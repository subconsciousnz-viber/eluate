"""Before/after evaluation, loudness-matched A/B and difference renders (spec 3.7).

Contract:
- `measure_track(x, fs, sections, crossovers_hz) -> dict` (JSON-safe): S95,
  loud-section integrated, integrated, LRA, the 1 s RMS of the loudest section
  and the loudest 1 s (meter.rms_1s, plain dBFS), true peak at 8x (meter, the
  constrained reading), 32x (FFT zero-padding, stages/limiter.true_peak_fft,
  the reading the ceiling is checked against, audit 05 D.23) and 4x (BS.1770
  Annex 2, compliance), sample peak, PLR, PSR p95, per-section integrated
  loudness, per-band crest p95 over M / S / stereo (multiband.band_crest, the
  solver's re-split measure), per-band L/R correlation and S/M ratio
  (features.stereo), fitted slope per view and sub shelf (features.tilt), and
  the LTAS per view. Non-finite numbers are written as null.
- `report(x, y, fs, out_dir, ...) -> dict`: measures both signals, writes
  out_dir/report.json, out_dir/report.md and, when matplotlib is installed, PNG plots (ltas.png: LTAS M / S /
  stereo before vs after; loudness.png: short-term loudness over time with the
  sections; gr.png: multiband compressor and ceiling GR per band and channel,
  glue GR, and the limiter's GR estimated as the 50 ms peak ratio of output
  to driven input, from `z` or precomputed by `limiter_gr_estimate` and passed
  as `limiter_gr` so the caller can free the pre-limiter audio first). Deltas are after - before, except crest where the solver's
  convention (loss = before - after) is reported too. Fitted tilt is shown
  against the target slope. The notch list (rendered or not, with time spans),
  per-stage GR summaries, the solver summary and the planner's record are
  copied in when given. `chain` holds each stage's decisions from the realised
  plan in plain units (the input normalisation gain and the output format
  included), and `flags` lists heuristic warnings for the listener
  (section balance, loudness range, sub shelf, L/R correlation, crest below the
  maintainer's masters, crest loss over the old limits under crest_mode "report", a noise
  floor the master exposed or that was already there; thresholds FLAG_* and
  noise.FLAG_DB). FLAG_SECTION_LU and FLAG_SECTION_IGNORE_LU are the one section-balance
  threshold and gate (S9, sprint 2026-10-21): the section flag here (a judged section
  rising more than 3 LU against the loudest), ui-data's, the process log's, and the
  solver's contrast term and keep test (R13) all read them. `flags_structured` holds every check, short or not, as {"kind":
  loudness | tilt | tp | section | lra | sub | stereo (with "band") | crest |
  crest_loss | noise, "value", "target", "tol", "short", "text"}: loudness, tilt and
  tp are the solver's constraints as measured (their text is the miss line), the
  rest the FLAG_* heuristics; `flags` is the text of the short heuristic ones. The
  crest reading is meter.crest_db (`crest_db` in before / after). `noise` holds
  noise.assess of the premaster's and the master's floors. `warnings` holds
  the solver's structured warnings (budget warnings and the loudness shortfall;
  the plan's budget warnings without a solve) and `loudness` the target and the
  RMS (plain and AES17) and S95 before and after. report.md (chunk 4) opens
  with Warnings (each warning in plain language, `warning_text`, naming the
  stage, band and M / S / both, needed against cap, and any loudness
  shortfall, "at least" when the limiter's needed GR comes from the measured shortfall,
  the shortfall marked within the tolerance when the RMS was met, then the
  other targets missed), then Gain reduction against budget
  (per stage, band and channel: the average peak GR, the outlier peak GR, the
  budget, the outlier cap and the status ok / over / under, maintainer 2026-10-19), Loudness (RMS with the
  AES17 value in brackets, S95), Flags and What was done (stage by stage, then
  every solver move kept or reverted, with the section contrast when a move was
  reverted for flattening the sections). Numbers carry 1 or 2 decimals and no
  section prints a raw Python list or dict.
- A/B (`write_ab`): ab/original_matched.wav and ab/master_matched.wav, matched
  on S95 by attenuating the louder one by the S95 difference (the quieter one
  is untouched), and ab/difference.wav = master_matched - original_matched
  (what the chain changed at matched loudness), all 32-bit float. The matched
  pair's S95 values are re-measured and reported. Gate-aware (audit 07 A18):
  when either S95 is not finite or below -60 LUFS (near the -70 LUFS gate) the
  pair is matched on gated integrated loudness instead, and not at all when
  that is not finite either; `matched_on` says which.
- Distance to the reference master (`distance_to_master`): when a master file of
  the same mix is supplied, each global feature of our master and of theirs is
  measured the same way and reported side by side with the difference (ours -
  theirs): S95, integrated, PLR, PSR, true peak, slope, sub shelf, LTAS RMS
  difference after removing the mean level (stereo, 30 Hz to 16 kHz), per-band
  crest p95 and correlation. No time alignment is needed (global features); a
  master at another sample rate is resampled (resample_poly) first.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from eluate import bands, io, meter, noise
from eluate.features import stereo as stereo_feature
from eluate.features import tilt as tilt_feature
from eluate.features._common import Context
from eluate.plan import MULTIBAND_MAX_GR_DB, TONE_MAX_DB
from eluate.stages import limiter as lim
from eluate.stages import multiband as mb_stage

BEFORE = "#2a78d6"   # categorical slot 1
AFTER = "#eb6834"    # categorical slot 2
GRID = "#e4e3df"
INK = "#52514e"

TONE_REPORT_HZ = (50.0, 100.0, 1000.0, 10000.0)
# the one section-balance threshold (S9): report flag, ui-data, process log, solver (R13)
FLAG_SECTION_LU = 3.0         # a section rising this much against the loudest one
FLAG_SECTION_IGNORE_LU = 20.0  # sections this far below the loudest are not judged
FLAG_LRA_LU = 5.0
FLAG_SUB_DB = 3.0
FLAG_CORR_DROP = 0.25
FLAG_CREST_DB = 7.1           # the maintainer's masters' lowest crest (meter.crest_db, work/corpus)


def _clean(v):
    """JSON-safe copy: numpy to Python, non-finite floats to None."""
    if isinstance(v, dict):
        return {str(k): _clean(w) for k, w in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(w) for w in v]
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def section_loudness(x, fs, sections) -> list[float]:
    from eluate.solver import section_loudness as sl
    return sl(x, fs, sections)


def measure_track(x: np.ndarray, fs: int, sections=(), crossovers_hz=bands.DEFAULT_CROSSOVERS,
                  short_term: bool = True) -> dict:
    x = np.asarray(x, dtype=np.float64)
    m = meter.measure(x, fs)
    m["true_peak_32x_dbtp"] = lim.true_peak_fft(x)
    m["section_lufs"] = section_loudness(x, fs, sections) if sections else []
    m["crest_p95_db"] = mb_stage.band_crest(x, fs, crossovers_hz)["p95_db"]
    ctx = Context(x, fs, crossovers_hz)
    st = stereo_feature.analyse(x, fs, ctx)
    m["band_corr"] = st["corr_summary"]
    m["band_sm_ratio_db"] = st["sm_ratio_summary_db"]
    tl = tilt_feature.analyse(x, fs, ctx)
    m["slope_db_oct"] = tl["slope_db_oct"]
    m["sub_shelf_db"] = tl["sub_shelf_db"]
    m["ltas_centres_hz"] = tl["centres"]
    m["ltas_db"] = tl["ltas_db"]
    m["ltas_fine_hz"] = tl["fine_centres"]
    m["ltas_fine_db"] = tl["fine_db"]
    if short_term:
        t_s, s_st = _short_term_series(x, fs)
        m["short_term_t"], m["short_term_lufs"] = t_s, s_st
    return m


def _short_term_series(x, fs):
    r = meter.short_term(x, fs)
    t, s = (r[0], r[1]) if isinstance(r, tuple) else (np.arange(len(r)) * meter.HOP_S, r)
    return np.asarray(t, dtype=np.float64), np.asarray(s, dtype=np.float64)


def _ltas_rms_diff(a_db, b_db, centres) -> float:
    c = np.asarray(centres)
    d = np.asarray(a_db) - np.asarray(b_db)
    sel = (c >= 30.0) & (c <= 16000.0) & np.isfinite(d) & (np.asarray(b_db) > -150)
    if not np.any(sel):
        return math.nan
    d = d[sel] - d[sel].mean()
    return float(np.sqrt(np.mean(d * d)))


# ------------------------------------------------------------------ A/B

AB_GATE_LUFS = -60.0     # below this S95 sits too close to the -70 LUFS gate to match on


def write_ab(x: np.ndarray, y: np.ndarray, fs: int, out_dir) -> dict:
    out = Path(out_dir) / "ab"
    sx = meter.loud_passage_lufs(x, fs)["s95"]
    sy = meter.loud_passage_lufs(y, fs)["s95"]
    matched_on = "s95"
    if not (math.isfinite(sx) and math.isfinite(sy)) or min(sx, sy) < AB_GATE_LUFS:
        # gate-aware (audit 07 A18): near or below the gate S95 is unstable; use the
        # gated integrated loudness, and match nothing if that is not finite either
        sx, sy = meter.integrated(x, fs), meter.integrated(y, fs)
        matched_on = "integrated"
    gx = gy = 0.0
    if math.isfinite(sx) and math.isfinite(sy):
        if sy > sx:
            gy = sx - sy
        else:
            gx = sy - sx
    xm = np.asarray(x, dtype=np.float64) * 10.0 ** (gx / 20.0)
    ym = np.asarray(y, dtype=np.float64) * 10.0 ** (gy / 20.0)
    io.write(out / "original_matched.wav", xm, fs)
    io.write(out / "master_matched.wav", ym, fs)
    io.write(out / "difference.wav", ym - xm, fs)
    return {"matched_on": matched_on if math.isfinite(sx) and math.isfinite(sy) else "none",
            "original_gain_db": gx, "master_gain_db": gy,
            "original_matched_s95": meter.loud_passage_lufs(xm, fs)["s95"],
            "master_matched_s95": meter.loud_passage_lufs(ym, fs)["s95"],
            "difference_s95": meter.loud_passage_lufs(ym - xm, fs)["s95"],
            "files": ["ab/original_matched.wav", "ab/master_matched.wav", "ab/difference.wav"]}


# ------------------------------------------------------------------ master distance

def distance_to_master(ours: np.ndarray, theirs: np.ndarray, fs: int, fs_theirs: int,
                       crossovers_hz=bands.DEFAULT_CROSSOVERS) -> dict:
    theirs = np.asarray(theirs, dtype=np.float64)
    if fs_theirs != fs:
        from scipy.signal import resample_poly
        fr = Fraction(int(fs), int(fs_theirs)).limit_denominator(1000)
        theirs = resample_poly(theirs, fr.numerator, fr.denominator, axis=0)
    a = measure_track(ours, fs, (), crossovers_hz, short_term=False)
    b = measure_track(theirs, fs, (), crossovers_hz, short_term=False)
    keys = ("s95_lufs", "integrated_lufs", "plr_db", "psr_p95_db", "true_peak_dbtp",
            "true_peak_32x_dbtp", "lra_lu")
    d = {k: {"ours": a[k], "theirs": b[k], "diff": a[k] - b[k]} for k in keys}
    d["slope_stereo"] = {"ours": a["slope_db_oct"]["stereo"], "theirs": b["slope_db_oct"]["stereo"],
                         "diff": a["slope_db_oct"]["stereo"] - b["slope_db_oct"]["stereo"]}
    d["sub_shelf_stereo"] = {"ours": a["sub_shelf_db"]["stereo"],
                             "theirs": b["sub_shelf_db"]["stereo"],
                             "diff": a["sub_shelf_db"]["stereo"] - b["sub_shelf_db"]["stereo"]}
    d["ltas_rms_diff_db"] = _ltas_rms_diff(a["ltas_db"]["stereo"], b["ltas_db"]["stereo"],
                                           a["ltas_centres_hz"])
    d["crest_p95_diff_db"] = (np.asarray(a["crest_p95_db"]) - np.asarray(b["crest_p95_db"]))
    d["band_corr_diff"] = np.asarray(a["band_corr"]) - np.asarray(b["band_corr"])
    return _clean(d)


# ------------------------------------------------------------------ plots

def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color="#0b0b0b")
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8)


def _plots(out: Path, before: dict, after: dict, sections, plan, y_gr, fs) -> list[str]:
    try:
        import matplotlib
    except ImportError:            # a dev dependency; the desktop engine ships without it
        return []
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    c = np.asarray(before["ltas_centres_hz"])
    for ax, v in zip(axes, ("M", "S", "stereo")):
        ax.semilogx(c, before["ltas_db"][v], color=BEFORE, lw=2, label="before")
        ax.semilogx(c, after["ltas_db"][v], color=AFTER, lw=2, label="after")
        _style(ax, f"LTAS {v} (dB, mean PSD per 1/3 oct)")
        ax.set_xlabel("Hz", fontsize=8, color=INK)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ltas.png", dpi=110)
    plt.close(fig)
    files.append("ltas.png")

    fig, ax = plt.subplots(figsize=(13, 3.6))
    for m, col, lab in ((before, BEFORE, "before"), (after, AFTER, "after")):
        s = np.asarray(m["short_term_lufs"], dtype=np.float64)
        ax.plot(m["short_term_t"], np.where(s > -70, s, np.nan), color=col, lw=1.5, label=lab)
    for s in sections or ():
        ax.axvline(s["start"], color=GRID, lw=1)
        ax.text(s["start"], 0.98, str(s.get("label") or ""), transform=ax.get_xaxis_transform(),
                fontsize=7, color=INK, va="top")
    _style(ax, "Short-term loudness (LUFS) with sections")
    ax.set_xlabel("s", fontsize=8, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "loudness.png", dpi=110)
    plt.close(fig)
    files.append("loudness.png")

    if plan is not None:
        mb = plan.multiband
        nb = len(mb.gr_db) if mb is not None else 0
        rows = nb + 1
        fig, axes = plt.subplots(rows, 1, figsize=(13, 1.4 * rows + 0.6), sharex=True)
        axes = np.atleast_1d(axes)
        dec = 50
        for b in range(nb):
            ax = axes[b]
            for ch, col, lab in ((0, BEFORE, "M"), (1, AFTER, "S")):
                v = np.maximum(mb.gr_db[b][ch].values, -3.0)
                if mb.ceiling_gr_db is not None:
                    cg = mb.ceiling_gr_db[b][ch].values
                    k = min(v.size, cg.size) if v.size > 1 and cg.size > 1 else max(v.size, cg.size)
                    v = (np.resize(v, k) + np.resize(cg, k))
                t = np.arange(v.size)[::dec] / 1000.0
                ax.plot(t, v[::dec], color=col, lw=1, label=lab)
            _style(ax, f"multiband band {b} GR, compressor + ceiling (dB)")
            if b == 0:
                ax.legend(frameon=False, fontsize=7, loc="lower right")
        ax = axes[-1]
        if plan.glue is not None and plan.glue.gain_db.values.size > 1:
            g = plan.glue.gain_db.values
            ax.plot(np.arange(g.size)[::dec] / 1000.0, g[::dec], color=BEFORE, lw=1, label="glue")
        if y_gr is not None:
            ax.plot(y_gr[0], y_gr[1], color=AFTER, lw=1, label="limiter (est.)")
        _style(ax, "glue and limiter GR (dB)")
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        ax.set_xlabel("s", fontsize=8, color=INK)
        fig.tight_layout()
        fig.savefig(out / "gr.png", dpi=100)
        plt.close(fig)
        files.append("gr.png")
    return files


def limiter_gr_estimate(z_driven: np.ndarray, y: np.ndarray, fs: int, win_s: float = 0.05):
    """(t, GR dB) per 50 ms window: peak(y) / peak(driven limiter input)."""
    w = max(1, int(round(win_s * fs)))
    k = y.shape[0] // w
    if k == 0:
        return np.zeros(0), np.zeros(0)
    a = np.max(np.abs(z_driven[:k * w]).reshape(k, w, -1), axis=(1, 2))
    b = np.max(np.abs(y[:k * w]).reshape(k, w, -1), axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        gr = np.where(a > 1e-9, 20 * np.log10(np.maximum(b, 1e-12) / a), 0.0)
    return (np.arange(k) + 0.5) * win_s, np.minimum(gr, 0.0)


# ------------------------------------------------------------------ report

def _gr_summary(plan, lim_info) -> dict:
    out = {}
    if plan is None:
        return out
    mb = plan.multiband
    if mb is not None:
        cap = MULTIBAND_MAX_GR_DB
        out["multiband_comp_gr_max_db"] = [[float(-max(np.min(e.values), -cap)) for e in p]
                                           for p in mb.gr_db]
        out["multiband_comp_gr_mean_db"] = [[float(-np.mean(np.maximum(e.values, -cap)))
                                             for e in p] for p in mb.gr_db]
        if mb.ceiling_gr_db is not None:
            out["multiband_ceiling_gr_max_db"] = [[float(-np.min(e.values)) for e in p]
                                                  for p in mb.ceiling_gr_db]
        out["multiband_side_makeup_db"] = mb.side_makeup()
    if plan.glue is not None:
        g = plan.glue.gain_db.values
        out["glue_gr_max_db"] = float(-np.min(g))
        out["glue_gr_mean_db"] = float(-np.mean(g))
        out["glue_mix"] = plan.glue.mix
    if lim_info:
        out["limiter"] = {k: v for k, v in lim_info.items() if k != "qp"}
    return out


def _chain_summary(plan) -> dict:
    """The decisions each stage of the realised plan carries, in plain units (static
    parameters, or time-averages of an envelope where the stage is dynamic)."""
    out = {}
    if plan is None:
        return out
    if plan.input is not None:
        out["input"] = {"gain_db": float(plan.input.gain_db), "r_ref_dbfs": plan.input.r_ref_dbfs,
                        "measured_dbfs": plan.input.measured_dbfs}
    eq = plan.inputeq
    if eq is not None and not plan.preserve_mix:
        from eluate.planner import inputeq_sections, inputeq_text
        out["inputeq"] = {"sections": inputeq_sections(eq), "text": inputeq_text(eq),
                          "phase": eq.phase,
                          "rotation_db_oct": float((eq.design or {}).get("rotation", 0.0))}
    if plan.notch is not None:
        out["notch"] = {"count": len(plan.notch.notches),
                        "f0_hz": [float(n.f0_hz) for n in plan.notch.notches]}
    if plan.sidestrip is not None:
        out["sidestrip"] = {"fc_hz": float(plan.sidestrip.fc_hz)}
    mb = plan.multiband
    if mb is not None:
        nb = len(mb.crossovers_hz) + 1
        out["multiband"] = {"crossovers_hz": list(mb.crossovers_hz),
                            "ratio": _band_table(mb.ratio, nb),
                            "threshold_db": _band_table(mb.threshold_db, nb),
                            "knee_db": _band_table(mb.knee_db, nb),
                            "attack_ms": mb.attack_ms, "release_ms": mb.release_ms,
                            "side_makeup_db": mb.side_makeup(),
                            "ceiling_lookahead_ms": mb.lookahead_ms(),
                            "makeup": (mb.budget or {}).get("makeup")}
    if plan.tone is not None:
        f = np.asarray(plan.tone.nodes_hz, dtype=float)
        g = np.clip([float(np.mean(e.values)) for e in plan.tone.gain_db], -TONE_MAX_DB,
                    TONE_MAX_DB)
        i, j = int(np.argmax(g)), int(np.argmin(g))
        out["tone"] = {"max_boost_db": float(g[i]), "max_boost_hz": float(f[i]),
                       "max_cut_db": float(g[j]), "max_cut_hz": float(f[j]),
                       "at_hz": {str(int(h)): float(np.interp(np.log2(h), np.log2(f), g))
                                 for h in TONE_REPORT_HZ if f[0] <= h <= f[-1]}}
    if plan.saturate is not None:
        out["saturate"] = {"drive_db": plan.saturate.drive_db,
                           "symmetry": float(plan.saturate.symmetry)}
    gl = plan.glue
    if gl is not None:
        out["glue"] = {"threshold_db": gl.threshold_db, "ratio": gl.ratio, "mix": gl.mix,
                       "attack_ms": gl.attack_ms, "release_ms": gl.release_ms,
                       "makeup_db": gl.makeup_db, "curve": gl.curve,
                       "mode": (gl.budget or {}).get("mode")}
    sl = getattr(plan, "speclim", None)
    if sl is not None:
        out["speclim"] = {"crossovers_hz": list(sl.crossovers_hz),
                          "allow_db": (sl.budget or {}).get("allow_db"),
                          "max_cut_db": (sl.budget or {}).get("max_cut_db"),
                          "lookahead_ms": sl.lookahead_ms(),
                          "bands": (sl.budget or {}).get("result")}
    if plan.keyeq is not None:
        out["keyeq"] = {"enabled": bool(plan.keyeq.enabled)}
    lp = plan.limiter
    if lp is not None:
        kick = lp.clip_threshold_db is None and lp.clip_depth_db > 0.0 and bool(lp.clip_onsets_s)
        out["limiter"] = {"drive_db": float(lp.input_gain_db), "ceiling_dbtp": float(lp.ceiling_dbtp),
                          "requested_gain_db": lp.requested_gain_db,
                          "target_rms_dbfs": lp.target_rms_dbfs,
                          "clip_depth_db": float(lp.clip_depth_db),
                          "clip_onsets": len(lp.clip_onsets_s or [])}
        out["clipper"] = {"mode": ("whole signal" if lp.clip_threshold_db is not None else
                                   "kick detective" if kick else "off"),
                          "threshold_db": lp.clip_threshold_db, "knee_db": float(lp.clip_knee_db),
                          "target_db": lp.clip_target_db, "split_hz": lp.clip_split_hz,
                          "depth_db": float(lp.clip_depth_db),
                          "onsets": len(lp.clip_onsets_s or [])}
    if plan.output is not None:
        out["output"] = {"bits": plan.output.bits, "dither": plan.output.dither}
    return out


def _band_table(v, nb: int) -> list[list[float]]:
    """A per-band [M, S] table from a scalar, a per-band list or a table."""
    if v is None:
        return [[None, None] for _ in range(nb)]
    if np.ndim(v) == 0:
        return [[float(v), float(v)] for _ in range(nb)]
    return [[None if c is None else float(c) for c in (r if np.ndim(r) else (r, r))] for r in v]


def _check(kind, value, target, tol, short, text=None, **kw) -> dict:
    return {"kind": kind, "value": value, "target": target, "tol": tol, "short": bool(short),
            "text": text if short else None, **kw}


def _checks(rep: dict) -> list[dict]:
    """Every check the listener is pointed at, as {"kind", "value", "target", "tol",
    "short", "text"} (module contract): the solver's loudness, tilt and true-peak
    constraints as measured on the master, then the FLAG_* heuristics. `short` marks
    the ones to show in amber; `text` is their plain-language line."""
    out = []
    names = {"rms": "loudness", "slope": "tilt", "true_peak": "tp"}
    for c in (rep.get("solver") or {}).get("constraints") or []:
        if c.get("name") in names and c.get("value") is not None:
            # a reported-only slope (the live profile) is flagged on its own tolerance
            short = (abs(c["value"] - c["target"]) > c["tol"] if c.get("mode") == "report"
                     else not c.get("ok", True))
            out.append(_check(names[c["name"]], c["value"], c.get("target"), c.get("tol"),
                              short, miss_text({**c, "excess": abs(c["value"] - c["target"])
                                                - c["tol"]} if c.get("mode") == "report" else c)))
    secs = [s for s in rep.get("sections") or []
            if s.get("before_lufs") is not None and s.get("after_lufs") is not None]
    if secs:
        ref = max(secs, key=lambda s: s["before_lufs"])
        live = [s for s in secs if s["before_lufs"] > ref["before_lufs"] - FLAG_SECTION_IGNORE_LU]
        rises = [((s["after_lufs"] - ref["after_lufs"]) - (s["before_lufs"] - ref["before_lufs"]), s)
                 for s in live]
        moved = [(d, s) for d, s in rises if d > FLAG_SECTION_LU]
        text = None
        if moved:
            d, w = max(moved, key=lambda t: t[0])
            louder = [s for _, s in moved if s["after_lufs"] > ref["after_lufs"]]
            text = (f"Section balance: {len(moved)} section(s) rose more than {FLAG_SECTION_LU:g} LU "
                    f"against the loudest section ({ref['label']} at {ref['start']:.0f} s); worst: "
                    f"{w['label']} at {w['start']:.0f} s, +{d:.1f} LU"
                    + (f"; {len(louder)} now louder than it" if louder else "") + ".")
        out.append(_check("section", max(d for d, _ in rises), 0.0, FLAG_SECTION_LU, moved, text))
    b, a = rep.get("before") or {}, rep.get("after") or {}
    if b.get("lra_lu") is not None and a.get("lra_lu") is not None:
        out.append(_check("lra", a["lra_lu"], b["lra_lu"], FLAG_LRA_LU,
                          b["lra_lu"] - a["lra_lu"] > FLAG_LRA_LU,
                          f"Loudness range: {b['lra_lu']:.1f} -> {a['lra_lu']:.1f} LU."))
    t = rep.get("tilt") or {}
    sb, sa = ((t.get("sub_shelf_before") or {}).get("stereo"),
              (t.get("sub_shelf_after") or {}).get("stereo"))
    if sb is not None and sa is not None:
        out.append(_check("sub", sa, sb, FLAG_SUB_DB, abs(sa - sb) > FLAG_SUB_DB,
                          f"Sub: the shelf below 100 Hz moved {sa - sb:+.1f} dB ({sb:.1f} -> {sa:.1f})."))
    co = rep.get("correlation") or {}
    for i, (p, q) in enumerate(zip(co.get("before") or [], co.get("after") or [])):
        if p is None or q is None:
            continue
        out.append(_check("stereo", q, p, FLAG_CORR_DROP, (q < 0.0 <= p) or p - q > FLAG_CORR_DROP,
                          f"Stereo: band {i} L/R correlation {p:.2f} -> {q:.2f}"
                          + (" (now negative: check mono)" if q < 0.0 else "") + ".", band=i))
    if a.get("crest_db") is not None:
        out.append(_check("crest", a["crest_db"], FLAG_CREST_DB, 0.0, a["crest_db"] < FLAG_CREST_DB,
                          f"Crest: {a['crest_db']:.1f} dB, below the reference masters "
                          f"({FLAG_CREST_DB:g} dB and up)."))
    cr = rep.get("crest") or {}
    lims = cr.get("limits_db")
    if lims and (rep.get("targets") or {}).get("crest_mode") == "report":
        rows = [(i, max(v for v in row if v is not None), lims[i])
                for i, row in enumerate(cr.get("loss_db") or []) if any(v is not None for v in row)]
        over = [r for r in rows if r[1] > r[2]]
        if rows:
            out.append(_check("crest_loss", max(r[1] - r[2] for r in rows), 0.0, 0.0, over,
                              "Crest (reported only): loss above the old limit in "
                              + ", ".join(f"band {i} {v:.1f} dB > {lm:g}" for i, v, lm in over) + "."))
    nz = rep.get("noise") or {}
    if nz.get("after_db") is not None:
        short = nz.get("status") in ("exposed", "present")
        text = None
        if nz.get("status") == "exposed":
            text = (f"Noise: the hiss floor (BS.468 weighted) rose from {nz['before_db']:.1f} to "
                    f"{nz['after_db']:.1f} dB, past the {nz['threshold_db']:g} dB threshold; the "
                    "master made it audible.")
        elif short:
            text = (f"Noise: the hiss floor (BS.468 weighted) is {nz['before_db']:.1f} dB in the "
                    f"premaster and {nz['after_db']:.1f} dB after the master, above the "
                    f"{nz['threshold_db']:g} dB threshold in both.")
        out.append(_check("noise", nz["after_db"], nz.get("threshold_db"), 0.0, short, text))
    return out


_SOLVER_KINDS = ("loudness", "tilt", "tp")   # report.md lists these under Warnings


def _flags(rep: dict, checks: list[dict] | None = None) -> list[str]:
    """Plain-language warnings about what the master did to the mix: the short
    heuristic checks (FLAG_*), meant to point the listener at something, not to fail."""
    return [c["text"] for c in (checks if checks is not None else _checks(rep))
            if c["short"] and c["kind"] not in _SOLVER_KINDS]


def report(x: np.ndarray, y: np.ndarray, fs: int, out_dir, plan=None, targets=None,
           plan_rec=None, solve_summary=None, lim_info=None, sections=None, z=None,
           master_ref=None, extra=None, limiter_gr=None) -> dict:
    """Write report.json, report.md and plots to out_dir (module contract)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    xo = list(targets.crossovers_hz) if targets is not None else list(bands.DEFAULT_CROSSOVERS)
    secs = sections if sections is not None else (targets.sections if targets else [])
    before = measure_track(x, fs, secs, xo)
    after = measure_track(y, fs, secs, xo)
    gr_est = limiter_gr
    if gr_est is None and z is not None and plan is not None and plan.limiter is not None:
        gr_est = limiter_gr_estimate(z * 10.0 ** (plan.limiter.input_gain_db / 20.0), y, fs)
    plots = _plots(out, before, after, secs, plan, gr_est, fs)
    scal = ("s95_lufs", "loud_section_lufs", "integrated_lufs", "lra_lu", "true_peak_dbtp",
            "true_peak_32x_dbtp", "true_peak_4x_dbtp", "sample_peak_dbfs", "plr_db",
            "psr_p95_db", "momentary_max_lufs", "short_term_max_lufs", "rms_meter_dbfs",
            "rms_meter_max_dbfs", "rms_1s_loud_dbfs", "rms_1s_max_dbfs", "crest_db")
    rep = {"before": {k: before[k] for k in scal}, "after": {k: after[k] for k in scal}}
    rep["delta"] = {k: after[k] - before[k] for k in scal}
    rep["sections"] = [{"start": s["start"], "end": s["end"], "label": s.get("label"),
                        "before_lufs": a, "after_lufs": b, "delta_lu": b - a}
                       for s, a, b in zip(secs, before["section_lufs"], after["section_lufs"])]
    cb, ca = np.asarray(before["crest_p95_db"]), np.asarray(after["crest_p95_db"])
    rep["crest"] = {"views": list(mb_stage.CREST_VIEWS), "before_p95_db": cb, "after_p95_db": ca,
                    "delta_db": ca - cb, "loss_db": cb - ca,
                    "limits_db": list(targets.crest_limits_db) if targets else None,
                    "exempt": list(targets.crest_exempt) if targets else []}
    rep["correlation"] = {"before": before["band_corr"], "after": after["band_corr"],
                          "delta": np.asarray(after["band_corr"]) - np.asarray(before["band_corr"]),
                          "sm_ratio_before_db": before["band_sm_ratio_db"],
                          "sm_ratio_after_db": after["band_sm_ratio_db"]}
    rep["tilt"] = {"before": before["slope_db_oct"], "after": after["slope_db_oct"],
                   "target_stereo": targets.slope_db_oct if targets else None,
                   "error_stereo": (after["slope_db_oct"]["stereo"] - targets.slope_db_oct)
                   if targets else None,
                   "sub_shelf_before": before["sub_shelf_db"], "sub_shelf_after": after["sub_shelf_db"]}
    rep["ltas"] = {"centres_hz": before["ltas_centres_hz"], "before": before["ltas_db"],
                   "after": after["ltas_db"],
                   "fine": {"centres_hz": before["ltas_fine_hz"], "before": before["ltas_fine_db"],
                            "after": after["ltas_fine_db"]}}
    rep["notches"] = (plan_rec or {}).get("notches", [])
    rep["gain_reduction"] = _gr_summary(plan, lim_info)
    rep["chain"] = _chain_summary(plan)
    # per-stage peak GR against its budget, with the warnings
    from eluate import planner
    rep["budget"] = ((solve_summary or {}).get("reported") or {}).get("budget") \
        or (planner.budget_report(plan, lim_info) if plan is not None else None)
    rep["solver"] = solve_summary
    rep["warnings"] = ((solve_summary or {}).get("warnings")
                       if (solve_summary or {}).get("warnings") is not None
                       else (rep["budget"] or {}).get("warnings") or [])
    # the target's reading is the maintainer's meter (meter.rms_meter: 300 ms, AES17, loudest
    # passage); the 1 s plain readings stay for reference
    rep["loudness"] = {"target_rms_dbfs": targets.rms_dbfs if targets is not None else None,
                       "reading": "300 ms RMS, AES17, loudest passage (p95)",
                       "aes17_offset_db": meter.AES17_OFFSET_DB,
                       "before_rms_dbfs": before["rms_meter_dbfs"],
                       "after_rms_dbfs": after["rms_meter_dbfs"],
                       "before_rms_aes17_dbfs": before["rms_meter_dbfs"],
                       "after_rms_aes17_dbfs": after["rms_meter_dbfs"],
                       "before_rms_1s_dbfs": before["rms_1s_loud_dbfs"],
                       "after_rms_1s_dbfs": after["rms_1s_loud_dbfs"],
                       "before_s95_lufs": before["s95_lufs"], "after_s95_lufs": after["s95_lufs"]}
    nb = noise.floor(x, fs)
    rep["noise"] = noise.assess(nb, noise.floor(y, fs, quiet=nb["quiet"]))
    rep["planner"] = {k: v for k, v in (plan_rec or {}).items() if k != "notches"}
    rep["targets"] = targets.to_dict() if targets is not None else None
    rep["ab"] = write_ab(x, y, fs, out)
    rep["plots"] = plots
    if master_ref is not None:
        rep["distance_to_master"] = distance_to_master(y, master_ref[0], fs, master_ref[1], xo)
    if extra:
        rep.update(extra)
    rep = _clean(rep)
    rep["flags_structured"] = _checks(rep)
    rep["flags"] = _flags(rep, rep["flags_structured"])
    io.write_json(out / "report.json", rep)
    io.write_text(out / "report.md", report_md(rep))
    return rep


def _f(v, nd=2):
    return "n/a" if v is None else f"{v:.{nd}f}"


def _pairs(rows, nd=1):
    """Per-band [M, S] pairs as 'M/S' cells joined by spaces."""
    return " ".join(f"{_f(m, nd)}/{_f(s, nd)}" for m, s in rows)


def _band_name(rep: dict, b) -> str:
    """'band 2 (400-1600 Hz)' from the chain's crossovers."""
    xo = ((rep.get("chain") or {}).get("multiband") or {}).get("crossovers_hz")
    if b is None:
        return "band ?"
    if not xo:
        return f"band {b}"
    edges = [None] + list(xo) + [None]
    lo, hi = edges[b], edges[b + 1]
    rng = (f"below {hi:.0f} Hz" if lo is None else f"above {lo:.0f} Hz" if hi is None
           else f"{lo:.0f}-{hi:.0f} Hz")
    return f"band {b} ({rng})"


_CHANNEL = {"M": "M", "S": "S", "MS": "M and S", "both": "M and S"}


def allow_text(allow) -> str:
    """speclim's allowance: one value, or per band ("2.0 dB, 3.5 dB in the upper bands")."""
    if allow is None:
        return "n/a"
    if np.ndim(allow) == 0:
        return f"{_f(allow, 1)} dB"
    lo, hi = min(allow), max(allow)
    return f"{_f(lo, 1)} dB" + (f", {_f(hi, 1)} dB in the upper bands" if hi > lo else "")


def warning_text(w: dict, rep: dict) -> str:
    """One plain-language line for a structured warning (planner.budget_report, the
    solver's loudness warning)."""
    st = w.get("stage")
    if st == "multiband" and w.get("kind") == "average":
        return (f"Multiband {_band_name(rep, w.get('band'))}, "
                f"{_CHANNEL.get(w.get('channel'), w.get('channel'))}: average peak gain "
                f"reduction {_f(w.get('achieved_db'), 2)} dB against the "
                f"{_f(w.get('target_db'), 1)} dB target (held at the outlier cap).")
    if st == "multiband":
        return (f"Multiband {_band_name(rep, w.get('band'))}, "
                f"{_CHANNEL.get(w.get('channel'), w.get('channel'))}: needed "
                f"{_f(w.get('needed_db'), 1)} dB of peak gain reduction, cap "
                f"{_f(w.get('cap_db'), 1)} dB; rendered at the cap.")
    if st == "inputeq" and w.get("kind") == "slope":
        at = w.get("at_bound") or []
        return (f"Slope: reached {_f(w.get('achieved_db_oct'))} dB/oct against "
                f"{_f(w.get('target_db_oct'))}"
                + (f"; at its limit: {', '.join(at)}" if at else "")
                + f". A Tilt of {_f(w.get('tilt_db'), 1)} dB/oct would make this slope the target.")
    if st == "glue":
        return (f"Glue ({w.get('mode')}): reached {_f(w.get('achieved_db'), 2)} dB of net average peak "
                f"gain reduction against the {_f(w.get('target_db'), 1)} dB target.")
    if st == "clipper":
        return (f"Clipper: the average output peak fell {_f(w.get('achieved_db'), 2)} dB against the "
                f"{_f(w.get('needed_db'), 1)} dB target (cap {_f(w.get('cap_db'), 1)} dB).")
    if st == "limiter":
        least = "at least " if w.get("needed_source") == "render" else ""
        s = (f"Limiter: the loudness target needed {least}{_f(w.get('needed_db'), 1)} dB of average "
             f"peak gain reduction, cap {_f(w.get('cap_db'), 1)} dB")
        if w.get("outlier_needed_db") is not None:
            s += (f" (outlier {least}{_f(w.get('outlier_needed_db'), 1)} dB, cap "
                  f"{_f(w.get('outlier_cap_db'), 1)} dB)")
        b = w.get("bound")
        s += ("; clamped at the outlier cap (the largest peaks set the gain)" if b == "outlier"
              else "; clamped at the average budget (the typical hits set the gain)"
              if b == "average" else "; clamped at the cap")
        if w.get("shortfall_db") is not None:
            est = " (estimated)" if w.get("shortfall_source") == "proxy" else ""
            s += f", so the master is {_f(w['shortfall_db'], 1)} dB short of the RMS target{est}"
            if w.get("accepted"):
                s += ", within the tolerance (not a miss)"
        if w.get("request_at_drive_limit"):
            s += "; the loudness solve found no gain inside its range"
        return s + "."
    if st == "loudness":
        return (f"Loudness: {_f(w.get('achieved_dbfs'), 1)} dB RMS (Ozone meter) against the "
                f"{_f(w.get('target_dbfs'), 1)} dB target, {_f(w.get('shortfall_db'), 1)} dB "
                f"{'short' if (w.get('shortfall_db') or 0) >= 0 else 'over'}"
                + (f" ({w['cause']})" if w.get("cause") else "") + ".")
    return f"{st}: " + ", ".join(f"{k} {_f(v) if isinstance(v, float) else v}"
                                 for k, v in w.items() if k != "stage") + "."


_MISS_NAME = {"rms": "Loudness (RMS)", "slope": "Slope", "true_peak": "True peak",
              "ltas_shape": "Spectrum shape", "section_deltas": "Section balance"}


def miss_text(c: dict) -> str:
    """One line for a target the solver missed."""
    n = c.get("name", "")
    name = _MISS_NAME.get(n, n.replace("crest_band", "Crest band "))
    return (f"{name}: {_f(c.get('value'))} against {_f(c.get('target'))} "
            f"+- {_f(c.get('tol'))}, missed by {_f(c.get('excess'))}.")


def _st(r: dict) -> str:
    """Status cell: the status, plus what held the stage back (a cap that bound, or a
    gated band)."""
    s = r.get("status", "n/a")
    if r.get("gated"):
        return "gated (near silent)"
    b = r.get("bound")
    if b == "outlier":
        return f"{s}, held by outlier cap"
    if b == "average":
        return f"{s}, held by average budget"
    if b == "sample cap":
        return f"{s}, sample cap reached"
    return s


def _budget_md(rep: dict) -> list[str]:
    """Per-stage average and outlier peak gain reduction against the budget and the
    outlier cap (planner.budget_report; rendered where known, else planned)."""
    br = rep.get("budget") or {}
    bud = br.get("stages") or {}
    if not bud:
        return []
    hits = br.get("hits")
    L = ["## Gain reduction against budget", "",
         "Gain reduction, dB. Average peak = the typical loud hit; outlier = the single "
         "largest moment." + (f" Hits: {hits}." if hits else ""), "",
         "| stage | band | channel | average peak | outlier peak | budget | outlier cap | status |",
         "|---|---|---|---|---|---|---|---|"]
    mb = bud.get("multiband")
    if mb:
        for b, row in enumerate(mb.get("bands") or []):
            for c, r in zip("MS", row):
                L.append(f"| multiband compressor | {_band_name(rep, b)} | {c} | "
                         f"{_f(r.get('comp_db'))} | {_f(r.get('comp_outlier_db'))} | "
                         f"{_f(r.get('comp_target_db'), 1)} | {_f(r.get('comp_cap_db'), 1)} | "
                         f"{_st(r)} |")
                if (r.get("limit_outlier_db") or 0.0) > 0.01:
                    L.append(f"| band limiter | {_band_name(rep, b)} | {c} | "
                             f"{_f(r.get('limit_db'))} | {_f(r.get('limit_outlier_db'))} | - | "
                             f"{_f(r.get('limit_cap_db'), 1)} | {r.get('status', 'n/a')} |")
    g = bud.get("glue")
    if g:
        L.append(f"| glue ({g.get('mode')}, net) | - | both | {_f(g.get('net_db'))} | "
                 f"{_f(g.get('net_outlier_db'))} | {_f(g.get('net_target_db'), 1)} | "
                 f"{_f(g.get('outlier_cap_db'), 1)} | {_st(g)} |")
    c = bud.get("clipper")
    if c:
        L.append(f"| clipper (output peak) | - | both | {_f(c.get('planned_db'))} | "
                 f"{_f(c.get('outlier_db'))} | {_f(c.get('target_db'), 1)} | "
                 f"{_f(c.get('outlier_cap_db'), 1)} | {_st(c)} |")
    lm = bud.get("limiter")
    if lm:
        L.append(f"| limiter | - | both | {_f(lm.get('planned_db'))} | "
                 f"{_f(lm.get('outlier_db'))} | {_f(lm.get('budget_db'), 1)} | "
                 f"{_f(lm.get('outlier_cap_db'), 1)} | {_st(lm)} |")
    return L + [""]


def _what_was_done_md(rep: dict) -> list[str]:
    c = rep.get("chain") or {}
    s = rep.get("solver") or {}
    pr = rep.get("planner") or {}
    if not c and not s:
        return []
    L = ["## What was done", ""]
    if "input" in c:
        i = c["input"]
        L.append(f"0. Input: gain {_f(i['gain_db'])} dB, premaster loudest-section 1 s RMS "
                 f"{_f(i['measured_dbfs'])} dBFS to the reference {_f(i['r_ref_dbfs'], 1)} dBFS.")
    items = []
    if "inputeq" in c:
        q = c["inputeq"]
        rot = q.get("rotation_db_oct") or 0.0
        er = pr.get("inputeq") or {}
        shape = er.get("low_mid_shape")
        why = ""
        if shape:
            why = (f" Low-mid excess {_f(er.get('low_mid_excess_db'), 1)} dB over "
                   f"{_f(er.get('low_mid_run_oct'), 2)} octaves above the line: {shape}.")
        items.append("Input EQ (static, " + ("zero" if q["phase"] == "zero" else "minimum")
                     + " phase): " + ("; ".join(q["text"]) if q["text"] else "every section off")
                     + (f" (fitted with a {_f(rot)} dB/oct solver rotation)" if rot else "") + "."
                     + why)
    if "notch" in c:
        n = c["notch"]
        items.append(f"Notch: {n['count']} notch(es)"
                     + (" at " + ", ".join(f"{f:.0f} Hz" for f in n["f0_hz"]) if n["f0_hz"] else "")
                     + " (details under Notches).")
    if "sidestrip" in c:
        items.append(f"Side strip: side channel high-passed below "
                     f"{c['sidestrip']['fc_hz']:.0f} Hz.")
    ch = pr.get("character") or {}
    if "multiband" in c:
        m = c["multiband"]
        xr = pr.get("crossovers") or {}
        txt = (f"Multiband (crossovers {', '.join(f'{x:.0f}' for x in m['crossovers_hz'])} Hz)"
               + (f", character {ch['preset']} ({ch.get('reason')})" if ch.get("preset") else "")
               + (f", makeup {m['makeup']}" if m.get("makeup") else "") + ".")
        if xr.get("reason"):
            txt += f"\n   Crossovers: {xr['reason']}."
        txt += "\n\n   | band | ratio M | ratio S | threshold M dBFS | threshold S dBFS | " \
               "attack ms M/S | release ms M/S | knee dB |\n   |---|---|---|---|---|---|---|---|"
        for b, (ra, th) in enumerate(zip(m["ratio"], m["threshold_db"])):
            att, rel = m["attack_ms"][b], m["release_ms"][b]
            kn = m["knee_db"][b][0] if m.get("knee_db") else None
            txt += (f"\n   | {_band_name(rep, b)} | {_f(ra[0])} | {_f(ra[1])} | {_f(th[0], 1)} | "
                    f"{_f(th[1], 1)} | {_f(att[0], 1)}/{_f(att[1], 1)} | "
                    f"{_f(rel[0], 0)}/{_f(rel[1], 0)} | {_f(kn, 1)} |")
        txt += ("\n\n   Side makeup dB per band: "
                + ", ".join(_f(v, 1) for v in m["side_makeup_db"]) + ".")
        if m.get("ceiling_lookahead_ms"):
            txt += ("\n   Band limiter lookahead ms per band: "
                    + ", ".join(_f(v, 0) for v in m["ceiling_lookahead_ms"]) + ".")
        items.append(txt)
    if "tone" in c:
        t = c["tone"]
        items.append(f"Tone (FIR): largest boost {_f(t['max_boost_db'])} dB at "
                     f"{t['max_boost_hz']:.0f} Hz, largest cut {_f(t['max_cut_db'])} dB at "
                     f"{t['max_cut_hz']:.0f} Hz; at "
                     + ", ".join(f"{k} Hz {_f(v)} dB" for k, v in t["at_hz"].items()) + ".")
    if "saturate" in c:
        rows = {r["band"]: r for r in (pr.get("saturation") or {}).get("bands") or []}
        parts = []
        for b, (dm, ds) in enumerate(c["saturate"]["drive_db"]):
            d = (rows.get(b) or {}).get("deficit_db")
            parts.append(f"{_band_name(rep, b)} M {_f(dm, 1)} / S {_f(ds, 1)} dB"
                         + (f" (deficit {_f(d, 1)} dB)" if d is not None and b > 0 else ""))
        items.append("Saturation (drive = deficit below the target line, 0.5 dB steps, at most "
                     "2 dB; high-mid S: width): " + "; ".join(parts) + ".")
    if "glue" in c:
        g = c["glue"]
        items.append(f"Glue ({g.get('mode') or 'threshold rule'}): threshold "
                     f"{_f(g['threshold_db'], 1)} dBFS, ratio {_f(g['ratio'], 1)}, "
                     f"mix {_f(g['mix'])}, attack {_f(g['attack_ms'], 1)} ms, release "
                     f"{_f(g['release_ms'], 0)} ms.")
    if "speclim" in c:
        s = c["speclim"]
        parts = []
        for b, row in enumerate(s.get("bands") or []):
            for ch, r in zip("MS", row):
                if r.get("gated") and r.get("peak_db") is not None:
                    parts.append(f"{_band_name(rep, b)} {ch}: gated (near silent)")
                elif r.get("ceiling_db") is not None:
                    parts.append(f"{_band_name(rep, b)} {ch}: up to {_f(r['cut_db'], 1)} dB "
                                 f"(typical hit {_f(r['typical_db'], 1)}, peak {_f(r['peak_db'], 1)} dBFS)")
        look = (f", lookahead {' / '.join(_f(v, 0) for v in s['lookahead_ms'])} ms"
                if s.get("lookahead_ms") else "")
        items.append(f"Frequency-dependent limiter (peaks more than {allow_text(s.get('allow_db'))} "
                     f"above the band's typical loud hit, at most {_f(s.get('max_cut_db'), 1)} dB"
                     f"{look}): "
                     + ("; ".join(parts) if parts else "no band needed it") + ".")
    if "keyeq" in c:
        items.append(f"Key EQ: {'on' if c['keyeq']['enabled'] else 'off'}.")
    if "clipper" in c:
        k = c["clipper"]
        why = ((pr.get("clipper") or {}).get("reason") or ch.get("kick_detective_reason"))
        if k["mode"] == "whole signal":
            t = (f"Clipper: whole signal, threshold {_f(k['threshold_db'])} dBFS (8x, before the "
                 f"limiter gain), knee {_f(k['knee_db'], 1)} dB, output-peak target "
                 f"{_f(k['target_db'], 1)} dB")
        elif k["mode"] == "kick detective":
            t = f"Clipper: kick detective, up to {_f(k['depth_db'], 1)} dB on {k['onsets']} onsets"
        else:
            t = "Clipper: off (kick detective with no usable kick onsets)"
        kd = "on" if k["mode"] != "whole signal" else "off"
        items.append(t + (f" (kick detective {kd}: {why})" if why else "") + ".")
    if "limiter" in c:
        lm = c["limiter"]
        req = lm.get("requested_gain_db")
        items.append(f"Limiter: gain {_f(lm['drive_db'])} dB into a {_f(lm['ceiling_dbtp'])} dBTP "
                     "ceiling"
                     + (f" (asked {_f(req)} dB for the {_f(lm.get('target_rms_dbfs'), 1)} dB RMS "
                        "target, Ozone meter)" if req is not None and lm.get("target_rms_dbfs") is not None
                        else "") + ".")
    if "output" in c:
        o = c["output"]
        items.append(f"Output: {'32-bit float' if o['bits'] == 32 else str(o['bits']) + '-bit PCM'}"
                     f"{', TPDF dither' if o['dither'] == 'tpdf' else ''}.")
    L += [f"{i}. {t}" for i, t in enumerate(items, 1)]
    hist = s.get("history") or []
    moves = [h for h in hist if "kept" in h or "reverted" in h]
    if moves:
        L += ["", "Solver moves (each judged on the summed constraint excess, lower is better):", ""]
        for h in moves:
            what = h.get("kept") or h.get("reverted")
            verdict = "kept" if "kept" in h else "reverted"
            sc = (f", score {_f(h['score_before'])} -> {_f(h['score_after'])}"
                  if h.get("score_before") is not None and h.get("score_after") is not None else "")
            if h.get("reason") == "section contrast":
                sc += (f", flattened the sections (contrast loss {_f(h.get('contrast_loss_before_lu'), 1)}"
                       f" -> {_f(h.get('contrast_loss_after_lu'), 1)} LU, flag {FLAG_SECTION_LU:g} LU)")
            L.append(f"- render {h.get('render', '-')}: {what}, {verdict}{sc}")
    L.append("")
    return L


_METRIC = {"rms_meter_dbfs": ("RMS, Ozone meter (300 ms, AES17, loudest passage)", "dB"),
           "rms_meter_max_dbfs": ("RMS, Ozone meter, loudest 300 ms", "dB"),
           "rms_1s_loud_dbfs": ("RMS, loudest section (1 s)", "dBFS"),
           "rms_1s_max_dbfs": ("RMS, loudest 1 s", "dBFS"), "s95_lufs": ("S95", "LUFS"),
           "loud_section_lufs": ("Loud-section integrated", "LUFS"),
           "integrated_lufs": ("Integrated", "LUFS"), "lra_lu": ("Loudness range", "LU"),
           "true_peak_dbtp": ("True peak 8x", "dBTP"),
           "true_peak_32x_dbtp": ("True peak 32x", "dBTP"),
           "true_peak_4x_dbtp": ("True peak 4x (BS.1770)", "dBTP"),
           "sample_peak_dbfs": ("Sample peak", "dBFS"), "plr_db": ("PLR", "dB"),
           "psr_p95_db": ("PSR p95", "dB"), "momentary_max_lufs": ("Momentary max", "LUFS"),
           "short_term_max_lufs": ("Short-term max", "LUFS"),
           "crest_db": ("Crest (sample peak - loudest-section 1 s RMS)", "dB")}


def report_md(rep: dict) -> str:
    b, a, d = rep["before"], rep["after"], rep["delta"]
    L = ["# Master report", ""]
    pr = (rep.get("planner") or {}).get("profile") or {}
    if pr.get("name") and pr["name"] != "electronic":
        L += [f"Profile: {pr['name']}" + ("" if pr.get("calibrated") else
              " (phase 1: the live profile defaults are uncalibrated)") + ".", ""]
    s = rep.get("solver") or {}
    warns = rep.get("warnings") or []
    miss = [m for m in (s.get("misses") or []) if m.get("name") != "rms"
            or not any(w.get("stage") == "loudness" for w in warns)]
    L += ["## Warnings", ""]
    L += [f"- {warning_text(w, rep)}" for w in warns] + [f"- {miss_text(m)}" for m in miss]
    if not warns and not miss:
        L.append("- none")
    L.append("")
    L += _budget_md(rep)
    ld = rep.get("loudness") or {}
    off = ld.get("aes17_offset_db", 3.01)
    L += ["## Loudness", ""]
    if ld.get("target_rms_dbfs") is not None:
        L.append(f"- Target: RMS {_f(ld['target_rms_dbfs'], 1)} dB on the Ozone meter "
                 "(300 ms, AES17, loudest passage).")
    for k, lab in (("before", "Before"), ("after", "After")):
        v, v1, sv = (rep[k].get("rms_meter_dbfs"), rep[k].get("rms_1s_loud_dbfs"),
                     rep[k].get("s95_lufs"))
        L.append(f"- {lab}: RMS {_f(v)} dB (Ozone meter); 1 s loudest section {_f(v1)} dBFS "
                 f"plain ({_f(None if v1 is None else v1 + off)} AES17); S95 {_f(sv)} LUFS.")
    if s:
        t_ = rep.get("targets") or {}
        L.append(f"- Solver: {s.get('renders')} renders, runtime {_f(s.get('runtime_s'), 0)} s; "
                 f"crest mode {t_.get('crest_mode')}, true-peak meter {t_.get('tp_meter')}.")
    L.append("")
    fl = rep.get("flags")
    if fl is not None:
        L += ["## Flags", ""] + ([f"- {x}" for x in fl] or ["- none"]) + [""]
    L += _what_was_done_md(rep)
    if s.get("constraints"):
        L += ["## Targets", "", "| target | value | aim | tolerance | excess | met |",
              "|---|---|---|---|---|---|"]
        for c in s["constraints"]:
            L.append(f"| {c['name']} | {_f(c['value'])} | {_f(c['target'])} | {_f(c['tol'])} "
                     f"| {_f(c['excess'])} | {'yes' if c['ok'] else 'no'} |")
        L.append("")
    L += ["## Loudness and peaks", "", "| metric | before | after | delta |", "|---|---|---|---|"]
    for k in b:
        name, unit = _METRIC.get(k, (k, ""))
        L.append(f"| {name}{' (' + unit + ')' if unit else ''} | {_f(b[k])} | {_f(a[k])} | "
                 f"{_f(d[k])} |")
    L += ["", "## Sections", "", "| start | end | label | before | after | delta |",
          "|---|---|---|---|---|---|"]
    for r in rep["sections"]:
        L.append(f"| {_f(r['start'], 1)} | {_f(r['end'], 1)} | {r['label']} | {_f(r['before_lufs'])} "
                 f"| {_f(r['after_lufs'])} | {_f(r['delta_lu'])} |")
    cr = rep["crest"]
    L += ["", "## Per-band crest p95 (dB), loss = before - after", "",
          "| band | " + " | ".join(f"{v} before | {v} after | {v} loss" for v in cr["views"])
          + " | limit |", "|---" * (3 * len(cr["views"]) + 2) + "|"]
    for i, (rb, ra, rl) in enumerate(zip(cr["before_p95_db"], cr["after_p95_db"], cr["loss_db"])):
        cells = " | ".join(f"{_f(p)} | {_f(q)} | {_f(r)}" for p, q, r in zip(rb, ra, rl))
        lim_v = cr["limits_db"][i] if cr.get("limits_db") else None
        L.append(f"| {i} | {cells} | {_f(lim_v, 1)} |")
    if cr.get("exempt"):
        L += ["", "Exempt from the crest constraint (removed by the sidestrip on purpose): "
              + ", ".join(f"band {b} {cr['views'][v]}" for b, v in cr["exempt"]) + "."]
    co = rep["correlation"]
    L += ["", "## Per-band L/R correlation", "", "| band | before | after | delta |",
          "|---|---|---|---|"]
    for i, (p, q, r) in enumerate(zip(co["before"], co["after"], co["delta"])):
        L.append(f"| {i} | {_f(p)} | {_f(q)} | {_f(r)} |")
    t = rep["tilt"]
    L += ["", "## Tilt (dB/oct, 100 Hz to 10 kHz)", "",
          f"Stereo: before {_f(t['before']['stereo'])}, after {_f(t['after']['stereo'])}, "
          f"target {_f(t['target_stereo'])}, error {_f(t['error_stereo'])}. "
          f"M {_f(t['before']['M'])} -> {_f(t['after']['M'])}, "
          f"S {_f(t['before']['S'])} -> {_f(t['after']['S'])}. Sub shelf (stereo) "
          f"{_f(t['sub_shelf_before']['stereo'])} -> {_f(t['sub_shelf_after']['stereo'])} dB.", ""]
    L += ["## Notches", "", "| f0 Hz | Q | channel | flag | max cut dB | rendered | spans (s) |",
          "|---|---|---|---|---|---|---|"]
    for n in rep["notches"]:
        sp = ", ".join(f"{p:.1f}-{q:.1f}" for p, q in n["spans"][:6]) + (
            " ..." if len(n["spans"]) > 6 else "")
        why = "" if n["rendered"] else f" ({n['reason']})"
        L.append(f"| {_f(n['f0_hz'], 1)} | {_f(n['q'], 1)} | {n['channel']} | {n['flag']} | "
                 f"{_f(n['max_cut_db'], 1)} | {'yes' if n['rendered'] else 'no'}{why} | {sp} |")
    mg = (rep.get("planner") or {}).get("merged_notches") or []
    if mg:
        L += ["", "Rendered (merged) notches: " + "; ".join(
            f"{_f(m_['f0_hz'], 1)} Hz Q {_f(m_['q'], 1)} max cut {_f(m_['max_cut_db'], 1)} dB "
            f"from candidates {', '.join(str(i) for i in m_['members'])}" for m_ in mg) + "."]
    L += _gain_reduction_md(rep)
    ab = rep["ab"]
    L += ["", "## A/B", "", f"Original gain {_f(ab['original_gain_db'])} dB, master gain "
          f"{_f(ab['master_gain_db'])} dB; matched S95 {_f(ab['original_matched_s95'])} / "
          f"{_f(ab['master_matched_s95'])} LUFS. Files: {', '.join(ab['files'])}.", ""]
    if rep.get("distance_to_master"):
        L += _distance_md(rep["distance_to_master"])
    if rep["plots"]:
        L += ["Plots: " + ", ".join(rep["plots"]), ""]
    return "\n".join(L)


def _gain_reduction_md(rep: dict) -> list[str]:
    """The time-domain GR summary: per-band compressor max / mean and band limiter max as
    tables, glue and limiter as lines."""
    g = rep.get("gain_reduction") or {}
    if not g:
        return []
    L = ["", "## Gain reduction", ""]
    mx, mn = g.get("multiband_comp_gr_max_db"), g.get("multiband_comp_gr_mean_db")
    if mx:
        cg = g.get("multiband_ceiling_gr_max_db")
        sm = g.get("multiband_side_makeup_db") or []
        L += ["| band | compressor max M/S | compressor mean M/S | band limiter max M/S | "
              "side makeup |", "|---|---|---|---|---|"]
        for b in range(len(mx)):
            lim_s = f"{_f(cg[b][0])}/{_f(cg[b][1])}" if cg else "-"
            L.append(f"| {_band_name(rep, b)} | {_f(mx[b][0])}/{_f(mx[b][1])} | "
                     f"{_f(mn[b][0])}/{_f(mn[b][1])} | {lim_s} | "
                     f"{_f(sm[b] if b < len(sm) else None, 1)} |")
        L.append("")
    if "glue_gr_max_db" in g:
        L.append(f"- Glue: max {_f(g['glue_gr_max_db'])} dB, mean {_f(g['glue_gr_mean_db'])} dB, "
                 f"mix {_f(g.get('glue_mix'))}.")
    lm = g.get("limiter")
    if lm:
        passes = lm.get("relimit_passes") or []
        L.append(f"- Limiter: max {_f(lm.get('max_gain_reduction_db'))} dB, clipper curve max "
                 f"{_f(lm.get('clip_max_reduction_db'))} dB on {lm.get('clip_spans', 0)} spans, "
                 f"true peak 8x {_f(lm.get('true_peak_8x_dbtp'))} / 32x "
                 f"{_f(lm.get('true_peak_32x_dbtp'))} dBTP, trim {_f(lm.get('trim_db'))} dB, "
                 f"{len(passes)} re-limit pass(es)"
                 + (" (" + ", ".join(f"{_f(p.get('max_reduction_db'))} dB on {p.get('samples')} "
                                     "samples" for p in passes) + ")" if passes else "") + ".")
    return L


def _distance_md(d: dict) -> list[str]:
    L = ["## Distance to the reference master", "", "| metric | ours | theirs | ours - theirs |",
         "|---|---|---|---|"]
    for k, v in d.items():
        if isinstance(v, dict) and "ours" in v:
            L.append(f"| {k} | {_f(v['ours'])} | {_f(v['theirs'])} | {_f(v['diff'])} |")
    if d.get("ltas_rms_diff_db") is not None:
        L.append(f"| LTAS RMS difference (dB) | - | - | {_f(d['ltas_rms_diff_db'])} |")
    cr, co = d.get("crest_p95_diff_db") or [], d.get("band_corr_diff") or []
    for b, row in enumerate(cr):
        L.append(f"| crest p95 band {b} M/S/stereo (dB) | - | - | "
                 + "/".join(_f(v) for v in row) + " |")
    for b, v in enumerate(co):
        L.append(f"| correlation band {b} | - | - | {_f(v)} |")
    return L + [""]
