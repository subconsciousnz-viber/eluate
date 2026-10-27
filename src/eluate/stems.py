"""Stems, phase A (sprint 2026-10-21 WP6; docs/noise-and-stems.md 2): master the sum of a
stem set with the stems profile's tighter ranges and name the stems responsible for every
limit the sum pass hit. No stem audio is changed.

Contract:
- `read_stems(paths, log=None) -> (stems, fs)`: stems (k, n, 2) float64; every file must
  share the sample rate and the length, except a render tail: a stem longer than the
  shortest by at most TRIM_MAX_S, whose extra samples peak below TRIM_TAIL_DB, is cut to it
  (logged); otherwise ValueError naming the stem. Mono is duplicated to two channels.
- `null_test(mix, total, fs) -> dict`: aligns the stems' sum `total` to the mix by one
  sample offset within +-NULL_MAX_OFFSET_MS (the per-channel cross-correlations over the
  first NULL_XCORR_S of both, summed) and one least-squares gain over the overlap, then
  residual_db = 10 log10(energy(mix - g total) / energy(mix)); either signal's tail
  outside the overlap, past the abs(offset) samples the shift drops, is residual (a mix
  that is only the stems' opening does not pass). verdict "pass" below
  NULL_PASS_DB; "processed" up to NULL_MISMATCH_DB (bus processing on the mix, or a stem
  missing); "mismatch" above (these stems do not make this mix). worst_band_hz is the
  octave band (31.5 Hz to 16 kHz) where the residual is largest against the mix there.
  A silent mix (no energy: nothing to null against) gives verdict "silent", residual_db
  and gain_db None, whether the stems are silent too or not (`null_message` says which).
- `attribute(work, stems, names, fs) -> dict`: reads the sum pass's report.json and
  plan.json in work and, for each limit it hit, ranks the stems (section 6.3):
  EQ sections at their bounds (each stem's projection share of the sum in the section's
  range, Re<stem, sum> / <sum, sum> from the cross spectra: the shares sum to 1 and a
  negative one cancels against the rest; and its level there against the sum's: the
  stem carrying the range is the one to brighten or cut);
  multiband bands at their exception or band limiter at its cap, and speclim cuts past
  SPECLIM_CUT_DB (each stem's band peak at the ATTR_MOMENTS moments of largest GR in that
  band and channel, against the sum's); the clipper short of its target or the limiter
  on its outlier cap (each stem's full-band 8x peak at the limiter's largest-GR moments);
  notches found (each stem's PSD at f0 over the notch's spans against its neighbours a
  third of an octave either side); a crest or section flag (each stem's crest and its
  projection share of each section's sum); noise exposed or present (each stem's own floor,
  noise.floor). Every entry: {"limit", "detail", "stems": [{"name", ...}] ranked}.
- `is_limit(entry) -> bool`: whether an attribution entry is a real limit, one a stem
  correction can act on and the stems loop scores: input EQ at a bound, a multiband band
  at its exception or cap, a frequency-dependent limiter cut, the clipper short or the
  limiter on its outlier cap, a notch found and not rendered. Not: the crest / section row
  (informational), noise (no correction acts on it), a rendered notch (the master took it).
- `pass_score(report, attribution) -> dict`: a stems pass's score (lower is better):
  rms_shortfall_db (report.json loudness: target_rms_dbfs - after_rms_dbfs, floored at 0)
  + LIMIT_WEIGHT per real limit; {"score", "rms_shortfall_db", "limits": [limit text]}.
- `write_report(work, rep)`: stems-report.json and stems-report.md; with rep["passes"]
  (cli.do_master_stems' loop) a Passes section: each pass's score, limits and fixes, the
  pass kept and why the loop stopped.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import signal

from eluate import bands, io, meter, noise
from eluate.plan import Plan

NULL_PASS_DB = -40.0
NULL_MISMATCH_DB = -20.0
NULL_MAX_OFFSET_MS = 10.0
NULL_XCORR_S = 30.0
ATTR_MOMENTS = 8               # largest-GR moments a band's ranking reads
ATTR_WINDOW_S = 0.05           # +- around each moment
ATTR_SPACING_S = 0.25          # moments at least this far apart
SPECLIM_CUT_DB = 3.0
EDGE_DB = 0.05                 # a gain this close to its bound is at it
OCTAVES_HZ = (31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)
TRIM_MAX_S = 0.1               # longest render tail read_stems cuts to the shortest stem
TRIM_TAIL_DB = -60.0           # ... and only when it peaks below this


def _db(v):
    return 20.0 * math.log10(v) if v > 0 else -math.inf


def _pdb(v):
    return 10.0 * math.log10(v) if v > 0 else -math.inf


def read_stems(paths, log=None) -> tuple[np.ndarray, int]:
    """Lengths may differ by a render tail: every stem is cut to the shortest when what
    is cut is at most TRIM_MAX_S long and peaks below TRIM_TAIL_DB (a DAW export's tail,
    the maintainer's stem set: 481 samples at -94.8 dBFS); anything more is an error."""
    out, fs0, names = [], None, []
    for p in paths:
        x, fs = io.read(p)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        x = x[:, :2]
        if fs0 is None:
            fs0 = fs
        elif fs != fs0:
            raise ValueError(f"stem {Path(p).name}: {fs} Hz, the first stem is {fs0} Hz")
        out.append(x)
        names.append(Path(p).name)
    if not out:
        raise ValueError("no stems")
    n = min(x.shape[0] for x in out)
    for name, x in zip(names, out):
        extra = x.shape[0] - n
        if not extra:
            continue
        peak = _db(float(np.max(np.abs(x[n:]))))
        if extra > TRIM_MAX_S * fs0 or peak > TRIM_TAIL_DB:
            raise ValueError(f"stem {name}: {x.shape[0]} samples, the shortest has {n} "
                             f"({extra / fs0 * 1000:.0f} ms more, peaking at {peak:.1f} dBFS)")
        if log:
            log(f"stem {name}: its last {extra} samples cut to the shortest stem "
                f"({extra / fs0 * 1000:.1f} ms, peak {peak:.1f} dBFS)")
    return np.stack([x[:n] for x in out]), int(fs0)


# ---------------------------------------------------------------- null test

def _offset(mix: np.ndarray, total: np.ndarray, fs: int) -> int:
    """The lag (samples) to shift total by so it lines up with mix (within the limit). The
    channels correlate jointly (a mono fold would null a side-only signal)."""
    m = min(len(mix), len(total), int(NULL_XCORR_S * fs))
    lim = int(round(NULL_MAX_OFFSET_MS * 1e-3 * fs))
    c = sum(signal.correlate(mix[:m, ch], total[:m, ch], mode="full", method="fft")
            for ch in range(mix.shape[1]))
    lags = np.arange(-m + 1, m)
    sel = np.abs(lags) <= lim
    return int(lags[sel][np.argmax(c[sel])])


def null_test(mix: np.ndarray, total: np.ndarray, fs: int) -> dict:
    mix = np.asarray(mix, dtype=np.float64)
    total = np.asarray(total, dtype=np.float64)
    if not np.any(mix):
        return {"residual_db": None, "gain_db": None, "offset_samples": 0, "verdict": "silent",
                "stems_silent": not np.any(total), "worst_band_hz": None,
                "worst_band_residual_db": None, "pass_db": NULL_PASS_DB,
                "mismatch_db": NULL_MISMATCH_DB}
    lag = _offset(mix, total, fs)
    if lag >= 0:
        a, b = mix[lag:], total[:len(total)]
    else:
        a, b = mix, total[-lag:]
    n = min(len(a), len(b))
    g = float(np.sum(a[:n] * b[:n]) / np.sum(b[:n] ** 2)) if np.any(b[:n]) else 1.0
    # residual over the longer of the two; the tail outside the overlap is residual too,
    # past the abs(lag) samples the shift itself drops off one end
    N = max(len(a), len(b))
    r = np.zeros((N, a.shape[1]))
    r[:n] = a[:n] - g * b[:n]
    t0 = n + abs(lag)
    r[t0:len(a)] = a[t0:]
    r[t0:len(b)] = g * b[t0:]
    a = np.concatenate([a, np.zeros((N - len(a), a.shape[1]))])
    res_db = _pdb(np.sum(r ** 2)) - _pdb(np.sum(a ** 2))
    verdict = ("pass" if res_db < NULL_PASS_DB else
               "processed" if res_db <= NULL_MISMATCH_DB else "mismatch")
    worst, worst_db = None, -math.inf
    if N >= 4096:
        nps = 8192 if N >= 8192 else 4096
        f, pr = signal.welch(r, fs, nperseg=nps, axis=0)
        _, pa = signal.welch(a, fs, nperseg=nps, axis=0)
        pr, pa = pr.sum(axis=1), pa.sum(axis=1)          # per channel, not a mono fold
        for fc in OCTAVES_HZ:
            if fc * math.sqrt(2) >= fs / 2:
                break
            sel = (f >= fc / math.sqrt(2)) & (f < fc * math.sqrt(2))
            if not np.any(sel) or pa[sel].sum() <= 0:
                continue
            d = _pdb(pr[sel].sum()) - _pdb(pa[sel].sum())
            if d > worst_db:
                worst, worst_db = fc, d
    return {"residual_db": float(res_db), "gain_db": _db(abs(g)) if g else -math.inf,
            "offset_samples": lag, "verdict": verdict, "worst_band_hz": worst,
            "worst_band_residual_db": None if worst is None else float(worst_db),
            "pass_db": NULL_PASS_DB, "mismatch_db": NULL_MISMATCH_DB}


def null_message(t: dict) -> str:
    if t["verdict"] == "silent":
        return ("The mix and the stems are both silent: nothing to master."
                if t.get("stems_silent") else
                "The mix is silent, so the stems cannot be checked against it: give the mix "
                "these stems were printed from.")
    if t["verdict"] == "pass":
        return f"Stems null against the mix: residual {t['residual_db']:.1f} dB."
    if t["verdict"] == "processed":
        return (f"The stems do not null against the mix: residual {t['residual_db']:.1f} dB "
                f"(pass is below {NULL_PASS_DB:g} dB), largest around {t['worst_band_hz']:g} Hz. "
                "The mix has bus processing the stems lack, or a stem is missing; print the "
                "stems without mix-bus processing.")
    return (f"These stems do not make this mix: residual {t['residual_db']:.1f} dB "
            f"(above {NULL_MISMATCH_DB:g} dB).")


# ---------------------------------------------------------------- attribution

def _moments(gr: np.ndarray, rate_hz: float) -> np.ndarray:
    """Times (s) of the ATTR_MOMENTS largest reductions (gr dB <= 0), ATTR_SPACING_S apart."""
    order = np.argsort(gr)
    gap = int(round(ATTR_SPACING_S * rate_hz))
    out: list[int] = []
    for i in order:
        if gr[i] >= -1e-6 or len(out) >= ATTR_MOMENTS:
            break
        if all(abs(i - j) >= gap for j in out):
            out.append(int(i))
    return np.asarray(out, dtype=np.float64) / rate_hz


def _peak_at(a: np.ndarray, fs: int, times: np.ndarray, tp: bool = False) -> float:
    """Mean over the moments of the peak (dBFS) within +-ATTR_WINDOW_S of each."""
    w = int(round(ATTR_WINDOW_S * fs))
    vals = []
    for t in times:
        i = int(round(t * fs))
        seg = a[max(0, i - w):i + w]
        if seg.size == 0:
            continue
        v = meter.true_peak(seg, fs) if tp else _db(float(np.max(np.abs(seg))))
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else -math.inf


def _rank(names, vals, ref, key="peak_db", top=2, below=True) -> list[dict]:
    rows = [{"name": n, key: v, "below_sum_db": (ref - v) if below and math.isfinite(v) else None}
            for n, v in zip(names, vals) if math.isfinite(v)]
    rows.sort(key=lambda r: -r[key])
    return rows[:top]


def _band_moments_rank(stems, names, total, fs, xo, k, c, env_db, rate_hz, order, phase):
    times = _moments(env_db, rate_hz)
    if times.size == 0:
        return None, []
    sb = bands.split(bands.to_ms(total), fs, xo, order=order, phase=phase)
    ref = _peak_at(sb[k, :, c], fs, times)
    del sb
    vals = []
    for s in stems:
        b = bands.split(bands.to_ms(s), fs, xo, order=order, phase=phase)
        vals.append(_peak_at(b[k, :, c], fs, times))
        del b
    return times, _rank(names, vals, ref)


def _env_db(plan_env, n_ctrl: int) -> np.ndarray:
    v = plan_env.values
    return np.full(n_ctrl, float(v[0])) if v.size == 1 else np.asarray(v[:n_ctrl], dtype=np.float64)


def _range_level(x: np.ndarray, fs: int, lo: float, hi: float) -> float:
    f, p = signal.welch(x.mean(axis=1), fs, nperseg=8192 if len(x) >= 8192 else 2048)
    sel = (f >= lo) & (f < hi)
    return float(p[sel].sum()) if np.any(sel) else 0.0


def _range_rows(names, stems, total, fs: int, lo: float, hi: float) -> list[dict]:
    """Each stem's projection share of the sum over [lo, hi): Re<s, total> / <total, total>
    from the per-channel cross spectra. The shares sum to 1; a negative one cancels against
    the rest. Ranked by share."""
    nps = 8192 if len(total) >= 8192 else 2048
    f, pt = signal.welch(total, fs, nperseg=nps, axis=0)
    sel = (f >= lo) & (f < hi)
    e_tt = float(pt[sel].sum()) if np.any(sel) else 0.0
    e_sum = _range_level(total, fs, lo, hi)
    rows = []
    for nm, s in zip(names, stems):
        e = _range_level(s, fs, lo, hi)
        if e <= 0 or e_sum <= 0 or e_tt <= 0:
            continue
        _, pst = signal.csd(s, total, fs, nperseg=nps, axis=0)
        rows.append({"name": nm, "share": float(np.real(pst[sel]).sum()) / e_tt,
                     "level_re_sum_db": _pdb(e) - _pdb(e_sum)})
    rows.sort(key=lambda r: -r["share"])
    return rows


SECTION_RANGES = {"low_shelf": lambda s: (20.0, s["fc_hz"]),
                  "bell": lambda s: (s["fc_hz"] / 2 ** 0.5, s["fc_hz"] * 2 ** 0.5),
                  "band_shelf": lambda s: (s["lo_hz"], s["hi_hz"]),
                  "high_shelf": lambda s: (s["fc_hz"], 20000.0),
                  "high_shelf_side": lambda s: (s["fc_hz"], 20000.0),
                  "air_shelf": lambda s: (s["fc_hz"], 20000.0),
                  "air_shelf_side": lambda s: (s["fc_hz"], 20000.0)}


def _eq_bounds() -> dict:
    from eluate import planner
    b = (planner.PROFILES["stems"]["eq_bounds"] or {})
    return {"gain_db": b.get("gain_db", 3.0), "hs_mid_db": b.get("hs_mid_db", 4.5),
            "hs_side_db": b.get("hs_side_db", 6.0)}


def attribute(work, stems: np.ndarray, names, fs: int) -> dict:
    work = Path(work)
    rep = json.loads((work / "report.json").read_text(encoding="utf-8"))
    plan = Plan.load(work / "plan.json")
    total = stems.sum(axis=0)
    names = list(names)
    out: list[dict] = []
    n = total.shape[0]
    from eluate.plan import CONTROL_RATE_HZ, control_length
    n_ctrl = control_length(n, fs, CONTROL_RATE_HZ)

    # EQ sections at their bounds
    bnd = _eq_bounds()
    secs = ((rep.get("chain") or {}).get("inputeq") or {}).get("sections") or {}
    for name, sec in secs.items():
        if sec is None or name == "highpass" or sec.get("gain_db") is None:
            continue
        g = float(sec["gain_db"])
        cap = (bnd["hs_mid_db"] if name in ("high_shelf", "air_shelf") else
               bnd["hs_side_db"] if name in ("high_shelf_side", "air_shelf_side")
               else bnd["gain_db"])
        if not (g >= cap - EDGE_DB or g <= -bnd["gain_db"] + EDGE_DB):
            continue
        lo, hi = SECTION_RANGES[name](sec)
        rows = _range_rows(names, stems, total, fs, lo, hi)
        detail = {"gain_db": g, "bound_db": cap, "range_hz": [lo, hi], "section": name}
        # the total gain the planner wanted in the range (its range_deficit_db), so stemfix
        # sizes the stem shelf from the excess over the bound
        caps = (rep.get("planner") or {}).get("inputeq") or {}
        caps = caps.get("side_caps" if name.endswith("_side") else "top_caps") or {}
        key = "presence" if name.startswith("high_shelf") else "air" if name.startswith("air") else None
        wanted = (caps.get(key) or {}).get("range_deficit_db") if key else None
        if isinstance(wanted, (int, float)):
            detail["wanted_db"] = float(wanted)
        out.append({"limit": f"input EQ {name.replace('_', ' ')} {g:+.1f} dB, at its bound",
                    "detail": detail, "stems": rows[:2]})

    # multiband exception / band limiter at its cap
    mbp = plan.multiband
    rows_mb = ((mbp.budget or {}).get("result") or []) if mbp is not None else []
    for k, row in enumerate(rows_mb):
        for c, r in enumerate(row):
            if r.get("gated"):
                continue
            at_cap = (r.get("limit_db") or 0.0) >= (r.get("limit_cap_db") or math.inf) - EDGE_DB
            if not (r.get("exception") or at_cap):
                continue
            env = np.clip(_env_db(mbp.gr_db[k][c], n_ctrl), -6.0, 0.0)
            if mbp.ceiling_gr_db is not None:
                env = env + _env_db(mbp.ceiling_gr_db[k][c], n_ctrl)
            times, ranked = _band_moments_rank(stems, names, total, fs, mbp.crossovers_hz, k, c,
                                               env, CONTROL_RATE_HZ, mbp.order, mbp.phase)
            out.append({"limit": f"multiband band {k} {'MS'[c]} "
                                 + ("at its exception" if r.get("exception") else "band limiter at its cap"),
                        "detail": {"band": k, "channel": "MS"[c], "comp_db": r.get("comp_db"),
                                   "limit_db": r.get("limit_db"),
                                   "moments_s": None if times is None else times.tolist()},
                        "stems": ranked})

    # speclim cuts past SPECLIM_CUT_DB
    sl = plan.speclim
    rows_sl = ((sl.budget or {}).get("result") or []) if sl is not None else []
    for k, row in enumerate(rows_sl):
        for c, r in enumerate(row):
            if (r.get("cut_db") or 0.0) <= SPECLIM_CUT_DB or sl.ceiling_gr_db is None:
                continue
            env = _env_db(sl.ceiling_gr_db[k][c], n_ctrl)
            times, ranked = _band_moments_rank(stems, names, total, fs, sl.crossovers_hz, k, c,
                                               env, CONTROL_RATE_HZ, sl.order, sl.phase)
            out.append({"limit": f"frequency-dependent limiter band {k} {'MS'[c]} cut "
                                 f"{r['cut_db']:.1f} dB",
                        "detail": {"band": k, "channel": "MS"[c], "cut_db": r["cut_db"],
                                   "moments_s": None if times is None else times.tolist()},
                        "stems": ranked})

    # clipper short of its target, limiter on its outlier cap
    warns = rep.get("warnings") or []
    lim_bound = next((w.get("bound") for w in warns if w.get("stage") == "limiter"), None)
    clip_w = any(w.get("stage") == "clipper" for w in warns)
    if (clip_w or lim_bound == "outlier") and (work / "limiter_gr.npz").exists():
        z = np.load(work / "limiter_gr.npz")
        t, gr = np.asarray(z["t"]), np.asarray(z["gr_db"])
        rate = 1.0 / float(np.median(np.diff(t))) if t.size > 1 else CONTROL_RATE_HZ
        times = _moments(-np.abs(gr), rate) + (float(t[0]) if t.size else 0.0)
        ref = _peak_at(total, fs, times, tp=True)
        vals = [_peak_at(s, fs, times, tp=True) for s in stems]
        out.append({"limit": ("clipper short of its target" if clip_w else "")
                    + (" and " if clip_w and lim_bound == "outlier" else "")
                    + ("limiter on its outlier cap" if lim_bound == "outlier" else ""),
                    "detail": {"moments_s": times.tolist()},
                    "stems": _rank(names, vals, ref)})

    # notches found
    for nt in rep.get("notches") or []:
        f0, spans = nt.get("f0_hz"), nt.get("spans") or []
        if not f0 or not spans:
            continue
        rows = []
        for nm, s in zip(names, stems):
            seg = np.concatenate([s[int(a * fs):int(b * fs)] for a, b in spans]
                                 or [s[:0]])
            if len(seg) < 2048:
                continue
            f, p = signal.welch(seg.mean(axis=1), fs, nperseg=4096 if len(seg) >= 4096 else 2048)
            at = p[np.argmin(np.abs(f - f0))]
            nb = (np.abs(np.log2(np.maximum(f, 1e-9) / f0)) <= 1 / 3) & (np.abs(f - f0) > f0 * 0.03)
            if at <= 0 or not np.any(nb):
                continue
            rows.append({"name": nm, "excess_db": _pdb(at) - _pdb(float(np.median(p[nb])))})
        rows.sort(key=lambda r: -r["excess_db"])
        out.append({"limit": f"notch at {f0:.0f} Hz" + (" (rendered)" if nt.get("rendered") else " (found)"),
                    "detail": {"f0_hz": f0, "spans": spans,
                               **{k: nt[k] for k in ("channel", "q", "flag", "max_cut_db")
                                  if nt.get(k) is not None}},
                    "stems": rows[:2]})

    # crest or section flag
    flags = {f["kind"]: f for f in rep.get("flags_structured") or []}
    if (flags.get("crest") or {}).get("short") or (flags.get("section") or {}).get("short"):
        rows = [{"name": nm, "crest_db": meter.crest_db(s, fs)} for nm, s in zip(names, stems)]
        rows.sort(key=lambda r: r["crest_db"] if math.isfinite(r["crest_db"]) else math.inf)
        share = []
        for sec in rep.get("sections") or []:
            a, b = int(sec["start"] * fs), int(sec["end"] * fs)
            # projection share of the sum's energy (sums to 1, negative where a stem cancels)
            tot = float(np.sum(total[a:b] ** 2)) if b > a else 0.0
            e = [float(np.sum(s[a:b] * total[a:b])) / tot if tot > 0 else 0.0 for s in stems]
            share.append({"label": sec.get("label"), "start": sec["start"],
                          "share": dict(zip(names, e))})
        out.append({"limit": "crest low or section contrast lost",
                    "detail": {"section_share": share}, "stems": rows})

    # noise exposed or present
    nz = rep.get("noise") or {}
    if nz.get("status") in ("exposed", "present"):
        rows = [{"name": nm, "floor_db": float(noise.floor(s, fs)["db"])}
                for nm, s in zip(names, stems)]
        rows.sort(key=lambda r: -r["floor_db"] if math.isfinite(r["floor_db"]) else math.inf)
        out.append({"limit": f"noise {nz['status']}", "detail": {"after_db": nz.get("after_db")},
                    "stems": rows[:2]})
    return {"stems": names, "attribution": out}


# ---------------------------------------------------------------- pass score

LIMIT_WEIGHT = 0.5             # score per real limit, in dB of RMS shortfall
_LIMIT_PREFIXES = ("input EQ ", "multiband band ", "frequency-dependent limiter ",
                   "clipper short", "limiter on its outlier cap")


def is_limit(e: dict) -> bool:
    lim = e.get("limit") or ""
    if lim.startswith("notch "):
        return lim.endswith("(found)")
    return lim.startswith(_LIMIT_PREFIXES)


def pass_score(report: dict, attribution: list) -> dict:
    ld = report.get("loudness") or {}
    tgt, got = ld.get("target_rms_dbfs"), ld.get("after_rms_dbfs")
    short = (max(0.0, float(tgt) - float(got))
             if tgt is not None and got is not None and math.isfinite(tgt) and math.isfinite(got)
             else 0.0)
    limits = [e["limit"] for e in attribution if is_limit(e)]
    return {"score": short + LIMIT_WEIGHT * len(limits), "rms_shortfall_db": short,
            "limits": limits}


# ---------------------------------------------------------------- report

def _f(v, nd=1):
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def _stem_line(e: dict, r: dict) -> str:
    if "below_sum_db" in r:
        return f"{r['name']}: peak {_f(r.get('peak_db'))} dBFS, {_f(r.get('below_sum_db'))} dB below the sum's"
    if "share" in r:
        if r["share"] < 0:
            return (f"{r['name']}: cancels against the rest there "
                    f"({100 * r['share']:.0f} % share of the sum)")
        return f"{r['name']}: {100 * r['share']:.0f} % share of the sum there"
    if "excess_db" in r:
        return f"{r['name']}: {_f(r['excess_db'])} dB above its neighbours at f0"
    if "crest_db" in r:
        return f"{r['name']}: crest {_f(r['crest_db'])} dB"
    if "floor_db" in r:
        return f"{r['name']}: floor {_f(r['floor_db'])} dB (BS.468 weighted)"
    return r["name"]


def write_report(work, rep: dict) -> None:
    from eluate import evaluate
    work = Path(work)
    rep = evaluate._clean(rep)
    io.write_json(work / "stems-report.json", rep)
    L = ["# Stems report", "", "Stems: " + ", ".join(rep["stems"]) + ".", ""]
    nt = rep.get("null_test")
    if nt is not None:
        L += ["## Null test", "", null_message(nt), ""]
    L += ["## Limits the sum hit, and the stems behind them" if not rep.get("kept") else
          f"## Limits the kept pass ({rep['kept']}) hit, and the stems behind them", ""]
    if not rep["attribution"]:
        L.append("None: the sum met its targets within the stems profile's ranges.")
    for e in rep["attribution"]:
        L.append(f"- {e['limit']}: " + ("; ".join(_stem_line(e, r) for r in e["stems"])
                                       or "no stem stands out") + ".")
    passes = rep.get("passes")
    if passes:
        L += ["", "## Passes", "",
              f"Score: the RMS shortfall to the target in dB, plus {LIMIT_WEIGHT:g} per limit "
              "hit (lower is better).", ""]
        for p in passes:
            kept = " (kept)" if p["index"] == rep.get("kept") else ""
            what = "the plain sum" if p["index"] == 0 else "corrected stems"
            L.append(f"- Pass {p['index']}{kept}, {what}: score {_f(p['score'], 2)} "
                     f"(RMS {_f(p['rms_shortfall_db'], 2)} dB short, "
                     f"{len(p['limits'])} limit{'s' if len(p['limits']) != 1 else ''}"
                     + (": " + "; ".join(p["limits"]) if p["limits"] else "") + ").")
            for line in p.get("fix_lines") or []:
                L.append(f"  - {line}")
        if rep.get("stopped"):
            L += ["", rep["stopped"]]
    io.write_text(work / "stems-report.md", "\n".join(L) + "\n")
