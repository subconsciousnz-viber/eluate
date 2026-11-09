"""Analysis pass (spec 3.3): one module per feature group plus `analyse`.

Contract of `analyse(x, fs, fcs=bands.DEFAULT_CROSSOVERS, hop_s=0.1, bpm=None,
fcs_from_key=None)`:
- `fcs_from_key`: a function of the key's tonic name (or None) returning the
  crossovers (planner.key_crossovers). When given, the key (pitch.key: chroma
  weighted by momentary-loudness power, neither of which reads the band
  split) is found right after loudness, before any band split, and the
  crossovers it returns replace `fcs` for the whole analysis;
  meta["crossover_rule"] is "key" and meta["crossover_tonic"] the tonic, else
  "fixed" and None. The key the pitch pass reports later is the same one (same
  chroma, same weights).
- `bpm`: the user's tempo (spec `--bpm`); when given it overrides tempo
  detection and seeds the beat grid, and structure.metrical reports how far the
  detected pulse was from it (structure.metrical_hypothesis).
- `x` is (n,), (n, 1) or (n, 2) L/R audio (mono is duplicated), any float
  dtype; NaN / Inf raise ValueError. Computation is float64 and deterministic
  (no random state is used anywhere; two runs on the same input give
  bit-identical results).
- Every time series is on the common grid: frame k covers [k h, (k + 1) h),
  stamped at its centre (k + 0.5) h, K = ceil(n / (h fs)) frames, h = hop_s.
  `t` holds the frame times.
- Order: loudness, tilt, crest, stereo, transients, structure (needs the onset
  envelope, kick onsets and short-term loudness), pitch (needs the beat and
  bar grid and the kick onsets), resonance (needs the sections for
  persistence), then per-section loudness.
- Returns a dict with keys `meta`, `t`, `loudness`, `tilt`, `crest`, `stereo`,
  `transients`, `structure`, `pitch`, `resonance`, `section_loudness`, `edges`
  and `plan_inputs`. Each module's own contract is in its docstring.
- `edges`: peak level (dBFS) of the first / last 10 ms and whether the track
  starts / ends abruptly (peak above -40 dBFS there): the zero-phase band
  split pre-rings on an abrupt start, which is reported, not suppressed
  (audit 02 D.9).
- `plan_inputs`: the scalars and lists the plan / solver need (spec 1, 3.5,
  3.6): S95, integrated, loud-section integrated, LRA, true peak (8x and 4x),
  PLR, PSR p95, sections with per-section loudness, tilt slope (M / S /
  stereo) and sub shelf, LTAS (stereo, for the preserve-mode +-1 dB check),
  crest p95 per band and view, per-band correlation, S/M ratio and
  decoherence (finite; `band_decoherence_measurable` flags the defaults), the
  side-strip crossover suggestion, the notch list (flag "resonance" or
  "musical", depth scale, channel M / S / both, Q at half the dB gain with
  its uncertainty), kick onsets with their scores, per-band spectral flatness
  (finite, flagged), key (combined edma mode / edmm tonic, with both
  profiles' labels and scores), bass root per beat with confidence and the
  keyeq gate (NSDF agreement), tempo (octave-checked), whether the beat grid is
  informative, beats, bars and edges.
- pitch.analyse receives only the kicks scoring >= transients.KICK_MASK_SCORE
  for its bass-root mask; resonance.analyse receives the key and the per-bar
  chords for its chord-invariance check; the side-strip suggestion receives the
  sub band's energy-weighted S/M ratio.
"""

from __future__ import annotations

import numpy as np

from .. import bands as _bands
from . import crest, loudness, pitch, resonance, stereo, structure, tilt, transients
from ._common import HOP_S, VIEWS, Context

__all__ = ["analyse", "Context", "crest", "loudness", "pitch", "resonance", "stereo",
           "structure", "tilt", "transients"]

EDGE_S = 0.010
ABRUPT_DBFS = -40.0


def _edges(x: np.ndarray, fs: float) -> dict:
    m = max(1, int(round(EDGE_S * fs)))
    with np.errstate(divide="ignore"):
        a = float(20 * np.log10(np.max(np.abs(x[:m])))) if np.any(x[:m]) else -np.inf
        b = float(20 * np.log10(np.max(np.abs(x[-m:])))) if np.any(x[-m:]) else -np.inf
    return {"start_peak_dbfs": a, "end_peak_dbfs": b,
            "abrupt_start": a > ABRUPT_DBFS, "abrupt_end": b > ABRUPT_DBFS}


def _frame_power(mom: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(mom), 10.0 ** (mom / 10.0), 0.0)


def analyse(x: np.ndarray, fs: float, fcs=_bands.DEFAULT_CROSSOVERS, hop_s: float = HOP_S,
            bpm: float | None = None, fcs_from_key=None) -> dict:
    ctx = Context(x, fs, fcs, hop_s)
    g = ctx.grid
    lo = loudness.analyse(ctx.x, ctx.fs, ctx)
    tonic = None
    if fcs_from_key is not None:
        tonic = pitch.key(ctx, _frame_power(lo["momentary"]))["key"].get("tonic")
        ctx.fcs = tuple(float(f) for f in fcs_from_key(tonic))
    out = {"meta": {"fs": ctx.fs, "n": ctx.x.shape[0], "duration_s": ctx.x.shape[0] / ctx.fs,
                    "hop_s": g.hop_s, "frames": g.k, "crossovers": ctx.fcs, "views": VIEWS,
                    "crossover_rule": "fixed" if fcs_from_key is None else "key",
                    "crossover_tonic": tonic},
           "t": g.times}
    out["loudness"] = lo
    out["tilt"] = tilt.analyse(ctx.x, ctx.fs, ctx)
    out["crest"] = crest.analyse(ctx.x, ctx.fs, ctx)
    out["stereo"] = stereo.analyse(ctx.x, ctx.fs, ctx)
    tr = transients.analyse(ctx.x, ctx.fs, ctx)
    out["transients"] = tr
    st = structure.analyse(ctx.x, ctx.fs, ctx, onset=(tr["onset_env_t"], tr["onset_env"]),
                           short_term=out["loudness"]["short_term"], onset_rate=tr["onset_rate"],
                           kick_onsets=tr["kick_onsets"], bpm=bpm)
    out["structure"] = st
    mom = out["loudness"]["momentary"]
    out["pitch"] = pitch.analyse(ctx.x, ctx.fs, ctx, beat_times=st["beat_times"],
                                 bar_times=st["bar_times"], kick_onsets=tr["kick_mask_onsets"],
                                 frame_power=_frame_power(mom))
    # nothing after this point reads these caches (resonance uses ctx.x and
    # ctx.ms, per_section ctx.x): drop them before the spectrograms (spec 1 memory)
    ctx.release("bands", "ms", "low_band_m", "mono_ar", "high_m", "onset_env", "chroma")
    out["resonance"] = resonance.analyse(ctx.x, ctx.fs, ctx, sections=st["segments"],
                                         key=out["pitch"]["key"], chords=out["pitch"]["chords"])
    out["section_loudness"] = loudness.per_section(ctx.x, ctx.fs, st["segments"], out["loudness"], g)
    out["edges"] = _edges(ctx.x, ctx.fs)

    lsum = out["loudness"]["summary"]
    tl = out["tilt"]
    sd = out["stereo"]
    br = out["pitch"].get("bass", {})
    out["plan_inputs"] = {
        "s95_lufs": lsum["s95_lufs"], "integrated_lufs": lsum["integrated_lufs"],
        "loud_section_lufs": lsum["loud_section_lufs"], "lra_lu": lsum["lra_lu"],
        "true_peak_dbtp": lsum["true_peak_dbtp"], "true_peak_4x_dbtp": lsum["true_peak_4x_dbtp"],
        "plr_db": lsum["plr_db"], "psr_p95_db": lsum["psr_p95_db"],
        "sections": out["section_loudness"],
        "tilt_slope_db_oct": tl["slope_db_oct"], "sub_shelf_db": tl["sub_shelf_db"],
        "ltas_centres_hz": tl["centres"], "ltas_stereo_db": tl["ltas_db"]["stereo"],
        "ltas_mid_db": tl["ltas_db"]["M"], "ltas_side_db": tl["ltas_db"]["S"],
        "crest_p95_db": out["crest"]["p95_db"], "crest_views": VIEWS,
        "band_corr": sd["corr_summary"], "band_sm_ratio_db": sd["sm_ratio_summary_db"],
        "band_decoherence": sd["decoherence"],
        "side_strip_hz": stereo.side_strip_suggestion(tl["ltas_db"]["M"], tl["ltas_db"]["S"],
                                                      tl["centres"], tl["ltas_db"]["stereo"],
                                                      float(sd["sm_ratio_summary_db"][0])),
        "band_decoherence_measurable": sd["decoherence_measurable"],
        "notches": [{k: r[k] for k in ("f0", "q", "q_uncertainty", "start", "end", "spans",
                                       "excess_db", "flag", "depth_scale", "channel")}
                    for r in out["resonance"]["resonances"] if r["flag"] in ("resonance", "musical")],
        "kick_onsets": tr["kick_onsets"], "kick_score": tr["kick_score"],
        "band_flatness": tl["flatness"], "band_flatness_measurable": tl["flatness_measurable"],
        "key": out["pitch"]["key"],
        "bass_root_midi": br.get("root_midi"), "bass_confidence": br.get("confidence"),
        "keyeq_eligible": out["pitch"].get("keyeq_eligible", False),
        "tempo_bpm": st["tempo_bpm"], "tempo_octave": st["tempo_octave"],
        "metrical": {k: v for k, v in st["metrical"].items()}, "kick_consistency": st["kick_consistency"],
        "grid_informative": st["grid_informative"], "bass_nsdf_agreement": out["pitch"].get("bass_nsdf_agreement"),
        "beat_times": st["beat_times"], "bar_times": st["bar_times"],
        "edges": out["edges"],
    }
    return out
