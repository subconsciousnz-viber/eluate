"""Per-band stereo decoherence (spec 3.3 row "stereo"; maintainer's decision:
L/R correlation plus S/M ratio).

Contract:
- Bands from bands.split as in crest.py; also the broadband signal (`full_*`).
- Per band and grid frame, over the 400 ms window centred on the grid time
  (in-signal samples only at the ends):
  - `corr`: Pearson correlation of L and R (means removed per window); NaN when
    either channel's variance is below 1e-12 of full scale or the frame is gated.
  - `sm_ratio_db`: 10 log10(E_S / E_M), energies of S = (L - R)/sqrt2 and
    M = (L + R)/sqrt2 (not mean-removed); -inf for pure mono, +inf for pure
    antiphase.
  - `gated`: band stereo RMS sqrt(mean (L^2 + R^2)/2) > -60 dBFS.
- Summaries over gated frames: `corr_summary[b]` = pooled Pearson coefficient
  sum(cov) / sqrt(sum(var_L) sum(var_R)) (energy weighted); `sm_ratio_summary_db[b]`
  = 10 log10(sum E_S / sum E_M); NaN when no frame is gated.
- `decoherence[b]` = clip(1 - corr_summary[b], 0, 1): 0 for mono, 1 for
  uncorrelated or anti-correlated. The multiband stage scales the high-side
  makeup by (1 - decoherence) (spec 3.5). When the correlation is not
  measurable (a hard-panned band, one channel silent, or a silent band) it is
  DECOHERENCE_DEFAULT = 1.0, i.e. no width makeup, and
  `decoherence_measurable[b]` is False (audit 04 A12: NaN reached the plan,
  whose dataclasses reject it).
- `side_strip_suggestion(ltas_m_db, ltas_s_db, centres, ltas_ref_db)`: see
  its docstring (spec 3.3 draft 0.4, audit 04 A13).
"""

from __future__ import annotations

import numpy as np

from ._common import Context, ensure_context, window_mean

WINDOW_S = 0.4
GATE_DBFS = -60.0
VAR_FLOOR = 1e-12
SIDE_STRIP_RATIO_DB = -20.0
SIDE_STRIP_RANGE = (60.0, 150.0)
SIDE_STRIP_DEFAULT = 100.0
SIDE_STRIP_GATE_DB = 60.0
SIDE_STRIP_SCAN_HZ = 160.0
SIDE_STRIP_SUB_HZ = 90.0  # the offending run must start below this
DECOHERENCE_DEFAULT = 1.0  # not measurable: no width makeup (the neutral choice for the multiband stage)


def _frame_stats(lr: np.ndarray, centres: np.ndarray, length: int) -> dict:
    left, right = lr[:, 0], lr[:, 1]
    # one column at a time: window_mean's cumsum runs down each column independently,
    # so this equals window_mean of the stacked (n, 5) array bit for bit while one
    # full-length product is alive instead of five plus their (n + 1, 5) cumsum
    makers = (lambda: left, lambda: right, lambda: left * left, lambda: right * right,
              lambda: left * right)
    ml, mr, ell, err, elr = (window_mean(f(), centres, length, inside=True) for f in makers)
    var_l = np.maximum(ell - ml * ml, 0.0)
    var_r = np.maximum(err - mr * mr, 0.0)
    cov = elr - ml * mr
    e_m = np.maximum(0.5 * (ell + err) + elr, 0.0)
    e_s = np.maximum(0.5 * (ell + err) - elr, 0.0)
    pw = 0.5 * (ell + err)
    return {"var_l": var_l, "var_r": var_r, "cov": cov, "e_m": e_m, "e_s": e_s, "pw": pw}


def _summaries(st: dict) -> dict:
    with np.errstate(divide="ignore", invalid="ignore"):
        rms_db = 10.0 * np.log10(st["pw"])
        gated = rms_db > GATE_DBFS
        ok = gated & (st["var_l"] > VAR_FLOOR) & (st["var_r"] > VAR_FLOOR)
        corr = np.where(ok, st["cov"] / np.sqrt(st["var_l"] * st["var_r"]), np.nan)
        corr = np.clip(corr, -1.0, 1.0)
        sm = np.where(gated, 10.0 * np.log10(st["e_s"] / st["e_m"]), np.nan)
        if np.any(ok):
            cs = float(np.sum(st["cov"][ok]) / np.sqrt(np.sum(st["var_l"][ok]) * np.sum(st["var_r"][ok])))
            cs = float(np.clip(cs, -1.0, 1.0))
        else:
            cs = np.nan
        sms = (float(10.0 * np.log10(np.sum(st["e_s"][gated]) / np.sum(st["e_m"][gated])))
               if np.any(gated) else np.nan)
    return {"corr": corr, "sm_ratio_db": sm, "gated": gated, "corr_summary": cs,
            "sm_ratio_summary_db": sms}


def decoherence_of(corr_summary: float) -> float:
    return float(np.clip(1.0 - corr_summary, 0.0, 1.0)) if np.isfinite(corr_summary) else np.nan


def side_strip_suggestion(ltas_m_db, ltas_s_db, centres, ltas_ref_db=None,
                          sub_sm_ratio_db: float | None = None) -> float:
    """Side high-pass crossover suggestion (spec 3.3 draft 0.4, audit 04 A13).

    A 1/3-octave band offends when its S PSD is within 20 dB of M (S/M > -20 dB)
    and the S level is within 60 dB of the maximum of `ltas_ref_db` (the stereo
    LTAS; M's when not given). Only side energy that starts in the sub counts:
    the offending region is the contiguous run of offending bands that begins at
    the lowest offending band centred below 90 Hz. No such band: the default
    100 Hz. Otherwise the upper edge (centre x 2^(1/6)) of the run's top band,
    clipped to [60, 150] Hz. Stereo pads and synths above 100 Hz are wanted width
    and do not move the crossover (revision 3 took the highest offending band up
    to 160 Hz and read 150 Hz on all five real premasters, whose sub is mono).
    Mono reads 100 Hz, uncorrelated stereo 150 Hz.
    `sub_sm_ratio_db` (the sub band's energy-weighted S/M from `analyse`), when
    given and below -20 dB, means the sub is mono and the answer is the default
    100 Hz whatever a weak band reads (a 79 Hz band at S/M -18 dB chained into the
    stereo low-mids and read 150 Hz on a track whose sub S/M is -28 dB)."""
    if sub_sm_ratio_db is not None and np.isfinite(sub_sm_ratio_db) and sub_sm_ratio_db < SIDE_STRIP_RATIO_DB:
        return SIDE_STRIP_DEFAULT
    m = np.asarray(ltas_m_db, dtype=np.float64)
    sd = np.asarray(ltas_s_db, dtype=np.float64)
    c = np.asarray(centres, dtype=np.float64)
    ref = m if ltas_ref_db is None else np.asarray(ltas_ref_db, dtype=np.float64)
    good = np.isfinite(ref) & (ref > -190.0)
    if not np.any(good):
        return SIDE_STRIP_DEFAULT
    gate = float(np.max(ref[good])) - SIDE_STRIP_GATE_DB
    off = (c <= SIDE_STRIP_SCAN_HZ) & np.isfinite(sd) & (sd - m > SIDE_STRIP_RATIO_DB) & (sd > gate)
    start = np.nonzero(off & (c < SIDE_STRIP_SUB_HZ))[0]
    if start.size == 0:
        return SIDE_STRIP_DEFAULT
    top = start[0]
    while top + 1 < c.size and off[top + 1]:
        top += 1
    f = float(c[top] * 2 ** (1 / 6))
    return float(np.clip(f, *SIDE_STRIP_RANGE))


def analyse(x: np.ndarray, fs: float, ctx: Context | None = None) -> dict:
    ctx = ensure_context(x, fs, ctx)
    g = ctx.grid
    length = int(round(WINDOW_S * ctx.fs))
    nb = len(ctx.fcs) + 1
    corr = np.full((nb, g.k), np.nan)
    sm = np.full((nb, g.k), np.nan)
    gated = np.zeros((nb, g.k), dtype=bool)
    cs = np.full(nb, np.nan)
    sms = np.full(nb, np.nan)
    for b, band in ctx.iter_bands():
        s = _summaries(_frame_stats(band, g.centres, length))
        corr[b], sm[b], gated[b] = s["corr"], s["sm_ratio_db"], s["gated"]
        cs[b], sms[b] = s["corr_summary"], s["sm_ratio_summary_db"]
        del band                # free this band before the iterator computes the next
    full = _summaries(_frame_stats(ctx.x, g.centres, length))
    return {
        "t": g.times, "corr": corr, "sm_ratio_db": sm, "gated": gated,
        "corr_summary": cs, "sm_ratio_summary_db": sms,
        "decoherence": np.array([d if np.isfinite(d) else DECOHERENCE_DEFAULT
                                 for d in (decoherence_of(c) for c in cs)]),
        "decoherence_measurable": np.isfinite(cs),
        "full_corr": full["corr"], "full_sm_ratio_db": full["sm_ratio_db"],
        "full_corr_summary": full["corr_summary"],
        "full_sm_ratio_summary_db": full["sm_ratio_summary_db"],
        "full_decoherence": (decoherence_of(full["corr_summary"]) if np.isfinite(full["corr_summary"])
                             else DECOHERENCE_DEFAULT),
        "full_decoherence_measurable": bool(np.isfinite(full["corr_summary"])),
        "window_s": WINDOW_S,
    }
