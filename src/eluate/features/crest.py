"""Per-band crest factor over time in M, S and stereo (spec 3.3 row "crest").

Contract:
- Bands from bands.split (zero phase, crossovers of the Context; default
  bands.DEFAULT_CROSSOVERS, 100 / 400 / 1600 / 6400 Hz, five bands). M/S per
  band via bands.to_ms.
- Per band b, view v in ("M", "S", "stereo") and grid frame k, over the 400 ms
  window centred on the grid time:
  - M / S: peak = max |v| (sample peak), RMS = sqrt(mean v^2).
  - stereo: peak = max(|L|, |R|), RMS = sqrt(mean (L^2 + R^2) / 2).
  Windows reaching past either end use the in-signal samples only.
  - `crest_db` = 20 log10(peak / RMS); `rms_dbfs` = 20 log10(RMS) (a
    full-scale square wave reads 0 dBFS, a full-scale sine -3.01).
  - `gated` = rms_dbfs > -60 dBFS; crest is NaN on ungated frames.
- `p95_db[b, v]` = 95th percentile of crest over gated interior frames
  (window at least 0.25 s inside both ends; all frames for very short input);
  NaN if none.
- Peaks are sample peaks, not true peaks: per band and view an 8x true-peak
  envelope would multiply the analysis cost by about 15 for a figure the
  solver's crest constraint compares before / after on the same terms.
Shapes: (bands, 3, K) for the series, (bands, 3) for the summary.
"""

from __future__ import annotations

import numpy as np

from ..bands import INV_SQRT2
from ._common import EPS, VIEWS, Context, ensure_context, window_max, window_mean

WINDOW_S = 0.4
GATE_DBFS = -60.0
EDGE_S = 0.25


def analyse(x: np.ndarray, fs: float, ctx: Context | None = None) -> dict:
    ctx = ensure_context(x, fs, ctx)
    g = ctx.grid
    length = int(round(WINDOW_S * ctx.fs))
    nb = len(ctx.fcs) + 1
    crest = np.full((nb, 3, g.k), np.nan)
    rms_db = np.full((nb, 3, g.k), -np.inf)
    for b, lr in ctx.iter_bands():

        def view(v):  # built one at a time: each is two full-length arrays
            if v == 2:
                return np.max(np.abs(lr), axis=1), 0.5 * (lr[:, 0] ** 2 + lr[:, 1] ** 2)
            sig = (lr[:, 0] + lr[:, 1]) * INV_SQRT2 if v == 0 else (lr[:, 0] - lr[:, 1]) * INV_SQRT2
            return np.abs(sig), sig * sig

        for v in range(3):
            pk_sig, pw_sig = view(v)
            pk = window_max(pk_sig, g.centres, length)
            pw = np.maximum(window_mean(pw_sig, g.centres, length, inside=True), 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = 10.0 * np.log10(pw)
                c = 20.0 * np.log10(np.maximum(pk, EPS)) - r
            rms_db[b, v] = r
            crest[b, v] = np.where(r > GATE_DBFS, c, np.nan)
            del pk_sig, pw_sig
        del lr                  # free this band before the iterator computes the next
    gated = rms_db > GATE_DBFS
    # p95 over the interior: frames whose window starts and ends at least
    # EDGE_S inside the track (audit 04 A17 / F19: the zero-phase split's pre-ring
    # at an abrupt start set the summary); all frames when none qualify
    dur = ctx.x.shape[0] / ctx.fs
    interior = (g.times - WINDOW_S / 2 >= EDGE_S) & (g.times + WINDOW_S / 2 <= dur - EDGE_S)
    if not np.any(interior):
        interior = np.ones(g.k, dtype=bool)
    p95 = np.full((nb, 3), np.nan)
    for b in range(nb):
        for v in range(3):
            vals = crest[b, v][gated[b, v] & interior]
            if vals.size:
                p95[b, v] = np.percentile(vals, 95.0)
    return {"t": g.times, "views": VIEWS, "crest_db": crest, "rms_dbfs": rms_db,
            "gated": gated, "p95_db": p95, "window_s": WINDOW_S, "gate_dbfs": GATE_DBFS}
