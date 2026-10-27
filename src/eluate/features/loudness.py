"""Loudness over time (spec 3.3 row "loudness"), via meter.py only.

Contract:
- `analyse(x, fs, ctx=None)` returns
  - `momentary` / `short_term`: BS.1770 momentary (400 ms) and short-term (3 s)
    loudness in LUFS on the common grid, each window centred on the grid time
    (the track is embedded in silence, so windows that reach past the ends see
    zeros; fully silent windows read -inf). Computed by meter.momentary /
    meter.short_term on a zero-padded copy, which places meter's window j on
    grid frame j.
  - `summary`: meter.measure(x, fs) (integrated, S95, loud-section integrated,
    max M / S, LRA, true peak 8x and 4x, sample peak, PLR, PSR p95).
- `per_section(x, fs, segments, series)` gives, per segment {start, end}:
  integrated loudness of the 400 ms blocks of the whole-track K-weighted signal
  that lie inside the segment (meter's block powers and gating; -inf when no
  block fits), the
  power mean of the short-term series over the segment's grid frames, and the
  maximum momentary and short-term values there.
"""

from __future__ import annotations

import math

import numpy as np

from .. import meter
from ._common import Context, Grid, ensure_context


def _centred_series(x: np.ndarray, fs: float, grid: Grid, win_s: float, fn) -> np.ndarray:
    hop = grid.hop_s
    front = int(round((win_s / 2.0 - hop / 2.0) * fs))
    length = int(round(win_s * fs))
    need = int(math.ceil((grid.k - 1) * hop * fs)) + length + 2
    back = max(0, need - front - x.shape[0])
    xp = np.pad(x, ((front, back), (0, 0)))
    _, v = fn(xp, fs, hop_s=hop)
    v = np.asarray(v, dtype=np.float64)
    if v.size < grid.k:  # cannot happen with the padding above; kept as a guard
        v = np.pad(v, (0, grid.k - v.size), constant_values=-np.inf)
    return v[: grid.k]


def series(x: np.ndarray, fs: float, grid: Grid) -> dict:
    return {
        "momentary": _centred_series(x, fs, grid, meter.BLOCK_S, meter.momentary),
        "short_term": _centred_series(x, fs, grid, meter.SHORT_S, meter.short_term),
    }


def _power_mean_lufs(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return -math.inf
    return float(10.0 * np.log10(np.mean(10.0 ** (v / 10.0))))


def per_section(x: np.ndarray, fs: float, segments, s: dict, grid: Grid) -> list[dict]:
    # K-weight the whole track once and gate the 400 ms blocks lying inside each
    # section (audit 04 A15: slicing first restarted the K-filter at every
    # section's first sample, a step into a running signal)
    ps = meter._kpower(x, fs, None)
    b_start, b_len, b_p = meter._window_powers(ps, fs, meter.BLOCK_S, meter.HOP_S)
    out = []
    for seg in segments:
        i0 = int(round(seg["start"] * fs))
        i1 = int(round(seg["end"] * fs))
        inside = (b_start >= i0) & (b_start + b_len <= i1)
        sel = grid.frames_in(seg["start"], seg["end"])
        st = s["short_term"][sel]
        mo = s["momentary"][sel]
        out.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "label": seg.get("label"),
            "integrated_lufs": float(meter._lufs(meter._gated_mean_power(b_p[inside]))),
            "short_term_mean_lufs": _power_mean_lufs(st),
            "short_term_max_lufs": float(np.max(st)) if st.size else -math.inf,
            "momentary_max_lufs": float(np.max(mo)) if mo.size else -math.inf,
        })
    return out


def analyse(x: np.ndarray, fs: float, ctx: Context | None = None) -> dict:
    ctx = ensure_context(x, fs, ctx)
    out = series(ctx.x, ctx.fs, ctx.grid)
    out["t"] = ctx.grid.times
    out["summary"] = meter.measure(ctx.x, ctx.fs)
    return out
