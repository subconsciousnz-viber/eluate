"""Broadband tone EQ: one smooth zero-phase FIR from a target curve
(spec 3.5 stage 4, audit 05 A11).

Contract: `apply(x, fs, plan) -> y`, pure, (n, ch) in, float64 (n, ch) out, the
same filter on every channel.
- Target curve: node gains (plan.gain_db[k] at plan.nodes_hz[k], default
  1/3-octave nodes 30 Hz to 16 kHz) clipped to +-3 dB, interpolated over
  log2(f) by PCHIP (smooth, never outside the node values), held flat below
  the first and above the last node, then clipped to +-3 dB again: the bound
  is applied to the curve itself.
- FIR by frequency sampling (`design_fir`): N = 2^ceil(log2(fs)) points
  (65536 at 44.1 / 48 kHz, 131072 at 96 kHz, about 0.7 Hz resolution); H[k] =
  10^(curve(f_k) / 20), real and even, so h = IDFT(H) is real, symmetric and
  zero phase. h is windowed by w = (v * v) / max, the autocorrelation of a
  Hann window of length N/2 (support N - 1 < N, w[0] = 1). The windowed
  filter's response is then (1/N) sum_k H[k] W(omega - omega_k) with
  W = |V|^2 >= 0 and sum_k W(omega - omega_k) / N = w[0] = 1: a convex
  combination of the target samples at every frequency. So the measured
  response is zero phase and can never leave [min target, max target], i.e.
  +-3 dB. The price is smoothing by W (first null at 4 fs / N, 2.9 Hz at
  48 kHz) against a node spacing of 7.3 Hz at the 30 Hz node, and a long
  filter (N - 1 taps, 1.4 s at 48 kHz; the long tails carry only the sharp
  low-frequency detail of the curve).
- Static node gains: one FIR over the whole track (fftconvolve, zero-padded
  edges, centred). Time-varying node gains: the FIR is redesigned at block
  centres every HOP = 16384 samples (0.34 s at 48 kHz) and the block outputs
  are cross-faded with raised-cosine windows that sum to 1 (each block output is zero phase and
  bounded; the cross-fade is a time-varying mix of them).
- Under preserve-mix the stage is disabled (stages.render skips it). The
  planner sets the targets (reference slope -4.5 dB/oct PSD over 100 Hz to
  10 kHz plus user tilt, sub shelf separately, mapped onto these nodes; the
  planner no longer renders this stage, docs/gain-staging.md chunk 2).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy import signal
from scipy.interpolate import PchipInterpolator

from eluate.plan import TONE_MAX_DB, TonePlan

HOP = 16384


def fir_length(fs: float) -> int:
    return 1 << int(math.ceil(math.log2(fs)))


@lru_cache(maxsize=4)
def _window(N: int) -> np.ndarray:
    v = signal.windows.hann(N // 2, sym=True)
    w = signal.fftconvolve(v, v)
    w = np.maximum(w, 0.0)  # FFT rounding below zero at the far tails
    return w / w[N // 2 - 1]


def target_curve(f: np.ndarray, nodes_hz, gains_db) -> np.ndarray:
    """Target gain in dB at frequencies f (see the module contract)."""
    nodes = np.asarray(nodes_hz, dtype=np.float64)
    g = np.clip(np.asarray(gains_db, dtype=np.float64), -TONE_MAX_DB, TONE_MAX_DB)
    if nodes.size == 1:
        return np.full(np.shape(f), g[0])
    lf = np.log2(np.clip(np.maximum(f, 1e-9), nodes[0], nodes[-1]))
    return np.clip(PchipInterpolator(np.log2(nodes), g)(lf), -TONE_MAX_DB, TONE_MAX_DB)


def design_fir(nodes_hz, gains_db, fs: float) -> np.ndarray:
    """Zero-phase FIR (odd length N - 1, centre at index N/2 - 1)."""
    N = fir_length(fs)
    f = np.fft.rfftfreq(N, 1.0 / fs)
    H = 10.0 ** (target_curve(f, nodes_hz, gains_db) / 20.0)
    h = np.roll(np.fft.irfft(H, N), N // 2)[1:]
    return h * _window(N)


def _filter(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    c = (h.size - 1) // 2
    return signal.fftconvolve(x, h[:, None], mode="full", axes=0)[c:c + x.shape[0]]


def apply(x: np.ndarray, fs: float, plan: TonePlan) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    x2 = x[:, None] if squeeze else x
    n = x2.shape[0]
    if plan.nodes_hz[-1] >= fs / 2:
        raise ValueError(f"tone node {plan.nodes_hz[-1]} Hz is not below fs/2")
    centres = np.arange(0, max(n - 1, 0) + HOP, HOP)
    centres = centres[centres < n + HOP - 1]
    G = np.stack([e.at_samples(np.minimum(centres, n - 1), fs) for e in plan.gain_db], axis=1)
    G = np.clip(G, -TONE_MAX_DB, TONE_MAX_DB)
    if np.all(G == G[0]):
        if not np.any(G[0]):
            y = x2.copy()
        else:
            y = _filter(x2, design_fir(plan.nodes_hz, G[0], fs))
        return y[:, 0] if squeeze else y
    y = np.zeros_like(x2)
    half = (fir_length(fs) - 2) // 2
    cache: dict[bytes, np.ndarray] = {}
    for b, c in enumerate(centres):
        t0, t1 = max(0, c - HOP + 1), min(n, c + HOP)
        if t1 <= t0:
            continue
        key = G[b].tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = design_fir(plan.nodes_hz, G[b], fs)
        h = cache[key]
        s0, s1 = max(0, t0 - half), min(n, t1 + half)
        seg = _filter(x2[s0:s1], h)[t0 - s0:t1 - s0]
        t = np.arange(t0, t1)
        w = 0.5 * (1.0 + np.cos(np.pi * (t - c) / HOP))
        y[t0:t1] += w[:, None] * seg
    return y[:, 0] if squeeze else y
