"""Processing resampler for the render stages (spec 3.5 item 7, audit 05 Q1).

Contract: `upsample(x, L)` / `decimate(y, L)`, the aligned pair the limiter and
the saturator process through, separate from the metering prototype in
meter.py (passband 0.93 x Nyquist, designed for true-peak accuracy, which
removed 22.5 to 24 kHz from a signal the limiter did not touch).
- Prototype: Kaiser-windowed sinc at the oversampled rate, passband edge
  PASS = 0.98 x base Nyquist, stopband edge 1.0 x base Nyquist with >= 60 dB
  attenuation (scipy.signal.kaiserord), cutoff at the midpoint 0.99. Length is
  odd with the centre on a multiple of L / 2, so `upsample` is time-aligned:
  y[L i + p] ~ x(i + p / L). About 363 base-rate taps (2905 at 8x).
- Kaiser passband ripple at 60 dB is +-0.0087 dB per pass; the round trip
  decimate(upsample(x)) applies the filter twice. Measured per-bin gain of the
  round trip: see tests/test_stage_limiter.py (gate TRANSPARENT_DB up to
  0.98 x Nyquist). Between 0.98 and 1.0 x Nyquist the response falls through
  the transition band.
- Edges: the signal is embedded in silence (full linear convolution), as in
  meter.py. Filtering is overlap-add FFT convolution (scipy oaconvolve): the
  2905-tap 8x prototype made direct polyphase filtering 3.5 times slower.
- Memory (spec 1): both directions run per column through `eluate.oa.oa_stream`,
  an overlap-add that reproduces scipy.signal.oaconvolve block for block (same
  block size, same FFT calls, same additions), so the output is bit-identical
  while neither the zero-stuffed signal nor the all-blocks FFT array exists;
  where oa_stream declines (one block, or scipy without `_calc_oa_lens`) the
  whole-array oaconvolve is used.
  The oversampled output itself is still whole (the limiter's blocks keep it
  short).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy import signal

from eluate.oa import oa_stream

PASS = 0.98
STOP_DB = 60.0
TRANSPARENT_DB = 0.02   # documented round-trip bound below PASS x Nyquist (tested)


@lru_cache(maxsize=8)
def prototype(factor: int) -> np.ndarray:
    """Interpolation prototype with DC gain `factor` (h[L k + p] = phase p)."""
    width = (1.0 - PASS) / factor            # transition, fraction of the oversampled Nyquist
    numtaps, beta = signal.kaiserord(STOP_DB, width)
    numtaps = int(math.ceil(numtaps / factor) * factor) + 1
    cutoff = 0.5 * (PASS + 1.0) / factor
    return signal.firwin(numtaps, cutoff, window=("kaiser", beta)) * factor


def _as2d(x):
    x = np.asarray(x, dtype=np.float64)
    return x[:, None] if x.ndim == 1 else x


def _oa_stream(get, s1: int, h: np.ndarray, put) -> bool:
    """Streamed overlap-add, bit-identical to scipy.signal.oaconvolve (eluate.oa,
    which also handles scipy without `_calc_oa_lens` by declining)."""
    return oa_stream(get, s1, h, put)


def _zero_padded(v: np.ndarray):
    n = v.shape[0]

    def get(a, b):
        out = np.zeros(b - a)
        if a < n:
            seg = v[a:min(b, n)]
            out[:seg.size] = seg
        return out
    return get


def upsample(x: np.ndarray, factor: int) -> np.ndarray:
    """Interpolate by `factor`. Computed per column with a streamed overlap-add that is
    bit-identical to oaconvolve on the zero-stuffed signal (spec 1 memory: no
    whole-track zero-stuffed copy, no all-blocks FFT array)."""
    xa = _as2d(x)
    L = int(factor)
    h = prototype(L)
    c = (h.size - 1) // 2
    n = xa.shape[0]
    m = n * L
    y = np.empty((m, xa.shape[1]))
    for col in range(xa.shape[1]):
        v = xa[:, col]

        def get(a, b, v=v):            # zero-stuffed v (v[i] at L i) over [a, b)
            out = np.zeros(b - a)
            i0 = -(-a // L)
            i1 = min(n, -(-b // L))
            if i1 > i0:
                out[i0 * L - a:(i1 - 1) * L - a + 1:L] = v[i0:i1]
            return out

        def put(start, vals, col=col):
            a, b = max(start, c), min(start + vals.size, c + m)
            if b > a:
                y[a - c:b - c, col] = vals[a - start:b - start]

        if not _oa_stream(get, m, h, put):
            z = np.zeros(m)
            z[::L] = v
            y[:, col] = signal.oaconvolve(z[:, None], h[:, None], mode="full",
                                          axes=0)[c:c + m, 0]
    return y[:, 0] if np.ndim(x) == 1 else y


def decimate(y: np.ndarray, factor: int) -> np.ndarray:
    """Low-pass and keep every factor-th sample; streamed per column, bit-identical to
    oaconvolve(...)[c:c + len][::factor]."""
    ya = _as2d(y)
    L = int(factor)
    h = prototype(L) / L
    c = (h.size - 1) // 2
    m = ya.shape[0]
    k = -(-m // L)
    z = np.empty((k, ya.shape[1]))
    for col in range(ya.shape[1]):
        def put(start, vals, col=col):
            # full index i = c + L j for j in [0, k)
            j0 = max(0, -(-(start - c) // L))
            j1 = min(k, (start + vals.size - 1 - c) // L + 1)
            if j1 > j0:
                z[j0:j1, col] = vals[c + L * j0 - start:c + L * (j1 - 1) - start + 1:L]

        if not _oa_stream(_zero_padded(ya[:, col]), m, h, put):
            z[:, col] = signal.oaconvolve(ya[:, col:col + 1], h[:, None], mode="full",
                                          axes=0)[c:c + m][::L, 0]
    return z[:, 0] if np.ndim(y) == 1 else z
