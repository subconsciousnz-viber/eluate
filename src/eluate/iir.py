"""Blocked IIR filtering with a subnormal flush (filter-maths audit F5, 2026-11-10).

An IIR filter's state decays exponentially after its input stops; in digital silence it
runs down into the subnormal range (below 2.2e-308), where every multiply is 10 to 100
times slower on x86. The audit measured the zero-phase splits and the input EQ 9 to 21
times slower on a track that is mostly silence (bands.split zero 6.8 s against 0.68 s
for the same length of noise).

Contract:
- `sosfilt(sos, x, zi=None, block=BLOCK)` -> (y, zf): scipy.signal.sosfilt along axis 0
  (x (n,) or (n, ...)), run in blocks of `block` samples with the state carried from
  block to block. zi None starts from rest. sosfilt is a per-sample recursion, so
  carrying the state reproduces one whole-array call bit for bit. Two departures:
  1. Flush: after a block, a state whose largest magnitude is below FLUSH_BELOW
     (1e-150) is set to exactly 0. A block whose input is all zero is run in
     SILENT_BLOCK sub-blocks with the same check after each, so a decaying tail is
     flushed long before it can reach the subnormal range (a sub-block cannot decay by
     the 158 decades between 1e-150 and 2.2e-308 for any filter in Eluate except the
     fastest poles, which then spend at most one sub-block there).
  2. Skip: a block (or sub-block) whose input is all zero with the state exactly 0 is
     not filtered; its output is 0, which is what the filter computes from rest.
  Without a flush the output equals scipy's bit for bit; a flush changes it by at
  most the flushed state times the filter's gain, |dy| < 1e-150 x gain, so the
  output differs from scipy's by far less than 1e-12 (tested), and digital silence
  stays exactly 0 once the tail is flushed.
- `sosfiltfilt_zero(sos, x, pl, block=BLOCK)`: the zero-phase pass Eluate uses
  everywhere (bands.split, inputeq, notch), scipy.signal.sosfiltfilt(sos,
  pad(x, pl zeros either side), axis 0, padtype=None)[pl:pl + n], computed with
  `sosfilt` above: forward from sosfilt_zi * the first padded sample (0: from rest),
  backward from sosfilt_zi * the forward output's last sample, as scipy does.
  Bit-identical to scipy when no flush occurs (music without digital silence).
"""

from __future__ import annotations

import numpy as np
from scipy import signal

BLOCK = 1 << 16          # samples per block (1.4 s at 48 kHz)
SILENT_BLOCK = 1024      # sub-block length while the input is digital silence
FLUSH_BELOW = 1e-150     # a state below this (max |zi|) is set to exactly 0


def _run(sos, x, zi, out, block):
    """Filter x into out (same shape) in place, block by block; returns the final state."""
    n = x.shape[0]
    zero = not np.any(zi)
    a = 0
    while a < n:
        b = min(n, a + block)
        xb = x[a:b]
        if not np.any(xb):
            if zero:
                out[a:b] = 0.0
                a = b
                continue
            # silence after signal: short sub-blocks until the tail is flushed
            for s0 in range(a, b, SILENT_BLOCK):
                s1 = min(b, s0 + SILENT_BLOCK)
                if zero:
                    out[s0:s1] = 0.0
                    continue
                out[s0:s1], zi = signal.sosfilt(sos, x[s0:s1], axis=0, zi=zi)
                if np.max(np.abs(zi)) < FLUSH_BELOW:
                    zi = np.zeros_like(zi)
                    zero = True
            a = b
            continue
        out[a:b], zi = signal.sosfilt(sos, xb, axis=0, zi=zi)
        zero = False
        if np.max(np.abs(zi)) < FLUSH_BELOW:
            zi = np.zeros_like(zi)
            zero = True
        a = b
    return zi


def _zi_shape(sos, x):
    return (sos.shape[0], 2) + x.shape[1:]


def sosfilt(sos: np.ndarray, x: np.ndarray, zi: np.ndarray | None = None,
            block: int = BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """scipy.signal.sosfilt(sos, x, axis=0, zi) in blocks with the subnormal flush
    (module contract); returns (y, zf). zi None starts from rest."""
    sos = np.asarray(sos, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if zi is None:
        zi = np.zeros(_zi_shape(sos, x))
    out = np.empty_like(x)
    zf = _run(sos, x, np.array(zi, dtype=np.float64, copy=True), out, max(1, int(block)))
    return out, zf


def sosfiltfilt_zero(sos: np.ndarray, x: np.ndarray, pl: int,
                     block: int = BLOCK) -> np.ndarray:
    """Zero-phase sos filtering of x (n,) or (n, ...) along axis 0 with pl zeros padded
    either side, as scipy.signal.sosfiltfilt(sos, padded, axis=0, padtype=None)[pl:pl+n]
    (module contract)."""
    sos = np.asarray(sos, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    pl = int(pl)
    ext = np.pad(x, [(pl, pl)] + [(0, 0)] * (x.ndim - 1))
    zi0 = signal.sosfilt_zi(sos).reshape((sos.shape[0], 2) + (1,) * (x.ndim - 1))
    y = np.empty_like(ext)
    _run(sos, ext, zi0 * ext[0:1], y, block)
    del ext
    yr = y[::-1]
    z = np.empty_like(y)
    _run(sos, yr, zi0 * yr[0:1], z, block)
    del y, yr
    return np.ascontiguousarray(z[::-1][pl:pl + n])
