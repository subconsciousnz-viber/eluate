"""Blocked IIR filtering with a subnormal flush (filter-maths audit F5, 2026-11-10).

An IIR filter's state decays exponentially after its input stops; in digital silence it
runs down into the subnormal range (below 2.2e-308), where every multiply is 10 to 100
times slower on x86. The audit measured the zero-phase splits and the input EQ 9 to 21
times slower on a track that is mostly silence (bands.split zero 6.8 s against 0.68 s
for the same length of noise).

Contract:
- `sosfilt(sos, x, zi=None, block=BLOCK)` -> (y, zf): scipy.signal.sosfilt along axis 0
  (x (n,) or (n, ...)) with the state carried. zi None starts from rest. The input is
  cut into blocks of `block` samples; consecutive blocks with any non-zero sample run
  as one sosfilt call, so music without digital silence is a single call, bit for
  bit scipy's (sosfilt is a per-sample recursion: carrying the state reproduces one
  whole-array call exactly). Two departures:
  1. Flush: after a run of non-silent blocks, a state whose largest magnitude is
     below FLUSH_BELOW (1e-150) is set to exactly 0. A silent block entered with a
     non-zero state is run in SILENT_BLOCK sub-blocks with the same check after each,
     so a decaying tail is flushed before it can spend long in the subnormal range
     (below 2.2e-308; only the fastest poles cross the 158 decades from 1e-150 within
     one sub-block, and then for at most that sub-block).
  2. Skip: silent blocks with the state exactly 0 are not filtered; their output is
     0, which is what the filter computes from rest.
  A flush changes the output by at most the flushed state times the filter's gain,
  |dy| < 1e-150 x gain: within 1e-12 of scipy (tested), and digital silence stays
  exactly 0 once the tail is flushed.
- `sosfiltfilt_zero(sos, x, pl, block=BLOCK)`: the zero-phase pass Eluate uses
  everywhere (bands.split, inputeq, notch), scipy.signal.sosfiltfilt(sos,
  pad(x, pl zeros either side), axis 0, padtype=None)[pl:pl + n], computed with
  `sosfilt` above, one column at a time: forward from sosfilt_zi * the first padded
  sample (0: from rest), backward from sosfilt_zi * the forward output's last sample,
  as scipy does. Bit-identical to scipy when no flush occurs (music without digital
  silence). Cost: about 10 to 30 % more than scipy on material without silence (the
  extra copies), 3x less on a split with 5 s of silence at either end of 20 s.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

BLOCK = 1 << 16          # samples per block (1.4 s at 48 kHz)
SILENT_BLOCK = 1024      # sub-block length while the input is digital silence
FLUSH_BELOW = 1e-150     # a state below this (max |zi|) is set to exactly 0


def _silent_sub(sos, x, zi, out):
    """x is all zero, zi not: run SILENT_BLOCK sub-blocks until the state flushes, then
    zeros. Writes out; returns the final state."""
    for s0 in range(0, x.shape[0], SILENT_BLOCK):
        s1 = min(x.shape[0], s0 + SILENT_BLOCK)
        out[s0:s1], zi = signal.sosfilt(sos, x[s0:s1], axis=0, zi=zi)
        if np.max(np.abs(zi)) < FLUSH_BELOW:
            out[s1:] = 0.0
            return np.zeros_like(zi)
    return zi


def _run(sos, x, zi, block):
    """Filter x along axis 0 from state zi (module contract): consecutive blocks with any
    non-zero input run as one sosfilt call, silent ones are skipped from rest or run in
    flushing sub-blocks. Returns (y, zf); y is sosfilt's own array when one call covers
    x (no copy)."""
    n = x.shape[0]
    nb = -(-n // block) if n else 0
    loud = [bool(np.any(x[i * block:(i + 1) * block])) for i in range(nb)]
    pieces = []
    i = 0
    while i < nb:
        j = i
        while j < nb and loud[j] == loud[i]:
            j += 1
        a, b = i * block, min(n, j * block)
        if loud[i]:
            y, zi = signal.sosfilt(sos, x[a:b], axis=0, zi=zi)
            if np.max(np.abs(zi)) < FLUSH_BELOW:
                zi = np.zeros_like(zi)
        elif not np.any(zi):
            y = np.zeros_like(x[a:b])
        else:
            y = np.empty_like(x[a:b])
            zi = _silent_sub(sos, x[a:b], zi, y)
        pieces.append(y)
        i = j
    if not pieces:
        return np.empty_like(x), zi
    return (pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)), zi


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
    return _run(sos, x, np.array(zi, dtype=np.float64, copy=True), max(1, int(block)))


def _zero_1d(sos, x, pl, block):
    n = x.shape[0]
    ext = np.pad(x, (pl, pl))
    zi0 = signal.sosfilt_zi(sos)
    y, _ = _run(sos, ext, zi0 * ext[0:1], block)
    del ext
    yr = y[::-1]
    z, _ = _run(sos, yr, zi0 * yr[0:1], block)
    del y, yr
    return z[::-1][pl:pl + n]


def sosfiltfilt_zero(sos: np.ndarray, x: np.ndarray, pl: int,
                     block: int = BLOCK) -> np.ndarray:
    """Zero-phase sos filtering of x (n,) or (n, ch) along axis 0 with pl zeros padded
    either side, as scipy.signal.sosfiltfilt(sos, padded, axis=0, padtype=None)[pl:pl+n]
    (module contract). Columns run one at a time (sosfilt treats them independently)."""
    sos = np.asarray(sos, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    block = max(1, int(block))
    if x.ndim == 1:
        return np.ascontiguousarray(_zero_1d(sos, x, int(pl), block))
    out = np.empty_like(x)
    flat = x.reshape(x.shape[0], -1)
    o2 = out.reshape(x.shape[0], -1)
    for c in range(flat.shape[1]):
        o2[:, c] = _zero_1d(sos, np.ascontiguousarray(flat[:, c]), int(pl), block)
    return out
