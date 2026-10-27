"""Streamed overlap-add convolution, bit-identical to scipy.signal.oaconvolve
(spec 1 memory, audit 07 A15 / F14).

Contract: `oa_stream(get, s1, h, put) -> bool` computes the full linear convolution
of a 1-D signal of length s1 with the 1-D kernel h, block by block exactly as
scipy.signal.oaconvolve does: the same block size, overlap and step from scipy's
`_calc_oa_lens`, the same `scipy.fft.rfftn` / `irfftn` calls over rows of `step`
samples zero-padded to the block size, and the same overlap-add (each output
sample is the block's head value plus the previous block's tail value, one
addition). Only ROWS_TARGET samples of blocks are transformed at a time.
- get(a, b) returns in1[a:b] with zeros beyond s1 (b - a is a whole number of
  steps); put(start, values) receives the full-convolution output in order,
  start being the index of values[0]; the last call ends at s1 + len(h) - 1.
- Returns False and does nothing when oaconvolve itself would not overlap-add
  (one block: it falls back to fftconvolve), when scipy's private
  `_calc_oa_lens` cannot be imported (a scipy release that moved it), or for an
  empty kernel; the caller then uses scipy.signal.oaconvolve on the whole
  array, which is correct, only not streamed. tests/test_oa.py asserts the
  import resolves to the function oaconvolve itself calls, so a scipy that
  moves it fails the tests instead of losing the memory bound silently.
- scipy's block choice always gives step >= overlap; for step < overlap (not
  reachable through _calc_oa_lens, kept for generality) a pending-tail
  accumulator spreads each tail over the following blocks; that order of
  additions is not scipy's (scipy cannot produce the case) and is only exact to
  rounding.
- Bit-identity holds for the scipy / numpy pair the plan's environment records
  (tests/test_memory_pass.py and tests/test_solver.py compare against
  oaconvolve over lengths 3 to 1 000 003).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import fft as sp_fft

try:
    from scipy.signal._signaltools import _calc_oa_lens
except ImportError:          # pragma: no cover - depends on the scipy release
    _calc_oa_lens = None

ROWS_TARGET = 1 << 18        # samples of blocks transformed per batch (memory only)
ROWS_MIN = 16


def oa_stream(get, s1: int, h: np.ndarray, put) -> bool:
    h = np.asarray(h, dtype=np.float64)
    s2 = h.size
    if _calc_oa_lens is None or s2 == 0 or s1 == 0:
        return False
    bs, overlap, step, _ = _calc_oa_lens(s1, s2)
    if step == s1 or overlap is None:
        return False
    total = s1 + s2 - 1
    nsteps = math.ceil((s1 + 1) / step)
    if (bs - overlap) * nsteps < total:
        nsteps += 1
    sp2 = sp_fft.rfftn(h[None, :], [bs], axes=[1])
    rows = max(ROWS_MIN, ROWS_TARGET // bs)
    tail = None                   # previous block's tail (step >= overlap)
    acc = np.zeros(max(overlap, 0))   # pending contributions (step < overlap)
    for j0 in range(0, nsteps, rows):
        j1 = min(nsteps, j0 + rows)
        blk = get(j0 * step, j1 * step).reshape(j1 - j0, step)
        ret = sp_fft.irfftn(sp_fft.rfftn(blk, [bs], axes=[1]) * sp2, [bs], axes=[1])
        if overlap == 0:
            heads = ret[:, :step]
        elif step >= overlap:
            heads = ret[:, :step].copy()
            # head + previous block's tail, as oaconvolve's ret[1:, :overlap] += overpart
            heads[1:, :overlap] = heads[1:, :overlap] + ret[:-1, step:]
            if tail is not None:
                heads[0, :overlap] = heads[0, :overlap] + tail
            tail = ret[-1, step:].copy()
        else:
            heads = np.empty((j1 - j0, step))
            for r in range(j1 - j0):
                row = ret[r]
                heads[r] = row[:step] + acc[:step]
                nxt = np.zeros(overlap)
                nxt[:overlap - step] = acc[step:]
                acc = nxt + row[step:]
        out = heads.reshape(-1)
        start = j0 * step
        if start >= total:
            return True
        put(start, out[:min(out.size, total - start)])
    return True
