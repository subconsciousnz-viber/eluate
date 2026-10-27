"""Export quantisation (plan.OutputPlan; docs/gain-staging.md 3 "Ceiling"): the last
stage of the chain.

Contract: `apply(x, fs, plan) -> y`, pure, float64 (n, ch) out, holding exactly
the samples the file will hold (io.write with plan.subtype writes them without
further rounding):
- bits 32: the float32 cast (what a 32-bit float file holds).
- bits 16 / 24: with q = 2^(bits - 1), y = clip(round(x q + d), -q, q - 1) / q,
  round half to even, d = 0 (dither "none") or TPDF noise u1 - u2 with u1, u2
  uniform on [0, 1) (dither "tpdf", +-1 LSB, triangular). The noise comes from
  numpy's PCG64 seeded with plan.dither_seed, drawn in blocks of DITHER_BLOCK
  samples as one (block, ch, 2) array each, so the same (x, plan) gives the
  same output on every run and the stage holds one block of noise at a time.
"""

from __future__ import annotations

import numpy as np

from eluate.plan import OutputPlan

DITHER_BLOCK = 1 << 16


def apply(x: np.ndarray, fs: float, plan: OutputPlan) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if plan.bits == 32:
        return x.astype(np.float32).astype(np.float64)
    q = 2.0 ** (plan.bits - 1)
    x2 = x[:, None] if x.ndim == 1 else x
    y = np.empty(x2.shape)
    rng = np.random.Generator(np.random.PCG64(plan.dither_seed)) if plan.dither == "tpdf" \
        else None
    for a in range(0, x2.shape[0], DITHER_BLOCK):
        b = min(x2.shape[0], a + DITHER_BLOCK)
        v = x2[a:b] * q
        if rng is not None:
            u = rng.random((b - a, x2.shape[1], 2))
            v += u[..., 0] - u[..., 1]
        y[a:b] = np.clip(np.round(v), -q, q - 1.0) / q
    return y[:, 0] if x.ndim == 1 else y
