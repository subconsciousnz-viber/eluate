"""Sub side-strip (spec 3.5 stage 2).

Contract: `apply(x, fs, plan) -> y`, pure, stereo (n, 2) in, float64 (n, 2) out.
- M, S = bands.to_ms(x); S_out = S - LP_zp(S), LP_zp = sosfiltfilt(butter(order,
  fc_hz)) with zero-padded edges (the high part of bands.split(S, fs, [fc_hz])),
  i.e. the squared Butterworth high-pass |HP|^2 = r^2N / (1 + r^2N), r = f / fc:
  -6.02 dB at fc, zero phase. y = bands.to_lr(M, S_out).
- Never mono-sums: M is passed through untouched (it changes only by the M/S
  round-trip rounding, < 1e-15), so L + R is unchanged at every frequency, and S
  above fc is kept (|HP|^2 > -0.02 dB above 4.6 fc at order 2).
- fc is a plan scalar. The analysis suggestion (lowest frequency where S/M
  exceeds -20 dB, bounded 60 to 150 Hz) is the planner's job; this stage only
  requires 0 < fc < fs/2.
"""

from __future__ import annotations

import numpy as np

from eluate import bands
from eluate.plan import SidestripPlan


def apply(x: np.ndarray, fs: float, plan: SidestripPlan) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("sidestrip needs stereo (n, 2) audio")
    ms = bands.to_ms(x)
    lo_hi = bands.split(ms[:, 1], fs, [plan.fc_hz], order=plan.order, phase="zero")
    ms[:, 1] = lo_hi[1]
    return bands.to_lr(ms)
