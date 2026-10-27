"""Input normalisation (docs/gain-staging.md 2 item 1): the first stage of the chain.

Contract: `apply(x, fs, plan) -> y`, pure, y = x * 10^(plan.gain_db / 20) in
float64 (a gain of 0 dB returns an unchanged copy). plan is a plan.InputPlan;
its record fields are not read.
"""

from __future__ import annotations

import numpy as np

from eluate.plan import InputPlan


def apply(x: np.ndarray, fs: float, plan: InputPlan) -> np.ndarray:
    y = np.array(x, dtype=np.float64, copy=True)
    if plan.gain_db != 0.0:
        y *= 10.0 ** (plan.gain_db / 20.0)
    return y
