"""Render stages (spec 3.5) and the fixed-order chain.

Every stage is `apply(x, fs, section) -> y`: a pure function of the audio, its
sample rate and its plan section (plan.py), with no hidden state. `render`
runs them in the maintainer's order: input (normalisation gain), inputeq, notch,
sidestrip, multiband, tone, saturate, glue, speclim (frequency-dependent limiting, a
ceiling-only multiband), keyeq, limiter, output (export quantisation). The planner leaves tone None since chunk 2 of
docs/gain-staging.md: the input EQ replaces it.
"""

from __future__ import annotations

import numpy as np

from eluate.plan import STAGE_ORDER, Plan, check_environment
from eluate.stages import (gain, glue, inputeq, keyeq, limiter, multiband, notch, output,
                              saturate, sidestrip, tone)

STAGES = {
    "input": gain.apply,
    "inputeq": inputeq.apply,
    "notch": notch.apply,
    "sidestrip": sidestrip.apply,
    "multiband": multiband.apply,
    "tone": tone.apply,
    "saturate": saturate.apply,
    "glue": glue.apply,
    "speclim": multiband.apply,
    "keyeq": keyeq.apply,
    "limiter": limiter.apply,
    "output": output.apply,
}


def render(x: np.ndarray, fs: float, plan: Plan, on_stage=None, keep_stages: bool = False):
    """Run the chain. Returns (audio, per_stage).

    per_stage maps each stage that ran, in chain order, to its output (float64,
    a separate array per stage) when keep_stages is True, else to None, so by
    default only the final audio stays resident (spec 1, audit 05 Q15).
    on_stage(name, audio), if given, is called after each stage with that
    stage's output (to write it to disk, measure it, or keep it). A stage whose
    section is None is skipped (glue included); inputeq and tone are also skipped when
    plan.preserve_mix is set, keyeq when its section has enabled False. An enabled keyeq section
    raises NotImplementedError (stub). Audio must be stereo (n, 2) with n >= 2
    and finite; fs must equal plan.fs. A RuntimeWarning is issued when
    plan.environment differs from the running numpy / scipy / eluate
    versions: the render is bit-exact only in the recorded environment. Same versions on
    another platform, machine, Python or BLAS / LAPACK only log (plan.check_environment).
    Deterministic: the same (x, fs, plan, environment) gives bit-identical output.
    """
    if fs != plan.fs:
        raise ValueError(f"plan is for {plan.fs} Hz, audio is {fs} Hz")
    check_environment(plan.environment)
    y = np.array(x, dtype=np.float64, copy=True)
    if y.ndim != 2 or y.shape[1] != 2 or y.shape[0] < 2:
        raise ValueError(f"render needs stereo (n, 2) audio with n >= 2, got {y.shape}")
    if not np.all(np.isfinite(y)):
        raise ValueError("audio contains NaN or Inf")
    per_stage: dict[str, np.ndarray | None] = {}
    for name in STAGE_ORDER:
        section = getattr(plan, name)
        if (section is None or (name in ("inputeq", "tone") and plan.preserve_mix)
                or (name == "keyeq" and not section.enabled)):
            continue
        y = STAGES[name](y, fs, section)
        per_stage[name] = y if keep_stages else None
        if on_stage is not None:
            on_stage(name, y)
    return y, per_stage
