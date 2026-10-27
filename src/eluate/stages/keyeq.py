"""Key and note-aware EQ (spec 3.5 stage 6). NOT IMPLEMENTED: stub.

Spec text (docs/spec-draft.md draft 0.3, section 3.5, item 6):

    `keyeq` (experimental, off by default): dynamic notch whose `f0` follows
    harmonics of the tracked bass root via a crossfaded bank of static
    zero-phase bandpasses (30 ms raised-cosine), never a swept IIR. Enabled only
    when pitch confidence >= 0.75.

Brief (chain item 5): master EQ moves that follow key or bass-root changes,
e.g. taming a resonance tied to the current root. Experimental; gated behind
the research findings.

Contract when built: `apply(x, fs, plan) -> y`, pure, like every other stage.
It depends on features/pitch.py (workstream C), which does not exist yet.
`stages.render` calls this whenever the plan carries a keyeq section, so a plan
with one raises until it is implemented; leave the section None.
"""

from __future__ import annotations

import numpy as np

from eluate.plan import KeyEqPlan


def apply(x: np.ndarray, fs: float, plan: KeyEqPlan) -> np.ndarray:
    raise NotImplementedError(__doc__)
