"""eluate.oa's private scipy import (report 3.5): a scipy release that moves or renames
`_calc_oa_lens` must fail here, not fall back silently to the whole-array oaconvolve
(correct, but not streamed: the memory bound of spec 1 is lost)."""

import numpy as np
from scipy import signal

from eluate import oa


def test_calc_oa_lens_resolves_to_what_oaconvolve_uses():
    # the fallback is decided by exactly this name being None after the import
    assert oa._calc_oa_lens is not None, "scipy.signal._signaltools._calc_oa_lens moved"
    # and the block sizes must be the ones oaconvolve itself uses (bit-identity)
    assert signal.oaconvolve.__globals__.get("_calc_oa_lens") is oa._calc_oa_lens
    # so a long signal is streamed, not declined
    rng = np.random.default_rng(0)
    a, h = rng.standard_normal(100_000), rng.standard_normal(1000)
    out = np.zeros(a.size + h.size - 1)

    def get(i0, i1):
        z = np.zeros(i1 - i0)
        if i0 < a.size:
            seg = a[i0:min(i1, a.size)]
            z[:seg.size] = seg
        return z

    def put(start, v):
        out[start:start + v.size] = v

    assert oa.oa_stream(get, a.size, h, put)
    assert np.array_equal(out, signal.oaconvolve(a, h, mode="full"))
