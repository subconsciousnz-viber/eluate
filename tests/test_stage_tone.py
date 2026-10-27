"""Gates for stages/tone.py (spec 3.5 stage 4; audit 05 A11)."""

import numpy as np
import pytest

from fixtures import synthetic as syn
from eluate.plan import Envelope, Plan, TonePlan, control_length, tone_nodes
from eluate.stages import tone

FS = 48000
RATES = [44100, 48000, 96000]
NODES = tone_nodes()
NN = len(NODES)


def _plan(gains_db):
    return TonePlan([Envelope.constant(g) for g in gains_db])


def _response(plan, fs, n=1 << 18):
    x = np.zeros((n, 1))
    x[n // 2] = 1.0
    h = tone.apply(x, fs, plan)[:, 0]
    H = np.fft.rfft(np.roll(h, -n // 2))
    return h, H, np.fft.rfftfreq(n, 1 / fs)


def test_default_nodes():
    assert NN == 28 and NODES[0] == 30.0 and abs(NODES[-1] - 16000.0) < 1e-9
    r = np.diff(np.log2(NODES))
    assert np.allclose(r, r[0])  # log-spaced, about 1/3 octave


def test_zero_gain_is_identity():
    x = syn.pink_noise(FS, FS, seed=1)
    assert np.array_equal(tone.apply(x, FS, _plan([0.0] * NN)), x)


@pytest.mark.parametrize("fs", RATES)
@pytest.mark.parametrize("node_hz", [100.0, 1000.0, 8000.0])
def test_single_node_reaches_its_gain(fs, node_hz):
    """Audit 05 A11: a +3 dB move on one node reaches +3 dB within 0.015 dB (filterbank:
    +2.01 dB); the residual at the 96 Hz node is the window's smoothing."""
    i = int(np.argmin(np.abs(np.array(NODES) - node_hz)))
    for g in (3.0, -3.0):
        gains = np.zeros(NN)
        gains[i] = g
        _, H, f = _response(_plan(gains), fs)
        mag = 20 * np.log10(np.abs(H.real))
        at = mag[np.argmin(np.abs(f - NODES[i]))]
        print(f"{fs} Hz node {NODES[i]:.0f} Hz {g:+.0f} dB: response at node {at:+.4f}")
        assert abs(at - g) < 0.015
        assert mag.max() <= 3.0 + 1e-9 and mag.min() >= -3.0 - 1e-9


@pytest.mark.parametrize("fs", RATES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_zero_phase_and_bounded(fs, seed):
    """Measured response: real (zero phase), symmetric impulse response, and inside
    [min node gain, max node gain] (a convex combination of the target)."""
    rng = np.random.default_rng(seed)
    plan = _plan(rng.uniform(-3, 3, NN))
    gains = np.array([e.values[0] for e in plan.gain_db])  # as stored (float32-representable)
    h, H, f = _response(plan, fs)
    n = h.size
    assert np.max(np.abs(h[n // 2 + 1:] - h[n // 2 - 1:0:-1][: n // 2 - 1])) < 1e-12
    assert np.max(np.abs(H.imag)) < 1e-10
    mag = 20 * np.log10(np.abs(H.real))
    print(f"{fs} Hz seed {seed}: nodes {gains.min():+.4f}..{gains.max():+.4f}, response "
          f"{mag.min():+.4f}..{mag.max():+.4f} dB")
    assert mag.min() >= gains.min() - 1e-9
    assert mag.max() <= gains.max() + 1e-9
    # the curve is followed, not diluted: at every node within 0.3 dB of the PCHIP curve
    f_n = np.array(NODES)
    at = mag[[int(np.argmin(np.abs(f - v))) for v in f_n]]
    curve = tone.target_curve(f[[int(np.argmin(np.abs(f - v))) for v in f_n]], NODES, gains)
    err = np.abs(at - curve)
    print(f"  node tracking error: max {err[1:].max():.4f} dB above the first node, "
          f"{err[0]:.4f} dB at 30 Hz")
    assert err[1:].max() < 0.3


def test_requests_beyond_3db_are_capped():
    _, H, _ = _response(_plan(np.tile([6.0, -9.0, 12.0, -12.0], NN // 4)), FS)
    mag = 20 * np.log10(np.abs(H.real))
    assert mag.max() <= 3.0 + 1e-9 and mag.min() >= -3.0 - 1e-9
    _, H, _ = _response(_plan([3.0] * NN), FS)
    assert np.allclose(20 * np.log10(np.abs(H.real)), 3.0, atol=1e-9)


@pytest.mark.parametrize("fs", RATES)
def test_tilt_shape(fs):
    gains = np.linspace(-3, 3, NN)
    _, H, f = _response(_plan(gains), fs)
    mag = 20 * np.log10(np.abs(H.real))
    band = (f > 40) & (f < 15000)
    assert np.all(np.diff(mag[band]) >= -1e-9)  # monotone
    slope = np.polyfit(np.log2(f[band]), mag[band], 1)[0]
    print(f"{fs} Hz tilt: fitted {slope:.4f} dB/oct (nodes {6 / np.log2(16000 / 30):.4f})")
    assert abs(slope - 6 / np.log2(16000 / 30)) < 0.05


def test_time_varying_gain_crossfade():
    """Node gains changing over time: regions where every block shares the static gains
    equal the static FIR output; the output stays finite and bounded."""
    n = 4 * FS
    nc = control_length(n, FS)
    g = np.zeros(nc)
    g[1000:3000] = 3.0  # +3 dB on one node from 1 s to 3 s
    i = int(np.argmin(np.abs(np.array(NODES) - 1000)))
    envs = [Envelope.constant(0.0) for _ in range(NN)]
    envs[i] = Envelope(g)
    x = syn.pink_noise(n, FS, seed=2)
    y = tone.apply(x, FS, TonePlan(envs))
    gains = np.zeros(NN)
    gains[i] = 3.0
    y_on = tone.apply(x, FS, _plan(gains))
    m = int(1.1 * tone.HOP)  # a sample sees the blocks centred within one hop of it
    on = slice(int(1.0 * FS) + m, int(3.0 * FS) - m)
    off = slice(0, int(1.0 * FS) - m)
    assert np.max(np.abs(y[on] - y_on[on])) < 1e-12
    assert np.max(np.abs(y[off] - x[off])) < 1e-12
    assert np.all(np.isfinite(y))


def test_nodes_above_nyquist_rejected():
    with pytest.raises(ValueError):
        tone.apply(np.zeros((1000, 2)), 22050, _plan([0.0] * NN))
    with pytest.raises(ValueError):
        Plan(fs=22050, tone=_plan([0.0] * NN))
