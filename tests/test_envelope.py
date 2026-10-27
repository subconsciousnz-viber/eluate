"""Gates for envelope.py (spec 3.4, section 7, draft 0.3)."""

import numpy as np
import pytest
from scipy import signal
from scipy.ndimage import maximum_filter1d

from fixtures import synthetic as syn
from eluate import envelope as env
from eluate import meter

FS = 48000


def _dense_greq(n=48000, seed=1):
    x = syn.pink_noise(n, FS, rms_dbfs=-14, seed=seed)[:, 0]
    x += syn.impulse_train(n, FS, 0.05, 0.9)[:, 0]
    return env.required_gain(maximum_filter1d(np.abs(x), 5), 10 ** (-6 / 20))


def _variants():
    return {
        "box": lambda g: env.box_smooth(g, attack=64, release=400),
        "box_norelease": lambda g: env.box_smooth(g, attack=64),
        "qp8": lambda g: env.qp_smooth(g, 8)[0],
        "qp64": lambda g: env.qp_smooth(g, 64)[0],
        "qp_attack_release": lambda g: env.qp_smooth(g, attack=64, release=400)[0],
    }


def test_required_gain():
    e = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
    assert np.allclose(env.required_gain(e, 1.0), [1, 1, 1, 0.5, 0.25])


@pytest.mark.parametrize("name", list(_variants()))
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_bound_holds_everywhere(name, seed):
    g_req = _dense_greq(seed=seed)
    g_req[:10] = 0.2  # dips at both edges
    g_req[-10:] = 0.3
    g = _variants()[name](g_req)
    assert g.shape == g_req.shape
    assert np.all(g <= np.minimum(g_req, 1.0))
    assert np.all(g >= env.G_MIN)


def test_bound_with_unclamped_input():
    g_req = 2.0 * _dense_greq(seed=4)  # values above 1 must still give g <= 1
    for name, fn in _variants().items():
        g = fn(g_req)
        assert np.all(g <= np.minimum(g_req, 1.0)), name


@pytest.mark.parametrize("name", list(_variants()))
def test_g_min_enforced_on_zero_input(name):
    g = _variants()[name](np.zeros(1000))
    assert np.all(g == env.G_MIN)


@pytest.mark.parametrize("fn", [lambda g: env.box_smooth(g, 8), lambda g: env.qp_smooth(g, 8),
                                lambda g: env.release_hull(g, 10.0)])
def test_non_finite_rejected(fn):
    g = np.ones(100)
    g[3] = np.nan
    with pytest.raises(ValueError):
        fn(g)


def test_box_bound_holds_before_rounding_clamp():
    g_req = _dense_greq(n=200000, seed=5)
    for attack, release in [(2, 0), (64, 0), (301, 2000), (1000, 50)]:
        raw, u = env._box_smooth_raw(g_req, attack, release)
        # cumulative-sum rounding only; measured 1.5e-11 at n = 200k
        assert np.max(raw - u) < 1e-9, (attack, release, np.max(raw - u))


@pytest.mark.parametrize("name", list(_variants()))
def test_smoother_than_raw(name):
    g_req = _dense_greq()
    g = _variants()[name](g_req)
    assert env.roughness(g) < 0.1 * env.roughness(np.minimum(g_req, 1.0))


@pytest.mark.parametrize("attack", [199, 200, 201])
def test_box_attack_extent_parity(attack):
    """Reduction starts exactly 2 * (attack // 2) samples before an isolated dip."""
    n, p = 20000, 10000
    g_req = np.ones(n)
    g_req[p] = 0.5
    g = env.box_smooth(g_req, attack=attack)
    red = np.flatnonzero(g < 1.0)
    assert p - red.min() == 2 * (attack // 2) <= attack
    assert red.max() - p == 2 * (attack // 2)
    assert g[p] == pytest.approx(0.5)


@pytest.mark.parametrize("variant", ["box", "qp"])
@pytest.mark.parametrize("attack", [100, 301, 1000])
def test_no_reduction_earlier_than_attack(variant, attack):
    """Isolated impulse, both variants (spec 7): no gain reduction more than
    `attack` samples before it; the release hull keeps it reduced after."""
    n, p = 30000, 12000
    g_req = np.ones(n)
    g_req[p] = 0.4
    if variant == "box":
        g = env.box_smooth(g_req, attack=attack, release=2000)
    else:
        g, info = env.qp_smooth(g_req, attack=attack, release=2000)
        assert info["converged"]
    red = np.flatnonzero(g < 1.0)
    assert p - red.min() <= attack, (variant, attack, p - red.min())
    # and the lookahead is actually used (QP with a release hull: 0.88-0.89 x attack)
    assert p - red.min() >= 0.8 * attack
    assert red.max() > p + 5000
    assert g[p] <= 0.4


def test_qp_extent_ratio_measured():
    """Support half-width / length for an isolated dip: 4.40 to 4.44, depth independent."""
    n, p = 40000, 20000
    for length in (25, 100, 200):
        for depth in (0.5, 0.05):
            g_req = np.ones(n)
            g_req[p] = depth
            g, info = env.qp_smooth(g_req, length)
            red = np.flatnonzero(g < 1.0)
            ratio = (p - red.min()) / length
            assert 4.3 <= ratio <= env.QP_EXTENT_RATIO, (length, depth, ratio)


def test_release_hull_is_exponential_in_db():
    n, p, tau = 5000, 100, 300.0
    u = np.ones(n)
    u[p] = 10 ** (-6 / 20)
    h = env.release_hull(u, tau)
    k = np.arange(1, 2000)
    depth = -20 * np.log10(h[p + k])
    assert np.allclose(depth, 6.0 * np.exp(-k / tau), rtol=1e-9, atol=1e-12)
    assert np.all(h[:p] == 1.0)
    assert np.all(h <= u)


def test_release_hull_precision_on_long_input():
    """Block-reset accumulation: 5e6 samples, tau 1e6, still exact to 1e-9 relative."""
    n, tau = 5_000_000, 1e6
    u = np.ones(n)
    u[0] = 10 ** (-6 / 20)
    h = env.release_hull(u, tau)
    k = np.array([1, 10, 1000, 10**5, 10**6, 4 * 10**6, n - 1])
    depth = -20 * np.log10(h[k])
    assert np.allclose(depth, 6.0 * np.exp(-k / tau), rtol=1e-9)


def test_release_hull_tiny_release_is_no_hull():
    u = _dense_greq(n=1000)
    assert np.array_equal(env.release_hull(u, 1e-13), u)


def test_qp_mu_parameter_equivalent_to_length():
    g_req = _dense_greq(n=20000)
    a, _ = env.qp_smooth(g_req, 16)
    b, info = env.qp_smooth(g_req, mu=16.0 ** -4)
    assert np.array_equal(a, b) and info["mu"] == 16.0 ** -4


@pytest.mark.parametrize("length", [8, 128, 512])
def test_qp_touches_bound(length):
    """At the optimum g must touch u somewhere (else a uniform raise of g lowers
    the fidelity term). Guards against a too-loose stopping rule."""
    g_req = _dense_greq(n=48000)
    g, info = env.qp_smooth(g_req, length)
    assert info["converged"]
    assert np.max(g - np.minimum(g_req, 1.0)) > -1e-6


def test_qp_matches_independent_solver():
    """Objective agrees with scipy L-BFGS-B on the same bound-constrained QP."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(0)
    n, length = 400, 6.0
    u = np.ones(n)
    for p in rng.choice(np.arange(20, n - 20), 8, replace=False):
        u[p] = rng.uniform(0.3, 0.9)
    mu = length**-4

    def obj(g):
        d2 = np.diff(g, 2)
        return 0.5 * d2 @ d2 + 0.5 * mu * (g - u) @ (g - u)

    def grad(g):
        return env._q_mul(g, mu) - mu * u

    ref = minimize(obj, u.copy(), jac=grad, method="L-BFGS-B",
                   bounds=[(None, ui) for ui in u],
                   options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-12})
    g, info = env.qp_smooth(u, length)
    assert info["converged"]
    assert obj(g) <= ref.fun * (1 + 1e-6) + 1e-12
    assert np.max(np.abs(g - ref.x)) < 1e-3


@pytest.mark.slow
def test_qp_runtime_240k_points():
    g_req = _dense_greq(n=240000)
    g, info = env.qp_smooth(g_req, 32)
    assert info["converged"]
    assert info["seconds"] < 10.0
    print(f"qp_smooth 240k points: {info['iterations']} iterations, "
          f"polished={info['polished']}, {info['seconds']:.2f} s")


# ------------------------------------------------ limiter gate at 8x

def _fractional_impulses(n, fs, period_s, amp, rng):
    """Band-limited impulses (Hann-windowed sinc, 129 taps) at random fractional positions."""
    x = np.zeros(n)
    t = np.arange(n)
    for t0 in np.arange(0.013, n / fs - 0.01, period_s):
        pos = t0 * fs + rng.uniform(0.1, 0.9)
        w = np.abs(t - pos) < 64
        seg = t[w] - pos
        x[w] += amp * np.sinc(seg) * (0.5 + 0.5 * np.cos(np.pi * seg / 64))
    return x


def _limiter_input(seconds=1.0, seed=0):
    n = int(seconds * FS)
    t = np.arange(n) / FS
    fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.01)
    sweep = 1.2 * signal.chirp(t, 20, t[-1], 20000, method="logarithmic") * fade
    x = sweep + _fractional_impulses(n, FS, 0.07, 1.0, np.random.default_rng(seed))
    return np.stack([x, 0.8 * x[::-1]], axis=1)


@pytest.mark.parametrize("name", ["box", "qp"])
@pytest.mark.slow
def test_limiter_at_8x_true_peak(name):
    """Spec 7 limiter gate: 1.2-amplitude log sweep plus fractional-phase impulses;
    detector = 8x true-peak envelope, gain applied at 8x, decimated, measured at 8x."""
    target_db = -1.0
    L = 8
    x = _limiter_input()
    assert meter.true_peak(x, FS, 8) > target_db + 6.0  # well above the ceiling
    x8 = meter.upsample(x, L)
    g_req = env.required_gain(np.max(np.abs(x8), axis=1), 10 ** (target_db / 20))
    attack = int(0.0015 * FS * L)
    release = 0.05 * FS * L
    if name == "box":
        g = env.box_smooth(g_req, attack=attack, release=release)
    else:
        g, info = env.qp_smooth(g_req, attack=attack, release=release)
        assert info["converged"]
    assert np.all(g <= np.minimum(g_req, 1.0))
    y = meter.decimate(x8 * g[:, None], L)
    tp8 = meter.true_peak(y, FS, 8)
    print(f"limiter {name}: TP8 after decimation {tp8:.3f} dBTP")
    assert tp8 <= target_db + 0.05, tp8


def test_limiting_at_4x_overshoots_at_8x():
    """Documents audit 02 A11: the same input limited at 4x (resample_poly) reads
    above the ceiling at 8x after decimation."""
    target_db = -1.0
    x = _limiter_input()
    x4 = signal.resample_poly(x, 4, 1, axis=0)
    g_req = env.required_gain(np.max(np.abs(x4), axis=1), 10 ** (target_db / 20))
    g = env.box_smooth(g_req, attack=int(0.0015 * FS * 4), release=0.05 * FS * 4)
    y = signal.resample_poly(x4 * g[:, None], 1, 4, axis=0)
    tp8 = meter.true_peak(y, FS, 8)
    print(f"limited at 4x: TP8 {tp8:.3f} dBTP")
    assert tp8 > target_db + 0.05
