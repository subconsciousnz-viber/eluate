"""Static parametric input EQ (docs/gain-staging.md chunk 2): each section's measured
zero-phase response against its specification, silence, DC, identity, the plan."""

import numpy as np
import pytest
from scipy import signal

from eluate.plan import (EqBell, EqHighPass, EqHighShelf, EqLowShelf, InputEqPlan, Plan)
from eluate.stages import inputeq

FS = 48000
N = 1 << 19                 # 10.9 s impulse buffer: 0.09 Hz bins, room for the 25 Hz tails


def _measured_db(plan, fs=FS):
    """Magnitude (dB) of apply() on a centred unit impulse, on the rfft grid."""
    x = np.zeros((N, 2))
    x[N // 2] = 1.0
    y = inputeq.apply(x, fs, plan)
    assert np.array_equal(y[:, 0], y[:, 1])        # same filter on both channels
    f = np.fft.rfftfreq(N, 1.0 / fs)
    with np.errstate(divide="ignore"):
        return f, 20.0 * np.log10(np.abs(np.fft.rfft(y[:, 0])))


def _spec_db(plan, f, fs=FS):
    _, h = signal.sosfreqz(inputeq.spec_sos(plan, fs), worN=f, fs=fs)
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.abs(h))


def _at(f, db, hz):
    return float(np.interp(np.log2(hz), np.log2(f[1:]), db[1:]))


def _check_against_spec(plan, tol=0.01):
    f, m = _measured_db(plan)
    s = _spec_db(plan, f)
    sel = (f > 5.0) & (s > -60.0)
    err = float(np.max(np.abs(m[sel] - s[sel])))
    print(f"{plan}: max deviation from the specification {err:.4f} dB")
    assert err < tol
    return f, m


@pytest.mark.parametrize("q", [0.5, 0.707, 1.0])
def test_highpass_24_db_per_octave(q):
    p = InputEqPlan(highpass=EqHighPass(25.0, q))
    f, m = _check_against_spec(p)
    assert _at(f, m, 25.0) == pytest.approx(20 * np.log10(q * q), abs=0.02)
    slope = (_at(f, m, 25.0 / 8) - _at(f, m, 25.0 / 16))     # one octave, far below fc
    assert slope == pytest.approx(24.0, abs=0.3)
    hi = (f >= 20 * 25.0) & (f < 0.45 * FS)
    assert np.max(np.abs(m[hi])) < 0.05


def test_highpass_lower_q_cuts_more_sub():
    f, m5 = _measured_db(InputEqPlan(highpass=EqHighPass(25.0, 0.5)))
    _, m7 = _measured_db(InputEqPlan(highpass=EqHighPass(25.0, 0.707)))
    for hz in (31.5, 50.0, 80.0):
        assert _at(f, m5, hz) < _at(f, m7, hz) - 0.5


@pytest.mark.parametrize("g,fc,q", [(3.0, 120.0, 0.5), (-3.0, 250.0, 0.5), (1.2, 60.0, 0.707),
                                    (-2.0, 100.0, 0.6)])
def test_low_shelf_12_db_per_octave(g, fc, q):
    p = InputEqPlan(low_shelf=EqLowShelf(g, fc, q))
    f, m = _check_against_spec(p)
    assert _at(f, m, fc) == pytest.approx(g / 2, abs=0.02)        # corner: half the gain
    assert _at(f, m, fc / 30) == pytest.approx(g, abs=0.05)       # shelf
    hi = (f >= 30 * fc) & (f < 0.45 * FS)
    assert np.max(np.abs(m[hi])) < 0.05                           # unity elsewhere
    # second order: at Q 0.707 the transition is steeper than a first-order shelf's
    # (at Q 0.5 the RBJ shelf is the square of two half-gain first-order shelves, so
    # within these gains it is nearly the first-order shape)
    first = _max_slope(*_measured_db(InputEqPlan(high_shelf=EqHighShelf(-g, fc))))
    if q == 0.707:
        assert _max_slope(f, m) > 1.3 * first


@pytest.mark.parametrize("g,fc,q", [(-3.0, 250.0, 0.7), (-1.5, 120.0, 4.0), (-2.0, 1000.0, 2.0)])
def test_bell_cut(g, fc, q):
    p = InputEqPlan(bell=EqBell(g, fc, q))
    f, m = _check_against_spec(p)
    assert _at(f, m, fc) == pytest.approx(g, abs=0.02)
    assert float(np.max(m[f > 5.0])) < 0.01                       # cuts only
    away = ((f > 5.0) & (f <= fc / 30)) | ((f >= 30 * fc) & (f < 0.45 * FS))
    assert np.max(np.abs(m[away])) < 0.05


def test_bell_is_cut_only_in_the_plan():
    with pytest.raises(ValueError):
        EqBell(1.0, 300.0, 1.0)
    with pytest.raises(ValueError):
        EqBell(-1.0, 1500.0, 1.0)
    with pytest.raises(ValueError):
        EqBell(-1.0, 300.0, 5.0)


@pytest.mark.parametrize("g,fc", [(2.0, 16000.0), (-3.0, 3000.0), (3.0, 1000.0)])
def test_high_shelf_6_db_per_octave(g, fc):
    p = InputEqPlan(high_shelf=EqHighShelf(g, fc))
    f, m = _check_against_spec(p)
    assert _at(f, m, fc) == pytest.approx(g / 2, abs=0.02)
    lo = (f > 5.0) & (f <= fc / 30)
    assert np.max(np.abs(m[lo])) < 0.05
    # first order: the steepest point of the transition is G ln(2) / 2 dB/oct (small-gain
    # form of the first-order shelf; bilinear warping steepens it near Nyquist)
    if fc <= 3000.0:
        assert _max_slope(f, m) == pytest.approx(abs(g) * np.log(2) / 2, rel=0.05)


def _max_slope(f, m):
    """Steepest slope of a response, dB/oct, on a 1/24-octave grid from 10 Hz to 20 kHz."""
    g = 10.0 * 2.0 ** (np.arange(0, 11 * 24) / 24.0)
    g = g[g < 20000.0]
    d = np.array([_at(f, m, h) for h in g])
    return float(np.max(np.abs(np.diff(d))) * 24.0)


def test_all_sections_cascade_and_minimum_phase_option():
    p = InputEqPlan(EqHighPass(25.0, 0.6), EqLowShelf(1.5, 150.0), EqBell(-2.0, 400.0, 2.0),
                    EqHighShelf(2.5, 12000.0))
    f, m = _check_against_spec(p)
    pm = InputEqPlan(p.highpass, p.low_shelf, p.bell, p.high_shelf, phase="minimum")
    _, mm = _measured_db(pm)
    sel = (f > 5.0) & (_spec_db(p, f) > -60.0)
    assert np.max(np.abs(mm[sel] - _spec_db(p, f)[sel])) < 1e-6   # exact, one causal pass
    # zero phase: the impulse response is symmetric about the impulse, except the
    # high-pass, which runs causally (R7e, T6): nothing of it rings ahead of a kick
    x = np.zeros((1 << 18, 2))
    c = x.shape[0] // 2
    x[c] = 1.0
    k = np.arange(1, 20000)
    y = inputeq.apply(x, FS, InputEqPlan(None, p.low_shelf, p.bell, p.high_shelf))[:, 0]
    assert np.max(np.abs(y[c + k] - y[c - k])) < 1e-12
    yh = inputeq.apply(x, FS, InputEqPlan(p.highpass))[:, 0]
    assert np.max(np.abs(yh[:c])) == 0.0 and abs(yh[c]) > 0.5
    assert response_close(p)


def response_close(p):
    """response_db (the planner's model) agrees with the measured zero-phase response."""
    f, m = _measured_db(p)
    sel = (f > 20.0) & (f < 20000.0)
    r = inputeq.response_db(p, f[sel], FS)
    return float(np.max(np.abs(r - m[sel]))) < 1e-6


def test_silence_identity_and_dc():
    x = np.zeros((FS, 2))
    p = InputEqPlan(EqHighPass(), EqLowShelf(2.0, 100.0), None, EqHighShelf(1.0))
    assert np.array_equal(inputeq.apply(x, FS, p), x)
    rng = np.random.default_rng(0)
    z = rng.standard_normal((FS, 2))
    off = inputeq.apply(z, FS, InputEqPlan())
    assert np.array_equal(off, z) and off is not z
    # DC, 10 s: the high-pass blocks it; without the high-pass it leaves at the shelf gain
    dc = np.full((10 * FS, 2), 0.1)
    mid = slice(4 * FS, 6 * FS)
    assert np.max(np.abs(inputeq.apply(dc, FS, p)[mid])) < 1e-6
    q = InputEqPlan(None, EqLowShelf(2.0, 100.0), None, EqHighShelf(1.0))
    assert np.allclose(inputeq.apply(dc, FS, q)[mid], 0.1 * 10 ** (2.0 / 20), rtol=1e-6)


@pytest.mark.parametrize("phase", ["zero", "minimum"])
def test_ms_top_end_shelves_mid_and_side_apart(phase):
    """Maintainer, 2026-10-20: separate high shelves on M (up to +4.5) and S (up to +6); the
    shared sections still act on both."""
    from eluate import bands
    p = InputEqPlan(low_shelf=EqLowShelf(-2.0, 120.0), high_shelf=EqHighShelf(4.5, 3000.0),
                    high_shelf_side=EqHighShelf(6.0, 2500.0), phase=phase)
    f = np.fft.rfftfreq(N, 1.0 / FS)
    for c, ch in enumerate("MS"):
        ms = np.zeros((N, 2))
        ms[N // 2, c] = 1.0
        y = bands.to_ms(inputeq.apply(bands.to_lr(ms), FS, p))
        assert np.max(np.abs(y[:, 1 - c])) < 1e-9              # no leak into the other channel
        with np.errstate(divide="ignore"):
            m = 20.0 * np.log10(np.abs(np.fft.rfft(y[:, c])))
        want = inputeq.response_db(p, f[1:], FS, ch)
        sel = (f[1:] > 20.0) & (f[1:] < 0.45 * FS)
        assert np.max(np.abs(m[1:][sel] - want[sel])) < 0.02
        assert _at(f, m, 20000.0) == pytest.approx(4.5 if ch == "M" else 6.0, abs=0.3)
        assert _at(f, m, 30.0) == pytest.approx(-2.0, abs=0.2)   # the shared low shelf
    # linked (no side shelf): the side hears the mid's shelf, as before
    lk = InputEqPlan(high_shelf=EqHighShelf(3.0, 3000.0))
    assert np.allclose(inputeq.response_db(lk, f[1:100], FS, "S"),
                       inputeq.response_db(lk, f[1:100], FS, "M"))
    with pytest.raises(ValueError):
        EqHighShelf(6.1)
    q = Plan.from_json(Plan(fs=FS, inputeq=p).to_json())
    assert q.inputeq == p


@pytest.mark.parametrize("phase", ["zero", "minimum"])
def test_presence_and_air_shelves_per_channel(phase):
    """Sprint 2026-10-21 T12: a presence and an air shelf on each of M and S; a side air
    shelf alone makes the top end M/S; the plan survives JSON."""
    from eluate import bands
    p = InputEqPlan(high_shelf=EqHighShelf(2.0, 3000.0), air_shelf=EqHighShelf(3.0, 12000.0),
                    high_shelf_side=EqHighShelf(1.0, 3000.0), air_shelf_side=EqHighShelf(4.0, 12000.0),
                    phase=phase)
    assert p.ms_top
    f = np.fft.rfftfreq(N, 1.0 / FS)
    for c, ch in enumerate("MS"):
        ms = np.zeros((N, 2))
        ms[N // 2, c] = 1.0
        y = bands.to_ms(inputeq.apply(bands.to_lr(ms), FS, p))
        assert np.max(np.abs(y[:, 1 - c])) < 1e-9
        with np.errstate(divide="ignore"):
            m = 20.0 * np.log10(np.abs(np.fft.rfft(y[:, c])))
        want = inputeq.response_db(p, f[1:], FS, ch)
        sel = (f[1:] > 20.0) & (f[1:] < 0.45 * FS)
        assert np.max(np.abs(m[1:][sel] - want[sel])) < 0.02
        assert _at(f, m, 20000.0) == pytest.approx(5.0, abs=0.6)       # both shelves' sum
    # an air shelf on S only still splits the top end
    assert InputEqPlan(air_shelf_side=EqHighShelf(1.0, 12000.0)).ms_top
    linked = InputEqPlan(high_shelf=EqHighShelf(2.0, 3000.0), air_shelf=EqHighShelf(3.0, 12000.0))
    assert not linked.ms_top
    assert np.allclose(inputeq.response_db(linked, f[1:100], FS, "S"),
                       inputeq.response_db(linked, f[1:100], FS, "M"))
    assert Plan.from_json(Plan(fs=FS, inputeq=p).to_json()).inputeq == p


def test_stereo_only_and_frequency_below_nyquist():
    p = InputEqPlan(high_shelf=EqHighShelf(1.0, 16000.0))
    with pytest.raises(ValueError):
        inputeq.apply(np.zeros(100), FS, p)
    with pytest.raises(ValueError):
        inputeq.apply(np.zeros((100, 2)), 32000, p)
    with pytest.raises(ValueError):
        Plan(fs=32000, inputeq=p)


def test_plan_bounds_and_json_round_trip(tmp_path):
    for bad in (lambda: EqHighPass(25.0, 0.4), lambda: EqHighPass(25.0, 1.1),
                lambda: EqLowShelf(3.5, 100.0), lambda: EqLowShelf(1.0, 300.0),
                lambda: EqLowShelf(1.0, 100.0, 0.9), lambda: EqHighShelf(-3.2),
                lambda: InputEqPlan(phase="linear")):
        with pytest.raises(ValueError):
            bad()
    p = Plan(fs=FS, inputeq=InputEqPlan(EqHighPass(25.0, 0.6), EqLowShelf(1.2, 120.0),
                                         EqBell(-1.5, 250.0, 2.0), EqHighShelf(1.0),
                                         design={"nodes_hz": [100.0, 200.0], "rotation": 0.5}))
    q = Plan.from_json(p.to_json())
    assert q.to_dict() == p.to_dict() and q.inputeq == p.inputeq
    p.save(tmp_path / "plan.json")
    assert Plan.load(tmp_path / "plan.json").inputeq == p.inputeq
    assert Plan.from_json(Plan(fs=FS).to_json()).inputeq is None
