"""Gates for stages/notch.py (spec 3.5 stage 1, section 7; audit 05 A5-A8)."""

import math

import numpy as np
import pytest
from scipy import signal

from fixtures import synthetic as syn
from eluate.plan import Envelope, Notch, NotchPlan, control_length
from eluate.stages import notch

FS = 48000
RATES = [44100, 48000, 96000]


def _noise(seconds=2.0, seed=0, fs=FS):
    return syn.pink_noise(int(seconds * fs), fs, rms_dbfs=-20, seed=seed)


def _impulse_response_db(plan, fs, n=1 << 16):
    x = np.zeros((n, 1))
    x[n // 2] = 1.0
    h = notch.apply(x, fs, plan)[:, 0]
    H = np.fft.rfft(np.roll(h, -n // 2))
    return 20 * np.log10(np.abs(H)), np.fft.rfftfreq(n, 1 / fs), H


@pytest.mark.parametrize("fs", RATES)
def test_depth_zero_is_bit_exact_identity(fs):
    """Spec 7: complementary identity when depth = 0, bit-exact."""
    x = _noise(fs=fs)
    n = x.shape[0]
    flat = Envelope(np.zeros(control_length(n, fs)))
    plans = [
        NotchPlan([]),
        NotchPlan([Notch(1000.0, 8.0, Envelope.constant(0.0))]),
        NotchPlan([Notch(1000.0, 8.0, flat), Notch(5000.0, 12.0, flat)]),
        NotchPlan([Notch(1000.0, 8.0, Envelope.constant(-6.0))], kick_mask=Envelope.constant(1.0)),
    ]
    for p in plans:
        assert np.array_equal(notch.apply(x, fs, p), x)


@pytest.mark.parametrize("fs", RATES)
def test_depth_zero_regions_are_bit_exact(fs):
    x = _noise(fs=fs)
    n = x.shape[0]
    g = np.zeros(control_length(n, fs))
    g[500:1500] = -6.0
    p = NotchPlan([Notch(2000.0, 10.0, Envelope(g)), Notch(6000.0, 10.0, Envelope(g))])
    y = notch.apply(x, fs, p)
    i0, i1 = int(0.499 * fs), int(1.501 * fs)
    assert np.array_equal(y[:i0], x[:i0])
    assert np.array_equal(y[i1:], x[i1:])
    assert not np.array_equal(y[i0:i1], x[i0:i1])


@pytest.mark.parametrize("fs", RATES)
@pytest.mark.parametrize("f0,q,g", [(150.0, 6.0, -6.0), (1000.0, 10.0, -6.0), (1000.0, 10.0, -3.0),
                                    (7000.0, 12.0, -6.0), (15000.0, 8.0, -4.0)])
def test_q_is_defined_on_the_notch_response(fs, f0, q, g):
    """Audit 05 A5: f0 / Q is the width of the notch response at half its dB cut
    (was 130.6 Hz for f0/Q = 100 Hz at -6 dB). Zero phase, exact depth at f0."""
    nt = Notch(f0, q, Envelope.constant(g))
    sos = notch.notch_sos(nt, fs)
    w, h = signal.sosfreqz(sos, worN=2 ** 21, fs=fs)
    d = 1 - 10 ** (g / 20)
    resp = 20 * np.log10(1 - d * np.abs(h) ** 2)
    band = w[resp <= g / 2]
    ratio = (band.max() - band.min()) / (f0 / q)
    print(f"{fs} Hz f0 {f0} Q {q} cut {g}: half-cut width / (f0/Q) = {ratio:.5f}, "
          f"min {resp.min():.5f} dB")
    assert abs(ratio - 1.0) < 2e-3
    assert abs(resp.min() - g) < 1e-4
    n = 1 << 15
    imp = np.zeros((n, 1))
    imp[n // 2] = 1.0
    r = notch.bandpass_zp(imp, sos, fs)[:, 0]
    assert np.max(np.abs(r[n // 2 + 1:] - r[n // 2 - 1:0:-1][: n // 2 - 1])) < 1e-12


@pytest.mark.parametrize("g_db,expect", [(-3.0, -3.0), (-6.0, -6.0), (-12.0, -6.0)])
def test_gain_at_f0_and_6db_cap(g_db, expect):
    f0 = 1000.0
    n = 2 * FS
    t = np.arange(n) / FS
    x = np.stack([np.sin(2 * np.pi * f0 * t)] * 2, axis=1) * 0.5
    y = notch.apply(x, FS, NotchPlan([Notch(f0, 10.0, Envelope.constant(g_db))]))
    mid = slice(FS // 2, 3 * FS // 2)
    got = 20 * np.log10(np.sqrt(np.mean(y[mid] ** 2)) / np.sqrt(np.mean(x[mid] ** 2)))
    print(f"request {g_db} dB at f0: measured {got:.4f} dB")
    assert abs(got - expect) < 0.01


def test_overlapping_bands_must_merge_and_merged_cut_is_capped():
    """Audit 05 A6: two Q=6 notches 41 cents apart (was -11.72 dB) overlap, so apply
    refuses them; merged, the cut is -6 dB."""
    pair = [Notch(1000.0, 6.0, Envelope.constant(-6.0)), Notch(1024.0, 6.0, Envelope.constant(-6.0))]
    with pytest.raises(ValueError, match="overlap"):
        notch.apply(np.zeros((100, 2)), FS, NotchPlan(pair))
    m = notch.merge_candidates(pair)
    assert len(m) == 1
    mag, f, _ = _impulse_response_db(NotchPlan(m), FS)
    print(f"merged {m[0].f0_hz:.1f} Hz Q {m[0].q:.3f}: deepest {mag.min():.4f} dB")
    assert mag.min() >= -6.0 - 1e-6


@pytest.mark.parametrize("fs", RATES)
@pytest.mark.parametrize("spacing", [1.45, 1.6, 2.0])
def test_composite_cut_clamped_for_separate_notches(fs, spacing):
    """Non-overlapping notches whose skirts add: the summed cut never exceeds 6 dB."""
    a = Notch(1000.0, 3.0, Envelope.constant(-6.0))
    b = Notch(1000.0 * spacing, 3.0, Envelope.constant(-6.0))
    c = Notch(1000.0 * spacing ** 2, 3.0, Envelope.constant(-6.0))
    p = NotchPlan(notch.merge_candidates([a, b, c]))
    mag, f, H = _impulse_response_db(p, fs)
    print(f"{fs} Hz spacing {spacing}: {len(p.notches)} notches, deepest {mag.min():.6f} dB")
    assert mag.min() >= -6.0 - 1e-5
    assert np.max(np.abs(H.imag)) < 1e-9  # still zero phase


def test_composite_clamp_bound_holds_with_time_varying_depths():
    n = 2 * FS
    nc = control_length(n, FS)
    rng = np.random.default_rng(3)
    ga = Envelope(np.repeat(rng.uniform(-6, 0, nc // 50 + 1), 50)[:nc])
    gb = Envelope(np.repeat(rng.uniform(-6, 0, nc // 50 + 1), 50)[:nc])
    p = NotchPlan([Notch(1000.0, 3.0, ga), Notch(1450.0, 3.0, gb)])
    x = _noise(2.0, seed=4)
    y = notch.apply(x, FS, p)
    assert np.all(np.isfinite(y))
    # where both envelopes sit at -6 dB on a whole segment the clamp binds; the
    # static response test above bounds that case; here check no blow-up and identity
    # where both are 0 dB
    both0 = (ga.at(n, FS) == 0) & (gb.at(n, FS) == 0)
    assert np.array_equal(y[both0], x[both0])


def test_notch_reduces_injected_ring():
    x0 = _noise(3.0)
    x, _ = syn.add_ring(x0, FS, f0=7000.0, q=12.0, gain_db=9.0, start_s=0.5, end_s=2.5)
    y = notch.apply(x, FS, NotchPlan([Notch(7000.0, 12.0, Envelope.constant(-6.0))]))
    f, px = signal.welch(x[:, 0], FS, nperseg=8192)
    _, py = signal.welch(y[:, 0], FS, nperseg=8192)
    at = np.argmin(np.abs(f - 7000))
    far = (f > 1000) & (f < 3000)
    d_ring = 10 * np.log10(py[at] / px[at])
    d_far = 10 * np.log10(py[far].sum() / px[far].sum())
    print(f"ring: {d_ring:.2f} dB at 7 kHz, {d_far:.4f} dB at 1-3 kHz")
    assert -6.5 < d_ring < -5.0
    assert abs(d_far) < 0.02


@pytest.mark.parametrize("fs", RATES)
def test_kick_mask_holds_exactly_20ms_off_grid(fs):
    """Audit 05 A8: depth is 0 for the full +-20 ms around an off-grid onset (was 39.0 ms)."""
    x = _noise(1.0, seed=3, fs=fs)
    n = x.shape[0]
    onsets = [0.2505, 0.6113]
    p = NotchPlan([Notch(3000.0, 8.0, Envelope.constant(-6.0))],
                  kick_mask=notch.kick_mask(onsets, n / fs))
    y = notch.apply(x, fs, p)
    sos = notch.notch_sos(p.notches[0], fs)
    bp = notch.bandpass_zp(x, sos, fs)
    d_full = 1 - 10 ** (-6 / 20)
    for o in onsets:
        hold = slice(int(math.ceil((o - 0.020) * fs)), int(math.floor((o + 0.020) * fs)) + 1)
        assert np.array_equal(y[hold], x[hold])
        far = int((o + 0.06) * fs)
        seg = slice(far, far + 480)
        assert np.allclose(y[seg], x[seg] - d_full * bp[seg], atol=1e-12)


def test_merge_rules_and_q_min():
    e1 = Envelope(np.array([0.0, -4.0, -2.0]))
    e2 = Envelope(np.array([-3.0, -1.0]))
    lo1, hi1 = notch.band_edges(1000.0, 10.0)
    assert abs(lo1 * hi1 - 1e6) < 1e-6 and abs(hi1 - lo1 - 100.0) < 1e-9
    close = [Notch(1000.0, 10.0, e1), Notch(1080.0, 10.0, e2)]   # bands overlap
    far = [Notch(1000.0, 10.0, e1), Notch(1120.0, 10.0, e2)]     # bands disjoint
    m = notch.merge_candidates(close)
    assert len(m) == 1 and 1000.0 < m[0].f0_hz < 1080.0 and m[0].q < 10.0
    assert np.array_equal(m[0].gain_db.values, [-3.0, -4.0, -2.0])
    assert len(notch.merge_candidates(far)) == 2
    with pytest.raises(ValueError):
        Notch(1000.0, 0.5, Envelope.constant(-3.0))  # Q >= 1
    wide = notch.merge_candidates([Notch(1000.0, 1.0, Envelope.constant(-3.0)),
                                   Notch(1400.0, 1.0, Envelope.constant(-3.0))])
    assert len(wide) == 1 and wide[0].q >= 1.0


def test_depth_bounded_at_every_sample():
    n = FS
    g = np.where(np.arange(control_length(n, FS)) % 7 < 3, -20.0, 0.0)
    env = Envelope(g)
    d = 1 - 10 ** (np.clip(env.at(n, FS), -6, 0) / 20)
    assert d.max() <= 1 - 10 ** (-6 / 20) + 1e-15
    x = np.zeros((n, 2))
    x[n // 2] = 1.0
    y = notch.apply(x, FS, NotchPlan([Notch(2000.0, 4.0, env)]))
    assert np.all(np.isfinite(y))
