"""Gates for stages/multiband.py (spec 3.5 stage 3; audit 05 A9, A10, A15; brief 06)."""

import numpy as np
import pytest
from scipy import signal

from fixtures import synthetic as syn
from fixtures.crest_fixtures import band_impulse_train, kick_plus_sustain
from eluate import bands
from eluate.plan import MULTIBAND_MAX_GR_DB, Envelope, MultibandPlan, Plan, control_length
from eluate.stages import multiband

FS = 48000
NB = 5
RATES = [44100, 48000, 96000]
OLD_XO = [100.0, 500.0, 2000.0, 8000.0]   # the crossovers these band-response tests were written on


@pytest.fixture
def cap45(monkeypatch):
    """The per-sample compressor clamp at its pre-meter value (4.5 dB), for the tests whose
    signals drive the compressor into the clamp so the ceiling engages on top of it (the
    clamp is 12 dB since the meter reading, 2026-10-19; the mechanism is unchanged)."""
    import sys
    monkeypatch.setattr(multiband, "MULTIBAND_MAX_GR_DB", 4.5)
    monkeypatch.setattr(sys.modules[__name__], "MULTIBAND_MAX_GR_DB", 4.5)
    return 4.5


def _flat(n_bands=NB, value=0.0):
    return [[Envelope.constant(value), Envelope.constant(value)] for _ in range(n_bands)]


def _ms_tone(f, n, channel, fs=FS, amp=0.3):
    t = np.arange(n) / fs
    ms = np.zeros((n, 2))
    ms[:, channel] = amp * np.sin(2 * np.pi * f * t)
    return bands.to_lr(ms)


def _band_weights(f, fs, fcs=(100, 500, 2000, 8000), order=2):
    """Zero-phase cascade band responses at f, from the digital Butterworth magnitude
    (independent of bands.split): lp_k = |B_k|^2, band k = lp_k * prod_{j<k} (1 - lp_j)."""
    lp = []
    for fc in fcs:
        sos = signal.butter(order, fc, fs=fs, output="sos")
        _, h = signal.sosfreqz(sos, worN=[f], fs=fs)
        lp.append(float(np.abs(h[0]) ** 2))
    w, rest = [], 1.0
    for v in lp:
        w.append(rest * v)
        rest *= 1.0 - v
    w.append(rest)
    return np.array(w)


def _expect_db(f, gains_db, fs=FS, fcs=(100, 500, 2000, 8000)):
    return 20 * np.log10(np.sum(_band_weights(f, fs, fcs) * 10 ** (np.asarray(gains_db) / 20)))


def _gain_db(y, x, channel, fs=FS):
    a, b = bands.to_ms(y)[:, channel], bands.to_ms(x)[:, channel]
    mid = slice(fs // 2, -fs // 2)
    return 20 * np.log10(np.sqrt(np.mean(a[mid] ** 2)) / np.sqrt(np.mean(b[mid] ** 2)))


@pytest.mark.parametrize("fs", RATES)
def test_unity_plan_is_identity_to_double_precision(fs):
    x = syn.pink_noise(2 * fs, fs, rms_dbfs=-14, seed=2)
    y = multiband.apply(x, fs, MultibandPlan(_flat(), side_makeup_db=0.0))
    assert np.max(np.abs(y - x)) < 1e-14


@pytest.mark.parametrize("phase", ["minimum", ["minimum", "zero", "zero", "zero"]])
def test_minimum_and_mixed_phase(phase):
    """Audit 05 A15: with a minimum-phase crossover the unity plan is an allpass
    (flat magnitude), and the compressor clamp still holds on a band."""
    n = 1 << 16
    x = np.zeros((n, 2))
    x[n // 2] = 1.0
    y = multiband.apply(x, FS, MultibandPlan(_flat(), phase=phase, side_makeup_db=0.0))
    for c in range(2):
        mag = 20 * np.log10(np.abs(np.fft.rfft(y[:, c])))
        f = np.fft.rfftfreq(n, 1 / FS)
        band = (f > 10) & (f < 0.45 * FS)
        assert np.max(np.abs(mag[band])) < 0.01
    xt = _ms_tone(1000.0, 2 * FS, 0)
    gr = _flat()
    gr[2][0] = Envelope.constant(-10.0)
    got = _gain_db(multiband.apply(xt, FS, MultibandPlan(gr, phase=phase, side_makeup_db=0.0)),
                   xt, 0)
    print(f"phase {phase}: 1 kHz M gain with -10 dB request {got:.4f} dB")
    assert got >= -MULTIBAND_MAX_GR_DB - 0.02
    p = MultibandPlan(_flat(), phase=phase)
    from eluate.plan import Plan
    q = Plan.from_json(Plan(fs=FS, multiband=p).to_json())
    assert q.multiband.phase == phase


@pytest.mark.parametrize("request_db", [-1.5, -3.0, -10.0, -40.0])
def test_gain_reduction_capped_at_the_clamp(request_db):
    n = 2 * FS
    x = _ms_tone(1000.0, n, 0)
    p = MultibandPlan(_flat(value=request_db), side_makeup_db=0.0)
    got = _gain_db(multiband.apply(x, FS, p), x, 0)
    print(f"GR request {request_db} dB -> {got:.4f} dB")
    assert abs(got - max(request_db, -MULTIBAND_MAX_GR_DB)) < 1e-3


@pytest.mark.parametrize("fs", RATES)
def test_cap_holds_between_control_points(fs):
    """A plan envelope alternating 0 / -20 dB every 1 ms: the applied per-sample gain
    never goes below the clamp (PCHIP does not overshoot the clipped control values)."""
    n = fs
    v = np.where(np.arange(control_length(n, fs)) % 2 == 0, 0.0, -20.0)
    gr = _flat()
    gr[2][0] = Envelope(v)
    x = _ms_tone(1000.0, n, 0, fs, amp=0.5)
    b2 = bands.split(bands.to_ms(x), fs, [100, 500, 2000, 8000])[2, :, 0]
    y = multiband.apply(x, fs, MultibandPlan(gr, crossovers_hz=OLD_XO, side_makeup_db=0.0))
    big = np.abs(b2) > 0.1
    g = 1 + (bands.to_ms(y)[:, 0] - bands.to_ms(x)[:, 0])[big] / b2[big]
    print(f"{fs} Hz alternating plan: applied gain {20 * np.log10(g.min()):.6f} dB minimum")
    assert g.min() >= 10 ** (-MULTIBAND_MAX_GR_DB / 20) - 1e-9
    assert g.max() <= 1.0 + 1e-9


@pytest.mark.parametrize("fs", RATES)
@pytest.mark.parametrize("decoh", [0.0, 0.5, 1.0])
def test_side_makeup_rule(fs, decoh):
    """S makeup 0 below the high side; high-mid / air S bands +2 dB * (1 - decoherence);
    M no makeup unless planned. Tone gains match the independent band-response prediction."""
    n = 2 * fs
    p = MultibandPlan(_flat(), crossovers_hz=OLD_XO, decoherence=[decoh] * NB)
    assert p.side_makeup() == [0.0, 0.0, 0.0, 2.0 * (1 - decoh), 2.0 * (1 - decoh)]
    side = np.array(p.side_makeup())
    for f in (50.0, 250.0, 1000.0, 4000.0, 14000.0):
        for ch, gains in ((0, np.zeros(NB)), (1, side)):
            x = _ms_tone(f, n, ch, fs)
            got = _gain_db(multiband.apply(x, fs, p), x, ch, fs)
            expect = _expect_db(f, gains, fs)
            assert abs(got - expect) < 2e-3, (f, ch, got, expect)
    x = _ms_tone(16000.0, n, 1, fs)
    assert abs(_gain_db(multiband.apply(x, fs, p), x, 1, fs) - 2.0 * (1 - decoh)) < 0.02


def test_side_makeup_keyed_on_geometric_centre():
    """Audit 05 A10: an 1800 Hz crossover keeps the high-mid widening (was 0)."""
    p = MultibandPlan(_flat(), crossovers_hz=[100.0, 500.0, 1800.0, 8000.0])
    assert p.side_makeup() == [0.0, 0.0, 0.0, 2.0, 2.0]
    # [500, 3000] Hz centres at sqrt(500 * 3000) = 1224.7 Hz, below 2 kHz: no widening
    p = MultibandPlan(_flat(), crossovers_hz=[100.0, 500.0, 3000.0, 8000.0])
    assert p.side_makeup() == [0.0, 0.0, 0.0, 2.0, 2.0]
    # [1000, 5000] Hz centres at 2236 Hz: widened
    p = MultibandPlan(_flat(), crossovers_hz=[100.0, 1000.0, 5000.0, 10000.0])
    assert p.side_makeup() == [0.0, 0.0, 2.0, 2.0, 2.0]


def test_mid_makeup_applies_to_m_only():
    n = 2 * FS
    p = MultibandPlan(_flat(), crossovers_hz=OLD_XO, mid_makeup_db=[0.0, 0.0, 1.5, 0.0, 0.0],
                      side_makeup_db=0.0)
    xm = _ms_tone(1000.0, n, 0)
    xs = _ms_tone(1000.0, n, 1)
    expect = _expect_db(1000.0, [0.0, 0.0, 1.5, 0.0, 0.0])
    assert abs(_gain_db(multiband.apply(xm, FS, p), xm, 0) - expect) < 2e-3
    assert abs(_gain_db(multiband.apply(xs, FS, p), xs, 1)) < 1e-6


@pytest.mark.parametrize("fs", RATES)
def test_design_bounds_and_lookahead(fs):
    n = 2 * fs
    x = syn.pink_noise(n, fs, rms_dbfs=-30, seed=4)
    burst = np.zeros(n)
    i0 = fs
    m = int(0.05 * fs)
    burst[i0:i0 + m] = 0.8 * np.sin(2 * np.pi * 1000 * np.arange(m) / fs)
    x = x + bands.to_lr(np.stack([burst, np.zeros(n)], axis=1))
    p = multiband.design(x, fs, np.full((NB, 2), -20.0), ratio=4.0, attack_ms=10.0,
                         release_ms=100.0)
    for pair in p.gr_db:
        for e in pair:
            assert e.values.min() >= -MULTIBAND_MAX_GR_DB - 1e-6 and e.values.max() <= 0.0
    g = p.gr_db[2][0].values
    k0 = int(i0 * 1000 / fs)
    assert g[k0] < -MULTIBAND_MAX_GR_DB + 0.1
    b2 = bands.split(bands.to_ms(x), fs, p.crossovers_hz)[2, :, 0]
    k_cross = int(np.flatnonzero(np.abs(b2) > 10 ** (-20 / 20))[0] * 1000 // fs)
    first = np.flatnonzero(g < -1e-9)[0]
    print(f"{fs} Hz design: GR starts {k_cross - first} ms before the threshold crossing "
          f"({k0 - first} ms before the burst), attack 10 ms")
    assert k_cross - first <= 10


@pytest.mark.parametrize("fs", RATES)
def test_design_bound_holds_between_control_points(fs):
    n = fs
    t = np.arange(n) / fs
    m = 0.9 * np.sin(2 * np.pi * 1000 * t) * (t > 0.5)
    x = bands.to_lr(np.stack([m, np.zeros(n)], axis=1))
    thr_db, ratio = -6.0, 100.0
    p = multiband.design(x, fs, np.full((NB, 2), thr_db), ratio=ratio, attack_ms=5.0,
                         release_ms=50.0)
    for k, pair in enumerate(p.gr_db):
        for c, e in enumerate(pair):
            if (k, c) != (2, 0):
                assert not np.any(e.values)
    y = multiband.apply(x, fs, p)
    b2 = bands.split(bands.to_ms(x), fs, p.crossovers_hz)[2, :, 0]
    big = np.abs(b2) > 0.05
    g = 1 + (bands.to_ms(y)[:, 0] - bands.to_ms(x)[:, 0])[big] / b2[big]
    g_req = np.maximum(np.minimum(1.0, 10 ** (thr_db / 20) / np.abs(b2[big])) ** (1 - 1 / ratio),
                       10 ** (-MULTIBAND_MAX_GR_DB / 20))
    print(f"{fs} Hz design: max(g - g_req) = {np.max(g - g_req):.2e}")
    assert np.all(g <= g_req + 1e-9)


def test_rejects_bad_plans():
    with pytest.raises(ValueError):
        MultibandPlan(_flat(4))
    with pytest.raises(ValueError):
        MultibandPlan(_flat(), decoherence=[1.5] * NB)
    with pytest.raises(ValueError):
        MultibandPlan(_flat(), crossovers_hz=[100.0, 2000.0, 500.0, 8000.0])


@pytest.mark.parametrize("fs", [44100, 48000, 88200, 96000])
def test_frame_assignment_matches_envelope_interpolation(fs):
    """Audit 05 A9: sample i belongs to the control segment Envelope.at interpolates
    it on, floor(i * 1000 / fs), for every sample (np.round edges put 44.1 kHz
    samples like 44 and 176 one frame late)."""
    n = fs // 5
    nc = control_length(n, fs)
    want = np.floor(np.arange(n) * 1000.0 / fs).astype(int)
    for i in list(range(0, 400)) + [4409, 4410, 4411, n - 1]:
        v = np.zeros(n)
        v[i] = 1.0
        assert np.argmax(multiband._frame_peaks(v, fs, nc, 1000.0)) == want[i], i


# ------------------------------------------------------------ per-band ceiling (brief 06)

CEIL_TOL_DB = 0.05


def _ceiling_case(fs, band, ch, ceiling=-12.0, thr=0.0, bed=None, **kw):
    x, _ = band_impulse_train(fs, band, peak_dbfs=-6.0, channel=ch, bed=bed)
    ceil = [[None, None] for _ in range(NB)]
    ceil[band][ch] = ceiling
    p = multiband.design(x, fs, thr, ceiling_db=ceil, **kw)
    y, info = multiband.apply_with_info(x, fs, p)
    return x, y, p, info


@pytest.mark.parametrize("band,ch,fs", [(b, c, 48000) for b in range(NB) for c in (0, 1)]
                         + [(2, 0, 44100), (2, 1, 96000), (0, 0, 96000), (4, 1, 44100)])
def test_ceiling_gate_band_limited_impulse_train(band, ch, fs):
    """Brief 06 gate: band-limited impulse train above the ceiling; the band's sample peak
    after the stage is <= ceiling + 0.05 dB, every other band is bit-identical, the sum of
    the untouched bands is still the input's, and the compressor cap holds."""
    ceiling = -12.0
    x, y, p, info = _ceiling_case(fs, band, ch, ceiling, side_makeup_db=0.0)
    b_in = bands.split(bands.to_ms(x), fs, p.crossovers_hz)
    assert info["peak_in_dbfs"][band, ch] > ceiling + 1.0  # the train is above the ceiling
    # independent measurement: only band k changed, so its output = input band + (y - x)
    out_band = b_in[band, :, ch] + (bands.to_ms(y) - bands.to_ms(x))[:, ch]
    peak = 20 * np.log10(np.max(np.abs(out_band)))
    print(f"{fs} band {band} ch {ch}: in {info['peak_in_dbfs'][band, ch]:.3f} dBFS, "
          f"out {peak:.9f} dBFS, ceiling GR {info['ceiling_gr_min_db'][band, ch]:.3f} dB")
    assert peak <= ceiling + CEIL_TOL_DB
    assert info["peak_out_dbfs"][band, ch] <= ceiling + CEIL_TOL_DB
    assert info["ceiling_gr_min_db"][band, ch] < -0.5
    for k in range(NB):
        for c in range(2):
            if (k, c) != (band, ch):
                assert np.array_equal(info["bands_out"][k, :, c], b_in[k, :, c]), (k, c)
                assert not np.any(p.ceiling_gr_db[k][c].values)
    rest_in = bands.to_ms(x) - b_in[band]
    rest_out = bands.to_ms(y) - info["bands_out"][band]
    assert np.max(np.abs(rest_out - rest_in)) < 1e-12
    assert info["comp_gr_min_db"].min() >= -MULTIBAND_MAX_GR_DB


@pytest.mark.parametrize("fs", RATES)
def test_ceiling_with_compressor_and_makeup(fs, cap45):
    """Compressor driven past its cap, +2 dB side makeup on the air band, pink bed: the
    compressor GR stays within 3 dB, the ceiling still holds after makeup, and the
    ceiling GR is reported separately (deeper than the compressor cap here)."""
    bed = syn.pink_noise(2 * fs, fs, rms_dbfs=-36, seed=5)
    x, y, p, info = _ceiling_case(fs, 4, 1, ceiling=-14.0, thr=-40.0, bed=bed, ratio=8.0)
    assert info["makeup_db"][4, 1] == 2.0
    assert info["comp_gr_min_db"].min() >= -MULTIBAND_MAX_GR_DB
    assert info["comp_gr_min_db"][4, 1] < -2.9
    assert info["peak_out_dbfs"][4, 1] <= -14.0 + CEIL_TOL_DB
    assert info["ceiling_gr_min_db"][4, 1] < -MULTIBAND_MAX_GR_DB
    print(f"{fs}: comp GR {info['comp_gr_min_db'][4, 1]:.3f}, ceiling GR "
          f"{info['ceiling_gr_min_db'][4, 1]:.3f}, out peak {info['peak_out_dbfs'][4, 1]:.6f}")
    for k in range(NB):
        for c in range(2):
            assert np.all(p.gr_db[k][c].values >= -MULTIBAND_MAX_GR_DB - 1e-6)


def test_ceiling_below_signal_is_identity():
    """A ceiling the band never reaches leaves a zero envelope and a bit-identical output."""
    x, _ = band_impulse_train(FS, 2, peak_dbfs=-20.0)
    p = multiband.design(x, FS, 0.0, ceiling_db=-6.0, side_makeup_db=0.0)
    assert all(not np.any(e.values) for pair in p.ceiling_gr_db for e in pair)
    assert np.array_equal(multiband.apply(x, FS, p), bands.to_lr(
        bands.split(bands.to_ms(x), FS, p.crossovers_hz).sum(axis=0)))
    assert np.max(np.abs(multiband.apply(x, FS, p) - x)) < 1e-14


def test_ceiling_lookahead_is_short():
    """Ceiling GR starts no earlier than the lookahead (2 ms) plus one control period
    (the two-segment detector) before the first sample over the ceiling."""
    x, _ = band_impulse_train(FS, 2, peak_dbfs=-6.0, seconds=1.0, period_s=0.5)
    ceil = [[None, None] for _ in range(NB)]
    ceil[2][0] = -12.0
    p = multiband.design(x, FS, 0.0, ceiling_db=ceil, ceiling_lookahead_ms=2.0,
                         side_makeup_db=0.0)
    b2 = bands.split(bands.to_ms(x), FS, p.crossovers_hz)[2, :, 0]
    first_over = np.flatnonzero(np.abs(b2) > 10 ** (-12 / 20))[0]
    g = p.ceiling_gr_db[2][0].values
    first = np.flatnonzero(g < 0)[0]
    lead_ms = first_over * 1000 / FS - first
    print(f"ceiling GR leads the first over by {lead_ms:.3f} ms (lookahead 2 ms)")
    assert lead_ms <= 2.0 + 1.0


def test_ceiling_lookahead_per_band():
    """Maintainer, 2026-10-23: a per-band lookahead. A longer lookahead on band 0 starts that
    band's ceiling GR earlier (by the window difference) and leaves every other band's
    envelope bit-identical; design_from and the JSON round trip keep the per-band list."""
    _, ms0 = band_impulse_train(FS, 0, peak_dbfs=-12.0, seconds=1.0, period_s=0.5)
    _, ms2 = band_impulse_train(FS, 2, peak_dbfs=-12.0, seconds=1.0, period_s=0.5)
    x = bands.to_lr(ms0 + ms2)
    ceil = [[None, None] for _ in range(NB)]
    ceil[0][0] = ceil[2][0] = -18.0
    kw = dict(ceiling_db=ceil, side_makeup_db=0.0)
    p = multiband.design(x, FS, 0.0, ceiling_lookahead_ms=2.0, **kw)
    q = multiband.design(x, FS, 0.0, ceiling_lookahead_ms=[20.0, 2.0, 2.0, 2.0, 2.0], **kw)
    assert p.ceiling_lookahead_ms == 2.0 and p.lookahead_ms() == [2.0] * NB
    assert q.ceiling_lookahead_ms == [20.0, 2.0, 2.0, 2.0, 2.0]

    def first(plan, k):
        return int(np.flatnonzero(plan.ceiling_gr_db[k][0].values < 0)[0])
    print(f"band 0 GR starts at {first(p, 0)} ms (2 ms), {first(q, 0)} ms (20 ms)")
    assert first(q, 0) <= first(p, 0) - 15        # box_smooth: 2 (L // 2) = 20 vs 2 samples
    assert np.array_equal(q.ceiling_gr_db[2][0].values, p.ceiling_gr_db[2][0].values)
    for k in (1, 3, 4):
        assert q.ceiling_gr_db[k] == p.ceiling_gr_db[k]
    r = multiband.design_from(q, x, FS)
    assert r.ceiling_gr_db == q.ceiling_gr_db and r.ceiling_lookahead_ms == q.ceiling_lookahead_ms
    s = Plan.from_json(Plan(fs=FS, multiband=q).to_json()).multiband
    assert s.ceiling_lookahead_ms == q.ceiling_lookahead_ms and s.ceiling_gr_db == q.ceiling_gr_db
    with pytest.raises(ValueError):
        multiband.design(x, FS, 0.0, ceiling_lookahead_ms=[20.0, 2.0], **kw)
    with pytest.raises(ValueError):
        MultibandPlan(_flat(), ceiling_lookahead_ms=[2.0, 2.0])
    with pytest.raises(ValueError):
        MultibandPlan(_flat(), ceiling_lookahead_ms=[2.0, 2.0, 0.0, 2.0, 2.0])


def test_scalar_ceiling_lookahead_still_loads():
    """A plan.json from before the per-band lookahead (a scalar ceiling_lookahead_ms, a
    limiter section without speclim_lookahead_ms) loads and renders as before."""
    import json
    from eluate import planner
    from eluate.plan import LimiterPlan
    x, _ = band_impulse_train(FS, 2, peak_dbfs=-6.0, seconds=1.0, period_s=0.5)
    p = multiband.design(x, FS, 0.0, ceiling_db=-12.0, side_makeup_db=0.0)
    d = json.loads(Plan(fs=FS, multiband=p, limiter=LimiterPlan()).to_json())
    d["multiband"]["ceiling_lookahead_ms"] = 2
    del d["limiter"]["speclim_lookahead_ms"]
    old = Plan.from_dict(d)
    assert old.multiband.ceiling_lookahead_ms == 2.0 and old.multiband.lookahead_ms() == [2.0] * NB
    assert old.limiter.speclim_lookahead_ms is None and planner.SPECLIM_LOOKAHEAD_MS == 2.0
    assert old.multiband.ceiling_gr_db == p.ceiling_gr_db
    assert np.array_equal(multiband.apply(x, FS, old.multiband), multiband.apply(x, FS, p))
    assert multiband.design_from(old.multiband, x, FS).ceiling_gr_db == p.ceiling_gr_db


def test_per_band_attack_release_recorded_and_effective():
    """attack_ms / release_ms are bands x [M, S] plan fields; changing one band's attack
    changes that band's lookahead only, and design_from rebuilds identical envelopes."""
    n = 2 * FS
    t = np.arange(n) / FS
    m = np.zeros(n)
    for f in (250.0, 1000.0):
        m += 0.4 * np.sin(2 * np.pi * f * t) * (t > 1.0)
    x = bands.to_lr(np.stack([m, np.zeros(n)], axis=1))
    att = np.full((NB, 2), 5.0)
    att[2, 0] = 40.0
    p = multiband.design(x, FS, -30.0, ratio=4.0, attack_ms=att, release_ms=[50, 60, 70, 80, 90],
                         ceiling_db=-20.0)
    assert np.array_equal(np.asarray(p.attack_ms), att)
    assert [r[0] for r in p.release_ms] == [50, 60, 70, 80, 90]

    split = bands.split(bands.to_ms(x), FS, p.crossovers_hz)

    def lead(k):  # ms from first GR to the band's first sample over the threshold
        cross = np.flatnonzero(np.abs(split[k, :, 0]) > 10 ** (-30 / 20))[0] * 1000 / FS
        return cross - np.flatnonzero(p.gr_db[k][0].values < -1e-9)[0]
    print(f"lookahead lead: band 1 {lead(1):.2f} ms (attack 5), "
          f"band 2 {lead(2):.2f} ms (attack 40)")
    # bound: attack + one control period (a sample in segment k sets control point k)
    assert lead(1) <= 5.0 + 1.0 and 30.0 <= lead(2) <= 40.0 + 1.0
    q = multiband.design_from(p, x, FS)
    assert q.gr_db == p.gr_db and q.ceiling_gr_db == p.ceiling_gr_db
    r = Plan.from_json(Plan(fs=FS, multiband=p).to_json()).multiband
    assert r.gr_db == p.gr_db and r.ceiling_gr_db == p.ceiling_gr_db
    assert r.attack_ms == p.attack_ms and r.ceiling_db == p.ceiling_db and r.ratio == p.ratio


def test_ceiling_plan_validation():
    bad = _flat()
    bad[0][0] = Envelope.constant(0.5)
    with pytest.raises(ValueError):
        MultibandPlan(_flat(), ceiling_gr_db=bad)
    with pytest.raises(ValueError):
        MultibandPlan(_flat(), ceiling_gr_db=_flat(4))
    with pytest.raises(ValueError):
        MultibandPlan(_flat(), attack_ms=[[0.0, 1.0]] * NB)
    with pytest.raises(ValueError):
        MultibandPlan(_flat(), ceiling_db=[[float("nan"), None]] * NB)
    with pytest.raises(ValueError):
        MultibandPlan(_flat(), ratio=1.0)
    with pytest.raises(ValueError):
        multiband.design_from(MultibandPlan(_flat()), np.zeros((100, 2)), FS)


# ------------------------------------------------------------ crest loss hook (brief 06)

def test_band_crest_matches_features_crest():
    """band_crest is an independent implementation of the spec 3.3 definition; it agrees
    with features/crest.py on the series and the p95 summary."""
    from eluate.features import crest as fcrest
    x, _ = kick_plus_sustain(FS)
    x = x + syn.pink_noise(x.shape[0], FS, rms_dbfs=-30, seed=3)
    a = multiband.band_crest(x, FS)
    b = fcrest.analyse(x, FS)
    assert np.allclose(a["p95_db"], b["p95_db"], atol=1e-9, equal_nan=True)
    assert np.allclose(a["crest_db"], b["crest_db"], atol=1e-9, equal_nan=True)


def test_crest_loss_identity_and_gain_invariance():
    x, _ = kick_plus_sustain(FS)
    r = multiband.crest_loss(x, x, FS)
    assert np.all(r["loss_db"][np.isfinite(r["loss_db"])] == 0.0) and r["ok"].all()
    r = multiband.crest_loss(x, 0.5 * x, FS)  # crest is level-independent
    assert np.nanmax(np.abs(r["loss_db"])) < 1e-9


def test_crest_loss_sees_the_ceiling():
    """Fixture: band-limited impulse train in the mid band (M) over a -24 dBFS pink bed.
    A ceiling under its peak costs the mid band M crest. The hook re-splits the render, so
    the change also reads in the adjacent bands (the order-2 zero-phase crossovers overlap:
    the band-2 train has content in analysis bands 1 and 3); the S view, untouched, and
    the non-adjacent bands lose under 0.01 dB."""
    bed = syn.pink_noise(4 * FS, FS, rms_dbfs=-24, seed=8)
    x, _ = band_impulse_train(FS, 2, seconds=4.0, peak_dbfs=-6.0, bed=bed)
    ceil = [[None, None] for _ in range(NB)]
    ceil[2][0] = -12.0
    p = multiband.design(x, FS, 0.0, ceiling_db=ceil, side_makeup_db=0.0)
    y = multiband.apply(x, FS, p)
    r = multiband.crest_loss(x, y, FS, limits_db=[1.5, 2.0, 0.5, 3.0, 3.0])
    print("crest loss dB (bands x M, S, stereo):", np.round(r["loss_db"], 3))
    assert r["loss_db"][2, 0] > 0.5 and not r["ok"][2, 0] and r["worst_excess_db"][2] > 0
    assert np.all(r["ok"][[0, 1, 3, 4]])
    assert np.nanmax(np.abs(r["loss_db"][:, 1])) < 1e-9
    assert np.nanmax(np.abs(r["loss_db"][[0, 4]])) < 0.01  # measured 2.2e-4 (filter tails)


def test_crest_loss_on_kick_fixture_under_ceiling():
    """Kick plus sustain, ceilings 3 dB under the sub and low band M peaks: crest loss is
    positive in those bands and reported against the spec 3.6 limits (1.5 / 2 dB)."""
    x, _ = kick_plus_sustain(FS)
    pk = [20 * np.log10(np.max(np.abs(b[:, 0])))
          for b in bands.split(bands.to_ms(x), FS, bands.DEFAULT_CROSSOVERS)]
    ceil = [[pk[0] - 3.0, None], [pk[1] - 3.0, None], [None, None], [None, None], [None, None]]
    p = multiband.design(x, FS, 0.0, ceiling_db=ceil, side_makeup_db=0.0)
    r = multiband.crest_loss(x, multiband.apply(x, FS, p), FS)
    print("kick fixture crest loss M:", np.round(r["loss_db"][:, 0], 3), "ok", r["ok"][:, 0])
    assert r["loss_db"][0, 0] > 0.2 and r["loss_db"][1, 0] > 0.2
    assert np.nanmax(np.abs(r["loss_db"][2:, 0])) < 0.2  # leakage into the analysis bands above
    assert r["ok"].all()


def test_crest_loss_rejects_mismatch():
    with pytest.raises(ValueError):
        multiband.crest_loss(np.zeros((100, 2)), np.zeros((99, 2)), FS)
    x = syn.pink_noise(FS, FS, seed=1)
    with pytest.raises(ValueError):
        multiband.crest_loss(x, x, FS, crossovers_hz=[100.0, 1000.0])


# ------------------------------------------------------------ revision 2 (audit 06 fix list)

def test_crest_loss_silenced_band_is_a_violation():
    """Audit 06 F5 / M4: a band gated before and silenced after is never ok, worst +inf."""
    x, _ = kick_plus_sustain(FS)
    x = x + syn.pink_noise(x.shape[0], FS, rms_dbfs=-30, seed=3)
    # brick-wall low-pass at 1.5 kHz: the air band's re-split content drops below -70 dBFS
    # (subtracting the air band is not enough: the zero-phase split is not idempotent)
    spec = np.fft.rfft(x, axis=0)
    spec[np.fft.rfftfreq(x.shape[0], 1 / FS) > 1500.0] = 0.0
    y = np.fft.irfft(spec, n=x.shape[0], axis=0)
    r = multiband.crest_loss(x, y, FS)
    print("silenced:", r["silenced"][:, 0], "worst", r["worst_excess_db"])
    assert r["silenced"][4].all() and not r["ok"][4].any() and r["worst_excess_db"][4] == np.inf
    assert not r["silenced"][:4].any()
    r = multiband.crest_loss(x, x * 1e-4, FS)  # -80 dB everywhere
    assert r["silenced"].all() and not r["ok"].any()
    assert np.all(r["worst_excess_db"] == np.inf)
    assert not np.any(r["worst_excess_db"] == -np.inf)


def test_crest_loss_unmeasured_band_and_never_minus_inf():
    """A band with no content before or after: loss NaN, ok, worst NaN (not -inf)."""
    t = np.arange(2 * FS) / FS
    s = 0.3 * np.sin(2 * np.pi * 1000 * t)
    x = np.stack([s, s], axis=1)
    r = multiband.crest_loss(x, x, FS)
    assert r["unmeasured"][:, 1].all()  # no S anywhere
    assert np.isnan(r["worst_excess_db"]).sum() + np.isfinite(r["worst_excess_db"]).sum() == 5
    assert r["ok"].all() and not np.any(r["worst_excess_db"] == -np.inf)


def test_crest_loss_gate_straddle_is_stable():
    """Audit 06 M6: the same signal at -59.9 and -60.1 dBFS band RMS gives loss 0 on one
    shared frame set, not NaN."""
    x = syn.pink_noise(3 * FS, FS, rms_dbfs=-20, seed=6)
    rb = multiband.band_crest(x, FS)["rms_dbfs"][2, 0]
    g_up = 10 ** ((-59.9 - np.median(rb)) / 20)
    g_dn = 10 ** ((-60.1 - np.median(rb)) / 20)
    r = multiband.crest_loss(x * g_up, x * g_dn, FS)
    print("straddle loss mid band:", r["loss_db"][2])
    assert np.all(np.isfinite(r["loss_db"][2, :2])) and np.nanmax(np.abs(r["loss_db"])) < 1e-9
    assert not r["silenced"].any()


def test_crest_loss_uses_plan_crossovers_and_early_transient():
    """Plan crossovers are used; a transient only in the first 0.65 s is still measured
    (audit 06 M13: interior trim falls back to all gated frames)."""
    p = MultibandPlan([[Envelope.constant(0.0)] * 2 for _ in range(4)],
                      crossovers_hz=[150.0, 1000.0, 6000.0])
    x = syn.pink_noise(2 * FS, FS, rms_dbfs=-20, seed=2)
    r = multiband.crest_loss(x, x, FS, plan=p, limits_db=[1.5, 2.0, 3.0, 3.0])
    assert r["loss_db"].shape == (4, 3)
    n = 3 * FS
    e = np.zeros((n, 2))
    e[:int(0.6 * FS)] = syn.pink_noise(int(0.6 * FS), FS, rms_dbfs=-20, seed=9)
    c = multiband.band_crest(e, FS)
    assert np.all(np.isfinite(c["p95_db"][:, 0]))


def test_ceiling_makeup_consistency_and_validation_after_design():
    """Audit 06 M7: the ceiling records the makeup it saw; a plan with different makeup, or a
    ceiling envelope above 0 dB, is rejected on construction and by apply."""
    x, _ = band_impulse_train(FS, 2, peak_dbfs=-6.0)
    ceil = [[None, None] for _ in range(NB)]
    ceil[2][0] = -12.0
    p = multiband.design(x, FS, 0.0, ceiling_db=ceil, side_makeup_db=0.0)
    assert p.ceiling_makeup_db[2][0] == 0.0
    import dataclasses
    with pytest.raises(ValueError):
        dataclasses.replace(p, mid_makeup_db=[0.0, 0.0, 3.0, 0.0, 0.0])
    p.mid_makeup_db = [0.0, 0.0, 3.0, 0.0, 0.0]  # mutated after construction
    with pytest.raises(ValueError):
        multiband.apply(x, FS, p)
    q = multiband.design(x, FS, 0.0, ceiling_db=ceil, side_makeup_db=0.0,
                         mid_makeup_db=[0.0, 0.0, 3.0, 0.0, 0.0])
    _, info = multiband.apply_with_info(x, FS, q)
    assert info["peak_out_dbfs"][2, 0] <= -12.0 + CEIL_TOL_DB  # designed with the makeup: holds
    q.ceiling_gr_db[1][1] = Envelope.constant(2.0)
    with pytest.raises(ValueError):
        multiband.apply(x, FS, q)


def test_ceiling_g_min_floor():
    """Audit 06 F4 / M9: a ceiling needing more than 120 dB (envelope.G_MIN) is refused."""
    x, _ = band_impulse_train(FS, 2, peak_dbfs=-6.0)
    ceil = [[None, None] for _ in range(NB)]
    ceil[2][0] = -136.0
    with pytest.raises(ValueError, match="envelope floor"):
        multiband.design(x, FS, 0.0, ceiling_db=ceil)
    ceil[2][0] = -120.0  # needs about 110 dB: within the floor, and the bound holds
    p = multiband.design(x, FS, 0.0, ceiling_db=ceil, side_makeup_db=0.0)
    _, info = multiband.apply_with_info(x, FS, p)
    print(f"ceiling -120 dBFS: out {info['peak_out_dbfs'][2, 0]:.6f}")
    assert info["peak_out_dbfs"][2, 0] <= -120.0 + CEIL_TOL_DB


def test_per_band_ceiling_release():
    """Audit 06 M11: ceiling_release_ms is bands x [M, S], recorded and effective; 0 rejected."""
    x, _ = band_impulse_train(FS, 0, peak_dbfs=-6.0, seconds=1.0, period_s=0.5)
    rel = [[20.0, 20.0]] * NB
    rel = [list(r) for r in rel]
    rel[0][0] = 400.0
    p = multiband.design(x, FS, 0.0, ceiling_db=[[-12.0, None]] + [[None, None]] * 4,
                         ceiling_release_ms=rel)
    q = multiband.design(x, FS, 0.0, ceiling_db=[[-12.0, None]] + [[None, None]] * 4,
                         ceiling_release_ms=20.0)
    assert p.ceiling_release_ms[0][0] == 400.0 and q.ceiling_release_ms[0][0] == 20.0
    k = 250 + 100  # 100 ms after the impulse
    assert p.ceiling_gr_db[0][0].values[k] < q.ceiling_gr_db[0][0].values[k] - 1.0
    r = Plan.from_json(Plan(fs=FS, multiband=p).to_json()).multiband
    assert r.ceiling_release_ms == p.ceiling_release_ms
    with pytest.raises(ValueError):
        MultibandPlan(_flat(), ceiling_release_ms=0.0)
    with pytest.raises(ValueError):
        multiband.design(x, FS, 0.0, ceiling_db=-12.0, ceiling_release_ms=0.0)


@pytest.mark.parametrize("phase", ["zero", "minimum", ["minimum", "zero", "zero", "zero"]])
def test_ceiling_under_time_varying_compressor(phase, cap45):
    """Audit 06 F7 / M15: bursts drive the mid-band compressor in and out of GR (envelope
    not pinned at the cap) with a ceiling on top; the processed band's sample peak stays at
    or below the ceiling at every sample, minimum-phase crossovers included."""
    n = 3 * FS
    t = np.arange(n) / FS
    env = np.where((t % 0.75) < 0.2, 1.0, 0.15)
    m = 0.5 * env * np.sin(2 * np.pi * 1000 * t) + 0.3 * np.sin(2 * np.pi * 300 * t) * (t > 1.5)
    x = bands.to_lr(np.stack([m, 0.1 * m], axis=1))
    ceil = [[None, None] for _ in range(NB)]
    ceil[2][0] = -12.0
    p = multiband.design(x, FS, -18.0, ratio=3.0, attack_ms=5.0, release_ms=80.0,
                         ceiling_db=ceil, phase=phase, side_makeup_db=0.0,
                         mid_makeup_db=[0.0, 0.0, 1.0, 0.0, 0.0])
    y, info = multiband.apply_with_info(x, FS, p)
    gr = np.clip(p.gr_db[2][0].at(n, FS), -MULTIBAND_MAX_GR_DB, 0.0)
    varying = np.mean((gr < -0.1) & (gr > -2.9))
    print(f"{phase}: compressor GR between -0.1 and -2.9 dB on {100 * varying:.1f} % of samples; "
          f"out peak {info['peak_out_dbfs'][2, 0]:.6f}, comp {info['comp_gr_min_db'][2, 0]:.3f}, "
          f"ceiling {info['ceiling_gr_min_db'][2, 0]:.3f}, total {info['total_gr_min_db'][2, 0]:.3f}")
    assert varying > 0.05
    assert info["peak_out_dbfs"][2, 0] <= -12.0 + CEIL_TOL_DB
    assert info["comp_gr_min_db"].min() >= -MULTIBAND_MAX_GR_DB
    assert info["ceiling_gr_min_db"][2, 0] < -0.5


def test_total_gr_reported_next_to_capped_compressor_gr(cap45):
    """Audit 06 F3 / M5: total band GR (compressor + ceiling) is reported; it may exceed the
    compressor's 3 dB cap, which still holds on its own."""
    bed = syn.pink_noise(2 * FS, FS, rms_dbfs=-36, seed=5)
    x, y, p, info = _ceiling_case(FS, 4, 1, ceiling=-14.0, thr=-40.0, bed=bed, ratio=8.0)
    tot = info["total_gr_min_db"][4, 1]
    print(f"air S: comp {info['comp_gr_min_db'][4, 1]:.3f}, ceiling "
          f"{info['ceiling_gr_min_db'][4, 1]:.3f}, total {tot:.3f}, total with makeup "
          f"{info['total_gain_min_db'][4, 1]:.3f} dB")
    assert info["comp_gr_min_db"][4, 1] >= -MULTIBAND_MAX_GR_DB
    assert tot < -MULTIBAND_MAX_GR_DB
    assert tot >= info["comp_gr_min_db"][4, 1] + info["ceiling_gr_min_db"][4, 1] - 1e-9
    assert abs(info["total_gain_min_db"][4, 1] - (tot + 2.0)) < 1e-9


def test_design_span_recorded():
    x = syn.pink_noise(int(1.5 * FS), FS, seed=1)
    p = multiband.design(x, FS, -20.0)
    assert p.design_span_s == 1.5
    assert Plan.from_json(Plan(fs=FS, multiband=p).to_json()).multiband.design_span_s == 1.5


def test_fixture_resplit_is_not_zero_in_other_bands():
    """Audit 06 M14: the band-2 train re-splits into the adjacent bands (documented)."""
    x, _ = band_impulse_train(FS, 2, peak_dbfs=-6.0)
    assert np.max(np.abs(x)) <= 0.95
    b = bands.split(bands.to_ms(x), FS, bands.DEFAULT_CROSSOVERS)
    pk = [20 * np.log10(np.max(np.abs(b[k, :, 0])) + 1e-300) for k in range(NB)]
    print("re-split band peaks dBFS:", np.round(pk, 2))
    assert pk[1] > -40 and pk[3] > -40
