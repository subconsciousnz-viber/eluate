"""Gates for stages/glue.py (spec 3.5 item 6; brief 06 revision 2, audit 06 fix list item 1)."""

import numpy as np
import pytest

from fixtures import synthetic as syn
from fixtures.crest_fixtures import FULL_SCALE_PEAK, graded_kicks, kick_plus_sustain
from eluate import envelope
from eluate.plan import (GLUE_MAX_GAIN_DB, STAGE_ORDER, Envelope, GluePlan, Plan,
                            SidestripPlan, control_length)
from eluate.stages import glue, render

FS = 48000
RATES = [44100, 48000, 96000]
CURVES = ["clean", "vintage"]
PEAK_TOL_DB = 0.5


def _heavy_plan(n, fs, curve="clean", mix=0.0):
    nc = control_length(n, fs)
    g = -6.0 * (0.5 + 0.5 * np.sin(np.arange(nc) / 37.0))
    return GluePlan(Envelope(g), mix=mix, makeup_db=4.0, curve=curve, threshold_db=-20.0)


def _attack_peaks(x, y, onsets, fs, g_db=None, mix=0.5):
    """Per kick: attack-peak change (first 5 ms) in dB, and, with g_db, the part not
    explained by the wet gain already present at the onset (release carry-over)."""
    absd, attack = [], []
    for o in onsets:
        i = int(round(o * fs))
        a = slice(i, i + int(0.005 * fs))
        d = 20 * np.log10(np.max(np.abs(y[a])) / np.max(np.abs(x[a])))
        absd.append(d)
        if g_db is not None:
            attack.append(d - 20 * np.log10((1 - mix) + mix * 10 ** (g_db[i] / 20)))
    return np.array(absd), np.array(attack)


def _burst(fs, on_s=0.5, dur_s=1.0, lo=0.05, hi=0.8):
    n = 2 * fs
    t = np.arange(n) / fs
    s = lo * np.sin(2 * np.pi * 200 * t)
    i0, i1 = int(on_s * fs), int((on_s + dur_s) * fs)
    s[i0:i1] = hi * np.sin(2 * np.pi * 200 * t[i0:i1])
    return np.stack([s, s], axis=1), i0


def test_fixtures_below_full_scale():
    for fx in (kick_plus_sustain(FS)[0], graded_kicks(FS)[0]):
        assert np.max(np.abs(fx)) <= FULL_SCALE_PEAK


@pytest.mark.parametrize("curve", CURVES)
def test_mix_zero_is_bit_exact_identity(curve):
    x = syn.pink_noise(FS, FS, rms_dbfs=-10, seed=1)
    y = glue.apply(x, FS, _heavy_plan(x.shape[0], FS, curve, mix=0.0))
    assert np.array_equal(y, x)


@pytest.mark.parametrize("fs", RATES)
@pytest.mark.parametrize("curve", CURVES)
def test_mix_one_below_threshold_is_identity(fs, curve):
    """Detector below the knee everywhere, makeup 0: zero envelope, zero vintage term."""
    x = syn.pink_noise(fs, fs, rms_dbfs=-30, seed=2)
    peak = 20 * np.log10(np.max(np.abs(x)))
    p = glue.design(x, fs, threshold_db=peak + 3.5, curve=curve, mix=1.0)
    assert not np.any(p.gain_db.values) and p.makeup_db == 0.0
    assert np.array_equal(glue.apply(x, fs, p), x)


@pytest.mark.parametrize("fs", RATES)
@pytest.mark.parametrize("curve", CURVES)
def test_lagging_attack_starts_after_onset_and_reaches_63pct(fs, curve):
    """Spec 3.5 item 6 gate: on a burst above threshold the wet-path GR is exactly 0 before
    the onset and reaches 63 % of its depth one attack time (30 ms) after it, within the
    1 to 2 ms causal detector delay plus a control period."""
    x, i0 = _burst(fs)
    p = glue.design(x, fs, threshold_db=-20.0, curve=curve, mix=1.0)
    g = glue.gain_trace_db(p, x.shape[0], fs)
    assert np.all(g[:i0 + 1] == 0.0)
    first = np.flatnonzero(g < 0)[0]
    depth = g[i0 + int(0.4 * fs)]
    t63 = (np.flatnonzero(g <= (1 - np.exp(-1)) * depth)[0] - i0) * 1000 / fs
    # the detector hears the burst through the 120 Hz sidechain high-pass (D3)
    from scipy import signal as sg
    _, hp = sg.sosfreqz(sg.butter(2, glue.SIDECHAIN_HP_HZ, "highpass", fs=fs, output="sos"),
                        worN=[200.0], fs=fs)
    static = glue.gain_computer(20 * np.log10(0.8 * abs(hp[0])), -20.0, 2.0, 6.0)
    print(f"{fs} {curve}: GR starts {(first - i0) * 1000 / fs:.3f} ms after the onset, "
          f"63 % at {t63:.2f} ms, depth {depth:.3f} dB (static {static:.3f})")
    assert first > i0
    assert 30.0 <= t63 <= 30.0 + 3.0
    # steady state follows the static curve; the 1 ms peak detector ripples on a 200 Hz
    # sine and the release recovers between segment peaks (0.055 dB clean, 0.11 vintage)
    assert abs(depth - static) < 0.15


@pytest.mark.parametrize("fs", RATES)
@pytest.mark.parametrize("curve", CURVES)
def test_graded_kicks_gate(fs, curve):
    """Gate (audit 06 M2): kicks at 0 to -12 dB, one per second, mix 0.5, makeup 0: every
    kick's attack peak within 0.5 dB of the input."""
    x, onsets, levels = graded_kicks(fs, bpm=60.0)
    p = glue.design(x, fs, threshold_db=-12.0, curve=curve, mix=0.5)
    y = glue.apply(x, fs, p)
    absd, attack = _attack_peaks(x, y, onsets, fs, glue.gain_trace_db(p, x.shape[0], fs))
    print(f"{fs} {curve} 60 bpm: deepest GR {p.gain_db.values.min():.2f} dB; attack peak change "
          f"per kick {np.round(absd, 3).tolist()} dB, attack-induced {np.round(attack, 3).tolist()}")
    assert np.max(np.abs(absd)) <= PEAK_TOL_DB


@pytest.mark.parametrize("curve", CURVES)
def test_graded_kicks_120bpm_attack_passes(curve):
    """At 120 bpm with the 300 ms release, part of each kick's GR is still present at the
    next onset (release carry-over, what a bus compressor does). The attack itself still
    passes: the peak change beyond the wet gain present at the onset is within 0.5 dB.
    The absolute change is reported (brief 06 D.2r)."""
    x, onsets, levels = graded_kicks(FS, bpm=120.0)
    p = glue.design(x, FS, threshold_db=-12.0, curve=curve, mix=0.5)
    y = glue.apply(x, FS, p)
    absd, attack = _attack_peaks(x, y, onsets, FS, glue.gain_trace_db(p, x.shape[0], FS))
    print(f"{curve} 120 bpm: absolute {np.round(absd, 3).tolist()} dB, "
          f"attack-induced {np.round(attack, 3).tolist()} dB")
    assert np.max(np.abs(attack)) <= PEAK_TOL_DB
    assert absd[0] > -PEAK_TOL_DB  # the first kick has no carry-over


def test_stereo_linked():
    """One gain for both channels: a transient in L alone reduces R by the same factor."""
    n = FS
    t = np.arange(n) / FS
    x = np.zeros((n, 2))
    x[:, 0] = 0.05 * np.sin(2 * np.pi * 200 * t)
    x[:, 1] = 0.05 * np.sin(2 * np.pi * 300 * t)
    x[FS // 2:FS // 2 + 4000, 0] += 0.9 * np.sin(2 * np.pi * 80 * t[:4000])
    p = glue.design(x, FS, threshold_db=-20.0, mix=1.0)
    assert p.gain_db.values.min() < -3.0
    y = glue.apply(x, FS, p)
    g = glue.gain_trace_db(p, n, FS)
    for c in range(2):
        big = np.abs(x[:, c]) > 1e-3
        assert np.allclose(y[big, c] / x[big, c], 10 ** (g[big] / 20), rtol=0, atol=1e-12)


def test_gain_computer_curve():
    lv = np.linspace(-40, 0, 4001)
    gr = glue.gain_computer(lv, -20.0, 2.0, 6.0)
    assert np.all(gr[lv <= -23.0] == 0.0)
    above = lv >= -17.0
    assert np.allclose(gr[above], -(lv[above] + 20.0) / 2.0)
    assert np.all(np.diff(gr) <= 1e-12)  # monotone
    assert np.max(np.abs(np.diff(gr, 2))) < 1e-3  # no corner at the knee edges (C1)
    hard = glue.gain_computer(lv, -20.0, 2.0, 0.0)
    assert np.allclose(hard, np.minimum(0.0, -(lv + 20.0) / 2.0))


def test_envelope_construction():
    """The envelope is the lagging one-pole on the program release of the causal
    detector's required gain, the detector on the 120 Hz high-passed input (D3; spec 3.5
    item 6 attack)."""
    from scipy import signal as sg
    x, _ = kick_plus_sustain(FS, beats=4)
    p = glue.design(x, FS, threshold_db=-12.0)
    side = sg.sosfilt(sg.butter(2, glue.SIDECHAIN_HP_HZ, "highpass", fs=FS, output="sos"), x,
                      axis=0)
    det = glue.causal_detector(np.max(np.abs(side), axis=1), FS,
                               control_length(x.shape[0], FS), 1000.0)
    with np.errstate(divide="ignore"):
        u = np.clip(10 ** (glue.gain_computer(20 * np.log10(det), -12.0, 2.0, 6.0) / 20),
                    envelope.G_MIN, 1.0)
    h = glue.program_release(u, 300.0)
    s = glue.lagging_attack(20 * np.log10(h), 30.0)
    from eluate.stages.multiband import to_env32
    assert p.gain_db == to_env32(10 ** (s / 20), 1000.0)
    # branching one-pole: never deeper than the hull, equal to it once released
    assert np.all(s >= 20 * np.log10(h) - 1e-12)


def test_lagging_attack_step_response():
    h = np.concatenate([np.zeros(10), np.full(200, -6.0), np.zeros(5)])
    s = glue.lagging_attack(h, 30.0)
    assert np.all(s[:10] == 0.0)
    assert abs(s[10 + 29] - (-6.0) * (1 - np.exp(-30 / 30.0))) < 1e-12
    assert np.all(s[210:] == 0.0)  # release follows the target (here an instant hull)


def test_vintage_release_is_program_dependent():
    """After a 20 ms burst the gain recovers faster than after a 1.5 s burst of the same
    level, on both curves (D3: the program release is the glue's release)."""
    fs = FS
    n = 4 * fs
    t = np.arange(n) / fs

    def burst(dur):
        x = 0.02 * np.sin(2 * np.pi * 220 * t)
        m = (t >= 1.0) & (t < 1.0 + dur)
        x[m] = 0.9 * np.sin(2 * np.pi * 220 * t[m])
        return np.stack([x, x], axis=1)

    def recovery_ms(p, end_s):
        g = p.gain_db.values
        k0 = int(end_s * 1000) + 3  # past the causal detector delay
        k_min = k0 - 5 + int(np.argmin(g[k0 - 5:k0 + 5]))
        depth = g[k_min]
        return int(np.flatnonzero(g[k_min:] > depth / np.e)[0])

    rec = {}
    for curve in CURVES:
        for dur in (0.02, 1.5):
            # 2 ms attack so both bursts reach full depth: the test isolates the release
            p = glue.design(burst(dur), fs, threshold_db=-20.0, curve=curve, attack_ms=2.0)
            rec[curve, dur] = recovery_ms(p, 1.0 + dur)
    print("time to recover to 1/e of the depth, ms:", rec)
    for curve in CURVES:
        assert rec[curve, 1.5] > 1.5 * rec[curve, 0.02]


@pytest.mark.parametrize("fs", RATES)
def test_vintage_adds_slight_even_harmonics_on_wet_only(fs):
    """220 Hz sine 6 dB over the threshold, mix 1, no gain change: vintage adds H2 well
    below the fundamental and no DC; clean adds nothing; the curve is purely even."""
    n = 2 * fs
    t = np.arange(n) / fs
    s = 0.5 * np.sin(2 * np.pi * 220 * t)
    x = np.stack([s, s], axis=1)
    p = GluePlan(Envelope.constant(0.0), mix=1.0, curve="vintage", threshold_db=-12.0)
    y = glue.apply(x, fs, p)
    mid = slice(fs // 2, -fs // 2)
    seg = (y - x)[mid, 0]
    w = np.hanning(seg.size)
    spec = np.abs(np.fft.rfft(seg * w))
    ref = np.abs(np.fft.rfft(x[mid, 0] * w))
    f = np.fft.rfftfreq(seg.size, 1 / fs)

    def at(h, sp):
        return sp[np.argmin(np.abs(f - 220 * h))]
    h2 = 20 * np.log10(at(2, spec) / at(1, ref))
    h3 = 20 * np.log10(at(3, spec) / at(1, ref))
    dc = abs(np.mean(seg))
    print(f"{fs}: vintage H2 {h2:.1f} dBc, H3 {h3:.1f} dBc, DC {dc:.2e}")
    assert -45.0 < h2 < -20.0 and h3 < h2 - 10.0 and dc < 1e-4
    clean = GluePlan(Envelope.constant(0.0), mix=1.0, curve="clean", threshold_db=-12.0)
    assert np.array_equal(glue.apply(x, fs, clean), x)
    half = GluePlan(Envelope.constant(0.0), mix=0.5, curve="vintage", threshold_db=-12.0)
    assert np.allclose(glue.apply(x, fs, half) - x, 0.5 * (y - x), atol=1e-15)


def test_vintage_high_pass_is_causal():
    """Audit 06 M10: nothing before the first wet sample above the threshold (the 20 Hz
    high-pass is causal); after the burst the removed DC decays as the filter's tail."""
    x, i0 = _burst(FS, on_s=0.8, dur_s=0.3, lo=0.0, hi=0.5)
    p = GluePlan(Envelope.constant(0.0), mix=1.0, curve="vintage", threshold_db=-12.0)
    e = glue.apply(x, FS, p)[:, 0] - x[:, 0]
    first_over = np.flatnonzero(np.abs(x[:, 0]) > 10 ** (-12 / 20))[0]
    pre = np.max(np.abs(e[:first_over - 400]))  # minus the 4x resampler's half length
    inside = np.max(np.abs(e[i0:i0 + int(0.3 * FS)]))
    tail = np.max(np.abs(e[i0 + int(0.4 * FS):]))
    print(f"vintage change: before burst {pre:.2e}, in burst {inside:.2e}, "
          f"from 100 ms after it {tail:.2e}")
    assert pre < 1e-12 and tail < 0.5 * inside


def test_vintage_engages_on_the_wet_signal():
    """With makeup, input below the threshold can exceed it on the wet path (wording of the
    contract, audit 06 M10): -14 dBFS sine, threshold -12 dBFS, makeup 6 dB."""
    t = np.arange(FS) / FS
    s = 10 ** (-14 / 20) * np.sin(2 * np.pi * 220 * t)
    x = np.stack([s, s], axis=1)
    p = GluePlan(Envelope.constant(0.0), mix=1.0, makeup_db=6.0, curve="vintage",
                 threshold_db=-12.0)
    assert np.any(glue.vintage_change(x * 10 ** (6 / 20), FS, -12.0))
    assert not np.any(glue.vintage_change(x, FS, -12.0))
    assert not np.array_equal(glue.apply(x, FS, p), x * 10 ** (6 / 20))


def test_positive_gain_allowed_within_bound():
    """Audit 06 M3: the schema admits upward gain up to GLUE_MAX_GAIN_DB."""
    x = syn.pink_noise(FS // 2, FS, rms_dbfs=-30, seed=4)
    p = GluePlan(Envelope.constant(3.0), mix=1.0)
    assert np.allclose(glue.apply(x, FS, p), x * 10 ** (3 / 20.0))
    with pytest.raises(ValueError):
        GluePlan(Envelope.constant(GLUE_MAX_GAIN_DB + 0.5))


def test_plan_validation_and_round_trip(tmp_path):
    with pytest.raises(ValueError):
        GluePlan(Envelope.constant(0.0), mix=1.5)
    with pytest.raises(ValueError):
        GluePlan(Envelope.constant(0.0), makeup_db=7.0)
    with pytest.raises(ValueError):
        GluePlan(Envelope.constant(0.0), curve="tube")
    with pytest.raises(ValueError):
        GluePlan(Envelope.constant(0.0), curve="vintage")  # needs threshold_db
    with pytest.raises(ValueError):
        GluePlan(Envelope.constant(0.0), ratio=1.0)
    with pytest.raises(ValueError):
        GluePlan(Envelope.constant(0.0), mix=float("nan"))
    with pytest.raises(ValueError):
        glue.design(np.zeros((100, 2)), FS, -12.0, attack_ms=0.0)
    x, _ = kick_plus_sustain(FS, beats=4)
    p = Plan(fs=FS, glue=glue.design(x, FS, -12.0, curve="vintage", makeup_db=1.5))
    assert p.glue.makeup_db == 1.5
    q = Plan.from_json(p.to_json())
    assert q.to_dict() == p.to_dict()
    p.save(tmp_path / "plan.json")
    r = Plan.load(tmp_path / "plan.json")
    assert r.glue.gain_db == p.glue.gain_db
    assert np.array_equal(glue.apply(x, FS, r.glue), glue.apply(x, FS, p.glue))


def test_render_order_and_skip():
    """Glue sits between saturate and keyeq; render skips it when its section is absent."""
    assert STAGE_ORDER == ("input", "inputeq", "notch", "sidestrip", "multiband", "tone",
                           "saturate", "glue", "speclim", "keyeq", "limiter", "output")
    x, _ = kick_plus_sustain(FS, beats=4)
    _, stages = render(x, FS, Plan(fs=FS, sidestrip=SidestripPlan()))
    assert list(stages) == ["sidestrip"]
    gp = glue.design(x, FS, -12.0, mix=0.5)
    y, stages = render(x, FS, Plan(fs=FS, sidestrip=SidestripPlan(), glue=gp), keep_stages=True)
    assert list(stages) == ["sidestrip", "glue"]
    assert np.array_equal(y, glue.apply(stages["sidestrip"], FS, gp))
