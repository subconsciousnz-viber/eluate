"""Spec 7 resonance gates and the resonance.py contract."""

import numpy as np
import pytest

from fixtures import synthetic as syn
from eluate.features import resonance

FS = 48000


@pytest.fixture(scope="module")
def ring_in_pink():
    x = syn.pink_noise(FS * 20, FS, seed=3)
    y, meta = syn.add_ring(x, FS, f0=7000.0, q=12.0, gain_db=9.0, start_s=6.0, end_s=14.0)
    return x, y, meta


def test_injected_ring_gate(ring_in_pink):
    """Gate: 7 kHz Q=12 +9 dB ring in pink noise for 40 % of the duration:
    f0 +-1 %, start/end +-200 ms; flagged as a resonance."""
    _, y, meta = ring_in_pink
    res = [r for r in resonance.analyse(y, FS)["resonances"] if r["flag"] == "resonance"]
    assert len(res) == 1, res
    r = res[0]
    assert abs(r["f0"] / 7000.0 - 1.0) < 0.01
    assert abs(r["start"] - meta["start_s"]) < 0.2
    assert abs(r["end"] - meta["end_s"]) < 0.2
    assert r["q"] > resonance.Q_MIN and r["depth_scale"] == 1.0
    assert 0.35 < r["persistence"][0] < 0.45
    assert r["excess_db"].shape == (200,)
    on = (np.arange(200) + 0.5) * 0.1
    assert np.all(r["excess_db"][(on > 6.5) & (on < 13.5)] > resonance.EXCESS_DB)
    assert np.all(r["excess_db"][(on < 5.5) | (on > 14.5)] == 0.0)


def test_pink_noise_alone_has_no_candidates(ring_in_pink):
    x, _, _ = ring_in_pink
    assert resonance.analyse(x, FS)["resonances"] == []


@pytest.mark.parametrize("f0,n_harm", [(220.0, 12), (110.0, 20), (330.0, 16), (55.0, 40)])
@pytest.mark.slow
def test_sustained_note_not_flagged(f0, n_harm):
    """Gate: sustained synth note with harmonics is not flagged as a resonance
    (its partials are persistent peaks, all classed musical or sub-exempt)."""
    x = syn.synth_note(FS * 20, FS, f0, n_harm)
    res = resonance.analyse(x, FS)["resonances"]
    assert len(res) > 0  # the partials are seen ...
    assert not [r for r in res if r["flag"] == "resonance"]  # ... and none is flagged
    for r in res:
        assert r["depth_scale"] < 1.0


@pytest.mark.parametrize("f0,start,end,want", [(65.0, 2.0, 18.0, "sub_exempt"),
                                               (120.0, 2.0, 18.0, "resonance"),
                                               (120.0, 12.0, 18.0, "sub_exempt")])
def test_low_exemption_rules(f0, start, end, want):
    """Spec 3.3 draft 0.4: below 80 Hz exempt; 80 to 150 Hz only with persistence
    >= 50 % (80 % of the time passes; 30 % does not, read as about 0.4 after the 3 s
    smoothing widens presence by about +-1.5 s) and half the depth."""
    x = syn.pink_noise(FS * 20, FS, seed=4)
    y, _ = syn.add_ring(x, FS, f0=f0, q=12.0, gain_db=12.0, start_s=start, end_s=end)
    res = resonance.analyse(y, FS)["resonances"]
    near = [r for r in res if abs(r["f0"] / f0 - 1) < 0.03]
    assert len(near) == 1 and near[0]["flag"] == want, [(r["f0"], r["flag"], r["persistence"]) for r in res]
    if want == "resonance":
        assert near[0]["depth_scale"] == resonance.LOW_DEPTH_SCALE
    else:
        assert near[0]["depth_scale"] == 0.0


@pytest.mark.slow
def test_persistence_is_per_section():
    """A ring present for 25 % of the track but 50 % of one section is persistent
    with sections given, not without."""
    x = syn.pink_noise(FS * 20, FS, seed=5)
    y, _ = syn.add_ring(x, FS, f0=5000.0, q=15.0, gain_db=10.0, start_s=10.0, end_s=15.0)
    assert not resonance.analyse(y, FS)["resonances"]
    secs = [{"start": 0.0, "end": 10.0}, {"start": 10.0, "end": 20.0}]
    res = resonance.analyse(y, FS, sections=secs)["resonances"]
    assert len(res) == 1 and res[0]["persistence"][1] > 0.4 and res[0]["persistence"][0] < 0.05


@pytest.mark.parametrize("f0,q", [(800.0, 8.0), (3000.0, 10.0), (450.0, 10.0), (200.0, 12.0)])
@pytest.mark.slow
def test_rings_across_resolutions(f0, q):
    """Rings in each STFT resolution over a level-stepped arrangement: found once
    (no split), f0 within 2 %, span within half the smoothing window: 1.5 s at
    N = 16384 (3 s moving mean below 500 Hz), 0.3 s above."""
    x, _ = syn.sectioned_track(44100)
    y, _ = syn.add_ring(x, 44100, f0=f0, q=q, gain_db=9.0, start_s=15.0, end_s=45.0)
    secs = [{"start": a, "end": a + 15.0} for a in (0.0, 15.0, 30.0, 45.0, 60.0)]
    res = resonance.analyse(y, 44100, sections=secs)["resonances"]
    near = [r for r in res if abs(r["f0"] / f0 - 1) < 0.05]
    assert len(near) == 1, [(r["f0"], r["flag"]) for r in res]
    r = near[0]
    assert abs(r["f0"] / f0 - 1) < 0.02
    tol = 1.5 if f0 < 500 else 0.3
    assert abs(r["start"] - 15.0) < tol and abs(r["end"] - 45.0) < tol


def test_parabolic_interpolation_off_bin():
    """A pure tone between bins is located well inside one bin (N = 2048 at 48 kHz:
    23.4 Hz bins)."""
    t = np.arange(FS * 10) / FS
    f = 5011.7
    y = 0.05 * np.sin(2 * np.pi * f * t)
    x = np.stack([y, y], 1) + syn.pink_noise(FS * 10, FS, rms_dbfs=-50, seed=6)
    res = resonance.analyse(x, FS)["resonances"]
    near = [r for r in res if abs(r["f0"] - f) < 50]
    assert near and abs(near[0]["f0"] - f) < 2.0


@pytest.mark.parametrize("f0,q", [(3000.0, 10.0), (1650.0, 15.0), (500.0, 12.0)])
@pytest.mark.slow
def test_ring_between_note_partials_is_flagged(f0, q):
    """A ring sitting between a 220 Hz note's partials is a resonance, the partials
    stay musical (the ladder rule; the peak rule calls all three rings musical)."""
    note = syn.synth_note(FS * 20, FS, 220.0, 12)
    y, _ = syn.add_ring(note, FS, f0=f0, q=q, gain_db=9.0, start_s=2.0, end_s=18.0)
    res = resonance.analyse(y, FS)["resonances"]
    ring = [r for r in res if abs(r["f0"] / f0 - 1) < 0.03]
    assert len(ring) == 1 and ring[0]["flag"] == "resonance", [(r["f0"], r["flag"]) for r in res]
    assert ring[0]["ladder_fraction"] < resonance.LADDER_MIN
    others = [r for r in res if r is not ring[0] and r["flag"] != "sub_exempt"]
    assert others and all(r["flag"] in ("musical", "tonal") for r in others)


@pytest.mark.slow
def test_side_only_ring_is_found_in_s():
    """Audit 04 A4: a +9 dB Q 12 ring in S only, S/M -10 dB, was invisible in the
    stereo power; now found with channel "S" (and an M-only ring with "M")."""
    from eluate import bands
    pm = syn.pink_noise(FS * 20, FS, seed=7)[:, 0]
    ps = syn.pink_noise(FS * 20, FS, seed=8)[:, 0] * 10 ** (-10 / 20)
    rs, _ = syn.add_ring(ps[:, None], FS, 3000.0, 12.0, 9.0, 4.0, 16.0)
    res = resonance.analyse(bands.to_lr(np.stack([pm, rs[:, 0]], 1)), FS)["resonances"]
    hit = [r for r in res if abs(r["f0"] / 3000 - 1) < 0.02]
    assert len(hit) == 1 and hit[0]["channel"] == "S" and hit[0]["flag"] == "resonance"
    rm, _ = syn.add_ring(pm[:, None], FS, 3000.0, 12.0, 9.0, 4.0, 16.0)
    res = resonance.analyse(bands.to_lr(np.stack([rm[:, 0], ps], 1)), FS)["resonances"]
    hit = [r for r in res if abs(r["f0"] / 3000 - 1) < 0.02]
    assert len(hit) == 1 and hit[0]["channel"] == "M"


@pytest.mark.slow
def test_ring_above_14th_partial_is_reported():
    """Audit 04 A1 / A3: a Q 18 ring at 3190 Hz over a 220 Hz note (between partials
    14 and 15) was not reported; now it is, and the partials stay musical."""
    note = syn.synth_note(FS * 20, FS, 220.0, 20)
    y, _ = syn.add_ring(note, FS, 3190.0, 18.0, 9.0, 2.0, 18.0)
    res = resonance.analyse(y, FS)["resonances"]
    ring = [r for r in res if abs(r["f0"] / 3190 - 1) < 0.01]
    assert len(ring) == 1 and ring[0]["flag"] == "resonance"
    assert all(r["flag"] != "resonance" for r in res if r is not ring[0])


def _harmonic(f0, n_h, detune_cents=0.0, B=0.0):
    t = np.arange(FS * 20) / FS
    y = np.zeros(t.size)
    for k in range(1, n_h + 1):
        fk = k * f0 * np.sqrt(1 + B * k * k)
        if fk >= 0.45 * FS:
            break
        for dc in ((0.0,) if detune_cents == 0 else (-detune_cents, detune_cents)):
            y += np.sin(2 * np.pi * fk * 2 ** (dc / 1200) * t + 0.3 * k) / k
    y *= 0.2 / np.max(np.abs(y))
    return np.stack([y, y], 1) + syn.pink_noise(t.size, FS, rms_dbfs=-40, seed=1)


@pytest.mark.parametrize("f0,n_h,detune,B", [(220.0, 16, 8.0, 0.0), (110.0, 24, 0.0, 4e-4)])
@pytest.mark.slow
def test_detuned_and_stiff_notes_are_musical(f0, n_h, detune, B):
    """Audit 04 F9: a detuned unison (+-8 cents) and a piano-like stiff string
    (B = 4e-4: the 12th to 14th partials 30 to 50 cents sharp) are not flagged."""
    res = resonance.analyse(_harmonic(f0, n_h, detune, B), FS)["resonances"]
    assert res and not [r["f0"] for r in res if r["flag"] == "resonance"]


@pytest.mark.parametrize("q,gain", [(12.0, 9.0), (20.0, 9.0), (12.0, 6.0)])
def test_q_matches_the_notch_convention(q, gain):
    """Audit 04 F5 / A2: Q is the width at half the dB boost (the notch stage's
    convention), analytic Q / sqrt(A) for the add_ring peaking response; within
    10 % (before: 9.8 to 24 against 7.15)."""
    x = syn.pink_noise(FS * 20, FS, seed=3)
    y, _ = syn.add_ring(x, FS, 7000.0, q, gain, 6.0, 14.0)
    r = [c for c in resonance.analyse(y, FS)["resonances"] if abs(c["f0"] / 7000 - 1) < 0.03][0]
    want = q / np.sqrt(10 ** (gain / 20))
    assert abs(r["q"] / want - 1) < 0.10, (r["q"], want)
    assert np.isfinite(r["q_uncertainty"]) and r["q_uncertainty"] > 0


def test_gated_low_ring_is_detected():
    """Audit 04 F7: a 300 Hz Q 10 +9 dB ring on for 0.5 s of every 1.25 s."""
    x = syn.pink_noise(FS * 20, FS, seed=5)
    for a in np.arange(0.0, 20.0, 1.25):
        x, _ = syn.add_ring(x, FS, 300.0, 10.0, 9.0, a, min(20.0, a + 0.5))
    res = resonance.analyse(x, FS)["resonances"]
    assert [r for r in res if abs(r["f0"] / 300 - 1) < 0.03 and r["flag"] == "resonance"]


def test_non_circular_ring_from_scipy_iirpeak():
    """A ring made with scipy.signal.iirpeak (not add_ring's RBJ bandpass)."""
    from scipy import signal
    x = syn.pink_noise(FS * 20, FS, seed=9)
    b, a = signal.iirpeak(5000.0, 12.0, fs=FS)
    y = x.copy()
    s0, s1 = 6 * FS, 14 * FS
    y[s0:s1] += (10 ** (9 / 20) - 1) * signal.lfilter(b, a, x[s0:s1], axis=0)
    res = [r for r in resonance.analyse(y, FS)["resonances"] if r["flag"] == "resonance"]
    assert len(res) == 1 and abs(res[0]["f0"] / 5000 - 1) < 0.01
    assert abs(res[0]["start"] - 6) < 0.2 and abs(res[0]["end"] - 14) < 0.2


def test_held_tone_is_tonal_not_resonance():
    """Real-material pass (brief-04 D.33): a sustained sine at 330 Hz over pink noise
    is a line as narrow as the window (and 30+ dB of excess): "tonal", depth 0."""
    t = np.arange(FS * 20) / FS
    y = 0.1 * np.sin(2 * np.pi * 330.0 * t)
    x = np.stack([y, y], 1) + syn.pink_noise(t.size, FS, rms_dbfs=-40, seed=2)
    res = [r for r in resonance.analyse(x, FS)["resonances"] if abs(r["f0"] / 330 - 1) < 0.01]
    assert len(res) == 1 and res[0]["flag"] == "tonal" and res[0]["depth_scale"] == 0.0
    assert res[0]["line"]


def test_chord_tone_rule():
    on = np.array([1.0, 3.0, 5.0])
    chords = [{"start": 0.0, "end": 2.0, "chord": "Am"}, {"start": 2.0, "end": 4.0, "chord": "F"},
              {"start": 4.0, "end": 6.0, "chord": "C"}]
    ok, share = resonance._chord_tone(440.0, on, None, chords)                  # A: in Am and F, not C
    assert ok and abs(share - 2 / 3) < 1e-9
    ok, share = resonance._chord_tone(329.63, on, None, chords)                 # E: in Am and C
    assert ok and abs(share - 2 / 3) < 1e-9
    assert resonance._chord_tone(311.13, on, None, chords)[0] is False          # D#: in none
    assert resonance._chord_tone(311.13, on, {"label": "D#m"}, chords)[0] is True  # tonic triad
    assert resonance._chord_tone(452.0, on, None, chords)[0] is False           # 47 cents off A
