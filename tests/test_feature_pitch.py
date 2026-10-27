"""Spec 7 pitch gate and the pitch.py contract."""

import numpy as np
import pytest

from fixtures import synthetic as syn
from eluate.features import pitch, transients
from eluate.features._common import Context

FS = 44100


def _truth(meta):
    return np.round(69 + 12 * np.log2(meta["roots_per_beat"] / 440.0))


@pytest.fixture(scope="module")
def bassline():
    x, meta = syn.kick_sub(FS, bars=16)
    return x, meta


def test_bass_root_gate_with_true_beats(bassline):
    """Gate: synthetic sub bassline + kick with 250 ms tail, sidechain ducked:
    root per beat >= 95 % correct (64 beats)."""
    x, meta = bassline
    k = transients.analyse(x, FS)["kick_onsets"]
    br = pitch.bass_root(x, FS, meta["beat_times"], k)
    acc = np.mean(br["root_midi"] == _truth(meta))
    assert acc >= 0.95, acc
    assert np.median(br["confidence"]) >= 0.75
    assert not br["glide"].any()


@pytest.mark.slow
def test_bass_root_gate_through_analyse(bassline):
    """Same gate with beats from structure.py, kicks from transients.py."""
    from eluate.features import analyse
    x, meta = bassline
    a = analyse(x, FS)
    beats = a["structure"]["beat_times"]
    truth = _truth(meta)
    # map each estimated beat to the true beat containing its midpoint
    bt = meta["beat_times"]
    ibi = np.median(np.diff(bt))
    idx = np.clip(np.searchsorted(bt, beats + ibi / 2) - 1, 0, bt.size - 1)
    roots = a["pitch"]["bass"]["root_midi"]
    acc = np.mean(roots == truth[idx])
    assert acc >= 0.95, acc
    assert a["plan_inputs"]["keyeq_eligible"]


def test_never_on_stereo_energy_and_per_channel(bassline):
    """Bass only in R (L silent): the per-channel route still finds the root."""
    x, meta = bassline
    y = x.copy()
    y[:, 0] = 0.0
    br = pitch.bass_root(y, FS, meta["beat_times"], meta["kick_onsets"])
    assert np.mean(br["root_midi"] == _truth(meta)) >= 0.95


def test_glide_flag():
    """A bass that slides 2 semitones up through the second half of a beat is flagged."""
    bpm = 120.0
    beat = 0.5
    n = int(8 * beat * FS)
    t = np.arange(n) / FS
    ph = (t % beat) / beat
    midi = 33 + np.where((t // beat) % 2 == 1, np.clip((ph - 0.5) * 2, 0, 1) * 2.0, 0.0)
    f = 440 * 2 ** ((midi - 69) / 12)
    y = 0.4 * np.sin(2 * np.pi * np.cumsum(f) / FS)
    br = pitch.bass_root(np.stack([y, y], 1), FS, np.arange(8) * beat)
    assert list(br["glide"][1::2][:3]) == [True, True, True]
    assert not br["glide"][0::2].any()


def test_silent_beats_have_no_root():
    br = pitch.bass_root(np.zeros((FS * 2, 2)), FS, np.arange(4) * 0.5)
    assert np.all(np.isnan(br["root_midi"])) and np.all(br["confidence"] == 0)


@pytest.mark.parametrize("chords,want", [(("C", "F", "G", "C"), "C"),
                                         (("D", "G", "A", "D"), "D"),
                                         (("Am", "Dm", "Em", "Am"), "Am"),
                                         (("Am", "Am", "F", "G"), "Am")])
def test_key_mode_from_edma_tonic_from_edmm(chords, want):
    """Combined key (spec 3.3 draft 0.4): major loops read major (edmm alone read
    them as the parallel minor), clear minor loops read minor with edmm's tonic."""
    x, meta = syn.chord_loop(FS, chords=chords, loops=2)
    r = pitch.analyse(x, FS, Context(x, FS), bar_times=meta["bar_times"])
    assert r["key"]["label"] == want, r["key"]
    assert r["key"]["edmm"]["label"] is not None and r["key"]["edma"]["label"] is not None
    if r["key"]["mode"] == "minor":
        assert r["key"]["rule"] == "edmm tonic, modes agree"
    assert [c["chord"] for c in r["chords"]] == meta["chords"]


def test_relative_major_ambiguity_is_reported():
    """Am-F-C-G is a vi-IV-I-V loop in C as much as i-VI-III-VII in A minor: edma
    reads C major, edmm A minor, and both are in the output."""
    x, meta = syn.chord_loop(FS, chords=("Am", "F", "C", "G"), loops=2)
    r = pitch.analyse(x, FS, Context(x, FS), bar_times=meta["bar_times"])
    assert r["key"]["label"] == "C" and r["key"]["edmm"]["label"] == "Am"


def test_edmm_major_profile_is_flat_so_major_never_wins():
    """Faraldo's edmm reports major keys as minor (Essentia key.cpp); edma does not."""
    s = pitch.key_from_chroma(np.roll(pitch.EDMA_MAJOR, 2), "edmm")
    assert s["mode"] == "minor" and np.all(np.isnan(s["scores"][:12]))
    e = pitch.key_from_chroma(np.roll(pitch.EDMA_MAJOR, 2), "edma")
    assert e["label"] == "D"


def _acc_through_analyse(x, meta, fs=FS):
    from eluate.features import analyse
    a = analyse(x, fs)
    bt = meta["beat_times"]
    b = a["structure"]["beat_times"]
    idx = np.clip(np.searchsorted(bt, b + np.median(np.diff(bt)) / 2) - 1, 0, bt.size - 1)
    truth = np.round(69 + 12 * np.log2(meta["roots_per_beat"] / 440.0))
    return a, float(np.mean(a["pitch"]["bass"]["root_midi"] == truth[idx]))


def test_174_bpm_is_not_read_at_half_tempo():
    """Audit 04 F2 / F11: librosa reads 174 BPM as 87; the kick-rate octave check
    doubles the grid, bass-root accuracy stays >= 0.95 and keyeq stays honest."""
    x, meta = syn.kick_sub(FS, bpm=174.0, bars=8)
    a, acc = _acc_through_analyse(x, meta)
    st = a["structure"]
    assert abs(st["tempo_bpm"] - 174.0) < 1.0  # metrical hypothesis reads 174 directly (no doubling needed)
    assert abs(st["metrical"]["tempo_bpm"] - 174.0) < 1.0
    assert st["beat_times"].size == meta["beat_times"].size
    assert acc >= 0.95


def test_preroll_does_not_fabricate_a_beat_at_zero():
    """Audit 04 A8: with 400 ms of silence before the first kick, no beat at 0 s."""
    x, meta = syn.kick_sub(FS, bars=8)
    x = np.pad(x, ((int(0.4 * FS), 0), (0, 0)))
    from eluate.features import analyse
    b = analyse(x, FS)["structure"]["beat_times"]
    assert b[0] > 0.35 and abs(b[0] - 0.4) < 0.02


def test_keyeq_gate_needs_nsdf_agreement():
    """Self-agreement is not enough (audit 04 F2): confident roots that the
    independent NSDF estimate does not confirm keep keyeq off."""
    n = 16
    br = {"root_midi": np.full(n, 33.0), "confidence": np.ones(n),
          "nsdf_agree": np.zeros(n, dtype=bool)}
    assert pitch.keyeq_gate(br) == (False, 0.0)
    br["nsdf_agree"][:12] = True
    assert pitch.keyeq_gate(br)[0] is True
    br["confidence"][:] = 0.5
    assert pitch.keyeq_gate(br)[0] is False


@pytest.mark.parametrize("pattern,bpm,kick", [("offbeat", 128.0, True), ("eighths", 140.0, True),
                                              ("beats", 128.0, False), ("offbeat", 128.0, False)])
def test_bass_attacks_do_not_mask_the_bass(pattern, bpm, kick):
    """Audit 04 A5 / F4: bass-note attacks are not kicks, so the pitch mask (kicks
    scoring >= 0.8 only) leaves the bass frames alone; accuracy >= 0.95 (before:
    0.59 with 8th-note bass at 140 BPM, 0.16 on a kickless off-beat bass)."""
    x, meta = syn.bassline(FS, bpm, pattern, kick_on_beats=kick)
    tr = transients.analyse(x, FS)
    br = pitch.bass_root(x, FS, meta["beat_times"], tr["kick_mask_onsets"])
    truth = np.round(69 + 12 * np.log2(meta["roots_per_beat"] / 440.0))
    assert np.mean(br["root_midi"] == truth) >= 0.95


@pytest.mark.parametrize("partials", [(0.0, 1.0, 0.7, 0.5, 0.3), (0.0, 1.0, 0.3, 0.2), (0.1, 1.0, 0.6, 0.4)])
def test_missing_or_weak_fundamental(partials):
    """A bass with no (or a -20 dB) fundamental still reads its root, not the octave
    (audit 04 A6's case; measured with SHS over 4 harmonics, brief-04 D.22)."""
    x, meta = syn.bassline(FS, 126.0, "beats", kick_on_beats=False, partials=partials,
                           roots_hz=(55.0, 55.0, 61.74, 49.0, 41.2, 73.42, 65.41, 46.25))
    br = pitch.bass_root(x, FS, meta["beat_times"])
    truth = np.round(69 + 12 * np.log2(meta["roots_per_beat"] / 440.0))
    assert np.mean(br["root_midi"] == truth) >= 0.95


def test_calibration_tables_are_monotone():
    for edges, p in (pitch.CALIB_PYIN, pitch.CALIB_SHS):
        assert np.all(np.diff(p) >= 0) and len(p) == len(edges)
