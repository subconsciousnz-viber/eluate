"""Spec 7 structure gate and the structure.py contract."""

import numpy as np
import pytest

from fixtures import synthetic as syn
from eluate.features import structure, transients
from eluate.features._common import Context

FS = 44100


def _run(x):
    ctx = Context(x, FS)
    tr = transients.analyse(x, FS, ctx)
    return structure.analyse(x, FS, ctx, onset=(tr["onset_env_t"], tr["onset_env"]),
                             onset_rate=tr["onset_rate"], kick_onsets=tr["kick_onsets"])


@pytest.mark.parametrize("bpm,levels", [(128.0, (-12.0, 0.0, -6.0, 0.0, -9.0)),
                                        (140.0, (0.0, -6.0, 0.0, -3.0)),
                                        (124.0, (-9.0, 0.0, -9.0, 0.0, -3.0, 0.0))])
def test_boundaries_within_one_bar(bpm, levels):
    """Gate: synthetic 8-bar sections with level steps, boundaries +-1 bar."""
    x, meta = syn.sectioned_track(FS, bpm=bpm, levels_db=levels)
    st = _run(x)
    bar = 240.0 / bpm
    got = np.array(st["boundaries"])
    want = meta["boundaries"]
    assert got.size == want.size, (got, want)
    assert np.max(np.abs(got - want)) <= bar
    assert abs(st["tempo_bpm"] - bpm) < 0.5
    # tighter than the gate on this fixture: downbeats are right, so exact to 50 ms
    assert np.max(np.abs(got - want)) < 0.05
    segs = st["segments"]
    assert segs[0]["start"] == 0.0 and abs(segs[-1]["end"] - x.shape[0] / FS) < 1e-9
    assert all(s["label"] in ("intro", "drop", "build", "breakdown", "body", "outro") for s in segs)
    loud = [s["loudness_lufs"] for s in segs]
    assert segs[0]["label"] == "intro" and segs[-1]["label"] == "outro"
    for s, lv in list(zip(segs, levels))[1:-1]:
        if lv == max(levels):
            assert s["label"] == "drop"


def test_bar_grid_from_downbeats():
    x, meta = syn.sectioned_track(FS, bpm=128.0, levels_db=(-6.0, 0.0, -6.0))
    st = _run(x)
    bars = st["bar_times"]
    assert np.max(np.abs(bars[: meta["bar_times"].size] - meta["bar_times"][: bars.size])) < 0.02
    assert st["grid_kind"] == "constant"


def test_dropout_detector():
    """A one-bar sub + low dropout before a section start is found, and its
    return lands on the boundary."""
    x, meta = syn.sectioned_track(FS, bpm=128.0, levels_db=(0.0, 0.0, 0.0, 0.0))
    from scipy import signal
    bar = 240.0 / 128.0
    t0, t1 = meta["boundaries"][1] - bar, meta["boundaries"][1]
    i0, i1 = int(t0 * FS), int(t1 * FS)
    hp = signal.sosfiltfilt(signal.butter(4, 600.0, "highpass", fs=FS, output="sos"), x[i0:i1], axis=0)
    y = x.copy()
    y[i0:i1] = hp
    st = _run(y)
    assert len(st["dropouts"]) >= 1
    ends = [d["end"] for d in st["dropouts"]]
    assert min(abs(e - t1) for e in ends) < 0.05
    assert min(abs(b - t1) for b in st["boundaries"]) < bar


def test_snap_grid():
    assert structure._snap_bar(7, 64) == 8
    assert structure._snap_bar(15, 64) == 16
    assert structure._snap_bar(13, 64) == 12
    assert structure._snap_bar(14, 64) == 14


@pytest.mark.parametrize("bpm,chords", [(128.0, ("Am", "F", "C", "G")), (140.0, ("C", "Am", "F", "G"))])
def test_chord_only_boundaries(bpm, chords):
    """Audit 04 F13: sections that differ only in their pad chord (no level step)
    are found on the right downbeat (before the feature floors: 0.5 to 0.75 bar
    late, phase off by 2 or 3 beats)."""
    x, meta = syn.sectioned_track(FS, bpm=bpm, levels_db=(0.0, 0.0, 0.0, 0.0), chords=chords)
    st = _run(x)
    got = np.array(st["boundaries"])
    assert got.size == meta["boundaries"].size
    assert np.max(np.abs(got - meta["boundaries"])) < 0.05


def test_one_bar_fill_boundary_lands_on_the_drop():
    """Audit 04 A9: the boundary after a 1-bar pre-drop fill is the drop (the
    dropout's return), not the fill's start."""
    x, meta = syn.sectioned_track(FS, bpm=128.0, levels_db=(0.0, 0.0, 0.0, 0.0))
    from scipy import signal
    bar = 240.0 / 128.0
    t1 = meta["boundaries"][1]
    i0, i1 = int((t1 - bar) * FS), int(t1 * FS)
    y = x.copy()
    y[i0:i1] = signal.sosfiltfilt(signal.butter(4, 600.0, "highpass", fs=FS, output="sos"), x[i0:i1], axis=0)
    st = _run(y)
    assert min(abs(b - t1) for b in st["boundaries"]) < 0.05
    assert all(abs(b - (t1 - bar)) > 0.5 * bar for b in st["boundaries"])


def test_align_guards_and_uninformative_grid():
    beats = np.arange(16) * 0.5
    out, shift = structure.align_to_onsets(beats, [1.0], 8.0)  # one onset: unchanged
    assert np.array_equal(out, beats) and shift == 0.0
    r = _run(syn.pink_noise(FS * 6, FS, seed=2) * 0.01)
    assert r["meter"] == "4/4 (assumed)"
    assert r["grid_informative"] == (r["grid_kind"] != "default")


def test_snap_from_unrounded_bar():
    assert structure._snap_bar(7.4, 64) == 8
    assert structure._snap_bar(14.6, 64) == 15  # 1.4 bars from 16; rounding first gave 15 -> 16
    assert structure._snap_bar(13.4, 64) == 13
    assert structure._snap_bar(16.2, 64, not_before=17.0) == 17


def test_octave_check_refuses_irregular_or_too_fast():
    """Real-material pass: doubling 120 to 240 from 8th-note bass "kicks" is refused
    (doubled tempo above 180 BPM); an irregular kick train is refused too."""
    beats = np.arange(0, 32) * 0.5          # 120 BPM
    kicks = np.sort(np.r_[beats, beats + 0.25])  # every 8th
    b, t, note = structure.octave_check(beats, 120.0, kicks, 16.0)
    assert note == "kept" and t == 120.0
    beats = np.arange(0, 32) * 0.8          # 75 BPM, doubled would be 150
    rng = np.random.default_rng(0)
    kicks = np.sort(beats + rng.uniform(0, 0.8, beats.size))  # irregular
    assert structure.octave_check(beats, 75.0, kicks, 26.0)[2] == "kept"
    kicks = np.sort(np.r_[beats, beats + 0.4])  # regular, half the period
    assert structure.octave_check(beats, 75.0, kicks, 26.0)[2] == "doubled from kick rate"


def test_dropouts_need_a_steady_low_end():
    """A low end that dips every other beat (half-time, broken) has no fills; a
    steady low end with a one-bar hole has one."""
    beats = np.arange(64) * 0.5
    half = np.where(np.arange(64) % 2 == 0, 0.0, -20.0)
    assert structure.dropouts(half, beats, 0) == []
    steady = np.zeros(64)
    steady[36:40] = -20.0
    d = structure.dropouts(steady, beats, 0)
    assert len(d) == 1 and d[0]["beat_end"] == 40


def _pulses(fs, dur, hat_bpm, kick_bpm):
    """Hats (800 Hz high-passed noise bursts) at hat_bpm and low-passed kicks at kick_bpm."""
    from scipy import signal
    n = int(dur * fs)
    y = np.zeros(n)
    rng = np.random.default_rng(3)
    hp = signal.butter(4, 800, "highpass", fs=fs, output="sos")  # hats and shakers: clap and hat bands
    h = signal.sosfilt(hp, rng.standard_normal(int(0.03 * fs))) * np.exp(-np.arange(int(0.03 * fs)) / (0.008 * fs))
    for t in np.arange(0, dur - 0.05, 60.0 / hat_bpm):
        i = int(t * fs)
        y[i:i + h.size] += 0.2 * h
    # the kick low-passed at 300 Hz: its 1.5 kHz click would put the kick pulse
    # into the clap and hat bands, which here must carry only the hats
    kk = signal.sosfilt(signal.butter(4, 300.0, "lowpass", fs=fs, output="sos"), syn.kick(fs))
    for t in np.arange(0, dur - 0.3, 60.0 / kick_bpm):
        i = int(t * fs)
        y[i:i + kk.size] += kk[: n - i]
    return np.stack([y, y], 1)


def test_metrical_hypothesis_flags_a_4_3_kick():
    """Interim rule (architect, 2026-10-17): hats at 94 BPM and a kick at 4/3 of it
    (Track A's pattern) read 94 with the kick pulse x4/3 and the feel flagged."""
    x = _pulses(FS, 40.0, 94.0, 94.0 * 4 / 3)
    h = structure.metrical_hypothesis(x, FS)
    assert abs(h["tempo_bpm"] - 94.0) < 1.0
    assert h["relation"] == "x4/3" and h["feel"] == "half-time or triplet feel"
    assert h["alternatives"] and all("relation" in a for a in h["alternatives"])


def test_metrical_hypothesis_prefers_the_faster_octave():
    """A pulse train correlates equally at twice its period; 140 must not read 70."""
    x, _ = syn.sectioned_track(FS, bpm=140.0, levels_db=(0.0, -6.0, 0.0, -3.0))
    h = structure.metrical_hypothesis(x, FS)
    assert abs(h["tempo_bpm"] - 140.0) < 1.0 and h["relation"] == "x1" and h["feel"] is None


def _grid_track(bpm, dur, hats, kicks=(), claps=(), ghost=0.0):
    """Hats ({sixteenth in bar: gain}, ghost gain elsewhere), low-passed kicks and
    band-passed claps (sixteenth indices in the bar) on a 16-step grid."""
    from scipy import signal
    n = int(dur * FS)
    y = np.zeros(n)
    rng = np.random.default_rng(3)
    h = signal.sosfilt(signal.butter(4, 800, "highpass", fs=FS, output="sos"),
                       rng.standard_normal(int(0.03 * FS))) * np.exp(-np.arange(int(0.03 * FS)) / (0.008 * FS))
    c = signal.sosfilt(signal.butter(2, (300, 1500), "bandpass", fs=FS, output="sos"),
                       rng.standard_normal(int(0.08 * FS))) * np.exp(-np.arange(int(0.08 * FS)) / (0.02 * FS))
    kk = signal.sosfilt(signal.butter(4, 300.0, "lowpass", fs=FS, output="sos"), syn.kick(FS))
    six = 15.0 / bpm
    for b in range(int(dur / (16 * six))):
        for s in range(16):
            i = int((b * 16 + s) * six * FS)
            g = hats.get(s, ghost)
            if g:
                y[i:i + h.size] += 0.2 * g * h[: n - i]
            if s in kicks:
                y[i:i + kk.size] += kk[: n - i]
            if s in claps:
                y[i:i + c.size] += 0.3 * c[: n - i]
    return np.stack([y, y], 1)


def test_lattice_overrides_a_4_3_pulse():
    """Track D's pattern (maintainer's tempo 120, strongest pulse 160): hats every 3
    sixteenths (3-3-3-3-4 per bar) over a half-time kick and clap. The strongest
    upper pulse is 4/3 (or 2/3) of the beat; every peak sits a whole number of
    sixteenths apart, and the lattice reads 120."""
    for kw, up in ((dict(hats={0: 1, 3: 1, 6: 1, 9: 1, 12: 1}, kicks=(0, 8), claps=(8,)), "x4/3"),
                   (dict(hats={0: 1, 2: .4, 3: 1, 6: 1, 9: 1, 10: .4, 12: 1}, kicks=(0, 8), claps=(4, 12)), "x2/3")):
        h = structure.metrical_hypothesis(_grid_track(120.0, 40.0, **kw), FS)
        assert abs(h["tempo_bpm"] - 120.0) < 0.5 and h["rule"] == "lattice"
        assert h["upper_relation"] == up
        assert h["lattice_score"] >= structure.MH_LATTICE_GAIN * h["pulse_score"]


def test_lattice_keeps_a_plain_pulse():
    """A 3-3-2 accent pattern with ghost sixteenths, and a bare four-on-the-floor
    track (explained just as well by a grid at 3/4 of the rate): the pulse stays."""
    h = structure.metrical_hypothesis(_grid_track(120.0, 40.0, {0: 1, 3: 1, 6: 1, 8: 1, 11: 1, 14: 1},
                                                  (0, 8), (8,), ghost=0.3), FS)
    assert abs(h["tempo_bpm"] - 120.0) < 0.5 and h["rule"] == "pulse"
    for bpm in (124.0, 128.0):
        x, _ = syn.sectioned_track(FS, bpm=bpm, levels_db=(0.0, -6.0, 0.0))
        h = structure.metrical_hypothesis(x, FS)
        assert abs(h["tempo_bpm"] - bpm) < 0.5 and h["rule"] == "pulse"


TRACK_F = dict(hats={0: 1, 3: 1, 6: 1, 8: 1, 11: 1, 14: 1}, kicks=(0, 8), claps=(4, 12))


def test_octave_doubles_a_clap_on_two_and_four():
    """S2 (Track F): at 145 BPM a clap on 2 and 4, a kick every other beat and 3-3-2
    hats repeat every two beats, so every pulse reads 72.5 and its autocorrelation
    is taller there; the peaks an odd number of sixteenths apart at 145 decide."""
    h = structure.metrical_hypothesis(_grid_track(145.0, 40.0, **TRACK_F), FS)
    o = h["octave"]
    assert abs(h["tempo_bpm"] - 145.0) < 0.5 and h["rule"] == "pulse + octave"
    assert o["decision"] == "doubled" and abs(o["from_bpm"] - 72.5) < 0.3
    lo, hi = sorted(o["candidates"], key=lambda c: c["bpm"])
    assert hi["lattice_score"] >= structure.MH_OCTAVE_GAIN * lo["lattice_score"]
    assert lo["upper_acf"] > hi["upper_acf"]  # the evidence that kept 72.5 before


def test_octave_keeps_a_plain_72_5():
    """A kick on every beat with off-beat hats at 72.5: both octaves' sixteenth grids
    explain the same peaks, so the pulse stays."""
    x, _ = syn.sectioned_track(FS, bpm=72.5, levels_db=(0.0, -6.0, 0.0))
    h = structure.metrical_hypothesis(x, FS)
    assert abs(h["tempo_bpm"] - 72.5) < 0.5 and h["rule"] == "pulse"
    assert h["octave"]["decision"] == "kept"


def test_octave_decision_reaches_the_grid_and_spares_the_users_bpm():
    """S3: the beat grid follows the doubled tempo (one beat per 145 BPM beat);
    a user's 72.5 is not second-guessed, and detection is reported against it."""
    x = _grid_track(145.0, 40.0, **TRACK_F)
    st = structure.analyse(x, FS)
    assert abs(st["tempo_bpm"] - 145.0) < 0.5 and st["tempo_octave"] == "doubled from lattice"
    assert abs(60.0 / np.median(np.diff(st["beat_times"])) - 145.0) < 1.0
    st = structure.analyse(x, FS, bpm=72.5)
    assert st["tempo_bpm"] == pytest.approx(72.5) and st["tempo_octave"] == "user"
    assert abs(60.0 / np.median(np.diff(st["beat_times"])) - 72.5) < 0.5
    assert st["metrical"]["detected_relation"] == "x2"


def test_groove_switch_is_a_boundary():
    """Track D's half-time to full-time switch (maintainer, 1:52): same sounds and
    levels, only the clap moves from beat 3 to beats 2 and 4 and the hats from
    quarters to eighths. A boundary within a bar of each switch. On these clean
    fixtures the timbre and level groups already find it (with or without the
    groove group, 0.5 s late: the downbeat vote lands one beat off); the groove
    group's own effect is measured only on Track D (test_feature_real.py)."""
    bpm, bars = 120.0, 24
    dur = bars * 240.0 / bpm
    a = _grid_track(bpm, dur, {0: 1, 4: 1, 8: 1, 12: 1}, (0, 8), (8,))
    b = _grid_track(bpm, dur, {0: 1, 2: 1, 4: 1, 6: 1, 8: 1, 10: 1, 12: 1, 14: 1}, (0, 8), (4, 12))
    x = np.concatenate([a, b, a])
    st = structure.analyse(x, FS)
    starts = np.array([s["start"] for s in st["segments"]])
    for t in (dur, 2 * dur):
        assert np.min(np.abs(starts - t)) <= 240.0 / bpm, (t, starts)


def test_user_bpm_overrides_detection():
    """Spec --bpm: the user's tempo seeds the grid; the detected pulse and its
    deviation are reported; the octave check does not second-guess it."""
    x, m = syn.sectioned_track(FS, bpm=128.0, levels_db=(0.0, -6.0, 0.0))
    h = structure.metrical_hypothesis(x, FS, bpm=64.0)
    assert h["tempo_bpm"] == 64.0 and h["rule"] == "user"
    assert abs(h["detected_bpm"] - 128.0) < 0.5 and h["detected_relation"] == "x2"
    assert abs(h["deviation_pct"] - 100.0) < 1.0
    st = structure.analyse(x, FS, bpm=127.0)
    assert abs(st["tempo_bpm"] - 127.0) < 0.5 and st["tempo_octave"] == "user"
    assert abs(st["metrical"]["deviation_pct"] - 100 * (128.0 / 127.0 - 1)) < 0.5


def test_kick_consistency():
    k = np.arange(0, 20, 60.0 / 124.8)
    c = structure.kick_consistency(k, 94.0, 124.8)
    assert c["consistent"] and c["matches"] == "kick pulse" and abs(c["ratio"] - 124.8 / 94.0) < 0.01
    c = structure.kick_consistency(np.arange(0, 20, 1.0), 120.0)
    assert c["consistent"] and c["matches"] == "tempo half"
    c = structure.kick_consistency(np.cumsum(np.full(30, 0.83)), 120.0, 95.0)
    assert not c["consistent"]
