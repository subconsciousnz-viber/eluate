"""Real-material gates on the maintainer's premasters, named by label (Tracks A to F; the
files are private and never committed). A TOML manifest maps each label to its file:
fixtures/premasters/manifest.toml (gitignored), or the one the environment variable
ELUATE_PREMASTERS names; fixtures/premasters/manifest.example.toml shows the form.
A label's value is a path or a list of paths, the first that exists used, relative
paths from the manifest's folder (the maintainer's lists a local copy first, then his
drive). Read in place, never written; skipped where the manifest or a file is absent.

Ground truth from the maintainer (via the architect, 2026-10-17): tempos Track D 120,
Track E 95, Track B 120, Track A 94, Track C 88; Track D switches from half-time to
full-time at about 1:52 and back at about 4:46. Kick patterns are unknown for all
five. Track F (a sixth VA premaster) is 145 (maintainer, 2026-10-21)."""
import os
import tomllib
from pathlib import Path

import numpy as np
import pytest

from eluate import io
from eluate.features import structure

MANIFEST = Path(os.environ.get("ELUATE_PREMASTERS")
                or Path(__file__).resolve().parents[1] / "fixtures" / "premasters" / "manifest.toml")


def _files(manifest):
    """{label: the first listed path that exists, else None}; {} without a manifest."""
    if not manifest.is_file():
        return {}
    with open(manifest, "rb") as fh:
        tracks = tomllib.load(fh).get("tracks", {})
    out = {}
    for label, v in tracks.items():
        paths = [Path(p).expanduser() for p in ([v] if isinstance(v, str) else v)]
        paths = [p if p.is_absolute() else manifest.parent / p for p in paths]
        out[label] = next((str(p) for p in paths if p.is_file()), None)
    return out


FILES = _files(MANIFEST)
TRACKS = {"A": 94.0, "B": 120.0, "C": 88.0, "D": 120.0, "E": 95.0}  # the maintainer's tempos
SWITCHES_S = (112.0, 286.0)  # Track D: half-time -> full-time -> half-time
SWITCH_TOL_S = 4.0  # two bars at 120 BPM ("about" 1:52 and 4:46)
SIMPLE = ("x1", "x2", "/2", "x4/3", "x3/4", "x3/2", "x2/3")

present = pytest.mark.skipif(not all(FILES.get(k) for k in TRACKS),
                             reason=f"Tracks A to E are not all found through {MANIFEST}")
# Track F, 145 BPM (maintainer, 2026-10-21): its clap on 2 and 4 read 72.5 before the octave
# decision (sprint S2); the test the five above pass by an octave relation is not enough
TRACK_F_BPM = 145.0


@pytest.fixture(scope="module")
def track_d():
    from eluate import features
    x, fs = io.read(FILES["D"])
    return features.analyse(x, fs)


@present
@pytest.mark.parametrize("label", list(TRACKS))
def test_tempo_against_the_owner(label):
    """Detected within 2 % of the maintainer's tempo, or the maintainer's tempo among the
    alternatives with the chosen one a simple ratio of it (spec 7)."""
    bpm = TRACKS[label]
    x, fs = io.read(FILES[label])
    h = structure.metrical_hypothesis(x, fs)
    if abs(h["tempo_bpm"] / bpm - 1.0) <= 0.02:
        return
    assert any(abs(a["bpm"] / bpm - 1.0) <= 0.02 for a in h["alternatives"])
    assert structure._relation(h["tempo_bpm"], bpm) in SIMPLE


@pytest.mark.skipif(not FILES.get("F"), reason=f"Track F is not found through {MANIFEST}")
def test_track_f_tempo_is_the_faster_octave():
    """Detected within 2 % of 145 itself, not at 72.5 (audit 13, Sol #9)."""
    bpm = TRACK_F_BPM
    x, fs = io.read(FILES["F"])
    h = structure.metrical_hypothesis(x, fs)
    assert abs(h["tempo_bpm"] / bpm - 1.0) <= 0.02, (h["tempo_bpm"], h["octave"])


@present
def test_user_bpm_reports_the_deviation():
    bpm = TRACKS["C"]
    x, fs = io.read(FILES["C"])
    h = structure.metrical_hypothesis(x, fs, bpm=bpm)
    assert h["tempo_bpm"] == bpm and h["rule"] == "user"
    assert np.isfinite(h["deviation_pct"]) and h["detected_relation"] is not None


@present
@pytest.mark.slow
def test_track_d_groove_switches_are_section_boundaries(track_d):
    starts = np.array([s["start"] for s in track_d["structure"]["segments"]])
    for t in SWITCHES_S:
        assert np.min(np.abs(starts - t)) <= SWITCH_TOL_S, t


@present
@pytest.mark.xfail(strict=True, reason="kick detection on dense real mixes is unverified (brief-04 D.36): "
                                       "the detector finds 0.1 to 0.4 kicks per beat here")
@pytest.mark.slow
def test_track_d_kick_rate_doubles_across_the_switch(track_d):
    k = np.asarray(track_d["transients"]["kick_onsets"])
    dur = track_d["meta"]["duration_s"]
    edges = (0.0,) + SWITCHES_S + (dur,)
    rate = [np.sum((k >= a) & (k < b)) / (b - a) for a, b in zip(edges[:-1], edges[1:])]
    assert abs(rate[1] / rate[0] - 2.0) < 0.3 and abs(rate[1] / rate[2] - 2.0) < 0.3
