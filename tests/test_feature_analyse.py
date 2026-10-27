"""features.analyse: common grid, plan inputs, determinism, input handling."""

import numpy as np
import pytest

from fixtures import synthetic as syn
from eluate.features import analyse

FS = 44100


@pytest.fixture(scope="module")
def track():
    x, meta = syn.sectioned_track(FS, levels_db=(-12.0, 0.0, -6.0, 0.0))
    y, ring = syn.add_ring(x, FS, f0=3000.0, q=10.0, gain_db=9.0, start_s=15.0, end_s=45.0)
    return y, meta


@pytest.fixture(scope="module")
def result(track):
    return analyse(track[0], FS)


def test_common_grid(result, track):
    n = track[0].shape[0]
    k = int(np.ceil(n / (0.1 * FS)))
    t = result["t"]
    assert t.size == k and np.allclose(t, (np.arange(k) + 0.5) * 0.1)
    for arr in (result["loudness"]["momentary"], result["loudness"]["short_term"],
                result["transients"]["onset_strength"], result["tilt"]["centroid_hz"]):
        assert arr.shape == (k,)
    assert result["crest"]["crest_db"].shape == (5, 3, k)
    assert result["stereo"]["corr"].shape == (5, k)
    assert result["pitch"]["chroma"].shape == (12, k)
    for r in result["resonance"]["resonances"]:
        assert r["excess_db"].shape == (k,)


def test_plan_inputs(result, track):
    _, meta = track
    pi = result["plan_inputs"]
    assert np.isfinite(pi["s95_lufs"]) and pi["s95_lufs"] > pi["integrated_lufs"]
    assert [round(s["start"], 1) for s in pi["sections"]][1:] == list(np.round(meta["boundaries"], 1))
    loud = [s["integrated_lufs"] for s in pi["sections"]]
    assert abs((loud[1] - loud[0]) - 12.0) < 0.5 and abs((loud[1] - loud[2]) - 6.0) < 0.5
    assert 60.0 <= pi["side_strip_hz"] <= 150.0
    assert pi["crest_p95_db"].shape == (5, 3)
    assert np.all(pi["band_decoherence"][np.isfinite(pi["band_decoherence"])] < 1e-6)  # mono fixture
    notch = [n for n in pi["notches"] if abs(n["f0"] / 3000 - 1) < 0.02]
    assert len(notch) == 1 and abs(notch[0]["start"] - 15) < 0.3 and abs(notch[0]["end"] - 45) < 0.3
    assert abs(pi["tempo_bpm"] - 128.0) < 0.5
    assert pi["edges"]["abrupt_start"]  # the fixture starts on a kick at full level
    assert pi["bass_root_midi"].size == pi["beat_times"].size
    assert np.all(pi["bass_root_midi"][np.isfinite(pi["bass_root_midi"])] == 33)  # A1, 55 Hz


@pytest.mark.slow
def test_deterministic(track):
    y = track[0][: FS * 20]
    a, b = analyse(y, FS), analyse(y, FS)
    for key in ("momentary", "short_term"):
        assert np.array_equal(a["loudness"][key], b["loudness"][key])
    assert np.array_equal(a["crest"]["crest_db"], b["crest"]["crest_db"], equal_nan=True)
    assert np.array_equal(a["structure"]["beat_times"], b["structure"]["beat_times"])
    assert np.array_equal(a["pitch"]["bass"]["root_midi"], b["pitch"]["bass"]["root_midi"], equal_nan=True)
    assert len(a["resonance"]["resonances"]) == len(b["resonance"]["resonances"])
    for ra, rb in zip(a["resonance"]["resonances"], b["resonance"]["resonances"]):
        assert ra["f0"] == rb["f0"] and np.array_equal(ra["excess_db"], rb["excess_db"])


def test_mono_input_and_rejects_non_finite():
    x = syn.pink_noise(FS * 4, FS, seed=1)[:, 0]
    r = analyse(x, FS)
    assert r["meta"]["n"] == x.size
    assert abs(r["stereo"]["full_corr_summary"] - 1.0) < 1e-9
    bad = np.ones((FS, 2))
    bad[10, 0] = np.nan
    with pytest.raises(ValueError):
        analyse(bad, FS)


def test_short_and_silent_input():
    r = analyse(np.zeros((FS * 2, 2)), FS)
    assert r["resonance"]["resonances"] == []
    assert np.isnan(r["plan_inputs"]["s95_lufs"])
    assert np.all(np.isneginf(r["loudness"]["momentary"]))
