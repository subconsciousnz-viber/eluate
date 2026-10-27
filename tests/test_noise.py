"""noise.py: the BS.468 curve, the quiet-passage floor and the exposure assessment."""
import numpy as np
import pytest

from eluate import noise

FS = 48000


def test_weighting_468_reference_points():
    w = noise.weighting_468_db([1000.0, 6300.0, 31.5, 20000.0])
    assert w[0] == pytest.approx(0.0, abs=0.1)
    assert w[1] == pytest.approx(12.2, abs=0.1)
    assert w[2] == pytest.approx(-29.9, abs=0.2)
    assert w[3] == pytest.approx(-22.2, abs=0.3)


def _track(noise_db, seconds=20, quiet_every=4.0, quiet_s=1.0, seed=3):
    """Loud tone-and-noise music with a quiet gap of only noise every quiet_every seconds."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    music = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.1 * rng.standard_normal(n)
    gate = ((t % quiet_every) >= quiet_s).astype(float)
    hiss = rng.standard_normal((n, 2)) * 10 ** (noise_db / 20)
    return (music * gate)[:, None] + hiss


def test_floor_reads_the_noise_in_the_gaps_not_the_music():
    x = _track(-80.0)
    alone = noise.floor(x[: int(1.0 * FS)], FS, quiet=np.ones(10_000, bool))   # a gap only
    got = noise.floor(x, FS)
    assert got["quiet_s"] > 3.0 and len(got["quiet_spans"]) == 5
    assert got["db"] == pytest.approx(alone["db"], abs=1.5)
    assert got["db"] < -60.0


def test_a_full_scale_sine_reads_its_weighting():
    t = np.arange(4 * FS) / FS
    s = np.sin(2 * np.pi * 4000.0 * t)
    got = noise.floor(np.stack([s, s], 1), FS, quiet=np.ones(10_000, bool))
    assert got["db"] == pytest.approx(float(noise.weighting_468_db(4000.0)), abs=0.5)


def test_master_gain_exposes_the_floor():
    x = _track(-80.0)
    before = noise.floor(x, FS)
    after = noise.floor(x * 10 ** (30 / 20), FS, quiet=before["quiet"])
    a = noise.assess(before, after)
    assert a["status"] == "exposed"
    assert a["after_db"] - a["before_db"] == pytest.approx(30.0, abs=0.1)
    assert noise.assess(before, before)["status"] is None
    loud = noise.floor(x * 10 ** (45 / 20), FS, quiet=before["quiet"])
    assert noise.assess(loud, loud)["status"] == "present"


def test_no_quiet_passage_means_no_reading():
    x = _track(-80.0, quiet_s=0.0)                  # the music never stops
    got = noise.floor(x, FS)
    assert got["db"] == float("-inf") and got["quiet_spans"] == []
    assert noise.assess(got, got)["status"] is None


def test_a_rate_with_no_band_above_2k_reads_nothing():
    rng = np.random.default_rng(0)
    x = np.concatenate([0.5 * rng.standard_normal(8000), 1e-4 * rng.standard_normal(8000)])
    got = noise.floor(x, 4000)                         # 0.45 fs = 1.8 kHz < 2 kHz
    assert got["bands_hz"] == [] and got["db"] == float("-inf")
    assert noise.assess(got, got)["status"] is None


def test_a_borrowed_mask_is_padded_to_a_longer_signal():
    """Report 12b: the master read on a shorter premaster's mask (a 5 s master on a 4 s
    premaster) raised IndexError; frames past the mask's end are not quiet."""
    pre = _track(-80.0, seconds=4)
    before = noise.floor(pre, FS)
    master = np.concatenate([pre, pre[: FS]])
    after = noise.floor(master, FS, quiet=before["quiet"])
    assert len(after["quiet"]) > len(before["quiet"])
    assert np.array_equal(after["quiet"][: len(before["quiet"])], before["quiet"])
    assert not after["quiet"][len(before["quiet"]):].any()
    assert after["db"] == pytest.approx(before["db"], abs=0.01)
