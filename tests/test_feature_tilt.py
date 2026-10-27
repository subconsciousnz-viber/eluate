"""Spec 7 tilt gate and the tilt.py contract."""

import numpy as np
import pytest

from fixtures import synthetic as syn
from eluate.features import tilt

FS = 48000


@pytest.mark.parametrize("fs", [44100, 48000, 96000])
def test_pink_noise_slope(fs):
    """Gate: pink noise reads -3.01 +- 0.1 dB/oct (mean PSD per 1/3-octave band)."""
    x = syn.pink_noise(fs * 30, fs, seed=1)
    r = tilt.analyse(x, fs)
    for v in ("M", "S", "stereo"):
        assert abs(r["slope_db_oct"][v] - (-3.0103)) < 0.1, (v, r["slope_db_oct"][v])


@pytest.mark.parametrize("slope", [-4.5, -6.0, 0.0])
def test_synthetic_slope_recovered(slope):
    """Gate: a synthetic -4.5 dB/oct is recovered within 0.1 (plus -6 and flat)."""
    x = syn.noise_psd_slope(FS * 30, FS, slope, seed=2)
    r = tilt.analyse(x, FS)
    assert abs(r["slope_db_oct"]["stereo"] - slope) < 0.1
    # the sub shelf of a pure power law sits on the extrapolated line
    assert abs(r["sub_shelf_db"]["stereo"]) < 0.5


def test_band_mean_not_band_sum():
    """Mean PSD per band: white noise is flat (0 dB/oct); band-summed power would
    read +3.01 dB/oct because 1/3-octave bands widen with frequency (audit F5)."""
    x = syn.noise_psd_slope(FS * 20, FS, 0.0, seed=3)
    r = tilt.analyse(x, FS)
    assert abs(r["slope_db_oct"]["stereo"]) < 0.1
    w = r["centres"] * (2 ** (1 / 6) - 2 ** (-1 / 6))
    summed = r["ltas_db"]["stereo"] + 10 * np.log10(w)
    s, _ = tilt.fit_slope(r["centres"], summed)
    assert abs(s - 3.0103) < 0.1


def test_sub_shelf_reported_separately():
    """A +6 dB shelf below 90 Hz leaves the 100 Hz-10 kHz slope alone and shows up
    in sub_shelf_db."""
    from scipy import signal
    x = syn.pink_noise(FS * 30, FS, seed=4)
    lo = signal.sosfiltfilt(signal.butter(4, 70, fs=FS, output="sos"), x, axis=0)
    y = x + (10 ** (6 / 20) - 1) * lo
    r0, r1 = tilt.analyse(x, FS), tilt.analyse(y, FS)
    assert abs(r1["slope_db_oct"]["stereo"] - r0["slope_db_oct"]["stereo"]) < 0.1
    assert r1["sub_shelf_db"]["stereo"] - r0["sub_shelf_db"]["stereo"] > 3.0


def test_views_and_peak_hold():
    """M/S/stereo: a hard-left signal has M = S; peak-hold >= average everywhere."""
    x = syn.pink_noise(FS * 10, FS, seed=5)
    x[:, 1] = 0.0
    r = tilt.analyse(x, FS)
    assert np.allclose(r["ltas_db"]["M"], r["ltas_db"]["S"], atol=1e-6)
    for v in ("M", "S", "stereo"):
        assert np.all(r["peak_hold_db"][v] >= r["ltas_db"][v] - 1e-9)


def test_band_defined_below_one_bin():
    """At 96 kHz the 20 Hz band is narrower than an 8192-point bin... it is still a
    finite mean (fractional bin overlap)."""
    x = syn.pink_noise(96000 * 5, 96000, seed=6)
    r = tilt.analyse(x, 96000)
    assert np.all(np.isfinite(r["ltas_db"]["stereo"]))
    assert r["centres"][0] < 20.0


def test_series_and_flatness():
    x = syn.pink_noise(FS * 10, FS, seed=7)
    r = tilt.analyse(x, FS)
    assert r["centroid_hz"].shape == r["t"].shape == r["slope_series"].shape
    assert np.nanmax(np.abs(r["slope_series"] + 3.01)) < 0.5
    assert np.all((r["flatness"] > 0.95) & (r["flatness"] <= 1.0))
    t = np.arange(FS * 10) / FS
    tone = 0.3 * np.sin(2 * np.pi * 1000 * t)
    r2 = tilt.analyse(np.stack([tone, tone], 1) + 1e-4 * x, FS)
    assert r2["flatness"][2] < 0.1  # 500-2000 Hz band holds a pure tone


def test_silence_is_finite():
    r = tilt.analyse(np.zeros((FS * 2, 2)), FS)
    assert np.all(r["ltas_db"]["stereo"] == -200.0)
    assert np.all(np.isnan(r["centroid_hz"]))


def _kellet_pink(n, seed=0):
    """Paul Kellet's refined pink filter (music-dsp, 1999; +-0.05 dB above 9.2 Hz
    at 44.1 kHz) on white Gaussian noise: independent of fixtures.synthetic's
    FFT-shaped generator, so the tilt gate is not circular (audit 04 F19)."""
    from scipy import signal
    w = np.random.default_rng(seed).standard_normal(n)
    poles = (0.99886, 0.99332, 0.96900, 0.86650, 0.55000, -0.7616)
    gains = (0.0555179, 0.0750759, 0.1538520, 0.3104856, 0.5329522, -0.0168980)
    y = sum(signal.lfilter([g], [1.0, -p], w) for p, g in zip(poles, gains))
    y = y + 0.5362 * w + 0.115926 * np.concatenate([[0.0], w[:-1]])
    return y / np.sqrt(np.mean(y ** 2)) * 0.1


def test_tilt_on_independent_pink():
    fs = 44100
    y = _kellet_pink(fs * 30)
    r = tilt.analyse(np.stack([y, y], 1), fs)
    assert abs(r["slope_db_oct"]["M"] - (-3.0103)) < 0.1, r["slope_db_oct"]["M"]


@pytest.mark.parametrize("fs", [44100, 96000])
def test_fine_display_spectrum(fs):
    """The 1/24-octave display LTAS: 20 Hz to under fs/2, finite, and on pink noise
    its own fit reads the same slope as the third-octave LTAS the tilt uses."""
    x = syn.pink_noise(fs * 20, fs, seed=4)
    r = tilt.analyse(x, fs)
    fc = r["fine_centres"]
    assert fc[0] == pytest.approx(20.0)
    assert np.allclose(np.diff(np.log2(fc)), 1 / 24)
    assert fc[-1] * 2 ** (1 / 12) <= 0.95 * fs / 2 + 1e-9
    assert fc.size > 200
    for v in ("M", "S", "stereo"):
        assert r["fine_db"][v].shape == fc.shape and np.all(np.isfinite(r["fine_db"][v]))
    s, _ = tilt.fit_slope(fc, r["fine_db"]["stereo"])
    assert abs(s - r["slope_db_oct"]["stereo"]) < 0.1
