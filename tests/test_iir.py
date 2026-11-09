"""iir.py: blocked IIR filtering with the subnormal flush (filter-maths audit F5)."""

import numpy as np
import pytest
from scipy import signal

from fixtures import synthetic as syn
from eluate import bands, iir
from eluate.stages import notch

FS = 48000


def _music(n, seed=0, channels=2):
    """Music-like: pink noise plus a few sustained partials, no digital silence."""
    x = syn.pink_noise(n, FS, rms_dbfs=-20.0, seed=seed, channels=channels)
    if channels == 1 and x.ndim == 2:
        x = x[:, 0]
    t = np.arange(n) / FS
    tone = 0.1 * np.sin(2 * np.pi * 55.0 * t) + 0.05 * np.sin(2 * np.pi * 440.0 * t + 0.3)
    return x + (tone[:, None] if x.ndim == 2 else tone)


def _sos_set():
    return [signal.butter(2, 100.0, "lowpass", fs=FS, output="sos"),
            signal.butter(2, 6400.0, "lowpass", fs=FS, output="sos"),
            notch.bandpass_sos(1000.0, 4.0, FS),
            np.vstack([signal.butter(2, 300.0, "highpass", fs=FS, output="sos"),
                       signal.butter(2, 3000.0, "lowpass", fs=FS, output="sos")])]


def _scipy_zero(sos, x, pl):
    xp = np.pad(x, [(pl, pl)] + [(0, 0)] * (x.ndim - 1))
    return signal.sosfiltfilt(sos, xp, axis=0, padtype=None)[pl:pl + x.shape[0]]


@pytest.mark.parametrize("k", range(4))
@pytest.mark.parametrize("channels", [1, 2])
def test_zero_phase_is_scipy_bit_for_bit_on_music(k, channels):
    """Without digital silence no flush happens: identical to scipy's sosfiltfilt."""
    sos = _sos_set()[k]
    x = _music(3 * FS + 17, seed=k, channels=channels)
    pl = 2048
    for block in (iir.BLOCK, 5000, 1 << 20):
        y = iir.sosfiltfilt_zero(sos, x, pl, block=block)
        assert np.array_equal(y, _scipy_zero(sos, x, pl))


@pytest.mark.parametrize("k", range(4))
def test_zero_phase_with_silence_within_1e12_and_silence_exact(k):
    """5 s of digital silence either side of 4 s of music: the flush moves the output by
    less than 1e-12 and the outer silence is exactly 0 (scipy leaves subnormal tails)."""
    sos = _sos_set()[k]
    sil = np.zeros(5 * FS)
    x = np.concatenate([sil, _music(4 * FS, seed=k, channels=1), sil])
    pl = 4096
    y = iir.sosfiltfilt_zero(sos, x, pl)
    ref = _scipy_zero(sos, x, pl)
    assert np.max(np.abs(y - ref)) < 1e-12
    assert np.all(y[:FS] == 0.0) and np.all(y[-FS:] == 0.0)
    if k == 0:      # the 100 Hz pole is still subnormal there in scipy (faster ones underflow)
        assert np.any(ref[-FS:] != 0.0) and np.max(np.abs(ref[-FS:])) < 1e-300


def test_causal_sosfilt_matches_scipy():
    sos = signal.butter(2, 25.0, "highpass", fs=FS, output="sos")
    x = _music(2 * FS, seed=3)
    y, zf = iir.sosfilt(sos, x, block=7000)
    ref, zref = signal.sosfilt(sos, x, axis=0, zi=np.zeros((1, 2, 2)))
    assert np.array_equal(y, ref) and np.array_equal(zf, zref)
    assert np.array_equal(y, signal.sosfilt(sos, x, axis=0))
    # with a silent tail: within 1e-12, and the state ends flushed to exactly 0
    xs = np.concatenate([x, np.zeros((6 * FS, 2))])
    y2, z2 = iir.sosfilt(sos, xs)
    ref2 = signal.sosfilt(sos, xs, axis=0)
    assert np.max(np.abs(y2 - ref2)) < 1e-12
    assert not np.any(z2) and np.all(y2[-FS:] == 0.0)
    # a given state is carried: two halves equal one call
    h = xs.shape[0] // 2
    a, za = iir.sosfilt(sos, xs[:h])
    b, _ = iir.sosfilt(sos, xs[h:], zi=za)
    assert np.max(np.abs(np.concatenate([a, b]) - ref2)) < 1e-12


def test_all_silence_and_short_inputs():
    sos = signal.butter(2, 100.0, "lowpass", fs=FS, output="sos")
    assert np.array_equal(iir.sosfiltfilt_zero(sos, np.zeros((1000, 2)), 512),
                          np.zeros((1000, 2)))
    for n in (1, 2, 3, 100):
        x = np.random.default_rng(n).standard_normal(n)
        assert np.array_equal(iir.sosfiltfilt_zero(sos, x, 512), _scipy_zero(sos, x, 512))


def test_split_uses_it_and_stays_exact():
    """bands.split (zero phase): the band sum is still x, and a track in silence gives
    bands that are exactly 0 in the outer silence."""
    sil = np.zeros((3 * FS, 2))
    x = np.concatenate([sil, _music(2 * FS, seed=5), sil])
    b = bands.split(x, FS, bands.DEFAULT_CROSSOVERS)
    assert np.max(np.abs(b.sum(axis=0) - x)) < 1e-15
    assert np.all(b[:, -FS:] == 0.0) and np.all(b[:, :FS] == 0.0)
