"""Gates for bands.py (spec 3.1, section 7, draft 0.3)."""

import warnings

import numpy as np
import pytest
from scipy import signal

from eluate import bands

FS_LIST = [44100, 48000, 96000]


def _dtft(h: np.ndarray, f: float, fs: float) -> complex:
    n = np.arange(h.shape[0])
    return complex(np.sum(h * np.exp(-2j * np.pi * f * n / fs)))


def _impulse(n, centre):
    x = np.zeros(n)
    x[n // 2 if centre else 0] = 1.0
    return x


@pytest.mark.parametrize("fs", FS_LIST)
@pytest.mark.parametrize("edge", ["zero", "odd"])
def test_zero_mode_sum_equals_input(fs, edge):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((fs * 3, 2)) * 0.5
    b = bands.split(x, fs, edge=edge)
    assert b.shape == (5, x.shape[0], 2)
    err = np.max(np.abs(b.sum(axis=0) - x))
    assert err < 1e-9, err


@pytest.mark.parametrize("fs", FS_LIST)
@pytest.mark.parametrize("phase", ["zero", "minimum"])
def test_magnitude_at_each_crossover_isolated(fs, phase):
    """Each crossover's two outputs are -6.02 +- 0.05 dB at fc."""
    x = _impulse(1 << 17, phase == "zero")
    for fc in bands.DEFAULT_CROSSOVERS:
        b = bands.split(x, fs, [fc], phase=phase)
        for band in (0, 1):
            mag_db = 20 * np.log10(abs(_dtft(b[band], fc, fs)))
            assert abs(mag_db - (-6.0206)) <= 0.05, (fc, band, mag_db)


@pytest.mark.parametrize("fs", FS_LIST)
@pytest.mark.parametrize("phase", ["zero", "minimum"])
@pytest.mark.parametrize("order,lo_tol,hi_tol", [(2, 0.07, 0.0), (4, 0.05, 0.05)])
def test_magnitude_at_each_crossover_in_cascade(fs, phase, order, lo_tol, hi_tol):
    """Order 2 in the 5-band cascade: -6.02 +0.00 / -0.07 dB (neighbouring
    crossovers 2 octaves away can only lower the gain at fc). Order 4: +-0.05."""
    x = _impulse(1 << 17, phase == "zero")
    b = bands.split(x, fs, bands.DEFAULT_CROSSOVERS, order=order, phase=phase)
    for k, fc in enumerate(bands.DEFAULT_CROSSOVERS):
        for band in (k, k + 1):  # the two bands meeting at fc
            mag_db = 20 * np.log10(abs(_dtft(b[band], fc, fs)))
            dev = mag_db - (-6.0206)
            assert -lo_tol - 1e-4 <= dev <= hi_tol + 1e-4, (fc, band, mag_db)


@pytest.mark.parametrize("fs", FS_LIST)
@pytest.mark.parametrize("phase", ["minimum", ["minimum", "zero", "zero", "zero"]])
def test_minimum_mode_sum_is_flat(fs, phase):
    x = _impulse(1 << 17, phase != "minimum")
    s = bands.split(x, fs, phase=phase).sum(axis=0)
    mag = np.abs(np.fft.rfft(s))
    f = np.fft.rfftfreq(s.shape[0], 1 / fs)
    sel = (f >= 10) & (f <= 0.45 * fs)
    dev = np.max(np.abs(20 * np.log10(mag[sel])))
    assert dev <= 0.05, dev


def test_minimum_mode_band_sum_is_the_allpass_product():
    fs = 48000
    x = _impulse(1 << 16, False)
    s = bands.split(x, fs, phase="minimum").sum(axis=0)
    ref = x.copy()
    for fc in bands.DEFAULT_CROSSOVERS:
        lp = signal.butter(2, fc, fs=fs, output="sos")
        ref = signal.sosfilt(bands.allpass_sos(lp), ref)
    assert np.max(np.abs(s - ref)) < 1e-12


@pytest.mark.parametrize("order", [2, 4, 6])
def test_modes_are_magnitude_identical_per_band(order):
    fs = 48000
    bz = bands.split(_impulse(1 << 17, True), fs, order=order, phase="zero")
    bm = bands.split(_impulse(1 << 17, False), fs, order=order, phase="minimum")
    for k in range(bz.shape[0]):
        mz = np.abs(np.fft.rfft(bz[k]))
        mm = np.abs(np.fft.rfft(bm[k]))
        assert np.max(np.abs(mz - mm)) < 1e-6, (order, k)


# ------------------------------------------------ edges (spec 7, draft 0.3)

FC_EDGE = 60.0


def _edge_stimulus(fs, seconds=2.0, seed=5):
    """Cosine plus mixed-phase sines, strong content around fc = 60 Hz."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    x = 0.3 * np.cos(2 * np.pi * 45 * t)
    for f, a in ((30, 0.2), (60, 0.15), (170, 0.1), (1000, 0.1), (5000, 0.05)):
        x = x + a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return x


def _zero_padded_reference(x, fs, pad_s=4.0):
    pad = int(pad_s * fs)
    z = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    return bands.split(z, fs, [FC_EDGE])[:, pad:pad + x.size]


def _edge_error_db(x, fs, edge):
    b = bands.split(x, fs, [FC_EDGE], edge=edge)
    return 20 * np.log10(np.max(np.abs(b - _zero_padded_reference(x, fs))))


@pytest.mark.parametrize("fs", FS_LIST)
@pytest.mark.parametrize("edge", ["zero", "odd"])
def test_edges_with_100ms_silence(fs, edge):
    """>= 100 ms digital silence at both ends: <= -100 dB vs zero-padded reference."""
    ns = int(0.1 * fs)
    x = np.concatenate([np.zeros(ns), _edge_stimulus(fs), np.zeros(ns)])
    assert _edge_error_db(x, fs, edge) <= -100


@pytest.mark.parametrize("fs", FS_LIST)
def test_edges_with_50ms_linear_fade(fs):
    """50 ms linear fade at both ends: gate -70 dB (met by edge='zero').

    edge='odd' is recorded, not gated: it measures about -52 dB on this
    stimulus (audit/brief-02.md D.9)."""
    x = _edge_stimulus(fs)
    nf = int(0.05 * fs)
    w = np.ones(x.size)
    w[:nf] = np.linspace(0, 1, nf)
    w[-nf:] = np.linspace(1, 0, nf)
    x = x * w
    err_zero = _edge_error_db(x, fs, "zero")
    err_odd = _edge_error_db(x, fs, "odd")
    print(f"fs={fs} 50 ms fade: zero {err_zero:.1f} dB, odd {err_odd:.1f} dB")
    assert err_zero <= -70
    assert err_odd > -70  # documents why 'zero' is the default


@pytest.mark.parametrize("edge", ["zero", "odd"])
def test_abrupt_full_scale_cosine_start(edge):
    """Abrupt full-scale cosine at fc = 60 Hz: finite, deterministic, band sum
    exact; the low band's deviation from steady state near the start is the
    zero-phase filter's inherent pre-ring and is only recorded."""
    fs = 48000
    t = np.arange(2 * fs) / fs
    x = np.cos(2 * np.pi * 45 * t)
    b1 = bands.split(x, fs, [FC_EDGE], edge=edge)
    b2 = bands.split(x, fs, [FC_EDGE], edge=edge)
    assert np.all(np.isfinite(b1)) and np.array_equal(b1, b2)
    assert np.max(np.abs(b1.sum(axis=0) - x)) < 1e-12
    lp = signal.butter(2, FC_EDGE, fs=fs, output="sos")
    steady = abs(signal.sosfreqz(lp, [45.0], fs=fs)[1][0]) ** 2 * x
    dev = 20 * np.log10(np.max(np.abs(b1[0, : int(0.01 * fs)] - steady[: int(0.01 * fs)])))
    print(f"abrupt cosine, edge={edge}: low-band deviation in first 10 ms {dev:.1f} dB")


def test_odd_padlen_cap_warns_and_short_input():
    assert bands.padlen(10**6, 48000, 100.0) == max(512, int(np.ceil(20 * 48000 / (2 * np.pi * 100))))
    with pytest.warns(RuntimeWarning):
        assert bands.padlen(100, 48000, 100.0) == 99
    x = np.random.default_rng(0).standard_normal((300, 2))
    with pytest.warns(RuntimeWarning):
        b = bands.split(x, 48000, edge="odd")
    assert np.max(np.abs(b.sum(axis=0) - x)) < 1e-9
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        b = bands.split(x, 48000)  # zero padding is never capped
    assert np.max(np.abs(b.sum(axis=0) - x)) < 1e-9


def test_ms_roundtrip_and_energy():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((10000, 2))
    ms = bands.to_ms(x)
    assert np.max(np.abs(bands.to_lr(ms) - x)) < 1e-12
    assert np.isclose(np.sum(ms**2), np.sum(x**2), rtol=1e-12)
    mono = np.stack([x[:, 0], x[:, 0]], axis=1)
    ms_mono = bands.to_ms(mono)
    assert np.max(np.abs(ms_mono[:, 1])) == 0.0
    assert np.allclose(ms_mono[:, 0], np.sqrt(2) * x[:, 0])


def test_rejects_bad_arguments():
    x = np.zeros((1000, 2))
    with pytest.raises(ValueError):
        bands.split(x, 48000, [500, 100])
    with pytest.raises(ValueError):
        bands.split(x, 48000, [100, 30000])
    with pytest.raises(ValueError):
        bands.split(x, 48000, [100], order=3, phase="minimum")
    with pytest.raises(ValueError):
        bands.split(x, 48000, [100, 200], phase=["zero"])
    with pytest.raises(ValueError):
        bands.split(x, 48000, edge="even")
    with pytest.raises(ValueError):
        bands.split(np.zeros((1, 2)), 48000)
    bad = x.copy()
    bad[5, 1] = np.inf
    with pytest.raises(ValueError):
        bands.split(bad, 48000)
