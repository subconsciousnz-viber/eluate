"""Sanity checks for fixtures/synthetic.py generators."""

import numpy as np
import pytest
from scipy import signal

from fixtures import synthetic as syn

FS = 48000


def _third_octave_psd_slope(x, fs, f_lo, f_hi):
    """OLS slope (dB/oct) of mean PSD per 1/3-octave band (spec 3.3 convention)."""
    f, p = signal.welch(x, fs, nperseg=1 << 16)
    centres = 1000.0 * 2.0 ** (np.arange(-30, 14) / 3.0)
    centres = centres[(centres >= f_lo) & (centres <= f_hi)]
    lvl = [10 * np.log10(p[(f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))].mean()) for c in centres]
    return np.polyfit(np.log2(centres), lvl, 1)[0]


@pytest.mark.parametrize("slope", [-3.0103, -4.5, 0.0])
def test_noise_psd_slope(slope):
    """Slope recovered within 0.1 dB/oct from 20 Hz to 16 kHz and within 0.1 dB/oct
    from 100 Hz to 10 kHz (the section 7 tilt gate)."""
    x = syn.noise_psd_slope(FS * 30, FS, slope, rms_dbfs=-20, seed=7)
    assert x.shape == (FS * 30, 2)
    assert abs(10 * np.log10(np.mean(x**2)) - (-20)) < 1e-9
    for c in range(2):
        assert abs(_third_octave_psd_slope(x[:, c], FS, 100, 10000) - slope) < 0.1
        assert abs(_third_octave_psd_slope(x[:, c], FS, 20, 16000) - slope) < 0.1


def test_noise_has_nothing_below_the_10hz_corner():
    n = FS * 20
    x = syn.noise_psd_slope(n, FS, -3.0103, f_lo=10.0, seed=3)[:, 0]
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(n, 1 / FS)
    below = spec[f < 10.0].sum()
    assert below <= 1e-20 * spec.sum()
    # and the band just above the corner is populated at the expected level
    band = spec[(f >= 10.0) & (f < 20.0)].mean() / spec[(f >= 20.0) & (f < 40.0)].mean()
    assert abs(10 * np.log10(band) - 3.0103) < 0.5  # one octave lower: +3.01 dB for pink


def test_noise_correlation():
    x = syn.noise_psd_slope(FS * 5, FS, correlation=0.8, seed=1)
    assert abs(np.corrcoef(x[:, 0], x[:, 1])[0, 1] - 0.8) < 0.02


def test_impulse_train():
    x = syn.impulse_train(FS, FS, 0.1, 0.5)
    assert np.count_nonzero(x[:, 0]) == 10 and x.max() == 0.5


def test_tone_sequence_levels():
    x = syn.tone_sequence([(-20, 1.0), (-30, 1.0)], FS)
    assert x.shape == (2 * FS, 2)
    assert abs(20 * np.log10(np.max(np.abs(x[:FS, 0]))) + 20) < 0.01
    assert abs(20 * np.log10(np.max(np.abs(x[FS:, 0]))) + 30) < 0.01


def test_kick_sub_meta():
    x, meta = syn.kick_sub(FS, bpm=120, bars=2)
    assert x.shape == (int(8 * 0.5 * FS), 2)
    assert len(meta["kick_onsets"]) == 8 and len(meta["roots_per_beat"]) == 8
    assert np.max(np.abs(x)) < 1.5
    g = meta["sidechain_gain"]
    i0 = int(meta["kick_onsets"][1] * FS)
    assert g[i0] == pytest.approx(10 ** (-18 / 20))
    assert g[i0 + int(0.3 * FS)] == 1.0


def test_kick_has_250ms_tail_at_f_end():
    k = syn.kick(FS)
    assert k.size == int(0.25 * FS)
    seg = k[int(0.05 * FS):int(0.2 * FS)]
    f, p = signal.periodogram(seg, FS, nfft=1 << 16)
    assert abs(f[np.argmax(p)] - 50.0) < 2.0


def test_add_ring_boost_at_f0():
    x = syn.pink_noise(FS * 10, FS, seed=3)
    y, meta = syn.add_ring(x, FS, f0=7000, q=12, gain_db=9, start_s=2.0, end_s=6.0)
    seg = slice(int(2.5 * FS), int(5.5 * FS))
    f, px = signal.welch(x[seg, 0], FS, nperseg=8192)
    _, py = signal.welch(y[seg, 0], FS, nperseg=8192)
    k = np.argmin(np.abs(f - 7000))
    assert abs(10 * np.log10(py[k] / px[k]) - 9.0) < 0.5
    # f0 within 1 % (section 7 resonance gate)
    ratio_db = 10 * np.log10(py / px)
    assert abs(f[np.argmax(ratio_db)] - 7000) <= 70
    assert np.array_equal(y[: int(1.9 * FS)], x[: int(1.9 * FS)])
    assert np.allclose(y[int(6.1 * FS):], x[int(6.1 * FS):])


def test_add_ring_time_extent_within_200ms():
    """Start / end recovered within 200 ms from the 7 kHz band-energy excess."""
    x = syn.pink_noise(FS * 10, FS, seed=4)
    y, _ = syn.add_ring(x, FS, f0=7000, q=12, gain_db=9, start_s=2.0, end_s=6.0)
    sos = signal.butter(4, [6800, 7200], btype="bandpass", fs=FS, output="sos")
    hop = int(0.02 * FS)

    def frames(v):
        e = signal.sosfiltfilt(sos, v[:, 0]) ** 2
        return np.array([e[i:i + hop].mean() for i in range(0, e.size - hop, hop)])

    excess = 10 * np.log10(frames(y) / frames(x))
    on = np.flatnonzero(excess > 4.5)
    t_on, t_off = on.min() * 0.02, (on.max() + 1) * 0.02
    assert abs(t_on - 2.0) <= 0.2 and abs(t_off - 6.0) <= 0.2, (t_on, t_off)


@pytest.mark.parametrize("start,end", [(-1.0, 0.01), (0.5, 0.505), (9.99, 20.0)])
def test_add_ring_short_and_out_of_range_spans(start, end):
    x = syn.pink_noise(FS * 10, FS, seed=5)
    y, _ = syn.add_ring(x, FS, start_s=start, end_s=end)
    assert y.shape == x.shape and np.all(np.isfinite(y))
    d = np.abs(y - x)[:, 0]
    i0 = int(np.clip(round(start * FS), 0, x.shape[0]))
    assert np.all(d[:i0] == 0.0)
