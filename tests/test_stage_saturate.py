"""Gates for stages/saturate.py (spec 3.5 stage 5; audit 05 A12, A13)."""

import math

import numpy as np
import pytest

from fixtures import synthetic as syn
from eluate import bands
from eluate.plan import SaturatePlan
from eluate.stages import saturate

FS = 48000
NB = 5
RATES = [44100, 48000, 96000]


def _plan(drive=None, harmonic=None, symmetry=0.3):
    drive = drive if drive is not None else [[0.0, 0.0]] * NB
    harmonic = harmonic if harmonic is not None else [0.0] * NB
    return SaturatePlan(drive_db=drive, harmonic=harmonic, symmetry=symmetry)


def _spectrum_db(v, fs):
    w = np.hanning(v.size)
    s = np.abs(np.fft.rfft(v * w)) / (w.sum() / 2)
    return 20 * np.log10(np.maximum(s, 1e-300)), np.fft.rfftfreq(v.size, 1 / fs)


def _level(spec, f, f0):
    return spec[np.argmin(np.abs(f - f0))]


def _tone(f_approx, fs, n=1 << 16, amp=0.5):
    f0 = fs * round(f_approx * n / fs) / n  # bin-centred
    return f0, amp * np.sin(2 * np.pi * f0 * np.arange(n) / fs)


@pytest.mark.parametrize("fs", RATES)
def test_zero_drive_or_zero_blend_is_identity(fs):
    x = syn.pink_noise(fs, fs, seed=3)
    assert np.max(np.abs(saturate.apply(x, fs, _plan()) - x)) < 1e-15
    y1 = saturate.apply(x, fs, _plan(drive=[[0, 0]] + [[12.0, 12.0]] * 4))
    assert np.max(np.abs(y1 - x)) < 1e-15  # harmonic 0
    band = np.random.default_rng(0).standard_normal(1000)
    assert np.array_equal(saturate.saturate_band(band, 0.0, 1.0, fs, 500.0), band)
    assert np.array_equal(saturate.saturate_band(band, 6.0, 0.0, fs, 500.0), band)


def test_tiny_drive_is_identity_not_an_exception():
    """Audit 05 A13: drive 1e-16 dB used to divide by zero."""
    band = 0.3 * np.sin(np.arange(4800) * 0.1)
    for d in (1e-16, 1e-13):
        assert np.array_equal(saturate.saturate_band(band, d, 1.0, FS, 500.0), band)
    out = saturate.saturate_band(band, 1e-6, 1.0, FS, 500.0)
    assert np.max(np.abs(out - band)) < 1e-6


def test_sub_drive_must_be_zero_and_defaults():
    with pytest.raises(ValueError, match="sub"):
        _plan(drive=[[3.0, 0.0]] + [[0.0, 0.0]] * 4)
    with pytest.raises(ValueError, match="sub"):
        _plan(drive=[[0.0, 1.0]] + [[0.0, 0.0]] * 4)
    d = SaturatePlan.defaults(width_drive_db=3.0)
    assert d.drive_db == [[0, 0], [0, 0], [0, 0], [0, 3.0], [0, 0]]  # air 0, high-mid S width
    assert d.symmetry == 0.3


@pytest.mark.parametrize("fs", RATES)
def test_unity_small_signal_slope_not_rms_equal(fs):
    """Audit 05 A12: no RMS normalisation. With the odd curve a passage 40 dB below the
    knee passes almost unchanged (unity slope, the change falls as level squared); the
    loud passage changes level, and the band's RMS is not forced back to the input's.
    With symmetry the small-signal change is the H2 term, linear in level."""
    n = 2 * fs
    f0, s = _tone(3000.0, fs, n, 1.0)
    env = np.where(np.arange(n) < n // 2, 0.5, 0.005)
    band = s * env
    quiet = slice(n // 2 + fs // 10, n - fs // 10)
    loud = slice(fs // 10, n // 2 - fs // 10)
    q = {}
    for sym in (0.0, 0.3):
        out = saturate.saturate_band(band, 12.0, 1.0, fs, 2000.0, symmetry=sym)
        q[sym] = np.max(np.abs(out[quiet] - band[quiet])) / np.max(np.abs(band[quiet]))
        if sym == 0.0:
            ratio_db = 20 * np.log10(np.sqrt(np.mean(out[loud] ** 2))
                                     / np.sqrt(np.mean(band[loud] ** 2)))
    print(f"{fs} Hz: quiet passage relative change {q[0.0]:.2e} (odd), {q[0.3]:.2e} "
          f"(symmetry 0.3); loud passage RMS change {ratio_db:+.3f} dB")
    assert q[0.0] < 1e-3
    assert q[0.3] < 1e-2
    assert abs(ratio_db) > 0.1


@pytest.mark.parametrize("fs", [44100, 48000])
def test_air_band_tone_gets_in_band_h2(fs):
    """Audit 05 A12: tanh is odd, so a 10 kHz tone had no harmonics below Nyquist
    (H2 at -268 dB). symmetry > 0 adds H2 at 20 kHz; symmetry 0 leaves none."""
    f0, band = _tone(10000.0, fs)
    h2 = {}
    for sym in (0.0, 0.3, 1.0):
        out = saturate.saturate_band(band, 12.0, 1.0, fs, 8000.0, symmetry=sym)
        spec, f = _spectrum_db(out, fs)
        h2[sym] = _level(spec, f, 2 * f0) - _level(spec, f, f0)
    print(f"{fs} Hz air tone H2 re fundamental: " +
          ", ".join(f"symmetry {k}: {v:.1f} dB" for k, v in h2.items()))
    assert h2[0.0] < -150
    assert h2[0.3] > -40 and h2[1.0] > h2[0.3]


@pytest.mark.parametrize("fs", RATES)
def test_oversampling_suppresses_aliases(fs):
    """7 kHz sine in the high-mid band, fully saturated: the aliases of H5 and H7
    (which would fold into the audio band at 1x) are removed by the 4x path; what is left
    is the decimation filter's >= 60 dB stopband applied to those harmonics."""
    f0, band = _tone(7000.0, fs)
    out = saturate.saturate_band(band, 12.0, 1.0, fs, 2000.0, symmetry=0.0)
    spec, f = _spectrum_db(out, fs)
    fund = _level(spec, f, f0)
    h3 = _level(spec, f, 3 * f0) if 3 * f0 < fs / 2 else -300.0
    alias = [(k * f0) % fs for k in (5, 7, 9) if k * f0 > fs / 2]   # harmonics that would fold
    alias = [min(a, fs - a) for a in alias]
    lv = [_level(spec, f, a) - fund for a in alias if abs(a - f0) > 200]
    print(f"{fs} Hz: H3 {h3 - fund:.1f} dB, alias levels {[round(v, 1) for v in lv]} dB")
    assert all(v < -90 for v in lv)  # the decimator's 60 dB stopband on harmonics >= -40 dB


@pytest.mark.parametrize("fs", RATES)
def test_knee_is_robust_to_a_spike(fs):
    """Audit 05 A13: one spike sample used to drop H3 by 25 dB (knee = whole-track peak)."""
    n = 2 * fs
    f0 = fs * round(1000.0 * n / fs) / n
    band = 0.2 * np.sin(2 * np.pi * f0 * np.arange(n) / fs)
    spiked = band.copy()
    spiked[0] = 1.0
    h3 = []
    for b in (band, spiked):
        out = saturate.saturate_band(b, 6.0, 1.0, fs, 500.0, symmetry=0.0)
        spec, f = _spectrum_db(out[fs // 10:], fs)
        h3.append(_level(spec, f, 3 * f0) - _level(spec, f, f0))
    print(f"{fs} Hz H3: clean {h3[0]:.2f} dB, with one spike {h3[1]:.2f} dB")
    assert abs(h3[0] - h3[1]) < 0.1


def test_no_dc_or_subsonic_from_even_term():
    """The even term rectifies; the change is high-passed at the band's lower edge."""
    f0, band = _tone(3000.0, FS, 1 << 17)
    out = saturate.saturate_band(band, 12.0, 1.0, FS, 2000.0, symmetry=1.0)
    change = out - band
    spec, f = _spectrum_db(change, FS)
    low = spec[(f > 0) & (f < 200)].max() - _level(spec, f, 2 * f0)
    print(f"change below 200 Hz re H2: {low:.1f} dB, DC {np.mean(change):.2e}")
    assert abs(np.mean(change)) < 1e-5
    assert low < -60


def test_harmonic_blend_is_monotone():
    n = 1 << 15
    f0 = FS * 1000 / n
    band = 0.4 * np.sin(2 * np.pi * f0 * np.arange(n) / FS)
    thd = []
    for h in (0.0, 0.25, 0.5, 0.75, 1.0):
        out = saturate.saturate_band(band, 9.0, h, FS, 500.0)
        spec, f = _spectrum_db(out, FS)
        fund = 10 ** (_level(spec, f, f0) / 20)
        harm = math.sqrt(sum((10 ** (_level(spec, f, k * f0) / 20)) ** 2 for k in (2, 3, 4, 5)))
        thd.append(20 * np.log10(max(harm / fund, 1e-30)))
    print("THD (H2-H5) vs harmonic 0..1:", " ".join(f"{t:.1f}" for t in thd))
    assert thd[0] < -200
    assert all(b > a for a, b in zip(thd, thd[1:]))


@pytest.mark.parametrize("fs", RATES)
def test_side_width_drive_changes_side_only(fs):
    x = syn.pink_noise(fs, fs, rms_dbfs=-14, seed=6)
    p = SaturatePlan.defaults(width_drive_db=9.0, harmonic=1.0)
    y = saturate.apply(x, fs, p)
    assert np.max(np.abs(bands.to_ms(y)[:, 0] - bands.to_ms(x)[:, 0])) < 1e-14
    assert np.max(np.abs(bands.to_ms(y)[:, 1] - bands.to_ms(x)[:, 1])) > 1e-4


@pytest.mark.parametrize("fs", [44100, 48000])
def test_air_band_8x_limits_foldback(fs):
    """Audit 05 Q5: a near-Nyquist air-band tone at the 12 dB drive cap; at 8x its
    harmonics above the 4x Nyquist no longer fold back into the audio band."""
    f0, band = _tone(0.875 * fs / 2, fs)
    worst = {}
    for L in (4, 8):
        out = saturate.saturate_band(band, 12.0, 1.0, fs, 8000.0, symmetry=0.3, oversample=L)
        spec, f = _spectrum_db(out, fs)
        fund = _level(spec, f, f0)
        away = (np.abs(f - f0) > 300) & (f > 20)
        worst[L] = float(spec[away].max() - fund)
    print(f"{fs} Hz tone {f0:.0f} Hz, drive 12: worst foldback re fundamental "
          f"4x {worst[4]:.1f} dB, 8x {worst[8]:.1f} dB")
    # what remains at 8x is the decimator's 60 dB stopband on H3 (about -23 dB)
    assert worst[8] < -80
    assert worst[8] < worst[4] - 20


def test_drive_capped_at_12_db():
    with pytest.raises(ValueError):
        _plan(drive=[[0, 0], [0, 0], [0, 0], [0, 0], [13.0, 0]], harmonic=[1.0] * NB)
