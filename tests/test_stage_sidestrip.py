"""Gates for stages/sidestrip.py (spec 3.5 stage 2)."""

import numpy as np
import pytest
from scipy import signal

from eluate import bands
from eluate.plan import SidestripPlan
from eluate.stages import sidestrip

RATES = [44100, 48000, 96000]


def _tone(f, n, fs, amp=0.3):
    return amp * np.sin(2 * np.pi * f * np.arange(n) / fs)


def _gain_db(y, x, fs):
    mid = slice(fs // 2, -fs // 2)
    return 20 * np.log10(np.sqrt(np.mean(y[mid] ** 2)) / np.sqrt(np.mean(x[mid] ** 2)))


@pytest.mark.parametrize("fs", RATES)
def test_mid_is_untouched_never_mono_sums(fs):
    """M (hence L + R) passes unchanged at every frequency; only S is filtered."""
    n = 2 * fs
    x = np.random.default_rng(0).standard_normal((n, 2)) * 0.1
    y = sidestrip.apply(x, fs, SidestripPlan(100.0))
    assert np.max(np.abs(bands.to_ms(y)[:, 0] - bands.to_ms(x)[:, 0])) < 1e-15
    assert np.max(np.abs((y[:, 0] + y[:, 1]) - (x[:, 0] + x[:, 1]))) < 1e-15


@pytest.mark.parametrize("fs", RATES)
@pytest.mark.parametrize("fc", [60.0, 100.0, 150.0])
def test_side_high_pass_response(fs, fc):
    """S gain = the squared digital Butterworth high-pass |HP|^2 (order 2, zero phase);
    near r^4 / (1 + r^4), r = f / fc."""
    n = 3 * fs
    sos = signal.butter(2, fc, btype="highpass", fs=fs, output="sos")
    for f in (0.3 * fc, 0.5 * fc, fc, 2 * fc, 4.6 * fc, 1000.0):
        s = _tone(f, n, fs)
        ms = np.stack([np.zeros(n), s], axis=1)
        y = bands.to_ms(sidestrip.apply(bands.to_lr(ms), fs, SidestripPlan(fc)))
        expect = 20 * np.log10(np.abs(signal.sosfreqz(sos, worN=[f], fs=fs)[1][0]) ** 2)
        got = _gain_db(y[:, 1], s, fs)
        print(f"{fs} Hz fc {fc}: S at {f:.0f} Hz {got:.3f} dB (expect {expect:.3f})")
        assert abs(got - expect) < 0.02
        assert np.max(np.abs(y[:, 0])) < 1e-15


@pytest.mark.parametrize("fs", RATES)
def test_side_kept_above_fc_and_stereo_preserved(fs):
    n = 2 * fs
    m = _tone(50.0, n, fs, 0.4)
    s = _tone(2000.0, n, fs, 0.2)
    x = bands.to_lr(np.stack([m, s], axis=1))
    y = sidestrip.apply(x, fs, SidestripPlan(100.0))
    assert abs(_gain_db(bands.to_ms(y)[:, 1], s, fs)) < 1e-3
    assert np.max(np.abs(y[:, 0] - y[:, 1])) > 0.25


def test_rejects_mono():
    with pytest.raises(ValueError):
        sidestrip.apply(np.zeros((100, 1)), 48000, SidestripPlan())
