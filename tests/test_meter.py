"""Gates for meter.py (spec 3.2, section 7, draft 0.3).

The official EBU loudness test set could not be fetched (tech.ebu.ch serves a
Cloudflare bot check to scripted downloads). `test_ebu_official_vectors` runs
those files if they are placed in fixtures/ebu/ and skips otherwise. The other
Tech 3341 / 3342 tests rebuild the cases from their published definitions:
Tech 3341 (2023) Table 1 was read from the EBU PDF; cases 7 and 8 (authentic
programme) cannot be rebuilt, and 11 and 14 are for live meters.
"""

import math
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from fixtures import synthetic as syn
from eluate import io, meter

FS = 48000
EBU_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "ebu"


def _faded(x, fs, fade_s=0.01):
    nf = int(fade_s * fs)
    w = np.ones(x.shape[0])
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(nf) / nf))
    w[:nf] = ramp
    w[-nf:] = ramp[::-1]
    return x * (w[:, None] if x.ndim == 2 else w)


def fft_true_peak_db(x, up=32):
    """Independent reference: FFT zero-padding interpolation (no polyphase code).

    The signal is embedded in silence (matching the meter's convention) and
    interpolated by `up` with scipy.signal.resample (frequency-domain)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64).T).T
    pad = np.zeros((4096, x.shape[1]))
    z = np.concatenate([pad, x, pad])
    y = signal.resample(z, z.shape[0] * up, axis=0)
    return 20 * np.log10(np.max(np.abs(y)))


# ------------------------------------------------ K-weighting calibration

def test_k_weighting_matches_bs1770_tables_at_48k():
    sos = meter.k_weighting_sos(48000)
    shelf_b = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    shelf_a = [1.0, -1.69065929318241, 0.73248077421585]
    hp_a = [1.0, -1.99004745483398, 0.99007225036621]
    assert np.allclose(sos[0, :3], shelf_b, atol=1e-14)
    assert np.allclose(sos[0, 3:], shelf_a, atol=1e-14)
    assert np.allclose(sos[1, :3], [1.0, -2.0, 1.0], atol=1e-14)
    assert np.allclose(sos[1, 3:], hp_a, atol=1e-14)


@pytest.mark.parametrize("fs", [44100, 48000, 88200, 96000, 192000])
def test_997hz_full_scale_reads_minus_3_01(fs):
    t = np.arange(fs * 5) / fs
    x = np.sin(2 * np.pi * 997 * t)
    got = meter.integrated(x, fs)
    print(f"997 Hz 0 dBFS at {fs}: {got:.4f} LUFS")
    assert abs(got - (-3.01)) <= 0.01


@pytest.mark.parametrize("fs", [44100, 48000, 96000, 192000])
def test_rlb_passband_gain_pinned_to_48k(fs):
    sos = meter.k_weighting_sos(fs)
    _, h = signal.sosfreqz(sos[1:2], worN=[0.45 * fs], fs=fs)
    ref = meter.k_weighting_sos(48000)
    _, h48 = signal.sosfreqz(ref[1:2], worN=[0.45 * 48000], fs=48000)
    assert abs(20 * np.log10(abs(h[0]) / abs(h48[0]))) < 2e-3


def test_dc_reads_minus_inf():
    x = np.full((FS * 3, 2), 0.5)
    assert meter.integrated(x, FS) == -np.inf


def test_gating_hops_uniform_within_one_sample():
    fs = 11025
    starts, _, _ = meter._window_powers(np.ones(fs * 5), fs, meter.BLOCK_S)
    hops = np.diff(starts)
    assert set(hops.tolist()) <= {1102, 1103}


# ------------------------------------------------ Tech 3341 (official definitions)

@pytest.mark.parametrize(
    "case,segments,expected",
    [
        (1, [(-23, 20)], -23.0),
        (2, [(-33, 20)], -33.0),
        (3, [(-36, 10), (-23, 60), (-36, 10)], -23.0),
        (4, [(-72, 10), (-36, 10), (-23, 60), (-36, 10), (-72, 10)], -23.0),
        (5, [(-26, 20), (-20, 20.1), (-26, 20)], -23.0),
    ],
)
def test_tech3341_integrated(case, segments, expected):
    x = syn.tone_sequence(segments, FS)
    assert abs(meter.integrated(x, FS) - expected) <= 0.1


@pytest.mark.parametrize("level", [-23, -33])
def test_tech3341_cases_1_2_momentary_and_short_term(level):
    x = syn.tone_sequence([(level, 20)], FS)
    _, m = meter.momentary(x, FS)
    _, s = meter.short_term(x, FS)
    assert np.all(np.abs(m - level) <= 0.1)
    assert np.all(np.abs(s - level) <= 0.1)


def test_tech3341_case6_five_channel_weights():
    lr = syn.tone_sequence([(-28, 20)], FS)[:, 0]
    c = syn.tone_sequence([(-24, 20)], FS)[:, 0]
    sur = syn.tone_sequence([(-30, 20)], FS)[:, 0]
    x = np.stack([lr, lr, c, sur, sur], axis=1)
    got = meter.integrated(x, FS, weights=[1.0, 1.0, 1.0, 1.41, 1.41])
    assert abs(got - (-23.0)) <= 0.1


def test_tech3341_case9_short_term_constant():
    x = syn.tone_sequence([(-20, 1.34), (-30, 1.66)] * 5, FS)
    _, s = meter.short_term(x, FS)
    assert np.all(np.abs(s - (-23.0)) <= 0.1), (s.min(), s.max())


@pytest.mark.parametrize("i", range(20))
def test_tech3341_case10_max_short_term_per_segment(i):
    x = syn.tone_sequence([(-200, i * 0.15), (-23, 3.0), (-200, 1.0)], FS)
    assert abs(meter.short_term_max(x, FS) - (-23.0)) <= 0.1


def test_tech3341_case12_momentary_constant():
    x = syn.tone_sequence([(-20, 0.18), (-30, 0.22)] * 25, FS)
    _, m = meter.momentary(x, FS)
    assert np.all(np.abs(m - (-23.0)) <= 0.1), (m.min(), m.max())


@pytest.mark.parametrize("i", range(20))
def test_tech3341_case13_max_momentary_per_segment(i):
    x = syn.tone_sequence([(-200, i * 0.02), (-23, 0.4), (-200, 1.0)], FS)
    assert abs(meter.momentary_max(x, FS) - (-23.0)) <= 0.1


def _tp_sine(fs, freq, amp, phase_deg, seconds=1.0):
    n = int(seconds * fs)
    k = np.arange(n)
    x = amp * np.sin(2 * np.pi * freq / fs * k + np.deg2rad(phase_deg))
    return _faded(np.stack([x, x], axis=1), fs, 0.01)


def _case20(fs, offset):
    """fs/6 sine at 0.5 with one period of fs/4 at 1.0, phase continuous,
    synthesised at 4 fs, anti-alias filtered and downsampled at `offset`."""
    up = 4
    n4 = int(1.0 * fs * up)
    f = np.full(n4, fs / 6.0)
    a = np.full(n4, 0.5)
    # start the burst at a zero crossing of the fs/6 tone so the waveform is continuous
    period6 = up * 6
    start = (n4 // 2) // period6 * period6
    burst = up * 4  # one period of fs/4 at the 4 fs rate
    f[start:start + burst] = fs / 4.0
    a[start:start + burst] = 1.0
    ph = 2 * np.pi * np.concatenate([[0.0], np.cumsum(f[:-1])]) / (fs * up)
    x4 = _faded(a * np.sin(ph), fs * up, 0.01)
    h = signal.firwin(255, 0.9 / up)
    y = signal.lfilter(h, 1.0, x4)[127 + offset::up]
    return np.stack([y, y], axis=1)


TP_CASES = [
    (15, lambda: _tp_sine(FS, FS / 4, 0.5, 0.0), -6.0),
    (16, lambda: _tp_sine(FS, FS / 4, 0.5, 45.0), -6.0),
    (17, lambda: _tp_sine(FS, FS / 6, 0.5, 60.0), -6.0),
    (18, lambda: _tp_sine(FS, FS / 8, 0.5, 67.5), -6.0),
    (19, lambda: _tp_sine(FS, FS / 4, 1.41, 45.0), 3.0),
    (20, lambda: _case20(FS, 0), 0.0),
    (21, lambda: _case20(FS, 1), 0.0),
    (22, lambda: _case20(FS, 2), 0.0),
    (23, lambda: _case20(FS, 3), 0.0),
]


@pytest.mark.parametrize("case,make,expected", TP_CASES)
@pytest.mark.parametrize("oversample", [4, 8])
def test_tech3341_true_peak_cases(case, make, expected, oversample):
    tp = meter.true_peak(make(), FS, oversample)
    print(f"3341 case {case} {oversample}x: {tp:.3f} dBTP (expected {expected})")
    assert expected - 0.4 <= tp <= expected + 0.2, (case, oversample, tp)


# ------------------------------------------------ Tech 3342 (rebuilt)

@pytest.mark.parametrize(
    "segments,expected",
    [
        ([(-20, 20), (-30, 20)], 10.0),
        ([(-20, 20), (-15, 20)], 5.0),
        ([(-40, 20), (-20, 20)], 20.0),
        ([(-50, 20), (-35, 20), (-20, 20), (-35, 20), (-50, 20)], 15.0),
    ],
)
def test_tech3342_lra(segments, expected):
    x = syn.tone_sequence(segments, FS)
    assert abs(meter.lra(x, FS) - expected) <= 0.1  # Tech 3342 tolerance is +-1 LU


def test_lra_percentile_rule_on_a_ramp():
    """8 s tone ramping -40 -> -20 dBFS: 51 short-term values, where the
    libebur128 rule (10.750 LU) and linear interpolation (10.625 LU) differ.
    Expected value computed analytically from the ramp, not by the meter."""
    T = 8.0
    n = int(T * FS)
    t = np.arange(n) / FS
    level = -40 + 20 * t / T
    x = 10 ** (level / 20) * np.sin(2 * np.pi * 997 * t)
    L, hop = 3 * FS, FS // 10
    c = np.concatenate([[0.0], np.cumsum(10 ** (level / 10))])
    st = np.arange(0, n - L + 1, hop)
    s = np.sort(10 * np.log10((c[st + L] - c[st]) / L))
    expected = s[int((s.size - 1) * 0.95 + 0.5)] - s[int((s.size - 1) * 0.10 + 0.5)]
    got = meter.lra(np.stack([x, x], axis=1), FS)
    assert abs(got - expected) <= 0.02, (got, expected)


# ------------------------------------------------ official vectors if present

def _ebu_file(pattern):
    hits = sorted(EBU_DIR.rglob(pattern)) if EBU_DIR.exists() else []
    return hits[0] if hits else None


EBU_CASES = [
    # (file pattern, quantity, expected, tolerance). Expected values from EBU Tech 3341 (2023 edition)
    # Table 1 and Tech 3342; the set is "EBU loudness test set v05" (2016).
    ("seq-3341-1-*.wav", "I", -23.0, 0.1),
    ("seq-3341-2-*.wav", "I", -33.0, 0.1),
    ("seq-3341-3-*.wav", "I", -23.0, 0.1),
    ("seq-3341-4-*.wav", "I", -23.0, 0.1),
    ("seq-3341-5-*.wav", "I", -23.0, 0.1),
    ("seq-3341-7_*.wav", "I", -23.0, 0.1),
    ("seq-3341-*8_seq-3342-6*.wav", "I", -23.0, 0.1),
    ("seq-3341-9-*.wav", "S", -23.0, 0.1),
    ("seq-3341-12-*.wav", "M", -23.0, 0.1),
    *[(f"seq-3341-10-{n}-*.wav", "S", -23.0, 0.1) for n in range(1, 21)],
    *[(f"seq-3341-13-{n}-*.wav*", "M", -23.0, 0.1) for n in range(1, 21)],
    *[(f"seq-3341-{n}-24bit*.wav", "TP", -6.0, (0.2, 0.4)) for n in (15, 16, 17, 18)],
    ("seq-3341-19-24bit*.wav", "TP", 3.0, (0.2, 0.4)),
    *[(f"seq-3341-{n}-24bit*.wav", "TP", 0.0, (0.2, 0.4)) for n in (20, 21, 22, 23)],
    ("seq-3342-1-*.wav", "LRA", 10.0, 1.0),
    ("seq-3342-2-*.wav", "LRA", 5.0, 1.0),
    ("seq-3342-3-*.wav", "LRA", 20.0, 1.0),
    ("seq-3342-4-*.wav", "LRA", 15.0, 1.0),
    ("seq-3341-7_seq-3342-5*.wav", "LRA", 5.0, 1.0),
    ("seq-3341-*8_seq-3342-6*.wav", "LRA", 15.0, 1.0),
]


@pytest.mark.parametrize("pattern,kind,expected,tol", EBU_CASES)
def test_ebu_official_vectors(pattern, kind, expected, tol):
    path = _ebu_file(pattern)
    if path is None:
        pytest.skip(f"official EBU vector {pattern} not present in fixtures/ebu/")
    x, fs = io.read(path)
    fn = {"I": meter.integrated, "LRA": meter.lra, "S": meter.short_term_max,
          "M": meter.momentary_max, "TP": lambda x, fs: meter.true_peak(x, fs, 8)}[kind]
    got = fn(x, fs)
    if kind == "TP":  # Tech 3341 true-peak tolerance is +0.2 / -0.4 dB
        assert expected - tol[1] <= got <= expected + tol[0], (path.name, got)
    else:
        assert abs(got - expected) <= tol, (path.name, got)


# ------------------------------------------------ pyloudnorm sanity check

@pytest.mark.parametrize("fs", [44100, 48000, 96000])
def test_pyloudnorm_crosscheck_pink_noise_bias_compensated(fs):
    """pyloudnorm normalises the RLB numerator to unity HF gain; BS.1770's table
    filter (and ours, at every rate) has a passband gain of a0(48k) = +0.0433 dB.
    With that known bias removed the two agree to 0.002 LU (measured)."""
    pyln = pytest.importorskip("pyloudnorm")
    k = math.tan(math.pi * 38.13547087602444 / 48000)
    bias = 20 * math.log10(1 + k / 0.5003270373238773 + k * k)
    x = syn.pink_noise(fs * 20, fs, rms_dbfs=-20.0, seed=fs)
    ours = meter.integrated(x, fs)
    theirs = pyln.Meter(fs).integrated_loudness(x)
    assert abs(ours - bias - theirs) <= 0.005, (ours, theirs)


# ------------------------------------------------ true peak

@pytest.mark.parametrize("fs", [44100, 48000])
@pytest.mark.parametrize("oversample,tol", [(4, 0.2), (8, 0.05)])
def test_true_peak_intersample_sine(fs, oversample, tol):
    n = fs
    k = np.arange(n)
    a = 10 ** (-1.0 / 20)
    x = a * np.sin(2 * np.pi * 0.25 * k + np.pi / 4)
    x = np.stack([x, x], axis=1)
    assert abs(meter.sample_peak(x) - (-1.0 - 3.0103)) < 1e-6
    tp = meter.true_peak(_faded(x, fs, 0.05), fs, oversample)
    assert abs(tp - (-1.0)) <= tol, tp


@pytest.mark.parametrize("fs", [44100, 48000])
@pytest.mark.parametrize("freq", [997.0, 5000.0, 11025.0, 15000.0, 19000.0, 20500.0])
def test_8x_matches_fft_reference_on_sines(fs, freq):
    rng = np.random.default_rng(int(freq))
    t = np.arange(fs // 2) / fs
    x = _faded(0.5 * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi)), fs, 0.02)
    assert abs(meter.true_peak(x, fs, 8) - fft_true_peak_db(x)) <= 0.05


@pytest.mark.parametrize("ratio,phase_deg,n_ms", [(3 / 8, 22.5, 2.0), (7 / 16, 11.25, 4.0), (1 / 4, 45.0, 1.0)])
def test_8x_matches_fft_reference_on_short_hf_bursts(ratio, phase_deg, n_ms):
    """Short Hann bursts at 44.1 kHz (16.5 / 19.3 / 11.0 kHz) whose samples sit a
    fixed angle off every crest, spectrum inside the 0.93 x Nyquist passband."""
    fs = 44100
    n = int(n_ms * 1e-3 * fs)
    k = np.arange(n)
    burst = np.hanning(n) * np.sin(2 * np.pi * ratio * k + np.deg2rad(90 + phase_deg))
    x = np.zeros(fs // 4)
    x[2000:2000 + n] = 0.9 * burst
    ref = fft_true_peak_db(x)
    assert ref - meter.sample_peak(x) > 0.1  # samples really miss the crest
    assert abs(meter.true_peak(x, fs, 8) - ref) <= 0.05


def test_8x_under_reads_content_beyond_its_passband():
    """Recorded limit: a 4 ms burst at 20.5 kHz (fs 44.1 kHz) spreads past 0.93 x
    Nyquist and reads about 0.1 dB low against the FFT reference."""
    fs = 44100
    n = int(0.004 * fs)
    t = np.arange(n) / fs
    x = np.zeros(fs // 4)
    x[2000:2000 + n] = 0.9 * np.hanning(n) * np.sin(2 * np.pi * 20500 * t + 0.7)
    d = meter.true_peak(x, fs, 8) - fft_true_peak_db(x)
    print(f"20.5 kHz burst: 8x - FFT = {d:.3f} dB")
    assert -0.15 <= d <= 0.0


def test_4x_vs_8x_difference_bounded():
    """Regression: 4x (Annex 2) minus 8x over a frequency x phase sweep, 48 kHz.
    Measured max |difference| 0.213 dB (4x reads high from its passband ripple)."""
    fs = 48000
    rng = np.random.default_rng(0)
    worst = 0.0
    for f in np.geomspace(100, 21000, 25):
        for ph in rng.uniform(0, 2 * np.pi, 3):
            t = np.arange(fs // 4) / fs
            x = _faded(0.5 * np.sin(2 * np.pi * f * t + ph), fs, 0.01)
            worst = max(worst, abs(meter.true_peak(x, fs, 4) - meter.true_peak(x, fs, 8)))
    print(f"max |TP4 - TP8| = {worst:.3f} dB")
    assert worst <= 0.25


def test_every_sample_owns_exactly_L_phases():
    """Tone with an odd period (47 samples): even and odd samples visit the same
    waveform phases, so their envelope means agree unless the phase-to-sample
    mapping gives alternate samples more phases (the old rint mapping: 9 / 7)."""
    fs = 48000
    k = np.arange(47 * 400)
    x = 0.5 * np.sin(2 * np.pi * k / 47)
    env = meter.true_peak_envelope(x, fs, 8)[470:-470]
    assert abs(env[0::2].mean() - env[1::2].mean()) < 1e-5 * env.mean()


def test_true_peak_includes_edge_overshoot_of_truncated_sine():
    fs = 48000
    t = np.arange(fs) / fs
    x = 0.5 * np.cos(2 * np.pi * 997 * t)  # starts at its crest: a 0.5 step
    assert meter.true_peak(x, fs, 8) > 20 * np.log10(0.5) + 0.3  # Gibbs at the step


def test_true_peak_never_below_sample_peak():
    x = syn.pink_noise(FS * 2, FS, rms_dbfs=-12, seed=3)
    assert meter.true_peak(x, FS, 4) >= meter.sample_peak(x)
    assert meter.true_peak(x, FS, 8) >= meter.sample_peak(x)


def test_upsample_decimate_roundtrip():
    x = syn.pink_noise(FS, FS, rms_dbfs=-20, seed=9)
    x = signal.sosfiltfilt(signal.butter(8, 0.85 * FS / 2, fs=FS, output="sos"), x, axis=0)
    y = meter.upsample(x, 8)
    assert y.shape == (FS * 8, 2)
    assert np.max(np.abs(y[::8] - x)[200:-200]) < 2e-3
    z = meter.decimate(y, 8)
    assert np.max(np.abs(z - x)[200:-200]) < 2e-3


# ------------------------------------------------ sentinels and bad input

def test_silence_short_and_empty_inputs():
    silent = np.zeros((FS * 5, 2))
    assert meter.integrated(silent, FS) == -np.inf
    assert math.isnan(meter.lra(silent, FS))
    assert math.isnan(meter.psr(silent, FS)[2])
    assert math.isnan(meter.loud_passage_lufs(silent, FS)["s95"])
    m = meter.measure(silent, FS)
    assert m["integrated_lufs"] == -np.inf and math.isnan(m["lra_lu"])

    short = syn.pink_noise(int(0.3 * FS), FS, seed=1)  # < 400 ms
    assert meter.integrated(short, FS) == -np.inf
    assert math.isnan(meter.lra(short, FS))
    assert math.isnan(meter.psr(short, FS)[2])
    assert math.isnan(meter.loud_passage_lufs(short, FS)["s95"])
    meter.measure(short, FS)

    empty = np.zeros((0, 2))
    assert meter.true_peak(empty, FS) == -np.inf
    assert meter.true_peak_envelope(empty, FS).size == 0
    assert meter.sample_peak(empty) == -np.inf


def test_s95_falls_back_to_momentary_between_0_4_and_3_s():
    x = syn.pink_noise(2 * FS, FS, rms_dbfs=-20, seed=2)
    lp = meter.loud_passage_lufs(x, FS)
    assert lp["window_s"] == meter.BLOCK_S
    assert abs(lp["s95"] - meter.integrated(x, FS)) < 1.0


def test_non_finite_input_rejected():
    x = np.zeros((FS, 2))
    x[100, 1] = np.nan
    for fn in (meter.integrated, meter.true_peak, meter.lra):
        with pytest.raises(ValueError):
            fn(x, FS)


# ------------------------------------------------ loud passage, PLR, PSR

def test_loud_passage_on_two_level_programme():
    fs = FS
    quiet = syn.pink_noise(fs * 40, fs, rms_dbfs=-30, seed=1)
    loud = syn.pink_noise(fs * 20, fs, rms_dbfs=-18, seed=2)
    x = np.concatenate([quiet, loud, quiet])
    lp = meter.loud_passage_lufs(x, fs)
    i_loud = meter.integrated(loud, fs)
    i_all = meter.integrated(x, fs)
    assert lp["window_s"] == meter.SHORT_S
    assert abs(lp["loud_section"] - i_loud) <= 0.1
    assert abs(lp["s95"] - i_loud) <= 0.3
    assert lp["s95"] > i_all + 1.0
    assert lp["s_max"] >= lp["s95"]


def test_plr_psr_on_steady_sine():
    x = _faded(syn.tone_sequence([(-23, 20)], FS, freq=997.0), FS, 0.05)
    assert abs(meter.plr(x, FS) - 0.0) <= 0.1
    _, series, p95 = meter.psr(x, FS)
    assert np.all(np.abs(series - 0.0) <= 0.1)
    assert abs(p95) <= 0.1


def test_measure_summary_keys():
    m = meter.measure(syn.pink_noise(FS * 10, FS, seed=4), FS)
    for key in ("integrated_lufs", "s95_lufs", "loud_section_lufs", "lra_lu",
                "true_peak_dbtp", "true_peak_4x_dbtp", "plr_db", "psr_p95_db",
                "momentary_max_lufs", "short_term_max_lufs"):
        assert np.isfinite(m[key]), key
    assert m["true_peak_oversample"] == 8


def test_crest_db_is_the_corpus_reading():
    """meter.crest_db equals work/corpus/measure.py's crest (sample peak minus the p95 of
    the 1 s RMS at a 100 ms hop, plain dBFS, windows above -70 dBFS), the reading the
    maintainer's masters were surveyed with; measure() carries the same value."""
    rng = np.random.default_rng(7)
    x = syn.pink_noise(FS * 12, FS, seed=5)
    x[:, 1] = 0.7 * x[:, 1] + 0.05 * rng.standard_normal(len(x))
    x[FS * 4:FS * 6] *= 3.0                                  # a loud passage
    p = np.mean(x ** 2, axis=1)
    c = np.concatenate([[0.0], np.cumsum(p)])
    idx = np.arange(0, len(p) - FS + 1, FS // 10)
    r = 10 * np.log10((c[idx + FS] - c[idx]) / FS + 1e-20)
    ref = 20 * np.log10(np.abs(x).max()) - np.percentile(r[r > -70], 95)
    assert meter.crest_db(x, FS) == pytest.approx(ref, abs=1e-9)
    assert meter.measure(x, FS)["crest_db"] == pytest.approx(ref, abs=1e-9)
    assert math.isnan(meter.crest_db(np.zeros((FS // 2, 2)), FS))
