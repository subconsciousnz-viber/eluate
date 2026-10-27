"""Gates for stages/limiter.py (spec 3.5 stage 7, section 7; audit 02 A11, D.10; audit 05 A1-A4)."""

import math

import numpy as np
import pytest
from scipy import signal

from fixtures import synthetic as syn
from eluate import meter
from eluate.plan import LimiterPlan
from eluate.stages import limiter

FS = 48000
TARGET = -1.0
RATES = [44100, 48000, 96000]


def fft_reference_db(x, up=32):
    """Independent whole-signal reference: FFT zero-padding interpolation of the
    signal embedded in 4096 samples of silence (no polyphase, no blocks)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64).T).T
    pad = np.zeros((4096, x.shape[1]))
    z = np.concatenate([pad, x, pad])
    y = signal.resample(z, z.shape[0] * up, axis=0)
    return 20 * np.log10(np.max(np.abs(y)))


def _fractional_impulses(n, fs, period_s, amp, rng):
    """Band-limited impulses (Hann-windowed sinc, 129 taps) at random fractional positions."""
    x = np.zeros(n)
    t = np.arange(n)
    for t0 in np.arange(0.013, n / fs - 0.01, period_s):
        pos = t0 * fs + rng.uniform(0.1, 0.9)
        w = np.abs(t - pos) < 64
        seg = t[w] - pos
        x[w] += amp * np.sinc(seg) * (0.5 + 0.5 * np.cos(np.pi * seg / 64))
    return x


def sweep_plus_impulses(seconds=1.0, seed=0, fs=FS):
    """Spec 7 stimulus: 1.2-amplitude log sweep 20 Hz to 20 kHz plus fractional-phase
    impulses (the construction of tests/test_envelope.py, audit 02 A11)."""
    n = int(seconds * fs)
    t = np.arange(n) / fs
    fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.01)
    sweep = 1.2 * signal.chirp(t, 20, t[-1], 20000, method="logarithmic") * fade
    x = sweep + _fractional_impulses(n, fs, 0.07, 1.0, np.random.default_rng(seed))
    return np.stack([x, 0.8 * x[::-1]], axis=1)


GATE_CASES = [(1.0, 0, "box", 48000), (2.0, 0, "box", 48000), (2.0, 1, "box", 48000),
              (3.0, 5, "box", 48000), (1.0, 0, "qp", 48000), (1.0, 3, "box", 44100),
              (1.0, 3, "box", 96000)]


def _report(tag, info, tp8, ref):
    p = info["relimit_passes"]
    print(f"{tag}: GR {info['max_gain_reduction_db']:.2f} dB; before re-limit 8x "
          f"{info['true_peak_8x_before_relimit_dbtp']:.4f} / 32x "
          f"{info['true_peak_32x_before_relimit_dbtp']:.4f}; re-limit passes {len(p)} "
          f"(max {max([q['max_reduction_db'] for q in p], default=0):.4f} dB); trim "
          f"{info['trim_db']:.4f}; after 8x {tp8:.4f}, FFT ref {ref:.4f} dBTP")


@pytest.mark.parametrize("seconds,seed,smoother,fs", GATE_CASES)
def test_true_peak_gate(seconds, seed, smoother, fs):
    """Spec 7: 1.2-amplitude log sweep plus fractional-phase impulses stays <= target
    + 0.05 dB. Asserted tighter: the 8x meter at or below the ceiling, the whole-signal
    32x FFT reference within 0.002 dB of it, and no global trim (audit 05 A1, A2)."""
    x = sweep_plus_impulses(seconds, seed, fs)
    assert meter.true_peak(x, fs, 8) > TARGET + 6.0
    y, info = limiter.apply_with_info(x, fs, LimiterPlan(ceiling_dbtp=TARGET, smoother=smoother))
    tp8, ref = meter.true_peak(y, fs, 8), fft_reference_db(y)
    _report(f"{seconds} s seed {seed} {smoother} {fs} Hz", info, tp8, ref)
    assert tp8 <= TARGET + 1e-9
    assert ref <= TARGET + 0.002
    assert info["trim_db"] == 0.0


def _bed_plus_hf_burst(fs=FS, burst_db=2.0, f_hf=10000.0, onset=0.5):
    """Audit 05 A2 stimulus: -2 dBFS 1 kHz bed plus a 5 ms Hann-windowed 10 kHz burst
    peaking at +2 dBFS at the onset."""
    n = fs
    t = np.arange(n) / fs
    bed = 10 ** (-2 / 20) * np.sin(2 * np.pi * 1000 * t)
    burst = np.zeros(n)
    i0, m = int(onset * fs), int(0.005 * fs)
    burst[i0:i0 + m] = np.hanning(m) * 10 ** (burst_db / 20) * np.sin(2 * np.pi * f_hf * t[:m])
    return np.stack([bed + burst] * 2, axis=1)


@pytest.mark.parametrize("fs", RATES)
def test_clipper_with_hf_content_at_onset(fs):
    """Clipped HF harmonics above base Nyquist are removed before the detector, so
    the limiter sees the rebound: no global trim, 8x at the ceiling, and the track
    before the onset equals the render without the clipper to rounding."""
    x = _bed_plus_hf_burst(fs)
    y0, i0 = limiter.apply_with_info(x, fs, LimiterPlan(ceiling_dbtp=TARGET))
    y, info = limiter.apply_with_info(x, fs, LimiterPlan(ceiling_dbtp=TARGET, clip_depth_db=3.0,
                                                         clip_onsets_s=[0.5]))
    tp8, ref = meter.true_peak(y, fs, 8), fft_reference_db(y)
    print(f"{fs} Hz: GR without clipper {i0['max_gain_reduction_db']:.2f} dB, with 3 dB "
          f"clipper {info['max_gain_reduction_db']:.2f} dB (clip {info['clip_max_reduction_db']:.2f})")
    _report(f"  {fs} Hz clipped", info, tp8, ref)
    assert info["clip_max_reduction_db"] == 3.0
    assert info["trim_db"] == 0.0
    assert tp8 <= TARGET + 1e-9 and ref <= TARGET + 0.002
    before = slice(0, int(0.45 * fs))
    # equal to FFT-convolution rounding (the processing resampler is overlap-add FFT)
    assert np.max(np.abs(y[before] - y0[before])) < 1e-12
    # the clipper still spares the limiter some reduction, net of the rebound
    assert info["max_gain_reduction_db"] < i0["max_gain_reduction_db"]


def test_local_relimit_leaves_the_rest_untouched():
    """The re-limit acts only around the excess: a quiet section 0.6 s before the only
    hot event equals (to rounding) the same section rendered alone through the limiter."""
    fs = FS
    x = _bed_plus_hf_burst(fs, burst_db=4.0, onset=0.8)
    y, info = limiter.apply_with_info(x, fs, LimiterPlan())
    assert info["relimit_passes"] and info["trim_db"] == 0.0
    early = slice(0, int(0.2 * fs))
    y_ref, _ = limiter.apply_with_info(x[: int(0.4 * fs)], fs, LimiterPlan())
    assert np.max(np.abs(y[early] - y_ref[early])) < 1e-12  # FFT-convolution rounding


@pytest.mark.parametrize("n", [1000, 40000, 100003])
def test_blockwise_fft_matches_whole_signal_fft(n):
    rng = np.random.default_rng(n)
    x = rng.standard_normal((n, 2)) * 0.2
    x = signal.sosfilt(signal.butter(6, 0.9, output="sos"), x, axis=0)
    d = limiter.true_peak_fft(x) - fft_reference_db(x)
    print(f"n {n}: blockwise - whole FFT = {d:.2e} dB")
    assert abs(d) < 0.005
    assert limiter.true_peak_fft(np.zeros((0, 2))) == -math.inf


def test_fft_counts_edge_tails():
    """Audit 05 A4: an abrupt start rings into the silence before the track; the
    blockwise FFT reads it like the whole-signal reference and the 8x meter do."""
    y = np.zeros((1000, 1))
    y[0, 0], y[1, 0] = 1.0, -1.0
    a, ref, tp8 = limiter.true_peak_fft(y), fft_reference_db(y), meter.true_peak(y, FS, 8)
    print(f"abrupt start: blockwise {a:.4f}, whole FFT {ref:.4f}, 8x {tp8:.4f} dBTP")
    assert abs(a - ref) < 0.005
    y = y[::-1].copy()  # abrupt end
    assert abs(limiter.true_peak_fft(y) - fft_reference_db(y)) < 0.005


def test_below_ceiling_no_gain_reduction():
    x = 0.3 * syn.pink_noise(FS, FS, rms_dbfs=-20, seed=1)
    _, info = limiter.apply_with_info(x, FS, LimiterPlan())
    assert info["max_gain_reduction_db"] == 0.0
    assert info["trim_db"] == 0.0 and not info["relimit_passes"]


def test_input_gain():
    x = 0.1 * syn.pink_noise(FS, FS, rms_dbfs=-20, seed=1)
    a = limiter.apply(x, FS, LimiterPlan(input_gain_db=0.0))
    b = limiter.apply(x, FS, LimiterPlan(input_gain_db=6.0))
    assert np.allclose(b, a * 10 ** (6 / 20), atol=1e-12)


# --------------------------------------------------------- onset-masked clipper

def _hot_kick_sub(fs=FS):
    x, meta = syn.kick_sub(fs, bars=2)
    return 1.8 * x, meta["kick_onsets"]


def _check_reduction(u, v, depth_db):
    a, b = np.abs(u), np.abs(v)
    big = a > 1e-9
    worst = float(np.max(20 * np.log10(a[big] / np.maximum(b[big], 1e-300))))
    assert worst <= depth_db + 1e-9, worst
    return worst


@pytest.mark.parametrize("fs", RATES)
def test_clipper_identity_outside_spans_and_bounded_reduction(fs):
    x, onsets = _hot_kick_sub(fs)
    fs8 = fs * 8
    u = meter.upsample(x, 8)
    ceiling = 10 ** (TARGET / 20)
    plan = LimiterPlan(clip_depth_db=3.0, clip_onsets_s=list(onsets))
    v, red, spans = limiter.clip_onsets(u, fs8, plan, ceiling)
    assert 0.0 < red <= 3.0
    touched = np.zeros(u.shape[0], dtype=bool)
    for a0, a1 in spans:
        touched[a0:a1] = True
        p_in, p_out = np.max(np.abs(u[a0:a1])), np.max(np.abs(v[a0:a1]))
        expect = min(3.0, 20 * np.log10(p_in / ceiling))
        assert abs(20 * np.log10(p_in / p_out) - expect) < 1e-9
    assert np.array_equal(v[~touched], u[~touched])
    worst = _check_reduction(u, v, 3.0)
    # the knee starts at most 1 dB x (depth / 3 dB) below the ceiling: samples at or
    # below that floor are untouched
    low = np.abs(u) <= ceiling * 10 ** (-1.0 / 20)
    assert np.array_equal(v[low], u[low])
    assert limiter.clip_onsets(u, fs8, LimiterPlan(), ceiling)[0] is u
    print(f"{fs} Hz clipper: {len(spans)} spans, worst per-sample reduction {worst:.3f} dB")


def test_clipper_ramp_sample_and_overlapping_onsets():
    """Audit 05 A3 cases: a ramp sample above the core peak (was 5.50 dB) and two
    onsets 2 ms apart (was 6.00 dB) both stay within the 3 dB depth."""
    fs8 = FS * 8
    ceiling = 0.5
    plan = LimiterPlan(clip_depth_db=3.0, clip_onsets_s=[0.1])
    u = np.zeros((fs8 // 4, 1))
    c = int(0.1 * fs8)
    pre = int(round(2e-3 * fs8))
    u[c - pre:c + 200, 0] = 0.75
    u[c - pre - 1, 0] = 1.0  # inside the ramp, above the core
    v, red, _ = limiter.clip_onsets(u, fs8, plan, ceiling)
    w1 = _check_reduction(u, v, 3.0)
    u2 = np.zeros((fs8 // 4, 1))
    u2[c:c + 2000, 0] = 1.0  # 6 dB over the ceiling
    v2, red2, spans2 = limiter.clip_onsets(u2, fs8, LimiterPlan(clip_depth_db=3.0,
                                                                clip_onsets_s=[0.1, 0.102]),
                                           ceiling)
    w2 = _check_reduction(u2, v2, 3.0)
    print(f"ramp sample: {w1:.3f} dB; overlapping onsets: {w2:.3f} dB in {len(spans2)} span")
    assert len(spans2) == 1


def test_clipper_spares_the_limiter():
    x, onsets = _hot_kick_sub()
    _, plain = limiter.apply_with_info(x, FS, LimiterPlan(ceiling_dbtp=TARGET))
    y, clipped = limiter.apply_with_info(
        x, FS, LimiterPlan(ceiling_dbtp=TARGET, clip_depth_db=3.0, clip_onsets_s=list(onsets)))
    print(f"kick+sub: limiter GR without clipper {plain['max_gain_reduction_db']:.2f} dB, "
          f"with 3 dB clipper {clipped['max_gain_reduction_db']:.2f} dB")
    assert clipped["max_gain_reduction_db"] < plain["max_gain_reduction_db"] - 2.0
    assert meter.true_peak(y, FS, 8) <= TARGET + 1e-9
    assert fft_reference_db(y) <= TARGET + 0.002
    assert clipped["trim_db"] == 0.0


def test_clipper_window_validation():
    with pytest.raises(ValueError):
        LimiterPlan(clip_pre_ms=1.0, clip_post_ms=5.0)
    with pytest.raises(ValueError):
        LimiterPlan(clip_depth_db=4.0)
    LimiterPlan(clip_pre_ms=4.0, clip_post_ms=8.0)


def test_deterministic():
    x = sweep_plus_impulses(1.0, 2)
    p = LimiterPlan(clip_depth_db=2.0, clip_onsets_s=[0.1, 0.3, 0.55])
    assert np.array_equal(limiter.apply(x, FS, p), limiter.apply(x, FS, p))


@pytest.mark.parametrize("gain", [1.0, 1.2, 1.5, 1.8, 2.5])
def test_small_overs_get_a_soft_knee(gain):
    """Spec 3.5 item 7 (audit 05 D.20): with the knee allowed 1 dB x (depth / 3 dB)
    below the ceiling, no clipped span is a hard clip: in every span the curve's slope
    is continuous (C1), i.e. the knee half-width is positive, and the window peak still
    lands exactly on T."""
    fs8 = FS * 8
    x, onsets = _hot_kick_sub()
    u = meter.upsample(gain * x / 1.8, 8)
    ceiling = 10 ** (TARGET / 20)
    plan = LimiterPlan(clip_depth_db=3.0, clip_onsets_s=list(onsets))
    spans, _ = limiter._spans(plan, fs8, u.shape[0])
    floor = ceiling * 10 ** (-1.0 / 20)
    hard = soft = 0
    for w0, w1, a0, a1 in spans:
        p = float(np.max(np.abs(u[a0:a1])))
        if p <= ceiling:
            continue
        r = min(3.0, 20 * np.log10(p / ceiling))
        t = max(ceiling, p * 10 ** (-r / 20))
        w = min(p - t, t - floor)
        soft, hard = (soft + 1, hard) if w > 0 else (soft, hard + 1)
    v, red, _ = limiter.clip_onsets(u, fs8, plan, ceiling)
    worst = _check_reduction(u, v, 3.0)
    print(f"gain {gain}: peak {20 * np.log10(np.max(np.abs(u)) / ceiling):+.2f} dB over the "
          f"ceiling; spans soft {soft}, hard {hard}; worst reduction {worst:.3f} dB")
    assert hard == 0


@pytest.mark.parametrize("fs", RATES)
def test_transparent_below_ceiling(fs):
    """Spec 3.5 item 7, audit 05 Q1: with no gain reduction the stage is transparent:
    per-bin gain of a below-ceiling log sweep (embedded in silence) within
    resample.TRANSPARENT_DB up to 0.98 x Nyquist. The metering resampler removed
    22.5 to 24 kHz (-0.35 / -6.87 / -31 dB at 22.5 / 23 / 23.5 kHz, 48 kHz)."""
    from eluate.stages import resample
    n = 4 * fs
    t = np.arange(n) / fs
    pad = fs // 2
    core = slice(pad, n - pad)
    x = np.zeros(n)
    tt = t[core] - t[pad]
    fade = np.minimum(1.0, np.minimum(tt, tt[-1] - tt) / 0.01)
    x[core] = 0.25 * signal.chirp(tt, 20, tt[-1], 0.995 * fs / 2, method="logarithmic") * fade
    x2 = np.stack([x, 0.8 * x], axis=1)
    y, info = limiter.apply_with_info(x2, fs, LimiterPlan())
    assert info["max_gain_reduction_db"] == 0.0 and not info["relimit_passes"]
    X, Y = np.fft.rfft(x2[:, 0]), np.fft.rfft(y[:, 0])
    f = np.fft.rfftfreq(n, 1 / fs)
    sig = (np.abs(X) > 1e-3 * np.abs(X).max()) & (f >= 30) & (f <= resample.PASS * fs / 2)
    g = 20 * np.log10(np.abs(Y[sig]) / np.abs(X[sig]))
    at = {v: 20 * np.log10(np.abs(Y[np.argmin(np.abs(f - v))]) / np.abs(X[np.argmin(np.abs(f - v))]))
          for v in (0.9375 * fs / 2, 0.958 * fs / 2, 0.979 * fs / 2)}
    print(f"{fs} Hz below ceiling: per-bin gain {g.min():+.4f}..{g.max():+.4f} dB up to "
          f"{resample.PASS * fs / 2:.0f} Hz; at " +
          ", ".join(f"{k:.0f} Hz {v:+.3f} dB" for k, v in at.items()))
    assert np.max(np.abs(g)) <= resample.TRANSPARENT_DB


def test_processing_resampler_round_trip_bound():
    """The documented bound from the impulse response: zero phase, |gain| <= 0.02 dB
    up to 0.98 x Nyquist, >= 60 dB down in the stopband after one pass."""
    from eluate.stages import resample
    n = 1 << 16
    x = np.zeros(n)
    x[n // 2] = 1.0
    for L in (4, 8):
        y = resample.decimate(resample.upsample(x, L), L)
        H = np.fft.rfft(np.roll(y, -n // 2))
        f = np.fft.rfftfreq(n)  # cycles per sample, Nyquist 0.5
        band = f <= resample.PASS * 0.5
        dev = np.max(np.abs(20 * np.log10(np.abs(H[band]))))
        print(f"{L}x round trip: max |gain| {dev:.4f} dB up to 0.98 Nyquist, "
              f"max phase {np.max(np.abs(np.angle(H[band]))):.1e} rad")
        assert dev <= resample.TRANSPARENT_DB
        assert np.max(np.abs(np.angle(H[band]))) < 1e-9
        h = resample.prototype(L) / L
        w, r = signal.freqz(h, worN=1 << 16)
        stop = w / np.pi >= 1.0 / L
        assert 20 * np.log10(np.max(np.abs(r[stop]))) <= -60.0 + 0.5
