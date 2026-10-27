# Reference functions copied from drumbusmaximus (SKIN verification lane D),
# tools/py/metrics/{stereo,spectral,pitch}.py, and from TRANSMUTATION
# engine/transients.py (detect_onsets, verbatim, as ref_detect_onsets).
# Copyright (C) 2026 subconsciousnz-viber
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cross-checks of features/ against independent implementations from the maintainer's
drumbusmaximus repository (docs/reuse-survey.md item 4). The reference functions
below are copied verbatim in logic (comments trimmed) and used only here:
`correlation` (stereo.py), `magnitude_spectrum` + `dominant_peak` (spectral.py)
and `frame_f0` (pitch.py, McLeod NSDF with first-peak selection)."""

import numpy as np
import pytest

from fixtures import synthetic as syn
from eluate.features import borrowed, pitch, resonance, stereo


# ------------------------------------------------- copied reference functions

def ref_correlation(left, right):
    l = np.asarray(left, dtype=np.float64)
    r = np.asarray(right, dtype=np.float64)
    n = min(l.size, r.size)
    l, r = l[:n], r[:n]
    ok = np.isfinite(l) & np.isfinite(r)
    if not ok.any():
        return 0.0
    l = l[ok] - np.mean(l[ok])
    r = r[ok] - np.mean(r[ok])
    denom = np.sqrt(float(np.dot(l, l)) * float(np.dot(r, r)))
    if denom <= 0:
        return 0.0
    return float(np.dot(l, r) / denom)


def ref_magnitude_spectrum(x, sample_rate, n_fft=None):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(n_fft or x.size)
    if n > x.size:
        x = np.pad(x, (0, n - x.size))
    else:
        x = x[:n]
    w = np.hanning(n)
    mag = np.abs(np.fft.rfft(x * w)) * (2.0 / max(np.sum(w), 1e-30))
    return np.fft.rfftfreq(n, 1.0 / sample_rate), mag


def ref_dominant_peak(x, sample_rate, f_min, f_max):
    freqs, mag = ref_magnitude_spectrum(x, sample_rate)
    band = (freqs >= f_min) & (freqs <= f_max)
    idx = np.flatnonzero(band)
    i = int(idx[int(np.argmax(mag[band]))])
    a, b, c = float(mag[i - 1]), float(mag[i]), float(mag[i + 1])
    denom = a - 2.0 * b + c
    delta = float(np.clip(0.5 * (a - c) / denom if denom != 0.0 else 0.0, -0.5, 0.5))
    peak_hz = float(freqs[i]) + delta * float(freqs[1] - freqs[0])
    med = float(np.median(mag[band]))
    return {"hz": peak_hz, "prominence_db": float(20.0 * np.log10(mag[i] / med))}


def _parabolic_peak(y, i):
    if i <= 0 or i >= y.size - 1:
        return float(i), float(y[i])
    a, b, c = float(y[i - 1]), float(y[i]), float(y[i + 1])
    denom = a - 2.0 * b + c
    if denom == 0.0:
        return float(i), b
    delta = 0.5 * (a - c) / denom
    return float(i) + delta, b - 0.25 * (a - c) * delta


def ref_frame_f0(frame, sample_rate, f_min=30.0, f_max=2000.0, peak_ratio=0.90):
    x = np.asarray(frame, dtype=np.float64)
    x = x - x.mean()
    if float(np.dot(x, x)) <= 0.0:
        return float("nan"), 0.0
    n = x.size
    lag_min = max(2, int(np.floor(sample_rate / f_max)))
    lag_max = min(n - 2, int(np.ceil(sample_rate / f_min)))
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(x, nfft)
    r = np.fft.irfft(spec * np.conj(spec), nfft)[:n]
    sq = x * x
    prefix = np.concatenate(([0.0], np.cumsum(sq)))
    lags = np.arange(n)
    m = prefix[n - lags] + (prefix[n] - prefix[lags])
    with np.errstate(divide="ignore", invalid="ignore"):
        nsdf = np.where(m > 0, 2.0 * r / m, 0.0)
    seg = nsdf[lag_min:lag_max + 1]
    interior = np.flatnonzero((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:])) + 1
    if interior.size == 0:
        return float("nan"), 0.0
    best = float(seg[interior].max())
    if best <= 0.0:
        return float("nan"), 0.0
    chosen = int(interior[np.flatnonzero(seg[interior] >= peak_ratio * best)[0]]) + lag_min
    lag, peak = _parabolic_peak(nsdf, chosen)
    return float(sample_rate / lag), float(max(0.0, min(1.0, peak)))


# ---------------------------------------- TRANSMUTATION detect_onsets, verbatim

def ref_detect_onsets(audio, sr, hop=512, n_fft=2048):
    """Salience-gated spectral-flux onset detector. Returns sorted sample
    indices.

    REBUILT 2026-08-18 (tri-audit C). The old function was
    librosa.onset_detect with an ABSOLUTE delta on a material-dependent
    envelope, falling back to a home-rolled detector when librosa was
    absent — and the two paths disagreed by 25 events on the same input.
    Measured: 25 onsets in 4 s of a pure 440 Hz sine, 20 in stationary
    white noise, a click at t=0 missed entirely, and every detection ~11 ms
    (one hop) late. Downstream, stft._if_morph discarded its accumulated
    phase several times a second on SUSTAINED material, transient_envelope
    yanked alpha at phantom events, and build_onset_warp fed phantom
    anchors to the DTW.

    Now ONE implementation, no librosa:
    * log-compressed superflux (per-bin max-filtered previous frame, so
      vibrato is not an onset);
    * MULTIPLICATIVE local-median threshold plus a global salience gate —
      a stationary signal's flux sits at its own median and can never
      fire (pinned: 0 onsets on a pure sine AND on stationary noise);
    * sample-level refinement of each detection to the local energy-rise
      point, so the reported index is the attack, not the frame grid
      (pinned: click-train timing error <= 3 ms, t=0 included).
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.size < 2 * hop or not np.any(np.abs(x) > 1e-9):
        return np.array([], dtype=int)
    from numpy.fft import rfft
    from scipy.signal import get_window
    win = get_window('hann', n_fft, fftbins=True)
    # Front-pad a full window of silence so an attack at t=0 has an
    # all-silence predecessor frame to flux against — the old detector
    # missed a click at sample 0 outright. Frame i then covers original
    # samples [i*hop - n_fft, i*hop): its END is at i*hop.
    xp = np.pad(x, (n_fft, hop))
    n_frames = 1 + (len(xp) - n_fft) // hop
    if n_frames < 3:
        return np.array([], dtype=int)
    nbins = n_fft // 2 + 1
    mags = np.empty((n_frames, nbins))
    for i in range(n_frames):
        s = i * hop
        mags[i] = np.abs(rfft(xp[s:s + n_fft] * win))
    peak = float(np.max(mags))
    if peak <= 1e-12:
        return np.array([], dtype=int)
    L = np.log1p(1000.0 * mags / peak)
    # Superflux: positive difference against the per-bin MAX of the
    # previous frame's 3-bin neighbourhood.
    prev = np.maximum(np.maximum(L[:-1, :-2], L[:-1, 1:-1]), L[:-1, 2:])
    d = np.maximum(L[1:, 1:-1] - prev, 0.0)
    flux = np.concatenate([[0.0], d.sum(axis=1)])
    fmax = float(np.max(flux))
    if fmax <= 1e-9:
        return np.array([], dtype=int)
    from scipy.ndimage import median_filter
    med_win = max(3, int(round(0.25 * sr / hop)) | 1)
    med = median_filter(flux, size=med_win, mode='nearest')
    thresh = 1.5 * med + 0.07 * fmax + 1e-12
    min_gap = max(1, int(round(0.05 * sr / hop)))

    cand = []
    last = -min_gap
    for i in range(1, n_frames - 1):
        if (flux[i] > thresh[i] and flux[i] >= flux[i - 1]
                and flux[i] >= flux[i + 1] and (i - last) >= min_gap):
            cand.append(i)
            last = i

    if not cand:
        return np.array([], dtype=int)

    # Sample-level refinement: the flux frame only brackets the attack
    # (its window covers the n_fft samples BEFORE its end); take the
    # steepest rise of a short energy envelope inside that span.
    env_w = 64
    kernel = np.ones(env_w) / env_w
    onsets = []
    for i in cand:
        end = i * hop             # end-of-window position, original coords
        s0 = max(0, end - n_fft - env_w)
        s1 = min(len(x), end + hop)
        if s1 - s0 < 2 * env_w:
            onsets.append(max(0, min(end, len(x) - 1)))
            continue
        seg = np.convolve(x[s0:s1] ** 2, kernel, mode='same')
        rise = np.diff(seg)
        onsets.append(s0 + int(np.argmax(rise)))
    onsets = sorted(set(onsets))
    # Re-apply the gap on the refined positions.
    gap = int(round(0.05 * sr))
    out = []
    for s in onsets:
        if not out or s - out[-1] >= gap:
            out.append(s)
    return np.array(out, dtype=int)


# ------------------------------------------------------------- cross-checks

FS = 48000


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9, -0.4])
def test_broadband_correlation_matches_reference(rho):
    """Our pooled per-window Pearson coefficient vs drumbusmaximus's whole-file one."""
    x = syn.noise_psd_slope(FS * 10, FS, -3.0103, correlation=rho, seed=11)
    ours = stereo.analyse(x, FS)["full_corr_summary"]
    assert abs(ours - ref_correlation(x[:, 0], x[:, 1])) < 0.01


def test_resonance_f0_matches_dominant_peak():
    """The gate ring's f0 vs drumbusmaximus dominant_peak on the ring section alone
    (a single FFT over 8 s: 0.125 Hz bins)."""
    x = syn.pink_noise(FS * 20, FS, seed=3)
    y, _ = syn.add_ring(x, FS, 7000.0, 12.0, 9.0, 6.0, 14.0)
    ours = [r for r in resonance.analyse(y, FS)["resonances"] if r["flag"] == "resonance"][0]["f0"]
    seg = 0.5 * (y[6 * FS:14 * FS, 0] + y[6 * FS:14 * FS, 1])
    ref = ref_dominant_peak(seg, FS, 6000.0, 8000.0)
    # the boost's top is flat to within the noise over about +-1 % of f0; both
    # estimators land on it, at different points
    assert abs(ours / ref["hz"] - 1.0) < 0.01


def test_bass_root_matches_nsdf_second_halves():
    """Our per-beat root vs drumbusmaximus NSDF f0 on the second half of each beat
    (the part the spec weights), low-passed the same way."""
    fs = 44100
    x, meta = syn.kick_sub(fs, bars=8)
    br = pitch.bass_root(x, fs, meta["beat_times"], meta["kick_onsets"])
    lo = pitch._bass_signals(x, fs)[2]  # M, low-passed and resampled to 4 kHz
    beat = np.median(np.diff(meta["beat_times"]))
    agree = 0
    for b, root in zip(meta["beat_times"], br["root_midi"]):
        a = int((b + 0.55 * beat) * pitch.BASS_FS)
        e = int((b + 0.95 * beat) * pitch.BASS_FS)
        f0, conf = ref_frame_f0(lo[a:e], pitch.BASS_FS, 30.0, 250.0)
        if conf > 0.5 and round(69 + 12 * np.log2(f0 / 440.0)) == root:
            agree += 1
    assert agree / len(br["root_midi"]) >= 0.95


@pytest.mark.parametrize("name", ["kick_sub", "arrangement", "pink", "clicks"])
def test_superflux_port_is_bit_identical(name):
    """borrowed.detect_onsets (vectorised rfft) returns the same sample indices as
    the verbatim TRANSMUTATION function above (audit 04 F16)."""
    fs = 44100
    if name == "kick_sub":
        y = syn.kick_sub(fs, bars=4)[0][:, 0]
    elif name == "arrangement":
        y = syn.sectioned_track(fs, levels_db=(0.0, -6.0))[0][:, 0]
    elif name == "pink":
        y = syn.pink_noise(fs * 4, fs)[:, 0]
    else:
        y = syn.impulse_train(fs * 4, fs, 0.25, 0.5)[:, 0] + syn.pink_noise(fs * 4, fs, rms_dbfs=-60)[:, 0]
    a = ref_detect_onsets(y, fs)
    b = borrowed.detect_onsets(y, fs)
    assert a.size > 0 and np.array_equal(a, b)
