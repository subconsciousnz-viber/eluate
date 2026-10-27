# Portions copied from TRANSMUTATION (audio morph workstation) and from
# drumbusmaximus / SKIN, both by the same author.
# Copyright (C) 2026 subconsciousnz-viber
# SPDX-License-Identifier: GPL-3.0-or-later
"""Detector code borrowed from the maintainer's sibling repositories (docs/reuse-survey.md).

Origins (copied, not imported; the maintainer holds the copyright):
- `detect_onsets`, `superflux`: TRANSMUTATION engine/transients.py `detect_onsets`
  (rebuilt 2026-08-18): log-compressed SuperFlux, multiplicative local-median
  threshold plus a global salience gate, 64-sample attack refinement, 50 ms gap.
  Logic unchanged. The per-frame rfft loop is vectorised in blocks and the flux
  computation is split out as `superflux` so the envelope can be reported as
  onset strength; `detect_onsets(x, sr)` returns the same sample indices as the
  original (tested against a verbatim copy).
- `detect_onsets_fb`: the same detector with a 12-per-octave log filterbank
  before the log compression and a local (+-4 s) threshold floor, as spec 3.3
  draft 0.4 asks (audit 04 A11). Measured worse on the gate fixtures (lower
  recall with the filterbank, lower precision with the local floor; brief-04
  D.21), so transients.py does not use it.
- `match_onsets`: TRANSMUTATION engine/fusion/onset_match.py, verbatim logic:
  greedy closest-first one-to-one matching inside a tolerance, precision /
  recall / coverage / matched-pair error. Greedy, not minimum-cost (Hungarian):
  exact when the tolerance is under half the event spacing, which holds for
  every gate here (no two reference onsets within 40 ms; audit 04 F17).
- `harmonic_salience`: the core of TRANSMUTATION engine/fusion/metrics.py
  `pitch_candidates` (harmonic sum S(f) = sum_h mag(h f) / h on a 10-cent grid,
  completeness = share of expected harmonics above -40 dB of the peak, score =
  S x completeness, peaks merged within 90 cents, subharmonic phantoms kept only
  if their own fundamental is above -30 dB of the peak). Adapted to run per
  frame on a magnitude matrix: the original takes the median spectrum of a
  clip's centre 50 %, which cannot follow a bassline beat by beat. Returns the
  best surviving candidate per frame and its score.
- `ladder_fraction`: TRANSMUTATION metrics.py `harmonic_noise_ratio`, i.e. the
  share of energy on a harmonic ladder (bands of +-60 cents around h f0, up to
  24 harmonics) normalised by the share the ladder covers (white noise 0, a pure
  ladder 1). Adapted to a given power spectrum with arbitrary bin widths (the
  caller passes a baseline-whitened spectrum, because the area normalisation
  assumes a flat noise floor) and to an optional excluded band (the candidate
  itself).
- `nsdf_f0`: drumbusmaximus tools/py/metrics/pitch.py `frame_f0` (McLeod-Wyvill
  normalised square difference function, first NSDF peak within 0.90 of the
  highest, parabolic refinement; confidence = the NSDF value there). Logic
  unchanged; used by pitch.py as the independent per-beat estimate for the
  keyeq gate (audit 04 F2).
- `attack_low_share`, `KICK_LOW_SHARE`: drumbusmaximus analysis/src/Onsets.cpp
  `measureOnsetTimbre` (attack = sqrt of the per-band power rise, Hann window of
  about 46 ms after the onset against the same window before it, the after
  window stopping at the next onset, 24 log bands from 30 Hz to 16 kHz) and
  analysis/src/DrumIdentity.cpp (a drum with >= 0.3 of its unit-sum attack shape
  in the bands below 150 Hz is low, i.e. a kick). DrumIdentity applies the rule
  to NMF-separated drum shapes; here it is applied to each onset's own attack,
  so a kick struck together with a loud hat reads a lower share. Added
  `lead_s`: both windows may start that long before the onset (transients.py
  uses 5 ms; the C++ uses 0, where the Hann window gives the attack sample no
  weight), and `mode`: "power" (default, audit 04 A11: each bin's power rise
  split over the bands by fractional overlap, share of power) or "sqrt" (the
  C++: integer binning, which leaves two of the six low bands empty at 48 kHz,
  and the share of per-band sqrt magnitudes). The power share runs higher, so
  the 0.3 threshold belongs to "sqrt" only; transients.py maps the power share
  with its own measured ramp.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import get_window

# ---------------------------------------------------------------- SuperFlux

_BLOCK = 4096  # frames per rfft block


def superflux(audio, sr, hop: int = 512, n_fft: int = 2048):
    """SuperFlux envelope of TRANSMUTATION's detect_onsets. Returns (flux, n_frames)
    with flux[i] belonging to the frame whose window ENDS at original sample i*hop
    (the signal is front-padded by n_fft); None when the input is too short or silent."""
    x = np.asarray(audio, dtype=np.float64)
    if x.size < 2 * hop or not np.any(np.abs(x) > 1e-9):
        return None
    win = get_window('hann', n_fft, fftbins=True)
    xp = np.pad(x, (n_fft, hop))
    n_frames = 1 + (len(xp) - n_fft) // hop
    if n_frames < 3:
        return None
    view = np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::hop][:n_frames]
    mags = np.empty((n_frames, n_fft // 2 + 1))
    for a in range(0, n_frames, _BLOCK):
        mags[a:a + _BLOCK] = np.abs(np.fft.rfft(view[a:a + _BLOCK] * win, axis=1))
    peak = float(np.max(mags))
    if peak <= 1e-12:
        return None
    L = np.log1p(1000.0 * mags / peak)
    prev = np.maximum(np.maximum(L[:-1, :-2], L[:-1, 1:-1]), L[:-1, 2:])
    d = np.maximum(L[1:, 1:-1] - prev, 0.0)
    flux = np.concatenate([[0.0], d.sum(axis=1)])
    return flux, n_frames


def detect_onsets(audio, sr, hop: int = 512, n_fft: int = 2048):
    """Salience-gated SuperFlux onsets (TRANSMUTATION). Sorted sample indices."""
    x = np.asarray(audio, dtype=np.float64)
    sf = superflux(x, sr, hop, n_fft)
    if sf is None:
        return np.array([], dtype=int)
    flux, n_frames = sf
    fmax = float(np.max(flux))
    if fmax <= 1e-9:
        return np.array([], dtype=int)
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

    env_w = 64
    kernel = np.ones(env_w) / env_w
    onsets = []
    for i in cand:
        end = i * hop
        s0 = max(0, end - n_fft - env_w)
        s1 = min(len(x), end + hop)
        if s1 - s0 < 2 * env_w:
            onsets.append(max(0, min(end, len(x) - 1)))
            continue
        seg = np.convolve(x[s0:s1] ** 2, kernel, mode='same')
        rise = np.diff(seg)
        onsets.append(s0 + int(np.argmax(rise)))
    onsets = sorted(set(onsets))
    gap = int(round(0.05 * sr))
    out = []
    for s in onsets:
        if not out or s - out[-1] >= gap:
            out.append(s)
    return np.array(out, dtype=int)


def log_filterbank(n_fft: int, sr: float, bands_per_octave: int = 12, f_lo: float = 30.0,
                   f_hi: float = 16000.0) -> np.ndarray:
    """(bands, bins) triangular filters equally spaced in log frequency (centres
    f_lo * 2^(k / bands_per_octave)), each normalised to unit area; bins narrower
    than a filter are split by fractional overlap (every band non-empty)."""
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    f_hi = min(f_hi, 0.95 * sr / 2.0)
    k = np.arange(int(np.floor(bands_per_octave * np.log2(f_hi / f_lo))) + 1)
    c = f_lo * 2.0 ** (k / bands_per_octave)
    lo = c * 2.0 ** (-1.0 / bands_per_octave)
    hi = c * 2.0 ** (1.0 / bands_per_octave)
    w = np.maximum(0.0, np.minimum((f[None, :] - lo[:, None]) / (c - lo)[:, None],
                                   (hi[:, None] - f[None, :]) / (hi - c)[:, None]))
    df = f[1] - f[0]
    empty = w.sum(axis=1) <= 0
    for i in np.nonzero(empty)[0]:  # filter narrower than a bin: nearest bin
        w[i, int(np.argmin(np.abs(f - c[i])))] = 1.0
    return w / np.maximum(w.sum(axis=1, keepdims=True), 1e-30)


def detect_onsets_fb(audio, sr, hop: int = 512, n_fft: int = 2048, bands_per_octave: int = 12,
                     local_s: float = 4.0, env_w: int = 64):
    """SuperFlux as in the literature (Boeck and Widmer 2013): the TRANSMUTATION
    detector with two changes (spec 3.3 draft 0.4, audit 04 A11): the magnitude
    spectrum goes through a log-frequency triangular filterbank before log
    compression and the 3-band max filter, so a kick's few low bins weigh as
    much per octave as a hat's many high ones; and the threshold floor is
    0.07 x the maximum flux within +-local_s instead of the whole track's, so
    a quiet intro keeps its onsets. Refinement, gap and the local-median term
    are unchanged. `env_w` is the refinement envelope length in samples."""
    x = np.asarray(audio, dtype=np.float64)
    if x.size < 2 * hop or not np.any(np.abs(x) > 1e-9):
        return np.array([], dtype=int)
    win = get_window('hann', n_fft, fftbins=True)
    xp = np.pad(x, (n_fft, hop))
    n_frames = 1 + (len(xp) - n_fft) // hop
    if n_frames < 3:
        return np.array([], dtype=int)
    fb = log_filterbank(n_fft, sr, bands_per_octave)
    view = np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::hop][:n_frames]
    mags = np.empty((n_frames, fb.shape[0]))
    for a in range(0, n_frames, _BLOCK):
        mags[a:a + _BLOCK] = np.abs(np.fft.rfft(view[a:a + _BLOCK] * win, axis=1)) @ fb.T
    peak = float(np.max(mags))
    if peak <= 1e-12:
        return np.array([], dtype=int)
    L = np.log1p(1000.0 * mags / peak)
    prev = np.maximum(np.maximum(L[:-1, :-2], L[:-1, 1:-1]), L[:-1, 2:])
    d = np.maximum(L[1:, 1:-1] - prev, 0.0)
    flux = np.concatenate([[0.0], d.sum(axis=1)])
    if float(np.max(flux)) <= 1e-9:
        return np.array([], dtype=int)
    from scipy.ndimage import maximum_filter1d
    med_win = max(3, int(round(0.25 * sr / hop)) | 1)
    med = median_filter(flux, size=med_win, mode='nearest')
    loc = maximum_filter1d(flux, size=2 * int(round(local_s * sr / hop)) + 1, mode='nearest')
    thresh = 1.5 * med + 0.07 * loc + 1e-12
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
    kernel = np.ones(env_w) / env_w
    onsets = []
    for i in cand:
        end = i * hop
        s0 = max(0, end - n_fft - env_w)
        s1 = min(len(x), end + hop)
        if s1 - s0 < 2 * env_w:
            onsets.append(max(0, min(end, len(x) - 1)))
            continue
        seg = np.convolve(x[s0:s1] ** 2, kernel, mode='same')
        onsets.append(s0 + int(np.argmax(np.diff(seg))))
    onsets = sorted(set(onsets))
    gap = int(round(0.05 * sr))
    out = []
    for s_ in onsets:
        if not out or s_ - out[-1] >= gap:
            out.append(s_)
    return np.array(out, dtype=int)


# ------------------------------------------------------------ onset matching

def match_onsets(ref, out, sr: int, tol_ms: float) -> dict:
    """One-to-one onset matching within tol_ms (TRANSMUTATION onset_match)."""
    r = np.sort(np.asarray(ref, dtype=np.float64).ravel())
    o = np.sort(np.asarray(out, dtype=np.float64).ravel())
    tol = float(tol_ms) * 1e-3 * float(sr)
    n_ref, n_out = int(r.size), int(o.size)
    cands = []
    for i in range(n_ref):
        lo = int(np.searchsorted(o, r[i] - tol, side='left'))
        hi = int(np.searchsorted(o, r[i] + tol, side='right'))
        for j in range(lo, hi):
            cands.append((abs(r[i] - o[j]), i, j))
    cands.sort()
    used_r = np.zeros(n_ref, dtype=bool)
    used_o = np.zeros(n_out, dtype=bool)
    pairs, errs = [], []
    for d, i, j in cands:
        if used_r[i] or used_o[j]:
            continue
        used_r[i] = used_o[j] = True
        pairs.append((int(r[i]), int(o[j])))
        errs.append(d / float(sr) * 1000.0)
    pairs.sort()
    matched = len(pairs)
    nan = float('nan')
    return {
        'matched': matched, 'missed': n_ref - matched, 'extra': n_out - matched,
        'n_ref': n_ref, 'n_out': n_out,
        'recall': (matched / n_ref) if n_ref else 1.0,
        'precision': (matched / n_out) if n_out else 1.0,
        'coverage': (matched / max(n_ref, n_out)) if max(n_ref, n_out) else 1.0,
        'mean_err_ms': float(np.mean(errs)) if errs else nan,
        'max_err_ms': float(np.max(errs)) if errs else nan,
        'matched_pairs': pairs,
        'missed_ref': [int(v) for v in r[~used_r]],
        'extra_out': [int(v) for v in o[~used_o]],
    }


# ------------------------------------------------------- harmonic salience

def _interp_rows(mag: np.ndarray, freqs_hz: np.ndarray, sr: float, n_fft: int) -> np.ndarray:
    """Row-wise linear magnitude lookup (TRANSMUTATION _interp_mag, per frame)."""
    bins = freqs_hz * n_fft / sr
    k0 = np.floor(bins).astype(int)
    frac = bins - k0
    k0 = np.clip(k0, 0, mag.shape[1] - 1)
    k1 = np.clip(k0 + 1, 0, mag.shape[1] - 1)
    return (1.0 - frac) * mag[:, k0] + frac * mag[:, k1]


def harmonic_salience(mag: np.ndarray, sr: float, n_fft: int, *, fmin: float, fmax: float,
                      n_harmonics: int = 8, cents_step: float = 10.0,
                      min_sep_cents: float = 90.0, sub_floor_db: float = -30.0,
                      present_db: float = -40.0, max_keep: int = 8,
                      return_dominance: bool = False):
    """Per-frame pitch_candidates core. `mag` is (frames, n_fft // 2 + 1).

    Returns (f0_hz, score) per frame: the best surviving candidate's frequency and
    its salience x completeness, normalised per frame to the frame's own magnitude
    peak (NaN / 0 for silent frames). With `return_dominance`, also
    1 - second-best / best score per frame (1 when only one candidate survives)."""
    n_steps = max(2, int(np.ceil(1200.0 * np.log2(fmax / fmin) / cents_step)))
    grid = fmin * 2.0 ** (np.arange(n_steps) * cents_step / 1200.0)
    harm = np.arange(1, n_harmonics + 1, dtype=np.float64)
    hf = grid[:, None] * harm[None, :]
    valid = hf < sr * 0.5
    nf = mag.shape[0]
    f_out = np.full(nf, np.nan)
    s_out = np.zeros(nf)
    d_out = np.zeros(nf)
    pk = mag.max(axis=1)
    good = pk > 1e-14
    if not np.any(good):
        return (f_out, s_out, d_out) if return_dominance else (f_out, s_out)
    m = np.where(good[:, None], mag / np.where(good, pk, 1.0)[:, None], 0.0)
    floor = 10.0 ** (present_db / 20.0)
    sub_floor = 10.0 ** (sub_floor_db / 20.0)
    hmag = _interp_rows(m, np.clip(hf, 0.0, sr * 0.5 - 1.0).ravel(), sr, n_fft).reshape(nf, *hf.shape)
    hmag = np.where(valid[None], hmag, 0.0)
    sal = (hmag / harm[None, None, :]).sum(axis=2)
    comp = (hmag > floor).sum(axis=2) / np.maximum(valid.sum(axis=1), 1)[None, :]
    score = sal * comp
    own = _interp_rows(m, grid, sr, n_fft)  # magnitude at each candidate's fundamental
    is_pk = np.zeros_like(sal, dtype=bool)
    is_pk[:, 1:-1] = (sal[:, 1:-1] > sal[:, :-2]) & (sal[:, 1:-1] >= sal[:, 2:])
    for i in np.nonzero(good)[0]:
        idx = np.nonzero(is_pk[i])[0]
        if idx.size == 0:
            idx = np.array([int(np.argmax(sal[i]))])
        sc = score[i, idx]
        order = np.argsort(sc)[::-1]
        kept = []
        for gi in idx[order]:
            f = grid[gi]
            if any(abs(1200.0 * np.log2(f / grid[k])) < min_sep_cents for k in kept):
                continue
            is_sub = False
            for k in kept:
                up = grid[k]
                if up <= f:
                    continue
                ratio = up / f
                mult = int(round(ratio))
                if mult >= 2 and abs(1200.0 * np.log2(ratio / mult)) < min_sep_cents:
                    is_sub = True
                    break
            if is_sub and own[i, gi] < sub_floor:
                continue
            kept.append(gi)
            if len(kept) >= max_keep:
                break
        if kept:
            ks = sorted((score[i, k] for k in kept), reverse=True)
            best = max(kept, key=lambda k: score[i, k])
            f_out[i] = grid[best]
            s_out[i] = score[i, best]
            d_out[i] = 1.0 - (ks[1] / ks[0]) if len(ks) > 1 and ks[0] > 0 else 1.0
    return (f_out, s_out, d_out) if return_dominance else (f_out, s_out)


# ------------------------------------------------------------ ladder share

def ladder_fraction(freqs: np.ndarray, power: np.ndarray, widths: np.ndarray, f0: float, *,
                    n_harmonics: int = 24, width_cents: float = 60.0,
                    exclude: tuple[float, float] | None = None, min_bins: float = 0.0) -> float:
    """Area-normalised share of power on the harmonic ladder of f0 (TRANSMUTATION
    harmonic_noise_ratio): (raw - area) / (1 - area), clipped to [0, 1]. `widths`
    are the bin widths in Hz (bins of several resolutions may be mixed); bins inside
    `exclude` = (lo, hi) Hz are left out of both the ladder and the total.
    `min_bins` (an addition) keeps each harmonic band at least that many bin
    widths wide, so a narrow cents tolerance still holds the nearest bin."""
    keep = np.ones(freqs.shape, dtype=bool)
    if exclude is not None:
        keep &= ~((freqs >= exclude[0]) & (freqs <= exclude[1]))
    f, p, w = freqs[keep], power[keep], widths[keep]
    tot_w = float(np.sum(w))
    total = float(np.sum(p * w))
    if total <= 1e-30 or tot_w <= 0:
        return 0.0
    half = 2.0 ** (width_cents / 1200.0)
    mask = np.zeros(f.shape, dtype=bool)
    top = float(f.max())
    for h in range(1, n_harmonics + 1):
        fh = f0 * h
        if fh >= top:
            break
        tol = np.maximum(fh * (half - 1.0), min_bins * w)  # at least min_bins of each bin's width
        mask |= np.abs(f - fh) <= tol
    raw = float(np.sum(p[mask] * w[mask]) / total)
    area = float(np.sum(w[mask]) / tot_w)
    if area >= 1.0 - 1e-9:
        return raw
    return float(np.clip((raw - area) / (1.0 - area), 0.0, 1.0))


# ------------------------------------------------------------- kick rule

KICK_LOW_SHARE = 0.3       # DrumIdentity kKickLow
ATTACK_BANDS = 24          # Onsets.h kAttackBands
ATTACK_LO_HZ, ATTACK_HI_HZ = 30.0, 16000.0
ATTACK_LOW_BANDS = 6       # DrumIdentity kLowBands: the bands below 150 Hz


def attack_low_share(mono: np.ndarray, sr: float, onsets: np.ndarray, lead_s: float = 0.0,
                     mode: str = "power"):
    """Per onset (sample index), the share of its attack in the bands below
    150 Hz (Onsets.cpp measureOnsetTimbre + DrumIdentity lowShare).
    `lead_s` (an addition, 0 = the C++) starts both windows that much before
    the onset. `mode="power"` (default, audit 04 A11): each bin's power rise is
    split over the bands by fractional overlap and the share is a power share;
    `mode="sqrt"` is the C++: integer binning (at 48 kHz two of the six low
    bands hold no bin) and the share of the per-band sqrt magnitudes."""
    x = np.asarray(mono, dtype=np.float64)
    ns = x.size
    n = int(2 ** round(np.log2(0.0464 * sr)))  # about 46 ms (2048 at 44.1 / 48 kHz)
    win = np.hanning(n + 1)[:-1]
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    edges = ATTACK_LO_HZ * (ATTACK_HI_HZ / ATTACK_LO_HZ) ** (np.arange(ATTACK_BANDS + 1) / ATTACK_BANDS)
    band = np.searchsorted(edges, freqs, side="right") - 1
    band[(freqs < ATTACK_LO_HZ) | (freqs >= ATTACK_HI_HZ)] = -1
    band[0] = -1
    band[-1] = -1  # Onsets.cpp sums bins 1 .. N/2 - 1
    df = freqs[1] - freqs[0]
    ov = np.clip(np.minimum(freqs[None, :] + df / 2, edges[1:, None])
                 - np.maximum(freqs[None, :] - df / 2, edges[:-1, None]), 0.0, None) / df
    ov[:, 0] = 0.0
    ov[:, -1] = 0.0
    onsets = np.asarray(onsets, dtype=np.int64)
    out = np.full(onsets.size, np.nan)
    lead = int(round(lead_s * sr))
    for i, s in enumerate(onsets):
        s = max(0, int(s) - lead)
        nxt = onsets[i + 1] if i + 1 < onsets.size else ns
        post = np.zeros(n)
        a1 = min(ns, s + n, max(s, nxt))
        post[:max(0, a1 - s)] = x[s:a1]
        pre = np.zeros(n)
        b0 = max(0, s - n)
        pre[n - (s - b0):] = x[b0:s]
        mp = np.abs(np.fft.rfft(post * win)) ** 2
        mq = np.abs(np.fft.rfft(pre * win)) ** 2
        r = np.maximum(0.0, mp - mq)
        if mode == "power":
            rise = ov @ r
            att = rise
        else:
            rise = np.zeros(ATTACK_BANDS)
            ok = band >= 0
            np.add.at(rise, band[ok], r[ok])
            att = np.sqrt(rise)
        tot = att.sum()
        if tot > 0:
            out[i] = att[:ATTACK_LOW_BANDS].sum() / tot
    return out


# ------------------------------------------------------------------ NSDF

def _parabolic_peak(y: np.ndarray, i: int) -> tuple[float, float]:
    if i <= 0 or i >= y.size - 1:
        return float(i), float(y[i])
    a, b, c = float(y[i - 1]), float(y[i]), float(y[i + 1])
    denom = a - 2.0 * b + c
    if denom == 0.0:
        return float(i), b
    delta = 0.5 * (a - c) / denom
    return float(i) + delta, b - 0.25 * (a - c) * delta


def nsdf_f0(frame: np.ndarray, sample_rate: float, f_min: float = 30.0, f_max: float = 2000.0,
            peak_ratio: float = 0.90) -> tuple[float, float]:
    """(f0_hz, confidence) of one frame by NSDF (drumbusmaximus frame_f0)."""
    x = np.asarray(frame, dtype=np.float64)
    x = np.where(np.isfinite(x), x, 0.0)
    if x.size < 8:
        return float("nan"), 0.0
    x = x - x.mean()
    if float(np.dot(x, x)) <= 0.0:
        return float("nan"), 0.0
    n = x.size
    lag_min = max(2, int(np.floor(sample_rate / f_max)))
    lag_max = min(n - 2, int(np.ceil(sample_rate / f_min)))
    if lag_max <= lag_min:
        return float("nan"), 0.0
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(x, nfft)
    r = np.fft.irfft(spec * np.conj(spec), nfft)[:n]
    if r[0] <= 0:
        return float("nan"), 0.0
    sq = x * x
    prefix = np.concatenate(([0.0], np.cumsum(sq)))
    lags = np.arange(n)
    m = prefix[n - lags] + (prefix[n] - prefix[lags])
    with np.errstate(divide="ignore", invalid="ignore"):
        nsdf = np.where(m > 0, 2.0 * r / m, 0.0)
    seg = nsdf[lag_min:lag_max + 1]
    if seg.size < 3:
        return float("nan"), 0.0
    interior = np.flatnonzero((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:])) + 1
    if interior.size == 0:
        return float("nan"), 0.0
    best = float(seg[interior].max())
    if best <= 0.0:
        return float("nan"), 0.0
    chosen = int(interior[np.flatnonzero(seg[interior] >= peak_ratio * best)[0]]) + lag_min
    lag, peak = _parabolic_peak(nsdf, chosen)
    if lag <= 0:
        return float("nan"), 0.0
    return float(sample_rate / lag), float(max(0.0, min(1.0, peak)))
