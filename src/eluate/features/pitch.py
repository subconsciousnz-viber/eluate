"""Key, chord chroma and bass-root tracking (spec 3.3 row "pitch").

Contract:
- Chroma: librosa.feature.chroma_cqt of the mono analysis signal at AR =
  20480 Hz, hop 2048 (100 ms, aligned to the common grid), librosa defaults
  otherwise (per-frame max normalisation, tuning estimated).
- Key: the frame chroma averaged with weights = frame power (from the
  momentary-loudness series when given, else the frame RMS of the analysis
  signal; silent frames weigh 0), then Pearson-correlated with the 12
  rotations of each profile. Profile "edmm" (Faraldo et al. 2017, values as
  shipped in Essentia's Key algorithm, src/algorithms/tonal/key.cpp): its major
  profile is flat by design ("it will report major modes ... as minor"), so its
  correlation is undefined and only minor keys are candidates. "edma" (the
  automatic major/minor pair from the same work) has a real major profile.
  Combined key (spec 3.3 draft 0.4, audit 04 F1 / A7): the mode comes from
  edma; when edma says minor (the modes agree) the tonic comes from edmm,
  otherwise from edma. `key` holds tonic, mode, label, score (of the profile
  that set the tonic), `rule` ("edmm tonic, modes agree" or "edma, major") and
  both profiles' own results; `key_edmm` / `key_edma` hold those in full
  (tonic, mode, score, the 24 scores).
- Chord chroma per bar: mean frame chroma over the grid frames whose centre is
  in each bar (bars from `bar_times`), plus the best major / minor triad
  template by cosine similarity, its label (e.g. "Am") and score.
- Bass root per beat (`beat_times` required; from structure.py in analyse):
  1. L, R and M = (L + R)/sqrt2 separately (never sqrt(L^2 + R^2), audit F10):
     zero-phase Butterworth low-pass, order 4 under sosfiltfilt (-6 dB at
     250 Hz), zero-padded edges, resampled to 4 kHz. R reuses L's estimates
     when the two are bit-identical.
  2. Per channel, 10 ms hop: (a) pYIN (librosa.pyin, 30 to 250 Hz, 128 ms
     frames, 30 s blocks with 2 s overlap each side to bound memory, the last
     block read through its final half frame), estimate = f0 where voiced;
     (b) subharmonic summation (Hermes 1988; the harmonic-product-spectrum
     family, summed in magnitude) on a 128 ms Hann frame zero-padded to 8192:
     S(c) = sum_{h=1..4} 0.84^(h-1) |X(h c)| on a 10-cent grid 30 to 250 Hz,
     candidates above 130 Hz weighted 0.5 (pads leaking under the LPF);
     estimate = argmax, raw confidence = 1 - mean(S) / max(S) times the frame's
     level ramp (0 at -40 dB of the loudest frame). Four harmonics, not the
     eight the audit fix list asks for: 6 or 8 measured worse (bass + pad
     0.875 -> 0.500) and 4 already reads missing-fundamental basses right
     (brief-04 D.22). The 2 f0 phantom rule was tried and dropped (it moved a
     correct missing-fundamental root an octave down on 3 of 8 beats).
  3. Calibration (audit 04 F3): each raw confidence (pYIN voiced probability,
     SHS peakiness) goes through CALIB_PYIN / CALIB_SHS, the measured
     P(within 50 cents of the true root) per confidence bin, so the two are on
     one scale before they are weighed together. pYIN estimates above 130 Hz
     are weighted 0.5 as well.
  4. Per beat, every estimate (3 channels x 2 methods x frames in the beat)
     gets weight w = phase weight x kick mask x calibrated confidence. Phase
     weight 0.1 in the first half of the beat, 1.0 in the second (where
     sidechain ducking has released); kick mask 0 from half an analysis frame
     (64 ms) before to 150 ms after a kick onset (analyse passes only kicks
     scoring >= 0.8, transients.KICK_MASK_SCORE), else 1.
  5. Root = weighted median of the estimates in MIDI note numbers, rounded to
     a semitone; `root_hz` = 440 * 2^((midi - 69) / 12) of the unrounded median.
     `confidence` = (sum of weights within 50 cents of the rounded root / sum
     of all weights) x (phase x mask weighted share of frames pYIN calls voiced,
     averaged over L, R, M). It measures how much the estimators agree with
     each other, not whether they are right: a half-tempo grid kept it at 1.0
     while accuracy fell to 0.50 (audit 04 F2). `glide` = the OLS slope of
     voiced pYIN pitch (M) over the beat's second half (frames whose window ends
     inside the beat) times the beat length exceeds 1 semitone.
  6. Independent check: `nsdf_midi` / `nsdf_confidence` per beat from
     borrowed.nsdf_f0 (McLeod NSDF, drumbusmaximus) on the M low band over 55 to
     95 % of the beat; `nsdf_agree` = NSDF confidence > 0.5 and within 50 cents
     of the root.
  Beats with no weight left give root NaN and confidence 0. keyeq eligibility
  (`keyeq_gate`): median confidence >= 0.75 AND NSDF agreement on >= 75 % of
  the beats with a root (spec 3.3 draft 0.4); `bass_nsdf_agreement` is that
  share.
- The Demucs bass-stem comparison is optional and not installed; not run.
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy import signal

from .. import bands as _bands
from . import borrowed
from ._common import AR, Context, ensure_context, librosa_on_grid, resample, window_mean
from .transients import kick_mask

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Essentia key.cpp (Faraldo et al. 2017): edmm major is flat by design.
EDMM_MAJOR = np.full(12, 0.083)
EDMM_MINOR = np.array([0.17235348, 0.04, 0.0761009, 0.12, 0.05621498, 0.08527853,
                       0.0497915, 0.13451001, 0.07458916, 0.05003023, 0.09187879, 0.05545106])
EDMA_MAJOR = np.array([1.00, 0.29, 0.50, 0.40, 0.60, 0.56, 0.32, 0.80, 0.31, 0.45, 0.42, 0.39])
EDMA_MINOR = np.array([1.00, 0.31, 0.44, 0.58, 0.33, 0.49, 0.29, 0.78, 0.43, 0.29, 0.53, 0.32])
PROFILES = {"edmm": (EDMM_MAJOR, EDMM_MINOR), "edma": (EDMA_MAJOR, EDMA_MINOR)}

BASS_LPF_HZ = 250.0
BASS_LPF_ORDER = 4
BASS_FS = 4000
BASS_FMIN, BASS_FMAX = 30.0, 250.0
BASS_FRAME = 512  # 128 ms at 4 kHz
BASS_HOP = 40  # 10 ms
SHS_NFFT = 8192
SHS_H = 4  # 6 or 8 measured worse (audit/brief-04.md D.22)
HIGH_ROOT_HZ = 130.0
# Calibration (audit 04 F3): P(estimate within 50 cents of the true root) per raw
# confidence bin, measured on frames in the second half of each beat of six
# sidechained 8-bar basslines (sine + h2, saw, weak fundamental, two missing-
# fundamental variants, sine + A minor pad; brief-04 C.7), made monotone by
# pooling adjacent violators. pYIN voiced probability and SHS peakiness are
# not on one scale: pYIN is right ~100 % from 0.4 up, SHS only from 0.6 up.
_CALIB_EDGES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9])
CALIB_PYIN = (_CALIB_EDGES, (0.463, 0.463, 1.0, 1.0, 1.0, 1.0))
CALIB_SHS = (_CALIB_EDGES, (0.408, 0.408, 0.408, 0.973, 1.0, 1.0))
HIGH_ROOT_WEIGHT = 0.5  # candidates above 130 Hz (pads leaking under the 250 Hz LPF)
SHS_DECAY = 0.84
SHS_RANGE_DB = 40.0
FIRST_HALF_WEIGHT = 0.1
KICK_MASK_S = 0.150
HALF_FRAME_S = BASS_FRAME / 2 / BASS_FS  # a frame centred this close before a kick already contains it
AGREE_CENTS = 50.0
GLIDE_SEMITONES = 1.0
KEYEQ_MIN_CONFIDENCE = 0.75
KEYEQ_MIN_AGREE = 0.75
NSDF_SPAN = (0.55, 0.95)  # share of the beat the independent NSDF estimate reads
NSDF_MIN_CONF = 0.5
SALIENCE = "shs"  # "shs" (subharmonic summation) or "hs" (borrowed harmonic salience)
CHUNK_FRAMES = 2048
PYIN_BLOCK_S = 30.0
PYIN_OVERLAP_S = 2.0


def _pearson(a, b):
    if np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return np.nan  # undefined for a constant vector (the flat edmm major profile)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def key_from_chroma(chroma_mean: np.ndarray, profile: str = "edmm") -> dict:
    major, minor = PROFILES[profile]
    scores = np.full(24, np.nan)
    for t in range(12):
        scores[t] = _pearson(chroma_mean, np.roll(major, t))
        scores[12 + t] = _pearson(chroma_mean, np.roll(minor, t))
    if not np.any(np.isfinite(scores)):
        return {"tonic": None, "mode": None, "score": np.nan, "scores": scores, "profile": profile}
    i = int(np.nanargmax(scores))
    return {"tonic": NOTE_NAMES[i % 12], "mode": "major" if i < 12 else "minor",
            "label": NOTE_NAMES[i % 12] + ("" if i < 12 else "m"),
            "score": float(scores[i]), "scores": scores, "profile": profile}


def combined_key(edmm: dict, edma: dict) -> dict:
    """Mode from edma; tonic from edmm when edma also says minor."""
    if edma.get("mode") is None:
        return {"tonic": None, "mode": None, "label": None, "score": np.nan, "rule": "no chroma",
                "edmm": _brief(edmm), "edma": _brief(edma)}
    if edma["mode"] == "minor" and edmm.get("tonic") is not None:
        src, rule = edmm, "edmm tonic, modes agree"
    else:
        src, rule = edma, "edma, major"
    return {"tonic": src["tonic"], "mode": edma["mode"],
            "label": src["tonic"] + ("m" if edma["mode"] == "minor" else ""),
            "score": src["score"], "rule": rule, "edmm": _brief(edmm), "edma": _brief(edma)}


def _brief(k: dict) -> dict:
    return {"label": k.get("label"), "score": k.get("score")}


def _triads():
    names, temps = [], []
    for r in range(12):
        for q, iv in (("", (0, 4, 7)), ("m", (0, 3, 7))):
            t = np.zeros(12)
            t[[(r + i) % 12 for i in iv]] = 1.0
            names.append(NOTE_NAMES[r] + q)
            temps.append(t / np.linalg.norm(t))
    return names, np.array(temps)


def chroma(ctx: Context) -> np.ndarray:
    if "chroma" not in ctx.cache:
        y = ctx.mono_ar[1024:].astype(np.float32)
        c = librosa.feature.chroma_cqt(y=y, sr=AR, hop_length=2048)
        ctx.cache["chroma"] = librosa_on_grid(c.astype(np.float64), ctx.grid.k)
    return ctx.cache["chroma"]


def chords_per_bar(ch: np.ndarray, grid, bar_times, end_s: float) -> list[dict]:
    names, temps = _triads()
    out = []
    bt = list(bar_times) + [end_s]
    for a, b in zip(bt[:-1], bt[1:]):
        sel = grid.frames_in(a, b)
        v = ch[:, sel].mean(axis=1) if np.any(sel) else np.zeros(12)
        nv = np.linalg.norm(v)
        if nv > 0:
            sc = temps @ (v / nv)
            i = int(np.argmax(sc))
            out.append({"start": float(a), "end": float(b), "chroma": v, "chord": names[i],
                        "score": float(sc[i])})
        else:
            out.append({"start": float(a), "end": float(b), "chroma": v, "chord": None, "score": 0.0})
    return out


def _bass_signals(x: np.ndarray, fs: float) -> list[np.ndarray]:
    ms = _bands.to_ms(x)
    sos = signal.butter(BASS_LPF_ORDER, BASS_LPF_HZ, btype="lowpass", fs=fs, output="sos")
    pad = int(np.ceil(20 * fs / (2 * np.pi * BASS_LPF_HZ)))
    out = []
    for y in (x[:, 0], x[:, 1], ms[:, 0]):
        yp = np.pad(y, (pad, pad))
        lo = signal.sosfiltfilt(sos, yp, padtype=None)[pad:pad + y.shape[0]]
        out.append(resample(lo, fs, BASS_FS))
    return out


def _shs(y: np.ndarray, n_frames: int):
    """Subharmonic summation per frame; returns (midi, confidence). Frames are
    processed in blocks so memory stays bounded on long tracks."""
    cand = BASS_FMIN * 2.0 ** (np.arange(0, 1200 * np.log2(BASS_FMAX / BASS_FMIN) + 1, 10) / 1200.0)
    yp = np.pad(y, (BASS_FRAME // 2, BASS_FRAME // 2 + BASS_HOP * n_frames))
    view = np.lib.stride_tricks.sliding_window_view(yp, BASS_FRAME)[::BASS_HOP][:n_frames]
    win = np.hanning(BASS_FRAME + 1)[:-1]
    fbin = np.arange(SHS_NFFT // 2 + 1) * BASS_FS / SHS_NFFT
    high_w = np.where(cand > HIGH_ROOT_HZ, HIGH_ROOT_WEIGHT, 1.0)
    best = np.empty(n_frames, dtype=np.int64)
    conf = np.empty(n_frames)
    lvl = np.empty(n_frames)
    for a in range(0, n_frames, CHUNK_FRAMES):
        v = view[a:a + CHUNK_FRAMES]
        mag = np.abs(np.fft.rfft(v * win, n=SHS_NFFT, axis=1))
        sal = np.zeros((v.shape[0], cand.size))
        for h in range(1, SHS_H + 1):
            sal += SHS_DECAY ** (h - 1) * _interp_rows(mag, fbin, h * cand)
        sal *= high_w[None, :]
        bi = np.argmax(sal, axis=1)
        smax = sal[np.arange(v.shape[0]), bi]
        with np.errstate(invalid="ignore", divide="ignore"):
            conf[a:a + v.shape[0]] = np.where(smax > 0, 1.0 - sal.mean(axis=1) / smax, 0.0)
        best[a:a + v.shape[0]] = bi
        lvl[a:a + v.shape[0]] = 20 * np.log10(np.maximum(np.sqrt(np.mean(v ** 2, axis=1)), 1e-12))
    rel = np.clip(1.0 - (lvl.max() - lvl) / SHS_RANGE_DB, 0.0, 1.0) if lvl.size else lvl
    midi = 69.0 + 12.0 * np.log2(cand[best] / 440.0)
    return midi, np.clip(conf, 0, 1) * rel


def _hs(y: np.ndarray, n_frames: int):
    """Borrowed harmonic salience (TRANSMUTATION pitch_candidates core, see
    borrowed.py) per frame on the same 128 ms Hann frames zero-padded to 8192.
    Returns (midi, confidence); confidence = 1 - second-best / best candidate
    score (the original's dominance, as a 0..1 number), times the level ramp
    used by `_shs`."""
    yp = np.pad(y, (BASS_FRAME // 2, BASS_FRAME // 2 + BASS_HOP * n_frames))
    view = np.lib.stride_tricks.sliding_window_view(yp, BASS_FRAME)[::BASS_HOP][:n_frames]
    win = np.hanning(BASS_FRAME + 1)[:-1]
    f0 = np.full(n_frames, np.nan)
    conf = np.zeros(n_frames)
    lvl = np.empty(n_frames)
    for a in range(0, n_frames, CHUNK_FRAMES):
        v = view[a:a + CHUNK_FRAMES]
        mag = np.abs(np.fft.rfft(v * win, n=SHS_NFFT, axis=1))
        f, sc, dom = borrowed.harmonic_salience(mag, BASS_FS, SHS_NFFT, fmin=BASS_FMIN,
                                                fmax=BASS_FMAX, return_dominance=True)
        f0[a:a + v.shape[0]] = f
        conf[a:a + v.shape[0]] = dom
        lvl[a:a + v.shape[0]] = 20 * np.log10(np.maximum(np.sqrt(np.mean(v ** 2, axis=1)), 1e-12))
    rel = np.clip(1.0 - (lvl.max() - lvl) / SHS_RANGE_DB, 0.0, 1.0) if lvl.size else lvl
    ok = np.isfinite(f0)
    midi = np.where(ok, 69.0 + 12.0 * np.log2(np.where(ok, f0, 440.0) / 440.0), np.nan)
    return midi, np.where(ok, conf, 0.0) * rel


def _pyin(y: np.ndarray, n_frames: int):
    """librosa.pyin in overlapping blocks of PYIN_BLOCK_S (PYIN_OVERLAP_S each
    side, only the centre kept): librosa holds every frame x threshold at once,
    several GB on a 6 min track. The block edges only shorten the Viterbi
    smoothing context, and 2 s of overlap is far longer than any bass note."""
    blk = int(round(PYIN_BLOCK_S * BASS_FS / BASS_HOP))
    ov = int(round(PYIN_OVERLAP_S * BASS_FS / BASS_HOP))
    f0 = np.full(n_frames, np.nan)
    vf = np.zeros(n_frames, dtype=bool)
    vp = np.zeros(n_frames)
    for a in range(0, n_frames, blk):
        b = min(n_frames, a + blk)
        a0 = max(0, a - ov)
        b0 = min(n_frames, b + ov)
        # through the half frame after the block's last centre, so the final block's
        # last frames see the real signal rather than padding (audit 04 A6)
        seg = y[a0 * BASS_HOP:min(y.size, (b0 - 1) * BASS_HOP + BASS_FRAME // 2 + 1)]
        if seg.size < BASS_FRAME // 2:
            seg = np.pad(seg, (0, BASS_FRAME // 2 - seg.size))
        f, v, p = librosa.pyin(seg, fmin=BASS_FMIN, fmax=BASS_FMAX, sr=BASS_FS,
                               frame_length=BASS_FRAME, hop_length=BASS_HOP, center=True)
        i0, i1 = a - a0, a - a0 + (b - a)
        f0[a:b], vf[a:b], vp[a:b] = f[i0:i1], v[i0:i1], p[i0:i1]
    return f0, vf, vp


def _interp_rows(mag, fbin, fq):
    """Row-wise linear interpolation of mag (frames, bins) at frequencies fq."""
    df = fbin[1] - fbin[0]
    pos = fq / df
    i0 = np.clip(np.floor(pos).astype(int), 0, mag.shape[1] - 2)
    fr = pos - i0
    return mag[:, i0] * (1 - fr) + mag[:, i0 + 1] * fr


def _calibrate(c: np.ndarray, table) -> np.ndarray:
    """Map a raw confidence through a piecewise-constant table of measured
    P(within 50 cents of the true root) per confidence bin."""
    edges, p = table
    k = np.clip(np.digitize(np.asarray(c, dtype=np.float64), edges) - 1, 0, len(p) - 1)
    return np.asarray(p)[k]


def _wmedian(v, w):
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, 0.5 * c[-1])])


def bass_root(x: np.ndarray, fs: float, beat_times, kick_onsets=None) -> dict:
    sigs = _bass_signals(x, fs)
    n = sigs[0].shape[0]
    n_frames = 1 + n // BASS_HOP
    tf = np.arange(n_frames) * BASS_HOP / BASS_FS
    est, conf = [], []
    pyin_m = None
    for c, y in enumerate(sigs):
        if c == 1 and np.array_equal(y, sigs[0]):  # R identical to L: same estimates
            est.extend(est[0:2])
            conf.extend(conf[0:2])
            continue
        f0, voiced, vprob = _pyin(y.astype(np.float64), n_frames)
        ok = voiced & np.isfinite(f0)
        m = np.where(ok, 69.0 + 12.0 * np.log2(np.where(ok, f0, 440.0) / 440.0), np.nan)
        q = _calibrate(np.where(np.isfinite(m), vprob, 0.0), CALIB_PYIN) * np.isfinite(m)
        q = np.where(np.isfinite(f0) & (f0 > HIGH_ROOT_HZ), HIGH_ROOT_WEIGHT * q, q)
        est.append(m)
        conf.append(q)
        if c == 2:
            pyin_m = m
        sm, sq = (_hs if SALIENCE == "hs" else _shs)(y, n_frames)
        if SALIENCE != "hs":
            sq = _calibrate(sq, CALIB_SHS) * (sq > 0)
        est.append(sm)
        conf.append(sq)
    est = np.array(est)
    conf = np.array(conf)
    beat_times = np.asarray(beat_times, dtype=np.float64)
    dur = n / BASS_FS
    if beat_times.size == 0:
        return {"beat_times": beat_times, "root_midi": np.zeros(0), "root_hz": np.zeros(0),
                "root_name": [], "confidence": np.zeros(0), "glide": np.zeros(0, bool),
                "nsdf_midi": np.zeros(0), "nsdf_confidence": np.zeros(0),
                "nsdf_agree": np.zeros(0, bool)}
    ibi = float(np.median(np.diff(beat_times))) if beat_times.size > 1 else dur - beat_times[0]
    ends = np.append(beat_times[1:], min(dur, beat_times[-1] + ibi))
    mask = np.ones(n_frames)
    if kick_onsets is not None and len(kick_onsets):
        mask = 1.0 - kick_mask(tf, kick_onsets, HALF_FRAME_S, KICK_MASK_S)
    roots, hz, names, cf, gl = [], [], [], [], []
    nsdf_m, nsdf_c = [], []
    for a, b in zip(beat_times, ends):
        i0 = int(round((a + NSDF_SPAN[0] * (b - a)) * BASS_FS))
        i1 = int(round((a + NSDF_SPAN[1] * (b - a)) * BASS_FS))
        fq, cq = borrowed.nsdf_f0(sigs[2][max(0, i0):max(0, i1)], BASS_FS, BASS_FMIN, BASS_FMAX)
        nsdf_m.append(69.0 + 12.0 * np.log2(fq / 440.0) if np.isfinite(fq) and fq > 0 else np.nan)
        nsdf_c.append(cq)
        sel = (tf >= a) & (tf < b)
        if b <= a or not np.any(sel):
            roots.append(np.nan); hz.append(np.nan); names.append(None); cf.append(0.0); gl.append(False)
            continue
        ph = (tf[sel] - a) / (b - a)
        pw = np.where(ph < 0.5, FIRST_HALF_WEIGHT, 1.0) * mask[sel]
        e = est[:, sel]
        w = conf[:, sel] * pw[None, :]
        ok = np.isfinite(e) & (w > 0)
        if not np.any(ok):
            roots.append(np.nan); hz.append(np.nan); names.append(None); cf.append(0.0); gl.append(False)
            continue
        med = _wmedian(e[ok], w[ok])
        r = int(round(med))
        agree = ok & (np.abs(e - r) * 100.0 <= AGREE_CENTS)
        # agreement among the confident evidence x share of the beat pYIN calls voiced
        voiced = np.isfinite(est[0::2, sel]) * pw[None, :]
        vfrac = voiced.sum() / (pw.sum() * voiced.shape[0]) if pw.sum() > 0 else 0.0
        roots.append(float(r))
        hz.append(440.0 * 2.0 ** ((med - 69.0) / 12.0))
        names.append(NOTE_NAMES[r % 12] + str(r // 12 - 1))
        cf.append(float(np.sum(w[agree]) / np.sum(w[ok]) * vfrac))
        pm = pyin_m[sel]
        second = (ph >= 0.5) & (tf[sel] <= b - HALF_FRAME_S) & np.isfinite(pm)
        g = False
        if np.count_nonzero(second) >= 4:
            sl = np.polyfit(tf[sel][second], pm[second], 1)[0]
            g = bool(abs(sl) * (b - a) > GLIDE_SEMITONES)
        gl.append(g)
    roots_a = np.array(roots)
    nm, nc = np.array(nsdf_m), np.array(nsdf_c)
    agree = (nc > NSDF_MIN_CONF) & np.isfinite(nm) & np.isfinite(roots_a) &         (np.abs(np.nan_to_num(nm) - np.nan_to_num(roots_a)) * 100.0 <= AGREE_CENTS)
    return {"beat_times": beat_times, "root_midi": roots_a, "root_hz": np.array(hz),
            "root_name": names, "confidence": np.array(cf), "glide": np.array(gl),
            "nsdf_midi": nm, "nsdf_confidence": nc, "nsdf_agree": agree}


def keyeq_gate(br: dict) -> tuple[bool, float]:
    """(eligible, NSDF agreement share): eligible when the median confidence is
    >= 0.75 AND the per-beat root agrees with the independent NSDF estimate on
    >= 0.75 of the beats that have a root (audit 04 F2)."""
    c = br["confidence"]
    has = np.isfinite(br["root_midi"])
    if not np.any(has):
        return False, 0.0
    share = float(np.mean(br["nsdf_agree"][has]))
    return bool(np.median(c) >= KEYEQ_MIN_CONFIDENCE and share >= KEYEQ_MIN_AGREE), share


def key(ctx: Context, frame_power=None) -> dict:
    """The key part of `analyse` (chroma, power-weighted mean, both profiles, combined
    key): {"chroma", "chroma_mean", "key_edmm", "key_edma", "key"}. It reads neither
    the band split nor the crossovers, so features.analyse can run it before the
    first split (key-locked crossovers); the chroma is cached on ctx."""
    g = ctx.grid
    ch = chroma(ctx)
    if frame_power is None:
        y = ctx.mono_ar
        hop = 2048
        cen = np.round(g.times * AR).astype(np.int64)
        frame_power = window_mean(y * y, cen, hop)
    w = np.where(np.isfinite(frame_power), frame_power, 0.0)
    cm = (ch * w[None, :]).sum(axis=1) / w.sum() if w.sum() > 0 else ch.mean(axis=1)
    out = {"chroma": ch, "chroma_mean": cm,
           "key_edmm": key_from_chroma(cm, "edmm"), "key_edma": key_from_chroma(cm, "edma")}
    out["key"] = combined_key(out["key_edmm"], out["key_edma"])
    return out


def analyse(x: np.ndarray, fs: float, ctx: Context | None = None, beat_times=None,
            bar_times=None, kick_onsets=None, frame_power=None) -> dict:
    ctx = ensure_context(x, fs, ctx)
    g = ctx.grid
    out = {"t": g.times, **key(ctx, frame_power)}
    ch = out["chroma"]
    dur = ctx.x.shape[0] / ctx.fs
    out["chords"] = chords_per_bar(ch, g, bar_times, dur) if bar_times is not None and len(bar_times) else []
    if beat_times is not None:
        br = bass_root(ctx.x, ctx.fs, beat_times, kick_onsets)
        out["bass"] = br
        c = br["confidence"]
        out["bass_confidence_median"] = float(np.median(c)) if c.size else 0.0
        out["keyeq_eligible"], out["bass_nsdf_agreement"] = keyeq_gate(br)
    return out
