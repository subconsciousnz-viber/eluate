"""Noise floor detection (docs/noise-and-stems.md §1, detection only; reduction is opt-in and
not built).

Contract:
- `floor(x, fs, quiet=None) -> dict`: the noise floor of a stereo or mono signal in its
  quiet passages, ITU-R BS.468 weighted (the noise-measurement curve, +12.2 dB at 6.3 kHz),
  over the high register (FLOOR_BANDS_HZ, hiss being the likely noise; maintainer, 2026-10-20),
  in dB re full scale with the AES17 convention (a full-scale sine reads 0).
  Frames (Hann, FRAME_S, half overlap, channels averaged) are quiet when their broadband
  level is more than QUIET_BELOW_DB below the track's loud level (the 95th percentile of
  the frame levels) and not digital silence: intros, gaps, tails and fades, where the music
  does not mask a noise floor. Loud passages are not measured: noise there is masked, and
  a dense mix leaves no gaps in which a floor can be told from the music. Per third-octave
  band, the floor is the QUIET_PERCENTILE of the band's weighted power over the quiet
  frames; the broadband floor is their sum. `quiet` passes another signal's quiet frames
  (the master is read on the premaster's, so both are measured at the same moments),
  trimmed to this signal's frame count or padded with False (frames past its end are
  not quiet).
  Returns {"db", "bands_hz", "bands_db", "quiet_s", "quiet_spans": [start, end] seconds of
  the quiet passages (runs closer than 0.5 s merged), "quiet": frame mask}; db is -inf when
  there are fewer than MIN_QUIET_S of quiet frames or the sample rate leaves no band in
  FLOOR_BANDS_HZ (no bands then).
- `assess(before, after, threshold_db=FLAG_DB) -> dict` (JSON-ready, no mask): status
  "exposed" = below the threshold before and at or above it after (the master made it
  audible), "present" = at or above it in both, None otherwise (or nothing to measure).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import stft

FLAG_DB = -60.0               # maintainer, 2026-10-20: -60 or -50 dB, spectrally weighted
FRAME_S = 0.046
QUIET_BELOW_DB = 35.0
QUIET_PERCENTILE = 20.0
MIN_QUIET_S = 0.5
SILENT_DB = -120.0
LOUD_PERCENTILE = 95.0
FLOOR_BANDS_HZ = (2000.0, 20000.0)
AES17_DB = 3.0103


def weighting_468_db(f) -> np.ndarray:
    """ITU-R BS.468 weighting in dB (0 dB at 1 kHz, +12.2 dB at 6.3 kHz)."""
    f = np.asarray(f, dtype=np.float64)
    h1 = (-4.737338981378384e-24 * f ** 6 + 2.043828333606125e-15 * f ** 4
          - 1.363894795463638e-07 * f ** 2 + 1.0)
    h2 = 1.306612257412824e-19 * f ** 5 - 2.118150887518656e-11 * f ** 3 + 5.559488023498642e-04 * f
    with np.errstate(divide="ignore"):
        r = 1.246332637532143e-04 * f / np.sqrt(h1 ** 2 + h2 ** 2)
        return 18.2 + 20.0 * np.log10(r)


def _centres(fs: float) -> np.ndarray:
    hi = min(FLOOR_BANDS_HZ[1], 0.45 * fs)
    k = np.arange(np.ceil(3 * np.log2(FLOOR_BANDS_HZ[0] / 1000.0)),
                  np.floor(3 * np.log2(hi / 1000.0)) + 1)
    return 1000.0 * 2.0 ** (k / 3.0)


def _spans(mask, hop_s, frame_s, gap_s=0.5) -> list:
    """[start, end] seconds of the runs of True frames, runs closer than gap_s merged."""
    idx = np.flatnonzero(mask)
    if not idx.size:
        return []
    spans, s, e = [], idx[0], idx[0]
    for i in idx[1:]:
        if (i - e) * hop_s > gap_s:
            spans.append([round(s * hop_s, 1), round(e * hop_s + frame_s, 1)])
            s = i
        e = i
    spans.append([round(s * hop_s, 1), round(e * hop_s + frame_s, 1)])
    return spans


def floor(x: np.ndarray, fs: int, quiet: np.ndarray | None = None) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = int(round(FRAME_S * fs))
    n += n % 2
    f, _, Z = stft(x.T, fs=fs, window=np.hanning(n), nperseg=n, noverlap=n // 2,
                   boundary=None, padded=False, scaling="psd")
    # one-sided power per bin (2 |Z|^2 df): summed over bins it is the frame's mean square,
    # so a full-scale sine reads -3.01 dB plain, 0 dB with the AES17 offset
    p = 2.0 * np.mean(np.abs(Z) ** 2, axis=0) * (f[1] - f[0])     # (bins, frames)
    centres = _centres(fs)
    w = 10.0 ** (weighting_468_db(np.maximum(f, 1e-9)) / 10.0)
    band_p = np.zeros((0, p.shape[1]))
    if centres.size:
        band_p = np.stack([(p[m] * w[m, None]).sum(axis=0) for m in
                           ((f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6)) for c in centres)])
    if quiet is None:
        level = 10 * np.log10(p.sum(axis=0) + 1e-30)
        live = level > SILENT_DB
        loud = np.percentile(level[live], LOUD_PERCENTILE) if np.any(live) else 0.0
        quiet = live & (level < loud - QUIET_BELOW_DB)
    # a borrowed mask fits this signal's frames: trimmed, or padded with False (not quiet)
    nf = band_p.shape[1]
    quiet = np.asarray(quiet, dtype=bool)[:nf]
    quiet = np.concatenate([quiet, np.zeros(nf - len(quiet), dtype=bool)])
    hop_s = (n // 2) / fs
    out = {"bands_hz": centres.tolist(), "quiet_s": float(quiet.sum() * hop_s), "quiet": quiet,
           "quiet_spans": _spans(quiet, hop_s, n / fs)}
    if quiet.sum() * hop_s < MIN_QUIET_S:
        return {**out, "db": float("-inf"), "bands_db": [float("-inf")] * len(centres)}
    fl = np.percentile(band_p[:, quiet], QUIET_PERCENTILE, axis=1)
    with np.errstate(divide="ignore"):
        bands_db = 10 * np.log10(fl) + AES17_DB
        db = 10 * np.log10(fl.sum()) + AES17_DB
    return {**out, "db": float(db), "bands_db": [float(v) for v in bands_db]}


def assess(before: dict, after: dict, threshold_db: float = FLAG_DB) -> dict:
    b, a = before["db"], after["db"]
    status = None
    if np.isfinite(a) and a >= threshold_db:
        status = "exposed" if b < threshold_db else "present"
    loudest = None
    if np.isfinite(a):
        loudest = after["bands_hz"][int(np.argmax(after["bands_db"]))]
    return {"before_db": b, "after_db": a, "threshold_db": threshold_db, "status": status,
            "weighting": "ITU-R BS.468", "quiet_s": before["quiet_s"],
            "quiet_spans": before["quiet_spans"], "loudest_band_hz": loudest,
            "bands_hz": after["bands_hz"], "before_bands_db": before["bands_db"],
            "after_bands_db": after["bands_db"]}
