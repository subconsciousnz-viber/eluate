"""Fixtures for the per-band ceiling, glue and crest-loss gates (audit brief 06).

Every fixture peaks at or below FULL_SCALE_PEAK (-0.45 dBFS) in L and R (audit 06 F10, M14).

- `band_impulse_train(fs, band, ...)`: an impulse train band-limited to one
  band of the default zero-phase split (the band-k output of bands.split of
  an impulse train), placed in M or S, scaled so its sample peak is
  `peak_dbfs`. Splitting the fixture again does *not* give zero in the other
  bands: the zero-phase split is not idempotent, so a re-split puts part of
  it into the adjacent bands (measured -28 / -16 dBFS in bands 1 / 3 for a
  -6 dBFS band-2 train).
- `kick_plus_sustain(fs, ...)`: synthetic.kick on every beat over a sustained
  three-note pad (sines at 220 / 277.2 / 329.6 Hz), mono in L and R, scaled
  down when the kick's click term would push it over FULL_SCALE_PEAK.
- `graded_kicks(fs, levels_db, ...)`: the same, with each kick at its own level
  relative to the loudest (default 0 to -12 dB in 2 dB steps), for per-kick
  gates (audit 06 M2).
"""

from __future__ import annotations

import numpy as np

from fixtures import synthetic as syn
from eluate import bands

FULL_SCALE_PEAK = 0.95


def _cap(x: np.ndarray) -> np.ndarray:
    pk = np.max(np.abs(x))
    return x * (FULL_SCALE_PEAK / pk) if pk > FULL_SCALE_PEAK else x


def band_impulse_train(fs: float, band: int, seconds: float = 2.0, period_s: float = 0.25,
                       peak_dbfs: float = -6.0, channel: int = 0,
                       crossovers_hz=bands.DEFAULT_CROSSOVERS, bed=None):
    """Returns (x_lr, band_ms) with band_ms the M/S train alone, (n, 2)."""
    n = int(round(seconds * fs))
    imp = syn.impulse_train(n, fs, period_s, offset_s=period_s / 2, channels=1)[:, 0]
    b = bands.split(imp, fs, crossovers_hz)[band]
    b *= 10.0 ** (peak_dbfs / 20.0) / np.max(np.abs(b))
    ms = np.zeros((n, 2))
    ms[:, channel] = b
    x = bands.to_lr(ms)
    if bed is not None:
        x = x + bed
    if np.max(np.abs(x)) > FULL_SCALE_PEAK:
        raise ValueError("band_impulse_train above full scale; lower peak_dbfs")
    return x, ms


def _kicks(fs, levels, bpm, pad_peak, kick_level):
    beat = 60.0 / bpm
    n = int(round(len(levels) * beat * fs))
    t = np.arange(n) / fs
    pad = sum(np.sin(2 * np.pi * f * t + p) for f, p in ((220.0, 0.0), (277.18, 1.0),
                                                       (329.63, 2.0)))
    pad *= pad_peak / np.max(np.abs(pad))
    x = pad.copy()
    k = syn.kick(fs, level=kick_level)
    onsets = np.arange(len(levels)) * beat
    for o, g in zip(onsets, levels):
        i0 = int(round(o * fs))
        m = min(k.size, n - i0)
        x[i0:i0 + m] += g * k[:m]
    return _cap(np.stack([x, x], axis=1)), onsets


def kick_plus_sustain(fs: float, bpm: float = 120.0, beats: int = 8, pad_peak: float = 0.12,
                      kick_level: float = 0.9):
    """Returns (x, onsets_s): kick on every beat over a sustained pad, peak <= -0.45 dBFS."""
    return _kicks(fs, [1.0] * beats, bpm, pad_peak, kick_level)


def graded_kicks(fs: float, levels_db=(0.0, -2.0, -4.0, -6.0, -8.0, -10.0, -12.0),
                 bpm: float = 120.0, pad_peak: float = 0.12, kick_level: float = 0.9):
    """Returns (x, onsets_s, levels_db): kicks at graded relative levels over a pad."""
    lv = [10.0 ** (d / 20.0) for d in levels_db]
    x, onsets = _kicks(fs, lv, bpm, pad_peak, kick_level)
    return x, onsets, list(levels_db)
