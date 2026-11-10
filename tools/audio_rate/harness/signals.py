"""Synthetic test signals (48 kHz stereo by default). No third-party audio."""
from __future__ import annotations

import numpy as np

from .common import FS, load_test_module


def _stereo(x):
    return np.stack([x, x], axis=1)


def sine(f0: float, fs: int = FS, seconds: float = 20.0, dbfs: float = -6.0,
         stereo_offset_db: float | None = None) -> np.ndarray:
    """S1: steady sine. Identical L/R (a mono sub) unless stereo_offset_db gives R a level
    offset (so S is not silent)."""
    n = int(round(seconds * fs))
    t = np.arange(n) / fs
    a = 10.0 ** (dbfs / 20.0)
    x = a * np.sin(2.0 * np.pi * f0 * t)
    if stereo_offset_db is None:
        return _stereo(x)
    return np.stack([x, x * 10.0 ** (stereo_offset_db / 20.0)], axis=1)


def kick(fs: int = FS, f_start: float = 150.0, f_end: float = 45.0, sweep_s: float = 0.06,
         decay_s: float = 0.15, length_s: float = 0.4, level: float = 1.0) -> np.ndarray:
    """Synthetic kick: sine swept f_start -> f_end (exponential) over sweep_s, held at f_end,
    with an exponential amplitude decay of time constant decay_s; 2 ms attack."""
    n = int(round(length_s * fs))
    t = np.arange(n) / fs
    k = np.log(f_end / f_start) / sweep_s
    f = np.where(t < sweep_s, f_start * np.exp(k * np.minimum(t, sweep_s)), f_end)
    ph = 2.0 * np.pi * np.cumsum(f) / fs
    env = np.exp(-t / decay_s) * (1.0 - np.exp(-t / 0.002))
    y = level * env * np.sin(ph)
    fade = int(0.01 * fs)
    y[-fade:] *= np.cos(np.linspace(0, np.pi / 2, fade)) ** 2
    return y


def kick_onsets(bpm: float, seconds: float, offset_s: float = 0.25) -> np.ndarray:
    beat = 60.0 / bpm
    return np.arange(offset_s, seconds - 0.5, beat)


def sine_plus_kick(fs: int = FS, seconds: float = 20.0, f_sub: float = 40.0, bpm: float = 128.0,
                   sub_dbfs: float = -12.0, kick_dbfs: float = -6.0):
    """S2: f_sub sine + kick at bpm. Returns (x, onsets_s)."""
    n = int(round(seconds * fs))
    t = np.arange(n) / fs
    x = 10.0 ** (sub_dbfs / 20.0) * np.sin(2.0 * np.pi * f_sub * t)
    kk = kick(fs, level=10.0 ** (kick_dbfs / 20.0))
    on = kick_onsets(bpm, seconds)
    for o in on:
        i = int(round(o * fs))
        m = min(kk.size, n - i)
        x[i:i + m] += kk[:m]
    return _stereo(x), on


def kick_loop(fs: int = FS, seconds: float = 20.0, bpm: float = 128.0, kick_dbfs: float = -6.0):
    """S7: kicks only."""
    n = int(round(seconds * fs))
    x = np.zeros(n)
    kk = kick(fs, level=10.0 ** (kick_dbfs / 20.0))
    on = kick_onsets(bpm, seconds)
    for o in on:
        i = int(round(o * fs))
        m = min(kk.size, n - i)
        x[i:i + m] += kk[:m]
    return _stereo(x), on


def drum_loop(fs: int = FS, beats: int = 40, seed: int = 3):
    """S3: the electronic loop from tests/test_average_peak.py (120 BPM: kick, snare, hats,
    55 Hz bass, pink pad)."""
    mod = load_test_module("test_average_peak")
    x = mod.drum_loop(fs=fs, beats=beats, seed=seed)
    return x, 120.0


def probe_1k(fs: int = FS, seconds: float = 20.0, tone_dbfs: float = -20.0,
             burst_dbfs: float = -3.0, burst_ms: float = 5.0, period_s: float = 0.5, seed: int = 7):
    """S5: 1 kHz sine at tone_dbfs plus a burst_ms white-noise burst at burst_dbfs peak every
    period_s. Returns (x, burst_starts_s)."""
    n = int(round(seconds * fs))
    t = np.arange(n) / fs
    x = np.zeros((n, 2))
    x[:, 0] = x[:, 1] = 10.0 ** (tone_dbfs / 20.0) * np.sin(2.0 * np.pi * 1000.0 * t)
    rng = np.random.default_rng(seed)
    nb = int(round(burst_ms * 1e-3 * fs))
    w = np.hanning(nb)
    starts = np.arange(0.25, seconds - 0.1, period_s)
    for s in starts:
        i = int(round(s * fs))
        b = rng.standard_normal((nb, 2)) * w[:, None]
        b *= 10.0 ** (burst_dbfs / 20.0) / np.max(np.abs(b))
        x[i:i + nb] += b
    return x, starts


def pink_bursts(fs: int = FS, seconds: float = 20.0, burst_s: float = 0.2, period_s: float = 0.5,
                fade_s: float = 0.05, dbfs: float = -3.0, seed: int = 11):
    """S6: pink-noise bursts with fade_s raised-cosine fades, peak dbfs."""
    from fixtures import synthetic as syn
    n = int(round(seconds * fs))
    pink = syn.pink_noise(n, fs, rms_dbfs=-20.0, seed=seed, channels=2)
    env = np.zeros(n)
    nb, nf = int(round(burst_s * fs)), int(round(fade_s * fs))
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(nf) / nf))
    for s in np.arange(0.2, seconds - burst_s, period_s):
        i = int(round(s * fs))
        env[i:i + nf] = np.maximum(env[i:i + nf], ramp)
        env[i + nf:i + nb - nf] = 1.0
        env[i + nb - nf:i + nb] = np.maximum(env[i + nb - nf:i + nb], ramp[::-1])
    x = pink * env[:, None]
    x *= 10.0 ** (dbfs / 20.0) / np.max(np.abs(x))
    return x


def hits_for(x, fs, bpm: float):
    """envelope.hits_for on a bpm grid (the planner's hits with --bpm)."""
    from eluate import envelope
    beat = 60.0 / bpm
    dur = x.shape[0] / fs
    return envelope.hits_for(x, fs, np.arange(beat, dur, beat))
