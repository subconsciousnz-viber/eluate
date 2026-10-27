"""Shared machinery for the analysis pass (spec 3.3): the common time grid,
centred-window statistics, and a cache of intermediates every module reuses.

Contract:
- Grid: frame k covers [k h, (k + 1) h) with h = `hop_s` (default 0.1 s) and is
  stamped at its centre t_k = (k + 0.5) h; K = ceil(n / (h fs)) frames, so the
  last frame may extend past the end of the signal. Every time series returned
  by features/ has length K on this grid.
- Windowed statistics (`window_mean`, `window_max`) cover the `length` samples
  [c - length // 2, c - length // 2 + length) around centre sample c. The
  signal is treated as embedded in digital silence (zero outside its span), the
  same convention as bands.py's default edge mode. `window_mean(inside=True)`
  averages over the in-signal part of the window only; crest and stereo use it
  so ratios at the first and last frames are not skewed by the zeros.
- `Context` holds the input as float64 (n, 2) L/R plus lazily computed shared
  intermediates: the 5-band split (bands.split, zero phase, default crossovers)
  and the mono analysis signal (L + R) / 2 resampled to AR = 20480 Hz, where a
  hop of 2048 samples is exactly 100 ms. Modules accept a Context so the split
  and the resampling are done once per analysis.
- No randomness anywhere: the analysis is a deterministic function of its input.
"""

from __future__ import annotations

import math
from functools import cached_property

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d

from .. import bands as _bands
from ..io import as_stereo

AR = 20480  # analysis rate for librosa features; 2048-sample hop = 100 ms
HOP_S = 0.1
VIEWS = ("M", "S", "stereo")
EPS = 1e-30


def as_audio(x: np.ndarray) -> np.ndarray:
    """(n,), (n, 1) or (n, 2) -> float64 (n, 2); NaN/Inf and n < 2 raise."""
    x = np.asarray(as_stereo(np.asarray(x)), dtype=np.float64)
    if x.shape[0] < 2:
        raise ValueError("need at least 2 samples")
    if not np.all(np.isfinite(x)):
        raise ValueError("audio contains NaN or Inf")
    return x


class Grid:
    """The common analysis grid (see module docstring)."""

    def __init__(self, n: int, fs: float, hop_s: float = HOP_S):
        self.n = int(n)
        self.fs = float(fs)
        self.hop_s = float(hop_s)
        self.k = max(1, int(math.ceil(self.n / (self.hop_s * self.fs) - 1e-9)))
        self.times = (np.arange(self.k) + 0.5) * self.hop_s
        self.centres = np.round(self.times * self.fs).astype(np.int64)

    def index(self, t) -> np.ndarray:
        """Grid frame index containing time(s) t, clipped to [0, K - 1]."""
        return np.clip(np.floor(np.asarray(t) / self.hop_s).astype(np.int64), 0, self.k - 1)

    def frames_in(self, t0: float, t1: float) -> np.ndarray:
        """Boolean mask of grid frames whose centre lies in [t0, t1)."""
        return (self.times >= t0) & (self.times < t1)

    def to_grid(self, t_src: np.ndarray, v_src: np.ndarray) -> np.ndarray:
        """Linear interpolation of a series stamped at t_src onto the grid
        (held constant beyond the ends; NaN-aware: NaN where both neighbours are NaN)."""
        t_src = np.asarray(t_src, dtype=np.float64)
        v_src = np.asarray(v_src, dtype=np.float64)
        if t_src.size == 0:
            return np.full(self.k, np.nan)
        good = np.isfinite(v_src)
        if not np.any(good):
            return np.full(self.k, np.nan)
        out = np.interp(self.times, t_src[good], v_src[good])
        # NaN on grid frames whose nearest source frame is NaN (true nearest
        # neighbour, not the next one: audit 04 A14)
        j = np.clip(np.searchsorted(t_src, self.times), 0, t_src.size - 1)
        jm = np.clip(j - 1, 0, t_src.size - 1)
        near = np.where(np.abs(t_src[jm] - self.times) <= np.abs(t_src[j] - self.times), jm, j)
        out[~good[near]] = np.nan
        return out


def window_mean(sig: np.ndarray, centres: np.ndarray, length: int, inside: bool = False) -> np.ndarray:
    """Mean of `sig` (n,) or (n, c) over centred windows, zero outside the span.
    With `inside`, the mean is over the part of each window inside the signal
    (so a window half past the end is not diluted by the silence there)."""
    sig = np.asarray(sig, dtype=np.float64)
    n = sig.shape[0]
    c = np.zeros((n + 1,) + sig.shape[1:])
    np.cumsum(sig, axis=0, out=c[1:])
    a = np.clip(centres - length // 2, 0, n)
    b = np.clip(centres - length // 2 + length, 0, n)
    if inside:
        cnt = np.maximum(b - a, 1).astype(np.float64)
        return (c[b] - c[a]) / (cnt if sig.ndim == 1 else cnt[:, None])
    return (c[b] - c[a]) / float(length)


def window_max(sig: np.ndarray, centres: np.ndarray, length: int) -> np.ndarray:
    """Max of `sig` (n,) over centred windows (sig >= 0 assumed; zero outside the span)."""
    sig = np.asarray(sig, dtype=np.float64)
    n = sig.shape[0]
    tail = max(0, int(centres.max()) + length - n + 1) if centres.size else 0
    s = np.pad(sig, (0, tail)) if tail else sig
    # size L, origin 0: position i covers [i - L//2, i + (L - 1)//2]
    m = maximum_filter1d(s, size=length, mode="constant", cval=0.0)
    return m[np.clip(centres, 0, s.shape[0] - 1)]


def db_power(p, floor: float = -200.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.maximum(10.0 * np.log10(np.asarray(p, dtype=np.float64)), floor)


def resample(y: np.ndarray, fs: float, fs_out: float) -> np.ndarray:
    """Polyphase resampling along axis 0 (deterministic)."""
    fs_i, fo_i = int(round(fs)), int(round(fs_out))
    if fs_i == fo_i:
        return np.asarray(y, dtype=np.float64)
    g = math.gcd(fs_i, fo_i)
    return signal.resample_poly(np.asarray(y, dtype=np.float64), fo_i // g, fs_i // g, axis=0)


def fft_size(n_ref: int, fs: float) -> int:
    """Spec FFT sizes are for 44.1 / 48 kHz; at 88.2 kHz and above they are scaled
    by powers of two so the frequency resolution in Hz is kept."""
    return int(n_ref * 2 ** max(0, round(math.log2(fs / 48000.0))))


class Context:
    """Input plus cached shared intermediates for one analysis."""

    def __init__(self, x: np.ndarray, fs: float, fcs=_bands.DEFAULT_CROSSOVERS,
                 hop_s: float = HOP_S):
        self.x = as_audio(x)
        self.fs = float(fs)
        self.fcs = tuple(float(f) for f in fcs)
        self.grid = Grid(self.x.shape[0], self.fs, hop_s)
        self.cache: dict = {}

    @cached_property
    def bands(self) -> np.ndarray:
        """(B, n, 2) zero-phase band split of L/R (bands.split defaults)."""
        return _bands.split(self.x, self.fs, self.fcs)

    @cached_property
    def ms(self) -> np.ndarray:
        return _bands.to_ms(self.x)

    @cached_property
    def mono_ar(self) -> np.ndarray:
        """(L + R) / 2 at AR Hz (a mono file keeps its level)."""
        return resample(0.5 * (self.x[:, 0] + self.x[:, 1]), self.fs, AR)

    def iter_bands(self):
        """Yield (b, (n, 2) band) low to high; bit-identical to self.bands[b]. Uses the
        cached split when it exists, else bands.split_iter, which holds one band and
        the residual instead of all of them (spec 1 memory)."""
        if "bands" in self.__dict__:
            yield from enumerate(self.bands)
            return
        yield from enumerate(_bands.split_iter(self.x, self.fs, self.fcs))

    def release(self, *names: str) -> None:
        """Drop cached intermediates (cached properties or cache keys) no later
        step needs; they are recomputed identically if asked for again."""
        for k in names:
            self.__dict__.pop(k, None)
            self.cache.pop(k, None)

    def band_ms(self, b: int) -> np.ndarray:
        """M/S of band b, (n, 2). Linear, so equal to the band of M/S."""
        return _bands.to_ms(self.bands[b])

    @cached_property
    def low_band_m(self) -> np.ndarray:
        """M component of the bands below the second crossover (sub + low)."""
        if "bands" in self.__dict__:
            lo = self.bands[0] + self.bands[1] if len(self.fcs) >= 2 else self.bands[0]
        else:
            it = self.iter_bands()
            lo = next(it)[1]
            if len(self.fcs) >= 2:
                lo = lo + next(it)[1]
            it.close()
        return _bands.to_ms(lo)[:, 0]


def ensure_context(x, fs, ctx: Context | None, hop_s: float = HOP_S) -> Context:
    return ctx if ctx is not None else Context(x, fs, hop_s=hop_s)


def librosa_on_grid(feat: np.ndarray, k: int) -> np.ndarray:
    """(d, frames) librosa feature computed at AR with hop 2048 on a signal whose
    first 1024 samples were dropped (so frame i is centred at (i + 0.5) * 0.1 s);
    trimmed or edge-padded to k frames."""
    if feat.shape[1] >= k:
        return feat[:, :k]
    return np.pad(feat, ((0, 0), (0, k - feat.shape[1])), mode="edge")
