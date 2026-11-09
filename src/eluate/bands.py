"""Band splitting and M/S conversion (spec 3.1).

Contract:
- `split(x, fs, fcs, order, phase)` returns an array of shape
  (len(fcs) + 1, n, ch), float64, bands ordered low to high. The split is a
  cascade upward from the lowest crossover: at crossover k the current residual
  r is split into a low part (band k) and a high part (the next residual).
- `phase="zero"` (per crossover): low = sosfiltfilt(butter(order, fc)) of r,
  high = r - low. Magnitude of low is |H_bw|^2 (-6.02 dB at fc) and high is
  exactly 1 - |H_bw|^2 = |H_bw,hp|^2. The sum of all bands equals x to double
  precision for any padding, because every high part is defined as a
  difference.
- `phase="minimum"` (per crossover): Linkwitz-Riley of order 2*order (order=2
  gives LR4): low = butter_lp(order) applied twice causally, high = butter_hp
  applied twice. Every band below a minimum-phase crossover is passed through
  that crossover's allpass (LP^2 + HP^2 = D(-s)/D(s)), so the sum of bands is
  an allpass: flat magnitude, not sample-identical to x. Requires even order.
- `phase` may be a single mode or one mode per crossover (e.g. minimum at the
  lowest crossover, zero elsewhere); the sum is then the product of the
  allpasses of the minimum-phase crossovers.
- For any even order the two modes are magnitude-identical per band: the
  zero-phase low part is |B_N|^2 and the Linkwitz-Riley low part is B_N^2, the
  same magnitude (tested for orders 2, 4, 6). Default order 2 (LR4).
- Edges (`edge`, zero-phase crossovers only). Padding length is
  max(512, ceil(PAD_TAU * fs / (2 pi fc_min))) samples per side.
  `edge="zero"` (default): the residual is padded with digital silence and
  filtered from rest, so the bands are those of the track embedded in silence,
  which is what the file is on playback. They match a long zero-padded
  reference to about -250 dB whether the track starts from silence, from a
  fade or abruptly; padding only has to outlast the filter tails (PAD_TAU = 20
  leaves about -120 dB). An abrupt start or end at level therefore shows the
  zero-phase filter's real pre-ring / post-ring in the bands (the band sum is
  still exact); it is reported, not suppressed.
  `edge="odd"`: odd reflection as in spec draft 0.3, capped at len(x) - 1 with
  a RuntimeWarning. It masks some of an abrupt start's transient (-12.4 vs
  -8.4 dB low-band deviation in the first 10 ms of a full-scale 45 Hz cosine,
  fc 60 Hz) but differs from the zero-padded reference by -52 dB on a 50 ms
  linear fade with strong content near fc, failing the -70 dB gate; see
  audit/brief-02.md D.9.
  Minimum-phase crossovers are causal and start from rest, i.e. the same
  silence assumption. Inputs shorter than 2 samples raise ValueError, as does
  NaN or Inf.
- `split_iter(...)` yields the same bands one at a time (bit-identical) and holds
  just the current band and the residual, for every phase mode (since 2026-11-10,
  when the planner made the lowest crossover minimum phase; before, a minimum-phase
  crossover computed the whole split first). `split` fills a preallocated output
  from it. Zero-phase
  filtering runs one column at a time (sosfilt treats columns independently,
  so the result is unchanged) to hold one padded column, not all. The edge "zero"
  pass is iir.sosfiltfilt_zero (2026-11-10, filter-maths audit F5): blocked, with
  filter state below 1e-150 flushed to 0 and silent blocks skipped, so digital
  silence no longer runs the tails in the subnormal range (10x slower); identical to
  scipy's sosfiltfilt bit for bit unless a flush happens, within 1e-12 when it does.
- `to_ms` / `to_lr`: M = (L + R) / sqrt(2), S = (L - R) / sqrt(2), and the
  inverse with the same factor. Orthonormal: energy-preserving and exactly
  self-inverse up to rounding.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np
from scipy import signal

from eluate import iir

# x4 spacing (docs/gain-staging.md 4); planner.key_crossovers moves the lowest with the key
DEFAULT_CROSSOVERS: tuple[float, ...] = (100.0, 400.0, 1600.0, 6400.0)
DEFAULT_ORDER = 2
INV_SQRT2 = 1.0 / math.sqrt(2.0)

# Padding in time constants 1 / (2 pi fc_min) (spec draft 0.3: 20). The
# butter(2) tails decay as exp(-2 pi fc t / sqrt(2)); 5 time constants leave
# about -31 dB, 20 leave about -120 dB.
PAD_TAU = 20.0
PAD_MIN = 512


def padlen(n: int | None, fs: float, fc_min: float, tau: float | None = None) -> int:
    """Pad length for zero-phase filtering. With `n` given (odd reflection) it is
    capped at n - 1 and a RuntimeWarning is issued when the cap bites."""
    tau = PAD_TAU if tau is None else tau
    want = max(PAD_MIN, math.ceil(tau * fs / (2.0 * math.pi * fc_min)))
    if n is not None and want > n - 1:
        warnings.warn(
            f"zero-phase padding capped at {n - 1} samples (wanted {want}); "
            "the band edges are no longer free of filter start-up transients",
            RuntimeWarning, stacklevel=3)
        return max(0, n - 1)
    return want


def _phases(phase: str | Sequence[str], n_xo: int) -> list[str]:
    if isinstance(phase, str):
        modes = [phase] * n_xo
    else:
        modes = list(phase)
    if len(modes) != n_xo:
        raise ValueError(f"need {n_xo} phase modes, got {len(modes)}")
    for m in modes:
        if m not in ("zero", "minimum"):
            raise ValueError(f"phase must be 'zero' or 'minimum', got {m!r}")
    return modes


def allpass_sos(lp_sos: np.ndarray) -> np.ndarray:
    """Allpass with the same poles as `lp_sos`: each section [a2, a1, a0]/[a0, a1, a2].

    For an even-order Butterworth pair, LP^2 + HP^2 equals this allpass exactly
    (analog identity D(s)D(-s) = 1 + s^2N, preserved by the bilinear transform).
    """
    ap = np.array(lp_sos, dtype=np.float64, copy=True)
    a = ap[:, 3:6]
    # first-order sections are stored as [b0, b1, 0, 1, a1, 0]
    for i in range(ap.shape[0]):
        if a[i, 2] == 0.0:
            ap[i, 0:3] = [a[i, 1], a[i, 0], 0.0]
        else:
            ap[i, 0:3] = a[i, ::-1]
    return ap


def _check_split(x, fs, fcs, order, phase, edge):
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    n = x.shape[0]
    if n < 2:
        raise ValueError("need at least 2 samples")
    if not np.all(np.isfinite(x)):
        raise ValueError("audio contains NaN or Inf")
    fcs = [float(f) for f in fcs]
    if any(b <= a for a, b in zip(fcs, fcs[1:])):
        raise ValueError("crossovers must be strictly ascending")
    if fcs and (fcs[0] <= 0 or fcs[-1] >= fs / 2):
        raise ValueError("crossovers must lie in (0, fs/2)")
    modes = _phases(phase, len(fcs))
    if "minimum" in modes and order % 2:
        raise ValueError("minimum-phase (Linkwitz-Riley) mode needs an even order")
    if edge not in ("zero", "odd"):
        raise ValueError(f"edge must be 'zero' or 'odd', got {edge!r}")
    return x, squeeze, fcs, modes


def _zero_low(r: np.ndarray, lp: np.ndarray, pl: int, edge: str) -> np.ndarray:
    """Zero-phase low part of r, one column at a time (sosfilt treats columns
    independently, so this is bit-identical to filtering (n, ch) at once and
    holds one padded column instead of all of them)."""
    lo = np.empty_like(r)
    for c in range(r.shape[1]):
        if edge == "zero":
            # blocked with the subnormal flush (iir.py): scipy's sosfiltfilt of the
            # zero-padded column bit for bit unless digital silence lets the tail flush
            lo[:, c] = iir.sosfiltfilt_zero(lp, r[:, c], pl)
        else:
            lo[:, c] = signal.sosfiltfilt(lp, r[:, c], padtype="odd", padlen=pl)
    return lo


def split_iter(
    x: np.ndarray,
    fs: float,
    fcs: Sequence[float] = DEFAULT_CROSSOVERS,
    order: int = DEFAULT_ORDER,
    phase: str | Sequence[str] = "zero",
    edge: str = "zero",
):
    """Yield the bands of `split` one at a time, low to high, each (n, ch) (or (n,)).

    At most the current band and the residual are alive (spec 1 memory), for every
    phase mode: band k is passed through the allpass of each minimum-phase crossover
    above it as soon as it is split (the allpasses depend only on the crossover
    frequencies), in the order of the crossovers, so the bands are bit-identical to
    the whole-split computation used before 2026-11-10 (tested against it)."""
    x2, squeeze, fcs, modes = _check_split(x, fs, fcs, order, phase, edge)
    del x                            # only r refers to the input from here: a caller that
    n = x2.shape[0]                  # passes a temporary lets it go after the first band
    pl = (padlen(None if edge == "zero" else n, fs, min(fcs))
          if fcs and "zero" in modes else 0)
    lps = [signal.butter(order, fc, btype="lowpass", fs=fs, output="sos") for fc in fcs]
    aps = [allpass_sos(lp) if m == "minimum" else None for lp, m in zip(lps, modes)]
    r = x2
    del x2
    own = False                      # r is the caller's array until the first residual
    for k, (fc, mode) in enumerate(zip(fcs, modes)):
        lp = lps[k]
        if mode == "zero":
            lo = _zero_low(r, lp, pl, edge)
            if own:
                np.subtract(r, lo, out=r)    # same values as r - lo, no new array
                hi = r
            else:
                hi = r - lo
        else:
            hp = signal.butter(order, fc, btype="highpass", fs=fs, output="sos")
            lo = signal.sosfilt(lp, signal.sosfilt(lp, r, axis=0), axis=0)
            hi = signal.sosfilt(hp, signal.sosfilt(hp, r, axis=0), axis=0)
        own = True
        # phase-align: band k passes through the allpass of every minimum-phase
        # crossover above it, so the band sum is the product of those allpasses
        for j in range(k + 1, len(fcs)):
            if aps[j] is not None:
                lo = signal.sosfilt(aps[j], lo, axis=0)
        yield lo[:, 0] if squeeze else lo
        del lo
        r = hi
    yield r[:, 0] if squeeze else r


def split(
    x: np.ndarray,
    fs: float,
    fcs: Sequence[float] = DEFAULT_CROSSOVERS,
    order: int = DEFAULT_ORDER,
    phase: str | Sequence[str] = "zero",
    edge: str = "zero",
) -> np.ndarray:
    """Split `x` (n,) or (n, ch) into len(fcs)+1 bands, shape (bands, n, ch)."""
    x, squeeze, fcs, modes = _check_split(x, fs, fcs, order, phase, edge)
    n = x.shape[0]
    # fill a preallocated output from split_iter (no second copy of every band)
    out = np.empty((len(fcs) + 1, n, x.shape[1]))
    for k, b in enumerate(split_iter(x, fs, fcs, order, phase, edge)):
        out[k] = b
    return out[:, :, 0] if squeeze else out


def to_ms(x: np.ndarray) -> np.ndarray:
    """L/R (n, 2) -> M/S (n, 2), M = (L+R)/sqrt2, S = (L-R)/sqrt2."""
    x = np.asarray(x, dtype=np.float64)
    m = (x[:, 0] + x[:, 1]) * INV_SQRT2
    s = (x[:, 0] - x[:, 1]) * INV_SQRT2
    return np.stack([m, s], axis=1)


def to_lr(ms: np.ndarray) -> np.ndarray:
    """M/S (n, 2) -> L/R (n, 2), L = (M+S)/sqrt2, R = (M-S)/sqrt2."""
    ms = np.asarray(ms, dtype=np.float64)
    left = (ms[:, 0] + ms[:, 1]) * INV_SQRT2
    right = (ms[:, 0] - ms[:, 1]) * INV_SQRT2
    return np.stack([left, right], axis=1)
