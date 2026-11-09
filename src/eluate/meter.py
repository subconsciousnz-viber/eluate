"""Loudness and true-peak metering (spec 3.2). Own ITU-R BS.1770-4 implementation.

Contract:
- Input `x` is (n,) mono or (n, ch). Channel weights default to 1.0 each;
  `weights` gives BS.1770 G_i (e.g. 1.41 for surrounds). NaN or Inf input raises
  ValueError. Arithmetic in float64.
- K-weighting: two biquads (high shelf f0 1681.974 Hz, Q 0.70717, +3.99984 dB;
  RLB high-pass f0 38.13547 Hz, Q 0.50033) designed at the native rate by the
  pre-warped bilinear transform (K = tan(pi f0 / fs), libebur128 form; the
  convention libebur128 and pyloudnorm also use). The RLB numerator [1, -2, 1]
  is scaled by a0(48 kHz) / a0(fs), a0 = 1 + K/Q + K^2, so the stage's passband
  gain equals the 48 kHz table filter's (+0.0433 dB) at every rate; unscaled it
  drifts with fs (audit 02 A5). At 48 kHz the coefficients equal BS.1770
  Tables 1 and 2 to < 1e-15. Measured 997 Hz full-scale readings: see tests.
- The K-filter starts from the steady state of the first sample
  (scipy.signal.sosfilt_zi), so DC reads -inf and an abrupt start does not
  inject a start-up block (audit 02 A16). This is deliberate and kept (metering
  audit 2026-11-10, finding 11): libebur128, pyloudnorm and ffmpeg start from a
  zero state, which moves only the first 400 ms block, by at most 0.04 LU
  either way; whole-track readings are unaffected.
- Loudness of a power p (weighted sum over channels of mean square of the
  K-weighted signal) is -0.691 + 10 log10(p) LUFS; silence gives -inf.
- Gating blocks are 400 ms at 75 % overlap; block j starts at round(0.1 j fs)
  (hops are uniform to within one sample at rates not divisible by 10, e.g.
  1102/1103 at 11025 Hz); incomplete trailing blocks are dropped. Integrated =
  mean power of blocks above -70 LUFS and above (abs-gated loudness - 10 LU);
  -inf when no block passes (silence, or input shorter than 400 ms).
- `momentary` / `short_term`: 400 ms / 3 s windows at 100 ms hop, full windows
  only, time stamp = window end. `momentary_max` / `short_term_max` slide the
  window one sample at a time (EBU Tech 3341 cases 10 and 13), as the maximum
  of c[i + L] - c[i] over the power's cumulative sum taken in chunks of
  SLIDE_CHUNK starts (no per-start index arrays; metering audit 2026-11-10,
  finding 5). A maximum window power below P_FLOOR (the cumulative-sum residue
  of a K-weighted DC, about -325 LUFS) reads -inf, as silence does (finding 10).
- `lra`: EBU Tech 3342: short-term values at 100 ms hop, -70 LUFS absolute
  gate, -20 LU relative gate, LRA = L95 - L10 taken from the sorted gated
  values at index floor((N - 1) p + 0.5), libebur128's rule (verified against
  its source). NaN when not measurable (input < 3 s or nothing above the gates).
- True peak: `oversample=8` (default; the meter's reading, which the solver
  constrains together with the 32x FFT reading at 24 bits and with the 4x
  reading too under the -1 dBTP deliveries; evaluate.measure_track's
  `true_peak_max_dbtp` is the maximum of all three) uses a Kaiser-windowed
  sinc interpolator: passband to 0.93 x Nyquist, -6 dB at 0.965, >= 60 dB
  image rejection from 1.0 x Nyquist (so it under-reads full-band material by
  up to 0.5 dB against the 32x reading). `oversample=4` uses the 48-tap,
  4-phase FIR of BS.1770-4 Annex 2 (coefficients verified against the
  BS.1770-5 PDF), reported for compliance.
  Each input sample owns exactly `oversample` interpolated values (floor
  mapping). The signal is silent outside its span, so reconstruction ringing at
  an abrupt start / end counts. True peak is never below the sample peak;
  empty input gives -inf. Returned in dBTP, maximum over channels.
- `upsample(x, factor)` / `decimate(y, factor)`: the same interpolation
  prototype used as a time-aligned resampler pair, for the 8x limiter path.
- `loud_passage_lufs`: S95 = 95th percentile of short-term loudness over windows
  above -70 LUFS (control target); loud-section integrated = gated integrated
  loudness over the 400 ms blocks lying entirely inside the union of windows
  with S >= S_max - 3 LU (reported). Inputs from 0.4 to 3 s fall back to
  momentary windows (`window_s` says which); shorter or silent input gives NaN.
- PLR = true peak - integrated. PSR(t) = true peak inside each short-term
  window - S(t); windows below -70 LUFS are NaN; `psr` also returns the 95th
  percentile, NaN when not measurable.
- 1 s RMS (docs/gain-staging.md 3, 9; the maintainer's iZotope output meter and the
  corpus metric): `rms_series` slides a RMS_WINDOW_S (1 s) window at a 100 ms
  hop over the unweighted signal, power = mean over channels of the mean
  square, full windows only (the same window placement as `momentary`), plain
  dBFS: a full-scale sine reads -3.01, full-scale DC 0. `rms_1s` returns the
  loudest-section value (p95 of the series over windows above RMS_GATE_DBFS,
  -70 dBFS) and the loudest 1 s value (the series maximum). `aes17=True` adds
  AES17_OFFSET_DB (10 log10 2 = 3.01 dB) so a full-scale sine reads 0 dBFS
  (iZotope's AES17 option); the gate is applied to the plain values either
  way. Input shorter than 1 s or with no window above the gate gives NaN.
- The maintainer's meter (`rms_meter`, 2026-10-19): the same RMS on a 300 ms window at a
  10 ms hop, AES17, p95 over gated windows = the loudest passage as read off the
  Ozone RMS bar. This is the loudness target's reading.
- `crest_db` (the one crest reading the CLI, the report, the app and the corpus survey
  share): sample peak minus the loudest-section 1 s RMS (`rms_1s`, plain dBFS). NaN when
  the RMS is not measurable.
- `measure` computes the K-weighted power, its cumulative sum, the 400 ms and 3 s
  window powers, the unweighted power's cumulative sum and the 8x envelope once each
  and shares them between the readings (integrated, S95 / loud section, PSR,
  short-term and momentary maxima, LRA, true peak, the RMS meters); every value equals
  the stand-alone function's (tests/test_meter.py, metering audit 2026-11-10 finding 5).
- The 24-bit export (stages/output.py) is rounded without dither, deliberately
  (docs/gain-staging.md 11, deviation 3; metering audit 2026-11-10 finding 15: the
  rounding error sits near -144 dBFS, far below any converter's floor); only the
  16-bit CD export is dithered (TPDF).
"""

from __future__ import annotations

import math
import threading
from functools import lru_cache

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d

from eluate import parallel
from eluate.oa import oa_stream

ABS_GATE = -70.0
REL_GATE = -10.0
LRA_REL_GATE = -20.0
BLOCK_S = 0.4
SHORT_S = 3.0
HOP_S = 0.1
OFFSET = -0.691
TP_OVERSAMPLE = 8
RMS_WINDOW_S = 1.0
RMS_GATE_DBFS = -70.0
AES17_OFFSET_DB = 10.0 * math.log10(2.0)
METER_WINDOW_S = 0.3              # the maintainer's meter (rms_meter): Ozone RMS, 300 ms
METER_HOP_S = 0.01
# a window power below this is silence (-inf), not a reading: the cumulative-sum residue of
# a K-weighted DC (about 1e-33, -325 LUFS; metering audit 2026-11-10 finding 10). One 24-bit
# step of noise is about 3e-15 (-144 LUFS), far above it.
P_FLOOR = 1e-30
SLIDE_CHUNK = 1 << 20             # window starts per chunk of the one-sample-hop maxima

# BS.1770-4 Annex 2, order-48 4-phase interpolating FIR. Row p = phase p.
ANNEX2_PHASES = np.array(
    [
        [0.0017089843750, 0.0109863281250, -0.0196533203125, 0.0332031250000,
         -0.0594482421875, 0.1373291015625, 0.9721679687500, -0.1022949218750,
         0.0476074218750, -0.0266113281250, 0.0148925781250, -0.0083007812500],
        [-0.0291748046875, 0.0292968750000, -0.0517578125000, 0.0891113281250,
         -0.1665039062500, 0.4650878906250, 0.7797851562500, -0.2003173828125,
         0.1015625000000, -0.0582275390625, 0.0330810546875, -0.0189208984375],
        [-0.0189208984375, 0.0330810546875, -0.0582275390625, 0.1015625000000,
         -0.2003173828125, 0.7797851562500, 0.4650878906250, -0.1665039062500,
         0.0891113281250, -0.0517578125000, 0.0292968750000, -0.0291748046875],
        [-0.0083007812500, 0.0148925781250, -0.0266113281250, 0.0476074218750,
         -0.1022949218750, 0.9721679687500, 0.1373291015625, -0.0594482421875,
         0.0332031250000, -0.0196533203125, 0.0109863281250, 0.0017089843750],
    ]
)


def _as2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x[:, None] if x.ndim == 1 else x
    if x.ndim != 2:
        raise ValueError(f"audio must be (n,) or (n, ch), got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("audio contains NaN or Inf")
    return x


def _lufs(p):
    with np.errstate(divide="ignore"):
        return OFFSET + 10.0 * np.log10(p)


# ---------------------------------------------------------------- K-weighting

_SHELF = (1681.974450955533, 3.999843853973347, 0.7071752369554196)
_RLB = (38.13547087602444, 0.5003270373238773)


def _rlb_a0(fs: float) -> float:
    f0, q = _RLB
    k = math.tan(math.pi * f0 / fs)
    return 1.0 + k / q + k * k


def k_weighting_sos(fs: float) -> np.ndarray:
    """Two-section SOS of the BS.1770 K-weighting filter at sample rate `fs`."""
    f0, gain_db, q = _SHELF
    k = math.tan(math.pi * f0 / fs)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / q + k * k
    shelf = [
        (vh + vb * k / q + k * k) / a0,
        2.0 * (k * k - vh) / a0,
        (vh - vb * k / q + k * k) / a0,
        1.0,
        2.0 * (k * k - 1.0) / a0,
        (1.0 - k / q + k * k) / a0,
    ]
    f0, q = _RLB
    k = math.tan(math.pi * f0 / fs)
    a0 = 1.0 + k / q + k * k
    gain = _rlb_a0(48000.0) / a0  # passband gain pinned to the 48 kHz table filter
    hp = [gain, -2.0 * gain, gain, 1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]
    return np.array([shelf, hp])


def k_weight(x: np.ndarray, fs: float) -> np.ndarray:
    """K-weighted signal, (n, ch) float64, filter state = steady state of x[0]."""
    x = _as2d(x)
    sos = k_weighting_sos(fs)
    if x.shape[0] == 0:
        return x.copy()
    zi = signal.sosfilt_zi(sos)[:, :, None] * x[0][None, None, :]
    y, _ = signal.sosfilt(sos, x, axis=0, zi=zi)
    return y


def _weighted_power(y: np.ndarray, weights) -> np.ndarray:
    w = np.ones(y.shape[1]) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (y.shape[1],):
        raise ValueError(f"need {y.shape[1]} channel weights, got {w.shape}")
    return (y * y) @ w


# ------------------------------------------------------------- block powers

def _cumsum(ps: np.ndarray) -> np.ndarray:
    """c with c[0] = 0 and c[i] = sum(ps[:i]), so a window's summed power is c[i + L] - c[i]."""
    c = np.zeros(ps.shape[0] + 1)
    np.cumsum(ps, out=c[1:])
    return c


def _windows_from_cumsum(c: np.ndarray, fs: float, win_s: float, hop_s: float = HOP_S):
    """`_window_powers` from the power's cumulative sum `c` (shared between readings)."""
    n = c.shape[0] - 1
    length = int(round(win_s * fs))
    hop = hop_s * fs
    if n < length or length == 0:
        return np.zeros(0, dtype=np.int64), length, np.zeros(0)
    count = int(math.floor((n - length) / hop + 1e-9)) + 1
    starts = np.round(np.arange(count) * hop).astype(np.int64)
    starts = starts[starts + length <= n]
    return starts, length, (c[starts + length] - c[starts]) / length


def _window_powers(ps: np.ndarray, fs: float, win_s: float, hop_s: float = HOP_S):
    """Mean of the per-sample weighted power `ps` over full windows at a fixed hop.

    Returns (starts, length, powers). Window j starts at round(j * hop_s * fs).
    """
    return _windows_from_cumsum(_cumsum(ps), fs, win_s, hop_s)


def _gated_mean_power(p: np.ndarray, rel_gate: float = REL_GATE) -> float:
    """BS.1770 two-stage gating on block powers; returns gated mean power (0 if none)."""
    if p.size == 0:
        return 0.0
    p_abs = p[_lufs(p) > ABS_GATE]
    if p_abs.size == 0:
        return 0.0
    gamma_r = _lufs(p_abs.mean()) + rel_gate
    p_rel = p_abs[_lufs(p_abs) > gamma_r]
    return float(p_rel.mean()) if p_rel.size else 0.0


def _kpower(x, fs, weights):
    return _weighted_power(k_weight(x, fs), weights)


# ---------------------------------------------------------------- loudness

def momentary(x: np.ndarray, fs: float, hop_s: float = HOP_S, weights=None):
    """Momentary loudness (400 ms). Returns (t_end_s, lufs)."""
    starts, length, p = _window_powers(_kpower(x, fs, weights), fs, BLOCK_S, hop_s)
    return (starts + length) / fs, _lufs(p)


def short_term(x: np.ndarray, fs: float, hop_s: float = HOP_S, weights=None):
    """Short-term loudness (3 s). Returns (t_end_s, lufs)."""
    starts, length, p = _window_powers(_kpower(x, fs, weights), fs, SHORT_S, hop_s)
    return (starts + length) / fs, _lufs(p)


def _sliding_max_power(c: np.ndarray, length: int, chunk: int = SLIDE_CHUNK) -> float:
    """max over every window start i of c[i + length] - c[i] (the window's summed power),
    taken in chunks of `chunk` starts so no per-start index array exists. NaN when no full
    window fits."""
    n = c.shape[0] - 1
    if length == 0 or n < length:
        return math.nan
    last = n - length
    m = -math.inf
    for i in range(0, last + 1, chunk):
        j = min(i + chunk, last + 1)
        m = max(m, float(np.max(c[i + length:j + length] - c[i:j])))
    return m


def _sliding_max_lufs(ps: np.ndarray, fs: float, win_s: float, c: np.ndarray | None = None) -> float:
    """Loudness of the loudest window of `win_s` over every start (1-sample hop); `c` is
    the power's cumulative sum when the caller has it. -inf below P_FLOOR (silence, or
    the residue of a K-weighted DC); NaN when no full window fits."""
    length = int(round(win_s * fs))
    m = _sliding_max_power(_cumsum(ps) if c is None else c, length)
    if math.isnan(m):
        return math.nan
    p = m / length
    return -math.inf if p < P_FLOOR else float(_lufs(p))


def momentary_max(x: np.ndarray, fs: float, weights=None) -> float:
    """Maximum momentary loudness over every window position (1-sample hop)."""
    return _sliding_max_lufs(_kpower(x, fs, weights), fs, BLOCK_S)


def short_term_max(x: np.ndarray, fs: float, weights=None) -> float:
    """Maximum short-term loudness over every window position (1-sample hop)."""
    return _sliding_max_lufs(_kpower(x, fs, weights), fs, SHORT_S)


def _integrated_blocks(p: np.ndarray) -> float:
    """Gated integrated loudness from the 400 ms block powers at the 100 ms hop."""
    return float(_lufs(_gated_mean_power(p)))


def integrated(x: np.ndarray, fs: float, weights=None) -> float:
    """BS.1770-4 gated integrated loudness, LUFS (-inf if nothing passes the gates)."""
    _, _, p = _window_powers(_kpower(x, fs, weights), fs, BLOCK_S, HOP_S)
    return _integrated_blocks(p)


def _lra_from_short_term(s: np.ndarray) -> float:
    s = s[s > ABS_GATE]
    if s.size == 0:
        return math.nan
    p = 10.0 ** ((s - OFFSET) / 10.0)
    gate = _lufs(p.mean()) + LRA_REL_GATE
    s = np.sort(s[s > gate])
    if s.size == 0:
        return math.nan
    lo = s[int((s.size - 1) * 0.10 + 0.5)]
    hi = s[int((s.size - 1) * 0.95 + 0.5)]
    return float(hi - lo)


def lra(x: np.ndarray, fs: float, weights=None) -> float:
    """Loudness range per EBU Tech 3342 (libebur128 percentile rule), LU; NaN if not measurable."""
    _, s = short_term(x, fs, weights=weights)
    return _lra_from_short_term(s)


# --------------------------------------------------------------- true peak

@lru_cache(maxsize=8)
def _interp_prototype(factor: int, kind: str) -> np.ndarray:
    """Interpolation prototype h with h[L*k + p] = phase p, tap k."""
    if kind == "annex2":
        if factor != 4:
            raise ValueError("the Annex 2 filter is 4x only")
        return ANNEX2_PHASES.T.reshape(-1).copy()
    # Kaiser-windowed sinc: passband 0.93, stopband 1.0 (fractions of input Nyquist)
    width = 0.07 / factor  # normalised to the oversampled Nyquist
    numtaps, beta = signal.kaiserord(60.0, width)
    numtaps = int(math.ceil(numtaps / factor) * factor) + 1  # (numtaps-1)/2 is a multiple of factor/2
    return signal.firwin(numtaps, 0.965 / factor, window=("kaiser", beta)) * factor


def _kind(oversample: int, kind: str | None) -> str:
    return kind if kind is not None else ("annex2" if oversample == 4 else "kaiser")


def _centre(h: np.ndarray) -> int:
    return int(math.floor((h.shape[0] - 1) / 2.0 + 0.5))


def true_peak_envelope(x: np.ndarray, fs: float, oversample: int = TP_OVERSAMPLE,
                       kind: str | None = None) -> np.ndarray:
    """Per-input-sample max |interpolated value| over its phases and all channels.

    Output has length n. Interpolated index m = L*j + p maps to input sample
    (m - c) // L, c = round(filter centre), so every sample owns exactly L
    values; values that fall outside the signal span (filter tails at the
    edges) are folded into the first / last sample.
    """
    x = _as2d(x)
    n = x.shape[0]
    if n == 0:
        return np.zeros(0)
    if oversample == 1:
        return np.max(np.abs(x), axis=1)
    L = int(oversample)
    h = _interp_prototype(L, _kind(L, kind))
    c = _centre(h)
    env = np.zeros(n)
    # One channel at a time (oaconvolve along axis 0 treats columns independently,
    # so this is bit-identical), the convolution streamed block by block
    # (eluate.oa) and the index map folded in with slices: value m of phase p
    # belongs to sample (L m + p - c) // L = m + (p - c) // L, clipped to
    # [0, n - 1]. max is exact, so the result equals the maximum.at form while no
    # full-length convolution exists (spec 1 memory, audit 07 A15). The phase and
    # channel pairs run on parallel.pmap and fold under a lock: the order of the maxima
    # does not change the result (sprint WP8).
    lock = threading.Lock()

    def one(task):
        p, ch = task
        hp = h[p::L]
        off = (p - c) // L
        col = x[:, ch]

        def fold(start, vals):
            # convolution values [start, start + len) land on samples m + off
            a = np.abs(vals)
            m0, m1 = start, start + a.size
            j0, j1 = max(m0, -off), min(m1, n - off)
            with lock:
                if j1 > j0:
                    seg = env[j0 + off:j1 + off]
                    np.maximum(seg, a[j0 - m0:j1 - m0], out=seg)
                if m0 < -off:
                    env[0] = max(env[0], float(a[:min(a.size, -off - m0)].max()))
                if m1 > n - off:
                    env[n - 1] = max(env[n - 1], float(a[max(0, n - off - m0):].max()))

        def get(a0, b0):
            out = np.zeros(b0 - a0)
            if a0 < n:
                seg = col[a0:min(b0, n)]
                out[:seg.size] = seg
            return out

        # streamed (eluate.oa, bit-identical to oaconvolve; audit 07 A15)
        if not oa_stream(get, n, hp, fold):
            fold(0, signal.oaconvolve(col, hp, mode="full"))

    parallel.pmap(one, [(p, ch) for p in range(L) if h[p::L].size
                        for ch in range(x.shape[1])])
    for ch in range(x.shape[1]):
        np.maximum(env, np.abs(x[:, ch]), out=env)
    return env


def _env_db(env: np.ndarray) -> float:
    """The peak of a true-peak envelope in dBTP; -inf for an empty or silent one."""
    if env.size == 0:
        return -math.inf
    with np.errstate(divide="ignore"):
        return float(20.0 * np.log10(env.max()))


def true_peak(x: np.ndarray, fs: float, oversample: int = TP_OVERSAMPLE,
              kind: str | None = None) -> float:
    """True peak in dBTP (max over channels); -inf for empty or silent input."""
    return _env_db(true_peak_envelope(x, fs, oversample, kind))


def sample_peak(x: np.ndarray) -> float:
    x = _as2d(x)
    if x.size == 0:
        return -math.inf
    with np.errstate(divide="ignore"):
        return float(20.0 * np.log10(np.max(np.abs(x))))


def upsample(x: np.ndarray, factor: int = TP_OVERSAMPLE, kind: str | None = None) -> np.ndarray:
    """Interpolate by `factor` with the true-peak prototype: y[factor*i + p] ~ x(i + p/factor).

    Shape (n * factor, ch) (or (n * factor,) for 1-D input). Holds the whole
    oversampled signal in memory.
    """
    xa = _as2d(x)
    h = _interp_prototype(int(factor), _kind(factor, kind))
    c = _centre(h)
    y = signal.upfirdn(h, xa, up=int(factor), axis=0)[c:c + xa.shape[0] * int(factor)]
    return y[:, 0] if np.ndim(x) == 1 else y


def decimate(y: np.ndarray, factor: int = TP_OVERSAMPLE) -> np.ndarray:
    """Inverse of `upsample`: low-pass with the same Kaiser prototype, keep every factor-th."""
    ya = _as2d(y)
    h = _interp_prototype(int(factor), "kaiser") / factor
    c = _centre(h)
    z = signal.upfirdn(h, ya, up=1, down=1, axis=0)[c:c + ya.shape[0]][:: int(factor)]
    return z[:, 0] if np.ndim(y) == 1 else z


# ------------------------------------------------------ derived measures

def plr(x: np.ndarray, fs: float, oversample: int = TP_OVERSAMPLE, weights=None) -> float:
    """Peak-to-loudness ratio: true peak - integrated, dB."""
    return true_peak(x, fs, oversample) - integrated(x, fs, weights)


def psr(x: np.ndarray, fs: float, oversample: int = TP_OVERSAMPLE, hop_s: float = HOP_S,
        weights=None):
    """Peak-to-short-term-loudness ratio per 3 s window.

    Returns (t_end_s, psr_db, p95_db). Windows below -70 LUFS are NaN; p95 is
    NaN when no window is measurable (silence, input < 3 s).
    """
    starts, length, p = _window_powers(_kpower(x, fs, weights), fs, SHORT_S, hop_s)
    if starts.size == 0:
        return np.zeros(0), np.zeros(0), math.nan
    return _psr_from(starts, length, p, true_peak_envelope(x, fs, oversample), fs)


def _psr_from(starts, length, p, env, fs):
    """`psr` from the short-term window powers and the true-peak envelope."""
    if starts.size == 0:
        return np.zeros(0), np.zeros(0), math.nan
    # origin -(length // 2) makes wmax[i] = max(env[i : i + length])
    wmax = maximum_filter1d(env, size=length, origin=-(length // 2), mode="nearest")
    s = _lufs(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = 20.0 * np.log10(wmax[starts]) - s
    val = np.where(s > ABS_GATE, val, np.nan)
    good = val[np.isfinite(val)]
    p95 = float(np.percentile(good, 95.0)) if good.size else math.nan
    return (starts + length) / fs, val, p95


def _unweighted_cumsum(x: np.ndarray) -> np.ndarray:
    """Cumulative sum of the per-sample power, mean over channels (the RMS meters' base)."""
    xa = _as2d(x)
    return _cumsum(np.mean(xa * xa, axis=1))


def _rms_series_c(c: np.ndarray, fs: float, window_s: float, hop_s: float, aes17: bool = False):
    """`rms_series` from the unweighted power's cumulative sum."""
    starts, length, p = _windows_from_cumsum(c, fs, window_s, hop_s)
    with np.errstate(divide="ignore"):
        db = 10.0 * np.log10(p)
    if aes17:
        db = db + AES17_OFFSET_DB
    return (starts + length) / fs, db


def rms_series(x: np.ndarray, fs: float, window_s: float = RMS_WINDOW_S, hop_s: float = HOP_S,
               aes17: bool = False):
    """Sliding RMS, dBFS (module contract). Returns (t_end_s, dbfs)."""
    return _rms_series_c(_unweighted_cumsum(x), fs, window_s, hop_s, aes17)


def _rms_1s_db(db: np.ndarray, aes17: bool) -> dict:
    off = AES17_OFFSET_DB if aes17 else 0.0
    gated = db[db > RMS_GATE_DBFS]
    if gated.size == 0:
        return {"loud_section_dbfs": math.nan, "max_dbfs": math.nan, "aes17": bool(aes17)}
    return {"loud_section_dbfs": float(np.percentile(gated, 95.0)) + off,
            "max_dbfs": float(gated.max()) + off, "aes17": bool(aes17)}


def rms_1s(x: np.ndarray, fs: float, aes17: bool = False) -> dict:
    """1 s RMS over the loudest section (p95 over gated windows) and the loudest 1 s."""
    _, db = rms_series(x, fs)
    return _rms_1s_db(db, aes17)


def crest_db(x: np.ndarray, fs: float) -> float:
    """Sample peak minus the loudest-section 1 s RMS, dB (module contract)."""
    return sample_peak(x) - rms_1s(x, fs)["loud_section_dbfs"]


def rms_meter(x: np.ndarray, fs: float) -> dict:
    """The maintainer's meter (Ozone 11 RMS bar, 300 ms integration, AES17: a full-scale sine
    reads 0 dB; calibrated 2026-10-19 with a -6 dBFS sine reading -6 and 200 ms bursts
    reading -7.4): a METER_WINDOW_S sliding RMS at a METER_HOP_S hop, AES17. Returns
    {"loud_dbfs": the p95 over windows above RMS_GATE_DBFS (plain), the bar's top in the
    loudest passage as the maintainer reads it (his masters: -5.2 and -5.4 where he read
    "about -5"), "max_dbfs": the series maximum}. NaN when nothing passes the gate."""
    _, db = rms_series(x, fs, METER_WINDOW_S, METER_HOP_S)
    return _rms_meter_db(db)


def _rms_meter_db(db: np.ndarray) -> dict:
    gated = db[db > RMS_GATE_DBFS]
    if gated.size == 0:
        return {"loud_dbfs": math.nan, "max_dbfs": math.nan}
    return {"loud_dbfs": float(np.percentile(gated, 95.0)) + AES17_OFFSET_DB,
            "max_dbfs": float(gated.max()) + AES17_OFFSET_DB}


def loud_passage_lufs(x: np.ndarray, fs: float, weights=None) -> dict:
    """Loud-passage loudness: S95 (control) and loud-section integrated (report)."""
    ps = _kpower(x, fs, weights)
    c = _cumsum(ps)
    return _loud_passage_from(ps.shape[0], fs, _windows_from_cumsum(c, fs, BLOCK_S, HOP_S),
                              _windows_from_cumsum(c, fs, SHORT_S, HOP_S))


def _loud_passage_from(n: int, fs: float, blocks, windows) -> dict:
    """`loud_passage_lufs` from the 400 ms block powers and the 3 s window powers (both at
    the 100 ms hop) of an n-sample signal; the blocks stand in for the windows when the
    input is shorter than 3 s."""
    nan = {"s95": math.nan, "loud_section": math.nan, "s_max": math.nan,
           "loud_fraction": 0.0, "window_s": math.nan}
    win_s = SHORT_S if n >= round(SHORT_S * fs) else BLOCK_S
    w_starts, w_len, w_p = windows if win_s == SHORT_S else blocks
    if w_starts.size == 0:
        return nan
    s = _lufs(w_p)
    gated = s > ABS_GATE
    if not np.any(gated):
        return nan
    s95 = float(np.percentile(s[gated], 95.0))
    s_max = float(s.max())
    sel = s >= s_max - 3.0
    # union of selected windows as a sample mask, via a difference array
    d = np.zeros(n + 1, dtype=np.int64)
    np.add.at(d, w_starts[sel], 1)
    np.add.at(d, w_starts[sel] + w_len, -1)
    mask = np.cumsum(d[:-1]) > 0
    b_starts, b_len, b_p = blocks
    cm = np.concatenate([[0], np.cumsum(mask)])
    inside = (cm[b_starts + b_len] - cm[b_starts]) == b_len
    loud = float(_lufs(_gated_mean_power(b_p[inside])))
    return {"s95": s95, "loud_section": loud, "s_max": s_max,
            "loud_fraction": float(mask.mean()), "window_s": win_s}


def measure(x: np.ndarray, fs: float, oversample: int = TP_OVERSAMPLE, weights=None) -> dict:
    """Summary metrics for reports. true_peak_dbtp is at `oversample` (8x by
    default); true_peak_4x_dbtp is the BS.1770 Annex 2 reading for compliance.
    The shared intermediates (module contract) make every value equal the
    stand-alone function's."""
    x = _as2d(x)
    ps = _kpower(x, fs, weights)
    c = _cumsum(ps)
    blocks = _windows_from_cumsum(c, fs, BLOCK_S, HOP_S)
    windows = _windows_from_cumsum(c, fs, SHORT_S, HOP_S)
    env = true_peak_envelope(x, fs, oversample)
    tp = _env_db(env)
    i = _integrated_blocks(blocks[2])
    lp = _loud_passage_from(ps.shape[0], fs, blocks, windows)
    _, _, psr95 = _psr_from(*windows, env, fs)
    cu = _unweighted_cumsum(x)
    r = _rms_1s_db(_rms_series_c(cu, fs, RMS_WINDOW_S, HOP_S)[1], False)
    rm = _rms_meter_db(_rms_series_c(cu, fs, METER_WINDOW_S, METER_HOP_S)[1])
    sp = sample_peak(x)
    return {
        "rms_meter_dbfs": rm["loud_dbfs"],
        "rms_meter_max_dbfs": rm["max_dbfs"],
        "integrated_lufs": i,
        "s95_lufs": lp["s95"],
        "loud_section_lufs": lp["loud_section"],
        "loud_window_s": lp["window_s"],
        "short_term_max_lufs": _sliding_max_lufs(ps, fs, SHORT_S, c),
        "momentary_max_lufs": _sliding_max_lufs(ps, fs, BLOCK_S, c),
        "lra_lu": _lra_from_short_term(_lufs(windows[2])),
        "true_peak_dbtp": tp,
        "true_peak_oversample": oversample,
        "true_peak_4x_dbtp": true_peak(x, fs, 4),
        "sample_peak_dbfs": sp,
        "plr_db": tp - i,
        "psr_p95_db": psr95,
        "rms_1s_loud_dbfs": r["loud_section_dbfs"],
        "rms_1s_max_dbfs": r["max_dbfs"],
        "crest_db": sp - r["loud_section_dbfs"],
    }
