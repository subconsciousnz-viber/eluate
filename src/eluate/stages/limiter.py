"""Final loudness stage: onset-masked soft clipper, band-limit, 8x true-peak
limiter, local re-limit (spec 3.5 stage 7, audit 02 A11 / D.10, audit 05 A1-A4).

Contract: `apply(x, fs, plan) -> y` (and `apply_with_info` -> (y, info)), pure,
(n, ch) in, float64 (n, ch) out. The plan holds scalars only; every envelope
is computed here.
1. Input gain: x * 10^(input_gain_db / 20) (the solver's loudness knob).
2. u = resample.upsample(x, 8): the processing resampler (stages/resample.py,
   passband 0.98 x Nyquist, 60 dB at Nyquist), not the metering prototype,
   so the stage is transparent below the ceiling (audit 05 Q1).
3a. Whole-signal soft clipper (`clip_whole`, docs/gain-staging.md 8, the planner's
   default since chunk 3), at 8x, when clip_threshold_db is set. T =
   10^((clip_threshold_db + input_gain_db) / 20) (the threshold is set on the
   signal before the input gain, so the clipper's reduction does not move with
   the drive), knee half-width w = T (1 - 10^(-clip_knee_db / 20)): the same
   curve as below, identity up to T - w (clip_knee_db below T), quadratic up to
   T + w, T above (w = 0: hard clip at T). With plan.clip_max_gr_db set, no sample
   loses more than that: the output is max(curve(|x|), |x| 10^(-cap/20)), so what
   passes the cap goes on to the limiter. No onset mask: every run of 8x samples
   whose max over channels exceeds T - w is clipped; runs closer than the
   band-limit's reach (2 BL_MARGIN base samples) are merged into one span, and
   outside every span the 8x signal is untouched bit for bit. With
   plan.clip_split_hz set (the planner's default since 2026-10-20: the lowest
   multiband crossover), the clipper acts on the part of the signal above that
   frequency only (`clip_hi_part`: the high part of a zero-phase second-order
   split at base rate, then upsampled): spans are found on it, the curve is
   applied to it and the change u_hi_clipped - u_hi is added to the full signal,
   so the sub passes untouched to the limiter and the reported reduction is the
   high part's. The reported
   reduction is the curve's at each span's peak (concave curve: the largest
   reduction is at the peak). `clip_threshold_for_peak_gr` solves T for a peak
   reduction (1 dB aimed, CLIP_MAX_GR_DB = 1.5 at most) on the track's 8x peak.
   The clipper is also a tonal lever: the harmonics it adds are what chunk 4
   will use (maintainer, docs/gain-staging.md 8); here it is set for peak reduction
   only.
3b. Onset-masked soft clipper (`clip_onsets`, the "kick detective" toggle), at
   8x, when clip_depth_db > 0.
   Each onset o gives a core window [o - clip_pre_ms, o + clip_post_ms] (8 to
   12 ms); cores whose ramped spans (clip_ramp_ms either side) touch are merged
   into one, so every sample is clipped at most once. For each merged span,
   P = max |u| over the whole active span (ramps included) and channels, C =
   ceiling. If P > C: r = min(clip_depth_db, 20 log10(P / C)), T = P 10^(-r/20)
   (so T >= C), knee floor F = C 10^(-k/20) with k = KNEE_BELOW_DB x
   (clip_depth_db / 3 dB) (spec 3.5 item 7, audit 05 D.20), knee half-width
   w = min(P - T, T - F); the curve is the identity up to T - w (never below
   F, i.e. at most k dB below the ceiling), a quadratic knee
   a - (a - T + w)^2 / (4 w) up to T + w, and T above. Small overs therefore
   get a soft knee rather than a hard clip at the ceiling.
   The curve is concave with f(P) = T, so f(a) / a >= T / P for every
   a <= P: no sample is reduced by more than r <= clip_depth_db. The clipped
   span is cross-faded in with raised-cosine ramps; outside every span the 8x
   signal is untouched bit for bit.
4. Band-limit (audit 05 A2): the clipper's change u_c - u is decimated and
   re-interpolated (resample.decimate then resample.upsample, locally per span) so
   its content above base-rate Nyquist is removed before the detector sees it;
   otherwise those harmonics vanish in the final decimation and the
   fundamental's crest rebounds unseen.
5. Limiter at 8x: detector = max over channels of |u_bl| (stereo-linked),
   g = envelope.box_smooth (or qp_smooth) of required_gain(detector, C), attack
   attack_ms, release release_ms at the 8x rate. y = resample.decimate(u_bl g, 8).
6. Local re-limit on the decimated output (at most RELIMIT_PASSES passes): the
   per-sample peak envelope E = max(8x true-peak envelope, 32x FFT envelope, 4x
   BS.1770 Annex 2 envelope) of y; where E exceeds C, g2 = box_smooth(required_gain(E, C - 0.005 dB))
   at base rate with the same attack / release, y *= g2. The gain is 1 away
   from the excess, so the rest of the track is untouched (to the rounding of
   the FFT-convolution resampler, below 1e-12).
7. Last resort: if max(8x, 32x) true peak still exceeds C, one global trim
   onto C, reported in info["trim_db"]; never seen in the tests.
info["detector_peak_dbfs"] is the peak of the limiter's 8x detector (after the
input gain, the clipper and the band-limit): the limiter's own peak GR is
max(0, that - ceiling_dbtp), exactly, since box_smooth keeps the minimum of the
required gain (envelope.py).
Memory: whole-track 8x arrays (about 2.6 GB of allocations per minute of
stereo); the blocked version is workstream E's (spec 1).
"""

from __future__ import annotations

import math
import threading

import numpy as np
from scipy import signal

from eluate import bands, envelope, meter, parallel
from eluate.stages import resample
from eluate.plan import CLIP_MAX_GR_DB, LimiterPlan

OVERSAMPLE = 8
CHECK_OVERSAMPLE = 32
FFT_BLOCK = 32768
FFT_MARGIN = 8192
FFT_EDGE_TAIL = 256     # base samples of silence read either side of the track
BL_MARGIN = 512         # base samples of context around a clipped span (2 x 182-tap half-length)
RELIMIT_PASSES = 4
RELIMIT_MARGIN_DB = 0.005
ATTACK_FLOOR = 2          # samples: box_smooth with attack < 2 does no smoothing
KNEE_BELOW_DB = 1.0       # knee may start this far below the ceiling at 3 dB depth
CLIP_TARGET_DB = 1.0      # whole-signal clipper: peak reduction aimed at (gain-staging 3)
PEAK_BLOCK = 1 << 17      # base samples per block of the blocked 8x peak reading
SOLVE_TOL_DB = 1e-4
SOLVE_ITER = 80


def true_peak_fft_envelope(y: np.ndarray, up: int = CHECK_OVERSAMPLE, block: int = FFT_BLOCK,
                           margin: int = FFT_MARGIN, tail: int = FFT_EDGE_TAIL) -> np.ndarray:
    """Per-sample max |x(t)| over t in [i, i+1) and channels, by FFT zero-padding interpolation.

    The signal is embedded in silence; each core block of `block` samples is
    interpolated with `margin` samples of context either side (scipy.signal.resample)
    and only the core is read. The `tail` samples of silence before the track and
    after it are read too (reconstruction ringing into the silence counts, as in
    meter.true_peak, audit 05 A4) and folded into the first / last sample. The blocks
    run on parallel.pmap; their maxima fold into the envelope under a lock (max is exact,
    so the result does not depend on the order).
    """
    y = np.asarray(y, dtype=np.float64)
    y = y[:, None] if y.ndim == 1 else y
    n = y.shape[0]
    env = np.zeros(n)
    if n == 0:
        return env
    yp = np.pad(y, ((margin, margin + block), (0, 0)))
    lock = threading.Lock()

    def one(start):
        blen = min(block, n - start)
        lo = margin - (tail if start == 0 else 0)
        hi = margin + blen + (tail if start + blen == n else 0)
        seg = yp[start:start + block + 2 * margin]
        z = np.max(np.abs(signal.resample(seg, seg.shape[0] * up, axis=0)), axis=1)
        a = z[lo * up:hi * up].reshape(-1, up).max(axis=1)   # one value per base sample
        del z
        idx = np.clip(np.arange(start - (margin - lo), start - (margin - lo) + a.size), 0, n - 1)
        with lock:
            np.maximum.at(env, idx, a)

    parallel.pmap(one, range(0, n, block), parallel.BLOCK_CAP)
    return env


def true_peak_fft(y: np.ndarray, up: int = CHECK_OVERSAMPLE, block: int = FFT_BLOCK,
                  margin: int = FFT_MARGIN) -> float:
    """True peak (dBTP, max over channels) by FFT zero-padding interpolation."""
    env = true_peak_fft_envelope(y, up, block, margin)
    if env.size == 0 or env.max() == 0.0:
        return -math.inf
    return float(20.0 * math.log10(env.max()))


def _knee_clip(v: np.ndarray, t: float, w: float, r: float = 0.0) -> np.ndarray:
    """Soft-knee clip at t with half-width w; r > 0 caps the reduction per sample
    (output at least r |v|, r = 10^(-cap/20))."""
    a = np.abs(v)
    if w <= 0.0:
        out = np.minimum(a, t)
    else:
        out = np.where(a <= t - w, a, np.where(a >= t + w, t, a - (a - t + w) ** 2 / (4.0 * w)))
    if r > 0.0:
        out = np.maximum(out, r * a)
    return np.sign(v) * out


def _spans(plan: LimiterPlan, fs_up: float, n: int):
    """Merged clip spans as (core0, core1, active0, active1) in oversampled samples."""
    pre = int(round(plan.clip_pre_ms * 1e-3 * fs_up))
    post = int(round(plan.clip_post_ms * 1e-3 * fs_up))
    ramp = int(round(plan.clip_ramp_ms * 1e-3 * fs_up))
    cores = []
    for o in sorted(plan.clip_onsets_s):
        c = int(round(o * fs_up))
        w0, w1 = max(0, c - pre), min(n, c + post + 1)
        if w1 > w0:
            cores.append([w0, w1])
    merged = []
    for w0, w1 in cores:
        if merged and w0 - ramp <= merged[-1][1] + ramp:
            merged[-1][1] = max(merged[-1][1], w1)
        else:
            merged.append([w0, w1])
    return [(w0, w1, max(0, w0 - ramp), min(n, w1 + ramp)) for w0, w1 in merged], ramp


def clip_onsets(u: np.ndarray, fs_up: float, plan: LimiterPlan, ceiling: float):
    """Onset-masked soft clipper on the oversampled signal u (n, ch).

    Returns (u_clipped, max_reduction_db, spans) with spans the (active0, active1)
    slices that were modified."""
    if plan.clip_depth_db <= 0.0 or not plan.clip_onsets_s:
        return u, 0.0, []
    out = u.copy()
    spans, ramp = _spans(plan, fs_up, u.shape[0])
    max_red = 0.0
    touched = []
    for w0, w1, a0, a1 in spans:
        seg = u[a0:a1]
        p = float(np.max(np.abs(seg)))
        if p <= ceiling:
            continue
        r_db = min(plan.clip_depth_db, 20.0 * math.log10(p / ceiling))
        t = max(ceiling, p * 10.0 ** (-r_db / 20.0))
        floor = ceiling * 10.0 ** (-KNEE_BELOW_DB * (plan.clip_depth_db / 3.0) / 20.0)
        kw = max(0.0, min(p - t, t - floor))
        idx = np.arange(a0, a1)
        wgt = np.ones(a1 - a0)
        if ramp > 0:
            dist = np.maximum((w0 - idx) / (ramp + 1.0), (idx - (w1 - 1)) / (ramp + 1.0))
            wgt = np.where(dist <= 0, 1.0, 0.5 * (1.0 + np.cos(np.pi * np.minimum(dist, 1.0))))
        out[a0:a1] = seg + wgt[:, None] * (_knee_clip(seg, t, kw) - seg)
        max_red = max(max_red, r_db)
        touched.append((a0, a1))
    return out, max_red, touched


def clip_curve(t: float, knee_db: float) -> tuple[float, float]:
    """(T, w) of the whole-signal curve: the knee starts knee_db below T."""
    return t, t * (1.0 - 10.0 ** (-knee_db / 20.0))


def clip_reduction_db(peak: float, t: float, w: float, r: float = 0.0) -> float:
    """Reduction (dB >= 0) of the soft-knee curve at an input level peak."""
    if peak <= 0.0:
        return 0.0
    f = float(_knee_clip(np.array([peak]), t, w, r)[0])
    return float(20.0 * math.log10(peak / f)) if f > 0 else math.inf


def whole_runs(a: np.ndarray, floor: float, offset: int = 0) -> list[list[int]]:
    """[start, end) runs (offset added) of a > floor; a = per-sample max over channels."""
    over = np.concatenate([[False], a > floor, [False]])
    d = np.flatnonzero(np.diff(over.astype(np.int8)))
    return [[int(i) + offset, int(j) + offset] for i, j in zip(d[0::2], d[1::2])]


def merge_runs(runs, gap: int) -> list[tuple[int, int]]:
    """Runs merged where the gap between them is at most gap samples."""
    out: list[list[int]] = []
    for a0, a1 in sorted(runs):
        if out and a0 - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], a1)
        else:
            out.append([a0, a1])
    return [(a0, a1) for a0, a1 in out]


SPAN_GAP = 2 * BL_MARGIN * OVERSAMPLE   # 8x samples: runs this close share one band-limit


def sample_cap_ratio(plan: LimiterPlan) -> float:
    """r of _knee_clip for plan.clip_max_gr_db (0 = uncapped)."""
    c = getattr(plan, "clip_max_gr_db", None)
    return 0.0 if c is None else 10.0 ** (-c / 20.0)


def whole_threshold(plan: LimiterPlan) -> tuple[float, float, float]:
    """(T, w, r) on the 8x signal after the input gain (module contract 3a); r the
    per-sample reduction cap (sample_cap_ratio)."""
    t, w = clip_curve(10.0 ** ((plan.clip_threshold_db + plan.input_gain_db) / 20.0),
                      plan.clip_knee_db)
    return t, w, sample_cap_ratio(plan)


def clip_hi_part(x: np.ndarray, fs: float, fc_hz: float) -> np.ndarray:
    """The part of x (n, ch) above fc_hz: the high band of a zero-phase second-order
    split (bands.split, complementary, so x minus it is the low part exactly)."""
    return bands.split(np.asarray(x, dtype=np.float64), fs, [float(fc_hz)], phase="zero")[1]


def clip_whole(u: np.ndarray, plan: LimiterPlan, u_hi: np.ndarray | None = None):
    """Whole-signal soft clipper on the oversampled signal u (n, ch), input gain applied.
    u_hi (the oversampled part of u above plan.clip_split_hz, same shape): the clipper
    acts on it and its change is added to u (module contract 3a).

    Returns (u_clipped, max_reduction_db, spans) with spans the (a0, a1) slices that
    were modified."""
    if plan.clip_threshold_db is None:
        return u, 0.0, []
    t, w, r = whole_threshold(plan)
    src = u if u_hi is None else u_hi
    spans = merge_runs(whole_runs(np.max(np.abs(src), axis=1), t - w), SPAN_GAP)
    out = u.copy()
    max_red = 0.0
    for a0, a1 in spans:
        seg = src[a0:a1]
        c = _knee_clip(seg, t, w, r)
        out[a0:a1] = c if u_hi is None else u[a0:a1] + (c - seg)
        max_red = max(max_red, clip_reduction_db(float(np.max(np.abs(seg))), t, w, r))
    return out, max_red, spans


def peak_8x_db(x: np.ndarray, block: int = PEAK_BLOCK) -> float:
    """Peak (dBFS) of resample.upsample(x, 8), max over channels, read per block (each
    block upsampled with the filter's reach of context either side)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[:, None] if x.ndim == 1 else x
    n = x.shape[0]
    L = OVERSAMPLE
    hb = (resample.prototype(L).size - 1) // 2 // L + 2
    pk = 0.0
    for c0 in range(0, n, block):
        c1 = min(n, c0 + block)
        r0, r1 = max(0, c0 - hb), min(n, c1 + hb)
        u = resample.upsample(x[r0:r1], L)
        pk = max(pk, float(np.max(np.abs(u[L * (c0 - r0):L * (c1 - r0)]))))
    return float(20.0 * math.log10(pk)) if pk > 0 else -math.inf


def clip_threshold_for_peak_gr(x: np.ndarray, target_db: float = CLIP_TARGET_DB,
                               knee_db: float = 1.0, block: int = PEAK_BLOCK) -> dict:
    """Whole-signal clipper threshold (dBFS, before the limiter's input gain) giving a
    peak reduction of target_db at the 8x peak of x, the audio entering the clipper
    (module contract 3a), by bisection on the curve. Returns {"threshold_db",
    "peak_gr_db", "peak_8x_dbfs"}; a silent x gives a threshold of 0 dBFS."""
    if not 0.0 <= target_db <= CLIP_MAX_GR_DB:
        raise ValueError(f"target_db must lie in [0, {CLIP_MAX_GR_DB}]")
    p_db = peak_8x_db(x, block)
    if not math.isfinite(p_db):
        return {"threshold_db": 0.0, "peak_gr_db": 0.0, "peak_8x_dbfs": p_db}
    peak = 10.0 ** (p_db / 20.0)
    if target_db <= 0.0:
        return {"threshold_db": p_db + 0.01, "peak_gr_db": 0.0, "peak_8x_dbfs": p_db}

    def red(t_db):
        return clip_reduction_db(peak, *clip_curve(10.0 ** (t_db / 20.0), knee_db))

    lo, hi = p_db - target_db - knee_db - 1.0, p_db + 0.01
    best = (hi, red(hi))
    for _ in range(SOLVE_ITER):
        if abs(best[1] - target_db) < SOLVE_TOL_DB:
            break
        mid = 0.5 * (lo + hi)
        r = red(mid)
        if abs(r - target_db) < abs(best[1] - target_db):
            best = (mid, r)
        if r > target_db:
            lo = mid
        else:
            hi = mid
    return {"threshold_db": float(best[0]), "peak_gr_db": float(best[1]), "peak_8x_dbfs": p_db}


def band_limit_change(u: np.ndarray, uc: np.ndarray, spans) -> np.ndarray:
    """u + LP(uc - u): the clipper's change band-limited to base-rate Nyquist,
    computed locally around each modified span on the 8x grid."""
    if not spans:
        return uc
    L = OVERSAMPLE
    n = u.shape[0]
    out = u.copy()
    m = BL_MARGIN * L
    for a0, a1 in spans:
        s0 = max(0, ((a0 - m) // L) * L)            # aligned to the base-rate grid
        s1 = min(n, a1 + m)
        s1 = s0 + ((s1 - s0) // L) * L
        d = np.zeros((s1 - s0, u.shape[1]))
        d[a0 - s0:a1 - s0] = uc[a0:a1] - u[a0:a1]   # this span's change only
        out[s0:s1] += resample.upsample(resample.decimate(d, L), L)
    return out


def _smooth(g_req, attack, release, smoother):
    if smoother == "box":
        return envelope.box_smooth(g_req, attack=attack, release=release), None
    return envelope.qp_smooth(g_req, attack=attack, release=release)


def _peak_envelopes(y, fs):
    """(8x true-peak envelope, 32x FFT envelope, 4x BS.1770 Annex 2 envelope), per base
    sample. The re-limit holds the maximum of the three, so the file's "worst" reading
    (solver.true_peaks) is at or below the ceiling by construction: the Annex 2 filter
    can read a few hundredths of a dB above 8x / 32x on limiter-shaped peaks."""
    return (meter.true_peak_envelope(y, fs, OVERSAMPLE), true_peak_fft_envelope(y),
            meter.true_peak_envelope(y, fs, 4))


def _db(v):
    return float(20.0 * np.log10(v)) if v > 0 else -math.inf


def apply_with_info(x: np.ndarray, fs: float, plan: LimiterPlan):
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    x2 = (x[:, None] if squeeze else x) * 10.0 ** (plan.input_gain_db / 20.0)
    ceiling = 10.0 ** (plan.ceiling_dbtp / 20.0)
    fs_up = fs * OVERSAMPLE
    u = resample.upsample(x2, OVERSAMPLE)
    if plan.clip_threshold_db is not None:
        u_hi = None
        if plan.clip_split_hz is not None:
            u_hi = resample.upsample(clip_hi_part(x2, fs, plan.clip_split_hz), OVERSAMPLE)
        uc, clip_db, spans = clip_whole(u, plan, u_hi)
        del u_hi
    else:
        uc, clip_db, spans = clip_onsets(u, fs_up, plan, ceiling)
    u = band_limit_change(u, uc, spans)
    del uc
    det = np.max(np.abs(u), axis=1)
    det_peak = _db(det.max(initial=0.0))
    g_req = envelope.required_gain(det, ceiling)
    attack = max(ATTACK_FLOOR, int(round(plan.attack_ms * 1e-3 * fs_up)))
    release = plan.release_ms * 1e-3 * fs_up
    g, qp_info = _smooth(g_req, attack, release, plan.smoother)
    y = resample.decimate(u * g[:, None], OVERSAMPLE)
    gr8 = float(-20.0 * np.log10(np.min(g))) if g.size else 0.0
    del u, g, g_req, det

    e8, e32, e4 = _peak_envelopes(y, fs)
    tp8_0, tp32_0 = _db(e8.max(initial=0.0)), _db(e32.max(initial=0.0))
    e = np.maximum(np.maximum(e8, e32), e4)
    passes = []
    target = ceiling * 10.0 ** (-RELIMIT_MARGIN_DB / 20.0)
    a_base = max(ATTACK_FLOOR, int(round(plan.attack_ms * 1e-3 * fs)))
    r_base = plan.release_ms * 1e-3 * fs
    for _ in range(RELIMIT_PASSES):
        if e.size == 0 or e.max() <= ceiling:
            break
        g2 = envelope.box_smooth(envelope.required_gain(e, target), attack=a_base,
                                 release=r_base)
        y = y * g2[:, None]
        passes.append({"samples": int(np.count_nonzero(g2 < 1.0)),
                       "max_reduction_db": float(-20.0 * np.log10(g2.min()))})
        e8, e32, e4 = _peak_envelopes(y, fs)
        e = np.maximum(np.maximum(e8, e32), e4)
    tp8, tp32 = _db(e8.max(initial=0.0)), _db(e32.max(initial=0.0))
    trim_db = 0.0
    worst = max(tp8, tp32, _db(e4.max(initial=0.0)))
    if worst > plan.ceiling_dbtp:
        trim_db = plan.ceiling_dbtp - worst
        y = y * 10.0 ** (trim_db / 20.0)
    info = {
        "clip_max_reduction_db": clip_db,
        "clip_spans": len(spans),
        "max_gain_reduction_db": gr8,
        "detector_peak_dbfs": det_peak,
        "true_peak_8x_before_relimit_dbtp": tp8_0,
        "true_peak_32x_before_relimit_dbtp": tp32_0,
        "relimit_passes": passes,
        "true_peak_8x_dbtp": tp8 + trim_db,
        "true_peak_32x_dbtp": tp32 + trim_db,
        "trim_db": trim_db,
        "qp": qp_info,
    }
    return (y[:, 0] if squeeze else y), info


def apply(x: np.ndarray, fs: float, plan: LimiterPlan) -> np.ndarray:
    return apply_with_info(x, fs, plan)[0]
