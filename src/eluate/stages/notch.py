"""Dynamic notching (spec 3.5 stage 1, audit 01 F14, audit 05 A5-A8).

Contract: `apply(x, fs, plan) -> y`, pure, (n, ch) in, float64 (n, ch) out.
- Parallel form: y = x - s(t) * sum_i d_i(t) BP_i(x),
      d_i(t) = (1 - m_kick(t)) * (1 - 10^(G_i(t) / 20)),  G_i clipped to [-6, 0] dB,
  so d_i <= 1 - 10^(-6/20) = 0.49881. With one notch active s = 1 and the gain
  at f0 is exactly 10^(G/20). G_i and m_kick are 1 ms envelopes interpolated by
  PCHIP (plan.Envelope.at).
- Composite clamp (audit 05 A6): s(t) <= 1 keeps the summed cut at every
  frequency within the cap, 1 - s sum_i d_i B_i(f) >= 10^(-6/20), where B_i is
  notch i's zero-phase bandpass magnitude. On each 1 ms control segment an
  upper bound of sum_i d_i(t) B_i(f) is evaluated from the larger of each
  notch's cut and of (1 - m_kick) at the two end points (PCHIP stays between
  them); its maximum over frequency is found on a 1/96-octave grid holding
  every f0, with the three largest local maxima refined by bounded scalar
  search. s at a control point is the minimum over its two segments, then
  eroded and averaged over S_SMOOTH_CTRL control points either side (a moving
  minimum followed by a moving average of the same width never exceeds the
  input), and is PCHIP-interpolated, so the bound holds between control points
  too. s = 1 exactly wherever fewer than two notches are active within
  2 S_SMOOTH_CTRL ms.
- BP_i: RBJ constant-0-dB-peak biquad bandpass run forward and backward
  (sosfiltfilt) from rest with zero padding (the track embedded in silence, as
  bands.split): zero phase, unity gain at f0, magnitude B = |H|^2.
- Q is defined on the notch response (audit 05 A5, cookbook convention): f0 /
  Q is the bandwidth at half the dB cut, taken at the notch's deepest planned
  cut G_ref = min G_i (clipped to [-6, 0]). The response 1 - d B reaches
  10^(G_ref/40) where B = 1 / (1 + 10^(G_ref/40)), i.e. where the analog
  prototype has Q_bq u = 10^(G_ref/80); the biquad Q is solved (`design_q`,
  Brent) so the digital width after the bilinear warp is exactly f0 / Q.
- Identity: where every d_i(t) == 0 the sample is untouched bit for bit, so
  depth 0 everywhere, an empty list, or a full kick mask is a bit-exact
  identity.
- Kick mask: m_kick in [0, 1] from `kick_mask`: 1 within +-half_ms (default
  20 ms) of an onset at audio rate (the hold is padded by one control period
  on the 1 ms grid, audit 05 A8), raised-cosine ramp of ramp_ms (default 5 ms)
  outside that.
- Merge (audit 05 A6): candidates whose half-cut bands overlap must be merged by
  the planner with `merge_candidates(notches, fs)`, within each channel. The bands
  are the realised ones (`notch_edges`, 2026-11-10, filter-maths audit F2): the
  digital notch's half-cut edges after the bilinear warp, fs / pi atan(tan(pi f0 /
  fs) o^(+-1)) with o from the solved biquad Q (`_notch_bandwidth`), so hi - lo =
  f0 / Q exactly and tan(pi lo / fs) tan(pi hi / fs) = tan(pi f0 / fs)^2. The
  analog geometric edges f0 (sqrt(1 + 1/(4Q^2)) +- 1/(2Q)) (`band_edges`, used
  before) are right only far below fs / 4: at 44.1 kHz and Q 2 they are 5 % high at
  12 kHz and 17 % at 18 kHz, so 12000 / 19695 Hz Q 2 (realised bands 8912-14912 and
  11756-21603 Hz) passed as disjoint. A merged notch spans the union of its
  members' realised bands: f0 the tan-domain geometric mean of the edges (always
  below fs / 2), Q = f0 / (hi - lo). `apply` raises ValueError on a pair of the same
  channel whose bands overlap under both the realised and the analog edges, so plans
  merged on either rule (saved plans used the analog one) render as before; the
  composite clamp holds the summed cut either way. Q >= 1 (plan.NOTCH_Q_MIN).
- Padding of the zero-phase bandpass: PAD_TAU time constants of the slowest pole,
  -1 / ln max|pole| (np.roots of the denominator; 2026-11-10, audit F3). For complex
  poles this is atanh(alpha), the formula used before (the same pads on 4203 grid
  cases); at high f0 and low Q the poles are real (alpha >= 1 from f0 / fs 0.225 at
  Q 1, where math.atanh raised a domain error, or a1^2 > 4 a2 before that, where the
  old pad followed the geometric mean of the two radii and came out short).
- Channel (features audit 2026-11-10 F7): a Notch with channel "both" is applied to
  every channel of x as above; one with channel "S" is applied to the side channel
  alone (L/R -> M/S with bands.to_ms, the parallel form on S, back with to_lr), after
  the "both" set, so a side-only resonance is cut where it rings and M is untouched.
  The composite clamp and the merge check run within each set; the planner keeps an S
  notch out of a stereo notch's band so the two sets never stack on S past the cap.
  The M/S conversion happens only when a side notch cuts somewhere, so the identity
  properties above hold bit for bit.
- Composite clamp cost: the bound is evaluated per distinct row of cut depths after
  rounding them up to BOUND_STEPS steps of the cap (conservative; the steps this
  leaves in s are smoothed as above), so the planner's envelopes need no quantisation
  of their own (the 0.5 dB steps raised the notch's
  modulation sidebands from -64 to -47 dB; audio-rate audit 2026-11-10).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.optimize import brentq, minimize_scalar

from eluate import bands, iir
from eluate.plan import (CONTROL_RATE_HZ, NOTCH_MAX_CUT_DB, Envelope, Notch, NotchPlan,
                            control_coord, control_length)

D_MAX = 1.0 - 10.0 ** (-NOTCH_MAX_CUT_DB / 20.0)
GRID_PER_OCTAVE = 96
REFINE_PEAKS = 3
BOUND_STEPS = 12       # composite_scale rounds the cut depths up to this many steps of the cap
S_SMOOTH_CTRL = 10     # s is eroded then box-averaged over this many control points each side


def _g_ref(nt: Notch) -> float:
    return float(np.clip(nt.gain_db.values.min(), -NOTCH_MAX_CUT_DB, 0.0))


def _notch_bandwidth(q_bq: float, f0: float, fs: float, g_ref_db: float) -> float:
    """Width (Hz) of the notch response 1 - d |H|^2 at half its dB cut, for biquad Q q_bq.

    RBJ's bandpass is exactly the bilinear transform of the analog
    (s/Q)/(s^2 + s/Q + 1) with Omega = tan(w/2) / tan(w0/2); |H|^2 = 1 / (1 + Q^2 u^2),
    u = Omega - 1/Omega, so the half-cut point is at Q u = 10^(g_ref/80)."""
    k = math.tan(math.pi * f0 / fs)
    c = 10.0 ** (g_ref_db / 80.0) / q_bq
    o2 = 0.5 * (c + math.sqrt(c * c + 4.0))
    return fs / math.pi * (math.atan(o2 * k) - math.atan(k / o2))


def design_q(f0: float, q: float, fs: float, g_ref_db: float = -NOTCH_MAX_CUT_DB) -> float:
    """Biquad Q giving the notch a half-cut bandwidth of exactly f0 / q."""
    target = f0 / q
    lo, hi = 1e-4, 1e5
    if not _notch_bandwidth(lo, f0, fs, g_ref_db) > target > _notch_bandwidth(hi, f0, fs, g_ref_db):
        raise ValueError(f"notch bandwidth {target:.1f} Hz not realisable at f0 {f0} Hz")
    return brentq(lambda qb: _notch_bandwidth(qb, f0, fs, g_ref_db) - target, lo, hi,
                  xtol=1e-12, rtol=1e-12)


def bandpass_sos(f0: float, q_bq: float, fs: float) -> np.ndarray:
    """RBJ constant-0-dB-peak bandpass with biquad Q q_bq (already solved)."""
    if not 0 < f0 < fs / 2:
        raise ValueError(f"notch f0 {f0} Hz outside (0, fs/2)")
    w0 = 2.0 * math.pi * f0 / fs
    alpha = math.sin(w0) / (2.0 * q_bq)
    a0 = 1.0 + alpha
    return np.array([[alpha / a0, 0.0, -alpha / a0, 1.0, -2.0 * math.cos(w0) / a0,
                      (1.0 - alpha) / a0]])


def notch_sos(nt: Notch, fs: float) -> np.ndarray:
    return bandpass_sos(nt.f0_hz, design_q(nt.f0_hz, nt.q, fs, _g_ref(nt)), fs)


def pad_samples(sos: np.ndarray, fs: float) -> int:
    """Zero padding per side for the bandpass: bands.padlen at the equivalent corner of
    the slowest pole, decay per sample -ln max|pole| (module contract)."""
    r = float(np.max(np.abs(np.roots([1.0, float(sos[0, 4]), float(sos[0, 5])]))))
    if not 0.0 < r < 1.0:
        raise ValueError(f"notch bandpass pole radius {r} outside (0, 1)")
    return bands.padlen(None, fs, fs * -math.log(r) / (2.0 * math.pi))


def bandpass_zp(x: np.ndarray, sos: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase bandpass of x (n, ch), zero-padded edges."""
    return iir.sosfiltfilt_zero(sos, x, pad_samples(sos, fs))   # blocked, subnormal flush


def kick_mask(onsets_s, duration_s: float, half_ms: float = 20.0, ramp_ms: float = 5.0,
              rate_hz: float = CONTROL_RATE_HZ) -> Envelope:
    """m_kick at the control rate: 1 within +-(half_ms + one control period) of an onset,
    cosine ramp over ramp_ms beyond that, so the audio-rate hold is at least +-half_ms."""
    n = int(math.ceil(duration_s * rate_hz)) + 1
    t = np.arange(n) / rate_hz
    m = np.zeros(n)
    h, r = half_ms / 1000.0 + 1.0 / rate_hz, ramp_ms / 1000.0
    for o in np.asarray(onsets_s, dtype=np.float64):
        dist = np.abs(t - o) - h
        if r > 0:
            w = np.where(dist <= 0, 1.0,
                         np.where(dist < r, 0.5 * (1.0 + np.cos(np.pi * dist / r)), 0.0))
        else:
            w = (dist <= 0).astype(np.float64)
        np.maximum(m, w, out=m)
    return Envelope(m, rate_hz)


def band_edges(f0: float, q: float) -> tuple[float, float]:
    """Analog geometric half-cut band edges: lo hi = f0^2, hi - lo = f0 / q (the rule
    before 2026-11-10; right far below fs / 4, see notch_edges)."""
    a = math.sqrt(1.0 + 0.25 / (q * q))
    return f0 * (a - 0.5 / q), f0 * (a + 0.5 / q)


def notch_edges(nt: Notch, fs: float | None) -> tuple[float, float]:
    """The notch's realised half-cut band edges (Hz) at fs: the digital response's
    points at half its dB cut G_ref after the bilinear warp (module contract);
    hi - lo = f0 / Q exactly. fs None: the analog band_edges."""
    if fs is None:
        return band_edges(nt.f0_hz, nt.q)
    g = _g_ref(nt)
    q_bq = design_q(nt.f0_hz, nt.q, fs, g)
    k = math.tan(math.pi * nt.f0_hz / fs)
    c = 10.0 ** (g / 80.0) / q_bq
    o2 = 0.5 * (c + math.sqrt(c * c + 4.0))
    return fs / math.pi * math.atan(k / o2), fs / math.pi * math.atan(o2 * k)


def _centre(lo: float, hi: float, fs: float | None) -> float:
    """The f0 whose band has edges lo and hi: geometric mean, in the tan (warped) domain
    when fs is given (always below fs / 2)."""
    if fs is None:
        return math.sqrt(lo * hi)
    return fs / math.pi * math.atan(math.sqrt(math.tan(math.pi * lo / fs)
                                              * math.tan(math.pi * hi / fs)))


def overlap(a: Notch, b: Notch, fs: float | None) -> bool:
    """True when the two notches' half-cut bands (notch_edges at fs) overlap or touch."""
    (lo1, hi1), (lo2, hi2) = sorted([notch_edges(a, fs), notch_edges(b, fs)])
    return lo2 <= hi1


def merge_candidates(notches: list[Notch], fs: float | None = None) -> list[Notch]:
    """Merge notches whose half-cut bands overlap (chained, sorted by lower edge), on the
    realised edges at fs (notch_edges; fs None: the analog edges, the rule before
    2026-11-10), within each channel ("both" and "S" notches act on different signals
    and are never merged with each other; the planner keeps an S notch out of a stereo
    notch's band).

    A merged notch spans the union of the members' bands: f0 the geometric mean of the
    union's edges (in the warped domain at fs), Q = f0 / (hi - lo), floored at Q_MIN
    (which narrows the merged band below the union); its gain envelope is the
    member-wise minimum (deepest), on the longest member's grid."""
    if not notches:
        return []
    out = []
    for ch in sorted({n.channel for n in notches}):
        out += _merge_channel([n for n in notches if n.channel == ch], fs)
    return out


def _merge_channel(notches: list[Notch], fs: float | None) -> list[Notch]:
    edges = {id(n): notch_edges(n, fs) for n in notches}
    srt = sorted(notches, key=lambda n: edges[id(n)][0])
    groups = [[srt[0]]]
    hi_run = edges[id(srt[0])][1]
    for n in srt[1:]:
        lo, hi = edges[id(n)]
        if lo <= hi_run:
            groups[-1].append(n)
            hi_run = max(hi_run, hi)
        else:
            groups.append([n])
            hi_run = hi
    out = []
    for g in groups:
        if len(g) == 1:
            out.append(g[0])
            continue
        lo = min(edges[id(n)][0] for n in g)
        hi = max(edges[id(n)][1] for n in g)
        f0 = _centre(lo, hi, fs)
        rate = g[0].gain_db.rate_hz
        if any(n.gain_db.rate_hz != rate for n in g):
            raise ValueError("cannot merge envelopes at different control rates")
        size = max(n.gain_db.values.size for n in g)
        vals = np.full(size, np.inf)
        for n in g:
            v = n.gain_db.values
            vals = np.minimum(vals, np.concatenate([v, np.full(size - v.size, v[-1])]))
        q = max(f0 / (hi - lo), 1.0)
        out.append(Notch(f0, q, Envelope(vals, rate), g[0].channel))
    return out


def _check_merged(notches: list[Notch], fs: float) -> None:
    """Raise on a pair of one channel whose bands overlap under both the realised edges
    at fs and the analog ones (module contract: a plan merged on either rule is
    accepted; "both" and "S" notches are never merged with each other)."""
    for i, a in enumerate(notches):
        for b in notches[i + 1:]:
            if a.channel == b.channel and overlap(a, b, None) and overlap(a, b, fs):
                lo1, hi1 = notch_edges(a, fs)
                lo2, hi2 = notch_edges(b, fs)
                raise ValueError(f"notch bands [{lo1:.1f}, {hi1:.1f}] and [{lo2:.1f}, "
                                 f"{hi2:.1f}] Hz ({a.channel}) overlap; merge them with "
                                 "merge_candidates")


def _freq_grid(f0s, fs: float) -> np.ndarray:
    lo, hi = 10.0, 0.499 * fs
    k = int(math.ceil(GRID_PER_OCTAVE * math.log2(hi / lo)))
    return np.unique(np.concatenate([lo * 2.0 ** (np.arange(k + 1) / GRID_PER_OCTAVE),
                                     np.asarray(f0s, dtype=np.float64)]))


def composite_scale(depths_ctrl: np.ndarray, keep_ctrl, soss, f0s, fs: float) -> np.ndarray:
    """s at control points. depths_ctrl: (n_ctrl, notches) cut depths 1 - 10^(G/20) on the
    control grid; keep_ctrl: (n_ctrl,) 1 - m_kick on the grid, or None."""
    nc, k = depths_ctrl.shape
    s = np.ones(nc)
    if k < 2 or nc < 2:
        return s
    # per segment: larger end-point cut times larger end-point (1 - m)
    up = np.maximum(depths_ctrl[:-1], depths_ctrl[1:])
    if keep_ctrl is not None:
        up = up * np.maximum(keep_ctrl[:-1], keep_ctrl[1:])[:, None]
    multi = np.count_nonzero(up > 0, axis=1) >= 2
    if not np.any(multi):
        return s
    f = _freq_grid(f0s, fs)

    def resp(freqs):
        return np.stack([np.abs(signal.sosfreqz(so, worN=freqs, fs=fs)[1]) ** 2 for so in soss])

    B = resp(f)
    # the bound is evaluated once per distinct row: the depths are rounded UP to
    # BOUND_STEPS steps of the cap (a larger assumed cut keeps the bound valid), so a
    # smooth envelope (no longer quantised by the planner, 2026-11-10) gives a few dozen
    # rows, not one per millisecond (a 10 s excerpt with 8 notches did not finish)
    step = D_MAX / BOUND_STEPS
    up_q = np.minimum(np.ceil(up[multi] / step - 1e-9) * step, 1.0)
    rows, inv = np.unique(up_q, axis=0, return_inverse=True)
    peaks = np.empty(rows.shape[0])
    for r, w in enumerate(rows):
        tot = w @ B
        # refine the largest grid local maxima between their grid neighbours
        loc = np.flatnonzero((tot[1:-1] >= tot[:-2]) & (tot[1:-1] >= tot[2:])) + 1
        best = float(tot.max())
        for j in loc[np.argsort(tot[loc])[::-1][:REFINE_PEAKS]]:
            res = minimize_scalar(lambda v: -float(w @ resp([v])[:, 0]),
                                  bounds=(f[j - 1], f[j + 1]), method="bounded",
                                  options={"xatol": 1e-9 * f[j]})
            best = max(best, -float(res.fun))
        peaks[r] = best
    seg = np.ones(nc - 1)
    seg[multi] = np.minimum(1.0, D_MAX / np.maximum(peaks[inv.reshape(-1)], 1e-300))
    s[:-1] = np.minimum(s[:-1], seg)
    s[1:] = np.minimum(s[1:], seg)
    # the rounded rows make s a staircase where the clamp binds (0.66 dB in one 1 ms step
    # on two overlapping ramps): a moving minimum then a moving average of the same width
    # smooths it and stays at or under the bound at every point
    w = 2 * S_SMOOTH_CTRL + 1
    if S_SMOOTH_CTRL > 0 and np.any(s < 1.0):
        s = np.minimum(uniform_filter1d(minimum_filter1d(s, w, mode="nearest"), w,
                                        mode="nearest"), s)
    return s


def apply(x: np.ndarray, fs: float, plan: NotchPlan) -> np.ndarray:
    y = np.array(x, dtype=np.float64, copy=True)
    if y.ndim == 1:
        y = y[:, None]
    n = y.shape[0]
    _check_merged(plan.notches, fs)
    if not plan.notches:
        return y if np.ndim(x) == 2 else y[:, 0]
    keep = None if plan.kick_mask is None else 1.0 - np.clip(plan.kick_mask.at(n, fs), 0.0, 1.0)
    for e in [nt.gain_db for nt in plan.notches] + ([plan.kick_mask] if keep is not None else []):
        if e.rate_hz != CONTROL_RATE_HZ:
            raise ValueError(f"notch envelopes must be at {CONTROL_RATE_HZ} Hz")
    both = [nt for nt in plan.notches if nt.channel != "S"]
    side = [nt for nt in plan.notches if nt.channel == "S"]
    if both:
        y = _apply_set(y, fs, both, plan.kick_mask, keep)
    if side and y.shape[1] == 2:
        # side-only notches cut S alone (features audit F7): L/R -> M/S, the cut on S, back.
        # Converted only when a side notch cuts somewhere, so a zero-depth plan stays a
        # bit-exact identity.
        if any(np.any(np.clip(nt.gain_db.values, -NOTCH_MAX_CUT_DB, 0.0) < 0.0) for nt in side):
            ms = bands.to_ms(y)
            ms[:, 1] = _apply_set(ms[:, 1:2], fs, side, plan.kick_mask, keep)[:, 0]
            y = bands.to_lr(ms)
    return y if np.ndim(x) == 2 else y[:, 0]


def _apply_set(y: np.ndarray, fs: float, notches: list[Notch], kick_mask, keep) -> np.ndarray:
    """The parallel notch set on y (n, ch), every notch on every channel of y."""
    n = y.shape[0]
    nc = control_length(n, fs)
    keep_ctrl = None if keep is None else 1.0 - np.clip(_ctrl_values(kick_mask, nc), 0, 1)
    acc = np.zeros_like(y)
    soss, d_ctrl, f0s = [], [], []
    for nt in notches:
        g_db = np.clip(nt.gain_db.at(n, fs), -NOTCH_MAX_CUT_DB, 0.0)
        d = 1.0 - 10.0 ** (g_db / 20.0)
        if keep is not None:
            d = keep * d
        if not np.any(d):
            continue
        sos = notch_sos(nt, fs)
        acc += d[:, None] * bandpass_zp(y, sos, fs)
        soss.append(sos)
        f0s.append(nt.f0_hz)
        # cut on the control grid; PCHIP keeps G and m between their end values,
        # so end-point maxima bound d on each segment
        dc = 1.0 - 10.0 ** (np.clip(_ctrl_values(nt.gain_db, nc), -NOTCH_MAX_CUT_DB, 0.0) / 20.0)
        d_ctrl.append(dc)
    if not soss:
        return y
    s = None
    if len(soss) >= 2:
        s_ctrl = composite_scale(np.stack(d_ctrl, axis=1), keep_ctrl, soss, f0s, fs)
        if np.any(s_ctrl < 1.0):
            t = np.clip(control_coord(np.arange(n), fs), 0.0, nc - 1.0)
            s = PchipInterpolator(np.arange(nc, dtype=np.float64), s_ctrl)(t)
    return y - (acc if s is None else s[:, None] * acc)


def _ctrl_values(e: Envelope, nc: int) -> np.ndarray:
    """Envelope values on the 1 ms control grid, end value held."""
    v = e.values
    if v.size >= nc:
        return v[:nc]
    return np.concatenate([v, np.full(nc - v.size, v[-1])])
