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
  search. s at a control point is the minimum over its two segments and is
  PCHIP-interpolated, so the bound holds between control points too. s = 1
  exactly wherever fewer than two notches are active.
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
- Merge (audit 05 A6): candidates whose -3 dB bands (half-cut bands, geometric
  edges f0 (sqrt(1 + 1/(4Q^2)) +- 1/(2Q))) overlap must be merged by the
  planner with `merge_candidates`; `apply` raises ValueError on an overlapping
  pair of the same channel. Q >= 1 (plan.NOTCH_Q_MIN).
- Channel (features audit 2026-11-10 F7): a Notch with channel "both" is applied to
  every channel of x as above; one with channel "S" is applied to the side channel
  alone (L/R -> M/S with bands.to_ms, the parallel form on S, back with to_lr), after
  the "both" set, so a side-only resonance is cut where it rings and M is untouched.
  The composite clamp and the merge check run within each set; the planner keeps an S
  notch out of a stereo notch's band so the two sets never stack on S past the cap.
  The M/S conversion happens only when a side notch cuts somewhere, so the identity
  properties above hold bit for bit.
- Composite clamp cost: the bound is evaluated per distinct row of cut depths after
  rounding them up to BOUND_STEPS steps of the cap (conservative), so the planner's
  envelopes need no quantisation of their own (the 0.5 dB steps raised the notch's
  modulation sidebands from -64 to -47 dB; audio-rate audit 2026-11-10).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, minimize_scalar

from eluate import bands
from eluate.plan import (CONTROL_RATE_HZ, NOTCH_MAX_CUT_DB, Envelope, Notch, NotchPlan,
                            control_coord, control_length)

D_MAX = 1.0 - 10.0 ** (-NOTCH_MAX_CUT_DB / 20.0)
GRID_PER_OCTAVE = 96
REFINE_PEAKS = 3
BOUND_STEPS = 12       # composite_scale rounds the cut depths up to this many steps of the cap


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


def bandpass_zp(x: np.ndarray, sos: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase bandpass of x (n, ch), zero-padded edges."""
    # pole radius r = sqrt((1 - alpha) / (1 + alpha)) from a2 = (1 - alpha) / (1 + alpha);
    # decay per sample -ln r = atanh(alpha), i.e. an equivalent corner fs atanh(alpha) / (2 pi)
    a2 = float(sos[0, 5])
    alpha = (1.0 - a2) / (1.0 + a2)
    pl = bands.padlen(None, fs, fs * math.atanh(alpha) / (2.0 * math.pi))
    xp = np.pad(x, ((pl, pl), (0, 0)))
    return signal.sosfiltfilt(sos, xp, axis=0, padtype=None)[pl:pl + x.shape[0]]


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
    """Exact geometric half-cut band edges: lo hi = f0^2, hi - lo = f0 / q."""
    a = math.sqrt(1.0 + 0.25 / (q * q))
    return f0 * (a - 0.5 / q), f0 * (a + 0.5 / q)


def merge_candidates(notches: list[Notch]) -> list[Notch]:
    """Merge notches whose half-cut bands overlap (chained, sorted by lower edge), within
    each channel ("both" and "S" notches act on different signals and are never merged
    with each other; the planner keeps an S notch out of a stereo notch's band).

    A merged notch spans the union of the members' bands: f0 = sqrt(lo hi),
    Q = f0 / (hi - lo), floored at Q_MIN by widening; its gain envelope is the
    member-wise minimum (deepest), on the longest member's grid."""
    if not notches:
        return []
    out = []
    for ch in sorted({n.channel for n in notches}):
        out += _merge_channel([n for n in notches if n.channel == ch])
    return out


def _merge_channel(notches: list[Notch]) -> list[Notch]:
    srt = sorted(notches, key=lambda n: band_edges(n.f0_hz, n.q)[0])
    groups = [[srt[0]]]
    hi_run = band_edges(srt[0].f0_hz, srt[0].q)[1]
    for n in srt[1:]:
        lo, hi = band_edges(n.f0_hz, n.q)
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
        lo = min(band_edges(n.f0_hz, n.q)[0] for n in g)
        hi = max(band_edges(n.f0_hz, n.q)[1] for n in g)
        f0 = math.sqrt(lo * hi)
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


def _check_merged(notches: list[Notch]) -> None:
    for ch in {n.channel for n in notches}:
        e = sorted(band_edges(n.f0_hz, n.q) for n in notches if n.channel == ch)
        for (lo1, hi1), (lo2, hi2) in zip(e, e[1:]):
            if lo2 <= hi1:
                raise ValueError(f"notch bands [{lo1:.1f}, {hi1:.1f}] and [{lo2:.1f}, {hi2:.1f}] "
                                 f"Hz ({ch}) overlap; merge them with merge_candidates")


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
    return s


def apply(x: np.ndarray, fs: float, plan: NotchPlan) -> np.ndarray:
    y = np.array(x, dtype=np.float64, copy=True)
    if y.ndim == 1:
        y = y[:, None]
    n = y.shape[0]
    _check_merged(plan.notches)
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
