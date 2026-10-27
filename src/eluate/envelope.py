"""Gain-envelope construction (spec 3.4, audit 01 F7, audit 02).

All functions take and return 1-D float64 arrays of gain at one control rate
(1 ms for multiband / notch / tone, the 8x audio rate for the limiter). NaN or
Inf input raises ValueError.

Contract:
- `required_gain(env, threshold)` = min(1, threshold / env), 1 where env <= threshold.
- Both smoothers bound against u = clip(g_req, g_min, 1), g_min = 1e-6 by
  default, so g is in [g_min, 1] and g <= u everywhere.
- `release_hull(u, release)`: h_dB[n] = min(u_dB[n], a * h_dB[n-1]),
  a = exp(-1/release): gain reduction recovers exponentially in dB with time
  constant `release` samples. h <= u. Closed form (log-domain cumulative max),
  restarted every block so precision does not degrade with track length.
- `box_smooth(g_req, attack, release)`:
    1. u as above; h = release_hull(u, release).
    2. Lookahead running minimum over offsets [-(L-1), L-1] (window 2L-1,
       scipy.ndimage.minimum_filter1d), L = attack // 2 + 1.
    3. Moving average over offsets [-(L-1), 0] then over [0, L-1]: a symmetric
       triangular kernel with support [-(L-1), L-1], C1 output.
  Because every kernel weight is >= 0, sums to 1, and every sample n - j in the
  kernel support has n inside its minimum window, g[n] <= h[n] <= u[n] holds
  exactly; a final min(g, u) removes only cumulative-sum rounding (1.5e-11
  measured at n = 200k, tested < 1e-9). Gain reduction for an isolated dip
  starts exactly 2 * (attack // 2) samples before it, i.e. `attack` rounded
  down to even, never earlier than `attack` (tested for 199 / 200 / 201).
  Without a hull it recovers over the same span after the dip.
  Why the release is a hull and not a longer average: with a matched minimum
  window, any kernel of support width W spreads an isolated dip by exactly W on
  both sides, so a longer averaging window on the release side cannot give a
  longer release than attack while keeping the bound by construction.
- `qp_smooth(g_req, length | attack=..., release=..., mu=...)`: banded QP
      minimise ||D2 g||^2 + mu ||g - h||^2   s.t.  g <= h = release_hull(u)
  mu defaults to length**-4 (the unconstrained smoother's -6 dB point is
  1 / length rad/sample): smoothing length and fidelity weight are one knob
  unless mu is given explicitly. The fidelity term is required; without it
  the problem is unbounded below. Passing `attack` sets length = attack /
  QP_EXTENT_RATIO so an isolated dip's reduction starts no earlier than
  `attack` samples before it (tested). Solver: Mehrotra primal-dual interior
  point on z = h - g >= 0 (pentadiagonal Newton systems, O(n)), then an
  active-set polish that solves the identified equality-constrained system
  exactly, corrects the active set until the KKT sign conditions hold, and
  falls back to the interior-point point if they never do. Measured on 240k
  points: see audit/brief-02.md section C.3. A primal-dual active-set method
  cycles on this problem (Q is not an M-matrix) and projected Newton needs
  hundreds of iterations at long lengths; both were tried and dropped. On exit
  g = clip(h - max(z, 0), g_min, h), so the bound holds whether or not the
  solver converged (reported in the returned info).
- `roughness(g)` = sum of squared second differences, used to compare variants.

Average peak GR (docs/gain-staging.md 3, maintainer 2026-10-19), shared by every dynamics
stage and the report:
- `hit_edges(duration_s, beats)`: the hits, [0, b_1, ..., b_m, duration] from the
  beat times strictly inside the track (the analysis grid, which follows the user's
  --bpm when given); fewer than two beats (no usable grid) gives HIT_FRAME_S (400 ms)
  frames from 0.
- `loud_hits(x, fs, edges_s)`: the loud section, per hit: the 1 s RMS (meter.rms_series,
  chunk 1's meter) of x, the audio entering the stage, at the window centred nearest
  the hit's centre is within LOUD_WITHIN_DB (3 dB) of x's loudest-section value (the
  p95 of the gated 1 s RMS). Fewer than LOUD_MIN_HITS (16) qualifying: the 16 loudest
  hits (every hit when there are fewer).
- `hit_values(gr_db, fs_env, edges_s)`: per hit, the maximum of a GR curve sampled at
  fs_env (sample i at i / fs_env s); hits with no sample are NaN. A curve of one value
  is constant over every hit.
- `average_peak_gr(gr_db, fs_env, edges_s, loud_mask)` -> {"average_db", "outlier_db",
  "hits", "loud_hits"}: average peak GR = the median of the hit values over the loud
  hits (every hit when loud_mask is None), outlier peak GR = the maximum over the whole
  curve. `summarise_hits(values, loud_mask)` is its second half, for a stage whose hit
  values are not the maximum of one curve (the clipper's output-peak fall per hit).
  With edges [0, duration] (one hit) the average equals the outlier: the old
  whole-track maximum reading.
- `hits_for(x, fs, beats, loud=None)` -> {"edges_s", "loud"}: the hits of x and its loud
  section, the form every stage's budget solve takes as `hits` (None: the whole track is
  one hit, the maximum reading). `loud` given (a frozen mask, sprint 2026-10-21 D2: the
  loud set of the stage-1 input, planner.freeze_hits) is used as the loud section instead
  of deriving it on x; its length must equal the number of hits (ValueError otherwise).
- `stage_gr(gr_db, fs_env, hits, reading="meter")`: average_peak_gr of a compressor's
  GR curve as a meter shows it (maintainer, 2026-10-19): the curve is first averaged over
  METER_S (300 ms, the maintainer's meter integration; `meter_smooth`), so a compressor's
  budget reads what its GR meter shows on the loud hits, not its per-sample spike.
  reading "peak" uses the curve as it is. The clipper and limiter read their peaks.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.linalg import solveh_banded
from scipy.ndimage import minimum_filter1d

from eluate import meter

G_MIN = 1e-6
SNAP = 1e-6
# Support half-width of the polished QP's response to an isolated dip, in
# units of `length`: measured 4.40 / 4.42 / 4.43 / 4.435 at lengths 25 / 50 /
# 100 / 200, independent of dip depth, approaching pi * sqrt(2) = 4.443. Rounded
# up so that attack -> length = attack / ratio never starts reduction early.
QP_EXTENT_RATIO = 4.45
# average peak GR (docs/gain-staging.md 3, maintainer 2026-10-19)
HIT_FRAME_S = 0.4
LOUD_WITHIN_DB = 3.0
LOUD_MIN_HITS = 16
OUTLIER_ALLOWANCE_DB = 1.5      # outlier cap = budget + this
AVERAGE_WARN_DB = 0.5           # an average this far below its target after the cap warns
# the meter reading (maintainer, 2026-10-19): a compressor's GR is budgeted as a GR meter shows
# it, the curve averaged over the maintainer's meter integration (300 ms), not its per-sample peak
METER_S = 0.3


def _check(g_req) -> np.ndarray:
    g = np.asarray(g_req, dtype=np.float64)
    if g.ndim != 1:
        raise ValueError(f"gain curves are 1-D, got shape {g.shape}")
    if not np.all(np.isfinite(g)):
        raise ValueError("gain curve contains NaN or Inf")
    return g


def _ceiling(g_req, g_min: float) -> np.ndarray:
    """u = clip(g_req, g_min, 1): the bound every smoother respects."""
    return np.clip(_check(g_req), g_min, 1.0)


def required_gain(env: np.ndarray, threshold: float) -> np.ndarray:
    env = np.asarray(env, dtype=np.float64)
    if not np.all(np.isfinite(env)):
        raise ValueError("detector contains NaN or Inf")
    g = np.ones_like(env)
    pos = env > threshold
    g[pos] = threshold / env[pos]
    return g


def roughness(g: np.ndarray) -> float:
    d2 = np.diff(np.asarray(g, dtype=np.float64), 2)
    return float(np.dot(d2, d2))


def _box(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Mean of x[i+lo .. i+hi] (inclusive) with edge-replicate padding."""
    n = x.shape[0]
    pad_l, pad_r = max(0, -lo), max(0, hi)
    xp = np.concatenate([np.full(pad_l, x[0]), x, np.full(pad_r, x[-1])])
    # c = [0, cumsum(xp)] and the window sums by slices: the same values as the
    # gather form c[i + hi + 1] - c[i + lo], i = arange(n) + pad_l, with fewer arrays
    c = np.empty(xp.size + 1)
    c[0] = 0.0
    np.cumsum(xp, out=c[1:])
    del xp
    a0, b0 = pad_l + hi + 1, pad_l + lo
    out = c[a0:a0 + n] - c[b0:b0 + n]
    out /= (hi - lo + 1)
    return out


def release_hull(u: np.ndarray, release: float) -> np.ndarray:
    """h[n] = min(u[n], h[n-1] recovered by exp(-1/release) in dB). h <= u.

    depth_h[n] = max_k a^(n-k) depth[k] is evaluated in the log domain as
    n ln a + cummax(log depth[k] - k ln a), restarted every B samples
    (B |ln a| <= 50, B >= 256) with the previous block's last value carried in,
    so the k ln a terms stay small. release < 0.01 samples means no hull.
    """
    u = _check(u)
    if release < 0.01 or u.size == 0:
        return u.copy()
    with np.errstate(divide="ignore"):
        # in place, the same operations as log(max(-20 log10(max(u, 1e-300)), 0))
        w = np.maximum(u, 1e-300)
        np.log10(w, out=w)
        w *= -20.0
        np.maximum(w, 0.0, out=w)       # depth
        np.log(w, out=w)                # -inf where no reduction
    ln_a = -1.0 / release
    n = u.shape[0]
    block = int(min(n, max(256, 50.0 / abs(ln_a))))
    k = np.arange(block, dtype=np.float64)
    hull_log = np.empty(n)
    carry = -np.inf
    for start in range(0, n, block):
        seg = w[start:start + block]
        kk = k[: seg.size]
        local = np.maximum.accumulate(seg - kk * ln_a) + kk * ln_a
        local = np.maximum(local, carry + (kk + 1.0) * ln_a)
        hull_log[start:start + seg.size] = local
        carry = local[-1]
    del w
    # in place, the same operations as 10 ** (-exp(hull_log) / 20)
    h = np.exp(hull_log, out=hull_log)
    np.negative(h, out=h)
    h /= 20.0
    np.power(10.0, h, out=h)
    # min(u) <= h <= u holds exactly; the clip removes log/exp rounding
    return np.clip(h, u.min(), u, out=h)


def _box_smooth_raw(g_req, attack: int, release: float = 0.0, g_min: float = G_MIN):
    u = _ceiling(g_req, g_min)
    if u.size == 0:
        return u, u
    h = release_hull(u, release)
    L = max(1, int(attack) // 2 + 1)
    if L == 1:
        return h.copy(), u
    m = minimum_filter1d(h, size=2 * L - 1, mode="nearest")  # offsets [-(L-1), L-1]
    g = _box(_box(m, -(L - 1), 0), 0, L - 1)
    return g, u


def box_smooth(g_req: np.ndarray, attack: int, release: float = 0.0,
               g_min: float = G_MIN) -> np.ndarray:
    """Lookahead min + twice-applied moving average; g <= clip(g_req, g_min, 1) exactly."""
    g, u = _box_smooth_raw(g_req, attack, release, g_min)
    return np.clip(g, g_min, u)


# ------------------------------------------------------------------ QP

def _q_bands(n: int, mu: float):
    """Diagonal, first and second off-diagonals of Q = D2'D2 + mu I (n >= 4)."""
    d0 = np.full(n, 6.0)
    d1 = np.full(max(n - 1, 0), -4.0)
    if n >= 1:
        d0[0] = d0[-1] = 1.0
    if n >= 2:
        d0[1] = d0[-2] = 5.0
        d1[0] = d1[-1] = -2.0
    d0 += mu
    d2 = np.ones(max(n - 2, 0))
    return d0, d1, d2


def _q_mul(v: np.ndarray, mu: float) -> np.ndarray:
    n = v.shape[0]
    r = mu * v
    if n >= 3:
        w = v[:-2] - 2.0 * v[1:-1] + v[2:]
        r[:-2] += w
        r[1:-1] -= 2.0 * w
        r[2:] += w
    return r


def _solve_free(fi, d0, d1p, d2p, rhs):
    """Solve Q_FF y = rhs; Q_FF is pentadiagonal in compressed indexing."""
    off1 = np.zeros(fi.size)
    off2 = np.zeros(fi.size)
    if fi.size >= 2:
        gap = np.diff(fi)
        head = fi[:-1]
        off1[1:] = np.where(gap == 1, d1p[head], np.where(gap == 2, d2p[head], 0.0))
    if fi.size >= 3:
        off2[2:] = np.where(fi[2:] - fi[:-2] == 2, 1.0, 0.0)
    return solveh_banded(np.vstack([off2, off1, d0[fi]]), rhs, check_finite=False)


def qp_smooth(
    g_req: np.ndarray,
    length: float | None = None,
    *,
    attack: float | None = None,
    release: float = 0.0,
    mu: float | None = None,
    g_min: float = G_MIN,
    max_iter: int = 100,
    tol: float = 1e-8,
    polish: bool = True,
) -> tuple[np.ndarray, dict]:
    """Smoothest envelope under g <= h = release_hull(clip(g_req, g_min, 1)).

    Give `length` (samples) or `attack` (samples; length = attack /
    QP_EXTENT_RATIO) or `mu` directly; mu defaults to length**-4.

    Interior point in z = h - g >= 0: minimise 1/2 z'Qz - b'z, Q = D2'D2 + mu I,
    b = D2'D2 h; stops when z.lambda/n <= tol*mu and the dual residual max-norm
    <= max(tol*mu, 1e-13) (cond(Q) ~ 16/mu puts the residual's rounding floor
    near 4e-16 absolute). Polish: active set A = {lambda/mu > z} from the
    interior point, seeded as A = {lambda/mu > 0.1 z}; solve Q_FF z_F = b_F
    with z_A = 0; move free points with z_F < 0 into A and active points with
    lambda_A = (Qz - b)_A < 0 out of A; repeat (at most 60 times) until both sign conditions hold (to 1e-12 relative), which makes it
    the exact optimum; otherwise keep the interior-point solution. Polishing removes the order-1e-5 uniform offset the
    interior point leaves in the flat, mu-curvature direction.
    """
    t0 = time.perf_counter()
    u = _ceiling(g_req, g_min)
    n = u.shape[0]
    if attack is not None:
        length = max(float(attack), 1.0) / QP_EXTENT_RATIO
    if mu is None:
        if length is None:
            raise ValueError("give length, attack or mu")
        mu = float(length) ** -4.0
    h = release_hull(u, release)
    if n < 4:
        return h.copy(), {"iterations": 0, "converged": True, "polished": False,
                          "seconds": 0.0, "mu": mu, "length": mu ** -0.25}
    d0, d1, d2 = _q_bands(n, mu)
    b = _q_mul(h, mu) - mu * h  # D2'D2 h
    rd_tol = max(tol * mu, 1e-13)
    # far from any dip the iterate sits on the central path with z ~ sqrt(mc / mu),
    # so complementarity is driven to tol^2 * mu to make that offset ~tol
    mc_tol = tol * tol * mu
    ab = np.zeros((3, n))  # upper banded storage of Q + D
    ab[0, 2:] = d2
    ab[1, 1:] = d1

    def solve(dg, rhs):
        ab[2] = d0 + dg
        return solveh_banded(ab, rhs, check_finite=False)

    def max_step(v, dv):
        neg = dv < 0.0
        return float(np.min(-v[neg] / dv[neg])) if neg.any() else math.inf

    z = np.full(n, 0.5)
    lam = np.ones(n)
    converged = False
    it = 0
    rd_norm = mc = math.inf
    for it in range(1, max_iter + 1):
        rd = _q_mul(z, mu) - b - lam
        mc = float(z @ lam) / n
        rd_norm = float(np.max(np.abs(rd)))
        if rd_norm <= rd_tol and mc <= mc_tol:
            converged = True
            break
        dg = lam / z
        # predictor (affine scaling)
        dz = solve(dg, -rd - lam)
        dl = -lam - dg * dz
        a = min(1.0, max_step(z, dz), max_step(lam, dl))
        mc_aff = float((z + a * dz) @ (lam + a * dl)) / n
        sigma = (mc_aff / mc) ** 3
        # corrector
        rc = z * lam + dz * dl - sigma * mc
        dz = solve(dg, -rd - rc / z)
        dl = (-rc - lam * dz) / z
        a = min(1.0, 0.99 * min(max_step(z, dz), max_step(lam, dl)))
        z = z + a * dz
        lam = lam + a * dl

    polished = False
    polish_steps = 0
    if polish:
        d1p, d2p = np.append(d1, 0.0), np.append(d2, [0.0, 0.0])
        zmax = float(np.max(z))
        scale = float(np.max(np.abs(b))) + mu
        # far from any dip the interior point sits on the central path with
        # z ~ lambda/mu; biasing the test 10x toward "active" classifies those
        # points correctly and leaves the support boundary to the corrections.
        active = lam / mu > 0.1 * z
        for polish_steps in range(1, 11):
            fi = np.flatnonzero(~active)
            zp = np.zeros(n)
            if fi.size:
                zp[fi] = _solve_free(fi, d0, d1p, d2p, b[fi])
            lam_p = _q_mul(zp, mu) - b
            bad_free = (~active) & (zp < -1e-12 * max(zmax, 1e-300))
            bad_act = active & (lam_p < -1e-12 * scale)
            if not bad_free.any() and not bad_act.any():
                z = zp
                polished = True
                break
            active = (active | bad_free) & ~bad_act
        if not polished:
            # snap: zeroing z only raises g toward h, so g <= h still holds
            z = np.where(z < SNAP * max(zmax, 1e-300), 0.0, z)
    g = np.clip(h - np.maximum(z, 0.0), g_min, h)
    info = {
        "iterations": it,
        "converged": converged,
        "polished": polished,
        "polish_steps": polish_steps,
        "dual_residual": rd_norm,
        "complementarity": mc,
        "seconds": time.perf_counter() - t0,
        "mu": mu,
        "length": float(mu ** -0.25),
    }
    return g, info


# ------------------------------------------------------------------ average peak GR

def hit_edges(duration_s: float, beats=None) -> np.ndarray:
    """Hit boundaries in seconds (module contract)."""
    d = float(duration_s)
    if not d > 0.0:
        return np.array([0.0, max(d, 0.0)])
    b = np.zeros(0) if beats is None else np.asarray(beats, dtype=np.float64).ravel()
    b = np.unique(b[np.isfinite(b) & (b > 0.0) & (b < d)])
    if b.size < 2:
        b = np.arange(HIT_FRAME_S, d, HIT_FRAME_S)
    return np.concatenate([[0.0], b, [d]])


def loud_hits(x: np.ndarray, fs: float, edges_s) -> np.ndarray:
    """Per hit, True inside the loud section of x (module contract)."""
    e = np.asarray(edges_s, dtype=np.float64)
    nh = e.size - 1
    if nh <= 0:
        return np.zeros(0, dtype=bool)
    t_end, db = meter.rms_series(x, fs)
    if db.size == 0:
        return np.ones(nh, dtype=bool)
    centres = t_end - meter.RMS_WINDOW_S / 2.0
    mid = 0.5 * (e[:-1] + e[1:])
    j = np.clip(np.searchsorted(centres, mid), 1, max(1, centres.size - 1))
    if centres.size > 1:
        j = np.where(np.abs(centres[j - 1] - mid) <= np.abs(centres[j] - mid), j - 1, j)
    else:
        j = np.zeros(nh, dtype=np.int64)
    lv = db[j]
    gated = db[db > meter.RMS_GATE_DBFS]
    if gated.size == 0:
        return np.ones(nh, dtype=bool)
    mask = lv >= float(np.percentile(gated, 95.0)) - LOUD_WITHIN_DB
    if mask.sum() < min(LOUD_MIN_HITS, nh):
        mask = np.zeros(nh, dtype=bool)
        mask[np.argsort(-lv, kind="stable")[:LOUD_MIN_HITS]] = True
    return mask


def hit_values(gr_db, fs_env: float, edges_s) -> np.ndarray:
    """Per-hit maximum of a GR curve (module contract)."""
    g = np.asarray(gr_db, dtype=np.float64).ravel()
    e = np.asarray(edges_s, dtype=np.float64)
    nh = max(0, e.size - 1)
    if g.size == 1:
        return np.full(nh, float(g[0]))
    idx = np.clip(np.ceil(e * fs_env - 1e-9).astype(np.int64), 0, g.size)
    out = np.full(nh, np.nan)
    full = idx[1:] > idx[:-1]
    if np.any(full) and g.size:
        out[full] = np.maximum.reduceat(g, idx[:-1][full])
        # reduceat runs to the next start; hits are contiguous, so only the last full hit
        # can run past its own end
        k = np.flatnonzero(full)[-1]
        out[k] = float(np.max(g[idx[k]:idx[k + 1]]))
    return out


def summarise_hits(values, loud_mask=None) -> dict:
    """Average (median over the loud hits) and outlier (maximum) of per-hit values."""
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    loud = ok if loud_mask is None else ok & np.asarray(loud_mask, dtype=bool)
    if not np.any(loud):
        loud = ok
    if not np.any(ok):
        return {"average_db": 0.0, "outlier_db": 0.0, "hits": int(v.size), "loud_hits": 0}
    return {"average_db": float(np.median(v[loud])) + 0.0, "outlier_db": float(np.max(v[ok])) + 0.0,
            "hits": int(v.size), "loud_hits": int(np.count_nonzero(loud))}


def average_peak_gr(gr_db, fs_env: float, edges_s, loud_mask=None) -> dict:
    """Average and outlier peak GR of a GR curve (dB, positive = reduction; module
    contract). edges_s None: the whole curve is one hit (the maximum reading)."""
    g = np.asarray(gr_db, dtype=np.float64).ravel()
    if edges_s is None:
        edges_s = [0.0, g.size / fs_env]
    out = summarise_hits(hit_values(g, fs_env, edges_s), loud_mask)
    if g.size:
        out["outlier_db"] = max(out["outlier_db"], float(np.max(g)) + 0.0)
    return out


def hits_for(x: np.ndarray, fs: float, beats=None, loud=None) -> dict:
    """The hits and loud section of x, the audio entering a stage; loud: a frozen mask
    used instead of deriving the loud section on x (module contract)."""
    x = np.asarray(x)
    e = hit_edges(x.shape[0] / fs, beats)
    if loud is None:
        return {"edges_s": e, "loud": loud_hits(x, fs, e)}
    m = np.asarray(loud, dtype=bool).ravel()
    if m.size != e.size - 1:
        raise ValueError(f"frozen loud mask has {m.size} hits, the audio has {e.size - 1}")
    return {"edges_s": e, "loud": m.copy()}


def meter_smooth(gr_db, fs_env: float, window_s: float = METER_S) -> np.ndarray:
    """A GR curve as a meter shows it: the centred window_s moving average (edges
    replicated). A curve of one point or a window under two points is returned as is."""
    g = np.asarray(gr_db, dtype=np.float64).ravel()
    w = int(round(window_s * fs_env))
    if g.size < 2 or w < 2:
        return g.copy()
    lo = -(w // 2)
    return _box(g, lo, lo + w - 1)


def stage_gr(gr_db, fs_env: float, hits: dict | None, reading: str = "meter") -> dict:
    """average_peak_gr of a compressor's GR curve over a stage's hits (None: one hit), on
    the meter reading (meter_smooth) by default; reading "peak" uses the curve itself."""
    if reading not in ("meter", "peak"):
        raise ValueError("reading must be 'meter' or 'peak'")
    g = meter_smooth(gr_db, fs_env) if reading == "meter" else gr_db
    if hits is None:
        return average_peak_gr(g, fs_env, None)
    return average_peak_gr(g, fs_env, hits["edges_s"], hits["loud"])
