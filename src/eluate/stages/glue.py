"""Single-band parallel "glue" bus compressor (spec 3.5 item 6; audit 06 F1, F2, M1 to M3, M10).

Contract: `apply(x, fs, plan) -> y`, pure, stereo (n, 2) in, float64 (n, 2) out.
- One stereo-linked gain envelope (plan.gain_db, 1 ms, PCHIP to audio rate)
  is applied to both channels of the wet path with the makeup:
      wet = x * 10^((G(t) + makeup_db) / 20)
      wet += E(wet)                        (vintage curve only, see below)
      y = x + mix * (wet - x)
  mix 0 returns the input bit-exactly; with G = 0 dB, makeup 0 and a zero
  vintage term, wet == x, so y == x bit-exactly for any mix. No crossover is
  involved (single band).
- G may be positive up to GLUE_MAX_GAIN_DB (plan validation) so that other
  topologies (upward compression) are expressible; `design` only produces
  G <= 0.
- Vintage curve E (wet path only): a slight even-harmonic soft curve on the
  wet signal (after the envelope and makeup), so it engages where the *wet*
  signal exceeds T = 10^(threshold_db / 20); with makeup > 0 that includes
  input below the threshold:
      E(v) = (VINTAGE_EVEN / 2) T tanh^2(max(|v| - T, 0) / T)
  per channel, evaluated at 4x through stages/resample.py (only the change is
  resampled), then high-passed by a causal 2nd-order Butterworth at
  VINTAGE_HP_HZ to remove the DC an even curve produces. The high-pass is
  causal, so nothing appears before the first sample above T; after a burst
  the removed DC decays as the filter's tail. E is a function of |v|, hence
  purely even. A wet signal that never exceeds T gets E == 0 exactly.

`design(x, fs, threshold_db, ...)` builds the envelope at the 1 ms rate:
- Causal stereo-linked detector: control point k = sample peak of
  max(|L|, |R|) of the input through a causal 2nd-order Butterworth high-pass at
  SIDECHAIN_HP_HZ (sprint 2026-10-21 D3: full band, the kick drove the glue as a
  beat-rate ducker) over the control segment [k - 2, k - 1) (segment j = the
  samples Envelope.at interpolates between points j and j + 1). The PCHIP
  segment between points k and k + 1 is flat unless point k + 1 differs from
  point k, and point k + 1 depends only on samples before point k, so no
  gain reduction appears before the first sample over the threshold. Cost:
  1 to 2 ms of detector delay.
- Gain computer, soft knee of width W = knee_db (Giannoulis, Massberg and
  Reiss, JAES 60(6), 2012): level L in dBFS, GR = 0 below T - W/2,
  (1/R - 1)(L - T + W/2)^2 / (2W) inside the knee, (1/R - 1)(L - T) above.
- Release: the program-dependent release on either curve (D3: a fixed 300 ms hull
  never recovered between beats), min(hull(u, R/2), hull(s, 2R)), s the causal mean
  of the required GR in dB over the preceding R (recovers at R/2 after short hits, up
  to 2R after sustained compression; the constants are the builder's, audit 06 D9).
- Lagging attack: a causal branching one-pole in dB on the hull h:
  s[n] = s[n-1] + (1 - a) (h[n] - s[n-1]) while h[n] < s[n-1] (more reduction
  wanted), a = exp(-1 / attack samples); s[n] = h[n] otherwise (the release
  follows the hull). A step in the required GR reaches 63 % of its depth one
  attack time after the detector sees it; the transient passes the wet path
  nearly unreduced. At the 1 ms control rate an attack under 1 ms is a near-
  immediate step after the detector's 1 to 2 ms delay: the presets' 0.1 to 0.8 ms
  differ little in effect (D3), they are not sub-millisecond attacks. Unlike the multiband and limiter envelopes there is no
  g <= g_req bound: letting the attack through is the point.
- makeup_db is explicit (the solver's), default 0, in [0, 6] dB.

Budget solve (docs/gain-staging.md 3, gain-staging chunk 3; the average-peak reading,
maintainer 2026-10-19). The GR solved for is the average peak GR over `hits`
(envelope.hits_for on the glue's input: the median over the loud hits of each hit's
maximum GR, on the meter reading: envelope.stage_gr averages the GR over 300 ms first);
the outlier is the maximum over the track on the same reading. hits None makes the
track one hit (the maximum, the chunk 3 reading). The wet path's GR curve is
-gain_db; the net (output) one, with makeup 0, is
-20 log10(1 - mix + mix 10^(gain_db / 20)), monotonic in the wet gain.
- `threshold_for_peak_gr(x, fs, target_db, ratio, knee_db, attack_ms,
  release_ms, curve, hits=None, floor_db=None)`: the threshold whose envelope (this
  design, lagging attack included) has a wet average peak GR of target_db on x, by
  bisection. floor_db: the wet path's GR clamp (the required gain floored there
  before the release hull), how the outlier cap is held.
- `design_budget(x, fs, mode, ..., hits=None)`: "serial" (the default): mix 1, the
  threshold solved for a net average peak GR of net_db (1.5 dB, range 1 to 2),
  every GR read on the meter reading (envelope.stage_gr, 2026-10-19), the wet GR
  clamped per sample at SAMPLE_MAX_GR_DB and the outlier reported against
  OUTLIER_CAP_DB (4 dB on the meter reading);
  "parallel": the wet path solved for wet_db (3 dB, range 2 to 4) and the mix
  chosen so the net average peak GR equals net_db (`mix_for_net`), then, when the
  wet outlier passes the wet GR at which the net reaches SAMPLE_MAX_GR_DB at that mix,
  the wet path re-solved with its GR clamped there (the safety clamp); the net outlier
  is reported against OUTLIER_CAP_DB, not enforced (docs/gain-staging.md 18). A net average more than
  envelope.AVERAGE_WARN_DB (0.5 dB) below net_db (a silent input, a lagging attack
  that lets every peak through, or the clamp reaching into the loud hits) gives a
  warning {"stage": "glue", "mode", "target_db", "achieved_db"}. The record is
  GluePlan.budget.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from eluate import envelope
from eluate.plan import CONTROL_RATE_HZ, Envelope, GluePlan, control_length
from eluate.stages import resample
from eluate.stages.multiband import _frame_peaks, to_env32

SERIAL_GR_DB = 1.5              # net peak GR, serial (range 1 to 2)
SERIAL_RANGE_DB = (1.0, 2.0)
WET_GR_DB = 3.0                 # wet-path peak GR, parallel (range 2 to 4)
WET_RANGE_DB = (2.0, 4.0)
WARN_DB = 0.5                   # = envelope.AVERAGE_WARN_DB
OUTLIER_CAP_DB = 4.0            # net outlier GR cap on the meter reading (dynamics-macros 3)
SAMPLE_MAX_GR_DB = 10.0         # per-sample safety clamp of the wet GR (the meter reading)
SOLVE_TOL_DB = 1e-4
SOLVE_ITER = 80
BRACKET_STEP_DB = 6.0
BRACKET_MAX_DB = 60.0
VINTAGE_EVEN = 0.1
VINTAGE_HP_HZ = 20.0
OVERSAMPLE = 4
DETECTOR_DELAY = 2  # control segments (see module contract)
SIDECHAIN_HP_HZ = 120.0         # D3: the detector does not hear the sub


def gain_computer(level_db, threshold_db: float, ratio: float, knee_db: float) -> np.ndarray:
    """Static soft-knee curve: gain reduction in dB (<= 0) for detector level in dBFS."""
    lv = np.asarray(level_db, dtype=np.float64)
    over = lv - threshold_db
    slope = 1.0 / ratio - 1.0
    gr = np.where(over > 0.0, slope * over, 0.0)
    if knee_db > 0.0:
        half = knee_db / 2.0
        inside = np.abs(over) <= half
        gr = np.where(inside, slope * (over + half) ** 2 / (2.0 * knee_db), gr)
        gr = np.where(over < -half, 0.0, gr)
    return np.minimum(gr, 0.0)


def program_release(u: np.ndarray, release: float) -> np.ndarray:
    """Vintage release (see module contract); u, release in control samples. Result <= u."""
    if release < 0.01:
        return envelope.release_hull(u, 0.0)
    fast = envelope.release_hull(u, release / 2.0)
    u_db = 20.0 * np.log10(np.clip(u, envelope.G_MIN, 1.0))
    w = max(1, int(round(release)))
    c = np.concatenate([[0.0], np.cumsum(u_db)])
    i = np.arange(u.size)
    lo = np.maximum(i + 1 - w, 0)
    s_db = (c[i + 1] - c[lo]) / (i + 1 - lo)
    slow = envelope.release_hull(10.0 ** (s_db / 20.0), 2.0 * release)
    return np.minimum(fast, slow)


def lagging_attack(h_db: np.ndarray, attack: float) -> np.ndarray:
    """Causal branching one-pole in dB (see module contract); attack in control samples."""
    a = float(np.exp(-1.0 / attack))
    out = np.empty_like(h_db)
    s = 0.0
    for i, v in enumerate(h_db.tolist()):
        s = s + (1.0 - a) * (v - s) if v < s else v
        out[i] = s
    return out


def causal_detector(v: np.ndarray, fs: float, n_ctrl: int, rate_hz: float) -> np.ndarray:
    """Point k = peak |v| over control segment k - DETECTOR_DELAY (0 before the start)."""
    pk = _frame_peaks(v, fs, n_ctrl, rate_hz)
    return np.concatenate([np.zeros(DETECTOR_DELAY), pk[:-DETECTOR_DELAY]])[:n_ctrl]


def _gain_env(level, threshold_db, ratio, knee_db, attack_ms, release_ms, curve,
              rate_hz, floor_db: float | None = None) -> Envelope:
    """The design's envelope from the detector level (dBFS) at the control rate; floor_db
    clamps the required GR before the release hull."""
    g_min = envelope.G_MIN if floor_db is None else max(envelope.G_MIN,
                                                        10.0 ** (-floor_db / 20.0))
    u = np.clip(10.0 ** (gain_computer(level, threshold_db, ratio, knee_db) / 20.0),
                g_min, 1.0)
    release = release_ms * rate_hz / 1000.0
    h = program_release(u, release)
    g_db = lagging_attack(20.0 * np.log10(h), attack_ms * rate_hz / 1000.0)
    return to_env32(10.0 ** (g_db / 20.0), rate_hz)


def _level(x: np.ndarray, fs: float, rate_hz: float) -> np.ndarray:
    n_ctrl = control_length(x.shape[0], fs, rate_hz)
    sos = signal.butter(2, SIDECHAIN_HP_HZ, "highpass", fs=fs, output="sos")
    side = signal.sosfilt(sos, np.asarray(x, dtype=np.float64), axis=0)
    det = causal_detector(np.max(np.abs(side), axis=1), fs, n_ctrl, rate_hz)
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(det)


def wet_peak_gr_db(plan: GluePlan) -> float:
    """Peak GR of the wet path (dB >= 0), the envelope alone."""
    return float(max(0.0, -np.min(plan.gain_db.values)))


def net_peak_gr_db(plan: GluePlan) -> float:
    """Peak GR of the output y = x + mix (wet - x) (dB >= 0), makeup included."""
    g = 10.0 ** ((float(np.min(plan.gain_db.values)) + plan.makeup_db) / 20.0)
    return float(max(0.0, -20.0 * np.log10(1.0 - plan.mix + plan.mix * g)))


def net_gr_curve_db(plan: GluePlan) -> np.ndarray:
    """The output's GR curve (dB >= 0) at the control rate, makeup included."""
    g = 10.0 ** ((plan.gain_db.values + plan.makeup_db) / 20.0)
    return np.maximum(-20.0 * np.log10(1.0 - plan.mix + plan.mix * g), 0.0)


def wet_floor_for_net(mix: float, net_cap_db: float) -> float | None:
    """The wet GR at which the net GR reaches net_cap_db at this mix (None: never)."""
    a = 10.0 ** (-net_cap_db / 20.0) - (1.0 - mix)
    if mix <= 0.0 or a <= 0.0:
        return None
    return float(-20.0 * np.log10(a / mix))


def mix_for_net(wet_gr_db: float, net_db: float) -> float:
    """Wet share giving a net peak GR of net_db from a wet peak GR of wet_gr_db (makeup
    0): 1 - m + m 10^(-wet / 20) = 10^(-net / 20); at most 1."""
    if wet_gr_db <= 0.0:
        return 1.0
    return float(min(1.0, (1.0 - 10.0 ** (-net_db / 20.0)) / (1.0 - 10.0 ** (-wet_gr_db / 20.0))))


def threshold_for_peak_gr(x: np.ndarray, fs: float, target_db: float, ratio: float = 3.0,
                          knee_db: float = 6.0, attack_ms: float = 30.0,
                          release_ms: float = 300.0, curve: str = "clean",
                          rate_hz: float = CONTROL_RATE_HZ, hits: dict | None = None,
                          floor_db: float | None = None) -> dict:
    """Threshold (dBFS) whose envelope has a wet average peak GR of target_db on x over
    hits (the maximum without hits; module contract). Returns {"threshold_db",
    "peak_gr_db" (the average reached), "outlier_gr_db"}; when no threshold within
    BRACKET_MAX_DB below the peak reaches the target, the lowest one tried."""
    from eluate.stages.multiband import bisect_peak_gr
    x = np.asarray(x, dtype=np.float64)
    level = _level(x, fs, rate_hz)
    top = float(np.max(level)) if level.size else -np.inf
    if not np.isfinite(top) or target_db <= 0.0:
        return {"threshold_db": 0.0, "peak_gr_db": 0.0, "outlier_gr_db": 0.0}
    last = {}

    def gr(t):
        env = _gain_env(level, t, ratio, knee_db, attack_ms, release_ms, curve, rate_hz,
                        floor_db)
        r = envelope.stage_gr(np.maximum(-env.values, 0.0), rate_hz, hits)
        last[t] = r["outlier_db"]
        return r["average_db"]

    hi = top + knee_db / 2.0 + 0.01
    lo = top - target_db / (1.0 - 1.0 / ratio) - knee_db
    while gr(lo) < target_db and lo > top - BRACKET_MAX_DB:
        lo -= BRACKET_STEP_DB
    t, g = bisect_peak_gr(gr, lo, hi, target_db)
    return {"threshold_db": float(t), "peak_gr_db": float(g), "outlier_gr_db": float(last[t])}


def design_budget(x: np.ndarray, fs: float, mode: str = "serial", ratio: float = 3.0,
                  knee_db: float = 6.0, attack_ms: float = 30.0, release_ms: float = 300.0,
                  curve: str = "clean", net_db: float = SERIAL_GR_DB,
                  wet_db: float = WET_GR_DB, rate_hz: float = CONTROL_RATE_HZ,
                  hits: dict | None = None) -> GluePlan:
    """Serial or parallel glue solved for its average peak GR budget over hits (module
    contract)."""
    if mode not in ("serial", "parallel"):
        raise ValueError("mode must be 'serial' or 'parallel'")
    if not SERIAL_RANGE_DB[0] <= net_db <= SERIAL_RANGE_DB[1]:
        raise ValueError(f"net_db must lie in {SERIAL_RANGE_DB}")
    if mode == "parallel" and not WET_RANGE_DB[0] <= wet_db <= WET_RANGE_DB[1]:
        raise ValueError(f"wet_db must lie in {WET_RANGE_DB}")
    serial = mode == "serial"
    target = net_db if serial else wet_db
    floor = SAMPLE_MAX_GR_DB if serial else None
    kw = dict(ratio=ratio, knee_db=knee_db, attack_ms=attack_ms, release_ms=release_ms,
              curve=curve, rate_hz=rate_hz)
    sol = threshold_for_peak_gr(x, fs, target, **kw, hits=hits, floor_db=floor)
    mix = 1.0 if serial else mix_for_net(sol["peak_gr_db"], net_db)
    if not serial:
        floor = wet_floor_for_net(mix, SAMPLE_MAX_GR_DB)
        if floor is not None and sol["outlier_gr_db"] > floor + 1e-9:
            sol = threshold_for_peak_gr(x, fs, target, **kw, hits=hits, floor_db=floor)
        else:
            floor = None
    p = design(x, fs, sol["threshold_db"], mix=mix, floor_db=floor, **kw)
    wet = envelope.stage_gr(np.maximum(-p.gain_db.values, 0.0), rate_hz, hits)
    net = envelope.stage_gr(net_gr_curve_db(p), rate_hz, hits)
    warn = []
    if net["average_db"] < net_db - WARN_DB:
        warn.append({"stage": "glue", "mode": mode, "target_db": float(net_db),
                     "achieved_db": net["average_db"]})
    p.budget = {"mode": mode, "net_target_db": float(net_db),
                "wet_target_db": float(target), "wet_db": wet["average_db"],
                "wet_outlier_db": wet["outlier_db"], "net_db": net["average_db"],
                "net_outlier_db": net["outlier_db"], "net_cap_db": OUTLIER_CAP_DB,
                "wet_floor_db": floor, "mix": float(mix),
                "reading": "max" if hits is None else "average", "warnings": warn}
    return p


def design(x: np.ndarray, fs: float, threshold_db: float, ratio: float = 2.0,
           knee_db: float = 6.0, attack_ms: float = 30.0, release_ms: float = 300.0,
           curve: str = "clean", mix: float = 0.5, makeup_db: float = 0.0,
           rate_hz: float = CONTROL_RATE_HZ, floor_db: float | None = None) -> GluePlan:
    """Stereo-linked lagging-attack glue envelope (see module contract); floor_db clamps
    the wet GR per sample (the budget solve's SAMPLE_MAX_GR_DB safety clamp)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("glue needs stereo (n, 2) audio")
    if not ratio > 1.0:
        raise ValueError("ratio must be > 1")
    if curve not in ("clean", "vintage"):
        raise ValueError("curve must be 'clean' or 'vintage'")
    if not attack_ms > 0:
        raise ValueError("attack_ms must be > 0")
    env = _gain_env(_level(x, fs, rate_hz), threshold_db, ratio, knee_db, attack_ms, release_ms,
                    curve, rate_hz, floor_db)
    return GluePlan(gain_db=env if np.any(env.values) else Envelope.constant(0.0), mix=mix,
                    makeup_db=float(makeup_db) + 0.0, curve=curve, threshold_db=threshold_db,
                    ratio=ratio, knee_db=knee_db, attack_ms=attack_ms, release_ms=release_ms)


def vintage_change(wet: np.ndarray, fs: float, threshold_db: float) -> np.ndarray:
    """E(wet) for both channels (see module contract); exactly 0 where |wet| <= T throughout."""
    t_lin = 10.0 ** (threshold_db / 20.0)
    sos = signal.butter(2, VINTAGE_HP_HZ, btype="highpass", fs=fs, output="sos")
    out = np.zeros_like(wet)
    for c in range(wet.shape[1]):
        if not np.any(np.abs(wet[:, c]) > t_lin):
            continue
        u = resample.upsample(wet[:, c], OVERSAMPLE)
        e = 0.5 * VINTAGE_EVEN * t_lin * np.tanh(np.maximum(np.abs(u) - t_lin, 0.0) / t_lin) ** 2
        out[:, c] = signal.sosfilt(sos, resample.decimate(e, OVERSAMPLE))
    return out


def apply(x: np.ndarray, fs: float, plan: GluePlan) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("glue needs stereo (n, 2) audio")
    if plan.mix == 0.0:
        return x.copy()
    total = plan.gain_db.at(x.shape[0], fs) + plan.makeup_db
    wet = x * (10.0 ** (total / 20.0))[:, None] if np.any(total) else x.copy()
    if plan.curve == "vintage":
        wet = wet + vintage_change(wet, fs, plan.threshold_db)
    return x + plan.mix * (wet - x)


def gain_trace_db(plan: GluePlan, n: int, fs: float) -> np.ndarray:
    """Wet-path gain in dB at audio rate (envelope + makeup), for reporting."""
    return plan.gain_db.at(n, fs) + plan.makeup_db
