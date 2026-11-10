"""Drive each dynamic stage as the planner drives it (planner.character_settings mirrored
without an analysis dict), and extract the audio-rate gain each stage applies."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .common import FS  # noqa: F401  (sys.path setup)
from eluate import bands, envelope, meter, planner  # noqa: E402
from eluate.plan import (CONTROL_RATE_HZ, MULTIBAND_MAX_GR_DB, Envelope, GluePlan,  # noqa: E402
                         LimiterPlan, MultibandPlan)
from eluate.stages import glue as gl, limiter as lim, multiband as mb, resample  # noqa: E402

CHARS = ("transparent", "slow", "moderate", "fast", "aggressive")
PRESETS = planner.CHARACTER_PRESETS
PROFILES = planner.PROFILES
XO = (100.0, 400.0, 1600.0, 6400.0)


def timing(character: str, profile: str = "electronic", nb: int = 5) -> dict:
    """planner.character_settings without the analysis (same rules)."""
    pr = PRESETS[character]
    prof = PROFILES[profile]
    rel = list(pr["mb_release_ms"])[:nb]
    rel[0] = max(rel[0], planner.LOW_BAND_RELEASE_MIN_MS)
    lim_t = PRESETS[prof["limiter_timing"]] if prof["limiter_timing"] else pr
    glue_att = prof["glue_attack_ms"] if prof["glue_attack_ms"] is not None else pr["glue_attack_ms"]
    return {"mb_attack_ms": list(pr["mb_attack_ms"])[:nb], "mb_release_ms": rel,
            "mb_knee_db": pr["mb_knee_db"], "band_lookahead_ms": list(pr["band_lookahead_ms"])[:nb],
            "speclim_lookahead_ms": list(lim_t["band_lookahead_ms"])[:nb],
            "glue_attack_ms": glue_att, "glue_release_ms": pr["glue_release_ms"],
            "clip_knee_db": pr["clip_knee_db"], "limiter_attack_ms": lim_t["limiter_attack_ms"],
            "limiter_release_ms": max(lim_t["limiter_release_ms"], planner.LIMITER_RELEASE_MIN_MS),
            "limiter_release_raw_ms": lim_t["limiter_release_ms"]}


# ------------------------------------------------------------------ multiband

def mb_plan(character: str, profile: str = "electronic", crossovers=XO, makeup: str = "peak") -> MultibandPlan:
    """planner.plan_multiband_record -> multiband_unrealised, without an analysis."""
    t = timing(character, profile)
    prof = PROFILES[profile]
    nb = len(crossovers) + 1
    zero = [[Envelope.constant(0.0), Envelope.constant(0.0)] for _ in range(nb)]
    return MultibandPlan(gr_db=zero, crossovers_hz=list(crossovers), mid_makeup_db=[0.0] * nb,
                         side_makeup_db=prof["side_makeup_db"], decoherence=[0.0] * nb,
                         ceiling_gr_db=None, ceiling_db=None,
                         threshold_db=[[0.0, 0.0] for _ in range(nb)], ratio=planner.MB_RATIO,
                         attack_ms=[[a, a] for a in t["mb_attack_ms"]],
                         release_ms=[[r, r] for r in t["mb_release_ms"]],
                         knee_db=t["mb_knee_db"], ceiling_lookahead_ms=t["band_lookahead_ms"],
                         budget={**mb.default_budget(makeup), **prof["mb_budget"]},
                         ms_link=prof["ms_link"], **new_mb_fields(crossovers, nb))


def new_mb_fields(crossovers, nb) -> dict:
    """plan_multiband_record's 2026-11-10 fields when the src has them (else nothing)."""
    out = {}
    if hasattr(mb, "default_detector_window_ms"):
        out["detector_window_ms"] = mb.default_detector_window_ms(list(crossovers))
    if hasattr(planner, "CEILING_RELEASE_MS"):
        out["ceiling_release_ms"] = [[planner.CEILING_RELEASE_MS[min(k, len(planner.CEILING_RELEASE_MS) - 1)]] * 2
                                     for k in range(nb)]
    if hasattr(planner, "xover_phases"):
        out["phase"] = planner.xover_phases("minimum", list(crossovers))
    return out


def mb_realise(x, fs, plan: MultibandPlan, hits) -> MultibandPlan:
    return mb.design_from(plan, x, fs, hits=hits)


def mb_total_gain_db(plan: MultibandPlan, k: int, c: int, n: int, fs: float) -> np.ndarray:
    """The gain band k channel c gets in multiband._stream (dB, audio rate)."""
    makeup = plan.makeup()
    if plan.ms_link:
        red = []
        for cc in range(2):
            r = np.clip(plan.gr_db[k][cc].at(n, fs), -MULTIBAND_MAX_GR_DB, 0.0)
            if plan.ceiling_gr_db is not None:
                r += plan.ceiling_gr_db[k][cc].at(n, fs)
            red.append(r)
        total = np.minimum(red[0], red[1])
        if plan.mode_makeup_db is not None:
            total += plan.mode_makeup_db[k][0].at(n, fs)
        return total + makeup[k][0]
    total = np.clip(plan.gr_db[k][c].at(n, fs), -MULTIBAND_MAX_GR_DB, 0.0)
    if plan.ceiling_gr_db is not None:
        total += plan.ceiling_gr_db[k][c].at(n, fs)
    if plan.mode_makeup_db is not None:
        total += plan.mode_makeup_db[k][c].at(n, fs)
    return total + makeup[k][c]


def mb_comp_gain_db(plan, k, c, n, fs):
    return np.clip(plan.gr_db[k][c].at(n, fs), -MULTIBAND_MAX_GR_DB, 0.0)


def mb_ceil_gain_db(plan, k, c, n, fs):
    return np.zeros(n) if plan.ceiling_gr_db is None else plan.ceiling_gr_db[k][c].at(n, fs)


def ms_bands(x, fs, crossovers=XO):
    """(bands, n, 2) M/S zero-phase split."""
    return bands.split(bands.to_ms(np.asarray(x, dtype=np.float64)), fs, list(crossovers))


def ceiling_only_plan(x, fs, cut_db: float, band: int, lookahead_ms, release_ms,
                      crossovers=XO, channel: int = 0) -> MultibandPlan:
    """A ceiling-only multiband (compressor off) cutting band `band` channel M by cut_db
    below its sample peak: the band ceiling / speclim mechanism on a steady signal."""
    b = ms_bands(x, fs, crossovers)
    nb = b.shape[0]
    pk = float(np.max(np.abs(b[band, :, channel])))
    ceil = [[None, None] for _ in range(nb)]
    ceil[band][channel] = 20.0 * math.log10(pk) - cut_db
    kw = {}
    if hasattr(mb, "default_detector_window_ms"):
        kw["detector_window_ms"] = mb.default_detector_window_ms(list(crossovers))
    return mb.design(x, fs, mb.SPECLIM_OFF_THRESHOLD_DB, ratio=2.0, crossovers_hz=list(crossovers),
                     ceiling_db=ceil, ceiling_lookahead_ms=lookahead_ms,
                     ceiling_release_ms=release_ms, mid_makeup_db=[0.0] * nb, side_makeup_db=0.0,
                     decoherence=[0.0] * nb, **kw)


# ------------------------------------------------------------------ glue

def glue_plan(x, fs, character: str, hits, profile: str = "electronic", mode: str = "serial",
              attack_override_ms: float | None = None) -> GluePlan:
    t = timing(character, profile)
    prof = PROFILES[profile]
    net = gl.SERIAL_GR_DB if prof["glue_net_db"] is None else prof["glue_net_db"]
    att = t["glue_attack_ms"] if attack_override_ms is None else attack_override_ms
    return gl.design_budget(x, fs, mode, ratio=planner.GLUE_RATIO, knee_db=planner.GLUE_KNEE_DB,
                            attack_ms=att, release_ms=t["glue_release_ms"], hits=hits, net_db=net)


def glue_gain_db(plan: GluePlan, n: int, fs: float) -> np.ndarray:
    return gl.gain_trace_db(plan, n, fs)


# ------------------------------------------------------------------ speclim

def speclim_plan(y, fs, character: str, hits, profile: str = "electronic", crossovers=XO):
    """planner.design_speclim without a Plan."""
    t = timing(character, profile)
    prof = PROFILES[profile]
    nb = len(crossovers) + 1
    rel = [planner.SPECLIM_RELEASE_MS[min(k, len(planner.SPECLIM_RELEASE_MS) - 1)] for k in range(nb)]
    edges = [0.0] + list(crossovers)
    allow = [max(prof["speclim_allow_db"], planner.SPECLIM_TOP_ALLOW_DB)
             if edges[k] >= planner.SPECLIM_TOP_FROM_HZ else prof["speclim_allow_db"] for k in range(nb)]
    kw = {}
    if hasattr(mb, "default_detector_window_ms"):
        kw["detector_window_ms"] = mb.default_detector_window_ms(list(crossovers))
    if hasattr(planner, "xover_phases"):
        kw["phase"] = planner.xover_phases("minimum", list(crossovers))
    return mb.design_speclim(y, fs, list(crossovers), allow, prof["speclim_max_cut_db"],
                             t["speclim_lookahead_ms"], rel, hits=hits, **kw)


# ------------------------------------------------------------------ limiter

def limiter_plan(character: str, profile: str = "electronic", input_gain_db: float = 0.0,
                 clip: bool = True, release_raw: bool = False, split_hz: float | None = 100.0,
                 budget: bool = True, knee_override: float | None = None,
                 sample_cap: bool = True, attack_override_ms: float | None = None,
                 release_override_ms: float | None = None) -> LimiterPlan:
    t = timing(character, profile)
    prof = PROFILES[profile]
    rel = t["limiter_release_raw_ms"] if release_raw else t["limiter_release_ms"]
    if release_override_ms is not None:
        rel = release_override_ms
    att = t["limiter_attack_ms"] if attack_override_ms is None else attack_override_ms
    kw = {}
    if clip:
        kw = dict(clip_target_db=prof["clip_target_db"],
                  clip_max_gr_db=prof["clip_sample_cap_db"] if sample_cap else None,
                  clip_split_hz=split_hz,
                  clip_knee_db=t["clip_knee_db"] if knee_override is None else knee_override)
    if budget:
        kw.update(max_gr_db=prof["limiter_max_gr_db"], max_outlier_gr_db=prof["limiter_outlier_cap_db"])
    if hasattr(planner, "LIMITER_HOLD_MS"):
        kw["hold_ms"] = planner.LIMITER_HOLD_MS
    if hasattr(planner, "LIMITER_AIM_MARGIN_DB"):
        kw["aim_margin_db"] = planner.LIMITER_AIM_MARGIN_DB
    return LimiterPlan(ceiling_dbtp=planner.CEILING_DBTP, input_gain_db=input_gain_db,
                       attack_ms=att, release_ms=rel, requested_gain_db=input_gain_db, **kw)


def limiter_solve(y, fs, lp: LimiterPlan, hits, requested_db: float) -> LimiterPlan:
    """The planner's design_limiter: clipper threshold over hits, then the gain clamp."""
    return planner.design_limiter(y, fs, lp, requested_db=requested_db, hits=hits)


def clip_threshold_direct(y, fs, lp: LimiterPlan, target_db: float = 1.0) -> LimiterPlan:
    """Clipper threshold from the curve on the 8x peak of the part the clipper acts on."""
    src = y if lp.clip_split_hz is None else lim.clip_hi_part(y, fs, lp.clip_split_hz)
    sol = lim.clip_threshold_for_peak_gr(src, target_db, lp.clip_knee_db)
    return replace(lp, clip_threshold_db=sol["threshold_db"])


def limiter_run(x, fs, plan: LimiterPlan, smooth=None, keep8: bool = True) -> dict:
    """Fork of limiter.apply_with_info that also returns the 8x gain and signals.
    smooth(g_req, attack, release) -> g lets a prototype smoother replace box_smooth."""
    x = np.asarray(x, dtype=np.float64)
    x2 = x * 10.0 ** (plan.input_gain_db / 20.0)
    ceiling = 10.0 ** (plan.ceiling_dbtp / 20.0)
    fs_up = fs * lim.OVERSAMPLE
    u = resample.upsample(x2, lim.OVERSAMPLE)
    u_pre = u.copy() if keep8 else None
    u_hi_keep = None
    if plan.clip_threshold_db is not None:
        u_hi = None
        if plan.clip_split_hz is not None:
            u_hi = resample.upsample(lim.clip_hi_part(x2, fs, plan.clip_split_hz), lim.OVERSAMPLE)
        uc, clip_db, spans = lim.clip_whole(u, plan, u_hi)
        u_hi_keep = u_hi if keep8 else None
        del u_hi
    else:
        uc, clip_db, spans = lim.clip_onsets(u, fs_up, plan, ceiling)
    u_bl = lim.band_limit_change(u, uc, spans)
    uc_keep = uc if keep8 else None
    del uc, u
    det = np.max(np.abs(u_bl), axis=1)
    det_peak = float(20.0 * np.log10(det.max())) if det.max() > 0 else -math.inf
    new = hasattr(lim, "relimit")
    g_req = envelope.required_gain(det, lim.aim(ceiling, plan) if new else ceiling)
    attack = max(lim.ATTACK_FLOOR, int(round(plan.attack_ms * 1e-3 * fs_up)))
    release = plan.release_ms * 1e-3 * fs_up
    if smooth is None and new:
        g = envelope.box_smooth(g_req, attack=attack, release=release,
                                hold=lim.hold_samples(plan, fs_up))
    elif smooth is None:
        g = envelope.box_smooth(g_req, attack=attack, release=release)
    else:
        g = smooth(g_req, attack, release)
    y = resample.decimate(u_bl * g[:, None], lim.OVERSAMPLE)
    gr8 = float(-20.0 * np.log10(np.min(g)))
    if new:
        y, rinfo = lim.relimit(y, fs, plan, ceiling)
        info = {"clip_max_reduction_db": clip_db, "clip_spans": len(spans), "max_gain_reduction_db": gr8,
                "detector_peak_dbfs": det_peak, **rinfo}
        return {"y": y, "info": info, "g8": g, "g2": None, "det": det, "u_pre": u_pre,
                "uc": uc_keep, "u_hi": u_hi_keep, "u_bl": u_bl if keep8 else None, "spans": spans,
                "fs8": fs_up, "attack8": attack, "release8": release, "plan": plan}
    e8, e32, e4 = lim._peak_envelopes(y, fs)
    e = np.maximum(np.maximum(e8, e32), e4)
    passes = []
    target = ceiling * 10.0 ** (-lim.RELIMIT_MARGIN_DB / 20.0)
    a_base = max(lim.ATTACK_FLOOR, int(round(plan.attack_ms * 1e-3 * fs)))
    r_base = plan.release_ms * 1e-3 * fs
    g2_total = np.ones(y.shape[0])
    for _ in range(lim.RELIMIT_PASSES):
        if e.size == 0 or e.max() <= ceiling:
            break
        g2 = envelope.box_smooth(envelope.required_gain(e, target), attack=a_base, release=r_base)
        y = y * g2[:, None]
        g2_total *= g2
        passes.append({"samples": int(np.count_nonzero(g2 < 1.0)),
                       "max_reduction_db": float(-20.0 * np.log10(g2.min()))})
        e8, e32, e4 = lim._peak_envelopes(y, fs)
        e = np.maximum(np.maximum(e8, e32), e4)
    trim_db = 0.0
    worst = max(lim._db(e8.max(initial=0.0)), lim._db(e32.max(initial=0.0)), lim._db(e4.max(initial=0.0)))
    if worst > plan.ceiling_dbtp:
        trim_db = plan.ceiling_dbtp - worst
        y = y * 10.0 ** (trim_db / 20.0)
    info = {"clip_max_reduction_db": clip_db, "clip_spans": len(spans), "max_gain_reduction_db": gr8,
            "detector_peak_dbfs": det_peak, "relimit_passes": passes, "trim_db": trim_db,
            "true_peak_8x_dbtp": lim._db(e8.max(initial=0.0)) + trim_db,
            "true_peak_32x_dbtp": lim._db(e32.max(initial=0.0)) + trim_db}
    return {"y": y, "info": info, "g8": g, "g2": g2_total, "det": det, "u_pre": u_pre,
            "uc": uc_keep, "u_hi": u_hi_keep, "u_bl": u_bl if keep8 else None, "spans": spans,
            "fs8": fs_up, "attack8": attack, "release8": release, "plan": plan}


def apply_gain_at_8x(tone, fs, g8):
    """tone (n, 2) through an 8x gain curve (upsample, multiply, decimate)."""
    u = resample.upsample(np.asarray(tone, dtype=np.float64), lim.OVERSAMPLE)
    m = min(u.shape[0], g8.size)
    return resample.decimate(u[:m] * g8[:m, None], lim.OVERSAMPLE)


def rms_meter_db(y, fs) -> float:
    return float(meter.rms_meter(y, fs)["loud_dbfs"])


def crest_db(y, fs) -> float:
    return float(20 * np.log10(np.max(np.abs(y))) - meter.rms_1s(y, fs)["loud_section_dbfs"])
