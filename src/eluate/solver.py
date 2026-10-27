"""Constraint solver and blocked limiter (spec 1, 3.6).

Contract:
- `limit_blocked(x, fs, plan, block=BLOCK) -> (y, info)`: the final stage of
  spec 3.5 item 7 (stages/limiter.py) computed in base-rate blocks, so no
  whole-track 8x array exists (spec 1). Each block of `block` base samples is
  processed over an extended region that covers every dependency of its
  output: the 8x interpolation filter (resample.py), the clip spans whose
  band-limited change reaches the needed range (with the same global span
  bookkeeping and base-grid alignment as limiter.band_limit_change), the box
  smoother's lookahead (+-2 (L - 1) samples at 8x) and the decimation
  filter. The release hull is causal and recursive: its state is carried
  exactly from the previous block (the hull value just before the region is
  prepended as one synthetic sample, which reproduces the recursion
  h[n] = min(u[n], a h[n-1]) in dB). Then the local re-limit and the
  last-resort trim run on the decimated base-rate output as in limiter.py
  (8x true-peak envelope and 32x FFT envelope, both base-rate arrays).
  Agreement with limiter.apply_with_info: equal up to FFT-convolution and
  log-domain rounding (tested: max abs difference < 1e-9 on synthetic
  signals with clipping and re-limiting). info has the same keys.
  Threads (sprint WP8, eluate.parallel): each block's clipper, band-limit and
  detector, and after the hull its smoother, gain and decimation, run on
  parallel.pmap, parallel.workers(BLOCK_CAP) blocks at a time; the hull runs block by
  block in order between the two, and the whole-signal clipper's spans are found the
  same way. The output is bit-identical at any thread count (tests/test_parallel.py).
  Memory: the 8x arrays are per block (block + margins, about 16 MB per
  array at the default block; 40 to 50 MB per block in flight, at most
  parallel.BLOCK_CAP at once); the peak is set by the base-rate re-limit
  pass (8x true-peak envelope, 32x FFT envelope, box smoother on whole-track
  base-rate arrays). Measured before the memory pass at about 180 bytes per
  input sample against about 970 for limiter.apply_with_info (tracemalloc,
  48 s at 44.1 kHz); see brief-07 revision 3 for the current figure.
- `limit_proxy(z, fs, plan, drive_db)`: a base-rate model of the limiter
  (the clipper at base rate, sample-peak detector, same box smoother) used
  inside the solver and the planner to solve the limiter's gain cheaply;
  `proxy_rms` reads the loudness target's reading (meter.rms_meter) and `gain_for_rms` brackets
  the gain (brentq) at which that reading plus a bias equals a target; with `info` (a dict)
  it sets info["at_limit"] when no bracket was found and the gain returned is an end of
  DRIVE_RANGE_DB (S5: a failed solve, flagged, not a gain). The
  solver calibrates the bias (real render minus proxy) after every real render.
- `clip_threshold_for_output_peak(y, fs, plan, target_db)`: the whole-signal
  clipper's threshold for a target fall of the output peak (the 8x peak after
  the curve and its band-limit, read by `limit_blocked(measure_only=True)`),
  maintainer 2026-10-18 (docs/gain-staging.md 14).
- `render_plan(x, fs, plan, pre=None)`: the whole chain (stages.render
  without its limiter and output stages) followed by `limit_blocked` and the
  output stage (`finish`: the export quantisation of plan.output, so the
  master returned is the file's samples). This is the render the CLI and the
  solver use; it differs from stages.render only by the blocked limiter.
  `render_pre` (cached by the solver) is the input gain, input EQ, notch and
  sidestrip.
- `solve(x, fs, plan0, targets, ...) -> SolveResult` (docs/gain-staging.md 14,
  chunk 4). Every dynamics stage is solved for its peak-GR budget on its real
  input (multiband, saturate, glue, clipper: `Chain.pre_limiter`, cached per
  tone rotation, with the stage 1-2 output and every cached array spilled to
  disk). The solver's variables (`Levers`, written by `apply_levers`) are the
  limiter's input gain, the tone rotation and the limiter ceiling. Each render:
  the gain is solved for the loudness target (targets.rms_dbfs, the master's RMS on
  the maintainer's meter: meter.rms_meter, 300 ms, AES17, loudest passage) on the proxy
  limiter with the
  calibrated bias (`gain_for_rms`), then clamped to the limiter's peak-GR budget
  (planner.clamp_limiter_gain: budgets win, with a warning), the blocked
  limiter and the output stage run, and `measure` reads the constraints.
  Constraints (`measure`): RMS within +-targets.rms_tol_db (0.3 dB) of the
  target; stereo slope within
  +-0.3 dB/oct of reference + tilt (reshape) or the
  stereo LTAS within +-1 dB of the premaster's shape per 1/3-octave band over
  30 Hz to 16 kHz after removing the mean level difference, bands gated
  relative to the premaster's LTAS maximum (60 dB) and the same bands used for
  both signals (preserve); per-band crest through multiband.crest_loss (shared
  frames, silenced rule) under targets.crest_mode: "loss" = loss within 1.5 /
  2 / 3 / 3 / 3 dB, "floor" = after-p95 at or above targets.crest_floors_db,
  "report" (default) = reported, never failing; the sidestripped S views
  exempt only when targets.crest_exempt is set; true peak <=
  targets.true_peak_max_dbtp on targets.tp_meter ("worst" = max of the 4x
  Annex 2, 8x and 32x readings, "8x" = max of 8x and 32x), every constraint
  measured on the master as written (after plan.output's quantisation and
  dither, then the float32 cast, exact for PCM); the crest reference is the
  premaster after the plan's input gain; under preserve, each section's
  loudness relative to the premaster's loudest section within +-1 LU of the
  premaster's (a counted section lost in the render reads SECTION_LOST_LU, a finite
  penalty, S8). S95 (and the RMS with the AES17 offset) are reported, not
  targets. Per-stage GR is reported against its budget (planner.budget_report).
  Section contrast (sprint 2026-10-21 D2, R13), in both modes: each counted section's
  level against the loudest, change from the premaster's (section_delta_change_lu; > 0 =
  the section rose toward the loudest, less contrast), and section_contrast_loss_lu = the
  largest rise (0 when none rose); `section_change_by_stage` the same change after every
  stage of the chain (input EQ / notch / sidestrip, multiband, tone, saturation, glue,
  frequency limiter, then the limiter and output on the master), so the loss can be
  traced to the stage that made it. One threshold, evaluate.FLAG_SECTION_LU (3 LU), is
  shared with the report's section flag, ui-data and the process log (S9).
  Method: render 1 is the planned state. When the constrained true-peak reading
  is over, a render with the limiter ceiling lowered by the overshoot comes
  next and is the baseline. Slope first: while the slope constraint is violated
  (reshape mode, input EQ present), up to SLOPE_STEPS renders add the slope
  error to the tone rotation (planner.refit_inputeq), each kept only if
  `lever_score` (the summed normalised excess without the RMS, true-peak and
  section-contrast terms: the slope is primary, R7a) falls. Then up to LOUDNESS_STEPS
  renders while the RMS is outside target +- tolerance with the limiter's budget not
  clamping, or the true peak reads over, each re-solving the gain on the recalibrated
  proxy (and the lowered ceiling), kept when the full `score` does not rise and the
  render does not flatten the sections (`judge_move`, `flattens`, R13): a move is reverted, with
  reason "section contrast", when the kept render's loudness is already accepted (ok)
  and the move leaves section_contrast_loss_lu past FLAG_SECTION_LU and more than
  CONTRAST_SLACK_LU above the kept render's. `score` = the summed normalised constraint
  excess plus the contrast term max(0, loss - FLAG_SECTION_LU) / FLAG_SECTION_LU, so
  among renders that differ in nothing else the flatter one scores worse. At most
  max_outer renders in all
  for the rotation and loudness passes (`render_bound(max_outer)`: the most renders a
  solve can make, the true-peak ceiling render counted). `progress(n)`, when given, is
  called as render n starts (the desktop app's progress line; nothing else depends on
  it). History logs every render, kept and
  reverted move. The result's `misses` are the constraints the kept render
  misses (`misses`), `warnings` the budget warnings (planner.budget_report; a
  clamped limiter's warning carries the measured loudness shortfall_db and, S5, needed_db
  / outlier_needed_db derived from it: the GR at the clamp plus the shortfall, a lower
  bound, needed_source "render") plus a {"stage": "loudness", "target_dbfs",
  "achieved_dbfs", "shortfall_db", "cause"} warning when the RMS constraint is not ok,
  plus a {"stage": "inputeq", "kind": "slope", "at_bound", "tilt_db", ...} warning when
  the slope is missed (T1: the sections at their bound and the Tilt that would make the
  reached slope the target); `ok` when there is no miss and no warning, a limiter clamp
  warning excepted when the RMS is within its tolerance (that warning then carries
  accepted True). A tone-rotation step whose refit cannot move the predicted slope by
  ROT_SLOPE_MIN is skipped ("EQ at its limit" in the history, S7).
- Memory (spec 1, brief-07 revision 3): the Chain spills its cached stage
  outputs and the best render to .npy files and loads them when needed; the
  input gain of the blocked limiter is applied per block; the solver holds at
  most the current render's master and pre-limiter audio. With the streamed
  band split, resampler, envelope evaluation and true-peak envelope in the
  modules it calls, every step's arrays are bit-identical to the whole-array
  forms (tests/test_memory_pass.py). Peak working set is read from the OS
  (`peak_memory_mb`). `sweep_spill()` removes spill dirs a killed engine left in the
  temp dir, only those untouched for SPILL_STALE_S (6 h), so a concurrent render's
  live dir is never removed.
"""

from __future__ import annotations

import copy
import ctypes
import math
import sys
import time
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.optimize import brentq

from eluate import envelope, meter, parallel, planner
from eluate.evaluate import FLAG_SECTION_IGNORE_LU, FLAG_SECTION_LU
from eluate.features import tilt as tilt_feature
from eluate.plan import CLIP_MAX_GR_DB, MULTIBAND_MAX_GR_DB, LimiterPlan, Plan
from eluate.stages import gain as gain_stage
from eluate.stages import glue as glue_stage
from eluate.stages import inputeq as eq_stage
from eluate.stages import limiter as lim
from eluate.stages import multiband as mb_stage
from eluate.stages import notch as notch_stage
from eluate.stages import output as output_stage
from eluate.stages import resample
from eluate.stages import saturate as sat_stage
from eluate.stages import sidestrip as side_stage
from eluate.stages import tone as tone_stage

BLOCK = 1 << 17
MAX_OUTER = 8
DRIVE_RANGE_DB = (-24.0, 36.0)
SECTION_IGNORE_LU = FLAG_SECTION_IGNORE_LU    # one gate with the report (S9)
SECTION_LOST_LU = FLAG_SECTION_IGNORE_LU      # a counted section lost in the render (S8)
CONTRAST_SLACK_LU = 0.1       # contrast change a move may make before it "flattens" (R13)
LTAS_RANGE_HZ = (30.0, 16000.0)

TONE_ROT_MAX = 3.0            # dB/oct of extra rotation the slope pass may ask for
SLOPE_STEPS = 2               # tone-rotation renders
ROT_SLOPE_MIN = 0.05          # dB/oct: a refit moving the predicted slope less is skipped (S7)
LOUDNESS_STEPS = 3            # loudness / true-peak re-renders after the slope pass
TP_MARGIN_DB = 0.01           # extra ceiling drop when the constrained meter reads over
LTAS_GATE_DB = 60.0           # preserve-mode LTAS bands within this of the premaster's maximum
CLIP_SEARCH_DB = 12.0         # lowest clipper threshold tried, below the 8x peak
CLIP_SEARCH_STEP_DB = 1.0
CLIP_SOLVE_XTOL_DB = 0.005

# ------------------------------------------------------------------ memory

def peak_memory_mb() -> float:
    """Peak resident memory of this process, MB (Windows PeakWorkingSetSize, else ru_maxrss)."""
    if sys.platform == "win32":
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC),
                                               ctypes.c_ulong]
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return math.nan
        return pmc.PeakWorkingSetSize / 2 ** 20
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 20 if sys.platform == "darwin" else r / 1024.0


# ------------------------------------------------------------------ blocked limiter

def _hull_carry(u: np.ndarray, release: float, carry_db: float | None) -> np.ndarray:
    """envelope.release_hull continued from a previous hull value (dB depth >= 0)."""
    if release < 0.01 or carry_db is None or carry_db <= 0.0:
        return envelope.release_hull(u, release)
    u0 = 10.0 ** (-carry_db / 20.0)
    return envelope.release_hull(np.concatenate([[u0], u]), release)[1:]


def _box_tail(h: np.ndarray, L: int) -> np.ndarray:
    """Steps 2-3 of envelope.box_smooth on a hull (no clip)."""
    if L == 1:
        return h.copy()
    m = minimum_filter1d(h, size=2 * L - 1, mode="nearest")
    return envelope._box(envelope._box(m, -(L - 1), 0), 0, L - 1)


def _clip_span(seg: np.ndarray, w0: int, w1: int, a0: int, ramp: int, plan: LimiterPlan,
               ceiling: float):
    """limiter.clip_onsets for one merged span; returns (clipped, reduction_db) or None."""
    p = float(np.max(np.abs(seg)))
    if p <= ceiling:
        return None
    r_db = min(plan.clip_depth_db, 20.0 * math.log10(p / ceiling))
    t = max(ceiling, p * 10.0 ** (-r_db / 20.0))
    floor = ceiling * 10.0 ** (-lim.KNEE_BELOW_DB * (plan.clip_depth_db / 3.0) / 20.0)
    kw = max(0.0, min(p - t, t - floor))
    idx = np.arange(a0, a0 + seg.shape[0])
    wgt = np.ones(seg.shape[0])
    if ramp > 0:
        dist = np.maximum((w0 - idx) / (ramp + 1.0), (idx - (w1 - 1)) / (ramp + 1.0))
        wgt = np.where(dist <= 0, 1.0, 0.5 * (1.0 + np.cos(np.pi * np.minimum(dist, 1.0))))
    return seg + wgt[:, None] * (lim._knee_clip(seg, t, kw) - seg), r_db


def _whole_spans_blocked(x2: np.ndarray, gain: float, plan: LimiterPlan, block: int,
                         hb: int):
    """The whole-signal clipper's spans (limiter.clip_whole) found per block: each block
    upsampled with hb base samples of context, its runs above the knee start collected
    in global 8x indices, then merged as clip_whole merges them."""
    L = lim.OVERSAMPLE
    n = x2.shape[0]
    t, w, _ = lim.whole_threshold(plan)

    def block_runs(c0):
        c1 = min(n, c0 + block)
        r0, r1 = max(0, c0 - hb), min(n, c1 + hb)
        u = resample.upsample(x2[r0:r1] * gain, L)
        a = np.max(np.abs(u[L * (c0 - r0):L * (c1 - r0)]), axis=1)
        del u
        return lim.whole_runs(a, t - w, L * c0)

    runs = [r for rs in parallel.pmap(block_runs, range(0, n, block), parallel.BLOCK_CAP)
            for r in rs]
    return lim.merge_runs(lim.merge_runs(runs, 0), lim.SPAN_GAP)


def limit_blocked(x: np.ndarray, fs: float, plan: LimiterPlan, block: int = BLOCK,
                  measure_only: bool = False, detector_env: bool = False,
                  x_hi: np.ndarray | None = None):
    """Blocked version of limiter.apply_with_info (see module contract). measure_only:
    stop after the clipper and band-limit and return only {"detector_peak_dbfs",
    "clip_max_reduction_db", "clip_spans"} (the limiter's input, for its budget), and,
    with detector_env, "detector_env": per base sample, the maximum of the 8x detector
    over its 8 subsamples (linear), for the per-hit readings. x_hi: the part of x above
    plan.clip_split_hz (limiter.clip_hi_part), computed here when the plan splits and
    it is not given; a caller that measures many thresholds passes it once."""
    if plan.smoother != "box":
        raise ValueError(f"limit_blocked implements the box smoother only, not {plan.smoother!r}"
                         " (audit 07 A1)")
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    x2 = x[:, None] if squeeze else x
    gain = 10.0 ** (plan.input_gain_db / 20.0)   # applied per block: same values, no copy
    n, nch = x2.shape
    split = plan.clip_threshold_db is not None and plan.clip_split_hz is not None
    if split:
        if x_hi is None:
            x_hi = lim.clip_hi_part(x2, fs, plan.clip_split_hz)
        x_hi = np.asarray(x_hi, dtype=np.float64)
        x_hi = x_hi[:, None] if x_hi.ndim == 1 else x_hi
        if x_hi.shape != x2.shape:
            raise ValueError("x_hi must have the shape of x")
    else:
        x_hi = None
    L = lim.OVERSAMPLE
    n8 = n * L
    fs_up = fs * L
    ceiling = 10.0 ** (plan.ceiling_dbtp / 20.0)
    h = resample.prototype(L)
    c8 = (h.size - 1) // 2                       # filter half-length, 8x samples
    hb = c8 // L + 2                             # in base samples, with slack
    attack = max(lim.ATTACK_FLOOR, int(round(plan.attack_ms * 1e-3 * fs_up)))
    release = plan.release_ms * 1e-3 * fs_up
    Lb = max(1, attack // 2 + 1)
    sup = 2 * (Lb - 1)                           # box smoother support, 8x samples
    m_bl = lim.BL_MARGIN * L
    spans, ramp = ([], 0)
    whole = plan.clip_threshold_db is not None
    if whole:
        spans = [(a0, a1, a0, a1) for a0, a1 in
                 _whole_spans_blocked(x2 if x_hi is None else x_hi, gain, plan, block, hb)]
        t_w = lim.whole_threshold(plan)
    elif plan.clip_depth_db > 0.0 and plan.clip_onsets_s:
        spans, ramp = lim._spans(plan, fs_up, n8)
    infl = []                                     # band-limit influence range per span
    for w0, w1, a0, a1 in spans:
        s0 = max(0, ((a0 - m_bl) // L) * L)
        s1 = min(n8, a1 + m_bl)
        s1 = s0 + ((s1 - s0) // L) * L
        infl.append((s0, s1))
    span_lo = np.array([s for s, _ in infl], dtype=np.int64)
    span_hi = np.array([e for _, e in infl], dtype=np.int64)

    y = None if measure_only else np.zeros((n, nch))
    denv = np.zeros(n) if measure_only and detector_env else None

    def front(c0):
        """Clipper, band-limit and detector of one block: independent of every other."""
        c1 = min(n, c0 + block)
        p0, p1 = max(0, (c0 - hb) * L), min(n8, (c1 + hb) * L)      # decimation input
        g0, g1 = max(0, p0 - sup), min(n8, p1 + sup)                # gain needed here
        sel = np.flatnonzero((span_hi > g0) & (span_lo < g1)) if spans else np.zeros(0, int)
        r8_lo = min([g0] + [int(span_lo[i]) for i in sel])
        r8_hi = max([g1] + [int(span_hi[i]) for i in sel])
        r0 = max(0, r8_lo // L - hb)
        r1 = min(n, -(-r8_hi // L) + hb)
        u = resample.upsample(x2[r0:r1] * gain, L)                  # 8x indices [L r0, L r1)
        off = L * r0
        ubl = u.copy()
        u_hi = (resample.upsample(x_hi[r0:r1] * gain, L)
                if x_hi is not None and len(sel) else None)
        clips = []                                # reductions of the spans this block reports
        for i in sel:
            w0, w1, a0, a1 = spans[i]
            if whole:
                src = u if u_hi is None else u_hi
                seg = src[a0 - off:a1 - off]
                seg_c = lim._knee_clip(seg, *t_w)
                if u_hi is not None:
                    seg_c = u[a0 - off:a1 - off] + (seg_c - seg)
                r_db = lim.clip_reduction_db(float(np.max(np.abs(seg))), *t_w)
            else:
                res = _clip_span(u[a0 - off:a1 - off], w0, w1, a0, ramp, plan, ceiling)
                if res is None:
                    continue
                seg_c, r_db = res
            s0, s1 = infl[i]
            d = np.zeros((s1 - s0, nch))
            d[a0 - s0:a1 - s0] = seg_c - u[a0 - off:a1 - off]
            ubl[s0 - off:s1 - off] += resample.upsample(resample.decimate(d, L), L)
            if c0 <= w0 // L < c1:                # report each span once
                clips.append(r_db)
        del u, u_hi
        det = np.max(np.abs(ubl[g0 - off:g1 - off]), axis=1)
        f = {"c0": c0, "c1": c1, "p0": p0, "p1": p1, "g0": g0, "clips": clips,
             "det_max": float(det.max(initial=0.0))}
        if denv is not None:
            denv[c0:c1] = det[L * c0 - g0:L * c1 - g0].reshape(c1 - c0, L).max(axis=1)
        if not measure_only:
            f["uu"] = np.clip(envelope.required_gain(det, ceiling), envelope.G_MIN, 1.0)
            f["v"] = ubl[p0 - off:p1 - off]
        return f

    def back(f):
        """Box smoother, gain and decimation of one block, given its hull; the block's
        minimum gain."""
        c0, c1, p0, p1, g0 = f["c0"], f["c1"], f["p0"], f["p1"], f["g0"]
        g = np.clip(_box_tail(f.pop("hh"), Lb), envelope.G_MIN, f.pop("uu"))
        k0, k1 = L * c0 - g0, L * c1 - g0
        gmin = float(g[k0:k1].min())
        v = f.pop("v") * g[p0 - g0:p1 - g0, None]
        del g
        z = resample.decimate(v, L)                                  # base [p0/L, p1/L)
        y[c0:c1] = z[c0 - p0 // L:c1 - p0 // L]
        return gmin

    det_max = 0.0
    gmin_all = 1.0
    clip_max = 0.0
    clip_count = 0
    carry_next = None                             # (8x index, hull depth dB) for the next block
    starts = list(range(0, n, block))
    wave = parallel.workers(parallel.BLOCK_CAP)   # blocks in flight: bounds the 8x memory
    for w in range(0, len(starts), wave):
        fronts = parallel.pmap(front, starts[w:w + wave], parallel.BLOCK_CAP)
        for f in fronts:
            det_max = max(det_max, f["det_max"])
            for r_db in f["clips"]:
                clip_max = max(clip_max, r_db)
                clip_count += 1
        if measure_only:
            continue
        for f in fronts:                          # the hull is recursive: block by block
            g0, c1 = f["g0"], f["c1"]
            carry = None
            if carry_next is not None and g0 > 0:
                if carry_next[0] != g0 - 1:
                    raise RuntimeError("hull carry index does not match the block start")
                carry = carry_next[1]
            hh = _hull_carry(f["uu"], release, carry)
            # the next block's region starts at g0' and needs the hull just before it:
            # carry that one scalar (audit 07 A3)
            if c1 < n:
                g0n = max(0, max(0, (c1 - hb) * L) - sup)
                j = g0n - 1 - g0
                if not 0 <= j < hh.size:
                    raise RuntimeError("hull carry index outside this block")
                with np.errstate(divide="ignore"):
                    carry_next = (g0n - 1, max(float(-20.0 * np.log10(hh[j])), 0.0))
            f["hh"] = hh
        gmin_all = min([gmin_all] + parallel.pmap(back, fronts, parallel.BLOCK_CAP))
        del fronts
    det_db = float(20.0 * math.log10(det_max)) if det_max > 0 else -math.inf
    if measure_only:
        out = {"detector_peak_dbfs": det_db, "clip_max_reduction_db": clip_max,
               "clip_spans": clip_count}
        if denv is not None:
            out["detector_env"] = denv
        return out
    gr8 = float(-20.0 * math.log10(gmin_all))
    y, rinfo = _relimit(y, fs, plan, ceiling)
    info = {"clip_max_reduction_db": clip_max, "clip_spans": clip_count,
            "max_gain_reduction_db": gr8, "detector_peak_dbfs": det_db, **rinfo, "qp": None,
            "blocked": True, "block": int(block)}
    return (y[:, 0] if squeeze else y), info


def limiter_peaks(z: np.ndarray, fs: float, plan: LimiterPlan, hits: dict | None = None,
                  block: int = BLOCK) -> dict:
    """The limiter's 8x detector peaks at 0 dB input gain on the clipper's output
    (limit_blocked measure_only): {"peak_dbfs" (the track's, the outlier reading),
    "average_peak_dbfs" (the median over the loud hits of each hit's peak; the track's
    without hits)}. The whole-signal clipper's threshold sits before the gain, so both
    move dB for dB with it."""
    m = limit_blocked(z, fs, replace(plan, input_gain_db=0.0), block, measure_only=True,
                      detector_env=hits is not None)
    p = m["detector_peak_dbfs"]
    avg = p
    if hits is not None and math.isfinite(p):
        pk = hit_peaks_db(m["detector_env"], fs, hits["edges_s"])
        avg = envelope.summarise_hits(pk, hits["loud"])["average_db"] if np.any(
            np.isfinite(pk)) else p
    return {"peak_dbfs": p, "average_peak_dbfs": avg}


def limiter_gain_max(p_max: float, p_avg: float, ceiling_dbtp: float, cap_db: float,
                     outlier_cap_db: float | None = None) -> float:
    """The largest input gain within the limiter's budget: an average peak GR of at most
    cap_db (p_avg + g - ceiling) and an outlier peak GR of at most outlier_cap_db
    (p_max + g - ceiling; default cap_db + envelope.OUTLIER_ALLOWANCE_DB, maintainer
    2026-10-19). The limiter's own GR at a peak P is max(0, P + g - ceiling) exactly
    (box_smooth keeps the required gain's minimum)."""
    ocap = cap_db + envelope.OUTLIER_ALLOWANCE_DB if outlier_cap_db is None else outlier_cap_db
    return min(ceiling_dbtp + cap_db - p_avg, ceiling_dbtp + ocap - p_max)


def limiter_bound(p_max: float, p_avg: float, ceiling_dbtp: float, cap_db: float,
                  outlier_cap_db: float | None = None) -> str:
    """Which reading sets limiter_gain_max: "average" or "outlier"."""
    ocap = cap_db + envelope.OUTLIER_ALLOWANCE_DB if outlier_cap_db is None else outlier_cap_db
    return "outlier" if ceiling_dbtp + ocap - p_max < ceiling_dbtp + cap_db - p_avg else "average"


def limiter_input_gain(z: np.ndarray, fs: float, plan: LimiterPlan,
                       requested_db: float | None = None, block: int = BLOCK,
                       hits: dict | None = None) -> dict:
    """The limiter's input gain within its budget plan.max_gr_db (docs/gain-staging.md 3,
    chunk 3; the average-peak reading, maintainer 2026-10-19), given the clipper's output:
    the detector peaks at 0 dB gain (limiter_peaks: the track's P and, over hits, the
    loud hits' median P_avg; without hits both are P) give g_max = limiter_gain_max.
    requested_db (default plan.input_gain_db) is the gain asked for (the loudness
    target's); it is clamped to g_max, with a warning {"stage": "limiter",
    "requested_db", "max_db", "needed_db" (average peak GR asked for), "cap_db",
    "outlier_needed_db", "outlier_cap_db"} when that binds. Returns {"input_gain_db",
    "max_gain_db", "requested_gain_db", "planned_gr_db" (average), "planned_outlier_gr_db",
    "clipped_peak_dbfs", "clipped_average_peak_dbfs", "warnings"}."""
    req = float(plan.input_gain_db if requested_db is None else requested_db)
    pk = limiter_peaks(z, fs, plan, hits, block)
    p, pa = pk["peak_dbfs"], pk["average_peak_dbfs"]
    cap = plan.max_gr_db
    if cap is None or not math.isfinite(p):
        fin = math.isfinite(p)
        return {"input_gain_db": req, "max_gain_db": None, "requested_gain_db": req,
                "planned_gr_db": max(0.0, pa + req - plan.ceiling_dbtp) if fin else 0.0,
                "planned_outlier_gr_db": max(0.0, p + req - plan.ceiling_dbtp) if fin else 0.0,
                "clipped_peak_dbfs": p if fin else None,
                "clipped_average_peak_dbfs": pa if fin else None, "warnings": []}
    ocap = (cap + envelope.OUTLIER_ALLOWANCE_DB if plan.max_outlier_gr_db is None
            else plan.max_outlier_gr_db)
    g_max = limiter_gain_max(p, pa, plan.ceiling_dbtp, cap, ocap)
    g = min(req, g_max)
    warn = []
    if req > g_max + 1e-9:
        warn.append({"stage": "limiter", "requested_db": req, "max_db": g_max,
                     "needed_db": pa + req - plan.ceiling_dbtp, "cap_db": cap,
                     "outlier_needed_db": p + req - plan.ceiling_dbtp,
                     "outlier_cap_db": ocap,
                     "bound": limiter_bound(p, pa, plan.ceiling_dbtp, cap, ocap)})
    return {"input_gain_db": g, "max_gain_db": g_max, "requested_gain_db": req,
            "planned_gr_db": max(0.0, pa + g - plan.ceiling_dbtp),
            "planned_outlier_gr_db": max(0.0, p + g - plan.ceiling_dbtp),
            "clipped_peak_dbfs": p, "clipped_average_peak_dbfs": pa, "warnings": warn}


def _relimit(y: np.ndarray, fs: float, plan: LimiterPlan, ceiling: float):
    """limiter.apply_with_info steps 6-7 on the decimated output."""
    e8, e32, e4 = lim._peak_envelopes(y, fs)
    tp8_0, tp32_0 = lim._db(e8.max(initial=0.0)), lim._db(e32.max(initial=0.0))
    e = np.maximum(np.maximum(e8, e32), e4)
    passes = []
    target = ceiling * 10.0 ** (-lim.RELIMIT_MARGIN_DB / 20.0)
    a_base = max(lim.ATTACK_FLOOR, int(round(plan.attack_ms * 1e-3 * fs)))
    r_base = plan.release_ms * 1e-3 * fs
    for _ in range(lim.RELIMIT_PASSES):
        if e.size == 0 or e.max() <= ceiling:
            break
        g2 = envelope.box_smooth(envelope.required_gain(e, target), attack=a_base,
                                 release=r_base)
        y = y * g2[:, None]
        passes.append({"samples": int(np.count_nonzero(g2 < 1.0)),
                       "max_reduction_db": float(-20.0 * np.log10(g2.min()))})
        e8, e32, e4 = lim._peak_envelopes(y, fs)
        e = np.maximum(np.maximum(e8, e32), e4)
    tp8, tp32 = lim._db(e8.max(initial=0.0)), lim._db(e32.max(initial=0.0))
    trim_db = 0.0
    worst = max(tp8, tp32, lim._db(e4.max(initial=0.0)))
    if worst > plan.ceiling_dbtp:
        trim_db = plan.ceiling_dbtp - worst
        y = y * 10.0 ** (trim_db / 20.0)
    return y, {"true_peak_8x_before_relimit_dbtp": tp8_0,
               "true_peak_32x_before_relimit_dbtp": tp32_0, "relimit_passes": passes,
               "true_peak_8x_dbtp": tp8 + trim_db, "true_peak_32x_dbtp": tp32 + trim_db,
               "trim_db": trim_db}


def clip_hi_for(z: np.ndarray, fs: float, plan: LimiterPlan) -> np.ndarray | None:
    """limiter.clip_hi_part of z when the plan's whole-signal clipper splits, else None."""
    if plan.clip_threshold_db is None or plan.clip_split_hz is None:
        return None
    return lim.clip_hi_part(z, fs, plan.clip_split_hz)


def limit_proxy(z: np.ndarray, fs: float, plan: LimiterPlan, drive_db: float,
                z_hi: np.ndarray | None = None) -> np.ndarray:
    """Base-rate limiter model for bracketing (see module contract). z_hi: clip_hi_for(z)
    when the plan splits (computed here if not given)."""
    g = 10.0 ** (drive_db / 20.0)
    x2 = np.asarray(z, dtype=np.float64) * g
    ceiling = 10.0 ** (plan.ceiling_dbtp / 20.0)
    if plan.clip_threshold_db is not None:
        t_w = lim.whole_threshold(replace(plan, input_gain_db=drive_db))
        if plan.clip_split_hz is None:
            x2 = lim._knee_clip(x2, *t_w)
        else:
            zh = (clip_hi_for(z, fs, plan) if z_hi is None else np.asarray(z_hi)) * g
            x2 = x2 + (lim._knee_clip(zh, *t_w) - zh)
    elif plan.clip_depth_db > 0.0 and plan.clip_onsets_s:
        x2 = lim.clip_onsets(x2, fs, plan, ceiling)[0]
    det = np.max(np.abs(x2), axis=1)
    a = max(lim.ATTACK_FLOOR, int(round(plan.attack_ms * 1e-3 * fs)))
    g = envelope.box_smooth(envelope.required_gain(det, ceiling), attack=a,
                            release=plan.release_ms * 1e-3 * fs)
    return x2 * g[:, None]


# ------------------------------------------------------------------ render

def render_pre(x: np.ndarray, fs: float, plan: Plan) -> np.ndarray:
    """Input gain, input EQ and stages 1-2 (input, inputeq, notch, sidestrip): depend only
    on the analysis and the tone rotation; cached by the solver. The gain comes first
    (docs/gain-staging.md 2): the EQ is linear and time-invariant, so the order does not
    change the audio, but it keeps the chain the maintainer's."""
    y = np.asarray(x, dtype=np.float64)
    if plan.input is not None:
        y = gain_stage.apply(y, fs, plan.input)
    if plan.inputeq is not None and not plan.preserve_mix:
        y = eq_stage.apply(y, fs, plan.inputeq)
    if plan.notch is not None:
        y = notch_stage.apply(y, fs, plan.notch)
    if plan.sidestrip is not None:
        y = side_stage.apply(y, fs, plan.sidestrip)
    return y


def render_chain(pre: np.ndarray, fs: float, plan: Plan, on_stage=None) -> np.ndarray:
    """Stages 3-6 and the frequency-dependent limiter of a realised plan on the stage 1-2
    output (keyeq must be disabled)."""
    if plan.keyeq is not None and plan.keyeq.enabled:
        raise NotImplementedError("keyeq is a stub")
    y = pre
    for name, fn in (("multiband", mb_stage.apply), ("tone", tone_stage.apply),
                     ("saturate", sat_stage.apply), ("glue", glue_stage.apply),
                     ("speclim", mb_stage.apply)):
        sec = getattr(plan, name)
        if sec is None or (name == "tone" and plan.preserve_mix):
            continue
        y = fn(y, fs, sec)
        if on_stage is not None:
            on_stage(name, y)
    return y


def finish(y: np.ndarray, fs: float, plan: Plan) -> np.ndarray:
    """The output stage (export quantisation) of plan.output; y unchanged without one."""
    return y if plan.output is None else output_stage.apply(y, fs, plan.output)


def render_plan(x: np.ndarray, fs: float, plan: Plan, pre=None, on_stage=None):
    """Whole chain with the blocked limiter. Returns (master, limiter info, pre-limiter audio)."""
    from eluate.plan import check_environment
    if fs != plan.fs:
        raise ValueError(f"plan is for {plan.fs} Hz, audio is {fs} Hz")
    check_environment(plan.environment)
    if pre is None:
        pre = render_pre(x, fs, plan)
        if on_stage is not None:
            on_stage("pre", pre)
    z = render_chain(pre, fs, plan, on_stage)
    if plan.limiter is None:
        return finish(z.copy(), fs, plan), {}, z
    y, info = limit_blocked(z, fs, plan.limiter)
    if on_stage is not None:
        on_stage("limiter", y)
    if plan.output is not None:
        y = finish(y, fs, plan)
        if on_stage is not None:
            on_stage("output", y)
    return y, info, z


# ------------------------------------------------------------------ loudness and clipper

def proxy_rms(z: np.ndarray, fs: float, plan: LimiterPlan, drive_db: float,
              z_hi: np.ndarray | None = None) -> float:
    """The loudness target's reading (meter.rms_meter: 300 ms RMS, AES17, loudest
    passage) of the proxy limiter's output."""
    return float(meter.rms_meter(limit_proxy(z, fs, plan, drive_db, z_hi), fs)["loud_dbfs"])


def gain_for_rms(z: np.ndarray, fs: float, plan: LimiterPlan, target_dbfs: float,
                 start_db: float = 0.0, bias_db: float = 0.0, info: dict | None = None) -> float:
    """The limiter input gain at which the proxy limiter's loudness reading (proxy_rms) plus
    bias_db (the real render's offset from the proxy, calibrated by the solver) equals
    target_dbfs: bracket from start_db, then brentq (_bracket_drive), within
    DRIVE_RANGE_DB. A silent z returns start_db. info: set info["at_limit"] (module
    contract)."""
    z_hi = clip_hi_for(z, fs, plan)          # the split is linear: once for every drive

    def f(d):
        v = proxy_rms(z, fs, plan, d, z_hi)
        return (v + bias_db - target_dbfs) if math.isfinite(v) else -1.0
    flag = {"at_limit": False}
    if info is not None:
        info.update(flag)
    if not math.isfinite(proxy_rms(z, fs, plan, start_db, z_hi)):
        return float(start_db)
    lo, hi = DRIVE_RANGE_DB
    g = float(_bracket_drive(f, float(np.clip(start_db, lo, hi)), lo, hi, flag=flag))
    if info is not None:
        info.update(flag)
    return g


def _bisect_down(f, lo: float, hi: float, target: float, xtol: float, ftol: float):
    """Bisection of a non-increasing f with f(lo) >= target >= f(hi): returns (x_near,
    x_hi), the evaluation closest to the target and the highest x seen at which
    f <= target (so f(x_hi) <= target holds, for a cap)."""
    fl, fh = f(lo), f(hi)
    best = (lo, fl) if abs(fl - target) < abs(fh - target) else (hi, fh)
    while hi - lo > xtol and abs(best[1] - target) > ftol:
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm - target) < abs(best[1] - target):
            best = (mid, fm)
        if fm > target:
            lo = mid
        else:
            hi = mid
    return best[0], hi


def hit_peaks_db(env: np.ndarray, fs: float, edges_s) -> np.ndarray:
    """Per-hit maximum (dBFS; NaN for an empty or silent hit) of a base-rate peak
    envelope."""
    hv = envelope.hit_values(env, fs, edges_s)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 20.0 * np.log10(hv)
    return np.where(np.isfinite(db), db, np.nan)


def clip_threshold_for_average_peak(y: np.ndarray, fs: float, plan: LimiterPlan,
                                    target_db: float, hits: dict,
                                    cap_db: float = CLIP_MAX_GR_DB,
                                    block: int = BLOCK) -> dict:
    """The whole-signal clipper's threshold (dBFS of the 8x signal before the limiter's
    input gain) for an average output-peak fall of target_db over hits (maintainer,
    2026-10-19): per hit, the fall is the hit's 8x input peak minus its 8x peak after
    the curve and its band-limit (limit_blocked measure_only, detector_env); the average
    is their median over the loud hits (envelope.summarise_hits), the outlier their
    maximum. Bisection from no clipping down to CLIP_SEARCH_DB below the loud hits'
    median input peak. Without a per-sample cap (plan.clip_max_gr_db None), when the
    outlier then exceeds cap_db the threshold is raised by bisection until it is at
    most cap_db (the stage clamps at the cap); with one, the curve holds the outlier
    and the threshold stays at the average's solution. Returns {"threshold_db",
    "peak_gr_db" (the average fall), "outlier_gr_db", "capped" (the average's solution
    put the outlier above cap_db: enforced as above without a per-sample cap, only
    reported with one, so capped with bound None is possible), "curve_gr_db" (the
    curve's reduction at the track's 8x peak), "peak_8x_dbfs", "bound" ("outlier",
    "sample cap" or None)}. With plan.clip_split_hz the solve runs on the part of y above
    it (limiter.clip_hi_part), the signal the clipper acts on: every reading (the input
    peaks, the falls, peak_8x_dbfs) is that part's."""
    if plan.clip_split_hz is not None:
        y = lim.clip_hi_part(np.asarray(y, dtype=np.float64), fs, plan.clip_split_hz)
    base = replace(plan, input_gain_db=0.0, clip_depth_db=0.0, clip_onsets_s=[],
                   clip_split_hz=None)
    e = hits["edges_s"]
    m0 = limit_blocked(y, fs, replace(base, clip_threshold_db=None), block, measure_only=True,
                       detector_env=True)
    p_db = m0["detector_peak_dbfs"]
    if not math.isfinite(p_db) or target_db <= 0.0:
        t = p_db + 0.01 if math.isfinite(p_db) else 0.0
        return {"threshold_db": float(t), "peak_gr_db": 0.0, "outlier_gr_db": 0.0,
                "capped": False, "curve_gr_db": 0.0, "peak_8x_dbfs": p_db}
    in_db = hit_peaks_db(m0["detector_env"], fs, e)
    del m0
    peak = 10.0 ** (p_db / 20.0)
    memo: dict[float, dict] = {}

    def reading(t):
        t = float(t)
        if t not in memo:
            m = limit_blocked(y, fs, replace(base, clip_threshold_db=t), block,
                              measure_only=True, detector_env=True)
            fall = np.maximum(in_db - hit_peaks_db(m["detector_env"], fs, e), 0.0)
            memo[t] = envelope.summarise_hits(fall, hits["loud"])
        return memo[t]

    r_cap = lim.sample_cap_ratio(plan)

    def curve(t):
        return lim.clip_reduction_db(peak, *lim.clip_curve(10.0 ** (t / 20.0), plan.clip_knee_db),
                                     r_cap)

    loud = np.asarray(hits["loud"], dtype=bool) & np.isfinite(in_db)
    ref = float(np.median(in_db[loud])) if np.any(loud) else p_db
    hi = p_db + 0.01
    lo = min(hi, ref - target_db - plan.clip_knee_db - 1.0)
    floor = ref - CLIP_SEARCH_DB
    while reading(lo)["average_db"] < target_db and lo > floor:
        lo = max(floor, lo - CLIP_SEARCH_STEP_DB)
    if reading(lo)["average_db"] < target_db:
        t = lo
    else:
        t, _ = _bisect_down(lambda v: reading(v)["average_db"], lo, hi, target_db,
                            CLIP_SOLVE_XTOL_DB, 0.01)
    capped = reading(t)["outlier_db"] > cap_db
    bound = None
    if r_cap > 0.0:
        # the per-sample cap holds the outlier in the curve itself: the average is never
        # lowered to make room (docs/dynamics-macros.md 3, step 2)
        if reading(t)["outlier_db"] >= plan.clip_max_gr_db - 0.1:
            bound = "sample cap"
    elif capped:
        _, t = _bisect_down(lambda v: reading(v)["outlier_db"], t, hi, cap_db,
                            CLIP_SOLVE_XTOL_DB, 0.0)
        bound = "outlier"
    r = reading(t)
    return {"threshold_db": float(t), "peak_gr_db": r["average_db"],
            "outlier_gr_db": r["outlier_db"], "capped": bool(capped), "curve_gr_db": curve(t),
            "peak_8x_dbfs": p_db, "bound": bound}


def clip_threshold_for_output_peak(y: np.ndarray, fs: float, plan: LimiterPlan,
                                   target_db: float, block: int = BLOCK) -> dict:
    """Whole-signal clipper threshold (dBFS of the 8x signal before the limiter's input
    gain) at which the clipper's output peak, the 8x peak after the soft-knee curve and
    its band-limit (limit_blocked measure_only, the limiter detector's input), sits
    target_db below the 8x peak of y (maintainer, 2026-10-18: the target is on the output
    peak, not the curve's reduction at the 8x peak). Bracketed from the curve's own
    solution (limiter.clip_threshold_for_peak_gr, whose output reduction is smaller,
    the band-limit letting some back) downward in CLIP_SEARCH_STEP_DB steps to at most
    CLIP_SEARCH_DB below the peak, then brentq. When even the lowest threshold falls
    short, that threshold is returned with the reduction it reaches. Returns
    {"threshold_db", "peak_gr_db" (output-peak reduction), "curve_gr_db" (the curve's
    reduction at the 8x peak), "peak_8x_dbfs"}; a silent y gives 0 dBFS and no
    reduction. With plan.clip_split_hz, on the part of y above it, as
    clip_threshold_for_average_peak."""
    if plan.clip_split_hz is not None:
        y = lim.clip_hi_part(np.asarray(y, dtype=np.float64), fs, plan.clip_split_hz)
    p_db = lim.peak_8x_db(y)
    if not math.isfinite(p_db) or target_db <= 0.0:
        t = p_db + 0.01 if math.isfinite(p_db) else 0.0
        return {"threshold_db": float(t), "peak_gr_db": 0.0, "curve_gr_db": 0.0,
                "peak_8x_dbfs": p_db}
    base = replace(plan, input_gain_db=0.0, clip_depth_db=0.0, clip_onsets_s=[],
                   clip_split_hz=None)
    peak = 10.0 ** (p_db / 20.0)

    def out_red(t):
        m = limit_blocked(y, fs, replace(base, clip_threshold_db=float(t)), block,
                          measure_only=True)
        return float(p_db - m["detector_peak_dbfs"])

    def curve(t):
        return lim.clip_reduction_db(peak, *lim.clip_curve(10.0 ** (t / 20.0), plan.clip_knee_db),
                                     lim.sample_cap_ratio(plan))

    hi = lim.clip_threshold_for_peak_gr(y, min(target_db, CLIP_MAX_GR_DB),
                                        plan.clip_knee_db)["threshold_db"]
    f_hi = out_red(hi) - target_db
    if f_hi >= 0.0:
        lo, hi = hi, p_db + 0.01
    else:
        lo, f_lo = hi, f_hi
        while f_lo < 0.0 and lo > p_db - CLIP_SEARCH_DB:
            hi, lo = lo, max(p_db - CLIP_SEARCH_DB, lo - CLIP_SEARCH_STEP_DB)
            f_lo = out_red(lo) - target_db
        if f_lo < 0.0:
            return {"threshold_db": float(lo), "peak_gr_db": f_lo + target_db,
                    "curve_gr_db": curve(lo), "peak_8x_dbfs": p_db}
    t = float(brentq(lambda v: out_red(v) - target_db, lo, hi, xtol=CLIP_SOLVE_XTOL_DB))
    return {"threshold_db": t, "peak_gr_db": out_red(t), "curve_gr_db": curve(t),
            "peak_8x_dbfs": p_db}


# ------------------------------------------------------------------ levers

@dataclass
class Levers:
    """The solver's variables (chunk 4): the limiter's input gain (solved for the RMS
    target each render, then clamped to the limiter's budget), the tone rotation
    (dB/oct added to the input EQ's target, refitted) and the limiter ceiling (lowered
    when the constrained true-peak meter reads over). Every other stage is solved for
    its peak-GR budget at design."""

    drive_db: float = 0.0
    tone_rotation: float = 0.0
    limiter_ceiling_dbtp: float = planner.CEILING_DBTP

    @classmethod
    def from_plan(cls, plan: Plan) -> "Levers":
        lp = plan.limiter
        return cls(lp.input_gain_db if lp is not None else 0.0, 0.0,
                   lp.ceiling_dbtp if lp is not None else planner.CEILING_DBTP)

    def key_pre(self):
        return (self.tone_rotation,)


def apply_levers(plan0: Plan, lv: Levers) -> Plan:
    """Unrealised plan with the levers written in: the input EQ refitted with the tone
    rotation added to its target (planner.refit_inputeq), the limiter's input gain and
    ceiling."""
    p = copy.deepcopy(plan0)
    if p.inputeq is not None and p.inputeq.design is not None and lv.tone_rotation != 0.0:
        p.inputeq = planner.refit_inputeq(p.inputeq, p.fs, lv.tone_rotation)
    if p.limiter is not None:
        p.limiter = replace(p.limiter, input_gain_db=float(lv.drive_db),
                            ceiling_dbtp=float(lv.limiter_ceiling_dbtp))
    return p


SPILL_PREFIX = "eluate_chain_"
SPILL_STALE_S = 6 * 3600.0        # a spill dir untouched this long belongs to no live render


def sweep_spill(tmp_dir=None, max_age_s: float = SPILL_STALE_S, now: float | None = None) -> int:
    """Removes the Chain spill dirs (SPILL_PREFIX*) in tmp_dir (tempfile.gettempdir())
    that a killed engine left behind (report 3.6): only those whose newest mtime (the
    dir's or any file's in it) is over max_age_s old, so the live dir of a concurrent
    render, which writes its spill files as it solves, is never touched. Errors (a file
    another process holds open) are skipped. Returns the number of dirs removed."""
    import shutil
    import tempfile
    from pathlib import Path
    root = Path(tmp_dir if tmp_dir is not None else tempfile.gettempdir())
    now = time.time() if now is None else now
    n = 0
    try:
        dirs = [d for d in root.iterdir() if d.name.startswith(SPILL_PREFIX) and d.is_dir()]
    except OSError:
        return 0
    for d in dirs:
        try:
            newest = max([d.stat().st_mtime] + [f.stat().st_mtime for f in d.iterdir()])
        except OSError:
            continue
        if now - newest > max_age_s:
            shutil.rmtree(d, ignore_errors=True)
            n += not d.exists()
    return n


class Chain:
    """Stage cache for the solver: the input gain + inputeq + stages 1-2 output and the
    pre-limiter audio (multiband, saturate and glue, each solved for its budget on its
    real input) per tone rotation, the clipper's threshold and the limiter's clipped peak
    per tone rotation and ceiling.

    Cached whole-track audio (the premaster, the stage 1-2 output and the pre-limiter
    audio) is spilled to .npy files in a temporary directory and loaded only when a
    step needs it (np.save / np.load round-trip float64 exactly), so at most the array
    a step is working on is resident (spec 1 memory). close() removes the directory;
    solve() calls it in a finally block for a Chain it created, and the Chain is also a
    context manager. A killed engine never reaches close(); sweep_spill removes such
    dirs once stale (app_engine and the CLI's `master` run it before a render).

    The loud hits are frozen on the stage-1 input of each tone rotation
    (planner.freeze_hits, D2) and carried in the returned plan's hits. With sections
    (the targets' [{start, end, ...}]) every stage's output is measured per section
    (section_loudness) and kept per rotation in `stage_sections`."""

    def __init__(self, x: np.ndarray, fs: int, plan0: Plan, spill_dir=None, sections=None):
        import tempfile
        self.fs = int(fs)
        self.plan0 = plan0
        self.sections = list(sections or [])
        self.stage_sections: list = []    # [(stage, per-section LUFS)] for the cached key
        self._tmp = tempfile.TemporaryDirectory(prefix=SPILL_PREFIX, dir=spill_dir)
        self._x = self._put("x", np.asarray(x, dtype=np.float64))
        t = time.perf_counter()
        self._pre = (Levers.from_plan(plan0).key_pre(),
                     self._put("pre", render_pre(np.asarray(x, dtype=np.float64), self.fs, plan0)))
        self.pre_s = time.perf_counter() - t
        self._z = (None, None, None)      # key, realised plan (multiband, glue), spill path
        self._lim = (None, None)          # key, limiter section (clipper threshold, peak)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _put(self, name: str, a: np.ndarray):
        from pathlib import Path
        path = Path(self._tmp.name) / f"{name}.npy"
        np.save(path, a, allow_pickle=False)
        return path

    @staticmethod
    def _get(path) -> np.ndarray:
        return np.load(path, allow_pickle=False)

    @property
    def pre(self) -> np.ndarray:
        """The current input + inputeq + stage 1-2 output (loaded from the spill file)."""
        return self._get(self._pre[1])

    def close(self) -> None:
        self._tmp.cleanup()

    def pre_limiter(self, lv: Levers):
        """(realised plan, pre-limiter audio) for a lever state; the limiter section carries
        the solved clipper threshold and the clipped peak, its gain as lv.drive_db."""
        p = apply_levers(self.plan0, lv)
        fs = self.fs
        key = lv.key_pre()
        if self._pre[0] != key:
            # the tone rotation refits the input EQ ahead of notch and sidestrip
            self._pre = (key, self._put("pre", render_pre(self._get(self._x), fs, p)))
        if self._z[0] != key:
            # the loud hits of the stage-1 input, frozen for every stage below (D2)
            hits = planner.freeze_hits(p, self.pre, fs).hits
            p.hits = hits
            secs = []

            def tap(name, a):
                if self.sections:
                    secs.append((name, section_loudness(a, fs, self.sections)))
            tap("input EQ, notch, sidestrip", self.pre)
            # inputs loaded inline (no local name) so a stage that releases its input
            # after the M/S conversion actually frees it (spec 1 memory)
            mb = (mb_stage.design_from(p.multiband, self.pre, fs,
                                       hits=planner.stage_hits(p, self.pre, fs))
                  if p.multiband else None)
            y = mb_stage.apply(self.pre, fs, mb) if mb else self.pre
            if mb:
                tap("multiband", y)
            if p.tone is not None and not p.preserve_mix:
                y = tone_stage.apply(y, fs, p.tone)
                tap("tone", y)
            if p.saturate is not None:
                y = sat_stage.apply(y, fs, p.saturate)
                tap("saturation", y)
            gl = None
            if p.glue is not None:
                gl = planner.design_glue(y, fs, p.glue, planner.stage_hits(p, y, fs))
                y = glue_stage.apply(y, fs, gl)
                tap("glue", y)
            sl = planner.design_speclim(p, y, fs)
            if sl is not None:
                y = mb_stage.apply(y, fs, sl)
                tap("frequency limiter", y)
            self._z = (key, (mb, gl, sl, hits), self._put("z", y))
            self.stage_sections = secs
            self._lim = (None, None)
        p.multiband, p.glue, p.speclim, p.hits = self._z[1]
        y = self._get(self._z[2])
        # the kick detective clips relative to the ceiling, so its ceiling is in the key
        lkey = key + ((lv.limiter_ceiling_dbtp,) if p.limiter is not None
                      and p.limiter.clip_target_db is None else ())
        if p.limiter is not None and self._lim[0] != lkey:
            h = planner.stage_hits(p, y, fs)
            self._lim = (lkey, planner.limiter_peak(y, fs, planner.solve_clipper(y, fs, p.limiter,
                                                                                 h), h))
        if p.limiter is not None:
            lp = self._lim[1]
            rec = copy.deepcopy(lp.budget)
            if rec and rec.get("limiter"):
                rec["limiter"]["ceiling_dbtp"] = p.limiter.ceiling_dbtp
            p.limiter = replace(lp, input_gain_db=p.limiter.input_gain_db,
                                ceiling_dbtp=p.limiter.ceiling_dbtp, budget=rec)
        return p, y


# ------------------------------------------------------------------ measurement

def section_loudness(y: np.ndarray, fs: float, sections) -> list[float]:
    """Gated integrated loudness of each section (blocks inside it), the
    features.loudness.per_section rule on a K-weighting of the whole signal."""
    ps = meter._kpower(y, fs, None)
    b_start, b_len, b_p = meter._window_powers(ps, fs, meter.BLOCK_S, meter.HOP_S)
    out = []
    for s in sections:
        i0, i1 = int(round(s["start"] * fs)), int(round(s["end"] * fs))
        inside = (b_start >= i0) & (b_start + b_len <= i1)
        out.append(float(meter._lufs(meter._gated_mean_power(b_p[inside]))))
    return out


def section_deltas(levels, ignore_lu: float = SECTION_IGNORE_LU, ref_levels=None):
    """Loudness of each section relative to the loudest; NaN where ignored. With
    ref_levels (the premaster's), the gate and the reference section are the
    premaster's for both signals (shared frames, audit 07 A13): a section counts when
    the premaster's is within ignore_lu of the premaster's loudest, and every delta is
    taken against that same section. A counted section, or the reference, with no
    measurable level in `levels` gives +inf (lost, a miss), not NaN (ignored)."""
    v = np.asarray(levels, dtype=np.float64)
    r = v if ref_levels is None else np.asarray(ref_levels, dtype=np.float64)
    fin = np.isfinite(r)
    if not np.any(fin):
        return np.full(v.shape, np.nan)
    top_i = int(np.argmax(np.where(fin, r, -np.inf)))
    keep = fin & (r >= r[top_i] - ignore_lu)
    with np.errstate(invalid="ignore"):
        d = np.where(keep, v - v[top_i], np.nan)
    return np.where(keep & ~np.isfinite(d), np.inf, d)


class Reference:
    """Premaster data the constraints compare against. Holds the premaster itself (by
    reference, no copy) for the shared-frame crest loss of multiband.crest_loss."""

    def __init__(self, x: np.ndarray, fs: int, targets: planner.Targets):
        self.targets = targets
        self.x = x
        self.ltas_c = np.asarray(targets.ltas_centres_hz)
        self.ltas = np.asarray(targets.ltas_ref_db)
        self.sec_in = [s["integrated_lufs"] for s in targets.sections]
        self.sec_delta_in = section_deltas(self.sec_in)


def crest_excess(ref_p95: np.ndarray, out_p95: np.ndarray, limits, exempt=()) -> tuple:
    """(loss, excess) bands x views from two p95 tables (kept for callers comparing
    summaries; the solver uses crest_constraints on multiband.crest_loss). Excess > 0
    violates; gated before and not after: +inf (audit 06 F5); ungated before: NaN;
    an exempt [band, view]: NaN."""
    loss = ref_p95 - out_p95
    ex = loss - np.asarray(limits, dtype=np.float64)[:, None]
    ex = np.where(np.isfinite(ref_p95) & ~np.isfinite(out_p95), np.inf, ex)
    for b, v in exempt:
        ex[b, v] = np.nan
    return loss, ex


def _worst(ex: np.ndarray) -> np.ndarray:
    return np.max(np.where(np.isnan(ex), -np.inf, ex), axis=1)


def crest_constraints(cl: dict, targets: planner.Targets) -> dict:
    """Per-band crest excess (dB, > 0 violates) from multiband.crest_loss under the
    targets' crest_mode: "loss" = loss - limit (silenced view +inf), "floor" = floor -
    after p95 (silenced +inf), "report" = loss reported, excess 0 (nothing fails).
    Exempt [band, view] pairs (targets.crest_exempt, empty unless side_band0_exempt)
    are left out. Returns {"excess": (B, V), "worst": (B,), "value": (B,)}."""
    loss = np.asarray(cl["loss_db"], dtype=np.float64)
    after = np.asarray(cl["after_p95_db"], dtype=np.float64)
    sil = np.asarray(cl["silenced"], dtype=bool)
    nb = loss.shape[0]
    mode = targets.crest_mode
    if mode == "floor":
        fl = np.asarray(targets.crest_floors_db, dtype=np.float64)
        ex = fl[:, None] - after
        val = after
    else:
        ex = loss - np.asarray(targets.crest_limits_db, dtype=np.float64)[:, None]
        val = loss
    ex = np.where(sil, np.inf, ex)
    for b, v in targets.crest_exempt:
        ex[b, v] = np.nan
    if mode == "report":
        ex = np.where(np.isnan(ex), np.nan, 0.0)
    worst = np.full(nb, math.nan)
    vals = np.full(nb, math.nan)
    for b in range(nb):
        live = ~np.isnan(ex[b])
        if np.any(live):
            worst[b] = float(np.max(ex[b][live]))
            v = val[b][live]
            v = v[np.isfinite(v)]
            if v.size:
                vals[b] = float(np.max(v))
    return {"excess": ex, "worst": worst, "value": vals}


def true_peaks(y: np.ndarray, fs: int, lim_info: dict | None = None) -> dict:
    """True-peak readings of the master as written (the output stage's samples, cast to
    float32, which is exact for PCM; a master rendered without an output stage is
    read after the float32 cast a FLOAT file applies), dBTP: 4x (BS.1770-4
    Annex 2), 8x (meter), 32x (FFT zero-padding), and the constrained readings
    "worst" = max of all three, "8x" = max of 8x and 32x (spec 3.6 draft 0.5)."""
    y32 = np.asarray(y, dtype=np.float32).astype(np.float64)
    tp4 = meter.true_peak(y32, fs, 4)
    tp8 = meter.true_peak(y32, fs, 8)
    tp32 = lim.true_peak_fft(y32)
    return {"tp4": tp4, "tp8": tp8, "tp32": tp32, "worst": max(tp4, tp8, tp32),
            "8x": max(tp8, tp32)}


def measure(y: np.ndarray, fs: int, plan: Plan, lim_info: dict, ref: Reference,
            z: np.ndarray | None = None) -> dict:
    """Constraint values and status for one render (module contract)."""
    t = ref.targets
    out: dict = {}
    # the loudness target's reading: the maintainer's meter (300 ms RMS, AES17, loudest passage)
    rms = float(meter.rms_meter(y, fs)["loud_dbfs"])
    out["rms_dbfs"] = rms
    out["rms_aes17_dbfs"] = rms
    out["rms_1s_dbfs"] = float(meter.rms_1s(y, fs)["loud_section_dbfs"])
    out["s95_lufs"] = meter.loud_passage_lufs(y, fs)["s95"]     # reported, not a target
    tl = tilt_feature.analyse(y, fs)
    out["slope_db_oct"] = float(tl["slope_db_oct"]["stereo"])
    ltas = np.asarray(tl["ltas_db"]["stereo"])
    c = np.asarray(tl["centres"])
    sel = (c >= LTAS_RANGE_HZ[0]) & (c <= LTAS_RANGE_HZ[1]) & np.isin(np.round(c, 6),
                                                                       np.round(ref.ltas_c, 6))
    ref_sel = np.isin(np.round(ref.ltas_c, 6), np.round(c[sel], 6))
    dv = ltas[sel] - ref.ltas[ref_sel]
    # relative gate on the premaster's LTAS, shared by both signals (audit 07 A13)
    rl = ref.ltas[ref_sel]
    good = np.isfinite(rl) & (rl > (np.max(rl[np.isfinite(rl)]) if np.any(np.isfinite(rl))
                                    else 0.0) - LTAS_GATE_DB)
    dev = dv - (np.mean(dv[good]) if np.any(good) else 0.0)
    out["ltas_dev_db"] = np.where(good, dev, np.nan).tolist()
    out["ltas_centres_hz"] = c[sel].tolist()
    cl = mb_stage.crest_loss(ref.x, y, fs, t.crossovers_hz, limits_db=t.crest_limits_db)
    cc = crest_constraints(cl, t)
    out["crest_p95_db"] = np.asarray(cl["after_p95_db"]).tolist()
    out["crest_before_p95_db"] = np.asarray(cl["before_p95_db"]).tolist()
    out["crest_loss_db"] = np.asarray(cl["loss_db"]).tolist()
    out["crest_silenced"] = np.asarray(cl["silenced"]).tolist()
    out["crest_excess_db"] = cc["excess"].tolist()
    comp, ceil = [], []
    mb = plan.multiband
    if mb is not None:
        comp = [[float(-max(np.min(e.values), -MULTIBAND_MAX_GR_DB)) for e in p] for p in mb.gr_db]
        if mb.ceiling_gr_db is not None:
            ceil = [[float(-np.min(e.values)) for e in p] for p in mb.ceiling_gr_db]
    # reported, not a solved constraint: the stage clamps it (audit 07 A12)
    out["comp_gr_max_db"] = comp
    out["ceiling_gr_max_db"] = ceil
    out["glue_gr_max_db"] = float(-np.min(plan.glue.gain_db.values)) if plan.glue else 0.0
    out["budget"] = planner.budget_report(plan, lim_info)
    tp = true_peaks(y, fs, lim_info)
    out["true_peak_4x_dbtp"], out["true_peak_8x_dbtp"] = tp["tp4"], tp["tp8"]
    out["true_peak_32x_dbtp"] = tp["tp32"]
    out["true_peak_constrained_dbtp"] = tp[t.tp_meter]
    sec = section_loudness(y, fs, t.sections) if t.sections else []
    out["section_lufs"] = sec
    dd = section_change(sec, ref)
    out["section_delta_change_lu"] = [None if not np.isfinite(v) else float(v) for v in dd]
    out["section_contrast_loss_lu"] = contrast_loss(dd)
    # a counted section lost in the render: a finite penalty, so a later render that
    # loses it too is judged on everything else (S8)
    dd = np.where(np.isinf(dd), SECTION_LOST_LU, dd)

    cons = []

    def add(name, value, target, tol, excess, detail=None, mode=None):
        cons.append({"name": name, "value": value, "target": target, "tol": tol,
                     "excess": float(excess), "ok": bool(excess <= 1e-9), "detail": detail,
                     "mode": mode})

    add("rms", rms, t.rms_dbfs, t.rms_tol_db,
        abs(rms - t.rms_dbfs) - t.rms_tol_db if math.isfinite(rms) else math.inf,
        {"meter": "300 ms RMS, AES17, loudest passage", "rms_1s_dbfs": out["rms_1s_dbfs"]})
    if t.preserve_mix:
        mx = float(np.nanmax(np.abs(dev[good]))) if np.any(good) else 0.0
        add("ltas_shape", mx, 0.0, t.ltas_tol_db, mx - t.ltas_tol_db)
    else:
        # the live profile reports the slope and flags it, never solves for it
        rep_only = getattr(t, "slope_mode", "constraint") == "report"
        dev = abs(out["slope_db_oct"] - t.slope_db_oct) - t.slope_tol
        add("slope", out["slope_db_oct"], t.slope_db_oct, t.slope_tol,
            min(dev, 0.0) if rep_only else dev, {"deviation_db": dev} if rep_only else None,
            mode="report" if rep_only else None)
    for b in range(cc["worst"].size):
        w = cc["worst"][b]
        tol = (float(t.crest_limits_db[b]) if t.crest_mode != "floor" else 0.5)
        target_v = 0.0 if t.crest_mode != "floor" else float(t.crest_floors_db[b])
        add(f"crest_band{b}", float(cc["value"][b]), target_v, tol,
            0.0 if math.isnan(w) else float(w),
            {"loss_db": np.asarray(cl["loss_db"])[b].tolist(),
             "after_p95_db": np.asarray(cl["after_p95_db"])[b].tolist(),
             "silenced": np.asarray(cl["silenced"])[b].tolist(),
             "views": list(mb_stage.CREST_VIEWS)}, mode=t.crest_mode)
    add("true_peak", tp[t.tp_meter], t.true_peak_max_dbtp, 0.0,
        tp[t.tp_meter] - t.true_peak_max_dbtp, {"meter": t.tp_meter, **tp})
    if t.preserve_mix and dd.size:
        fin = dd[~np.isnan(dd)]
        mx = float(np.max(np.abs(fin))) if fin.size else 0.0
        add("section_deltas", mx, 0.0, t.section_tol_lu, mx - t.section_tol_lu)
    out["constraints"] = cons
    out["all_ok"] = all(c["ok"] for c in cons)
    return out


def section_change(levels, ref: "Reference") -> np.ndarray:
    """Each section's level against the loudest, minus the premaster's (LU; > 0 = it rose
    toward the loudest), on the premaster's gate and reference section (section_deltas);
    NaN = not judged, +inf = a counted section lost."""
    if not len(levels):
        return np.zeros(0)
    return section_deltas(levels, ref_levels=ref.sec_in) - ref.sec_delta_in


def contrast_loss(change) -> float:
    """The largest rise of section_change (LU), 0 when no judged section rose; a counted
    section lost in the render (an infinite change) reads SECTION_LOST_LU, so the score
    penalises losing a section in reshape mode too (audit 13 #2), not only under
    Preserve."""
    d = np.asarray(change, dtype=np.float64)
    d = np.where(np.isinf(d), SECTION_LOST_LU, d)
    d = d[np.isfinite(d)]
    return float(max(0.0, np.max(d))) if d.size else 0.0


def contrast_term(m: dict) -> float:
    """The score's section-contrast term: the loss past FLAG_SECTION_LU, normalised by it."""
    loss = m.get("section_contrast_loss_lu") or 0.0
    return max(0.0, float(loss) - FLAG_SECTION_LU) / FLAG_SECTION_LU


def score(m: dict, skip=()) -> float:
    """Sum of normalised constraint excesses (0 when every constraint is met) plus the
    section-contrast term (contrast_term, named "section_contrast"), leaving out the terms
    named in skip."""
    s = sum(max(0.0, c["excess"]) / max(c["tol"], 0.1) for c in m["constraints"]
            if c["name"] not in skip)
    if "section_contrast" not in skip:
        s += contrast_term(m)
    return float(s)


def lever_score(m: dict) -> float:
    """score without the RMS, true-peak and section-contrast terms: the tone rotation is
    judged on what it changes, the slope first (R7a); the loudness is solved through the
    limiter gain and the true peak through the limiter ceiling, each in renders of its
    own, where the contrast is judged."""
    return score(m, skip=("rms", "true_peak", "section_contrast"))


def flattens(m_new: dict, m_kept: dict) -> bool:
    """R13: the new render's contrast loss is past FLAG_SECTION_LU and more than
    CONTRAST_SLACK_LU above the kept render's."""
    a = m_new.get("section_contrast_loss_lu") or 0.0
    b = m_kept.get("section_contrast_loss_lu") or 0.0
    return a > FLAG_SECTION_LU and a > b + CONTRAST_SLACK_LU


def judge_move(m_new: dict, m_kept: dict, loudness_accepted: bool) -> str | None:
    """The loudness / true-peak pass's keep test: None = keep; "section contrast" when the
    kept render's loudness is accepted and the move flattens (R13); "score" when the full
    score rises."""
    if loudness_accepted and flattens(m_new, m_kept):
        return "section contrast"
    return None if score(m_new) <= score(m_kept) else "score"


def stage_changes(stage_levels, ref: "Reference") -> list[dict]:
    """[{"stage", "change_lu", "contrast_loss_lu"}] per stage from [(stage, levels)]."""
    out = []
    for name, lv in stage_levels:
        d = section_change(lv, ref)
        out.append({"stage": name,
                    "change_lu": [None if not np.isfinite(v) else float(v) for v in d],
                    "contrast_loss_lu": contrast_loss(d)})
    return out


# ------------------------------------------------------------------ solver

@dataclass
class SolveResult:
    plan: Plan
    master: np.ndarray
    lim_info: dict
    metrics: dict
    levers: Levers
    renders: int
    history: list
    ok: bool
    misses: list
    warnings: list
    runtime_s: float
    pre_limiter: np.ndarray

    def summary(self) -> dict:
        return {"renders": self.renders, "ok": self.ok, "misses": self.misses,
                "warnings": self.warnings, "levers": asdict(self.levers),
                "runtime_s": self.runtime_s, "history": self.history,
                "constraints": self.metrics["constraints"],
                "reported": {k: self.metrics.get(k) for k in
                             ("budget", "rms_dbfs", "rms_aes17_dbfs", "s95_lufs",
                              "comp_gr_max_db", "ceiling_gr_max_db", "glue_gr_max_db",
                              "true_peak_4x_dbtp", "true_peak_8x_dbtp", "true_peak_32x_dbtp",
                              "section_delta_change_lu", "section_contrast_loss_lu",
                              "section_change_by_stage")}}


def _bracket_drive(f, d0: float, lo_lim: float, hi_lim: float, step: float = 6.0,
                   flag: dict | None = None):
    """Root of increasing f near d0 by bracketing + brentq; clamped to the limits (flag:
    flag["at_limit"] = True when no root lies inside them)."""
    a, b = max(lo_lim, d0 - step), min(hi_lim, d0 + step)
    fa, fb = f(a), f(b)
    while fa > 0 and a > lo_lim:
        a, b, fb = max(lo_lim, a - 2 * step), a, fa
        fa = f(a)
    while fb < 0 and b < hi_lim:
        a, fa, b = b, fb, min(hi_lim, b + 2 * step)
        fb = f(b)
    if fa > 0 or fb < 0:
        if flag is not None:
            flag["at_limit"] = True
        return a if fa > 0 else b
    return brentq(f, a, b, xtol=0.01)


def misses(cons: list[dict]) -> list[dict]:
    """The constraints a render misses (value, target, tolerance, excess), largest
    normalised excess first."""
    bad = [c for c in cons if not c["ok"]]
    bad.sort(key=lambda c: -c["excess"] / max(c["tol"], 0.1))
    return [{k: c[k] for k in ("name", "value", "target", "tol", "excess")} for c in bad]


def _same_refit(plan0, rot_now: float, rot_new: float) -> bool:
    """The input EQ refitted at rot_new cannot move the slope: its predicted slope is within
    ROT_SLOPE_MIN of the refit at rot_now (every section that would move it is at its
    bound; S7)."""
    eq = plan0.inputeq
    if eq is None or eq.design is None:
        return False

    def predicted(rot):
        d = dict(eq.design)
        d["rotation"] = float(rot)
        return planner.fit_inputeq(d, plan0.fs, eq.phase)[1]["predicted_slope_db_oct"]

    return abs(predicted(rot_new) - predicted(rot_now)) < ROT_SLOPE_MIN


def slope_warning(plan0, rotation: float, slope: float, targets) -> dict:
    """T1: the slope missed. The input EQ sections at their bound (planner.fit_inputeq's
    at_bound, at the kept rotation) and the Tilt that would make the reached slope the
    target (within planner.TILT_MAX)."""
    at = []
    eq = plan0.inputeq
    if eq is not None and eq.design is not None:
        d = dict(eq.design)
        d["rotation"] = float(rotation)
        at = planner.fit_inputeq(d, plan0.fs, eq.phase)[1]["at_bound"]
    tilt = float(np.clip(round(slope - planner.REFERENCE_SLOPE, 1), -planner.TILT_MAX,
                         planner.TILT_MAX))
    return {"stage": "inputeq", "kind": "slope", "achieved_db_oct": float(slope),
            "target_db_oct": float(targets.slope_db_oct), "at_bound": at, "tilt_db": tilt}


def _rotatable(plan: Plan) -> bool:
    """The tone rotation has something to act on: an input EQ with its design record."""
    return (plan.inputeq is not None and plan.inputeq.design is not None
            and not plan.preserve_mix)


def _clamped(plan: Plan) -> bool:
    """The limiter's budget clamped the gain the loudness target asked for."""
    lr = ((plan.limiter.budget or {}).get("limiter") or {}) if plan.limiter is not None else {}
    return bool(lr.get("warnings"))


def render_bound(max_outer: int = MAX_OUTER) -> int:
    """The most renders `solve` makes: the planned render and the true-peak ceiling
    render come before the max_outer checks, then SLOPE_STEPS and LOUDNESS_STEPS."""
    return min(2 + SLOPE_STEPS + LOUDNESS_STEPS, max(max_outer, 2))


def solve(x: np.ndarray, fs: int, plan0: Plan, targets: planner.Targets,
          max_outer: int = MAX_OUTER, log=None, chain: Chain | None = None,
          progress=None) -> SolveResult:
    """Chunk 4 loop (see module contract)."""
    own_chain = chain is None
    chain = chain or Chain(x, fs, plan0, sections=targets.sections)
    if not chain.sections and targets.sections:
        chain.sections = list(targets.sections)
    try:
        return _solve(x, fs, plan0, targets, max_outer, log, chain, progress)
    finally:
        if own_chain:
            chain.close()


def _solve(x, fs, plan0, targets, max_outer, log, chain, progress=None) -> SolveResult:
    t_start = time.perf_counter()
    say = log or (lambda *_: None)
    say(f"input, inputeq and stages 1-2 cached in {chain.pre_s:.1f} s")
    # crest reference: the premaster as the chain sees it (after the input gain), so the
    # gated crest frames do not depend on the premaster's level
    ref = Reference(x if plan0.input is None else
                    np.asarray(gain_stage.apply(x, fs, plan0.input), dtype=np.asarray(x).dtype),
                    fs, targets)
    lv = Levers.from_plan(plan0)
    history = []
    bias = {"rms": 0.0}
    renders = 0
    ceiling_now = {"v": lv.limiter_ceiling_dbtp}

    def tp_correct(m, rendered_ceiling):
        """Lower the limiter ceiling for later renders by the constrained meter's overshoot
(audit 07 A14; since 2026-10-20 the limiter's re-limit holds the 4x Annex 2 envelope
        as well as 8x and 32x, so this fires only on the quantised file's rounding).
        Never touches a rendered state's levers."""
        over = m["true_peak_constrained_dbtp"] - targets.true_peak_max_dbtp
        if over > 1e-9:
            ceiling_now["v"] = min(ceiling_now["v"],
                                   float(rendered_ceiling - over - TP_MARGIN_DB))

    def drop(st):
        for k in ("z", "y"):
            try:
                st[k].unlink()
            except OSError:
                pass

    def render(lv, note):
        """Pre-limiter (cached), the gain for the RMS target on the calibrated proxy,
        clamped to the limiter's budget, blocked limiter, measurement. Returns a state
        dict with the master and pre-limiter audio spilled to disk."""
        nonlocal renders
        if progress is not None:
            progress(renders + 1)
        plan, z = chain.pre_limiter(lv)
        ginfo = {}
        req = gain_for_rms(z, fs, plan.limiter, targets.rms_dbfs, lv.drive_db, bias["rms"],
                           info=ginfo)
        plan.limiter = planner.clamp_limiter_gain(plan.limiter, req)
        if ginfo.get("at_limit"):
            for w in ((plan.limiter.budget or {}).get("limiter") or {}).get("warnings") or []:
                w["request_at_drive_limit"] = True
        lv.drive_db = plan.limiter.input_gain_db
        y, info = limit_blocked(z, fs, plan.limiter)
        y = finish(y, fs, plan)
        renders += 1
        m = measure(y, fs, plan, info, ref, z)
        m["section_change_by_stage"] = stage_changes(
            chain.stage_sections + ([("limiter, output", m["section_lufs"])]
                                    if m["section_lufs"] else []), ref)
        pr = proxy_rms(z, fs, plan.limiter, lv.drive_db)
        if math.isfinite(m["rms_dbfs"]) and math.isfinite(pr):
            bias["rms"] = m["rms_dbfs"] - pr
        st = {"lv": copy.deepcopy(lv), "plan": plan, "info": info, "m": m,
              "score": lever_score(m), "full_score": score(m), "clamped": _clamped(plan),
              "note": note,
              "z": chain._put(f"z{renders}", z), "y": chain._put(f"y{renders}", y)}
        tp_correct(m, lv.limiter_ceiling_dbtp)
        history.append({"render": renders, "move": note, "levers": asdict(lv),
                        "requested_gain_db": float(req), "clamped": st["clamped"],
                        "rms_dbfs": m["rms_dbfs"], "s95": m["s95_lufs"],
                        "slope": m["slope_db_oct"],
                        "true_peak": m["true_peak_constrained_dbtp"], "score": st["score"],
                        "full_score": st["full_score"],
                        "section_contrast_loss_lu": m["section_contrast_loss_lu"],
                        "request_at_drive_limit": bool(ginfo.get("at_limit")),
                        "misses": [c["name"] for c in misses(m["constraints"])],
                        "t_s": time.perf_counter() - t_start})
        say(f"render {renders} ({note}): RMS {m['rms_dbfs']:.2f} dBFS, S95 "
            f"{m['s95_lufs']:.2f}, slope {m['slope_db_oct']:.2f}, limiter gain "
            f"{lv.drive_db:+.2f} dB" + (" (clamped)" if st["clamped"] else ""))
        return st

    def ok(st, name):
        return next((c for c in st["m"]["constraints"] if c["name"] == name), {"ok": True})["ok"]

    def on_target(st):
        """The RMS within target +- tolerance."""
        c = next((c for c in st["m"]["constraints"] if c["name"] == "rms"), None)
        return c is None or c["excess"] <= 1e-9

    kept = render(lv, "planned")
    if ceiling_now["v"] < lv.limiter_ceiling_dbtp - 1e-12:
        # the constrained true-peak meter read over: a ceiling-correction render of its
        # own, which becomes the baseline the rotation is judged against
        lv.limiter_ceiling_dbtp = ceiling_now["v"]
        st = render(lv, "true-peak ceiling")
        history.append({"kept": "true-peak ceiling", "render": renders,
                        "limiter_ceiling_dbtp": lv.limiter_ceiling_dbtp})
        drop(kept)
        kept = st

    def slope_error(st):
        """Target minus measured slope when the slope constraint is violated and the input
        EQ can be refitted, else None."""
        c = next((c for c in st["m"]["constraints"] if c["name"] == "slope"), None)
        if c is None or c["ok"] or not _rotatable(plan0):
            return None
        return targets.slope_db_oct - st["m"]["slope_db_oct"]

    # slope first: the tone rotation is the only lever that moves the slope
    for _ in range(SLOPE_STEPS):
        err = slope_error(kept)
        if err is None or renders >= max_outer:
            break
        lv_t = copy.deepcopy(kept["lv"])
        lv_t.limiter_ceiling_dbtp = ceiling_now["v"]
        rot = float(np.clip(lv_t.tone_rotation + err, -TONE_ROT_MAX, TONE_ROT_MAX))
        if abs(rot - lv_t.tone_rotation) < 1e-9:
            break
        if _same_refit(plan0, lv_t.tone_rotation, rot):
            # S7 (sprint 2026-10-21): every section the rotation moves is at its bound
            history.append({"skipped": "tone rotation", "reason": "EQ at its limit",
                            "tone_rotation": rot})
            break
        lv_t.tone_rotation = rot
        st = render(lv_t, "tone rotation")
        rec = {"render": renders, "score_before": kept["score"], "score_after": st["score"],
               "tone_rotation": rot}
        if st["score"] < kept["score"]:
            history.append({"kept": "tone rotation", **rec})
            drop(kept)
            kept = st
        else:
            history.append({"reverted": "tone rotation", **rec})
            drop(st)
            break

    # loudness and true peak: re-render on the recalibrated proxy (and the corrected
    # ceiling) while the RMS is off target with gain left in the budget, or the true
    # peak reads over; kept when the full score does not rise and, once the loudness is
    # accepted, the render does not flatten the sections past the flag (R13)
    contrast_revert = False
    for _ in range(LOUDNESS_STEPS):
        want_rms = not on_target(kept) and not kept["clamped"]
        if not (want_rms or not ok(kept, "true_peak")) or renders >= max_outer:
            break
        lv_c = copy.deepcopy(kept["lv"])
        lv_c.limiter_ceiling_dbtp = ceiling_now["v"]
        st = render(lv_c, "loudness" if want_rms else "true-peak ceiling")
        rec = {"render": renders, "score_before": kept["full_score"],
               "score_after": st["full_score"],
               "contrast_loss_before_lu": kept["m"]["section_contrast_loss_lu"],
               "contrast_loss_after_lu": st["m"]["section_contrast_loss_lu"]}
        why = judge_move(st["m"], kept["m"], ok(kept, "rms"))
        if why is None:
            history.append({"kept": st["note"], **rec})
            drop(kept)
            kept = st
        else:
            history.append({"reverted": st["note"], **rec, "reason": why})
            contrast_revert = contrast_revert or why == "section contrast"
            drop(st)
            break

    lv, plan, info, m = copy.deepcopy(kept["lv"]), kept["plan"], kept["info"], kept["m"]
    z = chain._get(kept["z"])
    y = chain._get(kept["y"])
    rms_c = next(c for c in m["constraints"] if c["name"] == "rms")
    final_plan = replace(plan, limiter=replace(plan.limiter, input_gain_db=lv.drive_db))
    if _clamped(final_plan):
        # the measured shortfall replaces the planner's proxy estimate
        lr = final_plan.limiter.budget["limiter"]
        short = (float(targets.rms_dbfs - m["rms_dbfs"]) if math.isfinite(m["rms_dbfs"])
                 else None)
        # S5: what the budget would have had to allow, from the measured shortfall (a
        # lower bound: the RMS rises at most 1 dB per dB of gain), not the proxy's request
        need = ({} if short is None else
                {"needed_db": float(lr.get("planned_gr_db") or 0.0) + max(short, 0.0),
                 "outlier_needed_db": float(lr.get("planned_outlier_gr_db") or 0.0)
                 + max(short, 0.0), "needed_source": "render"})
        lr["warnings"] = [dict(w, shortfall_db=short, shortfall_source="render",
                               target_rms_dbfs=targets.rms_dbfs, achieved_rms_dbfs=m["rms_dbfs"],
                               accepted=bool(rms_c["ok"]), **need)
                          for w in lr["warnings"]]
    m["budget"] = planner.budget_report(final_plan, info)
    warns = list(m["budget"]["warnings"])
    if not rms_c["ok"]:
        warns.append({"stage": "loudness", "target_dbfs": targets.rms_dbfs,
                      "achieved_dbfs": m["rms_dbfs"],
                      "shortfall_db": float(targets.rms_dbfs - m["rms_dbfs"])
                      if math.isfinite(m["rms_dbfs"]) else None,
                      "cause": "limiter budget" if _clamped(final_plan) else
                      "section contrast" if contrast_revert else "solver"})
    slope_c = next((c for c in m["constraints"] if c["name"] == "slope"), None)
    if slope_c is not None and not slope_c["ok"]:
        warns.append(slope_warning(plan0, lv.tone_rotation, m["slope_db_oct"], targets))
    miss = misses(m["constraints"])
    # a limiter held at its budget is reported, but fails nothing while the RMS is within
    # its tolerance
    blocking = [w for w in warns if not (w.get("stage") == "limiter" and w.get("accepted"))]
    res = SolveResult(final_plan, y, info, m, lv, renders, history, not miss and not blocking,
                      miss, warns, time.perf_counter() - t_start, z)
    res.history.append({"renders": renders, "clamped": _clamped(final_plan)})
    return res

