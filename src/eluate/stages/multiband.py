"""M/S multiband compressor with per-band ceilings (spec 3.5 stage 3, 3.4).

Contract: `apply(x, fs, plan) -> y`, pure, stereo (n, 2) in, float64 (n, 2) out.
- M, S = bands.to_ms(x); bands.split into len(crossovers) + 1 bands (default
  bands.DEFAULT_CROSSOVERS, 100 / 400 / 1600 / 6400 Hz, five bands, zero phase unless `phase` says
  otherwise per crossover).
- Band k, channel c is multiplied by
      10^((clip(GR_kc(t), -MULTIBAND_MAX_GR_DB, 0) + makeup_kc + C_kc(t)) / 20)
  GR_kc is the compressor envelope, C_kc the ceiling (per-band limiter)
  envelope, both the plan's 1 ms envelopes interpolated by PCHIP. The hard cap
  (MULTIBAND_MAX_GR_DB = 6 dB since D5, a per-sample clamp: the budgets read the GR
  as a meter shows it, maintainer 2026-10-19, and the budget's outlier cap is
  MULTIBAND_OUTLIER_CAP_DB on that reading) is the compressor's and holds on GR at
  every audio sample
  whatever the plan holds; C is not capped (reported separately,
  plan.ceiling_gr_db, and in `apply_with_info` with the total) and is absent
  when plan.ceiling_gr_db is None. apply runs plan.check_ceiling() first: a
  ceiling envelope above 0 dB, or a makeup different from the one the ceiling
  was designed with, raises ValueError (audit 06 M7).
- Makeup: M bands take plan.mid_makeup_db (planner: toward the tone targets).
  S makeup is fixed by the maintainer's rule (MultibandPlan.side_makeup): bands whose
  geometric centre (plan.band_centres) is at or above 2 kHz (high-mid, air:
  widens the top) get side_makeup_db (default +2 dB) * (1 - decoherence_k),
  scaled back when the band is already decoherent; every other band, sub and
  low included (narrows the bottom), gets 0.
- ms_link (MultibandPlan.ms_link, the live profile): M and S of a band take one gain,
  the larger reduction of the two at each instant, with the M makeup. The budget is
  designed per channel, so a channel's applied GR can exceed its own budget's reading.
- Bands are summed and converted back to L/R. With every GR, ceiling and makeup
  at 0 dB the output equals the input to double precision (zero-phase bands
  sum exactly); with a minimum-phase crossover the sum is an allpass
  (bands.py). A band whose gain is identically 1 is passed bit-identically, so
  the sum of the untouched bands is exactly the input's.

`design(x, fs, threshold_db, ...)` builds both envelope sets with envelope.py:
- Compressor: detector at control point k (t = k ms) = band peak over the two
  control segments either side of it (segment j = the samples Envelope.at
  interpolates between points j and j + 1, assigned with the same
  control_coord, audit 05 A9), required gain = (T / peak)^(1 - 1/ratio)
  floored at -MULTIBAND_MAX_GR_DB, then envelope.box_smooth (lookahead attack window, at least
  two control samples; release hull) at the 1 ms rate (ratio must be > 1),
  rounded to float32 toward more reduction. The detector covering both
  neighbouring segments makes g <= g_req hold on every interpolated segment.
  Reduction starts at most attack + one control period before the first
  sample over the threshold (a sample in segment k sets point k).
- Ceiling: the same detector run on the band *after* the compressor and makeup
  exactly as apply() computes them (the rounded compressor envelope through
  Envelope.at and the cap), required gain = min(1, ceiling / peak), then
  box_smooth with a short lookahead (ceiling_lookahead_ms, a scalar or one value
  per band: the character preset's band_lookahead_ms, longer on the low bands so a
  cut there is not timed inside the waveform; at least two
  control samples) and a per-band release hull (ceiling_release_ms, bands x
  [M, S]), rounded to float32 toward more reduction. Since PCHIP stays between
  its two neighbouring control values and both are at or below the required
  gain of every sample between them, the processed band's sample peak is at
  or below its ceiling at every sample, up to the rounding of the final
  product (gate 0.05 dB; measured below 5e-7 dB) and **provided the required
  gain stays above envelope.G_MIN = 1e-6 (-120 dB)**: box_smooth floors gains
  there, so a ceiling more than 120 dB below the processed band peak cannot be
  met, and design raises ValueError instead (audit 06 F4, M9).
  The bound is on the band as the stage processes it; a re-split of the
  output reads differently (spec 3.6; the solver iterates on the re-split).
- attack_ms, release_ms, thresholds, ceilings and ceiling releases broadcast
  to bands x [M, S] (the ceiling lookahead is per band, M and S alike, and is
  recorded as given, a scalar or the per-band list) and are recorded in the plan
  with the makeup the ceiling
  saw and the design span (s), so the solver can vary them per band and
  rebuild with `design_from(plan, x, fs)`.

- knee_db (scalar or bands x [M, S], default 0): with a knee > 0 the compressor's
  required gain comes from the soft-knee static curve (Giannoulis, Massberg and
  Reiss 2012, as the glue's) on the detector level in dBFS; knee 0 keeps the
  original (T / peak)^(1 - 1/ratio). ratio may also be bands x [M, S].
- mode_makeup_db (chunk 3): per band and channel makeup envelopes added after
  the ceiling (see design_budget); the ceiling detector does not see them.

Budget solve (docs/gain-staging.md 3 and 5, gain-staging chunk 3; the average-peak
reading, maintainer 2026-10-19). Every solve takes `hits` (envelope.hits_for: the hit
edges and the loud section of the stage's input); the GR it solves for is the
average peak GR (envelope.stage_gr: the GR averaged over 300 ms as a meter shows it,
then the median over the loud hits of each hit's maximum), and the outlier is the
maximum over the track on the same reading. hits None makes the whole track one hit,
so the average is the maximum: the chunk 3 reading. The pure solves bisect on the
envelope design above, on the band as the stage splits it:
- `threshold_for_peak_gr(v, fs, target_db, ratio, attack_ms, release_ms, knee_db,
  hits=None)`: the compressor threshold whose (capped, float32-rounded) envelope
  has an average peak GR of target_db on band signal v; returns its outlier as
  well (reported against MULTIBAND_OUTLIER_CAP_DB; only the per-sample clamp
  MULTIBAND_MAX_GR_DB holds it).
- `ceiling_for_peak_gr(v, fs, target_db, lookahead_ms, release_ms)`: the band
  limiter ceiling whose envelope has a peak GR (maximum) of target_db on v (the
  band after compressor and makeup, the ceiling detector's input). The band
  limiter stays an outlier catcher, so it is solved on the maximum.
- `band_crest_db(v, fs)`: the band's peak-to-loud-RMS crest, the p99 of the
  400 ms frame peak (100 ms hop) minus the p95 of the 1 s RMS (100 ms hop,
  windows above -70 dBFS), the corpus metric of section 9. `ratio_for_crest`
  maps it through the section 5 table: the band's corpus p10 crest gives 2:1,
  its p90 1.3:1, linear in between, clamped to [1.3, 2.0]; M and S each on
  their own corpus table (the S table since chunk 4).
- `limit_need_db(v, fs)`: how far the band's highest peak sits above the p99
  of its 400 ms frame peaks: the peaks that get past the compressor, which the
  band limiter takes.
- `design_budget(x, fs, plan, hits=None)`: per band and channel, the ratio from
  the crest, the compressor threshold for an average peak GR of
  budget["comp_gr_db"] (2.5 dB, on the meter reading: envelope.stage_gr), then the
  limiter for the need, solved on its per-sample peak with the band's own lookahead
  (plan.lookahead_ms()), at most limit_max_db (the
  combined meter reading is reported against total_normal_db). Only where the need
  exceeds that (the exception) is the
  compressor re-solved at comp_gr_max_db (3 dB) and the limiter allowed up to
  limit_max_db (3 dB, a maximum) more; a band that needs more still is
  rendered at the caps and gets a warning {"stage": "multiband", "band",
  "channel": "M" | "S" | "both", "needed_db", "cap_db"}. A compressor whose
  average falls more than envelope.AVERAGE_WARN_DB (0.5 dB) below its target
  (the outlier clamp reaching into the loud hits) warns {"stage": "multiband",
  "band", "channel", "kind": "average", "target_db", "achieved_db"}. The
  envelopes are then built by `design` with the solved values, so they are
  exactly the ones solved. Makeup mode budget["makeup"]: "peak" = + the band's
  combined average peak GR (compressor + limiter; the maximum without hits),
  "rms" = + the mean combined GR over the control points whose detector level
  is above -60 dBFS, "auto" = a time-varying makeup following the 3 s centred
  moving average of the combined GR. The mid and side makeup rules apply on top.
  A band's M and S solves are independent and run on eluate.parallel.pmap (the
  same values at any thread count).

`crest_loss(before, after, fs, ...)`: the solver's per-band crest constraint
(spec 3.6) on the spec 3.3 crest definition; see its docstring.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from eluate import bands, envelope, meter, parallel
from eluate.plan import (CONTROL_RATE_HZ, MAKEUP_MODES, MODE_MAKEUP_MAX_DB,
                            MULTIBAND_LIMIT_MAX_GR_DB, MULTIBAND_MAX_GR_DB,
                            MULTIBAND_OUTLIER_CAP_DB, Envelope,
                            MultibandPlan, control_coord, control_length)


ATTACK_FLOOR = 2  # control samples

# spec 3.6 crest-loss tolerances for the default five bands, dB
CREST_LOSS_LIMITS_DB = (1.5, 2.0, 3.0, 3.0, 3.0)
CREST_WINDOW_S = 0.4
CREST_HOP_S = 0.1
CREST_GATE_DBFS = -60.0
CREST_EDGE_S = 0.25
CREST_VIEWS = ("M", "S", "stereo")
CREST_HYST_DB = 10.0          # "silenced" = below gate - hysteresis after processing
CREST_SILENCED_FRACTION = 0.5  # of the frames gated before

# Budget solve (docs/gain-staging.md 3, 5; chunk 3)
COMP_GR_DB = 2.5                # compressor average peak GR, normal range 2 to 3
COMP_GR_MAX_DB = 3.0            # compressor average peak GR in the exception
TOTAL_NORMAL_DB = 3.0           # comp + band limiter, normal tally
RATIO_MAX, RATIO_MIN = 2.0, 1.3
# corpus band crest (dB, 44 premasters, crossovers 100 / 400 / 1600 / 6400), M and S
# (the S values: maintainer, 2026-10-18): the p10 maps to RATIO_MAX, the p90 to RATIO_MIN;
# bands past the table use the last row
CREST_P10_DB = (8.5, 13.5, 15.7, 19.8, 20.6)
CREST_P90_DB = (12.0, 17.4, 20.4, 25.4, 27.5)
CREST_P10_S_DB = (11.1, 12.4, 13.4, 16.7, 21.8)
CREST_P90_S_DB = (17.0, 17.4, 19.4, 26.5, 32.3)
BAND_PEAK_FRAME_S = 0.4
BAND_RMS_WINDOW_S = 1.0
BAND_HOP_S = 0.1
BAND_RMS_GATE_DBFS = -70.0
LIMIT_NEED_PERCENTILE = 99.0
RMS_MAKEUP_GATE_DBFS = -60.0
# a band channel whose loudest-section 1 s RMS sits this far below the stage input's is
# left uncompressed (e.g. the low S band after the side strip: about 34 dB below)
GATE_BELOW_INPUT_DB = 24.0       # D8: at 30 the side top band was compressed 7 to 12 dB
SPECLIM_GATE_BELOW_INPUT_DB = 30.0   # speclim's (WP4): a stray peak in a quiet band is its job
AUTO_MAKEUP_S = 3.0
SOLVE_TOL_DB = 1e-4
SOLVE_ITER = 80
BRACKET_STEP_DB = 6.0
BRACKET_MAX_DB = 48.0


def _gains_db(plan: MultibandPlan, n: int, fs: float):
    """Per band, lazily: [(capped compressor GR, makeup, ceiling GR or None) for M, S],
    dB at audio rate. A generator, so only one band's four audio-rate curves are
    alive at a time (all twenty were 3.2 GB on a 7-minute 48 kHz track)."""
    makeup = plan.makeup()
    for k in range(len(plan.gr_db)):
        row = []
        for c in range(2):
            gr = np.clip(plan.gr_db[k][c].at(n, fs), -MULTIBAND_MAX_GR_DB, 0.0)
            cg = None if plan.ceiling_gr_db is None else plan.ceiling_gr_db[k][c].at(n, fs)
            mk = makeup[k][c]
            if plan.mode_makeup_db is not None:
                mk = mk + plan.mode_makeup_db[k][c].at(n, fs)
            row.append((gr, mk, cg))
        if plan.ms_link:
            red = [g if cg is None else g + cg for g, _, cg in row]
            link = np.minimum(red[0], red[1])
            row = [(link, row[0][1], None), (link, row[0][1], None)]
        yield row


def _check_stereo(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("multiband needs stereo (n, 2) audio")
    return x


def _stream(ms: np.ndarray, n: int, fs: float, plan: MultibandPlan) -> np.ndarray:
    """The non-info path: bands streamed, one band, the residual and the running sum
    alive, not all bands; the sum is accumulated in band order (in place), as
    b.sum(axis=0) adds the slices of its outer axis, so the output is bit-identical.
    Gains one channel at a time and in place: the same operations as
    10 ** ((gr + cg + makeup) / 20) in _gains_db's form."""
    acc = None
    makeup = plan.makeup()
    it = bands.split_iter(ms, fs, plan.crossovers_hz, order=plan.order, phase=plan.phase)
    del ms
    for k, band in enumerate(it):
        if plan.ms_link:
            _apply_linked(band, k, n, fs, plan, makeup)
        else:
            for c in range(2):
                total = np.clip(plan.gr_db[k][c].at(n, fs), -MULTIBAND_MAX_GR_DB, 0.0)
                if plan.ceiling_gr_db is not None:
                    total += plan.ceiling_gr_db[k][c].at(n, fs)
                if plan.mode_makeup_db is not None:
                    total += plan.mode_makeup_db[k][c].at(n, fs)
                total += makeup[k][c]
                if np.any(total):
                    total /= 20.0
                    np.power(10.0, total, out=total)
                    band[:, c] *= total
                del total
        if acc is None:
            acc = band
        else:
            acc += band
        del band
    return bands.to_lr(acc)


def _apply_linked(band: np.ndarray, k: int, n: int, fs: float, plan: MultibandPlan,
                  makeup) -> None:
    """ms_link: M and S of band k take one gain, the larger reduction of the two, with
    the M makeup (MultibandPlan.ms_link), in place."""
    red = []
    for c in range(2):
        r = np.clip(plan.gr_db[k][c].at(n, fs), -MULTIBAND_MAX_GR_DB, 0.0)
        if plan.ceiling_gr_db is not None:
            r += plan.ceiling_gr_db[k][c].at(n, fs)
        red.append(r)
    total = np.minimum(red[0], red[1])
    del red
    if plan.mode_makeup_db is not None:
        total += plan.mode_makeup_db[k][0].at(n, fs)
    total += makeup[k][0]
    if np.any(total):
        band *= (10.0 ** (total / 20.0))[:, None]


def _process(x: np.ndarray, fs: float, plan: MultibandPlan, info: bool):
    x = _check_stereo(x)
    plan.check_ceiling()
    n = x.shape[0]
    if not info:
        return _stream(bands.to_ms(x), n, fs, plan), None
    b = bands.split(bands.to_ms(x), fs, plan.crossovers_hz, order=plan.order, phase=plan.phase)
    nb = b.shape[0]
    meta = None
    if info:
        meta = {k: np.zeros((nb, 2)) for k in
                ("comp_gr_min_db", "ceiling_gr_min_db", "total_gr_min_db", "total_gain_min_db",
                 "makeup_db", "peak_in_dbfs", "peak_out_dbfs")}
        meta["ceiling_db"] = [[None if plan.ceiling_db is None else plan.ceiling_db[k][c]
                               for c in range(2)] for k in range(nb)]
    for k, row in enumerate(_gains_db(plan, n, fs)):
        for c, (gr, makeup, cg) in enumerate(row):
            red = gr if cg is None else gr + cg
            total = red + makeup
            if info:
                meta["peak_in_dbfs"][k, c] = _db(np.max(np.abs(b[k, :, c])))
                meta["comp_gr_min_db"][k, c] = float(gr.min())
                meta["makeup_db"][k, c] = float(np.max(makeup))
                meta["ceiling_gr_min_db"][k, c] = 0.0 if cg is None else float(cg.min())
                meta["total_gr_min_db"][k, c] = float(red.min())
                meta["total_gain_min_db"][k, c] = float(total.min())
            if np.any(total):
                b[k, :, c] *= 10.0 ** (total / 20.0)
            if info:
                meta["peak_out_dbfs"][k, c] = _db(np.max(np.abs(b[k, :, c])))
    if info:
        meta["bands_out"] = b.copy()
    return bands.to_lr(b.sum(axis=0)), meta


def apply(x: np.ndarray, fs: float, plan: MultibandPlan) -> np.ndarray:
    x = _check_stereo(x)
    plan.check_ceiling()
    n = x.shape[0]
    ms = bands.to_ms(x)
    del x                   # released once in M/S (freed when the caller passed a temporary)
    return _stream(ms, n, fs, plan)


def apply_with_info(x: np.ndarray, fs: float, plan: MultibandPlan):
    """apply() plus metadata, (bands, 2) arrays over [M, S]: comp_gr_min_db (deepest
    capped compressor GR, the quantity spec 3.6 caps at 3 dB), ceiling_gr_min_db
    (deepest ceiling GR, 0 without ceilings), total_gr_min_db (deepest compressor
    plus ceiling reduction, makeup excluded), total_gain_min_db (the same with
    makeup), makeup_db, peak_in_dbfs / peak_out_dbfs (sample peak of each M or S
    band before and after processing, in the stage's own split), ceiling_db (the
    plan's, None where absent), and bands_out (bands, n, 2) M/S, whose sum over
    bands is the output in M/S."""
    return _process(x, fs, plan, info=True)


def _db(v: float) -> float:
    return float(20.0 * np.log10(v)) if v > 0 else -np.inf


def _frame_peaks(v: np.ndarray, fs: float, n_ctrl: int, rate_hz: float) -> np.ndarray:
    """Peak |v| per control segment k: the samples whose control coordinate
    (plan.control_coord, the mapping Envelope.at uses) lies in [k, k + 1)."""
    n = v.shape[0]
    k = np.minimum(np.floor(control_coord(np.arange(n), fs, rate_hz)).astype(np.int64),
                   n_ctrl - 1)
    peaks = np.zeros(n_ctrl)
    starts = np.flatnonzero(np.diff(k, prepend=-1))
    peaks[k[starts]] = np.maximum.reduceat(np.abs(v), starts)
    return peaks


def detector(v: np.ndarray, fs: float, n_ctrl: int, rate_hz: float) -> np.ndarray:
    """Control-point detector: peak over the segments either side of each point."""
    pk = _frame_peaks(v, fs, n_ctrl, rate_hz)
    return np.maximum(pk, np.concatenate([[0.0], pk[:-1]]))


def to_env32(g_lin: np.ndarray, rate_hz: float) -> Envelope:
    """dB envelope from a linear gain, float32-representable, rounded toward more reduction."""
    g_db = 20.0 * np.log10(g_lin)
    v32 = g_db.astype(np.float32)
    v32 = np.where(v32 > g_db, np.nextafter(v32, np.float32(-np.inf)), v32)
    return Envelope(np.minimum(v32.astype(np.float64), 0.0), rate_hz)


def _lookahead(v, nb: int) -> list[float]:
    """Per-band ceiling lookahead (ms) from a scalar or one value per band."""
    if np.ndim(v) == 0:
        return [float(v)] * nb
    a = [float(t) for t in v]
    if len(a) != nb:
        raise ValueError(f"ceiling_lookahead_ms must be a scalar or {nb} values")
    return a


def _table(v, nb: int, name: str) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    if a.ndim == 1 and a.size == nb:
        a = a[:, None]
    try:
        return np.array(np.broadcast_to(a, (nb, 2)))
    except ValueError:
        raise ValueError(f"{name} must broadcast to {nb} bands x [M, S]") from None


def knee_gr_db(level_db, threshold_db: float, ratio: float, knee_db: float) -> np.ndarray:
    """Soft-knee static curve: gain reduction in dB (<= 0) for a detector level in dBFS
    (the glue's gain computer)."""
    lv = np.asarray(level_db, dtype=np.float64)
    over = lv - threshold_db
    slope = 1.0 / ratio - 1.0
    gr = np.where(over > 0.0, slope * over, 0.0)
    if knee_db > 0.0:
        half = knee_db / 2.0
        gr = np.where(np.abs(over) <= half, slope * (over + half) ** 2 / (2.0 * knee_db), gr)
        gr = np.where(over < -half, 0.0, gr)
    return np.minimum(gr, 0.0)


def comp_envelope(det: np.ndarray, threshold_db: float, ratio: float, attack_ms: float,
                  release_ms: float, knee_db: float = 0.0,
                  rate_hz: float = CONTROL_RATE_HZ) -> Envelope:
    """One band and channel's compressor envelope from its detector (module contract)."""
    floor = 10.0 ** (-MULTIBAND_MAX_GR_DB / 20.0)
    if knee_db <= 0.0:
        g_lin = envelope.required_gain(det, 10.0 ** (threshold_db / 20.0))
        g_req = np.maximum(g_lin ** (1.0 - 1.0 / ratio), floor)
    else:
        with np.errstate(divide="ignore"):
            level = 20.0 * np.log10(det)
        g_req = np.maximum(10.0 ** (knee_gr_db(level, threshold_db, ratio, knee_db) / 20.0),
                           floor)
    # floor of two control samples: box_smooth with attack < 2 does no
    # smoothing (audit 05 Q13)
    attack = max(ATTACK_FLOOR, int(round(attack_ms * rate_hz / 1000.0)))
    g = envelope.box_smooth(g_req, attack=attack, release=release_ms * rate_hz / 1000.0)
    return to_env32(g, rate_hz)


def ceiling_envelope(det: np.ndarray, ceiling_db: float, lookahead_ms: float,
                     release_ms: float, rate_hz: float = CONTROL_RATE_HZ) -> Envelope:
    """One band and channel's ceiling envelope from the detector of the band after the
    compressor and makeup (module contract)."""
    g_req = envelope.required_gain(det, 10.0 ** (ceiling_db / 20.0))
    look = max(ATTACK_FLOOR, int(round(lookahead_ms * rate_hz / 1000.0)))
    g = envelope.box_smooth(g_req, attack=look, release=release_ms * rate_hz / 1000.0)
    env = to_env32(g, rate_hz)
    return env if np.any(env.values) else Envelope.constant(0.0)


def comp_peak_gr_db(env: Envelope) -> float:
    """Peak GR (dB >= 0) of a compressor envelope as the stage applies it (capped)."""
    return float(-max(np.min(env.values), -MULTIBAND_MAX_GR_DB)) + 0.0


def bisect_peak_gr(f, lo: float, hi: float, target: float):
    """x in [lo, hi] with f(x) = target, f non-increasing in x (more threshold, less GR).
    Returns (x, f(x)) of the evaluation closest to the target; shared by the budget
    solves of every stage."""
    best = (hi, f(hi))
    flo = f(lo)
    if abs(flo - target) < abs(best[1] - target):
        best = (lo, flo)
    for _ in range(SOLVE_ITER):
        if abs(best[1] - target) < SOLVE_TOL_DB or hi - lo < 1e-9:
            break
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm - target) < abs(best[1] - target):
            best = (mid, fm)
        if fm > target:
            lo = mid
        else:
            hi = mid
    return best


def _loud_top_db(det: np.ndarray, rate_hz: float, hits: dict | None) -> float:
    """The lowest loud hit's detector peak (dBFS; the track's peak without hits): every
    loud hit reaches a GR target once the threshold is that far below it."""
    if hits is None:
        return float(20.0 * np.log10(det.max()))
    hv = envelope.hit_values(det, rate_hz, hits["edges_s"])
    sel = np.isfinite(hv) & np.asarray(hits["loud"], dtype=bool) & (hv > 0.0)
    if not np.any(sel):
        return float(20.0 * np.log10(det.max()))
    return float(20.0 * np.log10(hv[sel].min()))


def threshold_for_peak_gr(v: np.ndarray, fs: float, target_db: float, ratio: float,
                          attack_ms: float, release_ms: float, knee_db: float = 0.0,
                          rate_hz: float = CONTROL_RATE_HZ, hits: dict | None = None) -> dict:
    """Compressor threshold (dBFS) giving an average peak GR of target_db on band signal
    v over hits (the maximum without hits), by bisection on comp_envelope (module
    contract). Returns {"threshold_db", "peak_gr_db" (the average peak GR reached),
    "outlier_gr_db"}; a silent band gives 0 dBFS and 0 dB, a zero target a threshold
    just above the knee's reach."""
    v = np.asarray(v, dtype=np.float64)
    if not 0.0 <= target_db <= MULTIBAND_MAX_GR_DB:
        raise ValueError(f"target_db must lie in [0, {MULTIBAND_MAX_GR_DB}]")
    det = detector(v, fs, control_length(v.shape[0], fs, rate_hz), rate_hz)
    pk = float(det.max()) if det.size else 0.0
    if pk <= 0.0:
        return {"threshold_db": 0.0, "peak_gr_db": 0.0, "outlier_gr_db": 0.0}
    top = 20.0 * np.log10(pk)
    if target_db <= 0.0:
        return {"threshold_db": float(top + knee_db / 2.0 + 0.01), "peak_gr_db": 0.0,
                "outlier_gr_db": 0.0}
    lo = _loud_top_db(det, rate_hz, hits) - target_db / (1.0 - 1.0 / ratio) - knee_db - 1.0
    hi = top + knee_db / 2.0 + 0.01
    # at the cap every lower threshold reads the cap too: aim just under it so the solve
    # returns the highest threshold that reaches it (the least compression elsewhere)
    aim = min(target_db, MULTIBAND_MAX_GR_DB - 1e-3)
    last = {}

    def gr(t):
        env = comp_envelope(det, t, ratio, attack_ms, release_ms, knee_db, rate_hz)
        r = envelope.stage_gr(np.clip(-env.values, 0.0, MULTIBAND_MAX_GR_DB), rate_hz, hits)
        last[t] = r["outlier_db"]
        return r["average_db"]

    # the meter reading averages the GR, so the target can need a threshold below the
    # peak-reading bracket: step down until it is reached (at most BRACKET_MAX_DB)
    floor = lo - BRACKET_MAX_DB
    while gr(lo) < aim and lo > floor:
        lo -= BRACKET_STEP_DB
    t, g = bisect_peak_gr(gr, lo, hi, aim)
    return {"threshold_db": float(t), "peak_gr_db": float(g), "outlier_gr_db": float(last[t])}


def ceiling_for_peak_gr(v: np.ndarray, fs: float, target_db: float,
                        lookahead_ms: float = 2.0, release_ms: float = 50.0,
                        rate_hz: float = CONTROL_RATE_HZ) -> dict:
    """Band limiter ceiling (dBFS) giving a peak GR of target_db on v, the band after the
    compressor and makeup, by bisection on ceiling_envelope. Returns {"ceiling_db",
    "peak_gr_db"}; ceiling_db None for a silent band or a zero target."""
    v = np.asarray(v, dtype=np.float64)
    det = detector(v, fs, control_length(v.shape[0], fs, rate_hz), rate_hz)
    pk = float(det.max()) if det.size else 0.0
    if pk <= 0.0 or target_db <= 0.0:
        return {"ceiling_db": None, "peak_gr_db": 0.0}
    top = 20.0 * np.log10(pk)

    def gr(c):
        return float(-np.min(ceiling_envelope(det, c, lookahead_ms, release_ms,
                                              rate_hz).values))

    c, g = bisect_peak_gr(gr, top - target_db - 1.0, top + 0.01, target_db)
    return {"ceiling_db": float(c), "peak_gr_db": float(g)}


def _frames(n: int, fs: float, win_s: float, hop_s: float):
    """(starts, ends) of windows of win_s every hop_s (one window for a shorter signal)."""
    w = max(1, int(round(win_s * fs)))
    h = max(1, int(round(hop_s * fs)))
    if n <= w:
        return np.array([0]), np.array([n])
    a = np.arange(0, n - w + 1, h)
    return a, a + w


def _frame_peak_db(v: np.ndarray, fs: float) -> np.ndarray:
    """Sample peak (dBFS) of each 400 ms frame at a 100 ms hop."""
    a, e = _frames(v.shape[0], fs, BAND_PEAK_FRAME_S, BAND_HOP_S)
    if a.size == 1:
        pk = np.array([np.max(np.abs(v[a[0]:e[0]]))])
    else:
        w = int(e[0] - a[0])
        # a centred window of w at a + w // 2 covers [a, a + w)
        pk = maximum_filter1d(np.abs(v), size=w, mode="constant", cval=0.0)[a + w // 2]
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(pk)


def band_crest_db(v: np.ndarray, fs: float) -> float:
    """The band's peak-to-loud-RMS crest (module contract); NaN for a silent band."""
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0 or not np.any(v):
        return float("nan")
    pk = _frame_peak_db(v, fs)
    a, e = _frames(v.shape[0], fs, BAND_RMS_WINDOW_S, BAND_HOP_S)
    c = np.concatenate([[0.0], np.cumsum(v * v)])
    with np.errstate(divide="ignore"):
        rms = 10.0 * np.log10(np.maximum((c[e] - c[a]) / (e - a), 0.0))
    rms = rms[rms > BAND_RMS_GATE_DBFS]
    pk = pk[np.isfinite(pk)]
    if rms.size == 0 or pk.size == 0:
        return float("nan")
    return float(np.percentile(pk, 99.0) - np.percentile(rms, 95.0))


def ratio_for_crest(crest_db: float, band: int, channel: int = 0) -> float:
    """Section 5 map: the band's corpus p10 crest -> 2:1, p90 -> 1.3:1, linear between,
    clamped, on the M table (channel 0) or the S table (channel 1); a NaN crest (silent
    band) gives 2:1."""
    if not np.isfinite(crest_db):
        return RATIO_MAX
    p10, p90 = (CREST_P10_DB, CREST_P90_DB) if channel == 0 else (CREST_P10_S_DB, CREST_P90_S_DB)
    k = min(band, len(p10) - 1)
    lo, hi = p10[k], p90[k]
    t = float(np.clip((crest_db - lo) / (hi - lo), 0.0, 1.0))
    return float(RATIO_MAX + t * (RATIO_MIN - RATIO_MAX))


def limit_need_db(v: np.ndarray, fs: float) -> float:
    """dB by which the band's highest sample peak exceeds the LIMIT_NEED_PERCENTILE of its
    400 ms frame peaks (>= 0; 0 for a silent band)."""
    pk = _frame_peak_db(np.asarray(v, dtype=np.float64), fs)
    pk = pk[np.isfinite(pk)]
    if pk.size == 0:
        return 0.0
    return float(max(0.0, pk.max() - np.percentile(pk, LIMIT_NEED_PERCENTILE)))


def default_budget(makeup: str = "peak", ratio_from_crest: bool = True) -> dict:
    """The multiband budget record's targets (docs/gain-staging.md 3). A profile may
    override them and add "ratio_max", the highest ratio the crest map may give (the
    live profile: 1.5)."""
    if makeup not in MAKEUP_MODES:
        raise ValueError(f"makeup must be one of {MAKEUP_MODES}")
    return {"comp_gr_db": COMP_GR_DB, "comp_gr_max_db": COMP_GR_MAX_DB,
            "total_normal_db": TOTAL_NORMAL_DB, "limit_max_db": MULTIBAND_LIMIT_MAX_GR_DB,
            "makeup": makeup, "ratio_from_crest": bool(ratio_from_crest)}


def _ctrl(env: Envelope, m: int) -> np.ndarray:
    """An envelope's control values over m points (a constant envelope broadcast)."""
    v = env.values
    return np.full(m, float(v[0])) if v.size == 1 else v[:m]


def mode_makeup(gr_total_db: np.ndarray, gate: np.ndarray, mode: str,
                rate_hz: float = CONTROL_RATE_HZ, hits: dict | None = None) -> Envelope:
    """Makeup envelope (dB >= 0) of one band and channel from its combined compressor +
    limiter GR (dB <= 0) at the control rate and the detector gate (module contract);
    "peak" reads the average peak GR over hits (the maximum without hits)."""
    red = np.clip(-np.asarray(gr_total_db, dtype=np.float64), 0.0, MODE_MAKEUP_MAX_DB)
    if mode == "peak":
        return Envelope.constant(envelope.stage_gr(red, rate_hz, hits)["average_db"]
                                 if red.size else 0.0)
    if mode == "rms":
        return Envelope.constant(float(red[gate].mean()) if np.any(gate) else 0.0)
    if mode != "auto":
        raise ValueError(f"makeup must be one of {MAKEUP_MODES}")
    w = max(1, int(round(AUTO_MAKEUP_S * rate_hz)))
    m = np.clip(uniform_filter1d(red, size=w, mode="nearest"), 0.0, MODE_MAKEUP_MAX_DB)
    return Envelope(m, rate_hz) if np.any(m) else Envelope.constant(0.0)


def design_budget(x: np.ndarray, fs: float, plan: MultibandPlan,
                  rate_hz: float = CONTROL_RATE_HZ, hits: dict | None = None) -> MultibandPlan:
    """Budget solve and envelopes (module contract). plan supplies the budget targets,
    the timing (attack_ms, release_ms, knee_db, ceiling timing), the ratio when
    budget["ratio_from_crest"] is off, the crossovers and the makeup rules; hits
    (envelope.hits_for on x) the average-peak reading, None the maximum."""
    x = _check_stereo(x)
    b = {k: v for k, v in plan.budget.items() if k not in ("result", "warnings")}
    mode = b.get("makeup", "peak")
    if mode not in MAKEUP_MODES:
        raise ValueError(f"makeup must be one of {MAKEUP_MODES}")
    n = x.shape[0]
    xo = list(plan.crossovers_hz)
    nb = len(xo) + 1
    att = _table(plan.attack_ms, nb, "attack_ms")
    rel = _table(plan.release_ms, nb, "release_ms")
    knee = _table(plan.knee_db, nb, "knee_db")
    crel = _table(plan.ceiling_release_ms, nb, "ceiling_release_ms")
    look = _lookahead(plan.ceiling_lookahead_ms, nb)
    rat = _table(2.0 if plan.ratio is None else plan.ratio, nb, "ratio")
    makeup_kc = plan.makeup()
    n_ctrl = control_length(n, fs, rate_hz)
    comp_t, comp_max = float(b["comp_gr_db"]), float(b["comp_gr_max_db"])
    normal, lim_max = float(b["total_normal_db"]), float(b["limit_max_db"])
    thr = np.zeros((nb, 2))
    ceil: list[list[float | None]] = [[None, None] for _ in range(nb)]
    rows: list[list[dict]] = []
    gates = []
    warn: list[dict] = []
    warn_avg: list[dict] = []
    full_db = meter.rms_1s(x, fs)["loud_section_dbfs"]

    def channel(k, c, v):
        """One band channel's budget solve: (row, makeup gate, warnings, average
        warnings). Writes only its own [k, c] entries of rat, thr and ceil."""
        warns, warns_avg = [], []
        det = detector(v, fs, n_ctrl, rate_hz)
        with np.errstate(divide="ignore"):
            gate = 20.0 * np.log10(det) > RMS_MAKEUP_GATE_DBFS
        crest = band_crest_db(v, fs)
        if b.get("ratio_from_crest", True):
            rat[k, c] = min(ratio_for_crest(crest, k, c), float(b.get("ratio_max", RATIO_MAX)))

        def comp(target, v=v, det=det, k=k, c=c):
            sol = threshold_for_peak_gr(v, fs, target, rat[k, c], att[k, c], rel[k, c],
                                        knee[k, c], rate_hz, hits)
            env = comp_envelope(det, sol["threshold_db"], rat[k, c], att[k, c], rel[k, c],
                                knee[k, c], rate_hz)
            g = np.clip(env.at(n, fs), -MULTIBAND_MAX_GR_DB, 0.0)
            return sol, v * 10.0 ** ((g + makeup_kc[k][c]) / 20.0)

        lvl = meter.rms_1s(v, fs)["loud_section_dbfs"] if np.any(v) else -np.inf
        gated = bool(np.isfinite(full_db) and not lvl >= full_db - GATE_BELOW_INPUT_DB)
        if gated:
            sol, _ = comp(0.0)
            thr[k, c] = sol["threshold_db"]
            return ({"crest_db": float(crest) if np.isfinite(crest) else None,
                     "ratio": float(rat[k, c]), "threshold_db": float(thr[k, c]),
                     "comp_target_db": 0.0, "limit_need_db": 0.0,
                     "limit_target_db": 0.0, "exception": False, "gated": True,
                     "level_db": float(lvl) if np.isfinite(lvl) else None,
                     "budget_db": float(normal)}, gate, warns, warns_avg)
        sol, after = comp(comp_t)
        need = limit_need_db(after, fs)
        # the band limiter is an outlier catcher solved on its per-sample peak; on the
        # meter reading its spikes barely register, so the tally (compressor + limiter
        # as the meter shows them) is reported, and the limiter may take up to
        # lim_max before the compressor goes to its exception target
        allowance = lim_max
        exception = need > allowance + 1e-9
        if exception:
            # only where peaks get past the normal tally: 3 dB comp + up to 3 dB limit
            sol, after = comp(comp_max)
            need = limit_need_db(after, fs)
            if need > lim_max + 0.05:
                warns.append({"stage": "multiband", "band": k, "channel": "MS"[c],
                              "needed_db": float(sol["peak_gr_db"] + need),
                              "cap_db": float(comp_max + lim_max)})
        target_now = comp_max if exception else comp_t
        if sol["peak_gr_db"] < target_now - envelope.AVERAGE_WARN_DB and np.any(v):
            warns_avg.append({"stage": "multiband", "band": k, "channel": "MS"[c],
                              "kind": "average", "target_db": float(target_now),
                              "achieved_db": float(sol["peak_gr_db"])})
        lim_t = min(need, lim_max if exception else allowance)
        cs = (ceiling_for_peak_gr(after, fs, lim_t, look[k], crel[k, c],
                                  rate_hz) if lim_t > 0.01
              else {"ceiling_db": None, "peak_gr_db": 0.0})
        thr[k, c] = sol["threshold_db"]
        ceil[k][c] = cs["ceiling_db"]
        del after, det
        return ({"crest_db": float(crest) if np.isfinite(crest) else None,
                 "ratio": float(rat[k, c]), "threshold_db": float(thr[k, c]),
                 "comp_target_db": comp_max if exception else comp_t,
                 "limit_need_db": float(need), "limit_target_db": float(lim_t),
                 "exception": bool(exception),
                 "budget_db": float(comp_max + lim_max if exception else normal)},
                gate, warns, warns_avg)

    for k, band in enumerate(bands.split_iter(bands.to_ms(x), fs, xo, order=plan.order,
                                              phase=plan.phase)):
        # the two channels in parallel (eluate.parallel); all ten pairs at once would
        # hold every band and its temporaries (sprint WP8)
        row, grow = [], []
        for r, g, w, wa in parallel.pmap(lambda c: channel(k, c, band[:, c]), range(2)):
            row.append(r)
            grow.append(g)
            warn += w
            warn_avg += wa
        rows.append(row)
        gates.append(grow)
        del band
    has_ceil = any(c is not None for r in ceil for c in r)
    p = design(x, fs, thr.tolist(), ratio=rat.tolist(), attack_ms=att.tolist(),
               release_ms=rel.tolist(), crossovers_hz=xo, order=plan.order, phase=plan.phase,
               rate_hz=rate_hz, ceiling_db=ceil if has_ceil else None,
               ceiling_lookahead_ms=plan.ceiling_lookahead_ms,
               ceiling_release_ms=crel.tolist(), knee_db=knee.tolist(),
               mid_makeup_db=list(plan.mid_makeup_db), side_makeup_db=plan.side_makeup_db,
               decoherence=list(plan.decoherence))
    mm = []
    for k in range(nb):
        pair = []
        for c in range(2):
            gr = np.clip(_ctrl(p.gr_db[k][c], n_ctrl), -MULTIBAND_MAX_GR_DB, 0.0)
            cg = (np.zeros(n_ctrl) if p.ceiling_gr_db is None
                  else _ctrl(p.ceiling_gr_db[k][c], n_ctrl))
            tot = gr + cg
            r = rows[k][c]
            sc, sl, st = (envelope.stage_gr(-a, rate_hz, hits) for a in (gr, cg, tot))
            r["comp_db"], r["comp_outlier_db"] = sc["average_db"], sc["outlier_db"]
            r["limit_db"], r["limit_outlier_db"] = sl["average_db"], sl["outlier_db"]
            r["total_db"], r["total_outlier_db"] = st["average_db"], st["outlier_db"]
            r["comp_cap_db"] = MULTIBAND_OUTLIER_CAP_DB
            r["limit_cap_db"] = lim_max
            r["ceiling_db"] = None if p.ceiling_db is None else p.ceiling_db[k][c]
            env = mode_makeup(tot, gates[k][c], mode, rate_hz, hits)
            r["makeup_db"] = float(env.values.max()) + 0.0
            pair.append(env)
        mm.append(pair)
    # one warning per band: M and S both over their caps read "both"
    merged: dict[int, dict] = {}
    for w in warn:
        if w["band"] in merged:
            o = merged[w["band"]]
            o["channel"] = "both"
            o["needed_db"] = max(o["needed_db"], w["needed_db"])
        else:
            merged[w["band"]] = dict(w)
    b.update(result=rows, warnings=[merged[k] for k in sorted(merged)] + warn_avg,
             reading="max" if hits is None else "average")
    return replace(p, mode_makeup_db=mm, budget=b)


SPECLIM_OFF_THRESHOLD_DB = 40.0   # compressor threshold that never engages (ceiling-only)


def design_speclim(x: np.ndarray, fs: float, crossovers_hz, allow_db,
                   max_cut_db: float, lookahead_ms, release_ms,
                   hits: dict | None = None, order: int = 2,
                   phase="zero") -> MultibandPlan | None:
    """Frequency-dependent limiting (maintainer, 2026-10-19; the Ozone Maximizer's spectral
    limiting is the model): a ceiling-only multiband on x, the audio entering the clipper.
    Per band and M / S: the typical loud hit's peak P50 (median over hits["loud"] of each
    hit's sample peak; hits None = envelope.hits_for(x) on 400 ms frames) and the band's
    largest peak P; the ceiling is max(P50 + allow_db, P - max_cut_db) (allow_db a scalar
    or one value per band), none when P is
    already within it. So only peaks that stand out in their own band are limited, by at
    most max_cut_db, and the rest of the spectrum is left alone. lookahead_ms: the
    ceilings' lookahead, a scalar or one value per band (design). No makeup. Returns None
    when no band needs a ceiling; the plan's budget records each band's P50, P, ceiling
    and planned cut. A band channel whose loudest-section 1 s RMS sits
    SPECLIM_GATE_BELOW_INPUT_DB below x's (near silent, e.g. the low S band after the side strip) gets no ceiling, as
    the compressor leaves it alone; its row says "gated" (audit 08 #11)."""
    x = _check_stereo(x)
    full_db = meter.rms_1s(x, fs)["loud_section_dbfs"]
    if hits is None:
        hits = envelope.hits_for(x, fs, None)
    e = np.asarray(hits["edges_s"], dtype=np.float64)
    loud = np.asarray(hits["loud"], dtype=bool)
    nb = len(crossovers_hz) + 1
    allow = [float(a) for a in (allow_db if np.ndim(allow_db) else [allow_db] * nb)]
    ceil: list[list[float | None]] = [[None, None] for _ in range(nb)]
    rows = []
    for k, band in enumerate(bands.split_iter(bands.to_ms(x), fs, list(crossovers_hz),
                                              order=order, phase=phase)):
        row = []
        for c in range(2):
            a = np.abs(band[:, c])
            pk = float(a.max()) if a.size else 0.0
            if pk <= 0.0:
                row.append({"typical_db": None, "peak_db": None, "ceiling_db": None, "cut_db": 0.0,
                            "gated": True})
                continue
            lvl = meter.rms_1s(band[:, c], fs)["loud_section_dbfs"]
            if np.isfinite(full_db) and not lvl >= full_db - SPECLIM_GATE_BELOW_INPUT_DB:
                row.append({"typical_db": None, "peak_db": 20.0 * np.log10(pk), "ceiling_db": None,
                            "cut_db": 0.0, "gated": True})
                continue
            hv = envelope.hit_values(a, fs, e)
            with np.errstate(divide="ignore"):
                hd = 20.0 * np.log10(hv)
            ok = np.isfinite(hd) & (loud if loud.size == hd.size else True)
            if not np.any(ok):
                ok = np.isfinite(hd)
            p50 = float(np.median(hd[ok]))
            p_db = 20.0 * np.log10(pk)
            cl = max(p50 + allow[k], p_db - max_cut_db)
            if p_db > cl + 0.05:
                ceil[k][c] = cl
            row.append({"typical_db": p50, "peak_db": p_db,
                        "ceiling_db": ceil[k][c], "cut_db": max(0.0, p_db - cl), "gated": False})
        rows.append(row)
        del band
    if all(c is None for r in ceil for c in r):
        return None
    p = design(x, fs, SPECLIM_OFF_THRESHOLD_DB, ratio=2.0, crossovers_hz=list(crossovers_hz),
               order=order, phase=phase, ceiling_db=ceil, ceiling_lookahead_ms=lookahead_ms,
               ceiling_release_ms=release_ms, mid_makeup_db=[0.0] * nb, side_makeup_db=0.0)
    p.budget = {"kind": "speclim", "allow_db": allow,
                "max_cut_db": float(max_cut_db), "result": rows}
    return p


def design(x: np.ndarray, fs: float, threshold_db, ratio: float = 2.0,
           attack_ms=10.0, release_ms=100.0,
           crossovers_hz=bands.DEFAULT_CROSSOVERS, order: int = 2,
           phase="zero", rate_hz: float = CONTROL_RATE_HZ, ceiling_db=None,
           ceiling_lookahead_ms=2.0, ceiling_release_ms=50.0, knee_db=0.0,
           **plan_kw) -> MultibandPlan:
    """Compressor and ceiling envelopes from band peaks (see module contract).

    threshold_db, attack_ms, release_ms, ceiling_release_ms: scalar, per band
    (length bands) or bands x [M, S]. ceiling_db: None (no ceiling stage) or the
    same shapes, an entry None meaning no ceiling for that band and channel
    (dBFS sample peak of the M or S band as processed). ratio and knee_db: scalar or
    the same shapes (a scalar is recorded as a scalar). ceiling_lookahead_ms: scalar or
    per band (band k's ceiling uses its own; recorded as given). plan_kw go to
    MultibandPlan (mid_makeup_db, side_makeup_db, decoherence); the ceiling
    detector sees the makeup, which the plan records (ceiling_makeup_db).
    Raises ValueError when a ceiling needs more than 120 dB of reduction
    (envelope.G_MIN)."""
    x = _check_stereo(x)
    n = x.shape[0]
    nb = len(crossovers_hz) + 1
    rat = _table(ratio, nb, "ratio")
    kn = _table(knee_db, nb, "knee_db")
    if not np.all(rat > 1.0):
        raise ValueError("ratio must be > 1 (ratio 1 is a no-op)")
    if np.any(kn < 0.0):
        raise ValueError("knee_db must be >= 0")
    ms = bands.to_ms(x)
    thr = _table(threshold_db, nb, "threshold_db")
    att = _table(attack_ms, nb, "attack_ms")
    rel = _table(release_ms, nb, "release_ms")
    crel = _table(ceiling_release_ms, nb, "ceiling_release_ms")
    look = _lookahead(ceiling_lookahead_ms, nb)
    if np.any(att <= 0) or np.any(rel < 0) or np.any(crel <= 0):
        raise ValueError("attack_ms and ceiling_release_ms must be > 0, release_ms >= 0")
    ceil = None
    if ceiling_db is not None:
        c_obj = np.empty((nb, 2), dtype=object)
        src = np.asarray(ceiling_db, dtype=object)
        if src.ndim == 1 and src.size == nb:
            src = src[:, None]
        c_obj[...] = np.broadcast_to(src, (nb, 2))
        ceil = [[None if c_obj[k, c] is None else float(c_obj[k, c]) for c in range(2)]
                for k in range(nb)]
    n_ctrl = control_length(n, fs, rate_hz)
    gr = []
    # bands streamed twice (compressor, then ceiling) instead of held: same values
    for k, band in enumerate(bands.split_iter(ms, fs, crossovers_hz, order=order, phase=phase)):
        pair = []
        for c in range(2):
            det = detector(band[:, c], fs, n_ctrl, rate_hz)
            pair.append(comp_envelope(det, thr[k, c], rat[k, c], att[k, c], rel[k, c],
                                      kn[k, c], rate_hz))
        gr.append(pair)
        del band, det           # free this band before split_iter computes the next
    common = dict(gr_db=gr, crossovers_hz=list(crossovers_hz), order=order, phase=phase,
                  threshold_db=thr.tolist(),
                  ratio=float(ratio) if np.ndim(ratio) == 0 else rat.tolist(),
                  knee_db=float(knee_db) if np.ndim(knee_db) == 0 else kn.tolist(),
                  attack_ms=att.tolist(),
                  release_ms=rel.tolist(), ceiling_db=ceil,
                  ceiling_lookahead_ms=(float(ceiling_lookahead_ms)
                                        if np.ndim(ceiling_lookahead_ms) == 0 else look),
                  ceiling_release_ms=crel.tolist(),
                  design_span_s=n / fs, **plan_kw)
    plan = MultibandPlan(**common)
    if ceil is None:
        return plan
    cg = []
    makeup_kc = plan.makeup()
    for k, band in enumerate(bands.split_iter(ms, fs, crossovers_hz, order=order, phase=phase)):
        pair = []
        for c in range(2):
            if ceil[k][c] is None:
                pair.append(Envelope.constant(0.0))
                continue
            # one channel's compressor curve at a time (the values _gains_db gives)
            g_comp = np.clip(plan.gr_db[k][c].at(n, fs), -MULTIBAND_MAX_GR_DB, 0.0)
            pre = band[:, c] * 10.0 ** ((g_comp + makeup_kc[k][c]) / 20.0)
            del g_comp
            det = detector(pre, fs, n_ctrl, rate_hz)
            g_min = float(envelope.required_gain(det, 10.0 ** (ceil[k][c] / 20.0)).min())
            if g_min < envelope.G_MIN:
                raise ValueError(
                    f"band {k} {'MS'[c]}: ceiling {ceil[k][c]} dBFS needs "
                    f"{-20 * np.log10(g_min):.1f} dB of reduction, beyond the "
                    f"{-20 * np.log10(envelope.G_MIN):.0f} dB envelope floor")
            pair.append(ceiling_envelope(det, ceil[k][c], look[k], crel[k, c], rate_hz))
        cg.append(pair)
    # constructed, not assigned, so MultibandPlan validation runs on the ceiling (audit 06 M7)
    return MultibandPlan(**common, ceiling_gr_db=cg, ceiling_makeup_db=plan.makeup())


def design_from(plan: MultibandPlan, x: np.ndarray, fs: float,
                rate_hz: float = CONTROL_RATE_HZ, hits: dict | None = None) -> MultibandPlan:
    """Rebuild a plan's envelopes from its design record (the solver's levers:
    per-band thresholds, attack / release windows, ceilings and ceiling timing). A plan
    with a budget record is re-solved on x (design_budget, over hits)."""
    if plan.budget is not None:
        return replace(design_budget(x, fs, plan, rate_hz, hits), ms_link=plan.ms_link)
    if plan.threshold_db is None or plan.ratio is None or plan.attack_ms is None \
            or plan.release_ms is None:
        raise ValueError("plan has no design record (threshold_db, ratio, attack_ms, release_ms)")
    return replace(design(x, fs, plan.threshold_db, ratio=plan.ratio, attack_ms=plan.attack_ms,
                          release_ms=plan.release_ms, crossovers_hz=plan.crossovers_hz,
                          order=plan.order, phase=plan.phase, rate_hz=rate_hz,
                          ceiling_db=plan.ceiling_db,
                          ceiling_lookahead_ms=plan.ceiling_lookahead_ms,
                          ceiling_release_ms=plan.ceiling_release_ms,
                          mid_makeup_db=list(plan.mid_makeup_db),
                          side_makeup_db=plan.side_makeup_db,
                          decoherence=list(plan.decoherence), knee_db=plan.knee_db),
                   ms_link=plan.ms_link)


# ------------------------------------------------------------------ crest loss

def _interior(t: np.ndarray, n: int, fs: float) -> np.ndarray:
    dur = n / fs
    return (t - CREST_WINDOW_S / 2 >= CREST_EDGE_S) & (t + CREST_WINDOW_S / 2 <= dur - CREST_EDGE_S)


def _p95(v: np.ndarray) -> float:
    return float(np.percentile(v, 95.0)) if v.size else np.nan


def band_crest(x: np.ndarray, fs: float, crossovers_hz=bands.DEFAULT_CROSSOVERS,
               hop_s: float = CREST_HOP_S) -> dict:
    """Per-band crest factor over time, spec 3.3 row "crest", computed here from
    bands.split (zero phase) independently of features/crest.py.

    Bands: zero-phase split of L/R at `crossovers_hz` (pass the plan's). Views
    M, S (bands.to_ms) and stereo. Frame k is centred on sample
    round((k + 0.5) hop fs) and covers the 400 ms window
    [c - L // 2, c - L // 2 + L), in-signal samples only.
    M / S: peak = max |v|, RMS = sqrt(mean v^2); stereo: peak = max(|L|, |R|),
    RMS = sqrt(mean (L^2 + R^2) / 2). crest_raw_db = 20 log10(peak / RMS) on
    every frame (non-finite where RMS is 0); crest_db the same, NaN where the
    frame's RMS <= -60 dBFS. p95_db[b, v] = 95th percentile of crest_db over
    gated frames whose window lies at least 0.25 s inside both ends, or over
    all gated frames when no gated frame is interior (audit 06 M13: a transient
    only in the first 0.65 s is still measured); NaN when no frame is gated.
    features/crest.py falls back only when no frame at all is interior.
    Returns {"crest_db", "crest_raw_db", "rms_dbfs": (B, 3, K), "p95_db": (B, 3),
    "t": (K,), "interior": (K,)}."""
    x = _check_stereo(x)
    n = x.shape[0]
    if n < 2 or not np.all(np.isfinite(x)):
        raise ValueError("need finite audio with n >= 2")
    kf = max(1, int(np.ceil(n / (hop_s * fs) - 1e-9)))
    t = (np.arange(kf) + 0.5) * hop_s
    centres = np.round(t * fs).astype(np.int64)
    length = int(round(CREST_WINDOW_S * fs))
    a = np.clip(centres - length // 2, 0, n)
    e = np.clip(centres - length // 2 + length, 0, n)
    cnt = np.maximum(e - a, 1).astype(np.float64)
    nb = len(crossovers_hz) + 1
    crest_raw = np.full((nb, 3, kf), np.nan)
    rms_db = np.full((nb, 3, kf), -np.inf)
    interior = _interior(t, n, fs)
    pad = max(0, int(centres.max()) + length - n + 1)
    # bands streamed (bands.split_iter) and each view's two signals built only when
    # used: the same values as the full split and view tuple, a fraction of the memory
    for bi, lr in enumerate(bands.split_iter(x, fs, crossovers_hz)):

        def sig(v):
            # M and S with bands.to_ms's arithmetic, (L +- R) / sqrt 2, one at a time
            if v == 0:
                return (lr[:, 0] + lr[:, 1]) * bands.INV_SQRT2
            return (lr[:, 0] - lr[:, 1]) * bands.INV_SQRT2

        for v in range(3):
            # each full-length intermediate is dropped as soon as it is used (the same
            # values as building the view pair, cumsum and padded copy together)
            if v < 2:
                m = sig(v)
                pw_sig = m ** 2
            else:
                pw_sig = 0.5 * (lr[:, 0] ** 2 + lr[:, 1] ** 2)
            c = np.empty(n + 1)
            c[0] = 0.0
            np.cumsum(pw_sig, out=c[1:])
            del pw_sig
            pw = np.maximum((c[e] - c[a]) / cnt, 0.0)
            del c
            if v < 2:
                np.abs(m, out=m)
                pk_sig = m
                del m
            else:
                pk_sig = np.max(np.abs(lr), axis=1)
            s = np.pad(pk_sig, (0, pad))
            del pk_sig
            mf = maximum_filter1d(s, size=length, mode="constant", cval=0.0)
            pk = mf[np.clip(centres, 0, s.size - 1)]
            del s, mf
            with np.errstate(divide="ignore", invalid="ignore"):
                r = 10.0 * np.log10(pw)
                crest_raw[bi, v] = 20.0 * np.log10(np.maximum(pk, 1e-30)) - r
            rms_db[bi, v] = r
        del lr                  # free this band before split_iter computes the next
    gated = rms_db > CREST_GATE_DBFS
    crest = np.where(gated, crest_raw, np.nan)
    p95 = np.full((nb, 3), np.nan)
    for bi in range(nb):
        for v in range(3):
            g = gated[bi, v]
            sel = g & interior if np.any(g & interior) else g
            p95[bi, v] = _p95(crest[bi, v][sel])
    return {"crest_db": crest, "crest_raw_db": crest_raw, "rms_dbfs": rms_db, "p95_db": p95,
            "t": t, "interior": interior, "views": CREST_VIEWS}


def crest_loss(before: np.ndarray, after: np.ndarray, fs: float, crossovers_hz=None,
               limits_db=None, plan: MultibandPlan | None = None) -> dict:
    """Per-band crest loss between a premaster and a render (spec 3.6 constraint).

    Bands: crossovers_hz, else plan.crossovers_hz, else the defaults; both
    signals are split the same way (zero phase, spec 3.3) and measured on the
    same 100 ms grid. Per band b and view v:
    - Shared frames F = frames where max(RMS_before, RMS_after) > -60 dBFS (one
      frame set for both signals, so a level change straddling the gate does
      not move frames in or out: audit 06 M6), interior frames only when any
      qualify (as band_crest).
    - silenced[b, v]: more than half of the frames gated before are below
      -60 - CREST_HYST_DB (= -70) dBFS after. A silenced band is a violation:
      ok False, worst_excess_db +inf (spec 3.6: never "nothing to lose"; audit
      06 F5, M4). dropped_fraction reports that share.
    - unmeasured[b, v]: F is empty (no content before or after): loss NaN,
      ok True, excess NaN.
    - loss_db = p95(crest_before over F) - p95(crest_after over F), frames where
      either crest is non-finite excluded; positive = crest lost. This is the
      difference of the spec 3.3 summaries; frame_loss_p95_db = p95 of the
      per-frame differences is reported alongside.
    limits_db: per-band tolerance (CREST_LOSS_LIMITS_DB for five bands, else
    required). ok = not silenced and loss <= limit (NaN loss with a measured
    band counts as ok). worst_excess_db[b] = max over views of loss - limit,
    +inf if any view is silenced, NaN only if every view is unmeasured; never
    -inf. Signals must have the same shape."""
    before = _check_stereo(before)
    after = _check_stereo(after)
    if before.shape != after.shape:
        raise ValueError("before and after must have the same shape")
    if crossovers_hz is None:
        crossovers_hz = plan.crossovers_hz if plan is not None else bands.DEFAULT_CROSSOVERS
    cb = band_crest(before, fs, crossovers_hz)
    ca = band_crest(after, fs, crossovers_hz)
    nb = cb["p95_db"].shape[0]
    if limits_db is None:
        if nb != len(CREST_LOSS_LIMITS_DB):
            raise ValueError(f"limits_db needed for {nb} bands")
        limits_db = CREST_LOSS_LIMITS_DB
    lim = np.asarray(limits_db, dtype=np.float64)
    if lim.shape != (nb,):
        raise ValueError(f"limits_db needs {nb} values")
    shape = (nb, 3)
    loss, frame_loss = np.full(shape, np.nan), np.full(shape, np.nan)
    p_b, p_a = np.full(shape, np.nan), np.full(shape, np.nan)
    silenced = np.zeros(shape, dtype=bool)
    unmeasured = np.zeros(shape, dtype=bool)
    dropped = np.zeros(shape)
    interior = cb["interior"]
    for bi in range(nb):
        for v in range(3):
            rb, ra = cb["rms_dbfs"][bi, v], ca["rms_dbfs"][bi, v]
            gb = rb > CREST_GATE_DBFS
            if np.any(gb):
                dropped[bi, v] = np.mean(ra[gb] <= CREST_GATE_DBFS - CREST_HYST_DB)
                silenced[bi, v] = dropped[bi, v] > CREST_SILENCED_FRACTION
            shared = np.maximum(rb, ra) > CREST_GATE_DBFS
            if not np.any(shared):
                unmeasured[bi, v] = True
                continue
            sel = shared & interior if np.any(shared & interior) else shared
            xb, xa = cb["crest_raw_db"][bi, v][sel], ca["crest_raw_db"][bi, v][sel]
            fin = np.isfinite(xb) & np.isfinite(xa)
            p_b[bi, v], p_a[bi, v] = _p95(xb[fin]), _p95(xa[fin])
            loss[bi, v] = p_b[bi, v] - p_a[bi, v]
            frame_loss[bi, v] = _p95(xb[fin] - xa[fin])
    excess = loss - lim[:, None]
    ok = ~silenced & ~(excess > 0)
    worst = np.full(nb, np.nan)
    for bi in range(nb):
        if silenced[bi].any():
            worst[bi] = np.inf
        elif np.any(np.isfinite(excess[bi])):
            worst[bi] = float(np.nanmax(excess[bi]))
    return {"loss_db": loss, "frame_loss_p95_db": frame_loss, "before_p95_db": p_b,
            "after_p95_db": p_a, "limits_db": lim, "ok": ok, "silenced": silenced,
            "unmeasured": unmeasured, "dropped_fraction": dropped, "worst_excess_db": worst,
            "views": CREST_VIEWS}
