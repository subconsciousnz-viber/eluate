"""Budgets on the average peak (docs/gain-staging.md 3, maintainer 2026-10-19).

Average peak GR = the median over the loud hits of each hit's maximum GR; outlier peak
GR = the maximum over the track. Every solve hits its average target, outliers are
capped at the budget + 1.5 dB. Synthetic signals only."""

import numpy as np
import pytest

from eluate import envelope, planner, solver
from eluate.plan import PLAN_VERSION, HitsPlan, LimiterPlan, Plan
from eluate.stages import glue, multiband as mb

FS = 44100
TOL = 0.1
BPM = 120.0
BEAT = 60.0 / BPM


# ------------------------------------------------------------ the shared function

def test_hit_edges_from_beats_and_the_400_ms_fallback():
    e = envelope.hit_edges(2.0, [0.0, 0.5, 1.0, 1.5, 2.5])
    assert e.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]
    # no usable grid (none, or a single beat): 400 ms frames from 0
    for beats in (None, [], [0.7]):
        assert np.allclose(envelope.hit_edges(2.0, beats), [0.0, 0.4, 0.8, 1.2, 1.6, 2.0])


def test_average_peak_gr_on_a_known_envelope():
    rate = 1000.0
    g = np.zeros(4000)                       # 4 s, beats every 0.5 s: 8 hits
    peaks = [1.0, 3.0, 2.0, 2.5, 0.5, 9.0, 2.2, 1.8]
    for k, v in enumerate(peaks):
        g[k * 500 + 100] = v                 # one peak inside each beat
        g[k * 500 + 300] = v / 2             # a smaller one that the hit maximum ignores
    beats = np.arange(0.5, 4.0, 0.5)
    e = envelope.hit_edges(4.0, beats)
    assert envelope.hit_values(g, rate, e).tolist() == peaks
    r = envelope.average_peak_gr(g, rate, e)
    assert r["average_db"] == pytest.approx(np.median(peaks))
    assert r["outlier_db"] == 9.0 and r["hits"] == 8 and r["loud_hits"] == 8
    # the loud mask: the median over the loud hits only
    loud = np.array([False, True, True, True, False, True, True, False])
    r = envelope.average_peak_gr(g, rate, e, loud)
    assert r["average_db"] == pytest.approx(np.median([3.0, 2.0, 2.5, 9.0, 2.2]))
    assert r["outlier_db"] == 9.0 and r["loud_hits"] == 5
    # one hit (no edges): the old whole-track maximum
    one = envelope.average_peak_gr(g, rate, None)
    assert one["average_db"] == one["outlier_db"] == 9.0
    # the 400 ms fallback: hits are frames, the median of their maxima
    fr = envelope.average_peak_gr(g, rate, envelope.hit_edges(4.0, None))
    vals = envelope.hit_values(g, rate, envelope.hit_edges(4.0, None))
    assert vals.size == 10 and fr["average_db"] == pytest.approx(np.median(vals))
    # a constant curve reads its value on every hit
    assert envelope.average_peak_gr(np.array([1.5]), rate, e)["average_db"] == 1.5


def _sections(fs=FS):
    """32 beats of noise: 8 quiet (breakdown, -12 dB), 24 loud: the drop."""
    n = int(32 * BEAT * fs)
    rng = np.random.default_rng(2)
    x = 0.1 * rng.standard_normal((n, 2))
    x[: int(8 * BEAT * fs)] *= 10 ** (-12 / 20)
    return x


def test_loud_section_mask():
    x = _sections()
    e = envelope.hit_edges(x.shape[0] / FS, np.arange(1, 32) * BEAT)
    loud = envelope.loud_hits(x, FS, e)
    assert loud.size == 32
    # the breakdown is out, except the beat whose 1 s window reaches into the drop
    assert not loud[:7].any() and loud[8:].all()


def test_loud_section_takes_the_16_loudest_when_too_few_qualify():
    n = int(32 * BEAT * FS)
    rng = np.random.default_rng(3)
    lv = np.linspace(-20.0, 0.0, 32)        # a steady 20 dB ramp, one level per beat
    x = rng.standard_normal((n, 2)) * 0.05
    for k in range(32):
        x[int(k * BEAT * FS):int((k + 1) * BEAT * FS)] *= 10 ** (lv[k] / 20)
    e = envelope.hit_edges(n / FS, np.arange(1, 32) * BEAT)
    loud = envelope.loud_hits(x, FS, e)
    # only the top few beats are within 3 dB of the p95; the fallback keeps 16
    assert loud.sum() == 16 and loud[16:].all() and not loud[:16].any()
    # fewer hits than 16: every hit
    short = x[: int(8 * BEAT * FS)]
    assert envelope.loud_hits(short, FS, envelope.hit_edges(8 * BEAT, np.arange(1, 8) * BEAT)).all()


# ------------------------------------------------------------ the solves

def _loop(fs=FS, beats=32, seed=1, spike_at=(), spike_db=0.0, spread_db=4.0):
    """Noise hits on every beat, levels varying over spread_db, a pad underneath; spikes
    (spike_db above the loudest hit) at the given beats."""
    rng = np.random.default_rng(seed)
    n = int(beats * BEAT * fs)
    x = 0.01 * rng.standard_normal((n, 2))
    th = np.arange(int(0.06 * fs)) / fs
    for b in range(beats):
        lvl = 10 ** ((spike_db if b in spike_at else rng.uniform(-spread_db, 0.0)) / 20)
        i = int(b * BEAT * fs)
        x[i:i + th.size] += 0.3 * lvl * rng.standard_normal((th.size, 2)) \
            * np.exp(-th / 0.015)[:, None]
    return x


def _tone_loop(fs=FS, beats=32, seed=1, spike_at=(), spike_db=0.0, spread_db=1.0):
    """Decaying 200 Hz hits on every beat, levels varying over spread_db (deterministic
    peaks, unlike noise bursts), over a quiet pad; spikes at the given beats."""
    rng = np.random.default_rng(seed)
    n = int(beats * BEAT * fs)
    x = 0.01 * rng.standard_normal((n, 2))
    th = np.arange(int(0.1 * fs)) / fs
    hit = np.sin(2 * np.pi * 200.0 * th) * np.exp(-th / 0.03)
    for b in range(beats):
        lvl = 10 ** ((spike_db if b in spike_at else rng.uniform(-spread_db, 0.0)) / 20)
        i = int(b * BEAT * fs)
        x[i:i + th.size] += 0.6 * lvl * hit[:, None]
    return x


def _hits(x):
    return envelope.hits_for(x, FS, np.arange(1, int(x.shape[0] / FS / BEAT)) * BEAT)


@pytest.mark.parametrize("target", [1.0, 2.5, 3.0])
def test_multiband_threshold_hits_the_average_peak(target):
    x = _loop()
    h = _hits(x)
    v = x[:, 0]
    sol = mb.threshold_for_peak_gr(v, FS, target, 1.7, 10.0, 100.0, 3.0, hits=h)
    det = mb.detector(v, FS, mb.control_length(v.size, FS, 1000.0), 1000.0)
    env = mb.comp_envelope(det, sol["threshold_db"], 1.7, 10.0, 100.0, 3.0)
    # the stage's per-sample clamp, as the solve reads it (D5: 6 dB)
    r = envelope.stage_gr(np.minimum(-env.values, mb.MULTIBAND_MAX_GR_DB), 1000.0, h)
    assert r["average_db"] == pytest.approx(target, abs=TOL)
    assert sol["peak_gr_db"] == pytest.approx(r["average_db"], abs=1e-9)
    assert sol["outlier_gr_db"] == pytest.approx(r["outlier_db"], abs=1e-9)
    # the typical hit reaches the target: the loudest goes past it
    assert r["outlier_db"] > r["average_db"] + 0.3
    old = mb.threshold_for_peak_gr(v, FS, target, 1.7, 10.0, 100.0, 3.0)
    assert sol["threshold_db"] < old["threshold_db"] - 0.5


def test_multiband_outlier_spike_is_capped():
    x = _loop(spike_at=(9,), spike_db=12.0)
    h = _hits(x)
    sol = mb.threshold_for_peak_gr(x[:, 0], FS, 3.0, 2.0, 10.0, 100.0, 0.0, hits=h)
    assert sol["peak_gr_db"] == pytest.approx(3.0, abs=TOL)
    # the spike reads well past the typical hit on the meter, within the per-sample clamp
    assert 3.3 < sol["outlier_gr_db"] <= mb.MULTIBAND_MAX_GR_DB + 1e-6


def test_multiband_design_budget_on_the_average():
    x = _loop(spike_at=(20,), spike_db=8.0)
    p0 = mb.design(x, FS, -20.0, ratio=1.5)
    p0.budget = mb.default_budget("peak")
    p = mb.design_budget(x, FS, p0, hits=_hits(x))
    assert p.budget["reading"] == "average"
    for row in p.budget["result"]:
        for r in row:
            assert r["comp_db"] == pytest.approx(r["comp_target_db"], abs=TOL)
            # the meter-reading outlier is reported against its cap (4.5), held only by
            # the per-sample clamp
            assert r["comp_outlier_db"] <= mb.MULTIBAND_MAX_GR_DB + 1e-6
            assert r["comp_cap_db"] == mb.MULTIBAND_OUTLIER_CAP_DB
            assert r["limit_outlier_db"] <= r["limit_cap_db"] + 1e-6
            # peak makeup: + the combined average peak GR
            assert r["makeup_db"] == pytest.approx(r["total_db"], abs=1e-6)
    # without hits the whole track is one hit: the old maximum reading
    q = mb.design_budget(x, FS, p0)
    assert q.budget["reading"] == "max"
    for row in q.budget["result"]:
        for r in row:
            assert r["comp_db"] == r["comp_outlier_db"]


@pytest.mark.parametrize("mode", ["serial", "parallel"])
def test_glue_hits_the_net_average_peak_and_caps_the_outlier(mode):
    x = _loop(spike_at=(7, 23), spike_db=10.0)
    g = glue.design_budget(x, FS, mode, hits=_hits(x))
    b = g.budget
    assert b["net_db"] == pytest.approx(1.5, abs=TOL)
    assert b["net_outlier_db"] <= glue.SAMPLE_MAX_GR_DB + 1e-6 and glue.OUTLIER_CAP_DB == 4.0
    assert b["net_outlier_db"] > b["net_db"] + 0.3
    assert b["warnings"] == []
    if mode == "parallel":
        assert b["wet_db"] == pytest.approx(3.0, abs=TOL) and 0.0 < g.mix < 1.0
    # the stage's own curve reads the record
    r = envelope.stage_gr(glue.net_gr_curve_db(g), 1000.0, _hits(x))
    assert r["average_db"] == pytest.approx(b["net_db"], abs=1e-9)


def test_glue_warns_only_when_the_average_falls_half_a_db_short():
    x = _loop()
    assert glue.design_budget(x, FS, hits=_hits(x)).budget["warnings"] == []
    w = glue.design_budget(np.zeros((FS, 2)), FS, hits=None).budget["warnings"]
    assert w and w[0]["stage"] == "glue"


@pytest.mark.slow
def test_clipper_hits_the_average_output_peak_fall():
    # hits within 1 dB of each other: the 1.5 dB outlier cap does not bind
    x = _tone_loop(seed=4, spread_db=1.0)
    h = _hits(x)
    lp = LimiterPlan(clip_knee_db=1.0, clip_target_db=1.0)
    sol = solver.clip_threshold_for_average_peak(x, FS, lp, 1.0, h)
    assert sol["peak_gr_db"] == pytest.approx(1.0, abs=TOL)
    assert sol["outlier_gr_db"] <= 1.5 + 1e-6
    # the per-hit fall read independently of the solve
    e = h["edges_s"]
    m0 = solver.limit_blocked(x, FS, LimiterPlan(), measure_only=True, detector_env=True)
    m1 = solver.limit_blocked(x, FS, LimiterPlan(clip_threshold_db=sol["threshold_db"],
                                                 clip_knee_db=1.0),
                              measure_only=True, detector_env=True)
    fall = solver.hit_peaks_db(m0["detector_env"], FS, e) - solver.hit_peaks_db(
        m1["detector_env"], FS, e)
    assert float(np.median(fall[h["loud"]])) == pytest.approx(sol["peak_gr_db"], abs=1e-6)
    assert not sol["capped"]
    old = solver.clip_threshold_for_output_peak(x, FS, lp, 1.0)
    assert sol["threshold_db"] < old["threshold_db"]


@pytest.mark.slow
def test_clipper_outlier_spike_is_capped():
    x = _tone_loop(seed=5, spike_at=(12,), spike_db=6.0)
    lp = LimiterPlan(clip_knee_db=1.0, clip_target_db=1.0)
    sol = solver.clip_threshold_for_average_peak(x, FS, lp, 1.0, _hits(x))
    assert sol["capped"] and sol["outlier_gr_db"] <= 1.5 + 1e-6
    assert sol["outlier_gr_db"] == pytest.approx(1.5, abs=0.05)
    # the cap reaches into the loud hits: the planner warns when the average falls
    # more than 0.5 dB short
    rec = planner.solve_clipper(x, FS, lp, _hits(x)).budget["clipper"]
    assert rec["capped"] and rec["reading"] == "average"
    assert bool(rec["warnings"]) == (rec["peak_gr_db"] < 1.0 - 0.5)


# ------------------------------------------- audible maxima (dynamics-macros step 1)

def test_knee_clip_sample_cap_limits_the_reduction_per_sample():
    from eluate.stages import limiter as lim
    v = np.linspace(-4.0, 4.0, 4001)
    t, w = lim.clip_curve(0.5, 1.0)
    r = 10 ** (-3.0 / 20)
    out = lim._knee_clip(v, t, w, r)
    red = 20 * np.log10(np.abs(v[v != 0]) / np.abs(out[v != 0]))
    assert red.max() <= 3.0 + 1e-9
    # below the cap the curve is unchanged; above it the output follows the input -3 dB
    assert np.allclose(lim._knee_clip(v, t, w, r)[np.abs(v) < 0.6],
                       lim._knee_clip(v, t, w)[np.abs(v) < 0.6])
    assert np.allclose(out[np.abs(v) > 1.0], r * v[np.abs(v) > 1.0])


@pytest.mark.slow
def test_clipper_spike_with_the_sample_cap_keeps_the_average():
    # the same spike that held the uncapped clipper at 1.5 dB: with the 3 dB per-sample
    # cap the average reaches its 1 dB and the spike takes at most about 3 dB
    x = _tone_loop(seed=5, spike_at=(12,), spike_db=6.0)
    lp = LimiterPlan(clip_knee_db=1.0, clip_target_db=1.0, clip_max_gr_db=3.0)
    sol = solver.clip_threshold_for_average_peak(x, FS, lp, 1.0, _hits(x))
    assert sol["peak_gr_db"] == pytest.approx(1.0, abs=TOL)
    assert sol["outlier_gr_db"] <= 3.0 + 0.05
    old = solver.clip_threshold_for_average_peak(
        x, FS, LimiterPlan(clip_knee_db=1.0, clip_target_db=1.0), 1.0, _hits(x))
    assert sol["peak_gr_db"] > old["peak_gr_db"] + 0.3


def test_limiter_gain_max_reads_the_outlier_cap_and_names_the_bound():
    # typical hits at -5, the largest at 0, ceiling -0.3: outlier cap 5 gives +4.7,
    # the average budget (3) +7.7: the outlier binds
    assert solver.limiter_gain_max(0.0, -5.0, -0.3, 3.0, 5.0) == pytest.approx(4.7)
    assert solver.limiter_bound(0.0, -5.0, -0.3, 3.0, 5.0) == "outlier"
    # a tight spread: the average binds
    assert solver.limiter_gain_max(-4.0, -5.0, -0.3, 3.0, 5.0) == pytest.approx(7.7)
    assert solver.limiter_bound(-4.0, -5.0, -0.3, 3.0, 5.0) == "average"
    # default: the old budget + 1.5
    assert solver.limiter_gain_max(0.0, -5.0, -0.3, 3.0) == pytest.approx(4.2)


def test_speclim_limits_only_the_band_whose_peak_stands_out():
    from scipy.signal import butter, sosfilt

    from eluate import bands
    x = _tone_loop(seed=6, spread_db=1.0)                  # 200 Hz hits, band 1 of 100/400/...
    rng = np.random.default_rng(2)
    i = int(13 * BEAT * FS)
    burst = 0.5 * rng.standard_normal((int(0.004 * FS), 2))
    sos = butter(4, 8000, "hp", fs=FS, output="sos")
    x[i:i + burst.shape[0]] += sosfilt(sos, burst, axis=0)  # one hi-hat spike above 6.4 kHz
    xo = [100.0, 400.0, 1600.0, 6400.0]
    p = mb.design_speclim(x, FS, xo, 2.0, 6.0, 2.0, 50.0, hits=_hits(x))
    assert p is not None and p.budget["kind"] == "speclim"
    rows = p.budget["result"]
    assert rows[4][0]["ceiling_db"] is not None and rows[4][0]["cut_db"] > 2.0
    assert rows[4][0]["cut_db"] <= 6.0 + 1e-9
    assert rows[1][0]["ceiling_db"] is None              # the 200 Hz hits sit within 2 dB
    y = mb.apply(x, FS, p)
    bx = bands.split(bands.to_ms(x), FS, xo)
    by = bands.split(bands.to_ms(y), FS, xo)
    pk = lambda b, k: 20 * np.log10(np.abs(b[k, :, 0]).max())  # noqa: E731
    assert pk(by, 4) <= rows[4][0]["ceiling_db"] + 0.05
    assert pk(by, 1) == pytest.approx(pk(bx, 1), abs=0.05)


def test_multiband_gates_a_near_silent_band_channel():
    x = _loop()
    x[:, 1] = x[:, 0] * (1 + 1e-3)          # S about 66 dB below M: near silent
    p0 = mb.design(x, FS, -20.0, ratio=1.5)
    p0.budget = mb.default_budget("peak")
    p = mb.design_budget(x, FS, p0, hits=_hits(x))
    for row in p.budget["result"]:
        assert not row[0].get("gated")
        assert row[1]["gated"] and row[1]["comp_db"] == pytest.approx(0.0, abs=1e-6)
        assert row[1]["makeup_db"] == pytest.approx(0.0, abs=1e-6)


def test_speclim_gates_a_near_silent_band_channel():
    """Audit 08 #11: speclim leaves a near-silent band channel alone, as the compressor does,
    however far a stray peak stands above its typical hit."""
    x = _loop()
    x[:, 1] = x[:, 0] * (1 + 2e-2)          # S about 36 dB below the input
    i = len(x) // 2
    x[i:i + 3, 0] += 0.9                     # a click: band 4 M needs a ceiling, and band 4 S
    x[i:i + 3, 1] = x[i:i + 3, 0] * (1 + 2e-2) + 5e-2   # would get one too without the gate
    p = mb.design_speclim(x, FS, [100.0, 400.0, 1600.0, 6400.0], 2.0, 6.0, 2.0, 50.0, hits=_hits(x))
    rows = p.budget["result"]
    assert rows[4][0]["ceiling_db"] is not None and not rows[4][0]["gated"]
    assert rows[4][1]["gated"] and rows[4][1]["ceiling_db"] is None and rows[4][1]["cut_db"] == 0.0


def test_limiter_budget_on_the_average_with_the_outlier_cap():
    x = _loop(seed=6)
    h = _hits(x)
    lp = LimiterPlan(ceiling_dbtp=-0.3, max_gr_db=3.0)
    b = solver.limiter_input_gain(x, FS, lp, requested_db=30.0, hits=h)
    assert b["warnings"] and b["warnings"][0]["stage"] == "limiter"
    assert b["planned_gr_db"] <= 3.0 + 1e-9 and b["planned_outlier_gr_db"] <= 4.5 + 1e-9
    assert max(b["planned_gr_db"] - 3.0, b["planned_outlier_gr_db"] - 4.5) == pytest.approx(0.0,
                                                                                         abs=1e-9)
    from dataclasses import replace
    _, info = solver.limit_blocked(x, FS, replace(lp, input_gain_db=b["input_gain_db"]))
    assert info["max_gain_reduction_db"] <= 4.5 + 0.05
    # the rendered limiter's per-hit GR has the planned average
    m = solver.limit_blocked(x, FS, replace(lp, input_gain_db=b["input_gain_db"]),
                             measure_only=True, detector_env=True)
    gr = np.maximum(solver.hit_peaks_db(m["detector_env"], FS, h["edges_s"]) + 0.3, 0.0)
    assert float(np.median(gr[h["loud"]])) == pytest.approx(b["planned_gr_db"], abs=TOL)
    old = solver.limiter_input_gain(x, FS, lp, requested_db=30.0)
    assert b["input_gain_db"] > old["input_gain_db"] + 0.5


def test_limiter_outlier_spike_binds_the_cap():
    x = _loop(seed=6, spike_at=(5,), spike_db=8.0)
    lp = LimiterPlan(ceiling_dbtp=-0.3, max_gr_db=3.0)
    b = solver.limiter_input_gain(x, FS, lp, requested_db=30.0, hits=_hits(x))
    assert b["planned_outlier_gr_db"] == pytest.approx(4.5, abs=1e-9)
    assert b["planned_gr_db"] < 3.0
    w = b["warnings"][0]
    assert w["outlier_cap_db"] == 4.5 and w["outlier_needed_db"] > 4.5
    # the pure clamp reads the same record
    rec = LimiterPlan(ceiling_dbtp=-1.0, max_gr_db=3.0,
                      budget={"limiter": {"clipped_peak_dbfs": -4.0,
                                          "clipped_average_peak_dbfs": -6.0}})
    out = planner.clamp_limiter_gain(rec, 20.0)
    # average: -1 + 3 + 6 = 8; outlier: -1 + 4.5 + 4 = 7.5 binds
    assert out.input_gain_db == pytest.approx(7.5)
    lr = out.budget["limiter"]
    assert lr["planned_outlier_gr_db"] == pytest.approx(4.5)
    assert lr["planned_gr_db"] == pytest.approx(-6.0 + 7.5 + 1.0)


# ------------------------------------------------------------ plan and report

def test_hits_plan_round_trip_and_validation():
    p = Plan(fs=FS, hits=HitsPlan(beats_s=[0.5, 1.0], source="user bpm"))
    q = Plan.from_json(p.to_json())
    assert q.hits == p.hits and q.version == PLAN_VERSION
    with pytest.raises(ValueError):
        HitsPlan(reading="loudest")
    assert planner.stage_hits(Plan(fs=FS), np.zeros((FS, 2)), FS) is None
    assert planner.stage_hits(Plan(fs=FS, hits=HitsPlan(reading="max")), np.zeros((FS, 2)),
                              FS) is None


def test_plan_hits_follow_the_grid():
    a = {"plan_inputs": {"beat_times": [0.5, 1.0, 1.5], "grid_informative": False}}
    assert planner.plan_hits(a, planner.Controls()).beats_s is None
    assert planner.plan_hits(a, planner.Controls(bpm=120.0)).source == "user bpm"
    a["plan_inputs"]["grid_informative"] = True
    h = planner.plan_hits(a, planner.Controls())
    assert h.beats_s == [0.5, 1.0, 1.5] and h.source == "analysis beat grid"


def test_report_table_shows_average_outlier_budget_cap_and_status():
    from eluate import evaluate
    rep = {"budget": {"hits": "user bpm", "stages": {
        "multiband": {"bands": [[
            {"comp_db": 2.5, "comp_outlier_db": 4.1, "comp_target_db": 2.5, "comp_cap_db": 4.5,
             "limit_db": 0.0, "limit_outlier_db": 0.4, "limit_cap_db": 3.0, "status": "ok"},
            {"comp_db": 2.5, "comp_outlier_db": 2.9, "comp_target_db": 2.5, "comp_cap_db": 4.5,
             "limit_db": 0.0, "limit_outlier_db": 0.0, "limit_cap_db": 3.0, "status": "ok"}]]},
        "glue": {"mode": "serial", "net_db": 1.5, "net_outlier_db": 3.2, "net_target_db": 1.5,
                 "outlier_cap_db": 3.5, "status": "ok"},
        "clipper": {"planned_db": 0.4, "outlier_db": 1.5, "target_db": 1.0,
                    "outlier_cap_db": 1.5, "status": "under"},
        "limiter": {"planned_db": 3.0, "outlier_db": 4.2, "budget_db": 3.0,
                    "outlier_cap_db": 4.5, "status": "ok"}}}}
    md = "\n".join(evaluate._budget_md(rep))
    assert "average peak = the typical loud hit; outlier = the single largest moment" in md.lower()
    assert "| stage | band | channel | average peak | outlier peak | budget | outlier cap | status |" in md
    assert "| glue (serial, net) | - | both | 1.50 | 3.20 | 1.5 | 3.5 | ok |" in md
    assert "| clipper (output peak) | - | both | 0.40 | 1.50 | 1.0 | 1.5 | under |" in md
    assert "| limiter | - | both | 3.00 | 4.20 | 3.0 | 4.5 | ok |" in md
    assert md.count("| band limiter |") == 1


# ------------------------------------------------------------ the master

def drum_loop(fs=FS, beats=32, seed=3):
    """A 120 BPM loop: a kick on every beat (30 ms rise, levels varying over 8 dB), a
    snare on the off beats, hats between, a pumping 55 Hz bass and a pink-noise pad."""
    from scipy.signal import lfilter
    rng = np.random.default_rng(seed)
    n = int(beats * BEAT * fs)
    t = np.arange(n) / fs
    x = np.zeros((n, 2))
    tk = np.arange(int(0.25 * fs)) / fs
    kick = np.sin(2 * np.pi * (50 + 120 * np.exp(-tk / 0.02)) * tk) * np.exp(-tk / 0.12) \
        * (1 - np.exp(-tk / 0.03))
    ts = np.arange(int(0.15 * fs)) / fs
    th = np.arange(int(0.04 * fs)) / fs
    for b in range(beats):
        i = int(b * BEAT * fs)
        lvl = 10 ** (rng.uniform(-8.0, 0.0) / 20)
        seg = 0.5 * lvl * kick[: n - i]
        x[i:i + seg.size] += seg[:, None]
        if b % 2 == 1:
            sn = 0.2 * lvl * rng.standard_normal((ts.size, 2)) * np.exp(-ts / 0.04)[:, None]
            x[i:i + ts.size] += sn[: n - i]
        j = int((b + 0.5) * BEAT * fs)
        hh = 0.06 * rng.standard_normal((th.size, 2)) * np.exp(-th / 0.01)[:, None]
        x[j:j + th.size] += hh[: max(0, n - j)]
    w = rng.standard_normal((n, 2))
    pink = lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786],
                   [1, -2.494956002, 2.017265875, -0.522189400], w, axis=0)
    x += 0.02 * pink / np.std(pink)
    x += 0.04 * np.sin(2 * np.pi * 55 * t)[:, None] \
        * (0.6 + 0.4 * np.sin(2 * np.pi * t / (4 * BEAT)) ** 2)[:, None]
    return x


def _crest_db(y):
    """The corpus crest: sample peak minus the loudest-section 1 s RMS."""
    from eluate import meter
    return float(20 * np.log10(np.max(np.abs(y))) - meter.rms_1s(y, FS)["loud_section_dbfs"])


@pytest.mark.slow
def test_average_reading_reduces_the_crest_more_than_the_max_reading():
    from dataclasses import replace
    x = drum_loop()
    a = planner.analyse(x, FS, bpm=BPM)
    p0 = planner.plan_from_analysis(a, planner.Controls(bpm=BPM, character="fast"))
    assert p0.hits.source == "user bpm" and len(p0.hits.beats_s) >= 30
    red = {}
    for reading in ("max", "average"):
        p = planner.realise(replace(p0, hits=replace(p0.hits, reading=reading)), x, FS)
        y, info, _ = solver.render_plan(x, FS, p)
        red[reading] = _crest_db(x) - _crest_db(y)
        b = planner.budget_report(p, info)["stages"]
        assert b["limiter"]["planned_db"] <= 3.0 + 1e-9
        assert b["limiter"]["outlier_db"] <= planner.LIMITER_OUTLIER_CAP_DB + 0.05
        assert b["glue"]["net_outlier_db"] <= glue.SAMPLE_MAX_GR_DB + 1e-6
        assert b["clipper"]["outlier_db"] <= planner.CLIP_SAMPLE_CAP_DB + 0.05
    print(f"crest reduction: max reading {red['max']:.2f} dB, average {red['average']:.2f} dB")
    # both readings take the compressors on the meter now (2026-10-19), which narrows the
    # gap the average reading opened on the clipper and limiter
    assert red["average"] >= red["max"] + 0.5


# ------------------------------------------------------------ split clipper (2026-10-20)

@pytest.mark.slow
def test_split_clipper_threshold_ignores_the_sub():
    from dataclasses import replace
    from eluate.stages import limiter as lim
    x = _tone_loop(seed=4, spread_db=1.0)
    h = _hits(x)
    t = np.arange(x.shape[0]) / FS
    xs = x + (0.5 * np.sin(2 * np.pi * 40.0 * t))[:, None]       # a sub note under the loop
    lp = LimiterPlan(clip_knee_db=1.0, clip_target_db=1.0, clip_split_hz=100.0,
                     clip_max_gr_db=3.0)      # the planner's per-sample cap, no outlier re-solve
    a = solver.clip_threshold_for_average_peak(x, FS, lp, 1.0, h)
    b = solver.clip_threshold_for_average_peak(xs, FS, lp, 1.0, h)
    assert a["peak_gr_db"] == pytest.approx(1.0, abs=TOL)
    assert b["peak_gr_db"] == pytest.approx(1.0, abs=TOL)
    assert abs(a["threshold_db"] - b["threshold_db"]) < 0.3     # the sub does not move it
    c = solver.clip_threshold_for_average_peak(xs, FS, replace(lp, clip_split_hz=None), 1.0, h)
    assert c["threshold_db"] > b["threshold_db"] + 1.0           # the full-band one sat on the sine
    # the solve's reading is the high part's per-hit output-peak fall
    xh = lim.clip_hi_part(xs, FS, 100.0)
    e = h["edges_s"]
    m0 = solver.limit_blocked(xh, FS, LimiterPlan(), measure_only=True, detector_env=True)
    m1 = solver.limit_blocked(xh, FS, LimiterPlan(clip_threshold_db=b["threshold_db"],
                                                  clip_knee_db=1.0),
                              measure_only=True, detector_env=True)
    fall = solver.hit_peaks_db(m0["detector_env"], FS, e) - solver.hit_peaks_db(
        m1["detector_env"], FS, e)
    assert float(np.median(fall[h["loud"]])) == pytest.approx(b["peak_gr_db"], abs=1e-6)
    rec = planner.solve_clipper(xs, FS, lp, h).budget["clipper"]
    assert rec["split_hz"] == 100.0 and "above 100 Hz" in rec["measured_on"]
    # the proxy clips the high part too: its output keeps the sine where the full-band one did not
    y_split = solver.limit_proxy(xs, FS, replace(lp, clip_threshold_db=b["threshold_db"],
                                                 ceiling_dbtp=0.0), 0.0)
    low = lim.bands.split(y_split - xs, FS, [100.0], phase="zero")[0]
    assert 20 * np.log10(np.sqrt(np.mean(low ** 2)) / np.sqrt(np.mean(xs ** 2))) < -40
