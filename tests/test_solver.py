"""Solver and blocked limiter (spec 1, 3.6)."""

import copy
import tracemalloc

import numpy as np
import pytest

from fixtures import synthetic as syn
from fixtures.e_signals import short_analysis
from eluate import envelope, meter, parallel, planner, solver
from eluate.plan import LimiterPlan
from eluate.stages import limiter as lim

FS = 44100


def _hot(fs, seconds=3.0):
    x, meta = syn.kick_sub(fs, bars=2)
    n = int(seconds * fs)
    x = np.resize(x, (n, 2))
    rng = np.random.default_rng(1)
    x = x + 0.2 * rng.standard_normal((n, 2)) * np.linspace(0.2, 1, n)[:, None]
    t = np.arange(n) / fs
    x[:, 0] += 0.5 * np.sin(2 * np.pi * 15000 * t) * (t > seconds / 2)
    per = len(meta["kick_onsets"]) * 60.0 / meta["bpm"]
    on = [float(o) for o in np.concatenate([meta["kick_onsets"], meta["kick_onsets"] + per])
          if o < seconds]
    return x, on


@pytest.mark.parametrize("fs", [44100, 48000])
@pytest.mark.parametrize("kind", ["plain", "clip", "clip_slow"])
def test_blocked_limiter_matches_whole_track(fs, kind):
    x, on = _hot(fs)
    plan = {"plain": LimiterPlan(ceiling_dbtp=-1.0, input_gain_db=6.0),
            "clip": LimiterPlan(ceiling_dbtp=-1.0, input_gain_db=8.0, clip_depth_db=2.5,
                                clip_onsets_s=on),
            "clip_slow": LimiterPlan(ceiling_dbtp=-1.0, input_gain_db=3.0, clip_depth_db=3.0,
                                     clip_onsets_s=on, release_ms=200.0)}[kind]
    y0, i0 = lim.apply_with_info(x, fs, plan)
    for blk in (4096, 20000, 1 << 17):
        y1, i1 = solver.limit_blocked(x, fs, plan, block=blk)
        assert np.max(np.abs(y1 - y0)) < 1e-9
        for k in ("max_gain_reduction_db", "clip_max_reduction_db", "clip_spans", "trim_db"):
            assert i1[k] == pytest.approx(i0[k], abs=1e-9), k
        assert len(i1["relimit_passes"]) == len(i0["relimit_passes"])
        assert max(i1["true_peak_8x_dbtp"], i1["true_peak_32x_dbtp"]) <= plan.ceiling_dbtp + 1e-9


@pytest.mark.slow
def test_blocked_limiter_memory_below_whole_track():
    parallel.set_threads(1)        # one block in flight (tests/test_parallel.py: more)
    fs = 44100
    x, on = _hot(fs, seconds=24.0)
    plan = LimiterPlan(input_gain_db=6.0, clip_depth_db=2.0, clip_onsets_s=on)
    one_8x = x.shape[0] * 8 * 2 * 8          # bytes of one whole-track stereo 8x array
    tracemalloc.start()
    solver.limit_blocked(x, fs, plan)
    blocked = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    tracemalloc.start()
    lim.apply_with_info(x, fs, plan)
    whole = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    # measured: whole-track 1022 MB, blocked 199 MB (the base-rate re-limit pass and
    # the 32x FFT check dominate; no 8x array spans more than one block)
    assert whole > 4 * one_8x
    assert blocked < 0.35 * whole


def test_hull_carry_is_exact_recursion():
    rng = np.random.default_rng(3)
    u = np.clip(1 - 0.5 * rng.random(5000) ** 8, 0.05, 1.0)
    full = envelope.release_hull(u, 300.0)
    with np.errstate(divide="ignore"):
        carry = float(-20 * np.log10(full[2499]))
    tail = solver._hull_carry(u[2500:], 300.0, carry)
    assert np.max(np.abs(tail - full[2500:])) < 1e-12


def test_proxy_limits_near_ceiling():
    x, on = _hot(FS)
    lp = LimiterPlan(ceiling_dbtp=-1.0, clip_depth_db=1.0, clip_onsets_s=on)
    y = solver.limit_proxy(x, FS, lp, 6.0)
    assert y.shape == x.shape
    assert 20 * np.log10(np.max(np.abs(y))) <= -1.0 + 1e-9


@pytest.fixture(scope="module")
def setup():
    x, a = short_analysis(FS)
    c = planner.Controls()
    return x, a, planner.plan_from_analysis(a, c), planner.targets_from(a, c)


def test_apply_levers_writes_gain_rotation_and_ceiling(setup):
    x, a, p0, t = setup
    lv = solver.Levers.from_plan(p0)
    assert set(vars(lv)) == {"drive_db", "tone_rotation", "limiter_ceiling_dbtp"}
    lv.drive_db, lv.limiter_ceiling_dbtp = 9.0, -1.2
    p = solver.apply_levers(p0, lv)
    assert p.limiter.input_gain_db == 9.0 and p.limiter.ceiling_dbtp == -1.2
    # every dynamics stage keeps its budget record: the solve at design sets it
    assert p.multiband == p0.multiband and p.glue == p0.glue
    assert p.limiter.clip_target_db == planner.CLIP_TARGET_DB
    assert p.limiter.target_rms_dbfs == t.rms_dbfs == planner.DEFAULT_TARGET_RMS_DBFS
    assert p0.limiter.input_gain_db != 9.0                       # plan0 untouched


def test_tone_rotation_lever_refits_the_input_eq(setup):
    _, a, p0, _ = setup
    lv = solver.Levers.from_plan(p0)
    assert solver.apply_levers(p0, lv).inputeq == p0.inputeq      # no rotation: as planned
    lv.tone_rotation = 0.5
    p = solver.apply_levers(p0, lv)
    # the same EQ the planner fits with the rotation added to its target
    ref, rec = planner.plan_inputeq(a, planner.Controls(), FS, rotation=0.5)
    assert p.inputeq == ref and p.inputeq.design["rotation"] == 0.5
    assert p0.inputeq.design["rotation"] == 0.0                    # plan0 untouched
    # the short track pins most sections at their bounds, so only the direction is tested
    # here (test_planner checks the slope the fit reaches on unbounded material)
    _, rec0 = planner.plan_inputeq(a, planner.Controls(), FS)
    _, recm = planner.plan_inputeq(a, planner.Controls(), FS, rotation=-0.5)
    assert recm["eq_slope_db_oct"] < rec0["eq_slope_db_oct"] < rec["eq_slope_db_oct"]
    assert p.tone is None


@pytest.mark.slow
def test_chain_rerenders_pre_stages_when_the_rotation_changes(setup):
    """The input EQ sits before the cached stages 1-2, so a tone rotation must re-render
    them and everything after; other levers reuse the cache."""
    x, _, p0, _ = setup
    xs = x[: 4 * FS]
    with solver.Chain(xs, FS, p0) as ch:
        lv = solver.Levers.from_plan(p0)
        pre0 = ch.pre
        assert np.array_equal(pre0, solver.render_pre(xs, FS, p0))
        _, z0 = ch.pre_limiter(lv)
        lv_d = copy.deepcopy(lv)
        lv_d.drive_db += 3.0                         # drive only: nothing re-rendered
        path = ch._pre[1]
        _, z_d = ch.pre_limiter(lv_d)
        assert ch._pre[1] == path and np.array_equal(z_d, z0)
        lv_r = copy.deepcopy(lv)
        lv_r.tone_rotation = 0.6
        p_r, z_r = ch.pre_limiter(lv_r)
        assert p_r.inputeq != p0.inputeq
        assert np.array_equal(ch.pre, solver.render_pre(xs, FS, p_r))
        assert not np.array_equal(ch.pre, pre0) and not np.array_equal(z_r, z0)
        # the cached result equals an uncached chain on the refitted plan
        with solver.Chain(xs, FS, solver.apply_levers(p0, lv_r)) as fresh:
            lv_f = copy.deepcopy(lv)
            _, z_f = fresh.pre_limiter(lv_f)
        assert np.array_equal(z_r, z_f)
        _, z_back = ch.pre_limiter(lv)               # back to no rotation
        assert np.array_equal(z_back, z0)


def test_crest_excess_flags_band_pushed_below_gate():
    ref = np.array([[10.0, np.nan, 10.0], [12.0, 12.0, 12.0]])
    out = np.array([[9.0, np.nan, np.nan], [8.0, 11.5, 12.0]])
    loss, ex = solver.crest_excess(ref, out, [1.5, 2.0])
    assert ex[0, 0] == pytest.approx(-0.5)
    assert np.isnan(ex[0, 1])          # ungated in the premaster: no constraint
    assert ex[0, 2] == np.inf          # gated before, not after: violation (audit 06 F5)
    assert ex[1, 0] == pytest.approx(2.0)
    assert list(solver._worst(ex)) == [np.inf, pytest.approx(2.0)]


def test_section_deltas_and_misses():
    d = solver.section_deltas([-10.0, -12.0, -35.0, -np.inf])
    assert d[0] == 0.0 and d[1] == -2.0 and np.isnan(d[2]) and np.isnan(d[3])
    cons = [{"name": "crest_band0", "value": 3.0, "target": 0.0, "excess": 0.3, "tol": 1.5,
             "ok": False},
            {"name": "rms", "value": -8.2, "target": -7.7, "excess": 0.2, "tol": 0.3,
             "ok": False},
            {"name": "true_peak", "value": -0.4, "target": -0.3, "excess": -0.1, "tol": 0.0,
             "ok": True}]
    ms = solver.misses(cons)
    assert [m["name"] for m in ms] == ["rms", "crest_band0"]       # largest normalised first
    assert set(ms[0]) == {"name", "value", "target", "tol", "excess"}
    assert solver.misses([cons[2]]) == []


@pytest.mark.slow
def test_render_plan_equals_stages_render_with_blocked_limiter(setup):
    from eluate import stages
    x, a, p0, _ = setup
    p = planner.realise(p0, x, FS)
    xs = x[: 6 * FS]
    ps = copy.deepcopy(p)
    ps.notch = None                       # notch envelopes span the full track; skip here
    ps.output = None                      # compared before the export rounding
    y_ref, _ = stages.render(xs, FS, ps)
    y_blk, info, _ = solver.render_plan(xs, FS, ps)
    assert np.max(np.abs(y_blk - y_ref)) < 1e-9
    assert info["blocked"] is True


@pytest.mark.slow
def test_solve_reports_misses_and_warnings(setup):
    x, a, p0, t = setup
    logs = []
    res = solver.solve(x, FS, p0, t, log=logs.append)
    assert res.renders <= solver.MAX_OUTER + 2
    tp = max(res.lim_info["true_peak_8x_dbtp"], res.lim_info["true_peak_32x_dbtp"])
    assert tp <= t.true_peak_max_dbtp + 1e-9
    rms = meter.rms_meter(res.master, FS)["loud_dbfs"]
    assert rms == pytest.approx(res.metrics["rms_dbfs"])
    assert res.metrics["s95_lufs"] == pytest.approx(meter.loud_passage_lufs(res.master, FS)["s95"])
    assert "s95" not in {c["name"] for c in res.metrics["constraints"]}   # reported only
    # S4: a limiter held at its budget fails nothing while the loudness is accepted
    blocking = [w for w in res.warnings if not (w["stage"] == "limiter" and w.get("accepted"))]
    assert res.ok == (not res.misses and not blocking)
    assert any("render 1" in m for m in logs)
    summ = res.summary()
    assert summ["renders"] == res.renders and summ["misses"] == res.misses
    assert summ["reported"]["s95_lufs"] == res.metrics["s95_lufs"]
    # the contrast per stage, the master last (D2)
    by = summ["reported"]["section_change_by_stage"]
    assert by[0]["stage"] == "input EQ, notch, sidestrip" and by[-1]["stage"] == "limiter, output"
    assert by[-1]["contrast_loss_lu"] == pytest.approx(summ["reported"]["section_contrast_loss_lu"])
    assert res.plan.hits.loud is not None                          # the frozen mask is saved
    assert all("section_contrast_loss_lu" in h for h in res.history if "move" in h)
    assert res.plan.limiter.input_gain_db == pytest.approx(res.levers.drive_db)


@pytest.fixture(scope="module")
def dense():
    from fixtures.e_signals import dense_planner_analysis
    from dataclasses import replace
    x, a = dense_planner_analysis(FS)
    c = planner.Controls()
    p0 = planner.plan_from_analysis(a, c)
    # the fixture is a pure 55 Hz harmonic stack with fixed phases: the minimum-phase
    # high-pass (R7e) rotates its fundamental against the harmonics and lifts its peaks
    # by about 1.2 dB, which no real premaster showed (-0.5 to +0.3 dB on the VA tracks);
    # these tests are about the budgets, so the fixture runs without it
    p0 = replace(p0, inputeq=replace(p0.inputeq, highpass=None))
    return x, a, p0, planner.targets_from(a, c)


@pytest.mark.slow
def test_solve_meets_the_rms_target_when_the_budgets_suffice(dense):
    x, a, p0, t = dense
    res = solver.solve(x, FS, p0, t)
    rms = meter.rms_meter(res.master, FS)["loud_dbfs"]
    assert abs(rms - planner.DEFAULT_TARGET_RMS_DBFS) <= 0.3
    assert not any(w["stage"] in ("limiter", "loudness") for w in res.warnings)
    lb = res.metrics["budget"]["stages"]["limiter"]
    assert lb["rendered_db"] <= planner.LIMITER_MAX_GR_DB + 0.05
    assert "rms" not in [m["name"] for m in res.misses]
    # another target, same track: still met
    t2 = copy.deepcopy(t)
    t2.rms_dbfs = -9.0
    res2 = solver.solve(x, FS, p0, t2, max_outer=4)
    assert abs(meter.rms_meter(res2.master, FS)["loud_dbfs"] + 9.0) <= 0.3


@pytest.mark.slow
def test_solve_clamps_at_the_limiter_budget_and_states_the_shortfall():
    """A very dynamic track: the target needs more than 3 dB of limiter peak GR, so the
    gain is clamped at the budget, with the measured shortfall in the warning."""
    from fixtures.e_signals import short_planner_analysis
    x, a = short_planner_analysis(FS)
    c = planner.Controls(target_rms=-2.5)
    p0 = planner.plan_from_analysis(a, c)
    t = planner.targets_from(a, c)
    res = solver.solve(x, FS, p0, t, max_outer=3)
    rms = meter.rms_meter(res.master, FS)["loud_dbfs"]
    lw = [w for w in res.warnings if w["stage"] == "limiter"]
    assert lw, res.warnings
    w = lw[0]
    assert w["needed_db"] > w["cap_db"] == planner.LIMITER_MAX_GR_DB
    assert w["shortfall_db"] == pytest.approx(-2.5 - rms) and w["shortfall_db"] > 0.3
    assert w["shortfall_source"] == "render"
    ld = [w for w in res.warnings if w["stage"] == "loudness"]
    assert ld and ld[0]["cause"] == "limiter budget"
    assert ld[0]["shortfall_db"] == pytest.approx(w["shortfall_db"])
    assert "rms" in [m["name"] for m in res.misses] and not res.ok
    # the clamp holds the limiter at its budget
    assert res.metrics["budget"]["stages"]["limiter"]["planned_db"] == pytest.approx(
        planner.LIMITER_MAX_GR_DB, abs=1e-6)
    assert all(h.get("clamped") for h in res.history if "move" in h)
    assert not any(h.get("move") == "loudness" for h in res.history)   # nothing left to gain


@pytest.mark.slow
def test_solve_slope_first_then_loudness(setup):
    x, a, p0, t = setup
    t3 = copy.deepcopy(t)
    t3.slope_db_oct = t.slope_db_oct - 1.5       # force the slope out of tolerance
    res = solver.solve(x, FS, p0, t3, max_outer=4)
    moves = [h["move"] for h in res.history if "move" in h]
    assert "tone rotation" in moves, res.history
    assert set(moves) <= {"planned", "true-peak ceiling", "tone rotation", "loudness"}
    for h in res.history:
        if h.get("kept") == "tone rotation":
            assert h["score_after"] < h["score_before"]


@pytest.mark.slow
def test_gain_for_rms_hits_the_target_on_the_proxy(setup):
    x, a, p0, t = setup
    p = planner.realise(p0, x, FS)
    z = solver.render_chain(solver.render_pre(x, FS, p), FS, p)
    g = solver.gain_for_rms(z, FS, p.limiter, -14.0, 0.0)
    assert solver.proxy_rms(z, FS, p.limiter, g) == pytest.approx(-14.0, abs=0.02)
    g2 = solver.gain_for_rms(z, FS, p.limiter, -14.0, 0.0, bias_db=1.0)
    assert solver.proxy_rms(z, FS, p.limiter, g2) == pytest.approx(-15.0, abs=0.02)


def test_crest_excess_exempt_view():
    ref = np.array([[10.0, 12.0, 10.0]])
    out = np.array([[9.0, 5.0, 9.5]])
    _, ex = solver.crest_excess(ref, out, [1.5], exempt=[[0, 1]])
    assert np.isnan(ex[0, 1]) and solver._worst(ex)[0] == pytest.approx(-0.5)


@pytest.mark.slow
def test_measure_preserve_mode_gain_only_is_ok(setup):
    x, a, p0, t = setup
    tp = copy.deepcopy(t)
    tp.preserve_mix = True
    tp.rms_dbfs = meter.rms_meter(x * 0.5, FS)["loud_dbfs"]
    ref = solver.Reference(x, FS, tp)
    plan = planner.realise(planner.plan_from_analysis(a, planner.Controls(preserve_mix=True)),
                           x, FS)
    m = solver.measure(x * 0.5, FS, plan, {}, ref)
    names = {c["name"]: c for c in m["constraints"]}
    assert "comp_gr" not in names and m["comp_gr_max_db"]          # reported only (A12)
    assert "slope" not in names and names["ltas_shape"]["ok"] and names["section_deltas"]["ok"]
    assert names["ltas_shape"]["value"] < 1e-6
    assert names["section_deltas"]["value"] < 1e-6
    assert all(names[f"crest_band{b}"]["ok"] for b in range(5))
    assert names["rms"]["ok"] and names["rms"]["value"] == pytest.approx(tp.rms_dbfs)



def test_blocked_limiter_rejects_qp():
    x, on = _hot(FS, 1.0)
    with pytest.raises(ValueError, match="box"):
        solver.limit_blocked(x, FS, LimiterPlan(smoother="qp"))


@pytest.mark.parametrize("n", [2, 5, 333, 4096, 4097, 8191])
def test_blocked_limiter_short_and_odd_lengths(n):
    rng = np.random.default_rng(n)
    x = rng.standard_normal((n, 2)) * 1.5
    plan = LimiterPlan(input_gain_db=3.0, clip_depth_db=2.0, clip_onsets_s=[0.0, n / FS],
                       release_ms=0.0)
    y0, i0 = lim.apply_with_info(x, FS, plan)
    y1, i1 = solver.limit_blocked(x, FS, plan, block=1024)
    assert np.max(np.abs(y1 - y0)) < 1e-9
    assert i1["clip_spans"] == i0["clip_spans"]


def test_blocked_limiter_mono_and_spans_on_block_edges():
    fs = 48000
    x, on = _hot(fs, 2.0)
    blk = 4000
    edges = [k * blk / fs for k in range(1, 20)]          # onsets exactly on block edges
    plan = LimiterPlan(input_gain_db=6.0, clip_depth_db=3.0, clip_onsets_s=edges)
    y0, _ = lim.apply_with_info(x, fs, plan)
    y1, _ = solver.limit_blocked(x, fs, plan, block=blk)
    assert np.max(np.abs(y1 - y0)) < 1e-9
    m0, _ = lim.apply_with_info(x[:, 0], fs, plan)
    m1, _ = solver.limit_blocked(x[:, 0], fs, plan, block=blk)
    assert m1.ndim == 1 and np.max(np.abs(m1 - m0)) < 1e-9


def test_blocked_limiter_peak_at_the_ceiling():
    """audit 07 A2: a peak within 1e-9 of the ceiling sits on a threshold branch, where
    the FFT rounding difference between the two implementations (about 2e-10) can flip
    the branch; the documented bound there is 0.01 dB, and both hold the ceiling."""
    fs = 48000
    t = np.arange(fs) / fs
    x = np.stack([np.sin(2 * np.pi * 997 * t)] * 2, axis=1)
    x *= 10 ** (-1.0 / 20) / meter.true_peak_envelope(x, fs, 8).max() * (1 + 1e-10)
    plan = LimiterPlan(ceiling_dbtp=-1.0)
    y0, i0 = lim.apply_with_info(x, fs, plan)
    y1, i1 = solver.limit_blocked(x, fs, plan, block=8192)
    big = np.abs(y0) > 1e-3
    d = np.max(np.abs(20 * np.log10(np.abs(y1[big]) / np.abs(y0[big]))))
    assert d < 0.01
    assert max(i1["true_peak_8x_dbtp"], i1["true_peak_32x_dbtp"]) <= -1.0 + 1e-9


@pytest.mark.parametrize("s1,s2", [(1, 5), (5, 1), (7, 7), (100, 300), (2048, 91), (2049, 91),
                                   (100003, 2905), (65536, 23)])
def test_oa_stream_edges(s1, s2):
    from eluate import oa
    from scipy import signal
    rng = np.random.default_rng(s1 + s2)
    a, h = rng.standard_normal(s1), rng.standard_normal(s2)
    out = np.full(s1 + s2 - 1, np.nan)

    def get(i0, i1):
        z = np.zeros(i1 - i0)
        if i0 < s1:
            seg = a[i0:min(i1, s1)]
            z[:seg.size] = seg
        return z

    def put(start, v):
        out[start:start + v.size] = v

    if oa.oa_stream(get, s1, h, put):
        assert np.array_equal(out, signal.oaconvolve(a, h, mode="full"))
    else:
        assert np.all(np.isnan(out))          # declined: the caller uses oaconvolve


def test_oa_accumulator_branch_matches_direct_convolution(monkeypatch):
    """step < overlap is never produced by scipy's _calc_oa_lens; the accumulator that
    handles it is exact to rounding (not bit-identical, documented)."""
    from eluate import oa
    rng = np.random.default_rng(3)
    a, h = rng.standard_normal(5000), rng.standard_normal(300)
    monkeypatch.setattr(oa, "_calc_oa_lens", lambda s1, s2: (512, s2 - 1, 512 - s2 + 1, s2))
    out = np.zeros(a.size + h.size - 1)

    def get(i0, i1):
        z = np.zeros(i1 - i0)
        if i0 < a.size:
            seg = a[i0:min(i1, a.size)]
            z[:seg.size] = seg
        return z

    def put(start, v):
        out[start:start + v.size] = v

    assert oa.oa_stream(get, a.size, h, put)
    assert np.max(np.abs(out - np.convolve(a, h))) < 1e-9
    monkeypatch.setattr(oa, "_calc_oa_lens", None)
    assert not oa.oa_stream(get, a.size, h, put)       # import fallback declines


def test_crest_constraint_modes(setup):
    _, _, _, t = setup
    cl = {"loss_db": np.array([[1.0, 5.0, 2.0], [0.5, 0.5, 0.5]]),
          "after_p95_db": np.array([[9.0, 4.0, 8.0], [12.0, 12.0, 12.0]]),
          "silenced": np.array([[False, False, False], [False, True, False]])}
    tt = copy.deepcopy(t)
    tt.crest_limits_db = [1.5, 2.0]
    tt.crest_mode = "loss"
    tt.crest_exempt = []
    c = solver.crest_constraints(cl, tt)
    assert c["worst"][0] == pytest.approx(3.5) and c["worst"][1] == np.inf
    tt.crest_exempt = [[0, 1]]
    assert solver.crest_constraints(cl, tt)["worst"][0] == pytest.approx(0.5)
    tt.crest_mode = "report"
    r = solver.crest_constraints(cl, tt)
    assert np.nanmax(r["worst"]) == 0.0
    tt.crest_mode, tt.crest_floors_db, tt.crest_exempt = "floor", [10.0, 11.0], []
    f = solver.crest_constraints(cl, tt)
    assert f["worst"][0] == pytest.approx(6.0) and f["worst"][1] == np.inf


def test_true_peaks_worst_meter_on_float32():
    rng = np.random.default_rng(4)
    y = rng.standard_normal((FS, 2)) * 0.2
    tp = solver.true_peaks(y, FS)
    assert tp["worst"] == max(tp["tp4"], tp["tp8"], tp["tp32"])
    assert tp["8x"] == max(tp["tp8"], tp["tp32"])
    y32 = y.astype(np.float32).astype(np.float64)
    assert tp["tp8"] == meter.true_peak(y32, FS, 8)


def test_section_deltas_shared_gate():
    ref = [-10.0, -12.0, -35.0]
    out = [-4.0, -9.0, -6.0]            # the quiet section rose: still gated on ref
    d = solver.section_deltas(out, ref_levels=ref)
    assert d[0] == 0.0 and d[1] == -5.0 and np.isnan(d[2])
    # a counted section, or the reference itself, silenced in the render is a miss (audit 08)
    lost = solver.section_deltas([-4.0, -np.inf, -6.0], ref_levels=ref)
    assert lost[1] == np.inf and np.isnan(lost[2])
    assert np.all(np.isinf(solver.section_deltas([np.nan, -9.0, -6.0], ref_levels=ref)[:2]))


@pytest.mark.slow
def test_preserve_mode_solve_under_real_processing(setup):
    x, a, _, _ = setup
    c = planner.Controls(preserve_mix=True)
    p0 = planner.plan_from_analysis(a, c)
    t = planner.targets_from(a, c)
    res = solver.solve(x, FS, p0, t, max_outer=2)
    names = {k["name"]: k for k in res.metrics["constraints"]}
    assert "ltas_shape" in names and "section_deltas" in names and "slope" not in names
    assert np.isfinite(names["ltas_shape"]["value"])
    assert np.isfinite(names["section_deltas"]["value"])
    assert res.plan.tone is None and res.plan.saturate is None and res.plan.inputeq is None


def test_rotation_refit_and_slope_warning():
    """Sprint 2026-10-21 S7 and T1: a rotation whose refit equals the current fit is
    skipped; a missed slope gets a warning naming the sections at their bound and the
    Tilt that would make the reached slope the target."""
    from eluate import evaluate, planner
    from eluate.plan import Plan
    nodes = [31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
             1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000]
    d = {"nodes_hz": nodes, "residual_db": [0.0] * len(nodes), "measured_slope": -9.0,
         "target_slope": -4.5, "sub_shelf_db": 0.0, "rotation": 0.0}
    eq, rec = planner.fit_inputeq(d, 44100)
    p0 = Plan(fs=44100, inputeq=eq)
    # far too dark: every top section is at its bound, so more rotation changes nothing
    assert any("presence shelf" in a for a in rec["at_bound"])
    assert solver._same_refit(p0, 1.0, 2.0)
    assert not solver._same_refit(p0, 0.0, -2.0)
    t = planner.Targets(slope_db_oct=-4.5)
    w = solver.slope_warning(p0, 0.0, -6.2, t)
    assert w["stage"] == "inputeq" and w["tilt_db"] == pytest.approx(-1.7)
    assert w["at_bound"]
    txt = evaluate.warning_text(w, {})
    assert "A Tilt of -1.7 dB/oct" in txt and "at its limit" in txt
