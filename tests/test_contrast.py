"""Section contrast and reporting (sprint 2026-10-21, WP7 Group C: D2, R13, S4, S5, S8, S9)."""

import copy
import json

import numpy as np
import pytest

from fixtures.e_signals import short_analysis
from eluate import envelope, evaluate, planner, processlog, solver
from eluate.plan import PLAN_VERSION, HitsPlan, Plan

FS = 44100


def _drop_and_build(fs=FS, beat=0.5):
    """A loud drop (8 s) then a build 4 dB under it (8 s), noise bursts on every beat."""
    rng = np.random.default_rng(2)
    n = 16 * fs
    t = np.arange(n) / fs
    env = np.exp(-((t % beat) / 0.08))
    lvl = np.where(t < 8.0, 1.0, 10 ** (-4.0 / 20))
    x = 0.3 * rng.standard_normal((n, 2)) * (env * lvl)[:, None]
    beats = list(np.arange(beat, 16.0, beat))
    return x, beats


# ------------------------------------------------------------------ D2: frozen loud mask

def test_hits_for_uses_a_frozen_mask_and_checks_its_length():
    x, beats = _drop_and_build()
    h = envelope.hits_for(x, FS, beats)
    assert h["loud"][:10].all() and not h["loud"][-10:].any()     # the drop only
    flat = x.copy()
    flat[8 * FS:] *= 10 ** (4.0 / 20)                             # the build lifted to the drop
    assert envelope.hits_for(flat, FS, beats)["loud"].sum() > h["loud"].sum()
    fz = envelope.hits_for(flat, FS, beats, loud=h["loud"])
    assert np.array_equal(fz["loud"], h["loud"]) and np.array_equal(fz["edges_s"], h["edges_s"])
    with pytest.raises(ValueError, match="frozen loud mask"):
        envelope.hits_for(flat, FS, beats, loud=h["loud"][:-1])


def test_freeze_hits_and_stage_hits_budget_every_stage_on_the_stage1_set():
    x, beats = _drop_and_build()
    p = Plan(fs=FS, hits=HitsPlan(beats_s=beats, source="user bpm"))
    derived = planner.stage_hits(p, x, FS)["loud"]
    pf = planner.freeze_hits(p, x, FS)
    assert p.hits.loud is None                                    # input not modified
    assert pf.hits.loud == [bool(v) for v in derived]
    flat = x.copy()
    flat[8 * FS:] *= 10 ** (4.0 / 20)
    assert np.array_equal(planner.stage_hits(pf, flat, FS)["loud"], derived)
    assert planner.stage_hits(p, flat, FS)["loud"].sum() > derived.sum()    # unfrozen: drifts
    # nothing to freeze without hits or under the maximum reading
    assert planner.freeze_hits(Plan(fs=FS), x, FS).hits is None
    pm = Plan(fs=FS, hits=HitsPlan(beats_s=beats, reading="max"))
    assert planner.freeze_hits(pm, x, FS).hits.loud is None


def test_hits_mask_round_trips_and_an_older_plan_still_loads():
    p = Plan(fs=FS, hits=HitsPlan(beats_s=[0.5, 1.0, 1.5], source="user bpm",
                                  loud=[True, False, True, True]))
    q = Plan.from_json(p.to_json())
    assert q.hits == p.hits and q.hits.loud == [True, False, True, True]
    d = json.loads(p.to_json())
    del d["hits"]["loud"]                       # a plan saved before the frozen mask
    assert d["version"] == PLAN_VERSION
    old = Plan.from_dict(d)
    assert old.hits.loud is None and old.hits.beats_s == [0.5, 1.0, 1.5]


@pytest.fixture(scope="module")
def setup():
    x, a = short_analysis(FS)
    c = planner.Controls()
    return x, a, planner.plan_from_analysis(a, c), planner.targets_from(a, c)


@pytest.mark.slow
def test_realise_and_chain_freeze_the_mask_on_the_multiband_input(setup):
    x, a, p0, t = setup
    pr = planner.realise(p0, x, FS)
    pre = solver.render_pre(x, FS, p0)
    want = [bool(v) for v in envelope.hits_for(pre, FS, p0.hits.beats_s)["loud"]]
    assert pr.hits.loud == want and p0.hits.loud is None
    assert planner.realise(p0, x, FS) == pr                       # deterministic
    with solver.Chain(x, FS, p0, sections=t.sections) as ch:
        p, _ = ch.pre_limiter(solver.Levers.from_plan(p0))
        assert p.hits.loud == want
        names = [n for n, _ in ch.stage_sections]
        assert names[0] == "input EQ, notch, sidestrip" and "multiband" in names
        assert "glue" in names
        assert all(len(v) == len(t.sections) for _, v in ch.stage_sections)
        # the cached path hands back the same frozen hits
        lv = solver.Levers.from_plan(p0)
        lv.drive_db += 2.0
        p2, _ = ch.pre_limiter(lv)
        assert p2.hits == p.hits


# ------------------------------------------------------------------ R13: score and keep test

def _m(loss, rms_excess=0.0):
    return {"constraints": [{"name": "rms", "excess": rms_excess, "tol": 0.3, "ok": True},
                            {"name": "slope", "excess": 0.0, "tol": 0.3, "ok": True}],
            "section_contrast_loss_lu": loss}


def test_score_has_a_contrast_term_past_the_flag_and_the_lever_score_does_not():
    flag = evaluate.FLAG_SECTION_LU
    assert solver.FLAG_SECTION_LU == flag == 3.0                     # one threshold (S9)
    assert solver.SECTION_IGNORE_LU == evaluate.FLAG_SECTION_IGNORE_LU
    assert solver.score(_m(flag)) == 0.0
    assert solver.score(_m(flag + 1.5)) == pytest.approx(0.5)
    assert solver.score(_m(flag + 1.5), skip=("section_contrast",)) == 0.0
    assert solver.lever_score(_m(flag + 1.5)) == 0.0                # slope first (R7a)
    assert solver.score(_m(flag + 2.0)) > solver.score(_m(flag + 1.0))


def test_judge_move_reverts_a_flatter_render_once_the_loudness_is_accepted():
    kept, flatter = _m(3.2, rms_excess=0.4), _m(3.6, rms_excess=0.0)
    # the score alone keeps the louder render (0.4 / 0.3 of RMS excess gone)
    assert solver.score(flatter) < solver.score(kept)
    assert solver.judge_move(flatter, kept, loudness_accepted=False) is None
    assert solver.judge_move(flatter, kept, loudness_accepted=True) == "section contrast"
    # within the slack, or below the flag, the move stands
    assert solver.judge_move(_m(3.25), kept, True) is None
    assert solver.judge_move(_m(2.9), _m(1.0, 0.4), True) is None
    assert solver.judge_move(_m(1.0, 0.9), _m(1.0, 0.4), True) == "score"


def test_contrast_loss_and_section_change():
    ref = solver.Reference(np.zeros((10, 2)), FS, planner.Targets(sections=[
        {"start": 0, "end": 1, "integrated_lufs": -10.0},
        {"start": 1, "end": 2, "integrated_lufs": -16.0},
        {"start": 2, "end": 3, "integrated_lufs": -40.0}]))
    d = solver.section_change([-8.0, -11.0, -12.0], ref)
    assert d[0] == 0.0 and d[1] == pytest.approx(3.0) and np.isnan(d[2])
    assert solver.contrast_loss(d) == pytest.approx(3.0)
    lost = solver.section_change([-8.0, -np.inf, -12.0], ref)
    # a lost section counts as SECTION_LOST_LU (audit 13 #2: in reshape mode it was free)
    assert lost[1] == np.inf and solver.contrast_loss(lost) == solver.SECTION_LOST_LU
    assert solver.contrast_loss([-1.0, np.nan]) == 0.0
    st = solver.stage_changes([("multiband", [-9.0, -14.0, -40.0])], ref)
    assert st[0]["stage"] == "multiband" and st[0]["change_lu"][1] == pytest.approx(1.0)
    assert st[0]["change_lu"][2] is None and st[0]["contrast_loss_lu"] == pytest.approx(1.0)


# ------------------------------------------------------------------ S8: finite penalty

@pytest.mark.slow
def test_a_lost_section_is_a_finite_penalty_under_preserve(setup):
    x, a, _, t = setup
    tp = copy.deepcopy(t)
    tp.preserve_mix = True
    assert len(tp.sections) >= 2
    s = max(tp.sections[1:], key=lambda s: s["integrated_lufs"])
    y = x * 0.5
    y[int(s["start"] * FS):int(s["end"] * FS)] = 0.0
    plan = planner.realise(planner.plan_from_analysis(a, planner.Controls(preserve_mix=True)),
                           x, FS)
    m = solver.measure(y, FS, plan, {}, solver.Reference(x, FS, tp))
    c = next(c for c in m["constraints"] if c["name"] == "section_deltas")
    assert c["value"] == pytest.approx(solver.SECTION_LOST_LU) and not c["ok"]
    assert np.isfinite(solver.score(m))


# ------------------------------------------------------------------ S4: the Loudness control

@pytest.mark.slow
def test_rms_is_met_only_within_tolerance_of_the_loudness_control(setup):
    x, a, p0, t = setup
    plan = planner.realise(p0, x, FS)
    ref = solver.Reference(x, FS, t)
    from eluate import meter
    r0 = meter.rms_meter(x, FS)["loud_dbfs"]

    def rms_con(level_dbfs, tt):
        y = x * 10 ** ((level_dbfs - r0) / 20)
        return next(c for c in solver.measure(y, FS, plan, {}, ref if tt is t else
                                              solver.Reference(x, FS, tt))["constraints"]
                    if c["name"] == "rms")
    # S4 ruled 2026-10-21: no acceptance floor under the Loudness control
    assert rms_con(t.rms_dbfs - 0.2, t)["ok"]
    c = rms_con(-4.7, t)
    assert not c["ok"] and c["excess"] == pytest.approx(0.3, abs=0.02)
    assert not rms_con(-3.5, t)["ok"]


# ------------------------------------------------------------------ S5: needed and the bracket

@pytest.mark.slow
def test_gain_for_rms_flags_a_failed_bracket(setup):
    x, a, p0, t = setup
    p = planner.realise(p0, x, FS)
    z = solver.render_chain(solver.render_pre(x, FS, p), FS, p)
    info = {}
    g = solver.gain_for_rms(z, FS, p.limiter, -14.0, 0.0, info=info)
    assert info == {"at_limit": False} and solver.DRIVE_RANGE_DB[0] < g < solver.DRIVE_RANGE_DB[1]
    g = solver.gain_for_rms(z, FS, p.limiter, +6.0, 0.0, info=info)  # above any ceiling
    assert info["at_limit"] and g == solver.DRIVE_RANGE_DB[1]


def test_warning_text_needed_from_the_render_and_accepted():
    rep = {"chain": {}}
    w = {"stage": "limiter", "needed_db": 3.7, "cap_db": 3.0, "outlier_needed_db": 5.4,
         "outlier_cap_db": 4.5, "bound": "average", "shortfall_db": 0.7,
         "shortfall_source": "render", "needed_source": "render", "accepted": True}
    s = evaluate.warning_text(w, rep)
    assert "needed at least 3.7 dB" in s and "outlier at least 5.4" in s
    assert "within the tolerance (not a miss)" in s
    s2 = evaluate.warning_text(dict(w, needed_source=None, accepted=False,
                                    request_at_drive_limit=True), rep)
    assert "at least" not in s2 and "not a miss" not in s2 and "no gain inside" in s2


# ------------------------------------------------------------------ S9 and the process log

def _rep(changes_by_stage=None):
    secs = [{"start": 0.0, "end": 10.0, "label": "intro", "before_lufs": -20.0, "after_lufs": -12.0},
            {"start": 10.0, "end": 20.0, "label": "drop", "before_lufs": -8.0, "after_lufs": -6.0},
            {"start": 20.0, "end": 30.0, "label": "outro", "before_lufs": -16.0, "after_lufs": -20.0}]
    rep = {"sections": secs, "targets": {"section_tol_lu": 1.0}, "planner": {"controls": {}}}
    if changes_by_stage:
        rep["solver"] = {"reported": {"section_change_by_stage": changes_by_stage}}
    return rep


def test_process_log_calls_out_rises_only_and_shows_the_change_per_stage():
    from eluate.plan import Envelope, GluePlan
    plan = Plan(fs=FS, glue=GluePlan(gain_db=Envelope(np.array([-1.0]))))
    by = [{"stage": "multiband", "change_lu": [2.0, 0.0, -1.0], "contrast_loss_lu": 2.0},
          {"stage": "glue", "change_lu": [3.5, 0.0, -2.0], "contrast_loss_lu": 3.5},
          {"stage": "limiter, output", "change_lu": [6.0, 0.0, -6.0], "contrast_loss_lu": 6.0}]
    d = processlog.Doc()
    worst = processlog._sections(d, _rep(by), plan, None, None)
    md = d.md()
    assert worst == pytest.approx(6.0)
    # the intro rose 6 LU (called out); the outro fell 6 LU (more contrast: not a loss)
    assert "1 section(s) rose more than 3.0 LU" in md
    assert "Contrast change after each stage" in md
    assert "| multiband | glue | limiter, output |" in md
    assert "+1.5" in md and "+2.5" in md                         # the intro's per-stage shares
    # the same calls as evaluate's section flag (one threshold, rises only)
    chk = next(c for c in evaluate._checks(_rep()) if c["kind"] == "section")
    assert chk["tol"] == evaluate.FLAG_SECTION_LU and chk["short"]
    assert "1 section(s) rose" in chk["text"] and "intro" in chk["text"]
