"""Planner / solver / CLI fixes from the 2026-11-10 audits: planner-solver S3-1 (short and
silent inputs), S3-2 (the side-shelf fit at low sample rates), S3-2b (the loud-hit gate),
S2-3 (the multiband threshold floor), S2-4 (the clipper's rendered figure), S2-4b / S4-7
(a --bpm that disagrees with the detected grid), S4-5 (level invariance of the input EQ
fit), S4-6 (the re-limit in the limiter row), S4-8 (the key gate), S4-9 (a failed run's
work dir); hygiene F1, F3, F4, F5, F18; features F4, F7, F21; the notch envelope's
quantisation (audio-rate audit). Synthetic signals only."""

import copy
import json
import math
import os
import stat
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fixtures.e_signals import (features_with_key_confidence, short_planner_analysis,
                                short_track)
from eluate import bands, cli, envelope, meter, planner, solver
from eluate.plan import Envelope, InputPlan, LimiterPlan, Notch, NotchPlan, Plan
from eluate.stages import multiband as mb
from eluate.stages import notch as notch_stage

FS = 44100


@pytest.fixture(scope="module")
def pana():
    return short_planner_analysis(FS, 2, 0.0)


# ------------------------------------------------------------ S3-2b: the loud-hit gate

def _two_bursts(floor=1e-27):
    """Two half-second noise bursts over a near-silent floor (an HPF's DC residue), 20 hits."""
    n = 10 * FS
    x = np.full((n, 2), floor)
    rng = np.random.default_rng(0)
    for b in (3, 11):
        i0 = int(b * 0.5 * FS)
        x[i0:i0 + FS // 2] = 0.3 * rng.standard_normal((FS // 2, 2))
    return x, np.arange(0.5, 10.0, 0.5)


@pytest.mark.parametrize("floor", [1e-27, 0.0])
def test_loud_hits_never_take_hits_below_the_gate(floor):
    """The 16-loudest fallback took near-silent hits (the DC case: multiband thresholds
    at -545 dBFS); now only hits whose 1 s window passes the -70 dBFS gate are loud."""
    x, beats = _two_bursts(floor)
    h = envelope.hits_for(x, FS, beats)
    loud = np.flatnonzero(h["loud"])
    assert 0 < loud.size < envelope.LOUD_MIN_HITS
    t_end, db = meter.rms_series(x, FS)
    centres = t_end - meter.RMS_WINDOW_S / 2.0
    for k in loud:                                    # every loud hit's window is gated in
        mid = 0.5 * (h["edges_s"][k] + h["edges_s"][k + 1])
        assert db[np.argmin(np.abs(centres - mid))] > meter.RMS_GATE_DBFS
    # the budget solves then read the bursts, not the floor
    hv = envelope.hit_values(np.max(np.abs(x), axis=1), FS, h["edges_s"])
    with np.errstate(divide="ignore"):
        med = 20 * np.log10(np.median(hv[h["loud"]]))
    assert med > -30.0
    assert mb._loud_top_db(np.max(np.abs(x), axis=1), FS, h) > -30.0


def test_loud_hits_fallback_still_fills_sixteen_from_gated_hits():
    n = int(32 * 0.5 * FS)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((n, 2)) * 0.05
    lv = np.linspace(-20.0, 0.0, 32)
    for k in range(32):
        x[int(k * 0.5 * FS):int((k + 1) * 0.5 * FS)] *= 10 ** (lv[k] / 20)
    e = envelope.hit_edges(n / FS, np.arange(1, 32) * 0.5)
    assert envelope.loud_hits(x, FS, e).sum() == envelope.LOUD_MIN_HITS


def test_no_hit_above_the_gate_means_no_loud_hits():
    """A hit is loud only when its 1 s window and its own RMS pass the -70 dBFS gate; when
    no hit does, the mask is all False (no loud hits), never the 16 loudest of the floor."""
    n = 10 * FS
    x = np.full((n, 2), 1e-27)
    x[3 * FS:3 * FS + 64] = 0.5                       # a 1.5 ms click: its hit is loud
    e = np.array([0.0, 2.0, 2.9999, 3.01, 5.0, 10.0])
    assert envelope.loud_hits(x, FS, e).tolist() == [False, False, True, False, False]
    # one 0.1 sample: its 1 s window reads -66 dBFS (gated in), but the 9 s hit holding
    # it reads -76 dBFS and the other hit is the floor: no hit is loud
    x = np.full((n, 2), 1e-27)
    x[FS // 2] = 0.1
    e = np.array([0.0, 9.0, 10.0])
    assert envelope.loud_hits(x, FS, e).tolist() == [False, False]
    assert envelope.has_loud(None) and envelope.has_loud({"loud": None})
    assert not envelope.has_loud({"edges_s": e, "loud": np.zeros(2, dtype=bool)})


def test_every_budget_solve_leaves_its_stage_alone_without_loud_hits():
    """S3-2b, the callers: with no loud hit the multiband, glue, clipper, frequency-dependent
    limiter and the limiter's average term do not compress (no near-silent reading)."""
    x, beats = _two_bursts(1e-27)
    h = envelope.hits_for(x, FS, beats)
    none = {"edges_s": h["edges_s"], "loud": np.zeros_like(h["loud"])}
    v = bands.split(bands.to_ms(x), FS, [100.0, 400.0, 1600.0, 6400.0])[2, :, 0]
    sol = mb.threshold_for_peak_gr(v, FS, 2.5, 2.0, 10.0, 100.0, 6.0, 1000.0, none)
    assert sol["peak_gr_db"] == 0.0 and sol["bound"] is None
    p0 = mb.design(x, FS, -20.0, crossovers_hz=[100.0, 400.0, 1600.0, 6400.0])
    p0.budget = mb.default_budget("peak")
    p = mb.design_budget(x, FS, p0, hits=none)
    assert all(r["no_loud_hits"] and r["comp_target_db"] == 0.0 and r["limit_target_db"] == 0.0
               for row in p.budget["result"] for r in row)
    assert all(np.all(np.asarray(g.values) == 0.0) for row in p.gr_db for g in row)
    assert p.ceiling_db is None
    g = planner.design_glue(x, FS, planner.glue_unrealised(None, mode="serial"), none)
    assert np.all(g.gain_db.values == 0.0) and g.budget["no_loud_hits"]
    lp = LimiterPlan(input_gain_db=0.0, clip_threshold_db=-20.0, clip_knee_db=1.0)
    c = solver.clip_threshold_for_average_peak(x, FS, lp, 1.0, none)
    assert c["peak_gr_db"] == 0.0 and c["threshold_db"] > -20.0
    pk = solver.limiter_peaks(x, FS, replace(lp, clip_threshold_db=None), none)
    assert pk["average_peak_dbfs"] == pk["peak_dbfs"]
    plan = Plan(fs=FS, hits=planner.HitsPlan(beats_s=list(beats), loud=[False] * len(none["loud"])),
                limiter=LimiterPlan(speclim_allow_db=3.0))
    assert planner.design_speclim(plan, x, FS) is None
    (w,) = [w for w in planner.budget_report(plan)["warnings"] if w["stage"] == "hits"]
    assert w["kind"] == "no loud hits" and "do not compress" in w["text"]


@pytest.mark.slow
def test_dc_premaster_budgets_on_its_real_hits():
    """The audit's DC case (0.5 DC, 8 s, edge_cases.log "DC 0.5 8s"): the input EQ's HPF
    leaves -540 dBFS between the edge transients; the multiband thresholds sat at -545 to
    -842 dBFS, the clipper's at -732 dBFS and the limiter asked for its 36 dB drive-range
    end. Now no budget is solved on a hit below the gate."""
    n = 8 * FS
    x = np.full((n, 2), 0.5)
    a = planner.analyse(x, FS)
    assert a["meta"]["crossovers"] == list(bands.DEFAULT_CROSSOVERS) or         tuple(a["meta"]["crossovers"]) == tuple(bands.DEFAULT_CROSSOVERS)   # no key lock
    p = planner.realise(planner.plan_from_analysis(a, planner.Controls(), x), x, FS)
    thr = np.asarray(p.multiband.threshold_db, dtype=np.float64)
    assert np.all(np.isfinite(thr)) and thr.min() > -120.0
    lb = p.limiter.budget
    if lb.get("clipper") is not None:
        assert p.limiter.clip_threshold_db is None or p.limiter.clip_threshold_db > -120.0
    lim = lb["limiter"]
    assert lim["clipped_average_peak_dbfs"] is None or lim["clipped_average_peak_dbfs"] > -120.0
    assert lim["requested_gain_db"] is None or lim["requested_gain_db"] < 30.0
    if p.glue is not None and p.glue.threshold_db is not None:
        assert p.glue.threshold_db > -120.0


# ------------------------------------------------------------ S2-3: the threshold floor

def _sparse_top():
    """A 110 Hz pad with 10 ms impulses every 0.5 s: the upper bands hold only the impulses,
    whose 100 ms release averaged over 300 ms cannot read the 2.5 dB target."""
    n = 4 * FS
    t = np.arange(n) / FS
    x = 0.05 * np.sin(2 * np.pi * 110 * t)
    rng = np.random.default_rng(9)
    for b in np.arange(0.1, 4.0, 0.5):
        i0, m = int(b * FS), int(0.01 * FS)
        x[i0:i0 + m] += (0.6 + 0.3 * rng.random()) * np.exp(-np.arange(m) / (0.002 * FS))
    xs = np.stack([x, x * 0.9], axis=1)
    return xs, envelope.hits_for(xs, FS, np.arange(0, 4.0, 0.5))


def test_multiband_threshold_never_sinks_below_the_band_level_floor():
    xs, hits = _sparse_top()
    bb = bands.split(bands.to_ms(xs), FS, [100.0, 400.0, 1600.0, 6400.0])
    for k in (2, 4):
        v = bb[k, :, 0]
        body = mb.body_level_db(v, FS, hits)
        # the band's level at the loud hits: a little under the loudest section here
        assert 0.0 <= meter.rms_1s(v, FS)["loud_section_dbfs"] - body < 1.0
        sol = mb.threshold_for_peak_gr(v, FS, 2.5, 2.0, 10.0, 100.0, 6.0, 1000.0, hits)
        assert sol["bound"] == "level floor"
        assert sol["threshold_db"] == pytest.approx(body - mb.BRACKET_BELOW_LEVEL_DB, abs=1e-6)
        assert sol["floor_db"] == pytest.approx(sol["threshold_db"], abs=1e-9)
        assert 1.5 < sol["peak_gr_db"] < 2.5 - mb.UNREACHABLE_TOL_DB
        # the old step-down went 48 dB down into the leakage floor (-70 to -76 dBFS)
        assert sol["threshold_db"] > -62.0
    # a dense band reaches its target with the floor out of the way
    sol0 = mb.threshold_for_peak_gr(bb[0, :, 0], FS, 2.5, 2.0, 30.0, 200.0, 6.0, 1000.0, hits)
    assert sol0["bound"] is None and sol0["peak_gr_db"] == pytest.approx(2.5, abs=0.01)


def test_threshold_floor_follows_the_loud_hits_not_a_quiet_intro():
    """The floor's body is the band's level at the loud hits (multiband.body_level_db). A
    median over every gated window followed a quiet intro instead: 12 s of hiss before 8 s
    of the sparse pattern put it 23 dB under the loud section and the search ran to
    -73.5 dBFS, the descent the floor exists to stop (final audit 2026-11-10)."""
    n, lead = 20 * FS, 12 * FS
    rng = np.random.default_rng(9)
    x = 0.05 * np.sin(2 * np.pi * 110 * np.arange(n) / FS)
    x[:lead] += 10 ** (-50.0 / 20) * rng.standard_normal(lead)
    for b in np.arange(12.1, 20.0, 0.5):
        i0, m = int(b * FS), int(0.01 * FS)
        x[i0:i0 + m] += (0.6 + 0.3 * rng.random()) * np.exp(-np.arange(m) / (0.002 * FS))
    xs = np.stack([x, x * 0.9], axis=1)
    hits = envelope.hits_for(xs, FS, np.arange(0, 20.0, 0.5))
    v = bands.split(bands.to_ms(xs), FS, [100.0, 400.0, 1600.0, 6400.0])[2, :, 0]
    _, db = meter.rms_series(v, FS)
    loud = meter.rms_1s(v, FS)["loud_section_dbfs"]
    assert loud - np.median(db[db > meter.RMS_GATE_DBFS]) > 20.0      # the intro owns the median
    body = mb.body_level_db(v, FS, hits)
    assert 0.0 <= loud - body < 1.5
    sol = mb.threshold_for_peak_gr(v, FS, 2.5, 2.0, 10.0, 100.0, 6.0, 1000.0, hits)
    assert sol["bound"] == "level floor" and sol["threshold_db"] > -62.0
    assert sol["threshold_db"] == pytest.approx(body - mb.BRACKET_BELOW_LEVEL_DB, abs=1e-6)
    # without hits the body is the loudest-section 1 s RMS
    assert mb.body_level_db(v, FS) == pytest.approx(loud)
    assert mb.body_level_db(np.zeros(2 * FS), FS) == -np.inf


def test_design_budget_reports_an_unreachable_band_as_such():
    xs, hits = _sparse_top()
    p0 = mb.design(xs, FS, -20.0, ratio=1.5, crossovers_hz=[100.0, 400.0, 1600.0, 6400.0])
    p0.budget = mb.default_budget("peak")
    p0.attack_ms = [[10.0, 10.0]] * 5
    p0.release_ms = [[100.0, 100.0]] * 5
    p0.knee_db = 6.0
    p = mb.design_budget(xs, FS, p0, hits=hits)
    rep = planner.budget_report(Plan(fs=FS, multiband=p))
    rows = rep["stages"]["multiband"]["bands"]
    unreachable = [k for k, row in enumerate(rows) if row[0]["status"] == "target unreachable"]
    assert unreachable and 4 in unreachable and 0 not in unreachable
    for k in unreachable:
        r = rows[k][0]
        assert r["bound"] == "level floor" and r["threshold_floor_db"] is not None
        assert r["threshold_db"] == pytest.approx(r["threshold_floor_db"], abs=1e-6)
        assert r["threshold_floor_db"] < r["level_db"] - mb.BRACKET_BELOW_LEVEL_DB + 1.0
    kinds = {(w["band"], w["channel"]): w["kind"] for w in rep["warnings"]
             if w["stage"] == "multiband"}
    assert all(kinds.get((k, "M")) == "unreachable" for k in unreachable)
    assert "average" not in [kinds.get((k, "M")) for k in unreachable]
    w = next(w for w in rep["warnings"] if w.get("kind") == "unreachable")
    assert w["bound"] == "level floor" and w["achieved_db"] < w["target_db"]
    assert {"threshold_db", "floor_db", "level_db"} <= set(w)


def test_status_reads_target_unreachable_only_on_a_floor_bound():
    assert planner._status(2.1, 2.5, 3.0, 4.5, "level floor") == "target unreachable"
    assert planner._status(2.1, 2.5, 3.0, 4.5, "bracket floor") == "target unreachable"
    assert planner._status(2.1, 2.5, 3.0, 4.5, None) == "ok"
    assert planner._status(1.9, 2.5, 3.0, 4.5, None) == "under"
    assert planner._status(2.1, 2.5, 5.0, 4.5, "level floor") == "over"


# ------------------------------------------------------------ S3-2: low sample rates

def _design(fs, side=True):
    nodes = [float(v) for v in np.geomspace(30, 18000, 29) if v < 0.5 * fs]
    return {"nodes_hz": nodes, "residual_db": [0.0] * len(nodes), "measured_slope": -4.5,
            "target_slope": -4.5, "sub_pull": True, "sub_shelf_db": 0.0, "rotation": 0.0,
            "side_residual_db": [0.5] * len(nodes) if side else None,
            "noise": {"floor_db": None, "gain_db": 20.0}, "top_decoherence": 0.1}


@pytest.mark.parametrize("fs", [16000, 22050, 22222])
def test_input_eq_fit_with_a_side_spectrum_at_low_sample_rates(fs):
    """0.45 fs is at or below the air shelf's 10 kHz range bottom: the corner range no
    longer collapses and the side fit's bounds are never inverted (scipy raised)."""
    lo, hi = planner._corner_range(planner.EQ_AIR_FC_HZ, fs)
    assert lo < hi <= planner.EQ_HS_MAX_FS_SHARE * fs
    eq, rec = planner.fit_inputeq(_design(fs), fs)
    assert eq.air_shelf_side is not None and eq.frequencies()
    assert all(f < fs / 2 for f in eq.frequencies())
    Plan(fs=fs, inputeq=eq)                           # validates every frequency


# ------------------------------------------------------------ S3-1: short and silent inputs

def test_input_plan_accepts_an_unmeasurable_level():
    assert InputPlan(0.0, -18.0, None).measured_dbfs is None
    with pytest.raises(ValueError):
        InputPlan(0.0, -18.0, math.nan)


def test_analyse_records_none_for_an_unmeasurable_level():
    a = planner.analyse(np.zeros((2 * FS, 2)), FS)
    assert a["meta"]["input"] == {"gain_db": 0.0, "r_ref_dbfs": -18.0, "measured_dbfs": None}
    p = planner.plan_from_analysis(a, planner.Controls())
    assert p.input == InputPlan(0.0, -18.0, None)


def test_cli_refuses_a_short_and_a_silent_input(tmp_path, capsys):
    """`master` refuses an input shorter than 1 s, and a silent one, before any analysis,
    with one plain line (S3-1); the work dir is left as it was."""
    rng = np.random.default_rng(1)
    half = tmp_path / "half.wav"
    sf.write(half, 0.3 * rng.standard_normal((FS // 2, 2)), FS)
    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros((10 * FS, 2)), FS)
    out = tmp_path / "out"
    assert cli.main(["master", str(half), "-o", str(out)]) == 2
    err = capsys.readouterr().err
    assert "too short" in err and "0.50 s" in err and f"{cli.MIN_INPUT_S:g} s" in err
    assert not out.exists()
    assert cli.main(["master", str(silent), "-o", str(out)]) == 2
    err = capsys.readouterr().err
    assert "silent" in err and "nothing to master" in err
    assert not list(out.glob("*.npz"))                # no analysis cache of silence
    assert cli.main(["analyse", str(silent), "-o", str(out)]) == 2
    assert "silent" in capsys.readouterr().err
    with pytest.raises(ValueError, match="too short"):
        cli.check_audio(half, np.zeros((FS // 2, 2)), FS)


def test_engine_refuses_a_silent_input_with_one_error_line(tmp_path, capsys):
    from eluate import app_engine
    silent = tmp_path / "s.wav"
    sf.write(silent, np.zeros((3 * FS, 2)), FS)
    assert app_engine.main(["analyse", str(silent), "-o", str(tmp_path / "w")]) == 2
    msgs = [json.loads(l[len(app_engine.PREFIX):]) for l in capsys.readouterr().out.splitlines()
            if l.startswith(app_engine.PREFIX)]
    assert [m["event"] for m in msgs] == ["stage", "error"] and "silent" in msgs[-1]["msg"]
    assert not (tmp_path / "w" / cli.ANALYSIS_FILE).exists()


# ------------------------------------------------------------ hygiene F3, F4, F5, S4-9

def _fake_done(work: Path, inp: Path, run_id="abc"):
    """A finished run's work dir: every product present, run.json last."""
    work.mkdir(parents=True, exist_ok=True)
    assert set(cli.RUN_PRODUCTS) <= set(cli.CLEARED_AT_START)
    assert cli.CLEARED_AT_START[0] == "run.json"
    for name in cli.CLEARED_AT_START:
        (work / name).write_text("{}", encoding="utf-8")
    (work / "ab").mkdir(exist_ok=True)
    (work / "ab" / "difference.wav").write_bytes(b"")
    (work / "plan.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    (work / "run.json").write_text(json.dumps({"run_id": run_id, "controls": {}}),
                                   encoding="utf-8")
    (work / "source.json").write_text(json.dumps({"path": str(inp)}), encoding="utf-8")


def test_input_is_checked_before_the_previous_run_is_touched(tmp_path, monkeypatch):
    """Hygiene F4: a typo'd input path, or an input inside the output folder (F3), is
    refused before run.json and the process log are removed."""
    inp = tmp_path / "a.wav"
    sf.write(inp, 0.1 * np.random.default_rng(0).standard_normal((2 * FS, 2)), FS)
    work = tmp_path / "work"
    _fake_done(work, inp)
    with pytest.raises(ValueError, match="no such file"):
        cli.do_master(tmp_path / "typo.wav", work, planner.Controls(), log=lambda m: None)
    assert all((work / f).exists() for f in cli.CLEARED_AT_START + cli.RUN_PRODUCTS)
    # the previous master as the input of a new run in the same folder (`eluate master
    # out/master.wav -o out/` overwrote its own input and exited 0)
    sf.write(work / "master.wav", 0.1 * np.random.default_rng(1).standard_normal((2 * FS, 2)),
             FS)
    with pytest.raises(ValueError, match="a file this run writes or removes"):
        cli.do_master(work / "master.wav", work, planner.Controls(), log=lambda m: None)
    assert cli.main(["master", str(work / "master.wav"), "-o", str(work)]) == 2
    for rel in ("ab/difference.wav", "passes/x 1 - plain sum.wav", "pass-1/master.wav",
                "report.json", "features.npz"):
        assert cli.run_writes(work / rel, work), rel
    assert all((work / f).exists() for f in cli.CLEARED_AT_START)
    # other files in the folder are not the run's: the stems flow's own sums, a premaster
    sum_path = work / cli.STEMS_SUM
    sf.write(sum_path, 0.1 * np.random.default_rng(0).standard_normal((2 * FS, 2)), FS)
    assert cli.check_input(sum_path, work) == (2 * FS, FS)
    (work / "pass-1").mkdir()
    sf.write(work / "pass-1" / cli.CORRECTED_SUM, np.ones((2 * FS, 2)) * 0.1, FS)
    assert cli.check_input(work / "pass-1" / cli.CORRECTED_SUM, work / "pass-1") == (2 * FS, FS)
    with pytest.raises(ValueError, match="writes or removes"):
        cli.check_input(work / "pass-1" / cli.CORRECTED_SUM, work)
    sf.write(work / "premaster.wav", np.ones((2 * FS, 2)) * 0.1, FS)
    assert not cli.run_writes(work / "premaster.wav", work)
    # the stems flow: no stem may be the sum it writes, or a file a master there writes
    for stem in (work / cli.STEMS_SUM, work / "master.wav"):
        with pytest.raises(ValueError, match="writes or removes"):
            cli.do_master_stems([stem, inp], work, planner.Controls(), log=lambda m: None)
    assert (work / "master.wav").exists() and (work / "run.json").exists()
    assert cli.check_input(inp, work) == (2 * FS, FS)
    with pytest.raises(ValueError, match="too short"):
        cli.check_input(_short(tmp_path), work)


def _short(d: Path) -> Path:
    p = d / "short.wav"
    sf.write(p, np.zeros((FS // 4, 2)), FS)
    return p


def test_locked_output_is_refused_before_anything_is_removed(tmp_path):
    """Hygiene F5: a read-only (or open elsewhere) master.wav fails fast, before the
    solve and before the previous run is cleared."""
    inp = tmp_path / "a.wav"
    sf.write(inp, 0.1 * np.random.default_rng(0).standard_normal((2 * FS, 2)), FS)
    work = tmp_path / "work"
    _fake_done(work, inp)
    target = work / "master.wav"
    os.chmod(target, stat.S_IREAD)
    try:
        with pytest.raises(ValueError, match="cannot replace"):
            cli.probe_output(work)
        with pytest.raises(ValueError, match="cannot replace"):
            cli.do_master(inp, work, planner.Controls(), log=lambda m: None)
        assert all((work / f).exists() for f in cli.CLEARED_AT_START)
    finally:
        os.chmod(target, stat.S_IREAD | stat.S_IWRITE)
    cli.probe_output(work)                            # writable again: fine
    assert not list(work.glob(".eluate-probe-*"))


def test_a_run_clears_every_previous_product_and_report_checks_the_run(tmp_path, monkeypatch):
    """S4-9: once the input and output are checked, every product of the previous run goes
    (the analysis cache stays), so a failed run never leaves the old master next to the new
    plan; `report` refuses a work dir without run.json or with mismatched run ids."""
    inp = tmp_path / "a.wav"
    sf.write(inp, 0.1 * np.random.default_rng(0).standard_normal((2 * FS, 2)), FS)
    work = tmp_path / "work"
    _fake_done(work, inp)
    for name in ("features.npz", "analysis.json"):
        (work / name).write_text("x", encoding="utf-8")
    (work / "pass-1").mkdir()
    (work / "passes").mkdir()

    def stopped(*a, **k):
        raise RuntimeError("stopped")
    monkeypatch.setattr(cli, "_reuse_analysis", stopped)
    with pytest.raises(RuntimeError):
        cli.do_master(inp, work, planner.Controls(), log=lambda m: None)
    assert not any((work / f).exists() for f in cli.CLEARED_AT_START + cli.RUN_PRODUCTS)
    assert not (work / "ab").exists() and not (work / "pass-1").exists()
    assert not (work / "passes").exists()
    assert all((work / f).exists() for f in ("features.npz", "analysis.json", "source.json"))
    with pytest.raises(ValueError, match="no run.json"):
        cli.do_report(work, log=lambda m: None)
    # mismatched run ids: an unfinished re-render
    _fake_done(work, inp)
    (work / "plan.json").write_text(json.dumps({"run_id": "other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="different runs"):
        cli.check_run_files(work)
    (work / "plan.json").write_text(json.dumps({"run_id": "abc"}), encoding="utf-8")
    cli.check_run_files(work)


def test_report_refuses_a_changed_premaster(tmp_path):
    """Hygiene F1: `report` compares the premaster's SHA-256 with source.json's."""
    inp = tmp_path / "a.wav"
    sf.write(inp, 0.1 * np.random.default_rng(0).standard_normal((2 * FS, 2)), FS)
    work = tmp_path / "work"
    _fake_done(work, inp)
    (work / "source.json").write_text(json.dumps({"path": str(inp), "sha256": "0" * 64}),
                                      encoding="utf-8")
    with pytest.raises(ValueError, match="changed since the master was made"):
        cli.check_source(work)
    with pytest.raises(ValueError, match="changed since"):
        cli.do_report(work, log=lambda m: None)
    (work / "source.json").write_text(json.dumps({"path": str(inp),
                                                  "sha256": cli.file_sha256(inp)}),
                                      encoding="utf-8")
    assert cli.check_source(work)["path"] == str(inp)
    (work / "source.json").write_text(json.dumps({"path": str(tmp_path / "gone.wav")}),
                                      encoding="utf-8")
    with pytest.raises(ValueError, match="not there"):
        cli.check_source(work)


# ------------------------------------------------------------ F18: --subside

def test_subside_request_is_recorded_with_the_clip(pana):
    _, a = pana
    sc = planner.sidestrip_choice(a, planner.Controls(subside_hz=20.0))
    assert sc == {"requested_hz": 20.0, "rule_hz": 20.0, "used_hz": 60.0, "clipped": True,
                  "rule": "--subside", "range_hz": [60.0, 150.0]}
    assert planner.subside_hz(a, planner.Controls(subside_hz=20.0)) == 60.0
    sc = planner.sidestrip_choice(a, planner.Controls(subside_hz=90.0))
    assert sc["used_hz"] == 90.0 and not sc["clipped"]
    # live (audit 2026-11-10 A10): 60 Hz only on a decoherent sub; this fixture's is mono
    sc = planner.sidestrip_choice(a, planner.Controls(profile="live"))
    assert sc["rule"].startswith("live profile: off") and sc["used_hz"] is None
    b = {**a, "plan_inputs": {**a["plan_inputs"], "band_decoherence":
                              [0.5] + list(a["plan_inputs"]["band_decoherence"][1:])}}
    sc = planner.sidestrip_choice(b, planner.Controls(profile="live"))
    assert sc["rule"].startswith("live profile: sub band decoherence") and sc["used_hz"] == 60.0
    rec = planner.plan_record(a, planner.Controls(subside_hz=20.0))
    assert rec["sidestrip"]["clipped"] and rec["controls"]["subside_hz"] == 20.0
    lines = []
    assert cli._plan_notes(rec, lines.append) == []
    assert lines == ["side strip: --subside asked for 20 Hz, clipped to 60 Hz (the 60 to 150 Hz "
                     "range)"]


# ------------------------------------------------------------ F4: the character boundary

@pytest.mark.parametrize("bpm,name,note", [
    (69.0, "slow", "< 70"), (69.3, "moderate", "within 1 % of the 70 BPM boundary"),
    (69.9, "moderate", "within 1 %"), (70.0, "moderate", "in 70 to 140"),
    (140.0, "moderate", "in 70 to 140"), (140.09, "moderate", "within 1 % of the 140 BPM boundary"),
    (141.4, "moderate", "within 1 %"), (141.5, "fast", "> 140")])
def test_auto_character_boundary_tolerance(bpm, name, note):
    """A true 140 BPM track read 140.09 and flipped to fast: +-1 % at both boundaries
    resolves to moderate, and the reason quotes the tempo and the rule."""
    p, why = planner.character_choice("auto", bpm)
    assert p == name and note in why and f"{bpm:g} BPM" in why
    p2, why2 = planner.character_choice("auto", bpm, "analysis")
    assert p2 == ("moderate" if name == "slow" else name) and "detected" in why2


# ------------------------------------------------------------ F21, S2-4b / S4-7: tempo

def _tempo_analysis(user_bpm=None, detected=128.0, beats=None):
    pi = {"tempo_bpm": user_bpm if user_bpm is not None else detected,
          "tempo_octave": "user" if user_bpm is not None else "kept",
          "grid_informative": True, "beat_times": beats if beats is not None else
          list(np.arange(0.0, 8.0, 60.0 / (user_bpm or detected))),
          "metrical": {"rule": "user" if user_bpm is not None else "pulse",
                       "octave": {"decision": "kept"}}}
    if user_bpm is not None:
        rel = {2.0: "x2", 0.5: "/2"}.get(round(detected / user_bpm, 6))
        pi["metrical"].update(user_bpm=user_bpm, detected_bpm=detected,
                              deviation_pct=100.0 * (detected / user_bpm - 1.0),
                              detected_relation=rel)
    return {"plan_inputs": pi, "meta": {"crossovers": [100.0, 400.0, 1600.0, 6400.0],
                                        "duration_s": 8.0}}


def test_a_tempo_given_at_analyse_stays_the_users_at_plan():
    """Features F21: `eluate analyse --bpm 65` then `eluate plan` (no --bpm) read the tempo
    as the user's, so the character resolves to slow and the hits say "user bpm"."""
    a = _tempo_analysis(user_bpm=65.0, detected=130.0)
    assert planner.plan_tempo(a, planner.Controls()) == (65.0, "user")
    c = planner.character_settings(a, planner.Controls(), nb=5)
    assert (c["bpm"], c["bpm_source"], c["preset"]) == (65.0, "user", "slow")
    assert c["tempo_octave"] == "user" and c["kick_detective"]
    assert planner.plan_hits(a, planner.Controls()).source == "user bpm"
    # a tempo given now still wins, and a detected one is still "analysis"
    assert planner.plan_tempo(a, planner.Controls(bpm=130.0)) == (130.0, "user")
    b = _tempo_analysis(detected=65.0)
    assert planner.plan_tempo(b, planner.Controls()) == (65.0, "analysis")
    assert planner.character_settings(b, planner.Controls(), nb=5)["preset"] == "moderate"


def test_a_bpm_given_at_analyse_is_the_users(tmp_path):
    """Features F21 end to end: `eluate analyse --bpm 65` records the tempo as the user's
    (source.json bpm_source, the analysis's metrical rule) and `plan` (no --bpm) reads it
    so: slow, the user's grid, and the tempo record says "user"."""
    x, _ = short_track(FS, 1)
    wav = tmp_path / "in.wav"
    sf.write(wav, x[: 6 * FS], FS)
    work = tmp_path / "w"
    cli.do_analyse(wav, work, None, log=lambda m: None, bpm=65.0)
    src = json.loads((work / "source.json").read_text(encoding="utf-8"))
    assert src["bpm"] == 65.0 and src["bpm_source"] == "user"
    a = cli._load_analysis(work)
    assert planner.plan_tempo(a, planner.Controls()) == (65.0, "user")
    rec = planner.plan_record(a, planner.Controls())
    assert rec["character"]["bpm_source"] == "user" and rec["character"]["preset"] == "slow"
    assert rec["tempo"]["source"] == "user" and rec["tempo"]["detected_bpm"] is not None
    cli.do_analyse(wav, work, None, log=lambda m: None)
    assert json.loads((work / "source.json").read_text(encoding="utf-8"))["bpm_source"] is None


def test_tempo_record_flags_a_user_tempo_that_disagrees_with_the_grid():
    rec = planner.tempo_record(_tempo_analysis(user_bpm=256.0, detected=128.0),
                               planner.Controls())
    assert rec["source"] == "user" and rec["detected_bpm"] == 128.0
    assert rec["deviation_pct"] == pytest.approx(-50.0) and rec["relation"] == "/2"
    (w,) = rec["warnings"]
    assert w["stage"] == "tempo" and w["kind"] == "disagreement"
    assert w["ratio"] == pytest.approx(2.0)
    assert w["text"].startswith("tempo given 256 BPM; detected 128.0 BPM, ratio 2.00 (/2)")
    assert "--bpm" in w["text"]
    # within the tolerance: recorded, no warning
    rec = planner.tempo_record(_tempo_analysis(user_bpm=128.0, detected=129.0),
                               planner.Controls())
    assert rec["detected_bpm"] == 129.0 and rec["warnings"] == []
    # no user tempo: nothing to compare
    rec = planner.tempo_record(_tempo_analysis(detected=128.0), planner.Controls())
    assert rec["source"] == "analysis" and rec["detected_bpm"] is None and not rec["warnings"]
    # a tempo given now against an analysis made without one: the analysis's own tempo
    rec = planner.tempo_record(_tempo_analysis(detected=128.0), planner.Controls(bpm=97.0))
    assert rec["detected_bpm"] == 128.0 and rec["relation"] is None
    assert rec["warnings"][0]["ratio"] == pytest.approx(97.0 / 128.0)
    lines = []
    assert cli._plan_notes({"tempo": rec}, lines.append) == rec["warnings"]
    assert lines == ["warning: " + rec["warnings"][0]["text"]]


def test_plan_record_carries_tempo_and_the_warning(pana):
    _, a = pana
    rec = planner.plan_record(a, planner.Controls(bpm=256.0))
    assert rec["tempo"]["source"] == "user" and rec["tempo"]["bpm"] == 256.0
    assert rec["tempo"]["detected_bpm"] == pytest.approx(a["plan_inputs"]["tempo_bpm"])
    assert rec["tempo"]["warnings"][0]["ratio"] == pytest.approx(
        256.0 / a["plan_inputs"]["tempo_bpm"], abs=1e-9)
    assert planner.plan_record(a, planner.Controls())["tempo"]["warnings"] == []


def test_process_log_states_the_tempo_disagreement():
    from eluate import processlog
    d = processlog.Doc()
    rep = {"planner": {"controls": {"tilt_db": 0.0}, "character": {"bpm": 256.0,
                       "bpm_source": "user", "preset": "fast", "reason": "auto"},
                       "profile": {"name": "electronic", "calibrated": True},
                       "tempo": {"bpm": 256.0, "source": "user", "detected_bpm": 128.0,
                                 "deviation_pct": -50.0, "relation": "/2", "tolerance_pct": 3.0}},
           "loudness": {"target_rms_dbfs": -4.1}, "targets": {"slope_db_oct": -4.5},
           "warnings": [{"stage": "tempo", "kind": "disagreement", "text": "tempo given 256 BPM; "
                         "detected 128.0 BPM, ratio 2.00"}]}
    processlog._settings(d, rep, {"path": "x.wav"}, None)
    processlog._flags(d, rep)
    md = d.md()
    assert "Tempo: 256.0 BPM (user); detected 128.0 BPM, ratio 2.00 (/2)" in md
    assert "check --bpm" in md and "tempo given 256 BPM; detected 128.0 BPM, ratio 2.00" in md


# ------------------------------------------------------------ S4-8 / features F2: key gate

def test_key_gate_and_confidence():
    assert planner.analysis_key_confidence({"plan_inputs": {}, "meta": {}}) is None
    assert planner.analysis_key_confidence({"plan_inputs": {"key_confidence": 0.63}}) == 0.63
    assert planner.analysis_key_confidence({"plan_inputs": {"key_confidence": math.nan}}) is None
    assert planner.analysis_key_confidence({"plan_inputs": {}, "meta": {"key_confidence": 1.4}}) == 1.0
    # the record's raw profile score is not a confidence: absent means unknown
    assert planner.analysis_key_confidence(
        {"plan_inputs": {"key": {"tonic": "A", "score": 0.9}}}) is None
    g = planner.key_gate("C", 0.29)
    assert not g["passed"] and g["min"] == planner.KEY_CONFIDENCE_MIN == 0.5
    assert g["reason"] == "key C confidence 0.29 < 0.5"
    assert planner.key_gate("A", 0.63)["passed"]
    assert planner.key_gate("A", None)["reason"] == "key confidence unknown"
    assert planner.key_gate(None, 0.9)["reason"] == "no key found"


@pytest.mark.parametrize("conf", [0.29, None])
def test_an_unconfident_or_unknown_key_leaves_the_default_crossovers(monkeypatch, conf):
    """Pink noise read Cm and moved every band split; below the gate, or with no confidence
    at all (an analysis without key_confidence), the analysis keeps
    bands.DEFAULT_CROSSOVERS and records why (features audit F2)."""
    from eluate.features import pitch
    real = pitch.key

    def c_minor(ctx, frame_power=None):
        out = real(ctx, frame_power)
        out["key"] = {**out["key"], "tonic": "C", "label": "Cm"}
        if conf is None:               # pitch.key now gives a confidence (F2): drop it
            out["key"]["confidence"] = None
        return out
    monkeypatch.setattr(pitch, "key", c_minor)
    if conf is not None:
        features_with_key_confidence(monkeypatch, conf)
    x, _ = short_track(FS, 1)
    a = planner.analyse(x[: 2 * FS], FS)
    assert planner.key_crossovers("C") != bands.DEFAULT_CROSSOVERS
    assert tuple(a["meta"]["crossovers"]) == bands.DEFAULT_CROSSOVERS
    gate = a["meta"]["crossover_gate"]
    assert gate["passed"] is False and gate["applied"] and gate["confidence"] == conf
    assert a["meta"]["crossover_rule"] == "key" and a["meta"]["crossover_tonic"] == "C"
    rec = planner.crossover_choice(a, planner.Controls())
    want = "key C confidence 0.29 < 0.5" if conf else "key confidence unknown"
    assert rec["key_gate"] == gate and rec["reason"].startswith(f"key not locked ({want})")
    assert rec["hz"] == list(bands.DEFAULT_CROSSOVERS)


def test_a_confident_key_still_locks_the_crossovers(monkeypatch):
    from eluate.features import pitch
    real = pitch.key

    def c_sharp(ctx, frame_power=None):
        out = real(ctx, frame_power)
        out["key"] = {**out["key"], "tonic": "C#", "label": "C#m"}
        return out
    monkeypatch.setattr(pitch, "key", c_sharp)
    features_with_key_confidence(monkeypatch, 0.8)
    x, _ = short_track(FS, 1)
    a = planner.analyse(x[: 2 * FS], FS)
    gate = a["meta"]["crossover_gate"]
    assert gate["passed"] and gate["confidence"] == 0.8
    assert tuple(a["meta"]["crossovers"]) == planner.key_crossovers("C#") !=         bands.DEFAULT_CROSSOVERS
    rec = planner.crossover_choice(a, planner.Controls())
    assert rec["key_gate"]["passed"] and rec["reason"].startswith("key-locked")
    # an analysis from before the gate reads it for the record only (applied False)
    b = copy.deepcopy(a)
    del b["meta"]["crossover_gate"]
    b["plan_inputs"]["key_confidence"] = 0.1
    rec = planner.crossover_choice(b, planner.Controls())
    assert rec["key_gate"]["applied"] is False and rec["key_gate"]["confidence"] == 0.1
    assert rec["reason"].startswith("key-locked")


# ------------------------------------------------------------ features F7: side-only notches

def test_s_only_notch_is_rendered_on_the_side(pana):
    _, a = pana
    b = copy.deepcopy(a)
    for r in b["plan_inputs"]["notches"]:
        r["channel"] = "S"
    p, rows, merged = planner.plan_notches(b)
    assert p.notches and all(n.channel == "S" for n in p.notches)
    assert all(r["rendered"] for r in rows if r["reason"] == "")
    assert all(m["channel"] == "S" for m in merged)
    assert planner.notch_rendered("S") and planner.notch_channel("S") == "S"
    assert planner.notch_channel("M") == planner.notch_channel("both") == "both"
    # the plan round-trips the channel; a plan saved without one loads as "both"
    d = Plan(fs=FS, notch=p).to_dict()
    assert d["notch"]["notches"][0]["channel"] == "S"
    assert Plan.from_dict(d).notch.notches[0].channel == "S"
    del d["notch"]["notches"][0]["channel"]
    assert Plan.from_dict(d).notch.notches[0].channel == "both"


def test_s_notch_inside_a_stereo_notch_band_is_left_out(pana):
    _, a = pana
    b = copy.deepcopy(a)
    n0 = b["plan_inputs"]["notches"][0]
    both = dict(n0, channel="both")
    side = dict(n0, channel="S", f0=n0["f0"] * 1.01)
    far = dict(n0, channel="S", f0=n0["f0"] * 1.5)
    b["plan_inputs"]["notches"] = [both, side, far]
    p, rows, merged = planner.plan_notches(b)
    assert rows[0]["rendered"] and not rows[1]["rendered"] and rows[2]["rendered"]
    assert "inside a stereo notch's band" in rows[1]["reason"]
    assert sorted(n.channel for n in p.notches) == ["S", "both"]


def test_notch_stage_cuts_the_side_alone():
    """A -6 dB side notch at 1 kHz: a 1 kHz tone on S falls to its notch gain, M (a
    500 Hz tone) is untouched to rounding, and a zero-depth side notch is a bit-exact
    identity."""
    n = 2 * FS
    t = np.arange(n) / FS
    ms = np.stack([0.3 * np.sin(2 * np.pi * 500.0 * t), 0.2 * np.sin(2 * np.pi * 1000.0 * t)],
                  axis=1)
    x = bands.to_lr(ms)
    y = notch_stage.apply(x, FS, NotchPlan([Notch(1000.0, 8.0, Envelope.constant(-6.0), "S")]))
    out = bands.to_ms(y)
    mid = slice(FS // 2, -FS // 2)
    assert np.max(np.abs(out[mid, 0] - ms[mid, 0])) < 1e-9
    gain = np.max(np.abs(out[mid, 1])) / np.max(np.abs(ms[mid, 1]))
    assert gain == pytest.approx(10 ** (-6 / 20), abs=0.01)
    # the same notch on "both" cuts M and S alike at their own frequencies: the 1 kHz S
    # tone falls as much, the 500 Hz M tone barely
    yb = notch_stage.apply(x, FS, NotchPlan([Notch(1000.0, 8.0, Envelope.constant(-6.0))]))
    ob = bands.to_ms(yb)
    assert np.max(np.abs(ob[mid, 1])) / np.max(np.abs(ms[mid, 1])) == pytest.approx(gain, abs=0.01)
    z = notch_stage.apply(x, FS, NotchPlan([Notch(1000.0, 8.0, Envelope.constant(0.0), "S")]))
    assert np.array_equal(z, x)
    # the kick mask holds the side notch too: around an onset S passes uncut
    km = notch_stage.kick_mask([1.0], n / FS)
    yk = notch_stage.apply(x, FS, NotchPlan([Notch(1000.0, 8.0, Envelope.constant(-6.0), "S")],
                                            km))
    ok = bands.to_ms(yk)
    hold = slice(FS - int(0.004 * FS), FS + int(0.004 * FS))
    assert np.max(np.abs(ok[hold, 1] - ms[hold, 1])) < 0.02 * np.max(np.abs(ms[:, 1]))
    assert np.max(np.abs(ok[hold, 0] - ms[hold, 0])) < 1e-9
    assert np.max(np.abs(ok[mid.start:FS // 2 + FS // 4, 1])) == pytest.approx(
        np.max(np.abs(out[mid.start:FS // 2 + FS // 4, 1])), abs=1e-6)
    # the cap holds on S: a deeper request is clipped to -6 dB
    y12 = notch_stage.apply(x, FS, NotchPlan([Notch(1000.0, 8.0, Envelope.constant(-12.0),
                                                    "S")]))
    assert np.max(np.abs(y12 - y)) < 1e-12
    # overlapping notches must merge within a channel, and never across channels
    pair = [Notch(1000.0, 6.0, Envelope.constant(-6.0), "S"),
            Notch(1024.0, 6.0, Envelope.constant(-6.0), "S")]
    with pytest.raises(ValueError, match="overlap"):
        notch_stage.apply(x, FS, NotchPlan(pair))
    m = notch_stage.merge_candidates(pair)
    assert len(m) == 1 and m[0].channel == "S"
    mixed = [Notch(1000.0, 6.0, Envelope.constant(-6.0), "S"),
             Notch(1024.0, 6.0, Envelope.constant(-6.0), "both")]
    assert len(notch_stage.merge_candidates(mixed)) == 2
    notch_stage.apply(x, FS, NotchPlan(mixed))        # separate sets: accepted
    with pytest.raises(ValueError, match="channel"):
        Notch(1000.0, 8.0, Envelope.constant(-1.0), "M")


# ------------------------------------------------------------ the notch envelope (audio-rate)

def test_notch_envelope_is_smooth_not_stepped():
    """0.5 dB steps at 1 ms raised the notch's modulation sidebands from -64 to -47 dB; the
    envelope keeps the 200 ms smoothing and no quantisation."""
    hop, n = 0.1, 6 * FS
    ex = np.concatenate([np.zeros(10), np.linspace(6.0, 18.0, 20), np.full(20, 18.0),
                         np.zeros(10)])
    g = planner._notch_envelope(ex, hop, 1.0, n, FS)
    assert g.min() == pytest.approx(-6.0, abs=1e-9) and g.max() == 0.0
    steps = np.abs(np.diff(g))
    assert steps.max() < 0.1                          # no 0.5 dB step anywhere
    assert np.unique(np.round(g, 6)).size > 200        # a ramp, not a staircase
    # the smoothed ramp changes by less than 0.05 dB per ms where it moves
    moving = steps > 0
    assert moving.any() and steps[moving].max() < 0.05
    assert not hasattr(planner, "NOTCH_QUANT_DB")


def test_composite_clamp_bound_holds_with_smooth_envelopes():
    """The clamp now rounds its own bound rows: with two smooth envelopes it still holds
    the summed cut at the cap and runs in a bounded number of rows."""
    n = 2 * FS
    nc = planner.control_length(n, FS)
    ramp = -6.0 * np.clip(np.linspace(-0.5, 1.5, nc), 0.0, 1.0)
    p = NotchPlan([Notch(1000.0, 3.0, Envelope(ramp)), Notch(1450.0, 3.0, Envelope(ramp))])
    x = np.zeros((n, 1))
    x[n // 2] = 1.0
    h = notch_stage.apply(x, FS, p)[:, 0]
    H = np.fft.rfft(np.roll(h, -n // 2))
    # at the impulse both notches sit at the cap: the summed cut stays within 6 dB
    assert 20 * np.log10(np.abs(H)).min() >= -6.0 - 1e-3
    d = 1.0 - 10.0 ** (ramp / 20.0)
    s = notch_stage.composite_scale(np.stack([d, d], axis=1), None,
                                    [notch_stage.notch_sos(nt, FS) for nt in p.notches],
                                    [1000.0, 1450.0], FS)
    assert s.min() < 1.0 and s.max() == 1.0 and np.all(np.diff(s[nc // 4: 3 * nc // 4]) <= 1e-12)
    # the rounded rows leave no staircase in s: under 0.1 dB per 1 ms control step (it
    # was 0.66 dB on overlapping ramps before the erosion and average)
    t = np.linspace(0.0, 1.0, nc)
    r1 = -6.0 * np.clip(1.3 * np.sin(np.pi * t), 0.0, 1.0)
    r2 = -6.0 * np.clip(1.2 * np.sin(0.8 * np.pi * t), 0.0, 1.0)
    nts = [Notch(1000.0, 3.0, Envelope(r1)), Notch(1450.0, 3.0, Envelope(r2))]
    d2 = np.stack([1.0 - 10.0 ** (r / 20.0) for r in (r1, r2)], axis=1)
    s2 = notch_stage.composite_scale(d2, None, [notch_stage.notch_sos(nt, FS) for nt in nts],
                                     [1000.0, 1450.0], FS)
    assert s2.min() < 0.9 and np.max(np.abs(np.diff(20 * np.log10(s2)))) < 0.1
    h2 = notch_stage.apply(x, FS, NotchPlan(nts))[:, 0]
    for c in (nc // 3, nc // 2, 2 * nc // 3):              # the cap holds where both cut
        xi = np.zeros((n, 1))
        xi[c * FS // 1000] = 1.0
        hi = notch_stage.apply(xi, FS, NotchPlan(nts))[:, 0]
        Hi = np.fft.rfft(np.roll(hi, -(c * FS // 1000)))
        assert 20 * np.log10(np.abs(Hi)).min() >= -6.0 - 1e-3
    assert np.all(np.isfinite(h2))


# ------------------------------------------------------------ S2-4, S4-6: budget_report

def _limiter_plan(split, full=None):
    clip = {"target_db": 1.0, "peak_gr_db": 0.9, "outlier_gr_db": 1.4,
            "budget_db": 3.0, "outlier_cap_db": 3.0, "curve_gr_db": 1.2, "threshold_db": -6.0,
            "input_peak_8x_dbfs": -9.6, "split_hz": split, "warnings": []}
    if full is not None:
        clip["input_peak_8x_full_dbfs"] = full
    lp = LimiterPlan(clip_threshold_db=-6.0, clip_target_db=1.0, max_gr_db=3.0,
                     input_gain_db=10.0, clip_split_hz=split,
                     budget={"clipper": clip,
                             "limiter": {"planned_gr_db": 2.0, "planned_outlier_gr_db": 3.0,
                                         "budget_db": 3.0, "outlier_cap_db": 5.0,
                                         "input_gain_db": 10.0, "requested_gain_db": 12.0,
                                         "clipped_peak_dbfs": -10.5, "warnings": []}})
    return Plan(fs=FS, limiter=lp)


def test_clipper_rendered_fall_compares_like_with_like():
    """S2-4: the rendered track-peak fall reads the full signal's 8x input peak against the
    render's (full-signal) detector peak, with or without clip_split_hz; the solve's input
    peak above the split is never mixed with the full detector peak."""
    info = {"detector_peak_dbfs": -0.8, "max_gain_reduction_db": 2.3, "relimit_passes": [],
            "clip_max_reduction_db": 1.25}
    cs = planner.budget_report(_limiter_plan(None), info)["stages"]["clipper"]
    assert cs["rendered_db"] == pytest.approx(-9.6 + 10.0 + 0.8)
    assert cs["planned_track_db"] == pytest.approx(-9.6 + 10.5)
    assert cs["rendered_curve_db"] == 1.25 and cs["rendered_on"] == "track peak, full signal"
    # with a split: the full-signal peak (-8.4), not the part above the split (-9.6)
    cs = planner.budget_report(_limiter_plan(87.3, full=-8.4), info)["stages"]["clipper"]
    assert cs["rendered_db"] == pytest.approx(-8.4 + 10.0 + 0.8)
    assert cs["planned_track_db"] == pytest.approx(-8.4 + 10.5)
    # an old plan with a split and no full-signal peak: not measured, said so
    cs = planner.budget_report(_limiter_plan(87.3), info)["stages"]["clipper"]
    assert cs["rendered_db"] is None and "above 87.3 Hz" in cs["rendered_note"]
    assert "planned_track_db" not in cs
    assert cs["planned_db"] == 0.9 and cs["status"] == "ok"
    assert "rendered_db" not in planner.budget_report(_limiter_plan(87.3, -8.4))["stages"]["clipper"]


def test_limiter_row_carries_the_relimit():
    """S4-6: the base-rate re-limit's passes, largest reduction and share of samples, read
    from lim_info["relimit"] when the limiter reports it, else from relimit_passes."""
    info = {"detector_peak_dbfs": -0.8, "max_gain_reduction_db": 2.3, "trim_db": 0.0,
            "samples": 1_000_000,
            "relimit_passes": [{"samples": 980_000, "max_reduction_db": 0.29},
                               {"samples": 1200, "max_reduction_db": 0.02}]}
    st = planner.budget_report(_limiter_plan(None), info)["stages"]["limiter"]
    assert st["rendered_db"] == 2.3 and st["outlier_db"] == 2.3
    assert st["relimit"] == {"passes": 2, "max_db": 0.29, "samples": 980_000, "fraction": 0.98,
                             "trim_db": 0.0}
    assert st["rendered_total_max_db"] == pytest.approx(2.59)
    # the limiter's own record wins (a count of passes, or a list)
    info2 = dict(info, relimit={"passes": 1, "max_db": 0.4, "fraction": 0.5})
    st = planner.budget_report(_limiter_plan(None), info2)["stages"]["limiter"]
    assert st["relimit"]["passes"] == 1 and st["relimit"]["max_db"] == 0.4
    assert st["relimit"]["fraction"] == 0.5 and st["rendered_total_max_db"] == pytest.approx(2.7)
    info3 = {"max_gain_reduction_db": 1.0, "samples": 1000,
             "relimit": {"passes": [{"samples": 250, "max_reduction_db": 0.1}]}}
    st = planner.budget_report(_limiter_plan(None), info3)["stages"]["limiter"]
    assert st["relimit"] == {"passes": 1, "max_db": 0.1, "samples": 250, "fraction": 0.25,
                             "trim_db": None}
    # None-safe: no re-limit figures in the info, or relimit None
    for i in ({"max_gain_reduction_db": 1.0}, {"max_gain_reduction_db": 1.0, "relimit": None}):
        st = planner.budget_report(_limiter_plan(None), i)["stages"]["limiter"]
        assert st["relimit"] is None and "rendered_total_max_db" not in st
    assert "relimit" not in planner.budget_report(_limiter_plan(None))["stages"]["limiter"]


@pytest.mark.slow
@pytest.mark.parametrize("split", [True, False])
def test_clipper_planned_and_rendered_track_fall_agree(pana, split):
    """On the synthetic loop, with the planner's clip_split_hz and with the clipper on the
    whole signal, the render's track-peak fall agrees with the solve's within 0.2 dB, and
    the render reports its sample count for the re-limit share."""
    x, a = pana
    p = planner.plan_from_analysis(a, planner.Controls())
    if not split:
        p = replace(p, limiter=replace(p.limiter, clip_split_hz=None))
    assert (p.limiter.clip_split_hz is not None) == split
    p = planner.realise(p, x, FS)
    y, info, _ = solver.render_plan(x, FS, p)
    assert info["samples"] == x.shape[0]
    cs = planner.budget_report(p, info)["stages"]["clipper"]
    assert cs["rendered_db"] is not None and cs["rendered_db"] >= -0.05
    assert abs(cs["rendered_db"] - cs["planned_track_db"]) < 0.2
    assert abs(cs["rendered_curve_db"] - cs["curve_db"]) < 0.2
    assert cs["rendered_db"] <= cs["curve_db"] + 0.2
    st = planner.budget_report(p, info)["stages"]["limiter"]
    assert st["relimit"]["passes"] == len(info["relimit_passes"])
    assert 0.0 <= (st["relimit"]["fraction"] or 0.0) <= 1.0


# ------------------------------------------------------------ S4-5: the EQ fit's inputs

def test_input_eq_fit_inputs_are_quantised(pana):
    """Two float32 roundings of one premaster differ by about 1e-6 dB: the fit's inputs
    are rounded to 0.01 dB (slope 0.001 dB/oct), so the design and the EQ are identical."""
    _, a = pana
    b = copy.deepcopy(a)
    rng = np.random.default_rng(5)
    for k in ("ltas_stereo_db", "ltas_side_db"):
        v = np.asarray(b["plan_inputs"][k], dtype=np.float64)
        b["plan_inputs"][k] = (v + 1e-6 * rng.standard_normal(v.size)).tolist()
    b["plan_inputs"]["tilt_slope_db_oct"] = dict(b["plan_inputs"]["tilt_slope_db_oct"])
    b["plan_inputs"]["tilt_slope_db_oct"]["stereo"] += 1e-6
    b["plan_inputs"]["sub_shelf_db"] = dict(b["plan_inputs"]["sub_shelf_db"])
    b["plan_inputs"]["sub_shelf_db"]["stereo"] -= 1e-6
    da, db = planner.inputeq_design(a, planner.Controls()), planner.inputeq_design(b, planner.Controls())
    assert da == db
    for v in da["residual_db"]:
        assert v == pytest.approx(round(v / planner.EQ_INPUT_QUANT_DB) * planner.EQ_INPUT_QUANT_DB,
                                  abs=1e-12)
    ea, _ = planner.plan_inputeq(a, planner.Controls(), FS)
    eb, _ = planner.plan_inputeq(b, planner.Controls(), FS)
    assert planner.inputeq_sections(ea) == planner.inputeq_sections(eb)
    assert planner._quant(1.2345, 0.01) == 1.23 and planner._quant(-0.004, 0.01) == 0.0
    assert math.isnan(planner._quant(math.nan, 0.01))
