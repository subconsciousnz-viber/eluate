"""eluate CLI (spec 4)."""

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fixtures.e_signals import short_track
from eluate import cli, io, solver
from eluate.plan import Plan

FS = 44100
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def wav(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    x, _ = short_track(FS)
    p = d / "in.wav"
    io.write(p, x, FS)
    return p


def test_entry_point_declared():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert meta["project"]["scripts"]["eluate"] == "eluate.cli:main"


def test_parser_controls():
    a = cli.build_parser().parse_args(
        ["master", "in.wav", "-o", "out", "--tilt", "-1", "--target-rms", "-5",
         "--preserve", "--xover", "100,500,2000,8000", "--subside", "90"])
    c = cli._controls(a)
    assert (c.tilt_db, c.target_rms, c.preserve_mix, c.subside_hz) == (-1.0, -5.0, True, 90.0)
    assert c.rms_target == -5.0
    with pytest.raises(SystemExit):              # genre presets retired (maintainer, 2026-10-20)
        cli.build_parser().parse_args(["master", "in.wav", "-o", "out", "--genre", "techno"])
    k = cli._controls(cli.build_parser().parse_args(
        ["master", "in.wav", "-o", "out", "--character", "slow", "--makeup", "auto",
         "--kick-detective", "--glue-mode", "parallel", "--streaming"]))
    assert (k.character, k.makeup, k.kick_detective, k.glue_mode, k.streaming) == \
        ("slow", "auto", True, "parallel", True)
    assert cli._controls(cli.build_parser().parse_args(
        ["master", "in.wav", "-o", "out", "--no-kick-detective"])).kick_detective is False
    for gone in (["--target", "-7"], ["--harmonic", "0.4"]):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["master", "in.wav", "-o", "out", *gone])
    assert c.crossovers_hz == (100.0, 500.0, 2000.0, 8000.0)
    assert cli.build_parser().parse_args(["analyse", "a.wav", "-o", "w", "--bpm", "94"]).bpm == 94.0
    b = cli.build_parser().parse_args(
        ["master", "in.wav", "-o", "out", "--crest-mode", "floor", "--crest-floor",
         "8,10,12,12,12", "--side-band0-exempt", "--tp-meter", "8x"])
    cb = cli._controls(b)
    assert (cb.crest_mode, cb.crest_floors_db, cb.side_band0_exempt, cb.tp_meter) == \
        ("floor", (8.0, 10.0, 12.0, 12.0, 12.0), True, "8x")
    d = cli._controls(cli.build_parser().parse_args(["master", "in.wav", "-o", "out"]))
    assert (d.crest_mode, d.tp_meter, d.side_band0_exempt) == ("report", "8x", False)   # S6
    assert d.target_rms is None and d.rms_target == -4.1 and d.kick_detective is None
    assert cli.build_parser().parse_args(["master", "a.wav", "-o", "w", "--bpm", "120"]).bpm == 120.0


def test_bad_input_exits_2(tmp_path, capsys):
    assert cli.main(["analyse", str(tmp_path / "missing.wav"), "-o", str(tmp_path)]) == 2
    assert cli.main(["plan", str(tmp_path), "--tilt", "9"]) == 2


@pytest.mark.slow
def test_analyse_plan_render(wav, tmp_path):
    work = tmp_path / "work"
    assert cli.main(["analyse", str(wav), "-o", str(work)]) == 0
    for f in ("features.npz", "analysis.json", "source.json"):
        assert (work / f).exists()
    assert not (work / "features.tmp.npz").exists()
    src = json.loads((work / "source.json").read_text(encoding="utf-8"))
    assert src["path"] == str(wav.resolve()) and src["fs"] == FS and src["bpm"] is None
    assert src["sha256"] == cli.file_sha256(wav) and len(src["sha256"]) == 64
    # no --xover: key-locked crossovers, recorded with the request (none) and R_ref
    assert src["crossovers_requested"] is None and src["crossover_rule"] == "key"
    a = cli._load_analysis(work)
    assert src["crossovers_hz"] == list(cli.planner.key_crossovers(a["meta"]["crossover_tonic"]))
    assert src["r_ref_dbfs"] == -18.0 and src["input_gain_db"] == a["meta"]["input"]["gain_db"]
    assert cli.main(["plan", str(work), "--tilt", "0.5"]) == 0
    for f in ("plan.json", "plan.npz", "targets.json", "plan_record.json"):
        assert (work / f).exists()
    assert json.loads((work / "targets.json").read_text(encoding="utf-8"))["slope_db_oct"] == -4.0
    out = tmp_path / "m.wav"
    assert cli.main(["render", str(work / "plan.json"), "-o", str(out)]) == 0
    y, fs = io.read(out)
    x, _ = io.read(wav)
    yr, info, _ = solver.render_plan(x, FS, Plan.load(work / "plan.json"))
    assert np.max(np.abs(y - yr.astype(np.float32))) == 0.0
    for f in work.glob("*.json"):
        assert b"\r\n" not in f.read_bytes()


def test_analysis_cache_is_data_only(tmp_path):
    a = {"meta": {"fs": 44100, "crossovers": (100.0, 400.0), "gain": np.float64(-1.5),
                  "nan": float("nan"), "flag": np.bool_(True)},
         "rows": [{"x": np.arange(5.0), "m": np.zeros((2, 3), dtype=np.float32)}, None, "s"]}
    cli._save_analysis(tmp_path, a)
    b = cli._load_analysis(tmp_path)
    assert b["meta"]["crossovers"] == (100.0, 400.0) and b["meta"]["gain"] == -1.5
    assert np.isnan(b["meta"]["nan"]) and b["meta"]["flag"] is True
    assert np.array_equal(b["rows"][0]["x"], np.arange(5.0))
    assert b["rows"][0]["m"].dtype == np.float32 and b["rows"][1:] == [None, "s"]
    with pytest.raises(ValueError):
        cli._save_analysis(tmp_path, {"o": np.array([{}], dtype=object)})
    # object arrays are refused on load (no pickle), and a truncated cache is a ValueError
    np.savez(tmp_path / "features.npz", __tree__=np.frombuffer(b'{"o":{"__nd__":"a0"}}', np.uint8),
             a0=np.array([{}], dtype=object))
    with pytest.raises(ValueError):
        cli._load_analysis(tmp_path)
    cli._save_analysis(tmp_path, a)
    data = (tmp_path / "features.npz").read_bytes()
    (tmp_path / "features.npz").write_bytes(data[: len(data) // 2])
    with pytest.raises(ValueError):
        cli._load_analysis(tmp_path)


@pytest.mark.slow
def test_analysis_reuse(wav, tmp_path):
    """A matching cache is reused and source.json follows the input's path; a broken
    cache or a stopped analysis (no source.json) is analysed again; a pickle is never read."""
    work = tmp_path / "w"
    assert cli.main(["analyse", str(wav), "-o", str(work)]) == 0
    moved = tmp_path / "moved.wav"
    moved.write_bytes(wav.read_bytes())
    assert cli._reuse_analysis(work, moved, log=lambda m: None) is not None
    assert json.loads((work / "source.json").read_text(encoding="utf-8"))["path"] == \
        str(moved.resolve())
    # a cache from another analysis version, or from before versioning, is analysed again
    # (audit 13, Sol: a stale tempo or a missing field otherwise survives a re-render)
    current = (work / "source.json").read_text(encoding="utf-8")
    src = json.loads(current)
    assert src["analysis_version"] == cli.ANALYSIS_VERSION
    assert cli._matches(work / "source.json", moved)
    for old in (cli.ANALYSIS_VERSION - 1, None):
        if old is None:
            del src["analysis_version"]
        else:
            src["analysis_version"] = old
        (work / "source.json").write_text(json.dumps(src), encoding="utf-8")
        assert not cli._matches(work / "source.json", moved), old
        assert cli._reuse_analysis(work, moved, log=lambda m: None) is None, old
    (work / "source.json").write_text(current, encoding="utf-8")
    good = (work / "features.npz").read_bytes()
    (work / "features.npz").write_bytes(good[:1000])
    msgs = []
    assert cli._reuse_analysis(work, moved, log=msgs.append) is None
    assert msgs and "not reused" in msgs[0]
    (work / "features.npz").write_bytes(good)
    (work / "source.json").unlink()                      # an analysis stopped before the end
    assert cli._reuse_analysis(work, moved, log=lambda m: None) is None
    assert cli.main(["analyse", str(wav), "-o", str(work)]) == 0
    (work / "features.npz").unlink()
    (work / "features.pkl").write_bytes(b"not a pickle")  # a cache from elsewhere
    assert cli._reuse_analysis(work, wav, log=lambda m: None) is None
    assert cli.main(["plan", str(work)]) == 2


@pytest.mark.slow
def test_master_and_report(wav, tmp_path, monkeypatch):
    out = tmp_path / "out"
    run = cli.do_master(wav, out, cli.planner.Controls(), max_renders=3, log=lambda m: None)
    for f in ("master.wav", "plan.json", "plan.npz", "solve.json", "report.json", "report.md",
              "run.json", "ab/master_matched.wav"):
        assert (out / f).exists(), f
    assert run["renders"] >= 1 and run["peak_memory_mb"] > 0
    assert run["memory"]["available_mb"] > 0 and run["memory"]["estimate_mb"] > 0
    assert run["memory"]["threads"] == run["controls"]["threads"]
    assert set(run["constraints_met"]) >= {"rms", "slope", "true_peak"}
    assert "s95" not in run["constraints_met"] and run["s95_lufs"] is not None
    assert "comp_gr" not in run["constraints_met"]                 # reported only
    assert (out / "limiter.json").exists() and (out / "limiter_gr.npz").exists()
    sv = json.loads((out / "solve.json").read_text(encoding="utf-8"))
    blocking = [w for w in sv["warnings"] if not (w["stage"] == "limiter" and w.get("accepted"))]
    assert run["ok"] == (not run["misses"] and not blocking)          # S4
    assert not (out / "master_feasible.wav").exists()
    # the plain-language process log (processlog; its failures are logged, not raised)
    log_md = (out / "process-log.md").read_text(encoding="utf-8")
    assert (out / "process-log.html").exists()
    heads = ["## Settings used", "## Result", "## What each stage did", "## Renders",
             "## Sections and the contrast between them", "## Preserve mix", "## Warnings and flags"]
    assert all(h in log_md for h in heads)
    assert [log_md.index(h) for h in heads] == sorted(log_md.index(h) for h in heads)
    # a band with no measurable deficit (seen on a 40 s excerpt) does not stop the log
    rj = json.loads((out / "report.json").read_text(encoding="utf-8"))
    rj["planner"]["saturation"]["bands"][-1]["deficit_db"] = None
    (out / "report.json").write_text(json.dumps(rj), encoding="utf-8")
    from eluate import processlog
    assert "no deficit measurable" in processlog.write(out).read_text(encoding="utf-8")
    # a section with no measurable loudness (silent, or shorter than a 400 ms block) is
    # left out of the contrast, not a crash (audit 08)
    from eluate import uidata
    rj["sections"][0]["before_lufs"] = None
    rj["sections"][-1]["after_lufs"] = None
    (out / "report.json").write_text(json.dumps(rj), encoding="utf-8")
    ui = uidata.from_render(out)
    assert ui["sections"][0]["change"] is None and not ui["sections"][0]["flag"]
    assert ui["sections"][-1]["change"] is None
    assert "Sections and the contrast" in processlog.write(out).read_text(encoding="utf-8")
    assert "### 10. Final limiter" in log_md and "Render 1: the planned settings" in log_md
    before = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert cli.main(["report", str(out)]) == 0
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    # structured flags (WP1): the solver's constraints as measured, and the text flags are
    # their short heuristic checks; the app reads the same list with the run's controls
    ch = {f["kind"]: f for f in rep["flags_structured"] if f["kind"] != "stereo"}
    missed = {m["name"] for m in rep["solver"]["misses"]}
    for name, kind in (("rms", "loudness"), ("slope", "tilt"), ("true_peak", "tp")):
        assert ch[kind]["short"] == (name in missed), kind
    assert rep["flags"] == [f["text"] for f in rep["flags_structured"]
                            if f["short"] and f["kind"] not in ("loudness", "tilt", "tp")]
    assert rep["after"]["crest_db"] == pytest.approx(
        rep["after"]["sample_peak_dbfs"] - rep["after"]["rms_1s_loud_dbfs"])
    ui = uidata.from_render(out)
    assert ui["flags"] == rep["flags_structured"] and ui["crest"]["after"] == round(
        rep["after"]["crest_db"], 2)
    assert ui["controls"] == json.loads((out / "run.json").read_text(encoding="utf-8"))["controls"]
    # 8x and 32x on the written master (tp_meter "8x" at 24 bits, sprint S6), at the
    # -0.3 dBTP default ceiling
    assert max(rep["after"]["true_peak_dbtp"], rep["after"]["true_peak_32x_dbtp"]) <= -0.3 + 1e-6
    assert rep["targets"]["tp_meter"] == "8x"
    assert sf.info(str(out / "master.wav")).subtype == "PCM_24"
    # input normalisation and the export are in the report
    assert rep["chain"]["input"]["r_ref_dbfs"] == -18.0
    assert rep["chain"]["output"] == {"bits": 24, "dither": "none"}
    assert rep["after"]["rms_1s_loud_dbfs"] is not None
    assert rep["planner"]["crossovers"]["rule"] == "key"
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "0. Input: gain" in md and "Output: 24-bit PCM." in md and "key-locked" in md
    assert md.index("## Warnings") < md.index("## Gain reduction against budget") \
        < md.index("## What was done")
    assert rep["loudness"]["target_rms_dbfs"] == -4.1
    # the report command keeps the limiter data (audit 07 A16)
    assert rep["gain_reduction"]["limiter"] == before["gain_reduction"]["limiter"]
    # run.json (the completion marker) and plan.json carry one run id per run; a run
    # without a limiter GR estimate leaves no limiter_gr.npz from the previous run (3.1)
    assert json.loads((out / "plan.json").read_text(encoding="utf-8"))["run_id"] == run["run_id"]
    solve = cli.solver.solve

    def no_gr(*a, **k):
        res = solve(*a, **k)
        res.pre_limiter = None
        return res
    monkeypatch.setattr(cli.solver, "solve", no_gr)
    again = cli.do_master(wav, out, cli.planner.Controls(), max_renders=1, log=lambda m: None)
    assert again["phases"]["analyse_s"] == 0.0         # analysis reused (same SHA-256)
    assert again["run_id"] != run["run_id"] and not (out / "limiter_gr.npz").exists()
    assert json.loads((out / "run.json").read_text(encoding="utf-8"))["run_id"] == again["run_id"]
    assert Plan.load(out / "plan.json").limiter is not None
    assert not list(out.glob("*.tmp*"))
    assert not cli._matches(out / "source.json", wav, bpm=120.0)
    assert not cli._matches(out / "source.json", wav, crossovers=[120.0, 500.0, 2000.0, 8000.0])
    assert cli._matches(out / "source.json", wav)


def test_a_failed_master_is_not_done(tmp_path, monkeypatch):
    """Report 3.1: a re-render that fails or is stopped removes the previous run's
    completion marker (run.json) and process log first, so the list does not show the
    track as done with a mix of two runs' files; an unreadable run.json is not done."""
    from eluate import uidata
    inp = tmp_path / "a.wav"
    work = uidata.work_dir(inp)
    work.mkdir()
    side = {"rms_meter_dbfs": -4.2, "true_peak_dbtp": -0.3, "sample_peak_dbfs": -0.4,
            "rms_1s_loud_dbfs": -7.0, "lra_lu": 5.0, "crest_db": 6.6}
    rep = {"before": side, "after": side, "tilt": {"after": {"stereo": -4.6}},
           "flags_structured": []}
    (work / "report.json").write_text(json.dumps(rep), encoding="utf-8")
    (work / "source.json").write_text(json.dumps({"path": str(inp)}), encoding="utf-8")
    for f in cli.CLEARED_AT_START:
        (work / f).write_text("{}", encoding="utf-8")
    assert uidata.summary(inp)["state"] == "done"

    def stopped(*a, **k):
        raise RuntimeError("stopped")
    monkeypatch.setattr(cli, "_reuse_analysis", stopped)
    with pytest.raises(RuntimeError):
        cli.do_master(inp, work, cli.planner.Controls(), log=lambda m: None)
    assert not any((work / f).exists() for f in cli.CLEARED_AT_START)
    assert (work / "report.json").exists() and uidata.summary(inp)["state"] == "none"
    (work / "run.json").write_text('{"controls": ', encoding="utf-8")
    assert uidata.summary(inp)["state"] == "none"


@pytest.mark.slow
def test_master_streaming(wav, tmp_path):
    """--streaming: -1 dBTP ceiling, 16-bit with TPDF dither, true peak read on the file."""
    out = tmp_path / "out"
    assert cli.main(["master", str(wav), "-o", str(out), "--streaming", "--max-renders", "1"]) == 0
    assert sf.info(str(out / "master.wav")).subtype == "PCM_16"
    y, _ = io.read(out / "master.wav")
    assert solver.true_peaks(y, FS)["worst"] <= -1.0 + 1e-9
    plan = Plan.load(out / "plan.json")
    assert plan.output.bits == 16 and plan.output.dither == "tpdf"
    assert plan.limiter.ceiling_dbtp <= -1.0
    t = json.loads((out / "targets.json").read_text(encoding="utf-8"))
    assert t["true_peak_max_dbtp"] == -1.0
    # a saved plan renders the same file
    r = tmp_path / "r.wav"
    assert cli.main(["render", str(out / "plan.json"), "-o", str(r)]) == 0
    assert np.array_equal(io.read(r)[0], y)
