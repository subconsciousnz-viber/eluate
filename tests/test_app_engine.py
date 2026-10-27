"""The desktop engine's protocol (eluate.app_engine) and the preview data (uidata)."""
import json

import numpy as np
import pytest
import soundfile as sf

from eluate import app_engine, uidata

FS = 48000


def _messages(out: str) -> list[dict]:
    return [json.loads(line[len(app_engine.PREFIX):]) for line in out.splitlines()
            if line.startswith(app_engine.PREFIX)]


def test_preview_sits_on_the_target(tmp_path, capsys):
    t = np.arange(4 * FS) / FS
    x = 0.1 * np.sin(2 * np.pi * 220 * t)
    p = tmp_path / "tone.wav"
    sf.write(p, np.stack([x, x], axis=1), FS)
    assert app_engine.main(["preview", str(p), "--target-rms", "-6"]) == 0
    msgs = _messages(capsys.readouterr().out)
    assert [m["event"] for m in msgs] == ["data"]
    ui = msgs[0]["ui"]
    assert ui["kind"] == "preview" and ui["duration"] == 4.0
    assert len(ui["env"]["before"]) == uidata.COLUMNS == len(ui["meter"]["before"])
    # a steady tone: every full 300 ms window reads the target after the shift
    mid = ui["meter"]["before"][100:-100]
    assert max(abs(v + 6.0) for v in mid) < 0.05


def test_missing_input_is_one_error_line(tmp_path, capsys):
    assert app_engine.main(["preview", str(tmp_path / "none.wav")]) == 2
    msgs = _messages(capsys.readouterr().out)
    assert len(msgs) == 1 and msgs[0]["event"] == "error" and msgs[0]["msg"]


def test_summary_with_and_without_a_report(tmp_path, capsys):
    """One data message per input, read from the work dir's JSON without audio; a report
    made before flags_structured / crest_db gets both rebuilt from what it has."""
    done, fresh = tmp_path / "a.wav", tmp_path / "b.wav"
    work = uidata.work_dir(done)
    assert work == tmp_path / "a - eluate"
    work.mkdir()
    side = {"rms_meter_dbfs": -4.2, "true_peak_dbtp": -0.31, "sample_peak_dbfs": -0.4,
            "rms_1s_loud_dbfs": -7.0, "lra_lu": 5.0}
    rep = {"before": {**side, "rms_1s_loud_dbfs": -12.0, "lra_lu": 6.0}, "after": side,
           "tilt": {"after": {"stereo": -4.6}},
           "solver": {"constraints": [{"name": "rms", "value": -4.2, "target": -4.1,
                                       "tol": 0.3, "ok": True}]}}
    (work / "report.json").write_text(json.dumps(rep), encoding="utf-8")
    (work / "source.json").write_text(json.dumps({"path": str(done), "bpm": 128.0}),
                                      encoding="utf-8")
    (work / "run.json").write_text(json.dumps({"controls": {"tilt_db": 0.5}}), encoding="utf-8")
    assert app_engine.main(["summary", str(done), str(fresh)]) == 0
    msgs = [m["ui"] for m in _messages(capsys.readouterr().out)]
    assert [m["kind"] for m in msgs] == ["summary", "summary"]
    a, b = msgs
    assert a["state"] == "done" and a["track"] == "a" and a["bpm"] == 128.0
    assert (a["loudness"], a["tilt"], a["truePeak"]) == (-4.2, -4.6, -0.31)
    assert a["crest"] == 6.6 and a["crestBefore"] == 11.6 and a["controls"] == {"tilt_db": 0.5}
    kinds = {f["kind"]: f["short"] for f in a["flags"]}
    assert kinds == {"loudness": False, "lra": False, "crest": True}
    assert b == {"kind": "summary", "track": "b", "input": str(fresh),
                 "work": str((tmp_path / "b - eluate").resolve()), "state": "none"}


def test_engine_flags_name_no_files():
    """Audit 13 #4: the shell grant-checks the input but passes the page's flags through,
    so no engine command may take a flag whose value is a path (--master-ref lives on the
    CLI only). Except the ones the shell owns: -o, and master's --stems / --mix (Open
    stems), which the shell refuses from the page and appends itself from the stem set
    the user picked (lib.rs engine_args, page_args_ok)."""
    from pathlib import Path
    p = app_engine._parser()
    sub = next(a for a in p._actions if a.__class__.__name__ == "_SubParsersAction")
    for name, sp in sub.choices.items():
        for a in sp._actions:
            # -o: the shell appends its own work dir after the page's flags and argparse
            # keeps the last, so the page cannot point it elsewhere (lib.rs start_job)
            shell_owned = "-o" in a.option_strings or (
                name in ("master", "stems-sum") and a.option_strings in (["--stems"], ["--mix"]))
            if a.option_strings and not shell_owned:
                assert a.type is not Path, (name, a.option_strings)


def test_ui_data_reads_only_the_granted_input(tmp_path, capsys, monkeypatch):
    """Audit 13 (GPT Sol): `ui-data` takes the input the shell granted, derives the work
    dir itself and refuses a source.json that names another file; a source.json naming
    the same file (by another spelling) or no file reads the given input."""
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    for p in (a, b):
        sf.write(p, np.zeros((FS // 10, 2)), FS)
    work = uidata.work_dir(a)
    work.mkdir()
    (work / "report.json").write_text("{}", encoding="utf-8")
    (work / "source.json").write_text(json.dumps({"path": str(b)}), encoding="utf-8")
    assert app_engine.main(["ui-data", str(a)]) == 2
    msgs = _messages(capsys.readouterr().out)
    assert len(msgs) == 1 and msgs[0]["event"] == "error" and "b.wav" in msgs[0]["msg"]
    monkeypatch.chdir(tmp_path)
    assert uidata._premaster({"path": "a.wav"}, a) == a
    assert uidata._premaster({}, a) == a
    assert uidata._premaster({"path": str(b)}, None) == b      # the maintainer's CLI form


def test_master_sweeps_stale_spill_dirs(tmp_path, capsys, monkeypatch):
    """Report 3.6: a killed engine leaves its solver spill dir behind; `master` sweeps the
    stale ones first. Age-based: a fresh dir, or an old dir with a freshly written spill
    file (a concurrent render's live dir), stays; other names stay."""
    import os
    import time
    from eluate import cli, solver
    now = time.time()
    old_t = now - solver.SPILL_STALE_S - 60
    stale, fresh, live = (tmp_path / f"{solver.SPILL_PREFIX}{n}" for n in ("a", "b", "c"))
    other = tmp_path / "other_old"
    for d in (stale, fresh, live, other):
        d.mkdir()
        (d / "x.npy").write_bytes(b"0")
    for d in (stale, live, other):
        os.utime(d / "x.npy", (old_t, old_t))
        os.utime(d, (old_t, old_t))
    (live / "pre.npy").write_bytes(b"0")
    os.utime(live, (old_t, old_t))        # a file rewritten in place leaves the dir's mtime
    assert solver.sweep_spill(tmp_path, now=now) == 1
    assert not stale.exists() and fresh.exists() and live.exists() and other.exists()
    assert solver.sweep_spill(tmp_path / "missing") == 0
    # the engine runs the sweep (default: the system temp dir) before a render
    calls = []
    monkeypatch.setattr(solver, "sweep_spill", lambda *a, **k: calls.append(a) or 0)

    def stop(*a, **k):
        raise RuntimeError("stop")
    monkeypatch.setattr(cli, "do_master", stop)
    assert app_engine.main(["master", str(tmp_path / "in.wav"), "-o", str(tmp_path / "w")]) == 2
    assert calls == [()]
    capsys.readouterr()


def test_estimate_phases_and_threads():
    """estimate (maintainer feedback 2026-10-24): the fitted phase times scale with the frames
    and fall with the threads; the time left falls phase by phase, a reused analysis costs
    nothing, and the solver's measured renders replace the fit for the later ones."""
    from eluate import estimate as est
    n = 10_000_000
    assert est.seconds("analyse", 2 * n, 30) == 2 * est.seconds("analyse", n, 30)
    assert est.seconds("render1", n, 1) > est.seconds("render1", n, 6) > est.seconds("render1", n, 30)
    assert est.seconds("render", n, 30) == est.LATER_RENDER * est.seconds("render1", n, 30)
    # 10 M frames on 30 threads: (5.74 + 20.3 / 30) s per million frames of analysis
    assert abs(est.seconds("analyse", n, 30) - 64.17) < 0.01
    lefts = [est.left(p, n, 30) for p in ("analyse", "plan", "solve", "report", "load")]
    assert lefts == sorted(lefts, reverse=True) and lefts[-1] > 0
    saved = est.left("analyse", n, 30) - est.left("analyse", n, 30, reused=True)
    assert saved == pytest.approx(est.seconds("analyse", n, 30))
    assert est.left("analyse", n, 30, reused=True) == est.left("plan", n, 30)
    assert est.left("solve", n, 30) == est.left("render", n, 30, render=1)
    tail = est.seconds("report", n, 30) + est.seconds("load", n, 30)
    # two renders expected: at render 2 one is left, timed from the measured first
    assert est.left("render", n, 30, render=2, measured=[100.0]) == pytest.approx(
        est.LATER_RENDER * 100.0 + tail)
    # past the expected count, the render that started is still counted, from the
    # measured later renders
    assert est.left("render", n, 30, render=4, measured=[100.0, 60.0, 80.0]) == \
        pytest.approx(70.0 + tail)
    with pytest.raises(ValueError):
        est.left("mix", n, 30)


def test_master_progress_stages(tmp_path, capsys, monkeypatch):
    """`master` sends one stage per phase from cli.do_master's progress callback: step 1 to
    4, the solver's renders with their bound, then Loading; the time left never rises
    except where the solver's measured renders replace the fit."""
    from eluate import cli, solver
    p = tmp_path / "a.wav"
    sf.write(p, np.zeros((FS * 2, 2)), FS)
    monkeypatch.setattr(solver, "sweep_spill", lambda *a, **k: 0)
    monkeypatch.setattr(app_engine, "_frames", lambda path: 48_000_000)

    def fake_master(inp, out, controls, log=print, bpm=None, progress=None, **k):
        progress("analyse", frames=48_000_000, threads=30, reused=True)
        for ph in ("plan", "solve"):
            progress(ph)
        for n in (1, 2):
            progress("render", render=n, renders_max=solver.render_bound())
        progress("report")
    monkeypatch.setattr(cli, "do_master", fake_master)
    from eluate import uidata
    monkeypatch.setattr(uidata, "from_render", lambda *a, **k: {"kind": "render"})
    assert app_engine.main(["master", str(p), "-o", str(tmp_path / "w"), "--threads", "30"]) == 0
    st = [m for m in _messages(capsys.readouterr().out) if m["event"] == "stage"]
    assert [(m["stage"], m.get("step")) for m in st] == [
        ("Analysing", 1), ("Analysing", 1), ("Planning", 2), ("Mastering", 3),
        ("Mastering", 3), ("Report", 4), ("Loading", None)]
    assert [m.get("render") for m in st[3:5]] == [1, 2]
    assert st[3]["renders_max"] == solver.render_bound() == 7
    assert solver.render_bound(1) == 2 and solver.render_bound(3) == 3
    # the first message knows the frames but not the reuse; once reused, no analysis
    assert st[0]["eta_s"] > st[1]["eta_s"] == st[2]["eta_s"] > st[3]["eta_s"]
    # render 2 is timed from render 1 as measured (here: no time at all)
    assert st[3]["eta_s"] > st[4]["eta_s"] >= st[5]["eta_s"] > st[6]["eta_s"] > 0


def test_analyse_writes_the_cache_master_reuses(tmp_path, capsys):
    """`analyse` (maintainer feedback 2026-10-24): the analysis a master with the same bpm and
    crossovers would make, cached in the work dir, so cli._reuse_analysis (what do_master
    calls) takes it; run again, it keeps the cache; another bpm does not match it."""
    from fixtures.e_signals import short_track
    from eluate import cli, parallel
    x, _ = short_track(44100, bars_per_section=1)
    inp = tmp_path / "a.wav"
    sf.write(inp, x, 44100, subtype="PCM_24")
    work = uidata.work_dir(inp)
    args = ["analyse", str(inp), "-o", str(work), "--bpm", "128", "--threads", "2"]
    assert app_engine.main(args) == 0
    msgs = _messages(capsys.readouterr().out)
    st = [m for m in msgs if m["event"] == "stage"]
    assert [m["stage"] for m in st] == ["Analysing"] and not any(m["event"] == "error" for m in msgs)
    assert st[0]["eta_s"] > 0 and st[0]["then_s"] > st[0]["eta_s"]
    assert parallel.threads() == 2
    assert (work / cli.ANALYSIS_FILE).exists() and (work / "source.json").exists()
    assert cli._reuse_analysis(work, inp, 128.0, None, log=lambda m: None) is not None
    assert app_engine.main(args) == 0
    msgs = _messages(capsys.readouterr().out)
    assert not [m for m in msgs if m["event"] == "stage"]
    assert any(m["event"] == "log" and m["msg"] == "analysis reused" for m in msgs)
    assert cli._reuse_analysis(work, inp, None, None, log=lambda m: None) is None


@pytest.mark.slow
def test_master_after_analyse_reuses_it(tmp_path, capsys):
    """The app's flow: `analyse` after the preview, then `master` with the same tempo
    reuses that analysis (log "analysis reused", no analysis time in run.json)."""
    from fixtures.e_signals import short_track
    x, _ = short_track(44100, bars_per_section=1)
    inp = tmp_path / "a.wav"
    sf.write(inp, x, 44100, subtype="PCM_24")
    work = uidata.work_dir(inp)
    assert app_engine.main(["analyse", str(inp), "-o", str(work), "--bpm", "128"]) == 0
    capsys.readouterr()
    assert app_engine.main(["master", str(inp), "-o", str(work), "--bpm", "128",
                            "--threads", "4"]) == 0
    msgs = _messages(capsys.readouterr().out)
    assert any(m["event"] == "log" and m["msg"] == "analysis reused" for m in msgs)
    st = [m for m in msgs if m["event"] == "stage"]
    assert [m["stage"] for m in st][:3] == ["Analysing", "Analysing", "Planning"]
    assert {m["stage"] for m in st} >= {"Mastering", "Report", "Loading"}
    assert msgs[-1]["event"] == "data" and msgs[-1]["ui"]["kind"] == "render"
    run = json.loads((work / "run.json").read_text(encoding="utf-8"))
    assert run["phases"]["analyse_s"] == 0.0


def _stem_files(tmp_path, n=3, seconds=3):
    rng = np.random.default_rng(5)
    paths = []
    for i in range(n):
        x = 0.05 * rng.standard_normal((seconds * FS, 2))
        p = tmp_path / f"song stem {i}.wav"
        sf.write(p, x, FS, subtype="FLOAT")
        paths.append(p)
    return paths


def test_stems_sum_writes_the_sum_in_the_set_work_dir(tmp_path, capsys):
    """stems-sum (Open stems): the sum lands in the set's work dir as stems-sum.wav, one data
    message names it, and uidata.work_dir keeps that file in its own folder."""
    paths = _stem_files(tmp_path)
    work = tmp_path / "song stem - eluate"
    mix = tmp_path / "mix.wav"
    total = sum(sf.read(p)[0] for p in paths)
    sf.write(mix, total, FS, subtype="FLOAT")
    assert app_engine.main(["stems-sum", *map(str, paths), "--mix", str(mix), "-o", str(work)]) == 0
    msgs = _messages(capsys.readouterr().out)
    data = [m["ui"] for m in msgs if m["event"] == "data"]
    assert len(data) == 1 and data[0]["kind"] == "stems" and data[0]["stems"] == 3
    assert data[0]["null"]["verdict"] == "pass"
    s = work / "stems-sum.wav"
    assert data[0]["sum"] == str(s.resolve()) and s.exists()
    np.testing.assert_allclose(sf.read(s)[0], total, atol=1e-6)
    assert uidata.work_dir(s) == work
    assert uidata.work_dir(tmp_path / "stems-sum.wav") == tmp_path / "stems-sum - eluate"


def test_stems_sum_stops_on_a_failed_null_test(tmp_path, capsys):
    paths = _stem_files(tmp_path)
    mix = tmp_path / "other.wav"
    sf.write(mix, 0.05 * np.random.default_rng(9).standard_normal((3 * FS, 2)), FS, subtype="FLOAT")
    work = tmp_path / "x - eluate"
    assert app_engine.main(["stems-sum", *map(str, paths), "--mix", str(mix), "-o", str(work)]) == 2
    msgs = _messages(capsys.readouterr().out)
    assert msgs[-1]["event"] == "error" and "do not make this mix" in msgs[-1]["msg"]
    assert not (work / "stems-sum.wav").exists()


def test_stems_master_progress_names_its_passes(tmp_path, capsys, monkeypatch):
    """A stem set's correction passes repeat steps 1 to 4 with pass / passes_max on every
    stage (the sum pass has none), each opened by Correcting stems; a kept correction
    pass ends with one Report stage, no pass, before Loading."""
    from eluate import cli, solver, uidata
    p = tmp_path / "stems-sum.wav"
    sf.write(p, np.zeros((FS, 2)), FS)
    monkeypatch.setattr(solver, "sweep_spill", lambda *a, **k: 0)
    monkeypatch.setattr(app_engine, "_frames", lambda path: 48_000_000)

    def one_pass(progress, reused):
        progress("analyse", frames=48_000_000, threads=30, reused=reused)
        progress("plan")
        progress("solve")
        progress("render", render=1, renders_max=solver.render_bound())
        progress("report")

    def fake_stems(paths, out, controls, mix=None, log=print, bpm=None, progress=None, **k):
        progress("pass", index=0, passes_max=3)
        one_pass(progress, True)
        progress("pass", index=1, passes_max=3)
        one_pass(progress, False)
        progress("keep", index=1)
    monkeypatch.setattr(cli, "do_master_stems", fake_stems)
    monkeypatch.setattr(uidata, "from_render", lambda *a, **k: {"kind": "render"})
    assert app_engine.main(["master", str(p), "-o", str(tmp_path), "--threads", "30",
                            "--stems", "a.wav", "b.wav"]) == 0
    st = [m for m in _messages(capsys.readouterr().out) if m["event"] == "stage"]
    assert [(m["stage"], m.get("step"), m.get("pass")) for m in st] == [
        ("Analysing", 1, None),
        ("Analysing", 1, None), ("Planning", 2, None), ("Mastering", 3, None), ("Report", 4, None),
        ("Correcting stems", None, 2),
        ("Analysing", 1, 2), ("Planning", 2, 2), ("Mastering", 3, 2), ("Report", 4, 2),
        ("Report", 4, None), ("Loading", None, None)]
    assert all(m["passes_max"] == 3 for m in st if m.get("pass"))
    # a correction pass's sum is new: its analysis is in the time left
    assert st[5]["eta_s"] > st[4]["eta_s"] and st[6]["eta_s"] > st[7]["eta_s"]
