"""Stems: the null test, reading, attribution, the CLI flow (phase A) and the correction loop (phase B)."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fixtures import synthetic as syn
from eluate import bands, stems

FS = 44100
BPM = 128.0
BEAT = 60.0 / BPM


def _stem_set(bars: int = 4):
    """Kick, bass, pad and hat stems whose sum is the mix (stereo, 44.1 kHz)."""
    n = int(round(4 * bars * BEAT * FS))
    rng = np.random.default_rng(1)
    kick = np.zeros(n)
    k1 = syn.kick(FS)
    for b in range(4 * bars):
        i = int(round(b * BEAT * FS))
        m = min(len(k1), n - i)
        kick[i:i + m] += k1[:m]
    bass, _ = syn.bassline(FS, bpm=BPM, bars=bars, kick_on_beats=False, level=0.3)
    bass = bass[:n] if bass.ndim == 1 else bass[:n, 0]
    pad, _ = syn.chord_loop(FS, bpm=BPM, loops=max(1, bars // 4), level=0.08)
    pad = np.resize(pad[:, 0], n) * np.linspace(0.8, 1.0, n)
    hat = np.zeros(n)
    for e in range(8 * bars):
        i = int(round(e * BEAT / 2 * FS))
        h = syn._hat(FS, rng, level=0.25)
        m = min(len(h), n - i)
        hat[i:i + m] += h[:m]
    out = []
    for mono, w in ((kick, 0.0), (bass, 0.0), (pad, 0.4), (hat, 0.3)):
        st = np.stack([mono * (1 - w), mono * (1 + w) * 0.9], axis=1)
        out.append(st)
    return np.stack(out), ["kick", "bass", "pad", "hat"]


@pytest.fixture(scope="module")
def stem_set():
    return _stem_set()


def test_null_passes_on_the_sum_and_finds_the_offset(stem_set):
    st, _ = stem_set
    mix = st.sum(axis=0)
    t = stems.null_test(mix, st.sum(axis=0), FS)
    assert t["verdict"] == "pass" and t["residual_db"] < -100
    shifted = np.concatenate([np.zeros((37, 2)), mix])[: len(mix)]
    t = stems.null_test(shifted, mix, FS)
    assert t["offset_samples"] == 37 and t["verdict"] == "pass"


def test_null_aligns_a_side_only_sum(stem_set):
    """Antiphase channels fold to zero in mono; the joint correlation still finds the lag."""
    st, _ = stem_set
    s = st.sum(axis=0)[:, 0]
    side = np.stack([s, -s], axis=1)
    shifted = np.concatenate([np.zeros((5, 2)), side])[: len(side)]
    t = stems.null_test(shifted, side, FS)
    assert t["offset_samples"] == 5 and t["verdict"] == "pass"


def test_null_counts_a_missing_tail():
    """A mix that is only the stems' opening does not null: the tail is residual."""
    total = np.random.default_rng(3).standard_normal((5000, 2)) * 0.1
    t = stems.null_test(total[:2000], total, FS)
    assert t["verdict"] != "pass" and t["residual_db"] > stems.NULL_MISMATCH_DB
    t = stems.null_test(total, total[:2000], FS)            # and the other way round
    assert t["verdict"] != "pass"


def test_null_fails_on_a_stem_off_by_1_db(stem_set):
    st, _ = stem_set
    mix = st.sum(axis=0)
    wrong = st.copy()
    wrong[1] *= 10 ** (1 / 20)                         # the bass stem 1 dB hot
    t = stems.null_test(mix, wrong.sum(axis=0), FS)
    # the figure: what is left of the 1 dB error after the best overall gain
    b = wrong.sum(axis=0)
    g = np.sum(mix * b) / np.sum(b * b)
    expect = 10 * math.log10(np.mean((mix - g * b) ** 2) / np.mean(mix ** 2))
    assert t["residual_db"] == pytest.approx(expect, abs=0.05)
    assert t["verdict"] in ("processed", "mismatch") and t["residual_db"] > stems.NULL_PASS_DB
    assert t["worst_band_hz"] <= 250.0                  # the bass lives there
    assert "do not null" in stems.null_message(t) or "do not make" in stems.null_message(t)


def test_null_on_a_silent_mix_is_silent_not_nan(stem_set):
    """Report 12c: a silent mix gave NaN and "mismatch"; it is "silent", with a message
    that says whether the stems are silent too."""
    st, _ = stem_set
    quiet = np.zeros_like(st[0])
    for total, both in ((st.sum(axis=0), False), (quiet, True)):
        t = stems.null_test(quiet, total, FS)
        assert t["verdict"] == "silent" and t["residual_db"] is None
        assert t["stems_silent"] is both
        msg = stems.null_message(t)
        assert ("both silent" in msg) is both and "silent" in msg
        json.dumps(t, allow_nan=False)


def test_read_stems_needs_one_rate_and_length(tmp_path, stem_set):
    st, _ = stem_set
    # a is longer by 10 audible samples (not a silent render tail): an error
    sf.write(tmp_path / "a.wav", np.vstack([st[0], np.full((10, 2), 0.5)]), FS)
    sf.write(tmp_path / "b.wav", st[1], FS)
    with pytest.raises(ValueError, match="samples"):
        stems.read_stems([tmp_path / "a.wav", tmp_path / "b.wav"])
    sf.write(tmp_path / "c.wav", st[1], 48000)
    with pytest.raises(ValueError, match="Hz"):
        stems.read_stems([tmp_path / "a.wav", tmp_path / "c.wav"])


def test_read_stems_cuts_a_silent_render_tail(tmp_path, stem_set):
    """A stem longer by a near-silent tail (a DAW export) is cut to the shortest and logged;
    a silent tail longer than TRIM_MAX_S is still an error."""
    st, _ = stem_set
    tail = np.full((481, 2), 10 ** (-95 / 20))
    sf.write(tmp_path / "a.wav", st[0], FS, subtype="FLOAT")
    sf.write(tmp_path / "b.wav", np.vstack([st[1], tail]), FS, subtype="FLOAT")
    logs = []
    got, fs = stems.read_stems([tmp_path / "a.wav", tmp_path / "b.wav"], log=logs.append)
    assert got.shape == (2, st[0].shape[0], 2) and fs == FS
    assert len(logs) == 1 and "481 samples" in logs[0]
    long = np.zeros((int(stems.TRIM_MAX_S * FS) + 10, 2))
    sf.write(tmp_path / "c.wav", np.vstack([st[1], long]), FS, subtype="FLOAT")
    with pytest.raises(ValueError, match="ms more"):
        stems.read_stems([tmp_path / "a.wav", tmp_path / "c.wav"])


def test_band_moments_name_the_kick_low_and_the_hat_high(stem_set):
    st, names = stem_set
    total = st.sum(axis=0)
    xo = [100.0, 400.0, 1600.0, 6400.0]
    rate = 100.0
    n_ctrl = int(len(total) / FS * rate)
    t = np.arange(n_ctrl) / rate
    # a band-0 GR envelope dipping on the kicks, a band-4 one on the off-beat hats
    on_kick = np.min([np.abs(t - b * BEAT) for b in range(16)], axis=0) < 0.02
    on_hat = np.min([np.abs(t - (b + 0.5) * BEAT) for b in range(16)], axis=0) < 0.02
    _, low = stems._band_moments_rank(st, names, total, FS, xo, 0, 0, np.where(on_kick, -3.0, 0.0),
                                      rate, 2, "zero")
    _, high = stems._band_moments_rank(st, names, total, FS, xo, 4, 0, np.where(on_hat, -3.0, 0.0),
                                       rate, 2, "zero")
    assert low[0]["name"] == "kick"
    assert high[0]["name"] == "hat"


def test_range_shares_are_projections_of_the_sum():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((FS * 2, 2)) * 0.1
    # stem B cancels half of A: shares 2 and -1, summing to 1
    st = np.stack([a, -0.5 * a])
    rows = stems._range_rows(["a", "b"], st, st.sum(axis=0), FS, 200.0, 2000.0)
    sh = {r["name"]: r["share"] for r in rows}
    assert sum(sh.values()) == pytest.approx(1.0, abs=1e-9)
    assert sh["a"] == pytest.approx(2.0, abs=1e-6) and sh["b"] == pytest.approx(-1.0, abs=1e-6)
    assert "cancels" in stems._stem_line({}, rows[1])
    # uncorrelated: the louder stem first, shares still summing to 1
    st = np.stack([0.3 * rng.standard_normal((FS * 2, 2)) * 0.1, a])
    rows = stems._range_rows(["quiet", "loud"], st, st.sum(axis=0), FS, 200.0, 2000.0)
    assert rows[0]["name"] == "loud" and rows[0]["share"] > 0.8
    assert sum(r["share"] for r in rows) == pytest.approx(1.0, abs=1e-9)
    assert "share of the sum" in stems._stem_line({}, rows[0])


@pytest.mark.slow
def test_master_stems_end_to_end(tmp_path, stem_set):
    from eluate import cli
    st, names = stem_set
    paths = []
    for nm, s in zip(names, st):
        p = tmp_path / f"{nm}.wav"
        sf.write(p, s, FS, subtype="FLOAT")
        paths.append(str(p))
    sf.write(tmp_path / "mix.wav", st.sum(axis=0), FS, subtype="FLOAT")
    out = tmp_path / "out"
    assert cli.main(["master", "--stems", *paths, "--mix", str(tmp_path / "mix.wav"), "-o", str(out),
                     "--max-renders", "1"]) == 0
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["controls"]["profile"] == "stems"
    rep = json.loads((out / "stems-report.json").read_text(encoding="utf-8"))
    assert rep["stems"] == names and rep["null_test"]["verdict"] == "pass"
    assert (out / "stems-report.md").read_text(encoding="utf-8").startswith("# Stems report")
    # a mix that is not the stems' sum stops before mastering, with exit 2
    sf.write(tmp_path / "other.wav", st[0] + st[1], FS, subtype="FLOAT")
    assert cli.main(["master", "--stems", *paths, "--mix", str(tmp_path / "other.wav"),
                     "-o", str(tmp_path / "out2")]) == 2
    assert not (tmp_path / "out2" / "master.wav").exists()
    assert cli.main(["master", "a.wav", "--stems", *paths, "-o", str(out)]) == 2


# ---------------------------------------------------------------- phase B: the loop
# cli.do_master_stems with a fake master, attribution and stemfix: each pass is scripted
# as (RMS shortfall dB, limits hit); the score is shortfall + 0.5 per limit.

TARGET = -4.1
EQ = "input EQ high shelf +2.5 dB, at its bound"
LIM = "limiter on its outlier cap"
CREST = "crest low or section contrast lost"


class _Loop:
    def __init__(self, tmp_path, monkeypatch, script, plans=None):
        from eluate import cli
        self.cli, self.script = cli, script
        self.plans = plans                       # what stemfix.plan returns, per call
        self.plan_calls, self.reports, self.progress = [], [], []
        rng = np.random.default_rng(3)
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.paths = []
        for nm in ("kick", "bass"):
            p = tmp_path / f"{nm}.wav"
            sf.write(p, 0.1 * rng.standard_normal((4096, 2)), FS, subtype="FLOAT")
            self.paths.append(str(p))
        self.out = tmp_path / "set - eluate"
        loop = self

        def index(work):
            return 0 if Path(work) == loop.out else int(Path(work).name.split("-")[1])

        def fake_master(inp, out, controls, log=print, bpm=None, progress=None, **k):
            assert controls.profile == "stems"
            out.mkdir(parents=True, exist_ok=True)
            i = index(out)
            short, _ = loop.script[i]
            cli.io.write_json(out / "report.json", {"loudness": {"target_rms_dbfs": TARGET,
                                                                "after_rms_dbfs": TARGET - short}})
            (out / "master.wav").write_text(f"master of pass {i}", encoding="utf-8")
            (out / "plan.json").write_text(f"plan of pass {i}", encoding="utf-8")
            (out / "source.json").write_text("plain sum" if i == 0 else f"sum of pass {i}",
                                             encoding="utf-8")
            (out / "ab").mkdir(exist_ok=True)
            (out / "ab" / "master_matched.wav").write_text("ab", encoding="utf-8")
            cli.io.write_json(out / "run.json", {"run_id": f"r{i}", "input": str(inp), "bpm": bpm})
            return {"run_id": f"r{i}", "input": str(inp), "bpm": bpm}

        def fake_attribute(work, st, names, fs):
            _, limits = loop.script[index(work)]
            return {"stems": list(names),
                    "attribution": [{"limit": lim, "detail": {}, "stems": []} for lim in limits]}

        class Fix:
            @staticmethod
            def plan(rep, st, names, fs, prior=None, bpm=None):
                loop.plan_calls.append({"rep": rep, "st": st, "prior": prior, "bpm": bpm})
                if loop.plans is not None:
                    return loop.plans[len(loop.plan_calls) - 1]
                n = len(loop.plan_calls)
                return [{"stem": names[0], "kind": "eq", "params": {"n": j}, "reason": "r"}
                        for j in range(n)]

            @staticmethod
            def apply(st, fs, names, fixes):
                return st * 0.5

            @staticmethod
            def apply_sum(st, fs, names, fixes):
                return (st * 0.5).sum(axis=0)

            @staticmethod
            def describe(f):
                return f"{f['stem']}: {f['kind']} {f['params']['n']}"

        monkeypatch.setattr(cli, "do_master", fake_master)
        monkeypatch.setattr(cli.stems, "attribute", fake_attribute)
        monkeypatch.setattr(cli, "_stemfix", lambda: Fix)
        monkeypatch.setattr(cli, "do_report",
                            lambda out, ref=None, log=print, process_log=True: loop.reports.append(out))
        monkeypatch.setattr(cli, "_process_log", lambda out, log: None)

    def run(self, bpm=None):
        return self.cli.do_master_stems(
            self.paths, self.out, self.cli.planner.Controls(), log=lambda m: None, bpm=bpm,
            progress=lambda ph, **k: self.progress.append((ph, k)))

    def report(self):
        return json.loads((self.out / "stems-report.json").read_text(encoding="utf-8"))


def test_loop_stops_when_no_limit_is_hit(tmp_path, monkeypatch):
    lp = _Loop(tmp_path, monkeypatch, {0: (1.0, [CREST, "notch at 398 Hz (rendered)"])})
    run = lp.run()
    rep = lp.report()
    assert lp.plan_calls == [] and rep["kept"] == 0 and rep["stopped_why"] == "no_limit"
    assert [p["index"] for p in rep["passes"]] == [0] and rep["passes"][0]["score"] == 1.0
    assert run["stems"]["kept"] == 0 and run["stems"]["passes"] == 1
    assert json.loads((lp.out / "run.json").read_text(encoding="utf-8"))["stems"]["kept"] == 0
    assert not (lp.out / "pass-1").exists() and lp.reports == []
    assert [ph for ph, _ in lp.progress] == ["pass"]


def test_loop_stops_without_improvement_and_keeps_the_earlier_pass(tmp_path, monkeypatch):
    # pass 1 is 0.2 lower: not an improvement, and a tie for the kept pass
    lp = _Loop(tmp_path, monkeypatch, {0: (1.0, [EQ, LIM]), 1: (0.8, [EQ, LIM])})
    lp.run()
    rep = lp.report()
    assert rep["stopped_why"] == "no_improvement" and rep["kept"] == 0
    assert [round(p["score"], 6) for p in rep["passes"]] == [2.0, 1.8]
    assert (lp.out / "master.wav").read_text(encoding="utf-8") == "master of pass 0"
    assert lp.reports == []                      # nothing promoted
    # pass 1 ran in its own dir from its corrected sum; its A/B is dropped
    d = lp.out / "pass-1"
    assert (d / "corrected-sum.wav").exists() and (d / "master.wav").exists()
    assert not (d / "ab").exists()
    md = (lp.out / "stems-report.md").read_text(encoding="utf-8")
    assert "did not improve on pass 0" in md and "no later pass was at least 0.25 lower" in md


def test_loop_runs_at_most_two_passes_and_promotes_the_best(tmp_path, monkeypatch):
    lp = _Loop(tmp_path, monkeypatch, {0: (2.0, [EQ, LIM]), 1: (1.0, [EQ, LIM]), 2: (0.5, [LIM])})
    run = lp.run()
    rep = lp.report()
    assert len(lp.plan_calls) == 2 and rep["stopped_why"] == "max_passes" and rep["kept"] == 2
    # stemfix plans from the original stems every time, with the cumulative fixes before
    orig = stems.read_stems(lp.paths)[0]
    assert all(np.array_equal(c["st"], orig) for c in lp.plan_calls)
    assert lp.plan_calls[0]["prior"] is None and lp.plan_calls[1]["prior"] == [
        {"stem": "kick", "kind": "eq", "params": {"n": 0}, "reason": "r"}]
    # the kept pass's outputs are in the root, the plain sum's analysis cache stays
    assert (lp.out / "master.wav").read_text(encoding="utf-8") == "master of pass 2"
    assert (lp.out / "plan.json").read_text(encoding="utf-8") == "plan of pass 2"
    assert (lp.out / "source.json").read_text(encoding="utf-8") == "plain sum"
    assert not (lp.out / "corrected-sum.wav").exists() and (lp.out / "stems-sum.wav").exists()
    assert lp.reports == [lp.out]                # the report rebuilt against the plain sum
    assert run["run_id"] == "r2" and run["stems"]["kept"] == 2
    disk = json.loads((lp.out / "run.json").read_text(encoding="utf-8"))
    assert disk["run_id"] == "r2" and disk["stems"]["passes"] == 3
    assert [p["fix_lines"] for p in rep["passes"]] == [[], ["kick: eq 0"], ["kick: eq 0", "kick: eq 1"]]
    assert rep["passes"][2]["limits"] == [LIM]
    md = (lp.out / "stems-report.md").read_text(encoding="utf-8")
    assert "## Passes" in md and "- Pass 2 (kept), corrected stems: score 1.00" in md
    assert "  - kick: eq 1" in md and "after its 2 correction passes" in md
    assert "## Limits the kept pass (2) hit" in md
    # progress: a pass marker before each pass, then the kept pass's report
    assert [(ph, k.get("index")) for ph, k in lp.progress] == [
        ("pass", 0), ("pass", 1), ("pass", 2), ("keep", 2)]
    assert lp.progress[0][1]["passes_max"] == 3
    # every pass's master under a numbered name; pass 0's still holds pass 0 after the promote
    named = {f.name: f.read_text(encoding="utf-8") for f in (lp.out / "passes").iterdir()}
    assert named == {"stems 1 - plain sum.wav": "master of pass 0",
                     "stems 2 - pass 1.wav": "master of pass 1",
                     "stems 3 - pass 2 (kept).wav": "master of pass 2"}


def test_loop_keeps_the_best_when_the_last_pass_is_worse(tmp_path, monkeypatch):
    lp = _Loop(tmp_path, monkeypatch, {0: (2.0, [EQ, LIM]), 1: (0.5, [LIM]), 2: (1.0, [LIM])})
    lp.run()
    rep = lp.report()
    assert rep["stopped_why"] == "no_improvement" and rep["kept"] == 1
    assert (lp.out / "master.wav").read_text(encoding="utf-8") == "master of pass 1"
    assert (lp.out / "pass-2" / "master.wav").exists() and lp.reports == [lp.out]
    assert rep["attribution"] == [{"limit": LIM, "detail": {}, "stems": []}]


def test_loop_keeps_pass_0_when_a_correction_pass_fails(tmp_path, monkeypatch):
    lp = _Loop(tmp_path, monkeypatch, {0: (2.0, [EQ, LIM])})
    master = lp.cli.do_master

    def failing(inp, out, controls, **k):
        if Path(out) != lp.out:
            raise MemoryError("commit headroom")
        return master(inp, out, controls, **k)

    monkeypatch.setattr(lp.cli, "do_master", failing)
    run = lp.run()
    rep = lp.report()
    assert rep["stopped_why"] == "pass_failed" and rep["kept"] == 0 and run["stems"]["kept"] == 0
    assert (lp.out / "master.wav").read_text(encoding="utf-8") == "master of pass 0"
    assert (lp.out / "run.json").exists() and not (lp.out / "pass-1").exists()
    assert "correction pass 1 failed" in (lp.out / "stems-report.md").read_text(encoding="utf-8")


def test_loop_stops_when_stemfix_has_nothing_more(tmp_path, monkeypatch):
    fix = [{"stem": "bass", "kind": "comp", "params": {"n": 0}, "reason": "r"}]
    lp = _Loop(tmp_path, monkeypatch, {0: (2.0, [LIM]), 1: (1.0, [LIM])}, plans=[fix, list(fix)])
    (lp.out / "pass-2").mkdir(parents=True)       # an earlier render's
    lp.run(bpm=140.0)
    rep = lp.report()
    assert rep["stopped_why"] == "nothing_to_fix" and rep["kept"] == 1 and len(lp.plan_calls) == 2
    assert not (lp.out / "pass-2").exists()
    assert all(c["bpm"] == 140.0 for c in lp.plan_calls)
    # an empty plan on pass 0 stops before any correction pass
    lp2 = _Loop(tmp_path / "b", monkeypatch, {0: (2.0, [LIM])}, plans=[[]])
    lp2.run()
    assert lp2.report()["stopped_why"] == "nothing_to_fix" and not (lp2.out / "pass-1").exists()


def test_correction_passes_use_the_tempo_pass_0_found(tmp_path, monkeypatch):
    lp = _Loop(tmp_path, monkeypatch, {0: (2.0, [LIM]), 1: (2.0, [LIM])})
    from eluate import cli
    fake = cli.do_master

    def with_analysis(inp, out, controls, **k):
        r = fake(inp, out, controls, **k)
        if out == lp.out:
            cli.io.write_json(out / "analysis.json", {"structure": {"tempo_bpm": 172.1}})
        return r
    monkeypatch.setattr(cli, "do_master", with_analysis)
    lp.run()
    assert lp.plan_calls[0]["bpm"] == 172.1
    assert json.loads((lp.out / "pass-1" / "run.json").read_text(encoding="utf-8"))["bpm"] == 172.1


def test_pass_score_counts_real_limits_only():
    att = [{"limit": EQ}, {"limit": "multiband band 1 M at its exception"},
           {"limit": "frequency-dependent limiter band 3 S cut 4.0 dB"},
           {"limit": "clipper short of its target"}, {"limit": LIM},
           {"limit": "notch at 398 Hz (found)"}, {"limit": "notch at 398 Hz (rendered)"},
           {"limit": CREST}, {"limit": "noise exposed"}]
    rep = {"loudness": {"target_rms_dbfs": -4.1, "after_rms_dbfs": -6.2}}
    s = stems.pass_score(rep, att)
    assert len(s["limits"]) == 6 and math.isclose(s["rms_shortfall_db"], 2.1)
    assert math.isclose(s["score"], 2.1 + 3.0)
    # louder than the target is no shortfall; a report without loudness scores the limits
    assert stems.pass_score({"loudness": {"target_rms_dbfs": -4.1, "after_rms_dbfs": -3.0}}, [])["score"] == 0
    assert stems.pass_score({}, [{"limit": LIM}])["score"] == 0.5


@pytest.mark.slow
def test_master_stems_loop_end_to_end(tmp_path, stem_set, monkeypatch):
    """A real render of two passes: a fake stemfix halves the hat; the kept pass's master,
    report and A/B are in the root, which is a complete render."""
    from eluate import cli
    st, names = stem_set
    paths = []
    for nm, s in zip(names, st):
        p = tmp_path / f"{nm}.wav"
        sf.write(p, s, FS, subtype="FLOAT")
        paths.append(str(p))

    class Fix:
        @staticmethod
        def plan(rep, st, names, fs, prior=None, bpm=None):
            return [{"stem": "hat", "kind": "eq", "params": {"gain_db": -6.0}, "reason": "test"}]

        @staticmethod
        def apply(st, fs, names, fixes):
            out = st.copy()
            out[names.index("hat")] *= 0.5
            return out

        @staticmethod
        def describe(f):
            return f"{f['stem']}: {f['params']['gain_db']:+.0f} dB"
    monkeypatch.setattr(cli, "_stemfix", lambda: Fix)
    # every pass counts a limit, so pass 1 always runs
    real = stems.attribute
    monkeypatch.setattr(cli.stems, "attribute", lambda *a: {
        **real(*a), "attribution": real(*a)["attribution"] + [{"limit": LIM, "detail": {}, "stems": []}]})
    out = tmp_path / "out"
    run = cli.do_master_stems(paths, out, cli.planner.Controls(), max_renders=1, log=lambda m: None)
    rep = json.loads((out / "stems-report.json").read_text(encoding="utf-8"))
    assert len(rep["passes"]) == 2 and rep["stopped_why"] in ("nothing_to_fix", "no_improvement")
    assert (out / "pass-1" / "corrected-sum.wav").exists()
    disk = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert disk["stems"]["kept"] == rep["kept"] == run["stems"]["kept"]
    for f in ("master.wav", "report.json", "report.md", "ab/master_matched.wav", "process-log.md"):
        assert (out / f).exists(), f
    src = json.loads((out / "source.json").read_text(encoding="utf-8"))
    assert Path(src["path"]).name == cli.STEMS_SUM
    if rep["kept"]:
        assert (out / "master.wav").read_bytes() == (out / "pass-1" / "master.wav").read_bytes()


# ---------------------------------------------------------------- stems audit 2026-11-10

def _kinds(t):
    return [d["kind"] for d in t["diagnosis"]]


def test_null_recognises_polarity_and_large_offsets(stem_set):
    """Item 6: |argmax| aligns a polarity-inverted mix (it passes, reported); the search
    reaches +-250 ms (a 12 ms lag was a junk lag and "mismatch")."""
    st, _ = stem_set
    tot = st.sum(axis=0)
    t = stems.null_test(-tot, tot, FS)
    assert t["verdict"] == "pass" and t["polarity_inverted"] and t["offset_samples"] == 0
    assert "polarity" in _kinds(t) and "polarity" in stems.null_message(t)
    for ms in (12.0, 200.0):
        lag = int(round(ms * 1e-3 * FS))
        mix = np.concatenate([np.zeros((lag, 2)), tot])[: len(tot)]
        t = stems.null_test(mix, tot, FS)
        assert t["offset_samples"] == lag and t["verdict"] == "pass", ms
    # the inverted and delayed one together
    lag = int(0.05 * FS)
    t = stems.null_test(-np.concatenate([np.zeros((lag, 2)), tot])[: len(tot)], tot, FS)
    assert t["offset_samples"] == lag and t["verdict"] == "pass" and t["polarity_inverted"]


def test_null_names_what_differs(stem_set):
    """Item 6: gain mismatch, length / tail, a sub-sample offset and a limited reference
    are named, and "processed" says the run continues."""
    st, _ = stem_set
    tot = st.sum(axis=0)
    t = stems.null_test(2.0 * tot, tot, FS)                       # 6 dB louder: passes, said
    assert t["verdict"] == "pass" and "gain" in _kinds(t) and "6.0 dB louder" in stems.null_message(t)
    # a mix longer by an audible tail: the overlap nulls, the tail is the difference
    rng = np.random.default_rng(9)
    tail = 0.02 * rng.standard_normal((FS // 2, 2)) * np.exp(-np.arange(FS // 2) / (0.1 * FS))[:, None]
    t = stems.null_test(np.vstack([tot, tail]), tot, FS)
    assert "length" in _kinds(t) and t["overlap_residual_db"] < -100
    if t["verdict"] != "pass":
        assert "tail" in _kinds(t)
    # half a sample late (a resampled or SRC'd print): processed, named, the shift found
    n = len(tot)
    w = 2 * np.pi * np.fft.rfftfreq(n)
    half = np.fft.irfft(np.fft.rfft(tot, axis=0) * np.exp(-0.5j * w)[:, None], n=n, axis=0)
    t = stems.null_test(half, tot, FS)
    # (residual -17 dB, past the mismatch line, on this hat-heavy set: the shift explains
    # it, so the stems do make the mix: processed, not "do not make this mix")
    assert t["verdict"] == "processed" and "subsample" in _kinds(t)
    assert "The run continues" in stems.null_message(t)
    # through a hard limiter 3 dB up: the residual sits at the peaks
    pk = np.max(np.abs(tot))
    lim = np.clip(tot * 10 ** (3 / 20), -pk, pk)
    t = stems.null_test(lim, tot, FS)
    assert t["verdict"] in ("processed", "pass") and t["verdict"] != "mismatch"
    if t["verdict"] == "processed":
        assert "limited" in _kinds(t)


def test_exact_null_serialises_and_the_report_writes(tmp_path, stem_set):
    """Item 6: an exact null read -inf, became None in the JSON and crashed null_message
    (`--stems a.wav --mix a.wav`); it is floored at -200 dB."""
    st, names = stem_set
    t = stems.null_test(st[0], st[0], FS)
    assert t["residual_db"] == stems.NULL_RESIDUAL_FLOOR_DB and t["verdict"] == "pass"
    json.dumps(t, allow_nan=False)
    stems.write_report(tmp_path, {"stems": ["kick"], "attribution": [], "null_test": t})
    md = (tmp_path / "stems-report.md").read_text(encoding="utf-8")
    assert "residual -200.0 dB" in md
    # a report already cleaned (None residual) still writes
    t2 = dict(t, residual_db=None, gain_db=None)
    stems.write_report(tmp_path, {"stems": ["kick"], "attribution": [], "null_test": t2})
    assert "n/a" in (tmp_path / "stems-report.md").read_text(encoding="utf-8")


def test_processed_is_a_warning_mismatch_stops(tmp_path, stem_set):
    """Item 6: "processed" lets sum_stems continue (logged); "mismatch" stops it."""
    from eluate import cli
    st, names = stem_set
    paths = []
    for nm, s in zip(names, st):
        p = tmp_path / f"{nm}.wav"
        sf.write(p, s, FS, subtype="FLOAT")
        paths.append(p)
    tot = st.sum(axis=0)
    n = len(tot)
    w = 2 * np.pi * np.fft.rfftfreq(n)
    half = np.fft.irfft(np.fft.rfft(tot, axis=0) * np.exp(-0.5j * w)[:, None], n=n, axis=0)
    sf.write(tmp_path / "half.wav", half, FS, subtype="FLOAT")
    logs = []
    _, _, nt, sum_path = cli.sum_stems(paths, tmp_path / "w", tmp_path / "half.wav", log=logs.append)
    assert nt["verdict"] == "processed" and sum_path.exists()
    assert any(m.startswith("warning:") for m in logs)
    sf.write(tmp_path / "other.wav", st[0] + st[1], FS, subtype="FLOAT")
    with pytest.raises(ValueError, match="do not make this mix"):
        cli.sum_stems(paths, tmp_path / "w2", tmp_path / "other.wav", log=lambda m: None)


def test_check_paths_refuses_duplicates_and_missing(tmp_path, stem_set):
    """Item 9: one basename twice (fixes and rows go by name), the same file twice, a
    missing file, more than two channels: refused with the stem named."""
    st, _ = stem_set
    (tmp_path / "dup").mkdir()
    for p in (tmp_path / "kick.wav", tmp_path / "dup" / "kick.wav", tmp_path / "bass.wav"):
        sf.write(p, st[0], FS, subtype="FLOAT")
    with pytest.raises(ValueError, match="distinct file names: kick"):
        stems.check_paths([tmp_path / "kick.wav", tmp_path / "dup" / "kick.wav"])
    with pytest.raises(ValueError, match="given twice"):
        stems.read_stems([tmp_path / "kick.wav", tmp_path / "." / "kick.wav"])
    with pytest.raises(ValueError, match="not found"):
        stems.read_stems([tmp_path / "kick.wav", tmp_path / "nope.wav"])
    sf.write(tmp_path / "quad.wav", np.zeros((1000, 4)), FS)
    with pytest.raises(ValueError, match="quad.wav: 4 channels"):
        stems.read_stems([tmp_path / "kick.wav", tmp_path / "quad.wav"])
    assert [p.name for p in stems.check_paths([tmp_path / "kick.wav", tmp_path / "bass.wav"])] \
        == ["kick.wav", "bass.wav"]


def test_loo_rank_puts_a_cancelling_stem_last():
    """Item 7: three stems with 1 kHz bursts, carrier +0.5, in phase +0.3, anti-phase -0.45:
    by own peak the anti-phase one ranked second; by leave-one-out it cancels."""
    n = FS * 2
    t = np.arange(n) / FS
    burst = np.zeros(n)
    for c in (0.5, 1.0, 1.5):
        i = int(c * FS)
        burst[i:i + 2205] = np.sin(2 * np.pi * 1000 * t[:2205]) * np.hanning(2205)
    st = np.stack([np.stack([g * burst] * 2, axis=1) for g in (-0.45, 0.3, 0.5)])
    names = ["anti", "in", "carrier"]
    rows = stems._loo_rank(names, st, st.sum(axis=0), FS, np.array([0.525, 1.025, 1.525]), top=3)
    assert rows[-1]["name"] == "anti" and rows[-1]["peak_drop_db"] < 0
    assert all(r["peak_drop_db"] > 0 for r in rows[:2])
    assert stems._db(0.45) > stems._db(0.3)              # by its own peak it outranked "in"
    assert "cancels there" in stems._stem_line({}, rows[-1])
    # a stem that is the whole peak: a finite drop that survives JSON
    solo = stems._loo_rank(["only"], st[2:], st[2], FS, np.array([0.525]))
    assert math.isfinite(solo[0]["peak_drop_db"]) and solo[0]["peak_drop_db"] > 100


def _attr_work(tmp_path, monkeypatch, st, names, rep):
    from types import SimpleNamespace
    monkeypatch.setattr(stems.Plan, "load", staticmethod(lambda p: SimpleNamespace(multiband=None, speclim=None)))
    (tmp_path / "plan.json").write_text("{}", encoding="utf-8")
    (tmp_path / "report.json").write_text(json.dumps(rep), encoding="utf-8")
    return stems.attribute(tmp_path, st, names, FS)


def test_attribution_is_ms_aware_and_counts_the_budget(tmp_path, monkeypatch, stem_set):
    """Items 4, 5, 7: a side-only stem is present on the S shelf and absent from the M one;
    a notch on S reads its ring on S; the limiter held by its average budget and the
    loudness shortfall are one limit, ranked by leave-one-out; rejected notches are not
    limits; the planner's wanted gain rides on every EQ entry."""
    st, names = stem_set
    n = st.shape[1]
    t = np.arange(n) / FS
    rng = np.random.default_rng(5)
    from scipy import signal as sg
    hs = sg.sosfilt(sg.butter(4, 2000, "highpass", fs=FS, output="sos"), rng.standard_normal(n)) * 0.05
    wide = np.stack([hs, -hs], axis=1)                     # L = -R: no M at all
    pad = st[2].copy()
    ring = 0.05 * np.sin(2 * np.pi * 660.0 * t)
    pad[:, 0] += ring
    pad[:, 1] -= ring                                       # an S-only ring in the pad
    S = np.concatenate([st[:2], pad[None], st[3:], wide[None]])
    nm = names + ["wide"]
    gr_t = np.arange(0, n / FS, 0.01)
    gr = np.where(np.min([np.abs(gr_t - b * BEAT) for b in range(16)], axis=0) < 0.02, -2.0, -0.5)
    np.savez(tmp_path / "limiter_gr.npz", t=gr_t, gr_db=gr)
    rep = {"chain": {"inputeq": {"sections": {
               "high_shelf": {"gain_db": 2.5, "fc_hz": 1340.0},
               "high_shelf_side": {"gain_db": 3.0, "fc_hz": 1340.0},
               "bell": {"gain_db": -1.5, "fc_hz": 500.0, "q": 0.7}}}},
           "planner": {"inputeq": {"top_caps": {"presence": {"range_deficit_db": 0.4}},
                                   "side_caps": {"presence": {"range_deficit_db": 6.0}},
                                   "fit_nodes_hz": [250.0, 397.0, 500.0, 630.0, 1000.0],
                                   "fit_deficit_db": [0.0, -2.0, -3.3, -12.0, 0.0]}},
           "warnings": [{"stage": "limiter", "bound": "average", "needed_db": 5.3, "cap_db": 2.0},
                        {"stage": "loudness", "shortfall_db": 3.3, "cause": "limiter budget"}],
           "notches": [{"f0_hz": 660.0, "spans": [[0.0, n / FS]], "channel": "S", "q": 8.0,
                        "rendered": False, "reason": "S-only resonance; notch stage is not channel-aware"},
                       {"f0_hz": 9000.0, "spans": [[0.0, n / FS]], "rendered": False,
                        "reason": "suspect: spans cover 91 % of the track"},
                       {"f0_hz": 440.0, "spans": [[0.0, n / FS]], "rendered": False, "flag": "musical",
                        "max_cut_db": 0.0, "reason": "below the top 5 merged notches by peak excess"}],
           "loudness": {"target_rms_dbfs": -4.1, "after_rms_dbfs": -7.4}}
    att = _attr_work(tmp_path, monkeypatch, S, nm, rep)["attribution"]
    by = {e["detail"].get("section") or e["limit"]: e for e in att}
    side = [r["name"] for r in stems._range_rows(nm, S, S.sum(axis=0), FS, 1340.0, 20000.0, "S")]
    mid = [r["name"] for r in stems._range_rows(nm, S, S.sum(axis=0), FS, 1340.0, 20000.0, "M")]
    assert side[0] == "wide" and "wide" not in mid
    assert by["high_shelf_side"]["stems"][0]["name"] == "wide"
    assert by["high_shelf_side"]["detail"]["channel"] == "S"
    assert by["high_shelf"]["detail"]["wanted_db"] == 0.4
    assert by["bell"]["detail"]["wanted_db"] == pytest.approx(-3.3)
    assert by["bell"]["detail"]["wanted_from"] == "fit_deficit"
    notch = [e for e in att if e["kind"] == "notch"]
    s_notch = next(e for e in notch if e["detail"]["f0_hz"] == 660.0)
    assert s_notch["detail"]["read_on"] == "S" and s_notch["stems"][0]["name"] == "pad"
    assert s_notch["stems"][0]["excess_db"] > 20.0
    assert stems.is_limit(s_notch) and s_notch["limit"].endswith("(found)")
    for e in notch:
        if e is not s_notch:
            assert not stems.is_limit(e) and "skipped" in e["limit"]
    budget = next(e for e in att if e["kind"] == "budget")
    assert budget["detail"]["which"] == "limiter average budget and loudness shortfall"
    assert "peak_drop_db" in budget["stems"][0] and stems.is_limit(budget)
    sc = stems.pass_score(rep, att)
    assert budget["limit"] in sc["limits"] and len(sc["limits"]) == 5    # 3 EQ, budget, S notch
    assert sc["rms_shortfall_db"] == pytest.approx(3.3) and sc["dominant"] == "rms_shortfall"
    assert sc["shortfall_cause"] == "limiter budget"
    from eluate import evaluate
    json.dumps(evaluate._clean(att), allow_nan=False)


def test_spectra_are_read_once_per_attribution(monkeypatch, stem_set):
    """Item 10: every channel of the sum and of each stem is transformed once (scipy's csd
    re-transformed both inputs per pair, five calls a stem), and the result is Welch's:
    the shares and levels match scipy's welch / csd."""
    from scipy import signal as sg
    st, names = stem_set
    tot = st.sum(axis=0)
    calls = []
    real = stems._seg_fft
    monkeypatch.setattr(stems, "_seg_fft", lambda *a: calls.append(a[2:]) or real(*a))
    sp = stems._Spectra(st, tot, FS)
    nseg = 1 + (len(tot) - sp.nps) // (sp.nps // 2)
    blocks = -(-nseg // stems.SPECTRA_BLOCK)
    assert len(calls) == blocks * 2 * (1 + len(names))
    for lo, hi, ch in ((20.0, 120.0, "LR"), (350.0, 700.0, "LR"), (1340.0, 20000.0, "M"),
                       (1340.0, 20000.0, "S")):
        sp.rows(names, lo, hi, ch)
    assert len(calls) == blocks * 2 * (1 + len(names))      # the sections read, never recompute
    # against scipy (the per-pair welch / csd the attribution used before)
    nps = sp.nps
    f, pt = sg.welch(tot, FS, nperseg=nps, axis=0)
    sel = (f >= 350.0) & (f < 700.0)
    e_t = pt[sel].sum()
    want = {}
    for nm, s in zip(names, st):
        _, x = sg.csd(s, tot, FS, nperseg=nps, axis=0)
        _, ps = sg.welch(s, FS, nperseg=nps, axis=0)
        want[nm] = (float(np.real(x[sel]).sum() / e_t), 10 * np.log10(ps[sel].sum() / e_t))
    for r in sp.rows(names, 350.0, 700.0, "LR"):
        assert r["share"] == pytest.approx(want[r["name"]][0], abs=1e-9)
        assert r["level_re_sum_db"] == pytest.approx(want[r["name"]][1], abs=1e-6)


def test_notch_rows_read_the_spans_and_the_channel(stem_set):
    """Item 10: the notch readings, one pass per stem: a ring only inside a notch's spans
    is read there and not by a notch spanning elsewhere; an S ring is read on S."""
    st, names = stem_set
    n = st.shape[1]
    t = np.arange(n) / FS
    pad = st[2].copy()
    half = n // 2
    ring = np.where(t < half / FS, 0.05 * np.sin(2 * np.pi * 1500.0 * t), 0.0)
    pad[:, 0] += ring
    pad[:, 1] -= ring
    S = np.concatenate([st[:2], pad[None], st[3:]])
    nts = [{"f0_hz": 1500.0, "spans": [[0.0, half / FS]], "channel": "S"},
           {"f0_hz": 1500.0, "spans": [[half / FS + 0.1, n / FS]], "channel": "S"},
           {"f0_hz": 1500.0, "spans": [[0.0, half / FS]]}]
    on, off, lr = stems._notch_rows(names, S, nts, FS)
    pad_on = next(r for r in on if r["name"] == "pad")
    assert on[0]["name"] == "pad" and pad_on["excess_db"] > 20.0
    assert next(r for r in off if r["name"] == "pad")["excess_db"] < pad_on["excess_db"] - 15.0
    assert lr[0]["name"] == "pad"
