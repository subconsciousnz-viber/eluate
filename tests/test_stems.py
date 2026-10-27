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
