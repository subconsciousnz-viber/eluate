"""Before/after report, A/B and master-distance hook (spec 3.7)."""

import json
import math

import numpy as np
import pytest

from fixtures.e_signals import short_track
from eluate import evaluate, io, meter

FS = 44100


@pytest.fixture(scope="module")
def pair():
    x, meta = short_track(FS)
    x = x[: 10 * FS]
    y = np.tanh(3.0 * x) / np.tanh(3.0) * 0.89     # louder, compressed stand-in for a master
    return x, y


def test_clean_maps_non_finite():
    d = evaluate._clean({"a": np.nan, "b": [np.inf, 1.0], "c": np.array([1, 2]),
                         "d": np.float32(2.5), "e": np.bool_(True)})
    assert d == {"a": None, "b": [None, 1.0], "c": [1, 2], "d": 2.5, "e": True}
    json.dumps(d, allow_nan=False)


def test_ab_matches_on_s95_by_attenuating_louder(pair, tmp_path):
    x, y = pair
    r = evaluate.write_ab(x, y, FS, tmp_path)
    assert r["original_gain_db"] == 0.0 and r["master_gain_db"] < 0.0
    assert r["master_matched_s95"] == pytest.approx(r["original_matched_s95"], abs=0.01)
    xm, _ = io.read(tmp_path / "ab" / "original_matched.wav")
    ym, _ = io.read(tmp_path / "ab" / "master_matched.wav")
    dm, _ = io.read(tmp_path / "ab" / "difference.wav")
    assert np.max(np.abs(xm - x.astype(np.float32))) == 0.0     # quieter one untouched
    assert np.max(np.abs(dm - (ym.astype(np.float64) - xm))) < 1e-6
    r2 = evaluate.write_ab(y, x, FS, tmp_path / "rev")
    assert r2["original_gain_db"] < 0.0 and r2["master_gain_db"] == 0.0


def test_measure_track_fields(pair):
    x, _ = pair
    m = evaluate.measure_track(x, FS, [{"start": 0.0, "end": 5.0}, {"start": 5.0, "end": 10.0}])
    for k in ("s95_lufs", "true_peak_dbtp", "true_peak_32x_dbtp", "true_peak_4x_dbtp", "plr_db",
              "psr_p95_db", "section_lufs", "crest_p95_db", "band_corr", "slope_db_oct",
              "ltas_db", "short_term_lufs"):
        assert k in m, k
    assert len(m["section_lufs"]) == 2
    assert np.asarray(m["crest_p95_db"]).shape == (5, 3)
    assert m["true_peak_32x_dbtp"] >= m["sample_peak_dbfs"] - 1e-9


def test_report_writes_json_md_plots(pair, tmp_path):
    x, y = pair
    secs = [{"start": 0.0, "end": 5.0, "label": "a"}, {"start": 5.0, "end": 10.0, "label": "b"}]
    rec = {"notches": [{"f0_hz": 3150.0, "q": 8.0, "channel": "both", "flag": "resonance",
                        "max_cut_db": 3.0, "rendered": True, "reason": "",
                        "spans": [[1.0, 9.0]]}]}
    rep = evaluate.report(x, y, FS, tmp_path, sections=secs, plan_rec=rec,
                          master_ref=(y, FS))
    for f in ("report.json", "report.md", "ltas.png", "loudness.png",
              "ab/original_matched.wav", "ab/master_matched.wav", "ab/difference.wav"):
        assert (tmp_path / f).exists(), f
    text = (tmp_path / "report.json").read_text(encoding="utf-8")
    json.loads(text)
    assert "NaN" not in text and "Infinity" not in text
    assert b"\r\n" not in (tmp_path / "report.md").read_bytes()
    assert rep["delta"]["s95_lufs"] == pytest.approx(rep["after"]["s95_lufs"]
                                                     - rep["before"]["s95_lufs"])
    assert len(rep["sections"]) == 2 and rep["notches"][0]["f0_hz"] == 3150.0
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## Notches" in md and "3150.0" in md and "## A/B" in md
    assert rep["distance_to_master"]["s95_lufs"]["diff"] == pytest.approx(0.0, abs=1e-12)


def test_distance_to_master_resamples(pair):
    x, y = pair
    from scipy.signal import resample_poly
    y48 = resample_poly(y, 160, 147, axis=0)
    d = evaluate.distance_to_master(y, y48, FS, 48000)
    assert abs(d["s95_lufs"]["diff"]) < 0.05
    assert abs(d["slope_stereo"]["diff"]) < 0.05
    assert d["ltas_rms_diff_db"] < 0.5


def test_limiter_gr_estimate():
    z = np.ones((44100, 2))
    y = z * 0.5
    t, gr = evaluate.limiter_gr_estimate(z, y, 44100)
    assert np.allclose(gr, 20 * math.log10(0.5)) and t.size == 20


def test_ab_gate_aware_matching(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((FS * 6, 2)) * 10 ** (-75 / 20)      # S95 below the gate
    y = x * 10.0
    r = evaluate.write_ab(x, y, FS, tmp_path)
    assert r["matched_on"] in ("integrated", "none")
    z = rng.standard_normal((FS * 6, 2)) * 0.1
    assert evaluate.write_ab(z, z * 2, FS, tmp_path / "b")["matched_on"] == "s95"


def test_flags_on_synthetic_report():
    rep = {"sections": [{"start": 0.0, "label": "intro", "before_lufs": -28.0, "after_lufs": -7.0},
                        {"start": 60.0, "label": "drop", "before_lufs": -20.0, "after_lufs": -8.0},
                        {"start": 200.0, "label": "outro", "before_lufs": -45.0, "after_lufs": -15.0}],
           "before": {"lra_lu": 10.0}, "after": {"lra_lu": 2.0},
           "tilt": {"sub_shelf_before": {"stereo": 2.8}, "sub_shelf_after": {"stereo": -3.5}},
           "correlation": {"before": [1.0, 0.04], "after": [0.99, -0.13]},
           "crest": {"limits_db": [1.5, 2.0], "loss_db": [[0.1, 0.2, 0.3], [2.5, 1.0, None]]},
           "targets": {"crest_mode": "report"}}
    fl = evaluate._flags(rep)
    text = " ".join(fl)
    assert "Section balance: 1 section(s)" in text and "now louder" in text   # outro ignored
    assert "Loudness range" in text and "Sub:" in text
    assert "band 1 L/R correlation" in text and "band 0" not in text.split("Crest")[0]
    assert "band 1 2.5 dB > 2" in text
    rep["targets"]["crest_mode"] = "loss"
    assert not any(f.startswith("Crest") for f in evaluate._flags(rep))


def test_structured_flags_carry_every_check():
    rep = {"sections": [{"start": 0.0, "label": "drop", "before_lufs": -8.0, "after_lufs": -5.0},
                        {"start": 60.0, "label": "break", "before_lufs": -14.0, "after_lufs": -6.0}],
           "before": {"lra_lu": 6.0}, "after": {"lra_lu": 5.0, "crest_db": 6.4},
           "noise": {"before_db": -64.0, "after_db": -57.0, "threshold_db": -60.0,
                     "status": "exposed"},
           "solver": {"constraints": [
               {"name": "rms", "value": -4.8, "target": -4.1, "tol": 0.3, "excess": 0.4, "ok": False},
               {"name": "slope", "value": -4.6, "target": -4.5, "tol": 0.3, "excess": 0.0, "ok": True},
               {"name": "true_peak", "value": -0.31, "target": -0.3, "tol": 0.0, "excess": -0.01,
                "ok": True},
               {"name": "crest_band0", "value": 0.5, "target": 0.0, "tol": 1.5, "ok": True}]}}
    ch = {c["kind"]: c for c in evaluate._checks(rep)}
    assert set(ch) == {"loudness", "tilt", "tp", "section", "lra", "crest", "noise"}
    assert ch["loudness"]["short"] and ch["loudness"]["text"] == evaluate.miss_text(
        rep["solver"]["constraints"][0])
    assert not ch["tilt"]["short"] and ch["tilt"]["text"] is None and not ch["tp"]["short"]
    assert ch["section"]["short"] and ch["section"]["value"] == pytest.approx(5.0)
    assert ch["crest"] == {"kind": "crest", "value": 6.4, "target": evaluate.FLAG_CREST_DB,
                           "tol": 0.0, "short": True, "text": ch["crest"]["text"]}
    assert ch["noise"]["short"] and not ch["lra"]["short"]
    # the text flags are the short heuristic checks; the solver's misses stay in Warnings
    assert evaluate._flags(rep) == [ch[k]["text"] for k in ("section", "crest", "noise")]
    rep["noise"]["status"] = None
    assert not {c["kind"]: c for c in evaluate._checks(rep)}["noise"]["short"]


def test_what_was_done_lists_the_input_eq_first():
    from eluate.plan import (EqBell, EqHighPass, EqHighShelf, EqLowShelf, InputEqPlan, Plan,
                                SidestripPlan)
    p = Plan(fs=FS, inputeq=InputEqPlan(EqHighPass(25.0, 0.6), EqLowShelf(1.2, 120.0),
                                         EqBell(-1.5, 250.0, 2.0), EqHighShelf(1.0),
                                         design={"rotation": 0.25}),
             sidestrip=SidestripPlan(100.0))
    c = evaluate._chain_summary(p)
    assert c["inputeq"]["sections"]["low_shelf"]["gain_db"] == 1.2 and "tone" not in c
    md = "\n".join(evaluate._what_was_done_md({"chain": c}))
    first = md.split("\n")[2]
    assert first.startswith("1. Input EQ (static, zero phase): "
                            "high-pass 24 dB/oct at 25 Hz, Q 0.60")
    for t in ("low shelf +1.2 dB at 120 Hz", "bell -1.5 dB at 250 Hz, Q 2.00",
              "presence shelf +1.0 dB at 16000 Hz", "0.25 dB/oct solver rotation"):
        assert t in first, t
    assert "2. Side strip" in md and "Tone" not in md
    p.preserve_mix = True
    assert "inputeq" not in evaluate._chain_summary(p)


def _no_raw_lists(md: str) -> bool:
    import re
    return re.search(r"\[\s*-?\d|\[\[|\{'|\[None|\[\]", md) is None


def test_warning_text_names_stage_band_and_channel():
    rep = {"chain": {"multiband": {"crossovers_hz": [110.0, 440.0, 1760.0, 7040.0]}}}
    t = evaluate.warning_text({"stage": "multiband", "band": 2, "channel": "both",
                               "needed_db": 7.43, "cap_db": 6.0}, rep)
    assert t == ("Multiband band 2 (440-1760 Hz), M and S: needed 7.4 dB of peak gain "
                 "reduction, cap 6.0 dB; rendered at the cap.")
    assert "band 0 (below 110 Hz), S" in evaluate.warning_text(
        {"stage": "multiband", "band": 0, "channel": "S", "needed_db": 6.2, "cap_db": 6.0}, rep)
    t = evaluate.warning_text({"stage": "limiter", "needed_db": 8.91, "cap_db": 3.0,
                               "shortfall_db": 4.05, "shortfall_source": "render"}, rep)
    assert "needed 8.9 dB" in t and "cap 3.0 dB" in t and "4.0 dB short of the RMS target" in t
    t = evaluate.warning_text({"stage": "loudness", "target_dbfs": -7.7, "achieved_dbfs": -11.75,
                               "shortfall_db": 4.05, "cause": "limiter budget"}, rep)
    assert t == "Loudness: -11.8 dB RMS (Ozone meter) against the -7.7 dB target, 4.0 dB short " \
                "(limiter budget)."
    assert "Glue (serial)" in evaluate.warning_text(
        {"stage": "glue", "mode": "serial", "target_db": 1.5, "achieved_db": 1.2}, rep)
    assert "Clipper: the average output peak fell 0.80 dB" in evaluate.warning_text(
        {"stage": "clipper", "needed_db": 1.0, "achieved_db": 0.8, "cap_db": 1.5}, rep)


@pytest.mark.slow
def test_report_md_warnings_and_budget_sections(tmp_path):
    from fixtures.e_signals import short_planner_analysis
    from eluate import planner, solver
    x, a = short_planner_analysis(FS)
    # a target the budgets cannot reach on this short dynamic track (Ozone-meter scale)
    c = planner.Controls(target_rms=-2.5, character="moderate")
    p = planner.plan_from_analysis(a, c, x)
    t = planner.targets_from(a, c)
    rec = planner.plan_record(a, c)
    y, info, _ = solver.render_plan(x, FS, p)
    rep = evaluate.report(x, y, FS, tmp_path, plan=p, targets=t, plan_rec=rec, lim_info=info)
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    lines = md.split("\n")
    assert lines[2] == "## Warnings"                         # at the very top
    assert md.index("## Warnings") < md.index("## Gain reduction against budget") \
        < md.index("## Loudness") < md.index("## What was done")
    assert any(w["stage"] == "limiter" for w in rep["warnings"])
    assert "- Limiter: the loudness target needed" in md and "(estimated)" in md
    # one compressor row per band and channel, the glue, the clipper and the limiter, each
    # with its average and outlier peak GR, budget, outlier cap and status (2026-10-19)
    assert "| stage | band | channel | average peak | outlier peak | budget | outlier cap | " \
        "status |" in md
    assert "Average peak = the typical loud hit; outlier = the single largest moment." in md
    rows = [ln for ln in lines if ln.startswith("| multiband compressor | band ")]
    assert len(rows) == 10 and all(r.split("|")[-2].strip().startswith(
        ("ok", "under", "over", "gated")) for r in rows)
    for stage in ("| glue (serial, net) |", "| clipper (output peak) |", "| limiter |"):
        assert any(ln.startswith(stage) for ln in lines), stage
    assert "RMS -" in md and "AES17" in md and "S95" in md
    # the target's reading is already AES17 (the maintainer's meter); the 1 s reading stays
    assert rep["loudness"]["after_rms_aes17_dbfs"] == rep["loudness"]["after_rms_dbfs"]
    assert rep["loudness"]["after_rms_dbfs"] == pytest.approx(rep["after"]["rms_meter_dbfs"])
    assert rep["loudness"]["after_rms_1s_dbfs"] == pytest.approx(rep["after"]["rms_1s_loud_dbfs"])
    # what was done: character and why, makeup, ratios and thresholds, glue and clipper
    # modes, saturation drives
    for s in ("character moderate (chosen)", "makeup peak", "| ratio M | ratio S |",
              "Glue (serial)", "Clipper: whole signal", "Saturation (drive = deficit",
              "Limiter: gain", "0. Input: gain"):
        assert s in md, s
    assert _no_raw_lists(md), [ln for ln in lines if not _no_raw_lists(ln)]
    # 1 to 2 decimals: no number in report.md carries more
    import re
    assert not re.search(r"\d\.\d{3,}", md)
