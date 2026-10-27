"""Profiles (sprint 2026-10-21 WP5): the live profile's table, its plan and the M/S link."""
import numpy as np
import pytest

from fixtures.e_signals import short_planner_analysis
from eluate import planner
from eluate.stages import multiband as mb

FS = 44100


@pytest.fixture(scope="module")
def pana():
    return short_planner_analysis(FS)


def test_every_profile_sets_every_key():
    keys = set(planner.PROFILES["electronic"])
    for name, prof in planner.PROFILES.items():
        assert set(prof) == keys, name
    assert planner.PROFILES["electronic"]["calibrated"]
    assert not planner.PROFILES["live"]["calibrated"]          # until R9
    with pytest.raises(ValueError):
        planner.Controls(profile="jazz")


def test_live_plan_switches_off_and_scales_down(pana):
    x, a = pana
    c = planner.Controls(profile="live", character="moderate")
    assert c.rms_target == -10.0 and c.ceiling_dbtp == -0.3           # R8: -0.3, not -1
    p = planner.plan_from_analysis(a, c, x)
    assert p.saturate is None
    lp = p.limiter
    assert lp.clip_target_db is None and lp.clip_threshold_db is None   # clipper off
    assert lp.clip_depth_db == 0.0 and lp.clip_onsets_s == []
    assert p.notch is None or p.notch.kick_mask is None
    assert (lp.max_gr_db, lp.max_outlier_gr_db) == (1.5, 3.0)
    assert (lp.speclim_allow_db, lp.speclim_max_cut_db) == (4.0, 3.0)
    tr = planner.CHARACTER_PRESETS["transparent"]
    assert (lp.attack_ms, lp.release_ms) == (tr["limiter_attack_ms"], tr["limiter_release_ms"])
    # band lookahead (2026-10-23): the band ceilings follow the character, the frequency-
    # dependent limiter the limiter timing (transparent under live)
    nb = len(p.multiband.crossovers_hz) + 1
    mo = planner.CHARACTER_PRESETS["moderate"]
    assert p.multiband.ceiling_lookahead_ms == planner._per_band(mo["band_lookahead_ms"], nb)
    assert lp.speclim_lookahead_ms == planner._per_band(tr["band_lookahead_ms"], nb)
    if p.speclim is not None:
        assert p.speclim.ceiling_lookahead_ms == lp.speclim_lookahead_ms
    assert p.sidestrip.fc_hz == 60.0                                     # R8: on at 60 Hz
    m = p.multiband
    assert m.ms_link and m.side_makeup_db == 0.0
    assert m.budget["comp_gr_db"] == 1.0 and m.budget["limit_max_db"] == 1.5
    assert all(r[c]["ratio"] <= 1.5 + 1e-9 for r in m.budget["result"] for c in range(2))
    assert p.glue.attack_ms == 20.0 and p.glue.budget["net_target_db"] == pytest.approx(1.0)
    eq = p.inputeq
    assert (eq.highpass.fc_hz, eq.highpass.q) == (25.0, planner.EQ_HPF_Q_BUTTERWORTH)
    for sec in (eq.low_shelf, eq.bell, eq.band_shelf):
        assert sec is None or abs(sec.gain_db) <= 1.5 + 1e-9
    for sec in (eq.high_shelf, eq.high_shelf_side):
        assert sec is None or -1.5 - 1e-9 <= sec.gain_db <= 2.0 + 1e-9
    t = planner.targets_from(a, c)
    assert t.slope_mode == "report" and t.rms_dbfs == -10.0
    rec = planner.plan_record(a, c)
    assert rec["profile"] == {"name": "live", "calibrated": False}
    assert rec["clipper"]["mode"] == "off"


def test_live_tone_target_is_the_mix_own_slope(pana):
    """R9 (maintainer, 2026-10-26): under live the input EQ does not steer toward the
    electronic reference slope. The target is the mix's own slope, Tilt moves it, and
    a thin sub is not pulled up; electronic keeps REFERENCE_SLOPE + Tilt."""
    _, a = pana
    measured = a["plan_inputs"]["tilt_slope_db_oct"]["stereo"]
    for tilt in (0.0, 1.0, -0.5):
        c = planner.Controls(profile="live", tilt_db=tilt)
        assert planner.targets_from(a, c).slope_db_oct == pytest.approx(measured + tilt)
        d = planner.inputeq_design(a, c)
        assert d["target_slope"] == pytest.approx(measured + tilt) and d["sub_pull"] is False
    e = planner.Controls(tilt_db=1.0)
    assert planner.targets_from(a, e).slope_db_oct == pytest.approx(planner.REFERENCE_SLOPE + 1.0)
    assert planner.inputeq_design(a, e)["sub_pull"] is True
    # at tilt 0 the rotation is zero, and without the sub pull every node's deficit is 0
    dc = planner.tone_deficit(a, planner.Controls(profile="live"))
    assert dc["delta_slope"] == pytest.approx(0.0) and dc["sub_shelf_correction_db"] == 0.0
    assert np.allclose(dc["deficit_db"], 0.0)


def test_iso226_contour_and_audibility_weights():
    """ISO 226:2003 at 1 kHz gives back its phon; the 80-phon weights rise from the bass to
    the 3 kHz region and fall in the top octave; above 12.5 kHz (no data) they never
    exceed the 10 kHz weight."""
    for ph in (40.0, 60.0, 80.0):
        assert planner.iso226_spl(ph, [1000.0])[0] == pytest.approx(ph, abs=0.05)
    w = dict(zip([31.5, 100, 1000, 3150, 10000, 16000, 18000],
                 planner.audibility_weights([31.5, 100, 1000, 3150, 10000, 16000, 18000])))
    assert w[31.5] < w[100] < w[1000] < w[3150]
    assert w[1000] == pytest.approx(1.0)
    assert w[16000] <= w[10000] + 1e-9 and w[18000] <= w[10000] + 1e-9
    assert all(planner.EQ_WEIGHT_RANGE[0] <= v <= planner.EQ_WEIGHT_RANGE[1] for v in w.values())


def test_live_eq_fit_is_audibility_weighted(pana):
    """Only live weights the input EQ fit (electronic and stems unchanged); on the same
    track the weighted fit matches the target better where the ear is most sensitive
    (1 to 5 kHz). Weighting moves accuracy there; it can make a shelf work harder."""
    _, a = pana
    live = planner.inputeq_design(a, planner.Controls(profile="live"))
    assert live["weight_phon"] == planner.EQ_WEIGHT_PHON
    assert "weight_phon" not in planner.inputeq_design(a, planner.Controls())
    assert planner.PROFILES["stems"]["eq_weighting"] is None
    raw = dict(live); raw.pop("weight_phon")
    _, wrec = planner.fit_inputeq(live, FS)
    _, rrec = planner.fit_inputeq(raw, FS)

    def mid_err(rec):
        f = np.asarray(rec["fit_nodes_hz"])
        e = np.asarray(rec["fit_db"]) - np.asarray(rec["fit_deficit_db"])
        return float(np.sqrt(np.mean(e[(f >= 1000) & (f <= 5000)] ** 2)))
    assert mid_err(wrec) <= mid_err(rrec) + 0.05


def test_live_residual_stops_at_the_fit_window():
    """Under live the EQ corrects the residual only from 100 Hz to 10 kHz, where the line
    was fitted; a deep top-octave roll-off (the extended line overshoots it) then drives
    no air boost. Electronic still corrects up to EQ_FIT_RANGE_HZ's top."""
    nodes = np.array([100.0, 1000.0, 8000.0, 10000.0, 12500.0, 16000.0])
    assert list(planner._residual_nodes(nodes, {"residual_window": True})) == [True] * 4 + [False] * 2
    assert all(planner._residual_nodes(nodes, {}))
    assert planner.PROFILES["live"]["eq_residual_window"]
    assert not planner.PROFILES["electronic"]["eq_residual_window"]
    # a flat mix whose top octave falls away hard: the residual says "boost the air"
    c = 1000.0 * 2.0 ** (np.arange(-15, 13) / 3.0)
    c = c[c <= 16500]
    res = np.where(c > 10000, -20.0, 0.0)
    design = {"nodes_hz": c.tolist(), "residual_db": res.tolist(), "measured_slope": -4.5,
              "target_slope": -4.5, "sub_shelf_db": 0.0, "rotation": 0.0, "sub_pull": False,
              "side_residual_db": None, "hpf": {"fc_hz": 25.0, "q": 0.707},
              "bounds": {"gain_db": 1.5, "hs_mid_db": 2.0, "hs_side_db": 2.0}}
    extended, _ = planner.fit_inputeq(design, FS)
    windowed, _ = planner.fit_inputeq({**design, "residual_window": True}, FS)
    gain = lambda e: 0.0 if e.air_shelf is None else e.air_shelf.gain_db
    assert gain(extended) > 1.0 and abs(gain(windowed)) < 0.2


def test_deficit_curve_sub_pull_off_leaves_a_thin_sub():
    nodes = [31.5, 40.0, 63.0, 125.0, 1000.0]
    on = planner.deficit_curve(nodes, -6.0, -6.0, shelf=-8.0)
    off = planner.deficit_curve(nodes, -6.0, -6.0, shelf=-8.0, sub_pull=False)
    assert on["sub_shelf_correction_db"] == pytest.approx(8.0 - planner.SUB_SHELF_MAX_DB)
    assert off["sub_shelf_correction_db"] == 0.0 and np.allclose(off["deficit_db"], 0.0)


def test_live_clipper_only_under_fast_and_aggressive(pana):
    _, a = pana
    for ch, on in (("transparent", False), ("slow", False), ("fast", True), ("aggressive", True)):
        p = planner.plan_from_analysis(a, planner.Controls(profile="live", character=ch))
        assert (p.limiter.clip_target_db is not None) == on, ch
        assert p.limiter.clip_depth_db == 0.0                          # no kick detective


def test_electronic_plan_unchanged_by_the_table(pana):
    _, a = pana
    p = planner.plan_from_analysis(a, planner.Controls(character="moderate"))
    assert not p.multiband.ms_link and p.multiband.side_makeup_db == 2.0
    assert p.limiter.max_gr_db == planner.LIMITER_MAX_GR_DB
    assert p.limiter.clip_target_db == planner.CLIP_TARGET_DB
    assert p.saturate is not None
    assert planner.targets_from(a, planner.Controls()).slope_mode == "constraint"


def test_ms_link_gives_m_and_s_one_gain(pana):
    x, _ = pana
    x = x[: 6 * FS].copy()
    x[:, 1] = 0.5 * x[:, 1] + 0.05 * np.random.default_rng(3).standard_normal(len(x))   # wide
    xo = [100.0, 400.0, 1600.0, 6400.0]
    # M and S compressed very differently, with a ceiling on one channel only
    thr = [[-30.0, -12.0] for _ in range(5)]
    ceil = [[None, -40.0] for _ in range(5)]
    p = mb.design(x, FS, thr, ratio=3.0, crossovers_hz=xo, ceiling_db=ceil,
                  side_makeup_db=0.0, mid_makeup_db=[0.0] * 5)
    p.ms_link = True
    y, info = mb.apply_with_info(x, FS, p)
    from eluate import bands
    bi = bands.split(bands.to_ms(x), FS, xo)
    bo = info["bands_out"]
    for k in range(5):
        m = (np.abs(bi[k, :, 0]) > 1e-6) & (np.abs(bi[k, :, 1]) > 1e-6)
        assert m.sum() > FS
        gm, gs = bo[k, m, 0] / bi[k, m, 0], bo[k, m, 1] / bi[k, m, 1]
        assert np.allclose(gm, gs, rtol=1e-9), k
    # the streamed path is the same signal
    assert np.allclose(mb.apply(x, FS, p), y, atol=1e-12)
    # unlinked, the two channels' gains differ
    p.ms_link = False
    _, info2 = mb.apply_with_info(x, FS, p)
    k = 2
    m = (np.abs(bi[k, :, 0]) > 1e-6) & (np.abs(bi[k, :, 1]) > 1e-6)
    assert not np.allclose(info2["bands_out"][k, m, 0] / bi[k, m, 0],
                           info2["bands_out"][k, m, 1] / bi[k, m, 1], rtol=1e-3)


def test_reported_slope_is_flagged_not_missed():
    """Live: the slope is a reported constraint (ok, so no tone refits) that the flags
    still mark when it is off its tolerance."""
    from eluate import evaluate
    rep = {"solver": {"constraints": [{"name": "slope", "value": -6.0, "target": -4.5, "tol": 0.3,
                                       "excess": 0.0, "ok": True, "mode": "report"}]}}
    tilt = {c["kind"]: c for c in evaluate._checks(rep)}["tilt"]
    assert tilt["short"] and "missed by 1.20" in tilt["text"]
    rep["solver"]["constraints"][0]["value"] = -4.6
    assert not {c["kind"]: c for c in evaluate._checks(rep)}["tilt"]["short"]


@pytest.mark.slow
def test_live_master_end_to_end(pana, tmp_path):
    import json

    import soundfile as sf

    from eluate import cli
    x, _ = pana
    wav = tmp_path / "live.wav"
    sf.write(wav, x, FS, subtype="PCM_24")
    out = tmp_path / "out"
    args = cli.build_parser().parse_args(["master", str(wav), "-o", str(out), "--profile", "live"])
    run = cli.do_master(wav, out, cli._controls(args), max_renders=2, log=lambda m: None)
    assert run["controls"]["profile"] == "live"
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    slope = next(c for c in rep["solver"]["constraints"] if c["name"] == "slope")
    assert slope["mode"] == "report" and slope["ok"]
    assert rep["chain"].get("saturate") is None
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "Profile: live (phase 1: the live profile defaults are uncalibrated)." in md
    assert "Profile: live" in (out / "process-log.md").read_text(encoding="utf-8")


def test_electronic_profile_is_the_module_constants():
    """Audit 13 #7: the electronic row is the constants the planner used before the table
    (D7's 80 ms release floor is a separate, intended change of every profile)."""
    from eluate.stages import multiband as mb
    e = planner.PROFILES["electronic"]
    assert e["target_rms_dbfs"] == planner.DEFAULT_TARGET_RMS_DBFS
    assert e["mb_budget"] == {} and not e["ms_link"] and e["side_makeup_db"] == 2.0
    assert e["saturate"] and e["glue_net_db"] is None and e["glue_attack_ms"] is None
    assert e["clipper_presets"] is None and e["kick"]
    assert (e["clip_target_db"], e["clip_sample_cap_db"], e["sat_max_db"]) == (
        planner.CLIP_TARGET_DB, planner.CLIP_SAMPLE_CAP_DB, planner.SAT_MAX_DRIVE_DB)
    assert (e["speclim_allow_db"], e["speclim_max_cut_db"]) == (
        planner.SPECLIM_ALLOW_DB, planner.SPECLIM_MAX_CUT_DB)
    assert (e["limiter_max_gr_db"], e["limiter_outlier_cap_db"]) == (
        planner.LIMITER_MAX_GR_DB, planner.LIMITER_OUTLIER_CAP_DB)
    assert e["limiter_timing"] is None and e["sidestrip_hz"] is None
    assert e["eq_bounds"] is None and e["eq_hpf"] is None and e["slope"] == "constraint"
    assert {**mb.default_budget("peak"), **e["mb_budget"]} == mb.default_budget("peak")
