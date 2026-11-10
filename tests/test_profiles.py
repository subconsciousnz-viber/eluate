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
    # live audit 2026-11-10 B.4: calibrated on synthetic fixtures and published statistics,
    # "calibrated" stays False until the maintainer's own live masters are measured (R9)
    assert not planner.PROFILES["live"]["calibrated"]
    with pytest.raises(ValueError):
        planner.Controls(profile="jazz")


def test_live_plan_switches_off_and_scales_down(pana):
    x, a = pana
    c = planner.Controls(profile="live", character="moderate")
    assert c.rms_target == -11.5 and c.ceiling_dbtp == -0.3           # B.4; R8: -0.3, not -1
    p = planner.plan_from_analysis(a, c, x)
    assert p.saturate is None
    lp = p.limiter
    assert lp.clip_target_db is None and lp.clip_threshold_db is None   # clipper off
    assert lp.clip_depth_db == 0.0 and lp.clip_onsets_s == []
    assert p.notch is None or p.notch.kick_mask is None
    # not sustained (the fixture's 4.3 onsets/s): the profile's table
    assert not planner.sustained_choice(a, c)["sustained"]
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
    # B.4 / A10: the side strip only for a decoherent sub band; this fixture's is 0
    assert planner.PROFILES["live"]["sidestrip_hz"] is None
    assert p.sidestrip is None and a["plan_inputs"]["band_decoherence"][0] < 0.3
    m = p.multiband
    assert m.ms_link and m.side_makeup_db == 0.0
    assert m.budget["comp_gr_db"] == 1.0 and m.budget["limit_max_db"] == 1.5
    assert all(r[c]["ratio"] <= 1.5 + 1e-9 for r in m.budget["result"] for c in range(2))
    assert p.glue.attack_ms == 20.0 and p.glue.budget["net_target_db"] == pytest.approx(1.0)
    eq = p.inputeq
    assert (eq.highpass.fc_hz, eq.highpass.q) == (25.0, planner.EQ_HPF_Q_BUTTERWORTH)
    # B.4 / A3: +-1 dB, and at Tilt 0 the high-pass and a narrow bell only
    for sec in (eq.low_shelf, eq.high_shelf, eq.high_shelf_side, eq.air_shelf,
                eq.air_shelf_side, eq.band_shelf):
        assert sec is None
    assert eq.bell is None or (abs(eq.bell.gain_db) <= 1.0 + 1e-9 and eq.bell.q >= 2.0)
    e1 = planner.plan_from_analysis(a, planner.Controls(profile="live", tilt_db=0.5)).inputeq
    for sec in (e1.low_shelf, e1.bell, e1.band_shelf, e1.high_shelf, e1.high_shelf_side,
                e1.air_shelf, e1.air_shelf_side):
        assert sec is None or abs(sec.gain_db) <= 1.0 + 1e-9
    t = planner.targets_from(a, c)
    assert t.slope_mode == "report" and t.rms_dbfs == -11.5
    rec = planner.plan_record(a, c)
    assert rec["profile"] == {"name": "live", "calibrated": False}
    assert rec["clipper"]["mode"] == "off"
    assert rec["sidestrip"]["used_hz"] is None and "off" in rec["sidestrip"]["rule"]


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


def test_live_clipper_never_under_auto(pana):
    """Live audit A1 (B.4): auto resolving to fast (a tempo above 140 BPM) never turns the
    clipper on under live; a user who chooses fast or aggressive gets it (the explicit-
    character gate). Electronic clips under auto as before."""
    _, a = pana
    for ch, on in (("transparent", False), ("slow", False), ("moderate", False),
                   ("fast", True), ("aggressive", True)):
        p = planner.plan_from_analysis(a, planner.Controls(profile="live", character=ch))
        assert (p.limiter.clip_target_db is not None) == on, ch
        assert p.limiter.clip_depth_db == 0.0                          # no kick detective
    for bpm in (150.0, 200.0):                     # auto -> fast, a given tempo
        c = planner.Controls(profile="live", bpm=bpm)
        ch = planner.character_settings(a, c)
        assert ch["preset"] == "fast" and ch["clipper"] == "off"
        assert "never under auto" in ch["clipper_reason"]
        assert planner.plan_from_analysis(a, c).limiter.clip_target_db is None
        assert planner.plan_record(a, c)["clipper"]["mode"] == "off"
        e = planner.plan_from_analysis(a, planner.Controls(bpm=bpm))
        assert e.limiter.clip_target_db == planner.CLIP_TARGET_DB


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
    assert ("Profile: live (calibrated on synthetic fixtures and published statistics; not yet "
            "on the maintainer's own live masters).") in md
    log = (out / "process-log.md").read_text(encoding="utf-8")
    assert "Profile: live (calibrated on synthetic fixtures and published statistics" in log
    assert rep["loudness"]["target_rms_dbfs"] == -11.5
    kinds = {f["kind"]: f for f in rep["flags_structured"]}
    assert {"psr", "plr", "crest_drop", "slope_reference", "clipper_live", "notches_tonal",
            "beat_grid"} <= set(kinds)
    assert not kinds["clipper_live"]["short"] and not kinds["beat_grid"]["short"]
    assert "sidestrip" not in rep["chain"] and "Side strip: off" in md


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
    assert e["eq_tilt0"] is None and e["sidestrip_decoherent"] is None
    assert e["notch_flags"] == planner.NOTCH_FLAGS
    assert e["sustained_onsets_per_s"] is None and e["sustained"] is None


# ------------------------------------------------ live calibration (live audit 2026-11-10 B.4)

def _with(a, **pi):
    """The analysis with plan_inputs entries replaced (shallow copies; the cached fixture
    is not modified)."""
    return {**a, "plan_inputs": {**a["plan_inputs"], **pi}}


def test_live_sustained_rule_switches_the_overrides(pana):
    """B.4: sustained = transients onsets_per_s < 0.5 under live: multiband 0.5 / 0.75 /
    0.75 / 1.0, glue 0.5, limiter 1.0 / 2.0, no speclim, no notches, 400 ms hits with the
    "no beat grid" source (A6), all in the plan record. Electronic and stems have no rule."""
    _, a = pana
    c = planner.Controls(profile="live", character="moderate")
    s = _with(a, onsets_per_s=0.09)                # the bowed-strings fixture's rate
    rec = planner.sustained_choice(s, c)
    assert rec["applies"] and rec["sustained"] and rec["threshold_onsets_per_s"] == 0.5
    p = planner.plan_from_analysis(s, c)
    b = p.multiband.budget
    assert (b["comp_gr_db"], b["comp_gr_max_db"], b["total_normal_db"], b["limit_max_db"]) == (
        0.5, 0.75, 0.75, 1.0)
    assert b["ratio_max"] == 1.5                   # the rest of the live budget stays
    assert p.glue.budget["net_db"] == 0.5
    assert (p.limiter.max_gr_db, p.limiter.max_outlier_gr_db) == (1.0, 2.0)
    assert p.limiter.speclim_allow_db is None
    assert p.notch.notches == []
    assert p.hits.beats_s is None and p.hits.source == planner.SUSTAINED_HITS_SOURCE
    r = planner.plan_record(s, c)
    assert r["sustained"]["sustained"] and r["sustained"]["overrides"]["glue_net_db"] == 0.5
    assert r["glue"]["net_target_db"] == 0.5 and r["limiter"]["max_gr_db"] == 1.0
    assert r["notches"] and all(not n["rendered"] and "sustained" in n["reason"]
                                for n in r["notches"])
    # the quiet-passage gain the top shelves' noise cap reads follows the lower budgets
    assert planner.quiet_gain_db(c, planner.profile_for(s, c)) < planner.quiet_gain_db(c)
    # a tempo the user gave still sets the hits
    pu = planner.plan_from_analysis(s, planner.Controls(profile="live", bpm=100.0))
    assert pu.hits.source.startswith("user bpm")
    # at the threshold and above: the profile's table
    for rate in (0.5, 3.0):
        q = planner.plan_from_analysis(_with(a, onsets_per_s=rate), c)
        assert q.limiter.max_gr_db == 1.5 and q.limiter.speclim_allow_db == 4.0
        assert q.multiband.budget["comp_gr_db"] == 1.0 and q.glue.budget["net_db"] == 1.0
        assert q.hits.source != planner.SUSTAINED_HITS_SOURCE
    # the rate is read from plan_inputs, else the transients; unknown is not sustained
    old = {k: v for k, v in a.items() if k != "transients"}
    old["plan_inputs"] = {k: v for k, v in a["plan_inputs"].items() if k != "onsets_per_s"}
    assert planner.analysis_onsets_per_s(old) is None and not planner.is_sustained(old, c)
    assert planner.is_sustained({**old, "transients": {"onsets_per_s": 0.2}}, c)
    assert a["plan_inputs"]["onsets_per_s"] == a["transients"]["onsets_per_s"] > 0.5
    # electronic and stems: no rule, the same plan whatever the onset rate
    for prof in ("electronic", "stems"):
        e = planner.Controls(profile=prof)
        assert not planner.sustained_choice(s, e)["applies"]
        assert planner.plan_from_analysis(s, e).to_json() == \
            planner.plan_from_analysis(a, e).to_json()


def test_live_sustained_plan_realises(pana):
    """The sustained budgets are inside every stage's range (the glue's serial net range
    starts at 0.5 dB since B.4) and the realised plan reads 400 ms frames."""
    x, a = pana
    s = _with(a, onsets_per_s=0.09)
    p = planner.plan_from_analysis(s, planner.Controls(profile="live"), x)
    assert p.glue.budget["net_target_db"] == pytest.approx(0.5)
    assert p.speclim is None and p.hits.beats_s is None
    assert all(r[c]["comp_target_db"] <= 0.75 + 1e-9 for r in p.multiband.budget["result"]
               for c in range(2) if r[c].get("comp_target_db") is not None)
    rep = planner.budget_report(p)
    assert rep["hits"] == planner.SUSTAINED_HITS_SOURCE
    assert rep["stages"]["limiter"]["budget_db"] == 1.0


def test_onset_rate_tells_sustained_from_transient_material():
    """The feature the rule reads: a held chord reads below 0.5 onsets/s, the fixture's
    kick and hat track well above it."""
    from eluate.features import transients
    from fixtures import synthetic as syn
    from fixtures.e_signals import short_track
    held = transients.analyse(syn.sustained_chord(FS, 12.0), FS)["onsets_per_s"]
    beat = transients.analyse(short_track(FS, 2)[0], FS)["onsets_per_s"]
    assert held < planner.PROFILES["live"]["sustained_onsets_per_s"] < beat


def test_live_side_strip_only_on_a_decoherent_sub(pana):
    """A10 (B.4): live sidestrip_hz None; 60 Hz only when band_decoherence[0] >= 0.3, else
    no side strip (a panned bass keeps its image). --subside still wins; electronic keeps
    its rule."""
    _, a = pana
    live = planner.Controls(profile="live")
    dec = list(a["plan_inputs"]["band_decoherence"])
    for d0, on in ((0.0, False), (0.29, False), (0.3, True), (0.6, True)):
        b = _with(a, band_decoherence=[d0] + dec[1:])
        sc = planner.sidestrip_choice(b, live)
        assert sc["used_hz"] == (60.0 if on else None) and not sc["clipped"], d0
        assert ("off" in sc["rule"]) != on
        p = planner.plan_from_analysis(b, live)
        assert (p.sidestrip.fc_hz if p.sidestrip is not None else None) == sc["used_hz"]
    assert planner.subside_hz(a, planner.Controls(profile="live", subside_hz=90.0)) == 90.0
    assert planner.subside_hz(a, planner.Controls()) == planner.SUBSIDE_DEFAULT_HZ
    t = planner.targets_from(a, planner.Controls(profile="live", side_band0_exempt=True))
    assert t.crest_exempt == []


def _tilt0_design(peak_db: float, rule: bool) -> dict:
    """A design whose residual asks for every section: a darker top (shelves), a one-node
    low-mid peak at 250 Hz of peak_db (the bell), and the same on the side."""
    c = 1000.0 * 2.0 ** (np.arange(-15, 13) / 3.0)
    c = c[c <= 16500]
    res = np.where(c >= 3000.0, -2.0, 0.0)
    res = np.where((c > 70.0) & (c < 110.0), 1.5, res)
    res[int(np.argmin(np.abs(c - 250.0)))] = peak_db
    d = {"nodes_hz": c.tolist(), "residual_db": res.tolist(), "measured_slope": -4.5,
         "target_slope": -4.5, "sub_shelf_db": 0.0, "rotation": 0.0, "sub_pull": False,
         "side_residual_db": res.tolist(), "hpf": {"fc_hz": 25.0, "q": 0.707},
         "bounds": dict(planner.PROFILES["live"]["eq_bounds"]), "residual_window": True}
    if rule:
        d["tilt0"] = dict(planner.PROFILES["live"]["eq_tilt0"])
    return d


def test_live_tilt0_eq_is_the_high_pass_and_a_narrow_bell(pana):
    """A3 (B.4): live at Tilt 0 fits no low, presence or air shelf (M or S) and no band
    shelf; a bell only on a low-mid excess above 4 dB, at Q 2 or more, within the +-1 dB
    bounds. Off Tilt 0 the full fit within +-1 dB; electronic is not affected."""
    full, _ = planner.fit_inputeq(_tilt0_design(6.0, False), FS)
    assert full.high_shelf is not None and full.air_shelf is not None
    assert full.high_shelf_side is not None
    eq, rec = planner.fit_inputeq(_tilt0_design(6.0, True), FS)
    assert rec["tilt0"]["shelves"] is False
    for sec in (eq.low_shelf, eq.high_shelf, eq.high_shelf_side, eq.air_shelf,
                eq.air_shelf_side, eq.band_shelf):
        assert sec is None
    assert eq.highpass.fc_hz == 25.0 and eq.highpass.q == 0.707
    assert eq.bell is not None and eq.bell.q >= planner.LIVE_BELL_Q_MIN
    assert -1.0 - 1e-9 <= eq.bell.gain_db < 0.0
    assert not any("shelf" in s for s in rec["at_bound"])
    # 3.5 dB is a bell for the full fit (above EQ_BELL_GATE_DB) but not at live Tilt 0
    eq35, _ = planner.fit_inputeq(_tilt0_design(3.5, True), FS)
    assert eq35.bell is None and eq35.low_shelf is None and eq35.high_shelf is None
    assert planner.fit_inputeq(_tilt0_design(3.5, False), FS)[0].bell is not None
    # the planner sets the rule at Tilt 0 exactly, under live only
    _, a = pana
    assert "tilt0" in planner.inputeq_design(a, planner.Controls(profile="live"))
    assert "tilt0" not in planner.inputeq_design(a, planner.Controls(profile="live", tilt_db=0.5))
    assert "tilt0" not in planner.inputeq_design(a, planner.Controls())
    assert planner.PROFILES["live"]["eq_bounds"] == {"gain_db": 1.0, "hs_mid_db": 1.0,
                                                     "hs_side_db": 1.0}


def test_live_renders_only_resonance_notches(pana):
    """A2 (B.4): notch_flags ("resonance",) under live; electronic and stems keep every
    flag. The same candidate flagged "musical" is rendered by electronic, not by live."""
    _, a = pana
    assert planner.PROFILES["live"]["notch_flags"] == ("resonance",)
    for prof in ("electronic", "stems"):
        assert planner.PROFILES[prof]["notch_flags"] == planner.NOTCH_FLAGS
    live, el = planner.Controls(profile="live"), planner.Controls()
    assert planner.plan_from_analysis(a, live).notch.notches       # the 3150 Hz ring
    m = _with(a, notches=[{**n, "flag": "musical"} for n in a["plan_inputs"]["notches"]])
    assert planner.plan_from_analysis(m, live).notch.notches == []
    rows = planner.plan_record(m, live)["notches"]
    assert rows and all(not r["rendered"] and 'renders only "resonance"' in r["reason"]
                        for r in rows)
    assert len(planner.plan_from_analysis(m, el).notch.notches) == \
        len(planner.plan_from_analysis(a, el).notch.notches) > 0


def _live_rep(**after) -> dict:
    rep = {"planner": {"profile": {"name": "live", "calibrated": False},
                       "clipper": {"mode": "off"}, "merged_notches": [],
                       "sustained": {"applies": True, "sustained": False, "onsets_per_s": 3.0,
                                     "threshold_onsets_per_s": 0.5}},
           "before": {"lra_lu": 14.0, "psr_p95_db": 14.0, "plr_db": 16.0, "crest_db": 15.0},
           "after": {"lra_lu": 12.5, "psr_p95_db": 12.0, "plr_db": 14.0, "crest_db": 13.0},
           "tilt": {"before": {"stereo": -4.7}, "after": {"stereo": -4.8}},
           "notches": [], "budget": {"hits": "analysis beat grid"}}
    rep["after"].update(after)
    return rep


def test_live_report_flags():
    """A5 / A7 (B.4): the live dynamics flags (PSR p95 < 10, amber < 8; PLR < 12; LRA drop
    > 3 LU on LRA > 12; crest drop > 3 dB), the informational slope check against
    REFERENCE_SLOPE (+-2 dB/oct) and the provenance flags. None on an electronic report."""
    from eluate import evaluate

    def checks(rep):
        return {c["kind"]: c for c in evaluate._checks(rep)}
    live_kinds = {"psr", "plr", "crest_drop", "slope_reference", "clipper_live",
                  "notches_tonal", "beat_grid"}
    k = checks(_live_rep())
    assert live_kinds <= set(k) and not any(c["short"] for c in k.values())
    k = checks(_live_rep(psr_p95_db=9.0, plr_db=11.0, lra_lu=10.5, crest_db=11.5))
    assert k["psr"]["short"] and k["psr"]["level"] == "flag" and "below 10 dB" in k["psr"]["text"]
    assert k["plr"]["short"] and k["crest_drop"]["short"]
    assert k["lra"]["short"] and k["lra"]["tol"] == 3.0          # 3.5 LU off a 14 LU premaster
    assert checks(_live_rep(psr_p95_db=7.5))["psr"]["level"] == "amber"
    rep = _live_rep(lra_lu=6.5)
    rep["before"]["lra_lu"] = 10.0                               # not wide: the 5 LU rule
    assert not checks(rep)["lra"]["short"] and checks(rep)["lra"]["tol"] == evaluate.FLAG_LRA_LU
    # B.4's sanity ranges (informational): S-max -13.5..-11, integrated by the LRA
    rep = _live_rep(integrated_lufs=-14.5, short_term_max_lufs=-12.3)
    rep["before"]["lra_lu"] = 8.0
    assert not checks(rep)["live_loudness"]["short"]
    rep["after"]["integrated_lufs"] = -12.5                      # above -13.5
    lv = checks(rep)["live_loudness"]
    assert lv["short"] and lv["level"] == "info" and lv["target"] == [-16.5, -13.5]
    rep["before"]["lra_lu"], rep["after"]["integrated_lufs"] = 20.0, -16.0
    assert not checks(rep)["live_loudness"]["short"]
    assert checks(rep)["live_loudness"]["target"] == [-18.5, -15.0]
    rep["after"]["short_term_max_lufs"] = -10.0                  # above -11
    assert checks(rep)["live_loudness"]["short"]
    rep = _live_rep()
    rep["tilt"] = {"before": {"stereo": -7.1}, "after": {"stereo": -7.2}}
    s = checks(rep)["slope_reference"]
    assert s["short"] and s["level"] == "info" and "darker" in s["text"]
    assert "Tilt +2.6" in s["text"]
    rep["tilt"]["after"]["stereo"] = -6.4                        # within 2 dB/oct
    assert not checks(rep)["slope_reference"]["short"]
    rep = _live_rep()
    rep["chain"] = {"clipper": {"mode": "whole signal"}}
    rep["planner"]["character"] = {"clipper_reason": "live profile: on only when chosen"}
    rep["planner"]["clipper"] = {"mode": "whole signal", "reason": "off: live profile"}
    c = checks(rep)["clipper_live"]
    assert c["short"] and c["text"] == ("Clipper on under live (whole signal; live profile: on "
                                        "only when chosen).")
    rep = _live_rep()
    rep["planner"]["sustained"].update(sustained=True, onsets_per_s=0.09)
    rep["planner"]["merged_notches"] = [{"f0_hz": 2014.0}]
    rep["budget"]["hits"] = planner.SUSTAINED_HITS_SOURCE
    k = checks(rep)
    assert k["notches_tonal"]["short"]
    assert "1 notch(es) rendered on sustained" in k["notches_tonal"]["text"]
    assert k["beat_grid"]["short"] and "No beat grid: sustained material" in k["beat_grid"]["text"]
    rep = _live_rep()
    rep["planner"]["merged_notches"] = [{"f0_hz": 2014.0}]
    rep["notches"] = [{"rendered": True, "flag": "musical", "merged_into": 0}]
    assert checks(rep)["notches_tonal"]["short"]
    # electronic: none of the live kinds, the 5 LU range rule
    rep = _live_rep(psr_p95_db=7.0, plr_db=8.0, lra_lu=10.5, crest_db=9.0)
    rep["planner"]["profile"] = {"name": "electronic", "calibrated": True}
    k = checks(rep)
    assert not live_kinds & set(k) and not k["lra"]["short"]


def test_live_texts_name_the_calibration_and_both_defaults(capsys):
    """A8 / A15: --target-rms names both defaults; --profile says how the live profile is
    calibrated (no "uncalibrated" left; the report and process log: the slow end-to-end)."""
    from eluate import cli
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["master", "--help"])
    h = " ".join(capsys.readouterr().out.split())
    assert "(default -4.1 electronic / -11.5 live)" in h
    assert "calibrated on synthetic fixtures and published statistics" in h
    assert "uncalibrated" not in h
