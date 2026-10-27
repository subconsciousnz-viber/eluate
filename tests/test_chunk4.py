"""Chunk 4 of docs/gain-staging.md: the maintainer's answers of 2026-10-18 to the open
questions of chunks 2 and 3, and saturation as the inverse of the slope (6).
Synthetic signals only."""

import math

import numpy as np
import pytest

from fixtures.e_signals import short_planner_analysis
from eluate import planner, solver
from eluate.plan import EqBandShelf, EqBell, InputEqPlan, LimiterPlan, SaturatePlan
from eluate.stages import inputeq as eq
from eluate.stages import limiter as lim

FS = 44100


@pytest.fixture(scope="module")
def pana():
    return short_planner_analysis(FS)


# ------------------------------------------------------------ band shelf (PR 2, item 4)

def _lr4(f, fc, kind):
    """Analog 4th-order Linkwitz-Riley magnitude (the digital one prewarps; checked far
    below Nyquist)."""
    r = (f / fc) ** 4
    return 1.0 / (1.0 + r) if kind == "lp" else r / (1.0 + r)


@pytest.mark.parametrize("gain,lo,hi", [(-3.0, 150.0, 1200.0), (-1.5, 200.0, 800.0),
                                        (-2.0, 300.0, 2400.0)])
def test_band_shelf_magnitude(gain, lo, hi):
    p = InputEqPlan(band_shelf=EqBandShelf(gain, lo, hi))
    f = np.geomspace(20.0, 12000.0, 200)
    r = eq.response_db(p, f, FS)
    g = 10.0 ** (gain / 20.0)
    want = 20 * np.log10(1.0 - (1.0 - g) * _lr4(f, lo, "hp") * _lr4(f, hi, "lp"))
    sel = f < 6000.0
    assert np.max(np.abs(r[sel] - want[sel])) < 0.02
    assert np.all(r <= 1e-9) and r.min() >= gain - 1e-9           # cut only, bounded
    # flat bottom: within 0.25 dB of the full cut from an octave above the low corner to
    # an octave below the high one when they are three octaves apart; 24 dB/oct
    # transitions: one octave outside a corner the dip has fallen to 1/17 of its depth
    # (amplitude), (1/2)^4 / (1 + (1/2)^4)
    if hi / lo >= 8:
        mid = (f > 2 * lo) & (f < hi / 2)
        assert np.all(r[mid] < gain + 0.25) and np.ptp(r[mid]) < 0.1
    d = 20 * np.log10(1.0 - (1.0 - g) / 17.0)
    assert eq.response_db(p, [lo / 2.0], FS)[0] == pytest.approx(d, abs=0.05 * abs(d) + 0.01)
    # no ripple: the dip falls monotonically to its centre and rises after it
    k = int(np.argmin(r))
    assert np.all(np.diff(r[:k + 1]) <= 1e-9) and np.all(np.diff(r[k:]) >= -1e-9)


@pytest.mark.parametrize("phase", ["zero", "minimum"])
def test_band_shelf_stage_matches_its_response(phase):
    p = InputEqPlan(band_shelf=EqBandShelf(-3.0, 200.0, 1600.0), phase=phase)
    n = 4 * FS
    x = np.zeros((n, 2))
    x[n // 2 if phase == "zero" else 0] = 1.0
    y = eq.apply(x, FS, p)
    H = np.abs(np.fft.rfft(y[:, 0]))
    fr = np.fft.rfftfreq(n, 1.0 / FS)
    f = np.array([50.0, 150.0, 200.0, 560.0, 1600.0, 4000.0])
    meas = 20 * np.log10(np.interp(f, fr, H))
    assert np.max(np.abs(meas - eq.response_db(p, f, FS))) < 0.01
    assert np.array_equal(y[:, 0], y[:, 1])


def test_band_shelf_validation_and_round_trip():
    with pytest.raises(ValueError):
        EqBandShelf(0.5, 200.0, 800.0)                  # cut only
    with pytest.raises(ValueError):
        EqBandShelf(-3.5, 200.0, 800.0)                 # bounded to -3 dB
    with pytest.raises(ValueError):
        EqBandShelf(-1.0, 200.0, 210.0)                 # corners too close
    from eluate.plan import Plan
    p = Plan(fs=FS, inputeq=InputEqPlan(band_shelf=EqBandShelf(-2.0, 180.0, 900.0)))
    assert Plan.from_json(p.to_json()).inputeq == p.inputeq


def _design(res_by_hz):
    """An input EQ design record with a flat measured slope on target and the given
    residuals (dB) at the 1/3-octave nodes."""
    nodes = np.array([31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                      1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
                      12500, 16000])
    res = np.array([res_by_hz.get(float(f), 0.0) for f in nodes])
    return {"nodes_hz": nodes.tolist(), "residual_db": res.tolist(), "measured_slope": -4.5,
            "target_slope": -4.5, "sub_shelf_db": 0.0, "rotation": 0.0}


def test_planner_uses_the_band_shelf_for_a_broad_excess():
    # T3 (sprint 2026-10-21): gated against the neighbours, 100 to 500 Hz, at most 1.5 oct
    broad = {200.0: 4.0, 250.0: 4.5, 315.0: 4.5, 400.0: 4.5, 500.0: 4.0}
    e, rec = planner.fit_inputeq(_design(broad), FS)
    assert rec["low_mid_shape"] == "band shelf" and rec["low_mid_run_oct"] > 1.0
    assert e.band_shelf is not None and e.bell is None
    bs = e.band_shelf
    assert -3.0 <= bs.gain_db < -1.0 and bs.lo_hz < 250.0 and bs.hi_hz > 400.0
    assert bs.hi_hz <= bs.lo_hz * 2.0 ** planner.EQ_BAND_SHELF_MAX_OCT + 1.0
    assert any(t.startswith("band shelf") for t in planner.inputeq_text(e))
    # a narrow peak keeps the bell, cut no deeper than its excess
    narrow = {315.0: 3.0, 400.0: 3.5, 500.0: 3.0}
    e2, rec2 = planner.fit_inputeq(_design(narrow), FS)
    assert rec2["low_mid_shape"] == "bell" and rec2["low_mid_run_oct"] == pytest.approx(1.0)
    assert e2.bell is not None and e2.band_shelf is None
    assert e2.bell.gain_db >= -rec2["low_mid_excess_db"] - 0.05
    # below the gate (the natural spread of a line fit): neither
    e3, rec3 = planner.fit_inputeq(_design({400.0: 2.5}), FS)
    assert rec3["low_mid_shape"] is None and e3.bell is None and e3.band_shelf is None


def test_low_mid_run_width():
    nodes = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000]
    res = [0, 0, 3.6, 3.8, 4.2, 3.9, 0.5, 0, 0, 0, 0]
    r = planner.low_mid_run(nodes, res)
    assert r["run_hz"] == [160.0, 315.0] and r["run_oct"] == pytest.approx(4 / 3)
    assert r["shape"] == "band shelf" and r["excess_db"] == pytest.approx(4.2)
    assert r["neighbour_db"] == pytest.approx(0.0)
    # the same shape lifted 3 dB everywhere: still measured against its neighbours
    r2 = planner.low_mid_run(nodes, [v + 3.0 for v in res])
    assert r2["excess_db"] == pytest.approx(4.2)


# ------------------------------------------------------------ HPF Q (PR 2, item 3)

def test_hpf_q_never_rises_for_a_sub_deficit():
    # T11 (sprint 2026-10-21): the Q > 0.707 branch is gone; the excess rule is unchanged
    for v in (0.0, -3.0, -6.0, -9.0, -20.0):
        assert planner.hpf_q(v) == 0.71
    assert planner.hpf_q(9.0) == 0.5 and planner.hpf_q(6.0) == pytest.approx(0.6, abs=0.01)


def test_a_thin_sub_is_the_low_shelfs_to_lift():
    d = _design({})
    d["sub_shelf_db"] = -10.0                     # far below the target line
    e, rec = planner.fit_inputeq(d, FS)
    assert rec["sub_shelf_predicted_db"] < -9.0
    assert e.highpass.q == 0.71
    assert e.low_shelf is not None and e.low_shelf.gain_db > 0.0


# ------------------------------------------------------------ kick detective in auto (PR 3, 3)

@pytest.mark.parametrize("preset,on", [("slow", True), ("transparent", True),
                                       ("moderate", False), ("fast", False),
                                       ("aggressive", False)])
def test_auto_kick_detective_by_character(pana, preset, on):
    x, a = pana
    assert planner.kick_detective_choice(None, preset)[0] is on
    c = planner.character_settings(a, planner.Controls(character=preset))
    assert c["kick_detective"] is on and c["kick_detective_reason"].startswith("auto")
    p = planner.plan_from_analysis(a, planner.Controls(character=preset))
    assert (p.limiter.clip_target_db is None) is on
    # the user's choice wins either way
    assert planner.character_settings(
        a, planner.Controls(character=preset, kick_detective=not on))["kick_detective"] is (not on)


def test_auto_character_slow_turns_the_kick_detective_on(pana):
    _, a = pana
    c = planner.character_settings(a, planner.Controls(bpm=60.0))
    assert c["preset"] == "slow" and c["kick_detective"]
    rec = planner.plan_record(a, planner.Controls(bpm=60.0))
    assert rec["clipper"]["mode"] == "kick detective"
    assert "slow" in rec["clipper"]["reason"]


# ------------------------------------------------------------ clipper target (PR 3, 4)

def _dynamic(seconds=4.0, seed=7):
    n = int(seconds * FS)
    t = np.arange(n) / FS
    x = 0.05 * np.random.default_rng(seed).standard_normal((n, 2)) \
        * (1.0 + 0.7 * np.sin(2 * np.pi * 0.3 * t))[:, None]
    for k, at in enumerate((1.0, 2.6)):
        i = int(at * FS)
        x[i:i + int(0.03 * FS)] *= 4.0 + k
    return x


@pytest.mark.parametrize("knee", [0.0, 1.0])
def test_clipper_target_is_the_output_peak(knee):
    x = _dynamic()
    lp = LimiterPlan(ceiling_dbtp=-0.3, clip_knee_db=knee, clip_target_db=1.0)
    sol = solver.clip_threshold_for_output_peak(x, FS, lp, 1.0)
    p_in = lim.peak_8x_db(x)
    # the output peak (after the curve and its band-limit) falls by the target
    m = solver.limit_blocked(x, FS, LimiterPlan(clip_threshold_db=sol["threshold_db"],
                                                clip_knee_db=knee), measure_only=True)
    assert p_in - m["detector_peak_dbfs"] == pytest.approx(1.0, abs=0.02)
    assert sol["peak_gr_db"] == pytest.approx(1.0, abs=0.02)
    # the curve's own reduction at the 8x peak is larger: the band-limit lets some back
    assert sol["curve_gr_db"] > sol["peak_gr_db"]
    old = lim.clip_threshold_for_peak_gr(x, 1.0, knee)["threshold_db"]
    assert sol["threshold_db"] < old
    # and it does not move with the limiter's input gain
    _, info = solver.limit_blocked(x, FS, LimiterPlan(input_gain_db=5.0,
                                                      clip_threshold_db=sol["threshold_db"],
                                                      clip_knee_db=knee))
    assert p_in + 5.0 - info["detector_peak_dbfs"] == pytest.approx(1.0, abs=0.02)


@pytest.mark.slow
def test_planner_clipper_record_reads_the_output_peak(pana):
    x, a = pana
    p = planner.plan_from_analysis(a, planner.Controls(character="moderate"), x)
    c = p.limiter.budget["clipper"]
    assert c["measured_on"].startswith("output peak") and c["reading"] == "average"
    assert c["split_hz"] == p.multiband.crossovers_hz[0]   # the clipper acts above the sub
    # the average fall hits the target; the per-sample cap holds the outlier
    assert c["outlier_gr_db"] <= planner.CLIP_SAMPLE_CAP_DB + 0.05
    assert c["peak_gr_db"] == pytest.approx(1.0, abs=0.02)
    assert c["curve_gr_db"] >= c["peak_gr_db"] - 1e-6
    st = planner.budget_report(p)["stages"]["clipper"]
    assert st["planned_db"] == pytest.approx(c["peak_gr_db"])
    # the max reading (one hit): the track's output peak falls by the target
    from dataclasses import replace
    q = planner.realise(replace(planner.plan_from_analysis(a, planner.Controls(
        character="moderate")), hits=None), x, FS)
    cq = q.limiter.budget["clipper"]
    assert cq["reading"] == "max" and cq["peak_gr_db"] == pytest.approx(1.0, abs=0.02)


def test_silent_input_gives_no_clipping():
    sol = solver.clip_threshold_for_output_peak(np.zeros((FS, 2)), FS, LimiterPlan(), 1.0)
    assert sol["peak_gr_db"] == 0.0 and sol["threshold_db"] == 0.0


# ------------------------------------------------------------ saturation (6)

@pytest.mark.parametrize("deficit,drive", [(-1.0, 0.0), (0.0, 0.0), (0.2, 0.0), (0.25, 0.5),
                                           (0.6, 0.5), (0.75, 1.0), (1.3, 1.5), (1.74, 1.5),
                                           (1.75, 2.0), (2.6, 2.0), (9.0, 2.0),
                                           (math.nan, 0.0)])
def test_saturation_drive_follows_the_deficit(deficit, drive):
    assert planner.saturation_drive_db(deficit) == drive


def test_plan_saturate_per_band(pana, monkeypatch):
    _, a = pana
    c = planner.resolve(a, planner.Controls())
    monkeypatch.setattr(planner, "band_deficits", lambda an, co: [5.0, 0.6, -1.0, 1.8, 3.0])
    sat, rec = planner.plan_saturate(a, c)
    assert isinstance(sat, SaturatePlan)
    # the sub never, M and S alike, except the high-mid S band (width rule, 2 dB)
    assert sat.drive_db == [[0.0, 0.0], [0.5, 0.5], [0.0, 0.0], [2.0, 2.0], [2.0, 2.0]]
    assert sat.harmonic == [1.0] * 5
    assert [r["deficit_db"] for r in rec["bands"]] == [5.0, 0.6, -1.0, 1.8, 3.0]
    assert [r["drive_m_db"] for r in rec["bands"]] == [0.0, 0.5, 0.0, 2.0, 2.0]
    monkeypatch.setattr(planner, "band_deficits", lambda an, co: [0.0, 0.0, 0.0, 0.0, 0.0])
    sat, _ = planner.plan_saturate(a, c)
    assert sat.drive_db[3] == [0.0, 2.0]                     # the width drive stays
    assert max(d for p in sat.drive_db for d in p) <= planner.SAT_MAX_DRIVE_DB
    assert planner.plan_saturate(a, planner.Controls(preserve_mix=True))[0] is None


def test_band_deficits_are_measured_before_the_eq(pana):
    _, a = pana
    c = planner.Controls()
    d = planner.band_deficits(a, c)
    des = planner.inputeq_design(a, planner.resolve(a, c))
    nodes = np.asarray(des["nodes_hz"])
    res = np.asarray(des["residual_db"])
    delta = des["target_slope"] - des["measured_slope"]
    line = delta * np.log2(np.clip(nodes, 100.0, 10000.0) / 1000.0) - res
    xo = planner.resolve(a, c).crossovers_hz
    b2 = (nodes >= xo[1]) & (nodes < xo[2])
    assert d[2] == pytest.approx(float(np.mean(line[b2])))
    # a steeper target (more top-end deficit) raises the top band's deficit
    d2 = planner.band_deficits(a, planner.Controls(tilt_db=1.0))
    assert d2[4] > d[4] and d2[0] < d[0]
    rec = planner.plan_record(a, c)
    assert [r["deficit_db"] for r in rec["saturation"]["bands"]] == pytest.approx(d)


# ------------------------------------------------------------ loudness controls

def test_loudness_target_default():
    # the maintainer's meter (300 ms, AES17, loudest passage): his masters' median
    assert planner.DEFAULT_TARGET_RMS_DBFS == -4.1
    assert planner.Controls().rms_target == -4.1
    assert not hasattr(planner, "GENRE_PRESETS")   # retired (maintainer, 2026-10-20)
    with pytest.raises(ValueError):
        planner.Controls(target_rms=1.0)


@pytest.mark.slow
def test_planner_realise_solves_the_gain_for_the_target(pana):
    x, a = pana
    p = planner.plan_from_analysis(a, planner.Controls(target_rms=-14.0), x)
    lp = p.limiter
    assert lp.target_rms_dbfs == -14.0
    lr = lp.budget["limiter"]
    assert lr["requested_gain_db"] == pytest.approx(lp.requested_gain_db)
    assert lp.input_gain_db <= lr["max_gain_db"] + 1e-9


def test_bell_still_used_for_a_narrow_peak_in_a_plan():
    e = InputEqPlan(bell=EqBell(-2.0, 400.0, 2.0))
    assert e.band_shelf is None and "band_shelf" in planner.inputeq_sections(e)
