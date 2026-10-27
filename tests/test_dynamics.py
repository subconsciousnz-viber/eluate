"""Chunk 3 of docs/gain-staging.md: dynamics solved for their peak-GR budgets (3, 5, 7, 8).

The stage solves called without hits read peak GR as the maximum over the whole track
(per band and channel where that applies), the chunk 3 reading; the planner and the
solver read the average peak (maintainer, 2026-10-19; tests/test_average_peak.py).
Synthetic signals only."""

import numpy as np
import pytest

from fixtures.e_signals import short_planner_analysis
from eluate import planner, solver
from eluate.plan import LimiterPlan, MultibandPlan, Envelope, Plan
from eluate.stages import glue, limiter as lim, multiband as mb

FS = 44100
TOL = 0.1


def _noise(n, seed=0, ch=2):
    return np.random.default_rng(seed).standard_normal((n, ch))


def _dynamic(fs=FS, seconds=6.0, seed=1):
    """Noise with a slow level swell and a few short bursts: something to compress."""
    n = int(seconds * fs)
    t = np.arange(n) / fs
    x = 0.05 * _noise(n, seed) * (1.0 + 0.7 * np.sin(2 * np.pi * 0.3 * t))[:, None]
    for k, at in enumerate((1.0, 2.6, 4.1)):
        i = int(at * fs)
        x[i:i + int(0.03 * fs)] *= 4.0 + k
    return x


@pytest.fixture(scope="module")
def pana():
    return short_planner_analysis(FS)


def _meter(gain_db_values, rate=1000.0):
    """A GR curve's meter reading (envelope.stage_gr, one hit): the 300 ms average's peak."""
    from eluate import envelope
    return envelope.stage_gr(np.maximum(-np.asarray(gain_db_values, dtype=float), 0.0), rate,
                             None)["average_db"]


# ------------------------------------------------------------ solve functions

@pytest.mark.parametrize("target", [1.0, 2.0, 2.5, 3.0])
@pytest.mark.parametrize("knee", [0.0, 6.0])
def test_multiband_threshold_hits_peak_gr(target, knee):
    v = _dynamic()[:, 0]
    sol = mb.threshold_for_peak_gr(v, FS, target, 1.7, 10.0, 100.0, knee)
    assert sol["peak_gr_db"] == pytest.approx(target, abs=TOL)
    # the envelope the stage builds from that threshold reads that GR on the meter (300 ms)
    det = mb.detector(v, FS, mb.control_length(v.size, FS, 1000.0), 1000.0)
    env = mb.comp_envelope(det, sol["threshold_db"], 1.7, 10.0, 100.0, knee)
    assert _meter(env.values) == pytest.approx(target, abs=TOL)
    # the per-sample peak goes past the meter reading
    assert mb.comp_peak_gr_db(env) > target


def test_multiband_threshold_on_the_band_as_split():
    x = _dynamic()
    p0 = mb.design(x, FS, -40.0, ratio=1.5, attack_ms=5.0, release_ms=80.0, knee_db=3.0)
    _, info0 = mb.apply_with_info(x, FS, p0)
    from eluate import bands
    b = bands.split(bands.to_ms(x), FS, p0.crossovers_hz)
    thr = [[mb.threshold_for_peak_gr(b[k, :, c], FS, 2.5, 1.5, 5.0, 80.0, 3.0)["threshold_db"]
            for c in range(2)] for k in range(b.shape[0])]
    p = mb.design(x, FS, thr, ratio=1.5, attack_ms=5.0, release_ms=80.0, knee_db=3.0)
    _, info = mb.apply_with_info(x, FS, p)
    got = [[_meter(np.clip(p.gr_db[k][c].values, -mb.MULTIBAND_MAX_GR_DB, 0.0))
            for c in range(2)] for k in range(b.shape[0])]
    assert np.allclose(got, 2.5, atol=TOL)
    assert not np.allclose(info0["comp_gr_min_db"], info["comp_gr_min_db"])


@pytest.mark.parametrize("target", [0.5, 1.0, 3.0])
def test_multiband_ceiling_hits_peak_gr(target):
    v = _dynamic(seed=3)[:, 1]
    sol = mb.ceiling_for_peak_gr(v, FS, target, 2.0, 50.0)
    assert sol["peak_gr_db"] == pytest.approx(target, abs=TOL)
    assert mb.ceiling_for_peak_gr(v, FS, 0.0)["ceiling_db"] is None
    assert mb.ceiling_for_peak_gr(np.zeros(FS), FS, 1.0) == {"ceiling_db": None,
                                                           "peak_gr_db": 0.0}


def test_silent_band_threshold():
    assert mb.threshold_for_peak_gr(np.zeros(FS), FS, 2.5, 2.0, 10, 100) == \
        {"threshold_db": 0.0, "peak_gr_db": 0.0, "outlier_gr_db": 0.0}
    # the per-sample clamp (6 dB since D5, sprint 2026-10-21); the outlier cap is 4.5 on
    # the meter reading
    assert mb.MULTIBAND_MAX_GR_DB == 6.0 and mb.MULTIBAND_OUTLIER_CAP_DB == 4.5
    with pytest.raises(ValueError):
        mb.threshold_for_peak_gr(np.ones(FS), FS, 6.1, 2.0, 10, 100)


@pytest.mark.parametrize("target", [1.0, 1.5, 2.0, 3.0, 4.0])
def test_glue_threshold_hits_peak_gr(target):
    x = _dynamic(seed=5)
    sol = glue.threshold_for_peak_gr(x, FS, target, 3.0, 6.0, 30.0, 300.0)
    assert sol["peak_gr_db"] == pytest.approx(target, abs=TOL)
    p = glue.design(x, FS, sol["threshold_db"], ratio=3.0, knee_db=6.0, attack_ms=30.0,
                    release_ms=300.0, mix=1.0)
    assert _meter(p.gain_db.values) == pytest.approx(target, abs=TOL)


@pytest.mark.parametrize("target,knee", [(0.5, 1.0), (1.0, 1.0), (1.0, 1.5), (1.0, 0.0),
                                         (1.5, 0.5)])
def test_clipper_threshold_hits_peak_gr(target, knee):
    x = _dynamic(seed=7)
    sol = lim.clip_threshold_for_peak_gr(x, target, knee)
    assert sol["peak_gr_db"] == pytest.approx(target, abs=TOL)
    plan = LimiterPlan(ceiling_dbtp=-1.0, input_gain_db=4.0, clip_threshold_db=sol["threshold_db"],
                       clip_knee_db=knee)
    _, info = lim.apply_with_info(x, FS, plan)
    # the reduction does not move with the drive: the threshold sits before the gain
    assert info["clip_max_reduction_db"] == pytest.approx(target, abs=TOL)
    with pytest.raises(ValueError):
        lim.clip_threshold_for_peak_gr(x, 1.6)


def test_limiter_input_gain_holds_its_budget():
    x = _dynamic(seed=9)
    clip = lim.clip_threshold_for_peak_gr(x, 1.0, 1.0)["threshold_db"]
    lp = LimiterPlan(ceiling_dbtp=-0.3, input_gain_db=0.0, clip_threshold_db=clip,
                     max_gr_db=3.0)
    b = solver.limiter_input_gain(x, FS, lp, requested_db=30.0)
    assert b["planned_gr_db"] == pytest.approx(3.0, abs=1e-9)
    assert b["warnings"] and b["warnings"][0]["stage"] == "limiter"
    assert b["warnings"][0]["needed_db"] > 3.0
    y, info = solver.limit_blocked(x, FS, replace_gain(lp, b["input_gain_db"]))
    assert info["max_gain_reduction_db"] == pytest.approx(3.0, abs=TOL)
    assert info["max_gain_reduction_db"] <= 3.0 + 1e-6
    # a request inside the budget is kept, with no warning
    b2 = solver.limiter_input_gain(x, FS, lp, requested_db=b["max_gain_db"] - 2.0)
    assert b2["input_gain_db"] == b["max_gain_db"] - 2.0 and not b2["warnings"]
    assert b2["planned_gr_db"] == pytest.approx(1.0, abs=1e-6)


def replace_gain(lp, g):
    from dataclasses import replace
    return replace(lp, input_gain_db=g)


def test_clamp_limiter_gain_is_the_chunk4_hook():
    lp = LimiterPlan(ceiling_dbtp=-1.0, max_gr_db=3.0,
                     budget={"limiter": {"clipped_peak_dbfs": -10.0}})
    out = planner.clamp_limiter_gain(lp, 20.0)
    # S5: the request is capped at the maximum gain + 6 dB, the raw request recorded
    assert out.input_gain_db == pytest.approx(12.0)
    assert out.requested_gain_db == pytest.approx(12.0 + planner.LIMITER_REQUEST_OVER_DB)
    w = out.budget["limiter"]["warnings"][0]
    assert w["cap_db"] == 3.0 and w["request_capped"] and w["requested_uncapped_db"] == 20.0
    assert w["needed_db"] == pytest.approx(3.0 + planner.LIMITER_REQUEST_OVER_DB)
    ok = planner.clamp_limiter_gain(lp, 5.0)
    assert ok.input_gain_db == 5.0 and ok.budget["limiter"]["warnings"] == []
    assert ok.budget["limiter"]["planned_gr_db"] == 0.0


# ------------------------------------------------------------ ratio map

def test_ratio_map_endpoints():
    for k, (p10, p90) in enumerate(zip(mb.CREST_P10_DB, mb.CREST_P90_DB)):
        assert mb.ratio_for_crest(p10, k) == pytest.approx(2.0)
        assert mb.ratio_for_crest(p90, k) == pytest.approx(1.3)
        assert mb.ratio_for_crest(p10 - 5, k) == 2.0 and mb.ratio_for_crest(p90 + 5, k) == 1.3
        assert mb.ratio_for_crest(0.5 * (p10 + p90), k) == pytest.approx(1.65)
    assert mb.CREST_P10_DB == (8.5, 13.5, 15.7, 19.8, 20.6)
    assert mb.CREST_P90_DB == (12.0, 17.4, 20.4, 25.4, 27.5)
    assert mb.ratio_for_crest(30.0, 9) == 1.3                # extra bands use the last row
    # the S table (maintainer, 2026-10-18)
    assert mb.CREST_P10_S_DB == (11.1, 12.4, 13.4, 16.7, 21.8)
    assert mb.CREST_P90_S_DB == (17.0, 17.4, 19.4, 26.5, 32.3)
    for k, (p10, p90) in enumerate(zip(mb.CREST_P10_S_DB, mb.CREST_P90_S_DB)):
        assert mb.ratio_for_crest(p10, k, 1) == pytest.approx(2.0)
        assert mb.ratio_for_crest(p90, k, 1) == pytest.approx(1.3)
        assert mb.ratio_for_crest(0.5 * (p10 + p90), k, 1) == pytest.approx(1.65)
    # band 4 at 27.5 dB: 1.3:1 on the M table, between the rows on the S table
    assert mb.ratio_for_crest(27.5, 4, 0) == pytest.approx(1.3)
    assert mb.ratio_for_crest(27.5, 4, 1) == pytest.approx(2.0 - 0.7 * (27.5 - 21.8) / 10.5)
    assert mb.ratio_for_crest(float("nan"), 0) == 2.0


def test_band_crest_metric():
    # a full-scale sine: p99 frame peak 0 dBFS, 1 s RMS -3.01 dBFS
    t = np.arange(3 * FS) / FS
    assert mb.band_crest_db(np.sin(2 * np.pi * 997 * t), FS) == pytest.approx(3.01, abs=0.01)
    assert np.isnan(mb.band_crest_db(np.zeros(FS), FS))


# ------------------------------------------------------------ makeup modes

def test_makeup_modes_from_a_gr_curve():
    gr = np.zeros(10000)
    gr[2000:3000] = -2.0
    gr[2500] = -3.0
    gate = np.zeros(10000, bool)
    gate[1000:5000] = True
    # PEAK makeup: + the GR's meter reading (the 300 ms average's peak), not its spike
    pk = mb.mode_makeup(gr, gate, "peak").values.tolist()
    assert pk == [pytest.approx(-gr[2000:3000].mean(), abs=0.01)]
    rms = mb.mode_makeup(gr, gate, "rms").values
    assert rms.tolist() == [pytest.approx(-gr[gate].mean())]
    auto = mb.mode_makeup(gr, gate, "auto").values
    assert auto.size == gr.size and auto.max() <= 3.0
    assert auto[2500] == pytest.approx(-gr[1000:4000].mean(), abs=1e-6)   # centred 3 s mean
    assert auto[9000] == 0.0
    assert mb.mode_makeup(np.zeros(5), np.ones(5, bool), "auto").values.tolist() == [0.0]
    with pytest.raises(ValueError):
        mb.mode_makeup(gr, gate, "loud")


def _budget_plan(makeup="peak", xo=(400.0,)):
    nb = len(xo) + 1
    zero = [[Envelope.constant(0.0)] * 2 for _ in range(nb)]
    return MultibandPlan(gr_db=zero, crossovers_hz=list(xo), threshold_db=[[0.0, 0.0]] * nb,
                         ratio=2.0, attack_ms=[[10.0, 10.0]] * nb,
                         release_ms=[[100.0, 100.0]] * nb, knee_db=6.0,
                         budget=mb.default_budget(makeup))


@pytest.mark.parametrize("mode", ["peak", "rms", "auto"])
def test_makeup_modes_in_the_stage(mode):
    x = _dynamic(seed=11)
    p = mb.design_budget(x, FS, _budget_plan(mode))
    for k in range(2):
        for c in range(2):
            r = p.budget["result"][k][c]
            env = p.mode_makeup_db[k][c]
            if mode == "peak":
                assert env.values.tolist() == [pytest.approx(r["total_db"])]
            elif mode == "rms":
                assert env.values.size == 1 and 0.0 < env.values[0] < r["total_db"]
            else:
                assert env.values.size > 1 and env.values.max() < r["total_db"]
    _, info = mb.apply_with_info(x, FS, p)
    if mode == "peak":          # +peak GR on the meter reading: less than the deepest spike
        assert np.all(info["makeup_db"] <= -info["total_gr_min_db"] + 1e-6)


# ------------------------------------------------------------ multiband budget

def test_design_budget_hits_the_normal_tally():
    x = _dynamic(seed=13)
    p = mb.design_budget(x, FS, _budget_plan())
    _, info = mb.apply_with_info(x, FS, p)
    for k in range(2):
        for c in range(2):
            r = p.budget["result"][k][c]
            assert r["comp_db"] == pytest.approx(2.5, abs=TOL) or r["exception"]
            assert r["total_db"] <= r["budget_db"] + 1e-6
            g = np.clip(p.gr_db[k][c].values, -mb.MULTIBAND_MAX_GR_DB, 0.0)
            assert _meter(g) == pytest.approx(r["comp_db"], abs=1e-6)
            assert -info["comp_gr_min_db"][k, c] >= r["comp_db"] - 1e-6
            assert 1.3 <= r["ratio"] <= 2.0
    assert p.budget["warnings"] == []
    # design_from re-solves on the audio it is given (the solver's Chain)
    q = mb.design_from(p, 0.5 * x, FS)
    flat = lambda pl, k: [r[k] for row in pl.budget["result"] for r in row]  # noqa: E731
    assert flat(q, "comp_db") == pytest.approx(flat(p, "comp_db"), abs=TOL)
    # solved on the audio entering the stage: 6 dB quieter, thresholds 6 dB lower
    assert np.allclose(np.subtract(flat(p, "threshold_db"), flat(q, "threshold_db")),
                       20 * np.log10(2.0), atol=0.01)


def _spiky(fs, seconds, left_only):
    """Steady noise plus one huge 1 kHz burst: band 2 needs far more than 6 dB of peak GR."""
    n = int(seconds * fs)
    x = 0.02 * _noise(n, 21)
    t = np.arange(int(0.03 * fs)) / fs
    i = n // 2
    burst = 0.9 * np.sin(2 * np.pi * 1000.0 * t)
    x[i:i + t.size, 0] += burst
    if not left_only:
        x[i:i + t.size, 1] += burst
    return x


@pytest.mark.parametrize("left_only,channel", [(False, "M"), (True, "both")])
def test_warning_when_a_band_is_forced_over_its_cap(left_only, channel):
    fs = 16000
    x = _spiky(fs, 40.0, left_only)
    p = mb.design_budget(x, fs, _budget_plan(xo=(100.0, 400.0, 1600.0, 6400.0)))
    w = [w for w in p.budget["warnings"] if w["band"] == 2]
    assert len(w) == 1
    assert w[0] == {"stage": "multiband", "band": 2, "channel": channel,
                    "needed_db": pytest.approx(w[0]["needed_db"]), "cap_db": 6.0}
    assert w[0]["needed_db"] > 6.0
    r = p.budget["result"][2][0]
    assert r["exception"] and r["budget_db"] == 6.0
    # rendered at the caps: the compressor at 3 dB on the meter reading, the band limiter
    # at its 3 dB per-sample peak (an outlier catcher: its spike barely moves the meter)
    assert r["comp_db"] == pytest.approx(3.0, abs=TOL)
    assert 0.0 < r["limit_outlier_db"] <= 3.0 + 0.05
    assert r["total_db"] <= 6.0 + 1e-3   # the solve tolerance


# ------------------------------------------------------------ glue

def test_glue_serial_and_parallel_give_the_same_net_peak_gr():
    x = _dynamic(seed=17)
    s = glue.design_budget(x, FS, "serial")
    p = glue.design_budget(x, FS, "parallel")
    assert s.mix == 1.0 and 0.0 < p.mix < 1.0
    # on the meter reading: both nets at 1.5 dB, the parallel wet path at 3 dB
    assert s.budget["net_db"] == pytest.approx(1.5, abs=TOL)
    assert p.budget["net_db"] == pytest.approx(1.5, abs=TOL)
    assert p.budget["wet_db"] == pytest.approx(3.0, abs=TOL)
    assert s.ratio == 3.0 and p.ratio == 3.0
    assert s.budget["warnings"] == [] and p.budget["warnings"] == []
    # the output's GR as the stage applies it, read on the meter
    for g in (s, p):
        wet = 10.0 ** (g.gain_db.values / 20.0)
        net = 1.0 - g.mix + g.mix * wet
        assert _meter(20 * np.log10(net)) == pytest.approx(1.5, abs=TOL)
        assert -20 * np.log10(net.min()) >= 1.5 - 1e-6


def test_glue_warns_when_the_target_cannot_be_met():
    g = glue.design_budget(np.zeros((FS, 2)), FS, "serial")
    assert g.budget["warnings"] and g.budget["warnings"][0]["stage"] == "glue"
    assert g.budget["warnings"][0]["achieved_db"] == 0.0
    with pytest.raises(ValueError):
        glue.design_budget(np.zeros((FS, 2)), FS, "serial", net_db=2.5)
    assert glue.mix_for_net(3.0, 1.5) == pytest.approx(
        (1 - 10 ** (-1.5 / 20)) / (1 - 10 ** (-3 / 20)))


# ------------------------------------------------------------ clipper

def _kicks_and_snare(fs=FS):
    """Kicks (decaying 55 Hz bursts) every 0.5 s and one isolated snare-like noise burst,
    the loudest peak, between two kicks."""
    n = int(3.0 * fs)
    x = np.zeros((n, 2))
    onsets = [0.25 + 0.5 * k for k in range(6)]
    tk = np.arange(int(0.15 * fs)) / fs
    kick = 0.5 * np.sin(2 * np.pi * 55 * tk) * np.exp(-tk / 0.05)
    for o in onsets:
        i = int(o * fs)
        x[i:i + tk.size] += kick[:, None]
    snare_at = 1.5
    ts = np.arange(int(0.08 * fs)) / fs
    sn = 0.9 * np.random.default_rng(4).standard_normal(ts.size) * np.exp(-ts / 0.02)
    i = int(snare_at * fs)
    x[i:i + ts.size] += sn[:, None]
    return x, onsets, snare_at


def test_whole_signal_clipper_acts_on_the_snare():
    x, onsets, snare_at = _kicks_and_snare()
    sol = lim.clip_threshold_for_peak_gr(x, 1.0, 1.0)
    plan = LimiterPlan(ceiling_dbtp=-1.0, clip_threshold_db=sol["threshold_db"], clip_knee_db=1.0)
    u = lim.resample.upsample(x, lim.OVERSAMPLE)
    uc, red, spans = lim.clip_whole(u, plan)
    assert 0.0 < red <= 1.5 + 1e-9 and red == pytest.approx(1.0, abs=TOL)
    fs8 = FS * lim.OVERSAMPLE
    t_spans = [(a0 / fs8, a1 / fs8) for a0, a1 in spans]
    assert any(a <= snare_at + 0.08 and b >= snare_at for a, b in t_spans)
    assert not all(any(abs(a - o) < 0.012 for o in onsets) for a, _ in t_spans)
    # the output's peak reduction (after the band-limit) is at most 1.5 dB
    ub = lim.band_limit_change(u, uc, spans)
    drop = 20 * np.log10(np.max(np.abs(u)) / np.max(np.abs(ub)))
    assert 0.0 < drop <= 1.5
    # outside every span the 8x signal is untouched
    mask = np.ones(u.shape[0], bool)
    for a0, a1 in spans:
        mask[a0:a1] = False
    assert np.array_equal(uc[mask], u[mask])
    # through the limiter stage, and the blocked limiter agrees
    _, info = lim.apply_with_info(x, FS, plan)
    assert info["clip_max_reduction_db"] == pytest.approx(1.0, abs=TOL)
    _, ib = solver.limit_blocked(x, FS, plan, block=8192)
    assert ib["clip_max_reduction_db"] == pytest.approx(info["clip_max_reduction_db"], abs=1e-9)
    assert ib["clip_spans"] == info["clip_spans"]


def test_kick_detective_keeps_the_onset_mask():
    x, onsets, snare_at = _kicks_and_snare()
    plan = LimiterPlan(ceiling_dbtp=-12.0, clip_depth_db=1.5, clip_onsets_s=onsets)
    u = lim.resample.upsample(x, lim.OVERSAMPLE)
    uc, red, spans = lim.clip_onsets(u, FS * lim.OVERSAMPLE, plan, 10 ** (-12 / 20))
    fs8 = FS * lim.OVERSAMPLE
    i0, i1 = int(snare_at * fs8), int((snare_at + 0.08) * fs8)
    assert red > 0 and spans
    assert np.array_equal(uc[i0:i1], u[i0:i1])            # the snare is not clipped
    with pytest.raises(ValueError):
        LimiterPlan(clip_depth_db=1.5, clip_onsets_s=onsets, clip_threshold_db=-3.0)


@pytest.mark.parametrize("fs", [44100, 48000])
@pytest.mark.slow
def test_blocked_whole_clipper_matches_whole_track(fs):
    x, _, _ = _kicks_and_snare(fs)
    x = np.resize(x, (int(2.2 * x.shape[0]), 2))
    sol = lim.clip_threshold_for_peak_gr(x, 1.0, 0.5)
    plan = LimiterPlan(ceiling_dbtp=-1.0, input_gain_db=6.0,
                       clip_threshold_db=sol["threshold_db"], clip_knee_db=0.5)
    y0, i0 = lim.apply_with_info(x, fs, plan)
    for blk in (4096, 20000, 1 << 17):
        y1, i1 = solver.limit_blocked(x, fs, plan, block=blk)
        assert np.max(np.abs(y1 - y0)) < 1e-9
        for k in ("max_gain_reduction_db", "clip_max_reduction_db", "clip_spans",
                  "detector_peak_dbfs"):
            assert i1[k] == pytest.approx(i0[k], abs=1e-8), k
        m = solver.limit_blocked(x, fs, plan, block=blk, measure_only=True)
        assert m["detector_peak_dbfs"] == pytest.approx(i0["detector_peak_dbfs"], abs=1e-9)


# ------------------------------------------------------------ character presets

@pytest.mark.parametrize("bpm,name", [(60.0, "slow"), (69.9, "slow"), (70.0, "moderate"),
                                      (120.0, "moderate"), (140.0, "moderate"),
                                      (140.1, "fast"), (174.0, "fast"), (None, "moderate")])
def test_auto_character_by_tempo(bpm, name):
    assert planner.character_choice("auto", bpm)[0] == name
    assert planner.character_choice("aggressive", bpm)[0] == "aggressive"


@pytest.mark.slow
def test_preset_values(pana):
    _, a = pana
    t = planner.CHARACTER_PRESETS
    assert t["transparent"]["mb_knee_db"] == 10.0 and t["aggressive"]["clip_knee_db"] == 0.5
    # maintainer, 2026-10-23 (quality review 3): no hard clipper knee in any preset
    assert all(t[n]["clip_knee_db"] >= 0.5 for n in planner.CHARACTERS[1:])
    # maintainer, 2026-10-23 (quality review 1): band ceilings' lookahead per band, low to high
    look = {"transparent": (20.0, 10.0, 5.0, 2.0, 2.0), "slow": (20.0, 8.0, 4.0, 2.0, 2.0),
            "moderate": (15.0, 6.0, 3.0, 2.0, 2.0), "fast": (10.0, 5.0, 2.0, 2.0, 2.0),
            "aggressive": (6.0, 4.0, 2.0, 2.0, 2.0)}
    assert set(look) == set(planner.CHARACTERS[1:]) == set(t)
    for name, v in look.items():
        assert t[name]["band_lookahead_ms"] == v and len(t[name]["band_lookahead_ms"]) == 5
    assert t["aggressive"]["mb_knee_db"] == 1.0           # maintainer, 2026-10-18: not hard
    assert t["moderate"]["mb_attack_ms"] == (20.0, 15.0, 10.0, 5.0, 3.0)
    assert t["fast"]["mb_release_ms"] == (120.0, 100.0, 60.0, 40.0, 30.0)
    # FET-style glue (1176 attack range 0.02 to 0.8 ms; maintainer, 2026-10-19)
    assert (t["slow"]["glue_attack_ms"], t["slow"]["glue_release_ms"]) == (0.8, 400.0)
    assert planner.GLUE_RATIO == 2.0
    assert (t["fast"]["limiter_attack_ms"], t["fast"]["limiter_release_ms"]) == (1.0, 50.0)
    for name in planner.CHARACTERS[1:]:
        c = planner.character_settings(a, planner.Controls(character=name))
        pr = t[name]
        assert c["preset"] == name and c["mb_release_ms"][0] >= 25.0
        assert c["mb_attack_ms"] == list(pr["mb_attack_ms"])
        p = planner.plan_from_analysis(a, planner.Controls(character=name))
        assert p.multiband.attack_ms == [[v, v] for v in pr["mb_attack_ms"]]
        assert p.multiband.knee_db == pr["mb_knee_db"]
        assert (p.glue.attack_ms, p.glue.release_ms) == (pr["glue_attack_ms"],
                                                         pr["glue_release_ms"])
        assert p.limiter.clip_knee_db == pr["clip_knee_db"]
        nb = len(p.multiband.crossovers_hz) + 1
        want = planner._per_band(pr["band_lookahead_ms"], nb)
        assert c["band_lookahead_ms"] == want == c["speclim_lookahead_ms"]
        assert p.multiband.ceiling_lookahead_ms == want == p.multiband.lookahead_ms()
        assert p.limiter.speclim_lookahead_ms == want             # electronic: the character's
        q = Plan.from_json(p.to_json())
        assert q.multiband.ceiling_lookahead_ms == want
        assert q.limiter.speclim_lookahead_ms == want
        assert planner.plan_record(a, planner.Controls(character=name))["multiband"][
            "ceiling_lookahead_ms"] == want
        assert (p.limiter.attack_ms, p.limiter.release_ms) == (
            pr["limiter_attack_ms"], max(pr["limiter_release_ms"], planner.LIMITER_RELEASE_MIN_MS))


def test_low_band_release_floor():
    c = planner.character_settings({"plan_inputs": {}, "meta": {"crossovers": [100.0]}},
                                   planner.Controls(character="aggressive"), nb=5)
    assert c["mb_release_ms"][0] == 80.0
    pr = dict(planner.CHARACTER_PRESETS)
    try:
        planner.CHARACTER_PRESETS["aggressive"] = dict(pr["aggressive"],
                                                       mb_release_ms=(10.0, 10.0))
        c = planner.character_settings({"plan_inputs": {}, "meta": {"crossovers": [100.0]}},
                                       planner.Controls(character="aggressive"), nb=2)
        assert c["mb_release_ms"] == [25.0, 10.0]
    finally:
        planner.CHARACTER_PRESETS.clear()
        planner.CHARACTER_PRESETS.update(pr)


def test_auto_reads_the_users_bpm_first(pana):
    _, a = pana
    c = planner.character_settings(a, planner.Controls(bpm=150.0))
    assert c["preset"] == "fast" and c["bpm_source"] == "user"
    c = planner.character_settings(a, planner.Controls())
    assert c["bpm_source"] == "analysis" and c["bpm"] == a["plan_inputs"]["tempo_bpm"]
    assert c["preset"] == planner.character_choice("auto", c["bpm"])[0]
    with pytest.raises(ValueError):
        planner.Controls(character="punchy")


# ------------------------------------------------------------ planner and record

@pytest.mark.slow
def test_planner_chain_order_and_budget_record(pana):
    x, a = pana
    p = planner.plan_from_analysis(a, planner.Controls(character="moderate"), x)
    r = planner.budget_report(p)
    st = r["stages"]
    assert set(st) == {"multiband", "glue", "clipper", "limiter"}
    for row in st["multiband"]["bands"]:
        for b in row:
            assert b["total_db"] <= b["budget_db"] + 1e-6
    assert st["glue"]["net_db"] == pytest.approx(1.5, abs=TOL) and st["glue"]["mode"] == "serial"
    # the average output-peak fall hits 1 dB; the per-sample cap (3 dB) holds the outlier,
    # so the average is never lowered to make room (dynamics-macros step 1)
    cl = st["clipper"]
    assert cl["outlier_cap_db"] == planner.CLIP_SAMPLE_CAP_DB
    assert cl["outlier_db"] <= cl["outlier_cap_db"] + 0.05
    assert cl["planned_db"] == pytest.approx(1.0, abs=TOL)
    assert cl["planned_db"] <= cl["budget_db"] == 1.5
    assert st["limiter"]["planned_db"] <= 3.0 + 1e-9 and st["limiter"]["budget_db"] == 3.0
    assert st["limiter"]["outlier_db"] <= planner.LIMITER_OUTLIER_CAP_DB + 1e-9
    assert r["reading"] == "average" and p.hits is not None
    assert p.limiter.clip_threshold_db is not None and p.limiter.clip_depth_db == 0.0
    # the whole plan survives the JSON round trip
    from eluate.plan import Plan
    q = Plan.from_json(p.to_json())
    assert q.to_dict() == p.to_dict()
    rec = planner.plan_record(a, planner.Controls(character="moderate"))
    assert rec["character"]["preset"] in planner.CHARACTER_PRESETS
    assert rec["clipper"]["mode"] == "whole signal"


@pytest.mark.slow
def test_kick_detective_control_restores_the_onset_clipper(pana):
    x, a = pana
    p = planner.plan_from_analysis(a, planner.Controls(kick_detective=True), x)
    lp = p.limiter
    assert lp.clip_threshold_db is None and lp.clip_target_db is None
    assert lp.clip_onsets_s and lp.clip_depth_db == planner.CLIP_DEPTH_DB
    assert planner.plan_record(a, planner.Controls(kick_detective=True))["clipper"]["mode"] \
        == "kick detective"


@pytest.mark.slow
def test_parallel_glue_and_rms_makeup_controls(pana):
    x, a = pana
    p = planner.plan_from_analysis(a, planner.Controls(glue_mode="parallel", makeup="rms"), x)
    assert p.glue.budget["mode"] == "parallel" and p.glue.mix < 1.0
    assert p.glue.budget["net_db"] == pytest.approx(1.5, abs=TOL)
    assert glue.net_peak_gr_db(p.glue) <= glue.OUTLIER_CAP_DB + 1e-6
    assert p.multiband.budget["makeup"] == "rms"


@pytest.mark.slow
def test_solver_keeps_every_budget(pana):
    x, a = pana
    c = planner.Controls()
    res = solver.solve(x, FS, planner.plan_from_analysis(a, c), planner.targets_from(a, c),
                       max_outer=2)
    b = res.metrics["budget"]
    # rendered: the outlier, within the 5 dB cap; planned: the average, within 3 dB
    assert b["stages"]["limiter"]["rendered_db"] <= planner.LIMITER_OUTLIER_CAP_DB + 0.05
    assert b["stages"]["limiter"]["planned_db"] <= 3.0 + 1e-9
    assert b["stages"]["clipper"]["rendered_db"] <= planner.CLIP_SAMPLE_CAP_DB + 0.05
    for row in b["stages"]["multiband"]["bands"]:
        for r in row:
            assert r["total_db"] <= r["budget_db"] + 1e-6
    assert res.summary()["reported"]["budget"] == b
    # the requested drive is recorded; a clamp leaves a warning naming the limiter
    lp = res.plan.limiter
    assert lp.requested_gain_db is not None and lp.input_gain_db <= lp.requested_gain_db + 1e-9
    if lp.requested_gain_db > lp.input_gain_db + 1e-6:
        assert any(w["stage"] == "limiter" for w in b["warnings"])


# ------------------------------------------------------------ split clipper (2026-10-20)

def _sub_and_clicks(fs=FS, seconds=4.0):
    """A 50 Hz sine at -2 dBFS with a 3 ms click at 3 kHz every half second."""
    n = int(seconds * fs)
    t = np.arange(n) / fs
    x = 0.79 * np.sin(2 * np.pi * 50.0 * t)
    m = int(0.003 * fs)
    click = 0.6 * np.sin(2 * np.pi * 3000.0 * np.arange(m) / fs) * np.hanning(m)
    for k in range(1, int(seconds / 0.5)):
        i = int(k * 0.5 * fs)
        x[i:i + m] += click
    return np.stack([x, 0.9 * x], axis=1)


def test_split_clipper_clips_the_clicks_and_leaves_the_sub_alone():
    from dataclasses import replace
    x = _sub_and_clicks()
    full = LimiterPlan(ceiling_dbtp=-0.3, clip_threshold_db=-6.0, clip_knee_db=1.0)
    split = replace(full, clip_split_hz=200.0)
    u = lim.resample.upsample(x, lim.OVERSAMPLE)
    u_hi = lim.resample.upsample(lim.clip_hi_part(x, FS, 200.0), lim.OVERSAMPLE)
    uc_f, _, spans_f = lim.clip_whole(u, full)
    uc_s, red_s, spans_s = lim.clip_whole(u, split, u_hi)
    cover = lambda sp: sum(b - a for a, b in sp) / u.shape[0]
    assert cover(spans_f) > 0.9          # the full-band clipper sits on the sine
    assert 0 < cover(spans_s) < 0.05     # the split clipper touches the clicks only
    assert len(spans_s) == 7 and red_s > 0.5
    # the change is the high part's: below 100 Hz it is more than 60 dB under the sine
    d = lim.resample.decimate(uc_s - u, lim.OVERSAMPLE)
    low = lim.bands.split(d, FS, [100.0], phase="zero")[0]
    assert 20 * np.log10(np.sqrt(np.mean(low ** 2)) / np.sqrt(np.mean(x ** 2))) < -60
    # outside every span the signal is untouched
    mask = np.ones(u.shape[0], bool)
    for a0, a1 in spans_s:
        mask[a0:a1] = False
    assert np.array_equal(uc_s[mask], u[mask])
    # the stage and the blocked limiter agree
    y0, i0 = lim.apply_with_info(x, FS, split)
    y1, i1 = solver.limit_blocked(x, FS, split, block=8192)
    assert np.max(np.abs(y0 - y1)) < 1e-9
    assert i1["clip_spans"] == i0["clip_spans"] == 7
    assert i1["clip_max_reduction_db"] == pytest.approx(i0["clip_max_reduction_db"], abs=1e-9)
    # a plan without the split is the old clipper, bit for bit
    uc_old, _, _ = lim.clip_whole(u, full, None)
    assert np.array_equal(uc_old, uc_f)
    with pytest.raises(ValueError, match="clip_split_hz"):
        LimiterPlan(clip_split_hz=0.0)
