"""Chunk 1 of docs/gain-staging.md: key-locked x4 crossovers (4), the 1 s RMS meter
(3, 9), input normalisation (2 item 1) and the ceiling / export format (3); with
chunk 2, normalisation followed by the input EQ (2 items 1 and 2)."""

import json
import math

import numpy as np
import pytest
import soundfile as sf

from fixtures.e_signals import short_planner_analysis
from eluate import bands, cli, features, io, meter, planner, solver
from eluate.plan import (EqBell, EqHighPass, EqHighShelf, EqLowShelf, InputEqPlan, InputPlan,
                            LimiterPlan, OutputPlan, Plan)
from eluate.stages import gain as gain_stage
from eluate.stages import inputeq as eq_stage
from eluate.stages import output as output_stage

FS = 44100
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _sine(f, amp, dur, fs, ch=2):
    t = np.arange(int(round(dur * fs))) / fs
    s = amp * np.sin(2 * np.pi * f * t)
    return np.repeat(s[:, None], ch, axis=1)


# ------------------------------------------------------------ crossovers (4)

def test_default_crossovers_are_x4():
    assert bands.DEFAULT_CROSSOVERS == (100.0, 400.0, 1600.0, 6400.0)
    assert planner.Controls().crossovers_hz is None           # the analysis's (key-locked)
    assert planner.Targets().crossovers_hz == [100.0, 400.0, 1600.0, 6400.0]


@pytest.mark.parametrize("tonic, p, xo", [
    ("C#", 138.591, (110.0, 440.0, 1760.0, 7040.0)),
    ("D#", 155.563, (123.471, 493.883, 1975.53, 7902.13)),
])
def test_key_crossovers_examples(tonic, p, xo):
    assert planner.tonic_partial_hz(tonic) == pytest.approx(p, abs=1e-3)
    assert planner.key_crossovers(tonic) == pytest.approx(xo, rel=1e-5)


@pytest.mark.parametrize("pc", range(12))
def test_key_crossovers_every_pitch_class(pc):
    name = NAMES[pc]
    p = planner.tonic_partial_hz(name)
    assert 80.0 <= p < 160.0
    # the lowest one at or above 80 Hz, of the right pitch class
    assert p / 2 < 80.0
    assert planner.cents_to_pitch_class(p, pc) < 1e-6
    xo = planner.key_crossovers(name)
    assert 80.0 <= xo[0] <= 160.0
    assert [b / a for a, b in zip(xo, xo[1:])] == pytest.approx([4.0, 4.0, 4.0], rel=1e-12)
    # p is the first tonic partial in band 2 (above fc1, below fc2), a third of an
    # octave above fc1 unless the 80 Hz clip binds
    assert xo[0] < p < xo[1]
    if xo[0] > 80.0:
        assert p / xo[0] == pytest.approx(2 ** (1 / 3), rel=1e-12)
    else:
        assert p / 2 ** (1 / 3) < 80.0
    assert xo[3] < FS / 2


def test_key_crossovers_names_and_fallback():
    assert planner.key_crossovers("Db") == planner.key_crossovers("C#")
    assert planner.key_crossovers("Ebm") == planner.key_crossovers("D#")
    assert planner.key_crossovers(None) == bands.DEFAULT_CROSSOVERS
    assert planner.key_crossovers("H") == bands.DEFAULT_CROSSOVERS
    base = (100.0, 500.0, 2000.0, 8000.0)
    assert planner.key_crossovers("C#", base) == pytest.approx((110.0, 550.0, 2200.0, 8800.0),
                                                               rel=1e-5)
    # the clip: E2 = 82.4 Hz would put fc1 at 65.4 Hz
    assert planner.key_crossovers("E")[0] == 80.0


@pytest.fixture(scope="module")
def pana():
    return short_planner_analysis(FS)


def test_analysis_crossovers_follow_the_key(pana):
    x, a = pana
    m = a["meta"]
    assert m["crossover_rule"] == "key"
    assert m["crossover_tonic"] == a["pitch"]["key"]["tonic"] is not None
    assert tuple(m["crossovers"]) == planner.key_crossovers(m["crossover_tonic"])
    assert np.asarray(a["crest"]["p95_db"]).shape[0] == 5
    c = planner.Controls()
    plan = planner.plan_from_analysis(a, c)
    assert plan.multiband.crossovers_hz == list(m["crossovers"])
    assert plan.saturate.crossovers_hz == list(m["crossovers"])
    assert planner.targets_from(a, c).crossovers_hz == list(m["crossovers"])
    rec = planner.plan_record(a, c)["crossovers"]
    assert rec["rule"] == "key" and rec["hz"] == list(m["crossovers"])
    assert rec["tonic"] == m["crossover_tonic"] and 80.0 <= rec["tonic_partial_hz"] < 160.0
    assert "key-locked" in rec["reason"]
    # given crossovers must match the analysis's
    with pytest.raises(ValueError):
        planner.plan_from_analysis(a, planner.Controls(crossovers_hz=bands.DEFAULT_CROSSOVERS))


@pytest.mark.slow
def test_key_locked_analysis_equals_a_fixed_one():
    """Finding the key before the first split changes nothing but the crossovers: the
    analysis equals a fixed-crossover analysis at the same crossovers."""
    x, a = short_planner_analysis(FS)
    xn = np.asarray(x, dtype=np.float64) * 10 ** (a["meta"]["input"]["gain_db"] / 20)
    b = features.analyse(xn, FS, fcs=a["meta"]["crossovers"])
    assert b["meta"]["crossover_rule"] == "fixed"
    for k in ("crest", "stereo", "tilt"):
        for kk, v in a[k].items():
            if isinstance(v, np.ndarray):
                assert np.array_equal(v, b[k][kk], equal_nan=True), (k, kk)
    assert a["pitch"]["key"] == b["pitch"]["key"]
    assert np.array_equal(a["plan_inputs"]["kick_onsets"], b["plan_inputs"]["kick_onsets"])


@pytest.mark.slow
def test_cache_reuse_with_key_crossovers(tmp_path):
    x, a = short_planner_analysis(FS)
    wav = tmp_path / "in.wav"
    io.write(wav, x, FS)
    work = tmp_path / "w"
    cli.do_analyse(wav, work, None, log=lambda m: None)
    s = json.loads((work / "source.json").read_text(encoding="utf-8"))
    assert s["crossovers_requested"] is None and s["crossover_rule"] == "key"
    assert s["crossovers_hz"] == pytest.approx(list(a["meta"]["crossovers"]))
    assert s["r_ref_dbfs"] == planner.R_REF_DBFS
    # a request without --xover reuses it whatever crossovers the key gave
    assert cli._matches(work / "source.json", wav)
    assert not cli._matches(work / "source.json", wav, crossovers=s["crossovers_hz"])
    assert not cli._matches(work / "source.json", wav, crossovers=bands.DEFAULT_CROSSOVERS)
    # a source.json from before input normalisation never matches
    old = {k: v for k, v in s.items() if k not in ("crossovers_requested", "r_ref_dbfs")}
    (work / "source.json").write_text(json.dumps(old), encoding="utf-8")
    assert not cli._matches(work / "source.json", wav)


# ------------------------------------------------------------ 1 s RMS meter (3, 9)

@pytest.mark.parametrize("fs", [44100, 48000, 96000])
def test_rms_full_scale_sine(fs):
    r = meter.rms_1s(_sine(997.0, 1.0, 5.0, fs), fs)
    assert r["loud_section_dbfs"] == pytest.approx(-3.0103, abs=0.005)
    assert r["max_dbfs"] == pytest.approx(-3.0103, abs=0.005)
    ra = meter.rms_1s(_sine(997.0, 1.0, 5.0, fs), fs, aes17=True)
    assert ra["loud_section_dbfs"] == pytest.approx(0.0, abs=0.005) and ra["aes17"]
    assert ra["max_dbfs"] - r["max_dbfs"] == pytest.approx(meter.AES17_OFFSET_DB, abs=1e-12)


def test_rms_level_channels_and_dc():
    assert meter.rms_1s(_sine(440.0, 0.1, 3.0, FS), FS)["max_dbfs"] == \
        pytest.approx(-20.0 - 3.0103, abs=0.005)
    one = _sine(440.0, 1.0, 3.0, FS)
    one[:, 1] = 0.0                                   # mean of the channels' squares
    assert meter.rms_1s(one, FS)["max_dbfs"] == pytest.approx(-6.0206, abs=0.005)
    assert meter.rms_1s(np.full((FS * 2, 2), 0.5), FS)["max_dbfs"] == \
        pytest.approx(20 * math.log10(0.5), abs=1e-9)
    mono = meter.rms_1s(_sine(440.0, 0.5, 3.0, FS)[:, 0], FS)["max_dbfs"]
    assert mono == pytest.approx(meter.rms_1s(_sine(440.0, 0.5, 3.0, FS), FS)["max_dbfs"])


def test_rms_noise():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((FS * 20, 2)) * 0.05
    r = meter.rms_1s(x, FS)
    assert r["loud_section_dbfs"] == pytest.approx(20 * math.log10(0.05), abs=0.05)
    # p95 of the 1 s series sits a little above the long-term RMS, the max above that
    assert r["max_dbfs"] >= r["loud_section_dbfs"] >= 20 * math.log10(0.05) - 0.01


def test_rms_loudest_section_and_gate():
    quiet, loud = 10 ** (-30 / 20), 10 ** (-10 / 20)
    x = np.concatenate([np.full((20 * FS, 2), quiet), np.full((5 * FS, 2), loud)])
    r = meter.rms_1s(x, FS)
    # 41 of the 241 windows lie inside the loud 5 s: more than the top 5 %
    assert r["loud_section_dbfs"] == pytest.approx(-10.0, abs=1e-9)
    assert r["max_dbfs"] == pytest.approx(-10.0, abs=1e-9)
    t, db = meter.rms_series(x, FS)
    assert t.size == 241 and t[0] == pytest.approx(1.0) and t[1] - t[0] == pytest.approx(0.1)
    # silence is gated out: 10 s of digital silence before 5 s at -20 dBFS DC
    y = np.concatenate([np.zeros((10 * FS, 2)), np.full((5 * FS, 2), 0.1)])
    assert meter.rms_1s(y, FS)["loud_section_dbfs"] == pytest.approx(-20.0, abs=1e-9)
    nan = meter.rms_1s(np.zeros((3 * FS, 2)), FS)
    assert math.isnan(nan["loud_section_dbfs"]) and math.isnan(nan["max_dbfs"])
    assert math.isnan(meter.rms_1s(np.full((FS // 2, 2), 0.5), FS)["max_dbfs"])
    m = meter.measure(_sine(997.0, 1.0, 4.0, FS), FS)
    assert m["rms_1s_loud_dbfs"] == pytest.approx(-3.0103, abs=0.005)


# ------------------------------------------------------------ input normalisation (2 item 1)

def test_input_gain():
    g, m = planner.input_gain(_sine(997.0, 10 ** (-6 / 20), 5.0, FS), FS)
    assert m == pytest.approx(-9.0103, abs=0.005)
    assert g == pytest.approx(planner.R_REF_DBFS - m)
    assert planner.R_REF_DBFS == -18.0
    g0, m0 = planner.input_gain(np.zeros((FS * 3, 2)), FS)       # silence: no gain
    assert g0 == 0.0 and math.isnan(m0)


def test_analysis_runs_on_the_normalised_premaster(pana):
    x, a = pana
    inp = a["meta"]["input"]
    g, m = planner.input_gain(x, FS)
    assert inp == {"gain_db": g, "r_ref_dbfs": -18.0, "measured_dbfs": m}
    assert a["loudness"]["summary"]["rms_1s_loud_dbfs"] == pytest.approx(-18.0, abs=1e-9)
    plan = planner.plan_from_analysis(a, planner.Controls())
    assert plan.input == InputPlan(g, -18.0, m)
    assert plan.input is not None and Plan.from_json(plan.to_json()).input == plan.input
    rec = planner.plan_record(a, planner.Controls())
    assert rec["input"] == inp


def test_gain_stage_and_render_pre():
    x = _sine(100.0, 0.25, 0.5, FS)
    y = gain_stage.apply(x, FS, InputPlan(6.0))
    assert np.allclose(y, x * 10 ** 0.3, rtol=0, atol=1e-15)
    assert np.array_equal(gain_stage.apply(x, FS, InputPlan(0.0)), x)
    p = Plan(fs=FS, input=InputPlan(-6.0))
    assert np.allclose(solver.render_pre(x, FS, p), x * 10 ** -0.3, rtol=0, atol=1e-15)
    from eluate import stages
    assert stages.render(x, FS, p)[0] == pytest.approx(x * 10 ** -0.3)
    with pytest.raises(ValueError):
        InputPlan(math.nan)


@pytest.fixture(scope="module")
def two_levels():
    out = {}
    for lvl in (0.0, -17.0):
        x, a = short_planner_analysis(FS, 2, lvl)
        c = planner.Controls()
        res = solver.solve(x, FS, planner.plan_from_analysis(a, c), planner.targets_from(a, c),
                           max_outer=2)
        out[lvl] = (x, a, res)
    return out


@pytest.mark.slow
def test_output_independent_of_premaster_level(two_levels):
    """The chunk's main property: a premaster at any level gives the same master up to
    float rounding (one 24-bit step where rounding lands either side of a half)."""
    (xa, aa, ra), (xb, ab, rb) = two_levels[0.0], two_levels[-17.0]
    assert np.max(np.abs(xb - xa * 10 ** (-17 / 20))) < 1e-15
    ga, gb = aa["meta"]["input"]["gain_db"], ab["meta"]["input"]["gain_db"]
    assert gb - ga == pytest.approx(17.0, abs=1e-9)
    assert aa["meta"]["crossovers"] == ab["meta"]["crossovers"]
    # every analysis-derived level is the same: thresholds, ceilings, starting drive
    pa, pb = planner.plan_from_analysis(aa, planner.Controls()), \
        planner.plan_from_analysis(ab, planner.Controls())
    assert np.allclose(pa.multiband.threshold_db, pb.multiband.threshold_db, atol=1e-9)
    assert pa.limiter.input_gain_db == pytest.approx(pb.limiter.input_gain_db, abs=1e-9)
    # and every level solved on the audio (chunk 3): thresholds, ceilings, clipper threshold
    nan = lambda t: np.array([[math.nan if c is None else c for c in r] for r in t])  # noqa: E731
    ma, mb_ = ra.plan.multiband, rb.plan.multiband
    assert np.allclose(ma.threshold_db, mb_.threshold_db, atol=1e-6)
    if ma.ceiling_db is not None or mb_.ceiling_db is not None:
        assert np.allclose(nan(ma.ceiling_db), nan(mb_.ceiling_db), atol=1e-6, equal_nan=True)
    assert ra.plan.limiter.clip_threshold_db == pytest.approx(rb.plan.limiter.clip_threshold_db,
                                                              abs=1e-6)
    assert ra.plan.glue.threshold_db == pytest.approx(rb.plan.glue.threshold_db, abs=1e-9)
    assert ra.levers.drive_db == pytest.approx(rb.levers.drive_db, abs=1e-6)
    assert np.max(np.abs(ra.pre_limiter - rb.pre_limiter)) < 1e-9
    lsb = 2.0 ** -23
    assert np.max(np.abs(ra.master - rb.master)) <= lsb + 1e-15
    assert np.mean(ra.master != rb.master) < 1e-3


@pytest.mark.slow
def test_input_eq_fit_is_level_invariant(two_levels):
    """Normalisation is a pure gain ahead of the input EQ, and the EQ is fitted on the
    normalised analysis: a premaster at 0 and -17 dB gets the same EQ, and the solver's
    rotation refit (which reads only the design record) agrees too."""
    (_, aa, ra), (_, ab, rb) = two_levels[0.0], two_levels[-17.0]
    c = planner.Controls()
    ea, reca = planner.plan_inputeq(aa, c, FS)
    eb, recb = planner.plan_inputeq(ab, c, FS)
    assert ea is not None
    assert planner.inputeq_sections(ea) == planner.inputeq_sections(eb)
    assert np.allclose(ea.design["residual_db"], eb.design["residual_db"], atol=1e-9)
    assert reca["residual_rms_db"] == pytest.approx(recb["residual_rms_db"], abs=1e-9)
    assert ra.plan.inputeq is not None
    assert planner.inputeq_sections(ra.plan.inputeq) == planner.inputeq_sections(rb.plan.inputeq)
    assert planner.inputeq_sections(planner.refit_inputeq(ea, FS, 0.4)) == \
        planner.inputeq_sections(planner.refit_inputeq(eb, FS, 0.4))


def _eq():
    return InputEqPlan(EqHighPass(25.0, 0.6), EqLowShelf(1.5, 120.0), EqBell(-2.0, 300.0, 1.4),
                       EqHighShelf(2.0, 12000.0))


def test_render_pre_runs_gain_then_eq():
    rng = np.random.default_rng(3)
    x = 0.1 * rng.standard_normal((FS, 2))
    p = Plan(fs=FS, input=InputPlan(-6.0), inputeq=_eq())
    want = eq_stage.apply(gain_stage.apply(x, FS, p.input), FS, p.inputeq)
    got = solver.render_pre(x, FS, p)
    assert np.array_equal(got, want)
    # the EQ is linear: the same audio as EQ then gain, up to float rounding
    assert np.max(np.abs(got - eq_stage.apply(x, FS, p.inputeq) * 10 ** -0.3)) < 1e-12
    from eluate import stages
    y, per = stages.render(x, FS, p)
    assert list(per) == ["input", "inputeq"] and np.array_equal(y, got)
    # preserve_mix drops the EQ, never the gain
    pm = Plan(fs=FS, preserve_mix=True, input=InputPlan(-6.0), inputeq=_eq())
    assert np.array_equal(solver.render_pre(x, FS, pm), gain_stage.apply(x, FS, pm.input))


def test_chain_rotation_keeps_the_input_gain(pana):
    """The solver's pre cache with both front stages: a tone rotation re-renders input
    gain + refitted EQ + notch + side strip, and equals an uncached render."""
    x, a = pana
    p0 = planner.plan_from_analysis(a, planner.Controls())
    assert p0.input is not None and p0.input.gain_db != 0.0 and p0.inputeq is not None
    xs = x[: 3 * FS]
    with solver.Chain(xs, FS, p0) as ch:
        assert np.array_equal(ch.pre, solver.render_pre(xs, FS, p0))
        lv = solver.Levers.from_plan(p0)
        lv.tone_rotation = 0.5
        p_r, _ = ch.pre_limiter(lv)
        assert p_r.input == p0.input and p_r.inputeq != p0.inputeq
        assert np.array_equal(ch.pre, solver.render_pre(xs, FS, p_r))
        no_gain = solver.render_pre(xs, FS, Plan(fs=FS, inputeq=p_r.inputeq, notch=p_r.notch,
                                                 sidestrip=p_r.sidestrip))
        assert not np.allclose(ch.pre, no_gain)


@pytest.mark.slow
def test_master_is_the_written_file(two_levels, tmp_path):
    x, a, r = two_levels[0.0]
    assert r.plan.output == OutputPlan(24, "none") and r.plan.limiter.ceiling_dbtp <= -0.3
    k = r.master * 2.0 ** 23
    assert np.array_equal(k, np.round(k))
    io.write(tmp_path / "m.wav", r.master, FS, r.plan.output.subtype)
    y, _ = io.read(tmp_path / "m.wav")
    assert sf.info(str(tmp_path / "m.wav")).subtype == "PCM_24"
    assert np.array_equal(y.astype(np.float64), r.master)
    tp = solver.true_peaks(y, FS)
    assert tp["8x"] <= -0.3 + 1e-9                  # the 24-bit default meter (S6)
    assert r.metrics["true_peak_constrained_dbtp"] == pytest.approx(tp["8x"], abs=1e-12)


# ------------------------------------------------------------ ceiling and export (3)

def test_ceiling_defaults_and_streaming(pana):
    assert planner.CEILING_DBTP == -0.3 and planner.STREAMING_CEILING_DBTP == -1.0
    assert LimiterPlan().ceiling_dbtp == -0.3
    assert planner.Targets().true_peak_max_dbtp == -0.3
    assert planner.Controls().output_plan() == OutputPlan(24, "none")
    s = planner.Controls(streaming=True)
    assert s.ceiling_dbtp == -1.0 and s.output_plan() == OutputPlan(16, "tpdf")
    _, a = pana
    for c, ceil, bits in ((planner.Controls(), -0.3, 24), (s, -1.0, 16)):
        p = planner.plan_from_analysis(a, c)
        assert p.limiter.ceiling_dbtp == ceil and p.output.bits == bits
        assert planner.targets_from(a, c).true_peak_max_dbtp == ceil
        assert Plan.from_json(p.to_json()).output == p.output
    args = cli.build_parser().parse_args(["master", "in.wav", "-o", "o", "--streaming"])
    assert cli._controls(args).streaming
    assert not cli._controls(cli.build_parser().parse_args(["master", "i", "-o", "o"])).streaming


def test_output_plan_validation():
    assert OutputPlan().subtype == "PCM_24" and OutputPlan(16, "tpdf").subtype == "PCM_16"
    assert OutputPlan(32).subtype == "FLOAT"
    for bad in ({"bits": 20}, {"dither": "rpdf"}, {"bits": 32, "dither": "tpdf"},
                {"dither_seed": -1}):
        with pytest.raises(ValueError):
            OutputPlan(**bad)


def test_quantise_24_and_16():
    rng = np.random.default_rng(3)
    x = rng.uniform(-0.95, 0.95, (200000, 2))
    y24 = output_stage.apply(x, FS, OutputPlan(24))
    k = y24 * 2.0 ** 23
    assert np.array_equal(k, np.round(k)) and np.max(np.abs(y24 - x)) <= 0.5 * 2.0 ** -23
    y16 = output_stage.apply(x, FS, OutputPlan(16, "tpdf", 5))
    k = y16 * 2.0 ** 15
    assert np.array_equal(k, np.round(k))
    e = (y16 - x) * 2.0 ** 15
    assert np.max(np.abs(e)) <= 1.5
    # TPDF + rounding: zero mean, variance 1/6 + 1/12 LSB^2
    assert abs(float(e.mean())) < 0.01 and float(e.var()) == pytest.approx(0.25, rel=0.03)
    assert np.array_equal(y16, output_stage.apply(x, FS, OutputPlan(16, "tpdf", 5)))
    assert not np.array_equal(y16, output_stage.apply(x, FS, OutputPlan(16, "tpdf", 6)))
    # clipped to the integer range
    full = output_stage.apply(np.array([[1.0, -1.0], [2.0, -2.0]]), FS, OutputPlan(16))
    assert np.array_equal(full, [[32767 / 32768, -1.0], [32767 / 32768, -1.0]])
    f32 = output_stage.apply(x, FS, OutputPlan(32))
    assert np.array_equal(f32, x.astype(np.float32).astype(np.float64))


def test_tpdf_dither_keeps_a_signal_below_one_step():
    """A sine of a quarter of a 16-bit step: rounding alone gives digital silence, TPDF
    dither keeps it (the dithered output's mean is the input)."""
    n = 1 << 18
    a = 0.25 * 2.0 ** -15
    s = a * np.sin(2 * np.pi * 1000.0 * np.arange(n) / FS)
    x = np.stack([s, s], axis=1)
    assert not np.any(output_stage.apply(x, FS, OutputPlan(16)))
    y = output_stage.apply(x, FS, OutputPlan(16, "tpdf"))
    est = 2.0 * float(np.mean(y[:, 0] * np.sin(2 * np.pi * 1000.0 * np.arange(n) / FS)))
    assert est == pytest.approx(a, rel=0.1)


def test_pcm_write_is_exact(tmp_path):
    rng = np.random.default_rng(4)
    x = rng.uniform(-1.0, 1.0, (5000, 2))
    for op in (OutputPlan(24), OutputPlan(16, "tpdf")):
        y = output_stage.apply(x, FS, op)
        p = tmp_path / f"{op.bits}.wav"
        io.write(p, y, FS, op.subtype)
        r, _ = io.read(p)
        assert sf.info(str(p)).subtype == op.subtype
        assert np.array_equal(r.astype(np.float64), y)


def test_render_plan_applies_the_output_stage():
    x = _sine(997.0, 0.5, 1.0, FS)
    p = Plan(fs=FS, limiter=LimiterPlan(input_gain_db=6.0), output=OutputPlan(16, "tpdf"))
    y, info, z = solver.render_plan(x, FS, p)
    k = y * 2.0 ** 15
    assert np.array_equal(k, np.round(k))
    assert solver.true_peaks(y, FS)["8x"] <= -0.3 + 0.01
