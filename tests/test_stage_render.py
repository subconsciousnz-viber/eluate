"""Plan (de)serialisation and the fixed-order render chain (spec 1, 3.5; audit 05 A14)."""

import json

import numpy as np
import pytest

from fixtures import synthetic as syn
from eluate import meter
from eluate.plan import (STAGE_ORDER, EqBell, EqHighPass, EqHighShelf, EqLowShelf, Envelope,
                            InputEqPlan, InputPlan, KeyEqPlan, LimiterPlan, MultibandPlan, Notch,
                            NotchPlan, Plan, RUN_ID_KEY, SaturatePlan, SidestripPlan, TonePlan,
                            control_length, tone_nodes)
from eluate.stages import STAGES, multiband, notch, render

FS = 48000
RATES = [44100, 48000, 96000]


def _signal(seconds=3.0, fs=FS):
    x, meta = syn.kick_sub(fs, bars=2)
    n = int(seconds * fs)
    x = np.resize(x, (n, 2))
    x = x + syn.noise_psd_slope(n, fs, -4.5, rms_dbfs=-24, seed=7, correlation=0.6)
    x, _ = syn.add_ring(x, fs, f0=3150.0, q=10.0, gain_db=8.0, start_s=0.5, end_s=2.5)
    period = len(meta["kick_onsets"]) * 60.0 / meta["bpm"]
    onsets = [o for o in np.concatenate([meta["kick_onsets"], meta["kick_onsets"] + period])
              if o < seconds]
    return x, onsets


def _full_plan(x, onsets, fs=FS):
    n = x.shape[0]
    nc = control_length(n, fs)
    g = np.zeros(nc)
    g[500:2500] = -5.0
    mb = multiband.design(x, fs, np.full((5, 2), -24.0), ratio=2.0,
                          decoherence=[0.0, 0.1, 0.3, 0.4, 0.8],
                          mid_makeup_db=[0.0, 0.5, 0.0, 0.0, 0.0])
    nodes = tone_nodes()
    tilt = np.linspace(-1.5, 1.5, len(nodes))
    return Plan(
        fs=fs,
        inputeq=InputEqPlan(EqHighPass(25.0, 0.6), EqLowShelf(1.5, 120.0),
                            EqBell(-2.0, 300.0, 2.0), EqHighShelf(1.0, 16000.0)),
        notch=NotchPlan([Notch(3150.0, 10.0, Envelope(g)), Notch(8000.0, 6.0, Envelope(g * 0.6))],
                        kick_mask=notch.kick_mask(onsets, n / fs)),
        sidestrip=SidestripPlan(100.0),
        multiband=mb,
        tone=TonePlan([Envelope(np.linspace(0, 1, nc) * t) for t in tilt]),
        saturate=SaturatePlan(drive_db=[[0, 0], [3, 0], [2, 1], [1, 4], [0, 0]],
                              harmonic=[0.0, 0.4, 0.3, 0.6, 0.7]),
        keyeq=KeyEqPlan(enabled=False),
        limiter=LimiterPlan(ceiling_dbtp=-1.0, input_gain_db=4.0, clip_depth_db=2.0,
                            clip_onsets_s=onsets),
    )


def test_plan_json_round_trip_exact():
    x, onsets = _signal()
    p = _full_plan(x, onsets)
    text = p.to_json()
    q = Plan.from_json(text)
    assert q.to_dict() == p.to_dict()
    assert q.to_json() == text


def test_save_load_with_npz_sidecar(tmp_path):
    """Spec 1: plan.json holds scalars and sidecar keys, plan.npz the float32 envelopes;
    the reloaded plan is identical and renders bit-identically."""
    x, onsets = _signal(1.5)
    p = _full_plan(x, onsets)
    path = tmp_path / "plan.json"
    npz = p.save(path)
    raw = path.read_bytes()
    assert b"\r" not in raw
    d = json.loads(raw)
    assert d["sidecar"] == "plan.npz" and npz.exists()
    assert "values" not in json.dumps(d)  # no envelope data in the JSON
    with np.load(npz) as z:
        assert all(z[k].dtype == np.float32 for k in z.files if k != RUN_ID_KEY)
        assert str(z[RUN_ID_KEY]) == d["run_id"] and len(d["run_id"]) == 32
    q = Plan.load(path)
    assert q.to_dict() == p.to_dict()
    print(f"plan.json {len(raw)} bytes, plan.npz {npz.stat().st_size} bytes for 1.5 s; "
          f"inline JSON {len(p.to_json())} bytes")
    y1, _ = render(x, FS, p)
    y2, _ = render(x, FS, q)
    assert np.array_equal(y1, y2)
    # a sidecar from another run: a clear error, not a KeyError (audit 08); caught by the
    # run id (report 3.1), and by the missing key for a plan.json written before run ids
    np.savez(npz, unrelated=np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError, match="plan.json and plan.npz are from different runs"):
        Plan.load(path)
    del d["run_id"]
    path.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError, match="not in the plan.npz sidecar"):
        Plan.load(path)


def test_save_pairs_json_and_sidecar_by_run_id(tmp_path):
    """Report 3.1: a run stopped between the two writes leaves a pair from two runs, which
    loads as an error; a plan.json without a run id (older) loads unchecked. The writes
    are atomic (report 3.2): no temporary file is left behind."""
    x, onsets = _signal(1.0)
    p = _full_plan(x, onsets)
    a, b = tmp_path / "a", tmp_path / "b"
    p.save(a / "plan.json", run_id="1" * 32)
    p.save(b / "plan.json")
    assert json.loads((a / "plan.json").read_text(encoding="utf-8"))["run_id"] == "1" * 32
    assert Plan.load(a / "plan.json").to_dict() == p.to_dict()
    assert sorted(f.name for f in a.iterdir()) == ["plan.json", "plan.npz"]
    (a / "plan.npz").write_bytes((b / "plan.npz").read_bytes())        # npz of another run
    with pytest.raises(ValueError, match="plan.json and plan.npz are from different runs"):
        Plan.load(a / "plan.json")
    d = json.loads((a / "plan.json").read_text(encoding="utf-8"))
    del d["run_id"]
    (a / "plan.json").write_text(json.dumps(d), encoding="utf-8")
    assert Plan.load(a / "plan.json").to_dict() == p.to_dict()


def test_load_rejects_a_malformed_plan_and_a_sidecar_path(tmp_path):
    """Report 12d, 12e: a malformed plan.json is a ValueError naming the file, and the
    sidecar must be a bare .npz file name (it is joined to the plan's folder)."""
    x, onsets = _signal(1.0)
    path = tmp_path / "w" / "plan.json"
    _full_plan(x, onsets).save(path)
    good = json.loads(path.read_text(encoding="utf-8"))
    (tmp_path / "other.npz").write_bytes((path.parent / "plan.npz").read_bytes())
    for name in ("../other.npz", str(tmp_path / "other.npz"), "plan.json", "sub/plan.npz",
                 7, None):
        path.write_text(json.dumps({**good, "sidecar": name}), encoding="utf-8")
        with pytest.raises(ValueError, match="not a .npz file name"):
            Plan.load(path)
    for bad in ({k: v for k, v in good.items() if k != "fs"},              # KeyError
                {**good, "limiter": {**good["limiter"], "no_such_field": 1}},   # TypeError
                {**good, "sidestrip": 100.0}):
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match="plan.json"):
            Plan.load(path)
    for text in ("{", "[1, 2]"):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="plan.json"):
            Plan.load(path)


@pytest.mark.parametrize("fs", RATES)
@pytest.mark.slow
def test_render_chain_order_skip_and_determinism(fs):
    x, onsets = _signal(3.0, fs)
    p = _full_plan(x, onsets, fs)
    y, stages = render(x, fs, p, keep_stages=True)
    # keyeq disabled, glue, speclim, input and output absent (no section): all skipped
    assert list(stages) == [s for s in STAGE_ORDER
                            if s not in ("keyeq", "glue", "speclim", "input", "output")]
    assert stages["limiter"] is y
    tp8 = meter.true_peak(y, fs, 8)
    print(f"{fs} Hz chain: true peak {tp8:.4f} dBTP")
    assert tp8 <= -1.0 + 1e-9
    y2, _ = render(x, fs, p)
    y3, _ = render(x, fs, Plan.from_json(p.to_json()))
    assert np.array_equal(y, y2) and np.array_equal(y, y3)
    prev = np.asarray(x, dtype=np.float64)
    for name, out in stages.items():
        assert np.array_equal(STAGES[name](prev, fs, getattr(p, name)), out)
        prev = out


def test_absent_sections_are_skipped():
    x, _ = _signal(1.0)
    y, stages = render(x, FS, Plan(fs=FS))
    assert stages == {} and np.array_equal(y, x)
    _, stages = render(x, FS, Plan(fs=FS, sidestrip=SidestripPlan(), limiter=LimiterPlan()))
    assert list(stages) == ["sidestrip", "limiter"]


def test_preserve_mix_disables_tone_and_inputeq():
    x, _ = _signal(1.0)
    nc = control_length(x.shape[0], FS)
    nodes = tone_nodes()
    p = Plan(fs=FS, preserve_mix=True, tone=TonePlan([Envelope(np.full(nc, 2.0))] * len(nodes)),
             inputeq=InputEqPlan(EqHighPass(), EqLowShelf(2.0, 100.0)))
    y, stages = render(x, FS, p)
    assert "tone" not in stages and "inputeq" not in stages and np.array_equal(y, x)


def test_inputeq_follows_the_input_gain():
    assert STAGE_ORDER[:3] == ("input", "inputeq", "notch")
    x, _ = _signal(1.0)
    p = Plan(fs=FS, input=InputPlan(-3.0), inputeq=InputEqPlan(EqHighPass()),
             sidestrip=SidestripPlan())
    _, stages = render(x, FS, p)
    assert list(stages) == ["input", "inputeq", "sidestrip"]


def test_keyeq_enabled_is_a_stub_disabled_is_skipped():
    x, _ = _signal(0.5)
    with pytest.raises(NotImplementedError, match="bass root"):
        render(x, FS, Plan(fs=FS, keyeq=KeyEqPlan(enabled=True)))
    y, stages = render(x, FS, Plan(fs=FS, keyeq=KeyEqPlan()))
    assert stages == {} and np.array_equal(y, x)


def test_render_input_checks():
    with pytest.raises(ValueError):
        render(np.zeros((100, 2)), 44100, Plan(fs=FS))
    for bad in (np.zeros(100), np.zeros((100, 1)), np.zeros((100, 3)), np.zeros((1, 2))):
        with pytest.raises(ValueError):
            render(bad, FS, Plan(fs=FS))
    with pytest.raises(ValueError):
        render(np.full((100, 2), np.nan), FS, Plan(fs=FS))


def test_envelope_pchip_no_overshoot_and_constant_runs():
    v = np.array([0.0, 0.0, -3.0, -3.0, 0.0, -6.0, -6.0, -6.0, 0.0])
    e = Envelope(v)
    a = e.at(9 * 48, FS)
    assert a.min() >= -6.0 and a.max() <= 0.0
    assert np.all(a[0:48] == 0.0)
    assert np.all(a[2 * 48:3 * 48] == -3.0)
    assert np.all(a[5 * 48:7 * 48] == -6.0)
    assert np.all(e.at(20 * 48, FS)[8 * 48:] == 0.0)
    assert np.all(Envelope.constant(1.5).at(10, FS) == 1.5)
    # values are float32-representable (sidecar round trip is exact)
    assert Envelope(np.array([0.1])).values[0] == float(np.float32(0.1))


def test_plan_validation():
    with pytest.raises(ValueError):
        Envelope(np.array([0.0, np.nan]))
    with pytest.raises(ValueError):
        LimiterPlan(smoother="iir")
    with pytest.raises(ValueError):
        SaturatePlan(drive_db=[[0, 0]] * 5, harmonic=[0, 0, 2.0, 0, 0])
    with pytest.raises(ValueError):
        SaturatePlan(drive_db=[[0, 0]] * 5, harmonic=[0] * 5, symmetry=1.5)
    with pytest.raises(ValueError):
        Plan.from_dict({"fs": FS, "version": "9"})
    flat = [[Envelope.constant(0)] * 2] * 5
    with pytest.raises(ValueError):
        MultibandPlan(flat, side_makeup_db=-1.0)
    with pytest.raises(ValueError):
        MultibandPlan(flat, crossovers_hz=[100.0, 100.0, 2000.0, 8000.0])
    with pytest.raises(ValueError):
        TonePlan([Envelope.constant(0)] * 2, nodes_hz=[1000.0, 500.0])
    with pytest.raises(ValueError):
        Plan(fs=0)
    with pytest.raises(ValueError):
        Plan(fs=48000.5)
    with pytest.raises(ValueError):
        Plan(fs=16000, multiband=MultibandPlan(               # 8 kHz crossover at Nyquist
            flat, crossovers_hz=[100.0, 500.0, 2000.0, 8000.0]))
    with pytest.raises(ValueError):
        Plan(fs=FS, notch=NotchPlan([Notch(30000.0, 4.0, Envelope.constant(-3))]))
    with pytest.raises(ValueError):
        Plan(fs=FS, sidestrip=SidestripPlan(30000.0))


def test_fs_coerced_to_int():
    """Audit 05 A14: Plan(fs=48000.0) round-trips."""
    p = Plan(fs=48000.0)
    assert p.fs == 48000 and isinstance(p.fs, int)
    assert Plan.from_json(p.to_json()).to_json() == p.to_json()


def test_per_stage_outputs_optional_via_callback():
    """Spec 1 / audit 05 Q15: by default only the final audio is kept; on_stage sees
    every stage's output in order."""
    x, onsets = _signal(1.0)
    p = _full_plan(x, onsets)
    seen = []
    y, stages = render(x, FS, p, on_stage=lambda name, a: seen.append((name, a.copy())))
    assert list(stages) == [n for n, _ in seen]
    assert all(v is None for v in stages.values())
    assert np.array_equal(seen[-1][1], y)
    _, kept = render(x, FS, p, keep_stages=True)
    for name, a in seen:
        assert np.array_equal(kept[name], a)


def test_environment_recorded_and_mismatch_warns(tmp_path):
    """Audit 05 Q11: the plan records numpy, scipy and package versions; a render with
    a plan from another environment warns."""
    import warnings

    import scipy

    from eluate import __version__
    p = Plan(fs=FS, sidestrip=SidestripPlan())
    assert {k: p.environment[k] for k in ("numpy", "scipy", "eluate")} == {
        "numpy": np.__version__, "scipy": scipy.__version__, "eluate": __version__}
    p.save(tmp_path / "plan.json")
    q = Plan.load(tmp_path / "plan.json")
    assert q.environment == p.environment
    x = np.zeros((1000, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        render(x, FS, q)  # same environment: no warning
    d = p.to_dict()
    d["environment"]["numpy"] = "0.0.0"
    with pytest.warns(RuntimeWarning, match="bit-identical"):
        render(x, FS, Plan.from_dict(d))


def test_validation_bounds_from_second_audit():
    """Audit 05 Q5, Q6, Q12, Q13."""
    flat = [[Envelope.constant(0)] * 2] * 5
    MultibandPlan(flat, side_makeup_db=3.0, mid_makeup_db=[-3.0, 3.0, 0, 0, 0])
    for bad in (dict(side_makeup_db=3.5), dict(side_makeup_db=-0.1),
                dict(mid_makeup_db=[0, 0, -40.0, 0, 0]), dict(order=float("nan")),
                dict(order=2.5), dict(order=0)):
        with pytest.raises(ValueError):
            MultibandPlan(flat, **bad)
    SaturatePlan(drive_db=[[0, 0]] + [[12.0, 12.0]] * 4, harmonic=[0.5] * 5)
    with pytest.raises(ValueError):
        SaturatePlan(drive_db=[[0, 0]] + [[12.5, 0]] * 4, harmonic=[0.5] * 5)
    with pytest.raises(ValueError):
        SaturatePlan(drive_db=[[0, 0]] * 5, harmonic=[0] * 5, order=float("nan"))
    with pytest.raises(ValueError):
        SidestripPlan(order=float("inf"))
    for c in (float("nan"), -0.1, 1.5):
        with pytest.raises(ValueError):
            KeyEqPlan(min_confidence=c)
    with pytest.raises(ValueError):
        LimiterPlan(clip_ramp_ms=0.0)
    x, _ = _signal(0.5)
    with pytest.raises(ValueError, match="ratio"):
        multiband.design(x, FS, np.full((5, 2), -20.0), ratio=1.0)
    # attack floor: 0.5 ms requests clamp to two control samples, so smoothing still happens
    p = multiband.design(x, FS, np.full((5, 2), -40.0), ratio=4.0, attack_ms=0.5)
    q = multiband.design(x, FS, np.full((5, 2), -40.0), ratio=4.0, attack_ms=2.0)
    assert all(a == b for pa, pb in zip(p.gr_db, q.gr_db) for a, b in zip(pa, pb))
