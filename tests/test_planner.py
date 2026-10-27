"""Planner: analysis + controls -> Plan (spec 1, 3.2, 3.5, 3.6)."""

import copy
import pickle
from pathlib import Path
import math

import numpy as np
import pytest

from fixtures.e_signals import short_analysis
from eluate import planner
from eluate.plan import NOTCH_MAX_CUT_DB, PLATFORM_KEYS, VERSION_KEYS, Plan, environment

FS = 44100


@pytest.fixture(scope="module")
def ana():
    return short_analysis(FS)


def _pn(a):
    p, rows, _ = planner.plan_notches(a)
    return p, rows


planner.plan_notches3 = _pn          # two-value view used by the older tests


# ------------------------------------------------------------ tempo octave (S2, S3)

_DOUBLED = {"decision": "doubled", "from_bpm": 72.5, "to_bpm": 145.0, "gain": 1.1,
            "candidates": [{"bpm": 72.5, "lattice_score": 2.17, "upper_acf": 0.11, "kick_acf": 0.1},
                           {"bpm": 145.0, "lattice_score": 2.56, "upper_acf": 0.05,
                            "kick_acf": 0.02}]}


def test_auto_character_on_a_detected_tempo():
    """A detected tempo never resolves to slow (an octave error must not turn the kick
    detective on); a given one does. A moved octave is quoted with its evidence."""
    assert planner.character_choice("auto", 65.0)[0] == "slow"
    p, why = planner.character_choice("auto", 65.0, "analysis")
    assert p == "moderate" and "--bpm" in why
    p, why = planner.character_choice("auto", 145.0, "analysis", _DOUBLED)
    assert p == "fast" and "octave doubled from 72.5" in why and "2.56 against 2.17" in why
    assert "octave" not in planner.character_choice("auto", 145.0, "user", _DOUBLED)[1]
    a = {"plan_inputs": {"tempo_bpm": 145.0, "tempo_octave": "doubled from lattice",
                         "metrical": {"octave": _DOUBLED}}, "meta": {"crossovers": [100.0]}}
    c = planner.character_settings(a, planner.Controls(), nb=2)
    assert c["preset"] == "fast" and c["tempo_octave"] == "doubled from lattice"
    c = planner.character_settings(a, planner.Controls(bpm=72.5), nb=2)
    assert c["preset"] == "moderate" and c["bpm_source"] == "user" and "octave" not in c["reason"]


def test_plan_hits_one_per_beat_at_the_tempo():
    """S3: a grid at half the plan's tempo (the user's, else the analysis's) is
    subdivided, so each hit is one beat, not the louder of two."""
    grid = list(np.arange(0.0, 8.0, 60.0 / 72.5))
    a = {"plan_inputs": {"beat_times": grid, "grid_informative": True, "tempo_bpm": 145.0},
         "meta": {"duration_s": 8.0}}
    h = planner.plan_hits(a, planner.Controls())
    assert h.source == "analysis beat grid, subdivided x2 to 145.0 BPM"
    assert np.allclose(np.diff(h.beats_s), 60.0 / 145.0) and max(h.beats_s) < 8.0
    a["plan_inputs"]["tempo_bpm"] = 72.5
    h = planner.plan_hits(a, planner.Controls())
    assert h.beats_s == grid and h.source == "analysis beat grid"
    assert "subdivided x2" in planner.plan_hits(a, planner.Controls(bpm=145.0)).source


def test_plan_hits_follow_the_doubled_octave():
    """End to end on the S2 fixture: analysis at 145 (not 72.5), one hit per beat."""
    from test_feature_structure import TRACK_F, _grid_track
    from eluate.features import structure
    x = _grid_track(145.0, 40.0, **TRACK_F)
    st = structure.analyse(x, FS)
    a = {"plan_inputs": {"beat_times": st["beat_times"], "grid_informative": st["grid_informative"],
                         "tempo_bpm": st["tempo_bpm"]}}
    h = planner.plan_hits(a, planner.Controls())
    assert h.source == "analysis beat grid"
    assert abs(60.0 / np.median(np.diff(h.beats_s)) - 145.0) < 1.0


def test_controls_validation_and_targets():
    assert planner.Controls().rms_target == -4.1
    assert planner.Controls(target_rms=-5.0).rms_target == -5.0
    with pytest.raises(ValueError):
        planner.Controls(tilt_db=3.5)
    with pytest.raises(TypeError):
        planner.Controls(genre="techno")         # genre presets retired (maintainer, 2026-10-20)
    with pytest.raises(TypeError):
        planner.Controls(harmonic=0.5)          # retired with the flatness share (chunk 4)


def test_every_section_filled(ana):
    x, a = ana
    p = planner.plan_from_analysis(a, planner.Controls())
    for name in ("inputeq", "notch", "sidestrip", "multiband", "saturate", "glue", "keyeq",
                 "limiter", "output"):
        assert getattr(p, name) is not None, name
    assert p.tone is None                        # the FIR tone stage is off (chunk 2)
    assert p.input is None            # features.analyse without normalisation (no meta input)
    assert p.keyeq.enabled is False
    assert p.limiter.ceiling_dbtp == -0.3 and p.output.bits == 24
    assert p.glue.makeup_db == 0.0
    assert 60.0 <= p.sidestrip.fc_hz <= 150.0
    assert p.limiter.clip_onsets_s == planner.usable_kick_onsets(a)
    assert p.environment == environment()


def test_preserve_mix_disables_tone_and_saturation(ana):
    _, a = ana
    p = planner.plan_from_analysis(a, planner.Controls(preserve_mix=True))
    assert p.preserve_mix and p.tone is None and p.saturate is None and p.inputeq is None
    assert all(m == 0.0 for m in p.multiband.mid_makeup_db)


def test_notches_capped_quantised_and_ring_found(ana):
    x, a = ana
    p, rows = planner.plan_notches3(a)
    assert len(p.notches) <= planner.NOTCH_MAX
    assert any(abs(n.f0_hz - 3150.0) / 3150.0 < 0.03 for n in p.notches)
    for n in p.notches:
        v = n.gain_db.values
        assert v.min() >= -NOTCH_MAX_CUT_DB and v.max() <= 0.0
        q = np.round(-v / planner.NOTCH_QUANT_DB) * planner.NOTCH_QUANT_DB
        assert np.allclose(-v, q, atol=1e-6)
        assert 1.0 <= n.q <= planner.NOTCH_Q_HI   # merging may widen below NOTCH_Q_LO
    assert p.kick_mask is not None
    assert all("rendered" in r and "spans" in r for r in rows)


def test_s_only_notch_not_rendered(ana):
    _, a = ana
    b = copy.deepcopy(a)
    for r in b["plan_inputs"]["notches"]:
        r["channel"] = "S"
    p, rows = planner.plan_notches3(b)
    assert p.notches == []
    assert all(not r["rendered"] and "S-only" in r["reason"] for r in rows)
    assert planner.notch_rendered("M") and planner.notch_rendered("both")


def test_notch_depth_rule():
    ex = np.array([0.0, 6.0, 10.0, 18.0, 40.0] * 20)
    g = planner._notch_envelope(np.repeat(ex, 1), 0.1, 1.0, 44100 * 10, 44100)
    assert g.min() >= -6.0 and g.max() <= 0.0
    assert planner._notch_envelope(np.full(100, 6.0), 0.1, 1.0, 441000, 44100).min() == 0.0
    mid = planner._notch_envelope(np.full(100, 10.0), 0.1, 1.0, 441000, 44100)
    assert mid[5000] == pytest.approx(-2.0)
    half = planner._notch_envelope(np.full(100, 10.0), 0.1, 0.5, 441000, 44100)
    assert half[5000] == pytest.approx(-1.0)


def test_tone_deficit_rotation_and_sub(ana):
    _, a = ana
    b = copy.deepcopy(a)
    b["plan_inputs"]["tilt_slope_db_oct"] = {"M": -6.0, "S": -6.0, "stereo": -6.0}
    b["plan_inputs"]["sub_shelf_db"] = {"M": 0.0, "S": 0.0, "stereo": 0.0}
    r = planner.tone_deficit(b, planner.Controls())
    nodes, d = np.array(r["nodes_hz"]), np.array(r["deficit_db"])
    assert r["delta_slope"] == pytest.approx(1.5)
    k = np.argmin(np.abs(nodes - 1000.0))
    assert d[k] == pytest.approx(1.5 * math.log2(nodes[k] / 1000.0))
    hi = nodes > 10000.0
    assert np.allclose(d[hi], 1.5 * math.log2(10.0))
    # T2: the rotation cuts no node more than EQ_LOW_ROT_CAP_DB (1.5 dB from 500 Hz down
    # here); the sub holds the capped 100 Hz deficit, predicted shelf inside +-3
    mid = (nodes >= 100.0) & (nodes < 1000.0)
    assert np.allclose(d[mid], np.maximum(1.5 * np.log2(nodes[mid] / 1000.0),
                                          -planner.EQ_LOW_ROT_CAP_DB))
    lo = nodes < 100.0
    assert np.allclose(d[lo], -planner.EQ_LOW_ROT_CAP_DB)
    b["plan_inputs"]["sub_shelf_db"]["stereo"] = 5.0
    r2 = planner.tone_deficit(b, planner.Controls())
    assert r2["sub_shelf_correction_db"] == pytest.approx(3.0 - 6.5)
    r3 = planner.tone_deficit(b, planner.Controls(tilt_db=1.5))
    assert r3["target_slope"] == -3.0 and r3["delta_slope"] == pytest.approx(3.0)


def test_saturation_follows_the_deficit_below_the_line(ana):
    _, a = ana
    b = copy.deepcopy(a)
    b["plan_inputs"]["tilt_slope_db_oct"] = {"M": -7.0, "S": -7.0, "stereo": -7.0}
    sat, rec = planner.plan_saturate(b, planner.Controls())
    d = planner.band_deficits(b, planner.Controls())
    assert sat.drive_db[0] == [0.0, 0.0]                         # the sub never
    for k in range(1, 5):
        assert sat.drive_db[k][0] == planner.saturation_drive_db(d[k])
        assert sat.drive_db[k][1] == (2.0 if k == 3 else sat.drive_db[k][0])
    assert max(v for row in sat.drive_db for v in row) <= planner.SAT_MAX_DRIVE_DB == 2.0
    # a dull mix (steep measured slope against -4.5): the top bands sit below the line
    assert d[4] > 0.0 and sat.drive_db[4][0] > 0.0
    assert [r["deficit_db"] for r in rec["bands"]] == d
    assert not hasattr(planner, "harmonic_blend") and not hasattr(planner, "SAT_DRIVE_PER_DB")

def test_multiband_record(ana):
    _, a = ana
    rec = planner.plan_multiband_record(a, planner.Controls())
    lv = planner.band_peak_levels(a)
    for b in range(5):
        for c in range(2):
            if lv[b, c].size:
                assert rec["threshold_db"][b][c] == pytest.approx(
                    np.percentile(lv[b, c], 90) - planner.MB_THRESHOLD_BELOW_P90_DB)
    # chunk 3: the ceilings are solved at realise, the timing is the character's
    assert rec["ceiling_db"] is None
    assert rec["budget"] == planner.mb_stage.default_budget("peak")
    ch = planner.character_settings(a, planner.Controls())
    assert rec["attack_ms"] == [[v, v] for v in ch["mb_attack_ms"]]
    assert rec["knee_db"] == ch["mb_knee_db"]
    mb = planner.multiband_unrealised(rec)
    assert mb.budget == rec["budget"]
    sm = mb.side_makeup()
    assert sm[0] == 0.0 and sm[1] == 0.0
    assert sm[4] == pytest.approx(2.0 * (1 - rec["decoherence"][4]))


@pytest.mark.slow
def test_realise_and_sidecar_round_trip(ana, tmp_path):
    x, a = ana
    p = planner.plan_from_analysis(a, planner.Controls(), x)
    assert any(np.any(e.values < 0) for pr in p.multiband.gr_db for e in pr)
    assert p.glue.gain_db.values.size > 1
    npz = p.save(tmp_path / "plan.json")
    assert npz.exists()
    q = Plan.load(tmp_path / "plan.json")
    assert q.to_dict() == p.to_dict()
    assert set(q.environment) == set(VERSION_KEYS + PLATFORM_KEYS)


def test_crossover_mismatch_raises(ana):
    _, a = ana
    with pytest.raises(ValueError, match="re-run the analysis"):
        planner.plan_from_analysis(a, planner.Controls(crossovers_hz=(120, 500, 2000, 8000)))


def test_targets(ana):
    _, a = ana
    t = planner.targets_from(a, planner.Controls(tilt_db=-1.0, target_rms=-2.6))
    assert t.rms_dbfs == -2.6 and t.slope_db_oct == -5.5 and t.rms_tol_db == 0.3
    assert t.crest_limits_db == [1.5, 2.0, 3.0, 3.0, 3.0]
    assert len(t.sections) == len(a["plan_inputs"]["sections"])
    assert planner.crest_limits([100, 1000]) == [1.5, 2.0, 3.0]


def test_deterministic(ana):
    x, a = ana
    p1 = planner.plan_from_analysis(a, planner.Controls())
    p2 = planner.plan_from_analysis(a, planner.Controls())
    assert p1.to_dict() == p2.to_dict()


def test_crest_exempt_follows_sidestrip(ana):
    _, a = ana
    assert planner.crest_exempt([100, 500, 2000, 8000], 100.0) == [[0, 1]]
    assert planner.crest_exempt([100, 500, 2000, 8000], 40.0) == []
    assert planner.crest_exempt([100, 250, 2000], 150.0) == [[0, 1], [1, 1]]
    t = planner.targets_from(a, planner.Controls(subside_hz=100.0))
    assert t.crest_exempt == []                       # off by default (audit 07 F17)
    t2 = planner.targets_from(a, planner.Controls(subside_hz=100.0, side_band0_exempt=True))
    assert t2.crest_exempt == [[0, 1]]


def _with_notch(a, f0=3150.0, excess=12.0, cover=0.4, channel="both"):
    b = copy.deepcopy(a)
    dur = b["meta"]["n"] / b["meta"]["fs"]
    k = len(b["t"]) if not isinstance(b["t"], dict) else b["meta"]["frames"]
    ex = np.zeros(int(b["meta"]["frames"]))
    m = int(cover * ex.size)
    ex[:m] = excess
    b["plan_inputs"]["notches"] = [{"f0": f0, "q": 8.0, "q_uncertainty": 1.0, "start": 0.0,
                                    "end": cover * dur, "spans": [[0.0, cover * dur]],
                                    "excess_db": ex, "flag": "resonance", "depth_scale": 1.0,
                                    "channel": channel}]
    b["plan_inputs"]["key"] = {"tonic": "F#"}
    b["pitch"] = {"chords": []}
    return b


def test_pitch_helpers():
    assert planner.pitch_class("C#m") == 1 and planner.pitch_class("Bb") == 10
    assert planner.pitch_class(None) is None and planner.pitch_class("N") is None
    assert planner.cents_to_pitch_class(440.0, 9) == pytest.approx(0.0)
    assert planner.cents_to_pitch_class(880.0 * 2 ** (30 / 1200), 9) == pytest.approx(30.0)
    assert planner.cents_to_pitch_class(261.63, 9) == pytest.approx(300.0, abs=0.1)


def test_suspect_candidates_not_rendered(ana):
    _, a = ana
    ok = _with_notch(a)
    p, rows = planner.plan_notches3(ok)
    assert rows[0]["rendered"] and len(p.notches) == 1
    p, rows = planner.plan_notches3(_with_notch(a, excess=planner.NOTCH_SUSPECT_EXCESS_DB + 1))
    assert not rows[0]["rendered"] and rows[0]["reason"].startswith("suspect: peak excess")
    p, rows = planner.plan_notches3(_with_notch(a, cover=0.7))
    assert not rows[0]["rendered"] and "cover" in rows[0]["reason"] and p.notches == []


def test_tonic_and_fifth_guard(ana):
    _, a = ana
    b = _with_notch(a, f0=440.0 * 2 ** (20 / 1200))      # A + 20 cents
    b["plan_inputs"]["key"] = {"tonic": "D"}             # fifth of D is A
    _, rows = planner.plan_notches3(b)
    assert not rows[0]["rendered"] and "tonic or fifth" in rows[0]["reason"]
    b["plan_inputs"]["key"] = {"tonic": "C"}
    _, rows = planner.plan_notches3(b)
    assert rows[0]["rendered"]


def test_chord_root_guard_is_per_bar(ana):
    _, a = ana
    b = _with_notch(a, f0=440.0, cover=0.5)
    b["plan_inputs"]["key"] = {"tonic": "C"}
    b["pitch"] = {"chords": [{"start": 1.0, "end": 3.0, "chord": "Am"},
                             {"start": 3.0, "end": 5.0, "chord": "F"}]}
    p, rows = planner.plan_notches3(b)
    assert rows[0]["rendered"] and rows[0]["chord_guarded_s"] == pytest.approx(2.0)
    v = p.notches[0].gain_db.values
    # guard applied before smoothing: zero well inside the bar, ramps at its edges
    assert np.all(v[1500:2501] == 0.0) and v[3500] < 0.0
    assert np.max(np.abs(np.diff(v))) <= planner.NOTCH_QUANT_DB + 1e-9
    b["pitch"] = {"chords": [{"start": 0.0, "end": 60.0, "chord": "A"}]}
    p, rows = planner.plan_notches3(b)
    assert not rows[0]["rendered"] and rows[0]["reason"].startswith("no cut")


def test_notch_cap_ranked_by_peak_excess(ana):
    _, a = ana
    b = _with_notch(a)
    base = b["plan_inputs"]["notches"][0]
    b["plan_inputs"]["notches"] = []
    for i in range(planner.NOTCH_MAX + 3):
        r = copy.deepcopy(base)
        r["f0"] = 1000.0 * 1.3 ** i
        r["excess_db"] = np.where(np.asarray(base["excess_db"]) > 0, 8.0 + i, 0.0)
        b["plan_inputs"]["notches"].append(r)
    b["plan_inputs"]["key"] = {}
    _, rows = planner.plan_notches3(b)
    on = [r for r in rows if r["rendered"]]
    assert len(on) == planner.NOTCH_MAX
    assert min(r["peak_excess_db"] for r in on) > max(
        r["peak_excess_db"] for r in rows if not r["rendered"])


def test_sparse_kicks_give_no_mask_and_no_clip(ana):
    _, a = ana
    b = copy.deepcopy(a)
    dur = b["meta"]["n"] / b["meta"]["fs"]
    b["plan_inputs"]["kick_onsets"] = np.array([1.0, 6.0])
    assert planner.usable_kick_onsets(b) == []
    p = planner.plan_from_analysis(b, planner.Controls())
    assert p.notch.kick_mask is None and p.limiter.clip_onsets_s == []
    assert p.limiter.clip_depth_db == 0.0
    b["plan_inputs"]["kick_onsets"] = np.arange(0.25, dur, 0.4687) + 1.23e-4
    on = planner.usable_kick_onsets(b)
    assert len(on) == len(b["plan_inputs"]["kick_onsets"])
    assert np.allclose(np.asarray(on) * 1000, np.round(np.asarray(on) * 1000))


def test_subside_default_when_sub_not_decoherent(ana):
    _, a = ana
    b = copy.deepcopy(a)
    b["plan_inputs"]["side_strip_hz"] = 150.0
    b["plan_inputs"]["band_decoherence"] = [0.05] + list(b["plan_inputs"]["band_decoherence"][1:])
    assert planner.subside_hz(b, planner.Controls()) == planner.SUBSIDE_DEFAULT_HZ
    b["plan_inputs"]["band_decoherence"][0] = 0.3
    assert planner.subside_hz(b, planner.Controls()) == 150.0
    assert planner.subside_hz(b, planner.Controls(subside_hz=80.0)) == 80.0


def test_crest_and_tp_controls(ana):
    _, a = ana
    with pytest.raises(ValueError):
        planner.Controls(crest_mode="floor")
    with pytest.raises(ValueError):
        planner.Controls(crest_mode="maybe")
    with pytest.raises(ValueError):
        planner.Controls(tp_meter="4x")
    c = planner.Controls(crest_mode="floor", crest_floors_db=(8, 10, 12, 12, 12), tp_meter="8x")
    t = planner.targets_from(a, c)
    assert t.crest_mode == "floor" and t.crest_floors_db == [8.0, 10.0, 12.0, 12.0, 12.0]
    assert t.tp_meter == "8x"
    assert planner.targets_from(a, planner.Controls()).tp_meter == "8x"            # S6
    assert planner.targets_from(a, planner.Controls(streaming=True)).tp_meter == "worst"
    # without --xover the floors are counted against the analysis's crossovers (audit 08)
    three = {"meta": {"crossovers": [100.0, 400.0, 1600.0]}}
    four = planner.Controls(crest_mode="floor", crest_floors_db=(8, 10, 12, 12))
    assert planner.resolve(three, four).crest_floors_db == (8.0, 10.0, 12.0, 12.0)
    with pytest.raises(ValueError, match="one value per band"):
        planner.resolve(three, c)
    with pytest.raises(ValueError, match="one value per band"):
        planner.Controls(crossovers_hz=(100.0, 400.0), crest_floors_db=(1, 2))


def test_q_only_when_fit_ok():
    assert planner.q_fit_ok(10.0, 4.9) and not planner.q_fit_ok(10.0, 5.0)
    assert not planner.q_fit_ok(float("nan"), 1.0) and not planner.q_fit_ok(10.0, float("inf"))
    assert planner._notch_q({"q": 10.0, "q_uncertainty": 6.0}) == planner.NOTCH_Q_DEFAULT
    assert planner._notch_q({"q": 10.0, "q_uncertainty": 2.0}) == 10.0


def test_glue_threshold_from_glue_input():
    fs = 48000
    t = np.arange(fs * 4) / fs
    y = np.stack([0.5 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1)
    thr = planner.glue_threshold_audio(y, fs)
    assert thr == pytest.approx(20 * np.log10(0.5) - 6.0, abs=1e-3)
    assert planner.glue_threshold_audio(np.zeros((fs, 2)), fs) == 0.0


@pytest.mark.slow
def test_glue_unrealised_threshold_is_set_at_realise(ana):
    x, a = ana
    p0 = planner.plan_from_analysis(a, planner.Controls())
    assert p0.glue.threshold_db is None
    p = planner.realise(p0, x, FS)
    assert p.glue.threshold_db is not None and p.glue.threshold_db <= 0.0
    assert float(np.min(p.glue.gain_db.values)) < 0.0          # the glue acts


def test_merge_then_cap_reports_merged(ana):
    _, a = ana
    b = _with_notch(a)
    base = b["plan_inputs"]["notches"][0]
    b["plan_inputs"]["key"] = {}
    rows_in = []
    for i in range(planner.NOTCH_MAX + 2):
        for df in (1.0, 1.02):                 # each pair overlaps: merges to one notch
            r = copy.deepcopy(base)
            r["f0"] = 1000.0 * 1.5 ** i * df
            r["excess_db"] = np.where(np.asarray(base["excess_db"]) > 0, 8.0 + i, 0.0)
            rows_in.append(r)
    b["plan_inputs"]["notches"] = rows_in
    p, rows, merged = planner.plan_notches(b)
    assert len(p.notches) == len(merged) == planner.NOTCH_MAX     # cap after merging
    assert all(len(m["members"]) == 2 for m in merged)
    on = [r for r in rows if r["rendered"]]
    assert len(on) == 2 * planner.NOTCH_MAX
    assert {r["merged_into"] for r in on} == set(range(planner.NOTCH_MAX))


# The maintainer's own renders (gitignored, skipped when absent): features.npz, and the
# features.pkl caches written before it, which are pickle-loaded. Those are his files on his
# machine, so the pickle is acceptable here; drop the .pkl branch once they are regenerated.
REAL = sorted(Path(__file__).resolve().parents[1].glob("work/renders/*/features.[np][pk][zl]"))


@pytest.mark.skipif(not REAL, reason="no real analysis in work/renders")
@pytest.mark.slow
def test_guards_reject_on_a_real_analysis():
    """audit 07 F10: the tonic / fifth and chord-root guards parse what the real
    analysis provides and reject or zero at least one candidate somewhere."""
    from eluate import cli
    hits = 0
    for f in REAL:
        if f.suffix == ".npz":
            a = cli._load_analysis(f.parent)
        else:                                  # the maintainer's caches from before features.npz
            with open(f, "rb") as fh:
                a = pickle.load(fh)
        assert planner.pitch_class(a["plan_inputs"]["key"]["tonic"]) is not None
        assert planner.bar_roots(a), f
        _, rows, _ = planner.plan_notches(a)
        hits += sum(1 for r in rows if "tonic or fifth" in r["reason"] or r["chord_guarded_s"] > 0)
    assert hits >= 1


# ---------------------------------------------------------------- input EQ (chunk 2)

def _ltas_analysis(slope=-4.5, bump=None, top=0.0, sub=0.0, fs=FS):
    """Minimal analysis for the input EQ fit: a stereo LTAS on the tilt module's 1/3-octave
    centres, a power law of `slope` dB/oct, plus a bell-shaped bump (dB, Hz) in the low
    mids, `top` dB above 10 kHz and `sub` dB below 90 Hz."""
    from eluate.features import tilt
    c = tilt.third_octave_centres(fs)
    L = -30.0 + slope * np.log2(c / 1000.0)
    if bump is not None:
        g, f0 = bump
        L = L + g * np.exp(-0.5 * (np.log2(c / f0) / 0.4) ** 2)
    L = np.where(c >= 10000.0, L + top, L)
    L = np.where(c <= 90.0, L + sub, L)
    s, i = tilt.fit_slope(c, L)
    sel = (c >= 30.0) & (c <= 90.0)
    shelf = float(np.mean(L[sel] - (i + s * np.log2(c[sel]))))
    v = {"M": float(s), "S": float(s), "stereo": float(s)}
    return {"meta": {"fs": fs},
            "plan_inputs": {"tilt_slope_db_oct": v, "sub_shelf_db": {"stereo": shelf},
                            "ltas_centres_hz": c, "ltas_stereo_db": L}}


def test_inputeq_ms_top_end_boosts_a_darker_side_more():
    """Maintainer, 2026-10-20: a dark top end gets a mid shelf (up to +4.5) and its own side
    shelf (up to +6); a side darker than the mid up top gets the bigger boost."""
    a = _ltas_analysis(slope=-6.0)
    c = np.asarray(a["plan_inputs"]["ltas_centres_hz"])
    L = np.asarray(a["plan_inputs"]["ltas_stereo_db"])
    # side: 8 dB lower overall, and a further -1.5 dB/oct above 2 kHz
    S = L - 8.0 - np.where(c > 2000.0, 1.5 * np.log2(c / 2000.0), 0.0)
    a["plan_inputs"]["ltas_side_db"] = S
    eq, rec = planner.plan_inputeq(a, planner.Controls(), FS)
    assert eq.ms_top and eq.high_shelf is not None
    top = lambda *s: sum(x.gain_db for x in s if x is not None)    # noqa: E731
    m, sd = top(eq.high_shelf, eq.air_shelf), top(eq.high_shelf_side, eq.air_shelf_side)
    assert 0.0 < m and m < sd
    for x in (eq.high_shelf, eq.air_shelf):
        assert x is None or x.gain_db <= 4.5
    for x in (eq.high_shelf_side, eq.air_shelf_side):
        assert x is None or x.gain_db <= 6.0
    assert any(" S " in t and t.endswith("(6 dB/oct)") for t in rec["text"])
    # the solver's refit keeps the side shelves
    assert planner.refit_inputeq(eq, FS, 0.3).ms_top
    # no side data: linked, as before
    eq2, _ = planner.plan_inputeq(_ltas_analysis(slope=-6.0), planner.Controls(), FS)
    assert not eq2.ms_top
    # a near-silent side: linked
    a["plan_inputs"]["ltas_side_db"] = L - 40.0
    assert not planner.plan_inputeq(a, planner.Controls(), FS)[0].ms_top
    # T4: a decoherent top band gets no side boost
    a["plan_inputs"]["ltas_side_db"] = S
    a["plan_inputs"]["band_decoherence"] = [0.0, 0.1, 0.2, 0.3, 0.6]
    e3, r3 = planner.plan_inputeq(a, planner.Controls(), FS)
    assert top(e3.high_shelf_side, e3.air_shelf_side) <= 0.0
    assert r3["side_caps"]["air"]["reason"] == "top already decoherent"


def test_inputeq_flat_target_needs_almost_nothing():
    eq, rec = planner.plan_inputeq(_ltas_analysis(), planner.Controls(), FS)
    assert eq.highpass.fc_hz == 25.0 and eq.highpass.q == pytest.approx(0.71)
    assert eq.bell is None
    for s in (eq.low_shelf, eq.high_shelf):
        assert s is None or abs(s.gain_db) <= 0.2
    assert rec["residual_rms_db"] < 0.1
    assert rec["text"][0] == "high-pass 24 dB/oct at 25 Hz, Q 0.71"


def test_inputeq_bell_only_above_the_low_mid_gate():
    """T3: the bell fires on an excess of more than EQ_BELL_GATE_DB over its neighbours
    (not over the fit line) between 100 and 500 Hz, and cuts no more than the excess."""
    eq, rec = planner.plan_inputeq(_ltas_analysis(bump=(5.0, 250.0)), planner.Controls(), FS)
    assert rec["low_mid_excess_db"] > planner.EQ_BELL_GATE_DB and rec["bell_considered"]
    assert eq.bell is not None and -rec["low_mid_excess_db"] - 0.05 <= eq.bell.gain_db < -1.5
    assert 180.0 <= eq.bell.fc_hz <= 350.0
    assert any(t.startswith("bell -") and "Hz" in t for t in rec["text"])
    eq1, rec1 = planner.plan_inputeq(_ltas_analysis(bump=(3.0, 250.0)), planner.Controls(), FS)
    assert rec1["low_mid_excess_db"] < planner.EQ_BELL_GATE_DB      # spread over its neighbours
    assert eq1.bell is None and not rec1["bell_considered"]
    eq2, _ = planner.plan_inputeq(_ltas_analysis(bump=(5.0, 800.0)), planner.Controls(), FS)
    assert eq2.bell is None                      # above the low-mid range (T3: to 500 Hz)


def test_inputeq_top_shelves_from_the_top_end_deficit():
    """T12: a top-end deficit above 10 kHz is the air shelf's (corner 10 to 16 kHz); its
    boost is capped by the deficit in its own range (T4)."""
    eq, rec = planner.plan_inputeq(_ltas_analysis(top=-4.0), planner.Controls(), FS)
    assert eq.air_shelf is not None and eq.air_shelf.gain_db > 1.0
    assert 10000.0 <= eq.air_shelf.fc_hz <= 16000.0
    assert eq.air_shelf.gain_db <= rec["top_caps"]["air"]["cap_db"] + 1e-9
    eq2, _ = planner.plan_inputeq(_ltas_analysis(top=3.0), planner.Controls(), FS)
    tops = [s.gain_db for s in (eq2.high_shelf, eq2.air_shelf) if s is not None]
    assert tops and min(tops) < 0.0


def test_inputeq_air_boost_stays_under_the_noise_headroom():
    """T4: with a measured quiet-passage floor, the presence and air boosts together keep
    the master's floor under noise.FLAG_DB."""
    from eluate import noise
    a = _ltas_analysis(top=-6.0)
    gain = planner.quiet_gain_db(planner.Controls())
    assert gain == pytest.approx(-4.1 + 18.0 - 3.0103 + 2.5 + 1.5 + 1.0 + 3.0)
    a["plan_inputs"]["noise_floor_db"] = noise.FLAG_DB - gain - 1.0       # 1 dB of headroom
    eq, rec = planner.plan_inputeq(a, planner.Controls(), FS)
    assert rec["top_caps"]["noise_headroom_db"] == pytest.approx(1.0, abs=1e-6)
    boost = sum(max(0.0, s.gain_db) for s in (eq.high_shelf, eq.air_shelf) if s is not None)
    assert boost <= 1.0 + 1e-9
    assert rec["top_caps"]["air"]["reason"] == "noise headroom"
    # the side's pair is held to the same headroom (audit 13 #1)
    c = np.asarray(a["plan_inputs"]["ltas_centres_hz"])
    L = np.asarray(a["plan_inputs"]["ltas_stereo_db"])
    a["plan_inputs"]["ltas_side_db"] = L - 8.0 - np.where(c > 2000.0, 1.5 * np.log2(c / 2000.0), 0.0)
    es, _ = planner.plan_inputeq(a, planner.Controls(), FS)
    assert es.ms_top
    assert sum(max(0.0, s.gain_db) for s in (es.high_shelf_side, es.air_shelf_side)
               if s is not None) <= 1.0 + 1e-9
    del a["plan_inputs"]["ltas_side_db"]
    # a quiet-passage level already past the flag (music in the breakdowns) sets no cap
    a["plan_inputs"]["noise_floor_db"] = noise.FLAG_DB - gain + 3.0
    eq2, rec2 = planner.plan_inputeq(a, planner.Controls(), FS)
    assert rec2["top_caps"]["noise_headroom_db"] is None
    assert sum(max(0.0, s.gain_db) for s in (eq2.high_shelf, eq2.air_shelf) if s is not None) > 2.0


def test_inputeq_hpf_q_from_the_sub_excess():
    assert planner.hpf_q(0.0) == pytest.approx(0.71) and planner.hpf_q(3.0) == pytest.approx(0.71)
    assert planner.hpf_q(6.0) == pytest.approx(0.6, abs=0.01)
    assert planner.hpf_q(9.0) == 0.5 and planner.hpf_q(20.0) == 0.5
    eq, rec = planner.plan_inputeq(_ltas_analysis(sub=6.0), planner.Controls(), FS)
    assert rec["sub_shelf_predicted_db"] > planner.SUB_SHELF_MAX_DB
    assert eq.highpass.q < 0.65
    # the lowered Q and the low shelf together bring the sub back toward the window
    assert eq.low_shelf is not None and eq.low_shelf.gain_db < 0.0
    nodes = np.array(rec["fit_nodes_hz"])
    sub = nodes < 100.0
    assert np.mean(np.array(rec["fit_db"])[sub]) == pytest.approx(
        np.mean(np.array(rec["fit_deficit_db"])[sub]), abs=0.5)


def test_inputeq_bounds_rounding_and_record():
    for kw in ({"slope": -7.5}, {"slope": -1.5}, {"bump": (8.0, 400.0)}, {"sub": 12.0},
               {"top": -12.0}):
        eq, rec = planner.plan_inputeq(_ltas_analysis(**kw), planner.Controls(), FS)
        for s, top in ((eq.low_shelf, 3.0), (eq.bell, 3.0), (eq.high_shelf, 4.5),
                       (eq.high_shelf_side, 6.0), (eq.air_shelf, 4.5), (eq.air_shelf_side, 6.0)):
            if s is not None:
                assert -3.0 <= s.gain_db <= top and s.gain_db == round(s.gain_db, 1)
                assert s.fc_hz == round(s.fc_hz)
        if eq.low_shelf is not None:
            assert 60.0 <= eq.low_shelf.fc_hz <= 250.0
        if eq.bell is not None:
            assert -3.0 <= eq.bell.gain_db < 0.0 and 0.7 <= eq.bell.q <= 4.0
        assert 0.5 <= eq.highpass.q <= 0.71                     # T11: never above
        assert len(rec["fit_db"]) == len(rec["fit_nodes_hz"]) == len(rec["fit_deficit_db"])
        assert rec["fit_nodes_hz"][0] >= 30.0 and rec["fit_nodes_hz"][-1] <= 18000.0
        assert rec["residual_max_db"] >= rec["residual_rms_db"] >= 0.0
        assert eq.design is not None and set(rec["sections"]) == {
            "highpass", "low_shelf", "bell", "band_shelf", "high_shelf", "high_shelf_side",
            "air_shelf", "air_shelf_side"}
    a = _ltas_analysis(slope=-3.5)
    assert planner.plan_inputeq(a, planner.Controls(preserve_mix=True), FS)[0] is None
    e1, _ = planner.plan_inputeq(a, planner.Controls(), FS)
    e2, _ = planner.plan_inputeq(a, planner.Controls(), FS)
    assert e1 == e2                                # deterministic
    assert planner.refit_inputeq(e1, FS, 0.0) == e1
    et, rt = planner.plan_inputeq(a, planner.Controls(tilt_db=0.5), FS)
    assert rt["target_slope"] == pytest.approx(-4.0)
    assert rt["predicted_slope_db_oct"] == pytest.approx(-4.0, abs=0.15)


def test_inputeq_tilt_error_corrected_on_pink_noise():
    """Done-criterion of chunk 2: pink-ish noise with a known tilt error comes out of the
    input EQ alone within +-0.3 dB/oct of the target slope (features/tilt.py reading)."""
    from fixtures import synthetic as syn
    from eluate.features import tilt
    from eluate.stages import inputeq
    n = 20 * FS
    for err in (-0.6, 0.6):
        x = syn.noise_psd_slope(n, FS, planner.REFERENCE_SLOPE + err, rms_dbfs=-20, seed=3,
                                correlation=0.5)
        t = tilt.analyse(x, FS)
        assert t["slope_db_oct"]["stereo"] == pytest.approx(planner.REFERENCE_SLOPE + err, abs=0.1)
        a = {"meta": {"fs": FS},
             "plan_inputs": {"tilt_slope_db_oct": t["slope_db_oct"],
                             "sub_shelf_db": t["sub_shelf_db"], "ltas_centres_hz": t["centres"],
                             "ltas_stereo_db": t["ltas_db"]["stereo"]}}
        eq, rec = planner.plan_inputeq(a, planner.Controls(), FS)
        out = tilt.analyse(inputeq.apply(x, FS, eq), FS)["slope_db_oct"]["stereo"]
        print(f"tilt error {err:+.1f}: {t['slope_db_oct']['stereo']:.2f} -> {out:.2f} dB/oct "
              f"with {rec['text']}")
        assert abs(out - planner.REFERENCE_SLOPE) <= 0.3
        assert out == pytest.approx(rec["predicted_slope_db_oct"], abs=0.1)


def test_plan_record_carries_the_input_eq(ana):
    _, a = ana
    rec = planner.plan_record(a, planner.Controls())
    eq = rec["inputeq"]
    assert eq["text"] and eq["text"][0].startswith("high-pass 24 dB/oct at 25 Hz")
    import json
    json.dumps(eq)


def test_low_notch_q_and_onset():
    """R7e, T7: below 500 Hz a notch's Q is at most 4, and its cut does not start before
    the resonance does, despite the analysis's ~3 s centred smoothing of the excess."""
    assert planner._notch_q({"f0": 120.0, "q": 16.0, "q_uncertainty": 0.1}) == 4.0
    assert planner._notch_q({"f0": 2000.0, "q": 16.0, "q_uncertainty": 0.1}) == 16.0
    hop, fs = 0.05, 44100.0
    t = np.arange(400) * hop                       # 20 s grid, a resonance from 8 to 14 s
    raw = np.where((t >= 8.0) & (t < 14.0), 20.0, 0.0)
    S = 3.0
    k = np.ones(int(S / hop)) / int(S / hop)
    ex = np.convolve(raw, k, mode="same")           # centred, as resonance.py smooths it
    n = int(20 * fs)
    g = planner._notch_envelope(ex, hop, 1.0, n, fs, erode_s=S)
    on = np.flatnonzero(g < 0)[0] / planner.CONTROL_RATE_HZ
    assert 8.0 - 0.1 <= on <= 8.6
    g0 = planner._notch_envelope(ex, hop, 1.0, n, fs)
    assert np.flatnonzero(g0 < 0)[0] / planner.CONTROL_RATE_HZ < 7.6     # the old lead
