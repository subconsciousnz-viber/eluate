"""Stems, phase B corrections (stemfix): the plan / apply / describe contract on synthetic
stems."""
import json

import numpy as np
import pytest
from scipy import signal

from eluate import stemfix

FS = 44100
BPM = 120.0
BEAT = 60.0 / BPM
DUR = 4.0
N = int(DUR * FS)
ONSETS = np.arange(0.25, DUR - 0.3, BEAT)


def _stereo(m):
    return np.stack([m, m], axis=1)


def _set():
    """kick, bass, pad, hat (neutral names are given per test); the bass sustains under
    every kick."""
    t = np.arange(N) / FS
    rng = np.random.default_rng(3)
    kick = np.zeros(N)
    for o in ONSETS:
        i = int(round(o * FS))
        tt = np.arange(int(0.25 * FS)) / FS
        k = 0.8 * np.sin(2 * np.pi * (50 * tt + 60 * 0.03 * (1 - np.exp(-tt / 0.03)))) * np.exp(-tt / 0.08)
        kick[i:i + k.size] += k[:N - i]
    bass = 0.3 * np.sin(2 * np.pi * 41.0 * t) + 0.05 * np.sin(2 * np.pi * 82.0 * t)
    pad = sum(0.05 * np.sin(2 * np.pi * f * t + f) for f in (330.0, 440.0, 660.0, 1320.0, 2640.0))
    noise = signal.sosfilt(signal.butter(4, 5000, "highpass", fs=FS, output="sos"),
                           rng.standard_normal(N))
    gate = np.zeros(N)
    for o in ONSETS + BEAT / 2:
        i = int(round(o * FS))
        gate[i:i + int(0.05 * FS)] = 1.0
    hat = 0.2 * noise * gate
    return np.stack([_stereo(kick), _stereo(bass), _stereo(pad), _stereo(hat)])


NAMES = ["one", "two", "three", "four"]       # kick, bass, pad, hat: no name hints


@pytest.fixture(scope="module")
def S():
    return _set()


def _rms_db(x):
    return 10 * np.log10(np.mean(x ** 2))


def _eq_rep(section="high shelf", gain=2.5, lo=1340.0, hi=20000.0, rows=None):
    return {"stems": NAMES, "attribution": [{
        "limit": f"input EQ {section} {gain:+.1f} dB, at its bound",
        "detail": {"gain_db": gain, "bound_db": abs(gain), "range_hz": [lo, hi]},
        "stems": rows or [{"name": "three", "share": 0.6, "level_re_sum_db": -3.0},
                          {"name": "four", "share": 0.1, "level_re_sum_db": -9.0}]}]}


def _lim_rep(names_rows, moments):
    return {"stems": NAMES, "attribution": [{
        "limit": "limiter on its outlier cap", "detail": {"moments_s": list(moments)},
        "stems": [{"name": n, "peak_db": p, "below_sum_db": 3.0} for n, p in names_rows]}]}


def test_empty_rep_gives_no_fixes(S):
    assert stemfix.plan({"stems": NAMES, "attribution": []}, S, NAMES, FS) == []
    assert stemfix.plan({}, S, NAMES, FS) == []
    prior = [{"stem": "three", "kind": "eq", "reason": "x",
              "params": {"shape": "high_shelf", "fc_hz": 1340.0, "q": None, "channel": "LR",
                         "gain_db": 1.5}}]
    assert stemfix.plan({"attribution": []}, S, NAMES, FS, prior=prior) == prior


def test_roles_by_content(S):
    assert stemfix.roles(S, NAMES, FS) == (0, 1)
    on, holds = stemfix.kick_onsets(S[0], FS)
    assert on.size == ONSETS.size                       # one per kick, none on its ripple
    assert np.max(np.abs(on / 1000.0 - ONSETS)) < 0.003


def test_contract_shape_and_serialisable(S):
    rep = _eq_rep()
    rep["attribution"] += _lim_rep([("two", -6.0), ("four", -15.0)], ONSETS[:4] + 0.02)["attribution"]
    rep["attribution"].append({"limit": "notch at 660 Hz (found)",
                               "detail": {"f0_hz": 660.0, "spans": [[0.0, 2.0]]},
                               "stems": [{"name": "three", "excess_db": 8.0}]})
    fixes = stemfix.plan(rep, S, NAMES, FS, bpm=BPM)
    assert json.loads(json.dumps(fixes)) == fixes
    kinds = {(f["stem"], f["kind"]) for f in fixes}
    assert ("three", "eq") in kinds and ("two", "duck") in kinds
    assert all(set(f) == {"stem", "kind", "params", "reason"} for f in fixes)
    assert all(f["stem"] in ("two", "three") for f in fixes)   # only named stems
    for f in fixes:
        line = stemfix.describe(f)
        assert line.startswith(f["stem"] + ": ") and f["reason"] in line
    shelf = next(f for f in fixes if f["params"]["shape"] == "high_shelf")
    assert "+1.5 dB at 1.34 kHz" in stemfix.describe(shelf)


def test_caps_hold_across_passes(S):
    rep = _eq_rep()
    rep["attribution"] += _lim_rep([("two", -6.0)], ONSETS[:4] + 0.02)["attribution"]
    rep["attribution"] += _lim_rep([("three", -6.0)], [1.0, 2.0])["attribution"]
    p1 = stemfix.plan(rep, S, NAMES, FS, bpm=BPM)
    p2 = stemfix.plan(rep, S, NAMES, FS, prior=p1, bpm=BPM)
    p3 = stemfix.plan(rep, S, NAMES, FS, prior=p2, bpm=BPM)
    get = lambda fx, st, k: next(f["params"] for f in fx if f["stem"] == st and f["kind"] == k)
    assert get(p1, "three", "eq")["gain_db"] == pytest.approx(1.5)
    assert get(p2, "three", "eq")["gain_db"] == pytest.approx(stemfix.EQ_MAX_DB)
    assert get(p2, "two", "duck")["depth_db"] == pytest.approx(stemfix.DUCK_MAX_DB)
    assert get(p2, "three", "comp")["max_gr_db"] == pytest.approx(stemfix.COMP_MAX_DB)
    assert len(p1) == len(p2)
    # the duck at its cap: a further hit on the bass compresses it instead, up to its cap
    assert get(p3, "two", "comp")["max_gr_db"] == pytest.approx(stemfix.COMP_STEP_DB)
    p4 = stemfix.plan(rep, S, NAMES, FS, prior=p3, bpm=BPM)
    p5 = stemfix.plan(rep, S, NAMES, FS, prior=p4, bpm=BPM)
    assert p5 == p4                                     # every fix at its cap: nothing more
    assert get(p3, "three", "eq")["gain_db"] == pytest.approx(stemfix.EQ_MAX_DB)


def test_wanted_gain_sizes_the_shelf(S):
    rep = _eq_rep()
    rep["attribution"][0]["detail"]["wanted_db"] = 2.8
    g = stemfix.plan(rep, S, NAMES, FS)[0]["params"]["gain_db"]
    # 1 + (g - 1) 0.6 = 10^(0.3/20): about +0.57 dB on the one stem with share >= 0.2
    assert g == pytest.approx(20 * np.log10(1 + (10 ** (0.3 / 20) - 1) / 0.6), abs=0.01)


def test_apply_untouched_and_deterministic(S):
    rep = _eq_rep()
    rep["attribution"] += _lim_rep([("two", -6.0)], ONSETS[:4] + 0.02)["attribution"]
    rep["attribution"] += _lim_rep([("four", -6.0)], [1.0, 2.0])["attribution"]
    fixes = stemfix.plan(rep, S, NAMES, FS, bpm=BPM)
    before = S.copy()
    a = stemfix.apply(S, FS, NAMES, fixes)
    b = stemfix.apply(S, FS, NAMES, fixes)
    assert np.array_equal(S, before)
    assert a is not S and a.shape == S.shape
    assert np.array_equal(a, b)
    assert np.array_equal(a[0], S[0])                   # the kick is not named: untouched
    assert not np.array_equal(a[1], S[1])
    assert np.array_equal(stemfix.apply(S, FS, NAMES, []), S)


def test_ducking_only_at_kick_onsets(S):
    fx = [{"stem": "two", "kind": "duck", "reason": "r",
           "params": {"kick": "one", "depth_db": 3.0, "release_ms": stemfix._release_ms(BPM),
                      "lookahead_ms": 5.0}}]
    y = stemfix.apply(S, FS, NAMES, fx)
    at, away = [], []
    for o in ONSETS:
        i = int(round(o * FS))
        a = slice(i + int(0.002 * FS), i + int(0.015 * FS))
        w = slice(i + int(0.30 * FS), i + int(0.45 * FS))
        at.append(_rms_db(y[1][a]) - _rms_db(S[1][a]))
        away.append(_rms_db(y[1][w]) - _rms_db(S[1][w]))
    assert max(at) < -2.5 and min(at) > -3.05
    assert max(abs(v) for v in away) < 0.01
    # no bass under the kicks: nothing to duck
    T = S.copy()
    T[1, : N // 2] = 0.0
    z = stemfix.apply(T, FS, NAMES, fx)
    assert np.array_equal(z[1, : N // 2], T[1, : N // 2])


def test_limiter_on_bass_ducks_at_kicks_else_compresses(S):
    on_kicks = stemfix.plan(_lim_rep([("two", -6.0)], ONSETS + 0.02), S, NAMES, FS, bpm=BPM)
    assert [(f["stem"], f["kind"]) for f in on_kicks] == [("two", "duck")]
    off = stemfix.plan(_lim_rep([("two", -6.0)], ONSETS + BEAT / 2), S, NAMES, FS, bpm=BPM)
    assert [(f["stem"], f["kind"]) for f in off] == [("two", "comp")]
    y = stemfix.apply(S, FS, NAMES, off)
    gr = 20 * np.log10(np.max(np.abs(S[1])) / np.max(np.abs(y[1])))
    assert 0.5 < gr <= stemfix.COMP_STEP_DB + 0.05


def test_eq_moves_the_band_the_right_way(S):
    for gain, sign in ((2.5, 1), (-1.5, -1)):
        fixes = stemfix.plan(_eq_rep(gain=gain), S, NAMES, FS)
        assert [f["stem"] for f in fixes] == ["three"]   # "four" carries too little
        y = stemfix.apply(S, FS, NAMES, fixes)
        f, p0 = signal.welch(S[2][:, 0], FS, nperseg=8192)
        _, p1 = signal.welch(y[2][:, 0], FS, nperseg=8192)
        hi = (f >= 2000) & (f < 3000)
        lo = (f >= 300) & (f < 500)
        d_hi = 10 * np.log10(p1[hi].sum() / p0[hi].sum())
        d_lo = 10 * np.log10(p1[lo].sum() / p0[lo].sum())
        assert sign * d_hi > 0.8 and abs(d_lo) < 0.3
    low = stemfix.plan(_eq_rep("low shelf", -1.5, 20.0, 120.0,
                               rows=[{"name": "two", "share": 0.9}]), S, NAMES, FS)
    y = stemfix.apply(S, FS, NAMES, low)
    assert _rms_db(y[1]) < _rms_db(S[1]) - 0.8


def test_notch_found_cut_rendered_skipped_crest_none(S):
    ent = {"limit": "notch at 660 Hz (found)", "detail": {"f0_hz": 660.0, "spans": [[0.0, 1.0]]},
           "stems": [{"name": "three", "excess_db": 8.0}, {"name": "four", "excess_db": 1.0}]}
    fixes = stemfix.plan({"attribution": [ent]}, S, NAMES, FS)
    assert len(fixes) == 1 and fixes[0]["params"]["shape"] == "notch"
    assert fixes[0]["params"]["gain_db"] == pytest.approx(-stemfix.NOTCH_STEP_DB)
    y = stemfix.apply(S, FS, NAMES, fixes)
    f, p0 = signal.welch(S[2][:, 0], FS, nperseg=16384)
    _, p1 = signal.welch(y[2][:, 0], FS, nperseg=16384)
    k = np.argmin(np.abs(f - 660.0))
    assert 10 * np.log10(p1[k] / p0[k]) == pytest.approx(-1.5, abs=0.2)
    rendered = dict(ent, limit="notch at 660 Hz (rendered)")
    crest = {"limit": "crest low or section contrast lost", "detail": {},
             "stems": [{"name": "two", "crest_db": 6.0}]}
    assert stemfix.plan({"attribution": [rendered, crest]}, S, NAMES, FS) == []


def test_silent_stem_never_fixed(S):
    T = S.copy()
    T[3] = 0.0
    rep = _eq_rep(rows=[{"name": "four", "share": 0.5}, {"name": "three", "share": 0.5}])
    fixes = stemfix.plan(rep, T, NAMES, FS)
    assert [f["stem"] for f in fixes] == ["three"]
    # the JSON report's form: a silent stem's infinities read as None
    rep2 = _lim_rep([("four", None)], [1.0])
    assert stemfix.plan(rep2, T, NAMES, FS) == []


def test_apply_sum_matches_apply(S):
    rep = _eq_rep()
    rep["attribution"] += _lim_rep([("two", -6.0)], ONSETS[:4] + 0.02)["attribution"]
    rep["attribution"] += _lim_rep([("four", -6.0)], [1.0, 2.0])["attribution"]
    fixes = stemfix.plan(rep, S, NAMES, FS, bpm=BPM)
    before = S.copy()
    tot = stemfix.apply_sum(S, FS, NAMES, fixes)
    assert tot.shape == S.shape[1:] and np.array_equal(S, before)
    assert np.allclose(tot, stemfix.apply(S, FS, NAMES, fixes).sum(axis=0), atol=1e-12)


def test_detail_section_and_wanted(S):
    rep = _eq_rep(section="presence", gain=2.5)            # unparseable name: detail wins
    rep["attribution"][0]["detail"].update(section="high_shelf", wanted_db=2.8)
    p = stemfix.plan(rep, S, NAMES, FS)[0]["params"]
    assert p["shape"] == "high_shelf" and p["fc_hz"] == 1340.0
    assert p["gain_db"] == pytest.approx(20 * np.log10(1 + (10 ** (0.3 / 20) - 1) / 0.6), abs=0.01)


def test_notch_channel_q_and_musical_cap(S):
    T = S.copy()
    T[2, :, 1] *= 0.5                                     # the pad gets a side
    ent = {"limit": "notch at 660 Hz (found)",
           "detail": {"f0_hz": 660.0, "spans": [[0.0, 1.0]], "channel": "S", "q": 6.0,
                      "flag": "musical", "max_cut_db": 1.0},
           "stems": [{"name": "three", "excess_db": 8.0}]}
    p1 = stemfix.plan({"attribution": [ent]}, T, NAMES, FS)
    p2 = stemfix.plan({"attribution": [ent]}, T, NAMES, FS, prior=p1)
    assert p1[0]["params"]["gain_db"] == pytest.approx(-1.0)   # the musical cap, not 1.5
    assert p2 == p1
    assert p1[0]["params"]["q"] == 6.0 and p1[0]["params"]["channel"] == "S"
    assert "on S" in stemfix.describe(p1[0])
    y = stemfix.apply(T, FS, NAMES, p1)[2]
    m0, s0 = T[2].sum(axis=1) / 2, (T[2][:, 0] - T[2][:, 1]) / 2
    m1, s1 = y.sum(axis=1) / 2, (y[:, 0] - y[:, 1]) / 2
    assert np.allclose(m1, m0, atol=1e-12)
    f, a = signal.welch(s0, FS, nperseg=16384)
    _, b = signal.welch(s1, FS, nperseg=16384)
    k = np.argmin(np.abs(f - 660.0))
    assert 10 * np.log10(b[k] / a[k]) == pytest.approx(-1.0, abs=0.15)
    # not musical: the 3 dB cap
    ent["detail"]["flag"] = None
    q = stemfix.plan({"attribution": [ent]}, T, NAMES, FS, prior=stemfix.plan({"attribution": [ent]}, T, NAMES, FS))
    assert q[0]["params"]["gain_db"] == pytest.approx(-stemfix.NOTCH_MAX_DB)


def test_a_moved_shelf_corner_deepens_the_same_fix(S):
    # the re-solved sum's shelf corner moves between passes: still one fix per section
    p1 = stemfix.plan(_eq_rep(lo=1340.0), S, NAMES, FS)
    p2 = stemfix.plan(_eq_rep(lo=1420.0), S, NAMES, FS, prior=p1)
    eq1 = [f for f in p1 if f["stem"] == "three" and f["kind"] == "eq"]
    eq2 = [f for f in p2 if f["stem"] == "three" and f["kind"] == "eq"]
    assert len(eq1) == len(eq2) == 1
    assert eq2[0]["params"]["fc_hz"] == eq1[0]["params"]["fc_hz"] == 1340.0
    assert eq2[0]["params"]["gain_db"] > eq1[0]["params"]["gain_db"]
