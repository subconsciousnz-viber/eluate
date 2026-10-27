"""loudness.py, crest.py, stereo.py and transients.py contracts."""

import math

import numpy as np
import pytest

from fixtures import synthetic as syn
from eluate import meter
from eluate.features import borrowed, crest, loudness, stereo, transients
from eluate.features._common import Context, Grid, window_max, window_mean

FS = 48000


def test_grid_convention():
    g = Grid(int(FS * 1.05), FS, 0.1)
    assert g.k == 11
    assert np.allclose(g.times, (np.arange(11) + 0.5) * 0.1)
    assert g.index(0.0) == 0 and g.index(0.0999) == 0 and g.index(0.1) == 1


def test_window_stats_zero_outside():
    sig = np.ones(100)
    c = np.array([0, 50, 99])
    assert np.allclose(window_mean(sig, c, 20), [0.5, 1.0, 0.55])
    assert np.allclose(window_max(sig, c, 20), [1.0, 1.0, 1.0])
    assert window_max(np.r_[np.zeros(50), np.ones(50)], np.array([30]), 20)[0] == 0.0


def test_loudness_series_centred_and_matches_meter():
    """A 1 kHz tone stepping -20 -> -30 dBFS at 5 s: momentary reads meter's
    steady value away from the step, and the step sits at 5 s on the grid."""
    x = syn.tone_sequence([(-20, 5.0), (-30, 5.0)], FS)
    r = loudness.analyse(x, FS)
    t, m = r["t"], r["momentary"]
    ref_hi = meter.integrated(x[: 4 * FS], FS)
    ref_lo = meter.integrated(x[6 * FS:], FS)
    assert np.all(np.abs(m[(t > 1) & (t < 4.7)] - ref_hi) < 0.01)
    assert np.all(np.abs(m[(t > 5.3) & (t < 9.7)] - ref_lo) < 0.01)
    # centred 400 ms windows straddling the step mix the two powers by the time
    # each level occupies: the frame at 4.95 s holds 0.25 s of -20 and 0.15 s of -30
    for tc in (4.85, 4.95, 5.05, 5.15):
        a = np.clip((5.0 - (tc - 0.2)) / 0.4, 0, 1)
        want = 10 * np.log10(a * 10 ** (ref_hi / 10) + (1 - a) * 10 ** (ref_lo / 10))
        k = int(np.argmin(np.abs(t - tc)))
        assert abs(m[k] - want) < 0.05, (tc, m[k], want)
    s = r["short_term"]
    assert np.all(np.abs(s[(t > 2) & (t < 3.4)] - ref_hi) < 0.01)
    assert r["summary"]["integrated_lufs"] == meter.integrated(x, FS)
    assert m.shape == t.shape == s.shape


def test_loudness_edges_and_sections():
    x = syn.tone_sequence([(-20, 4.0)], FS)
    r = loudness.analyse(x, FS)
    assert np.isfinite(r["momentary"][0]) and r["momentary"][0] < r["momentary"][10]
    g = Grid(x.shape[0], FS)
    secs = [{"start": 0.0, "end": 2.0, "label": "a"}, {"start": 2.0, "end": 4.0, "label": "b"},
            {"start": 4.0, "end": 4.0}]
    ps = loudness.per_section(x, FS, secs, r, g)
    assert abs(ps[0]["integrated_lufs"] - meter.integrated(x[: 2 * FS], FS)) < 1e-12
    assert ps[1]["label"] == "b"
    assert ps[2]["integrated_lufs"] == -math.inf


@pytest.mark.parametrize("band,freq", [(0, 50), (1, 250), (2, 1000), (3, 4000), (4, 12000)])
def test_crest_of_a_sine(band, freq):
    """One sine (a sum of two would leak across crossovers and change the crest):
    3.01 dB in M and stereo in its own band; S silent, gated out."""
    t = np.arange(FS * 4) / FS
    y = 0.1 * np.sin(2 * np.pi * freq * t)
    r = crest.analyse(np.stack([y, y], 1), FS)
    assert r["crest_db"].shape == (5, 3, r["t"].size)
    for v in (0, 2):
        c = r["crest_db"][band, v][r["gated"][band, v]]
        assert abs(np.median(c) - 3.0103) < 0.01
        # p95 includes the first / last frames, where the abrupt start and end
        # show the zero-phase split's inherent pre-ring (spec 3.1: reported)
        assert 3.0 < r["p95_db"][band, v] < 3.5
    assert np.isnan(r["p95_db"][band, 1]) and not r["gated"][:, 1].any()


def test_crest_gate_and_impulses():
    x = syn.impulse_train(FS * 3, FS, 0.5, 0.5) + syn.pink_noise(FS * 3, FS, rms_dbfs=-50, seed=1)
    r = crest.analyse(x, FS)
    assert np.nanmax(r["crest_db"][4, 2]) > 15.0  # impulses are peaky in the air band
    q = crest.analyse(1e-5 * syn.pink_noise(FS * 2, FS, seed=2), FS)
    assert not q["gated"].any() and np.all(np.isnan(q["p95_db"]))


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_stereo_correlation_and_ratio(rho):
    x = syn.noise_psd_slope(FS * 20, FS, -3.0103, correlation=rho, seed=3)
    r = stereo.analyse(x, FS)
    # the sub band (10-100 Hz) of 20 s of noise has ~3600 degrees of freedom: sd ~0.017
    assert np.all(np.abs(r["corr_summary"] - rho) < 0.05), r["corr_summary"]
    # equal-power channels: E_S / E_M = (1 - rho) / (1 + rho)
    want = 10 * np.log10((1 - rho) / (1 + rho))
    assert np.all(np.abs(r["sm_ratio_summary_db"] - want) < 0.3)
    assert np.allclose(r["decoherence"], np.clip(1 - r["corr_summary"], 0, 1))


def test_stereo_mono_and_antiphase():
    y = syn.pink_noise(FS * 3, FS, seed=4)[:, 0]
    mono = stereo.analyse(np.stack([y, y], 1), FS)
    assert np.allclose(mono["corr_summary"], 1.0) and np.all(mono["decoherence"] == 0.0)
    assert np.all(mono["sm_ratio_summary_db"] < -100)
    anti = stereo.analyse(np.stack([y, -y], 1), FS)
    assert np.allclose(anti["corr_summary"], -1.0) and np.all(anti["decoherence"] == 1.0)


def test_side_strip_suggestion():
    """Spec 3.3 draft 0.4 (audit 04 A13): default 100 Hz unless the side has
    low-frequency energy within 20 dB of M (and within 60 dB of the LTAS
    maximum); then just above the highest such band, clipped to [60, 150]."""
    c = np.array([31.5, 40, 50, 63, 80, 100, 125, 160, 200])
    m = np.zeros(c.size)
    assert stereo.side_strip_suggestion(m, m - 40.0, c) == 100.0          # clean low side
    assert stereo.side_strip_suggestion(m, m.copy(), c) == 150.0          # uncorrelated stereo
    low = np.where(c <= 80, -10.0, -40.0)
    assert abs(stereo.side_strip_suggestion(m, low, c) - 80 * 2 ** (1 / 6)) < 1e-9
    sub = np.where(c <= 50, -10.0, -40.0)
    assert stereo.side_strip_suggestion(m, sub, c) == 60.0                # clipped up
    quiet = np.full(c.size, -90.0)                                        # S/M -20 dB but 90 dB down
    assert stereo.side_strip_suggestion(m - 70.0, quiet, c, ltas_ref_db=m) == 100.0
    assert stereo.side_strip_suggestion(np.full(c.size, -200.0), m, c) == 100.0


def test_side_strip_on_audio():
    """Mono reads 100 Hz, uncorrelated stereo 150 Hz (before: 150 and 60)."""
    from eluate.features import tilt
    y = syn.pink_noise(FS * 10, FS, seed=2)[:, 0]
    tm = tilt.analyse(np.stack([y, y], 1), FS)
    assert stereo.side_strip_suggestion(tm["ltas_db"]["M"], tm["ltas_db"]["S"], tm["centres"],
                                        tm["ltas_db"]["stereo"]) == 100.0
    tu = tilt.analyse(syn.pink_noise(FS * 10, FS, seed=3), FS)
    assert stereo.side_strip_suggestion(tu["ltas_db"]["M"], tu["ltas_db"]["S"], tu["centres"],
                                        tu["ltas_db"]["stereo"]) == 150.0


def test_unmeasurable_bands_get_finite_defaults():
    """Hard-panned input: decoherence 1.0 (no width makeup) flagged not measurable,
    accepted by plan.MultibandPlan (audit 04 A12); bands with no power get the
    0.5 flatness default, flagged."""
    from eluate.features import tilt
    from eluate.plan import Envelope, MultibandPlan
    x = syn.pink_noise(FS * 6, FS, seed=1)
    x[:, 1] = 0.0
    r = stereo.analyse(x, FS)
    assert np.all(np.isfinite(r["decoherence"])) and not r["decoherence_measurable"].any()
    env = Envelope(np.zeros(4, dtype=np.float32), 1000.0)
    MultibandPlan(gr_db=[[env, env] for _ in range(5)], decoherence=list(r["decoherence"]))
    tl = tilt.analyse(np.zeros((FS * 2, 2)), FS)
    assert np.all(tl["flatness"] == tilt.FLATNESS_DEFAULT) and not tl["flatness_measurable"].any()


@pytest.mark.parametrize("fs", [44100, 48000, 96000])
def test_kick_onsets_on_sidechained_bassline(fs):  # noqa: D103
    """Every kick of kick_sub found once, within 5 ms (the ducked bass makes the
    sub + low energy rise only ~5 dB; see transients docstring)."""
    x, meta = syn.kick_sub(fs, bars=4)
    r = transients.analyse(x, fs)
    k = r["kick_onsets"]
    assert k.size == meta["kick_onsets"].size
    assert np.max(np.abs(k - meta["kick_onsets"])) < 0.005


def test_onsets_rate_and_mask():
    x = syn.impulse_train(FS * 6, FS, 0.25, 0.5, offset_s=0.1) + syn.pink_noise(FS * 6, FS, rms_dbfs=-60)
    r = transients.analyse(x, FS)
    assert abs(r["onsets_per_s"] - 4.0) < 0.5
    assert r["onset_strength"].shape == r["t"].shape
    mid = (r["t"] > 1.5) & (r["t"] < 4.5)
    assert np.all(np.abs(r["onset_rate"][mid] - 4.0) <= 0.5)
    t = np.array([0.95, 1.0, 1.1, 1.2])
    assert list(transients.kick_mask(t, [1.0], 0.04, 0.15)) == [0.0, 1.0, 1.0, 0.0]
    assert transients.kick_mask(t, [], 0.04, 0.15).sum() == 0


def test_context_shares_split():
    x = syn.pink_noise(FS * 2, FS, seed=9)
    ctx = Context(x, FS)
    a = crest.analyse(x, FS, ctx)
    b = crest.analyse(x, FS)
    assert ctx.bands.shape == (5, x.shape[0], 2)
    assert np.array_equal(a["p95_db"], b["p95_db"])


def _onset_cases(fs):
    beat = 60.0 / 128.0
    x = syn.impulse_train(fs * 6, fs, 0.25, 0.5, offset_s=0.1) + syn.pink_noise(fs * 6, fs, rms_dbfs=-60)
    yield "clicks", x, np.arange(0.1, 6, 0.25)
    xk, mk = syn.kick_sub(fs, bars=8)
    yield "kick_sub", xk, mk["kick_onsets"]
    xs, ms = syn.sectioned_track(fs, levels_db=(0.0, -6.0, 0.0))
    yield "kicks+hats", xs, np.sort(np.r_[ms["beat_times"], ms["beat_times"] + beat / 2])


@pytest.mark.parametrize("fs", [44100, 48000])
def test_onset_gate_one_to_one(fs):
    """Onsets (SuperFlux + kick union) matched one-to-one (borrowed.match_onsets,
    20 ms): precision >= 0.95, recall >= 0.99, mean error <= 3 ms (measured: one
    hat of 192 missed at 48 kHz)."""
    for name, x, truth in _onset_cases(fs):
        r = transients.analyse(x, fs)
        m = borrowed.match_onsets(np.round(truth * fs), np.round(r["onset_times"] * fs), fs, 20.0)
        assert m["recall"] >= 0.99 and m["precision"] >= 0.95, (name, m["recall"], m["precision"])
        assert m["mean_err_ms"] <= 3.0, (name, m["mean_err_ms"])


def test_superflux_silent_on_stationary_input():
    t = np.arange(FS * 6) / FS
    ramp = np.minimum(1, np.minimum(t, t[-1] - t) / 0.05)
    for y in (0.3 * np.sin(2 * np.pi * 440 * t) * ramp, syn.pink_noise(FS * 6, FS, seed=4)[:, 0] * ramp):
        on = borrowed.detect_onsets(y, FS)
        assert np.all((on < 0.1 * FS) | (on > 5.9 * FS))  # only at the fades, if at all


def test_kick_rule_rejects_clicks_and_keeps_layered_kicks():
    """Low-band flux candidates pass only with >= 0.3 of their attack below 150 Hz
    (drumbusmaximus DrumIdentity): a click train gives none; kicks struck with a
    loud hat all pass."""
    x = syn.impulse_train(FS * 6, FS, 0.25, 0.5, offset_s=0.1) + syn.pink_noise(FS * 6, FS, rms_dbfs=-60)
    r = transients.analyse(x, FS)
    assert r["kick_onsets"].size == 0 and r["kick_debug"]["candidates"].size > 10
    beat = 60.0 / 128.0
    kk = syn.kick(FS)
    n = int(16 * beat * FS)
    y = np.zeros(n)
    rng = np.random.default_rng(1)
    from scipy import signal
    hp = signal.butter(4, 6000, "highpass", fs=FS, output="sos")
    for i in range(16):
        s = int(i * beat * FS)
        y[s:s + kk.size] += kk[: n - s]
        if i % 2:
            h = signal.sosfilt(hp, rng.standard_normal(int(0.04 * FS))) * np.exp(-np.arange(int(0.04 * FS)) / (0.01 * FS))
            y[s:s + h.size] += 0.6 * h
    r = transients.analyse(np.stack([y, y], 1), FS)
    m = borrowed.match_onsets(np.round(np.arange(16) * beat * FS), np.round(r["kick_onsets"] * FS), FS, 20.0)
    assert m["recall"] == 1.0 and m["precision"] == 1.0
    assert np.all(r["kick_low_share"] >= borrowed.KICK_LOW_SHARE)


@pytest.mark.parametrize("pattern,bpm,kick", [("offbeat", 128.0, True), ("eighths", 140.0, True),
                                              ("beats", 128.0, False), ("offbeat", 128.0, False)])
def test_bass_attacks_are_not_kicks(pattern, bpm, kick):
    """Audit 04 A5: a kick needs a falling pitch (sweep) or a click above 500 Hz;
    low-passed stationary bass attacks are neither. Every true kick scores 1.0."""
    x, meta = syn.bassline(FS, bpm, pattern, kick_on_beats=kick)
    r = transients.analyse(x, FS)
    m = borrowed.match_onsets(np.round(meta["kick_onsets"] * FS), np.round(r["kick_onsets"] * FS), FS, 20.0)
    assert m["extra"] == 0 and m["missed"] == 0, (m["extra"], m["missed"])
    d = r["kick_debug"]
    assert np.all(d["sweep"][~d["kept"]] == 0.0)
    if kick:
        assert np.all(r["kick_score"] >= transients.KICK_MASK_SCORE)


def test_kick_score_shape_evidence():
    """kick_sub kicks carry a sweep; the strict mask list is the kicks scoring >= 0.8."""
    x, meta = syn.kick_sub(FS, bars=4)
    r = transients.analyse(x, FS)
    d = r["kick_debug"]
    assert np.all(d["sweep"][d["kept"]] == 1.0)
    assert np.array_equal(r["kick_mask_onsets"], r["kick_onsets"][r["kick_score"] >= transients.KICK_MASK_SCORE])


def test_non_circular_crest_and_correlation():
    """Fixtures built outside fixtures/synthetic (audit 04 F19): a 50 Hz square wave
    reads a lower crest than a sine in the sub band; L = cos(a) u + sin(a) v,
    R = cos(a) u - sin(a) v from independent Gaussian noise has correlation cos(2a)."""
    t = np.arange(FS * 4) / FS
    sq = 0.2 * np.sign(np.sin(2 * np.pi * 50.0 * t + 0.1))
    r = crest.analyse(np.stack([sq, sq], 1), FS)
    assert r["p95_db"][0, 0] < 3.0
    rng = np.random.default_rng(42)
    u, v = rng.standard_normal((2, FS * 10)) * 0.1
    for a in (0.3, 0.6):
        x = np.stack([np.cos(a) * u + np.sin(a) * v, np.cos(a) * u - np.sin(a) * v], 1)
        s = stereo.analyse(x, FS)
        assert abs(s["full_corr_summary"] - np.cos(2 * a)) < 0.02


def test_loudness_series_on_official_ebu_tone():
    """Non-circular (audit 04 F19): the EBU loudness set's 1 kHz -20 LUFS file reads
    -20 +- 0.1 LUFS in every momentary and short-term frame whose window is
    inside the tone."""
    from pathlib import Path
    from eluate import io
    p = Path(__file__).resolve().parents[1] / "fixtures" / "ebu" / "ebu-loudness-test-setv05" / "1kHz Sine -20 LUFS-16bit.wav"
    if not p.exists():
        pytest.skip("EBU loudness test set not present")
    x, fs = io.read(p)
    r = loudness.analyse(x, fs)
    t = r["t"]
    dur = x.shape[0] / fs
    inner_m = (t > 0.3) & (t < dur - 0.3)
    inner_s = (t > 1.6) & (t < dur - 1.6)
    assert np.all(np.abs(r["momentary"][inner_m] + 20.0) < 0.1)
    assert np.all(np.abs(r["short_term"][inner_s] + 20.0) < 0.1)


def test_side_strip_default_when_the_sub_is_mono():
    """Real-material pass (item 5): stereo low-mids above 100 Hz and a weak 79 Hz
    side component do not move the crossover when the sub band itself is mono
    (S/M -28 dB measured on a premaster that read 150 Hz before)."""
    c = np.array([31.5, 40, 50, 63, 80, 100, 125, 160, 200])
    m = np.zeros(c.size)
    s = np.array([-34, -36, -39, -23, -18, -15, -12, -3, -2.0])
    assert stereo.side_strip_suggestion(m, s, c) == 150.0  # the band rule alone
    assert stereo.side_strip_suggestion(m, s, c, sub_sm_ratio_db=-28.0) == 100.0
    assert stereo.side_strip_suggestion(m, s, c, sub_sm_ratio_db=-12.0) == 150.0
    mids = np.where(c >= 100, -10.0, -40.0)  # side width only above 100 Hz
    assert stereo.side_strip_suggestion(m, mids, c) == 100.0
