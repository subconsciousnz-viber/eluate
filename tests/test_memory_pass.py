"""Memory pass (spec 1, brief-07 revision 3): every streamed or in-place rewrite is
bit-identical to the whole-array formulation it replaced, written out here as the
reference."""

import math

import numpy as np
import pytest
from scipy import signal
from scipy.interpolate import PchipInterpolator

from eluate import bands, envelope, meter
from eluate.plan import EVAL_CHUNK, Envelope, SaturatePlan, control_coord
from eluate.stages import multiband, resample, saturate

RNG = np.random.default_rng(11)


def _ref_split(x, fs, fcs=bands.DEFAULT_CROSSOVERS, order=2):
    x = np.asarray(x, dtype=np.float64)
    x = x[:, None] if x.ndim == 1 else x
    n = x.shape[0]
    pl = bands.padlen(None, fs, min(fcs))
    out, r = [], x
    for fc in fcs:
        lp = signal.butter(order, fc, btype="lowpass", fs=fs, output="sos")
        lo = signal.sosfiltfilt(lp, np.pad(r, ((pl, pl), (0, 0))), axis=0, padtype=None)[pl:pl + n]
        out.append(lo)
        r = r - lo
    out.append(r)
    return np.stack(out)


@pytest.mark.parametrize("n", [2, 999, 48000 * 3 + 7])
def test_split_and_split_iter_match_whole_array_form(n):
    x = RNG.standard_normal((n, 2))
    ref = _ref_split(x, 48000)
    assert np.array_equal(bands.split(x, 48000), ref)
    for k, b in enumerate(bands.split_iter(x, 48000)):
        assert np.array_equal(b, ref[k])
    x0 = x.copy()
    list(bands.split_iter(x, 48000))
    assert np.array_equal(x, x0)                    # the caller's array is never modified


def _ref_up(x, L):
    h = resample.prototype(L)
    c = (h.size - 1) // 2
    z = np.zeros((x.shape[0] * L, x.shape[1]))
    z[::L] = x
    return signal.oaconvolve(z, h[:, None], mode="full", axes=0)[c:c + z.shape[0]]


def _ref_down(y, L):
    h = resample.prototype(L) / L
    c = (h.size - 1) // 2
    return signal.oaconvolve(y, h[:, None], mode="full", axes=0)[c:c + y.shape[0]][::L]


@pytest.mark.parametrize("n", [3, 400, 20011, 300007])
@pytest.mark.parametrize("L", [4, 8])
def test_streamed_resampler_is_bit_identical(n, L):
    x = RNG.standard_normal((n, 2))
    assert np.array_equal(resample.upsample(x, L), _ref_up(x, L))
    y = RNG.standard_normal((n * L + 5, 2))
    assert np.array_equal(resample.decimate(y, L), _ref_down(y, L))
    assert np.array_equal(resample.upsample(x[:, 0], L), _ref_up(x[:, :1], L)[:, 0])


def _ref_tp_env(x, L):
    h = meter._interp_prototype(L, meter._kind(L, None))
    c = meter._centre(h)
    n = x.shape[0]
    env = np.zeros(n)
    for p in range(L):
        hp = h[p::L]
        a = np.max(np.abs(signal.oaconvolve(x, hp[:, None], mode="full", axes=0)), axis=1)
        idx = np.clip((L * np.arange(a.shape[0]) + p - c) // L, 0, n - 1)
        np.maximum.at(env, idx, a)
    return np.maximum(env, np.max(np.abs(x), axis=1))


@pytest.mark.parametrize("n", [1, 2, 9, 5000, 200003])
@pytest.mark.parametrize("L", [4, 8])
def test_true_peak_envelope_matches_maximum_at_form(n, L):
    x = RNG.standard_normal((n, 2)) * 0.5
    assert np.array_equal(meter.true_peak_envelope(x, 48000, L), _ref_tp_env(x, L))


@pytest.mark.parametrize("n", [100, EVAL_CHUNK, EVAL_CHUNK + 3, 3 * EVAL_CHUNK + 11])
def test_envelope_chunked_evaluation(n):
    e = Envelope(RNG.standard_normal(3001))
    t = np.clip(control_coord(np.arange(n), 48000), 0.0, 3000.0)
    ref = PchipInterpolator(np.arange(3001.0), e.values)(t)
    assert np.array_equal(e.at(n, 48000), ref)
    idx = RNG.integers(0, n, size=min(n, EVAL_CHUNK + 17))
    ref_i = PchipInterpolator(np.arange(3001.0), e.values)(
        np.clip(control_coord(idx, 48000), 0.0, 3000.0))
    assert np.array_equal(e.at_samples(idx, 48000), ref_i)


def _ref_box(x, lo, hi):
    n = x.shape[0]
    pad_l, pad_r = max(0, -lo), max(0, hi)
    xp = np.concatenate([np.full(pad_l, x[0]), x, np.full(pad_r, x[-1])])
    c = np.concatenate([[0.0], np.cumsum(xp)])
    i = np.arange(n) + pad_l
    return (c[i + hi + 1] - c[i + lo]) / (hi - lo + 1)


def _ref_hull(u, release):
    with np.errstate(divide="ignore"):
        depth = np.maximum(-20.0 * np.log10(np.maximum(u, 1e-300)), 0.0)
        w = np.log(depth)
    ln_a = -1.0 / release
    n = u.shape[0]
    block = int(min(n, max(256, 50.0 / abs(ln_a))))
    k = np.arange(block, dtype=np.float64)
    hl = np.empty(n)
    carry = -np.inf
    for start in range(0, n, block):
        seg = w[start:start + block]
        kk = k[: seg.size]
        local = np.maximum.accumulate(seg - kk * ln_a) + kk * ln_a
        local = np.maximum(local, carry + (kk + 1.0) * ln_a)
        hl[start:start + seg.size] = local
        carry = local[-1]
    return np.clip(10.0 ** (-np.exp(hl) / 20.0), u.min(), u)


@pytest.mark.parametrize("n", [3, 1000, 250001])
def test_envelope_in_place_forms(n):
    u = np.clip(1 - RNG.random(n) ** 6, 1e-7, 1.0)
    for rel in (0.5, 37.0, 2205.0):
        assert np.array_equal(envelope.release_hull(u, rel), _ref_hull(u, rel))
    for lo, hi in ((-5, 0), (0, 5), (-40, 40), (0, 0)):
        assert np.array_equal(envelope._box(u, lo, hi), _ref_box(u, lo, hi))


def _ref_sat_band(band, drive_db, harmonic, fs, f_lo, symmetry, oversample):
    g = math.expm1(drive_db * math.log(10.0) / 20.0)
    k = math.sqrt(2.0) * saturate.gated_rms(band, fs)
    u = _ref_up(band[:, None], oversample)[:, 0]
    t = np.tanh(u * (g / k))
    s = (k / g) * (t + 0.5 * symmetry * t * t)
    change = _ref_down((harmonic * (s - u))[:, None], oversample)[:, 0]
    change = _ref_split(change, fs, [f_lo])[1][:, 0]
    return band + change


def test_saturate_matches_whole_array_form():
    fs = 48000
    x = RNG.standard_normal((fs * 2, 2)) * 0.2
    plan = SaturatePlan(drive_db=[[0, 0], [4, 0], [2, 1], [1, 5], [0, 0]],
                        harmonic=[0.0, 0.5, 0.4, 0.7, 0.6])
    b = _ref_split(bands.to_ms(x), fs, plan.crossovers_hz)
    lower = [0.0, *plan.crossovers_hz]
    for k in range(5):
        for c in range(2):
            if plan.drive_db[k][c] > 0 and plan.harmonic[k] > 0:
                b[k, :, c] = _ref_sat_band(b[k, :, c], plan.drive_db[k][c], plan.harmonic[k],
                                           fs, lower[k], plan.symmetry, 4)
    assert np.array_equal(saturate.apply(x, fs, plan), bands.to_lr(b.sum(axis=0)))


def test_multiband_streamed_path_matches_info_path():
    fs = 48000
    x = RNG.standard_normal((fs * 2, 2)) * 0.3
    p = multiband.design(x, fs, -20.0, ceiling_db=-14.0, mid_makeup_db=[0, 1, 0, -1, 0])
    y = multiband.apply(x, fs, p)
    y_info, meta = multiband.apply_with_info(x, fs, p)
    assert np.array_equal(y, y_info)
    assert np.array_equal(bands.to_lr(meta["bands_out"].sum(axis=0)), y)
