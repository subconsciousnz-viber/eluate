"""Prototype fixes as monkey-patches (never touching the repo):
F1  asymmetric hold in box_smooth (min over offsets [-(L-1)-H, L-1]) for the multiband
    compressor, band ceilings and speclim (per band), and the final limiter.
F2  detector: per-control-point peak over a trailing window of W segments (>= the two
    the stage uses) instead of two adjacent segments.
F3  ms_link: the M/S minimum taken at the control rate, then PCHIP.
F4  glue: (a) attack floor (ms) for the lagging attack, (b) a trailing hold on the
    required gain before the program release, (c) gain applied above a split only.
"""
from __future__ import annotations

import contextlib

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import minimum_filter1d

from eluate import bands, envelope
from eluate.plan import CONTROL_RATE_HZ, MULTIBAND_MAX_GR_DB, Envelope, control_length
from eluate.stages import glue as gl, multiband as mb

_BAND = {"k": 0}


def minfilt(h: np.ndarray, left: int, right: int) -> np.ndarray:
    """m[n] = min h[n - left .. n + right], edges replicated. O(n) (van Herk)."""
    n = h.size
    if left == 0 and right == 0:
        return h.copy()
    size = left + right + 1
    hp = np.pad(h, (left, right), mode="edge")
    out = minimum_filter1d(hp, size, mode="nearest")
    # a centred window of `size` at padded index q covers [q - size//2, q - size//2 + size - 1];
    # for output n the window must start at padded index n (= n + left - left)
    q0 = size // 2
    return out[q0:q0 + n]


def _selftest():
    rng = np.random.default_rng(0)
    h = rng.random(200)
    for left, right in ((0, 0), (2, 2), (5, 2), (2, 5), (30, 1), (0, 7), (7, 0), (4, 3)):
        ref = np.array([h[max(0, i - left):i + right + 1].min() for i in range(h.size)])
        got = minfilt(h, left, right)
        assert np.array_equal(ref, got), (left, right)


_selftest()


def box_smooth_hold(g_req, attack: int, release: float = 0.0, hold: int = 0,
                    g_min: float = envelope.G_MIN) -> np.ndarray:
    """envelope.box_smooth with a hold of `hold` samples after each dip: the lookahead
    minimum runs over offsets [-(L-1)-H, L-1]; the two moving averages (support
    [-(L-1), L-1]) stay inside it, so g <= h <= u holds by the same argument."""
    u = envelope._ceiling(g_req, g_min)
    if u.size == 0:
        return u
    h = envelope.release_hull(u, release)
    L = max(1, int(attack) // 2 + 1)
    H = max(0, int(hold))
    if L == 1 and H == 0:
        return np.clip(h, g_min, u)
    m = minfilt(h, L - 1 + H, L - 1)
    if L > 1:
        g = envelope._box(envelope._box(m, -(L - 1), 0), 0, L - 1)
    else:
        g = m
    return np.clip(g, g_min, u)


def _split_tracker(orig):
    def wrapper(*a, **kw):
        for k, b in enumerate(orig(*a, **kw)):
            _BAND["k"] = k
            yield b
    return wrapper


@contextlib.contextmanager
def f1(hold_comp_ms, hold_ceil_ms=None):
    """F1 on the multiband compressor and the ceilings (band ceilings and speclim): per-band
    hold in ms (list, low to high)."""
    hold_ceil_ms = hold_comp_ms if hold_ceil_ms is None else hold_ceil_ms
    orig_split, orig_comp, orig_ceil = bands.split_iter, mb.comp_envelope, mb.ceiling_envelope

    def comp_envelope(det, threshold_db, ratio, attack_ms, release_ms, knee_db=0.0,
                      rate_hz=CONTROL_RATE_HZ):
        floor = 10.0 ** (-MULTIBAND_MAX_GR_DB / 20.0)
        if knee_db <= 0.0:
            g_lin = envelope.required_gain(det, 10.0 ** (threshold_db / 20.0))
            g_req = np.maximum(g_lin ** (1.0 - 1.0 / ratio), floor)
        else:
            with np.errstate(divide="ignore"):
                level = 20.0 * np.log10(det)
            g_req = np.maximum(10.0 ** (mb.knee_gr_db(level, threshold_db, ratio, knee_db) / 20.0), floor)
        attack = max(mb.ATTACK_FLOOR, int(round(attack_ms * rate_hz / 1000.0)))
        hold = int(round(hold_comp_ms[min(_BAND["k"], len(hold_comp_ms) - 1)] * rate_hz / 1000.0))
        g = box_smooth_hold(g_req, attack, release_ms * rate_hz / 1000.0, hold)
        return mb.to_env32(g, rate_hz)

    def ceiling_envelope(det, ceiling_db, lookahead_ms, release_ms, rate_hz=CONTROL_RATE_HZ):
        g_req = envelope.required_gain(det, 10.0 ** (ceiling_db / 20.0))
        look = max(mb.ATTACK_FLOOR, int(round(lookahead_ms * rate_hz / 1000.0)))
        hold = int(round(hold_ceil_ms[min(_BAND["k"], len(hold_ceil_ms) - 1)] * rate_hz / 1000.0))
        g = box_smooth_hold(g_req, look, release_ms * rate_hz / 1000.0, hold)
        env = mb.to_env32(g, rate_hz)
        return env if np.any(env.values) else Envelope.constant(0.0)

    bands.split_iter = _split_tracker(orig_split)
    mb.comp_envelope, mb.ceiling_envelope = comp_envelope, ceiling_envelope
    try:
        yield
    finally:
        bands.split_iter, mb.comp_envelope, mb.ceiling_envelope = orig_split, orig_comp, orig_ceil


def limiter_smoother(hold_ms: float, fs8: float):
    """smooth(g_req, attack, release) for drive.limiter_run with an F1 hold at 8x."""
    hold = int(round(hold_ms * 1e-3 * fs8))

    def smooth(g_req, attack, release):
        return box_smooth_hold(g_req, attack, release, hold)
    return smooth


@contextlib.contextmanager
def f2(window_ms):
    """F2: detector at point k = peak over the trailing W segments [k - (W - 1), k] (W >= 2;
    W = 2 is the stage's own detector). Per band, ms."""
    orig_split, orig_det = bands.split_iter, mb.detector

    def detector(v, fs, n_ctrl, rate_hz, _window=None):
        # the stage now passes its own window (built from this prototype, 2026-11-10); the
        # prototype's window_ms overrides it so the experiment measures what it says
        pk = mb._frame_peaks(v, fs, n_ctrl, rate_hz)
        w = max(2, int(round(window_ms[min(_BAND["k"], len(window_ms) - 1)] * rate_hz / 1000.0)))
        pp = np.concatenate([np.zeros(w - 1), pk])
        return sliding_window_view(pp, w).max(axis=1)

    bands.split_iter = _split_tracker(orig_split)
    mb.detector = detector
    try:
        yield
    finally:
        bands.split_iter, mb.detector = orig_split, orig_det


@contextlib.contextmanager
def f3():
    """F3: ms_link minimum at the control rate, then PCHIP."""
    orig = mb._apply_linked

    def _apply_linked(band, k, n, fs, plan, makeup):
        total = linked_gain_f3(plan, k, n, fs)
        if np.any(total):
            band *= (10.0 ** (total / 20.0))[:, None]

    mb._apply_linked = _apply_linked
    try:
        yield
    finally:
        mb._apply_linked = orig


def linked_gain_f3(plan, k, n, fs):
    """The gain F3 applies to band k (dB), for comparison with drive.mb_total_gain_db."""
    n_ctrl = control_length(n, fs)
    red = []
    for c in range(2):
        r = np.clip(mb._ctrl(plan.gr_db[k][c], n_ctrl), -MULTIBAND_MAX_GR_DB, 0.0)
        if plan.ceiling_gr_db is not None:
            r = r + mb._ctrl(plan.ceiling_gr_db[k][c], n_ctrl)
        red.append(r)
    total = Envelope(np.minimum(red[0], red[1])).at(n, fs)
    if plan.mode_makeup_db is not None:
        total += plan.mode_makeup_db[k][0].at(n, fs)
    return total + plan.makeup()[k][0]


@contextlib.contextmanager
def f4(attack_floor_ms: float | None = None, hold_ms: float | None = None):
    """F4 on the glue design: (a) attack floor, (b) trailing hold (min over the past
    hold_ms of the required gain) before the program release."""
    orig_attack, orig_release = gl.lagging_attack, gl.program_release

    def lagging_attack(h_db, attack):
        if attack_floor_ms is not None:
            attack = max(attack, attack_floor_ms * CONTROL_RATE_HZ / 1000.0)
        return orig_attack(h_db, attack)

    def program_release(u, release):
        if hold_ms is not None:
            u = minfilt(u, int(round(hold_ms * CONTROL_RATE_HZ / 1000.0)), 0)
        return orig_release(u, release)

    gl.lagging_attack, gl.program_release = lagging_attack, program_release
    try:
        yield
    finally:
        gl.lagging_attack, gl.program_release = orig_attack, orig_release


def glue_apply_split(x, fs, plan, split_hz: float = 100.0):
    """F4c: the glue's gain applied to the part above split_hz only (zero-phase split)."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = bands.split(x, fs, [split_hz], phase="zero")
    total = plan.gain_db.at(x.shape[0], fs) + plan.makeup_db
    wet = hi * (10.0 ** (total / 20.0))[:, None]
    return lo + hi + plan.mix * (wet - hi)
