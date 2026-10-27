"""Static parametric input EQ (docs/gain-staging.md section 2 item 2, chunk 2).

Contract: `apply(x, fs, plan) -> y`, pure, stereo (n, 2) in, float64 (n, 2) out.
The same filter runs on L and R, so M and S get the same EQ, except the top end
when a side shelf is set (plan.ms_top): then the shared sections run on L and R and the
top shelves run on M (high_shelf, air_shelf) and S (high_shelf_side, air_shelf_side)
separately (maintainer, 2026-10-20; the air shelves: sprint 2026-10-21 T12). It is static over the whole track (the maintainer: a level-following EQ
would boost pads and ambience in every breakdown with no sub).
- Sections, in the maintainer's order, each optional (None = off):
  high-pass 24 dB/oct (fc, Q in [0.5, 1]), low shelf 12 dB/oct (gain, corner
  in [60, 250] Hz, Q in [0.5, 0.707]), bell (cut only, centre in [120, 1000]
  Hz, Q in [0.7, 4]), band shelf (chunk 4, cut only, below), high shelf 6 dB/oct
  (gain, corner, default 16 kHz; the presence shelf), air shelf 6 dB/oct (gain, corner).
  Every gain is bounded to +-3 dB (plan.INPUTEQ_MAX_DB = TONE_MAX_DB), except the
  high shelves' boosts: +4.5 dB on M or linked, +6 dB on S (plan.INPUTEQ_HS_*).
- Specified response (`spec_sos`): the single-pass digital filter of that
  order at the full gain, bilinear transform prewarped at the corner or centre:
  high-pass = two cascaded RBJ high-pass biquads with the section's Q (the
  squared biquad: Q 0.707 is the squared Butterworth, i.e. Linkwitz-Riley,
  -6.02 dB at fc as in sidestrip and bands; the gain at fc is 20 log10(Q^2));
  low shelf = RBJ low shelf with the section's Q (gain/2 at the corner);
  bell = RBJ peaking EQ (the full cut at the centre); high shelf = first-order
  shelf H(s) = (sqrt(A) s + 1) / (s / sqrt(A) + 1) at s = j f / fc
  (gain/2 at the corner).
- Band shelf (chunk 4, cut only, gain g = 10^(gain_db / 20) <= 1): the magnitude
  D(f) = 1 - (1 - g) |HP_lo(f)| |LP_hi(f)|, with HP_lo and LP_hi the 4th-order
  Linkwitz-Riley high-pass at lo_hz and low-pass at hi_hz (squared digital
  2nd-order Butterworth, bilinear, so |LP4| + |HP4| = 1): flat at g between the
  corners, 24 dB/oct transitions, no ripple, half the dip (in amplitude) at
  each corner when they are far apart. It is a parallel structure, not a
  biquad: under phase "zero" the stage subtracts (1 - g) times the zero-phase
  band (forward-backward 2nd-order Butterworth high-pass and low-pass, exactly
  |HP_lo| |LP_hi|) from the output of the other sections; under "minimum" it is
  the in-phase three-way Linkwitz-Riley sum low + g mid + high (magnitude D,
  phase that of two 2nd-order allpasses).
- phase "zero" (default): one forward-backward pass (sosfiltfilt) of the
  compensated per-pass filter (`pass_sos`), so the magnitude is |H_pass|^2 and
  the phase is zero, except the high-pass, which runs once, causally, at its
  specified response (R7e, sprint 2026-10-21 T6: a zero-phase 24 dB/oct high-pass
  put a pre-echo 12 to 20 ms ahead of every kick); its magnitude is the same either
  way, so `response_db` is unchanged. Compensation: the high-pass runs one RBJ biquad per pass
  (12 dB/oct, Q unchanged), so the double pass is exactly the specified
  24 dB/oct cascade; the shelves and the bell run the same design at half
  the dB gain, which gives the specified gain exactly at DC / Nyquist, at the
  corner and at the centre, and keeps the rest of the curve within 0.005 dB
  of the specification at +-3 dB (exact for the low shelf at Q 0.5, where the
  RBJ shelf is the square of a first-order shelf). The track is embedded in
  digital silence (zero padding of PAD_TAU time constants of the slowest
  pole, at least bands.PAD_MIN samples per side) and filtered from rest, as
  bands.split and the notch stage do.
- phase "minimum": the specified filter run once, causally, from rest. Exact
  magnitude, minimum phase (kept as an option; see the chunk 2 PR for why zero
  phase is the default).
- Identity: a plan with every section None returns a bit-exact copy. Silence
  stays silence. DC: blocked by the high-pass; with the high-pass off a DC
  input leaves at the low shelf's gain.
- The planner fits the sections to the whole-track spectral target
  (planner.plan_inputeq); this stage never reads the plan's design record.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from scipy import signal

from eluate import bands
from eluate.plan import InputEqPlan

PAD_TAU = 20.0     # time constants of the slowest pole (bands.PAD_TAU)


def _rbj_highpass(fc: float, q: float, fs: float) -> np.ndarray:
    w0 = 2.0 * math.pi * fc / fs
    c, al = math.cos(w0), math.sin(w0) / (2.0 * q)
    b = [(1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0]
    a = [1.0 + al, -2.0 * c, 1.0 - al]
    return _sos(b, a)


def _rbj_low_shelf(gain_db: float, fc: float, q: float, fs: float) -> np.ndarray:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * fc / fs
    c, al = math.cos(w0), math.sin(w0) / (2.0 * q)
    k = 2.0 * math.sqrt(A) * al
    b = [A * ((A + 1) - (A - 1) * c + k), 2 * A * ((A - 1) - (A + 1) * c),
         A * ((A + 1) - (A - 1) * c - k)]
    a = [(A + 1) + (A - 1) * c + k, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - k]
    return _sos(b, a)


def _rbj_peak(gain_db: float, fc: float, q: float, fs: float) -> np.ndarray:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * fc / fs
    c, al = math.cos(w0), math.sin(w0) / (2.0 * q)
    b = [1.0 + al * A, -2.0 * c, 1.0 - al * A]
    a = [1.0 + al / A, -2.0 * c, 1.0 - al / A]
    return _sos(b, a)


def _first_order_high_shelf(gain_db: float, fc: float, fs: float) -> np.ndarray:
    """Bilinear transform of (sqrt(A) s + 1) / (s / sqrt(A) + 1), s normalised to fc."""
    r = math.sqrt(10.0 ** (gain_db / 20.0))
    K = math.tan(math.pi * fc / fs)
    b = [r / K + 1.0, 1.0 - r / K, 0.0]
    a = [1.0 / (r * K) + 1.0, 1.0 - 1.0 / (r * K), 0.0]
    return _sos(b, a)


def _sos(b, a) -> np.ndarray:
    a0 = a[0]
    return np.array([[b[0] / a0, b[1] / a0, b[2] / a0, 1.0, a[1] / a0, a[2] / a0]])


def _butter2(fc: float, fs: float, kind: str) -> np.ndarray:
    return signal.butter(2, fc, btype=kind, fs=fs, output="sos")


def band_shelf_db(plan: InputEqPlan, f, fs: float) -> np.ndarray:
    """Magnitude (dB) of the band shelf alone at f (Hz); zeros when it is off."""
    f = np.asarray(f, dtype=np.float64)
    bs = plan.band_shelf
    if bs is None:
        return np.zeros(f.shape)
    _, hp = signal.sosfreqz(_butter2(bs.lo_hz, fs, "highpass"), worN=f, fs=fs)
    _, lp = signal.sosfreqz(_butter2(bs.hi_hz, fs, "lowpass"), worN=f, fs=fs)
    g = 10.0 ** (bs.gain_db / 20.0)
    return 20.0 * np.log10(1.0 - (1.0 - g) * np.abs(hp) ** 2 * np.abs(lp) ** 2)


def _band_shelf_zero(y: np.ndarray, fs: float, plan: InputEqPlan) -> np.ndarray:
    bs = plan.band_shelf
    sos = np.vstack([_butter2(bs.lo_hz, fs, "highpass"), _butter2(bs.hi_hz, fs, "lowpass")])
    n, pl = y.shape[0], padlen(sos)
    g = 10.0 ** (bs.gain_db / 20.0)
    out = np.empty_like(y)
    for c in range(y.shape[1]):
        mid = signal.sosfiltfilt(sos, np.pad(y[:, c], (pl, pl)), padtype=None)[pl:pl + n]
        out[:, c] = y[:, c] - (1.0 - g) * mid
    return out


def _band_shelf_causal(y: np.ndarray, fs: float, plan: InputEqPlan) -> np.ndarray:
    bs = plan.band_shelf
    g = 10.0 ** (bs.gain_db / 20.0)

    def lr4(v, fc, kind):
        s = _butter2(fc, fs, kind)
        return signal.sosfilt(np.vstack([s, s]), v, axis=0)

    lo, rest = lr4(y, bs.lo_hz, "lowpass"), lr4(y, bs.lo_hz, "highpass")
    lo = lr4(lo, bs.hi_hz, "lowpass") + lr4(lo, bs.hi_hz, "highpass")     # allpass at hi
    return lo + g * lr4(rest, bs.hi_hz, "lowpass") + lr4(rest, bs.hi_hz, "highpass")


def section_sos(plan: InputEqPlan, fs: float, passes: int) -> dict[str, np.ndarray]:
    """Per active section, the sos of one pass when the response is run `passes` times
    (2 = the zero-phase design, 1 = the specified response)."""
    out = {}
    h = 1.0 / passes
    if plan.highpass is not None:
        s = _rbj_highpass(plan.highpass.fc_hz, plan.highpass.q, fs)
        out["highpass"] = s if passes == 2 else np.vstack([s, s])
    if plan.low_shelf is not None:
        ls = plan.low_shelf
        out["low_shelf"] = _rbj_low_shelf(ls.gain_db * h, ls.fc_hz, ls.q, fs)
    if plan.bell is not None:
        bl = plan.bell
        out["bell"] = _rbj_peak(bl.gain_db * h, bl.fc_hz, bl.q, fs)
    if plan.high_shelf is not None:
        hs = plan.high_shelf
        out["high_shelf"] = _first_order_high_shelf(hs.gain_db * h, hs.fc_hz, fs)
    if plan.air_shelf is not None:
        a = plan.air_shelf
        out["air_shelf"] = _first_order_high_shelf(a.gain_db * h, a.fc_hz, fs)
    return out


def spec_sos(plan: InputEqPlan, fs: float) -> np.ndarray | None:
    """The specified (single-pass, full-gain) response of every active section."""
    s = section_sos(plan, fs, 1)
    return np.vstack(list(s.values())) if s else None


def pass_sos(plan: InputEqPlan, fs: float) -> np.ndarray | None:
    """The compensated filter run forward and backward under phase "zero"."""
    s = section_sos(plan, fs, 2)
    return np.vstack(list(s.values())) if s else None


def response_db(plan: InputEqPlan, f, fs: float, channel: str = "M") -> np.ndarray:
    """Magnitude (dB) of the stage at frequencies f (Hz) for the plan's phase mode, as
    channel "M" or "S" hears it (the same unless the top end is M/S)."""
    plan = channel_plan(plan, channel)
    f = np.asarray(f, dtype=np.float64)
    if plan.phase == "minimum":
        sos, p = spec_sos(plan, fs), 1.0
    else:
        sos, p = pass_sos(plan, fs), 2.0
    if sos is None:
        return band_shelf_db(plan, f, fs)
    _, h = signal.sosfreqz(sos, worN=f, fs=fs)
    with np.errstate(divide="ignore"):
        return p * 20.0 * np.log10(np.abs(h)) + band_shelf_db(plan, f, fs)


def padlen(sos: np.ndarray) -> int:
    """Zero padding per side: PAD_TAU time constants of the slowest pole."""
    r = 0.0
    for row in sos:
        r = max(r, float(np.max(np.abs(np.roots(row[3:6])))) if np.any(row[4:6]) else 0.0)
    if r <= 0.0:
        return bands.PAD_MIN
    return max(bands.PAD_MIN, int(math.ceil(PAD_TAU / -math.log(r))))


def channel_plan(plan: InputEqPlan, channel: str) -> InputEqPlan:
    """The linked-form plan one channel ("M" or "S") hears: with an M/S top end
    (plan.ms_top), S takes the _side shelves in place of high_shelf and air_shelf."""
    if not plan.ms_top:
        return plan
    if channel == "M":
        return replace(plan, high_shelf_side=None, air_shelf_side=None)
    return replace(plan, high_shelf=plan.high_shelf_side, air_shelf=plan.air_shelf_side,
                   high_shelf_side=None, air_shelf_side=None)


def _shelf(v: np.ndarray, fs: float, shelves, phase: str) -> np.ndarray:
    """One channel through its top shelves alone (None entries skipped), in the stage's
    phase mode."""
    shelves = [s for s in shelves if s is not None]
    if not shelves:
        return v.copy()
    if phase == "minimum":
        sos = np.vstack([_first_order_high_shelf(s.gain_db, s.fc_hz, fs) for s in shelves])
        return signal.sosfilt(sos, v)
    sos = np.vstack([_first_order_high_shelf(s.gain_db / 2.0, s.fc_hz, fs) for s in shelves])
    n, pl = v.shape[0], padlen(sos)
    return signal.sosfiltfilt(sos, np.pad(v, (pl, pl)), padtype=None)[pl:pl + n]


def apply(x: np.ndarray, fs: float, plan: InputEqPlan) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("inputeq needs stereo (n, 2) audio")
    for f in plan.frequencies():
        if not f < fs / 2.0:
            raise ValueError(f"inputeq frequency {f} Hz is not below fs/2")
    if plan.ms_top:
        # M/S top end: the shared sections on L and R (the same filter on both is the
        # same on M and S), then each channel's top shelves on it
        y = bands.to_ms(apply(x, fs, replace(plan, high_shelf=None, high_shelf_side=None,
                                             air_shelf=None, air_shelf_side=None)))
        y[:, 0] = _shelf(y[:, 0], fs, (plan.high_shelf, plan.air_shelf), plan.phase)
        y[:, 1] = _shelf(y[:, 1], fs, (plan.high_shelf_side, plan.air_shelf_side), plan.phase)
        return bands.to_lr(y)
    if plan.phase == "minimum":
        sos = spec_sos(plan, fs)
        y = x.copy() if sos is None else signal.sosfilt(sos, x, axis=0)
        return y if plan.band_shelf is None else _band_shelf_causal(y, fs, plan)
    if plan.highpass is not None:
        hp = section_sos(replace(plan, low_shelf=None, bell=None, high_shelf=None,
                                 air_shelf=None, band_shelf=None), fs, 1)["highpass"]
        x = signal.sosfilt(hp, x, axis=0)
        plan = replace(plan, highpass=None)
    sos = pass_sos(plan, fs)
    if sos is None:
        y = x.copy()
    else:
        n, pl = x.shape[0], padlen(sos)
        y = np.empty_like(x)
        for c in range(x.shape[1]):
            xp = np.pad(x[:, c], (pl, pl))
            y[:, c] = signal.sosfiltfilt(sos, xp, padtype=None)[pl:pl + n]
    return y if plan.band_shelf is None else _band_shelf_zero(y, fs, plan)
