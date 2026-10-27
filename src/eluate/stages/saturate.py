"""Band saturation (spec 3.5 stage 5, audit 05 A12, A13).

Contract: `apply(x, fs, plan) -> y`, pure, stereo (n, 2) in, float64 (n, 2) out.
- M, S = bands.to_ms(x), split into zero-phase bands (default
  bands.DEFAULT_CROSSOVERS, 100 / 400 / 1600 / 6400 Hz). For band k, channel
  c with drive D = drive_db[k][c] and blend
  h = harmonic[k] > 0 (otherwise the band passes untouched):
    G = expm1(D ln10 / 20)                 (= 10^(D/20) - 1, no cancellation)
    K = sqrt(2) * gated RMS of the band     (400 ms blocks above -60 dBFS)
    u = resample.upsample(band, L)        L = 4, or 8 for the top (air) band
    s = (K / G) [tanh(G u / K) + (sigma / 2) tanh^2(G u / K)],  sigma = symmetry
    w = u + h (s - u)                                          harmonic blend
    v = band + HP(resample.decimate(w - u, L))                 only the change
  s has unity small-signal slope (ds/du = sech^2 (1 + sigma tanh) = 1 at 0), so
  the compensation is for the curve's slope only, not RMS-equal: what the
  saturator adds is real band energy (and the peaks it compresses), measured by
  evaluate.py with the solver closing the loop (spec 3.5 item 5). The curve is
  monotone for sigma <= 1. The odd tanh term gives odd harmonics; the even
  term (sigma > 0) gives H2, so fundamentals up to fs/4 (the air band) get
  in-band harmonics. The even term also rectifies (DC and envelope-rate
  content), so the change is high-passed at the band's lower crossover (the
  high part of bands.split, zero phase) before it is added; the sub band never
  saturates, so every driven band has a lower edge.
- The knee is scaled to a robust band level (gated RMS x sqrt 2, the peak of a
  sine at that RMS), never the whole-track peak, so one spike does not mute the
  saturator. G < 1e-12, h = 0, or a band with no block above the gate leaves
  the band bit-identical.
- Only the saturation's change passes through the resampler (the processing
  pair of stages/resample.py, passband 0.98 x Nyquist), so the dry path is
  never resampled. The top (air) band runs at 8x (audit 05 Q5): its harmonics
  above the 4x Nyquist would otherwise fold back into the audio band. Drive is
  capped at 12 dB by SaturatePlan.
- Sub band drive is 0 on M and S (enforced by SaturatePlan); air band drive is
  0 in SaturatePlan.defaults and the high-mid S band gets the width drive.
- `harmonic[k]` in [0, 1] is the wet fraction of band k's saturated path. The
  planner (chunk 4, docs/gain-staging.md 6) sets it to 1 on every band and
  writes the drive from the band's deficit below the target slope line
  (planner.plan_saturate); the drive is in this stage's own units until the
  Saturn calibration is done.
"""

from __future__ import annotations

import math

import numpy as np

from eluate import bands
from eluate.stages import resample
from eluate.plan import SaturatePlan

OVERSAMPLE = 4
OVERSAMPLE_AIR = 8
GATE_DBFS = -60.0
BLOCK_S = 0.4
CURVE_CHUNK = 1 << 20   # samples per chunk of the elementwise curve (memory only)


def gated_rms(v: np.ndarray, fs: float) -> float:
    """RMS over 400 ms blocks (50 % overlap) whose own RMS exceeds -60 dBFS; 0 if none.
    Inputs shorter than one block use the whole signal as one block."""
    v = np.asarray(v, dtype=np.float64)
    L = int(round(BLOCK_S * fs))
    if v.size < L or L == 0:
        p = np.array([np.mean(v * v)]) if v.size else np.zeros(0)
    else:
        c = np.concatenate([[0.0], np.cumsum(v * v)])
        starts = np.arange(0, v.size - L + 1, max(1, L // 2))
        p = (c[starts + L] - c[starts]) / L
    gate = 10.0 ** (GATE_DBFS / 10.0)
    p = p[p > gate]
    return float(math.sqrt(p.mean())) if p.size else 0.0


def saturate_band(band: np.ndarray, drive_db: float, harmonic: float, fs: float,
                  f_lo: float, symmetry: float = 0.3, oversample: int = OVERSAMPLE) -> np.ndarray:
    """One band, one channel (1-D), lower crossover f_lo. See the module contract."""
    band = np.asarray(band, dtype=np.float64)
    g = math.expm1(drive_db * math.log(10.0) / 20.0) if drive_db > 0 else 0.0
    if g < 1e-12 or harmonic <= 0.0:
        return band.copy()
    k = math.sqrt(2.0) * gated_rms(band, fs)
    if k == 0.0:
        return band.copy()
    u = resample.upsample(band, oversample)
    # elementwise, in chunks, written back into u: the same operations in the same
    # order as  t = tanh(u (g/k)); s = (k/g) (t + 0.5 sigma t t); harmonic (s - u)
    # (bit-identical; one oversampled array alive instead of six)
    for a in range(0, u.size, CURVE_CHUNK):
        uc = u[a:a + CURVE_CHUNK]
        t = uc * (g / k)
        np.tanh(t, out=t)
        w = (0.5 * symmetry) * t
        w *= t
        w += t
        w *= (k / g)
        w -= uc
        w *= harmonic
        uc[...] = w
    change = resample.decimate(u, oversample)
    del u
    if f_lo > 0:
        it = bands.split_iter(change, fs, [f_lo], phase="zero")
        next(it)
        change = next(it)
    return band + change


def apply(x: np.ndarray, fs: float, plan: SaturatePlan) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("saturate needs stereo (n, 2) audio")
    lower = [0.0, *plan.crossovers_hz]
    nb = len(plan.crossovers_hz) + 1
    # bands streamed and summed in band order (as b.sum(axis=0), in place): bit-identical,
    # one band and the residual alive instead of all bands; the input is released
    # once converted to M/S (freed when the caller passed a temporary)
    ms = bands.to_ms(x)
    del x
    acc = None
    it = bands.split_iter(ms, fs, plan.crossovers_hz, order=plan.order, phase="zero")
    del ms
    for k, band in enumerate(it):
        for c in range(2):
            if plan.drive_db[k][c] > 0.0 and plan.harmonic[k] > 0.0:
                band[:, c] = saturate_band(
                    band[:, c], plan.drive_db[k][c], plan.harmonic[k], fs, lower[k],
                    plan.symmetry, OVERSAMPLE_AIR if k == nb - 1 else OVERSAMPLE)
        if acc is None:
            acc = band
        else:
            acc += band
        del band
    return bands.to_lr(acc)
