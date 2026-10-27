"""Long-term spectral tilt, LTAS and spectral centroid (spec 3.3 row "tilt").

Contract:
- PSD: Welch, Hann window, N = 8192 (scaled by powers of two at 88.2 kHz and
  above, see _common.fft_size), 50 % overlap, constant detrend, density
  scaling, computed as the mean over the frames of scipy.signal.spectrogram
  (identical to scipy.signal.welch with the same settings). Views: M, S and
  stereo = (PSD_L + PSD_R) / 2 = (PSD_M + PSD_S) / 2.
- LTAS = **mean PSD per 1/3-octave band** (not band-summed power, audit F5):
  nominal centres 1000 * 2^(k/3) from 19.7 Hz to 16 kHz (k = -17..12, a band
  whose upper edge is above 0.95 fs/2 dropped), band edges centre * 2^(+-1/6). Each FFT bin is a
  rectangle of width df centred on its frequency; the band value is the mean of
  that piecewise-constant PSD over the band (fractional bin overlap), so every
  band is defined even when it is narrower than a bin. In dB (10 log10 PSD,
  floor -200).
- Slope: OLS of LTAS dB against log2(centre) over the centres in [100, 10000]
  Hz, dB/oct, per view. For pink noise the band means of a power law are an
  exact power law, so the fit is unbiased.
- Sub shelf: mean of (LTAS - extrapolated fit line) over centres in [30, 90] Hz
  (31.5 to 80 Hz nominal), per view, dB; positive means the sub sits above the
  line. Reported separately and never part of the slope fit.
- Peak-hold LTAS: per-frame band PSDs smoothed by a 1 s moving mean over frames,
  maximum over time, dB, per view.
- Time series on the common grid: `centroid_hz` (power-weighted mean frequency
  of the M spectrum per frame, NaN on silent frames) and `slope_series` (stereo
  slope of the 3 s moving-mean band PSDs), interpolated from frame centres.
- `flatness[b]`: spectral flatness (geometric / arithmetic mean) of the stereo
  LTAS bins inside crossover band b (20 Hz to min(20 kHz, 0.95 fs/2) overall),
  after dividing out the fitted tilt line; near 1 for noise-like bands, low for
  sparse tonal ones. Input for the saturate stage's harmonic/linear default.
  A band with fewer than 2 bins of power reads FLATNESS_DEFAULT = 0.5 with
  `flatness_measurable[b]` False (audit 04 A12).
- Display LTAS (`fine_centres`, `fine_db`): the same band mean on centres every
  1/24 octave from 20 Hz to 20 kHz (an upper edge above 0.95 fs/2 dropped), each
  band 1/6 octave wide, per view, dB. For drawing only; the slope, shelf and
  flatness use the third-octave LTAS.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from ._common import VIEWS, Context, db_power, ensure_context, fft_size

N_REF = 8192
FIT_RANGE = (100.0, 10000.0)
SUB_RANGE = (30.0, 90.0)
PEAK_HOLD_S = 1.0
SLOPE_SERIES_S = 3.0
FLATNESS_DEFAULT = 0.5  # not measurable (empty band): neutral, flagged
FINE_STEP_OCT = 1.0 / 24.0
FINE_WIDTH_OCT = 1.0 / 6.0


def third_octave_centres(fs: float) -> np.ndarray:
    c = 1000.0 * 2.0 ** (np.arange(-17, 13) / 3.0)
    return c[c * 2 ** (1 / 6) <= 0.95 * fs / 2.0 + 1e-9]


def fine_centres(fs: float) -> np.ndarray:
    k = np.arange(0, int(round(np.log2(1000.0) / FINE_STEP_OCT)) + 1)
    c = 20.0 * 2.0 ** (k * FINE_STEP_OCT)
    return c[c * 2 ** (FINE_WIDTH_OCT / 2) <= 0.95 * fs / 2.0 + 1e-9]


def band_weights(f: np.ndarray, centres: np.ndarray, width_oct: float = 1.0 / 3.0) -> np.ndarray:
    """(bands, bins) weights; W @ psd = mean PSD over each band (fractional bins)."""
    df = f[1] - f[0]
    lo_b = f - df / 2.0
    hi_b = f + df / 2.0
    lo = centres * 2.0 ** (-width_oct / 2.0)
    hi = centres * 2.0 ** (width_oct / 2.0)
    ov = np.clip(np.minimum(hi_b[None, :], hi[:, None]) - np.maximum(lo_b[None, :], lo[:, None]), 0.0, None)
    return ov / (hi - lo)[:, None]


def fit_slope(centres: np.ndarray, level_db: np.ndarray, rng=FIT_RANGE):
    """OLS slope and intercept of level_db (bands,) or (bands, frames) vs log2 f."""
    sel = (centres >= rng[0] - 1e-9) & (centres <= rng[1] + 1e-9)
    xx = np.log2(centres[sel])
    coef = np.polyfit(xx, level_db[sel], 1)
    return coef[0], coef[1]


def _spectrogram(y: np.ndarray, fs: float, n: int):
    nper = min(n, y.shape[0])
    f, t, p = signal.spectrogram(y, fs, window="hann", nperseg=nper, noverlap=nper // 2,
                                 detrend="constant", scaling="density", mode="psd")
    return f, t, p


def analyse(x: np.ndarray, fs: float, ctx: Context | None = None) -> dict:
    ctx = ensure_context(x, fs, ctx)
    n_fft = fft_size(N_REF, ctx.fs)
    centres = third_octave_centres(ctx.fs)
    ms = ctx.ms
    mean_psd = {}
    band_frames = {}
    centroid = None
    f = t = None
    for v, y in (("M", ms[:, 0]), ("S", ms[:, 1])):
        f, t, p = _spectrogram(y, ctx.fs, n_fft)
        mean_psd[v] = p.mean(axis=1)
        w = band_weights(f, centres)
        band_frames[v] = w @ p
        if v == "M":
            tot = p.sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                centroid = np.where(tot > 1e-20, (f @ p) / tot, np.nan)
        del p
    mean_psd["stereo"] = 0.5 * (mean_psd["M"] + mean_psd["S"])
    band_frames["stereo"] = 0.5 * (band_frames["M"] + band_frames["S"])
    w = band_weights(f, centres)
    fc = fine_centres(ctx.fs)
    wf = band_weights(f, fc, FINE_WIDTH_OCT)
    fine = {v: db_power(wf @ mean_psd[v]) for v in VIEWS}

    ltas, peak, slope, icpt, shelf = {}, {}, {}, {}, {}
    hop_s = (t[1] - t[0]) if t.size > 1 else 1.0
    ph = max(1, int(round(PEAK_HOLD_S / hop_s)))
    sub_sel = (centres >= SUB_RANGE[0]) & (centres <= SUB_RANGE[1])
    for v in VIEWS:
        ltas[v] = db_power(w @ mean_psd[v])
        peak[v] = db_power(np.max(uniform_filter1d(band_frames[v], ph, axis=1, mode="nearest"), axis=1))
        slope[v], icpt[v] = (float(c) for c in fit_slope(centres, ltas[v]))
        line = icpt[v] + slope[v] * np.log2(centres[sub_sel])
        shelf[v] = float(np.mean(ltas[v][sub_sel] - line))

    # 3 s slope series (stereo)
    ss = max(1, int(round(SLOPE_SERIES_S / hop_s)))
    bf = uniform_filter1d(band_frames["stereo"], ss, axis=1, mode="nearest")
    lvl = db_power(bf)
    s_series, _ = fit_slope(centres, lvl)
    silent = bf.sum(axis=0) < 1e-20
    s_series = np.where(silent, np.nan, s_series)

    # flatness per crossover band, tilt removed
    edges = [20.0, *ctx.fcs, min(20000.0, 0.95 * ctx.fs / 2.0)]
    with np.errstate(divide="ignore"):
        line_bins = icpt["stereo"] + slope["stereo"] * np.log2(np.maximum(f, 1e-3))
    pw = mean_psd["stereo"] / 10.0 ** (line_bins / 10.0)
    flat = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (f >= lo) & (f < hi) & (pw > 0)
        if np.count_nonzero(sel) < 2:
            flat.append(np.nan)
            continue
        v = pw[sel]
        flat.append(float(np.exp(np.mean(np.log(v))) / np.mean(v)))

    g = ctx.grid
    flat = np.array(flat)
    flat_ok = np.isfinite(flat)
    flat = np.where(flat_ok, flat, FLATNESS_DEFAULT)
    return {
        "centres": centres, "ltas_db": ltas, "peak_hold_db": peak,
        "slope_db_oct": slope, "intercept_db": icpt, "sub_shelf_db": shelf,
        "fit_range": FIT_RANGE, "sub_range": SUB_RANGE, "n_fft": n_fft,
        "fine_centres": fc, "fine_db": fine,
        "psd_f": f, "psd": mean_psd,
        "t": g.times, "centroid_hz": g.to_grid(t, centroid),
        "slope_series": g.to_grid(t, s_series),
        "flatness": flat, "flatness_measurable": flat_ok, "flatness_edges": np.array(edges),
    }
