"""Measurements: gain velocity, gain spectra, harmonics, envelope modulation, sidebands,
kick-region gain look, band energies, aliasing residue."""
from __future__ import annotations

import numpy as np
from scipy import signal

from .common import pdb


def mono(y):
    y = np.asarray(y, dtype=np.float64)
    return y if y.ndim == 1 else y.mean(axis=1)


def velocity(g_db, fs) -> dict:
    """|dG/dt| in dB/ms: max, p99.9, p99; also the max second difference in dB/ms^2."""
    g = np.asarray(g_db, dtype=np.float64)
    d = np.abs(np.diff(g)) * fs / 1000.0
    if d.size < 2:
        return {"vel_max": 0.0, "vel_p999": 0.0, "vel_p99": 0.0, "acc_max": 0.0}
    d2 = np.abs(np.diff(g, 2)) * (fs / 1000.0) ** 2
    return {"vel_max": float(d.max()), "vel_p999": float(np.percentile(d, 99.9)),
            "vel_p99": float(np.percentile(d, 99.0)), "acc_max": float(d2.max())}


def ripple_pp_db(g_db, fs, skip_s=1.0) -> float:
    g = np.asarray(g_db, dtype=np.float64)
    k = int(skip_s * fs)
    seg = g[k:-k] if g.size > 2 * k + 10 else g
    return float(seg.max() - seg.min())


def gain_spectrum(g_lin, fs, edges=(20.0, 50.0, 100.0, 500.0),
                  lines=(1000.0, 2000.0, 3000.0, 4000.0), skip_s=0.5) -> dict:
    """Spectrum of the modulation signal m = g / mean(g) - 1 (Hann). Energy fractions above
    the edges (of the AC energy); line levels: the largest amplitude within +-5 Hz of each
    line, as the sideband level it would put on a partial: 20 log10(A / 2) dB re carrier."""
    g = np.asarray(g_lin, dtype=np.float64)
    k = int(skip_s * fs)
    g = g[k:-k] if g.size > 2 * k + 10 else g
    m = g / g.mean() - 1.0
    n = m.size
    w = np.hanning(n)
    X = np.fft.rfft(m * w)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    A = 2.0 * np.abs(X) / w.sum()
    P = np.abs(X) ** 2
    tot = P[1:].sum()
    out = {"mod_pp_db": float(20.0 * np.log10((1.0 + m.max()) / (1.0 + m.min()))),
           "mod_rms_db": float(20.0 * np.log10(max(np.sqrt(np.mean(m * m)), 1e-300)))}
    for e in edges:
        out[f"frac_gt_{int(e)}"] = float(P[f > e].sum() / tot) if tot > 0 else 0.0
    for ln in lines:
        sel = np.abs(f - ln) <= 5.0
        a = float(A[sel].max()) if sel.any() else 0.0
        out[f"line_{int(ln)}_db"] = float(20.0 * np.log10(max(a / 2.0, 1e-300)))
    return out


def harmonics(y, fs, f0, nh=12, skip_s=1.0, bins=3, fmax=20000.0) -> dict:
    """Hann FFT of the steady-state portion (channel mean). Fundamental and harmonic
    powers from +-bins around k f0; THD, H2, H3 re fundamental (dB); non-harmonic residue
    (everything else in [20, fmax] Hz) re fundamental; the top non-harmonic lines."""
    v = mono(y)
    k = int(skip_s * fs)
    v = v[k:-k] if v.size > 2 * k + 10 else v
    n = v.size
    w = np.hanning(n)
    X = np.fft.rfft(v * w)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    P = np.abs(X) ** 2
    df = fs / n
    harm_mask = np.zeros(f.size, dtype=bool)

    def band(fc):
        i = int(round(fc / df))
        lo, hi = max(0, i - bins), min(f.size, i + bins + 1)
        harm_mask[lo:hi] = True
        return float(P[lo:hi].sum())

    fund = band(f0)
    hk = {}
    for kk in range(2, nh + 1):
        if kk * f0 < fs / 2:
            hk[kk] = band(kk * f0)
    thd = sum(hk.values())
    inband = (f >= 20.0) & (f <= fmax)
    resid = float(P[inband & ~harm_mask].sum())
    Pn = np.where(inband & ~harm_mask, P, 0.0)
    peaks, _ = signal.find_peaks(Pn, distance=max(1, int(2.0 / df)))
    top = peaks[np.argsort(Pn[peaks])[::-1][:5]]
    lines = [(round(float(f[i]), 2), round(float(pdb(Pn[i] / fund)), 1)) for i in top if fund > 0]
    amp = np.sqrt(fund) * 2.0 / w.sum() if fund > 0 else 0.0
    out = {"fund_dbfs": float(20.0 * np.log10(max(amp, 1e-300))),
           "thd_db": float(pdb(thd / fund)) if fund > 0 else -np.inf,
           "h2_db": float(pdb(hk.get(2, 0.0) / fund)) if fund > 0 else -np.inf,
           "h3_db": float(pdb(hk.get(3, 0.0) / fund)) if fund > 0 else -np.inf,
           "h5_db": float(pdb(hk.get(5, 0.0) / fund)) if fund > 0 else -np.inf,
           "nonharm_db": float(pdb(resid / fund)) if fund > 0 else -np.inf,
           "nonharm_lines": lines}
    return out


def env_modulation(y, fs, f0, skip_s=1.0) -> dict:
    """Analytic envelope of the (channel-mean) output: p-p in dB and the modulation lines
    at 2 f0 and 4 f0 (20 log10 of the AC amplitude re the mean envelope)."""
    v = mono(y)
    k = int(skip_s * fs)
    v = v[k:-k] if v.size > 2 * k + 10 else v
    env = np.abs(signal.hilbert(v))
    e = env[int(0.1 * fs):-int(0.1 * fs)]
    m = e / e.mean() - 1.0
    n = m.size
    w = np.hanning(n)
    X = np.fft.rfft(m * w)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    A = 2.0 * np.abs(X) / w.sum()
    out = {"env_pp_db": float(20.0 * np.log10(e.max() / e.min()))}
    for mult in (2, 4):
        sel = np.abs(f - mult * f0) <= 1.0
        a = float(A[sel].max()) if sel.any() else 0.0
        out[f"env_line_{mult}f_db"] = float(20.0 * np.log10(max(a, 1e-300)))
    return out


def sidebands(y, fs, fc=1000.0, lo=20.0, hi=2000.0, excl=5.0, skip_s=0.5) -> dict:
    """Sideband energy within [lo, hi] Hz either side of fc (excluding +-excl) re the
    carrier (+-excl), dB; near (<= 200 Hz) and far parts; and the energy within +-20 Hz of
    fc +- 1 kHz (the control-grid line)."""
    v = mono(y)
    k = int(skip_s * fs)
    v = v[k:-k] if v.size > 2 * k + 10 else v
    n = v.size
    w = np.hanning(n)
    X = np.fft.rfft(v * w)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    P = np.abs(X) ** 2
    d = np.abs(f - fc)
    carrier = P[d <= excl].sum()
    sb = P[(d >= lo) & (d <= hi)].sum()
    pk = P[(np.abs(d - 1000.0) <= 20.0)].sum()
    near = P[(d >= lo) & (d <= 200.0)].sum()
    far = P[(d > 200.0) & (d <= hi)].sum()
    return {"sb_db": float(pdb(sb / carrier)), "sb_near200_db": float(pdb(near / carrier)),
            "sb_far_db": float(pdb(far / carrier)), "sb_1k_db": float(pdb(pk / carrier))}


def kick_look(g_db, fs, onsets_s, pre_ms=5.0, post_ms=80.0) -> dict:
    """Per onset: the deepest gain in [onset - pre, onset + post], its time after the onset,
    the largest drop over any 1 / 2 / 5 ms, and the recovery time to 63 % of the depth.
    Returns medians and maxima over onsets."""
    g = np.asarray(g_db, dtype=np.float64)
    pre, post = int(pre_ms * 1e-3 * fs), int(post_ms * 1e-3 * fs)
    rows = []
    for o in onsets_s:
        i = int(round(o * fs))
        a, b = max(0, i - pre), min(g.size, i + post)
        seg = g[a:b]
        if seg.size < 10:
            continue
        j = int(np.argmin(seg))
        depth = -float(seg[j])
        r = {"depth_db": depth, "t_min_ms": (a + j - i) * 1000.0 / fs}
        for ms in (1, 2, 5):
            s = int(ms * 1e-3 * fs)
            r[f"drop_{ms}ms_db"] = float(np.max(seg[:-s] - seg[s:])) if seg.size > s else 0.0
        rec = seg[j:]
        kk = np.flatnonzero(rec >= seg[j] + 0.63 * depth)
        r["rec63_ms"] = (kk[0] * 1000.0 / fs) if kk.size else float("nan")
        rows.append(r)
    out = {}
    for key in ("depth_db", "t_min_ms", "drop_1ms_db", "drop_2ms_db", "drop_5ms_db", "rec63_ms"):
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        out[key + "_med"] = float(np.nanmedian(vals)) if vals.size else float("nan")
        out[key + "_max"] = float(np.nanmax(vals)) if vals.size else float("nan")
    out["n_onsets"] = len(rows)
    return out


def band_energy_db(y, fs, lo, hi, skip_s=0.0) -> float:
    """10 log10 of the mean-square energy of the channel mean within [lo, hi] Hz (Parseval:
    a full-scale sine inside the band reads -3.01)."""
    v = mono(y)
    k = int(skip_s * fs)
    v = v[k:-k] if (k > 0 and v.size > 2 * k + 10) else v
    n = v.size
    X = np.fft.rfft(v)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    P = np.abs(X) ** 2 / (n * n)
    P[1:-1] *= 2.0
    sel = (f >= lo) & (f <= hi)
    return float(pdb(P[sel].sum()))


def third_octaves(y, fs, lo=100.0, hi=1000.0) -> list[tuple[float, float]]:
    out = []
    f = lo
    while f < hi * 1.001:
        out.append((round(f, 1), band_energy_db(y, fs, f / 2 ** (1 / 6), f * 2 ** (1 / 6))))
        f *= 2 ** (1 / 3)
    return out


def above_nyquist_fraction(u8, fs8, fs_base) -> float:
    """Energy fraction of a (mono) oversampled signal above the base-rate Nyquist."""
    v = mono(u8)
    X = np.fft.rfft(v)
    f = np.fft.rfftfreq(v.size, 1.0 / fs8)
    P = np.abs(X) ** 2
    return float(P[f > fs_base / 2].sum() / max(P.sum(), 1e-300))


def window_energy_db(y, fs, centres_s, pre_ms, post_ms, lo, hi) -> float:
    """Median over windows of the band energy in [lo, hi] Hz in [c - pre, c + post]."""
    v = mono(y)
    vals = []
    for c in centres_s:
        i = int(round(c * fs))
        a, b = max(0, i - int(pre_ms * 1e-3 * fs)), min(v.size, i + int(post_ms * 1e-3 * fs))
        if b - a < 16:
            continue
        vals.append(band_energy_db(v[a:b] * np.hanning(b - a), fs, lo, hi))
    return float(np.median(vals)) if vals else float("nan")
