"""Transient density, onset strength and kick onsets (spec 3.3 row "transients").

Contract:
- Onsets: SuperFlux (borrowed.detect_onsets, TRANSMUTATION's detector, verbatim
  logic, tested bit-identical) on the mono signal (L + R)/2 at the native
  rate, hop 512 / n_fft 2048 at 44.1 and 48 kHz (scaled by powers of two above,
  _common.fft_size): log-compressed flux against the max-filtered previous
  frame, multiplicative local-median threshold plus a global salience gate,
  attack refined to the sample, 50 ms minimum gap. Its flux sums over linear
  bins, so a kick's few low bins can lose to broadband hats under the global
  gate (43 of 96 kicks on a kick + hat fixture); the kick onsets not within
  50 ms of a SuperFlux onset are therefore added, and `onset_times` (s) is that
  union. The spec row (draft 0.4) asks for a log filterbank and a local
  threshold floor; both are in borrowed.detect_onsets_fb and both measured
  worse on the gate fixtures (brief-04 D.21), so they are not used.
  Measured with borrowed.match_onsets at 20 ms: clicks, kick_sub and kicks +
  off-beat hats P >= 0.96, R >= 0.995, mean error <= 2.6 ms at 44.1 / 48 kHz.
- `onset_strength`: the SuperFlux envelope, maximum per grid frame (frame i of
  the flux ends at sample i * hop, stamped at its window centre).
  `onset_env` / `onset_env_t`: librosa.onset.onset_strength of the mono signal
  at AR = 20480 Hz, hop 256 (12.5 ms); used only by structure.py's beat
  tracker, which librosa's beat_track is built around.
- `onset_rate` = onsets per second in a 2 s window centred on each grid frame;
  `onsets_per_s` = count / duration of the non-silent span (first to last grid
  frame with signal, or the whole file).
- Kick onsets with a score (spec 3.3 draft 0.4, audit 04 F4 / A5 / A10 / A16):
  1. Candidates from the sub + low band energy flux: the M component of the
     bands below the second crossover (400 Hz by default), resampled to 2 kHz,
     STFT with a 20 ms Hann window every 2 ms (64-point FFT, bins up to
     500 Hz), magnitudes in dB floored 80 dB below the loudest bin. Flux
     d(t) = weighted mean over bins of max(0, A(f, t) - A(f, t - 10 ms)), bins
     in 40 to 200 Hz weighted 2: a band-limited spectral flux in dB rather than
     the flux of total band energy (a sidechained bass ducks as the kick lands,
     so total sub + low energy rises by only about 5 dB on kick_sub while the
     per-bin flux peaks near 25 dB). A candidate is a local maximum of d within
     +-50 ms with d > 10 dB and band energy within 40 dB of the maximum within
     +-4 s (a local gate, not the whole track's), at least 100 ms after the
     previous one; the signal is preceded by silence so an onset at t = 0
     counts. Each candidate is moved to the steepest rise of a 1.45 ms moving
     energy (64 samples at 44.1 kHz, scaled with fs) of the low band within
     +-15 ms: on the low band, so a hat 10 ms away cannot take the refinement.
  2. Evidence, each in [0, 1]: low share s = borrowed.attack_low_share in its
     power mode (fractional bins, power share below 150 Hz; 46 ms Hann windows
     starting 5 ms before the attack, the C++ starts them at the onset where
     the window is zero), mapped 0.6 -> 0 .. 0.85 -> 1 (kicks measured >= 0.92,
     every non-kick <= 0.68); sweep (the low band's instantaneous frequency
     falls by 1.15x -> 0 .. 1.5x -> 1 from [1, 12] ms to [20, 40] ms after the
     attack); click (power above the second crossover rises 6 -> 0 .. 18 dB ->
     1 over the first 5 ms against [-25, -5] ms, times its level against the
     low band over the same 5 ms, -40 -> 0 .. -30 dB -> 1).
     score = s x max(sweep, click). `kick_onsets` are the candidates with
     score >= 0.5 (grid alignment, onset union, notch duck); `kick_mask_onsets`
     those >= 0.8, the only ones that mask bass-root frames (spec: "kick mask
     applied only above a stricter kick score").
  Measured (44.1 kHz, 20 ms match): kick_sub, kicks + off-beat hats, kicks
  struck with a loud hat, off-beat bass + kick at 128 BPM and 8th-note bass +
  kick at 140 BPM all P = R = 1.0, max error <= 0.8 ms on kick_sub; kickless
  basslines (notes on the beat or the off-beat, 5 ms attacks) 0 kicks (before:
  one per note); a click train 0. Every bass attack measured sweep 0 and click
  <= 0 after the level term (click splatter -43 dB or less of its low band;
  the synthetic kick's click -19 to -35 dB). A bass note struck together with
  a hat would still pass the click test; not tested. `kick_score` and
  `kick_low_share` are per kick.
  On five real premasters (dense mixes with a bassline in the same band, half-
  time and triplet grooves) this finds 0.05 to 0.33 kicks per beat: the
  analytic-signal sweep and the click test are swamped by the bass and hats,
  and without labelled kicks no variant tried could be verified (brief-04
  D.36). structure.kick_consistency reports the kick rate against the tempo
  hypothesis so the planner can see when to distrust it.
- `kick_mask(t, onsets, before_s, after_s)`: 1 where any onset lies within
  [t - after_s, t + before_s], else 0; used by notch (40 ms), pitch (150 ms)
  and the clipper.
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d

from . import borrowed
from ._common import AR, Context, db_power, ensure_context, fft_size, resample, window_mean

ONSET_HOP = 256
SF_HOP = 512
SF_NFFT = 2048
UNION_GAP_S = 0.05
REFINE_ENV_S = 64 / 44100  # 64 samples at 44.1 kHz, scaled with fs (audit 04 A10)
KICK_SHARE_LEAD_S = 0.005
KICK_BAND = (40.0, 200.0)
KICK_LOCAL_S = 4.0
SHARE_LO, SHARE_HI = 0.6, 0.85  # power share; 0.725 maps to 0.5 (kicks >= 0.92, others <= 0.68 measured)
SWEEP_LO, SWEEP_HI = 1.15, 1.5
CLICK_LO_DB, CLICK_HI_DB = 6.0, 18.0
CLICK_REL_LO_DB, CLICK_REL_HI_DB = -40.0, -30.0
KICK_SCORE_MIN = 0.5   # reported as a kick (grid alignment, onset union)
KICK_MASK_SCORE = 0.8  # strict: only these mask bass-root frames (spec 3.3, audit 04 F4)
REFINE_BEFORE_S = 0.015
REFINE_AFTER_S = 0.015
RATE_WINDOW_S = 2.0
KICK_RATE = 2000
KICK_FMAX = 500.0
KICK_WIN_S = 0.020
KICK_STEP_S = 0.002
KICK_LAG_S = 0.010
KICK_FLOOR_DB = 80.0
KICK_FLUX_DB = 10.0
KICK_OFFSET_S = 0.0
KICK_RANGE_DB = 40.0
KICK_NEIGHBOUR_S = 0.050
KICK_MIN_GAP_S = 0.100


def onset_envelope(ctx: Context) -> tuple[np.ndarray, np.ndarray]:
    if "onset_env" not in ctx.cache:
        y = ctx.mono_ar.astype(np.float32)
        env = librosa.onset.onset_strength(y=y, sr=AR, hop_length=ONSET_HOP)
        t = np.arange(env.size) * ONSET_HOP / AR
        ctx.cache["onset_env"] = (t, env.astype(np.float64))
    return ctx.cache["onset_env"]


def kick_onsets(ctx: Context) -> tuple[np.ndarray, dict]:
    """Kick onsets (s) with a score; see the module docstring. Returns the onsets
    whose score >= KICK_SCORE_MIN and a debug dict with every candidate's time,
    low share, sweep and click evidence and score."""
    sig = resample(ctx.low_band_m, ctx.fs, KICK_RATE)
    dur = sig.shape[0] / KICK_RATE
    nper = int(round(KICK_WIN_S * KICK_RATE))
    hop = int(round(KICK_STEP_S * KICK_RATE))
    # leading silence so an onset at t = 0 has a "before" to rise from
    pre = nper + int(round(KICK_LAG_S * KICK_RATE))
    f, t, z = signal.stft(np.pad(sig, (pre, 0)), KICK_RATE, window="hann", nperseg=nper,
                          noverlap=nper - hop, nfft=64, boundary="zeros", padded=True)
    t = t - pre / KICK_RATE
    sel = f <= KICK_FMAX
    a = 20.0 * np.log10(np.abs(z[sel]) + 1e-12)
    a = np.maximum(a, a.max() - KICK_FLOOR_DB)
    e = db_power(np.mean(np.abs(z[sel]) ** 2, axis=0))
    lag = int(round(KICK_LAG_S / KICK_STEP_S))
    # kick-range bins (40 to 200 Hz) weigh double in the flux (audit 04 A16)
    wb = np.where((f[sel] >= KICK_BAND[0]) & (f[sel] <= KICK_BAND[1]), 2.0, 1.0)
    d = np.zeros(a.shape[1])
    d[lag:] = (np.maximum(a[:, lag:] - a[:, :-lag], 0.0) * wb[:, None]).sum(axis=0) / wb.sum()
    nb = int(round(KICK_NEIGHBOUR_S / KICK_STEP_S))
    local_max = d >= maximum_filter1d(d, size=2 * nb + 1, mode="nearest")
    inside = (t >= -KICK_WIN_S / 2.0) & (t <= dur - KICK_WIN_S / 2.0)
    # energy gate relative to the local (+-4 s) maximum, not the whole track's
    e_loc = maximum_filter1d(e, size=2 * int(round(KICK_LOCAL_S / KICK_STEP_S)) + 1, mode="nearest")
    cand = np.nonzero(inside & local_max & (d > KICK_FLUX_DB) & (e > e_loc - KICK_RANGE_DB))[0]
    out = []
    last = -np.inf
    for i in cand:
        ti = t[i] - KICK_OFFSET_S
        if ti - last >= KICK_MIN_GAP_S:
            out.append(ti)
            last = ti
    cand_t = np.maximum(np.array(out, dtype=np.float64), 0.0)
    mono = 0.5 * (ctx.x[:, 0] + ctx.x[:, 1])
    low = ctx.low_band_m
    cand_s = _refine(low, ctx.fs, cand_t)  # on the low band: a hat 10 ms away cannot win
    cand_t = cand_s / ctx.fs
    share = borrowed.attack_low_share(mono, ctx.fs, cand_s, lead_s=KICK_SHARE_LEAD_S)
    sweep, click = _kick_shape(low, _high(ctx), ctx.fs, cand_s)
    s_share = np.clip((np.nan_to_num(share, nan=0.0) - SHARE_LO) / (SHARE_HI - SHARE_LO), 0.0, 1.0)
    score = s_share * np.maximum(sweep, click)
    keep = score >= KICK_SCORE_MIN
    return cand_t[keep], {"t": t, "energy_db": e, "flux_db": d, "candidates": cand_t,
                          "low_share": share, "sweep": sweep, "click": click, "score": score,
                          "kept": keep}


def _high(ctx: Context) -> np.ndarray:
    """M component of the bands above the second crossover (>= 400 Hz by default)."""
    if "high_m" not in ctx.cache:
        x_m = ctx.ms[:, 0]
        ctx.cache["high_m"] = x_m - ctx.low_band_m
    return ctx.cache["high_m"]


def _kick_shape(low: np.ndarray, high: np.ndarray, fs: float, starts: np.ndarray):
    """Per candidate, (sweep, click) evidence in [0, 1].

    sweep: instantaneous frequency of the low band (analytic signal) as the
    median over [1, 12] ms after the attack against [20, 40] ms; ratio 1.15
    gives 0, 1.5 or more gives 1 (a kick's pitch falls; a bass note's does not).
    click: mean power above the second crossover over the first 5 ms against
    [-25, -5] ms (a rise of 6 dB gives 0, 18 dB or more 1), times its level
    against the low band over the same 5 ms (-40 dB gives 0, -30 dB 1): a short
    bass attack rising from silence splatters above 500 Hz too, but at -43 dB
    or less of its low band, where the synthetic kick's click sits at -19 to
    -35 dB."""
    n = low.size
    sweep = np.zeros(starts.size)
    click = np.zeros(starts.size)
    ms = lambda v: int(round(v * 1e-3 * fs))
    for i, s in enumerate(np.asarray(starts, dtype=np.int64)):
        a, b = max(0, s - ms(20)), min(n, s + ms(60))
        if b - a > ms(45):
            z = signal.hilbert(low[a:b])
            ph = np.unwrap(np.angle(z))
            inst = np.diff(ph) * fs / (2 * np.pi)
            def med(t0, t1):
                u, v = s - a + ms(t0), s - a + ms(t1)
                seg = inst[max(0, u):max(0, v)]
                return float(np.median(seg)) if seg.size else np.nan
            fe, fl = med(1, 12), med(20, 40)
            if np.isfinite(fe) and np.isfinite(fl) and fl > 1.0:
                sweep[i] = np.clip((fe / fl - SWEEP_LO) / (SWEEP_HI - SWEEP_LO), 0.0, 1.0)
        p_on = high[s:min(n, s + ms(5))]
        p_pre = high[max(0, s - ms(25)):max(0, s - ms(5))]
        l_on = low[s:min(n, s + ms(5))]
        if p_on.size:
            e_on = float(np.mean(p_on ** 2))
            e_pre = float(np.mean(p_pre ** 2)) if p_pre.size else 0.0
            e_low = float(np.mean(l_on ** 2))
            rise = 10 * np.log10(e_on / e_pre) if e_pre > 0 and e_on > 0 else (60.0 if e_on > 0 else 0.0)
            rel = 10 * np.log10(e_on / e_low) if e_on > 0 and e_low > 0 else -200.0
            click[i] = (np.clip((rise - CLICK_LO_DB) / (CLICK_HI_DB - CLICK_LO_DB), 0.0, 1.0)
                        * np.clip((rel - CLICK_REL_LO_DB) / (CLICK_REL_HI_DB - CLICK_REL_LO_DB), 0.0, 1.0))
    return sweep, click


def _refine(mono: np.ndarray, fs: float, t: np.ndarray) -> np.ndarray:
    """Move each time to the steepest rise of a 64-sample moving energy within
    t +- 15 ms (the refinement of borrowed.detect_onsets), as sample
    indices. The attack-share windows must start at the attack: a few ms late,
    a click's window holds only the noise after it."""
    n = mono.size
    cs = np.concatenate([[0.0], np.cumsum(mono * mono)])
    h = max(1, int(round(REFINE_ENV_S * fs)) // 2)
    out = np.empty(t.size, dtype=np.int64)
    for i, ti in enumerate(t):
        c = int(round(ti * fs))
        # centres j in [a, b): energy over [j - h, j + h), fully inside the signal
        a = max(h, c - int(round(REFINE_BEFORE_S * fs)))
        b = min(n - h, c + int(round(REFINE_AFTER_S * fs)))
        if b - a < 2:
            out[i] = min(max(c, 0), n - 1)
            continue
        j = np.arange(a, b)
        e = cs[j + h] - cs[j - h]
        out[i] = a + 1 + int(np.argmax(np.diff(e)))
    return out


def kick_mask(t, onsets, before_s: float, after_s: float) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    onsets = np.sort(np.asarray(onsets, dtype=np.float64))
    if onsets.size == 0:
        return np.zeros(t.shape)
    # nearest onset at or after t - after_s
    i = np.searchsorted(onsets, t - after_s, side="left")
    i = np.clip(i, 0, onsets.size - 1)
    o = onsets[i]
    return ((o >= t - after_s) & (o <= t + before_s)).astype(np.float64)


def superflux_onsets(ctx: Context) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(onset times s, flux frame times s, flux) of SuperFlux on (L + R)/2."""
    mono = 0.5 * (ctx.x[:, 0] + ctx.x[:, 1])
    hop, nfft = fft_size(SF_HOP, ctx.fs), fft_size(SF_NFFT, ctx.fs)  # scaled with fs (audit 04 A10)
    on = borrowed.detect_onsets(mono, ctx.fs, hop, nfft) / ctx.fs
    sf = borrowed.superflux(mono, ctx.fs, hop, nfft)
    if sf is None:
        return on, np.zeros(0), np.zeros(0)
    flux, nf = sf
    tf = (np.arange(nf) * hop - nfft / 2.0) / ctx.fs  # window centre
    return on, tf, flux


def analyse(x: np.ndarray, fs: float, ctx: Context | None = None) -> dict:
    ctx = ensure_context(x, fs, ctx)
    g = ctx.grid
    te, env = onset_envelope(ctx)
    sf_on, tf, flux = superflux_onsets(ctx)
    strength = np.zeros(g.k)
    if flux.size:
        np.maximum.at(strength, g.index(np.maximum(tf, 0.0)), flux)
    kicks, kdbg = kick_onsets(ctx)
    add = [k for k in kicks if sf_on.size == 0 or np.min(np.abs(sf_on - k)) > UNION_GAP_S]
    onsets = np.sort(np.concatenate([sf_on, np.asarray(add, dtype=np.float64)]))
    half = RATE_WINDOW_S / 2.0
    lo = np.searchsorted(onsets, g.times - half)
    hi = np.searchsorted(onsets, g.times + half)
    rate = (hi - lo) / RATE_WINDOW_S
    p = np.mean(ctx.x ** 2, axis=1)
    active = window_mean(p, g.centres, int(round(g.hop_s * ctx.fs))) > 1e-10
    if np.any(active):
        k0, k1 = np.nonzero(active)[0][[0, -1]]
        dur = (k1 - k0 + 1) * g.hop_s
    else:
        dur = ctx.x.shape[0] / ctx.fs
    return {
        "t": g.times, "onset_strength": strength, "onset_times": onsets,
        "superflux_onsets": sf_on, "onset_rate": rate,
        "onsets_per_s": float(onsets.size / dur) if dur > 0 else 0.0,
        "onset_env": env, "onset_env_t": te, "kick_onsets": kicks,
        "kick_low_share": kdbg["low_share"][kdbg["kept"]],
        "kick_score": kdbg["score"][kdbg["kept"]],
        "kick_mask_onsets": kicks[kdbg["score"][kdbg["kept"]] >= KICK_MASK_SCORE],
        "kick_debug": kdbg,
    }
