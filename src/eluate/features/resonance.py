"""Persistent narrowband resonance detection (spec 3.3 row "resonance", draft 0.4).

Contract:
- Channels (audit 04 A4): the whole detector runs on M, on S and on the stereo
  power (|X_L|^2 + |X_R|^2) / 2 separately; a channel that is all zeros is
  skipped. A +9 dB ring in S alone at S/M -10 dB was invisible in the stereo
  power. Each reported resonance has `channel` "M" (seen in M, not S), "S" (S,
  not M) or "both", and `channels_seen`, so the notch stage cuts only there.
- Spectrum per channel: multi-resolution STFT with a Hann window, N = 16384
  for 30 to 500 Hz, 4096 for 500 to 2500 Hz, 2048 from 2500 Hz to
  min(20 kHz, 0.95 fs/2) (N scaled by powers of two at 88.2 kHz and above,
  _common.fft_size), hop N/4; each resolution only reports peaks whose f0 lies
  in its own range. Each frame's spectrum is normalised to unit power (so a
  louder neighbouring section cannot drag its shape across a level step), then
  averaged over time by a centred moving mean of max(0.5 s, 8 N / fs) (0.5 s at
  N = 2048 and 4096, about 3 s at 16384 / 44.1 kHz) with frames weighted by
  activity: frames more than 60 dB below the loudest neither count nor dilute
  (audit 04 A4), and a frame is active when its window is mostly active. That
  averages roughly 8 to 20 independent periodograms depending on the
  resolution, about 1.5 to 2 dB of residual ripple (audit 04 F8 corrected the
  earlier "16, 1 dB"); persistence, not smoothing, carries the noise rejection.
  Below 500 Hz a second pass with 0.5 s smoothing (audit 04 F7) adds its peaks
  at a 9 dB excess threshold (at 6 dB its ~2 periodograms fire on noise); on
  the tested fixtures it adds no detection the long pass misses
  (brief-04 D.24).
- Baseline (spec 3.3 draft 0.4: flank median): per frame, the mean of two
  medians in dB over the flanks [f 2^-1/2, f 2^-1/6] and [f 2^1/6, f 2^1/2] (a
  1-octave window with its central 1/3 octave excluded; a linear slope
  cancels). Where a flank has fewer than 3 bins or runs past the kept bins
  (near Nyquist, at the bottom of a resolution) the other flank alone is used
  (audit 04 A4). On a 1/24-octave grid, interpolated in log f to the bins;
  BASELINE_MODE = "median" gives a centred median of width BASELINE_OCT for
  comparison. Excess E(f, t) = level - baseline.
- Peaks: per frame, local maxima of the level with E > 6 dB in active frames;
  a peak inside the -3 dB band of a higher peak in the same frame is dropped.
  f0 by parabolic interpolation of the three dB values around the maximum.
  The per-frame Q test (Q > 5) uses the -3 dB width of the level peak
  (`q_peak` reports the median); it is a peakiness test, not the reported Q.
- Candidates: per channel and resolution, peaks are clustered greedily by
  frequency (repeatedly the frequency with the most peak-seconds within
  max(2 % of f, 1.5 bins), centre = median f0, until less than 1 s remains).
  Merge (audit 04 A3): candidates of any resolution and channel join a
  stronger one only when each one's centre lies inside the other's half-power
  band f0 (1 +- 1 / 2Q); a narrow partial inside a broad ring's band stays
  separate (the old 5 % rule absorbed a Q 18 ring at 3190 Hz into a 220 Hz
  note's partials). Merged f0 = peak-seconds-weighted mean; Q recomputed in the
  strongest member's resolution over the merged frames.
- Q (audit 04 F5 / A2): a peaking response 10 log10((A^2 + x^2) / (1 + x^2)) - d,
  x = Qs (f / f0 - f0 / f), fitted by least squares to the excess averaged in
  dB over the candidate's frames within f0 2^(+-1/2); reported `q` = Qs /
  sqrt(A), the width at half the dB gain, which is the notch stage's convention
  (plan.Notch). `q_uncertainty` comes from the fit covariance; it understates
  the error on broad rings (measured: Q 12 / 20 rings within 4 % of analytic,
  a Q 6 ring 27 % high with a stated +-0.11).
- Persistence: presence on the common grid (share of the candidate's frames in
  each grid frame >= 0.5, nearest frame where none), then per section
  (`sections`, whole track if None) the share of the section's grid frames
  present; persistent if any section exceeds 30 %. Below 500 Hz the 3 s
  smoothing widens presence by about +-1.5 s (a ring 30 % of the time read
  about 0.4). Spans: frame presence after closing gaps < 0.25 s and dropping
  runs < 0.25 s, unioned over the merged members.
- Classification (`flag`, `depth_scale`):
  Order: sub_exempt, tonal, musical, resonance.
  - "sub_exempt" (depth 0): f0 < 80 Hz, or 80 to 150 Hz with no section at
    >= 50 % persistence (spec 3.3 draft 0.4, audit 04 F6). From 80 to 150 Hz
    the depth scale of a flagged candidate is halved.
  - "musical" (depth MUSICAL_DEPTH_SCALE = 0.25): the harmonic-ladder check
    (`_ladder`, COHERENCE = "ladder") passes: the best area-normalised ladder
    share >= LADDER_MIN = 0.1 over fundamentals f0 / m (m up to 64 and below
    the candidate's Q), stiff-string partial positions f1 k sqrt(1 + B k^2) for
    B from 0 to 0.0128, +-60 / m cents per partial, a ladder fundamental and at
    least one neighbouring partial required; see `_ladder`. Measured over 122
    candidates: harmonic, detuned and piano-like partials 0.448 to 0.823, rings
    in noise or in the arrangement 0.000 to 0.004, rings between a note's
    partials 0.000 (the top partial of a B = 0.01 bell reads 0.071 and is
    flagged; brief-04 D.25). The share is reported (`ladder_fraction`, with
    `ladder_m`, `ladder_b`). COHERENCE = "peaks" is the revision-1 rule
    (partner peaks at 2-4 f0, f0/2-4 and the series extension), reported per
    candidate in `partials` / `series`; it calls rings between partials
    musical.
  - "tonal" (depth 0; real-material pass, brief-04 D.33): a held tone, not a
    resonance: the candidate is a line (its per-frame -3 dB width at most 3
    bins, i.e. as narrow as the window) or its excess exceeds 24 dB. A
    resonance raises broadband content by its filter's gain and is as wide as
    the filter; on five real premasters every candidate above 24 dB was a bass
    or chord tone (40 to 66 dB).
  - "musical" also when the chord-invariance check passes (`_chord_tone`, with
    `key` and `chords` from pitch.py): the pitch class within 35 cents of equal
    temperament is in the key's tonic triad or in the triad of at least half of
    the bars the candidate is present in.
  - "resonance" (depth 1): otherwise.
  Dropped before classification: candidates whose reliable peaking fit has a
  half-dB Q below 2 (a shelf or a bump, e.g. at a mix's high-frequency rolloff).
- Output `analyse(...)["resonances"]`: list of dicts, persistent candidates
  only, sorted by f0: f0, q, q_uncertainty, q_peak, start, end, spans,
  excess_db (common grid, 0 where absent), excess_max_db, persistence (per
  section), flag, depth_scale, channel, channels_seen, partials, series,
  resolution (N), ladder_fraction, ladder_m, ladder_b, members, line,
  bw_peak_hz, chord_tone, chord_share.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import least_squares

from . import borrowed
from ..bands import INV_SQRT2
from ._common import Context, ensure_context, fft_size

RESOLUTIONS = ((16384, 30.0, 500.0), (4096, 500.0, 2500.0), (2048, 2500.0, 20000.0))
EXCESS_DB = 6.0
Q_MIN = 5.0
PERSISTENCE = 0.30
TOL_REL = 0.02
TOL_BINS = 1.5
SUB_EXEMPT_HZ = 80.0
LOW_RULE_HZ = 150.0  # 80 to 150 Hz: persistence >= 0.5 and half the depth
LOW_PERSISTENCE = 0.5
LOW_DEPTH_SCALE = 0.5
SMOOTH_S = 0.5
SMOOTH_FRAMES = 8  # window lengths: about 16 independent periodograms at 50 % overlap
FRAME_RANGE_DB = 60.0
MIN_CLUSTER_S = 1.0
GAP_CLOSE_S = 0.25
MIN_RUN_S = 0.25
PARTIAL_DB = 3.0
PARTIAL_FRACTION = 0.5
PARTIAL_TOL_REL = 0.006  # partials sit on exact ratios; 2 % would match neighbours
MUSICAL_DEPTH_SCALE = 0.25
PEAK_CHUNK_ELEMS = 1 << 22   # peaks x span elements per chunk in _Res.peaks (memory only)
LINE_BINS = 3.0         # -3 dB width at or under 3 bins: a line, i.e. a held tone (Hann main lobe 1.44)
EXCESS_TONAL_DB = 24.0  # above this a peak is a sustained tone, not a filter boost (brief-04 D.33)
Q_FIT_MIN = 2.0         # a fitted half-dB Q below this is a shelf or a bump, not narrowband
CHORD_CENTS = 35.0
CHORD_SHARE = 0.5
COHERENCE = "ladder"  # "ladder" (default, borrowed HNR ladder), "peaks" or "either"
LADDER_MIN = 0.1
LADDER_MAX_M = 64
LADDER_CENTS = 60.0  # at m = 1; +-60 / m cents at f0 / m
LADDER_B = (0.0,) + tuple(1e-4 * 2 ** (k / 4) for k in range(29))  # 0, 1e-4 .. 1.28e-2 in 2^(1/4) steps
CHANNELS = ("M", "S", "stereo")
SHORT_PASS_BELOW_HZ = 500.0
SHORT_SMOOTH_S = 0.5
SHORT_EXCESS_DB = 9.0  # about 2 periodograms in 0.5 s at N = 16384: 6 dB fires on noise
FLANK_MIN_BINS = 3
SPEC_PARTIALS = (2.0, 3.0, 4.0, 1 / 2, 1 / 3, 1 / 4)
SERIES_M = range(2, 33)
BASE_STEP_OCT = 1.0 / 24.0
BASELINE_MODE = "flank"  # "flank" (default) or "median" (spec: 1/3-octave median); brief-04 D.2
BASELINE_OCT = 1.0  # total window width, octaves
BASELINE_CORE_OCT = 1.0 / 3.0  # excluded centre ("flank" mode)


class _Res:
    """One STFT resolution of one channel: smoothed dB level and excess."""

    def __init__(self, x: np.ndarray, fs: float, n_ref: int, f_lo: float, f_hi: float,
                 smooth_s: float | None = None):
        self.n = fft_size(n_ref, fs)
        self.hop = self.n // 4
        self.fs = fs
        self.df = fs / self.n
        self.f_lo, self.f_hi = f_lo, min(f_hi, 0.95 * fs / 2.0)
        k0 = max(1, int(np.floor(self.f_lo * 2 ** (-BASELINE_OCT) / self.df)))
        k1 = min(self.n // 2, int(np.ceil(self.f_hi * 2 ** BASELINE_OCT / self.df)) + 1)
        self.k0, self.k1 = k0, k1
        self.f = np.arange(k0, k1) * self.df
        p, self.t = self._power(x)
        if smooth_s is None:
            smooth_s = max(SMOOTH_S, SMOOTH_FRAMES * self.n / fs)
        self.smooth_s = smooth_s
        smooth = max(1, int(round(smooth_s * fs / self.hop)))
        tot = p.sum(axis=1)
        act = (tot > tot.max() * 10 ** (-FRAME_RANGE_DB / 10)) if tot.max() > 0 else np.zeros(tot.shape, bool)
        # activity-weighted mean of the unit-power spectral shapes: silent frames
        # neither count nor dilute (audit 04 A4); a frame is active when its
        # window's weight is mostly active
        w = act.astype(np.float64)
        wsum = uniform_filter1d(w, smooth, mode="nearest")
        self.active = wsum > 0.5
        # in place, same operations in the same order as the expression form
        # p / max(tot) * w, smooth, / max(wsum), 10 log10(max(., 0)), max(., -300):
        # bit-identical, one frames x bins float64 array alive instead of five
        p /= np.maximum(tot, 1e-300)[:, None]
        p *= w[:, None]
        pn = uniform_filter1d(p, smooth, axis=0, mode="nearest")
        del p
        pn /= np.maximum(wsum, 1e-12)[:, None]
        np.maximum(pn, 0.0, out=pn)
        with np.errstate(divide="ignore"):
            np.log10(pn, out=pn)
        pn *= 10.0
        np.maximum(pn, -300.0, out=pn)
        self.db = pn.astype(np.float32)
        del pn
        base = self._baseline()
        self.ex = (self.db - base).astype(np.float32)
        del base
        self.frame_s = self.hop / fs

    def _power(self, x):
        """Mean over the columns of x (one channel, or L and R) of |X|^2."""
        n, hop = self.n, self.hop
        xp = np.pad(x, ((n // 2, n // 2 + hop), (0, 0)))
        frames = 1 + (x.shape[0] + hop - 1) // hop
        win = np.hanning(n + 1)[:-1]
        out = np.empty((frames, self.k1 - self.k0), dtype=np.float64)
        chunk = max(1, (1 << 22) // n)
        view = np.lib.stride_tricks.sliding_window_view(xp, n, axis=0)[::hop]  # (frames, ch, n)
        for a in range(0, frames, chunk):
            b = min(frames, a + chunk)
            spec = np.fft.rfft(view[a:b] * win, axis=-1)[..., self.k0:self.k1]
            out[a:b] = np.mean(np.abs(spec) ** 2, axis=1)
        t = np.arange(frames) * hop / self.fs
        return out, t

    def _baseline(self):
        lf = np.log2(self.f)
        grid = np.arange(lf[0], lf[-1] + BASE_STEP_OCT, BASE_STEP_OCT)
        vals = np.empty((self.db.shape[0], grid.size), dtype=np.float32)
        half, core = BASELINE_OCT / 2, BASELINE_CORE_OCT / 2
        nbin = lf.size
        for i, g in enumerate(grid):
            if BASELINE_MODE == "median":
                a = np.searchsorted(lf, g - half)
                b = max(a + 1, np.searchsorted(lf, g + half))
                vals[:, i] = np.median(self.db[:, a:b], axis=1)
                continue
            # flanks: medians either side of the excluded core, averaged (their
            # centres are symmetric in log f, so a linear slope cancels)
            a0 = min(np.searchsorted(lf, g - half), nbin - 1)
            a1 = max(np.searchsorted(lf, g - core), a0 + 1)
            b0 = min(np.searchsorted(lf, g + core), nbin - 1)
            b1 = max(np.searchsorted(lf, g + half), b0 + 1)
            lo_ok = a1 - a0 >= FLANK_MIN_BINS and lf[a0] <= g - half + BASE_STEP_OCT
            hi_ok = b1 - b0 >= FLANK_MIN_BINS and lf[min(b1, nbin) - 1] >= g + half - BASE_STEP_OCT
            lo_m = np.median(self.db[:, a0:a1], axis=1)
            hi_m = np.median(self.db[:, b0:b1], axis=1)
            if lo_ok and not hi_ok:  # near the top of the kept bins: mirror the lower flank
                vals[:, i] = lo_m
            elif hi_ok and not lo_ok:
                vals[:, i] = hi_m
            else:
                vals[:, i] = 0.5 * (lo_m + hi_m)
        out = np.empty_like(self.db)
        for j in range(self.db.shape[0]):
            out[j] = np.interp(lf, grid, vals[j])
        return out

    def q_half(self, f0: float, frames: np.ndarray) -> tuple[float, float, float]:
        """(Q, Q uncertainty, fitted gain dB) of the resonance at f0 over `frames`.

        A peaking response E(f) = 10 log10((A^2 + x^2) / (1 + x^2)) - d,
        x = Qs (f / f0' - f0' / f), is fitted by least squares to the excess
        spectrum averaged in dB over the frames, within f0 * 2^(+-1/2). The
        offset d absorbs the baseline lift from the ring's own skirts, which
        reach into the flanks (a direct half-height reading of the excess read
        Q 9.8 for an analytic 7.15). Returned Q is the fitted shape's width at
        half its dB gain, Qs / sqrt(A): the notch stage's convention (plan.Notch,
        bandwidth at half the deepest dB cut). Uncertainty: from the fit's
        covariance (residual variance x (J^T J)^-1), propagated to Qs / sqrt(A).
        Falls back to the direct half-height width when the fit fails."""
        frames = np.asarray(frames, dtype=np.int64)
        if frames.size == 0:
            return float("nan"), float("nan"), float("nan")
        e = self.ex[frames].astype(np.float64).mean(axis=0)
        sel = (self.f >= f0 * 2 ** -0.5) & (self.f <= f0 * 2 ** 0.5)
        fsel, esel = self.f[sel], e[sel]
        q0, peak = self._width_q(e, f0)
        if fsel.size < 6 or not np.isfinite(peak) or peak <= 0:
            return q0, float("nan"), peak
        qi = q0 * np.sqrt(10 ** (peak / 20)) if np.isfinite(q0) else 10.0

        def model(p):
            g, qs, fc, d = p
            xq = qs * (fsel / fc - fc / fsel)
            a2 = 10 ** (g / 10)
            return 10 * np.log10((a2 + xq * xq) / (1 + xq * xq)) - d

        try:
            sol = least_squares(lambda p: model(p) - esel, [peak + 1.0, qi, f0, 1.0],
                                bounds=([0.5, 1.0, f0 * 0.97, -6.0], [40.0, 500.0, f0 * 1.03, 12.0]))
        except (ValueError, np.linalg.LinAlgError):
            return q0, float("nan"), peak
        g, qs, fc, d = sol.x
        qh = qs / np.sqrt(10 ** (g / 20))
        unc = float("nan")
        j = sol.jac
        dof = max(1, fsel.size - 4)
        try:
            cov = np.linalg.inv(j.T @ j) * float(np.sum(sol.fun ** 2)) / dof
            # d(qh) = d(qs) / sqrt(A) - qh * ln(10) / 40 * d(g)
            grad = np.array([-qh * np.log(10) / 40.0, 1.0 / np.sqrt(10 ** (g / 20)), 0.0, 0.0])
            unc = float(np.sqrt(max(0.0, grad @ cov @ grad)))
        except np.linalg.LinAlgError:
            pass
        return float(qh), unc, float(g)

    def _width_q(self, e: np.ndarray, f0: float) -> tuple[float, float]:
        k = int(np.clip(np.argmin(np.abs(self.f - f0)), 1, self.f.size - 2))
        # the local maximum next to f0
        while k + 1 < self.f.size - 1 and e[k + 1] > e[k] and abs(self.f[k + 1] - f0) < 2 * self.df:
            k += 1
        while k - 1 > 0 and e[k - 1] > e[k] and abs(self.f[k - 1] - f0) < 2 * self.df:
            k -= 1
        peak = e[k]
        if not np.isfinite(peak) or peak <= 0:
            return float("nan"), float(peak)
        half = peak / 2.0
        edges = []
        for side in (-1, 1):
            j = k
            while 0 < j < self.f.size - 1 and e[j + side] > half:
                j += side
            nxt = j + side
            if not 0 <= nxt < self.f.size:
                return float("nan"), float(peak)
            frac = (e[j] - half) / max(e[j] - e[nxt], 1e-12)
            edges.append(self.f[j] + side * frac * self.df)
        bw = edges[1] - edges[0]
        return (float(f0 / bw) if bw > 0 else float("nan")), float(peak)

    def tol(self, f):
        return np.maximum(TOL_REL * np.asarray(f), TOL_BINS * self.df)

    def in_range(self, f) -> bool:
        return self.f_lo <= f < self.f_hi

    def peaks(self, excess_db: float = EXCESS_DB, whole: bool = False):
        """Arrays (frame, f0, excess, q) of qualifying peaks: local maxima with
        excess > excess_db and Q > Q_MIN, f0 in this resolution's range (or in
        all kept bins when `whole`)."""
        db, ex = self.db, self.ex
        sel = np.ones(self.f.size, bool) if whole else (self.f >= self.f_lo) & (self.f < self.f_hi)
        ks = np.nonzero(sel)[0]
        ks = ks[(ks >= 1) & (ks <= db.shape[1] - 2)]
        mid = db[:, ks]
        is_pk = (mid > db[:, ks - 1]) & (mid >= db[:, ks + 1]) & (ex[:, ks] > excess_db)
        is_pk &= self.active[:, None]
        fr, ki = np.nonzero(is_pk)
        if fr.size == 0:
            return (np.zeros(0, int),) + (np.zeros(0),) * 3
        k = ks[ki]
        a = db[fr, k - 1].astype(np.float64)
        b = db[fr, k].astype(np.float64)
        c = db[fr, k + 1].astype(np.float64)
        den = a - 2 * b + c
        p = np.where(den < 0, 0.5 * (a - c) / np.where(den < 0, den, -1.0), 0.0)
        p = np.clip(p, -0.5, 0.5)
        f0 = self.f[0] + (k + p) * self.df
        peak_db = b - 0.25 * (a - c) * p
        thr = peak_db - 3.0
        span = int(np.ceil(self.f_hi / (2.0 * Q_MIN * self.df))) + 2
        nbin = db.shape[1]
        edges = [np.empty(fr.size), np.empty(fr.size)]
        offs = np.arange(1, span + 1)
        # rows (peaks) are independent: in chunks the (peaks x span) work arrays stay
        # bounded (all at once they reached 3.5 GB on a 7-minute 48 kHz track); the
        # result is bit-identical to the one-shot form
        step = max(1, PEAK_CHUNK_ELEMS // span)
        for r0 in range(0, fr.size, step):
            r1 = min(fr.size, r0 + step)
            kc, frc, thc, bc = k[r0:r1], fr[r0:r1], thr[r0:r1], b[r0:r1]
            for e, side in zip(edges, (-1, 1)):
                idx = kc[:, None] + side * offs[None, :]
                valid = (idx >= 0) & (idx < nbin)
                vals = np.where(valid, db[frc[:, None], np.clip(idx, 0, nbin - 1)], np.inf)
                below = vals < thc[:, None]
                found = below.any(axis=1)
                j = np.argmax(below, axis=1)  # first offset below threshold
                v_out = vals[np.arange(j.size), j]
                v_in = np.where(j > 0, vals[np.arange(j.size), np.maximum(j - 1, 0)], bc)
                frac = np.where(v_in > v_out, (v_in - thc) / np.maximum(v_in - v_out, 1e-12), 1.0)
                pos = kc + side * (j + frac)  # j is 0-based offset-1
                e[r0:r1] = np.where(found, pos, np.nan)
        bw = (edges[1] - edges[0]) * self.df
        q = f0 / bw
        ok = np.isfinite(q) & (q > Q_MIN)
        fr, f0, k, q, peak_db = fr[ok], f0[ok], k[ok], q[ok], peak_db[ok]
        lo_f = self.f[0] + edges[0][ok] * self.df
        hi_f = self.f[0] + edges[1][ok] * self.df
        keep = _suppress_inside(fr, f0, peak_db, lo_f, hi_f)
        return fr[keep], f0[keep], ex[fr[keep], k[keep]].astype(np.float64), q[keep]

    def partial_present(self, g: float, frames: np.ndarray) -> float:
        """Fraction of `frames` holding a partial peak (excess > PARTIAL_DB,
        Q > Q_MIN) within tol of g."""
        if frames.size == 0:
            return 0.0
        if not hasattr(self, "_pp"):
            fr, f0, _, _ = self.peaks(PARTIAL_DB, whole=True)
            o = np.argsort(f0)
            self._pp = (fr[o], f0[o])
        fr, f0 = self._pp
        tol = max(PARTIAL_TOL_REL * g, TOL_BINS * self.df)
        a, b = np.searchsorted(f0, g - tol), np.searchsorted(f0, g + tol, side="right")
        hit = np.zeros(self.t.size, dtype=bool)
        hit[fr[a:b]] = True
        return float(np.mean(hit[frames]))


def _suppress_inside(fr, f0, peak_db, lo_f, hi_f, reach: int = 4) -> np.ndarray:
    """Per frame, drop a peak lying inside the -3 dB band of a higher peak:
    noise ripple on a broad resonance's top would otherwise split it."""
    keep = np.ones(f0.size, dtype=bool)
    if f0.size == 0:
        return keep
    o = np.lexsort((f0, fr))
    frs, fs_, pk, lo, hi = fr[o], f0[o], peak_db[o], lo_f[o], hi_f[o]
    sup = np.zeros(f0.size, dtype=bool)
    for d in range(1, reach + 1):
        for sgn in (-1, 1):
            j = np.arange(f0.size) + sgn * d
            v = (j >= 0) & (j < f0.size)
            jj = np.clip(j, 0, f0.size - 1)
            same = v & (frs[jj] == frs)
            inside = (fs_ >= lo[jj]) & (fs_ <= hi[jj])
            sup |= same & inside & (pk[jj] > pk)
    keep[o] = ~sup
    return keep


def _band(c: dict) -> tuple[float, float]:
    q = c["q"] if np.isfinite(c["q"]) and c["q"] > 0 else Q_MIN
    return c["f0"] * (1 - 0.5 / q), c["f0"] * (1 + 0.5 / q)


def _merge(cands: list[dict]) -> list[list[dict]]:
    """Group candidates (any resolution, any channel) whose half-power bands
    contain each other's centres (mutual containment), strongest first. A
    narrow partial inside a broad ring's band is not merged into it: the ring's
    centre is not inside the partial's band (audit 04 A1 / A3)."""
    order = sorted(cands, key=lambda c: -c["weight"])
    groups: list[list[dict]] = []
    for c in order:
        lo_c, hi_c = _band(c)
        for grp in groups:
            r = grp[0]
            lo_r, hi_r = _band(r)
            if lo_r <= c["f0"] <= hi_r and lo_c <= r["f0"] <= hi_c:
                grp.append(c)
                break
        else:
            groups.append([c])
    return groups


def _cluster(fr, f0, ex, q, res: _Res):
    """Greedy frequency clustering; yields (centre, member mask)."""
    order = np.argsort(f0)
    fr, f0, ex, q = fr[order], f0[order], ex[order], q[order]
    alive = np.ones(f0.size, dtype=bool)
    w = np.full(f0.size, res.frame_s)
    out = []
    while np.any(alive):
        cw = np.concatenate([[0.0], np.cumsum(np.where(alive, w, 0.0))])
        tol = res.tol(f0)
        a = np.searchsorted(f0, f0 - tol)
        b = np.searchsorted(f0, f0 + tol, side="right")
        score = np.where(alive, cw[b] - cw[a], -1.0)
        i = int(np.argmax(score))
        if score[i] < MIN_CLUSTER_S:
            break
        members = alive & (np.arange(f0.size) >= a[i]) & (np.arange(f0.size) < b[i])
        centre = float(np.median(f0[members]))
        t_c = res.tol(centre)
        members = alive & (np.abs(f0 - centre) <= t_c)
        if not np.any(members):
            alive[i] = False
            continue
        out.append((float(np.median(f0[members])), fr[members], ex[members], q[members]))
        alive &= ~members
    return out


def _ladder(f0: float, q: float, tol_hz: float, t_on: np.ndarray, res: list) -> tuple[float, int]:
    """Largest area-normalised ladder share (borrowed.ladder_fraction, from
    TRANSMUTATION's harmonic_noise_ratio) over fundamentals f0 / m, m = 1 ..
    min(32, Q - 1), on the baseline-whitened spectrum 10^(E/10) of the
    candidate's channel averaged over its frames, every resolution over its own
    range, the candidate's own band (+- f0 / 2Q or the tolerance) left out.
    Ladder bands are +-60 / m cents (at least +-0.6 bin) and reach 2 m + 2
    harmonics (audit 04 A1: with a fixed +-60 cents the bands tile the axis
    above the 14th harmonic, so any ring high above a note found a ladder). For
    m > 1 the ladder counts only if the whitened power within the band of f0 / m
    exceeds 3 dB (the subharmonic-phantom rule of pitch_candidates).
    Whitening matters: the original's area normalisation assumes a flat floor.
    Stiff-string ladders (audit 04 F9): the partials may sit at
    f1 k sqrt(1 + B k^2) for B in LADDER_B (0 to 0.0128, piano-like to
    bell-like), with f1 = f0 / (m sqrt(1 + B m^2)); with exact ladders only, the
    12th to 14th partials of a B = 4e-4 note (30 to 50 cents sharp) read as
    resonances. The share is the same area-normalised measure as
    borrowed.ladder_fraction, computed with prefix sums (equal on an exact
    ladder, tested). Neighbour rule: the ladder's partials m - 1 and m + 1
    (partial 2 when m = 1): at least one must also be > 3 dB above the baseline
    (the top partial of a series has no m + 1), otherwise a
    stretched ladder threads a ring between a note's partials (a 1650 Hz ring
    over a 220 Hz note read 0.41 without it). Returns (share, m, B)."""
    fr, pw, wd = [], [], []
    for r in res:
        sel = (r.f >= r.f_lo) & (r.f < r.f_hi)
        if not np.any(sel) or t_on.size == 0:
            continue
        fi = np.unique(np.clip(np.searchsorted(r.t, t_on), 0, r.t.size - 1))
        p = np.mean(10.0 ** (r.ex[np.ix_(fi, np.nonzero(sel)[0])].astype(np.float64) / 10.0), axis=0)
        fr.append(r.f[sel])
        pw.append(p)
        wd.append(np.full(p.size, r.df))
    if not fr:
        return 0.0, 0, 0.0
    fr, pw, wd = np.concatenate(fr), np.concatenate(pw), np.concatenate(wd)
    o = np.argsort(fr, kind="stable")
    fr, pw, wd = fr[o], pw[o], wd[o]
    q = q if np.isfinite(q) and q > 0 else Q_MIN
    half = max(f0 / (2.0 * q), tol_hz)
    excl = (f0 - half, f0 + half)
    keep = ~((fr >= excl[0]) & (fr <= excl[1]))
    fr, pw, wd = fr[keep], pw[keep], wd[keep]
    tot_w = float(np.sum(wd))
    tot_p = float(np.sum(pw * wd))
    cp = np.concatenate([[0.0], np.cumsum(pw * wd)])
    cw = np.concatenate([[0.0], np.cumsum(wd)])
    best, best_m, best_b = 0.0, 0, 0.0
    floor = 10.0 ** (PARTIAL_DB / 10.0)
    if tot_p <= 0 or fr.size == 0:
        return 0.0, 0, 0.0
    for m in range(1, LADDER_MAX_M + 1):
        if m > 1 and m >= q:
            break
        cents = LADDER_CENTS / m
        c = 2.0 ** (cents / 1200.0)
        k = np.arange(1, 2 * m + 3, dtype=np.float64)
        for B in LADDER_B:
            fund = f0 / (m * np.sqrt(1.0 + B * m * m))  # stiff string: f_k = k f1 sqrt(1 + B k^2)
            if m > 1:
                near = np.abs(fr - fund) <= np.maximum(fund * (c - 1.0), 0.6 * wd)
                if not np.any(near) or float(np.max(pw[near])) < floor:
                    continue
            fh = fund * k * np.sqrt(1.0 + B * k * k)
            # neighbour rule: the ladder's partials next to the candidate (m - 1
            # and m + 1; 2 when m = 1) must carry energy, or a stretched ladder
            # can thread a ring between a note's partials
            nb = [m - 1, m + 1] if m > 1 else [2]
            hits = 0
            for kk in nb:
                fk = fund * kk * np.sqrt(1.0 + B * kk * kk)
                if fk >= fr[-1]:
                    continue
                band = np.abs(fr - fk) <= np.maximum(fk * (c - 1.0), 0.6 * wd)
                hits += bool(np.any(band) and float(np.max(pw[band])) >= floor)
            if hits == 0:  # at least one neighbour (a series' top partial has no m + 1)
                continue
            fh = fh[fh < fr[-1]]
            if fh.size == 0:
                continue
            j = np.clip(np.searchsorted(fr, fh), 0, fr.size - 1)
            tol = np.maximum(fh * (c - 1.0), 0.6 * wd[j])
            a = np.searchsorted(fr, fh - tol, side="left")
            b = np.searchsorted(fr, fh + tol, side="right")
            mark = np.zeros(fr.size + 1, dtype=np.int64)  # union of the bands
            np.add.at(mark, a, 1)
            np.add.at(mark, b, -1)
            on = np.cumsum(mark[:-1]) > 0
            raw = float(np.sum((pw * wd)[on]) / tot_p)
            area = float(np.sum(wd[on]) / tot_w)
            share = raw if area >= 1.0 - 1e-9 else float(np.clip((raw - area) / (1.0 - area), 0.0, 1.0))
            if share > best:
                best, best_m, best_b = share, m, B
    return best, best_m, best_b


def _runs(mask: np.ndarray, frame_s: float, t: np.ndarray):
    """Close gaps < GAP_CLOSE_S, drop runs < MIN_RUN_S; return [(start_s, end_s)]."""
    m = mask.astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts = list(np.nonzero(d == 1)[0])
    ends = list(np.nonzero(d == -1)[0])  # exclusive
    runs = []
    for s, e in zip(starts, ends):
        if runs and (s - runs[-1][1]) * frame_s < GAP_CLOSE_S:
            runs[-1][1] = e
        else:
            runs.append([s, e])
    out = []
    for s, e in runs:
        if (e - s) * frame_s >= MIN_RUN_S:
            out.append((float(t[s]), float(t[e - 1] + frame_s)))
    return out


def _grid_presence(g, r: _Res, pres: np.ndarray, exf: np.ndarray):
    nfr = r.t.size
    gi = g.index(r.t)  # r.t are frame centre times
    cnt = np.bincount(gi, minlength=g.k).astype(float)
    hit = np.bincount(gi, weights=pres.astype(float), minlength=g.k)
    # nearest frame for grid frames that hold no frame centre (audit 04 A14)
    j = np.clip(np.searchsorted(r.t, g.times), 1, max(1, nfr - 1))
    near = np.where(np.abs(r.t[np.minimum(j, nfr - 1)] - g.times) < np.abs(r.t[j - 1] - g.times),
                    np.minimum(j, nfr - 1), j - 1) if nfr > 1 else np.zeros(g.k, int)
    frac = np.where(cnt > 0, hit / np.maximum(cnt, 1), pres[near].astype(float))
    gp = frac >= 0.5
    exg = np.zeros(g.k)
    np.maximum.at(exg, gi, exf)
    exg = np.where(gp, np.where(cnt > 0, exg, exf[near]), 0.0)
    return gp, exg


def _union_spans(spans):
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + GAP_CLOSE_S:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _channel_of(chs: set) -> str:
    if "M" in chs and "S" not in chs:
        return "M"
    if "S" in chs and "M" not in chs:
        return "S"
    return "both"


_PC = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def _triad(label):
    """Pitch classes of a triad label such as "Am" or "C#"; empty when unknown."""
    if not label:
        return set()
    minor = label.endswith("m")
    root = _PC.get(label[:-1] if minor else label)
    if root is None:
        return set()
    return {root, (root + (3 if minor else 4)) % 12, (root + 7) % 12}


def _chord_tone(f0: float, on_t: np.ndarray, key, chords) -> tuple[bool, float]:
    """(is a chord tone, share of the candidate's bars whose triad holds its pitch
    class). The pitch class must be within CHORD_CENTS of equal temperament; a
    member of the key's tonic triad counts on its own (spec 3.3: chord-invariance
    check via pitch.py)."""
    m = 69.0 + 12.0 * np.log2(f0 / 440.0)
    if abs(m - round(m)) * 100.0 > CHORD_CENTS:
        return False, 0.0
    pc = int(round(m)) % 12
    if key and pc in _triad(key.get("label")):
        return True, 1.0
    share = 0.0
    if chords and on_t.size:
        hits = tot = 0
        for b in chords:
            if np.any((on_t >= b["start"]) & (on_t < b["end"])):
                tot += 1
                hits += pc in _triad(b.get("chord"))
        share = hits / tot if tot else 0.0
    return share >= CHORD_SHARE, share


def analyse(x: np.ndarray, fs: float, ctx: Context | None = None, sections=None,
            key=None, chords=None) -> dict:
    ctx = ensure_context(x, fs, ctx)
    g = ctx.grid
    dur = ctx.x.shape[0] / ctx.fs
    if not sections:
        sections = [{"start": 0.0, "end": dur}]
    # channel inputs built one at a time with bands.to_ms's arithmetic, (L +- R) / sqrt 2
    # (the same values as ctx.ms's columns, one column alive instead of both)
    inputs = {"M": lambda: ((ctx.x[:, 0] + ctx.x[:, 1]) * INV_SQRT2)[:, None],
              "S": lambda: ((ctx.x[:, 0] - ctx.x[:, 1]) * INV_SQRT2)[:, None],
              "stereo": lambda: ctx.x}
    res_by = {}
    raw = []
    for ch in CHANNELS:
        xin = inputs[ch]()
        if not np.any(xin):
            continue
        rl = [_Res(xin, ctx.fs, n, lo, hi) for n, lo, hi in RESOLUTIONS if lo < 0.95 * ctx.fs / 2]
        res_by[ch] = rl
        for r in rl:
            fr, f0, ex, q = r.peaks()
            if r.f_hi <= SHORT_PASS_BELOW_HZ and r.smooth_s > SHORT_SMOOTH_S:
                # short detection pass (audit 04 F7): same STFT, 0.5 s smoothing,
                # its peaks added to the long pass's before clustering
                rs = _Res(xin, ctx.fs, RESOLUTIONS[0][0], r.f_lo, r.f_hi, smooth_s=SHORT_SMOOTH_S)
                f2, f02, ex2, q2 = rs.peaks(SHORT_EXCESS_DB)
                fr, f0 = np.concatenate([fr, f2]), np.concatenate([f0, f02])
                ex, q = np.concatenate([ex, ex2]), np.concatenate([q, q2])
            if f0.size == 0:
                continue
            for centre, cfr, cex, cq in _cluster(fr, f0, ex, q, r):
                pres = np.zeros(r.t.size, dtype=bool)
                pres[cfr] = True
                exf = np.zeros(r.t.size)
                np.maximum.at(exf, cfr, cex)
                qh, qu, _ = r.q_half(centre, np.nonzero(pres)[0])
                raw.append({"f0": centre, "q": qh if np.isfinite(qh) else float(np.median(cq)),
                            "q_unc": qu, "q_peak": float(np.median(cq)), "pres": pres, "exf": exf,
                            "res": r, "channel": ch, "weight": cfr.size * r.frame_s})

    cands = []
    for grp in _merge(raw):
        rep = grp[0]
        r = rep["res"]
        gp = np.zeros(g.k, dtype=bool)
        exg = np.zeros(g.k)
        spans = []
        for c in grp:
            p_, e_ = _grid_presence(g, c["res"], c["pres"], c["exf"])
            gp |= p_
            exg = np.maximum(exg, e_)
            spans += _runs(c["pres"], c["res"].frame_s, c["res"].t)
        pers = [float(np.mean(gp[g.frames_in(s_["start"], s_["end"])]))
                if np.any(g.frames_in(s_["start"], s_["end"])) else 0.0 for s_ in sections]
        if max(pers) <= PERSISTENCE:
            continue
        wsum = sum(c["weight"] for c in grp)
        f0 = float(sum(c["f0"] * c["weight"] for c in grp) / wsum)
        # Q recomputed in the strongest member's resolution over the merged frames
        on_t = g.times[gp]
        fi = np.unique(np.clip(np.searchsorted(r.t, on_t), 0, r.t.size - 1))
        qh, qu, exm = r.q_half(f0, fi)
        cands.append({"f0": f0, "q": qh if np.isfinite(qh) else rep["q"],
                      "q_unc": qu if np.isfinite(qh) else rep["q_unc"], "q_peak": rep["q_peak"],
                      "res": r, "channel_set": {c["channel"] for c in grp},
                      "rep_channel": rep["channel"], "excess_db": exg, "persistence": pers,
                      "spans": _union_spans(spans), "on_t": on_t, "members": len(grp)})

    out = []
    for c in sorted(cands, key=lambda c: c["f0"]):
        f0 = c["f0"]
        r = c["res"]
        rl = res_by[c["rep_channel"]]

        def res_for(fq):
            for rr in rl:
                if rr.in_range(fq):
                    return rr
            return None

        def partial(gf):
            rr = res_for(gf)
            if rr is None:
                return False
            fi = np.unique(np.clip(np.searchsorted(rr.t, c["on_t"]), 0, rr.t.size - 1))
            return rr.partial_present(gf, fi) >= PARTIAL_FRACTION

        present = {f"{h:.4g}": partial(h * f0) for h in SPEC_PARTIALS}
        # a pure line is not a peaking-filter shape: when the fit is that loose,
        # the per-frame -3 dB peak Q bounds m and the excluded band instead
        q_lad = c["q"] if (np.isfinite(c["q"]) and np.isfinite(c["q_unc"])
                           and c["q_unc"] < 0.5 * c["q"]) else c["q_peak"]
        lad, lad_m, lad_b = _ladder(f0, q_lad, float(r.tol(f0)), c["on_t"], rl)
        musical_peaks = any(present.values())
        series_hit = None
        if not musical_peaks:
            for m in SERIES_M:
                if m >= c["q_peak"]:
                    break
                fr0 = f0 / m
                hits = [j for j in range(1, m + 3) if j != m and partial(j * fr0)]
                if len(hits) >= 2 and (m - 1 in hits or m + 1 in hits):
                    musical_peaks = True
                    series_hit = {"m": m, "j": hits}
                    break
        musical_ladder = lad >= LADDER_MIN
        musical = {"peaks": musical_peaks, "ladder": musical_ladder,
                   "either": musical_peaks or musical_ladder}[COHERENCE]
        low = SUB_EXEMPT_HZ <= f0 < LOW_RULE_HZ
        ex_max = float(np.max(c["excess_db"]))
        bw_peak = f0 / c["q_peak"] if c["q_peak"] > 0 else np.inf
        line = bw_peak <= LINE_BINS * r.df  # as narrow as the window: a held tone
        fit_ok = np.isfinite(c["q"]) and np.isfinite(c["q_unc"]) and c["q_unc"] < 0.5 * c["q"]
        broad = fit_ok and c["q"] < Q_FIT_MIN
        chord_tone, chord_share = _chord_tone(f0, c["on_t"], key, chords)
        if broad:
            continue  # a shelf-like bump (e.g. at a mix's high-frequency rolloff), not narrowband
        if f0 < SUB_EXEMPT_HZ or (low and max(c["persistence"]) < LOW_PERSISTENCE):
            flag, depth = "sub_exempt", 0.0
        elif line or ex_max > EXCESS_TONAL_DB:
            flag, depth = "tonal", 0.0
        elif musical or chord_tone:
            flag, depth = "musical", MUSICAL_DEPTH_SCALE
        else:
            flag, depth = "resonance", 1.0
        if low and depth > 0:
            depth *= LOW_DEPTH_SCALE
        spans = c["spans"]
        out.append({
            "f0": f0, "q": c["q"], "q_uncertainty": c["q_unc"], "q_peak": c["q_peak"],
            "start": spans[0][0] if spans else float("nan"),
            "end": spans[-1][1] if spans else float("nan"),
            "spans": spans, "excess_db": c["excess_db"],
            "excess_max_db": float(np.max(c["excess_db"])),
            "persistence": c["persistence"], "flag": flag, "depth_scale": depth,
            "channel": _channel_of(c["channel_set"]), "channels_seen": sorted(c["channel_set"]),
            "partials": present, "series": series_hit, "resolution": r.n,
            "ladder_fraction": lad, "ladder_m": lad_m, "ladder_b": lad_b, "members": c["members"],
            "line": bool(line), "bw_peak_hz": float(bw_peak), "chord_tone": bool(chord_tone),
            "chord_share": float(chord_share),
        })
    return {"t": g.times, "resonances": out, "sections": sections,
            "params": {"excess_db": EXCESS_DB, "q_min": Q_MIN, "persistence": PERSISTENCE,
                       "sub_exempt_hz": SUB_EXEMPT_HZ, "low_rule_hz": LOW_RULE_HZ,
                       "channels": [ch for ch in CHANNELS if ch in res_by],
                       "resolutions": [(n, lo, hi) for n, lo, hi in RESOLUTIONS]}}
