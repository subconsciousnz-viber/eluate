"""Tempo, beat / bar grid and section segmentation (spec 3.3 row "structure").

Contract:
- Tempo first, as a metrical hypothesis (`metrical_hypothesis`, real-material
  pass): the pulse that the clap and hat bands' autocorrelation carry, with the
  kick band's pulse and its relation to it (x1, x2, /2, x4/3, ...), a "half-time
  or triplet feel" flag on 4/3, and every band's ranked alternatives; returned
  in `metrical`. A subdivision lattice (all strong peaks a whole number of
  sixteenths apart) overrides the strongest pulse when it explains clearly more
  of the peaks (the maintainer's tempos: Track C 88 and Track D 120, whose
  strongest pulses are 4/3 of that). An octave decision follows (S2, sprint
  2026-10-21): the half or double tempo replaces the choice when its
  sixteenth grid explains MH_OCTAVE_GAIN x the peak weight, recorded with its
  evidence in `metrical["octave"]` (Track F: 72.5 read as 145). A user `bpm`
  overrides all of it and is never second-guessed. Nothing assumes four-on-the-floor (the architect's interim
  rule): on the real premasters the kick band runs at 4/3 of the hat pulse on
  one track and at 2/3 on another.
- Beats: librosa.beat.beat_track on the onset envelope from transients.py
  (AR = 20480 Hz, hop 256), trim off, with its tempo fixed to the hypothesis
  (librosa's own estimate read Track B at 60 and Track D at 161.5 on 90 s
  excerpts); with a user `bpm` the period is exactly 60 / bpm and only the
  phase is fitted (`tempo_octave` "user"). If the tracked beats fit a constant
  tempo (least squares t = t0 + i T, the beat indices i counted interval by
  interval as round(diff / median interval), at least 1, so a slightly wrong
  interval cannot drift them; residual RMS < 10 % of T), the grid is that constant
  grid extended over the whole file (`grid_kind` "constant"); otherwise the
  tracked beats are kept and extrapolated at the fitted T to both ends
  ("tracked"). `tempo_bpm` = 60 / T in both cases. When kick onsets are given (from
  transients.py), the grid is shifted by the median offset from each beat to
  its nearest kick onset (beats with an onset within a quarter interval, if at
  least half have one); librosa's beats trail the attack by about 30 ms, which
  puts the next kick's attack inside the previous beat interval.
  A tracked grid at half the hypothesis tempo is subdivided (midpoints), so
  the grid, `tempo_bpm` and the plan's hits (one per beat) follow the octave
  decision (S3); `tempo_octave` is then "doubled from lattice" unless the kick check
  below doubles. Only doubling: a tempo read at twice the true one is not corrected by
  either path (audit 13 #6; no VA track is one).
  Octave check (audit 04 F2 / F11): when the median kick interval is half the
  beat period (ratio 2 +- 0.2, at least 8 kicks) the grid is doubled
  (`tempo_octave` "doubled from kick rate"): librosa reads 174 BPM as 87, and
  a half-tempo grid halves bass-root accuracy while every confidence still
  agrees with itself. The doubling also needs a regular kick train and a
  doubled tempo <= 180 BPM; with the hypothesis in place it rarely fires.
  `kick_consistency`: the kick rate (60 / median kick interval) against the
  tempo pulse, its half, the kick pulse or its half (`ratio`, `consistent`,
  `matches`). A beat that the grid or the shift puts within a quarter
  interval before 0 s is kept (clipped to 0) only if the first 20 ms hold
  signal above -40 dB of the track peak; a 400 ms pre-roll otherwise grew a
  beat at 0.000 s (audit 04 A8). `grid_informative` is False for the
  beatless default grid (120 BPM from t = 0), which the plan must not trust;
  `meter` is "4/4 (assumed)".
- Beat-synchronous features: per beat, the mean over the grid frames centred
  in the middle half of it (25 to 75 %) of MFCC 1-12 (librosa.feature.mfcc, 13 coefficients, c0 dropped, hop
  2048 at AR = 100 ms), chroma (librosa.feature.chroma_stft, 200 ms window,
  same hop; pitch.chroma's CQT smears chord changes over several beats), and
  two levels computed per
  beat interval directly from the samples: broadband mean power
  (L^2 + R^2)/2 and sub + low band M power, dB, floored at -100 (grid-frame
  averages would alias against the beat), over the middle half of each beat
  only (25 % to 75 %): with whole-beat intervals, 2.9 ms of the next kick's
  attack inside the previous interval was enough to move the downbeat vote by
  one beat.
  Each dimension is z-scored over beats with its standard deviation floored
  at a physical scale (1 for MFCC, 0.05 for chroma, 1 dB for the levels), and
  each of the four groups is divided by sqrt(its dimension) so the groups
  weigh equally. The novelty (not the downbeat vote) also gets a fifth group,
  `groove`: per beat, each metrical band's onset-function autocorrelation over
  the two bars around it at 1/2, 1 and 2 beats (floor 0.05). Track D's
  half-time to full-time switch back at 4:46 (maintainer) has no level or timbre
  change the other groups see: without the groove group its novelty peak (at
  286.3 s) had prominence 8.2 under a threshold of 12.4; with it the boundary
  lands at 287.8 s.
- Downbeats (4/4 assumed): the phase p in 0..3 maximising the summed feature
  change |F[b] - F[b - 1]| over beats b = p mod 4 (sections, chords and
  arrangement changes land on downbeats); ties go to the lowest phase. Bars are
  4 beats from there; a pickup before the first downbeat belongs to no bar.
- Novelty: Foote (2000) checkerboard kernel, half-width 16 beats (4 bars),
  Gaussian taper (sigma = half-width / 2), over the self-similarity matrix
  exp(-d^2 / (2 s^2)) of the beat features (d Euclidean, s = median d),
  zero-padded at the edges. Peaks: scipy find_peaks, at least 8 bars apart,
  prominence >= 0.15 x (max - median); peaks within 2 bars of either end are
  dropped. In the downbeat vote only outlier changes count (above median +
  3 MAD of all beat changes; all changes if none is an outlier), and the last
  beat, cut short by the end of the file, does not vote.
- Low-band dropout detector (pre-drop fills): runs of 1 to 8 beats whose sub +
  low level is >= 10 dB below the median of the 16 beats before the run, ended
  by a beat back within 3 dB of that median. The beat of return is a boundary
  candidate; runs are reported in `dropouts` as {start, end}.
- Snapping: from the unrounded bar position of each candidate to the nearest
  point of the 16-, 8- or 4-bar grid (anchored at the first downbeat, coarsest
  first) if that point is within 1 bar, else to the nearest bar; a boundary
  from the dropout detector is never moved before the dropout's return bar.
  Candidates closer than 4 bars are merged (the one with higher novelty kept).
- Segments {start, end, label, bar_start, bar_end}: from 0 to the first
  boundary, between boundaries, last to the end of the file. Labels are
  heuristic, from per-segment power-mean short-term loudness L, mean sub + low
  level B and onset rate: "intro" / "outro" for the first / last segment (when
  there are at least two; audit 04 A9: on a flat-level track the first and
  last read "drop"); else "drop" if L >= L_max - 2 LU and B >= B_max - 3 dB;
  else "build" for a
  segment right before a drop whose loudness or onset rate rises over it or
  that ends in a dropout; else "breakdown" if B < B_max - 6 dB; else "body".
  Labels are for reporting and preserve-mode deltas only, never the loudness
  target (spec 3.3).
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist

from . import pitch as _pitch
from ._common import AR, Context, db_power, ensure_context, librosa_on_grid, resample, window_mean

BEATS_PER_BAR = 4
CONST_RESIDUAL = 0.10
PHASE_MAD = 3.0
OCTAVE_MIN_KICKS = 8
MH_RATE = 4000
MH_HOP = 0.002
MH_TOL = 0.015
MH_RANGE = (70.0, 180.0)
MH_PEAK_RANGE = (50.0, 250.0)
MH_BANDS = (("kick", (35.0, 160.0)), ("clap", (300.0, 1500.0)), ("hats", (1500.0, None)))
MH_RELATIONS = (("x1", 1.0), ("x2", 2.0), ("/2", 0.5), ("x4/3", 4 / 3), ("x3/4", 0.75),
                ("x3/2", 1.5), ("x2/3", 2 / 3), ("x4", 4.0), ("/4", 0.25))
MH_REL_TOL = 0.03
MH_OCTAVE = 0.8
MH_GRID_PER_BEAT = 4  # grid units (sixteenths) per beat for the lattice; an assumption, stated in the output
MH_LATTICE_K = 8  # peaks may lie 1 to 8 grid units apart
MH_LATTICE_TOL = 0.015
MH_LATTICE_PEAKS = 8  # per autocorrelation
MH_LATTICE_KICK_WEIGHT = 0.5
MH_LATTICE_GAIN = 1.5  # the lattice tempo must explain this much more peak weight than the pulse
MH_OCTAVE_GAIN = 1.1  # the other octave's sixteenth grid must explain this much more (S2: Track F 1.18)
OCTAVE_REGULAR = 0.6
OCTAVE_MAX_BPM = 180.0
START_ACTIVE_S = 0.020
START_ACTIVE_DB = -40.0
FEATURE_FLOORS = [1.0, 0.05, 1.0, 1.0]  # MFCC (dB-like), chroma, level dB, low-band dB
BEAT_CORE = (0.25, 0.75)  # share of each beat the level features average over
NOVELTY_HALF_BEATS = 16
PEAK_MIN_BARS = 8  # 4 gave a 2 to 3 bar median section on real premasters
PEAK_PROMINENCE = 0.15
EDGE_BARS = 2
GROOVE_NOVELTY = True  # the groove group joins the novelty features (not the downbeat vote)
GROOVE_BARS = 2
GROOVE_LAGS = (0.5, 1.0, 2.0)
GROOVE_FLOOR = 0.05
MERGE_BARS = 4  # 2 left a fill return and the nearby novelty peak as two boundaries
SNAP_GRIDS = (16, 8, 4)
SNAP_TOL_BARS = 1
DROP_DB = 10.0
DROP_RETURN_DB = 3.0
DROP_MAX_BEATS = 8
DROP_REF_BEATS = 16
DROP_STEADY_IQR_DB = 6.0
DROP_AFTER_BEATS = 16
DROP_MIN_GAP_BEATS = 32


def beat_grid(onset_t, onset_env, dur: float, start_active: bool = True, bpm: float | None = None,
              fixed: bool = False):
    """`bpm` seeds librosa's tracker; with `fixed` (the user's tempo) the grid
    period is 60 / bpm exactly and only its phase is fitted (median offset)."""
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env.astype(np.float32), sr=AR,
                                           hop_length=256, units="time", trim=False, bpm=bpm)
    beats = np.asarray(beats, dtype=np.float64)
    if beats.size < 4:
        t = 60.0 / (bpm if fixed else 120.0)
        return np.arange(0.0, dur, t), 60.0 / t, "default"
    ibi = 60.0 / bpm if fixed else float(np.median(np.diff(beats)))
    # count beats interval by interval so a slightly wrong ibi cannot drift the indices
    idx = np.concatenate([[0.0], np.cumsum(np.maximum(1.0, np.round(np.diff(beats) / ibi)))])
    if fixed:
        period = ibi
        t0 = float(np.median(beats - period * idx))
    else:
        a = np.vstack([np.ones_like(idx), idx]).T
        (t0, period), *_ = np.linalg.lstsq(a, beats, rcond=None)
    resid = beats - (t0 + period * idx)
    if fixed or np.sqrt(np.mean(resid ** 2)) < CONST_RESIDUAL * period:
        i0 = -int(np.floor(t0 / period))
        i1 = int(np.ceil((dur - t0) / period))
        grid = t0 + period * np.arange(i0 - 1, i1)
        # a beat just before 0 is the first beat, but only if the track has signal there
        lo = -period / 4.0 if start_active else -1e-9
        grid = grid[(grid >= lo) & (grid < dur)]
        return np.maximum(grid, 0.0), 60.0 / period, "constant"
    # the fitted mean interval, not the median: librosa's beats sit on 12.5 ms
    # frames, and the median of 50 / 51-frame intervals read 95 BPM as 94.12
    pre = beats[0] - period * np.arange(int(beats[0] / period), 0, -1)
    post = beats[-1] + period * np.arange(1, int((dur - beats[-1]) / period) + 1)
    grid = np.concatenate([pre, beats, post])
    return grid[(grid >= 0) & (grid < dur)], 60.0 / period, "tracked"


def _band_odf(y: np.ndarray, band) -> np.ndarray:
    """Positive 10 ms rise of the band's 20 ms log energy, 2 ms hop, at MH_RATE."""
    lo, hi = band
    if hi is None:
        sos = signal.butter(4, lo, "highpass", fs=MH_RATE, output="sos")
    else:
        sos = signal.butter(4, (lo, hi), "bandpass", fs=MH_RATE, output="sos")
    b = signal.sosfiltfilt(sos, y)
    w, h = int(0.02 * MH_RATE), int(MH_HOP * MH_RATE)
    c = np.concatenate([[0.0], np.cumsum(b * b)])
    idx = np.arange(0, max(1, b.size - w), h)
    e = (c[np.minimum(idx + w, b.size)] - c[idx]) / w
    le = 10 * np.log10(e + 1e-10 * max(float(e.max()), 1e-30))
    L = int(round(0.01 / MH_HOP))
    d = np.zeros_like(le)
    d[L:] = np.maximum(le[L:] - le[:-L], 0.0)
    return d


def _acf(d: np.ndarray) -> np.ndarray:
    f = d - d.mean()
    if not np.any(f):
        return np.zeros(f.size)
    F = np.fft.rfft(f, 2 * f.size)
    ac = np.fft.irfft(F * np.conj(F))[:f.size]
    return ac / ac[0]


def _acf_at(ac: np.ndarray, bpm: float) -> float:
    lag = 60.0 / bpm / MH_HOP
    if lag >= ac.size - 2:
        return 0.0
    w = max(1, int(round(MH_TOL * lag)))
    k = int(round(lag))
    return float(np.max(ac[max(1, k - w):k + w + 1]))


def _acf_peaks(ac: np.ndarray, n: int = 4):
    lags = np.arange(ac.size) * MH_HOP
    sel = np.nonzero((lags >= 60 / MH_PEAK_RANGE[1]) & (lags <= 60 / MH_PEAK_RANGE[0]))[0]
    if sel.size < 3:
        return []
    a = ac[sel]
    pk = sel[1:-1][(a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:])]
    pk = pk[np.argsort(-ac[pk])][:n]
    out = []
    for k in pk:
        y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
        den = y0 - 2 * y1 + y2
        p_ = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        out.append((float(60.0 / ((k + p_) * MH_HOP)), float(y1)))
    return out


def _relation(a: float, b: float):
    for name, r in MH_RELATIONS:
        if abs(a / b / r - 1.0) <= MH_REL_TOL:
            return name
    return None


def _pulse(ac: np.ndarray):
    """Strongest interpolated autocorrelation peak with a lag inside MH_RANGE BPM,
    moved to the faster octave when the peak at half the lag is in range and at
    least MH_OCTAVE of its height (a periodic train correlates equally at every
    multiple of its period, so the slower octave is never evidence on its own).
    Returns (bpm, height) or (nan, 0)."""
    pk = [(b_, h) for b_, h in _acf_peaks(ac, n=12) if MH_RANGE[0] <= b_ <= MH_RANGE[1]]
    if not pk:
        return float("nan"), 0.0
    bpm, h = max(pk, key=lambda t: t[1])
    for b2, h2 in pk:
        if abs(b2 / (2 * bpm) - 1.0) <= MH_REL_TOL and h2 >= MH_OCTAVE * h:
            bpm, h = b2, h2
    return bpm, h


def _lattice_weights(ac_upper: np.ndarray, ac_kick: np.ndarray):
    """(bpm, weight) of the MH_LATTICE_PEAKS strongest positive peaks of each
    autocorrelation, heights normalised to the strongest; kick peaks weigh
    MH_LATTICE_KICK_WEIGHT."""
    out = []
    for ac, wt in ((ac_upper, 1.0), (ac_kick, MH_LATTICE_KICK_WEIGHT)):
        pk = [(b_, h) for b_, h in _acf_peaks(ac, MH_LATTICE_PEAKS) if h > 0]
        if pk:
            m = max(h for _, h in pk)
            out += [(b_, wt * h / m) for b_, h in pk]
    return out


def _lattice_hits(unit: float, pw):
    """Peaks an integer 1 to MH_LATTICE_K grid units of `unit` BPM apart (within
    MH_LATTICE_TOL), as (implied unit, weight)."""
    out = []
    for b_, w in pw:
        k = round(unit / b_)
        if 1 <= k <= MH_LATTICE_K and abs(b_ * k / unit - 1.0) <= MH_LATTICE_TOL:
            out.append((b_ * k, w))
    return out


def _lattice_score(unit: float, pw) -> float:
    return float(sum(w for _, w in _lattice_hits(unit, pw)))


def _lattice(pw):
    """Grid unit (BPM of one subdivision, MH_GRID_PER_BEAT x a tempo in MH_RANGE)
    that explains the most peak weight, refined to the weighted mean of the
    peaks it explains. Returns (unit, score) or (nan, 0)."""
    lo, hi = MH_GRID_PER_BEAT * MH_RANGE[0], MH_GRID_PER_BEAT * MH_RANGE[1]
    best = (float("nan"), 0.0)
    for b_, _ in pw:
        for k in range(1, MH_LATTICE_K + 1):
            u = b_ * k
            if lo <= u <= hi:
                hits = _lattice_hits(u, pw)
                sc = float(sum(w for _, w in hits))
                if sc > best[1] + 1e-9:
                    best = (float(np.average([h[0] for h in hits], weights=[h[1] for h in hits])), sc)
    return best


def _octave(tempo: float, pw, up: np.ndarray, kick: np.ndarray):
    """The octave decision (sprint 2026-10-21 S2): of tempo and its double (inside
    MH_RANGE), the one whose sixteenth grid (_lattice_score at
    MH_GRID_PER_BEAT x the candidate) explains the most peak weight, taken only when
    that is at least MH_OCTAVE_GAIN x what tempo's own grid explains. Peaks an odd
    number of sixteenths apart at the faster octave are the evidence; the
    autocorrelation heights at each candidate's beat lag are recorded, not voted
    (a clap on 2 and 4 correlates at two beats, so the slower octave always reads
    taller). Doubling only: a sixteenth grid at half the tempo is an eighth grid at the
    tempo, so the slower octave explains a straight pattern with fewer slots and would
    halve a correct 174 BPM read (tests/test_feature_pitch.py); the failure this fixes
    (Track F 72.5 for 145) is always the half. Returns (tempo, record)."""
    if not np.isfinite(tempo):
        return tempo, {"decision": "none", "from_bpm": tempo, "to_bpm": tempo,
                       "gain": MH_OCTAVE_GAIN, "candidates": []}
    cands = []
    for m in (1.0, 2.0):
        c = tempo * m
        if m == 1.0 or MH_RANGE[0] <= c <= MH_RANGE[1]:
            cands.append({"bpm": c, "lattice_score": _lattice_score(MH_GRID_PER_BEAT * c, pw),
                          "upper_acf": _acf_at(up, c), "kick_acf": _acf_at(kick, c)})
    own = next(c for c in cands if c["bpm"] == tempo)
    best = max((c for c in cands if c is not own), key=lambda c: c["lattice_score"], default=None)
    out = tempo
    decision = "kept"
    if best is not None and best["lattice_score"] > 0 \
            and best["lattice_score"] >= MH_OCTAVE_GAIN * own["lattice_score"]:
        out = best["bpm"]
        decision = "doubled"
    return out, {"decision": decision, "from_bpm": tempo, "to_bpm": out, "gain": MH_OCTAVE_GAIN,
                 "candidates": cands}


def band_odfs(x: np.ndarray, fs: float) -> dict:
    """`_band_odf` of every MH_BANDS band of the mono mix at MH_RATE."""
    y = resample(0.5 * (x[:, 0] + x[:, 1]), fs, MH_RATE)
    return {name: _band_odf(y, band) for name, band in MH_BANDS}


def groove(odfs: dict, beats: np.ndarray) -> np.ndarray:
    """Per beat, the autocorrelation of each band's onset function over the
    GROOVE_BARS bars centred on the beat at lags of 1/2, 1 and 2 beats (local
    beat period): a groove descriptor that does not depend on the beat's place
    in the bar. Half-time against full-time changes the 1-beat lag against the
    2-beat lag; level, timbre and chroma features need not change at all."""
    nb = beats.size
    out = np.zeros((nb, len(MH_BANDS) * len(GROOVE_LAGS)))
    if nb < 3:
        return out
    per = np.gradient(beats)
    half = GROOVE_BARS * BEATS_PER_BAR // 2
    for i in range(nb):
        a, b = max(0, i - half), min(nb - 1, i + half)
        i0, i1 = int(beats[a] / MH_HOP), int(beats[b] / MH_HOP)
        p = per[i] / MH_HOP
        for j, (name, _) in enumerate(MH_BANDS):
            seg = odfs[name][i0:i1]
            seg = seg - seg.mean() if seg.size else seg
            e = float(np.dot(seg, seg))
            for m, lag in enumerate(GROOVE_LAGS):
                L = int(round(lag * p))
                if e <= 0 or L >= seg.size:
                    continue
                out[i, j * len(GROOVE_LAGS) + m] = float(np.dot(seg[:-L], seg[L:])) / e
    return out


def metrical_hypothesis(x: np.ndarray, fs: float, bpm: float | None = None, odfs: dict | None = None) -> dict:
    """Tempo as a metrical hypothesis (spec 3.3 structure row; the architect's
    interim rule for the real-material pass, 2026-10-17).

    Onset functions (10 ms rise of the 20 ms log energy) in three bands: kick 35
    to 160 Hz, clap 300 to 1500 Hz, hats above 1.5 kHz, autocorrelated over the
    whole track. Upper pulse: `_pulse` of the mean of the clap and hat
    autocorrelations; kick pulse: `_pulse` of the kick band's. The tempo is the
    upper pulse (the kick pulse when the upper bands have none in range);
    `relation` names how the kick pulse relates to it ("x1", "x2", "/2", "x4/3",
    "x3/4", "x3/2", "x2/3", "x4", "/4" within 3 %, else None); `feel` is "half-time
    or triplet feel" for a 4/3 or 3/4 relation. `alternatives`: each band's ranked
    autocorrelation peaks (50 to 250 BPM) with their relation to the tempo.

    Subdivision lattice (after the maintainer's tempos): on two of the five real
    premasters the strongest upper pulse is 4/3 of the beat (Track C 117.3
    for 88, Track D 160.2 for 120), while every strong peak sits a whole
    number of sixteenths apart (Track D: 2, 3, 4, 5, 6 and 8 sixteenths at
    120 BPM; the 4-sixteenth peak, the beat itself, is the weakest of them).
    So the strongest peaks of the upper and kick autocorrelations vote for the
    grid unit that spaces them (`_lattice`), and the lattice tempo (unit /
    MH_GRID_PER_BEAT, at the octave inside MH_RANGE nearest the upper pulse; the
    nearest octave alone read a 120 BPM fixture as 60) replaces the
    upper pulse when it is not an octave of it and explains at least
    MH_LATTICE_GAIN times the peak weight the pulse's own grid does (`rule`
    "lattice"; else "pulse"). Four grid units per beat is an assumption
    (`grid_per_beat`); a bare kick train is explained equally well by a grid at
    3/4 of its rate, which the gain margin refuses. The upper pulse at 4/3 or
    3/4 of the tempo also sets `feel`.

    Octave (S2, sprint 2026-10-21): the lattice above never moves the tempo by
    an octave; `_octave` then compares the chosen tempo's sixteenth grid with
    its half's and double's (inside MH_RANGE) and moves to the other octave
    when that grid explains at least MH_OCTAVE_GAIN x the peak weight (`rule`
    gains " + octave"). Track F read 72.5 on its clap (a clap on 2 and 4 at
    145) with the lattice unit at 4 x 145 (score 2.56 against 2.17, 1.18 x);
    on the six other VA premasters the other octave is outside MH_RANGE or
    scores at most 0.97 x (Track G's half) and 0.79 x (Track C's double).
    `octave`: {decision "kept" / "doubled" / "none",
    from_bpm, to_bpm, gain, candidates [{bpm, lattice_score, upper_acf,
    kick_acf}]}, the evidence the report quotes.

    `bpm` (the user's tempo, spec `--bpm`) overrides the choice and seeds the
    grid (`rule` "user"): `tempo_bpm` is the user's, `detected_bpm` what
    detection chose (octave decision included), `deviation_pct` how far it was
    and `detected_relation` its ratio to the user's tempo. A MIDI tempo map
    (WP10) will take the user's place."""
    odfs = band_odfs(x, fs) if odfs is None else odfs
    acs = {name: _acf(odfs[name]) for name in odfs}
    up = 0.5 * (acs["clap"] + acs["hats"])
    t_up, h_up = _pulse(up)
    t_k, h_k = _pulse(acs["kick"])
    if not np.isfinite(t_up):
        t_up, h_up = t_k, h_k
    pw = _lattice_weights(up, acs["kick"])
    unit, l_score = _lattice(pw)
    p_score = _lattice_score(MH_GRID_PER_BEAT * t_up, pw) if np.isfinite(t_up) else 0.0
    tempo, rule = t_up, "pulse"
    octs = [unit / MH_GRID_PER_BEAT * 2.0 ** e for e in (-1, 0, 1)] if np.isfinite(unit) else []
    octs = [t_ for t_ in octs if MH_RANGE[0] <= t_ <= MH_RANGE[1]]
    if octs and np.isfinite(t_up):
        t_l = min(octs, key=lambda t_: abs(np.log2(t_ / t_up)))
        if l_score >= MH_LATTICE_GAIN * p_score and _relation(t_l, t_up) not in ("x1", "x2", "/2"):
            tempo, rule = t_l, "lattice"
    tempo, octave = _octave(tempo, pw, up, acs["kick"])
    if octave["decision"] == "doubled":
        rule += " + octave"
    detected = tempo
    if bpm is not None:
        tempo, rule = float(bpm), "user"
    fin = np.isfinite(tempo)
    rel = _relation(t_k, tempo) if fin and np.isfinite(t_k) else None
    up_rel = _relation(t_up, tempo) if fin and np.isfinite(t_up) else None
    feel = "half-time or triplet feel" if {rel, up_rel} & {"x4/3", "x3/4"} else None
    alts = []
    for n, _ in MH_BANDS:
        for b_, h in _acf_peaks(acs[n]):
            alts.append({"band": n, "bpm": b_, "acf": h, "relation": _relation(b_, tempo) if fin else None})
    alts.sort(key=lambda a_: -a_["acf"])
    out = {"tempo_bpm": tempo, "upper_pulse_bpm": t_up, "kick_pulse_bpm": t_k, "relation": rel,
           "upper_relation": up_rel, "feel": feel, "rule": rule,
           "lattice_unit_bpm": unit, "lattice_score": l_score, "pulse_score": p_score,
           "grid_per_beat": MH_GRID_PER_BEAT, "octave": octave, "alternatives": alts}
    if bpm is not None:
        ok = np.isfinite(detected)
        out.update({"user_bpm": float(bpm), "detected_bpm": detected,
                    "deviation_pct": 100.0 * (detected / float(bpm) - 1.0) if ok else float("nan"),
                    "detected_relation": _relation(detected, float(bpm)) if ok else None})
    return out


def kick_consistency(kicks, tempo: float, kick_pulse: float | None = None) -> dict:
    """Kick onsets against the hypothesis pulses (the architect's interim rule 2):
    `ratio` = kick rate / tempo, the kick rate being 60 / median kick interval
    (1 = a kick per beat, 0.5 = every other beat); `consistent` when the kick rate
    matches the tempo pulse, its half, the kick-band pulse or its half within 10 %
    (`matches` says which). Only these four pulses count: matching any of the
    listed alternatives would make almost any rate "consistent"."""
    k = np.sort(np.asarray(kicks if kicks is not None else [], dtype=np.float64))
    if k.size < 4 or not np.isfinite(tempo):
        return {"ratio": float("nan"), "consistent": False, "matches": None}
    rate = 60.0 / float(np.median(np.diff(k)))
    pulses = [("tempo", tempo)]
    if kick_pulse is not None and np.isfinite(kick_pulse):
        pulses.append(("kick pulse", kick_pulse))
    for name, p_ in pulses:
        for mult, lab in ((1.0, ""), (0.5, " half")):
            if abs(rate / (p_ * mult) - 1.0) <= 0.10:
                return {"ratio": rate / tempo, "consistent": True, "matches": name + lab}
    return {"ratio": rate / tempo, "consistent": False, "matches": None}


def octave_check(beats: np.ndarray, tempo: float, kicks, dur: float, start_active: bool = True):
    """Double a half-tempo grid when the kicks say so (audit 04 F2 / F11, A8).

    Four-on-the-floor has about one kick per beat. If the median kick interval
    (intervals shorter than 1.5 beats) is half the beat period within 10 % and
    at least 8 kicks support it, the grid is doubled (midpoints inserted) -- but
    only when the kicks form a regular train (>= 60 % of the intervals within
    10 % of the median) and the doubled tempo is at most 180 BPM: on a real
    premaster (Track B, 120 BPM) 8th-note bass attacks counted as kicks made
    the interval half a beat and the check read 240.
    Returns (beats, tempo, note) with note "kept", "doubled from kick rate" or
    "no kicks"."""
    if kicks is None or len(kicks) < OCTAVE_MIN_KICKS or beats.size < 4:
        return beats, tempo, "no kicks"
    period = float(np.median(np.diff(beats)))
    d = np.diff(np.sort(np.asarray(kicks, dtype=np.float64)))
    d = d[d < 1.5 * period]
    if d.size < OCTAVE_MIN_KICKS - 1:
        return beats, tempo, "kept"
    med = float(np.median(d))
    ratio = period / med
    regular = float(np.mean(np.abs(d / med - 1.0) <= 0.1))  # share of intervals within 10 % of the median
    if abs(ratio - 2.0) <= 0.2 and regular >= OCTAVE_REGULAR and 2.0 * tempo <= OCTAVE_MAX_BPM:
        return _subdivide(beats, dur), tempo * 2.0, "doubled from kick rate"
    return beats, tempo, "kept"


def _subdivide(beats: np.ndarray, dur: float) -> np.ndarray:
    """The grid at twice the rate: midpoints inserted, one more after the last beat
    (kept only before dur)."""
    period = float(np.median(np.diff(beats)))
    mids = beats[:-1] + 0.5 * np.diff(beats)
    out = np.sort(np.concatenate([beats, mids, [beats[-1] + period / 2.0]]))
    return out[out < dur]


def align_to_onsets(beats: np.ndarray, onsets, dur: float, start_active: bool = True):
    """Shift the beat grid by the median offset to the nearest kick onset (beats
    with an onset within a quarter interval only), when at least half the beats
    have one. librosa's beats trail the kick attack by tens of ms, enough to put
    the next kick's attack inside a beat interval."""
    if onsets is None or len(onsets) < 2 or beats.size < 4:
        return beats, 0.0
    onsets = np.sort(np.asarray(onsets, dtype=np.float64))
    ibi = float(np.median(np.diff(beats)))
    i = np.clip(np.searchsorted(onsets, beats), 1, onsets.size - 1)
    near = np.where(np.abs(onsets[i] - beats) < np.abs(onsets[i - 1] - beats), onsets[i], onsets[i - 1])
    off = near - beats
    ok = np.abs(off) < ibi / 4.0
    if np.count_nonzero(ok) < beats.size / 2:
        return beats, 0.0
    shift = float(np.median(off[ok]))
    out = beats + shift
    if out[0] - ibi >= -1e-9 and shift > 0:
        out = np.concatenate([[out[0] - ibi], out])
    lo = -ibi / 4.0 if start_active else -1e-9  # a first beat shifted just before 0 survives only over signal
    out = out[(out >= lo) & (out < dur)]
    return np.maximum(out, 0.0), shift


def _beat_sync(feat: np.ndarray, grid, beats: np.ndarray, dur: float) -> np.ndarray:
    """(d, K) grid feature -> (beats, d) means over the grid frames centred in the
    middle half of each beat (BEAT_CORE), nearest frame when none: the CQT's
    long low-frequency windows smear a chord change into the previous beat, and
    with whole-beat means a chord-only boundary moved the downbeat by 3 beats."""
    ends = np.append(beats[1:], dur)
    out = np.empty((beats.size, feat.shape[0]))
    for i, (a, b) in enumerate(zip(beats, ends)):
        sel = grid.frames_in(a + BEAT_CORE[0] * (b - a), a + BEAT_CORE[1] * (b - a))
        if np.any(sel):
            out[i] = feat[:, sel].mean(axis=1)
        else:
            out[i] = feat[:, int(grid.index(0.5 * (a + b)))]
    return out


def _beat_power_db(p: np.ndarray, beats: np.ndarray, fs: float) -> np.ndarray:
    """Mean of the per-sample power p over each beat interval, dB, floored at -100."""
    c = np.concatenate([[0.0], np.cumsum(p)])
    # the middle half of each beat only: away from both kick attacks, so a grid
    # a few ms off the kicks cannot move a level step into the neighbouring beat
    ends = np.append(beats[1:], p.shape[0] / fs)
    span = ends - beats
    lo = np.clip(np.round((beats + BEAT_CORE[0] * span) * fs).astype(np.int64), 0, p.shape[0])
    hi = np.clip(np.round((beats + BEAT_CORE[1] * span) * fs).astype(np.int64), 0, p.shape[0])
    a, b = lo, hi
    with np.errstate(invalid="ignore"):
        m = np.where(b > a, (c[b] - c[a]) / np.maximum(b - a, 1), 0.0)
    return np.maximum(db_power(m), -100.0)


def _standardise(groups, floors=None) -> np.ndarray:
    """z-score each dimension, dividing by max(std, the group's floor): a feature
    that barely moves (sub-dB level noise on a constant-level track) must not be
    blown up to unit variance and outvote the features that do change (it
    moved the downbeat on chord-only boundaries by up to 3 beats)."""
    cols = []
    floors = floors or [1e-9] * len(groups)
    for gmat, fl in zip(groups, floors):
        sd = np.maximum(gmat.std(axis=0), fl)
        z = (gmat - gmat.mean(axis=0)) / sd
        cols.append(z / np.sqrt(gmat.shape[1]))
    return np.hstack(cols)


def foote_novelty(feat: np.ndarray, half: int = NOVELTY_HALF_BEATS) -> np.ndarray:
    """Foote novelty: sum(K * S) along the diagonal, K = +taper on the two
    within-segment quadrants and -taper on the cross quadrants."""
    n = feat.shape[0]
    d = cdist(feat, feat)
    s = np.median(d[np.triu_indices(n, 1)]) if n > 1 else 1.0
    s = s if s > 0 else 1.0
    ssm = np.exp(-(d ** 2) / (2 * s * s))
    u = np.arange(-half, half) + 0.5
    taper = np.exp(-0.5 * (u / (half / 2.0)) ** 2)
    k = np.outer(taper, taper) * np.outer(np.sign(u), np.sign(u))
    p = np.pad(ssm, half)
    out = np.empty(n)
    for i in range(n):
        out[i] = (k * p[i:i + 2 * half, i:i + 2 * half]).sum()
    return out


def dropouts(low_db_beats: np.ndarray, beats: np.ndarray, phase: int | None = None):
    """Runs of 1 to 8 beats with the sub + low level >= 10 dB under the median of
    the 16 beats before, ended by a beat back within 3 dB of it. Kept only as a
    pre-drop fill when the 16 reference beats were a steady low end (IQR <= 6 dB:
    a half-time or broken low end dips every bar and is not a fill) and, with a
    downbeat phase given, the return lands within one beat of a downbeat, the low
    end then stays up (median of the next 16 beats within 3 dB of the reference:
    the drop itself), and it is at least 32 beats after the previous fill. On the
    real premasters the unfiltered rule found 18 to 68 "fills" per track; with
    the steadiness and downbeat tests alone, still one every 2 bars on a track
    whose low end dips in a 2-bar pattern."""
    out = []
    n = low_db_beats.size
    b = DROP_REF_BEATS
    while b < n:
        refw = low_db_beats[b - DROP_REF_BEATS:b]
        ref = float(np.median(refw))
        steady = float(np.subtract(*np.percentile(refw, [75, 25]))) <= DROP_STEADY_IQR_DB
        if steady and low_db_beats[b] <= ref - DROP_DB:
            e = b
            while e < n and low_db_beats[e] <= ref - DROP_DB and e - b < DROP_MAX_BEATS + 1:
                e += 1
            on_bar = phase is None or min((e - phase) % BEATS_PER_BAR,
                                          BEATS_PER_BAR - (e - phase) % BEATS_PER_BAR) <= 1
            after = low_db_beats[e:e + DROP_AFTER_BEATS]
            sustained = after.size == DROP_AFTER_BEATS and float(np.median(after)) >= ref - DROP_RETURN_DB
            spaced = not out or e - out[-1]["beat_end"] >= DROP_MIN_GAP_BEATS
            if (1 <= e - b <= DROP_MAX_BEATS and e < n and low_db_beats[e] >= ref - DROP_RETURN_DB
                    and on_bar and sustained and spaced):
                out.append({"start": float(beats[b]), "end": float(beats[e]), "beat_start": b, "beat_end": e})
            b = e + 1
        else:
            b += 1
    return out


def _snap_bar(bar: float, n_bars: int, not_before: float | None = None) -> int:
    """Snap an unrounded bar position: the nearest point of the 16-, 8- then 4-bar
    grid within SNAP_TOL_BARS of the unrounded position, else the nearest bar
    (one rounding only: rounding first and snapping after moved boundaries by
    up to 1.25 bars, audit 04 A9). With `not_before` (a dropout's return bar)
    the result is never earlier than it, so a boundary after a 1-bar pre-drop
    fill stays on the drop, not on the fill's start."""
    out = int(round(bar))
    for g in SNAP_GRIDS:
        q = int(round(bar / g)) * g
        if abs(q - bar) <= SNAP_TOL_BARS and 0 < q < n_bars:
            out = q
            break
    if not_before is not None and out < int(np.ceil(not_before - 1e-6)):
        out = int(np.ceil(not_before - 1e-6))
    return out


def _label(segs, st_series, low_series, rate_series, grid, drop_ends):
    feats = []
    for s in segs:
        sel = grid.frames_in(s["start"], s["end"])
        st = st_series[sel]
        st = st[np.isfinite(st)]
        loud = 10 * np.log10(np.mean(10 ** (st / 10))) if st.size else -np.inf
        low = float(np.mean(low_series[sel])) if np.any(sel) else -100.0
        r = rate_series[sel] if rate_series is not None else np.zeros(np.count_nonzero(sel))
        half = max(1, r.size // 2)
        rising_rate = r.size >= 2 and r[half:].mean() > r[:half].mean() * 1.2
        sth = st_series[sel]
        sth = sth[np.isfinite(sth)]
        hh = max(1, sth.size // 2)
        pm = lambda v: 10 * np.log10(np.mean(10 ** (v / 10)))  # power mean, not dB mean
        rising_loud = sth.size >= 2 and pm(sth[hh:]) > pm(sth[:hh]) + 1.0
        feats.append((loud, low, rising_rate or rising_loud))
    if not feats:
        return
    lmax = max(f[0] for f in feats)
    bmax = max(f[1] for f in feats)
    is_drop = [f[0] >= lmax - 2.0 and f[1] >= bmax - 3.0 for f in feats]
    for i, s in enumerate(segs):
        # intro / outro first (audit 04 A9: on a flat-level track every section,
        # first and last included, was a "drop"); a single segment is a "drop"
        if len(segs) > 1 and i == 0:
            lab = "intro"
        elif len(segs) > 1 and i == len(segs) - 1:
            lab = "outro"
        elif is_drop[i]:
            lab = "drop"
        elif i + 1 < len(segs) and is_drop[i + 1] and (feats[i][2] or any(abs(e - s["end"]) < 1e-6 for e in drop_ends)):
            lab = "build"
        elif feats[i][1] < bmax - 6.0:
            lab = "breakdown"
        else:
            lab = "body"
        s["label"] = lab
        s["loudness_lufs"] = float(feats[i][0])
        s["low_db"] = float(feats[i][1])


def analyse(x: np.ndarray, fs: float, ctx: Context | None = None, onset=None,
            short_term=None, onset_rate=None, kick_onsets=None, bpm: float | None = None) -> dict:
    ctx = ensure_context(x, fs, ctx)
    g = ctx.grid
    dur = ctx.x.shape[0] / ctx.fs
    if onset is None:
        from .transients import onset_envelope
        onset = onset_envelope(ctx)
    if short_term is None:
        from .loudness import series
        short_term = series(ctx.x, ctx.fs, g)["short_term"]
    m0 = max(1, int(round(START_ACTIVE_S * ctx.fs)))
    peak = float(np.max(np.abs(ctx.x))) if ctx.x.size else 0.0
    start_active = peak > 0 and float(np.max(np.abs(ctx.x[:m0]))) > peak * 10 ** (START_ACTIVE_DB / 20)
    odfs = band_odfs(ctx.x, ctx.fs)
    mh = metrical_hypothesis(ctx.x, ctx.fs, bpm, odfs=odfs)
    beats, tempo, kind = beat_grid(onset[0], onset[1], dur, start_active,
                                   bpm=mh["tempo_bpm"] if np.isfinite(mh["tempo_bpm"]) else None,
                                   fixed=bpm is not None)
    if bpm is None and beats.size >= 2 and _relation(tempo, mh["tempo_bpm"]) == "/2":
        beats, tempo = _subdivide(beats, dur), tempo * 2.0  # the tracker fell back an octave
    if bpm is None:
        beats, tempo, octave = octave_check(beats, tempo, kick_onsets, dur, start_active)
        moved = mh["octave"]["decision"]
        if moved == "doubled" and octave != "doubled from kick rate":
            octave = f"{moved} from lattice"
    else:
        octave = "user"  # the user's tempo is not second-guessed
    kc = kick_consistency(kick_onsets, tempo, mh["kick_pulse_bpm"])
    beats, beat_shift = align_to_onsets(beats, kick_onsets, dur, start_active)

    y = ctx.mono_ar[1024:].astype(np.float32)
    mf = librosa_on_grid(librosa.feature.mfcc(y=y, sr=AR, n_mfcc=13, n_fft=4096, hop_length=2048)
                         .astype(np.float64), g.k)[1:]
    # a short STFT chroma here, not pitch.chroma's CQT: CQT filters at the low
    # end are seconds long and smear a chord change over several beats
    ch = librosa_on_grid(librosa.feature.chroma_stft(y=y, sr=AR, n_fft=4096, hop_length=2048)
                         .astype(np.float64), g.k)
    lo = ctx.low_band_m
    low_db = np.maximum(db_power(window_mean(lo * lo, g.centres, int(round(0.4 * ctx.fs)))), -100.0)
    # level features per beat straight from the samples: averaging 100 ms grid
    # frames would alias against the beat (4 or 5 frames per beat)
    loud_b = _beat_power_db(np.mean(ctx.x ** 2, axis=1), beats, ctx.fs)
    low_b = _beat_power_db(lo * lo, beats, ctx.fs)
    fb = [_beat_sync(m, g, beats, dur) for m in (mf, ch)] + [loud_b[:, None], low_b[:, None]]
    feat = _standardise(fb, FEATURE_FLOORS)
    nb = beats.size
    dchg = np.zeros(nb)
    if nb > 1:
        dchg[1:] = np.linalg.norm(np.diff(feat, axis=0), axis=1)
    # only outlier changes vote: z-scored features that merely carry noise would
    # otherwise outvote a few real section changes
    med = np.median(dchg[1:]) if nb > 1 else 0.0
    mad = np.median(np.abs(dchg[1:] - med)) if nb > 1 else 0.0
    excess = np.maximum(dchg - (med + PHASE_MAD * mad), 0.0)
    if not np.any(excess > 0):
        excess = dchg
    excess[-1] = 0.0  # the last beat is cut by the end of the file
    score = [excess[p::BEATS_PER_BAR].sum() for p in range(BEATS_PER_BAR)]
    phase = int(np.argmax(score))
    downbeats = beats[phase::BEATS_PER_BAR]
    n_bars = downbeats.size
    bar_of_beat = (np.arange(nb) - phase) / BEATS_PER_BAR

    feat_nov = np.hstack([feat, _standardise([groove(odfs, beats)], [GROOVE_FLOOR])]) if GROOVE_NOVELTY else feat
    nov = foote_novelty(feat_nov) if nb > 2 * BEATS_PER_BAR else np.zeros(nb)
    cands = []
    if nb > 2 * BEATS_PER_BAR and np.ptp(nov) > 0:
        prom = PEAK_PROMINENCE * (nov.max() - np.median(nov))
        pk, _ = find_peaks(nov, distance=PEAK_MIN_BARS * BEATS_PER_BAR, prominence=max(prom, 1e-12))
        for p in pk:
            cands.append((float(bar_of_beat[p]), float(nov[p]), None))
    lb = fb[3][:, 0]
    drops = dropouts(lb, beats, phase)
    for dr in drops:
        ret = float(bar_of_beat[dr["beat_end"]])
        cands.append((ret, float(nov.max()) if nov.size else 1.0, ret))

    bars = []
    for bar, strength, floor_bar in cands:
        bi = _snap_bar(bar, n_bars, floor_bar)
        if bi < EDGE_BARS or bi > n_bars - EDGE_BARS or bi >= n_bars:
            continue
        bars.append((bi, strength))
    bars.sort()
    merged = []
    for bi, st in bars:
        if merged and bi - merged[-1][0] < MERGE_BARS:
            if st > merged[-1][1]:
                merged[-1] = (bi, st)
        else:
            merged.append((bi, st))
    bounds = [b for b, _ in merged]
    edges_t = [0.0] + [float(downbeats[b]) for b in bounds] + [dur]
    segs = []
    for i, (a, b) in enumerate(zip(edges_t[:-1], edges_t[1:])):
        segs.append({"start": a, "end": b,
                     "bar_start": 0 if i == 0 else bounds[i - 1],
                     "bar_end": bounds[i] if i < len(bounds) else n_bars})
    _label(segs, short_term, low_db, onset_rate, g, [d["end"] for d in drops])
    return {
        "tempo_bpm": float(tempo), "grid_kind": kind, "grid_informative": kind != "default",
        "tempo_octave": octave, "meter": "4/4 (assumed)", "metrical": mh, "kick_consistency": kc,
        "beat_times": beats, "beat_shift_s": beat_shift,
        "downbeat_phase": phase, "downbeat_times": downbeats, "bar_times": downbeats,
        "beats_per_bar": BEATS_PER_BAR, "novelty": nov, "boundaries": [float(downbeats[b]) for b in bounds],
        "boundary_bars": bounds, "segments": segs, "dropouts": drops, "low_db": low_db,
    }

