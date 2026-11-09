"""Stems, phase B corrections (docs/noise-and-stems.md 2, "Phase B decisions"): turn the
attribution of a sum pass (stems.attribute) into gentle fixes on the stems it names, and
render them. The maintainer's stems are printed after his mix bus (they null to the mix), so
every fix is small and capped; the loop that re-sums and re-masters (at most two
correction passes) is not here.

Contract:
- `plan(rep, stems, names, fs, prior=None, bpm=None) -> list[dict]`: rep is
  stems.attribute() output of the latest pass (in memory, or its stems-report.json, where
  an infinite value reads as None); stems are the ORIGINAL uncorrected stems (k, n, 2)
  float64. Returns the full cumulative fix list: prior (kept in its order) merged with this
  pass's fixes, appended. A fix is keyed by (stem, kind, band); a key already in prior is
  deepened by this pass's step and clamped to its cap on the TOTAL, so a second pass may
  take a fix to its cap, never past it. A key gets at most one step per pass however many
  limits name it. An empty list, or one equal to prior, means nothing more to do.
  Each fix: {"stem": name, "kind": "eq"|"comp"|"duck", "params": {...}, "reason": the
  limit text of the latest pass that moved it}, JSON-serialisable (floats, str, None).
- `apply_sum(stems, fs, names, fixes) -> np.ndarray (n, 2)`: the corrected stems' sum,
  built one stem at a time (extra memory one stem plus the sum); use it on full sets.
- `apply(stems, fs, names, fixes) -> np.ndarray`: a new array of the same shape; stems is
  not modified. Every stem's fixes run on that stem as given (the loop passes the
  originals and the cumulative list, so nothing compounds), in the order EQ, duck,
  compression. Deterministic: no randomness, the duck's kick onsets are found again from
  the kick stem passed in. A fix naming a stem not in names is ignored.
- `describe(fix) -> str`: one report line, e.g. "pads: high shelf +1.5 dB at 1.34 kHz
  (input EQ high shelf +2.5 dB, at its bound)".

Mapping (limit -> fix):
- EQ section at its bound ("input EQ <section> <gain> dB, at its bound"): a shelf, same
  shape and corner as the sum's section, in the same direction, on each named stem whose
  projection share there is at least EQ_SHARE_MIN (a stem cancelling against the rest is
  never boosted to fix the sum). The section is detail["section"], else parsed from the
  limit text. Sizing: when the entry's detail carries "wanted_db" (the total gain the
  sum's section wanted: the planner's range_deficit_db for the shelves, the median of its
  fit's node deficits in the range for the low shelf, bell and band shelf; stems.attribute),
  excess = wanted_db - gain_db (the
  applied bound) and the per-stem gain g solves 1 + (g - 1) S = 10^(excess / 20), S the
  chosen stems' summed share (a stem's projection moves the sum's level in the range by its share);
  a known excess of 0 or less (the section sits at its bound for the slope target, having
  wanted less than it got in the range) proposes nothing; EQ_STEP_DB only when wanted_db
  is unknown (a report without the planner's record). Shapes: high and air shelves the inputeq first-order high shelf (the
  _side ones on the stem's S only), low shelf the RBJ low shelf at Q EQ_LOW_SHELF_Q, bell
  and band shelf an RBJ peak at the range's geometric centre (Q = centre / width; the
  band shelf's flat cut approximated). All zero phase, forward-backward at half the dB
  gain, as the input EQ stage runs them (exact at DC, Nyquist and the corner).
- Notch found and not applied ("notch at <f0> Hz (found)"): a static notch on the stem
  with the largest excess at f0 (the first row), in the notch stage's parallel form
  x - d BP(x) with its zero-phase bandpass and design_q, the planner's Q for an unmeasured
  notch on its detail["channel"] (M or S of the stem; else L and R), with its
  detail["q"] (else the planner's unmeasured default: planner.NOTCH_Q_DEFAULT, at most
  NOTCH_LOW_Q_MAX below NOTCH_LOW_HZ), NOTCH_STEP_DB per pass, the total capped at
  NOTCH_MAX_DB, or at detail["max_cut_db"] when detail["flag"] is "musical" (the cap is
  kept in the fix's params so later passes hold it; a cap of 0.0 is a cap, never "no
  cap"). No fix for a notch the planner itself held back on (stems.notch_rejected: a
  "suspect" reason, a chord-tone guard, no cut, or a 0 dB cap); a rendered notch already
  took it out of the sum: no fix.
- Limiter on its outlier cap, clipper short of its target, the limiter held by its
  average budget or the loudness shortfall (PEAK_LIMITS), a multiband band at its
  exception or its band limiter at its cap, a frequency-dependent limiter cut: peak
  compression on the first stem the entry names (its peaks at the moments of largest GR)
  and on the second when it is within COMP_SECOND_DB of the first on the entry's ranking
  (the leave-one-out drop of the sum's peak, "peak_drop_db", else its own peak). When the
  entry is full band or band 0 (the lowest) and names the bass (or the kick and the bass
  as its two rows), and the kick and bass stems are identified, and at least
  DUCK_MOMENT_SHARE of the entry's moments fall on a kick onset (or it lists none), the
  bass is ducked under the kick instead; once that duck sits at its cap, further hits
  compress the bass.
- Crest low or section contrast lost: no fix. The entry ranks stems by lowest crest, so it
  names the stems already most compressed; compressing them lowers the crest further, and
  neither EQ nor ducking restores section contrast. Noise: no fix here.
- A stem the entry does not name is never touched; a silent stem (peak below SILENT_DB,
  e.g. crest n/a) never gets a fix.

Compression (offline, a lookahead peak compressor): detector = per 1 ms control frame the
peak of |x| over both channels, peak-held over ENV_HOLD_MS (_held: a 1 ms frame of a sub
falls between its peaks, and the bare frame peaks would ripple the gain at twice the
fundamental); required gain from glue.gain_computer (hard knee, ratio
COMP_RATIO), clipped to the fix's max_gr_db; held over +-1 frame (so the linear
interpolation between control points keeps the full reduction at every frame's peak),
then envelope.release_hull forward with COMP_RELEASE_MS and backward (time reversed) with
COMP_LOOKAHEAD_MS / 3, so the gain is down before each peak and never reduces less than
required. Threshold: set in plan so the stem's peaks at the entry's moments (mean of the
peak within +-stems.ATTR_WINDOW_S of each; else its 99.9th percentile frame peak) take the
fix's depth, measured on the stem with its other fixes applied; the depth starts at
COMP_STEP_DB and is capped at COMP_MAX_DB.

Ducking (whole-track knowledge, offline): kick and bass found by content (`roles`): stems
whose energy below ROLE_LOW_HZ is at least ROLE_LOW_SHARE of their own and within
ROLE_LOW_RANGE_DB of the loudest such band; the kick has the highest low-band crest, the
bass the most low-band energy among the rest that may take the role (`bass_ok`: a
BASS_HINTS token in its name, else its low band ROLE_BASS_LOW_OVER_HI_DB over the rest of
its own energy, a tonal low band (`_tonal`: ROLE_TONAL_MIN of its power in a few lines)
and under ROLE_BROADBAND_SHARE of the set's energy above ROLE_LOW_HZ, so a "music" stem
carrying the sub with its pads and leads is never ducked full band); a name token
(KICK_HINTS, BASS_HINTS) adds ROLE_HINT_CREST_DB to a kick's crest or ROLE_HINT_ENERGY_DB
to a bass's energy (a tiebreak); without both hints the kick's crest must beat the bass's
by ROLE_CREST_GAP_DB. No bass that passes: no duck (the entry's stems are compressed).
A bass already dipping under the kick (a sidechain in the mix) is not ducked again: per
onset `existing_dip_db` (its low-band RMS before and after the kick against its RMS
DUCK_DURING_MS under it, when at least DUCK_EXISTING_MIN_DB) comes off the depth.
Kick onsets: the kick's low-band 1 ms envelope (peak held over ENV_HOLD_MS) rising ONSET_RISE_DB above its minimum
over the previous ONSET_RISE_MS, within ONSET_RANGE_DB of its loud level (99.5th
percentile), at least ONSET_GAP_MS apart. Per onset the bass gain dips by depth x w on a
raised cosine over DUCK_LOOKAHEAD_MS before the onset, holds while the kick's low band
stays within KICK_BODY_DB of its peak (on the held envelope, so up to ENV_HOLD_MS past
the body; clamped to DUCK_HOLD_MS), and recovers on a raised
cosine over the release: DUCK_RELEASE_BEATS of a beat when bpm is known (clamped to
DUCK_RELEASE_MS), else DUCK_RELEASE_DEFAULT_MS. w in [0, 1] is the overlap: the bass's
low-band peak over that hold against its loud level (95th percentile), 1 at it, 0
OVERLAP_RANGE_DB below, so a kick with no bass under it ducks nothing. Overlapping dips
take the deeper. Full band on the bass (a 3 dB dip at most; a band split would keep its
harmonics but adds a crossover to a stem that nulls to the mix). Depth DUCK_STEP_DB per
pass, capped at DUCK_MAX_DB.
"""

from __future__ import annotations

import math
import re

import numpy as np
from scipy import signal
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import minimum_filter1d

from eluate import envelope
from eluate.plan import CONTROL_RATE_HZ
from eluate.stems import ATTR_WINDOW_S, notch_rejected
from eluate.stages import glue, inputeq, notch

EQ_STEP_DB = 1.5            # per pass when the sum's wanted gain is unknown: two passes reach the cap
EQ_MAX_DB = 3.0             # total per stem per band: twice the stems profile's own +-1.5 dB
EQ_SHARE_MIN = 0.2          # a stem carrying less of the range moves the sum too little to fix it
EQ_LOW_SHELF_Q = 0.707      # the input EQ's low shelf Q (plan default)
NOTCH_STEP_DB = 1.5         # per pass
NOTCH_MAX_DB = 3.0          # total cut: half the notch stage's 6 dB cap (stems are post mix bus)
COMP_STEP_DB = 1.0          # first pass peak GR on the stem
COMP_MAX_DB = 2.0           # total peak GR on one stem (maintainer: gentle)
COMP_RATIO = 4.0            # peak control: steep enough that the cap, not the ratio, sets the depth
COMP_LOOKAHEAD_MS = 5.0     # the gain is down before a kick's or bass note's first cycle peaks
COMP_RELEASE_MS = 60.0      # fast: back before the next 16th at 172 bpm (87 ms)
COMP_SECOND_DB = 3.0        # the second-ranked stem is compressed too when this close to the first
DUCK_STEP_DB = 2.0          # first pass dip
DUCK_MAX_DB = 3.0           # total dip on the bass (maintainer: about 3 dB)
DUCK_LOOKAHEAD_MS = 5.0     # the dip is complete at the kick's onset
DUCK_HOLD_MS = (20.0, 120.0)    # the kick's body, clamped: a click to a long 808-style tail
DUCK_RELEASE_BEATS = 0.25   # a 16th note: recovered well before the next beat's kick
DUCK_RELEASE_MS = (40.0, 200.0)
DUCK_RELEASE_DEFAULT_MS = 100.0  # 0.25 beat at 150 bpm, the middle of the electronic range
DUCK_MOMENT_SHARE = 0.5     # this share of the limiter's moments on a kick onset makes it a duck
DUCK_MOMENT_WINDOW_S = (0.03, 0.06)  # a moment counts from 30 ms before an onset (the
                                     # limiter's lookahead) to 60 ms after (the kick's body)
KICK_BODY_DB = 6.0          # the hold lasts while the kick's low band is within this of its peak
OVERLAP_RANGE_DB = 12.0     # bass this far below its loud level under a kick: no duck there
ROLE_LOW_HZ = 150.0         # kick fundamentals and the bass's sub sit below this
ROLE_LOW_SHARE = 0.4        # a kick or bass stem has at least this share of its energy below it
ROLE_LOW_RANGE_DB = 20.0    # ... and a low band within this of the loudest stem's low band
ROLE_CREST_GAP_DB = 3.0     # the kick's low-band crest above the bass's (content alone)
ROLE_HINT_CREST_DB = 3.0    # a kick-named stem's crest bonus (tiebreak)
ROLE_HINT_ENERGY_DB = 6.0   # a bass-named stem's low-band energy bonus (tiebreak)
ROLE_BASS_LOW_OVER_HI_DB = 15.0  # an unnamed bass: its low band this far over the rest of its energy
                                 # (measured: sub / bass stems 21 to 32 dB, a sub + pads + lead
                                 # "music" stem 12 dB, kicks 10 to 23 dB)
ROLE_TONAL_MIN = 0.4        # ... and tonal: this share of its low-band power in a few lines
ROLE_TONAL_RES_HZ = 5.0     # the line test's resolution (a 128 bpm kick train's lines merge)
ROLE_TONAL_BINS = 3         # the lines counted: a root, its octave, one note change
ROLE_BROADBAND_SHARE = 0.5  # ... and not the set's dominant broadband stem (its share of the
                            # set's energy above ROLE_LOW_HZ under this)
KICK_HINTS = ("kick", "kik", "kck", "bd", "bassdrum", "drum", "drums")
BASS_HINTS = ("bass", "sub", "808")
DUCK_PRE_MS = (55, 5)       # existing_dip_db: the bass before an onset, [i - 55 ms, i - 5 ms]
DUCK_DURING_MS = (10, 60)   # ... under the kick, [i + 10 ms, i + 60 ms]: two 41 Hz periods
                            # where a sidechain's dip is deepest, past a note's own attack and
                            # the lowpass's delay (a note starting with the kick reads 0.6 dB
                            # from i on, 0.1 dB from i + 10 ms)
DUCK_EXISTING_MIN_DB = 1.0  # an existing dip under this is a note's envelope, not a sidechain
DUCK_POST_MS = (100, 200)   # ... and after it, [i + 100 ms, i + 200 ms]
ENV_HOLD_MS = 25            # the low-band envelope's peak hold: one period at 40 Hz
ONSET_RISE_DB = 9.0         # a kick's attack rises at least this within ONSET_RISE_MS (real kicks
ONSET_RISE_MS = 10          # rise 30 dB or more from their own decayed tail or silence)
ONSET_RANGE_DB = 18.0       # ghost hits and tails further below the kick's loud level are not onsets
ONSET_GAP_MS = 60           # one onset per kick: a 32nd at 250 bpm, faster than any kick roll ducks
SILENT_DB = -100.0          # a stem peaking below this is silent: never fixed

_EQ_RE = re.compile(r"^input EQ (.+?) [+-]?\d+(?:\.\d+)? dB, at its bound$")
_MB_RE = re.compile(r"^(?:multiband|frequency-dependent limiter) band (\d+) ")
# full-band peak limits (stems.attribute's texts): peak compression, or the duck
PEAK_LIMITS = ("limiter on its outlier cap", "clipper short of its target",
               "limiter held by its average budget", "loudness ")
_RATE = CONTROL_RATE_HZ


def _db(v):
    return 20.0 * math.log10(v) if v > 0 else -math.inf


def _num(v):
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None


def _silent(x: np.ndarray) -> bool:
    return _db(float(np.max(np.abs(x)))) < SILENT_DB if x.size else True


# ---------------------------------------------------------------- control-rate helpers

def _frame_starts(n: int, fs: int) -> np.ndarray:
    k = np.arange(int(math.ceil(n * _RATE / fs)))
    return np.unique(np.minimum(np.ceil(k * fs / _RATE).astype(np.int64), n - 1))


def _frame_peaks(v: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Per 1 ms frame the peak of v (mono, >= 0); returns (peaks, frame start samples)."""
    st = _frame_starts(len(v), fs)
    return np.maximum.reduceat(v, st), st


def _to_audio(g_db: np.ndarray, st: np.ndarray, n: int) -> np.ndarray:
    """Control-rate dB at the frame starts, linearly interpolated to every sample, linear."""
    return 10.0 ** (np.interp(np.arange(n), st, g_db) / 20.0)


def _held(pk: np.ndarray) -> np.ndarray:
    """Causal peak hold over the last ENV_HOLD_MS frames: a 1 ms frame of a 40 Hz wave can
    fall between its peaks, so the bare frame peaks ripple by tens of dB."""
    h = ENV_HOLD_MS
    return sliding_window_view(np.concatenate([np.full(h - 1, pk[0]), pk]), h).max(axis=1)


def _low_mono(x: np.ndarray, fs: int) -> np.ndarray:
    """(L + R) / 2 through the 4th-order ROLE_LOW_HZ lowpass the roles and the duck read."""
    sos = signal.butter(4, ROLE_LOW_HZ, "lowpass", fs=fs, output="sos")
    return signal.sosfilt(sos, 0.5 * (x[:, 0] + x[:, 1]))


def _env_db(x: np.ndarray, fs: int, lowpass: bool) -> tuple[np.ndarray, np.ndarray]:
    """Mono (L + R) / 2 envelope in dB at the 1 ms frame starts, held (_held)."""
    m = _low_mono(x, fs) if lowpass else 0.5 * (x[:, 0] + x[:, 1])
    pk, st = _frame_peaks(np.abs(m), fs)
    pk = _held(pk)
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.maximum(pk, 1e-12)), st


# ---------------------------------------------------------------- roles and onsets

def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]


def _hint(name: str, hints) -> bool:
    return any(t in hints or any(t.startswith(h) for h in hints if len(h) > 2)
               for t in _tokens(name))


def _tonal(lo: np.ndarray, fs: int) -> float:
    """How much of the low band's power sits in its ROLE_TONAL_BINS largest ROLE_TONAL_RES_HZ
    bins (20 Hz to ROLE_LOW_HZ): a sustained bass root reads as a few lines (0.7 to 1.0), a
    kick's sweep and tail as a continuum (about 0.3)."""
    nps = int(fs / ROLE_TONAL_RES_HZ)
    if lo.size < nps:
        return 0.0
    f, p = signal.welch(lo, fs, nperseg=nps)
    sel = (f >= 20.0) & (f <= ROLE_LOW_HZ)
    p = p[sel]
    tot = float(p.sum())
    return float(np.sort(p)[-ROLE_TONAL_BINS:].sum() / tot) if tot > 0 else 0.0


def bass_ok(f: dict, name: str) -> bool:
    """Whether a low-band candidate may take the bass role (module docstring, Ducking): a
    BASS_HINTS token in its name, else its low band ROLE_BASS_LOW_OVER_HI_DB over the rest
    of its own energy, tonal (_tonal >= ROLE_TONAL_MIN) and not the set's dominant
    broadband stem (under ROLE_BROADBAND_SHARE of the set's energy above ROLE_LOW_HZ)."""
    if _hint(name, BASS_HINTS):
        return True
    return (f["low_db"] - f["hi_db"] >= ROLE_BASS_LOW_OVER_HI_DB and f["tonal"] >= ROLE_TONAL_MIN
            and f["hi_share"] < ROLE_BROADBAND_SHARE)


def roles(stems: np.ndarray, names, fs: int) -> tuple[int | None, int | None]:
    """(kick index, bass index) by content, name tokens as a tiebreak; None when unsure.
    The bass index is None when no stem passes `bass_ok` (a music stem that carries the
    sub with its pads and leads is not ducked)."""
    sos = signal.butter(4, ROLE_LOW_HZ, "lowpass", fs=fs, output="sos")
    feats = []
    for i, s in enumerate(stems):
        if _silent(s):
            continue
        m = 0.5 * (s[:, 0] + s[:, 1])
        e = float(np.sum(m ** 2))
        lo = signal.sosfilt(sos, m)
        el = float(np.sum(lo ** 2))
        if e <= 0 or el <= 0:
            continue
        pk = _held(_frame_peaks(np.abs(lo), fs)[0])
        pk_db = 20.0 * np.log10(np.maximum(pk, 1e-12))
        loud = float(np.percentile(pk_db, 99.5))
        act = pk_db > loud - 40.0
        rms_db = 10.0 * math.log10(float(np.mean(pk[act] ** 2))) if np.any(act) else -math.inf
        hi = max(e - el, 0.0)
        feats.append({"i": i, "share": el / e, "low_db": 10.0 * math.log10(el),
                      "hi_db": 10.0 * math.log10(hi) if hi > 0 else -math.inf, "hi": hi,
                      "crest": loud - rms_db, "tonal": _tonal(lo, fs)})
    if not feats:
        return None, None
    hi_tot = sum(f["hi"] for f in feats)
    for f in feats:
        f["hi_share"] = f["hi"] / hi_tot if hi_tot > 0 else 0.0
    top = max(f["low_db"] for f in feats)
    cand = [f for f in feats if f["share"] >= ROLE_LOW_SHARE and f["low_db"] >= top - ROLE_LOW_RANGE_DB]
    if len(cand) < 2:
        return None, None
    kick = max(cand, key=lambda f: (f["crest"] + (ROLE_HINT_CREST_DB if _hint(names[f["i"]], KICK_HINTS) else 0.0), -f["i"]))
    rest = [f for f in cand if f is not kick and bass_ok(f, names[f["i"]])]
    if not rest:
        return kick["i"], None
    bass = max(rest, key=lambda f: (f["low_db"] + (ROLE_HINT_ENERGY_DB if _hint(names[f["i"]], BASS_HINTS) else 0.0), -f["i"]))
    named = _hint(names[kick["i"]], KICK_HINTS) and _hint(names[bass["i"]], BASS_HINTS)
    if not named and kick["crest"] - bass["crest"] < ROLE_CREST_GAP_DB:
        return None, None
    return kick["i"], bass["i"]


def kick_onsets(kick: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """(onset frames, hold frames) on the 1 ms control grid, from the kick stem alone."""
    env, _ = _env_db(kick, fs, lowpass=True)
    loud = float(np.percentile(env, 99.5))
    r = ONSET_RISE_MS
    prev_min = sliding_window_view(np.concatenate([np.full(r, env[0]), env]), r)[:env.size].min(axis=1)
    cand = np.nonzero((env >= loud - ONSET_RANGE_DB) & (env - prev_min >= ONSET_RISE_DB))[0]
    on, last = [], -10 ** 9
    for i in cand.tolist():
        if i - last >= ONSET_GAP_MS:
            on.append(i)
            last = i
    on = np.asarray(on, dtype=np.int64)
    holds = np.empty(on.size, dtype=np.int64)
    hmax = int(DUCK_HOLD_MS[1])
    for j, i in enumerate(on.tolist()):
        seg = env[i:i + hmax + 1]
        p = int(np.argmax(seg))
        below = np.nonzero(seg[p:] < seg[p] - KICK_BODY_DB)[0]
        h = p + (int(below[0]) if below.size else seg.size - p)
        holds[j] = int(np.clip(h, DUCK_HOLD_MS[0], DUCK_HOLD_MS[1]))
    return on, holds


def _release_ms(bpm) -> float:
    if bpm and bpm > 0:
        return float(np.clip(DUCK_RELEASE_BEATS * 60000.0 / bpm, *DUCK_RELEASE_MS))
    return DUCK_RELEASE_DEFAULT_MS


def existing_dip_db(lo: np.ndarray, fs: int, onset_frame: int) -> float:
    """How far the bass's low band (lo, its lowpassed mono) already dips under a kick at
    this onset (1 ms frames): its level outside the dip, the louder of its RMS over the
    DUCK_PRE_MS window before the onset and the DUCK_POST_MS window after (a slow
    sidechain release still holds the bass down 100 ms on; a note starting at the kick has
    no level before it and reads its own level after), minus its RMS over DUCK_DURING_MS
    after the onset (a note ending at the kick, silent there: 0, and its overlap weight is
    0 anyway); under DUCK_EXISTING_MIN_DB it reads 0. On a sub sidechained 6 dB with a 100 ms
    release it reads about 3.8 dB, so a 3 dB duck adds none; 3 dB / 60 ms reads 1.5 (the
    duck adds 1.5); 1.5 dB / 30 ms and an unducked sub read 0."""
    spf = fs / _RATE
    n = lo.shape[0]

    def rms_db(a_ms, b_ms):
        a, b = int(round(a_ms * spf)), int(round(b_ms * spf))
        a, b = max(0, min(n, a)), max(0, min(n, b))
        if b - a < 2:
            return None
        v = float(np.mean(lo[a:b] ** 2))
        return 10.0 * math.log10(v) if v > 0 else -math.inf

    i = onset_frame
    pre = rms_db(i - DUCK_PRE_MS[0], i - DUCK_PRE_MS[1])
    post = rms_db(i + DUCK_POST_MS[0], i + DUCK_POST_MS[1])
    during = rms_db(i + DUCK_DURING_MS[0], i + DUCK_DURING_MS[1])
    ref = max((v for v in (pre, post) if v is not None), default=None)
    if ref is None or during is None or not math.isfinite(during):
        return 0.0
    d = ref - during
    return d if d >= DUCK_EXISTING_MIN_DB else 0.0


def duck_gain_db(kick: np.ndarray, bass: np.ndarray, fs: int, depth_db: float,
                 release_ms: float, lookahead_ms: float = DUCK_LOOKAHEAD_MS) -> tuple[np.ndarray, np.ndarray]:
    """(the bass's duck gain, dB <= 0, at its 1 ms frame starts; those starts, samples)."""
    on, holds = kick_onsets(kick, fs)
    lo = _low_mono(bass, fs)
    pk, st = _frame_peaks(np.abs(lo), fs)
    with np.errstate(divide="ignore"):
        benv = 20.0 * np.log10(np.maximum(_held(pk), 1e-12))
    nf = st.size
    g = np.zeros(nf)
    act = benv > float(np.max(benv)) - 60.0
    if on.size == 0 or not np.any(act):
        return g, st
    ref = float(np.percentile(benv[act], 95.0))
    la, rl = int(round(lookahead_ms)), max(1, int(round(release_ms)))
    for i, h in zip(on.tolist(), holds.tolist()):
        if i >= nf:
            continue
        b = float(np.max(benv[i:min(nf, i + h + 1)]))
        w = float(np.clip((b - (ref - OVERLAP_RANGE_DB)) / OVERLAP_RANGE_DB, 0.0, 1.0))
        if w <= 0.0:
            continue
        d = max(0.0, depth_db * w - existing_dip_db(lo, fs, i))
        if d <= 1e-9:
            continue
        a0 = max(0, i - la)
        t = np.arange(a0, i)
        shape = [(t, 0.5 * (1.0 - np.cos(np.pi * (t - (i - la)) / max(la, 1))))]
        shape.append((np.arange(i, min(nf, i + h)), None))
        t = np.arange(i + h, min(nf, i + h + rl))
        shape.append((t, 0.5 * (1.0 + np.cos(np.pi * (t - (i + h)) / rl))))
        for t, f in shape:
            if t.size:
                v = -d * (1.0 if f is None else f)
                g[t] = np.minimum(g[t], v)
    return g, st


# ---------------------------------------------------------------- compression

def _comp_gain_db(x: np.ndarray, fs: int, p: dict) -> tuple[np.ndarray, np.ndarray]:
    pk, st = _frame_peaks(np.max(np.abs(x), axis=1), fs)
    pk = _held(pk)      # a 1 ms frame of a sub falls between its peaks: no gain ripple at 2 f0
    with np.errstate(divide="ignore"):
        lv = 20.0 * np.log10(np.maximum(pk, 1e-12))
    gr = np.maximum(glue.gain_computer(lv, p["threshold_db"], p["ratio"], 0.0), -p["max_gr_db"])
    u = 10.0 ** (minimum_filter1d(gr, size=3, mode="nearest") / 20.0)
    h = envelope.release_hull(u, p["release_ms"] * _RATE / 1000.0)
    h = envelope.release_hull(h[::-1].copy(), p["lookahead_ms"] * _RATE / 3000.0)[::-1]
    return 20.0 * np.log10(np.maximum(h, envelope.G_MIN)), st


def _comp_threshold(x: np.ndarray, fs: int, depth: float, moments) -> float:
    w = int(round(ATTR_WINDOW_S * fs))
    a = np.max(np.abs(x), axis=1)
    vals = []
    for t in moments or []:
        i = int(round(float(t) * fs))
        seg = a[max(0, i - w):i + w]
        if seg.size and np.max(seg) > 0:
            vals.append(_db(float(np.max(seg))))
    if vals:
        ref = float(np.mean(vals))
    else:
        pk, _ = _frame_peaks(a, fs)
        ref = _db(float(np.percentile(pk, 99.9)))
    return ref - depth / (1.0 - 1.0 / COMP_RATIO)


# ---------------------------------------------------------------- EQ

def _eq_sos(p: dict, fs: int) -> np.ndarray | None:
    """Per-pass (half dB gain) section for a forward-backward run; None for a notch."""
    g = p["gain_db"] / 2.0
    if p["shape"] == "high_shelf":
        return inputeq._first_order_high_shelf(g, p["fc_hz"], fs)
    if p["shape"] == "low_shelf":
        return inputeq._rbj_low_shelf(g, p["fc_hz"], p["q"], fs)
    if p["shape"] == "bell":
        return inputeq._rbj_peak(g, p["fc_hz"], p["q"], fs)
    return None


def _zp(v: np.ndarray, sos: np.ndarray) -> np.ndarray:
    n, pl = v.shape[0], inputeq.padlen(sos)
    return signal.sosfiltfilt(sos, np.pad(v, ((pl, pl), (0, 0))), axis=0, padtype=None)[pl:pl + n]


def _apply_eq(x: np.ndarray, fs: int, fixes: list[dict]) -> np.ndarray:
    y = x
    for ch in ("LR", "S", "M"):
        rows = [_eq_sos(f["params"], fs) for f in fixes
                if f["params"].get("channel", "LR") == ch and f["params"]["shape"] != "notch"]
        rows = [r for r in rows if r is not None]
        if not rows:
            continue
        sos = np.vstack(rows)
        if ch == "LR":
            y = _zp(y, sos)
        else:
            m, s = 0.5 * (y[:, 0] + y[:, 1]), 0.5 * (y[:, 0] - y[:, 1])
            v = _zp((s if ch == "S" else m)[:, None], sos)[:, 0]
            m, s = (m, v) if ch == "S" else (v, s)
            y = np.stack([m + s, m - s], axis=1)
    for f in fixes:
        p = f["params"]
        if p["shape"] != "notch" or p["gain_db"] >= 0:
            continue
        sos = notch.bandpass_sos(p["fc_hz"], notch.design_q(p["fc_hz"], p["q"], fs, p["gain_db"]), fs)
        d = 1.0 - 10.0 ** (p["gain_db"] / 20.0)
        ch = p.get("channel", "LR")
        if ch == "LR":
            y = y - d * notch.bandpass_zp(y, sos, fs)
        else:
            m, s = 0.5 * (y[:, 0] + y[:, 1]), 0.5 * (y[:, 0] - y[:, 1])
            v = s if ch == "S" else m
            v = v - d * notch.bandpass_zp(v[:, None], sos, fs)[:, 0]
            m, s = (m, v) if ch == "S" else (v, s)
            y = np.stack([m + s, m - s], axis=1)
    return y


# ---------------------------------------------------------------- apply

def _apply_stem(stems, fs, names, i, fixes, upto=("eq", "duck", "comp")) -> np.ndarray:
    x = np.asarray(stems[i], dtype=np.float64)
    mine = [f for f in fixes if f["stem"] == names[i]]
    eq = [f for f in mine if f["kind"] == "eq"]
    if eq and "eq" in upto:
        x = _apply_eq(x, fs, eq)
    n = x.shape[0]
    for f in mine:
        p = f["params"]
        if f["kind"] == "duck" and "duck" in upto and p.get("kick") in names and p["depth_db"] > 0:
            k = names.index(p["kick"])
            g, st = duck_gain_db(np.asarray(stems[k], dtype=np.float64), np.asarray(stems[i]), fs,
                                 p["depth_db"], p["release_ms"], p["lookahead_ms"])
            x = x * _to_audio(g, st, n)[:, None]
    for f in mine:
        if f["kind"] == "comp" and "comp" in upto and f["params"]["max_gr_db"] > 0:
            g, st = _comp_gain_db(x, fs, f["params"])
            x = x * _to_audio(g, st, n)[:, None]
    return x


def apply_sum(stems: np.ndarray, fs: int, names, fixes: list[dict]) -> np.ndarray:
    """The corrected stems' sum (n, 2), one stem at a time: extra memory is one stem and
    the sum, not a copy of the set. Equals apply(...).sum(axis=0) up to summation order."""
    names = list(names)
    total = np.zeros(np.shape(stems)[1:], dtype=np.float64)
    for i, nm in enumerate(names):
        if any(f["stem"] == nm for f in fixes):
            total += _apply_stem(stems, fs, names, i, fixes)
        else:
            total += stems[i]
    return total


def apply(stems: np.ndarray, fs: int, names, fixes: list[dict]) -> np.ndarray:
    names = list(names)
    out = np.array(stems, dtype=np.float64, copy=True)
    for i, nm in enumerate(names):
        if any(f["stem"] == nm for f in fixes):
            out[i] = _apply_stem(stems, fs, names, i, fixes)
    return out


# ---------------------------------------------------------------- plan

def _key(f: dict) -> tuple:
    p = f["params"]
    if f["kind"] == "eq" and p.get("section"):
        # one fix per sum section: a later pass's corner moves with the re-solved sum, the
        # fix keeps its first corner and deepens
        return (f["stem"], "eq", p["section"])
    if f["kind"] == "eq":
        return (f["stem"], "eq", p["shape"], p.get("channel", "LR"), round(float(p["fc_hz"]), 1))
    return (f["stem"], f["kind"])


def _eq_shape(section: str, lo: float, hi: float) -> dict:
    if section in ("high_shelf", "air_shelf"):
        return {"shape": "high_shelf", "fc_hz": float(lo), "q": None, "channel": "LR"}
    if section in ("high_shelf_side", "air_shelf_side"):
        return {"shape": "high_shelf", "fc_hz": float(lo), "q": None, "channel": "S"}
    if section == "low_shelf":
        return {"shape": "low_shelf", "fc_hz": float(hi), "q": EQ_LOW_SHELF_Q, "channel": "LR"}
    fc = math.sqrt(lo * hi)
    return {"shape": "bell", "fc_hz": float(fc), "q": float(fc / max(hi - lo, 1e-9)), "channel": "LR"}


def _notch_q(f0: float) -> float:
    """The notch stage's planner Q without a measured one: NOTCH_Q_DEFAULT, at most
    NOTCH_LOW_Q_MAX below NOTCH_LOW_HZ (a high-Q bandpass at low f0 pre-rings)."""
    from eluate import planner
    return float(planner.NOTCH_LOW_Q_MAX if f0 < planner.NOTCH_LOW_HZ else planner.NOTCH_Q_DEFAULT)


def _notch_cap(det: dict) -> float:
    """The total cut: NOTCH_MAX_DB, or the planner's max_cut_db when it flagged the notch
    musical (a chord tone the planner itself held back on)."""
    mc = _num(det.get("max_cut_db"))
    if det.get("flag") == "musical" and mc is not None:
        return float(min(abs(mc), NOTCH_MAX_DB))
    return NOTCH_MAX_DB


def _onset_hit(moments, on_s: np.ndarray) -> float:
    if moments is None or len(moments) == 0:
        return 1.0
    if on_s.size == 0:
        return 0.0
    hits = 0
    for t in moments:
        j = np.searchsorted(on_s, float(t) - DUCK_MOMENT_WINDOW_S[1])
        if j < on_s.size and on_s[j] <= float(t) + DUCK_MOMENT_WINDOW_S[0]:
            hits += 1
    return hits / len(moments)


def plan(rep: dict, stems: np.ndarray, names, fs: int, prior: list[dict] | None = None,
         bpm: float | None = None) -> list[dict]:
    names = list(names)
    out = [dict(f, params=dict(f["params"])) for f in (prior or [])]
    idx = {_key(f): j for j, f in enumerate(out)}
    if not rep or not rep.get("attribution"):
        return out
    silent = {nm for nm, s in zip(names, stems) if _silent(s)}
    steps: dict[tuple, tuple[dict, float, str]] = {}   # key -> (template fix, delta, reason)
    _roles = []

    def role():
        if not _roles:
            k, b = roles(stems, names, fs)
            on = (kick_onsets(np.asarray(stems[k]), fs)[0] / _RATE) if k is not None else np.zeros(0)
            _roles.append((k, b, on))
        return _roles[0]

    def propose(fix: dict, delta: float, reason: str):
        if fix["stem"] in silent or fix["stem"] not in names or delta == 0:
            return
        k = _key(fix)
        if k not in steps or abs(delta) > abs(steps[k][1]):
            steps[k] = (fix, delta, reason)

    for e in rep["attribution"]:
        limit, det, rows = e.get("limit") or "", e.get("detail") or {}, e.get("stems") or []
        m = _EQ_RE.match(limit)
        if m:
            sec = det.get("section") or m.group(1).replace(" ", "_")
            g, lo_hi = _num(det.get("gain_db")), det.get("range_hz")
            if g is None or not lo_hi or g == 0:
                continue
            sign = 1.0 if g > 0 else -1.0
            chosen = [r for r in rows if (_num(r.get("share")) or 0.0) >= EQ_SHARE_MIN]
            wanted = _num(det.get("wanted_db"))
            if wanted is not None:
                if (wanted - g) * sign <= 0 or not chosen:
                    # the sum's section sits at its bound for the slope target, not for
                    # this range (it wanted less than it got): no stem fix moves that
                    continue
                S = sum(r["share"] for r in chosen)
                lin = 1.0 + (10.0 ** ((wanted - g) / 20.0) - 1.0) / S
                step = _db(lin) if lin > 0 else -EQ_MAX_DB
            else:
                step = sign * EQ_STEP_DB
            for r in chosen:
                shape = _eq_shape(sec, float(lo_hi[0]), float(lo_hi[1]))
                propose({"stem": r["name"], "kind": "eq", "params": {**shape, "section": sec, "gain_db": 0.0}},
                        step, limit)
            continue
        if limit.startswith("notch at "):
            f0, cap = _num(det.get("f0_hz")), _notch_cap(det)
            # a notch the planner itself held back on (suspect timbre, a chord tone, a 0 dB
            # cap) is not cut on a stem either; a rendered one is already out of the sum
            if (limit.endswith("(found)") and notch_rejected(det) is None and cap > 0
                    and f0 and rows and 0 < f0 < fs / 2):
                propose({"stem": rows[0]["name"], "kind": "eq",
                         "params": {"shape": "notch", "fc_hz": float(f0),
                                    "q": _num(det.get("q")) or _notch_q(f0),
                                    "channel": det.get("channel") if det.get("channel") in ("M", "S") else "LR",
                                    "max_cut_db": cap, "gain_db": 0.0}},
                        -NOTCH_STEP_DB, limit)
            continue
        peaky = any(s in limit for s in PEAK_LIMITS)
        mb = _MB_RE.match(limit)
        if not (peaky or mb) or not rows:
            continue
        low = peaky or int(mb.group(1)) == 0
        top = [r["name"] for r in rows[:2]]
        # the second stem is compressed too when it is within COMP_SECOND_DB of the first
        # on the entry's ranking: its leave-one-out drop of the sum's peak where it is
        # given (stems.attribute, limiter moments), else its own peak. A stem whose removal
        # does not lower the sum's peak (it cancels there) is never compressed for it.
        loo = any(r.get("peak_drop_db") is not None for r in rows[:2])
        pkey = "peak_drop_db" if loo else "peak_db"
        p0 = _num(rows[0].get(pkey))
        named = [rows[0]["name"]] + [r["name"] for r in rows[1:2]
                                     if p0 is not None and _num(r.get(pkey)) is not None
                                     and p0 - r[pkey] <= COMP_SECOND_DB]
        if loo:
            named = [nm for nm, r in zip(named, rows) if (_num(r.get(pkey)) or 0.0) > 0.0]
        moments = det.get("moments_s")
        if low:
            k, b, on = role()
            if k is not None and b is not None:
                kn, bn = names[k], names[b]
                if (top[0] == bn or set(top) == {kn, bn}) and _onset_hit(moments, on) >= DUCK_MOMENT_SHARE:
                    cur = out[idx[(bn, "duck")]]["params"]["depth_db"] if (bn, "duck") in idx else 0.0
                    if cur < DUCK_MAX_DB - 1e-9:
                        propose({"stem": bn, "kind": "duck",
                                 "params": {"kick": kn, "depth_db": 0.0, "release_ms": _release_ms(bpm),
                                            "lookahead_ms": DUCK_LOOKAHEAD_MS}},
                                DUCK_STEP_DB, limit)
                        named = [nm for nm in named if nm not in (kn, bn)]
                    else:
                        named = [bn] + [nm for nm in named if nm not in (kn, bn)]
        for nm in named:
            propose({"stem": nm, "kind": "comp",
                     "params": {"threshold_db": 0.0, "ratio": COMP_RATIO, "max_gr_db": 0.0,
                                "lookahead_ms": COMP_LOOKAHEAD_MS, "release_ms": COMP_RELEASE_MS,
                                "moments_s": None if moments is None else [float(t) for t in moments]}},
                    COMP_STEP_DB, limit)

    # merge onto prior, the total clamped to its cap
    changed = set()
    for k, (fix, delta, reason) in steps.items():
        cap = {"duck": DUCK_MAX_DB, "comp": COMP_MAX_DB}.get(fix["kind"])
        if fix["kind"] == "eq":
            mc = _num(fix["params"].get("max_cut_db"))     # 0.0 is a cap (the planner's), not "none"
            cap = (NOTCH_MAX_DB if mc is None else abs(mc)) if fix["params"]["shape"] == "notch" else EQ_MAX_DB
        field = {"eq": "gain_db", "duck": "depth_db", "comp": "max_gr_db"}[fix["kind"]]
        cur = out[idx[k]]["params"][field] if k in idx else 0.0
        new = round(float(np.clip(cur + delta, -cap, cap)), 2)
        if fix["kind"] == "eq" and fix["params"]["shape"] == "notch":
            new = min(new, 0.0)
        if abs(new - cur) < 1e-9:
            continue
        if k in idx:
            f = out[idx[k]]
            f["params"][field] = new
            f["reason"] = reason
            if fix["kind"] == "comp":
                f["params"]["moments_s"] = fix["params"]["moments_s"]
        else:
            f = {"stem": fix["stem"], "kind": fix["kind"],
                 "params": {**fix["params"], field: new}, "reason": reason}
            idx[k] = len(out)
            out.append(f)
        changed.add(f["stem"])
    # compression thresholds on each changed stem with its other fixes applied
    for f in [f for f in out if f["kind"] == "comp" and f["stem"] in changed and f["stem"] in names]:
        i = names.index(f["stem"])
        x = _apply_stem(stems, fs, names, i, out, upto=("eq", "duck"))
        p = f["params"]
        p["threshold_db"] = round(_comp_threshold(x, fs, p["max_gr_db"], p.get("moments_s")), 2)
    return out


# ---------------------------------------------------------------- describe

def _hz(f: float) -> str:
    return f"{f / 1000:.3g} kHz" if f >= 1000 else f"{f:.0f} Hz"


def describe(fix: dict) -> str:
    p, k = fix["params"], fix["kind"]
    if k == "eq":
        names = {"high_shelf": "high shelf", "low_shelf": "low shelf", "bell": "bell",
                 "notch": "notch"}
        ch = {"S": " on S", "M": " on M"}.get(p.get("channel", "LR"), "")
        q = f", Q {p['q']:.2g}" if p["shape"] in ("bell", "notch") and p.get("q") else ""
        body = f"{names[p['shape']]}{ch} {p['gain_db']:+.1f} dB at {_hz(p['fc_hz'])}{q}"
    elif k == "comp":
        body = (f"peak compression, up to {p['max_gr_db']:.1f} dB GR (threshold "
                f"{p['threshold_db']:.1f} dBFS, {p['ratio']:g}:1, {p['lookahead_ms']:g} ms lookahead, "
                f"{p['release_ms']:g} ms release)")
    else:
        body = (f"ducked under {p['kick']} by up to {p['depth_db']:.1f} dB at its onsets "
                f"(release {p['release_ms']:.0f} ms)")
    return f"{fix['stem']}: {body} ({fix.get('reason') or ''})".replace(" ()", "")
