"""Stems, phase A (sprint 2026-10-21 WP6; docs/noise-and-stems.md 2): master the sum of a
stem set with the stems profile's tighter ranges and name the stems responsible for every
limit the sum pass hit. No stem audio is changed.

Contract:
- `read_stems(paths, log=None) -> (stems, fs)`: stems (k, n, 2) float64, filled one stem
  at a time into one preallocated array (peak memory the set plus one stem); every file
  must share the sample rate and the length, except a render tail: a stem longer than the
  shortest by at most TRIM_MAX_S, whose extra samples peak below TRIM_TAIL_DB, is cut to it
  (logged); otherwise ValueError naming the stem. The same file twice (resolved paths) is
  refused. Mono is duplicated to two channels.
- `null_test(mix, total, fs) -> dict`: aligns the stems' sum `total` to the mix by one
  sample offset within +-NULL_MAX_OFFSET_MS (the per-channel cross-correlations over the
  first NULL_XCORR_S of both, summed; the magnitude's peak, so a polarity-inverted mix
  aligns and its gain comes out negative: "polarity_inverted") and one least-squares
  gain over the overlap, then residual_db = 10 log10(energy(mix - g total) / energy(mix)),
  floored at NULL_RESIDUAL_FLOOR_DB (an exact null serialises as a number); either
  signal's tail outside the overlap, past the abs(offset) samples the shift drops, is
  residual (a mix that is only the stems' opening does not pass). verdict "pass" below
  NULL_PASS_DB; "processed" up to NULL_MISMATCH_DB (bus processing on the mix, or a stem
  differing); "mismatch" above (these stems do not make this mix). worst_band_hz is the
  octave band (31.5 Hz to 16 kHz) where the residual is largest against the mix there.
  "diagnosis" names what differs, each {"kind", "text"}: "polarity"; "gain" (the mix
  more than NULL_GAIN_NOTE_DB louder or quieter than the sum); "length" (the lengths and
  the extra samples' peak); and when the verdict is not pass, one of "no_alignment" (the
  offset sits at the window's edge), "silent_stems", "subsample" (a half-sample shift
  of the sum drops the overlap's residual by NULL_SUBSAMPLE_GAIN_DB), "limited" (the
  residual sits at the mix's peaks: a limited or clipped mix), "tail" (the overlap nulls,
  the extra length is the residual), "processed" or "mismatch". `null_message(t)` is the
  verdict line with the diagnosis appended; a "processed" verdict is a warning the CLI
  lets the run continue on (the sum is mastered as given), "mismatch" and "silent" stop it.
  A silent mix (no energy: nothing to null against) gives verdict "silent", residual_db
  and gain_db None, whether the stems are silent too or not (`null_message` says which).
- `attribute(work, stems, names, fs) -> dict`: reads the sum pass's report.json and
  plan.json in work and, for each limit it hit, ranks the stems (section 6.3):
  EQ sections at their bounds (each stem's projection share of the sum in the section's
  range, Re<stem, sum> / <sum, sum> from the cross spectra, read on the channel the
  section runs on (SECTION_CHANNEL: the high and air shelves M, the _side ones S, the
  low shelf and bell L+R), so a side-only stem is present for the S shelves and absent
  from the M ones: the shares sum to 1 and a negative one cancels against the rest; and
  its level there against the sum's, on the same channel: the stem carrying the range is
  the one to brighten or cut; every stem's spectra are read once per attribution,
  `_Spectra`, and integrated over each section's range);
  multiband bands at their exception or band limiter at its cap, and speclim cuts past
  SPECLIM_CUT_DB (each stem's band peak at the ATTR_MOMENTS moments of largest GR in that
  band and channel, against the sum's; the band is split around the moments only);
  the clipper short of its target or the limiter on its outlier cap (ranked by the
  leave-one-out drop of the sum's 8x true peak at the limiter's largest-GR moments,
  "peak_drop_db" = peak(sum) - peak(sum - stem): a stem in anti-phase at the peaks has
  a negative drop and ranks last, whatever its own peak, which is carried as "peak_db");
  the limiter held by its average budget, or the loudness shortfall (report.json's
  warnings; one entry, detail "which"), ranked the same way at the moments of the
  limiter's median reduction (its typical work in the loud section), else by projection
  share of the whole sum; notches found (each stem's PSD at f0 over the notch's spans on
  the notch's channel, M, S, else L+R, against its neighbours a third of an octave either
  side; one the planner itself held back on, `notch_rejected`, is labelled "found,
  skipped: <why>" and is not a limit); a crest or section flag (each stem's crest and its
  projection share of each section's sum); noise exposed or present (each stem's own floor,
  noise.floor). Every entry: {"kind", "limit", "detail", "stems": [{"name", ...}] ranked};
  kind one of eq, multiband, speclim, peak, budget, notch, crest, noise.
- `notch_rejected(notch) -> str | None`: why the planner held a found notch back (its
  reason starting "suspect", the chord-tone guard "within ...", "no cut", or a 0 dB cap),
  None when it ran out of room or the notch stage could not take it (a stem fix may).
- `is_limit(entry) -> bool`: whether an attribution entry is a real limit, one a stem
  correction can act on and the stems loop scores: input EQ at a bound, a multiband band
  at its exception or cap, a frequency-dependent limiter cut, the clipper short or the
  limiter on its outlier cap, the limiter held by its average budget or a loudness
  shortfall, a notch found, not rendered and not rejected by the planner. Not: the crest
  / section row (informational), noise (no correction acts on it), a rendered or
  planner-rejected notch. Read from "kind" where the entry carries one, else its text.
- `pass_score(report, attribution) -> dict`: a stems pass's score (lower is better):
  rms_shortfall_db (report.json loudness: target_rms_dbfs - after_rms_dbfs, floored at 0)
  + LIMIT_WEIGHT per real limit; {"score", "rms_shortfall_db", "limits": [limit text],
  "dominant": which term is the larger ("rms_shortfall" or "limits", None when both are
  0), "shortfall_cause": the loudness warning's cause, when there is one}.
- `write_report(work, rep)`: stems-report.json and stems-report.md; with rep["passes"]
  (cli.do_master_stems' loop) a Passes section: each pass's score, limits and fixes, the
  pass kept and why the loop stopped.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import maximum_filter1d

from eluate import bands, io, meter, noise
from eluate.plan import Plan

NULL_PASS_DB = -40.0
NULL_MISMATCH_DB = -20.0
NULL_MAX_OFFSET_MS = 250.0     # linear-phase EQ or lookahead limiting on the mix bus puts
                               # 20 to 100 ms between a mix print and stems printed without it
NULL_XCORR_S = 30.0
NULL_RESIDUAL_FLOOR_DB = -200.0  # an exact null reads this, not -inf (which serialises as None)
NULL_GAIN_NOTE_DB = 0.5        # a mix this much louder or quieter than the sum is reported
NULL_LIMITED_GAIN_DB = 1.0     # ... and this much louder with the residual at its peaks: limited
NULL_PEAK_SHARE = 0.01         # the mix's top 1 % of samples are its peaks
NULL_PEAK_WINDOW_MS = 5.0      # +- around each peak sample
NULL_PEAK_CONC = 3.0           # residual energy per sample at the peaks this many times the rest
NULL_SUBSAMPLE_GAIN_DB = 10.0  # a half-sample shift dropping the overlap's residual by this
ATTR_MOMENTS = 8               # largest-GR moments a band's ranking reads
ATTR_WINDOW_S = 0.05           # +- around each moment
ATTR_SPACING_S = 0.25          # moments at least this far apart
ATTR_SPLIT_MARGIN_S = 0.5      # a band split around a moment runs this far either side of the
                               # window (the crossovers' transients die well inside it)
SPECTRA_NPERSEG = 8192         # the per-stem Welch / cross spectra (2048 under 8192 samples)
SPECLIM_CUT_DB = 3.0
EDGE_DB = 0.05                 # a gain this close to its bound is at it
OCTAVES_HZ = (31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)
TRIM_MAX_S = 0.1               # longest render tail read_stems cuts to the shortest stem
TRIM_TAIL_DB = -60.0           # ... and only when it peaks below this


def _db(v):
    return 20.0 * math.log10(v) if v > 0 else -math.inf


def _pdb(v):
    return 10.0 * math.log10(v) if v > 0 else -math.inf


def read_stems(paths, log=None) -> tuple[np.ndarray, int]:
    """Lengths may differ by a render tail: every stem is cut to the shortest when what
    is cut is at most TRIM_MAX_S long and peaks below TRIM_TAIL_DB (a DAW export's tail,
    the maintainer's stem set: 481 samples at -94.8 dBFS); anything more is an error.
    The rates and lengths are read from the headers first (soundfile.info), so the set is
    filled into one preallocated (k, n, 2) array, one stem at a time."""
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("no stems")
    seen: dict[str, Path] = {}
    for p in paths:
        key = os.path.normcase(os.path.abspath(str(p)))
        if key in seen:
            raise ValueError(f"stem {p.name} is given twice ({seen[key]} and {p})")
        seen[key] = p
    infos = [sf.info(str(p)) for p in paths]
    fs0 = int(infos[0].samplerate)
    for p, inf in zip(paths, infos):
        if int(inf.samplerate) != fs0:
            raise ValueError(f"stem {p.name}: {inf.samplerate} Hz, the first stem is {fs0} Hz")
    n = min(int(inf.frames) for inf in infos)
    out = np.empty((len(paths), n, 2), dtype=np.float64)
    for i, p in enumerate(paths):
        x, fs = io.read(p)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        x = x[:, :2]
        if x.shape[0] < n:
            raise ValueError(f"stem {p.name}: {x.shape[0]} samples read, its header says "
                             f"{infos[i].frames}")
        extra = x.shape[0] - n
        if extra:
            peak = _db(float(np.max(np.abs(x[n:]))))
            if extra > TRIM_MAX_S * fs0 or peak > TRIM_TAIL_DB:
                raise ValueError(f"stem {p.name}: {x.shape[0]} samples, the shortest has {n} "
                                 f"({extra / fs0 * 1000:.0f} ms more, peaking at {peak:.1f} dBFS)")
            if log:
                log(f"stem {p.name}: its last {extra} samples cut to the shortest stem "
                    f"({extra / fs0 * 1000:.1f} ms, peak {peak:.1f} dBFS)")
        out[i] = x[:n]
        del x
    return out, fs0


# ---------------------------------------------------------------- null test

def _offset(mix: np.ndarray, total: np.ndarray, fs: int) -> tuple[int, bool]:
    """The lag (samples) to shift total by so it lines up with mix (within the limit), and
    whether it sits at the window's edge (no alignment inside it). The channels correlate
    jointly (a mono fold would null a side-only signal); the correlation's magnitude is
    read, so a polarity-inverted mix aligns too (its gain then comes out negative)."""
    m = min(len(mix), len(total), int(NULL_XCORR_S * fs))
    lim = min(int(round(NULL_MAX_OFFSET_MS * 1e-3 * fs)), m - 1)
    c = sum(signal.correlate(mix[:m, ch], total[:m, ch], mode="full", method="fft")
            for ch in range(mix.shape[1]))
    lags = np.arange(-m + 1, m)
    sel = np.abs(lags) <= lim
    lag = int(lags[sel][np.argmax(np.abs(c[sel]))])
    return lag, abs(lag) >= lim > 0


def _fit(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(the least-squares gain of b onto a, signed; the residual's energy re a's, dB) over
    their common length."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    bb = float(np.sum(b ** 2))
    g = float(np.sum(a * b) / bb) if bb > 0 else 1.0
    return g, _pdb(float(np.sum((a - g * b) ** 2))) - _pdb(float(np.sum(a ** 2)))


def _frac_shift(x: np.ndarray, d: float) -> np.ndarray:
    """x delayed by d samples (fractional) through an FFT phase ramp; circular, so the
    ends wrap (they are near silent in a track)."""
    n = x.shape[0]
    w = 2 * np.pi * np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(x, axis=0) * np.exp(-1j * w * d)[:, None], n=n, axis=0)


def _subsample(a: np.ndarray, b: np.ndarray, fs: int) -> tuple[float, float, float] | None:
    """A sub-sample offset between the aligned mix a and sum b: the fractional shift d in
    [-0.5, 0.5] of b that minimises the least-squares residual over the first NULL_XCORR_S
    (a golden-section search on the residual energy, _frac_shift), reported as (d, the
    residual before, after; dB) when it drops by NULL_SUBSAMPLE_GAIN_DB; else None."""
    m = min(len(a), len(b), int(NULL_XCORR_S * fs))
    if m < 16:
        return None
    a, b = a[:m], b[:m]
    before = max(_fit(a, b)[1], NULL_RESIDUAL_FLOOR_DB)

    def e(d):
        return max(_fit(a, _frac_shift(b, d))[1], NULL_RESIDUAL_FLOOR_DB)

    lo, hi = -0.5, 0.5
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = hi - phi * (hi - lo), lo + phi * (hi - lo)
    f1, f2 = e(x1), e(x2)
    for _ in range(10):                 # the bracket closes to under 0.01 sample
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - phi * (hi - lo)
            f1 = e(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + phi * (hi - lo)
            f2 = e(x2)
    d, after = (x1, f1) if f1 < f2 else (x2, f2)
    if after < before - NULL_SUBSAMPLE_GAIN_DB:
        return float(d), float(before), float(after)
    return None


def _peak_concentration(a: np.ndarray, r: np.ndarray, fs: int) -> float:
    """How much more the residual r concentrates at the mix's peaks than the mix a itself
    does: each one's energy per sample within +-NULL_PEAK_WINDOW_MS of a's top
    NULL_PEAK_SHARE of samples over its energy per sample elsewhere, the residual's ratio
    over the mix's. 1 means the residual is spread like the signal (a gain or channel
    error, a small time shift); a limiter's or clipper's residual sits at the peaks only."""
    env = np.max(np.abs(a), axis=1)
    if env.size == 0 or not np.any(env):
        return 1.0
    mask = env >= np.percentile(env, 100.0 * (1.0 - NULL_PEAK_SHARE))
    w = int(round(NULL_PEAK_WINDOW_MS * 1e-3 * fs))
    mask = maximum_filter1d(mask.astype(np.uint8), size=2 * w + 1).astype(bool)
    k = int(mask.sum())
    if k == 0 or k == mask.size:
        return 1.0

    def conc(x):
        e = np.sum(x ** 2, axis=1)
        at, away = float(e[mask].sum()) / k, float(e[~mask].sum()) / (mask.size - k)
        return at / away if away > 0 else math.inf

    cr, ca = conc(r), conc(a)
    if not math.isfinite(cr):
        return math.inf
    return cr / ca if ca > 0 and math.isfinite(ca) else 1.0


def null_test(mix: np.ndarray, total: np.ndarray, fs: int) -> dict:
    mix = np.asarray(mix, dtype=np.float64)
    total = np.asarray(total, dtype=np.float64)
    base = {"pass_db": NULL_PASS_DB, "mismatch_db": NULL_MISMATCH_DB,
            "mix_frames": int(len(mix)), "sum_frames": int(len(total)),
            "max_offset_ms": NULL_MAX_OFFSET_MS}
    if not np.any(mix):
        return {"residual_db": None, "gain_db": None, "offset_samples": 0, "offset_ms": 0.0,
                "verdict": "silent", "stems_silent": not np.any(total), "worst_band_hz": None,
                "worst_band_residual_db": None, "polarity_inverted": False, "diagnosis": [],
                **base}
    lag, at_edge = _offset(mix, total, fs) if np.any(total) else (0, False)
    if lag >= 0:
        a, b = mix[lag:], total[:len(total)]
    else:
        a, b = mix, total[-lag:]
    n = min(len(a), len(b))
    g, overlap_db = _fit(a[:n], b[:n]) if np.any(b[:n]) else (1.0, 0.0)
    # residual over the longer of the two; the tail outside the overlap is residual too,
    # past the abs(lag) samples the shift itself drops off one end
    N = max(len(a), len(b))
    r = np.zeros((N, a.shape[1]))
    r[:n] = a[:n] - g * b[:n]
    t0 = n + abs(lag)
    r[t0:len(a)] = a[t0:]
    r[t0:len(b)] = g * b[t0:]
    extra_peak = _db(float(np.max(np.abs(r[t0:])))) if t0 < N else -math.inf
    a = np.concatenate([a, np.zeros((N - len(a), a.shape[1]))])
    res_db = max(_pdb(np.sum(r ** 2)) - _pdb(np.sum(a ** 2)), NULL_RESIDUAL_FLOOR_DB)
    overlap_db = max(overlap_db, NULL_RESIDUAL_FLOOR_DB)
    verdict = ("pass" if res_db < NULL_PASS_DB else
               "processed" if res_db <= NULL_MISMATCH_DB else "mismatch")
    worst, worst_db = None, -math.inf
    if N >= 4096:
        nps = 8192 if N >= 8192 else 4096
        f, pr = signal.welch(r, fs, nperseg=nps, axis=0)
        _, pa = signal.welch(a, fs, nperseg=nps, axis=0)
        pr, pa = pr.sum(axis=1), pa.sum(axis=1)          # per channel, not a mono fold
        for fc in OCTAVES_HZ:
            if fc * math.sqrt(2) >= fs / 2:
                break
            sel = (f >= fc / math.sqrt(2)) & (f < fc * math.sqrt(2))
            if not np.any(sel) or pa[sel].sum() <= 0:
                continue
            d = _pdb(pr[sel].sum()) - _pdb(pa[sel].sum())
            if d > worst_db:
                worst, worst_db = fc, d
    gain_db = _db(abs(g)) if g else NULL_RESIDUAL_FLOOR_DB
    # what differs: named, so the message says what to do instead of guessing
    diag: list[dict] = []
    if g < 0:
        diag.append({"kind": "polarity", "text": "the mix is polarity-inverted against the "
                     "stems' sum (the sign is taken into account)"})
    if abs(gain_db) > NULL_GAIN_NOTE_DB:
        diag.append({"kind": "gain", "text": f"the mix is {abs(gain_db):.1f} dB "
                     f"{'louder' if gain_db > 0 else 'quieter'} than the stems' sum (the gain "
                     "is taken into account)"})
    if len(mix) != len(total):
        longer = "mix" if len(mix) > len(total) else "stems"
        ms = abs(len(mix) - len(total)) / fs * 1000.0
        diag.append({"kind": "length", "text": f"the mix is {ms:.0f} ms "
                     f"{'longer' if longer == 'mix' else 'shorter'} than the stems ({len(mix)} vs "
                     f"{len(total)} samples); the extra {longer} samples "
                     + (f"peak at {extra_peak:.1f} dBFS and count as residual"
                        if math.isfinite(extra_peak) else "are silent")})
    if verdict != "pass":
        if at_edge:
            diag.append({"kind": "no_alignment",
                         "text": f"no alignment found within +-{NULL_MAX_OFFSET_MS:g} ms"})
        if not np.any(total):
            diag.append({"kind": "silent_stems", "text": "the stems are silent"})
        else:
            sub = _subsample(a[:n], b[:n], fs) if n >= 2 else None
            # the channels: swapped, or at different gains
            swap_db = max(_fit(a[:n], b[:n, ::-1])[1], NULL_RESIDUAL_FLOOR_DB) if n >= 2 else 0.0
            ch_db = [_db(abs(_fit(a[:n, c:c + 1], b[:n, c:c + 1])[0])) for c in range(a.shape[1])] \
                if n >= 2 else []
            if sub is not None:
                diag.append({"kind": "subsample", "text": f"a sub-sample offset: shifting the "
                             f"stems' sum by {sub[0]:+.2f} sample takes the residual over the first "
                             f"{NULL_XCORR_S:g} s of the overlap from {_f(sub[1])} to {_f(sub[2])} dB "
                             "(the stems are aligned to one sample; print the mix and the stems "
                             "from one session)"})
            elif overlap_db < NULL_PASS_DB and len(mix) != len(total):
                diag.append({"kind": "tail", "text": f"the overlap nulls (residual {_f(overlap_db)} "
                             "dB): the difference is the extra length"})
            elif swap_db < NULL_PASS_DB:
                diag.append({"kind": "swapped", "text": "the mix's channels are swapped against "
                             f"the stems' sum (swapped, the residual is {_f(swap_db)} dB)"})
            elif (len(ch_db) == 2 and all(math.isfinite(v) for v in ch_db)
                  and abs(ch_db[0] - ch_db[1]) > NULL_GAIN_NOTE_DB):
                diag.append({"kind": "channel_gain", "text": "the mix's channels sit at different "
                             f"gains against the stems' sum (L {ch_db[0]:+.1f} dB, R {ch_db[1]:+.1f} dB)"})
            elif _peak_concentration(a[:n], r[:n], fs) >= NULL_PEAK_CONC:
                diag.append({"kind": "limited", "text": "the mix is limited or clipped (the "
                             "residual sits at its peaks"
                             + (f", and it is {gain_db:.1f} dB louder" if gain_db > NULL_LIMITED_GAIN_DB else "")
                             + "); the stems are mastered as given"})
            elif verdict == "processed":
                diag.append({"kind": "processed", "text": "the mix carries processing the stems "
                             "lack (a bus stage), or a stem differs slightly; the stems are "
                             "mastered as given"})
            else:
                diag.append({"kind": "mismatch", "text": "a stem is missing or this is another "
                             "mix; omit --mix, or give the mix these stems were printed from"})
    return {"residual_db": float(res_db), "overlap_residual_db": float(overlap_db),
            "gain_db": float(gain_db), "polarity_inverted": bool(g < 0),
            "offset_samples": lag, "offset_ms": lag / fs * 1000.0, "verdict": verdict,
            "worst_band_hz": worst,
            "worst_band_residual_db": None if worst is None else float(worst_db),
            "diagnosis": diag, **base}


def null_message(t: dict) -> str:
    if t["verdict"] == "silent":
        return ("The mix and the stems are both silent: nothing to master."
                if t.get("stems_silent") else
                "The mix is silent, so the stems cannot be checked against it: give the mix "
                "these stems were printed from.")
    res = _f(t.get("residual_db"))
    notes = [d["text"] for d in t.get("diagnosis") or [] if d.get("text")]
    body = "; ".join(notes)
    tail = (" " + body[0].upper() + body[1:] + ".") if body else ""
    if t["verdict"] == "pass":
        pol = ", polarity inverted" if t.get("polarity_inverted") else ""
        return (f"Stems null against the mix: residual {res} dB (offset {t.get('offset_samples', 0)} "
                f"samples, gain {_f(t.get('gain_db'))} dB{pol})." + tail)
    if t["verdict"] == "processed":
        return (f"The stems do not null against the mix: residual {res} dB "
                f"(pass is below {NULL_PASS_DB:g} dB), largest around {_f(t.get('worst_band_hz'), 0)} Hz. "
                "The run continues: the sum is mastered as given and the report says the mix "
                "was processed." + tail)
    return (f"These stems do not make this mix: residual {res} dB "
            f"(above {NULL_MISMATCH_DB:g} dB)." + tail)


# ---------------------------------------------------------------- attribution

def _moments(gr: np.ndarray, rate_hz: float) -> np.ndarray:
    """Times (s) of the ATTR_MOMENTS largest reductions (gr dB <= 0), ATTR_SPACING_S apart."""
    order = np.argsort(gr)
    gap = int(round(ATTR_SPACING_S * rate_hz))
    out: list[int] = []
    for i in order:
        if gr[i] >= -1e-6 or len(out) >= ATTR_MOMENTS:
            break
        if all(abs(i - j) >= gap for j in out):
            out.append(int(i))
    return np.asarray(out, dtype=np.float64) / rate_hz


def _peak_at(a: np.ndarray, fs: int, times: np.ndarray, tp: bool = False) -> float:
    """Mean over the moments of the peak (dBFS) within +-ATTR_WINDOW_S of each."""
    w = int(round(ATTR_WINDOW_S * fs))
    vals = []
    for t in times:
        i = int(round(t * fs))
        seg = a[max(0, i - w):i + w]
        if seg.size == 0:
            continue
        v = meter.true_peak(seg, fs) if tp else _db(float(np.max(np.abs(seg))))
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else -math.inf


def _rank(names, vals, ref, key="peak_db", top=2, below=True) -> list[dict]:
    rows = [{"name": n, key: v, "below_sum_db": (ref - v) if below and math.isfinite(v) else None}
            for n, v in zip(names, vals) if math.isfinite(v)]
    rows.sort(key=lambda r: -r[key])
    return rows[:top]


def _band_peak_at(x, fs, xo, k, c, times, order, phase) -> float:
    """Mean over the moments of band k's peak (dBFS) on M/S channel c within +-ATTR_WINDOW_S
    of each. The split runs on the window plus ATTR_SPLIT_MARGIN_S either side only (the
    crossovers' transients die inside the margin), never on the whole stem."""
    w, mg = int(round(ATTR_WINDOW_S * fs)), int(round(ATTR_SPLIT_MARGIN_S * fs))
    n = x.shape[0]
    vals = []
    for t in times:
        i = int(round(t * fs))
        a, b = max(0, i - w - mg), min(n, i + w + mg)
        lo, hi = max(0, i - w) - a, min(n, i + w) - a
        if b - a < 2 or hi <= lo:
            continue
        seg = bands.split(bands.to_ms(x[a:b]), fs, xo, order=order, phase=phase)[k, lo:hi, c]
        v = _db(float(np.max(np.abs(seg))))
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else -math.inf


def _band_moments_rank(stems, names, total, fs, xo, k, c, env_db, rate_hz, order, phase):
    times = _moments(env_db, rate_hz)
    if times.size == 0:
        return None, []
    ref = _band_peak_at(total, fs, xo, k, c, times, order, phase)
    vals = [_band_peak_at(s, fs, xo, k, c, times, order, phase) for s in stems]
    return times, _rank(names, vals, ref)


def _loo_rank(names, stems, total, fs: int, times: np.ndarray, top: int = 2) -> list[dict]:
    """Rank the stems at the moments by the leave-one-out drop of the sum's true peak,
    peak_drop_db = peak(sum) - peak(sum - stem) (means over the moments of the peak within
    +-ATTR_WINDOW_S): the stem that builds the peak ranks first; one in anti-phase there
    has a negative drop (removing it raises the peak) and ranks last whatever its own
    peak, carried as peak_db. Silent stems at the moments are left out."""
    w = int(round(ATTR_WINDOW_S * fs))
    idx = [(max(0, int(round(t * fs)) - w), int(round(t * fs)) + w) for t in times]
    idx = [(a, b) for a, b in idx if b > a and a < total.shape[0]]
    if not idx:
        return []

    def mean_tp(get):
        vals = [meter.true_peak(get(a, b), fs) for a, b in idx]
        vals = [v for v in vals if math.isfinite(v)]
        return float(np.mean(vals)) if vals else -math.inf

    ref = mean_tp(lambda a, b: total[a:b])
    rows = []
    for nm, s in zip(names, stems):
        own = mean_tp(lambda a, b: s[a:b])
        if not math.isfinite(own):
            continue
        without = mean_tp(lambda a, b: total[a:b] - s[a:b])
        rows.append({"name": nm, "peak_db": own,
                     "below_sum_db": (ref - own) if math.isfinite(ref) else None,
                     "peak_drop_db": (ref - without) if math.isfinite(ref) and math.isfinite(without)
                     else (math.inf if math.isfinite(ref) else 0.0)})
    rows.sort(key=lambda r: -r["peak_drop_db"])
    return rows[:top]


def _moments_median(gr: np.ndarray, rate_hz: float) -> np.ndarray:
    """Times (s) of ATTR_MOMENTS reductions nearest the median of the active ones (gr < 0),
    ATTR_SPACING_S apart: the limiter's typical work, not its outliers."""
    act = np.nonzero(gr < -1e-6)[0]
    if act.size == 0:
        return np.zeros(0)
    med = float(np.median(gr[act]))
    order = act[np.argsort(np.abs(gr[act] - med), kind="stable")]
    gap = int(round(ATTR_SPACING_S * rate_hz))
    out: list[int] = []
    for i in order:
        if len(out) >= ATTR_MOMENTS:
            break
        if all(abs(i - j) >= gap for j in out):
            out.append(int(i))
    return np.asarray(sorted(out), dtype=np.float64) / rate_hz


def _env_db(plan_env, n_ctrl: int) -> np.ndarray:
    v = plan_env.values
    return np.full(n_ctrl, float(v[0])) if v.size == 1 else np.asarray(v[:n_ctrl], dtype=np.float64)


class _Spectra:
    """Every stem's spectra, read once per attribution and integrated over each range the
    sections ask for: the Welch PSDs of its L and R, its L-R cross spectrum, and its
    cross spectra with the sum's L and R (all four pairings), SPECTRA_NPERSEG points. From
    those the PSD and the projection on any channel set follow by bilinearity: "LR" sums L
    and R; "M" reads (L+R)/2, PSD_M = (P_L + P_R + 2 Re C_LR) / 4 and the projection
    Re<M_s, M_t> = Re(C_LL + C_LR + C_RL + C_RR) / 4; "S" reads (L-R)/2 with the cross
    terms negated. The sum's own PSDs the same way."""

    def __init__(self, stems, total, fs: int):
        self.stems, self.total, self.fs = stems, total, fs
        self.nps = SPECTRA_NPERSEG if total.shape[0] >= SPECTRA_NPERSEG else 2048
        self.f, pt = signal.welch(total, fs, nperseg=self.nps, axis=0)
        _, c = signal.csd(total[:, 0], total[:, 1], fs, nperseg=self.nps)
        self.t = {"ll": pt[:, 0], "rr": pt[:, 1], "lr": np.real(c)}
        self._s: dict[int, dict] = {}

    def stem(self, i: int) -> dict:
        if i not in self._s:
            s, t, fs, nps = self.stems[i], self.total, self.fs, self.nps
            _, ps = signal.welch(s, fs, nperseg=nps, axis=0)
            _, c = signal.csd(s[:, 0], s[:, 1], fs, nperseg=nps)
            _, x = signal.csd(s, t, fs, nperseg=nps, axis=0)           # L_s L_t, R_s R_t
            _, xlr = signal.csd(s[:, 0], t[:, 1], fs, nperseg=nps)      # L_s R_t
            _, xrl = signal.csd(s[:, 1], t[:, 0], fs, nperseg=nps)      # R_s L_t
            self._s[i] = {"ll": ps[:, 0], "rr": ps[:, 1], "lr": np.real(c),
                          "x_ll": np.real(x[:, 0]), "x_rr": np.real(x[:, 1]),
                          "x_lr": np.real(xlr), "x_rl": np.real(xrl)}
        return self._s[i]

    @staticmethod
    def _psd(d: dict, channel: str) -> np.ndarray:
        if channel == "LR":
            return d["ll"] + d["rr"]
        sign = 1.0 if channel == "M" else -1.0
        return 0.25 * (d["ll"] + d["rr"] + 2.0 * sign * d["lr"])

    @staticmethod
    def _proj(d: dict, channel: str) -> np.ndarray:
        if channel == "LR":
            return d["x_ll"] + d["x_rr"]
        sign = 1.0 if channel == "M" else -1.0
        return 0.25 * (d["x_ll"] + d["x_rr"] + sign * (d["x_lr"] + d["x_rl"]))

    def rows(self, names, lo: float, hi: float, channel: str = "LR") -> list[dict]:
        """`_range_rows` over [lo, hi) on the channel set."""
        sel = (self.f >= lo) & (self.f < hi)
        e_t = float(self._psd(self.t, channel)[sel].sum()) if np.any(sel) else 0.0
        rows = []
        if e_t <= 0:
            return rows
        for i, nm in enumerate(names):
            d = self.stem(i)
            e = float(self._psd(d, channel)[sel].sum())
            if e <= 0:
                continue
            rows.append({"name": nm, "share": float(self._proj(d, channel)[sel].sum()) / e_t,
                         "level_re_sum_db": _pdb(e) - _pdb(e_t)})
        rows.sort(key=lambda r: -r["share"])
        return rows


def _range_rows(names, stems, total, fs: int, lo: float, hi: float, channel: str = "LR",
                spectra: _Spectra | None = None) -> list[dict]:
    """Each stem's projection share of the sum over [lo, hi): Re<s, total> / <total, total>
    from the cross spectra on the channel set ("LR": L and R summed; "M"; "S"), and its
    level there against the sum's on the same channels. The shares sum to 1; a negative
    one cancels against the rest. Ranked by share. A side-only stem has no M and reads
    its full level on S; a mono fold would lose it."""
    return (spectra or _Spectra(stems, total, fs)).rows(names, lo, hi, channel)


SECTION_RANGES = {"low_shelf": lambda s: (20.0, s["fc_hz"]),
                  "bell": lambda s: (s["fc_hz"] / 2 ** 0.5, s["fc_hz"] * 2 ** 0.5),
                  "band_shelf": lambda s: (s["lo_hz"], s["hi_hz"]),
                  "high_shelf": lambda s: (s["fc_hz"], 20000.0),
                  "high_shelf_side": lambda s: (s["fc_hz"], 20000.0),
                  "air_shelf": lambda s: (s["fc_hz"], 20000.0),
                  "air_shelf_side": lambda s: (s["fc_hz"], 20000.0)}
# the channel each input EQ section runs on (the planner's "presence shelf (M)", "(S)")
SECTION_CHANNEL = {"high_shelf": "M", "air_shelf": "M",
                   "high_shelf_side": "S", "air_shelf_side": "S"}


def _eq_bounds() -> dict:
    from eluate import planner
    b = (planner.PROFILES["stems"]["eq_bounds"] or {})
    return {"gain_db": b.get("gain_db", 3.0), "hs_mid_db": b.get("hs_mid_db", 4.5),
            "hs_side_db": b.get("hs_side_db", 6.0)}


def attribute(work, stems: np.ndarray, names, fs: int) -> dict:
    work = Path(work)
    rep = json.loads((work / "report.json").read_text(encoding="utf-8"))
    plan = Plan.load(work / "plan.json")
    total = stems.sum(axis=0)
    names = list(names)
    out: list[dict] = []
    n = total.shape[0]
    from eluate.plan import CONTROL_RATE_HZ, control_length
    n_ctrl = control_length(n, fs, CONTROL_RATE_HZ)

    # EQ sections at their bounds: every stem's spectra once, integrated per section
    bnd = _eq_bounds()
    secs = ((rep.get("chain") or {}).get("inputeq") or {}).get("sections") or {}
    spectra = None
    for name, sec in secs.items():
        if sec is None or name == "highpass" or sec.get("gain_db") is None:
            continue
        g = float(sec["gain_db"])
        cap = (bnd["hs_mid_db"] if name in ("high_shelf", "air_shelf") else
               bnd["hs_side_db"] if name in ("high_shelf_side", "air_shelf_side")
               else bnd["gain_db"])
        if not (g >= cap - EDGE_DB or g <= -bnd["gain_db"] + EDGE_DB):
            continue
        lo, hi = SECTION_RANGES[name](sec)
        ch = SECTION_CHANNEL.get(name, "LR")
        if spectra is None:
            spectra = _Spectra(stems, total, fs)
        rows = spectra.rows(names, lo, hi, ch)
        detail = {"gain_db": g, "bound_db": cap, "range_hz": [lo, hi], "section": name,
                  "channel": ch}
        # the total gain the planner wanted in the range (its range_deficit_db), so stemfix
        # sizes the stem shelf from the excess over the bound
        caps = (rep.get("planner") or {}).get("inputeq") or {}
        caps = caps.get("side_caps" if name.endswith("_side") else "top_caps") or {}
        key = "presence" if name.startswith("high_shelf") else "air" if name.startswith("air") else None
        wanted = (caps.get(key) or {}).get("range_deficit_db") if key else None
        if isinstance(wanted, (int, float)):
            detail["wanted_db"] = float(wanted)
        out.append({"kind": "eq",
                    "limit": f"input EQ {name.replace('_', ' ')} {g:+.1f} dB, at its bound",
                    "detail": detail, "stems": rows[:2]})
    del spectra

    # multiband exception / band limiter at its cap
    mbp = plan.multiband
    rows_mb = ((mbp.budget or {}).get("result") or []) if mbp is not None else []
    for k, row in enumerate(rows_mb):
        for c, r in enumerate(row):
            if r.get("gated"):
                continue
            at_cap = (r.get("limit_db") or 0.0) >= (r.get("limit_cap_db") or math.inf) - EDGE_DB
            if not (r.get("exception") or at_cap):
                continue
            env = np.clip(_env_db(mbp.gr_db[k][c], n_ctrl), -6.0, 0.0)
            if mbp.ceiling_gr_db is not None:
                env = env + _env_db(mbp.ceiling_gr_db[k][c], n_ctrl)
            times, ranked = _band_moments_rank(stems, names, total, fs, mbp.crossovers_hz, k, c,
                                               env, CONTROL_RATE_HZ, mbp.order, mbp.phase)
            out.append({"kind": "multiband",
                        "limit": f"multiband band {k} {'MS'[c]} "
                                 + ("at its exception" if r.get("exception") else "band limiter at its cap"),
                        "detail": {"band": k, "channel": "MS"[c], "comp_db": r.get("comp_db"),
                                   "limit_db": r.get("limit_db"),
                                   "moments_s": None if times is None else times.tolist()},
                        "stems": ranked})

    # speclim cuts past SPECLIM_CUT_DB
    sl = plan.speclim
    rows_sl = ((sl.budget or {}).get("result") or []) if sl is not None else []
    for k, row in enumerate(rows_sl):
        for c, r in enumerate(row):
            if (r.get("cut_db") or 0.0) <= SPECLIM_CUT_DB or sl.ceiling_gr_db is None:
                continue
            env = _env_db(sl.ceiling_gr_db[k][c], n_ctrl)
            times, ranked = _band_moments_rank(stems, names, total, fs, sl.crossovers_hz, k, c,
                                               env, CONTROL_RATE_HZ, sl.order, sl.phase)
            out.append({"kind": "speclim",
                        "limit": f"frequency-dependent limiter band {k} {'MS'[c]} cut "
                                 f"{r['cut_db']:.1f} dB",
                        "detail": {"band": k, "channel": "MS"[c], "cut_db": r["cut_db"],
                                   "moments_s": None if times is None else times.tolist()},
                        "stems": ranked})

    # clipper short of its target, limiter on its outlier cap: the stems building the
    # sum's peak at the limiter's largest reductions (leave-one-out)
    warns = rep.get("warnings") or []
    lim_w = next((w for w in warns if w.get("stage") == "limiter"), None) or {}
    lim_bound = lim_w.get("bound")
    clip_w = any(w.get("stage") == "clipper" for w in warns)
    loud_w = next((w for w in warns if w.get("stage") == "loudness"), None) or {}
    short = loud_w.get("shortfall_db")
    short = float(short) if isinstance(short, (int, float)) and math.isfinite(short) and short > 0 else None
    t = gr = None
    if (work / "limiter_gr.npz").exists():
        z = np.load(work / "limiter_gr.npz")
        t, gr = np.asarray(z["t"], dtype=np.float64), np.asarray(z["gr_db"], dtype=np.float64)
        rate = 1.0 / float(np.median(np.diff(t))) if t.size > 1 else CONTROL_RATE_HZ
        t_off = float(t[0]) if t.size else 0.0
    if (clip_w or lim_bound == "outlier") and gr is not None:
        times = _moments(-np.abs(gr), rate) + t_off
        out.append({"kind": "peak",
                    "limit": ("clipper short of its target" if clip_w else "")
                    + (" and " if clip_w and lim_bound == "outlier" else "")
                    + ("limiter on its outlier cap" if lim_bound == "outlier" else ""),
                    "detail": {"moments_s": times.tolist(), "which": "largest reductions"},
                    "stems": _loo_rank(names, stems, total, fs, times)})

    # the limiter held by its average budget, or the loudness shortfall: the master is
    # short of the target and the loop must see where the limiter's typical work goes
    if lim_bound == "average" or short is not None:
        parts, which = [], []
        if lim_bound == "average":
            parts.append(f"limiter held by its average budget ({_f(lim_w.get('needed_db'))} dB "
                         f"needed, cap {_f(lim_w.get('cap_db'))} dB)")
            which.append("limiter average budget")
        if short is not None:
            cause = loud_w.get("cause")
            parts.append(f"loudness {short:.1f} dB short of the target"
                         + (f" ({cause})" if cause and lim_bound != "average" else ""))
            which.append("loudness shortfall")
        detail = {"bound": lim_bound, "needed_db": lim_w.get("needed_db"),
                  "cap_db": lim_w.get("cap_db"), "shortfall_db": short,
                  "cause": loud_w.get("cause"), "which": " and ".join(which)}
        rows: list[dict] = []
        if gr is not None:
            times = _moments_median(gr, rate) + t_off
            if times.size:
                detail["moments_s"] = times.tolist()
                detail["ranked_by"] = "leave-one-out peak drop at the limiter's median reductions"
                rows = _loo_rank(names, stems, total, fs, times)
        if not rows:
            tot = float(np.sum(total ** 2))
            rows = [{"name": nm, "share": float(np.sum(s * total)) / tot}
                    for nm, s in zip(names, stems) if tot > 0]
            rows.sort(key=lambda r: -r["share"])
            detail["ranked_by"] = "projection share of the sum"
            rows = rows[:2]
        out.append({"kind": "budget", "limit": ", ".join(parts), "detail": detail, "stems": rows})

    # notches found, on the notch's channel
    for nt in rep.get("notches") or []:
        f0, spans = nt.get("f0_hz"), nt.get("spans") or []
        if not f0 or not spans:
            continue
        ch = nt.get("channel") if nt.get("channel") in ("M", "S") else "LR"
        why = notch_rejected(nt)
        rows = []
        for nm, s in zip(names, stems):
            seg = np.concatenate([s[int(a * fs):int(b * fs)] for a, b in spans]
                                 or [s[:0]])
            if len(seg) < 2048:
                continue
            nps = 4096 if len(seg) >= 4096 else 2048
            if ch == "LR":
                f, p = signal.welch(seg, fs, nperseg=nps, axis=0)
                p = p.sum(axis=1)                      # L and R, not a mono fold
            else:
                v = 0.5 * (seg[:, 0] + seg[:, 1]) if ch == "M" else 0.5 * (seg[:, 0] - seg[:, 1])
                f, p = signal.welch(v, fs, nperseg=nps)
            at = p[np.argmin(np.abs(f - f0))]
            nb = (np.abs(np.log2(np.maximum(f, 1e-9) / f0)) <= 1 / 3) & (np.abs(f - f0) > f0 * 0.03)
            if at <= 0 or not np.any(nb):
                continue
            rows.append({"name": nm, "excess_db": _pdb(at) - _pdb(float(np.median(p[nb])))})
        rows.sort(key=lambda r: -r["excess_db"])
        label = ("rendered" if nt.get("rendered") else "found" if why is None
                 else f"found, skipped: {why}")
        out.append({"kind": "notch", "limit": f"notch at {f0:.0f} Hz ({label})",
                    "detail": {"f0_hz": f0, "spans": spans, "read_on": ch,
                               "actionable": why is None and not nt.get("rendered"),
                               **{k: nt[k] for k in ("channel", "q", "flag", "max_cut_db", "reason")
                                  if nt.get(k) is not None}},
                    "stems": rows[:2]})

    # crest or section flag
    flags = {f["kind"]: f for f in rep.get("flags_structured") or []}
    if (flags.get("crest") or {}).get("short") or (flags.get("section") or {}).get("short"):
        rows = [{"name": nm, "crest_db": meter.crest_db(s, fs)} for nm, s in zip(names, stems)]
        rows.sort(key=lambda r: r["crest_db"] if math.isfinite(r["crest_db"]) else math.inf)
        share = []
        for sec in rep.get("sections") or []:
            a, b = int(sec["start"] * fs), int(sec["end"] * fs)
            # projection share of the sum's energy (sums to 1, negative where a stem cancels)
            tot = float(np.sum(total[a:b] ** 2)) if b > a else 0.0
            e = [float(np.sum(s[a:b] * total[a:b])) / tot if tot > 0 else 0.0 for s in stems]
            share.append({"label": sec.get("label"), "start": sec["start"],
                          "share": dict(zip(names, e))})
        out.append({"kind": "crest", "limit": "crest low or section contrast lost",
                    "detail": {"section_share": share}, "stems": rows})

    # noise exposed or present
    nz = rep.get("noise") or {}
    if nz.get("status") in ("exposed", "present"):
        rows = [{"name": nm, "floor_db": float(noise.floor(s, fs)["db"])}
                for nm, s in zip(names, stems)]
        rows.sort(key=lambda r: -r["floor_db"] if math.isfinite(r["floor_db"]) else math.inf)
        out.append({"kind": "noise", "limit": f"noise {nz['status']}",
                    "detail": {"after_db": nz.get("after_db")}, "stems": rows[:2]})
    return {"stems": names, "attribution": out}


# ---------------------------------------------------------------- pass score

LIMIT_WEIGHT = 0.5             # score per real limit, in dB of RMS shortfall
_LIMIT_PREFIXES = ("input EQ ", "multiband band ", "frequency-dependent limiter ",
                   "clipper short", "limiter on its outlier cap",
                   "limiter held by its average budget", "loudness ")
_LIMIT_KINDS = {"eq", "multiband", "speclim", "peak", "budget"}


def notch_rejected(nt: dict) -> str | None:
    """Why the planner itself held this found notch back (module contract), or None when a
    stem fix may take it: it ran out of room ("below the top N ..."), or the notch stage
    could not take an S-only resonance (stemfix can, on the stem's S)."""
    reason = str(nt.get("reason") or "")
    if reason.startswith(("suspect", "within", "no cut")):
        return reason
    mc = nt.get("max_cut_db")
    if isinstance(mc, (int, float)) and math.isfinite(mc) and abs(mc) <= 0.0:
        return "the planner's cap is 0 dB" + (" (musical)" if nt.get("flag") == "musical" else "")
    return None


def is_limit(e: dict) -> bool:
    kind = e.get("kind")
    if kind is not None:
        if kind == "notch":
            return bool((e.get("detail") or {}).get("actionable"))
        return kind in _LIMIT_KINDS
    lim = e.get("limit") or ""
    if lim.startswith("notch "):
        return lim.endswith("(found)")
    return lim.startswith(_LIMIT_PREFIXES)


def pass_score(report: dict, attribution: list) -> dict:
    ld = report.get("loudness") or {}
    tgt, got = ld.get("target_rms_dbfs"), ld.get("after_rms_dbfs")
    short = (max(0.0, float(tgt) - float(got))
             if tgt is not None and got is not None and math.isfinite(tgt) and math.isfinite(got)
             else 0.0)
    limits = [e["limit"] for e in attribution if is_limit(e)]
    lw = LIMIT_WEIGHT * len(limits)
    cause = next((w.get("cause") for w in report.get("warnings") or []
                  if isinstance(w, dict) and w.get("stage") == "loudness"), None)
    return {"score": short + lw, "rms_shortfall_db": short, "limits": limits,
            "dominant": ("rms_shortfall" if short > lw else "limits") if short + lw > 0 else None,
            "shortfall_cause": cause}


# ---------------------------------------------------------------- report

def _f(v, nd=1):
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def _stem_line(e: dict, r: dict) -> str:
    if "peak_drop_db" in r:
        d = r["peak_drop_db"]
        how = (f"removing it lowers the sum's peak by {_f(d)} dB" if d is None or d >= 0 else
               f"it cancels there (removing it raises the sum's peak by {_f(-d)} dB)")
        return f"{r['name']}: peak {_f(r.get('peak_db'))} dBFS, {how}"
    if "below_sum_db" in r:
        return f"{r['name']}: peak {_f(r.get('peak_db'))} dBFS, {_f(r.get('below_sum_db'))} dB below the sum's"
    if "share" in r:
        if r["share"] < 0:
            return (f"{r['name']}: cancels against the rest there "
                    f"({100 * r['share']:.0f} % share of the sum)")
        return f"{r['name']}: {100 * r['share']:.0f} % share of the sum there"
    if "excess_db" in r:
        return f"{r['name']}: {_f(r['excess_db'])} dB above its neighbours at f0"
    if "crest_db" in r:
        return f"{r['name']}: crest {_f(r['crest_db'])} dB"
    if "floor_db" in r:
        return f"{r['name']}: floor {_f(r['floor_db'])} dB (BS.468 weighted)"
    return r["name"]


def write_report(work, rep: dict) -> None:
    from eluate import evaluate
    work = Path(work)
    rep = evaluate._clean(rep)
    io.write_json(work / "stems-report.json", rep)
    L = ["# Stems report", "", "Stems: " + ", ".join(rep["stems"]) + ".", ""]
    nt = rep.get("null_test")
    if nt is not None:
        L += ["## Null test", "", null_message(nt), ""]
    L += ["## Limits the sum hit, and the stems behind them" if not rep.get("kept") else
          f"## Limits the kept pass ({rep['kept']}) hit, and the stems behind them", ""]
    if not rep["attribution"]:
        L.append("None: the sum met its targets within the stems profile's ranges.")
    for e in rep["attribution"]:
        L.append(f"- {e['limit']}: " + ("; ".join(_stem_line(e, r) for r in e["stems"])
                                       or "no stem stands out") + ".")
    passes = rep.get("passes")
    if passes:
        L += ["", "## Passes", "",
              f"Score: the RMS shortfall to the target in dB, plus {LIMIT_WEIGHT:g} per limit "
              "hit (lower is better).", ""]
        for p in passes:
            kept = " (kept)" if p["index"] == rep.get("kept") else ""
            what = "the plain sum" if p["index"] == 0 else "corrected stems"
            cause = f", {p['shortfall_cause']}" if p.get("shortfall_cause") else ""
            dom = {"rms_shortfall": "; the shortfall dominates the score",
                   "limits": "; the limits dominate the score"}.get(p.get("dominant") or "", "")
            L.append(f"- Pass {p['index']}{kept}, {what}: score {_f(p['score'], 2)} "
                     f"(RMS {_f(p['rms_shortfall_db'], 2)} dB short{cause}, "
                     f"{len(p['limits'])} limit{'s' if len(p['limits']) != 1 else ''}"
                     + (": " + "; ".join(p["limits"]) if p["limits"] else "") + dom + ").")
            for line in p.get("fix_lines") or []:
                L.append(f"  - {line}")
        if rep.get("stopped"):
            L += ["", rep["stopped"]]
    io.write_text(work / "stems-report.md", "\n".join(L) + "\n")
