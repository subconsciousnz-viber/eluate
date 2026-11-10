# Offline Auto-Mastering Engine: orchestration spec (draft 0.5)

Companion to `mastering-engine-brief.md`. The brief is the intent; this is the
work breakdown, module contracts and gates so the work can be split across
sessions/agents. Draft 0.2 folds in the maintainer's decisions of 2026-10-17 and the
accepted findings from `audit/01-research-review.md` (review of
`research/report.md`). Items marked **[Q]** are still open.

> **Status 2026-11-10: historical draft.** This spec is the 2026-10-17 work breakdown and is kept
> for its reasoning and audit trail; it is **not** the current contract. Where it disagrees with
> the code, the code, its module docstrings and `docs/gain-staging.md` (cited as GS §n) win. Known
> contradictions, with what supersedes each:
>
> 1. **Loudness target** (3.6, 3.7, 9): S95 is retired as a target. The constraint is the RMS
>    target +-0.3 dB on the 300 ms AES17 meter (`Targets.rms_dbfs`, default
>    `planner.DEFAULT_TARGET_RMS_DBFS` = -4.1); S95 is still reported. GS §17, §18; retirement
>    list in GS §14.2.
> 2. **True-peak ceiling** (3.6): the default is -0.3 dBTP on the 8x meter, 24-bit output;
>    -1 dBTP with the worst-of-4x/8x/32x meter applies only under `--streaming`. GS §3
>    "Ceiling", §11 "Ceiling and export".
> 3. **Solver method** (3.6 "Method", 1 "solver loop"): the lever classes, coordinate descent
>    and nearest-feasible-point reporting are retired; the solver is a fixed sequence of planned,
>    true-peak, slope and loudness renders (`solver.py`). GS §14.2 "Retired".
> 4. **CLI** (4): `--target`, `--genre` and `--harmonic` no longer exist (abbreviations are off,
>    so `--target` errors). Section 4 below is corrected; `eluate <command> --help` is
>    authoritative. GS §14.2 "CLI".
> 5. **`tone` stage** (3.5 item 4, 9): the planner leaves tone off; the spectral move is the
>    input EQ (`stages/inputeq.py`). The chain (`stages/__init__.py`) is input gain, inputeq,
>    notch, sidestrip, multiband, tone, saturate, glue, speclim (a ceiling-only multiband),
>    keyeq, limiter, output; this list omits input, inputeq, speclim and output. GS §10, §12,
>    §18, §19.
> 6. **`saturate`** (3.5 item 5): drive is capped at 2 dB (`planner.SAT_MAX_DRIVE_DB`) and set
>    from the band deficit; the harmonic-vs-linear blend is retired. GS §6, §14.2.
> 7. **`glue`** (3.5 item 6): default `glue_mode="serial"`, FET-style, 2:1, 0.4 ms attack
>    (`planner.GLUE_ATTACK_MS`); parallel is an option. GS §13, §18.
> 8. **Crossovers** (9): default 100 / 400 / 1600 / 6400 Hz, and without `--xover` they are
>    key-locked to the track's tonic. Corrected in 9 below. GS §4, §11.
> 9. **Licence** (8, 9): decided, AGPL-3.0-or-later (`LICENSE`, `pyproject.toml`). Corrected
>    in 8 below.
> 10. **Pipeline and repository layout** (1, 2): no `offline-master/` root, no
>    `docs/architecture.md`, and many modules are missing from the list. Section 2 is
>    corrected; the environment table in 0 is also stale (the pinned set is Python 3.12 and
>    `requirements/lock-*.txt`, see `requirements/README.md`).
>
> References to `audit/NN-*.md` below point to the maintainer's internal review logs, which
> are not part of the public tree; they are kept as the historical record.

## 0. Environment (verified 2026-10-17)

| Item | State |
|---|---|
| `agy` CLI | present, `%USERPROFILE%\.local\bin\agy.exe`; non-interactive via `agy -p "<prompt>" --model gemini-4-argon-high --output-format json` |
| Auditor model | `gemini-4-argon-high` (also `-medium`, `-low`) |
| Python | 3.14 default (numba broken against numpy 2.5); **3.12 available via `py -3.12`**, use a 3.12 venv |
| Installed (3.14) | numpy 2.5.3, scipy 1.17.1, librosa 0.10.2, soundfile 0.13.1, matplotlib 3.10.8, pyloudnorm |
| Missing | torch, demucs (optional extra only), uv |

## 1. Pipeline shape

```
audio.wav --> analyse --> features.npz + analysis.json
                              |
   user controls -------------+--> plan --> plan.json   (every envelope + scalar the renderer needs)
                              |              |
                              +--> render <--+ --> master.wav + per-stage renders
                                      |
                                   measure --> report.json / report.md / plots
                                      |
                                   solver loop: adjust plan.json until constraints met or infeasible
```

Design rules that make this orchestratable:

- **Every stage is a pure function of (audio, plan).** No hidden state, no
  parameter modulation at render time: the plan holds precomputed gain
  envelopes and scalars.
- **Control rates.** Notch, multiband and tone envelopes are stored at 1 ms and
  interpolated to audio rate with PCHIP (shape-preserving, so caps that hold at the
  control points hold between them; audit 05 D.2). The limiter envelope is never
  stored: `limiter.py` computes and applies it at the **8x** oversampled audio rate
  from scalars in the plan (audit 02 A11, 05 D.3).
- **`plan.json` is the product**, but envelopes live in a `plan.npz` sidecar
  (float32 arrays keyed by stage and band) and the JSON holds scalars plus the
  sidecar's key names; 1 ms envelopes as JSON text run to 120 MB per track
  (audit 05 D.13). The plan records the numpy, scipy and package versions it was made with, and a
  render from a saved plan is bit-exact reproducible **in that environment**; render warns on a
  version mismatch (audit 05 Q11). Since 2026-10-22 it also records the platform, machine,
  Python and BLAS / LAPACK vendors; a difference there only logs "rendered on a different
  platform" (macOS report 3.4). Per-stage outputs are optional, via callback (audit 05 Q15).
- **Whole-track 8x arrays are not acceptable for real tracks** (13 GB for five
  minutes, audit 05 D.12). The limiter runs in blocks with carried hull state;
  the box smoother is local, so this is bookkeeping, not a design change.
  Workstream E owns it.
- **Analysis is cached** (`features.npz`), so the solver loop re-renders but
  never re-analyses. Stages 1-2 (notch, sidestrip) depend only on analysis and
  are rendered once and cached before the solver loop.
- **Everything is measured in M, S and stereo** via one shared band split and
  one shared metering module, used identically by analysis, measure and tests.

## 2. Repository layout

As of 2026-11-10 (the draft's original layout, rooted at `offline-master/` with a
`docs/architecture.md` that was never written, is superseded):

```
(repository root)
  mastering-engine-brief.md     # the intent
  research/                     # Phase 1 research report
  docs/                         # spec-draft.md (this file), gain-staging.md, dynamics-macros.md,
                                # noise-and-stems.md, batch-mode.md, UI-brief.md, release notes
  src/eluate/
    io.py          # read/write (soundfile), LF-only text writes, 32f internal
    bands.py       # zero-phase complementary band split, minimum-phase option, M/S conversion
    meter.py       # BS.1770 K-weighting, momentary/short-term/integrated, true peak, RMS meter, PLR/PSR
    oa.py          # streamed overlap-add convolution, bit-identical to scipy's oaconvolve
    envelope.py    # gain-envelope construction: required gain, min-filter + box smoothing, banded QP variant
    features/      # analysis pass: loudness crest stereo transients tilt resonance structure pitch borrowed
    planner.py     # analysis -> plan: targets, crossovers, every stage's design
    plan.py        # dataclasses + JSON schema for plan.json (+ plan.npz sidecar)
    stages/        # render stages, each apply(x, fs, section) -> y, run in a fixed order:
      gain.py  inputeq.py  notch.py  sidestrip.py  multiband.py  tone.py  saturate.py  glue.py
      keyeq.py  limiter.py  resample.py  output.py
    solver.py      # fixed render sequence: planned, true peak, slope, loudness
    evaluate.py    # before/after report, A/B loudness-matched renders, difference render
    stems.py  stemfix.py  noise.py          # stem-set mastering, stem corrections, noise-floor detection
    memory.py  estimate.py  parallel.py     # memory guard, time-left estimate, engine threads
    processlog.py  app_engine.py  uidata.py # plain-language process log, desktop engine process, UI numbers
    cli.py         # eluate analyse|plan|render|master|report
  tests/           # synthetic-signal tests per module (section 7); `slow` marker for long ones
  fixtures/        # synthetic signals; EBU vectors and private premasters are gitignored
  requirements/    # exact lock files per platform
  ui/  desktop/    # the desktop shell (HTML UI, Tauri)
  tools/           # notices and analysis helpers
  pyproject.toml   # python>=3.12,<3.14; extras: [dev], [demucs]
```

Package name `eluate` (maintainer, 2026-10-21; it was `offmaster` until then).

## 3. Module contracts

### 3.1 bands.py
- `split(x, fcs, order, phase)`: cascaded upward from the lowest crossover.
  `zero` mode: `LP_zp = sosfiltfilt(butter(order, fc))`, `HP = x - LP_zp`. Sum of bands equals `x`
  to float precision. `minimum` mode: LR4 cascade, allpass-flat magnitude sum.
- Zero-phase Butterworth under filtfilt has magnitude `|H|^2`: -6 dB at fc, and the complementary
  HP `1 - |H_lp|^2` is exactly the squared Butterworth high-pass (audit F2).
- Zero-phase splits pre-ring symmetrically once bands get different gains (audit F1). Default is
  `zero` for all crossovers except the lowest, which is decided by listening in gate D with
  `minimum` as the expected winner. **Decided 2026-11-10** (filter-maths audit F1 and the final audit,
  docs/gain-staging.md 21): zero phase stays the default (the maintainer's blind-test pick; the LR4
  allpass raises the peaks of clipped, low-crest material by 1.6 to 5.8 dB), and
  `--xover-phase minimum` makes the lowest crossover of the multiband and the frequency-dependent
  limiter `minimum` (recorded in the plan); the zero-phase
  100 Hz split pre-rang a kick by -14.9 dB re its peak 6 ms early at -6 dB of sub GR, the LR4 puts
  nothing before the onset; the band sum is the LR4 allpass (flat magnitude, 5.3 / 4.5 ms of group
  delay at 50 / 100 Hz, none above 1 kHz). Saturate's split stays zero phase; saved plans keep
  their recorded phase. `split_iter` streams every mode (one band alive at a time).
- Zero-phase passes run blocked with a subnormal flush (`iir.py`, 2026-11-10): bit-identical to
  `sosfiltfilt` without digital silence, within 1e-12 with it.
- Edge handling: **zero padding** of `max(512, ceil(20 fs / (2 pi fc_min)))` samples (capped at
  len(x) - 1 with a warning), with odd reflection available as `edge="odd"` (audit 02 D.9). Real
  tracks start from digital silence or a fade, and silence is the true continuation; odd reflection
  of a fade is a mirrored ramp. No padding can continue arbitrary curvature: a track that starts
  abruptly at level pre-rings under zero-phase filtering and that is reported, not suppressed.
- **Crossover defaults: 100 / 500 / 2000 / 8000 Hz, giving 5 bands** (sub, low, mid, high-mid, air).
  The brief said "4-band"; the maintainer's stated crossovers imply five. Band count is configurable; five
  is the default. **[Q] confirm five bands is intended.**
- `to_ms`, `to_lr` with the 1/sqrt(2) convention stated once and used everywhere.

### 3.2 meter.py
- Own BS.1770-4 implementation with K-weighting biquads derived per sample rate by pre-warped
  bilinear transform (audit F21), 400 ms blocks at 75 % overlap, -70 LUFS absolute + -10 LU relative
  gating; `pyloudnorm` as a cross-check in tests.
- True peak: the **8x** windowed-sinc meter is the one the solver, limiter and report constrain
  against; the 4x BS.1770-4 Annex 2 FIR reading is reported alongside for compliance. The two differ
  by up to about 0.2 dB on steady tones and more on transients (audit 02 F4), so constraining at 4x
  would miss the ceiling at report time. Native-rate K-weighting (pre-warped bilinear) is the
  convention, with the RLB high-pass numerator scaled by a0(48 kHz)/a0(fs) so the stage's passband
  gain matches the 48 kHz calibration at every rate; unnormalised, it drifts by 0.02 dB at 96 kHz
  (audit 02 A5). The K-filter starts from the steady state of the first sample so DC or an abrupt
  start does not read as a loud block (audit 02 A16).
- `loud_passage_lufs(x)`: two variants reported, one used as control:
  - **S95**: 95th percentile of short-term (3 s) loudness. Control target.
  - **Loud-section integrated**: BS.1770 integrated over windows with `S >= S_max - 3 LU`. Reported.
- PLR = true peak - integrated; PSR = true peak - short-term, as a time series plus 95th percentile.
- **Default S95 targets by genre preset** (maintainer's figures, applied as S95):

| Preset | Default S95 | Range |
|---|---|---|
| downtempo | -9 LUFS | -12 to -7 |
| techno / trance / house | -6.5 LUFS | -8 to -5 |
| festival EDM / bass / dubstep / dnb | -5 LUFS | -6 to -4 |

S95 sits roughly 0.5 to 2 LU above integrated on dynamic arrangements, so these land slightly
quieter integrated than the same numbers read as integrated targets. Evaluate on fixtures.

### 3.2a User controls (addendum)
- **BPM (optional).** The maintainer's point: users know their tempo. `--bpm` overrides detection, seeds
  the beat grid and the bar snap; the analysis then reports the detected pulse's ratio to it.
  Detection stays for batch use and as the fallback, and always returns a metrical hypothesis
  (chosen pulse plus ranked alternatives with their ratios, such as 4/3 for triplet or shuffle feel).
  Ground truth for the five maintainer premasters: Track D 120 (half-time to full-time at about 1:52
  and back at about 4:46), Track E 95, Track B 120, Track A 94, Track C 88.

### 3.3 features/
| Module | Output (time series at a common hop, e.g. 100 ms, plus summaries) | Method | Risk |
|---|---|---|---|
| loudness | M (400 ms), S (3 s), I, LRA; per section | meter.py | low |
| crest | per band x {M,S,stereo}: peak/RMS over 400 ms, dB; gated to frames with band RMS > -60 dBFS; summary = 95th percentile over interior frames (0.25 s inside both ends), or over all of that band and view's gated frames when none is interior, the rule of stages/multiband.band_crest (audit features F14) | bands.py | low |
| stereo | per band: L/R Pearson correlation and S/M energy ratio (dB); decoherence scalar from correlation, with a finite neutral default and a not-measurable flag for hard-panned or silent bands (audit 04 A12), hard-panned including a quieter channel more than 40 dB under the louder (bleed: correlation is scale-invariant, audit features F10); side-strip suggestion = default 100 Hz when no low-side energy exceeds the ratio, otherwise just above the offending region, gated relative to the LTAS maximum (A13) | bands.py | low |
| transients | onset strength, onset times, onsets/s, kick onsets with a score (not a boolean) | SuperFlux onsets (borrowed, log filterbank, local threshold floor); kick = low-band flux rise (gate 1.5 x the 90th percentile of the flux maxima, clipped to 6 to 10 dB, audit features F5) plus either a downward frequency sweep in the first 30 ms or a click component above 500 Hz, so low-passed stationary bass attacks are not kicks (audit 04 A5); refinement on the low band; sample-count constants scaled with fs (A10), the attack window and onset refinement as fixed durations (features F17) | low-med |
| tilt | LTAS as **mean PSD per 1/3-octave band** (not band-summed power, audit F5), 20 Hz to 16 kHz, for M/S/stereo; OLS slope of dB vs log2(f) over 100 Hz to 10 kHz; sub shelf (30 to 90 Hz) reported separately; peak-hold LTAS | Welch N=8192 | low |
| resonance | list of {f0, Q +- uncertainty, start, end, excess_dB(t), channel in {M, S, both}} | multi-resolution STFT (N = 16384 below 500 Hz, 4096 to 2.5 kHz, 2048 above) with parabolic sub-bin interpolation; baseline = flank median (audit 04 brief D.2); excess > 6 dB, Q > 5, persistence > 30 % of a section; tolerance max(2 %, 1.5 bins); **sub-bass exemption below 80 Hz, and 80 to 150 Hz only with persistence >= 50 % and half the depth cap** (audit 04 F6); harmonic-ladder check with the share reported, tuned on detuned and inharmonic fixtures too (F9); a short smoothing pass below 500 Hz for detection (F7), blind where its smoothing reaches a mostly zero-padded frame at the file's ends (features F13); **run on M and S separately** as well as stereo, channel recorded so the notch stage cuts only where the resonance is (a +9 dB ring in S alone is invisible in the stereo power spectrum at S/M -10 dB, audit 04 A4) | medium-high |
| structure | tempo, downbeats, bar grid (4/4 declared; meter estimation is future work); segments {start, end, label}; `grid_informative` with `grid_reason` and `grid_evidence`: informative only with at least 8 beats, 8 onsets, an onset within 40 ms at half the beats and the onset strength at the beats 1.8 x its mean (a user tempo always; librosa returns beats on noise and pads, audit features F1 / F16) | librosa beat track with an **octave check against the kick-onset rate** (a half-tempo grid halves bass-root accuracy and is invisible to agreement-based confidence, audit 04 F2/F11); a detected constant grid refitted to the onsets and shifted onto them (all onsets, not only kicks: features F6 / F16); Foote novelty on MFCC+chroma+loudness SSM; low-band dropout detector for pre-drop fills; snap boundaries to 4/8/16-bar grid; heuristic labels. Labels are for reporting and preserve-mode deltas only, never the loudness target | medium |
| pitch | global key (tonic, mode, both profiles' scores) and `key_confidence` in [0, 1] (None without chroma); chord chroma per bar; bass root per beat with a confidence that is documented as what it measures, and a glide flag | key: CQT chroma; **mode from Faraldo edma, tonic from edmm when the modes agree** (edmm's major profile is flat, so on its own it can only ever report minor, audit 04 F1). When edma says major and edmm's minor tonic and its relative major both score within 0.3 of edma's best, the bass decides: the bass-root histogram when it favours one by 1.1 x, else the first stable bass root (features F3). key_confidence = ramp(edma best score, 0.3 to 0.8) x ramp(its margin over the best other tonic, 0 to 0.15) x (0.4 when edmm's tonic differs) x ramp(chroma max / mean, 1.3 to 2.0); the key lock keeps the fixed crossovers below about 0.5 (features F2). Bass root: LPF < 250 Hz, subharmonic summation plus pYIN, run on L, R and M separately and confidence-weighted (never on sqrt(L^2+R^2), audit 01 F10), frames weighted toward the second half of each beat, kick mask applied only above a stricter kick score (F4); **keyeq eligibility requires per-beat agreement with the independent NSDF estimate**, not self-agreement (F2); Demucs bass stem as optional extra | key low-med; bass root high |

### 3.4 envelope.py
A gain envelope `g(t) in (0,1]` is built as (audit F7):
1. `g_req(t)` = required gain per sample from the detector (e.g. `min(1, T / env(t))`).
2. Lookahead: running minimum over window `w` (`scipy.ndimage.minimum_filter1d`).
3. **Primary smoothing: moving average of length w applied twice** (triangular kernel, C1). With the
   minimum window and average window matched, `g <= g_req` holds by construction; no iteration.
4. Asymmetry: attack window and release window are different lengths; the release side uses a longer
   average (or a one-sided exponential-hull followed by the same smoothing).
5. **Optimal variant: banded QP** minimising `||D2 g||^2 + mu ||g - u||^2` subject to `g <= u`,
   `u = min(g_req, 1)`, with `mu` a documented parameter (default length^-4). The fidelity term is
   required; without it the problem is unbounded below (audit 02 D.4). The release hull from step 4
   is applied to `u` before the QP so the QP does not reduce gain earlier than the lookahead (audit
   02 F11). Solved by a banded interior-point method, O(n). Compared against 3-4 in gate D.
6. The smooth-then-project iteration from draft 0.1 is dropped.
7. Rates: multiband/notch/tone at 1 ms with cubic interpolation to audio rate; limiter at 8x audio rate.
8. Hold (2026-11-10, `box_smooth(..., hold)`): the running minimum reaches `hold` samples further
   back, `[-(L-1)-hold, L-1]`, then the same two averages. The kernel support stays inside every
   minimum window, so `g <= g_req` still holds by construction; reduction never starts earlier;
   an isolated dip recovers exactly `hold` samples later. The final limiter holds 25 ms (one 40 Hz
   period) so it does not ripple inside a sub cycle; hold 0 is the original, bit for bit.

### 3.5 stages/ (order accepted by maintainer)
1. `notch`: per confirmed resonance, `y = x - d(t) * BP_zp(x; f0, Q)`, `BP_zp` fixed-frequency
   zero-phase bandpass, `d(t) = (1 - m_kick(t)) * (1 - 10^(G_dB(t)/20))`, cap -6 dB (d <= 0.4988).
   **Q is defined on the notch response**: f0 over the bandwidth at half the dB cut (cookbook
   convention), solved through the bilinear warp (audit 05 A5). Kick mask holds depth at zero for
   +-20 ms around each onset at audio rate, padded by one control period on the 1 ms grid (audit 05
   A8). **Merge candidates whose -3 dB bands overlap**, and clamp the composite cut at any frequency
   to the cap (audit 05 A6). Q >= 1. The bands are the realised (bilinear-warped) half-cut edges
   since 2026-11-10 (the analog edges were 5 to 17 % off above fs / 4 at Q 2); the bandpass pad
   follows the slowest pole, real or complex.
2. `sidestrip`: zero-phase HP on S only below `fc_sub` (default 100 Hz; analysis suggestion = lowest
   frequency where S/M ratio exceeds -20 dB, bounded 60 to 150 Hz). Never mono-sums.
3. `multiband`: N-band M/S (default 5). Per band x {M,S}: a **compressor envelope** (threshold,
   ratio, attack and release as window lengths, from 3.4, max GR 3 dB hard cap) multiplied by a
   **per-band limiter envelope** (ceiling, short lookahead, from 3.4). Threshold shapes the band's
   dynamics; the ceiling catches its peaks, as in iZotope's multiband dynamics (maintainer ruling
   2026-10-17). Per-band crest is steered here, not at the final stage. Detector (2026-11-10): the
   peak over the trailing window of one period of the band's lower edge, floored at 2 ms (25, 10,
   3, 2, 2 ms on the default crossovers), so the low bands do not ripple inside a cycle; band
   ceilings release 100 / 80 / 50 / 50 / 50 ms; ms_link takes its minimum on the control grid.
   S in the sub and low bands: makeup 0. S in the high-mid and air bands: makeup +2 dB scaled by
   `(1 - decoherence)`, bounded to [0, +3] dB; M makeup bounded to +-3 dB at validation (audit 05
   Q6). Band targets come from `tone`.
4. `tone`: **one smooth zero-phase FIR** designed by frequency sampling from the planner's target
   curve (log-spaced nodes from 30 Hz to 16 kHz, smoothed, the +-3 dB bound applied to the curve
   itself), moving the LTAS toward `reference_slope + user_tilt`; disabled under preserve-mix. Not a
   filterbank: an order-2 octave filterbank reaches only two thirds of a single band's requested
   gain (audit 05 A11). Reference slope = -4.5 dB/oct PSD over 100 Hz to 10 kHz with the sub shelf
   below 100 Hz as its own target (audit 01 F4, F5). The gate is on the measured response.
5. `saturate`: per band, 4x oversampled soft clip with unity small-signal gain (compensation is
   for the curve's slope, **not** RMS-equal: an RMS-normalised saturator can never add band energy,
   audit 05 A12). The energy it adds is measured by evaluate.py and the solver closes the loop. A
   `symmetry` parameter in [0, 1] (default 0.3) adds an even-harmonic component so fundamentals up to
   fs/4 produce in-band H2; pure tanh is odd and gives the air band nothing below Nyquist. Sub band
   drive 0; air band drive 0 by default and 8x oversampled when driven; drive capped at 12 dB
   (audit 05 Q5); high-mid S band gets width drive. The knee is scaled to a
   robust band level (gated RMS x sqrt 2), never the whole-track peak (audit 05 A13).
   **Harmonic-vs-linear blend (maintainer decision):** `harmonic` in [0, 1] sets how much of each band's
   deficit is met by saturation versus by `tone`. **Planner rule (audit 07 A8):** tone always carries
   the full linear deficit; saturation with unity small-signal slope removes band energy as drive
   rises (measured -1.65 dB at 6 dB drive, -6.9 dB at 12 dB), so it is a timbre choice, never a level
   lever; default drive cap 6 dB until the drive-to-energy map is calibrated; the solver's slope pass
   absorbs its level effect. Default per band from analysis: harmonically dense
   bands lean linear; clean, sine-like bands lean harmonic. Eventually a UI toggle.
6. `glue`: a single-band **parallel** compressor after saturate and before the clipper: a
   **lagging attack** (causal one-sided attack window or one-pole over the past, default 30 ms; the
   gain never reduces before the onset), the release hull from 3.4 (default 300 ms, optional
   program-dependent), gentle ratio, soft knee, wet/dry mix, explicit makeup set by the solver
   (default 0), optional "vintage" even-harmonic curve on the wet path. Lookahead smoothing is
   wrong here: it ducks before each hit and only auto makeup gives the transient back, which is
   upward compression, not a bus compressor (audit 06 F1, F2). Gate: on a burst above threshold,
   wet-path gain reduction starts after the onset and reaches 63 % of its depth at about the attack
   time; at mix 0.5 the transient peak is within 0.5 dB of the input; the old "sustain rises" gate is
   dropped because the fixture's sustain sat below threshold and the gate measured makeup. Stored at
   1 ms. This is the second crest lever after the multiband stage (maintainer ruling 2026-10-17).
   **Planner rule (audit 07 A9):** threshold = p95 of the 400 ms peak level of the signal entering
   the glue stage minus 6 dB, never a sum of loudness and PSR percentiles.
7. `keyeq` (experimental, off by default): dynamic notch whose `f0` follows harmonics of the tracked
   bass root via a crossfaded bank of static zero-phase bandpasses (30 ms raised-cosine), never a
   swept IIR. Enabled only when pitch confidence >= 0.75.
8. `limiter`: onset-masked oversampled soft clipper (8 to 12 ms around kick onsets, 0 to 3 dB,
   peak measured over the whole active window, overlapping windows merged, knee allowed to start up
   to 1 dB x (depth / 3 dB) below the ceiling so small overs still get a soft knee rather than a hard
   clip, audit 05 D.20), **then band-limited to base-rate Nyquist at 8x before the detector** (the clipper's
   harmonics above Nyquist otherwise vanish in decimation and the fundamental crest rebounds unseen,
   audit 05 A2), then the true-peak gain-envelope limiter at 8x, decimate, then a 32x FFT true-peak
   check. Any remaining excess is removed by a **local re-limit** on the decimated output, never a
   global trim; the trim against max(8x, 32x) readings is the last resort and is reported. The
   limiter's up/down resampler is a **processing** design with passband >= 0.98 Nyquist, separate
   from the metering prototype: reusing the meter's filter removed 22.5 to 24 kHz at zero gain
   reduction (audit 05 Q1). Gate: a below-ceiling sweep passes within a documented bound. Limiting at 4x and measuring at 8x overshoots a -1 dBTP ceiling by about 0.4 dB
   on hot high-frequency material (audit 02 A11).

### 3.6 solver.py
Constraints (all measurable with meter/features):
- `S95 == target +- 0.3 LU`
- fitted slope `== reference + tilt +- 0.3 dB/oct` (reshape mode), or LTAS within +-1 dB per band of
  the original shape (preserve mode)
- per-band crest loss <= 1.5 dB sub, 2 dB low, 3 dB mid / high-mid / air (maintainer accepted).
  **Interpretation (maintainer ruling 2026-10-17):** crest will change on almost every track; the
  constraint bounds the loss per band relative to the premaster, it does not preserve crest. The
  levers, in order: multiband per-band attack and release, per-band ceiling, glue mix and release,
  clip depth, input gain. The final limiter is broadband and is not where per-band crest is steered.
- per-band **compressor** GR <= 3 dB (the per-band ceiling's reduction is separate, uncapped,
  reported, and bounded indirectly by the crest constraint; audit 06 F3)
- per-band peak and crest are measured by **re-splitting the rendered output**, which reads up to
  0.65 dB above the ceiling the stage enforced on the band as processed; the solver sets ceilings
  iteratively against the re-split measurement (audit 06 F6). A band that falls below the crest
  gate after processing is a violation, never "nothing to lose" (F5).
- true peak <= -1 dBTP on the **worst of the 4x (BS.1770 Annex 2), 8x and 32x readings**
  (`tp_meter="worst"`, default; `"8x"` is available). Constraining on 8x alone left all five real
  masters at -0.81 to -0.86 dBTP on the 4x delivery meter (audit 07 A14). Checked after the float32
  cast.
- **Crest constraint mode** (audit 07 F1/F17, maintainer ruling pending): `crest_mode` in {"loss",
  "report", "floor"}. "loss" is the per-band loss limit above; "report" prints per-band loss and
  fails nothing; "floor" takes absolute per-band crest floors on the master, to be calibrated from
  the maintainer's masters. **Ruling 2026-10-18 (architect, with the maintainer's agreement that crest factor
  was Claude's suggestion and not worth further iteration): default is "report".** At -6.5 LUFS
  from -17 LUFS premasters the loss limits are unreachable by construction on four of five real
  tracks. The band-0 side-view exemption after the side-strip is an
  explicit option, off by default.
- preserve mode: section loudness deltas within +-1 LU of the original deltas

Method: cache stages 1-2; correct the slope first through the tone rotation (the only lever that
moves it); the lever classes below run only while a constraint other than S95, slope and true peak
is violated; ordered coordinate descent exploiting monotonicity (limiter threshold vs
loudness; makeup vs band level; per-band attack and release, separately, and ceiling vs that band's
crest loss; glue mix vs broadband crest; clipper depth vs kick crest). Levers move in **both**
directions: relaxing upstream compression, and tightening it (lower ceilings on the bands that drive
the broadband limiter, deeper onset clipping) so the final limiter works less (audit 07 A4/F2). One
lever class per render; a reverted move is judged against the excess before it was applied, bracket + Brent per variable, outer
loop <= 8 renders, then one proportional cross-coupling refinement pass. On infeasibility, report the
binding constraint and the nearest feasible point. Differentiable DSP is not used (audit F16).

### 3.7 evaluate.py
Before/after: S95, per-section loudness, PLR/PSR, per-band crest delta, per-band correlation delta,
fitted tilt vs target, true peak (8x), notch list with time ranges. Outputs `report.json`,
`report.md`, PNG plots (LTAS M/S/stereo, loudness timeline with sections, GR envelopes). A/B:
`ab/original_matched.wav` and `ab/master_matched.wav` matched on S95 by attenuating the louder one;
plus `ab/difference.wav`. When the maintainer's own master of the same mix is supplied, report
per-feature distance to it.

## 4. CLI

As built (2026-11-10; `eluate <command> --help` is authoritative, and the draft's `--target`,
`--genre` and `--harmonic` are gone):

```
eluate analyse  in.wav  -o work/ [--xover 100,400,1600,6400] [--bpm N]
eluate plan     work/   [controls]
eluate render   work/plan.json -o master.wav [--input in.wav]
eluate master   in.wav -o out/ [--stems a.wav b.wav ...] [--mix M] [controls]   # analyse + plan + solve + render + report + A/B
eluate report   out/    [--master-ref ref.wav]

controls: --tilt DB/OCT  --target-rms DBFS  --preserve  --xover F1,F2,F3,F4  --subside HZ
          --profile {electronic,live}  --streaming  --character {auto,transparent,slow,moderate,fast,aggressive}
          --makeup {peak,rms,auto}  --glue-mode {serial,parallel}  --[no-]kick-detective  --[no-]speclim
          --crest-mode {loss,report,floor}  --crest-floor  --side-band0-exempt  --tp-meter {worst,8x}
          --threads N
```

## 5. Workstreams (parallelisable)

| WS | Scope | Depends on | State |
|---|---|---|---|
| A | `research/report.md` + audit 01 | none | done (Gemini), reviewed |
| B | `io`, `bands`, `meter`, `envelope`, synthetic fixtures, tests | none | done, audit 02 closed |
| C | `features/*` | B | done, audit 04 closed, real-material pass closed (kick recall open) |
| D | `stages/*` + listening fixtures | B | done, audit 05 closed |
| D2 | per-band ceilings, glue compressor, crest-loss hook | D | done, audit 06 closed |
| E | `plan` sidecar, planner (targets from analysis), `solver`, `evaluate`, `cli`, blocked limiter | C, D | |
| F | `docs/architecture.md` | A | |
| G | MCP server wrapping the Python API (`analyse`, `render`, `report`, `ab`) | E | nice-to-have |

Audits: after F; after each of B, C, D, E; before each milestone.

## 6. Auditing

See `docs/audit-protocol.md`: rotating primary and second-opinion auditors across model families,
a fixed ledger per audit, `tools/audit_dispatch.py` for API, agy and local providers, manual relay
for GPT Astra. Builders write `audit/brief-NN.md` and stop; the architect dispatches and logs
`audit/NN-topic.md`. Roles: Fable 5.1 architect, Opus 5.5 builders, everything else audits.
Production target recorded 2026-10-17: JUCE-free C++ core library with a JUCE shell.

## 7. Test gates (synthetic, numeric; no ears needed)

| Module | Test | Pass |
|---|---|---|
| bands | sum of bands vs input (zero mode) | max abs error < 1e-9 |
| bands | magnitude at each fc | -6.02 +- 0.05 dB per band |
| bands | minimum mode magnitude sum | flat +- 0.05 dB |
| bands | cosine plus mixed-phase sines, >= 100 ms digital silence at both ends, fc = 60 Hz | boundary error <= -100 dB vs zero-padded reference |
| bands | same stimulus with a 50 ms linear fade at both ends | error reported; gate -70 dB |
| bands | abrupt full-scale cosine start, fc = 60 Hz | finite, deterministic, reported as inherent pre-ring (no numeric gate) |
| bands | magnitude at each fc, order 2, in the 5-band cascade | -6.02 +0.00 / -0.07 dB |
| limiter | log sweep at 1.2 amplitude plus fractional-phase impulses, limited at 8x, decimated, measured at 8x | true peak <= target + 0.05 dB |
| meter | 997 Hz full scale at 44.1 / 48 / 88.2 / 96 / 192 kHz | -3.01 +- 0.01 LUFS |
| meter | DC input | -inf LUFS (steady-state filter initialisation) |
| meter | 4x vs 8x true peak over a frequency and phase sweep | difference bounded and recorded (regression) |
| meter | FFT zero-padding true-peak reference vs 8x meter | +-0.05 dB |
| meter | silence and < 400 ms input | integrated -inf, lra and psr NaN |
| envelope | isolated impulse, both variants | no gain reduction earlier than `attack` samples before it |
| meter | EBU Tech 3341/3342 vectors | within +-0.1 LU / +-0.2 dBTP |
| meter | pyloudnorm cross-check on pink noise at 44.1 / 48 / 96 kHz | +-0.05 LU |
| tilt | pink noise (mean PSD per band) | -3.01 +- 0.1 dB/oct; synthetic -4.5 recovered +-0.1 |
| resonance | injected 7 kHz Q=12 +9 dB ring in pink noise for 40 % of duration | f0 +-1 %, start/end +-200 ms |
| resonance | sustained synth note with harmonics | not flagged |
| envelope | required-gain bound, both variants | `g <= g_req` everywhere |
| limiter | sweep + impulse train, measured at 8x after decimation | true peak <= target + 0.05 dB |
| notch | complementary identity when depth = 0 | bit-exact |
| pitch | synthetic sub bassline + kick with 250 ms tail, sidechain ducked | root per beat >= 95 % correct |
| structure | synthetic 8-bar sections with level steps | boundaries +-1 bar |

## 8. Licence

Decided: AGPL-3.0-or-later (`LICENSE`, `pyproject.toml`). (Draft 0.5 had it deferred; the
research report recommended GPL-3.0-or-later with AGPL as the SaaS-proof alternative.)

## 9. Decisions (2026-10-17) and remaining questions

Decided: stage order (3.5, now with a parallel `glue` compressor before the limiter and per-band ceilings in `multiband`); explicit `tone` stage; onset-masked clipper; correlation = L/R
correlation plus S/M ratio; saturation drive follows tilt deficit with a harmonic-vs-linear control;
S95 as control with genre presets; crest tolerances; crossovers 100/500/2000/8000; fixtures to be
supplied when needed; name and licence deferred; language deferred until after the prototype.

Superseded since (2026-11-10): crossovers are 100/400/1600/6400 Hz by default and key-locked to
the tonic without `--xover` (GS §4, §11); the name is `eluate` and the licence AGPL-3.0-or-later;
S95 control and genre presets are retired for the RMS target (GS §14.2, §17); `tone` is replaced
by the input EQ and the harmonic-vs-linear control is retired (GS §12, §14.2). Open item 3 is
closed.

Open:
1. **[Q]** Five bands (from four crossovers) rather than the brief's four. Assumed yes.
2. **[Q]** Replace the research report's unsourced genre slope table with slopes measured on your
   material and a handful of commercial releases (gate C), or adopt -4.5 dB/oct flat as the reference.
3. ~~**[Q]** Licence, when ready.~~ Closed: AGPL-3.0-or-later.
