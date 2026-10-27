# Gain staging (draft 0.1, 2026-10-18)

The maintainer's mastering chain and rules, given after the first report-mode renders,
written down as the design that replaces the current planner's dynamics and the
solver's loudness-by-limiter-drive approach. Status: **draft for maintainer
correction; all four chunks are built: 1 (foundations, §11), 2 (input EQ, §12),
3 (dynamics, §13) and 4 (loudness on the RMS target, the budget report,
saturation as the inverse of the slope, §14). The maintainer's answers of
2026-10-18 to the chunk 2 and 3 questions are in §14.1. Budgets read on the
average peak since 2026-10-19 (§3, built in §15); audible maxima in §16; the
loudness target on the maintainer's Ozone meter in §17; compressor budgets on the meter
reading, FET-style glue and frequency-dependent limiting in §18. Where §2 to §8
give older values (glue 3:1 / 30 ms, outlier caps of budget + 1.5, the 1 s RMS
target of -7.7), §16 to §18 supersede them.**
Numbers marked (corpus) wait on the
corpus survey in `work/corpus/` (`measure.py`, results not committed: maintainer
audio).

## 1. Why

On Track A, as built, upstream stages did about 1 dB of work and the final
limiter about 20 dB of peak gain reduction (median 19.8 dB in the drop, swinging
11 dB between 50 ms frames, 31.4 dB of drive). That caused the squash, the
distortion and the pumping the maintainer heard. The rule below spreads the work so
no stage does more than about 3 dB.

## 2. The chain (maintainer's order)

1. **Input normalisation.** The premaster is gained so its loudest-section
   1 s RMS equals the reference R_ref (corpus). Every threshold below is then
   relative to a known level.
2. **Input EQ** (maintainer's chain): high-pass 24 dB/oct at 25 Hz (its Q is the
   lever for a mix with too much sub), low shelf 12 dB/oct with low Q if needed,
   a bell (12 or 24 dB/oct band shelf only when a peak is targeted; most often
   the second-harmonic build-up in the low mids), high shelf 6 dB/oct from
   16 kHz, usually a boost. Static for the whole track (maintainer: a level-following
   EQ would boost pads and ambience in every breakdown that has no sub). Tone
   is this EQ plus the saturator. It replaces the current FIR tone stage, its
   gains fitted to the same whole-track spectrum target (§10; maintainer approved).
3. **Dynamic resonance suppression** (the notch stage, kept: maintainer, 2026-10-18) and
   side strip. Narrow cuts at resonant peaks that go out of bounds, each cut's
   depth following how far its peak exceeds the spectrum around it over time.
4. **Multiband dynamics**, M/S, crossovers in §4: compressor plus per-band
   limiter. The band limiter is normally gentle; where peaks get past the
   compressor's 3 dB, it may take up to 3 dB more.
5. **Saturation**, the same bands, tube or tape character (§6).
6. **Glue**, 3:1.
7. **Clipper.**
8. **Limiter.**

## 3. Budgets (average peak gain reduction, with an outlier cap)

**Changed 2026-10-19 (maintainer).** The budgets are read on the **average peak**: the
gain reduction on the typical hits across the loud sections ("the red box"), not
the single largest spike. The maintainer aims a stage at what its meter shows on the
loud hits; a few outlier spikes above that do not count, except for a bounded
allowance. Reading the budget as the maximum over the whole track (chunk 3) let
only the loudest moment reach it: on two real premasters every stage stayed in
budget, the limiter sat at a median of 0.24 to 0.40 dB when active, and the masters
came out 4.8 and 7.3 dB short of the loudness target, with the crest reduced by
3.6 and 5.4 dB against 7.8 dB on the maintainer's masters. Input normalisation is
unchanged (-18 dBFS loudest-section RMS).

Definitions, for a stage's gain-reduction curve (dB, positive = reduction, per band
and channel where that applies):

1. **Hits**: the track split into beats from the analysis beat grid (the user's
   `--bpm` when given). Without a usable grid, 400 ms frames. A hit's value is the
   maximum GR inside it.
2. **Loud section**: the hits whose 1 s RMS of the audio *entering the stage* is
   within 3 dB of that signal's loudest-section value (the p95 of the 1 s RMS,
   chunk 1's meter). If fewer than 16 hits qualify, the 16 loudest.
3. **Average peak GR**: the median of the hit values over the loud section.
4. **Outlier peak GR**: the maximum over the whole track.

| Stage | Average peak target | Outlier cap | Above the cap |
|---|---|---|---|
| Multiband compressor, per band and M / S | 2.5 dB (normal), 3 dB (exception) | 3 + 1.5 = 4.5 dB (the stage's safety clamp) | clamped; warn if the average falls more than 0.5 dB below target |
| Band limiter, per band and M / S | none: it stays an outlier catcher | at most +3 dB (maximum) | render at the caps, warn naming the band and M / S / both |
| Glue (net) | 1.5 dB (serial; parallel: 3 dB wet, the same net) | 2 + 1.5 = 3.5 dB | clamped; warn as above |
| Clipper (output-peak fall) | 1.0 dB | 1.5 dB (the maintainer's maximum) | clamped; warn as above |
| Limiter | at most 3 dB, as the loudness target needs | 3 + 1.5 = 4.5 dB | clamped; warn with the loudness shortfall |

The multiband tally is judged on the average reading: the compressor's average
plus the band limiter stays within 3 dB normally; where peaks get past the
compressor the exception is 3 dB of compressor average plus up to 3 dB of band
limiter, and past that the band renders at the caps with a warning. The engine
knows the whole track, so each threshold is **solved** (bisection, no trial
renders) for its average target on the audio actually entering the stage, in chain
order; when the outlier then passes its cap the stage is clamped at the cap. The
limiter's requested gain is solved for the loudness target and clamped so its
average peak GR stays within 3 dB and its outlier within 4.5 dB. *Assumption: when
the budgets cannot reach the loudness target, the budgets win and the report states
how far short the master is and which stage was at its cap.*

*Before 2026-10-19: peak GR was the maximum over the track (the maintainer reads the
meter's peak), each threshold solved to give the budgeted peak GR exactly; the
multiband compressor was clamped at 3 dB.*

**Makeup** (multiband, per band): PEAK = +peak GR (the maintainer's usual: GR hits
-3, band gets +3; since 2026-10-19 the band's average peak GR, *assumption*), RMS = +mean GR, AUTO = makeup following a slow average of
the band's GR. One toggle in the UI. *Assumption for AUTO: a 2 to 4 s average.*

**Loudness target**: iZotope-style RMS on the output, 1 s window, read over
the loudest section, plain dBFS (a full-scale sine reads -3.01; the report
gives the AES17 value, +3.01 dB, in brackets). Default -7.7 dBFS (§9),
`--target-rms` to change it (chunk 4, §14).

**Ceiling**: -0.3 dBTP, 24-bit, by default (maintainer, 2026-10-18; -0.1 is too
high for inter-sample peaks). A streaming render option: -1 dBTP at 16 bits.

## 4. Crossovers

Default 100 / 400 / 1600 / 6400 Hz, every crossover 4x the one below. The
lowest crossover follows the key within 80 to 160 Hz (E2 to D#3, so every pitch
class has one position): it sits below the tonic's first partial above 80 Hz
by a margin set by the crossover slope, so that partial (f1 or f2 of the bass
tonic) is the first tonic harmonic in band 2 and sits clear of the crossover.
Margin one third of an octave (maintainer, 2026-10-18). Examples: C# minor (Track A) C#3 =
138.6 Hz, crossovers 110 / 440 / 1760 / 7040; D# minor (Track G) D#3 = 155.6 Hz,
crossovers 123 / 494 / 1976 / 7904.

## 5. Multiband ratio and threshold

The higher a band's crest, the lower its ratio and threshold (maintainer: a lever
for detail per band). Draft map on the band's peak-to-loud-RMS crest
(the p99 of the 400 ms frame peak minus the p95 of the 1 s RMS):

| Band crest | Ratio |
|---|---|
| 20 dB or more | 1.3:1 |
| 17 dB | 1.5:1 |
| 14 dB | 1.7:1 |
| 11 dB or less | 2:1 |

Linear between rows. The threshold is then solved for the budgeted peak GR at
that ratio.

*Revised by the corpus (§9): crest rises with frequency (premaster medians
10 / 15 / 18 / 23 / 24 dB, band 0 to 4), so one map would put every top band
at 1.3:1 and the sub always at 2:1. Proposal: map each band's crest against
that band's own corpus range, the p10 of the band giving 2:1 and the p90
giving 1.3:1.* Corpus values (44 premasters, M, crossovers 100 / 400 / 1600 /
6400; S from results.json, given by the maintainer on 2026-10-18):

| Band | M p10 crest (2:1) | M p90 crest (1.3:1) | S p10 crest (2:1) | S p90 crest (1.3:1) |
|---|---|---|---|---|
| 0 | 8.5 dB | 12.0 dB | 11.1 dB | 17.0 dB |
| 1 | 13.5 | 17.4 | 12.4 | 17.4 |
| 2 | 15.7 | 20.4 | 13.4 | 19.4 |
| 3 | 19.8 | 25.4 | 16.7 | 26.5 |
| 4 | 20.6 | 27.5 | 21.8 | 32.3 |

Built: M bands map on the M columns, S bands on the S columns (chunk 4).

## 6. Saturation

After the multiband, before the glue, the same bands. Tube or tape character
(the maintainer uses FabFilter Saturn in those modes only). Drive in 0.5 dB steps,
at most 2 dB (maintainer: never above 2 dB on a mastering pass), output
auto-compensated as Saturn does. Pulled only when a band sits below the target
slope **and** is harmonically sparse; a harmonically rich band (much bass music
is already distorted) gets none, and the deficit goes to EQ instead. Shape:
widen the top end, emphasise the low mids and mids.

Maintainer ruling (2026-10-18), calibration against Saturn set aside for now:
harmonics are the inverse of the slope. A band that sits below the target
slope line (too dull) gets drive in proportion to its deficit: drive = the
band's deficit in dB, rounded to 0.5 dB, at most 2 dB; bands on or above the
line get none. *Assumption: the input EQ still corrects level; the saturator
(unity small-signal gain, so it adds harmonic density rather than level) acts on
the deficit measured before the EQ, so a dull band gets both. Drive is in the
engine's own units until the Saturn calibration is done
(`work/calibration/saturn/calibration_tones.wav` is ready for it).*

## 7. Character presets

Transparent, Slow, Moderate, Fast, Aggressive. Auto picks by tempo: under
70 BPM Slow, 70 to 140 Moderate, above 140 Fast. Draft values for the maintainer
to correct (bands low to high; ms):

| Preset | Multiband attack | Multiband release | MB knee | Glue attack / release | Clipper knee | Limiter attack / release | Band lookahead |
|---|---|---|---|---|---|---|---|
| Transparent | 30 / 25 / 15 / 10 / 5 | 300 / 250 / 200 / 150 / 120 | 10 dB | 30 / 600 | soft, 1.5 dB | 5 / 200 | 20 / 10 / 5 / 2 / 2 |
| Slow | 30 / 25 / 15 / 10 / 5 | 300 / 250 / 200 / 150 / 120 | 6 dB | 30 / 400 | soft, 1 dB | 3 / 150 | 20 / 8 / 4 / 2 / 2 |
| Moderate | 20 / 15 / 10 / 5 / 3 | 200 / 150 / 100 / 80 / 60 | 6 dB | 30 / 300 | soft, 1 dB | 2 / 100 | 15 / 6 / 3 / 2 / 2 |
| Fast | 10 / 8 / 5 / 3 / 1 | 120 / 100 / 60 / 40 / 30 | 3 dB | 10 / 150 | soft, 0.5 dB | 1 / 50 | 10 / 5 / 2 / 2 / 2 |
| Aggressive | 5 / 4 / 3 / 1 / 0.5 | 80 / 60 / 40 / 30 / 20 | 1 dB (maintainer, 2026-10-18: "1 dB is aggressive enough") | 3 / 80 | soft, 0.5 dB (maintainer, 2026-10-23; was hard) | 0.5 / 30 | 6 / 4 / 2 / 2 / 2 |

Low-band release never below 25 ms (one 40 Hz cycle) in any preset: faster
than that, the gain moves inside the bass waveform and distorts.

Band lookahead (maintainer, 2026-10-23; docs/quality-review-2026-10-22.md items 1 and 3, with
the clipper knee floor): the lookahead of the multiband's band ceilings
(`MultibandPlan.ceiling_lookahead_ms`) and of the frequency-dependent limiter
(`LimiterPlan.speclim_lookahead_ms`), per band, ms. It was 2 ms on every band; a 3 to 6 dB
cut on a 40 to 60 Hz band then lands inside a tenth of a cycle (amplitude modulation,
heard at 100 to 500 Hz). The band ceilings take the character's; the frequency-dependent
limiter takes the limiter timing's (the live profile's transparent, else the character's).
Plans made before keep 2 ms.

## 8. Clipper scope

Maintainer, 2026-10-18: the kick-only mask is wrong (the kick is already the loudest
thing in most tracks). The clipper works on the whole signal, aims for 1 dB of
peak reduction, never more than 1.5 dB, and is a **tonal lever** as well as a
dynamic one, because it generates harmonics. A "kick detective" toggle keeps the
old behaviour (clip only around detected kick onsets); in full auto it is the
less aggressive option. As built, the clipper acts only around kick onsets, and
kick detection found none on Track B, so the clipper did nothing there.
Maintainer, 2026-10-18: in auto the kick detective is on when the character
resolves to slow or transparent; the 1 dB target (at most 1.5) is the fall of
the output peak, after the band-limit (built in chunk 4).

## 9. Corpus survey (2026-10-18)

44 unique premasters and 18 of the maintainer's masters (17 matched to their
premaster), from the folders the maintainer listed (`work/corpus/paths.txt`, audio
not in the repo). Metric: 1 s RMS (sliding, plain dBFS, a full-scale sine
reads -3.01), loudest section = p95 of the 1 s RMS; crest = sample peak minus
that; band crest = p99 of the 400 ms frame peak minus the band's p95 1 s RMS,
crossovers 100 / 400 / 1600 / 6400.

| | Premasters (44) | Maintainer's masters (18) |
|---|---|---|
| Sample peak | -10.6 to 0.0 dBFS, median -4.3 | -0.2 to 0.0 |
| Loudest-section RMS | -27.1 to -13.9, median -19.2 | -8.9 to -7.2, median -7.7 |
| Loudest 1 s RMS | -24.1 to -12.5 | -8.2 to -5.6, median -6.8 |
| Crest | 9.2 to 18.9 dB, median 14.9 | 7.1 to 8.7, median 7.6 |
| Loudest-section RMS with the peak at -3 dBFS | -21.9 (most dynamic) to -12.2, median -17.9 | |
| S95 | -26.6 to -14.5 LUFS | -8.7 to -6.7, median -8.1 |

Maintainer's masters against their premasters (17 pairs, medians): crest reduced
7.8 dB, loudest-section RMS raised 10.6 dB, band crest reduced (M)
2.4 / 1.8 / 1.4 / 1.2 / 1.1 dB, at most 3.5 / 3.0 / 3.9 / 2.1 / 2.3: the
"about 3 dB per band, never much more" rule, measured.

The engine's renders on the same metric: Track A loudest-section RMS -7.4,
loudest 1 s -4.9, crest 6.2 (the maintainer's masters: crest never below 7.1);
Track B -8.2, -6.6, 7.0. Track A's premaster has a crest of 19.9 dB, more
dynamic than any premaster in the corpus (band 3 and 4 crest 26.5 and 31.4 dB,
above the corpus maxima), so reaching the maintainer's usual 7.6 dB crest needs
about 12 dB of reduction against the corpus median of 7.8: the case the
budget warnings exist for.

Proposed from this: R_ref (input normalisation) = loudest-section RMS of
-18 dBFS, the corpus median with the peak at -3 dBFS; default loudness target
= loudest-section RMS of -7.7 dBFS (the maintainer's median), the maintainer's
"1.5 dB quieter than Track A" agreeing within 0.2 dB. *Open: the ceiling. The
maintainer's masters peak at -0.1 to -0.2 dBFS sample peak; the engine uses
-1 dBTP.*

## 10. The tone stage as built, and the proposal

As built, tone is one static zero-phase FIR (stages/tone.py), bounded to +-3 dB,
designed from the whole-track long-term spectrum: the mean power per 1/3 octave
over every frame (power averaging weights loud sections far above breakdowns),
its fitted slope over 100 Hz to 10 kHz rotated onto -4.5 dB/oct about 1 kHz,
and the sub (30 to 90 Hz) held to the 100 Hz correction and kept within +-3 dB
of the line. It does not follow level over time, so it does not boost breakdowns.
It sits after the multiband. The only time-varying EQ is the notch stage: at
most 8 narrow cuts at persistent resonances found by the analysis, each depth
following that resonance's excess over time (up to -6 dB), ducked around kicks.
That is a targeted resonance remover, not a soothe-style broadband suppressor;
there is none in the code.

Approved by the maintainer (2026-10-18): the maintainer's static parametric input EQ
(§2 item 2) replaces the FIR, fitted to the same whole-track target; the notch
stage stays as its own stage: the high-pass Q from the sub excess,
the low shelf from the sub shelf, the bell from the low-mid excess against the
line (the second-harmonic build-up), the high shelf from the 10 to 16 kHz
deficit. Same target, the maintainer's tools, and settings a person can read.

## 11. Built in chunk 1 (foundations, 2026-10-18)

What landed (§2 item 1, §3 "Ceiling" and "Loudness target", §4, §9):

- **Crossovers x4.** `bands.DEFAULT_CROSSOVERS` = 100 / 400 / 1600 / 6400 Hz,
  and every default that copied it (plan, multiband, saturate). Tests about
  something else are pinned to the old 100 / 500 / 2000 / 8000.
- **Key-locked lowest crossover.** `planner.key_crossovers(tonic)`: p = the
  tonic's lowest frequency at or above 80 Hz, fc1 = p / 2^(1/3) clipped to
  80 to 160 Hz, then x4, x16, x64 (C# gives 110 / 440 / 1760 / 7040, D#
  123.5 / 493.9 / 1976 / 7902). Without `--xover` the analysis finds the key
  first (chroma weighted by momentary loudness, which reads no band split) and
  runs every band-dependent feature on the key-locked crossovers. With
  `--xover` the given ones are used. `plan_record.json` keeps the crossovers,
  the rule, the tonic, p and a one-line reason. `source.json` keeps the
  crossovers used, the ones requested (null = key-locked) and R_ref. The
  analysis cache is reused when the request matches, so key-derived
  crossovers do not break reuse.
- **1 s RMS meter.** `meter.rms_series` / `meter.rms_1s`: a 1 s window at a
  100 ms hop, the mean of the channels' squares, plain dBFS (a full-scale
  sine reads -3.01). It returns the loudest section (p95 over windows above
  -70 dBFS) and the loudest 1 s. `aes17=True` adds 3.01 dB. Both readings are
  in `meter.measure` and in the report, before and after.
- **Input normalisation.** `planner.R_REF_DBFS` = -18.0 (§9). `planner.analyse`
  measures the premaster, gains it to R_ref and analyses the gained audio.
  `Plan.input` (`InputPlan`: gain, R_ref, measured value) is the first stage
  (`stages/gain.py`, also run by `solver.render_pre` and `planner.realise`).
  It is in plan.json and in the report ("0. Input"). A premaster rendered at
  0 and -17 dB gives the same pre-limiter audio to 1e-9 and the same master
  to one 24-bit step (`tests/test_gain_staging.py`).
- **Ceiling and export.** -0.3 dBTP by default (`planner.CEILING_DBTP`,
  `LimiterPlan`, `Targets`). `Plan.output` (`OutputPlan`) is the last stage
  (`stages/output.py`): 24-bit PCM, rounded, by default. `--streaming` gives
  -1 dBTP and 16-bit PCM with TPDF dither (seeded, so a saved plan renders
  the same file). The solver measures every constraint, the true peak
  included, on the quantised samples, and master.wav holds exactly those.

Deviations, with reasons:

1. **The analysis runs on the normalised premaster.** The alternative was to
   analyse the raw premaster and add the gain to every analysis-derived
   level. The analysis also has absolute gates (the -60 dBFS crest gate, the
   -70 LUFS loudness gate, the onset detector's log compression), so shifting
   thresholds alone would not make the output independent of input level.
   Analysing the gained audio does: the multiband thresholds and ceilings,
   the limiter's starting drive and the glue threshold all come out already
   relative to R_ref. Consequence: levels in features.pkl and analysis.json
   are those of the normalised audio. The report's "before" column still
   measures the premaster as delivered.
2. **The 80 Hz clip binds for E, F, F# and G.** For those keys p / 2^(1/3)
   is 65.4 to 77.8 Hz, so fc1 is held at 80 Hz. The margin is then less than
   a third of an octave: E2 (82.4 Hz) sits half a semitone above the
   crossover. Built as specified; the maintainer to rule.
3. **24-bit output is rounded without dither.** §3 names dither only for the
   16-bit streaming option.
4. **Plan version 0.5.** Plans saved by earlier code no longer load. That is
   the existing rule; re-run `master`.

Not built here (chunks 2 to 4): the input EQ, the dynamics redesign, the
loudness target on the 1 s RMS meter (the solver still targets S95) and the
budget warnings.

## 12. Built in chunk 2 (input EQ)

`stages/inputeq.py` follows the input normalisation (chunk 1) and comes before
the notch and side strip: high-pass 24 dB/oct at 25 Hz, low shelf 12 dB/oct
(Q 0.707, corner 60 to 250 Hz), bell (cut only, 120 Hz to 1 kHz, Q 0.7 to 4, used only when the
low-mid excess above the line passes 1.5 dB), high shelf 6 dB/oct (corner 1 to
16 kHz), every gain within +-3 dB, static for the whole track, the same filter
on M and S. Zero phase (forward-backward, each section designed at half its dB
gain so the double pass gives the specified response); minimum phase is an
option. `planner.plan_inputeq` fits it by least squares on the 1/3-octave LTAS
nodes from 30 Hz to 18 kHz against the same target as the FIR (§10), and the
plan record reads e.g. "low shelf +1.2 dB at 120 Hz". The high-pass Q drops
from 0.707 toward 0.5 as the predicted sub shelf exceeds +3 dB. The solver's
tone rotation now refits this EQ. The FIR tone stage is out of the default
chain (`stages/tone.py` kept, unused).

Merged with chunk 1 (2026-10-18): the chain is input gain, input EQ, notch,
side strip, multiband, saturate, glue, keyeq, limiter, output. The analysis
the EQ is fitted from runs on the normalised premaster, and the fit reads
only the LTAS shape against its own fit line and its loudest node, so the
fitted EQ does not move with the premaster's level (a premaster at 0 and
-17 dB gives the same EQ and the same pre-limiter audio,
`tests/test_gain_staging.py`). Both branches had bumped the plan to 0.5 with
different sections, so the joined plan is version 0.6.

## 13. Built in chunk 3 (dynamics, 2026-10-18)

Every dynamics stage is solved for its peak-GR budget (§3) on the audio actually
entering it, in chain order (multiband, saturate, glue, clipper, limiter), in
`planner.realise` and in the solver's Chain. Peak GR is the maximum over the
whole track of a stage's gain reduction, per band and channel where that
applies. Each stage has a pure solve that bisects on the stage's own envelope
design: `multiband.threshold_for_peak_gr` / `ceiling_for_peak_gr`,
`glue.threshold_for_peak_gr`, `limiter.clip_threshold_for_peak_gr`, and
`solver.limiter_input_gain`.

- **Multiband** (`multiband.design_budget`, per band and M / S): the ratio comes
  from the band's crest (p99 of the 400 ms frame peak minus p95 of the 1 s
  RMS) through the §5 table: the band's p10 crest gives 2:1, its p90 1.3:1,
  linear between, clamped. The compressor threshold is solved for 2.5 dB of
  peak GR. The band limiter takes the peaks above the band's p99 400 ms frame
  peak, at most 3 dB minus the compressor's (normal tally 3 dB). Only where
  that is not enough is the compressor re-solved at 3 dB and the limiter
  allowed up to 3 dB more. Past that the band renders at the caps and the plan
  record gets `{"stage": "multiband", "band", "channel": "M" | "S" | "both",
  "needed_db", "cap_db"}`. Makeup is `--makeup peak` (default: + the band's
  combined peak GR), `rms` (+ the mean GR over 1 ms points whose detector is
  above -60 dBFS) or `auto` (a time-varying makeup following the 3 s centred
  moving average of the band's GR). The side makeup rule applies on top.
  `MULTIBAND_MAX_GR_DB` stays the stage's hard clamp on the compressor.
- **Glue** (`glue.design_budget`), 3:1: `--glue-mode serial` (default) solves
  the threshold for 1.5 dB net peak GR. `parallel` solves the wet path for
  3 dB and sets the mix so the net peak GR is the same 1.5 dB. A miss of more
  than 0.1 dB gives a warning.
- **Clipper**: whole signal by default (`LimiterPlan.clip_threshold_db`). Every
  span whose 8x level passes the knee is clipped by the existing soft-knee
  curve and band-limit, with no onset mask. The threshold is solved for 1 dB
  of peak reduction (at most 1.5 dB). `--kick-detective` keeps the old
  onset-masked clipper. The code notes that the clipper is a tonal lever whose
  harmonics chunk 4 will use.
- **Limiter**: at the chunk 1 ceiling, its input gain is clamped so its own
  peak GR on the clipper's output is at most 3 dB. The hook for chunk 4 is
  `planner.clamp_limiter_gain(lp, requested_db)`: the requested gain
  (`LimiterPlan.requested_gain_db`, today the solver's S95 drive) is clamped
  to the budget, with a warning naming the needed and capped GR.
- **Character** (`--character`, §7 table): `planner.CHARACTER_PRESETS` sets the
  multiband attack, release and knee, the glue attack and release, the clipper
  knee, the limiter attack and release and the per-band lookahead of the band
  ceilings and the frequency-dependent limiter. `auto` picks by tempo (the user's
  `--bpm`, else the analysis's): under 70 BPM slow, 70 to 140 moderate, above
  140 fast, moderate without a tempo. The analysis's tempo is octave-checked
  (features/structure.py `_octave`, sprint 2026-10-21 S2) and the reason quotes a
  moved octave's evidence; a detected tempo under 70 reads moderate, not slow:
  the boundaries are an octave apart, and an octave error must not turn the kick
  detective on (slow needs `--bpm`, later a MIDI tempo map). The low band's release is never below
  25 ms, including after the solver's release lever.
- **Report data**: `planner.budget_report(plan, lim_info)` gives per-stage peak
  GR against budget, planned and rendered, plus every warning. It goes in
  `plan_record.json` (`budget`, `warnings`), `solve.json` (`reported.budget`)
  and `report.json` (`budget`). Formatting it in report.md is chunk 4.

Deviations, with reasons:

1. **When peaks "get past the compressor".** The compressor's attack is a
   lookahead window, so no peak arrives before its reduction. "Past" is read as
   the band's outlier peaks: the band's highest peak above the p99 of its
   400 ms frame peaks, measured after the compressor. On short material the
   p99 of a few frames sits near the maximum, so the band limiter rarely acts
   there.
2. **The ratio map uses the M corpus table for M and S.** The S values are in
   `results.json`, which is not in the repository (maintainer audio).
3. **Soft knee on the multiband compressor.** §7 gives an MB knee per preset.
   The compressor gained the glue's soft-knee curve; a knee of 0 keeps the
   original curve bit for bit.
4. **Clipper threshold before the limiter gain.** It is stored in dBFS of the
   8x signal before the limiter's input gain, so the clipper's reduction does
   not move with the drive, and the clipper's 1 dB is planned once per glue
   output. The knee (`clip_knee_db`) is how far below the threshold the soft
   curve starts. The 1 dB is the curve's reduction at the track's 8x peak; the
   band-limit lets some of it back (about 0.5 dB on the test signals), so the
   output peak falls by less.
5. **Budgets win over S95.** The solver still brackets its drive on S95 (chunk
   4 moves it to the RMS target), but the limiter budget clamps it. On material
   that cannot reach the target inside the budgets, the solve reports S95 as
   binding, the shortfall and the limiter warning (§3's assumption). With the
   budgets on, the solver's ceiling-offset and input-gain lever classes are
   off, because the thresholds and ceilings are solved. The glue lever moves
   only the release (the mix is solved), and the clip-depth lever applies only
   to the kick detective.
6. **Kick detective in auto.** It is recorded as the less aggressive clipper
   option (`plan_record.character.clipper`). Auto does not switch it on by
   itself; the maintainer to rule.
7. **Plan version 0.7.**

## 14. Built in chunk 4 (loudness, report, saturation, 2026-10-18)

### 14.1 The maintainer's answers (2026-10-18), applied

Chunk 3 (dynamics, PR #3):

1. **Band limiter trigger**: kept as built (the band's peak above the p99 of
   its 400 ms frame peaks, measured after the compressor; §13 deviation 1).
2. **S ratio map**: the S corpus values, p10 -> 2:1 and p90 -> 1.3:1, per band
   0 to 4: 11.1-17.0, 12.4-17.4, 13.4-19.4, 16.7-26.5, 21.8-32.3 dB. In the §5
   table; `multiband.CREST_P10_S_DB` / `CREST_P90_S_DB`, and
   `ratio_for_crest(crest, band, channel)` reads the S table for S. Replaces
   §13 deviation 2.
3. **Kick detective in auto**: on when the character resolves to slow or
   transparent (`planner.kick_detective_choice`; `--kick-detective` /
   `--no-kick-detective` override it). Replaces §13 deviation 6.
4. **Clipper target**: the 1 dB target (at most 1.5) is the fall of the
   output peak, after the band-limit, not the curve's reduction at the 8x peak.
   `solver.clip_threshold_for_output_peak` brackets the threshold from the
   curve's own solution downward and solves (brentq) on the 8x peak after the
   clipper and band-limit (`limit_blocked(measure_only=True)`). The curve's
   reduction at the 8x peak is now at least 1 dB and can be well above the
   old 1.5 dB cap (1.0 dB on the short synthetic track, where the band-limit
   lets nothing back; 2.3 dB on noise with 30 ms bursts) and is recorded as
   `curve_gr_db`. Clip knee meanings unchanged.
   Replaces §13 deviation 4's last sentences.
5. **AUTO makeup**: the centred 3 s average is correct; unchanged.
6. **Aggressive multiband knee**: 1 dB, not hard. §7 table updated.

Chunk 2 (input EQ, PR #2):

1. **Low shelf**: Q 0.707 is fine; unchanged.
2. **High shelf corner**: the fitted 1 to 16 kHz range stays; the maintainer
   judges the result by ear.
3. **HPF Q for a thin sub**: a sub deficit raises the high-pass Q from 0.707
   toward 1.0, mirroring the excess rule: `planner.hpf_q` rises linearly once
   the predicted sub shelf is below -3 dB (SUB_SHELF_MAX_DB) and reaches 1.0
   at 6 dB of deficit beyond that window (-9 dB), as the excess rule reaches
   0.5 at 6 dB beyond +3.
4. **Band-shelf bell**: built (`plan.EqBandShelf`, `InputEqPlan.band_shelf`).
   A cut-only, flat-bottomed dip between a low and a high corner with 24
   dB/oct transitions: magnitude 1 - (1 - g) |HP4(lo)| |LP4(hi)|, the
   4th-order Linkwitz-Riley high-pass and low-pass, so no ripple and half the
   dip (in amplitude) at each corner; depth within -3 dB like the other
   sections; low corner 100 Hz to 1 kHz, high corner 1/3 to 3 octaves above
   it. Zero phase by subtracting the zero-phase band; under "minimum" phase
   the in-phase three-way Linkwitz-Riley sum. The planner uses it instead of
   the bell when the low-mid excess above the line is broad: the contiguous
   run of 1/3-octave nodes above 1.5 dB around the largest excess (120 Hz to
   1 kHz) spans more than one octave (four or more nodes); a narrower peak
   keeps the bell (`planner.low_mid_run`).
5. **25 Hz HPF**: stays a rumble filter outside the tonal fit; unchanged.

### 14.2 What landed

- **Loudness target.** The master's loudest-section 1 s RMS (chunk 1's meter,
  plain dBFS), default `planner.DEFAULT_TARGET_RMS_DBFS` = -7.7, `--target-rms`
  to change it. `Targets.rms_dbfs` with a 0.3 dB tolerance replaces the S95
  target; S95 is measured and reported, never a target. The chain: input
  normalisation to -18, every stage solved for its budget with its makeup,
  then the limiter, whose requested gain is solved for the target:
  `solver.gain_for_rms` brackets and brentqs the proxy limiter's RMS plus a
  bias, recalibrated (real render minus proxy) after every real render, as
  the S95 drive was. `LimiterPlan.target_rms_dbfs` carries the target, so
  `planner.realise` (`eluate plan`) solves the gain on the uncalibrated
  proxy as well.
- **Budgets win.** The requested gain is clamped to the limiter's 3 dB
  (`planner.clamp_limiter_gain`); the limiter warning then carries the
  loudness shortfall in dB (`shortfall_db`: the proxy's estimate at plan time,
  the measured one after the solve), and the solve adds a
  `{"stage": "loudness", "target_dbfs", "achieved_dbfs", "shortfall_db",
  "cause"}` warning whenever the RMS target is missed.
- **Genre presets as RMS targets** (retired 2026-10-20, maintainer: loudness by genre is the
  wrong framing; `--genre` and `planner.GENRE_PRESETS` removed). The S95 presets' LU
  differences from their -6.5 LUFS default, as dB offsets from -7.7 dBFS:

  | Genre | S95 preset (old) | RMS target | Range |
  |---|---|---|---|
  | downtempo | -9.0 LUFS | -10.2 dBFS | -13.2 to -8.2 |
  | techno, trance, house | -6.5 | -7.7 | -9.2 to -6.2 |
  | festival, edm, bass, dubstep, dnb | -5.0 | -6.2 | -7.2 to -5.2 |

- **Solver.** The variables are the limiter's input gain (solved for the
  target, then clamped), the tone rotation (the slope-first pass, refitting
  the input EQ) and the limiter ceiling (the true-peak correction). Render
  order: planned; a true-peak ceiling render when the constrained meter reads
  over; up to 2 tone-rotation renders while the slope is out (kept when the
  summed excess without the RMS and true-peak terms falls); up to 3 loudness /
  true-peak renders while the target is missed with gain left in the budget
  or the true peak reads over (kept when the full score does not rise). The
  history keeps every render, kept and reverted. The old binding / infeasible
  report and the nearest-feasible search are replaced by `misses` (the
  targets the kept render misses) and `warnings` (the budget warnings plus
  the loudness warning); `master_feasible.wav` / `plan_feasible.json` are no
  longer written. The Chain caches the pre-limiter audio per tone rotation
  (every dynamics stage solved on its real input) and the clipper threshold
  per rotation (and, for the kick detective, per ceiling).
- **Saturation as the inverse of the slope.** Per band b (never the sub),
  drive_b = the band's deficit below the target slope line, measured on the
  analysed spectrum before the input EQ (`planner.band_deficits`: the mean
  over the band's input EQ fit nodes of the target line minus the LTAS, the
  target line being the measured fit line rotated onto the target slope about
  1 kHz), rounded to 0.5 dB (half up), at most 2 dB
  (`planner.saturation_drive_db`); bands on or above the line get none. M and
  S alike, except the high-mid S band, which keeps the width rule at 2 dB.
  The wet share is 1 on every band. `plan_record.json` `saturation` lists
  each band's deficit and drive.
- **Report** (`evaluate.report_md`): **Warnings** at the very top, one plain
  line per warning (`evaluate.warning_text`: stage, band with its frequency
  range, M / S / both, needed against cap, the loudness shortfall), then the
  other targets missed; **Gain reduction against budget**: per band and
  M / S the compressor, band limiter and total peak GR, the glue's net, the
  clipper's output-peak fall and the limiter's peak GR, each with its budget
  and marked over or under; **Loudness**: the target, and before / after RMS
  with the AES17 value in brackets and S95. "What was done" adds the low-mid
  shape and why, the character and why auto picked it, the makeup mode, the
  per-band ratio, threshold, timing and knee table, the saturation drives with
  their deficits, the glue and clipper modes, the kick detective's reason and
  the limiter's requested gain. Numbers carry 1 or 2 decimals; the old "Gain
  reduction" section and the distance-to-master section no longer print raw
  lists or dicts. report.json gains `warnings` and `loudness`.
- **CLI.** `--character {auto, transparent, slow, moderate, fast,
  aggressive}`, `--makeup {peak, rms, auto}`, `--kick-detective` /
  `--no-kick-detective` (default auto), `--glue-mode {serial, parallel}`,
  `--target-rms`, `--streaming`; `--genre` mapped to its RMS target (retired 2026-10-20). `--target`
  (S95) and `--harmonic` are gone. Option abbreviations are off, so an old `--target -6.5` fails
  instead of being read as `--target-rms`. `eluate report` rebuilds the new
  report from the files `master` writes. run.json reports `ok`, `misses`,
  `warnings`, the RMS and S95.
- **Plan version 0.8** (the band shelf and the limiter's RMS target).

Retired (code and tests removed): the solver's lever classes
(`_lever_classes`: multiband attack / release scales, band ceiling offsets,
glue mix and release, clip depth, threshold offset) with their `Levers`
fields and constants (MB_SCALE_MAX, MB_ATTACK_*, MB_RELEASE_MAX_MS,
MB_CEIL_*, DRIVING_BANDS, GLUE_MIX_STEP, GLUE_MIX_MAX, GLUE_RELEASE_*,
CLIP_STEP_DB, CLIP_MAX_DB, THR_STEP_DB, THR_MAX_DB, IMPROVE_SCORE),
`driving_bands`, `binding`, `_nearest_feasible`, `_finite_or` and
NEAREST_*, `SolveResult.feasible / binding / nearest_*`, the S95 target
(`Targets.s95_lufs`, `Controls.target_s95`, `Controls.s95_target`,
`DEFAULT_TARGET_S95`), the 1 - flatness harmonic share
(`planner.harmonic_blend`, `Controls.harmonic`, HARMONIC_RANGE,
`saturate.split_deficit`), SAT_DRIVE_PER_DB, SAT_DRIVE_CAP_DB,
SAT_WIDTH_DRIVE_DB, `planner.plan_tone_saturate` (replaced by
`plan_saturate`; `tone_deficit` stays as the record) and `design_glue`'s
threshold offset.

Deviations and assumptions, with reasons:

1. **HPF Q "6 dB of deficit"** is read as 6 dB beyond the -3 dB window, the
   mirror of the excess rule's span (EQ_HPF_Q_SPAN_DB). If the maintainer meant a
   predicted shelf of -6 dB, the span becomes 3 dB.
2. **Band-shelf corners** are where half the dip (in amplitude) is reached,
   the Linkwitz-Riley crossover points. Between corners less than about two
   octaves apart the bottom does not reach the full depth (corners two octaves
   apart at -3 dB give -2.6 dB in the middle); the fit chooses the corners for
   the shape it needs.
3. **The deficit for saturation** is the mean over the band's 1/3-octave
   nodes. The top band includes the nodes above 10 kHz, where the target line
   runs parallel to the measured fit (as in the EQ's deficit). The air band
   can now be driven (the old rule never drove it).
4. **The kick detective in auto** also applies when the user names the slow or
   transparent character (auto refers to the toggle's default, not only to
   `--character auto`).
5. **The loudness re-renders stop when the limiter is clamped**: more renders
   cannot add gain the budget does not allow.
6. **Crest modes** "loss" and "floor" are still measured and reported as
   misses, but no lever acts on them now; the default stays "report".

## 15. Built: budgets on the average peak (maintainer, 2026-10-19)

The §3 reading as built, in every dynamics stage and the report:

- **One shared reading.** `envelope.hit_edges` (beats from the analysis grid, which
  follows `--bpm`; 400 ms frames without a usable grid), `envelope.loud_hits` (the
  loud section on the audio entering the stage: the 1 s RMS window centred nearest
  each hit's centre within 3 dB of the p95; at least the 16 loudest),
  `envelope.average_peak_gr` (median of the per-hit maxima over the loud hits, and
  the track's maximum) and `envelope.summarise_hits`. `envelope.hits_for(x, fs,
  beats)` is the form every stage takes as `hits`; `hits=None` is the whole track
  as one hit, so the average equals the maximum (the chunk 3 reading, kept for
  plans without hits and as the comparison in the tests).
- **The plan carries the grid.** `Plan.hits` (`HitsPlan`: beats, source,
  reading "average" or "max"), filled by `planner.plan_hits`; `planner.stage_hits`
  gives each stage its hits on its real input, in `planner.realise` and the
  solver's Chain. Plan version 0.9.
- **Multiband.** `threshold_for_peak_gr(..., hits)` bisects the threshold for
  the average target and returns the outlier. `MULTIBAND_MAX_GR_DB` is now 4.5 dB
  (the 3 dB budget + 1.5), the compressor's floor at every sample, so the outlier
  cap holds by construction; the exception target is its own constant
  (`COMP_GR_MAX_DB`, 3 dB). The band limiter is still solved on its maximum
  (an outlier catcher, at most 3 dB). The tally and the exception read the
  compressor's average. PEAK makeup is + the combined average peak GR.
- **Glue.** The wet path's required gain is floored at the outlier cap
  (`glue.OUTLIER_CAP_DB` = 3.5 dB net; in parallel, the wet GR at which the net
  reaches 3.5 dB at the solved mix, re-solved only when it binds), then the
  threshold is bisected for the net average. Warns only when the net average is
  more than 0.5 dB short.
- **Clipper.** `solver.clip_threshold_for_average_peak`: per hit, the 8x input
  peak minus the 8x peak after the curve and its band-limit
  (`limit_blocked(measure_only=True, detector_env=True)`, a per-sample detector
  envelope); bisection for a 1 dB median fall, then, when the largest hit's fall
  passes 1.5 dB, the threshold is raised by bisection until it is 1.5 dB.
- **Limiter.** `solver.limiter_peaks` records the clipped detector peak of the
  track and the loud hits' median; `solver.limiter_gain_max` = min(ceiling + 3 -
  median, ceiling + 4.5 - peak). The requested gain is solved for the loudness
  target as before and clamped to that; the warning carries both readings and the
  shortfall.
- **Report.** `planner.budget_report` gives each stage's average and outlier with
  its budget, its outlier cap and a status (ok, over, under); report.md's table
  has those five columns and the sentence "Average peak = the typical loud hit;
  outlier = the single largest moment."

Deviations and assumptions, with reasons:

1. **Loud section for the limiter.** Read on the clipper's input (the glue's
   output), not its output, which exists only inside the blocked limiter; the
   clipper changes the 1 s RMS by far less than the 3 dB window.
2. **The loud section is broadband.** Every band of the multiband uses the loud
   hits of the stage's full input, not of the band, so a band that is quiet in
   the drop is still measured on the drop.
3. **"Clamps at the cap".** For the multiband and the glue, a floor on the
   required gain (the outlier never passes the cap, and the average is solved
   with the floor in place); for the clipper, the threshold raised until the
   largest hit's fall is 1.5 dB; for the limiter, the gain clamp.
4. **The limiter's per-hit GR** is max(0, hit detector peak + gain - ceiling), the
   required gain at each hit's peak (box_smooth keeps its minimum); the lookahead
   of a neighbouring hit is not counted.
5. **PEAK makeup** follows the new reading (+ the average peak GR, not the
   outlier), so the typical loud hit is restored to unity, as "GR hits -3, band
   gets +3" did on the meter's reading.

## 16. Built: audible maxima (docs/dynamics-macros.md step 1, 2026-10-19)

Why: on the first real renders of §15 (Track A, Track B) the fixed outlier caps bound
before the averages. The loudest hits sat 4.6 to 4.9 dB (Track A) and 2.1 dB (Track B,
at the clipper) above the typical loud hit, so the clipper's average fell to 0.0 and
0.05 dB, and on Track A the limiter's average to 0.0 dB with its outlier at the 4.5 dB
cap, which set the gain (6.0 and 3.9 dB short of the target). The maintainer asked for the
architect's recommended values, built locally, then a listen.

- **Clipper: a per-sample cap of 3 dB** (`LimiterPlan.clip_max_gr_db`,
  `planner.CLIP_SAMPLE_CAP_DB`): the curve's output is max(curve(x), |x| 10^(-3/20)),
  so no sample loses more than 3 dB and what passes it goes to the limiter. The
  threshold is solved for the 1 dB average and is never raised to make room for the
  outlier (it was raised until the largest hit fell 1.5 dB). Kick-detective plans
  are unchanged.
- **Limiter outlier cap 5 dB** (`LimiterPlan.max_outlier_gr_db`,
  `planner.LIMITER_OUTLIER_CAP_DB`), average budget 3 dB as before. Not 6: on Track A
  4 dB on a lone hit after silence with the 100 ms release already ducks the next bar.
- **Glue outlier cap 4 dB** (`glue.OUTLIER_CAP_DB`, was 3.5).
- **Multiband unchanged at 4.5 dB.** With PEAK makeup it gave back most of what it
  took from a stray (0.7 dB of hit-spread reduction measured on Track A); a deeper
  per-band cut on broadband drop hits would flip the band balance for the release time.
- **Gate:** a band channel whose loudest-section 1 s RMS is more than 30 dB below the
  stage input's (`multiband.GATE_BELOW_INPUT_DB`) is not compressed (the low S band
  after the side strip sat about 34 dB down and was solved to a -60 dBFS threshold).
- **Report:** the status column names what held a stage back ("held by outlier cap",
  "held by average budget", "sample cap reached", "gated (near silent)"), and the
  limiter warning says which reading set the gain.
- Plan version 0.10. Not built: the job sizing, shares, ρ and the macros (steps 2 to 5).

## 17. The loudness target on the maintainer's meter (2026-10-19)

The maintainer reads loudness off the Ozone 11 RMS bar (meter type RMS + Peak, integration
300 ms (VU), stereo) at the loudest passage. Calibrated with test files: a -6 dBFS
1 kHz sine reads -6 (so the meter is AES17: a full-scale sine reads 0, 3.01 dB above
the plain dBFS this engine used); 200 ms bursts every 2 s read -7.4 (a sliding window of
about 280 ms, or an exponential of about 155 ms; a flat 300 ms window predicts -7.8 and
the two agree within 0.2 dB on music). His reading of "about -5" on the Track B and
Track A masters matches the p95 of the 300 ms series (-5.2 and -5.4), not its maximum
(-3.5 and -3.8).

- `meter.rms_meter`: 300 ms sliding RMS at a 10 ms hop, AES17, p95 over windows above
  -70 dBFS (and the maximum). `meter.measure` reports it as `rms_meter_dbfs` /
  `rms_meter_max_dbfs`.
- The loudness target (`Targets.rms_dbfs`, `--target-rms`, the solver's measure and
  proxy) reads it. **Default -4.1**, the median of the maintainer's 18 masters on this
  reading (range -5.0 to -3.7; `work/corpus/meter300.py`). Genre presets kept their
  offsets here until they were retired the next day (§14). The old -7.7 was the same
  corpus median on the 1 s / p95 / plain reading, so this changes the scale, not the
  intent.
- The input normalisation (R_ref -18, 1 s plain) and the multiband gate stay on the 1 s
  reading: they are internal reference levels. The report prints both readings.
- Measured on this reading: the maintainer's Track B and Track A masters -5.2 / -5.4, gs3
  -7.4 / -8.3 (2.2 and 2.9 dB short). The gap is the dynamics: the maintainer's Ozone multiband simulated on the premasters takes about 3 to
  6 dB of peak GR on the typical loud hit (6 to 12 dB at most), where the engine allows
  2.5 to 3 dB; his "3 dB" is a meter reading, not a per-hit peak.

## 18. Built 2026-10-20: the meter reading, FET glue, frequency-dependent limiting

The maintainer's FL Patcher masters for Track B and Track A (`trackb-FLPatcher-reference.fst`,
`tracka-FLPatcher-reference.fst`, decoded with a private FL preset decoder; the files
were removed from the repo root afterwards, the values below are what they held) run
soothe2, Pro-Q 2, Ozone 11 Dynamics (4 bands, M/S, 2:1 everywhere, 10 dB knee, a 10:1
band limiter per band, about 0 dB band makeup), UAD 1176AE (2:1; Track A input -54,
output -3, mix 100 %), Saturn, Ozone 11 Maximizer (IRC mode 3, soft clip on, gain
+3 dB, ceiling -0.3) and Pro-L (gain +1.3 / +2.0 dB, ceiling -0.3). Maintainer, on the
Ozone multiband settings: they vary per project; the offline engine must earn its
keep by adapting, so they are not copied. Built, measured one step at a time on
Track A (maintainer's master -5.4 on his meter, crest 9.0):

| render | change | RMS (maintainer's meter) | crest | limiter held by |
|---|---|---|---|---|
| gs3 | §16 audible maxima | -8.3 | 11.4 | outlier cap |
| gs4 | 1. compressor budgets on the meter reading | -6.5 | 9.8 | outlier cap |
| gs5 | 3. FET-style glue | -6.2 | 9.4 | outlier cap |
| gs6 | 4. frequency-dependent limiter | -5.3 | 8.6 | average budget |

1. **The meter reading** (`envelope.stage_gr`, `meter_smooth`): a compressor's GR curve
   is averaged over 300 ms (the maintainer's meter integration) before the per-hit and
   loud-section reading, for the multiband compressor and the glue; the band limiter,
   clipper and limiter keep their per-sample peaks. Simulating the maintainer's Ozone
   settings on the premasters gave about 3 to 6 dB of per-hit peak GR where his meter
   reads about 3 (comp and limiter combined, maintainer). The per-sample clamps rose to
   safety values (multiband 12 dB, glue 10 dB); the outlier caps (multiband 4.5, glue
   4.0) are on the meter reading and reported, not enforced. The band limiter may take
   its 3 dB peak before the compressor goes to its exception target; the combined
   meter reading is reported against the 3 / 6 dB tally.
2. **FET-style glue** (maintainer open to it): 2:1, knee 3 dB, attack 0.1 to 0.8 ms by
   character (the 1176's 20 to 800 us range), releases unchanged; the causal detector
   still lets the first 1 to 2 ms of a transient through. The 30 ms glue had measured
   0.0 dB of crest change on both real tracks.
3. **Frequency-dependent limiter** (`Plan.speclim`, `multiband.design_speclim`,
   `planner.design_speclim`; `--no-speclim` turns it off): after the glue, before the
   clipper, a ceiling-only multiband on the multiband's crossovers. Per band and M / S,
   peaks more than 2 dB above the band's typical loud hit are limited in that band, by
   at most 6 dB (lookahead 2 ms, release 100 to 40 ms low to high; the lookahead is
   per band from the character since 2026-10-23, §7). On Track A it cut
   up to 6 dB in the top band (M and S) and 3 to 4 dB in band 3, where the stray
   peaks are; the limiter then came off its outlier cap.

Not built from the maintainer's chain: copying the Ozone settings (maintainer: they vary per
project), the band layout (4 bands at 120 / 1000 to 1200 / 4790 Hz), near-zero band
makeup, soothe2 and Pro-Q 2 (not decoded), and the Saturn state (identical on both
tracks, not decoded). Plan version 0.10.

## 19. Built 2026-10-20: the M/S top end of the input EQ

Track G (gs6) finished 2.3 dB/oct darker than the target with every input EQ section at
its +-3 dB bound. The maintainer's own input EQ on that track (screenshot, 2026-10-20) had
separate high shelves: mid about +3 dB at 12 to 13 kHz rising toward +4, side about
+6 there rising toward +8, both starting around 2 to 3 kHz. Maintainer ruling: allow M/S on
the top end, the side shelf up to +6 dB and the mid up to +4.5 dB, the shelf frequency
dependent on the material.

- `InputEqPlan.high_shelf_side` (stages/inputeq.py): when set, the shared sections
  (high-pass, low shelf, bell, band shelf) run on L and R as before and the high shelves
  run apart, `high_shelf` on M and `high_shelf_side` on S, in the same zero-phase or
  minimum-phase mode. A side shelf of 0 dB still makes the top end M/S (side flat, mid
  shelved); `None` keeps the shelf linked (old plans, no side data, a near-silent side).
  `response_db(plan, f, fs, channel)` reads either channel.
- Limits (plan.py): `INPUTEQ_HS_MID_MAX_DB` = 4.5 (the mid or linked shelf),
  `INPUTEQ_HS_SIDE_MAX_DB` = 6.0; cuts stay at -3 dB, the other sections at +-3.
- Fit (planner.py): the analysis now carries the M and S spectra
  (`plan_inputs.ltas_mid_db`, `ltas_side_db`). The mid shelf is fitted as before on the
  stereo spectrum, with its boost allowed to +4.5. `inputeq_design` adds the side's
  residual from the same fit line after aligning its level to the stereo spectrum over
  500 Hz to 2 kHz (`EQ_SIDE_ALIGN_HZ`; a side more than 30 dB down there,
  `EQ_SIDE_SILENT_DB`, gets none); `_fit_side_shelf` then fits only the side shelf's
  gain and corner (1 to 16 kHz, from the material) on the nodes from 1 kHz up, with the
  shared sections fixed. The solver's tone-rotation refit refits both shelves.
- Analysis caches made before this change have no side spectrum, so they plan a
  linked shelf until re-analysed (a fresh render directory re-analyses).

## 20. Built 2026-10-20: the clipper acts above the sub

Why: the whole-signal clipper was clipping sustained sub-bass, not transients. On sub-heavy
tracks the per-hit peak is the kick and bass waveform, so a 1 dB median fall put the threshold
inside the bass: Track C's drop had clip spans up to 633 ms (p90 136 ms), the clipper
engaged a third of the time, and the limiter did nothing (the master's crest 6.7 was the
clipper's work). A 1 to 2 dB soft clip on a sine is -27 to -21 dB THD where the limiter doing
3 dB measures -42 dB.

- `LimiterPlan.clip_split_hz` (planner: the lowest multiband crossover, the engine's own sub
  boundary, 80 to 160 Hz by key). The clipper acts on the part of the signal above it
  (`limiter.clip_hi_part`: the high band of a zero-phase second-order split at base rate,
  upsampled with the signal): spans are found on that part, the curve is applied to it and
  the change is added to the full signal, so the sub passes to the limiter untouched. The
  threshold is solved on that part's per-hit output peak (`solver.clip_threshold_for_average_peak`
  runs on the high part); the proxy limiter clips the high part too. The record and the
  process log say "output peak above N Hz". Plans without the field render as before.
- Measured on the saved plans' drop excerpts, re-solving the clipper only (same limiter gain):

| track | clipped time (20 s) | span p90 / max (ms) | full-signal per-hit fall | clipper's change below 150 Hz, re signal | limiter GR max |
|---|---|---|---|---|---|
| Track C, full-band | 6.7 s | 136 / 633 | 1.20 dB | -57.9 dB | 0.0 |
| Track C, split at 80 Hz | 3.8 s | 72 / 158 | 0.23 dB | -62.9 dB | 1.0 |
| Track G, full-band | 5.1 s | 46 / 166 | 1.43 dB | -51.8 dB | 3.0 |
| Track G, split at 123 Hz | 1.7 s | 25 / 91 | 0.23 dB | -58.9 dB | 4.7 |

  The clipper now takes its 1 dB from the transient part only and the limiter takes the sub
  peaks; at the same gain the meter reads 0.06 to 0.13 dB lower, and in a full solve the
  limiter's budget may hold the gain a little more. Open (maintainer, by ear): the per-sample cap
  (3 dB, the architect's §16 value; the maintainer's §3 table said 1.5) and whether the split
  should sit higher than the key crossover on bass-heavy material (Track C's 80 Hz split
  still leaves 72 ms spans in the 80 to 200 Hz bass).

- Full renders with the split (same controls as before; `trackc_gs8`, `trackg_gs8`):

| track | meter RMS | crest | clipper avg / outlier | limiter avg / outlier | LRA | largest section move |
|---|---|---|---|---|---|---|
| Track C gs6 (full-band) | -4.10 | 6.7 | 1.0 / 2.7 | 0.0 / 0.0 | 5.0 | 5.6 LU |
| Track C gs8 (split 80 Hz) | -4.10 | 7.0 | 1.0 / 3.1 | 0.1 / 0.9 | 5.5 | 4.6 LU |
| Track G gs7 (full-band) | -4.55 | 7.3 | 1.0 / 3.0 | 3.0 / 3.3 | 5.6 | 8.3 LU |
| Track G gs8 (split 123 Hz) | -4.82 | 7.6 | 1.0 / 3.0 | 3.0 / 4.1 | 6.1 | 7.8 LU |

  Same loudness on Track C with 0.35 dB more crest and the limiter finally doing the peak
  work; Track G 0.27 dB quieter (the limiter's average budget clamps the gain once it carries
  the sub peaks), 0.3 dB more crest, 0.5 LU less contrast loss on both. Listening decides.
- Also 2026-10-20: the limiter's re-limit holds the 4x BS.1770 Annex 2 envelope as well as
  8x and 32x (`limiter._peak_envelopes`), so the file's "worst" reading sits at or below the
  ceiling by construction; the solver's ceiling render becomes rare. Found by the CLI test
  after the split: with the sub reaching the limiter, the Annex 2 reading sat 0.0003 dB over.
  The two gs8 renders predate this change (their ceiling render handled it).
