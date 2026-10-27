"""Planner: analysis + user controls -> Plan (spec 1, 3.2, 3.5, 3.6).

Contract:
- `analyse(x, fs, crossovers=None, bpm=None) -> dict` is the analysis the
  planner expects (docs/gain-staging.md 2 item 1, 4): the premaster's
  loudest-section 1 s RMS is measured (meter.rms_1s), the gain that brings
  it to R_REF_DBFS is applied (`input_gain`: 0 dB when the reading is not
  finite), and `features.analyse` runs on the normalised audio, so every
  analysis-derived level the planner turns into a threshold (multiband
  thresholds and ceilings from the crest pass) and the glue threshold (measured on the audio entering the glue
  stage) already account for the gain; the output of the chain is then
  independent of the premaster's level up to float rounding. meta["input"]
  records gain_db, r_ref_dbfs and measured_dbfs. Crossovers: the given ones
  (rule "explicit"), else `key_crossovers` of the key the analysis finds
  before its first band split (features.analyse fcs_from_key; rule "key").
- `key_crossovers(tonic, base=bands.DEFAULT_CROSSOVERS)`: docs/gain-staging.md
  4. p = the lowest frequency of the tonic's pitch class at or above
  KEY_PARTIAL_MIN_HZ (80 Hz; equal temperament, A4 = 440 Hz), so p lies in
  [80, 160); fc1 = p / 2^KEY_MARGIN_OCT (one third of an octave below p, maintainer
  2026-10-18), clipped to KEY_XOVER_RANGE_HZ (80 to 160 Hz); the others keep
  the base's ratios to its lowest (x4, x16, x64 for the default). An unknown
  or unparsable tonic gives the base. For E, F, F# and G the clip binds (p / 2^(1/3)
  is 65 to 78 Hz), so the crossover sits at 80 Hz, less than a third of an
  octave below p. `crossover_choice` is the record (crossovers, rule, tonic,
  p) in plan_record["crossovers"].
- `plan_from_analysis(analysis, controls, x=None) -> Plan`. `analysis` is the
  full dict returned by `analyse` or `features.analyse` (its `plan_inputs`,
  plus `meta` for fs / length / crossovers / input gain and `crest` for
  per-band peak levels). `controls` is a `Controls`. Every plan section is
  filled: input (from meta["input"]; None for an analysis made without
  normalisation), inputeq (None under preserve_mix), notch, sidestrip,
  multiband, saturate (None under preserve_mix), glue, keyeq (disabled),
  limiter, output. tone is None: the static input EQ replaces the FIR tone
  stage (docs/gain-staging.md chunk 2); stages/tone.py and TonePlan are kept,
  unused by the planner. The analysis (and so the input EQ fit) runs on the
  normalised premaster, and the EQ is a linear, time-invariant filter after
  a pure gain, so the fitted EQ does not depend on the premaster's level.
  Dynamics (docs/gain-staging.md chunk 3): the multiband carries its budget
  record (multiband.default_budget, makeup mode from controls.makeup), the
  glue its mode (controls.glue_mode; FET-style 2:1), the limiter the whole-signal
  clipper target (CLIP_TARGET_DB, on the output peak) or, with the kick
  detective, the old onset clipper, and its peak-GR budget LIMITER_MAX_GR_DB
  with the RMS target (chunk 4); the timing comes from the character preset
  (character_settings, the per-band lookahead of the band ceilings and of the
  frequency-dependent limiter included; auto by tempo, `character_choice`: a detected tempo, already
  octave-checked by the analysis, never resolves to slow). `realise` solves them in
  chain order on each stage's real input (multiband.design_budget, design_glue, design_limiter);
  `budget_report` collects per-stage peak GR against budget and the warnings.
  Average-peak reading (maintainer, 2026-10-19): Plan.hits (`plan_hits`: the analysis
  beat grid, which follows the user's --bpm, when it is informative or the user gave a
  tempo, one hit per beat at the plan's octave-checked tempo; else 400 ms frames) gives
  each stage its hits, and `stage_hits(plan, y, fs)` the hits of the audio y entering
  that stage and their loud section (envelope.hits_for).
  The loud section is frozen on the stage-1 input (sprint 2026-10-21 D2): `freeze_hits(plan,
  pre, fs)` writes the loud-hit mask of the multiband's input into Plan.hits.loud
  (`realise` and the solver's Chain call it before the multiband is designed), and
  stage_hits then hands every stage that mask, so the glue, the frequency-dependent limiter,
  the clipper and the limiter budget the same hits as the multiband instead of a loud set
  re-derived on their flatter inputs (on Track G, 185 of 561 hits at stage 1, 272 on the
  master: build hits entered the set and diluted the median). A plan whose hits carry no
  mask (saved before) derives the loud set per stage as before. Every budget is then solved on the average peak GR (the
  median over the loud hits of each hit's maximum GR); the compressors (multiband,
  glue) read their GR as a meter shows it (envelope.stage_gr: the 300 ms average),
  the clipper and limiter on their peaks. Outlier caps: multiband 4.5 and glue 4 dB
  (meter reading, reported), clipper 3 dB per sample (CLIP_SAMPLE_CAP_DB), limiter
  5 dB (LIMITER_OUTLIER_CAP_DB). Plan.hits None or reading "max": the whole track is
  one hit, the chunk 3 maximum reading. With limiter.speclim_allow_db set, `realise`
  also designs the frequency-dependent limiter (`design_speclim`) on the glue's output.
  Deterministic: the same inputs give an identical plan.
- Envelopes that are a function of the audio entering a stage (multiband
  compressor and ceiling envelopes, glue gain) cannot come from the analysis
  alone. Without `x` they are neutral (0 dB) and the plan carries the design
  record (thresholds, attack / release, ceilings, glue threshold / timing):
  an *unrealised* plan. With `x` (the premaster, (n, 2), at analysis fs) the
  plan is realised by `realise`: input gain, notch and sidestrip are rendered, the
  multiband envelopes designed on their output (multiband.design_from), the
  chain run to the glue input and the glue envelope designed there
  (glue.design). The solver realises with its own stage cache.
- Notches (spec 3.3, 3.5 item 1): one per analysis notch with flag
  "resonance" or "musical". Depth G(t) = -depth_scale * min(6, NOTCH_FRACTION
  * (excess(t) - 6 dB)) dB: half of the excess above the analysis detection
  threshold, capped at the stage's -6 dB, scaled by the analysis depth scale
  (1 resonance, 0.5 from 80 to 150 Hz, 0.25 musical). On the real premasters
  the median excess of a detection is 8 to 17 dB, so half the raw excess
  would pin nearly every notch at the -6 dB cap. excess(t) lives on the 100 ms analysis grid; it is linearly
  interpolated to 1 ms, smoothed by a triangular kernel of 2 x
  NOTCH_SMOOTH_MS, then quantised to NOTCH_QUANT_DB steps (each step is a
  1 ms PCHIP ramp of 0.5 dB). The quantisation, and kick onsets snapped to
  the 1 ms grid with the hold widened to KICK_HALF_MS, keep the number of
  distinct control rows the notch stage's composite clamp evaluates small: with
  smooth unquantised envelopes every 1 ms row is distinct and a 10 s excerpt
  with 8 notches did not finish in 2 minutes. Q: the measured Q is used only
  when the analyser's own fit test holds (`q_fit_ok`: finite, uncertainty below
  half the Q, features/resonance.py), else NOTCH_Q_DEFAULT; then clipped to
  [NOTCH_Q_LO, NOTCH_Q_HI] (the analysis reads Q high, workstream C hand-over).
  Channel awareness: the plan's Notch has no channel field and the notch stage
  cuts L and R identically, i.e. M and S together. A channel "both" or "M"
  notch is rendered (an M-only notch also cuts S at f0: S carries less energy
  there by construction of the attribution); a channel "S" notch is not
  rendered (cutting M where only S rings is the error audit 04 A4 was about)
  and is listed in `notch_report` with rendered False. This policy is
  `notch_rendered()`, the one place to change when Notch gains a channel.
  Defensive rules (NOTCH_SUSPECT_*, NOTCH_MUSICAL_GUARD_CENTS, NOTCH_MAX,
  KICK_MIN_RATE_HZ, in one block of constants, to relax when the analysis is
  fixed): a candidate whose peak excess exceeds 20 dB or whose spans cover
  more than 60 % of the track is suspect, logged and not rendered; no notch
  within 50 cents of any octave of the key's tonic or its fifth; within 50
  cents of a bar's chord root the excess is zeroed in that bar's analysis
  frames before interpolation and smoothing, so the guard ramps like the rest
  of the envelope (over the whole track the bar roots cover 6 to 10 of the 12
  pitch classes on the real premasters, so a whole-track guard would block most
  of the spectrum). Eligible candidates are grouped and merged first (the
  grouping notch.merge_candidates uses: half-cut band overlap), each merged
  notch ranked by its members' peak excess over unguarded frames, then at most
  NOTCH_MAX merged notches rendered (audit 07 A10). Every candidate is reported
  with rendered / reason / merged_into, peak excess and span share, and the
  rendered merged notches separately (`plan_notches` third value). The kick mask
  (notch.kick_mask) comes from `usable_kick_onsets`: none when the list is
  empty or sparser than KICK_MIN_RATE_HZ (0.5 kicks/s) over the track.
- Sidestrip: controls.subside_hz, else 100 Hz when the sub band's
  decoherence is below 0.1, else the analysis suggestion; clipped to
  [60, 150] Hz (spec 3.5 item 2).
- Multiband (spec 3.5 item 3): per band and M / S, a 400 ms frame peak level
  P(t) = rms_dbfs + crest_db from the analysis crest pass (gated frames).
  threshold = p90(P) - MB_THRESHOLD_BELOW_P90_DB at ratio MB_RATIO (about 1 dB
  of compressor GR at the p90 frame peak; the stage caps GR at 3 dB). Ceiling
  = p99(P) + MB_CEILING_ABOVE_P99_DB (a transient catcher on the loudest 1 %
  of frames). Attack / release per band from MB_ATTACK_MS / MB_RELEASE_MS.
  Makeup: mid 0 dB on every band (tone carries the spectral target); side
  makeup per the maintainer's rule (MultibandPlan.side_makeup: +2 dB x (1 -
  decoherence) on bands centred at or above 2 kHz, 0 below), decoherence from
  the analysis (finite by contract).
- Tone (spec 3.5 item 4): target slope = REFERENCE_SLOPE + tilt over 100 Hz to
  10 kHz (stereo LTAS). The node deficit is the rotation that turns the
  measured fit line into the target line about 1 kHz: delta (log2 f / 1 kHz),
  delta = target - measured slope, held at its 10 kHz value above 10 kHz. The
  sub (nodes below 100 Hz) is its own target: it holds the 100 Hz deficit (it
  does not rotate with the slope) and is pulled back into +-SUB_SHELF_MAX_DB of
  the target line if the predicted output shelf would leave that window. The
  nodes are stored unclipped; the tone stage applies the +-3 dB bound to the
  curve (TONE_MAX_DB), so the solver's rotation acts on the real deficit.
  Static over time (the stage redesigns per 0.34 s block only when the
  envelope varies; the planner never drives it at 1 ms). Since chunk 2 the
  deficit feeds the input EQ fit through `deficit_curve`; no TonePlan is
  rendered (`tone_deficit` stays as the record in plan_record["tone"]).
- Input EQ (docs/gain-staging.md section 2 item 2 and section 10, chunk 2):
  the stages/inputeq.py sections fitted to the same whole-track target. Fit
  nodes: the stereo LTAS 1/3-octave centres from 30 Hz to 18 kHz (nodes more
  than EQ_GATE_DB below the loudest are left out). Deficit D = the tone
  deficit (`deficit_curve`: rotation onto the target slope about 1 kHz plus
  the solver's rotation, sub held and pulled into +-SUB_SHELF_MAX_DB) minus,
  from 100 Hz up, the node's residual from the measured OLS line (so the
  low-mid build-up and the top-end roll-off count; the residuals have zero
  slope over 100 Hz to 10 kHz, so D's slope is the rotation's); the rotation
  cuts no node by more than EQ_LOW_ROT_CAP_DB (T2: brightness from the top, not
  from cutting the lows). High-pass: 25 Hz, Q = `hpf_q` of the predicted sub
  shelf (0.707 up to the window, down to 0.5 at EQ_HPF_Q_SPAN_DB above it; the
  Q > 0.707 branch is gone, T11); only its extra cut against the
  Butterworth Q counts in the fit (the baseline is the maintainer's rumble
  filter). Low mids (`low_mid_run`, T3): when the largest residual over 100 to
  500 Hz stands more than EQ_BELL_GATE_DB above its neighbours, the run of
  nodes around it is measured (EQ_NODE_OCT per node); a run of more than
  EQ_BAND_SHELF_OCT gets the band shelf (cut only, corners fitted, at most
  EQ_BAND_SHELF_MAX_OCT wide, 24 dB/oct), a narrower one the bell; either cuts
  at most the excess. The top end (T12): a presence shelf (corner
  EQ_PRES_FC_HZ) and an air shelf (EQ_AIR_FC_HZ), 6 dB/oct, each boost capped by
  `top_caps` (T4: the deficit in its own range, none where the range has no
  node above the gate, the noise headroom of the quiet-passage floor plus the
  master's predicted gain against noise.FLAG_DB; together within that
  headroom). The side gets its own presence and air shelves
  (`_fit_side_shelves`, at most the mid shelf + EQ_SIDE_OVER_MID_DB plus the
  side's own extra deficit, none at a top-band decoherence of
  EQ_SIDE_DECOHERENT or more). Least squares (scipy least_squares, trf,
  bounded, deterministic multi-start) over the low shelf gain and corner (Q
  EQ_LOW_SHELF_Q), the two top shelves' gains and corners (at most 0.45 fs)
  and, when enabled, the bell gain, centre and Q or the band shelf gain, low
  corner and width in octaves, every gain within +-3 dB (the profile's
  bounds; bell and band shelf <= 0). The fitted values are rounded (0.1 dB,
  1 Hz, the top shelves' corners 10 Hz, 0.01 Q); a section whose gain rounds to 0 is off; `at_bound` names the
  sections at their bound or cap. The record (`fit_inputeq`)
  holds the deficit, the fit, residual RMS and maximum, the predicted slope,
  the sections and one readable line each ("low shelf +1.2 dB at 120 Hz").
  InputEqPlan.design keeps the fit inputs so `refit_inputeq` (the solver's
  tone_rotation lever) refits with a rotation added to the target.
- Saturation (docs/gain-staging.md 6, chunk 4: harmonics are the inverse of the
  slope): per band b >= 1 (the sub never), drive_b = the band's deficit below
  the target slope line (`band_deficits`: the mean over the band's input EQ fit
  nodes of the target line minus the LTAS, on the analysed spectrum, i.e.
  before the input EQ), rounded to SAT_STEP_DB (0.5 dB), at most
  SAT_MAX_DRIVE_DB (2 dB); bands on or above the line get none
  (`saturation_drive_db`). M and S alike, except the high-mid S band, which
  keeps the width drive at SAT_MAX_DRIVE_DB. Wet share 1 on every band. The
  record (`plan_saturate`, plan_record["saturation"]) holds each band's
  deficit and drive. The input EQ still corrects the level, so a dull band
  gets both.
- Glue (spec 3.5 item 6, audit 07 A9): the unrealised plan carries threshold
  None; `realise` and the solver's Chain solve it for the glue's budget
  (glue.design_budget, serial or parallel) on the audio entering the glue
  stage; without a budget record the threshold rule `glue_threshold_audio`
  (p95 of the 400 ms frame peak minus 6 dB) applies. FET-style: ratio 2
  (GLUE_RATIO), knee 3 dB, attack 0.1 to 0.8 ms and release from the character
  preset, makeup 0 dB, clean curve.
- Limiter (spec 3.5 item 7, chunk 4): ceiling controls.ceiling_dbtp
  (CEILING_DBTP, -0.3 dBTP; STREAMING_CEILING_DBTP, -1 dBTP, with
  controls.streaming); target_rms_dbfs = controls.rms_target (the master's
  RMS on the maintainer's meter, meter.rms_meter: 300 ms, AES17, loudest passage);
  starting gain = the target minus R_ref and the AES17 offset (the analysed audio's
  level after normalisation), which `design_limiter`
  replaces by the gain the proxy limiter needs for the target and the solver
  by its calibrated solve; the budget clamps it (`clamp_limiter_gain`).
  Clipper: the whole-signal clipper, acting on the signal above the lowest crossover
  (LimiterPlan.clip_split_hz; the sub passes to the limiter untouched, 2026-10-20),
  solved for CLIP_TARGET_DB of output-peak
  reduction (`solve_clipper`), or with the kick detective (controls.kick_detective,
  None = auto: on when the character resolves to slow or transparent,
  `kick_detective_choice`) clip depth CLIP_DEPTH_DB at `usable_kick_onsets`
  (clip depth 0 when none).
- Output (docs/gain-staging.md 3 "Ceiling"): 24-bit PCM, rounded, by default;
  16-bit PCM with TPDF dither under controls.streaming. Targets.true_peak_max_dbtp
  is the same ceiling, read on the quantised master.
- Under preserve_mix: tone and saturate are None, side makeup is kept (the
  solver's LTAS check bounds it), mid makeup 0.
- `targets_from(analysis, controls) -> Targets`: the solver's constraint
  targets (spec 3.6), from the same inputs. The RMS is met within +-rms_tol_db of the
  Loudness control and nowhere else (maintainer, S4, 2026-10-21: the Loudness control is the
  trade between loudness and section contrast, not a hidden acceptance floor). `crest_exempt` lists the S views
  of bands the sidestrip removes on purpose (upper edge <= 2 fc_sub), only when
  controls.side_band0_exempt is set (off by default, audit 07 F17): on
  Track A, band 0 S read a 4.9 dB crest "loss" with the limiter not acting at
  all, which made every limiter drive infeasible.
- Controls: crossovers_hz None (default) means the analysis's own crossovers
  (key-locked when `analyse` found a key); a tuple must equal them.
  streaming selects the -1 dBTP / 16-bit TPDF export. threads (at least 1, default
  parallel.DEFAULT_THREADS) is the engine's thread count (eluate.parallel), recorded
  with the controls; the plan and master do not depend on it. tilt_db is read as
  dB/oct added to the reference slope (spec 3.5 item 4 "reference_slope +
  user_tilt", 3.6 "reference + tilt +- 0.3 dB/oct"), bounded to +-3 (the
  brief's "+-3 dB"). target_rms (dBFS, below 0), default
  DEFAULT_TARGET_RMS_DBFS (-4.1 dB on the maintainer's meter, meter.rms_meter).
  crest_mode in {"loss", "report", "floor"} (spec 3.6 draft 0.5; default
  "report" per the maintainer's budget ruling of 2026-10-18: per-band crest loss
  is reported and never fails a render; "loss" and "floor" stay available;
  "floor" needs crest_floors_db, one per band, no calibrated default); side_band0_exempt (default False);
  tp_meter "worst" (max of the 4x Annex 2, 8x and 32x readings) or "8x" (max of
  8x and 32x); None (default) reads "8x" at 24 bits and "worst" under streaming
  (sprint 2026-10-21 S6, R7g: the Annex 2 reading traded 0.2 to 0.5 dB of loudness
  for a compliance figure only the lossy-encode path needs).
- Profiles (PROFILES, Controls.profile; sprint 2026-10-21 sections 5 and 6): every
  number above that differs by material comes from the profile: "electronic" (the
  values above), "live" (phase 1, uncalibrated: lower budgets, the clipper only under
  the fast and aggressive characters, saturation off, M/S-linked multiband, side strip
  at 60 Hz, a narrow input EQ with a 25 Hz Butterworth high-pass, the tone target the
  mix's own slope + Tilt (target_slope) with no sub pull, the residual corrected only
  inside SLOPE_RANGE_HZ and the fit's error weighted by audibility (audibility_weights),
  the slope reported but not solved for) and "stems" (the sum of a stem set,
  `eluate master --stems`: tighter ranges). Electronic plans are unchanged by the table.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from eluate import bands, envelope, meter, parallel
from eluate.plan import (CLIP_MAX_GR_DB, CONTROL_RATE_HZ, MAKEUP_MODES, INPUTEQ_BELL_FC_RANGE, INPUTEQ_BELL_Q_RANGE,
                            INPUTEQ_HIGH_SHELF_FC_HZ, INPUTEQ_HPF_FC_HZ, INPUTEQ_HPF_Q_RANGE,
                            INPUTEQ_HS_MID_MAX_DB, INPUTEQ_HS_SIDE_MAX_DB,
                            INPUTEQ_LOW_SHELF_FC_RANGE, INPUTEQ_MAX_DB, NOTCH_MAX_CUT_DB,
                            NOTCH_Q_MIN, EqBandShelf, EqBell, EqHighPass,
                            EqHighShelf, INPUTEQ_BAND_SHELF_LO_RANGE,
                            INPUTEQ_BAND_SHELF_WIDTH_OCT,
                            EqLowShelf, Envelope, GluePlan, InputEqPlan, InputPlan, KeyEqPlan,
                            HitsPlan, LimiterPlan, MultibandPlan, Notch, NotchPlan, OutputPlan, Plan,
                            SaturatePlan, SidestripPlan, band_centres, control_length,
                            tone_nodes)
from eluate.stages import glue as glue_stage
from eluate.stages import multiband as mb_stage
from eluate.stages import notch as notch_stage

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- constants
REFERENCE_SLOPE = -4.5          # dB/oct, PSD, 100 Hz to 10 kHz (spec 3.5 item 4)
TILT_MAX = 3.0
SLOPE_RANGE_HZ = (100.0, 10000.0)
PIVOT_HZ = 1000.0
SUB_SHELF_MAX_DB = 3.0
SUB_CENTRE_HZ = 50.0            # representative sub frequency (30 to 90 Hz shelf)

EQ_FIT_RANGE_HZ = (30.0, 18000.0)   # 1/3-octave LTAS nodes the input EQ is fitted on
EQ_GATE_DB = 60.0                   # nodes this far below the loudest are left out
EQ_HPF_Q_BUTTERWORTH = 0.707
EQ_HPF_Q_SPAN_DB = 6.0              # sub excess (deficit) beyond SUB_SHELF_MAX_DB at which
                                    # Q hits 0.5 (1.0)
EQ_LOW_SHELF_Q = 0.707              # 0.5 would be nearly the 6 dB/oct shape at +-3 dB
EQ_BELL_GATE_DB = 3.0               # low-mid excess above its neighbours that enables the bell
EQ_BELL_RUN_DB = 1.5                # nodes within this of the peak belong to its run
EQ_BELL_NEIGHBOUR_OCT = 1.0         # neighbours: nodes up to this far outside the run
EQ_BAND_SHELF_OCT = 1.0             # excess run wider than this: the band shelf, not the bell
EQ_BAND_SHELF_MAX_OCT = 1.5         # the band shelf's widest (sprint 2026-10-21 T3)
EQ_NODE_OCT = 1.0 / 3.0             # width of one LTAS node
EQ_LOW_MID_RUN_HZ = (100.0, 500.0)  # nodes the low-mid excess run may span (T3)
EQ_LOW_ROT_CAP_DB = 1.5             # the rotation's largest cut below the pivot (T2)
EQ_PRES_FC_HZ = (1000.0, 6000.0)    # presence shelf corner (T12)
EQ_AIR_FC_HZ = (10000.0, 16000.0)   # air shelf corner (T12)
EQ_PRES_DEFICIT_HZ = (2000.0, 8000.0)   # the presence shelf's own range (its boost cap)
EQ_AIR_DEFICIT_HZ = (10000.0, 16000.0)  # the air shelf's own range (its boost cap)
# Audibility weighting of the input EQ fit (maintainer, 2026-10-26: shelves calibrated to how
# the ear hears at a mastering level, not to raw dB). ISO 226:2003 equal-loudness tables
# as in the IoSR Surrey toolbox (iosr.auditory.iso226). The standard's low end is measured
# only up to 80 phon, so the contour is 80 phon (the maintainer monitors at about 85 dB).
ISO226_F = (20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
            1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500)
ISO226_AF = (0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349, 0.330, 0.315,
             0.301, 0.288, 0.276, 0.267, 0.259, 0.253, 0.250, 0.246, 0.244, 0.243, 0.243,
             0.243, 0.242, 0.242, 0.245, 0.254, 0.271, 0.301)
ISO226_LU = (-31.6, -27.2, -23.0, -19.1, -15.9, -13.0, -10.3, -8.1, -6.2, -4.5, -3.1, -2.0,
             -1.1, -0.4, 0.0, 0.3, 0.5, 0.0, -2.7, -4.1, -1.0, 1.7, 2.5, 1.2, -2.1, -7.1,
             -11.2, -10.7, -3.1)
ISO226_TF = (78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4, 11.4, 8.6, 6.2,
             4.4, 3.0, 2.2, 2.4, 3.5, 1.7, -1.3, -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3)
EQ_WEIGHT_PHON = 80.0
EQ_WEIGHT_RANGE = (0.05, 2.0)       # weights clipped: no band ignored, none dominant
EQ_SIDE_OVER_MID_DB = 2.0           # a side shelf at most this above its mid shelf (T4)
EQ_SIDE_DECOHERENT = 0.5            # top-band decoherence from which S gets no boost (T4)
EQ_BELL_Q_START = 1.4
EQ_HS_MAX_FS_SHARE = 0.45           # high-shelf corner at most 0.45 fs
EQ_SIDE_ALIGN_HZ = (500.0, 2000.0)  # where the side spectrum is level-aligned to the stereo
EQ_SIDE_SILENT_DB = 30.0            # a side this far below the stereo there gets no shelf
EQ_LS_STARTS_HZ = (80.0, 200.0)
EQ_PRES_STARTS_HZ = (4000.0, 1500.0)
EQ_AIR_STARTS_HZ = (12000.0,)

# Loudness target (docs/gain-staging.md 3, 9, 17): the maintainer's meter (meter.rms_meter:
# Ozone RMS bar, 300 ms, AES17 so a full-scale sine reads 0, read at the loudest passage =
# the p95), the median of the maintainer's 18 masters by default (-4.06; range -5.0 to -3.7).
DEFAULT_TARGET_RMS_DBFS = -4.1
TARGET_RMS_TOL_DB = 0.3
# S5: the gain asked of the limiter is capped this far above the budget's maximum gain, so
# a failed proxy solve (the drive range's end) does not become the warning's "needed"
LIMITER_REQUEST_OVER_DB = 6.0

NOTCH_FRACTION = 0.5
NOTCH_SMOOTH_MS = 200.0
NOTCH_QUANT_DB = 0.5
KICK_HALF_MS = 20.5            # onsets snapped to the 1 ms grid: +-0.5 ms of hold restored
NOTCH_Q_DEFAULT = 8.0
NOTCH_Q_LO = 2.0
NOTCH_Q_HI = 16.0
NOTCH_EXCESS_FLOOR_DB = 6.0     # the analysis detection threshold (resonance.EXCESS_DB)

# Defensive limits while the analysis over-reports resonances and under-reports kicks
# (2026-10-18: 37 to 68 candidates per real track, excess up to 66 dB, mostly held
# bass and chord tones; one kick every 2 to 10 s). Relax here when workstream C lands.
NOTCH_MAX = 8                    # notches rendered per track, ranked by peak excess
NOTCH_LOW_HZ = 500.0             # below this a notch's Q is at most NOTCH_LOW_Q_MAX (T7)
NOTCH_LOW_Q_MAX = 4.0            # a Q 16 bandpass at 120 Hz pre-rang 140 ms
NOTCH_SUSPECT_EXCESS_DB = 20.0   # peak excess above this: suspect, not rendered
NOTCH_SUSPECT_SPAN = 0.6         # spans covering more than this share of the track: suspect
NOTCH_MUSICAL_GUARD_CENTS = 50.0  # no notch this close to the tonic, fifth or a bar's chord root
KICK_MIN_RATE_HZ = 0.5           # fewer kicks per second than this: no kick mask, no clip onsets
SUBSIDE_DEFAULT_HZ = 100.0       # sidestrip default when the sub is not decoherent
SUBSIDE_DECOHERENCE_MIN = 0.1    # sub-band decoherence below which the default applies

SUBSIDE_RANGE_HZ = (60.0, 150.0)

MB_RATIO = 2.0
MB_THRESHOLD_BELOW_P90_DB = 2.0
MB_CEILING_ABOVE_P99_DB = 0.0
MB_ATTACK_MS = (30.0, 20.0, 10.0, 5.0, 3.0)      # sub .. air; extra bands use the last
MB_RELEASE_MS = (200.0, 150.0, 100.0, 80.0, 60.0)

SAT_MAX_DRIVE_DB = 2.0           # docs/gain-staging.md 6: never above 2 dB on a mastering pass
SAT_STEP_DB = 0.5

GLUE_BELOW_PEAK_DB = 6.0
GLUE_MIX = 0.35                  # the solver's starting mix for a glue without a budget
# FET-style glue (maintainer, 2026-10-19: his 1176AE at 2:1; a 1176's attack is 20 to 800 us,
# so it catches the start of a transient that a slow bus compressor lets through)
GLUE_RATIO = 2.0
GLUE_KNEE_DB = 3.0
GLUE_ATTACK_MS = 0.4
GLUE_RELEASE_MS = 300.0
GLUE_MODES = ("serial", "parallel")

CEILING_DBTP = -0.3              # 24-bit default (maintainer, 2026-10-18; -0.1 is too high)
STREAMING_CEILING_DBTP = -1.0    # with 16-bit TPDF output (--streaming)
CLIP_DEPTH_DB = 1.5              # kick-detective clipper depth
CLIP_TARGET_DB = 1.0             # whole-signal clipper output-peak reduction (at most 1.5)
CLIP_WARN_DB = envelope.AVERAGE_WARN_DB   # a clipper this far short of its target warns
LIMITER_MAX_GR_DB = 3.0          # limiter average peak GR budget (docs/gain-staging.md 3)
# audible maxima (docs/dynamics-macros.md step 1, maintainer 2026-10-19): the clipper takes
# the hit spread first, per sample up to this, then the limiter up to its outlier cap
CLIP_SAMPLE_CAP_DB = 3.0
LIMITER_OUTLIER_CAP_DB = 5.0
# frequency-dependent limiting ahead of the clipper (maintainer, 2026-10-19; the model is the
# Ozone Maximizer's spectral limiting): per band and M / S, peaks more than ALLOW above
# the band's typical loud hit are limited in that band, by at most MAX_CUT
SPECLIM_ALLOW_DB = 2.0
SPECLIM_MAX_CUT_DB = 6.0
# the lookahead of a limiter section without speclim_lookahead_ms (plans made before
# 2026-10-23); the planner takes it per band from the character preset
SPECLIM_LOOKAHEAD_MS = 2.0
SPECLIM_RELEASE_MS = (100.0, 80.0, 80.0, 80.0, 80.0)   # bands low to high (D6: at least 80 ms)
SPECLIM_TOP_FROM_HZ = 1600.0     # D6: bands from here up are allowed SPECLIM_TOP_ALLOW_DB
SPECLIM_TOP_ALLOW_DB = 3.5       # (at 2 dB it ducked the top band on 7 to 12 % of the hits)

# Character presets (docs/gain-staging.md 7; ms, bands low to high, extra bands use the
# last value). clip_knee_db 0 would be hard; every preset is soft, at least 0.5 dB (maintainer,
# 2026-10-23: a hard knee's aliasing, docs/quality-review-2026-10-22.md 3).
# band_lookahead_ms: the band ceilings' and the frequency-dependent limiter's lookahead
# (maintainer, 2026-10-23; quality review 1: at 2 ms a 3 to 6 dB cut on a 40 to 60 Hz band lands
# inside a tenth of a cycle, amplitude modulation heard as distortion at 100 to 500 Hz).
# auto: under 70 BPM slow, 70 to 140 moderate, above 140 fast.
CHARACTERS = ("auto", "transparent", "slow", "moderate", "fast", "aggressive")
CHARACTER_PRESETS = {
    "transparent": {"mb_attack_ms": (30.0, 25.0, 15.0, 10.0, 5.0),
                    "mb_release_ms": (300.0, 250.0, 200.0, 150.0, 120.0), "mb_knee_db": 10.0,
                    "glue_attack_ms": 0.8, "glue_release_ms": 600.0, "clip_knee_db": 1.5,
                    "limiter_attack_ms": 5.0, "limiter_release_ms": 200.0,
                    "band_lookahead_ms": (20.0, 10.0, 5.0, 2.0, 2.0)},
    "slow": {"mb_attack_ms": (30.0, 25.0, 15.0, 10.0, 5.0),
             "mb_release_ms": (300.0, 250.0, 200.0, 150.0, 120.0), "mb_knee_db": 6.0,
             "glue_attack_ms": 0.8, "glue_release_ms": 400.0, "clip_knee_db": 1.0,
             "limiter_attack_ms": 3.0, "limiter_release_ms": 150.0,
             "band_lookahead_ms": (20.0, 8.0, 4.0, 2.0, 2.0)},
    "moderate": {"mb_attack_ms": (20.0, 15.0, 10.0, 5.0, 3.0),
                 "mb_release_ms": (200.0, 150.0, 100.0, 80.0, 60.0), "mb_knee_db": 6.0,
                 "glue_attack_ms": 0.4, "glue_release_ms": 300.0, "clip_knee_db": 1.0,
                 "limiter_attack_ms": 2.0, "limiter_release_ms": 100.0,
                 "band_lookahead_ms": (15.0, 6.0, 3.0, 2.0, 2.0)},
    "fast": {"mb_attack_ms": (10.0, 8.0, 5.0, 3.0, 1.0),
             "mb_release_ms": (120.0, 100.0, 60.0, 40.0, 30.0), "mb_knee_db": 3.0,
             "glue_attack_ms": 0.2, "glue_release_ms": 150.0, "clip_knee_db": 0.5,
             "limiter_attack_ms": 1.0, "limiter_release_ms": 50.0,
             "band_lookahead_ms": (10.0, 5.0, 2.0, 2.0, 2.0)},
    "aggressive": {"mb_attack_ms": (5.0, 4.0, 3.0, 1.0, 0.5),
                   "mb_release_ms": (80.0, 60.0, 40.0, 30.0, 20.0), "mb_knee_db": 1.0,
                   "glue_attack_ms": 0.1, "glue_release_ms": 80.0, "clip_knee_db": 0.5,
                   "limiter_attack_ms": 0.5, "limiter_release_ms": 30.0,
                   "band_lookahead_ms": (6.0, 4.0, 2.0, 2.0, 2.0)},
}
AUTO_BPM_SLOW, AUTO_BPM_FAST = 70.0, 140.0
AUTO_DEFAULT = "moderate"        # auto without a tempo
KICK_DETECTIVE_AUTO = ("slow", "transparent")   # presets on which auto turns it on
LOW_BAND_RELEASE_MIN_MS = 25.0   # one 40 Hz cycle: the lowest band never releases faster
LIMITER_RELEASE_MIN_MS = 80.0    # D7: faster limiter releases ripple on sustained sub

# Input normalisation (docs/gain-staging.md 2 item 1, 9): the corpus median of the
# premasters' loudest-section 1 s RMS with their peak at -3 dBFS (44 premasters,
# -17.9 dBFS, rounded).
R_REF_DBFS = -18.0

# Key-locked lowest crossover (docs/gain-staging.md 4)
KEY_PARTIAL_MIN_HZ = 80.0        # the tonic partial is the lowest one at or above this
KEY_MARGIN_OCT = 1.0 / 3.0       # crossover this far below it (maintainer, 2026-10-18)
KEY_XOVER_RANGE_HZ = (80.0, 160.0)

CREST_MODES = ("loss", "report", "floor")   # spec 3.6 draft 0.5, maintainer ruling pending
TP_METERS = ("worst", "8x")
CREST_LIMITS_DB = (1.5, 2.0)    # sub, low; every higher band 3 dB (spec 3.6)
CREST_LIMIT_HIGH_DB = 3.0

# Profiles (sprint 2026-10-21 section 5; maintainer R8, 2026-10-21): every number that differs
# between electronic music and a live or acoustic recording. "live" is phase 1: its
# loudness default and crest tables are not calibrated on live masters yet (R9), and the
# report and the process log say so. None keeps the electronic (module) value.
PROFILES = {
    "electronic": {
        "calibrated": True,
        "target_rms_dbfs": DEFAULT_TARGET_RMS_DBFS,
        "mb_budget": {},                    # multiband.default_budget as it stands
        "ms_link": False,
        "side_makeup_db": 2.0,
        "saturate": True,
        "glue_net_db": None,                # glue.SERIAL_GR_DB
        "glue_attack_ms": None,             # the character preset's
        "clipper_presets": None,            # every preset
        "clip_target_db": CLIP_TARGET_DB,
        "clip_sample_cap_db": CLIP_SAMPLE_CAP_DB,
        "sat_max_db": SAT_MAX_DRIVE_DB,
        "kick": True,                       # kick detective (auto) and the notch kick mask
        "speclim_allow_db": SPECLIM_ALLOW_DB,
        "speclim_max_cut_db": SPECLIM_MAX_CUT_DB,
        "limiter_max_gr_db": LIMITER_MAX_GR_DB,
        "limiter_outlier_cap_db": LIMITER_OUTLIER_CAP_DB,
        "limiter_timing": None,             # the character preset's
        "sidestrip_hz": None,               # the key-dependent rule (subside_hz)
        "eq_bounds": None,                  # INPUTEQ_MAX_DB and the high shelves' maxima
        "eq_weighting": None,               # the fit's error unweighted (raw dB per node)
        "eq_residual_window": False,        # the residual corrects up to EQ_FIT_RANGE_HZ's top
        "eq_hpf": None,                     # INPUTEQ_HPF_FC_HZ, Q from the sub (hpf_q)
        "slope": "constraint",
    },
    "live": {
        "calibrated": False,
        "target_rms_dbfs": -10.0,           # uncalibrated until R9
        "mb_budget": {"comp_gr_db": 1.0, "comp_gr_max_db": 1.5, "total_normal_db": 1.5,
                      "limit_max_db": 1.5, "ratio_max": 1.5},
        "ms_link": True,                    # the image is the room: it must not move
        "side_makeup_db": 0.0,
        "saturate": False,
        "glue_net_db": 1.0,
        "glue_attack_ms": 20.0,
        "clipper_presets": ("fast", "aggressive"),   # maintainer: off unless asked for
        "clip_target_db": CLIP_TARGET_DB,
        "clip_sample_cap_db": CLIP_SAMPLE_CAP_DB,
        "sat_max_db": SAT_MAX_DRIVE_DB,
        "kick": False,                      # a floor tom or a double bass is not a kick
        "speclim_allow_db": 4.0,
        "speclim_max_cut_db": 3.0,
        "limiter_max_gr_db": 1.5,
        "limiter_outlier_cap_db": 3.0,
        "limiter_timing": "transparent",
        "sidestrip_hz": 60.0,               # maintainer: on, below 60 Hz (room ambience above stays)
        "eq_bounds": {"gain_db": 1.5, "hs_mid_db": 2.0, "hs_side_db": 2.0},
        "eq_weighting": EQ_WEIGHT_PHON,     # the fit's error weighted by audibility (ISO 226)
        "eq_residual_window": True,         # the residual corrects only inside SLOPE_RANGE_HZ
        "eq_hpf": {"fc_hz": 25.0, "q": EQ_HPF_Q_BUTTERWORTH},   # rumble only: flat above 63 Hz
        "slope": "report",                  # the mix's own slope is the target (Tilt moves it); no sub pull
    },
    # the sum of a stem set (WP6, section 6.2): tighter ranges, so a sum that cannot reach
    # the target within them is the signal that a stem needs correcting (phase B)
    "stems": {
        "calibrated": True,
        "target_rms_dbfs": DEFAULT_TARGET_RMS_DBFS,
        "mb_budget": {"comp_gr_db": 1.5, "comp_gr_max_db": 2.0, "total_normal_db": 2.0,
                      "limit_max_db": 2.0},
        "ms_link": False,
        "side_makeup_db": 2.0,
        "saturate": True,
        "glue_net_db": 1.0,
        "glue_attack_ms": None,
        "clipper_presets": None,
        "clip_target_db": 0.5,
        "clip_sample_cap_db": 2.0,
        "sat_max_db": 1.0,
        "kick": True,
        "speclim_allow_db": 3.0,
        "speclim_max_cut_db": 4.0,
        "limiter_max_gr_db": 2.0,
        "limiter_outlier_cap_db": 4.0,
        "limiter_timing": None,
        "sidestrip_hz": None,
        "eq_bounds": {"gain_db": 1.5, "hs_mid_db": 2.5, "hs_side_db": 3.0},
        "eq_weighting": None,
        "eq_residual_window": False,
        "eq_hpf": None,
        "slope": "constraint",
    },
}
PROFILE_NAMES = tuple(PROFILES)
USER_PROFILES = ("electronic", "live")   # the stems profile comes with --stems


# ---------------------------------------------------------------- controls

@dataclass
class Controls:
    """User controls (spec 3.2, 4). See the module contract for tilt units."""

    tilt_db: float = 0.0
    target_rms: float | None = None          # master RMS on the maintainer's meter (rms_meter)
    preserve_mix: bool = False
    crossovers_hz: tuple | None = None       # None: the analysis's (key-locked) crossovers
    subside_hz: float | None = None
    crest_mode: str = "report"               # "loss" | "report" | "floor" (spec 3.6)
    crest_floors_db: tuple | None = None     # per-band floors for crest_mode "floor"
    side_band0_exempt: bool = False          # exempt sidestripped S views from crest
    tp_meter: str | None = None              # "worst" of 4x/8x/32x, "8x"; None: by output
    streaming: bool = False                  # -1 dBTP ceiling, 16-bit TPDF output
    character: str = "auto"                  # docs/gain-staging.md 7 (CHARACTERS)
    bpm: float | None = None                 # the user's tempo; auto character reads it first
    makeup: str = "peak"                     # multiband makeup mode: peak | rms | auto
    glue_mode: str = "serial"                # serial | parallel
    kick_detective: bool | None = None       # clip only around kick onsets; None = auto
    speclim: bool = True                     # frequency-dependent limiting before the clipper
    profile: str = "electronic"              # PROFILES
    threads: int = parallel.DEFAULT_THREADS  # engine threads (eluate.parallel); no effect on the result

    def __post_init__(self):
        self.threads = int(self.threads)
        if self.threads < 1:
            raise ValueError("threads must be at least 1")
        if self.profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILE_NAMES}")
        if self.character not in CHARACTERS:
            raise ValueError(f"character must be one of {CHARACTERS}")
        if self.bpm is not None:
            self.bpm = float(self.bpm)
            if not (math.isfinite(self.bpm) and self.bpm > 0):
                raise ValueError("bpm must be positive")
        if self.makeup not in MAKEUP_MODES:
            raise ValueError(f"makeup must be one of {MAKEUP_MODES}")
        if self.glue_mode not in GLUE_MODES:
            raise ValueError(f"glue_mode must be one of {GLUE_MODES}")
        if not (math.isfinite(self.tilt_db) and abs(self.tilt_db) <= TILT_MAX):
            raise ValueError(f"tilt must lie in [-{TILT_MAX}, {TILT_MAX}]")
        if self.target_rms is not None:
            self.target_rms = float(self.target_rms)
            if not (math.isfinite(self.target_rms) and self.target_rms < 0.0):
                raise ValueError("target RMS must be finite and below 0 dBFS")
        if self.crossovers_hz is not None:
            self.crossovers_hz = tuple(float(f) for f in self.crossovers_hz)
        if self.subside_hz is not None and not self.subside_hz > 0:
            raise ValueError("subside must be positive")
        if self.crest_mode not in CREST_MODES:
            raise ValueError(f"crest_mode must be one of {CREST_MODES}")
        if self.crest_floors_db is not None:
            self.crest_floors_db = tuple(float(v) for v in self.crest_floors_db)
            if self.crossovers_hz is not None:     # else checked on the analysis's (resolve)
                _check_floors(self.crest_floors_db, self.crossovers_hz)
        if self.crest_mode == "floor" and self.crest_floors_db is None:
            raise ValueError("crest_mode 'floor' needs crest_floors_db (no calibrated default)")
        if self.tp_meter is None:
            self.tp_meter = "worst" if self.streaming else "8x"
        if self.tp_meter not in TP_METERS:
            raise ValueError(f"tp_meter must be one of {TP_METERS}")

    @property
    def rms_target(self) -> float:
        """The loudness target: target_rms, else the profile's default."""
        return self.target_rms if self.target_rms is not None \
            else PROFILES[self.profile]["target_rms_dbfs"]

    @property
    def prof(self) -> dict:
        return PROFILES[self.profile]

    @property
    def ceiling_dbtp(self) -> float:
        return STREAMING_CEILING_DBTP if self.streaming else CEILING_DBTP

    def output_plan(self) -> OutputPlan:
        return OutputPlan(16, "tpdf") if self.streaming else OutputPlan(24)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["crossovers_hz"] = None if self.crossovers_hz is None else list(self.crossovers_hz)
        if self.crest_floors_db is not None:
            d["crest_floors_db"] = list(self.crest_floors_db)
        d["rms_target"] = self.rms_target
        d["ceiling_dbtp"] = self.ceiling_dbtp
        return d


@dataclass
class Targets:
    """Solver constraints (spec 3.6), all from analysis + controls."""

    rms_dbfs: float = DEFAULT_TARGET_RMS_DBFS
    rms_tol_db: float = TARGET_RMS_TOL_DB
    slope_db_oct: float = REFERENCE_SLOPE
    slope_tol: float = 0.3
    preserve_mix: bool = False
    ltas_tol_db: float = 1.0
    ltas_centres_hz: list = field(default_factory=list)
    ltas_ref_db: list = field(default_factory=list)
    crest_limits_db: list = field(default_factory=list)
    true_peak_max_dbtp: float = CEILING_DBTP
    sections: list = field(default_factory=list)   # [{start, end, label, integrated_lufs}]
    section_tol_lu: float = 1.0
    crossovers_hz: list = field(default_factory=lambda: list(bands.DEFAULT_CROSSOVERS))
    crest_exempt: list = field(default_factory=list)   # [band, view index] pairs
    crest_mode: str = "report"
    crest_floors_db: list | None = None
    tp_meter: str = "worst"
    slope_mode: str = "constraint"         # "report" (live): measured and flagged only

    def to_dict(self) -> dict:
        return asdict(self)


def crest_limits(crossovers_hz) -> list[float]:
    nb = len(crossovers_hz) + 1
    return [CREST_LIMITS_DB[k] if k < len(CREST_LIMITS_DB) else CREST_LIMIT_HIGH_DB
            for k in range(nb)]


def _check_crossovers(analysis: dict, controls: Controls) -> None:
    a = [float(f) for f in analysis["meta"]["crossovers"]]
    if controls.crossovers_hz is not None and list(controls.crossovers_hz) != a:
        raise ValueError(f"controls crossovers {list(controls.crossovers_hz)} differ from the "
                         f"analysis crossovers {a}; re-run the analysis with them")


def resolve(analysis: dict, controls: Controls) -> Controls:
    """controls with crossovers_hz set to the analysis's (checked when given)."""
    _check_crossovers(analysis, controls)
    if controls.crossovers_hz is not None:
        return controls
    return replace(controls, crossovers_hz=tuple(float(f) for f in analysis["meta"]["crossovers"]))


def _check_floors(floors, crossovers_hz) -> None:
    if len(floors) != len(crossovers_hz) + 1:
        raise ValueError(f"crest_floors_db needs one value per band ({len(crossovers_hz) + 1}, "
                         f"from the crossovers {list(crossovers_hz)}), got {len(floors)}")


# ---------------------------------------------------------------- input and crossovers

def input_gain(x: np.ndarray, fs: float) -> tuple[float, float]:
    """(gain dB, measured loudest-section 1 s RMS dBFS): the gain that brings the
    premaster's loudest-section 1 s RMS to R_REF_DBFS; 0 dB when it is not finite
    (input shorter than 1 s, or silent)."""
    m = float(meter.rms_1s(x, fs)["loud_section_dbfs"])
    return (R_REF_DBFS - m if math.isfinite(m) else 0.0), m


def tonic_partial_hz(tonic) -> float | None:
    """Lowest frequency of the tonic's pitch class at or above KEY_PARTIAL_MIN_HZ (equal
    temperament, A4 = 440 Hz), in [80, 160); None for an unknown tonic."""
    pc = pitch_class(tonic)
    if pc is None:
        return None
    f = 440.0 * 2.0 ** ((pc - 9) / 12.0)
    return float(f / 2.0 ** math.floor(math.log2(f / KEY_PARTIAL_MIN_HZ)))


def key_crossovers(tonic, base=bands.DEFAULT_CROSSOVERS) -> tuple[float, ...]:
    """Key-locked crossovers (module contract); the base for an unknown tonic."""
    base = tuple(float(f) for f in base)
    p = tonic_partial_hz(tonic)
    if p is None:
        return base
    fc1 = float(np.clip(p / 2.0 ** KEY_MARGIN_OCT, *KEY_XOVER_RANGE_HZ))
    return tuple(fc1 * f / base[0] for f in base)


def crossover_choice(analysis: dict, controls: Controls) -> dict:
    """Record of the crossovers the plan uses and why (plan_record["crossovers"])."""
    meta = analysis["meta"]
    xo = [float(f) for f in meta["crossovers"]]
    rule = meta.get("crossover_rule", "fixed")
    tonic = meta.get("crossover_tonic")
    rec = {"hz": xo, "rule": rule, "tonic": tonic, "tonic_partial_hz": None,
           "requested": None if controls.crossovers_hz is None else list(controls.crossovers_hz)}
    if rule == "key":
        p = tonic_partial_hz(tonic)
        rec["tonic_partial_hz"] = p
        if p is None:
            rec["reason"] = "no key found: the default crossovers"
        else:
            fc = p / 2.0 ** KEY_MARGIN_OCT
            clip = "" if KEY_XOVER_RANGE_HZ[0] <= fc <= KEY_XOVER_RANGE_HZ[1] else (
                f", clipped from {fc:.1f} Hz to the {KEY_XOVER_RANGE_HZ[0]:g} to "
                f"{KEY_XOVER_RANGE_HZ[1]:g} Hz range")
            rec["reason"] = (f"key-locked: tonic {tonic}, lowest partial at or above "
                             f"{KEY_PARTIAL_MIN_HZ:g} Hz = {p:.2f} Hz, lowest crossover one third "
                             f"of an octave below it{clip}, the others x4")
    else:
        rec["reason"] = "given crossovers (--xover or the analysis's own)"
    return rec


def analyse(x: np.ndarray, fs: float, crossovers=None, bpm: float | None = None) -> dict:
    """The planner's analysis: input normalisation, then features.analyse with the given
    or the key-locked crossovers (module contract)."""
    from eluate import features
    g, m = input_gain(x, fs)
    xn = np.asarray(x, dtype=np.float64) * 10.0 ** (g / 20.0)
    kw = {} if bpm is None else {"bpm": float(bpm)}
    if crossovers is None:
        a = features.analyse(xn, fs, fcs_from_key=key_crossovers, **kw)
    else:
        a = features.analyse(xn, fs, fcs=tuple(float(f) for f in crossovers), **kw)
    a["meta"]["input"] = {"gain_db": float(g), "r_ref_dbfs": R_REF_DBFS,
                          "measured_dbfs": float(m)}
    from eluate import noise
    fl = noise.floor(xn, fs)["db"]
    a["plan_inputs"]["noise_floor_db"] = float(fl) if math.isfinite(fl) else None
    return a


def subside_hz(analysis: dict, controls: Controls) -> float:
    """controls.subside_hz, else the profile's (live: 60 Hz), else the analysis suggestion,
    except SUBSIDE_DEFAULT_HZ when
    the sub band's decoherence is below SUBSIDE_DECOHERENCE_MIN (the suggestion read
    150 Hz on every real track); clipped to SUBSIDE_RANGE_HZ."""
    pi = analysis["plan_inputs"]
    if controls.subside_hz is not None:
        fc = controls.subside_hz
    elif controls.prof["sidestrip_hz"] is not None:
        fc = controls.prof["sidestrip_hz"]
    elif float(pi["band_decoherence"][0]) < SUBSIDE_DECOHERENCE_MIN:
        fc = SUBSIDE_DEFAULT_HZ
    else:
        fc = float(pi["side_strip_hz"])
    return float(np.clip(fc, *SUBSIDE_RANGE_HZ))


def crest_exempt(crossovers_hz, fc_sub: float) -> list[list[int]]:
    """S views the sidestrip removes on purpose: bands whose upper edge is at or below
    2 fc_sub (the squared order-2 high-pass is -12 dB at fc / 2 and the band's S
    level falls toward the -60 dBFS crest gate, so its crest is the strip's, not a
    loss the solver can steer). View index 1 = S."""
    return [[b, 1] for b, f in enumerate(crossovers_hz) if f <= 2.0 * fc_sub]


def targets_from(analysis: dict, controls: Controls) -> Targets:
    controls = resolve(analysis, controls)
    pi = analysis["plan_inputs"]
    secs = [{"start": float(s["start"]), "end": float(s["end"]), "label": s.get("label"),
             "integrated_lufs": float(s["integrated_lufs"])} for s in pi["sections"]]
    return Targets(
        rms_dbfs=controls.rms_target,
        slope_db_oct=target_slope(analysis, controls),
        preserve_mix=controls.preserve_mix,
        ltas_centres_hz=[float(v) for v in pi["ltas_centres_hz"]],
        ltas_ref_db=[float(v) for v in pi["ltas_stereo_db"]],
        crest_limits_db=crest_limits(controls.crossovers_hz),
        true_peak_max_dbtp=controls.ceiling_dbtp,
        sections=secs,
        crossovers_hz=list(controls.crossovers_hz),
        crest_exempt=(crest_exempt(controls.crossovers_hz, subside_hz(analysis, controls))
                      if controls.side_band0_exempt else []),
        crest_mode=controls.crest_mode,
        crest_floors_db=None if controls.crest_floors_db is None else list(controls.crest_floors_db),
        tp_meter=controls.tp_meter,
        slope_mode=controls.prof["slope"],
    )


# ---------------------------------------------------------------- notches

def notch_rendered(channel: str) -> bool:
    """Channel policy while plan.Notch has no channel (see module contract)."""
    return channel in ("both", "M")


def _notch_envelope(excess_db, hop_s: float, depth_scale: float, n: int, fs: float,
                    guard=None, erode_s: float = 0.0) -> np.ndarray:
    """1 ms cut envelope (dB <= 0). guard: boolean mask on the analysis grid; guarded
    frames have zero excess before interpolation and smoothing, so the guard ramps
    with the same smoothing as everything else (audit 07 A10). erode_s: the analysis's
    centred smoothing of the excess (resonance: about 3 s below 500 Hz), which widens a
    resonance by half of it on each side; a minimum filter of half that width (the
    NOTCH_EXCESS_FLOOR_DB already trims the ramp's foot) takes the lead back (T7: low
    notches engaged more than a second early). The NOTCH_SMOOTH_MS ramps then
    trail the cut (causal), so it starts at the eroded onset, not before."""
    ex = np.asarray(excess_db, dtype=np.float64)
    ex = np.where(np.isfinite(ex), np.maximum(ex, 0.0), 0.0)
    nc = control_length(n, fs)
    if not ex.size:
        return np.zeros(nc)
    if guard is not None:
        guard = np.asarray(guard, dtype=bool)
        if guard.size != ex.size:
            raise ValueError(f"notch guard has {guard.size} frames, the excess {ex.size}")
        ex = np.where(guard, 0.0, ex)
    cut = depth_scale * np.minimum(NOTCH_MAX_CUT_DB,
                                  NOTCH_FRACTION * np.maximum(ex - NOTCH_EXCESS_FLOOR_DB, 0.0))
    if erode_s > 0.0:
        from scipy.ndimage import minimum_filter1d
        # half the smoothing window: the excess floor already trims the ramp's foot
        cut = minimum_filter1d(cut, max(1, int(round(0.5 * erode_s / hop_s))), mode="nearest")
    t_grid = (np.arange(ex.size) + 0.5) * hop_s
    t_ctrl = np.arange(nc) / CONTROL_RATE_HZ
    c = np.interp(t_ctrl, t_grid, cut, left=cut[0], right=cut[-1])
    w = max(1, int(round(NOTCH_SMOOTH_MS * CONTROL_RATE_HZ / 1000.0)))
    k = np.ones(w) / w
    if erode_s > 0.0:
        c = np.convolve(np.convolve(c, k)[:nc], k)[:nc]      # trailing ramps
    else:
        c = np.convolve(np.convolve(c, k, mode="same"), k, mode="same")
    c = np.clip(np.round(c / NOTCH_QUANT_DB) * NOTCH_QUANT_DB, 0.0, NOTCH_MAX_CUT_DB)
    return -c


def _analysis_smooth_s(r: dict, fs: float) -> float:
    """The centred smoothing (s) resonance.py applied to a candidate's excess: its
    resolution's 8 window lengths, at least 0.5 s (resonance.SMOOTH_FRAMES, SMOOTH_S)."""
    from eluate.features import resonance
    nres = r.get("resolution")
    if nres is None:
        return 0.0
    return max(resonance.SMOOTH_S, resonance.SMOOTH_FRAMES * float(nres) / fs)


def q_fit_ok(q: float, qu: float) -> bool:
    """The analyser's own fit test (features/resonance.py: finite, uncertainty below half
    the Q), the only case in which the measured Q is used (audit 07 F9)."""
    return math.isfinite(q) and math.isfinite(qu) and qu < 0.5 * q


def _notch_q(r: dict) -> float:
    q, qu = float(r.get("q", math.nan)), float(r.get("q_uncertainty", math.nan))
    if not q_fit_ok(q, qu):
        q = NOTCH_Q_DEFAULT
    hi = NOTCH_LOW_Q_MAX if float(r.get("f0", math.inf)) < NOTCH_LOW_HZ else NOTCH_Q_HI
    return float(np.clip(q, max(NOTCH_Q_LO, NOTCH_Q_MIN), hi))


_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def pitch_class(name) -> int | None:
    """Pitch class of a note or chord name ("C#m" -> 1, "Bb" -> 10); None if unparsable."""
    if not name or name[0] not in _PC:
        return None
    pc = _PC[name[0]]
    if len(name) > 1 and name[1] in "#b":
        pc += 1 if name[1] == "#" else -1
    return pc % 12


def protected_pitch_classes(analysis: dict) -> set[int]:
    """Key tonic and its fifth (guarded for the whole track)."""
    key = analysis["plan_inputs"].get("key") or {}
    t = pitch_class(key.get("tonic"))
    return set() if t is None else {t, (t + 7) % 12}


def bar_roots(analysis: dict) -> list[tuple[float, float, int]]:
    """(start s, end s, root pitch class) per bar with a parsable chord."""
    out = []
    for c in (analysis.get("pitch") or {}).get("chords") or []:
        pc = pitch_class(c.get("chord"))
        if pc is not None:
            out.append((float(c["start"]), float(c["end"]), pc))
    return out


def cents_to_pitch_class(f0: float, pc: int) -> float:
    """Distance in cents from f0 to the nearest octave of pitch class pc (A4 = 440 Hz)."""
    c = 1200.0 * math.log2(f0 / 440.0) - 100.0 * (pc - 9)
    return abs((c + 600.0) % 1200.0 - 600.0)


def usable_kick_onsets(analysis: dict) -> list[float]:
    """Kick onsets snapped to the 1 ms grid, or [] when the list is empty or sparser than
    KICK_MIN_RATE_HZ over the track (treated as no kick information, not a mask)."""
    meta, pi = analysis["meta"], analysis["plan_inputs"]
    on = np.asarray(pi["kick_onsets"], dtype=np.float64)
    dur = float(meta["n"]) / float(meta["fs"])
    if on.size == 0 or on.size / max(dur, 1e-9) < KICK_MIN_RATE_HZ:
        return []
    return [float(v) for v in np.round(on * CONTROL_RATE_HZ) / CONTROL_RATE_HZ]


def _guard_frames(analysis: dict, f0: float) -> np.ndarray:
    """Analysis-grid frames whose bar's chord root is within the guard of f0."""
    meta = analysis["meta"]
    k = int(meta["frames"])
    hop = float(meta["hop_s"])
    t = (np.arange(k) + 0.5) * hop
    m = np.zeros(k, dtype=bool)
    for a, b, pc in bar_roots(analysis):
        if cents_to_pitch_class(f0, pc) <= NOTCH_MUSICAL_GUARD_CENTS:
            m |= (t >= a) & (t < b)
    return m


def _merge_groups(items: list[tuple]) -> list[list[tuple]]:
    """Group (notch, ...) items whose half-cut bands overlap, chained in order of the lower
    edge: the grouping notch.merge_candidates uses."""
    srt = sorted(items, key=lambda it: notch_stage.band_edges(it[0].f0_hz, it[0].q)[0])
    groups: list[list[tuple]] = []
    hi_run = -math.inf
    for it in srt:
        lo, hi = notch_stage.band_edges(it[0].f0_hz, it[0].q)
        if groups and lo <= hi_run:
            groups[-1].append(it)
            hi_run = max(hi_run, hi)
        else:
            groups.append([it])
            hi_run = hi
    return groups


def plan_notches(analysis: dict, kick: bool = True) -> tuple[NotchPlan, list[dict], list[dict]]:
    """NotchPlan, a report row per analysis notch (rendered or why not) and the merged
    notches actually rendered (members, f0, Q, peak excess)."""
    meta, pi = analysis["meta"], analysis["plan_inputs"]
    fs, n, hop = float(meta["fs"]), int(meta["n"]), float(meta["hop_s"])
    dur = n / fs
    guard = protected_pitch_classes(analysis)
    rows, cands = [], []
    for i, r in enumerate(pi["notches"]):
        q = _notch_q(r)
        f0 = float(r["f0"])
        gmask = _guard_frames(analysis, f0)
        ex = np.asarray(r["excess_db"], dtype=np.float64)
        ex = np.where(np.isfinite(ex), ex, 0.0)
        g = _notch_envelope(ex, hop, float(r["depth_scale"]), n, fs, guard=gmask,
                            erode_s=_analysis_smooth_s(r, fs))
        integ = float(-g.sum() / CONTROL_RATE_HZ)
        peak_all = float(np.max(ex)) if ex.size else 0.0
        free = ex[~gmask[:ex.size]] if gmask.size >= ex.size else ex
        peak = float(np.max(free)) if free.size else 0.0      # ranked on unguarded frames
        cover = float(sum(max(0.0, b - a) for a, b in r["spans"])) / max(dur, 1e-9)
        near = [pc for pc in guard if cents_to_pitch_class(f0, pc) <= NOTCH_MUSICAL_GUARD_CENTS]
        row = {"index": i, "f0_hz": f0, "q": q, "q_measured": float(r["q"]),
               "q_fit_ok": q_fit_ok(float(r["q"]), float(r.get("q_uncertainty", math.nan))),
               "channel": r["channel"], "flag": r["flag"], "depth_scale": float(r["depth_scale"]),
               "max_cut_db": float(-g.min()) if g.size else 0.0, "cut_db_s": integ,
               "peak_excess_db": peak_all, "peak_excess_unguarded_db": peak,
               "span_share": cover, "chord_guarded_s": float(gmask.sum() * hop),
               "spans": [[float(a), float(b)] for a, b in r["spans"]],
               "rendered": False, "merged_into": None, "reason": ""}
        if peak_all > NOTCH_SUSPECT_EXCESS_DB:
            row["reason"] = f"suspect: peak excess {peak_all:.1f} dB > {NOTCH_SUSPECT_EXCESS_DB} dB"
        elif cover > NOTCH_SUSPECT_SPAN:
            row["reason"] = f"suspect: spans cover {100 * cover:.0f} % of the track"
        elif near:
            row["reason"] = (f"within {NOTCH_MUSICAL_GUARD_CENTS:.0f} cents of the key's tonic "
                             f"or fifth (pitch class {sorted(near)})")
        elif not notch_rendered(r["channel"]):
            row["reason"] = "S-only resonance; notch stage is not channel-aware"
        elif integ <= 0.0 or not f0 < fs / 2:
            row["reason"] = "no cut" + (" (every active bar's chord root is within the guard)"
                                        if gmask.any() else "")
        else:
            cands.append((Notch(f0, q, Envelope(g)), i, peak))
        if row["reason"].startswith("suspect"):
            log.info("notch candidate %.1f Hz skipped, %s", f0, row["reason"])
        rows.append(row)
    # merge first, then cap (audit 07 A10): a merged notch ranks by its members' peak
    groups = _merge_groups(cands)
    ranked = sorted(groups, key=lambda grp: (-max(it[2] for it in grp), min(it[1] for it in grp)))
    keep, merged = ranked[:NOTCH_MAX], []
    for grp in ranked[NOTCH_MAX:]:
        for _, i, _ in grp:
            rows[i]["reason"] = f"below the top {NOTCH_MAX} merged notches by peak excess"
    keep = sorted(keep, key=lambda grp: min(it[1] for it in grp))
    notches = []
    for m_i, grp in enumerate(keep):
        nt = notch_stage.merge_candidates([it[0] for it in sorted(grp, key=lambda it: it[1])])
        assert len(nt) == 1
        notches.append(nt[0])
        for _, i, _ in grp:
            rows[i]["rendered"] = True
            rows[i]["merged_into"] = m_i
        merged.append({"f0_hz": nt[0].f0_hz, "q": nt[0].q,
                       "members": [it[1] for it in sorted(grp, key=lambda it: it[1])],
                       "peak_excess_unguarded_db": max(it[2] for it in grp),
                       "max_cut_db": float(-nt[0].gain_db.values.min())})
    on = usable_kick_onsets(analysis) if kick else []
    km = notch_stage.kick_mask(on, dur, half_ms=KICK_HALF_MS) if on else None
    return NotchPlan(notches, km), rows, merged


# ---------------------------------------------------------------- tone

def iso226_spl(phon: float, f) -> np.ndarray:
    """SPL (dB) of the ISO 226:2003 equal-loudness contour at `phon`, at frequencies f (Hz):
    the standard's formula on its 29 tabled frequencies, interpolated in log frequency,
    held below 20 Hz. The standard has no data above 12.5 kHz, where hearing falls off
    steeply, so the less sensitive of its 10 and 12.5 kHz levels is held there (the 10 to
    12.5 kHz step runs the other way and would make 16 kHz count more than 1 kHz)."""
    a, lu, tf = (np.asarray(v, dtype=np.float64) for v in (ISO226_AF, ISO226_LU, ISO226_TF))
    af = 4.47e-3 * (10.0 ** (0.025 * phon) - 1.15) + (0.4 * 10.0 ** ((tf + lu) / 10.0 - 9.0)) ** a
    spl = 10.0 / a * np.log10(af) - lu + 94.0
    x = np.log2(np.asarray(ISO226_F, dtype=np.float64))
    xf = np.log2(np.maximum(np.asarray(f, dtype=np.float64), ISO226_F[0]))
    out = np.interp(xf, x, spl)
    out[xf > x[-1]] = max(spl[-1], spl[-2])
    return out


def audibility_weights(f, phon: float = EQ_WEIGHT_PHON) -> np.ndarray:
    """Per-frequency weight of the input EQ fit's error: how much louder (> 1) or quieter
    (< 1) a band sounds than 1 kHz at the same level, as an amplitude ratio off the phon
    contour (31.5 Hz about 0.1 at 80 phon, 3 to 4 kHz above 1), within EQ_WEIGHT_RANGE."""
    s = iso226_spl(phon, f) - iso226_spl(phon, [1000.0])[0]
    return np.clip(10.0 ** (-s / 20.0), *EQ_WEIGHT_RANGE)


def target_slope(analysis: dict, controls: Controls) -> float:
    """The tone target (dB/oct): REFERENCE_SLOPE + Tilt, or under a profile whose slope is
    "report" (live) the mix's own measured slope + Tilt, so the input EQ does not steer
    acoustic material toward the electronic reference (maintainer, 2026-10-26: drumless
    material is naturally darker; Elowsson & Friberg, AES 2017)."""
    if controls.prof["slope"] == "report":
        return float(analysis["plan_inputs"]["tilt_slope_db_oct"]["stereo"]) + controls.tilt_db
    return REFERENCE_SLOPE + controls.tilt_db


def deficit_curve(nodes, measured: float, target: float, shelf: float,
                  sub_pull: bool = True) -> dict:
    """The tone target's node deficits (module contract, Tone): the rotation of the
    measured fit line onto the target line about PIVOT_HZ, no node cut more than
    EQ_LOW_ROT_CAP_DB by it (T2), the sub held to the 100 Hz deficit and pulled into
    +-SUB_SHELF_MAX_DB of the target line (not with sub_pull False: the live profile leaves
    a thin sub thin). The sub correction is not bound by the T2 cap:
    a node below 100 Hz can be cut by the cap plus the correction (audit 13, Sol #6; on
    the four VA tracks the correction cut at most 0.45 dB, Track C)."""
    nodes = np.asarray(nodes, dtype=np.float64)
    delta = target - measured
    lo, hi = SLOPE_RANGE_HZ
    # T2 (sprint 2026-10-21): brightness comes from the top, not from cutting the lows
    d = np.maximum(delta * np.log2(np.clip(nodes, lo, hi) / PIVOT_HZ), -EQ_LOW_ROT_CAP_DB)
    # holding the 100 Hz deficit below 100 Hz moves the sub against the target
    # line by delta * log2(100 / f): predicted output shelf at the sub centre
    pred = shelf + delta * math.log2(lo / SUB_CENTRE_HZ)
    corr = float(np.clip(pred, -SUB_SHELF_MAX_DB, SUB_SHELF_MAX_DB) - pred) if sub_pull else 0.0
    d = np.where(nodes < lo, d + corr, d)
    return {"nodes_hz": nodes.tolist(), "deficit_db": d.tolist(), "measured_slope": measured,
            "target_slope": target, "delta_slope": delta, "sub_shelf_db": shelf,
            "sub_shelf_predicted_db": pred, "sub_shelf_correction_db": corr}


def tone_deficit(analysis: dict, controls: Controls, nodes_hz=None, rotation: float = 0.0) -> dict:
    """Node deficits (dB, before the harmonic split and the +-3 dB clip). rotation (dB/oct)
    is added to the target slope (the solver's tone_rotation lever)."""
    pi = analysis["plan_inputs"]
    nodes = tone_nodes() if nodes_hz is None else nodes_hz
    return deficit_curve(nodes, float(pi["tilt_slope_db_oct"]["stereo"]),
                         target_slope(analysis, controls) + rotation,
                         float(pi["sub_shelf_db"]["stereo"]), sub_pull=controls.prof["slope"] != "report")


# ---------------------------------------------------------------- input EQ

def inputeq_design(analysis: dict, controls: Controls) -> dict:
    """The input EQ's fit inputs (stored as InputEqPlan.design): the 1/3-octave LTAS
    nodes from EQ_FIT_RANGE_HZ, each node's residual from the measured fit line
    (stereo LTAS minus the OLS line over 100 Hz to 10 kHz, the line features/tilt.py
    fits), the measured and target slopes and the sub shelf. Nodes more than
    EQ_GATE_DB below the loudest node are left out (band-limited material)."""
    from eluate.features import tilt as tilt_feature
    pi = analysis["plan_inputs"]
    c = np.asarray(pi["ltas_centres_hz"], dtype=np.float64)
    L = np.asarray(pi["ltas_stereo_db"], dtype=np.float64)
    slope, icpt = tilt_feature.fit_slope(c, L)
    r = L - (icpt + slope * np.log2(c))
    lo, hi = EQ_FIT_RANGE_HZ
    sel = (c >= lo - 1e-9) & (c <= hi + 1e-9) & (L > np.max(L) - EQ_GATE_DB)
    out = {"nodes_hz": c[sel].tolist(), "residual_db": r[sel].tolist(),
           "measured_slope": float(pi["tilt_slope_db_oct"]["stereo"]),
           "target_slope": target_slope(analysis, controls),
           "sub_pull": controls.prof["slope"] != "report",
           "sub_shelf_db": float(pi["sub_shelf_db"]["stereo"]), "rotation": 0.0,
           "side_residual_db": None}
    # T4: the quiet-passage floor (None when the analysis predates it or the track has no
    # quiet passage) and the master's predicted gain over the normalised audio
    fl = pi.get("noise_floor_db")
    out["noise"] = {"floor_db": None if fl is None or not math.isfinite(fl) else float(fl),
                    "gain_db": quiet_gain_db(controls)}
    dec = pi.get("band_decoherence")
    out["top_decoherence"] = float(dec[-1]) if dec is not None and len(dec) else None
    if controls.prof["eq_bounds"] is not None:
        out["bounds"] = dict(controls.prof["eq_bounds"])
    if controls.prof["eq_hpf"] is not None:
        out["hpf"] = dict(controls.prof["eq_hpf"])
    if controls.prof["eq_weighting"] is not None:
        out["weight_phon"] = float(controls.prof["eq_weighting"])
    if controls.prof["eq_residual_window"]:
        out["residual_window"] = True
    # the side spectrum for the M/S top end: its residual from the same fit line, after
    # aligning its level to the stereo spectrum over EQ_SIDE_ALIGN_HZ (the side's shape,
    # not its level, decides its shelf). None when the analysis predates it or the side
    # is near silent there.
    S = pi.get("ltas_side_db")
    if S is not None:
        S = np.asarray(S, dtype=np.float64)
        al = (c >= EQ_SIDE_ALIGN_HZ[0]) & (c <= EQ_SIDE_ALIGN_HZ[1]) & np.isfinite(S)
        if np.any(al):
            off = float(np.median(S[al] - L[al]))
            if off > -EQ_SIDE_SILENT_DB:
                rs = S - off - (icpt + slope * np.log2(c))
                ok = np.isfinite(rs) & (S > np.max(S[np.isfinite(S)]) - EQ_GATE_DB)
                out["side_residual_db"] = [float(v) if o else None
                                           for v, o in zip(rs[sel], ok[sel])]
    return out


def _eq_from_params(hp_q: float, p: dict, fs: float,
                    hp_fc: float = INPUTEQ_HPF_FC_HZ) -> InputEqPlan:
    """InputEqPlan from fitted parameters (sections with a zero gain are off)."""
    ls = (EqLowShelf(p["ls_gain"], p["ls_fc"], EQ_LOW_SHELF_Q) if p.get("ls_gain") else None)
    bl = (EqBell(p["bell_gain"], p["bell_fc"], p["bell_q"]) if p.get("bell_gain") else None)
    hs = (EqHighShelf(p["hs_gain"], p["hs_fc"]) if p.get("hs_gain") else None)
    bs = (EqBandShelf(p["bs_gain"], p["bs_lo"], p["bs_hi"]) if p.get("bs_gain") else None)
    # a side shelf of 0 dB still makes the top end M/S (the side stays flat while the mid
    # is shelved); absent means linked. The presence shelf is high_shelf.
    hss = EqHighShelf(p["hss_gain"], p["hss_fc"]) if "hss_gain" in p else None
    air = EqHighShelf(p["air_gain"], p["air_fc"]) if p.get("air_gain") else None
    airs = EqHighShelf(p["airs_gain"], p["airs_fc"]) if "airs_gain" in p else None
    return InputEqPlan(highpass=EqHighPass(hp_fc, hp_q), low_shelf=ls, bell=bl,
                       high_shelf=hs, band_shelf=bs, high_shelf_side=hss, air_shelf=air,
                       air_shelf_side=airs)


def hpf_q(sub_shelf_predicted_db: float) -> float:
    """High-pass Q from the sub excess: Butterworth (0.707) while the predicted output
    shelf is at most +SUB_SHELF_MAX_DB; falling linearly to the lower Q bound (0.5) when
    it exceeds that by EQ_HPF_Q_SPAN_DB or more (a lower Q cuts more sub, up to ~100 Hz).
    A thin sub is the low shelf's to lift, not the high-pass's (sprint 2026-10-21 T11:
    the Q 1.0 branch lifted 35 Hz by 2.5 dB while the same plan cut 80 to 250 Hz)."""
    v = float(sub_shelf_predicted_db)
    ex = float(np.clip((v - SUB_SHELF_MAX_DB) / EQ_HPF_Q_SPAN_DB, 0.0, 1.0))
    lo, _ = INPUTEQ_HPF_Q_RANGE
    return round(EQ_HPF_Q_BUTTERWORTH - ex * (EQ_HPF_Q_BUTTERWORTH - lo), 2)


def low_mid_run(nodes, residual_db) -> dict:
    """The low-mid excess (the second-harmonic build-up, sprint 2026-10-21 T3): the
    largest residual over EQ_LOW_MID_RUN_HZ, measured against its neighbours (the median
    residual of the nodes up to EQ_BELL_NEIGHBOUR_OCT outside the run), not against the
    fit line, whose natural spread is 2 to 4 dB. The run is the plateau around the peak:
    the contiguous nodes within EQ_LOW_MID_RUN_HZ no more than EQ_BELL_RUN_DB below it;
    width EQ_NODE_OCT per node. shape: None (excess at most EQ_BELL_GATE_DB), "bell" (a
    run of at most EQ_BAND_SHELF_OCT) or "band shelf" (broader, at most
    EQ_BAND_SHELF_MAX_OCT when fitted). excess_db is what the cut may take."""
    nodes = np.asarray(nodes, dtype=np.float64)
    res = np.asarray(residual_db, dtype=np.float64)
    lo_hz, hi_hz = EQ_LOW_MID_RUN_HZ
    lm = (nodes >= lo_hz - 1e-9) & (nodes <= hi_hz + 1e-9)
    none = {"excess_db": 0.0, "shape": None, "run_hz": None, "run_oct": 0.0, "peak_index": None,
            "neighbour_db": None}
    if not np.any(lm):
        return none
    k = int(np.argmax(np.where(lm, res, -np.inf)))

    def neighbours(a, b):
        lo_n, hi_n = nodes[a] / 2.0 ** EQ_BELL_NEIGHBOUR_OCT, nodes[b] * 2.0 ** EQ_BELL_NEIGHBOUR_OCT
        nb = ((nodes >= lo_n - 1e-9) & (nodes < nodes[a] - 1e-9)) |              ((nodes > nodes[b] + 1e-9) & (nodes <= hi_n + 1e-9))
        return float(np.median(res[nb])) if np.any(nb) else float(np.median(res))

    # the run: the plateau around the peak, nodes within EQ_BELL_RUN_DB of it
    a = b = k
    while a > 0 and lm[a - 1] and res[a - 1] >= res[k] - EQ_BELL_RUN_DB:
        a -= 1
    while b < nodes.size - 1 and lm[b + 1] and res[b + 1] >= res[k] - EQ_BELL_RUN_DB:
        b += 1
    ref = neighbours(a, b)
    excess = float(res[k] - ref)
    if excess <= EQ_BELL_GATE_DB:
        return {**none, "excess_db": excess, "peak_index": k, "neighbour_db": ref}
    width = (b - a + 1) * EQ_NODE_OCT
    return {"excess_db": excess, "shape": "band shelf" if width > EQ_BAND_SHELF_OCT + 1e-9
            else "bell", "run_hz": [float(nodes[a]), float(nodes[b])], "run_oct": float(width),
            "peak_index": k, "neighbour_db": ref,
            "run_mean_db": float(np.mean(res[a:b + 1]) - ref)}


def _range_mean(nodes, v, rng) -> float | None:
    sel = (nodes >= rng[0] - 1e-9) & (nodes <= rng[1] + 1e-9)
    return float(np.mean(v[sel])) if np.any(sel) else None


def quiet_gain_db(controls: Controls) -> float:
    """The master's predicted gain over a quiet passage of the normalised audio: the
    loudness target over R_ref (AES17) plus the lift the compressors give what they leave
    alone, the sum of the profile's average budgets (multiband, glue, clipper, limiter).
    On Track G (2026-10-20) the floor rose 29 dB with +6 dB of shelving; this predicts 19.
    """
    prof = controls.prof
    b = {**mb_stage.default_budget(controls.makeup), **prof["mb_budget"]}
    glue = glue_stage.SERIAL_GR_DB if prof["glue_net_db"] is None else prof["glue_net_db"]
    clip = prof["clip_target_db"] if prof["clipper_presets"] is None else 0.0
    lift = float(b["comp_gr_db"]) + glue + clip + prof["limiter_max_gr_db"]
    return controls.rms_target - (R_REF_DBFS + meter.AES17_OFFSET_DB) + lift


def top_caps(design: dict, nodes, D, fs: float, mid: dict | None = None,
             side: bool = False) -> dict:
    """Boost caps of the presence and air shelves (sprint 2026-10-21 T4): each at most
    the profile's bound, at most the deficit measured in its own range
    (EQ_PRES_DEFICIT_HZ, EQ_AIR_DEFICIT_HZ; no boost where the range has no node above
    the gate, e.g. an air band under the noise floor), and each channel's pair together
    at most the noise headroom (the M pair in fit_inputeq, the S pair in
    _fit_side_shelves; the stereo floor then rises by at most that): the premaster's quiet-passage floor (noise.floor, BS.468, on the normalised
    audio) plus the master's predicted quiet-passage gain (`quiet_gain_db`) must stay
    under noise.FLAG_DB; a floor already past it (music in the quiet passages, or a floor
    exposed anyway) sets no cap. For S, also at
    most the mid shelf + EQ_SIDE_OVER_MID_DB plus the side's own extra deficit, and no
    boost at a top-band decoherence of EQ_SIDE_DECOHERENT or more."""
    bounds = design.get("bounds") or {}
    gh = float(bounds.get("hs_side_db" if side else "hs_mid_db",
                          INPUTEQ_HS_SIDE_MAX_DB if side else INPUTEQ_HS_MID_MAX_DB))
    nodes = np.asarray(nodes, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    out = {}
    for name, rng in (("presence", EQ_PRES_DEFICIT_HZ), ("air", EQ_AIR_DEFICIT_HZ)):
        m = _range_mean(nodes, D, (rng[0], min(rng[1], EQ_HS_MAX_FS_SHARE * fs)))
        cap = min(gh, max(0.0, m)) if m is not None else 0.0
        why = "range deficit" if m is not None and m < gh else "bound" if m is not None \
            else "no node above the gate"
        out[name] = {"cap_db": cap, "range_deficit_db": m, "reason": why}
    nz = design.get("noise") or {}
    head = None
    if nz.get("floor_db") is not None and nz.get("gain_db") is not None:
        from eluate import noise as noise_mod
        head = noise_mod.FLAG_DB - (nz["floor_db"] + nz["gain_db"])
        # a quiet-passage level already past the flag before any shelving is music (pads,
        # tails) or a floor the master cannot hide either way: no cap. Below it, the top
        # shelves may not be what lifts it past (the hiss the master would expose).
        if head > 0.0:
            for name in ("presence", "air"):
                if out[name]["cap_db"] > head:
                    out[name] = {**out[name], "cap_db": head, "reason": "noise headroom"}
        else:
            head = None
    out["noise_headroom_db"] = head
    if side and mid is not None:
        if (design.get("top_decoherence") or 0.0) >= EQ_SIDE_DECOHERENT:
            for name in ("presence", "air"):
                out[name] = {**out[name], "cap_db": 0.0, "reason": "top already decoherent"}
        else:
            for name in ("presence", "air"):
                lim = max(0.0, mid.get(name, 0.0)) + EQ_SIDE_OVER_MID_DB \
                    + max(0.0, mid.get(name + "_side_extra", 0.0))
                if out[name]["cap_db"] > lim:
                    out[name] = {**out[name], "cap_db": lim, "reason": "mid + 2 dB"}
    return out


def _corner_range(rng, fs: float) -> tuple[float, float]:
    top = min(rng[1], EQ_HS_MAX_FS_SHARE * fs)
    return min(rng[0], top), top


def _top_fc(v) -> float:
    """A top shelf's corner rounded to 10 Hz: its cost is flat near a gain bound, so 1 Hz
    would read the optimiser's noise (and differ between two levels of one premaster)."""
    return float(round(float(v) / 10.0) * 10.0)


def _top_starts(g_pres: float, g_air: float, pr, ar) -> list[list[float]]:
    return [[g_pres, math.log2(float(np.clip(fp, *pr))), g_air, math.log2(float(np.clip(fa, *ar)))]
            for fp in EQ_PRES_STARTS_HZ for fa in EQ_AIR_STARTS_HZ]


def _residual_nodes(nodes: np.ndarray, design: dict) -> np.ndarray:
    """The nodes whose residual from the fit line the EQ corrects: from SLOPE_RANGE_HZ's
    bottom up (the sub is the deficit curve's), and under residual_window (live) only up to
    its top. Above 10 kHz the residual is from the line extended past the range it was
    fitted on, which a spectrum's natural top roll-off always undershoots: on the R9 tracks
    by 7 to 27 dB at 12.5 to 16 kHz, enough to drive the air shelves to their bound."""
    sel = nodes >= SLOPE_RANGE_HZ[0]
    if design.get("residual_window"):
        sel &= nodes <= SLOPE_RANGE_HZ[1]
    return sel


def fit_inputeq(design: dict, fs: float, phase: str = "zero") -> tuple[InputEqPlan, dict]:
    """Least-squares fit of the input EQ to the deficit curve (module contract, Input
    EQ). Returns the plan section and the fit record.

    Sprint 2026-10-21 Group A: the low-mid cut only on a real excess against its
    neighbours, no deeper than that excess (low_mid_run, T3); the top end as a presence
    shelf (corner EQ_PRES_FC_HZ) and an air shelf (EQ_AIR_FC_HZ), each boost capped by
    top_caps (its own range's deficit, the noise headroom; T4, T12); the presence and
    air boosts together within the noise headroom. The side gets its own pair
    (_fit_side_shelves)."""
    from scipy.optimize import least_squares

    from eluate.stages import inputeq as eq_stage
    nodes = np.asarray(design["nodes_hz"], dtype=np.float64)
    res = np.asarray(design["residual_db"], dtype=np.float64)
    target = float(design["target_slope"]) + float(design.get("rotation", 0.0))
    dc = deficit_curve(nodes, float(design["measured_slope"]), target,
                       float(design["sub_shelf_db"]), sub_pull=design.get("sub_pull", True))
    D = np.asarray(dc["deficit_db"]) - np.where(_residual_nodes(nodes, design), res, 0.0)
    # the profile may fix the high-pass (live: 25 Hz Butterworth) and the gain bounds
    hpf = design.get("hpf") or {}
    hp_fc = float(hpf.get("fc_hz", INPUTEQ_HPF_FC_HZ))
    q = float(hpf["q"]) if "q" in hpf else hpf_q(dc["sub_shelf_predicted_db"])
    # the high-pass is the maintainer's rumble filter: only the extra sub cut of a lowered
    # Q (against the Butterworth baseline) counts toward the deficit
    hp_extra = (eq_stage.response_db(InputEqPlan(EqHighPass(hp_fc, q)), nodes, fs)
                - eq_stage.response_db(InputEqPlan(EqHighPass(hp_fc, EQ_HPF_Q_BUTTERWORTH)),
                                       nodes, fs))
    run = low_mid_run(nodes, res)
    excess = run["excess_db"]
    use_bell = run["shape"] == "bell"
    use_bs = run["shape"] == "band shelf"
    bounds = design.get("bounds") or {}
    G = float(bounds.get("gain_db", INPUTEQ_MAX_DB))
    caps = top_caps(design, nodes, D, fs)
    pr, ar = _corner_range(EQ_PRES_FC_HZ, fs), _corner_range(EQ_AIR_FC_HZ, fs)
    cp, ca = caps["presence"]["cap_db"], caps["air"]["cap_db"]
    cut = min(G, excess)                          # T3: the cut takes the excess, no more
    lo = [-G, math.log2(INPUTEQ_LOW_SHELF_FC_RANGE[0]), -G, math.log2(pr[0]), -G, math.log2(ar[0])]
    hi = [G, math.log2(INPUTEQ_LOW_SHELF_FC_RANGE[1]), cp, math.log2(pr[1]), ca, math.log2(ar[1])]
    bell_fc = (max(INPUTEQ_BELL_FC_RANGE[0], EQ_LOW_MID_RUN_HZ[0]),
               min(INPUTEQ_BELL_FC_RANGE[1], EQ_LOW_MID_RUN_HZ[1]))
    if use_bell:
        lo += [-cut, math.log2(bell_fc[0]), math.log2(INPUTEQ_BELL_Q_RANGE[0])]
        hi += [0.0, math.log2(bell_fc[1]), math.log2(INPUTEQ_BELL_Q_RANGE[1])]
    if use_bs:
        lo += [-cut, math.log2(INPUTEQ_BAND_SHELF_LO_RANGE[0]), INPUTEQ_BAND_SHELF_WIDTH_OCT[0]]
        hi += [0.0, math.log2(min(INPUTEQ_BAND_SHELF_LO_RANGE[1], EQ_LOW_MID_RUN_HZ[1])),
               EQ_BAND_SHELF_MAX_OCT]
    for a in range(len(lo)):                      # a zero cap leaves a degenerate gain range
        if hi[a] <= lo[a]:
            hi[a] = lo[a] + 1e-6

    def params(t) -> dict:
        p = {"ls_gain": t[0], "ls_fc": 2.0 ** t[1], "hs_gain": t[2], "hs_fc": 2.0 ** t[3],
             "air_gain": t[4], "air_fc": 2.0 ** t[5]}
        if use_bell:
            p.update(bell_gain=t[6], bell_fc=2.0 ** t[7], bell_q=2.0 ** t[8])
        if use_bs:
            p.update(bs_gain=t[6], bs_lo=2.0 ** t[7], bs_hi=2.0 ** (t[7] + t[8]))
        return p

    def model(p: dict) -> np.ndarray:
        e = _eq_from_params(q, p, fs, hp_fc)
        e.highpass = None
        return eq_stage.response_db(e, nodes, fs) + hp_extra

    # live: each node's error counts by its audibility (audibility_weights); else 1
    wph = design.get("weight_phon")
    w = audibility_weights(nodes, wph) if wph is not None else np.ones_like(nodes)

    def resid(t):
        return (model(params(t)) - D) * w

    best = None
    sub = D[nodes < SLOPE_RANGE_HZ[0]]
    g_ls0 = float(np.clip(sub.mean() if sub.size else 0.0, -G * 0.99, G * 0.99))
    starts = []
    for fl in EQ_LS_STARTS_HZ:
        for top in _top_starts(0.5 * cp, 0.5 * ca, pr, ar):
            t0 = [g_ls0, math.log2(fl), *top]
            if use_bell:
                k = run["peak_index"]
                t0 += [-0.9 * cut, math.log2(float(np.clip(nodes[k], *bell_fc))),
                       math.log2(EQ_BELL_Q_START)]
            if use_bs:
                # corners half a node outside the run's first and last node
                a_hz = run["run_hz"][0] / 2.0 ** (EQ_NODE_OCT / 2.0)
                b_hz = run["run_hz"][1] * 2.0 ** (EQ_NODE_OCT / 2.0)
                t0 += [-0.9 * cut, math.log2(a_hz), math.log2(b_hz / a_hz)]
            starts.append(np.clip(t0, np.array(lo) + 1e-9, np.array(hi) - 1e-9))
    for t0 in starts:
        r = least_squares(resid, t0, bounds=(lo, hi), method="trf", x_scale="jac")
        if best is None or r.cost < best.cost - 1e-12:
            best = r
    p = params(best.x)
    # readable settings: 0.1 dB, 1 Hz, 0.01 Q (the residual is that of the rounded EQ)
    rp = {k: (round(float(v), 1) + 0.0 if k.endswith("gain") else
              round(float(v), 2) if k.endswith("_q") else
              _top_fc(v) if k in ("hs_fc", "air_fc") else float(round(float(v))))
          for k, v in p.items()}
    if use_bs:
        # the rounded high corner stays inside the width bounds of the rounded low corner
        w = (INPUTEQ_BAND_SHELF_WIDTH_OCT[0], EQ_BAND_SHELF_MAX_OCT)
        rp["bs_hi"] = float(np.clip(rp["bs_hi"], math.ceil(rp["bs_lo"] * 2.0 ** w[0]),
                                    math.floor(rp["bs_lo"] * 2.0 ** w[1])))
    head = caps["noise_headroom_db"]
    if head is not None and max(0.0, rp["hs_gain"]) + max(0.0, rp["air_gain"]) > head + 1e-9:
        rp["air_gain"] = round(min(rp["air_gain"], max(0.0, head - max(0.0, rp["hs_gain"]))), 1)
    for k in ("hs", "air"):                        # rounding never crosses a cap
        c = cp if k == "hs" else ca
        if rp[k + "_gain"] > c:
            rp[k + "_gain"] = math.floor(c * 10.0) / 10.0
    side, side_rec = _fit_side_shelves(design, dc, nodes, q, hp_fc, rp, hp_extra, fs, w)
    if side is not None:
        rp.update(side)
    eq = _eq_from_params(q, rp, fs, hp_fc)
    eq.phase = phase
    eq.design = dict(design)
    fit = model(rp)
    err = fit - D
    full = eq_stage.response_db(eq, nodes, fs)
    fsel = (nodes >= SLOPE_RANGE_HZ[0] - 1e-9) & (nodes <= SLOPE_RANGE_HZ[1] + 1e-9)
    eq_slope = float(np.polyfit(np.log2(nodes[fsel]), full[fsel], 1)[0]) if fsel.sum() > 1 else 0.0
    at_bound = _at_bound(rp, G, caps, side_rec)
    rec = {**dc, "fit_nodes_hz": nodes.tolist(), "fit_deficit_db": D.tolist(),
           "fit_db": fit.tolist(), "eq_response_db": full.tolist(),
           "residual_rms_db": float(np.sqrt(np.mean(err ** 2))) if err.size else 0.0,
           "residual_max_db": float(np.max(np.abs(err))) if err.size else 0.0,
           "low_mid_excess_db": excess, "bell_considered": bool(use_bell),
           "band_shelf_considered": bool(use_bs), "low_mid_shape": run["shape"],
           "low_mid_run_hz": run["run_hz"], "low_mid_run_oct": run["run_oct"],
           "low_mid_neighbour_db": run["neighbour_db"],
           "top_caps": caps, "side_caps": side_rec, "at_bound": at_bound,
           "eq_slope_db_oct": eq_slope,
           "predicted_slope_db_oct": float(design["measured_slope"]) + eq_slope,
           "rotation": float(design.get("rotation", 0.0)),
           "sections": inputeq_sections(eq), "text": inputeq_text(eq)}
    return eq, rec


def _at_bound(rp: dict, G: float, caps: dict, side_rec: dict | None) -> list[str]:
    """The sections whose gain sits at its bound or cap, with the cap's reason (T1: the
    report names them when the slope is missed)."""
    out = []
    if abs(rp.get("ls_gain", 0.0)) >= G - 0.05:
        out.append("low shelf")
    ms = " (M)" if side_rec else ""
    for k, name, c in (("hs", "presence shelf", caps["presence"]),
                       ("air", "air shelf", caps["air"])):
        if rp.get(k + "_gain", 0.0) >= c["cap_db"] - 0.05:
            out.append(f"{name}{ms}: {c['reason']}")
    if side_rec:
        for k, name, c in (("hss", "presence shelf (S)", side_rec["presence"]),
                           ("airs", "air shelf (S)", side_rec["air"])):
            if rp.get(k + "_gain", 0.0) >= c["cap_db"] - 0.05:
                out.append(f"{name}: {c['reason']}")
    return out


def _fit_side_shelves(design: dict, dc: dict, nodes: np.ndarray, hp_q: float, hp_fc: float,
                      rp: dict, hp_extra: np.ndarray, fs: float, weights: np.ndarray | None = None):
    """The side's own presence and air shelves (maintainer, 2026-10-20; Group A), fitted after
    the shared sections: the side deficit is the deficit curve with the side's residual
    (design["side_residual_db"], level-aligned) in place of the stereo one; the shared
    sections (rp without its top shelves) are fixed and only the two side shelves are
    free, fitted on the nodes from the lowest presence corner up, their boosts capped by
    top_caps(side=True). Returns ({"hss_gain", "hss_fc", "airs_gain", "airs_fc"}, caps)
    rounded like the other settings, or (None, None) (no side data: the top end stays
    linked)."""
    from scipy.optimize import least_squares

    from eluate.stages import inputeq as eq_stage
    rs = design.get("side_residual_db")
    if rs is None:
        return None, None
    pr, ar = _corner_range(EQ_PRES_FC_HZ, fs), _corner_range(EQ_AIR_FC_HZ, fs)
    ok = np.array([v is not None for v in rs]) & (nodes >= pr[0] - 1e-9)
    if ok.sum() < 3:
        return None, None
    r_s = np.array([0.0 if v is None else v for v in rs], dtype=np.float64)
    r_m = np.asarray(design["residual_db"], dtype=np.float64)
    defi = np.asarray(dc["deficit_db"])
    D_s_all = defi - np.where(_residual_nodes(nodes, design), r_s, 0.0)
    D_m_all = defi - np.where(_residual_nodes(nodes, design), r_m, 0.0)
    extra = {}
    for name, rng in (("presence", EQ_PRES_DEFICIT_HZ), ("air", EQ_AIR_DEFICIT_HZ)):
        sel = ok & (nodes >= rng[0] - 1e-9) & (nodes <= rng[1] + 1e-9)
        extra[name + "_side_extra"] = float(np.mean(D_s_all[sel] - D_m_all[sel])) if np.any(sel) else 0.0
    mid = {"presence": rp.get("hs_gain", 0.0), "air": rp.get("air_gain", 0.0), **extra}
    bounds = design.get("bounds") or {}
    G = float(bounds.get("gain_db", INPUTEQ_MAX_DB))
    caps = top_caps(design, nodes[ok], D_s_all[ok], fs, mid=mid, side=True)
    D_s = D_s_all[ok]
    shared = {k: v for k, v in rp.items() if not (k.startswith("hs") or k.startswith("air"))}
    base = _eq_from_params(hp_q, shared, fs, hp_fc)
    base.highpass = None
    common = (eq_stage.response_db(base, nodes, fs) + hp_extra)[ok]
    f = nodes[ok]
    ws = np.ones_like(f) if weights is None else np.asarray(weights)[ok]   # the mid fit's weights

    def resid(t):
        shelves = InputEqPlan(high_shelf=EqHighShelf(float(t[0]), float(2.0 ** t[1])),
                              air_shelf=EqHighShelf(float(t[2]), float(2.0 ** t[3])))
        return (common + eq_stage.response_db(shelves, f, fs) - D_s) * ws

    cp, ca = caps["presence"]["cap_db"], caps["air"]["cap_db"]
    lo = [-G, math.log2(pr[0]), -G, math.log2(ar[0])]
    hi = [max(cp, -G + 1e-6), math.log2(pr[1]), max(ca, -G + 1e-6), math.log2(ar[1])]
    best = None
    for t0 in _top_starts(0.5 * cp, 0.5 * ca, pr, ar):
        t0 = np.clip(t0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        r = least_squares(resid, t0, bounds=(lo, hi), method="trf", x_scale="jac")
        if best is None or r.cost < best.cost - 1e-12:
            best = r
    out = {"hss_gain": round(float(best.x[0]), 1) + 0.0, "hss_fc": _top_fc(2.0 ** best.x[1]),
           "airs_gain": round(float(best.x[2]), 1) + 0.0, "airs_fc": _top_fc(2.0 ** best.x[3])}
    for k, c in (("hss", cp), ("airs", ca)):
        if out[k + "_gain"] > c:
            out[k + "_gain"] = math.floor(c * 10.0) / 10.0
    # audit 13 #1: S's pair within the noise headroom too (each channel's floor rises by at
    # most its own pair's boost, so the stereo floor by at most the larger of the two)
    head = caps["noise_headroom_db"]
    if head is not None and max(0.0, out["hss_gain"]) + max(0.0, out["airs_gain"]) > head + 1e-9:
        out["airs_gain"] = math.floor(max(0.0, head - max(0.0, out["hss_gain"])) * 10.0) / 10.0 + 0.0
        if out["hss_gain"] > head:
            out["hss_gain"] = math.floor(head * 10.0) / 10.0
    return out, {**caps, **extra}


def inputeq_sections(eq: InputEqPlan) -> dict:
    """Section parameters as plain dicts (None = off)."""
    from dataclasses import asdict as _ad
    return {k: (None if getattr(eq, k) is None else _ad(getattr(eq, k)))
            for k in ("highpass", "low_shelf", "bell", "band_shelf", "high_shelf",
                      "high_shelf_side", "air_shelf", "air_shelf_side")}


def inputeq_text(eq: InputEqPlan) -> list[str]:
    """One readable line per active section, e.g. "low shelf +1.2 dB at 120 Hz"."""
    out = []
    if eq.highpass is not None:
        out.append(f"high-pass 24 dB/oct at {eq.highpass.fc_hz:g} Hz, Q {eq.highpass.q:.2f}")
    if eq.low_shelf is not None:
        out.append(f"low shelf {eq.low_shelf.gain_db:+.1f} dB at {eq.low_shelf.fc_hz:.0f} Hz "
                   f"(12 dB/oct, Q {eq.low_shelf.q:.2f})")
    if eq.bell is not None:
        out.append(f"bell {eq.bell.gain_db:+.1f} dB at {eq.bell.fc_hz:.0f} Hz, Q {eq.bell.q:.2f}")
    if eq.band_shelf is not None:
        bs = eq.band_shelf
        out.append(f"band shelf {bs.gain_db:+.1f} dB from {bs.lo_hz:.0f} to {bs.hi_hz:.0f} Hz "
                   "(24 dB/oct)")
    ms = eq.ms_top
    for sec, name in ((eq.high_shelf, "presence shelf"), (eq.air_shelf, "air shelf")):
        if sec is not None:
            out.append(f"{name}{' M' if ms else ''} {sec.gain_db:+.1f} dB at {sec.fc_hz:.0f} Hz "
                       "(6 dB/oct)")
    for sec, name in ((eq.high_shelf_side, "presence shelf"), (eq.air_shelf_side, "air shelf")):
        if sec is not None:
            out.append(f"{name} S {sec.gain_db:+.1f} dB at {sec.fc_hz:.0f} Hz (6 dB/oct)")
    return out


def plan_inputeq(analysis: dict, controls: Controls, fs: float, rotation: float = 0.0):
    """(InputEqPlan | None, fit record). None under preserve_mix."""
    design = inputeq_design(analysis, controls)
    design["rotation"] = float(rotation)
    eq, rec = fit_inputeq(design, fs)
    if controls.preserve_mix:
        return None, rec
    return eq, rec


def refit_inputeq(eq: InputEqPlan, fs: float, rotation: float) -> InputEqPlan:
    """The planned EQ refitted with `rotation` dB/oct about PIVOT_HZ added to its target
    (the solver's tone_rotation lever). Needs the design record."""
    if eq.design is None:
        raise ValueError("inputeq has no design record to refit from")
    d = dict(eq.design)
    d["rotation"] = float(rotation)
    return fit_inputeq(d, fs, eq.phase)[0]


def _band_of(f: np.ndarray, crossovers) -> np.ndarray:
    return np.searchsorted(np.asarray(crossovers, dtype=np.float64), f, side="right")


def saturation_drive_db(deficit_db: float, max_db: float = SAT_MAX_DRIVE_DB) -> float:
    """docs/gain-staging.md 6 (maintainer, 2026-10-18): drive = the band's deficit below the
    target slope line, rounded to SAT_STEP_DB (half up), at most SAT_MAX_DRIVE_DB; none
    for a band on or above the line."""
    d = float(deficit_db)
    if not (math.isfinite(d) and d > 0.0):
        return 0.0
    return float(min(max_db, math.floor(d / SAT_STEP_DB + 0.5) * SAT_STEP_DB))


def band_deficits(analysis: dict, controls: Controls) -> list[float | None]:
    """Each band's deficit below the target slope line (dB, > 0 = below), measured on the
    spectrum before the input EQ: the mean over the band's input EQ fit nodes
    (inputeq_design) of the target line minus the LTAS, the target line being the
    measured fit line rotated onto the target slope about PIVOT_HZ (held outside 100 Hz
    to 10 kHz, as in deficit_curve). None for a band with no node."""
    controls = resolve(analysis, controls)
    d = inputeq_design(analysis, controls)
    nodes = np.asarray(d["nodes_hz"], dtype=np.float64)
    res = np.asarray(d["residual_db"], dtype=np.float64)
    delta = float(d["target_slope"]) - float(d["measured_slope"])
    line = delta * np.log2(np.clip(nodes, *SLOPE_RANGE_HZ) / PIVOT_HZ) - res
    xo = list(controls.crossovers_hz)
    band = _band_of(nodes, xo)
    out = []
    for b in range(len(xo) + 1):
        sel = band == b
        out.append(float(np.mean(line[sel])) if np.any(sel) else None)
    return out


def plan_saturate(analysis: dict, controls: Controls):
    """(SaturatePlan | None, record): harmonics are the inverse of the slope
    (docs/gain-staging.md 6). Per band b >= 1 (the sub never), drive_b =
    saturation_drive_db(the band's deficit, band_deficits), on M and S alike; the high-mid
    S band keeps the width rule, at SAT_MAX_DRIVE_DB. The wet share is 1 on every band
    (the drive alone sets the amount). None under preserve_mix."""
    controls = resolve(analysis, controls)
    xo = list(controls.crossovers_hz)
    nb = len(xo) + 1
    dfc = band_deficits(analysis, controls)
    drive = [[0.0, 0.0] for _ in range(nb)]
    rows = []
    for b in range(nb):
        smax = controls.prof["sat_max_db"]
        dm = 0.0 if b == 0 else saturation_drive_db(dfc[b] if dfc[b] is not None else 0.0, smax)
        ds = smax if (nb >= 3 and b == nb - 2) else dm
        drive[b] = [dm, ds]
        rows.append({"band": b, "deficit_db": dfc[b], "drive_m_db": dm, "drive_s_db": ds,
                     "rule": ("sub: never" if b == 0 else
                              "deficit" + ("; S: width rule" if nb >= 3 and b == nb - 2
                                           else ""))})
    rec = {"bands": rows, "step_db": SAT_STEP_DB, "max_db": SAT_MAX_DRIVE_DB,
           "rule": "drive = deficit below the target slope line (before the input EQ), "
                   "rounded to 0.5 dB, at most 2 dB; high-mid S band: width drive 2 dB"}
    if controls.preserve_mix or not controls.prof["saturate"]:
        return None, rec
    return SaturatePlan(drive, [1.0] * nb, xo), rec


# ---------------------------------------------------------------- multiband

def band_peak_levels(analysis: dict) -> np.ndarray:
    """(bands, 2) arrays of gated 400 ms frame peak levels (dBFS) for M and S, as an
    object array of 1-D series (rms_dbfs + crest_db, gated frames only)."""
    cr = analysis["crest"]
    rms = np.asarray(cr["rms_dbfs"], dtype=np.float64)
    cdb = np.asarray(cr["crest_db"], dtype=np.float64)
    nb = rms.shape[0]
    out = np.empty((nb, 2), dtype=object)
    for b in range(nb):
        for c in range(2):
            p = rms[b, c] + cdb[b, c]
            out[b, c] = p[np.isfinite(p)]
    return out


def _per_band(v, nb: int) -> list[float]:
    return [float(v[min(b, len(v) - 1)]) for b in range(nb)]


def _octave_note(octave: dict | None) -> str:
    """The analysis's octave decision (structure._octave) as report text, '' when kept."""
    if not octave or octave.get("decision") != "doubled":
        return ""
    sc = {round(c["bpm"], 6): c["lattice_score"] for c in octave.get("candidates", [])}
    a, b = octave["from_bpm"], octave["to_bpm"]
    return (f"; octave {octave['decision']} from {a:.1f} BPM, sixteenth grid at {b:.1f} "
            f"explains {sc.get(round(b, 6), float('nan')):.2f} against "
            f"{sc.get(round(a, 6), float('nan')):.2f}")


def character_choice(name: str, bpm: float | None, source: str = "user",
                     octave: dict | None = None) -> tuple[str, str]:
    """(preset, reason): a named preset as given; auto by tempo (docs/gain-staging.md 7).
    source "user" (or, with WP10, "midi") is an exact tempo; "analysis" a detected one,
    whose octave is the analysis's decision (octave, quoted in the reason). The
    AUTO_BPM_* boundaries are an octave apart, so every tempo's octave neighbour lies
    across one; an octave error must not change the chain's topology, so a detected
    tempo never resolves to slow (which turns the kick detective on): below
    AUTO_BPM_SLOW it reads moderate. Slow needs an exact tempo."""
    if name != "auto":
        return name, "chosen"
    if bpm is None or not math.isfinite(bpm):
        return AUTO_DEFAULT, "auto, no tempo"
    note = _octave_note(octave) if source == "analysis" else ""
    if bpm < AUTO_BPM_SLOW:
        if source == "analysis":
            return "moderate", (f"auto, detected {bpm:g} BPM < {AUTO_BPM_SLOW:g}: moderate, "
                                f"slow needs a given tempo (--bpm){note}")
        return "slow", f"auto, {bpm:g} BPM < {AUTO_BPM_SLOW:g}"
    if bpm > AUTO_BPM_FAST:
        return "fast", f"auto, {bpm:g} BPM > {AUTO_BPM_FAST:g}{note}"
    return "moderate", f"auto, {AUTO_BPM_SLOW:g} <= {bpm:g} BPM <= {AUTO_BPM_FAST:g}{note}"


def kick_detective_choice(requested: bool | None, preset: str) -> tuple[bool, str]:
    """(on, reason): the user's choice when given; auto (None) turns the kick detective
    on when the character resolves to a preset in KICK_DETECTIVE_AUTO (maintainer,
    2026-10-18: slow or transparent)."""
    if requested is not None:
        return bool(requested), "chosen"
    on = preset in KICK_DETECTIVE_AUTO
    return on, f"auto: character {preset}" + (" (slow or transparent)" if on else "")


def character_settings(analysis: dict, controls: Controls, nb: int | None = None) -> dict:
    """The character preset's settings for nb bands (the analysis's crossovers when nb is
    None), the low band's release held at LOW_BAND_RELEASE_MIN_MS or more. The tempo is
    the user's (controls.bpm) if given, else the analysis's (octave-checked; its octave
    decision goes to character_choice and `tempo_octave` records the analysis's note).
    band_lookahead_ms (per band) is the band ceilings' lookahead, from the character like
    the rest of the multiband timing; speclim_lookahead_ms the frequency-dependent
    limiter's, from the limiter timing (the profile's limiter_timing preset when it sets
    one, the live profile's transparent, else the character's)."""
    if nb is None:
        nb = len(resolve(analysis, controls).crossovers_hz) + 1
    bpm, src = controls.bpm, "user"
    pi = analysis.get("plan_inputs") or {}
    if bpm is None:
        t = pi.get("tempo_bpm")
        bpm, src = (float(t), "analysis") if t is not None and math.isfinite(float(t)) \
            else (None, None)
    octave = (pi.get("metrical") or {}).get("octave")
    name, why = character_choice(controls.character, bpm, src or "analysis", octave)
    pr = CHARACTER_PRESETS[name]
    rel = _per_band(pr["mb_release_ms"], nb)
    rel[0] = max(rel[0], LOW_BAND_RELEASE_MIN_MS)
    kd, kd_why = kick_detective_choice(controls.kick_detective, name)
    prof = controls.prof
    if not prof["kick"]:
        kd, kd_why = False, f"off: {controls.profile} profile"
    clip = prof["clipper_presets"] is None or name in prof["clipper_presets"]
    lim_t = CHARACTER_PRESETS[prof["limiter_timing"]] if prof["limiter_timing"] else pr
    glue_att = prof["glue_attack_ms"] if prof["glue_attack_ms"] is not None \
        else pr["glue_attack_ms"]
    return {"character": controls.character, "preset": name, "reason": why, "bpm": bpm,
            "bpm_source": src, "tempo_octave": pi.get("tempo_octave") if src == "analysis"
            else src, "mb_attack_ms": _per_band(pr["mb_attack_ms"], nb),
            "mb_release_ms": rel, "mb_knee_db": pr["mb_knee_db"],
            "band_lookahead_ms": _per_band(pr["band_lookahead_ms"], nb),
            "speclim_lookahead_ms": _per_band(lim_t["band_lookahead_ms"], nb),
            "glue_attack_ms": glue_att, "glue_release_ms": pr["glue_release_ms"],
            "clip_knee_db": pr["clip_knee_db"], "limiter_attack_ms": lim_t["limiter_attack_ms"],
            "limiter_release_ms": max(lim_t["limiter_release_ms"], LIMITER_RELEASE_MIN_MS),
            "kick_detective": kd and clip, "kick_detective_reason": kd_why,
            "clipper": ("off" if not clip else
                        "kick detective (less aggressive)" if kd else "whole signal"),
            "clipper_reason": ("on" if prof["clipper_presets"] is None else
                               f"{controls.profile} profile: on under "
                               + " and ".join(prof["clipper_presets"]))}


def plan_multiband_record(analysis: dict, controls: Controls) -> dict:
    """Multiband design record from the analysis and the controls: timing and knee from
    the character preset, the budget targets (chunk 3; the thresholds, ratios and
    ceilings are solved on the multiband's input at realise), and as starting values
    the analysis's thresholds (p90 of the band's 400 ms peak - 2 dB)."""
    xo = list(resolve(analysis, controls).crossovers_hz)
    nb = len(xo) + 1
    lv = band_peak_levels(analysis)
    thr = np.zeros((nb, 2))
    for b in range(nb):
        for c in range(2):
            p = lv[b, c]
            # a silent band: threshold at 0 dBFS
            thr[b, c] = float(np.percentile(p, 90.0)) - MB_THRESHOLD_BELOW_P90_DB if p.size \
                else 0.0
    ch = character_settings(analysis, controls, nb)
    dec = [float(np.clip(v, 0.0, 1.0)) for v in analysis["plan_inputs"]["band_decoherence"]]
    return {"threshold_db": thr.tolist(), "ratio": MB_RATIO,
            "attack_ms": [[a, a] for a in ch["mb_attack_ms"]],
            "release_ms": [[r, r] for r in ch["mb_release_ms"]],
            "knee_db": ch["mb_knee_db"], "ceiling_lookahead_ms": ch["band_lookahead_ms"],
            "ceiling_db": None, "crossovers_hz": xo,
            "decoherence": dec, "mid_makeup_db": [0.0] * nb,
            "side_makeup_db": controls.prof["side_makeup_db"],
            "budget": {**mb_stage.default_budget(controls.makeup), **controls.prof["mb_budget"]},
            "ms_link": controls.prof["ms_link"], "character": ch["preset"]}


def multiband_unrealised(rec: dict) -> MultibandPlan:
    nb = len(rec["crossovers_hz"]) + 1
    zero = [[Envelope.constant(0.0), Envelope.constant(0.0)] for _ in range(nb)]
    return MultibandPlan(gr_db=zero, crossovers_hz=list(rec["crossovers_hz"]),
                         mid_makeup_db=list(rec["mid_makeup_db"]),
                         side_makeup_db=rec["side_makeup_db"], decoherence=list(rec["decoherence"]),
                         ceiling_gr_db=None, ceiling_db=rec["ceiling_db"],
                         threshold_db=rec["threshold_db"], ratio=rec["ratio"],
                         attack_ms=rec["attack_ms"], release_ms=rec["release_ms"],
                         knee_db=rec.get("knee_db", 0.0),
                         ceiling_lookahead_ms=rec.get("ceiling_lookahead_ms",
                                                      MultibandPlan.ceiling_lookahead_ms),
                         budget=None if rec.get("budget") is None else dict(rec["budget"]),
                         ms_link=bool(rec.get("ms_link", False)))


def glue_unrealised(threshold_db: float | None, mix: float = GLUE_MIX,
                    release_ms: float = GLUE_RELEASE_MS, mode: str | None = None,
                    attack_ms: float = GLUE_ATTACK_MS, net_db: float | None = None) -> GluePlan:
    """Glue section before realise. mode "serial" / "parallel": the budget solve (chunk 3)
    sets the threshold and, in parallel, the mix at realise; None: the threshold rule of
    glue_threshold_audio and the given mix."""
    return GluePlan(gain_db=Envelope.constant(0.0), mix=1.0 if mode == "serial" else mix,
                    makeup_db=0.0, curve="clean", threshold_db=threshold_db, ratio=GLUE_RATIO,
                    knee_db=GLUE_KNEE_DB, attack_ms=attack_ms, release_ms=release_ms,
                    budget=None if mode is None else
                    {"mode": mode, **({} if net_db is None else {"net_db": float(net_db)})})


HITS_SUBDIVIDE_TOL = 0.1   # a grid interval within 10 % of k beats (k >= 2) is subdivided


def plan_hits(analysis: dict, controls: "Controls") -> HitsPlan:
    """The plan's hits (module contract): the analysis beat grid when it is informative or
    follows the user's --bpm, else 400 ms frames. One hit per beat at the plan's tempo
    (the user's, else the analysis's octave-checked one; sprint S3: a 2-beat hit's
    maximum is the louder beat, which moved every budget 0.85 to 0.94 dB toward less GR):
    a grid whose median interval is k beats (k >= 2, within HITS_SUBDIVIDE_TOL) is
    subdivided k times, and the source says so."""
    pi = analysis.get("plan_inputs") or {}
    beats = pi.get("beat_times")
    if beats is not None:
        beats = [b for b in np.asarray(beats, dtype=np.float64).ravel() if math.isfinite(b)]
    if beats is not None and len(beats) >= 2 and (controls.bpm is not None
                                                 or pi.get("grid_informative")):
        src = "user bpm" if controls.bpm is not None else "analysis beat grid"
        b = np.asarray(beats)
        tempo = controls.bpm if controls.bpm is not None else pi.get("tempo_bpm")
        if tempo is not None and math.isfinite(float(tempo)) and tempo > 0:
            r = float(np.median(np.diff(b))) * float(tempo) / 60.0
            k = int(round(r))
            if k >= 2 and abs(r / k - 1.0) <= HITS_SUBDIVIDE_TOL:
                d = np.diff(b)
                d = np.append(d, d[-1])
                b = (b[:, None] + d[:, None] * np.arange(k)[None, :] / k).ravel()
                dur = (analysis.get("meta") or {}).get("duration_s")
                if dur is not None:
                    b = b[b < float(dur)]
                src += f", subdivided x{k} to {float(tempo):.1f} BPM"
        return HitsPlan(beats_s=[float(v) for v in b], source=src)
    return HitsPlan(beats_s=None, source="400 ms frames")


def design_speclim(plan: Plan, y: np.ndarray, fs: float):
    """The frequency-dependent limiter on y, the glue's output (multiband.design_speclim on
    the multiband's crossovers, over the stage's hits), or None when the limiter section
    does not ask for it or no band needs a ceiling."""
    lp = plan.limiter
    if lp is None or lp.speclim_allow_db is None:
        return None
    xo = (plan.multiband.crossovers_hz if plan.multiband is not None
          else list(bands.DEFAULT_CROSSOVERS))
    nb = len(xo) + 1
    rel = [SPECLIM_RELEASE_MS[min(k, len(SPECLIM_RELEASE_MS) - 1)] for k in range(nb)]
    edges = [0.0] + list(xo)
    allow = [max(lp.speclim_allow_db, SPECLIM_TOP_ALLOW_DB) if edges[k] >= SPECLIM_TOP_FROM_HZ
             else lp.speclim_allow_db for k in range(nb)]
    cut = SPECLIM_MAX_CUT_DB if lp.speclim_max_cut_db is None else lp.speclim_max_cut_db
    look = SPECLIM_LOOKAHEAD_MS if lp.speclim_lookahead_ms is None else lp.speclim_lookahead_ms
    if np.ndim(look):
        look = _per_band(look, nb)
    return mb_stage.design_speclim(y, fs, xo, allow, cut, look, rel,
                                   hits=stage_hits(plan, y, fs))


def stage_hits(plan: Plan, y: np.ndarray, fs: float) -> dict | None:
    """The hits of y, the audio entering a stage, with the loud mask frozen in plan.hits
    (freeze_hits) or, without one, the loud section of y (envelope.hits_for); None (the
    maximum reading) without plan.hits or under reading "max"."""
    h = plan.hits
    if h is None or h.reading == "max":
        return None
    return envelope.hits_for(y, fs, h.beats_s, loud=h.loud)


def freeze_hits(plan: Plan, pre: np.ndarray, fs: float) -> Plan:
    """plan with Plan.hits.loud set to the loud-hit mask of pre, the stage-1 (multiband)
    input (module contract, D2); plan unchanged when it has no hits to read."""
    h = plan.hits
    if h is None or h.reading == "max":
        return plan
    m = envelope.hits_for(pre, fs, h.beats_s)["loud"]
    return replace(plan, hits=replace(h, loud=[bool(v) for v in m]))


def design_glue(y: np.ndarray, fs: float, g: GluePlan, hits: dict | None = None) -> GluePlan:
    """The glue section realised on its input y: the budget solve (glue.design_budget,
    over hits) when g has a budget, else the threshold rule at g's mix."""
    if g.budget is not None:
        return glue_stage.design_budget(y, fs, g.budget["mode"], ratio=g.ratio,
                                        knee_db=g.knee_db, attack_ms=g.attack_ms,
                                        release_ms=g.release_ms, curve=g.curve, hits=hits,
                                        net_db=g.budget.get("net_db", glue_stage.SERIAL_GR_DB))
    base = glue_threshold_audio(y, fs) if g.threshold_db is None else g.threshold_db
    thr = min(base, 0.0)
    return glue_stage.design(y, fs, thr, ratio=g.ratio, knee_db=g.knee_db,
                             attack_ms=g.attack_ms, release_ms=g.release_ms, curve=g.curve,
                             mix=g.mix, makeup_db=g.makeup_db)


GLUE_FRAME_S = 0.4
GLUE_PEAK_PERCENTILE = 95.0
GLUE_FRAME_GATE_DBFS = -70.0


def glue_threshold_audio(y: np.ndarray, fs: float) -> float:
    """Glue threshold from the signal entering the glue stage (spec 3.5 item 6, audit 07
    A9): the 95th percentile over non-overlapping 400 ms frames of the frame's sample
    peak of max(|L|, |R|) (frames above -70 dBFS), minus 6 dB, at most 0 dBFS. A signal
    with no frame above the gate gets 0 dBFS (the glue then does nothing)."""
    y = np.asarray(y, dtype=np.float64)
    L = max(1, int(round(GLUE_FRAME_S * fs)))
    k = max(1, -(-y.shape[0] // L))
    pk = np.zeros(k)
    for j in range(k):
        seg = y[j * L:(j + 1) * L]
        pk[j] = float(np.max(np.abs(seg))) if seg.size else 0.0
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(pk)
    db = db[db > GLUE_FRAME_GATE_DBFS]
    if db.size == 0:
        return 0.0
    return float(min(0.0, np.percentile(db, GLUE_PEAK_PERCENTILE) - GLUE_BELOW_PEAK_DB))


def solve_clipper(y: np.ndarray, fs: float, lp: LimiterPlan,
                  hits: dict | None = None) -> LimiterPlan:
    """The whole-signal clipper's threshold solved on its input y (the glue's output) so the
    output peak (the 8x peak after the curve and its band-limit) falls by lp.clip_target_db
    (maintainer, 2026-10-18): over hits, on the average peak with the outlier capped at
    CLIP_MAX_GR_DB (solver.clip_threshold_for_average_peak, maintainer 2026-10-19); without
    hits on the track's peak (solver.clip_threshold_for_output_peak). lp unchanged
    without a target. A fall more than CLIP_WARN_DB short of the target (the
    band-limit, or the outlier cap reaching into the loud hits) gives a warning
    {"stage": "clipper", "needed_db", "achieved_db", "cap_db"}."""
    if lp.clip_target_db is None:
        return lp
    from eluate import solver
    if hits is None:
        sol = solver.clip_threshold_for_output_peak(y, fs, lp, lp.clip_target_db)
        sol = dict(sol, outlier_gr_db=sol["peak_gr_db"], capped=False)
    else:
        sol = solver.clip_threshold_for_average_peak(y, fs, lp, lp.clip_target_db, hits,
                                                     CLIP_MAX_GR_DB)
    ocap = CLIP_MAX_GR_DB if lp.clip_max_gr_db is None else lp.clip_max_gr_db
    rec = dict(lp.budget or {})
    warn = []
    if sol["peak_gr_db"] < lp.clip_target_db - CLIP_WARN_DB:
        warn.append({"stage": "clipper", "needed_db": float(lp.clip_target_db),
                     "achieved_db": float(sol["peak_gr_db"]), "cap_db": ocap})
    rec["clipper"] = {"target_db": lp.clip_target_db, "peak_gr_db": sol["peak_gr_db"],
                      "outlier_gr_db": sol["outlier_gr_db"], "capped": sol["capped"],
                      "curve_gr_db": sol["curve_gr_db"], "threshold_db": sol["threshold_db"],
                      "knee_db": lp.clip_knee_db, "input_peak_8x_dbfs": sol["peak_8x_dbfs"],
                      "budget_db": CLIP_MAX_GR_DB, "outlier_cap_db": ocap,
                      "sample_cap_db": lp.clip_max_gr_db,
                      "bound": sol.get("bound"),
                      "split_hz": lp.clip_split_hz,
                      "measured_on": ("output peak" if lp.clip_split_hz is None
                                      else f"output peak above {lp.clip_split_hz:g} Hz"),
                      "reading": "max" if hits is None else "average", "warnings": warn}
    return replace(lp, clip_threshold_db=sol["threshold_db"], budget=rec)


def limiter_peak(y: np.ndarray, fs: float, lp: LimiterPlan,
                 hits: dict | None = None) -> LimiterPlan:
    """Record the limiter's 8x detector peaks at 0 dB input gain on its input y (after the
    clipper and band-limit: solver.limiter_peaks), the track's and, over hits, the loud
    hits' median, which clamp_limiter_gain reads. lp unchanged without a budget
    (max_gr_db None)."""
    if lp.max_gr_db is None:
        return lp
    from eluate import solver
    pk = solver.limiter_peaks(y, fs, lp, hits)
    fin = math.isfinite(pk["peak_dbfs"])
    rec = dict(lp.budget or {})
    rec["limiter"] = {"clipped_peak_dbfs": pk["peak_dbfs"] if fin else None,
                      "clipped_average_peak_dbfs": pk["average_peak_dbfs"] if fin else None,
                      "budget_db": lp.max_gr_db,
                      "outlier_cap_db": limiter_outlier_cap(lp),
                      "ceiling_dbtp": lp.ceiling_dbtp,
                      "reading": "max" if hits is None else "average"}
    return replace(lp, budget=rec)


def limiter_outlier_cap(lp: LimiterPlan) -> float:
    """The limiter's outlier cap: lp.max_outlier_gr_db, else max_gr_db +
    envelope.OUTLIER_ALLOWANCE_DB (plans before the audible maxima)."""
    if lp.max_outlier_gr_db is not None:
        return float(lp.max_outlier_gr_db)
    return float(lp.max_gr_db) + envelope.OUTLIER_ALLOWANCE_DB


def clamp_limiter_gain(lp: LimiterPlan, requested_db: float) -> LimiterPlan:
    """The limiter's input gain: requested_db (the gain the loudness target asks for)
    clamped to the budget: an average peak GR of at most max_gr_db and an outlier peak
    GR of at most limiter_outlier_cap(lp) (solver.limiter_gain_max on
    the recorded clipped peaks at 0 dB, limiter_peak; a record without the average
    peak reads the track's peak for both); a warning in the budget record when the
    clamp binds. The request is first capped at the maximum gain + LIMITER_REQUEST_OVER_DB
    (S5: requested_uncapped_db and request_capped record it), so the warning's needed_db
    (the GR the capped request would take) never reads a failed proxy solve. Pure."""
    from eluate import solver
    req = float(requested_db)
    rec = dict(lp.budget or {})
    lr = dict(rec.get("limiter") or {})
    p = lr.get("clipped_peak_dbfs")
    if lp.max_gr_db is None or p is None or not math.isfinite(p):
        return replace(lp, input_gain_db=req, requested_gain_db=req)
    pa = lr.get("clipped_average_peak_dbfs")
    pa = p if pa is None or not math.isfinite(pa) else pa
    ocap = limiter_outlier_cap(lp)
    g_max = solver.limiter_gain_max(p, pa, lp.ceiling_dbtp, lp.max_gr_db, ocap)
    raw = req
    req = min(req, g_max + LIMITER_REQUEST_OVER_DB)
    g = min(req, g_max)
    warn = []
    bound = None
    if req > g_max + 1e-9:
        bound = solver.limiter_bound(p, pa, lp.ceiling_dbtp, lp.max_gr_db, ocap)
        warn.append({"stage": "limiter", "requested_db": req, "max_db": g_max,
                     "needed_db": pa + req - lp.ceiling_dbtp, "cap_db": lp.max_gr_db,
                     "outlier_needed_db": p + req - lp.ceiling_dbtp, "outlier_cap_db": ocap,
                     "bound": bound, "request_capped": raw > req,
                     "requested_uncapped_db": raw})
    lr.update(requested_gain_db=req, max_gain_db=g_max, input_gain_db=g, bound=bound,
              outlier_cap_db=ocap,
              planned_gr_db=max(0.0, pa + g - lp.ceiling_dbtp),
              planned_outlier_gr_db=max(0.0, p + g - lp.ceiling_dbtp), warnings=warn)
    rec["limiter"] = lr
    return replace(lp, input_gain_db=g, requested_gain_db=req, budget=rec)


def design_limiter(y: np.ndarray, fs: float, lp: LimiterPlan,
                   requested_db: float | None = None, hits: dict | None = None) -> LimiterPlan:
    """Clipper threshold, then the limiter's input gain within its budget, on the glue's
    output y (chain order), both over hits. requested_db, when given, is the gain asked
    for; else, with lp.target_rms_dbfs, the gain at which the proxy limiter's output
    reaches that reading on the maintainer's meter (solver.gain_for_rms, uncalibrated: the solver
    calibrates the proxy on real renders); else lp.requested_gain_db or lp.input_gain_db.
    When the budget clamps the gain, the limiter warning carries the proxy's estimate of
    the loudness shortfall (shortfall_db; the solver replaces it with the measured
    one)."""
    lp = limiter_peak(y, fs, solve_clipper(y, fs, lp, hits), hits)
    from eluate import solver
    if requested_db is not None:
        req = requested_db
    elif lp.target_rms_dbfs is not None:
        req = solver.gain_for_rms(y, fs, lp, lp.target_rms_dbfs, lp.input_gain_db)
    else:
        req = lp.requested_gain_db if lp.requested_gain_db is not None else lp.input_gain_db
    if lp.max_gr_db is None:
        return replace(lp, input_gain_db=float(req), requested_gain_db=float(req))
    lp = clamp_limiter_gain(lp, req)
    if lp.target_rms_dbfs is not None:
        lr = lp.budget["limiter"]
        if lr.get("warnings"):
            est = solver.proxy_rms(y, fs, lp, lp.input_gain_db)
            lr["warnings"] = [dict(w, shortfall_db=float(lp.target_rms_dbfs - est),
                                   shortfall_source="proxy") for w in lr["warnings"]]
    return lp


def _status(avg, target, outlier, cap) -> str:
    """"over" when the average passes its target or the outlier its cap, "under" when the
    average is more than envelope.AVERAGE_WARN_DB below its target, else "ok"."""
    if avg is None:
        return "n/a"
    if (target is not None and avg > target + 0.05) or (
            outlier is not None and cap is not None and outlier > cap + 0.05):
        return "over"
    if target is not None and avg < target - envelope.AVERAGE_WARN_DB:
        return "under"
    return "ok"


def budget_report(plan: Plan, lim_info: dict | None = None) -> dict:
    """Per-stage average and outlier peak GR against the budget and the outlier cap
    (docs/gain-staging.md 3, maintainer 2026-10-19), with every warning, from a realised plan's
    records; lim_info (the limiter's info of a render) adds the rendered clipper and
    limiter peak GR. JSON-safe."""
    out: dict = {"stages": {}, "warnings": [],
                 "reading": "max" if plan.hits is None else plan.hits.reading,
                 "hits": None if plan.hits is None else plan.hits.source}
    mb = plan.multiband
    if mb is not None and mb.budget is not None and mb.budget.get("result") is not None:
        rows = []
        for row in mb.budget["result"]:
            pair = []
            for r in row:
                d = {k: r.get(k) for k in ("comp_db", "comp_outlier_db", "limit_db",
                                           "limit_outlier_db", "total_db", "total_outlier_db",
                                           "budget_db", "comp_target_db", "comp_cap_db",
                                           "limit_cap_db", "exception", "ratio", "makeup_db",
                                           "gated")}
                comp_cap = r.get("comp_cap_db")
                st = _status(r.get("comp_db"), r.get("comp_target_db"),
                             r.get("comp_outlier_db"), comp_cap)
                if st != "over" and (r.get("limit_outlier_db") or 0.0) > \
                        (r.get("limit_cap_db") or math.inf) + 0.05:
                    st = "over"
                if st != "over" and (r.get("total_db") or 0.0) > (r.get("budget_db") or math.inf) + 0.05:
                    st = "over"
                d["status"] = st
                pair.append(d)
            rows.append(pair)
        out["stages"]["multiband"] = {"makeup": mb.budget.get("makeup"), "bands": rows}
        out["warnings"] += list(mb.budget.get("warnings") or [])
    g = plan.glue
    if g is not None and g.budget is not None and "net_db" in g.budget:
        serial = g.budget["mode"] == "serial"
        cap = g.budget.get("net_cap_db", glue_stage.OUTLIER_CAP_DB)
        out["stages"]["glue"] = {
            "mode": g.budget["mode"], "net_db": g.budget["net_db"],
            "net_outlier_db": g.budget.get("net_outlier_db"), "wet_db": g.budget["wet_db"],
            "mix": g.budget["mix"], "net_target_db": g.budget["net_target_db"],
            "budget_db": glue_stage.SERIAL_RANGE_DB[1] if serial else glue_stage.WET_RANGE_DB[1],
            "outlier_cap_db": cap,
            "status": _status(g.budget["net_db"], g.budget["net_target_db"],
                              g.budget.get("net_outlier_db"), cap)}
        out["warnings"] += list(g.budget.get("warnings") or [])
    lp = plan.limiter
    if lp is not None and lp.budget is not None:
        c = lp.budget.get("clipper")
        if c is not None:
            out["stages"]["clipper"] = {"planned_db": c["peak_gr_db"], "target_db": c["target_db"],
                                        "outlier_db": c.get("outlier_gr_db", c["peak_gr_db"]),
                                        "budget_db": c["budget_db"],
                                        "outlier_cap_db": c.get("outlier_cap_db", c["budget_db"]),
                                        "curve_db": c.get("curve_gr_db"),
                                        "bound": c.get("bound"),
                                        "measured_on": c.get("measured_on", "output peak")}
            p_in, det = c.get("input_peak_8x_dbfs"), (lim_info or {}).get("detector_peak_dbfs")
            if det is not None and p_in is not None and math.isfinite(p_in):
                # the track peak's fall: the clipper's 8x input peak at the rendered gain
                # minus the limiter detector's peak (after the clipper and band-limit)
                out["stages"]["clipper"]["rendered_db"] = float(p_in + lp.input_gain_db - det)
            cs = out["stages"]["clipper"]
            cs["status"] = _status(cs["planned_db"], c["target_db"], cs["outlier_db"],
                                   cs["outlier_cap_db"])
            out["warnings"] += list(c.get("warnings") or [])
        lr = lp.budget.get("limiter")
        if lr is not None:
            st = {"planned_db": lr.get("planned_gr_db"),
                  "outlier_db": lr.get("planned_outlier_gr_db"), "budget_db": lr.get("budget_db"),
                  "outlier_cap_db": lr.get("outlier_cap_db"),
                  "input_gain_db": lr.get("input_gain_db"),
                  "requested_gain_db": lr.get("requested_gain_db"),
                  "target_rms_dbfs": lp.target_rms_dbfs, "bound": lr.get("bound")}
            if lim_info:
                st["rendered_db"] = lim_info.get("max_gain_reduction_db")
                st["outlier_db"] = st["rendered_db"]
            # the limiter's target is the loudness: its budget is a ceiling, never "under"
            st["status"] = _status(st["planned_db"], None, st["outlier_db"], st["outlier_cap_db"])
            if st["planned_db"] is not None and st["budget_db"] is not None \
                    and st["planned_db"] > st["budget_db"] + 0.05:
                st["status"] = "over"
            out["stages"]["limiter"] = st
            out["warnings"] += list(lr.get("warnings") or [])
    return out


# ---------------------------------------------------------------- top level

def plan_from_analysis(analysis: dict, controls: Controls, x: np.ndarray | None = None) -> Plan:
    """Fill every plan section from the analysis and the controls (module contract)."""
    controls = resolve(analysis, controls)
    meta, pi = analysis["meta"], analysis["plan_inputs"]
    fs = int(round(float(meta["fs"])))
    notch, _, _ = plan_notches(analysis, kick=controls.prof["kick"])
    side = SidestripPlan(subside_hz(analysis, controls))
    mb = multiband_unrealised(plan_multiband_record(analysis, controls))
    sat, _ = plan_saturate(analysis, controls)   # the FIR tone stage is off (chunk 2)
    inputeq, _ = plan_inputeq(analysis, controls, fs)
    inp = meta.get("input")
    # starting gain for the RMS target: the analysed audio sits at R_ref after the input
    # normalisation (S95 stands in for an analysis made without it); design_limiter and
    # the solver solve the real one. R_ref is a 1 s plain reading; the target is on the
    # maintainer's meter (AES17), so the start adds the AES17 offset
    s95 = float(pi["s95_lufs"])
    lvl = (float(inp["r_ref_dbfs"]) if inp is not None else s95) + meter.AES17_OFFSET_DB
    gain0 = controls.rms_target - lvl if math.isfinite(lvl) else 0.0
    prof = controls.prof
    onsets = usable_kick_onsets(analysis) if prof["kick"] else []
    ch = character_settings(analysis, controls)
    clip = ({} if ch["clipper"] == "off" else
            {"clip_depth_db": CLIP_DEPTH_DB if onsets else 0.0} if ch["kick_detective"] else
            {"clip_target_db": prof["clip_target_db"], "clip_max_gr_db": prof["clip_sample_cap_db"],
             "clip_split_hz": float(mb.crossovers_hz[0])})
    lim = LimiterPlan(ceiling_dbtp=controls.ceiling_dbtp, input_gain_db=float(gain0),
                      attack_ms=ch["limiter_attack_ms"], release_ms=ch["limiter_release_ms"],
                      clip_onsets_s=onsets, max_gr_db=prof["limiter_max_gr_db"],
                      requested_gain_db=float(gain0), clip_knee_db=ch["clip_knee_db"],
                      target_rms_dbfs=controls.rms_target,
                      max_outlier_gr_db=prof["limiter_outlier_cap_db"],
                      speclim_allow_db=prof["speclim_allow_db"] if controls.speclim else None,
                      speclim_max_cut_db=prof["speclim_max_cut_db"],
                      speclim_lookahead_ms=ch["speclim_lookahead_ms"], **clip)
    plan = Plan(fs=fs, preserve_mix=controls.preserve_mix,
                input=None if inp is None else InputPlan(**inp), inputeq=inputeq, notch=notch,
                sidestrip=side, multiband=mb, tone=None, saturate=sat,
                glue=glue_unrealised(None, mode=controls.glue_mode,
                                     attack_ms=ch["glue_attack_ms"],
                                     release_ms=ch["glue_release_ms"],
                                     net_db=prof["glue_net_db"]),
                keyeq=KeyEqPlan(enabled=False),
                limiter=lim, output=controls.output_plan(),
                hits=plan_hits(analysis, controls))
    if x is not None:
        plan = realise(plan, x, fs)
    return plan


def realise(plan: Plan, x: np.ndarray, fs: int, pre=None) -> Plan:
    """Design the audio-dependent envelopes (multiband, glue) from their design records,
    and the clipper threshold and limiter gain (chunk 3), each solved in chain order on
    the previous stage's real output.

    pre: the input gain + inputeq + notch + sidestrip output if already rendered (the
    solver's cache); otherwise it is rendered here, in chain order (the gain first, so
    every level-dependent design below sees the normalised audio). The loud hits are
    frozen on pre (freeze_hits) and returned in the plan's hits. Returns a new Plan
    (the input is not modified)."""
    from eluate.stages import gain as _g, inputeq as _eq, notch as _n, saturate as _s
    from eluate.stages import sidestrip as _ss, tone as _t
    x = np.asarray(x, dtype=np.float64)
    if pre is None:
        pre = x if plan.input is None else _g.apply(x, fs, plan.input)
        if plan.inputeq is not None and not plan.preserve_mix:
            pre = _eq.apply(pre, fs, plan.inputeq)
        if plan.notch is not None:
            pre = _n.apply(pre, fs, plan.notch)
        if plan.sidestrip is not None:
            pre = _ss.apply(pre, fs, plan.sidestrip)
    plan = freeze_hits(plan, pre, fs)
    mb = plan.multiband
    y = pre
    if mb is not None:
        mb = mb_stage.design_from(mb, pre, fs, hits=stage_hits(plan, pre, fs))
        y = mb_stage.apply(pre, fs, mb)
    if plan.tone is not None and not plan.preserve_mix:
        y = _t.apply(y, fs, plan.tone)
    if plan.saturate is not None:
        y = _s.apply(y, fs, plan.saturate)
    gl = plan.glue
    if gl is not None:
        gl = design_glue(y, fs, gl, stage_hits(plan, y, fs))
        y = glue_stage.apply(y, fs, gl)
    sl = design_speclim(plan, y, fs)
    if sl is not None:
        y = mb_stage.apply(y, fs, sl)
    lp = plan.limiter
    if lp is not None:
        lp = design_limiter(y, fs, lp, hits=stage_hits(plan, y, fs))
    return replace(plan, multiband=mb, glue=gl, speclim=sl, limiter=lp)


def plan_record(analysis: dict, controls: Controls) -> dict:
    """Everything the planner decided, for the report (JSON-safe)."""
    xo_rec = crossover_choice(analysis, controls)
    controls = resolve(analysis, controls)
    _, rows, merged = plan_notches(analysis, kick=controls.prof["kick"])
    tone = tone_deficit(analysis, controls)
    _, sat = plan_saturate(analysis, controls)
    fs = float(analysis["meta"]["fs"])
    _, eq = plan_inputeq(analysis, controls, fs)
    ch = character_settings(analysis, controls)
    prof = controls.prof
    net = glue_stage.SERIAL_GR_DB if prof["glue_net_db"] is None else prof["glue_net_db"]
    return {"controls": controls.to_dict(), "crossovers": xo_rec,
            "profile": {"name": controls.profile, "calibrated": prof["calibrated"]},
            "input": analysis["meta"].get("input"),
            "output": asdict(controls.output_plan()), "ceiling_dbtp": controls.ceiling_dbtp,
            "notches": rows, "merged_notches": merged,
            "tone": tone, "inputeq": eq, "saturation": sat,
            "multiband": plan_multiband_record(analysis, controls),
            "character": ch,
            "glue": {"mode": controls.glue_mode, "ratio": GLUE_RATIO,
                     "net_target_db": net,
                     "wet_target_db": (net if controls.glue_mode == "serial"
                                       else glue_stage.WET_GR_DB)},
            "clipper": ({"mode": "off", "reason": ch["clipper_reason"]}
                        if ch["clipper"] == "off" else
                        {"mode": "kick detective", "depth_db": CLIP_DEPTH_DB,
                         "reason": ch["kick_detective_reason"]}
                        if ch["kick_detective"] else
                        {"mode": "whole signal", "target_db": prof["clip_target_db"],
                         "max_db": CLIP_MAX_GR_DB, "measured_on": "output peak",
                         "reason": ch["kick_detective_reason"]}),
            "limiter": {"max_gr_db": prof["limiter_max_gr_db"],
                        "outlier_cap_db": prof["limiter_outlier_cap_db"],
                        "ceiling_dbtp": controls.ceiling_dbtp},
            "loudness": {"target_rms_dbfs": controls.rms_target,
                         "tolerance_db": TARGET_RMS_TOL_DB,
                         "meter": "300 ms RMS, AES17, loudest passage (an Ozone-style RMS meter)"},
            "glue_threshold_rule": ("solved for the glue's peak GR budget on its input (set at "
                                    "realise)"),
            "band_centres_hz": band_centres(controls.crossovers_hz)}
