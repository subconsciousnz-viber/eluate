# Live profile audit: calibration proposal and wrong-for-live behaviours (2026-11-10)

Repo `<repo>/` (branch audit/2026-11-10), read-only; no repo file was changed.
Scratch: `<scratch>/audit/live/`
(`fx/` fixtures + `*.beats.json` ground truth, `out/<fixture>-<profile>/` full CLI output + `audit_measure.json`
+ `*.measure.log`, `scripts/` = `make_fixtures.py`, `run_batch.py`, `measure_run.py`, `aggregate.py`,
`analysis_peek.py`; `evidence_tables.md` = tables T1-T6, reproduced below).

## 0. Method

Four synthetic 48 kHz stereo fixtures, 36-44 s, no third-party audio (`scripts/make_fixtures.py`):

- `jazz_trio`: swing ride (inharmonic partials + HP noise), feathered kick (85->45 Hz sweep), cross-stick /
  snare comping, upright bass (9 partials, 22 ms attack), inharmonic additive piano (B = 4e-4), 120 BPM with
  +-8 ms jitter and 0.4 %/bar tempo drift, head / solo crescendo / quiet bass solo / out; 1.1 s decorrelated
  noise-IR room at -16 dB.
- `folk_gtr_vocal`: Karplus-Strong 6-string strums (D-DU-UDU) at 76 BPM with a ritardando over the last
  three bars, body resonance, formant-filtered harmonic "vocal" (5 vowels, 5.6 Hz vibrato, portamento,
  breaths), verse -5 dB / chorus; 1.4 s room at -17 dB.
- `rock_band`: two tanh-saturated additive-saw power-chord guitars hard L/R through a cab filter, saw bass,
  kick / snare / hats / crashes, intro (clean) / verse (chugs) / chorus (open chords), 140 BPM; 0.45 s drum room.
- `classical_strings`: five-voice additive string ensemble (bass, cello, viola, vln II, vln I; 2-4 players per
  voice with +-7 cents detune and independent 5-6 Hz vibrato), 13 chords with rubato lengths 2.0-6.5 s,
  pp -> mf -> p -> ff -> ppp curve (-28 .. +1 .. -34 dB); 2.2 s hall at -9 dB.

Premasters (engine meter, `meter.measure`):

| fixture | peak dBFS / dBTP | integrated LUFS | S-max LUFS | LRA LU | crest dB (peak - loud 1 s RMS) | meter p95 (300 ms AES17) | PLR | PSR p95 |
|---|---|---|---|---|---|---|---|---|
| jazz_trio | -2.0 / -1.97 | -17.96 | -14.2 | 7.4 | 15.7 | -14.36 | 16.0 | 14.6 |
| folk_gtr_vocal | -2.0 / -1.99 | -19.18 | -16.4 | 5.8 | 17.5 | -15.99 | 17.2 | 17.2 |
| rock_band | -1.5 / -1.42 | -15.78 | -13.5 | 12.1 | 14.2 | -11.86 | 14.4 | 15.8 |
| classical_strings | -3.0 / -3.0 | -18.30 | -13.0 | 21.0 | 12.5 | -12.44 | 15.3 | 12.5 |

(Bowed strings have no transients, so the classical crest is 12.5 dB by nature; rock sits at the asked 12-14 dB;
folk/rock integrated is 2-3 dB above -18 because their loud sections are dense.)

Each fixture was mastered with `.venv/Scripts/python.exe -m eluate.cli master <wav> -o <dir> --profile live`
and `--profile electronic` (defaults otherwise, 12 threads; 8 runs, all rc = 0). Measurements
(`scripts/measure_run.py`): before/after `meter.measure`, `multiband.band_crest`, `features.tilt`,
`features.stereo`, sub-image (L/R level difference and S energy below 60 Hz); listen-proxies from the saved plan:
multiband / glue / speclim gain envelopes (`plan.npz`, 1 ms), limiter GR (`limiter_gr.npz`, 50 ms), lock-in of
each GR series at the fixture's true beat rate over the loud section (pumping depth, dB), pre-onset energy
2-22 ms before every detected onset (`transients.onset_times`) relative to the 20 ms after it, output minus
input (pre-ring), notch list, speclim rows, tempo grid against the fixture's true beats, budget table, flags.
The output is sample-aligned with the input on every run (cross-correlation lag 0), so all deltas are direct.

---

## A. Findings (sorted by severity)

### S1 - audible / wrong audio for live material

**A1. The whole-signal clipper switches itself on under live whenever the detected tempo is above 140 BPM.**
`planner.py:1721-1743` (`character_choice`: bpm > AUTO_BPM_FAST -> "fast"), `planner.py:508`
(`PROFILES["live"]["clipper_presets"] = ("fast", "aggressive")`, comment "maintainer: off unless asked for"),
`planner.py:1781` (`clip = prof["clipper_presets"] is None or name in prof["clipper_presets"]` - it cannot tell
an auto-resolved "fast" from a user-chosen one).
Evidence: `rock_band-live` (analysis tempo 140.108 BPM): report "Clipper: whole signal, threshold -9.89 dBFS,
knee 0.5 dB, output-peak target 1.0 dB"; clipper curve max 2.61 dB on 106 spans; clipper planned 0.99 dB avg /
2.52 dB outlier; the limiter did 0.00 dB. `character_settings` with the jazz analysis and `bpm=141` or `150`
also returns "fast / clipper whole signal". A 141 BPM live band gets clipped with a 0.5 dB knee; a 139 BPM one does not.
Minimal fix: `"clipper_presets": ()` for live (clipper off), or keep the tuple but gate it on an explicit
character: `clip = prof["clipper_presets"] is None or (controls.character != "auto" and name in prof["clipper_presets"])`
(one line in `character_settings`). Confidence: high.

**A2. Notches are rendered on the partials of sustained notes.** `planner.py:975-1060` (`plan_notches`) renders
candidates flagged "musical" at depth 0.25 (`resonance.py:131 MUSICAL_DEPTH_SCALE`); the suspect-span guard
(`NOTCH_SUSPECT_SPAN 0.6`, `planner.py:361`) and the tonic/fifth guard catch only some.
Evidence: `classical_strings-live` (an additive ensemble with no resonance): 24 candidates, **9 rendered**
(merged: 2014 Hz Q 6 1.5 dB, 2480 Hz Q 8 0.5 dB, 3612 Hz Q 3.4 1.0 dB, 5212 Hz Q 6.6 1.0 dB, 8875 Hz Q 16 1.0 dB;
all flag "musical", channel both/M); 10 others were only saved by the 60 % span rule (e.g. 6031 Hz spans 95 % of
the track). On the transient fixtures the "tonal" flag correctly suppressed 75/78 (jazz), 89/91 (folk), 62/65 (rock)
candidates, 0 rendered. So the failure mode is specifically long held chords with vibrato (strings, organ, choir):
the vibrato smears the line past `LINE_BINS`, the ladder test says "musical", and 0.5-1.5 dB dynamic notches then
ride the partials of the music.
Minimal fix: under live render only `flag == "resonance"` candidates (depth_scale 1), never "musical"; or set
`NOTCH_MAX = 0` when the material is sustained (onsets_per_s < 0.5, see A6). Confidence: high (mechanism is
generic; depth is small but it is a moving notch on the music).

**A3. The live input EQ still reshapes the spectrum by +-1.5 dB broadband at Tilt 0.** The "mix's own slope"
rule (`planner.py:1073-1080`) only zeroes the slope term; the residual correction then flattens the LTAS toward
a straight line inside 100 Hz-10 kHz (`eq_residual_window`, `planner.py:521`, `_residual_nodes 1330`,
`fit_inputeq 1342`) with every section allowed +-1.5 / +2.0 dB (`eq_bounds`, `planner.py:519`). For acoustic
material the departure of the LTAS from a straight line *is* the instrumentation (ride-cymbal band, bass-note
region, string formants), not an error.
Evidence (live runs, Tilt 0): `jazz_trio`: bell -1.5 dB at 131 Hz Q 3.2 (the walking-bass note region), low
shelf -0.8 dB, presence M -1.5 dB at 2980 Hz, air M -1.5 dB at 10 kHz, presence S -1.5 dB at 1230 Hz, air S
-1.5 dB; LTAS change (mean removed) 250-800 Hz +1.3..+1.9 dB against 12.7-16 kHz -0.5..-0.8 dB, i.e. a 2.5 dB
tonal rotation that the slope figure (-4.70 -> -4.85) does not show. `classical_strings`: low shelf +1.5 dB at
220 Hz, bell -1.5 dB at 398 Hz Q 0.7, presence M and S -1.5 dB at 1000 Hz: LTAS 31-125 Hz +1.1..+1.7 dB against
2-16 kHz -1.0..-1.4 dB. `folk`: low shelf +1.5 dB at 168 Hz, bell -1.5 dB at 419 Hz, presence M +1.1 dB.
Every section sits at its bound, which is the signature of a fit chasing the material rather than a defect.
Minimal fix: at `tilt_db == 0` under live, disable the presence / air shelves (M and S) and the low shelf; keep
the 25 Hz Butterworth HPF and a bell only for narrow (Q >= 2) excesses above +4 dB; or halve the live bounds to
+-0.75 dB. The electronic runs show what the full fit does on the same material (folk: air S +5.4 dB, classical:
presence S +6.0 dB, air +4.5/+6.0 dB), so the live rule is the right direction, just not finished. Confidence: medium-high.

### S2 - wrong numbers / reports

**A4. Under `ms_link` the side channel receives up to 2.5 dB more reduction than the table reports, and a
"gated" S band is not left alone.** `multiband.py:199-203` and `:247-262` apply `min(M, S)` gain to both; the
budget is solved and reported per channel (`design_budget` rows, `planner.budget_report 2126-2157`); the module
docstring (`multiband.py:26-28`) admits "a channel's applied GR can exceed its own budget's reading".
Evidence: `jazz_trio-live` band 4 S row says `gated: true, comp 0.0` while the applied linked GR on S is 0.42 dB
mean / 3.72 dB max; band 1 S extra over its own solve 0.17 dB mean / 0.88 p95 / 2.47 max; `rock_band-live`
band 0 S extra 0.38 mean / 2.45 max; `classical_strings-live` band 2 S extra up to 2.11 dB. Pumping of the room
is not the problem (lock-in at the beat rate 0.08-0.42 dB on every live band, glue <= 0.18 dB): the problem is
that the report's S numbers are not what was rendered.
Minimal fix: solve the linked bands on the linked detector (max of M and S peaks) so one threshold and one
reported row describe what is applied; or report the applied (linked) average/outlier per band in
`budget_report`. Confidence: high.

**A5. The tilt check is structurally dead under live.** `planner.target_slope` (1073-1080) sets the target to
the measured slope + Tilt; `evaluate._checks` (444-452) flags `|value - target| > tol`, so a mix that is 3 dB/oct
too dark or too bright is never flagged and the user is not told.
Evidence: with the jazz analysis re-tilted to -7.70 and -1.70 dB/oct, live gives target -7.70 / -1.70, predicted
-7.83 / -1.83, `tilt` flag False with no text; electronic would have corrected toward -4.5 (predicted -6.19 /
-2.38). The UI brief says "Tilt under Live moves from the mix's own slope" - fine as a control law, but the
report should still say where the mix sits: "the mix is 3.2 dB/oct darker than the -4.5 reference; Tilt +1.5
would ...". Minimal fix: a live-only informational check against `REFERENCE_SLOPE` with a wide tolerance
(+-2 dB/oct; drumless material is naturally darker, Elowsson & Friberg 2017). Confidence: high.

**A6. A fictitious beat grid is trusted on rubato / sustained material, and the budget table and character
reason cite it.** `structure.py:822` (`grid_informative = kind != "default"`), `planner.plan_hits 1862-1890`
(uses the grid whenever it is "informative"), `character_settings 1769-1773` (tempo from the analysis).
Evidence: `classical_strings` (13 rubato chords, 4 detected onsets in 44 s): structure returns a "tracked"
grid at 105.4 BPM, `grid_informative = True`, 78 beats vs 13 chord onsets, median grid error 825 ms, 0 % within
50 ms; the plan's hits are "analysis beat grid", the character reason reads "auto, 70 <= 105.41 BPM <= 140".
Audio consequence on this fixture is small (the reading becomes "the 16 loudest 0.57 s segments", 1.0 dB, where
400 ms frames would read 1.08-1.36 and the true chord onsets 0.00 because fewer than half the chords are loud),
but the provenance in report.md and process-log is false and the tempo-driven character is noise. On the
transient fixtures the grid is good (jazz 118.6 BPM, 57 % of beats within 50 ms of the swung truth, folk 86 %,
rock 93 %) and the GR reading moves by at most 0.23 dB between the analysis grid and the true beats; 400 ms
frames would under-read by 0.1-0.4 dB (so the solver would compress 10-40 % harder), a half-tempo grid over-reads
by 0.2-0.6 dB. Minimal fix: `grid_informative` also requires `transients.onsets_per_s >= 1` and at least half
the beats within a quarter interval of an onset; otherwise 400 ms frames plus a flag "no beat grid: sustained
material". Confidence: high.

**A7. The report's dynamics flags are electronic flags.** `evaluate.py:96-99` (`FLAG_LRA_LU 5`,
`FLAG_CREST_DB 7.1` = the electronic masters' lowest crest), `:492`. Live runs lost 1.1-4.0 dB of crest,
1.4-4.4 dB of PLR and 2.1-2.4 LU of LRA with **zero flags** on all four; nothing reports PSR or PLR at all,
nothing says "clipper engaged under live", "N notches on sustained tonal material", "beat grid not trusted".
Minimal fix (live): flag PSR p95 after < 10 dB (Dynameter "Competitive" floor; < 8 = Shepherd's minimum, amber),
PLR after < 12 dB (Katz "good"), LRA drop > 3 LU when LRA before > 12, crest drop > 3 dB, plus the three
provenance flags above. Confidence: high.

**A8. `--target-rms` help text states the electronic default under every profile.** `cli.py:481-483`
("default -4.1") while `Controls.rms_target` (`planner.py:625-627`) returns -10 under live. S5-level, listed
here because the proposal changes the number. Fix: "default -4.1 (electronic) / <live> (live)".

### S3 - crash / misbehaviour

**A9. Live-length material.** No crash in 8 runs, but peak memory was 1.17-1.35 GB and analysis alone 150-181 s
for 36-44 s of 48 kHz audio (12 threads; whole master 153-846 s). Live recordings are concerts: at this scaling a
60-minute set needs tens of GB and the memory guard (`memory.guard`) refuses or drops to one thread, and the
resonance detector (3 resolutions x 3 channels) and structure pass dominate the time. Not fixable inside
PROFILES; noted because "right for live material" includes being able to master it. Confidence: medium (scaling
extrapolated from 44 s files; release notes give 3 GB / 6 min at 44.1 kHz).

### S4 - hygiene

**A10. The 60 Hz side strip is unconditional under live.** `planner.subside_hz 779-792`: the profile value
precedes the sub-decoherence rule, so a stereo-miked or panned bass loses its image below 60 Hz whatever the
sub's correlation. Evidence (`classical_strings`, bass panned right; `sidestrip.apply` at 60 Hz): L/R level
difference below 60 Hz -6.8 -> -3.6 dB while 60-150 Hz stays -5.0 dB (a frequency-dependent image), band-0 L/R
correlation 0.77 -> 0.92, S below 60 Hz -41.5 -> -47.2 dBFS; the change signal is -22 dB re the full signal.
Audibility is low (LF localisation is weak) but the LF decorrelation of a spaced pair is part of a classical
recording's spaciousness. Fix: live `sidestrip_hz: None` with a rule: 60 Hz only when `band_decoherence[0]`
>= 0.3 or band-0 correlation < 0.5 (all four fixtures: 0.02-0.23 -> off). Confidence: medium.

**A11. Live budgets are read on a grid that is +-40 ms off on swing.** Not wrong, but the maintainer should
know the reading's sensitivity: jazz band 0 M reads 1.00 on the analysis grid, 0.77 on the true beats, 0.59 on
400 ms frames, 1.23 on a half-tempo grid (`envelope.stage_gr`, `LOUD_MIN_HITS 16`). Document as a tolerance
(+-0.3 dB) on the 1.0 dB live budget.

**A12. Pre-ring.** Output energy in the 20 ms before onsets, relative to the hit, minus the same on the input:
live median +0.09 / +0.65 / +0.50 dB (jazz / folk / rock), p90 1.1 / 2.7 / 1.6 dB, worst strum +5.1 dB (6 of 77
folk onsets over +3 dB); electronic +1.3 / +2.4 / +1.3 dB median, 17-30 onsets over +3 dB, worst +7.6 dB. The
live profile's causal HPF, 5 ms "transparent" limiter lookahead and 20 ms glue attack keep it near inaudible;
the remaining energy is the zero-phase crossovers at 65-104 Hz. Acceptable; S4 only.

### S5 - nits

**A13.** ISO 226 weighting is fixed at 80 phon (`planner.py:325`). At the live level (6 dB lower -> about 74 phon
at the same monitor gain) the weights differ by <= 0.06 at 100-200 Hz and <= 0.03 above 1 kHz (31.5-63 Hz are
pinned by the 0.05 floor either way), so it behaves. Nit: derive phon from the target (80 + target - (-4.1)).
**A14.** The classical run's character reason quotes a tempo that does not exist (A6). **A15.** README line 70
and release notes say "uncalibrated"; evaluate.py:932, processlog.py:161, cli.py:489, tests/test_profiles.py
lines 27, 34-36, 43, 46 and `assert not PROFILES["live"]["calibrated"]` pin the current numbers and must move
with the proposal.

---

## B. Calibration proposal (headline)

### B.1 What the current live profile does on the fixtures (T1)

| run | wall s | renders | limiter gain dB | meter p95 dBFS | integrated LUFS | S-max LUFS | LRA LU | TP dBTP | PLR dB | PSR p95 dB | crest dB | target / achieved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jazz_trio-live | 524 | 1 | 7.23 | -14.36 -> -10.01 | -18.0 -> -12.5 | -14.2 -> -10.1 | 7.4 -> 5.0 | -2.0 -> -0.3 | 16.0 -> 12.2 | 14.6 -> 12.9 | 15.7 -> 13.1 | -10.0 / -10.01 |
| jazz_trio-electronic | 846 | 1 | 11.60 | -14.36 -> -6.37 | -18.0 -> -7.9 | -14.2 -> -6.7 | 7.4 -> 2.4 | -2.0 -> -0.3 | 16.0 -> 7.6 | 14.6 -> 9.1 | 15.7 -> 9.4 | -4.1 / -6.37 (clamped) |
| folk_gtr_vocal-live | 517 | 1 | 6.18 | -15.99 -> -10.25 | -19.2 -> -13.1 | -16.4 -> -11.5 | 5.7 -> 3.5 | -2.0 -> -0.3 | 17.2 -> 12.8 | 17.2 -> 16.1 | 17.5 -> 13.5 | -10.0 / -10.25 (limiter avg clamp) |
| folk_gtr_vocal-electronic | 460 | 1 | 10.82 | -15.99 -> -6.13 | -19.2 -> -9.1 | -16.4 -> -7.2 | 5.7 -> 3.6 | -2.0 -> -0.3 | 17.2 -> 8.8 | 17.2 -> 12.4 | 17.5 -> 9.5 | -4.1 / -6.13 (clamped) |
| rock_band-live | 252 | 1 | 6.15 | -11.86 -> -10.00 | -15.8 -> -13.6 | -13.5 -> -11.4 | 12.0 -> 9.6 | -1.4 -> -0.6 | 14.4 -> 13.0 | 15.8 -> 14.1 | 14.2 -> 12.9 | -10.0 / -10.00 (clipper on, A1) |
| rock_band-electronic | 344 | 2 | 11.04 | -11.86 -> -7.14 | -15.8 -> -9.6 | -13.5 -> -7.8 | 12.0 -> 6.5 | -1.4 -> -0.3 | 14.4 -> 9.3 | 15.8 -> 12.5 | 14.2 -> 10.0 | -4.1 / -7.14 (clamped) |
| classical_strings-live | 153 | 1 | 6.84 | -12.44 -> -10.00 | -18.3 -> -15.3 | -13.0 -> -10.8 | 21.0 -> 18.9 | -3.0 -> -1.8 | 15.3 -> 13.4 | 12.5 -> 12.4 | 12.5 -> 11.4 | -10.0 / -10.00 |
| classical_strings-electronic | 231 | 2 | 12.02 | -12.44 -> -6.42 | -18.3 -> -10.1 | -13.0 -> -7.1 | 21.0 -> 16.0 | -3.0 -> -0.3 | 15.3 -> 9.8 | 12.5 -> 12.3 | 12.5 -> 9.3 | -4.1 / -6.42 (clamped) |

Budgets as reported (T3, live): multiband comp 1.0 dB average on every non-gated band (outliers 1.3-2.2),
band limiter 0, glue net 1.0 (outlier 1.45-1.93), limiter average **0.0** on jazz / rock / classical and 1.5
(clamped, 0.3 dB short) on folk, limiter outlier 0-2.9 (cap 3.0), speclim only band 0: jazz M 2.3 / S 1.0 dB,
rock S 0.6 dB, nothing on cymbals. Note what this means: at -10 the loudness is reached almost entirely by gain
plus compressor makeup, glue and speclim; the limiter is idle while the chain still takes 1.1-4.0 dB of crest.

### B.2 Meter conversion measured on the fixtures (T2)

| material | meter p95 - integrated, premaster | meter p95 - integrated, live master | meter p95 - S-max (pre / live) | LRA |
|---|---|---|---|---|
| jazz_trio | 3.59 | 2.52 | -0.17 / 0.12 | 7.4 -> 5.0 |
| folk_gtr_vocal | 3.19 | 2.87 | 0.41 / 1.20 | 5.7 -> 3.5 |
| rock_band | 3.92 | 3.59 | 1.65 / 1.37 | 12.0 -> 9.6 |
| classical_strings | 5.86 | 5.25 | 0.52 / 0.80 | 21.0 -> 18.9 |

Rule of thumb for this meter (300 ms AES17 RMS, p95 over the track): **meter = S-max + 0.8 +- 0.6 dB**, and
**meter = integrated + 3.0 +- 0.6 dB for LRA 5-12 material, + 5.3..5.9 dB for LRA ~ 20**. (A stereo full-scale
sine reads meter 0.0 / -0.7 LUFS; music's K-weighting and the p95 window account for the rest.) So the current
**-10** target is S-max ~ -11 LUFS and integrated ~ -13 (band / jazz / folk) or ~ -15 (wide classical).

### B.3 What modern practice does (published)

Platforms and recommendations:
- Spotify normalises to -14 LUFS integrated (Loud -11 / Quiet -19 options on some regional pages), asks for
  <= -1 dBTP (<= -2 dBTP for masters louder than -14) and since 2021 no longer limits quiet tracks upward
  ([Spotify for Artists](https://support.spotify.com/artists/article/loudness-normalization/),
  [MeterPlugs 2021](https://www.meterplugs.com/blog/2021/07/15/spotify-disables-their-limiter.html)).
  YouTube, Tidal, Amazon: -14 (third-party reporting; no first-party figure), Apple Sound Check ~ -16
  ([iZotope guide](https://www.izotope.com/en/learn/mastering-for-streaming-platforms), [Orphiq 2026 review](https://orphiq.com/resources/mastering-for-streaming)).
- AES TD1008 (2021): distribution loudness -16 LUFS for track-normalised music, album normalisation with the
  loudest track at -14, speech -18; "fine arts programming may need lower integrated loudness"; upward
  normalisation is a problem for high-PLR content ([Radio World](https://www.radioworld.com/?p=115138),
  [Production Advice](https://productionadvice.co.uk/td1008/), [Grimm Audio](https://www.grimmaudio.com/?p=4698);
  Grimm's 4.2 M-album Tidal study: 87 % of albums have a loudest track above -14, [HKU/Grimm](https://www.grimmaudio.com/?p=2874)).
- EBU R128: -23 LUFS, max -1 dBTP (-2 for lossy chains); R128 s1 short-form: max short-term -18 LUFS; LRA is a
  production tool with no maximum ([EBU Tech 3343 v2023](https://tech.ebu.ch/docs/tech/tech3343.pdf) sections 9.3.2
  and 11.2; s1 [tech.ebu.ch/publications/r128s1](https://tech.ebu.ch/publications/r128s1)). Tech 3343 9.3.2 notes
  music programmes (wide-LRA classical, compressed rock) are listened to +2-3 LU louder than speech-anchored
  ones and recommends a "Loudness Offset / Low Loudness Flag" for deliberately quiet classical programmes
  rather than a different target.

Measured masters:
- 246,705 tagged tracks (2026 scan, median integrated LUFS, IQR): metal -8.4, rock -9.2 (-11.0..-7.8),
  country/folk -9.6, singer-songwriter -10.6 (-12.4..-9.2), jazz -12.8 (-15.2..-10.8), instrumental/score -12.9
  (-16.5..-10.1), ambient -14.1; 90 % of all tracks louder than -14 ([Freshly Baked Studios](https://freshlybakedstudios.com/blog/average-lufs-by-genre); no classical row).
- Deruty & Pachet 2015, 7200 mainstream tracks 1967-2014 ([ISMIR paper](https://archives.ismir.net/ismir2015/paper/000136.pdf), Fig. 1):
  EBU 3341 integrated median ~ -17 LUFS before 1990 rising to ~ -11..-12 at the 2007 peak; crest factor
  ~ 12.5-13 dB before 1990, ~ 10 dB at the 2008 peak; EBU 3342 LRA ~ 5-6 LU throughout (macrodynamics untouched);
  dynamics depend on year, not genre. Deruty 2011 (SoS, 4,500 tracks): ~5 dB louder than the 70s, crest -3 dB
  ([SoS](https://www.soundonsound.com/sound-advice/dynamic-range-loudness-war)). Deruty & Tardieu 2014 JAES 62(1/2)
  42-55 is the source of the descriptors (paywalled; abstract only).
- Kirchberger & Russo 2016 (Trends in Hearing, 1000 songs, 10 genres): dynamic range (99th-30th percentile
  per third-octave) speech > classical (orchestra, opera, chamber, piano, choir) > jazz > pop/rock/rap/schlager
  ([PMC4753356](https://pmc.ncbi.nlm.nih.gov/articles/PMC4753356/); values are in figures only).
- Classical albums measured by users: Bach violin sonata -19.1 LUFS / LRA 8.9; Mahler 1 -16.9 / 23.5; live
  Mahler 1 -20.9 / 23.1 ([Steinberg forum](https://forums.steinberg.net/t/how-to-use-loudness-meter/623612));
  vendor guidance -18..-23 ([Waves](https://www.waves.com/how-many-lufs-should-my-master-be)).
- Micro-dynamics guidance: Dynameter minimum-PSR presets Limited 8 / Competitive 10 / Balanced 12 / Wide 14;
  unprocessed acoustic drums read PSR > 12; a classic loud rock master "minimum PSR 9, PLR 11 = healthy";
  loudness-war victims PSR 6 / PLR 8 ([Dynameter manual](https://www.meterplugs.com/files/dynameter-guide.pdf) pp. 9-13);
  Katz: PLR 12 good, 14 rare, 20 the limit ([Lentz summary](https://world.hey.com/larslentz/9-20-23-plr-and-microdynamics-psr-1cffb024)).
- The DR database gives no genre statistics (its FAQ: rock/metal DR >= 8 acceptable, electronic DR 5 can be fine;
  [dr.loudness-war.info](https://dr.loudness-war.info/faq)); EBU Tech 3342 only gives test-signal LRAs (5 and 15 LU).

Reading of this for the engine: (1) every platform normalises at -14 (Apple -16); a live master above -14
integrated is turned down and buys nothing but lost crest; (2) the measured market for acoustic genres sits at
-12.8 (jazz) to -10.6 (singer-songwriter) integrated, i.e. the current -10 (integrated ~ -12.5..-13.6) is at the
jazz median and 2 dB hotter than TD1008's track value; (3) at the loudest passage the agreed floor is PSR 8,
the comfortable zone PSR 10-12.

### B.4 Recommendation for this engine

**Target.** `PROFILES["live"]["target_rms_dbfs"] = -11.5` (meter p95). By B.2 that is S-max ~ -12.3 LUFS and
integrated ~ -14.5 +- 0.6 for LRA 5-12 material (rock band, jazz, folk) and ~ -17 for LRA ~ 20 (classical),
which lands exactly at the Spotify / YouTube / Tidal normalisation point for band material and at the TD1008
"fine arts lower" / measured-classical range for wide material, with a loud-passage PSR of ~ 12 dB (TP -0.3 minus
S-max; Dynameter "Balanced") and PLR 14-16. The 1.5 dB difference from -10 is the crest the fixtures lost for
nothing: at -10 the live chain cost 2.6 (jazz) / 4.0 (folk) / 1.3 (rock) / 1.1 (classical) dB of crest and
3.8 / 4.4 / 1.4 / 1.9 dB of PLR, half of it to reach a level the platforms immediately undo. Sanity ranges to
print in the report: S-max -13.5..-11.0 LUFS; integrated -16.5..-13.5 (LRA <= 12) or -18.5..-15.0 (LRA > 15);
PSR p95 >= 10. Alternatives, documented: -10 (current) = "2026 jazz median, louder than every normalisation
point"; -13 = TD1008 track normalisation (-16 integrated), the purist choice, at which most live premasters
need almost no dynamics at all.

**One profile, one measured rule (no presets).** The loud-passage target already scales with macro-dynamics by
itself (fixed meter target -> integrated -13.6 for the band, -15.3 for the strings at -10), so the target needs
no class rule. What differs between a band / trio and sustained material (strings, organ, choir) is the
processing, not the level. Rule: `sustained = transients.onsets_per_s < 0.5` (fixtures: classical 0.09/s; jazz
3.3, folk 1.9, rock 3.8). When sustained: multiband `comp_gr_db 0.5, comp_gr_max_db 0.75, total_normal_db 0.75,
limit_max_db 1.0`, `glue_net_db 0.5`, `limiter_max_gr_db 1.0, limiter_outlier_cap_db 2.0`, `speclim_allow_db None`
(off), notches off (A2), hits = 400 ms frames with the flag of A6. Otherwise the table below. This is one
boolean derived from a feature the analysis already computes, not a user-facing preset.

**Exact constants** (`src/eluate/planner.py` PROFILES["live"], lines 498-524):

| key | now | proposed | reason / evidence |
|---|---|---|---|
| `calibrated` | False | True once the maintainer has run `meter.rms_meter` over his own live masters (same method as the electronic -4.1 median); until then keep False but change the report text to "calibrated on synthetic fixtures and published statistics" | A15 |
| `target_rms_dbfs` | -10.0 | **-11.5** | B.2-B.4; integrated lands at -14/-16, PSR ~ 12 |
| `mb_budget` | comp 1.0 / max 1.5 / total 1.5 / limit 1.5 / ratio_max 1.5 | keep; x0.5 when sustained | measured avg exactly 1.0, outliers <= 2.2; crest loss 0.1-1.0 dB/band (T6); pumping <= 0.42 dB |
| `ms_link` | True | keep True, but solve the linked bands on the linked detector (or report the applied GR) | A4 |
| `side_makeup_db` | 0.0 | keep | correlation unchanged (T5) |
| `saturate` | False | keep | - |
| `glue_net_db` / `glue_attack_ms` | 1.0 / 20 | keep; 0.5 when sustained | glue lock-in <= 0.18 dB; outliers <= 1.9 |
| `clipper_presets` | ("fast","aggressive") | **()** (or explicit-character gate) | A1: auto "fast" at 140.1 BPM clipped the rock band |
| `kick` | False | keep | 0-2 false kicks found on feathered kick / upright bass; no kick mask |
| `speclim_allow_db` / `speclim_max_cut_db` | 4.0 / 3.0 | keep; None when sustained | only band 0 fired (2.3 dB jazz M, 0.6 dB rock S), no cymbal cuts |
| `limiter_max_gr_db` / `limiter_outlier_cap_db` | 1.5 / 3.0 | keep (1.0 / 2.0 when sustained) | measured avg 0-1.5, outlier 0-2.9; at -11.5 it rarely binds |
| `limiter_timing` | "transparent" | keep | pre-ring median <= 0.65 dB (A12) |
| `sidestrip_hz` | 60.0 | **None** + rule: 60 Hz only when `band_decoherence[0] >= 0.3` | A10 |
| `eq_bounds` | gain 1.5 / hs 2.0 / 2.0 | **gain 1.0 / hs 1.0 / 1.0**, and at Tilt 0 no presence / air / low shelves (HPF + narrow bell only) | A3: +-1.5 dB broadband rotation at Tilt 0 |
| `eq_weighting` | 80 phon | keep (nit: 80 + target + 4.1) | A13 |
| `eq_residual_window` / `eq_hpf` | True / 25 Hz Butterworth | keep | verified (sub shelf unchanged within 0.5 dB on 3 of 4) |
| `slope` | "report" | keep, plus the informational check of A5 (|measured - (-4.5)| > 2 dB/oct) | A5 |
| new: `notch_flags` | (all) | ("resonance",) under live | A2 |
| new: `sustained_onsets_per_s` | - | 0.5 | B.4 rule |
| new flags (evaluate) | - | PSR p95 < 10 / < 8, PLR < 12, LRA drop > 3 LU when LRA > 12, crest drop > 3 dB, "clipper on under live", "N notches on tonal material", "no beat grid" | A7 |

Tests that pin the old numbers and must move: `tests/test_profiles.py` lines 27 (`rms_target == -10.0`),
34-36 (limiter 1.5/3.0, speclim 4.0/3.0), 43 (sidestrip 60), 46 (mb_budget), 60-62 (eq bounds),
`test_live_clipper_only_under_fast_and_aggressive` (becomes "never under auto"), and the
`calibrated` asserts (lines 19, 68, 71). Current fast tier: 12 passed (195 s).

### B.5 Remaining evidence tables

T4 listen-proxies (live rows; electronic for contrast):

| run | mb band0/2/4 M lock-in at beat rate dB | glue mean / lock-in | limiter mean / p95 / lock-in | S extra from ms_link mean,max (band0 / band2) | pre-ring delta median / p90 / max dB (n > 3 dB / n) | rendered notches | speclim band0 M/S |
|---|---|---|---|---|---|---|---|
| jazz_trio-live | 0.08 / 0.19 / 0.11 | 0.28 / 0.16 | 0.04 / 0.20 / 0.05 | 0.07,2.53 / 0.04,1.93 | 0.09 / 1.05 / 2.59 (0 / 138) | 0 | 2.3 / 1.0 |
| jazz_trio-electronic | 0.26 / 0.37 / 0.21 | 0.65 / 0.16 | 1.11 / 3.58 / 0.32 | - | 1.32 / 3.23 / 6.40 (17 / 138) | 0 | - |
| folk_gtr_vocal-live | 0.00 / 0.09 / 0.25 | 0.33 / 0.18 | 0.21 / 1.15 / 0.25 | 0.00,0.00 / 0.15,1.65 | 0.65 / 2.72 / 5.13 (6 / 77) | 0 | - |
| folk_gtr_vocal-electronic | 0.00 / 0.20 / 0.29 | 0.55 / 0.19 | 0.74 / 3.16 / 0.28 | - | 2.44 / 4.45 / 6.93 (30 / 77) | 0 | - |
| rock_band-live | 0.23 / 0.07 / 0.42 | 0.49 / 0.18 | 0.05 / 0.35 / 0.10 | 0.38,2.45 / 0.14,2.57 | 0.50 / 1.55 / 3.82 (1 / 138) | 0 | 0 / 0.6 |
| rock_band-electronic | 0.28 / 0.13 / 0.68 | 0.83 / 0.15 | 1.13 / 3.20 / 0.34 | - | 1.27 / 3.59 / 7.58 (21 / 138) | 0 | - |
| classical_strings-live | - (no beat) | 0.18 | 0.00 | 0.03,1.10 / 0.03,2.11 | 0.09 / 0.26 / 0.27 (0 / 4) | **9** (1972-8875 Hz, 0.5-1.5 dB) | - |
| classical_strings-electronic | - | 0.27 | 0.54 / 2.85 | - | 0.27 / 0.95 / 1.16 (0 / 4) | 9 | - |

T5 grid / tilt / stereo / sub image (live rows):

| run | analysis BPM (true) | grid informative | grid median error ms / within 50 ms | slope before -> after | band L/R corr before -> after (0..4) | L/R diff below 60 Hz before -> after dB | flags |
|---|---|---|---|---|---|---|---|
| jazz_trio-live | 118.6 (120) | True | 42 / 57 % | -4.70 -> -4.85 | 0.96->0.98, 0.97, 0.97, 0.94->0.95, 0.96 | 0.14 -> -0.01 | none |
| folk_gtr_vocal-live | 75.2 (76, rit.) | True | 35 / 86 % | -2.86 -> -2.96 | unchanged (0.96-0.98) | 4.68 -> 3.32 | none |
| rock_band-live | 140.1 (140) | True | 31 / 93 % | -4.83 -> -4.95 | 0.98->0.99, 0.68, 0.56->0.55, 0.57->0.51, 0.80->0.72 | -0.11 -> -0.08 | none |
| classical_strings-live | 105.4 (rubato) | **True (false)** | 825 / 0 % | -6.63 -> -6.95 | 0.77->0.90, 0.59->0.62, 0.67, 0.71, 0.73 | -6.81 -> -3.40 | none |

T6 per-band crest p95 loss (stereo view, dB), live: jazz 0.1 / 0.4 / 0.0 / 1.0 / 0.6; folk -0.7 / 0.1 / 0.7 / 0.9 /
0.8; rock -0.3 / 0.3 / 0.5 / 0.6 / 0.7; classical 0.6 / 0.2 / 0.0 / 0.1 / 0.2 (electronic up to 2.4 / 2.2 / 1.9).

---

## C. Verified correct

- Chain latency 0 samples on all 8 runs; true peak -0.30 dBTP held on 8x and 32x (classical-live ends at -1.82
  because -10 was reached without touching the ceiling).
- No pumping under live: GR lock-in at the beat rate <= 0.42 dB on every multiband band, <= 0.18 dB glue,
  <= 0.25 dB limiter (electronic: up to 0.68 / 0.83 / 0.34).
- Budgets honoured and reported as solved: multiband comp average 1.00 +- 0.01 dB on every non-gated band,
  outliers <= 2.2 < 4.5; glue net 1.00, outliers <= 1.93 < 4.0; limiter outlier <= 2.92 <= 3.0 cap; the one
  clamp (folk) is warned correctly ("needed 1.8, cap 1.5, 0.3 dB short, within tolerance").
- Kick detective off and no kick mask under live; the detector itself finds only 0-2 "kicks" on the feathered
  jazz kick, upright bass and rock bass (`KICK_MIN_RATE_HZ` keeps them unusable), so "a floor tom or a double
  bass is not a kick" holds on these signals.
- Resonance "tonal" flag suppresses held notes on transient material (75/78, 89/91, 62/65 candidates); the
  failure is confined to vibrato ensembles (A2).
- Speclim under live only acted on band 0 (the accented kick: 2.3 dB M / 1.0 dB S jazz, 0.6 dB S rock); no cuts
  on ride, hats or crashes (top-band ceilings None on all runs).
- Saturation off, side makeup 0, 25 Hz Butterworth HPF causal (pre-ring median <= 0.65 dB), S high-pass does
  not move the centred jazz / rock bass (L/R diff below 60 Hz changes <= 0.15 dB).
- Section contrast: live lost 0.0-1.7 LU of section contrast (flag 3 LU); electronic 4.0 LU on the strings (flagged).
- ISO 226 weighting is insensitive to the 6 dB level difference (A13).
- Tempo grid on swing with jitter and drift (118.6 vs 120) and on a ritardando (75.2 vs 76, "tracked") is
  sane; the GR reading moves <= 0.23 dB between the analysis grid and the true beats.
- `pytest tests/test_profiles.py -m "not slow"`: 12 passed.
- The electronic profile on the same material is unambiguously wrong for it (crest 9.3-10.0, PLR 7.6-9.8,
  LRA -5 LU, air shelves +4.5..+6 dB, limiter clamped 2-3 dB short): the live profile's direction is right.

## D. Open questions for the maintainer

1. -11.5 (at the normalisation point, PSR ~ 12) or the purist -13 (TD1008 track -16)? Both are defensible; -10 is not.
2. The -11.5 is calibrated on synthetic fixtures plus published statistics. The electronic -4.1 came from your 18
   masters on `meter.rms_meter`; the same one-liner over your live/acoustic masters (if any) would replace B.4's
   number with a measured median - do you have such a set?
3. "Sustained" as an automatic rule (`onsets_per_s < 0.5`) or a visible state in the UI ("no beat: sustained
   material", read-only)? I recommend automatic plus the read-only label.
4. Should the live loudness come from the limiter (peaks) or from compressor makeup (density)? Today at -10 the
   limiter averages 0 dB and the compressor + glue + speclim supply the level; the budgets say "ok" while 2.6-4
   dB of crest went. If you want live loudness to be "gain until the peaks say stop", the live multiband makeup
   should be 0 (as your Ozone settings had) and the limiter budget carries the lift.
5. At Tilt 0, should live be an identity EQ apart from the HPF (my recommendation) or keep correcting narrow
   build-ups? Where is the line between "a resonance" and "the bass note the song is in"?
6. Long live recordings (60-120 min) will hit the memory guard; is a blocked / sectioned analysis in scope for
   the live profile, or is "export the concert in pieces" the answer?
7. PSR floor: a report flag (my recommendation) or a hard clamp on the limiter gain (PSR p95 >= 10)?
8. The `ms_link` fix: solve on the linked detector (one threshold, changes audio slightly) or just report the
   applied GR (no audio change)?
