# Eluate audit: planner / solver / plan / app flow (2026-11-10)

Scope: `src/eluate/planner.py`, `solver.py`, `plan.py`, `estimate.py`, `memory.py`,
`parallel.py`, `app_engine.py`, `cli.py` and the matching tests. Branch `audit/2026-11-10`,
venv numpy 2.5.4 / scipy 1.18.1 / Python 3.12.13, Windows 11, 12 logical cores (the
experiments ran four scripts concurrently, so wall-clock figures below are contended).
Synthetic signals only (`fixtures/synthetic.sectioned_track` tiled, made truly stereo with
panned pads, decorrelated hats and -42 dBFS pink noise; impulse trains; silence; DC).

Scripts and logs (the audit's session scratch, not in the repository): `scratchpad/audit/planner-solver/` (`common.py`, `mono_sweep.py`,
`edge_cases.py`, `cli_master.py`, `misc.py`, `followup1..3.py`, `paths.py`, `*.log`,
work dirs `out_ref`, `out_low`, `out_hot`, `out_ref_preserve`, `out_silence`, `out_half`).

Fast test tier for the in-scope files: **297 passed, 43 deselected (slow)** in 16 min
(`pytest_fast.log`).

Severity: S1 audible / wrong audio; S2 wrong numbers in plan or report; S3 crash or
misbehaviour on plausible input; S4 hygiene / perf / documentation that misleads; S5 nit.

---

## Findings

### S3

**1. Any input shorter than 1 s, or whose gated 1 s RMS is empty (digital silence, a
near-silent file), runs the full analysis and then crashes the planner with
"input measured_dbfs contains NaN or Inf".**
`planner.py:706-711` (`input_gain` returns `(0.0, nan)` when `meter.rms_1s` has no
gated window), `planner.py:771-772` (`analyse` stores `"measured_dbfs": float(m)`),
`plan.py:336-343` (`InputPlan.__post_init__` rejects a non-finite record field),
`planner.py:2249` (`InputPlan(**inp)`).
Evidence: `followup1.log` A (silence 8 s, 0.3 s and 0.99 s loops: `measured=nan`,
`InputPlan RAISES ValueError`); `eluate master wav/silence.wav` and `wav/half.wav`:
"analysed silence.wav in 26.0 s ... eluate: input measured_dbfs contains NaN or Inf"
(`tasks/be3lxrqii.output`); `edge_cases.log` 0.3 s @ 44.1 k and @ 48 k. The work dir
is left with features.npz / analysis.json / source.json and no plan.
Fix: `"measured_dbfs": float(m) if math.isfinite(m) else None` (InputPlan already allows
None), and refuse inputs shorter than `meter.RMS_WINDOW_S` up front in `cli.do_master` /
`app_engine` with a plain message before the 25 s analysis. Confidence: high.

**2. `plan_from_analysis` crashes at any sample rate at or below 22 222 Hz whenever the
analysis carries a side spectrum (every modern analysis does): scipy
"Each lower bound must be strictly less than each upper bound".**
`planner.py:1314-1316` (`_corner_range` collapses the air-shelf corner range to
`(0.45 fs, 0.45 fs)` once `0.45 fs <= 10 000`), `planner.py:1549-1550`
(`_fit_side_shelves` builds `lo`/`hi` from it with no guard), while the mid fit has the
guard at `planner.py:1391-1393`.
Evidence: `misc.log` section 4 (`analyse/plan @ 22050` and `@ 16000` raise, traceback
through `plan_inputeq`); `followup3`/`followup1.log` B: `side=False: ok`,
`side=True: RAISES` at fs 22050 and 22222; ok from 22223 up. A 22.05 kHz premaster is
unusual but accepted everywhere else in the chain (`bands.split` is fine down to 16 kHz;
plan validation checks every frequency < fs/2).
Fix: apply the same `if hi[a] <= lo[a]: hi[a] = lo[a] + 1e-6` loop in
`_fit_side_shelves`, or make `_corner_range` return `(top / 2**(1/3), top)` when it
collapses. Confidence: high.

**2b. The loud-hit fallback ("fewer than 16 qualify: the 16 loudest") ignores the
-70 dBFS gate, so a short track whose quiet hits are numerically non-silent (HPF residue
of a DC offset, dither, room tone) budgets every stage on near-silent hits.**
`envelope.py:420-423` (`loud_hits`: `mask[np.argsort(-lv)[:16]] = True` with no gate),
`multiband.py:444-453` (`_loud_top_db` brackets from the lowest loud hit),
`solver.py:442-457` (`limiter_peaks`: the median of the loud hits' peaks).
Evidence: `edge_cases.log` "DC 0.5 8s": the zero-phase 25 Hz HPF leaves 1.2e-27
(-540 dBFS) everywhere but the edges (reproduced: HPF test B), 16 of 20 hits are taken
as loud, the multiband thresholds come out at **-545.8 / -787.7 / -841.6 dBFS** with
warnings "needed 53 / 96 / 170 dB, cap 6", the glue threshold at -82.6 dBFS (net 0.07,
outlier 7.5), the clipper threshold at **-731.8 dBFS**, `clipped_average_peak_dbfs
-542.8`, requested gain 36 dB (the drive-range end), `needed_db -506.5`; reproduced on a
minimal signal (2 loud hits + 18 hits of 1e-27: `loud count 16`, median -540 dBFS).
Exact digital silence is harmless (the HPF keeps exact zeros, hit peaks read NaN and
are skipped), so the realistic trigger is a short clip (fewer than 16 hits within 3 dB
of the p95) with a tail or lead of near-silence: the limiter's average term then reads
tens of dB too low, the gain clamps on the outlier term alone (5 dB on the peak instead
of 3 dB on the typical hit) and a multiband average with half its loud set silent is
unreachable (finding 3).
Fix: in `loud_hits` take the 16 loudest among hits above `RMS_GATE_DBFS` (all hits only
when none pass); treat hit values below about -120 dBFS as NaN in `hit_peaks_db` /
`_loud_top_db`. Confidence: high (mechanism), medium (how often real clips trigger it).

### S2

**3. A band whose meter-smoothed average GR cannot reach its 2.5 dB target is driven
48 dB down the bracket into its noise / leakage floor, then reported "ok" (or "over")
with no warning.** `multiband.py:488-493` (step-down loop to `floor = lo -
BRACKET_MAX_DB`), `multiband.py:422-441` (`bisect_peak_gr` returns the closest
evaluation, i.e. the floor, when `f(lo) < target`), `multiband.py:694-698` (the
`kind: "average"` warning fires only at a 0.5 dB shortfall), `planner.py:2113-2123`
(`_status` reads "ok" unless 0.5 dB short). `glue.py:240-243` shares the pattern
(60 dB bracket).
Evidence: `followup1.log` C on a 110 Hz pad plus 10 ms impulses every 0.5 s: band 2 M
(400-1600 Hz) — the sweep's average saturates at 2.2-3.2 dB because a 100 ms release
blob averaged over 300 ms cannot read 2.5 dB; the solver returns threshold **-70.1 dBFS**
(top of the band -8 dBFS), per-sample GR 6.0 dB (the clamp) with GR > 0.01 dB for
**100 % of the time**; `design_budget` lands at -75.9 / -76.1 / -75.6 dBFS on bands 2-4
with comp_db 2.27 / 2.18 / 2.17, outlier 5.3 (> 4.5 cap, status "over"), PEAK makeup
+2.2 dB, `warnings: []`. `mono_sweep.log` "sparse" section shows the same for knee 1
(thresholds -95 to -97 dBFS). On a real track with a sparse top band (hats only, nothing
else above 5.6 kHz) this turns the band down by several dB continuously and slams every
hit at the 6 dB clamp — potentially audible, not demonstrated on music.
Fix: bound the search by the band's program level (never below the band channel's
loud-section 1 s RMS minus ~20 dB), and when the floor binds record `bound: "bracket
floor"` and emit the average warning regardless of the 0.5 dB rule; `budget_report`
should print it. Confidence: high (mechanism), medium (audibility).

**4. The clipper's "rendered" output-peak fall in plan_record.json / solve.json /
report.json is wrong whenever `clip_split_hz` is set (the planner's default).**
`planner.py:2182-2186`: `rendered_db = input_peak_8x_dbfs + input_gain_db -
detector_peak_dbfs`, but `input_peak_8x_dbfs` is the peak of the part *above the split*
(`solver.py:726-727`) while `detector_peak_dbfs` is the full signal's.
Evidence: `followup2.log`: `planned_db 0.996, rendered_db -0.59` (= -9.59 + 13.31 - 4.31).
Not printed in report.md (`evaluate._budget_md` prints planned/outlier only) but present
in all three JSON files. Fix: skip `rendered_db` when `clip_split_hz` is set, or record the
full-signal 8x input peak alongside and use it. Confidence: high.

**4b. A wrong `--bpm` silently redefines every budget's hits, and an octave error turns
the frequency-dependent limiter onto the kick band at its 6 dB maximum.** The user's
tempo seeds the grid with no second-guessing (`structure.analyse`, `fixed=True`;
`planner.plan_hits` "user bpm"); `multiband.design_speclim` (`multiband.py:814-824`)
takes the typical loud hit P50 as the median of per-hit peaks, so half-beat hits (every
other hit has no kick) halve P50 in the bands where only the kick lives.
Evidence (`edge_cases.log`, 8 s 128 BPM loop): detected grid -> speclim cut in band 1
(87-349 Hz, the kick body) 0.0 dB; `--bpm 256` -> **6.0 dB** (the maximum) in band 1 M,
3.3 / 1.7 / 2.6 dB in bands 2-4, 34 hits with 18 loud; `--bpm 97` -> 13 hits, every
budget re-solved on 0.62 s hits; `--bpm 64` -> character resolves to slow, the kick
detective replaces the whole-signal clipper (chain topology changes). In no case does
any output mention the detected tempo or the disagreement (finding 7). The analysis
records `detected_relation` "x2" for the 256 case and nobody reads it.
Fix: when `detected_relation` is "x2" / "/2" (or `deviation_pct` is large) warn in the
report and the process log; consider refusing an octave-doubled user tempo for the
speclim P50 (or taking P50 over hits that contain a kick). Confidence: high.

### S4

**5. Level invariance is weaker than documented and tested.** docs/gain-staging.md §11
and `tests/test_gain_staging.py:244-271` claim "the same master to one 24-bit step". On
real files (32-bit float WAVs at -6, -26 and 0 dBFS peak, i.e. independent float32
roundings of 6e-8 relative) the masters differ by **658 24-bit steps (-82 dBFS), on 99 %
of samples, RMS difference -104 dB** (`cli_master.log` "invariance"). The first
divergence is the input EQ's rounded air-shelf corner: **11370 Hz (ref) vs 11380 Hz (low,
hot)** — `planner.py:1319-1322` (`_top_fc` rounds to 10 Hz) and `1437-1442`; the flip
then moves the crest-derived ratios by 1e-5, thresholds by up to 1.3e-3 dB
(`multiband.gr_db.3.0` max diff 2.9e-4 dB), the glue threshold by 2.7e-4 dB and the
limiter gain by 1.4e-4 dB. The bisection itself is stable (a 1e-7 perturbation moves a
threshold by 5e-7 dB, `followup3.log`). The test passes only because its fixture scales a
float64 array exactly: the same loop scaled in memory to -40 dBFS peak gives the
identical plan (clipper threshold -13.881443744592 in both, `edge_cases.log` "peak -40
dBFS") and the same master RMS, so the engine's arithmetic is level-invariant and the
divergence is entirely the EQ fit's rounding of independently rounded float32 files.
Not audible; the claim and the test are overstated.
Fix: state the real tolerance (about 1e-3 dB, -80 dB) or quantise the fit inputs
(residuals / LTAS nodes to 0.01 dB) so a rounding flip needs a spectral change; add a
test that writes the two levels through `io.write`. Confidence: high.

**6. The limiter's reported gain reduction excludes the base-rate re-limit, which is not
rare: on the 30 s loop it reduced 0.29 dB on 1 296 143 of 1 323 000 samples (98 %); on the
crest-loss / `tp worst` path 0.375 dB on 91 %.** The 8x limiter's output read
-0.011 dBTP (8x meter) against the -0.3 ceiling before the re-limit (`out_*/limiter.json`
`true_peak_8x_before_relimit_dbtp`, `relimit_passes`); on the aggressive + tilt +2 path it took 0.64 dB on 352 780 of
352 800 samples plus a second pass (`paths.log`). `planner.py:2199-2201` reports
`max_gain_reduction_db` (gr8, `solver.py:434`) as the rendered outlier; the stage's
contract (`stages/limiter.py:61-66`) says the re-limit leaves "the rest of the track
untouched". The ceiling is met (-0.3047 on all three meters). Effect: the budget table's
limiter outlier understates by ~0.3 dB and a second limiter layer runs across the whole
track. Fix: add the re-limit's max reduction to the rendered figure and the GR plot, and
investigate why the 8x stage overshoots by 0.29 dB after decimation (processing vs
metering resampler). Confidence: high (numbers), medium (cause).

**7. A `--bpm` that disagrees with the detected grid is never surfaced** (the reporting
half of 4b). `structure.py:470-477` records `detected_bpm`, `deviation_pct`,
`detected_relation`, but `evaluate.py`, `processlog.py` and `uidata.py` never read them
(grep: no hits); the process log's "Tempo: 97.0 BPM (user)" is all the user sees. Spec
3.2a promises "the analysis then reports the detected pulse's ratio to it". Fix: print
the detected tempo and relation under "Settings used" / "What was done" when they differ
by more than a few percent. Confidence: high.

**8. The key lock has no confidence gate.** `features/pitch.py:153-163` (`combined_key`)
returns a tonic whenever chroma exists; `planner.key_crossovers` uses it unconditionally.
Atonal or noisy material therefore gets a lowest crossover anywhere in 80-123.5 Hz (and
the clipper split with it) from an effectively random pitch class (`misc.log` section 4:
the table of tonic -> crossovers; the planner analysis of a loop picks A regardless of
level, but the score is 0.66 with no threshold). Fix: fall back to the default crossovers
below a key score threshold and say so in `crossover_choice`. Confidence: medium.

**9. A failed or killed `master` leaves a mixed-run work dir, and `eluate report` will
happily rebuild from it.** `cli.py:194, 362-363` clears only run.json and the process log
at start; `targets.json` and `plan_record.json` are written before the solve
(`cli.py:385-386`), while `plan.json`, `master.wav`, `solve.json`, `report.*` from the
previous run stay. `cli.do_report` (`cli.py:439-468`) never checks run.json, so after a
failure it pairs the old master with the new plan_record. (`uidata.summary` and
`from_render` do require run.json.) Evidence: `out_silence/`, `out_half/` after the
crash in finding 1 contain the analysis cache only; a second failed run after a good one
would keep the good run's master. Fix: clear every product (or write to a temp subdir
and promote), and make `do_report` refuse without run.json. Confidence: high.

**10. On a silent or near-silent input the solver would render the same levers up to
four times** (by reading, `solver.py:1575-1581`: with `rms` NaN the excess is inf,
`clamped` is False because the limiter record has no finite peak, so `want_rms` stays
True and nothing changes between renders). **Not run**: the planner crashes first
(finding 1), so the loop was never reached; re-check once finding 1 is fixed. On the DC
input, which does reach the solver, it made one render, skipped the rotation ("EQ at
its limit") and ended with a finite master at -28.8 dB RMS and a 24.7 dB shortfall.

**11. `uidata.from_render` sends the electronic shelf limits for every profile**
(`uidata.py`, `from_render` "shelves"/"max": `INPUTEQ_HS_MID_MAX_DB`,
`INPUTEQ_HS_SIDE_MAX_DB`), so a live (2.0 / 2.0) or stems (2.5 / 3.0) render shows the
shelf against the wrong bound. Fix: read `planner.inputeq.bounds` from the record.
Confidence: high.

**12. `estimate.py` constants are machine-specific and far off on this machine.**
30 s at 44.1 kHz on 4 threads: `estimate.left("analyse") = 107 s`; measured 388, 417
and 398 s (analyse 60-126 s against 14.3 estimated, one solver render 192-212 s against
46.1, report 78-122 s against 8.8; `cli_master.log`). The run was contended (four
scripts), so the ratio, not the absolute, is the point: the progress line will say
"about 2 min left" when 6 remain. Fix: scale by a one-off calibration (time the
analysis, then scale the remaining phases by measured / fitted). Confidence: medium.

### S5

13. `planner.py:1681` records `"max_db": SAT_MAX_DRIVE_DB` (2 dB) while the cap applied
    is `prof["sat_max_db"]` (1 dB under stems, `planner.py:1673`); `evaluate.py`
    `_what_was_done_md` hard-codes "at most 2 dB" in the saturation line.
14. `solver.py:678-681` `_bisect_down` docstring says it returns "the highest x seen at
    which f <= target"; it returns the lowest such x (correct for the cap, wrong words).
15. `multiband.bisect_peak_gr` returns `lo` silently on a bad bracket (`lo > hi`) and the
    lowest x on an exact plateau (`misc.log` section 12); callers avoid both today.
16. `planner.py:449, 1776`: `LOW_BAND_RELEASE_MIN_MS` (25 ms) can never bind (every
    preset's low band releases at 80 ms or more) and the docstring still mentions the
    retired solver release lever.
17. `character_choice` reason reads "auto, 70 <= 120 BPM <= 140" on a beatless default
    grid (tempo 120 is the default, `grid_informative` False; 1 s loops in
    `edge_cases.log`), implying a detected tempo.
18. `plan_from_analysis` and `plan_record` each run the multi-start input-EQ fit
    (`planner.py:2223`, `2315`); `plan_s` was 12.7-21.8 s on 30 s audio (contended).
19. `report.json["loudness"]` carries `before_rms_dbfs` and `before_rms_aes17_dbfs` with
    identical values (`evaluate.py:report`), a leftover of the 1 s reading.
20. librosa `UserWarning: n_fft=512 is too large for input signal of length=304/144` leaks
    to stderr on 1 s and 0.5 s inputs; "Trying to estimate tuning from empty frequency
    set" on silence.
21. The report's saturation line and `plan_record["saturation"]["rule"]` say "at most 2 dB"
    for every profile.
22. `limiter_gain_max(-inf, -inf, ...)` returns `inf` (callers guard with isfinite).

---

## Verified correct (coverage)

- **Monotonicity (item 1).** On two 4-5 s stereo loops and an impulse-on-pad signal,
  sweeping the threshold in 0.05 dB steps over 50-60 dB: `multiband.threshold_for_peak_gr`'s
  objective (meter-smoothed per-hit median, knees 1 and 6, ratios 1.5 and 2) and its
  outlier were non-increasing in the threshold on every band and channel (120 sweeps,
  0 violations); `glue.threshold_for_peak_gr` likewise, with float32 (`to_env32`) jitter
  of at most 9.5e-7 dB (below SOLVE_TOL 1e-4); `ceiling_for_peak_gr` exact
  (`gr = top - ceiling`); `clip_threshold_for_average_peak`'s average monotone over 133
  thresholds (0 violations), its outlier jitters by up to 3.1e-3 dB (used only in the
  no-sample-cap branch, which was monotone); `proxy_rms` non-decreasing in the drive
  over -24..36 dB (0 violations). Every reachable solve hit its target within 1e-4 dB
  and returned the highest threshold reaching it (`mono_sweep.log`). The theory holds
  too: every operator in the chain (knee curve, min filter, moving averages, release
  hull, lagging one-pole, float32 round-down, clamp, median) is order-preserving.
- **Bracketing (item 2).** Loops of 1, 3, 8, 60 s at 44.1 and 48 kHz plan and solve
  (1 s: 400 ms frames, 3 hits all loud; 3 s: 7 beats; 8 s: 18 beats, 16 loud; 60 s:
  128 beats, 50 loud). `gain_for_rms` flags `at_limit` and `clamp_limiter_gain` caps the
  request at `g_max + 6` with `request_capped` when the proxy saturates (1 s and 3 s
  loops). `hit_edges` drops beats outside (0, d), duplicates and NaN; a 0.3 s track is one
  hit; 0.5 s gives [0, 0.4, 0.5].
- **Hits consistency (item 3).** The loud mask is frozen on the stage-1 input and every
  stage receives it through `stage_hits`; `hits_for` raises on a mask whose length
  differs from the hit count; beats are identical between the three levels.
- **Key crossovers (item 4).** Table for all 12 pitch classes incl. enharmonics
  (`misc.log`): E/F/F#/G clip to 80 Hz; top crossover at most 7902 Hz, below fs/2 down
  to 16 kHz; `bands.split` raises cleanly at 8 kHz. Unknown tonic gives the defaults.
- **Solver loop (item 5).** Terminates (2 + 2 + 3 renders at most, `render_bound`);
  the kept/reverted logic and `judge_move` behave as documented in `paths.log` and the
  CLI runs; the proxy bias is -0.07..-0.17 dB (real quieter than proxy) on the loop; the
  warning "needed at least 4.9 dB ... 1.9 dB short" in report.md matches solve.json and
  run.json (`misses: ['rms']`, `warnings: ['limiter','loudness']`); true peak met on the
  worst of 4x/8x/32x (-0.3047 / -0.3050 / -0.3050). 30 s loop at 4 threads: 1 render,
  388-417 s wall (contended), 813 MB peak.
- **Profiles (item 6).** All 24 PROFILES keys are read (grep of every `prof[...]`);
  no key is read but undefined; live's `slope: "report"` gives `Targets.slope_mode`
  "report", a never-failing slope constraint, no tone rotation, no sub pull; stems
  without `--stems` is rejected by the CLI (`USER_PROFILES`) and works programmatically.
- **Character (item 7).** `character_choice` table verified (None/NaN -> moderate;
  69.9 user -> slow, 69.9 analysis -> moderate; 140.01 -> fast; octave note quoted);
  live `limiter_timing` gives transparent's 5 ms / 200 ms and its speclim lookahead
  while the band ceilings keep the character's.
- **Serialisation (item 8).** `to_dict`/`from_dict` and `save`/`load` round-trip equal
  (dataclass equality incl. envelopes); npz keys are dotted field paths
  (`multiband.gr_db.0.1`, 43 arrays); version 0.9 rejected with a clear message; run-id
  mismatch rejected; `check_environment` warns on a version difference and only logs on a
  platform difference; `to_env32` never rounds above the exact dB (max shortfall 9.5e-7
  dB, one float32 ulp at -6 dB), the "0.000001 dB" figure is right between -3 and -12 dB.
- **estimate / memory / parallel (item 9).** `estimate.left` arithmetic as documented;
  `memory.guard` drops to one thread and refuses correctly (1000-1182 MB free against a
  3.0 GB one-thread estimate); `design_budget`/`realise` give identical plans at 1, 4 and
  12 threads; `limit_blocked` and the whole render are bit-identical at 1 and 4 threads
  and on repeat; `eluate render` of the saved plan equals master.wav bit for bit
  (`cli_master.log`).
- **app_engine (item 9).** Every command emits only `@@eluate` JSON lines (no stray
  stdout): `summary` gives `state: "none"` rows for an unrendered and for a nonexistent
  file (exit 0); `analyse` on a matching cache logs "analysis reused" with no stage, with
  a different `--bpm` it emits one Analysing stage with `eta_s` / `then_s` then
  re-analyses; `preview` returns env / meter arrays (all-null arrays and `matchDb: null`
  on silence, still valid JSON); `master` on a missing file emits one error line and exit
  2; `ui-data` on a track without a render emits the raw `[Errno 2] ... report.json`
  message and exit 2 (an S5: a plain "no render yet" would be kinder). Cancel is the
  shell's `child.kill()` (`desktop/src-tauri/src/lib.rs:407-409`); the engine has no
  cancel protocol, which is safe because run.json is the completion marker and the
  Chain's spill dir is swept after 6 h (`solver.sweep_spill`). **Not run**: a kill
  mid-render (reasoned from the code and from `test_master_sweeps_stale_spill_dirs`).
- **Preserve (item 10).** `--preserve` on the 30 s loop: `inputeq` and `saturate` None,
  multiband / glue / speclim / clipper / limiter planned exactly as without it; the
  solver adds `ltas_shape` and `section_deltas` constraints (both missed here: 3.38 dB
  shape change, 1.86 LU section rise) and drops `slope`; nothing acts on either (no
  lever), the rotation is unavailable, and with the limiter clamped no re-render
  happens, so the run is one render. report.md lists "Section balance: 1.86 against
  0.00 +- 1.00, missed by 0.86" under Warnings and in the Targets table; the process
  log's "Preserve mix: what it does in this build" states exactly this (turns off EQ and
  saturation, adds two measured checks that count in the score, dynamics unchanged) —
  truthful. The section deltas are therefore reported and scored, not constrained.
- **Reproducibility (item 11).** As above: bitwise.
- **Odd inputs (item 2, `edge_cases.log`).** Mono-in-stereo (S gated in every band, 2
  renders, ok, RMS -4.25), hard-panned (S not gated, outlier-clamped, finite), a 1 kHz
  sine (limiter idle, target met by gain, slope miss and rotation skipped), four noise
  bursts (5 hits all loud, tempo read as 30 BPM, clamped), peak -40 dBFS (input gain
  +38 dB, plan identical to the -6 dBFS loop to 1e-12), live and stems profiles
  (clipper off under live's moderate, 0.5 dB target and 2 / 4 dB limiter caps under
  stems), 48 kHz at every length: all plan and solve without error.
- **Control paths (`paths.py`).** `--crest-mode floor` with 8 dB floors, `--character
  slow` (kick detective, 12 onsets), `transparent --makeup auto --glue-mode parallel
  --streaming` (16-bit grid, -1.005 dBTP), explicit `--xover 100,500,2000,8000 --subside
  80 --crest-mode loss --tp-meter worst --no-speclim --makeup rms`, `aggressive --tilt 2
  --target-rms -2` (3 renders, two rotations kept, slope -3.20 -> -2.93 against -2.5,
  loudness fell 1 dB per the design that judges the rotation without the RMS term),
  `--profile live --preserve --bpm 128`: every plan solves and round-trips.
- Control paths (`paths.log`): slow + auto kick detective (12 onsets, 1.5 dB), transparent
  + makeup auto + parallel glue + streaming (16-bit grid, -1.005 dBTP), explicit xover +
  crest loss + tp worst + makeup rms, aggressive + tilt + hot target: all plan, solve and
  round-trip.

## Experiments not run

- A full `eluate master` at `--threads 1` on the 30 s loop (only `render_plan` at 1 vs 4
  threads and `eluate render` were compared, bitwise equal; `test_parallel.py::
  test_master_bit_identical_at_any_thread_count` covers the full run on the fixture).
- A full solve of the 60 s loops (planned and realised only: 171 s analysis + 270-400 s
  realise under contention).
- A full master at 22.05 kHz (blocked by finding 2) and on silence / < 1 s input (blocked
  by finding 1).
- A kill mid-render (cancel) and the silent-input solver loop (finding 10).
- `--profile stems` without `--stems` through the CLI (rejected by `USER_PROFILES`;
  exercised through `Controls` only).

## Open questions

1. How often does finding 3 bind on real premasters (a sparse top band over a dense low
   end)? The gate (24 dB below the input's 1 s RMS) does not catch it when the hits are
   loud.
2. The 0.29 dB everywhere-overshoot of the 8x limiter before the re-limit (finding 6): is
   it the decimation filter's Gibbs on flat-topped peaks or the metering prototype? It
   decides whether the re-limit is a correction or a second limiter by design.
3. A clamped render that lands *over* the target by more than 0.3 dB is never re-rendered
   (`want_rms` requires `not clamped`, `solver.py:1576`); the bias measured here is
   negative, so it did not occur, but a positive bias of > 0.3 dB would leave a loud miss
   attributed to the "limiter budget".
4. `--max-renders 1` still allows two renders (the true-peak ceiling render is outside
   the count), documented in `render_bound`; intended?
5. The tone rotation is judged without the RMS term (R7a): on the aggressive + tilt path
   two kept rotations cost 1.0 dB of loudness for 0.27 dB/oct of slope while the master
   was already clamped and short. Is a rotation that deepens an RMS miss meant to be kept?
6. The key lock on atonal material (a 1 kHz sine reads "B", noise bursts "E") moves the
   sub crossover and the clipper split: should a key score below some value fall back to
   the defaults (finding 8)?
