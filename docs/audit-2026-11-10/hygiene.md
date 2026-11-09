# Eluate hygiene, test-health and input-robustness audit (branch audit/2026-11-10, HEAD fb082d1)

Auditor scope: code hygiene, test health, input robustness, packaging, doc drift. Machine: Windows 11, 12 logical cores,
32 GB; heavily contended during the audit (3 to 5 other auditors' test suites and masters running at once), so every
wall time below is inflated by roughly 3x against an idle machine. Scratch evidence (logs, outputs) lives in
`<scratch>/audit/hygiene/`
(`fast.log`, `slow.log`, `robust.log`, `runs/*.stdout|stderr`, `prof.out`, `ruff_full.txt`, `vulture_src.txt`).
No tracked repo file was modified; ruff 0.17.4, vulture 2.16 and psutil 7.2.2 were added to the disposable venv.

Severity key: S1 wrong output; S2 wrong numbers/reports; S3 crash or data loss on plausible input; S4
hygiene/maintainability/perf; S5 nit.

## Findings

Counts: S1 0, S2 2, S3 6, S4 13, S5 6 (27 findings).

### S1

None found in this scope (output correctness of the DSP is other auditors' scope).

### S2

**F1 (S2). `eluate report` re-reads the "before" audio from `source.json`'s path without checking its SHA-256.**
`src/eluate/cli.py:439-442` (`do_report`): `x, fs = io.read(src["path"])`. source.json records `sha256`, `size`, `mtime`
(and `do_master` relies on the hash for cache reuse), but report never compares them. If the premaster was edited,
replaced or overwritten after the master (see F3, where it is overwritten by the master itself), the rebuilt report,
A/B files and graphs silently compare against the wrong audio; nothing in the report says so. The desktop shell's
"rebuild report" goes through the same path (`_promote` -> `do_report`, cli.py:604).
Evidence: code read; F3's run leaves `runs/same/source.json` with sha `b8d8443d...` while the file at that path now
hashes `072a60d4...` (it is the master), so a later `eluate report runs/same` compares the master with itself.
Fix: in do_report hash the file (file_sha256 exists, cli.py:136) and raise ValueError("the premaster at ... changed
since the master was made") on mismatch (or warn and record `source_changed: true` in report.json).
Confidence: high.

**F2 (S2). Under `--preserve`, `--tilt` is accepted, silently ignored, and the report still prints a tilt target and an
"error" against it.** `src/eluate/evaluate.py:563-566` builds `rep["tilt"]["target_stereo"] = targets.slope_db_oct`
and `error_stereo` regardless of mode; `evaluate.py:997-1001` prints them. In preserve mode inputeq/tone/saturate
are None (`planner.py:221`, `1610`, `1684`) and the slope is not a constraint (`misses` = `['rms', 'ltas_shape']`).
Evidence: `eluate master stereo10s.wav --preserve --tilt 2` -> exit 0, no message about the tilt; report.md:
"Stereo: before -5.74, after -5.99, target -2.50, error -3.49" (runs/o_pt/report.md). run.json controls record
`tilt_db: 2.0`. A reader sees a 3.5 dB/oct miss on a target nothing aimed at.
Fix: in preserve mode write `target_stereo: null` / print "target: n/a (preserve mix)"; in `Controls.__post_init__`
(planner.py:584) either reject `tilt_db != 0 and preserve_mix` or log "--tilt has no effect with --preserve".
Confidence: high.

### S3

**F3 (S3, data loss). An input that lives in the output directory as `master.wav` is overwritten by the master;
exit 0, no warning.** `src/eluate/cli.py:355-` (`do_master`) never compares `inp` with `out / "master.wav"` (or any
other file it writes: `stems-sum.wav`, `ab/*.wav`, `pass-k/corrected-sum.wav`). The README promises "Input files are
not modified." Plausible trigger: re-mastering a previous master in place (`eluate master out/master.wav -o out/`), or
a desktop batch pointed at its own work folder.
Evidence: `runs/same/master.wav` sha256 before `b8d8443d5dc6...`, after `072a60d4c5d9...`; rc=0, "master done in 535 s"
(robust.log `same_dir_master`).
Fix: early in do_master / do_master_stems: `if any(Path(inp).resolve() == (out / n).resolve() for n in WRITTEN_AUDIO)`
(or `os.path.samefile` when it exists) raise ValueError("the input is one of the files this run writes; choose another
output folder"). Confidence: high.

**F4 (S3, data loss of the completion marker). A typo in the input path (or a memory-guard refusal) deletes the
previous good run's `run.json` and process log before anything is validated.** `src/eluate/cli.py:362-368`: the
`CLEARED_AT_START` loop (`run.json`, `process-log.md`, `process-log.html`) runs before `sf.info(str(inp))` and before
`memory.guard`. The previous master is then reported as incomplete by the app (run.json is the completion marker) and
its process log is gone.
Evidence: copied a finished run to `runs/out_typo`, ran `eluate master in/typo_missing.wav -o runs/out_typo` -> rc 2,
"Error opening '...typo_missing.wav': System error."; afterwards `run.json`, `process-log.md`, `process-log.html`
are missing, every other file is left.
Fix: move the three unlinks after `sf.info` and `memory.guard` (and after `Controls` validation in the stems path,
`do_master_stems` -> `sum_stems`). Confidence: high.

**F5 (S3/S4). A locked or read-only output file fails only after the whole solve, with a raw WinError naming the
temporary file, and leaves a half-updated directory.** `src/eluate/io.py:62-71` (`atomic_write`, `os.replace`).
Evidence: `runs/out_ro/master.wav` set read-only (`attrib +R`; the same happens when the file is open in a DAW or
media player on Windows): rc 2 after 387 s (three full renders), message
`eluate: [WinError 5] Access is denied: '...\out_ro\master.ff07fb6f9f62.tmp.wav' -> '...\out_ro\master.wav'`. The
directory then holds this run's plan.json/plan.npz/plan_record.json/targets.json/source.json next to the previous
run's master/report/solve.json (run.json removed, so it is at least marked incomplete).
Fix: probe writability of `out/master.wav` (open for append, or `os.access` plus a try-rename of a temp file) right
after the memory guard, and catch PermissionError in `atomic_write` to re-raise "cannot replace X: is it open in
another program?". Confidence: high.

**F6 (S3/S4). Short inputs fail with library-internal messages; the documented floor is wrong.** The contract and
`multiband.band_crest` say "need at least 2 samples"; in practice analysis needs about 0.1 s.
Evidence (`eluate analyse` / `master`): 0 and 1 frames -> "eluate: need at least 2 samples" (good); 2 frames ->
"The length of the input vector x must be greater than padlen, which is 27." (scipy filtfilt); 100 frames -> same, plus
four librosa UserWarnings on stderr ("n_fft=2048 is too large for input signal of length=47"); 1000 frames ->
"window is longer than input signal"; 4410 frames (0.1 s) -> OK. Inputs under about 0.4 s also get "input gain
+0.00 dB" (the gated loudness is -inf, so normalisation is silently skipped; 0.5 s noise at -20 dBFS got +0.00, 1 s
got +1.99).
Fix: one guard in `planner.analyse`/`io.read` callers: `if n < MIN_FRAMES (e.g. round(0.5 * fs)): raise ValueError(
f"{path}: {n / fs:.2f} s is too short to master (at least 0.5 s)")`. Confidence: high.

**F7 (S3). Any 22.05 kHz (and, by the same arithmetic, 24 kHz) input crashes in planning:
"Each lower bound must be strictly less than each upper bound."** `src/eluate/planner.py:1314-1316`
(`_corner_range`) clamps the air-shelf corner range `EQ_AIR_FC_HZ = (10000, 16000)` (planner.py:308) to
`EQ_HS_MAX_FS_SHARE * fs = 0.45 * 22050 = 9922.5 Hz`, giving the degenerate range (9922.5, 9922.5); the side-shelf
fit then calls `least_squares(..., bounds=(lo, hi))` with `log2(ar[0]) == log2(ar[1])` (planner.py:1549-1554) and scipy
raises. Traceback (re-run of `do_plan` on the cached analysis in runs/o_22k): `plan_from_analysis` (2223) ->
`plan_inputeq` (1609) -> `fit_inputeq` (1455) -> `_fit_side_shelves` (1554) -> scipy `least_squares` ValueError.
Evidence: `eluate master fs22050.wav` -> rc 2 after the analysis, message above, nothing tells the user that the
sample rate is the cause; the analysis cache is left in the output dir. 44.1, 48 and 192 kHz master fine.
Fix: when `ar[1] - ar[0] < 1 Hz` drop the air shelf from the fit (fix its gain at 0 and its parameter out of the
optimisation), same for presence; or refuse fs < 32 kHz up front in `do_master` with "Eluate needs 32 kHz or more".
Add a 22.05 kHz case to tests (none exists: `grep 22050 tests` finds only meter/resample unit tests).
Confidence: high.

**F8 (S3). Silent (or digital-black) input runs the whole analysis, then fails with "input measured_dbfs contains NaN
or Inf".** `src/eluate/plan.py:336-343` (`InputPlan.__post_init__` -> `_finite`) rejects the -inf loudness that
`planner.analyse` measured, but only at plan time. Evidence: 10 s of zeros -> rc 2 after 22 s; 20 min of zeros at
44.1 kHz -> the memory guard lets it through (estimate 10 516 MiB vs 13 366 MiB free, 2 threads kept), analysis runs
1495 s (25 min, contended) and then rc 2 with the same message; stderr also carries two librosa "Trying to estimate
tuning from empty frequency set" UserWarnings. features.npz/analysis.json/source.json of a silent track are left
behind. Fix: in `do_master`/`do_analyse` right after `io.read`: `if not np.any(x): raise ValueError(f"{inp} is
silent")` (or gate on the 1 s RMS below -90 dBFS), before the analysis. Confidence: high.

### S4

**F9 (S4, packaging). `build-system.requires = ["setuptools>=68"]` cannot build this pyproject; the SPDX
`license = "AGPL-3.0-or-later"` plus `license-files` need setuptools >= 77.** `pyproject.toml:2,8-9`.
Evidence (sdist of HEAD built with `setuptools.build_meta.build_wheel` in a throwaway venv): 68.2.2 -> FAIL "invalid
pyproject.toml config: `project.license` ... must be valid exactly by one definition"; 76.1.0 -> same FAIL;
77.0.3 -> OK `eluate-0.1.0b0-py3-none-any.whl`. pip's default isolated build fetches the newest setuptools, so CI and
most users never see it; `pip install --no-build-isolation`, offline mirrors and distro builds with an older pinned
setuptools fail.
Fix: `requires = ["setuptools>=77"]`. Confidence: high.

**F10 (S4, LF rule). Two source files are committed with CRLF, one with mixed endings; there is no `.gitattributes`.**
`git ls-files --eol`: `i/crlf src/eluate/cli.py`, `i/crlf src/eluate/stemfix.py`, `i/mixed tests/test_stemfix.py`
(143 others `i/lf`). The maintainer machine has `core.autocrlf=true`, so the whole working tree is CRLF and new files
are easy to commit either way. Runtime text output is clean: every JSON/MD/HTML the baseline run wrote is LF
(`grep -c $'\r'` = 0) and has no NaN/Infinity tokens.
Fix: add `.gitattributes` with `* text=auto eol=lf` (and `*.png binary`), then `git add --renormalize .`.
Confidence: high.

**F11 (S4, dead code and dead constants with docstring drift).**
- `planner.py:372-374` `MB_CEILING_ABOVE_P99_DB`, `MB_ATTACK_MS`, `MB_RELEASE_MS` are never read; the module docstring
  (`planner.py:125-128`) still says "Attack / release per band from MB_ATTACK_MS / MB_RELEASE_MS"; the values actually
  come from the character presets (`planner.py:420-444`, `mb_attack_ms`, read at `planner.py:1787`).
- `stages/glue.py:101-102` `SOLVE_TOL_DB`, `SOLVE_ITER` unused (glue bisects with `multiband.bisect_peak_gr`).
- Never called anywhere: `evaluate._pairs` (617), `features/_common.Context.band_ms` (178), `glue.wet_peak_gr_db`
  (181), `features/borrowed.detect_onsets_fb` (167; only named in docstrings).
- Called only by tests: `envelope.roughness`, `meter.plr`, `solver.limiter_input_gain`, `solver.crest_excess`,
  `solver._worst`, `glue.net_peak_gr_db`, `glue.gain_trace_db`, `multiband.comp_peak_gr_db`.
- Unused imports: `features/structure.py:115` (`pitch as _pitch`), `features/resonance.py:108` (`borrowed`),
  `planner.py:269` (`INPUTEQ_HIGH_SHELF_FC_HZ`); unused locals `borrowed.py:160 df`, `resonance.py:523-524 cp, cw`,
  `structure.py:165 onset_t`, `structure.py:440/443 h_up`.
- `stages/keyeq.py` is a stub (fine, plan always disables it, `planner.py:2255`), but its docstring says
  "It depends on features/pitch.py (workstream C), which does not exist yet" (keyeq.py:15) - pitch.py exists and
  computes `keyeq_eligible`. An enabled keyeq in a hand-edited plan.json raises NotImplementedError
  (`solver.py:602`), which `cli.main` does not catch (only ValueError/OSError/SoundFileError) -> traceback.
- `stages/tone.py` is still imported and applied by solver/planner when `plan.tone` is set, but the planner always
  sets `tone=None` (`planner.py:2250`); live only through tests and old plan files.
- Section header "levers" (`solver.py:840`) survives the retirement of the lever classes; the docs/gain-staging.md
  retired list is otherwise confirmed gone (0 hits for `_lever_classes`, `MB_SCALE_MAX`, `DRIVING_BANDS`, `GLUE_MIX_*`,
  `CLIP_STEP_DB`, `THR_*`, `IMPROVE_SCORE`, `_nearest_feasible`, `_finite_or`, `NEAREST_*`, `target_s95`,
  `DEFAULT_TARGET_S95`, `HARMONIC_RANGE`, `split_deficit`, `SAT_DRIVE_CAP_DB`, `SAT_WIDTH_DRIVE_DB`,
  `plan_tone_saturate`, `--genre`, `GENRE_PRESETS`, `master_feasible.wav`, `plan_feasible.json`; `harmonic_blend` and
  `SAT_DRIVE_PER_DB` appear only in a test asserting their absence; `SaturatePlan.harmonic` remains as a plan field,
  always 1.0).
Fix: delete or move to tests; fix the two docstrings. Confidence: high (vulture 60 % hits checked by grep).

**F12 (S4, duplicated logic).**
- `ATTACK_FLOOR = 2` defined twice: `stages/multiband.py:144` (control samples) and `stages/limiter.py:97` (audio
  samples); solver uses the limiter one (`solver.py:306, 526, 574`).
- `SOLVE_TOL_DB = 1e-4` / `SOLVE_ITER = 80` in `glue.py:101`, `limiter.py:101`, `multiband.py:179`, and three
  bisection loops: `multiband.bisect_peak_gr` (422), the inline loop in `limiter.clip_threshold_for_peak_gr`
  (324-338, same algorithm), `solver._bisect_down` (678); plus brentq-based roots at `solver.py:835, 1361`.
- `_db(v)` (20 log10 with -inf floor) four times: `limiter.py:376`, `multiband.py:322`, `stemfix.py:156`,
  `stems.py:74`; `_pdb` in stems.py.
- `_frame_peaks` twice with the same name and different semantics: `multiband.py:326` (peak per control segment via
  control_coord) vs `stemfix.py:175` (peak per 1 ms frame via ceil starts).
- Crest: `features/crest.py:analyse` (analysis), `multiband.band_crest` (965, used by evaluate for before/after),
  `multiband.band_crest_db` (541, ratio map), `meter.crest_db` (458). crest.py and band_crest compute the same
  quantity with a deliberately different interior-frame fallback (band_crest falls back to all gated frames when no
  gated frame is interior; crest.py only when no frame is interior), and different frame grids (analysis grid vs
  `CREST_HOP_S`). Before and after both use band_crest, so the constraint is self-consistent, but the analysis
  `p95_db` and the report's `before_p95_db` can differ on the same audio.
- Crossover defaults: `bands.DEFAULT_CROSSOVERS` is the single definition (good), but default-argument copies
  `crossovers_hz=bands.DEFAULT_CROSSOVERS` remain in stage signatures while the run actually uses key-locked
  crossovers (87.3/349.2/1396.9/5587.7 Hz on every synthetic run here), so any caller that forgets to pass the plan's
  crossovers measures different bands silently.
Fix: one `eluate/dsp_util.py` (db, pdb, bisect_monotone, ATTACK_FLOOR_SAMPLES) and one crest kernel with a
`fallback=` argument. Confidence: high.

**F13 (S4, private imports).** `stages/glue.py:92` imports `_frame_peaks` from `stages/multiband` (and
`glue.py:223` re-imports `bisect_peak_gr` lazily from the module it already imports at top level, redundant).
`oa.py:39` imports scipy's private `scipy.signal._signaltools._calc_oa_lens`: present in the pinned scipy 1.18.1,
signature `(s1, s2)`, `_calc_oa_lens(100000, 2049) = (24192, 2048, 22144, 2049)`. Fallback: `ImportError` -> `None`
-> `oa_stream` returns False and callers use whole-array `scipy.signal.oaconvolve` (correct, not memory-bounded);
`tests/test_oa.py` asserts the identity. Not covered: a scipy that keeps the name but changes the signature or return
arity raises TypeError/ValueError at `oa.py:52` instead of falling back. pyproject's `scipy<1.19` bound is the real
guard. Fix: wrap the call in `try/except (TypeError, ValueError): return False` and rename `_frame_peaks` public.
Confidence: high.

**F14 (S4, structure).** No genuine top-level import cycles (the SCCs a naive graph shows are package `__init__`
re-exports), but four latent cycles are broken only by function-level imports: planner -> solver (planner.py lazy,
solver imports planner at top), evaluate -> planner/solver, stems/stemfix -> planner, processlog -> evaluate.
`planner.py` is 2347 lines with a 285-line module docstring; natural split points are its own section markers:
constants + PROFILES/CHARACTERS (285-559), Controls (559-704), input/crossovers/notches (704-1047), tone + input EQ
(1047-1689, about 640 lines), multiband/dynamics/limiter (1689-2212), top level (2212-). `solver.py` (1637) likewise
splits at blocked limiter (213-580) / measurement (1030-1315) / solve (1315-).
Confidence: high.

**F15 (S4, docs reference files that are not in the public tree).** `tools/public_snapshot.py:36-52` excludes
`audit/`, `docs/dev/`, `docs/macos-report-verdicts.md`, `docs/quality-review-*.md`, `docs/reuse-survey.md`, `work/`,
but the public code cites them: `memory.py:2` (macos-report-verdicts.md), `memory.py:9` and `estimate.py:19`
(work/renders), `planner.py:413` (quality-review-2026-10-22.md), `bands.py:39`, `envelope.py:44`, `pitch.py:101`
(audit/brief-0N.md), `borrowed.py:5` and `tests/test_feature_crosscheck.py:7` (reuse-survey.md),
`docs/spec-draft.md:6` (audit/01-research-review.md), plus 126 bare "report N.N" / "audit NN Xn" ids in src/.
`public_snapshot.py` itself lists the internal file names it excludes (SESSION_HANDOFF_*, macos-claude-report.md,
tools/audit_dispatch.py), and in the public tree it always exits with "docs/dev/public-snapshot-patterns.txt is
missing" (the patterns file is excluded by design). `borrowed.py:25-35` names an external project ("TRANSMUTATION
engine/fusion/metrics.py") 18 times; check that against the brief's "no third-party names" rule (it is code
provenance, not an artist, so probably fine).
Fix: replace internal-doc citations with the public doc section or drop them; ship public_snapshot.py only in the
private repo. Confidence: high.

**F16 (S4, perf). Every analysis JIT-compiles librosa's beat-tracker gufuncs (`cache=False` upstream).**
cProfile of `eluate analyse` on a 0.1 s file: 51 s total, of which 25.4 s in numba `ufuncbuilder._compile_element`
(13 compilations, 4 calls from `librosa/beat.py`, whose `@numba.guvectorize(..., cache=False)` at beat.py:565, 604,
655, 704, 724) and about 30 s in imports (contended). Idle-machine cost is roughly 8 to 10 s fixed per run, paid again
for every stems pass. Separately, `cli.py:92-100` imports evaluate/planner/solver/stems (scipy, librosa, numba,
matplotlib) before argparse, so `--help` and every validation error cost a full import (`python -W error -c "import
eluate, eluate.cli, ..."` = 4.2 s; each rejected flag in the robustness matrix took 14 to 26 s under contention).
Fix: parse args before the heavy imports (lazy-import in `main`); for the frozen engine, warm/precompile the beat
gufuncs once or use a cached reimplementation of `__beat_local_score`/`__beat_track_dp`. Confidence: medium (exact
idle figure not measured).
Profile of the 60 s / 48 kHz master (`--threads 2`, see the profile table): solve 1901 s of 2279 s (83 %); the
clipper threshold search (`planner.solve_clipper` -> `solver.clip_threshold_for_average_peak` -> `_bisect_down`,
solver.py:678-779) costs 527 s, 28 % of the solve, because each of its 42 bisection steps re-runs a whole-track
`reading` (11.7 s each: 8x upsample + limiter proxy); 36 `limit_blocked` calls cost 947 s; `envelope.release_hull`
(envelope.py:159) is the hottest pure-Python frame (81 s self time). Fix: bracket + brentq (as `gain_for_rms` does)
instead of a fixed-tolerance bisection, and cache the upsampled pre-clipper signal across the steps.

**F17 (S4, test tiers).** `slow` is defined as "takes more than about 5 s" (pyproject.toml:37), but the fast tier holds
many tests far above that even allowing for 3x contention: `test_feature_analyse::test_common_grid` setup 60.9 s,
`test_stage_limiter::test_true_peak_gate[1.0-0-qp-48000]` 50.2 s, `test_planner::test_every_section_filled` setup
46.2 s, `test_profiles::test_live_plan_switches_off_and_scales_down` 33.7 s, `test_gain_staging::
test_chain_rotation_keeps_the_input_gain` 31.5 s (top 20 all > 16 s). Fast tier: 30 min 42 s here; CONTRIBUTING says
"a few minutes". The heavy setups are module fixtures (a full analysis) shared by many fast tests, so marking the
consumers slow would empty the fast tier; better to shrink the fixture signal (e.g. 4 s instead of the current
length) or cache the analysis in a session fixture. CI fast tier command (`python -m pytest -m "not slow" -q`) matches
CONTRIBUTING and README; CI installs from the locks while README/CONTRIBUTING tell contributors `pip install -e
".[dev]"` (ranges), so a contributor can run newer numpy/scipy/librosa than CI and see bit-exactness tests differ.
Confidence: high for the numbers, medium for the cause split. The slow tier (73 tests) took 61 min here; its top three
(`test_cli::test_master_and_report` 350.7 s, `test_average_peak::test_average_reading_reduces_the_crest_more_than_the_max_reading`
347.6 s, `test_app_engine::test_master_after_analyse_reuses_it` 343.6 s) are full masters. The 33 slow tests below the
40-row durations cut each took < 17.3 s under contention; whether some are < 5 s idle (mis-marked) was not
determinable from this run. Wall-clock/randomness/thread dependence: `test_app_engine.py:116` uses `time.time()` but
passes `now=` explicitly (deterministic); `test_cli.py:172-173` and `test_memory.py:24` read the real machine's free
memory (pass on any machine with > 0 MiB free); `test_parallel.py:141` asserts the default thread count equals
`parallel.DEFAULT_THREADS` (cores - 2, machine dependent but self-consistent); no unseeded RNG found (`grep` for
`np.random.*(`/`default_rng()` in tests: none). A second ResourceWarning in the slow tier comes from
`cli._reuse_analysis` (cli.py:339), same np.load cause as F24.

**F18 (S4, controls silently changed).** `--subside 20` is clipped to 60 Hz with no message
(`planner.py:792`, `np.clip(fc, *SUBSIDE_RANGE_HZ)`), while run.json `controls.subside_hz` records 20.0 and
plan.json `sidestrip.fc_hz` 60.0 (runs/o_sub20). Same pattern for `--preserve --tilt` (F2). `--stems` with a single
file is accepted and runs the full three-pass stems flow (1025 s here) although there is nothing to attribute.
Fix: validate `subside_hz` against SUBSIDE_RANGE_HZ in `Controls.__post_init__` (planner.py:608) like tilt; require
two or more stems. Confidence: high.

**F19 (S4, JSON/text output consistency).** `io.write_json` uses `indent=2` and `allow_nan=True` (io.py:105) while
`Plan.save` uses `indent=1, allow_nan=False` (plan.py:1096, 1113); NaN safety of the other JSONs depends on every
caller remembering `evaluate._clean` (they do today; no NaN/Infinity token in any output file of the runs here).
`processlog.write` uses `Path.write_text(..., newline="\n")` directly (processlog.py:571-572), not
`io.write_text`, so the process log is the only text output that is not atomic (documented in cli.py:86). The PNG
plots are not atomic either. Fix: `allow_nan=False` in write_json (fail loudly), one indent.
Confidence: high.

**F20 (S4, latent closure bugs flagged by ruff).** B023/F821 at `features/crest.py:47-48` and
`stages/multiband.py:1007-1008`: closures `view`/`sig` capture loop variable `lr`, which the loop then `del`s. Safe
today because each closure is called only inside its iteration, but any refactor that defers the call reads the
next band or raises NameError. Same B023 pattern at `resonance.py:740,749`, `transients.py:205-206`, `notch.py:219`,
`multiband.py:719` (lambda over `k`, `band` passed to `parallel.pmap`, consumed immediately). Fix: bind as default
args (`def view(v, lr=lr)`). Confidence: high that it is latent only.

**F21 (S4, documentation drift).**
README (checked against code): every flag it names exists (`--target-rms`, `--bpm`, `--stems`, `--mix`, `--profile
live`, `--streaming`); defaults match (`planner.DEFAULT_TARGET_RMS_DBFS = -4.1`, `CEILING_DBTP = -0.3`,
`STREAMING_CEILING_DBTP = -1.0`, `OutputPlan(24)` / `OutputPlan(16, "tpdf")`, stems profile with limiter budget 2.0 /
outlier 4.0). Wrong: "Input files are not modified" (F3). CONTRIBUTING: "fast tier (a few minutes)" (30 min here,
probably 8 to 10 idle; F17). docs/gain-staging.md 14.2: "default `planner.DEFAULT_TARGET_RMS_DBFS` = -7.7" and "chunk
1's meter, plain dBFS, 1 s" - code is -4.1 on the 300 ms AES17 meter (the section is a historical log but reads as
the current contract).
docs/spec-draft.md (draft 0.5) items the code contradicts, top 10:
1. 3.6: "`S95 == target +- 0.3 LU`" - S95 is retired as a target; the constraint is the RMS target +-0.3 dB
   (`Targets.rms_dbfs`).
2. 3.6: "true peak <= -1 dBTP on the worst of 4x/8x/32x (`tp_meter="worst"`, default)" - default ceiling is -0.3 dBTP
   with the 8x meter at 24 bits; -1 dBTP and "worst" only under `--streaming` (planner.py:390-391, 617).
3. 3.6 Method: lever classes, coordinate descent, "On infeasibility, report the binding constraint and the nearest
   feasible point" - all retired (gain-staging 14.2 list); the solver is planned render + TP render + 2 slope + 3
   loudness renders (`solver.py:167-176`, `render_bound`).
4. 4 CLI: `eluate plan work/ --tilt 0 --target -6.5 [--genre techno] ... [--harmonic 0.5]` - `--target`, `--genre`,
   `--harmonic` do not exist (allow_abbrev=False, so `--target` now errors); missing `--target-rms`, `--character`,
   `--streaming`, `--profile`, `--threads`, `--stems`.
5. 3.5 item 4 / 9 "explicit `tone` stage" - the planner always leaves tone None (`planner.py:2250`); the spectral move
   is the input EQ (stages/inputeq.py), which the spec's stage list does not mention (nor speclim, gain, output).
6. 3.5 item 5 saturate: "drive capped at 12 dB ... default drive cap 6 dB", "harmonic-vs-linear blend" - code:
   `SAT_MAX_DRIVE_DB = 2.0`, harmonic share retired, drive = band deficit (planner.saturation_drive_db).
7. 3.5 item 6 glue: "single-band parallel compressor ... lagging attack default 30 ms" - default `glue_mode="serial"`,
   `GLUE_ATTACK_MS = 0.4` (FET style), `GLUE_RATIO = 2.0`.
8. 9 Decisions: "crossovers 100/500/2000/8000", "S95 as control with genre presets" - code default
   100/400/1600/6400 and, without `--xover`, key-locked crossovers (87.3/349.2/1396.9/5587.7 for every A-key synthetic
   here); genre presets retired.
9. 8 Licence: "Deferred by maintainer" - licensed AGPL-3.0-or-later (pyproject.toml:8, LICENSE).
10. 1 and 2: pipeline "solver loop: adjust plan.json until constraints met or infeasible"; layout `offline-master/`,
   `docs/architecture.md` (does not exist), module list without planner, stems, stemfix, noise, memory, parallel, oa,
   estimate, processlog, app_engine, uidata, stages/gain|glue|inputeq|output|resample. 0: environment table is
   stale (Python 3.14 / librosa 0.10.2 vs the pinned 3.12 / librosa 1.0.0) and describes the internal `agy` audit
   tooling.
Spec 7's test gates do all have tests (bands 1e-9 / -6.02 dB, meter 997 Hz at five rates, pitch root >= 95 %,
structure +-1 bar, etc.); the EBU vector gates skip without the downloaded vectors (72 skips).
Fix: mark spec-draft.md as historical and point to gain-staging.md / module docstrings, or update 3.5, 3.6, 4, 8, 9.
Confidence: high.

### S5

**F22 (S5).** Error messages that do not tell the user what to do: missing input -> "Error opening '...': System
error." (libsndfile; say "file not found"); `--crest-mode floor` without floors -> "crest_mode 'floor' needs
crest_floors_db (no calibrated default)" (internal field name; say `--crest-floor a,b,c,d,e`); read-only output
location -> raw "[WinError 5] Access is denied: 'C:\Windows\System32\eluate_audit_out'" (acceptable).
Validation failures after `out.mkdir` leave an empty output directory (o_zero, o_xdesc, ...).

**F23 (S5).** report.md lists the slope miss twice under Warnings ("Slope: reached -5.22 dB/oct against -4.50; at its
limit ..." from `warning_text`, evaluate.py:663, and "Slope: -5.22 against -4.50 +- 0.30, missed by 0.42." from
`miss_text`, evaluate.py:708). Merge or drop the second.

**F24 (S5).** ResourceWarning in the fast tier: `tests/test_cli.py:120` - the truncated-npz path of
`cli._load_analysis` (cli.py:244) leaves numpy's file handle to the GC on BadZipFile. On Windows an open handle can
block the immediate re-analysis's `os.replace` in the long-lived app engine. Fix: open the file yourself
(`with open(p, "rb") as fh, np.load(fh, allow_pickle=False) as z`).

**F25 (S5).** No linter config in pyproject although the code is written to a 100-column limit (ruff E501 at 88:
1345 in src, at 100: 187). Add `[tool.ruff] line-length = 100` and a CI `ruff check` with F, B (minus B905), UP.

**F26 (S5).** `optional-dependencies.demucs = ["demucs"]` is unbounded and unlocked (pulls torch, which may pin numpy
outside `<2.6`). make_lock.py: correct for runtime and dev closures (markers evaluated with `extra=""`, extras
followed, eluate removed), but imports `packaging`, which is not a declared dev dependency (it is present only
transitively through pooch/librosa); the lock omits build requirements (setuptools), so F9 is not caught by a lock
install either. `uv pip check`: "All installed packages are compatible" (45 packages; venv matches the lock).

**F27 (S5, docstring spot-check, 10 checked).** Correct: cli.py:65 (-4.1), memory.py:5-13 (180 MiB/M frames + 240,
"6 min 44.1 kHz about 3.0 GB" = 3.10 GB, "10 min 96 kHz about 10 GB" = 10.6 GB), crest.py:4-5 (DEFAULT_CROSSOVERS),
planner.py:120-122 (sidestrip 100 Hz / 0.1 / [60, 150]), planner.py:117 (KICK_MIN_RATE_HZ 0.5), cli.py:46-60
(STEM_PASSES_MAX = 2, LIMIT_WEIGHT 0.5, STEM_IMPROVE 0.25), solver.py:94-110 (SLOPE_STEPS 2, LOUDNESS_STEPS 3,
MAX_OUTER 8), cli `--tilt` help "around -4.5, +-3" (REFERENCE_SLOPE -4.5, TILT_MAX 3.0). Wrong: planner.py:127-128
(MB_ATTACK_MS / MB_RELEASE_MS, F11) and keyeq.py:15 (pitch.py "does not exist yet", F11). An automated scan of every
`NAME (value)` quote in src and docs against the live constants found no other mismatch (the hits were unit
conversions such as `METER_S (300 ms)`).

## Test runs

All runs: `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider -rs -W default`, pinned lock venv, machine
contended by other auditors (times about 3x idle). pytest-timeout is not installed, so no `--timeout`. The full
suite was run as its two disjoint tiers (fast, then `-m slow`); together they cover all 1152 collected tests.

| Tier | Command | Passed | Failed | Skipped | Deselected | Warnings | Wall time |
|---|---|---|---|---|---|---|---|
| fast | `-m "not slow" --durations=20` | 1004 | 0 | 72 | 76 | 13 | 30 min 42 s |
| slow | `-m slow --durations=40` | 73 | 0 | 3 | 1076 | 17 | 61 min 22 s |
| full (sum) | | 1077 | 0 | 75 | - | 30 | 92 min |
| CI fast job (`python -m pytest -m "not slow" -q`) | same selection as the fast row | | | | | | |

No failures, so no tracebacks. Skip reasons: 64 EBU Tech 3341/3342 vectors absent from `fixtures/ebu/`
(test_meter.py:267), 1 EBU loudness set absent (test_feature_levels.py:294), 9 private premasters absent
(test_feature_real.py:61/75/84/93/101 via fixtures/premasters/manifest.toml), 1 `work/renders` absent
(test_planner.py:449). Warnings: 28 librosa `UserWarning: Trying to estimate tuning from empty frequency set`
(librosa/core/pitch.py:105, raised on synthetic fixtures with no tonal content; harmless but noisy, could be filtered
in `features/pitch.py` with `warnings.catch_warnings`), 2 `ResourceWarning: unclosed file ... features.npz`
(test_cli.py:120 via cli.py:244, and cli.py:339 in test_analysis_reuse; F24). No DeprecationWarning, no
RuntimeWarning (divide/invalid are wrapped in `np.errstate` throughout).
`python -W error -c "import eluate, eluate.cli, eluate.app_engine, eluate.planner, eluate.solver"`: clean, 4.2 s.

### Slowest 20, fast tier (contended)

| s | phase | test |
|---|---|---|
| 60.89 | setup | test_feature_analyse.py::test_common_grid |
| 50.19 | call | test_stage_limiter.py::test_true_peak_gate[1.0-0-qp-48000] |
| 46.19 | setup | test_planner.py::test_every_section_filled |
| 33.67 | call | test_profiles.py::test_live_plan_switches_off_and_scales_down |
| 31.50 | call | test_gain_staging.py::test_chain_rotation_keeps_the_input_gain |
| 26.45 | call | test_parallel.py::test_true_peak_envelopes_bit_identical |
| 25.81 | call | test_planner.py::test_inputeq_tilt_error_corrected_on_pink_noise |
| 23.34 | call | test_stage_render.py::test_save_load_with_npz_sidecar |
| 22.26 | setup | test_chunk4.py::test_auto_kick_detective_by_character[slow-True] |
| 21.14 | call | test_parallel.py::test_limit_blocked_thread_sweep_partial_block[7] |
| 21.14 | call | test_feature_structure.py::test_groove_switch_is_a_boundary |
| 20.75 | call | test_parallel.py::test_limit_blocked_thread_sweep_partial_block[2] |
| 18.79 | call | test_solver.py::test_blocked_limiter_matches_whole_track[clip_slow-48000] |
| 18.71 | call | test_solver.py::test_blocked_limiter_matches_whole_track[clip-48000] |
| 18.62 | call | test_evaluate.py::test_report_writes_json_md_plots |
| 18.12 | call | test_solver.py::test_blocked_limiter_matches_whole_track[clip_slow-44100] |
| 17.81 | call | test_parallel.py::test_limit_blocked_thread_sweep_partial_block[16] |
| 17.02 | call | test_solver.py::test_blocked_limiter_matches_whole_track[plain-44100] |
| 16.97 | call | test_solver.py::test_blocked_limiter_matches_whole_track[plain-48000] |
| 16.94 | call | test_solver.py::test_blocked_limiter_matches_whole_track[clip-44100] |

Slow tier top 10 (contended): test_cli::test_master_and_report 350.7; test_average_peak::
test_average_reading_reduces_the_crest_more_than_the_max_reading 347.6; test_app_engine::
test_master_after_analyse_reuses_it 343.6; test_cli::test_analyse_plan_render 164.6; test_average_peak::
test_clipper_outlier_spike_is_capped 153.9; test_parallel::test_master_bit_identical_at_any_thread_count 146.4;
test_chunk4::test_planner_clipper_record_reads_the_output_peak 145.0; test_average_peak::
test_split_clipper_threshold_ignores_the_sub 139.1; test_gain_staging::test_output_independent_of_premaster_level
(setup) 132.3; test_average_peak::test_clipper_spike_with_the_sample_cap_keeps_the_average 120.4. Full list in
`slow.log`.

## Ruff summary

`ruff 0.17.4 check --isolated --select E,F,W,B,UP,SIM,NPY` (no ruff config exists in the repo; E501 at the default
88 columns). Full listing: `ruff_full.txt`.

| Rule | src | tests | tools + requirements | Note |
|---|---|---|---|---|
| E501 line-too-long | 1345 | 719 | 97 | the code is written to ~100 cols: 187 in src at 100 |
| B905 zip without strict | 38 | 14 | 0 | |
| B023 loop variable in closure | 21 | 1 | 0 | latent, F20 |
| F821 undefined name | 12 | 0 | 0 | all the `del lr` closures, F20 (not live bugs) |
| UP037 quoted annotation | 10 | 0 | 0 | |
| E702 semicolon statements | 8 | 1 | 0 | |
| F841 unused variable | 4 | 3 | 0 | F11 |
| B007 unused loop variable | 4 | 2 | 0 | |
| SIM300 yoda | 4 | 2 | 0 | |
| F401 unused import | 3 | 2 | 0 | F11 |
| E731 lambda assignment | 2 | 3 | 0 | |
| E741 ambiguous name | 0 | 3 | 0 | |
| SIM117 / SIM108 / SIM102 / SIM105 | 3 | 4 | 0 | |
| UP035 / UP028 | 2 | 0 | 1 | io.py:30 `typing.Callable` |
| NPY (numpy deprecations) | 0 | 0 | 0 | clean |
| W (whitespace) | 0 | 0 | 0 | clean |

Ruff's own default rule set (0.17) reports 198 in src beyond the selection above, mostly C408, RUF007, RUF100
(9 stale `noqa`), FURB167. vulture (60 %): 29 hits in src, triaged in F11.

## Robustness matrix

`eluate master <input> -o <dir> --threads 2` unless noted. Times are wall times under contention.
"Left behind" = what the output directory holds after a failure.

| Input / flags | Exit | Time | Message / result | Left behind |
|---|---|---|---|---|
| 10 s stereo 24-bit 44.1 kHz (baseline) | 0 | 192 s | master done, peak 491 MB (estimate 1069) | 22 files, below |
| mono 16-bit | 0 | 702 s | stereo (dual-mono) 24-bit master | full set |
| 8-bit unsigned (PCM_U8) | 0 | 578 s | 24-bit master | full set |
| 32-bit float | not run as a master | - | float32 reader path exercised by the NaN/Inf cases | - |
| 64-bit float | 0 | 530 s | 24-bit master | full set |
| 3 channels / 6 channels | 2 | 17 s | "stereo only: got 3 channels" / "6 channels" | empty dir |
| 0 frames | 2 | 11 s | "need at least 2 samples" | empty dir |
| 1 frame | 2 | 10 s | "need at least 2 samples" | empty dir |
| 2 frames (`analyse`) | 2 | 41 s | scipy "length of the input vector x must be greater than padlen, which is 27" (F6) | - |
| 100 frames | 2 | 23 s | same scipy padlen message + 4 librosa UserWarnings (F6) | empty dir |
| 1000 frames (`analyse`) | 2 | 44 s | "window is longer than input signal" (F6) | - |
| 4410 frames = 0.1 s (`analyse`) | 0 | 39 s | analyses; 0.5 s gets input gain +0.00 (normalisation skipped) | cache |
| float32 with one NaN / one Inf | 2 | 12 / 16 s | "<path>: audio contains NaN or Inf" | empty dir |
| missing input file | 2 | 15 s | "Error opening '...': System error." (F22) | nothing |
| missing input into a finished out dir | 2 | ~15 s | same; previous run.json + process-log.* deleted (F4) | old run minus marker |
| non-ASCII + spaces in input and output paths (`dir with spaces é ü 日本/tråck ñ.wav` -> `o ü 日本 out`) | 0 | 409 s | full master | full set |
| output dir holding a previous run, rerun with `--streaming` | 0 | 480 s | every product replaced (16-bit master), analysis cache reused; foreign files and an old `pass-1/` left untouched | merged dir |
| input == `out/master.wav` | 0 | 551 s | input overwritten by the master (F3) | input lost |
| `out/master.wav` read-only | 2 | 387 s | "[WinError 5] Access is denied: '...master.<id>.tmp.wav' -> '...master.wav'" after 3 renders (F5) | new plan files + old master/report, no run.json |
| output under `C:/Windows/System32` | 2 | 20 s | "[WinError 5] Access is denied: 'C:\Windows\System32\eluate_audit_out'" | nothing |
| 22.05 kHz | 2 | 32 s | "Each lower bound must be strictly less than each upper bound." (F7) | analysis cache |
| 192 kHz 24-bit | 0 | 865 s | 24-bit 192 kHz master, peak 720 MB (estimate 1336) | full set |
| FLAC 24-bit | 0 | 328 s | WAV master | full set |
| AIFF 24-bit | 0 | 664 s | WAV master | full set |
| 10 s digital silence | 2 | 22 s | "input measured_dbfs contains NaN or Inf" + 2 librosa warnings (F8) | analysis cache |
| 20 min 44.1 kHz digital silence | 2 | 1526 s | memory guard passes (estimate 10 516 MiB vs 13 366 free, 2 threads), analysis 1495 s, then the F8 message | analysis cache |
| `--preserve --tilt 2` | 0 | 239 s | tilt silently ignored; report prints target -2.50 / error -3.49 (F2) | full set |
| `--stems one.wav` | 0 | 1025 s | accepted; full 3-pass stems flow (F18) | full set + pass-1 |
| input and `--stems` | 2 | 17 s | "give one input, or --stems" | nothing |
| `--mix` without `--stems` | 2 | 19 s | "--mix goes with --stems" | nothing |
| `--bpm 0` | 2 | 18 s | "bpm must be positive" | nothing |
| `--target-rms 5` | 2 | 14 s | "target RMS must be finite and below 0 dBFS" | nothing |
| `--tilt 5` | 2 | 16 s | "tilt must lie in [-3.0, 3.0]" | nothing |
| `--threads 0` | 2 | 16 s | "threads must be at least 1" | nothing |
| `--xover 6400,1600,400,100` | 2 | 25 s | "crossovers must be strictly ascending" | empty dir |
| `--xover 100,400,1600,30000` | 2 | 26 s | "crossovers must lie in (0, fs/2)" | empty dir |
| `--xover 100,abc` | 2 | 18 s | argparse usage + "invalid _xover value" | nothing |
| `--subside 20` | 0 | 538 s | clipped to 60 Hz silently; run.json says 20 (F18) | full set |
| `--crest-mode floor` (no floors) | 2 | 17 s | "crest_mode 'floor' needs crest_floors_db (no calibrated default)" (F22) | nothing |

Baseline end-to-end (`stereo10s.wav`, 10 s 44.1 kHz 24-bit, `--threads 2`): rc 0, 192 s wall (analyse 37, plan 16,
solve 125 over 3 renders, report 14), peak 491 MB, run.json `ok: false`, misses rms (-7.39 vs -4.1, limiter budget)
and slope. Outputs (bytes): ab/difference.wav 3 528 088, ab/master_matched.wav 3 528 088, ab/original_matched.wav
3 528 088, analysis.json 49 290, features.npz 398 409, gr.png 346 114, limiter.json 549, limiter_gr.npz 3 698,
loudness.png 26 669, ltas.png 65 418, master.wav 2 646 044 (24-bit), plan.json 23 687, plan.npz 1 015 048,
plan_record.json 27 027, process-log.html 12 932, process-log.md 10 551, report.json 135 056, report.md 9 665,
run.json 5 841, solve.json 18 385, source.json 613, targets.json 1 998 (22 files, 13.6 MB). All text files UTF-8,
LF only, no NaN tokens.

## Profile: 60 s 48 kHz synthetic electronic loop, `--threads 2`

`python -m cProfile -o prof.out -m eluate.cli master loop60_48k.wav -o o_prof --threads 2`: rc 0, 2306 s wall
under contention and profiler overhead (run.json: analyse 212, plan 32, solve 1901, report 131 s; 3 renders).
cProfile sees only the main thread, so work done in `parallel.pmap` worker threads shows up as
`futures.result`/`threading.wait` (1605 s) rather than under its own frames; a matplotlib mathtext chain that cProfile
lists at the top with 2287 s cumulative is a recursion/thread attribution artefact (`740/1` primitive calls, it
cannot exceed the 131 s report phase) and is left out.

| # | cumulative s | calls | function |
|---|---|---|---|
| 1 | 1900.9 | 1 | solver.py:1422 solve |
| 2 | 1830.2 | 3 | solver.py:1469 render (the solver's renders) |
| 3 | 1156.4 | 515 | parallel.py:75 pmap |
| 4 | 1126.3 | 3 | solver.py:966 pre_limiter |
| 5 | 992.4 | 120877 | scipy.fft dispatch (`__ua_function__`) |
| 6 | 946.7 | 36 | solver.py:271 limit_blocked |
| 7 | 527.3 | 3 | planner.py:1975 solve_clipper -> solver.py:705 clip_threshold_for_average_peak |
| 8 | 490.7 | 42 | solver.py:743 reading (inside the clipper bisection) |
| 9 | 431.4 | 8591 | stages/resample.py:78 upsample |
| 10 | 424.2 | 3 | solver.py:678 _bisect_down |
| 11 | 350.6 | 792 | solver.py:331 front |
| 12 | 285.8 | 31508 | oa.py:47 oa_stream |
| 13 | 257.7 | 11 | stages/limiter.py:105 true_peak_fft_envelope |
| 14 | 213.7 | 3 | solver.py:1144 measure |
| 15 | 210.8 | 1 | cli.py:175 do_analyse (features.analyse 209.1) |

Top self time: pocketfft c2r 273.9 s, ufunc.reduce 190.5, pocketfft r2c 156.2, envelope.py:159 release_hull 80.7,
ndarray.copy 53.0, scipy sosfilt 52.2, limiter.py:126 `one` 48.6, resample.py:100 put 44.0, oa_stream 42.6.
Memory: run.json `peak_memory_mb` (Windows PeakWorkingSetSize) 920 MB against `memory.estimate_mb(2 880 000, 2)` =
1508 MiB (estimate 64 % above the measured peak; conservative, as designed). 10 s / 44.1 kHz: 491 MB vs 1069;
10 s / 192 kHz: 720 MB vs 1336. tracemalloc was not used (it would miss numpy/FFT native buffers);
PeakWorkingSetSize is the better measure here.

## Verified correct

- `python -W error -c "import eluate, eluate.cli, eluate.app_engine, eluate.planner, eluate.solver"` imports cleanly
  (4.2 s); `eluate.__version__` = 0.1.0b0 = pyproject = `_FALLBACK_VERSION`; desktop package.json, tauri.conf.json,
  Cargo.toml all 0.1.0-beta, tag `v0.1.0-beta` exists; release notes' figures (-4.1 RMS, -0.3 / -1 dBTP, 24 / 16 bit,
  up to two stem correction passes) match code. Entry point `eluate = "eluate.cli:main"` works.
- `uv pip check`: all 45 installed packages compatible; the venv equals `requirements/lock-*-win-x64-py312.txt`
  (numpy 2.5.4, scipy 1.18.1, librosa 1.0.0, numba 0.68.0, soundfile 0.14.0) within the pyproject bounds.
- `io.read` rejects NaN and Inf (float32 WAV with one NaN / one Inf -> "audio contains NaN or Inf", exit 2, nothing
  written), 3 and 6 channels ("stereo only: got N channels", exit 2), 0 and 1 frame ("need at least 2 samples").
- Control validation is good and early (before any file is touched): `--bpm 0`, `--target-rms 5`, `--tilt 5`,
  `--threads 0`, `--xover` descending ("strictly ascending"), `--xover` above Nyquist ("must lie in (0, fs/2)"),
  `--xover 100,abc` (argparse usage), input plus `--stems`, `--mix` without `--stems`, `--crest-mode floor`
  without floors - all exit 2 with one line.
- Mono, 8-bit unsigned, 64-bit float, FLAC, AIFF, 192 kHz and non-ASCII / space-containing input and output
  paths (see matrix) master to completion as 24-bit stereo at the input rate (22.05 kHz does not: F7; 32-bit float
  not run as a master, the f32 reader path is the one the NaN/Inf cases exercise); text outputs are UTF-8 + LF, JSON has no NaN tokens; writes are
  atomic (temporary sibling + os.replace) and a rerun into a finished directory replaces every product (stale
  non-eluate files and an old `pass-1/` are left alone, which is acceptable).
- The retired-function list in docs/gain-staging.md 14.2 is gone from code and tests (F11).
- No NPY (numpy-deprecation) ruff hits; no CRLF in runtime outputs; `make_lock.py` writes LF and evaluates markers
  per interpreter; public_snapshot.py extracts `git archive` with `filter="data"` and refuses an OUT_DIR inside the repo.
- Memory guard: estimate for a 60 s 48 kHz master at 2 threads = 1508 MiB (MIB_PER_MFRAME 180 x 2.88 + 240 + 750);
  for 20 min 44.1 kHz = 10 516 MiB (2 threads) / 9 766 (1 thread) against 13 366 MiB free -> it proceeds with 2
  threads (10.5 GB < 0.85 x 13.4 GB).

## Open questions

1. Should `--preserve` reject `--tilt` (and `--subside` outside 60-150 Hz) or warn? (F2, F18)
2. Is the analysis/report crest pair (features/crest.py vs multiband.band_crest) meant to keep its different interior
   fallback, or is that drift from audit 06 M13 applied to one copy only? (F12)
3. Mono input produces a stereo (dual-mono) master. Intended? (No option for mono output.)
4. Should `eluate report` without targets.json measure on `bands.DEFAULT_CROSSOVERS` (evaluate.py:537) when the master
   was made on key-locked crossovers? It silently labels different bands than the original report.
5. Spill-dir sweeping (`solver.sweep_spill`, system temp, 6 h age) runs at every `master`; with several engines on one
   machine (batch + CLI), is the 6 h mtime rule enough for a 20 min / 192 kHz render that writes nothing for hours?
