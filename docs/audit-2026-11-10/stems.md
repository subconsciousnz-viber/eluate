# Eluate stems audit (2026-11-10, branch audit/2026-11-10)

Scope: `src/eluate/stems.py`, `stemfix.py`, `cli.py` (`sum_stems`, `do_master_stems`), `app_engine.py`
(`stems-sum`, `master --stems`), `uidata.py`, `ui/app.js`, `desktop/src-tauri/src/lib.rs` (stems parts),
against `docs/noise-and-stems.md` §2 (the maintainer's definition, phases A/B/C), `docs/gain-staging.md`
§3/§16–20 and `planner.PROFILES["stems"]`. Read-only; nothing in the repo was changed.

Evidence lives in `<scratch>/audit/stems/`:
`mkstems.py` (the synthetic 6-stem set: kick, sub, perc, pad, lead, vox; 48 kHz stereo, 30 s, 128 BPM,
bars 0–3 intro without kick/sub, bars 8–15 with lead; `set/` holds the WAVs and the exact-sum `mix.wav`),
`exp1_null.py` (null test / read_stems), `exp2_attr.py` (attribution), `exp3_comp.py` (stem compressor),
`exp4_roles.py` (roles, onsets, duck), `exp5_eq.py` (stem EQ), `run1/` + `run1.log` (the timed
`master --stems … --threads 2` run), `run_dup/` + `run_dup.log` (duplicate basenames), `run_one/` +
`run_one.log` (one stem with itself as `--mix`). Baseline: `tests/test_stems.py` + `tests/test_stemfix.py`
fast tier, 32 passed.

Severity: S1 audible / wrong audio; S2 wrong numbers or reports; S3 crash or misbehaviour on plausible
input; S4 hygiene / performance; S5 nit. Confidence is in how sure I am the finding is real and the fix
right, not in how often it bites.

---

## Findings

### S1

**1. A bass-carrying "music" stem is ducked full-band under the drums.** `stemfix.py:216-248`
(`roles`), `:563-576` (the duck decision). `roles()` makes any stem whose energy is at least 40 % below
150 Hz and within 20 dB of the loudest low band a kick/bass candidate; in a 3-stem delivery (drums /
music / vocals, common for a "stem master") the music stem (sub + pads + lead) qualifies as the bass by
content, and a limiter entry naming it at kick moments plans `music: ducked under drums by up to 2.0 dB`
(pass 2: 3 dB). The duck is full-band (by design, docstring l.92), so pads and lead dip with the sub.
Evidence: in-process `stemfix.plan` on {drums, music, vox} from `mkstems`; the gain measured on the music
stem 10–50 ms after a kick is −1.90 dB on the pad and lead content too; `roles()` returns
`('kick', 'music')` for {kick, sub+pad+lead}. Fix (minimal): for a stem without a `BASS_HINTS` token
require a higher low share (0.75) *and* its energy above `ROLE_LOW_HZ` at least ~10 dB below its low band
before it may take the bass role; or, when the bass candidate carries substantial content above 150 Hz,
duck only its low band (the docstring rejects a band split because the stem nulls to the mix; the null
test is on the *input*, a crossover on a corrected stem does not break it). Confidence: high that it
triggers; medium on audibility (2–3 dB, ~100 ms, weighted by the overlap `w`).

**2. A bass that was already sidechained is ducked again (8 dB under each kick).** `stemfix.py:281-311`
(`duck_gain_db`). The depth weight `w` (l.296-297) compares the bass's low-band peak *over the kick's hold*
with its own 95th-percentile loud level: a bass already dipping 6 dB under the kick is still "within
12 dB of its loud level", so `w ≈ 0.75` and it is ducked a further 2.2 dB (3 dB cap). Nothing compares the
bass under the kick with the bass just *before* the kick. Evidence (`exp4_roles.py`): the synthetic sub
with a 6 dB / 100 ms sidechain applied → `duck_gain_db` dips −2.23 dB → 8.2 dB total under the kick
(the un-sidechained sub: −3.00 dB). Fix: per onset, measure the bass's held low-band level over
[i − 60 ms, i − 5 ms] and over the hold; if it already falls by ≥ `DUCK_STEP_DB` skip the dip (or reduce
the depth by the existing dip). Confidence: high.

**3. A notch the planner flagged *musical* with a 0.0 dB cap is cut 1.5 dB, then 3 dB, on the stem.**
`stemfix.py:589`: `cap = (fix["params"].get("max_cut_db") or NOTCH_MAX_DB)` — a cap of `0.0` is falsy
and reads as "no cap", so `_notch_cap`'s correct 0.0 (l.473-479, "the planner itself held back") is
replaced by 3 dB. The planner's `max_cut_db: 0.0` means "a chord tone, do not cut"; the stem gets a
Q 16 static notch on it anyway. Evidence: run1 pass 1 log, `perc: notch -1.5 dB at 10.4 kHz, Q 16 (notch
at 10396 Hz (found))` where report.json has that notch `flag: musical, max_cut_db: 0.0, rendered: False`;
in-process: `plan` → −1.5, a second pass → −3.0 (expected 0.0). Fix: `cap = p["max_cut_db"] if
p.get("max_cut_db") is not None else NOTCH_MAX_DB`, and skip the fix when the cap is 0. Confidence: high
(audibility depends on the tone: a hat peak at 10 kHz is subtle, a chord tone at 400 Hz is a note thinned).

### S2

**4. Null test: a polarity-inverted mix is reported as "These stems do not make this mix" with a junk
offset.** `stems.py:129` (`np.argmax(c[sel])` on the signed correlation), `:146` (the gain is signed and
would handle it, but at the wrong lag). Evidence (`exp1_null.py`): `mix = -sum` → verdict `mismatch`,
residual −1.4 dB, `offset_samples = −448`, gain −5.6 dB. A flipped mix print is a plausible export
state and trivially fixable. Fix: `np.argmax(np.abs(c[sel]))`, keep `g` signed, say "polarity inverted"
when `g < 0`. Confidence: high.

**5. The "processed" / "mismatch" advice is backwards for the common cases, and the pass line omits the
gain.** `stems.py:185-194`. Measured verdicts (`exp1_null.py`): mix through a 3 dB limiter (+3 dB) →
processed −25.4 dB; through a 1 dB soft clip → processed −37.9 dB; a 0.5-sample fractional delay →
processed −21.5 dB (worst band 16 kHz, offset 0); a 0.25-sample delay → −27.2 dB; the mix 2 s longer
with a −34 dBFS decaying tail → processed −39.4 dB; the mix 50 ms shorter → processed −38.6 dB. In every
one of these the message says "The mix has bus processing the stems lack, or a stem is missing; print
the stems without mix-bus processing" — the stems are fine, it is the *mix* that differs, and the
remedy is to omit `--mix` or print the mix without the limiter. A mix 6 dB louder than the sum passes
with "residual −inf dB" and nothing about the 6 dB (`gain_db` is in stems-report.json only). A missing
stem at a normal level lands in "mismatch" (−14 dB pad, −15.7 dB lead), not in "processed" as the
docstring (l.15-16) and message claim. Fix: diagnose before advising: (a) `|gain_db| > 0.5` → "the mix
is X dB louder/quieter than the stems' sum"; (b) `gain_db > 1` and residual concentrated at the sum's
peaks (within ±5 ms of the top-1 % peaks vs elsewhere) → "the mix is limited or clipped; the stems are
mastered as given"; (c) lengths differ → state both lengths and the tail's level; (d) worst band
≥ 8 kHz with offset 0 → "a sub-sample offset; the stems are aligned to one sample". Print via `_f` so
−inf reads "below −150 dB". Confidence: high.

**6. `_range_rows` reads each stem's level on a mono fold: a side-only stem vanishes from the EQ
ranking, a wide one reads 10 dB low.** `stems.py:254-255` (`_range_level` uses `x.mean(axis=1)`),
`:272` (a zero mono level drops the row). The *share* (per-channel `csd`, correct) and the level
disagree. Evidence (`exp2_attr.py`): a pure-S stem (L = −R) is MISSING from the rows; a mostly-S stem
(R = −0.8 L) reports `level_re_sum −10.4 dB` where the per-channel truth is −0.5 dB (share +0.89). The
report prints a wrong level and stemfix can skip the stem. Fix: `signal.welch(x, fs, axis=0)` and
`p.sum(axis=1)` as the share already does. Confidence: high.

**7. Notch attribution ignores the notch's channel.** `stems.py:400` (`seg.mean(axis=1)`), although
`detail["channel"]` (M/S) is carried and stemfix (l.548) applies the fix on that channel. A side-only
resonance cancels in the fold. Evidence (`exp2_attr.py`): a 660 Hz ring added to the pad as L+ring /
R−ring reads +7.9 dB on the fold (the pad's own E5 harmonic, not the ring) and +28.2 dB on S. Fix: take
M = (L+R)/2 or S = (L−R)/2 per `nt["channel"]`, L+R only when unset. Confidence: high.

**8. Limiter / clipper moments are attributed by each stem's own true peak, not by its contribution to
the sum's peak.** `stems.py:380-386` (`_peak_at(s, …, tp=True)` per stem, ranked by `_rank`). A stem
in anti-phase at the peaks ranks high and gets compressed, which *raises* the sum's peak. Evidence
(`exp2_attr.py`): three stems with 1 kHz bursts, carrier +0.5, in-phase +0.3, anti-phase −0.45: the
anti-phase stem ranks second (−6.9 dBFS) above the in-phase one (−10.5) although removing it would
raise the sum's peak from −9.1 to −6.0 dBFS. Fix: rank by leave-one-out peak drop,
`peak(sum) − peak(sum − stem)` over the moment windows (one true-peak call per stem, same cost), or by
`Re<stem, sum>` over the windows. Confidence: high on the metric, medium on how often real stems are
anti-phase at limiter hits (rare, but the duck/compress fix then acts the wrong way).

**9. A limiter held by its *average* budget is not a "limit": the loop stops with "no limit" while the
master is short of the target.** `stems.py:376` (only `bound == "outlier"` is attributed),
`cli.py:664-665` (a pass with no limit ends the loop), `is_limit` l.447. The stems profile's limiter
average budget is 2 dB (`planner.py:544`), and gain-staging §18 shows the average budget is the usual
binding reading on the maintainer's tracks. Evidence (run1): pass 0's `report.json` warnings are
`limiter {bound: "average", needed 5.27 dB, cap 2.0, outlier 5.76 / cap 4.0, shortfall 3.27}` and
`loudness {shortfall 3.27, cause: "limiter budget"}`; the attribution has no limiter entry (12 limits:
6 EQ bounds + 6 found notches), so the 3.27 dB that dominates the score (9.27) is never attributed to a
stem; had the EQ bounds not been hit the loop would have stopped "no limit" 3.3 dB short. Fix: attribute
a `limiter` warning with `bound == "average"` (and the `loudness` shortfall) as a limit, ranking stems by
their projection share of the sum over the loud section's hits (`_moments` already reads
`limiter_gr.npz`; use the median-GR hits rather than the largest). Confidence: high on the logic.

**10. Notches the planner rejected as "suspect" or "musical" count as limits and are fixed on the
stem.** `stems.py:395-411` (every notch in `rep["notches"]` becomes an entry; the planner's `reason`,
`flag`, `max_cut_db` are not read for the verdict), `:447-450` (`is_limit`: any "(found)" notch),
`stemfix.py:542-551`. In run1 all six unrendered notches carry `reason: "suspect: spans cover 63…91 % of
the track"` (a sustained HF peak is the hat's timbre, which is why the planner did not render them) and
one is `flag: musical, max_cut_db: 0.0`. The loop counted them as 6 of pass 0's 12 limits (3.0 of the
9.27 score), stemfix put six static Q 8–16 notches at 9–16 kHz on the perc, and pass 1's sum then had
four *new* found notches (7634, 14663, 15894, 18106 Hz) — whack-a-mole on a hat. Fix: `is_limit` false
(and no stem fix) for a notch whose `reason` starts with "suspect", whose `flag` is "musical", or whose
`max_cut_db` is 0; carry `reason` into the entry's detail. Confidence: high.

**11. When the planner wanted *less* than the applied bound, the stem gets the blind +1.5 dB step
anyway.** `stemfix.py:531-536`: the sized branch runs only when `(wanted − g) · sign > 0`; otherwise
`step = sign × EQ_STEP_DB`. Evidence (run1): the air shelf sat at +2.5 (its bound) with
`wanted_db 0.42` — the planner's own range deficit says 0.4 dB of air was wanted — and pass 1 added
`perc: high shelf +1.5 dB at 10 kHz`; reproduced in-process. The fallback belongs to the case where
`wanted_db` is *unknown* (low shelf, bell, band shelf), not to a known excess ≤ 0. Fix: when
`wanted is not None` and the excess is ≤ 0, propose nothing (the shelf is at its bound for the slope
target, not for the range; a stem fix cannot help). Confidence: high.

**12. The pass score is blind to what a correction pass can break.** `stems.py:454-460`,
`cli.py:699-702`. Score = RMS shortfall + 0.5 per limit. A pass that compresses the vocal 2 dB
(`exp3_comp.py`: 85 % of its frames in gain reduction, 0.9 dB/ms slews, 60 ms release = a level change
with pumping between syllables), ducks the bass, or shelves a stem's S 3 dB can clear one limit (−0.5)
and be kept while the crest, section contrast, tonal balance or stereo image got worse — none enter the
score; the crest/section row is informational. Evidence (run1): pass 1's 13 fixes raised the sum's
crest from 10.5 to 13.2 dB and the master came out 2.9 dB *quieter* (RMS 6.15 dB short vs 3.27) — here
the RMS term caught it (score 10.65 > 9.27, pass 0 kept), but a pass that trades 0.3 dB of RMS for one
EQ bound (−0.5) is kept with +2 dB of crest and nothing said. Fix (minimal): add penalties the report
already computes — crest change vs pass 0, the section-contrast flag, the slope miss, a stereo width
change — or at least print those deltas per pass in stems-report.md. Confidence: high that the blind
spot exists; the weights are a listening question.

**13. The stem compressor's detector has no hold: sustained sub bass gets gain ripple at 2·f0.**
`stemfix.py:317` (`_frame_peaks` bare), while `_held` (l.186-190) exists for exactly this and is used by
the duck and roles. Evidence (`exp3_comp.py`): a 41.2 Hz sine at 2 dB GR → gain ripple 0.25 dB,
harmonics 2–5 rise from −124 dB to −42.8 dB re the fundamental; 60 Hz → −48.8 dB; 110 Hz → −62 dB.
With `pk = _held(pk)` in `_comp_gain_db`: ripple 0.000 dB, harmonics −124 dB. The artefact is bounded
(the hull's 60 ms release limits the ripple to ≤ 0.25 dB at 41 Hz) but 80 dB worse than the chain's
own stages, which hold. Fix: one line, `pk = _held(pk)` after `_frame_peaks` in `_comp_gain_db` (the
backward lookahead hull still pre-empts the attack). Confidence: high.

### S3

**14. `write_report` crashes on an exact null, after the whole master has run.** `stems.py:491`
(`rep = evaluate._clean(rep)` turns `residual_db = −inf` into `None`), `:496` → `null_message` `:187`
formats `None` with `:.1f` → `TypeError`, outside `cli.main`'s `except (ValueError, OSError,
SoundFileError)` (`cli.py:786`). Triggers whenever `mix − g·sum` is bit-exact: `--stems a.wav --mix
a.wav` (one stem; the CLI accepts one), a set where only one stem is non-silent, a 32-bit-float mix the
DAW summed identically. Evidence: `run_one.log` — `master --stems perc.wav --mix perc.wav` runs the
sum pass (12 s) and a correction pass, prints "Kept pass 0", then the traceback above; exit 1, no
`run.json` (state "none" in the app), `stems-report.json` written, `stems-report.md` not. Fix: build
the null line before `_clean`, or make `null_message` use `_f`. Confidence: high.

**15. A "processed" verdict aborts the master, for mixes that are commonly "processed".**
`cli.py:525-526` (`raise ValueError` unless `pass`), and the app's `stems-sum` job fails the same way.
The verdicts in finding 5 (a limited reference mix, a sub-sample offset, a longer tail) all stop the
run; the user must re-pick the set without the mix. The mix is only a check and the stems are what
gets mastered. Fix: treat "processed" as a warning (log, stems-report, the app's track row) and abort
only on "mismatch" / "silent"; or add `--mix-check strict|warn`. The maintainer's rule says "check
first", not "refuse". Confidence: high (behaviour); the policy is the maintainer's call.

**16. The ±10 ms offset search silently returns a wrong lag and "mismatch" for larger offsets.**
`stems.py:62` (`NULL_MAX_OFFSET_MS = 10`), `:129`. Linear-phase EQ or lookahead limiting on the mix bus
puts 20–100 ms between a mix print and stems printed without it (or with PDC differences). Evidence
(`exp1_null.py`): a 12 ms-delayed mix → `mismatch`, residual −1.3 dB, offset 443 (a random peak inside
the window); a 10 ms delay passes. The cross-correlation is an FFT over 30 s, so a ±250 ms window costs
nothing. Fix: widen the window; when the peak sits at the edge or the best residual is near 0 dB, say
"no alignment found within ±X ms". Confidence: high.

**17. Duplicate stem names (or the same file twice) are accepted and corrupt the correction pass.**
`cli.py:651` (`names = [Path(p).stem …]`), `stems.py:read_stems` (no check), `stemfix.py:401, 409, 426`
(fixes are matched by name; `names.index(kick)` takes the first). Evidence: `run_dup.log` —
`--stems kick.wav dup/kick.wav sub.wav` (two different files, the same basename) runs to exit 0 in
61 s; the log lists one set of "kick" fixes (`kick: low shelf -1.5 dB …`, six of them) that `apply`
puts on *both* kick files, and the attribution rows cannot be told apart. `read_stems([base, base])`
(the same file twice) is accepted and sums it +6 dB silently (`exp1_null.py`). Fix: reject duplicate
resolved paths and duplicate stems in `read_stems` (or disambiguate names with the parent folder).
Confidence: high.

**18. The memory guard does not know about the stems held in memory.** `cli.py:366-367`
(`memory.guard(frames, …)` on the *sum's* frames only, inside every `do_master`), `:653` (`st` held
through the whole loop), `:688` (`fx.apply` makes a second full set), `stems.py:89, 115` (`read_stems`
holds the float64 list and the `np.stack` copy at once), `stems.py:234-246` (`bands.split` of a
full-length stem = 5 bands). For 8 stems × 6 min × 48 kHz (17.3 M frames): one stem 264 MiB, the set
2.1 GiB; `read_stems` peaks at ~4.2 GiB (list + stack); during each pass's master: set 2.1 + sum 0.26 +
the chain's own 4.1 GiB (`memory.estimate_mb`, 2 threads) ≈ 6.5 GiB against a guard estimate of
4.1 GiB; after the master: set 2.1 + `fx.apply` copy 2.1 + band split 1.3 + `to_ms` 0.26 ≈ 5.8 GiB. A
machine with 8 GB free passes the guard and hits `MemoryError` in pass 0's master (uncaught there; the
loop catches it only from pass 1 on). Measured (run1, 6 stems × 30 s, 2 threads): peak working set
782 MB (pass 0) / 816 MB (pass 1) against the guard's 1249 MB estimate; the set adds 138 MB (+138 MB
for the `apply` copy), absorbed here by the estimate's fixed terms (BASE 240 + THREADED 750 MiB), which
cannot absorb 2.1 GiB at 6 min. Fix: `guard(frames, threads, extra_mib = k·n·16/2^20 (×2 while `apply`
is used))` from `do_master_stems`; preallocate `(k, n, 2)` in `read_stems`; attribute from the
one-stem-at-a-time path instead of `fx.apply` (finding 20). Confidence: high on the arithmetic.

### S4

**19. `attribute` is slow even without band splits, and the band-split path multiplies it.**
`stems.py:260-279` (`_range_rows`: one `welch` of the total plus a `welch` and a `csd` per stem, 8192
points, repeated for *each* of the 6 sections), `:395-411` (a PSD per stem per notch), `:234-246`
(`_band_moments_rank`: `bands.split` of `total` and of every stem per (band, channel) entry — with E
entries, E × (k + 1) full-length five-band zero-phase splits, 1.3 GiB of output each at 6 min).
Measured (run1): `attribute()` 29.1 s for pass 0 and 22.1 s for pass 1 on 6 × 30 s with *no*
multiband / speclim / limiter entries at all; `plan` 0.1 s, `apply_sum` 1.4 s, `apply` 1.5 s. Scaled to
8 stems × 6 min that is ~7 min of stems-specific work per pass before any band split, on top of each
pass's full master. Fix: one `csd(stem, total)` and one `welch(stem)` per stem, integrated over every
section's range (6× fewer FFTs); one PSD per stem over the union of notch spans; split each stem once
(or only the bands the entries need), or read the band at the moment windows only (±50 ms around ≤ 8
moments: filter 0.8 s of audio, not 6 min). Confidence: high.

**20. Every fix is computed twice per pass and the corrected set is materialised.** `cli.py:682`
(`apply_sum`, one stem at a time, for the master) then `:688` (`apply`, the full set, for the
attribution). Fix: have `apply_sum` hand each corrected stem to a callback the attribution consumes,
or attribute per stem inside the same loop. Confidence: high.

**21. The promoted `plan.json` does not match the root's `source.json`.** `cli.py:597-609`. After a
kept correction pass the root holds pass-k's `plan.json` (solved on `pass-k/corrected-sum.wav`) and
pass 0's `source.json` (`stems-sum.wav`). `eluate report out/` is fine (it reads master.wav), but
`eluate render out/plan.json` renders that plan on the plain sum — a different master from
`master.wav` — and the process log's input section describes the plain sum. Fix: a `rendered_from`
field in run.json / source.json, or promote `corrected-sum.wav` as the plan's render input.
Confidence: high.

**22. The app never shows the null test or the stems report.** `ui/app.js:197` stores
`j.track.nullTest = ui.null` and nothing reads it (l.1354 only initialises it); `uidata.from_render`
(l.178) has no stems fields, so `stems-report.md`, the kept pass, `passes/` and the per-stem rows are
invisible in the UI except as lines in the job's log pane (200-line ring). Fix: a "Stems" block in
`from_render` (null verdict + residual + gain, kept pass, per-pass score / limits / fix lines, the
report's path for Reveal). Confidence: high.

**23. `--profile live` (or `electronic`) with `--stems` is overridden silently.** `cli.py:650`. Log one
line. Confidence: high.

**24. `read_stems` reads float32, copies to float64 per stem, then stacks once more.**
`stems.py:89, 115` (see 18). Fix: read the lengths with `sf.info` first, preallocate
`np.empty((k, n, 2))` and fill it. Confidence: high.

**25. Each correction pass re-analyses from scratch (64 s here) and the master itself dominates the
wall time.** `cli.py:685-686` (a fresh `do_master` on `corrected-sum.wav`, no cache). Run1: 38 min for
30 s of audio (pass 0 master 734 s, pass 1 master 1315 s). Not a stems bug — the solver's speed is the
engine's — but the stems flow pays it up to three times; a cheap pre-check (the proxy chain on the
corrected sum: crest and RMS estimate) could skip a pass like run1's pass 1 before its 22 minutes.
Confidence: high on the numbers.

### S5

**26.** The stem *bell* fix ignores the sum bell's Q (`stemfix.py:462-463`: always `fc/(hi − lo)` =
1.41 for the fixed-octave `SECTION_RANGES["bell"]`), and the band shelf is approximated by the same
bell; the sum's `q`, `lo_hz`, `hi_hz` are available in the section dict.

**27.** A `high_shelf_side` fix on a stem whose S is negligible changes nothing (`stemfix.py:365-378`;
measured max |y − x| = 0 on a mono stem) but counts as a step; the next pass then stops
"nothing_to_fix" or deepens it to the cap. Skip S-only fixes when the stem's S is > 30 dB below its M
(the planner's `EQ_SIDE_SILENT_DB` rule).

**28.** `is_limit` / `stemfix.plan` parse the limit *text* (`stems.py:447-451`, `stemfix.py:151-152`);
an edited label breaks the loop silently. Carry a structured `kind` in each entry (it would also carry
finding 10's `reason`).

**29.** The CLI accepts one stem (`--stems a.wav`), the app requires two (`lib.rs:588`). Harmless, but
it is the one-file path of finding 14.

**30.** A 4-channel stem fails with "stereo only: got 4 channels" (io.as_stereo) without the stem's
name; a missing stem reports soundfile's "Error opening '…': System error." rather than "not found".

**31.** The kick onset time carries the 150 Hz 4th-order lowpass's group delay (~2 ms at 50 Hz): the
duck is −1.96 dB at the true onset and −3 dB 2 ms later (`exp4_roles.py`); the 5 ms lookahead covers
it. A note, not a bug.

**32.** `null_test` with silent stems and a non-silent mix returns `offset_samples = −480` (the window's
edge) and "These stems do not make this mix"; say "the stems are silent".

**33.** The trim rule (`stems.py:70-71`, 100 ms / −60 dBFS) is enforced as written; a 50 ms tail at
−40 dBFS (a short reverb tail on one stem) is refused with a correct message. Maintainer rule; no
change proposed.

---

## Measured: the timed run (run1)

`eluate master --stems kick sub perc pad lead vox --mix mix.wav -o run1 --threads 2` on the synthetic
30 s / 48 kHz set, Windows 11, this machine (32 GB, ~18.6 GB free at the time). Exit 0.

- **Wall time 38 min 5 s** for 30 s of audio. Pass 0's master 734 s (analyse 64.2 s, plan 2.9 s, solve
  577.7 s for 2 renders, report 86.5 s); pass 1's master 1315 s (3 renders, a fresh analysis);
  stems-specific work (sum + null test, two attributions, plan, apply_sum, apply, report) ≈ 230 s, of
  which the two attributions are 51 s.
- **Null test**: pass, residual −149.0 dB (float32 files of a float64 sum), offset 0, gain 0 dB.
- **Pass 0**: RMS −7.37 dBFS vs the −4.1 target (3.27 dB short), crest 10.5 dB, limiter held by its
  *average* budget (needed 5.27 dB against 2.0; outlier 5.76 against 4.0), every input EQ section at its
  stems bound (low shelf −1.5 @ 88 Hz, bell −1.5 @ 500 Hz, presence +2.5 @ 1.61 kHz, presence S +3.0 @
  1 kHz, air +2.5 and air S +3.0 @ 10 kHz), 8 notches found of which 2 rendered (the 6 others
  "suspect: spans cover 63–91 % of the track", one also "musical", cap 0.0). Score 9.27 = 3.27 +
  0.5 × 12 (6 EQ bounds, 6 found notches). The attribution rows are plausible: sub 87 % of the
  low-shelf range, vox 81 % of the bell range, perc 58 % / vox 19 % of presence, vox 49 % / perc 30 %
  of presence S, perc 100 % of air.
- **Pass 1** (13 fixes: sub low shelf −1.5 dB; vox bell −1.5 dB and S shelf +3.0 dB; perc shelf +3.0 dB,
  S shelf +3.0 dB, air +1.5 dB (wanted 0.42, finding 11) and air S +1.5 dB, six static notches
  −0.5…−1.5 dB at 9–16 kHz (findings 3, 10)): RMS −10.25 dBFS (**6.15 dB short, 2.9 dB quieter than
  pass 0**), crest **13.2 dB** (+2.7), the same six EQ sections still at their bounds (air S at 2.9),
  four *new* found notches. Score 10.65 → "did not improve on pass 0 by at least 0.25", pass 0 kept,
  `passes/` holds both masters, the root is pass 0's complete render. The loop's verdict is right; what
  it shows is that brightening the stems that carry the ranges the sum's shelves wanted raises the
  sum's crest, which the stems profile's 2 dB limiter budget turns into less loudness — the EQ-bound
  term and the RMS term of the score pull against each other.
- **Memory**: peak working set 782 MB / 816 MB; guard estimate 1249 MB (for the sum alone).
- **Not run**: a 6-minute / 8-stem set (finding 18's arithmetic stands in for it); `--threads 1`; the
  desktop shell (static reading of `lib.rs` / `app.js` only); the slow-tier end-to-end tests
  (`-m slow`); a real limiter/clipper/multiband attribution path (run1 hit none of those limits, so
  `_band_moments_rank` and the limiter-moment ranking were exercised in-process only).

---

## Design notes for the maintainer (not bugs; the docs' rulings stand)

**Stem sets in practice.** Two deliveries dominate: the 4-stem "stem master" (drums / bass / music /
vocals, sometimes 5 with FX or the lead vocal apart) and the producer's own 6–12 track-group stems
(kick, sub, snare/clap, hats/perc, pads, leads, vocals, FX). `roles()` is tuned for the second (a
separate kick and a separate bass); findings 1 and 2 are what happens on the first.

**Master the sum, then correct stems** (the maintainer's definition) is the right order for stems
printed after the mix bus: it keeps the mix's balance unless the sum cannot meet spec. What a mastering
engineer expects to control in that flow and the engine does not expose: per-stem gain offsets (±1–2 dB
trims are the first thing tried, and a trim is the gentlest fix for "stem carries the range"; the loop
has no trim), mute/solo checks (`passes/` gives per-pass masters but not per-stem contributions),
stem-level M/S (S-only shelves exist, no width control), and a hard "do not touch this stem" (vocals
are usually off limits). `--stem-lock vox` and `--stem-trim vox=-1` would cover most of it.

**Where phase-B corrections can degrade a mix.** (a) The vocal compressor: 4:1, 60 ms release,
threshold at the moment peaks → on a vocal stem 85 % of frames sit in reduction (a level change with
pumping), while the chain's own stages use 300 ms meter smoothing and hull releases; a stem "peak" fix
should be a lookahead peak catcher on the top 1 % of frames (the `_moments` windows), not a program
compressor. (b) Ducking a bass that was mixed with a sidechain (2) or a music stem (1). (c) Zero-phase
shelf pre-ring on a kick: measured −19 dB re peak in the last 1 ms, −34 dB at 2–5 ms, −44 dB at
5–10 ms, −71 dB at 10–20 ms for ±3 dB at 120 Hz (`exp5_eq.py`): bounded and pre-masked, acceptable.
(d) Compressing a stem that ranks by its own peak but cancels at the sum's peak (8). (e) Static notches
on a stem for peaks the planner judged timbre (10). (f) Brightening the stems that carry a shelf's range
raises the sum's crest and, under the 2 dB limiter budget, lowers its loudness (run1 pass 1); the loop
catches the big case by score but not the small one (12).

**What a stem master's report should include** (none of it is in stems-report.md today; all of it is
one pass over data the engine already has): per stem, its share of the sum's loud-section RMS
(`Re<s,sum>/<sum,sum>` over the loud hits), its share of the limiter's work (leave-one-out peak drop
at the limiter's hits, finding 8's metric), its correlation to the sum (`Re<s,sum>/√(<s,s><sum,sum>)`:
near 0 masked, below 0 cancelling), its crest and 300 ms-meter level, its band shares at the plan's
crossovers, its noise floor (`noise.floor`, already computed when noise is flagged), and for the kept
pass: per stem what changed and the before/after of the above. The null-test line should carry the gain
and the alignment; the "Passes" section the crest / slope / section deltas per pass (12).

**Tighter ranges.** The stems profile halves the EQ bounds (±1.5 / 2.5 / 3) and the limiter's average
budget (2 dB). With the average budget binding (9) the loop never sees the limiter as a limit; either the
budget is a limit the stem pass may address, or the profile should keep the limiter's budget and
tighten only the EQ, so a shortfall shows up where the loop can act.

**Stems before or after the bus (open in the docs).** The null test as built assumes after. Stems
printed before the bus never null against a processed mix; the "processed" band and finding 15's hard
stop then block the most common delivery. If the hard stop stays, the docs and the message should say
the mix must be printed from the same stems with no bus processing, or `--mix` omitted.

---

## Verified correct

- Null test: integer offsets found and nulled at ±1, 37, 480 samples; a 6 dB louder/quieter mix passes
  with the least-squares gain (±6.02 dB); 16-bit TPDF dither passes at −81 dB; a 0.3 dB soft clip passes
  (−48.6 dB); a mix 2 s longer with a silent tail passes; the tail rule (a mix that is only the stems'
  opening fails, +11.8 dB; the reverse fails too); shares in `_range_rows` sum to 1 to 1e-9 and a
  cancelling stem reads negative; the joint-channel correlation finds a side-only signal's lag; run1's
  on-disk float32 set nulls at −149 dB.
- `read_stems`: a 50 ms / −70 dBFS tail is cut and logged with the sample count; 50 ms at −40, 200 ms
  and 2 s at −70 are refused naming the stem, the extra ms and the peak; a 44.1 kHz stem among 48 kHz
  is refused naming the stem and both rates (also when it is first); mono is duplicated L = R at the
  same level; a silent stem is accepted; 24-bit PCM reads.
- Attribution: band-moment ranking names the kick in band 0 and the hat in band 4 (`test_stems`); the
  EQ-range shares on run1 are plausible (sub 87 % of the low-shelf range, vox 81 % of the bell range,
  perc 58 % of presence, perc 100 % of air); `wanted_db` is carried from the planner's
  `range_deficit_db` (4.57 presence, 6.75 presence S); `is_limit` matches every label `attribute`
  emits, including the combined "clipper … and limiter …" one.
- stemfix: `roles()` by content with neutral names (kick, sub) and with swapped names (content wins);
  an 808 kick with a plucky bass and no name hints → `(None, None)` (conservative, no duck); kick onsets
  48/48 within 2 ms, holds 57–58 ms on a 60 ms kick and 120 ms (the clamp) on an 808; the duck shape
  (5 ms raised cosine before, hold through the kick body, 117 ms raised cosine release at 128 BPM = a
  16th, 0 dB 200 ms after; no dip where the bass is absent); caps hold across passes (EQ ±3, comp 2,
  duck 3, then the duck at its cap hands further hits to the bass compressor); `apply` works on the
  originals with the cumulative list (no compounding), leaves unnamed and silent stems untouched, is
  deterministic; `apply_sum` equals `apply().sum()` to 1e-12; the zero-phase half-gain
  forward-backward shelf is exact at the corner (1.5 dB at 1.34 kHz, 3.0 dB at 20 kHz for a 3 dB
  shelf); S-only fixes leave M bit-identical; `np.interp` of the 1 ms gain differs from PCHIP by at
  most 0.0045 (linear) and both stay inside the control points; the compressor's lookahead hull has the
  gain down before the peak; the sizing formula `1 + (g − 1) S = 10^(excess/20)` is applied and clamped
  at ±3 when the excess is positive.
- The loop (`test_stems` phase-B tests, read against the code, and run1 / run_dup): stops on no limit /
  nothing to fix / no improvement / max passes / a failed pass; keeps the lowest score with the 0.25
  tie rule (run1 kept pass 0 over a worse pass 1; run_dup kept pass 2 after two improving passes);
  promotes the kept pass's files and rebuilds the root report against the plain sum; removes stale
  `pass-j` dirs beyond the last pass and rebuilds `passes/` (hard links, copy fallback; run1:
  "stems 1 - plain sum (kept).wav", "stems 2 - pass 1.wav"); `run.json` is written last with the
  `stems` record; the pass dirs' A/B is removed; `_promote` unlinks before copying so `passes/` keeps
  pass 0's master. Cancellation mid-loop leaves the root without `run.json` (state "none" in the app,
  by design) with pass 0's master on disk and a reusable analysis cache; a partial `pass-k/` is reused
  and cleared by the next run.
- CLI: `--mix` without `--stems`, input together with `--stems`, and `--stems` with no files are
  refused with one line; the stems profile is forced; the app requires two stems and names the set by
  the common prefix (fallback to the folder name) with the work dir `<prefix> - eluate` beside the first
  stem (`lib.rs:352-372`, `uidata.work_dir`).

---

## Open questions

1. Is the hard stop on "processed" (15) the maintainer's intent, or should the null test be a warning
   that the report carries? The docs say "check first".
2. Should a limiter held by its average budget count as a limit the stem pass may address (9), or is
   the 2 dB average budget the stems profile's deliberate ceiling on loudness?
3. Should notches the planner rejected as "suspect" (timbre) ever be fixed on a stem (10), or only
   the "resonance"-flagged ones it ran out of room for?
4. For the 3-stem delivery (1): no duck at all, or a low-band-only duck?
5. Should the stems report and the UI expose per-stem trims and locks, given the maintainer's listen
   found pass 1 "only just" better and run1's pass 1 was worse on every count?
6. Null test window: ±10 ms is the maintainer's figure for post-bus stems; is ±250 ms acceptable?
