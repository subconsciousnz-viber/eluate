# Stem and premaster export (draft, 2026-11-21)

Status: **proposal, nothing built.** It extends the stems flow in
[noise-and-stems.md](noise-and-stems.md) §2: phases A, B and C, the phase B rulings and the
2026-11-10 addendum. The questions for the maintainer are at the end.

## 1. What the user gets

A stem set is mastered as it is today: sum it, master the sum, attribute the limits, then
run at most two gentle correction passes on the stems and keep the best pass. Export then
offers four files or folders, all from the kept pass:

| Export | What it is | Sums to |
|---|---|---|
| **Master** | `master.wav`, as today | n/a |
| **Premaster** | The corrected stems summed, before the master chain, at the level it was mastered from | n/a |
| **Premaster stems** | Each stem (or group) with only its own corrections (EQ, notch, duck, compression) | the premaster, exactly |
| **Master stems** | Each corrected stem (or group) through the master chain, plus one **colour** stem | the master, to float rounding |

Groups (section 5) let a 40-channel DAW export come in as, say, eight named groups: drums,
bass, music, vox and so on. A group is the sum of its files and is treated as one stem
everywhere below.

**Not in scope:** an automixer. The corrections keep their phase B caps: ±3 dB EQ per band,
2 dB of compression, 3 dB of ducking. They keep the ruling that stems are printed after the
mix bus. Making a mix to hit the slope target from raw, pre-bus tracks is a different
product (section 8, question 4).

## 2. Why master stems are possible: the chain is linear once the plan is fixed

Every stage is `apply(x, fs, section)` with no hidden state (`stages/__init__.py`). With the
plan fixed, as solved on the sum, most stages are linear in x: static filters, or gains that
vary with time but come from the plan and not from x. For such a chain L,
L(a + b) = L(a) + L(b). So running each stem through L with the sum's plan gives stems that
add up to L(sum).

| Stage | Linear with the plan fixed? | Evidence |
|---|---|---|
| input (gain) | yes | `stages/gain.py`: scalar gain |
| inputeq | yes (LTI) | `stages/inputeq.py`: static sections (zero phase except the high-pass) |
| notch | yes (LTV) | `stages/notch.py`: the depth, kick-mask and clamp envelopes all come from the plan |
| sidestrip | yes (LTI) | `stages/sidestrip.py` |
| multiband, speclim | yes (LTV) | `stages/multiband.py`: GR and ceiling envelopes from the plan; zero-phase split |
| tone | yes (LTI) | `stages/tone.py` (None since chunk 2 in any case) |
| glue, `clean` curve | yes (LTV) | `stages/glue.py`: one envelope from the plan, a mix of x and the wet signal |
| glue, `vintage` curve | **no** | E(wet) is a soft curve on the signal |
| saturate | **no** | `stages/saturate.py`: tanh on the band. On in the electronic and stems profiles (≤ 1 dB drive under stems) |
| limiter: clipper | **no** | `stages/limiter.py` 3a/3b: whole-signal or onset clipper at 8x. On by default (about 1 dB peak) |
| limiter: 8x gain, re-limit, trim | linear **if the gains are captured**. They are computed inside the stage from its input (`apply_with_info`): `g` at 8x, `g2` per re-limit pass, the trim | |
| output | quantisation | per-stem float32; see section 3 |

L is the chain with saturate, the clippers and the vintage term left out. The limiter's
gains are not recomputed: the ones captured from the sum's real render are replayed.

## 3. The colour stem

Let M be the sum's real master in float, before output quantisation. Because L is linear,
the stems' master renders add up to L(sum):

    Σ L(stem_i) = L(sum)

So the stem that makes the set null against the master is

    colour = M − L(sum)

It needs **one** extra render of the sum, not one per stem. It holds everything the nonlinear
stages added: saturation harmonics, the clipper's cut and its band-limited change, the vintage
curve. In each case that is after every later stage's gains, because later stages are linear
with the plan fixed. Expect it well below the master: about 1 dB of clipping and at most
1–2 dB of drive. Phase 1 measures its level on the fixtures.

**How exact it is:**
- The stems are written as 32-bit float, so the set nulls against M to float32 rounding
  (below −130 dBFS).
- Against a 24-bit `master.wav`, the remainder is the master's own quantisation and dither,
  at 24-bit LSB level.
- `--streaming` writes 24-bit and `--cd` resamples. Master stems are not offered under
  `--cd`, which is not available with `--stems` anyway.

**Alternative (question 1):** a "stem-safe" master with saturate and the clippers off, so
there is no colour stem. That is a different and quieter master, since the limiter takes the
clipper's work. 0.2.0 is already short of its target on the fixture.

## 4. Engine design

The new code goes in `src/eluate/stemexport.py`, with a contract docstring in the house
style. The flow:

1. **Capture.** A variant of `limiter.apply_with_info` (a `capture=` argument, or a sibling
   function) returns `g` (8x, stereo-linked), every re-limit pass's `g2` and the trim. The
   kept pass's final render runs it once, on the same plan. The gains are saved under
   `export/` as float32 `.npy`, opened memory-mapped when used. `g` is one curve for both
   channels; at 8x in float32 it is about 0.55 GB for 6 minutes at 48 kHz, so it is never
   held twice.
2. **`render_linear(x, fs, plan, gains)`**, in `stages/__init__.py` or `stemexport.py`. It
   runs `STAGE_ORDER` and:
   - skips saturate;
   - runs glue with the vintage term off;
   - runs the limiter as input gain, then `upsample`, then `× g`, then `decimate`, then `× g2`
     for each pass, then the trim, with no clipper and no band-limit.

   The same pure-function contract as `render`. Its test: `render_linear(sum)` +
   colour = `render(sum)` to 1e-9. The checked-in identity is
   Σ `render_linear(stem_i)` = `render_linear(sum)`.
3. **Corrected stems one at a time.** Add `stemfix.apply_one(stems, fs, names, i, fixes)`,
   the per-stem body `apply_sum` already loops over. The duck reads the kick stem from the
   set as it does now, so each corrected stem is made without a second copy of the set.
4. **Write:**
   - `export/premaster stems/<name>.wav` (float32);
   - `export/premaster.wav` = Σ of those. It is the kept pass's `corrected-sum.wav`, or
     `stems-sum.wav` when pass 0 is kept, written once.
   - `export/master stems/<name>.wav` = `render_linear(corrected stem)`;
   - `export/master stems/eluate colour.wav`;
   - `export/export.json`: the run id, the pass kept, the plan's sha256, the files and the
     measured null (section 7).

   Writes go stem by stem. Peak memory is one stem's limiter-rate arrays (about 2.6 GB per
   minute of stereo at 8x, `limiter.py`'s note), as a render today.
5. **CLI:** `eluate export <work> [--stems-out master|premaster|both] [--premaster]`. It runs
   after `master --stems` on a finished work dir (`check_run_files`). It checks that the
   stems' sha256 still match `run.json` and recomputes the corrected stems from the recorded
   fixes, so export never re-solves. Running it as its own step keeps a master's runtime
   unchanged for those who don't export.
6. **Levels:** the premaster and the premaster stems keep the level they were mastered from.
   No normalisation, so they null against the files the user gave. Question 3 covers adding
   headroom.

**Time:** about one full-chain render per stem, minus saturation and the clipper, plus one
for the colour stem. The master's solve already runs up to `solver.MAX_OUTER` (8) outer
renders, so eight groups add roughly one master's time. Forty ungrouped stems would add
about five. That is the case for groups, beyond workflow.

## 5. Groups

- **CLI:** `--group drums=kick.wav,snare.wav,hats.wav --group bass=sub.wav,bass.wav ...`, or
  `--groups groups.json` (`{"drums": [...], ...}`). Files not in any group stay as their own
  stems. A file in two groups, an empty group, or a group name equal to a file's stem is
  refused, naming it (`stems.check_paths` rules).
- **`stems.read_stems`** sums a group's files into one stem as it reads them: one file at a
  time into the preallocated array, so 40 files never sit in memory at once. The
  render-tail rule applies per file, before summing. The memory guard counts groups, not
  files.
- **Names matter.** Roles for the duck are found by content, with names as evidence
  (`stemfix.roles`, `bass_ok`: bass, sub, 808). The UI should suggest group names such as
  Kick, Bass, Drums, Music and Vox, and say why.
- **The null test is unchanged:** the groups' sum is the stems' sum.
- **App:** the Stems button picks the files and then opens a grouping sheet. It lists the
  files, has a group column (type-ahead from the names already used), and auto-groups by
  common prefix or DAW numbering as a starting point. The shell passes `--group` itself;
  the page never builds argument strings (audit 13 #4).

## 6. App: export

After a stems render, the results view gets an **Export** button with the four checkboxes
from section 1 and a destination folder. The default is `<set> - eluate/export/`. Export runs
the engine's `export` job in its own slot, with progress per stem. The A/B is unchanged.
Previewing a single stem stays out of scope.

## 7. Tests and acceptance

- Linearity, on `fixtures.e_signals.short_track` split into synthetic stems (kick, bass,
  pads, hats), for every profile and every character preset:
  - |Σ master stems + colour − M| ≤ 1e-9 in float;
  - after writing float32, the null is below −120 dBFS.
- The premaster stems sum to `corrected-sum.wav` bit for bit (same summation order), or to
  1e-12.
- A pass 0 keep (no corrections) gives premaster stems equal to the inputs.
- Report the colour stem's RMS and peak relative to M in `export.json`. It is a
  regression gauge, not a gate.
- Groups:
  - files summed equal the group read as one file;
  - a 40-file set at 48 kHz × 6 min stays inside the memory guard;
  - refusals for bad group specs.
- `eluate export` on a work dir whose stems changed refuses, naming the stem.
- Rust: grant checks for the `export` job's destination, as for the other output folders.

## 8. Questions for the maintainer

1. **Colour stem, or a stem-safe master with no saturation or clipper?** The colour stem is
   proposed: same master, exact null.
2. **Export master stems by default, or only on request?** Their cost is about one render
   per stem.
3. **Premaster level:** as mastered from (proposed), or normalised to a peak such as
   −6 dBFS for handing to another engineer? Normalising breaks the null against the inputs
   but not between the premaster stems and the premaster.
4. **Raw pre-bus multitracks:** support them only as groups mastered as a sum, as now, with
   the null test's "processed" band as the guard? Or open the "stems printed before the bus"
   design note (noise-and-stems.md addendum) as its own project?
5. **Trims and locks** (`--stem-trim vox=-1`, `--stem-lock vox`, from the addendum's design
   notes): build them with groups, as a group-level control, or later?

## 9. Phases

| Phase | Work | Estimate (Claude session time, see below) |
|---|---|---|
| 1 | Limiter gain capture, `render_linear`, colour stem, linearity tests, colour levels measured on the fixtures | 3–5 h |
| 2 | `eluate export`: premaster, premaster stems, master stems, export.json, stem sha256 checks | 3–5 h |
| 3 | Groups in the engine (CLI, read_stems, memory guard, tests) | 2–4 h |
| 4 | App: grouping sheet, Export button, shell job and grants, smoke on Windows CI | 5–8 h |
| 5 | Maintainer listen on a real stem set and a 40-channel export; fixes | depends on the listen |

These are hours of Claude sessions, not calendar time. The limits are the maintainer's
listening and decisions (section 8) and Windows CI runs of about 35 minutes each.
