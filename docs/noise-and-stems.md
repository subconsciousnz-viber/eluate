# Noise exposure and stem input (2026-10-20)

Both features use what the engine already knows before rendering: the whole track and the
planned gain at every moment.

**Maintainer rulings, 2026-10-20:** noise processing is **opt-in**; the engine **detects** noise
and **flags it on the render** so the user knows the option exists; threshold **-60 or
-50 dB, spectrally weighted**, hiss in the high register being the likely noise. Stems: see
§2, now the maintainer's definition.

**Built 2026-10-20: detection** (`eluate/noise.py`, in report.json as `noise`, in the
report's flags, the process log and the app's footer). The estimator differs from the
proposal below: minimum statistics on the whole track read dense mixes' own pads and
reverb as "floor" (all four VA tracks read -33 to -41 dB before the master). It now reads
only the **quiet passages** (frames more than 35 dB below the track's loud level: intros,
gaps, tails, fades), where noise is not masked and is what the master exposes; the 20th
percentile of the BS.468-weighted power per third-octave band from **2 kHz up**, summed;
the master read at the premaster's quiet moments. Threshold **-60 dB** (`noise.FLAG_DB`).
On the VA: Track G -86.6 to -58.0 (exposed), Track A -67.5 to -51.9 (exposed), Track F -70.8
to -51.8 (exposed), Track C -56.9 to -37.9 (present before). A quiet musical tail reads
the same as noise, so the log lists the passages measured, for a check by ear. Not built:
the reduction (step 4) and the exposure prediction before rendering (steps 2 and 3).

## 1. Noise the master would expose

**Aim (maintainer, 2026-10-20):** remove noise that was inaudible or barely audible in the
premaster but would become audible after the master pass, and leave everything else alone.

**Why the engine can do this better than a denoiser plug-in:** the plan says how much gain
each band gets at each moment before anything renders: input gain, the input EQ (high
shelves up to +4.5 M / +6 S), multiband makeup where the compressors are not working, and the
limiter's drive (about +15 to +17 dB on the VA tracks). In quiet passages (intro, breakdown,
fade, the gaps before drops, tails) almost all of that reaches the noise floor, because
nothing is compressing there. So the noise level after the master can be predicted per band
and per moment, and reduction applied only where the prediction crosses audibility.

**Method:**
1. **Noise floor, per band (ERB or 1/3 octave), M and S, over time:** minimum statistics
   (Martin 2001) on the premaster STFT. This tracks the stationary floor and ignores
   musical events, so white-noise risers, vinyl crackle beds and other deliberate noise read
   as signal, not as floor.
2. **Exposure:** apply the plan's gain to the floor: static gain, the EQ response, the
   per-band makeup and envelope, and the limiter gain where the limiter is idle.
3. **Audibility, before and after:** the floor against (a) the masking threshold of the music
   in the same band and moment (simultaneous masking with spreading), and (b) the absolute
   threshold at a reference playback level. "Became audible" = below threshold, or within a
   margin of it, before the master, and above it after.
4. **Reduction:** only in those bands and moments, by just enough to put the floor back where
   it was relative to its threshold, capped (e.g. 10 dB). Offline and zero-phase: the gain
   is an envelope like every other stage, smoothed in time and frequency so it does not
   "burble". It goes at the start of the chain, before the input EQ. The solver loop stays
   as it is.
5. **Report and UI:** the exposure map as flagged spans on the timeline ("noise +x dB" in
   amber), what was reduced, and a residual file in `ab/` (what was taken out), so it can be
   checked by ear.

**The two suggestions:**
- **noise-repellent** (GPL-3.0, JUCE plug-in) is a front end for **libspecbleach**
  (LGPL-2.1, C17, zero runtime dependencies). That library is the useful part: adaptive
  estimators (SPP-MMSE, Brandt, Martin minimum statistics), 2D non-local means smoothing
  against musical noise, transient protection, a masking veto, tonal reduction. Both licences
  are compatible with the engine's AGPL-3.0. Using it means a C build per platform, a cffi or
  ctypes binding and packaging it with the engine. Its design is causal and real-time
  (latency, block processing); offline we would use its estimators and smoothing, with our
  exposure model deciding where it may act.
- **noise_remover-webapp** is not usable: it takes the noise profile from the first frames,
  does plain spectral subtraction, converts stereo to mono and needs ffmpeg. Nothing to take
  beyond the general idea.

**Recommendation:** steps 1 to 3 and the report in numpy first (no new dependency, all
platforms). The reduction, step 4, in numpy as a smoothed Wiener-style gain within the
exposure mask. Bring in libspecbleach only if listening shows musical-noise artefacts that
our smoothing does not handle.

**Risks:** deliberate noise in electronic music (risers, crackle, hiss textures, reverb
tails). The minimum-statistics floor and a "music, not floor" check on events are the guard;
the residual file is the check by ear. Hum (50/60 Hz and harmonics) is tonal and belongs to the
existing dynamic notch stage, not here.

## 2. Stem input (maintainer's definition, 2026-10-20)

1. Sum the stems and run the usual process on the sum, **with tighter default ranges than
   the single stereo file gets** (EQ limits, budgets).
2. If the result does not meet spec, do **not** widen the usual ranges. Find which stem
   the problem comes from and correct it there (static EQ, compression; kick and bass
   ducking with whole-track knowledge), then re-sum and master again.

Problems to attribute: shelves at their limits, a multiband band at its exception, the
frequency-dependent limiter cutting hard, the clipper at its cap, notches found but not
applied, low crest, lost section contrast, and noise (a stem's silent stretches show its
floor directly). Check first that the stems sum to the mix when the mix is supplied (a null
test). Phases: A, the sum with tighter ranges plus an attribution report (no stem audio
changes); B, stem corrections and the loop; C, stems in the UI. The tighter ranges, and
whether the stems come before or after mix-bus processing, are still open.

**Built: phase A** (sprint 2026-10-21 WP6, amended by audit 13 and the macOS report;
`eluate master --stems ... [--mix mix.wav]`, `src/eluate/stems.py`, contract in its
docstring). The null test aligns the stems' sum to the mix by one offset found on the
cross-correlations of both channels summed (one offset for the pair) and one least-squares
gain; either signal's tail outside the overlap counts as residual, so a mix that is only the
stems' opening does not pass. Verdicts: pass below -40 dB, processed up to -20 dB, mismatch
above, and silent for a mix with no energy. The attribution ranks the stems for the EQ and
section limits by their projection share of the sum (Re<stem, sum> / <sum, sum>: the shares
sum to 1, a negative one cancels against the rest), not by their own energy. Stems whose
lengths differ only by a render tail (at most 100 ms, peaking below -60 dBFS: a DAW export,
the maintainer's stem set) are cut to the shortest and logged; any other difference is an
error.

**Built: phase C** (maintainer, 2026-10-27). The app's **Stems** button picks the stems, then
optionally the stereo mix (Cancel skips the null test). The set becomes one track named
after the stems' common prefix, with its work dir `<prefix> - eluate/` beside the first
stem. The engine sums it there (`stems-sum`, null test included) and that sum is the
track: preview, background analysis, graphs and A/B as for any track. Render runs the
stems flow above (`master --stems`, the stems profile and stems-report.md). The shell
keeps each set's files and adds `--stems` / `--mix` itself; the page cannot pass them
(audit 13 #4).

**Phase B decisions** (maintainer, 2026-10-27): the stems profile's ranges stay as they are (a
bound hit hands the work to the stem pass); the maintainer's stems are printed **after** his mix
bus (they sum to the mix, so stem corrections stay gentle); the loop runs at most two
correction passes and stops early when no limit is hit or a pass does not improve on the
last, keeping the best pass; corrections built: static EQ, compression, kick and bass
ducking.

**Built: phase B** (2026-10-27; `src/eluate/stemfix.py` contract in its docstring, the
loop in `cli.do_master_stems`). After the sum pass, up to two correction passes: stemfix
plans cumulative fixes from the last pass's attribution, always on the original stems
(an EQ shelf on the stems carrying a range the sum's shelf hit its bound in, sized from the
planner's wanted gain; a notch on the notch's channel, capped at the planner's cut when it
is flagged musical; peak compression on the stem behind a limiter, clipper or multiband
limit; the bass ducked under the kick's onsets when the limiter's moments fall on them),
re-sums one stem at a time and masters again in `<work>/pass-k/`. Score: RMS shortfall to
the target plus 0.5 per real limit; a pass must improve by 0.25 to count or to be kept. The
kept pass's outputs are copied to the work dir root; a failed correction pass falls back to
the best pass before it. Per-stem caps: ±3 dB EQ per band, 2 dB compression, 3 dB duck. No
fix for the crest / section row (it names the most compressed stems already).

Maintainer listen, on the maintainer's stem set (2026-10-27): pass 1 (kept) over the plain sum and pass 2, "but only
just. more natural open sounding top end". On paper pass 1 only cleared the found notch:
loudness 0.1 dB lower, both shelves still at their bounds (the sum wanted +6.5 dB air
against the 2.5 dB bound; perc and snare took the stem cap of +3 dB).

**Addendum, 2026-11-10: the stems audit's fixes.** The phase B decisions above stand (ranges
unchanged, stems printed after the mix bus, at most two correction passes, corrections
limited to static EQ, compression and kick/bass ducking). What changed in behaviour:

- *Notches.* A notch the planner itself held back on (a "suspect" reason, the chord-tone
  guard, no cut, or a 0 dB cap) is listed as "found, skipped: <why>", is not a limit and is
  never cut on a stem. A 0.0 dB musical cap is a cap (it read as "no cap" and the stem got
  1.5 then 3 dB). A notch that ran out of room, or an S-only one the notch stage cannot
  take, may still be cut on the stem.
- *Ducking.* The bass role needs evidence: a bass/sub/808 name, or a low band at least
  15 dB over the rest of the stem's energy, tonal, and not the set's dominant broadband
  stem. A 3-stem delivery's "music" stem (sub + pads + lead) is never ducked full band;
  its limiter entry compresses instead. A bass already dipping under the kick (a sidechain
  in the mix: its level 10-60 ms under the kick against the louder of 50 ms before and
  100-200 ms after) has that dip taken off the duck; the audit's 6 dB sidechained sub gets
  none (it got 2.2 dB more).
- *Limits.* The limiter held by its average budget and the loudness shortfall are one limit
  (detail "which" says which), ranked by the leave-one-out drop of the sum's peak at the
  limiter's median reductions. Limiter and clipper moments rank the same way: a stem
  cancelling at the peaks ranks last and is never compressed for them.
- *EQ.* A stem shelf or bell only when the planner wanted more than the bound it applied
  (shelves: its range deficit; low shelf, bell, band shelf: the median of its fit's node
  deficits in the range), sized from that excess. Run1's air shelf (+2.5 applied, 0.42
  wanted) now proposes nothing.
- *Attribution is M/S aware.* Each EQ section's shares and levels are read on its own
  channel (presence and air M, the side shelves S, the rest L+R); a notch on its channel.
  A side-only stem shows up for the S shelves.
- *Stem compressor.* Its detector is peak-held over 25 ms, so a sub gets no gain ripple at
  twice its fundamental (harmonics stay at -124 dB, were -43 dB).
- *Null test.* The magnitude of the correlation aligns a polarity-inverted mix (reported,
  the run proceeds); the offset search is +-250 ms; an exact null reads -200 dB (it crashed
  the report: `--stems a.wav --mix a.wav`). The message names what differs: polarity, gain,
  length / tail, a sub-sample offset, swapped or unequal channels, a limited or clipped
  reference. "Processed" is a warning: logged, carried in the stems report and the app's
  stems-sum message, and the sum is mastered as given; "mismatch" and "silent" still stop.
  A "mismatch" that a sub-sample shift takes below the pass line is "processed".
- *Inputs and memory.* Duplicate file names, the same file twice, a missing file and more
  than two channels are refused naming the stem. The memory guard counts the set: stems x
  frames x 2 channels x 8 bytes x 2 (the set held through every pass, plus stemfix.apply's
  copy for the attribution; read_stems fills one preallocated array). 8 stems x 6 min at
  48 kHz adds 4.1 GiB to the sum's estimate; the record is in run.json "stems".
- *Speed.* Every stem's spectra are computed once per attribution (one transform per
  channel, all cross spectra from those) and the notch readings in one pass per stem.
  `attribute()` on the audit's 6 x 30 s run1: 35.0 s before, 4.5 s after.

*Design notes for the maintainer (not implemented).*

- *Per-stem trims and locks.* A +-1-2 dB stem trim is the gentlest fix for "this stem
  carries the range" and the first thing an engineer tries; the loop has none. A lock
  (`--stem-lock vox`: never corrected) and a trim (`--stem-trim vox=-1`) would cover most of
  what a stem master needs and give the loop a fourth, mildest correction.
- *Stems printed before the bus.* The null test and the gentle corrections assume stems
  printed after the mix bus (the ruling above). Stems printed before it never null against
  a processed mix: with "processed" now a warning such a set masters, but the corrections
  then act on stems that were not the mix's. If that delivery is to be supported, the null
  test's verdict should select a mode (corrections off, or a wider "processed" band), and
  the docs should say which delivery the engine expects.
- *What the stems report should show.* Per stem: its share of the sum's loud-section level,
  its share of the limiter's work (the leave-one-out drop now computed), its correlation to
  the sum (near 0 masked, below 0 cancelling), its crest and 300 ms level, its band shares
  at the plan's crossovers and its noise floor; for the kept pass, per stem what changed
  and the before / after of those. The Passes section should carry each pass's crest,
  slope and section-contrast deltas: the score sees only RMS and limits, and a pass can
  trade a small RMS gain for more crest (the audit's run1 pass 1: crest +2.7 dB).

## Questions for the maintainer

Noise (open): -60 or -50 dB as the threshold (built at -60); the maximum reduction;
whether S gets its own treatment (the side shelf at +6 dB is where hiss is most exposed).

Stems: answered 2026-10-27 (phase B decisions above).
