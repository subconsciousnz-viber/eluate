# Dynamics macros (draft 0.1, 2026-10-19)

A proposal, for maintainer correction, to turn the fixed budgets of gain-staging.md §3
into rules that scale with the material, steered by a few macro controls that can
become the UI. Nothing here is built. Numbers come from synthetic loops (the
drum loop in `tests/test_average_peak.py` and variants); values marked
*(corpus)* need the calibration run on the maintainer's machine (§7).

**Update 2026-10-20:** step 1 (§8) is built with the architect's values: clipper
per-sample cap 3 dB, limiter outlier cap 5 dB, glue 4 dB, multiband 4.5 dB
(gain-staging.md §16), then the compressors' budgets moved to a meter reading and a
frequency-dependent limiter took the stray peaks (§18), which together addressed E1
(hit spread) and E3 (the glue now FET-style). Steps 2 to 5 (job sizing, shares, ρ,
the macros) and the §9 questions are open; the maintainer wants a UI-levers conversation
next, which is where the macros belong.

**Maintainer rulings, 2026-10-20 (UI levers):** the names Punch, Hardness and Control and the
§3 ranges are accepted, but **none of them goes in the UI for now** (no Ozone-style
complexity; a fine-tune panel may come later). In the UI, **Character is the timing
preset** (auto from tempo, or transparent / slow / moderate / fast / aggressive, as
`--character` already does), so the §4 renaming to `--timing` is dropped. **Loudness
belongs in the character range:** part of the roughly 14 dB of gain reduction across the
stages should scale with the loudness target (step 2). Genre loudness presets are retired,
so the "genre target" column in §4 no longer applies. The macros may still steer the
engine internally (steps 2 to 5), with the characters as their settings.

## 1. What blocked us, measured

Five edges, all from the same cause: absolute numbers that fit a typical,
well-mixed track and stop fitting as the material gets more dynamic or faster.

| # | Edge | Measured (synthetic) | What the fixed rule did |
|---|---|---|---|
| E1 | **Hit spread.** The loudest hit sits well above the typical hit. | Limiter input: loudest hit 2.8 dB above the median loud hit. Glue output: 4.3 dB. | The +1.5 dB outlier cap bound first, so the average was pulled down instead: limiter 1.7 dB average (target 3), clipper 0.0 dB (target 1). More dynamic material got *less* reduction (maintainer: wrong). |
| E2 | **Peak and average do not scale together.** | Per band, outlier minus average = 0.85 to 2.0 dB, tracking the band's hit spread (§2). | One fixed allowance for every band and stage. |
| E3 | **Fast attacks against the glue's lagging attack.** | Glue crest change (dB, average reading): +1.1 (10 ms glue attack, instant kick), +0.45 (10 ms, kick peaking at about 26 ms), +0.06 (3 ms, same kick), -0.43 (10 ms, kick peaking at about 48 ms), +0.20 (30 ms, same kick). | The glue let transients through and squeezed the body, so the crest *rose*; more glue (the average reading) made it worse. |
| E4 | **The size of the job varies.** | Premaster crest 9.2 to 19.9 dB against the maintainer's masters at 7.1 to 8.7 *(corpus §9)*: a crest job of about 1 to 12 dB. | Fixed budgets sum to about 8 dB of typical-hit work, so dynamic tracks fell short (Track B 4.8 dB, Track A 7.3 dB). |
| E5 | **Bands differ.** | Band crest rises with frequency (premaster medians 10 / 15 / 18 / 23 / 24 dB); the maintainer's masters reduce it by 2.4 / 1.8 / 1.4 / 1.2 / 1.1 dB *(corpus §9)*. | 2.5 dB average on every band and M / S. |

## 2. The law that links average and outlier

For a stage with static slope σ (σ = 1 - 1/ratio for a compressor; σ = 1 for
the limiter, and for the clipper above its knee), solved for an average peak GR
G, the loudest hit gets about

    outlier ≈ G + σ · T,    T = (loudest hit peak) - (median loud hit peak), on the stage's input

Measured on the drum loop (multiband, per band, M and S): predicted σ·T against
measured outlier minus average, 0.94 / 0.85, 1.22 / 0.99, 1.37 / 1.23, 1.76 / 1.48,
1.72 / 1.57, 1.43 / 1.10, 2.24 / 2.00, 2.10 / 1.86, 2.17 / 2.00, and 2.62 against
2.00 where the 4.5 dB clamp bound. The prediction runs about 10 % high (attack and
release smoothing, the knee). For the limiter it is exact: T = 2.83 dB, so a
4.5 dB outlier cap leaves 4.5 - 2.83 = 1.67 dB average, which is what the solve
returned.

Reading: the maintainer's +1.5 dB is this law on a track whose loudest hit sits about
1.5 dB above its typical hit at the limiter (*hypothesis, to check in §7*). A
fixed allowance is right only for that track. The proposal makes the allowance a
consequence of the material (σ·T), bounded by a per-stage **audible maximum**
that the character sets, so a more dynamic track gets more reduction (maintainer,
2026-10-19).

## 3. The algorithm

Measure, size the job, share it out, solve each stage on its real input, hand any
excess downstream. With the default macros on a typical track it reproduces the
maintainer's numbers (2.5 / 1.5 / 1 / 3 average, about +1.5 outlier).

**Descriptors** (the hit reading already built, on each stage's input; per band
and M / S for the multiband):
- P50, the median loud-hit peak; T, the loudest hit's peak above P50 (T90 = p90 - P50
  as a robust variant);
- R, the loudest-section 1 s RMS; H = P50 - R, the typical hit above the body;
  crest C = H + T;
- τ, the median time from a loud hit's onset to its peak (1 ms peak envelope);
- per band, the crest against the corpus band table (§5 of gain-staging.md).

**Job.** The target crest is C* = ceiling - RMS target (-0.3 - (-7.7) = 7.4 dB by
default). The total job is J = C_in - C*, split into
- the tail job J_T = (1 - ρ) · T_in: the outliers brought towards the typical
  hit, keeping a fraction ρ of the input's hit spread (the Control macro);
- the hit job J_H = J - J_T: the typical hit brought towards the body.

**Shares.** The maintainer's averages are the shares on a reference track:
G_s = k · G_s(maintainer) · b_s(Punch), with k = J_H / J_H(ref) (J_H(ref) from the
corpus) and b_s the Punch bend (§4). Each G_s is clamped to its stage range
(table below). What one stage cannot take moves to the next stage in chain order.

**Solve, in chain order, on the real input** (bisection as now):
1. Threshold for the average G_s.
2. Outliers follow the law (G_s + σ_s · T_s). They are held at the stage's audible
   maximum M_s per sample, and the average is never lowered to make room:
   - multiband and glue: the gain floor already built, now at M_s instead of
     budget + 1.5;
   - clipper: its reduction capped per sample at M_s, so the curve becomes
     max(curve(x), x · 10^(-M_s / 20)) and anything beyond goes to the limiter;
   - limiter: the last stage. Past M_lim the loudness gives way (shortfall warning)
     or, with Hardness high, the limiter accepts it.
3. Crest guard (multiband and glue): measure the stage's crest change on the real
   audio; if it exceeds δ(Punch), shorten the attack by bisection until it does
   not. This fixes E3 from the audio itself instead of a fitted rule, since the
   glue figures above do not follow attack / τ alone.
4. Re-measure the descriptors on the stage's output and carry the remaining job.

**Ranges** (proposal, maintainer to correct; averages, then audible maxima at Control 50):

| Stage | Average range | Audible max M_s |
|---|---|---|
| Multiband compressor, per band and M / S | 1.5 to 3.5 dB (exception: 3 + band limiter 3, as now) | 6 dB |
| Glue (net) | 0.75 to 2.5 dB | 4 dB |
| Clipper (output-peak fall) | 0.5 to 2 dB | 3 dB for transients under 2 ms, 2 dB otherwise |
| Limiter | 1 to 4 dB | 6 dB |

Audible maxima scale 0.7x (Control 0) to 1.3x (Control 100).

## 4. The macros

Four dynamics macros (0 to 100) plus the existing loudness target. Each bends
named parameters; nothing else moves.

| Macro | 0 | 100 | What it bends |
|---|---|---|---|
| **Loudness** (exists: `--target-rms`) | quiet | loud | C*, so the total job J |
| **Punch** | smooth | punchy | multiband and glue attack: 0.3x to 3x the measured τ (tempo timing as the base); crest guard δ: -0.5 dB (the glue must lower the crest) to +0.5 dB (transients may rise); hit-job shares: smooth puts more on the multiband and glue (the body), punchy more on the clipper and limiter (the peaks only) |
| **Hardness** | soft | hard | the share of the remaining tail taken by the clipper against the limiter (0.3 to 0.8); clipper knee 1.5 to 0.5 dB (no hard knee: maintainer, 2026-10-23, gain-staging.md §7); clipper audible max 1.5 to 3 dB; saturation drive 0.5x to 1x the deficit rule |
| **Control** | light | firm | ρ, the hit spread kept: 0.5 to 0.15; the audible maxima 0.7x to 1.3x |
| **Band balance** | uniform | per band | 0: the same average on every band (as now); 100: each band's average from its own crest job against the corpus master band crest *(corpus)* |

**Characters** are points in that space (starting values for the maintainer to move):

| Character | Punch | Hardness | Control | Band balance | Loudness default |
|---|---|---|---|---|---|
| transparent | 50 | 0 | 35 | 50 | genre target |
| light | 60 | 20 | 10 | 30 | genre target - 1.5 dB |
| smooth | 10 | 20 | 70 | 60 | genre target |
| punchy | 90 | 50 | 40 | 50 | genre target |
| hard | 60 | 100 | 90 | 70 | genre target + 1 dB |

Naming: `--character` today is the §7 timing preset (transparent / slow / moderate /
fast / aggressive, auto from tempo). Proposal: rename it to `--timing` (still auto
from tempo, and the base the Punch macro scales) and give `--character` these five,
with `--punch`, `--hardness`, `--control` and `--band-balance` to override one at a
time.

## 5. The limiter's loud section (point 5, restated)

The loud section is chosen per stage from the 1 s RMS of that stage's input. For
the limiter, its input is the clipper's output, which exists only inside the blocked
limiter, so the beats are chosen on the clipper's input. On the drum loop the
clipper changes the 1 s RMS by at most 0.009 dB and the p95 by 0.005 dB, and all
33 beats land in the same place (loud or not) either way. Choosing them on the
clipper's output would need one extra render pass, for no measured difference.

## 6. The multiband's loud section (point 3)

What is measured now, per band and M / S:
- **the GR**: that band's own compressor envelope, so the hit values are that
  band's GR in each beat (per band, as the maintainer meters);
- **which beats count**: the beats where the *whole* stage input (stereo,
  broadband) is within 3 dB of its loudest section, the same beats for every band.

The alternative is per band: the beats where *that band's* 1 s RMS is within 3 dB
of its own loudest section. The two differ when a band is loudest somewhere other
than the full mix's loudest part, e.g. pads or vocals peaking in a breakdown, or a
sub that drops out of the last chorus. On the drum loop (no arrangement) they
differ by 2 to 4 of 33 beats per band; on real arrangements it will be more.
Proposal: per band, since each compressor is driven by its own band and the maintainer
meters per band. Maintainer to rule.

## 7. Calibration on the maintainer's masters (local, numbers only)

The 17 premaster and master pairs (§9 of gain-staging.md) are the training data
for this mapping. A local script (`work/corpus/calibrate.py`, like `measure.py`:
audio never leaves the machine, it writes numbers only) would measure §3's
descriptors on each premaster and master, then:
1. read J_H(ref) and the typical tail T at each stage position (checks the §2
   hypothesis that the maintainer's +1.5 dB is the law at a typical tail);
2. fit ρ and the band-balance targets so that the default character reproduces the
   pairs' median crest, band crest and tail;
3. place each of the maintainer's masters in macro space, so the five characters start
   from where the maintainer's own work sits.

## 8. Build order (if approved)

1. The law and the audible maxima in place of budget + 1.5 (E1, E2); the clipper's
   per-sample reduction cap.
2. The job and the shares (E4), the Loudness link, ranges.
3. The crest guard and the τ-scaled attacks (E3): the Punch macro.
4. Hardness, Control, characters and the CLI.
5. Band balance and per-band loud sections (E5, §6) once the calibration has run.

## 9. Questions for the maintainer

1. The audible maxima and average ranges in §3.
2. ρ: how much of a premaster's hit spread a master should keep, by default.
3. Per-band loud sections (§6).
4. The five characters' starting points (§4), and whether these four macros are the
   right axes.
5. Renaming the timing presets to `--timing`.
