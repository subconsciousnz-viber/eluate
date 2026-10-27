# Brief: Offline Auto-Mastering Engine

## Why

I want to build a free, offline, non-causal auto-mastering tool for electronic music and give it away. Existing assistants like Ozone's push things too loud and too bright, ignore per-band peaks, and target a static average spectrum instead of how a track behaves over time.

The core idea: offline, lookahead is effectively unlimited and the whole track is known before any processing happens. That allows time-varying processing nobody would ever automate by hand.

Format is a standalone app, not a VST. Start with a batch CLI, add a GUI later, and keep ARA as an optional future bridge into DAWs.

The approach is informed by the way I approach mastering for people - challenge my assertions but respect my experience and knowledge.

## Phase 1: Research dive (do this first)

Produce `research/report.md` covering:

1. **Prior art.** Cover commercial tools (LANDR, eMastered, BandLab Mastering, Ozone Assistant, any FL Studio or Ableton auto-mastering, Sonible smart:range and smart:limit, and anything else you find). Cover open source and academic work too (Matchering, DeepAFx-ST, Steinmetz et al. on differentiable mastering, reference-matching and intelligent music production literature). For each, note what it actually does over time versus statically. I believe none of them track key or note changes. Verify that claim, don't assume it.
2. **Feasibility, per feature** in the spec below: approach, known methods, hard parts, risk rating.
3. **The −4.5 dB/oct claim.** Is that the average long-term slope of commercially mastered music? Find the primary sources (e.g. Pestana et al. on spectral characteristics of commercial recordings) and report the actual figure and its variance, including any genre split.
4. **Gain-curve dynamics versus modulated-parameter compressors.** Is precomputing an ideal gain envelope and applying it directly better than modulating ratio and attack? I find parameter modulation artifacts badly, even when rendered.
5. **Key and bass-note tracking robustness on dense electronic mixes.** Compare chroma/CQT on the full mix with pYIN on a source-separated bass stem (e.g. Demucs).
6. **Whether differentiable DSP is viable** for the constraint solver, compared with simple iterative render, measure, adjust.

Cite primary sources. Mark anything you couldn't verify.

## Spec

### Analysis pass (whole track, before any processing)
- Short-term (3 s) and momentary (400 ms) loudness over time, per ITU-R BS.1770 / EBU R128
- Per-band crest factor over time, in mid, side and stereo
- Per-band mid/side correlation (stereo decoherence)
- Transient density and onset strength
- Spectral centroid and long-term tilt (peak-hold and average)
- Key, chord and bass-root tracking
- Section segmentation (intro, breakdown, build, drop, etc.)
- Persistent narrowband resonances, meaning peaks above a smoothed spectral baseline that last over time

### User controls (keep this minimal)
- **Tilt:** ±3 dB maximum, around the reference slope from research item 3. This lets the user flatten the curve or steepen it.
- **Loudness target:** measured over the loud phrases only, never integrated across the whole track. Propose a definition, e.g. a high percentile of short-term LUFS, or the mean over the loudest detected sections.
- **Preserve mix toggle:** "Is your mix and dynamic balance already right?"
  - Yes: tonal balance and between-section dynamics are locked within tolerance, and the engine only gains loudness.
  - No: the engine may reshape tone and section dynamics.
- **Crossover frequencies** for the M/S bands, including the sub side-strip frequency.

### Default chain (this is how I master; encode it)
I can do a passable master from the RTA alone, looking at mid, side and stereo views. The chain:

1. **4-band M/S multiband compressor/limiter** as the first tone shaper and transient catcher.
   - No more than −3 dB gain reduction on any band.
   - Low sides: compress with **no makeup gain**, which narrows the bottom.
   - High sides: compress with a **couple of dB extra makeup**, which widens the top.
   - Scale back or skip the widening when the per-band correlation shows the mix is already very stereo-decoherent.
2. **Sub handling:** if the sub is boosted, boost it in the mid and strip the sides below a variable crossover. Never mono-sum. Use a variable high-pass on the side channel only.
3. **Dynamic notching:** detect persistent resonances and apply notches whose depth varies over time, engaging only while the resonance is actually present. This is the automated notch I usually can't be bothered to draw.
4. **Saturation:** gentle, oversampled, weighted to follow the tilt curve. Saturate the top on the sides for width, and don't drive the sub.
5. **Key and note-aware EQ:** master EQ moves that follow key or bass-root changes, e.g. taming a resonance tied to the current root. This is experimental; gate it behind the research findings.
6. **Final loudness stage:** a true-peak limiter with offline lookahead. It must respect per-band peak and crest-factor constraints, so the kick and sub keep their punch.

### Constraint solver
Targets are expressed as constraints, not presets. For example: "hit the loud-passage target and the tilt setting, with no band losing more than X dB of crest factor, and true peak ≤ −1 dBTP." Iterate render → measure → adjust until the constraints are met, or report which ones can't be met and why.

### DSP notes
- **Precompute gain envelopes rather than modulating compressor parameters.** Smooth control curves with zero-phase forward-backward filtering (`filtfilt`) to avoid zipper noise and pumping.
- **Forward-backward filtering of audio squares the magnitude response.** It's zero-phase with no pre-ringing, but attenuation doubles. Design for that.
- **Zero-phase crossovers:** build the band splits complementarily (HP = x − LP_zero-phase) so the bands sum back perfectly. Don't filtfilt a Linkwitz-Riley pair and assume it still sums flat.
- **Measure everything** in mid, side and stereo.

## Engineering constraints
- **Prototype first** in Python (numpy, scipy, librosa or equivalent, pyloudnorm or your own BS.1770 implementation). Choose the production language after the prototype proves out, and justify the choice.
- **Write LF line endings.** Don't rewrite files with Python text-mode defaults on Windows; use `open(..., newline="\n")`.
- **Any GUI text** is limited to control names, live values, must-know state and navigation. No taglines, hints, badges, captions or explanatory copy anywhere in the UI.
- **Licence:** it's a gift to the world, so recommend an open-source licence and say why.

## Evaluation
For every render, report the following before and after processing:
- Loud-passage loudness and per-section loudness
- PLR / PSR
- Per-band crest factor change
- Per-band M/S correlation change
- Fitted tilt against the target
- True peak
- Notches applied, with their time ranges

Also produce loudness-matched A/B renders, so I can judge by ear without being fooled by level.

## Auditor protocol: Gemini 4 Argon via `agy` CLI

Use Gemini 4 Argon, through the `agy` CLI, as an independent auditor and second opinion. You own the work; it reviews it.

**Setup:** run `agy --help` first to confirm non-interactive invocation and how to select Gemini 4 Argon. If the CLI or model isn't available, tell me and continue without it. Don't fake audit results.

**When to call it (minimum):**
1. After the Phase 1 report: check for missed prior art, hallucinated or wrong citations, and overstated feasibility.
2. After the architecture and spec decisions.
3. After each DSP module: check the maths (filter design, loudness measurement, true-peak, crossover reconstruction, gain-curve smoothing), the edge cases and the test coverage.
4. Before declaring any milestone done.

**How to brief it:**
- It has no access to this conversation, so give it full context every time: the artefact, the relevant spec section, and what you claim the artefact does.
- Ask it to find errors, unstated assumptions, missing cases and better-known methods, with a confidence rating per finding. Tell it explicitly that agreement isn't useful and that it should argue against the work.
- Never send credentials, tokens or private data.

**How to handle findings:**
- Log each audit to `audit/NN-topic.md`. Include its findings verbatim and your response to each: accepted, rejected, or unresolved, with reasoning.
- Settle factual disagreements with evidence: a test, a measurement or a primary source. Don't defer to either model's confidence.
- Never paste auditor code into the codebase unreviewed.
- Escalate anything still unresolved to me, with both positions summarised.

## Deliverables, in order
1. `research/report.md` plus its audit log
2. `docs/architecture.md`: the module breakdown, data flow (analysis → plan → render → measure loop), and a prototype plan
3. A working Python prototype of the analysis pass and the default M/S multiband chain, with evaluation output
4. Questions for me, collected at the end, not asked piecemeal. Work through everything you can decide yourself first.
