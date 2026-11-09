# Eluate audit: analysis features (src/eluate/features/, noise.py, planner consumption)

Branch audit/2026-11-10, read-only. All signals were synthetic: fixtures/synthetic.py, fixtures/e_signals.py, and the audit's own generator `audit/features/gen.py`, which builds kick + off-beat hats + bass root per bar + pad triads, plus an ambient pad. Scripts and logs are in `audit/features/` next to this file. Each finding cites its script.

Timing caveat: most experiments ran in parallel with each other on 12 cores, so their wall times are inflated. The clean timing table comes from `finish.py`, run alone after a numba warm-up.

Existing gates: `pytest tests/test_feature_*.py tests/test_noise.py tests/test_synthetic.py` → **158 passed, 10 skipped** (23 min 50 s; `pytest.log`).

Severity scale: S1 = wrong processing, S2 = wrong numbers in reports, S3 = crash or misbehaviour, S4 = hygiene or performance, S5 = nit.

---

## Findings (by severity)

### S1: leads to wrong processing

**F1. Beatless material gets a "tracked", *informative* beat grid.**
- **Where:** `structure.py:172-174`, `:822`.
- **What is wrong:** `grid_kind` is "default" (uninformative) only when librosa returns fewer than 4 beats. On noise or a pad, librosa always returns beats, so the grid is "tracked" and `grid_informative` is True. `planner.plan_hits` (planner.py:1873-1874) then uses these random beats as hits, and the auto character uses the invented tempo.
- **Evidence** (`seg_exp.py`, `robust.py`):

  | Input | Tempo | Grid | `informative` |
  |---|---|---|---|
  | ambient pad, 30 s | 135.3 BPM (175.5 BPM at 48 kHz) | tracked | True |
  | pink noise, 30 s | 113.5 BPM (rule "lattice") | tracked | True |

  The pad read `upper_pulse` 124.9 and `kick_pulse` 164.9, with no relation between the two and no kicks.
- **Fix:** mark the grid uninformative when the metrical evidence is weak. Candidate conditions:
  - no kicks, and `kick_consistency` not consistent;
  - the upper and kick pulses have no relation (`metrical.relation is None`);
  - the peak ACF height is below a floor;
  - for a "tracked" grid, a high residual.

  Report the reason.
- **Confidence:** high.

**F2. The key-locked crossovers are moved by a key that has no confidence gate.**
- **Where:** `features/__init__.py:87-89`; `pitch.combined_key` `pitch.py:153-164`.
- **What is wrong:** the tonic is taken whatever its score. The resulting crossover then sets every band split for the whole analysis and plan. The notch guard (tonic and fifth, planner.py:911-915) and resonance's tonic-triad rule also follow this key.
- **Evidence** (`pitch_exp.py`, `robust.py`):

  | Input | Key reported | Score | Crossover |
  |---|---|---|---|
  | pink noise | Cm | 0.29 | follows Cm |
  | drums only | G#m | 0.22 | follows G#m |
  | DC only | Gm | 0.71 | follows Gm |
  | DC only (robust.py, 5 s at 48 kHz) | C# | — | 100 → 110 Hz |
- **Fix:** require an edma score of at least about 0.6 (and/or a margin over the second-best key) before locking the crossovers. Otherwise use `bands.DEFAULT_CROSSOVERS` with `crossover_rule` "fixed (key not confident)".
- **Confidence:** high.

**F3. The combined key rule reads the most common EDM minor progression as its relative major.**
- **Where:** `pitch.py:158-161` (mode always from edma).
- **Evidence** (`pitch_exp.py`):

  | Progression | Wanted | Result | edmm alone |
  |---|---|---|---|
  | i–VI–III–VII (Em C G D) | Em | **G** | Em |
  | i–VI–III–VII (C#m A E B) | C#m | **E** (chord_loop), **A** (audit track) | C#m |
  | Am F C G | Am / C (ambiguous) | C or F | — |
  | Dm Gm A Dm (audit track) | Dm | D major | — |

  Overall score: 10/12 on fixtures.chord_loop and 9/12 on the audit track. Clean diatonic I–IV–V / i–iv–V progressions were all correct (C, G, D#, Bb, E, Am, F#m, Dm, Fm).
- **Consequences:**
  - C#m → E or A moves fc1 from 110 to 80 or 87 Hz.
  - The protected tonic and fifth change from C#/G# to E/B.
- **Fix:** when the edma major winner is the relative major of the edmm minor winner and the two scores are within a margin, report minor (the EDM prior Faraldo's edmm encodes), or report the key as ambiguous and use the fixed crossovers.
- **Confidence:** high for the measurement, medium for the fix.

**F4. Tempo octave and the character boundary.**
- **Where:** structure `_pulse` / `_octave` (`structure.py:262-275`, `:323-353`); `planner.character_choice` (`planner.py:446`, `:1736`, `bpm > AUTO_BPM_FAST`).
- **Evidence** (`struct_exp.py`):

  | Fixture | Detected tempo |
  |---|---|
  | 70 BPM four-on-the-floor with off-beat hats | **140.11** (kick pulse 70, upper pulse 140); `kick_consistency` "tempo half", so nothing is flagged |
  | 140 BPM DnB pattern | 140.09 |
  | 85, 128, 150 and 174 BPM four-on-the-floor | within 0.04 BPM, no octave error |
  | 174 BPM DnB | within 0.04 BPM, no octave error |
  | 87 and 90 BPM hip-hop | within 0.04 BPM, no octave error |

- **What is wrong:**
  - The detected tempo carries a small upward bias of +0.06 %.
  - Because the boundary test is `> 140`, any true 140 BPM track (dubstep, grime, a common tempo) lands randomly in "fast" or "moderate".
  - The 70 → 140 doubling is the known limitation "a tempo read at twice the true one is not corrected" (audit 13 #6). It doubles the number of hits and selects "fast".
- **Fix:**
  - Add a hysteresis band of ±1 % around AUTO_BPM_FAST (and around SLOW) for detected tempos, or round the detected tempo to 0.1 BPM before comparing.
  - When the kick pulse is exactly /2 of the chosen tempo and the upper pulse comes from off-beat hats, prefer the kick octave, or at least flag it.
- **Confidence:** high for the measurement, medium for the impact.

**F5. Kick candidates are lost under a sustained (non-sidechained) bass plus pad.**
- **Where:** `transients.py:108` (KICK_FLUX_DB = 10), `:151`.
- **Evidence** (`kick_track.py`, `kick_flux.py`):

  | Audit track content | Kicks found |
  |---|---|
  | kick, hats, bass, pad | **20/43** |
  | bass removed | 43/43 |
  | pad removed | 43/43 |

  - The band-mean dB flux at the true kicks ranged 7.9 to 61.7 dB.
  - 23 of the 43 kicks peaked below the 10 dB gate.
  - All the kicks that were found scored about 1.0, so the score stage is fine; only the candidate gate loses them.
- **Consequences:**
  - `align_to_onsets` needs half the beats to have a kick (`structure.py:551`), so alignment is skipped silently.
  - `usable_kick_onsets` feeds the kick detective, clipper and notch duck (KICK_MIN_RATE 0.5/s).
  - The kick mask for bass root reads these onsets.
- **Fix:** weight the flux by bins whose level rises above their own recent median (sustained-tone bins contribute about 0), or lower the gate to about 7 dB and let the share/sweep/click score reject non-kicks (it already does: bass notes, toms and snares scored 0).
- **Confidence:** high.

**F6. The beat grid trails the attack by 18 to 22 ms when the kicks are not on every beat.**
- **Where:** `structure.py:538-559` (alignment needs kicks on at least half the beats).
- **Evidence** (`struct_exp.py`):

  | Pattern | Median grid lag behind the true beat |
  |---|---|
  | DnB at 174 BPM | 18.1 ms |
  | DnB at 140 BPM | 17.4 ms |
  | hip-hop at 87 BPM | 22.2 ms |
  | hip-hop at 90 BPM | 21.5 ms |
  | four-on-the-floor | 0.9 to 5 ms |

- **What is wrong:** the docstring's own failure mode applies: the next attack falls inside the previous hit. Every hit's maximum is then the following beat's transient (`envelope.hit_edges`).
- **Fix:** when the kicks are too sparse, align to the SuperFlux onsets nearest the beats, using the same median-offset rule.
- **Confidence:** high for the lag, medium for the impact.

### S2: wrong numbers in reports / defeated safeguards

**F7. Planner: S-only resonances are detected (audit A4) but never cut.**
- **Where:** `planner.py:830-832`, `:1011-1012` ("S-only resonance; notch stage is not channel-aware").
- **Evidence** (`reson_exp.py` case 5): the ring was found correctly with channel "S", f0 5001 Hz. `dense_exp.py` shows two S-only candidates skipped for this reason.
- **What is wrong:** the spec's channel contract is implemented in features but dropped by the consumer. This matters for the user-visible promise.
- **Fix:** give plan.Notch a channel, or document the S-only gap in the report.
- **Confidence:** high.

**F8. Noise floor: louder hiss is *not* flagged.**
- **Where:** `noise.py:94-98`, `:106-107`; consumer `planner.py:773-775`.
- **What is wrong:** quiet frames must be more than 35 dB below the p95 loud level. A hiss at or above that level makes no frame quiet, so the result is −inf, no flag, and `noise_floor_db` None, which removes the top-shelf noise cap (`planner.py:1289-1297`).
- **Evidence** (`noise_exp.py`): 40 s track peak-normalised, with a 5 s gap and a 3 s tail. The floor is the BS.468-weighted reading; −60 dB is the flag threshold.

  | Hiss added | Floor (BS.468-weighted) | Flagged at −60? |
  |---|---|---|
  | white −70 dBFS | −61.1 dB | no |
  | white −60 dBFS | −51.1 dB | yes |
  | **white −50 dBFS** | **−inf** | no |
  | **pink −50 dBFS** | **−inf** | no |
  | **white −40 dBFS** | **−inf** | no |

- **Fix:** when no frame qualifies, fall back to the lowest-level 5 % of non-silent frames (or minimum statistics) and report that the floor may include music. Never return −inf when non-silent frames exist.
- **Confidence:** high.

**F9. The keyeq gate passes on a half-tempo grid.**
- **Where:** `pitch.py:122` (NSDF_SPAN 0.55–0.95), `:405`, `:411-420`.
- **Evidence** (`bass_exp.py`): kick_sub fixture with the root changing every beat, analysed on a grid at half the tempo.
  - Accuracy against the root at each grid beat: 0/16.
  - Median confidence 0.91.
  - NSDF agreement 1.00.
  - keyeq_eligible: True.

  The "independent" NSDF reads the same late part of the beat as the weighted root, so the F2 fix cannot see the failure that motivated it.
- **Impact:** limited today, because keyeq_eligible is report-only (no planner consumer).
- **Fix:** also require agreement over the first half (0.1–0.45 of the beat), or require that the per-half roots agree.
- **Confidence:** high.

**F10. Hard-panned with bleed reads as "mono".**
- **Where:** `stereo.py:369` (absolute VAR_FLOOR 1e-12), `:400`.
- **Evidence** (`stereo_exp.py`):

  | Input | Correlation | Decoherence | Measurable | S/M |
  |---|---|---|---|---|
  | R = 0.001·L | 1.0 | 0 | True | 0 dB |
  | R = 0 | NaN | 1.0 (default) | False | — |

  Correlation is scale-invariant, so a −60 dB bleed escapes the "hard-panned: not measurable" rule (spec A12). The multiband stage then applies full high-side makeup (it scales by 1 − decoherence) on a hard-panned band.
- **Fix:** add a relative test: not measurable when min(var_L, var_R) / max(var_L, var_R) < 1e-4. Alternatively, flag disagreement between the correlation (1.0) and an S/M near 0 dB.
- **Confidence:** medium.

**F11. Musical demotion of rings by pitch class.**
- **Where:** `resonance.py:651-652`; planner guard `planner.py:363`, `:1009`.
- **Evidence** (`reson_exp.py` 1 vs 1b): the spec gate's own 7 kHz ring (−10 cents from A8) becomes "musical" with depth 0.25 as soon as the key is Am. The planner guard (±50 cents of tonic or fifth, for the whole track) would skip it entirely.
- **What is wrong:** between the ±35-cent tonic-triad rule (3 of 12 pitch classes, 70 % of each semitone) and the ±50-cent tonic/fifth guard, about 17 % of all frequencies can never get a full-depth notch, independent of whether the energy is a chord tone.
- **Fix:** apply the pitch-class rules only below about 2 kHz (where chord tones live), or only when a ladder or partial is also present.
- **Confidence:** medium (design trade-off).

**F12. Notch false positives on dense synthetic material.**
- **Evidence** (`dense_exp.py`): a 48 s loop (kick, hats, bass, pad, supersaw ±12 cents, 16th arpeggio) with **no** resonance.
  - Feature level: 7 notch candidates (4 "resonance", 3 "musical").
  - Planner: rendered 3 notches, with max cuts 1.0 dB at 1832 Hz, 0.5 dB at 2198 Hz and **3.0 dB at 19263 Hz**.
  - Blocked: 2 "suspect" by span (1977 Hz with a 6 dB cut proposed, 13.3 kHz) and 2 S-only.
  - Robustness clips: the kick's 1.5 kHz click was flagged "resonance" on the 2 s and 5 s clips, 13.7 kHz on 5 s, and 12.5 kHz on the 12 s / 44.1 kHz clip.
  - Controls: e_signals.dense_track gave 0 candidates. Pink noise alone gave 0.
- **What is wrong:** the planner's defensive rules (span > 60 %, excess > 20 dB, NOTCH_MAX) carry most of the false-positive rejection, not the detector.
- **Fix:**
  - Require persistence in more than one section, or a minimum duration in seconds, for short files.
  - Treat bands within an octave of the top of the kept range (f > 0.8 · f_hi) with a two-sided baseline only.
- **Confidence:** medium. The 19.3 kHz origin was not isolated.

**F13. Spurious 0.26 s span at t = 0 for rings below 500 Hz.**
- **Where:** the `resonance.py:686-692` short pass, combined with the zero padding in `_Res._power`.
- **Evidence** (`reson300.py`): rings at 300 and 450 Hz present over 9–21 s reported spans [(0.0, 0.26), (8.5, 21.6)]. As a result:
  - `start` = 0.00 instead of 9.0;
  - the excess is above 0 at grid frame 0, so a brief cut at the file start.

  200 Hz and 7 kHz were clean.
- **Fix:** drop the short-pass peaks in frames whose window is more than 25 % padding.
- **Confidence:** high for the symptom, medium for the cause.

**F14. Two per-band crest implementations disagree on edge-only frames.**
- **Where:** `crest.py:234-235` vs `stages/multiband.py:978-981`.
- **Evidence** (`crest_exp.py`): the same 20 s signal.
  - All finite values are identical (max |diff| 0.0 dB).
  - Band 4 / S: features gives NaN, while band_crest gives 26.19 dB, taken from non-interior (pre-ring) frames.
  - planner.py:1694 reads features' crest, while solver and evaluate read band_crest. Before/after and plan figures can therefore differ in which bands are measured.
- **Fix:** make one function call the other, or share `_interior` and the fallback rule.
- **Confidence:** high.

### S3: crash / misbehaviour

**F15. Planning crashes on silent or sub-1 s input.**
- **Where:** `planner.py:772` writes `measured_dbfs = float(nan)`; `plan.py:339` then rejects it.
- **Evidence** (`robust.py`): `silence_10s` and `dur_0.5s`: features.analyse succeeds, then `planner.plan_from_analysis` raises `ValueError: input measured_dbfs contains NaN or Inf`.
- **Fix:** write None when the value is not finite, and give a clear "nothing to master" message for silence.
- **Confidence:** high.

**F16. Short clips: tempo error and fabricated structure.**
- **Evidence** (`robust.py`): at 128 BPM:

  | Clip length | Tempo read | Error |
  |---|---|---|
  | 5 s | 131.29 | +2.6 % |
  | 12 s | 128.77 | +0.6 % |
  | 60 s | 128.01 | — |

  On the 2 s and 5 s clips a "resonance" notch was flagged on the kick click (see F12).
- **Fix:** below about 16 beats, report the grid as uninformative.
- **Confidence:** medium.

**F17. Kick score depends on the sample rate.**
- **Where:** `borrowed.attack_low_share` window `n = 2**round(log2(0.0464 fs))` (`borrowed.py:414`): 46.4 ms at 44.1 / 88.2 kHz, 42.7 ms at 48 / 96 kHz.
- **Evidence** (`clap_exp.py`): kick plus a 12 kHz low-passed snare at 0.4 on beats 2 and 4.

  | Sample rate | Kicks found under the snare | Share on those kicks |
  |---|---|---|
  | 44.1 kHz | **5/16** | 0.72 |
  | 96 kHz | **16/16** | 0.77 |

  With a white-noise snare (`trans_exp.py`): 16/32 at 44.1 and 48 kHz, 32/32 at 96 kHz.

  A clap of any level was fine at both rates (32/32).
- **Fix:** use a fixed-duration window (`round(0.0464 fs)`) with zero-padding to a power of 2.
- **Confidence:** medium.

**F18. A user `--bpm` that is slightly wrong drifts the grid silently.**
- **Where:** `structure.py:175-192` (fixed period, phase from the median).
- **Evidence** (`struct_exp.py`): `--bpm 172` on a 174 BPM loop. `deviation_pct` of 1.15 % is recorded, but the constant grid drifts about 0.37 s over 32 s. No warning is raised.
- **Fix:** warn when |deviation| is between 0.3 and 3 %.
- **Confidence:** high (design choice; "never second-guessed").

### S4: hygiene / performance

**F19. numba cold start dominates first runs.**
- **Where:** `desktop/engine/entry.py:8`; `desktop/src-tauri/src/lib.rs:463` (the shell sets NUMBA_CACHE_DIR to the app cache, correctly).
- **Evidence** (`bass_exp.py`): first `librosa.pyin` call with an empty cache took **62 s**, the next call 0.73 s. A new process with a populated cache took **11.3 s** for its first call. Both figures were measured under CPU contention, so the true values are lower.
- **What is wrong:** the first master after install (or after the OS clears the temp/cache dir) pays this cost. The fallback cache dir is `%TEMP%/eluate-numba`, which Windows Storage Sense can purge.
- **Fix:** warm the cache in the installer or on first launch, and show "preparing analysis" in the UI.
- **Confidence:** medium (contention).

**F20. The analysis cache version is manual.**
- **Where:** `cli.py:150-155`, `:312-326`.
- **What is wrong:** reuse checks ANALYSIS_VERSION (= 1), SHA-256, requested crossovers, bpm and R_ref. Nothing ties the cache to the code of `features/*`, `noise.py`, `planner.key_crossovers` or `planner.analyse`. Any edit that forgets the bump reuses stale analyses. The repo has only 2 commits, so drift since version 1 cannot be traced. Crossover mismatch between plan and analysis is guarded (`planner.py:684-686` raises).
- **Fix:** add a hash of those sources (`cli.source_hashes` already exists) to source.json and to `_matches`.
- **Confidence:** high.

**F21. Split CLI path: `--bpm` given at `analyse` is treated as a detected tempo at `plan`.**
- **Where:** `planner.py:1766-1771`.
- **What is wrong:** `eluate plan` has no --bpm, so `controls.bpm` is None and the source becomes "analysis". A user tempo below 70 then resolves to "moderate", not "slow", and the kick detective auto setting differs from `eluate master --bpm`.
- **Fix:** use `src = "user"` when `plan_inputs.metrical.rule == "user"`.
- **Confidence:** high (static).

**F22. borrowed.py licence and attribution.**
- **Where:** `borrowed.py:1-4`.
- **What it is:** SPDX `GPL-3.0-or-later`, copied from the maintainer's own TRANSMUTATION and drumbusmaximus/SKIN. That code is:
  - SuperFlux onsets;
  - match_onsets;
  - harmonic salience;
  - the ladder / HNR;
  - the McLeod NSDF;
  - the DrumIdentity low share.

  No third-party code is borrowed: it is not madmom, essentia or librosa code. The algorithms are published (Böck & Widmer SuperFlux, McLeod & Wyvill, Hermes SHS).
- **Licence compatibility:** GPL-3.0-or-later combines lawfully with AGPL-3.0 (GPLv3 §13), so there is no incompatibility. The header differs from the project's AGPL-3.0-or-later and its "the Eluate contributors" copyright line, and THIRD_PARTY_NOTICES (generated from installed packages only) does not mention it. Not legally required for self-authored code.
- **Data from Essentia:** the Faraldo edma/edmm profiles in `pitch.py:86-91` match Essentia key.cpp, which is AGPL. The values are numeric data from a published paper and are credited in the docstring.
- **Fix:** relicense the header to AGPL-3.0-or-later (same author), or add a "Portions" note to NOTICES.
- **Confidence:** high.

### S5: nits

**F23.** `tilt.py:127`: for a mono file the S LTAS is floored at −200 dB everywhere, so `slope_db_oct["S"]` and `sub_shelf_db["S"]` read 0.0 instead of NaN or a flag (`tilt_exp.py`). Not consumed by the planner (it reads stereo only).

**F24.** `noise.py` docstring: "+12.2 dB at 6.3 kHz" is correct. Note that the BS.468 weighting puts white hiss about +9 dB above its unweighted level (−60 dBFS white reads −51.1 dB). The −60 dB flag is therefore about −69 dBFS of white hiss, worth stating in the report text.

**F25.** Silence produces a librosa "Trying to estimate tuning from empty frequency set" UserWarning (`pitch.py:185`). Pass `tuning=0.0` when the signal is silent.

---

## Verified correct (with numbers)

- **Tilt** (`tilt_exp.py`):
  - Pink noise reads −3.003 to −3.029 dB/oct (M/S/stereo, 44.1/48/96 kHz, 3 seeds): inside the ±0.1 gate.
  - Synthetic −4.5 dB/oct reads −4.498 (gate ±0.1); −6 reads −5.999; 0 reads +0.004.
  - Method (audit F5): "mean PSD per band" is confirmed. A band-summed control reads about 0.0 on pink (−0.02 to +0.03), so the implementation uses the right one.
  - Correlated 0.9 pink: −3.01 for M, S and stereo.
  - Sub shelf: a +6 dB / −6 dB 90 Hz zero-phase low boost reads +5.42 / −5.41 against analytic +5.71 / −5.50.
  - Power averaging: 50 s pink at −30 plus 10 s of −6 dB/oct at −10 reads −4.80 (the loud 1/6 of the track dominates; equal levels give −3.10). This is expected for a power LTAS; worth stating in the report.
- **Resonance gate** (`reson_exp.py`):
  - 7 kHz Q12 +9 dB ring over 40 % of the track: f0 6997.3 (−0.04 %), start 8.85 (−150 ms), end 21.07 (+70 ms). All within the ±1 % and ±200 ms gates.
  - Q half-gain 7.21 against analytic 12/√2.818 = 7.15 (+0.8 %).
  - Other rings:

    | Ring | Q read | Analytic | Error |
    |---|---|---|---|
    | 1 kHz Q8 | 4.94 | 4.77 | +3.6 % |
    | 300 Hz | 5.92 | 5.96 | — |
    | 3150 Hz Q20 +6.5 | 12.41 | 13.76 | −10 % (a wider notch than needed) |

  - Pink noise alone: 0 detections.
  - Sustained 8-harmonic notes at 110, 220 and 523 Hz: all "tonal", depth 0.
  - Stiff string (B = 0.005), church-bell ratios, ±0.4 % detuned harmonics: all "tonal", depth 0; ladder 0.21 to 0.95 except two bell partials at 0.0 / 0.01 that the line rule caught.
  - Sub rules: 60 Hz → sub_exempt. 120 Hz at 40 % persistence → sub_exempt. 120 Hz at 81 % → resonance with depth 0.5.
  - M/S: an S-only ring at S/M −10 dB was found in S only (`channels_seen ['S']`); an M-only ring was reported M.
- **Transients** (`trans_exp.py`):
  - SuperFlux on a 72-event drum loop: P = R = 1.000 at 44.1 and 48 kHz. At 96 kHz SuperFlux-only recall is 0.958; after the kick union P = R = 1.000.
  - Onsets at 44.1 kHz against 48 / 96 kHz: median |Δ| 0.36 / 0.39 ms, but max 3.7 / 4.3 ms. This is outside the "within 1 ms" goal for a few events (S5; the hop is scaled by a power of 2, not exactly).
  - Kicks 44.1 vs 48 kHz: max |Δ| 0.02 ms.
  - Kick rule:
    - kick with click 16/16; without click 16/16; short 120 → 55 Hz sweep 16/16;
    - floor toms (90 and 70 Hz, 8 % drop): 1 false kick per 16;
    - hard-attack bass notes (55 Hz / 1 ms, 41 Hz / 0.2 ms): 0;
    - snare: 0;
    - clap on 2 and 4: 32/32.
- **Structure** (`struct_exp.py`, `seg_exp.py`):
  - 85, 128, 150 and 174 BPM four-on-the-floor, 174 DnB, 87/90 hip-hop: tempo within 0.1 % with no half/double error (the 70 and 140 exceptions are in F4).
  - Foote segmentation on sectioned_track at 128 and 140 BPM (−12/0/−6/0/−9 dB steps), chords-only, and 2 dB steps: all 4 boundaries at 0.00 bars error, no extras, downbeat phase correct.
  - `--bpm` path: the period is exactly 60/bpm, `rule` "user", `detected_bpm` and `deviation_pct` are reported.
  - `hit_edges` keeps only beats strictly inside (0, duration).
  - Beats with a silent 10 s intro: see the robustness table.
- **Pitch** (`bass_exp.py`):
  - Bass root on kick_sub at 126 BPM: 32/32 (1.000) with the true kicks and with detected kicks. 174 BPM with roots E1–D1: 32/32.
  - Off-beat, eighths and missing-fundamental basslines: 32/32.
  - Spec gate (≥ 95 %): met.
- **Stereo** (`stereo_exp.py`):

  | Signal | Correlation | S/M | Decoherence | Side strip |
  |---|---|---|---|---|
  | mono | 1.0 | −inf | 0 | 100 Hz |
  | uncorrelated | about 0 | 0 dB | 1.0 | 150 Hz |
  | anti-phase | −1 | +inf | 1 | 150 Hz |
  | R = 0 | — (not measurable) | — | 1.0 | 150 Hz |
  | corr 0.5 | 0.50 | −4.8 dB | 0.50 | 150 Hz |

  Side-strip rule with a stereo sub below fc (60 / 80 / 120 / 200 Hz): 70.2 / 88.4 / 140.3 / 150 Hz (clipped). Mono sub with wide content above 100 Hz: 100 Hz. All inside the 60–150 Hz bound.
- **Noise** (`noise_exp.py`):
  - BS.468 implementation within 0.09 dB of the published table at 21 frequencies (31.5 Hz – 31.5 kHz).
  - Calibration: white −60 dBFS reads −51.11 against an analytic mean of −50.60 (20th-percentile bias −0.5 dB).
  - Clean track: −189 dB, not flagged.
  - White and pink −60 hiss: flagged. White −70: not flagged.
  - A quiet low-register pad intro and a noise riser were not false positives. A bright quiet musical tail was **not run** (the doc already admits it reads as noise).
- **Crest:** values identical to multiband.band_crest wherever both are finite (F14 covers the NaN pattern).
- **Robustness:** no crashes in features.analyse on any of the 19 inputs (results below).

### Robustness sweep (features.analyse with fcs_from_key=planner.key_crossovers; then planner.analyse + plan_from_analysis)

| Case | Result |
|---|---|
| silence 10 s | tempo 120 default grid, key None, fixed crossovers, all-NaN loudness. **Plan crash (F15).** |
| DC only 5 s | key C# moves the crossovers (F2); S/M −inf / NaN. Plan OK. |
| music + 0.2 DC | 128.26 BPM; DC causes no NaN. OK. |
| mono L = R | S/M −inf (documented); decoherence 0. OK. |
| hard-panned R = 0 | correlation NaN → decoherence 1.0, not measurable; side strip 150 Hz. |
| 0.5 s | 1 beat, default grid. **Plan crash (F15).** |
| 2 s | 129.7 BPM; 4 notch candidates including the kick click at 1.5 kHz (F12). |
| 5 s | 131.3 BPM (F16); 2 false "resonance" notches. |
| 12 s / 60 s | 128.77 / 128.01 BPM. |
| 44.1 / 48 / 96 / 192 kHz, 12 s | 128.79 / 128.77 / 128.76 / 128.77 BPM; consistent. 44.1 kHz had one false 12.5 kHz "resonance"; 192 kHz found 11 kicks against 13. |
| ambient pad 30 s | **tempo 175.5, tracked grid, informative (F1).** |
| tempo change 120 → 140 at 20 s (40 s) | tracked grid follows it (beat intervals 0.500 → 0.425 s, i.e. 120 → 141 BPM). `tempo_bpm` is one number, 130.5, which feeds the character choice. Boundary at 20.0 s is correct. |
| 7-beat bars, 128 BPM (30 s) | 128.02 BPM; 4/4 is assumed (declared). A boundary is invented at 15.5 s on a uniform loop, and a false "resonance" is flagged at 9006 Hz (F12). |
| peak −40 dBFS (raw, not normalised) | analysis is OK; upper bands fall under the absolute −60 dBFS gates, so `band_corr` and S/M read NaN. The planner path normalises first, so this is harmless there. |
| 10 s digital silence then music | constant grid extended back over the silence (21 beats in silence, harmless). First boundary at 8.2 s, not 10.0 s (about 1 bar early, at the edge of the ±1-bar snap). Section 0 loudness is −inf (silent). |

Timings for these four cases (finish.log, after warm-up): 27.4 / 21.0 / 14.0 / 29.6 s.

No negative durations. No NaN/Inf other than the documented ones:
- loudness on silence;
- `crest_p95` NaN for gated-out S bands;
- `band_corr` NaN when not measurable, with flags;
- `metrical.*` NaN on beatless input;
- `kick_consistency.ratio` NaN.

---

## Timing (60 s, 48 kHz, audit track, numba warm; `finish.py`, run alone)

| Feature (in features.analyse order) | Seconds | Share |
|---|---|---|
| loudness (meter series + measure) | 3.28 | 7 % |
| key, run before the split (chroma CQT) | 5.37 | 11 % |
| tilt | 0.75 | 2 % |
| crest (includes the 5-band split) | 2.88 | 6 % |
| stereo | 1.28 | 3 % |
| transients (SuperFlux, librosa onset, kicks) | 2.07 | 4 % |
| structure (metrical ACF, beat_track, MFCC/chroma, Foote) | 2.03 | 4 % |
| **pitch: bass root** (pyin on L/R/M in 30 s blocks + SHS + NSDF) | **16.39** | **33 %** |
| **resonance** (3 channels x 3 STFT resolutions + short pass) | **15.23** | **31 %** |
| per-section loudness | 0.17 | <1 % |
| **Total** | **49.5** (cProfile run of features.analyse: 50.1) | |

cProfile (features.analyse, 50.1 s):
- `pitch.bass_root` 20.8 s cumulative, of which `librosa.pyin` (9 calls) is 17.1 s. Inside pyin, the `__pyin_helper` beta-distribution pmf / localmin work is 12.1 s and the Viterbi 3.6 s.
- `resonance.analyse` 18.2 s, of which `_Res.__init__` (12 instances) is 10.8 s: `_power` 5.0 s, `_baseline` 4.7 s (median in a Python loop), `peaks` 2.4 s, `_ladder` 2.3 s.
- Top self-time: pocketfft 4.4 s, the Viterbi helper 3.2 s, `ndarray.partition` (medians) 2.8 s.

Cheap wins (S4):
1. pyin: run it once on M and skip L/R when L and R correlate above 0.99 in the low band. Today R is skipped only when bit-identical.
2. resonance: skip the "stereo" channel when both M and S were analysed. Its spectrum is (P_M + P_S)/2, so it adds about 1/3 of the cost and is only needed for rings visible in neither channel alone.
3. `_baseline`: vectorise the per-grid-point medians with a sorted sliding window.

Cold start (F19): add 62 s on a first-ever pyin call, or 11 s on a first call per process with a populated numba cache (both measured under contention).

---

## Open questions

1. Should a low-confidence key fall back to the fixed crossovers (F2), and should the relative-major ambiguity default to minor for EDM (F3)? This is a maintainer decision on the Faraldo prior.
2. Is the notch stage's channel awareness (F7) planned? Until it is, the S detection is report-only, and the report should say so.
3. AUTO_BPM_FAST = 140 with a strict ">": is 140 BPM meant to be "moderate"? Detection bias puts it on either side (F4).
4. What criterion should mark a tracked grid uninformative (F1)? The spec's "grid_kind flags a beatless default grid" never fires on real beatless audio because librosa always returns beats.
5. Noise: should a track whose quiet passages are themselves the noise (F8) be reported as "floor ≥ X" rather than −inf?
6. Was ANALYSIS_VERSION bumped for every analysis change since the cache was introduced? The history (2 commits) cannot show it (F20).
7. The 19.3 kHz rendered notch on the dense loop (F12): baseline edge effect or a real spectral feature of the synthetic hats? Not isolated.
