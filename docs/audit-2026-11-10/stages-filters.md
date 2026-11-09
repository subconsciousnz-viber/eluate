# Eluate audit: filters and stage alignment (mathematical verification)

Branch audit/2026-11-10, read-only. Python 3.12.13, numpy 2.5.4, scipy 1.18.1 (Windows x64).
Scripts and logs: `<scratch>/audit/stages-filters/s1_inputeq.py … s10_extra2.py` with `s1.log … s10.log`;
`pytest.log` = the seven in-scope test files, 171 passed in 341.7 s. Every number below is measured
by the named script unless marked "derived".

Severity: S1 audible/wrong audio, S2 wrong numbers, S3 crash/misbehaviour on plausible input,
S4 hygiene/perf, S5 nit.

---

## Findings

### F1 (S1, confidence medium on audibility, high on the numbers) — zero-phase 100 Hz crossover pre-rings a kick by −15 dB, 6–10 ms ahead, once the sub band is gained
- `src/eluate/bands.py:141-153` (`_zero_low`, phase "zero"); default phase `"zero"` for every crossover in `src/eluate/plan.py:496` (MultibandPlan), and the planner never selects "minimum" (grep of planner.py / solver.py / cli.py: no `phase=`). `docs/spec-draft.md:98` says the lowest crossover "is decided by listening in gate D with `minimum` as the expected winner"; that decision was never taken.
- Evidence (`s6_bands.py` 6c, synthetic kick 50→110 Hz sweep + click after 0.5 s of silence, fs 48 k):
  - sub band alone: pre-onset energy −27 dB re the sub band, pre-onset peak −8.9 dB re kick peak, above −40 dB from 8.0 ms before onset, above −60 dB from 10.2 ms, above −80 dB from 16.7 ms.
  - sum with the sub band at −3 dB (what a 3 dB multiband GR does): pre-echo peak −19.6 dB re kick peak, above −60 dB 9.3 ms before the onset.
  - sum with the sub band at −6 dB (the compressor's cap, MULTIBAND_MAX_GR_DB): pre-echo peak −14.9 dB, above −40 dB 6.3 ms and above −60 dB 9.8 ms before the onset; identical in the 5-band default split.
  - phase "minimum": exactly zero pre-onset output in every case.
  - For scale: the zero-phase 24 dB/oct HPF the maintainer removed for its pre-echo (R7e) measures −21 dB / 41 ms (Q 0.707) on the same kick (`s1_inputeq.py` 1d2). The multiband pre-echo is 6 dB louder and shorter.
- The band sum itself stays exact (4.4e-16); the pre-echo appears only when bands get different gains (multiband, speclim, saturate's change is also zero-phase but small).
- Minimal fix: run the lowest crossover in "minimum" (`phase=["minimum","zero","zero","zero"]`) for multiband/speclim/saturate (allpass sum, verified flat to 2.7e-12 dB, `s6` 6f), or make it a listening-gated plan field; at minimum document the measured pre-echo in gain-staging.md.

### F2 (S2, confidence high) — notch merge/overlap rules use analog band edges; they are wrong above ~fs/4 at the planner's low Q
- `src/eluate/stages/notch.py:130-133` (`band_edges`), used by `merge_candidates` (136-173) and `_check_merged` (176-181). The edges are the analog geometric pair f0·(√(1+1/4Q²) ∓ 1/2Q); the filter is designed through the bilinear warp (`_notch_bandwidth` is exact), so the realised half-cut edges are the warped ones.
- Evidence (`s2_notch.py` 2a, `s10_extra2.py` 10a; 44.1 k, −6 dB, Q 2 = planner NOTCH_Q_LO): edge error (digital vs `band_edges`) at 8 k −1.9 %/−1.1 %, 12 k −4.9 %/−3.0 %, 16 k −11.2 %/−6.8 %, 18 k −16.8 %/−10.2 %; Q 4 at 18 k −4.2 %/−3.2 %; Q 8 ≤ 1 %. At 0.45 fs Q 1: analog 12265–32110 Hz vs digital 2151–21996 Hz.
  - Pair 12000/19695 Hz Q 2: `_check_merged` accepts (analog bands disjoint) while the digital −3 dB bands 8912–14912 and 11756–21603 Hz overlap by 3156 Hz; pair 12000/16730 Hz Q 3 overlaps by 573 Hz and is accepted.
- Consequence: wrong grouping / false "overlap" errors near the top; the composite clamp still holds the summed cut at exactly −6.0000 dB (verified, 2b), so no cap is breached. The notch width itself is exact: |width/(f0/Q) − 1| ≤ 5.1e-4 over f0 40 Hz…0.45 fs, Q 1…50, g −1…−6 dB, three rates.
- Minimal fix: map the analog edges through the warp used by `_notch_bandwidth`: `f_edge = fs/π · atan(Ω± · tan(π f0/fs))`, Ω± = √(1+1/4Q²) ± 1/2Q (and invert the same way when the merged notch is rebuilt from lo/hi). The resonance detector runs up to 0.95·fs/2 (`features/resonance.py:165`), so the region is reachable.

### F3 (S3, confidence high) — `bandpass_zp` raises `ValueError: math domain error` for low-Q notches above ~fs/4
- `src/eluate/stages/notch.py:106`: `math.atanh(alpha)` with alpha = sin(w0)/(2 q_bq). For the biquad Q that `design_q` returns at high f0 and low notch Q, alpha ≥ 1 (a2 < 0: the poles are real; the filter is still stable, pole radius 0.2–0.55).
- Evidence (`s9_extra.py` 9b): alpha ≥ 1 from f0/fs ≥ 0.225 at Q 1 (10.8 kHz at 48 k), ≥ 0.335 at Q 1.5, ≥ 0.45 at Q 2 (19.8 kHz at 44.1 k); never for Q ≥ 3. `notch.apply` with Notch(12000, 1.0) at 48 k, Notch(15000, 1.5) or Notch(10000, 1.0) at 44.1 k raises; `Plan` validation accepts them (only f0 < fs/2 and Q ≥ 1 are checked). Reachable from the planner through `merge_candidates`, which floors the merged Q at 1 (planner NOTCH_Q_LO is 2, but merged notches drop below it): 18 k + 20 k Q 1 at 44.1 k merges to f0 18 974 Hz Q 1.000, `Plan` accepts, `apply` raises (2c).
- Minimal fix: derive the pad from the real pole radius: `r = max|roots([1, a1, a2])|; pl = max(PAD_MIN, ceil(PAD_TAU / -ln r))` (17 samples for the 12 kHz Q 1 case instead of an exception). Optionally reject Q < 3 above fs/4 at plan validation with a clear message.

### F4 (S2, confidence high) — the band saturator's sub-band leakage is odd-order IMD, not only even-term rectification, and the 24 dB/oct zero-phase HP leaves −33 dB of it at 12 dB drive
- `src/eluate/stages/saturate.py:20-23` (docstring: "the even term also rectifies … so the change is high-passed at the band's lower crossover") and `:103` (the HP is the complement of a zero-phase 2nd-order Butterworth: |HP| = r⁴/(1+r⁴): −8 dB at 0.9 fc, −19 dB at 0.6 fc, −42 dB at 0.3 fc, `s3` 3a/3b).
- Evidence (`s4_saturate.py` 4b; band 1 = 100–400 Hz, tones 150.15 + 180.18 Hz at 0.25 each, harmonic 1):
  - residual below 100 Hz re the fundamentals: drive 2 dB −92 / −76 / −66 dB (sym 0 / 0.3 / 1); 6 dB −51 / −50 / −47 dB; 12 dB −33.5 / −33.3 / −31.6 dB. It is nearly independent of symmetry: the products are 5th/7th/9th-order odd differences (3f1−2f2 = 90 Hz passes the HP at −8 dB; 4f1−3f2 = 60 Hz at −47 dB re fundamentals at 12 dB drive even with sym 0; 5f1−4f2 = 30 Hz −81 dB).
  - the even term's own products at 12 dB drive, sym 0.3: f2−f1 (30 Hz) −68 dB, 2(f2−f1) (60 Hz) −47 dB, f1+f2 (330 Hz) −27 dB, 2f1 −38 dB, 2f2 −52 dB.
  - in-band non-fundamental power: −43.5 / −23.3 / −15.3 dB at 2 / 6 / 12 dB drive (sym 0; 3rd-order 2f2−f1 at −15 dB dominates); sym 1.0 adds 1–2 dB.
  - band level change on the two-tone: −0.19 / −2.03 / −6.83 dB.
- Consequence: "the sub band never saturates" holds for the sub band's own signal, but a driven band 1 injects sub-region intermodulation at −33 dB (12 dB drive) / −51 dB (6 dB) re its fundamentals. Minimal fix: document the odd-order products; if sub cleanliness matters, high-pass the change with a steeper or higher-order zero-phase filter (two cascaded LR4 complements, or fc·1.2), or keep band-1 drive ≤ 6 dB (the spec's own "default drive cap 6 dB until calibrated").

### F5 (S4, confidence high) — IIR tails in digital silence run in subnormal range: 9–21× slower
- `src/eluate/stages/inputeq.py:266` (causal HPF, every zero-phase render with a high-pass), `:276` and `bands.py:141-153` (sosfiltfilt over the zero pads), `bands.py:229` (minimum phase), `glue.py:320`.
- Evidence (`s8_hygiene.py` 8c, `s9_extra.py` 9a; 20 s stereo, a 48-sample burst then silence, vs 20 s of noise): inputeq HPF 1440 vs 161 ms (8.9×); inputeq minimum 3302 vs 158 ms (20.9×); `bands.split` zero 6793 vs 680 ms (10.0×); minimum 14 887 vs 1395 ms (10.7×); multiband neutral 3956 vs 1921 ms (2.1×). After the burst the HPF output has 1 303 124 subnormal samples and zero exact zeros in the last 10 s (min |y| 4.6e-319).
- Minimal fix: filter block-wise with `zi` and zero the state (and output) once the block input is silent and max|zi| < 1e-150; this keeps "silence stays silence" bit-exact. (A DC dither of 1e-30 would also work but breaks the bit-exact silence/identity tests.)

### F6 (S5) — docstring precision in inputeq.py
- `src/eluate/stages/inputeq.py:45-47`: "within 0.005 dB of the specification at ±3 dB" is true (worst 0.00427 dB over gains ±3, corners 60–250 Hz / 1–16 kHz, the plan's Q ranges, fs 44.1/48/96 k; `s1` 1b) but the high shelves may run at +4.5 dB (M) and +6 dB (S): deviation 0.0143 dB and 0.0334 dB. "Exact for the low shelf at Q 0.5" is false: 1.07e-3 dB (at 1.9 fc). Harmless because `response_db` returns the real double-pass curve (matches the FFT of `apply()` to 2.7e-10 dB, 1f) and the planner fits on it.

### F7 (S5) — band shelf depth at narrow widths
- `src/eluate/stages/inputeq.py:25-30` "flat at g between the corners … half the dip at each corner when they are far apart". Measured (`s9` 9c, −3 dB request, lo 300 Hz): width 1/3 oct deepest −1.01 dB (corners −0.96), 1/2 oct −1.21, 1 oct −1.80 (corners −1.28), 2 oct −2.60, 3 oct −2.89 (corners −1.37 = the half-dip −1.371). The planner uses it only for runs > 1 octave and fits on `response_db`, so this is documentation only.

### F8 (S5) — saturate level-loss numbers in the spec
- `docs/spec-draft.md:208` "−1.65 dB at 6 dB drive, −6.9 dB at 12 dB". Measured on a sine at the knee level K (`s4` 4c, harmonic 1, sym 0 or 0.3, band HP at 400 Hz included): RMS −0.14 / −1.73 / −7.30 dB at 2 / 6 / 12 dB; fundamental −0.14 / −1.75 / −7.55 dB (the band HP at 2.5 fc returns 0.22 dB of the change's fundamental). With harmonic 0.5 (SaturatePlan.defaults): −0.07 / −0.83 / −2.96 dB.

### F9 (S5) — `merge_candidates` docstring "floored at Q_MIN by widening"
- `src/eluate/stages/notch.py:140`. Flooring Q at 1 narrows the merged band below the union: Q 1 notches at 1000/1500 Hz merge to f0 1224.7 Hz Q 1.000, band 757–1982 Hz (1225 Hz wide) against the union 618–2427 Hz (1809 Hz). Part of each member's band is left uncut; disjointness of the output groups is preserved (`s2` 2c).

### F10 (S5) — padding definitions differ between bands.py and notch.py (both sufficient)
- `bands.py:69-70`: PAD_TAU 20 "time constants 1/(2π fc)" are 14.1 time constants of the actual butter(2) pole (decay √2/(2π fc)); the tail at the pad end is −123 dB (docstring's "about −120 dB" is right), and the split matches a 6 s zero-padded reference to −254 dB (80 Hz / 96 k), −266 dB (100 Hz / 48 k), −260 dB (60 Hz / 44.1 k) (`s6` 6d). `notch.py:102-106` uses true pole time constants: −174 dB at the pad end, −221…−269 dB vs 4× padding, but 214 260 samples (4.46 s) per side for Q 50 at 60 Hz and 642 779 (6.70 s) at 40 Hz / 96 k (`s2` 2f); 52–152 ms per 2 s of audio. PAD_TAU ≈ 14 would match bands.py and halve the padding.

### F11 (S5) — `gated_rms` with one loud block in silence reads low
- `src/eluate/stages/saturate.py:59-72`: a 400 ms burst (RMS 0.212) in 4 s of silence gives 0.173 (−1.8 dB): the half-overlapping neighbour blocks pass the −60 dBFS gate at half power and are averaged in (`s4` 4e). Short/quiet/empty/n=2 inputs behave as documented (0.0 → bit-exact identity).

### F12 (S5) — stale comment
- `src/eluate/stages/limiter.py:94` "2 x 182-tap half-length": the processing prototype is 363 base taps (1453/2905 at 4×/8×); the band-limit round trip's measured reach is 2746 8× samples (343 base) < BL_MARGIN·8 = 4096, so the margin is still sufficient (`s10` 10b).

---

## Stage alignment / latency table

Method (`s7_latency.py`, `s10_extra2.py` 10b): 2 s stereo at 48 k (−4.5 dB/oct noise at −20 dBFS, correlation 0.6, three kick+click events); lag = argmax of the FFT cross-correlation with parabolic refinement, full band and per band (<100 Hz / 100 Hz–1 kHz / >1 kHz via bands.split). Positive lag = output later.

| stage | neutral plan | bit-exact under neutral | max \|y−x\| neutral | delay (samples) | notes |
|---|---|---|---|---|---|
| input | gain 0 | yes | 0 | 0 | −3 dB: exact scale, lag 0 |
| inputeq | all None | yes | 0 | 0 | with a high-pass (zero or minimum mode) the HPF runs causally: full-band xcorr −144 samples (noise, LF-weighted); per band sub −170 / mid −5.8 / high −0.01 samples; on a kick alone the <100 Hz band lags 3.5 ms (Q 0.707), 4.3 ms (Q 0.5), 2.7 ms (Q 1.0); group delay at 25/50/100/200 Hz: Q 0.5 12.7/5.1/1.5/0.4 ms, Q 0.707 18.0/5.3/1.2/0.3 ms, Q 1.0 25.5/4.9/0.9/0.2 ms (fs-independent, `s1` 1d). Shelves/bell/band shelf alone: lag 0.000, symmetric to 1.2e-16 |
| notch | empty / depth 0 / full kick mask | yes (all 7 identity cases, 1-D and 2-D) | 0 | 0 | zero phase (max \|Im H\| 1.9e-15) |
| sidestrip | none (always filters S) | no: M 1.1e-16, L+R 2.2e-16 | 1.2e-1 (the S HP) | 0 | mono input: 1.1e-16, not bit-exact |
| multiband | GR 0, side makeup 0 | no (3.3e-16: M/S round trip + band sum) | 3.3e-16 | 0 | default side makeup +2 dB on bands ≥ 2 kHz is not neutral (−36 dB re peak) |
| tone (unused) | 0 dB nodes | yes | 0 | 0 | ±1.5 dB tilt: lag 0 |
| saturate | drive 0 | no (3.3e-16) | 3.3e-16 | 0 | change about an impulse symmetric to 1.8e-16 (4×) / 2.7e-16 (8×) |
| glue | gain 0 (any mix) or mix 0 | yes | 0 | 0 | vintage change path (4× resampler + causal 20 Hz HP on the change only): lag 0.004 samples |
| speclim | ceiling 0 | no (3.3e-16) | 3.3e-16 | 0 | |
| limiter | LimiterPlan() below ceiling | no: processing resampler round trip | 9.5e-2 on full-band noise (−18 dB re peak: 23.5–24 kHz removed at −2.8 dB, 20–20 kHz error −56 dB, ripple ±0.018 dB) | 0 (refined 0.000; 0.3/1/3 dB GR: ≤0.01/0.02/0.08 samples; the −2.2-sample sub reading at 12.8 dB GR is gain modulation of the kick, not delay) | clip_hi_part + band_limit_change: symmetric to 3.5e-18 (zero delay); whole clipper with/without split, kick clipper: lag 0.000 |
| output | 32-bit float | no: float32 rounding 3.0e-8; 24-bit 6.0e-8; 16-bit TPDF 4.5e-5 | | 0 | |

Whole chain under a neutral plan (input, inputeq None, notch empty, multiband 0, saturate 0, glue 0, limiter default, output 32): lag 0.000, max |y−x| 9.5e-2 — entirely the limiter's resampler on full-band noise (at 44.1 k the removed band is 21.6–22.05 kHz; 20–20 k error −61 dB, `s9` 9d). The stems null test and level-matched A/B are therefore safe on alignment; the only time misalignment in the chain is the inputeq high-pass's frequency-dependent group delay (deliberate, R7e).

---

## Verified correct (with numbers)

inputeq (`s1`)
- RBJ high-pass, low shelf (with Q), peaking, and the first-order shelf bilinear of (√A s+1)/(s/√A+1): coefficients equal the literal cookbook to 0.0 and the prewarped analog prototypes to ≤2.6e-11 (25 Hz) / ≤1.3e-15 (shelf). Shelf: DC 0, corner g/2, Nyquist g to 1e-5 dB; bell: centre g, half-gain band edges at Ω−1/Ω = ±1/Q exactly g/2.
- HPF one biquad per pass squared = the 24 dB/oct cascade (2.8e-14 dB). Zero-phase sections symmetric about the impulse (1.2e-16).
- Band shelf: |LP2|²+|HP2|² = 1 to 1.3e-12; `band_shelf_db` = D(f) exactly; zero form |H| = D to 3e-12 dB, phase 1.4e-14 rad; causal form |H| = D to 4e-12 dB and phase = arg(AP_lo·AP_hi) to 8e-13 rad; `apply(phase="minimum")` bit-equals `_band_shelf_causal`; no pre-impulse energy.
- ms_top: M and S impulses leak 0.0 into the other channel; `response_db(…, "M"/"S")` matches the measured response to 2e-10 dB (zero and minimum); air_shelf_side alone makes the top end M/S correctly (+1.87 dB at 20 kHz, not the mid's +3). Linked plans: `response_db` vs FFT of `apply()` 2.7e-10 (zero) / 2.9e-10 dB (minimum); L == R bit-exact.

notch (`s2`, `s9`, `s10`)
- Half-cut bandwidth = f0/Q after `design_q`: |error| ≤ 5.1e-4 (grid-limited) for f0 ∈ {40, 60, 150, 1000, fs/4, 0.45 fs}, Q ∈ {1, 3, 10, 50}, g ∈ {−1, −3, −6}, three rates; depth at f0 exact to 1e-4 dB.
- Composite clamp: two full-depth notches with bands just disjoint (Q 1/3/10; spacings 1.0001× the touching ratio up to 2×): composite minimum −6.0000 dB (unclamped would be −8.5 to −8.8 dB), zero phase retained; the notches themselves read −5.77…−6.00 dB at f0.
- `merge_candidates`: touching bands merge; chained overlaps form one group independent of input order; envelopes of different lengths hold the end value then take the minimum; different rates raise; empty/single pass through.
- `kick_mask` at audio rate (three rates, on- and off-grid onsets): hold = 1 from −20.35…−21.0 ms to +20.5…+21.0 ms (≥ ±20 ms, ≤ ±21 ms), ramps 4.98–6.01 ms (5 ms + ≤ one control period), range [0, 1], monotone; two onsets 12 ms apart hold 1 between them.
- Identity: empty list, constant 0, flat 0 (two notches), full mask (constant and array, one and two notches), mask > 1, positive gain: all bit-exact, 1-D and 2-D, and a copy.
- Narrow-notch padding: tail at pad end −174 dB; vs 4× padding −221…−269 dB.

sidestrip (`s3`)
- |HP|² = r⁴/(1+r⁴) with r = f/fc agrees with the digital complement to ≤1.3e-3 dB (warping) and with the prewarped r to ≤1.6e-5 dB; −6.0206 dB at fc; −0.0194 dB at 4.6 fc (claim "> −0.02 dB above 4.6 fc" holds, min above −0.0194); S gain at 30/100/200/460/1000/10000 Hz: −41.90/−6.02/−0.53/−0.019/−0.0009/0.0000 dB.
- M unchanged to 1.1e-16, L+R to 2.2e-16, to_lr(to_ms(x)) − x ≤ 1.7e-16 (three rates).

saturate (`s4`)
- Aliasing: 15 kHz air-band tone at 8×, drives 6/12 dB, sym 0/0.3, 44.1 k and 48 k: no non-harmonic product above −80 dB re fundamental (at 4× for comparison: 11.4 kHz −48.9 dB and 18.6 kHz −57.3 dB at 44.1 k; 12 kHz −58.3 and 18 kHz −65.5 at 48 k). 5 kHz and 6.3 kHz band-3 tones at 4×: no product above −80 dB; 8× gives the same (the limit is the decimator's 60 dB stopband on H5: −22 − 60 = −82 dB landing at 23 kHz / 19.1 kHz, derived and consistent with the spectra). 4× is enough for band 3.
- Curve: monotone for sym ≤ 1 (min slope 3e-10 at sym 1, 1.7e-5 at 0.3), slope 1.000000 at 0; sym 1.2 non-monotone (plan rejects > 1).
- gated_rms: short quiet → 0 → bit-exact identity; short loud = plain RMS; empty 0; n = 2 ok; −59 dBFS band finite.
- Change path has zero delay (symmetric about an impulse to 1.8e-16); the round trip's impulse peak is at offset 0.

resample / oa (`s5`, `s9`)
- prototype(4) 1453 taps, prototype(8) 2905 taps (363 base taps each), centre on a multiple of L/2, DC gain L. Single pass: ripple +0.0089/−0.0077 dB to 0.98 Nyquist, −6.02 dB at 0.99, −60.9 dB at 1.0, stopband max −59.9 dB, −74.9 dB above 1.05 Nyquist. Round trip: zero-phase symmetric kernel (727 base taps, symmetric to 0.0), DC gain 1.000000, max |gain| 0.0178 dB to 0.98 Nyquist (TRANSPARENT_DB 0.02 holds), −0.015 dB at 0.98, −12.0 dB at 0.99, −37.9 dB at 0.995, −116 dB at 1.0. Metering prototype for reference: −2.85 dB at 0.95 Nyquist.
- Alignment: impulse peak at L·i exactly (value 0.990), round-trip peak at i (0.9878), symmetric to 1.9e-16; sine phase shift 0.00 rad at 100 Hz…0.97 Nyquist (4× and 8×).
- Edges: abrupt full-scale start: round-trip error 8.4e-2 (−22 dBFS) in the first 50 samples, 4e-4 in 50–500, 2.9e-5 (−91 dB) interior, 7.6e-3 in the last 50; overshoot |z| max 1.0011.
- `oa_stream` vs `oaconvolve`: bit-identical with no gaps for every streamed case (s1 100…1 000 003 × s2 7/363/1000/2905); scipy always gives step ≥ overlap; declines (single-block or kernel-dominated cases, e.g. s1 ≤ 10 000 with the 2905-tap kernel) fall back to the whole-array path. `resample.upsample/decimate` bit-identical to whole-array oaconvolve for n = 2, 100, 1000, 4097, 100 003 at 4× and 8×.
- Private import: `oa._calc_oa_lens is signal.oaconvolve.__globals__["_calc_oa_lens"]` (scipy 1.18.1), signature (s1, s2); scipy's nsteps logic matches oa.py line for line. `_calc_oa_lens` has lived in `scipy.signal._signaltools` since the 1.8 private-module rename (1.4–1.7: `scipy.signal.signaltools`, import would fail and oa.py declines); every version in the pyproject range (≥1.13, <1.19) has it; tests/test_oa.py guards a move.

bands (`s6`)
- Band sum = input to 4.4e-16 (44.1/48 k) and 8.9e-16 (96 k) on 3 s of noise (claim 1e-9); mixed phases: sum = AP100(x) to 4.4e-13.
- Isolated crossovers −6.021/−6.021 dB at every fc, zero and minimum; in the 5-band cascade −6.021…−6.088 dB.
- split_iter == split bit-exact for zero, mixed and minimum; inputs (48001,), (48001,1), (2,2), (3,), (2,), (513,2) all split with exact sums; n = 1 raises; the caller's array is untouched.
- Minimum phase: sum − allpass product 4.8e-15, |sum| flat to 2.7e-12 dB; LP² and HP² in phase at fc (≤2.5e-13 rad); `allpass_sos` on a first-order section flat to 6.9e-15.

render / hygiene (`s7`, `s8`)
- Every stage returns float64 for float32 input; no float32 cast mid-chain (Envelope values are float32-representable by design; output bits 32 is the documented cast). `np.errstate` is used where logs are taken (inputeq.response_db, glue, multiband, meter).
- Silence through every active stage: finite, exactly zero (16-bit TPDF: ±1 LSB noise by design); render of silence (full plan): zero, no warnings.
- Very short inputs n = 2, 3, 100, 1000, 4800 through every stage (inputeq zero/ms/minimum, notch with kick mask, sidestrip, multiband, tone, saturate, glue vintage, limiter whole/kick clipper, output 24/16): all ok, finite, right shape; only `underflow` RuntimeWarnings under `np.errstate(all="warn")` (ignored by default).
- NaN input: inputeq/notch/tone/glue propagate; bands-based stages and the limiter raise ValueError; `render` rejects NaN up front. gated_rms = 0 path returns a bit-exact copy.
- The seven in-scope test files pass (171 tests, 341.7 s).

---

## Open questions

1. F1: is the lowest-crossover phase choice ("minimum" was the spec's expected winner) still planned as a listening gate, or should the plan carry it explicitly? The numbers above (−15 dB / 6–10 ms at 6 dB sub GR) suggest it is audible on a kick after a breakdown.
2. F4: is sub-region IMD from band-1 saturation acceptable at the 6 dB default cap (−51 dB re the band's fundamentals), or should the change's HP be steeper?
3. notch: `merge_candidates` can produce a merged f0 above Nyquist when members have very different Q (f0 = √(lo_min·hi_max)); not reproduced with same-Q members (merged f0 ≤ max member f0). Worth a guard at plan validation.
4. `composite_scale` refines the three largest grid maxima on a 1/96-octave grid; a Q 16 notch spans ~8.7 grid steps, so this is fine for the planner's range, but a hand-built Q 50 notch (0.03 octave, 3 steps) relies on the refinement bracket; not tested beyond Q 10 pairs.
5. Performance of F5 on real tracks depends on how much digital silence they contain (intros, outros, batch gaps); measured here on a worst case (20 s of silence).
