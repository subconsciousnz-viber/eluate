# Eluate audit: audio-rate gain and per-sample nonlinearities

Branch audit/2026-11-10, 2026-11-10. Read-only audit; nothing in the repo was modified.
Harness: `<scratch>/audit/audio-rate/run.py`
(`python run.py h1|h2|h3|h4|h5|h6|h7|fixes|mbkick|shape|all`, helpers in `harness/`, results in `out/*.md|json`, logs in `logs/`).
Every stage is driven as the planner drives it (`planner.character_settings` mirrored without an analysis dict:
`CHARACTER_PRESETS` timings, `LOW_BAND_RELEASE_MIN_MS`, `LIMITER_RELEASE_MIN_MS`, `multiband.design_budget` with the
electronic/live budgets, `glue.design_budget`, `multiband.design_speclim`, `planner.design_limiter`). Signals are
synthetic (48 kHz stereo; 44.1 kHz spot checks): S1 sines 30/40/60 Hz at -6 dBFS; S2 40 Hz sine (-12 dBFS) + swept
kick (150->45 Hz over 60 ms, 150 ms decay) at 128 BPM; S3 `tests/test_average_peak.py::drum_loop`; S5 1 kHz at
-20 dBFS + 5 ms noise bursts at -3 dBFS every 500 ms; S6 pink-noise bursts with 50 ms fades; S7 kick-only loop.
Units: ripple = p-p of the applied gain in dB; vel = |dG/dt| in dB/ms; THD/H2/H3 re fundamental; "re carrier" = re
the partial being modulated. "As planned" means the values the planner actually writes (limiter release clamped to
>= 80 ms; band ceilings' release = the `MultibandPlan` default 50 ms, which the planner never sets).

## Findings, by severity

### F1 (S1, audible on sparse / sub-heavy material; fast and aggressive worst, every electronic character affected). The FET glue's sub-millisecond attack is a 1 ms full-band gain step.

`stages/glue.py:141` (`lagging_attack`), `:152` (`causal_detector`, `DETECTOR_DELAY = 2` at `:108`), `:168`
(`_gain_env` applies the attack at the 1 ms control rate), `:330` (`apply`: the gain is full-band; the sidechain is
high-passed at `SIDECHAIN_HP_HZ = 120`, `:109`). `planner.CHARACTER_PRESETS` glue attacks 0.8/0.8/0.4/0.2/0.1 ms.
At a 1 ms control rate a = exp(-1/0.1) = 4.5e-5: the one-pole is a one-sample step, which PCHIP turns into a 1 ms
smoothstep landing 1-2 ms after the onset, applied to the sub as well as the transient.

Evidence (H4, `out/h4.md`; S2 = sub + kick at 128 BPM, budget-solved, net 1.5 dB met in every row):

| char | peak GR dB | vel max dB/ms | largest fall in 1 ms (median / max) | in 5 ms | sub x gain: energy 100-1000 Hz re sub | H3 region 110-130 Hz | onset 1-5 kHz energy in -> out (dBFS, 22 ms window) |
|---|---|---|---|---|---|---|---|
| transparent | 2.4 | 1.5 | 0.97 / 1.11 | 1.84 | -49.1 | -54.7 | -92 -> -81 |
| slow | 2.9 | 1.8 | 1.12 / 1.30 | 2.33 | -47.1 | -52.6 | -92 -> -80 |
| moderate | 3.4 | 3.0 | 1.70 / 2.09 | 3.19 | -43.8 | -50.1 | -92 -> -75 |
| fast | 5.3 | 5.75 | 2.73 / 3.90 | 4.84 | -40.5 | -46.7 | -92 -> -70 |
| aggressive | 6.1 | 5.74 | 2.77 / 4.05 | 5.00 | -40.2 | -46.1 | -92 -> -71 |

On a steady sub (H1 glue rows, S1 at -6 dBFS; this is the glue compressing the HP residue of the sub, which a real
track's kicks would normally do instead): aggressive THD -35.3 / -38.3 / -44.9 dB at 30 / 40 / 60 Hz (H3 -36.5 /
-39.6 / -45.8), fast -40.3 / -43.5 / -50.5, moderate -46.3 / -49.8 / -57.0; gain ripple up to 0.48 dB and
velocity 3.7-5.8 dB/ms at every character (the step is there even at transparent; its depth scales with the budget).
On the S5 probe the glue puts modulation products at -44 dB re carrier in the 200 Hz-2 kHz band around a 1 kHz tone
(fast / aggressive; the multiband puts -60 to -88 there). The 1 kHz PCHIP line itself is only -82 to -100 dB: the step
is the problem, not the interpolation (H2 below).

Audibility: a 4 dB gain fall in 1 ms on a 40 Hz sub produces a broadband tick 40 dB below the sub's own level and
raises the 1-5 kHz energy around each onset by 21 dB. In this synthetic case the tick sits ~55 dB under the kick's
onset (-16 dBFS in 100-1000 Hz), so simultaneous masking by a real kick's click will usually hide it; on a soft kick
over a loud clean sub, or when the glue is driven by a snare/hat while the sub sustains, it is the "spit" the
maintainer describes. THD of -35 dB on a sustained sub (1.8 %, mostly H3) is at the detection threshold for
low-frequency distortion on pure tones (about 1 %, -40 dB; loudspeaker-distortion literature, Geddes & Lee 2003,
Temme) and clearly audible in isolation at -30 dB. The live profile (glue attack 20 ms) is unaffected.

Minimal fix (prototyped, `harness/fixes.py::f4`): floor the lagging attack at >= 3 control samples (3 ms) in
`glue._gain_env` (F4a). Measured on S2: vel 5.74 -> 1.63 dB/ms, largest 1 ms fall 4.05 -> 1.39 dB, sub splatter
-40.2 -> -46.0 dB (H3 region -46 -> -51), onset 1-5 kHz -71 -> -88 dBFS (within 4 dB of the input), peak GR 6.1 ->
5.8 dB, net budget still 1.5 dB. Adding a 25 ms trailing hold before the program release (F4b) gives -49 dB
splatter but the budget solve then meets its 1.5 dB meter reading with a shallower peak GR (6.1 -> 3.7 dB): a
character change, not a free fix. Applying the glue gain above the sidechain split only (F4c, `glue_apply_split`)
leaves the sub untouched (-73 dB splatter) but changes the stage's nature (the sub no longer ducks with the hits).
Recommendation: F4a now; the module docstring already concedes that "an attack under 1 ms is a near-immediate step
after the detector's 1 to 2 ms delay", so the presets' 0.1-0.8 ms values buy nothing. Confidence: high.

### F2 (S1 on sub-heavy drops; moderate / fast / aggressive). The final 8x limiter ripples inside each sub cycle at 3 dB of GR.

`stages/limiter.py:402` (`_smooth` -> `envelope.box_smooth` at 8x), `envelope.py:205-208` (L = attack//2 + 1, min
over +-(L-1)), `planner.py:450` (`LIMITER_RELEASE_MIN_MS = 80`, applied at `:1793`). H1 confirmed with the
architect's formula within 0.1 dB for this stage (the hump of the required gain is narrow for a limiter):

| char (attack / release as planned) | 30 Hz ripple / THD / H3 | 40 Hz ripple / THD / H3 | 60 Hz ripple / THD / H3 | raw preset release: 30 Hz / 40 Hz THD |
|---|---|---|---|---|
| transparent 5 / 200 | 0.13 / -47.8 / -48.3 | 0.07 / -53.6 / -53.9 | 0.02 / -65.2 / -65.5 | (same) |
| slow 3 / 150 | 0.22 / -42.2 / -43.1 | 0.15 / -46.0 / -46.7 | 0.07 / -53.1 / -53.5 | (same) |
| moderate 2 / 100 | 0.35 / -37.8 / -39.0 | 0.25 / -40.8 / -41.8 | 0.15 / -46.0 / -46.7 | (same) |
| fast 1 / 80 (preset 50) | 0.46 / -35.2 / -36.7 | 0.35 / -37.6 / -39.0 | 0.23 / -41.4 / -42.7 | -31.9 / -34.2 |
| aggressive 0.5 / 80 (preset 30) | 0.49 / -34.9 / -36.5 | 0.37 / -37.1 / -38.8 | 0.25 / -40.4 / -42.1 | -28.3 / -30.2 |

H2 is -183 to -195 dB (symmetric), the distortion is odd (H3, H5: -44.5 dB at 40 Hz aggressive). The envelope of the
output ripples 0.29-0.37 dB p-p (`env_pp_db`). The D7 clamp to 80 ms halves the ripple against the raw 30 / 50 ms
presets but leaves -35 to -38 dB THD at 3 dB of GR; since the clipper is split at the lowest crossover (§20) the sub
peaks go to this limiter by design, so this is the normal operating point on a sub-heavy drop. Identical at 44.1 kHz
(H7). Audibility: -35 dB THD (H3 1.5 %) on a sustained 30-40 Hz sub is around the pure-tone threshold; in a mix the
bass's own upper partials mask the H3 at 90-120 Hz unless the sub is clean. The amplitude-modulation view gives the
same verdict: 0.4-0.5 dB p-p = 20 log m ~ -31 dB, near the AM-detection floor for low rates (-25 to -30 dB,
Viemeister 1979; Kohlrausch, Fassel & Dau 2000).

Minimal fix (prototyped): an asymmetric hold in the 8x `box_smooth` (`harness/fixes.py::box_smooth_hold`, min over
offsets [-(L-1)-H, L-1], H = 25 ms = 9600 samples at 384 kHz). Measured: ripple 0.37-0.49 -> 0.000 dB, THD -35/-37
-> -78/-80 dB at 30/40 Hz for fast and aggressive; 12.5 ms is not enough at 30 Hz (0.08-0.10 dB, THD -49/-51), 33 ms
is the same as 25. g <= g_req still holds (the two moving averages' support [-(L-1), L-1] stays inside the min
window; tested numerically on 63 (attack, hold) cases, max violation 0). Cost in the chain (next section): <= 0.14 dB
of meter RMS at equal gain, limiter max GR +0.1 to +0.3 dB. Confidence: high.

### F3 (S2 numbers / S3 robustness). Band ceilings and speclim ripple on 30-40 Hz at fast / aggressive, and the band ceilings' release is an unplanned default.

`stages/multiband.py:406-411` (`ceiling_envelope`), `plan.py:507` (`ceiling_release_ms` default 50.0),
`planner.py:1823` (the multiband record carries no `ceiling_release_ms`, so every band ceiling releases in 50 ms,
while speclim gets `SPECLIM_RELEASE_MS` 100/80/80/80/80 at `:407`). The architect's H1 assumed 100 ms. A 6 dB cut
on the sub band (ceiling-only `multiband.design`, S1 at -6 dBFS):

| stage (lookahead = preset band_lookahead[0]) | char | 30 Hz ripple / THD | 40 Hz ripple / THD | 60 Hz |
|---|---|---|---|---|
| band ceiling, release 50 ms (as planned) | fast (10 ms) | 0.12 / -51.5 | 0.00 / -84 | 0 / -219 |
| band ceiling, release 50 ms (as planned) | aggressive (6 ms) | 0.48 / -39.3 | 0.18 / -48.2 | 0 / -90.8 |
| band ceiling, release 100 ms | aggressive | 0.28 / -43.7 | 0.11 / -52.9 | 0 |
| speclim, release 100 ms (as planned) | aggressive | 0.28 / -43.7 | 0.11 / -52.9 | 0 |
| any of the above | transparent / slow / moderate (20 / 20 / 15 ms) | 0 / -207 | 0 / -212 | 0 / -219 |

The formula overestimates here (1.43 predicted vs 0.48 measured at 30 Hz aggressive; 1.04 vs 0.18 at 40 Hz) because
the required gain itself is a hump of +-4 ms around each peak, which the hull fills. The budget gives the sub band a
ceiling only when a peak exceeds its p99 by more than the allowance, so this is rarer than F2; when it does, a 6 dB
cut at aggressive on a 30 Hz note is -39 dB THD (audible in isolation). Minimal fixes: give the band ceilings a
per-band release (sub >= 100 ms, as speclim already has) — halves the ripple on its own — and/or the F1/F2 hold
below, which removes it (THD -207 dB). Confidence: high.

### F4 (S3). The multiband compressor's sub band ripples at aggressive on 30-40 Hz; the architect's numbers overestimate it.

`stages/multiband.py:338-341` (detector: two adjacent 1 ms segments), `:386-403` (`comp_envelope`:
`box_smooth(attack, release)` with attack 5 ms -> L = 3). Budget-solved at 2.5 dB (meter reading) on S1:

| char (sub attack / release) | 30 Hz ripple / THD / H3 | 40 Hz | 60 Hz | predicted (formula) 30 / 40 Hz |
|---|---|---|---|---|
| aggressive 5 / 80 | 0.22 / -44.7 / -45.1 | 0.11 / -51.4 / -51.8 | 0.03 / -64.2 | 0.42 / 0.30 |
| fast 10 / 120 | 0.03 / -63.9 / -64.0 | 0.00 / -89.8 | 0.00 / -219 | 0.23 / 0.15 |
| moderate 20 / 200, slow, transparent | 0.00 / -207 | 0.00 / -212 | 0.00 / -219 | <= 0.08 |

The detector itself ripples 12.1 dB at 2f (a 2 ms peak window on a 40 Hz sine), but with a 1 dB knee at 2:1 the
required-gain hump is wide, the release hull fills it, and the +-2 ms min window plus the triangular average leave
0.11-0.22 dB. Below the pure-tone threshold at 40 Hz (-51 dB), marginal at 30 Hz (-45 dB), nil at 60 Hz and at every
other character. The per-period shape (det, g_req, control envelope) is in `out/shape.md`. Fix: F1/F2 (below) remove
it entirely at a cost of raising the solved sub thresholds by 0.6-1.3 dB for the same meter reading. Confidence:
high.

### F5 (S4 hygiene). The planner's notch envelopes are a 0.5 dB staircase at 1 ms.

`planner.py:348` (`NOTCH_QUANT_DB = 0.5`), `:870` (quantisation after the 200 ms smoothing). Each 0.5 dB step is
one control sample, which PCHIP renders as a 1 ms smoothstep: notch-gain velocity 0.75 dB/ms (vs 0.05 for the same
ramp unquantised), 16-48 steps per 6 dB excursion. On a tone at the notch frequency the modulation sidebands are
-47.5 dB re carrier within +-200 Hz and -60 dB in 200 Hz-2 kHz, against -64.5 / -94.6 dB for the unquantised
envelope (H5, no kick mask). Below the AM threshold (-26 to -30 dB) and masked by the resonance's own level in a mix,
so hygiene: the float32 sidecar makes the quantisation pointless; drop it (or quantise before the smoothing).
Confidence: high on the numbers, medium on it ever being heard.

### F6 (S4 / open question). The kick mask opens the notch 20 ms before each kick with a 5 ms ramp.

`stages/notch.py:111` (`kick_mask`, `half_ms = 20`, `ramp_ms = 5`). The product d(t) = keep(t) x depth(t) is C1 and
the composite s(t) is PCHIP-smooth as designed (H5 "smooth" rows: velocity 0.05 dB/ms, s(t) min 0.754 where three
Q 8 notches at 500/580/660 Hz overlap, velocity of s 0.09 dB/ms). But the 5 ms raised-cosine mask ramp on a 4-6 dB
notch puts -27 to -33 dB sidebands within +-200 Hz on the notched partial (tone-only measurement), 45x per 20 s at
128 BPM, and the opening ramp lands 20-25 ms before the kick, where the kick cannot mask it (pre-masking is
5-10 ms). Whether a 4-6 dB level change in 5 ms on a sustained resonance partial is heard as a "bump" depends on the
mix; with `ramp_ms` at 15-20 ms the sidebands fall by ~10 dB. Design choice for the maintainer, not a defect.

### F7 (S5). ms_link's audio-rate `np.minimum` kinks are real and inaudible.

`stages/multiband.py:201` and `:257`. Live profile on S3 (H3): 37-717 crossings per 20 s per band; the kink
raises the gain's maximum second difference 3 to 66 dB/ms^2 (vs 0.2 to 7 for the control-rate minimum) and the two
gains differ by at most 0.01-0.30 dB; the output difference is -77 to -94 dB re the signal (its part above 500 Hz
-79 to -95 dB). On a stereo 40 Hz sine (R = L - 6 dB) base and F3 give identical THD (-58.6 dB aggressive, from the
compressor's own ripple). F3 (`harness/fixes.py::f3`, the minimum of the control values then PCHIP) is a free,
bit-stable cleanup; the stored envelopes are already float32 so no new rounding. Confidence: high.

### F8 (S5). Stale rows in docs/gain-staging.md §7.

`docs/gain-staging.md:182-188`: the table lists glue attack / release 30/600, 30/400, 30/300, 10/150, 3/80 while
`planner.CHARACTER_PRESETS` (`planner.py:419-444`) holds 0.8, 0.8, 0.4, 0.2, 0.1 ms (the §18 FET change); it lists
limiter release 50 / 30 for fast / aggressive while the planner clamps to 80 (`LIMITER_RELEASE_MIN_MS`, D7, not
mentioned in §7); it says nothing about the band ceilings' release (50 ms default) vs speclim's 100/80.

## Hypotheses: verdicts with numbers

- **H1 (sub-cycle ripple).** Confirmed in mechanism, overestimated in size for the compressor and ceilings, accurate
  for the limiter. Measured (aggressive, 40 Hz): compressor 0.11 dB / THD -51 (predicted 0.30 -> "H2 near -34" is
  wrong: the distortion is odd, H3 -52, H2 -83); band ceiling 6 dB 0.18 / -48 (predicted 1.04 at the planned 50 ms
  release, 0.54 at 100); limiter 3 dB 0.37 / -37 (predicted 0.43 at the planned 80 ms; 0.85 / -30 at the raw 30 ms,
  predicted 1.01). The formula fails for the compressor because the required gain is a wide hump (|sin| within the
  5 dB knee for +-4 ms) that the hull fills; only the remaining 12.5 - 2(L-1) - ~8 ms decays.
- **H2 (PCHIP 1 kHz lines).** Refuted. The share of the gain's AC energy above the control Nyquist (500 Hz) is 0.0-0.1 %
  for every envelope on S3 (the aggressive glue, the most curved, 0.1 %); the 1 / 2 / 3 kHz lines are -87 / -105 /
  -115 dB re carrier at worst (aggressive glue), -98 to -134 dB for the compressor and ceiling envelopes (a C2 cubic
  spline on the same knots: -135 to -204). On S5 the +-1 kHz sidebands a stage puts on the 1 kHz tone are -67 to
  -141 dB re carrier; the near sidebands (-27 to -41 dB) are the intended ducking at the burst rate. PCHIP's C1 kink
  doubles the second derivative versus a C2 spline (e.g. 2.04 vs 0.93 dB/ms^2) with no spectral consequence.
- **H3 (ms_link kinks).** Confirmed to exist, quantified as inaudible: see F7.
- **H4 (glue 1 ms steps).** Confirmed and the largest finding: see F1.
- **H5 (notch smoothness).** Confirmed for the stage: d(t) and s(t) are smooth products of PCHIP curves (velocity
  0.05-0.09 dB/ms, no energy above 500 Hz, 1 kHz line <= -101 dB). The planner's inputs are not smooth (F5) and the
  mask ramp is fast by choice (F6).
- **H6 (clipper).** Verified correct. Clipping per span is bit-identical to applying the curve to every sample
  (`span_vs_full` = 0 in every case; the curve is the identity below the knee, so spans are only an optimisation).
  The band-limited change has 0.0 energy above the base Nyquist (the raw change of clipped pink noise has 17-21 %),
  and the local window never truncates (edge value 0 on S6; the non-zero edge values on the sines are the signals'
  own abrupt starts). Aliasing / non-harmonic residue: -94 to -128 dB re fundamental with the presets' soft knees,
  -84 to -99 with a hard knee. THD is the curve's: 1 dB of peak reduction = -29 to -30 dB THD (H3 -30 to -32)
  whatever the knee (0.5-1.5 dB), 2 dB = -22, 3 dB = -17/-18 (§20's own figures); the 3 dB per-sample cap never
  binds at the 1 dB target. The 8x limiter after it leaves one re-limit pass of <= 0.24 dB in every case and
  trim 0; true peak -0.304 / -0.305 dBTP (8x / 32x) against -0.3.
- **H7 (44.1 kHz).** Confirmed clean. `control_coord` at 44.1 kHz gives segment lengths {44, 45} with the 10 ms
  boundaries exact (0 ulp) and the sampled coordinates exact to 0 ulp against `Fraction`; the 40 Hz results at 44.1 k
  equal 48 k to three digits (compressor -51.4 / -51.4 dB THD, ceiling -48.2 / -48.2, limiter -37.1 / -37.1).

## Fix prototypes: effect and cost

Prototypes are monkey-patches in `harness/fixes.py` (F1 `f1`, `box_smooth_hold`, `limiter_smoother`; F2 `f2`;
F3 `f3`; F4 `f4`, `glue_apply_split`). Holds / windows per band [25, 10, 2.5|3, 1|2, 0|2] ms (one period of the band's
lower edge, sub >= 25 ms); limiter hold 25 ms.

Chain multiband -> glue -> speclim -> clipper + limiter, budget-solved per stage with the frozen loud mask (as
`realise`), the same requested limiter gain for every variant (`out/fixes.md`, S3 16 s; S2 fast / aggressive):

| signal | char | variant | sub threshold dB | sub comp avg / outlier | limiter max GR | meter RMS at equal gain | crest |
|---|---|---|---|---|---|---|---|
| S3 | transparent | base / F1 / F2 | -22.6 / -22.0 / -22.1 | 2.5 / 3.23-3.26 | 1.10 / 1.11 / 1.11 | -7.18 / -7.18 / -7.18 | 12.0 |
| S3 | slow | base / F1 / F2 | -22.6 / -22.0 / -22.1 | 2.5 / 3.23-3.26 | 1.07 / 1.09 / 1.08 | -7.20 / -7.24 / -7.24 | 12.1-12.2 |
| S3 | moderate | base / F1 / F2 | -23.6 / -22.8 / -22.8 | 2.5 / 3.07-3.11 | 1.47 / 1.61 / 1.60 | -6.41 / -6.55 / -6.49 | 11.2 |
| S3 | fast | base / F1 / F2 | -25.1 / -23.9 / -24.0 | 2.5 / 3.17-3.20 | 1.49 / 1.72 / 1.70 | -6.23 / -6.32 / -6.25 | 11.0 |
| S3 | aggressive | base / F1 / F2 | -26.1 / -24.8 / -24.9 | 2.5 / 3.26-3.27 | 1.47 / 1.75 / 1.73 | -6.26 / -6.36 / -6.28 | 11.0 |
| S2 | fast | base / F1 / F2 | -10.6 / -10.0 / -10.1 | 2.5 / 2.52-2.55 | 2.75 / 2.97 / 2.95 | -1.09 / -1.20 / -1.13 | 4.4-4.5 |
| S2 | aggressive | base / F1 / F2 | -11.4 / -10.5 / -10.6 | 2.5 / 2.54-2.56 | 2.61 / 2.91 / 2.89 | -1.08 / -1.07 / -1.00 | 4.3-4.4 |

The glue and speclim readings are unchanged (glue net 1.5 dB everywhere; speclim max cut 3.3-4.7 dB on S3, none on
S2); the limiter's budget clamp never bound. Cost of F1: 0 to 0.14 dB of meter RMS at equal gain (the multiband
takes less peak GR for the same meter reading, so 0.1-0.3 dB more reaches the limiter); F2: 0 to 0.08 dB. Effect on
S1 (`out/fixes.md` lower table): compressor and ceiling ripple 0.11-0.48 -> 0.000 dB, THD -45/-51 -> -207/-212 dB
for both F1 and F2; limiter ripple 0.37-0.49 -> 0.000 dB, THD -35/-37 -> -78/-80 dB with a 25 ms hold (12.5 ms
leaves 0.08-0.10 dB at 30 Hz). Recommendation ranking:

1. Glue attack floor 3 ms (F4a) — removes the only wideband artefact; no budget cost.
2. Limiter hold 25 ms in the 8x `box_smooth` (F1) — removes the sub ripple at 3 dB GR; <= 0.14 dB loudness in the
   chain tests, which includes the multiband holds.
3. Multiband compressor / band ceilings / speclim: F2 (trailing detector window [25, 10, 3, 2, 2] ms) is the cheaper
   of the two (no change to `box_smooth`, the bound holds a fortiori because det_new >= det_old), F1 is equivalent in
   effect; either way set the band ceilings' release per band (sub >= 100 ms) in `plan_multiband_record`.
4. F3 ms_link control-rate minimum — free, cosmetic.
5. Docs §7 rows; drop `NOTCH_QUANT_DB`; consider `kick_mask(ramp_ms=15..20)`.

## Verified correct

- `envelope.box_smooth`: g <= clip(g_req) holds exactly, and the asymmetric hold keeps it (kernel-support argument
  and 63 numeric cases, max violation 0). The hold's extent is exactly [-(2(L-1)), 2(L-1)+H] around an isolated dip.
- `plan.control_coord` at 44.1 kHz: no jitter; segments 44 / 45 samples, boundaries exact; results equal 48 kHz.
- PCHIP interpolation: no 1 kHz lines above -86 dB, no energy above 500 Hz beyond 0.1 %; caps hold between knots.
- The multiband compressor at transparent / slow / moderate: zero sub-cycle ripple at 30-60 Hz (THD -207 to -219).
- `LOW_BAND_RELEASE_MIN_MS` and `LIMITER_RELEASE_MIN_MS` are applied as documented in code (the latter halves the
  limiter ripple vs the raw presets).
- Clipper: span clipping == whole-signal curve; band-limit removes everything above Nyquist; aliasing residue
  <= -84 dB (hard knee) / <= -94 dB (presets); sample cap and knee behave as documented; ceiling held to within
  0.005 dB with one re-limit pass <= 0.24 dB and no trim.
- Notch: d(t) and s(t) are C1 products of PCHIP curves; the composite clamp engages (s min 0.754) and is smooth.
- Output stage (`stages/output.py`): plain rounding / TPDF per block, no gain involved; not measured further.
- Saturate (`stages/saturate.py`): a static waveshaper at 4x / 8x with only the change resampled; no time-varying
  gain, so outside the audio-rate-automation class. Not measured (the planner's drive cap is 2 dB).

## Open questions

- Masking on real material: all audibility statements above are for synthetic signals; the maintainer's ear on a
  clean-sub track at aggressive / fast with the glue step (F1) and the limiter ripple (F2) decides S1 vs S2.
- The hold delays every sub-band release by 25 ms; on a bass line with staccato notes the compressor's "pump" shape
  changes slightly. The chain tests show <= 0.14 dB of loudness, but no listening was done.
- Whether the speclim's per-band lookahead should come with a per-band hold as well (same mechanism as the band
  ceilings; same fix).
- The glue's realised attack is 1-3 ms (detector delay + step) for every preset; if the 1176 character is wanted,
  the glue envelope would need its own control rate (e.g. 0.1 ms) with band-limited interpolation — a larger
  change than the 3 ms floor.
- The kick mask's 5 ms ramp before the kick (F6): design intent or oversight?
- Not run: nothing from the brief was skipped. The saturator and output stages were read but not measured (no
  time-varying gain). No listening tests.

## Add-on measurements

### Multiband and speclim around kicks (deliverable 2e for the multiband; `out/mbkick.md`, S2 / S7, 12 s, budget-solved)

| char | band 0 (sub) M: peak GR / vel max / largest fall in 1 ms / in 5 ms | sub x gain splatter 100-1000 Hz re sub (S2 / S7) | band 1 (100-400) M: peak GR / vel max / fall in 1 ms / in 5 ms | output onset 100-1000 Hz in -> out (S2) | onset 1-5 kHz in -> out |
|---|---|---|---|---|---|
| transparent | 3.1 dB / 0.09 dB/ms / 0.09 / 0.43 | -81 / -83 | 3.9 / 0.27 / 0.18 / 0.64 | -15.8 -> -16.7 | -92.2 -> -92.3 |
| slow | 3.3 / 0.10 / 0.10 / 0.47 | -81 / -84 | 3.9 / 0.28 / 0.18 / 0.65 | -15.8 -> -16.7 | -92.2 -> -92.3 |
| moderate | 3.5 / 0.15 / 0.15 / 0.66 | -69 / -81 | 5.2 / 0.57 / 0.51 / 1.96 | -15.8 -> -17.5 | -92.2 -> -92.3 |
| fast | 3.8 / 0.28 / 0.27 / 1.13 | -52 / -70 | 6.0 / 1.05 / 1.01 / 3.54 | -15.8 -> -18.1 | -92.2 -> -92.4 |
| aggressive | 4.7 / 0.59 / 0.55 / 1.71 | -46 / -54 | 6.0 / 1.72 / 1.56 / 2.86 | -15.8 -> -18.2 | -92.2 -> -91.8 |

The multiband's sub-band gain moves at most 0.55 dB in any millisecond around a kick (the 5 ms attack's +-4 ms
lookahead window spreads it), the low band at most 1.6 dB; the output's 1-5 kHz onset energy is unchanged, so the
multiband adds no "spit" at any character (compare the glue's +21 dB, F1). Speclim designed no ceiling on either
signal (uniform kick peaks: nothing stands out above the typical loud hit), so its kick behaviour is covered by the
H1 ceiling rows only.

### Gain shape over one 40 Hz period (`out/shape.md`, S1 -6 dBFS)

Multiband sub band, aggressive (threshold -8.30 dBFS, knee 1 dB), control grid 1 ms: detector -15.3 / -9.6 / -6.6 /
-4.7 / -3.7 / -3.2 / -3.2 / -3.2 / -3.4 / -4.1 / -5.5 / -7.8 / -11.9 / -12.0 dBFS over the half period (12.1 dB
ripple); required gain 0 / 0 / -0.87 / -1.79 / -2.32 / -2.53 / -2.54 / -2.54 / -2.46 / -2.11 / -1.41 / -0.24 / 0 / 0 dB
(a +-5 ms hump, not a spike); the smoothed envelope -2.43 .. -2.54 dB (0.11 dB). Fast (knee 3 dB, attack 10 ms):
envelope -2.500 .. -2.499 dB: flat. Limiter aggressive 0.5 / 80 ms, 3 dB over: the 8x gain is a sawtooth -3.00 ->
-2.63 dB every half period (-3.00 -> -2.15 dB with the raw 30 ms release), which is the odd-harmonic distortion in F2.

## Measurement tables

Full tables: `out/h1.md` (per stage x character x 30/40/60 Hz: ripple, predicted, velocity, gain-spectrum
fractions, 1 kHz line, THD / H2 / H3 / non-harmonic residue, envelope ripple), `out/h2.md` (PCHIP vs cubic vs
band-limited; S5 sidebands), `out/h3.md` (ms_link), `out/h4.md` (glue around kicks, F4 variants, third-octave
in / out spectra in the JSON), `out/h5.md` (notch), `out/h6.md` (clipper / limiter), `out/h7.md` (44.1 k),
`out/fixes.md` (chain cost and S1 with fixes). JSON twins hold every number, including the top non-harmonic lines.
