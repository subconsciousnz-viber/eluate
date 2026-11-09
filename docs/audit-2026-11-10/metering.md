# Eluate metering, loudness, true-peak and delivery audit (2026-11-10)

Scope: `src/eluate/meter.py`, `features/loudness.py`, `stages/output.py`, `io.py`, `evaluate.py`, `uidata.py`, the limiter's true-peak envelopes (`stages/limiter.py`), `solver.true_peaks`, and the tests `test_meter.py`, `test_evaluate.py` and `test_io.py`. The tests give 138 passed and 64 skipped; the skips are the official EBU files, which are absent, so every EBU case below was rebuilt synthetically. Read-only: nothing in the repo was changed.

Scripts and raw outputs are in `audit/metering/` next to this file, one pair (`*.py` / `*.out`) per check:

| Script | What it checks |
|---|---|
| `a_kweight` | K-weighting and pyloudnorm |
| `b_gating` | Gating, Tech 3341, Tech 3342 |
| `c_truepeak` | True-peak worst cases |
| `d_ceiling` | Real limiter, then quantisation, then all true-peak meters plus an independent 64x FFT reference |
| `e_output` | Dither, quantisation, WAV I/O |
| `f_edges` | Edge cases, meter calibration, A/B files |
| `g_perf` | 10 min at 96 kHz: time and memory |
| `h_meter_vs_lufs` | 300 ms meter against LUFS |
| `i_abmatch` | UI level match against the A/B files' match |

The BS.1770-5 PDF was fetched from itu.int and parsed into `audit/metering/dl/bs1770-5.txt`.

**Headline.** The measurement core is correct:

- K-weighting, gating, and the momentary, short-term and integrated readings.
- LRA.
- The Annex 2 coefficients, which are bit-exact against the BS.1770-5 PDF.
- Dither and quantisation.
- The ceiling, which holds after quantisation on every meter, including an independent 64x reference.

What is wrong is what the report and the UI tell the user about delivery:

- At the default target, a master reads about -7.5 to -9.3 LUFS integrated at -0.30 dBTP. Nothing tells the user how platforms will play it back, or that it breaks Spotify's "-2 dBTP above -14 LUFS" advice.
- The UI level-matches the premaster on a different meter from the A/B files the user hears, a 0.6 to 0.7 dB mismatch at target loudness.
- The UI's true peak is the 8x reading, not the reading that is constrained.

## Findings (sorted by severity)

Severity: S1 wrong audio / ceiling violated; S2 wrong or missing numbers in report / UI; S3 crash / misbehaviour; S4 hygiene / performance; S5 nit.

**No S1 found.** The ceiling holds on the 4x, 8x and 32x meters and on a 64x reference after 16-bit TPDF and 24-bit rounding, at 44.1 and 48 kHz, +6 and +12 dB limiter drive (`d_ceiling.out`; see the compliance table).

---

### 1. [S2] No delivery information: integrated-LUFS playback offsets, platform true-peak rules, sample rate

- **Where:**
  - `evaluate.py:927-1026` (`report_md`)
  - `evaluate.py:583-594` (`rep["loudness"]`)
  - `uidata.py:218-258` (`from_render`) and `uidata.py:143-148` (`summary`)
  - `ui/app.js:1281-1291` (`drawStats`)
  - `planner.py:389-390` (ceilings)
- **What is wrong:** The loudness target is a 300 ms unweighted AES17 RMS, read as the p95. It has no fixed relation to BS.1770 integrated loudness, which is the quantity every platform normalises on.
  - report.md's "Loudness and peaks" table does print integrated, LRA, the 4x / 8x / 32x true peaks, PLR, PSR p95 and short-term max.
  - Nothing translates these into delivery terms: no per-platform playback gain, and no check of the true peak against platform recommendations at that loudness.
  - The sample rate is never printed. The bit depth appears only inside "What was done", and only when a plan was passed.
  - The UI shows crest, LRA, true peak (8x) and noise, and never shows a LUFS value.
- **Evidence:**
  - I ran the same electronic premaster through the real blocked limiter (`i_abmatch.out`, -0.3 dBTP ceiling):

    | Limiter drive | 300 ms meter | Integrated | S95 | Worst true peak |
    |---|---|---|---|---|
    | +12 dB | -4.82 dB | -9.30 LUFS | | -0.30 dBTP |
    | +16 dB | -3.76 dB | -7.48 LUFS | -7.03 LUFS | -0.30 dBTP |

    The default target is -4.1.
  - On this bass-led material the meter sits 5.2 dB above integrated (`d_ceiling.out`, every row).
  - Across signal types the gap ranges from -0.2 to +6.3 dB (`h_meter_vs_lufs.out`). So the target "-4.1 dB RMS" lands anywhere from about -4 to about -10 LUFS depending on the spectrum.
  - At -7.5 LUFS, platforms play the master back with these gains:

    | Platform | Reference | Playback gain |
    |---|---|---|
    | Spotify Normal, YouTube, Amazon, Tidal | -14 LUFS | -6.5 dB |
    | Deezer | -15 LUFS | -7.5 dB |
    | Apple Sound Check | -16 LUFS | -8.5 dB |
    | Spotify Loud | -11 LUFS | -3.5 dB |

  - Spotify's artist guidance says "keep it below -1dB TP (True Peak) max", and "If your master is louder than -14dB integrated LUFS, keep True Peak below -2dB".
    - The default 24-bit render (-0.3 dBTP) breaks both rules.
    - The `--streaming` render (-1 dBTP) breaks the second.
    - Neither the report nor the UI flags this.
- **Minimal fix:**
  - Add a "Delivery" block to report.md and report.json (exact lines under "Recommended additions" below).
  - Add a structured flag `kind: "delivery"`. It is short when `tp_max > -1.0`, or when `integrated > -14` and `tp_max > -2.0`. It shows amber like the other flags: the value plus amber is the must-know state, so no explanatory copy is needed.
  - Add value-only UI rows: Integrated, Short-term max, Playback gain (Spotify / Apple), Format.
  - Optionally offer a -2 dBTP ceiling for streaming uploads of loud masters.
- **Confidence:** high. The numbers are measured; the platform values are cited below.

### 2. [S2] The UI and the A/B files use different level matches

- **Where:**
  - `uidata.py:191-192`: `match_db` is the after-minus-before `rms_meter` reading. It is applied to the drawn premaster envelope and meter, and `matchDb` is also used at `ui/app.js:902`.
  - `evaluate.py:170-196`: `write_ab` matches on S95 by attenuating the louder signal. `ab/original_matched.wav` and `ab/master_matched.wav` are what the UI plays (`uidata.py:215-217`).
- **What is wrong:** The waveform and meter overlay show the premaster at one gain while the A/B audio plays it at another. The gap grows with limiting, because the unweighted 300 ms RMS and the K-weighted S95 react differently to compression and to the low end.
- **Evidence** (`i_abmatch.out`):

  | Limiter drive | 300 ms meter match | S95 match (A/B files) | Mismatch | Integrated match |
  |---|---|---|---|---|
  | +6 dB | +5.54 dB | +5.66 dB | 0.11 dB | +5.73 dB |
  | +12 dB | +9.42 dB | +9.98 dB | 0.56 dB | |
  | +16 dB (at target) | +10.49 dB | +11.19 dB | 0.70 dB | +12.34 dB |

  0.5 to 0.7 dB is enough to bias a level-matched comparison.
- **Minimal fix:** Use one reference in both places. The smallest change is one line: have `from_render` use `rep["ab"]["master_gain_db"] - rep["ab"]["original_gain_db"]` (the S95 match, spec 3.7) as `matchDb`. The alternative is to match the A/B files on `rms_meter`. Integrated LUFS, which is what platforms do, is the third defensible option.
- **Confidence:** high.

### 3. [S2] The UI, summary and process log show the 8x true peak, not the constrained reading

- **Where:**
  - `uidata.py:147` and `uidata.py:246` read `rep["after"]["true_peak_dbtp"]`, which is the 8x reading (`meter.measure`, `meter.py:528`).
  - `processlog.py:211` does the same.
  - The evaluate contract (`evaluate.py:6-9`) calls 8x "the constrained reading".
- **What is wrong:** The solver constrains `max(8x, 32x)` at 24-bit and `max(4x, 8x, 32x)` under `--streaming` (`solver.py:1130-1141`). The limiter's re-limit holds all three (`limiter.py:367-373`). The 8x Kaiser passband ends at 0.93 x Nyquist, so it under-reads full-band material:

  | Signal | 8x under-read |
  |---|---|
  | +0.5-sample band-limited impulse | 0.31 dB |
  | +0.125-sample band-limited impulse | 0.22 dB |
  | White noise (10 min, 96 kHz) | 0.51 dB (8x -4.75 dBTP, 32x -4.24 dBTP) |

  Sources: `c_truepeak.out`, `g_perf.out`.

  On Eluate's own masters the re-limit keeps all three readings within about 0.001 dB of each other (`d_ceiling.out`: 8x -0.3041, 32x -0.3048, 4x -0.3049), so the "after" value is right to a few thousandths. The "before" value of a bright or clipped premaster, and any `distance_to_master` comparison, can be 0.5 dB low.
- **Minimal fix:** Add `true_peak_max_dbtp = max(4x, 8x, 32x)` to `measure_track`. Show it as "True peak" in the UI, the summary and the process log, and keep all three in report.md.
- **Confidence:** high.

### 4. [S3] `uidata._load` silently keeps the first two channels of a multichannel file; `io.read` rejects the same file

- **Where:** `uidata.py:51-53` (`x[:, :2]`) against `io.py:94-99`.
- **What is wrong:** For a 5.1 or quad input, `preview()` draws a level-matched preview of L/R only, and then `master` fails in `io.read` with "stereo only". The preview and the engine disagree about what the input is.
- **Minimal fix:** Make `_load` use `io.as_stereo`, which duplicates mono and raises on more than two channels, or call `io.read`.
- **Confidence:** high (code reading).

### 5. [S4] `meter.measure` repeats its expensive work; the one-sample-hop sliding maxima are memory-heavy

- **Where:**
  - `meter.py:508-537`: `integrated`, `loud_passage_lufs`, `psr`, `short_term_max`, `momentary_max` and `lra` each K-weight the whole signal again (`_kpower`), six times in all. `true_peak(x, fs, 8)` and `psr` each build the 8x envelope, and the 4x reading is a third pass.
  - `evaluate.measure_track` adds the 32x FFT pass. The report runs all of this for both before and after.
  - `meter.py:233-235` (`_sliding_max_lufs` via `_window_powers` with hop 1/fs) materialises full-length arrays: `np.arange` as float, `np.round`, the `starts` int64, the cumsum, and the power array.
- **Evidence** (`g_perf.out`: 10 min at 96 kHz stereo, 57.6 M samples, 0.92 GB float64 input; 12-core machine with other audit jobs running concurrently, so times are pessimistic):

  | Call | Time | Peak extra RSS |
  |---|---|---|
  | `integrated` | 11.8 s | +2.30 GB |
  | `short_term_max` | 27.1 s | +2.73 GB |
  | `momentary_max` | 28.1 s | +2.76 GB |
  | `lra` | 7.0 s | +2.30 GB |
  | `true_peak` 8x | 49.9 s | +0.92 GB |
  | `true_peak` 4x | 41.2 s | +0.92 GB |
  | `true_peak_fft` 32x | 190.1 s | +2.09 GB |
  | `rms_meter` | 9.4 s | +1.39 GB |
  | `psr` | 56.1 s | +2.31 GB |
  | **`meter.measure` total (no 32x)** | **263.0 s** | **+2.76 GB** |

  The report calls `measure_track` twice, so before and after together take roughly 2 × (263 + 190) s, about 15 min, of metering alone on a 10-minute 96 kHz track.

- **Minimal fix:**
  - In `measure`, compute `ps = _kpower(...)` once and pass it to private helpers (`_integrated_ps`, `_short_term_ps`, and so on). Compute the 8x envelope once and reuse it for `true_peak` and `psr`.
  - In `_sliding_max_lufs`, compute `max(c[i+L] - c[i])` over chunks of about 2^20 positions instead of materialising every start. That removes roughly 4 × 8 × n bytes, about 1.8 GB at 10 min and 96 kHz.
  - The 32x FFT envelope is the slowest reading by far. For the report, consider a 16x FFT or a 32-phase windowed-sinc polyphase.
- **Confidence:** high.

### 6. [S4] `--streaming` means 16-bit at the input rate, with no sample-rate conversion and no 24-bit streaming master

- **Where:** `planner.py:637-638` (`OutputPlan(16, "tpdf") if streaming else OutputPlan(24)`) and `cli.py:399` (`io.write(..., fs, ...)`). master.wav is always written at the input rate, and nothing resamples for delivery.
- **What is wrong:** One switch couples three things: the -1 dBTP ceiling, 16 bits, and dither.
  - Distributors and platforms ingest 24-bit files, up to 192 kHz, and Apple Digital Masters asks for 24-bit sources (third-party summaries; Apple's own document could not be fetched).
  - 16-bit only makes sense as a 16-bit / 44.1 kHz CD deliverable. From a 48, 88.2 or 96 kHz premaster the engine writes 16-bit at 48 to 96 kHz, which is neither.
  - The README and gain-staging §3 wording ("streaming selects 16-bit") reinforces this.
- **Minimal fix:**
  - Make `--streaming` mean -1 dBTP at 24-bit.
  - Add a separate `--cd` option: SRC to 44.1 kHz with `resample_poly`, 16-bit TPDF, and the true peak re-checked after SRC.
  - Print the written sample rate and bit depth in report.md.
- **Confidence:** medium-high. The practice is long-standing; Apple's exact wording is second-hand.

### 7. [S4] The WAV output carries no metadata, and the input's metadata is dropped

- **Where:** `io.py:116-126`.
- **What is wrong:** `sf.write` produces plain `RIFF/WAVE/fmt /data` chunks (plus `fact` for float) (`e_output.out`). There is:
  - no BWF `bext` chunk, so none of the EBU Tech 3285 v2 loudness fields (LoudnessValue, LoudnessRange, MaxTruePeakLevel, MaxMomentaryLoudness, MaxShortTermLoudness);
  - no ISRC;
  - no `LIST/INFO`.

  Any premaster tags are discarded. This is acceptable for a hand-off to a distributor, but cheap to improve.
- **Minimal fix:** Copy the premaster's `title` and `artist` strings (soundfile exposes them). Optionally append a `bext` chunk with the measured loudness fields inside `atomic_write`; libsndfile does not write `bext` itself.
- **Confidence:** high.

### 8. [S4] report.json `loudness.before_rms_dbfs` / `after_rms_dbfs` duplicate the AES17 fields although the contract says plain

- **Where:** `evaluate.py:588-591`, against the contract at `evaluate.py:41-42`: "the RMS (plain and AES17)".
- **What is wrong:** `before_rms_dbfs == before_rms_aes17_dbfs`; both hold the AES17 `rms_meter` reading. A consumer that reads `*_rms_dbfs` as plain dBFS is 3.01 dB off.
  - Today's consumers (`processlog.py:182`, `stems.py:456`, `cli.py:627`) treat the field as the meter reading, so nothing prints a wrong number yet.
  - The plain 1 s value lives in `before_rms_1s_dbfs`.
- **Minimal fix:** Fix the contract wording ("both are the 300 ms AES17 meter reading") or drop the duplicate keys.
- **Confidence:** high.

### 9. [S5] The target meter is unweighted, so DC and infrasonic energy count as loudness

- **Where:** `meter.py:434-475` (`rms_series` / `rms_meter`).
- **Evidence** (`f_edges.out`): DC at 0.5 reads -3.01 dB on the target meter and -inf LUFS. A premaster with DC offset or sub-sonic rumble therefore reads louder on the target and comes out mastered quieter.
- **Fix:** If Ozone's RMS bar is also unweighted, this is by design (§17). Otherwise, high-pass the meter at 10 to 20 Hz, which is inaudible on music. In either case, add a DC-offset flag.
- **Confidence:** medium. It depends on Ozone's internals.

### 10. [S5] Sliding maxima on DC-like input return about -326 LUFS instead of -inf / NaN

- **Where:** `meter.py:233-245`.
- **Evidence:** 5 s of DC at 0.5 gives `short_term_max_lufs = momentary_max_lufs = -325.8`. These are cumsum-difference residues of a K-weighted DC that should be about zero. `integrated` correctly returns -inf. The finite number lands in report.json.
- **Fix:** Return -inf when `p.max() < 1e-30`, or apply the -70 LUFS absolute-gate convention.
- **Confidence:** high.

### 11. [S5] The steady-state K-filter start (`sosfilt_zi` scaled by x[0]) is not neutral for a signal that starts at level

- **Where:** `meter.py:15-17`, `meter.py:171`.
- **Evidence** (`a_kweight.out`, first 400 ms block; the true steady value of the 997 Hz case is -3.010):

  | Signal | zi(x[0]) | Zero state |
  |---|---|---|
  | 997 Hz cosine starting at +1 | -2.992 LUFS | -3.010 LUFS |
  | 50 Hz cosine | -7.671 LUFS | -7.630 LUFS |
  | Pink noise + 0.1 DC | -16.600 LUFS | -16.584 LUFS |

  - The first block moves 0.02 to 0.04 dB in either direction; there is no consistent bias.
  - Integrated readings on a whole track are unaffected.
  - The first-block value differs from libebur128, pyloudnorm and ffmpeg, which all start from zero state.
- **Fix:** None needed. It does do what audit 02 A16 wanted: DC reads -inf.
- **Confidence:** high.

### 12. [S5] At 44.1 kHz the 4x Annex 2 reading oversamples to 176.4 kHz, below Annex 2's "at least 192 kHz" for the dB TP label

- **Where:** `meter.py:530`; the report label at `evaluate.py:920` reads "True peak 4x (BS.1770)".
- **What is wrong:** BS.1770-5 Annex 2 says: "Meters that follow these guidelines, and that use an oversampled sampling rate of at least 192 kHz, should indicate the result in the units of dB TP." At 44.1 kHz the 4x figure is strictly not dB TP. This is cosmetic: 8x and 32x are also measured and constrained.
- **Fix:** Label it "4x (BS.1770 Annex 2 filter)".
- **Confidence:** high.

### 13. [S5] Docs and contract drift

- `meter.py:1` and spec §3.2 say "BS.1770-4". The current edition is BS.1770-5 (2023-11). For channel-based audio, Annex 1 and Annex 2 are unchanged, and I verified the coefficients against -5.
- `docs/gain-staging.md` §3 "Loudness target" still describes the 1 s plain-dBFS reading with a -7.7 default. §17 superseded it (300 ms AES17, -4.1), and §3 should be marked as superseded.
- `meter.measure` accepts more than two channels with unit weights, with no LFE exclusion and no 1.41 surround weighting: 6 identical channels read +7.8 LU over 1 (`f_edges.out`). Only the API can reach this, because `io.read` rejects more than two channels. Either document it or raise.

### 14. [S5] RLB normalisation reads 0.02 to 0.03 LU above libebur128 / ffmpeg / pyloudnorm-DeMan at 88.2 to 192 kHz (deliberate, and correct)

- **Where:** `meter.py:157-161`.
- **Evidence** (`a_kweight.out`): differences against pyloudnorm's `DeMan` class, which is the libebur128 form, on pink noise, sines and bursts:

  | Sample rate | Difference |
  |---|---|
  | 44.1 kHz | -0.004 LU |
  | 48 kHz | 0.000 LU |
  | 88.2 kHz | +0.020 LU |
  | 96 kHz | +0.022 LU |
  | 176.4 kHz | +0.031 LU |
  | 192 kHz | +0.033 LU |

  - These are exactly 20·log10(a0(48k)/a0(fs)).
  - Against pyloudnorm's default RBJ "K-weighting" class, the difference is +0.03 to +0.044 at every rate.
  - Everything is within ±0.05 LU.
  - Eluate's choice is the literal BS.1770-5 reading: coefficients "chosen to provide the same frequency response that the specified filter provides at 48 kHz". Its deviation from the 48 kHz table response is at most 0.0075 dB from 20 Hz to 20 kHz at every rate.
- **Fix:** None. Mention it in the README so users comparing against ffmpeg's `ebur128` at 96 kHz are not surprised.

### 15. [S5] 24-bit output is rounded without dither (deliberate: gain-staging §11, deviation 3)

- **24-bit rounding:**
  - The industry norm is to dither whenever word length is reduced.
  - At 24 bits, rounding error is about -144 dBFS RMS (2^-23 / √12 relative to full scale), 20 to 30 dB below the self-noise of any converter.
  - Many mastering tools ship with 24-bit dither off or optional.
  - The signal reaching the stage is dense processed float audio.
  - Acceptable.
- **16-bit dither (TPDF, no noise shaping):**
  - This is the safe choice for loud masters: the flat floor of about -101 dBFS sits about 95 dB below programme.
  - Noise shaping would buy 10 to 15 dB of perceived floor, but it moves noise energy towards Nyquist, which lossy encoders spend bits on and which adds a little true peak.
  - The trade-off is reasonable.
  - One refinement: no dither on stretches of exact digital silence (auto-black) for fade tails and gaps. Today, silence becomes ±1 LSB noise at 16-bit (`e_output.out`: 25 % of samples non-zero).

---

## Recommended additions

### report.md: a "Delivery" section after "Loudness" (the report may explain; the numbers are illustrative, from the drive +16 render in `i_abmatch.out`: I -7.48, S95 -7.03, TP -0.30; LRA, S-max and M-max are placeholders)

```
## Delivery

- File: 24-bit PCM WAV, 48000 Hz, stereo, no dither.            (16-bit: "TPDF dither, seed 0")
- Integrated -7.5 LUFS, loudness range 4.1 LU, short-term max -6.9 LUFS, momentary max -6.0 LUFS.
- True peak -0.30 dBTP (4x BS.1770 -0.30, 8x -0.30, 32x -0.30); sample peak -0.31 dBFS; PLR 7.2 dB; PSR p95 6.8 dB.
- Playback on loudness-normalised services (integrated LUFS, track normalisation; reference levels are
  approximate and change):

  | service | reference | playback gain | peak at playback |
  |---|---|---|---|
  | Spotify (Normal) | -14 LUFS | -6.5 dB | -6.8 dBTP |
  | Spotify (Loud) | -11 LUFS | -3.5 dB | -3.8 dBTP |
  | Apple Music (Sound Check) | -16 LUFS | -8.5 dB | -8.8 dBTP |
  | YouTube | -14 LUFS | -6.5 dB | -6.8 dBTP |
  | Amazon Music | -14 LUFS | -6.5 dB | -6.8 dBTP |
  | Tidal | -14 LUFS | -6.5 dB | -6.8 dBTP |
  | Deezer | -15 LUFS | -7.5 dB | -7.8 dBTP |

  (Services that raise quiet tracks: gain = min(ref - I, -1 dBTP - TP) for Spotify; YouTube and Tidal only lower.)
- Lossy-codec headroom: true peak -0.30 dBTP is above the -1 dBTP that Spotify and AES TD1008 recommend for
  lossy delivery, and above Spotify's -2 dBTP for masters louder than -14 LUFS. For streaming uploads use
  --streaming (-1 dBTP) or a -2 dBTP ceiling.
```

### report.json: a new `delivery` object plus one structured flag

```
"delivery": {
  "fs": 48000, "bits": 24, "subtype": "PCM_24", "dither": "none", "dither_seed": null,
  "integrated_lufs": -7.48, "lra_lu": 4.1, "short_term_max_lufs": -6.9, "momentary_max_lufs": -6.0,
  "true_peak_max_dbtp": -0.30,            // max(4x, 8x, 32x): the constrained reading
  "true_peak_4x_dbtp": -0.30, "sample_peak_dbfs": -0.31, "plr_db": 7.2, "psr_p95_db": 6.8,
  "platforms": [
    {"name": "spotify", "ref_lufs": -14, "gain_db": -6.5, "peak_at_playback_dbtp": -6.8,
     "tp_recommended_dbtp": -2.0, "tp_ok": false},
    {"name": "apple_music", "ref_lufs": -16, ...}, {"name": "youtube", "ref_lufs": -14, ...},
    {"name": "amazon_music", "ref_lufs": -14, ...}, {"name": "tidal", "ref_lufs": -14, ...},
    {"name": "deezer", "ref_lufs": -15, ...}],
  "codec_headroom_ok": false               // true_peak_max_dbtp <= -1.0
}
flags_structured += {"kind": "delivery", "value": -0.30, "target": -2.0, "tol": 0.0, "short": true,
                     "text": "True peak -0.30 dBTP at -7.5 LUFS: above Spotify's -2 dBTP for loud masters."}
```

Put the platform reference levels in one table, for example `evaluate.PLATFORMS`, dated, so they can be updated without touching code paths.

### UI: values only, per the brief's GUI-text rule (`uidata.from_render`, `summary`, `drawStats`)

- `Integrated` -7.5 LUFS <small>Before -19.4 LUFS</small>
- `Short-term max` -6.9 LUFS
- `True peak` showing `true_peak_max_dbtp` (not 8x), amber when the `delivery` flag is short
- `Spotify` -6.5 dB and `Apple` -8.5 dB (playback gain); other services on hover or in the batch CSV
- `Format` 24-bit 48 kHz
- Batch row: an `integrated` column next to `loudness`; the CSV export (`ui/app.js:509`) gains `integrated`, `tp_max` and `spotify_gain`.

---

## Standards compliance table

| Requirement | Status | Evidence |
|---|---|---|
| BS.1770-5 Table 1 and Table 2 K-weighting at 48 kHz | Pass (< 1e-14) | `test_k_weighting_matches_bs1770_tables_at_48k`; `a_kweight.out` |
| Other rates: "same frequency response that the specified filter provides at 48 kHz" | Pass: ≤ 0.0075 dB from 20 Hz to 20 kHz at 44.1 / 88.2 / 96 / 176.4 / 192 kHz | `a_kweight.out` |
| 997 Hz at 0 dBFS in one channel reads -3.01 LUFS | Pass: -3.011 (44.1 kHz) to -3.005 (192 kHz); stereo 0.00 | `a_kweight.out` |
| Channel weights G_i | Pass via `weights` (the 5-channel case passes); default is unit weights with no LFE handling | `test_meter.py:115`; finding 13 |
| Block of 400 ms "to the nearest sample", 75 % overlap | Pass: length round(0.4 fs), block j starts at round(0.1 j fs) | `meter.py:185-200` |
| "Incomplete gating blocks at the end of the measurement interval are not used" | Pass: a 1.05 s input gives 7 blocks, the last ending at 1.0 s | BS.1770-5 Annex 1; `b_gating.out` |
| Absolute gate -70 LKFS, relative gate -10 LU, strict ">" | Pass | `meter.py:203-212` |
| Tech 3341 cases 1-5, integrated within ±0.1 LU, at 44.1 / 48 / 96 kHz | Pass; worst error -0.022 LU; matches pyloudnorm-DeMan | `b_gating.out` |
| Tech 3341 case 6 (5 channels) | Pass | repo test |
| Tech 3341 cases 7 and 8 (authentic programme) | Not testable: files absent | n/a |
| Tech 3341 case 9 (constant S) and case 12 (constant M) | Pass: S -22.99, M -22.96 to -22.97 | `b_gating.out` |
| Tech 3341 cases 10 and 13 (max S / M over every window position) | Pass (≤ 0.006 LU). A 100 ms hop would miss case 13 by 0.46 LU, so the 1-sample hop matters | `b_gating.out` |
| Tech 3341 cases 15-23 (true peak within +0.2 / -0.4 dB) | Pass at 4x and 8x | `test_tech3341_true_peak_cases` |
| Tech 3342 cases 1-4 (LRA within ±1 LU) | Pass, exact, at 44.1 / 48 / 96 kHz; agrees with pyloudnorm | `b_gating.out` |
| Tech 3342 cases 5 and 6 (programme) | Not testable | n/a |
| Tech 3342 gates (-70 LUFS absolute, -20 LU relative) and the 10th / 95th percentiles | Pass (libebur128 index rule) | `meter.py:254-265`; ramp test |
| BS.1770-5 Annex 2 FIR (48 taps, 4 phases) | Pass: bit-exact against the table parsed from the PDF | `c_truepeak.out` ("max \|diff\| 0.0") |
| Annex 2 "oversampled sampling rate of at least 192 kHz" | 4x at 44.1 kHz gives 176.4 kHz (finding 12); 8x and 32x comply | BS.1770-5 Annex 2 |
| True peak is never below the sample peak | Pass | `meter.py:360-361`; repo test |
| Ceiling on the file as written, on 4x, 8x, 32x and a 64x reference | Pass at -1.0 dBTP with 16-bit TPDF and at -0.3 dBTP with 24-bit, at 44.1 and 48 kHz, with +6 and +12 dB drive. Worst at the -0.3 ceiling: 4x -0.3049, 64x -0.3008 | `d_ceiling.out` |
| Delivery: -1 dBTP for lossy codecs (Spotify, AES TD1008 / TD1004) | Only with `--streaming`; the default master is -0.3 dBTP; not reported | finding 1 |
| Delivery: Spotify's -2 dBTP when louder than -14 LUFS | Not met in either mode; not reported | finding 1 |
| 16-bit TPDF, ±1 LSB, white, deterministic | Pass: on silence P(-1, 0, +1) = 0.125 / 0.750 / 0.125; error variance 0.250 LSB²; flat PSD (-49.8 dB re LSB²/Hz in every band); L/R correlation 0.002; seeded; prefix-stable | `e_output.out` |
| master.wav holds exactly the samples the solver measured | Pass: PCM_16, PCM_24 and FLOAT read back bit-identical; the float32 cast is exact on 16-bit and 24-bit grids | `e_output.out`; `solver.py:1130-1141` |

## Verified correct

- **K-weighting and loudness**
  - Pre-warped bilinear K-weighting at all six rates.
  - The RLB numerator scaling a0(48k)/a0(fs) does what the code says: +0.0433 dB passband at every rate. Without it the reading would drift by +0.004 dB at 44.1 kHz, -0.022 dB at 96 kHz and -0.032 dB at 192 kHz.
  - Mono input (1-D) and stereo input are both handled.
- **Cross-check against pyloudnorm** (pink noise, sines at 60 Hz, 997 Hz, 5 kHz and 12 kHz, and noise bursts, at six rates):
  - against the DeMan class: ±0.004 at 44.1 / 48 kHz, +0.020 to +0.033 at 88.2 to 192 kHz;
  - against the default class: +0.030 to +0.044;
  - all within ±0.05 LU.
- **Time windows**
  - Momentary and short-term use full windows only, time-stamped at the window end.
  - The one-sample-hop maxima are correct for Tech 3341 cases 10 and 13.
  - `psr`'s `maximum_filter1d(origin = -(L//2))` equals max(env[i:i+L]) for both parities of L.
- **8x Kaiser prototype:** 833 taps, β 5.65; flat within ±0.006 dB to 0.93 × Nyquist, -6.02 dB at 0.965, -62 dB at 1.0, and at least 60.2 dB of image rejection over 1.0 to 8 × Nyquist, as documented.
- **Annex 2 filter as an interpolator:**
  - +0.10 dB ripple at 0.5 × Nyquist, -1.26 dB at 0.9, -6.2 dB at Nyquist, and only 13 dB of rejection at 1.07 × Nyquist.
  - So it over-reads an fs/4 sine at 45° by 0.045 dB.
  - On a +0.5-sample band-limited impulse, 4x under-reads by 0.80 dB and 8x by 0.31 dB, while 32x is within 0.004 dB of the 64x reference.
  - On worst-phase sines at 0.375 to 0.47 fs, 4x reads -0.11 to +0.03 dB and 8x -0.04 to +0.01 dB.
- **Edge handling:**
  - Reconstruction ringing into the surrounding silence counts towards the true peak; this is conservative.
    - A 1 kHz cosine starting at +1 reads +1.13 (32x), +1.08 (8x) and +0.99 dBTP (4x).
    - An abrupt fs/4 "++--" burst reads +3.10 dBTP.
  - `true_peak_fft_envelope` matches the 64x reference to within 0.001 dB.
- **Re-limit:** It holds the maximum of the 4x, 8x and 32x envelopes, so the 4x Annex 2 reading that platforms use stays at or under the ceiling. After quantisation the measured margin is 0.0047 to 0.0049 dB on 4x, and an independent 64x reading sits 0.0006 to 0.002 dB under the ceiling. The 0.005 dB re-limit margin covers 16-bit TPDF, which adds at most 1.5 LSB, about 0.0005 dB at -1 dBFS.
- **Output stage:**
  - Rounds half to even and clips to [-q, q-1]: +1.0 becomes 32767 / 8388607.
  - u1 - u2 is exact ±1 LSB TPDF with zero bias.
  - PCG64 noise is drawn in blocks and is prefix-stable.
  - Dither lowers the harmonics of a -90 dBFS tone at 16-bit from -22.7 dB to -57.9 dB.
  - Output is at the input sample rate; there is no SRC.
- **io:**
  - PCM_24 is written as int32 << 8 and PCM_16 as int16, both exact.
  - Reading a 24-bit file through float32 is exact.
  - Float input above 0 dBFS is preserved.
  - NaN or Inf raises.
  - Mono is duplicated; more than two channels raises.
- **A/B files:** 32-bit float, so they cannot clip, and correctly undithered. Only the louder file is attenuated; the quieter one is bit-identical. The matched S95 values agree within 0.01 LU, the difference file is exact, and the gate-aware fallback to integrated works.
- **Edge cases:**
  - Silence gives -inf / NaN with no crash.
  - Under 400 ms: integrated is -inf and PLR is inf, written as null in JSON.
  - 0.4 to 3 s: S95 falls back to momentary windows; LRA and S-max are NaN.
  - Empty and one-sample inputs are handled.
  - 8 kHz and 192 kHz are handled.
  - NaN raises ValueError.
  - The UI meter works on a 0.1 s input.
  - The UI's 1400-column meter p95 agrees with `rms_meter` within 0.002 dB.
- **`rms_meter` calibration:**
  - A -6 dBFS 1 kHz sine reads -6.00.
  - -6 dBFS 1 kHz 200 ms bursts every 2 s read -7.76, matching §17's "a flat 300 ms window predicts -7.8". Ozone reads -7.4, a known 0.36 dB difference on sparse bursts.
- **What the reading means in LUFS terms:**
  - On a kick / bass / hat / pad loop, the 300 ms meter sits 5.2 dB above integrated before and after limiting (`d_ceiling.out`, `i_abmatch.out`).
  - On other signals the gap is -0.2 dB (pink noise), +0.9 (chords), +2.5 (200 Hz hits) and +6.3 (sub-heavy sections).
  - S95 sits 0.45 to 1.5 LU above integrated on the limited masters.
  - At the default -4.1 target, electronic material lands at about -7.5 to -9.3 LUFS integrated.

## Open questions

1. **Platform reference levels.**
   - Spotify's values come from its own page: -14 LUFS, Loud -11, Quiet -19, -1 dBTP, -2 dBTP above -14 LUFS, and a Loud-mode limiter at -1 dB sample.
   - Apple (-16, Sound Check), YouTube, Amazon and Tidal (-14) and Deezer (-15) are not published by the platforms. These figures come from consistent third-party measurements (iZotope, LANDR, Production Expert, MeterPlugs) and can change.
   - Label them "approximate" with a date in report.md.
2. **AES TD1008 (2021) true-peak wording.** I could not fetch it: the PDF copy I found was unreachable.
   - Secondary sources agree on -16 LUFS overall for music, with the loudest track at -14 under album normalisation, and -18 for speech.
   - One quotes "Maximum True Peak level not exceed -1 dBTP at the codec input"; TD1004 said -1 dBTP.
   - Confirm against the AES PDF before quoting it in report.md.
3. **Ingest meters.** libebur128 and ffmpeg's `ebur128` use their own polyphase interpolator, and only 2x at 96 kHz and above. Their passband ripple may read a few hundredths above Annex 2 on limiter-shaped peaks. With the 0.005 dB re-limit margin, a -1.000 dBTP master could read about -0.99 on another meter. Consider a 0.05 dB margin under `--streaming`, plus a CI check with `ffmpeg -af ebur128=peak=true`.
4. **Integrated LUFS in the UI.** Should the report and UI show integrated LUFS next to the maintainer's meter? The meter remains the target by design (§17), but users and platforms think in LUFS.
5. **24-bit `--streaming`** (finding 6). Is the maintainer happy with it? It changes the README and §3.
6. **Ozone's RMS bar.** Does it high-pass or remove DC (finding 9)? If it does, `rms_meter` should too.

## Sources

- ITU-R BS.1770-5 (11/2023), fetched and parsed: https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-5-202311-I!!PDF-E.pdf (Annex 1 Tables 1 and 2 and the gating text; Annex 2 coefficient table and "at least 192 kHz").
- Spotify for Artists, "Loudness normalization on Spotify": https://support.spotify.com/artists/article/loudness-normalization/
- AES TD1008 summaries: https://productionadvice.co.uk/td1008/ ; https://www.grimmaudio.com/news/aes-td1008-ends-the-loudness-war/ ; https://www.production-expert.com/production-expert-1/aes-update-loudness-recommendations-for-audio-only-streaming
- Platform reference levels (third-party): https://www.izotope.com/en/learn/mastering-for-streaming-platforms ; https://blog.landr.com/lufs-loudness-metering ; https://www.production-expert.com/production-expert-1/apple-choose-16lufs-loudness-level-for-apple-music-heres-why ; https://meterplugs.com/blog/2019/10/15/loudness-penalty-amazon-deezer.html
- Apple Digital Masters, 24-bit and -1 dBTP (third-party summary): https://www.sonarworks.com/blog/learn/mastering-for-apple-digital-masters
- EBU Tech 3341 / 3342 case definitions: as encoded in `tests/test_meter.py` (Tech 3341, 2023, Table 1).
