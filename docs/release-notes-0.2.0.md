# Eluate 0.2.0

Eluate is a free, offline auto-mastering app. It analyses the whole track before it
processes anything, then plans every gain change ahead instead of reacting to the signal.

0.2.0 carries the November 2026 audit sprint: ten evidence-based reviews of 0.1.0-beta and
the fixes that came out of them. The findings, measurements and every fix are in
[`docs/audit-2026-11-10.md`](audit-2026-11-10.md) and the area reports beside it.

**Masters made with 0.2.0 differ from 0.1.0-beta's.** The loudness target is still met; the
changes below move limiter gain reduction by about 0.1 to 0.3 dB and sub-band thresholds by
0.6 to 1.3 dB at the same meter reading, and they remove the audio-rate distortion described
under Sound.

## Sound

- **Audio-rate automation removed from the dynamics.** The envelope smoother gains a hold
  window (one period of the lowest frequency of interest), the band ceilings get per-band
  detector windows and releases, the glue's control-rate attack has a 3 ms floor, and notch
  depth envelopes are no longer quantised. On low sub material the final limiter's ripple and
  odd-order distortion at the fast and aggressive characters were the largest of these.
- **Lowest crossover:** zero phase stays the default, as in 0.1.0-beta (the maintainer's
  blind-test ruling). Minimum phase (Linkwitz-Riley) is available as the opt-in
  `--xover-phase minimum`.
- **Multiband:** a band's threshold floor now sits on the band's level at the loud hits.
- **Live profile calibrated:** target -11.5 dBFS on the engine's meter (about -14.5 LUFS
  integrated on band material), no clipper under the Auto character, a sustained-material
  rule, a conditional side strip, and EQ bounds of ±1 dB. The calibration uses synthetic
  fixtures and published statistics; real live masters may still move it.
- Fixes across the planner and solver, analysis features and stems (audit sections 2.2 to 2.4).

## Delivery and metering

- `--streaming` now writes 24-bit PCM at a -1 dBTP ceiling (it was 16-bit with dither at
  the input's rate).
- New `--cd`: resamples to 44.1 kHz before the chain and writes 16-bit PCM with TPDF dither at
  -1 dBTP, so the ceiling holds at the delivery rate. `--streaming` and `--cd` are exclusive;
  `--cd` is not yet available with `--stems`.
- report.md gains a Delivery section (format, integrated loudness, each platform's playback
  gain, codec-headroom rules); report.json gains a `delivery` object. Its flag is amber only
  for a streaming or CD delivery that breaks a rule. The default master's ceiling stays
  -0.3 dBTP.
- The true peak shown by the app, the batch row and the process log is the worst of the 4x,
  8x and 32x readings, and the app's level match is the A/B files' own, so the drawn
  premaster and the audio heard agree.
- Metering is about 1.5x faster, with identical numbers.

## App

- UI fixes from the audit (section 2.7): the report, delivery data, character and votes in
  the results view.
- Vote sharing: the consent text names every field a shared vote carries, and sharing reads
  as off only after the outbox is cleared.
- Engine processes now end with the app: a Job Object on Windows and `PR_SET_PDEATHSIG` on
  Linux. macOS has no equivalent yet (desktop/README.md).
- The engine reads its output pipes to the end whatever bytes they carry, paths are
  canonical (with a fallback for RAM-disk, cloud-sync and SMB volumes), and the frozen
  engine no longer bundles test suites or scikit-learn.

## Platforms

| Platform | Status |
|---|---|
| Windows 10 / 11 x64 | Installer built, installed silently, rendered from and uninstalled in CI (windows-latest). |
| macOS 11+ (Apple Silicon only) | Builds; the packaged app launches and its engine renders. Ad-hoc signed, not notarised. |
| Linux x64 (`.deb`, `.AppImage`) | **Untested** for 0.2.0. |

No Intel Mac build.

## Known limits

- Every installer is unsigned.
  - Windows: SmartScreen warns; choose **More info** → **Run anyway**.
  - macOS: Gatekeeper blocks the first launch; open System Settings → Privacy & Security →
    **Open Anyway**.
- Memory: a render holds the whole track, about 3 GB for a 6-minute 44.1 kHz track and about
  10 GB for 10 minutes at 96 kHz.
- No real premaster corpus was measured in the sprint; every number in the audit comes from
  synthetic signals.

## Source

AGPL-3.0-or-later. The source of this release is the tag
[v0.2.0](https://github.com/subconsciousnz-viber/eluate/tree/v0.2.0); third-party licences
are in THIRD_PARTY_NOTICES.txt, installed with the app.
