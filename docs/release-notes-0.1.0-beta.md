# Eluate 0.1.0-beta

Eluate is a free, offline auto-mastering app for Windows. It analyses the whole track before
it processes anything, then plans every gain change ahead instead of reacting to the signal.

## In this beta

- **Desktop app** with a command line (`eluate master`) behind it: open one track or a whole
  folder (batch, one render at a time), Render, then compare before / after on the level
  timeline and spectrum.
- **Profiles**: Electronic (default, loudness target -4.1 dB RMS) and Live (acoustic and live
  material, quieter default, tone target from the mix's own slope). Delivery **Master**
  (24-bit, -0.3 dBTP) or **Streaming** (16-bit with dither, -1 dBTP). Character presets set
  the dynamics' timing (Auto picks from the tempo).
- **Stems**: master a stem set as its sum, with an optional null test against the mix, a
  report naming the stem behind each limit, and up to two correction passes on the stems
  (EQ, gentle compression, bass ducked under the kick) before mastering the re-sum again.
- **Blind A/B**: the level-matched before / after play as A and B in a random order; vote
  Prefer A / B / Can't tell. Votes are kept in the track's work folder.
- **Opt-in vote sharing**: off unless you say yes. Only the vote, the settings and the level
  measurements are sent, never audio, file names or paths. Privacy: jective.nz/eluate/privacy.

## Platforms

| Platform | Status |
|---|---|
| Windows 10 / 11 x64 | Tested. Installer attached. |
| macOS 11+ (Apple Silicon only) | **Untested.** Builds; no render has been run from the packaged app. |
| Linux x64 (`.deb`, `.AppImage`; Ubuntu 24.04 build) | **Untested.** Builds; package contents not verified. |

macOS and Linux packages, where attached, are provided as is for early testers. Expect
failures; reports welcome (issues). No Intel Mac build; no `.rpm` for this beta (the
AppImage runs on Fedora / openSUSE).

## Known limits

- Every installer is unsigned.
  - Windows: SmartScreen warns; choose **More info** → **Run anyway**.
  - macOS: ad-hoc signed, not notarised. Gatekeeper blocks the first launch; open
    System Settings → Privacy & Security → **Open Anyway** (on macOS 15, right-click →
    Open no longer bypasses it).
- Linux `.deb`: the beta's package version sorts after a later `0.1.0`; uninstall the beta
  before installing a final release.
- Memory: a render holds the whole track, about 3 GB for a 6-minute 44.1 kHz track and about
  10 GB for 10 minutes at 96 kHz. The engine drops to one thread, or refuses with a message,
  when the machine does not have that much free.
- Beta: the Live profile's loudness default is not calibrated yet, and output may change
  between versions.

## Source

AGPL-3.0-or-later. The source of this release is the tag
[v0.1.0-beta](https://github.com/subconsciousnz-viber/eluate/tree/v0.1.0-beta); third-party
licences are in THIRD_PARTY_NOTICES.txt, installed with the app.
