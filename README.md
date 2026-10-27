# Eluate

**Offline adaptive dynamics solver with infinite lookahead.**

Offline auto-mastering engine and desktop app. Whole-track analysis drives
precomputed gain envelopes and constraint-based processing.

**0.1.0-beta — first public beta.** Windows tested; macOS (Apple Silicon) and Linux builds untested.
Source development uses Python 3.12.

## Install

Download the installer from [GitHub Releases](../../releases) (macOS and Linux packages, where attached, are untested).
The installer is unsigned: in Windows SmartScreen, select **More info** → **Run anyway**.

## CLI quickstart

From a source checkout, in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
eluate master input.wav -o out/ --target-rms -4.1 --bpm 120
```

`--target-rms` and `--bpm` are optional. Output includes `master.wav`, a saved plan,
measurement reports, a process log and level-matched A/B audio.

For stems, supply aligned files at the same sample rate and length:

```powershell
eluate master --stems drums.wav bass.wav other.wav -o out-stems/
```

Add `--mix mix.wav` to verify that the stems sum to the supplied mix before mastering.
Input files are not modified. Run `eluate master --help` for all controls.

## Build the desktop app

Requires Rust, Node.js and the Python 3.12 environment above. From `desktop/`, run
`npm install`, `npm run engine`, then `npm run build`.
See [desktop/README.md](desktop/README.md) for build and development details.

## Tests

```powershell
pytest -m "not slow"  # Fast tier
pytest               # Full suite
```

Download the EBU test vectors from `tech.ebu.ch` into `fixtures/ebu/`; they are not
redistributed. Tests requiring those vectors or private audio skip when it is absent.
CI runs the fast tier on pull requests and the full suite weekly or on manual dispatch.

## Contributing

Sign off commits under the DCO with `git commit -s`; see [CONTRIBUTING.md](CONTRIBUTING.md).
For security reports, see [SECURITY.md](SECURITY.md).

## License

[AGPL-3.0-or-later](LICENSE). Source for each release is the matching git tag.
Bundled dependencies retain their own licenses; see [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt).

Copyright (C) 2026 the Eluate contributors

## Appendix

- **Profiles:** `electronic` is the default; `--profile live` uses gentler processing
  and remains uncalibrated. Stems mode automatically uses a separate profile with
  tighter processing limits.
- **Loudness:** `--target-rms` measures the loudest passage using 300 ms AES17 RMS,
  not integrated LUFS. The electronic default is -4.1 dBFS. Processing limits can
  prevent reaching a target; reports record shortfalls.
- **Delivery:** default output is 24-bit PCM with a -0.3 dBTP ceiling;
  `--streaming` selects 16-bit PCM with TPDF dither and a -1 dBTP ceiling.
- **Pipeline:** `analyse`, `plan`, `render` and `report` expose individual stages;
  `master` runs the complete workflow. See the [engine specification](docs/spec-draft.md)
  and [gain-staging design](docs/gain-staging.md) for rationale and signal-chain detail.
