# Contributing to Eluate

## Setup

Follow the source setup in [README.md](README.md) (Python 3.12 venv, `pip install -e ".[dev]"`);
the desktop shell's build is in [desktop/README.md](desktop/README.md).

## Tests

- `pytest -m "not slow"`: the fast tier (a few minutes); run it before every pull request.
- `pytest`: the full suite, including the `slow` marker; CI runs it weekly.
- Tests that need the EBU loudness vectors (`fixtures/ebu/`, from tech.ebu.ch) or private
  premasters (`fixtures/premasters/manifest.toml`, see `manifest.example.toml`) skip when
  those files are absent.
- Desktop shell: `cargo test` in `desktop/src-tauri`; UI: `node --check ui/app.js`.

## Sign-off (DCO)

Contributions are accepted under the [Developer Certificate of Origin](https://developercertificate.org/):
sign off every commit with `git commit -s`, which adds a `Signed-off-by:` line stating you
have the right to submit the work under the project's licence (AGPL-3.0-or-later).

## No third-party audio or track names

Never commit audio, renders or analysis data of music you do not own, and never put other
artists' names or track titles in code, tests, docs, file names or commit messages. Tests
use synthetic signals (`fixtures/`) or refer to real material by label (Track A, Track B, ...)
through the gitignored manifest.
