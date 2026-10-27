# Dependency locks

Maintainer ruling 2026-10-22 (macOS report 3.5): pin the Python dependencies. `pyproject.toml`
keeps ranges (lower bounds plus an upper bound one minor past the verified versions; scipy
held to its verified minor for `oa.py`'s private `_calc_oa_lens`). The exact, verified
sets live here, one pair per platform:

| File | What |
|---|---|
| `lock-<os>-<arch>-py312.txt` | every package `pyproject.toml`'s `dependencies` reach on that platform, `==` pinned |
| `lock-dev-<os>-<arch>-py312.txt` | what the `dev` extra adds on top (pytest, pyloudnorm, matplotlib and theirs) |

Present: `win-x64-py312` (Windows 11 x64, Python 3.12.10). Owed: `macos-arm64-py312`.

## How a lock is made

No uv or pip-tools on the Windows machine, so the lock comes from the venv the suite
passed in:

1. `python -m venv .venv`, then `pip install -e ".[dev]"` (or from an existing lock).
2. Run the full suite; it must pass.
3. `.venv/Scripts/python requirements/make_lock.py` (`.venv/bin/python` on a Mac).

`make_lock.py` walks the requirement tree from `pyproject.toml` through the installed
packages (environment markers evaluated for that interpreter, so Windows-only packages
such as colorama appear only in the Windows lock) and writes both files for the running
platform. It leaves out eluate itself and anything installed but not required
(PyInstaller for the desktop engine build, stray packages); a plain `pip freeze` would not.

## How to install from a lock

```
pip install -r requirements/lock-win-x64-py312.txt && pip install -e . --no-deps
pip install -r requirements/lock-dev-win-x64-py312.txt     # tests
```

`--no-deps` keeps pip from re-resolving eluate's ranges past the pins. Then
`pip check` should print "No broken requirements found."

## The Mac

Same procedure on the M3, with a native arm64 Python 3.12 (`uv python install 3.12` or the
python.org installer; not Homebrew's `python@3.12` on that machine, whose Homebrew is the
Intel install under Rosetta and would give an x86_64 Python; `platform.machine()` must
print `arm64`). Install with `pip install -e ".[dev]"`, run the suite, run
`.venv/bin/python requirements/make_lock.py`, and commit
`lock-macos-arm64-py312.txt` and `lock-dev-macos-arm64-py312.txt`. The Mac's versions may
differ from Windows' (wheels differ by platform); each lock is the set verified there.

## Updating

Raise a bound in `pyproject.toml` only after the suite passes on the new version, then
regenerate the lock on each platform. A plan records the numpy and scipy versions and the
platform (`plan.environment`); renders are bit-exact only within one lock on one platform.
