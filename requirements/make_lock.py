"""Write this platform's lock files from the active environment (report 3.5).

Run with the venv's Python after `pip install -e ".[dev]"` and a passing test suite:

    python requirements/make_lock.py

It walks pyproject.toml's `dependencies` (runtime) and the `dev` extra through the
installed packages' own requirements (markers evaluated for this interpreter) and writes
exact pins to requirements/lock-<os>-<arch>-py<XY>.txt and
requirements/lock-dev-<os>-<arch>-py<XY>.txt (dev = what the dev extra adds beyond the
runtime set). eluate itself and anything installed but not required (PyInstaller,
stray packages) are left out. It fails if a required package is not installed.
"""

from __future__ import annotations

import platform
import sys
import tomllib
from importlib import metadata
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parent.parent
OS = {"win32": "win", "darwin": "macos"}.get(sys.platform, sys.platform)
ARCH = {"amd64": "x64", "x86_64": "x64", "aarch64": "arm64"}.get(
    platform.machine().lower(), platform.machine().lower())
TAG = f"{OS}-{ARCH}-py{sys.version_info.major}{sys.version_info.minor}"


def closure(roots: list[str]) -> dict[str, tuple[str, str]]:
    """{canonical name: (display name, installed version)} of roots and their requirements."""
    env = default_environment()
    seen: dict[str, tuple[str, str]] = {}
    stack = [(Requirement(r), "") for r in roots]
    while stack:
        req, extra = stack.pop()
        if req.marker is not None and not req.marker.evaluate({**env, "extra": extra}):
            continue
        key = canonicalize_name(req.name)
        dist = metadata.distribution(req.name)      # PackageNotFoundError: install first
        if key not in seen:
            seen[key] = (dist.metadata["Name"], dist.version)
            stack += [(Requirement(r), "") for r in dist.requires or []]
        for ex in req.extras:                       # requested extras of this package
            stack += [(Requirement(r), ex) for r in dist.requires or []]
    return seen


def write(path: Path, pins: dict[str, tuple[str, str]], what: str) -> None:
    head = [f"# {what} for {TAG} (Python {platform.python_version()}), made by",
            "# requirements/make_lock.py from the verified venv; see requirements/README.md.",
            "# Exact pins of every package pyproject.toml's requirements reach on this platform."]
    lines = [f"{name}=={ver}" for name, ver in sorted(pins.values(), key=lambda p: p[0].lower())]
    path.write_text("\n".join(head + lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{path.relative_to(ROOT)}: {len(lines)} pins")


def main() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    runtime = closure(project["dependencies"])
    dev_all = closure(project["optional-dependencies"]["dev"])
    dev = {k: v for k, v in dev_all.items() if k not in runtime}
    for pins in (runtime, dev):
        pins.pop("eluate", None)
    write(ROOT / "requirements" / f"lock-{TAG}.txt", runtime, "Runtime lock")
    write(ROOT / "requirements" / f"lock-dev-{TAG}.txt", dev, "Dev lock (beyond runtime)")


if __name__ == "__main__":
    main()
