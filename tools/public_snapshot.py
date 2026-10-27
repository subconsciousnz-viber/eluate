"""Export the public tree of this repository into a directory and prove it clean.

The public repository is made from one squashed commit of this export. The export is every
tracked file minus EXCLUDE (internal working documents), and the result is scanned for names
and paths that must never be published; any hit fails the run (exit 1) and lists file:line.

    .venv/Scripts/python -I tools/public_snapshot.py OUT_DIR [--ref HEAD | --worktree]
    .venv/Scripts/python -I tools/public_snapshot.py --check       # scan only, no export

--ref (default HEAD) exports the files as committed at that ref (`git archive`); --worktree
copies the working tree instead, including untracked files that are not gitignored, so a
change can be checked before it is committed. OUT_DIR must not exist or be empty.

Patterns: one regular expression per line (case-insensitive; blank lines and lines starting
with # ignored) in docs/dev/public-snapshot-patterns.txt. That file is itself excluded from
the export, so the names it lists are not published by the tool that guards them. Built in:
the word "owner" (public text says "the maintainer"), except in licence texts and on lines
that say "copyright owner".
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATTERNS_FILE = "docs/dev/public-snapshot-patterns.txt"

# Internal documents: never exported. A trailing "/" excludes a directory, otherwise a glob.
EXCLUDE = (
    ".archive/",                          # old session handoffs
    "SESSION_HANDOFF_*",                  # session handoffs
    "audit/",                             # audit briefs, raw auditor replies, triage logs, providers.toml
    "docs/dev/",                          # release checklist, snapshot patterns
    "docs/audit-protocol.md",             # internal multi-model audit process
    "docs/for-auditors.md",
    "docs/sprint-*.md",                   # sprint logs
    "docs/quality-review-*.md",
    "docs/macos-handoff.md",
    "docs/macos-report-verdicts.md",
    "docs/reuse-survey.md",
    "macos-claude-report.md",
    "tools/audit_dispatch.py",            # the internal audit process's dispatcher (audit/providers.toml)
    "trash/",
    "work/",
)

# The "owner" rule does not apply to licence texts (they say "copyright owner" and the like).
OWNER_EXEMPT = (
    "LICENSE",
    "THIRD_PARTY_NOTICES.txt",
    "tools/licenses/*",
    "ui/fonts/*",
    "tools/public_snapshot.py",
)
OWNER_RE = re.compile(r"\bowners?\b", re.I)
COPYRIGHT_OWNER_RE = re.compile(r"copyright[ -]owner", re.I)


def excluded(rel: str) -> bool:
    for pat in EXCLUDE:
        if pat.endswith("/"):
            if rel.startswith(pat):
                return True
        elif fnmatch.fnmatchcase(rel, pat):
            return True
    return False


def _git(*args: str, root: Path = ROOT) -> bytes:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout


def public_files(root: Path = ROOT, worktree: bool = True) -> list[str]:
    """Tracked files (and, for the working tree, untracked ones not ignored) minus EXCLUDE."""
    args = ["ls-files", "-z", "--cached"] + (["--others", "--exclude-standard"] if worktree else [])
    names = {n for n in _git(*args, root=root).decode("utf-8").split("\0") if n}
    out = sorted(n for n in names if not excluded(n))
    return [n for n in out if (root / n).is_file()] if worktree else out


def load_patterns(root: Path = ROOT) -> list[re.Pattern]:
    p = root / PATTERNS_FILE
    if not p.is_file():
        sys.exit(f"public_snapshot: {PATTERNS_FILE} is missing; refusing to export unchecked")
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    pats = [re.compile(ln, re.I) for ln in lines if ln and not ln.startswith("#")]
    if not pats:
        sys.exit(f"public_snapshot: {PATTERNS_FILE} has no patterns")
    return pats


def scan(base: Path, rels: list[str], pats: list[re.Pattern]) -> list[str]:
    """Every hit as 'path:line: [pattern] text'. Binary files are checked by name only."""
    hits = []
    for rel in rels:
        for pat in pats:
            if pat.search(rel):
                hits.append(f"{rel}: [file name: {pat.pattern}]")
        data = (base / rel).read_bytes()
        if b"\0" in data:
            continue
        text = data.decode("utf-8", errors="replace")
        owner_ok = any(fnmatch.fnmatchcase(rel, g) for g in OWNER_EXEMPT)
        for i, line in enumerate(text.splitlines(), 1):
            for pat in pats:
                if pat.search(line):
                    hits.append(f"{rel}:{i}: [{pat.pattern}] {line.strip()[:160]}")
            if not owner_ok and OWNER_RE.search(line) and not COPYRIGHT_OWNER_RE.search(line):
                hits.append(f"{rel}:{i}: [owner] {line.strip()[:160]}")
    return hits


def export(out: Path, ref: str | None) -> list[str]:
    """Write the public files into out; returns their relative paths."""
    if ref is None:
        rels = public_files(ROOT, worktree=True)
        for rel in rels:
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dst)
        return rels
    rels = public_files(ROOT, worktree=False)
    tar = _git("archive", "--format=tar", ref, "--", *rels)
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        tf.extractall(out, filter="data")
    return rels


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", type=Path, help="target directory (must not exist or be empty)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--ref", default="HEAD", help="export this commit (default HEAD)")
    src.add_argument("--worktree", action="store_true", help="export the working tree instead")
    ap.add_argument("--check", action="store_true", help="scan the working tree's public files only")
    a = ap.parse_args(argv)

    pats = load_patterns()
    if a.check:
        base, rels = ROOT, public_files(ROOT, worktree=True)
    else:
        if a.out is None:
            ap.error("OUT_DIR is required unless --check")
        out = a.out.resolve()
        if out.exists() and any(out.iterdir()):
            sys.exit(f"public_snapshot: {out} is not empty")
        if out == ROOT or ROOT in out.parents:
            sys.exit("public_snapshot: OUT_DIR must be outside the repository")
        out.mkdir(parents=True, exist_ok=True)
        export(out, None if a.worktree else a.ref)
        # scan what actually landed, not the list we meant to write
        base = out
        rels = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
        leaked = [r for r in rels if excluded(r)]
        if leaked:
            print("public_snapshot: FAILED, excluded files in the export:", *leaked, sep="\n  ")
            return 1

    hits = scan(base, rels, pats)
    if hits:
        print(f"public_snapshot: FAILED, {len(hits)} hit(s) in {base}:")
        for h in hits:
            print("  " + h)
        return 1
    print(f"public_snapshot: clean, {len(rels)} files scanned against {len(pats)} patterns + owner rule ({base})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
