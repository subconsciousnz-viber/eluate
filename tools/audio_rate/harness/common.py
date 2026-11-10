"""Paths, repo imports, small helpers."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent          # tools/audio_rate
# the checkout under test: this one by default, or ELUATE_REPO (e.g. a worktree of an older commit)
REPO = Path(os.environ.get("ELUATE_REPO", HERE.parents[1]))
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

for p in (REPO / "src", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

FS = 48000


def load_test_module(name: str):
    """Import a tests/*.py module by path (for the drum loop fixture)."""
    path = REPO / "tests" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def db(v):
    v = np.asarray(v, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.maximum(v, 1e-300))


def pdb(p):
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(np.maximum(np.asarray(p, dtype=np.float64), 1e-300))


def save_json(name: str, obj) -> Path:
    path = OUT / name

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, float) and not np.isfinite(o):
            return str(o)
        raise TypeError(type(o))
    with open(path, "w", newline="\n") as f:
        json.dump(obj, f, indent=1, default=conv)
    return path


def table(rows: list[dict], cols: list[str], fmt: dict | None = None) -> str:
    """Markdown table from dicts."""
    fmt = fmt or {}
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            f = fmt.get(c)
            if isinstance(v, float):
                cells.append(f"{v:{f}}" if f else f"{v:.3g}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_md(name: str, text: str) -> Path:
    path = OUT / name
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write(text)
    return path
