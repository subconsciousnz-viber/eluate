"""Audio and text I/O.

Contract:
- `read(path)` returns `(x, fs)` with `x` float32, shape (n, 2). Mono files are
  duplicated to both channels; files with more than two channels raise
  ValueError (the engine is stereo-only), as do NaN or Inf samples (possible
  in float WAV).
- `write(path, x, fs, subtype)` accepts (n,), (n, 1) or (n, 2) arrays of any
  float dtype, enforces stereo the same way, and writes via soundfile. Default
  subtype is FLOAT (32-bit float WAV) so no quantisation happens between stages.
  PCM_16 / PCM_24 are written from integers rounded here (round(x 2^(bits-1)),
  clipped to the integer range; PCM_24 as int32 holding the value << 8), so
  samples already on that grid (stages/output.py) are written exactly, whatever
  libsndfile's own float scaling.
- `write_text` / `write_json` always write UTF-8 with LF line endings,
  regardless of platform (`open(..., newline="\n")`).
- `write_npz(path, **arrays)`: np.savez (read it back with allow_pickle=False).
- Every write is atomic (`atomic_write(path, fn)`, report 3.2): fn writes a temporary
  file in the same directory with the same extension, which then replaces path
  (os.replace), so a stopped run leaves the old file or the new one, never a partial
  one. The temporary file is removed when fn raises.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf


PCM_BITS = {"PCM_16": 16, "PCM_24": 24}


def as_stereo(x: np.ndarray) -> np.ndarray:
    """Return `x` as shape (n, 2). Mono is duplicated; >2 channels is an error."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"audio must be 1-D or 2-D, got shape {x.shape}")
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] != 2:
        raise ValueError(f"stereo only: got {x.shape[1]} channels")
    return x


def read(path: str | Path) -> tuple[np.ndarray, int]:
    """Read an audio file as float32 stereo, shape (n, 2)."""
    x, fs = sf.read(str(path), dtype="float32", always_2d=True)
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{path}: audio contains NaN or Inf")
    return np.ascontiguousarray(as_stereo(x), dtype=np.float32), int(fs)


def atomic_write(path: str | Path, fn: Callable[[Path], Any]) -> Path:
    """fn(tmp) writes a temporary sibling of path (same extension), which then replaces path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{uuid.uuid4().hex[:12]}.tmp{path.suffix}")
    try:
        fn(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def write(path: str | Path, x: np.ndarray, fs: int, subtype: str = "FLOAT") -> None:
    """Write stereo audio: float32 for FLOAT, rounded integers for PCM_16 / PCM_24."""
    if subtype in PCM_BITS:
        q = 2.0 ** (PCM_BITS[subtype] - 1)
        k = np.clip(np.round(np.asarray(as_stereo(x), dtype=np.float64) * q), -q, q - 1.0)
        y = (k.astype(np.int16) if subtype == "PCM_16"
             else k.astype(np.int32) * np.int32(256))
    else:
        y = as_stereo(x).astype(np.float32, copy=False)
    y = np.ascontiguousarray(y)
    atomic_write(path, lambda p: sf.write(str(p), y, int(fs), subtype=subtype))


def write_text(path: str | Path, text: str) -> None:
    """Write UTF-8 text with LF line endings only."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    def fn(p: Path) -> None:
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    atomic_write(path, fn)


def write_npz(path: str | Path, **arrays: np.ndarray) -> None:
    """np.savez, atomically (path should end .npz: np.savez appends it otherwise)."""
    atomic_write(path, lambda p: np.savez(p, **arrays))


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    """Write JSON with LF line endings and a trailing newline."""
    write_text(path, json.dumps(obj, indent=indent, sort_keys=False) + "\n")
