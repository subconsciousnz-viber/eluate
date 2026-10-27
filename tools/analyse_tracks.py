"""Run the analysis pass on a list of audio files and write one JSON plus a summary table.

Usage:
    .venv/Scripts/python tools/analyse_tracks.py --out work/analysis file1.wav file2.wav ...

Audio stays where it is; only numbers are written. Arrays in the result are summarised
(length, min, max) so the JSON stays small; the full result is pickled next to it.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from eluate import features, io


def summarise(obj):
    if isinstance(obj, dict):
        return {k: summarise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if obj and all(isinstance(v, (int, float, np.floating, np.integer)) for v in obj) and len(obj) > 16:
            a = np.asarray(obj, dtype=float)
            return {"n": int(a.size), "min": float(np.nanmin(a)), "max": float(np.nanmax(a)), "mean": float(np.nanmean(a))}
        return [summarise(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.size > 16:
            a = obj.astype(float)
            return {"shape": list(obj.shape), "min": float(np.nanmin(a)), "max": float(np.nanmax(a)), "mean": float(np.nanmean(a))}
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="work/analysis")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in args.files:
        p = Path(f)
        t0 = time.time()
        x, fs = io.read(str(p))
        r = features.analyse(x, fs)
        dt = time.time() - t0
        stem = p.stem[:40].replace(" ", "_")
        with open(out / f"{stem}.pkl", "wb") as fh:
            pickle.dump(r, fh)
        with open(out / f"{stem}.json", "w", encoding="utf-8", newline="\n") as fh:
            json.dump(summarise(r), fh, indent=1)
        pi = r.get("plan_inputs", {})
        lo = r.get("loudness", {})
        st = r.get("structure", {})
        ti = r.get("tilt", {})
        rows.append({
            "track": p.stem[:40], "fs": fs, "dur_s": round(x.shape[0] / fs, 1), "analyse_s": round(dt, 1),
            "S95": pi.get("s95_lufs", lo.get("s95")), "integrated": lo.get("integrated_lufs", lo.get("integrated")),
            "true_peak_dbtp": lo.get("true_peak_dbtp", lo.get("true_peak")),
            "tempo": st.get("tempo_bpm"), "grid": st.get("grid_kind"), "sections": len(st.get("segments", []) or []),
            "slope_db_oct": ti.get("slope_db_oct", ti.get("slope")), "key": (pi.get("key") or {}).get("label"),
            "keyeq_eligible": pi.get("keyeq_eligible"), "resonances": len(pi.get("notches", []) or []),
            "side_strip_hz": pi.get("side_strip_hz"),
        })
        print(json.dumps(rows[-1]), flush=True)
    with open(out / "summary.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
