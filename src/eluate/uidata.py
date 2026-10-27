"""Numbers for the desktop UI (ui/): a premaster preview and a finished render.

Contract:
- `preview(path, target_rms)`: the premaster's peak envelope and 300 ms meter series
  (AES17) at COLUMNS points, shifted so its loudest passage (p95 of the meter series)
  sits on the loudness target, i.e. where a level-matched comparison will put it.
- `from_render(work, premaster=None)`: everything the UI draws for a finished `master`
  output dir: the premaster level-matched to the master on the maintainer's meter reading, the master,
  the limiter GR (positive dB), sections with their contrast change against the
  loudest section (flagged when it rose past evaluate.FLAG_SECTION_LU, as report.md; ignored past
  FLAG_SECTION_IGNORE_LU or when a section has no measurable loudness), LTAS M / S / stereo, tilt, loudness and what held it,
  crest (meter.crest_db, read from the report), LRA, true peak, the
  noise floor before and after with its status (None for reports made before noise.py),
  crossovers, the side high-pass, the input EQ high shelves and their limits, the
  paths of master.wav, the A/B files and process-log.html (None when absent), the
  render's `controls` (run.json; None when absent) and `flags` (report.json
  `flags_structured`, rebuilt by evaluate._checks for reports made before it).
  `premaster` is the audio the caller vouches for (the app engine passes the input the
  shell granted, audit 13): it is the one read, and a source.json whose path does not
  resolve to the same file raises ValueError. Without it the premaster is source.json's
  path (the maintainer's own CLI below).
- `work_dir(input)`: the work dir the app writes for a premaster, "<stem> - eluate"
  next to it (the shell's rule, desktop/src-tauri/src/lib.rs).
- `summary(input)`: kind "summary", what a batch row needs, from report.json / run.json / source.json
  without loading audio: track, input, work, state "done", loudness (the maintainer's meter),
  tilt (stereo slope), crest, crestBefore, truePeak, flags, controls, bpm. Done means a
  readable run.json (cli.do_master's completion marker, written last and removed first,
  report 3.1) with a report.json; anything else (no render, a run that failed or was
  stopped) gives {"kind", "track", "input", "work", "state": "none"}.
Numbers only; no audio is copied.

`python -m eluate.uidata <work> ui/data.js` writes the browser-preview data the UI
falls back to outside the desktop shell.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from eluate import evaluate
from eluate.plan import INPUTEQ_HS_MID_MAX_DB, INPUTEQ_HS_SIDE_MAX_DB

COLUMNS = 1400
AES17_DB = 3.0103


def _load(p):
    x, fs = sf.read(str(p), always_2d=True, dtype="float32")
    return x[:, :2], fs


def _columns_peak(x, n):
    a = np.abs(x).max(axis=1)
    edges = np.linspace(0, len(a), n + 1).astype(int)
    return np.array([a[edges[i]:max(edges[i + 1], edges[i] + 1)].max() for i in range(n)])


def _meter_series(x, fs, n, win=0.3):
    p = np.mean(x.astype(np.float64) ** 2, axis=1)
    c = np.concatenate([[0.0], np.cumsum(p)])
    w = min(int(win * fs), len(p))
    centres = np.linspace(w // 2, len(p) - w // 2 - 1, n).astype(int)
    lo = np.clip(centres - w // 2, 0, len(p) - w)
    return 10 * np.log10((c[lo + w] - c[lo]) / w + 1e-20) + AES17_DB


def _meter_p95(series):
    s = series[series > -70.0]
    return float(np.percentile(s, 95)) if s.size else float("nan")


def _r(v, d=2):
    if v is None:
        return None
    v = float(v)
    return round(v, d) if np.isfinite(v) else None


def _db_list(arr):
    return [_r(v) for v in arr]


def _fine(fine):
    """The 1/24-octave display spectrum, 0.1 dB; None in reports made before it existed."""
    if not fine:
        return None
    return {"hz": [round(float(f), 1) for f in fine["centres_hz"]],
            **{k: {v: _db_list(fine[k][v]) for v in fine[k]} for k in ("before", "after")}}


def _crest(side: dict):
    """The report's crest_db; sample peak minus the 1 s loudest-section RMS for reports
    made before it (the same reading, meter.crest_db)."""
    if side.get("crest_db") is not None:
        return side["crest_db"]
    if side.get("sample_peak_dbfs") is None or side.get("rms_1s_loud_dbfs") is None:
        return None
    return side["sample_peak_dbfs"] - side["rms_1s_loud_dbfs"]


def _flags(rep: dict) -> list[dict]:
    if rep.get("flags_structured") is not None:
        return rep["flags_structured"]
    rep = {**rep, "after": {**rep["after"], "crest_db": _crest(rep["after"])}}
    return evaluate._clean(evaluate._checks(rep))


def _run(work: Path) -> dict | None:
    """run.json, the completion marker; None when absent or unreadable."""
    try:
        run = json.loads((work / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return run if isinstance(run, dict) else None


def _controls(work: Path):
    run = _run(work)
    return run.get("controls") if run else None


def work_dir(path) -> Path:
    """`<name> - eluate/` beside the input; a stem set's sum (cli.STEMS_SUM) is already in
    its set's work dir, which is then its own (the shell's work_dir has the same rule)."""
    path = Path(path)
    if path.name == "stems-sum.wav" and path.parent.name.endswith(" - eluate"):
        return path.parent
    return path.parent / f"{path.stem} - eluate"


def summary(path) -> dict:
    work = work_dir(path)
    row = {"kind": "summary", "track": Path(path).stem, "input": str(path), "work": str(work.resolve())}
    run = _run(work)
    if run is None or not (work / "report.json").exists():
        return {**row, "state": "none"}
    rep = json.loads((work / "report.json").read_text(encoding="utf-8"))
    src = json.loads((work / "source.json").read_text(encoding="utf-8"))
    return {**row, "state": "done",
            "loudness": _r(rep["after"]["rms_meter_dbfs"]),
            "tilt": _r(rep["tilt"]["after"]["stereo"]),
            "crest": _r(_crest(rep["after"])), "crestBefore": _r(_crest(rep["before"])),
            "truePeak": _r(rep["after"]["true_peak_dbtp"]),
            "flags": _flags(rep), "controls": run.get("controls"), "bpm": src.get("bpm")}


def preview(path, target_rms: float) -> dict:
    x, fs = _load(path)
    meter = _meter_series(x, fs, COLUMNS)
    shift = target_rms - _meter_p95(meter)
    g = 10 ** (shift / 20)
    return {
        "kind": "preview",
        "track": Path(path).stem,
        "input": str(path),
        "duration": _r(len(x) / fs),
        "fs": fs,
        "matchDb": _r(shift),
        "env": {"before": _db_list(20 * np.log10(_columns_peak(x, COLUMNS) * g + 1e-9))},
        "meter": {"before": _db_list(meter + shift)},
    }


def _premaster(src: dict, premaster) -> Path:
    if premaster is None:
        return Path(src["path"])
    premaster = Path(premaster)
    if src.get("path") and Path(src["path"]).resolve() != premaster.resolve():
        raise ValueError(f"{premaster.name}: the render was made from another file "
                         f"({Path(src['path']).name}); master it again")
    return premaster


def from_render(work, premaster=None) -> dict:
    work = Path(work)
    rep = json.loads((work / "report.json").read_text(encoding="utf-8"))
    src = json.loads((work / "source.json").read_text(encoding="utf-8"))
    path = _premaster(src, premaster)
    pre, fs = _load(path)
    mst, fs2 = _load(work / "master.wav")
    if fs != fs2:
        raise ValueError("master and source sample rates differ")
    n = min(len(pre), len(mst))
    pre, mst = pre[:n], mst[:n]
    dur = n / fs

    match_db = rep["after"]["rms_meter_dbfs"] - rep["before"]["rms_meter_dbfs"]
    g = 10 ** (match_db / 20)
    tc = (np.arange(COLUMNS) + 0.5) * dur / COLUMNS
    gr = np.zeros(COLUMNS)
    if (work / "limiter_gr.npz").exists():
        lim = np.load(work / "limiter_gr.npz")
        gr = -np.interp(tc, lim["t"], lim["gr_db"])

    secs = rep["sections"]
    measured = [s for s in secs if s["before_lufs"] is not None and s["after_lufs"] is not None]
    ref = max(measured, key=lambda s: s["before_lufs"]) if measured else None
    out_secs = []
    for s in secs:
        live = s in measured and s["before_lufs"] > ref["before_lufs"] - evaluate.FLAG_SECTION_IGNORE_LU
        change = ((s["after_lufs"] - ref["after_lufs"]) - (s["before_lufs"] - ref["before_lufs"])
                  if live else None)
        out_secs.append({"start": _r(s["start"]), "end": _r(s["end"]), "label": s["label"],
                         "change": _r(change) if live else None,
                         "flag": bool(live and change > evaluate.FLAG_SECTION_LU),
                         "ref": s is ref})

    lim_w = next((w for w in rep.get("warnings") or [] if w.get("stage") == "limiter"), {})
    secs_eq = (rep["chain"].get("inputeq") or {}).get("sections") or {}
    tg = rep["targets"]
    ab = {k: str((work / "ab" / f).resolve()) for k, f in
          (("before", "original_matched.wav"), ("after", "master_matched.wav"))
          if (work / "ab" / f).exists()}
    return {
        "kind": "render",
        "track": path.stem,
        "input": src.get("path") or str(path),
        "work": str(work.resolve()),
        "master": str((work / "master.wav").resolve()),
        "log": str((work / "process-log.html").resolve()) if (work / "process-log.html").exists() else None,
        "ab": ab,
        "duration": _r(dur),
        "fs": fs,
        "bpm": src.get("bpm"),
        "matchDb": _r(match_db),
        "env": {"before": _db_list(20 * np.log10(_columns_peak(pre, COLUMNS) * g + 1e-9)),
                "after": _db_list(20 * np.log10(_columns_peak(mst, COLUMNS) + 1e-9))},
        "meter": {"before": _db_list(_meter_series(pre, fs, COLUMNS) + match_db),
                  "after": _db_list(_meter_series(mst, fs, COLUMNS))},
        "limiterGr": _db_list(gr),
        "sections": out_secs,
        "ltas": {"hz": rep["ltas"]["centres_hz"], "before": rep["ltas"]["before"],
                 "after": rep["ltas"]["after"]},
        "ltasFine": _fine(rep["ltas"].get("fine")),
        "tilt": {"before": rep["tilt"]["before"], "after": rep["tilt"]["after"],
                 "target": tg["slope_db_oct"], "tol": tg["slope_tol"]},
        "loudness": {"target": tg["rms_dbfs"], "tol": tg["rms_tol_db"],
                     "reached": _r(rep["after"]["rms_meter_dbfs"]),
                     "heldBy": lim_w.get("bound")},
        "crest": {"before": _r(_crest(rep["before"])), "after": _r(_crest(rep["after"]))},
        "lra": {"before": _r(rep["before"]["lra_lu"]), "after": _r(rep["after"]["lra_lu"])},
        "truePeak": _r(rep["after"]["true_peak_dbtp"]),
        "noise": ({k: (_r(v) if isinstance(v, float) else v) for k, v in rep["noise"].items()
                   if k in ("before_db", "after_db", "threshold_db", "status")}
                  if rep.get("noise") else None),
        "crossovers": [_r(f) for f in src["crossovers_hz"]],
        "sideHp": (rep["chain"].get("sidestrip") or {}).get("fc_hz"),
        "shelves": {"M": secs_eq.get("high_shelf"), "S": secs_eq.get("high_shelf_side"),
                    "airM": secs_eq.get("air_shelf"), "airS": secs_eq.get("air_shelf_side"),
                    "max": {"M": INPUTEQ_HS_MID_MAX_DB, "S": INPUTEQ_HS_SIDE_MAX_DB,
                            "airM": INPUTEQ_HS_MID_MAX_DB, "airS": INPUTEQ_HS_SIDE_MAX_DB}},
        "controls": _controls(work),
        "flags": _flags(rep),
    }


if __name__ == "__main__":
    import sys
    from eluate import io
    io.write_text(Path(sys.argv[2]), "window.DRAFT = "
                  + json.dumps(from_render(sys.argv[1]), separators=(",", ":")) + ";\n")
