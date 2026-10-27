"""Command line: eluate analyse | plan | render | master | report (spec 4).

Contract:
- `eluate analyse in.wav -o work/ [--xover 100,400,1600,6400] [--bpm B]`
  runs planner.analyse (input normalisation to planner.R_REF_DBFS, then
  features.analyse on the normalised audio with the given crossovers, or
  key-locked ones without --xover; bpm overrides tempo detection, spec 3.2a)
  and writes work/features.npz (the full analysis dict: its nesting as JSON in
  the "__tree__" entry, every array as its own entry; loaded with
  allow_pickle=False, so a work dir from elsewhere cannot run code, and written
  under a temporary name then renamed, source.json removed first and written
  last, so a stopped run never leaves a cache that matches the wrong analysis),
  work/analysis.json (arrays summarised as shape / min / max / mean) and
  work/source.json (input path, size, mtime, the audio file's SHA-256, fs,
  length, the crossovers used, the crossovers requested (null = key-locked),
  bpm, R_ref, the input gain and ANALYSIS_VERSION). The audio is read in place
  and never copied.
- `eluate plan work/ [--tilt T] [--target-rms R] [--preserve]
  [--xover ...] [--subside F]` writes work/plan.json +
  work/plan.npz (realised on the source audio), work/targets.json and
  work/plan_record.json (the planner's decisions, including every notch).
- `eluate render work/plan.json -o master.wav [--input in.wav]` renders a
  saved plan (stages + blocked limiter, solver.render_plan); the input
  defaults to the source.json next to the plan.
- `eluate master in.wav -o out/ [controls] [--master-ref theirs.wav]
  [--max-renders N] [--bpm B]`: analyse (reusing out/features.npz when
  source.json matches the audio's SHA-256, the requested crossovers (a
  key-locked analysis is reused for a request without --xover, whatever
  crossovers the key gave), bpm, R_ref and ANALYSIS_VERSION, and it loads; on reuse source.json
  takes the input's current path), plan, solve, write
  out/master.wav (24-bit PCM, or 16-bit with TPDF dither under --streaming:
  plan.output, the samples the solver measured), out/plan.json, out/solve.json, the report
  with A/B, and out/run.json (runtime per phase, peak memory, solver
  renders, SHA-256 of every eluate module at the start of the run, and "memory":
  eluate.memory's estimate and the free memory), and
  out/limiter.json / limiter_gr.npz, and out/process-log.md / .html (processlog: the
  render's decisions in plain language). solve.json and run.json carry the
  solver's target misses and budget warnings (the loudness shortfall included).
  run.json is the completion marker (report 3.1): the previous run's run.json and
  process log are removed before anything else, run.json is written after the master
  and the report, and it carries the run id (uuid4 hex) that plan.json / plan.npz
  carry too; a run without a limiter GR estimate removes the previous limiter_gr.npz.
- `eluate master --stems a.wav b.wav ... [--mix mix.wav] -o out/ [controls]` (stems,
  phase A, sprint 2026-10-21 WP6): reads the stems (stems.read_stems: one rate and
  length, a silent render tail cut), and with --mix runs the null test (stems.null_test),
  stopping with the figure unless it passes; writes their sum to out/stems-sum.wav
  (sum_stems, which the app's stems-sum runs alone) and masters it as above under the
  stems profile
  (planner.PROFILES["stems"], tighter ranges) and attributes its limits (stems.attribute:
  the stems behind every limit the sum hit). Phase B's loop (do_master_stems' docstring)
  then corrects the original stems (stemfix) for at most STEM_PASSES_MAX more passes, each
  mastered in out/pass-<k>/ from its corrected sum, keeps the best pass by score (its
  outputs copied to out/), and writes out/stems-report.md / .json (the kept pass's limits
  and every pass's score, limits and fixes) and run.json last. The stem files are never
  changed or written.
- `eluate report out/` rebuilds the report from out/ (source.json,
  master.wav, plan.json, targets.json, plan_record.json, solve.json, and the
  limiter.json / limiter_gr.npz that `master` writes, so the limiter data and
  GR plot survive), and the process log with it.
- Controls flags (spec 3.6 draft 0.5): --crest-mode loss|report|floor,
  --crest-floor a,b,c,d,e (floor mode), --side-band0-exempt, --tp-meter
  worst|8x; --streaming (docs/gain-staging.md 3: -1 dBTP ceiling, 16-bit
  output with TPDF dither, instead of -0.3 dBTP at 24 bits); the loudness
  target --target-rms R (the master's RMS on the maintainer's Ozone meter: 300 ms,
  AES17, loudest passage; default planner.DEFAULT_TARGET_RMS_DBFS = -4.1);
  dynamics: --character
  auto|transparent|slow|moderate|fast|aggressive, --makeup peak|rms|auto,
  --glue-mode serial|parallel, --kick-detective / --no-kick-detective (default
  auto: on for the slow and transparent characters), --speclim / --no-speclim
  (frequency-dependent limiting before the clipper, default on), --profile
  electronic|live (planner.PROFILES; live is phase 1, uncalibrated); --threads N (engine
  threads, eluate.parallel; default all cores but two, recorded in run.json's controls;
  the master and plan are bit-identical at any N). Once the completion marker is cleared,
  `master` runs eluate.memory.guard on the input's frame count (soundfile.info, nothing
  loaded): one thread when the threaded estimate is over 85% of the free memory (run.json's
  controls record the threads used), a ValueError when the one-thread estimate is over
  all of it. plan_record.json gains
  the per-stage peak GR against budget and the warnings (budget, warnings).
  `do_master(..., progress=f)` calls f(phase, **info) at each phase's start, for the
  desktop app's progress line (nothing in the run depends on it): "analyse" (frames,
  threads: after the memory guard; reused: whether the cached analysis is used), "plan",
  "solve", "render" (render: the solver's render about to start, of up to renders_max =
  solver.render_bound) and "report".
- `render` writes the plan's output format too (plan.output.subtype).
- Every text file is written UTF-8 with LF (io.write_text / write_json); the audio,
  JSON and npz files are written atomically (io.atomic_write; the report's plots and the
  process log are not).
- Exit status 0 on success; a ValueError, an OS error (missing file) or an
  unreadable audio file prints one line and exits 2.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from eluate import evaluate, io, memory, parallel, planner, processlog, solver, stems
from eluate.plan import MAKEUP_MODES
from eluate.plan import Plan


def _xover(s: str) -> tuple:
    return tuple(float(v) for v in s.split(","))


def _summarise(obj):
    if isinstance(obj, dict):
        return {str(k): _summarise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if len(obj) > 16 and all(isinstance(v, (int, float, np.floating, np.integer)) for v in obj):
            a = np.asarray(obj, dtype=float)
            return _summarise(a)
        return [_summarise(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.size > 16:
            a = obj.astype(float)
            fin = a[np.isfinite(a)]
            return {"shape": list(obj.shape),
                    "min": float(fin.min()) if fin.size else None,
                    "max": float(fin.max()) if fin.size else None,
                    "mean": float(fin.mean()) if fin.size else None}
        return evaluate._clean(obj)
    return evaluate._clean(obj)


def file_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _requested(crossovers) -> list | None:
    return None if crossovers is None else [float(f) for f in crossovers]


# The analysis code's version, written into source.json with every cached analysis; a
# cache of another version (or none) is never reused (audit 13, Sol). Bump it whenever a
# change to planner.analyse or eluate.features alters what the analysis holds (a value,
# a field added or removed), so every cached features.npz is analysed again once.
# 1: first versioned cache (S2 tempo octave decision, noise_floor_db).
ANALYSIS_VERSION = 1


def _source_record(path: Path, fs: int, n: int, analysis: dict, crossovers=None,
                   bpm=None) -> dict:
    """What the cached analysis was made from: path, size, mtime, the audio file's
    SHA-256, fs, length, the crossovers used and requested, bpm, R_ref, the input
    gain (audit 07 A17, F15) and ANALYSIS_VERSION (audit 13)."""
    st = path.stat()
    meta = analysis["meta"]
    inp = meta.get("input") or {}
    return {"analysis_version": ANALYSIS_VERSION,
            "path": str(path.resolve()), "size": st.st_size, "mtime": st.st_mtime,
            "sha256": file_sha256(path), "fs": int(fs), "n": int(n),
            "crossovers_hz": [float(f) for f in meta["crossovers"]],
            "crossovers_requested": _requested(crossovers),
            "crossover_rule": meta.get("crossover_rule"), "bpm": bpm,
            "r_ref_dbfs": inp.get("r_ref_dbfs"), "input_gain_db": inp.get("gain_db")}


def do_analyse(inp: Path, work: Path, xover=None, log=print, bpm: float | None = None):
    work.mkdir(parents=True, exist_ok=True)
    x, fs = io.read(inp)
    (work / "source.json").unlink(missing_ok=True)
    (work / "features.pkl").unlink(missing_ok=True)      # the old pickled cache, never read
    t = time.perf_counter()
    a = planner.analyse(x, fs, xover, bpm)
    dt = time.perf_counter() - t
    _save_analysis(work, a)
    io.write_json(work / "analysis.json", _summarise(a))
    io.write_json(work / "source.json", _source_record(inp, fs, x.shape[0], a, xover, bpm))
    log(f"analysed {inp.name} in {dt:.1f} s (input gain {a['meta']['input']['gain_db']:+.2f} dB, "
        f"crossovers {', '.join(f'{f:.1f}' for f in a['meta']['crossovers'])} Hz)")
    return a, x, fs, dt


ANALYSIS_FILE = "features.npz"
STEMS_SUM = "stems-sum.wav"         # a stem set's sum, in its work dir (uidata.work_dir keeps it there)
# removed at the start of `master`: run.json is the completion marker, written last
CLEARED_AT_START = ("run.json", "process-log.md", "process-log.html")
_TREE, _ND, _TUPLE = "__tree__", "__nd__", "__tuple__"


def _pack(v, arrays: dict):
    if isinstance(v, dict):
        if not all(isinstance(k, str) and k not in (_ND, _TUPLE) for k in v):
            raise ValueError("analysis dict keys must be strings")
        return {k: _pack(w, arrays) for k, w in v.items()}
    if isinstance(v, tuple):
        return {_TUPLE: [_pack(w, arrays) for w in v]}
    if isinstance(v, list):
        return [_pack(w, arrays) for w in v]
    if isinstance(v, np.ndarray):
        if v.dtype.hasobject:
            raise ValueError("analysis arrays must not hold Python objects")
        key = f"a{len(arrays)}"
        arrays[key] = v
        return {_ND: key}
    if isinstance(v, np.generic):
        return v.item()
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    raise ValueError(f"analysis value of type {type(v).__name__} cannot be cached")


def _unpack(v, arrays: dict):
    if isinstance(v, dict):
        if _ND in v:
            return arrays[v[_ND]]
        if _TUPLE in v:
            return tuple(_unpack(w, arrays) for w in v[_TUPLE])
        return {k: _unpack(w, arrays) for k, w in v.items()}
    if isinstance(v, list):
        return [_unpack(w, arrays) for w in v]
    return v


def _save_analysis(work: Path, a: dict) -> None:
    arrays: dict = {}
    tree = json.dumps(_pack(a, arrays), separators=(",", ":"))
    arrays[_TREE] = np.frombuffer(tree.encode("utf-8"), dtype=np.uint8)
    io.write_npz(work / ANALYSIS_FILE, **arrays)


def _load_analysis(work: Path) -> dict:
    p = work / ANALYSIS_FILE
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: run analyse first")
    try:
        with np.load(p, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        tree = json.loads(arrays.pop(_TREE).tobytes().decode("utf-8"))
        return _unpack(tree, arrays)
    except (OSError, ValueError, KeyError, TypeError, EOFError, RecursionError,
            zipfile.BadZipFile) as e:
        raise ValueError(f"{p} is unreadable ({type(e).__name__}): run analyse again") from e


def _controls(args) -> planner.Controls:
    return planner.Controls(tilt_db=args.tilt, target_rms=args.target_rms,
                            preserve_mix=args.preserve,
                            crossovers_hz=args.xover,
                            subside_hz=args.subside,
                            crest_mode=args.crest_mode, crest_floors_db=args.crest_floor,
                            side_band0_exempt=args.side_band0_exempt, tp_meter=args.tp_meter,
                            streaming=args.streaming, character=args.character,
                            bpm=getattr(args, "bpm", None), makeup=args.makeup,
                            glue_mode=args.glue_mode, kick_detective=args.kick_detective,
                            speclim=args.speclim, profile=args.profile, threads=args.threads)


def do_plan(work: Path, controls: planner.Controls, x=None, log=print):
    parallel.set_threads(controls.threads)
    a = _load_analysis(work)
    src = json.loads((work / "source.json").read_text(encoding="utf-8"))
    if x is None:
        x, fs = io.read(src["path"])
    plan = planner.plan_from_analysis(a, controls, x)
    plan.save(work / "plan.json")
    targets = planner.targets_from(a, controls)
    io.write_json(work / "targets.json", targets.to_dict())
    rec = planner.plan_record(a, controls)
    rec["budget"] = planner.budget_report(plan)
    rec["warnings"] = rec["budget"]["warnings"]
    io.write_json(work / "plan_record.json", evaluate._clean(rec))
    log(f"plan written to {work / 'plan.json'}")
    return plan, targets, a


def _targets_load(path: Path) -> planner.Targets:
    # a key since dropped (rms_accept_floor_dbfs, S4) is ignored, so older work dirs load
    names = {f.name for f in dataclasses.fields(planner.Targets)}
    d = json.loads(path.read_text(encoding="utf-8"))
    return planner.Targets(**{k: v for k, v in d.items() if k in names})


def do_render(plan_path: Path, out: Path, inp: Path | None, log=print):
    plan = Plan.load(plan_path)
    if inp is None:
        src = json.loads((plan_path.parent / "source.json").read_text(encoding="utf-8"))
        inp = Path(src["path"])
    x, fs = io.read(inp)
    y, info, _ = solver.render_plan(x, fs, plan)
    io.write(out, y, fs, _subtype(plan))
    log(f"rendered {out} (true peak 8x {info.get('true_peak_8x_dbtp', float('nan')):.2f}, "
        f"32x {info.get('true_peak_32x_dbtp', float('nan')):.2f} dBTP)")
    return y, info


def _subtype(plan: Plan) -> str:
    return "FLOAT" if plan.output is None else plan.output.subtype


def _matches(src_file: Path, inp: Path, bpm: float | None = None, crossovers=None) -> bool:
    """Reuse the cached analysis only for the same audio content (SHA-256), the same
    requested crossovers (None = key-locked: the crossovers the key gave are a function
    of the audio, so they need no check of their own), bpm, R_ref and the analysis code's
    version. A source.json from before input normalisation (no r_ref_dbfs) or of another
    ANALYSIS_VERSION (or none) never matches."""
    if not src_file.exists():
        return False
    try:
        s = json.loads(src_file.read_text(encoding="utf-8"))
        if s.get("analysis_version") != ANALYSIS_VERSION:
            return False
        if ("crossovers_requested" not in s or s["crossovers_requested"] != _requested(crossovers)
                or s.get("r_ref_dbfs") != planner.R_REF_DBFS):
            return False
        if s.get("bpm") != bpm or s.get("size") != inp.stat().st_size:
            return False
        return s.get("sha256") == file_sha256(inp)
    except (KeyError, ValueError, OSError):
        return False


def _reuse_analysis(out: Path, inp: Path, bpm=None, crossovers=None, log=print) -> dict | None:
    """The cached analysis in out/ when it matches inp and loads (None: analyse again);
    source.json then takes inp's current path, so a moved track reads from where it is."""
    if not ((out / ANALYSIS_FILE).exists() and _matches(out / "source.json", inp, bpm, crossovers)):
        return None
    try:
        a = _load_analysis(out)
    except ValueError as e:
        log(f"analysis not reused: {e}")
        return None
    src = json.loads((out / "source.json").read_text(encoding="utf-8"))
    src["path"], src["mtime"] = str(inp.resolve()), inp.stat().st_mtime
    io.write_json(out / "source.json", src)
    log("analysis reused")
    return a


def source_hashes() -> dict:
    """SHA-256 of every eluate module, so a run records the code it ran with."""
    import hashlib
    root = Path(__file__).resolve().parent
    return {str(p.relative_to(root.parent)).replace("\\", "/"):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*.py"))}


def do_master(inp: Path, out: Path, controls: planner.Controls, master_ref: Path | None = None,
              max_renders: int = solver.MAX_OUTER, log=print, bpm: float | None = None,
              progress=None) -> dict:
    t0 = time.perf_counter()
    step = progress or (lambda *a, **k: None)
    # the completion marker goes first, so a failed or stopped run is never read as done
    # with the previous run's files (report 3.1); its process log goes with it
    for name in CLEARED_AT_START:
        (out / name).unlink(missing_ok=True)
    run_id = uuid.uuid4().hex
    # then the memory guard, before any audio is read: one thread, or a refusal
    frames = sf.info(str(inp)).frames
    threads, mem = memory.guard(frames, controls.threads, log)
    controls = dataclasses.replace(controls, threads=threads)
    parallel.set_threads(controls.threads)
    code = source_hashes()
    out.mkdir(parents=True, exist_ok=True)
    phases = {}
    a = _reuse_analysis(out, inp, bpm, controls.crossovers_hz, log)
    step("analyse", frames=frames, threads=threads, reused=a is not None)
    if a is not None:
        x, fs = io.read(inp)
        phases["analyse_s"] = 0.0
    else:
        a, x, fs, phases["analyse_s"] = do_analyse(inp, out, controls.crossovers_hz, log, bpm)
    step("plan")
    t = time.perf_counter()
    plan0 = planner.plan_from_analysis(a, controls)
    targets = planner.targets_from(a, controls)
    rec = planner.plan_record(a, controls)
    io.write_json(out / "targets.json", targets.to_dict())
    io.write_json(out / "plan_record.json", evaluate._clean(rec))
    phases["plan_s"] = time.perf_counter() - t
    step("solve")
    bound = solver.render_bound(max_renders)
    t = time.perf_counter()
    res = solver.solve(x, fs, plan0, targets, max_outer=max_renders, log=log,
                       progress=lambda n: step("render", render=n, renders_max=bound))
    phases["solve_s"] = time.perf_counter() - t
    # per-stage peak GR against budget and the warnings, from the solved plan (chunk 3)
    rec["budget"] = planner.budget_report(res.plan, res.lim_info)
    rec["warnings"] = rec["budget"]["warnings"]
    io.write_json(out / "plan_record.json", evaluate._clean(rec))
    res.plan.save(out / "plan.json", run_id)
    io.write(out / "master.wav", res.master, fs, _subtype(res.plan))
    summ = evaluate._clean(res.summary())
    io.write_json(out / "solve.json", summ)
    step("report")
    t = time.perf_counter()
    mref = None
    if master_ref is not None:
        mx, mfs = io.read(master_ref)
        mref = (mx, mfs)
    lim_gr = None
    if res.pre_limiter is not None and res.plan.limiter is not None:
        lim_gr = evaluate.limiter_gr_estimate(
            res.pre_limiter * 10.0 ** (res.plan.limiter.input_gain_db / 20.0), res.master, fs)
    res.pre_limiter = None             # only the GR estimate needs it (spec 1 memory)
    # kept for `eluate report` (audit 07 A16)
    io.write_json(out / "limiter.json", evaluate._clean(res.lim_info))
    if lim_gr is not None:
        io.write_npz(out / "limiter_gr.npz", t=lim_gr[0], gr_db=lim_gr[1])
    else:
        (out / "limiter_gr.npz").unlink(missing_ok=True)     # not the previous run's
    evaluate.report(x, res.master, fs, out, plan=res.plan, targets=targets, plan_rec=rec,
                    solve_summary=summ, lim_info=res.lim_info, limiter_gr=lim_gr,
                    master_ref=mref)
    phases["report_s"] = time.perf_counter() - t
    run = {"run_id": run_id, "input": str(inp), "fs": fs, "duration_s": x.shape[0] / fs,
           "controls": controls.to_dict(), "phases": phases,
           "runtime_s": time.perf_counter() - t0, "peak_memory_mb": solver.peak_memory_mb(),
           "memory": mem,
           "renders": res.renders, "ok": res.ok,
           "misses": [c["name"] for c in res.misses],
           "warnings": [w.get("stage") for w in res.warnings],
           "rms_dbfs": res.metrics.get("rms_dbfs"), "s95_lufs": res.metrics.get("s95_lufs"),
           "constraints_met": {c["name"]: c["ok"] for c in res.metrics["constraints"]},
           "source_sha256": code}
    io.write_json(out / "run.json", evaluate._clean(run))
    _process_log(out, log)
    log(f"master done in {run['runtime_s']:.0f} s, peak memory {run['peak_memory_mb']:.0f} MB")
    return run


def do_report(out: Path, master_ref: Path | None = None, log=print,
              process_log: bool = True) -> dict:
    src = json.loads((out / "source.json").read_text(encoding="utf-8"))
    x, fs = io.read(src["path"])
    y, fy = io.read(out / "master.wav")
    if fy != fs:
        raise ValueError("master and source sample rates differ")
    plan = Plan.load(out / "plan.json") if (out / "plan.json").exists() else None
    targets = _targets_load(out / "targets.json") if (out / "targets.json").exists() else None
    rec = json.loads((out / "plan_record.json").read_text(encoding="utf-8")) \
        if (out / "plan_record.json").exists() else None
    summ = json.loads((out / "solve.json").read_text(encoding="utf-8")) \
        if (out / "solve.json").exists() else None
    mref = None
    if master_ref is not None:
        mx, mfs = io.read(master_ref)
        mref = (mx, mfs)
    lim_info = json.loads((out / "limiter.json").read_text(encoding="utf-8")) \
        if (out / "limiter.json").exists() else None
    lim_gr = None
    if (out / "limiter_gr.npz").exists():
        with np.load(out / "limiter_gr.npz") as zf:
            lim_gr = (zf["t"], zf["gr_db"])
    rep = evaluate.report(x, y, fs, out, plan=plan, targets=targets, plan_rec=rec,
                          solve_summary=summ, lim_info=lim_info, limiter_gr=lim_gr,
                          master_ref=mref)
    log(f"report written to {out / 'report.md'}")
    if process_log:
        _process_log(out, log)
    return rep


def _process_log(out: Path, log) -> None:
    """process-log.md / .html; a failure here is reported, not raised (the master stands)."""
    try:
        log(f"process log written to {processlog.write(out)}")
    except Exception as e:                       # noqa: BLE001
        log(f"process log not written: {type(e).__name__}: {e}")


def _add_controls(p):
    p.add_argument("--tilt", type=float, default=0.0, help="dB/oct around -4.5, +-3")
    p.add_argument("--target-rms", type=float, default=None,
                   help="master RMS on the Ozone meter: 300 ms, AES17, loudest passage (default "
                        f"{planner.DEFAULT_TARGET_RMS_DBFS:g})")
    p.add_argument("--preserve", action="store_true")
    p.add_argument("--xover", type=_xover, default=None)
    p.add_argument("--subside", type=float, default=None)
    p.add_argument("--crest-mode", default="report", choices=planner.CREST_MODES)
    p.add_argument("--profile", default="electronic", choices=planner.USER_PROFILES,
                   help="live: a live or acoustic recording (phase 1, uncalibrated)")
    p.add_argument("--crest-floor", type=_xover, default=None,
                   help="per-band crest floors, dB, for --crest-mode floor")
    p.add_argument("--side-band0-exempt", action="store_true")
    p.add_argument("--tp-meter", default=None, choices=planner.TP_METERS,
                   help="default 8x at 24 bits, worst (with BS.1770 Annex 2) under --streaming")
    p.add_argument("--streaming", action="store_true",
                   help="-1 dBTP ceiling, 16-bit output with TPDF dither")
    p.add_argument("--character", default="auto", choices=planner.CHARACTERS)
    p.add_argument("--makeup", default="peak", choices=MAKEUP_MODES,
                   help="multiband makeup: +peak GR, +mean GR, or following its 3 s average")
    p.add_argument("--glue-mode", default="serial", choices=planner.GLUE_MODES)
    p.add_argument("--kick-detective", action=argparse.BooleanOptionalAction, default=None,
                   help="clip only around kick onsets (default auto: on for the slow and "
                        "transparent characters)")
    p.add_argument("--speclim", action=argparse.BooleanOptionalAction, default=True,
                   help="frequency-dependent limiting before the clipper (default on)")
    p.add_argument("--threads", type=int, default=parallel.DEFAULT_THREADS,
                   help=f"engine threads (default {parallel.DEFAULT_THREADS}: all cores but two); "
                        "the master does not depend on it")


def sum_stems(paths, out: Path, mix: Path | None = None, log=print):
    """Reads the stems (stems.read_stems), runs the null test against the mix when there
    is one (a verdict other than pass stops with its message), and writes their sum to
    out/STEMS_SUM (32-bit float). Returns (stems, fs, null test or None, the sum's path).
    The same sum bytes every time, so an analysis cached for it stays valid."""
    st, fs = stems.read_stems(paths, log=log)
    total = st.sum(axis=0)
    nt = None
    if mix is not None:
        m, mfs = io.read(mix)
        if mfs != fs:
            raise ValueError(f"the mix is {mfs} Hz, the stems {fs} Hz")
        nt = stems.null_test(m, total, fs)
        log(stems.null_message(nt))
        if nt["verdict"] != "pass":
            raise ValueError(stems.null_message(nt))
    out.mkdir(parents=True, exist_ok=True)
    sum_path = out / STEMS_SUM
    io.write(sum_path, total, fs, "FLOAT")
    return st, fs, nt, sum_path


STEM_PASSES_MAX = 2       # correction passes after the sum pass (maintainer, 2026-10-27)
STEM_IMPROVE = 0.25       # a pass improves when its score drops at least this much
STEM_PASS_DIR = "pass-{}"   # a correction pass's work dir, inside the set's work dir
CORRECTED_SUM = "corrected-sum.wav"   # a correction pass's sum, in its pass dir
PASSES_DIR = "passes"     # every pass's master under a numbered name (maintainer: dragged into a
                          # DAW they are all "master"); hard links where the volume allows
# a winning correction pass's files that do not go to the work dir root: the root keeps
# its analysis cache (the plain sum's, which the app's background analysis made and the
# next Render reuses), the A/B is rebuilt there, and run.json is written last
_NOT_PROMOTED = ("features.npz", "analysis.json", "source.json", CORRECTED_SUM, "run.json",
                 "process-log.md", "process-log.html")
_STOPPED = {
    "no_limit": "The loop stopped: pass {last} hit no limit.",
    "nothing_to_fix": "The loop stopped: no further stem correction for what pass {last} hit.",
    "no_improvement": "The loop stopped: pass {last} did not improve on pass {prev} by at least "
                      f"{STEM_IMPROVE:g}.",
    "max_passes": f"The loop stopped after its {STEM_PASSES_MAX} correction passes.",
    "pass_failed": "The loop stopped: correction pass {fail} failed (the process log has why).",
}


def _stemfix():
    """eluate.stemfix, imported on first use (tests replace this)."""
    from eluate import stemfix
    return stemfix


def _detected_bpm(work: Path) -> float | None:
    """The tempo the sum pass's analysis found (analysis.json), so the correction passes
    and stemfix work on the same beat grid; None when it is not there."""
    try:
        a = json.loads((work / "analysis.json").read_text(encoding="utf-8"))
        v = (a.get("structure") or {}).get("tempo_bpm")
        return float(v) if isinstance(v, (int, float)) and v > 0 else None
    except (OSError, ValueError, AttributeError):
        return None


def _score_pass(index: int, work: Path, att: dict, fixes: list, lines: list) -> dict:
    rep = json.loads((work / "report.json").read_text(encoding="utf-8"))
    sc = stems.pass_score(rep, att["attribution"])
    return {"index": index, "dir": "." if index == 0 else work.name, **sc,
            "fixes": list(fixes), "fix_lines": list(lines)}


def _name_passes(out: Path, names, passes: list, best: int) -> None:
    """out/passes/<set> <n> - <what>.wav for pass n-1, numbered from 1 in render order,
    "(kept)" on the kept one; run before _promote, which writes the root's files fresh so
    pass 0's link keeps pass 0's master."""
    d = out / PASSES_DIR
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    prefix = os.path.commonprefix([str(n) for n in names]).strip(" -_") or "stems"
    for p in passes:
        i = p["index"]
        src = out / "master.wav" if i == 0 else out / STEM_PASS_DIR.format(i) / "master.wav"
        what = "plain sum" if i == 0 else f"pass {i}"
        dst = d / f"{prefix} {i + 1} - {what}{' (kept)' if i == best else ''}.wav"
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def _promote(src: Path, out: Path, log) -> None:
    """A correction pass won: its outputs go to the work dir root, where the app and the
    smoke test read them. The report is rebuilt there against the plain sum (the root's
    source.json), so its before, the A/B and the graphs compare the master with what the
    stems gave, corrections included."""
    (out / "run.json").unlink(missing_ok=True)      # the completion marker first
    for f in src.iterdir():
        if f.is_file() and f.name not in _NOT_PROMOTED:
            (out / f.name).unlink(missing_ok=True)      # a new file: passes/ links the old
            shutil.copy2(f, out / f.name)
    if not (src / "limiter_gr.npz").exists():
        (out / "limiter_gr.npz").unlink(missing_ok=True)
    do_report(out, None, log, process_log=False)


def do_master_stems(paths, out: Path, controls: planner.Controls, mix: Path | None = None,
                    max_renders: int = solver.MAX_OUTER, log=print,
                    bpm: float | None = None, progress=None) -> dict:
    """The stems flow (module contract) with stems phase B's loop (docs/noise-and-stems.md
    2, maintainer 2026-10-27).

    Pass 0 sums the stems (null test included), masters the sum in out/ under the stems
    profile and attributes its limits. Then at most STEM_PASSES_MAX correction passes:
    pass k asks stemfix.plan for the cumulative fixes (from the last pass's attribution,
    on the original stems), applies them to the original stems, writes their sum to
    out/pass-k/corrected-sum.wav, masters it in out/pass-k/ (its own analysis cache, the
    tempo pass 0 found unless --bpm gave one) and attributes the corrected stems. The
    corrected stems themselves are not written.

    Score (stems.pass_score, lower is better): the RMS shortfall to the profile target in
    dB (report.json loudness: target_rms_dbfs - after_rms_dbfs, floored at 0) plus
    stems.LIMIT_WEIGHT (0.5) per real limit (stems.is_limit: EQ bounds, multiband, the
    frequency-dependent limiter, clipper / limiter caps, notches found; not the crest /
    section row, noise or a rendered notch).

    The loop stops when the last pass hit no real limit, when stemfix has nothing more
    (empty, or the fixes it already had), when a pass's score does not drop by at least
    STEM_IMPROVE on the pass before it, or after STEM_PASSES_MAX passes. The pass kept has
    the lowest score; a later pass replaces an earlier one only when it is at least
    STEM_IMPROVE lower (a smaller gain counts as a tie, and ties go to the less processed
    pass). A kept correction pass's outputs are copied to out/ (_promote), so out/ always
    holds the final master, report, plots and A/B; the pass dirs keep their own master
    and report (not their A/B); out/passes/ names every pass's master by its number
    (_name_passes). stems-report.md / .json describe the kept pass's limits
    and every pass (rep["passes"], "kept", "stopped"); run.json is the kept pass's with
    "stems": {kept, passes, stopped, runtime_s}, written last.

    progress gets ("pass", index=k, passes_max=1 + STEM_PASSES_MAX) before each pass
    (before do_master's own phases) and ("keep", index=k) before a kept correction pass's
    report is rebuilt in out/.
    """
    step = progress or (lambda *a, **k: None)
    t0 = time.perf_counter()
    controls = dataclasses.replace(controls, profile="stems")
    names = [Path(p).stem for p in paths]
    total = 1 + STEM_PASSES_MAX
    st, fs, nt, sum_path = sum_stems(paths, out, mix, log)
    step("pass", index=0, passes_max=total)
    runs = [do_master(sum_path, out, controls, max_renders=max_renders, log=log, bpm=bpm,
                      progress=progress)]
    (out / "run.json").unlink(missing_ok=True)     # done only once the loop is (written last)
    atts = [stems.attribute(out, st, names, fs)]
    passes = [_score_pass(0, out, atts[0], [], [])]
    log(f"stems pass 0: score {passes[0]['score']:.2f}, {len(passes[0]['limits'])} limit(s)")
    best, fixes, why = 0, [], None
    pass_bpm = bpm if bpm is not None else _detected_bpm(out)
    for k in range(1, STEM_PASSES_MAX + 1):
        if not passes[-1]["limits"]:
            why = "no_limit"
            break
        try:
            fx = _stemfix()
            new = fx.plan(atts[-1], st, names, fs, prior=fixes or None, bpm=pass_bpm)
            if not new or list(new) == fixes:
                why = "nothing_to_fix"
                break
            fixes = list(new)
            lines = [fx.describe(f) for f in fixes]
            step("pass", index=k, passes_max=total)
            for line in lines:
                log(f"stems pass {k}: {line}")
            d = out / STEM_PASS_DIR.format(k)
            d.mkdir(parents=True, exist_ok=True)
            # the sum one stem at a time (stemfix.apply_sum): no corrected set is held through
            # the master (memory); the set is made once the master is done, for the attribution
            total_k = (fx.apply_sum(st, fs, names, fixes) if hasattr(fx, "apply_sum")
                       else fx.apply(st, fs, names, fixes).sum(axis=0))
            io.write(d / CORRECTED_SUM, total_k, fs, "FLOAT")
            del total_k
            runs.append(do_master(d / CORRECTED_SUM, d, controls, max_renders=max_renders, log=log,
                                  bpm=pass_bpm, progress=progress))
            corrected = fx.apply(st, fs, names, fixes)
            atts.append(stems.attribute(d, corrected, names, fs))
            del corrected
            shutil.rmtree(d / "ab", ignore_errors=True)   # the root's A/B is the one played
            passes.append(_score_pass(k, d, atts[-1], fixes, lines))
        except Exception as e:      # MemoryError included: pass 0 is still a whole master
            log(f"stems pass {k} failed ({type(e).__name__}: {e}); keeping the passes before it")
            why = "pass_failed"
            break
        p, prev = passes[-1], passes[-2]
        log(f"stems pass {k}: score {p['score']:.2f}, {len(p['limits'])} limit(s)")
        if p["score"] <= passes[best]["score"] - STEM_IMPROVE:
            best = k
        if p["score"] > prev["score"] - STEM_IMPROVE:
            why = "no_improvement"
            break
    if why is None:
        why = "no_limit" if not passes[-1]["limits"] else "max_passes"
    last = passes[-1]["index"]
    for j in range(last + 1, STEM_PASSES_MAX + 1):     # an earlier render's, not this one's
        shutil.rmtree(out / STEM_PASS_DIR.format(j), ignore_errors=True)
    low = min(p["score"] for p in passes)
    stopped = (_STOPPED[why].format(last=last, prev=max(0, last - 1), fail=last + 1)
               + f" Kept pass {best} (score {passes[best]['score']:.2f}"
               + ("" if len(passes) == 1 else ", the lowest" if passes[best]["score"] == low
                  else f": no later pass was at least {STEM_IMPROVE:g} lower") + ").")
    log(stopped)
    _name_passes(out, names, passes, best)
    if best:
        step("keep", index=best)
        _promote(out / STEM_PASS_DIR.format(best), out, log)
    run = dict(runs[best])
    run["stems"] = {"kept": best, "passes": len(passes), "stopped": why,
                    "runtime_s": time.perf_counter() - t0}
    rep = dict(atts[best])
    rep.update(null_test=nt, passes=passes, kept=best, stopped=stopped, stopped_why=why)
    stems.write_report(out, rep)
    log(f"stems report written to {out / 'stems-report.md'}")
    io.write_json(out / "run.json", evaluate._clean(run))     # the completion marker, last
    if best:
        _process_log(out, log)
    return run


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="eluate", allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("analyse", allow_abbrev=False)
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--work", type=Path, required=True)
    p.add_argument("--xover", type=_xover, default=None)
    p.add_argument("--bpm", type=float, default=None)
    p = sub.add_parser("plan", allow_abbrev=False)
    p.add_argument("work", type=Path)
    _add_controls(p)
    p = sub.add_parser("render", allow_abbrev=False)
    p.add_argument("plan", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--input", type=Path, default=None)
    p = sub.add_parser("master", allow_abbrev=False)
    p.add_argument("input", type=Path, nargs="?")
    p.add_argument("--stems", type=Path, nargs="+", default=None)
    p.add_argument("--mix", type=Path, default=None)
    p.add_argument("-o", "--work", type=Path, required=True)
    _add_controls(p)
    p.add_argument("--master-ref", type=Path, default=None)
    p.add_argument("--max-renders", type=int, default=solver.MAX_OUTER)
    p.add_argument("--bpm", type=float, default=None)
    p = sub.add_parser("report", allow_abbrev=False)
    p.add_argument("work", type=Path)
    p.add_argument("--master-ref", type=Path, default=None)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    log = lambda m: print(m, flush=True)  # noqa: E731
    try:
        if args.cmd == "analyse":
            do_analyse(args.input, args.work, args.xover, log, args.bpm)
        elif args.cmd == "plan":
            do_plan(args.work, _controls(args), log=log)
        elif args.cmd == "render":
            do_render(args.plan, args.output, args.input, log)
        elif args.cmd == "master":
            if (args.input is None) == (args.stems is None):
                raise ValueError("give one input, or --stems")
            solver.sweep_spill()     # a killed run's spill dirs (report 3.6)
            if args.stems is not None:
                do_master_stems(args.stems, args.work, _controls(args), args.mix,
                                args.max_renders, log, args.bpm)
            else:
                if args.mix is not None:
                    raise ValueError("--mix goes with --stems")
                do_master(args.input, args.work, _controls(args), args.master_ref,
                          args.max_renders, log, args.bpm)
        elif args.cmd == "report":
            do_report(args.work, args.master_ref, log)
    except (ValueError, OSError, sf.SoundFileError) as e:
        print(f"eluate: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
