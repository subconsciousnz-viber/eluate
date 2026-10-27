"""Engine process for the desktop app (desktop/src-tauri spawns it; ui/ draws its output).

Contract:
- `preview <input> [--target-rms R]`, `master <input> -o <work> [cli controls] [--bpm B]
  [--stems S... [--mix M]]` (with --stems, input is the set's sum and the master runs
  cli.do_master_stems: the stems profile and stems-report), `stems-sum <stem>... [--mix M]
  -o <work>` (cli.sum_stems: the null test, the sum written to <work>/stems-sum.wav; one
  data message {"kind": "stems", "sum", "fs", "stems", "null"}),
  `analyse <input> -o <work> [--bpm B] [--xover a,b,c,d] [--threads N]`,
  `ui-data <input>` (uidata.from_render of uidata.work_dir(input), reading that input
  and refusing a work dir whose source.json names another file), `summary <input>...`
  (one data message per input, uidata.summary, no audio loaded).
- Talks to the shell in JSON lines on stdout, each prefixed with PREFIX so stray
  library output cannot be mistaken for a message: {"event": "stage", "stage": ...}
  (with step / steps, render / renders_max and eta_s where they apply, below),
  {"event": "log", "msg": ...}, {"event": "data", "ui": {...}} (uidata), and
  {"event": "error", "msg": ...}. Exit status 0 on success, 2 on an error.
- `master` runs cli.do_master (the work dir is the same as the CLI's, process log
  included) with its progress callback, reporting the stages Analysing (step 1 of 4),
  Planning (2), Mastering (3; once per solver render with render and renders_max =
  solver.render_bound) and Report (4), then Loading, and sends uidata.from_render. Every
  stage carries eta_s: estimate.left, the seconds the master has left from there (the
  frame count and the threads after the memory guard; a reused analysis costs nothing,
  and the solver's later renders are timed from its earlier ones). Before it,
  solver.sweep_spill removes stale solver spill dirs a killed engine left in the temp dir
  (only `master` spills, and only it loads the solver). A stem set's master runs up to
  1 + cli.STEM_PASSES_MAX passes, each with those four steps: from the second pass on,
  every stage carries pass (1-based) and passes_max, a "Correcting stems" stage (no step)
  opens each correction pass, and a kept correction pass ends with one more Report stage
  (step 4 of 4, no pass) while its report is rebuilt in the work dir root. eta_s is the
  current pass's (the passes that follow are not known ahead).
- `analyse` (maintainer feedback 2026-10-24, "it analyses again"): the analysis `master`
  would make for the same input, bpm and crossovers, written to the same cache in the
  work dir (cli.do_analyse: features.npz, source.json last), so the master that follows
  reuses it (cli._reuse_analysis). A cache that already matches is kept (no stage, log
  "analysis reused"). Otherwise one Analysing stage with eta_s (the analysis) and then_s
  (estimate.left of the master after it, the analysis reused). Runs on --threads engine
  threads (the shell sets them as for a master); no memory guard (the master that
  follows runs its own).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

PREFIX = "@@eluate "


def emit(event: str, **kw) -> None:
    sys.stdout.write(PREFIX + json.dumps({"event": event, **kw}, separators=(",", ":"),
                                         allow_nan=False, default=str) + "\n")
    sys.stdout.flush()


def _parser() -> argparse.ArgumentParser:
    from eluate import cli
    p = argparse.ArgumentParser(prog="eluate-engine", allow_abbrev=False)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("preview", allow_abbrev=False)
    q.add_argument("input", type=Path)
    q.add_argument("--target-rms", type=float, default=None)
    q = sub.add_parser("master", allow_abbrev=False)
    q.add_argument("input", type=Path)
    q.add_argument("-o", "--work", type=Path, required=True)
    q.add_argument("--bpm", type=float, default=None)
    q.add_argument("--stems", type=Path, nargs="+", default=None)
    q.add_argument("--mix", type=Path, default=None)
    cli._add_controls(q)
    q = sub.add_parser("stems-sum", allow_abbrev=False)
    q.add_argument("stems", type=Path, nargs="+")
    q.add_argument("--mix", type=Path, default=None)
    q.add_argument("-o", "--work", type=Path, required=True)
    q = sub.add_parser("analyse", allow_abbrev=False)
    q.add_argument("input", type=Path)
    q.add_argument("-o", "--work", type=Path, required=True)
    q.add_argument("--bpm", type=float, default=None)
    q.add_argument("--xover", type=cli._xover, default=None)
    q.add_argument("--threads", type=int, default=None)
    q = sub.add_parser("ui-data", allow_abbrev=False)
    q.add_argument("input", type=Path)
    q = sub.add_parser("summary", allow_abbrev=False)
    q.add_argument("inputs", type=Path, nargs="+")
    return p


STEPS = {"analyse": ("Analysing", 1), "plan": ("Planning", 2), "solve": ("Mastering", 3),
         "render": ("Mastering", 3), "report": ("Report", 4)}


class _Progress:
    """cli.do_master's progress callback: one stage message per phase (module contract)."""

    def __init__(self, frames: int, threads: int):
        self.frames, self.threads = frames, threads
        self.reused = False
        self.starts: list[float] = []        # perf_counter at each solver render's start
        self.pass_no: int | None = None      # a stem set's pass (1-based), from the second on
        self.passes_max: int | None = None

    def eta(self, phase: str, render: int = 1) -> float:
        from eluate import estimate
        measured = [b - a for a, b in zip(self.starts, self.starts[1:])]
        return round(estimate.left(phase, self.frames, self.threads, self.reused, render,
                                   measured), 1)

    def __call__(self, phase: str, frames=None, threads=None, reused=None, render=None,
                 renders_max=None, index=None, passes_max=None) -> None:
        if phase == "pass":                  # cli.do_master_stems, before each pass
            self.starts = []
            if not index:
                return                       # the sum pass: the plain four steps
            self.pass_no, self.passes_max = index + 1, passes_max
            self.reused = False               # a new sum: its analysis is not cached
            emit("stage", stage="Correcting stems", **self._pass(),
                 eta_s=self.eta("analyse"))
            return
        if phase == "keep":                  # a correction pass won: its report, in the root
            self.pass_no = self.passes_max = None
            emit("stage", stage="Report", step=4, steps=4, eta_s=self.eta("report"))
            return
        if phase == "analyse":
            self.frames, self.threads, self.reused = frames, threads, bool(reused)
        extra = {}
        if phase == "render":
            self.starts.append(time.perf_counter())
            extra = {"render": render, "renders_max": renders_max}
        elif phase == "solve":
            return                           # the stage comes with render 1, just after
        name, step = STEPS[phase]
        emit("stage", stage=name, step=step, steps=4,
             eta_s=self.eta(phase, render or 1), **extra, **self._pass())

    def _pass(self) -> dict:
        return {} if self.pass_no is None else {"pass": self.pass_no,
                                                 "passes_max": self.passes_max}


def _frames(path: Path) -> int:
    import soundfile as sf
    return sf.info(str(path)).frames


def _master(args) -> None:
    from eluate import cli, solver, uidata

    solver.sweep_spill()
    frames = _frames(args.input)
    prog = _Progress(frames, args.threads)
    prog("analyse", frames=frames, threads=args.threads, reused=False)
    log = lambda m: emit("log", msg=str(m))   # noqa: E731
    if args.stems:                            # a stem set: input is its sum (cli.STEMS_SUM)
        cli.do_master_stems(args.stems, args.work, cli._controls(args), args.mix, log=log,
                            bpm=args.bpm, progress=prog)
    else:
        cli.do_master(args.input, args.work, cli._controls(args), log=log, bpm=args.bpm,
                      progress=prog)
    emit("stage", stage="Loading", eta_s=prog.eta("load"))
    emit("data", ui=uidata.from_render(args.work, args.input))


def _stems_sum(args) -> None:
    from eluate import cli
    emit("stage", stage="Summing stems")
    _, fs, nt, sum_path = cli.sum_stems(args.stems, args.work, args.mix,
                                        log=lambda m: emit("log", msg=str(m)))
    emit("data", ui={"kind": "stems", "sum": str(sum_path.resolve()), "fs": fs,
                     "stems": len(args.stems), "null": nt})


def _analyse(args) -> None:
    from eluate import cli, estimate, parallel

    if args.threads is not None:
        parallel.set_threads(args.threads)
    threads = parallel.threads()
    log = lambda m: emit("log", msg=str(m))   # noqa: E731
    frames = _frames(args.input)
    if cli._reuse_analysis(args.work, args.input, args.bpm, args.xover, log) is not None:
        return
    emit("stage", stage="Analysing",
         eta_s=round(estimate.seconds("analyse", frames, threads), 1),
         then_s=round(estimate.left("plan", frames, threads), 1))
    cli.do_analyse(args.input, args.work, args.xover, log, args.bpm)


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.cmd == "preview":
            from eluate import planner, uidata
            t = planner.DEFAULT_TARGET_RMS_DBFS if args.target_rms is None else args.target_rms
            emit("data", ui=uidata.preview(args.input, t))
        elif args.cmd == "master":
            _master(args)
        elif args.cmd == "analyse":
            _analyse(args)
        elif args.cmd == "stems-sum":
            _stems_sum(args)
        elif args.cmd == "summary":
            from eluate import uidata
            for p in args.inputs:
                emit("data", ui=uidata.summary(p))
        else:
            from eluate import uidata
            emit("data", ui=uidata.from_render(uidata.work_dir(args.input), args.input))
    except Exception as e:                # the shell shows one line; the trace goes to stderr
        traceback.print_exc(file=sys.stderr)
        emit("error", msg=f"{type(e).__name__}: {e}" if not isinstance(e, (ValueError, OSError))
             else str(e))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
