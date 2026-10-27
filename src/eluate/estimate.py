"""Time left in a render, for the desktop app's progress line (app_engine; maintainer feedback
2026-10-24: "there's no estimate").

Contract:
- `seconds(phase, frames, threads)`: the fitted time of one phase of `master` for
  `frames` frames on `threads` engine threads, (A + B / threads) seconds per million
  frames with (A, B) = RATES[phase]. Phases: "analyse" (planner.analyse), "plan",
  "render1" (the solver's first render, the stages it caches before it included),
  "render" (each later render: LATER_RENDER x render1), "report" (the limiter GR estimate
  and evaluate.report), "load" (uidata.from_render after the master).
- `left(phase, frames, threads, ...)`: the seconds left in a master at the start of
  `phase` ("analyse", "plan", "solve", "render", "report", "load"): that phase and every
  one after it, the load included. A reused analysis costs nothing. The solver's render
  count is not known ahead: EXPECTED_RENDERS are assumed, and once render n has started
  at least n. At "render" n (1-based), `measured` holds the seconds the renders before it
  took; later renders are timed from them (the mean of the measured later renders, else
  LATER_RENDER x the measured first), not from the fit.
- Constants, fitted on this machine (32 cores, Windows 11) from the eight run.json /
  solve.json in work/renders whose controls record the threads (2026-10-21 to 10-23):
  the kicktest renders A and C of Tracks B and E (128 s and 202 s excerpts, 30 threads)
  and the *_s2 renders of Tracks A, C, F and G (full tracks, 230 to 356 s, 6 threads). Per phase,
  the mean seconds per million frames at each thread count, the line A + B / threads
  through both means: analyse 6.42 at 30 threads, 9.13 at 6; render1 (solve_s minus the
  later renders, from solve.json history t_s) 15.91 and 27.57; report 4.16 and 5.64; plan
  0.06 at both. Later renders (the history's t_s steps, five, all at 6 threads) took
  0.82 of the first. EXPECTED_RENDERS = 2: the median render count of the 30 run.json
  there. load: uidata.from_render timed on Track G's *_s2 (1.02 s, 13.85 M frames) and
  Track B's kicktest C (0.47 s, 6.16 M), one thread. Refit when the solver's cost moves.
- An estimate, not a promise: the page shows it rounded ("about 6 min left").
"""

from __future__ import annotations

RATES = {                       # (A, B): seconds per million frames = A + B / threads
    "analyse": (5.74, 20.3),
    "plan": (0.06, 0.0),
    "render1": (13.0, 87.5),
    "report": (3.79, 11.2),
    "load": (0.075, 0.0),
}
LATER_RENDER = 0.82             # a later render's time, against the first's
EXPECTED_RENDERS = 2
PHASES = ("analyse", "plan", "solve", "render", "report", "load")


def seconds(phase: str, frames: int, threads: int) -> float:
    """The fitted time of one phase (module contract)."""
    if phase == "render":
        return LATER_RENDER * seconds("render1", frames, threads)
    a, b = RATES[phase]
    return (a + b / max(1, int(threads))) * frames / 1e6


def _renders_left(frames, threads, render, measured) -> float:
    """Render `render` (1-based, about to start) and the expected ones after it."""
    measured = list(measured or [])
    if len(measured) >= 2:
        later = sum(measured[1:]) / (len(measured) - 1)
    elif measured:
        later = LATER_RENDER * measured[0]
    else:
        later = seconds("render", frames, threads)
    this = seconds("render1", frames, threads) if render <= 1 else later
    return this + max(0, EXPECTED_RENDERS - render) * later


def left(phase: str, frames: int, threads: int, reused: bool = False, render: int = 1,
         measured=None) -> float:
    """Seconds left in a master at the start of `phase` (module contract)."""
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}")
    i = PHASES.index(phase)
    s = seconds("load", frames, threads)
    if i <= PHASES.index("report"):
        s += seconds("report", frames, threads)
    if i <= PHASES.index("render"):
        s += _renders_left(frames, threads, render if phase == "render" else 1,
                           measured if phase == "render" else None)
    if i <= PHASES.index("plan"):
        s += seconds("plan", frames, threads)
    if i == 0 and not reused:
        s += seconds("analyse", frames, threads)
    return s
