"""Engine threads (sprint 2026-10-21 WP8): one process-wide thread count and the helpers
that spend it.

Contract:
- `set_threads(n)` / `threads()`: the count, DEFAULT_THREADS (os.cpu_count() - 2, at
  least 1) until set; the CLI sets it from Controls.threads before a run.
- `pmap(fn, items, cap=None)`: [fn(i) for i in items], run on a shared
  ThreadPoolExecutor of min(threads(), cap) workers, results in item order. Sequential
  when that is 1, for one item, or when called from inside a pmap task (no nested
  pools). `workers(cap)` is that count. BLOCK_CAP (8) is the cap for whole-block 8x
  and 32x work (the limiter's blocks, the FFT true-peak blocks): measured on 24 s to
  60 s of stereo, the speed-up stops near 8 workers (memory bandwidth) while every
  block in flight holds 40 to 50 MB. Callers pass only
  independent work whose result does not depend on when it runs, so every result is
  bit-identical to the sequential loop at any count. An exception is raised as the
  sequential loop would raise it (the first failing item's), after the items not yet
  started are cancelled and the running ones have finished (no task outlives the call,
  and a failed render does not keep the shared pool busy).
Threads, not processes: a Windows process pool would copy the audio. numpy and
scipy.fft release the GIL in large array operations; pure-Python loops do not speed up.
scipy.fft's own `workers` is not used: on the batched transforms of eluate.oa and
the true-peak resample it measured no faster (scipy 1.18).
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait

DEFAULT_THREADS = max(1, (os.cpu_count() or 1) - 2)
BLOCK_CAP = 8

_threads = DEFAULT_THREADS
_pools: dict[int, ThreadPoolExecutor] = {}
_lock = threading.Lock()
_local = threading.local()


def set_threads(n: int) -> None:
    global _threads
    n = int(n)
    if n < 1:
        raise ValueError("threads must be at least 1")
    _threads = n


def threads() -> int:
    return _threads


def _in_task() -> bool:
    return getattr(_local, "in_task", False)


def workers(cap: int | None = None) -> int:
    return 1 if _in_task() else max(1, min(_threads, cap or _threads))


def _executor(k: int) -> ThreadPoolExecutor:
    with _lock:
        if k not in _pools:
            _pools[k] = ThreadPoolExecutor(max_workers=k, thread_name_prefix="eluate")
        return _pools[k]


def _task(fn, item):
    _local.in_task = True
    try:
        return fn(item)
    finally:
        _local.in_task = False


def pmap(fn, items, cap: int | None = None) -> list:
    items = list(items)
    k = workers(cap)
    if k <= 1 or len(items) <= 1:
        return [fn(i) for i in items]
    futs = [_executor(k).submit(_task, fn, i) for i in items]
    try:
        return [f.result() for f in futs]
    except BaseException:
        for f in futs:
            f.cancel()
        wait(futs)
        raise
