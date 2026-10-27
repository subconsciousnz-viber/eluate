"""Memory guard for a render (maintainer, 2026-10-22: a render must respect the machine's
memory; docs/macos-report-verdicts.md 7).

Contract:
- `estimate_mb(frames, threads)`: a `master` run's peak resident memory in MiB for a
  track of `frames` stereo frames: MIB_PER_MFRAME per million frames plus BASE_MIB, plus
  THREADED_MIB when threads > 1. Fitted to run.json `peak_memory_mb` (PeakWorkingSetSize,
  Windows) against duration_s x fs: one thread, the four VA premasters at 11.0 to
  15.7 M frames (work/renders/*_gA and *_wp4, 2026-10-21; least squares 180.6 MiB per
  M frames + 238 MiB, within 3%); threaded, the same four at --threads 6 (*_s2: 409 to
  542 MiB over one thread) and Track A at --threads 30 (2026-10-22: 2861 MiB, 631 over
  one thread); THREADED_MIB rounds the largest up. So a 6 min 44.1 kHz track needs about
  3.0 GB on one thread, 10 min at 96 kHz about 10 GB.
- `available_mb()`: (available, total) physical memory in MiB, or None when the query
  fails. Windows: GlobalMemoryStatusEx (ullAvailPhys, ullTotalPhys); macOS: `vm_stat`
  free + inactive + speculative pages and `sysctl -n hw.memsize`; Linux and others:
  /proc/meminfo MemAvailable and MemTotal, else os.sysconf (SC_AVPHYS_PAGES).
- `guard(frames, threads, log)` -> (threads, record): with threads > 1 and an estimate
  over SAFETY (0.85) of the available memory, one thread (logged); if the one-thread
  estimate is still over the available memory, ValueError with a plain message (the CLI
  and the engine print it as one line). A failed query skips the guard (logged). record
  (run.json "memory"): estimate_mb (for the threads the run uses), available_mb,
  total_mb, threads_requested, threads; None where the query failed.
No new dependency: ctypes and the system's own tools only.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

MIB_PER_MFRAME = 180.0     # one thread, per million stereo frames
BASE_MIB = 240.0           # interpreter, libraries, fixed buffers
THREADED_MIB = 750.0       # extra with threads > 1 (blocks and band solves in flight)
SAFETY = 0.85              # threads > 1 only while the estimate is under this share


def estimate_mb(frames: int, threads: int = 1) -> float:
    return (MIB_PER_MFRAME * frames / 1e6 + BASE_MIB
            + (THREADED_MIB if threads > 1 else 0.0))


def _windows() -> tuple[float, float]:
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.WinDLL("kernel32").GlobalMemoryStatusEx(ctypes.byref(st)):
        raise OSError("GlobalMemoryStatusEx failed")
    return st.ullAvailPhys / 2 ** 20, st.ullTotalPhys / 2 ** 20


def _macos() -> tuple[float, float]:
    total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                               text=True, check=True, timeout=10).stdout.strip())
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True,
                        timeout=10).stdout
    page = int(re.search(r"page size of (\d+) bytes", vm).group(1))
    pages = {m.group(1): int(m.group(2))
             for m in re.finditer(r"^Pages (\w+):\s+(\d+)\.", vm, re.MULTILINE)}
    if "free" not in pages:
        raise OSError("vm_stat has no free pages line")
    free = sum(pages[k] for k in ("free", "inactive", "speculative") if k in pages)
    return free * page / 2 ** 20, total / 2 ** 20


def _linux() -> tuple[float, float]:
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            info = {k: int(v.split()[0]) for k, v in
                    (line.split(":", 1) for line in f if ":" in line)}
        return info["MemAvailable"] / 1024.0, info["MemTotal"] / 1024.0
    except (OSError, KeyError, ValueError):
        page = os.sysconf("SC_PAGE_SIZE")
        return (os.sysconf("SC_AVPHYS_PAGES") * page / 2 ** 20,
                os.sysconf("SC_PHYS_PAGES") * page / 2 ** 20)


def available_mb() -> tuple[float, float] | None:
    """(available, total) physical memory in MiB, or None when the query fails."""
    try:
        q = (_windows if sys.platform == "win32" else
             _macos if sys.platform == "darwin" else _linux)
        avail, total = q()
    except (OSError, ValueError, AttributeError, KeyError, subprocess.SubprocessError):
        return None
    if not (0 < avail <= total):
        return None
    return float(avail), float(total)


def _gb(mib: float) -> str:
    return f"{mib / 1024:.1f} GB"


def guard(frames: int, threads: int, log=print) -> tuple[int, dict]:
    """The thread count to run with and the run.json record (module contract)."""
    rec = {"estimate_mb": None, "available_mb": None, "total_mb": None,
           "threads_requested": threads, "threads": threads}
    mem = available_mb()
    if mem is None:
        log("memory check skipped: the free memory could not be read")
        return threads, rec
    avail, total = mem
    rec["available_mb"], rec["total_mb"] = round(avail), round(total)
    need = estimate_mb(frames, threads)
    if threads > 1 and need > SAFETY * avail:
        log(f"this render needs about {_gb(need)} with {threads} threads and "
            f"{_gb(avail)} is free: running on one thread")
        threads = 1
        need = estimate_mb(frames, threads)
    rec["estimate_mb"], rec["threads"] = round(need), threads
    if need > avail:
        raise ValueError(f"This render needs about {_gb(need)} of memory; {_gb(avail)} is "
                         f"free. Close other programs, or use a shorter or lower-rate file.")
    return threads, rec
