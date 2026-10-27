"""Memory guard (eluate.memory; maintainer, 2026-10-22)."""

import pytest

from fixtures.e_signals import short_track
from eluate import cli, io, memory

FS = 44100


def test_estimate_arithmetic():
    m = memory
    assert m.estimate_mb(0, 1) == m.BASE_MIB
    assert m.estimate_mb(10_000_000, 1) == pytest.approx(10 * m.MIB_PER_MFRAME + m.BASE_MIB)
    assert m.estimate_mb(10_000_000, 8) == pytest.approx(
        m.estimate_mb(10_000_000, 1) + m.THREADED_MIB)
    assert m.estimate_mb(10_000_000, 2) == m.estimate_mb(10_000_000, 30)
    # the measured anchors (module contract): Track A 11.03 M frames, 2230 MiB on one thread
    assert m.estimate_mb(11_030_000, 1) == pytest.approx(2230, rel=0.05)
    assert m.estimate_mb(15_700_000, 1) == pytest.approx(3014, rel=0.05)


def test_query_is_sane():
    mem = memory.available_mb()
    assert mem is not None
    avail, total = mem
    assert 0 < avail <= total and total > 256


def _fake(monkeypatch, avail, total=64 * 1024):
    monkeypatch.setattr(memory, "available_mb", lambda: (float(avail), float(total)))


def test_guard_keeps_threads_with_room(monkeypatch):
    _fake(monkeypatch, 32 * 1024)
    logs = []
    threads, rec = memory.guard(10_000_000, 8, logs.append)
    assert threads == 8 and not logs
    assert rec == {"estimate_mb": round(memory.estimate_mb(10_000_000, 8)),
                   "available_mb": 32 * 1024, "total_mb": 64 * 1024,
                   "threads_requested": 8, "threads": 8}


def test_guard_drops_to_one_thread(monkeypatch):
    frames = 10_000_000
    one, many = memory.estimate_mb(frames, 1), memory.estimate_mb(frames, 8)
    # threaded estimate over 85% of the free memory, one thread fits
    avail = (one + many * memory.SAFETY) / 2 / memory.SAFETY
    assert many > memory.SAFETY * avail and one <= avail
    _fake(monkeypatch, avail)
    logs = []
    threads, rec = memory.guard(frames, 8, logs.append)
    assert threads == 1 and rec["threads"] == 1 and rec["threads_requested"] == 8
    assert rec["estimate_mb"] == round(one)
    assert len(logs) == 1 and "one thread" in logs[0]
    # one thread asked for: no drop, no log
    logs.clear()
    assert memory.guard(frames, 1, logs.append)[0] == 1 and not logs


def test_guard_refuses_what_does_not_fit(monkeypatch):
    _fake(monkeypatch, 1024)                      # 1 GB free
    with pytest.raises(ValueError, match=r"needs about 2\.\d GB of memory; 1\.0 GB is free"):
        memory.guard(10_000_000, 8, lambda m: None)


def test_guard_skips_when_the_query_fails(monkeypatch):
    monkeypatch.setattr(memory, "available_mb", lambda: None)
    logs = []
    threads, rec = memory.guard(10 ** 9, 8, logs.append)
    assert threads == 8 and rec["available_mb"] is None and "skipped" in logs[0]


def test_query_failure_is_none(monkeypatch):
    def boom():
        raise OSError("no")
    monkeypatch.setattr(memory, "_windows", boom)
    monkeypatch.setattr(memory, "_macos", boom)
    monkeypatch.setattr(memory, "_linux", boom)
    assert memory.available_mb() is None


def test_macos_parse(monkeypatch):
    vm = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
          "Pages free:                               10000.\n"
          "Pages active:                             90000.\n"
          "Pages inactive:                           20000.\n"
          "Pages speculative:                         1000.\n")

    class R:
        def __init__(self, out):
            self.stdout = out

    def run(cmd, **k):
        return R(str(16 * 2 ** 30) if cmd[0] == "sysctl" else vm)
    monkeypatch.setattr(memory.subprocess, "run", run)
    avail, total = memory._macos()
    assert total == 16 * 1024 and avail == pytest.approx(31000 * 16384 / 2 ** 20)


def test_master_refuses_cleanly(tmp_path, monkeypatch, capsys):
    """The refusal comes before any audio is read and prints one line, exit 2."""
    x, _ = short_track(FS)
    inp = tmp_path / "in.wav"
    io.write(inp, x, FS)
    _fake(monkeypatch, 1.0)
    monkeypatch.setattr(cli, "_reuse_analysis", lambda *a, **k: pytest.fail("audio read"))
    assert cli.main(["master", str(inp), "-o", str(tmp_path / "out")]) == 2
    err = capsys.readouterr()
    assert "This render needs about" in err.out + err.err
    assert not (tmp_path / "out" / "run.json").exists()
