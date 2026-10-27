"""Engine threads (sprint 2026-10-21 WP8): every threaded path gives the bytes one thread
gives."""

import json
import tracemalloc

import numpy as np
import pytest

from fixtures.e_signals import short_track
from eluate import cli, io, meter, parallel, solver
from eluate import plan as plan_mod
from eluate.plan import LimiterPlan, Plan
from eluate.stages import limiter as lim

FS = 44100


@pytest.fixture(scope="module")
def hot():
    x, _ = short_track(FS)
    return x * 10.0 ** (8.0 / 20.0)        # well over the ceiling: clipper and limiter act


def _both(fn):
    parallel.set_threads(1)
    one = fn()
    parallel.set_threads(8)
    return one, fn()


def test_pmap_order_errors_and_nesting():
    parallel.set_threads(4)
    assert parallel.pmap(lambda i: i * i, range(20)) == [i * i for i in range(20)]
    assert parallel.workers(2) == 2 and parallel.workers() == 4

    def bad(i):
        if i in (3, 7):
            raise ValueError(i)
        return i
    with pytest.raises(ValueError, match="3"):
        parallel.pmap(bad, range(10))
    # a pmap inside a task runs sequentially (no nested pools)
    assert parallel.pmap(lambda i: parallel.workers(), range(4)) == [1] * 4
    with pytest.raises(ValueError):
        parallel.set_threads(0)


PLANS = {
    "plain": LimiterPlan(ceiling_dbtp=-1.0, input_gain_db=2.0),
    "onsets": LimiterPlan(ceiling_dbtp=-1.0, input_gain_db=2.0, clip_depth_db=2.5,
                          clip_onsets_s=[0.5 * k for k in range(40)]),
    "whole_split": LimiterPlan(ceiling_dbtp=-1.0, input_gain_db=2.0, clip_threshold_db=-3.0,
                               clip_split_hz=110.0, release_ms=120.0),
}


@pytest.mark.parametrize("kind", sorted(PLANS))
@pytest.mark.slow
def test_limit_blocked_bit_identical(hot, kind):
    plan = PLANS[kind]
    (y1, i1), (y8, i8) = _both(lambda: solver.limit_blocked(hot, FS, plan, block=8192))
    assert hot.shape[0] > 20 * 8192                     # many blocks, several waves
    assert np.array_equal(y1, y8)
    assert json.dumps(i1, sort_keys=True) == json.dumps(i8, sort_keys=True)
    m1, m8 = _both(lambda: solver.limit_blocked(hot, FS, plan, block=8192, measure_only=True,
                                                detector_env=True))
    assert np.array_equal(m1.pop("detector_env"), m8.pop("detector_env")) and m1 == m8


@pytest.mark.parametrize("threads", [2, 3, 7, 16])
def test_limit_blocked_thread_sweep_partial_block(hot, threads):
    """Audit 13 #3: every thread count, on a length that leaves a partial last block."""
    x = hot[: 20 * 8192 + 1234]
    plan = PLANS["onsets"]
    parallel.set_threads(1)
    y1, i1 = solver.limit_blocked(x, FS, plan, block=8192)
    parallel.set_threads(threads)
    yt, it = solver.limit_blocked(x, FS, plan, block=8192)
    assert np.array_equal(y1, yt)
    assert json.dumps(i1, sort_keys=True) == json.dumps(it, sort_keys=True)


def test_true_peak_envelopes_bit_identical(hot):
    a, b = _both(lambda: lim.true_peak_fft_envelope(hot, block=4096, margin=1024))
    assert np.array_equal(a, b)
    for L in (4, 8):
        a, b = _both(lambda: meter.true_peak_envelope(hot, FS, L))
        assert np.array_equal(a, b)


@pytest.mark.slow
def test_block_memory_is_capped(hot):
    """At most BLOCK_CAP blocks in flight, whatever the thread count."""
    plan = PLANS["onsets"]
    peaks = {}
    for t in (parallel.BLOCK_CAP, 4 * parallel.BLOCK_CAP):
        parallel.set_threads(t)
        tracemalloc.start()
        solver.limit_blocked(hot, FS, plan, block=8192)
        peaks[t] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    assert peaks[4 * parallel.BLOCK_CAP] < 1.2 * peaks[parallel.BLOCK_CAP]


@pytest.mark.slow
def test_master_bit_identical_at_any_thread_count(tmp_path):
    x, _ = short_track(FS)
    wav = tmp_path / "in.wav"
    io.write(wav, x, FS)
    out = {}
    for t in (1, 8):
        out[t] = tmp_path / f"t{t}"
        run = cli.do_master(wav, out[t], cli.planner.Controls(threads=t), max_renders=2,
                            log=lambda m: None)
        assert run["controls"]["threads"] == t
        assert json.loads((out[t] / "run.json").read_text(encoding="utf-8"))["controls"]["threads"] == t
    a, b = out[1], out[8]
    assert (a / "master.wav").read_bytes() == (b / "master.wav").read_bytes()
    pa, pb = (json.loads((d / "plan.json").read_text(encoding="utf-8")) for d in (a, b))
    assert pa.pop("run_id") != pb.pop("run_id")          # one fresh id per run (report 3.1)
    assert pa == pb
    Plan.load(a / "plan.json")
    with np.load(a / "plan.npz") as za, np.load(b / "plan.npz") as zb:
        assert sorted(za.files) == sorted(zb.files)
        for k in za.files:
            if k != plan_mod.RUN_ID_KEY:
                assert np.array_equal(za[k], zb[k]), k
    def untimed(v):                                     # wall-clock fields differ
        if isinstance(v, dict):
            return {k: untimed(x) for k, x in v.items() if k not in ("runtime_s", "t_s")}
        return [untimed(x) for x in v] if isinstance(v, list) else v
    sa, sb = (untimed(json.loads((d / "solve.json").read_text(encoding="utf-8"))) for d in (a, b))
    assert sa == sb


def test_threads_flag():
    a = cli.build_parser().parse_args(["master", "in.wav", "-o", "out", "--threads", "3"])
    assert cli._controls(a).threads == 3
    a = cli.build_parser().parse_args(["master", "in.wav", "-o", "out"])
    assert cli._controls(a).threads == parallel.DEFAULT_THREADS >= 1
    with pytest.raises(ValueError):
        cli.planner.Controls(threads=0)


def test_pmap_cancels_the_rest_on_a_failure():
    """Report 12h: the first failure cancels the items not yet started and waits for the
    running ones, so nothing runs on after pmap raises and the shared pool is free."""
    import threading
    parallel.set_threads(2)
    ran = []
    lock = threading.Lock()

    def fn(i):
        if i == 0:
            raise ValueError("first")
        threading.Event().wait(0.2)       # keep both workers busy while pmap cancels
        with lock:
            ran.append(i)
        return i
    with pytest.raises(ValueError, match="first"):
        parallel.pmap(fn, range(20))
    done = sorted(ran)
    assert len(done) <= 2                 # items 1 and 2 at most; 3 to 19 cancelled
    threading.Event().wait(0.3)
    assert sorted(ran) == done            # nothing ran after the raise
    assert parallel.pmap(lambda i: i, range(10)) == list(range(10))
    parallel.set_threads(parallel.DEFAULT_THREADS)
