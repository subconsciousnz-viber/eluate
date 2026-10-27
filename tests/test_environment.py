"""plan.environment and plan.check_environment (macOS report 3.4), eluate.__version__."""

import logging
import platform
import sys
import tomllib
import warnings
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest

import eluate
from eluate import plan as plan_mod, solver
from eluate.plan import (PLATFORM_KEYS, VERSION_KEYS, Plan, SidestripPlan, check_environment,
                            environment)
from eluate.stages import render

FS = 48000
SOFT = "rendered on a different platform"


def _plan(**changes):
    d = Plan(fs=FS, sidestrip=SidestripPlan()).to_dict()
    d["environment"].update(changes)
    return Plan.from_dict(d)


def test_environment_keys_and_values():
    e = environment()
    assert list(e) == list(VERSION_KEYS + PLATFORM_KEYS)
    assert e["platform"] == sys.platform
    assert e["machine"] == platform.machine()
    assert e["python"] == f"{sys.version_info.major}.{sys.version_info.minor}"
    assert e["eluate"] == eluate.__version__
    for k in ("numpy_blas", "scipy_lapack"):
        assert isinstance(e[k], str) and e[k]
    e["machine"] = "changed"                        # a fresh dict each call
    assert environment()["machine"] == platform.machine()


def test_build_dep_is_defensive():
    class Raises:
        @staticmethod
        def show_config(mode=None):
            raise TypeError("no mode argument")

    class Missing:
        @staticmethod
        def show_config(mode=None):
            return {"Build Dependencies": {}}

    class Found:
        @staticmethod
        def show_config(mode=None):
            return {"Build Dependencies": {"blas": {"name": "accelerate", "version": "14"}}}

    assert plan_mod._build_dep(Raises, "blas") == "unknown"
    assert plan_mod._build_dep(Missing, "blas") == "unknown"
    assert plan_mod._build_dep(Found, "blas") == "accelerate 14"


def test_environment_round_trips(tmp_path):
    p = Plan(fs=FS, sidestrip=SidestripPlan())
    p.save(tmp_path / "plan.json")
    assert Plan.load(tmp_path / "plan.json").environment == environment()


@pytest.mark.parametrize("changes, hard, soft", [
    ({}, False, False),
    ({"machine": "arm64", "numpy_blas": "accelerate 14"}, False, True),
    ({"platform": "darwin"}, False, True),
    ({"scipy": "0.0.0", "machine": "arm64"}, True, False),   # versions win: one warning
])
def test_check_environment(caplog, changes, hard, soft):
    rec = {**environment(), **changes}
    with caplog.at_level(logging.WARNING, logger="eluate.plan"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_environment(rec)
    assert any("bit-identical" in str(x.message) for x in w) is hard
    assert (SOFT in caplog.text) is soft
    if soft:
        for k, v in changes.items():
            assert f"{k} {v} -> {environment()[k]}" in caplog.text


def test_plan_without_platform_keys_is_quiet(caplog):
    """A plan saved before report 3.4 records only the versions: no warning, no log."""
    rec = {k: environment()[k] for k in VERSION_KEYS}
    with caplog.at_level(logging.WARNING, logger="eluate.plan"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_environment(rec)
    assert caplog.text == ""


@pytest.mark.parametrize("run", [
    lambda x, p: render(x, FS, p),
    lambda x, p: solver.render_plan(x, FS, p),
])
def test_renders_log_platform_difference_without_warning(caplog, run):
    x = np.zeros((1000, 2))
    with caplog.at_level(logging.WARNING, logger="eluate.plan"):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            run(x, _plan(machine="some-other-cpu"))
    assert SOFT in caplog.text
    caplog.clear()
    with pytest.warns(RuntimeWarning, match="bit-identical"):
        run(x, _plan(numpy="0.0.0"))
    assert SOFT not in caplog.text


def test_version_single_source():
    """pyproject's version is the one source: the metadata reports it and the fallback
    (a source tree or frozen bundle without metadata) matches it."""
    root = Path(__file__).resolve().parent.parent
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert eluate._FALLBACK_VERSION == pyproject["project"]["version"]
    assert eluate.__version__ == metadata.version("eluate")
