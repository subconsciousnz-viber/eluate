"""Short synthetic tracks for the workstream E tests (planner, solver, evaluate, cli).

Contract: `short_track(fs, seconds_per_section)` returns (x, meta) where x is a
deterministic (n, 2) float64 track: `synthetic.sectioned_track` (kick, hats,
sub, pad, level steps per section) with a +9 dB Q 10 ring at 3150 Hz over
25 % to 65 % of the track (under the planner's suspect-span limit), scaled
so its sample peak is -6 dBFS.
`short_analysis(fs, bars)` returns (x, analysis) with the analysis cached per
process (features.analyse is the slow part of these tests).
`short_planner_analysis(fs, bars, level_db)` returns (x scaled by level_db,
planner.analyse of it): input normalisation and key-locked crossovers, cached
the same way.
`dense_track(fs)` / `dense_planner_analysis(fs)`: a dense, low-crest track
whose budgets reach the default RMS target (chunk 4 loudness tests).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from fixtures import synthetic as syn


def short_track(fs: int = 44100, bars_per_section: int = 2):
    x, meta = syn.sectioned_track(fs, bars_per_section=bars_per_section)
    dur = x.shape[0] / fs
    x, _ = syn.add_ring(x, fs, f0=3150.0, q=10.0, gain_db=9.0, start_s=0.25 * dur,
                        end_s=0.65 * dur)
    x = x * (10 ** (-6 / 20) / np.max(np.abs(x)))
    return x, meta


@lru_cache(maxsize=4)
def short_analysis(fs: int = 44100, bars_per_section: int = 2):
    from eluate import features
    x, _ = short_track(fs, bars_per_section)
    return x, features.analyse(x, fs)


@lru_cache(maxsize=4)
def short_planner_analysis(fs: int = 44100, bars_per_section: int = 2, level_db: float = 0.0):
    from eluate import planner
    x, _ = short_track(fs, bars_per_section)
    x = x * 10.0 ** (level_db / 20.0)
    return x, planner.analyse(x, fs)


def features_with_key_confidence(monkeypatch, conf: float = 0.9) -> None:
    """features.analyse as the features branch has it (audit features F2, 2026-11-10):
    the key's confidence goes to a fcs_from_key taking `key_confidence` and to
    plan_inputs["key_confidence"], so planner.analyse's key gate can pass. That branch
    has landed (features.analyse passes pitch.key's own confidence); this pins the value
    so a test does not depend on what pitch.key reads from a synthetic track."""
    from eluate import features
    real = features.analyse

    def analyse(x, fs, *a, fcs_from_key=None, **kw):
        fn = None if fcs_from_key is None else (
            lambda tonic: fcs_from_key(tonic, key_confidence=conf))
        out = real(x, fs, *a, fcs_from_key=fn, **kw)
        out["plan_inputs"]["key_confidence"] = conf
        out["meta"]["key_confidence"] = conf
        return out
    monkeypatch.setattr(features, "analyse", analyse)


@lru_cache(maxsize=2)
def keyed_planner_analysis(fs: int = 44100, bars_per_section: int = 2):
    """short_planner_analysis with a confident key (features_with_key_confidence): the
    key-locked crossovers."""
    import pytest
    with pytest.MonkeyPatch.context() as mp:
        features_with_key_confidence(mp)
        return short_planner_analysis.__wrapped__(fs, bars_per_section)


def dense_track(fs: int = 44100, seconds: float = 8.0, seed: int = 5):
    """A dense, low-crest stereo track (a harmonic stack on 55 Hz plus pink noise, through
    a tanh), sample peak -6 dBFS: material whose budgets reach the default RMS target
    (planner.DEFAULT_TARGET_RMS_DBFS)."""
    from scipy.signal import lfilter
    n = int(seconds * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 2))
    for f, a in ((55, 0.5), (110, 0.3), (220, 0.25), (330, 0.2), (440, 0.15), (660, 0.12),
                 (880, 0.1), (1760, 0.06), (3520, 0.04)):
        x += a * np.sin(2 * np.pi * f * t[:, None] + rng.random(2) * 2 * np.pi)
    w = rng.standard_normal((n, 2))
    pink = lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786],
                   [1, -2.494956002, 2.017265875, -0.522189400], w, axis=0)
    x += 0.18 * pink / np.std(pink)
    x = np.tanh(1.5 * x) / 1.5
    return x * (10 ** (-6 / 20) / np.max(np.abs(x)))


@lru_cache(maxsize=2)
def dense_planner_analysis(fs: int = 44100):
    from eluate import planner
    x = dense_track(fs)
    return x, planner.analyse(x, fs)
