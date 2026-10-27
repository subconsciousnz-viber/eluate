"""Plan dataclasses and (de)serialisation (spec 1, 3.5; audit 05 A14).

Contract:
- A `Plan` holds one optional section per render stage, in the fixed chain
  order input, inputeq, notch, sidestrip, multiband, tone, saturate, glue,
  keyeq, limiter, output. A section that is None is skipped by
  `stages.render` (so is a keyeq section with enabled False, and inputeq and
  tone under preserve_mix). The planner leaves tone None (the FIR tone stage
  is out of the default chain since chunk 2 of docs/gain-staging.md; a
  hand-built tone section still renders). The plan is the whole input of a
  render besides the audio and its sample rate: every stage is a pure
  function of (audio, fs, section).
- `input` (InputPlan, docs/gain-staging.md 2 item 1): the input normalisation
  gain, applied first. Its record fields (the reference and the premaster's
  measured loudest-section 1 s RMS) are not read by the render.
- `output` (OutputPlan, docs/gain-staging.md 3 "Ceiling"): the export format
  of the master, applied last: 24-bit PCM rounded (default), 16-bit PCM with
  TPDF dither (the streaming option), or 32-bit float (no quantisation). The
  render returns the samples exactly as written (k / 2^(bits - 1)), so the
  true peak measured on it is the true peak of the file. The dither is
  seeded (dither_seed) so a render from a saved plan stays bit-exact.
- Time-varying controls are `Envelope`s at the 1 ms control rate
  (CONTROL_RATE_HZ). Control sample k sits at t = k / rate_hz; audio sample i
  sits at control coordinate `control_coord(i, fs, rate_hz)` = i * rate_hz / fs
  (product first, one rounding; every stage that maps samples to control
  frames uses this same function). `Envelope.at(n, fs)` interpolates with PCHIP
  (shape-preserving cubic, scipy PchipInterpolator) and holds the end values:
  PCHIP never leaves the range of the two neighbouring control values, so a cap
  that holds at the control points holds at every sample, and a run of equal
  control values interpolates to exactly that value. A one-value envelope is a
  constant.
- Envelope values are stored as float32-representable float64 (rounded to
  float32 on construction), so the plan.npz sidecar (float32, spec 1) reloads
  bit-identically and a render from a saved plan equals the render from the
  in-memory plan.
- The limiter section stores scalars only (spec 1, audit 01 F8); its onset list
  is a list of scalars, and speclim_lookahead_ms a scalar or one scalar per band.
- Analysis features that workstream C has not delivered (kick onsets, resonance
  list, per-band decoherence, band deficits) are plan inputs here.
- Validation on construction: NaN / Inf raise; crossovers and tone nodes must be
  positive and strictly ascending; `Plan` coerces fs to int, requires fs > 0,
  and requires every frequency in every section to lie below fs / 2.
- Bounds at validation: multiband side makeup [0, 3] dB, mid makeup [-3, 3] dB;
  saturation drive [0, 12] dB; every `order` a positive integer; keyeq
  min_confidence in [0, 1]; limiter clip ramp > 0 (audit 05 Q5, Q6, Q12, Q13);
  multiband ceiling envelopes <= 0 dB and makeup equal to the makeup the
  ceiling was designed with (MultibandPlan.check_ceiling, also run by the
  stage), ratio > 1, attack > 0, release >= 0, ceiling release > 0;
  glue mix in [0, 1], makeup in [0, 6] dB, gain envelope <= +6 dB, curve in
  {clean, vintage}.
- `InputEqPlan` (chunk 2): optional high-pass (fc, Q in [0.5, 1]), low shelf
  (gain, corner in [60, 250] Hz, Q in [0.5, 0.707]), bell (gain in [-3, 0],
  centre in [120, 1000] Hz, Q in [0.7, 4]), band shelf (chunk 4: gain in
  [-3, 0], low corner in [100, 1000] Hz, high corner 1/3 to 3 octaves above it)
  and high shelf (gain, corner), every gain within +-INPUTEQ_MAX_DB except the high
  shelves' boosts (INPUTEQ_HS_MID_MAX_DB, and INPUTEQ_HS_SIDE_MAX_DB for the optional
  side shelf high_shelf_side, which makes the top end M/S); phase "zero" or "minimum"; `design`
  holds the planner's fit inputs (read back by planner.refit_inputeq, never
  by the stage).
- Version 0.9 (maintainer, 2026-10-19: budgets on the average peak) adds `hits`
  (HitsPlan: the beat grid the budget solves read their hits from, its source and
  the reading, "average" or "max"; None = the whole track as one hit, the maximum).
  `HitsPlan.loud` (sprint 2026-10-21 D2, no version change: a plan without it loads with
  None) is the loud-hit mask frozen on the stage-1 input (the multiband's input,
  planner.freeze_hits), one bool per hit; every later stage budgets these hits instead of
  re-deriving the loud set on its own, flatter, input. None: each stage derives it.
  Version 0.8 (chunk 4: the input EQ's band shelf and the limiter's RMS
  target; 0.7, chunk 3: multiband ratio and knee tables, mode makeup and the
  budget record, the glue budget record, the whole-signal clipper and the
  limiter budget fields; 0.6 joined chunk 1's input and output sections with
  chunk 2's inputeq section, each of which was 0.5 on its own branch; 0.4
  added the multiband ceiling and design-record fields and the glue section);
  `from_dict` accepts only the current version.
- `environment` records the numpy, scipy and eluate versions the plan was
  made with, and the platform, machine, Python major.minor and the numpy BLAS / scipy
  LAPACK builds (current values by default, the stored ones on load); render is
  bit-exact only in that environment. `check_environment` (called by stages.render and
  solver.render_plan) warns (RuntimeWarning) on a version mismatch and only logs when
  just the platform keys differ (audit 05 Q11, report 3.4).
- `to_json` / `from_json` embed envelopes as lists (exact round trip).
  `save(path, run_id=None)` writes path (UTF-8 JSON, LF only) holding scalars plus
  sidecar key names, and path with suffix .npz holding the envelopes as float32
  arrays, each atomically (io.atomic_write), sidecar first; both carry run_id (a fresh
  uuid4 hex by default; the JSON's "run_id", the sidecar's RUN_ID_KEY), so a pair from
  two runs (a run stopped between the writes) is caught on load (report 3.1).
  `load(path)` reads both and raises ValueError naming the file when the JSON is not
  valid or not a plan (a KeyError / TypeError from `from_dict` included), when the
  sidecar name is not a bare file name ending .npz (it is joined to path's folder,
  report 12e), and when the run ids differ ("plan.json and plan.npz are from different
  runs"); a plan.json without a run id (written before it) loads unchecked.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import platform
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from eluate import bands, io

CONTROL_RATE_HZ = 1000.0
EVAL_CHUNK = 1 << 21          # audio samples per PCHIP evaluation chunk (memory only)
PLAN_VERSION = "0.10"
RUN_ID_KEY = "__run_id__"     # the plan.npz entry holding the run id
STAGE_ORDER = ("input", "inputeq", "notch", "sidestrip", "multiband", "tone", "saturate",
               "glue", "speclim", "keyeq", "limiter", "output")

# Hard caps from spec 3.5 (enforced inside the stages, recorded here once).
NOTCH_MAX_CUT_DB = 6.0
NOTCH_Q_MIN = 1.0
# compressor safety clamp: its 3 dB average-peak budget + the 1.5 dB outlier allowance
# (maintainer, 2026-10-19; was the 3 dB peak cap)
MULTIBAND_MAX_GR_DB = 6.0         # per-sample clamp of the compressor (D5: 12 hid 4 to 8.6 dB
                                  # of per-hit GR in the upper bands behind a 2.5 dB reading)
MULTIBAND_OUTLIER_CAP_DB = 4.5    # the compressor's outlier cap, on the meter reading
MULTIBAND_LIMIT_MAX_GR_DB = 3.0   # band limiter on top of the compressor (gain-staging 3)
MODE_MAKEUP_MAX_DB = MULTIBAND_MAX_GR_DB + MULTIBAND_LIMIT_MAX_GR_DB   # makeup modes, chunk 3
MAKEUP_MODES = ("peak", "rms", "auto")
GLUE_MODES = ("serial", "parallel")
MAKEUP_MAX_DB = 3.0          # side makeup in [0, 3] dB, mid makeup in [-3, 3] dB (audit 05 Q6)
SATURATE_MAX_DRIVE_DB = 12.0  # audit 05 Q5
TONE_MAX_DB = 3.0
INPUTEQ_MAX_DB = TONE_MAX_DB  # every input EQ gain (docs/gain-staging.md chunk 2)
INPUTEQ_HPF_FC_HZ = 25.0
INPUTEQ_HPF_Q_RANGE = (0.5, 1.0)          # 0.707: Butterworth sections
INPUTEQ_LOW_SHELF_FC_RANGE = (60.0, 250.0)
INPUTEQ_LOW_SHELF_Q_RANGE = (0.5, 0.707)
INPUTEQ_BELL_FC_RANGE = (120.0, 1000.0)
INPUTEQ_BELL_Q_RANGE = (0.7, 4.0)
INPUTEQ_BAND_SHELF_LO_RANGE = (100.0, 1000.0)   # band-shelf low corner (chunk 4)
INPUTEQ_BAND_SHELF_WIDTH_OCT = (1.0 / 3.0, 3.0)  # high corner = low corner x 2^width
INPUTEQ_HIGH_SHELF_FC_HZ = 16000.0
INPUTEQ_PHASES = ("zero", "minimum")
CLIP_MAX_DEPTH_DB = 3.0
CLIP_MAX_GR_DB = 1.5          # whole-signal clipper: at most 1.5 dB of peak reduction (3)
CLIP_SAMPLE_CAP_MAX_DB = 6.0  # per-sample reduction cap of the whole-signal curve, upper bound
CLIP_KNEE_MAX_DB = 6.0
GLUE_MAX_MAKEUP_DB = 6.0      # glue wet-path makeup in [0, 6] dB
GLUE_MAX_GAIN_DB = 6.0        # glue gain envelope <= +6 dB (audit 06 M3)
GLUE_CURVES = ("clean", "vintage")
HIGH_SIDE_MIN_HZ = 2000.0   # S bands whose centre is at or above this get the widening makeup
OUTPUT_BITS = (16, 24, 32)  # 16 / 24 = PCM, 32 = float (no quantisation)
DITHERS = ("none", "tpdf")


def _finite(name: str, v) -> None:
    a = np.asarray(v, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or Inf")


def _ascending(name: str, v) -> None:
    v = [float(f) for f in v]
    _finite(name, v)
    if any(f <= 0 for f in v) or any(b <= a for a, b in zip(v, v[1:])):
        raise ValueError(f"{name} must be positive and strictly ascending")


def _order(name: str, v) -> int:
    f = float(v)
    if not (math.isfinite(f) and f == int(f) and f >= 1):
        raise ValueError(f"{name} must be a positive integer, got {v!r}")
    return int(f)


# environment() keys: a difference in VERSION_KEYS warns that the render may not be
# bit-identical; a difference in PLATFORM_KEYS only logs (report 3.4).
VERSION_KEYS = ("numpy", "scipy", "eluate")
PLATFORM_KEYS = ("platform", "machine", "python", "numpy_blas", "scipy_lapack")
_log = logging.getLogger(__name__)


def _build_dep(module, key: str) -> str:
    """'name version' of a library's build dependency `key` ("blas", "lapack") from its
    show_config(mode="dicts"), or "unknown" when the release does not report it."""
    try:
        dep = module.show_config(mode="dicts")["Build Dependencies"][key]
        return f"{dep.get('name', 'unknown')} {dep.get('version', '')}".strip()
    except Exception:  # noqa: BLE001 - an older or unusual build: record, never fail
        return "unknown"


@functools.lru_cache(maxsize=1)
def _environment() -> tuple:
    import scipy

    from eluate import __version__
    return (("numpy", np.__version__), ("scipy", scipy.__version__),
            ("eluate", __version__), ("platform", sys.platform),
            ("machine", platform.machine()),
            ("python", f"{sys.version_info.major}.{sys.version_info.minor}"),
            ("numpy_blas", _build_dep(np, "blas")), ("scipy_lapack", _build_dep(scipy, "lapack")))


def environment() -> dict:
    """What a render's bit-exactness depends on (spec 1, audit 05 Q11, report 3.4): the
    numpy, scipy and eluate versions (VERSION_KEYS), then sys.platform,
    platform.machine(), the Python major.minor, and the BLAS / LAPACK numpy and scipy
    were built against (PLATFORM_KEYS). A fresh dict each call."""
    return dict(_environment())


def check_environment(recorded: dict, stacklevel: int = 3) -> None:
    """Compare a plan's recorded environment with the running one. A VERSION_KEYS
    difference issues a RuntimeWarning (the render may not be bit-identical); otherwise a
    PLATFORM_KEYS difference logs a warning (the last bits may differ: SIMD, FMA and
    BLAS / LAPACK vary by platform). A key the plan does not record (a plan saved before
    report 3.4) is not compared."""
    import warnings

    now = environment()
    was = {k: recorded.get(k) for k in VERSION_KEYS}
    cur = {k: now[k] for k in VERSION_KEYS}
    if was != cur:
        warnings.warn(f"plan was made with {was}, rendering with {cur}; "
                      "the render may not be bit-identical", RuntimeWarning,
                      stacklevel=stacklevel)
        return
    diff = {k: (recorded[k], now[k]) for k in PLATFORM_KEYS
            if k in recorded and recorded[k] != now[k]}
    if diff:
        _log.warning("rendered on a different platform: results may differ in the last "
                     "bits (%s)", ", ".join(f"{k} {a} -> {b}" for k, (a, b) in diff.items()))


def control_coord(i, fs: float, rate_hz: float = CONTROL_RATE_HZ):
    """Control-grid coordinate of audio sample index i (array or scalar)."""
    return np.asarray(i, dtype=np.float64) * rate_hz / fs


def control_length(n: int, fs: float, rate_hz: float = CONTROL_RATE_HZ) -> int:
    """Control samples needed to cover n audio samples (last point at or after the end)."""
    return int(math.ceil(float(control_coord(n - 1, fs, rate_hz)))) + 1 if n > 1 else 1


# ------------------------------------------------------------------ Envelope

@dataclass(eq=False)
class Envelope:
    """Control curve at `rate_hz` (default 1 kHz). Units are the maintainer's (dB or 0..1)."""

    values: np.ndarray
    rate_hz: float = CONTROL_RATE_HZ

    def __post_init__(self):
        v = np.array(self.values, dtype=np.float64).reshape(-1)
        if v.size == 0:
            raise ValueError("envelope needs at least one value")
        _finite("envelope", v)
        v32 = v.astype(np.float32)
        if not np.all(np.isfinite(v32)):
            raise ValueError("envelope value outside float32 range")
        if not self.rate_hz > 0:
            raise ValueError("rate_hz must be positive")
        self.values = v32.astype(np.float64)
        self.rate_hz = float(self.rate_hz)

    @classmethod
    def constant(cls, value: float) -> "Envelope":
        return cls(np.array([float(value)]))

    def at_samples(self, idx, fs: float) -> np.ndarray:
        """Curve at audio sample indices idx (PCHIP, end values held). Evaluated in
        chunks of EVAL_CHUNK indices (PCHIP evaluation is pointwise, so the values are
        the same as one call; spec 1 memory)."""
        v = self.values
        idx = np.asarray(idx)
        if v.size == 1:
            return np.full(idx.shape, v[0])
        f = PchipInterpolator(np.arange(v.size, dtype=np.float64), v)
        if idx.size <= EVAL_CHUNK:
            return f(np.clip(control_coord(idx, fs, self.rate_hz), 0.0, v.size - 1.0))
        flat = idx.reshape(-1)
        out = np.empty(flat.shape)
        for a in range(0, flat.size, EVAL_CHUNK):
            t = np.clip(control_coord(flat[a:a + EVAL_CHUNK], fs, self.rate_hz), 0.0,
                        v.size - 1.0)
            out[a:a + EVAL_CHUNK] = f(t)
        return out.reshape(idx.shape)

    def at(self, n: int, fs: float) -> np.ndarray:
        """Audio-rate curve of length n."""
        v = self.values
        if v.size == 1 or n <= EVAL_CHUNK:
            return self.at_samples(np.arange(n), fs)
        f = PchipInterpolator(np.arange(v.size, dtype=np.float64), v)
        out = np.empty(n)
        for a in range(0, n, EVAL_CHUNK):
            b = min(n, a + EVAL_CHUNK)
            t = np.clip(control_coord(np.arange(a, b), fs, self.rate_hz), 0.0, v.size - 1.0)
            out[a:b] = f(t)
        return out

    def to_dict(self) -> dict:
        return {"rate_hz": self.rate_hz, "values": self.values.tolist()}

    @classmethod
    def from_dict(cls, d: dict, sidecar=None) -> "Envelope":
        if "npz" in d:
            if sidecar is None:
                raise ValueError(f"envelope {d['npz']!r} needs the plan.npz sidecar")
            if d["npz"] not in sidecar:
                raise ValueError(f"envelope {d['npz']!r} is not in the plan.npz sidecar "
                                 "(plan.json and plan.npz from different runs?)")
            return cls(np.asarray(sidecar[d["npz"]], dtype=np.float64), d["rate_hz"])
        return cls(np.array(d["values"], dtype=np.float64), d.get("rate_hz", CONTROL_RATE_HZ))

    def __eq__(self, other) -> bool:
        return (isinstance(other, Envelope) and self.rate_hz == other.rate_hz
                and np.array_equal(self.values, other.values))


# ------------------------------------------------------------------ sections

@dataclass
class InputPlan:
    """Input normalisation: the premaster is multiplied by 10^(gain_db / 20) before the
    notch stage. Record (None on a hand-built plan, never read by the render):
    r_ref_dbfs, the reference loudest-section 1 s RMS (planner.R_REF_DBFS), and
    measured_dbfs, the premaster's own (meter.rms_1s)."""

    gain_db: float = 0.0
    r_ref_dbfs: float | None = None
    measured_dbfs: float | None = None

    def __post_init__(self):
        self.gain_db = float(self.gain_db)
        _finite("input gain", [self.gain_db])
        for name in ("r_ref_dbfs", "measured_dbfs"):
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, float(v))
                _finite(f"input {name}", [v])


@dataclass
class OutputPlan:
    """Export format of the master (module contract). bits 16 or 24: PCM, rounded to
    the nearest step of 2^-(bits - 1) after the dither and clipped to
    [-1, 1 - 2^-(bits - 1)]; bits 32: float, untouched. dither "tpdf": triangular
    noise of +-1 LSB (two uniform draws) from numpy's PCG64 seeded with dither_seed."""

    bits: int = 24
    dither: str = "none"
    dither_seed: int = 0

    def __post_init__(self):
        self.bits = _order("output bits", self.bits)
        if self.bits not in OUTPUT_BITS:
            raise ValueError(f"output bits must be one of {OUTPUT_BITS}")
        if self.dither not in DITHERS:
            raise ValueError(f"dither must be one of {DITHERS}")
        if self.bits == 32 and self.dither != "none":
            raise ValueError("float output takes no dither")
        f = float(self.dither_seed)
        if not (math.isfinite(f) and f == int(f) and f >= 0):
            raise ValueError("dither_seed must be a non-negative integer")
        self.dither_seed = int(f)

    @property
    def subtype(self) -> str:
        """soundfile subtype of the written file."""
        return {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}[self.bits]


@dataclass
class Notch:
    """One confirmed resonance. gain_db(t) <= 0 is the notch gain at f0 (capped at -6 dB).
    q = f0 / bandwidth of the notch response at half its deepest dB cut (spec 3.5 item 1)."""

    f0_hz: float
    q: float
    gain_db: Envelope

    def __post_init__(self):
        _finite("notch", [self.f0_hz, self.q])
        if self.f0_hz <= 0:
            raise ValueError("notch f0 must be positive")
        if self.q < NOTCH_Q_MIN:
            raise ValueError(f"notch q must be >= {NOTCH_Q_MIN}")


@dataclass
class NotchPlan:
    notches: list[Notch] = field(default_factory=list)
    kick_mask: Envelope | None = None   # m_kick(t) in [0, 1]; None = no ducking


@dataclass
class SidestripPlan:
    fc_hz: float = 100.0
    order: int = 2

    def __post_init__(self):
        _finite("sidestrip", [self.fc_hz])
        if self.fc_hz <= 0:
            raise ValueError("fc_hz must be positive")
        self.order = _order("sidestrip order", self.order)


def band_centres(crossovers_hz) -> list[float]:
    """Geometric centre of each band; the bottom band's is fc_1 / 2, the top band's 2 fc_N."""
    e = [float(f) for f in crossovers_hz]
    if not e:
        return [1000.0]
    out = [e[0] / 2.0]
    out += [math.sqrt(a * b) for a, b in zip(e, e[1:])]
    out.append(2.0 * e[-1])
    return out


def _band_pairs(name: str, v, nb: int, allow_none: bool = False) -> list[list] | None:
    """Validate a bands x [M, S] table of finite floats (None entries if allowed)."""
    if v is None:
        return None
    if len(v) != nb or any(len(p) != 2 for p in v):
        raise ValueError(f"{name} must be {nb} bands x [M, S]")
    out = []
    for p in v:
        row = []
        for e in p:
            if e is None and allow_none:
                row.append(None)
                continue
            f = float(e)
            if not math.isfinite(f):
                raise ValueError(f"{name} contains NaN or Inf")
            row.append(f)
        out.append(row)
    return out


@dataclass
class MultibandPlan:
    """gr_db[k] = [M envelope, S envelope] of band k, dB <= 0 (capped at -3 dB).

    S makeup is not free: bands whose geometric centre is at or above 2 kHz get
    side_makeup_db * (1 - decoherence[k]), every other band 0 (spec 3.5;
    audit 05 A10).

    Per-band ceiling (spec 3.5 item 3, maintainer ruling 2026-10-17):
    ceiling_gr_db[k] = [M, S] limiter envelopes, dB <= 0, multiplied into the
    compressor's gain; reported separately from gr_db and not subject to the
    3 dB compressor cap (spec 3.6: the cap is the compressor's). None = no
    ceiling stage. ceiling_db[k][c] is the ceiling the envelope was designed
    for, None where that band and channel has none. Its domain is the *band as
    the stage splits and processes it* (sample peak, dBFS, of the M or S band
    after compressor and makeup), not a re-split of the output, which reads
    differently (audit 06 F6, M8; the solver iterates on the re-split).
    ceiling_makeup_db[k][c] records the makeup (dB) the ceiling was designed
    with; check_ceiling() rejects a plan whose makeup has changed since
    (audit 06 M7), and envelopes > 0 dB. The ceiling holds only with the
    compressor envelopes it was designed on: rebuild with
    multiband.design_from after changing either.

    Design record (the solver's levers, read back by multiband.design_from):
    threshold_db, attack_ms, release_ms and ceiling_release_ms are
    bands x [M, S] (a scalar ceiling_release_ms is broadcast); ratio a scalar;
    ceiling_lookahead_ms (ms, > 0) a scalar for every band (plans before 2026-10-23,
    2.0) or one value per band (the character preset's band_lookahead_ms; the band
    ceiling of band k uses `lookahead_ms()[k]`); design_span_s the duration (s) of the audio
    the envelopes were designed on (audit 06 M12: beyond it the envelopes hold
    their last value and nothing is guaranteed). None on a hand-built plan.
    ratio is a scalar or bands x [M, S]; knee_db (compressor soft knee, dB, 0 = the
    hard knee of the original design) likewise.

    Gain-staging chunk 3 (docs/gain-staging.md 3, 5): `budget` is the design record
    of the budget solve (multiband.design_budget): the targets (comp_gr_db,
    comp_gr_max_db, total_normal_db, limit_max_db, makeup mode, ratio from crest)
    and, once designed, the result per band and channel and the warnings. With
    budget set, design_from solves the thresholds, ratios and ceilings on the
    audio it is given; threshold_db, ratio and ceiling_db then record the solve.
    mode_makeup_db[k] = [M, S] makeup envelopes of the makeup mode (dB, 0 to
    MODE_MAKEUP_MAX_DB), added after the ceiling on top of mid_makeup_db and the
    side makeup; None = none. The ceiling detector does not see them.
    apply() uses only the envelopes and makeup, never the design record.

    ms_link (the live profile, sprint 2026-10-21 L8): M and S of a band take one gain, the
    larger reduction of the two (compressor + ceiling) at each instant with the M makeup,
    so the band's image does not move when M and S would compress differently. The design
    record and the budget stay per channel (what each detector asked for)."""

    gr_db: list[list[Envelope]]
    crossovers_hz: list[float] = field(default_factory=lambda: list(bands.DEFAULT_CROSSOVERS))
    order: int = 2
    phase: str | list[str] = "zero"
    mid_makeup_db: list[float] | None = None
    side_makeup_db: float = 2.0
    decoherence: list[float] | None = None
    ceiling_gr_db: list[list[Envelope]] | None = None
    ceiling_db: list[list[float | None]] | None = None
    threshold_db: list[list[float]] | None = None
    ratio: float | list[list[float]] | None = None
    attack_ms: list[list[float]] | None = None
    release_ms: list[list[float]] | None = None
    ceiling_lookahead_ms: float | list[float] = 2.0
    ceiling_release_ms: float | list[list[float]] = 50.0
    ceiling_makeup_db: list[list[float]] | None = None
    design_span_s: float | None = None
    knee_db: float | list[list[float]] = 0.0
    mode_makeup_db: list[list[Envelope]] | None = None
    budget: dict | None = None
    ms_link: bool = False

    def __post_init__(self):
        _ascending("crossovers_hz", self.crossovers_hz)
        nb = len(self.crossovers_hz) + 1
        if len(self.gr_db) != nb or any(len(p) != 2 for p in self.gr_db):
            raise ValueError(f"gr_db must be {nb} bands x [M, S]")
        if self.ceiling_gr_db is not None:
            if len(self.ceiling_gr_db) != nb or any(len(p) != 2 for p in self.ceiling_gr_db):
                raise ValueError(f"ceiling_gr_db must be {nb} bands x [M, S]")
        self.ceiling_db = _band_pairs("ceiling_db", self.ceiling_db, nb, allow_none=True)
        self.threshold_db = _band_pairs("threshold_db", self.threshold_db, nb)
        self.attack_ms = _band_pairs("attack_ms", self.attack_ms, nb)
        self.release_ms = _band_pairs("release_ms", self.release_ms, nb)
        self.ceiling_makeup_db = _band_pairs("ceiling_makeup_db", self.ceiling_makeup_db, nb)
        if isinstance(self.ceiling_release_ms, (int, float)):
            self.ceiling_release_ms = [[float(self.ceiling_release_ms)] * 2 for _ in range(nb)]
        self.ceiling_release_ms = _band_pairs("ceiling_release_ms", self.ceiling_release_ms, nb)
        if self.attack_ms is not None and any(a <= 0 for p in self.attack_ms for a in p):
            raise ValueError("attack_ms must be > 0")
        if self.release_ms is not None and any(r < 0 for p in self.release_ms for r in p):
            raise ValueError("release_ms must be >= 0")
        if any(r <= 0 for p in self.ceiling_release_ms for r in p):
            raise ValueError("ceiling_release_ms must be > 0 (audit 06 M11)")
        if self.ratio is not None:
            if isinstance(self.ratio, (int, float, np.floating)):
                self.ratio = float(self.ratio)
                rs = [self.ratio]
            else:
                self.ratio = _band_pairs("ratio", self.ratio, nb)
                rs = [r for p in self.ratio for r in p]
            if not all(math.isfinite(r) and r > 1.0 for r in rs):
                raise ValueError("ratio must be > 1")
        if isinstance(self.knee_db, (int, float, np.floating)):
            self.knee_db = float(self.knee_db)
            ks = [self.knee_db]
        else:
            self.knee_db = _band_pairs("knee_db", self.knee_db, nb)
            ks = [k for p in self.knee_db for k in p]
        if not all(math.isfinite(k) and k >= 0.0 for k in ks):
            raise ValueError("knee_db must be finite and >= 0")
        if self.mode_makeup_db is not None:
            if len(self.mode_makeup_db) != nb or any(len(p) != 2 for p in self.mode_makeup_db):
                raise ValueError(f"mode_makeup_db must be {nb} bands x [M, S]")
            for p in self.mode_makeup_db:
                for e in p:
                    if np.any(e.values < 0.0) or np.any(e.values > MODE_MAKEUP_MAX_DB + 1e-9):
                        raise ValueError(f"mode_makeup_db must lie in [0, {MODE_MAKEUP_MAX_DB}] dB")
        if isinstance(self.ceiling_lookahead_ms, (int, float, np.floating)):
            self.ceiling_lookahead_ms = float(self.ceiling_lookahead_ms)
        else:
            self.ceiling_lookahead_ms = [float(v) for v in self.ceiling_lookahead_ms]
            if len(self.ceiling_lookahead_ms) != nb:
                raise ValueError(f"ceiling_lookahead_ms needs one value or {nb} values")
        _finite("multiband ceiling lookahead", self.lookahead_ms())
        if any(v <= 0 for v in self.lookahead_ms()):
            raise ValueError("ceiling_lookahead_ms must be > 0")
        if self.design_span_s is not None:
            self.design_span_s = float(self.design_span_s)
            if not (math.isfinite(self.design_span_s) and self.design_span_s > 0):
                raise ValueError("design_span_s must be positive")
        if self.mid_makeup_db is None:
            self.mid_makeup_db = [0.0] * nb
        if self.decoherence is None:
            self.decoherence = [0.0] * nb
        if len(self.mid_makeup_db) != nb or len(self.decoherence) != nb:
            raise ValueError(f"mid_makeup_db and decoherence need {nb} values")
        _finite("multiband", [*self.mid_makeup_db, *self.decoherence, self.side_makeup_db])
        if any(not 0.0 <= d <= 1.0 for d in self.decoherence):
            raise ValueError("decoherence must lie in [0, 1]")
        if not 0.0 <= self.side_makeup_db <= MAKEUP_MAX_DB:
            raise ValueError(f"side_makeup_db must lie in [0, {MAKEUP_MAX_DB}] dB")
        if any(abs(m) > MAKEUP_MAX_DB for m in self.mid_makeup_db):
            raise ValueError(f"mid_makeup_db must lie in [-{MAKEUP_MAX_DB}, {MAKEUP_MAX_DB}] dB")
        self.order = _order("multiband order", self.order)
        self.check_ceiling()

    def lookahead_ms(self) -> list[float]:
        """The band ceilings' lookahead per band, ms (a scalar broadcast)."""
        v = self.ceiling_lookahead_ms
        nb = len(self.crossovers_hz) + 1
        return [float(v)] * nb if np.ndim(v) == 0 else list(v)

    def side_makeup(self) -> list[float]:
        """Per-band S makeup in dB from the maintainer's rule."""
        return [self.side_makeup_db * (1.0 - d) if c >= HIGH_SIDE_MIN_HZ else 0.0
                for c, d in zip(band_centres(self.crossovers_hz), self.decoherence)]

    def makeup(self) -> list[list[float]]:
        """Applied makeup per band x [M, S], dB."""
        return [[float(m), float(s)] for m, s in zip(self.mid_makeup_db, self.side_makeup())]

    def check_ceiling(self) -> None:
        """Raise ValueError if the ceiling envelopes are > 0 dB anywhere, or if a band and
        channel with a ceiling has a makeup different from the one it was designed with.
        Called on construction and by multiband.apply (the dataclass is mutable)."""
        if self.ceiling_gr_db is None:
            return
        if any(np.any(e.values > 0.0) for p in self.ceiling_gr_db for e in p):
            raise ValueError("ceiling_gr_db must be <= 0 dB")
        if self.ceiling_db is None or self.ceiling_makeup_db is None:
            return
        now = self.makeup()
        for k, row in enumerate(self.ceiling_db):
            for c, ceil in enumerate(row):
                if ceil is not None and now[k][c] != self.ceiling_makeup_db[k][c]:
                    raise ValueError(
                        f"band {k} {'MS'[c]}: ceiling designed with makeup "
                        f"{self.ceiling_makeup_db[k][c]} dB, plan has {now[k][c]} dB; "
                        "rebuild with multiband.design_from")


def tone_nodes() -> list[float]:
    """Default tone nodes: 28 log-spaced nodes from 30 Hz to 16 kHz (about 1/3 octave)."""
    return [float(v) for v in np.geomspace(30.0, 16000.0, 28)]


@dataclass
class TonePlan:
    """gain_db[k] is the target-curve gain envelope at nodes_hz[k], dB (curve capped at +-3 dB)."""

    gain_db: list[Envelope]
    nodes_hz: list[float] = field(default_factory=tone_nodes)

    def __post_init__(self):
        _ascending("nodes_hz", self.nodes_hz)
        if len(self.gain_db) != len(self.nodes_hz):
            raise ValueError(f"gain_db needs {len(self.nodes_hz)} envelopes")


def _in_range(name: str, v: float, rng) -> float:
    v = float(v)
    if not (math.isfinite(v) and rng[0] - 1e-9 <= v <= rng[1] + 1e-9):
        raise ValueError(f"{name} must lie in [{rng[0]}, {rng[1]}], got {v!r}")
    return v


_EQ_GAIN = (-INPUTEQ_MAX_DB, INPUTEQ_MAX_DB)
# the high shelves' boost limits (maintainer, 2026-10-20: his Track G input EQ boosted the side
# top about twice the mid): mid (or linked) shelf up to +4.5 dB, side shelf up to +6 dB
INPUTEQ_HS_MID_MAX_DB = 4.5
INPUTEQ_HS_SIDE_MAX_DB = 6.0


@dataclass
class EqHighPass:
    """24 dB/oct high-pass: two biquad sections of Q q (0.707 = Butterworth sections)."""

    fc_hz: float = INPUTEQ_HPF_FC_HZ
    q: float = 0.707

    def __post_init__(self):
        self.fc_hz = _in_range("high-pass fc_hz", self.fc_hz, (1e-3, math.inf))
        self.q = _in_range("high-pass q", self.q, INPUTEQ_HPF_Q_RANGE)


@dataclass
class EqLowShelf:
    """12 dB/oct low shelf; gain_db/2 at the corner fc_hz."""

    gain_db: float
    fc_hz: float
    q: float = 0.707

    def __post_init__(self):
        self.gain_db = _in_range("low shelf gain_db", self.gain_db, _EQ_GAIN)
        self.fc_hz = _in_range("low shelf fc_hz", self.fc_hz, INPUTEQ_LOW_SHELF_FC_RANGE)
        self.q = _in_range("low shelf q", self.q, INPUTEQ_LOW_SHELF_Q_RANGE)


@dataclass
class EqBell:
    """Bell, cut only: gain_db <= 0 at the centre fc_hz."""

    gain_db: float
    fc_hz: float
    q: float = 1.4

    def __post_init__(self):
        self.gain_db = _in_range("bell gain_db", self.gain_db, (-INPUTEQ_MAX_DB, 0.0))
        self.fc_hz = _in_range("bell fc_hz", self.fc_hz, INPUTEQ_BELL_FC_RANGE)
        self.q = _in_range("bell q", self.q, INPUTEQ_BELL_Q_RANGE)


@dataclass
class EqBandShelf:
    """Band shelf, cut only (chunk 4): a flat-bottomed dip of gain_db <= 0 between the
    corners lo_hz and hi_hz, with 24 dB/oct (Linkwitz-Riley 4th-order) transitions on
    both sides; each corner is where half the dip (in linear amplitude) is reached."""

    gain_db: float
    lo_hz: float
    hi_hz: float

    def __post_init__(self):
        self.gain_db = _in_range("band shelf gain_db", self.gain_db, (-INPUTEQ_MAX_DB, 0.0))
        self.lo_hz = _in_range("band shelf lo_hz", self.lo_hz, INPUTEQ_BAND_SHELF_LO_RANGE)
        w = INPUTEQ_BAND_SHELF_WIDTH_OCT
        self.hi_hz = _in_range("band shelf hi_hz", self.hi_hz,
                               (self.lo_hz * 2.0 ** w[0], self.lo_hz * 2.0 ** w[1]))


@dataclass
class EqHighShelf:
    """6 dB/oct (first-order) high shelf; gain_db/2 at the corner fc_hz."""

    gain_db: float
    fc_hz: float = INPUTEQ_HIGH_SHELF_FC_HZ

    def __post_init__(self):
        self.gain_db = _in_range("high shelf gain_db", self.gain_db,
                                 (-INPUTEQ_MAX_DB, INPUTEQ_HS_SIDE_MAX_DB))
        self.fc_hz = _in_range("high shelf fc_hz", self.fc_hz, (1e-3, math.inf))


@dataclass
class InputEqPlan:
    """Static input EQ (stages/inputeq.py), sections in the maintainer's order, None = off.
    design: the planner's fit inputs (planner.refit_inputeq), None on a hand-built plan.
    The top end has two 6 dB/oct shelves (sprint 2026-10-21 T12): high_shelf, the
    presence shelf (planner: corner 1 to 6 kHz), and air_shelf (corner 10 to 16 kHz).
    high_shelf_side and air_shelf_side both None: the top shelves run on L and R alike
    (linked); either set: the top end is M/S, high_shelf and air_shelf on M only (at most
    INPUTEQ_HS_MID_MAX_DB, planner) and the _side shelves on S (at most
    INPUTEQ_HS_SIDE_MAX_DB)."""

    highpass: EqHighPass | None = None
    low_shelf: EqLowShelf | None = None
    bell: EqBell | None = None
    high_shelf: EqHighShelf | None = None
    phase: str = "zero"
    design: dict | None = None
    band_shelf: EqBandShelf | None = None
    high_shelf_side: EqHighShelf | None = None
    air_shelf: EqHighShelf | None = None
    air_shelf_side: EqHighShelf | None = None

    def __post_init__(self):
        if self.phase not in INPUTEQ_PHASES:
            raise ValueError(f"phase must be one of {INPUTEQ_PHASES}")

    @property
    def ms_top(self) -> bool:
        """The top end runs M/S (a side shelf is set)."""
        return self.high_shelf_side is not None or self.air_shelf_side is not None

    def frequencies(self) -> list[float]:
        out = [s.fc_hz for s in (self.highpass, self.low_shelf, self.bell, self.high_shelf,
                                 self.high_shelf_side, self.air_shelf, self.air_shelf_side)
               if s is not None]
        if self.band_shelf is not None:
            out += [self.band_shelf.lo_hz, self.band_shelf.hi_hz]
        return out


@dataclass
class SaturatePlan:
    """drive_db[k] = [M, S] drive of band k, dB >= 0; band 0 (sub) must be 0.
    harmonic[k] in [0, 1] is the harmonic-vs-linear blend (wet fraction);
    symmetry in [0, 1] adds the even-harmonic component (0 = odd only)."""

    drive_db: list[list[float]]
    harmonic: list[float]
    crossovers_hz: list[float] = field(default_factory=lambda: list(bands.DEFAULT_CROSSOVERS))
    order: int = 2
    symmetry: float = 0.3

    def __post_init__(self):
        _ascending("crossovers_hz", self.crossovers_hz)
        nb = len(self.crossovers_hz) + 1
        if len(self.drive_db) != nb or any(len(p) != 2 for p in self.drive_db):
            raise ValueError(f"drive_db must be {nb} bands x [M, S]")
        if len(self.harmonic) != nb:
            raise ValueError(f"harmonic needs {nb} values")
        _finite("saturate", [*self.harmonic, self.symmetry,
                             *[d for p in self.drive_db for d in p]])
        if any(not 0.0 <= d <= SATURATE_MAX_DRIVE_DB for p in self.drive_db for d in p):
            raise ValueError(f"drive_db must lie in [0, {SATURATE_MAX_DRIVE_DB}] dB")
        self.order = _order("saturate order", self.order)
        if any(d != 0.0 for d in self.drive_db[0]):
            raise ValueError("sub band drive must be 0 (spec 3.5)")
        if any(not 0.0 <= h <= 1.0 for h in self.harmonic):
            raise ValueError("harmonic must lie in [0, 1]")
        if not 0.0 <= self.symmetry <= 1.0:
            raise ValueError("symmetry must lie in [0, 1]")

    @classmethod
    def defaults(cls, width_drive_db: float = 3.0, harmonic: float = 0.5,
                 crossovers_hz=bands.DEFAULT_CROSSOVERS) -> "SaturatePlan":
        """Spec 3.5 defaults: sub and air drive 0, width drive on the high-mid S band only
        (the band below the top band), every other drive 0."""
        nb = len(crossovers_hz) + 1
        drive = [[0.0, 0.0] for _ in range(nb)]
        if nb >= 3:
            drive[nb - 2][1] = float(width_drive_db)
        return cls(drive, [float(harmonic)] * nb, list(crossovers_hz))


@dataclass
class GluePlan:
    """Single-band parallel compressor (spec 3.5 item 6, maintainer ruling 2026-10-17).

    gain_db: stereo-linked gain envelope of the wet path, dB <= +6 (1 ms; glue.design
    produces <= 0, positive values allow other topologies, audit 06 M3).
    makeup_db in [0, 6] dB on the wet path, explicit, default 0; mix in [0, 1] is the wet share
    (y = x + mix (wet - x)); curve in {"clean", "vintage"}. Design record:
    threshold_db (dBFS, sample peak of max(|L|, |R|)), ratio > 1, knee_db >= 0,
    attack_ms > 0, release_ms >= 0. threshold_db is required for the vintage
    curve, whose even-harmonic term engages above it. budget (chunk 3, glue.design_budget):
    the design record of the budget solve, {"mode": "serial" | "parallel", the
    targets, and once designed the wet and net peak GR and the warnings}; with it
    set the threshold (and in parallel mode the mix) are solved on the glue input."""

    gain_db: Envelope
    mix: float = 0.5
    makeup_db: float = 0.0
    curve: str = "clean"
    threshold_db: float | None = None
    ratio: float = 2.0
    knee_db: float = 6.0
    attack_ms: float = 30.0
    release_ms: float = 300.0
    budget: dict | None = None

    def __post_init__(self):
        if self.budget is not None and self.budget.get("mode") not in GLUE_MODES:
            raise ValueError(f"glue budget mode must be one of {GLUE_MODES}")
        _finite("glue", [self.mix, self.makeup_db, self.ratio, self.knee_db, self.attack_ms,
                         self.release_ms]
                + ([] if self.threshold_db is None else [self.threshold_db]))
        if np.any(self.gain_db.values > GLUE_MAX_GAIN_DB):
            raise ValueError(f"glue gain_db must be <= {GLUE_MAX_GAIN_DB} dB")
        if not 0.0 <= self.mix <= 1.0:
            raise ValueError("mix must lie in [0, 1]")
        if not 0.0 <= self.makeup_db <= GLUE_MAX_MAKEUP_DB:
            raise ValueError(f"makeup_db must lie in [0, {GLUE_MAX_MAKEUP_DB}] dB")
        if self.curve not in GLUE_CURVES:
            raise ValueError(f"curve must be one of {GLUE_CURVES}")
        if self.curve == "vintage" and self.threshold_db is None:
            raise ValueError("the vintage curve needs threshold_db")
        if not self.ratio > 1.0:
            raise ValueError("ratio must be > 1")
        if self.knee_db < 0 or self.attack_ms <= 0 or self.release_ms < 0:
            raise ValueError("knee_db >= 0, attack_ms > 0 and release_ms >= 0 required")


@dataclass
class KeyEqPlan:
    """Placeholder for the experimental key/note-aware EQ (not implemented)."""

    enabled: bool = False
    min_confidence: float = 0.75

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        c = float(self.min_confidence)
        if not (math.isfinite(c) and 0.0 <= c <= 1.0):
            raise ValueError("min_confidence must lie in [0, 1]")


@dataclass
class LimiterPlan:
    """Scalars (and two records). Clipper window per onset: [onset - clip_pre_ms,
    onset + clip_post_ms], total 8 to 12 ms, with clip_ramp_ms raised-cosine edges
    outside it (the kick-detective clipper: clip_depth_db > 0 with clip_onsets_s).

    Whole-signal clipper (chunk 3, docs/gain-staging.md 8, the default the planner
    writes): clip_threshold_db not None. Every span whose 8x peak exceeds the knee
    start is clipped by the soft-knee curve; the threshold is in dBFS of the 8x
    signal *before* input_gain_db (so the clipper's reduction does not move with
    the limiter drive), clip_knee_db the knee width below it (0 = hard).
    clip_target_db records the peak reduction the threshold was solved for (at
    most CLIP_MAX_GR_DB). The two clippers are exclusive.

    max_gr_db (chunk 3): the limiter's peak GR budget; when set, the planner and
    the solver clamp input_gain_db so the limiter's own peak GR on the clipper's
    output stays within it. requested_gain_db records the gain asked for before
    that clamp; budget holds the record (the clamp, the planned GR, a warning).
    target_rms_dbfs (chunk 4): the master's RMS on the maintainer's meter (meter.rms_meter:
    300 ms, AES17, loudest passage; 1 s plain dBFS before 2026-10-19) the
    requested gain is solved for (planner.design_limiter on the proxy limiter, the
    solver with its calibration); None keeps input_gain_db as given.

    Audible maxima (docs/dynamics-macros.md step 1, maintainer 2026-10-19):
    clip_max_gr_db caps the whole-signal curve's reduction per sample (the output
    is max(curve(x), |x| 10^(-cap/20)), so what passes the cap goes to the
    limiter); None leaves the curve uncapped. max_outlier_gr_db is the limiter's
    outlier cap (its largest peak GR); None reads max_gr_db +
    envelope.OUTLIER_ALLOWANCE_DB. speclim_allow_db switches on the frequency-dependent
    limiter ahead of the clipper (Plan.speclim): per band and M / S, peaks more than
    this above the band's typical loud hit are limited in that band, by at most
    speclim_max_cut_db (None: planner.SPECLIM_MAX_CUT_DB; the live profile 3 dB), with a
    lookahead of speclim_lookahead_ms (ms, > 0; a scalar or one value per band, extra
    bands taking the last; None: planner.SPECLIM_LOOKAHEAD_MS, 2 ms, the plans made
    before 2026-10-23; the planner writes the limiter timing preset's band_lookahead_ms).

    clip_split_hz (2026-10-20, the clipper was clipping sustained sub-bass): when set,
    the whole-signal clipper acts only on the signal above this frequency (a zero-phase
    second-order split, limiter.clip_hi_part): spans are found on that part, the curve
    is applied to it and the change added back, so the sub passes to the limiter
    untouched. The threshold is then solved on that part's per-hit output peak. None
    (old plans) keeps the full-band clipper."""

    ceiling_dbtp: float = -0.3
    input_gain_db: float = 0.0
    attack_ms: float = 1.5
    release_ms: float = 50.0
    smoother: str = "box"
    clip_depth_db: float = 0.0
    clip_onsets_s: list[float] = field(default_factory=list)
    clip_pre_ms: float = 2.0
    clip_post_ms: float = 8.0
    clip_ramp_ms: float = 1.0
    clip_threshold_db: float | None = None
    clip_knee_db: float = 1.0
    clip_target_db: float | None = None
    max_gr_db: float | None = None
    requested_gain_db: float | None = None
    budget: dict | None = None
    target_rms_dbfs: float | None = None
    clip_max_gr_db: float | None = None
    max_outlier_gr_db: float | None = None
    speclim_allow_db: float | None = None
    clip_split_hz: float | None = None
    speclim_max_cut_db: float | None = None
    speclim_lookahead_ms: float | list[float] | None = None

    def __post_init__(self):
        if self.speclim_lookahead_ms is not None:
            v = self.speclim_lookahead_ms
            self.speclim_lookahead_ms = float(v) if np.ndim(v) == 0 else [float(a) for a in v]
            vs = [self.speclim_lookahead_ms] if np.ndim(v) == 0 else self.speclim_lookahead_ms
            if not vs or not all(math.isfinite(a) and a > 0.0 for a in vs):
                raise ValueError("speclim_lookahead_ms must be finite and > 0")
        if self.speclim_allow_db is not None and not (
                math.isfinite(self.speclim_allow_db) and self.speclim_allow_db >= 0.0):
            raise ValueError("speclim_allow_db must be finite and >= 0")
        if self.speclim_max_cut_db is not None and not (
                math.isfinite(self.speclim_max_cut_db) and self.speclim_max_cut_db > 0.0):
            raise ValueError("speclim_max_cut_db must be finite and > 0")
        if self.clip_split_hz is not None:
            self.clip_split_hz = float(self.clip_split_hz)
            if not (math.isfinite(self.clip_split_hz) and self.clip_split_hz > 0.0):
                raise ValueError("clip_split_hz must be finite and > 0")
        opt = [v for v in (self.clip_threshold_db, self.clip_target_db, self.max_gr_db,
                           self.requested_gain_db, self.target_rms_dbfs, self.clip_max_gr_db,
                           self.max_outlier_gr_db) if v is not None]
        if self.clip_max_gr_db is not None and not 0.0 < self.clip_max_gr_db <= CLIP_SAMPLE_CAP_MAX_DB:
            raise ValueError(f"clip_max_gr_db must lie in (0, {CLIP_SAMPLE_CAP_MAX_DB}]")
        if self.max_outlier_gr_db is not None and self.max_outlier_gr_db < 0.0:
            raise ValueError("max_outlier_gr_db must be >= 0")
        _finite("limiter", [self.clip_knee_db, *opt])
        if not 0.0 <= self.clip_knee_db <= CLIP_KNEE_MAX_DB:
            raise ValueError(f"clip_knee_db must lie in [0, {CLIP_KNEE_MAX_DB}]")
        if self.clip_target_db is not None and not 0.0 <= self.clip_target_db <= CLIP_MAX_GR_DB:
            raise ValueError(f"clip_target_db must lie in [0, {CLIP_MAX_GR_DB}]")
        if self.clip_threshold_db is not None and self.clip_depth_db > 0.0 \
                and self.clip_onsets_s:
            raise ValueError("the whole-signal clipper (clip_threshold_db) and the "
                             "kick-detective clipper (clip_depth_db) are exclusive")
        if self.max_gr_db is not None and self.max_gr_db < 0.0:
            raise ValueError("max_gr_db must be >= 0")
        _finite("limiter", [self.ceiling_dbtp, self.input_gain_db, self.attack_ms,
                            self.release_ms, self.clip_depth_db, self.clip_pre_ms,
                            self.clip_post_ms, self.clip_ramp_ms, *self.clip_onsets_s])
        if self.smoother not in ("box", "qp"):
            raise ValueError("smoother must be 'box' or 'qp'")
        if not 0.0 <= self.clip_depth_db <= CLIP_MAX_DEPTH_DB:
            raise ValueError(f"clip_depth_db must lie in [0, {CLIP_MAX_DEPTH_DB}]")
        if not 8.0 <= self.clip_pre_ms + self.clip_post_ms <= 12.0:
            raise ValueError("clipper window (pre + post) must be 8 to 12 ms")
        if self.attack_ms <= 0 or self.release_ms < 0 or self.clip_ramp_ms <= 0:
            raise ValueError("attack and clip ramp must be > 0, release >= 0")
        if self.clip_pre_ms < 0 or self.clip_post_ms < 0:
            raise ValueError("clip_pre_ms and clip_post_ms must be >= 0")


HIT_READINGS = ("average", "max")


@dataclass
class HitsPlan:
    """The hits every budget solve reads its GR over (docs/gain-staging.md 3, maintainer
    2026-10-19): beats_s the beat times (the analysis grid, the user's --bpm when given),
    None for 400 ms frames (no usable grid); reading "average" (the median of the
    per-hit maximum GR over the loud hits; the compressors' GR read as a meter shows
    it, envelope.stage_gr) or "max" (the whole track as one hit, the chunk 3 reading).
    loud: the loud-hit mask frozen on the stage-1 input (one bool per hit), None until
    planner.freeze_hits sets it."""

    beats_s: list[float] | None = None
    source: str = "400 ms frames"
    reading: str = "average"
    loud: list[bool] | None = None

    def __post_init__(self):
        if self.reading not in HIT_READINGS:
            raise ValueError(f"reading must be one of {HIT_READINGS}")
        if self.beats_s is not None:
            self.beats_s = [float(b) for b in self.beats_s]
            _finite("hits", self.beats_s)
        if self.loud is not None:
            self.loud = [bool(v) for v in self.loud]


@dataclass
class Plan:
    fs: int
    preserve_mix: bool = False
    input: InputPlan | None = None
    inputeq: InputEqPlan | None = None
    notch: NotchPlan | None = None
    sidestrip: SidestripPlan | None = None
    multiband: MultibandPlan | None = None
    tone: TonePlan | None = None
    saturate: SaturatePlan | None = None
    glue: GluePlan | None = None
    # frequency-dependent limiting before the clipper (maintainer, 2026-10-19): a ceiling-only
    # multiband (multiband.design_speclim), designed at realise when
    # limiter.speclim_allow_db is set
    speclim: MultibandPlan | None = None
    keyeq: KeyEqPlan | None = None
    limiter: LimiterPlan | None = None
    output: OutputPlan | None = None
    hits: HitsPlan | None = None
    version: str = PLAN_VERSION
    environment: dict = field(default_factory=environment)

    def __post_init__(self):
        fs = float(self.fs)
        if not (math.isfinite(fs) and fs > 0 and fs == int(fs)):
            raise ValueError(f"fs must be a positive integer, got {self.fs!r}")
        self.fs = int(fs)
        nyq = self.fs / 2.0
        freqs: list[tuple[str, float]] = []
        if self.inputeq is not None:
            freqs += [("inputeq frequency", f) for f in self.inputeq.frequencies()]
        if self.notch is not None:
            freqs += [("notch f0", n.f0_hz) for n in self.notch.notches]
        if self.sidestrip is not None:
            freqs.append(("sidestrip fc", self.sidestrip.fc_hz))
        for name in ("multiband", "saturate", "speclim"):
            sec = getattr(self, name)
            if sec is not None:
                freqs += [(f"{name} crossover", f) for f in sec.crossovers_hz]
        if self.tone is not None:
            freqs += [("tone node", f) for f in self.tone.nodes_hz]
        if self.limiter is not None and self.limiter.clip_split_hz is not None:
            freqs.append(("limiter clip_split_hz", self.limiter.clip_split_hz))
        for what, f in freqs:
            if not f < nyq:
                raise ValueError(f"{what} {f} Hz is not below fs/2 = {nyq} Hz")

    # -------------------------------------------------------------- JSON
    def to_dict(self) -> dict:
        return _encode(self)

    @classmethod
    def from_dict(cls, d: dict, sidecar=None) -> "Plan":
        if d.get("version") != PLAN_VERSION:
            raise ValueError(f"plan version {d.get('version')!r}, expected {PLAN_VERSION!r}")
        env = lambda e: Envelope.from_dict(e, sidecar)  # noqa: E731
        return cls(
            fs=d["fs"],
            preserve_mix=bool(d.get("preserve_mix", False)),
            input=_opt(d, "input", lambda s: InputPlan(**s)),
            inputeq=_opt(d, "inputeq", _inputeq_from),
            notch=_opt(d, "notch", lambda s: _notch_from(s, env)),
            sidestrip=_opt(d, "sidestrip", lambda s: SidestripPlan(**s)),
            multiband=_opt(d, "multiband", lambda s: _multiband_from(s, env)),
            tone=_opt(d, "tone", lambda s: _tone_from(s, env)),
            saturate=_opt(d, "saturate", lambda s: SaturatePlan(**s)),
            glue=_opt(d, "glue", lambda s: _glue_from(s, env)),
            speclim=_opt(d, "speclim", lambda s: _multiband_from(s, env)),
            keyeq=_opt(d, "keyeq", lambda s: KeyEqPlan(**s)),
            limiter=_opt(d, "limiter", lambda s: LimiterPlan(**s)),
            output=_opt(d, "output", lambda s: OutputPlan(**s)),
            hits=_opt(d, "hits", lambda s: HitsPlan(**s)),
            environment=dict(d.get("environment") or {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text: str, sidecar=None) -> "Plan":
        return cls.from_dict(json.loads(text), sidecar)

    def save(self, path: str | Path, run_id: str | None = None) -> Path:
        """Write path (JSON, scalars + sidecar keys) and its .npz sidecar, both carrying
        run_id (a fresh one by default). Returns the npz path."""
        path = Path(path)
        npz = path.with_suffix(".npz")
        run_id = run_id or uuid.uuid4().hex
        arrays: dict[str, np.ndarray] = {}
        d = _encode(self, arrays, ())
        d["sidecar"] = npz.name
        d["run_id"] = run_id
        io.write_npz(npz, **arrays, **{RUN_ID_KEY: np.array(run_id)})
        io.write_text(path, json.dumps(d, indent=1, allow_nan=False) + "\n")
        return npz

    @classmethod
    def load(cls, path: str | Path) -> "Plan":
        path = Path(path)
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except ValueError as e:
            raise ValueError(f"{path} is not valid JSON: {e}") from e
        if not isinstance(d, dict):
            raise ValueError(f"{path} is not a plan")
        run_id = d.pop("run_id", None)
        sidecar = None
        if "sidecar" in d:
            name = d.pop("sidecar")
            if not (isinstance(name, str) and name.endswith(".npz") and Path(name).name == name):
                raise ValueError(f"{path}: sidecar {name!r} is not a .npz file name")
            with np.load(path.parent / name, allow_pickle=False) as z:
                sidecar = {k: z[k] for k in z.files}
            got = sidecar.pop(RUN_ID_KEY, None)
            if run_id is not None and (got is None or str(got) != run_id):
                raise ValueError(f"{path.name} and {name} are from different runs")
        try:
            return cls.from_dict(d, sidecar)
        except (KeyError, TypeError, AttributeError) as e:
            raise ValueError(f"{path} is malformed ({type(e).__name__}: {e})") from e


def _encode(obj: Any, arrays: dict | None = None, key: tuple = ()) -> Any:
    if isinstance(obj, Envelope):
        if arrays is None:
            return obj.to_dict()
        name = ".".join(str(k) for k in key)
        arrays[name] = obj.values.astype(np.float32)
        return {"rate_hz": obj.rate_hz, "npz": name}
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _encode(getattr(obj, k), arrays, key + (k,)) for k in obj.__dataclass_fields__}
    if isinstance(obj, (list, tuple)):
        return [_encode(v, arrays, key + (i,)) for i, v in enumerate(obj)]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _opt(d: dict, key: str, fn):
    v = d.get(key)
    return None if v is None else fn(v)


def _notch_from(s: dict, env) -> NotchPlan:
    notches = [Notch(n["f0_hz"], n["q"], env(n["gain_db"])) for n in s["notches"]]
    km = s.get("kick_mask")
    return NotchPlan(notches, None if km is None else env(km))


def _multiband_from(s: dict, env) -> MultibandPlan:
    s = dict(s)
    s["gr_db"] = [[env(e) for e in pair] for pair in s["gr_db"]]
    if s.get("ceiling_gr_db") is not None:
        s["ceiling_gr_db"] = [[env(e) for e in pair] for pair in s["ceiling_gr_db"]]
    if s.get("mode_makeup_db") is not None:
        s["mode_makeup_db"] = [[env(e) for e in pair] for pair in s["mode_makeup_db"]]
    return MultibandPlan(**s)


def _glue_from(s: dict, env) -> GluePlan:
    s = dict(s)
    s["gain_db"] = env(s["gain_db"])
    return GluePlan(**s)


def _tone_from(s: dict, env) -> TonePlan:
    s = dict(s)
    s["gain_db"] = [env(e) for e in s["gain_db"]]
    return TonePlan(**s)


def _inputeq_from(s: dict) -> InputEqPlan:
    s = dict(s)
    for k, cls in (("highpass", EqHighPass), ("low_shelf", EqLowShelf), ("bell", EqBell),
                   ("high_shelf", EqHighShelf), ("band_shelf", EqBandShelf),
                   ("high_shelf_side", EqHighShelf), ("air_shelf", EqHighShelf),
                   ("air_shelf_side", EqHighShelf)):
        s[k] = _opt(s, k, lambda v, cls=cls: cls(**v))
    return InputEqPlan(**s)
