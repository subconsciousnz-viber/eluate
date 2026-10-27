"""eluate: offline, non-causal auto-mastering engine (Python prototype).

Array convention used by every module: audio is shaped (n_samples, n_channels),
channel 0 = L (or M), channel 1 = R (or S). Storage dtype is float32 (io.py);
DSP modules compute in float64 and return float64 so that identities such as
band reconstruction hold to double precision. Callers cast back to float32 for
storage.
"""

from importlib import metadata as _metadata

# pyproject.toml's [project] version is the single source; the fallback serves a run from a
# source tree or a frozen bundle without the package metadata, and is kept equal to it
# (tests/test_environment.py).
_FALLBACK_VERSION = "0.1.0b0"
try:
    __version__ = _metadata.version("eluate")
except _metadata.PackageNotFoundError:
    __version__ = _FALLBACK_VERSION
