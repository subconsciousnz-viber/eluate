# PyInstaller spec for the engine the desktop shell runs (desktop/README.md).
# Onedir: the scientific stack would otherwise self-extract on every launch.
import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, "..", ".."))

# Test suites are not collected (audit 2026-11-10 desktop #12): no `tests` / `conftest`
# submodules and no files under a tests/ folder (scipy's alone is ~17 MB with its data).
# `testing` packages are dropped too, except the two the engine imports at run time
# (checked from source with sklearn blocked: scipy._lib._array_api imports numpy.testing,
# scipy's array_api_extra its own testing module). The analysis still follows every real
# import, so the filter only stops the forced collection.
RUNTIME_TESTING = ("numpy.testing", "scipy._external.array_api_extra.testing")


def not_a_test(module):
    parts = module.split(".")[1:]
    if any(p in ("tests", "conftest") for p in parts):
        return False
    return "testing" not in parts or module.startswith(RUNTIME_TESTING)


NO_TEST_DATA = ["**/tests", "**/tests/**", "**/conftest.py"]

# scikit-learn is not collected and is excluded below: librosa imports it only lazily,
# inside librosa.decompose and librosa.segment functions the engine never calls (beat_track,
# onset_strength, pyin, mfcc, chroma_stft and chroma_cqt do not). Checked from source:
# eluate.app_engine, cli, planner, solver, uidata and those librosa functions import with
# sys.modules["sklearn"] = None. Re-check with `node scripts/smoke.mjs` on a built engine.
# Expected saving with the tests above: roughly 40-55 MB of ~370 MB (sklearn 27 MB; tests
# of numpy 6.7, scipy 16.6, numba 4.7 MB, measured in site-packages, less in the PYZ).
datas, binaries, hidden = [], [], []
for pkg in ("numpy", "scipy", "librosa", "numba", "llvmlite", "soundfile",
            "_soundfile_data", "soxr", "lazy_loader", "joblib", "msgpack", "pooch",
            "audioread", "decorator"):
    try:
        d, b, h = collect_all(pkg, filter_submodules=not_a_test, exclude_datas=NO_TEST_DATA)
    except TypeError:
        raise  # an older PyInstaller without these arguments: fail, never collect nothing
    except Exception:
        continue
    datas += d
    binaries += b
    hidden += h
hidden += collect_submodules("eluate")

a = Analysis(
    [os.path.join(SPECPATH, "entry.py")],
    pathex=[os.path.join(ROOT, "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    excludes=["matplotlib", "tkinter", "pytest", "IPython", "PyQt5", "PyQt6", "PySide6",
              "torch", "tensorflow", "demucs", "pyloudnorm",
              "sklearn",  # see above: librosa's lazy, unused import
              # Pulled in only by test, build-time or plotting modules that collect_all sweeps
              # up (traced in build-engine/xref-eluate-engine.html); never imported by the
              # engine. PIL: sklearn.datasets image loaders and pygments' image formatter.
              # cycler: librosa.display (matplotlib is excluded). pytest's own deps:
              # joblib.testing, sklearn.conftest, numpy tests. setuptools/distutils:
              # numba.pycc (ahead-of-time compiler), numba tests and cffi's compile API (the
              # engine's cffi use is soundfile's ABI-mode dlopen). pygments: numba's
              # NUMBA_HIGHLIGHT_DUMPS debug output (guarded import); colorama: numba's
              # optional coloured errors (falls back to plain). PyInstaller and its deps:
              # numpy's PyInstaller hook module.
              "PIL", "cycler",
              "_pytest", "pluggy", "py", "iniconfig",
              "setuptools", "_distutils_hack", "pkg_resources",
              "pygments", "colorama",
              "PyInstaller", "altgraph", "pefile", "ordlookup", "win32ctypes"],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="eluate-engine",
    console=True,          # the shell reads stdout and starts it without a window
    upx=False,
    strip=False,
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, upx=False, strip=False,
               name="eluate-engine")
