# PyInstaller spec for the engine the desktop shell runs (desktop/README.md).
# Onedir: the scientific stack would otherwise self-extract on every launch.
import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, "..", ".."))

datas, binaries, hidden = [], [], []
for pkg in ("numpy", "scipy", "librosa", "numba", "llvmlite", "sklearn", "soundfile",
            "_soundfile_data", "soxr", "lazy_loader", "joblib", "msgpack", "pooch",
            "audioread", "decorator"):
    try:
        d, b, h = collect_all(pkg)
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
