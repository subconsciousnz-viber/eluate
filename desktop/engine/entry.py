"""Frozen entry point of the engine (eluate-engine.exe). The shell sets NUMBA_CACHE_DIR;
the fallback keeps librosa's cached numba functions working when run by hand, since a
frozen app has no writable source tree for numba's default cache locator."""
import os
import sys
import tempfile

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "eluate-numba"))

from eluate.app_engine import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
