import pytest

from eluate import parallel


@pytest.fixture(autouse=True)
def _engine_threads():
    """A test that sets the engine's thread count (directly or through cli.do_master)
    does not change it for the next."""
    n = parallel.threads()
    yield
    parallel.set_threads(n)
