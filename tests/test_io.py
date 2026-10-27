"""Tests for io.py."""

import numpy as np
import pytest

from eluate import io


def test_roundtrip_float32_stereo(tmp_path):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((4800, 2)) * 0.1).astype(np.float32)
    p = tmp_path / "a.wav"
    io.write(p, x, 48000)
    y, fs = io.read(p)
    assert fs == 48000 and y.dtype == np.float32 and y.shape == (4800, 2)
    assert np.array_equal(x, y)


def test_mono_is_duplicated(tmp_path):
    x = np.linspace(-0.5, 0.5, 1000)
    p = tmp_path / "m.wav"
    io.write(p, x, 44100)
    y, _ = io.read(p)
    assert y.shape == (1000, 2) and np.array_equal(y[:, 0], y[:, 1])


def test_multichannel_rejected():
    with pytest.raises(ValueError):
        io.as_stereo(np.zeros((10, 3)))


def test_text_writer_is_lf_only(tmp_path):
    p = tmp_path / "t.md"
    io.write_text(p, "a\r\nb\rc\nd")
    raw = p.read_bytes()
    assert b"\r" not in raw and raw == b"a\nb\nc\nd"
    q = tmp_path / "j.json"
    io.write_json(q, {"k": [1, 2]})
    assert b"\r" not in q.read_bytes()


def test_read_rejects_non_finite(tmp_path):
    x = np.zeros((100, 2), dtype=np.float32)
    x[10, 0] = np.nan
    p = tmp_path / "nan.wav"
    io.write(p, x, 48000)
    with pytest.raises(ValueError):
        io.read(p)


def test_writes_are_atomic(tmp_path, monkeypatch):
    """Report 3.2: every writer replaces the file whole; a write that fails leaves the old
    file and no temporary file behind."""
    p = tmp_path / "a.wav"
    io.write(p, np.zeros((100, 2)), 48000, "PCM_24")
    io.write_json(tmp_path / "j.json", {"k": 1})
    io.write_npz(tmp_path / "z.npz", t=np.arange(3.0))
    with np.load(tmp_path / "z.npz") as z:
        assert np.array_equal(z["t"], np.arange(3.0))
    assert sorted(f.name for f in tmp_path.iterdir()) == ["a.wav", "j.json", "z.npz"]
    old = p.read_bytes()

    def broken(path, *a, **k):
        with open(path, "wb") as f:
            f.write(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(io.sf, "write", broken)
    with pytest.raises(OSError):
        io.write(p, np.ones((100, 2)) * 0.5, 48000)
    assert p.read_bytes() == old
    with pytest.raises(OSError):
        io.write(tmp_path / "new" / "b.wav", np.ones((100, 2)) * 0.5, 48000)
    assert sorted(f.name for f in tmp_path.rglob("*")) == ["a.wav", "j.json", "new", "z.npz"]
