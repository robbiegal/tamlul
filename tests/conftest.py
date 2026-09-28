from __future__ import annotations

import pathlib
import sys
import wave

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def data_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def synthetic_wav(tmp_path: pathlib.Path) -> pathlib.Path:
    """The first 14 s of the bundled CC-BY-4.0 FLEURS fixture (two clips + a gap): real speech, so
    Silero fires and the plan has two groups; the fake backend never looks at the samples."""
    sr = 16000
    src = ROOT / "src" / "tamlul" / "fixtures" / "fleurs_he.wav"
    with wave.open(str(src)) as w:
        sig = np.frombuffer(w.readframes(sr * 14), dtype="<i2").astype(np.float32) / 32768
    p = tmp_path / "synthetic.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(sig, -1, 1) * 32767).astype("<i2").tobytes())
    return p


@pytest.fixture(autouse=True)
def _utf8(monkeypatch):
    monkeypatch.setenv("PYTHONUTF8", "1")
    monkeypatch.delenv("TAMLUL_DATA", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_DATA", raising=False)
