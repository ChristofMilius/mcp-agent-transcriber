"""tests/test_whisper_engine.py — lazy load + caching, without touching torch."""
from __future__ import annotations

import pytest

from mcp_agent_transcriber.whisper_engine import WhisperEngine


class FakeModel:
    def transcribe(self, path, language=None, verbose=False, fp16=False):
        return {
            "text": "hello from whisper",
            "language": language or "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello from whisper"}],
        }


class FakeWhisper:
    def __init__(self):
        self.loaded = []

    def load_model(self, name, device):
        self.loaded.append((name, device))
        return FakeModel()


class _FakeTorchCuda:
    @staticmethod
    def is_available():
        return False


class FakeTorch:
    cuda = _FakeTorchCuda


@pytest.fixture
def fake_runtime(monkeypatch):
    """Whisper + torch stubs, wheels present, CUDA disabled."""
    whisper_mod = FakeWhisper()
    monkeypatch.setattr("mcp_agent_transcriber.whisper_engine._has_wheel", lambda name: True)
    monkeypatch.setattr("mcp_agent_transcriber.whisper_engine._load_runtime", lambda: (whisper_mod, FakeTorch))
    return whisper_mod


class TestEngine:
    def test_status_reports_installed_wheels(self, monkeypatch):
        monkeypatch.setattr("mcp_agent_transcriber.whisper_engine._has_wheel", lambda name: True)
        status = WhisperEngine("turbo").status()
        assert status["whisper_installed"] is True
        assert status["torch_installed"] is True
        assert status["model_loaded"] is False

    def test_load_raises_when_wheels_missing(self, monkeypatch):
        monkeypatch.setattr("mcp_agent_transcriber.whisper_engine._has_wheel", lambda name: False)
        with pytest.raises(RuntimeError, match="uv sync"):
            WhisperEngine("turbo").load()

    def test_load_caches_model(self, fake_runtime):
        engine = WhisperEngine("turbo")
        model = engine.load()
        assert engine.is_loaded()
        assert engine.device == "cpu"
        assert fake_runtime.loaded == [("turbo", "cpu")]
        # second load reuses the cache — no second whisper.load_model
        assert engine.load() is model
        assert fake_runtime.loaded == [("turbo", "cpu")]

    def test_transcribe_shapes_result(self, fake_runtime, tmp_path):
        media = tmp_path / "clip.m4a"
        media.write_bytes(b"not really audio")
        engine = WhisperEngine("turbo")
        out = engine.transcribe(str(media), language="de")
        assert out["text"] == "hello from whisper"
        assert out["language"] == "de"
        assert out["device"] == "cpu"
        assert out["fp16"] is False
        assert out["segments"][0]["text"] == "hello from whisper"
