"""tests/test_config.py — env-driven path resolution."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def patched_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("TRANSCRIBER_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("TRANSCRIBER_DOWNLOADS_DIR", str(tmp_path / "sounds"))
    monkeypatch.setenv("TRANSCRIBER_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("TRANSCRIBER_MODEL", "turbo")
    monkeypatch.setenv("TRANSCRIBER_LANGUAGE", "de")
    return tmp_path


class TestConfig:
    def test_absolute_paths_resolve(self, patched_env):
        from mcp_agent_transcriber.config import Config

        cfg = Config()
        assert cfg.output_dir == patched_env / "out"
        assert cfg.downloads_dir == patched_env / "sounds"
        assert cfg.logs_dir == patched_env / "logs"
        assert cfg.model_name == "turbo"
        assert cfg.default_language == "de"

    def test_relative_paths_resolve_to_project_root(self, monkeypatch):
        from mcp_agent_transcriber import config as config_mod
        from mcp_agent_transcriber.config import Config

        monkeypatch.delenv("TRANSCRIBER_OUTPUT_DIR", raising=False)
        monkeypatch.delenv("TRANSCRIBER_DOWNLOADS_DIR", raising=False)
        monkeypatch.delenv("TRANSCRIBER_LOGS_DIR", raising=False)
        monkeypatch.delenv("TRANSCRIBER_MODEL", raising=False)
        monkeypatch.delenv("TRANSCRIBER_LANGUAGE", raising=False)

        base = config_mod._BASE
        cfg = Config()
        assert cfg.output_dir == (base / "transcript_output").resolve()
        assert cfg.downloads_dir == (base / "transcript_output" / "downloads").resolve()
        assert cfg.logs_dir == (base / "logs").resolve()
        assert cfg.model_name == "turbo"
        assert cfg.default_language == "en"

    def test_defaults_are_absolute_path_objects(self, monkeypatch):
        monkeypatch.delenv("TRANSCRIBER_OUTPUT_DIR", raising=False)
        from mcp_agent_transcriber.config import Config

        cfg = Config()
        assert cfg.output_dir.is_absolute()
        assert cfg.downloads_dir.is_absolute()
