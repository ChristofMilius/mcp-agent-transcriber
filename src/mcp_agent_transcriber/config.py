"""
config.py — All configuration in one place
==========================================
Loads from environment variables. No secrets involved.

Path resolution:
  - All path fields are resolved to absolute paths at construction time.
  - Relative values (TRANSCRIBER_OUTPUT_DIR, ...) are resolved relative to
    the project root (the directory containing the package), NOT the process
    CWD. Agent harnesses set an unpredictable CWD when spawning MCP server
    subprocesses.
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root — the directory containing the package (src/mcp_agent_transcriber).
# All relative path env vars are resolved against this, never against CWD.
_BASE = Path(__file__).resolve().parent.parent.parent


def _resolve_path(raw: str) -> Path:
    """
    Resolve a path string to an absolute path.

    If the value from the environment is already absolute, return it
    normalized. If relative, resolve relative to the project root (_BASE),
    not the process CWD.
    """
    p = Path(raw)
    if p.is_absolute():
        return p
    return (_BASE / p).resolve()


class Config:
    def __init__(self) -> None:
        self.output_dir: Path = _resolve_path(
            os.getenv("TRANSCRIBER_OUTPUT_DIR", "transcript_output")
        )
        self.downloads_dir: Path = _resolve_path(
            os.getenv("TRANSCRIBER_DOWNLOADS_DIR", str(self.output_dir / "downloads"))
        )
        self.logs_dir: Path = _resolve_path(
            os.getenv("TRANSCRIBER_LOGS_DIR", "logs")
        )
        self.model_name: str = os.getenv("TRANSCRIBER_MODEL", "turbo")
        self.default_language: str = os.getenv("TRANSCRIBER_LANGUAGE", "en")

    def __repr__(self) -> str:
        return (
            f"Config("
            f"output_dir={self.output_dir!r}, "
            f"downloads_dir={self.downloads_dir!r}, "
            f"logs_dir={self.logs_dir!r}, "
            f"model_name={self.model_name!r}, "
            f"default_language={self.default_language!r}, "
            f")"
        )


__all__ = ["Config"]
