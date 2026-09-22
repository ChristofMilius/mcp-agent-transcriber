"""
mcp_agent_transcriber — MCP server wrapping the whisper_transcriber workflow
============================================================================
Turn any common tube-platform link (YouTube, Vimeo, Dailymotion, Twitch,
TikTok, ...) into an agent-friendly transcript.

Two paths, adapted from the standalone whisper_transcriber repo:
  * direct  — grab the platform's own captions (YouTube auto-captions first
              via youtube-transcript-api, other platforms via yt-dlp subs).
              Fast: no model, no download.
  * whisper — download the audio at the requested quality with yt-dlp and
              transcribe locally with OpenAI Whisper (turbo, CUDA/CPU
              fallback). Highest fidelity, ~1.6 GB model.

The Whisper checkpoint is never loaded at server startup — it loads on the
first transcribe call and stays cached for the process lifetime.
"""

from __future__ import annotations

__version__ = "0.1.0"


def main() -> int:
    """Console entry point (`mcp-agent-transcriber`). Dispatches to the CLI."""
    from mcp_agent_transcriber.cli import main as _cli_main

    return _cli_main()


__all__ = ["__version__", "main"]
