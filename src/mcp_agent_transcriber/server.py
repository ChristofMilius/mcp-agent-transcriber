"""
server.py — MCP server assembly
================================
Wires Config → WhisperEngine into an AppContext, builds an MCPServer, and
registers the tool surface. Uses mcp 2.x (`mcp.server.mcpserver.MCPServer`).
"""

from __future__ import annotations

import logging
from typing import Literal

from mcp.server.mcpserver import MCPServer

from mcp_agent_transcriber import __version__
from mcp_agent_transcriber.config import Config
from mcp_agent_transcriber.context import AppContext
from mcp_agent_transcriber.logging_setup import setup_logging
from mcp_agent_transcriber.tool_surface import register_all
from mcp_agent_transcriber.whisper_engine import WhisperEngine

logger = logging.getLogger(__name__)

SERVER_NAME = "mcp-agent-transcriber"

Transport = Literal["stdio", "sse", "streamable-http"]

INSTRUCTIONS = """
Video transcription for agents. Turn any common tube-platform link (YouTube,
Vimeo, Dailymotion, Twitch, TikTok, ...) into an agent-friendly transcript.

Two paths, kept deliberately explicit:

  * direct — grab the platform's own captions. Fast: no download, no model.
  * whisper — download the audio at the requested quality (yt-dlp) and
    transcribe locally with OpenAI Whisper (turbo, CUDA/CPU). Highest
    fidelity; the ~1.6 GB checkpoint loads on first use and stays cached.

Workflow:
  1. supported_platforms() / video_info(url) → platform, caption tracks,
     and audio availability for a URL.
  2. fetch_transcript(url) → direct transcript when the platform exposes
     one. No model needed.
  3. download_audio(url, quality=...) → pull the audio track on demand.
  4. transcribe_file(path) / transcribe_video(url) → Whisper the audio.
     transcribe_video(method="auto") does the whole route for you: captions
     first, Whisper as fallback when the platform has none.
""".strip()


def build_context() -> AppContext:
    """Construct the full application object graph."""
    cfg = Config()
    setup_logging(str(cfg.logs_dir))
    return AppContext(cfg=cfg, engine=WhisperEngine(cfg.model_name))


def create_server(ctx: AppContext | None = None) -> MCPServer:
    """Build an MCPServer with the full tool surface registered."""
    if ctx is None:
        ctx = build_context()

    server = MCPServer(name=SERVER_NAME, instructions=INSTRUCTIONS)
    register_all(server, ctx)

    logger.info(
        "[server] %s v%s ready (output=%s, downloads=%s, model=%s)",
        SERVER_NAME, __version__, ctx.cfg.output_dir, ctx.cfg.downloads_dir, ctx.cfg.model_name,
    )
    return server


def run(transport: Transport = "stdio", host: str = "127.0.0.1", port: int = 8000) -> None:
    """Build and run the server.

    transport: "stdio" (default), "sse", or "streamable-http".

    MCPServer.run() is overloaded with one signature per transport literal, and
    host/port only exist on the two HTTP ones. Each branch therefore passes its
    own literal rather than forwarding `transport`: a union of the two HTTP
    literals matches no single overload.
    """
    server = create_server()
    if transport == "stdio":
        server.run(transport="stdio")
    elif transport == "sse":
        server.run(transport="sse", host=host, port=port)
    else:
        server.run(transport="streamable-http", host=host, port=port)


__all__ = ["SERVER_NAME", "Transport", "build_context", "create_server", "run"]
