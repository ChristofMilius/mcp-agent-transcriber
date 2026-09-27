"""
cli.py — command-line interface
================================
Subcommands:
  serve    Run the MCP server (default). --http switches to streamable-http.
  doctor   Offline diagnostics (paths, model/engine state, ffmpeg, yt-dlp).

The CLI mirrors the mail / docparser siblings' shape. It is the operator
console, not the model-facing surface; output only reports paths/state,
no secrets.
"""

from __future__ import annotations

import argparse
import sys

from mcp_agent_transcriber import __version__


def _cmd_serve(args) -> int:
    from mcp_agent_transcriber.server import Transport, run

    transport: Transport = "streamable-http" if args.http else "stdio"
    try:
        run(transport=transport, host=args.host, port=args.port)
    except KeyboardInterrupt:
        pass
    return 0


def _cmd_doctor(args) -> int:
    from mcp_agent_transcriber.config import Config
    from mcp_agent_transcriber.platforms import COMMON_PLATFORMS
    from mcp_agent_transcriber.transcript import youtube_id
    from mcp_agent_transcriber.whisper_engine import WhisperEngine

    print(f"mcp-agent-transcriber {__version__} — doctor\n")

    try:
        cfg = Config()
    except Exception as e:
        print(f"[config] {type(e).__name__}: {e}")
        return 1

    print(f"  output   : {cfg.output_dir}")
    print(f"  downloads: {cfg.downloads_dir}")
    print(f"  logs     : {cfg.logs_dir}")
    print(f"  model    : {cfg.model_name}")

    status = WhisperEngine(cfg.model_name).status()
    print("\n  Engine status:")
    for k, v in status.items():
        print(f"    {k:16s} {v}")

    try:
        import importlib.metadata

        version = importlib.metadata.version("yt-dlp")
    except Exception:
        version = "?"
    print(f"\n  yt-dlp        : {version}")
    print(f"  known platforms: {len(COMMON_PLATFORMS)} curated common tube platforms")
    # youtube_id sanity (no network)
    sample = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    print(f"  youtube URL   : {sample}")
    print(f"    video id    : {youtube_id(sample)}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-agent-transcriber",
        description="MCP server for video transcription — direct captions or Whisper, from any common tube platform.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="Run the MCP server (default).")
    p_serve.add_argument("--http", action="store_true", help="Use streamable-http transport.")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=_cmd_serve)

    p_doc = sub.add_parser("doctor", help="Diagnose configuration and engine state.")
    p_doc.set_defaults(func=_cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        # Default action: serve over stdio (what MCP harnesses expect).
        args.http = False
        args.host = "127.0.0.1"
        args.port = 8000
        return _cmd_serve(args)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["build_parser", "main"]
