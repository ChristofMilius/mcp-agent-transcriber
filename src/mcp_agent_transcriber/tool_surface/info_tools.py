"""
info_tools — video/platform discovery
=====================================
video_info: one yt-dlp metadata pass over a URL — the entry point agents use
to decide between the caption path (fast) and the whisper path (fidelity).
supported_platforms: which common tube platforms the installed yt-dlp covers.
"""

from __future__ import annotations

import json

from mcp_agent_transcriber.errors import tool_error
from mcp_agent_transcriber.platforms import supported_platforms as _platform_report


def register(server, ctx) -> None:
    @server.tool()
    def video_info(url: str) -> str:
        """
        Resolve a video URL with one yt-dlp metadata pass: platform, title,
        duration, uploader, available caption tracks, and the audio formats
        found. Returns JSON; known failures (unsupported URL, private video,
        sign-in required, network) come back as a status:failed payload.
        """
        try:
            from mcp_agent_transcriber.download import describe_video

            return json.dumps(describe_video(url), indent=2, ensure_ascii=False)
        except Exception as e:
            return tool_error("video_info", e)

    @server.tool()
    def supported_platforms() -> str:
        """
        List the common tube platforms (YouTube, Vimeo, Dailymotion, Twitch,
        TikTok, ...) and whether the installed yt-dlp ships an extractor for
        each. Use before evaluating an unfamiliar host.
        """
        try:
            return json.dumps(_platform_report(), indent=2, ensure_ascii=False)
        except Exception as e:
            return tool_error("supported_platforms", e)


__all__ = ["register"]
