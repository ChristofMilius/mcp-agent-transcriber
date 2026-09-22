"""
download_tools — audio download
================================
download_audio: pull a video's audio track at the requested quality into the
downloads dir via yt-dlp. The foundation for the whisper path — Whisper gets
the best available input, whatever the platform sends.
"""

from __future__ import annotations

import json

from mcp_agent_transcriber.errors import tool_error


def register(server, ctx) -> None:
    @server.tool()
    def download_audio(
        url: str,
        quality: str = "best",
        output_format: str = "orig",
    ) -> str:
        """
        Download a video's audio track into the downloads dir via yt-dlp.

        quality: "best" (default, highest bitrate), "high" (>=128 kbps floor),
        "standard" (<=128 kbps, smaller), or "low". output_format: "orig"
        keeps the platform's container; one of mp3/wav/m4a/opus/flac/aac
        re-encodes with ffmpeg as a postprocess.
        """
        try:
            from mcp_agent_transcriber.download import download_audio as _download

            result = _download(
                url,
                ctx.cfg.downloads_dir,
                quality=quality,
                output_format=output_format,
            )
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as e:
            return tool_error("download_audio", e)


__all__ = ["register"]
