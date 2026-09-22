"""
transcript_tools — direct caption grabbing
==========================================
list_available_transcripts: inspect which caption tracks a URL exposes.
fetch_transcript: grab one and emit a clean, agent-friendly transcript file.
The no-download, no-model path.
"""

from __future__ import annotations

import json

from mcp_agent_transcriber.errors import tool_error


def register(server, ctx) -> None:
    @server.tool()
    def list_available_transcripts(url: str) -> str:
        """
        List the caption tracks a video URL exposes: language codes plus
        whether each is a manual or auto-generated track. Uses yt-dlp
        metadata only — no download.
        """
        try:
            from mcp_agent_transcriber.download import describe_video

            result = describe_video(url)
            if result.get("status") != "ok":
                return json.dumps(result, indent=2, ensure_ascii=False)
            info = result["info"]
            return json.dumps({
                "status": "ok",
                "platform": info["extractor"],
                "title": info["title"],
                "subtitles": info["subtitles"],
                "note": "fetch_transcript(language=...) picks the best track for you.",
            }, indent=2, ensure_ascii=False)
        except Exception as e:
            return tool_error("list_available_transcripts", e)

    @server.tool()
    def fetch_transcript(
        url: str,
        language: str | None = None,
        prefer_auto: bool = True,
        want_vtt: bool = False,
    ) -> str:
        """
        Grab an agent-friendly transcript for a video URL without downloading
        the media. YouTube uses youtube-transcript-api (fast); every other
        platform falls back to the yt-dlp caption pass. Writes a .txt (plus
        .vtt on request) into the output dir and returns the cleaned text.
        """
        try:
            from mcp_agent_transcriber.transcript import grab_transcript

            result = grab_transcript(
                url,
                language=language,
                prefer_auto=prefer_auto,
                want_vtt=want_vtt,
                output_dir=ctx.cfg.output_dir,
                tmp_dir=ctx.cfg.downloads_dir,
            )
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as e:
            return tool_error("fetch_transcript", e)


__all__ = ["register"]
