"""
transcribe_tools — the Whisper path
===================================
transcriber_status: model/device/ffmpeg readiness (no model load).
transcribe_file: Whisper a local media file — the original standalone repo's
core call, adapted.
transcribe_video: download then transcribe, or run the whole auto route
(captions first, Whisper fallback) in one call.
"""

from __future__ import annotations

import json
from pathlib import Path

from mcp_agent_transcriber.cleaning import render_vtt
from mcp_agent_transcriber.errors import tool_error
from mcp_agent_transcriber.names import transcript_names


def _emit_transcript(title: str, out: dict, output_dir: Path, want_vtt: bool) -> dict:
    """Write text/vtt artifacts for a whisper result and build the JSON payload."""
    txt_name, vtt_name = transcript_names(title, out.get("language") or "auto")
    (output_dir / txt_name).write_text(out["text"] + "\n", encoding="utf-8")
    result = {
        "status": "ok",
        "source": "whisper",
        "title": title,
        "language": out.get("language"),
        "text": out["text"],
        "chars": len(out["text"]),
        "device": out.get("device"),
        "fp16": out.get("fp16"),
        "output_dir": str(output_dir),
        "filename": txt_name,
    }
    if want_vtt:
        (output_dir / vtt_name).write_text(render_vtt(out["segments"]) + "\n", encoding="utf-8")
        result["vtt_filename"] = vtt_name
    return result


def register(server, ctx) -> None:
    engine = ctx.engine
    output_dir = ctx.cfg.output_dir

    @server.tool()
    def transcriber_status() -> str:
        """
        Report Whisper runtime readiness: model, device (cuda/cpu), whether
        the checkpoint is already cached/loaded, and ffmpeg availability.
        Does NOT load the model.
        """
        try:
            return json.dumps(engine.status(), indent=2, ensure_ascii=False)
        except Exception as e:
            return tool_error("transcriber_status", e)

    @server.tool()
    def transcribe_file(
        path: str,
        language: str | None = None,
        want_vtt: bool = True,
        return_segments: bool = True,
    ) -> str:
        """
        Transcribe a local video/audio file with Whisper (turbo). The model
        loads on first use and stays cached. Writes a .txt (plus .vtt on
        request) into the output dir and returns the transcript.

        language: ISO code to force (e.g. "en"); None auto-detects.
        """
        try:
            if not Path(path).is_file():
                return json.dumps(
                    {"status": "failed", "reason": "file_not_found", "path": path}, indent=2
                )
            output_dir.mkdir(parents=True, exist_ok=True)
            out = engine.transcribe(path, language=language or None)
            title = Path(path).stem
            result = _emit_transcript(title, out, output_dir, want_vtt)
            if return_segments:
                result["segments"] = out["segments"]
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as e:
            return tool_error("transcribe_file", e)

    @server.tool()
    def transcribe_video(
        url: str,
        method: str = "auto",
        language: str | None = None,
        quality: str = "best",
        prefer_auto: bool = True,
        keep_audio: bool = False,
        want_vtt: bool = True,
        return_segments: bool = True,
    ) -> str:
        """
        One-call video → transcript. method: "auto" (default) grabs the
        platform's captions first and falls back to Whisper only when none
        exist; "captions" uses captions only; "whisper" downloads the audio
        (quality: best/high/standard/low) and transcribes locally.
        keep_audio=true retains the downloaded audio file in the downloads
        dir; otherwise it is deleted after transcription.
        """
        try:
            from mcp_agent_transcriber.pipeline import transcribe_video as _run

            result = _run(
                url,
                method=method,
                language=language,
                quality=quality,
                prefer_auto=prefer_auto,
                keep_audio=keep_audio,
                want_vtt=want_vtt,
                return_segments=return_segments,
                output_dir=output_dir,
                downloads_dir=ctx.cfg.downloads_dir,
                engine=engine,
            )
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as e:
            return tool_error("transcribe_video", e)


__all__ = ["register"]
