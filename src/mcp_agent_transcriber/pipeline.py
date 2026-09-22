"""
pipeline.py — end-to-end video → transcript orchestration
==========================================================
transcribe_video() is the one-call convenience: while the caption and
whisper paths stay individually callable, this routes a URL through either
(or both) depending on `method`:

  * "captions" — direct caption grab only (no model, no download);
  * "whisper"  — download audio at the requested quality, then transcribe
                  locally with Whisper (highest fidelity);
  * "auto"     — captions first; only when the platform exposes none does it
                  fall back to the whisper path, reporting why.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from mcp_agent_transcriber.cleaning import render_vtt
from mcp_agent_transcriber.download import download_audio
from mcp_agent_transcriber.names import transcript_names

if TYPE_CHECKING:
    from mcp_agent_transcriber.whisper_engine import WhisperEngine

logger = logging.getLogger(__name__)

_METHODS = ("auto", "captions", "whisper")


def _whisper_route(
    url: str,
    language: str | None,
    quality: str,
    keep_audio: bool,
    want_vtt: bool,
    return_segments: bool,
    output_dir: Path,
    downloads_dir: Path,
    engine: WhisperEngine,
) -> dict:
    """Download the audio and transcribe it with Whisper."""
    dl = download_audio(url, downloads_dir, quality=quality, output_format="orig")
    if dl.get("status") != "ok":
        return dl

    audio_path = downloads_dir / dl["filename"]
    if not audio_path.is_file():
        return {
            "status": "failed",
            "reason": "audio_missing",
            "note": "the downloaded audio file was not found on disk.",
        }

    out = engine.transcribe(str(audio_path), language=language or None)
    title = dl.get("title") or "video"
    txt_name, vtt_name = transcript_names(title, out.get("language") or language or "auto")
    (output_dir / txt_name).write_text(out["text"] + "\n", encoding="utf-8")

    result = {
        "status": "ok",
        "source": "whisper",
        "platform": dl.get("platform"),
        "title": title,
        "language": out.get("language"),
        "text": out["text"],
        "chars": len(out["text"]),
        "device": out.get("device"),
        "fp16": out.get("fp16"),
        "output_dir": str(output_dir),
        "filename": txt_name,
        "audio": {
            "filename": dl["filename"],
            "ext": dl["ext"],
            "size_bytes": dl["size_bytes"],
            "quality_preset": dl["quality_preset"],
            "format_selector": dl["format_selector"],
        },
    }
    if want_vtt:
        (output_dir / vtt_name).write_text(render_vtt(out["segments"]) + "\n", encoding="utf-8")
        result["vtt_filename"] = vtt_name
    if return_segments:
        result["segments"] = out["segments"]

    if not keep_audio:
        audio_path.unlink(missing_ok=True)
        result["audio"]["saved"] = False
    else:
        result["audio"]["saved"] = True
    return result


def transcribe_video(
    url: str,
    method: str = "auto",
    language: str | None = None,
    quality: str = "best",
    prefer_auto: bool = True,
    keep_audio: bool = False,
    want_vtt: bool = True,
    return_segments: bool = True,
    output_dir: Path | None = None,
    downloads_dir: Path | None = None,
    engine: WhisperEngine | None = None,
) -> dict:
    """Route a video URL to captions and/or Whisper and return the transcript."""
    output_dir = Path(output_dir)
    downloads_dir = Path(downloads_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if method not in _METHODS:
        return {
            "status": "failed",
            "reason": "bad_method",
            "method": method,
            "allowed": list(_METHODS),
        }

    if method == "captions":
        from mcp_agent_transcriber import transcript

        return transcript.grab_transcript(
            url,
            language=language,
            prefer_auto=prefer_auto,
            want_vtt=want_vtt,
            output_dir=output_dir,
            tmp_dir=downloads_dir,
        )

    if method == "whisper":
        return _whisper_route(
            url, language, quality, keep_audio, want_vtt, return_segments,
            output_dir, downloads_dir, engine,
        )

    # auto — captions first, whisper as the fallback.
    from mcp_agent_transcriber import transcript

    caption = transcript.grab_transcript(
        url,
        language=language,
        prefer_auto=prefer_auto,
        want_vtt=want_vtt,
        output_dir=output_dir,
        tmp_dir=downloads_dir,
    )
    if caption.get("status") == "ok":
        return caption

    logger.info("[pipeline] captions unavailable (%s) — falling back to whisper", caption.get("status"))
    result = _whisper_route(
        url, language, quality, keep_audio, want_vtt, return_segments,
        output_dir, downloads_dir, engine,
    )
    result["captions_failed_reason"] = caption.get("status")
    return result


__all__ = ["transcribe_video"]
