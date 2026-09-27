"""
download.py — resolve video metadata and download audio via yt-dlp
==================================================================
yt-dlp is the multi-platform backbone: it identifies the extractor for the
URL (YouTube, Vimeo, Dailymotion, Twitch, ...), lists caption tracks, and
pulls the audio track at the requested quality.

Failure policy:
  * Known, reproducible conditions (unsupported URL, private video,
    authentication required, missing format, network) are decoded into
    static, path-free JSON payloads — exception *messages* are never
    returned because they can leak local output paths.
  * Unexpected exceptions bubble up to the tool boundary and become
    sanitized tool_error() responses.

`describe_video` runs extract_info(download=False) — one metadata pass the
info tools and the caption path share. `download_audio` is the second,
heavier pass that actually downloads.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, ExtractorError, UnsupportedError

from mcp_agent_transcriber.names import safe_slug

AUDIO_OUTPUT_FORMATS = ("mp3", "wav", "m4a", "opus", "flac", "aac")

# Quality preset -> yt-dlp format selector. abr filters nudge toward the
# headline quality; `bestaudio/best` is always the fallback term.
_QUALITY_SELECTORS = {
    "best": "bestaudio/best",
    "high": "bestaudio[abr>=128]/bestaudio/best",
    "standard": "bestaudio[abr<=128]/bestaudio/best",
    "low": "worstaudio/worst",
}

_MAX_FORMAT_SAMPLE = 5


def quality_selector(quality: str) -> str:
    """Map a quality preset name to a yt-dlp format selector string."""
    return _QUALITY_SELECTORS.get(quality, _QUALITY_SELECTORS["best"])


def _first_entry(raw: dict) -> dict:
    """Unwrap playlist extraction to the first entry (operate on one video)."""
    if raw.get("_type") == "playlist" and raw.get("entries"):
        return raw["entries"][0]
    return raw


def _extract_info(ydl: YoutubeDL, url: str, *, download: bool) -> dict:
    """
    Run one extract_info pass and normalize its union return type to a mapping.

    yt-dlp declares the result as `_InfoDict | PagedList | list | None`, but every
    caller in this module treats it as a single info dict. PagedList and list are
    playlist containers rather than video info, and None means the extractor
    produced nothing at all, so those cases are raised as a DownloadError for
    the existing classifier to turn into a static, path-free payload rather than
    crashing further down on an AttributeError.
    """
    raw = ydl.extract_info(url, download=download)
    if not isinstance(raw, dict):
        raise DownloadError("extractor returned no info dict")
    return cast(dict, raw)


def safe_info(entry: dict) -> dict:
    """Pick the whitelisted metadata fields an agent can act on."""
    return {
        "id": entry.get("id"),
        "title": entry.get("title"),
        "extractor": entry.get("extractor"),
        "extractor_key": entry.get("extractor_key"),
        "webpage_url": entry.get("webpage_url"),
        "uploader": entry.get("uploader") or entry.get("channel") or entry.get("uploader_id"),
        "duration": round(entry.get("duration") or 0),
        "upload_date": entry.get("upload_date"),
        "is_live": bool(entry.get("is_live")),
    }


def list_caption_tracks(entry: dict) -> list[dict]:
    """Flatten manual + auto-generated caption tracks into lang entries."""
    manual = entry.get("subtitles") or {}
    auto = entry.get("automatic_captions") or {}
    langs = sorted(set(manual) | set(auto))
    return [
        {
            "lang": lang,
            "manual": lang in manual,
            "auto_generated": lang in auto,
            "formats": sorted(
                {
                    f.get("ext")
                    for f in (manual.get(lang, []) + auto.get(lang, []))
                    if f.get("ext")
                }
            ),
        }
        for lang in langs
    ]


def audio_formats_summary(entry: dict) -> dict:
    """Summarize the audio-only formats yt-dlp found, best bitrate first."""
    formats = entry.get("formats") or []
    audio = [
        f
        for f in formats
        if (f.get("vcodec") in (None, "none")) and f.get("acodec") not in (None, "none")
    ]
    audio.sort(key=lambda f: f.get("tbr") or f.get("abr") or 0, reverse=True)
    sample = [
        {
            "format_id": f.get("format_id"),
            "ext": f.get("ext"),
            "abr": f.get("abr"),
            "tbr": round(f.get("tbr") or 0),
        }
        for f in audio[:_MAX_FORMAT_SAMPLE]
    ]
    return {
        "total_formats": len(formats),
        "audio_count": len(audio),
        "best": sample,
        "note": "download_audio(quality=...) picks among these; best = highest bitrate.",
    }


def describe_video(url: str) -> dict:
    """
    One metadata pass over a video URL.

    Returns {"status": "ok", "info": {...}} or a static failure payload.
    """
    try:
        with YoutubeDL(cast(Any, {"quiet": True, "no_warnings": True, "skip_download": True})) as ydl:
            raw = _extract_info(ydl, url, download=False)
    except Exception as exc:  # handled / classified below
        return _failure_payload(exc, context="metadata")

    entry = _first_entry(raw)
    info = safe_info(entry)
    info["slug"] = safe_slug(info.get("title") or "video")
    info["subtitles"] = list_caption_tracks(entry)
    info["audio_formats"] = audio_formats_summary(entry)
    if raw.get("_type") == "playlist":
        info["is_playlist"] = True
        info["playlist_count"] = len(raw.get("entries") or [])
    return {"status": "ok", "info": info, "requested_url": url}


def download_audio(
    url: str,
    download_dir: Path,
    quality: str = "best",
    output_format: str = "orig",
) -> dict:
    """
    Download the audio track of a video URL into download_dir.

    `output_format`: "orig" keeps the platform's container; one of
    mp3/wav/m4a/opus/flac/aac re-encodes with ffmpeg as a postprocess.
    Returns a status payload; on success the file lives at
    `download_dir / filename`.
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    selector = quality_selector(quality)
    opts: dict[str, Any] = {
        "format": selector,
        "outtmpl": str(download_dir / "%(title)s [%(id)s].%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    if output_format in AUDIO_OUTPUT_FORMATS:
        opts["postprocessors"] = [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": output_format,
        }]

    with YoutubeDL(cast(Any, opts)) as ydl:
        try:
            raw = _extract_info(ydl, url, download=True)
        except Exception as exc:  # classified failure payload
            return _failure_payload(exc, context="download")

    entry = _first_entry(raw)
    if not entry.get("id"):
        return {"status": "failed", "reason": "no_entry", "note": "no video entry could be extracted."}

    # cast(): prepare_filename is declared against the stub's `_InfoDict`, which
    # exists only inside the type checker (see the note on the params cast in
    # download_audio). At runtime yt-dlp reads plain keys off the mapping.
    final = Path(ydl.prepare_filename(cast(Any, entry)))
    if output_format in AUDIO_OUTPUT_FORMATS:
        final = final.with_suffix(f".{output_format}")
    if not final.is_file():
        return {
            "status": "failed",
            "reason": "output_missing",
            "note": "download completed but no file landed in the downloads dir.",
        }

    info = safe_info(entry)
    return {
        "status": "ok",
        "platform": info["extractor"],
        "title": info["title"],
        "id": info["id"],
        "duration": info["duration"],
        "filename": final.name,
        "ext": final.suffix.lstrip("."),
        "size_bytes": final.stat().st_size,
        "codec": output_format if output_format in AUDIO_OUTPUT_FORMATS else "original",
        "quality_preset": quality,
        "format_selector": selector,
        "output_dir": str(download_dir),
    }


def _classify(exc: Exception) -> tuple[str, str]:
    """Map a yt-dlp exception to (reason, static note). Messages never leak."""
    msg = re.sub(r"\s+", " ", str(exc)).lower()
    if isinstance(exc, UnsupportedError):
        return "unsupported_url", "no extractor for this URL/host — not a tube-platform link."
    if isinstance(exc, DownloadError):
        if "unsupported url" in msg or "is not a valid url" in msg:
            return "unsupported_url", "the URL is not a supported tube-platform link."
        if any(s in msg for s in (
            "private video", "video unavailable", "unavailable", "removed",
            "has been deleted", "not available", "no longer available",
        )):
            return "unavailable", "the video is private, removed, or region-blocked."
        if any(s in msg for s in (
            "sign in", "login", "age-restricted", "age restricted",
            "members only", "members-only", "account",
        )):
            return "login_required", "the video requires authentication or age verification."
        if any(s in msg for s in (
            "no video formats", "requested format is not available",
            "no audio formats", "format not found",
        )):
            return "format_unavailable", "no matching audio format exists for this video."
        if any(s in msg for s in (
            "timed out", "connection error", "errno", "failed to resolve",
            "name or service not known", "network",
        )):
            return "network_error", "a network error occurred while contacting the platform."
        return "download_error", "yt-dlp failed during extraction or download — check the server log."
    if isinstance(exc, ExtractorError):
        return "extractor_error", "the platform extractor failed — check the server log."
    return "unexpected", "an unexpected error occurred — check the server log."


def _failure_payload(exc: Exception, context: str) -> dict:
    reason, note = _classify(exc)
    return {"status": "failed", "reason": reason, "note": note, "context": context}


__all__ = [
    "AUDIO_OUTPUT_FORMATS",
    "audio_formats_summary",
    "describe_video",
    "download_audio",
    "list_caption_tracks",
    "quality_selector",
    "safe_info",
]
