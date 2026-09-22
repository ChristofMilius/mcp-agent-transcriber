"""
transcript.py — grab captions directly from the platform
==========================================================
The "no download, no model" path. Two sources:

  * YouTube — youtube-transcript-api (fast, lightweight; the platform's own
    caption track, auto-generated or manual).
  * Everywhere else (and as a YouTube fallback when the API chokes) —
    yt-dlp writes the caption track with --skip-download, exactly the
    workflow the old YT_Transcript_downloader notes documented.

Both return segment dicts that cleaning.py turns into agent-friendly text.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    InvalidVideoId,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)
from yt_dlp import YoutubeDL

from mcp_agent_transcriber.cleaning import (
    parse_vtt_cues,
    render_vtt,
    text_from_segments,
    text_from_vtt,
)
from mcp_agent_transcriber.download import _failure_payload, describe_video
from mcp_agent_transcriber.names import transcript_names

logger = logging.getLogger(__name__)

_YOUTUBE_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?[^#]*v=|shorts/|embed/|live/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})"
)


def youtube_id(url: str) -> str | None:
    """Extract an 11-char YouTube video id from common URL shapes, if present."""
    m = _YOUTUBE_ID_RE.search(url or "")
    return m.group(1) if m else None


def _pick_transcript(players, language: str | None, prefer_auto: bool):
    """Pick the best Transcript object: requested kind first, then language."""
    if not players:
        return None
    wanted_auto = prefer_auto
    desired = (language or "").lower()

    def rank(t):
        kind_ok = bool(t.is_generated) == wanted_auto
        code = (t.language_code or "").lower()
        if desired:
            if code == desired:
                lang_rank = 0
            elif code.split("-")[0] == desired.split("-")[0]:
                lang_rank = 1
            else:
                lang_rank = 2
        else:
            lang_rank = 0 if code == "en" else (1 if code.split("-")[0] == "en" else 2)
        return (0 if kind_ok else 1, lang_rank, 0 if not t.is_generated else 1)

    return min(players, key=rank)


def _normalized_segments(segments) -> list[dict]:
    """
    Normalize youtube-transcript-api fetch results into plain dicts.

    Current versions return FetchedTranscriptSnippet objects (.start /
    .duration / .text); older ones returned raw dicts. Both become the
    {"start", "end", "duration", "text"} shape cleaning.py consumes.
    """
    out: list[dict] = []
    for s in segments:
        if isinstance(s, dict):
            out.append(
                {
                    "start": s.get("start"),
                    "end": s.get("end"),
                    "duration": s.get("duration"),
                    "text": s.get("text"),
                }
            )
        else:
            out.append(
                {
                    "start": getattr(s, "start", None),
                    "end": getattr(s, "end", None),
                    "duration": getattr(s, "duration", None),
                    "text": getattr(s, "text", None),
                }
            )
    return out


def _youtube_direct(vid: str, language: str | None, prefer_auto: bool) -> dict:
    """Grab a YouTube caption track via youtube-transcript-api."""
    try:
        api = YouTubeTranscriptApi()
    except Exception as exc:  # pragma: no cover - import-time issues
        logger.error("[transcript] youtube-transcript-api init failed: %s", exc, exc_info=True)
        return {"status": "youtube_api_error", "platform": "youtube"}

    try:
        transcript_list = api.list(vid)
    except TranscriptsDisabled:
        return {"status": "captions_disabled", "platform": "youtube"}
    except VideoUnavailable:
        return {"status": "video_unavailable", "platform": "youtube"}
    except InvalidVideoId:
        return {"status": "invalid_video", "platform": "youtube"}
    except Exception as exc:  # network / API hiccups land here
        logger.warning("[transcript] youtube list failed: %s", exc)
        return {"status": "youtube_error", "platform": "youtube", "class": type(exc).__name__}

    players = [t for t in transcript_list]
    transcript = _pick_transcript(players, language, prefer_auto)
    if transcript is None:
        return {"status": "no_transcript", "platform": "youtube", "available": []}
    try:
        segments = transcript.fetch()
    except (NoTranscriptFound, CouldNotRetrieveTranscript):
        return {
            "status": "no_transcript",
            "platform": "youtube",
            "available": sorted({t.language_code for t in players}),
        }
    except Exception as exc:  # pragma: no cover - fetch transport errors
        logger.warning("[transcript] youtube fetch failed: %s", exc)
        return {"status": "youtube_error", "platform": "youtube", "class": type(exc).__name__}

    return {
        "status": "ok",
        "platform": "youtube",
        "segments": _normalized_segments(segments),
        "language": transcript.language_code,
        "generated": bool(transcript.is_generated),
    }


def _pick_caption_track(tracks: list[dict], language: str | None, prefer_auto: bool) -> dict | None:
    """Pick the best caption track: requested kind first, then language."""
    if not tracks:
        return None
    desired = (language or "").lower()

    def rank(t):
        kind_ok = bool(t["auto_generated"]) if prefer_auto else bool(t["manual"])
        code = (t["lang"] or "").lower()
        if desired:
            if code == desired:
                lang_rank = 0
            elif code.split("-")[0] == desired.split("-")[0]:
                lang_rank = 1
            else:
                lang_rank = 2
        else:
            lang_rank = 0 if code == "en" else (1 if code.split("-")[0] == "en" else 2)
        return (0 if kind_ok else 1, lang_rank, 0 if t["manual"] else 1)

    return min(tracks, key=rank)


def _grab_subs_ytdlp(
    url: str,
    language: str | None,
    prefer_auto: bool,
    tmp_dir: Path,
    title: str,
    platform: str,
    desc: dict | None = None,
) -> dict:
    """
    Write the caption track via yt-dlp (skip_download + writesubtitles) and
    hand back parsed segments. This mirrors the yt-dlp --write-auto-sub
    workflow from the original experiment notes, done in-process.
    """
    if desc is None or desc.get("status") != "ok":
        desc = describe_video(url)
    if desc.get("status") != "ok":
        return desc

    info = desc["info"]
    tracks = info.get("subtitles") or []
    if not tracks:
        return {"status": "no_captions", "platform": platform, "title": title, "available": []}

    chosen = _pick_caption_track(tracks, language, prefer_auto)
    if chosen is None:
        return {
            "status": "no_captions",
            "platform": platform,
            "title": title,
            "available": [t["lang"] for t in tracks],
        }

    lang, is_auto = chosen["lang"], bool(chosen["auto_generated"])
    tmp_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "outtmpl": str(tmp_dir / "%(id)s"),
        "writesubtitles": True,
        "writeautomaticsub": is_auto,
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt",
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    with YoutubeDL(opts) as ydl:
        try:
            ydl.extract_info(url, download=True)
        except Exception as exc:
            return _failure_payload(exc, context="captions")

    cand_auto = tmp_dir / f"{info['id']}.{lang}.auto.vtt"
    cand_manual = tmp_dir / f"{info['id']}.{lang}.vtt"
    produced = cand_auto if cand_auto.is_file() else (cand_manual if cand_manual.is_file() else None)
    if produced is None:
        return {
            "status": "no_captions",
            "platform": platform,
            "title": title,
            "available": [t["lang"] for t in tracks],
        }
    try:
        vtt = produced.read_text(encoding="utf-8", errors="replace")
    finally:
        for f in (cand_auto, cand_manual):
            if f.is_file():
                f.unlink()

    return {
        "status": "ok",
        "platform": platform,
        "segments": parse_vtt_cues(vtt),
        "text": text_from_vtt(vtt),
        "language": lang,
        "generated": is_auto,
        "raw_vtt": vtt,
    }


def _emit_captions(
    payload: dict,
    title: str,
    output_dir: Path,
    want_vtt: bool,
    source: str,
    raw_vtt: str | None = None,
) -> dict:
    """Write the transcript artifact(s) and build the tool result payload."""
    text = payload.get("text") or text_from_segments(
        payload.get("segments") or [], rolling=True
    )
    txt_name, vtt_name = transcript_names(title, payload.get("language") or "captions")
    (output_dir / txt_name).write_text(text + "\n", encoding="utf-8")
    result = {
        "status": "ok",
        "source": source,
        "platform": payload.get("platform"),
        "title": title,
        "language": payload.get("language"),
        "generated": payload.get("generated"),
        "text": text,
        "segments": payload.get("segments") or [],
        "chars": len(text),
        "output_dir": str(output_dir),
        "filename": txt_name,
    }
    if want_vtt:
        vtt = raw_vtt or render_vtt(payload.get("segments") or [])
        (output_dir / vtt_name).write_text(vtt + "\n", encoding="utf-8")
        result["vtt_filename"] = vtt_name
    return result


def grab_transcript(
    url: str,
    language: str | None = None,
    prefer_auto: bool = True,
    want_vtt: bool = False,
    output_dir: Path | None = None,
    tmp_dir: Path | None = None,
) -> dict:
    """
    Return an agent-friendly transcript for a video URL without downloading
    the media. YouTube goes through youtube-transcript-api first; every other
    platform (or a YouTube API failure) falls back to the yt-dlp caption pass.
    """
    lang = (language or "").strip() or None
    output_dir = Path(output_dir)
    tmp_dir = Path(tmp_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    desc = describe_video(url)
    if desc.get("status") != "ok":
        return desc

    info = desc["info"]
    title = info.get("title") or "video"
    platform = info.get("extractor") or "unknown"
    sid = info.get("id")

    if platform == "youtube" and sid:
        payload = _youtube_direct(sid, lang, prefer_auto)
        if payload.get("status") == "ok":
            return _emit_captions(
                payload, title, output_dir, want_vtt, source="youtube_transcript_api"
            )
        if payload.get("status") == "captions_disabled":
            payload["title"] = title
            return payload  # yt-dlp cannot do better; short-circuit
        logger.warning(
            "[transcript] youtube direct failed (%s) — falling back to yt-dlp captions",
            payload.get("status"),
        )

    payload = _grab_subs_ytdlp(url, lang, prefer_auto, tmp_dir, title, platform, desc)
    if payload.get("status") != "ok":
        return payload
    return _emit_captions(
        payload, title, output_dir, want_vtt, source="yt_dlp_captions", raw_vtt=payload.get("raw_vtt")
    )


__all__ = ["grab_transcript", "youtube_id"]
