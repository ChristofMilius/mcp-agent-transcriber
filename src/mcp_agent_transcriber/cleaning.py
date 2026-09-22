"""
cleaning.py — turn raw captions / VTT into agent-friendly text
===============================================================
Concepts ported from the standalone whisper_transcriber workflow and the
early YT_Transcript_downloader experiment it documented:

  * strip inline timing tags (`<00:00:01.234><c>`), the artifact of
    word-by-word rolling captions;
  * drop cue headers, indexes, and timestamp rows;
  * dedupe consecutive identical lines (the same rolling-captions
    artifact repeats the previous words in every new cue);
  * collapse runs of whitespace.

Two entry points feed this module:
  * whisper segments (the original repo's `result["segments"]`), and
  * parsed VTT cues (youtube-transcript-api has no VTT at all; yt-dlp
    caption downloads do), which are unified into segment dicts so the
    same text-from-segments path and VTT renderer serve all sources.
"""

from __future__ import annotations

import re

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")


def strip_inline_tags(line: str) -> str:
    """Remove inline VTT tags like <00:00:01.234><c> or <i>."""
    return _TAG_RE.sub("", line)


def clean_line(line: str) -> str:
    """Collapse whitespace and strip inline tags from a single caption line."""
    return _WS_RE.sub(" ", strip_inline_tags(line)).strip()


def _ts_to_sec(ts: str) -> float:
    """Parse an HH:MM:SS.mmm (or MM:SS.mmm) VTT/RFC3339-style timestamp."""
    parts = ts.split(":")
    secs = float(parts[-1])
    if len(parts) >= 2:
        secs += int(parts[-2]) * 60
    if len(parts) >= 3:
        secs += int(parts[-3]) * 3600
    return secs


def parse_vtt_cues(vtt: str) -> list[dict]:
    """
    Parse VTT text into segment dicts: {"start", "end", "text"}.

    Tolerates YouTube flavor: a leading index line per cue, `Kind:`/`WEBVTT`
    headers, and blank-line separated cues.
    """
    cues: list[dict] = []
    cur: dict | None = None
    for raw in vtt.splitlines():
        line = raw.strip()
        if "-->" in line:
            start, rest = line.split("-->", 1)
            cur = {
                "start": _ts_to_sec(start.strip()),
                "end": _ts_to_sec(rest.strip().split()[0]),
                "parts": [],
            }
            cues.append(cur)
        elif line == "":
            cur = None
        elif cur is not None:
            if cur["parts"] or not line.isdigit():
                cleaned = clean_line(line)
                if cleaned:
                    cur["parts"].append(cleaned)
    for cue in cues:
        cue["text"] = " ".join(cue["parts"])
    return cues


def _fmt_ts(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def render_vtt(segments: list[dict]) -> str:
    """
    Render segment dicts ({"start", "end", "text"} shape — end falls back to
    start + duration) as a WEBVTT document. Mirrors the original repo's
    per-segment VTT writer, generalized to consumed cue streams.
    """
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(segments, 1):
        start = float(seg.get("start") or 0)
        end = seg.get("end")
        if end is None:
            end = start + float(seg.get("duration") or 0)
        lines.append(f"{i}")
        lines.append(f"{_fmt_ts(start)} --> {_fmt_ts(end)}")
        lines.append((seg.get("text") or "").strip())
        lines.append("")
    return "\n".join(lines)


def _overlap_extra(prev: list[str], cur: list[str]) -> list[str]:
    """
    Words of `cur` not already covered by the tail of `prev`.

    Rolling (word-by-word) captions repeat the end of the previous cue at the
    start of the next one; the longest matching suffix/prefix overlap is cut
    so the repeated words are only emitted once.
    """
    if not prev:
        return cur
    overlap = 0
    for n in range(1, min(len(prev), len(cur)) + 1):
        if prev[-n:] == cur[:n]:
            overlap = n
    return cur[overlap:]


def text_from_segments(segments: list[dict], rolling: bool = False) -> str:
    """
    Join segment text into one deduped, whitespace-collapsed transcript.

    Default: consecutive identical lines are dropped once. With
    `rolling=True` (caption dumps) the word-by-word repeat artifact is
    handled via `_overlap_extra`, which also collapses partial repetitions.
    """
    if rolling:
        out: list[str] = []
        prev: list[str] = []
        for seg in segments:
            tokens = _WS_RE.sub(" ", strip_inline_tags(seg.get("text") or "")).strip().split()
            if not tokens or tokens == prev:
                continue
            out.extend(_overlap_extra(prev, tokens))
            prev = tokens
        return " ".join(out)
    lines: list[str] = []
    last = ""
    for seg in segments:
        text = _WS_RE.sub(" ", strip_inline_tags(seg.get("text") or "")).strip()
        if not text or text == last:
            continue
        lines.append(text)
        last = text
    return " ".join(lines)


def text_from_vtt(vtt: str) -> str:
    """Convenience: parse VTT then clean it into plain transcript text."""
    return text_from_segments(parse_vtt_cues(vtt), rolling=True)


__all__ = [
    "clean_line",
    "parse_vtt_cues",
    "render_vtt",
    "strip_inline_tags",
    "text_from_segments",
    "text_from_vtt",
]
