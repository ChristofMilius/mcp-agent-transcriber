"""
names.py — filesystem-safe filename helpers for emitted artifacts
=================================================================
The standalone whisper_transcriber named outputs `{stem}_transcript.txt` /
`.vtt` next to the input. The MCP server emits into a shared output dir, so
artifacts are namespaced with a sanitized title slug, the language, and a
timestamp to survive repeated runs without collisions.
"""

from __future__ import annotations

import re
from datetime import datetime

MAX_SLUG_LEN = 80


def safe_slug(title: str) -> str:
    """Derive a filesystem-safe slug from a video/record title."""
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return slug[:MAX_SLUG_LEN].strip("_") or "video"


def timestamp_tag() -> str:
    """Compact, sortable timestamp used inside emitted filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def transcript_names(title: str, language: str) -> tuple[str, str]:
    """
    Return (txt_name, vtt_name) sharing one timestamp, so a run's files line up.
    """
    stamp = timestamp_tag()
    stem = f"{safe_slug(title)}_{language}"
    return f"{stem}_{stamp}.txt", f"{stem}_{stamp}.vtt"


__all__ = ["transcript_names", "safe_slug", "timestamp_tag"]
