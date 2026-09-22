"""
platforms.py — what "any common tube platform" means here
==========================================================
yt-dlp already knows thousands of sites; the transcriber names the common
tube/video platforms an agent is likely to meet and reports which of them the
installed yt-dlp build actually ships extractors for.
"""

from __future__ import annotations

from yt_dlp.extractor import gen_extractors

COMMON_PLATFORMS: list[tuple[str, str]] = [
    ("youtube", "YouTube"),
    ("vimeo", "Vimeo"),
    ("dailymotion", "Dailymotion"),
    ("twitch", "Twitch"),
    ("tiktok", "TikTok"),
    ("facebook", "Facebook"),
    ("instagram", "Instagram"),
    ("bilibili", "Bilibili"),
    ("soundcloud", "SoundCloud"),
    ("rumble", "Rumble"),
    ("odysee", "Odysee"),
    ("peertube", "PeerTube"),
    ("vk", "VK"),
    ("rutube", "Rutube"),
    ("youku", "Youku"),
    ("twitter", "X / Twitter"),
]


def available_extractor_names() -> set[str]:
    """Lowercased IE_NAME of every extractor in the installed yt-dlp."""
    return {ie.IE_NAME.lower() for ie in gen_extractors()}


def _has_extractor(slug: str, avail: set[str]) -> bool:
    # Twitch ships as twitch:streams / twitch:videos, Facebook as multiple,
    # etc. — an extractor matches on the slug prefix or exact name.
    return slug in avail or any(name.startswith(f"{slug}:") for name in avail)


def supported_platforms() -> dict:
    """Report which common tube platforms the installed yt-dlp can resolve."""
    avail = available_extractor_names()
    items = [
        {"slug": slug, "name": name, "available": _has_extractor(slug, avail)}
        for slug, name in COMMON_PLATFORMS
    ]
    return {
        "count": len(items),
        "available_count": sum(1 for i in items if i["available"]),
        "platforms": items,
        "note": "YouTube also supports direct caption grabbing via "
                "youtube-transcript-api (no download, no model).",
    }


__all__ = ["COMMON_PLATFORMS", "available_extractor_names", "supported_platforms"]
