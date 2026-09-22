"""tests/test_platforms.py — platform discovery against the installed yt-dlp."""
from __future__ import annotations

from mcp_agent_transcriber.platforms import (
    COMMON_PLATFORMS,
    available_extractor_names,
    supported_platforms,
)


class TestPlatforms:
    def test_common_platforms_are_tuples(self):
        assert all(len(p) == 2 for p in COMMON_PLATFORMS)
        slugs = {p[0] for p in COMMON_PLATFORMS}
        assert "youtube" in slugs and "vimeo" in slugs

    def test_available_extractors_include_youtube(self):
        names = available_extractor_names()
        assert isinstance(names, set)
        assert "youtube" in names

    def test_supported_platforms_shape_and_youtube(self):
        result = supported_platforms()
        assert result["count"] == len(COMMON_PLATFORMS)
        assert any(p["slug"] == "youtube" and p["available"] for p in result["platforms"])
