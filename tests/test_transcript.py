"""tests/test_transcript.py — direct caption grabbing."""
from __future__ import annotations

import pytest
from conftest import FakeSubYDL, FakeTranscript, FakeYDL, FakeYouTubeApi
from youtube_transcript_api._errors import TranscriptsDisabled

from mcp_agent_transcriber import download as dl
from mcp_agent_transcriber import transcript as tr
from mcp_agent_transcriber.transcript import grab_transcript, youtube_id


class _DisabledApi(FakeYouTubeApi):
    def __init__(self):
        super().__init__(list_fail=TranscriptsDisabled("v123"))


class _EmptyApi(FakeYouTubeApi):
    def __init__(self):
        super().__init__(transcripts=[])


class _VimeoYDL(FakeYDL):
    def __init__(self, opts=None):
        entry = dict(FakeYDL().entry)
        entry["extractor"] = "vimeo"
        super().__init__(opts=opts, entry=entry)


@pytest.fixture
def patch_ytdlp(monkeypatch):
    """Route both yt-dlp uses (metadata + caption pass) to fakes."""
    monkeypatch.setattr(dl, "YoutubeDL", FakeYDL)
    monkeypatch.setattr(tr, "YoutubeDL", lambda opts=None: FakeSubYDL(opts=opts))


class TestYoutubeId:
    @pytest.mark.parametrize("url", [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ?start=5",
        "https://www.youtube.com/watch?foo=1&v=dQw4w9WgXcQ&t=30",
    ])
    def test_extracts_id(self, url):
        assert youtube_id(url) == "dQw4w9WgXcQ"

    @pytest.mark.parametrize("url", [
        "https://vimeo.com/123456",
        "https://www.youtube.com/",
        "https://example.com/watch?v=dQw4w9WgXcQ",
    ])
    def test_no_id(self, url):
        assert youtube_id(url) is None


class TestGrabTranscript:
    def test_youtube_direct_ok(self, tmp_path, monkeypatch, patch_ytdlp):
        monkeypatch.setattr(tr, "YouTubeTranscriptApi", FakeYouTubeApi)
        result = grab_transcript(
            "https://www.youtube.com/watch?v=x", output_dir=tmp_path / "out", tmp_dir=tmp_path / "dl"
        )
        assert result["status"] == "ok"
        assert result["source"] == "youtube_transcript_api"
        assert result["language"] == "en"
        assert result["text"] == "hello world test transcript"
        assert (tmp_path / "out" / result["filename"]).is_file()

    def test_youtube_language_preference(self, tmp_path, monkeypatch, patch_ytdlp):
        monkeypatch.setattr(tr, "YouTubeTranscriptApi", FakeYouTubeApi)
        result = grab_transcript(
            "https://www.youtube.com/watch?v=x", language="de",
            output_dir=tmp_path / "out", tmp_dir=tmp_path / "dl",
        )
        assert result["status"] == "ok"
        assert result["language"] == "de"
        assert result["text"] == "hallo welt"

    def test_captions_disabled_short_circuits(self, tmp_path, monkeypatch, patch_ytdlp):
        monkeypatch.setattr(tr, "YouTubeTranscriptApi", _DisabledApi)
        result = grab_transcript(
            "https://www.youtube.com/watch?v=x", output_dir=tmp_path / "out", tmp_dir=tmp_path / "dl"
        )
        assert result["status"] == "captions_disabled"
        assert result["title"] == "Demo Video"

    def test_youtube_no_transcript_falls_back_to_ytdlp(self, tmp_path, monkeypatch, patch_ytdlp):
        monkeypatch.setattr(tr, "YouTubeTranscriptApi", _EmptyApi)
        result = grab_transcript(
            "https://www.youtube.com/watch?v=x", output_dir=tmp_path / "out", tmp_dir=tmp_path / "dl"
        )
        assert result["status"] == "ok"
        assert result["source"] == "yt_dlp_captions"
        assert result["language"] == "en"
        assert result["generated"] is True
        assert result["text"] == "hello world test transcript"

    def test_non_youtube_platform(self, tmp_path, monkeypatch, patch_ytdlp):
        monkeypatch.setattr(dl, "YoutubeDL", _VimeoYDL)
        result = grab_transcript(
            "https://vimeo.com/123456", output_dir=tmp_path / "out", tmp_dir=tmp_path / "dl"
        )
        assert result["status"] == "ok"
        assert result["source"] == "yt_dlp_captions"
        assert result["platform"] == "vimeo"

    def test_want_vtt_writes_second_file(self, tmp_path, monkeypatch, patch_ytdlp):
        monkeypatch.setattr(tr, "YouTubeTranscriptApi", FakeYouTubeApi)
        result = grab_transcript(
            "https://www.youtube.com/watch?v=x", want_vtt=True,
            output_dir=tmp_path / "out", tmp_dir=tmp_path / "dl",
        )
        assert result["vtt_filename"].endswith(".vtt")
        assert (tmp_path / "out" / result["vtt_filename"]).is_file()


class TestPickTranscript:
    def test_prefers_requested_kind_and_language(self):
        picks = [
            FakeTranscript("en", False, []),
            FakeTranscript("de", True, []),
            FakeTranscript("en", True, []),
        ]
        chosen = tr._pick_transcript(picks, None, prefer_auto=True)
        assert chosen is not None
        assert chosen.language_code == "en" and chosen.is_generated is True

    def test_when_no_match_returns_a_transcript_anyway(self):
        picks = [FakeTranscript("fr", True, [])]
        chosen = tr._pick_transcript(picks, "en", prefer_auto=False)
        assert chosen is not None
        assert chosen.language_code == "fr"

    def test_empty_list(self):
        assert tr._pick_transcript([], "en", True) is None
