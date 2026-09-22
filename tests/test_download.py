"""tests/test_download.py — yt-dlp metadata + audio download logic."""
from __future__ import annotations

from pathlib import Path

from conftest import FakeYDL
from yt_dlp.utils import DownloadError, UnsupportedError

from mcp_agent_transcriber import download as dl
from mcp_agent_transcriber.download import (
    _classify,
    audio_formats_summary,
    describe_video,
    download_audio,
    list_caption_tracks,
    quality_selector,
    safe_info,
)


class TestQuality:
    def test_default_is_best(self):
        assert quality_selector("best") == "bestaudio/best"

    def test_unknown_falls_back_to_best(self):
        assert quality_selector("bogus") == "bestaudio/best"

    def test_high_and_low_map(self):
        assert quality_selector("high").startswith("bestaudio[abr")
        assert quality_selector("low") == "worstaudio/worst"


class TestMetadata:
    def test_safe_info_whitelists_fields(self):
        info = safe_info({"id": "x", "title": "T", "extractor": "youtube",
                          "secret_field": "nope"})
        assert info["id"] == "x"
        assert "secret_field" not in info

    def test_caption_tracks_merge_manual_and_auto(self):
        entry = {
            "subtitles": {"en": [{"ext": "vtt"}]},
            "automatic_captions": {"en": [{"ext": "json3"}], "de": [{"ext": "vtt"}]},
        }
        tracks = {t["lang"]: t for t in list_caption_tracks(entry)}
        assert tracks["en"]["manual"] is True
        assert tracks["en"]["auto_generated"] is True
        assert tracks["de"]["manual"] is False

    def test_audio_formats_summary_counts(self):
        summary = audio_formats_summary(FakeYDL().entry)
        assert summary["audio_count"] == 2
        assert summary["best"][0]["format_id"] == "251"  # highest tbr first


class TestDescribeVideo:
    def test_ok(self, monkeypatch):
        monkeypatch.setattr(dl, "YoutubeDL", FakeYDL)
        result = describe_video("https://youtu.be/dQw4w9WgXcQ")
        assert result["status"] == "ok"
        info = result["info"]
        assert info["title"] == "Demo Video"
        assert info["extractor"] == "youtube"
        assert info["slug"] == "demo_video"
        assert "subtitles" in info and "audio_formats" in info

    def test_unsupported_url_classified(self, monkeypatch):
        def boom(*a, **kw):
            raise UnsupportedError("Unsupported URL: https://weird.example/x")

        class BadYDL(FakeYDL):
            def extract_info(self, url, download=False):
                boom()

        monkeypatch.setattr(dl, "YoutubeDL", BadYDL)
        result = describe_video("https://weird.example/x")
        assert result["status"] == "failed"
        assert result["reason"] == "unsupported_url"

    def test_playlist_unwraps_to_first_entry(self, monkeypatch):
        class PlaylistYDL(FakeYDL):
            def extract_info(self, url, download=False):
                return {"_type": "playlist", "title": "PL", "entries": [dict(self.entry)]}

        monkeypatch.setattr(dl, "YoutubeDL", PlaylistYDL)
        result = describe_video("https://youtube.com/playlist?list=abc")
        assert result["status"] == "ok"
        assert result["info"]["is_playlist"] is True
        assert result["info"]["id"] == "v123"


class TestDownloadAudio:
    def test_orig_download(self, tmp_path: Path, monkeypatch):
        base = tmp_path / "downloads"
        monkeypatch.setattr(dl, "YoutubeDL", FakeYDL)
        result = download_audio("https://youtu.be/dQw4w9WgXcQ", base)
        assert result["status"] == "ok"
        assert result["ext"] == "m4a"
        assert (base / result["filename"]).is_file()
        assert result["size_bytes"] == len(b"fake audio payload")

    def test_transcode_writes_codec_suffix(self, tmp_path: Path, monkeypatch):
        base = tmp_path / "downloads"
        monkeypatch.setattr(dl, "YoutubeDL", FakeYDL)
        result = download_audio("https://youtu.be/dQw4w9WgXcQ", base, output_format="mp3")
        assert result["status"] == "ok"
        assert result["ext"] == "mp3"
        assert (base / result["filename"]).is_file()
        assert result["filename"].endswith(".mp3")

    def test_failure_returns_static_payload(self, tmp_path: Path, monkeypatch):
        class FailingYDL(FakeYDL):
            def extract_info(self, url, download=True):
                raise DownloadError("ERROR: [youtube] v123: The video is unavailable")

        monkeypatch.setattr(dl, "YoutubeDL", FailingYDL)
        result = download_audio("https://youtu.be/x", tmp_path)
        assert result["status"] == "failed"
        assert result["reason"] == "unavailable"


class TestClassify:
    def test_private_video(self):
        assert _classify(DownloadError("ERROR: Video is not available"))[0] == "unavailable"

    def test_login_required(self):
        assert _classify(DownloadError("ERROR: Sign in to confirm your age"))[0] == "login_required"

    def test_network_error(self):
        assert _classify(DownloadError("ERROR: [Errno -2] Name or service not known"))[0] == "network_error"

    def test_processor_note_is_static(self):
        reason, note = _classify(UnsupportedError("Unsupported URL: foobar"))
        assert reason == "unsupported_url"
        assert "foobar" not in note
