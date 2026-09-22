"""tests/test_tools.py — tool surface through a fake MCPServer."""
from __future__ import annotations

import json

import pytest
from conftest import FakeServer, FakeYDL, FakeYouTubeApi, build_ctx

from mcp_agent_transcriber import download as dl
from mcp_agent_transcriber import transcript as tr
from mcp_agent_transcriber.server import INSTRUCTIONS, SERVER_NAME
from mcp_agent_transcriber.tool_surface import register_all

EXPECTED_TOOLS = {
    "get_current_datetime",
    "supported_platforms",
    "video_info",
    "list_available_transcripts",
    "fetch_transcript",
    "download_audio",
    "transcriber_status",
    "transcribe_file",
    "transcribe_video",
}


@pytest.fixture
def ctx(tmp_project):
    root, env = tmp_project
    return build_ctx(root, env)


@pytest.fixture
def tools(ctx):
    fake = FakeServer()
    register_all(fake, ctx)
    return fake.tools


def _parse(payload: str) -> dict:
    return json.loads(payload)


class TestServerMeta:
    def test_server_name_and_instructions(self):
        assert SERVER_NAME == "mcp-agent-transcriber"
        assert "whisper" in INSTRUCTIONS.lower()
        assert "fetch_transcript" in INSTRUCTIONS
        assert "transcribe_video" in INSTRUCTIONS

    def test_all_tools_registered(self, tools):
        assert EXPECTED_TOOLS <= set(tools)


class TestUtilityTools:
    def test_get_current_datetime(self, tools):
        out = _parse(tools["get_current_datetime"]())
        assert "local" in out and "utc" in out and "weekday" in out

    def test_supported_platforms(self, tools):
        out = _parse(tools["supported_platforms"]())
        assert out["count"] == len(out["platforms"])
        assert any(p["slug"] == "youtube" and p["available"] for p in out["platforms"])

    def test_transcriber_status(self, tools, ctx):
        out = _parse(tools["transcriber_status"]())
        assert out["model"] == ctx.cfg.model_name


class TestInfoTools:
    def test_video_info_ok(self, tools, monkeypatch):
        monkeypatch.setattr(dl, "YoutubeDL", FakeYDL)
        out = _parse(tools["video_info"](url="https://youtu.be/dQw4w9WgXcQ"))
        assert out["status"] == "ok"
        assert out["info"]["extractor"] == "youtube"
        assert out["info"]["subtitles"] and out["info"]["audio_formats"]

    def test_video_info_unsupported(self, tools, monkeypatch):
        from yt_dlp.utils import UnsupportedError

        class BadYDL(FakeYDL):
            def extract_info(self, url, download=False):
                raise UnsupportedError("Unsupported URL: https://x.test/a")

        monkeypatch.setattr(dl, "YoutubeDL", BadYDL)
        out = _parse(tools["video_info"](url="https://x.test/a"))
        assert out["status"] == "failed"
        assert out["reason"] == "unsupported_url"


class TestTranscriptTools:
    def test_list_available_transcripts(self, tools, monkeypatch):
        monkeypatch.setattr(dl, "YoutubeDL", FakeYDL)
        out = _parse(tools["list_available_transcripts"](url="https://youtu.be/x"))
        assert out["status"] == "ok"
        langs = {t["lang"] for t in out["subtitles"]}
        assert {"en", "de"} <= langs

    def test_fetch_transcript_youtube(self, tools, ctx, monkeypatch):
        monkeypatch.setattr(dl, "YoutubeDL", FakeYDL)
        monkeypatch.setattr(tr, "YouTubeTranscriptApi", FakeYouTubeApi)
        out = _parse(tools["fetch_transcript"](url="https://youtu.be/x"))
        assert out["status"] == "ok"
        assert out["source"] == "youtube_transcript_api"
        assert (ctx.cfg.output_dir / out["filename"]).is_file()


class TestDownloadTools:
    def test_download_audio_ok(self, tools, ctx, monkeypatch):
        monkeypatch.setattr(dl, "YoutubeDL", FakeYDL)
        out = _parse(tools["download_audio"](url="https://youtu.be/x"))
        assert out["status"] == "ok"
        assert (ctx.cfg.downloads_dir / out["filename"]).is_file()

    def test_download_audio_bad_format(self, tools, monkeypatch):
        monkeypatch.setattr(dl, "YoutubeDL", FakeYDL)
        out = _parse(tools["download_audio"](url="https://youtu.be/x", output_format="ogg_vorbis"))
        # unknown codec falls back to the plain format selector path
        assert out["status"] == "ok"
        assert out["ext"] == "m4a"


class TestTranscribeTools:
    def test_transcribe_file_ok(self, tools, ctx, tmp_path):
        media = tmp_path / "clip.m4a"
        media.write_bytes(b"xx")
        out = _parse(tools["transcribe_file"](path=str(media), want_vtt=True, return_segments=False))
        assert out["status"] == "ok"
        assert out["text"].startswith("hello world")
        assert (ctx.cfg.output_dir / out["filename"]).is_file()
        assert "segments" not in out

    def test_transcribe_file_missing(self, tools):
        out = _parse(tools["transcribe_file"](path=r"C:\does\not\exist.m4a"))
        assert out["status"] == "failed"
        assert out["reason"] == "file_not_found"

    def test_transcribe_video_tool_captions(self, tools, ctx, monkeypatch):
        monkeypatch.setattr(dl, "YoutubeDL", FakeYDL)
        monkeypatch.setattr(tr, "YouTubeTranscriptApi", FakeYouTubeApi)
        out = _parse(tools["transcribe_video"](url="https://youtu.be/x", method="captions"))
        assert out["status"] == "ok"
        assert out["source"] == "youtube_transcript_api"


class TestToolErrors:
    def test_unknown_tool_name_reported(self, tools):
        from mcp_agent_transcriber.errors import tool_error

        msg = tool_error("fake_tool", ValueError("boom"))
        assert "fake_tool" in msg
        assert "ValueError" in msg
        assert "boom" not in msg  # exception message is never surfaced
