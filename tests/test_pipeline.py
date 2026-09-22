"""tests/test_pipeline.py — transcribe_video orchestration."""
from __future__ import annotations

from pathlib import Path

from conftest import FakeYDL, FakeYouTubeApi, build_ctx
from youtube_transcript_api._errors import TranscriptsDisabled

from mcp_agent_transcriber import download as dl
from mcp_agent_transcriber import transcript as tr
from mcp_agent_transcriber.pipeline import transcribe_video


class _FailingApi(FakeYouTubeApi):
    def __init__(self):
        super().__init__(list_fail=TranscriptsDisabled("v123"))


def _params(tmp_project):
    root, env = tmp_project
    return {
        "output_dir": Path(env["TRANSCRIBER_OUTPUT_DIR"]),
        "downloads_dir": Path(env["TRANSCRIBER_DOWNLOADS_DIR"]),
        "engine": build_ctx(root, env).engine,
    }


class TestPipelineRoute:
    def test_method_captions_uses_direct_grab(self, tmp_project, monkeypatch):
        monkeypatch.setattr(dl, "YoutubeDL", FakeYDL)
        monkeypatch.setattr(tr, "YouTubeTranscriptApi", FakeYouTubeApi)
        result = transcribe_video("https://youtube.com/watch?v=x", method="captions", **_params(tmp_project))
        assert result["status"] == "ok"
        assert result["source"] == "youtube_transcript_api"

    def test_method_whisper_downloads_and_transcribes(self, tmp_project, monkeypatch):
        monkeypatch.setattr(dl, "YoutubeDL", FakeYDL)
        params = _params(tmp_project)
        result = transcribe_video("https://youtu.be/x", method="whisper", **params)
        assert result["status"] == "ok"
        assert result["source"] == "whisper"
        assert result["text"] == "hello world test transcript"
        assert (params["output_dir"] / result["filename"]).is_file()
        # keep_audio=False → the downloaded audio file is cleaned up
        audio = params["downloads_dir"] / result["audio"]["filename"]
        assert not audio.exists()
        assert result["audio"]["saved"] is False

    def test_method_whisper_keep_audio(self, tmp_project, monkeypatch):
        monkeypatch.setattr(dl, "YoutubeDL", FakeYDL)
        params = _params(tmp_project)
        result = transcribe_video("https://youtu.be/x", method="whisper", keep_audio=True, **params)
        assert result["audio"]["saved"] is True
        assert (params["downloads_dir"] / result["audio"]["filename"]).exists()

    def test_method_auto_falls_back_to_whisper(self, tmp_project, monkeypatch):
        monkeypatch.setattr(dl, "YoutubeDL", FakeYDL)
        monkeypatch.setattr(tr, "YouTubeTranscriptApi", _FailingApi)
        result = transcribe_video("https://youtube.com/watch?v=x", method="auto", **_params(tmp_project))
        assert result["status"] == "ok"
        assert result["source"] == "whisper"
        assert result["captions_failed_reason"] == "captions_disabled"

    def test_method_auto_prefers_captions(self, tmp_project, monkeypatch):
        monkeypatch.setattr(dl, "YoutubeDL", FakeYDL)
        monkeypatch.setattr(tr, "YouTubeTranscriptApi", FakeYouTubeApi)
        result = transcribe_video("https://youtube.com/watch?v=x", method="auto", **_params(tmp_project))
        assert result["status"] == "ok"
        assert result["source"] == "youtube_transcript_api"

    def test_bad_method(self, tmp_project, monkeypatch):
        result = transcribe_video("https://youtu.be/x", method="nope", **_params(tmp_project))
        assert result["status"] == "failed"
        assert result["reason"] == "bad_method"
        assert result["allowed"] == ["auto", "captions", "whisper"]

    def test_whisper_download_failure_passthrough(self, tmp_project, monkeypatch):
        from yt_dlp.utils import DownloadError

        class FailingYDL(FakeYDL):
            def extract_info(self, url, download=True):
                raise DownloadError("ERROR: [youtube] x: Video unavailable")

        monkeypatch.setattr(dl, "YoutubeDL", FailingYDL)
        result = transcribe_video("https://youtu.be/x", method="whisper", **_params(tmp_project))
        assert result["status"] == "failed"
        assert result["reason"] == "unavailable"
