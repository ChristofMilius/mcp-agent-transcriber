"""tests/conftest.py — shared fixtures."""
from __future__ import annotations

import types
from pathlib import Path

import pytest


class FakeServer:
    """Collects tools registered via @server.tool() so we can call them."""

    def __init__(self):
        self.tools = {}

    def tool(self):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


class FakeEngine:
    """Cheap stand-in for WhisperEngine: no model, deterministic results."""

    def __init__(self, model_name: str = "turbo"):
        self.model_name = model_name
        self._device = "cpu"

    @property
    def device(self):
        return self._device

    def is_loaded(self):
        return False

    def status(self):
        return {
            "model": self.model_name,
            "whisper_installed": True,
            "torch_installed": True,
            "cuda_available": False,
            "device": self._device,
            "model_loaded": False,
            "weights_cached": False,
            "ffmpeg_available": True,
        }

    def transcribe(self, path: str, language: str | None = None):
        return {
            "text": "hello world test transcript",
            "language": language or "en",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello world"},
                {"start": 1.0, "end": 2.0, "text": "test transcript"},
            ],
            "device": self._device,
            "fp16": False,
        }


@pytest.fixture
def tmp_project(tmp_path: Path):
    """Return (root, env_dict) for a self-contained project tree."""
    env = {
        "TRANSCRIBER_OUTPUT_DIR": str(tmp_path / "transcript_output"),
        "TRANSCRIBER_DOWNLOADS_DIR": str(tmp_path / "downloads"),
        "TRANSCRIBER_LOGS_DIR": str(tmp_path / "logs"),
        "TRANSCRIBER_MODEL": "turbo",
        "TRANSCRIBER_LANGUAGE": "en",
    }
    return tmp_path, env


def build_ctx(root: Path, env: dict):
    """Build a context SimpleNamespace straight from env (no Config)."""
    cfg = types.SimpleNamespace(
        output_dir=Path(env["TRANSCRIBER_OUTPUT_DIR"]),
        downloads_dir=Path(env["TRANSCRIBER_DOWNLOADS_DIR"]),
        logs_dir=Path(env["TRANSCRIBER_LOGS_DIR"]),
        model_name=env.get("TRANSCRIBER_MODEL", "turbo"),
        default_language=env.get("TRANSCRIBER_LANGUAGE", "en"),
    )
    return types.SimpleNamespace(cfg=cfg, engine=FakeEngine(cfg.model_name))


class FakeYDL:
    """
    In-process yt-dlp stand-in. extract_info returns a fixed video entry and,
    when download=True, materializes the expected output file (honoring an
    FFmpegExtractAudio postprocessor) so download logic can stat/clean it.
    """

    def __init__(self, opts=None, entry=None):
        self.opts = opts or {}
        self.entry = entry or {
            "id": "v123",
            "title": "Demo Video",
            "extractor": "youtube",
            "extractor_key": "Youtube",
            "webpage_url": "https://youtu.be/dQw4w9WgXcQ",
            "duration": 120,
            "uploader": "John Smith",
            "ext": "m4a",
            "subtitles": {"en": [{"ext": "vtt"}]},
            "automatic_captions": {"de": [{"ext": "vtt"}], "en": [{"ext": "vtt"}]},
            "formats": [
                {"format_id": "140", "ext": "m4a", "acodec": "mp4a.40.2", "vcodec": "none", "abr": 128, "tbr": 128.5},
                {"format_id": "251", "ext": "webm", "acodec": "opus", "vcodec": "none", "abr": 160, "tbr": 160.1},
            ],
        }
        self.written = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def prepare_filename(self, entry):
        out = Path(self.opts.get("outtmpl", ".")).parent
        return str(out / f"{entry['title']} [{entry['id']}].{entry.get('ext', 'm4a')}")

    def extract_info(self, url, download=False):
        if download:
            path = Path(self.prepare_filename(self.entry))
            for pp in self.opts.get("postprocessors") or []:
                if pp.get("key") == "FFmpegExtractAudio":
                    path = path.with_suffix(f".{pp.get('preferredcodec')}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake audio payload")
            self.written.append(path)
        return dict(self.entry)


class FakeTranscript:
    def __init__(self, language_code, is_generated, segments):
        self.language_code = language_code
        self.language = language_code
        self.is_generated = is_generated
        self._segments = segments

    def fetch(self):
        return self._segments


SEGMENTS_EN = [
    {"text": "hello world", "start": 0.0, "duration": 1.0},
    {"text": "test transcript", "start": 1.0, "duration": 1.0},
]


class FakeYouTubeApi:
    """Stand-in for youtube_transcript_api.YouTubeTranscriptApi."""

    def __init__(self, list_fail=None, transcripts=None):
        self.list_fail = list_fail
        if transcripts is None:
            transcripts = [
                FakeTranscript("en", True, [dict(s) for s in SEGMENTS_EN]),
                FakeTranscript("de", True, [{"text": "hallo welt", "start": 0.0, "duration": 2.0}]),
            ]
        self.transcripts = transcripts

    def list(self, video_id):
        if self.list_fail is not None:
            raise self.list_fail
        return list(self.transcripts)


class FakeSubYDL:
    """
    Stand-in for the yt-dlp caption pass: on extract_info it writes the vtt
    file at {outtmpl-parent}/{id}.{lang}.auto.vtt (or .vtt for manual).
    """

    def __init__(self, opts=None, vtt_text: str | None = None):
        self.opts = opts or {}
        self.vtt_text = vtt_text or (
            "WEBVTT\n\n"
            "1\n00:00:00.000 --> 00:00:01.000\n"
            "hello <00:00:00.500><c> world\n\n"
            "2\n00:00:01.000 --> 00:00:02.000\n"
            "test transcript\n"
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def extract_info(self, url, download=False):
        lang = self.opts.get("subtitleslangs", ["en"])[0]
        is_auto = bool(self.opts.get("writeautomaticsub"))
        base = Path(self.opts.get("outtmpl", ".")).parent
        name = f"v123.{lang}.auto.vtt" if is_auto else f"v123.{lang}.vtt"
        target = base / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.vtt_text, encoding="utf-8")
        return {"id": "v123"}


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """WhisperEngine._MODEL_CACHE must not leak between tests."""
    import mcp_agent_transcriber.whisper_engine as we

    we._MODEL_CACHE.clear()
    yield
    we._MODEL_CACHE.clear()


__all__ = [
    "FakeEngine",
    "FakeServer",
    "FakeSubYDL",
    "FakeTranscript",
    "FakeYDL",
    "FakeYouTubeApi",
    "SEGMENTS_EN",
    "build_ctx",
]
