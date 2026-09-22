"""
whisper_engine.py — lazy Whisper engine
========================================
The standalone whisper_transcriber loaded the turbo checkpoint at startup
and blocked on it. An MCP server must boot fast and stay responsive for
caption-only workflows, so the model (~1.6 GB) is loaded on the first
transcribe call and cached for the process lifetime. The cache is class-level
so repeated create_server() calls (tests, reloads) share one model instead of
reloading gigabytes each time.

Device selection and fp16 mirror the original: CUDA when available, CPU
otherwise (fp16 only on CUDA).
"""

from __future__ import annotations

import importlib.util
import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Whisper checkpoint names: model id -> filename within the whisper cache.
_MODEL_FILES = {
    "turbo": "large-v3-turbo.pt",
    "large-v2": "large-v2.pt",
    "large-v3": "large-v3.pt",
    "medium": "medium.pt",
    "small": "small.pt",
    "base": "base.pt",
    "tiny": "tiny.pt",
}

_MODEL_CACHE: dict[str, Any] = {}


def _weights_cached(model_name: str) -> bool:
    """Check whether the checkpoint already exists in the standard cache."""
    fname = _MODEL_FILES.get(model_name)
    if not fname:
        return False
    cache = Path.home() / ".cache" / "whisper"
    if Path(fname).is_absolute():
        cache = Path(fname)
    return (cache / fname).is_file()


def _has_wheel(spec_name: str) -> bool:
    """True when a module is importable (find_spec). Indirection for tests."""
    return importlib.util.find_spec(spec_name) is not None


def _load_runtime():
    """Import whisper + torch. Separate function so tests can stub it."""
    import torch  # noqa: PLC0415
    import whisper  # noqa: PLC0415

    return whisper, torch


class WhisperEngine:
    def __init__(self, model_name: str = "turbo") -> None:
        self.model_name = model_name
        self._device: str | None = None

    # -- model lifecycle -----------------------------------------------------
    def is_loaded(self) -> bool:
        return self.model_name in _MODEL_CACHE

    @property
    def device(self) -> str | None:
        return self._device

    def load(self) -> Any:
        """Load the whisper model, reusing a process-wide cache."""
        if self.model_name in _MODEL_CACHE:
            return _MODEL_CACHE[self.model_name]
        if not _has_wheel("whisper") or not _has_wheel("torch"):
            raise RuntimeError(
                "openai-whisper / torch not installed — run `uv sync` in the server project."
            )

        whisper, torch = _load_runtime()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        logger.info("[whisper] loading model %s on %s (first use, cached after)", self.model_name, device)
        try:
            model = whisper.load_model(self.model_name, device=device)
        except Exception as exc:  # noqa: BLE001
            logger.error("[whisper] model load failed: %s", exc, exc_info=True)
            raise RuntimeError("failed to load the whisper model — check the server log.") from exc
        _MODEL_CACHE[self.model_name] = model
        return model

    # -- status ----------------------------------------------------------------
    def status(self) -> dict:
        """Report runtime readiness without loading the model."""
        whisper_ok = _has_wheel("whisper")
        torch_ok = _has_wheel("torch")
        cuda = None
        if torch_ok and self._device is None:
            try:
                import torch

                cuda = bool(torch.cuda.is_available())
                self._device = "cuda" if cuda else "cpu"
            except Exception:  # noqa: BLE001
                cuda = None
        return {
            "model": self.model_name,
            "whisper_installed": whisper_ok,
            "torch_installed": torch_ok,
            "cuda_available": cuda,
            "device": self._device,
            "model_loaded": self.is_loaded(),
            "weights_cached": _weights_cached(self.model_name) if whisper_ok else None,
            "ffmpeg_available": shutil.which("ffmpeg") is not None,
        }

    # -- transcription --------------------------------------------------------
    def transcribe(self, path: str, language: str | None = None) -> dict:
        """
        Whisper-transcribe a local media file.

        Returns {"text", "language", "segments", "device", "fp16"} where
        segments are {"start", "end", "text"} — the same shape the original
        repo's `result["segments"]` rendered into VTT.
        """
        model = self.load()
        fp16 = self._device == "cuda"
        logger.info("[whisper] transcribing %s (lang=%s, fp16=%s)", path, language or "auto", fp16)
        result = model.transcribe(
            path,
            language=language,
            verbose=False,
            fp16=fp16,
        )
        if result is None:
            raise RuntimeError("whisper returned no result — check the server log.")
        segments = [
            {
                "start": s.get("start"),
                "end": s.get("end"),
                "text": (s.get("text") or "").strip(),
            }
            for s in (result.get("segments") or [])
        ]
        return {
            "text": (result.get("text") or "").strip(),
            "language": result.get("language"),
            "segments": segments,
            "device": self._device,
            "fp16": fp16,
        }


__all__ = ["WhisperEngine"]
