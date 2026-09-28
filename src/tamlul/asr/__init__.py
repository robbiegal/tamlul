"""ASR backends. v1 ships one engine (CTranslate2 via faster-whisper); the
Protocol exists so v1.1 engines can be added behind the calibration gate
without touching the pipeline (design §4.2)."""

from __future__ import annotations

from .base import AsrBackend, RawSegment, RawWord
from .decode import DecodeOpts

__all__ = ["AsrBackend", "DecodeOpts", "RawSegment", "RawWord", "get_backend"]


def get_backend(name: str) -> AsrBackend:
    if name == "ct2":
        from .ct2 import Ct2Backend
        return Ct2Backend()
    if name == "fake":
        from .fake import FakeBackend
        return FakeBackend()
    raise KeyError(f"unknown ASR backend {name!r} (v1 ships 'ct2')")
