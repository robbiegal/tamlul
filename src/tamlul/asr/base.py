from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from .decode import DecodeOpts


@dataclass
class RawWord:
    s: float            # seconds, in the caller's time base
    e: float
    t: str              # raw token text as the engine produced it (may carry bidi marks, leading space)
    p: float            # word probability


def merge_fragments(words: list[RawWord]) -> list[RawWord]:
    """Re-attach word fragments that Whisper's splitter cut inside a word.

    faster-whisper starts a new word at every space *and at every punctuation token*, so a Hebrew
    geresh word (``צ'יפס``, ``ג'ירפה``) comes back as ``צ`` + ``'יפס`` and a thousands separator
    splits ``1,000``. Only a word that begins with whitespace is a real word start; anything else is
    the tail of the previous word. Concatenating ``t`` restores the segment text exactly (the
    phase-4 ablation put the split at -0.25 WER points on eval-whatsapp).
    """
    out: list[RawWord] = []
    for w in words:
        if out and w.t and not w.t[0].isspace():
            prev = out[-1]
            out[-1] = RawWord(s=prev.s, e=max(prev.e, w.e), t=prev.t + w.t, p=min(prev.p, w.p))
        else:
            out.append(w)
    return out


@dataclass
class RawSegment:
    s: float
    e: float
    text: str
    words: list[RawWord] = field(default_factory=list)
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    temperature: float | None = None
    seek: int | None = None


@dataclass
class ModelRef:
    id: str
    dir: str
    revision: str
    commit: str
    sha256: str


@runtime_checkable
class AsrBackend(Protocol):
    name: str
    word_timestamps: bool

    def load(self, model: ModelRef, device: str, threads: int, compute_type: str) -> dict: ...

    def transcribe(self, audio: np.ndarray, clips: list[tuple[float, float]] | None, opts: DecodeOpts) -> list[RawSegment]: ...

    def unload(self) -> None: ...
