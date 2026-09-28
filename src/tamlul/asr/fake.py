"""A deterministic backend for tests and CI: no model, no audio understanding.

Emits one segment per clip with synthetic Hebrew words spaced evenly, so the
plan/attribute/annotate/export stages can be exercised end-to-end without a
1.6 GB download (design §4.19).
"""

from __future__ import annotations

import numpy as np

from .base import ModelRef, RawSegment, RawWord
from .decode import DecodeOpts

WORDS = ["שלום", "מה", "נשמע", "אני", "חושב", "שזה", "רעיון", "טוב", "בסדר", "כן", "לא", "נכון", "אז", "בואו", "נתחיל"]


class FakeBackend:
    name = "fake"
    word_timestamps = True

    def __init__(self) -> None:
        self.info: dict = {}

    def load(self, model: ModelRef, device: str, threads: int, compute_type: str) -> dict:
        self.info = {"engine": "fake", "model": model.id, "dir": model.dir, "revision": model.revision,
                     "commit": model.commit, "model_sha256": model.sha256, "device": device,
                     "compute_type": compute_type, "threads": threads}
        return self.info

    def unload(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray, clips: list[tuple[float, float]] | None, opts: DecodeOpts) -> list[RawSegment]:
        n = len(audio) / 16000.0
        clips = clips or [(0.0, n)]
        out = []
        k = 0
        for s, e in clips:
            dur = max(0.0, min(e, n) - s)
            nwords = max(1, int(dur / 0.4))
            step = dur / nwords if nwords else dur
            words = []
            for j in range(nwords):
                ws = s + j * step
                words.append(RawWord(s=round(ws, 3), e=round(min(ws + step * 0.8, e), 3), t=" " + WORDS[(k + j) % len(WORDS)], p=0.9))
            k += nwords
            text = "".join(w.t for w in words) + "."
            out.append(RawSegment(s=round(s, 3), e=round(e, 3), text=text, words=words, avg_logprob=-0.2,
                                  no_speech_prob=0.01, compression_ratio=1.5, temperature=0.0))
        return out
