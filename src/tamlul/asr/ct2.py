"""faster-whisper / CTranslate2 backend - the v1 engine (design §4.2).

Called directly, never through ``ivrit-py`` (it drops decoder kwargs, #26).
The model is a LOCAL directory; ``HF_HUB_OFFLINE=1`` is already set by the
CLI so a hub id would fail fast. Audio arrives as a float32 slice of one plan
group with clip timestamps relative to the slice; the caller offsets times.
"""

from __future__ import annotations

import inspect
import logging
import os

import numpy as np

from .base import ModelRef, RawSegment, RawWord, merge_fragments
from .decode import DecodeOpts

log = logging.getLogger("tamlul.asr.ct2")


class Ct2Backend:
    name = "ct2"
    word_timestamps = True

    def __init__(self) -> None:
        self._model = None
        self._batched = None
        self.info: dict = {}

    def load(self, model: ModelRef, device: str, threads: int, compute_type: str) -> dict:
        from faster_whisper import WhisperModel
        if not os.path.isdir(model.dir):
            raise FileNotFoundError(f"model directory missing: {model.dir}")
        kwargs = dict(device=device, compute_type=compute_type, num_workers=1)
        if device == "cpu":
            kwargs["cpu_threads"] = int(threads)
        self._model = WhisperModel(model.dir, **kwargs)
        self._batched = None
        self.info = {"engine": "ct2", "model": model.id, "dir": model.dir, "revision": model.revision,
                     "commit": model.commit, "model_sha256": model.sha256, "device": device,
                     "compute_type": compute_type, "threads": int(threads)}
        return self.info

    def unload(self) -> None:
        self._model = None
        self._batched = None

    def _kwargs(self, opts: DecodeOpts) -> dict:
        assert self._model is not None
        sig = inspect.signature(self._model.transcribe).parameters
        wanted = {
            "language": opts.language, "task": opts.task, "beam_size": opts.beam_size, "best_of": opts.best_of,
            "patience": opts.patience, "temperature": list(opts.temperature),
            "compression_ratio_threshold": opts.compression_ratio_threshold,
            "log_prob_threshold": opts.log_prob_threshold, "no_speech_threshold": opts.no_speech_threshold,
            "condition_on_previous_text": opts.condition_on_previous_text,
            "initial_prompt": opts.initial_prompt, "hotwords": opts.hotwords,
            "suppress_blank": opts.suppress_blank, "suppress_tokens": list(opts.suppress_tokens),
            "word_timestamps": opts.word_timestamps, "without_timestamps": opts.without_timestamps,
            "max_initial_timestamp": opts.max_initial_timestamp,
            "prepend_punctuations": opts.prepend_punctuations, "append_punctuations": opts.append_punctuations,
            "hallucination_silence_threshold": opts.hallucination_silence_threshold,
            "repetition_penalty": opts.repetition_penalty, "no_repeat_ngram_size": opts.no_repeat_ngram_size,
            "vad_filter": False,
        }
        out = {}
        for k, v in wanted.items():
            if k in sig:
                out[k] = v
            else:
                log.warning("faster-whisper %s does not accept %s; option not applied", self._fw_version(), k)
        return out

    @staticmethod
    def _fw_version() -> str:
        try:
            from importlib import metadata
            return metadata.version("faster-whisper")
        except Exception:  # noqa: BLE001
            return "?"

    def transcribe(self, audio: np.ndarray, clips: list[tuple[float, float]] | None, opts: DecodeOpts) -> list[RawSegment]:
        assert self._model is not None, "load() first"
        kwargs = self._kwargs(opts)
        if clips and "clip_timestamps" in inspect.signature(self._model.transcribe).parameters:
            # 1.2.1 wants a flat list: start, end, start, end, ... (seconds, relative to the slice)
            flat: list[float] = []
            for s, e in clips:
                if e > s:
                    flat += [float(s), float(e)]
            if flat:
                kwargs["clip_timestamps"] = flat
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if opts.batched:
            from faster_whisper import BatchedInferencePipeline
            if self._batched is None:
                self._batched = BatchedInferencePipeline(self._model)
            for k in ("hallucination_silence_threshold", "max_initial_timestamp", "patience"):
                kwargs.pop(k, None)
            segments, _ = self._batched.transcribe(audio, batch_size=opts.batch_size, **kwargs)
        else:
            segments, _ = self._model.transcribe(audio, **kwargs)
        out: list[RawSegment] = []
        for s in segments:
            words = merge_fragments([RawWord(s=float(w.start), e=float(w.end), t=w.word, p=float(w.probability))
                                     for w in (s.words or [])])
            out.append(RawSegment(s=float(s.start), e=float(s.end), text=s.text, words=words,
                                  avg_logprob=float(s.avg_logprob) if s.avg_logprob is not None else None,
                                  no_speech_prob=float(s.no_speech_prob) if s.no_speech_prob is not None else None,
                                  compression_ratio=float(s.compression_ratio) if s.compression_ratio is not None else None,
                                  temperature=float(s.temperature) if s.temperature is not None else None,
                                  seek=int(s.seek) if getattr(s, "seek", None) is not None else None))
        return out
