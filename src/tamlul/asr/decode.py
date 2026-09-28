"""The Hebrew decoding contract (design §4.2, invariants 2-5, 8).

Mapped onto every backend; a backend that cannot honour a knob logs it, never
silently drops it. The values are not folklore: each was measured on Hebrew
decoding windows during design (see the invariants in CONTRIBUTING.md).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DecodeOpts:
    language: str = "he"
    task: str = "transcribe"
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    temperature: tuple[float, ...] = (0.0, 0.2, 0.4)
    compression_ratio_threshold: float = 2.8
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = False
    initial_prompt: str | None = None
    hotwords: str | None = None
    suppress_blank: bool = True
    suppress_tokens: tuple[int, ...] = (-1,)
    word_timestamps: bool = True
    without_timestamps: bool = False
    max_initial_timestamp: float = 1.0
    prepend_punctuations: str = "\"'“¿([{-"
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、…"
    hallucination_silence_threshold: float = 2.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    batched: bool = False
    batch_size: int = 4

    def as_dict(self) -> dict:
        d = asdict(self)
        d["temperature"] = list(self.temperature)
        d["suppress_tokens"] = list(self.suppress_tokens)
        return d

    def validate(self) -> list[str]:
        """The invariants, enforced. Returns a list of violations (empty = ok)."""
        v = []
        if self.language != "he":
            v.append("language must be 'he' (invariant 2)")
        if self.condition_on_previous_text:
            v.append("condition_on_previous_text must be False (invariant 3)")
        if self.compression_ratio_threshold < 2.8:
            v.append("compression_ratio_threshold must be >= 2.8 (invariant 4)")
        if any(t > 0.4 for t in self.temperature):
            v.append("temperature ladder must stay <= 0.4 (invariant 4)")
        for name, val in (("initial_prompt", self.initial_prompt), ("hotwords", self.hotwords)):
            if val and (any(c.isdigit() for c in val) or any("a" <= c.lower() <= "z" for c in val)):
                v.append(f"{name} must not contain digits or Latin script (invariant 5)")
        if self.no_repeat_ngram_size != 0:
            v.append("no_repeat_ngram_size must be 0 (invariant 8)")
        if self.batched and self.batch_size > 4:
            v.append("batch_size must be <= 4 (design §2.3)")
        return v


def quality_max_opts(base: DecodeOpts) -> DecodeOpts:
    """--quality max changes the model, not the contract."""
    return base
