"""Deterministic clean-up, always on (design §4.6 `clean.py`, invariants 7-8).

Operates on raw ASR words/segments and never touches ``t_raw``. Every change is
returned as a ``text_edits`` entry so the transcript can say what happened.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .._vendor.asr_training import BIDI_AND_ZERO_WIDTH

_BIDI = {ord(c): None for c in BIDI_AND_ZERO_WIDTH}
CREDIT_PHRASES = ("תודה רבה", "להתראות", "תודה שצפיתם", "כתוביות", "תודה על הצפייה", "תרגום")
# Knesset plenary boilerplate: the ivrit.ai fine-tunes emit it on non-speech, e.g. trailing seconds
# where Silero hears no speech (design §2)
KNESSET_PHRASES = ("חברי הכנסת", "חברות וחברי הכנסת", "היושב-ראש", "היושב ראש", "אדוני היושב", "גברתי היושבת")
HALLUCINATION_PHRASES = CREDIT_PHRASES + KNESSET_PHRASES
MIN_MS_PER_WORD = 50    # 3+ words packed tighter than this cannot have been spoken
REPEAT_MIN = 4          # a 1-4-word n-gram repeated >= 4 times consecutively is a loop
LOW_WORD_PROB = 0.3


@dataclass
class Edit:
    stage: str
    kind: str
    before: str
    after: str
    reason: str
    segment: int | None = None
    word: int | None = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class CleanResult:
    edits: list[Edit] = field(default_factory=list)
    flags: dict[int, list[str]] = field(default_factory=dict)   # segment index -> flags


def strip_bidi(s: str) -> str:
    return s.translate(_BIDI)


def clean_word_text(t: str) -> str:
    """The rendered form of one word: bidi stripped, NFC, trimmed, leading dash removed."""
    t = unicodedata.normalize("NFC", strip_bidi(t)).strip()
    t = re.sub(r"^[-–—]+(?=\S)", "", t)
    return t


_PUNCT_SPACE = re.compile(r"\s+([,.;:!?])")
_MULTI_PUNCT = re.compile(r"([,.;:!?])\1{1,}")
_SPACES = re.compile(r"[ \t]+")


def clean_segment_text(text: str) -> str:
    text = unicodedata.normalize("NFC", strip_bidi(text)).strip()
    text = re.sub(r"^[-–—,]+\s*", "", text)          # subtitle-style leading dash / comma
    text = _PUNCT_SPACE.sub(r"\1", text)
    text = _MULTI_PUNCT.sub(r"\1", text)
    text = _SPACES.sub(" ", text)
    return text.strip()


def repetition_loop(words: list[str], *, min_repeats: int = REPEAT_MIN, max_n: int = 4) -> tuple[int, int, int] | None:
    """Find the first n-gram (n<=4) repeated >= min_repeats times back to back.

    Returns (start, n, repeats) or None. Verbatim Hebrew has real repeats, so
    the caller keeps two copies and FLAGS the segment; nothing is deleted.
    """
    L = len(words)
    for n in range(1, max_n + 1):
        i = 0
        while i + n * min_repeats <= L:
            gram = words[i:i + n]
            k = 1
            while i + (k + 1) * n <= L and words[i + k * n:i + (k + 1) * n] == gram:
                k += 1
            if k >= min_repeats:
                return i, n, k
            i += 1
    return None


def credit_phrase_suspect(text: str, *, speech_fraction: float, no_speech_prob: float | None,
                          duration_ms: int | None = None, n_words: int | None = None) -> bool:
    """A hallucinated segment - only ever when the audio agrees (VAD speech fraction < 0.2) and one of:
    a credit/Knesset boilerplate phrase, ``no_speech_prob > 0.5``, or 3+ words packed into less than
    ``MIN_MS_PER_WORD`` each (Whisper's DTW collapses words it cannot place onto one instant)."""
    if speech_fraction >= 0.2:
        return False
    stripped = strip_bidi(text).strip(" .!?,")
    if any(p in stripped for p in HALLUCINATION_PHRASES):
        return True
    if no_speech_prob is not None and no_speech_prob > 0.5:
        return True
    return bool(n_words is not None and duration_ms is not None and n_words >= 3
                and duration_ms < MIN_MS_PER_WORD * n_words)


def canonical_quotes(text: str, *, machine: bool) -> str:
    """Human formats keep gershayim/geresh; machine formats use ASCII."""
    if machine:
        return text.replace("״", '"').replace("׳", "'")
    # models write geresh as ASCII ' 94:2 - render the Hebrew mark inside words
    text = re.sub(r"(?<=[א-ת])'(?=[א-ת]|\s|$)", "׳", text)
    text = re.sub(r"(?<=[א-ת])\"(?=[א-ת])", "״", text)
    return text
