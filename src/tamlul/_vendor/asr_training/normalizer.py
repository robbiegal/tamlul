# derived from https://github.com/ivrit-ai/asr-training @ 760572069c3a, evaluate_model.py, MIT License
# Copyright (c) 2024 ivrit.ai (licence text: LICENSE next to this file). Modifications: only the normalizer pieces are kept; `hebrew.Hebrew.no_niqqud`
# is reproduced locally (same code points) so the core install needs no extra package.
# Upstream file sha256: f2f135d3ab27f38def04122d9c976d8531c4c4b144820c836d69b98eef0229b4
"""The ivrit.ai leaderboard normalizer, byte-for-byte (design section 4.6, `ivrit-lb`)."""

from __future__ import annotations

from ..openai_whisper_normalizer import BasicTextNormalizer

# The same code points `hebrew.chars.NIQQUD_CHARS` strips (hebrew==0.8.1):
# U+05B0..U+05BC, U+05C1, U+05C2, U+05C4, U+05C5, U+05C7. Meteg (05BD), rafe (05BF)
# and cantillation (0591..05AF) are NOT stripped - that is the leaderboard's behaviour.
NIQQUD = "".join(chr(c) for c in list(range(0x05B0, 0x05BD)) + [0x05C1, 0x05C2, 0x05C4, 0x05C5, 0x05C7])
_NIQQUD_TABLE = {ord(c): None for c in NIQQUD}

BIDI_AND_ZERO_WIDTH = (
    "\u061C"                      # Arabic letter mark
    "\u200b\u200C\u200D"          # zero-width space, non-joiner, joiner
    "\u200E\u200F"                # LTR and RTL marks
    "\u202A\u202B\u202C\u202D\u202E"  # LTR/RTL embedding, pop, override
    "\u2066\u2067\u2068\u2069"      # isolate controls
    "\uFEFF"                      # zero-width no-break space
)
_BIDI_TABLE = {ord(c): None for c in BIDI_AND_ZERO_WIDTH}


def clean_some_unicode_from_text(text: str) -> str:
    return text.translate(_BIDI_TABLE)


def remove_niqqud(text: str) -> str:
    try:  # byte-for-byte with the leaderboard when the library is present
        from hebrew import Hebrew  # type: ignore
        return Hebrew(text).no_niqqud().string
    except ImportError:
        return text.translate(_NIQQUD_TABLE)


class HebrewTextNormalizer:
    def __init__(self) -> None:
        self.whisper_normalizer = BasicTextNormalizer()

    def __call__(self, text: str) -> str:
        text = clean_some_unicode_from_text(text)
        text = remove_niqqud(text)
        text = text.replace('"', "").replace("'", "")
        return self.whisper_normalizer(text)
