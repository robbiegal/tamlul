"""Evaluation normalizers and a dependency-free WER (design §4.6, §4.15).

Two named profiles, and every reported number says which one it used:

* ``ivrit-lb``  - byte-for-byte the leaderboard's ``HebrewTextNormalizer``
  (vendored): strip bidi/zero-width, strip niqqud, delete ASCII quotes, then
  OpenAI's ``BasicTextNormalizer`` (Unicode M/S/P -> space, so gershayim
  U+05F4 splits ``ד״ר`` into two tokens while ``ד"ר`` becomes ``דר``).
* ``heb-strict`` - internal default: canonicalise ``״``/``"``, ``׳``/``'``,
  ``’``/``'`` and maqaf ``־``/``-`` FIRST; delete quotes inside words only;
  strip every Hebrew mark U+0591-05AF, U+05B0-05BD, U+05BF, U+05C1/2/4/5/7
  (never U+05BE/05C0/05C3/05C6, which are punctuation); bidi strip; NFKC;
  punctuation -> space; keep finals and digits. Flags: fold_finals,
  numerals_to_words (needs num2words), drop_fillers.
"""

from __future__ import annotations

import re
import unicodedata

from .._vendor.asr_training import BIDI_AND_ZERO_WIDTH, HebrewTextNormalizer

_lb = HebrewTextNormalizer()

FINALS = str.maketrans({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})
HEBREW_MARKS = "".join(chr(c) for c in list(range(0x0591, 0x05B0)) + list(range(0x05B0, 0x05BE)) + [0x05BF, 0x05C1, 0x05C2, 0x05C4, 0x05C5, 0x05C7])
_MARKS_TABLE = {ord(c): None for c in HEBREW_MARKS}
_BIDI_TABLE = {ord(c): None for c in BIDI_AND_ZERO_WIDTH}
FILLERS = {"אה", "אהה", "אממ", "אמ", "מממ", "אהם", "הממ", "אממם"}

_CANON = {"״": '"', "׳": "'", "’": "'", "‘": "'", "“": '"', "”": '"', "־": "-", "–": "-", "—": "-"}
_QUOTE_IN_WORD = re.compile(r"(?<=\w)[\"'](?=\w)")


def ivrit_lb(text: str) -> str:
    return _lb(text)


def heb_strict(text: str, *, fold_finals: bool = False, numerals_to_words: bool = False,
               drop_fillers: bool = False) -> str:
    for k, v in _CANON.items():
        text = text.replace(k, v)
    text = _QUOTE_IN_WORD.sub("", text)
    text = text.translate(_MARKS_TABLE).translate(_BIDI_TABLE)
    text = unicodedata.normalize("NFKC", text)
    text = "".join(" " if unicodedata.category(c)[0] in "MSP" else c for c in text)
    text = text.lower()
    if fold_finals:
        text = text.translate(FINALS)
    if numerals_to_words:
        text = re.sub(r"\d+", lambda m: _num_to_words(m.group(0)), text)
    toks = text.split()
    if drop_fillers:
        toks = [t for t in toks if t not in FILLERS]
    return " ".join(toks)


def _num_to_words(digits: str) -> str:
    try:
        from num2words import num2words
        return num2words(int(digits), lang="he")
    except Exception:  # noqa: BLE001 - optional dependency
        return digits


PROFILES = {"ivrit-lb": ivrit_lb, "heb-strict": heb_strict}


def normalize(text: str, profile: str = "heb-strict", **flags) -> str:
    fn = PROFILES[profile]
    return fn(text, **flags) if profile == "heb-strict" else fn(text)


# ------------------------------------------------------------------ WER

def edit_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    """(substitutions, deletions, insertions, hits) by Levenshtein alignment."""
    n, m = len(ref), len(hyp)
    # dp[i][j] = (cost, s, d, i, h)
    prev = [(j, 0, 0, j, 0) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0, 0)]
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                c, s, d, ins, h = prev[j - 1]
                cur.append((c, s, d, ins, h + 1))
            else:
                sub = prev[j - 1]
                dele = prev[j]
                inse = cur[j - 1]
                best = min((sub[0] + 1, 0, sub), (dele[0] + 1, 1, dele), (inse[0] + 1, 2, inse), key=lambda x: (x[0], x[1]))
                c, kind, src = best
                _, s, d, ins, h = src
                if kind == 0:
                    cur.append((c, s + 1, d, ins, h))
                elif kind == 1:
                    cur.append((c, s, d + 1, ins, h))
                else:
                    cur.append((c, s, d, ins + 1, h))
        prev = cur
    c, s, d, ins, h = prev[m]
    return s, d, ins, h


def wer(refs: list[str], hyps: list[str]) -> dict:
    """Corpus-level (micro) WER exactly as the leaderboard: totals over the whole list."""
    S = D = Ins = H = N = 0
    for r, h in zip(refs, hyps):
        rt, ht = r.split(), h.split()
        s, d, i, hit = edit_ops(rt, ht)
        S += s
        D += d
        Ins += i
        H += hit
        N += len(rt)
    return {"wer": (S + D + Ins) / N if N else 0.0, "S": S, "D": D, "I": Ins, "H": H, "N": N}


def cer(refs: list[str], hyps: list[str], *, space_insensitive: bool = False) -> float:
    E = N = 0
    for r, h in zip(refs, hyps):
        if space_insensitive:
            r, h = r.replace(" ", ""), h.replace(" ", "")
        s, d, i, _ = edit_ops(list(r), list(h))
        E += s + d + i
        N += len(r)
    return E / N if N else 0.0
