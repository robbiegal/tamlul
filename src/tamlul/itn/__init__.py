"""Hebrew inverse text normalisation (design §4.6 `itn.py`).

A pure-Python port of the NeMo-text-processing Hebrew ITN grammar
(Apache-2.0; `pynini` has no Windows wheel). The NeMo test files under
``data/nemo/`` are the acceptance suite. Operates on ``words[]`` so that a
multi-word number collapses into the first word's ``t`` and the others render
empty; ``t_raw`` is never changed.
"""

from __future__ import annotations

from .grammar import itn_text, itn_tokens  # noqa: F401


def itn_words(words: list[dict], edits: list[dict]) -> int:
    """Apply ITN in place over the rendered tokens. Returns the number of changes."""
    from ..text.clean import Edit
    toks = [w["t"] for w in words]
    new = itn_tokens(toks)
    n = 0
    for w, before, after in zip(words, toks, new):
        if before != after:
            edits.append(Edit("text", "itn", before, after, "inverse text normalisation", word=w["i"]).as_dict())
            w["t"] = after
            w["edited"] = True
            n += 1
    return n
