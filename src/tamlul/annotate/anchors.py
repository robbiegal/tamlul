"""Evidence anchors (design §4.8, invariant 12).

An anchor is ``{"w": [i0, i1], "quote": "...", "s": ms, "e": ms}``: a raw word
index range, the verbatim rendering of that range, and its time span. Turn
ids are derived and informational. After speaker edits the anchors are
re-resolved (the words are immutable, so ``w`` stays valid; only the derived
``turn`` field and the ``spans_speakers`` flag change).
"""

from __future__ import annotations

from ..document import render

ANCHORED_LISTS = ("topics", "decisions", "action_items", "questions", "key_quotes", "speaker_names",
                  "term_restorations", "boundary_corrections", "speaker_summaries")


def iter_items(annotations: dict):
    for key in ANCHORED_LISTS:
        for k, item in enumerate(annotations.get(key, []) or []):
            yield key, k, item
    summ = annotations.get("summary")
    if isinstance(summ, dict):
        yield "summary", 0, summ


def check_anchor(doc: dict, anchor: dict, *, tolerance_ms: int = 1500) -> str | None:
    words = doc["words"]
    w = anchor.get("w")
    if not (isinstance(w, list) and len(w) == 2 and all(isinstance(x, int) for x in w)):
        return "anchor.w must be [i0, i1] integers"
    i0, i1 = w
    if i0 < 0 or i1 >= len(words) or i0 > i1:
        return f"anchor.w {w} out of range (0..{len(words)-1})"
    quote = anchor.get("quote", "")
    rendered = render(words, i0, i1)
    if not isinstance(quote, str) or _norm(quote) != _norm(rendered):
        return f"anchor.quote does not match words[{i0}:{i1}] ({rendered[:60]!r})"
    s, e = anchor.get("s"), anchor.get("e")
    if s is not None and abs(int(s) - words[i0]["s"]) > tolerance_ms:
        return f"anchor.s {s} is not the start of words[{i0}] ({words[i0]['s']})"
    if e is not None and abs(int(e) - words[i1]["e"]) > tolerance_ms:
        return f"anchor.e {e} is not the end of words[{i1}] ({words[i1]['e']})"
    return None


def _norm(s: str) -> str:
    return " ".join(s.replace("\u200F", "").split())


def resolve_anchor(doc: dict, anchor: dict) -> dict:
    """Fill the derived fields: exact times, turn id, whether the range spans speakers."""
    words = doc["words"]
    i0, i1 = anchor["w"]
    out = dict(anchor)
    out["s"], out["e"] = words[i0]["s"], words[i1]["e"]
    spks = {words[i].get("spk") for i in range(i0, i1 + 1)}
    out["spans_speakers"] = len(spks) > 1
    out["spk"] = next(iter(spks)) if len(spks) == 1 else None
    out["turn"] = None
    for t in doc.get("turns", []):
        if t["w"][0] <= i0 and i1 <= t["w"][1]:
            out["turn"] = t["id"]
            break
    return out


def validate_anchors(doc: dict) -> list[str]:
    errs = []
    for key, k, item in iter_items(doc.get("annotations") or {}):
        for j, a in enumerate(item.get("evidence", []) or []):
            e = check_anchor(doc, a)
            if e:
                errs.append(f"annotations.{key}[{k}].evidence[{j}]: {e}")
    return errs


def reresolve_all(doc: dict) -> int:
    n = 0
    for _key, _k, item in iter_items(doc.get("annotations") or {}):
        ev = item.get("evidence") or []
        item["evidence"] = [resolve_anchor(doc, a) if check_anchor(doc, a) is None else a for a in ev]
        n += len(ev)
    return n
