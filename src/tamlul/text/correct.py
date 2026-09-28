"""Glossary / participant post-correction and code-switch candidates (design §4.6 `correct.py`).

Lanes:
 (a) whole-token canonical+variants replacement from glossary.yaml (deterministic);
 (b) confidence-gated (< 0.55) phonetically weighted fuzzy match with clitic
     splitting - constants verbatim from hebrew_correct.py @ 7ca0992
     (KoganTheDev/hebrew-audio-transcriber, MIT), a-priori, validated not tuned;
 (c) uroman consonant-skeleton candidates for Latin glossary terms;
 (d) flagged words only -> Claude restoration with evidence anchors.

Never digits in the glossary (design invariant 5).
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field

from .clean import Edit

# --- constants vendored from hebrew_correct.py @ 7ca0992 (MIT) ---
CONFIDENCE_GATE = 0.55
CONFUSION_GROUPS = [("אהע", 0.25), ("כק", 0.25), ("טת", 0.25), ("סשצ", 0.35), ("בו", 0.35), ("יא", 0.4), ("גז", 0.5)]
MAX_RELATIVE_DISTANCE = 0.34
MARGIN = 0.5
CLITICS = ("וש", "ומ", "וב", "ול", "וה", "וכ", "ש", "ו", "ב", "ל", "מ", "ה", "כ")
_SUB_COST: dict[tuple[str, str], float] = {}
for group, cost in CONFUSION_GROUPS:
    for a in group:
        for b in group:
            if a != b:
                _SUB_COST[(a, b)] = cost
FINAL_MAP = str.maketrans({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})


@dataclass
class Term:
    canonical: str
    variants: list[str] = field(default_factory=list)
    latin: str = ""
    kind: str = "term"
    source: str = "glossary"


def load_glossary(path: pathlib.Path) -> list[Term]:
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = data.get("terms", data) if isinstance(data, dict) else data
    out = []
    for it in items or []:
        if isinstance(it, str):
            it = {"canonical": it}
        canon = str(it.get("canonical", "")).strip()
        if not canon or any(c.isdigit() for c in canon):
            continue  # invariant 5: never digits
        out.append(Term(canonical=canon, variants=[str(v) for v in it.get("variants", []) if not any(c.isdigit() for c in str(v))],
                        latin=str(it.get("latin", "")), kind=str(it.get("kind", "term")),
                        source=str(it.get("source", "glossary"))))
    return out


def weighted_distance(a: str, b: str) -> float:
    a, b = a.translate(FINAL_MAP), b.translate(FINAL_MAP)
    n, m = len(a), len(b)
    prev = [float(j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [float(i)]
        for j in range(1, m + 1):
            sub = prev[j - 1] + (0.0 if a[i - 1] == b[j - 1] else _SUB_COST.get((a[i - 1], b[j - 1]), 1.0))
            cur.append(min(sub, prev[j] + 1.0, cur[j - 1] + 1.0))
        prev = cur
    return prev[m]


def split_clitic(word: str) -> tuple[str, str]:
    for c in CLITICS:
        if word.startswith(c) and len(word) - len(c) >= 2:
            return c, word[len(c):]
    return "", word


def _is_hebrew(t: str) -> bool:
    return any("א" <= c <= "ת" for c in t)


def _is_latin(t: str) -> bool:
    return any("a" <= c.lower() <= "z" for c in t)


def match_term(word: str, terms: list[Term]) -> tuple[Term, float, float] | None:
    """Best (term, distance, second-best distance) for the confidence-gated lane."""
    core = word.strip(".,!?;:\"'״׳()[]")
    prefix, stem = split_clitic(core)
    cands = []
    for t in terms:
        for form in [t.canonical, *t.variants]:
            if not _is_hebrew(form):
                continue
            for probe in ({stem, core} if prefix else {core}):
                d = weighted_distance(probe, form)
                rel = d / max(1, len(form))
                cands.append((rel, t, form))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    best = cands[0]
    second = next((c for c in cands[1:] if c[1] is not best[1]), None)
    if best[0] > MAX_RELATIVE_DISTANCE:
        return None
    if second is not None and (second[0] - best[0]) < MARGIN * best[0] and best[0] > 0:
        return None
    return best[1], best[0], (second[0] if second else 1.0)


def apply_glossary(doc: dict, terms: list[Term]) -> list[dict]:
    """Lane (a) exact + lane (b) gated fuzzy. Returns flagged code-switch candidates for lane (d)."""
    words = doc["words"]
    exact: dict[str, Term] = {}
    for t in terms:
        for form in [t.canonical, *t.variants]:
            exact[form.translate(FINAL_MAP)] = t
    flagged: list[dict] = []
    for w in words:
        core = w["t"].strip(".,!?;:\"'״׳()[]")
        trail = w["t"][len(core):] if w["t"].endswith(tuple(".,!?;:")) else ""
        if not core:
            continue
        prefix, stem = split_clitic(core)
        key_full, key_stem = core.translate(FINAL_MAP), stem.translate(FINAL_MAP)
        t = exact.get(key_full) or (exact.get(key_stem) if prefix else None)
        if t is not None:
            target = _render_term(t, prefix if t is exact.get(key_stem) and not exact.get(key_full) else "")
            if target != core:
                _replace(doc, w, target + trail, "glossary_exact", f"whole-token match for {t.canonical}")
            continue
        if w["p"] >= CONFIDENCE_GATE:
            continue
        m = match_term(core, terms)
        if m:
            t, rel, _ = m
            target = _render_term(t, prefix)
            if target != core:
                _replace(doc, w, target + trail, "glossary_fuzzy",
                         f"p={w['p']:.2f} < {CONFIDENCE_GATE}, weighted distance {rel:.2f} to {t.canonical}")
        elif _looks_transliterated(core):
            flagged.append({"w": [w["i"], w["i"]], "text": core, "p": w["p"], "reason": "low confidence, skeleton",
                            "candidates": _skeleton_candidates(core, terms)})
    return flagged


def _render_term(t: Term, prefix: str) -> str:
    target = t.latin if t.latin else t.canonical
    if prefix:
        return f"{prefix}-{target}" if _is_latin(target) else prefix + target
    return target


def _replace(doc: dict, w: dict, new: str, kind: str, reason: str) -> None:
    doc["text_edits"].append(Edit("text", kind, w["t"], new, reason, word=w["i"]).as_dict())
    w["t"] = new
    w["edited"] = True


# ------------------------------------------------ lane (c): uroman skeletons

_VOWELS = set("aeiou")
_HEB_SKELETON_DROP = set("אהוי")


def _looks_transliterated(word: str) -> bool:
    """Cheap heuristic for 'probably an English term in Hebrew letters': long, no clitic sense, low p handled by caller."""
    return _is_hebrew(word) and len(word) >= 4 and not word.endswith(("ים", "ות"))


def hebrew_skeleton(word: str) -> str:
    return "".join(c for c in word.translate(FINAL_MAP) if c not in _HEB_SKELETON_DROP)


def latin_skeleton(term: str) -> str:
    """uroman-style consonant skeleton of a Latin term mapped onto Hebrew letters."""
    t = term.lower()
    t = re.sub(r"[^a-z]", "", t)
    mapping = {"b": "ב", "c": "ק", "d": "ד", "f": "פ", "g": "ג", "h": "", "j": "ג", "k": "ק", "l": "ל", "m": "מ", "n": "נ",
               "p": "פ", "q": "ק", "r": "ר", "s": "ס", "t": "ט", "v": "ו", "w": "ו", "x": "קס", "y": "", "z": "ז"}
    out = "".join(mapping.get(c, "") for c in t if c not in _VOWELS)
    return out.replace("ו", "")


def _skeleton_candidates(word: str, terms: list[Term]) -> list[str]:
    sk = hebrew_skeleton(word)
    out = []
    for t in terms:
        if t.latin and latin_skeleton(t.latin) == sk:
            out.append(t.latin)
    return out[:5]


def detect_code_switch_candidates(doc: dict) -> list[dict]:
    """Without a glossary: flag low-confidence Hebrew words that look transliterated, for Claude (lane d)."""
    out = []
    for w in doc["words"]:
        core = w["t"].strip(".,!?;:\"'״׳()[]")
        if w["p"] < CONFIDENCE_GATE and _looks_transliterated(core):
            out.append({"w": [w["i"], w["i"]], "text": core, "p": w["p"], "reason": "low confidence, skeleton", "candidates": []})
    return out
