"""Stage 8: ITN and glossary/participant correction on the rendered text (design §4.6).

``words[].t_raw`` is never touched; ``t`` is re-rendered; every change lands in
``text_edits[]``. ITN follows the profile (on for meeting/interview/voice-note,
off for ca/legal/--verbatim).
"""

from __future__ import annotations

import pathlib

from ..config import Settings
from ..document import render


def apply_text_layers(doc: dict, settings: Settings, *, run_dir: pathlib.Path | None = None) -> None:
    words = doc["words"]
    if settings.itn_enabled():
        from ..itn import itn_words
        n = itn_words(words, doc["text_edits"])
        doc["pipeline"]["itn"] = {"enabled": True, "changes": n}
    else:
        doc["pipeline"]["itn"] = {"enabled": False}
    gl = settings.glossary or ""
    if gl and pathlib.Path(gl).is_file():
        from .correct import apply_glossary, load_glossary
        glossary = load_glossary(pathlib.Path(gl))
        flagged = apply_glossary(doc, glossary)
        doc["pipeline"]["glossary"] = {"path": gl, "terms": len(glossary), "flagged": len(flagged)}
    else:
        from .correct import detect_code_switch_candidates
        flagged = detect_code_switch_candidates(doc)
        doc["pipeline"]["glossary"] = {"path": None, "terms": 0, "flagged": len(flagged)}
    doc.setdefault("candidates", {})["code_switch"] = flagged
    # re-render segment text from words so ITN/corrections show in every export
    for seg in doc["segments"]:
        if seg.get("w"):
            seg["text"] = render(words, seg["w"][0], seg["w"][1])
