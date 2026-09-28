"""Deterministic annotation and anchors on synthetic timelines (design §4.7, §4.8)."""

from __future__ import annotations

from tamlul.annotate.anchors import check_anchor, resolve_anchor, validate_anchors
from tamlul.annotate.deterministic import annotate_deterministic
from tamlul.document import new_document


def _doc(words_spec):
    """words_spec: list of (start_ms, end_ms, text, spk)."""
    doc = new_document({"path": "x.wav", "sha256": "", "duration_ms": words_spec[-1][1] + 500}, {}, "meeting")
    for k, (s, e, t, spk) in enumerate(words_spec):
        doc["words"].append({"i": k, "s": s, "e": e, "t_raw": t, "t": t, "p": 0.9, "spk": spk, "spk_conf": 0.9,
                             "overlap": False, "spk2": []})
    doc["segments"].append({"id": 0, "s": words_spec[0][0], "e": words_spec[-1][1], "spk": None, "text": "",
                            "w": [0, len(words_spec) - 1], "asr": {}, "edited": False, "flags": []})
    spks = sorted({w[3] for w in words_spec if w[3]})
    doc["speakers"] = [{"id": s, "label": s, "diar_label": s, "name": "", "name_source": "none", "stats": {}} for s in spks]
    return doc


def test_utterances_turns_events_and_stats():
    doc = _doc([
        (0, 300, "שלום", "S1"), (350, 700, "לכולם.", "S1"),
        (1900, 2200, "מה", "S1"), (2250, 2600, "שלומכם?", "S1"),      # 1.2 s within-speaker pause before
        (2650, 2900, "כן", "S2"),                                       # backchannel inside S1's floor
        (3000, 3400, "אני", "S1"), (3450, 3900, "מתחיל.", "S1"),
        (5500, 5900, "אוקיי", "S2"), (5950, 6300, "בוא", "S2"), (6350, 6800, "נתחיל.", "S2"),
    ])
    annotate_deterministic(doc, profile="meeting")
    assert len(doc["utterances"]) == 5
    assert doc["utterances"][1]["is_question"] is True
    turns = doc["turns"]
    assert [t["spk"] for t in turns] == ["S1", "S2"]
    assert turns[1]["transition"] == "gap" and turns[1]["fto_ms"] == 5500 - 3900
    kinds = {e["kind"] for e in doc["events"]}
    assert "pause" in kinds and "gap" in kinds
    assert doc["candidates"]["backchannels"][0]["text"] == "כן"
    st = {s["id"]: s["stats"] for s in doc["speakers"]}
    assert st["S1"]["turns"] == 1 and st["S1"]["questions"] == 1 and st["S2"]["backchannels"] == 1
    assert abs(st["S1"]["share"] + st["S2"]["share"] - 1.0) < 1e-6
    assert doc["stats"]["thresholds"]["gap_ms"] == 200
    assert turns[0]["tid"].startswith("S1:")


def test_anchor_validation_and_reresolution():
    doc = _doc([(0, 300, "אנחנו", "S1"), (350, 700, "נפגשים", "S1"), (750, 1100, "מחר", "S2")])
    annotate_deterministic(doc, profile="meeting")
    good = {"w": [0, 1], "quote": "אנחנו נפגשים", "s": 0, "e": 700}
    assert check_anchor(doc, good) is None
    assert "does not match" in check_anchor(doc, {"w": [0, 1], "quote": "משהו אחר", "s": 0, "e": 700})
    assert "out of range" in check_anchor(doc, {"w": [0, 9], "quote": "x"})
    assert "must be" in check_anchor(doc, {"w": "0-1", "quote": "x"})
    r = resolve_anchor(doc, good)
    assert r["spk"] == "S1" and r["spans_speakers"] is False and r["turn"] == 0
    r2 = resolve_anchor(doc, {"w": [1, 2], "quote": "נפגשים מחר"})
    assert r2["spans_speakers"] is True and r2["spk"] is None
    doc["annotations"] = {"decisions": [{"text": "x", "evidence": [good, {"w": [0, 0], "quote": "לא", "s": 0, "e": 300}]}]}
    errs = validate_anchors(doc)
    assert len(errs) == 1 and "evidence[1]" in errs[0]
