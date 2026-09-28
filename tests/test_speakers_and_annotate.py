"""Speaker overlay commands and the Claude annotation contract on a fake-backend run."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("faster_whisper", reason="the plan stage needs faster-whisper's bundled Silero VAD")

from tamlul.annotate.claude import apply_annotations, render_input  # noqa: E402
from tamlul.cli import main  # noqa: E402
from tamlul.diarize import DiarResult  # noqa: E402
from tamlul.diarize.attribution import attribute  # noqa: E402
from tamlul.jobs import RunDir  # noqa: E402
from tamlul.paths import DataLayout  # noqa: E402


def _fake_run(data_dir, wav, monkeypatch, capsys) -> RunDir:
    monkeypatch.setenv("TAMLUL_FAKE_ASR", "1")
    assert main(["--data", str(data_dir), "transcribe", str(wav), "--diarization", "off"]) == 0
    run_id = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("run: "))[5:]
    return RunDir.open(DataLayout(data_dir), run_id)


def _diarize_fake(doc: dict) -> None:
    """Two speakers alternating by plan group: exact timelines built from the words themselves."""
    words = doc["words"]
    turns = []
    cur, spk = None, "A"
    for w in words:
        if cur and w["s"] - cur["e"] > 600:
            turns.append(cur)
            spk = "B" if spk == "A" else "A"
            cur = None
        if cur is None:
            cur = {"s": w["s"], "e": w["e"], "spk": spk}
        cur["e"] = w["e"]
    if cur:
        turns.append(cur)
    res = DiarResult(turns=turns, exclusive=list(turns), labels=["A", "B"], info={"engine": "test"})
    attribute(doc, res)
    from tamlul.annotate.deterministic import annotate_deterministic
    annotate_deterministic(doc, profile="meeting")


def test_attribution_then_speakers_commands(data_dir, synthetic_wav, monkeypatch, capsys):
    rd = _fake_run(data_dir, synthetic_wav, monkeypatch, capsys)
    doc = rd.read_transcript()
    _diarize_fake(doc)
    assert {s["id"] for s in doc["speakers"]} == {"S1", "S2"}
    assert all(w["spk"] in ("S1", "S2") for w in doc["words"])
    assert doc["turns"] and doc["turns"][0]["spk"] in ("S1", "S2")
    rd.write_transcript(doc)
    d = str(data_dir)
    assert main(["--data", d, "speakers", rd.id, "show"]) == 0
    assert main(["--data", d, "speakers", rd.id, "rename", "S2", "יוסי"]) == 0
    doc = rd.read_transcript()
    assert next(s for s in doc["speakers"] if s["id"] == "S2")["name"] == "יוסי"
    assert doc["review"]["edited"] is True and rd.edits_jsonl.is_file()
    assert main(["--data", d, "speakers", rd.id, "merge", "S2", "into", "S1"]) == 0
    doc = rd.read_transcript()
    assert [s["id"] for s in doc["speakers"]] == ["S1"]
    assert all(w["spk"] == "S1" for w in doc["words"])
    assert main(["--data", d, "validate", rd.id]) == 0
    assert main(["--data", d, "speakers", rd.id, "reassign", "T0", "S9"]) == 3   # unknown speaker


def test_annotation_contract_apply(data_dir, synthetic_wav, monkeypatch, capsys):
    rd = _fake_run(data_dir, synthetic_wav, monkeypatch, capsys)
    doc = rd.read_transcript()
    _diarize_fake(doc)
    rd.write_transcript(doc)
    text = render_input(doc, participants=["דנה", "יוסי"], glossary=None)
    assert "T0 [" in text and "w0-" in text
    t0 = doc["turns"][0]
    words = doc["words"]
    # the fake backend cycles 15 words, so a short quote is ambiguous inside a turn (and must be rejected);
    # the whole turn is unique
    short = " ".join(words[i]["t"] for i in range(t0["w"][0], t0["w"][0] + 3))
    quote = t0["text"]
    ann = {
        "model": "test",
        "summary": {"text": "סיכום", "evidence": [{"turn": 0, "quote": quote}]},
        "speaker_names": [
            {"speaker": t0["spk"], "name": "דנה", "confidence": 0.9, "evidence": [{"turn": 0, "quote": quote}]},
            {"speaker": t0["spk"], "name": "משה", "confidence": 0.9, "evidence": [{"turn": 0, "quote": quote}]},   # not in participants
        ],
        "decisions": [{"text": "x", "evidence": [{"turn": 0, "quote": "מילים שלא נאמרו כאן"}]},   # bad quote
                      {"text": "x2", "evidence": [{"turn": 0, "quote": short}]}],                 # ambiguous quote
        "action_items": [{"text": "y", "owner": "S1", "evidence": [{"w": [t0["w"][0], t0["w"][0]], "quote": words[t0["w"][0]]["t"]}]}],
        "term_restorations": [{"w": [0, 0], "text": "Gemini", "evidence": [{"turn": 0, "quote": quote}]}],   # not a candidate
        "boundary_corrections": [{"turn": 0, "move_words": 9, "to": "S2", "evidence": [{"turn": 0, "quote": quote}]}],
    }
    report = apply_annotations(doc, ann, context={"participants": ["דנה", "יוסי"]})
    assert report["accepted"]["summary"] == 1
    assert report["accepted"]["speaker_names"] == 1
    assert report["accepted"]["action_items"] == 1
    reasons = {r["key"]: r["reason"] for r in report["rejected"]}
    assert "not in the participant list" in reasons["speaker_names"]
    assert "quote not found" in reasons["decisions"]
    assert sum(1 for r in report["rejected"] if r["key"] == "decisions") == 2
    assert "code-switch candidate" in reasons["term_restorations"]
    assert "move_words" in reasons["boundary_corrections"]
    assert next(s for s in doc["speakers"] if s["id"] == t0["spk"])["name"] == "דנה"
    assert doc["annotations"]["source"] == "claude" and doc["annotations"]["summary"]["text"] == "סיכום"
    # anchors carry resolved fields
    ev = doc["annotations"]["summary"]["evidence"][0]
    assert ev["w"][0] == t0["w"][0] and ev["turn"] == 0 and ev["spk"] == t0["spk"]
    # through the CLI: prepare prints the contract; apply writes and re-exports
    d = str(data_dir)
    rd.write_transcript(doc)
    assert main(["--data", d, "annotate", "prepare", rd.id]) == 0
    out = capsys.readouterr().out
    assert "annotations.json" in out and "ANCHOR" in out
    rd.annotations_json.write_text(json.dumps({"model": "t", "decisions": [{"text": "z", "evidence": [{"turn": 0, "quote": quote}]}]},
                                              ensure_ascii=False), encoding="utf-8")
    assert main(["--data", d, "annotate", "apply", rd.id, str(rd.annotations_json)]) == 0
    doc = rd.read_transcript()
    assert doc["annotations"]["decisions"][0]["text"] == "z"
    md = (rd.exports / "synthetic.md").read_text(encoding="utf-8")
    assert "## החלטות" in md and "- z" in md
