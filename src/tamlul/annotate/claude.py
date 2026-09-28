"""`annotate prepare` and `annotate apply` (design §4.9): the contract between the
CLI and the Claude annotator. Claude proposes; Python validates and applies.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from importlib import resources

from .. import SCHEMA_VERSION, __version__
from ..document import hms, render
from ..jobs import RunDir
from ..paths import DataLayout
from .anchors import check_anchor, resolve_anchor

MAX_QUOTE_WORDS = 40
MAX_BOUNDARY_MOVE = 3


# ------------------------------------------------------------------ prepare

def render_input(doc: dict, *, participants: list[str] | None, glossary: list[dict] | None) -> str:
    out = [f"# {pathlib.Path(doc['media']['path']).name} - annotation input", ""]
    st = doc["stats"]
    out += ["## Recording", "",
            f"- profile: {doc['pipeline'].get('profile')}; duration {hms(st.get('duration_ms', 0))}; "
            f"speech {hms(st.get('speech_ms', 0))}; words {st.get('words')}; turns {st.get('turns')}; "
            f"speaker changes {st.get('speaker_changes')}; overlap {st.get('overlap_ms', 0)/1000:.1f}s",
            f"- events: {st.get('events')}", ""]
    if doc["speakers"]:
        out += ["## Speakers (ids are fixed; names are what you may propose)", "",
                "| id | label | current name | share | turns | words | questions | backchannels | interruptions made/received | flags |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for s in doc["speakers"]:
            x = s.get("stats", {})
            out.append(f"| {s['id']} | {s['label']} | {s.get('name') or '-'} | {x.get('share', 0):.0%} | {x.get('turns', 0)} | "
                       f"{x.get('words', 0)} | {x.get('questions', 0)} | {x.get('backchannels', 0)} | "
                       f"{x.get('interruptions_made', 0)}/{x.get('interruptions_received', 0)} | "
                       f"{'possibly_spurious' if s.get('possibly_spurious') else ''} |")
        out.append("")
    else:
        out += ["## Speakers", "", "(no speaker layer in this run - do not propose speaker names)", ""]
    if participants:
        out += ["## Participants (use ONLY these names)", ""] + [f"- {p}" for p in participants] + [""]
    if glossary:
        out += ["## Glossary", ""] + [f"- {g.get('canonical')}" + (f" / {g['latin']}" if g.get('latin') else "") for g in glossary] + [""]
    cands = doc.get("candidates", {})
    if cands.get("questions"):
        out += ["## Question candidates (deterministic)", ""]
        out += [f"- turn-time {hms(q['s'])} {q.get('spk') or ''}: {q['text']}" for q in cands["questions"][:60]] + [""]
    if cands.get("backchannels"):
        out += ["## Backchannel candidates (deterministic; confirm or ignore)", ""]
        out += [f"- {hms(b['s'])} {b['spk']}: {b['text']}" for b in cands["backchannels"][:40]] + [""]
    cs = cands.get("code_switch") or []
    if cs:
        out += ["## Code-switch candidates (the ONLY words term_restorations may target)", ""]
        out += [f"- w {c['w'][0]}: {c['text']} (p={c['p']:.2f})" + (f" candidates {c['candidates']}" if c.get("candidates") else "")
                for c in cs[:80]] + [""]
    out += ["## Transcript (turn id, time, speaker, raw word range; quote words exactly as written)", ""]
    words = doc["words"]
    for t in doc["turns"]:
        spk = t["spk"] or "-"
        name = ""
        for s in doc["speakers"]:
            if s["id"] == t["spk"] and s.get("name"):
                name = f" ({s['name']})"
        out.append(f"T{t['id']} [{hms(t['s'])}] {spk}{name} w{t['w'][0]}-{t['w'][1]}:")
        out.append(render(words, t["w"][0], t["w"][1]))
        out.append("")
    return "\n".join(out)


def prepare(rd: RunDir, *, participants_file: str | None, glossary_file: str | None) -> pathlib.Path:
    doc = rd.read_transcript()
    participants = None
    if participants_file and pathlib.Path(participants_file).is_file():
        participants = [ln.strip() for ln in pathlib.Path(participants_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    glossary = None
    if glossary_file and pathlib.Path(glossary_file).is_file():
        from ..text.correct import load_glossary
        glossary = [t.__dict__ for t in load_glossary(pathlib.Path(glossary_file))]
    text = render_input(doc, participants=participants, glossary=glossary)
    p = rd.path / "annotate_input.md"
    p.write_text(text, encoding="utf-8")
    (rd.path / "annotate_context.json").write_text(json.dumps({"participants": participants, "glossary": glossary},
                                                              ensure_ascii=False), encoding="utf-8")
    return p


def contract_text(rd: RunDir, input_path: pathlib.Path) -> str:
    tpl = resources.files("tamlul").joinpath("prompts/annotate.md").read_text(encoding="utf-8")
    apply_cmd = f'tamlul annotate apply {rd.id} "{rd.annotations_json}"'
    return tpl.format(version=__version__, schema=SCHEMA_VERSION, run_id=rd.id, input_path=input_path,
                      output_path=rd.annotations_json, apply_cmd=apply_cmd)


# -------------------------------------------------------------------- apply

def _find_quote_in_turn(doc: dict, turn_id: int, quote: str) -> tuple[int, int] | None:
    turns = doc["turns"]
    if not (0 <= turn_id < len(turns)):
        return None
    t = turns[turn_id]
    words = doc["words"]
    q = [x for x in quote.replace("\u200f", "").split() if x]
    if not q or len(q) > MAX_QUOTE_WORDS:
        return None
    toks = [(i, words[i]["t"]) for i in range(t["w"][0], t["w"][1] + 1) if words[i]["t"]]
    hits = []
    for k in range(len(toks) - len(q) + 1):
        if [x[1] for x in toks[k:k + len(q)]] == q:
            hits.append((toks[k][0], toks[k + len(q) - 1][0]))
    return hits[0] if len(hits) == 1 else None


def _resolve(doc: dict, anchor: dict) -> tuple[dict | None, str | None]:
    if not isinstance(anchor, dict):
        return None, "anchor must be an object"
    a = dict(anchor)
    if "w" not in a:
        if "turn" not in a or "quote" not in a:
            return None, "anchor needs w or turn+quote"
        rng = _find_quote_in_turn(doc, int(a["turn"]), str(a["quote"]))
        if rng is None:
            return None, f"quote not found exactly once in turn {a['turn']}: {str(a['quote'])[:50]!r}"
        a["w"] = list(rng)
    if "quote" not in a:
        a["quote"] = render(doc["words"], a["w"][0], a["w"][1])
    err = check_anchor(doc, a)
    if err:
        return None, err
    return resolve_anchor(doc, a), None


def apply_annotations(doc: dict, ann: dict, *, context: dict | None = None) -> dict:
    """Validate and merge. Returns a report; mutates doc for accepted items."""
    report = {"accepted": {}, "rejected": []}
    context = context or {}
    participants = context.get("participants") or None
    accepted: dict = {"source": "claude", "model": str(ann.get("model", "")), "applied": __import__("time").strftime("%Y-%m-%dT%H:%M:%S")}
    spk_ids = {s["id"] for s in doc["speakers"]}
    cs_targets = {tuple(c["w"]) for c in (doc.get("candidates") or {}).get("code_switch", [])}

    def take(key: str, items, checker=None):
        ok = []
        for k, item in enumerate(items or []):
            if not isinstance(item, dict):
                report["rejected"].append({"key": key, "index": k, "reason": "item must be an object"})
                continue
            ev_in = item.get("evidence") or []
            if not ev_in and key != "summary":
                report["rejected"].append({"key": key, "index": k, "reason": "no evidence"})
                continue
            ev_out, bad = [], None
            for a in ev_in:
                r, err = _resolve(doc, a)
                if err:
                    bad = err
                    break
                ev_out.append(r)
            if bad:
                report["rejected"].append({"key": key, "index": k, "reason": bad})
                continue
            if checker:
                err = checker(item, ev_out)
                if err:
                    report["rejected"].append({"key": key, "index": k, "reason": err})
                    continue
            item = dict(item)
            item["evidence"] = ev_out
            ok.append(item)
        if ok or key in ann:
            accepted[key] = ok
            report["accepted"][key] = len(ok)
        return ok

    def chk_speaker(item, ev):
        sp = item.get("speaker")
        if sp not in spk_ids:
            return f"unknown speaker {sp}"
        name = str(item.get("name", "")).strip()
        if not name:
            return "empty name"
        if participants and name not in participants:
            return f"name {name!r} is not in the participant list"
        return None

    def chk_owner(item, ev):
        o = item.get("owner")
        return None if (o is None or o in spk_ids) else f"unknown owner {o}"

    def chk_term(item, ev):
        w = item.get("w")
        if not (isinstance(w, list) and len(w) == 2 and tuple(w) in cs_targets):
            return "w is not a code-switch candidate"
        if not any("a" <= c.lower() <= "z" for c in str(item.get("text", ""))):
            return "restored term must be Latin script"
        return None

    def chk_boundary(item, ev):
        t = item.get("turn")
        if not isinstance(t, int) or not (0 <= t < len(doc["turns"])):
            return "unknown turn"
        mv = item.get("move_words")
        if not isinstance(mv, int) or mv == 0 or abs(mv) > MAX_BOUNDARY_MOVE:
            return f"move_words must be a non-zero integer within +-{MAX_BOUNDARY_MOVE}"
        if item.get("to") not in spk_ids:
            return f"unknown speaker {item.get('to')}"
        return None

    summ = ann.get("summary")
    if isinstance(summ, dict) and summ.get("text"):
        ok = take("summary", [summ])
        accepted["summary"] = ok[0] if ok else None
        report["accepted"]["summary"] = 1 if ok else 0
    names = take("speaker_names", ann.get("speaker_names"), chk_speaker)
    take("speaker_summaries", ann.get("speaker_summaries"), lambda it, ev: None if it.get("speaker") in spk_ids else "unknown speaker")
    take("topics", ann.get("topics"))
    take("decisions", ann.get("decisions"))
    take("action_items", ann.get("action_items"), chk_owner)
    take("questions", ann.get("questions"))
    take("key_quotes", ann.get("key_quotes"))
    terms = take("term_restorations", ann.get("term_restorations"), chk_term)
    bounds = take("boundary_corrections", ann.get("boundary_corrections"), chk_boundary)

    # ---- side effects on the document (speaker names, term restorations, boundary moves)
    from ..speakers import rename
    from ..text.clean import Edit
    for n in names:
        rename(doc, n["speaker"], n["name"], source="llm")
        for s in doc["speakers"]:
            if s["id"] == n["speaker"]:
                s["name_confidence"] = float(n.get("confidence", 0.5))
                s["name_evidence"] = n["evidence"][0] if n["evidence"] else None
    for t in terms:
        w = doc["words"][t["w"][0]]
        doc["text_edits"].append(Edit("annotate", "term_restoration", w["t"], t["text"], "claude restoration on a flagged word",
                                      word=w["i"]).as_dict())
        w["t"] = str(t["text"])
        w["edited"] = True
    moved = 0
    for b in bounds:
        turn = doc["turns"][b["turn"]]
        i0, i1 = turn["w"]
        if b["move_words"] < 0:
            rng = range(max(0, i0 + b["move_words"]), i0)
        else:
            rng = range(i0, min(i1 + 1, i0 + b["move_words"]))
        for i in rng:
            doc["words"][i]["spk"] = b["to"]
            doc["words"][i]["spk_conf"] = 0.7
            moved += 1
    accepted["boundary_words_moved"] = moved
    doc["annotations"] = accepted
    doc["review"]["status"] = "annotated"
    return report


def annotate_command(args: argparse.Namespace, layout: DataLayout) -> int:
    rd = RunDir.open(layout, args.run)
    if args.acmd == "prepare":
        p = prepare(rd, participants_file=args.participants, glossary_file=args.glossary)
        print(contract_text(rd, p))
        return 0
    if args.acmd == "apply":
        path = pathlib.Path(args.annotations)
        if not path.is_file():
            print(f"tamlul annotate: no such file {path}", file=sys.stderr)
            return 3
        try:
            ann = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            print(json.dumps({"accepted": {}, "rejected": [{"key": "*", "reason": f"annotations.json is not valid JSON: {exc}"}]}, ensure_ascii=False, indent=2))
            return 2
        doc = rd.read_transcript()
        ctx = {}
        cpath = rd.path / "annotate_context.json"
        if cpath.is_file():
            ctx = json.loads(cpath.read_text(encoding="utf-8"))
        report = apply_annotations(doc, ann, context=ctx)
        from ..annotate.anchors import reresolve_all
        from ..annotate.deterministic import annotate_deterministic
        from ..export import rewrite_existing
        from ..schema import validate_document
        from ..speakers import write_overlay
        annotate_deterministic(doc, profile=doc["pipeline"]["profile"])
        reresolve_all(doc)
        errs = validate_document(doc)
        if errs:
            print(json.dumps({"accepted": {}, "rejected": [{"key": "*", "reason": e} for e in errs]}, ensure_ascii=False, indent=2))
            return 2
        rd.write_transcript(doc)
        write_overlay(rd, doc)
        written = rewrite_existing(rd, doc)
        report["exports"] = written
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if not report["rejected"] else 2
    return 3
