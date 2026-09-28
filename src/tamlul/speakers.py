"""Speaker overlay: speakers.yaml, sample clips and the `speakers` commands (design §4.10).

Ids never change. Names are an overlay with provenance. Every edit is
appended to edits.jsonl, deterministic layers recompute, anchors re-resolve,
and the exports already present in the run directory are re-rendered.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time

import yaml

from .jobs import RunDir
from .paths import DataLayout

TIME_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?$")


def write_overlay(rd: RunDir, doc: dict) -> None:
    data = {"speakers": [{"id": s["id"], "label": s["label"], "name": s.get("name") or "",
                          "name_source": s.get("name_source", "none"), "share": s.get("stats", {}).get("share"),
                          "sample": s.get("sample")} for s in doc["speakers"]],
            "note": "Edit through `tamlul speakers <run> ...`; hand edits here are overwritten."}
    rd.speakers_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _parse_time(s: str) -> int:
    m = TIME_RE.match(s.strip())
    if not m:
        raise ValueError(f"time must be hh:mm:ss[.mmm], got {s!r}")
    h, mi, se, ms_ = m.groups()
    return (int(h) * 3600 + int(mi) * 60 + int(se)) * 1000 + int((ms_ or "0").ljust(3, "0"))


def _log_edit(rd: RunDir, op: dict) -> None:
    op = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "actor": "human", **op}
    with rd.edits_jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(op, ensure_ascii=False) + "\n")


def _speaker(doc: dict, sid: str) -> dict:
    for s in doc["speakers"]:
        if s["id"] == sid:
            return s
    raise KeyError(f"no speaker {sid}; known: {[s['id'] for s in doc['speakers']]}")


def _recompute(rd: RunDir, doc: dict, op: dict) -> None:
    from .annotate.anchors import reresolve_all
    from .annotate.deterministic import annotate_deterministic
    from .export import rewrite_existing
    annotate_deterministic(doc, profile=doc["pipeline"]["profile"])
    reresolve_all(doc)
    doc["review"]["edited"] = True
    doc["review"]["edits"].append(op)
    rd.write_transcript(doc)
    write_overlay(rd, doc)
    _log_edit(rd, op)
    rewrite_existing(rd, doc)


def show(doc: dict) -> str:
    lines = []
    for s in doc["speakers"]:
        st = s.get("stats", {})
        lines.append(f"{s['id']:4} {s['label']:10} name={s.get('name') or '-':16} ({s.get('name_source','none')})  "
                     f"share={st.get('share', 0):.0%} turns={st.get('turns', 0)} words={st.get('words', 0)}"
                     f"{'  possibly_spurious' if s.get('possibly_spurious') else ''}  sample={s.get('sample') or '-'}")
    if not lines:
        lines.append("(no speaker layer in this run)")
    return "\n".join(lines)


def rename(doc: dict, sid: str, name: str, *, source: str = "user") -> dict:
    s = _speaker(doc, sid)
    op = {"op": "rename", "id": sid, "before": s.get("name"), "after": name}
    s["name"], s["name_source"], s["name_confidence"] = name, source, 1.0 if source == "user" else s.get("name_confidence")
    return op


def merge(doc: dict, src: str, dst: str) -> dict:
    _speaker(doc, src)
    _speaker(doc, dst)
    for w in doc["words"]:
        if w["spk"] == src:
            w["spk"] = dst
        if src in w.get("spk2", []):
            w["spk2"] = [dst if x == src else x for x in w["spk2"] if x != dst] or []
    for seg in doc["segments"]:
        if seg.get("spk") == src:
            seg["spk"] = dst
    doc["speakers"] = [s for s in doc["speakers"] if s["id"] != src]
    return {"op": "merge", "from": src, "into": dst}


def reassign_range(doc: dict, s_ms: int, e_ms: int, sid: str) -> dict:
    _speaker(doc, sid)
    n = 0
    for w in doc["words"]:
        if w["s"] >= s_ms and w["e"] <= e_ms:
            w["spk"] = sid
            w["spk_conf"] = 1.0
            n += 1
    return {"op": "reassign", "s": s_ms, "e": e_ms, "to": sid, "words": n}


def reassign_turn(doc: dict, turn_ref: str, sid: str) -> dict:
    idx = int(turn_ref.upper().lstrip("T"))
    t = doc["turns"][idx]
    return {**reassign_range(doc, t["s"], t["e"], sid), "turn": idx}


def split(doc: dict, sid: str, at_ms: int) -> dict:
    old = _speaker(doc, sid)
    n = 1 + max((int(s["id"][1:]) for s in doc["speakers"] if s["id"][1:].isdigit()), default=0)
    new_id = f"S{n}"
    doc["speakers"].append({"id": new_id, "label": f"דובר {n}", "diar_label": old.get("diar_label"), "name": "",
                            "name_source": "none", "name_confidence": None, "name_evidence": None,
                            "gender_hint": None, "stats": {}, "possibly_spurious": False, "sample": None,
                            "split_from": sid})
    k = 0
    for w in doc["words"]:
        if w["spk"] == sid and w["s"] >= at_ms:
            w["spk"] = new_id
            k += 1
    return {"op": "split", "id": sid, "at": at_ms, "new": new_id, "words": k}


def speakers_command(args: argparse.Namespace, layout: DataLayout) -> int:
    rd = RunDir.open(layout, args.run)
    doc = rd.read_transcript()
    op = args.op
    a = args.args
    if op == "show":
        print(show(doc))
        return 0
    try:
        if op == "rename" and len(a) >= 2:
            change = rename(doc, a[0], " ".join(a[1:]).strip('"'))
        elif op == "merge" and len(a) == 3 and a[1] == "into":
            change = merge(doc, a[0], a[2])
        elif op == "reassign" and len(a) == 2:
            if a[0].upper().startswith("T") and a[0][1:].isdigit():
                change = reassign_turn(doc, a[0], a[1])
            else:
                s, e = a[0].split("-", 1)
                change = reassign_range(doc, _parse_time(s), _parse_time(e), a[1])
        elif op == "split" and len(a) == 3 and a[1] == "at":
            change = split(doc, a[0], _parse_time(a[2]))
        else:
            print("usage: speakers <run> show | rename S2 \"name\" | merge S3 into S1 | reassign T42 S2 | "
                  "reassign hh:mm:ss-hh:mm:ss S1 | split S1 at hh:mm:ss", file=sys.stderr)
            return 3
    except (KeyError, ValueError, IndexError) as exc:
        print(f"tamlul speakers: {exc}", file=sys.stderr)
        return 3
    _recompute(rd, doc, change)
    print(json.dumps(change, ensure_ascii=False))
    print(show(doc))
    return 0
