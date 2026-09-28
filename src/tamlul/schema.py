"""transcript.json schema checks (invariant 16) and migration.

A hand-written structural validator (no jsonschema dependency): the checks
that matter are cross-field - word index ranges that exist, monotonic times,
anchors that resolve - and JSON Schema cannot express those anyway. A JSON
Schema file ships for external tools in ``schema/transcript-1.0.json``.
"""

from __future__ import annotations

import json
from importlib import resources

from . import SCHEMA_VERSION
from .jobs import RunDir
from .paths import DataLayout

MAJOR = SCHEMA_VERSION.split(".")[0]


def _schema_version(doc: dict) -> tuple[str, str] | None:
    s = doc.get("schema", "")
    if not isinstance(s, str) or "/" not in s:
        return None
    v = s.rsplit("/", 1)[1]
    parts = v.split(".")
    return (parts[0], parts[1] if len(parts) > 1 else "0")


def validate_document(doc: dict) -> list[str]:
    errs: list[str] = []
    ver = _schema_version(doc)
    if ver is None:
        return ["missing or malformed `schema`"]
    if ver[0] != MAJOR:
        return [f"schema major {ver[0]} is not readable by this version (needs migrate)"]
    for key in ("media", "pipeline", "speakers", "words", "segments", "utterances", "turns", "events", "stats",
                "text_edits", "annotations", "review"):
        if key not in doc:
            errs.append(f"missing block `{key}`")
    if errs:
        return errs
    words = doc["words"]
    prev_s = -1
    for k, w in enumerate(words):
        for f in ("i", "s", "e", "t_raw", "t", "p"):
            if f not in w:
                errs.append(f"words[{k}] missing {f}")
                break
        else:
            if w["i"] != k:
                errs.append(f"words[{k}].i != {k}")
            if not isinstance(w["s"], int) or not isinstance(w["e"], int):
                errs.append(f"words[{k}] times must be integer ms")
            elif w["e"] < w["s"]:
                errs.append(f"words[{k}] end before start")
            elif w["s"] < prev_s - 50:
                errs.append(f"words[{k}] starts before words[{k-1}] (non-monotonic)")
            prev_s = max(prev_s, w["s"])
            for c in w["t"]:
                if c in "\u202A\u202B\u202C\u202D\u202E\u200E\u200F\u2066\u2067\u2068\u2069\uFEFF":
                    errs.append(f"words[{k}].t contains a bidi control")
                    break
        if len(errs) > 20:
            break
    spk_ids = {s["id"] for s in doc["speakers"]}
    for k, w in enumerate(words):
        if w.get("spk") is not None and w["spk"] not in spk_ids:
            errs.append(f"words[{k}].spk {w['spk']} not in speakers")
            break
    for k, seg in enumerate(doc["segments"]):
        rng = seg.get("w")
        if rng and (rng[0] < 0 or rng[1] >= len(words) or rng[0] > rng[1]):
            errs.append(f"segments[{k}].w out of range")
    for name in ("utterances", "turns"):
        for k, u in enumerate(doc[name]):
            rng = u.get("w")
            if not rng or rng[0] < 0 or rng[1] >= len(words) or rng[0] > rng[1]:
                errs.append(f"{name}[{k}].w out of range")
                break
    for s in doc["speakers"]:
        for f in ("id", "label", "name_source"):
            if f not in s:
                errs.append(f"speaker {s.get('id')} missing {f}")
    # annotation anchors
    from .annotate.anchors import validate_anchors
    errs.extend(validate_anchors(doc)[:20])
    return errs


def validate_run(layout: DataLayout, ref: str) -> list[str]:
    rd = RunDir.open(layout, ref)
    if not rd.transcript_json.is_file():
        return ["no transcript.json in run"]
    try:
        doc = rd.read_transcript()
    except ValueError as exc:
        return [f"transcript.json is not valid JSON: {exc}"]
    return validate_document(doc)


def migrate_run(layout: DataLayout, ref: str) -> dict:
    rd = RunDir.open(layout, ref)
    doc = rd.read_transcript()
    ver = _schema_version(doc)
    if ver and ver[0] == MAJOR:
        return {"run": rd.id, "migrated": False, "schema": doc["schema"]}
    # No earlier major exists yet; a future migration lives here.
    raise RuntimeError(f"no migration from schema {doc.get('schema')} to {SCHEMA_VERSION}")


def json_schema() -> dict:
    with resources.files("tamlul").joinpath("schema/transcript-1.0.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)
