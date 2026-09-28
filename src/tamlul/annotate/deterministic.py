"""Deterministic annotation (design §4.7): utterances, turns, Heldner & Edlund
events, candidates and per-speaker stats. Pure functions over the document;
every threshold used is written into ``stats.thresholds``.

Works with or without a speaker layer: without one, every word has
``spk=None`` and the whole recording is one floor-holding "speaker".
"""

from __future__ import annotations

import re
from collections import Counter

from ..document import profile_thresholds, render

BACKCHANNEL_LEXICON = {"כן", "אה", "אהה", "מממ", "אמ", "אוקיי", "אוקי", "בסדר", "נכון", "ברור", "יופי", "וואו",
                       "טוב", "כן כן", "אהם", "הממ", "אממ"}
WH_WORDS = {"מה", "מי", "למה", "איך", "כמה", "מתי", "איפה", "האם", "איזה", "אילו", "לאן", "מאיפה", "מדוע", "איזו"}
_SENT_END = re.compile(r"[.?!]+$")


def _spk(w: dict) -> str | None:
    return w.get("spk")


def build_utterances(doc: dict, thr: dict) -> list[dict]:
    words = doc["words"]
    utts: list[dict] = []
    cur: list[int] = []

    def flush() -> None:
        if not cur:
            return
        i0, i1 = cur[0], cur[-1]
        text = render(words, i0, i1)
        first = words[i0]["t"]
        is_q = bool(_SENT_END.search(text) and text.rstrip().endswith("?")) or first in WH_WORDS
        utts.append({"id": len(utts), "s": words[i0]["s"], "e": words[i1]["e"], "spk": _spk(words[i0]),
                     "w": [i0, i1], "text": text, "is_question": is_q,
                     "kind": "question" if is_q else "statement"})
        cur.clear()

    for w in words:
        if cur:
            prev = words[cur[-1]]
            gap = w["s"] - prev["e"]
            if _spk(w) != _spk(prev) or gap >= thr["utt_gap_ms"] or _SENT_END.search(prev["t"]):
                flush()
        cur.append(w["i"])
    flush()
    return utts


def is_backchannel(utt: dict, thr: dict) -> bool:
    dur = utt["e"] - utt["s"]
    toks = utt["text"].strip(" .,!?").split()
    if dur > thr["backchannel_max_ms"] or len(toks) > thr["backchannel_max_tokens"]:
        return False
    core = " ".join(toks)
    return core in BACKCHANNEL_LEXICON or all(t.strip("?.!,") in BACKCHANNEL_LEXICON for t in toks)


def build_turns(doc: dict, utts: list[dict], thr: dict) -> tuple[list[dict], list[dict]]:
    """Floor-holding units: consecutive same-speaker utterances until another
    speaker's NON-backchannel utterance intervenes. Backchannels become candidates."""
    words = doc["words"]
    turns: list[dict] = []
    candidates: list[dict] = []
    cur: dict | None = None
    for u in utts:
        bc = is_backchannel(u, thr) and cur is not None and u["spk"] != cur["spk"] and u["spk"] is not None
        if bc:
            candidates.append({"utt": u["id"], "spk": u["spk"], "s": u["s"], "e": u["e"], "text": u["text"],
                               "inside_turn": cur["id"], "w": u["w"]})
            continue
        if cur is not None and u["spk"] == cur["spk"]:
            cur["e"] = u["e"]
            cur["w"][1] = u["w"][1]
            cur["utts"].append(u["id"])
            continue
        prev = cur
        cur = {"id": len(turns), "spk": u["spk"], "s": u["s"], "e": u["e"], "w": [u["w"][0], u["w"][1]],
               "utts": [u["id"]], "fto_ms": None, "transition": "start"}
        if prev is not None:
            fto = cur["s"] - prev["e"]
            cur["fto_ms"] = fto
            if abs(fto) <= thr["latch_ms"]:
                cur["transition"] = "latch"
            elif fto < 0:
                cur["transition"] = "overlap"
            else:
                cur["transition"] = "gap"
        turns.append(cur)
    for t in turns:
        t["text"] = render(words, t["w"][0], t["w"][1])
        t["tid"] = f"{t['spk'] or 'S0'}:{t['w'][0]}-{t['w'][1]}"   # content-derived, informational (invariant 12)
    return turns, candidates


def build_events(doc: dict, utts: list[dict], turns: list[dict], candidates: list[dict], thr: dict) -> list[dict]:
    events: list[dict] = []
    words = doc["words"]
    # within-speaker pauses and between-speaker gaps/latches/overlaps
    for k in range(1, len(utts)):
        a, b = utts[k - 1], utts[k]
        gap = b["s"] - a["e"]
        if a["spk"] == b["spk"]:
            if gap >= thr["pause_ms"]:
                events.append({"kind": "pause", "s": a["e"], "e": b["s"], "spk": a["spk"], "ms": gap})
        else:
            if gap > thr["gap_ms"]:
                events.append({"kind": "gap", "s": a["e"], "e": b["s"], "from": a["spk"], "to": b["spk"], "ms": gap})
            elif abs(gap) <= thr["latch_ms"]:
                events.append({"kind": "latch", "s": b["s"], "e": b["s"], "from": a["spk"], "to": b["spk"], "ms": gap})
    # overlaps from the word layer (set by attribution when the regular diarization overlaps)
    run: list[int] = []
    for w in words:
        if w.get("overlap"):
            run.append(w["i"])
        elif run:
            events.append(_overlap_event(words, run, candidates))
            run = []
    if run:
        events.append(_overlap_event(words, run, candidates))
    # interruptions: B starts >= lead before A ends, B is not a backchannel, A yields, B holds
    for k in range(1, len(turns)):
        a, b = turns[k - 1], turns[k]
        if a["spk"] is None or b["spk"] is None or a["spk"] == b["spk"]:
            continue
        lead = a["e"] - b["s"]
        if lead >= thr["interruption_lead_ms"] and (a["e"] - b["s"]) <= thr["interruption_yield_ms"] \
                and (b["e"] - b["s"]) >= thr["interruption_hold_ms"]:
            events.append({"kind": "interruption", "s": b["s"], "e": a["e"], "by": b["spk"], "of": a["spk"],
                           "fto_ms": -lead})
    # long silences from the plan (regions with no speech)
    for seg in doc["segments"]:
        if "low_confidence" in seg.get("flags", []):
            events.append({"kind": "low_confidence", "s": seg["s"], "e": seg["e"], "segment": seg["id"]})
    events.sort(key=lambda e: (e["s"], e["kind"]))
    return events


def _overlap_event(words: list[dict], run: list[int], candidates: list[dict]) -> dict:
    s, e = words[run[0]]["s"], words[run[-1]]["e"]
    spks = sorted({words[i]["spk"] for i in run if words[i]["spk"]} | {x for i in run for x in words[i].get("spk2", [])})
    bc = any(c["s"] <= e and c["e"] >= s for c in candidates)
    return {"kind": "overlap", "s": s, "e": e, "spks": spks, "subtype": "backchannel" if bc else "competitive",
            "w": [run[0], run[-1]]}


def question_candidates(utts: list[dict]) -> list[dict]:
    return [{"utt": u["id"], "spk": u["spk"], "s": u["s"], "e": u["e"], "text": u["text"], "w": u["w"]}
            for u in utts if u["is_question"]]


def speaker_stats(doc: dict, utts: list[dict], turns: list[dict], candidates: list[dict], events: list[dict],
                  diar_speech_ms: dict[str, int] | None, thr: dict) -> None:
    words = doc["words"]
    total_speech = 0
    per: dict[str, dict] = {}
    for sp in doc["speakers"]:
        per[sp["id"]] = {"speech_ms": 0, "share": 0.0, "turns": 0, "words": 0, "wpm": 0.0, "longest_turn_ms": 0,
                         "questions": 0, "backchannels": 0, "interruptions_made": 0, "interruptions_received": 0,
                         "mean_fto_ms": None}
    for w in words:
        if w["spk"] in per:
            per[w["spk"]]["words"] += 1
    if diar_speech_ms:
        for k, v in diar_speech_ms.items():
            if k in per:
                per[k]["speech_ms"] = v
    else:
        for t in turns:
            if t["spk"] in per:
                per[t["spk"]]["speech_ms"] += t["e"] - t["s"]
    total_speech = sum(p["speech_ms"] for p in per.values())
    ftos: dict[str, list[int]] = {k: [] for k in per}
    for t in turns:
        if t["spk"] in per:
            p = per[t["spk"]]
            p["turns"] += 1
            p["longest_turn_ms"] = max(p["longest_turn_ms"], t["e"] - t["s"])
            if t["fto_ms"] is not None:
                ftos[t["spk"]].append(t["fto_ms"])
    for u in utts:
        if u["is_question"] and u["spk"] in per:
            per[u["spk"]]["questions"] += 1
    for c in candidates:
        if c["spk"] in per:
            per[c["spk"]]["backchannels"] += 1
    for e in events:
        if e["kind"] == "interruption":
            if e["by"] in per:
                per[e["by"]]["interruptions_made"] += 1
            if e["of"] in per:
                per[e["of"]]["interruptions_received"] += 1
    for k, p in per.items():
        p["share"] = round(p["speech_ms"] / total_speech, 4) if total_speech else 0.0
        p["wpm"] = round(p["words"] / (p["speech_ms"] / 60000), 1) if p["speech_ms"] else 0.0
        p["mean_fto_ms"] = round(sum(ftos[k]) / len(ftos[k])) if ftos[k] else None
        p["possibly_spurious"] = bool(total_speech and p["share"] < thr["min_speaker_share"])
    for sp in doc["speakers"]:
        sp["stats"] = per[sp["id"]]
        sp["possibly_spurious"] = per[sp["id"]].pop("possibly_spurious")


def annotate_deterministic(doc: dict, *, profile: str, plan: dict | None = None,
                           diar_speech_ms: dict[str, int] | None = None) -> None:
    thr = profile_thresholds(profile)
    utts = build_utterances(doc, thr)
    turns, candidates = build_turns(doc, utts, thr)
    events = build_events(doc, utts, turns, candidates, thr)
    doc["utterances"] = utts
    doc["turns"] = turns
    doc["events"] = events
    doc["candidates"] = {"backchannels": candidates, "questions": question_candidates(utts),
                         "code_switch": doc.get("candidates", {}).get("code_switch", [])}
    speaker_stats(doc, utts, turns, candidates, events, diar_speech_ms, thr)
    words = doc["words"]
    duration = doc["media"].get("duration_ms") or (words[-1]["e"] if words else 0)
    speech_ms = int((plan or {}).get("speech_s", 0) * 1000) if plan else sum(t["e"] - t["s"] for t in turns)
    overlap_ms = sum(e["e"] - e["s"] for e in events if e["kind"] == "overlap")
    kinds = Counter(e["kind"] for e in events)
    ftos = [t["fto_ms"] for t in turns if t["fto_ms"] is not None]
    doc["stats"] = {
        "duration_ms": duration, "speech_ms": speech_ms, "silence_ms": max(0, duration - speech_ms),
        "overlap_ms": overlap_ms, "words": len(words), "segments": len(doc["segments"]),
        "utterances": len(utts), "turns": len(turns), "speaker_changes": sum(1 for t in turns if t["transition"] != "start"),
        "median_fto_ms": sorted(ftos)[len(ftos) // 2] if ftos else None,
        "events": dict(kinds), "backchannel_candidates": len(candidates),
        "question_candidates": len(doc["candidates"]["questions"]),
        "low_confidence_segments": sum(1 for s in doc["segments"] if "low_confidence" in s.get("flags", [])),
        "flagged_segments": sum(1 for s in doc["segments"] if s.get("flags")),
        "thresholds": thr,
    }
