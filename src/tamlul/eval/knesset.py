"""ivrit-ai/knesset-committees sessions as a *silver* meeting reference (design §9).

Per session folder (as listed on the Hub, 2026-09-16): ``audio.m4a``,
``raw.protocol.txt`` (the edited, non-verbatim protocol), ``speakers.txt``,
``speakers.segments.txt`` (``[{"start_char","end_char","speaker_ids":[...]}]``
character ranges into the protocol; synthetic ids >= 10,000,000 for unmatched
speakers), ``transcript.aligned.json`` (stable-ts ``WhisperResult``: segments
with word timings from a forced alignment of the protocol to the audio) and
``metadata.json`` (``per_segment_quality_scores[{start,end,probability}]``).

What is derived, and what it is worth:

* **speaker-attributed text** (cpWER) from the human speaker ranges over the
  protocol - reliable labels, but the protocol is edited text, so cpWER
  against it measures verbatim-vs-protocol differences too (silver);
* **turn timings** by walking the aligned words in order against the protocol
  tokens in char order (the alignment *is* the protocol, token for token, up
  to normalization) and cutting at speaker changes; the timing is machine-made
  by the same model family we ship, so DER on it is circular and is reported
  as "silver, in-domain" (turbo 0513 pre-trained on ~4,700 h of plenums).

Unverified against a real session until one is downloaded with the token
(gated); the layout above is the documented one and the unit test uses it.
"""

from __future__ import annotations

import json
import pathlib
import re
from collections import defaultdict

from ..text.normalize import ivrit_lb

_TOKEN = re.compile(r"\S+")


def load_session(d: pathlib.Path) -> dict:
    protocol = (d / "raw.protocol.txt").read_text(encoding="utf-8")
    segs = json.loads((d / "speakers.segments.txt").read_text(encoding="utf-8"))
    aligned = json.loads((d / "transcript.aligned.json").read_text(encoding="utf-8"))
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8")) if (d / "metadata.json").is_file() else {}
    return {"protocol": protocol, "segments": segs, "aligned": aligned, "metadata": meta, "id": d.name}


def _speaker_at(segs: list[dict], pos: int) -> str:
    for s in segs:
        if s["start_char"] <= pos < s["end_char"]:
            ids = s.get("speaker_ids") or []
            return "+".join(str(i) for i in ids) if ids else "?"
    return "?"


def _aligned_words(aligned: dict) -> list[dict]:
    out = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []):
            t = (w.get("word") or "").strip()
            if t:
                out.append({"t": t, "s": int(round(float(w["start"]) * 1000)), "e": int(round(float(w["end"]) * 1000)),
                            "p": float(w.get("probability", 1.0))})
    return out


def to_reference(session: dict, *, min_probability: float = 0.7) -> dict:
    """Text streams per speaker (all tokens) + silver turns (from aligned words in good-quality regions)."""
    protocol, segs = session["protocol"], session["segments"]
    tokens = [(m.start(), m.group(0)) for m in _TOKEN.finditer(protocol)]
    streams: dict[str, list[str]] = defaultdict(list)
    spk_of_token = []
    for pos, tok in tokens:
        spk = _speaker_at(segs, pos)
        spk_of_token.append(spk)
        streams[spk].append(tok)
    words = _aligned_words(session["aligned"])
    # walk both sequences; the alignment is the protocol in order, so a simple two-pointer match on
    # normalized tokens with bounded look-ahead recovers the token index of each aligned word
    good = [(int(q["start"] * 1000), int(q["end"] * 1000)) for q in session["metadata"].get("per_segment_quality_scores", [])
            if float(q.get("probability", 1.0)) >= min_probability]
    turns: list[dict] = []
    j = 0
    cur: dict | None = None
    for w in words:
        nw = ivrit_lb(w["t"])
        k = None
        for look in range(j, min(j + 8, len(tokens))):
            if ivrit_lb(tokens[look][1]) == nw:
                k = look
                break
        if k is None:
            continue
        j = k + 1
        spk = spk_of_token[k]
        if good and not any(s <= w["s"] < e for s, e in good):
            cur = None
            continue
        if cur and cur["spk"] == spk and w["s"] - cur["e"] < 2000:
            cur["e"] = max(cur["e"], w["e"])
            cur["text"] += " " + w["t"]
        else:
            cur = {"s": w["s"], "e": w["e"], "spk": spk, "text": w["t"]}
            turns.append(cur)
    return {"id": session["id"], "speakers": sorted(streams), "text_streams": {k: " ".join(v) for k, v in streams.items()},
            "turns": turns, "words_aligned": len(words), "words_protocol": len(tokens),
            "quality_regions": len(good), "note": "silver: edited protocol text; machine timings (circular for DER)"}


def score(doc: dict, ref: dict) -> dict:
    from . import speakers as spk_metrics
    hyp = spk_metrics.doc_turns(doc)
    cp = spk_metrics.cpwer(ref["text_streams"], spk_metrics.streams_from_turns(hyp))
    out = {"id": ref["id"], "cpwer_silver": round(cp["cpwer"], 4), "cpwer_n": cp["n"],
           "speaker_count": spk_metrics.speaker_count(len(ref["speakers"]), len({t["spk"] for t in hyp}))}
    if ref["turns"]:
        d = spk_metrics.der(ref["turns"], hyp, collar_s=0.25)
        out["der_silver_collar0.25"] = round(d["der"], 4)
        out["der_note"] = "silver, in-domain, timings from the same model family (circular)"
    return out
