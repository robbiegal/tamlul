"""The canonical transcript.json (design §4.8): building it from raw ASR output,
and the small helpers everything else uses to read it.

Rules: integer milliseconds; ``words[].t_raw`` immutable; ``t`` is the rendered
text after clean/ITN/correction; every derived layer carries ``source``;
evidence anchors are raw word index ranges.
"""

from __future__ import annotations

import time
from typing import Any

from . import SCHEMA_VERSION, __version__
from .asr.base import RawSegment
from .text import clean as C

SCHEMA_ID = f"tamlul/annotation/{SCHEMA_VERSION}"


def ms(x: float) -> int:
    return int(round(x * 1000))


def new_document(media: dict, pipeline: dict, profile: str) -> dict:
    return {
        "schema": SCHEMA_ID,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plugin_version": __version__,
        "media": media,
        "pipeline": {**pipeline, "profile": profile},
        "speakers": [],
        "words": [],
        "segments": [],
        "utterances": [],
        "turns": [],
        "events": [],
        "stats": {},
        "text_edits": [],
        "annotations": {},
        "review": {"status": "raw", "speakers_confirmed": False, "edited": False, "edits": []},
    }


def add_raw_segments(doc: dict, raw: list[RawSegment], *, offset_s: float, group: int,
                     speech_regions: list[list[float]] | None = None) -> None:
    """Append one plan group's segments/words (times offset back to file time)."""
    words = doc["words"]
    segs = doc["segments"]
    for rs in raw:
        s_ms, e_ms = ms(rs.s + offset_s), ms(rs.e + offset_s)
        first = len(words)
        for w in rs.words:
            t_raw = w.t
            t = C.clean_word_text(t_raw)
            if not t:
                continue
            words.append({"i": len(words), "s": ms(w.s + offset_s), "e": ms(w.e + offset_s),
                          "t_raw": t_raw, "t": t, "p": round(float(w.p), 4),
                          "spk": None, "spk_conf": None, "overlap": False, "spk2": []})
        last = len(words) - 1
        text = C.clean_segment_text(rs.text)
        seg = {"id": len(segs), "s": s_ms, "e": e_ms, "spk": None, "text": text,
               "w": [first, last] if last >= first else None, "group": group,
               "asr": {"avg_logprob": rs.avg_logprob, "no_speech_prob": rs.no_speech_prob,
                       "compression_ratio": rs.compression_ratio, "temperature": rs.temperature},
               "edited": False, "flags": []}
        if speech_regions is not None:
            seg["speech_fraction"] = round(_speech_fraction(s_ms / 1000, e_ms / 1000, speech_regions), 3)
        segs.append(seg)


def _speech_fraction(s: float, e: float, regions: list[list[float]]) -> float:
    if e <= s:
        return 0.0
    cov = 0.0
    for a, b in regions:
        lo, hi = max(a, s), min(b, e)
        if hi > lo:
            cov += hi - lo
    return cov / (e - s)


def apply_clean_flags(doc: dict) -> None:
    """Repetition guard, hallucination filter, low-confidence flags.

    Invariant 8 holds: nothing is deleted - a repetition loop is only flagged, and a segment judged a
    hallucination (``credit_phrase_suspect``) is flagged AND its words are blanked in the rendered
    text (``t`` = "", ``t_raw`` untouched, one ``text_edits`` entry per word), the same way ITN hides
    the words it merges, so no human-readable export prints boilerplate the audio never contained."""
    words = doc["words"]
    for seg in doc["segments"]:
        if not seg["w"]:
            continue
        i0, i1 = seg["w"]
        toks = [words[i]["t"] for i in range(i0, i1 + 1)]
        loop = C.repetition_loop(toks)
        if loop:
            start, n, k = loop
            seg["flags"].append(f"repetition_loop:{n}x{k}")
            doc["text_edits"].append(C.Edit("clean", "repetition_flag", " ".join(toks[start:start + n * k]),
                                            " ".join(toks[start:start + n * 2]) + " ...",
                                            f"{n}-gram repeated {k}x; kept in words[], flagged", segment=seg["id"]).as_dict())
        sf = seg.get("speech_fraction", 1.0)
        spoken = [i for i in range(i0, i1 + 1) if words[i]["t"]]
        if C.credit_phrase_suspect(seg["text"], speech_fraction=sf, no_speech_prob=seg["asr"].get("no_speech_prob"),
                                   duration_ms=seg["e"] - seg["s"], n_words=len(spoken)):
            seg["flags"].append("credit_phrase_suspect")
            for i in spoken:
                doc["text_edits"].append(C.Edit("clean", "hallucination_filter", words[i]["t"], "",
                                                f"no speech under the segment (VAD fraction {sf:.2f}); kept in t_raw",
                                                segment=seg["id"], word=i).as_dict())
                words[i]["t"] = ""
                words[i]["edited"] = True
            seg["text"] = ""
            continue
        probs = [words[i]["p"] for i in range(i0, i1 + 1)]
        if probs and sum(probs) / len(probs) < C.LOW_WORD_PROB:
            seg["flags"].append("low_confidence")


def render(words: list[dict], i0: int, i1: int) -> str:
    return " ".join(words[i]["t"] for i in range(i0, i1 + 1) if words[i]["t"])


def speaker_name(doc: dict, spk: str | None) -> str:
    if spk is None:
        return ""
    for s in doc["speakers"]:
        if s["id"] == spk:
            return s.get("name") or s["label"]
    return spk


def hms(ms_value: int, *, with_ms: bool = False) -> str:
    m, s = divmod(ms_value // 1000, 60)
    h, m = divmod(m, 60)
    base = f"{h:02d}:{m:02d}:{s:02d}"
    return f"{base}.{ms_value % 1000:03d}" if with_ms else base


def profile_thresholds(profile: str) -> dict[str, Any]:
    from .config import DEFAULT_PROFILE, PROFILES
    p = PROFILES.get(profile, PROFILES[DEFAULT_PROFILE])
    return {"utt_gap_ms": 700, "pause_ms": int(p["pause_s"] * 1000), "gap_ms": int(p["gap_s"] * 1000),
            "latch_ms": 100, "backchannel_max_ms": 1000, "backchannel_max_tokens": 3,
            "interruption_lead_ms": 200, "interruption_yield_ms": 1000, "interruption_hold_ms": 1000,
            "silence_ms": 2000, "min_speaker_share": 0.02, "low_word_prob": C.LOW_WORD_PROB}
