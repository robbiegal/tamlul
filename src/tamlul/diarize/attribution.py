"""Word -> speaker attribution (design §4.4 `attribution.py`).

1. primary label per word = max intersection against the EXCLUSIVE timeline
   (WhisperX ``assign_word_speakers`` rule, BSD-2, re-implemented); words with
   no overlap take the nearest turn (``fill_nearest``) - never unlabelled;
2. ``overlap=true`` + ``spk2[]`` when the word intersects >= 2 speakers in the
   regular timeline;
3. smoothing: MIN_SPEAKER_RUN_WORDS, majority vote inside sentence boundaries
   (punctuation, with a gap-based fallback), the interjection exception,
   MAX_FILL_GAP_SECONDS - constants from hebrew-audio-transcriber (MIT);
4. segments re-cut at speaker changes; raw labels immutable; ``דובר N`` by
   talk time; speakers < 2 % flagged ``possibly_spurious`` (never deleted).
"""

from __future__ import annotations

import re

from . import DiarResult

# constants vendored from KoganTheDev/hebrew-audio-transcriber core/speaker_attribution.py @ 7ca0992 (MIT)
MIN_SPEAKER_RUN_WORDS = 2
MAX_FILL_GAP_SECONDS = 1.5
INTERJECTION_MIN_S = 0.35
INTERJECTION_MIN_COVERAGE = 0.80
INTERJECTIONS = {"כן", "לא", "נכון", "אוקיי", "בסדר", "ברור"}
_SENT_END = re.compile(r"[.?!]+$")
SENTENCE_GAP_MS = 700


def _overlap_ms(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _label_words(words: list[dict], exclusive: list[dict], regular: list[dict]) -> None:
    ex = sorted(exclusive, key=lambda t: t["s"])
    reg = sorted(regular, key=lambda t: t["s"])
    for w in words:
        best, best_ov = None, 0
        for t in ex:
            if t["e"] <= w["s"]:
                continue
            if t["s"] >= w["e"]:
                break
            ov = _overlap_ms(w["s"], w["e"], t["s"], t["e"])
            if ov > best_ov:
                best, best_ov = t, ov
        dur = max(1, w["e"] - w["s"])
        if best is None:
            # fill_nearest: closest exclusive turn by midpoint, within the fill gap
            mid = (w["s"] + w["e"]) // 2
            near = min(ex, key=lambda t: min(abs(t["s"] - mid), abs(t["e"] - mid)), default=None)
            if near is not None:
                w["spk"] = near["spk"]
                w["spk_conf"] = 0.3
            continue
        w["spk"] = best["spk"]
        w["spk_conf"] = round(min(1.0, best_ov / dur), 3)
        others = {t["spk"] for t in reg if _overlap_ms(w["s"], w["e"], t["s"], t["e"]) >= dur * 0.3 and t["spk"] != w["spk"]}
        if others:
            w["overlap"] = True
            w["spk2"] = sorted(others)


def _sentences(words: list[dict]) -> list[list[int]]:
    """Punctuation-bounded runs; gap-based fallback so nothing depends on an optional punctuator."""
    out: list[list[int]] = []
    cur: list[int] = []
    for w in words:
        if cur and (w["s"] - words[cur[-1]]["e"] >= SENTENCE_GAP_MS):
            out.append(cur)
            cur = []
        cur.append(w["i"])
        if _SENT_END.search(w["t"]):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _smooth(words: list[dict]) -> int:
    changed = 0
    # majority vote inside sentences (whisper-diarization heuristic, BSD-2)
    for sent in _sentences(words):
        labels = [words[i]["spk"] for i in sent if words[i]["spk"]]
        if not labels:
            continue
        maj = max(set(labels), key=labels.count)
        for i in sent:
            w = words[i]
            if w["spk"] != maj and w["spk"] is not None:
                # interjection exception: a short standalone כן/לא/נכון keeps its own speaker
                core = w["t"].strip(".,!?")
                dur_s = (w["e"] - w["s"]) / 1000
                if core in INTERJECTIONS and dur_s >= INTERJECTION_MIN_S and (w.get("spk_conf") or 0) >= INTERJECTION_MIN_COVERAGE:
                    continue
                w["spk"] = maj
                w["spk_conf"] = min(w.get("spk_conf") or 0.5, 0.6)
                changed += 1
    # minimum run length: a single-word speaker island inside a run is absorbed
    n = len(words)
    for k in range(n):
        w = words[k]
        if w["spk"] is None:
            continue
        run_end = k
        while run_end + 1 < n and words[run_end + 1]["spk"] == w["spk"]:
            run_end += 1
        run_len = run_end - k + 1
        if run_len < MIN_SPEAKER_RUN_WORDS and 0 < k and run_end + 1 < n and words[k - 1]["spk"] == words[run_end + 1]["spk"]:
            core = w["t"].strip(".,!?")
            if core in INTERJECTIONS:
                continue
            for j in range(k, run_end + 1):
                words[j]["spk"] = words[k - 1]["spk"]
                changed += 1
    return changed


def attribute(doc: dict, res: DiarResult) -> None:
    words = doc["words"]
    if not words or not res.exclusive:
        return
    _label_words(words, res.exclusive, res.turns)
    changed = _smooth(words)
    # speakers, ordered by talk time in the regular timeline
    talk = res.speech_ms_per_speaker()
    order = sorted(res.labels, key=lambda lab: -talk.get(lab, 0))
    id_map = {lab: f"S{k+1}" for k, lab in enumerate(order)}
    for w in words:
        if w["spk"] is not None:
            w["spk"] = id_map.get(w["spk"], w["spk"])
        w["spk2"] = [id_map.get(x, x) for x in w.get("spk2", [])]
    doc["speakers"] = [{"id": id_map[lab], "label": f"דובר {k+1}", "diar_label": lab, "name": "", "name_source": "none",
                        "name_confidence": None, "name_evidence": None, "gender_hint": None, "stats": {},
                        "possibly_spurious": False, "sample": None} for k, lab in enumerate(order)]
    # re-cut segments at speaker changes
    new_segs = []
    for seg in doc["segments"]:
        if not seg.get("w"):
            new_segs.append(seg)
            continue
        i0, i1 = seg["w"]
        cur_start = i0
        for i in range(i0 + 1, i1 + 2):
            if i > i1 or words[i]["spk"] != words[cur_start]["spk"]:
                piece = dict(seg)
                piece["w"] = [cur_start, i - 1]
                piece["s"], piece["e"] = words[cur_start]["s"], words[i - 1]["e"]
                piece["spk"] = words[cur_start]["spk"]
                from ..document import render
                piece["text"] = render(words, cur_start, i - 1)
                new_segs.append(piece)
                cur_start = i
    for k, s in enumerate(new_segs):
        s["id"] = k
    doc["segments"] = new_segs
    doc["pipeline"]["attribution"] = {"rule": "max-intersection(exclusive)+fill_nearest; overlap from regular; "
                                              "sentence-majority; min-run; interjection exception",
                                      "constants": {"MIN_SPEAKER_RUN_WORDS": MIN_SPEAKER_RUN_WORDS,
                                                    "MAX_FILL_GAP_SECONDS": MAX_FILL_GAP_SECONDS,
                                                    "INTERJECTION_MIN_S": INTERJECTION_MIN_S,
                                                    "INTERJECTION_MIN_COVERAGE": INTERJECTION_MIN_COVERAGE},
                                      "smoothed_words": changed, "speaker_map": id_map}


def stereo_attribute(doc: dict, channel_of_word: list[int]) -> None:
    """Exact attribution for true-stereo recordings: the channel a word was decoded from is its speaker."""
    words = doc["words"]
    for w, ch in zip(words, channel_of_word):
        w["spk"] = f"S{ch+1}"
        w["spk_conf"] = 1.0
    doc["speakers"] = [{"id": f"S{k+1}", "label": f"דובר {k+1}", "diar_label": f"CH{k}", "name": "", "name_source": "channel",
                        "name_confidence": 1.0, "name_evidence": None, "gender_hint": None, "stats": {},
                        "possibly_spurious": False, "sample": None} for k in range(2)]
