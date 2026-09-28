"""Score a transcript.json against a synthetic reference (design §4.15 ``score_ts.py``).

Per reference clip: onset error = |first hypothesis word start inside/near the
clip - clip start|, offset error likewise with the last word end; reported as
the fraction within 25/50/100/200 ms, mean and p95 - and only meaningful at
collars >= the tier's precision floor (manifest). Also DER on clip-level
speaker labels (collar 0 and 0.25 s), word->speaker agreement after optimal
mapping, and health counters (words outside any clip, clips with no words).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from .speakers import der
from .synth import load_reference

COLLARS_MS = (25, 50, 100, 200)


def _words(doc: dict) -> list[dict]:
    return [w for w in doc.get("words", []) if w.get("t_raw", w.get("t", "")).strip()]


def score(doc: dict, synth_dir: pathlib.Path) -> dict:
    refs, manifest = load_reference(synth_dir)
    words = _words(doc)
    starts = np.asarray([w["s"] for w in words], dtype=np.float64)
    ends = np.asarray([w["e"] for w in words], dtype=np.float64)
    mids = (starts + ends) / 2
    on_err, off_err = [], []
    empty_clips = 0
    clip_of_word = np.full(len(words), -1, dtype=np.int64)
    for k, r in enumerate(refs):
        inside = np.nonzero((mids >= r["s"]) & (mids < r["e"]))[0]
        if len(inside) == 0:
            empty_clips += 1
            continue
        clip_of_word[inside] = k
        on_err.append(abs(float(starts[inside[0]]) - r["s"]))
        off_err.append(abs(float(ends[inside[-1]]) - r["e"]))
    outside = int(np.sum(clip_of_word < 0))
    on, off = np.asarray(on_err or [np.nan]), np.asarray(off_err or [np.nan])

    def summ(a: np.ndarray) -> dict:
        a = a[~np.isnan(a)]
        if len(a) == 0:
            return {"n": 0}
        return {"n": int(len(a)), "mean_ms": round(float(a.mean()), 1), "median_ms": round(float(np.median(a)), 1),
                "p95_ms": round(float(np.percentile(a, 95)), 1),
                **{f"within_{c}ms": round(float(np.mean(a <= c)), 4) for c in COLLARS_MS}}

    out = {"synth": str(synth_dir), "clips": len(refs), "words": len(words), "empty_clips": empty_clips,
           "words_outside_clips": outside, "precision_floor_ms": manifest.get("trim", {}).get("precision_floor_ms"),
           "offset_floor_ms": manifest.get("trim", {}).get("offset_floor_ms"),
           "onset": summ(on), "offset": summ(off)}
    # speakers: clip-level reference turns vs the document's turns
    ref_turns = [{"s": r["s"], "e": r["e"], "spk": r["spk"]} for r in refs]
    hyp_turns = [{"s": t["s"], "e": t["e"], "spk": t["spk"]} for t in doc.get("turns", [])]
    if hyp_turns and len({r["spk"] for r in refs}) > 1:
        d0 = der(ref_turns, hyp_turns, collar_s=0.0)
        d25 = der(ref_turns, hyp_turns, collar_s=0.25)
        mapping = d25["mapping"]
        agree = total = 0
        for i, w in enumerate(words):
            k = clip_of_word[i]
            if k < 0 or not w.get("spk"):
                continue
            total += 1
            if mapping.get(w["spk"]) == refs[k]["spk"]:
                agree += 1
        out["speakers"] = {"der_collar0": round(d0["der"], 4), "der_collar0.25": round(d25["der"], 4),
                           "confusion_collar0.25": round(d25["confusion"], 4),
                           "ref_speakers": d25["ref_speakers"], "hyp_speakers": d25["hyp_speakers"],
                           "word_speaker_agreement": round(agree / total, 4) if total else None,
                           "overlap": manifest.get("overlap")}
    return out


def score_run(run_dir: pathlib.Path, synth_dir: pathlib.Path) -> dict:
    doc = json.loads((run_dir / "transcript.json").read_text(encoding="utf-8"))
    return score(doc, synth_dir)
