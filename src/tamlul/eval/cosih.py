"""CoSIH (Corpus of Spoken Israeli Hebrew, Tel Aviv University) as a conversation reference (design §9).

Non-commercial licence, evaluated on and never shipped. Per recording the site
offers ``<id>.wav`` and ``<id>_ELAN.zip``; the ELAN file has one aligned tier
per speaker (``sp1..spN``: intonation units in standard orthography with
prosodic marks) plus ``ref_spN`` mirror tiers and, in some files, unidentified
or group tiers. Conversion (review acc-3):

* speaker set = the ``spN`` tiers; ``unidentified``/``undefined``/group tiers
  (``sp4_male-voices``, ``PA&radio``...) become UEM-excluded regions and the
  excluded seconds go into the manifest;
* RTTM-style turns carry every IU; the STM-style text strips prosodic marks
  (``|``, ``||``, ``\\``, ``/``), non-verbal ``<...>``, elided ``(x)`` and drops
  IUs with ``@`` (unintelligible) from the text streams only;
* tier B (timestamps): IU onsets whose previous IU on the same tier ended
  >= 150 ms earlier and whose text does not start with a truncation ``-``.
"""

from __future__ import annotations

import json
import pathlib
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict

from ..paths import DataLayout
from . import speakers as spk_metrics
from .sets import eval_root

BASE = "http://cosih.com/CoSIH_files"
# 31 usable recordings (C711_2 returns 404 as of 2026-09-16)
IDS = ["C1624", "C514_1", "C514_2", "C612_2", "C612_3", "C612_4", "C711_0", "C711_1", "C711_3", "C711_4", "C712_2",
       "C714", "C842", "D142", "D932", "D933", "OCD_1", "OCD_2", "OCh", "OM", "P311_2", "P423_1", "P423_2", "P931_1",
       "P931_2", "P931_3", "Y111", "Y311", "Y32", "Y33", "Y34"]

SPEAKER_TIER = re.compile(r"^sp\d+$")
_MARKS = re.compile(r"[|\\/]+")
_NONVERBAL = re.compile(r"<[^>]*>")
_ELIDED = re.compile(r"\([^)]*\)")


def cosih_dir(layout: DataLayout) -> pathlib.Path:
    return eval_root(layout) / "cosih"


def download(layout: DataLayout, ids: list[str] | None = None, *, audio: bool = True, log=print) -> list[pathlib.Path]:
    import requests
    out = []
    for rid in ids or IDS:
        d = cosih_dir(layout) / rid
        d.mkdir(parents=True, exist_ok=True)
        for name in ([f"{rid}_ELAN.zip"] + ([f"{rid}.wav"] if audio else [])):
            dest = d / name
            if dest.is_file() and dest.stat().st_size > 0:
                continue
            url = f"{BASE}/{rid}/{name}"
            r = requests.get(url, timeout=600, stream=True)
            if r.status_code != 200:
                log(f"{url}: HTTP {r.status_code}")
                continue
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
            log(f"fetched {url} ({dest.stat().st_size/1e6:.1f} MB)")
        out.append(d)
    return out


def parse_eaf(text: str) -> dict[str, list[tuple[int, int, str]]]:
    root = ET.fromstring(text)
    slots = {ts.get("TIME_SLOT_ID"): int(ts.get("TIME_VALUE")) for ts in root.iter("TIME_SLOT") if ts.get("TIME_VALUE")}
    tiers: dict[str, list[tuple[int, int, str]]] = {}
    for tier in root.iter("TIER"):
        tid = tier.get("TIER_ID") or ""
        items = []
        for ann in tier.iter("ALIGNABLE_ANNOTATION"):
            s, e = slots.get(ann.get("TIME_SLOT_REF1")), slots.get(ann.get("TIME_SLOT_REF2"))
            if s is None or e is None or e <= s:
                continue
            val = ann.findtext("ANNOTATION_VALUE") or ""
            items.append((s, e, val.strip()))
        items.sort()
        tiers[tid] = items
    return tiers


def read_eaf_zip(path: pathlib.Path) -> dict[str, list[tuple[int, int, str]]]:
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith(".eaf") and "__MACOSX" not in n]
        if not names:
            raise RuntimeError(f"{path}: no .eaf inside")
        return parse_eaf(z.read(names[0]).decode("utf-8"))


def clean_iu(text: str) -> str | None:
    """STM text of an intonation unit; None when it must be dropped from text scoring."""
    if "@" in text:
        return None
    t = _NONVERBAL.sub(" ", text)
    t = _ELIDED.sub("", t)
    t = _MARKS.sub(" ", t)
    toks = [w for w in t.split() if w not in ("-", "--")]
    return " ".join(toks) if toks else None


def to_reference(tiers: dict[str, list[tuple[int, int, str]]]) -> dict:
    speakers = sorted(t for t in tiers if SPEAKER_TIER.match(t))
    excluded_tiers = [t for t in tiers if not SPEAKER_TIER.match(t) and not t.startswith("ref_") and t != "Note"]
    turns, texts = [], []
    boundaries = []
    for t in speakers:
        prev_end = -10**9
        for s, e, val in tiers[t]:
            turns.append({"s": s, "e": e, "spk": t})
            ct = clean_iu(val)
            if ct:
                texts.append({"s": s, "e": e, "spk": t, "text": ct})
            if s - prev_end >= 150 and not val.startswith("-") and ct:
                boundaries.append({"s": s, "spk": t})
            prev_end = e
    excluded = [(s, e) for t in excluded_tiers for s, e, _ in tiers[t]]
    end = max([e for t in speakers for _, e, _ in tiers[t]] + [e for _, e in excluded] + [0])
    return {"speakers": speakers, "excluded_tiers": excluded_tiers, "turns": sorted(turns, key=lambda x: x["s"]),
            "text_turns": sorted(texts, key=lambda x: x["s"]), "excluded_regions_ms": sorted(excluded),
            "excluded_s": round(sum(e - s for s, e in excluded) / 1000, 1), "end_ms": end,
            "tier_b_onsets": sorted(boundaries, key=lambda x: x["s"]),
            "words": sum(len(x["text"].split()) for x in texts)}


def convert(layout: DataLayout, rid: str) -> pathlib.Path:
    d = cosih_dir(layout) / rid
    ref = to_reference(read_eaf_zip(d / f"{rid}_ELAN.zip"))
    ref["id"] = rid
    (d / "reference.json").write_text(json.dumps(ref, ensure_ascii=False), encoding="utf-8")
    return d / "reference.json"


def _uem(ref: dict) -> list[tuple[int, int]]:
    """Scoring region: [0, end] minus the excluded (unidentified/group) tier regions."""
    from .speakers import _subtract
    region = _subtract([(0.0, ref["end_ms"] / 1000.0)], [(s / 1000.0, e / 1000.0) for s, e in ref["excluded_regions_ms"]])
    return [(int(s * 1000), int(e * 1000)) for s, e in region]


def score(doc: dict, ref: dict) -> dict:
    hyp = spk_metrics.doc_turns(doc)
    uem = _uem(ref)
    d0 = spk_metrics.der(ref["turns"], hyp, collar_s=0.0, uem=uem)
    d25 = spk_metrics.der(ref["turns"], hyp, collar_s=0.25, uem=uem)
    cp = spk_metrics.cpwer(spk_metrics.streams_from_turns(ref["text_turns"]), spk_metrics.streams_from_turns(hyp))
    # tier B: nearest hypothesis *word* onset to each eligible IU onset (IUs are prosodic phrases; the
    # document's turns merge many IUs, so turn starts would measure segmentation, not timing)
    hyp_starts = sorted(w["s"] for w in doc.get("words", []) if w.get("t_raw", w.get("t", "")).strip())
    errs = []
    for b in ref["tier_b_onsets"]:
        if hyp_starts:
            errs.append(min(abs(h - b["s"]) for h in hyp_starts))
    tier_b = {"n": len(errs)}
    if errs:
        import numpy as np
        a = np.asarray(errs, dtype=float)
        tier_b.update({"median_ms": round(float(np.median(a)), 1), "p95_ms": round(float(np.percentile(a, 95)), 1),
                       "within_100ms": round(float(np.mean(a <= 100)), 4), "within_200ms": round(float(np.mean(a <= 200)), 4)})
    return {"id": ref.get("id"), "der_collar0": round(d0["der"], 4), "der_collar0.25": round(d25["der"], 4),
            "miss": round(d25["miss"], 4), "false_alarm": round(d25["false_alarm"], 4), "confusion": round(d25["confusion"], 4),
            "jer_like": round(d25["jer_like"], 4), "cpwer": round(cp["cpwer"], 4), "cpwer_n": cp["n"],
            "speaker_count": spk_metrics.speaker_count(len(ref["speakers"]), len({t["spk"] for t in hyp})),
            "excluded_s": ref["excluded_s"], "excluded_tiers": ref["excluded_tiers"], "tier_b_onsets": tier_b,
            "licence": "CoSIH, TAU, non-commercial; aggregates only are published"}


def aggregate(results: list[dict]) -> dict:
    """Corpus-level numbers over recordings (DER weighted by reference speech is what the design publishes;
    here: mean and median across files plus the exact-count rate) - per-file numbers stay local."""
    import numpy as np
    if not results:
        return {}
    keys = ["der_collar0", "der_collar0.25", "cpwer"]
    out = {k: {"mean": round(float(np.mean([r[k] for r in results])), 4),
               "median": round(float(np.median([r[k] for r in results])), 4)} for k in keys}
    out["speaker_count_exact_rate"] = round(sum(1 for r in results if r["speaker_count"]["exact"]) / len(results), 3)
    out["files"] = len(results)
    return out


def load_reference(layout: DataLayout, rid: str) -> dict:
    p = cosih_dir(layout) / rid / "reference.json"
    if not p.is_file():
        convert(layout, rid)
    return json.loads(p.read_text(encoding="utf-8"))


def stats(ref: dict) -> dict:
    per = defaultdict(int)
    for t in ref["turns"]:
        per[t["spk"]] += t["e"] - t["s"]
    return {"speakers": len(ref["speakers"]), "ius": len(ref["turns"]), "words": ref["words"],
            "labelled_s": round(sum(per.values()) / 1000, 1), "excluded_s": ref["excluded_s"],
            "tier_b_onsets": len(ref["tier_b_onsets"])}
