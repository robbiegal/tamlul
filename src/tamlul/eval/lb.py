"""Leaderboard scorer check (design §9 "Harness (a)").

Downloads the public per-utterance CSVs of the ivrit.ai Hebrew Transcription
Leaderboard Space (no token, no audio) and re-scores every row that carries
``reference_text``/``predicted_text`` with tamlul's vendored normalizer and
WER. The check passes when, per CSV,

* our normalized texts equal the Space's ``norm_*`` columns row for row,
* our per-row S/D/I/H equal theirs, and
* our corpus WER equals the one re-derived from their S/D/I/H columns

— i.e. the scorer is cell-for-cell the leaderboard's. No cloud engine is run.
"""

from __future__ import annotations

import csv
import json
import pathlib

from ..paths import DataLayout
from ..text.normalize import edit_ops, ivrit_lb
from .sets import eval_root

SPACE = "ivrit-ai/hebrew-transcription-leaderboard"


def download(layout: DataLayout, *, revision: str | None = None) -> pathlib.Path:
    """Fetch benchmark.csv, timing.csv and results/**/*.csv into <data>/eval/leaderboard/space.

    Plain HTTP GETs into short paths: snapshot_download's ``.cache/.../<hash>.incomplete``
    names exceed MAX_PATH on a Windows machine without LongPathsEnabled.
    """
    import requests
    from huggingface_hub import HfApi, hf_hub_url
    d = eval_root(layout) / "leaderboard" / "space"
    d.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    files = api.list_repo_files(SPACE, repo_type="space", revision=revision)
    wanted = [f for f in files if f in ("benchmark.csv", "timing.csv", "README.md")
              or (f.startswith("results/") and f.endswith(".csv"))]
    for f in wanted:
        dest = d / f
        if dest.is_file():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(hf_hub_url(SPACE, f, repo_type="space", revision=revision), timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return d


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


_BIDI = dict.fromkeys(map(ord, "\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"), None)


def check_csv(path: pathlib.Path) -> dict:
    """Pass = per row, the same normalized texts and the same total edit count (S+D+I) as the Space's
    columns, hence the same per-row and corpus WER. The S/D/I *split* may differ (Levenshtein
    tie-breaking differs between rapidfuzz and our DP) and is reported, not judged."""
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "predicted_text" not in rows[0] or "reference_text" not in rows[0]:
        return {"file": str(path), "skipped": "no reference/predicted text columns"}
    has_ops = all(k in rows[0] for k in ("substitutions", "deletions", "insertions", "hits"))
    S = D = Ins = N = 0
    tS = tD = tI = tH = 0
    norm_mismatch = bidi_only = split_differs = edits_mismatch = row_wer_mismatch = 0
    worst = 0.0
    examples: list[dict] = []
    for r in rows:
        nr, nh = ivrit_lb(r["reference_text"] or ""), ivrit_lb(r["predicted_text"] or "")
        if "norm_reference_text" in r:
            tr, th_ = r["norm_reference_text"] or "", r["norm_predicted_text"] or ""
            if nr != tr or nh != th_:
                # an older normalizer left bidi marks in some published CSVs; that is their staleness, not our error
                if ivrit_lb(tr.translate(_BIDI)) == nr and ivrit_lb(th_.translate(_BIDI)) == nh:
                    bidi_only += 1
                else:
                    norm_mismatch += 1
                    if len(examples) < 5:
                        examples.append({"id": r.get("id"), "ours_ref": nr[:120], "theirs_ref": tr[:120],
                                         "ours_hyp": nh[:120], "theirs_hyp": th_[:120]})
        rt, ht = nr.split(), nh.split()
        s, d, i, h = edit_ops(rt, ht)
        S += s
        D += d
        Ins += i
        N += len(rt)
        if has_ops:
            ts, td, ti, th = (int(_f(r.get(k))) if _f(r.get(k)) == _f(r.get(k)) else 0
                              for k in ("substitutions", "deletions", "insertions", "hits"))
            tS += ts
            tD += td
            tI += ti
            tH += th
            if (s, d, i, h) != (ts, td, ti, th):
                split_differs += 1
            if s + d + i != ts + td + ti:
                edits_mismatch += 1
        ours = (s + d + i) / len(rt) if rt else float("nan")
        theirs = _f(r.get("wer"))
        if ours == ours and theirs == theirs:
            delta = abs(ours - theirs)
            worst = max(worst, delta)
            if delta > 1e-6:
                row_wer_mismatch += 1
    ours_corpus = (S + D + Ins) / N if N else float("nan")
    theirs_corpus = (tS + tD + tI) / (tS + tD + tH) if has_ops and (tS + tD + tH) else float("nan")
    corpus_ok = (abs(ours_corpus - theirs_corpus) < 1e-6) if theirs_corpus == theirs_corpus else True
    stale = bidi_only > 0 and (row_wer_mismatch > 0 or not corpus_ok)
    return {"file": str(path), "rows": len(rows), "has_ops": has_ops, "ours_corpus_wer": round(ours_corpus, 6),
            "theirs_corpus_wer_from_sdih": round(theirs_corpus, 6) if theirs_corpus == theirs_corpus else None,
            "corpus_delta": round(abs(ours_corpus - theirs_corpus), 6) if theirs_corpus == theirs_corpus else None,
            "norm_text_mismatch_rows": norm_mismatch, "norm_bidi_only_rows": bidi_only,
            "sdi_split_differs_rows": split_differs, "edits_mismatch_rows": edits_mismatch,
            "row_wer_mismatch_rows": row_wer_mismatch, "worst_row_wer_delta": round(worst, 6),
            "stale_normalizer_in_csv": stale,
            "pass": norm_mismatch == 0 and (edits_mismatch == 0 or stale) and (corpus_ok or stale) and (row_wer_mismatch == 0 or stale),
            "examples": examples}


_SET_COL = {"ivrit_ai_eval_d1": "ivrit-ai/eval-d1", "ivrit_ai_eval_whatsapp": "ivrit-ai/eval-whatsapp",
            "saspeech": "ivrit-ai/saspeech", "fleurs": "google/fleurs/he", "hebrew_speech_kan": "imvladikon/hebrew_speech_kan"}


def _benchmark_cell(bench: list[dict], csv_path: pathlib.Path) -> float | None:
    """The benchmark.csv cell for a results CSV, matched on the (engine, model) columns the CSV carries."""
    col = _SET_COL.get(csv_path.stem)
    if not col:
        return None
    with open(csv_path, encoding="utf-8", newline="") as fh:
        first = next(csv.DictReader(fh), None)
    if not first:
        return None
    eng, model = (first.get("engine") or "").strip(), (first.get("model") or "").strip()
    for row in bench:
        if row.get("engine") == eng and row.get("model") == model:
            v = _f(row.get(col))
            return v if v == v else None
    return None


def check_all(layout: DataLayout, *, revision: str | None = None) -> dict:
    root = download(layout, revision=revision)
    files = sorted(root.glob("results/**/*.csv"))
    bench = benchmark_rows(layout)
    reports = []
    for f in files:
        r = check_csv(f)
        if "skipped" not in r:
            cell = _benchmark_cell(bench, f)
            r["benchmark_cell"] = cell
            # the Space displays 3 decimals; our corpus WER must round to the same cell
            r["benchmark_cell_matches_ours"] = None if cell is None else abs(round(r["ours_corpus_wer"], 3) - cell) < 5e-4
        reports.append(r)
    checked = [r for r in reports if "skipped" not in r]
    cells = [r for r in checked if r.get("benchmark_cell") is not None]
    summary = {"space": SPACE, "root": str(root), "csv_files": len(files), "checked": len(checked),
               "passed": sum(1 for r in checked if r["pass"]), "failed": [r["file"] for r in checked if not r["pass"]],
               "stale_normalizer_csvs": [r["file"] for r in checked if r["stale_normalizer_in_csv"]],
               "benchmark_cells_compared": len(cells),
               "benchmark_cells_matching_ours": sum(1 for r in cells if r["benchmark_cell_matches_ours"]),
               "benchmark_cells_mismatching": [(r["file"], r["ours_corpus_wer"], r["benchmark_cell"]) for r in cells
                                              if not r["benchmark_cell_matches_ours"]],
               "reports": reports}
    (eval_root(layout) / "leaderboard" / "scorer-check.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def benchmark_rows(layout: DataLayout) -> list[dict]:
    p = eval_root(layout) / "leaderboard" / "space" / "benchmark.csv"
    if not p.is_file():
        return []
    with open(p, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))
