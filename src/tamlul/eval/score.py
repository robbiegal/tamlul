"""ASR scoring over per-utterance CSVs (design §4.15 ``score_asr.py``).

* corpus (micro) WER exactly as the leaderboard: Σ(S+D+I)/ΣN over the whole list,
  recomputed here from the texts with ``ivrit_lb`` (never trusted from the CSV);
* CER and space-insensitive CER (clitic/segmentation errors isolated);
* filler-insensitive WER (אה/אמ/הממ... dropped from both sides);
* Latin-token error rate, its transliteration-tolerant twin (a Hebrew word whose
  consonant skeleton matches the Latin reference token counts as correct), and
  the spurious-Latin rate (Latin tokens in the hypothesis where the reference has none);
* paired bootstrap 95 % CIs (10k resamples over utterances) for every rate and for
  the difference between two runs on the same ids.
"""

from __future__ import annotations

import pathlib
import re
from collections.abc import Sequence

import numpy as np

from ..text.correct import hebrew_skeleton, latin_skeleton
from ..text.normalize import FILLERS, edit_ops, ivrit_lb
from .runasr import load_csv

_LATIN = re.compile(r"^[A-Za-z][A-Za-z0-9'.-]*$")


def _counts(ref: str, hyp: str) -> tuple[int, int, int, int, int]:
    rt, ht = ref.split(), hyp.split()
    s, d, i, h = edit_ops(rt, ht)
    return s, d, i, h, len(rt)


def _char_counts(ref: str, hyp: str, *, space_insensitive: bool) -> tuple[int, int]:
    if space_insensitive:
        ref, hyp = ref.replace(" ", ""), hyp.replace(" ", "")
    s, d, i, _ = edit_ops(list(ref), list(hyp))
    return s + d + i, len(ref)


def _drop_fillers(text: str) -> str:
    return " ".join(t for t in text.split() if t not in FILLERS)


def _latin_counts(ref: str, hyp: str) -> tuple[int, int, int, int]:
    """(latin ref tokens, latin ref tokens missed, transliteration-tolerant misses, spurious latin hyp tokens)."""
    rt, ht = ref.split(), hyp.split()
    lat_ref = [t for t in rt if _LATIN.match(t)]
    if not lat_ref:
        spurious = sum(1 for t in ht if _LATIN.match(t))
        return 0, 0, 0, spurious
    hyp_lower = {t.lower() for t in ht}
    hyp_skel = {hebrew_skeleton(t) for t in ht if not _LATIN.match(t)}
    missed = tolerant_missed = 0
    for t in lat_ref:
        if t.lower() in hyp_lower:
            continue
        missed += 1
        if latin_skeleton(t) not in hyp_skel:
            tolerant_missed += 1
    return len(lat_ref), missed, tolerant_missed, 0


def per_utt(rows: Sequence[dict]) -> dict[str, np.ndarray]:
    """Per-utterance numerators/denominators so the bootstrap resamples utterances."""
    cols: dict[str, list] = {k: [] for k in ("err", "n", "cerr", "cn", "scerr", "scn", "ferr", "fn",
                                              "lat_n", "lat_miss", "lat_tol_miss", "spurious", "hyp_n")}
    for r in rows:
        ref = ivrit_lb(r.get("reference_text") or "")
        hyp = ivrit_lb(r.get("predicted_text") or "")
        s, d, i, _, n = _counts(ref, hyp)
        ce, cn = _char_counts(ref, hyp, space_insensitive=False)
        sce, scn = _char_counts(ref, hyp, space_insensitive=True)
        fs, fd, fi, _, fn = _counts(_drop_fillers(ref), _drop_fillers(hyp))
        ln, lm, ltm, sp = _latin_counts(ref, hyp)
        for k, v in (("err", s + d + i), ("n", n), ("cerr", ce), ("cn", cn), ("scerr", sce), ("scn", scn),
                     ("ferr", fs + fd + fi), ("fn", fn), ("lat_n", ln), ("lat_miss", lm), ("lat_tol_miss", ltm),
                     ("spurious", sp), ("hyp_n", len(hyp.split()))):
            cols[k].append(v)
    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}


RATES = {"wer": ("err", "n"), "cer": ("cerr", "cn"), "cer_nospace": ("scerr", "scn"), "wer_nofiller": ("ferr", "fn"),
         "latin_err": ("lat_miss", "lat_n"), "latin_err_tolerant": ("lat_tol_miss", "lat_n"),
         "spurious_latin": ("spurious", "hyp_n")}


def _rate(c: dict[str, np.ndarray], key: str, idx: np.ndarray | None = None) -> float:
    num, den = RATES[key]
    a, b = c[num], c[den]
    if idx is not None:
        a, b = a[idx], b[idx]
    d = float(b.sum())
    return float(a.sum()) / d if d else float("nan")


def bootstrap_ci(c: dict[str, np.ndarray], key: str, *, resamples: int = 10_000, seed: int = 0) -> tuple[float, float]:
    n = len(c["n"])
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    num, den = RATES[key]
    a, b = c[num], c[den]
    idx = rng.integers(0, n, size=(resamples, n))
    nums = a[idx].sum(axis=1)
    dens = b[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.where(dens > 0, nums / np.maximum(dens, 1e-9), np.nan)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def score(rows: Sequence[dict], *, resamples: int = 10_000) -> dict:
    c = per_utt(rows)
    out = {"n_utts": int(len(c["n"])), "ref_words": int(c["n"].sum()),
           "audio_s": round(float(sum(float(r.get("audio_duration") or 0) for r in rows)), 1),
           "wall_s": round(float(sum(float(r.get("transcription_time") or 0) for r in rows)), 1)}
    if out["wall_s"]:
        out["rtf"] = round(out["audio_s"] / out["wall_s"], 2)
    for k in RATES:
        v = _rate(c, k)
        lo, hi = bootstrap_ci(c, k, resamples=resamples)
        out[k] = {"value": round(v, 4) if v == v else None, "ci95": [round(lo, 4), round(hi, 4)] if lo == lo else None,
                  "n": int(c[RATES[k][1]].sum())}
    return out


def paired_diff(rows_a: Sequence[dict], rows_b: Sequence[dict], *, key: str = "wer", resamples: int = 10_000,
                seed: int = 0) -> dict:
    """b - a on the shared ids, with a paired bootstrap CI (same resampled utterances on both sides)."""
    by_a = {r["id"]: r for r in rows_a}
    by_b = {r["id"]: r for r in rows_b}
    ids = [i for i in by_a if i in by_b]
    ca, cb = per_utt([by_a[i] for i in ids]), per_utt([by_b[i] for i in ids])
    num, den = RATES[key]
    n = len(ids)
    if n == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    va = ca[num][idx].sum(axis=1) / np.maximum(ca[den][idx].sum(axis=1), 1e-9)
    vb = cb[num][idx].sum(axis=1) / np.maximum(cb[den][idx].sum(axis=1), 1e-9)
    diff = vb - va
    return {"n": n, "key": key, "a": round(_rate(ca, key), 4), "b": round(_rate(cb, key), 4),
            "diff": round(_rate(cb, key) - _rate(ca, key), 4),
            "diff_ci95": [round(float(np.percentile(diff, 2.5)), 4), round(float(np.percentile(diff, 97.5)), 4)],
            "p_b_worse": round(float(np.mean(diff > 0)), 3)}


def score_file(path: pathlib.Path, **kw) -> dict:
    return score(load_csv(path), **kw)


def compare_files(a: pathlib.Path, b: pathlib.Path, **kw) -> dict:
    return paired_diff(load_csv(a), load_csv(b), **kw)
