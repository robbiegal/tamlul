"""Speaker metrics without meeteval/pyannote.metrics (design §4.15 ``score_diar.py``/``score_sa.py``).

* ``der``: NIST-style diarization error rate with a collar (seconds removed on
  each side of every reference boundary) and optional UEM regions; multi-speaker
  overlap counted per speaker; optimal speaker mapping by Hungarian assignment
  on overlap seconds. Also returns JER-style per-speaker error, miss/FA/confusion.
* ``cpwer``: concatenated minimum-permutation WER (meeteval's cpWER definition):
  per-speaker word streams concatenated, the assignment that minimises total
  edits (exhaustive for <= 8 speakers, Hungarian on pairwise edit counts above).
* ``speaker_count``: exact-match and |Δ|.

Turns are dicts ``{"s": ms, "e": ms, "spk": str}``; UEM is a list of (s, e) ms.
pyannote.metrics (eval extra) cross-checks ``der`` in the tests when importable.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

from ..text.normalize import edit_ops, ivrit_lb

Turn = dict


# ------------------------------------------------------------------ intervals

def _merge(iv: list[tuple[float, float]]) -> list[tuple[float, float]]:
    iv = sorted((a, b) for a, b in iv if b > a)
    out: list[tuple[float, float]] = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _intersect(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []
    i = j = 0
    a, b = _merge(a), _merge(b)
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _subtract(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []
    for s, e in _merge(a):
        cur = s
        for bs, be in _merge(b):
            if be <= cur or bs >= e:
                continue
            if bs > cur:
                out.append((cur, bs))
            cur = max(cur, be)
            if cur >= e:
                break
        if cur < e:
            out.append((cur, e))
    return out


def _total(iv: list[tuple[float, float]]) -> float:
    return sum(e - s for s, e in _merge(iv))


# ------------------------------------------------------------------ DER

def _hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    except ImportError:
        n, m = cost.shape
        if min(n, m) <= 8:
            best, best_val = [], float("inf")
            rows = list(range(n))
            for perm in itertools.permutations(range(m), min(n, m)):
                pairs = list(zip(rows, perm))
                v = sum(cost[i, j] for i, j in pairs)
                if v < best_val:
                    best, best_val = pairs, v
            return best
        # greedy fallback
        pairs, used_r, used_c = [], set(), set()
        for i, j in sorted(((i, j) for i in range(n) for j in range(m)), key=lambda p: cost[p]):
            if i in used_r or j in used_c:
                continue
            pairs.append((i, j))
            used_r.add(i)
            used_c.add(j)
        return pairs


def der(ref: list[Turn], hyp: list[Turn], *, collar_s: float = 0.0, uem: list[tuple[int, int]] | None = None,
        skip_overlap: bool = False) -> dict:
    """DER = (miss + false alarm + confusion) / reference speech, all in seconds, with collar and UEM."""
    def by_spk(turns):
        d: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for t in turns:
            d[str(t["spk"])].append((t["s"] / 1000.0, t["e"] / 1000.0))
        return {k: _merge(v) for k, v in d.items()}

    R, H = by_spk(ref), by_spk(hyp)
    # scoring region: UEM minus collars around every reference boundary
    if uem:
        region = _merge([(s / 1000.0, e / 1000.0) for s, e in uem])
    else:
        allv = [iv for v in list(R.values()) + list(H.values()) for iv in v]
        region = [(0.0, max((e for _, e in allv), default=0.0))]
    if collar_s > 0:
        collars = [(b - collar_s, b + collar_s) for v in R.values() for s, e in v for b in (s, e)]
        region = _subtract(region, collars)
    if skip_overlap:
        segs = sorted((s, e) for v in R.values() for s, e in v)
        ov = []
        for (s1, e1), (s2, e2) in itertools.combinations(segs, 2):
            if min(e1, e2) > max(s1, s2):
                ov.append((max(s1, s2), min(e1, e2)))
        region = _subtract(region, ov)
    R = {k: _intersect(v, region) for k, v in R.items()}
    H = {k: _intersect(v, region) for k, v in H.items()}
    rk, hk = sorted(R), sorted(H)
    total_ref = sum(_total(v) for v in R.values())
    if rk and hk:
        overlap = np.zeros((len(rk), len(hk)))
        for i, a in enumerate(rk):
            for j, b in enumerate(hk):
                overlap[i, j] = _total(_intersect(R[a], H[b]))
        pairs = _hungarian(-overlap)
        mapping = {hk[j]: rk[i] for i, j in pairs}
    else:
        mapping = {}
    # frame-free accounting on the union of boundaries
    bounds = sorted({b for v in list(R.values()) + list(H.values()) for s, e in v for b in (s, e)})
    miss = fa = conf = 0.0
    per_spk_err: dict[str, float] = defaultdict(float)
    per_spk_dur: dict[str, float] = defaultdict(float)
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        mid = (a + b) / 2
        dur = b - a
        r_active = {k for k, v in R.items() if any(s <= mid < e for s, e in v)}
        h_active = {mapping.get(k, f"?{k}") for k, v in H.items() if any(s <= mid < e for s, e in v)}
        nr, nh = len(r_active), len(h_active)
        correct = len(r_active & h_active)
        miss += max(0, nr - nh) * dur
        fa += max(0, nh - nr) * dur
        conf += (min(nr, nh) - correct) * dur
        for k in r_active:
            per_spk_dur[k] += dur
            if k not in h_active:
                per_spk_err[k] += dur
    jer = float(np.mean([per_spk_err[k] / per_spk_dur[k] for k in per_spk_dur if per_spk_dur[k] > 0])) if per_spk_dur else float("nan")
    return {"der": (miss + fa + conf) / total_ref if total_ref else float("nan"),
            "miss": miss / total_ref if total_ref else float("nan"),
            "false_alarm": fa / total_ref if total_ref else float("nan"),
            "confusion": conf / total_ref if total_ref else float("nan"),
            "jer_like": jer, "ref_speech_s": total_ref, "collar_s": collar_s, "mapping": mapping,
            "ref_speakers": len(rk), "hyp_speakers": len(hk)}


# ------------------------------------------------------------------ cpWER

def cpwer(ref_streams: dict[str, str], hyp_streams: dict[str, str]) -> dict:
    """Streams: speaker -> full text (any order of turns, concatenated in time). Normalized with ivrit_lb."""
    R = {k: ivrit_lb(v).split() for k, v in ref_streams.items()}
    H = {k: ivrit_lb(v).split() for k, v in hyp_streams.items()}
    rk, hk = sorted(R), sorted(H)
    n_ref = sum(len(v) for v in R.values())
    if not rk:
        return {"cpwer": float("nan"), "n": 0}
    # pad with empty streams so every ref/hyp speaker gets a partner
    size = max(len(rk), len(hk))
    rk2 = rk + [f"__r{i}" for i in range(size - len(rk))]
    hk2 = hk + [f"__h{i}" for i in range(size - len(hk))]
    R2 = {**R, **{k: [] for k in rk2 if k not in R}}
    H2 = {**H, **{k: [] for k in hk2 if k not in H}}
    cost = np.zeros((size, size))
    ops = {}
    for i, a in enumerate(rk2):
        for j, b in enumerate(hk2):
            s, d, ins, _ = edit_ops(R2[a], H2[b])
            cost[i, j] = s + d + ins
            ops[(i, j)] = (s, d, ins)
    pairs = _hungarian(cost) if size > 8 else min(
        (list(zip(range(size), perm)) for perm in itertools.permutations(range(size))),
        key=lambda ps: sum(cost[i, j] for i, j in ps))
    err = sum(cost[i, j] for i, j in pairs)
    S = sum(ops[(i, j)][0] for i, j in pairs)
    D = sum(ops[(i, j)][1] for i, j in pairs)
    Ins = sum(ops[(i, j)][2] for i, j in pairs)
    return {"cpwer": err / n_ref if n_ref else float("nan"), "S": int(S), "D": int(D), "I": int(Ins), "n": n_ref,
            "assignment": {hk2[j]: rk2[i] for i, j in pairs if not hk2[j].startswith("__") and not rk2[i].startswith("__")},
            "ref_speakers": len(rk), "hyp_speakers": len(hk)}


def streams_from_turns(turns: list[Turn], text_key: str = "text") -> dict[str, str]:
    d: dict[str, list[str]] = defaultdict(list)
    for t in sorted(turns, key=lambda t: t["s"]):
        d[str(t["spk"])].append(t.get(text_key, ""))
    return {k: " ".join(v) for k, v in d.items()}


def speaker_count(n_ref: int, n_hyp: int) -> dict:
    return {"ref": n_ref, "hyp": n_hyp, "exact": n_ref == n_hyp, "abs_delta": abs(n_ref - n_hyp)}


# ------------------------------------------------------------------ transcript.json adapters

def doc_turns(doc: dict) -> list[Turn]:
    return [{"s": t["s"], "e": t["e"], "spk": t["spk"], "text": t.get("text", "")} for t in doc.get("turns", [])]


def rttm_turns(text: str) -> list[Turn]:
    out = []
    for ln in text.splitlines():
        p = ln.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            s = float(p[3])
            out.append({"s": int(round(s * 1000)), "e": int(round((s + float(p[4])) * 1000)), "spk": p[7]})
    return out
