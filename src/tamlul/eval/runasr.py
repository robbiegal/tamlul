"""Run one ASR configuration over an evaluation set -> per-utterance CSV (design §4.15).

Column layout is the leaderboard's (``asr-training/evaluate_model.py``) so any
row can be re-scored by their code:

    id, reference_text, predicted_text, norm_reference_text, norm_predicted_text,
    wer, wil, substitutions, deletions, insertions, hits, audio_duration, transcription_time

Two decoding columns:

* ``staff``  - faster-whisper defaults + ``language="he"`` (what produced every
  published ivrit.ai number: beam 5, best_of 5, the full temperature ladder,
  condition_on_previous_text=True, no VAD, whole utterance in one call);
* ``plugin`` - the plugin's ``DecodeOpts`` on the plugin's plan (VAD groups,
  RMS gate), i.e. what a user gets.

Rows are appended as they finish and skipped on re-run, so a 3-6 h CPU run
survives interruption. ``machine.json`` next to the CSV records versions and
hardware.
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import platform
import time
from dataclasses import replace

import numpy as np

from .. import __version__, registry
from ..asr import get_backend
from ..asr.base import ModelRef
from ..asr.decode import DecodeOpts
from ..asr.recover import recover_missed
from ..doctor import _pkg_versions, physical_cores
from ..media import SR
from ..paths import DataLayout
from ..plan import CLIP_MODE, build_plan, decode_clips
from ..text.normalize import edit_ops, ivrit_lb
from .sets import SETS, Utt, eval_root, utterances

COLUMNS = ["id", "reference_text", "predicted_text", "norm_reference_text", "norm_predicted_text", "wer", "wil",
           "substitutions", "deletions", "insertions", "hits", "audio_duration", "transcription_time"]

STAFF_OPTS = DecodeOpts(temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold=2.4,
                        condition_on_previous_text=True, word_timestamps=False)


def model_ref(model_id: str, layout: DataLayout) -> ModelRef:
    art = registry.artefacts()[model_id]
    if not registry.is_installed(art, layout):
        raise SystemExit(f"{model_id} is not installed: tamlul models install {model_id} --yes")
    return ModelRef(id=model_id, dir=str(registry.local_dir(art, layout)), revision=art.revision, commit=art.commit,
                    sha256=art.files.get(art.primary, {}).get("sha256", ""))


def result_path(layout: DataLayout, set_name: str, model_id: str, compute_type: str, column: str,
                tag: str = "") -> pathlib.Path:
    safe = model_id.replace("/", "__").replace("@", "-")
    name = f"{safe}__{compute_type}__{column}" + (f"__{tag}" if tag else "") + ".csv"
    return eval_root(layout) / "results" / set_name / name


def _wil(s: int, d: int, i: int, h: int, n_ref: int, n_hyp: int) -> float:
    # jiwer's WIL: 1 - (H/N_ref)*(H/N_hyp)
    if n_ref == 0 or n_hyp == 0:
        return 1.0 if (n_ref or n_hyp) else 0.0
    return 1.0 - (h / n_ref) * (h / n_hyp)


def score_row(ref: str, hyp: str) -> dict:
    nr, nh = ivrit_lb(ref), ivrit_lb(hyp)
    rt, ht = nr.split(), nh.split()
    s, d, i, h = edit_ops(rt, ht)
    n = len(rt)
    return {"norm_reference_text": nr, "norm_predicted_text": nh,
            "wer": (s + d + i) / n if n else (1.0 if ht else 0.0), "wil": _wil(s, d, i, h, n, len(ht)),
            "substitutions": s, "deletions": d, "insertions": i, "hits": h}


def _decode_staff(backend, utt: Utt, opts: DecodeOpts) -> str:
    segs = backend.transcribe(utt.audio, None, opts)
    return " ".join(s.text.strip() for s in segs if s.text.strip())


def _join(segs, join: str) -> list[str]:
    """``words``: the plugin's word list (what the transcript carries); ``text``: the segment text."""
    out: list[str] = []
    for seg in segs:
        if join == "words" and seg.words:
            out += [w.t.strip() for w in seg.words if w.t.strip()]
        else:
            out += seg.text.split()
    return out


def _decode_plugin(backend, utt: Utt, opts: DecodeOpts, *, plan: bool = True, join: str = "words",
                   clip_mode: str = CLIP_MODE, recover: bool = True) -> str:
    if not plan:
        return " ".join(_join(backend.transcribe(utt.audio, None, opts), join))
    p = build_plan(utt.audio)
    words: list[str] = []
    for g in p.groups:
        sl = utt.audio[int(g.start_s * SR): int(g.end_s * SR)]
        segs = [(seg, g.start_s) for seg in backend.transcribe(sl, decode_clips(g, clip_mode), opts)]
        if recover:   # as the pipeline does: re-decode raw speech the decoder skipped, splice by time
            segs = sorted(segs + recover_missed(backend, utt.audio, g, segs, p.raw_regions, opts),
                          key=lambda x: x[0].s + x[1])
        words += _join([x for x, _ in segs], join)
    return " ".join(words)


def parse_overrides(items: list[str]) -> dict:
    """``key=value`` overrides for DecodeOpts, coerced from the field's default type
    (bool: true/false; tuples: comma-separated)."""
    from dataclasses import fields
    types = {f.name: f.default for f in fields(DecodeOpts)}
    out: dict = {}
    for it in items:
        k, _, v = it.partition("=")
        if k not in types:
            raise SystemExit(f"unknown DecodeOpts field {k!r}; known: {', '.join(sorted(types))}")
        d = types[k]
        if isinstance(d, bool):
            out[k] = v.lower() in ("1", "true", "yes", "on")
        elif isinstance(d, tuple):
            out[k] = tuple(type(d[0])(x) for x in v.split(",") if x != "") if d else tuple(v.split(","))
        elif d is None or isinstance(d, str):
            out[k] = v or None
        else:
            out[k] = type(d)(v)
    return out


def run(set_name: str, model_id: str, layout: DataLayout, *, column: str = "plugin", compute_type: str = "int8",
        threads: int = 0, limit: int | None = None, tag: str = "", temperature0: bool = False,
        overrides: dict | None = None, plan: bool = True, join: str = "words", itn: bool = False,
        clip_mode: str = CLIP_MODE, recover: bool = True, log=print) -> pathlib.Path:
    """``overrides``/``plan``/``join``/``itn`` are the ablation knobs (phase 4): DecodeOpts field overrides,
    whole-utterance decoding instead of the plugin's plan, segment-text instead of word join, and the
    pipeline's ITN applied to the hypothesis (what a user gets on the meeting/voice-note profiles)."""
    s = SETS[set_name]
    out = result_path(layout, set_name, model_id, compute_type, column, tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out.is_file():
        with open(out, encoding="utf-8", newline="") as fh:
            done = {r["id"] for r in csv.DictReader(fh)}
    threads = threads or physical_cores()
    model = model_ref(model_id, layout)
    be = get_backend("ct2")
    t0 = time.time()
    be.load(model, "cpu", threads, compute_type)
    log(f"{set_name}: {model_id} {compute_type} threads={threads} column={column} loaded in {time.time()-t0:.1f}s; "
        f"{len(done)} rows already done -> {out}")
    if column == "staff":
        opts = STAFF_OPTS
    elif column == "plugin":
        opts = DecodeOpts()
        assert not opts.validate()
    else:
        raise SystemExit("column must be staff|plugin")
    if temperature0:
        opts = replace(opts, temperature=(0.0,))
    if overrides:
        opts = replace(opts, **overrides)
    if itn:
        from ..itn import itn_text
    ablation = {"plan": plan, "clip_mode": clip_mode, "recover": recover, "join": join, "itn": itn,
                "overrides": overrides or {}}
    machine = {"ablation": ablation, "tamlul": __version__, "model": model.id, "revision": model.revision, "commit": model.commit,
               "model_sha256": model.sha256, "compute_type": compute_type, "threads": threads, "column": column,
               "decode_opts": opts.as_dict(), "packages": _pkg_versions(), "cpu": platform.processor(),
               "platform": platform.platform(), "logical_cpus": os.cpu_count(), "set": set_name,
               "licence_note": s.licence, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out.with_name(out.stem + ".machine.json").write_text(json.dumps(machine, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
    new_file = not out.is_file()
    n = len(done)
    audio_total = wall_total = 0.0
    with open(out, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        for utt in utterances(set_name, layout, limit=limit):
            if utt.id in done:
                continue
            t1 = time.time()
            hyp = (_decode_staff(be, utt, opts) if column == "staff"
                   else _decode_plugin(be, utt, opts, plan=plan, join=join, clip_mode=clip_mode, recover=recover))
            if itn:
                hyp = itn_text(hyp)
            dt = time.time() - t1
            row = {"id": utt.id, "reference_text": utt.text, "predicted_text": hyp,
                   "audio_duration": round(utt.duration_s, 3), "transcription_time": round(dt, 3)}
            row.update(score_row(utt.text, hyp))
            w.writerow(row)
            fh.flush()
            n += 1
            audio_total += utt.duration_s
            wall_total += dt
            if n % 10 == 0:
                log(f"  {n} rows, x{audio_total/wall_total:.2f} RT so far, last wer {row['wer']:.3f}")
    be.unload()
    log(f"done: {n} rows in {out}")
    return out


def load_csv(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def strip_for_publication(path: pathlib.Path, set_name: str, dest: pathlib.Path) -> pathlib.Path:
    """Publication rule (§4.15): reference text leaves the CSV for every set but FLEURS."""
    s = SETS[set_name]
    rows = load_csv(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    keep = COLUMNS if s.publish_refs else [c for c in COLUMNS if c not in ("reference_text", "norm_reference_text")]
    with open(dest, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keep)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keep})
    return dest


def rehydrate(published: pathlib.Path, set_name: str, layout: DataLayout, dest: pathlib.Path) -> pathlib.Path:
    """Re-join stripped references from the user's own download of the gated set."""
    refs = {u.id: u.text for u in utterances(set_name, layout)}
    rows = load_csv(published)
    with open(dest, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            ref = refs.get(r["id"], "")
            r = dict(r, reference_text=ref)
            r.update(score_row(ref, r["predicted_text"]))
            w.writerow({k: r.get(k, "") for k in COLUMNS})
    return dest


def audio_seconds(rows: list[dict]) -> float:
    return float(np.sum([float(r.get("audio_duration") or 0) for r in rows]))
