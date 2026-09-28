"""The diarization child process (design §4.4).

Owns torch and its OpenMP runtime so CTranslate2's never shares a process
with it. Reads ``audio.16k.f32`` from the run directory, runs pyannote
community-1 from the local model directory (zero Hub requests), and writes
``diarization.rttm`` and ``diarization.json`` atomically.

    python -m tamlul.diarize.child --run <dir> --model <dir> [--num-speakers N]
        [--min-speakers N] [--max-speakers N] [--fast-embeddings on|off]
        [--threads 6] [--equality-check]  [--embeddings-out file.npz]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Telemetry off BEFORE any pyannote import (the OTLP exporter is created at import time).
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

SR = 16000


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _annotation_to_turns(ann) -> list[dict]:
    out = []
    for seg, _, label in ann.itertracks(yield_label=True):
        out.append({"s": int(round(seg.start * 1000)), "e": int(round(seg.end * 1000)), "spk": str(label)})
    out.sort(key=lambda t: (t["s"], t["spk"]))
    return out


def _rttm(turns: list[dict], file_id: str) -> str:
    lines = [f"SPEAKER {file_id} 1 {t['s']/1000:.3f} {(t['e']-t['s'])/1000:.3f} <NA> <NA> {t['spk']} <NA> <NA>"
             for t in turns]
    return "\n".join(lines) + ("\n" if lines else "")


def load_pipeline(model_dir: str, *, fast: bool):
    from pyannote.audio import Pipeline
    pipe = Pipeline.from_pretrained(model_dir)
    if fast:
        from .._vendor.pyannote_fast import FastSpeakerDiarization, supports_multi_speaker_masks
        if supports_multi_speaker_masks(pipe._embedding):
            pipe.__class__ = FastSpeakerDiarization
            return pipe, True
        _log("fast embeddings requested but the embedding model does not support stacked masks; using stock")
    return pipe, False


def run(run_dir: str, model_dir: str, *, num_speakers: int | None, min_speakers: int | None,
        max_speakers: int | None, fast: bool, threads: int, equality_check: bool, embeddings_out: str | None,
        audio_file: str | None = None) -> int:
    import torch
    torch.set_num_threads(max(1, threads))
    t0 = time.time()
    audio_path = audio_file or os.path.join(run_dir, "audio.16k.f32")
    audio = np.fromfile(audio_path, dtype="<f4").astype(np.float32)
    wav = torch.from_numpy(audio)[None]
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = int(num_speakers)
    if min_speakers:
        kwargs["min_speakers"] = int(min_speakers)
    if max_speakers:
        kwargs["max_speakers"] = int(max_speakers)

    def hook(step, *_, total=None, completed=None, **__):
        if total:
            prog = os.path.join(run_dir, "progress.json")
            try:
                cur = json.loads(open(prog, encoding="utf-8").read()) if os.path.isfile(prog) else {}
            except ValueError:
                cur = {}
            cur.update({"stage": "diarize", "substage": step, "sub_pct": round(100 * completed / total, 1),
                        "last_update": time.strftime("%Y-%m-%dT%H:%M:%S")})
            tmp = prog + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(cur, fh)
            os.replace(tmp, prog)

    pipe, is_fast = load_pipeline(model_dir, fast=fast)
    t_load = time.time() - t0
    _log(f"pyannote loaded in {t_load:.1f}s (fast_embeddings={is_fast}, threads={threads})")
    t1 = time.time()
    out = pipe({"waveform": wav, "sample_rate": SR}, hook=hook, **kwargs)
    t_apply = time.time() - t1
    turns = _annotation_to_turns(out.speaker_diarization)
    exclusive = _annotation_to_turns(out.exclusive_speaker_diarization)
    labels = sorted({t["spk"] for t in turns})
    emb = {}
    if getattr(out, "speaker_embeddings", None) is not None:
        try:
            for k, lab in enumerate(out.speaker_diarization.labels()):
                emb[str(lab)] = [float(x) for x in np.asarray(out.speaker_embeddings)[k]]
        except Exception:  # noqa: BLE001
            emb = {}
    result = {
        "turns": turns, "exclusive": exclusive, "labels": labels, "embeddings": emb,
        "info": {"engine": "pyannote", "pipeline": type(pipe).__name__, "model_dir": model_dir,
                 "fast_embeddings": is_fast, "threads": threads, "hints": kwargs,
                 "load_s": round(t_load, 1), "apply_s": round(t_apply, 1), "audio_s": round(len(audio) / SR, 1),
                 "rtf": round(len(audio) / SR / t_apply, 2) if t_apply else None},
    }
    if equality_check:
        result["info"]["equality"] = _equality(model_dir, wav, kwargs, turns, hook=None)
    file_id = os.path.basename(os.path.normpath(run_dir))
    rttm_tmp = os.path.join(run_dir, "diarization.rttm.tmp")
    with open(rttm_tmp, "w", encoding="utf-8") as fh:
        fh.write(_rttm(turns, file_id))
    os.replace(rttm_tmp, os.path.join(run_dir, "diarization.rttm"))
    js_tmp = os.path.join(run_dir, "diarization.json.tmp")
    with open(js_tmp, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False)
    os.replace(js_tmp, os.path.join(run_dir, "diarization.json"))
    _log(f"diarization done: {len(labels)} speakers, {len(turns)} turns, apply {t_apply:.1f}s "
         f"(x{result['info']['rtf']} RT)")
    return 0


def _equality(model_dir: str, wav, kwargs: dict, fast_turns: list[dict], hook=None) -> dict:
    """Stock vs fast on the same audio: identical RTTM and allclose embeddings (design §4.4)."""
    stock, _ = load_pipeline(model_dir, fast=False)
    out = stock({"waveform": wav, "sample_rate": SR}, **kwargs)
    stock_turns = _annotation_to_turns(out.speaker_diarization)
    same = stock_turns == fast_turns
    return {"identical_rttm": bool(same), "stock_turns": len(stock_turns), "fast_turns": len(fast_turns)}


def probe() -> dict:
    """Import the diarization stack the way ``run`` does and report what breaks, if anything.

    Metadata says nothing about whether the compiled modules load: Windows App Control / Smart App
    Control can block unsigned Cython .pyd files (scipy, scikit-learn) in an otherwise healthy venv,
    and every transcription then silently comes back without speakers. ``tamlul doctor`` runs this
    in the child so torch never enters the CLI process."""
    out: dict = {"ok": False, "steps": []}
    t0 = time.time()
    try:
        import torch  # noqa: F401
        out["steps"].append("torch")
        import sklearn.cluster  # noqa: F401  (pyannote's clustering; needs scipy + sklearn .pyd files)
        out["steps"].append("sklearn.cluster")
        from pyannote.audio import Pipeline  # noqa: F401
        out["steps"].append("pyannote.audio.Pipeline")
        from .._vendor.pyannote_fast import FastSpeakerDiarization  # noqa: F401
        out["steps"].append("tamlul._vendor.pyannote_fast")
        out["ok"] = True
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe", action="store_true", help="import the stack, print a JSON report, exit 0/2")
    ap.add_argument("--run")
    ap.add_argument("--model")
    ap.add_argument("--audio", default=None, help="float32 16 kHz file (default <run>/audio.16k.f32)")
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument("--fast-embeddings", default="on", choices=["on", "off"])
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "6") or 6))
    ap.add_argument("--equality-check", action="store_true")
    ap.add_argument("--embeddings-out", default=None)
    a = ap.parse_args(argv)
    if a.probe:
        rep = probe()
        print(json.dumps(rep))
        return 0 if rep["ok"] else 2
    if not a.run or not a.model:
        ap.error("--run and --model are required")
    try:
        return run(a.run, a.model, num_speakers=a.num_speakers, min_speakers=a.min_speakers,
                   max_speakers=a.max_speakers, fast=(a.fast_embeddings == "on"), threads=a.threads,
                   equality_check=a.equality_check, embeddings_out=a.embeddings_out, audio_file=a.audio)
    except Exception as exc:  # noqa: BLE001
        _log(f"diarization failed: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
