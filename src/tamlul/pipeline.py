"""The transcription pipeline (design §3.1): probe -> extract -> plan -> asr ->
clean -> diarize (child process) -> attribute -> text -> annotate -> write -> export.

Every stage checkpoints under the run directory; ``--resume`` continues from
the last complete plan group and refuses to mix models/options/plans.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import sys
import time
from importlib import metadata
from typing import Any

import numpy as np

from . import __version__, registry
from .asr import DecodeOpts, get_backend
from .asr.base import ModelRef
from .asr.recover import recover_missed
from .config import Settings, resolve, resolve_auto_profile
from .document import add_raw_segments, apply_clean_flags, new_document
from .jobs import RunDir, file_sha256, resume_hash
from .media import SR, extract, load_f32
from .paths import DataLayout
from .plan import (
    SNAP_LATE_MS,
    SNAP_MAX_MS,
    SNAP_MIN_GAP_MS,
    SNAP_TOL_MS,
    Plan,
    build_plan,
    build_stereo_plan,
    decode_clips,
    load_plan,
    save_plan,
    snap_words,
)

MEDIA_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".amr", ".wma", ".webm",
                  ".mp4", ".mkv", ".mov", ".avi", ".3gp", ".mpeg", ".mpg", ".wmv"}
EXIT_OK, EXIT_ERROR, EXIT_NOT_READY = 0, 2, 4


class NotReady(Exception):
    pass


def _log(rd: RunDir, msg: str) -> None:
    rd.log_line(msg)
    print(msg, file=sys.stderr, flush=True)


def _lib_versions() -> dict[str, str | None]:
    out = {}
    for n in ("faster-whisper", "ctranslate2", "onnxruntime", "pyannote.audio", "torch", "numpy"):
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def _calibration(layout: DataLayout) -> dict:
    from .doctor import calibration_status
    st = calibration_status(layout)
    if not st.get("present") or st.get("stale"):
        return {}
    try:
        return json.loads(layout.calibration.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def choose_runtime(settings: Settings, layout: DataLayout, artefact: registry.Artefact) -> dict:
    """threads / compute_type / concurrency from flags > calibration > shipped decision > hard defaults."""
    from .doctor import physical_cores
    cal = _calibration(layout)
    choices = cal.get("choices", {})
    compute = settings.compute_type
    if compute == "auto":
        compute = choices.get("compute_type") or (artefact.compute_type or {}).get("decision") or "int8"
    threads = int(settings.threads) if settings.threads else int(choices.get("threads") or physical_cores())
    device = "cpu"
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            device = "cuda"
            compute = "float16"
    except Exception:  # noqa: BLE001
        pass
    return {"device": device, "compute_type": compute, "threads": threads,
            "concurrent_diarization": bool(choices.get("concurrent_diarization", False)),
            "fast_embeddings": choices.get("fast_embeddings"), "calibration_id": cal.get("id")}


def transcribe_command(args: argparse.Namespace, layout: DataLayout, cli_settings: dict[str, Any]) -> int:
    settings = resolve(cli_settings, layout)
    if settings.models_dir:
        os.environ["TAMLUL_MODELS_DIR"] = settings.models_dir
    layout.ensure()
    if args.resume:
        rd = RunDir.open(layout, args.resume)
        return run_one(rd, pathlib.Path(rd.read_run()["media"]["path"]), settings, layout, args, resume=True)
    if not args.media:
        print("tamlul: transcribe needs a media file or folder (or --resume <run id>)", file=sys.stderr)
        return EXIT_ERROR
    src = pathlib.Path(args.media).expanduser().resolve()
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in MEDIA_SUFFIXES)
        if not files:
            print(f"tamlul: no media files in {src}", file=sys.stderr)
            return EXIT_ERROR
        rc = EXIT_OK
        summary = []
        for f in files:
            rd = RunDir.create(layout, f, None)
            r = run_one(rd, f, settings, layout, args, resume=False)
            summary.append({"media": str(f), "run": rd.id, "rc": r})
            (layout.runs / "batch-latest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            rc = max(rc, r)
        return rc
    if not src.is_file():
        print(f"tamlul: no such file {src}", file=sys.stderr)
        return EXIT_ERROR
    rd = RunDir.create(layout, src, args.out)
    return run_one(rd, src, settings, layout, args, resume=False)


def run_one(rd: RunDir, src: pathlib.Path, settings: Settings, layout: DataLayout, args: argparse.Namespace,
            *, resume: bool) -> int:
    print(f"run: {rd.id}", flush=True)
    print(f"dir: {rd.path}", flush=True)
    t0 = time.time()
    try:
        return _run(rd, src, settings, layout, args, resume=resume, t0=t0)
    except NotReady as exc:
        _log(rd, f"not ready: {exc}")
        rd.set_progress("error", 0, error=str(exc))
        return EXIT_NOT_READY
    except Exception as exc:
        _log(rd, f"error: {type(exc).__name__}: {exc}")
        rd.set_progress("error", 0, error=f"{type(exc).__name__}: {exc}")
        raise


def _run(rd: RunDir, src: pathlib.Path, settings: Settings, layout: DataLayout, args: argparse.Namespace,
         *, resume: bool, t0: float) -> int:
    run = rd.read_run() if resume else {}

    # ---- 1-2. probe + extract (skipped on resume when the audio exists)
    if resume and rd.audio.is_file() and run.get("media"):
        media = run["media"]
        audio = load_f32(rd.audio)
        per_channel = [load_f32(rd.path / f"audio.ch{i}.16k.f32") for i in range(2)
                       if (rd.path / f"audio.ch{i}.16k.f32").is_file()]
        _log(rd, "resume: reusing decoded audio")
    else:
        rd.set_progress("probe", 1)
        sha = file_sha256(src)
        info, audio, per_channel = extract(src, rd.path, channels_mode=settings.channels, sha256=sha)
        media = info.as_dict()
        rd.media_json.write_text(json.dumps(media, ensure_ascii=False, indent=2), encoding="utf-8")
        _log(rd, f"decoded {media['duration_ms']/1000:.1f}s via {media['decoder']}, channels={media['channels']}, "
                 f"true_stereo={media['true_stereo']}, gain={media['gain_db']} dB")
    duration_s = len(audio) / SR

    # ---- profile
    profile = settings.profile
    if profile == "auto":
        profile = resolve_auto_profile(duration_s, media.get("channels"), src.suffix)
        _log(rd, f"profile auto -> {profile}")
    settings.profile = profile
    pdef = settings.profile_defaults()

    # ---- model
    quality_max = settings.quality == "max"
    aid = registry.asr_id_for(pdef["revision"], quality_max=quality_max)
    art = registry.artefacts()[aid]
    fake = bool(os.environ.get("TAMLUL_FAKE_ASR"))
    if fake:
        _log(rd, "TAMLUL_FAKE_ASR is set: using the deterministic fake backend (tests/CI only)")
    if not fake and not registry.is_installed(art, layout):
        if getattr(args, "yes", False):
            _log(rd, f"model {aid} missing; downloading ({art.size_bytes/1e9:.2f} GB) because --yes was given")
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            if registry.install([aid], layout, yes=True, log=lambda m: _log(rd, m)) != 0:
                raise NotReady(f"model {aid} could not be installed")
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            raise NotReady(f"model {aid} is not installed ({art.size_bytes/1e9:.2f} GB): run `tamlul models install "
                           f"--profile {profile} --yes` or pass --yes")
    model = ModelRef(id=aid, dir=str(registry.local_dir(art, layout)), revision=art.revision, commit=art.commit,
                     sha256=art.files.get(art.primary, {}).get("sha256", ""))
    rt = choose_runtime(settings, layout, art)
    opts = DecodeOpts(batched=bool(settings.fast))
    bad = opts.validate()
    if bad:
        raise RuntimeError("decode contract violated: " + "; ".join(bad))

    # ---- 3. plan
    if resume and rd.chunks_json.is_file():
        plan, plan_sha = load_plan(rd.chunks_json)
        _log(rd, f"resume: reusing plan ({len(plan.groups)} groups)")
    else:
        rd.set_progress("plan", 3)
        plan = build_stereo_plan(per_channel) if per_channel else build_plan(audio)
        plan_sha = save_plan(plan, rd.chunks_json)
        _log(rd, f"plan: {len(plan.regions)} speech regions, {plan.speech_s:.1f}s speech, {len(plan.groups)} groups, "
                 f"{len(plan.gated)} gated, {plan.parts} part(s)")

    audio_sha = _sha_of_array(audio)
    rh = resume_hash(model.sha256, model.revision, rt["compute_type"], opts.as_dict(), plan_sha, audio_sha)
    if resume:
        if run.get("resume_hash") != rh:
            raise RuntimeError("refusing to resume: model, options, plan or audio differ from run.json "
                               f"({run.get('resume_hash')} != {rh}); start a new run")
    prev_recovered = (run.get("asr") or {}).get("recovered_segments", 0) if resume else 0
    run = {
        "id": rd.id, "media": media, "profile": profile, "settings": settings.as_dict(),
        "settings_sources": settings.sources,
        "asr": {"engine": "fake" if fake else "ct2", "model": model.id, "dir": model.dir, "revision": model.revision,
                "commit": model.commit, "model_sha256": model.sha256, **rt},
        "decode": opts.as_dict(), "plan": {"source": plan.source, "sha": plan_sha, "params": plan.params,
                                           "groups": len(plan.groups), "parts": plan.parts},
        "audio_sha": audio_sha, "resume_hash": rh, "plugin_version": __version__,
        "libraries": _lib_versions(), "platform": platform.platform(), "python": platform.python_version(),
        "started": run.get("started") or time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    rd.write_run(run)

    # ---- 4. asr per group, resumable
    done = rd.groups_done() if resume else set()
    backend = get_backend("fake" if fake else "ct2")
    rd.set_progress("asr", 5, groups_done=len(done), groups_total=len(plan.groups))
    t_load = time.time()
    binfo = backend.load(model, rt["device"], rt["threads"], rt["compute_type"])
    _log(rd, f"asr: {binfo['engine']} {model.id} {rt['compute_type']} device={rt['device']} threads={rt['threads']} "
             f"loaded in {time.time()-t_load:.1f}s; backend line: {_ct2_backend_line()}")
    processed_s = 0.0
    recovered = 0
    t_asr = time.time()
    for g in plan.groups:
        if g.index in done:
            processed_s += g.duration_s
            continue
        i0, i1 = int(g.start_s * SR), int(g.end_s * SR)
        source = per_channel[g.channel] if g.channel >= 0 and per_channel else audio
        slice_ = source[i0:i1]
        raw = [(seg, g.start_s) for seg in backend.transcribe(slice_, decode_clips(g), opts)]
        # speech the decoder skipped (a window tail, typically): decode it on its own and splice it in
        extra = recover_missed(backend, source, g, raw, plan.raw_for(g.channel), opts, log=lambda m: _log(rd, m))
        if extra:
            recovered += len(extra)
            raw = sorted(raw + extra, key=lambda x: x[0].s + x[1])
        # offset back to file time and validate monotonicity across the boundary
        payload = []
        for seg, off in raw:
            payload.append({"s": round(seg.s + off, 3), "e": round(seg.e + off, 3), "text": seg.text,
                            "channel": g.channel,
                            "words": [{"s": round(w.s + off, 3), "e": round(w.e + off, 3), "t": w.t, "p": w.p}
                                      for w in seg.words],
                            "avg_logprob": seg.avg_logprob, "no_speech_prob": seg.no_speech_prob,
                            "compression_ratio": seg.compression_ratio, "temperature": seg.temperature})
        rd.append_segments(g.index, payload)
        processed_s += g.duration_s
        done.add(g.index)
        elapsed = time.time() - t_asr
        rtf = processed_s / elapsed if elapsed > 0 else 0.0
        remaining = sum(x.duration_s for x in plan.groups if x.index not in done)
        pct = 5 + 55 * (len(done) / max(1, len(plan.groups)))
        rd.set_progress("asr", pct, groups_done=len(done), groups_total=len(plan.groups), rtf=round(rtf, 2),
                        eta_s=int(remaining / rtf) if rtf else None)
    backend.unload()
    run["asr"]["recovered_segments"] = prev_recovered + recovered   # a resume only decodes the groups left
    rd.write_run(run)
    _log(rd, f"asr done: {processed_s:.1f}s of audio in {time.time()-t_asr:.1f}s; {recovered} skipped segment(s) recovered")

    # ---- 5. build the document from segments.jsonl (the checkpoint is the source of truth)
    doc = new_document(media, {"asr": run["asr"], "decode": run["decode"], "plan": run["plan"],
                               "calibration_id": rt.get("calibration_id"), "plugin_version": __version__,
                               "libraries": run["libraries"], "machine": run["platform"]}, profile)
    _fill_from_checkpoint(doc, rd, plan)
    # boundary snap (design §9 aligner b): Whisper puts a clip's first word at the clip start, i.e. inside
    # the 300 ms VAD pad; measured on the synthetic tier as a -200 ms onset mode before this step
    moved = snap_words(doc["words"], plan)
    doc["pipeline"]["align"] = {"method": "dtw+snap", "snapped_boundaries": moved, "snap_max_ms": SNAP_MAX_MS,
                                "snap_tol_ms": SNAP_TOL_MS, "snap_min_gap_ms": SNAP_MIN_GAP_MS,
                                "snap_late_ms": SNAP_LATE_MS, "raw_regions": len(plan.raw_regions)}
    _log(rd, f"align: {moved} word boundaries snapped to unpadded speech edges")
    apply_clean_flags(doc)

    # ---- 6-7. diarization + attribution (child process)
    diar_speech_ms = None
    if per_channel:
        from .diarize.attribution import stereo_attribute
        stereo_attribute(doc, [w.get("channel", 0) for w in doc["words"]])
        doc["pipeline"]["diarization"] = {"engine": "stereo", "reason": "true-stereo recording: one speaker per channel"}
        _log(rd, "diarization: true stereo -> channel assignment (exact)")
    elif settings.diarization_enabled():
        rd.set_progress("diarize", 62)
        from .diarize import run_diarization
        result = run_diarization(rd, layout, settings, audio_duration_s=duration_s,
                                 fast_embeddings=rt.get("fast_embeddings"), log=lambda m: _log(rd, m))
        if result is not None:
            from .diarize.attribution import attribute
            attribute(doc, result)
            id_map = doc["pipeline"].get("attribution", {}).get("speaker_map", {})
            diar_speech_ms = {id_map.get(k, k): v for k, v in result.speech_ms_per_speaker().items()}
            doc["pipeline"]["diarization"] = result.info
        else:
            doc["pipeline"]["diarization"] = {"engine": None, "reason": "unavailable; no speaker layer (invariant 10)"}
    else:
        doc["pipeline"]["diarization"] = {"engine": None, "reason": "off for this profile/flags"}

    _write_samples(rd, doc, audio)

    # ---- 8. text layers (ITN, glossary)
    rd.set_progress("text", 85)
    from .text.layers import apply_text_layers
    apply_text_layers(doc, settings, run_dir=rd.path)

    # ---- 9. deterministic annotation
    rd.set_progress("annotate", 90)
    from .annotate.deterministic import annotate_deterministic
    annotate_deterministic(doc, profile=profile, plan=plan.to_json(), diar_speech_ms=diar_speech_ms)

    # ---- 10. write + export
    rd.set_progress("write", 95)
    from .schema import validate_document
    errors = validate_document(doc)
    if errors:
        raise RuntimeError("transcript.json failed validation (invariant 16): " + "; ".join(errors[:5]))
    rd.write_transcript(doc)
    from .speakers import write_overlay
    write_overlay(rd, doc)
    from .export import write_exports
    formats = [f.strip() for f in (getattr(args, "formats", None) or "md,txt,srt,vtt,csv,html,rttm,eaf,textgrid").split(",")]
    written = write_exports(rd, doc, formats, log=lambda m: _log(rd, m))
    run["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    run["wall_s"] = round(time.time() - t0, 1)
    rd.write_run(run)
    rd.set_progress("done", 100, wall_s=run["wall_s"], transcript=str(rd.transcript_json), exports=written)
    _log(rd, f"done in {run['wall_s']}s -> {rd.transcript_json}")
    print(json.dumps({"run": rd.id, "dir": str(rd.path), "transcript": str(rd.transcript_json), "exports": written,
                      "duration_s": round(duration_s, 1), "wall_s": run["wall_s"], "speakers": len(doc["speakers"]),
                      "words": len(doc["words"])}, ensure_ascii=False))
    return EXIT_OK


def _fill_from_checkpoint(doc: dict, rd: RunDir, plan: Plan) -> None:
    from .asr.base import RawSegment, RawWord
    rows = rd.read_segments()
    rows.sort(key=lambda r: (r["group"], r["s"]))
    for r in rows:
        seg = RawSegment(s=r["s"], e=r["e"], text=r["text"],
                         words=[RawWord(s=w["s"], e=w["e"], t=w["t"], p=w["p"]) for w in r["words"]],
                         avg_logprob=r.get("avg_logprob"), no_speech_prob=r.get("no_speech_prob"),
                         compression_ratio=r.get("compression_ratio"), temperature=r.get("temperature"))
        first = len(doc["words"])
        add_raw_segments(doc, [seg], offset_s=0.0, group=r["group"], speech_regions=plan.regions)
        ch = r.get("channel", -1)
        if ch is not None and ch >= 0:
            doc["segments"][-1]["channel"] = ch
            for w in doc["words"][first:]:
                w["channel"] = ch
    if any((r.get("channel") or -1) >= 0 for r in rows):
        # true stereo: interleave the two channels' words by time so indices are chronological
        doc["words"].sort(key=lambda w: (w["s"], w.get("channel", 0)))
        for k, w in enumerate(doc["words"]):
            w["i"] = k
        _rebuild_segment_ranges(doc)
    # monotonicity across group boundaries: a word may not start before the previous one
    prev_e = 0
    for w in doc["words"]:
        if w["s"] < prev_e - 50:
            w["s"] = prev_e
        w["e"] = max(w["e"], w["s"])
        prev_e = w["e"]


def _write_samples(rd: RunDir, doc: dict, audio: np.ndarray, seconds: float = 5.0) -> None:
    """A 5-s clip per speaker from their longest confident stretch, so the user can listen (design §4.10)."""
    if not doc["speakers"]:
        return
    from .media import write_wav
    (rd.path / "samples").mkdir(exist_ok=True)
    words = doc["words"]
    for sp in doc["speakers"]:
        runs_: list[tuple[int, int]] = []
        start = None
        for w in words:
            if w["spk"] == sp["id"] and not w.get("overlap"):
                if start is None:
                    start = w["s"]
                end = w["e"]
            elif start is not None:
                runs_.append((start, end))
                start = None
        if start is not None:
            runs_.append((start, end))
        if not runs_:
            continue
        s_ms, e_ms = max(runs_, key=lambda r: r[1] - r[0])
        e_ms = min(e_ms, s_ms + int(seconds * 1000))
        clip = audio[int(s_ms / 1000 * SR): int(e_ms / 1000 * SR)]
        if clip.size < SR // 2:
            continue
        out = rd.path / "samples" / f"{sp['id']}_1.wav"
        write_wav(out, clip)
        sp["sample"] = str(out)
        sp["sample_span"] = [s_ms, e_ms]


def _rebuild_segment_ranges(doc: dict) -> None:
    """After re-ordering words (stereo), give each segment the index range of its own words by time+channel."""
    words = doc["words"]
    for seg in doc["segments"]:
        ch = seg.get("channel", -1)
        ids = [w["i"] for w in words if seg["s"] <= w["s"] < max(seg["e"], seg["s"] + 1) and w.get("channel", -1) == ch]
        seg["w"] = [min(ids), max(ids)] if ids else None


def _sha_of_array(a: np.ndarray) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(a, dtype="<f4").tobytes()[: 64 * 1024 * 1024])
    h.update(str(len(a)).encode())
    return h.hexdigest()[:24]


def _ct2_backend_line() -> str:
    try:
        import ctranslate2
        return f"ctranslate2 {ctranslate2.__version__}, cuda devices {ctranslate2.get_cuda_device_count()}"
    except Exception:  # noqa: BLE001
        return "n/a"
