"""Per-machine calibration (design §4.14): speed, engine disagreement and silence behaviour.

NEVER accuracy. The int8-vs-float32 accuracy decision was made on the eval
sets before release (§1.3); here the reference-free disagreement proxy only
catches a broken int8 kernel path on this machine (warn >= 1.5 %, switch to
float32 above 3 % - the research thresholds).

``--quick`` (~3 min): 90-s slice, one thread count, int8 vs float32 proxy,
one silence case with the gate on and off, pyannote fast-vs-stock equality on
60 s. ``--full``: the whole 5-min fixture, thread sweep {6, 8, 12, 16} x beam
{1, 5}, pyannote at {6, 12} threads, all silence cases.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
import time
import wave
from importlib import resources
from typing import Any

import numpy as np

from . import __version__, registry
from .asr import DecodeOpts, get_backend
from .asr.base import ModelRef
from .config import resolve
from .doctor import _pkg_versions, physical_cores
from .media import SR
from .paths import DataLayout
from .plan import build_plan, decode_clips
from .text.normalize import ivrit_lb, wer

PROXY_WARN = 0.015
PROXY_SWITCH = 0.03
PROXY_MIN_WORDS = 400   # below this a 3 % difference is a handful of words: warn, never switch (run --full)
FAST_EMB_MIN_GAIN = 1.0   # keep fast embeddings if identical output (speed is a bonus, never a condition)


def _load_fixture() -> tuple[np.ndarray, list[dict]]:
    fx = resources.files("tamlul").joinpath("fixtures")
    with fx.joinpath("fleurs_he.wav").open("rb") as fh, wave.open(fh) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    with fx.joinpath("fleurs_he.jsonl").open("r", encoding="utf-8") as fh:
        refs = [json.loads(line) for line in fh if line.strip()]
    return audio, refs


def _decode(backend, audio: np.ndarray, opts: DecodeOpts) -> tuple[str, float]:
    plan = build_plan(audio)
    t0 = time.time()
    words: list[str] = []
    for g in plan.groups:
        sl = audio[int(g.start_s * SR): int(g.end_s * SR)]
        for seg in backend.transcribe(sl, decode_clips(g), opts):
            words += [w.t.strip() for w in seg.words if w.t.strip()]
    return " ".join(words), time.time() - t0


def _asr_point(model: ModelRef, audio: np.ndarray, *, compute_type: str, threads: int, beam: int, log) -> dict:
    be = get_backend("ct2")
    t0 = time.time()
    be.load(model, "cpu", threads, compute_type)
    load_s = time.time() - t0
    opts = DecodeOpts(beam_size=beam, best_of=beam, temperature=(0.0,))   # deterministic for the proxy
    text, dt = _decode(be, audio, opts)
    be.unload()
    rtf = (len(audio) / SR) / dt if dt else 0.0
    log(f"  ct2 {compute_type:8} threads={threads:2} beam={beam}: {len(audio)/SR:5.1f}s audio in {dt:5.1f}s -> x{rtf:.2f} RT (load {load_s:.1f}s)")
    return {"compute_type": compute_type, "threads": threads, "beam": beam, "audio_s": round(len(audio) / SR, 1),
            "wall_s": round(dt, 1), "rtf": round(rtf, 2), "load_s": round(load_s, 1), "text": text}


def _silence_cases(full: bool) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    zeros = np.zeros(SR * 30, np.float32)
    cases = {"zeros_30s": zeros}
    if full:
        pink = np.cumsum(rng.standard_normal(SR * 30)).astype(np.float32)
        pink = (pink - pink.mean()) / (np.abs(pink).max() + 1e-9) * 10 ** (-30 / 20)
        room = (rng.standard_normal(SR * 30) * 10 ** (-55 / 20)).astype(np.float32)
        cases["pink_noise_30s"] = pink.astype(np.float32)
        cases["room_tone_30s"] = room
    return cases


def _silence_test(model: ModelRef, audio_fixture: np.ndarray, threads: int, compute_type: str, full: bool, log) -> dict:
    be = get_backend("ct2")
    be.load(model, "cpu", threads, compute_type)
    opts = DecodeOpts(temperature=(0.0,))
    out = {}
    cases = _silence_cases(full)
    # a real clip with 30 s of digital silence spliced in the middle (from the bundled fixture)
    clip = audio_fixture[: SR * 12]
    cases["clip_with_30s_silence"] = np.concatenate([clip[: SR * 6], np.zeros(SR * 30, np.float32), clip[SR * 6:]])
    for name, sig in cases.items():
        plan = build_plan(sig)
        gated_words = 0
        for g in plan.groups:
            sl = sig[int(g.start_s * SR): int(g.end_s * SR)]
            gated_words += sum(len(s.words) for s in be.transcribe(sl, decode_clips(g), opts))
        raw_segments = be.transcribe(sig, None, opts)
        raw_words = sum(len(s.words) for s in raw_segments)
        raw_text = " ".join(s.text.strip() for s in raw_segments)[:120]
        out[name] = {"through_pipeline_words": gated_words, "model_only_words": raw_words,
                     "model_only_sample": raw_text, "plan_groups": len(plan.groups)}
        log(f"  silence {name:22}: pipeline words={gated_words} (groups {len(plan.groups)}), model-only words={raw_words}")
    be.unload()
    return out


def _pyannote_equality(layout: DataLayout, audio: np.ndarray, threads: int, log) -> dict | None:
    art = registry.artefacts()["diarize/pyannote-community-1"]
    if not registry.is_installed(art, layout):
        log("  pyannote model not installed; skipping the equality check")
        return None
    with tempfile.TemporaryDirectory(prefix="tamlul_cal_") as tmp:
        audio_path = pathlib.Path(tmp) / "audio.16k.f32"
        audio.astype("<f4").tofile(audio_path)
        cmd = [sys.executable, "-m", "tamlul.diarize.child", "--run", tmp, "--model", str(registry.local_dir(art, layout)),
               "--fast-embeddings", "on", "--equality-check", "--threads", str(threads)]
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(threads)
        t0 = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        wall = time.time() - t0
        if proc.returncode != 0:
            log("  pyannote equality check failed to run: " + (proc.stderr or "")[-300:])
            return {"ran": False}
        data = json.loads((pathlib.Path(tmp) / "diarization.json").read_text(encoding="utf-8"))
    info = data["info"]
    res = {"ran": True, "identical_rttm": info.get("equality", {}).get("identical_rttm"), "fast_apply_s": info.get("apply_s"),
           "fast_rtf": info.get("rtf"), "speakers": len(data["labels"]), "wall_s": round(wall, 1), "threads": threads}
    log(f"  pyannote fast vs stock on {len(audio)/SR:.0f}s: identical_rttm={res['identical_rttm']}, fast x{res['fast_rtf']} RT, {res['speakers']} speakers")
    return res


def calibrate(layout: DataLayout, *, full: bool, settings, log=print) -> dict:
    audio, refs = _load_fixture()
    pdef = settings.profile_defaults() if settings.profile != "auto" else None
    revision = (pdef or {"revision": "2025.05.13"})["revision"]
    aid = registry.asr_id_for(revision, quality_max=(settings.quality == "max"))
    art = registry.artefacts()[aid]
    if not registry.is_installed(art, layout):
        raise RuntimeError(f"model {aid} is not installed; run `tamlul models install --yes` first")
    model = ModelRef(id=aid, dir=str(registry.local_dir(art, layout)), revision=art.revision, commit=art.commit,
                     sha256=art.files.get(art.primary, {}).get("sha256", ""))
    shipped = (art.compute_type or {}).get("decision") or "int8"
    phys = physical_cores()
    slice_ = audio if full else audio[: SR * 90]
    log(f"calibration {'full' if full else 'quick'}: {aid}, {len(slice_)/SR:.0f}s of the fixture, physical cores ~{phys}")

    measurements: dict[str, Any] = {"asr": [], "silence": {}, "pyannote": None}
    # ---- speed (+ the int8 output for the proxy)
    thread_list = [6, 8, 12, 16] if full else [phys]
    beams = [1, 5] if full else [5]
    best = None
    for th in thread_list:
        for b in beams:
            pt = _asr_point(model, slice_, compute_type=shipped, threads=th, beam=b, log=log)
            measurements["asr"].append({k: v for k, v in pt.items() if k != "text"})
            if b == 5 and (best is None or pt["rtf"] > best["rtf"]):
                best = pt
    # ---- int8 vs float32 disagreement proxy at the chosen thread count
    other = "float32" if shipped == "int8" else "int8"
    pt2 = _asr_point(model, slice_, compute_type=other, threads=best["threads"], beam=5, log=log)
    measurements["asr"].append({k: v for k, v in pt2.items() if k != "text"})
    a, b_ = ivrit_lb(best["text"]), ivrit_lb(pt2["text"])
    d = wer([b_ if shipped == "int8" else a], [a if shipped == "int8" else b_])   # int8 against float32
    proxy = {"int8_vs_float32_disagreement": round(d["wer"], 4), "words": d["N"], "S": d["S"], "D": d["D"], "I": d["I"],
             "threshold_warn": PROXY_WARN, "threshold_switch": PROXY_SWITCH}
    measurements["proxy"] = proxy
    log(f"  proxy: int8 vs float32 disagreement {d['wer']:.2%} over {d['N']} words (warn >= {PROXY_WARN:.1%}, switch > {PROXY_SWITCH:.1%})")
    # reference WER on the slice, reported for information only (read speech; NOT an accuracy decision)
    ref_text = " ".join(r["text"] for r in refs if r["end_ms"] <= len(slice_) / SR * 1000 + 500)
    measurements["fixture_wer_info_only"] = {
        "ivrit-lb_" + shipped: round(wer([ivrit_lb(ref_text)], [a])["wer"], 4),
        "ivrit-lb_" + other: round(wer([ivrit_lb(ref_text)], [b_])["wer"], 4),
        "note": "read speech, information only; accuracy is decided on the eval sets (design §1.3)"}
    log(f"  fixture WER (info only): {shipped} {measurements['fixture_wer_info_only']['ivrit-lb_' + shipped]:.3f}, "
        f"{other} {measurements['fixture_wer_info_only']['ivrit-lb_' + other]:.3f}")
    # ---- silence
    measurements["silence"] = _silence_test(model, audio, best["threads"], shipped, full, log)
    # ---- pyannote equality
    measurements["pyannote"] = _pyannote_equality(layout, audio[: SR * (120 if full else 60)], 6, log)

    # ---- choices
    compute = shipped
    reason = "shipped decision"
    dis = proxy["int8_vs_float32_disagreement"]
    if dis > PROXY_SWITCH and proxy["words"] >= PROXY_MIN_WORDS:
        compute = "float32"
        reason = f"int8 kernel disagreement {dis:.2%} > {PROXY_SWITCH:.0%} over {proxy['words']} words on this machine"
    elif dis > PROXY_SWITCH:
        reason = (f"kept {shipped}; disagreement {dis:.2%} but only {proxy['words']} words ({proxy['S']+proxy['D']+proxy['I']} differ) "
                  f"- inconclusive, run `calibrate --full` to decide on {PROXY_MIN_WORDS}+ words")
    elif dis >= PROXY_WARN:
        reason = f"kept {shipped}; disagreement {dis:.2%} is above the warn level"
    pe = measurements["pyannote"] or {}
    fast_ok = bool(pe.get("identical_rttm"))
    # the gate is judged on pure-silence cases only; the spliced clip legitimately carries speech words
    silence_ok = all(v["through_pipeline_words"] == 0 for k, v in measurements["silence"].items()
                     if k != "clip_with_30s_silence")
    choices = {"compute_type": compute, "compute_type_reason": reason, "threads": best["threads"],
               "beam": 5, "concurrent_diarization": False,
               "concurrent_reason": "v1 runs diarization after ASR in a child process; not measured",
               "fast_embeddings": fast_ok, "fast_embeddings_reason": "identical RTTM vs stock" if fast_ok else "not identical or not checked -> stock",
               "silence_gate_ok": silence_ok, "expected_rtf_beam5": best["rtf"]}
    cal = {"id": time.strftime("%Y%m%d-%H%M%S"), "mode": "full" if full else "quick", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "plugin_version": __version__, "packages": _pkg_versions(), "cpu": platform.processor(),
           "platform": platform.platform(), "logical_cpus": os.cpu_count(), "physical_cores_estimate": phys,
           "model": aid, "fixture_s": round(len(slice_) / SR, 1), "measurements": measurements, "choices": choices}
    layout.calibration.write_text(json.dumps(cal, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"calibration written: {layout.calibration}")
    log(f"choices: compute_type={compute} ({reason}); threads={best['threads']}; fast_embeddings={fast_ok}; "
        f"expected x{best['rtf']} RT at beam 5; silence gate ok={silence_ok}")
    return cal


def calibrate_command(args: argparse.Namespace, layout: DataLayout, cli_settings: dict) -> int:
    settings = resolve(cli_settings, layout)
    full = bool(args.full) and not args.quick
    try:
        calibrate(layout, full=full, settings=settings, log=lambda m: print(m, flush=True))
    except RuntimeError as exc:
        print(f"tamlul calibrate: {exc}", file=sys.stderr)
        return 4
    return 0
