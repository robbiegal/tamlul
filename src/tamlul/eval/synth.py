"""Synthetic tiers built from single-utterance sets (design §9, review eval-2).

Tier A (timestamps): clips concatenated with log-uniform 0.3-3 s gaps. Every
clip's true onset/offset is known by construction *up to the clip's own leading
and trailing silence*, so each clip is trimmed by two independent detectors
(energy threshold and Silero VAD); clips on which they disagree by more than
``max_disagree_ms`` are dropped and the residual disagreement (median, p95) is
published as the tier's precision floor - absolute accuracy is only claimed at
collars >= that floor.

Speaker tier: the same audio with the source utterance's speaker (the row's
``speaker`` column, or the utterance id when the set has one speaker per row,
e.g. eval-whatsapp voice notes) as the reference label -> non-circular DER /
attribution on consumer audio, with no natural overlap (stated in the manifest).

Optional noise beds: ``--noise-dir`` with any wav/mp3 files (MUSAN or the
user's own) mixed under a fraction of the clips at a fixed SNR.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from ..media import SR, write_wav
from ..paths import DataLayout
from ..plan import silero_regions
from .sets import Utt, decode_bytes, eval_root, utterances


def energy_bounds(x: np.ndarray, *, frame_ms: int = 10, above_floor_db: float = 20.0, hold: int = 3) -> tuple[float, float] | None:
    """First/last run of ``hold`` frames more than ``above_floor_db`` over the clip's noise floor
    (10th-percentile frame level), in seconds. Measured on FLEURS: this variant agrees with Silero's
    bias-corrected onset within 30 ms on ~60 % of clips; peak-relative thresholds fire on breath."""
    n = int(SR * frame_ms / 1000)
    if len(x) < n * hold:
        return None
    frames = x[: len(x) // n * n].reshape(-1, n)
    lvl = 20 * np.log10(np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) + 1e-9)
    on = lvl > np.percentile(lvl, 10) + above_floor_db
    first = next((i for i in range(len(on) - hold + 1) if on[i:i + hold].all()), None)
    if first is None:
        return None
    last = next((i + hold for i in range(len(on) - hold, -1, -1) if on[i:i + hold].all()), None)
    return first * n / SR, last * n / SR


def silero_bounds(x: np.ndarray) -> tuple[float, float] | None:
    regs = silero_regions(x, {"threshold": 0.5, "min_speech_duration_ms": 100, "max_speech_duration_s": 60.0,
                              "min_silence_duration_ms": 100, "speech_pad_ms": 0})
    if not regs:
        return None
    return float(regs[0][0]), float(regs[-1][1])


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def _load_noise(noise_dir: pathlib.Path | None) -> list[np.ndarray]:
    if not noise_dir or not noise_dir.is_dir():
        return []
    out = []
    for p in sorted(noise_dir.iterdir()):
        if p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"):
            try:
                out.append(decode_bytes(p.read_bytes()))
            except Exception:  # noqa: BLE001
                continue
    return [n for n in out if len(n) > SR]


def _mix(sig: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    if len(noise) < len(sig):
        noise = np.tile(noise, int(math.ceil(len(sig) / len(noise))))
    start = int(rng.integers(0, len(noise) - len(sig) + 1))
    n = noise[start: start + len(sig)]
    ps = float(np.mean(sig ** 2)) + 1e-9
    pn = float(np.mean(n ** 2)) + 1e-9
    gain = math.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    return np.clip(sig + gain * n, -0.99, 0.99).astype(np.float32)


def build(set_name: str, layout: DataLayout, *, name: str, minutes: float = 60.0, seed: int = 0,
          gap_s: tuple[float, float] = (0.3, 3.0), interleave_fraction: float = 0.2, noise_fraction: float = 0.2,
          noise_dir: pathlib.Path | None = None, snr_db: float = 10.0, max_disagree_ms: float = 30.0,
          speaker_from: str = "auto", log=print) -> pathlib.Path:
    """Write <data>/eval/synth/<name>/{audio.wav,reference.jsonl,manifest.json}."""
    rng = np.random.default_rng(seed)
    out = eval_root(layout) / "synth" / name
    out.mkdir(parents=True, exist_ok=True)
    noises = _load_noise(noise_dir)
    if noise_fraction > 0 and not noises:
        log("no noise files: noise beds disabled (pass --noise-dir with MUSAN or your own recordings)")
        noise_fraction = 0.0

    pieces: list[np.ndarray] = []
    refs: list[dict] = []
    dropped = 0
    disagreements: list[float] = []
    t = 0.0
    target = minutes * 60.0
    # pass 1: both detectors on every candidate. Silero's onset is systematically late relative to the
    # energy onset (it needs a few frames of evidence), so the two are compared after removing the
    # corpus-wide median offset; the residual is the random disagreement that defines the floor.
    cands: list[tuple[str, Utt, tuple[float, float], tuple[float, float]]] = []
    for u in utterances(set_name, layout):
        if len(u.audio) < SR * 0.5:
            continue
        spk = u.speaker if (speaker_from == "column" or (speaker_from == "auto" and u.speaker)) else u.id
        eb, sb = energy_bounds(u.audio), silero_bounds(u.audio)
        if eb is None or sb is None:
            dropped += 1
            continue
        cands.append((spk, u, eb, sb))
        if sum(c[2][1] - c[2][0] for c in cands) > target * 3:
            break
    if not cands:
        raise SystemExit("no usable clips (no detector fired?)")
    bias_on = float(np.median([sb[0] - eb[0] for _, _, eb, sb in cands])) * 1000
    bias_off = float(np.median([sb[1] - eb[1] for _, _, eb, sb in cands])) * 1000
    # pass 2: keep the clips on which the bias-corrected detectors agree; the energy onset is the boundary
    pool: list[tuple[str, dict, np.ndarray]] = []
    for spk, u, eb, sb in cands:
        d_on = abs((sb[0] * 1000 - bias_on) - eb[0] * 1000)
        d_off = abs((sb[1] * 1000 - bias_off) - eb[1] * 1000)
        disagreements.append(d_on)
        # the gate is on the onset only: offsets (trailing breath, decay) disagree far more between any two
        # detectors, so they get their own, wider floor instead of throwing the clip away
        if d_on > max_disagree_ms:
            dropped += 1
            continue
        on, off = eb
        clip = u.audio[int(on * SR): int(off * SR)]
        rms = float(np.sqrt(np.mean(clip.astype(np.float64) ** 2))) or 1e-6
        clip = np.clip(clip * (10 ** (-23 / 20) / rms), -0.99, 0.99).astype(np.float32)
        pool.append((spk, {"id": u.id, "text": u.text, "onset_disagree_ms": round(d_on, 1),
                           "offset_disagree_ms": round(d_off, 1)}, clip))
        if sum(len(c) for _, _, c in pool) / SR > target * 1.5:
            break
    if not pool:
        raise SystemExit("no usable clips (all dropped by the trim check?)")
    rng.shuffle(pool)   # type: ignore[arg-type]
    speakers = sorted({s for s, _, _ in pool})
    i = 0
    while t < target and i < len(pool):
        spk, meta, clip = pool[i]
        i += 1
        # interleave: with probability p, follow with a clip of a *different* speaker
        noisy = bool(noises) and rng.random() < noise_fraction
        if noisy:
            clip = _mix(clip, noises[int(rng.integers(0, len(noises)))], snr_db, rng)
        start = t
        pieces.append(clip)
        t += len(clip) / SR
        refs.append({**meta, "s": int(round(start * 1000)), "e": int(round(t * 1000)), "spk": spk, "noisy": noisy})
        gap = _log_uniform(rng, *gap_s)
        pieces.append(np.zeros(int(gap * SR), np.float32))
        t += gap
        if len(speakers) > 1 and rng.random() < interleave_fraction:
            j = next((k for k in range(i, len(pool)) if pool[k][0] != spk), None)
            if j is not None:
                pool[i], pool[j] = pool[j], pool[i]
    audio = np.concatenate(pieces)
    write_wav(out / "audio.wav", audio)
    with open(out / "reference.jsonl", "w", encoding="utf-8") as fh:
        for r in refs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    kept_on = np.asarray([r["onset_disagree_ms"] for r in refs] or [0.0])
    kept_off = np.asarray([r["offset_disagree_ms"] for r in refs] or [0.0])
    dis = kept_on
    all_dis = np.asarray(disagreements) if disagreements else np.zeros(1)
    manifest = {"name": name, "source_set": set_name, "seed": seed, "clips": len(refs), "dropped": dropped,
                "duration_s": round(len(audio) / SR, 1), "speakers": len({r["spk"] for r in refs}),
                "gap_s": list(gap_s), "interleave_fraction": interleave_fraction,
                "noise_fraction": noise_fraction if noises else 0.0, "snr_db": snr_db,
                "trim": {"detectors": ["energy: 3 x 10 ms frames > noise floor + 20 dB", "silero (faster_whisper.vad), pad 0; bias-corrected"],
                         "max_disagree_ms": max_disagree_ms,
                         "silero_minus_energy_bias_ms": {"onset": round(bias_on, 1), "offset": round(bias_off, 1)},
                         "candidates": len(cands),
                         "disagree_all_candidates_ms": {"median": round(float(np.median(all_dis)), 1),
                                                        "p95": round(float(np.percentile(all_dis, 95)), 1)},
                         "residual_onset_kept_ms": {"median": round(float(np.median(dis)), 1),
                                                    "p95": round(float(np.percentile(dis, 95)), 1)},
                         "residual_offset_kept_ms": {"median": round(float(np.median(kept_off)), 1),
                                                     "p95": round(float(np.percentile(kept_off, 95)), 1)},
                         "precision_floor_ms": round(float(np.percentile(dis, 95)), 1),
                         "offset_floor_ms": round(float(np.percentile(kept_off, 95)), 1)},
                "overlap": "none by construction (clips never overlap)",
                "note": "reference boundaries are the energy detector's after the Silero agreement gate on onsets; "
                        "onset accuracy is only claimed at collars >= precision_floor_ms, offsets at >= offset_floor_ms"}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"{name}: {len(refs)} clips, {manifest['duration_s']} s, {manifest['speakers']} speakers, "
        f"dropped {dropped}, precision floor {manifest['trim']['precision_floor_ms']} ms -> {out}")
    return out


def load_reference(synth_dir: pathlib.Path) -> tuple[list[dict], dict]:
    refs = [json.loads(ln) for ln in (synth_dir / "reference.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    manifest = json.loads((synth_dir / "manifest.json").read_text(encoding="utf-8"))
    return refs, manifest
