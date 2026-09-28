"""Speaker diarization (design §4.4): pyannote community-1 in a CHILD PROCESS.

The child (``tamlul-diarize`` / ``python -m tamlul.diarize.child``) owns torch
and its OpenMP runtime; this module only launches it, waits, and reads the
RTTM/JSON it writes atomically. If pyannote is not installed or the model is
missing, the run gets NO speaker layer and says so (invariant 10).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

from ..config import Settings
from ..jobs import RunDir
from ..paths import DataLayout


@dataclass
class DiarResult:
    turns: list[dict]                 # regular (overlap-aware): {s, e, spk}
    exclusive: list[dict]             # one speaker per frame: {s, e, spk}
    labels: list[str]
    info: dict
    embeddings: dict[str, list[float]] = field(default_factory=dict)

    def speech_ms_per_speaker(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self.turns:
            out[t["spk"]] = out.get(t["spk"], 0) + (t["e"] - t["s"])
        return out


def run_diarization(rd: RunDir, layout: DataLayout, settings: Settings, *, audio_duration_s: float,
                    fast_embeddings: str | bool | None, log: Callable[[str], None]) -> DiarResult | None:
    from .. import registry
    art = registry.artefacts()["diarize/pyannote-community-1"]
    if not registry.is_installed(art, layout):
        log("diarization: pyannote community-1 model not installed -> no speaker layer")
        return None
    cmd = [sys.executable, "-m", "tamlul.diarize.child", "--run", str(rd.path), "--model", str(registry.local_dir(art, layout))]
    if settings.speakers:
        cmd += ["--num-speakers", str(settings.speakers)]
    if settings.min_speakers:
        cmd += ["--min-speakers", str(settings.min_speakers)]
    if settings.max_speakers:
        cmd += ["--max-speakers", str(settings.max_speakers)]
    fe = settings.fast_embeddings
    if fe == "auto":
        fe = "on" if fast_embeddings in (True, "on") else "off"
    cmd += ["--fast-embeddings", fe]
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "6")
    env.setdefault("MKL_NUM_THREADS", "6")
    log(f"diarization: child process {' '.join(cmd[2:])}")
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError as exc:
        log(f"diarization: could not start child ({exc}) -> no speaker layer")
        return None
    for line in (proc.stderr or "").splitlines()[-15:]:
        log("  diarize> " + line)
    if proc.returncode != 0 or not rd.diarization_json.is_file():
        log(f"diarization: child exited {proc.returncode} -> no speaker layer")
        return None
    data = json.loads(rd.diarization_json.read_text(encoding="utf-8"))
    return DiarResult(turns=data["turns"], exclusive=data["exclusive"], labels=data["labels"], info=data["info"],
                      embeddings=data.get("embeddings", {}))
