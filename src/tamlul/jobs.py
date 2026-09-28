"""Run directories, progress, resume bookkeeping (design §4.12).

A run lives at ``<data>/runs/<id>/`` (never next to the media - read-only,
synced and WhatsApp-export folders, plus permission prompts outside cwd).
``run.json`` records everything a resume must match; ``progress.json`` is what
Claude polls; ``segments.jsonl`` is appended per plan group with fsync so an
interrupted job loses at most the group in flight.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shutil
import time
from typing import Any

from .paths import DataLayout

RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


def slug(text: str, limit: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9֐-׿._-]+", "-", text).strip("-.")
    return (s or "run")[:limit]


def new_run_id(media_name: str) -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + slug(pathlib.Path(media_name).stem)


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: pathlib.Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: pathlib.Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


class RunDir:
    def __init__(self, path: pathlib.Path):
        self.path = pathlib.Path(path)

    @classmethod
    def create(cls, layout: DataLayout, media: pathlib.Path, out: str | None = None) -> RunDir:
        rid = new_run_id(media.name)
        base = pathlib.Path(out).expanduser() if out else layout.runs / rid
        base.mkdir(parents=True, exist_ok=True)
        (base / "exports").mkdir(exist_ok=True)
        return cls(base)

    @classmethod
    def open(cls, layout: DataLayout, ref: str) -> RunDir:
        p = pathlib.Path(ref).expanduser()
        if p.is_dir() and (p / "run.json").is_file():
            return cls(p)
        cand = layout.runs / ref
        if cand.is_dir():
            return cls(cand)
        raise FileNotFoundError(f"no run at {ref} or {cand}")

    @property
    def id(self) -> str:
        return self.path.name

    # --- files
    @property
    def run_json(self) -> pathlib.Path: return self.path / "run.json"
    @property
    def progress_json(self) -> pathlib.Path: return self.path / "progress.json"
    @property
    def log(self) -> pathlib.Path: return self.path / "log.txt"
    @property
    def media_json(self) -> pathlib.Path: return self.path / "media.json"
    @property
    def audio(self) -> pathlib.Path: return self.path / "audio.16k.f32"
    @property
    def chunks_json(self) -> pathlib.Path: return self.path / "chunks.json"
    @property
    def segments_jsonl(self) -> pathlib.Path: return self.path / "segments.jsonl"
    @property
    def diarization_rttm(self) -> pathlib.Path: return self.path / "diarization.rttm"
    @property
    def diarization_json(self) -> pathlib.Path: return self.path / "diarization.json"
    @property
    def transcript_json(self) -> pathlib.Path: return self.path / "transcript.json"
    @property
    def speakers_yaml(self) -> pathlib.Path: return self.path / "speakers.yaml"
    @property
    def edits_jsonl(self) -> pathlib.Path: return self.path / "edits.jsonl"
    @property
    def annotations_json(self) -> pathlib.Path: return self.path / "annotations.json"
    @property
    def exports(self) -> pathlib.Path: return self.path / "exports"

    # --- run.json
    def read_run(self) -> dict:
        return json.loads(self.run_json.read_text(encoding="utf-8")) if self.run_json.is_file() else {}

    def write_run(self, data: dict) -> None:
        atomic_write_json(self.run_json, data)

    def read_transcript(self) -> dict:
        return json.loads(self.transcript_json.read_text(encoding="utf-8"))

    def write_transcript(self, data: dict) -> None:
        atomic_write_json(self.transcript_json, data)

    # --- progress / log
    def set_progress(self, stage: str, pct: float, **extra: Any) -> None:
        cur = {}
        if self.progress_json.is_file():
            try:
                cur = json.loads(self.progress_json.read_text(encoding="utf-8"))
            except ValueError:
                cur = {}
        cur.update({"stage": stage, "pct": round(float(pct), 1), "last_update": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "run_id": self.id, "pid": os.getpid()})
        cur.update(extra)
        atomic_write_json(self.progress_json, cur)

    def log_line(self, msg: str) -> None:
        with self.log.open("a", encoding="utf-8") as fh:
            fh.write(time.strftime("%H:%M:%S ") + msg + "\n")

    # --- segments.jsonl
    def append_segments(self, group_index: int, segments: list[dict]) -> None:
        with self.segments_jsonl.open("a", encoding="utf-8") as fh:
            for seg in segments:
                fh.write(json.dumps({"group": group_index, **seg}, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def read_segments(self) -> list[dict]:
        if not self.segments_jsonl.is_file():
            return []
        out = []
        with self.segments_jsonl.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue  # a torn last line after a crash
        return out

    def groups_done(self) -> set[int]:
        return {s["group"] for s in self.read_segments() if "group" in s}

    def resume_hash(self) -> str | None:
        run = self.read_run()
        return run.get("resume_hash")


def resume_hash(model_sha: str, revision: str, compute_type: str, decode_opts: dict,
                chunks_sha: str, audio_sha: str) -> str:
    payload = json.dumps([model_sha, revision, compute_type, decode_opts, chunks_sha, audio_sha],
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def list_runs(layout: DataLayout) -> list[dict]:
    out = []
    if not layout.runs.is_dir():
        return out
    for d in sorted(layout.runs.iterdir()):
        if not d.is_dir():
            continue
        rd = RunDir(d)
        run = rd.read_run()
        prog = {}
        if rd.progress_json.is_file():
            try:
                prog = json.loads(rd.progress_json.read_text(encoding="utf-8"))
            except ValueError:
                pass
        out.append({"id": d.name, "media": run.get("media", {}).get("path"), "profile": run.get("profile"),
                    "stage": prog.get("stage"), "pct": prog.get("pct"),
                    "transcript": rd.transcript_json.is_file(),
                    "size_mb": round(sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) / 1e6, 1)})
    return out


def prune_runs(layout: DataLayout, older_than_days: int) -> list[str]:
    removed = []
    cutoff = time.time() - older_than_days * 86400
    if not layout.runs.is_dir():
        return removed
    for d in layout.runs.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d.name)
    return removed


def pid_alive(pid: int) -> bool | None:
    """True/False when it can be told, None when the platform gives no answer."""
    if not pid:
        return None
    if os.name == "nt":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            h = k32.OpenProcess(0x1000, False, int(pid))   # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return False
            code = ctypes.c_ulong()
            ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
            k32.CloseHandle(h)
            return bool(ok) and code.value == 259          # STILL_ACTIVE
        except Exception:  # noqa: BLE001
            return None
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None


FINAL_STAGES = {"done", "error"}


def progress(layout: DataLayout, ref: str) -> dict:
    """progress.json plus liveness: a job launched from a Claude session dies with that
    session, so ``alive``/``stalled`` tell the caller to re-run transcribe (which resumes)."""
    rd = RunDir.open(layout, ref)
    cur: dict = {"run_id": rd.id, "stage": "unknown", "pct": 0}
    if rd.progress_json.is_file():
        try:
            cur = json.loads(rd.progress_json.read_text(encoding="utf-8"))
        except ValueError:
            pass
    if cur.get("stage") not in FINAL_STAGES:
        alive = pid_alive(int(cur.get("pid") or 0))
        cur["alive"] = alive
        cur["stalled"] = alive is False
        if alive is False:
            cur["hint"] = "the job's process is gone (its session ended?); run `transcribe --resume <run id>` to continue"
    return cur
