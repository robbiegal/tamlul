"""Model registry: the pin manifest, local directories, verified downloads.

Every run-time artefact is a local directory under ``<data>/models/<id>``,
installed once from Hugging Face at a pinned revision and hashed against
``models.json`` before it is marked usable (a ``.ok`` stamp written only after
hashing - ``hf_hub_download`` never computes SHA-256 on the HTTP path and the
fallback from Xet is silent). At run time engines receive the directory, never
a hub id, so ``HF_HUB_OFFLINE=1`` in the child env makes an accidental hub
lookup fail fast instead of downloading.

Nothing here downloads without ``yes=True``: the skills pre-approve the
launcher for a turn, and that grant must never be able to fetch gigabytes
silently (design §4.13).
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import resources

from .paths import DataLayout

ProgressCb = Callable[[str, int, int], None]


@dataclass(frozen=True)
class Artefact:
    id: str
    kind: str
    role: str
    repo: str
    revision: str
    commit: str
    license: str
    attribution: str
    allow_patterns: tuple[str, ...]
    primary: str
    files: dict
    size_bytes: int
    optional: bool = False
    compute_type: dict | None = None

    @property
    def dirname(self) -> str:
        return self.id.replace("/", "__").replace("@", "__at__")


def _manifest() -> dict:
    with resources.files("tamlul").joinpath("models.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def artefacts() -> dict[str, Artefact]:
    m = _manifest()
    out = {}
    for aid, a in m["artefacts"].items():
        out[aid] = Artefact(
            id=aid, kind=a["kind"], role=a["role"], repo=a["repo"], revision=a["revision"],
            commit=a.get("commit", ""), license=a["license"], attribution=a.get("attribution", ""),
            allow_patterns=tuple(a.get("allow_patterns", ())), primary=a["primary"],
            files=a.get("files", {}), size_bytes=int(a.get("size_bytes", 0)),
            optional=bool(a.get("optional", False)), compute_type=a.get("compute_type"),
        )
    return out


def ids_for(profile: str, *, quality_max: bool = False, diarization: bool | None = None) -> list[str]:
    m = _manifest()
    ids = list(m["profiles"].get(profile, m["profiles"]["meeting"]))
    if quality_max:
        ids = [i for i in ids if not i.startswith("asr/")] + [m["quality_max"]]
    if diarization is False:
        ids = [i for i in ids if not i.startswith("diarize/")]
    if diarization is True and not any(i.startswith("diarize/") for i in ids):
        ids.append("diarize/pyannote-community-1")
    return ids


def asr_id_for(revision: str, *, quality_max: bool = False) -> str:
    if quality_max:
        return _manifest()["quality_max"]
    for aid in artefacts():
        if aid.startswith("asr/turbo-ct2@") and aid.endswith(revision):
            return aid
    raise KeyError(f"no turbo-ct2 artefact pinned for revision {revision}")


def local_dir(a: Artefact, layout: DataLayout) -> pathlib.Path:
    return layout.models / a.dirname


def stamp_path(a: Artefact, layout: DataLayout) -> pathlib.Path:
    return local_dir(a, layout) / ".ok"


def is_installed(a: Artefact, layout: DataLayout) -> bool:
    return stamp_path(a, layout).is_file() and (local_dir(a, layout) / a.primary).is_file()


def sha256_of(path: pathlib.Path, cb: ProgressCb | None = None, label: str = "") -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            done += len(chunk)
            if cb:
                cb(label or path.name, done, total)
    return h.hexdigest()


def verify(a: Artefact, layout: DataLayout, cb: ProgressCb | None = None) -> dict:
    """Hash every pinned file; return a report. Does not write the stamp."""
    d = local_dir(a, layout)
    report = {"id": a.id, "dir": str(d), "ok": True, "files": {}}
    for rel, meta in a.files.items():
        p = d / rel
        entry = {"present": p.is_file()}
        if p.is_file():
            entry["size"] = p.stat().st_size
            entry["size_ok"] = entry["size"] == int(meta["size"])
            entry["sha256"] = sha256_of(p, cb, f"hash {rel}")
            entry["sha256_ok"] = entry["sha256"] == meta["sha256"]
            entry["ok"] = entry["size_ok"] and entry["sha256_ok"]
        else:
            entry["ok"] = False
        report["files"][rel] = entry
        report["ok"] = report["ok"] and entry["ok"]
    return report


def status(layout: DataLayout) -> dict:
    out = {}
    for aid, a in artefacts().items():
        d = local_dir(a, layout)
        out[aid] = {
            "installed": is_installed(a, layout),
            "dir": str(d),
            "size_bytes": a.size_bytes,
            "license": a.license,
            "optional": a.optional,
            "revision": a.revision,
            "compute_type": (a.compute_type or {}).get("decision"),
        }
    return out


def plan(ids: Iterable[str], layout: DataLayout) -> list[dict]:
    """What an install would fetch, with sizes, for the consent prompt."""
    arts = artefacts()
    rows = []
    for i in ids:
        a = arts[i]
        rows.append({"id": i, "repo": a.repo, "revision": a.revision, "size_bytes": a.size_bytes,
                     "installed": is_installed(a, layout), "dir": str(local_dir(a, layout)),
                     "license": a.license})
    return rows


def _write_stamp(a: Artefact, layout: DataLayout, report: dict) -> None:
    d = local_dir(a, layout)
    observed = {}
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.name != ".ok" and ".cache" not in p.relative_to(d).parts:
            rel = p.relative_to(d).as_posix()   # huggingface_hub's .cache/ bookkeeping is not model content
            observed[rel] = {"size": p.stat().st_size,
                             "sha256": report["files"].get(rel, {}).get("sha256")}
    stamp_path(a, layout).write_text(json.dumps({
        "id": a.id, "repo": a.repo, "revision": a.revision, "commit": a.commit,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": observed,
    }, indent=2), encoding="utf-8")


def install(ids: Iterable[str], layout: DataLayout, *, yes: bool, dry_run: bool = False,
            cb: ProgressCb | None = None, log: Callable[[str], None] = print) -> int:
    """Download + verify each artefact. Returns 0 on success, 2 on refusal/failure."""
    ids = list(ids)
    rows = plan(ids, layout)
    todo = [r for r in rows if not r["installed"]]
    total = sum(r["size_bytes"] for r in todo)
    for r in rows:
        log(f"{'installed' if r['installed'] else 'to fetch':>9}  {r['id']:36} {r['size_bytes']/1e9:6.2f} GB  {r['license']}")
    log(f"total to download: {total/1e9:.2f} GB into {layout.models}")
    if dry_run:
        return 0
    if not todo:
        return 0
    if not yes:
        log("refusing to download without --yes (design invariant 14)")
        return 2

    # Imported here: huggingface_hub is a core dependency but the offline
    # variables must not be set for this command (envblock handles that).
    from huggingface_hub import snapshot_download

    arts = artefacts()
    layout.ensure()
    rc = 0
    for r in todo:
        a = arts[r["id"]]
        dest = local_dir(a, layout)
        dest.mkdir(parents=True, exist_ok=True)
        log(f"downloading {a.repo}@{a.revision} -> {dest}")
        for attempt in range(1, 4):
            try:
                snapshot_download(
                    repo_id=a.repo, revision=a.revision, local_dir=str(dest),
                    allow_patterns=list(a.allow_patterns) or None,
                    max_workers=4,
                )
                break
            except Exception as exc:  # noqa: BLE001 - reported, then retried
                log(f"download attempt {attempt} failed: {exc}")
                time.sleep(3 * attempt)
        else:
            rc = 2
            continue
        report = verify(a, layout, cb)
        bad = [k for k, v in report["files"].items() if not v["ok"]]
        if bad:
            log(f"HASH MISMATCH for {a.id}: {bad} - quarantining the directory")
            quarantine = dest.with_name(dest.name + ".bad-" + time.strftime("%Y%m%d%H%M%S"))
            _rename_retry(dest, quarantine)
            rc = 2
            continue
        _write_stamp(a, layout, report)
        log(f"verified {a.id}")
    return rc


def _rename_retry(src: pathlib.Path, dst: pathlib.Path, tries: int = 5) -> None:
    """Windows keeps files locked briefly after a download; retry the rename."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(0.5 * (i + 1))
    shutil.move(str(src), str(dst))


def remove(a: Artefact, layout: DataLayout) -> None:
    d = local_dir(a, layout)
    if d.exists():
        shutil.rmtree(d)
