"""Evaluation sets: what they are, where they come from, how to iterate them.

Each set yields ``Utt`` rows (id, 16 kHz mono float32 audio, reference text,
duration). Gated ivrit.ai sets need ``--token`` once; the token goes to
``snapshot_download`` explicitly and is never stored by tamlul.

Publication rules (design §4.15) live here as data: ``publish_refs`` says
whether reference text may appear in a published per-utterance CSV.
"""

from __future__ import annotations

import io
import json
import pathlib
import subprocess
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from ..media import SR, find_ffmpeg
from ..paths import DataLayout


@dataclass(frozen=True)
class EvalSet:
    name: str
    repo: str
    split: str
    text_col: str
    gated: bool
    licence: str
    publish_refs: bool          # may reference text appear in a published CSV?
    publishable: bool           # may the set's numbers appear in published tables?
    kind: str                   # "parquet" | "fleurs" | "folder"
    note: str = ""


SETS: dict[str, EvalSet] = {
    "eval-whatsapp": EvalSet("eval-whatsapp", "ivrit-ai/eval-whatsapp", "test", "text", True,
                             "ivrit.ai licence (as presented at the gate; recorded at download)", False, True, "parquet",
                             "54 voice notes, 70.8 min; leaderboard set"),
    "eval-d1": EvalSet("eval-d1", "ivrit-ai/eval-d1", "test", "text", True,
                       "ivrit.ai licence (as presented at the gate; recorded at download)", False, True, "parquet",
                       "one 2-speaker podcast in 17 chunks, 1.43 h; no speaker labels"),
    "saspeech": EvalSet("saspeech", "upai-inc/saspeech", "test", "text", True,
                        "as tagged on the Hub (recorded at download)", False, True, "parquet",
                        "single-speaker read/spontaneous, 2,986 utterances"),
    "fleurs": EvalSet("fleurs", "google/fleurs", "test", "transcription", False, "CC BY 4.0", True, True, "fleurs",
                      "he_il test split, 792 utterances; the only set whose references may be published"),
    "kan": EvalSet("kan", "imvladikon/hebrew_speech_kan", "validation", "sentence", False,
                   "none stated (scraped subtitles)", False, False, "parquet",
                   "evaluated locally only; never in published tables"),
}


@dataclass
class Utt:
    id: str
    audio: np.ndarray
    text: str
    duration_s: float
    speaker: str = ""
    extra: dict | None = None


def eval_root(layout: DataLayout) -> pathlib.Path:
    return layout.root / "eval"


def set_dir(layout: DataLayout, s: EvalSet) -> pathlib.Path:
    return eval_root(layout) / s.name


# ------------------------------------------------------------------ download

def download(name: str, layout: DataLayout, *, token: str | None) -> dict:
    """snapshot_download the set into <data>/eval/<name>/hub; records the licence note.

    The token is passed explicitly; nothing implicit is ever used (env sets
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1).
    """
    s = SETS[name]
    from huggingface_hub import snapshot_download
    d = set_dir(layout, s)
    d.mkdir(parents=True, exist_ok=True)
    if s.gated and not token:
        raise SystemExit(f"{s.repo} is gated: pass --token (or HF_TOKEN) once; tamlul does not store it")
    kwargs = {"repo_id": s.repo, "repo_type": "dataset", "local_dir": str(d / "hub"), "token": token}
    if s.kind == "fleurs":
        kwargs["allow_patterns"] = ["data/he_il/test.tsv", "data/he_il/audio/test.tar.gz", "README.md"]
    else:
        kwargs["allow_patterns"] = ["*.parquet", "*.json", "*.jsonl", "*.csv", "*.md", "*.txt", "*.tsv",
                                    "*.wav", "*.mp3", "*.opus", "*.ogg", "*.m4a", "*.flac", "LICENSE*"]
    path = snapshot_download(**kwargs)
    card = pathlib.Path(path) / "README.md"
    licence_text = card.read_text(encoding="utf-8", errors="replace")[:4000] if card.is_file() else ""
    manifest = {"set": s.name, "repo": s.repo, "path": str(path), "licence_note": s.licence,
                "card_head": licence_text, "publish_refs": s.publish_refs, "publishable": s.publishable}
    (d / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def is_downloaded(name: str, layout: DataLayout) -> bool:
    return (set_dir(layout, SETS[name]) / "manifest.json").is_file()


# ------------------------------------------------------------------ decoding

def decode_bytes(data: bytes) -> np.ndarray:
    """Any container/codec -> 16 kHz mono float32, via ffmpeg (stdin) or PyAV."""
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        try:
            out = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-i", "pipe:0", "-vn", "-f", "f32le",
                                  "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(SR), "-"],
                                 input=data, capture_output=True, timeout=600)
            if out.returncode == 0 and len(out.stdout) >= 4:
                return np.frombuffer(out.stdout, dtype="<f4").astype(np.float32)
        except (OSError, subprocess.TimeoutExpired):
            pass
    import av
    container = av.open(io.BytesIO(data))
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=SR)
    chunks = []
    for frame in container.decode(stream):
        for rf in resampler.resample(frame):
            chunks.append(rf.to_ndarray()[0])
    for rf in resampler.resample(None):
        chunks.append(rf.to_ndarray()[0])
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


# ------------------------------------------------------------------ iteration

def _parquet_files(hub: pathlib.Path, split: str) -> list[pathlib.Path]:
    files = sorted(hub.rglob("*.parquet"))
    by_split = [f for f in files if split in f.name or split in {p.name for p in f.parents}]
    return by_split or files


def _iter_parquet(s: EvalSet, hub: pathlib.Path, limit: int | None) -> Iterator[Utt]:
    import pyarrow.parquet as pq
    n = 0
    for f in _parquet_files(hub, s.split):
        pf = pq.ParquetFile(f)
        cols = pf.schema_arrow.names
        audio_col = next((c for c in cols if c == "audio"), None) or next(
            (c for c in cols if "audio" in c.lower()), None)
        text_col = s.text_col if s.text_col in cols else next(
            (c for c in cols if c.lower() in ("text", "sentence", "transcription", "transcript")), None)
        if audio_col is None or text_col is None:
            raise RuntimeError(f"{f.name}: cannot find audio/text columns in {cols}")
        id_col = next((c for c in cols if c.lower() in ("id", "uuid", "name", "file", "path")), None)
        spk_col = next((c for c in cols if "speaker" in c.lower()), None)
        for batch in pf.iter_batches(batch_size=16):
            rows = batch.to_pylist()
            for row in rows:
                a = row[audio_col]
                data = a.get("bytes") if isinstance(a, dict) else None
                if data is None and isinstance(a, dict) and a.get("path"):
                    p = hub / a["path"]
                    data = p.read_bytes() if p.is_file() else None
                if data is None:
                    continue
                audio = decode_bytes(data)
                uid = str(row.get(id_col) if id_col else "") or (a.get("path") if isinstance(a, dict) else "") \
                    or f"{f.stem}-{n}"
                yield Utt(id=str(uid), audio=audio, text=str(row[text_col] or ""), duration_s=len(audio) / SR,
                          speaker=str(row.get(spk_col) or "") if spk_col else "",
                          extra={k: v for k, v in row.items() if k not in (audio_col,) and not isinstance(v, (bytes, dict))})
                n += 1
                if limit and n >= limit:
                    return


def fleurs_rows(hub: pathlib.Path) -> list[dict]:
    tsv = hub / "data" / "he_il" / "test.tsv"
    rows = []
    for line in tsv.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        rows.append({"id": parts[0], "file": parts[1], "raw_transcription": parts[2], "transcription": parts[3],
                     "gender": parts[6] if len(parts) > 6 else ""})
    return rows


def _iter_fleurs(hub: pathlib.Path, limit: int | None) -> Iterator[Utt]:
    rows = fleurs_rows(hub)
    wanted = {r["file"]: r for r in rows}
    n = 0
    with tarfile.open(hub / "data" / "he_il" / "audio" / "test.tar.gz", "r:gz") as tar:
        for member in tar:
            name = pathlib.Path(member.name).name
            row = wanted.get(name)
            if row is None:
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            audio = decode_bytes(fh.read())
            yield Utt(id=row["id"], audio=audio, text=row["transcription"], duration_s=len(audio) / SR,
                      extra={"raw_transcription": row["raw_transcription"], "file": name, "gender": row["gender"]})
            n += 1
            if limit and n >= limit:
                return


def _iter_folder(hub: pathlib.Path, s: EvalSet, limit: int | None) -> Iterator[Utt]:
    """audio files + a metadata.{jsonl,csv} with file_name/text (the Hub's AudioFolder layout)."""
    meta = next((p for p in hub.rglob("metadata.jsonl")), None) or next((p for p in hub.rglob("metadata.csv")), None)
    if meta is None:
        raise RuntimeError(f"no parquet and no metadata.jsonl/csv under {hub}")
    rows: list[dict] = []
    if meta.suffix == ".jsonl":
        rows = [json.loads(ln) for ln in meta.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        import csv
        with open(meta, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    n = 0
    for row in rows:
        p = meta.parent / row.get("file_name", "")
        if not p.is_file():
            continue
        audio = decode_bytes(p.read_bytes())
        yield Utt(id=row.get("id") or p.stem, audio=audio, text=str(row.get(s.text_col) or row.get("text") or ""),
                  duration_s=len(audio) / SR, speaker=str(row.get("speaker") or ""))
        n += 1
        if limit and n >= limit:
            return


def utterances(name: str, layout: DataLayout, *, limit: int | None = None) -> Iterator[Utt]:
    s = SETS[name]
    hub = set_dir(layout, s) / "hub"
    if s.kind == "fleurs" and not (hub / "data" / "he_il" / "test.tsv").is_file():
        # the calibration-fixture builder keeps the same files under <data>/eval/fleurs (no hub/)
        alt = set_dir(layout, s)
        if (alt / "data" / "he_il" / "test.tsv").is_file():
            hub = alt
    if not hub.is_dir():
        raise SystemExit(f"{name} is not downloaded: tamlul eval download {name}" + (" --token ..." if s.gated else ""))
    if s.kind == "fleurs":
        yield from _iter_fleurs(hub, limit)
    elif any(hub.rglob("*.parquet")):
        yield from _iter_parquet(s, hub, limit)
    else:
        yield from _iter_folder(hub, s, limit)
