"""Build the bundled calibration fixture from FLEURS he_il (CC BY 4.0).

Downloads `data/he_il/test.tsv` and `data/he_il/audio/test.tar.gz` (387 MB,
kept under <data>/eval/fleurs for the evaluation harness), takes test clips
in TSV order until the target duration is reached, resamples to 16 kHz mono, concatenates them
with 0.5 s of silence between clips, and writes:

    src/tamlul/fixtures/fleurs_he.wav        16-bit PCM, 16 kHz mono
    src/tamlul/fixtures/fleurs_he.jsonl      one row per clip: id, start_ms, end_ms, text
    src/tamlul/fixtures/ATTRIBUTION.md       CC BY 4.0 attribution block

Run once by the developer; the outputs are committed.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import wave

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "tamlul" / "fixtures"
DATA = pathlib.Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else pathlib.Path.home() / ".claude/plugins/data/tamlul-robbiegal-tools"
EVAL_DIR = DATA / "eval" / "fleurs"
TARGET_SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
GAP_S = 0.5
SR = 16000


def fetch_split() -> tuple[list[dict], pathlib.Path]:
    from huggingface_hub import hf_hub_download
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    tsv = hf_hub_download("google/fleurs", "data/he_il/test.tsv", repo_type="dataset", local_dir=str(EVAL_DIR))
    tar = hf_hub_download("google/fleurs", "data/he_il/audio/test.tar.gz", repo_type="dataset", local_dir=str(EVAL_DIR))
    rows = []
    for line in pathlib.Path(tsv).read_text(encoding="utf-8").splitlines():
        parts = line.split("	")
        if len(parts) < 4:
            continue
        rows.append({"id": parts[0], "file": parts[1], "raw_transcription": parts[2], "transcription": parts[3]})
    return rows, pathlib.Path(tar)


def decode_audio(data: bytes) -> np.ndarray:
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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pieces: list[np.ndarray] = []
    rows_out: list[dict] = []
    t = 0.0
    rows, tar_path = fetch_split()
    wanted = {r["file"]: r for r in rows}
    order = [r["file"] for r in rows]
    import tarfile
    got: dict[str, np.ndarray] = {}
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            name = pathlib.Path(member.name).name
            if name in wanted and name not in got:
                fh = tar.extractfile(member)
                if fh is None:
                    continue
                got[name] = decode_audio(fh.read())
                # the tar is not in TSV order; stop once the first N in TSV order are all present
                need = [f for f in order[:80] if f not in got]
                if not need:
                    break
    for fname in order:
        audio = got.get(fname)
        if audio is None or len(audio) < SR:
            continue
        row = wanted[fname]
        # per-clip loudness: FLEURS clips differ by tens of dB; bring each to -23 dBFS RMS so the
        # fixture exercises the pipeline at a normal level (a global peak normalisation left most
        # clips near -60 dBFS and Silero dropped them - found during phase 1 bring-up)
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) or 1e-6
        audio = np.clip(audio * (10 ** (-23 / 20) / rms), -0.99, 0.99).astype(np.float32)
        start_ms = int(round(t * 1000))
        pieces.append(audio)
        t += len(audio) / SR
        rows_out.append({"id": row["id"], "file": fname, "start_ms": start_ms, "end_ms": int(round(t * 1000)),
                         "text": row["transcription"], "raw_text": row["raw_transcription"]})
        pieces.append(np.zeros(int(GAP_S * SR), np.float32))
        t += GAP_S
        if t >= TARGET_SECONDS:
            break
    signal = np.concatenate(pieces)
    pcm = (signal * 32767).astype("<i2")
    with wave.open(str(OUT / "fleurs_he.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    with (OUT / "fleurs_he.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows_out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "ATTRIBUTION.md").write_text(
        "# Fixture attribution\n\n"
        "`fleurs_he.wav` / `fleurs_he.jsonl` are derived from the FLEURS dataset, Hebrew (he_il) test split.\n\n"
        "- Creator: Google (Conneau et al., \"FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech\", 2022)\n"
        "- Source: https://huggingface.co/datasets/google/fleurs\n"
        "- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/\n"
        f"- Modifications: {len(rows_out)} test clips concatenated with 0.5 s of digital silence between them, "
        "resampled to 16 kHz mono 16-bit PCM, each clip RMS-normalised to -23 dBFS. Reference transcriptions are unchanged.\n"
        "- Text origin: the reference sentences are the FLoRes Hebrew translations (Goyal et al., 2022; FLORES-200,\n"
        "  https://github.com/facebookresearch/flores, CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/), redistributed unchanged.\n"
        "- Used for: per-machine calibration (speed, engine disagreement, silence behaviour); never for accuracy claims.\n",
        encoding="utf-8")
    print(f"wrote {len(rows_out)} clips, {t:.1f} s, {len(pcm)*2/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
