"""Media probing, decoding and the true-stereo check (design §4.1).

Decoder chain: ffmpeg CLI -> PyAV (bundled with faster-whisper). The audio is
decoded ONCE, at its native channel count, so the stereo check can see both
channels; the mono 16 kHz float32 array the pipeline works on is then written
to the run directory (``audio.16k.f32``), and diarization reads that file
from its child process instead of decoding again.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
from dataclasses import asdict, dataclass

import numpy as np

SR = 16000

# is_true_stereo constants, vendored from KoganTheDev/hebrew-audio-transcriber
# core/audio_source.py @ 7ca0992 (MIT). Their field precision/recall is
# unmeasured (no public stereo Israeli call corpus exists); design §4.1.
STEREO_MAX_CORR = 0.5
STEREO_MIN_EXCLUSIVE_FRAC = 0.35
STEREO_LOUDER_RATIO = 4.0
STEREO_FRAME_MS = 100
STEREO_AUDIBLE_DBFS = -50.0

LOW_RMS_DBFS = -35.0        # speech level below this -> fixed gain (design §4.1)
TARGET_SPEECH_DBFS = -25.0
MAX_GAIN_DB = 30.0


@dataclass
class MediaInfo:
    path: str
    sha256: str
    duration_ms: int | None
    channels: int | None
    sample_rate: int | None
    codec: str | None
    container: str | None
    decoder: str
    true_stereo: bool = False
    stereo_corr: float | None = None
    stereo_exclusive_frac: float | None = None
    gain_db: float = 0.0
    integrated_rms_dbfs: float | None = None
    speech_level_dbfs: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ ffmpeg

def find_ffmpeg() -> str | None:
    from .doctor import find_ffmpeg as _f
    return _f()


def find_ffprobe(ffmpeg: str | None) -> str | None:
    p = shutil.which("ffprobe")
    if p:
        return p
    if ffmpeg:
        cand = pathlib.Path(ffmpeg).with_name("ffprobe" + pathlib.Path(ffmpeg).suffix)
        if cand.is_file():
            return str(cand)
    return None


def _run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def probe(path: pathlib.Path) -> dict:
    """ffprobe JSON -> ffmpeg banner parse -> PyAV. Returns a dict with decoder used."""
    ffmpeg = find_ffmpeg()
    ffprobe = find_ffprobe(ffmpeg)
    if ffprobe:
        try:
            out = _run([ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)], 60)
            if out.returncode == 0:
                info = json.loads(out.stdout.decode("utf-8", "replace"))
                streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
                s = streams[0] if streams else {}
                dur = info.get("format", {}).get("duration") or s.get("duration")
                return {"decoder": "ffprobe", "duration_ms": int(float(dur) * 1000) if dur else None,
                        "channels": int(s["channels"]) if s.get("channels") else None,
                        "sample_rate": int(s["sample_rate"]) if s.get("sample_rate") else None,
                        "codec": s.get("codec_name"), "container": info.get("format", {}).get("format_name")}
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    if ffmpeg:
        # Windows App Control can block ffprobe while ffmpeg still runs (design §3.1).
        try:
            out = _run([ffmpeg, "-hide_banner", "-i", str(path), "-f", "null", "-"], 600)
            text = out.stderr.decode("utf-8", "replace")
            import re
            dur = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
            aud = re.search(r"Audio:\s*(\w+).*?(\d+) Hz,\s*(mono|stereo|\d+ channels|[\d.]+)", text)
            ms = None
            if dur:
                h, m, s = dur.groups()
                ms = int((int(h) * 3600 + int(m) * 60 + float(s)) * 1000)
            ch = None
            if aud:
                layout = aud.group(3)
                ch = 1 if layout == "mono" else 2 if layout == "stereo" else (int(layout.split()[0]) if layout.endswith("channels") else None)
            return {"decoder": "ffmpeg", "duration_ms": ms, "channels": ch,
                    "sample_rate": int(aud.group(2)) if aud else None, "codec": aud.group(1) if aud else None,
                    "container": path.suffix.lstrip(".")}
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        import av
        with av.open(str(path)) as c:
            s = c.streams.audio[0]
            dur = c.duration / 1e6 if c.duration else (float(s.duration * s.time_base) if s.duration else None)
            return {"decoder": "pyav", "duration_ms": int(dur * 1000) if dur else None,
                    "channels": s.channels, "sample_rate": s.rate, "codec": s.codec_context.name,
                    "container": c.format.name}
    except Exception as exc:
        raise RuntimeError(f"cannot probe {path}: no ffmpeg/ffprobe and PyAV failed ({exc})") from exc


def decode(path: pathlib.Path, channels: int | None) -> tuple[np.ndarray, str]:
    """Decode to float32 at 16 kHz; shape (channels, n). Returns (array, decoder)."""
    ch = 2 if (channels or 1) >= 2 else 1
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        cmd = [ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-vn", "-f", "f32le", "-acodec", "pcm_f32le",
               "-ac", str(ch), "-ar", str(SR), "-"]
        try:
            out = _run(cmd, 3600)
            if out.returncode == 0 and len(out.stdout) >= 4:
                arr = np.frombuffer(out.stdout, dtype="<f4").astype(np.float32)
                return arr.reshape(-1, ch).T.copy(), "ffmpeg"
        except (OSError, subprocess.TimeoutExpired):
            pass
    import av
    frames = []
    with av.open(str(path)) as c:
        stream = c.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="stereo" if ch == 2 else "mono", rate=SR)
        for frame in c.decode(stream):
            for rf in resampler.resample(frame):
                frames.append(rf.to_ndarray())
        for rf in resampler.resample(None):
            frames.append(rf.to_ndarray())
    if not frames:
        return np.zeros((ch, 0), np.float32), "pyav"
    return np.concatenate(frames, axis=1).astype(np.float32), "pyav"


# ------------------------------------------------------------- stereo check

def is_true_stereo(left: np.ndarray, right: np.ndarray) -> tuple[bool, float, float]:
    """Two-party call recorded one voice per channel? Vendored thresholds.

    True when the channels are weakly correlated AND enough audible frames are
    much louder on one side. Duplicated mono has corr = 1 and can never pass,
    so the dangerous direction (fabricating two speakers) is closed.
    """
    n = min(len(left), len(right))
    if n < SR:
        return False, 1.0, 0.0
    lf_, rf_ = left[:n].astype(np.float64), right[:n].astype(np.float64)
    ls, rs = lf_.std(), rf_.std()
    corr = float(np.corrcoef(lf_, rf_)[0, 1]) if ls > 0 and rs > 0 else 1.0
    frame = int(SR * STEREO_FRAME_MS / 1000)
    m = n // frame
    lf = lf_[: m * frame].reshape(m, frame)
    rf = rf_[: m * frame].reshape(m, frame)
    le = np.sqrt((lf ** 2).mean(axis=1)) + 1e-12
    re = np.sqrt((rf ** 2).mean(axis=1)) + 1e-12
    thr = 10 ** (STEREO_AUDIBLE_DBFS / 20)
    audible = (le > thr) | (re > thr)
    if audible.sum() == 0:
        return False, corr, 0.0
    ratio = np.maximum(le / re, re / le)
    exclusive = float(((ratio >= STEREO_LOUDER_RATIO) & audible).sum() / audible.sum())
    ok = abs(corr) <= STEREO_MAX_CORR and exclusive >= STEREO_MIN_EXCLUSIVE_FRAC
    return bool(ok), corr, exclusive


def integrated_rms_dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return 20 * np.log10(rms + 1e-12)


def speech_level_dbfs(x: np.ndarray, frame_ms: int = 100) -> float:
    """The 90th percentile of 100-ms frame RMS: the level of the loud parts, silence excluded.

    Integrated RMS is dragged down by pauses and cannot tell a quiet recording
    from a sparse one; Silero VAD and the RMS gate are not level-invariant, so
    the gain decision must look at the speech itself.
    """
    if x.size < SR:
        return integrated_rms_dbfs(x)
    frame = int(SR * frame_ms / 1000)
    m = len(x) // frame
    f = x[: m * frame].astype(np.float64).reshape(m, frame)
    rms = np.sqrt((f ** 2).mean(axis=1))
    return float(20 * np.log10(np.percentile(rms, 90) + 1e-12))


# ----------------------------------------------------------------- extract

def extract(path: pathlib.Path, run_dir: pathlib.Path, *, channels_mode: str = "auto",
            sha256: str = "") -> tuple[MediaInfo, np.ndarray, list[np.ndarray]]:
    """Probe + decode + stereo check + gain; write audio.16k.f32 (+ per-channel files).

    Returns (info, mono, per_channel) where per_channel is empty unless the
    recording is treated as true stereo.
    """
    p = probe(path)
    arr, decoder = decode(path, p.get("channels"))
    info = MediaInfo(path=str(path), sha256=sha256, duration_ms=p.get("duration_ms") or int(arr.shape[1] / SR * 1000),
                     channels=p.get("channels") or arr.shape[0], sample_rate=p.get("sample_rate"),
                     codec=p.get("codec"), container=p.get("container"), decoder=decoder)
    per_channel: list[np.ndarray] = []
    if arr.shape[0] >= 2:
        ok, corr, excl = is_true_stereo(arr[0], arr[1])
        info.stereo_corr, info.stereo_exclusive_frac = round(corr, 4), round(excl, 4)
        if channels_mode == "split" or (channels_mode == "auto" and ok):
            info.true_stereo = True
            per_channel = [arr[0].copy(), arr[1].copy()]
    mono = arr.mean(axis=0).astype(np.float32) if arr.shape[0] > 1 else arr[0]
    rms = integrated_rms_dbfs(mono)
    level = speech_level_dbfs(mono)
    info.integrated_rms_dbfs = round(rms, 2)
    info.speech_level_dbfs = round(level, 2)
    if level < LOW_RMS_DBFS:
        gain_db = min(TARGET_SPEECH_DBFS - level, MAX_GAIN_DB)
        g = 10 ** (gain_db / 20)
        mono = np.clip(mono * g, -1.0, 1.0).astype(np.float32)
        per_channel = [np.clip(c * g, -1.0, 1.0).astype(np.float32) for c in per_channel]
        info.gain_db = round(gain_db, 2)
    mono.astype("<f4").tofile(run_dir / "audio.16k.f32")
    for i, c in enumerate(per_channel):
        c.astype("<f4").tofile(run_dir / f"audio.ch{i}.16k.f32")
    return info, mono, per_channel


def load_f32(path: pathlib.Path) -> np.ndarray:
    return np.fromfile(path, dtype="<f4").astype(np.float32)


def write_wav(path: pathlib.Path, audio: np.ndarray, sr: int = SR) -> None:
    import wave
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
