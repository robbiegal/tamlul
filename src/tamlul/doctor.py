"""`tamlul status` (JSON) and `tamlul doctor` (human report).

Neither downloads nor calibrates anything: the doctor reports and points at
the command that fixes what is missing (design §5, review calib-2).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from importlib import metadata

from . import __version__
from .envblock import switch_table
from .paths import DataLayout

CORE_PACKAGES = ("faster-whisper", "ctranslate2", "onnxruntime", "pyannote.audio", "torch",
                 "torchaudio", "torchcodec", "huggingface_hub", "numpy", "av")


def _pkg_versions() -> dict[str, str | None]:
    out = {}
    for name in CORE_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for g in (
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Kdenlive", "bin", "ffmpeg.exe"),
        os.path.join(os.environ.get("ProgramFiles", ""), "ffmpeg", "bin", "ffmpeg.exe"),
        "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg",
    ):
        if g and os.path.isfile(g):
            return g
    return None


def physical_cores() -> int:
    """Best effort: psutil is not a dependency, so fall back to os.cpu_count()."""
    try:
        import psutil  # type: ignore
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:  # noqa: BLE001
        pass
    n = os.cpu_count() or 4
    # Hybrid Intel parts report 2 threads per P-core only; halving 22 gives 11,
    # which is close to the 14 physical cores and safe. Calibration refines it.
    return max(2, n // 2)


def calibration_status(layout: DataLayout) -> dict:
    if not layout.calibration.is_file():
        return {"present": False, "stale": True, "reason": "never run"}
    try:
        cal = json.loads(layout.calibration.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"present": True, "stale": True, "reason": "unreadable"}
    reasons = []
    if cal.get("plugin_version") != __version__:
        reasons.append("plugin version changed")
    if cal.get("packages") != _pkg_versions():
        reasons.append("library versions changed")
    if cal.get("cpu") != platform.processor():
        reasons.append("CPU changed")
    return {"present": True, "stale": bool(reasons), "reason": "; ".join(reasons) or None,
            "id": cal.get("id"), "mode": cal.get("mode")}


def diarization_probe(timeout_s: float = 120.0) -> dict:
    """Ask the diarization child to import its stack (see ``tamlul.diarize.child.probe``)."""
    import subprocess
    if _pkg_versions().get("pyannote.audio") is None:
        return {"ok": False, "error": "pyannote.audio not installed (the diarize extra)", "steps": []}
    try:
        proc = subprocess.run([sys.executable, "-m", "tamlul.diarize.child", "--probe"], capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "steps": []}
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                break
    tail = (proc.stderr or "").strip().splitlines()[-1:] or [f"exit {proc.returncode}"]
    return {"ok": False, "error": tail[0], "steps": []}


def status(layout: DataLayout) -> dict:
    from . import registry
    from .config import resolve
    s = resolve({}, layout)
    return {
        "version": __version__,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "installed": True,
        "data_dir": str(layout.root),
        "models_dir": str(layout.models),
        "runs_dir": str(layout.runs),
        "ffmpeg": find_ffmpeg(),
        "pyav": _pkg_versions().get("av") is not None,
        "packages": _pkg_versions(),
        "logical_cpus": os.cpu_count(),
        "physical_cores_estimate": physical_cores(),
        "models": registry.status(layout),
        "profile_models": registry.ids_for(s.profile if s.profile != "auto" else "meeting",
                                           quality_max=(s.quality == "max"),
                                           diarization=None if s.diarization == "auto" else (s.diarization == "on")),
        "calibration": calibration_status(layout),
        "diarization": diarization_probe(),
        "config": {"effective": s.as_dict(), "sources": s.sources},
        "env": dict(switch_table()),
    }


def doctor_report(layout: DataLayout) -> str:
    st = status(layout)
    lines = [f"tamlul {st['version']}  ({st['python_version']} on {st['platform']})",
             f"data dir:   {st['data_dir']}",
             f"models dir: {st['models_dir']}",
             f"ffmpeg:     {st['ffmpeg'] or 'NOT FOUND (PyAV fallback: ' + ('available' if st['pyav'] else 'missing') + ')'}",
             "", "packages:"]
    for k, v in st["packages"].items():
        lines.append(f"  {k:18} {v or '-'}")
    lines += ["", "models (profile set marked *):"]
    for mid, m in st["models"].items():
        mark = "*" if mid in st["profile_models"] else " "
        lines.append(f" {mark} {mid:34} {'installed' if m['installed'] else 'missing':9} {m['size_bytes']/1e9:5.2f} GB  {m['license']}")
    missing = [mid for mid in st["profile_models"] if not st["models"][mid]["installed"]]
    if missing:
        total = sum(st["models"][m]["size_bytes"] for m in missing) / 1e9
        lines.append(f"  -> {len(missing)} model(s) missing for the active profile, {total:.2f} GB: "
                     "run  tamlul models install --yes")
    d = st["diarization"]
    lines += ["", "diarization stack (imported in the child process): "
              + (f"ok ({d.get('seconds')} s)" if d.get("ok") else "BROKEN -> transcripts will have no speaker layer")]
    if not d.get("ok"):
        lines.append(f"  {d.get('error')}")
        if "Application Control" in str(d.get("error", "")):
            lines.append("  Windows App Control / Smart App Control blocked a compiled module in the venv; "
                         "see Windows Security > App & browser control, or Event Viewer > CodeIntegrity/Operational")
    cal = st["calibration"]
    lines += ["", f"calibration: {'present' if cal['present'] else 'absent'}"
              + (f", stale ({cal['reason']})" if cal.get("stale") and cal.get("reason") else "")
              + ("  -> run  tamlul calibrate --quick" if cal.get("stale") else "")]
    lines += ["", "effective config (source):"]
    for k, v in st["config"]["effective"].items():
        src = st["config"]["sources"].get(k, "default")
        lines.append(f"  {k:16} {v!s:12} ({src})")
    lines += ["", "privacy switches in this process (process-scoped, never exported to the session):"]
    for k, v in st["env"].items():
        lines.append(f"  {k:28} {v if v is not None else '(unset)'}")
    return "\n".join(lines)
