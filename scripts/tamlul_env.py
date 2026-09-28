#!/usr/bin/env python3
"""Locate, inspect and build the environment tamlul runs in.

Standard library only, on purpose: this runs *before* anything is installed,
on whatever Python the machine happens to have (any CPython >= 3.8 can start
it; the venv it builds needs 3.11-3.14).

Two facts about the plugin runtime shape everything here.

* The plugin directory is version-scoped and disposable
  (``~/.claude/plugins/cache/<marketplace>/<plugin>/<version>/``), so the venv
  lives in the DATA directory and the install is a regular one, never
  ``pip install -e``.
* The data directory is passed in explicitly (``--data``, the skills pass
  ``${CLAUDE_PLUGIN_DATA}``), or comes from ``TAMLUL_DATA`` / ``CLAUDE_PLUGIN_DATA``,
  or falls back to the official marketplace id. It is never derived from this
  script's own path (a ``--plugin-dir`` checkout is not the cache).

Interpreter policy (design review fea-6 / fea-14): no interpreter name exists
on every platform (stock macOS has only ``python3``; under Git Bash on Windows
``python3`` is the Store stub that exits 49), so the venv is created from the
first candidate that actually runs and reports 3.11-3.14, preferring 3.12
because that is the version the dependency pins were verified on.

torch is installed from the PyTorch CPU index BEFORE the ``diarize`` extra:
on Linux the PyPI ``torch`` wheel is the CUDA build with multi-GB nvidia-*
dependencies (design review feasibility-missed-2).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parent.parent
OFFICIAL_DATA_ID = "tamlul-robbiegal-tools"
DEFAULT_DATA = pathlib.Path.home() / ".claude" / "plugins" / "data" / OFFICIAL_DATA_ID

SUPPORTED = ((3, 11), (3, 12), (3, 13), (3, 14))
PREFERRED = (3, 12)

TORCH_PINS = ["torch==2.14.0", "torchaudio==2.11.0", "torchcodec==0.16.0"]
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

EXTRA_MODULES = {
    "asr": "faster_whisper",
    "diarize": "pyannote.audio",
    "export": "pympi",
    "eval": "jiwer",
    "cuda": "faster_whisper",
    "whispercpp": "pywhispercpp",
    "sherpa": "sherpa_onnx",
    "docx": "docx",
}

# Two lists on purpose: ``shutil.ignore_patterns`` matches basenames at every
# depth, and the repo-level ``eval/`` (results, converters' inputs) would then
# also drop the ``src/tamlul/eval`` package from the staged build.
_IGNORE_EVERYWHERE = shutil.ignore_patterns(
    ".git", "*.egg-info", "__pycache__", ".pytest_cache", "venv", ".venv", "node_modules",
)
_IGNORE_AT_ROOT = {
    "build", "dist", "tests", "examples", ".github", ".omc", "docs",
    "evals", "eval", "runs", "tamlul-data", "results",
}


def _not_needed_to_build(root: str):
    root_path = pathlib.Path(root)

    def ignore(directory: str, names: list[str]) -> set[str]:
        out = set(_IGNORE_EVERYWHERE(directory, names))
        if pathlib.Path(directory) == root_path:
            out |= _IGNORE_AT_ROOT & set(names)
        return out

    return ignore


# ----------------------------------------------------------------- locations

def data_dir(explicit: str | None) -> pathlib.Path:
    for cand in (explicit, os.environ.get("TAMLUL_DATA"), os.environ.get("CLAUDE_PLUGIN_DATA")):
        if cand:
            return pathlib.Path(cand).expanduser()
    return DEFAULT_DATA


def venv_python(home: pathlib.Path) -> pathlib.Path | None:
    for rel in ("venv/Scripts/python.exe", "venv/bin/python"):
        cand = home / rel
        if cand.exists():
            return cand
    return None


# ------------------------------------------------------------- interpreters

def _candidates() -> list[list[str]]:
    cands: list[list[str]] = []
    here = (sys.version_info.major, sys.version_info.minor)
    if here in SUPPORTED:
        cands.append([sys.executable])
    if os.name == "nt":
        for v in ("3.12", "3.13", "3.11", "3.14"):
            cands.append(["py", f"-{v}"])
    for v in ("3.12", "3.13", "3.11", "3.14"):
        cands.append([f"python{v}"])
    cands.append(["python3"])
    cands.append(["python"])
    return cands


def probe_interpreters() -> list[dict]:
    """Run each candidate; the Store stub fails here instead of later."""
    out = []
    seen = set()
    for cmd in _candidates():
        exe = shutil.which(cmd[0]) if not os.path.isabs(cmd[0]) else cmd[0]
        if not exe:
            continue
        try:
            proc = subprocess.run(
                cmd + ["-c", "import sys,json;print(json.dumps([sys.version_info[0],sys.version_info[1],sys.executable]))"],
                capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode != 0 or not proc.stdout.strip():
            out.append({"cmd": cmd, "ok": False, "exit": proc.returncode})
            continue
        try:
            major, minor, path = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            continue
        if path in seen:
            continue
        seen.add(path)
        out.append({"cmd": cmd, "ok": (major, minor) in SUPPORTED, "version": f"{major}.{minor}", "path": path})
    return out


def choose_interpreter(probes: list[dict] | None = None) -> str | None:
    probes = probes if probes is not None else probe_interpreters()
    good = [p for p in probes if p.get("ok")]
    for p in good:
        if p["version"] == f"{PREFERRED[0]}.{PREFERRED[1]}":
            return p["path"]
    return good[0]["path"] if good else None


# ------------------------------------------------------------------ checks

def find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    guesses = [
        pathlib.Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Kdenlive/bin/ffmpeg.exe",
        pathlib.Path(os.environ.get("ProgramFiles", "")) / "ffmpeg/bin/ffmpeg.exe",
        pathlib.Path("/usr/bin/ffmpeg"),
        pathlib.Path("/usr/local/bin/ffmpeg"),
        pathlib.Path("/opt/homebrew/bin/ffmpeg"),
    ]
    for g in guesses:
        if g.is_file():
            return str(g)
    return None


def long_paths_enabled() -> bool | None:
    if os.name != "nt":
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
            return bool(winreg.QueryValueEx(k, "LongPathsEnabled")[0])
    except OSError:
        return False


def _probe_env(python: pathlib.Path) -> dict:
    code = (
        "import json,importlib.util as u\n"
        f"mods={EXTRA_MODULES!r}\n"
        "try:\n"
        "    from tamlul import __version__ as v\n"
        "except Exception:\n"
        "    v=None\n"
        "print(json.dumps({'version':v,'extras':{k:(u.find_spec(m) is not None) for k,m in mods.items()}}))\n"
    )
    try:
        out = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"version": None, "extras": {}, "error": str(exc)}
    if out.returncode != 0:
        return {"version": None, "extras": {}, "error": out.stderr.strip()[:300]}
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"version": None, "extras": {}, "error": "probe returned no JSON"}


def stale(home: pathlib.Path) -> bool:
    stamp = home / "installed.json"
    if not stamp.is_file():
        return True
    try:
        recorded = json.loads(stamp.read_text(encoding="utf-8")).get("pyproject")
    except ValueError:
        return True
    return recorded != (PLUGIN_ROOT / "pyproject.toml").read_text(encoding="utf-8")


def status(home: pathlib.Path, *, probe: bool = True) -> dict:
    python = venv_python(home)
    out = {
        "plugin_root": str(PLUGIN_ROOT),
        "data_dir": str(home),
        "venv_python": str(python) if python else None,
        "installed": False,
        "version": None,
        "extras": {k: False for k in EXTRA_MODULES},
        "stale": stale(home) if python else None,
        "ffmpeg": find_ffmpeg(),
        "host_python": sys.executable,
        "host_python_version": ".".join(map(str, sys.version_info[:3])),
        "platform": platform.platform(),
        "long_paths_enabled": long_paths_enabled(),
        "venv_path_length": len(str(home / "venv")),
    }
    if python:
        p = _probe_env(python)
        out["installed"] = bool(p.get("version"))
        out["version"] = p.get("version")
        out["extras"].update(p.get("extras") or {})
        if p.get("error"):
            out["error"] = p["error"]
    if probe:
        out["interpreters"] = probe_interpreters()
        out["chosen_interpreter"] = choose_interpreter(out["interpreters"])
    return out


# ----------------------------------------------------------------- install

def _pip(python: pathlib.Path, *args: str) -> int:
    cmd = [str(python), "-m", "pip", "install", "--disable-pip-version-check", *args]
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def install(home: pathlib.Path, extras: list[str], *, upgrade: bool = False) -> int:
    home.mkdir(parents=True, exist_ok=True)
    python = venv_python(home)

    if os.name == "nt" and long_paths_enabled() is False and len(str(home / "venv")) > 120:
        print("WARNING: Windows long paths are disabled and the venv path is long; torch's deepest "
              "files can exceed MAX_PATH. Enable LongPathsEnabled or set TAMLUL_DATA to a short path.",
              file=sys.stderr, flush=True)

    if python is None:
        interp = choose_interpreter()
        if interp is None:
            print("No usable Python 3.11-3.14 interpreter found (the Microsoft Store alias does not "
                  "count). Install CPython from python.org or `winget install Python.Python.3.12`.",
                  file=sys.stderr)
            return 2
        print(f"Creating environment in {home / 'venv'} with {interp} ...", flush=True)
        made = subprocess.run([interp, "-m", "venv", str(home / "venv")], capture_output=True, text=True)
        if made.returncode != 0:
            print(made.stderr.strip()[:600], file=sys.stderr)
            print("Could not create a virtual environment (Debian/Ubuntu: apt install python3-venv).",
                  file=sys.stderr)
            return 2
        python = venv_python(home)
    if python is None:
        print("Environment created but no interpreter found inside it.", file=sys.stderr)
        return 2

    if _pip(python, "--upgrade", "pip", "wheel") != 0:
        return 2

    extras = sorted({e.strip() for e in extras if e.strip()})
    if "diarize" in extras and "cuda" not in extras:
        # CPU torch first, from the PyTorch index; the extra's pins are then already satisfied.
        args = ["--index-url", TORCH_CPU_INDEX, *TORCH_PINS]
        if upgrade:
            args.insert(0, "--upgrade")
        if _pip(python, *args) != 0:
            print("torch CPU wheels failed to install; see the error above.", file=sys.stderr)
            return 2

    print(f"Installing {PLUGIN_ROOT}[{','.join(extras)}]" if extras else f"Installing {PLUGIN_ROOT}", flush=True)
    with tempfile.TemporaryDirectory(prefix="tamlul_build_") as tmp:
        staged = pathlib.Path(tmp) / "tamlul"
        shutil.copytree(PLUGIN_ROOT, staged, ignore=_not_needed_to_build(str(PLUGIN_ROOT)))
        spec = str(staged) + (f"[{','.join(extras)}]" if extras else "")
        args = [spec]
        if upgrade:
            args.insert(0, "--upgrade")
        rc = _pip(python, *args)
    if rc != 0:
        return 2

    (home / "pyproject.toml").write_bytes((PLUGIN_ROOT / "pyproject.toml").read_bytes())
    (home / "installed.json").write_text(json.dumps({
        "plugin_root": str(PLUGIN_ROOT),
        "extras": extras,
        "python": str(python),
        "pyproject": (PLUGIN_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
    }, indent=2), encoding="utf-8")
    print("environment ready", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", help="data directory (default: TAMLUL_DATA / CLAUDE_PLUGIN_DATA / official id)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status", help="JSON description of the environment")
    s.add_argument("--no-probe", action="store_true", help="skip probing interpreters (faster)")
    sub.add_parser("stale", help="exit 0 if the environment matches the bundled package")
    p = sub.add_parser("install", help="create the environment and install tamlul into it")
    p.add_argument("--extras", default="asr,diarize,export",
                   help="comma-separated extras (default: asr,diarize,export)")
    p.add_argument("--upgrade", action="store_true")
    sub.add_parser("upgrade", help="reinstall with --upgrade and the recorded extras")
    args = ap.parse_args(argv)
    home = data_dir(args.data)

    if args.cmd == "status":
        print(json.dumps(status(home, probe=not args.no_probe), indent=2))
        return 0
    if args.cmd == "stale":
        return 1 if stale(home) else 0
    if args.cmd == "upgrade":
        try:
            extras = json.loads((home / "installed.json").read_text(encoding="utf-8")).get("extras", [])
        except (OSError, ValueError):
            extras = ["asr", "diarize", "export"]
        return install(home, extras, upgrade=True)
    return install(home, [e for e in args.extras.split(",")], upgrade=args.upgrade)


if __name__ == "__main__":
    sys.exit(main())
