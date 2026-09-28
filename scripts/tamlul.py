#!/usr/bin/env python3
"""The one entry point every skill types.

    python  "<plugin root>/scripts/tamlul.py" --data "<data dir>" <command> ...   (Windows)
    python3 "<plugin root>/scripts/tamlul.py" --data "<data dir>" <command> ...   (macOS/Linux)

Standard library only and runs on any CPython >= 3.8, because it is the thing
that has to work BEFORE the environment exists. It locates the data directory
(``--data`` > ``TAMLUL_DATA`` > ``CLAUDE_PLUGIN_DATA`` > the official id), finds
the venv inside it, and re-executes ``python -m tamlul.cli`` there with the
Hebrew-safe encoding variables set. Everything else (telemetry switches,
offline mode at inference) is applied by the CLI to its own process, so
nothing leaks into the user's shell or into ``CLAUDE_ENV_FILE``.

Two commands work without a venv: ``env ...`` is delegated to
``tamlul_env.py`` on the host interpreter, and ``status --json`` reports that
the environment is missing (the skills use it to decide what to ask the user).
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parent.parent
OFFICIAL_DATA_ID = "tamlul-robbiegal-tools"
DEFAULT_DATA = pathlib.Path.home() / ".claude" / "plugins" / "data" / OFFICIAL_DATA_ID


def _split_data(argv: list[str]) -> tuple[str | None, list[str]]:
    """Pull ``--data DIR`` / ``--data=DIR`` out of argv wherever it appears."""
    data = None
    rest: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--data" and i + 1 < len(argv):
            data = argv[i + 1]
            i += 2
            continue
        if a.startswith("--data="):
            data = a.split("=", 1)[1]
            i += 1
            continue
        rest.append(a)
        i += 1
    return data, rest


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


def _missing_env(home: pathlib.Path, as_json: bool) -> int:
    py = "python" if os.name == "nt" else "python3"
    hint = f'{py} "{PLUGIN_ROOT / "scripts" / "tamlul.py"}" --data "{home}" env install'
    if as_json:
        print(json.dumps({"installed": False, "data_dir": str(home), "venv_python": None,
                          "plugin_root": str(PLUGIN_ROOT), "install_hint": hint,
                          "estimated_download": {"venv_with_torch_cpu_gb": 2.5}}, indent=2))
        return 0
    sys.stderr.write(
        "tamlul: the Python environment is not built yet. In Claude Code run /tamlul:doctor, "
        f"or directly:\n\n    {hint}\n")
    return 2


def _drift_warning(home: pathlib.Path) -> None:
    """One line, once per data dir per plugin version, when the venv predates the bundled package."""
    recorded = home / "pyproject.toml"
    bundled = PLUGIN_ROOT / "pyproject.toml"
    marker = home / ".drift-warned"
    try:
        if recorded.is_file() and recorded.read_bytes() != bundled.read_bytes():
            tag = str(PLUGIN_ROOT)
            if not (marker.is_file() and marker.read_text(encoding="utf-8") == tag):
                sys.stderr.write("tamlul: the plugin was updated but its Python environment still "
                                 "holds the previous version - run /tamlul:doctor to refresh it.\n")
                marker.write_text(tag, encoding="utf-8")
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    explicit, rest = _split_data(argv)
    home = data_dir(explicit)

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["TAMLUL_DATA"] = str(home)
    env["TAMLUL_PLUGIN_ROOT"] = str(PLUGIN_ROOT)

    if rest and rest[0] == "env":
        cmd = [sys.executable, str(PLUGIN_ROOT / "scripts" / "tamlul_env.py"), "--data", str(home), *rest[1:]]
        return subprocess.run(cmd, env=env).returncode

    py = venv_python(home)
    if py is None:
        return _missing_env(home, as_json=(rest[:1] == ["status"]))

    _drift_warning(home)
    cmd = [str(py), "-m", "tamlul.cli", *rest]
    try:
        return subprocess.run(cmd, env=env).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
