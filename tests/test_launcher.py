from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "scripts" / "tamlul.py"
ENV = ROOT / "scripts" / "tamlul_env.py"


def _load(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_launcher_splits_data_flag_anywhere():
    m = _load(LAUNCHER)
    assert m._split_data(["--data", "X", "status", "--json"]) == ("X", ["status", "--json"])
    assert m._split_data(["status", "--data=Y"]) == ("Y", ["status"])
    assert m._split_data(["transcribe", "f.wav"]) == (None, ["transcribe", "f.wav"])


def test_launcher_reports_missing_env_as_json(tmp_path):
    out = subprocess.run([sys.executable, str(LAUNCHER), "--data", str(tmp_path / "nowhere"), "status", "--json"],
                         capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0
    js = json.loads(out.stdout)
    assert js["installed"] is False and "install_hint" in js


def test_launcher_missing_env_other_command_exits_2(tmp_path):
    out = subprocess.run([sys.executable, str(LAUNCHER), "--data", str(tmp_path / "nowhere"), "doctor"],
                         capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 2
    assert "not built yet" in out.stderr


def test_env_status_runs_without_venv(tmp_path):
    out = subprocess.run([sys.executable, str(ENV), "--data", str(tmp_path), "status", "--no-probe"],
                         capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0
    js = json.loads(out.stdout)
    assert js["installed"] is False and js["data_dir"] == str(tmp_path)


def test_interpreter_probe_finds_current_python():
    m = _load(ENV)
    probes = m.probe_interpreters()
    assert any(p.get("ok") for p in probes), probes
    assert m.choose_interpreter(probes) is not None
