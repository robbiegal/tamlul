---
name: doctor
description: Check that tamlul can run - Python environment, models, ffmpeg, calibration, effective config - and build or repair it. Use before the first transcription, or when tamlul reports it is not set up.
disable-model-invocation: true
argument-hint: "[install|models|calibrate]"
allowed-tools: Bash(python "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Bash(python3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Bash(py -3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) PowerShell(python "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) PowerShell(py -3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Read
---

# Environment check

`PY` means `python` on Windows (`py -3` if it exits 49), `python3` elsewhere.
One launcher command per tool call: do not chain with `;`/`&&`, add `echo`, or
wrap it in a subshell - only the literal launcher command is pre-approved.
Nothing plugin-authored runs at plugin-install time, so the environment is
built here, on request, and every download is stated and confirmed first.

## 1. Look

```bash
PY "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" --data "${CLAUDE_PLUGIN_DATA}" env status --no-probe
PY "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" --data "${CLAUDE_PLUGIN_DATA}" doctor
```

Report a short table: environment path and version, extras present, ffmpeg,
which models are installed for the active profile, calibration state, and the
effective config with its sources.

## 2. Build or repair (each step asks first)

| Situation | Command | Say first |
|---|---|---|
| `installed: false` | `... env install` | ~2.5 GB of packages (torch CPU) into the data dir; a few minutes |
| models missing | `... models install --profile ${user_config.profile} --yes` | sizes from `models plan` (turbo 1.62 GB + pyannote 0.03 GB) |
| calibration absent/stale | `... calibrate --quick` | ~2-3 minutes; measures speed and the int8 kernel only |
| after a plugin update | `... env upgrade` | same size as the first install if pins changed |

Run installs and calibration in the background (`run_in_background: true`).

## 3. ffmpeg

`doctor` prints the path if it finds one (PATH or the usual side-installs).
Without it the PyAV fallback bundled with faster-whisper decodes most files;
say which decoder is active. To install: macOS `brew install ffmpeg`,
Debian/Ubuntu `sudo apt install ffmpeg`, Windows `winget install Gyan.FFmpeg`.

## Do not

- Do not `pip install -e` and do not install into the user's project
  environment; the venv lives in the plugin data directory.
- Do not export `HF_HUB_OFFLINE` or any tamlul variable into the session; the
  CLI scopes them to its own processes.
