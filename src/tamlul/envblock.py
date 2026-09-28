"""The privacy-hygiene environment block, applied to tamlul's OWN processes only.

This is not the invariant (the machine is online; Claude-side work is allowed).
It stops the libraries we load from phoning home about the user's files, and
it pins the HF/torch caches under the data directory. Verified reasons:

* pyannote.audio 4.0.7 creates an OTLP exporter to otel.pyannote.ai at import;
* huggingface_hub 1.32 fetches an agent-harness registry once per day and the
  ``hf`` CLI polls PyPI unless told not to; xet-core adds transfer telemetry
  gated by HF_XET_TELEMETRY_ENABLED;
* onnxruntime >= 1.29 Linux/macOS wheels carry a Microsoft 1DS uploader;
* ``import openvino`` (v1.1) sends a Google Analytics event unless a consent
  file contains ``0``.

The block is set with ``setdefault`` inside the CLI process before any heavy
import, and inherited by the diarization child process. It is NEVER written to
``CLAUDE_ENV_FILE`` or exported session-wide: that file is sourced before every
Bash command Claude runs in the session and would force unrelated Hugging Face
work offline (design review plug-2 / env-1).
"""

from __future__ import annotations

import os

from .paths import DataLayout

TELEMETRY_OFF = {
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_UPDATE_CHECK": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_XET_TELEMETRY_ENABLED": "0",
    "PYANNOTE_METRICS_ENABLED": "0",
    "OTEL_SDK_DISABLED": "true",
    "ORT_DISABLE_TELEMETRY": "1",
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONNOUSERSITE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

# Set at inference only. Models are local directories by then, so an accidental
# hub id fails fast instead of downloading; omitted for install/download.
OFFLINE = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}

# Subcommands that legitimately talk to the network.
ONLINE_COMMANDS = {"models", "env", "eval"}


def child_env(layout: DataLayout, *, offline: bool, base: dict | None = None) -> dict:
    env = dict(os.environ if base is None else base)
    env.update(TELEMETRY_OFF)
    env["HF_HOME"] = str(layout.hf_home)
    env["TORCH_HOME"] = str(layout.torch_home)
    env["TAMLUL_DATA"] = str(layout.root)
    if offline:
        env.update(OFFLINE)
    else:
        for k in OFFLINE:
            env.pop(k, None)
    return env


def apply_to_current_process(layout: DataLayout, *, offline: bool) -> None:
    """setdefault semantics: an explicit user override always wins."""
    for k, v in TELEMETRY_OFF.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("HF_HOME", str(layout.hf_home))
    os.environ.setdefault("TORCH_HOME", str(layout.torch_home))
    os.environ["TAMLUL_DATA"] = str(layout.root)
    if offline:
        for k, v in OFFLINE.items():
            os.environ.setdefault(k, v)
    else:
        for k in OFFLINE:
            os.environ.pop(k, None)


def switch_table() -> list[tuple[str, str | None]]:
    """What the doctor prints."""
    keys = list(TELEMETRY_OFF) + list(OFFLINE) + ["HF_HOME", "TORCH_HOME", "TAMLUL_DATA"]
    return [(k, os.environ.get(k)) for k in keys]
