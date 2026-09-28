"""Configuration precedence (design §4.16), stated once:

    CLI flag > TAMLUL_<KEY> environment variable > ./tamlul.toml in the working
    directory > <data>/config.json (``tamlul config set``) > profile default.

userConfig values reach us only because every SKILL.md passes the substituted
values as flags; an empty string means "unset" so a blank picker never
overrides a lower layer.
"""

from __future__ import annotations

import json
import os
import pathlib
import tomllib
from dataclasses import dataclass, field, fields
from typing import Any

from .paths import DataLayout

PROFILES: dict[str, dict[str, Any]] = {
    # revision: ivrit-ai/whisper-large-v3-turbo-ct2 tag. diarization: default on/off.
    "meeting": {"revision": "2025.05.13", "diarization": True, "itn": True, "verbatim": False,
                "pause_s": 1.0, "gap_s": 0.2, "annotation": "meeting"},
    "voice-note": {"revision": "2025.04.03", "diarization": False, "itn": True, "verbatim": False,
                   "pause_s": 1.0, "gap_s": 0.2, "annotation": "voice-note"},
    "interview": {"revision": "2025.05.13", "diarization": True, "itn": True, "verbatim": True,
                  "pause_s": 1.0, "gap_s": 0.2, "annotation": "interview"},
    "ca": {"revision": "2025.05.13", "diarization": True, "itn": False, "verbatim": True,
           "pause_s": 0.2, "gap_s": 0.2, "annotation": "ca"},
    "legal": {"revision": "2025.05.13", "diarization": True, "itn": False, "verbatim": True,
              "pause_s": 1.0, "gap_s": 0.2, "annotation": "legal"},
}
DEFAULT_PROFILE = "meeting"

WHATSAPP_CONTAINERS = {".opus", ".ogg", ".m4a", ".amr", ".oga"}

KEYS = ("profile", "quality", "compute_type", "threads", "diarization", "models_dir",
        "channels", "fast", "verbatim", "glossary", "participants", "speakers",
        "min_speakers", "max_speakers", "fast_embeddings", "align")


def resolve_auto_profile(duration_s: float | None, channels: int | None, suffix: str) -> str:
    """The ``auto`` heuristic of design §1.3; the choice is recorded in the run."""
    if (duration_s is not None and duration_s <= 600 and channels == 1
            and suffix.lower() in WHATSAPP_CONTAINERS):
        return "voice-note"
    return DEFAULT_PROFILE


@dataclass
class Settings:
    profile: str = DEFAULT_PROFILE
    quality: str = "default"          # default | max
    compute_type: str = "auto"        # auto | int8 | float32
    threads: int = 0                  # 0 = physical cores
    diarization: str = "auto"         # auto (profile) | on | off
    models_dir: str = ""
    channels: str = "auto"            # auto | mono | split
    fast: bool = False
    verbatim: str = "auto"            # auto | verbatim | clean
    glossary: str = ""
    participants: str = ""
    speakers: int = 0
    min_speakers: int = 0
    max_speakers: int = 0
    fast_embeddings: str = "auto"     # auto | on | off
    align: str = "dtw"                # dtw | ctc (v1.1)
    sources: dict[str, str] = field(default_factory=dict)

    def profile_defaults(self) -> dict[str, Any]:
        return PROFILES.get(self.profile, PROFILES[DEFAULT_PROFILE])

    def diarization_enabled(self) -> bool:
        if self.diarization in ("on", "off"):
            return self.diarization == "on"
        if self.speakers or self.min_speakers or self.max_speakers:
            return True
        return bool(self.profile_defaults()["diarization"])

    def verbatim_enabled(self) -> bool:
        if self.verbatim in ("verbatim", "clean"):
            return self.verbatim == "verbatim"
        return bool(self.profile_defaults()["verbatim"])

    def itn_enabled(self) -> bool:
        return bool(self.profile_defaults()["itn"]) and not self.verbatim_enabled()

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in KEYS}


_TYPES = {f.name: f.type for f in fields(Settings)}


def _coerce(key: str, value: Any) -> Any:
    t = _TYPES.get(key, "str")
    if value is None:
        return None
    if t == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if t == "bool":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    return str(value)


def _layer(values: dict[str, Any], source: str) -> dict[str, tuple[Any, str]]:
    out: dict[str, tuple[Any, str]] = {}
    for k, v in values.items():
        if k not in KEYS:
            continue
        if v is None or (isinstance(v, str) and v.strip() == ""):
            continue
        c = _coerce(k, v)
        if c is None:
            continue
        out[k] = (c, source)
    return out


def load_project_toml(cwd: pathlib.Path | None = None) -> dict[str, Any]:
    p = (cwd or pathlib.Path.cwd()) / "tamlul.toml"
    if not p.is_file():
        return {}
    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError):
        return {}
    return data.get("tamlul", data)


def load_data_config(layout: DataLayout) -> dict[str, Any]:
    if not layout.config.is_file():
        return {}
    try:
        return json.loads(layout.config.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def save_data_config(layout: DataLayout, values: dict[str, Any]) -> None:
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.config.write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")


def resolve(cli: dict[str, Any], layout: DataLayout, cwd: pathlib.Path | None = None) -> Settings:
    """Merge every layer; ``sources`` records which layer supplied each key."""
    merged: dict[str, tuple[Any, str]] = {}
    merged.update(_layer(load_data_config(layout), "config.json"))
    merged.update(_layer(load_project_toml(cwd), "tamlul.toml"))
    env = {k: os.environ.get("TAMLUL_" + k.upper()) for k in KEYS}
    merged.update(_layer(env, "env"))
    merged.update(_layer(cli, "flag"))
    s = Settings()
    for k, (v, src) in merged.items():
        setattr(s, k, v)
        s.sources[k] = src
    if s.profile not in PROFILES and s.profile != "auto":
        s.profile = DEFAULT_PROFILE
        s.sources["profile"] = "default(invalid value)"
    return s
