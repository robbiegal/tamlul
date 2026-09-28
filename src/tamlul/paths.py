"""Where tamlul keeps everything mutable.

One data directory holds the virtualenv, the models, the runs, the HF/torch
caches, the calibration and the persistent config. Its location is resolved
in this order and nowhere else:

1. an explicit ``--data`` flag (the skills pass ``${CLAUDE_PLUGIN_DATA}``);
2. ``TAMLUL_DATA``;
3. ``CLAUDE_PLUGIN_DATA`` (set for hook/MCP processes only);
4. the official id of the marketplace install, as a last resort for humans
   at a terminal.

It is never derived from the plugin's own path: for a ``--plugin-dir`` install
``CLAUDE_PLUGIN_ROOT`` is the source checkout, not the cache, and the data id
is ``tamlul-inline`` (design review plug-missed-3).
"""

from __future__ import annotations

import os
import pathlib

OFFICIAL_DATA_ID = "tamlul-robbiegal-tools"
DEFAULT_DATA = pathlib.Path.home() / ".claude" / "plugins" / "data" / OFFICIAL_DATA_ID


def data_dir(explicit: str | os.PathLike | None = None) -> pathlib.Path:
    for cand in (explicit, os.environ.get("TAMLUL_DATA"), os.environ.get("CLAUDE_PLUGIN_DATA")):
        if cand:
            return pathlib.Path(cand).expanduser()
    return DEFAULT_DATA


class DataLayout:
    """Typed view of the sub-directories, created lazily."""

    def __init__(self, root: pathlib.Path):
        self.root = pathlib.Path(root)

    @property
    def venv(self) -> pathlib.Path:
        return self.root / "venv"

    @property
    def models(self) -> pathlib.Path:
        override = os.environ.get("TAMLUL_MODELS_DIR")
        return pathlib.Path(override).expanduser() if override else self.root / "models"

    @property
    def runs(self) -> pathlib.Path:
        return self.root / "runs"

    @property
    def hf_home(self) -> pathlib.Path:
        return self.root / "hf"

    @property
    def torch_home(self) -> pathlib.Path:
        return self.root / "torch"

    @property
    def tmp(self) -> pathlib.Path:
        return self.root / "tmp"

    @property
    def config(self) -> pathlib.Path:
        return self.root / "config.json"

    @property
    def calibration(self) -> pathlib.Path:
        return self.root / "calibration.json"

    @property
    def installed_stamp(self) -> pathlib.Path:
        return self.root / "installed.json"

    def ensure(self) -> DataLayout:
        for p in (self.root, self.models, self.runs, self.hf_home, self.torch_home, self.tmp):
            p.mkdir(parents=True, exist_ok=True)
        return self


def venv_python(root: pathlib.Path) -> pathlib.Path | None:
    for rel in ("venv/Scripts/python.exe", "venv/bin/python"):
        cand = pathlib.Path(root) / rel
        if cand.exists():
            return cand
    return None
