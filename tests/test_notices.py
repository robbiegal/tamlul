"""Every model artefact and vendored source must have a notice (design §7, invariant 22)."""

from __future__ import annotations

import pathlib
import tomllib

from tamlul import registry

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTICES = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")


def test_every_artefact_has_a_notice():
    for aid, a in registry.artefacts().items():
        assert aid in NOTICES or a.repo in NOTICES, aid


def test_every_vendor_source_has_a_notice_and_a_dir():
    py = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    vendor = py["tool"]["tamlul"]["vendor"]
    vendor_dir = ROOT / "src/tamlul/_vendor"
    for name, meta in vendor.items():
        assert meta["repo"].split("github.com/")[-1].split("/")[1].lower() in NOTICES.lower() \
            or name in NOTICES, name
    for sub in vendor_dir.iterdir():
        if sub.is_dir() and sub.name != "__pycache__":
            assert sub.name in vendor, f"{sub.name} is vendored but not pinned in [tool.tamlul.vendor]"


def test_skills_use_literal_launcher_commands():
    for skill in (ROOT / "skills").glob("*/SKILL.md"):
        text = skill.read_text(encoding="utf-8")
        assert "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" in text, skill
        assert "$TAMLUL" not in text, f"{skill}: allowed-tools prefix rules refuse $VAR"
        head = text.split("---")[1]
        assert "description:" in head and "name:" in head
        assert len(head) < 3000
