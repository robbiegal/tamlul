from __future__ import annotations

from tamlul.config import PROFILES, Settings, resolve, resolve_auto_profile, save_data_config
from tamlul.paths import DataLayout


def test_precedence_flag_over_env_over_toml_over_config(data_dir, tmp_path, monkeypatch):
    layout = DataLayout(data_dir)
    save_data_config(layout, {"profile": "ca", "threads": "4"})
    (tmp_path / "tamlul.toml").write_text('[tamlul]\nprofile = "legal"\n', encoding="utf-8")
    monkeypatch.setenv("TAMLUL_PROFILE", "interview")
    s = resolve({"profile": "voice-note"}, layout, cwd=tmp_path)
    assert s.profile == "voice-note" and s.sources["profile"] == "flag"
    assert s.threads == 4 and s.sources["threads"] == "config.json"
    s2 = resolve({}, layout, cwd=tmp_path)
    assert s2.profile == "interview" and s2.sources["profile"] == "env"
    monkeypatch.delenv("TAMLUL_PROFILE")
    s3 = resolve({}, layout, cwd=tmp_path)
    assert s3.profile == "legal" and s3.sources["profile"] == "tamlul.toml"


def test_empty_string_means_unset(data_dir):
    layout = DataLayout(data_dir)
    save_data_config(layout, {"profile": "ca"})
    s = resolve({"profile": "", "threads": "${user_config.threads}"}, layout)
    assert s.profile == "ca"
    assert s.threads == 0


def test_invalid_profile_falls_back(data_dir):
    s = resolve({"profile": "nonsense"}, DataLayout(data_dir))
    assert s.profile == "meeting"


def test_profile_defaults_and_flags():
    s = Settings(profile="voice-note")
    assert not s.diarization_enabled()
    s.speakers = 2
    assert s.diarization_enabled()
    assert Settings(profile="ca").itn_enabled() is False
    assert Settings(profile="meeting").itn_enabled() is True
    assert Settings(profile="meeting", verbatim="verbatim").itn_enabled() is False
    for d in PROFILES.values():
        assert d["revision"] in ("2025.05.13", "2025.04.03")


def test_auto_profile_heuristic():
    assert resolve_auto_profile(120, 1, ".opus") == "voice-note"
    assert resolve_auto_profile(1200, 1, ".opus") == "meeting"
    assert resolve_auto_profile(120, 2, ".opus") == "meeting"
    assert resolve_auto_profile(120, 1, ".wav") == "meeting"
