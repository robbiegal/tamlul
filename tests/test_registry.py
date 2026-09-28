from __future__ import annotations

from tamlul import registry
from tamlul.paths import DataLayout


def test_manifest_is_consistent():
    arts = registry.artefacts()
    assert "asr/turbo-ct2@2025.05.13" in arts and "diarize/pyannote-community-1" in arts
    for a in arts.values():
        assert a.primary in a.files, a.id
        for rel, meta in a.files.items():
            assert len(meta["sha256"]) == 64 and int(meta["size"]) > 0, (a.id, rel)
        assert a.license
    for prof, ids in registry._manifest()["profiles"].items():
        for i in ids:
            assert i in arts, (prof, i)


def test_profile_ids():
    assert registry.ids_for("meeting") == ["asr/turbo-ct2@2025.05.13", "diarize/pyannote-community-1"]
    assert registry.ids_for("voice-note") == ["asr/turbo-ct2@2025.04.03"]
    assert registry.ids_for("voice-note", diarization=True)[-1] == "diarize/pyannote-community-1"
    assert registry.ids_for("meeting", quality_max=True)[-1] == "asr/large-v3-ct2@2025.05.13"
    assert registry.asr_id_for("2025.04.03") == "asr/turbo-ct2@2025.04.03"


def test_install_refuses_without_yes(data_dir, capsys):
    layout = DataLayout(data_dir)
    rc = registry.install(["diarize/pyannote-community-1"], layout, yes=False, log=print)
    assert rc == 2
    assert "refusing" in capsys.readouterr().out
    assert not (layout.models).exists() or not any(layout.models.iterdir())


def test_verify_detects_mismatch(data_dir):
    layout = DataLayout(data_dir)
    a = registry.artefacts()["diarize/pyannote-community-1"]
    d = registry.local_dir(a, layout)
    (d / "segmentation").mkdir(parents=True)
    (d / "segmentation" / "pytorch_model.bin").write_bytes(b"not the model")
    rep = registry.verify(a, layout)
    assert rep["ok"] is False
    f = rep["files"]["segmentation/pytorch_model.bin"]
    assert f["present"] and not f["sha256_ok"]
    assert not registry.is_installed(a, layout)
