from __future__ import annotations

import pathlib
import re

from tamlul.text.normalize import cer, edit_ops, heb_strict, ivrit_lb, wer

VENDOR = pathlib.Path(__file__).resolve().parent.parent / "src/tamlul/_vendor"


def test_ivrit_lb_edge_cases():
    # from benchmarks-datasets.md §4.2: gershayim splits, ASCII quote deletes, hyphen splits, brackets removed
    assert ivrit_lb('ד"ר') == "דר"
    assert ivrit_lb("ד״ר") == "ד ר"
    assert ivrit_lb("בן-גוריון") == "בן גוריון"
    # BasicTextNormalizer leaves a trailing space where punctuation was; jiwer splits on whitespace
    assert ivrit_lb("\u202Bשָׁלוֹם (ok) עולם!").split() == ["שלום", "עולם"]
    assert ivrit_lb("שלום  \u200Fעולם") == "שלום עולם"


def test_heb_strict_canonicalises_before_stripping():
    assert heb_strict("ד״ר") == heb_strict('ד"ר') == "דר"
    assert heb_strict("בן־גוריון") == "בן גוריון"
    assert heb_strict("שָׁלוֹם") == "שלום"
    assert heb_strict("אהה כן", drop_fillers=True) == "כן"
    assert heb_strict("ילדים", fold_finals=True) == "ילדימ"


def test_wer_micro_average():
    r = wer(["a b c", "d e"], ["a x c", "d e f"])
    assert r["S"] == 1 and r["I"] == 1 and r["D"] == 0 and r["N"] == 5
    assert abs(r["wer"] - 2 / 5) < 1e-9
    assert edit_ops(list("abc"), list("abd")) == (1, 0, 0, 2)
    assert cer(["שלום"], ["שלוט"]) == 0.25


def test_vendored_headers_and_hashes():
    for py in VENDOR.rglob("*.py"):
        if py.name == "__init__.py":
            continue
        head = py.read_text(encoding="utf-8").splitlines()[0]
        assert head.startswith("# derived from"), py
    for sub in VENDOR.iterdir():
        if sub.is_dir() and sub.name != "__pycache__":
            assert (sub / "VENDOR.md").is_file(), sub
    basic = (VENDOR / "openai_whisper_normalizer" / "basic.py").read_text(encoding="utf-8")
    m = re.search(r"Upstream file sha256: ([0-9a-f]{64})", basic)
    assert m and m.group(1) == "4742eaa040e0657fa1247a1361e0d856c62317a43326ea59a40c2e9edd8d2c38"
