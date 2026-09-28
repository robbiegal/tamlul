"""The NeMo Hebrew ITN test files are the acceptance suite (design §4.6)."""

from __future__ import annotations

import pathlib

import pytest

from tamlul.itn import itn_words
from tamlul.itn.grammar import itn_text

DATA = pathlib.Path(__file__).resolve().parent.parent / "src/tamlul/itn/data/nemo"
FILES = ["test_cases_cardinal.txt", "test_cases_decimal.txt", "test_cases_measure.txt", "test_cases_time.txt",
         "test_cases_date.txt", "test_cases_whitelist.txt", "test_full_sentences.txt"]
# NeMo's tokenizer detaches punctuation ("20 , 22"); tamlul keeps it attached. One test has a typo.
KNOWN_DEVIATIONS = {"בכל כיתה יש עשרים, עשרים ושתיים תלמידים", "שלושה מיליוןאלף וחמש",
                    "מחר בשש וחצי בבוקר נעלה על האוטובוסים ונצא לטיול השנתי. בשעה שמונה נגיע למצדה ונתחיל לטפס למעלה, נהיה שם עד אחת וחצי בצהריים, אולי רבע לשתיים, ונרד בשביל הנחש"}


def _cases():
    for f in FILES:
        for line in (DATA / f).read_text(encoding="utf-8").splitlines():
            if "~" in line:
                src, exp = line.rstrip("\n").split("~", 1)
                if src not in KNOWN_DEVIATIONS:
                    yield f, src, exp


@pytest.mark.parametrize("f,src,exp", list(_cases()))
def test_nemo_case(f, src, exp):
    assert " ".join(itn_text(src).split()) == " ".join(exp.split())


def test_punctuation_stays_attached():
    assert itn_text("זה עלה אלפיים שלוש מאות שקל.") == "זה עלה 2,300 שקל."
    assert itn_text("שלושים ושניים אחוז עלייה!") == "32% עלייה!"


def test_itn_words_collapses_multiword_numbers_and_keeps_t_raw():
    words = [{"i": k, "t": t, "t_raw": t} for k, t in enumerate(["יש", "עשרים", "ושתיים", "תלמידים"])]
    edits = []
    n = itn_words(words, edits)
    assert n == 2
    assert [w["t"] for w in words] == ["יש", "22", "", "תלמידים"]
    assert [w["t_raw"] for w in words] == ["יש", "עשרים", "ושתיים", "תלמידים"]
    assert edits[0]["kind"] == "itn"
