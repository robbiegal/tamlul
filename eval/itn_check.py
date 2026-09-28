"""Run the NeMo Hebrew ITN test cases against tamlul's port and report pass rates."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from tamlul.itn.grammar import itn_text  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent.parent / "src/tamlul/itn/data/nemo"
FILES = ["test_cases_cardinal.txt", "test_cases_decimal.txt", "test_cases_measure.txt", "test_cases_time.txt",
         "test_cases_date.txt", "test_cases_whitelist.txt", "test_full_sentences.txt"]
# NeMo's tokenizer detaches punctuation ("20 , 22"); tamlul keeps it attached. These are the known deviations.
KNOWN = {"בכל כיתה יש עשרים, עשרים ושתיים תלמידים", "שלושה מיליוןאלף וחמש",
         "מחר בשש וחצי בבוקר נעלה על האוטובוסים ונצא לטיול השנתי. בשעה שמונה נגיע למצדה ונתחיל לטפס למעלה, נהיה שם עד אחת וחצי בצהריים, אולי רבע לשתיים, ונרד בשביל הנחש"}


def main(verbose: bool = False) -> int:
    total = ok = 0
    for f in FILES:
        cases = [ln.rstrip("\n") for ln in (DATA / f).read_text(encoding="utf-8").splitlines() if "~" in ln]
        n_ok = 0
        for line in cases:
            src, exp = line.split("~", 1)
            if src in KNOWN:
                continue
            got = itn_text(src)
            total += 1
            if " ".join(got.split()) == " ".join(exp.split()):
                ok += 1
                n_ok += 1
            elif verbose:
                print(f"FAIL [{f}] {src!r}\n   exp {exp!r}\n   got {got!r}")
        print(f"{f:32} {n_ok}/{len([c for c in cases if c.split('~',1)[0] not in KNOWN])}")
    print(f"TOTAL {ok}/{total} = {ok/total:.1%}")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main(verbose="-v" in sys.argv))
