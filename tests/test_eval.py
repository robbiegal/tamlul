"""The evaluation harness on synthetic inputs: scorer, speaker metrics, leaderboard check, timestamps."""

from __future__ import annotations

import csv
import json
import pathlib

import pytest

from tamlul.eval import lb, score, speakers, timestamps
from tamlul.eval.runasr import COLUMNS, score_row


def _rows(pairs: list[tuple[str, str]]) -> list[dict]:
    out = []
    for k, (ref, hyp) in enumerate(pairs):
        r = {"id": str(k), "reference_text": ref, "predicted_text": hyp, "audio_duration": 2.0, "transcription_time": 1.0}
        r.update(score_row(ref, hyp))
        out.append(r)
    return out


def test_score_basic_rates():
    rows = _rows([("שלום עולם טוב", "שלום עולם טוב"),           # 0/3
                  ("אני הולך הביתה עכשיו", "אני הולך הביתה"),   # 1 deletion /4
                  ("אה זה בסדר", "זה בסדר")])                   # filler deleted: 1/3 as-is, 0/2 filler-insensitive
    s = score.score(rows, resamples=200)
    assert s["n_utts"] == 3 and s["ref_words"] == 10
    assert s["wer"]["value"] == pytest.approx(2 / 10, abs=1e-4)
    assert s["wer_nofiller"]["value"] == pytest.approx(1 / 9, abs=1e-4)
    lo, hi = s["wer"]["ci95"]
    assert lo <= s["wer"]["value"] <= hi
    assert s["cer"]["value"] > 0 and s["cer_nospace"]["value"] > 0
    assert s["rtf"] == pytest.approx(2.0)


def test_score_latin_metrics():
    rows = _rows([("עדכנתי את zoom אתמול", "עדכנתי את זום אתמול"),     # transliterated -> tolerant hit
                  ("זה docker פשוט", "זה בלה פשוט"),                     # missed both ways
                  ("שלום לכולם", "שלום hello לכולם")])                    # spurious latin
    s = score.score(rows, resamples=100)
    assert s["latin_err"]["n"] == 2 and s["latin_err"]["value"] == pytest.approx(1.0)
    assert s["latin_err_tolerant"]["value"] == pytest.approx(0.5)
    assert s["spurious_latin"]["value"] == pytest.approx(1 / (4 + 3 + 3))


def test_paired_diff_identical_and_worse():
    a = _rows([("אחד שתיים שלוש", "אחד שתיים שלוש"), ("ארבע חמש", "ארבע חמש")])
    same = score.paired_diff(a, a, resamples=200)
    assert same["diff"] == 0 and same["diff_ci95"] == [0.0, 0.0]
    b = _rows([("אחד שתיים שלוש", "אחד שלוש"), ("ארבע חמש", "ארבע חמש")])
    worse = score.paired_diff(a, b, resamples=200)
    assert worse["diff"] == pytest.approx(0.2) and worse["p_b_worse"] > 0.5


def test_der_and_cpwer():
    ref = [{"s": 0, "e": 10000, "spk": "A", "text": "שלום מה שלומך"},
           {"s": 10000, "e": 20000, "spk": "B", "text": "בסדר גמור תודה"}]
    hyp_swapped = [{"s": 0, "e": 10000, "spk": "S2", "text": "שלום מה שלומך"},
                   {"s": 10000, "e": 20000, "spk": "S1", "text": "בסדר גמור תודה"}]
    d = speakers.der(ref, hyp_swapped, collar_s=0.0)
    assert d["der"] == pytest.approx(0.0) and d["mapping"] == {"S2": "A", "S1": "B"}
    late = [{"s": 0, "e": 11000, "spk": "X"}, {"s": 11000, "e": 20000, "spk": "Y"}]   # 1 s of confusion
    d2 = speakers.der(ref, late, collar_s=0.0)
    assert d2["confusion"] == pytest.approx(1 / 20) and d2["der"] == pytest.approx(1 / 20)
    assert speakers.der(ref, late, collar_s=1.0)["der"] == pytest.approx(0.0)   # inside the collar
    cp = speakers.cpwer(speakers.streams_from_turns(ref), speakers.streams_from_turns(hyp_swapped))
    assert cp["cpwer"] == pytest.approx(0.0) and cp["assignment"] == {"S2": "A", "S1": "B"}
    one_wrong = [dict(hyp_swapped[0]), dict(hyp_swapped[1], text="בסדר גמור")]
    assert speakers.cpwer(speakers.streams_from_turns(ref), speakers.streams_from_turns(one_wrong))["cpwer"] == pytest.approx(1 / 6)
    assert speakers.speaker_count(2, 3) == {"ref": 2, "hyp": 3, "exact": False, "abs_delta": 1}


def test_der_matches_pyannote_metrics_when_available():
    try:                                   # not installed, or its compiled deps (pandas .pyd) blocked by
        import pyannote.metrics.diarization as pm  # Windows App Control: skip, don't fail
        from pyannote.core import Annotation, Segment
    except ImportError as exc:
        pytest.skip(f"pyannote.metrics not loadable here: {exc}")
    ref = [{"s": 0, "e": 5000, "spk": "A"}, {"s": 4000, "e": 9000, "spk": "B"}, {"s": 12000, "e": 15000, "spk": "A"}]
    hyp = [{"s": 200, "e": 5500, "spk": "1"}, {"s": 5500, "e": 8500, "spk": "2"}, {"s": 12500, "e": 16000, "spk": "1"}]

    def ann(turns):
        a = Annotation()
        for k, t in enumerate(turns):
            a[Segment(t["s"] / 1000, t["e"] / 1000), k] = t["spk"]
        return a

    theirs = pm.DiarizationErrorRate(collar=0.0, skip_overlap=False)(ann(ref), ann(hyp))
    ours = speakers.der(ref, hyp, collar_s=0.0)["der"]
    assert ours == pytest.approx(theirs, abs=1e-6)
    theirs_c = pm.DiarizationErrorRate(collar=0.5, skip_overlap=False)(ann(ref), ann(hyp))   # pyannote: total collar
    ours_c = speakers.der(ref, hyp, collar_s=0.25)["der"]                                    # ours: per side
    assert ours_c == pytest.approx(theirs_c, abs=1e-6)


def _write_csv(path: pathlib.Path, rows: list[dict], *, keep_bidi: bool = False) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS + ["engine", "model"])
        w.writeheader()
        for r in rows:
            r = dict(r, engine="e", model="m")
            if keep_bidi:
                r["norm_predicted_text"] = "\u202b" + r["norm_predicted_text"]
            w.writerow(r)


def test_lb_check_csv(tmp_path):
    rows = _rows([("שלום עולם", "שלום עולם"), ("מה נשמע היום", "מה נשמע")])
    good = tmp_path / "good.csv"
    _write_csv(good, rows)
    rep = lb.check_csv(good)
    assert rep["pass"] and rep["norm_text_mismatch_rows"] == 0 and rep["edits_mismatch_rows"] == 0
    assert rep["ours_corpus_wer"] == pytest.approx(1 / 5)
    stale = tmp_path / "stale.csv"
    bad_rows = [dict(r) for r in rows]
    bad_rows[0]["wer"] = 0.5   # the CSV's own number, computed with bidi marks kept
    _write_csv(stale, bad_rows, keep_bidi=True)
    rep2 = lb.check_csv(stale)
    assert rep2["norm_bidi_only_rows"] == 2 and rep2["stale_normalizer_in_csv"] and rep2["pass"]
    wrong = tmp_path / "wrong.csv"
    w_rows = [dict(r) for r in rows]
    w_rows[1]["norm_reference_text"] = "משהו אחר לגמרי"
    _write_csv(wrong, w_rows)
    assert not lb.check_csv(wrong)["pass"]


def test_timestamp_scorer(tmp_path):
    synth = tmp_path / "synth"
    synth.mkdir()
    refs = [{"id": "a", "s": 1000, "e": 3000, "spk": "A", "text": "x"}, {"id": "b", "s": 4000, "e": 6000, "spk": "B", "text": "y"}]
    (synth / "reference.jsonl").write_text("\n".join(json.dumps(r) for r in refs), encoding="utf-8")
    (synth / "manifest.json").write_text(json.dumps({"trim": {"precision_floor_ms": 30}, "overlap": "none"}), encoding="utf-8")
    doc = {"words": [{"s": 1040, "e": 1500, "t": "א", "spk": "S1"}, {"s": 1500, "e": 2900, "t": "ב", "spk": "S1"},
                     {"s": 4200, "e": 5000, "t": "ג", "spk": "S2"}, {"s": 5000, "e": 6100, "t": "ד", "spk": "S2"},
                     {"s": 9000, "e": 9500, "t": "ה", "spk": "S2"}],
           "turns": [{"s": 1040, "e": 2900, "spk": "S1"}, {"s": 4200, "e": 6100, "spk": "S2"}, {"s": 9000, "e": 9500, "spk": "S2"}]}
    out = timestamps.score(doc, synth)
    assert out["clips"] == 2 and out["words"] == 5 and out["words_outside_clips"] == 1 and out["empty_clips"] == 0
    assert out["onset"]["n"] == 2 and out["onset"]["mean_ms"] == pytest.approx((40 + 200) / 2)
    assert out["onset"]["within_50ms"] == 0.5 and out["onset"]["within_200ms"] == 1.0
    assert out["offset"]["within_50ms"] == 0.0 and out["offset"]["within_100ms"] == 1.0
    assert out["speakers"]["word_speaker_agreement"] == 1.0 and out["speakers"]["ref_speakers"] == 2


def test_knesset_silver_reference(tmp_path):
    from tamlul.eval import knesset
    d = tmp_path / "123"
    d.mkdir()
    protocol = "היו\"ר: בוקר טוב לכולם. אנחנו פותחים.\nחבר הכנסת לוי: תודה רבה, אדוני היושב ראש."
    (d / "raw.protocol.txt").write_text(protocol, encoding="utf-8")
    cut = protocol.index("חבר הכנסת")
    (d / "speakers.segments.txt").write_text(json.dumps([{"start_char": 0, "end_char": cut, "speaker_ids": [1]},
                                                          {"start_char": cut, "end_char": len(protocol), "speaker_ids": [2]}]),
                                             encoding="utf-8")
    toks = protocol.split()
    words = [{"word": t, "start": 0.5 * k, "end": 0.5 * k + 0.4, "probability": 0.9} for k, t in enumerate(toks)]
    (d / "transcript.aligned.json").write_text(json.dumps({"segments": [{"start": 0, "end": 6, "text": protocol, "words": words}]},
                                                          ensure_ascii=False), encoding="utf-8")
    (d / "metadata.json").write_text(json.dumps({"per_segment_quality_scores": [{"start": 0, "end": 100, "probability": 0.95}]}),
                                     encoding="utf-8")
    ref = knesset.to_reference(knesset.load_session(d))
    assert ref["speakers"] == ["1", "2"] and ref["words_aligned"] == len(toks)
    assert [t["spk"] for t in ref["turns"]] == ["1", "2"]
    assert ref["turns"][1]["s"] == int(0.5 * toks.index("חבר") * 1000)
    doc = {"turns": [{"s": 0, "e": 3000, "spk": "S1", "text": "בוקר טוב לכולם אנחנו פותחים"},
                     {"s": 3000, "e": 6000, "spk": "S2", "text": "תודה רבה אדוני היושב ראש"}], "words": []}
    sc = knesset.score(doc, ref)
    assert sc["speaker_count"]["exact"] and 0 < sc["cpwer_silver"] < 0.3   # "היו\"ר:" / "חבר הכנסת לוי:" are protocol-only tokens
