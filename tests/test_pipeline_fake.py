"""End-to-end through the CLI with the deterministic fake backend: no model, no download."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

pytest.importorskip("faster_whisper", reason="the plan stage needs faster-whisper's bundled Silero VAD")

from tamlul.cli import main
from tamlul.jobs import RunDir
from tamlul.paths import DataLayout


def _run(argv):
    return main(argv)


def test_transcribe_fake_end_to_end(data_dir, synthetic_wav, monkeypatch, capsys):
    monkeypatch.setenv("TAMLUL_FAKE_ASR", "1")
    rc = _run(["--data", str(data_dir), "transcribe", str(synthetic_wav), "--diarization", "off", "--profile", "meeting",
               "--formats", "md,txt,srt,vtt,csv,html,rttm"])
    assert rc == 0
    out = capsys.readouterr().out
    run_id = next(ln for ln in out.splitlines() if ln.startswith("run: "))[5:]
    rd = RunDir.open(DataLayout(data_dir), run_id)
    doc = rd.read_transcript()
    assert doc["schema"] == "tamlul/annotation/1.0"
    assert doc["words"] and doc["segments"] and doc["utterances"] and doc["turns"]
    assert all(isinstance(w["s"], int) and w["e"] >= w["s"] for w in doc["words"])
    assert doc["pipeline"]["asr"]["engine"] == "fake"
    assert doc["pipeline"]["decode"]["condition_on_previous_text"] is False
    assert doc["pipeline"]["decode"]["compression_ratio_threshold"] >= 2.8
    assert doc["stats"]["thresholds"]["pause_ms"] == 1000
    assert rd.chunks_json.is_file() and rd.segments_jsonl.is_file()
    for ext in ("md", "txt", "srt", "vtt", "csv", "html", "rttm"):
        assert (rd.exports / f"synthetic.{ext}").is_file(), ext
    srt = (rd.exports / "synthetic.srt").read_bytes()
    assert b"\r\n" in srt and not srt.startswith(b"\xef\xbb\xbf")
    assert "\u200F" in (rd.exports / "synthetic.txt").read_text(encoding="utf-8")
    # validate + progress + runs list through the CLI
    assert _run(["--data", str(data_dir), "validate", run_id]) == 0
    assert _run(["--data", str(data_dir), "progress", run_id]) == 0
    assert _run(["--data", str(data_dir), "runs"]) == 0
    prog = json.loads(rd.progress_json.read_text(encoding="utf-8"))
    assert prog["stage"] == "done"


def test_resume_refuses_different_options(data_dir, synthetic_wav, monkeypatch, capsys):
    monkeypatch.setenv("TAMLUL_FAKE_ASR", "1")
    assert _run(["--data", str(data_dir), "transcribe", str(synthetic_wav), "--diarization", "off"]) == 0
    run_id = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("run: "))[5:]
    # same options: resume is a no-op that succeeds
    assert _run(["--data", str(data_dir), "transcribe", str(synthetic_wav), "--diarization", "off", "--resume", run_id]) == 0
    # different compute type: must refuse (invariant 18)
    with pytest.raises(RuntimeError):
        _run(["--data", str(data_dir), "transcribe", str(synthetic_wav), "--diarization", "off", "--resume", run_id,
              "--compute-type", "float32"])


def test_run_dir_is_under_data_not_beside_media(data_dir, synthetic_wav, monkeypatch, capsys):
    monkeypatch.setenv("TAMLUL_FAKE_ASR", "1")
    assert _run(["--data", str(data_dir), "transcribe", str(synthetic_wav), "--diarization", "off"]) == 0
    assert not list(synthetic_wav.parent.glob("runs*"))
    assert list((pathlib.Path(data_dir) / "runs").iterdir())


def test_interval_exports_tolerate_overlapping_words(data_dir, synthetic_wav, monkeypatch, capsys):
    """Whisper can emit a zero-length word on the previous word's end; Praat/ELAN tiers must stay valid."""
    pytest.importorskip("praatio")
    pytest.importorskip("pympi")
    from tamlul.export import to_eaf, to_textgrid
    monkeypatch.setenv("TAMLUL_FAKE_ASR", "1")
    assert _run(["--data", str(data_dir), "transcribe", str(synthetic_wav), "--diarization", "off", "--formats", "md"]) == 0
    run_id = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("run: "))[5:]
    rd = RunDir.open(DataLayout(data_dir), run_id)
    doc = rd.read_transcript()
    w0, w1 = doc["words"][0], doc["words"][1]
    w1["s"] = w0["s"]            # overlapping starts
    w1["e"] = w0["s"] + 1        # and a 1-ms word
    tg = to_textgrid(doc, rd.exports)
    assert tg and "words" in tg
    eaf = to_eaf(doc, doc["media"]["path"], rd.exports)
    assert eaf and "-words" in eaf


def test_snap_to_speech_pulls_boundaries_out_of_silence():
    from tamlul.plan import snap_to_speech
    raw = [[1.0, 2.0], [3.0, 4.0], [6.0, 7.0]]           # unpadded speech, seconds
    words = [{"s": 700, "e": 1300, "t": "a"},           # starts 300 ms inside the pad -> snaps to 1000
             {"s": 1300, "e": 2250, "t": "b"},          # ends 250 ms past the offset -> snaps to 2000
             {"s": 2500, "e": 2700, "t": "c"},          # in silence: untouched
             {"s": 1900, "e": 3500, "t": "d"},          # starts 100 ms inside the previous tail, spans the 1 s gap -> 3000
             {"s": 3900, "e": 4100, "t": "e"},          # ends 100 ms past: snaps to 4000
             {"s": 6200, "e": 6600, "t": "f"},          # first word after the gap, 200 ms late -> 6000
             {"s": 6600, "e": 6900, "t": "g"}]          # not first after the gap: untouched
    moved = snap_to_speech(words, raw)
    assert moved == 5
    assert (words[0]["s"], words[0]["snap"]) == (1000, "s")
    assert (words[1]["e"], words[1]["snap"]) == (2000, "e")
    assert "snap" not in words[2] and "snap" not in words[6]
    assert (words[3]["s"], words[3]["snap"]) == (3000, "s")
    assert (words[4]["e"], words[4]["snap"]) == (4000, "e")
    assert (words[5]["s"], words[5]["snap"]) == (6000, "s")
    # a word that genuinely starts well inside the previous region is left alone (tail tolerance is 300 ms)
    words = [{"s": 1500, "e": 3500, "t": "x"}]
    assert snap_to_speech(words, raw) == 0 and "snap" not in words[0]


def test_decode_clips_group_mode_hands_the_whole_slice_to_the_backend():
    """Phase-4 ablation: per-region clip_timestamps cost +1.8 WER points and
    2.5x wall time on eval-whatsapp, so a group is decoded as one stretch; ``regions`` stays
    available to reproduce the old behaviour, and ``clips`` remain the speech regions."""
    import pytest

    from tamlul.plan import CLIP_MODE, Group, decode_clips

    g = Group(index=0, start_s=9.7, end_s=31.3, clips=[[10.0, 20.0], [21.0, 31.0]], part=0)
    assert CLIP_MODE == "group"
    assert decode_clips(g) is None
    got = decode_clips(g, "regions")
    assert [round(a, 3) for c in got for a in c] == [0.3, 10.3, 11.3, 21.3]
    assert g.clips == [[10.0, 20.0], [21.0, 31.0]]
    with pytest.raises(ValueError):
        decode_clips(g, "chunks")


def test_merge_fragments_restores_geresh_words_and_numbers():
    from tamlul.asr.base import RawWord, merge_fragments

    w = lambda s, e, t, p=0.9: RawWord(s=s, e=e, t=t, p=p)  # noqa: E731
    words = [w(0.0, 0.2, " צ", 0.8), w(0.2, 0.5, "'יפס", 0.6), w(0.5, 0.7, " של"), w(0.7, 0.8, " 1"), w(0.8, 0.9, ",000"),
             w(0.9, 1.2, " ג"), w(1.2, 1.5, "'ירפה")]
    got = merge_fragments(words)
    assert [x.t for x in got] == [" צ'יפס", " של", " 1,000", " ג'ירפה"]
    assert (got[0].s, got[0].e, got[0].p) == (0.0, 0.5, 0.6)
    assert "".join(x.t for x in got) == "".join(x.t for x in words)
    assert merge_fragments([]) == []


def test_uncovered_speech_finds_skipped_window_tails():
    from tamlul.plan import Group, uncovered_speech

    raw = [[1.0, 3.0], [3.5, 8.0], [8.3, 8.6], [20.0, 25.0], [26.0, 30.0], [40.0, 45.0]]
    g = Group(index=0, start_s=0.7, end_s=45.3, clips=[[1.0, 8.6], [20.0, 30.0], [40.0, 45.0]])
    segs = [(0.9, 3.1), (3.4, 5.0), (40.0, 45.0)]                   # the decoder stopped at 5.0 and resumed at 40
    got = uncovered_speech(raw, segs, g, min_s=1.0)
    # 3.5-8.0 is only 30 % covered -> missed; 8.3-8.6 is too short; 20-25 and 26-30 merge (gap < 2 s)
    assert [(round(a, 1), round(b, 1)) for a, b in got] == [(3.5, 8.0), (20.0, 30.0)]
    assert uncovered_speech(raw, [(0.0, 50.0)], g, min_s=1.0) == []
    assert uncovered_speech(raw, segs, g, min_s=2.5) == [(3.5, 8.0), (20.0, 30.0)]


def test_uncovered_speech_ignores_the_neighbouring_groups_speech():
    """Raw regions straddling a group boundary must not be read as skipped speech."""
    from tamlul.plan import Group, uncovered_speech

    raw = [[91.0, 100.0], [100.5, 120.0]]
    # padded slices as the pre-fix grouping produced them (overlapping by 0.6 s): the pre-fix check,
    # which only asked whether a raw region intersected the slice, flagged both regions
    g1 = Group(index=1, start_s=9.7, end_s=100.8, clips=[[10.0, 30.0], [90.0, 100.2]])
    g2 = Group(index=2, start_s=99.9, end_s=250.3, clips=[[100.2, 130.0], [220.0, 250.0]])
    assert uncovered_speech(raw, [(10.0, 100.1)], g1, min_s=2.5) == []      # 100.5-120.0 is g2's speech
    assert uncovered_speech(raw, [(100.2, 250.0)], g2, min_s=2.5) == []     # 91.0-100.0 is g1's speech
    # the part of a straddling region inside the group is still checked
    assert uncovered_speech([[95.0, 120.0]], [(10.0, 94.0)], g1, min_s=2.5) == [(95.0, 100.2)]


def test_long_groups_are_cut_at_a_pause_and_slices_never_overlap():
    from tamlul.plan import group_regions

    # continuous speech in 30-s Silero regions, one 0.4-s pause at 200 s (< two pads), contiguous (forced) cuts elsewhere
    regions = [[0.0, 30.0], [30.0, 60.0], [60.0, 90.0], [90.0, 120.0], [120.0, 150.0], [150.0, 180.0],
               [180.0, 199.6], [200.0, 230.0], [230.0, 260.0], [260.0, 290.0], [290.0, 310.0], [310.0, 334.0]]
    groups, parts = group_regions(regions, 400.0, group_max_s=300.0)
    assert parts == 1 and len(groups) == 2
    assert groups[0].clips[-1] == [180.0, 199.6] and groups[1].clips[0] == [200.0, 230.0]   # cut at the pause
    for a, b in zip(groups, groups[1:]):
        assert a.end_s <= b.start_s
    assert groups[0].end_s == groups[1].start_s == 199.8                    # pads meet at the midpoint
    # no pause in the second half at all: forced cut where the span runs out, still no overlap
    flat = [[float(k * 30), float(k * 30 + 30)] for k in range(12)]
    groups, _ = group_regions(flat, 400.0, group_max_s=300.0)
    assert len(groups) == 2 and groups[0].clips[-1] == [270.0, 300.0] and groups[0].end_s == groups[1].start_s == 300.0
    # ordinary silences still split with full pads
    groups, _ = group_regions([[1.0, 5.0], [9.0, 12.0]], 20.0)
    assert [(g.start_s, g.end_s) for g in groups] == [(0.7, 5.3), (8.7, 12.3)]


def test_group_cut_prefers_the_current_gap_and_ignores_first_half_pauses():
    from tamlul.plan import group_regions

    # the gap right here (1.5 s) is wider than any pause in the second half (0.3 s): cut here
    regions = [[0.0, 100.0], [100.0, 200.0], [200.3, 290.0], [291.5, 320.0]]
    groups, _ = group_regions(regions, 400.0, group_max_s=300.0)
    assert [g.clips[-1][1] for g in groups] == [290.0, 320.0]
    # a wide pause in the FIRST half is not used (it would leave a tiny first group and a huge second one)
    regions = [[0.0, 50.0], [51.8, 150.0], [150.0, 200.0], [200.4, 299.0], [299.0, 330.0]]
    groups, _ = group_regions(regions, 400.0, group_max_s=300.0)
    assert groups[0].clips[-1] == [150.0, 200.0] and groups[1].clips[0] == [200.4, 299.0]


def test_recovered_words_already_decoded_are_not_spliced_in_twice():
    from tamlul.asr.base import RawSegment, RawWord
    from tamlul.asr.decode import DecodeOpts
    from tamlul.asr.recover import recover_missed, trim_to_uncovered
    from tamlul.plan import Group

    words = [RawWord(s=0.3 + k, e=0.3 + k + 0.8, t=f" מילה{k}", p=0.9) for k in range(5)]   # 0.3-5.1 s of the slice
    seg = RawSegment(s=0.3, e=5.1, text="".join(w.t for w in words), words=words, avg_logprob=-0.2,
                     no_speech_prob=0.01, compression_ratio=1.3, temperature=0.0)
    covered = [(3.0, 5.0)]                       # the group already decoded file time 3.0-5.0
    got = trim_to_uncovered(seg, 3.2, covered)   # the slice starts at 3.2 s: words at 3.5..8.3 s file time
    assert [w.t for w in got.words] == [" מילה2", " מילה3", " מילה4"] and got.text == " מילה2 מילה3 מילה4"
    assert (got.s, got.e) == (2.3, 5.1)
    assert trim_to_uncovered(seg, 3.2, [(0.0, 100.0)]) is None
    assert trim_to_uncovered(seg, 3.2, []) is seg

    class Stub:
        def transcribe(self, audio, clips, opts):
            return [seg]

    g = Group(index=0, start_s=0.0, end_s=20.0, clips=[[0.5, 19.5]])
    extra = recover_missed(Stub(), np.zeros(16000 * 20, dtype=np.float32), g, [(RawSegment(s=0.5, e=5.0, text=" x"), 0.0)],
                           [[0.5, 5.0], [3.5, 12.0]], DecodeOpts())
    assert len(extra) == 1 and extra[0][1] == 3.2 and extra[0][0].text == " מילה2 מילה3 מילה4"


def test_doctor_reports_a_broken_diarization_stack(data_dir, monkeypatch):
    from tamlul import doctor
    from tamlul.paths import DataLayout

    layout = DataLayout(data_dir)
    monkeypatch.setattr(doctor, "diarization_probe", lambda: {
        "ok": False, "steps": ["torch"],
        "error": "ImportError: DLL load failed while importing _x: An Application Control policy has blocked this file."})
    rep = doctor.doctor_report(layout)
    assert "BROKEN -> transcripts will have no speaker layer" in rep
    assert "Application Control" in rep and "Smart App Control" in rep
    monkeypatch.setattr(doctor, "diarization_probe", lambda: {"ok": True, "steps": [], "seconds": 3.2})
    assert "diarization stack (imported in the child process): ok (3.2 s)" in doctor.doctor_report(layout)


def test_recovered_segments_must_be_plausible():
    from tamlul.asr.base import RawSegment
    from tamlul.asr.decode import DecodeOpts
    from tamlul.asr.recover import plausible

    o = DecodeOpts()

    def seg(text, lp=-0.3, nsp=0.05, cr=1.4):
        return RawSegment(s=0.0, e=2.0, text=text, words=[], avg_logprob=lp, no_speech_prob=nsp,
                          compression_ratio=cr, temperature=0.0)

    assert plausible(seg(" מעבר לאירוע של יום רביעי"), o)
    assert not plausible(seg(" כן אהההההההההההה"), o)           # one-letter loop
    assert not plausible(seg(" זה זה זה זה זה"), o)               # n-gram loop
    assert not plausible(seg(" מילים", nsp=0.9), o)              # no-speech
    assert not plausible(seg(" מילים", lp=-1.5), o)              # low confidence
    assert not plausible(seg(" מילים", cr=3.0), o)               # compression ratio
    assert not plausible(seg(""), o)


def test_long_turns_are_exported_as_timestamped_paragraphs():
    """A long monologue turn is exported as timestamped paragraphs, not one line under one timestamp."""
    from tamlul.export import PARA_MS, paragraphs, to_html, to_md, to_txt

    words, utts = [], []
    t_ms = 0
    for k in range(12):                        # 12 sentences of 3 words, 8 s each; a 3-s pause before sentence 9
        if k == 9:
            t_ms += 3000
        i0 = len(words)
        for j in range(3):
            words.append({"i": len(words), "s": t_ms + j * 2500, "e": t_ms + j * 2500 + 2000,
                          "t": f"מילה{k}{j}" + ("." if j == 2 else ""), "p": 0.9, "spk": "S1"})
        utts.append({"id": k, "s": t_ms, "e": t_ms + 7000, "spk": "S1", "w": [i0, len(words) - 1], "text": ""})
        t_ms += 8000
    words.insert(15, {"i": 15, "s": 41000, "e": 41300, "t": "כן", "p": 0.9, "spk": "S2"})   # a backchannel inside the turn
    for i, w in enumerate(words):
        w["i"] = i
    for u in utts:
        u["w"] = [u["w"][0] + (1 if u["w"][0] >= 15 else 0), u["w"][1] + (1 if u["w"][1] >= 15 else 0)]
    text = " ".join(w["t"] for w in words)
    turn = {"id": 0, "spk": "S1", "s": 0, "e": utts[-1]["e"], "w": [0, len(words) - 1], "utts": list(range(12)), "text": text}
    doc = {"media": {"path": "C:/x/sample.mp4"}, "words": words, "utterances": utts, "turns": [turn],
           "speakers": [{"id": "S1", "label": "דובר 1", "name": None, "stats": {}}], "annotations": {}}
    paras = paragraphs(doc, turn)
    assert " ".join(p["text"] for p in paras) == text                       # nothing dropped, backchannel kept
    assert [p["s"] for p in paras] == [0, 32000, 64000, 75000]                # 30-s rule twice, then the 3-s pause
    assert all(b["w"][0] == a["w"][1] + 1 for a, b in zip(paras, paras[1:]))
    assert paras[0]["s"] == turn["s"] and paras[-1]["e"] == turn["e"] and PARA_MS == 30_000
    txt = to_txt(doc).splitlines()
    assert len(txt) == 4 and "[00:00:32]" in txt[1] and "[00:01:04]" in txt[2] and "[00:01:15]" in txt[3]
    assert to_md(doc).count("[00:00:32]") == 1 and to_html(doc).count('class="para"') == 4
    # a turn without utterances still exports as one paragraph
    bare = dict(turn, utts=[])
    assert paragraphs(doc, bare) == [{"s": 0, "e": turn["e"], "w": turn["w"], "text": text}]


def test_stereo_recovery_and_snap_use_only_the_groups_own_channel():
    """Review of 2026-09-26: the stereo plan's raw regions were the union of both channels."""
    from tamlul.plan import Group, Plan, snap_words, uncovered_speech

    ch0 = [[0.0, 10.0], [11.9, 20.0], [90.3, 99.8], [100.2, 110.0]]
    ch1 = [[9.6, 12.6], [101.0, 104.0]]
    plan = Plan(source="silero+rms/stereo", params={}, duration_s=120.0, speech_s=0.0, regions=[], gated=[],
                raw_regions=sorted(ch0 + ch1), raw_by_channel=[ch0, ch1])
    assert plan.raw_for(0) is ch0 and plan.raw_for(1) is ch1 and plan.raw_for(-1) == sorted(ch0 + ch1)
    # channel 1 speaks in channel 0's pause: not a miss for channel 0
    g = Group(index=0, start_s=0.0, end_s=20.3, clips=[[0.0, 10.0], [11.9, 20.0]], channel=0)
    assert uncovered_speech(plan.raw_for(0), [(0.3, 9.7), (12.2, 19.7)], g, min_s=2.5) == []
    assert uncovered_speech(plan.raw_regions, [(0.3, 9.7), (12.2, 19.7)], g, min_s=2.5) == [(9.6, 12.6)]   # the old input
    # a genuine channel-0 miss is not shortened by channel 1's nested region, even with the union as input
    g2 = Group(index=1, start_s=90.0, end_s=110.3, clips=[[90.3, 99.8], [100.2, 110.0]], channel=0)
    assert uncovered_speech(plan.raw_for(0), [(90.0, 100.0)], g2, min_s=2.5) == [(100.2, 110.0)]
    assert uncovered_speech(plan.raw_regions, [(90.0, 100.0)], g2, min_s=2.5) == [(100.2, 110.0)]
    # snapping: a channel-0 word after a pause goes to channel 0's onset, not to channel 1's
    snap_plan = Plan(source="s", params={}, duration_s=10.0, speech_s=0.0, regions=[], gated=[],
                     raw_regions=[[1.0, 2.0], [3.0, 4.0], [3.2, 4.0]], raw_by_channel=[[[1.0, 2.0], [3.2, 4.0]], [[3.0, 4.0]]])
    words = [{"s": 1000, "e": 2000, "t": "a", "channel": 0}, {"s": 2700, "e": 3500, "t": "b", "channel": 0}]
    assert snap_words(words, snap_plan) == 1 and words[1]["s"] == 3200
    mono = Plan(source="s", params={}, duration_s=10.0, speech_s=0.0, regions=[], gated=[], raw_regions=[[1.0, 2.0], [3.2, 4.0]])
    words = [{"s": 1000, "e": 2000, "t": "a"}, {"s": 2700, "e": 3500, "t": "b"}]
    assert snap_words(words, mono) == 1 and words[1]["s"] == 3200
    assert Plan.from_json(snap_plan.to_json()).raw_by_channel == snap_plan.raw_by_channel


def test_hallucinations_are_flagged_and_blanked_from_the_exports():
    """Knesset boilerplate emitted on non-speech (VAD fraction 0) is flagged and blanked from every export."""
    from tamlul.document import apply_clean_flags
    from tamlul.export import to_srt, to_txt, to_vtt
    from tamlul.text.clean import credit_phrase_suspect

    knesset = "גברתי היושבת-ראש, חברות וחברי הכנסת,"
    assert credit_phrase_suspect(knesset, speech_fraction=0.0, no_speech_prob=1e-10)
    assert not credit_phrase_suspect(knesset, speech_fraction=0.9, no_speech_prob=1e-10)    # said aloud: kept
    assert credit_phrase_suspect("א ב ג ד ה", speech_fraction=0.0, no_speech_prob=1e-10, duration_ms=20, n_words=5)
    assert not credit_phrase_suspect("זהו זה", speech_fraction=0.0, no_speech_prob=1e-10, duration_ms=20, n_words=2)
    assert not credit_phrase_suspect("מילה אחת שתיים", speech_fraction=0.1, no_speech_prob=1e-10, duration_ms=900, n_words=3)

    words = [{"i": 0, "s": 1000, "e": 1500, "t": "זהו", "t_raw": " זהו", "p": 0.9, "spk": None},
             {"i": 1, "s": 1500, "e": 1900, "t": "סיימנו.", "t_raw": " סיימנו.", "p": 0.9, "spk": None}]
    words += [{"i": 2 + k, "s": 3500, "e": 3500 + (20 if k == 4 else 0), "t": w, "t_raw": " " + w, "p": 0.8, "spk": None}
              for k, w in enumerate(knesset.split())]
    asr = {"avg_logprob": -0.4, "no_speech_prob": 2e-10, "compression_ratio": 1.1, "temperature": 0.0}
    segs = [{"id": 0, "s": 1000, "e": 1900, "spk": None, "text": "זהו סיימנו.", "w": [0, 1], "flags": [], "asr": asr, "speech_fraction": 1.0},
            {"id": 1, "s": 3500, "e": 3520, "spk": None, "text": knesset, "w": [2, 6], "flags": [], "asr": asr, "speech_fraction": 0.0}]
    turn = {"id": 0, "spk": None, "s": 1000, "e": 3520, "w": [0, 6], "utts": [], "text": ""}
    doc = {"words": words, "segments": segs, "text_edits": [], "turns": [turn], "speakers": [], "utterances": [],
           "pipeline": {}, "media": {"path": "x.wav"}}
    apply_clean_flags(doc)
    assert segs[1]["flags"] == ["credit_phrase_suspect"] and segs[1]["text"] == "" and segs[0]["flags"] == []
    assert all(w["t"] == "" and w["t_raw"].strip() for w in words[2:]) and words[0]["t"] == "זהו"
    assert [e["kind"] for e in doc["text_edits"]] == ["hallucination_filter"] * 5
    turn["text"] = " ".join(w["t"] for w in words if w["t"])
    assert "הכנסת" not in to_txt(doc) and "זהו סיימנו." in to_txt(doc)
    srt = to_srt(doc)
    assert "הכנסת" not in srt and srt.count("-->") == 1 and srt.startswith("1\r\n")
    assert "הכנסת" not in to_vtt(doc)


def test_resume_keeps_the_recovered_segment_count(data_dir, synthetic_wav, monkeypatch, capsys):
    """A resume decodes only the groups left, so it must not reset asr.recovered_segments."""
    monkeypatch.setenv("TAMLUL_FAKE_ASR", "1")
    assert _run(["--data", str(data_dir), "transcribe", str(synthetic_wav), "--diarization", "off"]) == 0
    run_id = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("run: "))[5:]
    rd = RunDir.open(DataLayout(data_dir), run_id)
    run = rd.read_run()
    assert run["asr"]["recovered_segments"] == 0
    run["asr"]["recovered_segments"] = 5                     # as if the first pass had recovered five segments
    rd.write_run(run)
    assert _run(["--data", str(data_dir), "transcribe", "--diarization", "off", "--resume", run_id]) == 0
    assert rd.read_run()["asr"]["recovered_segments"] == 5
