"""Exports (design §4.11). All RTL-safe; store logical order, never reverse.

Conventions from the research: U+200F on lines that start with a timestamp,
digit or Latin word; LRI...PDI (U+2066/U+2069) around timestamps and Latin
runs in Markdown; SRT = CRLF, no BOM, U+200F per line; WebVTT has NO
``direction:`` cue setting - use U+200F + ``align:start`` + ``<v Name>``;
ELAN's SRT import merges speakers into one tier, so EAF is native; Praat tiers
cannot overlap, so one IntervalTier per speaker; speaker labels are Hebrew
(``דובר N``) because a Latin prefix flips the base direction.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import pathlib
import re
import shutil
from collections.abc import Callable

from ..document import hms, speaker_name
from ..jobs import RunDir
from ..paths import DataLayout

RLM = "\u200F"
LRI, PDI = "\u2066", "\u2069"
ALL_FORMATS = ("json", "md", "txt", "srt", "vtt", "csv", "html", "rttm", "eaf", "textgrid")
_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z0-9._\-']*(?:\s+[A-Za-z][A-Za-z0-9._\-']*)*")


def isolate_latin(text: str) -> str:
    return _LATIN_RUN.sub(lambda m: f"{LRI}{m.group(0)}{PDI}", text)


def _starts_ltr(line: str) -> bool:
    return bool(line) and (line[0].isdigit() or line[0] in "[(" or ("a" <= line[0].lower() <= "z"))


def rtl_line(line: str) -> str:
    return RLM + line if _starts_ltr(line) else line


def _name(doc: dict, spk: str | None) -> str:
    return speaker_name(doc, spk) if spk else ""


PARA_MS = 30_000        # a turn gets a new timestamped paragraph at the first utterance boundary 30 s in ...
PARA_PAUSE_MS = 2_000   # ... or at any pause of 2 s or more


def paragraphs(doc: dict, t: dict) -> list[dict]:
    """Split a turn into timestamped paragraphs of whole utterances: ``{s, e, w: [i0, i1], text}``.

    A monologue turn can run for many minutes; one timestamp per turn makes the TXT/MD/HTML exports
    unusable for finding anything (one line of tens of thousands of characters under a single
    [hh:mm:ss]). Paragraphs cover the turn's word range contiguously, so their texts joined
    with spaces are exactly the turn's text.
    """
    words = doc["words"]
    utts = doc.get("utterances") or []
    ids = [i for i in (t.get("utts") or []) if 0 <= i < len(utts)]
    spans: list[list[int]] = []          # [s, e, w0, w1]
    for uid in ids:
        u = utts[uid]
        if spans and (u["s"] - spans[-1][0] >= PARA_MS or u["s"] - spans[-1][1] >= PARA_PAUSE_MS):
            spans.append([u["s"], u["e"], u["w"][0], u["w"][1]])
        elif spans:
            spans[-1][1] = max(spans[-1][1], u["e"])
            spans[-1][3] = u["w"][1]
        else:
            spans.append([u["s"], u["e"], u["w"][0], u["w"][1]])
    if not spans:
        return [{"s": t["s"], "e": t["e"], "w": list(t["w"]), "text": t["text"]}] if t["text"].strip() else []
    spans[0][0], spans[0][2] = t["s"], t["w"][0]
    spans[-1][1], spans[-1][3] = t["e"], t["w"][1]
    for a, b in zip(spans, spans[1:]):   # contiguous word ranges: nothing between paragraphs is dropped
        b[2] = a[3] + 1
    out = []
    for s_ms, e_ms, i0, i1 in spans:
        text = " ".join(words[i]["t"] for i in range(i0, i1 + 1) if words[i]["t"])
        if text:
            out.append({"s": s_ms, "e": e_ms, "w": [i0, i1], "text": text})
    if out:
        return out
    return [{"s": t["s"], "e": t["e"], "w": list(t["w"]), "text": t["text"]}] if t["text"].strip() else []


# ------------------------------------------------------------------ writers

def to_md(doc: dict) -> str:
    out = [f"# {LRI}{pathlib.Path(doc['media']['path']).name}{PDI}", ""]
    ann = doc.get("annotations") or {}
    if ann.get("summary", {}).get("text"):
        out += ["## סיכום", "", ann["summary"]["text"], ""]
    if ann.get("decisions"):
        out += ["## החלטות", ""] + [f"- {d['text']}" for d in ann["decisions"]] + [""]
    if ann.get("action_items"):
        out += ["## משימות", "", "| משימה | אחראי | מועד | סטטוס |", "|---|---|---|---|"]
        out += [f"| {a['text']} | {_name(doc, a.get('owner')) or a.get('owner_name','')} | {a.get('due','')} | {a.get('status','open')} |"
                for a in ann["action_items"]] + [""]
    if ann.get("questions"):
        out += ["## שאלות", ""] + [f"- {q['text']}" + (" (נענתה)" if q.get("answered") else "") for q in ann["questions"]] + [""]
    if ann.get("key_quotes"):
        out += ["## ציטוטים", ""] + [f"> {q['text']}" for q in ann["key_quotes"]] + [""]
    if doc["speakers"]:
        out += ["## דוברים", "", "| דובר | חלק | תורות | מילים |", "|---|---|---|---|"]
        for s in doc["speakers"]:
            st = s.get("stats", {})
            out.append(f"| {s.get('name') or s['label']} | {st.get('share', 0):.0%} | {st.get('turns', 0)} | {st.get('words', 0)} |")
        out.append("")
    out += ["## תמלול", ""]
    for t in doc["turns"]:
        name = _name(doc, t["spk"])
        paras = paragraphs(doc, t)
        if name and paras:
            out += [rtl_line(f"**{name}**"), ""]
        for para in paras:
            out += [rtl_line(f"{LRI}[{hms(para['s'])}]{PDI} {isolate_latin(para['text'])}"), ""]
    return "\n".join(out)


def to_txt(doc: dict) -> str:
    """QDA flavour: [hh:mm:ss] Name: text, one line per paragraph (``paragraphs``); names <= 63 chars."""
    lines = []
    for t in doc["turns"]:
        name = (_name(doc, t["spk"]) or "")[:63]
        for para in paragraphs(doc, t):
            lines.append(rtl_line(f"[{hms(para['s'])}] {name + ': ' if name else ''}{para['text']}"))
    return "\n".join(lines) + "\n"


def _srt_time(ms: int) -> str:
    return hms(ms, with_ms=True).replace(".", ",")


def to_srt(doc: dict) -> str:
    out = []
    for k, seg in enumerate((x for x in doc["segments"] if x["text"].strip()), 1):   # blanked hallucinations skip
        name = _name(doc, seg.get("spk"))
        text = f"{name}: {seg['text']}" if name else seg["text"]
        out += [str(k), f"{_srt_time(seg['s'])} --> {_srt_time(seg['e'])}", RLM + text, ""]
    return "\r\n".join(out)


def to_vtt(doc: dict) -> str:
    p = doc["pipeline"]
    out = ["WEBVTT", "", f"NOTE tamlul {doc.get('plugin_version')} {p.get('asr', {}).get('model')} rev {p.get('asr', {}).get('revision')}", ""]
    for seg in doc["segments"]:
        if not seg["text"].strip():
            continue
        name = _name(doc, seg.get("spk"))
        text = f"<v {name}>{seg['text']}" if name else seg["text"]
        out += [f"{hms(seg['s'], with_ms=True)} --> {hms(seg['e'], with_ms=True)} align:start", RLM + text, ""]
    return "\n".join(out)


def to_csv(doc: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["id", "start_ms", "end_ms", "speaker_id", "speaker_name", "text", "asr_conf", "flags"])
    for seg in doc["segments"]:
        conf = seg.get("asr", {}).get("avg_logprob")
        w.writerow([seg["id"], seg["s"], seg["e"], seg.get("spk") or "", _name(doc, seg.get("spk")), seg["text"],
                    round(2.718281828 ** conf, 3) if conf is not None else "", ";".join(seg.get("flags", []))])
    return buf.getvalue()


def to_rttm(doc: dict, *, file_id: str) -> str:
    lines = []
    for t in doc["turns"]:
        if t["spk"] is None:
            continue
        lines.append(f"SPEAKER {file_id} 1 {t['s']/1000:.3f} {(t['e']-t['s'])/1000:.3f} <NA> <NA> {t['spk']} <NA> <NA>")
    return "\n".join(lines) + ("\n" if lines else "")


def _media_uri(path: str) -> str:
    p = pathlib.Path(path)
    try:
        return p.resolve().as_uri()
    except (ValueError, OSError):
        return path


def to_html(doc: dict) -> str:
    title = html.escape(pathlib.Path(doc["media"]["path"]).name)
    parts = [f"<!DOCTYPE html><html lang=\"he\" dir=\"rtl\"><head><meta charset=\"utf-8\"><title>{title}</title>",
             "<style>body{font-family:system-ui,'Segoe UI',Arial;max-width:900px;margin:2rem auto;padding:0 1rem;line-height:1.7}"
             ".turn{margin:1rem 0}.para{margin:.3rem 0}.spk{font-weight:600}.ts{color:#666;font-size:.85em;cursor:pointer;unicode-bidi:isolate}"
             ".w{cursor:pointer}.w:hover{background:#ffe9a8}.low{color:#a00}.ov{text-decoration:underline dotted}"
             "audio{width:100%;position:sticky;top:0;background:#fff}</style></head><body>",
             f"<h1>{title}</h1>",
             f"<audio id=\"a\" controls src=\"{html.escape(_media_uri(doc['media']['path']))}\"></audio>",
             "<script>function go(ms){var a=document.getElementById('a');a.currentTime=ms/1000;a.play();}</script>"]
    words = doc["words"]
    for t in doc["turns"]:
        name = html.escape(_name(doc, t["spk"]))
        anchor = f"ts_{t['s']}_{t['e']}_{t['spk'] or 'S0'}"
        paras = []
        for para in paragraphs(doc, t):
            spans = []
            for i in range(para["w"][0], para["w"][1] + 1):
                w = words[i]
                if not w["t"]:
                    continue
                cls = "w" + (" low" if w["p"] < 0.3 else "") + (" ov" if w.get("overlap") else "")
                spans.append(f"<span class=\"{cls}\" onclick=\"go({w['s']})\" title=\"{w['s']/1000:.2f}s p={w['p']:.2f}\">{html.escape(w['t'])}</span>")
            paras.append(f"<p class=\"para\"><span class=\"ts\" onclick=\"go({para['s']})\"><bdi>[{hms(para['s'])}]</bdi></span> "
                         f"{' '.join(spans)}</p>")
        if paras:
            parts.append(f"<div class=\"turn\" id=\"{anchor}\"><div class=\"spk\">{name}</div>{''.join(paras)}</div>")
    parts.append("</body></html>")
    return "\n".join(parts)


def to_eaf(doc: dict, media_path: str, tmp_dir: pathlib.Path) -> str | None:
    try:
        import pympi
    except ImportError:
        return None
    eaf = pympi.Elan.Eaf()
    eaf.add_linked_file(pathlib.Path(media_path).as_uri(), relpath=None, mimetype=None)
    spk_ids = [s["id"] for s in doc["speakers"]] or ["S0"]
    for sid in spk_ids:
        eaf.add_tier(_name(doc, sid) if sid != "S0" else "transcript")
        eaf.add_tier(f"{sid}-words")
    for sid in spk_ids:
        turns = _monotone([(t["s"], t["e"], t["text"]) for t in doc["turns"] if (t["spk"] or "S0") == sid])
        for s, e, text in turns:
            eaf.add_annotation(_name(doc, sid) if sid != "S0" else "transcript", int(s * 1000), int(e * 1000), text)
        words = _monotone([(w["s"], w["e"], w["t"]) for w in doc["words"] if w["t"] and (w["spk"] or "S0") == sid])
        for s, e, text in words:
            eaf.add_annotation(f"{sid}-words", int(s * 1000), int(e * 1000), text)
    eaf.add_tier("events")
    for s, e, kind in _monotone([(e["s"], e["e"], e["kind"]) for e in doc["events"]
                                 if e["kind"] in ("overlap", "pause", "gap", "latch", "interruption")]):
        eaf.add_annotation("events", int(s * 1000), int(e * 1000), kind)
    tmp = tmp_dir / ".tmp.eaf"
    eaf.to_file(str(tmp))
    text = tmp.read_text(encoding="utf-8")
    tmp.unlink(missing_ok=True)
    return text


def _monotone(items: list[tuple[int, int, str]]) -> list[tuple[float, float, str]]:
    """A Praat/ELAN interval tier cannot overlap: Whisper occasionally emits a zero-length word on the
    previous word's end, so starts are pushed to the previous end and every interval keeps >= 1 ms."""
    out: list[tuple[float, float, str]] = []
    prev_end = 0
    for s, e, text in sorted(items, key=lambda x: (x[0], x[1])):
        s = max(s, prev_end)
        e = max(e, s + 1)
        out.append((s / 1000, e / 1000, text))
        prev_end = e
    return out


def to_textgrid(doc: dict, tmp_dir: pathlib.Path) -> str | None:
    try:
        from praatio import textgrid
    except ImportError:
        return None
    dur = (doc["stats"].get("duration_ms") or doc["media"].get("duration_ms") or 0) / 1000
    tg = textgrid.Textgrid()
    spk_ids = [s["id"] for s in doc["speakers"]] or ["S0"]
    for sid in spk_ids:
        ints = _monotone([(t["s"], t["e"], t["text"]) for t in doc["turns"] if (t["spk"] or "S0") == sid])
        tg.addTier(textgrid.IntervalTier(_name(doc, sid) if sid != "S0" else "transcript", ints, 0, max(dur, ints[-1][1] if ints else 0)))
    words = _monotone([(w["s"], w["e"], w["t"]) for w in doc["words"] if w["t"]])
    tg.addTier(textgrid.IntervalTier("words", words, 0, max(dur, words[-1][1] if words else 0)))
    buf_path = tmp_dir / ".tmp.TextGrid"
    tg.save(str(buf_path), format="long_textgrid", includeBlankSpaces=True)
    text = buf_path.read_text(encoding="utf-8")
    buf_path.unlink(missing_ok=True)
    return text


WRITERS: dict[str, Callable] = {
    "md": lambda doc, rd: to_md(doc),
    "txt": lambda doc, rd: to_txt(doc),
    "srt": lambda doc, rd: to_srt(doc),
    "vtt": lambda doc, rd: to_vtt(doc),
    "csv": lambda doc, rd: to_csv(doc),
    "html": lambda doc, rd: to_html(doc),
    "rttm": lambda doc, rd: to_rttm(doc, file_id=rd.id),
    "eaf": lambda doc, rd: to_eaf(doc, doc["media"]["path"], rd.exports),
    "textgrid": lambda doc, rd: to_textgrid(doc, rd.exports),
}
EXT = {"textgrid": "TextGrid"}


def write_exports(rd: RunDir, doc: dict, formats: list[str], log: Callable[[str], None] = lambda m: None) -> list[str]:
    rd.exports.mkdir(exist_ok=True)
    stem = pathlib.Path(doc["media"]["path"]).stem
    written = []
    for f in formats:
        f = f.lower().strip()
        if f == "json" or f not in WRITERS:
            continue
        text = WRITERS[f](doc, rd)
        if text is None:
            log(f"export {f}: skipped (optional dependency missing)")
            continue
        p = rd.exports / f"{stem}.{EXT.get(f, f)}"
        newline = "" if f == "srt" else None
        with p.open("w", encoding="utf-8", newline=newline) as fh:
            fh.write(text)
        written.append(str(p))
    return written


def rewrite_existing(rd: RunDir, doc: dict) -> list[str]:
    present = []
    for p in rd.exports.glob("*"):
        ext = p.suffix.lstrip(".").lower()
        if ext == "textgrid":
            present.append("textgrid")
        elif ext in WRITERS:
            present.append(ext)
    return write_exports(rd, doc, sorted(set(present))) if present else []


def export_command(args: argparse.Namespace, layout: DataLayout) -> int:
    rd = RunDir.open(layout, args.run)
    doc = rd.read_transcript()
    formats = [f for f in (args.formats or ",".join(ALL_FORMATS)).split(",") if f]
    written = write_exports(rd, doc, formats, log=lambda m: print(m))
    if getattr(args, "beside_media", False):
        dest = pathlib.Path(doc["media"]["path"]).parent
        for w in written:
            try:
                shutil.copy2(w, dest / pathlib.Path(w).name)
            except OSError as exc:
                print(f"could not copy {w} beside the media: {exc}")
    for w in written:
        print(w)
    return 0
