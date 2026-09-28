---
name: tamlul
description: Transcribe and annotate Hebrew conversations locally - meetings, interviews, phone calls, WhatsApp voice notes, podcasts - into speaker-attributed, timestamped, verbatim Hebrew transcripts with turns, overlaps, summaries and action items. Use when someone asks to transcribe (לתמלל, תמלול, תמלל) a Hebrew recording, to turn a recording (הקלטה) of a meeting (פגישה, ישיבה), interview (ראיון), call (שיחה) or voice note (הודעה קולית) into text, to identify who said what (מי אמר מה, דוברים), or to summarise a Hebrew recording. Routes to the transcribe, annotate, speakers and export skills.
---

# Tamlul (תמלול)

Local Hebrew conversation transcription in two halves. **Python measures. You judge.**

Everything countable - words, timestamps, who spoke when, overlaps, pauses,
talk share, confidence - is computed by the CLI into `transcript.json`. Audio
never leaves the machine (ivrit.ai Whisper + pyannote, on CPU or CUDA). You
read the JSON and produce what a measurement cannot: names, topics, summaries,
decisions, action items, restored English terms - each anchored to the words
it came from.

## The workflow

| Step | What | Skill |
|---|---|---|
| 0 | First run: environment, models, quick calibration | `/tamlul:doctor` (or let `transcribe` guide you) |
| 1 | Transcribe a recording (background job) | `/tamlul:transcribe` |
| 2 | Judgement layers with evidence anchors | `/tamlul:annotate` |
| 3 | Name, merge or reassign speakers | `/tamlul:speakers` |
| 4 | Write MD/SRT/VTT/EAF/TextGrid/... | `/tamlul:export` |
| - | Check on a running job | `/tamlul:progress` |

## Rules that govern every reading

1. **Never recompute a measurement.** Durations, counts, shares, overlaps and
   confidences are in `transcript.json` (`stats`, `speakers[].stats`). If a
   number is not there, say it is unavailable.
2. **Never edit words or timestamps in place.** Corrections go through the
   CLI (`speakers`, `annotate apply`) so the raw layer stays immutable and
   every change is logged.
3. **Every judgement cites its evidence** - a raw word index range and a
   verbatim quote that the CLI validates. An unanchored claim is rejected.
4. **The transcript is verbatim by default.** Fillers, repeats and false
   starts were really said; do not clean them up unless the profile says so.
5. **Long jobs run in the background.** Never `TaskStop` a transcription;
   poll `progress.json` instead.

## Invocation

The launcher is a Python script; the interpreter name differs by OS. Prefer
the **Bash tool** with `run_in_background: true` for anything that runs more
than a minute:

- Windows: `python "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" --data "${CLAUDE_PLUGIN_DATA}" <command> ...`
  (if that prints nothing and exits 49, the Microsoft Store alias is in the
  way: use `py -3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" ...`)
- macOS / Linux: `python3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" --data "${CLAUDE_PLUGIN_DATA}" <command> ...`

`--data "${CLAUDE_PLUGIN_DATA}"` is mandatory: it is the only way the CLI
learns where its environment, models and runs live.

## Scope

Any audio or video ffmpeg can read, incl. WhatsApp `.opus/.ogg`, `.m4a`,
`.amr`. Profiles: `meeting` (default), `voice-note`, `interview`, `ca`
(conversation analysis), `legal`, `auto`. Hebrew is the language; English
terms inside Hebrew speech are kept or restored, never translated.
