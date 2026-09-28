---
name: export
description: Write or refresh exports of a tamlul transcript - Markdown, plain text for QDA tools, SRT/VTT subtitles, ELAN .eaf, Praat TextGrid, CSV, RTTM, HTML - all RTL-safe. Use when the user wants the transcript as a file, as subtitles (כתוביות), for ELAN/Praat/MAXQDA, or next to the recording.
argument-hint: "<run-id> [--formats md,srt,vtt,txt,eaf,textgrid,csv,rttm,html] [--beside-media]"
allowed-tools: Bash(python "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Bash(python3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Bash(py -3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) PowerShell(python "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) PowerShell(py -3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Read Glob
---

# Export

`PY` means `python` on Windows (`py -3` if it exits 49), `python3` elsewhere.
One launcher command per tool call: do not chain with `;`/`&&`, add `echo`, or
wrap it in a subshell - only the literal launcher command is pre-approved.

```bash
PY "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" --data "${CLAUDE_PLUGIN_DATA}" export <run> [--formats md,srt,vtt,txt,eaf,textgrid,csv,rttm,html] [--beside-media]
```

Exports land in `<run>/exports/`; `--beside-media` also copies them next to
the source recording. Formats:

| Format | For |
|---|---|
| `md` | Obsidian/Notion; turns with names and timestamps, then summary, decisions, action-item table when annotations exist |
| `txt` | MAXQDA/ATLAS.ti/NVivo/QualCoder (`[hh:mm:ss] שם: text`) |
| `srt`, `vtt` | subtitles/players; one cue per segment, RTL marks per line |
| `eaf` | ELAN: one tier per speaker + words + events (preserves overlap) |
| `textgrid` | Praat |
| `csv` | spreadsheets (segments) |
| `rttm` | diarization scoring |
| `html` | click-to-play local viewer (`dir="rtl"`) |

Report the paths; do not paste file contents. The HTML file is the default
viewer; an Artifact page is possible if the user explicitly asks for one.
