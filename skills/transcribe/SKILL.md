---
name: transcribe
description: Transcribe a Hebrew recording (meeting, interview, call, WhatsApp voice note, podcast) or a folder of recordings into a speaker-attributed, timestamped transcript.json plus exports, locally. Use when asked to transcribe (תמלל, לתמלל, תמלול) Hebrew audio or video, to convert a recording (הקלטה) or voice note (הודעה קולית) to text, or to get who-said-what (מי אמר מה) from a Hebrew conversation. Handles first-run setup by asking before any download.
argument-hint: "<audio|video|folder> [--profile meeting|voice-note|interview|ca|legal|auto] [--speakers N] [--quality max]"
allowed-tools: Bash(python "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Bash(python3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Bash(py -3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) PowerShell(python "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) PowerShell(py -3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Read Glob
---

# Transcribe

Use the Bash tool. `PY` below means `python` on Windows (`py -3` if that exits
49) and `python3` on macOS/Linux. One launcher command per tool call: do not
chain with `;`/`&&`, add `echo`, or wrap it in a subshell - only the literal
launcher command is pre-approved.

## 1. Check readiness (fast, no downloads)

```bash
PY "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" --data "${CLAUDE_PLUGIN_DATA}" status --json
```

Read the JSON:

- `installed: false` -> the Python environment is missing. **Tell the user the
  cost and ask**: roughly 2.5 GB of packages (torch CPU for speaker
  diarization) into `data_dir`. With consent, run in the background:
  `PY ... --data "${CLAUDE_PLUGIN_DATA}" env install`
- `models[...]` with `installed: false` for any id in `profile_models` -> models
  are missing. **State the sizes from `size_bytes` and ask** (turbo 1.62 GB,
  pyannote 0.03 GB for the default meeting profile). With consent:
  `PY ... --data "${CLAUDE_PLUGIN_DATA}" models install --profile ${user_config.profile} --yes`
  (`--yes` is the only way the CLI downloads anything.)
- `calibration.stale: true` -> after the models exist, run
  `PY ... --data "${CLAUDE_PLUGIN_DATA}" calibrate --quick` (about 2-3 minutes;
  measures speed and the int8 kernel on this machine, never accuracy). Without
  it the CLI uses safe defaults; say so and continue.

Never run `env install` or `models install` without the user's explicit yes in
this conversation.

## 2. Transcribe (background)

```bash
PY "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" --data "${CLAUDE_PLUGIN_DATA}" transcribe "<media>" --profile ${user_config.profile} --compute-type ${user_config.compute_type} --threads ${user_config.threads} --diarization ${user_config.diarization} --models-dir "${user_config.models_dir}" [--speakers N] [--quality max] [--glossary <file>] [--participants <file>]
```

Run it with `run_in_background: true`. Empty option values mean "use the
default"; a user's explicit request (e.g. "two speakers", "it is a voice note")
becomes `--speakers 2` / `--profile voice-note` and overrides the picker.

The run id is printed on the first line (`run: <id>`). Then:

- poll `<data>/runs/<id>/progress.json` with `Read` every few minutes
  (`stage`, `pct`, `eta_s`); a `Monitor` on `log.txt` may be used only if that
  tool exists in this session;
- `segments.jsonl` grows while the job runs - read it if the user wants an
  early look;
- **never `TaskStop` the job**; a transcription that looks idle is usually in
  diarization;
- the background task ends the turn: wait for its completion notification
  instead of ending the conversation with "I'll report back";
- if `... progress <id>` says `"stalled": true` (the worker's process is
  gone - it dies with the session that launched it), run
  `... transcribe --resume <id>` (same options): it continues from the last
  finished group.

## 3. Report

When `progress.json` says `stage: done`, reply with a short summary and the
paths: `transcript.json`, `exports/` (md, srt, vtt, txt, eaf, TextGrid, csv,
html), `speakers.yaml`. Do not paste the transcript inline. Offer
`/tamlul:annotate` for summaries/action items and `/tamlul:speakers` if the
speakers are unnamed (`דובר 1`, `דובר 2`).

## Do not

- Do not transcribe with any cloud service or a different local tool.
- Do not clean up fillers or "fix" what was said; the transcript is verbatim.
- Do not state a duration, word count or talk share you did not read from
  `transcript.json`.
