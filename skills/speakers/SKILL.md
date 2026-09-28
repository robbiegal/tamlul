---
name: speakers
description: Review, name, merge, split or reassign speakers in a tamlul transcript (דוברים, מי זה דובר 1). Use when the user names a speaker, says two labels are the same person, or says a stretch was said by someone else.
argument-hint: "<run-id> [rename S2 \"שם\" | merge S3 into S1 | reassign T42 S2 | reassign 00:12:03-00:12:09 S1 | split S1 at 00:20:00]"
allowed-tools: Bash(python "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Bash(python3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Bash(py -3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) PowerShell(python "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) PowerShell(py -3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Read Glob
---

# Speakers

`PY` means `python` on Windows (`py -3` if it exits 49), `python3` elsewhere.
One launcher command per tool call: do not chain with `;`/`&&`, add `echo`, or
wrap it in a subshell - only the literal launcher command is pre-approved.

Show the current overlay first:

```bash
PY "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" --data "${CLAUDE_PLUGIN_DATA}" speakers <run> show
```

It lists each speaker id, its current name, talk share, turn count, and the
5-second sample clip path (`samples/S1_1.wav`) the user can listen to.

Then apply what the user asked, one command per edit:

| The user says | Command |
|---|---|
| "דובר 2 זה יוסי" | `speakers <run> rename S2 "יוסי"` |
| "S3 and S1 are the same person" | `speakers <run> merge S3 into S1` |
| "turn 42 was actually S2" | `speakers <run> reassign T42 S2` |
| "from 12:03 to 12:09 it was S1" | `speakers <run> reassign 00:12:03-00:12:09 S1` |
| "S1 becomes someone else at 20:00" | `speakers <run> split S1 at 00:20:00` |

Ids never change; names are an overlay with provenance. After each edit the
CLI recomputes turns and stats, re-resolves every annotation anchor, and
re-renders the exports already in the run directory. Never edit
`transcript.json` or `speakers.yaml` by hand.
