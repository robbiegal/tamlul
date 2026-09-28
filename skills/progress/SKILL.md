---
name: progress
description: Show the progress of a running or finished tamlul transcription job.
disable-model-invocation: true
argument-hint: "[run-id]"
allowed-tools: Bash(python "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Bash(python3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Bash(py -3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) PowerShell(python "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) PowerShell(py -3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Read
---

# Progress

`PY` means `python` on Windows (`py -3` if it exits 49), `python3` elsewhere.
One launcher command per tool call: do not chain with `;`/`&&`, add `echo`, or
wrap it in a subshell - only the literal launcher command is pre-approved.

```bash
PY "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" --data "${CLAUDE_PLUGIN_DATA}" runs list
PY "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" --data "${CLAUDE_PLUGIN_DATA}" progress <run-id>
```

Report stage, percent, ETA and the realtime factor so far. A job in the
`diarize` stage can look idle for minutes; that is normal. Never stop a
running job.

If the JSON says `"stalled": true` (the worker's pid is gone - a job launched
from a session dies when that session ends), run
`... transcribe --resume <run-id>` (the original media path is in `run.json`);
it continues from the last finished group.
