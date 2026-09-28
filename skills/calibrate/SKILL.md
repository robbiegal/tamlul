---
name: calibrate
description: Measure tamlul's speed, int8 kernel sanity and silence behaviour on this machine (never accuracy) and record the defaults it picks. Use after installing models, after a plugin update, or when the doctor says calibration is stale.
disable-model-invocation: true
argument-hint: "[--quick|--full]"
allowed-tools: Bash(python "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Bash(python3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Bash(py -3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) PowerShell(python "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) PowerShell(py -3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Read
---

# Calibrate

`PY` means `python` on Windows (`py -3` if it exits 49), `python3` elsewhere.
One launcher command per tool call: do not chain with `;`/`&&`, add `echo`, or
wrap it in a subshell - only the literal launcher command is pre-approved.

```bash
PY "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" --data "${CLAUDE_PLUGIN_DATA}" calibrate --quick
```

`--quick` (~2-3 minutes, runs automatically after `models install`): realtime
factor at two thread counts, the int8-vs-float32 disagreement proxy (a broken
int8 kernel path switches this machine to float32; accuracy itself was decided
on the eval sets before release), one silence case with the gate on and off,
and the pyannote fast-embedding equality check.

`--full` (on demand, tens of minutes): thread and beam sweeps, sequential vs
concurrent diarization, all silence cases per installed revision, and any
optional engine against the reference engine.

Run it in the background and report `calibration.json`'s chosen defaults
(threads, compute type, concurrency, fast embeddings) in one table.
