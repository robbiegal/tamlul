---
name: annotate
description: Add judgement layers to a finished Hebrew transcript - summary, speaker names from context, topics, decisions, action items with owners, open questions, key quotes, restored English terms - every item anchored to the words it came from. Use after transcribe, or when asked for a summary (סיכום), action items (משימות), decisions (החלטות) or who-is-who from a transcribed Hebrew recording.
argument-hint: "<run-id> [--participants <file>] [--glossary <file>]"
context: fork
agent: conversation-annotator
allowed-tools: Bash(python "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Bash(python3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Bash(py -3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) PowerShell(python "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) PowerShell(py -3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Read Write Glob
---

# Annotate a run

You are a forked subagent and do not see the conversation. Everything you need
is in `$ARGUMENTS`: the run id (or run directory), plus optional
`--participants <file>` and `--glossary <file>` the main session wrote from
what the user said.

`PY` means `python` on Windows (`py -3` if it exits 49), `python3` elsewhere.
One launcher command per tool call: do not chain with `;`/`&&`, add `echo`, or
wrap it in a subshell - only the literal launcher command is pre-approved.

## 1. Get the contract, do not recall it

```bash
PY "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" --data "${CLAUDE_PLUGIN_DATA}" annotate prepare <run> [--participants F] [--glossary F]
```

This prints the path of `annotate_input.md` (the turn-level rendering with
raw word indices, the candidate lists, the stats you may quote) and the exact
schema of `annotations.json`. **Follow what it prints.** It is versioned with
the measurement code; a remembered copy names fields that no longer exist.

## 2. Write `annotations.json` into the run directory

Every item carries `evidence: [{ "w": [i0, i1], "quote": "...", "s": ms, "e": ms }]`
where `w` is a raw word index range and `quote` is the verbatim rendering of
that range. No anchor, no item.

## 3. Apply, and fix once if rejected

```bash
PY "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" --data "${CLAUDE_PLUGIN_DATA}" annotate apply <run> <run-dir>/annotations.json
```

The CLI validates every anchor (range exists, quote matches, times inside the
range, speaker ids exist, restored terms map to flagged words). If it rejects
items, read the reasons, fix `annotations.json` **once**, and apply again. Do
not loop further; report what was rejected.

Your final message is a short report: what was written, how many items,
what was rejected and why, and the paths of the refreshed exports.
