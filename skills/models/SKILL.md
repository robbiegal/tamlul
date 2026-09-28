---
name: models
description: List, plan, install, verify or remove the pinned models tamlul uses (ivrit.ai Whisper CT2, pyannote community-1). Sizes are stated before any download.
disable-model-invocation: true
argument-hint: "[list|plan|install|verify|remove] [ids...] [--profile P] [--quality max] [--all]"
allowed-tools: Bash(python "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Bash(python3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Bash(py -3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) PowerShell(python "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) PowerShell(py -3 "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" *) Read
---

# Models

`PY` means `python` on Windows (`py -3` if it exits 49), `python3` elsewhere.
One launcher command per tool call: do not chain with `;`/`&&`, add `echo`, or
wrap it in a subshell - only the literal launcher command is pre-approved.

```bash
PY "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" --data "${CLAUDE_PLUGIN_DATA}" models plan --profile ${user_config.profile}
```

shows what the active profile needs, with sizes and licences. Then, **after
the user agrees to the stated size**:

```bash
PY "${CLAUDE_PLUGIN_ROOT}/scripts/tamlul.py" --data "${CLAUDE_PLUGIN_DATA}" models install --profile ${user_config.profile} --yes
```

(`--quality max` adds the 3.09 GB large-v3 model; `--all` fetches every
non-optional artefact, 6.3 GB.) Run installs in the background. Every file is
hashed against the pinned SHA-256 before it is marked usable; `models verify`
re-checks, `models remove <id>` frees space. Models are downloaded once from
Hugging Face at a pinned revision; no token is needed.
