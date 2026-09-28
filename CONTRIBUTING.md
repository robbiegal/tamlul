# Working on tamlul

Instructions for an agent or a person editing **this codebase**. If you are
instead annotating a transcript, the prompt you want is printed by
`tamlul annotate prepare <run>`.

## The architecture rule everything else follows

> **Python measures. The agent judges. Neither does the other's job.**

The pipeline computes every countable thing from timestamps into
`transcript.json`. Claude reads that file and produces interpretation with
evidence anchors that Python validates. Claude is forbidden from recomputing
measurements and from editing words or timestamps in place.

## Invariants - do not "improve" these

1. **Audio never leaves the machine; no paid API anywhere.** No cloud ASR, no
   cloud fallback, no `anthropic` SDK, no API key - in the plugin or in CI.
2. **Always `language="he"` and a pinned revision + commit + sha256**, written
   into every transcript (`pipeline.asr`).
3. **No previous-text conditioning.** Hebrew costs ~2.9 Whisper tokens per
   word; the rolling prompt is what exhausts the 448-token window and VAD does
   not protect against it. `condition_on_previous_text=False`, always.
4. **`compression_ratio_threshold >= 2.8`, temperature ladder <= 0.4.** The 2.4
   default misfires on 5-20 % of dense Hebrew windows.
5. **Never a Latin name or a digit in the prompt or glossary.** Measured
   regressions. Correct after decoding, confidence-gated; Claude restores
   flagged words only.
6. **Explicit threads; sequential by default; batched opt-in and flagged;
   diarization in a child process** (two Intel OpenMP runtimes in one process
   is an abort waiting to happen).
7. **Strip bidi controls before anything else.** Every Whisper model emits
   U+202B at 10-25 % of segment starts.
8. **Verbatim survives**: `no_repeat_ngram_size=0`, fillers kept unless
   `--clean`; the repetition guard flags, never deletes.
9. **One plan per file on every machine** (`chunks.json`), immutable once ASR
   starts; digital silence is never decoded.
10. **No speaker layer is better than a fake one.**
11. **Word-level attribution, then re-segment**; raw diarization labels and
    `words[].t_raw` are immutable; names are an overlay with provenance.
12. **Evidence anchors reference raw word indices**; turn ids are never
    load-bearing.
13. **Segment-only engines never feed attribution.**
14. **Models are local directories at run time, hashed at install; nothing
    downloads without `--yes`.**
15. **Telemetry/offline variables are process-scoped** to tamlul's own
    subprocesses - never exported session-wide, never written to
    `CLAUDE_ENV_FILE`.
16. **A malformed `transcript.json` stops the run**; readers accept any `1.x`;
    a major bump ships `migrate`.
17. **Every tuned constant is written into the output** (`stats.thresholds`,
    `pipeline`).
18. **Long jobs checkpoint per group; a resume never mixes models, options or
    plans; skills forbid `TaskStop`.**
19. **Accuracy is decided on the eval sets with CIs; per-machine calibration
    measures speed, disagreement and silence only.**
20. **No accelerator becomes a default without the calibration gate.**
21. **Report WER only with the named normalizer profile and a CI; DER with the
    collar convention; every table labels gold vs silver.**
22. **Vendored code carries provenance** (`src/tamlul/_vendor/<source>/VENDOR.md`
    + a `derived from` header per file) and falls back when its upstream moves.

## Layout

- `scripts/tamlul.py` - stdlib launcher (any CPython >= 3.8); finds the venv
  under `--data` and re-executes `python -m tamlul.cli` there.
- `scripts/tamlul_env.py` - stdlib venv builder; installs torch from the
  PyTorch CPU index before the `diarize` extra.
- `src/tamlul/` - the package. Never `pip install -e` it into the data venv:
  the plugin directory is version-scoped and garbage-collected.
- `skills/`, `agents/` - the Claude Code surface. Every command in a SKILL.md
  is a literal substituted path; `allowed-tools` prefix rules refuse `$VAR`.
- `tests/` - runs without any model (fake backends + recorded fixtures).
- `eval/` - developer scripts and the harness notes; the harness code is
  `src/tamlul/eval/` (so the data venv can run `tamlul eval`). No audio and no
  gated reference text is ever committed.
- Package data must reach the *built* wheel (the venv installs a copy, not an
  editable link): CI unpacks the wheel and checks the ITN tables, schema,
  prompts, fixtures and VENDOR notes are inside. A missing glob in
  `[tool.setuptools.package-data]` fails there, not on a user's machine.
- One launcher command per tool call in every SKILL.md: only the literal
  `python "<root>/scripts/tamlul.py" ...` prefix is pre-approved; chaining with
  `;`/`&&` or a trailing `echo` is refused (verified live against Claude Code).

## Testing

```
pip install -e ".[dev,asr,export,eval]"
pytest -q
```

Without the `asr` extra the tests that import faster-whisper are skipped.

The suite must pass on a machine that has never downloaded a model.
