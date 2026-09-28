# tamlul (תמלול)

Local Hebrew conversation transcription and annotation, packaged as a
[Claude Code](https://claude.com/claude-code) plugin.

- **Audio inference is local.** ASR is ivrit.ai's Whisper large-v3-turbo Hebrew
  fine-tune through faster-whisper/CTranslate2 on CPU (CUDA when present);
  speaker diarization is pyannote.audio 4 with the community-1 weights. Audio
  never leaves the machine.
- **Claude does the judgement layers** - summaries, action items, speaker
  names from context, restored English terms - inside the Claude Code session,
  with every item anchored to the words it came from. No API key anywhere.
- **Output is annotation-grade**: word timestamps, speaker labels with
  confidence, overlaps, pauses, turns, per-speaker stats, and exports for
  Markdown, subtitles (SRT/VTT), QDA tools (TXT), ELAN (EAF), Praat (TextGrid),
  CSV, RTTM and a local RTL HTML viewer.

## Install

```
/plugin marketplace add robbiegal/plugin-marketplace
/plugin install tamlul@robbiegal-tools
/tamlul:doctor
```

The doctor builds a Python environment in the plugin's data directory
(~2.5 GB with torch CPU), installs the models for your profile after telling
you their size (1.62 GB + 33 MB for `meeting`), and runs a 2-3 minute
calibration that measures speed on your machine. Nothing downloads without
your yes.

Then: *"transcribe this meeting"* / *"תמלל את ההקלטה"* with a file path, or
`/tamlul:transcribe <file>`.

**Permission prompts.** Every skill pre-approves its own launcher command
(`allowed-tools`), so the slash form runs without prompts. When Claude picks
the skill itself from a natural-language request, Claude Code (2.1.274) asks
once to execute the skill and then — because the pre-approval is not carried
over on that path — once for the launcher command; answer "always allow" for
the launcher prefix and it stays approved for the project. To pre-approve it
yourself, add to `.claude/settings.json`:

```json
{ "permissions": { "allow": [ "Bash(python \"<plugin root>/scripts/tamlul.py\" *)" ] } }
```

where `<plugin root>` is the path `/tamlul:doctor` prints (the marketplace
cache directory, with forward slashes).

A transcription runs in the background of the session that started it and
dies with that session; `/tamlul:progress` says `stalled: true` in that case
and `transcribe --resume <id>` continues from the last finished group.

## Requirements

Python 3.11-3.14 (CPython, not the Microsoft Store alias), ffmpeg on PATH
(or the bundled PyAV fallback), ~4 GB of disk for environment + default
models, 8 GB RAM minimum (16 GB recommended). Windows 11, macOS and Linux.

## Evaluation

`eval/README.md` describes the harness (`tamlul eval ...`), which reproduces
every number below on your own machine (the gated ivrit.ai sets need your own
Hugging Face access). Measured on CPU with the shipped int8 model: the scorer
reproduces the ivrit.ai leaderboard's own per-utterance CSVs cell for cell;
the engine reproduces the published FLEURS and eval-whatsapp cells within 0.3
WER points; the shipped decoding path matches the leaderboard recipe on
eval-whatsapp (0.074 vs 0.073, not significantly different) and its GPU cell
on eval-d1 (0.052 vs 0.053); and int8 showed no measurable loss against
float32 on either set, so int8 is the default.

## Contributing

`CONTRIBUTING.md` explains the architecture rule and lists the invariants the
code must keep. Code comments cite sections of the internal design notes
("design §4.3"); those notes are not part of this repository.

## Publishing

`docs/marketplace-entry.json` is the catalogue entry for
`robbiegal/plugin-marketplace`.

## Licence

MIT for the plugin code. The models, vendored code, NeMo data and the test
fixture carry their own licences - see `THIRD_PARTY_NOTICES.md`; the licence
texts sit next to the files they cover and under `LICENSES/`.
