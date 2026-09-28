# Evaluation harness

Developer-only. Nothing here runs on the user path. The code lives in
`src/tamlul/eval/` (so the venv can run it); this folder holds the
developer scripts, this note, and nothing else — **no audio and no reference
text of any gated set is ever committed**. The only
redistributed audio in the repo is the CC-BY-4.0 calibration fixture.

Everything is driven by the launcher:

```
python scripts/tamlul.py --data <data> eval <op> ...
```

| Op | What | Needs |
|---|---|---|
| `sets` | the five leaderboard sets, licence notes, publication rules, what is downloaded | – |
| `download <set> [--token T]` | `snapshot_download` into `<data>/eval/<set>/hub`; the token is passed explicitly and never stored | HF token for `eval-whatsapp`, `eval-d1`, `saspeech` |
| `lb-check` | **scorer check**: re-scores every per-utterance CSV the leaderboard Space publishes with our vendored normalizer + WER | network, no token |
| `run <set> --column staff\|plugin --compute-type int8\|float32` | per-utterance CSV in the leaderboard's column layout + `machine.json`; resumable | the model installed |
| `score <csv…>` / `compare <a> <b>` | WER/CER/space-insensitive CER/filler-insensitive WER/Latin-token rates with paired-bootstrap 95 % CIs (10k resamples) | – |
| `grid` | {models} × {int8, float32} → the **`compute_type` decision per model** (float32 only if int8 is worse with a 95 % CI excluding 0 on any set) | the sets + models |
| `synth <set> --name N [--max-disagree-ms 30]` | tier-A synthetic timestamp/speaker set: clips trimmed by two detectors (noise-floor energy + Silero, bias-corrected; onset disagreement > 30 ms dropped; the residual is the onset precision floor, offsets get their own floor), log-uniform gaps, interleaved speakers, optional noise beds (`--noise-dir`, MUSAN or your own) | a downloaded set |
| `run <set> --opt FIELD=VALUE --no-plan --join text --clip-mode regions --no-recover --itn` | ablation knobs: DecodeOpts overrides, whole-utterance decoding, segment-text join, the pre-2026-09-21 per-region clips, no tail recovery, ITN on the hypothesis; `machine.json` records them | a run |
| `score-ts <run> <synth>` | onset/offset error at 25/50/100/200 ms, DER, word→speaker agreement | a run on the synthetic wav |
| `cosih download\|convert\|run\|score [ids]` | **CoSIH conversation tier**: ELAN → turns/UEM/text streams, full pipeline per recording, DER (collar 0 / 0.25, UEM), cpWER, speaker count, tier-B IU-onset error | network (cosih.com), non-commercial licence |
| `score-sa <transcript.json> <ref.rttm\|ref.json>` | DER + cpWER against any reference | – |
| `publish <csv> <set> <dest>` / `rehydrate` | strip reference text for every set but FLEURS; re-join from the user's own download | – |

## Publication rules

Per-utterance CSVs are published with `reference_text` / `norm_reference_text`
stripped for every set except FLEURS (`eval publish`); CoSIH and HUJI:
aggregates only; `hebrew_speech_kan`: never in published tables; ivrit.ai set
numbers carry "benchmark use per the leaderboard's own practice; confirmation
requested". Every results manifest records each set's licence note as
presented at download.
