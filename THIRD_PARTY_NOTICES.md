# Third-party notices

tamlul's own code is MIT (see `LICENSE`). It downloads, vendors or borrows
from the components below. The `tests/test_notices.py` check requires one
entry here for every artefact in `src/tamlul/models.json` and every source in
`[tool.tamlul.vendor]` of `pyproject.toml`.

## Models (downloaded by the installer at a pinned revision; not redistributed)

| id | source | licence | notes |
|---|---|---|---|
| `asr/turbo-ct2@2025.05.13`, `asr/turbo-ct2@2025.04.03` | `ivrit-ai/whisper-large-v3-turbo-ct2` | Apache-2.0 | ivrit.ai Hebrew fine-tune of OpenAI Whisper large-v3-turbo. Please cite Marmor, Y. et al., "ivrit.ai: ...", Interspeech 2025. |
| `asr/large-v3-ct2@2025.05.13` | `ivrit-ai/whisper-large-v3-ct2` | Apache-2.0 | same authors |
| `diarize/pyannote-community-1` | `pyannote-community/speaker-diarization-community-1` @ `8a527374` | CC BY 4.0 | pyannoteAI speaker-diarization-community-1 pipeline weights (segmentation, embedding, BUT PLDA); the embedding model is WeNet WeSpeaker ResNet34-LM (CC BY 4.0). Attribution: pyannoteAI, https://huggingface.co/pyannote/speaker-diarization-community-1 ; WeSpeaker, https://github.com/wenet-e2e/wespeaker . tamlul substitutes its pipeline class in code; the downloaded config is unmodified. |
| `align/wav2vec2-300m-lm-hebrew` | `imvladikon/wav2vec2-xls-r-300m-lm-hebrew` | Apache-2.0 (HF tag; no LICENSE file) | eval-only consistency check |

## Libraries (installed from PyPI / the PyTorch index)

faster-whisper (MIT, SYSTRAN), CTranslate2 (MIT, OpenNMT), onnxruntime (MIT,
Microsoft), PyAV (BSD-3), pyannote.audio (MIT, CNRS; `utils/vbx.py` Apache-2.0,
Brno University of Technology), pyannote.core / pyannote.metrics / pyannote.pipeline
(MIT), torch / torchaudio / torchcodec (BSD-3), huggingface_hub (Apache-2.0),
numpy (BSD-3), PyYAML (MIT), pympi-ling (MIT), praatio (MIT), jiwer (Apache-2.0),
hebrew (MIT), num2words (LGPL-2.1, used as a library), uroman (MIT-style;
"This project uses the universal romanizer software uroman written by Ulf
Hermjakob, USC Information Sciences Institute (2015-2020)").

Silero VAD v6 (MIT, snakers4) is used as bundled inside faster-whisper.

## Copied code and data (licence texts ship alongside)

| source | licence and copyright | where | what |
|---|---|---|---|
| pyannote-audio PR #2051 / #2048 (base 4.0.7) | MIT, Copyright (c) 2020 CNRS and contributors | `src/tamlul/_vendor/pyannote_fast/` (+ `LICENSE`) | stacked-mask `get_embeddings` (`FastSpeakerDiarization`) |
| ivrit-ai/asr-training @ 760572069c3a | MIT, Copyright (c) 2024 ivrit.ai | `src/tamlul/_vendor/asr_training/` (+ `LICENSE`) | `HebrewTextNormalizer` (leaderboard normalizer), bidi control list |
| openai/whisper v20250625 | MIT, Copyright (c) 2022 OpenAI | `src/tamlul/_vendor/openai_whisper_normalizer/` (+ `LICENSE`) | `BasicTextNormalizer` (40 lines, checksum-tested) |
| KoganTheDev/hebrew-audio-transcriber @ 7ca0992 | MIT, Copyright (c) 2026 Speech-to-Text Transcriber Contributors | constants in `src/tamlul/media.py`, `text/correct.py`, `diarize/attribution.py`; licence text in `LICENSES/hebrew-audio-transcriber.txt` | `is_true_stereo`, phonetic-corrector and attribution constants |
| NVIDIA NeMo-text-processing @ 4844e1fb1ffa, `inverse_text_normalization/he` | Apache-2.0, NVIDIA CORPORATION & AFFILIATES | `src/tamlul/itn/data/nemo/` (+ `LICENSE`, `VENDOR.md`) | grammar tables and test cases, unmodified (file names flattened); the specification for the pure-Python port in `tamlul/itn` |

## Fixtures

`src/tamlul/fixtures/` contains a calibration clip built from the FLEURS
Hebrew (he_il) test set: Google FLEURS, CC BY 4.0,
https://huggingface.co/datasets/google/fleurs . Modifications: clips
concatenated with 0.5 s silences, resampled to 16 kHz mono and each clip
RMS-normalised to -23 dBFS. The reference text ships alongside, unchanged; it is
the FLoRes Hebrew translation (Goyal et al., 2022; FLORES-200,
https://github.com/facebookresearch/flores), CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).

Ideas re-implemented without copying code: jianfch/stable-ts (MIT - silence
suppression and regrouping), m-bain/whisperX (BSD-2-Clause - the
`assign_word_speakers` rule), MahmoudAshraf97/whisper-diarization
(BSD-2-Clause - sentence-majority speaker smoothing), Tetnd/tiktuv (MIT - RTL
text conventions), IN-meetings (YuvInv, unlicensed - whole-token
post-corrector, RMS silence gate), blitzai (hoodini, no LICENSE file - engine
dataclasses, Windows fixes), noScribe (GPL-3 - overlap/pause markers),
ivrit-py (MIT by metadata - progress-callback contract).
