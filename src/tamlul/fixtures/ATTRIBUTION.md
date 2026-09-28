# Fixture attribution

`fleurs_he.wav` / `fleurs_he.jsonl` are derived from the FLEURS dataset, Hebrew (he_il) test split.

- Creator: Google (Conneau et al., "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech", 2022)
- Source: https://huggingface.co/datasets/google/fleurs
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
- Modifications: 31 test clips concatenated with 0.5 s of digital silence between them, resampled to 16 kHz mono 16-bit PCM, each clip RMS-normalised to -23 dBFS. Reference transcriptions are unchanged.
- Text origin: the reference sentences are the FLoRes Hebrew translations (Goyal et al., 2022; FLORES-200,
  https://github.com/facebookresearch/flores, CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/), redistributed unchanged.
- Used for: per-machine calibration (speed, engine disagreement, silence behaviour); never for accuracy claims.
