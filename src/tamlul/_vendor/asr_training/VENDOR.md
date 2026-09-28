# ivrit-ai/asr-training HebrewTextNormalizer

- Upstream: https://github.com/ivrit-ai/asr-training @ `760572069c3a` (master, 2026-05-03), file `evaluate_model.py`
- Licence: MIT, Copyright (c) 2024 ivrit.ai - full text in `LICENSE` next to this file
- Local diffs: only `clean_some_unicode_from_text`, `remove_niqqud` and `HebrewTextNormalizer` are kept;
  `remove_niqqud` falls back to a local table with the same code points when `hebrew` is not installed.
- Why vendored: leaderboard-comparable WER requires this exact normalizer (design section 4.6, invariant 21).
