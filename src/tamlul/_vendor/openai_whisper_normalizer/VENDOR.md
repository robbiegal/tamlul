# openai/whisper BasicTextNormalizer

- Upstream: https://github.com/openai/whisper, tag v20250625, file `whisper/normalizers/basic.py`
- Licence: MIT, Copyright (c) 2022 OpenAI - full text in `LICENSE` next to this file
- Local diffs: the `regex` import used only by `split_letters` is removed; `split_letters=True` raises.
- Why vendored: the leaderboard normalizer (`ivrit-lb`) must be byte-for-byte this code; pulling the
  `openai-whisper` package would drag torch, numba and tiktoken into the core install.
- Check: `tests/test_normalize.py` verifies the edge-case table and the recorded upstream sha256.
