# pyannote-audio stacked-mask embeddings (PR #2048 / #2051)

- Upstream: https://github.com/pyannote/pyannote-audio/pull/2048 (open) and /pull/2051 (closed by its
  author in favour of #2048); base blobs `e197d0312` (speaker_diarization.py) and `8c0edd878`
  (speaker_verification.py) match the released 4.0.7.
- Licence: MIT, Copyright (c) 2020 CNRS and pyannote contributors - full text in `LICENSE` next to this file
- Local diffs: written as `FastSpeakerDiarization(SpeakerDiarization)` overriding `get_embeddings`,
  with the model probe as a module function; the per-chunk loop is the PR's line for line.
- Why vendored: 3.1-4.1x end-to-end diarization on CPU with identical output (#2050: DER identical
  on all 216 VoxConverse dev files); upstream has not merged it and has not responded.
- Pin: `pyannote.audio==4.0.7`. On any pyannote bump the calibration equality check
  (`np.allclose` embeddings + identical RTTM vs stock) must pass again, otherwise the launcher falls
  back to the stock pipeline and the doctor says so (design §4.18).
