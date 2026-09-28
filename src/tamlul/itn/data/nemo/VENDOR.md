# NVIDIA NeMo-text-processing - Hebrew inverse text normalisation data

- Upstream: https://github.com/NVIDIA/NeMo-text-processing @ `4844e1fb1ffa` (2025-11-19):
  `nemo_text_processing/inverse_text_normalization/he/data/**/*.tsv` and
  `tests/nemo_text_processing/he/data_inverse_text_normalization/*.txt`
- Copyright: NVIDIA CORPORATION & AFFILIATES
- Licence: Apache-2.0 - full text in `LICENSE` next to this file
- Modifications: file names flattened (`<subdir>/<name>.tsv` -> `<subdir>_<name>.tsv`); contents unchanged.
- Use: the tables are the specification of the pure-Python port in `tamlul/itn/grammar.py`, and the
  test cases are run by `tests/test_itn.py`.
