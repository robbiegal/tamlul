"""Evaluation harness (design §4.15, §9): developer-only, never on the user path.

``tamlul eval ...`` drives it. Public leaderboard artefacts and FLEURS need no
token; the gated ivrit.ai sets take a one-time HF token passed explicitly to
``snapshot_download`` (the plugin's own environment disables implicit tokens).
No audio and no reference text of a gated set is ever written under the repo;
results land under ``<data>/eval/``.
"""
