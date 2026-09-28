# derived from https://github.com/pyannote/pyannote-audio pull request #2048 (and its predecessor #2051),
# src/pyannote/audio/pipelines/speaker_diarization.py and speaker_verification.py, MIT License
# Copyright (c) 2020- CNRS and pyannote contributors. Modifications: expressed as a subclass of the
# released 4.0.7 SpeakerDiarization pipeline instead of a patch; the multi-speaker-mask probe is
# a module function; everything else follows the PR line for line.
"""Stacked-mask embedding extraction: 3.1-4.1x end-to-end on CPU, output-identical.

Masks are only used by the pooling layer of the WeSpeaker embedding model; the
(much more expensive) layers before it do not depend on them. The PR therefore
runs the model once per chunk with the masks of all its speakers stacked,
instead of once per (chunk, speaker) pair. #2050 reported identical DER on all
216 VoxConverse dev files (max |delta embedding| 2.3e-6). The calibration's
equality check (np.allclose embeddings + identical RTTM vs the stock pipeline)
is the per-machine guard; on any mismatch the stock pipeline is used.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Optional

import numpy as np
import torch
from einops import rearrange
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pyannote.core import SlidingWindowFeature

PINNED_PYANNOTE = "4.0.7"


def supports_multi_speaker_masks(embedding) -> bool:
    """Whether the wrapped model accepts (batch, speakers, frames)-shaped masks.

    Models pooling with StatsPool do, but layers after the pooling may not be
    agnostic to the extra dimension; rather than enumerating classes, run the
    model once and look at the shape of its output (as the PR does).
    """
    try:
        model = embedding.model_
        device = embedding.device
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = model(torch.randn(1, 1, embedding.sample_rate).to(device), weights=torch.ones(1, 2, 16).to(device))
        return tuple(out.shape) == (1, 2, embedding.dimension)
    except Exception:  # noqa: BLE001 - any failure means "no"
        return False


class FastSpeakerDiarization(SpeakerDiarization):
    """SpeakerDiarization with per-chunk (stacked-mask) embedding extraction."""

    def get_embeddings(self, file, binary_segmentations: SlidingWindowFeature, exclude_overlap: bool = False,
                       hook: Optional[Callable] = None):
        if getattr(self, "training", False) or not supports_multi_speaker_masks(self._embedding):
            return super().get_embeddings(file, binary_segmentations, exclude_overlap=exclude_overlap, hook=hook)

        duration = binary_segmentations.sliding_window.duration
        num_frames = binary_segmentations.data.shape[1]

        if exclude_overlap:
            min_num_samples = self._embedding.min_num_samples
            num_samples = duration * self._embedding.sample_rate
            min_num_frames = math.ceil(num_frames * min_num_samples / num_samples)
            clean_frames = 1.0 * (np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2)
            clean_segmentations = SlidingWindowFeature(binary_segmentations.data * clean_frames,
                                                       binary_segmentations.sliding_window)
        else:
            min_num_frames = -1
            clean_segmentations = SlidingWindowFeature(binary_segmentations.data, binary_segmentations.sliding_window)

        return self._get_embeddings_per_chunk(file, binary_segmentations, clean_segmentations, min_num_frames, hook=hook)

    def _get_embeddings_per_chunk(self, file, binary_segmentations, clean_segmentations, min_num_frames, hook=None):
        num_chunks, _, num_speakers = binary_segmentations.data.shape
        chunks = binary_segmentations.sliding_window

        # embedding_batch_size keeps counting (chunk, speaker) pairs so peak memory is unchanged
        batch_size = max(1, self.embedding_batch_size // num_speakers)
        batch_count = math.ceil(num_chunks / batch_size)

        embeddings = np.empty((num_chunks, num_speakers, self._embedding.dimension), dtype=np.float32)

        if hook is not None:
            hook("embeddings", None, total=batch_count, completed=0)

        for i, start in enumerate(range(0, num_chunks, batch_size), 1):
            stop = min(start + batch_size, num_chunks)
            waveform_batch = torch.vstack([self._audio.crop(file, chunks[c], mode="pad")[0][None]
                                           for c in range(start, stop)])
            masks = np.nan_to_num(binary_segmentations.data[start:stop], nan=0.0)
            clean_masks = np.nan_to_num(clean_segmentations.data[start:stop], nan=0.0)
            used_masks = np.where(np.sum(clean_masks, axis=1, keepdims=True) > min_num_frames, clean_masks, masks)
            mask_batch = torch.from_numpy(np.ascontiguousarray(used_masks.transpose(0, 2, 1), dtype=np.float32))
            embedding_batch: np.ndarray = self._embedding(waveform_batch, masks=mask_batch)
            embeddings[start:stop] = embedding_batch
            if hook is not None:
                hook("embeddings", rearrange(embedding_batch, "c s d -> (c s) d"), total=batch_count, completed=i)

        return embeddings
