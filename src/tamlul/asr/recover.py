"""Re-decode raw speech the decoder skipped (design §4.3, added after the 2026-09-22 ablation).

Whisper stops a 30-s window early now and then and faster-whisper seeks to the window end, so
the speech in the tail is never decoded (synthetic tier A: a 4.3-s clip vanished at 103-108 s of
a 300-s group; eval-d1 lost ~19 words over 17 chunks). ``recover_missed`` finds raw Silero
regions that no segment covers, decodes each on its own and returns the extra segments in the
group's time base.

Two guards, both measured on eval-whatsapp (2026-09-22): a 1-s minimum span recovered mostly
hesitations Whisper skips on purpose (``אמ``, ``אה``) and turned 1-2-s slices of "umm" into
``אהההה…`` loops (+39 insertions, -4 deletions); so the minimum is 2.5 s and a recovered segment
is kept only if it passes the decoder's own sanity signals.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..media import SR
from ..plan import PAD_S, Group, uncovered_speech
from ..text.clean import repetition_loop, strip_bidi
from .base import RawSegment
from .decode import DecodeOpts

RECOVER_MIN_S = 2.5


def plausible(seg: RawSegment, opts: DecodeOpts) -> bool:
    """A recovered segment must look like speech the decoder believes in."""
    words = strip_bidi(seg.text).split()
    if not words:
        return False
    if seg.no_speech_prob is not None and seg.no_speech_prob > opts.no_speech_threshold:
        return False
    if seg.avg_logprob is not None and seg.avg_logprob < opts.log_prob_threshold:
        return False
    if seg.compression_ratio is not None and seg.compression_ratio > opts.compression_ratio_threshold:
        return False
    if repetition_loop(words) is not None:
        return False
    # a single token stretched into a loop of one letter (אהההה): 6+ characters drawn from at most two
    if any(len(w) >= 6 and len(set(w)) <= 2 for w in words):
        return False
    return True


def trim_to_uncovered(seg: RawSegment, off: float, covered: list[tuple[float, float]]) -> RawSegment | None:
    """Drop the words of a recovered segment whose midpoint (file time) lies in already-decoded time.

    A missed span is re-decoded whole, but it can be partly covered (the decoder stopped in the
    middle of it), and splicing the whole result in would transcribe those words twice. Words concatenate to the segment text (``merge_fragments``), so the text is rebuilt
    from the kept words."""
    def inside(t: float) -> bool:
        return any(a <= t < b for a, b in covered)

    if not seg.words:
        return None if inside(off + (seg.s + seg.e) / 2) else seg
    keep = [w for w in seg.words if not inside(off + (w.s + w.e) / 2)]
    if not keep:
        return None
    if len(keep) == len(seg.words):
        return seg
    return replace(seg, s=keep[0].s, e=keep[-1].e, text="".join(w.t for w in keep), words=keep)


def recover_missed(backend, audio: np.ndarray, g: Group, segs: list[tuple[RawSegment, float]],
                   raw_regions: list[list[float]], opts: DecodeOpts, *, min_s: float = RECOVER_MIN_S,
                   log=None) -> list[tuple[RawSegment, float]]:
    """``segs`` are (segment, offset_s) pairs already decoded for group ``g``; returns the extra pairs.

    ``audio`` is the whole file (or channel) at 16 kHz; each recovered span is decoded from its own
    padded slice, so the returned offsets differ from the group's."""
    covered = [(s.s + o, s.e + o) for s, o in segs]
    missed = uncovered_speech(raw_regions, covered, g, min_s=min_s)
    out: list[tuple[RawSegment, float]] = []
    for ms, me in missed:
        rs, re_ = max(g.start_s, ms - PAD_S), min(g.end_s, me + PAD_S)
        got = backend.transcribe(audio[int(rs * SR):int(re_ * SR)], None, opts)
        trimmed = [t for t in (trim_to_uncovered(s, rs, covered) for s in got) if t is not None]
        kept = [(s, rs) for s in trimmed if plausible(s, opts)]
        if log:
            log(f"asr: group {g.index}: skipped speech {ms:.1f}-{me:.1f}s re-decoded -> "
                f"{len(kept)}/{len(got)} segment(s) kept")
        out += kept
    return out
