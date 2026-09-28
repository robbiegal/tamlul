"""The decoding plan (design §4.3): Silero VAD + RMS gate -> groups -> chunks.json.

One plan per file on every machine. faster-whisper's bundled Silero v6 ONNX is
used through ``faster_whisper.vad`` (no torch, no ``silero-vad`` package). The
plan is immutable once ASR starts; its sha goes into ``run.json``.

Why a plan at all: faster-whisper feature-extracts the whole array passed to
``transcribe()`` regardless of ``clip_timestamps`` (#1030: an 11-h MP3 took
27 GB), so ASR is given a SLICE per group with relative clip timestamps and
the times are offset back. The RMS gate keeps digitally silent regions out of
the model entirely - the fine-tune emits Knesset boilerplate on silence.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import asdict, dataclass, field

import numpy as np

from .media import SR

VAD_PARAMS = {
    "threshold": 0.5,
    "neg_threshold": None,
    "min_speech_duration_ms": 250,
    "max_speech_duration_s": 30.0,
    "min_silence_duration_ms": 600,
    "speech_pad_ms": 300,
}
RMS_GATE = 1e-3            # IN-meetings ADR-003: below this the model hallucinates on silence
PAD_S = 0.3
GROUP_SPLIT_SILENCE_S = 2.0
GROUP_MAX_S = 300.0        # not one decoder window per group: 28.5-s groups cost +0.5 WER on eval-whatsapp
                           # (hard cuts in continuous speech); missed window tails are recovered instead
RECOVER_COVER = 0.5        # a raw region is "covered" when segments overlap at least this fraction of it
                           # (the minimum span to recover lives in asr/recover.py)
PART_MAX_S = 3600.0        # >= 2-h files are split into <= 60-min parts at VAD boundaries
LONG_FILE_S = 7200.0
CLIP_MODE = "group"        # what the backend is handed per group; see decode_clips()


@dataclass
class Group:
    index: int
    start_s: float          # slice start (padded) in file time
    end_s: float            # slice end (padded) in file time
    clips: list[list[float]]  # [start, end] in file time, unpadded speech regions
    part: int = 0
    channel: int = -1         # -1 = mono mix; 0/1 = a true-stereo channel

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class Plan:
    source: str
    params: dict
    duration_s: float
    speech_s: float
    regions: list[list[float]]      # all speech regions (file time), padded by speech_pad_ms
    gated: list[list[float]]        # regions dropped by the RMS gate
    groups: list[Group] = field(default_factory=list)
    parts: int = 1
    raw_regions: list[list[float]] = field(default_factory=list)   # unpadded Silero speech (for boundary snapping)
    # true stereo only: each channel's own unpadded speech. The union above mixes the two speakers,
    # and a channel-1 region read against a channel-0 group is a false "skipped speech" or a wrong
    # snap target (review of 2026-09-26)
    raw_by_channel: list[list[list[float]]] = field(default_factory=list)

    def raw_for(self, channel: int) -> list[list[float]]:
        """The raw speech regions that belong to ``channel`` (the whole file for a mono plan)."""
        if 0 <= channel < len(self.raw_by_channel):
            return self.raw_by_channel[channel]
        return self.raw_regions

    def to_json(self) -> dict:
        return {"source": self.source, "params": self.params, "duration_s": round(self.duration_s, 3),
                "speech_s": round(self.speech_s, 3), "regions": self.regions, "gated": self.gated,
                "parts": self.parts, "groups": [asdict(g) for g in self.groups], "raw_regions": self.raw_regions,
                "raw_by_channel": self.raw_by_channel}

    @classmethod
    def from_json(cls, d: dict) -> Plan:
        p = cls(source=d["source"], params=d["params"], duration_s=d["duration_s"], speech_s=d["speech_s"],
                regions=d["regions"], gated=d.get("gated", []), parts=d.get("parts", 1),
                raw_regions=d.get("raw_regions", []), raw_by_channel=d.get("raw_by_channel", []))
        p.groups = [Group(**g) for g in d["groups"]]
        return p


def silero_regions(audio: np.ndarray, params: dict | None = None) -> list[list[float]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    params = {**VAD_PARAMS, **(params or {})}
    opts = VadOptions(**{k: v for k, v in params.items() if v is not None})
    ts = get_speech_timestamps(audio, opts, sampling_rate=SR)
    return [[round(t["start"] / SR, 3), round(t["end"] / SR, 3)] for t in ts]


def rms_gate(audio: np.ndarray, regions: list[list[float]], gate: float = RMS_GATE) -> tuple[list[list[float]], list[list[float]]]:
    keep, dropped = [], []
    for s, e in regions:
        seg = audio[int(s * SR): int(e * SR)]
        rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) if seg.size else 0.0
        (keep if rms >= gate else dropped).append([s, e])
    return keep, dropped


def group_regions(regions: list[list[float]], duration_s: float, *, pad_s: float = PAD_S,
                  split_silence_s: float = GROUP_SPLIT_SILENCE_S, group_max_s: float = GROUP_MAX_S,
                  part_max_s: float = PART_MAX_S) -> tuple[list[Group], int]:
    """Group speech regions into decoding slices.

    A group ends at a silence >= ``split_silence_s``. When a group would outgrow ``group_max_s`` in
    continuous speech it is cut at the widest pause in its second half, not in front of the next
    region: Silero cuts regions longer than ``max_speech_duration_s`` with no gap at all, and with
    group decoding a cut there can split a word across two slices (the word's first half ends one
    group and the whole word begins the next). Padded slices of neighbouring groups never overlap -
    where the pads would, both stop at the midpoint between the two groups' speech - because a
    group is decoded whole and overlapping audio is transcribed twice.
    """
    groups: list[Group] = []
    cur: list[list[float]] = []
    parts = 1
    part_start = 0.0

    def flush(upto: int | None = None) -> None:
        """Emit ``cur[:upto]`` (all of it by default) as a group; the rest starts the next one."""
        nonlocal cur
        if not cur:
            return
        head, cur = (cur, []) if upto is None else (cur[:upto], cur[upto:])
        s = max(0.0, head[0][0] - pad_s)
        e = min(duration_s, head[-1][1] + pad_s)
        groups.append(Group(index=len(groups), start_s=round(s, 3), end_s=round(e, 3),
                            clips=[[round(a, 3), round(b, 3)] for a, b in head], part=parts - 1))

    def widest_pause() -> tuple[int | None, float]:
        """(index in ``cur`` to cut before, pause length) over the second half of the group."""
        best, best_gap = None, 0.0
        for k in range(1, len(cur)):
            gap = cur[k][0] - cur[k - 1][1]
            if cur[k][0] - cur[0][0] >= group_max_s / 2 and gap > best_gap:
                best, best_gap = k, gap
        return best, best_gap

    for s, e in regions:
        if cur:
            gap = s - cur[-1][1]
            if gap >= split_silence_s:
                flush()
            elif e - cur[0][0] > group_max_s:
                k, pause = widest_pause()
                flush(k if k is not None and pause > gap else None)
            if duration_s >= LONG_FILE_S and s - part_start >= part_max_s and gap >= split_silence_s:
                parts += 1
                part_start = s
        cur.append([s, e])
    flush()
    for a, b in zip(groups, groups[1:]):
        if a.end_s > b.start_s:
            mid = round((a.clips[-1][1] + b.clips[0][0]) / 2, 3)
            a.end_s, b.start_s = mid, mid
    return groups, parts


def decode_clips(g: Group, mode: str = CLIP_MODE) -> list[tuple[float, float]] | None:
    """Clip timestamps for the backend, relative to the group's slice.

    ``group`` (default): none - the whole padded group slice is decoded as one stretch and Whisper
    slides its 30-s windows over continuous speech, as it was trained. ``regions``: one clip per
    Silero region, which faster-whisper decodes as separate zero-context, 30-s-padded calls; the
    phase-4 ablation on eval-whatsapp measured that at +1.8 WER points and
    2.5x the wall time. The group's ``clips`` stay the speech regions for bookkeeping either way.
    """
    if mode == "group":
        return None
    if mode == "regions":
        return [(max(0.0, a - g.start_s), max(0.0, b - g.start_s)) for a, b in g.clips]
    raise ValueError(f"unknown clip mode {mode!r}")


def uncovered_speech(raw_regions: list[list[float]], segments: list[tuple[float, float]], g: Group,
                     *, min_s: float, cover: float = RECOVER_COVER,
                     join_gap_s: float = GROUP_SPLIT_SILENCE_S) -> list[tuple[float, float]]:
    """Raw speech regions of group ``g`` that the decoded segments miss (file time).

    Whisper stops a 30-s window early now and then and the decoder seeks to the window end, so the
    speech in the tail is never decoded (synthetic tier A, 2026-09-22: a 4.3-s clip vanished at
    103-108 s of a 300-s group). Regions are clipped to the group's own speech span - the first
    clip's start to the last clip's end - before anything is measured: a raw region that straddles
    into a neighbouring group is that group's speech, not a miss (without the clip, such regions
    triggered recoveries that inserted words decoded from ~0.4 s of audio). Clipped regions of at
    least ``min_s`` whose overlap with the segments is below ``cover`` of their length are returned,
    adjacent ones (gap < ``join_gap_s``) merged, so the caller can decode them on their own.
    """
    if not g.clips:
        return []
    lo, hi = g.clips[0][0], g.clips[-1][1]
    out: list[tuple[float, float]] = []
    for s, e in raw_regions:
        s, e = max(s, lo), min(e, hi)
        if e - s < min_s:
            continue
        covered = sum(max(0.0, min(e, b) - max(s, a)) for a, b in segments)
        if covered < cover * (e - s):
            if out and s - out[-1][1] < join_gap_s:
                out[-1] = (out[-1][0], max(out[-1][1], e))   # a nested region never shortens a span
            else:
                out.append((s, e))
    return out


RAW_VAD_PARAMS = {"speech_pad_ms": 0, "min_silence_duration_ms": 100, "min_speech_duration_ms": 100}


def build_plan(audio: np.ndarray, *, params: dict | None = None, source: str = "silero+rms") -> Plan:
    duration_s = len(audio) / SR
    regions = silero_regions(audio, params)
    kept, gated = rms_gate(audio, regions)
    groups, parts = group_regions(kept, duration_s)
    # a second, unpadded pass gives the physical speech onsets/offsets that snap_to_speech() uses:
    # Whisper places the first word of a clip at the clip start, i.e. inside the 300 ms pad
    raw = silero_regions(audio, {**(params or {}), **RAW_VAD_PARAMS})
    return Plan(source=source, params={**VAD_PARAMS, **(params or {}), "rms_gate": RMS_GATE, "pad_s": PAD_S,
                                       "group_split_silence_s": GROUP_SPLIT_SILENCE_S, "group_max_s": GROUP_MAX_S,
                                       "clip_mode": CLIP_MODE, "raw_vad": RAW_VAD_PARAMS},
                duration_s=duration_s, speech_s=sum(e - s for s, e in kept), regions=kept, gated=gated,
                groups=groups, parts=parts, raw_regions=raw)


SNAP_MAX_MS = 450      # offsets: pad (300) + Silero's own lag; a word end further past a raw offset is not padding
SNAP_TOL_MS = 300      # onsets: a word may "start" this far inside the tail of the previous raw region and still be
                       # the first word after the gap (Whisper's DTW starts a word where the previous one ended)
SNAP_MIN_GAP_MS = 250  # ... provided the silence it spans is at least this long
SNAP_LATE_MS = 300     # the first word after a gap that starts this late is pulled back to the raw onset


def snap_to_speech(words: list[dict], raw_regions: list[list[float]], *, max_ms: int = SNAP_MAX_MS,
                   tol_ms: int = SNAP_TOL_MS, min_gap_ms: int = SNAP_MIN_GAP_MS, late_ms: int = SNAP_LATE_MS) -> int:
    """Pull word boundaries out of silence to the unpadded (raw Silero) speech edges.

    Whisper places the first word after a pause where the previous word ended, i.e. the word swallows
    the silence, and with group decoding (``decode_clips``) that silence can be seconds long. Rules,
    words being ms ints in file time:

    * a word that starts in a gap - or no more than ``tol_ms`` inside the tail of the previous raw
      region - and ends past the next raw onset, the gap being at least ``min_gap_ms``, starts at
      that onset (no distance cap: nothing is spoken in the gap);
    * the first word after a gap that starts up to ``late_ms`` *after* the raw onset starts at the onset;
    * a word ending up to ``max_ms`` past a raw offset ends at that offset.

    On synthetic tier A, group decoding alone took onsets within 100 ms from 93 % to
    78 %; these rules bring it to 94 % (99 % within 200 ms). Returns how many words moved and flags
    each with ``snap``: "s", "e" or "se"."""
    if not raw_regions or not words:
        return 0
    starts = [int(round(s * 1000)) for s, _ in raw_regions]
    ends = [int(round(e * 1000)) for _, e in raw_regions]
    moved = 0
    import bisect
    spoken = [w for w in words if w.get("t_raw", w.get("t", "")).strip()]
    for idx, w in enumerate(spoken):
        flag = ""
        k = bisect.bisect_left(starts, w["s"])            # next raw onset at or after the word start
        j = bisect.bisect_right(starts, w["s"]) - 1       # raw region containing or preceding the word start
        if k < len(starts) and w["e"] > starts[k] and starts[k] > w["s"]:
            in_gap = j < 0 or ends[j] - w["s"] <= tol_ms
            gap = starts[k] - (ends[j] if j >= 0 else 0)
            if in_gap and gap >= min_gap_ms:
                w["s"] = starts[k]
                flag += "s"
        elif j >= 0 and w["s"] < ends[j] and 0 < w["s"] - starts[j] <= late_ms:
            if idx == 0 or spoken[idx - 1]["e"] <= starts[j]:
                w["s"] = starts[j]
                flag += "s"
        # nearest raw offset at or before the word end
        k = bisect.bisect_right(ends, w["e"]) - 1
        if k >= 0 and w["s"] < ends[k] and 0 < w["e"] - ends[k] <= max_ms:
            j = bisect.bisect_right(starts, w["e"] - 1) - 1
            if j < 0 or starts[j] <= ends[k]:     # the word end is not inside a later raw region
                w["e"] = max(ends[k], w["s"] + 1)
                flag += "e"
        if flag:
            w["snap"] = flag
            moved += 1
    return moved


def snap_words(words: list[dict], plan: Plan) -> int:
    """``snap_to_speech`` against the right raw regions: each channel's own for a true-stereo plan."""
    if not plan.raw_by_channel:
        return snap_to_speech(words, plan.raw_regions)
    return sum(snap_to_speech([w for w in words if w.get("channel", -1) == ch], raw)
               for ch, raw in enumerate(plan.raw_by_channel))


def build_stereo_plan(channels: list[np.ndarray], *, params: dict | None = None) -> Plan:
    """One plan per channel, merged: groups carry their channel; ASR decodes each from its own channel."""
    plans = [build_plan(c, params=params, source="silero+rms/stereo") for c in channels]
    merged = Plan(source="silero+rms/stereo", params=plans[0].params, duration_s=max(p.duration_s for p in plans),
                  speech_s=sum(p.speech_s for p in plans), regions=[r for p in plans for r in p.regions],
                  gated=[r for p in plans for r in p.gated], parts=max(p.parts for p in plans),
                  raw_regions=sorted(r for p in plans for r in p.raw_regions),
                  raw_by_channel=[p.raw_regions for p in plans])
    groups: list[Group] = []
    for ch, p in enumerate(plans):
        for g in p.groups:
            groups.append(Group(index=len(groups), start_s=g.start_s, end_s=g.end_s, clips=g.clips, part=g.part, channel=ch))
    groups.sort(key=lambda g: (g.start_s, g.channel))
    for k, g in enumerate(groups):
        g.index = k
    merged.groups = groups
    merged.regions.sort()
    return merged


def plan_from_regions(regions: list[list[float]], duration_s: float, source: str) -> Plan:
    """A plan from externally supplied speech regions (e.g. --diarize-first, tests)."""
    groups, parts = group_regions(regions, duration_s)
    return Plan(source=source, params={"pad_s": PAD_S, "group_split_silence_s": GROUP_SPLIT_SILENCE_S,
                                       "group_max_s": GROUP_MAX_S}, duration_s=duration_s,
                speech_s=sum(e - s for s, e in regions), regions=regions, gated=[], groups=groups, parts=parts)


def save_plan(plan: Plan, path: pathlib.Path) -> str:
    text = json.dumps(plan.to_json(), ensure_ascii=False, indent=1)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def load_plan(path: pathlib.Path) -> tuple[Plan, str]:
    text = path.read_text(encoding="utf-8")
    return Plan.from_json(json.loads(text)), hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
