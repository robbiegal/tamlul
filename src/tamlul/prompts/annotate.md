# Annotation contract (tamlul {version}, schema {schema})

You are annotating `{run_id}`. Read **`{input_path}`** (the turn-level rendering,
the candidate lists and the stats you may quote). Then write
**`{output_path}`** and run `annotate apply`.

## What you produce

`annotations.json`, one object. Every field is optional; omit what the
recording does not support. Every item carries `evidence`: a list of anchors.

```json
{{
  "model": "<the model you are>",
  "summary": {{"text": "...", "evidence": [ANCHOR, ...]}},
  "speaker_names": [{{"speaker": "S1", "name": "דנה", "confidence": 0.9, "reason": "self-introduction", "evidence": [ANCHOR]}}],
  "speaker_summaries": [{{"speaker": "S1", "text": "...", "evidence": [ANCHOR]}}],
  "topics": [{{"title": "...", "s": ms, "e": ms, "evidence": [ANCHOR]}}],
  "decisions": [{{"text": "...", "evidence": [ANCHOR]}}],
  "action_items": [{{"text": "...", "owner": "S2", "due": "", "status": "open", "evidence": [ANCHOR]}}],
  "questions": [{{"text": "...", "asked_by": "S1", "answered": true, "answer_evidence": [ANCHOR], "evidence": [ANCHOR]}}],
  "key_quotes": [{{"text": "...", "speaker": "S1", "evidence": [ANCHOR]}}],
  "term_restorations": [{{"w": [i, i], "text": "Gemini", "evidence": [ANCHOR]}}],
  "boundary_corrections": [{{"turn": 12, "move_words": -2, "to": "S2", "reason": "...", "evidence": [ANCHOR]}}]
}}
```

An **ANCHOR** is either
- `{{"turn": <turn id>, "quote": "<verbatim words from that turn>"}}` - the CLI
  finds the word range for you (the quote must be an exact run of words as
  rendered in the input, 1-40 words), or
- `{{"w": [i0, i1], "quote": "..."}}` - an explicit raw word index range.

Rules (the CLI enforces them; anything failing is rejected with a reason):

1. Never invent quotes. Copy words exactly as they appear in the input.
2. Speaker names come only from evidence in the transcript (self-introduction,
   vocative, role, gender agreement) or from the participant list; when a
   participant list is given, use only those names.
3. `term_restorations` may target only the words listed under
   "code-switch candidates" (`w` must be one of them); write the English term
   in Latin script.
4. `boundary_corrections` move a speaker change by at most 3 words and never
   change text or timestamps.
5. Do not state numbers you did not read from `stats` / the speaker table.
6. Hebrew stays Hebrew; do not translate; write in the language of the recording.
7. Empty lists are fine. Omit `summary` for recordings under 30 seconds.

## Then

```
{apply_cmd}
```

If items are rejected, read the reasons, fix `{output_path}` once, apply again,
and report what remained rejected.
