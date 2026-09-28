"""``tamlul eval ...`` - the developer evaluation harness entry point (design §4.15, §9).

    tamlul eval sets                                  what exists, what is downloaded, publication rules
    tamlul eval download <set> [--token T]            gated sets need the token once (or HF_TOKEN)
    tamlul eval lb-check                              scorer check against the leaderboard's own CSVs
    tamlul eval run <set> [--model ID] [--column plugin|staff] [--compute-type int8|float32]
                                     [--threads N] [--limit N] [--tag T] [--temperature0]
    tamlul eval score <csv> [<csv> ...]               metrics + bootstrap CIs
    tamlul eval compare <csv-a> <csv-b> [--key wer]   paired bootstrap on shared ids (b - a)
    tamlul eval grid [--sets a,b] [--models ...] [--threads 8]   int8 vs float32 -> compute_type decision (§1.3)
    tamlul eval synth <set> --name N [--minutes 60] [--noise-dir D] ...
    tamlul eval score-ts <run-id|run-dir> <synth-name|dir>
    tamlul eval score-sa <transcript.json> <ref.rttm|ref.json> [--collar 0.25]
    tamlul eval cosih download|convert|run|score [ids]   CoSIH conversation tier (DER/cpWER/tier-B onsets)
    tamlul eval publish <csv> <set> <dest.csv>        strip references per the publication rules
    tamlul eval rehydrate <published.csv> <set> <dest.csv>

Everything writes under <data>/eval/. Online commands (download, lb-check) are
allowed by the env block (ONLINE_COMMANDS includes "eval").
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

from .paths import DataLayout

EXIT_OK, EXIT_ERROR = 0, 2


def _p(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tamlul eval", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="op", required=True)
    sub.add_parser("sets")
    d = sub.add_parser("download")
    d.add_argument("set")
    d.add_argument("--token", default=None)
    lb = sub.add_parser("lb-check")
    lb.add_argument("--revision", default=None)
    r = sub.add_parser("run")
    r.add_argument("set")
    r.add_argument("--model", default="asr/turbo-ct2@2025.05.13")
    r.add_argument("--column", default="plugin", choices=["plugin", "staff"])
    r.add_argument("--compute-type", dest="compute_type", default="int8")
    r.add_argument("--threads", type=int, default=0)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--tag", default="")
    r.add_argument("--temperature0", action="store_true")
    r.add_argument("--opt", action="append", default=[], metavar="FIELD=VALUE",
                   help="override a DecodeOpts field (ablation); repeatable")
    r.add_argument("--no-plan", dest="plan", action="store_false", help="decode the whole utterance, no plan groups")
    r.add_argument("--clip-mode", dest="clip_mode", default="group", choices=["group", "regions"],
                   help="per-group clip policy handed to the backend (regions = the pre-2026-09-21 behaviour)")
    r.add_argument("--join", default="words", choices=["words", "text"], help="hypothesis from words[] or segment text")
    r.add_argument("--itn", action="store_true", help="apply the pipeline's ITN to the hypothesis")
    r.add_argument("--no-recover", dest="recover", action="store_false",
                   help="skip the pipeline's re-decoding of raw speech the decoder missed")
    s = sub.add_parser("score")
    s.add_argument("csv", nargs="+")
    s.add_argument("--resamples", type=int, default=10_000)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--key", default="wer")
    c.add_argument("--resamples", type=int, default=10_000)
    g = sub.add_parser("grid")
    g.add_argument("--sets", default="eval-whatsapp,eval-d1")
    g.add_argument("--models", default="asr/turbo-ct2@2025.04.03,asr/turbo-ct2@2025.05.13,asr/large-v3-ct2@2025.05.13")
    g.add_argument("--threads", type=int, default=8)
    g.add_argument("--column", default="plugin", choices=["plugin", "staff"])
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--resamples", type=int, default=10_000)
    sy = sub.add_parser("synth")
    sy.add_argument("set")
    sy.add_argument("--name", required=True)
    sy.add_argument("--minutes", type=float, default=60.0)
    sy.add_argument("--seed", type=int, default=0)
    sy.add_argument("--interleave", type=float, default=0.2)
    sy.add_argument("--noise-fraction", dest="noise_fraction", type=float, default=0.2)
    sy.add_argument("--noise-dir", dest="noise_dir", default=None)
    sy.add_argument("--snr-db", dest="snr_db", type=float, default=10.0)
    sy.add_argument("--speaker-from", dest="speaker_from", default="auto", choices=["auto", "column", "id"])
    sy.add_argument("--max-disagree-ms", dest="max_disagree_ms", type=float, default=30.0,
                    help="keep a clip only when the bias-corrected energy and Silero boundaries agree this closely")
    st = sub.add_parser("score-ts")
    st.add_argument("run")
    st.add_argument("synth")
    sa = sub.add_parser("score-sa")
    sa.add_argument("transcript")
    sa.add_argument("reference", help="RTTM (turn labels) or a JSON list of {s,e,spk,text} turns")
    sa.add_argument("--collar", type=float, default=0.25)
    co = sub.add_parser("cosih", help="CoSIH conversation tier: download | convert | run | score")
    co.add_argument("step", choices=["download", "convert", "run", "score"])
    co.add_argument("ids", nargs="*", help="recording ids (default: all 31)")
    co.add_argument("--profile", default="meeting")
    co.add_argument("--speakers-hint", dest="speakers_hint", action="store_true",
                    help="pass the reference speaker count to the pipeline (oracle count; reported as such)")
    pu = sub.add_parser("publish")
    pu.add_argument("csv")
    pu.add_argument("set")
    pu.add_argument("dest")
    rh = sub.add_parser("rehydrate")
    rh.add_argument("csv")
    rh.add_argument("set")
    rh.add_argument("dest")
    return ap


def _resolve_run(layout: DataLayout, ref: str) -> pathlib.Path:
    p = pathlib.Path(ref)
    if p.is_dir():
        return p
    from .jobs import RunDir
    return RunDir.open(layout, ref).path


def _resolve_synth(layout: DataLayout, ref: str) -> pathlib.Path:
    p = pathlib.Path(ref)
    if p.is_dir():
        return p
    from .eval.sets import eval_root
    return eval_root(layout) / "synth" / ref


def _grid(a: argparse.Namespace, layout: DataLayout) -> int:
    """{models} x {int8, float32} on the named sets; a model ships float32 only when int8 is worse
    by a point delta > 0 whose paired-bootstrap 95 % CI excludes 0 on any set (design §1.3)."""
    from .eval import runasr, score
    from .eval.sets import eval_root
    sets = [x for x in a.sets.split(",") if x]
    models = [x for x in a.models.split(",") if x]
    stamp = time.strftime("%Y-%m-%d")
    out_dir = eval_root(layout) / "results" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    grid: dict = {"date": stamp, "threads": a.threads, "column": a.column, "sets": sets, "models": {}}
    for m in models:
        grid["models"][m] = {"sets": {}, "decision": "int8", "reason": []}
        for s in sets:
            paths = {}
            for ct in ("int8", "float32"):
                paths[ct] = runasr.run(s, m, layout, column=a.column, compute_type=ct, threads=a.threads, limit=a.limit)
            sc = {ct: score.score_file(p, resamples=a.resamples) for ct, p in paths.items()}
            diff = score.compare_files(paths["float32"], paths["int8"], key="wer", resamples=a.resamples)  # b=int8 - a=float32
            grid["models"][m]["sets"][s] = {"int8": sc["int8"], "float32": sc["float32"], "int8_minus_float32": diff}
            lo, hi = diff.get("diff_ci95", [0, 0])
            if diff.get("diff", 0) > 0 and lo > 0:
                grid["models"][m]["decision"] = "float32"
                grid["models"][m]["reason"].append(f"{s}: int8 worse by {diff['diff']:.4f} (CI {lo:.4f}..{hi:.4f})")
        if grid["models"][m]["decision"] == "int8":
            grid["models"][m]["reason"].append("int8 not measurably worse on any set (CI includes 0 or delta <= 0)")
    (out_dir / "grid.json").write_text(json.dumps(grid, ensure_ascii=False, indent=2), encoding="utf-8")
    _p({m: {"decision": v["decision"], "reason": v["reason"],
            "wer": {s: {ct: v["sets"][s][ct]["wer"] for ct in ("int8", "float32")} for s in v["sets"]}}
        for m, v in grid["models"].items()})
    print(f"grid -> {out_dir / 'grid.json'}; record the decisions in models.json (`compute_type_decision`)",
          file=sys.stderr)
    return EXIT_OK


def _cosih(a: argparse.Namespace, layout: DataLayout) -> int:
    from .eval import cosih
    ids = a.ids or cosih.IDS
    if a.step == "download":
        cosih.download(layout, ids)
        return EXIT_OK
    if a.step == "convert":
        _p({rid: cosih.stats(cosih.load_reference(layout, rid)) for rid in ids})
        return EXIT_OK
    if a.step == "run":
        from .cli import main as tamlul_main
        rc = EXIT_OK
        for rid in ids:
            d = cosih.cosih_dir(layout) / rid
            wav = d / f"{rid}.wav"
            if not wav.is_file():
                print(f"{rid}: no audio (run `eval cosih download {rid}`)", file=sys.stderr)
                rc = EXIT_ERROR
                continue
            argv = ["--data", str(layout.root), "transcribe", str(wav), "--out", str(d / "run"), "--profile", a.profile,
                    "--diarization", "on"]
            if a.speakers_hint:
                argv += ["--speakers", str(len(cosih.load_reference(layout, rid)["speakers"]))]
            r = tamlul_main(argv)
            rc = max(rc, r)
        return rc
    if a.step == "score":
        results = []
        for rid in ids:
            tj = cosih.cosih_dir(layout) / rid / "run" / "transcript.json"
            if not tj.is_file():
                print(f"{rid}: no transcript (run `eval cosih run {rid}`)", file=sys.stderr)
                continue
            doc = json.loads(tj.read_text(encoding="utf-8"))
            res = cosih.score(doc, cosih.load_reference(layout, rid))
            res["oracle_speaker_count"] = bool(a.speakers_hint)
            (cosih.cosih_dir(layout) / rid / "score.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                                                      encoding="utf-8")
            results.append(res)
        agg = cosih.aggregate(results)
        (cosih.cosih_dir(layout) / "aggregate.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2),
                                                                encoding="utf-8")
        _p({"per_file": results, "aggregate": agg})
        return EXIT_OK
    return EXIT_ERROR


def eval_command(args: argparse.Namespace, layout: DataLayout) -> int:
    argv = [args.op] + list(args.rest or [])
    a = _parser().parse_args(argv)
    layout.ensure()
    if a.op == "sets":
        from .eval.sets import SETS, is_downloaded
        _p({k: {"repo": v.repo, "split": v.split, "gated": v.gated, "licence": v.licence,
                "publish_refs": v.publish_refs, "publishable": v.publishable, "downloaded": is_downloaded(k, layout),
                "note": v.note} for k, v in SETS.items()})
        return EXIT_OK
    if a.op == "download":
        from .eval.sets import download
        token = a.token or os.environ.get("HF_TOKEN") or None
        _p(download(a.set, layout, token=token))
        return EXIT_OK
    if a.op == "lb-check":
        from .eval.lb import check_all
        rep = check_all(layout, revision=a.revision)
        _p({k: v for k, v in rep.items() if k != "reports"})
        for r in rep["reports"]:
            if "skipped" in r:
                continue
            flag = "ok " if r["pass"] else "FAIL"
            theirs = r["theirs_corpus_wer_from_sdih"]
            print(f"{flag} {pathlib.Path(r['file']).parent.name}/{pathlib.Path(r['file']).name}: rows={r['rows']} "
                  f"ours={r['ours_corpus_wer']:.4f} theirs={theirs if theirs is None else f'{theirs:.4f}'} "
                  f"norm_mismatch={r['norm_text_mismatch_rows']} edits_mismatch={r['edits_mismatch_rows']} "
                  f"sdi_split_differs={r['sdi_split_differs_rows']}"
                  + (" [stale normalizer in the published CSV: bidi marks kept]" if r["stale_normalizer_in_csv"] else ""),
                  file=sys.stderr)
        return EXIT_OK if rep["checked"] and not rep["failed"] else EXIT_ERROR
    if a.op == "run":
        from .eval import runasr
        p = runasr.run(a.set, a.model, layout, column=a.column, compute_type=a.compute_type, threads=a.threads,
                       limit=a.limit, tag=a.tag, temperature0=a.temperature0,
                       overrides=runasr.parse_overrides(a.opt), plan=a.plan, join=a.join, itn=a.itn,
                       clip_mode=a.clip_mode, recover=a.recover)
        from .eval import score
        _p({"csv": str(p), **score.score_file(p, resamples=2000)})
        return EXIT_OK
    if a.op == "score":
        from .eval import score
        _p({c: score.score_file(pathlib.Path(c), resamples=a.resamples) for c in a.csv})
        return EXIT_OK
    if a.op == "compare":
        from .eval import score
        _p(score.compare_files(pathlib.Path(a.a), pathlib.Path(a.b), key=a.key, resamples=a.resamples))
        return EXIT_OK
    if a.op == "grid":
        return _grid(a, layout)
    if a.op == "synth":
        from .eval import synth
        synth.build(a.set, layout, name=a.name, minutes=a.minutes, seed=a.seed, interleave_fraction=a.interleave,
                    noise_fraction=a.noise_fraction, noise_dir=pathlib.Path(a.noise_dir) if a.noise_dir else None,
                    snr_db=a.snr_db, speaker_from=a.speaker_from, max_disagree_ms=a.max_disagree_ms)
        return EXIT_OK
    if a.op == "score-ts":
        from .eval import timestamps
        _p(timestamps.score_run(_resolve_run(layout, a.run), _resolve_synth(layout, a.synth)))
        return EXIT_OK
    if a.op == "score-sa":
        from .eval import speakers
        doc = json.loads(pathlib.Path(a.transcript).read_text(encoding="utf-8"))
        ref_path = pathlib.Path(a.reference)
        if ref_path.suffix.lower() == ".rttm":
            ref = speakers.rttm_turns(ref_path.read_text(encoding="utf-8"))
        else:
            ref = json.loads(ref_path.read_text(encoding="utf-8"))
        hyp = speakers.doc_turns(doc)
        out = {"der_collar0": speakers.der(ref, hyp, collar_s=0.0), "der_collar": speakers.der(ref, hyp, collar_s=a.collar),
               "speaker_count": speakers.speaker_count(len({t["spk"] for t in ref}), len({t["spk"] for t in hyp}))}
        if all("text" in t for t in ref):
            out["cpwer"] = speakers.cpwer(speakers.streams_from_turns(ref), speakers.streams_from_turns(hyp))
        _p(out)
        return EXIT_OK
    if a.op == "cosih":
        return _cosih(a, layout)
    if a.op == "publish":
        from .eval import runasr
        _p({"dest": str(runasr.strip_for_publication(pathlib.Path(a.csv), a.set, pathlib.Path(a.dest)))})
        return EXIT_OK
    if a.op == "rehydrate":
        from .eval import runasr
        _p({"dest": str(runasr.rehydrate(pathlib.Path(a.csv), a.set, layout, pathlib.Path(a.dest)))})
        return EXIT_OK
    return EXIT_ERROR
