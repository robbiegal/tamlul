"""tamlul command line (design §4.17). Never calls a language model.

Exit codes: 0 success, 2+ error. Never 1 - the Bash tool treats it specially.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import __version__
from .paths import DataLayout, data_dir

EXIT_OK = 0
EXIT_ERROR = 2
EXIT_USAGE = 3
EXIT_NOT_READY = 4


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", help="data directory (default: TAMLUL_DATA / CLAUDE_PLUGIN_DATA / official id)")
    p.add_argument("--json", action="store_true", help="machine-readable output where supported")


def _add_settings(p: argparse.ArgumentParser) -> None:
    """Flags that feed the configuration precedence (§4.16). Empty strings mean unset."""
    p.add_argument("--profile", default=None, help="meeting|voice-note|interview|ca|legal|auto")
    p.add_argument("--quality", default=None, help="default|max")
    p.add_argument("--compute-type", dest="compute_type", default=None, help="auto|int8|float32")
    p.add_argument("--threads", default=None, help="CPU threads for ASR (0 = physical cores)")
    p.add_argument("--diarization", default=None, help="auto|on|off")
    p.add_argument("--models-dir", dest="models_dir", default=None)
    p.add_argument("--channels", default=None, help="auto|mono|split")
    p.add_argument("--fast", action="store_const", const=True, default=None,
                   help="batched decoding (opt-in; flagged in the run)")
    p.add_argument("--verbatim", action="store_const", const="verbatim", default=None)
    p.add_argument("--clean", dest="verbatim", action="store_const", const="clean")
    p.add_argument("--glossary", default=None)
    p.add_argument("--participants", default=None)
    p.add_argument("--speakers", default=None, help="exact speaker count hint")
    p.add_argument("--min-speakers", dest="min_speakers", default=None)
    p.add_argument("--max-speakers", dest="max_speakers", default=None)
    p.add_argument("--fast-embeddings", dest="fast_embeddings", default=None, help="auto|on|off")
    p.add_argument("--align", default=None, help="dtw|ctc")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tamlul", description="Local Hebrew conversation transcription and annotation.")
    ap.add_argument("--version", action="version", version=f"tamlul {__version__}")
    _add_common(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("status", help="environment, models, calibration and config status (JSON)")
    st.add_argument("--json", action="store_true", help="accepted after the subcommand too (the skills spell it that way)")
    sub.add_parser("doctor", help="human-readable health report incl. the telemetry switch table")

    t = sub.add_parser("transcribe", help="transcribe a recording or a folder")
    t.add_argument("media", nargs="?", default=None, help="audio/video file or a folder of them (omit with --resume)")
    t.add_argument("--out", default=None, help="run directory (default: <data>/runs/<id>)")
    t.add_argument("--resume", default=None, help="resume run id")
    t.add_argument("--yes", action="store_true", help="allow model downloads if missing")
    t.add_argument("--formats", default=None, help="comma-separated export formats")
    _add_settings(t)

    a = sub.add_parser("annotate", help="prepare the agent's input, or apply its annotations")
    asub = a.add_subparsers(dest="acmd", required=True)
    ap1 = asub.add_parser("prepare", help="write the rendering the annotator reads")
    ap1.add_argument("run")
    ap1.add_argument("--participants", default=None)
    ap1.add_argument("--glossary", default=None)
    ap2 = asub.add_parser("apply", help="validate annotations.json and merge it")
    ap2.add_argument("run")
    ap2.add_argument("annotations")

    s = sub.add_parser("speakers", help="review and edit the speaker overlay")
    s.add_argument("run")
    s.add_argument("op", nargs="?", default="show", help="show|rename|merge|reassign|split")
    s.add_argument("args", nargs="*")

    e = sub.add_parser("export", help="(re)write exports for a run")
    e.add_argument("run")
    e.add_argument("--formats", default=None)
    e.add_argument("--beside-media", action="store_true")

    v = sub.add_parser("validate", help="validate transcript.json against the schema")
    v.add_argument("run")
    m = sub.add_parser("migrate", help="migrate a run to the current schema")
    m.add_argument("run")
    pr = sub.add_parser("progress", help="progress of a run")
    pr.add_argument("run")

    r = sub.add_parser("runs", help="list or prune runs")
    r.add_argument("op", nargs="?", default="list", help="list|prune")
    r.add_argument("--older-than-days", type=int, default=30)

    sub.add_parser("env", help="(handled by scripts/tamlul.py; here for completeness)")

    mo = sub.add_parser("models", help="list, plan, install, verify or remove pinned models")
    mo.add_argument("op", nargs="?", default="list", help="list|plan|install|verify|remove")
    mo.add_argument("ids", nargs="*", help="artefact ids, or none for the profile's set")
    mo.add_argument("--profile", default=None)
    mo.add_argument("--quality", default=None)
    mo.add_argument("--diarization", default=None)
    mo.add_argument("--all", action="store_true")
    mo.add_argument("--yes", action="store_true", help="required for any download")
    mo.add_argument("--dry-run", action="store_true")
    mo.add_argument("--no-calibrate", action="store_true", help="skip the automatic `calibrate --quick` after install")

    c = sub.add_parser("calibrate", help="measure speed, engine disagreement and silence behaviour")
    c.add_argument("--quick", action="store_true")
    c.add_argument("--full", action="store_true")
    _add_settings(c)

    cf = sub.add_parser("config", help="get/set persistent config in <data>/config.json")
    cf.add_argument("op", nargs="?", default="show", help="show|get|set|unset")
    cf.add_argument("key", nargs="?")
    cf.add_argument("value", nargs="?")

    ev = sub.add_parser("eval", help="developer: evaluation harness")
    ev.add_argument("op", help="download|run|score")
    ev.add_argument("rest", nargs=argparse.REMAINDER)
    return ap


def _layout(args: argparse.Namespace) -> DataLayout:
    return DataLayout(data_dir(getattr(args, "data", None)))


def _settings_from(args: argparse.Namespace) -> dict[str, Any]:
    from .config import KEYS
    return {k: getattr(args, k, None) for k in KEYS if getattr(args, k, None) is not None}


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    layout = _layout(args)

    # Privacy-hygiene block for THIS process only, before any heavy import.
    from .envblock import ONLINE_COMMANDS, apply_to_current_process
    offline = args.cmd not in ONLINE_COMMANDS
    apply_to_current_process(layout, offline=offline)

    try:
        return dispatch(args, layout)
    except KeyboardInterrupt:
        return 130
    except NotReady as exc:
        print(f"tamlul: {exc}", file=sys.stderr)
        return EXIT_NOT_READY
    except UserError as exc:
        print(f"tamlul: {exc}", file=sys.stderr)
        return EXIT_USAGE


class UserError(Exception):
    pass


class NotReady(Exception):
    pass


def dispatch(args: argparse.Namespace, layout: DataLayout) -> int:
    cmd = args.cmd
    if cmd == "status":
        from .doctor import status
        print(json.dumps(status(layout), indent=2, ensure_ascii=False))
        return EXIT_OK
    if cmd == "doctor":
        from .doctor import doctor_report
        print(doctor_report(layout))
        return EXIT_OK
    if cmd == "env":
        print("tamlul: use `python scripts/tamlul.py env install|status|upgrade` (host interpreter).",
              file=sys.stderr)
        return EXIT_USAGE
    if cmd == "models":
        return _models(args, layout)
    if cmd == "config":
        return _config(args, layout)
    if cmd == "runs":
        from .jobs import list_runs, prune_runs
        if args.op == "prune":
            removed = prune_runs(layout, args.older_than_days)
            print(json.dumps({"removed": removed}, indent=2))
            return EXIT_OK
        print(json.dumps(list_runs(layout), indent=2, ensure_ascii=False))
        return EXIT_OK
    if cmd == "transcribe":
        from .pipeline import transcribe_command
        return transcribe_command(args, layout, _settings_from(args))
    if cmd == "progress":
        from .jobs import progress
        print(json.dumps(progress(layout, args.run), indent=2, ensure_ascii=False))
        return EXIT_OK
    if cmd == "validate":
        from .schema import validate_run
        errors = validate_run(layout, args.run)
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2, ensure_ascii=False))
        return EXIT_OK if not errors else EXIT_ERROR
    if cmd == "migrate":
        from .schema import migrate_run
        print(json.dumps(migrate_run(layout, args.run), indent=2, ensure_ascii=False))
        return EXIT_OK
    if cmd == "export":
        from .export import export_command
        return export_command(args, layout)
    if cmd == "speakers":
        from .speakers import speakers_command
        return speakers_command(args, layout)
    if cmd == "annotate":
        from .annotate.claude import annotate_command
        return annotate_command(args, layout)
    if cmd == "calibrate":
        from .calibrate import calibrate_command
        return calibrate_command(args, layout, _settings_from(args))
    if cmd == "eval":
        from .evalcli import eval_command
        return eval_command(args, layout)
    raise UserError(f"unknown command {cmd}")


def _models(args: argparse.Namespace, layout: DataLayout) -> int:
    from . import registry
    op = args.op
    if op == "list":
        print(json.dumps(registry.status(layout), indent=2))
        return EXIT_OK
    if args.ids:
        ids = args.ids
    elif args.all:
        ids = [i for i, a in registry.artefacts().items() if not a.optional]
    else:
        from .config import resolve
        s = resolve(_settings_from(args), layout)
        ids = registry.ids_for(s.profile if s.profile != "auto" else "meeting",
                               quality_max=(s.quality == "max"),
                               diarization=None if s.diarization == "auto" else (s.diarization == "on"))
    if op == "plan":
        print(json.dumps(registry.plan(ids, layout), indent=2))
        return EXIT_OK
    if op == "install":
        rc = registry.install(ids, layout, yes=args.yes, dry_run=args.dry_run, cb=_hash_progress)
        if rc == 0 and not args.dry_run and not args.no_calibrate and any(i.startswith("asr/") for i in ids):
            # design §5.1: the quick calibration runs automatically after the models exist
            from .calibrate import calibrate
            from .config import resolve
            print("running calibrate --quick (about 3-4 minutes) ...", flush=True)
            try:
                calibrate(layout, full=False, settings=resolve(_settings_from(args), layout), log=lambda m: print(m, flush=True))
            except RuntimeError as exc:
                print(f"calibration skipped: {exc}", file=sys.stderr)
        return rc
    if op == "verify":
        arts = registry.artefacts()
        reports = [registry.verify(arts[i], layout, _hash_progress) for i in ids]
        print(json.dumps(reports, indent=2))
        return EXIT_OK if all(r["ok"] for r in reports) else EXIT_ERROR
    if op == "remove":
        arts = registry.artefacts()
        for i in ids:
            registry.remove(arts[i], layout)
            print(f"removed {i}")
        return EXIT_OK
    raise UserError(f"models: unknown op {op}")


def _hash_progress(label: str, done: int, total: int) -> None:
    if total and (done == total or done % (256 * 1024 * 1024) < 8 * 1024 * 1024):
        sys.stderr.write(f"\r{label}: {done / total * 100:5.1f}%")
        if done == total:
            sys.stderr.write("\n")
        sys.stderr.flush()


def _config(args: argparse.Namespace, layout: DataLayout) -> int:
    from .config import KEYS, load_data_config, resolve, save_data_config
    op = args.op
    if op == "show":
        s = resolve({}, layout)
        print(json.dumps({"effective": s.as_dict(), "sources": s.sources,
                          "config_json": load_data_config(layout)}, indent=2, ensure_ascii=False))
        return EXIT_OK
    if op == "get":
        if not args.key:
            raise UserError("config get <key>")
        print(getattr(resolve({}, layout), args.key, ""))
        return EXIT_OK
    if op in ("set", "unset"):
        if not args.key or args.key not in KEYS:
            raise UserError(f"config {op} <key> where key in {', '.join(KEYS)}")
        cur = load_data_config(layout)
        if op == "set":
            if args.value is None:
                raise UserError("config set <key> <value>")
            cur[args.key] = args.value
        else:
            cur.pop(args.key, None)
        save_data_config(layout, cur)
        print(json.dumps(cur, indent=2, ensure_ascii=False))
        return EXIT_OK
    raise UserError(f"config: unknown op {op}")


if __name__ == "__main__":
    sys.exit(main())
