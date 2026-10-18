"""Kairos command line.

  python -m kairos serve               start API + UI on http://localhost:8000
  python -m kairos replay pune-demo    run a demo script and print the event trace
  python -m kairos bench               run the dev benchmark (writes runs/bench/benchmark/latest.json)
  python -m kairos ablate              run ablation experiments (writes runs/bench/experiments/latest.json)
  python -m kairos train-stability     fit the stability model on training streams (self-supervised)
  python -m kairos tune-controller     tune controller cost weights on training streams
  python -m kairos record-demo         record demo traces as frontend fixtures
  python -m kairos reproduce           train → bench (2 repeats) → ablate, in one command
"""

from __future__ import annotations

import argparse
import json
import sys

from kairos.config import load_config, resolve_path


def _cfg(args):
    return load_config(args.config)


def cmd_serve(args) -> None:
    from kairos.api.server import dist_exists, run

    if not dist_exists():
        print("note: web/dist not found — API only. Build the UI with: npm --prefix web ci && npm --prefix web run build",
              file=sys.stderr)
    run(_cfg(args), args.host, args.port)


def cmd_replay(args) -> None:
    from kairos.engine import EngineContext
    from kairos.replay import run_script_sync
    from kairos.telemetry.validator import validate

    events = run_script_sync(EngineContext(_cfg(args)), args.script)
    for e in events:
        p = e.payload
        extra = ""
        if e.event_type == "TranscriptChunkReceived":
            extra = repr(p["text"])
        elif e.event_type == "RetrievalDecision":
            extra = f"{p['decision']} {p['reasons']}"
        elif e.event_type == "AnswerVersionCreated":
            extra = f"V{p['version']} {p['kind']}\n    " + p["rendering"].replace("\n", "\n    ")
        print(f"{e.timestamp:8.3f}  {e.event_type:24s} {e.sub_query_id or '':4s} {extra}")
    v = validate(events)
    print(f"\ntelemetry coverage: {v['coverage']:.3f} ({v['passed']}/{v['instances']} invariant instances)")


def cmd_bench(args) -> None:
    from benchmark.runner import run_benchmark

    report = run_benchmark(_cfg(args), split=args.split, repeats=args.repeats)
    print(json.dumps({"run_id": report["run_id"], "gates": report["gates"]}, indent=1))
    print(f"written: {resolve_path('runs/bench/benchmark/latest.json')}")


def cmd_ablate(args) -> None:
    from benchmark.ablations import run_ablations

    out = run_ablations(_cfg(args), only=args.only)
    for key, exp in out["experiments"].items():
        print(f"\n[{key}] {exp['title']} — {exp['status']}")
        for arm, data in exp.get("arms", {}).items():
            print(f"   {arm:18s} {data['metrics']}")


def cmd_train(args) -> None:
    from kairos.controller.train import train

    print(json.dumps(train(_cfg(args)), indent=1))


def cmd_tune(args) -> None:
    from kairos.controller.tune import tune

    r = tune(_cfg(args), max_premature=args.max_premature)
    print(json.dumps(r["best"], indent=1))
    print("apply these values in configs/default.yaml (controller section) to adopt them.")


def cmd_record(args) -> None:
    from kairos.engine import EngineContext
    from kairos.replay import dump_events, list_scripts, run_script_sync

    ctx = EngineContext(_cfg(args))
    out_dir = resolve_path(args.out)
    for s in list_scripts():
        events = run_script_sync(ctx, s["id"], session_id=f"rec-{s['id']}")
        dump_events(events, out_dir / f"{s['id']}.events.json")
        print(f"recorded {s['id']}: {len(events)} events")
    (out_dir / "index.json").write_text(json.dumps(list_scripts(), indent=1), encoding="utf-8")


def cmd_reproduce(args) -> None:
    cmd_train(args)
    args.split, args.repeats = "dev", 2
    cmd_bench(args)
    args.only = None
    cmd_ablate(args)


def cmd_import_corpus(args) -> None:
    from kairos.corpus.adapters import get_adapter

    adapter = get_adapter(args.dataset)
    if args.subset:
        adapter.spec["config"] = args.subset
    if args.split:
        adapter.spec["split"] = args.split
    result = adapter.run(
        args.source,
        resolve_path(args.destination or f"corpus/imported/{args.dataset}"),
        limit=args.limit,
        seed=args.seed,
        revision=args.revision,
        dry_run=args.dry_run,
        validate_only=args.validate_only,
        expected_checksum=args.expected_checksum,
    )
    print(json.dumps(result.manifest, indent=2, sort_keys=True))


def cmd_semantic_bench(args) -> None:
    from benchmark.semantic_runner import run_semantic_benchmark

    report = run_semantic_benchmark(_cfg(args))
    print(json.dumps({"run_id": report["run_id"], "metrics": report["metrics"],
                      "ablations": report["ablations"]}, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kairos", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML merged over configs/default.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("replay")
    p.add_argument("script")
    p.set_defaults(fn=cmd_replay)
    p = sub.add_parser("bench")
    p.add_argument("--split", default="dev")
    p.add_argument("--repeats", type=int, default=2)
    p.set_defaults(fn=cmd_bench)
    p = sub.add_parser("ablate")
    p.add_argument("--only", nargs="*")
    p.set_defaults(fn=cmd_ablate)
    sub.add_parser("train-stability").set_defaults(fn=cmd_train)
    p = sub.add_parser("tune-controller")
    p.add_argument("--max-premature", type=float, default=0.2)
    p.set_defaults(fn=cmd_tune)
    p = sub.add_parser("record-demo")
    p.add_argument("--out", default="web/src/mocks/fixtures")
    p.set_defaults(fn=cmd_record)
    sub.add_parser("reproduce").set_defaults(fn=cmd_reproduce)
    p = sub.add_parser("import-corpus", help="prepare a deterministic external Markdown corpus")
    p.add_argument("--dataset", required=True, choices=["scifact", "squad"])
    p.add_argument("--source", default=None, help="offline Dataset Viewer JSON; omit to fetch the public API")
    p.add_argument("--destination", default=None)
    p.add_argument("--revision", default="main", help="immutable source revision when known")
    p.add_argument("--subset", default=None, help="dataset configuration/subset")
    p.add_argument("--split", default=None)
    p.add_argument("--expected-checksum", default=None, help="SHA-256 of the source response/file")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.set_defaults(fn=cmd_import_corpus)
    sub.add_parser("semantic-bench", help="run V2 semantic/relevance/grounding/scale evaluations").set_defaults(
        fn=cmd_semantic_bench
    )
    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
