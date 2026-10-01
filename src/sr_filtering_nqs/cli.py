"""Unified command line interface for SR-filtering NQS experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("sr-filtering-nqs")
    except Exception:  # pragma: no cover - editable source without metadata
        return "0.1.0"


def _float_tuple(raw: str | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    values = tuple(float(item) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("Expected at least one comma-separated number")
    return values


def _artifact_fetch(args: argparse.Namespace) -> int:
    from sr_filtering_nqs.artifact_store import fetch_artifacts

    report = fetch_artifacts(
        args.manifest,
        output_root=args.output_root,
        experiment_ids=args.experiment or None,
        source_root=args.source_root,
        force=args.force,
        max_file_bytes=args.max_file_bytes,
        max_total_bytes=args.max_total_bytes,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _fixed_data(args: argparse.Namespace) -> int:
    systems = ("j1j2", "tfim") if args.system == "all" else (args.system,)
    quantiles = _float_tuple(args.lambda_quantiles)
    common = {
        "source_training_lambda": args.source_train_lambda,
        "source_steps": args.source_steps,
        "latest_only": args.latest_only,
        "repeat_count": args.repeat_count,
        "n_val": args.n_val,
        "validation_round_samples": args.validation_round_samples,
        "validation_sample_chunk": args.validation_sample_chunk,
        "jvp_chunk": args.jvp_chunk,
        "delta_batch_size": args.delta_batch_size,
        "train_seed_base": args.train_seed_base,
        "validation_seed_base": args.validation_seed_base,
        "spectrum_seed_base": args.spectrum_seed_base,
        "model_chunk_size_bwd": args.model_chunk_size_bwd,
    }
    if args.protocol == "figure2":
        if args.diagnostic_dir is not None:
            raise ValueError(
                "--diagnostic-dir is an input for figure3/bagged; "
                "figure2 writes diagnostics to --output-dir"
            )
        from sr_filtering_nqs.large_scale.core.fixed_data_protocol import (
            default_diagnostic_lambdas,
            run_fixed_data_fig2,
        )

        diagnostic_lambdas = _float_tuple(args.diag_lambdas) or default_diagnostic_lambdas()
        for system in systems:
            run_fixed_data_fig2(
                args.input_dir,
                output_root=args.output_dir,
                system_name=system,
                batch_slots=args.batch_slots,
                diagnostic_lambdas=diagnostic_lambdas,
                conn_chunk=args.conn_chunk,
                eval_chunk=args.eval_chunk,
                **common,
            )
    elif args.protocol == "figure3":
        from sr_filtering_nqs.large_scale.core.fixed_data_protocol import run_fixed_data_fig3

        for system in systems:
            run_fixed_data_fig3(
                args.input_dir,
                diagnostic_root=args.diagnostic_dir,
                output_root=args.output_dir,
                system_name=system,
                k=args.k,
                lambda_quantiles=quantiles,
                **common,
            )
    else:
        from sr_filtering_nqs.large_scale.core.bagged_lambda_protocol import (
            DEFAULT_BAGGED_LAMBDAS,
            run_bagged_fixed_lambda,
        )

        bagged_lambdas = _float_tuple(args.bagged_lambdas) or DEFAULT_BAGGED_LAMBDAS
        for system in systems:
            run_bagged_fixed_lambda(
                args.input_dir,
                diagnostic_root=args.diagnostic_dir,
                output_root=args.output_dir,
                system_name=system,
                bagged_lambdas=bagged_lambdas,
                k=args.k,
                lambda_quantiles=quantiles,
                **common,
            )
    return 0


def _online_train(args: argparse.Namespace) -> int:
    forwarded = list(args.trainer_args)
    if forwarded[:1] == ["--"]:
        forwarded.pop(0)
    if "--output-root" not in forwarded:
        forwarded = ["--output-root", str(args.output_root), *forwarded]
    if args.system == "j1j2":
        from sr_filtering_nqs.large_scale.cli import train_sweep as module

        forwarded = ["--system", "j1j2", *forwarded]
    else:
        from sr_filtering_nqs.large_scale.cli import train_tfim_fnqs as module

        forwarded = ["--system", "tfim", *forwarded]

    previous_argv = sys.argv
    try:
        sys.argv = [f"sr-filter online-train --system {args.system}", *forwarded]
        result = module.main()
    finally:
        sys.argv = previous_argv
    return int(result or 0)


def _benchmark(args: argparse.Namespace) -> int:
    from sr_filtering_nqs.large_scale.cli.benchmark_optimized_sr_cost import main as benchmark_main

    output = args.output or Path("outputs/benchmarks") / f"{args.method}_seed{args.seed}.json"
    argv = [
        "--checkpoint",
        str(args.checkpoint),
        "--method",
        args.method,
        "--output",
        str(output),
        "--seed",
        str(args.seed),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--k",
        str(args.k),
        "--diag-shift",
        str(args.diag_shift),
        "--quantiles",
        args.quantiles,
    ]
    if args.prediction_chunk is not None:
        argv.extend(("--prediction-chunk", str(args.prediction_chunk)))
    argv.append("--apply-update" if args.apply_update else "--no-apply-update")
    return int(benchmark_main(argv) or 0)


def _endpoint_eval(args: argparse.Namespace) -> int:
    from sr_filtering_nqs.large_scale.core.endpoint_eval import (
        aggregate_endpoint_files,
        atomic_write_json,
        evaluate_checkpoint_endpoint,
    )

    if args.action == "run":
        payload = evaluate_checkpoint_endpoint(
            args.checkpoint,
            experiment_id=args.experiment_id,
            system_name=args.system,
            rounds=args.rounds,
            burn_in=args.burn_in,
            seed=args.seed,
            samples_per_round=args.samples_per_round,
            method=args.method,
            lambda_policy=args.lambda_policy,
            k=args.k,
            branch_seed=args.branch_seed,
            replica=args.replica,
            repeat=args.repeat,
        )
        output = args.output or Path("outputs/endpoint-eval") / (
            f"{args.checkpoint.stem}_{args.method}_branch{args.branch_seed}_"
            f"replica{args.replica}_repeat{args.repeat}.json"
        )
    else:
        payload = aggregate_endpoint_files(args.inputs, arm=args.arm)
        output = args.output
    atomic_write_json(payload, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "energy_per_site": payload["energy_per_site"],
                "sigma_per_site": payload["sigma_per_site"],
            }
        )
    )
    return 0


def _reproduce(args: argparse.Namespace) -> int:
    from sr_filtering_nqs.reproduce import reproduce

    outputs = reproduce(
        args.recipe,
        input_root=args.input_root,
        output_dir=args.output_dir,
    )
    print(json.dumps({"recipe": args.recipe, "outputs": [str(path) for path in outputs]}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sr-filter", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    commands = parser.add_subparsers(dest="command", required=True)

    artifact = commands.add_parser("artifact", help="Manage checksum-verified experiment artifacts")
    artifact_actions = artifact.add_subparsers(dest="artifact_action", required=True)
    fetch = artifact_actions.add_parser("fetch", help="Fetch and verify files from a public manifest")
    fetch.add_argument("--manifest", default="artifacts/paper/manifest.json")
    fetch.add_argument("--output-root", type=Path, default=Path("outputs"))
    fetch.add_argument("--source-root", type=Path, default=None)
    fetch.add_argument("--experiment", action="append", default=[])
    fetch.add_argument("--force", action="store_true")
    fetch.add_argument("--max-file-bytes", type=int, default=512 * 1024 * 1024)
    fetch.add_argument("--max-total-bytes", type=int, default=2 * 1024 * 1024 * 1024)
    fetch.set_defaults(handler=_artifact_fetch)

    fixed = commands.add_parser("fixed-data", help="Run paired checkpoint-local SR protocols")
    fixed.add_argument("--protocol", choices=("figure2", "figure3", "bagged"), default="figure3")
    fixed.add_argument("--input-dir", type=Path, default=Path("outputs"))
    fixed.add_argument(
        "--diagnostic-dir",
        type=Path,
        default=None,
        help=(
            "Figure 2 output/cache root read by figure3 and bagged; defaults "
            "to --output-dir for a single-workspace run"
        ),
    )
    fixed.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Explicit destination for generated caches/results (default: ./outputs)",
    )
    fixed.add_argument("--system", choices=("all", "j1j2", "tfim"), default="all")
    fixed.add_argument("--source-train-lambda", type=float, default=1e-4)
    fixed.add_argument("--source-steps", default="all")
    fixed.add_argument("--latest-only", action="store_true")
    fixed.add_argument("--repeat-count", type=int, default=20)
    fixed.add_argument("--k", type=int, default=4)
    fixed.add_argument("--lambda-quantiles", default=None)
    fixed.add_argument("--diag-lambdas", default=None)
    fixed.add_argument("--bagged-lambdas", default=None)
    fixed.add_argument("--batch-slots", type=int, default=4)
    fixed.add_argument("--n-val", type=int, default=None)
    fixed.add_argument("--validation-round-samples", type=int, default=None)
    fixed.add_argument("--validation-sample-chunk", type=int, default=None)
    fixed.add_argument("--jvp-chunk", type=int, default=None)
    fixed.add_argument("--conn-chunk", type=int, default=4096)
    fixed.add_argument("--eval-chunk", type=int, default=4096)
    fixed.add_argument("--delta-batch-size", type=int, default=10)
    fixed.add_argument("--train-seed-base", type=int, default=None)
    fixed.add_argument("--validation-seed-base", type=int, default=None)
    fixed.add_argument("--spectrum-seed-base", type=int, default=None)
    fixed.add_argument("--model-chunk-size-bwd", type=int, default=None)
    fixed.set_defaults(handler=_fixed_data)

    online = commands.add_parser(
        "online-train",
        help="Run an online SR, MS-SR, or SPRING training arm",
        description="Pass native trainer options after --, for example: sr-filter online-train --system j1j2 -- --diag-shift 1e-4",
    )
    online.add_argument("--system", choices=("j1j2", "tfim"), required=True)
    online.add_argument("--output-root", type=Path, default=Path("outputs"))
    online.add_argument("trainer_args", nargs=argparse.REMAINDER)
    online.set_defaults(handler=_online_train)

    endpoint = commands.add_parser("endpoint-eval", help="Evaluate or aggregate final checkpoints")
    from sr_filtering_nqs.large_scale.cli.endpoint_eval import configure_parser

    configure_parser(endpoint)
    endpoint.set_defaults(handler=_endpoint_eval)

    benchmark = commands.add_parser("benchmark", help="Benchmark fair optimized SR/bagged/MS-SR steps")
    benchmark.add_argument("--checkpoint", type=Path, required=True)
    benchmark.add_argument("--method", choices=("sr", "bagged", "mssr"), required=True)
    benchmark.add_argument("--output", type=Path, default=None)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--warmup", type=int, default=2)
    benchmark.add_argument("--repeats", type=int, default=10)
    benchmark.add_argument("--k", type=int, default=4)
    benchmark.add_argument("--diag-shift", type=float, default=1e-4)
    benchmark.add_argument("--quantiles", default="0.9,0.7,0.4,0.1")
    benchmark.add_argument("--prediction-chunk", type=int, default=None)
    benchmark.add_argument("--apply-update", action="store_true")
    benchmark.set_defaults(handler=_benchmark)

    reproduce_parser = commands.add_parser("reproduce", help="Regenerate figures from compact artifacts")
    reproduce_parser.add_argument("--recipe", choices=("paper", "all", "curated", "online"), default="paper")
    reproduce_parser.add_argument("--input-root", type=Path, default=Path("."))
    reproduce_parser.add_argument("--output-dir", type=Path, default=Path("outputs/reproduced"))
    reproduce_parser.set_defaults(handler=_reproduce)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args) or 0)
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(f"sr-filter: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
