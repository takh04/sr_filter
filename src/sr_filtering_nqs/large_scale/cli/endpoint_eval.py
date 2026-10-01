"""CLI for checkpoint endpoint evaluation and independent-run pooling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sr_filtering_nqs.large_scale.core.endpoint_eval import (
    aggregate_endpoint_files,
    atomic_write_json,
    evaluate_checkpoint_endpoint,
)


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    actions = parser.add_subparsers(dest="action", required=True)

    run = actions.add_parser("run", help="Evaluate one checkpoint with fresh samples")
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--experiment-id", required=True)
    run.add_argument("--system", choices=("j1j2", "tfim"), required=True)
    run.add_argument("--rounds", type=int, default=512)
    run.add_argument("--burn-in", type=int, default=5)
    run.add_argument("--seed", type=int, default=12345)
    run.add_argument("--samples-per-round", type=int, default=8192)
    run.add_argument("--method", choices=("sr", "spring", "ms_sr", "bagged_sr"), required=True)
    run.add_argument("--lambda-policy", required=True)
    run.add_argument("--k", type=int, required=True)
    run.add_argument("--branch-seed", type=int, required=True)
    run.add_argument("--replica", type=int, required=True)
    run.add_argument("--repeat", type=int, required=True)
    run.add_argument("--output", type=Path, default=None)

    aggregate = actions.add_parser("aggregate", help="Pool independent endpoint JSON files")
    aggregate.add_argument("inputs", nargs="+", type=Path)
    aggregate.add_argument("--arm", default=None)
    aggregate.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/endpoint-eval/aggregate.json"),
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    return configure_parser(argparse.ArgumentParser(description=__doc__))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
    print(json.dumps({"output": str(output), "energy_per_site": payload["energy_per_site"], "sigma_per_site": payload["sigma_per_site"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
