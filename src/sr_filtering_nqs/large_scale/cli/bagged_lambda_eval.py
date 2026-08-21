from __future__ import annotations

import argparse

from sr_filtering_nqs.large_scale.core.bagged_lambda_protocol import (  # noqa: E402
    DEFAULT_BAGGED_LAMBDAS,
    run_bagged_fixed_lambda,
)
from sr_filtering_nqs.large_scale.core.fixed_data_protocol import (  # noqa: E402
    FIXED_DATA_REPEAT_COUNT,
)

SYSTEM_CHOICES = ("j1j2", "tfim", "ssm", "shastry")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bagged SR at fixed shifts on the fixed-data Figure 3 protocol"
    )
    parser.add_argument("--input-dir", default="outputs")
    parser.add_argument(
        "--diagnostic-dir",
        default=None,
        help="Figure 2 output/cache root; defaults to --output-dir.",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--system", choices=SYSTEM_CHOICES, required=True)
    parser.add_argument(
        "--bagged-lambdas",
        type=str,
        default=",".join(f"{lam:g}" for lam in DEFAULT_BAGGED_LAMBDAS),
        help="Comma-separated fixed shifts for the bagged solves",
    )
    parser.add_argument("--source-train-lambda", type=float, default=1e-4)
    parser.add_argument("--source-steps", type=str, default="all")
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--repeat-count", type=int, default=FIXED_DATA_REPEAT_COUNT)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--lambda-quantiles",
        default=None,
        help="Optional comma-separated, strictly decreasing NTK quantiles (K entries).",
    )
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--validation-round-samples", type=int, default=None)
    parser.add_argument("--validation-sample-chunk", type=int, default=None)
    parser.add_argument("--jvp-chunk", type=int, default=None)
    parser.add_argument("--delta-batch-size", type=int, default=10)
    parser.add_argument("--train-seed-base", type=int, default=None)
    parser.add_argument("--validation-seed-base", type=int, default=None)
    parser.add_argument("--spectrum-seed-base", type=int, default=None)
    parser.add_argument("--model-chunk-size-bwd", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bagged_lambdas = tuple(
        float(item) for item in args.bagged_lambdas.split(",") if item.strip()
    )
    lambda_quantiles = (
        None
        if args.lambda_quantiles is None
        else tuple(
            float(item)
            for item in args.lambda_quantiles.split(",")
            if item.strip()
        )
    )
    print(
        f"[bagged-lambda] system={args.system} lambdas={bagged_lambdas} "
        f"repeat_count={args.repeat_count} k={args.k}",
        flush=True,
    )
    run_bagged_fixed_lambda(
        args.input_dir,
        diagnostic_root=args.diagnostic_dir,
        output_root=args.output_dir,
        system_name=args.system,
        bagged_lambdas=bagged_lambdas,
        source_training_lambda=float(args.source_train_lambda),
        source_steps=args.source_steps,
        latest_only=args.latest_only,
        repeat_count=int(args.repeat_count),
        k=int(args.k),
        n_val=args.n_val,
        validation_round_samples=args.validation_round_samples,
        validation_sample_chunk=args.validation_sample_chunk,
        jvp_chunk=args.jvp_chunk,
        delta_batch_size=int(args.delta_batch_size),
        train_seed_base=args.train_seed_base,
        validation_seed_base=args.validation_seed_base,
        spectrum_seed_base=args.spectrum_seed_base,
        lambda_quantiles=lambda_quantiles,
        model_chunk_size_bwd=args.model_chunk_size_bwd,
    )


if __name__ == "__main__":
    main()
