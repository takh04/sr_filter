from __future__ import annotations

from pathlib import Path

from sr_filtering_nqs.cli import build_parser


def test_endpoint_cli_contract_and_defaults():
    args = build_parser().parse_args(
        [
            "endpoint-eval",
            "run",
            "--checkpoint",
            "state.mpack",
            "--experiment-id",
            "online.branch.seed0.v1",
            "--system",
            "j1j2",
            "--method",
            "ms_sr",
            "--lambda-policy",
            "quantile_stacking",
            "--k",
            "4",
            "--branch-seed",
            "0",
            "--replica",
            "2",
            "--repeat",
            "1",
        ]
    )

    assert args.samples_per_round == 8192
    assert args.rounds == 512
    assert args.burn_in == 5
    assert args.experiment_id == "online.branch.seed0.v1"
    assert args.method == "ms_sr"
    assert args.replica == 2


def test_public_cli_output_defaults_are_working_directory_relative():
    parser = build_parser()
    fetch = parser.parse_args(["artifact", "fetch"])
    fixed_data = parser.parse_args(["fixed-data"])
    reproduce = parser.parse_args(["reproduce", "--recipe", "curated"])

    assert fetch.output_root == Path("outputs")
    assert fixed_data.diagnostic_dir is None
    assert fixed_data.output_dir == Path("outputs")
    assert reproduce.output_dir == Path("outputs/reproduced")
