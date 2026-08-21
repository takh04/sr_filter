from __future__ import annotations

import pytest

from sr_filtering_nqs.large_scale.core.shared_checkpoint_protocol import (
    discover_source_checkpoints,
    shared_fig2_delta_bank_path,
    shared_fig2_metric_path,
    shared_fig3_bank_path,
    shared_fig3_eval_path,
)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint")


def test_discover_source_checkpoints_respects_explicit_steps(tmp_path):
    ckpt_dir = tmp_path / "large_scale" / "j1j2_results" / "1e-04" / "results" / "checkpoints"
    _touch(ckpt_dir / "checkpoint_step000900.pkl")
    _touch(ckpt_dir / "checkpoint_step001600.pkl")

    rows = discover_source_checkpoints(
        tmp_path / "large_scale",
        "j1j2",
        source_training_lambda=1e-4,
        source_steps="900,1600",
    )

    assert [step for step, _ in rows] == [900, 1600]


def test_discover_source_checkpoints_raises_for_missing_explicit_step(tmp_path):
    ckpt_dir = tmp_path / "large_scale" / "tfim_results" / "1e-04" / "results" / "checkpoints"
    _touch(ckpt_dir / "checkpoint_step001100.pkl")

    with pytest.raises(FileNotFoundError, match="Missing requested source checkpoints"):
        discover_source_checkpoints(
            tmp_path / "large_scale",
            "tfim",
            source_training_lambda=1e-4,
            source_steps="1100,1200",
        )


def test_shared_protocol_paths_encode_protocol_source_and_repeat(tmp_path):
    ckpt_path = (
        tmp_path
        / "large_scale"
        / "j1j2_results"
        / "1e-04"
        / "results"
        / "checkpoints"
        / "checkpoint_step000900.pkl"
    )
    _touch(ckpt_path)

    fig2_bank = shared_fig2_delta_bank_path(
        ckpt_path,
        protocol_id="shared_fig2_single",
        source_training_lambda=1e-4,
        source_step=900,
        diagnostic_lambda=1e-5,
        repeat_count=20,
    )
    fig2_rval = shared_fig2_metric_path(
        ckpt_path,
        protocol_id="shared_fig2_single",
        source_training_lambda=1e-4,
        source_step=900,
        diagnostic_lambda=1e-5,
        repeat_count=20,
        metric="rval",
    )
    fig3_bank = shared_fig3_bank_path(
        ckpt_path,
        protocol_id="shared_fig3_k4",
        source_training_lambda=1e-4,
        source_step=900,
        lambda_grid=(1e-2, 1e-3, 1e-4, 1e-5),
        repeat_count=20,
        oracle_lambda=1e-4,
    )
    fig3_eval = shared_fig3_eval_path(
        ckpt_path,
        protocol_id="shared_fig3_k4",
        source_training_lambda=1e-4,
        source_step=900,
        lambda_grid=(1e-2, 1e-3, 1e-4, 1e-5),
        repeat_count=20,
        oracle_lambda=1e-4,
    )

    assert fig2_bank.parent.name == "step000900"
    assert fig2_bank.parent.parent.name == "src_1e-04"
    assert fig2_bank.parent.parent.parent.name == "shared_fig2_single"
    assert fig2_bank.name == "delta_bank__diag_1e-05__m020.pkl"
    assert fig2_rval.name == "rval__diag_1e-05__m020.pkl"
    assert fig3_bank.name.startswith("mixture_bank__k4__grid_1e-02__1e-03__1e-04__1e-05")
    assert fig3_eval.name.endswith("__oracle_1e-04__m020.pkl")
