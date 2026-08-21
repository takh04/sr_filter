from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from sr_filtering_nqs.large_scale.cli.plot_online_comparison import (
    BRANCH_ARMS,
    GateError,
    budget_rolling_xy,
    load_online_artifacts,
    plot_online_comparison,
)

CURVE_FIELDS = (
    "experiment_id",
    "run_status",
    "system",
    "source_checkpoint_step",
    "branch_rng_seed",
    "arm",
    "optimizer",
    "curve_segment",
    "legacy_iteration",
    "optimizer_step_since_branch",
    "batches_per_update",
    "samples_per_batch",
    "sampling_budget_units_since_branch",
    "budget_equivalent_global_step",
    "energy_per_site",
    "energy_sigma_total",
    "energy_variance_total",
    "acceptance",
    "step_wall_time_s",
)
ENDPOINT_FIELDS = (
    "experiment_id",
    "run_status",
    "system",
    "branch_rng_seed",
    "arm",
    "checkpoint_step",
    "eval_run",
    "eval_seed",
    "rounds",
    "burn_in_rounds",
    "samples_per_round",
    "total_samples",
    "energy_per_site",
    "sigma_per_site",
    "sigma_round_level_per_site",
    "sigma_block8_level_per_site",
    "audit_status",
)


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _exact_audit():
    return {
        "status": "audited_complete",
        "update_protocol": "fixed_grid_exact_loo",
        "stacking_protocol": "exact_leave_one_batch_out",
        "lambda_policy": "fixed_grid",
        "lambda_grid": [1e-2, 1e-3, 1e-4, 1e-5],
        "k": 4,
        "optimizer_updates": 125,
        "samples_per_candidate_batch": 8192,
        "candidate_sample_counts_final_update": [8192, 8192, 8192, 8192],
        "candidate_batches_per_update": 4,
        "candidate_solves_per_update": 4,
        "sampling_solve_units": 500,
        "base_driver_sampling_calls": 0,
        "base_driver_solves": 0,
        "weight_fit_sampling_calls": 0,
        "weight_fit_solves": 0,
        "checkpoint_sha256": "a" * 64,
    }


def _make_online_branch(root, branch_id, *, seed, source_step, status="complete"):
    branch = root / "artifacts" / "online_training" / branch_id
    branch.mkdir(parents=True)
    experiment_id = f"online.j1j2.no_symmetry.branch{source_step}.seed{seed}.v1"
    common = {
        "experiment_id": experiment_id,
        "run_status": "complete",
        "system": "j1j2",
        "source_checkpoint_step": source_step,
        "branch_rng_seed": seed,
        "optimizer": "sr",
        "curve_segment": "branch",
        "legacy_iteration": 0,
        "samples_per_batch": 8192,
        "energy_sigma_total": 0.01,
        "energy_variance_total": 1.0,
        "acceptance": 0.5,
        "step_wall_time_s": 1.0,
    }
    shared_count = 550 if seed == 0 else 1
    curve_rows = []
    for index in range(shared_count):
        curve_rows.append(
            {
                **common,
                "arm": "shared_sr",
                "optimizer_step_since_branch": 0,
                "batches_per_update": 1,
                "sampling_budget_units_since_branch": 0,
                "budget_equivalent_global_step": source_step - shared_count + index + 1,
                "energy_per_site": -1.89 - 0.01 * seed - 1e-5 * index,
            }
        )
    for arm, steps, batches, offset in (
        ("sr", 500, 1, 0.01),
        ("spring", 500, 1, 0.02),
        ("ms_sr", 125, 4, 0.03),
    ):
        for local_step in range(1, steps + 1):
            budget = batches * local_step
            curve_rows.append(
                {
                    **common,
                    "arm": arm,
                    "optimizer": ("ms_sr_fixed_grid_exact_loo" if arm == "ms_sr" else arm),
                    "optimizer_step_since_branch": local_step,
                    "batches_per_update": batches,
                    "sampling_budget_units_since_branch": budget,
                    "budget_equivalent_global_step": source_step + budget,
                    "energy_per_site": (-1.90 - offset - 0.01 * seed - 0.05 * budget / 500.0),
                },
            )
    _write_csv(branch / "training_curve.csv", CURVE_FIELDS, curve_rows)

    endpoint_rows = []
    for arm_index, arm in enumerate(BRANCH_ARMS):
        for replica, eval_seed in ((1, 12345), (2, 98765)):
            endpoint_rows.append(
                {
                    "experiment_id": experiment_id,
                    "run_status": "complete",
                    "system": "j1j2",
                    "branch_rng_seed": seed,
                    "arm": arm,
                    "checkpoint_step": (
                        550 if seed == 0 and arm == "sr" else 125 if arm == "ms_sr" else 500
                    ),
                    "eval_run": replica,
                    "eval_seed": eval_seed,
                    "rounds": 512,
                    "burn_in_rounds": 5,
                    "samples_per_round": 8192,
                    "total_samples": 4_194_304,
                    "energy_per_site": -1.97 - 0.01 * arm_index - 1e-5 * replica,
                    "sigma_per_site": 2e-5 * replica,
                    "sigma_round_level_per_site": 1e-5,
                    "sigma_block8_level_per_site": 2e-5 * replica,
                    "audit_status": "pass",
                }
            )
    _write_csv(branch / "endpoint_evaluations.csv", ENDPOINT_FIELDS, endpoint_rows)
    (branch / "status.json").write_text(
        json.dumps(
            {
                "status": status,
                "camera_ready_gate": status == "complete",
                "experiment_id": experiment_id,
                "ms_sr_protocol_audit": _exact_audit(),
            }
        )
    )
    (branch / "provenance.json").write_text(
        json.dumps(
            {
                "branch_rng_seed": seed,
                "experiment_id": experiment_id,
                "ms_sr_protocol_audit": _exact_audit(),
            }
        )
    )


def _make_complete_online_tree(root):
    _make_online_branch(root, "branch_seed0", seed=0, source_step=2500)
    _make_online_branch(root, "branch_seed1", seed=1, source_step=1800)


def test_online_gate_requires_both_complete_branches(tmp_path):
    _make_online_branch(tmp_path, "branch_seed0", seed=0, source_step=2500)

    with pytest.raises(GateError, match="both online branches"):
        load_online_artifacts(tmp_path)


def test_online_gate_pools_two_replicas_and_uses_budget_grid(tmp_path):
    _make_complete_online_tree(tmp_path)

    first, second = load_online_artifacts(tmp_path)
    endpoint = first.endpoints["sr"]
    assert (first.seed, first.source_step) == (0, 2500)
    assert (second.seed, second.source_step) == (1, 1800)
    np.testing.assert_allclose(endpoint.energy_per_site, (-1.97001 - 1.97002) / 2)
    np.testing.assert_allclose(endpoint.sigma_per_site, np.sqrt((2e-5 / 2) ** 2 + (4e-5 / 2) ** 2))
    assert endpoint.replica_difference_z < 1.0
    x, y = budget_rolling_xy(first, "ms_sr", window_budget=25)
    assert x[0] == 24
    assert x[-1] == 500
    assert len(x) == len(y) == 477


def test_online_plot_writes_outputs_only_after_validation(tmp_path):
    _make_complete_online_tree(tmp_path)
    output_dir = tmp_path / "figures"
    repeat_dir = tmp_path / "figures_repeat"

    outputs = plot_online_comparison(tmp_path, output_dir)
    repeat_outputs = plot_online_comparison(tmp_path, repeat_dir)

    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    assert [path.read_bytes() for path in outputs] == [path.read_bytes() for path in repeat_outputs]
    payload = json.loads(outputs[2].read_text())
    assert payload["ready"] is True
    assert len(payload["branches"]) == 2
    assert payload["protocol"]["endpoint_burn_in_rounds"] == 5


def test_online_gate_rejects_statistically_inconsistent_replicas(tmp_path):
    _make_complete_online_tree(tmp_path)
    endpoint_path = (
        tmp_path / "artifacts" / "online_training" / "branch_seed0" / "endpoint_evaluations.csv"
    )
    with endpoint_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    sr_rows = [row for row in rows if row["arm"] == "sr"]
    sr_rows[1]["energy_per_site"] = str(float(sr_rows[0]["energy_per_site"]) + 1e-3)
    _write_csv(endpoint_path, ENDPOINT_FIELDS, rows)

    with pytest.raises(GateError, match="consistency gate"):
        load_online_artifacts(tmp_path)


def test_online_gate_rejects_weakened_exact_protocol_audit(tmp_path):
    _make_complete_online_tree(tmp_path)
    status_path = tmp_path / "artifacts" / "online_training" / "branch_seed0" / "status.json"
    payload = json.loads(status_path.read_text())
    payload["ms_sr_protocol_audit"]["weight_fit_sampling_calls"] = 1
    status_path.write_text(json.dumps(payload))

    with pytest.raises(GateError, match="weight_fit_sampling_calls"):
        load_online_artifacts(tmp_path)


def test_online_gate_rejects_truncated_curve_with_valid_final_row(tmp_path):
    _make_complete_online_tree(tmp_path)
    curve_path = tmp_path / "artifacts" / "online_training" / "branch_seed0" / "training_curve.csv"
    with curve_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in rows
        if not (row["arm"] == "ms_sr" and int(row["optimizer_step_since_branch"]) == 62)
    ]
    _write_csv(curve_path, CURVE_FIELDS, rows)

    with pytest.raises(GateError, match=r"ms_sr has 124 rows, expected 125"):
        load_online_artifacts(tmp_path)
