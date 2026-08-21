from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path

import pytest


GRID = [1e-2, 1e-3, 1e-4, 1e-5]


def _curation_module():
    script = Path(__file__).resolve().parents[1] / "scripts/curation/curate_release_artifacts.py"
    spec = importlib.util.spec_from_file_location("sr_filter_online_curation", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validator_module():
    script = Path(__file__).resolve().parents[1] / "scripts/curation/validate_release_artifacts.py"
    spec = importlib.util.spec_from_file_location("sr_filter_release_validator", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exact_run(root: Path, source_checkpoint: Path) -> Path:
    result = (
        root
        / "branch_seed1/mssr_exact/large_scale/j1j2_mixture_results/"
        "indep_batch__stacking__1e-02__1e-03__1e-04__1e-05/results"
    )
    checkpoint = result / "checkpoints/checkpoint_step000125.pkl"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"exact-checkpoint")
    train_log = result / "logs/train.log"
    train_log.parent.mkdir(parents=True)
    train_log.write_text("completed without dtype casts\n", encoding="utf-8")
    config = {
        "system": "j1j2",
        "seed": 1,
        "init_from": str(source_checkpoint),
        "init_from_step": 1800,
        "n_samples": 8192,
        "phase_samples": [8192],
        "phase_lengths": [125],
        "total_steps": 125,
        "schedule_total_steps": 125,
        "sr_update_protocol": "fixed_grid_exact_loo",
        "sr_variant": "indep_batch",
        "sr_weight_scheme": "stacking",
        "sr_lambda_grid": GRID,
        "sr_prediction_chunk": 2048,
        "sr_weight_fit_samples": None,
        "sr_lambda_quantiles": None,
        "sr_lambda_ema_rho": None,
        "sr_weight_ema_rho": None,
        "ngd_method": "sr",
    }
    config_path = result / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    metrics_path = result / "train_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("step", "energy_per_site"))
        writer.writeheader()
        for step in range(1, 126):
            writer.writerow({"step": step, "energy_per_site": -1.9 - step * 1e-5})
    summary = {
        "status": "completed",
        "sr_update_protocol": config["sr_update_protocol"],
        "sr_variant": config["sr_variant"],
        "sr_weight_scheme": config["sr_weight_scheme"],
        "sr_lambda_grid": GRID,
        "sr_prediction_chunk": 2048,
        "sr_lambda_quantiles": None,
        "sr_lambda_ema_rho": None,
        "sr_weight_ema_rho": None,
        "ngd_method": "sr",
        "init_from_step": 1800,
        "total_steps": 125,
        "checkpoint_steps_saved": [25, 50, 75, 100, 125],
        "latest_metrics": {"step": 125},
    }
    (result / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    metadata = {
        "method": "ms_sr",
        "variant": "indep_batch",
        "weight_scheme": "stacking",
        "update_protocol": "fixed_grid_exact_loo",
        "stacking_protocol": "exact_leave_one_batch_out",
        "lambda_source": "fixed_grid",
        "lambda_grid": GRID,
        "weights": [0.1, 0.2, 0.3, 0.4],
        "k": 4,
        "samples_per_candidate_batch": 8192,
        "candidate_sample_counts": [8192, 8192, 8192, 8192],
        "candidate_batches": 4,
        "candidate_solves": 4,
        "sampling_solve_units_this_update": 4,
        "sampling_solve_units_cumulative": 500,
        "base_driver_sampling_calls": 0,
        "base_driver_solves": 0,
        "weight_fit_sampling_calls": 0,
        "weight_fit_solves": 0,
        "training_energy_source": "pooled_candidate_batches",
        "training_energy_sample_count": 32768,
        "cross_prediction_launches": 4,
        "cross_prediction_directions": 12,
    }
    audit = {
        "schema": "sr_filter.online_optimizer_update_audit.v1",
        "status": "checkpoint_complete",
        "checkpoint_filename": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_step": 125,
        "config_filename": config_path.name,
        "config_sha256": _sha256(config_path),
        "train_metrics_filename": metrics_path.name,
        "train_metrics_sha256": _sha256(metrics_path),
        "run_contract": config,
        "optimizer_update": metadata,
    }
    checkpoint.with_suffix(".audit.json").write_text(json.dumps(audit), encoding="utf-8")
    return result


def _write_registered_online_result(
    root: Path,
    source_checkpoint: Path,
) -> tuple[Path, dict[str, object]]:
    result = root / "results"
    (result / "logs").mkdir(parents=True)
    config: dict[str, object] = {
        "system": "j1j2",
        "seed": 0,
        "ngd_method": "sr",
        "learning_rate": 0.01,
        "lr_schedule": "constant",
        "init_from": str(source_checkpoint),
        "init_from_step": 50,
        "n_samples": 8192,
        "total_steps": 500,
    }
    (result / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (result / "logs/train.log").write_text("completed\n", encoding="utf-8")
    return result, config


def test_registered_online_config_is_bound_to_settings_source_and_clean_log(tmp_path):
    module = _curation_module()
    source = tmp_path / "source.pkl"
    source.write_bytes(b"source")
    result, expected = _write_registered_online_result(tmp_path / "run", source)

    accepted = module.validate_registered_online_config(
        result,
        expected=expected,
        expected_init_checkpoint=source,
    )
    assert accepted == expected

    changed = dict(expected)
    changed["n_samples"] = 4096
    with pytest.raises(ValueError, match="registered online contract mismatch"):
        module.validate_registered_online_config(
            result,
            expected=changed,
            expected_init_checkpoint=source,
        )

    (result / "logs/train.log").write_text(
        "ComplexWarning: Casting complex values to real discards the imaginary part\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="complex-to-real solver cast"):
        module.validate_registered_online_config(
            result,
            expected=expected,
            expected_init_checkpoint=source,
        )


def test_exact_mssr_curation_requires_json_counter_audit(tmp_path):
    module = _curation_module()
    source = tmp_path / "source.pkl"
    source.write_bytes(b"source-checkpoint")
    result = _write_exact_run(tmp_path / "online", source)

    accepted = module.validated_exact_mssr_run(
        result,
        expected_seed=1,
        expected_source_native_step=1800,
        expected_source_sha256=_sha256(source),
    )

    assert accepted is not None
    rows, checkpoint, audit = accepted
    assert len(rows) == 125
    assert checkpoint.name == "checkpoint_step000125.pkl"
    assert audit["status"] == "audited_complete"
    assert audit["sampling_solve_units"] == 500

    audit_path = checkpoint.with_suffix(".audit.json")
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["optimizer_update"]["sampling_solve_units_cumulative"] = 504
    audit_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sampling_solve_units_cumulative"):
        module.validated_exact_mssr_run(
            result,
            expected_seed=1,
            expected_source_native_step=1800,
            expected_source_sha256=_sha256(source),
        )


def test_exact_mssr_curation_rejects_complex_to_real_solver_cast(tmp_path):
    module = _curation_module()
    source = tmp_path / "source.pkl"
    source.write_bytes(b"source-checkpoint")
    result = _write_exact_run(tmp_path / "online", source)
    (result / "logs/train.log").write_text(
        "FutureWarning: scatter inputs have incompatible types: complex128 to float64\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="complex-to-real solver cast"):
        module.validated_exact_mssr_run(
            result,
            expected_seed=1,
            expected_source_native_step=1800,
            expected_source_sha256=_sha256(source),
        )


def test_exact_result_resolution_never_uses_legacy_mssr_directory(tmp_path):
    module = _curation_module()
    legacy = tmp_path / "branch_seed0/mssr"
    legacy.mkdir(parents=True)

    resolved = module.exact_mssr_result_dir(tmp_path, "branch_seed0")

    assert "mssr_exact" in resolved.parts
    assert legacy not in resolved.parents


def test_branch_origin_is_extracted_from_recorded_checkpoint_history(tmp_path):
    module = _curation_module()
    checkpoint = tmp_path / "source.pkl"
    with checkpoint.open("wb") as handle:
        pickle.dump(
            {
                "step": 1800,
                "config": {"system": "j1j2", "n_samples": 8192},
                "energy_history": {
                    "iters": [1799, 1800],
                    "mean": [-127.6, -127.6474234330906],
                    "sigma": [0.02, 0.012748501494739724],
                    "variance": [2.0, 1.9696860878763605],
                },
            },
            handle,
        )

    row, provenance = module.branch_origin_row_from_checkpoint(
        checkpoint,
        experiment_id="online.j1j2.no_symmetry.branch1800.seed1.v1",
        branch_step=1800,
        branch_seed=1,
        n_sites=64,
    )

    assert row["arm"] == "shared_sr"
    assert row["curve_segment"] == "branch_origin"
    assert row["optimizer_step_since_branch"] == 0
    assert row["sampling_budget_units_since_branch"] == 0
    assert row["energy_per_site"] == pytest.approx(-127.6474234330906 / 64)
    assert provenance["source_field"] == "energy_history at iters=1800"
    assert provenance["checkpoint_sha256"] == _sha256(checkpoint)


def test_external_checkpoint_counts_follow_frozen_inventory_status(tmp_path):
    module = _validator_module()
    inventory_dir = tmp_path / "configs/experiments"
    artifact_dir = tmp_path / "artifacts"
    inventory_dir.mkdir(parents=True)
    artifact_dir.mkdir()
    ids = [f"checkpoint_{index}" for index in range(8)]
    statuses = ["complete"] * 5 + ["pending_gpu_execution"] * 3
    blocks = []
    manifest_rows = []
    for index, (identifier, status) in enumerate(zip(ids, statuses, strict=True)):
        role = "source" if index < 2 else "final_endpoint"
        lines = [
            "[[checkpoints]]",
            f'id = "{identifier}"',
            f'role = "{role}"',
            f'status = "{status}"',
        ]
        if status == "complete":
            lines.append(f'expected_original_sha256 = "{"a" * 64}"')
            manifest_rows.append(
                {
                    "id": identifier,
                    "role": role,
                    "status": "sanitized",
                    "bytes": 1,
                    "original_sha256": "a" * 64,
                    "sanitized_sha256": "b" * 64,
                    "pickle_roundtrip_verification": {"status": "passed"},
                }
            )
        else:
            manifest_rows.append(
                {
                    "id": identifier,
                    "role": role,
                    "status": status,
                    "original_sha256": None,
                    "sanitized_sha256": None,
                }
            )
        blocks.append("\n".join(lines))
    (inventory_dir / "external_checkpoint_inventory.toml").write_text(
        'schema = "sr_filter.external_checkpoint_inventory.v1"\n\n' + "\n\n".join(blocks) + "\n",
        encoding="utf-8",
    )
    manifest_path = artifact_dir / "external_checkpoints.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "sr_filter.external_checkpoint_manifest.v1",
                "status": "pending_gpu_outputs",
                "checkpoints": manifest_rows,
            }
        ),
        encoding="utf-8",
    )

    assert module.validate_external_checkpoints(tmp_path) == {
        "status": "pending_gpu_outputs",
        "sanitized": 5,
        "pending": 3,
    }

    manifest_rows[5].update(
        {
            "status": "sanitized",
            "bytes": 1,
            "original_sha256": "c" * 64,
            "sanitized_sha256": "d" * 64,
            "pickle_roundtrip_verification": {"status": "passed"},
        }
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "sr_filter.external_checkpoint_manifest.v1",
                "status": "pending_gpu_outputs",
                "checkpoints": manifest_rows,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Pending checkpoint status differs"):
        module.validate_external_checkpoints(tmp_path)
