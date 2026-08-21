#!/usr/bin/env python3
"""Validate the compact release artifacts and their claim-level invariants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tomllib
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

ONLINE_FIELDS = [
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
]

ENDPOINT_FIELDS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_checksum_file(category: Path) -> int:
    count = 0
    for line in (category / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = category / relative
        require(path.is_file(), f"Missing checksum target: {path}")
        require(sha256(path) == expected, f"Checksum mismatch: {path}")
        count += 1
    return count


def validate_manifest(root: Path) -> tuple[int, int]:
    manifest = json.loads((root / "artifacts/manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("schema") == "sr_filter.artifact_manifest.v1", "Wrong manifest schema")
    seen = set()
    for record in manifest["files"]:
        path = root / record["path"]
        require(path.is_file(), f"Manifest target missing: {path}")
        require(path.stat().st_size == int(record["bytes"]), f"Manifest byte mismatch: {path}")
        require(sha256(path) == record["sha256"], f"Manifest digest mismatch: {path}")
        require(record["path"] not in seen, f"Duplicate manifest path: {record['path']}")
        seen.add(record["path"])
    required_pickles = {
        "artifacts/small_scale/matched_noise/rbm490_vit10.pkl",
        "artifacts/small_scale/matched_state/alternating_polish_vit_step0100_to_direct_rbm_a2_figure1_style.pkl",
    }
    require(required_pickles <= seen, "Manifest must cover both bundled small-scale pickles")
    require("artifacts/external_checkpoints.json" in seen, "External checkpoint inventory is not manifested")
    for record in manifest["configs"]:
        path = root / record["path"]
        require(path.is_file() and path.stat().st_size == int(record["bytes"]), f"Config mismatch: {path}")
        require(sha256(path) == record["sha256"], f"Config digest mismatch: {path}")
        with path.open("rb") as handle:
            tomllib.load(handle)
    return len(manifest["files"]), len(manifest["configs"])


def validate_external_checkpoints(root: Path) -> dict[str, int | str]:
    payload = json.loads((root / "artifacts/external_checkpoints.json").read_text(encoding="utf-8"))
    with (root / "configs/experiments/external_checkpoint_inventory.toml").open("rb") as handle:
        inventory = tomllib.load(handle)
    require(payload.get("schema") == "sr_filter.external_checkpoint_manifest.v1", "Wrong external manifest schema")
    records = payload["checkpoints"]
    inventory_records = inventory["checkpoints"]
    require(len(records) == 8, "External inventory must contain two source and six final checkpoints")
    require(Counter(row["role"] for row in records) == {"source": 2, "final_endpoint": 6}, "Wrong external roles")
    expected_by_id = {row["id"]: row for row in inventory_records}
    require(len(expected_by_id) == 8, "Frozen external checkpoint inventory must have eight unique IDs")
    require({row["id"] for row in records} == set(expected_by_id), "External manifest/inventory IDs differ")
    sanitized = [row for row in records if row["status"] == "sanitized"]
    pending = [row for row in records if row["status"] != "sanitized"]
    for row in records:
        expected = expected_by_id[row["id"]]
        if expected["status"] == "complete":
            require(row["status"] == "sanitized", f"Completed checkpoint was not sanitized: {row['id']}")
            require(int(row["bytes"]) > 0, f"Missing sanitized byte size: {row['id']}")
            require(
                len(row["original_sha256"]) == 64 and len(row["sanitized_sha256"]) == 64,
                f"Missing checkpoint hashes: {row['id']}",
            )
            require(
                row["original_sha256"] == expected["expected_original_sha256"],
                f"Original checkpoint hash differs from frozen inventory: {row['id']}",
            )
            require(
                row["pickle_roundtrip_verification"]["status"] == "passed",
                f"Checkpoint load verification failed: {row['id']}",
            )
        else:
            require(
                row["status"] == expected["status"],
                f"Pending checkpoint status differs from frozen inventory: {row['id']}",
            )
            require(not row.get("original_sha256"), f"Pending checkpoint leaked a stale source hash: {row['id']}")
            require(not row.get("sanitized_sha256"), f"Pending checkpoint leaked a stale staged hash: {row['id']}")
    require(
        len(sanitized) == sum(row["status"] == "complete" for row in inventory_records),
        "External staging completion count differs from frozen inventory",
    )
    require(
        len(pending) == sum(row["status"] != "complete" for row in inventory_records),
        "External staging pending count differs from frozen inventory",
    )
    return {"status": payload["status"], "sanitized": len(sanitized), "pending": len(pending)}


def validate_online(root: Path) -> dict[str, object]:
    base = root / "artifacts/online_training"
    expected_ids = {
        "branch_seed0": "online.j1j2.no_symmetry.branch2500.seed0.v1",
        "branch_seed1": "online.j1j2.no_symmetry.branch1800.seed1.v1",
    }
    expected_curve_targets = {
        "branch_seed0": {"shared_sr": 550, "sr": 500, "spring": 500, "ms_sr": 125},
        "branch_seed1": {"shared_sr": 1, "sr": 500, "spring": 500, "ms_sr": 125},
    }
    reports = {}
    for branch, experiment_id in expected_ids.items():
        directory = base / branch
        fields, curves = csv_rows(directory / "training_curve.csv")
        require(fields == ONLINE_FIELDS, f"Unexpected online schema: {branch}")
        require(all(row["experiment_id"] == experiment_id for row in curves), f"Mixed IDs: {branch}")
        require(all(row["run_status"] == "complete" for row in curves), f"Incomplete curve row leaked: {branch}")
        counts = Counter(row["arm"] for row in curves)
        targets = expected_curve_targets[branch]
        for arm, count in counts.items():
            require(arm in targets, f"Unexpected {branch}/{arm} curve")
            require(count == targets[arm], f"Wrong {branch}/{arm} curve length: {count}")
        require(counts["shared_sr"] == targets["shared_sr"], f"Missing branch origin/source curve: {branch}")
        require(counts["sr"] == 500, f"Validated SR horizon is not 500: {branch}")
        if branch == "branch_seed0":
            require(counts["spring"] == 500, "Seed-0 validated SPRING curve is missing")
        else:
            require(counts["spring"] in (0, 500), "Seed-1 SPRING must be pending or complete")
        require(counts["ms_sr"] in (0, 125), f"MS-SR must be pending or exactly complete: {branch}")
        for row in curves:
            require(math.isfinite(float(row["energy_per_site"])), f"Non-finite curve energy: {branch}")
            if row["curve_segment"].startswith("post_branch"):
                require(int(row["sampling_budget_units_since_branch"]) <= 500, f"Budget >500: {branch}")

        by_arm = {
            arm: sorted(
                (row for row in curves if row["arm"] == arm),
                key=lambda row: int(row["optimizer_step_since_branch"] or 0),
            )
            for arm in targets
        }
        for arm, batches in (("sr", 1), ("spring", 1), ("ms_sr", 4)):
            rows = by_arm[arm]
            if not rows:
                continue
            require(
                [int(row["optimizer_step_since_branch"]) for row in rows]
                == list(range(1, len(rows) + 1)),
                f"Non-contiguous optimizer steps: {branch}/{arm}",
            )
            require(
                all(int(row["batches_per_update"]) == batches for row in rows),
                f"Wrong batches/update: {branch}/{arm}",
            )
            require(
                all(
                    int(row["sampling_budget_units_since_branch"])
                    == batches * int(row["optimizer_step_since_branch"])
                    for row in rows
                ),
                f"Wrong equal-budget accounting: {branch}/{arm}",
            )
            require(
                all(
                    int(row["budget_equivalent_global_step"])
                    == int(row["source_checkpoint_step"])
                    + int(row["sampling_budget_units_since_branch"])
                    for row in rows
                ),
                f"Wrong global budget mapping: {branch}/{arm}",
            )
        if by_arm["ms_sr"]:
            require(
                {row["optimizer"] for row in by_arm["ms_sr"]}
                == {"ms_sr_fixed_grid_exact_loo"},
                f"Legacy/non-exact MS-SR curve leaked: {branch}",
            )

        shared = by_arm["shared_sr"]
        if branch == "branch_seed1":
            origin = shared[0]
            require(origin["curve_segment"] == "branch_origin", "Seed-1 origin must be checkpoint-derived")
            require(int(origin["legacy_iteration"]) == 1800, "Seed-1 origin iteration drift")
            require(int(origin["optimizer_step_since_branch"]) == 0, "Seed-1 origin optimizer step drift")
            require(int(origin["sampling_budget_units_since_branch"]) == 0, "Seed-1 origin budget drift")
            require(int(origin["budget_equivalent_global_step"]) == 1800, "Seed-1 origin global step drift")
        else:
            require(max(int(row["legacy_iteration"]) for row in shared) == 2500, "Seed-0 source curve misses branch point")

        endpoint_fields, endpoints = csv_rows(directory / "endpoint_evaluations.csv")
        require(endpoint_fields == ENDPOINT_FIELDS, f"Unexpected endpoint schema: {branch}")
        require(len(endpoints) == 6, f"Endpoint contract must have six rows: {branch}")
        keys = {(row["arm"], int(row["eval_seed"])) for row in endpoints}
        require(
            keys == {(arm, seed) for arm in ("sr", "spring", "ms_sr") for seed in (12345, 98765)},
            f"Wrong endpoint arm/seed matrix: {branch}",
        )
        completed = Counter()
        for row in endpoints:
            require(int(row["rounds"]) == 512, f"Wrong endpoint rounds: {branch}")
            require(int(row["burn_in_rounds"]) == 5, f"Wrong endpoint burn-in: {branch}")
            require(int(row["samples_per_round"]) == 8192, f"Wrong endpoint samples: {branch}")
            require(int(row["total_samples"]) == 4_194_304, f"Wrong endpoint total samples: {branch}")
            if row["audit_status"] == "audited_complete":
                completed[row["arm"]] += 1
                for column in (
                    "energy_per_site",
                    "sigma_per_site",
                    "sigma_round_level_per_site",
                    "sigma_block8_level_per_site",
                ):
                    value = float(row[column])
                    require(math.isfinite(value), f"Non-finite endpoint {column}: {branch}")
                    if column.startswith("sigma"):
                        require(value > 0.0, f"Non-positive endpoint {column}: {branch}")
            else:
                require(
                    all(
                        row[column] == ""
                        for column in (
                            "energy_per_site",
                            "sigma_per_site",
                            "sigma_round_level_per_site",
                            "sigma_block8_level_per_site",
                        )
                    ),
                    f"Pending endpoint contains measurements: {branch}",
                )
        status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        require(
            status.get("curve_rows") == {arm: counts[arm] for arm in targets},
            f"Status curve counts disagree: {branch}",
        )
        require(status.get("target_curve_rows") == targets, f"Status curve targets disagree: {branch}")
        consistency = status.get("endpoint_replica_consistency")
        require(isinstance(consistency, dict), f"Missing endpoint consistency audit: {branch}")
        completed_arms = {arm for arm in ("sr", "spring", "ms_sr") if completed[arm] == 2}
        require(set(consistency) == completed_arms, f"Endpoint consistency arms disagree: {branch}")
        for arm in completed_arms:
            arm_rows = [row for row in endpoints if row["arm"] == arm]
            difference = abs(
                float(arm_rows[0]["energy_per_site"])
                - float(arm_rows[1]["energy_per_site"])
            )
            denominator = math.hypot(
                float(arm_rows[0]["sigma_per_site"]),
                float(arm_rows[1]["sigma_per_site"]),
            )
            expected_z = difference / denominator
            arm_audit = consistency[arm]
            require(arm_audit.get("status") == "pass", f"Endpoint consistency failed: {branch}/{arm}")
            require(
                float(arm_audit.get("max_allowed_replica_difference_z", -1)) == 5.0,
                f"Endpoint consistency threshold drift: {branch}/{arm}",
            )
            require(
                math.isclose(float(arm_audit.get("replica_difference_z", -1)), expected_z),
                f"Endpoint consistency statistic drift: {branch}/{arm}",
            )
            require(expected_z <= 5.0, f"Endpoint replicas disagree: {branch}/{arm}")
        protocol_audit = status.get("ms_sr_protocol_audit")
        require(isinstance(protocol_audit, dict), f"Missing MS-SR protocol audit: {branch}")
        if counts["ms_sr"] == 125:
            expected_audit = {
                "status": "audited_complete",
                "update_protocol": "fixed_grid_exact_loo",
                "stacking_protocol": "exact_leave_one_batch_out",
                "lambda_policy": "fixed_grid",
                "lambda_grid": [0.01, 0.001, 0.0001, 0.00001],
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
            }
            require(
                all(protocol_audit.get(key) == value for key, value in expected_audit.items()),
                f"Exact MS-SR audit drift: {branch}",
            )
            require(len(protocol_audit.get("checkpoint_sha256", "")) == 64, f"Missing exact checkpoint hash: {branch}")
        else:
            require(protocol_audit.get("status") == "pending_gpu_execution", f"Pending MS-SR audit drift: {branch}")
        gate = all(completed[arm] == 2 for arm in ("sr", "spring", "ms_sr")) and all(
            counts[arm] == targets[arm] for arm in targets if arm != "shared_sr"
        )
        require(bool(status["camera_ready_gate"]) == gate, f"Incorrect camera-ready gate: {branch}")
        require((status["status"] == "complete") == gate, f"Incorrect completion status: {branch}")
        if not gate:
            expected_status = (
                "pending_endpoint_evaluation"
                if all(counts[arm] == targets[arm] for arm in ("sr", "spring", "ms_sr"))
                else "pending_gpu_execution"
            )
            require(status["status"] == expected_status, f"Incorrect pending status: {branch}")

        provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        require(provenance.get("ms_sr_protocol_audit") == protocol_audit, f"MS-SR provenance/status mismatch: {branch}")
        if branch == "branch_seed1":
            origin_provenance = provenance.get("branch_origin")
            require(isinstance(origin_provenance, dict), "Seed-1 origin provenance is missing")
            require(origin_provenance.get("checkpoint_step") == 1800, "Seed-1 origin checkpoint drift")
            require(origin_provenance.get("n_sites") == 64, "Seed-1 origin site count drift")
            require(
                math.isclose(float(origin_provenance["energy_per_site"]), float(shared[0]["energy_per_site"])),
                "Seed-1 origin scalar/provenance mismatch",
            )
        reports[branch] = {"curve_rows": len(curves), "completed_endpoints": dict(completed), "gate": gate}

    diag_fields, diagnostics = csv_rows(base / "branch_seed0/endpoint_diagnostics.csv")
    require(diag_fields == ENDPOINT_FIELDS and len(diagnostics) == 1, "Missing seed-0 endpoint diagnostic")
    require(int(diagnostics[0]["rounds"]) == 128, "SPRING diagnostic must remain labeled 128 rounds")
    return reports


def validate_robustness(root: Path) -> dict[str, int]:
    base = root / "artifacts/robustness"
    expected_counts = {
        "shift_sweep_checkpoints.csv": 392,
        "shift_sweep_repeats.csv": 7840,
        "method_ablation_checkpoints.csv": 594,
        "method_ablation_repeats.csv": 11880,
        "bagged_fixed_shift_checkpoints.csv": 193,
        "bagged_fixed_shift_repeats.csv": 3860,
        "source_input_checksums.csv": 550,
    }
    report = {}
    for filename, expected in expected_counts.items():
        _, rows = csv_rows(base / filename)
        require(len(rows) == expected, f"Wrong row count for {filename}: {len(rows)}")
        report[filename] = len(rows)

    _, methods = csv_rows(base / "method_ablation_checkpoints.csv")
    tfim = [row for row in methods if row["regime"].startswith("tfim_") or row["regime"] == "tfim_canonical"]
    require({int(row["k"]) for row in tfim} >= {2, 3, 4, 5}, "Missing TFIM K=2..5 coverage")
    grids = {row["quantile_grid_id"] for row in tfim}
    require({"qgrid_a", "qgrid_b", "paper_k4", "k2_default", "k3_default", "k5_default"} <= grids, "Missing NTK grids")
    source_shifts = {float(row["source_training_shift"]) for row in tfim}
    require({1e-3, 1e-4, 1e-5} <= source_shifts, "Missing source-shift robustness")

    _, bagged = csv_rows(base / "bagged_fixed_shift_checkpoints.csv")
    require({int(row["k"]) for row in bagged if row["system"] == "tfim"} == {2, 3, 4, 5}, "Missing bagged K grid")
    require(all(row["coverage_status"] == "complete" for row in bagged), "Incomplete fixed-shift matrix leaked")
    require(
        not any(row["regime"] == "j1j2_no_symmetry" and int(row["checkpoint_step"]) == 1000 for row in bagged),
        "Excluded J1-J2 step-1000 fixed-shift checkpoint leaked",
    )

    _, ssm = csv_rows(base / "ssm_boundary_oracle_summary.csv")
    by_method = {row["method"]: row for row in ssm}
    require(len(ssm) == 2, "SSM negative-result summary must have two methods")
    require(math.isclose(float(by_method["bagged_grid_best_shift"]["r_val_raw_mean"]), 32.792262503627946), "SSM bagged risk drift")
    require(math.isclose(float(by_method["ms_sr"]["r_val_raw_mean"]), 158.57842083276023), "SSM MS-SR risk drift")
    require(all(row["lambda_star_at_diagnostic_boundary"] == "true" for row in ssm), "SSM boundary warning lost")
    require(
        by_method["bagged_grid_best_shift"]["result_label"] == "boundary-grid evaluation-oracle bagged SR",
        "SSM negative-result label drift",
    )
    return report


def validate_wall_clock(root: Path) -> dict[str, int]:
    base = root / "artifacts/wall_clock"
    _, timings = csv_rows(base / "timings_long.csv")
    require(len(timings) == 45, "Wall-clock table must contain 3 pairs x 3 methods x 5 repeats")
    require(Counter(row["method"] for row in timings) == {"sr": 15, "bagged_sr": 15, "ms_sr": 15}, "Wrong timing methods")
    require(all(row["apply_update"] == "false" for row in timings), "Timing claim accidentally includes parameter updates")
    _, pairing = csv_rows(base / "pairing_verification.csv")
    require(len(pairing) == 15, "Pairing table must contain 15 pair/repeat rows")
    require(all(row["first_batch_matches_sr"] == "true" and row["all_k4_batches_match"] == "true" for row in pairing), "Sample pairing failed")
    return {"timing_rows": len(timings), "pairing_rows": len(pairing), "raw_json": len(list((base / "raw").glob("*.json")))}


def validate_privacy(root: Path) -> int:
    forbidden = (
        b"/home/",
        b"/Users/",
        b"benign_" + b"overfitting_nqs",
        b"beingin_" + b"overfitting_nqs",
    )
    checked = 0
    for category in ("online_training", "robustness", "wall_clock"):
        for path in (root / "artifacts" / category).rglob("*"):
            if not path.is_file():
                continue
            payload = path.read_bytes()
            require(not any(token in payload for token in forbidden), f"Private path/name leaked: {path}")
            checked += 1
    return checked


def main() -> None:
    root = parse_args().repo_root.resolve()
    manifest_files, manifest_configs = validate_manifest(root)
    report = {
        "status": "ok",
        "manifest": {"files": manifest_files, "configs": manifest_configs},
        "category_checksums": {
            category: validate_checksum_file(root / "artifacts" / category)
            for category in ("online_training", "robustness", "wall_clock")
        },
        "online_training": validate_online(root),
        "external_checkpoints": validate_external_checkpoints(root),
        "robustness": validate_robustness(root),
        "wall_clock": validate_wall_clock(root),
        "privacy_files_checked": validate_privacy(root),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
