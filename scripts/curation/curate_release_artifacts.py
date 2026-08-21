#!/usr/bin/env python3
"""Build the public release artifact bundle from a trusted experiment checkout.

The private checkout contains pickle payloads and machine-local paths that must
not be published.  This script extracts only scalar/tabular result data, assigns
stable experiment identifiers, and writes deterministic CSV/JSON artifacts.

Usage:
    python scripts/curation/curate_release_artifacts.py \
        --source /path/to/trusted/experiment-checkout

The source is trusted: several historical inputs are Python pickle files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "artifacts"

METHOD_NAMES = {
    "single_best_fixed_lambda": "single_grid_best_shift",
    "same_batch_multilambda_uniform_k4": "same_batch_multishift_uniform",
    "same_batch_multilambda_stacking_k4": "same_batch_multishift_stacking",
    "indep_single_lambda_uniform_k4": "bagged_grid_best_shift",
    "indep_multilambda_uniform_k4": "independent_multishift_uniform",
    "indep_multilambda_stacking_k4": "ms_sr",
}

DIAGNOSTIC_GRID = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)

FIG2_SOURCES = (
    ("tfim_canonical", "tfim", 1e-4, "large_scale/20260422/results/fig2/tfim/step*/diag_*.pkl"),
    (
        "j1j2_symmetry_enabled",
        "j1j2",
        1e-4,
        "large_scale/20260422/results/fig2/j1j2/step*/diag_*.pkl",
    ),
    (
        "j1j2_no_symmetry",
        "j1j2",
        1e-4,
        "large_scale/20260428/no_symm_j1j2/results/fig2/j1j2/step*/diag_*.pkl",
    ),
    (
        "ssm_no_symmetry",
        "ssm",
        1e-4,
        "large_scale/20260428/no_symm_ssm/results/fig2/ssm/step*/diag_*.pkl",
    ),
    (
        "tfim_early_extension",
        "tfim",
        1e-4,
        "large_scale/20260428_early/results/fig2/tfim/step*/diag_*.pkl",
    ),
    (
        "tfim_source_shift_1e-3",
        "tfim",
        1e-3,
        "large_scale/20260428_lt1e3/results/fig2/tfim/step*/diag_*.pkl",
    ),
    (
        "tfim_source_shift_1e-5",
        "tfim",
        1e-5,
        "large_scale/20260428_lt1e5/results/fig2/tfim/step*/diag_*.pkl",
    ),
)

FIG3_SOURCES = (
    (
        "j1j2_symmetry_enabled",
        "j1j2",
        1e-4,
        "paper_k4",
        "large_scale/20260422/results/fig3/j1j2/step*/k4_eval.pkl",
    ),
    (
        "tfim_k2_default",
        "tfim",
        1e-4,
        "k2_default",
        "large_scale/20260428/results/fig3/tfim/step*/k2_eval.pkl",
    ),
    (
        "tfim_k3_default",
        "tfim",
        1e-4,
        "k3_default",
        "large_scale/20260428/results/fig3/tfim/step*/k3_eval.pkl",
    ),
    (
        "tfim_k4_default",
        "tfim",
        1e-4,
        "paper_k4",
        "large_scale/20260428/results/fig3/tfim/step*/k4_eval.pkl",
    ),
    (
        "tfim_k5_default",
        "tfim",
        1e-4,
        "k5_default",
        "large_scale/20260428/results/fig3/tfim/step*/k5_eval.pkl",
    ),
    (
        "tfim_qgrid_a",
        "tfim",
        1e-4,
        "qgrid_a",
        "large_scale/20260428_qgridA/results/fig3/tfim/step*/k4_eval.pkl",
    ),
    (
        "tfim_qgrid_b",
        "tfim",
        1e-4,
        "qgrid_b",
        "large_scale/20260428_qgridB/results/fig3/tfim/step*/k4_eval.pkl",
    ),
    (
        "tfim_source_shift_1e-3",
        "tfim",
        1e-3,
        "paper_k4",
        "large_scale/20260428_lt1e3/results/fig3/tfim/step*/k4_eval.pkl",
    ),
    (
        "tfim_source_shift_1e-5",
        "tfim",
        1e-5,
        "paper_k4",
        "large_scale/20260428_lt1e5/results/fig3/tfim/step*/k4_eval.pkl",
    ),
    (
        "tfim_early_extension",
        "tfim",
        1e-4,
        "paper_k4",
        "large_scale/20260428_early/results/fig3/tfim/step*/k4_eval.pkl",
    ),
    (
        "j1j2_no_symmetry",
        "j1j2",
        1e-4,
        "paper_k4",
        "large_scale/20260428/no_symm_j1j2/results/fig3/j1j2/step*/k4_eval.pkl",
    ),
    (
        "ssm_no_symmetry",
        "ssm",
        1e-4,
        "paper_k4",
        "large_scale/20260428/no_symm_ssm/results/fig3/ssm/step*/k4_eval.pkl",
    ),
)

BAGGED_SOURCES = (
    (
        "tfim_fixed_shift_k2",
        "tfim",
        "large_scale/20260428/results/fig3/tfim/step*/bagged_lambda_eval_k2.pkl",
    ),
    (
        "tfim_fixed_shift_k3",
        "tfim",
        "large_scale/20260428/results/fig3/tfim/step*/bagged_lambda_eval_k3.pkl",
    ),
    (
        "tfim_fixed_shift_k4",
        "tfim",
        "large_scale/20260428/results/fig3/tfim/step*/bagged_lambda_eval_k4.pkl",
    ),
    (
        "tfim_fixed_shift_k4_ext",
        "tfim",
        "large_scale/20260428/results/fig3/tfim/step*/bagged_lambda_eval_k4_ext.pkl",
    ),
    (
        "tfim_fixed_shift_k5",
        "tfim",
        "large_scale/20260428/results/fig3/tfim/step*/bagged_lambda_eval_k5.pkl",
    ),
    (
        "j1j2_no_symmetry_fixed_shift_k4",
        "j1j2",
        "large_scale/20260428/no_symm_j1j2/results/fig3/j1j2/step*/bagged_lambda_eval_k4.pkl",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Trusted private repository root")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Public artifacts directory")
    parser.add_argument(
        "--online-results-root",
        type=Path,
        default=None,
        help="Optional clean-run root containing branch_seed*/{spring,mssr_exact}",
    )
    parser.add_argument(
        "--endpoint-results-root",
        type=Path,
        default=None,
        help="Optional endpoint root containing branch_seed0 and branch_seed1 JSON results",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


COMPLEX_SOLVER_CAST_MARKERS = (
    "scatter inputs have incompatible types",
    "ComplexWarning: Casting complex values to real",
)


def require_no_complex_solver_cast(train_log_path: Path) -> None:
    """Reject optimization logs that silently cast a complex SR update to real."""

    if not train_log_path.is_file():
        raise ValueError(f"Registered online result lacks its training log: {train_log_path}")
    train_log = train_log_path.read_text(encoding="utf-8", errors="replace")
    if any(marker in train_log for marker in COMPLEX_SOLVER_CAST_MARKERS):
        raise ValueError(
            f"{train_log_path}: online result contains a complex-to-real solver cast"
        )


def validate_registered_online_config(
    result_dir: Path,
    *,
    expected: dict[str, Any],
    expected_init_checkpoint: Path,
) -> dict[str, Any]:
    """Bind a reused online trajectory to the complete registered run contract."""

    config_path = result_dir / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Registered online result lacks its config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mismatches = [
        f"{key}: expected {value!r}, got {config.get(key)!r}"
        for key, value in expected.items()
        if config.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"{config_path}: registered online contract mismatch: " + "; ".join(mismatches)
        )
    recorded_init = Path(str(config.get("init_from", "")))
    if (
        not expected_init_checkpoint.is_file()
        or not recorded_init.is_file()
        or sha256(recorded_init) != sha256(expected_init_checkpoint)
    ):
        raise ValueError(
            f"{config_path}: initialization checkpoint does not match the registered source"
        )
    require_no_complex_solver_cast(result_dir / "logs/train.log")
    return config


def source_revision(source: Path) -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(source), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in trusted input {path}")
    return payload


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def to_float(value: str | float | int | None) -> float | str:
    if value is None or value == "":
        return ""
    result = float(value)
    return "" if not math.isfinite(result) else result


def portable_checkpoint_id(system: str, training_shift: float, step: int) -> str:
    return f"{system}/train_shift_{training_shift:.0e}/step_{step:06d}"


def selection_label(method: str) -> str:
    if method in {"single_grid_best_shift", "bagged_grid_best_shift"}:
        return "evaluation_grid_best_same_validation"
    if "multishift" in method or method == "ms_sr":
        return "ntk_quantile_grid"
    return "fixed"


def is_grid_boundary(value: float) -> bool:
    return math.isclose(value, min(DIAGNOSTIC_GRID)) or math.isclose(value, max(DIAGNOSTIC_GRID))


def online_row(
    raw: dict[str, str],
    *,
    experiment_id: str,
    status: str,
    branch_step: int,
    branch_seed: int,
    arm: str,
    optimizer: str,
    segment: str,
    legacy_iteration: int,
    optimizer_step: int | str,
    batches: int,
    budget: int | str,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "run_status": status,
        "system": "j1j2",
        "source_checkpoint_step": branch_step,
        "branch_rng_seed": branch_seed,
        "arm": arm,
        "optimizer": optimizer,
        "curve_segment": segment,
        "legacy_iteration": legacy_iteration,
        "optimizer_step_since_branch": optimizer_step,
        "batches_per_update": batches,
        "samples_per_batch": 8192,
        "sampling_budget_units_since_branch": budget,
        "budget_equivalent_global_step": legacy_iteration if budget == "" else branch_step + int(budget),
        "energy_per_site": float(raw["energy_per_site"]),
        "energy_sigma_total": to_float(raw.get("energy_sigma")),
        "energy_variance_total": to_float(raw.get("energy_variance")),
        "acceptance": to_float(raw.get("acceptance")),
        "step_wall_time_s": to_float(raw.get("step_wall_time_s")),
    }


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

ONLINE_MSSR_FIXED_GRID = (1e-2, 1e-3, 1e-4, 1e-5)
ONLINE_MSSR_EXPECTED_STEPS = 125
ONLINE_MSSR_EXPECTED_BUDGET_UNITS = 500
MAX_ENDPOINT_REPLICA_DIFFERENCE_Z = 5.0
ONLINE_MSSR_RESULT_SUFFIX = Path(
    "large_scale/j1j2_mixture_results/"
    "indep_batch__stacking__1e-02__1e-03__1e-04__1e-05/results"
)


def online_branch_root(root: Path, branch: str) -> Path:
    """Resolve either the common online root or one explicitly scoped branch root."""

    candidate = root / branch
    return candidate if candidate.exists() else root


def exact_mssr_result_dir(root: Path, branch: str) -> Path:
    """Return the only directory eligible for the camera-ready MS-SR arm.

    In particular, this never resolves the historical ``mssr`` directories,
    whose callback implementation consumed extra, non-LOO batches.
    """

    return online_branch_root(root, branch) / "mssr_exact" / ONLINE_MSSR_RESULT_SUFFIX


def _require_exact_mssr_mapping(
    payload: dict[str, Any],
    *,
    context: str,
    expected_seed: int,
    expected_source_native_step: int,
) -> None:
    expected = {
        "sr_update_protocol": "fixed_grid_exact_loo",
        "sr_variant": "indep_batch",
        "sr_weight_scheme": "stacking",
        "ngd_method": "sr",
        "total_steps": ONLINE_MSSR_EXPECTED_STEPS,
        "schedule_total_steps": ONLINE_MSSR_EXPECTED_STEPS,
        "seed": expected_seed,
        "init_from_step": expected_source_native_step,
        "n_samples": 8192,
        "sr_prediction_chunk": 2048,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{context}: expected {key}={value!r}, got {payload.get(key)!r}")
    if tuple(float(value) for value in payload.get("sr_lambda_grid", ())) != ONLINE_MSSR_FIXED_GRID:
        raise ValueError(f"{context}: exact MS-SR fixed grid is missing or changed")
    for key in (
        "sr_weight_fit_samples",
        "sr_lambda_quantiles",
        "sr_lambda_ema_rho",
        "sr_weight_ema_rho",
    ):
        if payload.get(key) is not None:
            raise ValueError(f"{context}: {key} must be unset for exact online MS-SR")
    if [int(value) for value in payload.get("phase_lengths", ())] != [ONLINE_MSSR_EXPECTED_STEPS]:
        raise ValueError(f"{context}: exact MS-SR must be one 125-update phase")
    if [int(value) for value in payload.get("phase_samples", ())] != [8192]:
        raise ValueError(f"{context}: exact MS-SR must use 8192 samples per candidate batch")


def _require_exact_mssr_metadata(payload: dict[str, Any], *, context: str) -> None:
    expected = {
        "method": "ms_sr",
        "variant": "indep_batch",
        "weight_scheme": "stacking",
        "update_protocol": "fixed_grid_exact_loo",
        "stacking_protocol": "exact_leave_one_batch_out",
        "lambda_source": "fixed_grid",
        "k": 4,
        "samples_per_candidate_batch": 8192,
        "candidate_sample_counts": [8192, 8192, 8192, 8192],
        "candidate_batches": 4,
        "candidate_solves": 4,
        "sampling_solve_units_this_update": 4,
        "sampling_solve_units_cumulative": ONLINE_MSSR_EXPECTED_BUDGET_UNITS,
        "base_driver_sampling_calls": 0,
        "base_driver_solves": 0,
        "weight_fit_sampling_calls": 0,
        "weight_fit_solves": 0,
        "training_energy_source": "pooled_candidate_batches",
        "training_energy_sample_count": 4 * 8192,
        "cross_prediction_launches": 4,
        "cross_prediction_directions": 12,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{context}: expected {key}={value!r}, got {payload.get(key)!r}")
    if tuple(float(value) for value in payload.get("lambda_grid", ())) != ONLINE_MSSR_FIXED_GRID:
        raise ValueError(f"{context}: final update did not use the registered fixed grid")
    weights = payload.get("weights")
    if (
        not isinstance(weights, list)
        or len(weights) != 4
        or any(not math.isfinite(float(value)) or float(value) < 0.0 for value in weights)
        or not math.isclose(sum(float(value) for value in weights), 1.0, rel_tol=1e-7, abs_tol=1e-7)
    ):
        raise ValueError(f"{context}: final exact-LOO stacking weights are invalid")


def validated_exact_mssr_run(
    result_dir: Path,
    *,
    expected_seed: int,
    expected_source_native_step: int,
    expected_source_sha256: str,
) -> tuple[list[dict[str, str]], Path, dict[str, Any]] | None:
    """Audit a completed exact-budget MS-SR run, or return ``None`` while pending."""

    summary_path = result_dir / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed":
        return None

    metrics_path = result_dir / "train_metrics.csv"
    config_path = result_dir / "config.json"
    train_log_path = result_dir / "logs/train.log"
    checkpoint_path = result_dir / "checkpoints/checkpoint_step000125.pkl"
    audit_path = checkpoint_path.with_suffix(".audit.json")
    for path in (metrics_path, config_path, train_log_path, checkpoint_path, audit_path):
        if not path.is_file():
            raise ValueError(f"Completed exact MS-SR run lacks required file: {path}")

    require_no_complex_solver_cast(train_log_path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    _require_exact_mssr_mapping(
        config,
        context=str(config_path),
        expected_seed=expected_seed,
        expected_source_native_step=expected_source_native_step,
    )
    # The summary does not repeat every run field, so compare every field that
    # it does promise and require its saved-step audit separately.
    for key in (
        "sr_update_protocol",
        "sr_variant",
        "sr_weight_scheme",
        "sr_lambda_grid",
        "sr_lambda_quantiles",
        "sr_lambda_ema_rho",
        "sr_weight_ema_rho",
        "ngd_method",
        "init_from_step",
        "total_steps",
    ):
        if summary.get(key) != config.get(key):
            raise ValueError(f"{summary_path}: summary/config mismatch for {key}")
    if ONLINE_MSSR_EXPECTED_STEPS not in {
        int(value) for value in summary.get("checkpoint_steps_saved", ())
    }:
        raise ValueError(f"{summary_path}: final step 125 is not recorded as saved")
    if int((summary.get("latest_metrics") or {}).get("step", -1)) != ONLINE_MSSR_EXPECTED_STEPS:
        raise ValueError(f"{summary_path}: latest metrics are not from update 125")

    init_from = Path(str(config.get("init_from", "")))
    if not init_from.is_file() or sha256(init_from) != expected_source_sha256:
        raise ValueError(f"{config_path}: initialization checkpoint hash does not match the branch source")

    rows = load_csv(metrics_path)
    if len(rows) != ONLINE_MSSR_EXPECTED_STEPS:
        raise ValueError(f"{metrics_path}: expected exactly 125 metric rows")
    steps = [int(row["step"]) for row in rows]
    if steps != list(range(1, ONLINE_MSSR_EXPECTED_STEPS + 1)):
        raise ValueError(f"{metrics_path}: expected contiguous optimizer steps 1--125")
    if any(not math.isfinite(float(row["energy_per_site"])) for row in rows):
        raise ValueError(f"{metrics_path}: non-finite training energy")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("schema") != "sr_filter.online_optimizer_update_audit.v1"
        or audit.get("status") != "checkpoint_complete"
        or audit.get("checkpoint_filename") != checkpoint_path.name
        or audit.get("checkpoint_sha256") != sha256(checkpoint_path)
        or int(audit.get("checkpoint_step", -1)) != ONLINE_MSSR_EXPECTED_STEPS
        or audit.get("config_filename") != config_path.name
        or audit.get("config_sha256") != sha256(config_path)
        or audit.get("train_metrics_filename") != metrics_path.name
        or audit.get("train_metrics_sha256") != sha256(metrics_path)
    ):
        raise ValueError(f"{audit_path}: exact MS-SR audit bindings are invalid")
    checkpoint_config = audit.get("run_contract")
    if not isinstance(checkpoint_config, dict):
        raise ValueError(f"{audit_path}: missing run contract")
    _require_exact_mssr_mapping(
        checkpoint_config,
        context=str(audit_path),
        expected_seed=expected_seed,
        expected_source_native_step=expected_source_native_step,
    )
    metadata = audit.get("optimizer_update")
    if not isinstance(metadata, dict):
        raise ValueError(f"{audit_path}: missing exact MS-SR update audit")
    _require_exact_mssr_metadata(metadata, context=str(audit_path))
    return rows, checkpoint_path, {
        "status": "audited_complete",
        "update_protocol": "fixed_grid_exact_loo",
        "stacking_protocol": "exact_leave_one_batch_out",
        "lambda_policy": "fixed_grid",
        "lambda_grid": list(ONLINE_MSSR_FIXED_GRID),
        "k": 4,
        "optimizer_updates": ONLINE_MSSR_EXPECTED_STEPS,
        "samples_per_candidate_batch": 8192,
        "candidate_sample_counts_final_update": [8192, 8192, 8192, 8192],
        "candidate_batches_per_update": 4,
        "candidate_solves_per_update": 4,
        "sampling_solve_units": ONLINE_MSSR_EXPECTED_BUDGET_UNITS,
        "base_driver_sampling_calls": 0,
        "base_driver_solves": 0,
        "weight_fit_sampling_calls": 0,
        "weight_fit_solves": 0,
        "checkpoint_sha256": sha256(checkpoint_path),
    }


def branch_origin_row_from_checkpoint(
    checkpoint_path: Path,
    *,
    experiment_id: str,
    branch_step: int,
    branch_seed: int,
    n_sites: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract the recorded checkpoint energy at the exact branch origin."""

    checkpoint = load_pickle(checkpoint_path)
    if int(checkpoint.get("step", -1)) != branch_step:
        raise ValueError(f"{checkpoint_path}: source checkpoint step mismatch")
    config = checkpoint.get("config")
    history = checkpoint.get("energy_history")
    if not isinstance(config, dict) or not isinstance(history, dict):
        raise ValueError(f"{checkpoint_path}: source checkpoint lacks energy provenance")
    if config.get("system") != "j1j2" or int(config.get("n_samples", -1)) != 8192:
        raise ValueError(f"{checkpoint_path}: source system/sample contract mismatch")
    iterations = [int(value) for value in history.get("iters", ())]
    matching = [index for index, value in enumerate(iterations) if value == branch_step]
    if len(matching) != 1:
        raise ValueError(f"{checkpoint_path}: source energy history lacks a unique branch step")
    index = matching[0]
    try:
        energy_total = float(history["mean"][index])
        sigma_total = float(history["sigma"][index])
        variance_total = float(history["variance"][index])
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{checkpoint_path}: source energy history is incomplete") from error
    if not all(math.isfinite(value) for value in (energy_total, sigma_total, variance_total)):
        raise ValueError(f"{checkpoint_path}: source energy history contains non-finite values")
    row = online_row(
        {
            "energy_per_site": energy_total / int(n_sites),
            "energy_sigma": sigma_total,
            "energy_variance": variance_total,
        },
        experiment_id=experiment_id,
        status="complete",
        branch_step=branch_step,
        branch_seed=branch_seed,
        arm="shared_sr",
        optimizer="sr",
        segment="branch_origin",
        legacy_iteration=branch_step,
        optimizer_step=0,
        batches=1,
        budget=0,
    )
    provenance = {
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_step": branch_step,
        "source_field": f"energy_history at iters={branch_step}",
        "energy_total": energy_total,
        "energy_sigma_total": sigma_total,
        "energy_variance_total": variance_total,
        "n_sites": int(n_sites),
        "energy_per_site": energy_total / int(n_sites),
    }
    return row, provenance


def endpoint_row_from_payload(
    payload: dict[str, Any],
    *,
    experiment_id: str,
    branch_seed: int,
    arm: str,
    checkpoint_step: int,
    eval_run: int,
) -> dict[str, Any]:
    rounds = int(payload["rounds"])
    total_samples = int(payload["total_samples"])
    round_sigma = payload.get("sigma_round_level_per_site", payload.get("sigma_round_level"))
    block_sigma = payload.get("sigma_block8_level_per_site", payload.get("sigma_block8_level"))
    sigma = payload.get("sigma_per_site", payload.get("sigma_per_site_binned"))
    if sigma is None or round_sigma is None or block_sigma is None:
        raise ValueError("Endpoint payload lacks the required per-site standard errors")
    return {
        "experiment_id": experiment_id,
        "run_status": "complete",
        "system": "j1j2",
        "branch_rng_seed": branch_seed,
        "arm": arm,
        "checkpoint_step": checkpoint_step,
        "eval_run": eval_run,
        "eval_seed": int(payload.get("eval_seed", payload.get("evaluation_seed", payload["seed"]))),
        "rounds": rounds,
        "burn_in_rounds": int(payload.get("burn_in_rounds", payload.get("burn_in", 5))),
        "samples_per_round": int(payload.get("samples_per_round", total_samples // rounds)),
        "total_samples": total_samples,
        "energy_per_site": float(payload["energy_per_site"]),
        "sigma_per_site": float(sigma),
        "sigma_round_level_per_site": float(round_sigma),
        "sigma_block8_level_per_site": float(block_sigma),
        "audit_status": "audited_complete",
    }


def pending_endpoint_row(
    *,
    experiment_id: str,
    branch_seed: int,
    arm: str,
    checkpoint_step: int,
    eval_run: int,
    eval_seed: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "run_status": reason,
        "system": "j1j2",
        "branch_rng_seed": branch_seed,
        "arm": arm,
        "checkpoint_step": checkpoint_step,
        "eval_run": eval_run,
        "eval_seed": eval_seed,
        "rounds": 512,
        "burn_in_rounds": 5,
        "samples_per_round": 8192,
        "total_samples": 4194304,
        "energy_per_site": "",
        "sigma_per_site": "",
        "sigma_round_level_per_site": "",
        "sigma_block8_level_per_site": "",
        "audit_status": "pending_no_measurement",
    }


def validated_endpoint_overlays(
    root: Path | None,
    *,
    branch_seed: int,
    experiment_id: str,
    checkpoint_hashes: dict[str, str],
    checkpoint_steps: dict[str, int],
) -> dict[tuple[str, int], dict[str, Any]]:
    if root is None or not root.exists():
        return {}
    overlays: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("schema") != "sr_filter.endpoint_eval.v1":
            continue
        if int(payload.get("branch_seed", -1)) != branch_seed:
            continue
        if payload.get("experiment_id") != experiment_id:
            raise ValueError(f"Endpoint experiment ID mismatch in {path}")
        arm = str(payload["arm"])
        seed = int(payload.get("eval_seed", payload.get("evaluation_seed", payload["seed"])))
        if arm not in checkpoint_hashes:
            # A recognized arm cannot be overlaid until its exact accepted
            # training checkpoint is available. This also prevents stale
            # endpoint evaluations of either legacy MS-SR callback run from
            # entering the release while allowing curation to remain pending.
            if arm in checkpoint_steps:
                continue
            raise ValueError(f"Unexpected endpoint arm {arm!r} in {path}")
        expected_policy = {
            "sr": ("fixed_1e-4", 1),
            "spring": ("momentum_0.9_fixed_1e-4", 1),
            "ms_sr": ("fixed_grid_1e-2_1e-3_1e-4_1e-5_stacking", 4),
        }[arm]
        if payload.get("checkpoint_sha256") != checkpoint_hashes[arm]:
            raise ValueError(f"Endpoint checkpoint hash mismatch in {path}")
        if int(payload.get("checkpoint_step", -1)) != checkpoint_steps[arm]:
            raise ValueError(f"Endpoint checkpoint step mismatch in {path}")
        if payload.get("system") != "j1j2" or payload.get("method") != arm:
            raise ValueError(f"Endpoint system/method mismatch in {path}")
        if payload.get("lambda_policy") != expected_policy[0] or int(payload.get("k", -1)) != expected_policy[1]:
            raise ValueError(f"Endpoint lambda policy/K mismatch in {path}")
        if payload.get("run_status") != "complete" or payload.get("audit_status") != "pass":
            raise ValueError(f"Endpoint run is not audited-complete in {path}")
        if (
            int(payload["rounds"]) != 512
            or int(payload.get("burn_in_rounds", -1)) != 5
            or int(payload.get("samples_per_round", -1)) != 8192
            or int(payload["total_samples"]) != 4_194_304
        ):
            raise ValueError(f"Endpoint sampling contract mismatch in {path}")
        if seed not in (12345, 98765):
            raise ValueError(f"Unexpected endpoint seed {seed} in {path}")
        expected_replica = 1 if seed == 12345 else 2
        if (
            int(payload.get("evaluation_replica", -1)) != expected_replica
            or int(payload.get("eval_run", -1)) != expected_replica
            or int(payload.get("repeat_index", -1)) != 1
        ):
            raise ValueError(f"Endpoint replica metadata mismatch in {path}")
        if (
            int(payload.get("sampler_seed", -1)) != seed
            or int(payload.get("sampler_reset_count", -1)) != 1
            or payload.get("sampler_progression") != "single_seed_sequential_rounds"
            or payload.get("sampler_chain_name") != f"endpoint_eval_replica{expected_replica}"
        ):
            raise ValueError(f"Endpoint sampler progression mismatch in {path}")
        expected_all_indices = list(range(517))
        if (
            payload.get("all_round_sequence_indices") != expected_all_indices
            or payload.get("burn_in_round_sequence_indices") != expected_all_indices[:5]
            or payload.get("retained_round_sequence_indices") != expected_all_indices[5:]
        ):
            raise ValueError(f"Endpoint round sequence mismatch in {path}")
        all_energies = payload.get("all_round_energy_total")
        burn_energies = payload.get("burn_in_round_energy_total")
        retained_energies = payload.get("round_energy_total")
        if (
            not isinstance(all_energies, list)
            or not isinstance(burn_energies, list)
            or not isinstance(retained_energies, list)
            or len(all_energies) != 517
            or burn_energies != all_energies[:5]
            or retained_energies != all_energies[5:]
            or any(not math.isfinite(float(value)) for value in all_energies)
        ):
            raise ValueError(f"Endpoint per-round energies are incomplete in {path}")
        if (
            payload.get("all_round_sample_counts") != [8192] * 517
            or payload.get("burn_in_round_sample_counts") != [8192] * 5
            or payload.get("round_sample_counts") != [8192] * 512
        ):
            raise ValueError(f"Endpoint per-round sample counts are incomplete in {path}")
        environment = payload.get("environment")
        if not isinstance(environment, dict) or any(
            key not in environment
            for key in ("python", "jax", "jaxlib", "netket", "jax_backend", "device_count", "device_kind")
        ):
            raise ValueError(f"Endpoint environment metadata is incomplete in {path}")
        for field in ("energy_per_site", "sigma_per_site", "sigma_round_level_per_site", "sigma_block8_level_per_site"):
            value = float(payload[field])
            if not math.isfinite(value) or (field.startswith("sigma") and value < 0.0):
                raise ValueError(f"Endpoint field {field} is invalid in {path}")
        key = (arm, seed)
        if key in overlays:
            raise ValueError(f"Duplicate endpoint result for {key}: {path}")
        overlays[key] = payload
    return overlays


def endpoint_replica_consistency(
    overlays: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, dict[str, float | str]]:
    """Audit completed endpoint pairs before they can enter camera-ready artifacts."""

    result: dict[str, dict[str, float | str]] = {}
    for arm in ("sr", "spring", "ms_sr"):
        rows = [payload for (row_arm, _), payload in overlays.items() if row_arm == arm]
        if len(rows) < 2:
            continue
        if len(rows) != 2:
            raise ValueError(f"Expected two endpoint replicas for {arm}, found {len(rows)}")
        difference = abs(float(rows[0]["energy_per_site"]) - float(rows[1]["energy_per_site"]))
        denominator = math.hypot(
            float(rows[0]["sigma_per_site"]),
            float(rows[1]["sigma_per_site"]),
        )
        if denominator == 0.0:
            difference_z = 0.0 if difference == 0.0 else math.inf
        else:
            difference_z = difference / denominator
        if not math.isfinite(difference_z) or difference_z > MAX_ENDPOINT_REPLICA_DIFFERENCE_Z:
            raise ValueError(
                f"Endpoint replicas for {arm} fail the predeclared consistency gate: "
                f"z={difference_z:.6g} > {MAX_ENDPOINT_REPLICA_DIFFERENCE_Z:g}"
            )
        result[arm] = {
            "replica_difference_z": difference_z,
            "max_allowed_replica_difference_z": MAX_ENDPOINT_REPLICA_DIFFERENCE_Z,
            "status": "pass",
        }
    return result


def curate_online(
    source: Path,
    output: Path,
    revision: str,
    dirty: bool,
    *,
    online_results_root: Path | None,
    endpoint_results_root: Path | None,
) -> list[dict[str, Any]]:
    destination = output / "online_training"
    seed0_dir = destination / "branch_seed0"
    seed1_dir = destination / "branch_seed1"
    seed0_id = "online.j1j2.no_symmetry.branch2500.seed0.v1"
    seed1_id = "online.j1j2.no_symmetry.branch1800.seed1.v1"

    source_ckpt = source / (
        "large_scale/20260725/results/j1j2ns_sr2450to3000_seed0/"
        "j1j2_results/1e-04/results/checkpoints/checkpoint_step000050.pkl"
    )
    prebranch_ckpt = source / (
        "large_scale/20260725/results/j1j2ns_br1950_sr_seed0/"
        "j1j2_results/1e-04/results/checkpoints/checkpoint_step000500.pkl"
    )

    br1950 = source / (
        "large_scale/20260725/results/j1j2ns_br1950_sr_seed0/"
        "j1j2_results/1e-04/results/train_metrics.csv"
    )
    sr550 = source / (
        "large_scale/20260725/results/j1j2ns_sr2450to3000_seed0/"
        "j1j2_results/1e-04/results/train_metrics.csv"
    )
    spring500 = source / (
        "large_scale/20260725/results/j1j2ns_br2500_spring_seed0/"
        "j1j2_results/1e-04/results/train_metrics.csv"
    )
    common_online_expected = {
        "system": "j1j2",
        "lattice_size": 8,
        "n_sites": 64,
        "learning_rate": 0.01,
        "lr_schedule": "constant",
        "warmup_steps": 0,
        "n_samples": 8192,
        "phase_samples": [8192],
        "chunk_size_bwd": 16,
        "linear_solver": "cholesky",
        "diag_shift": 1e-4,
        "sr_variant": "single",
        "vit_depth": 8,
        "vit_d_model": 72,
        "vit_heads": 12,
        "vit_expansion": 4,
        "output_head": "Vanilla",
    }
    validate_registered_online_config(
        sr550.parent,
        expected={
            **common_online_expected,
            "seed": 0,
            "ngd_method": "sr",
            "chunk_size": 2048,
            "init_from_step": 500,
            "total_steps": 550,
            "schedule_total_steps": 550,
            "phase_lengths": [550],
        },
        expected_init_checkpoint=prebranch_ckpt,
    )
    validate_registered_online_config(
        spring500.parent,
        expected={
            **common_online_expected,
            "seed": 0,
            "ngd_method": "spring",
            "spring_momentum": 0.9,
            "chunk_size": 2048,
            "init_from_step": 50,
            "total_steps": 500,
            "schedule_total_steps": 500,
            "phase_lengths": [500],
        },
        expected_init_checkpoint=source_ckpt,
    )
    if sha256(source_ckpt) != "8e892b5aa13aeb391cd269d6f1201a6ba7f8832027e3aaca508253ec14e3aca5":
        raise ValueError("Seed-0 branch checkpoint hash does not match the registered source")
    first = load_csv(br1950)
    continuation = load_csv(sr550)
    spring = load_csv(spring500)
    if len(first) < 500 or len(continuation) < 550 or len(spring) != 500:
        raise ValueError("Seed-0 online inputs do not have the expected completed lengths")

    seed0_rows: list[dict[str, Any]] = []
    for raw in first[:500]:
        local = int(raw["step"])
        seed0_rows.append(
            online_row(
                raw,
                experiment_id=seed0_id,
                status="complete",
                branch_step=2500,
                branch_seed=0,
                arm="shared_sr",
                optimizer="sr",
                segment="pre_branch",
                legacy_iteration=1950 + local,
                optimizer_step="",
                batches=1,
                budget="",
            )
        )
    for raw in continuation[:50]:
        local = int(raw["step"])
        seed0_rows.append(
            online_row(
                raw,
                experiment_id=seed0_id,
                status="complete",
                branch_step=2500,
                branch_seed=0,
                arm="shared_sr",
                optimizer="sr",
                segment="pre_branch",
                legacy_iteration=2450 + local,
                optimizer_step="",
                batches=1,
                budget="",
            )
        )
    for raw in continuation[50:550]:
        local = int(raw["step"]) - 50
        seed0_rows.append(
            online_row(
                raw,
                experiment_id=seed0_id,
                status="complete",
                branch_step=2500,
                branch_seed=0,
                arm="sr",
                optimizer="sr",
                segment="post_branch",
                legacy_iteration=2500 + local,
                optimizer_step=local,
                batches=1,
                budget=local,
            )
        )
    for raw in spring:
        local = int(raw["step"])
        seed0_rows.append(
            online_row(
                raw,
                experiment_id=seed0_id,
                status="complete",
                branch_step=2500,
                branch_seed=0,
                arm="spring",
                optimizer="spring",
                segment="post_branch",
                legacy_iteration=2500 + local,
                optimizer_step=local,
                batches=1,
                budget=local,
            )
        )
    seed0_mssr_audit: dict[str, Any] = {
        "status": "pending_gpu_execution",
        "required_output": "branch_seed0/mssr_exact",
        "required_update_protocol": "fixed_grid_exact_loo",
    }
    seed0_curve_rows = {"shared_sr": 550, "sr": 500, "spring": 500, "ms_sr": 0}
    seed0_exact_checkpoint: Path | None = None
    if online_results_root is not None:
        exact_result = validated_exact_mssr_run(
            exact_mssr_result_dir(online_results_root, "branch_seed0"),
            expected_seed=0,
            expected_source_native_step=50,
            expected_source_sha256=sha256(source_ckpt),
        )
        if exact_result is not None:
            exact_rows, seed0_exact_checkpoint, seed0_mssr_audit = exact_result
            for raw in exact_rows:
                local = int(raw["step"])
                seed0_rows.append(
                    online_row(
                        raw,
                        experiment_id=seed0_id,
                        status="complete",
                        branch_step=2500,
                        branch_seed=0,
                        arm="ms_sr",
                        optimizer="ms_sr_fixed_grid_exact_loo",
                        segment="post_branch",
                        legacy_iteration=2500 + local,
                        optimizer_step=local,
                        batches=4,
                        budget=4 * local,
                    )
                )
            seed0_curve_rows["ms_sr"] = ONLINE_MSSR_EXPECTED_STEPS
    write_csv(seed0_dir / "training_curve.csv", ONLINE_FIELDS, seed0_rows)
    endpoint_ckpts = {
        "common_branch_step2500": source_ckpt,
        "sr_step500_after_branch": source
        / (
            "large_scale/20260725/results/j1j2ns_sr2450to3000_seed0/"
            "j1j2_results/1e-04/results/checkpoints/checkpoint_step000550.pkl"
        ),
        "spring_step500_after_branch": source
        / (
            "large_scale/20260725/results/j1j2ns_br2500_spring_seed0/"
            "j1j2_results/1e-04/results/checkpoints/checkpoint_step000500.pkl"
        ),
    }
    if seed0_exact_checkpoint is not None:
        endpoint_ckpts["ms_sr_step125_after_branch"] = seed0_exact_checkpoint
    checkpoint_hashes = {
        identifier: sha256(path) for identifier, path in endpoint_ckpts.items() if path.exists()
    }
    seed0_arm_hashes = {
        "sr": checkpoint_hashes["sr_step500_after_branch"],
        "spring": checkpoint_hashes["spring_step500_after_branch"],
    }
    if "ms_sr_step125_after_branch" in checkpoint_hashes:
        seed0_arm_hashes["ms_sr"] = checkpoint_hashes["ms_sr_step125_after_branch"]
    endpoint_steps = {"sr": 550, "spring": 500, "ms_sr": 125}
    overlays = validated_endpoint_overlays(
        endpoint_results_root,
        branch_seed=0,
        experiment_id=seed0_id,
        checkpoint_hashes=seed0_arm_hashes,
        checkpoint_steps=endpoint_steps,
    )
    seed0_replica_consistency = endpoint_replica_consistency(overlays)
    endpoint_rows = []
    for arm in ("sr", "spring", "ms_sr"):
        for eval_run, seed in ((1, 12345), (2, 98765)):
            if (arm, seed) in overlays:
                payload = overlays[(arm, seed)]
                endpoint_rows.append(
                    endpoint_row_from_payload(
                        payload,
                        experiment_id=seed0_id,
                        branch_seed=0,
                        arm=arm,
                        checkpoint_step=int(payload["checkpoint_step"]),
                        eval_run=eval_run,
                    )
                )
            else:
                endpoint_rows.append(
                    pending_endpoint_row(
                        experiment_id=seed0_id,
                        branch_seed=0,
                        arm=arm,
                        checkpoint_step=endpoint_steps[arm],
                        eval_run=eval_run,
                        eval_seed=seed,
                        reason=(
                            "pending_clean_evaluation"
                            if arm in seed0_arm_hashes
                            else "pending_exact_branch_completion"
                        ),
                    )
                )
    write_csv(seed0_dir / "endpoint_evaluations.csv", ENDPOINT_FIELDS, endpoint_rows)

    spring_diagnostic_payload = json.loads(
        (source / "large_scale/20260725/plots/hpeval_br2500_spring500.json").read_text(encoding="utf-8")
    )
    write_csv(
        seed0_dir / "endpoint_diagnostics.csv",
        ENDPOINT_FIELDS,
        [
            endpoint_row_from_payload(
                spring_diagnostic_payload,
                experiment_id=seed0_id,
                branch_seed=0,
                arm="spring",
                checkpoint_step=500,
                eval_run=0,
            )
        ],
    )
    seed0_completed_endpoints = {
        arm: sum(row["arm"] == arm and row["audit_status"] == "audited_complete" for row in endpoint_rows)
        for arm in ("sr", "spring", "ms_sr")
    }
    seed0_target_curve_rows = {"shared_sr": 550, "sr": 500, "spring": 500, "ms_sr": 125}
    seed0_curves_complete = seed0_curve_rows == seed0_target_curve_rows
    seed0_gate = seed0_curves_complete and all(
        value == 2 for value in seed0_completed_endpoints.values()
    )
    seed0_status = (
        "complete"
        if seed0_gate
        else "pending_endpoint_evaluation"
        if seed0_curves_complete
        else "pending_gpu_execution"
    )
    write_json(
        seed0_dir / "status.json",
        {
            "schema_version": 1,
            "experiment_id": seed0_id,
            "status": seed0_status,
            "camera_ready_gate": seed0_gate,
            "curve_rows": seed0_curve_rows,
            "target_curve_rows": seed0_target_curve_rows,
            "equal_sampling_budget_units": 500,
            "ms_sr_protocol_audit": seed0_mssr_audit,
            "completed_endpoint_runs": seed0_completed_endpoints,
            "supplementary_endpoint_diagnostics": {"spring_128_round": 1},
            "required_endpoint_replicas": {"seeds": [12345, 98765], "rounds_each": 512, "runs_per_arm": 2},
            "endpoint_error_column": "sigma_per_site",
            "endpoint_error_definition": "max(round-level SE, eight-round block SE)",
            "endpoint_replica_consistency": seed0_replica_consistency,
            "spring_high_precision_evaluations": "required_for_camera_ready_gate" if not seed0_gate else "complete",
            "excluded_legacy_spring_aggregate": True,
        },
    )
    write_json(
        seed0_dir / "provenance.json",
        {
            "schema_version": 1,
            "experiment_id": seed0_id,
            "trusted_source_revision": revision,
            "trusted_source_worktree_was_dirty": dirty,
            "checkpoint_sha256": checkpoint_hashes,
            "transformations": [
                "Joined the SR source trajectory at global steps 1951-2450 with the first 50 continuation steps.",
                "Mapped SR/SPRING optimizer steps one-to-one to sampling-budget units.",
                "Mapped each K=4 MS-SR optimizer step to four sampling-budget units.",
                "Removed machine-local paths; retained full-precision scalar metrics.",
                "Used only independently completed endpoint JSONs.",
            ],
            "ms_sr_protocol_audit": seed0_mssr_audit,
            "excluded_inputs": [
                {
                    "item": "legacy seed-0 125-update MS-SR callback run",
                    "reason": "the callback added a base SR batch/solve and a separate weight-fit batch and did not implement the registered exact-LOO four-unit update",
                }
            ],
            "interpretation": {
                "training_seed_field": "branch RNG seed; the shared source checkpoint is common to all arms",
                "ms_sr_shift_rule": "fixed grid [1e-2,1e-3,1e-4,1e-5], not NTK quantiles",
            },
        },
    )

    seed1_sr = source / (
        "large_scale/20260725/results/j1j2ns_br1800_sr_seed1/"
        "j1j2_results/1e-04/results/train_metrics.csv"
    )
    seed1_source_ckpt = source / (
        "large_scale/20260428/no_symm_fixed_data_sources/j1j2_results/1e-04/"
        "results/checkpoints/checkpoint_step001800.pkl"
    )
    seed1_sr_ckpt = seed1_sr.parent / "checkpoints/checkpoint_step000500.pkl"
    validate_registered_online_config(
        seed1_sr.parent,
        expected={
            **common_online_expected,
            "seed": 1,
            "ngd_method": "sr",
            "chunk_size": 4096,
            "init_from_step": 1800,
            "total_steps": 2000,
            "schedule_total_steps": 2000,
            "phase_lengths": [2000],
        },
        expected_init_checkpoint=seed1_source_ckpt,
    )
    if sha256(seed1_source_ckpt) != "b68eade1b8d3c9a6d10f57ccf130fdf52119a877f623a95d8062992c2462d319":
        raise ValueError("Seed-1 branch checkpoint hash does not match the registered source")
    seed1_origin, seed1_origin_provenance = branch_origin_row_from_checkpoint(
        seed1_source_ckpt,
        experiment_id=seed1_id,
        branch_step=1800,
        branch_seed=1,
        n_sites=64,
    )
    seed1_rows = [seed1_origin]
    seed1_sr_raw = load_csv(seed1_sr)
    if len(seed1_sr_raw) < 500 or sha256(seed1_sr_ckpt) != "b8308492bd1fed813963b1c65b8f3bac213bd41022e33e76a09e678fdc82c120":
        raise ValueError("Seed-1 SR 500-step anchor failed length or checkpoint-hash validation")
    for raw in seed1_sr_raw[:500]:
        local = int(raw["step"])
        seed1_rows.append(
            online_row(
                raw,
                experiment_id=seed1_id,
                status="complete",
                branch_step=1800,
                branch_seed=1,
                arm="sr",
                optimizer="sr",
                segment="post_branch",
                legacy_iteration=1800 + local,
                optimizer_step=local,
                batches=1,
                budget=local,
            )
        )

    seed1_arm_hashes = {"sr": sha256(seed1_sr_ckpt)}
    seed1_curve_rows = {"shared_sr": 1, "sr": 500, "spring": 0, "ms_sr": 0}
    seed1_mssr_audit: dict[str, Any] = {
        "status": "pending_gpu_execution",
        "required_output": "branch_seed1/mssr_exact",
        "required_update_protocol": "fixed_grid_exact_loo",
    }
    if online_results_root is not None:
        branch_root = online_branch_root(online_results_root, "branch_seed1")
        spring_result_dir = branch_root / "spring/large_scale/j1j2_results/1e-04/results"
        spring_metrics = spring_result_dir / "train_metrics.csv"
        spring_checkpoint = spring_result_dir / "checkpoints/checkpoint_step000500.pkl"
        spring_summary_path = spring_result_dir / "summary.json"
        spring_config_path = spring_result_dir / "config.json"
        if spring_summary_path.exists():
            spring_summary = json.loads(spring_summary_path.read_text(encoding="utf-8"))
            if spring_summary.get("status") == "completed":
                if not (spring_metrics.is_file() and spring_checkpoint.is_file() and spring_config_path.is_file()):
                    raise ValueError("Completed seed-1 SPRING run lacks metrics, config, or final checkpoint")
                spring_expected = {
                    **common_online_expected,
                    "ngd_method": "spring",
                    "spring_momentum": 0.9,
                    "init_from_step": 1800,
                    "total_steps": 500,
                    "schedule_total_steps": 500,
                    "seed": 1,
                    "chunk_size": 4096,
                    "phase_lengths": [500],
                }
                validate_registered_online_config(
                    spring_result_dir,
                    expected=spring_expected,
                    expected_init_checkpoint=seed1_source_ckpt,
                )
                spring_rows = load_csv(spring_metrics)
                if len(spring_rows) != 500 or [int(row["step"]) for row in spring_rows] != list(range(1, 501)):
                    raise ValueError("Completed seed-1 SPRING trajectory is not exactly steps 1--500")
                for raw in spring_rows:
                    local = int(raw["step"])
                    seed1_rows.append(
                        online_row(
                            raw,
                            experiment_id=seed1_id,
                            status="complete",
                            branch_step=1800,
                            branch_seed=1,
                            arm="spring",
                            optimizer="spring",
                            segment="post_branch",
                            legacy_iteration=1800 + local,
                            optimizer_step=local,
                            batches=1,
                            budget=local,
                        )
                    )
                seed1_curve_rows["spring"] = 500
                seed1_arm_hashes["spring"] = sha256(spring_checkpoint)

        exact_result = validated_exact_mssr_run(
            exact_mssr_result_dir(online_results_root, "branch_seed1"),
            expected_seed=1,
            expected_source_native_step=1800,
            expected_source_sha256=sha256(seed1_source_ckpt),
        )
        if exact_result is not None:
            exact_rows, checkpoint_path, seed1_mssr_audit = exact_result
            for raw in exact_rows:
                local = int(raw["step"])
                seed1_rows.append(
                    online_row(
                        raw,
                        experiment_id=seed1_id,
                        status="complete",
                        branch_step=1800,
                        branch_seed=1,
                        arm="ms_sr",
                        optimizer="ms_sr_fixed_grid_exact_loo",
                        segment="post_branch",
                        legacy_iteration=1800 + local,
                        optimizer_step=local,
                        batches=4,
                        budget=4 * local,
                    )
                )
            seed1_curve_rows["ms_sr"] = ONLINE_MSSR_EXPECTED_STEPS
            seed1_arm_hashes["ms_sr"] = sha256(checkpoint_path)
    seed1_rows.sort(key=lambda row: (row["arm"], int(row["optimizer_step_since_branch"])))
    write_csv(seed1_dir / "training_curve.csv", ONLINE_FIELDS, seed1_rows)

    seed1_endpoint_steps = {"sr": 500, "spring": 500, "ms_sr": 125}
    seed1_overlays = validated_endpoint_overlays(
        endpoint_results_root,
        branch_seed=1,
        experiment_id=seed1_id,
        checkpoint_hashes=seed1_arm_hashes,
        checkpoint_steps=seed1_endpoint_steps,
    )
    seed1_replica_consistency = endpoint_replica_consistency(seed1_overlays)
    seed1_endpoint_rows = []
    for arm in ("sr", "spring", "ms_sr"):
        for eval_run, seed in ((1, 12345), (2, 98765)):
            if (arm, seed) in seed1_overlays:
                payload = seed1_overlays[(arm, seed)]
                seed1_endpoint_rows.append(
                    endpoint_row_from_payload(
                        payload,
                        experiment_id=seed1_id,
                        branch_seed=1,
                        arm=arm,
                        checkpoint_step=int(payload["checkpoint_step"]),
                        eval_run=eval_run,
                    )
                )
            else:
                reason = "pending_endpoint_evaluation" if arm in seed1_arm_hashes else "pending_branch_completion"
                seed1_endpoint_rows.append(
                    pending_endpoint_row(
                        experiment_id=seed1_id,
                        branch_seed=1,
                        arm=arm,
                        checkpoint_step=seed1_endpoint_steps[arm],
                        eval_run=eval_run,
                        eval_seed=seed,
                        reason=reason,
                    )
                )
    write_csv(seed1_dir / "endpoint_evaluations.csv", ENDPOINT_FIELDS, seed1_endpoint_rows)

    seed1_completed_endpoints = {
        arm: sum(
            row["arm"] == arm and row["audit_status"] == "audited_complete"
            for row in seed1_endpoint_rows
        )
        for arm in ("sr", "spring", "ms_sr")
    }
    seed1_target_curve_rows = {"shared_sr": 1, "sr": 500, "spring": 500, "ms_sr": 125}
    seed1_curves_complete = seed1_curve_rows == seed1_target_curve_rows
    seed1_gate = seed1_curves_complete and all(value == 2 for value in seed1_completed_endpoints.values())
    seed1_status = (
        "complete"
        if seed1_gate
        else "pending_endpoint_evaluation"
        if seed1_curves_complete
        else "pending_gpu_execution"
    )
    write_json(
        seed1_dir / "status.json",
        {
            "schema_version": 1,
            "experiment_id": seed1_id,
            "status": seed1_status,
            "camera_ready_gate": seed1_gate,
            "source_checkpoint_step": 1800,
            "branch_rng_seed": 1,
            "curve_rows": seed1_curve_rows,
            "target_curve_rows": seed1_target_curve_rows,
            "equal_sampling_budget_units": 500,
            "ms_sr_protocol_audit": seed1_mssr_audit,
            "completed_endpoint_runs": seed1_completed_endpoints,
            "required_endpoint_replicas": {"seeds": [12345, 98765], "rounds_each": 512, "runs_per_arm": 2},
            "endpoint_error_column": "sigma_per_site",
            "endpoint_error_definition": "max(round-level SE, eight-round block SE)",
            "endpoint_replica_consistency": seed1_replica_consistency,
            "warning": (
                "The step-1800 source checkpoint is shared only by this branch's three "
                "optimizer arms; branch_seed0 uses a different step-2500 source checkpoint."
            ),
        },
    )
    write_json(
        seed1_dir / "provenance.json",
        {
            "schema_version": 1,
            "experiment_id": seed1_id,
            "trusted_source_revision": revision,
            "trusted_source_worktree_was_dirty": dirty,
            "source_checkpoint_sha256": sha256(seed1_source_ckpt),
            "branch_origin": seed1_origin_provenance,
            "arm_checkpoint_sha256": seed1_arm_hashes,
            "ms_sr_protocol_audit": seed1_mssr_audit,
            "published_data_scope": "validated 500-budget contract arms only",
            "excluded": [
                "SR metric rows 501-603 beyond the approved 500-budget horizon",
                "discontinuous 112-row exploratory seed-1 MS-SR callback run",
                "stopped seed-1 extra-batch/non-LOO MS-SR continuation under outputs/gpu/excluded",
                "one-step dense development attempts",
                "unmatched TFIM seed-1 MS-SR run",
            ],
        },
    )
    write_json(
        destination / "exclusions.json",
        {
            "schema_version": 1,
            "excluded_inputs": [
                {
                    "item": "legacy partial small-scale online comparison",
                    "reason": "explicitly excluded from camera-ready evidence",
                },
                {
                    "item": "external-cache-dependent online pilots",
                    "reason": "external-cache dependence and best-observed energy references prevent clean independent audit",
                },
                {
                    "item": "dense and learning-rate development probes",
                    "reason": "development probes or one-step/partial attempts",
                },
                {
                    "item": "legacy combined SPRING endpoint aggregate",
                    "reason": "claimed 8,388,608-sample aggregate cannot be exactly reconstructed from retained round-level data",
                    "replacement": "audited 128-round diagnostic in branch_seed0/endpoint_diagnostics.csv",
                },
                {
                    "item": "seed-1 SR tail after step 500, discontinuous exploratory MS-SR, and stopped extra-batch/non-LOO continuation",
                    "reason": "outside or incompatible with the approved exact-LOO equal-500-budget controlled branch contract",
                },
                {
                    "item": "legacy seed-0 125-update MS-SR callback run",
                    "reason": "used a base batch/solve and a separate weight-fit batch in addition to four candidate batches; replaced only by branch_seed0/mssr_exact",
                },
            ],
        },
    )
    write_json(
        destination / "provenance.json",
        {
            "schema_version": 1,
            "category": "online_training",
            "trusted_source_revision": revision,
            "portable_schema": "sr_filter.online_training.v1",
            "private_paths_retained": False,
        },
    )
    return [
        {
            "id": seed0_id,
            "category": "online_training",
            "system": "j1j2",
            "status": seed0_status,
            "config": "configs/experiments/online_j1j2_branch2500_seed0.toml",
            "primary_files": [
                "artifacts/online_training/branch_seed0/training_curve.csv",
                "artifacts/online_training/branch_seed0/endpoint_evaluations.csv",
                "artifacts/online_training/branch_seed0/status.json",
            ],
        },
        {
            "id": seed1_id,
            "category": "online_training",
            "system": "j1j2",
            "status": seed1_status,
            "config": "configs/experiments/online_j1j2_branch1800_seed1.toml",
            "primary_files": [
                "artifacts/online_training/branch_seed1/training_curve.csv",
                "artifacts/online_training/branch_seed1/endpoint_evaluations.csv",
                "artifacts/online_training/branch_seed1/status.json",
            ],
        },
    ]


FIG2_CHECKPOINT_FIELDS = [
    "experiment_id",
    "regime",
    "system",
    "source_checkpoint_id",
    "source_training_shift",
    "checkpoint_step",
    "solver_shift",
    "repeat_count",
    "n_validation_samples",
    "train_seed_base",
    "validation_seed",
    "r_val_raw",
    "r_val_norm",
    "var_hat_raw",
    "var_hat_norm",
    "target_norm_sq",
    "energy_variance_eval",
]

FIG2_REPEAT_FIELDS = [
    "experiment_id",
    "regime",
    "system",
    "source_training_shift",
    "checkpoint_step",
    "solver_shift",
    "repeat_index",
    "r_val_raw",
    "r_val_norm",
]

METHOD_CHECKPOINT_FIELDS = [
    "experiment_id",
    "regime",
    "system",
    "source_checkpoint_id",
    "source_training_shift",
    "checkpoint_step",
    "k",
    "quantile_grid_id",
    "lambda_quantiles_json",
    "lambda_grid_json",
    "lambda_star",
    "lambda_star_selection",
    "lambda_star_at_diagnostic_boundary",
    "method",
    "source_method_id",
    "selection_label",
    "repeat_count",
    "n_validation_samples",
    "train_seed_base",
    "validation_seed",
    "r_val_raw",
    "r_val_norm",
    "var_hat_raw",
    "var_hat_norm",
    "mean_weights_json",
]

METHOD_REPEAT_FIELDS = [
    "experiment_id",
    "regime",
    "system",
    "source_training_shift",
    "checkpoint_step",
    "k",
    "quantile_grid_id",
    "method",
    "repeat_index",
    "r_val_raw",
    "r_val_norm",
    "weights_json",
]

BAGGED_CHECKPOINT_FIELDS = [
    "experiment_id",
    "regime",
    "system",
    "source_checkpoint_id",
    "source_training_shift",
    "checkpoint_step",
    "k",
    "bagged_shift",
    "lambda_star",
    "shift_label",
    "lambda_star_at_diagnostic_boundary",
    "coverage_status",
    "repeat_count",
    "n_validation_samples",
    "train_seed_base",
    "validation_seed",
    "r_val_raw",
    "r_val_norm",
    "var_hat_raw",
    "var_hat_norm",
]

BAGGED_REPEAT_FIELDS = [
    "experiment_id",
    "regime",
    "system",
    "source_training_shift",
    "checkpoint_step",
    "k",
    "bagged_shift",
    "repeat_index",
    "r_val_raw",
    "r_val_norm",
]


def curate_robustness(source: Path, output: Path, revision: str, dirty: bool) -> list[dict[str, Any]]:
    destination = output / "robustness"
    fig2_checkpoints: list[dict[str, Any]] = []
    fig2_repeats: list[dict[str, Any]] = []
    trusted_patterns = []
    trusted_inputs: list[dict[str, Any]] = []

    for regime, expected_system, training_shift, pattern in FIG2_SOURCES:
        paths = sorted(source.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No trusted inputs for {pattern}")
        trusted_patterns.append(pattern)
        experiment_id = f"robustness.{regime}.shift_sweep.v1"
        for path in paths:
            payload = load_pickle(path)
            system = str(payload["system"])
            if system != expected_system:
                raise ValueError(f"Unexpected system {system} in {path}")
            step = int(payload["step"])
            solver_shift = float(payload["lambda"])
            source_shift = float(payload.get("source_training_lambda", training_shift))
            trusted_inputs.append(
                {
                    "input_id": f"{experiment_id}/step_{step:06d}/shift_{solver_shift:.0e}",
                    "sha256": sha256(path),
                }
            )
            fig2_checkpoints.append(
                {
                    "experiment_id": experiment_id,
                    "regime": regime,
                    "system": system,
                    "source_checkpoint_id": portable_checkpoint_id(system, source_shift, step),
                    "source_training_shift": source_shift,
                    "checkpoint_step": step,
                    "solver_shift": solver_shift,
                    "repeat_count": payload["repeat_count"],
                    "n_validation_samples": payload["n_val"],
                    "train_seed_base": payload["train_seed_base"],
                    "validation_seed": payload["validation_seed"],
                    "r_val_raw": payload["r_val_raw"],
                    "r_val_norm": payload["r_val_norm"],
                    "var_hat_raw": payload["var_hat_raw"],
                    "var_hat_norm": payload["var_hat_norm"],
                    "target_norm_sq": payload["target_norm_sq"],
                    "energy_variance_eval": payload["energy_variance_eval"],
                }
            )
            raw_values = payload["r_val_raw_per_repeat"]
            norm_values = payload["r_val_norm_per_repeat"]
            if len(raw_values) != int(payload["repeat_count"]) or len(norm_values) != len(raw_values):
                raise ValueError(f"Incomplete repeat arrays in {path}")
            for repeat_index, (raw_value, norm_value) in enumerate(zip(raw_values, norm_values, strict=True)):
                fig2_repeats.append(
                    {
                        "experiment_id": experiment_id,
                        "regime": regime,
                        "system": system,
                        "source_training_shift": source_shift,
                        "checkpoint_step": step,
                        "solver_shift": solver_shift,
                        "repeat_index": repeat_index,
                        "r_val_raw": raw_value,
                        "r_val_norm": norm_value,
                    }
                )
    fig2_checkpoints.sort(key=lambda row: (row["experiment_id"], row["checkpoint_step"], row["solver_shift"]))
    fig2_repeats.sort(
        key=lambda row: (row["experiment_id"], row["checkpoint_step"], row["solver_shift"], row["repeat_index"])
    )
    write_csv(destination / "shift_sweep_checkpoints.csv", FIG2_CHECKPOINT_FIELDS, fig2_checkpoints)
    write_csv(destination / "shift_sweep_repeats.csv", FIG2_REPEAT_FIELDS, fig2_repeats)

    method_checkpoints: list[dict[str, Any]] = []
    method_repeats: list[dict[str, Any]] = []
    for regime, expected_system, training_shift, quantile_id, pattern in FIG3_SOURCES:
        paths = sorted(source.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No trusted inputs for {pattern}")
        trusted_patterns.append(pattern)
        experiment_id = f"robustness.{regime}.method_ablation.v1"
        for path in paths:
            payload = load_pickle(path)
            system = str(payload["system"])
            if system != expected_system:
                raise ValueError(f"Unexpected system {system} in {path}")
            step = int(payload["step"])
            source_shift = float(payload.get("source_training_lambda", training_shift))
            lambda_star = float(payload["lambda_star"])
            k = int(payload["k"])
            trusted_inputs.append(
                {
                    "input_id": f"{experiment_id}/step_{step:06d}/k_{k}/{quantile_id}",
                    "sha256": sha256(path),
                }
            )
            for source_method, metrics in sorted(payload["methods"].items()):
                method = METHOD_NAMES.get(source_method, source_method.replace("_k4", ""))
                method_checkpoints.append(
                    {
                        "experiment_id": experiment_id,
                        "regime": regime,
                        "system": system,
                        "source_checkpoint_id": portable_checkpoint_id(system, source_shift, step),
                        "source_training_shift": source_shift,
                        "checkpoint_step": step,
                        "k": k,
                        "quantile_grid_id": quantile_id,
                        "lambda_quantiles_json": compact_json(payload["lambda_quantiles"]),
                        "lambda_grid_json": compact_json(payload["lambda_grid"]),
                        "lambda_star": lambda_star,
                        "lambda_star_selection": "evaluation_grid_best_same_validation",
                        "lambda_star_at_diagnostic_boundary": str(is_grid_boundary(lambda_star)).lower(),
                        "method": method,
                        "source_method_id": source_method,
                        "selection_label": selection_label(method),
                        "repeat_count": metrics["repeat_count"],
                        "n_validation_samples": payload["n_val"],
                        "train_seed_base": payload["train_seed_base"],
                        "validation_seed": payload["validation_seed"],
                        "r_val_raw": metrics["r_val_raw"],
                        "r_val_norm": metrics["r_val_norm"],
                        "var_hat_raw": metrics["var_hat_raw"],
                        "var_hat_norm": metrics["var_hat_norm"],
                        "mean_weights_json": compact_json(metrics.get("mean_weights", [])),
                    }
                )
                raw_values = metrics["r_val_raw_per_repeat"]
                norm_values = metrics["r_val_norm_per_repeat"]
                weights = metrics.get("weights_per_repeat", [[] for _ in raw_values])
                if not (len(raw_values) == len(norm_values) == len(weights) == int(metrics["repeat_count"])):
                    raise ValueError(f"Incomplete method repeat arrays in {path}: {source_method}")
                for repeat_index, (raw_value, norm_value, repeat_weights) in enumerate(
                    zip(raw_values, norm_values, weights, strict=True)
                ):
                    method_repeats.append(
                        {
                            "experiment_id": experiment_id,
                            "regime": regime,
                            "system": system,
                            "source_training_shift": source_shift,
                            "checkpoint_step": step,
                            "k": k,
                            "quantile_grid_id": quantile_id,
                            "method": method,
                            "repeat_index": repeat_index,
                            "r_val_raw": raw_value,
                            "r_val_norm": norm_value,
                            "weights_json": compact_json(repeat_weights),
                        }
                    )
    method_checkpoints.sort(
        key=lambda row: (row["experiment_id"], row["checkpoint_step"], row["k"], row["method"])
    )
    method_repeats.sort(
        key=lambda row: (
            row["experiment_id"],
            row["checkpoint_step"],
            row["k"],
            row["method"],
            row["repeat_index"],
        )
    )
    write_csv(destination / "method_ablation_checkpoints.csv", METHOD_CHECKPOINT_FIELDS, method_checkpoints)
    write_csv(destination / "method_ablation_repeats.csv", METHOD_REPEAT_FIELDS, method_repeats)

    bagged_checkpoints: list[dict[str, Any]] = []
    bagged_repeats: list[dict[str, Any]] = []
    expected_steps: dict[tuple[str, int], set[int]] = defaultdict(set)
    for regime, expected_system, pattern in BAGGED_SOURCES:
        paths = sorted(source.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No trusted inputs for {pattern}")
        trusted_patterns.append(pattern)
        canonical_regime = "tfim_canonical" if expected_system == "tfim" else "j1j2_no_symmetry"
        experiment_id = f"robustness.{regime}.bagged_fixed_shift.v1"
        for path in paths:
            payload = load_pickle(path)
            system = str(payload["system"])
            step = int(payload["step"])
            k = int(payload["k"])
            source_shift = float(payload["source_training_lambda"])
            lambda_star = float(payload["lambda_star"])
            if canonical_regime == "j1j2_no_symmetry" and step == 1000:
                # Only one fixed shift completed at this checkpoint.  Exclude
                # the whole checkpoint so every comparative series uses the
                # same nine-checkpoint support.
                continue
            trusted_inputs.append(
                {
                    "input_id": (
                        f"{experiment_id}/step_{step:06d}/k_{k}/"
                        f"{'extension' if path.stem.endswith('_ext') else 'primary'}"
                    ),
                    "sha256": sha256(path),
                }
            )
            expected_steps[(canonical_regime, k)].add(step)
            for metrics in payload["methods"].values():
                bagged_shift = float(metrics["bagged_lambda"])
                if math.isclose(bagged_shift, lambda_star):
                    shift_label = "evaluation_grid_best_same_validation"
                elif math.isclose(bagged_shift, source_shift):
                    shift_label = "source_training_shift"
                else:
                    shift_label = "fixed_candidate"
                base = {
                    "experiment_id": experiment_id,
                    "regime": canonical_regime,
                    "system": system,
                    "source_checkpoint_id": portable_checkpoint_id(system, source_shift, step),
                    "source_training_shift": source_shift,
                    "checkpoint_step": step,
                    "k": k,
                    "bagged_shift": bagged_shift,
                    "lambda_star": lambda_star,
                    "shift_label": shift_label,
                    "lambda_star_at_diagnostic_boundary": str(is_grid_boundary(lambda_star)).lower(),
                    "coverage_status": "",
                    "repeat_count": metrics["repeat_count"],
                    "n_validation_samples": payload["n_val"],
                    "train_seed_base": payload["train_seed_base"],
                    "validation_seed": payload["validation_seed"],
                    "r_val_raw": metrics["r_val_raw"],
                    "r_val_norm": metrics["r_val_norm"],
                    "var_hat_raw": metrics["var_hat_raw"],
                    "var_hat_norm": metrics["var_hat_norm"],
                }
                bagged_checkpoints.append(base)
                raw_values = metrics["r_val_raw_per_repeat"]
                norm_values = metrics["r_val_norm_per_repeat"]
                for repeat_index, (raw_value, norm_value) in enumerate(zip(raw_values, norm_values, strict=True)):
                    bagged_repeats.append(
                        {
                            "experiment_id": experiment_id,
                            "regime": canonical_regime,
                            "system": system,
                            "source_training_shift": source_shift,
                            "checkpoint_step": step,
                            "k": k,
                            "bagged_shift": bagged_shift,
                            "repeat_index": repeat_index,
                            "r_val_raw": raw_value,
                            "r_val_norm": norm_value,
                        }
                    )
    coverage_counts: dict[tuple[str, int, float], set[int]] = defaultdict(set)
    for row in bagged_checkpoints:
        coverage_counts[(row["regime"], row["k"], row["bagged_shift"])].add(row["checkpoint_step"])
    for row in bagged_checkpoints:
        expected = len(expected_steps[(row["regime"], row["k"])])
        observed = len(coverage_counts[(row["regime"], row["k"], row["bagged_shift"])])
        row["coverage_status"] = "complete" if observed == expected else f"partial_{observed}_of_{expected}_checkpoints"
    bagged_checkpoints.sort(
        key=lambda row: (row["regime"], row["k"], row["bagged_shift"], row["checkpoint_step"])
    )
    bagged_repeats.sort(
        key=lambda row: (row["regime"], row["k"], row["bagged_shift"], row["checkpoint_step"], row["repeat_index"])
    )
    write_csv(destination / "bagged_fixed_shift_checkpoints.csv", BAGGED_CHECKPOINT_FIELDS, bagged_checkpoints)
    write_csv(destination / "bagged_fixed_shift_repeats.csv", BAGGED_REPEAT_FIELDS, bagged_repeats)

    def summarize(
        rows: list[dict[str, Any]],
        group_fields: tuple[str, ...],
        metrics: tuple[str, ...],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[tuple(row[field] for field in group_fields)].append(row)
        output_rows = []
        for key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
            result = dict(zip(group_fields, key, strict=True))
            result["checkpoint_count"] = len(group)
            for metric in metrics:
                values = [float(row[metric]) for row in group]
                result[f"{metric}_mean"] = statistics.fmean(values)
                result[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
            output_rows.append(result)
        fields = list(group_fields) + ["checkpoint_count"]
        for metric in metrics:
            fields.extend((f"{metric}_mean", f"{metric}_std"))
        return fields, output_rows

    shift_fields, shift_summary = summarize(
        fig2_checkpoints,
        ("experiment_id", "regime", "system", "source_training_shift", "solver_shift"),
        ("r_val_raw", "r_val_norm", "var_hat_raw", "var_hat_norm"),
    )
    method_fields, method_summary = summarize(
        method_checkpoints,
        (
            "experiment_id",
            "regime",
            "system",
            "source_training_shift",
            "k",
            "quantile_grid_id",
            "lambda_quantiles_json",
            "method",
            "selection_label",
            "lambda_star_at_diagnostic_boundary",
        ),
        ("r_val_raw", "r_val_norm", "var_hat_raw", "var_hat_norm"),
    )
    bagged_fields, bagged_summary = summarize(
        bagged_checkpoints,
        (
            "regime",
            "system",
            "source_training_shift",
            "k",
            "bagged_shift",
            "shift_label",
            "coverage_status",
            "lambda_star_at_diagnostic_boundary",
        ),
        ("r_val_raw", "r_val_norm", "var_hat_raw", "var_hat_norm"),
    )
    write_csv(destination / "shift_sweep_summary.csv", shift_fields, shift_summary)
    write_csv(destination / "method_ablation_summary.csv", method_fields, method_summary)
    write_csv(destination / "bagged_fixed_shift_summary.csv", bagged_fields, bagged_summary)

    ssm_rows = [row for row in method_checkpoints if row["regime"] == "ssm_no_symmetry"]
    ssm_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ssm_rows:
        ssm_by_method[row["method"]].append(row)
    bagged_by_step = {row["checkpoint_step"]: row for row in ssm_by_method["bagged_grid_best_shift"]}
    mssr_by_step = {row["checkpoint_step"]: row for row in ssm_by_method["ms_sr"]}
    ssm_summary_rows = []
    for method in ("bagged_grid_best_shift", "ms_sr"):
        rows = sorted(ssm_by_method[method], key=lambda row: row["checkpoint_step"])
        risks = [float(row["r_val_raw"]) for row in rows]
        variances = [float(row["var_hat_raw"]) for row in rows]
        risk_std = statistics.pstdev(risks)
        variance_std = statistics.pstdev(variances)
        ssm_summary_rows.append(
            {
                "experiment_id": "robustness.ssm_no_symmetry.boundary_oracle.v1",
                "system": "ssm",
                "regime": "no_symmetry",
                "method": method,
                "result_label": (
                    "boundary-grid evaluation-oracle bagged SR"
                    if method == "bagged_grid_best_shift"
                    else "MS-SR"
                ),
                "lambda_star": 0.1,
                "lambda_star_selection": "evaluation_grid_best_same_validation",
                "lambda_star_at_diagnostic_boundary": "true",
                "checkpoint_count": len(rows),
                "r_val_raw_mean": statistics.fmean(risks),
                "r_val_raw_population_std": risk_std,
                "r_val_raw_checkpoint_se": risk_std / math.sqrt(len(rows)),
                "var_hat_raw_mean": statistics.fmean(variances),
                "var_hat_raw_population_std": variance_std,
                "var_hat_raw_checkpoint_se": variance_std / math.sqrt(len(rows)),
                "lower_r_val_checkpoint_count": (
                    sum(float(bagged_by_step[step]["r_val_raw"]) < float(mssr_by_step[step]["r_val_raw"]) for step in bagged_by_step)
                    if method == "bagged_grid_best_shift"
                    else sum(float(mssr_by_step[step]["r_val_raw"]) < float(bagged_by_step[step]["r_val_raw"]) for step in bagged_by_step)
                ),
                "lower_var_hat_checkpoint_count": (
                    sum(float(bagged_by_step[step]["var_hat_raw"]) < float(mssr_by_step[step]["var_hat_raw"]) for step in bagged_by_step)
                    if method == "bagged_grid_best_shift"
                    else sum(float(mssr_by_step[step]["var_hat_raw"]) < float(bagged_by_step[step]["var_hat_raw"]) for step in bagged_by_step)
                ),
            }
        )
    write_csv(
        destination / "ssm_boundary_oracle_summary.csv",
        [
            "experiment_id",
            "system",
            "regime",
            "method",
            "result_label",
            "lambda_star",
            "lambda_star_selection",
            "lambda_star_at_diagnostic_boundary",
            "checkpoint_count",
            "r_val_raw_mean",
            "r_val_raw_population_std",
            "r_val_raw_checkpoint_se",
            "var_hat_raw_mean",
            "var_hat_raw_population_std",
            "var_hat_raw_checkpoint_se",
            "lower_r_val_checkpoint_count",
            "lower_var_hat_checkpoint_count",
        ],
        ssm_summary_rows,
    )
    trusted_inputs.sort(key=lambda row: row["input_id"])
    write_csv(destination / "source_input_checksums.csv", ["input_id", "sha256"], trusted_inputs)

    write_json(
        destination / "provenance.json",
        {
            "schema_version": 1,
            "category": "robustness",
            "trusted_source_revision": revision,
            "trusted_source_worktree_was_dirty": dirty,
            "trusted_input_count": len(trusted_inputs),
            "trusted_input_checksums": "source_input_checksums.csv",
            "portable_schema": "sr_filter.robustness.v1",
            "statistics": "Summary standard deviations are population standard deviations across checkpoint-level aggregates.",
            "oracle_label_policy": "Grid-best shifts were selected on the same validation stream used for reporting and are never labeled independently held out.",
            "private_paths_retained": False,
        },
    )
    write_json(
        destination / "exclusions.json",
        {
            "schema_version": 1,
            "excluded_inputs": [
                {
                    "pattern": "raw pickle payloads and NTK caches",
                    "reason": "unsafe/non-portable serialization and embedded absolute paths; all scalar claim data are exported as CSV",
                },
                {
                    "pattern": "j1j2_no_symmetry fixed-shift matrix at step 1000",
                    "reason": "only shift 1e-4 completed; the entire checkpoint is excluded so every fixed-shift comparison uses the same nine checkpoints",
                },
            ],
            "negative_result_disclosure": {
                "experiment_id": "robustness.ssm_no_symmetry.method_ablation.v1",
                "label": "boundary-grid evaluation-oracle bagged SR",
                "caveat": "lambda*=0.1 at the maximum diagnostic-grid boundary and selected on the reporting validation stream",
            },
        },
    )
    return [
        {
            "id": "robustness.checkpoint_local.v1",
            "category": "robustness",
            "system": "tfim,j1j2,ssm",
            "status": "complete",
            "config": "configs/experiments/robustness_checkpoint_local.toml",
            "primary_files": [
                "artifacts/robustness/shift_sweep_checkpoints.csv",
                "artifacts/robustness/method_ablation_checkpoints.csv",
                "artifacts/robustness/bagged_fixed_shift_checkpoints.csv",
                "artifacts/robustness/method_ablation_summary.csv",
                "artifacts/robustness/ssm_boundary_oracle_summary.csv",
            ],
        }
    ]


TIMING_FIELDS = [
    "experiment_id",
    "pair",
    "physical_gpu_index",
    "seed",
    "method",
    "k",
    "repeat",
    "prepare_batches_s",
    "candidate_solves_s",
    "cross_prediction_s",
    "simplex_fit_s",
    "mix_s",
    "apply_update_s",
    "total_s",
    "apply_update",
]


def sanitize_timing_payload(payload: dict[str, Any]) -> dict[str, Any]:
    clean = dict(payload)
    clean["checkpoint"] = "tfim/train_shift_1e-4/step_002000"
    clean["environment"] = dict(clean["environment"])
    clean["environment"]["device"] = "visible_device_0"
    return clean


def curate_wall_clock(source: Path, output: Path, revision: str, dirty: bool) -> list[dict[str, Any]]:
    destination = output / "wall_clock"
    raw_destination = destination / "raw"
    source_dir = source / "rebuttal_assets/sr_cost/data"
    experiment_id = "wall_clock.tfim.h100.sr_bagged_ms_sr.v1"
    physical_gpus = {1: 2, 2: 3, 3: 4}
    method_names = {"sr": "sr", "bagged_sr_k4": "bagged_sr", "ms_sr_k4": "ms_sr"}
    timing_rows: list[dict[str, Any]] = []
    fingerprint_rows: list[dict[str, Any]] = []
    pairing_rows: list[dict[str, Any]] = []

    for pair in (1, 2, 3):
        pair_payloads = {}
        for file_method, public_method in method_names.items():
            path = source_dir / f"pair{pair}_{file_method}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            clean = sanitize_timing_payload(payload)
            write_json(raw_destination / path.name, clean)
            pair_payloads[public_method] = clean
            for row in clean["rows"]:
                timing_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "pair": pair,
                        "physical_gpu_index": physical_gpus[pair],
                        "seed": clean["seed"],
                        "method": public_method,
                        "k": 1 if public_method == "sr" else 4,
                        "repeat": row["repeat"],
                        "prepare_batches_s": row["prepare_batches_s"],
                        "candidate_solves_s": row["candidate_solves_s"],
                        "cross_prediction_s": row["cross_prediction_s"],
                        "simplex_fit_s": row["simplex_fit_s"],
                        "mix_s": row["mix_s"],
                        "apply_update_s": row["apply_update_s"],
                        "total_s": row["total_s"],
                        "apply_update": str(clean["apply_update"]).lower(),
                    }
                )
            for repeat, metadata in enumerate(clean["metadata_rows"], start=1):
                for batch_slot, fingerprint in enumerate(metadata["sample_fingerprints"]):
                    fingerprint_rows.append(
                        {
                            "experiment_id": experiment_id,
                            "pair": pair,
                            "seed": clean["seed"],
                            "repeat": repeat,
                            "method": public_method,
                            "batch_slot": batch_slot,
                            "sample_fingerprint": fingerprint,
                        }
                    )
        for repeat in range(5):
            sr_fp = pair_payloads["sr"]["metadata_rows"][repeat]["sample_fingerprints"]
            bag_fp = pair_payloads["bagged_sr"]["metadata_rows"][repeat]["sample_fingerprints"]
            ms_fp = pair_payloads["ms_sr"]["metadata_rows"][repeat]["sample_fingerprints"]
            pairing_rows.append(
                {
                    "experiment_id": experiment_id,
                    "pair": pair,
                    "repeat": repeat + 1,
                    "first_batch_matches_sr": str(sr_fp[0] == bag_fp[0] == ms_fp[0]).lower(),
                    "all_k4_batches_match": str(bag_fp == ms_fp).lower(),
                }
            )
    timing_rows.sort(key=lambda row: (row["pair"], row["method"], row["repeat"]))
    fingerprint_rows.sort(key=lambda row: (row["pair"], row["repeat"], row["method"], row["batch_slot"]))
    write_csv(destination / "timings_long.csv", TIMING_FIELDS, timing_rows)
    write_csv(
        destination / "sample_fingerprints.csv",
        ["experiment_id", "pair", "seed", "repeat", "method", "batch_slot", "sample_fingerprint"],
        fingerprint_rows,
    )
    write_csv(
        destination / "pairing_verification.csv",
        ["experiment_id", "pair", "repeat", "first_batch_matches_sr", "all_k4_batches_match"],
        pairing_rows,
    )

    for source_name, target_name in (
        ("timing_summary.csv", "timing_summary.csv"),
        ("paired_cluster_summary.csv", "paired_cluster_summary.csv"),
    ):
        rows = load_csv(source_dir / source_name)
        fields = list(rows[0])
        write_csv(destination / target_name, fields, rows)

    write_json(
        destination / "provenance.json",
        {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "trusted_source_revision": revision,
            "trusted_source_worktree_was_dirty": dirty,
            "hardware": {
                "gpu": "NVIDIA H100 80GB HBM3",
                "physical_gpu_indices": physical_gpus,
                "host_cpu_reported": "dual Intel Xeon Platinum 8481C",
            },
            "software": {"python": "3.12.13", "jax": "0.7.2", "netket": "3.21.0", "scipy": "1.18.0"},
            "protocol": {
                "warmups": 2,
                "timed_repeats": 5,
                "synchronized": True,
                "apply_update": False,
                "diagnostics": False,
                "solver": "Hermitian eigendecomposition pseudoinverse, rtol=1e-12",
                "samples_per_batch": 12000,
            },
            "interpretation": "Steady-state update-construction timing, not end-to-end training throughput.",
            "private_paths_retained": False,
        },
    )
    write_json(
        destination / "exclusions.json",
        {
            "schema_version": 1,
            "excluded": [
                {
                    "item": "process peak-memory comparison",
                    "reason": "reported allocator high-water marks include process/JIT lifetime and are not isolated step peaks",
                },
                {
                    "item": "full training throughput claim",
                    "reason": "benchmark intentionally used apply_update=false and fixed parameters",
                },
            ],
        },
    )
    return [
        {
            "id": experiment_id,
            "category": "wall_clock",
            "system": "tfim",
            "status": "complete",
            "config": "configs/experiments/wall_clock_h100.toml",
            "primary_files": [
                "artifacts/wall_clock/timings_long.csv",
                "artifacts/wall_clock/timing_summary.csv",
                "artifacts/wall_clock/paired_cluster_summary.csv",
                "artifacts/wall_clock/pairing_verification.csv",
            ],
        }
    ]


def write_category_checksums(category: Path) -> None:
    files = sorted(path for path in category.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    lines = [f"{sha256(path)}  {path.relative_to(category).as_posix()}" for path in files]
    (category / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def media_type(path: Path) -> str:
    return {
        ".csv": "text/csv",
        ".json": "application/json",
        ".toml": "application/toml",
        ".sha256": "text/plain",
    }.get(path.suffix, "application/octet-stream")


def write_manifest(output: Path, experiments: list[dict[str, Any]], revision: str, dirty: bool) -> None:
    files = []
    curated_paths = [output / "exclusions.json"]
    if (output / "external_checkpoints.json").is_file():
        curated_paths.append(output / "external_checkpoints.json")
    companion_release_path = output / "hf_dataset_release.json"
    companion_release = None
    if companion_release_path.is_file():
        companion_release = json.loads(companion_release_path.read_text(encoding="utf-8"))
        if companion_release.get("schema") != "sr_filter.hf_dataset_release.v1":
            raise ValueError("Unexpected companion dataset release schema")
        curated_paths.append(companion_release_path)
    for category in ("online_training", "robustness", "wall_clock"):
        curated_paths.extend(path for path in (output / category).rglob("*") if path.is_file())
    curated_paths.extend(path for path in (output / "small_scale").rglob("*.pkl") if path.is_file())
    for path in sorted(curated_paths):
        record = {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "media_type": media_type(path),
            "url": None,
        }
        if path.suffix == ".pkl":
            record.update(
                {
                    "distribution": "bundled_legacy_pickle",
                    "externalization_status": "pending_pinned_remote_upload",
                    "loading_warning": "Python pickle is unsafe for untrusted files; load only after SHA-256 verification and only from a trusted source.",
                }
            )
        files.append(record)
    configs = []
    config_paths = sorted((REPO_ROOT / "configs" / "experiments").glob("*.toml"))
    config_paths.extend(sorted((REPO_ROOT / "configs" / "release").glob("*.toml")))
    for path in config_paths:
        configs.append(
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    payload = {
        "schema": "sr_filter.artifact_manifest.v1",
        "schema_version": 1,
        "base_url": None,
        "generated_by": "scripts/curation/curate_release_artifacts.py",
        "trusted_source_revision": revision,
        "trusted_source_worktree_was_dirty": dirty,
        "experiments": list(experiments),
        "external_checkpoint_manifest": (
            "artifacts/external_checkpoints.json"
            if (output / "external_checkpoints.json").is_file()
            else None
        ),
        "files": files,
        "configs": configs,
    }
    if companion_release is not None:
        payload["companion_dataset"] = {
            key: companion_release.get(key)
            for key in (
                "schema",
                "status",
                "repo_id",
                "repo_type",
                "code_revision",
                "data_revision",
                "manifest_path",
                "manifest_url",
                "raw_base_url",
                "verification",
            )
        }
        if companion_release.get("status") == "verified_pinned":
            remote_files = companion_release.get("files")
            if not isinstance(remote_files, list) or not remote_files:
                raise ValueError("Pinned companion dataset release has no file records")
            payload["experiments"].append(
                {
                    "id": "companion.camera_ready_artifacts.v1",
                    "category": "companion_dataset",
                    "status": "complete",
                    "files": remote_files,
                }
            )
    write_json(output / "manifest.json", payload)


def write_root_exclusions(output: Path) -> None:
    companion_path = output / "hf_dataset_release.json"
    companion = (
        json.loads(companion_path.read_text(encoding="utf-8"))
        if companion_path.is_file()
        else {"status": "pending_local_bundle"}
    )
    if companion.get("status") == "verified_pinned":
        external_artifacts = {
            "status": "verified_pinned",
            "repo_id": companion.get("repo_id"),
            "revision": companion.get("data_revision"),
            "manifest_url": companion.get("manifest_url"),
        }
    else:
        external_artifacts = {
            "status": str(companion.get("status", "pending_local_bundle")),
            "planned_contents": [
                "sanitized source and final checkpoints",
                "endpoint per-round records",
                "extended timing and online provenance",
            ],
            "requirements": "Publish privately, read back, verify byte sizes and SHA-256 digests, then pin an immutable revision before making the dataset public.",
        }
    write_json(
        output / "exclusions.json",
        {
            "schema_version": 1,
            "policy": "Only completed, auditable scalar/tabular claim data are published; pending rows contain no fabricated measurements.",
            "excluded": [
                {
                    "item": "legacy review-period packaging",
                    "reason": "superseded by stable artifact identifiers and slated for removal from the public tree",
                },
                {
                    "item": "legacy small-scale online comparison",
                    "reason": "old partial Figure 5 experiments are outside the camera-ready evidence set",
                },
                {
                    "item": "release-curation input pickles, caches, raw logs, and unsanitized training checkpoints",
                    "reason": "large or path-bearing private inputs; logical checkpoint IDs and trusted-input hashes are emitted instead",
                    "exception": "two separately documented small-scale result pickles remain bundled until a pinned remote upload is available",
                },
                {
                    "item": "SPRING 8,388,608-sample combined endpoint claim",
                    "reason": "retained files do not permit exact reconstruction; clean seeds 12345 and 98765 are pending",
                },
                {
                    "item": "online branch seed 1 completion claim",
                    "reason": "the validated SR first 500 steps are reused; its 103-row tail, a discontinuous 112-row MS-SR attempt, and the stopped extra-batch/non-LOO continuation are excluded; only mssr_exact can satisfy the pending MS-SR contract",
                },
                {
                    "item": "legacy seed-0 MS-SR callback trajectory and endpoint checkpoint",
                    "reason": "the implementation consumed a base batch/solve plus four candidate batches/solves and a separate weight-fit batch; it is excluded even though it contains 125 rows",
                },
            ],
            "external_artifacts": external_artifacts,
        },
    )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    online_results_root = None if args.online_results_root is None else args.online_results_root.resolve()
    endpoint_results_root = None if args.endpoint_results_root is None else args.endpoint_results_root.resolve()
    if not (source / "large_scale").is_dir() or not (source / "rebuttal_assets").is_dir():
        raise SystemExit(f"Not a recognized trusted experiment checkout: {source}")
    output.mkdir(parents=True, exist_ok=True)
    revision, dirty = source_revision(source)
    experiments = []
    experiments.extend(
        curate_online(
            source,
            output,
            revision,
            dirty,
            online_results_root=online_results_root,
            endpoint_results_root=endpoint_results_root,
        )
    )
    experiments.extend(curate_robustness(source, output, revision, dirty))
    experiments.extend(curate_wall_clock(source, output, revision, dirty))
    write_root_exclusions(output)
    for category in ("online_training", "robustness", "wall_clock"):
        write_category_checksums(output / category)
    write_manifest(output, experiments, revision, dirty)
    print(f"Curated {len(experiments)} experiment groups into {output}")


if __name__ == "__main__":
    main()
