"""Validate and plot the two-branch equal-budget online comparison.

The validator deliberately refuses to produce a figure until both declared
branches and all audited endpoint replicas satisfy the camera-ready protocol.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sr_filtering_nqs.large_scale.core.endpoint_eval import (
    MAX_REPLICA_DIFFERENCE_Z,
    atomic_write_json,
    replica_difference_z,
)
from sr_filtering_nqs.plot_metadata import pdf_metadata

COMPLETE_VALUES = frozenset({"complete", "completed", "ready"})
AUDIT_VALUES = frozenset({"pass", "passed", "audited", "ok", "complete", "completed"})
BRANCH_ARMS = ("sr", "spring", "ms_sr")
REQUIRED_ARMS = ("shared_sr", *BRANCH_ARMS)
EXPECTED_BRANCHES = {
    "branch_seed0": {
        "seed": 0,
        "source_step": 2500,
        "experiment_id": "online.j1j2.no_symmetry.branch2500.seed0.v1",
        "shared_sr_rows": 550,
        "endpoint_steps": {"sr": 550, "spring": 500, "ms_sr": 125},
    },
    "branch_seed1": {
        "seed": 1,
        "source_step": 1800,
        "experiment_id": "online.j1j2.no_symmetry.branch1800.seed1.v1",
        "shared_sr_rows": 1,
        "endpoint_steps": {"sr": 500, "spring": 500, "ms_sr": 125},
    },
}
BRANCH_BUDGET_UNITS = 500
SAMPLES_PER_BATCH = 8192
ENDPOINT_RUNS_PER_ARM = 2
ENDPOINT_EVAL_SEEDS = (12345, 98765)
ENDPOINT_ROUNDS_PER_RUN = 512
ENDPOINT_BURN_IN_ROUNDS = 5
ENDPOINT_SAMPLES_PER_RUN = ENDPOINT_ROUNDS_PER_RUN * SAMPLES_PER_BATCH
EXACT_MSSR_GRID = (1e-2, 1e-3, 1e-4, 1e-5)
COLORS = {
    "shared_sr": "#777777",
    "sr": "#2570b7",
    "spring": "#238b57",
    "ms_sr": "#c43b46",
}
LABELS = {
    "shared_sr": "shared SR",
    "sr": "SR",
    "spring": "SPRING",
    "ms_sr": "MS-SR (K=4)",
}


class GateError(ValueError):
    """Raised when a required online artifact is missing or unaudited."""


@dataclass(frozen=True)
class PooledEndpoint:
    arm: str
    energy_per_site: float
    sigma_per_site: float
    total_samples: int
    replica_difference_z: float
    inputs: tuple[dict[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "aggregation": "sample_count_weighted_independent",
            "energy_per_site": self.energy_per_site,
            "sigma_per_site": self.sigma_per_site,
            "total_samples": self.total_samples,
            "replica_difference_z": self.replica_difference_z,
            "max_allowed_replica_difference_z": MAX_REPLICA_DIFFERENCE_Z,
            "replica_consistency_status": "pass",
            "input_count": len(self.inputs),
            "inputs": [dict(row) for row in self.inputs],
        }


@dataclass(frozen=True)
class OnlineBranch:
    branch_id: str
    seed: int
    source_step: int
    curves: Mapping[str, tuple[dict[str, str], ...]]
    endpoints: Mapping[str, PooledEndpoint]
    input_files: tuple[Path, ...]


def _normalized(value: Any) -> str:
    return str(value).strip().lower()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise GateError(f"missing required artifact: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise GateError(f"empty required artifact: {path}")
    return rows


def _finite_float(row: Mapping[str, str], field: str, context: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise GateError(f"{context}: invalid {field!r}") from error
    if not math.isfinite(value):
        raise GateError(f"{context}: non-finite {field!r}")
    return value


def _integer(row: Mapping[str, str], field: str, context: str) -> int:
    value = _finite_float(row, field, context)
    rounded = round(value)
    if abs(value - rounded) > 1e-9:
        raise GateError(f"{context}: {field!r} is not an integer")
    return int(rounded)


def _select_standard_error(row: Mapping[str, str], context: str) -> float:
    for field in (
        "sigma_per_site",
        "sigma_block8_level_per_site",
        "sigma_round_level_per_site",
    ):
        if row.get(field, "") in (None, ""):
            continue
        value = _finite_float(row, field, context)
        if value > 0.0:
            return value
    raise GateError(f"{context}: no positive audited per-site standard error")


def _pool_endpoints(
    rows: list[dict[str, str]],
    *,
    arm: str,
    branch_seed: int,
    experiment_id: str,
    expected_checkpoint_step: int,
    context: str,
) -> PooledEndpoint:
    selected = [
        row
        for row in rows
        if _normalized(row.get("arm")) == arm
        and row.get("rounds", "") not in (None, "")
        and _integer(row, "rounds", context) == ENDPOINT_ROUNDS_PER_RUN
    ]
    if len(selected) != ENDPOINT_RUNS_PER_ARM:
        raise GateError(
            f"{context}: expected exactly {ENDPOINT_RUNS_PER_ARM} audited "
            f"{ENDPOINT_ROUNDS_PER_RUN}-round endpoints for {arm}, found {len(selected)}"
        )

    energies: list[float] = []
    sigmas: list[float] = []
    sample_counts: list[int] = []
    eval_seeds: list[int] = []
    eval_runs: list[int] = []
    for index, row in enumerate(selected):
        row_context = f"{context}, {arm}[{index}]"
        if _normalized(row.get("run_status")) not in COMPLETE_VALUES:
            raise GateError(f"{row_context}: run_status is not complete")
        if _normalized(row.get("audit_status")) not in AUDIT_VALUES:
            raise GateError(f"{row_context}: audit_status is not passing")
        if _integer(row, "branch_rng_seed", row_context) != branch_seed:
            raise GateError(f"{row_context}: branch_rng_seed does not match the branch")
        if _normalized(row.get("system")) != "j1j2":
            raise GateError(f"{row_context}: system must be j1j2")
        if str(row.get("experiment_id", "")) != experiment_id:
            raise GateError(f"{row_context}: experiment_id does not match the branch")
        if _integer(row, "checkpoint_step", row_context) != expected_checkpoint_step:
            raise GateError(
                f"{row_context}: checkpoint_step does not match the registered endpoint"
            )
        if _integer(row, "burn_in_rounds", row_context) != ENDPOINT_BURN_IN_ROUNDS:
            raise GateError(f"{row_context}: burn_in_rounds must be {ENDPOINT_BURN_IN_ROUNDS}")
        samples_per_round = _integer(row, "samples_per_round", row_context)
        rounds = _integer(row, "rounds", row_context)
        total_samples = _integer(row, "total_samples", row_context)
        if samples_per_round != SAMPLES_PER_BATCH:
            raise GateError(
                f"{row_context}: samples_per_round={samples_per_round}, "
                f"expected {SAMPLES_PER_BATCH}"
            )
        if total_samples != ENDPOINT_SAMPLES_PER_RUN:
            raise GateError(
                f"{row_context}: total_samples={total_samples}, expected {ENDPOINT_SAMPLES_PER_RUN}"
            )
        if rounds * samples_per_round != total_samples:
            raise GateError(f"{row_context}: endpoint sample accounting is inconsistent")
        energies.append(_finite_float(row, "energy_per_site", row_context))
        sigmas.append(_select_standard_error(row, row_context))
        sample_counts.append(total_samples)
        eval_seeds.append(_integer(row, "eval_seed", row_context))
        eval_runs.append(_integer(row, "eval_run", row_context))

    if len(set(eval_seeds)) != ENDPOINT_RUNS_PER_ARM:
        raise GateError(f"{context}: endpoint seeds for {arm} are not distinct")
    if set(eval_seeds) != set(ENDPOINT_EVAL_SEEDS):
        raise GateError(f"{context}: endpoint seeds for {arm} must be {ENDPOINT_EVAL_SEEDS}")
    if set(eval_runs) != {1, 2}:
        raise GateError(f"{context}: endpoint eval_run values for {arm} must be 1 and 2")
    counts = np.asarray(sample_counts, dtype=np.float64)
    weights = counts / float(counts.sum())
    difference_z = replica_difference_z(energies, sigmas)
    if not math.isfinite(difference_z) or difference_z > MAX_REPLICA_DIFFERENCE_Z:
        raise GateError(
            f"{context}: endpoint replicas for {arm} fail the consistency gate: "
            f"z={difference_z:.6g} > {MAX_REPLICA_DIFFERENCE_Z:g}"
        )
    pooled_energy = float(np.dot(weights, np.asarray(energies, dtype=np.float64)))
    pooled_sigma = float(np.sqrt(np.sum(np.square(weights * np.asarray(sigmas, dtype=np.float64)))))
    return PooledEndpoint(
        arm=arm,
        energy_per_site=pooled_energy,
        sigma_per_site=pooled_sigma,
        total_samples=int(counts.sum()),
        replica_difference_z=difference_z,
        inputs=tuple(dict(row) for row in selected),
    )


def _resolve_artifact_root(root: str | Path) -> Path:
    root = Path(root)
    candidates = (
        root,
        root / "online_training",
        root / "artifacts" / "online_training",
    )
    for candidate in candidates:
        if all((candidate / branch_id).is_dir() for branch_id in EXPECTED_BRANCHES):
            return candidate
    expected = ", ".join(EXPECTED_BRANCHES)
    raise GateError(f"could not find both online branches ({expected}) below {root}")


def _load_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise GateError(f"missing required artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"invalid {label} artifact: {path}") from error
    if not isinstance(payload, Mapping):
        raise GateError(f"invalid {label} artifact: {path}")
    return payload


def _require_exact_mssr_audit(payload: Any, context: str) -> None:
    if not isinstance(payload, Mapping):
        raise GateError(f"{context}: missing exact MS-SR protocol audit")
    expected = {
        "status": "audited_complete",
        "update_protocol": "fixed_grid_exact_loo",
        "stacking_protocol": "exact_leave_one_batch_out",
        "lambda_policy": "fixed_grid",
        "k": 4,
        "optimizer_updates": 125,
        "samples_per_candidate_batch": 8192,
        "candidate_batches_per_update": 4,
        "candidate_solves_per_update": 4,
        "sampling_solve_units": 500,
        "base_driver_sampling_calls": 0,
        "base_driver_solves": 0,
        "weight_fit_sampling_calls": 0,
        "weight_fit_solves": 0,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise GateError(
                f"{context}: exact MS-SR audit expected {key}={value!r}, found {payload.get(key)!r}"
            )
    try:
        grid = tuple(float(value) for value in payload["lambda_grid"])
        counts = tuple(int(value) for value in payload["candidate_sample_counts_final_update"])
    except (KeyError, TypeError, ValueError) as error:
        raise GateError(f"{context}: malformed exact MS-SR grid/sample audit") from error
    if grid != EXACT_MSSR_GRID:
        raise GateError(f"{context}: exact MS-SR lambda grid is missing or changed")
    if counts != (8192, 8192, 8192, 8192):
        raise GateError(f"{context}: exact MS-SR candidate sample counts are invalid")
    checkpoint_sha = payload.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
        raise GateError(f"{context}: exact MS-SR checkpoint hash is invalid")
    try:
        int(checkpoint_sha, 16)
    except ValueError as error:
        raise GateError(f"{context}: exact MS-SR checkpoint hash is invalid") from error


def _load_status(path: Path, *, experiment_id: str) -> None:
    payload = _load_json_mapping(path, "status")
    tokens = {
        _normalized(payload[key])
        for key in ("status", "run_status", "completion_status", "artifact_status")
        if key in payload
    }
    if payload.get("complete") is not True and not tokens.intersection(COMPLETE_VALUES):
        raise GateError(f"{path}: branch status is not complete")
    if payload.get("camera_ready_gate") is not True:
        raise GateError(f"{path}: camera_ready_gate is not true")
    if payload.get("experiment_id") != experiment_id:
        raise GateError(f"{path}: experiment_id does not match the branch")
    _require_exact_mssr_audit(
        payload.get("ms_sr_protocol_audit"),
        f"{path}, ms_sr_protocol_audit",
    )


def _validate_provenance(path: Path, *, experiment_id: str) -> None:
    payload = _load_json_mapping(path, "provenance")
    if payload.get("experiment_id") != experiment_id:
        raise GateError(f"{path}: experiment_id does not match the branch")
    _require_exact_mssr_audit(
        payload.get("ms_sr_protocol_audit"),
        f"{path}, ms_sr_protocol_audit",
    )


def load_online_branch(artifact_root: Path, branch_id: str) -> OnlineBranch:
    if branch_id not in EXPECTED_BRANCHES:
        raise GateError(f"undeclared online branch: {branch_id}")
    expected = EXPECTED_BRANCHES[branch_id]
    branch_root = artifact_root / branch_id
    curve_path = branch_root / "training_curve.csv"
    endpoint_path = branch_root / "endpoint_evaluations.csv"
    status_path = branch_root / "status.json"
    provenance_path = branch_root / "provenance.json"
    experiment_id = str(expected["experiment_id"])
    _load_status(status_path, experiment_id=experiment_id)
    _validate_provenance(provenance_path, experiment_id=experiment_id)
    curve_rows = _read_csv(curve_path)
    endpoint_rows = _read_csv(endpoint_path)
    complete_rows = [
        row for row in curve_rows if _normalized(row.get("run_status")) in COMPLETE_VALUES
    ]
    if len(complete_rows) != len(curve_rows):
        raise GateError(f"{curve_path}: finalized curve contains non-complete rows")
    observed_arms = {_normalized(row.get("arm")) for row in complete_rows}
    if observed_arms != set(REQUIRED_ARMS):
        raise GateError(
            f"{curve_path}: expected exactly arms {REQUIRED_ARMS}, found {observed_arms}"
        )

    curves: dict[str, tuple[dict[str, str], ...]] = {}
    for arm in REQUIRED_ARMS:
        arm_rows = [row for row in complete_rows if _normalized(row.get("arm")) == arm]
        if not arm_rows:
            raise GateError(f"{curve_path}: no complete rows for arm {arm}")
        sort_field = (
            "budget_equivalent_global_step"
            if arm == "shared_sr"
            else "sampling_budget_units_since_branch"
        )
        arm_rows.sort(key=lambda row: _finite_float(row, sort_field, str(curve_path)))
        curves[arm] = tuple(arm_rows)

    seeds = {_integer(row, "branch_rng_seed", str(curve_path)) for row in complete_rows}
    source_steps = {
        _integer(row, "source_checkpoint_step", str(curve_path)) for row in complete_rows
    }
    if seeds != {expected["seed"]}:
        raise GateError(f"{curve_path}: expected branch seed {expected['seed']}, found {seeds}")
    if source_steps != {expected["source_step"]}:
        raise GateError(
            f"{curve_path}: expected source step {expected['source_step']}, found {source_steps}"
        )
    experiment_ids = {str(row.get("experiment_id", "")) for row in complete_rows}
    if experiment_ids != {expected["experiment_id"]}:
        raise GateError(
            f"{curve_path}: expected experiment_id {expected['experiment_id']!r}, "
            f"found {experiment_ids}"
        )
    systems = {_normalized(row.get("system")) for row in complete_rows}
    if systems != {"j1j2"}:
        raise GateError(f"{curve_path}: expected system j1j2, found {systems}")

    expected_steps = {"sr": 500, "spring": 500, "ms_sr": 125}
    expected_batches = {"sr": 1, "spring": 1, "ms_sr": 4}
    expected_optimizers = {
        "sr": "sr",
        "spring": "spring",
        "ms_sr": "ms_sr_fixed_grid_exact_loo",
    }
    shared_rows = curves["shared_sr"]
    expected_shared_rows = int(expected["shared_sr_rows"])
    if len(shared_rows) != expected_shared_rows:
        raise GateError(
            f"{curve_path}: shared_sr has {len(shared_rows)} rows, expected {expected_shared_rows}"
        )
    shared_global_steps = [
        _integer(row, "budget_equivalent_global_step", str(curve_path)) for row in shared_rows
    ]
    if shared_global_steps[-1] != int(expected["source_step"]) or any(
        right <= left for left, right in zip(shared_global_steps, shared_global_steps[1:])
    ):
        raise GateError(
            f"{curve_path}: shared_sr must be strictly ordered and end at the source step"
        )
    for arm in BRANCH_ARMS:
        rows = curves[arm]
        if len(rows) != expected_steps[arm]:
            raise GateError(
                f"{curve_path}: {arm} has {len(rows)} rows, expected {expected_steps[arm]}"
            )
        optimizer_steps = [
            _integer(row, "optimizer_step_since_branch", str(curve_path)) for row in rows
        ]
        expected_optimizer_steps = list(range(1, expected_steps[arm] + 1))
        if optimizer_steps != expected_optimizer_steps:
            raise GateError(
                f"{curve_path}: {arm} optimizer steps are not contiguous 1--{expected_steps[arm]}"
            )
        budgets = [
            _finite_float(row, "sampling_budget_units_since_branch", str(curve_path))
            for row in rows
        ]
        expected_budgets = [
            float(expected_batches[arm] * step) for step in expected_optimizer_steps
        ]
        if budgets != expected_budgets:
            raise GateError(f"{curve_path}: {arm} sampling-budget sequence is invalid")
        global_steps = [
            _finite_float(row, "budget_equivalent_global_step", str(curve_path)) for row in rows
        ]
        if global_steps != [
            float(int(expected["source_step"]) + budget) for budget in expected_budgets
        ]:
            raise GateError(f"{curve_path}: {arm} global-budget sequence is invalid")
        optimizers = {_normalized(row.get("optimizer")) for row in rows}
        if optimizers != {expected_optimizers[arm]}:
            raise GateError(f"{curve_path}: unexpected optimizer for {arm}: {optimizers}")
        max_budget = max(
            _finite_float(row, "sampling_budget_units_since_branch", str(curve_path))
            for row in rows
        )
        max_step = max(
            _integer(row, "optimizer_step_since_branch", str(curve_path)) for row in rows
        )
        batches = {_integer(row, "batches_per_update", str(curve_path)) for row in rows}
        samples = {_integer(row, "samples_per_batch", str(curve_path)) for row in rows}
        if abs(max_budget - BRANCH_BUDGET_UNITS) > 1e-8:
            raise GateError(
                f"{curve_path}: {arm} ends at budget {max_budget}, expected {BRANCH_BUDGET_UNITS}"
            )
        if max_step != expected_steps[arm]:
            raise GateError(
                f"{curve_path}: {arm} ends at update {max_step}, expected {expected_steps[arm]}"
            )
        if batches != {expected_batches[arm]}:
            raise GateError(f"{curve_path}: unexpected batches_per_update for {arm}: {batches}")
        if samples != {SAMPLES_PER_BATCH}:
            raise GateError(f"{curve_path}: unexpected samples_per_batch for {arm}: {samples}")

    expected_endpoint_rows = len(BRANCH_ARMS) * ENDPOINT_RUNS_PER_ARM
    if len(endpoint_rows) != expected_endpoint_rows:
        raise GateError(
            f"{endpoint_path}: expected exactly {expected_endpoint_rows} endpoint rows, "
            f"found {len(endpoint_rows)}"
        )
    endpoints = {
        arm: _pool_endpoints(
            endpoint_rows,
            arm=arm,
            branch_seed=int(expected["seed"]),
            experiment_id=str(expected["experiment_id"]),
            expected_checkpoint_step=int(expected["endpoint_steps"][arm]),
            context=str(endpoint_path),
        )
        for arm in BRANCH_ARMS
    }
    return OnlineBranch(
        branch_id=branch_id,
        seed=int(expected["seed"]),
        source_step=int(expected["source_step"]),
        curves=curves,
        endpoints=endpoints,
        input_files=(curve_path, endpoint_path, status_path, provenance_path),
    )


def load_online_artifacts(input_root: str | Path) -> tuple[OnlineBranch, OnlineBranch]:
    """Validate and load exactly the two preregistered online branches."""

    root = _resolve_artifact_root(input_root)
    branches = tuple(load_online_branch(root, branch_id) for branch_id in EXPECTED_BRANCHES)
    return branches[0], branches[1]


def _raw_branch_xy(branch: OnlineBranch, arm: str) -> tuple[np.ndarray, np.ndarray]:
    rows = branch.curves[arm]
    x = np.asarray([float(row["sampling_budget_units_since_branch"]) for row in rows], dtype=float)
    y = np.asarray([float(row["energy_per_site"]) for row in rows], dtype=float)
    source_energy = float(branch.curves["shared_sr"][-1]["energy_per_site"])
    if x[0] > 0.0:
        x = np.concatenate(([0.0], x))
        y = np.concatenate(([source_energy], y))
    return x, y


def budget_rolling_xy(
    branch: OnlineBranch,
    arm: str,
    *,
    window_budget: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate to one-budget units before applying a common rolling mean."""

    if window_budget <= 0 or window_budget > BRANCH_BUDGET_UNITS + 1:
        raise ValueError(f"invalid rolling window: {window_budget}")
    raw_x, raw_y = _raw_branch_xy(branch, arm)
    if raw_x[0] > 0.0 or raw_x[-1] < BRANCH_BUDGET_UNITS:
        raise GateError(
            f"{branch.branch_id}/{arm}: curve does not span budget 0--{BRANCH_BUDGET_UNITS}"
        )
    grid = np.arange(BRANCH_BUDGET_UNITS + 1, dtype=float)
    interpolated = np.interp(grid, raw_x, raw_y)
    kernel = np.ones(window_budget, dtype=float) / float(window_budget)
    smoothed = np.convolve(interpolated, kernel, mode="valid")
    return grid[window_budget - 1 :], smoothed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plot_online_comparison(
    input_root: str | Path,
    output_dir: str | Path,
    *,
    smoothing_window: int = 25,
) -> tuple[Path, Path, Path]:
    """Validate both branches, then write PDF/PNG and auditable pooled endpoints."""

    branches = load_online_artifacts(input_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        1,
        len(branches),
        figsize=(7.25, 2.85),
        squeeze=False,
        sharey=True,
    )
    for axis, branch in zip(axes[0], branches, strict=True):
        source_energy = float(branch.curves["shared_sr"][-1]["energy_per_site"])
        axis.scatter([0], [source_energy], marker="x", color="black", s=18, zorder=6)
        for arm in BRANCH_ARMS:
            raw_x, raw_y = _raw_branch_xy(branch, arm)
            axis.plot(raw_x, raw_y, color=COLORS[arm], linewidth=0.45, alpha=0.2)
            smooth_x, smooth_y = budget_rolling_xy(
                branch,
                arm,
                window_budget=smoothing_window,
            )
            axis.plot(
                smooth_x,
                smooth_y,
                color=COLORS[arm],
                linewidth=1.35,
                label=LABELS[arm],
            )
            endpoint = branch.endpoints[arm]
            axis.errorbar(
                [BRANCH_BUDGET_UNITS],
                [endpoint.energy_per_site],
                yerr=[endpoint.sigma_per_site],
                fmt="D",
                color=COLORS[arm],
                markeredgecolor="black",
                markeredgewidth=0.45,
                markersize=4.3,
                capsize=2.0,
                linewidth=0.8,
                zorder=5,
            )
        axis.set_title(f"seed {branch.seed}, branch at {branch.source_step}", fontsize=9)
        axis.set_xlabel("sampling-and-solve budget since branch", fontsize=8)
        axis.set_xlim(0, BRANCH_BUDGET_UNITS)
        axis.tick_params(labelsize=7)
        axis.grid(alpha=0.2, linewidth=0.5)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0, 0].set_ylabel("energy per site", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=7.5)
    figure.subplots_adjust(left=0.09, right=0.995, bottom=0.19, top=0.78, wspace=0.25)

    pdf_path = output_dir / "online_comparison.pdf"
    png_path = output_dir / "online_comparison.png"
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata=pdf_metadata("Equal-budget online SR comparison"),
    )
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    endpoint_path = output_dir / "pooled_online_endpoints.json"
    input_files = sorted({path.resolve() for branch in branches for path in branch.input_files})
    atomic_write_json(
        {
            "schema": "sr_filter.online_comparison.v1",
            "ready": True,
            "protocol": {
                "required_branches": list(EXPECTED_BRANCHES),
                "budget_units_per_branch": BRANCH_BUDGET_UNITS,
                "samples_per_batch": SAMPLES_PER_BATCH,
                "endpoint_runs_per_arm": ENDPOINT_RUNS_PER_ARM,
                "endpoint_rounds_per_run": ENDPOINT_ROUNDS_PER_RUN,
                "endpoint_burn_in_rounds": ENDPOINT_BURN_IN_ROUNDS,
                "endpoint_samples_per_run": ENDPOINT_SAMPLES_PER_RUN,
                "max_endpoint_replica_difference_z": MAX_REPLICA_DIFFERENCE_Z,
            },
            "branches": [
                {
                    "id": branch.branch_id,
                    "branch_seed": branch.seed,
                    "source_checkpoint_step": branch.source_step,
                    "endpoints": {arm: branch.endpoints[arm].as_dict() for arm in BRANCH_ARMS},
                }
                for branch in branches
            ],
            "inputs": [
                {
                    "path": "/".join(path.parts[-2:]),
                    "sha256": _sha256(path),
                }
                for path in input_files
            ],
        },
        endpoint_path,
    )
    return pdf_path, png_path, endpoint_path
