#!/usr/bin/env python3
"""Validate two online branches and generate Figure 5 plus TeX fragments.

This is intentionally a strict gate: no manuscript asset or prose fragment is
written unless both declared branches are complete and their endpoint rows pass
the artifact audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_ROOT.parent
DEFAULT_MANIFEST = PAPER_ROOT / "online_artifacts.json"
FIGURE_PATH = PAPER_ROOT / "figures" / "figure5_equal_budget_online.pdf"
RESULTS_TEX = PAPER_ROOT / "generated" / "online_comparison_results.tex"
PROTOCOL_TEX = PAPER_ROOT / "generated" / "online_protocol_table.tex"
STATUS_JSON = PAPER_ROOT / "generated" / "online_artifact_status.json"
DEFAULT_SOURCE_DATE_EPOCH = 1767225600  # 2026-01-01 00:00:00 UTC.
MAX_ENDPOINT_REPLICA_DIFFERENCE_Z = 5.0

COMPLETE_VALUES = {"complete", "completed", "ready"}
AUDIT_VALUES = {
    "pass",
    "passed",
    "audited",
    "audited_complete",
    "ok",
    "complete",
    "completed",
}
BRANCH_ARMS = ("sr", "spring", "ms_sr")
EXACT_MSSR_GRID = (1e-2, 1e-3, 1e-4, 1e-5)
COLORS = {"shared_sr": "#777777", "sr": "#2570b7", "spring": "#238b57", "ms_sr": "#c43b46"}
LABELS = {"shared_sr": "shared SR", "sr": "SR", "spring": "SPRING", "ms_sr": "MS-SR ($K=4$)"}


class GateError(RuntimeError):
    pass


def pdf_metadata(title: str) -> dict[str, object]:
    """Return stable PDF metadata controlled by SOURCE_DATE_EPOCH."""
    raw_epoch = os.environ.get("SOURCE_DATE_EPOCH", str(DEFAULT_SOURCE_DATE_EPOCH))
    try:
        epoch = int(raw_epoch)
        timestamp = datetime.fromtimestamp(epoch, tz=timezone.utc)
    except (ValueError, OverflowError, OSError) as error:
        raise GateError(f"invalid SOURCE_DATE_EPOCH={raw_epoch!r}") from error
    return {
        "Title": title,
        "Author": "Tak Hur",
        "Creator": "sr_filter manuscript figure generator",
        "Producer": "Matplotlib PDF backend",
        "CreationDate": timestamp,
        "ModDate": timestamp,
    }


@dataclass(frozen=True)
class Endpoint:
    arm: str
    energy: float
    standard_error: float
    total_samples: int
    runs: int
    replica_difference_z: float
    error_field: str
    eval_runs: tuple[int, ...]
    eval_seeds: tuple[int, ...]
    burn_in_rounds: tuple[int, ...]
    rounds: tuple[int, ...]
    samples_per_round: tuple[int, ...]


@dataclass
class Branch:
    branch_id: str
    seed: int
    source_step: int
    curves: dict[str, list[dict[str, str]]]
    endpoints: dict[str, Endpoint]
    input_files: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise GateError(f"missing required artifact: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_float(row: dict[str, str], field: str, context: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise GateError(f"{context}: invalid {field!r}") from error
    if not math.isfinite(value):
        raise GateError(f"{context}: non-finite {field!r}")
    return value


def integer(row: dict[str, str], field: str, context: str) -> int:
    value = finite_float(row, field, context)
    rounded = round(value)
    if abs(value - rounded) > 1e-9:
        raise GateError(f"{context}: {field!r} is not an integer")
    return int(rounded)


def normalized(value: Any) -> str:
    return str(value).strip().lower()


def load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise GateError(f"missing required artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"invalid {label} artifact: {path}") from error
    if not isinstance(payload, dict):
        raise GateError(f"invalid {label} artifact: {path}")
    return payload


def require_exact_mssr_audit(payload: Any, context: str) -> None:
    if not isinstance(payload, dict):
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


def select_standard_error(row: dict[str, str], context: str) -> tuple[float, str]:
    # The artifact contract defines sigma_per_site as the conservative reported
    # one-SE estimate: max(round-level, block-8-level) for each audited run.
    for field in (
        "sigma_per_site",
        "sigma_block8_level_per_site",
        "sigma_round_level_per_site",
    ):
        raw = row.get(field, "")
        if raw in (None, ""):
            continue
        value = finite_float(row, field, context)
        if value > 0:
            return value, field
    raise GateError(f"{context}: no positive audited standard-error field")


def pool_endpoints(
    rows: list[dict[str, str]],
    arm: str,
    context: str,
    *,
    expected_runs: int,
    expected_rounds: int,
    expected_burn_in_rounds: int,
    expected_samples: int,
    expected_seed: int,
    expected_experiment_id: str,
    expected_checkpoint_step: int,
    expected_eval_seeds: tuple[int, ...],
) -> Endpoint:
    selected = [
        row
        for row in rows
        if normalized(row.get("arm")) == arm
        and row.get("rounds", "") not in (None, "")
        and integer(row, "rounds", context) == expected_rounds
    ]
    if not selected:
        raise GateError(f"{context}: no {expected_rounds}-round endpoint for {arm}")
    if len(selected) != expected_runs:
        raise GateError(
            f"{context}: expected {expected_runs} {expected_rounds}-round endpoints "
            f"for {arm}, found {len(selected)}"
        )
    energies: list[float] = []
    errors: list[float] = []
    samples: list[int] = []
    fields: list[str] = []
    eval_runs: list[int] = []
    eval_seeds: list[int] = []
    burn_in_rounds: list[int] = []
    rounds: list[int] = []
    samples_per_round: list[int] = []
    for index, row in enumerate(selected):
        row_context = f"{context}, endpoint {arm}[{index}]"
        if normalized(row.get("run_status")) not in COMPLETE_VALUES:
            raise GateError(f"{row_context}: run_status is not complete")
        if normalized(row.get("audit_status")) not in AUDIT_VALUES:
            raise GateError(f"{row_context}: audit_status is not passing")
        if integer(row, "branch_rng_seed", row_context) != expected_seed:
            raise GateError(f"{row_context}: branch_rng_seed does not match the branch")
        if normalized(row.get("system")) != "j1j2":
            raise GateError(f"{row_context}: system must be j1j2")
        if str(row.get("experiment_id", "")) != expected_experiment_id:
            raise GateError(f"{row_context}: experiment_id does not match the branch")
        if integer(row, "checkpoint_step", row_context) != expected_checkpoint_step:
            raise GateError(
                f"{row_context}: checkpoint_step does not match the registered endpoint"
            )
        energies.append(finite_float(row, "energy_per_site", row_context))
        standard_error, field = select_standard_error(row, row_context)
        errors.append(standard_error)
        fields.append(field)
        count = integer(row, "total_samples", row_context)
        if count != expected_samples:
            raise GateError(f"{row_context}: total_samples={count}, expected {expected_samples}")
        samples.append(count)
        eval_runs.append(integer(row, "eval_run", row_context))
        eval_seeds.append(integer(row, "eval_seed", row_context))
        row_burn_in_rounds = integer(row, "burn_in_rounds", row_context)
        if row_burn_in_rounds != expected_burn_in_rounds:
            raise GateError(
                f"{row_context}: burn_in_rounds={row_burn_in_rounds}, "
                f"expected {expected_burn_in_rounds}"
            )
        burn_in_rounds.append(row_burn_in_rounds)
        row_rounds = integer(row, "rounds", row_context)
        row_samples_per_round = integer(row, "samples_per_round", row_context)
        if row_rounds * row_samples_per_round != count:
            raise GateError(
                f"{row_context}: rounds * samples_per_round does not equal total_samples"
            )
        rounds.append(row_rounds)
        samples_per_round.append(row_samples_per_round)

    if len(set(eval_seeds)) != expected_runs:
        raise GateError(f"{context}: endpoint evaluation seeds for {arm} are not distinct")
    if set(eval_seeds) != set(expected_eval_seeds):
        raise GateError(
            f"{context}: endpoint evaluation seeds for {arm} must be {expected_eval_seeds}"
        )
    if len(set(eval_runs)) != expected_runs:
        raise GateError(f"{context}: endpoint evaluation run IDs for {arm} are not distinct")
    if set(eval_runs) != set(range(1, expected_runs + 1)):
        raise GateError(
            f"{context}: endpoint evaluation run IDs for {arm} must be 1--{expected_runs}"
        )

    difference_z = abs(energies[0] - energies[1]) / math.hypot(errors[0], errors[1])
    if not math.isfinite(difference_z) or difference_z > MAX_ENDPOINT_REPLICA_DIFFERENCE_Z:
        raise GateError(
            f"{context}: endpoint replicas for {arm} fail the consistency gate: "
            f"z={difference_z:.6g} > {MAX_ENDPOINT_REPLICA_DIFFERENCE_Z:g}"
        )

    total = sum(samples)
    weights = np.asarray(samples, dtype=float) / total
    energy = float(np.dot(weights, np.asarray(energies, dtype=float)))
    standard_error = float(math.sqrt(np.sum((weights * np.asarray(errors, dtype=float)) ** 2)))
    return Endpoint(
        arm=arm,
        energy=energy,
        standard_error=standard_error,
        total_samples=total,
        runs=len(selected),
        replica_difference_z=difference_z,
        error_field="mixed" if len(set(fields)) > 1 else fields[0],
        eval_runs=tuple(eval_runs),
        eval_seeds=tuple(eval_seeds),
        burn_in_rounds=tuple(burn_in_rounds),
        rounds=tuple(rounds),
        samples_per_round=tuple(samples_per_round),
    )


def load_branch(root: Path, spec: dict[str, Any], manifest: dict[str, Any]) -> Branch:
    branch_id = str(spec["id"])
    branch_root = root / str(spec["directory"])
    curve_path = branch_root / str(manifest["training_curve_file"])
    endpoint_path = branch_root / str(manifest["endpoint_file"])
    status_path = branch_root / str(manifest["status_file"])
    provenance_path = branch_root / str(manifest["provenance_file"])
    status_payload = load_json_mapping(status_path, "status")
    provenance_payload = load_json_mapping(provenance_path, "provenance")
    expected_experiment_id = str(spec["expected_experiment_id"])
    status_tokens = {
        normalized(status_payload.get(key))
        for key in ("status", "run_status", "completion_status", "artifact_status")
        if key in status_payload
    }
    explicit_complete = status_payload.get("complete") is True
    if not explicit_complete and not (status_tokens & COMPLETE_VALUES):
        raise GateError(f"{status_path}: branch status is not complete")
    if status_payload.get("camera_ready_gate") is not True:
        raise GateError(f"{status_path}: camera_ready_gate is not true")
    if status_payload.get("experiment_id") != expected_experiment_id:
        raise GateError(f"{status_path}: experiment_id does not match the manifest")
    if provenance_payload.get("experiment_id") != expected_experiment_id:
        raise GateError(f"{provenance_path}: experiment_id does not match the manifest")
    require_exact_mssr_audit(
        status_payload.get("ms_sr_protocol_audit"),
        f"{status_path}, ms_sr_protocol_audit",
    )
    require_exact_mssr_audit(
        provenance_payload.get("ms_sr_protocol_audit"),
        f"{provenance_path}, ms_sr_protocol_audit",
    )

    curve_rows = read_csv(curve_path)
    endpoint_rows = read_csv(endpoint_path)
    if not curve_rows:
        raise GateError(f"{curve_path}: empty training curve")

    complete_rows = [
        row for row in curve_rows if normalized(row.get("run_status")) in COMPLETE_VALUES
    ]
    if len(complete_rows) != len(curve_rows):
        raise GateError(f"{curve_path}: finalized curve contains non-complete rows")
    observed_arms = {normalized(row.get("arm")) for row in complete_rows}
    required_arms = tuple(str(arm) for arm in manifest["required_arms"])
    if observed_arms != set(required_arms):
        raise GateError(
            f"{curve_path}: expected exactly arms {required_arms}, found {observed_arms}"
        )
    curves: dict[str, list[dict[str, str]]] = {}
    for arm in required_arms:
        arm_rows = [row for row in complete_rows if normalized(row.get("arm")) == arm]
        if not arm_rows:
            raise GateError(f"{curve_path}: no complete rows for arm {arm}")
        sort_field = (
            "budget_equivalent_global_step"
            if arm == "shared_sr"
            else "sampling_budget_units_since_branch"
        )
        arm_rows.sort(key=lambda row: finite_float(row, sort_field, str(curve_path)))
        curves[arm] = arm_rows

    seeds = {integer(row, "branch_rng_seed", str(curve_path)) for row in complete_rows}
    source_steps = {
        integer(row, "source_checkpoint_step", str(curve_path)) for row in complete_rows
    }
    if len(seeds) != 1 or len(source_steps) != 1:
        raise GateError(f"{curve_path}: expected one branch seed and one source step")
    seed = seeds.pop()
    source_step = source_steps.pop()
    if seed != int(spec["expected_seed"]):
        raise GateError(f"{curve_path}: expected seed {spec['expected_seed']}, found {seed}")
    if source_step != int(spec["expected_source_step"]):
        raise GateError(
            f"{curve_path}: expected source step {spec['expected_source_step']}, "
            f"found {source_step}"
        )
    experiment_ids = {str(row.get("experiment_id", "")) for row in complete_rows}
    if experiment_ids != {expected_experiment_id}:
        raise GateError(
            f"{curve_path}: expected experiment_id {expected_experiment_id!r}, "
            f"found {experiment_ids}"
        )
    systems = {normalized(row.get("system")) for row in complete_rows}
    if systems != {"j1j2"}:
        raise GateError(f"{curve_path}: expected system j1j2, found {systems}")

    target_budget = int(manifest["branch_budget_units"])
    expected_samples = int(manifest["samples_per_batch"])
    expected_steps = {"sr": target_budget, "spring": target_budget, "ms_sr": target_budget // 4}
    expected_batches = {"sr": 1, "spring": 1, "ms_sr": 4}
    expected_optimizers = {
        "sr": "sr",
        "spring": "spring",
        "ms_sr": "ms_sr_fixed_grid_exact_loo",
    }
    shared_rows = curves["shared_sr"]
    expected_shared_rows = int(spec["expected_shared_sr_rows"])
    if len(shared_rows) != expected_shared_rows:
        raise GateError(
            f"{curve_path}: shared_sr has {len(shared_rows)} rows, expected {expected_shared_rows}"
        )
    shared_global_steps = [
        integer(row, "budget_equivalent_global_step", str(curve_path)) for row in shared_rows
    ]
    if shared_global_steps[-1] != source_step or any(
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
            integer(row, "optimizer_step_since_branch", str(curve_path)) for row in rows
        ]
        expected_optimizer_steps = list(range(1, expected_steps[arm] + 1))
        if optimizer_steps != expected_optimizer_steps:
            raise GateError(
                f"{curve_path}: {arm} optimizer steps are not contiguous 1--{expected_steps[arm]}"
            )
        budgets = [
            finite_float(row, "sampling_budget_units_since_branch", str(curve_path)) for row in rows
        ]
        expected_budgets = [
            float(expected_batches[arm] * step) for step in expected_optimizer_steps
        ]
        if budgets != expected_budgets:
            raise GateError(f"{curve_path}: {arm} sampling-budget sequence is invalid")
        global_steps = [
            finite_float(row, "budget_equivalent_global_step", str(curve_path)) for row in rows
        ]
        if global_steps != [float(source_step + budget) for budget in expected_budgets]:
            raise GateError(f"{curve_path}: {arm} global-budget sequence is invalid")
        optimizers = {normalized(row.get("optimizer")) for row in rows}
        if optimizers != {expected_optimizers[arm]}:
            raise GateError(f"{curve_path}: unexpected optimizer for {arm}: {optimizers}")
        max_budget = max(
            finite_float(row, "sampling_budget_units_since_branch", str(curve_path)) for row in rows
        )
        max_step = max(integer(row, "optimizer_step_since_branch", str(curve_path)) for row in rows)
        if abs(max_budget - target_budget) > 1e-8:
            raise GateError(
                f"{curve_path}: {arm} ends at budget {max_budget}, expected {target_budget}"
            )
        if max_step != expected_steps[arm]:
            raise GateError(
                f"{curve_path}: {arm} ends at update {max_step}, expected {expected_steps[arm]}"
            )
        batches = {integer(row, "batches_per_update", str(curve_path)) for row in rows}
        samples_per_batch = {integer(row, "samples_per_batch", str(curve_path)) for row in rows}
        if batches != {expected_batches[arm]}:
            raise GateError(f"{curve_path}: unexpected batches_per_update for {arm}: {batches}")
        if samples_per_batch != {expected_samples}:
            raise GateError(
                f"{curve_path}: unexpected samples_per_batch for {arm}: {samples_per_batch}"
            )

    expected_endpoint_rows = len(BRANCH_ARMS) * int(manifest["endpoint_runs_per_arm"])
    if len(endpoint_rows) != expected_endpoint_rows:
        raise GateError(
            f"{endpoint_path}: expected exactly {expected_endpoint_rows} endpoint rows, "
            f"found {len(endpoint_rows)}"
        )
    endpoints = {
        arm: pool_endpoints(
            endpoint_rows,
            arm,
            str(endpoint_path),
            expected_runs=int(manifest["endpoint_runs_per_arm"]),
            expected_rounds=int(manifest["endpoint_rounds_per_run"]),
            expected_burn_in_rounds=int(manifest["endpoint_burn_in_rounds"]),
            expected_samples=int(manifest["endpoint_samples_per_run"]),
            expected_seed=seed,
            expected_experiment_id=expected_experiment_id,
            expected_checkpoint_step=int(spec["expected_endpoint_steps"][arm]),
            expected_eval_seeds=tuple(int(value) for value in manifest["endpoint_eval_seeds"]),
        )
        for arm in BRANCH_ARMS
    }
    return Branch(
        branch_id=branch_id,
        seed=seed,
        source_step=source_step,
        curves=curves,
        endpoints=endpoints,
        input_files=[curve_path, endpoint_path, status_path, provenance_path],
    )


def raw_branch_xy(branch: Branch, arm: str) -> tuple[np.ndarray, np.ndarray]:
    rows = branch.curves[arm]
    x = np.asarray([float(row["sampling_budget_units_since_branch"]) for row in rows], dtype=float)
    y = np.asarray([float(row["energy_per_site"]) for row in rows], dtype=float)
    source_energy = float(branch.curves["shared_sr"][-1]["energy_per_site"])
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or not math.isfinite(source_energy):
        raise GateError(f"{branch.branch_id}/{arm}: curve contains a non-finite value")
    if len(x) > 1 and np.any(np.diff(x) <= 0):
        raise GateError(f"{branch.branch_id}/{arm}: branch budget is not strictly increasing")
    if x[0] > 0:
        x = np.concatenate(([0.0], x))
        y = np.concatenate(([source_energy], y))
    return x, y


def budget_rolling_xy(
    branch: Branch, arm: str, *, window_budget: int = 25, max_budget: int = 500
) -> tuple[np.ndarray, np.ndarray]:
    raw_x, raw_y = raw_branch_xy(branch, arm)
    if raw_x[0] > 0 or raw_x[-1] < max_budget:
        raise GateError(f"{branch.branch_id}/{arm}: curve does not span budget 0--{max_budget}")
    # Interpolate every arm to the same one-budget-unit grid before smoothing.
    # This prevents a 25-point window from meaning 25 units for SR but 100 for
    # K=4 MS-SR.
    grid = np.arange(0, max_budget + 1, dtype=float)
    interpolated = np.interp(grid, raw_x, raw_y)
    kernel = np.ones(window_budget, dtype=float) / window_budget
    smoothed = np.convolve(interpolated, kernel, mode="valid")
    return grid[window_budget - 1 :], smoothed


def render_figure(branches: list[Branch]) -> None:
    fig, axes = plt.subplots(1, len(branches), figsize=(7.25, 2.85), squeeze=False, sharey=True)
    for ax, branch in zip(axes[0], branches):
        source_energy = float(branch.curves["shared_sr"][-1]["energy_per_site"])
        ax.scatter([0], [source_energy], marker="x", color="black", s=18, linewidth=0.8, zorder=6)
        for arm in BRANCH_ARMS:
            raw_x, raw_y = raw_branch_xy(branch, arm)
            ax.plot(raw_x, raw_y, color=COLORS[arm], linewidth=0.45, alpha=0.20)
            smooth_x, smooth_y = budget_rolling_xy(branch, arm, window_budget=25)
            ax.plot(smooth_x, smooth_y, color=COLORS[arm], linewidth=1.35, label=LABELS[arm])
        endpoint_x = 500
        for arm in BRANCH_ARMS:
            endpoint = branch.endpoints[arm]
            ax.errorbar(
                [endpoint_x],
                [endpoint.energy],
                yerr=[endpoint.standard_error],
                fmt="D",
                color=COLORS[arm],
                markeredgecolor="black",
                markeredgewidth=0.45,
                markersize=4.3,
                capsize=2.0,
                linewidth=0.8,
                zorder=5,
            )
        ax.set_title(f"seed {branch.seed}, branch at {branch.source_step}", fontsize=9)
        ax.set_xlabel("sampling-and-solve budget since branch", fontsize=8)
        ax.set_xlim(0, 500)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0, 0].set_ylabel("energy per site", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=7.5)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.19, top=0.78, wspace=0.25)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_PATH,
        bbox_inches="tight",
        metadata=pdf_metadata("Equal-budget online SR comparison"),
    )
    plt.close(fig)


def tex_number(value: float) -> str:
    return f"{value:.7f}"


def tex_signed_number(value: float) -> str:
    return f"{value:+.7f}"


def tex_se(value: float) -> str:
    exponent = math.floor(math.log10(value))
    mantissa = value / (10**exponent)
    return f"{mantissa:.2f}\\times10^{{{exponent}}}"


def arm_label(arm: str) -> str:
    return LABELS[arm].replace(" ($K=4$)", "")


def endpoint_ordering(branch: Branch) -> tuple[str, ...]:
    return tuple(sorted(BRANCH_ARMS, key=lambda arm: (branch.endpoints[arm].energy, arm)))


def endpoint_sentence(branch: Branch) -> str:
    values = []
    for arm in ("ms_sr", "sr", "spring"):
        endpoint = branch.endpoints[arm]
        values.append(
            f"{arm_label(arm)} ${tex_number(endpoint.energy)}"
            f"\\mathbin{{\\pm}}{tex_se(endpoint.standard_error)}$"
        )
    ordering = endpoint_ordering(branch)
    ordering_tex = " $<$ ".join(arm_label(arm) for arm in ordering)
    comparisons = []
    for comparator in ("sr", "spring"):
        difference = branch.endpoints["ms_sr"].energy - branch.endpoints[comparator].energy
        combined_se = math.hypot(
            branch.endpoints["ms_sr"].standard_error,
            branch.endpoints[comparator].standard_error,
        )
        z = difference / combined_se
        comparisons.append(
            f"MS-SR minus {arm_label(comparator)} is "
            f"${tex_signed_number(difference)}$ per site "
            f"(${z:+.2f}$ combined endpoint Monte Carlo standard errors)"
        )
    return (
        f"For seed {branch.seed} (branch step {branch.source_step}), "
        + ", ".join(values)
        + ". "
        + f"The numerical endpoint ordering (lower is better) is {ordering_tex}; "
        + "; ".join(comparisons)
        + "."
    )


def write_results_tex(branches: list[Branch]) -> None:
    sentences = "\n".join(endpoint_sentence(branch) for branch in branches)
    content = rf"""\begin{{figure}}[t]
    \centering
    \includegraphics[width=\textwidth]{{figures/figure5_equal_budget_online.pdf}}
    \caption{{Equal-budget late-stage online comparison on the $8\times8$
    $J_1$-$J_2$ model with a no-symmetry ViT. Each panel branches SR, SPRING,
    and fixed-grid MS-SR from identical parameters. One SR or SPRING update is
    one budget unit; one $K=4$ MS-SR update is four. Faint lines are raw
    training-energy estimates and solid lines are rolling means over a common
    25-budget-unit window. Diamonds are audited fresh-chain
    endpoint evaluations; bars are one conservative reported Monte Carlo
    standard error, not variation across seeds.}}
    \label{{fig:mssr_equal_budget_online}}
\end{{figure}}

Figure~\ref{{fig:mssr_equal_budget_online}} compares two controlled late-stage
branches at equal sampling-and-solve budget. Each arm receives $500$ units, so
SR and SPRING take $500$ updates while MS-SR takes $125$.
{sentences}
These are two individual branched trajectories; the endpoint uncertainties
measure Monte Carlo evaluation error and do not estimate across-seed variation.
"""
    RESULTS_TEX.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_TEX.write_text(content, encoding="utf-8")


def write_protocol_tex(branches: list[Branch]) -> None:
    rows = []
    for branch in branches:
        endpoint_counts = [
            f"{branch.endpoints[arm].total_samples:,}" for arm in ("sr", "spring", "ms_sr")
        ]
        rows.append(
            f"seed {branch.seed} & {branch.source_step} & 500 / 500 / 125 & "
            + " / ".join(endpoint_counts)
            + r" \\"
        )
    seed_descriptions = []
    for branch in branches:
        arm_seeds = [
            "/".join(str(seed) for seed in branch.endpoints[arm].eval_seeds)
            for arm in ("sr", "spring", "ms_sr")
        ]
        seed_descriptions.append(f"seed {branch.seed}: " + " / ".join(arm_seeds))
    protocol_groups: dict[tuple[int, int, int], list[str]] = {}
    for branch in branches:
        for arm in ("sr", "spring", "ms_sr"):
            endpoint = branch.endpoints[arm]
            for run_index, (rounds, burn_in, samples_per_round) in enumerate(
                zip(endpoint.rounds, endpoint.burn_in_rounds, endpoint.samples_per_round)
            ):
                key = (rounds, burn_in, samples_per_round)
                run_name = (
                    f"seed {branch.seed} "
                    f"{LABELS[arm].replace(' ($K=4$)', '')} "
                    f"run {endpoint.eval_runs[run_index]}"
                )
                protocol_groups.setdefault(key, []).append(run_name)
    if len(protocol_groups) == 1:
        rounds, burn_in, samples_per_round = next(iter(protocol_groups))
        protocol_description = (
            f"Each endpoint replica uses {rounds} evaluation rounds, "
            f"{burn_in} burn-in rounds, and {samples_per_round:,} samples per round."
        )
    else:
        descriptions = []
        for (rounds, burn_in, samples_per_round), run_names in sorted(protocol_groups.items()):
            descriptions.append(
                ", ".join(run_names)
                + f": {rounds} rounds, {burn_in} burn-in rounds, and "
                + f"{samples_per_round:,} samples per round"
            )
        protocol_description = "Endpoint protocols are " + "; ".join(descriptions) + "."
    protocol_line = (
        protocol_description
        + " Endpoint evaluation seeds (SR / SPRING / MS-SR) are "
        + "; ".join(seed_descriptions)
        + "."
    )
    content = (
        r"""\begin{table}[h]
\centering
\small
\begin{tabular}{lccc}
\toprule
Branch & Source step & Updates (SR / SPRING / MS-SR)
  & Endpoint samples (SR / SPRING / MS-SR) \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\caption{Online-branch accounting from the audited artifacts. Each optimizer
receives 500 budget units; all candidate batches contain 8192 samples.}
\label{tab:online_protocol}
\end{table}
"""
        + "\n"
        + protocol_line
        + "\n"
    )
    PROTOCOL_TEX.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL_TEX.write_text(content, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_status(branches: list[Branch], manifest_path: Path) -> None:
    inputs = sorted({path.resolve() for branch in branches for path in branch.input_files})
    payload = {
        "schema_version": 1,
        "ready": True,
        "required_branch_count": 2,
        "max_endpoint_replica_difference_z": MAX_ENDPOINT_REPLICA_DIFFERENCE_Z,
        "validated_branches": [
            {
                "id": branch.branch_id,
                "seed": branch.seed,
                "source_checkpoint_step": branch.source_step,
                "endpoint_samples": {
                    arm: branch.endpoints[arm].total_samples for arm in BRANCH_ARMS
                },
                "endpoint_replica_difference_z": {
                    arm: branch.endpoints[arm].replica_difference_z for arm in BRANCH_ARMS
                },
                "endpoint_energy_per_site": {
                    arm: branch.endpoints[arm].energy for arm in BRANCH_ARMS
                },
                "endpoint_standard_error_per_site": {
                    arm: branch.endpoints[arm].standard_error for arm in BRANCH_ARMS
                },
                "endpoint_ordering_lower_is_better": list(endpoint_ordering(branch)),
                "best_vs_next_best_gap_per_site": (
                    branch.endpoints[endpoint_ordering(branch)[1]].energy
                    - branch.endpoints[endpoint_ordering(branch)[0]].energy
                ),
                "ms_sr_minus_others_per_site": {
                    arm: branch.endpoints["ms_sr"].energy - branch.endpoints[arm].energy
                    for arm in ("sr", "spring")
                },
                "ms_sr_minus_others_combined_se_z": {
                    arm: (
                        branch.endpoints["ms_sr"].energy - branch.endpoints[arm].energy
                    )
                    / math.hypot(
                        branch.endpoints["ms_sr"].standard_error,
                        branch.endpoints[arm].standard_error,
                    )
                    for arm in ("sr", "spring")
                },
            }
            for branch in branches
        ],
        "inputs": [
            {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path)} for path in inputs
        ],
        "manifest_sha256": sha256(manifest_path),
        "outputs": [
            str(FIGURE_PATH.relative_to(REPO_ROOT)),
            str(RESULTS_TEX.relative_to(REPO_ROOT)),
            str(PROTOCOL_TEX.relative_to(REPO_ROOT)),
        ],
    }
    STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != 1:
        raise GateError(f"{manifest_path}: unsupported manifest schema")
    artifact_root = (manifest_path.parent / str(manifest["artifact_root"])).resolve()
    specs = manifest.get("required_branches", [])
    if len(specs) != 2:
        raise GateError(f"{manifest_path}: exactly two required branches are needed")
    branches = [load_branch(artifact_root, spec, manifest) for spec in specs]
    if args.check_only:
        print("online artifact gate: ready (2/2 branches)")
        return 0
    render_figure(branches)
    write_results_tex(branches)
    write_protocol_tex(branches)
    write_status(branches, manifest_path)
    print(FIGURE_PATH.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as error:
        raise SystemExit(f"online artifact gate: BLOCKED: {error}")
