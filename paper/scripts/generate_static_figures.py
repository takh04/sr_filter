#!/usr/bin/env python3
"""Regenerate manuscript figures that depend only on checked-in artifacts."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sr_filtering_nqs.artifact_store import (
    load_manifest,
    manifest_files,
    verify_artifact,
)
from sr_filtering_nqs.literature_ratio import plot_literature_ratio
from sr_filtering_nqs.plot_metadata import pdf_metadata
from sr_filtering_nqs.small_scale.plot_four_panel_draft import (
    load_pickle,
    plot_four_panel,
)


PAPER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_ROOT.parent
LITERATURE_INPUT = PAPER_ROOT / "literature_ratio.csv"
LITERATURE_OUTPUT = PAPER_ROOT / "figures" / "p_over_ns_literature.pdf"
MANIFEST_INPUT = REPO_ROOT / "artifacts" / "manifest.json"
MATCHED_NOISE_INPUT = (
    REPO_ROOT / "artifacts" / "small_scale" / "matched_noise" / "rbm490_vit10.pkl"
)
MATCHED_STATE_INPUT = (
    REPO_ROOT
    / "artifacts"
    / "small_scale"
    / "matched_state"
    / "alternating_polish_vit_step0100_to_direct_rbm_a2_figure1_style.pkl"
)
FIGURE2_OUTPUT = PAPER_ROOT / "figures" / "figure2.pdf"
TFIM_SHIFT_INPUT = REPO_ROOT / "artifacts" / "robustness" / "shift_sweep_checkpoints.csv"
FIGURE3_OUTPUT = PAPER_ROOT / "figures" / "figure3_tfim_fixed_data.pdf"
CHECKPOINT_INPUT = REPO_ROOT / "artifacts" / "robustness" / "method_ablation_checkpoints.csv"
FIGURE4_OUTPUT = PAPER_ROOT / "figures" / "figure4_mssr_ablation_oracle.pdf"
TFIM_SHIFT_EXPERIMENT = "robustness.tfim_canonical.shift_sweep.v1"
DIAGNOSTIC_SHIFTS = tuple(10.0**exponent for exponent in range(-8, 0))

METHODS = (
    "single_grid_best_shift",
    "same_batch_multishift_uniform",
    "same_batch_multishift_stacking",
    "bagged_grid_best_shift",
    "independent_multishift_uniform",
    "ms_sr",
)

LABELS = {
    "single_grid_best_shift": "oracle\nsingle\n$\\lambda^*$",
    "same_batch_multishift_uniform": "shared\nuniform",
    "same_batch_multishift_stacking": "shared\nstacked",
    "bagged_grid_best_shift": "oracle\nbagged\n$\\lambda^*$",
    "independent_multishift_uniform": "indep.\nuniform",
    "ms_sr": "MS-SR",
}

REGIMES = {
    "tfim": "tfim_k4_default",
    "j1j2": "j1j2_no_symmetry",
}

SELECTION_LABELS = {
    "single_grid_best_shift": "evaluation_grid_best_same_validation",
    "bagged_grid_best_shift": "evaluation_grid_best_same_validation",
    "same_batch_multishift_uniform": "ntk_quantile_grid",
    "same_batch_multishift_stacking": "ntk_quantile_grid",
    "independent_multishift_uniform": "ntk_quantile_grid",
    "ms_sr": "ntk_quantile_grid",
}


def load_verified_small_scale_inputs() -> tuple[dict, dict]:
    """Verify the two trusted pickle digests before deserializing them."""
    manifest, _ = load_manifest(MANIFEST_INPUT)
    records = {str(record["path"]): record for record in manifest_files(manifest)}
    payloads = []
    for path in (MATCHED_NOISE_INPUT, MATCHED_STATE_INPUT):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative not in records:
            raise ValueError(f"missing manifest record for Figure 2 input: {relative}")
        verify_artifact(path, records[relative])
        payloads.append(load_pickle(path))
    return payloads[0], payloads[1]


def load_checkpoint_summary(
    path: Path, *, system: str
) -> dict[str, dict[str, tuple[float, float]]]:
    """Aggregate the canonical K=4 checkpoint rows without summary inputs."""
    if not path.is_file():
        raise FileNotFoundError(f"missing curated Figure 4 input: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row.get("system") == system
        and row.get("regime") == REGIMES[system]
        and int(row.get("k", "-1")) == 4
        and row.get("quantile_grid_id") == "paper_k4"
        and math.isclose(float(row.get("source_training_shift", "nan")), 1e-4)
    ]
    if not selected:
        raise ValueError(f"{path}: no canonical K=4 rows for {system}")

    rows_by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        rows_by_method[row["method"]].append(row)
    unexpected = sorted(set(rows_by_method) - set(METHODS))
    missing = sorted(set(METHODS) - set(rows_by_method))
    if unexpected or missing:
        raise ValueError(
            f"{path}: {system} method mismatch; missing={missing}, unexpected={unexpected}"
        )

    checkpoint_sets: dict[str, set[int]] = {}
    summary: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for method in METHODS:
        method_rows = rows_by_method[method]
        selection_labels = {row["selection_label"] for row in method_rows}
        if selection_labels != {SELECTION_LABELS[method]}:
            raise ValueError(
                f"{path}: {system}/{method} selection labels are {selection_labels}"
            )
        checkpoints = [int(row["checkpoint_step"]) for row in method_rows]
        if len(checkpoints) != 10 or len(set(checkpoints)) != 10:
            raise ValueError(
                f"{path}: {system}/{method} needs 10 unique checkpoints, "
                f"found {len(checkpoints)}/{len(set(checkpoints))}"
            )
        checkpoint_sets[method] = set(checkpoints)
        for metric in ("r_val_raw", "var_hat_raw"):
            values = [float(row[metric]) for row in method_rows]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}: non-finite {system}/{method}/{metric}")
            # Released summaries use the population standard deviation across
            # checkpoint aggregates. Preserve that convention while deriving
            # the plotted SE directly from checkpoint-level rows.
            standard_error = statistics.pstdev(values) / math.sqrt(len(values))
            summary[method][metric] = (statistics.fmean(values), standard_error)
    if len({frozenset(steps) for steps in checkpoint_sets.values()}) != 1:
        raise ValueError(f"{path}: {system} methods do not share checkpoint support")
    return summary


def load_tfim_shift_summary(
    path: Path,
) -> dict[float, dict[str, tuple[float, float]]]:
    """Aggregate the canonical TFIM sweep from checkpoint-level records."""
    if not path.is_file():
        raise FileNotFoundError(f"missing curated Figure 3 input: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("experiment_id") == TFIM_SHIFT_EXPERIMENT
        ]
    if len(rows) != 80:
        raise ValueError(f"{path}: expected 80 canonical TFIM rows, found {len(rows)}")

    grouped: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("system") != "tfim"
            or row.get("regime") != "tfim_canonical"
            or not math.isclose(float(row["source_training_shift"]), 1e-4)
            or int(row["repeat_count"]) != 20
            or int(row["n_validation_samples"]) != 120_000
        ):
            raise ValueError(f"{path}: canonical TFIM protocol drift in {row}")
        grouped[float(row["solver_shift"])].append(row)
    if set(grouped) != set(DIAGNOSTIC_SHIFTS):
        raise ValueError(f"{path}: canonical TFIM shift-grid mismatch")

    checkpoint_sets: set[frozenset[int]] = set()
    summary: dict[float, dict[str, tuple[float, float]]] = defaultdict(dict)
    for shift in DIAGNOSTIC_SHIFTS:
        shift_rows = grouped[shift]
        checkpoints = frozenset(int(row["checkpoint_step"]) for row in shift_rows)
        if checkpoints != frozenset(range(1100, 2001, 100)):
            raise ValueError(f"{path}: canonical TFIM checkpoint mismatch at {shift:g}")
        checkpoint_sets.add(checkpoints)
        for metric in ("r_val_raw", "var_hat_raw"):
            values = [float(row[metric]) for row in shift_rows]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}: non-finite canonical TFIM {metric}")
            standard_error = statistics.pstdev(values) / math.sqrt(len(values))
            summary[shift][metric] = (statistics.fmean(values), standard_error)
    if len(checkpoint_sets) != 1:
        raise ValueError(f"{path}: shifts do not share checkpoint support")
    return summary


def render_tfim_shift_sweep(
    summary: dict[float, dict[str, tuple[float, float]]],
) -> None:
    shifts = np.asarray(DIAGNOSTIC_SHIFTS, dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.55))
    panels = (
        ("r_val_raw", r"$\widehat{\mathcal{E}}_{\mathrm{val}}$", "#24639b"),
        ("var_hat_raw", r"$\mathcal{V}_{\mathrm{mb}}$", "#93432f"),
    )
    for ax, (metric, ylabel, color) in zip(axes, panels, strict=True):
        means = np.asarray([summary[shift][metric][0] for shift in DIAGNOSTIC_SHIFTS])
        errors = np.asarray([summary[shift][metric][1] for shift in DIAGNOSTIC_SHIFTS])
        ax.errorbar(
            shifts,
            means,
            yerr=errors,
            marker="o",
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=1.1,
            color=color,
            linewidth=1.6,
            elinewidth=0.9,
            capsize=2.4,
        )
        ax.set_xscale("log")
        ax.set_xticks(shifts)
        ax.set_xticklabels([f"1e{int(np.log10(shift))}" for shift in shifts], rotation=35)
        ax.set_xlabel(r"Solve-time $\lambda$")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#dddddd", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    fig.subplots_adjust(left=0.085, right=0.992, bottom=0.29, top=0.96, wspace=0.34)
    fig.savefig(
        FIGURE3_OUTPUT,
        bbox_inches="tight",
        metadata=pdf_metadata("TFIM fixed-checkpoint shift sweep"),
    )
    plt.close(fig)


def render_panel(ax: plt.Axes, summary: dict[str, dict[str, tuple[float, float]]], title: str) -> None:
    x = np.arange(len(METHODS), dtype=float)
    width = 0.34
    risk = np.asarray([summary[method]["r_val_raw"][0] for method in METHODS])
    risk_se = np.asarray([summary[method]["r_val_raw"][1] for method in METHODS])
    variance = np.asarray([summary[method]["var_hat_raw"][0] for method in METHODS])
    variance_se = np.asarray([summary[method]["var_hat_raw"][1] for method in METHODS])

    ax.bar(
        x - width / 2,
        risk,
        width,
        yerr=risk_se,
        color="#eeeeee",
        edgecolor="#9a9a9a",
        linewidth=0.45,
        error_kw={"elinewidth": 0.8, "capsize": 2.2, "capthick": 0.8},
        label=r"$\widehat{\mathcal{E}}_{\mathrm{val}}$",
    )
    ax.bar(
        x + width / 2,
        variance,
        width,
        yerr=variance_se,
        color="#a9a9a9",
        edgecolor="#555555",
        linewidth=0.45,
        hatch="////",
        error_kw={"elinewidth": 0.8, "capsize": 2.2, "capthick": 0.8},
        label=r"$\mathcal{V}_{\mathrm{mb}}$",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[method] for method in METHODS], fontsize=6.2)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", color="#dddddd", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.set_ylabel("diagnostic value", fontsize=8)


def main() -> None:
    baseline_rc = matplotlib.rcParams.copy()
    plot_literature_ratio(LITERATURE_INPUT, LITERATURE_OUTPUT)
    matplotlib.rcParams.update(baseline_rc)

    matched_noise, matched_state = load_verified_small_scale_inputs()
    plot_four_panel(
        matched_noise=matched_noise,
        matched_state=matched_state,
        output_path=FIGURE2_OUTPUT,
    )
    matplotlib.rcParams.update(baseline_rc)

    tfim_shift = load_tfim_shift_summary(TFIM_SHIFT_INPUT)
    render_tfim_shift_sweep(tfim_shift)
    matplotlib.rcParams.update(baseline_rc)

    tfim = load_checkpoint_summary(CHECKPOINT_INPUT, system="tfim")
    j1j2 = load_checkpoint_summary(CHECKPOINT_INPUT, system="j1j2")

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.75))
    render_panel(axes[0], tfim, r"TFIM FNQS ($L=100$)")
    render_panel(axes[1], j1j2, r"$J_1$-$J_2$ ($8\times8$)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.29, top=0.79, wspace=0.28)
    FIGURE4_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE4_OUTPUT,
        bbox_inches="tight",
        metadata=pdf_metadata("Checkpoint-local MS-SR ablations"),
    )
    plt.close(fig)
    matplotlib.rcParams.update(baseline_rc)
    print(LITERATURE_OUTPUT.relative_to(REPO_ROOT))
    print(FIGURE2_OUTPUT.relative_to(REPO_ROOT))
    print(FIGURE3_OUTPUT.relative_to(REPO_ROOT))
    print(FIGURE4_OUTPUT.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
