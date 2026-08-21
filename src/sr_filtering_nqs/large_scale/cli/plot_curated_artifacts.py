from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sr_filtering_nqs.artifact_schema import normalize_artifact_record
from sr_filtering_nqs.plot_metadata import pdf_metadata


DEFAULT_ARTIFACT_ROOT = Path("artifacts")
DEFAULT_OUTPUT_DIR = Path("outputs")

METHOD_ORDER = (
    "single_grid_best_shift",
    "same_batch_multishift_uniform",
    "same_batch_multishift_stacking",
    "bagged_grid_best_shift",
    "independent_multishift_uniform",
    "ms_sr",
)

METHOD_LABELS = {
    "single_grid_best_shift": "single\nbest fixed",
    "same_batch_multishift_uniform": "same-batch\nmulti-lambda\nuniform",
    "same_batch_multishift_stacking": "same-batch\nmulti-lambda\nstacking",
    "bagged_grid_best_shift": "bagged\nsingle-lambda",
    "independent_multishift_uniform": "indep\nmulti-lambda\nuniform",
    "ms_sr": "MS-SR",
}

METHOD_COLORS = {
    "single_grid_best_shift": "#1f4e79",
    "same_batch_multishift_uniform": "#5b8c5a",
    "same_batch_multishift_stacking": "#89632a",
    "bagged_grid_best_shift": "#5f7ca1",
    "independent_multishift_uniform": "#a05b7f",
    "ms_sr": "#8d2d5a",
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), 0.0
    return float(np.mean(array)), float(np.std(array, ddof=1))


def plot_tfim_shift_sweep(artifact_root: Path, output_dir: Path) -> Path:
    rows = _read_rows(artifact_root / "large_scale/tfim_shared_shift_sweep/summary.csv")
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["subset"]][float(row["lambda"])].append(float(row["mean"]))

    fig, ax = plt.subplots(figsize=(5.8, 3.9), constrained_layout=True)
    colors = {"early5": "#6a7f3f", "late5": "#1f4e79"}
    labels = {"early5": "early checkpoints", "late5": "late checkpoints"}
    for subset in ("early5", "late5"):
        lambdas = np.asarray(sorted(grouped.get(subset, {})), dtype=np.float64)
        if lambdas.size == 0:
            continue
        stats = [_mean_std(grouped[subset][float(lam)]) for lam in lambdas]
        means = np.asarray([item[0] for item in stats], dtype=np.float64)
        stds = np.asarray([item[1] for item in stats], dtype=np.float64)
        ax.plot(lambdas, means, marker="o", linewidth=1.9, color=colors[subset], label=labels[subset])
        ax.fill_between(
            lambdas,
            np.maximum(means - stds, 1e-30),
            means + stds,
            color=colors[subset],
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Solve-time $\lambda$")
    ax.set_ylabel(r"Validation risk")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "figure3_tfim_shift_sweep_from_csv.pdf"
    fig.savefig(output_path, metadata=pdf_metadata("TFIM shift-sweep diagnostics"))
    plt.close(fig)
    return output_path


def _load_ms_sr_summary(path: Path) -> dict[str, dict[str, tuple[float, float]]]:
    rows = _read_rows(path)
    summary: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for row in rows:
        normalized = normalize_artifact_record(row)
        summary[normalized["method"]][normalized["metric"]] = (
            normalized["value"],
            float(row["std"]),
        )
    return summary


def _plot_ms_sr_metric(
    ax: plt.Axes,
    *,
    summary: dict[str, dict[str, tuple[float, float]]],
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    x = np.arange(len(METHOD_ORDER), dtype=np.float64)
    means = np.asarray([summary[method][metric][0] for method in METHOD_ORDER], dtype=np.float64)
    stds = np.asarray([summary[method][metric][1] for method in METHOD_ORDER], dtype=np.float64)
    ax.bar(x, means, color=[METHOD_COLORS[method] for method in METHOD_ORDER], alpha=0.92)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black", elinewidth=1.0, capsize=3.0)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[method] for method in METHOD_ORDER], fontsize=7.4)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.2)
    finite = means[np.isfinite(means) & (means > 0.0)]
    if finite.size:
        ax.set_yscale("log")


def plot_ms_sr_ablation(artifact_root: Path, output_dir: Path) -> Path:
    tfim = _load_ms_sr_summary(artifact_root / "large_scale/ms_sr_ablation/tfim_summary.csv")
    j1j2 = _load_ms_sr_summary(artifact_root / "large_scale/ms_sr_ablation/j1j2_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.6), constrained_layout=True)

    _plot_ms_sr_metric(
        axes[0, 0],
        summary=tfim,
        metric="r_val_raw",
        title="TFIM validation risk",
        ylabel=r"$R_{\mathrm{val}}$",
    )
    _plot_ms_sr_metric(
        axes[0, 1],
        summary=tfim,
        metric="var_hat_raw",
        title="TFIM variance estimate",
        ylabel=r"$V_{\mathrm{mb}}$",
    )
    _plot_ms_sr_metric(
        axes[1, 0],
        summary=j1j2,
        metric="r_val_raw",
        title="J1-J2 validation risk",
        ylabel=r"$R_{\mathrm{val}}$",
    )
    _plot_ms_sr_metric(
        axes[1, 1],
        summary=j1j2,
        metric="var_hat_raw",
        title="J1-J2 variance estimate",
        ylabel=r"$V_{\mathrm{mb}}$",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "figure4_ms_sr_ablation_from_csv.pdf"
    fig.savefig(output_path, metadata=pdf_metadata("MS-SR checkpoint-local ablation"))
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot figures from bundled compact CSV artifacts.")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure", choices=("all", "tfim-shift", "ms-sr"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root)
    output_dir = Path(args.output_dir)
    outputs: list[Path] = []
    if args.figure in ("all", "tfim-shift"):
        outputs.append(plot_tfim_shift_sweep(artifact_root, output_dir))
    if args.figure in ("all", "ms-sr"):
        outputs.append(plot_ms_sr_ablation(artifact_root, output_dir))
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
