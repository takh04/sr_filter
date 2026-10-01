"""Render the camera-ready main-text figures from compact, safe local artifacts.

The bundle contains derived diagnostics, not model checkpoints. Rendering needs
NumPy, SciPy, and Matplotlib only and performs no downloads or training. Figure 5
is assembled in the manuscript from its curve and two generated TeX tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import t

PDF_METADATA = {"CreationDate": None, "ModDate": None}
DEFAULT_ARTIFACT_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "paper"


def resolve_artifact_root(input_root: str | Path) -> Path:
    root = Path(input_root).resolve()
    for candidate in (root, root / "paper", root / "artifacts" / "paper"):
        if (candidate / "manifest.json").is_file() and (candidate / "figure1").is_dir():
            return candidate
    raise FileNotFoundError(f"No compact paper bundle found below {root}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def validate_artifacts(input_root: str | Path) -> dict:
    """Check every bundled file against its manifest before rendering."""
    root = resolve_artifact_root(input_root)
    manifest = json.loads((root / "manifest.json").read_text())
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe artifact path: {relative}")
        path = root / relative
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Artifact escapes bundle: {relative}")
        data = path.read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {relative}")
    return manifest


def _annotate_point(ax, point):
    label = point["label"]
    dx, dy = 0, 8
    ha = "left"
    va = "bottom"

    if label == "Carleo":
        dx, dy = 12, -4
        va = "top"
    elif label == "Choo":
        dx, dy = 8, 4
    elif label == "Zen":
        dx, dy = 10, -4
        va = "top"
    elif label == "Saito":
        dx, dy = 10, 8
    elif label == "Hibat-Allah":
        dx, dy = 10, 0
    elif label == "Pfau":
        dx, dy = 10, 1
    elif label == "Gutierrez":
        dx, dy = 8, -7
        va = "top"
    elif label == "von Glehn":
        dx, dy = -30, 7
        ha = "right"
    elif label == "Rende":
        dx, dy = -26, -8
        ha = "right"
        va = "top"
    elif label == "Chen":
        dx, dy = 14, 3
    elif label == "Sprague":
        dx, dy = -30, -3
        ha = "right"
        va = "top"

    ax.annotate(
        label,
        xy=(point["year"], point["ratio"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=6.4,
        ha=ha,
        va=va,
        arrowprops=dict(
            arrowstyle="-",
            color="#71717a",
            linewidth=0.55,
            shrinkA=2,
            shrinkB=4,
        ),
    )


def _draw_group_labels(ax, points, group_labels):
    groups = {}
    for point in points:
        if "group" in point:
            groups.setdefault(point["group"], []).append(point)

    for group, points in groups.items():
        label = group_labels[group]
        for point in points:
            ax.plot(
                [point["year"], label["x"]],
                [point["ratio"], label["y"]],
                color="#71717a",
                linewidth=0.65,
                alpha=0.65,
                zorder=2.5,
            )

        ax.text(
            label["x"],
            label["y"],
            group,
            fontsize=6.4,
            ha=label["ha"],
            va=label["va"],
            zorder=5,
        )


def _plot_literature(root: Path, output_dir: Path) -> list[Path]:
    payload = json.loads((root / "figure1/literature_points.json").read_text())
    points, styles, group_labels = (payload[key] for key in ("POINTS", "STYLES", "GROUP_LABELS"))

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(5.4, 3.15), constrained_layout=True)

    ax.axhspan(5e-3, 1, color="#f4f4f5", zorder=0)
    ax.axhspan(1, 5e3, color="#fff7ed", alpha=0.55, zorder=0)
    _draw_group_labels(ax, points, group_labels)

    for point in points:
        style = styles[point["kind"]]

        ax.scatter(
            point["year"],
            point["ratio"],
            s=54,
            marker=style["marker"],
            facecolor=style["color"],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="",
            marker=style["marker"],
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=6,
            label=kind,
        )
        for kind, style in styles.items()
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        fontsize=6.8,
        borderpad=0.2,
        handletextpad=0.4,
        labelspacing=0.3,
    )

    ax.axhline(1, color="#27272a", linewidth=1.0, linestyle="--", zorder=1)
    ax.text(
        2020.8,
        1.25,
        r"$P=N_s$ interpolation threshold",
        color="#27272a",
        fontsize=7.4,
        va="bottom",
    )

    ax.text(
        2026.25,
        0.72,
        "sample-rich",
        fontsize=6.8,
        color="#52525b",
        ha="right",
        va="top",
    )
    ax.text(
        2026.25,
        1.38,
        "parameter-rich",
        fontsize=6.8,
        color="#9a3412",
        ha="right",
        va="bottom",
    )

    for point in points:
        if "group" not in point and point.get("annotate", True) and point["label"]:
            _annotate_point(ax, point)

    ax.set_yscale("log")
    ax.set_xlim(2016.75, 2026.4)
    ax.set_ylim(5e-3, 5e3)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"Parameter-to-sample ratio $P/N_s$")
    ax.set_xticks([2017, 2020, 2022, 2024, 2026])
    ax.grid(True, which="major", axis="y", linewidth=0.5, color="#d4d4d8")
    ax.grid(True, which="minor", axis="y", linewidth=0.25, color="#e4e4e7")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = output_dir / "p_over_ns_literature.pdf"
    fig.savefig(path, dpi=300, metadata=PDF_METADATA)
    plt.close(fig)
    return [path]


METHOD_ORDER = (
    "single_train",
    "same_batch_multilambda_uniform_k4",
    "same_batch_multilambda_stacking_k4",
    "bagged_train",
    "bagged_tuned",
    "indep_multilambda_uniform_k4",
    "indep_multilambda_stacking_k4",
)

LABELS = {
    "single_train": "SR\n" + r"$\lambda_{\mathrm{train}}$",
    "same_batch_multilambda_uniform_k4": "shared-\nbatch\nuniform",
    "same_batch_multilambda_stacking_k4": "shared-\nbatch\nstacked",
    "bagged_train": "bagged\nSR\n" + r"$\lambda_{\mathrm{train}}$",
    "bagged_tuned": "bagged\nSR\n" + r"$\lambda_{\mathrm{tuned}}$",
    "indep_multilambda_uniform_k4": "indep.\nbatch\nuniform",
    "indep_multilambda_stacking_k4": "MS-SR",
}

METRICS = ("r_val_raw", "var_hat_raw")


def _render_ablation(summary: list[dict], output_dir: Path) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "hatch.linewidth": 0.45,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.6))
    fig.subplots_adjust(left=0.077, right=0.995, bottom=0.225, top=0.79, wspace=0.265)
    for ax, system, title in zip(
        axes,
        ("tfim", "j1j2"),
        (r"TFIM FNQS ($L=100$)", r"$J_1$-$J_2$ (8$\times$8)"),
        strict=True,
    ):
        methods = list(METHOD_ORDER)
        x = np.arange(len(methods))
        width = 0.34
        for index, (metric, label, color, hatch) in enumerate(
            (
                ("r_val_raw", r"$\hat{\mathcal{E}}_{\mathrm{val}}$", "#eaeaea", None),
                ("var_hat_raw", r"$\mathcal{V}_{\mathrm{mb}}$", "#9c9c9c", "////"),
            )
        ):
            rows = [
                next(
                    row
                    for row in summary
                    if row["system"] == system
                    and row["method"] == method
                    and row["metric"] == metric
                )
                for method in methods
            ]
            ax.bar(
                x + (index - 0.5) * width,
                [row["mean"] for row in rows],
                width=width,
                yerr=[row["sem"] for row in rows]
                if all(row["sem"] is not None for row in rows)
                else None,
                color=color,
                edgecolor="#555555",
                linewidth=0.3,
                hatch=hatch,
                label=label,
                error_kw={"elinewidth": 0.7, "capsize": 1.8, "capthick": 0.65},
            )
        ax.set_title(title, pad=9)
        ax.set_xticks(x, [LABELS[method] for method in methods])
        ax.tick_params(axis="x", length=0, pad=7)
        ax.tick_params(axis="y", width=0.65, length=3)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_size(7)
        ax.set_xlim(-0.6, len(methods) - 0.4)
        ax.set_ylim(bottom=0)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#d7d7d7", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("diagnostics value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.985),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=2.5,
    )
    path = output_dir / "Figure4_linear.pdf"
    fig.savefig(path, metadata=PDF_METADATA)
    plt.close(fig)
    return [path]


def _plot_amplitude(root: Path, output_dir: Path) -> list[Path]:
    from sr_filtering_nqs.small_scale.plot_four_panel_draft import plot_four_panel

    metadata = json.loads((root / "figure2/metadata.json").read_text())
    experiments = {"matched_noise": {"models": []}, "matched_state": {"models": []}}
    experiments["matched_state"]["matching"] = {"matched_fidelity": metadata["matched_fidelity"]}
    # allow_pickle=False is deliberate: the public bundle has plain numerical arrays.
    with np.load(root / "figure2/amplitude_diagnostics.npz", allow_pickle=False) as arrays:
        for index, panel in enumerate(metadata["panels"]):
            prefix = f"panel{index}_"
            risk_samples = arrays[prefix + "risk_samples"]
            model = {
                "model_desc": panel["model_desc"],
                "P_over_Ns": panel["p_over_ns"],
                "sigma_gap_sq": panel["sigma_gap_sq"],
                "noise_fraction": panel["noise_fraction"],
                "selected_checkpoint": {
                    "target_ideal": arrays[prefix + "target"],
                    "pi": arrays[prefix + "pi"],
                },
                "representative": {
                    "predictions": {"pred_small": arrays[prefix + "prediction"]},
                    "train_indices": arrays[prefix + "train_indices"],
                    "metrics": {
                        "r_train_hloc": [panel["train_residual"]],
                        "r_pop_ideal": [panel["representative_excess_risk"]],
                    },
                },
                "summary": {
                    "lambda_small_idx": 0,
                    "lambda_small": panel["lambda"],
                    "n_trials": panel["n_trials"],
                    "r_pop_ideal_mean": [panel["excess_risk"]],
                    "r_pop_ideal_stderr": [panel["excess_risk_standard_error"]],
                },
                "trials": [{"metrics": {"r_pop_ideal": [risk]}} for risk in risk_samples],
            }
            experiments[panel["protocol"]]["models"].append(model)
        path = output_dir / "figure2.pdf"
        plot_four_panel(
            matched_noise=experiments["matched_noise"],
            matched_state=experiments["matched_state"],
            output_path=path,
        )
    return [path]


def _mean_sem(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size != 10 or not np.isfinite(array).all():
        raise ValueError("Expected ten finite checkpoint diagnostics")
    # Retain the paper's checkpoint SD convention (ddof=0) / sqrt(10).
    return float(array.mean()), float(array.std(ddof=0) / math.sqrt(array.size))


def _plot_shift_sweep(root: Path, output_dir: Path) -> list[Path]:
    rows = _read_csv(root / "figure3/checkpoint_metrics.csv")
    shifts = sorted({float(row["lambda"]) for row in rows})
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.65))
    for ax, metric, label, color in zip(
        axes,
        METRICS,
        (r"$\widehat{\mathcal{E}}_{\mathrm{val}}$", r"$\mathcal{V}_{\mathrm{mb}}$"),
        ("#2166ac", "#a13c25"),
        strict=True,
    ):
        stats = [
            _mean_sem([float(row[metric]) for row in rows if float(row["lambda"]) == shift])
            for shift in shifts
        ]
        means, sems = np.asarray(stats).T
        x = np.arange(len(shifts))
        ax.errorbar(
            x,
            means,
            yerr=sems,
            marker="o",
            markersize=4,
            color=color,
            markerfacecolor="white",
            linewidth=1.3,
            capsize=2,
        )
        ax.set_xticks(x, [f"1e{int(round(math.log10(shift)))}" for shift in shifts])
        ax.tick_params(axis="x", labelrotation=35, labelsize=8)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.set_xlabel(r"Solve-time $\lambda$")
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.85", linewidth=0.6)
    fig.tight_layout(w_pad=3.2)
    path = output_dir / "figure3_tfim_fixed_data.pdf"
    fig.savefig(path, metadata=PDF_METADATA)
    plt.close(fig)
    return [path]


def _plot_ablation(root: Path, output_dir: Path) -> list[Path]:
    rows = _read_csv(root / "figure4/checkpoint_values.csv")
    summary = []
    for system in ("tfim", "j1j2"):
        for method in METHOD_ORDER:
            selected = [row for row in rows if row["system"] == system and row["method"] == method]
            if len({row["step"] for row in selected}) != 10:
                raise ValueError(f"Expected ten distinct checkpoints for {system}/{method}")
            for metric in METRICS:
                mean, sem = _mean_sem([float(row[metric]) for row in selected])
                summary.append(
                    {"system": system, "method": method, "metric": metric, "mean": mean, "sem": sem}
                )
    return _render_ablation(summary, output_dir)


def online_summary(root: str | Path) -> dict:
    """Return independently evaluated endpoint statistics for the five displayed pairs."""
    root = resolve_artifact_root(root)
    endpoints = _read_csv(root / "figure5/endpoint_pairs.csv")
    endpoints.sort(key=lambda row: int(row["branch_rng_seed"]))
    if [int(row["branch_rng_seed"]) for row in endpoints] != list(range(104, 109)):
        raise ValueError("Expected all five source2500 paired continuations")
    delta = np.array([float(row["delta_energy_historical_ms_sr_minus_sr"]) for row in endpoints])
    half_ci = t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / math.sqrt(len(delta))
    return {
        "pair_count": len(delta),
        "ms_sr_lower_count": int((delta < 0).sum()),
        "mean_delta_energy_per_site": float(delta.mean()),
        "paired_t_95_interval": [float(delta.mean() - half_ci), float(delta.mean() + half_ci)],
    }


def _write_endpoint_table(endpoints: list[dict], path: Path) -> None:
    table = [
        r"\setlength{\tabcolsep}{3pt}",
        r"\footnotesize",
        r"\begin{tabular}{@{}rrrr@{}}",
        r"\toprule",
        r"Pair & SR & MS-SR & $\Delta\;(10^{-5})$ \\",
        r"\midrule",
    ]
    for index, row in enumerate(endpoints, start=1):
        sr = float(row["sr_energy_per_site"])
        ms = float(row["historical_ms_sr_energy_per_site"])
        difference = float(row["delta_energy_historical_ms_sr_minus_sr"])
        se = math.hypot(float(row["sr_energy_mc_se"]), float(row["historical_ms_sr_energy_mc_se"]))
        sr_text, ms_text = f"{sr:.7f}", f"{ms:.7f}"
        if difference < 0:
            ms_text = r"\mathbf{" + ms_text + "}"
        else:
            sr_text = r"\mathbf{" + sr_text + "}"
        table.append(
            f"{index} & ${sr_text}$ & ${ms_text}$ & "
            f"${difference * 1e5:+.2f} \\pm {se * 1e5:.2f}$ " + r"\\"
        )
    sr_mean = np.mean([float(row["sr_energy_per_site"]) for row in endpoints])
    ms_mean = np.mean([float(row["historical_ms_sr_energy_per_site"]) for row in endpoints])
    delta = np.mean([float(row["delta_energy_historical_ms_sr_minus_sr"]) for row in endpoints])
    table.extend(
        [
            r"\midrule",
            f"Mean & ${sr_mean:.7f}$ & ${ms_mean:.7f}$ & ${delta * 1e5:+.2f}$ " + r"\\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    path.write_text("\n".join(table) + "\n")


def _write_timing_table(root: Path, path: Path) -> None:
    pairs = _read_csv(root / "figure5/paired_timing.csv")
    medians = [
        float(np.median([float(row[key]) for row in pairs]))
        for key in ("sr_median_s", "bagged_sr_median_s", "ms_sr_median_s")
    ]
    lines = [
        r"{\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"    \toprule",
        r"    Method & \shortstack{Batches/\\solves} & \shortstack{Time\\(s)} "
        r"& \shortstack{vs.\\SR} \\",
        r"    \midrule",
    ]
    for method, cost, value in zip(("SR", "Bagged SR", "MS-SR"), (1, 4, 4), medians):
        lines.append(
            f"    {method} & {cost}/{cost} & {value:.2f} & "
            f"${value / medians[0]:.2f}\\times$ " + r"\\"
        )
    lines.extend([r"    \bottomrule", r"\end{tabular}}"])
    path.write_text("\n".join(lines) + "\n")


def _plot_online(root: Path, output_dir: Path) -> list[Path]:
    training = _read_csv(root / "figure5/training_curves.csv")
    endpoints = _read_csv(root / "figure5/endpoint_pairs.csv")
    endpoints.sort(key=lambda row: int(row["branch_rng_seed"]))
    online_summary(root)
    endpoint_path = output_dir / "historical_online_endpoints.tex"
    timing_path = output_dir / "figure5_timing.tex"
    _write_endpoint_table(endpoints, endpoint_path)
    _write_timing_table(root, timing_path)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(3.2, 2.75))
    initial_values = {}
    bin_edges = np.arange(0, 501, 20)
    for method, color, nominal_cost, updates, endpoint_key in (
        ("SR", "#2166ac", 1, 500, "sr_energy_per_site"),
        ("MS-SR", "#d95f02", 4, 125, "historical_ms_sr_energy_per_site"),
    ):
        rows = [row for row in training if row["method"] == method]
        seeds = sorted({int(row["branch_seed"]) for row in rows})
        if seeds != list(range(104, 109)):
            raise ValueError(f"Incomplete training pairs for {method}")
        positions = nominal_cost * np.arange(updates)
        per_seed, initial = [], []
        for seed in seeds:
            records = sorted(
                [row for row in rows if int(row["branch_seed"]) == seed],
                key=lambda row: int(row["update"]),
            )
            if [int(row["update"]) for row in records] != list(range(1, updates + 1)):
                raise ValueError(f"Incomplete {method} trace for seed {seed}")
            values = np.array([float(row["energy_per_site"]) for row in records])
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite training energy")
            initial.append(values[0])
            per_seed.append(
                [
                    values[(positions >= left) & (positions < right) & (positions > 0)].mean()
                    for left, right in zip(bin_edges[:-1], bin_edges[1:])
                ]
            )
        initial_values[method] = np.array(initial)
        samples = np.stack(per_seed)
        mean, sd = samples.mean(axis=0), samples.std(axis=0, ddof=1)
        x = (bin_edges[:-1] + bin_edges[1:]) / 2
        final_mean = np.mean([float(row[endpoint_key]) for row in endpoints])
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.16, linewidth=0)
        ax.plot(
            np.r_[0, x, 500],
            np.r_[np.mean(initial), mean, final_mean],
            color=color,
            linewidth=1.3,
            label=method,
        )
        ax.plot(500, final_mean, "D", color=color, markersize=3.8, clip_on=False)
    np.testing.assert_array_equal(initial_values["SR"], initial_values["MS-SR"])
    ax.plot(0, initial_values["SR"].mean(), "o", color="0.2", markersize=3.5, clip_on=False)
    ax.set_title("Paired continuations ($n=5$)", loc="left", pad=8)
    ax.set_xlabel("SR-equivalent updates")
    ax.set_ylabel("Energy / site")
    ax.set_xlim(0, 500)
    ax.set_xticks([0, 250, 500])
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    ax.legend(frameon=False, loc="upper right", handlelength=1.3)
    fig.tight_layout(w_pad=1.1, pad=0.55)
    path = output_dir / "historical_online_cholesky.pdf"
    fig.savefig(path, bbox_inches="tight", metadata=PDF_METADATA)
    plt.close(fig)
    return [path, endpoint_path, timing_path]


def reproduce_paper(
    input_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    output_dir: str | Path = "outputs/paper",
    figures: list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    """Regenerate selected figure inputs, with no training or external data required.

    ``input_root`` may be a checkout, its artifacts directory, or the paper
    bundle itself. Figure identifiers are the main-text numbers ``1`` to ``5``.
    Existing output files with the corresponding figure names are replaced.
    """
    root = resolve_artifact_root(input_root)
    validate_artifacts(root)
    selected = list(figures) if figures is not None else list("12345")
    if any(str(figure) not in "12345" or len(str(figure)) != 1 for figure in selected):
        raise ValueError("Figure identifiers must be 1, 2, 3, 4, or 5")
    output_dir = Path(output_dir)
    if output_dir.resolve() == root or root in output_dir.resolve().parents:
        raise ValueError("Write figure outputs outside the immutable artifact bundle")
    output_dir.mkdir(parents=True, exist_ok=True)
    renderers = {
        "1": _plot_literature,
        "2": _plot_amplitude,
        "3": _plot_shift_sweep,
        "4": _plot_ablation,
        "5": _plot_online,
    }
    paths = []
    for figure in dict.fromkeys(map(str, selected)):
        with plt.rc_context():
            paths.extend(renderers[figure](root, output_dir))
    return paths


def reproduce_paper_figures(
    output_dir: str | Path,
    artifact_root: str | Path | None = None,
) -> dict[str, Path]:
    """Convenience wrapper keyed by output filename."""
    return {
        path.name: path
        for path in reproduce_paper(
            artifact_root if artifact_root is not None else DEFAULT_ARTIFACT_ROOT, output_dir
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--figures", nargs="+", choices=list("12345"))
    args = parser.parse_args()
    for path in reproduce_paper(args.artifact_root, args.output_dir, args.figures):
        print(path)


if __name__ == "__main__":
    main()
