"""Regenerate both appendix figures from the preserved paper data on a CPU."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sr_filtering_nqs.paper_figures import (
    DEFAULT_ARTIFACT_ROOT,
    PDF_METADATA,
    _mean_sem,
    _read_csv,
    resolve_artifact_root,
    validate_artifacts,
)


def _bagged_shifts(root: Path, output: Path) -> Path:
    sweep = [r for r in _read_csv(root / "figure4/full_sweep.csv") if r["system"] == "tfim"]
    refs = [
        r
        for r in _read_csv(root / "figure4/checkpoint_values.csv")
        if r["system"] == "tfim" and r["method"] == "indep_multilambda_stacking_k4"
    ]
    shifts = sorted({float(r["lambda"]) for r in sweep})
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.65))
    for ax, metric, title, label in zip(
        axes,
        ("r_val_raw", "var_hat_raw"),
        ("(a) Held-out residual", "(b) Update variance"),
        (r"$\widehat{\mathcal{E}}_{\mathrm{val}}$", r"$\mathcal{V}_{\mathrm{mb}}$"),
        strict=True,
    ):
        values = [
            _mean_sem([float(r[metric]) for r in sweep if float(r["lambda"]) == s]) for s in shifts
        ]
        mean, sem = np.asarray(values).T
        reference, uncertainty = _mean_sem([float(r[metric]) for r in refs])
        ax.axhspan(
            reference - uncertainty,
            reference + uncertainty,
            color="#d95f02",
            alpha=0.13,
            linewidth=0,
        )
        ax.axhline(reference, color="#d95f02", linestyle="--", label="MS-SR")
        ax.errorbar(
            shifts,
            mean,
            yerr=sem,
            color="#2166ac",
            marker="o",
            markersize=3.5,
            capsize=2.5,
            label="Bagged SR ($K=4$)",
        )
        ax.set(xscale="log", xlabel=r"Fixed bagging shift $\lambda$", ylabel=label)
        ax.set_title(title, loc="left")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
        ax.grid(axis="y", alpha=0.18)
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=2)
    path = output / "bagged_shift_robustness.pdf"
    fig.savefig(path, bbox_inches="tight", metadata=PDF_METADATA)
    plt.close(fig)
    return path


def _tracking(root: Path, output: Path) -> Path:
    points = _read_csv(root / "appendix/online_tracking/points.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharey=True, layout="constrained")
    models = ("rbm_a2", "rbm_a4", "vit_small")
    values = [float(r["log10_ratio"]) for r in points]
    ymin, ymax = min(min(values) - 0.08, -0.08), max(max(values) + 0.08, 0.08)
    for ax, system, title in zip(
        axes, ("tfim2d", "j1j2", "ssm"), ("TFIM", r"$J_1$–$J_2$", "SSM"), strict=True
    ):
        data = [
            [
                float(r["log10_ratio"])
                for r in points
                if r["system"] == system and r["model"] == model
            ]
            for model in models
        ]
        boxes = ax.boxplot(
            data,
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.45},
        )
        for patch, color, hatch in zip(
            boxes["boxes"], ("white", "#D9EAD3", "#D9D2E9"), ("", "///", ".."), strict=True
        ):
            patch.set(facecolor=color, hatch=hatch)
        ax.axhline(0, color="#333333", linestyle="--", linewidth=0.9)
        ax.axhspan(ymin, 0, color="#EAF3EA", alpha=0.45, zorder=0)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks([1, 2, 3], ["RBM\n" + r"($\alpha=2$)", "RBM\n" + r"($\alpha=4$)", "ViT"])
        ax.set_ylim(ymin, ymax)
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel(r"$\Delta_{\log r}$")
    path = output / "figure5_srel_logratio_hamiltonian_boxplots.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, metadata=PDF_METADATA)
    plt.close(fig)
    return path


def reproduce_appendix(input_root=DEFAULT_ARTIFACT_ROOT, output_dir="outputs/appendix"):
    root = resolve_artifact_root(input_root)
    validate_artifacts(root)
    output = Path(output_dir).resolve()
    if output.is_relative_to(root):
        raise ValueError("Write figures outside the immutable artifact bundle")
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42}):
        return [_bagged_shifts(root, output), _tracking(root, output)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/appendix"))
    args = parser.parse_args()
    for path in reproduce_appendix(args.artifact_root, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
