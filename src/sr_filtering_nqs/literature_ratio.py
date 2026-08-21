# ruff: noqa: I001
"""Audited parameter-to-sample literature plot."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sr_filtering_nqs.plot_metadata import pdf_metadata


DOMAINS = {
    "spin": {
        "label": "Spin/lattice NQS",
        "marker": "o",
        "color": "#2a6f97",
    },
    "electronic": {
        "label": "Electronic-structure NQS",
        "marker": "D",
        "color": "#15803d",
    },
}
EVIDENCE_CLASSES = {"explicit", "derived", "estimate"}
GROUP_LABELS = {
    "Saito": {"x": 2017.80, "y": 1.15, "ha": "right", "va": "bottom"},
    "Hibat-Allah": {"x": 2019.94, "y": 95.0, "ha": "right", "va": "center"},
    "Rende": {"x": 2023.43, "y": 68.0, "ha": "right", "va": "center"},
    "Sprague": {"x": 2023.53, "y": 1580.0, "ha": "right", "va": "center"},
}
ANNOTATIONS = {
    "Choo": {"dx": 8, "dy": 4, "ha": "left", "va": "bottom"},
    "Zen": {"dx": 10, "dy": -4, "ha": "left", "va": "top"},
    "Pfau": {"dx": 10, "dy": 1, "ha": "left", "va": "center"},
    "Gutierrez": {"dx": 8, "dy": -7, "ha": "left", "va": "top"},
    "von Glehn": {"dx": -30, "dy": 7, "ha": "right", "va": "bottom"},
    "Chen": {"dx": 14, "dy": 3, "ha": "left", "va": "bottom"},
}


def load_literature_points(path: Path) -> list[dict[str, object]]:
    """Load and strictly validate the machine-readable figure ledger."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 19:
        raise ValueError(f"{path}: expected 19 audited literature points, found {len(rows)}")

    points: list[dict[str, object]] = []
    identities: set[tuple[str, float]] = set()
    for row in rows:
        domain = str(row["domain"])
        evidence = str(row["evidence"])
        if domain not in DOMAINS:
            raise ValueError(f"{path}: unknown domain {domain!r}")
        if evidence not in EVIDENCE_CLASSES:
            raise ValueError(f"{path}: unknown evidence class {evidence!r}")
        publication_year = int(row["publication_year"])
        plot_year = float(row["plot_year"])
        parameters = int(row["parameters"])
        samples = int(row["samples"])
        ratio = float(row["ratio"])
        if publication_year < 2017 or plot_year < 2017 or parameters <= 0 or samples <= 0:
            raise ValueError(f"{path}: invalid positive count or year in {row}")
        expected_ratio = parameters / samples
        if not math.isclose(ratio, expected_ratio, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(
                f"{path}: ratio mismatch for {row['source']!r}: "
                f"stored={ratio}, computed={expected_ratio}"
            )
        source_url = str(row["source_url"])
        parsed = urlparse(source_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError(f"{path}: source URL is not HTTPS: {source_url!r}")
        identity = (str(row["source"]), plot_year)
        if identity in identities:
            raise ValueError(f"{path}: duplicate literature point {identity}")
        identities.add(identity)
        points.append(
            {
                **row,
                "publication_year": publication_year,
                "plot_year": plot_year,
                "parameters": parameters,
                "samples": samples,
                "ratio": ratio,
            }
        )
    return points


def _draw_group_labels(ax: plt.Axes, points: list[dict[str, object]]) -> None:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for point in points:
        group = str(point["group"])
        if group in GROUP_LABELS:
            grouped[group].append(point)
    if set(grouped) != set(GROUP_LABELS):
        raise ValueError("literature ledger is missing a configured plotted group")

    for group, group_points in grouped.items():
        label = GROUP_LABELS[group]
        for point in group_points:
            ax.plot(
                [point["plot_year"], label["x"]],
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


def _plot_literature_ratio(input_path: Path, output_path: Path) -> None:
    """Render the literature ratio figure from its audited CSV ledger."""
    points = load_literature_points(input_path)
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
    _draw_group_labels(ax, points)

    for point in points:
        style = DOMAINS[str(point["domain"])]
        ax.scatter(
            point["plot_year"],
            point["ratio"],
            s=54,
            marker=style["marker"],
            facecolor=style["color"],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.82 if point["evidence"] == "estimate" else 1.0,
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
            label=style["label"],
        )
        for style in DOMAINS.values()
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
    ax.text(2026.25, 0.72, "sample-rich", fontsize=6.8, color="#52525b", ha="right", va="top")
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
        group = str(point["group"])
        if group in GROUP_LABELS:
            continue
        annotation = ANNOTATIONS.get(group)
        if annotation is None:
            raise ValueError(f"literature ledger has no annotation layout for {group!r}")
        ax.annotate(
            group,
            xy=(point["plot_year"], point["ratio"]),
            xytext=(annotation["dx"], annotation["dy"]),
            textcoords="offset points",
            fontsize=6.4,
            ha=annotation["ha"],
            va=annotation["va"],
            arrowprops={
                "arrowstyle": "-",
                "color": "#71717a",
                "linewidth": 0.55,
                "shrinkA": 2,
                "shrinkB": 4,
            },
        )

    ax.set_yscale("log")
    ax.set_xlim(2017.1, 2026.4)
    ax.set_ylim(5e-3, 5e3)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"Parameter-to-sample ratio $P/N_s$")
    ax.set_xticks([2018, 2020, 2022, 2024, 2026])
    ax.grid(True, which="major", axis="y", linewidth=0.5, color="#d4d4d8")
    ax.grid(True, which="minor", axis="y", linewidth=0.25, color="#e4e4e7")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=300,
        metadata=pdf_metadata("NQS parameter-to-sample literature ratios"),
    )
    plt.close(fig)


def plot_literature_ratio(input_path: Path, output_path: Path) -> None:
    """Render without leaking manuscript-specific Matplotlib settings."""
    with matplotlib.rc_context():
        _plot_literature_ratio(input_path, output_path)
