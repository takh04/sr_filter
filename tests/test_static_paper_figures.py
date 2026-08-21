from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

from sr_filtering_nqs.literature_ratio import (
    load_literature_points,
    plot_literature_ratio,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_FIGURE_SCRIPT = REPO_ROOT / "paper/scripts/generate_static_figures.py"


def _load_static_figure_generator():
    spec = importlib.util.spec_from_file_location(
        "sr_filter_paper_static_figure_generator",
        STATIC_FIGURE_SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_literature_figure_uses_audited_numeric_ledger_and_is_deterministic(tmp_path):
    input_path = REPO_ROOT / "paper/literature_ratio.csv"
    points = load_literature_points(input_path)

    assert len(points) == 19
    assert {point["evidence"] for point in points} == {
        "explicit",
        "derived",
        "estimate",
    }
    assert all(
        math.isclose(
            point["parameters"] / point["samples"],
            point["ratio"],
            rel_tol=1e-12,
        )
        for point in points
    )

    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    plot_literature_ratio(input_path, first)
    plot_literature_ratio(input_path, second)
    assert first.read_bytes() == second.read_bytes()


def test_figure3_uses_canonical_checkpoint_means_and_standard_errors():
    generator = _load_static_figure_generator()
    summary = generator.load_tfim_shift_summary(
        REPO_ROOT / "artifacts/robustness/shift_sweep_checkpoints.csv"
    )

    assert summary[1e-8]["r_val_raw"] == pytest.approx(
        (0.011618817115603919, 0.0009056899411956953)
    )
    assert summary[1e-6]["r_val_raw"] == pytest.approx(
        (0.009697429615273737, 0.000767134751187018)
    )
    assert summary[1e-8]["var_hat_raw"] == pytest.approx(
        (0.006993152289115824, 0.0005845141705187629)
    )
    assert summary[1e-1]["var_hat_raw"] == pytest.approx(
        (0.0007439051542133191, 3.306987923394526e-05)
    )


def test_figure4_uses_checkpoint_level_standard_errors():
    generator = _load_static_figure_generator()
    summary = generator.load_checkpoint_summary(
        REPO_ROOT / "artifacts/robustness/method_ablation_checkpoints.csv",
        system="tfim",
    )

    risk_mean, risk_se = summary["ms_sr"]["r_val_raw"]
    assert risk_mean == pytest.approx(0.006802216546115481)
    assert risk_se == pytest.approx(0.0005043004938616207)
