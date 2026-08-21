from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

from sr_filtering_nqs.large_scale.cli import plot_fixed_data_fig2, plot_fixed_data_fig3
from sr_filtering_nqs.large_scale.core.fixed_data_protocol import fig2_checkpoint_lambda_path, fig3_checkpoint_eval_path


def _write_pickle(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def test_plot_fixed_data_fig2_renders_summary_outputs(tmp_path, monkeypatch):
    base_root = tmp_path / "large_scale"
    for system_name, step in (("j1j2", 900), ("tfim", 1100)):
        for lam, rval, var_hat in ((1e-4, 1.0, 0.2), (1e-5, 0.8, 0.1)):
            _write_pickle(
                fig2_checkpoint_lambda_path(base_root, system_name, step, lam),
                {
                    "system": system_name,
                    "step": step,
                    "lambda": lam,
                    "r_val_raw": rval,
                    "var_hat_raw": var_hat,
                },
            )

    output_dir = tmp_path / "plots"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_fixed_data_fig2.py",
            "--input-dir",
            str(base_root),
            "--output-dir",
            str(output_dir),
        ],
    )

    plot_fixed_data_fig2.main()

    assert (output_dir / "figure2_fixed_data_summary.csv").exists()
    assert (output_dir / "figure2_fixed_data_raw.pdf").exists()


def test_plot_fixed_data_fig3_renders_summary_outputs(tmp_path, monkeypatch):
    base_root = tmp_path / "large_scale"
    for system_name, step in (("j1j2", 900), ("tfim", 1100)):
        _write_pickle(
            fig3_checkpoint_eval_path(base_root, system_name, step, k=4),
            {
                "system": system_name,
                "methods": {
                    method_id: {
                        "r_val_raw": 1.0 + 0.1 * idx,
                        "var_hat_raw": 0.2 + 0.05 * idx,
                    }
                    for idx, method_id in enumerate(plot_fixed_data_fig3.FIG3_METHOD_IDS)
                },
            },
        )

    output_dir = tmp_path / "plots"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_fixed_data_fig3.py",
            "--input-dir",
            str(base_root),
            "--output-dir",
            str(output_dir),
            "--k",
            "4",
        ],
    )

    plot_fixed_data_fig3.main()

    assert (output_dir / "figure3_fixed_data_summary.csv").exists()
    assert (output_dir / "figure3_fixed_data_raw.pdf").exists()
