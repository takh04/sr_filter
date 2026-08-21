from __future__ import annotations

from pathlib import Path

from sr_filtering_nqs.large_scale.cli.plot_curated_artifacts import (
    plot_ms_sr_ablation,
    plot_tfim_shift_sweep,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_curated_pdf_reproduction_is_byte_deterministic(tmp_path):
    artifact_root = REPO_ROOT / "artifacts"
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_outputs = (
        plot_tfim_shift_sweep(artifact_root, first),
        plot_ms_sr_ablation(artifact_root, first),
    )
    second_outputs = (
        plot_tfim_shift_sweep(artifact_root, second),
        plot_ms_sr_ablation(artifact_root, second),
    )

    assert [path.read_bytes() for path in first_outputs] == [
        path.read_bytes() for path in second_outputs
    ]
