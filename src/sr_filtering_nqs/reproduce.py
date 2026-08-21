"""Named, portable reproduction recipes for the compact public artifacts."""

from __future__ import annotations

from pathlib import Path


def reproduce(
    recipe: str,
    *,
    input_root: str | Path,
    output_dir: str | Path = "outputs/reproduced",
) -> list[Path]:
    output_dir = Path(output_dir)
    outputs: list[Path] = []
    if recipe not in {"all", "curated", "online"}:
        raise ValueError(f"Unknown reproduction recipe {recipe!r}")
    # Validate the strict two-branch gate first so ``all`` never produces a
    # misleading partial output set when online artifacts are incomplete.
    if recipe in {"all", "online"}:
        from sr_filtering_nqs.large_scale.cli.plot_online_comparison import (
            plot_online_comparison,
        )

        outputs.extend(plot_online_comparison(input_root, output_dir))
    if recipe in {"all", "curated"}:
        from sr_filtering_nqs.large_scale.cli.plot_curated_artifacts import (
            plot_ms_sr_ablation,
            plot_tfim_shift_sweep,
        )

        artifact_root = Path(input_root)
        if not (artifact_root / "large_scale").is_dir():
            nested = artifact_root / "artifacts"
            if (nested / "large_scale").is_dir():
                artifact_root = nested
        outputs.append(plot_tfim_shift_sweep(artifact_root, output_dir))
        outputs.append(plot_ms_sr_ablation(artifact_root, output_dir))
    return outputs
