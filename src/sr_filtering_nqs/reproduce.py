"""Portable reproduction of the main paper figures from bundled summaries."""

from __future__ import annotations

from pathlib import Path


def reproduce(
    recipe: str,
    *,
    input_root: str | Path,
    output_dir: str | Path = "outputs/reproduced",
) -> list[Path]:
    from sr_filtering_nqs.paper_figures import reproduce_paper

    if recipe not in {"paper", "all", "curated", "online"}:
        raise ValueError(f"Unknown reproduction recipe {recipe!r}")
    figures = ["5"] if recipe == "online" else None
    paths = reproduce_paper(Path(input_root), Path(output_dir), figures=figures)
    if recipe == "all":
        from sr_filtering_nqs.appendix_figures import reproduce_appendix

        paths.extend(reproduce_appendix(input_root, output_dir))
    return paths
