#!/usr/bin/env python3
"""Preserve selected paper results from a TRUSTED original experiment checkout.

This one-time importer reads legacy pickle results. Never run it on untrusted
input. The public outputs are JSON/CSV/NPZ and require no pickle loading.
The original checkout is read only; unrelated experiments are not imported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def portable(value):
    """Keep numerical results, normalize archive paths, and encode NaN as null."""
    if isinstance(value, dict):
        return {str(portable(k)): portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(v) for v in value]
    if isinstance(value, np.ndarray):
        return portable(value.tolist())
    if isinstance(value, np.generic):
        return portable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        # These are provenance references, not runtime dependencies.
        for marker in ("benign_overfitting_nqs/", "codebase/nqs/", "sr_filter_artifacts_working/"):
            if marker in value:
                value = value.split(marker, 1)[1]
                break
        value = re.sub(r"/(?:home|data|tmp)/[^\s\"']+", "<local-path>", value)
        return value
    if value is None or isinstance(value, (int, float, bool)):
        return value
    raise TypeError(f"Unsupported archival value: {type(value)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trusted-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/paper")
    args = parser.parse_args()
    source, output = args.trusted_source.resolve(), args.output.resolve()
    provenance = []

    def read(path):
        data = path.read_bytes()
        provenance.append(
            {
                "source": str(path.relative_to(source)),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        return data

    def write_json(name, value):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(portable(value), indent=2, allow_nan=False) + "\n")

    def copy_file(relative, destination):
        path = source / relative
        data = read(path)
        target = output / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            write_json(destination, json.loads(data))
        else:
            target.write_bytes(data)

    # Full small result payloads preserve seeds, shifts, stacking weights,
    # checkpoint statistics, and every saved repeated residual.
    patterns = [
        "large_scale/20260422/results/fig2/tfim/step*/diag_*.pkl",
        "large_scale/20260428/results/fig3/tfim/step*/*eval*.pkl",
        "large_scale/20260428_qgridA/results/fig3/tfim/step*/k4_eval.pkl",
        "large_scale/20260428_qgridB/results/fig3/tfim/step*/k4_eval.pkl",
        "large_scale/20260428_lt1e3/results/fig[23]/tfim/step*/*.pkl",
        "large_scale/20260428_lt1e5/results/fig[23]/tfim/step*/*.pkl",
        "large_scale/20260918/j1j2_corrected/results/fig3/j1j2/step*/k4_eval.json",
        "large_scale/20260918/bagged_objective_sweep/results/*/step*/bagged_sweep.json",
    ]
    results = []
    for pattern in patterns:
        paths = sorted(source.glob(pattern))
        if not paths:
            raise FileNotFoundError(pattern)
        for path in paths:
            data = read(path)
            payload = pickle.loads(data) if path.suffix == ".pkl" else json.loads(data)
            results.append({"source": str(path.relative_to(source)), "result": payload})
    write_json("appendix/diagnostic_results.json", results)

    for path in sorted((source / "rebuttal_assets/checkpoint_local").glob("*.csv")):
        copy_file(path.relative_to(source), f"appendix/robustness/{path.name}")
    copy_file(
        "paper/figures/data/mssr_ablation/sweep_summary.csv",
        "appendix/robustness/sweep_summary.csv",
    )
    for path in sorted((source / "rebuttal_assets/sr_cost/data").glob("pair*_*.json")):
        copy_file(path.relative_to(source), f"figure5/timing_raw/{path.name}")

    stem = "Figure5_small/figure5_drafts/figure5_srel_logratio_hamiltonian_boxplots_seed0_seed1"
    for suffix, name in [
        ("_points.csv", "points.csv"),
        ("_box_stats.csv", "box_stats.csv"),
        ("_stats.json", "plot_statistics.json"),
    ]:
        copy_file(stem + suffix, f"appendix/online_tracking/{name}")
    points = list(csv.DictReader((source / (stem + "_points.csv")).open()))
    runs, trajectories, missing = [], [], []
    for point_id, point in enumerate(points):
        for method in ("sr", "spring", "mssr_ema_ntk_quantile"):
            directory = (
                source
                / "Figure5_small"
                / point["run_name"]
                / ("samples_" + point["samples"])
                / "results"
                / point["system"]
                / point["model"]
                / method
            )
            run = {
                "point_id": point_id,
                "method": method,
                "source_directory": str(directory.relative_to(source)),
            }
            for name in ("config.json", "summary.json"):
                path = directory / name
                run[name.removesuffix(".json")] = json.loads(read(path)) if path.is_file() else None
                if not path.is_file():
                    missing.append(str(path.relative_to(source)))
            path = directory / "update_tracking.csv"
            rows = list(csv.DictReader(read(path).decode().splitlines())) if path.is_file() else []
            if not path.is_file():
                missing.append(str(path.relative_to(source)))
            run["tracking_rows"] = len(rows)
            run["tracking_last_step"] = int(rows[-1]["update_step"]) if rows else None
            runs.append(run)
            trajectories.extend({"point_id": point_id, **row} for row in rows)
    write_json("appendix/online_tracking/runs.json", runs)
    trajectory_path = output / "appendix/online_tracking/trajectories.csv"
    with trajectory_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trajectories[0]))
        writer.writeheader()
        writer.writerows(trajectories)
    errors = defaultdict(list)
    for row in trajectories:
        errors[int(row["point_id"]), row["method"]].append(float(row["s_relative_error"]))
    reconstructed, disagree = [], []
    for point_id, point in enumerate(points):
        methods = ("sr", "spring", "mssr_ema_ntk_quantile")
        if not all(len(errors[point_id, method]) == 100 for method in methods):
            continue
        means = {method: float(np.mean(errors[point_id, method])) for method in methods}
        ratio = means["mssr_ema_ntk_quantile"] / min(means["sr"], means["spring"])
        record = {
            "point_id": point_id,
            "source_table_ratio": float(point["ratio"]),
            "trajectory_ratio": ratio,
            "trajectory_means": means,
        }
        reconstructed.append(record)
        if not math.isclose(ratio, float(point["ratio"]), rel_tol=1e-10):
            disagree.append(record)
    write_json(
        "appendix/online_tracking/reconstruction_audit.json",
        {
            "published_points": len(points),
            "complete_trajectory_triples": len(reconstructed),
            "matching_points": len(reconstructed) - len(disagree),
            "disagreements": disagree,
            "policy": "Preserve published source points and all available raw records; "
            "do not silently replace either. "
            "Missing/incomplete records are in archive_import.json.",
        },
    )
    copy_file(
        "audits/hamiltonian_20260919/online_report.json",
        "appendix/online_tracking/hamiltonian_audit.json",
    )
    for path in sorted((source / "Figure5_small/cache/exact_states").glob("*.npz")):
        if path.name in {"j1j2_L4_N16.npz", "ssm_L4_N16.npz", "tfim2d_L4_N16_h1.npz"}:
            copy_file(
                path.relative_to(source), f"appendix/online_tracking/exact_states/{path.name}"
            )

    # Only source2500 belongs to the manuscript; other online campaigns are excluded.
    online = json.loads(read(source / "paper/figures/data/historical_online/provenance.json"))
    selected = [entry for entry in online["inputs"] if "source2500/" in entry["path"]]
    write_json(
        "figure5/original_source_records.json",
        {
            "inputs": selected,
            "availability": "Original run directories are absent at their recorded paths. "
            "Training observations and endpoint estimates survive in the bundled tables.",
        },
    )
    write_json(
        "archive_import.json",
        {
            "schema_version": 1,
            "source_repository": "nqs_rev/benign_overfitting_nqs",
            "diagnostic_payloads": len(results),
            "tracking_points": len(points),
            "tracking_method_records": len(runs),
            "tracking_rows": len(trajectories),
            "missing_original_files": missing,
            "incomplete_tracking": [
                {k: run[k] for k in ("point_id", "method", "tracking_last_step")}
                for run in runs
                if run["tracking_rows"] and run["tracking_last_step"] < 100
            ],
            "transformations": [
                "Legacy result pickles converted to JSON without dropping result fields.",
                "Machine-local paths normalized; nonfinite JSON numbers encoded as null.",
                "Tracking CSVs concatenated with point_id referencing points.csv row order.",
                "Only manuscript cohorts imported; model weights and sample caches omitted.",
            ],
            "inputs": provenance,
        },
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    previous = {entry["path"]: entry for entry in manifest["files"]}
    manifest["scope"] = "Camera-ready main-text Figures 1–5 and appendix results"
    manifest["appendix_reproduction"] = "python -m sr_filtering_nqs.appendix_figures"
    manifest["files"] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        name, data = path.relative_to(output).as_posix(), path.read_bytes()
        record = previous.get(name, {"path": name, "figure": None})
        record.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        manifest["files"].append(record)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Imported {len(results)} diagnostics, {len(points)} tracking points, "
        f"{len(trajectories)} tracking rows; recorded {len(missing)} missing source files."
    )


if __name__ == "__main__":
    main()
