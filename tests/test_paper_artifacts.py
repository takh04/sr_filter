"""Scientific provenance and rendering checks for the released main-text bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from sr_filtering_nqs.paper_figures import (
    DEFAULT_ARTIFACT_ROOT,
    online_summary,
    reproduce_paper,
    resolve_artifact_root,
    validate_artifacts,
)


def rows(name: str) -> list[dict[str, str]]:
    with (DEFAULT_ARTIFACT_ROOT / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_bundle_manifest_is_complete_and_detects_tampering(tmp_path):
    manifest = validate_artifacts(DEFAULT_ARTIFACT_ROOT)
    paths = {entry["path"] for entry in manifest["files"]}
    assert paths == {
        str(path.relative_to(DEFAULT_ARTIFACT_ROOT))
        for path in DEFAULT_ARTIFACT_ROOT.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert {figure["id"] for figure in manifest["figures"]} == set("12345")
    assert all(Path(name).suffix in {".csv", ".json", ".npz", ".md"} for name in paths)
    copied = tmp_path / "bundle"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, copied)
    (copied / "figure5/endpoint_pairs.csv").write_text("corrupted\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_artifacts(copied)


def test_bundle_root_resolution_and_safe_paths(tmp_path):
    assert resolve_artifact_root(DEFAULT_ARTIFACT_ROOT.parent) == DEFAULT_ARTIFACT_ROOT
    assert resolve_artifact_root(DEFAULT_ARTIFACT_ROOT.parents[1]) == DEFAULT_ARTIFACT_ROOT
    copied = tmp_path / "bundle"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, copied)
    manifest = json.loads((copied / "manifest.json").read_text())
    manifest["files"][0]["path"] = "../outside.csv"
    (copied / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe artifact path"):
        validate_artifacts(copied)


def test_amplitude_arrays_are_safe_and_risk_uncertainties_match():
    metadata = json.loads((DEFAULT_ARTIFACT_ROOT / "figure2/metadata.json").read_text())
    assert metadata["matched_fidelity"] >= 0.98
    with np.load(
        DEFAULT_ARTIFACT_ROOT / "figure2/amplitude_diagnostics.npz", allow_pickle=False
    ) as data:
        for index, panel in enumerate(metadata["panels"]):
            prefix = f"panel{index}_"
            assert data[prefix + "pi"].shape == (65536,)
            assert data[prefix + "prediction"].shape == (65536,)
            assert data[prefix + "target"].shape == (65536,)
            assert len(data[prefix + "train_indices"]) == 4096
            np.testing.assert_allclose(data[prefix + "pi"].sum(), 1, atol=1e-12)
            risk = data[prefix + "risk_samples"]
            assert risk.size == (100 if panel["protocol"] == "matched_noise" else 10)
            np.testing.assert_allclose(risk.mean(), panel["excess_risk"], rtol=1e-12)
            np.testing.assert_allclose(
                risk.std(ddof=1) / np.sqrt(risk.size),
                panel["excess_risk_standard_error"],
                rtol=1e-12,
            )
            assert panel["lambda"] == 1e-9


def test_corrected_ablation_aggregates_and_tunes_bagged_objective():
    checkpoints = rows("figure4/checkpoint_values.csv")
    summary = rows("figure4/summary.csv")
    sweep = rows("figure4/full_sweep.csv")
    assert len(checkpoints) == 2 * 10 * 7
    assert len(sweep) == 2 * 10 * 8
    for row in summary:
        values = np.array(
            [
                float(item[row["metric"]])
                for item in checkpoints
                if item["system"] == row["system"] and item["method"] == row["method"]
            ]
        )
        assert values.size == 10
        np.testing.assert_allclose(values.mean(), float(row["mean"]), rtol=1e-12)
        np.testing.assert_allclose(values.std(ddof=0) / np.sqrt(10), float(row["sem"]), rtol=1e-12)
    for tuned in (row for row in checkpoints if row["method"] == "bagged_tuned"):
        candidates = [
            row
            for row in sweep
            if row["system"] == tuned["system"] and row["step"] == tuned["step"]
        ]
        best = min(candidates, key=lambda row: float(row["r_val_raw"]))
        assert float(tuned["lambda_bagged_tuned"]) == float(best["lambda"])
        np.testing.assert_allclose(float(tuned["r_val_raw"]), float(best["r_val_raw"]))
        np.testing.assert_allclose(float(tuned["var_hat_raw"]), float(best["var_hat_raw"]))


def test_online_scope_pairing_and_reported_statistics():
    training = rows("figure5/training_curves.csv")
    assert {row["source_step"] for row in training} == {"2500"}
    counts = Counter((row["method"], int(row["branch_seed"])) for row in training)
    assert counts == Counter(
        {
            (method, seed): count
            for method, count in (("SR", 500), ("MS-SR", 125))
            for seed in range(104, 109)
        }
    )
    for seed in range(104, 109):
        initial = [
            float(row["energy_per_site"])
            for row in training
            if int(row["branch_seed"]) == seed and row["update"] == "1"
        ]
        assert len(initial) == 2 and initial[0] == initial[1]
    summary = online_summary(DEFAULT_ARTIFACT_ROOT)
    assert summary["pair_count"] == 5
    assert summary["ms_sr_lower_count"] == 3
    np.testing.assert_allclose(summary["paired_t_95_interval"], [-7.05e-5, 3.50e-5], atol=5e-8)
    assert round(summary["mean_delta_energy_per_site"] * 1e5, 2) == -1.78
    timing = rows("figure5/timing_summary.csv")
    assert [round(float(row["median_seconds"]), 2) for row in timing] == [13.38, 53.55, 54.09]


def test_all_main_figure_inputs_render_deterministically(tmp_path):
    outputs = reproduce_paper(DEFAULT_ARTIFACT_ROOT, tmp_path / "first")
    assert {path.name for path in outputs} == {
        "p_over_ns_literature.pdf",
        "figure2.pdf",
        "figure3_tfim_fixed_data.pdf",
        "Figure4_linear.pdf",
        "historical_online_cholesky.pdf",
        "historical_online_endpoints.tex",
        "figure5_timing.tex",
    }
    for path in outputs:
        assert path.stat().st_size > 100
        if path.suffix == ".pdf":
            assert path.read_bytes().startswith(b"%PDF-")
    # All seven inputs must be reproducible within a fixed rendering environment.
    repeated = reproduce_paper(DEFAULT_ARTIFACT_ROOT, tmp_path / "second")
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in repeated
    ]
    endpoint_text = (tmp_path / "first/historical_online_endpoints.tex").read_text()
    assert "104" not in endpoint_text
    timing_text = (tmp_path / "first/figure5_timing.tex").read_text()
    assert "53.55" in timing_text and "54.09" in timing_text
