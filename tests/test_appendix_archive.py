"""Check archived evidence against the manuscript's appendix statistics."""

import csv
import json
from collections import defaultdict

import numpy as np

from sr_filtering_nqs.appendix_figures import reproduce_appendix
from sr_filtering_nqs.paper_figures import DEFAULT_ARTIFACT_ROOT as ROOT


def rows(name):
    with (ROOT / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_robustness_table_recovers_from_original_checkpoint_payloads():
    records = json.loads((ROOT / "appendix/diagnostic_results.json").read_text())
    assert len(records) == 274
    runs = {
        "k2": ("20260428", "k2_eval.pkl"),
        "k3": ("20260428", "k3_eval.pkl"),
        "k4 (shipped)": ("20260428", "k4_eval.pkl"),
        "k5": ("20260428", "k5_eval.pkl"),
        "k4 qgridA 0.95/0.8/0.5/0.2": ("20260428_qgridA", "k4_eval.pkl"),
        "k4 qgridB 0.8/0.6/0.3/0.05": ("20260428_qgridB", "k4_eval.pkl"),
        "k4 lt1e-3 (steps 1100/1500/2000)": ("20260428_lt1e3", "k4_eval.pkl"),
        "k4 lt1e-5 (steps 1100/1500/2000)": ("20260428_lt1e5", "k4_eval.pkl"),
    }
    methods = {
        "single@lambda*": "single_best_fixed_lambda",
        "same-batch uniform": "same_batch_multilambda_uniform_k4",
        "same-batch stacking": "same_batch_multilambda_stacking_k4",
        "bagged@lambda*": "indep_single_lambda_uniform_k4",
        "MS-SR uniform": "indep_multilambda_uniform_k4",
        "MS-SR": "indep_multilambda_stacking_k4",
    }
    for row in rows("appendix/robustness/k_quantile_ablation_summary.csv"):
        directory, filename = runs[row["run"]]
        selected = [
            r["result"]
            for r in records
            if r["source"].startswith(f"large_scale/{directory}/results/fig3/tfim/")
            and r["source"].endswith("/" + filename)
        ]
        assert len(selected) == int(row["n_checkpoints"])
        for field, prefix in (("r_val_raw", "e_val"), ("var_hat_raw", "v_mb")):
            values = [r["methods"][methods[row["method"]]][field] for r in selected]
            np.testing.assert_allclose(
                [np.mean(values), np.std(values)],
                [float(row[prefix + "_mean"]), float(row[prefix + "_std"])],
                atol=5.1e-7,
                rtol=0,
            )


def test_tracking_archive_preserves_points_and_exposes_raw_disagreements():
    points = rows("appendix/online_tracking/points.csv")
    assert len(points) == 96
    for box in rows("appendix/online_tracking/box_stats.csv"):
        values = [
            float(p["log10_ratio"])
            for p in points
            if p["system"] == box["system"] and p["model"] == box["model"]
        ]
        assert len(values) == int(box["n"])
        assert sum(v < 0 for v in values) == int(box["wins"])
        np.testing.assert_allclose(np.median(values), float(box["median_log10_ratio"]))
    trajectories = defaultdict(list)
    for row in rows("appendix/online_tracking/trajectories.csv"):
        trajectories[int(row["point_id"]), row["method"]].append(float(row["s_relative_error"]))
    assert len(trajectories) == 279
    assert sum(map(len, trajectories.values())) == 27769
    matched, mismatches = 0, []
    methods = ("sr", "spring", "mssr_ema_ntk_quantile")
    for index, point in enumerate(points):
        if all(len(trajectories[index, m]) == 100 for m in methods):
            a, b, ms = [np.mean(trajectories[index, m]) for m in methods]
            if np.isclose(ms / min(a, b), float(point["ratio"]), rtol=1e-10, atol=0):
                matched += 1
            else:
                mismatches.append(index)
    audit = json.loads((ROOT / "appendix/online_tracking/reconstruction_audit.json").read_text())
    assert matched == audit["matching_points"] == 81
    assert mismatches == [r["point_id"] for r in audit["disagreements"]]
    assert len(mismatches) == 2
    assert audit["complete_trajectory_triples"] == 83


def test_appendix_plots_need_only_bundled_inputs(tmp_path):
    paths = reproduce_appendix(ROOT, tmp_path)
    assert {p.name for p in paths} == {
        "bagged_shift_robustness.pdf",
        "figure5_srel_logratio_hamiltonian_boxplots.pdf",
    }
    assert all(p.read_bytes().startswith(b"%PDF-") for p in paths)
