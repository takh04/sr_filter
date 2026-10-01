from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pickle
import pytest

import sr_filtering_nqs.large_scale.core.fixed_data_protocol as fdp


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint")


def test_fixed_data_paths_live_under_stable_workspace(tmp_path):
    base_root = tmp_path / "large_scale"

    assert fdp.train_batches_cache_path(base_root, "j1j2", 900) == (
        base_root / "fixed_data" / "cache" / "j1j2" / "step000900" / "train_batches.pkl"
    )
    assert fdp.validation_cache_path(base_root, "tfim", 1100) == (
        base_root / "fixed_data" / "cache" / "tfim" / "step001100" / "validation.pkl"
    )
    assert fdp.spectrum_ref_path(base_root, "tfim", 1100) == (
        base_root / "fixed_data" / "cache" / "tfim" / "step001100" / "ntk" / "spectrum_ref.npy"
    )
    assert fdp.fig2_checkpoint_lambda_path(base_root, "j1j2", 900, 1e-4) == (
        base_root / "fixed_data" / "results" / "fig2" / "j1j2" / "step000900" / "diag_1e-04.pkl"
    )
    assert fdp.fig3_checkpoint_eval_path(base_root, "tfim", 1100, k=4) == (
        base_root / "fixed_data" / "results" / "fig3" / "tfim" / "step001100" / "k4_eval.pkl"
    )


def test_spectrum_quantile_lambda_grid_handles_zero_and_negative_values():
    spectrum = np.array([10.0, 3.0, 1e-15, 0.0, -1e-4])

    lambdas = fdp.spectrum_quantile_lambda_grid(spectrum)

    assert len(lambdas) == 4
    assert all(np.isfinite(lam) and lam > 0.0 for lam in lambdas)
    assert min(lambdas) >= 1e-15


def test_holdout_partition_counts_preserve_replica_groups_for_tfim():
    j1j2_counts = fdp.holdout_partition_counts(8192, n_replicas=None, holdout_fraction=0.25)
    tfim_counts = fdp.holdout_partition_counts(12000, n_replicas=6000, holdout_fraction=0.25)

    assert j1j2_counts == {
        "solve_count": 6144,
        "holdout_count": 2048,
        "solve_n_replicas": None,
        "holdout_n_replicas": None,
    }
    assert tfim_counts == {
        "solve_count": 9000,
        "holdout_count": 3000,
        "solve_n_replicas": 4500,
        "holdout_n_replicas": 1500,
    }


def test_holdout_partition_counts_align_to_sample_axis_granularity():
    counts = fdp.holdout_partition_counts(
        6000,
        n_replicas=None,
        holdout_fraction=0.25,
        sample_axis_granularity=8,
    )

    assert counts == {
        "solve_count": 4496,
        "holdout_count": 1504,
        "solve_n_replicas": None,
        "holdout_n_replicas": None,
    }


def test_holdout_partition_counts_keep_replica_groups_when_aligning_to_sample_axis_granularity():
    counts = fdp.holdout_partition_counts(
        12000,
        n_replicas=6000,
        holdout_fraction=0.25,
        sample_axis_granularity=8,
    )

    assert counts == {
        "solve_count": 9000,
        "holdout_count": 3000,
        "solve_n_replicas": 4500,
        "holdout_n_replicas": 1500,
    }


def test_compatible_chunk_size_bwd_uses_divisor_for_tfim_same_batch_splits():
    driver = SimpleNamespace(chunk_size_bwd=12000)

    assert fdp._compatible_chunk_size_bwd(driver, system_name="tfim", sample_count=12000) == 12000
    assert fdp._compatible_chunk_size_bwd(driver, system_name="tfim", sample_count=9000) == 3000
    assert fdp._compatible_chunk_size_bwd(driver, system_name="tfim", sample_count=3000) == 3000

    driver.chunk_size_bwd = 6000
    assert fdp._compatible_chunk_size_bwd(driver, system_name="tfim", sample_count=9000) == 3000


def test_exact_loo_stacking_weights_recovers_known_solution():
    weights_star = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    cross_predictions = np.array(
        [
            [[0.0, 0.0], [1.0, 0.5], [2.0, -0.5]],
            [[0.5, 1.0], [0.0, 0.0], [1.5, 1.5]],
            [[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    targets = []
    for j in range(len(weights_star)):
        coeffs = weights_star / (1.0 - weights_star[j])
        coeffs[j] = 0.0
        targets.append(np.tensordot(coeffs, cross_predictions[j], axes=(0, 0)))
    targets = np.asarray(targets, dtype=np.float64)

    fitted = fdp.exact_loo_stacking_weights(targets, cross_predictions)

    assert np.allclose(fitted.sum(), 1.0, atol=1e-8)
    assert np.all(fitted >= -1e-10)
    assert np.allclose(fitted, weights_star, atol=1e-5)


def test_metrics_from_prediction_bank_persists_per_repeat_statistics():
    prediction_bank = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[2.0, 3.0, 4.0, 5.0]],
        ],
        dtype=np.float64,
    )
    energy_channels = np.array([[0.0, 1.0, 0.0, 1.0]], dtype=np.float64)

    metrics = fdp.metrics_from_prediction_bank(
        prediction_bank,
        energy_channels,
        mode=None,
        n_replicas=None,
    )

    assert len(metrics["r_val_raw_per_repeat"]) == 2
    assert len(metrics["r_val_norm_per_repeat"]) == 2
    assert np.isclose(
        metrics["r_val_raw"],
        np.mean(np.asarray(metrics["r_val_raw_per_repeat"], dtype=np.float64)),
    )


def test_ensure_reference_spectrum_writes_npy_and_meta(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "large_scale" / "j1j2_results" / "1e-04" / "results" / "checkpoints" / "checkpoint_step000900.pkl"
    _touch(ckpt_path)

    monkeypatch.setattr(
        fdp,
        "_load_training_driver",
        lambda ckpt_path, system_name, **kwargs: {
            "driver": object(),
            "config": {"diag_shift": 1e-4},
            "step": 900,
        },
    )
    monkeypatch.setattr(fdp, "snapshot_driver_state", lambda driver: {"snap": True})
    monkeypatch.setattr(fdp, "restore_driver_state", lambda driver, snapshot: None)
    monkeypatch.setattr(
        fdp,
        "_sample_live_prepared_batch",
        lambda driver, seed, chain_name: SimpleNamespace(sample_count=8, n_replicas=None),
    )
    monkeypatch.setattr(
        fdp,
        "base_solve_record",
        lambda driver, batch, diag_shift, collect_residual_info: SimpleNamespace(
            ntk_eigenvalues=np.array([3.0, 1.0, 0.2], dtype=np.float64)
        ),
    )

    spectrum = fdp.ensure_reference_spectrum(
        ckpt_path,
        base_root=tmp_path / "large_scale",
        system_name="j1j2",
    )

    npy_path = fdp.spectrum_ref_path(tmp_path / "large_scale", "j1j2", 900)
    meta_path = fdp.spectrum_ref_meta_path(tmp_path / "large_scale", "j1j2", 900)
    assert npy_path.exists()
    assert meta_path.exists()
    assert np.allclose(spectrum, np.array([3.0, 1.0, 0.2]))


def test_run_fixed_data_fig2_writes_best_fixed_lambda_summary(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "large_scale" / "j1j2_results" / "1e-04" / "results" / "checkpoints" / "checkpoint_step000900.pkl"
    _touch(ckpt_path)

    monkeypatch.setattr(
        fdp,
        "canonical_fixed_data_checkpoints",
        lambda *args, **kwargs: [(900, ckpt_path)],
    )
    seen = {}

    def evaluate(*args, **kwargs):
        seen.update(kwargs)
        return [
            {"step": 900, "lambda": 1e-4, "r_val_raw": 2.0, "r_val_norm": 0.2},
            {"step": 900, "lambda": 1e-5, "r_val_raw": 1.0, "r_val_norm": 0.1},
        ]

    monkeypatch.setattr(fdp, "evaluate_fixed_data_fig2_checkpoint", evaluate)

    output_root = tmp_path / "generated"
    fdp.run_fixed_data_fig2(
        tmp_path / "large_scale",
        output_root=output_root,
        system_name="j1j2",
    )

    assert seen["base_root"] == output_root
    summary = fdp.load_best_fixed_lambda_summary(output_root)
    assert summary["systems"]["j1j2"]["900"]["lambda_star"] == 1e-5


def test_run_fixed_data_fig3_requires_fixed_data_caches(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "large_scale" / "tfim_results" / "1e-04" / "results" / "checkpoints" / "checkpoint_step001100.pkl"
    _touch(ckpt_path)

    monkeypatch.setattr(
        fdp,
        "canonical_fixed_data_checkpoints",
        lambda *args, **kwargs: [(1100, ckpt_path)],
    )
    monkeypatch.setattr(
        fdp,
        "evaluate_fixed_data_fig3_checkpoint",
        lambda *args, **kwargs: {"ok": True},
    )

    # Missing caches should fail before the mocked evaluator is reached.
    try:
        fdp.run_fixed_data_fig3(tmp_path / "large_scale", system_name="tfim")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected run_fixed_data_fig3 to require cached fixed-data artifacts")

    fdp.update_best_fixed_lambda_summary(
        tmp_path / "large_scale",
        system_name="tfim",
        checkpoint_summaries={1100: {"lambda_star": 1e-4, "r_val_raw": 1.0, "r_val_norm": 0.1}},
        diagnostic_lambdas=(1e-4,),
    )
    train_path = fdp.train_batches_cache_path(tmp_path / "large_scale", "tfim", 1100)
    val_path = fdp.validation_cache_path(tmp_path / "large_scale", "tfim", 1100)
    spec_path = fdp.spectrum_ref_path(tmp_path / "large_scale", "tfim", 1100)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    with train_path.open("wb") as f:
        pickle.dump({"ok": True}, f)
    with val_path.open("wb") as f:
        pickle.dump({"ok": True}, f)
    np.save(spec_path, np.array([1.0, 0.5], dtype=np.float64))

    results = fdp.run_fixed_data_fig3(tmp_path / "large_scale", system_name="tfim")
    assert results == [{"ok": True}]


@pytest.mark.parametrize("system_name", ["j1j2", "tfim"])
def test_evaluate_fixed_data_fig3_checkpoint_writes_six_method_schema(
    tmp_path, monkeypatch, system_name
):
    ckpt_path = tmp_path / "large_scale" / "j1j2_results" / "1e-04" / "results" / "checkpoints" / "checkpoint_step000900.pkl"
    _touch(ckpt_path)

    monkeypatch.setattr(
        fdp,
        "load_cached_train_batches",
        lambda base_root, system_name, step: {
            "repeat_count": 1,
            "batch_slots": 4,
            "repeats": [
                {
                    "slots": [{"slot_index": idx} for idx in range(4)],
                }
            ],
        },
    )
    monkeypatch.setattr(
        fdp,
        "load_cached_validation_payload",
        lambda base_root, system_name, step: {
            "samples": np.zeros((4, 2), dtype=np.int8),
            "energy_channels": np.zeros((1, 4), dtype=np.float64),
            "n_replicas": None,
            "mode": None,
        },
    )
    monkeypatch.setattr(fdp, "load_cached_spectrum", lambda *args, **kwargs: np.array([4.0, 2.0, 1.0, 0.5]))
    monkeypatch.setattr(fdp, "best_fixed_lambda_for_checkpoint", lambda *args, **kwargs: 1e-4)
    monkeypatch.setattr(
        fdp,
        "_load_training_driver",
        lambda ckpt_path, system_name, **kwargs: {
            "driver": SimpleNamespace(chunk_size_bwd=None),
            "config": {"diag_shift": 1e-4},
            "step": 900,
        },
    )
    monkeypatch.setattr(
        fdp,
        "ensure_reference_spectrum",
        lambda *args, **kwargs: np.array([4.0, 2.0, 1.0, 0.5]),
    )
    monkeypatch.setattr(fdp, "snapshot_driver_state", lambda driver: {"snap": True})
    monkeypatch.setattr(fdp, "restore_driver_state", lambda driver, snapshot: None)

    def fake_sample_live_prepared_batch(driver, *, seed, chain_name):
        slot_index = int(chain_name.rsplit("_s", 1)[1])
        return SimpleNamespace(
            sample_count=8,
            n_replicas=None,
            slot_index=slot_index,
        )

    monkeypatch.setattr(fdp, "_sample_live_prepared_batch", fake_sample_live_prepared_batch)
    solve_batches = []

    def fake_split(driver, batch, *, sample_axis_granularity):
        assert batch.slot_index == 0
        assert sample_axis_granularity == 8
        return (
            SimpleNamespace(sample_count=6, n_replicas=None, slot_index=0),
            SimpleNamespace(sample_count=2, n_replicas=None, slot_index=0),
            {"solve_count": 6, "holdout_count": 2,
             "solve_n_replicas": None, "holdout_n_replicas": None},
        )

    monkeypatch.setattr(fdp, "split_prepared_batch_for_same_batch_stacking", fake_split)

    def fake_solve(driver, batch, diag_shift, collect_residual_info):
        solve_batches.append(batch)
        return SimpleNamespace(delta=np.array([float(diag_shift)], dtype=np.float64))

    monkeypatch.setattr(
        fdp,
        "base_solve_record",
        fake_solve,
    )
    monkeypatch.setattr(
        fdp,
        "weighted_tree_sum",
        lambda deltas, weights: np.array([float(np.sum(weights))], dtype=np.float64),
    )

    def fake_same_batch_stacking_weights(driver, weight_batch, deltas):
        assert weight_batch.slot_index == (0 if system_name == "j1j2" else 1)
        assert weight_batch.sample_count == (2 if system_name == "j1j2" else 8)
        return np.array([0.1, 0.2, 0.3, 0.4]), {"weight_fit_samples": weight_batch.sample_count}

    monkeypatch.setattr(fdp, "compute_same_batch_stacking_weights", fake_same_batch_stacking_weights)
    monkeypatch.setattr(
        fdp,
        "compute_exact_loo_weights",
        lambda driver, batches, deltas: (
            np.array([0.25, 0.25, 0.25, 0.25]),
            {"loo_objective": 0.0},
        ),
    )

    def fake_method_metrics_live(
        training,
        *,
        base_root,
        system_name,
        final_deltas,
        n_val,
        validation_round_samples,
        validation_sample_chunk,
        validation_seed_base,
        jvp_chunk,
        delta_batch_size,
    ):
        return (
            {
                "r_val_raw": 1.0,
                "r_val_norm": 0.1,
                "r_val_raw_per_repeat": [1.0],
                "r_val_norm_per_repeat": [0.1],
                "var_hat_raw": 0.25,
                "var_hat_norm": 0.025,
                "snorm_diff_sq": 0.5,
                "target_norm_sq": 10.0,
                "energy_variance_eval": 2.5,
                "repeat_count": 1,
            },
            {"validation_seed": 123, "round_samples": 4},
        )

    monkeypatch.setattr(fdp, "_method_metrics_from_final_deltas_live", fake_method_metrics_live)

    payload = fdp.evaluate_fixed_data_fig3_checkpoint(
        ckpt_path,
        base_root=tmp_path / "large_scale",
        system_name=system_name,
        repeat_count=1,
    )

    assert tuple(payload["methods"]) == fdp.FIG3_METHOD_IDS
    same_batch_meta = payload["methods"]["same_batch_multilambda_stacking_k4"]["method_meta"]
    if system_name == "j1j2":
        assert same_batch_meta["solve_sample_count"] == 6
        assert same_batch_meta["weight_fit_samples"] == 2
        assert same_batch_meta["weight_fit_source"] == "heldout_quarter_of_slot0"
        assert same_batch_meta["weight_fit_batch_slot"] == 0
        assert [batch.sample_count for batch in solve_batches] == [8, 6, 6, 6, 6] + [8] * 8
    else:
        assert same_batch_meta["solve_sample_count"] == 8
        assert same_batch_meta["weight_fit_samples"] == 8
        assert same_batch_meta["weight_fit_source"] == "independent_batch"
        assert same_batch_meta["weight_fit_batch_slot"] == 1
        assert [batch.sample_count for batch in solve_batches] == [8] * 13
