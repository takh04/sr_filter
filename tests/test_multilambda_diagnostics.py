from __future__ import annotations

import numpy as np
import pytest

import sr_filtering_nqs.large_scale.core.multilambda_diagnostics as mdiag


def _dummy_bank_payload(lambda_grid):
    return {
        "system": "j1j2",
        "step": 100,
        "lambda": 1e-3,
        "lambda_grid": [float(x) for x in lambda_grid],
        "repeat_count": 2,
        "same_batch": {
            "batch_mode": "same_batch",
            "repeats": [
                {
                    "deltas": [{"params": np.array([1.0])}, {"params": np.array([2.0])}],
                    "gcv_y": np.array([1.0, 0.0]),
                    "gcv_mu": np.array([1.0, 0.5]),
                    "gcv_z": np.array([[1.0, 0.0], [0.0, 1.0]]),
                    "per_lambda": [],
                }
                for _ in range(2)
            ],
        },
        "indep_batch": {
            "batch_mode": "indep_batch",
            "repeats": [
                {
                    "deltas": [{"params": np.array([1.0])}, {"params": np.array([2.0])}],
                    "per_lambda": [],
                }
                for _ in range(2)
            ],
        },
    }


def test_precompute_mixture_bank_writes_expected_schema(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "results" / "checkpoints" / "checkpoint_step000100.pkl"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_bytes(b"checkpoint")
    driver = object()
    lambda_grid = (1e-3, 1e-4)

    monkeypatch.setattr(
        mdiag,
        "_load_training_driver",
        lambda ckpt_path, system_name: {
            "system": system_name,
            "config": {"diag_shift": 1e-3, "n_samples": 8},
            "step": 100,
            "driver": driver,
        },
    )
    monkeypatch.setattr(mdiag, "snapshot_driver_state", lambda driver: {"snap": True})
    monkeypatch.setattr(mdiag, "restore_driver_state", lambda driver, snapshot: None)
    monkeypatch.setattr(
        mdiag,
        "_same_batch_repeat_payload",
        lambda driver, lambda_grid: {
            "batch_mode": "same_batch",
            "deltas": [{"params": np.array([1.0])} for _ in lambda_grid],
            "per_lambda": [{"lambda": float(lam)} for lam in lambda_grid],
            "gcv_y": np.array([1.0, 0.0]),
            "gcv_mu": np.array([1.0, 0.5]),
            "gcv_z": np.ones((2, len(lambda_grid))),
        },
    )
    monkeypatch.setattr(
        mdiag,
        "_indep_batch_repeat_payload",
        lambda driver, lambda_grid: {
            "batch_mode": "indep_batch",
            "deltas": [{"params": np.array([2.0])} for _ in lambda_grid],
            "per_lambda": [{"lambda": float(lam)} for lam in lambda_grid],
        },
    )

    payload = mdiag.precompute_mixture_bank(
        ckpt_path,
        system_name="j1j2",
        repeat_count=2,
        lambda_grid=lambda_grid,
    )

    assert payload["repeat_count"] == 2
    assert payload["lambda_grid"] == [1e-3, 1e-4]
    assert len(payload["same_batch"]["repeats"]) == 2
    assert len(payload["indep_batch"]["repeats"]) == 2
    assert mdiag.mixture_bank_output_path(ckpt_path, 100).exists()


def test_evaluate_mixture_bank_uses_default_weight_fit_split_and_writes_all_methods(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "results" / "checkpoints" / "checkpoint_step000100.pkl"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_bytes(b"checkpoint")
    lambda_grid = (1e-3, 1e-4)
    bank_payload = _dummy_bank_payload(lambda_grid)

    monkeypatch.setattr(
        mdiag,
        "_load_training_driver",
        lambda ckpt_path, system_name: {
            "system": system_name,
            "config": {"diag_shift": 1e-3, "n_samples": 8},
            "step": 100,
            "driver": object(),
        },
    )
    monkeypatch.setattr(mdiag, "precompute_mixture_bank", lambda *args, **kwargs: bank_payload)
    monkeypatch.setattr(
        mdiag,
        "_method_weights",
        lambda bank_payload, method_id, **kwargs: [np.array([0.5, 0.5]), np.array([0.5, 0.5])],
    )
    monkeypatch.setattr(
        mdiag,
        "_evaluate_method",
        lambda ckpt_path, system_name, bank_payload, method_id, weights_per_repeat, eval_samples, validation_round_samples, jvp_chunk: {
            "method_id": method_id,
            "r_val_norm": 1.0,
            "var_hat_norm": 0.25,
            "weights_per_repeat": [[float(x) for x in weights] for weights in weights_per_repeat],
        },
    )

    payload = mdiag.evaluate_mixture_bank(
        ckpt_path,
        system_name="j1j2",
        repeat_count=2,
        lambda_grid=lambda_grid,
        n_val=40,
        weight_fit_samples=None,
    )

    assert payload["split_sizes"] == {"b_weight": 8, "c_eval": 32}
    assert set(payload["methods"]) == set(mdiag.MIXTURE_METHOD_IDS)
    assert mdiag.mixture_eval_output_path(ckpt_path, 100).exists()


def test_evaluate_mixture_bank_enforces_tfim_split_divisibility(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "results" / "checkpoints" / "checkpoint_step000100.pkl"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_bytes(b"checkpoint")
    lambda_grid = (1e-3, 1e-4)

    monkeypatch.setattr(
        mdiag,
        "_load_training_driver",
        lambda ckpt_path, system_name: {
            "system": system_name,
            "config": {
                "diag_shift": 1e-3,
                "n_samples": 8,
                "validation_h_count": 4,
            },
            "step": 100,
            "driver": object(),
        },
    )
    monkeypatch.setattr(mdiag, "precompute_mixture_bank", lambda *args, **kwargs: _dummy_bank_payload(lambda_grid))

    with pytest.raises(ValueError, match="validation_h_count"):
        mdiag.evaluate_mixture_bank(
            ckpt_path,
            system_name="tfim",
            repeat_count=2,
            lambda_grid=lambda_grid,
            n_val=22,
            weight_fit_samples=8,
        )


def test_indep_batch_gcv_weights_are_fit_from_same_batch_pilot(monkeypatch):
    lambda_grid = (1e-3, 1e-4)
    bank_payload = _dummy_bank_payload(lambda_grid)
    calls = []

    def fake_gcv(y, z, mu, lambdas):
        calls.append((y.copy(), z.copy(), mu.copy(), lambdas.copy()))
        return np.array([0.2, 0.8])

    monkeypatch.setattr(mdiag, "compute_gcv_mixture_weights", fake_gcv)

    weights = mdiag._method_weights(
        bank_payload,
        "indep_batch_gcv",
        ckpt_path="unused",
        system_name="j1j2",
        weight_fit_samples=8,
        validation_round_samples=None,
        jvp_chunk=None,
    )

    assert len(weights) == 2
    assert all(np.allclose(weight, np.array([0.2, 0.8])) for weight in weights)
    assert len(calls) == 2


def _dummy_shared_bank_payload(lambda_grid, *, oracle_lambda=1e-4, group_size=4):
    return {
        "system": "j1j2",
        "step": 100,
        "lambda": 1e-4,
        "source_training_lambda": 1e-4,
        "lambda_grid": [float(x) for x in lambda_grid],
        "repeat_count": 2,
        "oracle_lambda": float(oracle_lambda),
        "same_batch": {
            "batch_mode": "same_batch",
            "repeats": [
                {
                    "deltas": [{"params": np.array([1.0])}, {"params": np.array([2.0])}],
                    "gcv_y": np.array([1.0, 0.0]),
                    "gcv_mu": np.array([1.0, 0.5]),
                    "gcv_z": np.array([[1.0, 0.0], [0.0, 1.0]]),
                    "per_lambda": [],
                }
                for _ in range(2)
            ],
        },
        "indep_batch": {
            "batch_mode": "indep_batch",
            "repeats": [
                {
                    "deltas": [{"params": np.array([1.0])}, {"params": np.array([2.0])}],
                    "per_lambda": [],
                }
                for _ in range(2)
            ],
        },
        "single_oracle": {
            "batch_mode": "single_oracle",
            "lambda_value": float(oracle_lambda),
            "repeats": [
                {
                    "deltas": [{"params": np.array([3.0])}],
                    "per_lambda": [{"lambda": float(oracle_lambda)}],
                }
                for _ in range(2)
            ],
        },
        "indep_single_lambda": {
            "batch_mode": "indep_single_lambda",
            "lambda_value": float(oracle_lambda),
            "group_size": int(group_size),
            "repeats": [
                {
                    "deltas": [{"params": np.array([1.0])} for _ in range(group_size)],
                    "per_lambda": [{"lambda": float(oracle_lambda)} for _ in range(group_size)],
                }
                for _ in range(2)
            ],
        },
    }


def test_parse_mixture_method_id_supports_shared_checkpoint_methods():
    single_spec = mdiag.parse_mixture_method_id("single_oracle")
    indep_single_spec = mdiag.parse_mixture_method_id("indep_single_lambda_k4")
    same_batch_spec = mdiag.parse_mixture_method_id("same_batch_multilambda_gcv_k4")
    indep_multi_spec = mdiag.parse_mixture_method_id("indep_multilambda_uniform_k4")

    assert single_spec["branch"] == "single_oracle"
    assert indep_single_spec["branch"] == "indep_single_lambda"
    assert indep_single_spec["k"] == 4
    assert same_batch_spec["branch"] == "same_batch"
    assert same_batch_spec["weight_scheme"] == "gcv"
    assert indep_multi_spec["branch"] == "indep_batch"
    assert indep_multi_spec["weight_scheme"] == "uniform"


def test_required_mixture_branches_include_same_batch_pilot_for_indep_gcv():
    branches = mdiag.required_mixture_branches(("indep_multilambda_gcv_k4",))
    assert branches == ("indep_batch", "same_batch")


def test_method_weights_support_single_oracle_and_indep_single_lambda():
    bank_payload = _dummy_shared_bank_payload((1e-3, 1e-4, 1e-5, 1e-6), group_size=4)

    single_weights = mdiag._method_weights(
        bank_payload,
        "single_oracle",
        ckpt_path="unused",
        system_name="j1j2",
        weight_fit_samples=0,
        validation_round_samples=None,
        jvp_chunk=None,
    )
    indep_single_weights = mdiag._method_weights(
        bank_payload,
        "indep_single_lambda_k4",
        ckpt_path="unused",
        system_name="j1j2",
        weight_fit_samples=0,
        validation_round_samples=None,
        jvp_chunk=None,
    )

    assert len(single_weights) == 2
    assert all(np.allclose(weight, np.array([1.0])) for weight in single_weights)
    assert len(indep_single_weights) == 2
    assert all(np.allclose(weight, np.full(4, 0.25)) for weight in indep_single_weights)


def test_evaluate_mixture_bank_uses_full_eval_budget_without_stacking(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "results" / "checkpoints" / "checkpoint_step000100.pkl"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_bytes(b"checkpoint")
    lambda_grid = (1e-2, 1e-3, 1e-4, 1e-5)
    bank_payload = _dummy_shared_bank_payload(lambda_grid, group_size=4)
    method_ids = (
        "single_oracle",
        "indep_single_lambda_k4",
        "same_batch_multilambda_gcv_k4",
        "indep_multilambda_gcv_k4",
    )

    monkeypatch.setattr(
        mdiag,
        "_load_training_driver",
        lambda ckpt_path, system_name: {
            "system": system_name,
            "config": {"diag_shift": 1e-4, "n_samples": 8},
            "step": 100,
            "driver": object(),
        },
    )
    monkeypatch.setattr(mdiag, "precompute_mixture_bank", lambda *args, **kwargs: bank_payload)
    monkeypatch.setattr(
        mdiag,
        "_method_weights",
        lambda bank_payload, method_id, **kwargs: [np.array([1.0]), np.array([1.0])],
    )
    monkeypatch.setattr(
        mdiag,
        "_evaluate_method",
        lambda ckpt_path, system_name, bank_payload, method_id, weights_per_repeat, eval_samples, validation_round_samples, jvp_chunk: {
            "method_id": method_id,
            "eval_samples": int(eval_samples),
            "r_val_raw": 1.0,
            "var_hat_raw": 0.25,
        },
    )

    payload = mdiag.evaluate_mixture_bank(
        ckpt_path,
        system_name="j1j2",
        repeat_count=2,
        lambda_grid=lambda_grid,
        n_val=40,
        weight_fit_samples=None,
        method_ids=method_ids,
        oracle_lambda=1e-4,
        single_lambda_count=4,
    )

    assert payload["split_sizes"] == {"b_weight": 0, "c_eval": 40}
    assert tuple(payload["methods"]) == method_ids
