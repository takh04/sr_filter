from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import sr_filtering_nqs.large_scale.core.online_mssr as online_mssr
import sr_filtering_nqs.large_scale.core.optimized_sr_benchmark as benchmark
from sr_filtering_nqs.large_scale.cli.train_sweep import (
    TrainMetricsCallback,
    write_exact_optimizer_update_audit,
)
from sr_filtering_nqs.large_scale.core.multilambda_sr import PreparedBatch

GRID = (1e-2, 1e-3, 1e-4, 1e-5)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _batch(index: int) -> PreparedBatch:
    return PreparedBatch(
        afun=object(),
        variables={"params": jnp.asarray([1.0])},
        samples=jnp.asarray([[float(index)], [float(index + 1)]]),
        weights=jnp.ones((2,)),
        extra_args=(),
        local_grad=jnp.asarray([1.0, 2.0]),
        local_loss=jnp.asarray([float(index), float(index + 1)]),
        n_replicas=None,
    )


def test_fixed_grid_update_draws_and_solves_exactly_k_batches(monkeypatch):
    sample_calls = []
    solve_calls = []
    loo_calls = []

    def sample_prepared_batch(driver, *, n_samples, chain_name):
        sample_calls.append((driver, n_samples, chain_name))
        return _batch(len(sample_calls))

    def solve_fixed_grid_candidate(driver, batch, *, diag_shift):
        solve_calls.append((driver, batch, diag_shift))
        index = len(solve_calls)
        return {"params": jnp.asarray([float(index)])}, {}

    def exact_loo(driver, batches, deltas, *, prediction_chunk):
        loo_calls.append((driver, tuple(batches), tuple(deltas), prediction_chunk))
        return np.asarray([0.1, 0.2, 0.3, 0.4]), {
            "cross_prediction_s": 0.5,
            "simplex_fit_s": 0.1,
            "cross_prediction_launches": 4,
            "cross_prediction_directions": 12,
        }

    loss_stats = object()
    monkeypatch.setattr(benchmark, "sample_prepared_batch", sample_prepared_batch)
    monkeypatch.setattr(
        online_mssr,
        "solve_fixed_grid_candidate",
        solve_fixed_grid_candidate,
    )
    monkeypatch.setattr(online_mssr, "optimized_exact_loo_weights", exact_loo)
    monkeypatch.setattr(
        online_mssr,
        "_pooled_candidate_loss_statistics",
        lambda batches: loss_stats,
    )

    driver = SimpleNamespace(
        step_count=6,
        momentum=None,
        proj_reg=None,
        chunk_size_bwd=16,
    )
    actual_stats, delta, metadata = online_mssr.compute_fixed_grid_exact_loo_update(
        driver,
        lambda_grid=GRID,
        n_samples=2,
        prediction_chunk=16,
    )

    assert actual_stats is loss_stats
    assert len(sample_calls) == 4
    assert [call[1] for call in sample_calls] == [2] * 4
    assert [call[2] for call in sample_calls] == [
        "online_fixed_grid_mssr_0",
        "online_fixed_grid_mssr_1",
        "online_fixed_grid_mssr_2",
        "online_fixed_grid_mssr_3",
    ]
    assert len(solve_calls) == 4
    assert [call[2] for call in solve_calls] == list(GRID)
    assert len(loo_calls) == 1
    assert loo_calls[0][1] == tuple(call[1] for call in solve_calls)
    assert loo_calls[0][3] == 16
    np.testing.assert_allclose(delta["params"], np.asarray([3.0]))

    assert metadata["update_protocol"] == "fixed_grid_exact_loo"
    assert metadata["stacking_protocol"] == "exact_leave_one_batch_out"
    assert metadata["candidate_batches"] == 4
    assert metadata["candidate_solves"] == 4
    assert metadata["candidate_sample_counts"] == [2, 2, 2, 2]
    assert metadata["sampling_solve_units_this_update"] == 4
    assert metadata["sampling_solve_units_cumulative"] == 28
    assert metadata["base_driver_sampling_calls"] == 0
    assert metadata["base_driver_solves"] == 0
    assert metadata["weight_fit_sampling_calls"] == 0
    assert metadata["weight_fit_solves"] == 0
    assert metadata["cross_prediction_directions"] == 12


def test_exact_protocol_validation_forbids_diagnostic_extras():
    assert (
        online_mssr.validate_online_mssr_configuration(
            update_protocol="fixed_grid_exact_loo",
            sr_variant="indep_batch",
            sr_weight_scheme="stacking",
            sr_lambda_grid=GRID,
            sr_weight_fit_samples=None,
            sr_lambda_quantiles=None,
            sr_lambda_ema_rho=None,
            sr_weight_ema_rho=None,
            ngd_method="sr",
        )
        == GRID
    )

    with pytest.raises(ValueError, match="weight_fit_samples must be unset"):
        online_mssr.validate_online_mssr_configuration(
            update_protocol="fixed_grid_exact_loo",
            sr_variant="indep_batch",
            sr_weight_scheme="stacking",
            sr_lambda_grid=GRID,
            sr_weight_fit_samples=8192,
            sr_lambda_quantiles=None,
            sr_lambda_ema_rho=None,
            sr_weight_ema_rho=None,
            ngd_method="sr",
        )

    with pytest.raises(ValueError, match="fixed grid"):
        online_mssr.validate_online_mssr_configuration(
            update_protocol="fixed_grid_exact_loo",
            sr_variant="indep_batch",
            sr_weight_scheme="stacking",
            sr_lambda_grid=GRID,
            sr_weight_fit_samples=None,
            sr_lambda_quantiles=(0.9, 0.7, 0.4, 0.1),
            sr_lambda_ema_rho=None,
            sr_weight_ema_rho=None,
            ngd_method="sr",
        )


@pytest.mark.parametrize(
    ("n_replicas", "expected_distribution_key"),
    ((None, "importance_sampling_weights"), (4, "n_replicas")),
)
def test_candidate_solve_invokes_update_fn_once(n_replicas, expected_distribution_key):
    calls = []

    class Driver:
        _linear_solver_fn = object()
        mode = "real"
        proj_reg = None
        momentum = None
        step_count = 3
        _old_updates = None
        chunk_size_bwd = 8

        def update_fn(self, *args, **kwargs):
            calls.append((args, kwargs))
            return {"params": jnp.asarray([7.0])}, None, {"ok": True}

    batch = _batch(0)
    batch.n_replicas = n_replicas
    delta, info = online_mssr.solve_fixed_grid_candidate(
        Driver(),
        batch,
        diag_shift=1e-3,
    )

    assert len(calls) == 1
    assert expected_distribution_key in calls[0][1]
    unexpected_key = (
        "n_replicas"
        if expected_distribution_key == "importance_sampling_weights"
        else "importance_sampling_weights"
    )
    assert unexpected_key not in calls[0][1]
    assert calls[0][1]["diag_shift"] == 1e-3
    np.testing.assert_allclose(delta["params"], np.asarray([7.0]))
    assert info == {"ok": True}


def test_native_driver_method_does_not_call_base_compute(monkeypatch):
    expected_stats = object()
    expected_delta = {"params": jnp.asarray([2.0])}
    expected_metadata = {"candidate_batches": 4, "candidate_solves": 4}
    calls = []

    def compute(driver, **kwargs):
        calls.append((driver, kwargs))
        return expected_stats, expected_delta, expected_metadata

    monkeypatch.setattr(online_mssr, "compute_fixed_grid_exact_loo_update", compute)
    driver = SimpleNamespace(
        _mssr_lambda_grid=GRID,
        _mssr_n_samples=8192,
        _mssr_prediction_chunk=16,
        _mssr_chain_prefix="online_fixed_grid_mssr",
        _loss_stats=None,
        _dp=None,
        info=None,
    )

    actual_stats, actual_delta = (
        online_mssr.FixedGridExactLOOMSSRDriver.compute_loss_and_update.__wrapped__(driver)
    )

    assert calls == [
        (
            driver,
            {
                "lambda_grid": GRID,
                "n_samples": 8192,
                "prediction_chunk": 16,
                "chain_prefix": "online_fixed_grid_mssr",
            },
        )
    ]
    assert actual_stats is expected_stats
    assert actual_delta is expected_delta
    assert driver._loss_stats is expected_stats
    assert driver._dp is expected_delta
    assert driver.info == {"multilambda": expected_metadata}


def test_training_callback_does_not_sample_residual_batch_for_exact_protocol(tmp_path):
    class NoSamplesState:
        @property
        def samples(self):
            raise AssertionError("exact-budget callback attempted to sample a fifth batch")

    stats = SimpleNamespace(Mean=-10.0, Variance=2.0, Sigma=0.1)
    driver = SimpleNamespace(
        _loss_stats=stats,
        _stop_run=False,
        _dp={"params": jnp.asarray([1.0])},
        info={"multilambda": {"train_metrics_policy": "omitted_to_avoid_nonprotocol_diagnostics"}},
        state=NoSamplesState(),
    )
    callback = TrainMetricsCallback(
        results_dir=tmp_path,
        run_config={"n_sites": 5},
        phase_spec={"phase_index": 0, "name": "identity", "symmetry_index": 0},
        step_base=0,
        global_schedule=lambda step: 0.01,
        checkpoint_steps=(1,),
        existing_metric_steps=set(),
        saved_checkpoint_steps=[],
        latest_checkpoint_every=0,
        run_start_time=0.0,
    )

    callback.on_compute_update_end(0, {}, driver)

    assert callback.latest_metrics["step"] == 1
    assert np.isnan(callback.latest_metrics["r_train"])


def test_exact_checkpoint_writes_portable_hash_bound_json_audit(tmp_path):
    results = tmp_path / "results"
    checkpoint = results / "checkpoints/checkpoint_step000125.pkl"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"trusted-checkpoint-bytes")
    (results / "config.json").write_text('{"stable": true}\n', encoding="utf-8")
    (results / "train_metrics.csv").write_text("step,energy_per_site\n125,-1.99\n", encoding="utf-8")
    metadata = {
        "update_protocol": "fixed_grid_exact_loo",
        "candidate_batches": 4,
        "candidate_solves": 4,
        "sampling_solve_units_cumulative": 500,
        "base_driver_sampling_calls": 0,
        "weight_fit_sampling_calls": 0,
        "cross_prediction_directions": 12,
    }
    checkpoint_payload = {
        "step": 125,
        "config": {
            "system": "j1j2",
            "seed": 1,
            "sr_update_protocol": "fixed_grid_exact_loo",
            "sr_lambda_grid": list(GRID),
        },
        "optimizer_update": metadata,
    }

    audit_path = write_exact_optimizer_update_audit(results, checkpoint, checkpoint_payload)

    assert audit_path == checkpoint.with_suffix(".audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["schema"] == "sr_filter.online_optimizer_update_audit.v1"
    assert audit["checkpoint_filename"] == checkpoint.name
    assert audit["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert audit["checkpoint_step"] == 125
    assert audit["optimizer_update"] == metadata
    assert audit["run_contract"]["sr_lambda_grid"] == list(GRID)


@pytest.mark.parametrize(
    ("filename", "branch"),
    (
        ("online_j1j2_branch2500_seed0.toml", "branch_seed0"),
        ("online_j1j2_branch1800_seed1.toml", "branch_seed1"),
    ),
)
def test_frozen_online_commands_require_exact_native_driver(filename, branch):
    path = REPOSITORY_ROOT / "configs" / "experiments" / filename
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    ms_sr_run = next(run for run in payload["runs"] if run["arm"] == "ms_sr")

    assert payload["optimization"]["ms_sr_update_protocol"] == "fixed_grid_exact_loo"
    assert ms_sr_run["optimizer_steps"] == 125
    assert ms_sr_run["batches_per_update"] == 4
    assert ms_sr_run["output"] == f"outputs/gpu/online/{branch}/mssr_exact"
    assert "--sr-update-protocol fixed_grid_exact_loo" in ms_sr_run["command"]
    assert "--chunk-size 4096" in ms_sr_run["command"]
    assert payload["optimization"]["ms_sr_prediction_chunk"] == 2048
    assert "--sr-prediction-chunk 2048" in ms_sr_run["command"]
    assert "--sr-lambda-grid 1e-2,1e-3,1e-4,1e-5" in ms_sr_run["command"]
