from __future__ import annotations

import csv
import pickle
from types import SimpleNamespace

import pytest

from sr_filtering_nqs.large_scale.cli.train_sweep import (
    TrainMetricsCallback,
    load_resume_state,
    preserve_resume_contract,
)


def test_load_resume_state_retains_checkpoint_contract(tmp_path):
    path = tmp_path / "checkpoint.pkl"
    payload = {
        "step": 25,
        "phase_index": 0,
        "parameters": {"w": 1},
        "variables": {"params": {"w": 1}},
        "config": {
            "init_from": "/trusted/source.pkl",
            "init_from_step": 50,
            "sr_update_protocol": "fixed_grid_exact_loo",
        },
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)

    state = load_resume_state(str(path))

    assert state.step == 25
    assert state.source_step == 25
    assert state.checkpoint_config == payload["config"]


def test_preserve_resume_contract_keeps_original_source_provenance():
    requested = {
        "init_from": None,
        "init_from_step": None,
        "total_steps": 125,
        "sr_lambda_grid": [1e-2, 1e-3, 1e-4, 1e-5],
    }
    checkpoint = {
        **requested,
        "init_from": "/trusted/source.pkl",
        "init_from_step": 50,
    }

    preserved = preserve_resume_contract(requested, checkpoint)

    assert preserved["init_from"] == "/trusted/source.pkl"
    assert preserved["init_from_step"] == 50
    assert requested["init_from"] is None


def test_preserve_resume_contract_rejects_scientific_setting_change():
    requested = {
        "init_from": None,
        "init_from_step": None,
        "total_steps": 125,
        "sr_prediction_chunk": 2048,
    }
    checkpoint = {
        **requested,
        "init_from": "/trusted/source.pkl",
        "init_from_step": 50,
        "sr_prediction_chunk": 16,
    }

    with pytest.raises(ValueError, match="sr_prediction_chunk"):
        preserve_resume_contract(requested, checkpoint)


def test_resume_callback_uses_native_global_driver_step(tmp_path):
    callback = TrainMetricsCallback(
        results_dir=tmp_path,
        run_config={"n_sites": 2},
        phase_spec={"phase_index": 0, "name": "identity", "symmetry_index": 0},
        step_base=25,
        global_schedule=lambda step: step / 100,
        checkpoint_steps=(),
        existing_metric_steps=set(),
        saved_checkpoint_steps=[],
        latest_checkpoint_every=0,
        run_start_time=0.0,
    )
    driver = SimpleNamespace(
        _loss_stats=SimpleNamespace(Mean=-4.0, Variance=0.5, Sigma=0.1),
        _stop_run=False,
        info={
            "multilambda": {
                "train_metrics_policy": "omitted_to_avoid_nonprotocol_diagnostics"
            }
        },
    )

    # Advanced Drivers passes its native global counter (25), not a local
    # zero-based counter, to callbacks after a step-25 resume.
    callback.on_step_start(25, {}, driver)
    callback.on_compute_update_end(25, {}, driver)
    callback.on_step_end(25, {}, driver)

    assert callback.latest_metrics["step"] == 26
    with (tmp_path / "train_metrics.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert int(row["step"]) == 26
    assert float(row["learning_rate"]) == pytest.approx(0.26)
