from __future__ import annotations

import numpy as np

import sr_filtering_nqs.large_scale.core.multilambda_sr as ml
from sr_filtering_nqs.large_scale.core.multilambda_sr import MultiLambdaSRCallback


class _DummyDriver:
    def __init__(self):
        self._dp = {"params": np.array([1.0])}
        self.info = {}


def test_same_batch_gcv_callback_rewrites_driver_update(monkeypatch):
    driver = _DummyDriver()
    mixed_delta = {"params": np.array([2.0])}
    metadata = {
        "variant": "same_batch",
        "weight_scheme": "gcv",
        "lambda_grid": [1e-3, 1e-4],
        "lambda_source": "static",
        "weights": [0.3, 0.7],
        "train_metrics": {"residual_raw": 1.0},
    }
    monkeypatch.setattr(
        ml,
        "compute_multilambda_update",
        lambda driver, **kwargs: (mixed_delta, metadata),
    )

    callback = MultiLambdaSRCallback(
        sr_variant="same_batch",
        sr_weight_scheme="gcv",
        sr_lambda_grid=(1e-3, 1e-4),
    )
    log_data = {}
    callback.on_compute_update_end(0, log_data, driver)

    assert driver._dp == mixed_delta
    assert driver.info["multilambda"] == metadata
    assert log_data["multilambda_variant"] == "same_batch"
    assert log_data["multilambda_weight_scheme"] == "gcv"


def test_indep_uniform_callback_rewrites_driver_update(monkeypatch):
    driver = _DummyDriver()
    mixed_delta = {"params": np.array([3.0])}
    metadata = {
        "variant": "indep_batch",
        "weight_scheme": "uniform",
        "lambda_grid": [1e-3, 1e-4],
        "lambda_source": "static",
        "weights": [0.5, 0.5],
        "train_metrics": {"residual_raw": 0.5},
    }
    monkeypatch.setattr(
        ml,
        "compute_multilambda_update",
        lambda driver, **kwargs: (mixed_delta, metadata),
    )

    callback = MultiLambdaSRCallback(
        sr_variant="indep_batch",
        sr_weight_scheme="uniform",
        sr_lambda_grid=(1e-3, 1e-4),
    )
    callback.on_compute_update_end(0, {}, driver)

    assert driver._dp == mixed_delta
    assert driver.info["multilambda"]["variant"] == "indep_batch"


def test_indep_batch_gcv_callback_rewrites_driver_update(monkeypatch):
    driver = _DummyDriver()
    mixed_delta = {"params": np.array([4.0])}
    metadata = {
        "variant": "indep_batch",
        "weight_scheme": "gcv",
        "lambda_grid": [1e-3, 1e-4],
        "lambda_source": "static",
        "weights": [0.25, 0.75],
        "train_metrics": {"residual_raw": 0.25},
    }
    monkeypatch.setattr(
        ml,
        "compute_multilambda_update",
        lambda driver, **kwargs: (mixed_delta, metadata),
    )

    callback = MultiLambdaSRCallback(
        sr_variant="indep_batch",
        sr_weight_scheme="gcv",
        sr_lambda_grid=(1e-3, 1e-4),
    )
    callback.on_compute_update_end(0, {}, driver)

    assert driver._dp == mixed_delta
    assert driver.info["multilambda"]["weight_scheme"] == "gcv"


def test_single_variant_callback_is_noop():
    driver = _DummyDriver()
    original_dp = driver._dp
    callback = MultiLambdaSRCallback(
        sr_variant="single",
        sr_weight_scheme="uniform",
        sr_lambda_grid=(1e-3, 1e-4),
    )
    callback.on_compute_update_end(0, {}, driver)

    assert driver._dp == original_dp
    assert driver.info == {}


def test_callback_forwards_prediction_chunk(monkeypatch):
    driver = _DummyDriver()
    captured = {}

    def compute(driver, **kwargs):
        captured.update(kwargs)
        return driver._dp, {
            "variant": "same_batch",
            "weight_scheme": "uniform",
            "lambda_grid": [1e-3, 1e-4],
            "lambda_source": "static",
            "weights": [0.5, 0.5],
        }

    monkeypatch.setattr(ml, "compute_multilambda_update", compute)
    callback = MultiLambdaSRCallback(
        sr_variant="same_batch",
        sr_weight_scheme="uniform",
        sr_lambda_grid=(1e-3, 1e-4),
        prediction_chunk=37,
    )

    callback.on_compute_update_end(0, {}, driver)

    assert captured["prediction_chunk"] == 37


def test_callback_rejects_nonpositive_prediction_chunk():
    import pytest

    with pytest.raises(ValueError):
        MultiLambdaSRCallback(
            sr_variant="same_batch",
            sr_weight_scheme="uniform",
            sr_lambda_grid=(1e-3, 1e-4),
            prediction_chunk=0,
        )


def test_checkpoint_helpers_surface_multilambda_metadata():
    from sr_filtering_nqs.large_scale.cli.train_sweep import _multilambda_checkpoint_info as j1j2_checkpoint_info
    from sr_filtering_nqs.large_scale.cli.train_tfim_fnqs import _multilambda_checkpoint_info as tfim_checkpoint_info

    driver = _DummyDriver()
    driver.info = {
        "multilambda": {
            "variant": "same_batch",
            "weight_scheme": "gcv",
            "weights": [0.4, 0.6],
            "lambda_grid": [1e-3, 1e-4],
        }
    }

    assert j1j2_checkpoint_info(driver)["weights"] == [0.4, 0.6]
    assert tfim_checkpoint_info(driver)["lambda_grid"] == [1e-3, 1e-4]
