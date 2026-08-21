from __future__ import annotations

import numpy as np
import jax.numpy as jnp

import sr_filtering_nqs.large_scale.core.multilambda_sr as mlsr


class _DummyState:
    def __init__(self):
        self.variables = {"params": np.array([1.0])}
        self._apply_fun = object()
        self.calls = []

    def samples_distribution(self, distribution, *, variables=None, chain_name=None):
        self.calls.append(
            {
                "method": "samples_distribution",
                "distribution": distribution,
                "variables": variables,
                "chain_name": chain_name,
            }
        )
        return np.array([[1.0]])

    def sample_distribution(self, distribution, *, variables=None, n_samples=None, chain_name=None):
        self.calls.append(
            {
                "method": "sample_distribution",
                "distribution": distribution,
                "variables": variables,
                "n_samples": n_samples,
                "chain_name": chain_name,
            }
        )
        return np.array([[2.0]])


class _DummyDriver:
    def __init__(self):
        self.state = _DummyState()


def test_uniform_sample_weights_remain_real_for_complex_local_estimators():
    weights = mlsr._weights_for_samples(
        np.zeros((5, 2)),
        dtype=jnp.complex128,
    )

    assert weights.shape == (5,)
    assert weights.dtype == jnp.float64
    assert np.array_equal(np.asarray(weights), np.ones(5))


def test_sample_prepared_batch_uses_cached_distribution_when_n_samples_unspecified(monkeypatch):
    driver = _DummyDriver()

    monkeypatch.setattr(
        mlsr,
        "_prepared_batch_from_samples",
        lambda driver, samples, *, variables=None: {
            "samples": samples,
            "variables": variables,
        },
    )

    payload = mlsr.sample_prepared_batch(driver, chain_name="same_batch")

    assert driver.state.calls == [
        {
            "method": "samples_distribution",
            "distribution": driver.state._apply_fun,
            "variables": driver.state.variables,
            "chain_name": "same_batch",
        }
    ]
    assert np.array_equal(payload["samples"], np.array([[1.0]]))
    assert payload["variables"] is driver.state.variables


def test_sample_prepared_batch_uses_sample_distribution_for_explicit_n_samples(monkeypatch):
    driver = _DummyDriver()

    monkeypatch.setattr(
        mlsr,
        "_prepared_batch_from_samples",
        lambda driver, samples, *, variables=None: {
            "samples": samples,
            "variables": variables,
        },
    )

    payload = mlsr.sample_prepared_batch(driver, n_samples=17, chain_name="pilot")

    assert driver.state.calls == [
        {
            "method": "sample_distribution",
            "distribution": driver.state._apply_fun,
            "variables": driver.state.variables,
            "n_samples": 17,
            "chain_name": "pilot",
        }
    ]
    assert np.array_equal(payload["samples"], np.array([[2.0]]))
    assert payload["variables"] is driver.state.variables


class _FallbackDriver:
    def __init__(self):
        self.step_count = 0
        self.proj_reg = None
        self.momentum = None
        self._linear_solver_fn = object()
        self.mode = "real"
        self._old_updates = None
        self.chunk_size_bwd = None
        self.calls = []

    def update_fn(self, *args, **kwargs):
        if "importance_sampling_weights" in kwargs:
            raise TypeError("srt_onthefly() got an unexpected keyword argument 'importance_sampling_weights'")
        self.calls.append(kwargs)
        return {"params": jnp.asarray([1.0])}, None, {"used_pdf": kwargs.get("pdf")}


def test_solve_sr_on_batch_falls_back_to_pdf_when_driver_rejects_importance_weights(monkeypatch):
    driver = _FallbackDriver()
    batch = mlsr.PreparedBatch(
        afun=object(),
        variables={"params": jnp.asarray([2.0])},
        samples=jnp.asarray([[0.0], [1.0]]),
        weights=jnp.asarray([0.25, 0.75]),
        extra_args=(),
        local_grad=jnp.asarray([1.0, 2.0]),
        local_loss=jnp.asarray([3.0, 5.0]),
        n_replicas=None,
    )

    monkeypatch.setattr(mlsr, "statistics", lambda values: {"mean": float(np.asarray(values).mean())})
    monkeypatch.setattr(mlsr.jax, "block_until_ready", lambda value: value)

    delta, loss_stats, info = mlsr.solve_sr_on_batch(
        driver,
        batch,
        diag_shift=1e-4,
        collect_residual_info=False,
    )

    assert np.allclose(np.asarray(delta["params"]), np.array([1.0]))
    assert loss_stats == {"mean": 2.25}
    assert len(driver.calls) == 1
    assert np.allclose(
        np.asarray(driver.calls[0]["pdf"]),
        np.array([0.125, 0.375]),
    )
    assert np.allclose(np.asarray(info["used_pdf"]), np.array([0.125, 0.375]))


class _DoubleFallbackDriver:
    def __init__(self):
        self.step_count = 0
        self.proj_reg = None
        self.momentum = None
        self._linear_solver_fn = object()
        self.mode = "real"
        self._old_updates = None
        self.chunk_size_bwd = None
        self.calls = []

    def update_fn(self, *args, **kwargs):
        if "importance_sampling_weights" in kwargs:
            raise TypeError("srt_onthefly() got an unexpected keyword argument 'importance_sampling_weights'")
        if "pdf" in kwargs:
            raise TypeError("srt_onthefly() got an unexpected keyword argument 'pdf'")
        self.calls.append(kwargs)
        return {"params": jnp.asarray([2.0])}, None, {"used_unweighted": True}


def test_solve_sr_on_batch_falls_back_to_unweighted_call_when_driver_rejects_all_weight_kwargs(monkeypatch):
    driver = _DoubleFallbackDriver()
    batch = mlsr.PreparedBatch(
        afun=object(),
        variables={"params": jnp.asarray([2.0])},
        samples=jnp.asarray([[0.0], [1.0]]),
        weights=jnp.asarray([0.25, 0.75]),
        extra_args=(),
        local_grad=jnp.asarray([1.0, 2.0]),
        local_loss=jnp.asarray([3.0, 5.0]),
        n_replicas=None,
    )

    monkeypatch.setattr(mlsr, "statistics", lambda values: {"mean": float(np.asarray(values).mean())})
    monkeypatch.setattr(mlsr.jax, "block_until_ready", lambda value: value)

    delta, loss_stats, info = mlsr.solve_sr_on_batch(
        driver,
        batch,
        diag_shift=1e-4,
        collect_residual_info=False,
    )

    assert np.allclose(np.asarray(delta["params"]), np.array([2.0]))
    assert loss_stats == {"mean": 2.25}
    assert len(driver.calls) == 1
    assert "importance_sampling_weights" not in driver.calls[0]
    assert "pdf" not in driver.calls[0]
    assert info == {"used_unweighted": True}


class _ReplicaFallbackDriver:
    def __init__(self):
        self.step_count = 0
        self.proj_reg = None
        self.momentum = None
        self._linear_solver_fn = object()
        self.mode = "real"
        self._old_updates = None
        self.chunk_size_bwd = None
        self.calls = []

    def update_fn(self, *args, **kwargs):
        if "importance_sampling_weights" in kwargs:
            raise TypeError("srt_onthefly() got an unexpected keyword argument 'importance_sampling_weights'")
        if "pdf" in kwargs:
            raise TypeError("srt_onthefly() got an unexpected keyword argument 'pdf'")
        if "n_replicas" not in kwargs:
            raise TypeError("srt_onthefly() missing 1 required keyword-only argument: 'n_replicas'")
        self.calls.append(kwargs)
        return {"params": jnp.asarray([4.0])}, None, {"used_n_replicas": kwargs["n_replicas"]}


def test_solve_sr_on_batch_passes_n_replicas_when_driver_requires_it(monkeypatch):
    driver = _ReplicaFallbackDriver()
    batch = mlsr.PreparedBatch(
        afun=object(),
        variables={"params": jnp.asarray([2.0])},
        samples=jnp.asarray([[0.0], [1.0]]),
        weights=jnp.asarray([0.25, 0.75]),
        extra_args=(),
        local_grad=jnp.asarray([1.0, 2.0]),
        local_loss=jnp.asarray([3.0, 5.0]),
        n_replicas=6000,
    )

    monkeypatch.setattr(mlsr, "statistics", lambda values: {"mean": float(np.asarray(values).mean())})
    monkeypatch.setattr(mlsr.jax, "block_until_ready", lambda value: value)

    delta, loss_stats, info = mlsr.solve_sr_on_batch(
        driver,
        batch,
        diag_shift=1e-4,
        collect_residual_info=False,
    )

    assert np.allclose(np.asarray(delta["params"]), np.array([4.0]))
    assert loss_stats == {"mean": 2.25}
    assert len(driver.calls) == 1
    assert driver.calls[0]["n_replicas"] == 6000
    assert "importance_sampling_weights" not in driver.calls[0]
    assert "pdf" not in driver.calls[0]
    assert info == {"used_n_replicas": 6000}
