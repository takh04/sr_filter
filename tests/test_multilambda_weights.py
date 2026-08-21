from __future__ import annotations

import numpy as np

from sr_filtering_nqs.large_scale.core.multilambda_sr import (
    compute_gcv_mixture_weights,
    compute_stacking_mixture_weights,
)


def _assert_simplex(weights: np.ndarray) -> None:
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= -1e-10)
    assert np.isclose(np.sum(weights), 1.0, atol=1e-8)


def test_compute_gcv_mixture_weights_recovers_one_hot_solution():
    y = np.array([1.0, 0.0, 0.0])
    z_train = np.eye(3)
    mu = np.ones(3)
    lambdas = np.array([1e-6, 1e-3, 1.0])

    weights = compute_gcv_mixture_weights(y, z_train, mu, lambdas)

    _assert_simplex(weights)
    assert weights[0] > 0.999


def test_compute_gcv_mixture_weights_handles_near_interpolation_denominator():
    y = np.array([0.4, -0.2])
    z_train = np.array([[0.35, 0.2], [-0.15, -0.1]])
    mu = np.array([1e12, 1e12])
    lambdas = np.array([1e-12, 1e-9])

    weights = compute_gcv_mixture_weights(y, z_train, mu, lambdas)

    _assert_simplex(weights)


def test_compute_stacking_mixture_weights_recovers_one_hot_solution():
    y = np.array([0.0, 1.0, 0.0])
    z_train = np.eye(3)

    weights = compute_stacking_mixture_weights(y, z_train)

    _assert_simplex(weights)
    assert weights[1] > 0.999


def test_compute_stacking_mixture_weights_respects_simplex_for_generic_inputs():
    y = np.array([1.0, -0.5, 0.25, 0.75])
    z_train = np.array(
        [
            [1.0, 0.0, 0.2],
            [0.0, 1.0, -0.1],
            [0.2, 0.3, 1.0],
            [0.1, -0.2, 0.4],
        ]
    )

    weights = compute_stacking_mixture_weights(y, z_train)

    _assert_simplex(weights)
