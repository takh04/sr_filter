from __future__ import annotations

import numpy as np

from sr_filtering_nqs.large_scale.core.multilambda_sr import weighted_tree_sum


def _ridge_delta(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n_samples, n_features = x.shape
    gram = (x.T @ x) / float(n_samples)
    rhs = (x.T @ y) / float(n_samples)
    return np.linalg.solve(gram + float(lam) * np.eye(n_features), rhs)


def test_same_batch_one_hot_mixture_reproduces_single_lambda_update():
    x = np.diag([3.0, 1.5, 0.5])
    y = np.array([1.0, -2.0, 0.5])
    lambdas = (1e-2, 1.0)
    deltas = [{"params": _ridge_delta(x, y, lam)} for lam in lambdas]
    weights = np.array([1.0, 0.0])

    mixed_delta = weighted_tree_sum(deltas, weights)
    mixed_prediction = x @ np.asarray(mixed_delta["params"])
    weighted_prediction = sum(weight * (x @ delta["params"]) for weight, delta in zip(weights, deltas, strict=True))

    assert np.allclose(mixed_delta["params"], deltas[0]["params"])
    assert np.allclose(mixed_prediction, weighted_prediction)


def test_same_batch_mixture_changes_spectral_filter_shape():
    spectrum = np.array([0.05, 1.0, 20.0])
    lambdas = np.array([1e-2, 10.0])
    weights = np.array([0.4, 0.6])
    mixture_filter = np.sum(weights[None, :] * (spectrum[:, None] / (spectrum[:, None] + lambdas[None, :])), axis=1)

    candidate_lambdas = np.logspace(-4, 3, 2000)
    best_single_error = min(
        np.max(np.abs(mixture_filter - spectrum / (spectrum + lam)))
        for lam in candidate_lambdas
    )

    assert best_single_error > 1e-2


def test_independent_batch_averaging_reduces_empirical_prediction_variance():
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(30, 6))
    x_test = rng.normal(size=(40, 6))
    beta_star = rng.normal(size=6)
    lambdas = (0.05, 0.5)
    weights = np.array([0.5, 0.5])
    n_trials = 300
    sigma = 1.0

    prediction_trials = {lam: [] for lam in lambdas}
    ensemble_trials = []
    for _ in range(n_trials):
        per_lambda_predictions = []
        for lam in lambdas:
            noise = rng.normal(scale=sigma, size=x_train.shape[0])
            y = x_train @ beta_star + noise
            delta = _ridge_delta(x_train, y, lam)
            predictions = x_test @ delta
            prediction_trials[lam].append(predictions)
            per_lambda_predictions.append(predictions)
        ensemble_trials.append(sum(weight * pred for weight, pred in zip(weights, per_lambda_predictions, strict=True)))

    base_variances = []
    for lam in lambdas:
        preds = np.asarray(prediction_trials[lam])
        base_variances.append(float(np.var(preds, axis=0, ddof=1).mean()))
    ensemble_variance = float(np.var(np.asarray(ensemble_trials), axis=0, ddof=1).mean())

    assert ensemble_variance < min(base_variances)
