"""Complex tangent predictions must use the same channel layout as SR targets."""

import jax.numpy as jnp
import numpy as np
import pytest

from sr_filtering_nqs.large_scale.core.sr_diagnostics import (
    compute_O_delta_jvp,
    compute_O_delta_jvp_many,
)


def _complex_linear_model(variables, samples):
    weights = variables["params"]["weights"]
    return samples @ weights + 1j * (samples**2) @ weights


@pytest.mark.parametrize("mode", ["real", "complex"])
@pytest.mark.parametrize("chunk_size", [None, 1, 2, 3, 7])
def test_jvp_chunking_matches_analytic_complex_tangent(mode, chunk_size):
    samples = jnp.arange(14, dtype=float).reshape(7, 2) / 10
    params = {"weights": jnp.asarray([0.3, -0.2])}
    directions = jnp.asarray([[0.2, 0.8], [-0.7, 0.1]])
    analytic = np.asarray(samples) @ np.asarray(directions).T
    if mode == "complex":
        imaginary = np.asarray(samples**2) @ np.asarray(directions).T
        analytic = np.concatenate([analytic, imaginary], axis=0)

    bank = compute_O_delta_jvp_many(
        _complex_linear_model,
        params,
        {},
        samples,
        {"weights": directions},
        mode=mode,
        chunk_size=chunk_size,
    )
    np.testing.assert_allclose(bank, analytic.T, rtol=1e-6, atol=1e-7)
    for index, direction in enumerate(directions):
        single = compute_O_delta_jvp(
            _complex_linear_model,
            params,
            {},
            samples,
            {"weights": direction},
            mode=mode,
            chunk_size=chunk_size,
        )
        np.testing.assert_allclose(single, analytic[:, index], rtol=1e-6, atol=1e-7)
