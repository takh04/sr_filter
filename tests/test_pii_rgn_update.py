from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from sr_filtering_nqs.large_scale.core.pii_update import (
    compute_energy_shifted_ntk_matrix,
    compute_rgn_ntk_matrix,
    current_energy_tau,
)


def test_current_energy_tau_uses_weighted_local_energy_mean():
    local_loss = jnp.asarray([1.0, 3.0, 7.0])
    weights = jnp.asarray([1.0, 2.0, 1.0])

    assert current_energy_tau(local_loss, weights) == 3.5


def test_rgn_matrix_matches_pii_matrix_at_current_energy_shift():
    ntk = jnp.asarray(
        [
            [2.0, 0.5, 0.25],
            [0.5, 3.0, 0.75],
            [0.25, 0.75, 4.0],
        ]
    )
    local_loss = jnp.asarray([1.0, 4.0, 7.0])
    weights = jnp.asarray([1.0, 1.0, 2.0])
    tau = current_energy_tau(local_loss, weights)

    rgn_matrix = compute_rgn_ntk_matrix(
        ntk,
        local_loss,
        weights=weights,
        diag_shift=1e-3,
    )
    pii_matrix = compute_energy_shifted_ntk_matrix(
        ntk,
        local_loss,
        tau=tau,
        diag_shift=1e-3,
    )

    assert np.allclose(np.asarray(rgn_matrix), np.asarray(pii_matrix))
