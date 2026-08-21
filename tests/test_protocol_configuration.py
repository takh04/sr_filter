from __future__ import annotations

import pytest

from sr_filtering_nqs.large_scale.core.fixed_data_protocol import (
    fig3_method_ids,
    resolve_fig3_lambda_quantiles,
    validate_fig3_lambda_quantiles,
)
from sr_filtering_nqs.large_scale.core.optimized_sr_benchmark import validate_quantiles


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
def test_default_quantile_grid_supports_general_k(k):
    quantiles = resolve_fig3_lambda_quantiles(k)
    assert len(quantiles) == k
    assert all(left > right for left, right in zip(quantiles, quantiles[1:]))
    assert f"same_batch_multilambda_uniform_k{k}" in fig3_method_ids(k)
    assert f"indep_multilambda_stacking_k{k}" in fig3_method_ids(k)


@pytest.mark.parametrize(
    "k, quantiles",
    [
        (1, (0.5,)),
        (3, (0.9, 0.1)),
        (3, (0.9, 0.9, 0.1)),
        (3, (1.1, 0.5, 0.1)),
    ],
)
def test_invalid_quantile_grids_are_rejected(k, quantiles):
    with pytest.raises(ValueError):
        validate_fig3_lambda_quantiles(k, quantiles)


def test_optimized_quantile_validation_matches_k():
    assert validate_quantiles(3, (0.9, 0.5, 0.1)) == (0.9, 0.5, 0.1)
    with pytest.raises(ValueError):
        validate_quantiles(4, (0.9, 0.5, 0.1))

