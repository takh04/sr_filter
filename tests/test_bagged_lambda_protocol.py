import numpy as np
import pytest

from sr_filtering_nqs.large_scale.core.bagged_lambda_protocol import (
    DEFAULT_BAGGED_LAMBDAS,
    select_bagged_lambda,
)


def _method(shift, residuals, variance):
    return {
        "bagged_lambda": shift,
        "r_val_raw_per_repeat": residuals,
        "r_val_raw": float(np.mean(residuals)),
        "repeat_count": len(residuals),
        "var_hat_raw": variance,
    }


def test_bagged_selection_uses_mean_residual_and_keeps_selected_variance():
    methods = {
        "training_shift": _method(1e-4, [1.0, 9.0], 0.1),
        "best_bagged": _method(1e-3, [3.0, 3.0], 2.0),
        "single_sr_selected": _method(1e-1, [6.0, 4.0], 0.01),
    }
    selected = select_bagged_lambda(methods)
    assert selected["method_id"] == "best_bagged"
    assert selected["bagged_lambda"] == 1e-3
    assert selected["r_val_raw"] == 3.0
    # Variance is reported at the residual-minimizing shift, not minimized again.
    assert selected["var_hat_raw"] == 2.0
    assert DEFAULT_BAGGED_LAMBDAS == tuple(10.0**power for power in range(-8, 0))


def test_bagged_selection_rejects_incomplete_or_stale_metrics():
    method = _method(1e-4, [1.0, 2.0], 0.1)
    method["repeat_count"] = 3
    with pytest.raises(ValueError, match="Incomplete"):
        select_bagged_lambda({"candidate": method})
    method["repeat_count"] = 2
    method["r_val_raw"] = 0.5
    with pytest.raises(ValueError, match="disagrees"):
        select_bagged_lambda({"candidate": method})
