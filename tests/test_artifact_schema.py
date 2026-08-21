from __future__ import annotations

import pytest

from sr_filtering_nqs.artifact_schema import (
    NORMALIZED_FIELDS,
    normalize_artifact_record,
)


def test_legacy_k4_record_normalizes_to_public_long_form():
    normalized = normalize_artifact_record(
        {
            "system": "TFIM",
            "method_id": "indep_multilambda_stacking_k4",
            "checkpoint_step": "2000",
            "branch_rng_seed": "3",
            "repeat_index": "2",
            "metric": "r_val_raw",
            "mean": "0.0068",
        }
    )

    assert tuple(normalized) == NORMALIZED_FIELDS
    assert normalized == {
        "method": "ms_sr",
        "lambda_policy": "ntk_quantile_grid_stacking",
        "k": 4,
        "system": "tfim",
        "checkpoint": "2000",
        "branch_seed": 3,
        "repeat": 2,
        "metric": "r_val_raw",
        "value": 0.0068,
    }


def test_current_wide_record_normalizes_selected_metric():
    normalized = normalize_artifact_record(
        {
            "system": "j1j2",
            "method": "bagged_sr",
            "lambda_policy": "fixed_1e-4_uniform",
            "k": 5,
            "source_checkpoint_id": "j1j2/step_001500",
            "repeat": 0,
            "r_val_raw": 1.25,
        },
        metric="r_val_raw",
    )

    assert normalized["method"] == "bagged_sr"
    assert normalized["lambda_policy"] == "fixed_1e-4_uniform"
    assert normalized["k"] == 5
    assert normalized["checkpoint"] == "j1j2/step_001500"
    assert normalized["value"] == 1.25


def test_conflicting_legacy_k_suffix_is_rejected():
    with pytest.raises(ValueError, match="encodes K=4"):
        normalize_artifact_record(
            {
                "system": "tfim",
                "method_id": "indep_multilambda_uniform_k4",
                "k": 3,
                "metric": "r_val_raw",
                "value": 0.1,
            }
        )

