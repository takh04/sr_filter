"""Normalize historical experiment rows to the public long-form schema."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any


NORMALIZED_FIELDS = (
    "method",
    "lambda_policy",
    "k",
    "system",
    "checkpoint",
    "branch_seed",
    "repeat",
    "metric",
    "value",
)

LEGACY_METHODS = {
    "single_best_fixed_lambda": "single_grid_best_shift",
    "same_batch_multilambda_uniform": "same_batch_multishift_uniform",
    "same_batch_multilambda_stacking": "same_batch_multishift_stacking",
    "indep_single_lambda_uniform": "bagged_grid_best_shift",
    "indep_multilambda_uniform": "independent_multishift_uniform",
    "indep_multilambda_stacking": "ms_sr",
    "mssr": "ms_sr",
    "bagged": "bagged_sr",
}

DEFAULT_LAMBDA_POLICIES = {
    "single_grid_best_shift": "evaluation_grid_best_same_validation",
    "same_batch_multishift_uniform": "ntk_quantile_grid_uniform",
    "same_batch_multishift_stacking": "ntk_quantile_grid_stacking",
    "bagged_grid_best_shift": "evaluation_grid_best_same_validation",
    "independent_multishift_uniform": "ntk_quantile_grid_uniform",
    "ms_sr": "ntk_quantile_grid_stacking",
    "sr": "fixed_shift",
    "spring": "fixed_shift_with_momentum",
    "bagged_sr": "fixed_shift_uniform",
}

_K_SUFFIX = re.compile(r"_k(?P<k>[1-9][0-9]*)$")


def _optional_int(value: Any, *, field: str) -> int | None:
    if value in (None, ""):
        return None
    numeric = float(value)
    rounded = round(numeric)
    if not math.isfinite(numeric) or abs(numeric - rounded) > 1e-9:
        raise ValueError(f"{field} must be an integer or blank, got {value!r}")
    return int(rounded)


def normalize_method_id(method_id: str, *, k: int | None = None) -> tuple[str, int]:
    """Return a stable method name and candidate count from old or new IDs."""

    token = str(method_id).strip().lower().replace("-", "_")
    if not token:
        raise ValueError("method identifier must be nonempty")
    match = _K_SUFFIX.search(token)
    suffix_k = int(match.group("k")) if match else None
    if match:
        token = token[: match.start()]
    resolved_k = int(k) if k is not None else suffix_k
    method = LEGACY_METHODS.get(token, token)
    if resolved_k is None:
        resolved_k = 1 if method in {"sr", "spring", "single_grid_best_shift"} else 4
    if resolved_k <= 0:
        raise ValueError(f"k must be positive, got {resolved_k}")
    if suffix_k is not None and k is not None and suffix_k != resolved_k:
        raise ValueError(
            f"method identifier {method_id!r} encodes K={suffix_k}, "
            f"but the record declares K={resolved_k}"
        )
    return method, resolved_k


def normalize_artifact_record(
    record: Mapping[str, Any],
    *,
    metric: str | None = None,
) -> dict[str, Any]:
    """Normalize one scalar record without retaining machine-local metadata.

    Historical rows may use ``method_id``/``source_method_id``, ``_k4``
    suffixes, ``checkpoint_step``, ``branch_rng_seed``, ``repeat_index``, and a
    summary ``mean`` column. The returned mapping always has exactly the public
    long-form fields in :data:`NORMALIZED_FIELDS`.
    """

    raw_method = record.get("method", record.get("method_id", record.get("source_method_id")))
    if raw_method in (None, ""):
        raise ValueError("artifact record has no method identifier")
    declared_k = _optional_int(record.get("k"), field="k")
    method_name, resolved_k = normalize_method_id(str(raw_method), k=declared_k)

    system = str(record.get("system", "")).strip().lower()
    if not system:
        raise ValueError("artifact record has no system")
    lambda_policy = str(
        record.get(
            "lambda_policy",
            record.get(
                "selection_label",
                DEFAULT_LAMBDA_POLICIES.get(method_name, "unspecified"),
            ),
        )
    ).strip()
    if not lambda_policy:
        raise ValueError("artifact record has no lambda policy")

    metric_name = str(metric if metric is not None else record.get("metric", "")).strip()
    if not metric_name:
        raise ValueError("artifact record has no metric")
    if metric is not None:
        raw_value = record.get(metric_name)
    else:
        raw_value = record.get("value", record.get("mean"))
    if raw_value in (None, ""):
        raise ValueError(f"artifact record has no scalar value for metric {metric_name!r}")
    value = float(raw_value)
    if not math.isfinite(value):
        raise ValueError(f"artifact metric value must be finite, got {raw_value!r}")

    checkpoint = record.get(
        "checkpoint",
        record.get(
            "source_checkpoint_id",
            record.get("checkpoint_step", record.get("source_checkpoint")),
        ),
    )
    return {
        "method": method_name,
        "lambda_policy": lambda_policy,
        "k": resolved_k,
        "system": system,
        "checkpoint": checkpoint,
        "branch_seed": _optional_int(
            record.get("branch_seed", record.get("branch_rng_seed")),
            field="branch_seed",
        ),
        "repeat": _optional_int(
            record.get("repeat", record.get("repeat_index", record.get("repeat_number"))),
            field="repeat",
        ),
        "metric": metric_name,
        "value": value,
    }

