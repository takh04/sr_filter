"""Online fixed-grid MS-SR with an exact, auditable sampling budget.

The general ``MultiLambdaSRCallback`` is intentionally retained for the
checkpoint-local diagnostic variants.  It runs after the base VMC driver has
already sampled and solved an ordinary SR batch, however, and independent
stacking also draws a separate weight-fit batch.  It therefore cannot
implement the equal-budget online protocol.

This module provides a dedicated VMC driver for that protocol.  One optimizer
update draws exactly ``K`` named candidate batches, performs exactly one SR
solve on each batch with its corresponding fixed shift, fits the stacking
weights by exact leave-one-batch-out (LOO) prediction on those same batches,
and applies their weighted update.  No default-driver batch or separate
weight-fit batch is sampled.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import core as fcore
from netket.stats import statistics
from netket.utils import struct, timing

from advanced_drivers._src.driver.ngd.driver_vmc_ngd import VMC_NG
from netket_foundational._src.driver.ngd.driver_vmc_ngd import (
    VMC_NG as FoundationalVMCNG,
)
from sr_filtering_nqs.large_scale.core.optimized_sr_benchmark import (
    draw_prepared_batches,
    optimized_exact_loo_weights,
    weighted_delta,
)

DIAGNOSTIC_CALLBACK_PROTOCOL = "diagnostic_callback"
FIXED_GRID_EXACT_LOO_PROTOCOL = "fixed_grid_exact_loo"
SR_UPDATE_PROTOCOLS = (
    DIAGNOSTIC_CALLBACK_PROTOCOL,
    FIXED_GRID_EXACT_LOO_PROTOCOL,
)


def validate_fixed_grid(lambda_grid: Sequence[float]) -> tuple[float, ...]:
    """Validate and normalize a fixed one-shift-per-candidate grid."""

    values = tuple(float(value) for value in lambda_grid)
    if len(values) < 2:
        raise ValueError(f"Fixed-grid MS-SR requires K >= 2 shifts, got {values}")
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError(f"Fixed-grid MS-SR shifts must be finite and positive, got {values}")
    return values


def validate_online_mssr_configuration(
    *,
    update_protocol: str,
    sr_variant: str,
    sr_weight_scheme: str,
    sr_lambda_grid: Sequence[float],
    sr_weight_fit_samples: int | None,
    sr_lambda_quantiles: Sequence[float] | None,
    sr_lambda_ema_rho: float | None,
    sr_weight_ema_rho: float | None,
    ngd_method: str,
) -> tuple[float, ...]:
    """Reject configurations that would silently change the exact-budget arm."""

    if update_protocol not in SR_UPDATE_PROTOCOLS:
        raise ValueError(
            f"Unsupported sr_update_protocol={update_protocol!r}; "
            f"expected one of {SR_UPDATE_PROTOCOLS}"
        )
    grid = validate_fixed_grid(sr_lambda_grid)
    if update_protocol == DIAGNOSTIC_CALLBACK_PROTOCOL:
        return grid

    mismatches = []
    if sr_variant != "indep_batch":
        mismatches.append("sr_variant must be 'indep_batch'")
    if sr_weight_scheme != "stacking":
        mismatches.append("sr_weight_scheme must be 'stacking'")
    if sr_weight_fit_samples is not None:
        mismatches.append(
            "sr_weight_fit_samples must be unset (LOO reuses the K candidate batches)"
        )
    if sr_lambda_quantiles is not None:
        mismatches.append(
            "sr_lambda_quantiles must be unset (the online protocol uses a fixed grid)"
        )
    if sr_lambda_ema_rho is not None:
        mismatches.append("sr_lambda_ema_rho must be unset")
    if sr_weight_ema_rho is not None:
        mismatches.append(
            "sr_weight_ema_rho must be unset (weights are refit by exact LOO each update)"
        )
    if ngd_method != "sr":
        mismatches.append("ngd_method must be 'sr'")
    if mismatches:
        raise ValueError(
            f"{FIXED_GRID_EXACT_LOO_PROTOCOL} configuration is invalid: " + "; ".join(mismatches)
        )
    return grid


def _pooled_candidate_loss_statistics(batches):
    """Report energy from the K already-drawn candidate batches, without resampling."""

    weighted_local_losses = [
        jnp.ravel(jnp.asarray(batch.local_loss) * jnp.asarray(batch.weights)) for batch in batches
    ]
    return statistics(jnp.concatenate(weighted_local_losses, axis=0))


def _driver_scalar(value, *, step: int):
    return value(step) if callable(value) else value


def solve_fixed_grid_candidate(driver, batch, *, diag_shift: float):
    """Perform one and only one native SR solver call for a prepared batch."""

    model_state, params = fcore.pop(batch.variables, "params")
    common_kwargs = {
        "diag_shift": float(diag_shift),
        "solver_fn": driver._linear_solver_fn,
        "mode": driver.mode,
        "proj_reg": _driver_scalar(
            getattr(driver, "proj_reg", None),
            step=driver.step_count,
        ),
        "momentum": _driver_scalar(
            getattr(driver, "momentum", None),
            step=driver.step_count,
        ),
        "old_updates": driver._old_updates,
        "chunk_size": driver.chunk_size_bwd,
        "collect_quadratic_model": False,
        "collect_gradient_statistics": False,
        "collect_residual_info": False,
    }
    # The standard driver uses importance weights.  The foundational/FNQS
    # driver instead exposes the number of parameter replicas.  Selecting the
    # signature from the prepared batch avoids retrying update_fn and makes the
    # one-call-per-candidate invariant directly auditable.
    if batch.n_replicas is None:
        distribution_kwargs = {"importance_sampling_weights": batch.weights}
    else:
        distribution_kwargs = {"n_replicas": int(batch.n_replicas)}
    delta, _, info = driver.update_fn(
        batch.afun,
        batch.local_grad,
        params,
        model_state,
        batch.samples,
        **distribution_kwargs,
        **common_kwargs,
    )
    jax.block_until_ready(jax.tree_util.tree_leaves(delta))
    return delta, {} if info is None else info


def compute_fixed_grid_exact_loo_update(
    driver,
    *,
    lambda_grid: Sequence[float],
    n_samples: int,
    prediction_chunk: int | None = None,
    chain_prefix: str = "online_fixed_grid_mssr",
) -> tuple[Any, Any, dict[str, Any]]:
    """Compute one fixed-grid MS-SR update using exactly K samples and K solves.

    Returns ``(loss_statistics, mixed_delta, metadata)``.  The loss statistics
    pool the local energies from the K candidate batches; this provides a
    training trace without drawing a reporting-only batch.
    """

    grid = validate_fixed_grid(lambda_grid)
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if prediction_chunk is not None and int(prediction_chunk) <= 0:
        raise ValueError(f"prediction_chunk must be positive, got {prediction_chunk}")
    if (
        getattr(driver, "momentum", None) is not None
        or getattr(driver, "proj_reg", None) is not None
    ):
        raise ValueError("Fixed-grid online MS-SR requires plain SR without momentum or proj_reg")

    total_start = time.perf_counter()
    k = len(grid)
    prepare_start = time.perf_counter()
    batches = draw_prepared_batches(
        driver,
        k,
        n_samples=n_samples,
        chain_prefix=str(chain_prefix),
    )
    candidate_sample_counts = tuple(int(batch.sample_count) for batch in batches)
    if candidate_sample_counts != (n_samples,) * k:
        raise RuntimeError(
            "MS-SR candidate sampler did not return the configured batch size: "
            f"expected {(n_samples,) * k}, got {candidate_sample_counts}"
        )
    prepare_seconds = time.perf_counter() - prepare_start

    solve_start = time.perf_counter()
    deltas = []
    for batch, diag_shift in zip(batches, grid, strict=True):
        delta, _ = solve_fixed_grid_candidate(
            driver,
            batch,
            diag_shift=diag_shift,
        )
        deltas.append(delta)
    solve_seconds = time.perf_counter() - solve_start

    stacking_start = time.perf_counter()
    weights, stacking_meta = optimized_exact_loo_weights(
        driver,
        batches,
        deltas,
        prediction_chunk=prediction_chunk,
    )
    stacking_seconds = time.perf_counter() - stacking_start
    mix_and_loss_start = time.perf_counter()
    mixed_delta = weighted_delta(deltas, weights)
    jax.block_until_ready(jax.tree_util.tree_leaves(mixed_delta))
    loss_stats = _pooled_candidate_loss_statistics(batches)
    jax.block_until_ready(jax.tree_util.tree_leaves(loss_stats))
    mix_and_loss_seconds = time.perf_counter() - mix_and_loss_start
    total_seconds = time.perf_counter() - total_start

    completed_update = int(getattr(driver, "step_count", 0)) + 1
    metadata = {
        "method": "ms_sr",
        "variant": "indep_batch",
        "weight_scheme": "stacking",
        "update_protocol": FIXED_GRID_EXACT_LOO_PROTOCOL,
        "stacking_protocol": "exact_leave_one_batch_out",
        "lambda_source": "fixed_grid",
        "lambda_grid": [float(value) for value in grid],
        "weights": [float(value) for value in np.asarray(weights, dtype=np.float64)],
        "k": int(k),
        "samples_per_candidate_batch": int(n_samples),
        "candidate_sample_counts": list(candidate_sample_counts),
        "candidate_batches": int(k),
        "candidate_solves": int(k),
        "sampling_solve_units_this_update": int(k),
        "sampling_solve_units_cumulative": int(completed_update * k),
        "base_driver_sampling_calls": 0,
        "base_driver_solves": 0,
        "weight_fit_sampling_calls": 0,
        "weight_fit_solves": 0,
        "training_energy_source": "pooled_candidate_batches",
        "training_energy_sample_count": int(sum(candidate_sample_counts)),
        "train_metrics_policy": "omitted_to_avoid_nonprotocol_diagnostics",
        "prediction_chunk": None if prediction_chunk is None else int(prediction_chunk),
        "prepare_batches_s": float(prepare_seconds),
        "candidate_solves_s": float(solve_seconds),
        "stacking_total_s": float(stacking_seconds),
        "mix_and_loss_s": float(mix_and_loss_seconds),
        "total_update_s": float(total_seconds),
        **{
            key: (
                int(value)
                if key in {"cross_prediction_launches", "cross_prediction_directions"}
                else float(value)
            )
            for key, value in stacking_meta.items()
        },
    }
    return loss_stats, mixed_delta, metadata


class FixedGridExactLOOMSSRDriver(VMC_NG):
    """VMC driver whose native update is the exact-budget fixed-grid MS-SR step."""

    _mssr_lambda_grid: tuple[float, ...] = struct.field(
        pytree_node=False,
        serialize=False,
    )
    _mssr_n_samples: int = struct.field(pytree_node=False, serialize=False)
    _mssr_prediction_chunk: int | None = struct.field(
        pytree_node=False,
        serialize=False,
    )
    _mssr_chain_prefix: str = struct.field(pytree_node=False, serialize=False)

    def __init__(
        self,
        *args,
        lambda_grid: Sequence[float],
        n_samples: int,
        prediction_chunk: int | None = None,
        chain_prefix: str = "online_fixed_grid_mssr",
        **kwargs,
    ):
        if "diag_shift" in kwargs:
            raise TypeError("FixedGridExactLOOMSSRDriver derives diag_shift from lambda_grid")
        grid = validate_fixed_grid(lambda_grid)
        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if prediction_chunk is not None and int(prediction_chunk) <= 0:
            raise ValueError(f"prediction_chunk must be positive, got {prediction_chunk}")
        if kwargs.get("momentum") is not None or kwargs.get("proj_reg") is not None:
            raise ValueError(
                "Fixed-grid online MS-SR requires plain SR without momentum or proj_reg"
            )

        super().__init__(*args, diag_shift=max(grid), **kwargs)
        self._mssr_lambda_grid = grid
        self._mssr_n_samples = n_samples
        self._mssr_prediction_chunk = None if prediction_chunk is None else int(prediction_chunk)
        self._mssr_chain_prefix = str(chain_prefix)

    @timing.timed
    def compute_loss_and_update(self):
        loss_stats, delta, metadata = compute_fixed_grid_exact_loo_update(
            self,
            lambda_grid=self._mssr_lambda_grid,
            n_samples=self._mssr_n_samples,
            prediction_chunk=self._mssr_prediction_chunk,
            chain_prefix=self._mssr_chain_prefix,
        )
        self._loss_stats = loss_stats
        self._dp = delta
        self.info = {"multilambda": metadata}
        return loss_stats, delta


class FoundationalFixedGridExactLOOMSSRDriver(FoundationalVMCNG):
    """FNQS-compatible form of :class:`FixedGridExactLOOMSSRDriver`."""

    _mssr_lambda_grid: tuple[float, ...] = struct.field(
        pytree_node=False,
        serialize=False,
    )
    _mssr_n_samples: int = struct.field(pytree_node=False, serialize=False)
    _mssr_prediction_chunk: int | None = struct.field(
        pytree_node=False,
        serialize=False,
    )
    _mssr_chain_prefix: str = struct.field(pytree_node=False, serialize=False)

    def __init__(
        self,
        *args,
        lambda_grid: Sequence[float],
        n_samples: int,
        prediction_chunk: int | None = None,
        chain_prefix: str = "online_fixed_grid_mssr",
        **kwargs,
    ):
        if "diag_shift" in kwargs:
            raise TypeError(
                "FoundationalFixedGridExactLOOMSSRDriver derives diag_shift from lambda_grid"
            )
        grid = validate_fixed_grid(lambda_grid)
        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if prediction_chunk is not None and int(prediction_chunk) <= 0:
            raise ValueError(f"prediction_chunk must be positive, got {prediction_chunk}")
        if kwargs.get("momentum") is not None or kwargs.get("proj_reg") is not None:
            raise ValueError(
                "Fixed-grid online MS-SR requires plain SR without momentum or proj_reg"
            )

        super().__init__(*args, diag_shift=max(grid), **kwargs)
        self._mssr_lambda_grid = grid
        self._mssr_n_samples = n_samples
        self._mssr_prediction_chunk = None if prediction_chunk is None else int(prediction_chunk)
        self._mssr_chain_prefix = str(chain_prefix)

    @timing.timed
    def compute_loss_and_update(self):
        loss_stats, delta, metadata = compute_fixed_grid_exact_loo_update(
            self,
            lambda_grid=self._mssr_lambda_grid,
            n_samples=self._mssr_n_samples,
            prediction_chunk=self._mssr_prediction_chunk,
            chain_prefix=self._mssr_chain_prefix,
        )
        self._loss_stats = loss_stats
        self._dp = delta
        self.info = {"multilambda": metadata}
        return loss_stats, delta
