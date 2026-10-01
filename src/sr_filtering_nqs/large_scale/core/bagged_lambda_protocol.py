"""Training-shift and bagged-objective-tuned SR for the Figure 4 ablation.

Each shift uses the same K independent training batches and validation data.
The tuned shift minimizes the mean validation residual of the uniform bagged
updates, rather than reusing the shift selected for individual SR updates.
Both its residual and its variance come from that selected shift. Selection
uses the evaluation data, so this is an oracle diagnostic baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import numpy as np

from sr_filtering_nqs.large_scale.core.common import format_lambda_dir
from sr_filtering_nqs.large_scale.core.diagnostic_state import atomic_save
from sr_filtering_nqs.large_scale.core.fixed_data_protocol import (
    FIXED_DATA_REPEAT_COUNT,
    SCHEMA_VERSION,
    TRAIN_SEED_OFFSET,
    WORKSPACE_NAME,
    _live_batch_seed,
    _load_training_driver,
    _method_metrics_from_final_deltas_live,
    _resolve_seed_base,
    _sample_live_prepared_batch,
    best_fixed_lambda_for_checkpoint,
    canonical_fixed_data_checkpoints,
    checkpoint_step_from_path,
    ensure_reference_spectrum,
    fig3_step_dir,
    heldout_samples_for_system,
    load_best_fixed_lambda_summary,
    load_cached_spectrum,
    normalize_system_name,
    resolve_fig3_lambda_quantiles,
    spectrum_quantile_lambda_grid,
    spectrum_ref_path,
)
from sr_filtering_nqs.large_scale.core.multilambda_sr import (
    base_solve_record,
    restore_driver_state,
    snapshot_driver_state,
    uniform_mixture_weights,
    weighted_tree_sum,
)

DEFAULT_BAGGED_LAMBDAS = tuple(10.0**exponent for exponent in range(-8, 0))
BAGGED_SELECTION_OBJECTIVE = "mean_validation_residual_of_uniform_bagged_updates"


def select_bagged_lambda(methods: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Select one shift by its mean bagged residual, keeping its other metrics."""
    if not methods:
        raise ValueError("At least one bagged shift is required for selection")
    candidates = []
    for method_id, metrics in methods.items():
        residuals = np.asarray(metrics["r_val_raw_per_repeat"], dtype=np.float64)
        if residuals.ndim != 1 or residuals.size == 0 or not np.all(np.isfinite(residuals)):
            raise ValueError(f"Invalid bagged residual repeats for {method_id}")
        if residuals.size != int(metrics["repeat_count"]):
            raise ValueError(f"Incomplete bagged residual repeats for {method_id}")
        mean_residual = float(residuals.mean())
        if not np.isclose(mean_residual, float(metrics["r_val_raw"]), rtol=1e-12, atol=1e-14):
            raise ValueError(f"Bagged residual mean disagrees with repeats for {method_id}")
        shift = float(metrics["bagged_lambda"])
        if not np.isfinite(shift) or shift <= 0:
            raise ValueError(f"Invalid bagged shift for {method_id}")
        candidates.append((mean_residual, shift, method_id))
    _, _, selected_id = min(candidates)
    return {"method_id": selected_id, **methods[selected_id]}


def bagged_method_id(lam: float, *, k: int) -> str:
    return f"bagged_fixed_lambda_uniform_k{int(k)}_{format_lambda_dir(float(lam))}"


def bagged_lambda_eval_path(base_root, system_name: str, step: int, *, k: int = 4) -> Path:
    return fig3_step_dir(base_root, system_name, step) / f"bagged_lambda_eval_k{int(k)}.pkl"


def evaluate_bagged_fixed_lambda_checkpoint(
    ckpt_path,
    *,
    base_root,
    output_root=None,
    system_name: str,
    bagged_lambdas: tuple[float, ...] = DEFAULT_BAGGED_LAMBDAS,
    k: int = 4,
    repeat_count: int = FIXED_DATA_REPEAT_COUNT,
    n_val: int | None = None,
    validation_round_samples: int | None = None,
    validation_sample_chunk: int | None = None,
    jvp_chunk: int | None = None,
    delta_batch_size: int = 10,
    train_seed_base: int | None = None,
    validation_seed_base: int | None = None,
    spectrum_seed_base: int | None = None,
    lambda_quantiles=None,
    model_chunk_size_bwd: int | None = None,
) -> dict[str, Any]:
    """Evaluate bagged SR from diagnostics in ``base_root``.

    ``output_root`` controls only the result destination.  Keeping it separate
    lets several bagged grids reuse one immutable Figure 2 diagnostic cache.
    """

    system_name = normalize_system_name(system_name)
    result_root = base_root if output_root is None else output_root
    k = int(k)
    if k < 2:
        raise ValueError(f"Bagged SR requires K >= 2, got {k}")
    if int(repeat_count) <= 0:
        raise ValueError(f"repeat_count must be positive, got {repeat_count}")
    lambdas = tuple(float(lam) for lam in bagged_lambdas)
    if not lambdas:
        raise ValueError("At least one bagged lambda is required")
    if not all(np.isfinite(lam) and lam > 0.0 for lam in lambdas):
        raise ValueError(f"Bagged lambdas must be positive and finite, got {lambdas}")
    resolved_quantiles = resolve_fig3_lambda_quantiles(k, lambda_quantiles)
    step = int(checkpoint_step_from_path(ckpt_path))
    if n_val is None:
        n_val = int(heldout_samples_for_system(system_name))
    ensure_reference_spectrum(
        ckpt_path,
        base_root=base_root,
        system_name=system_name,
        spectrum_seed_base=spectrum_seed_base,
        model_chunk_size_bwd=model_chunk_size_bwd,
    )
    spectrum = load_cached_spectrum(base_root, system_name, step)
    lambda_star = best_fixed_lambda_for_checkpoint(base_root, system_name, step)
    quantile_lambda_grid = spectrum_quantile_lambda_grid(
        spectrum,
        quantiles=resolved_quantiles,
    )

    if validation_sample_chunk is None:
        if system_name == "tfim":
            validation_sample_chunk = 12000
        else:
            validation_sample_chunk = 4000
    if validation_round_samples is None:
        validation_round_samples = int(validation_sample_chunk)

    training = _load_training_driver(
        ckpt_path,
        system_name=system_name,
        chunk_size_bwd_override=model_chunk_size_bwd,
    )
    driver = training["driver"]
    resolved_train_seed_base = _resolve_seed_base(
        training["config"],
        step,
        train_seed_base,
        TRAIN_SEED_OFFSET,
    )
    snapshot = snapshot_driver_state(driver)
    original_chunk_size_bwd = getattr(driver, "chunk_size_bwd", None)

    method_ids = {lam: bagged_method_id(lam, k=k) for lam in lambdas}
    method_deltas: dict[str, list[Any]] = {mid: [] for mid in method_ids.values()}

    try:
        for repeat_index in range(int(repeat_count)):
            full_batches = [
                _sample_live_prepared_batch(
                    driver,
                    seed=_live_batch_seed(
                        resolved_train_seed_base,
                        repeat_index=repeat_index,
                        slot_index=slot_index,
                        batch_slots=int(k),
                    ),
                    chain_name=f"fixed_data_live_r{repeat_index:03d}_s{slot_index:02d}",
                )
                for slot_index in range(int(k))
            ]
            for lam in lambdas:
                records = [
                    base_solve_record(
                        driver,
                        batch,
                        diag_shift=float(lam),
                        collect_residual_info=False,
                    )
                    for batch in full_batches
                ]
                weights = uniform_mixture_weights(len(records))
                bagged_delta = weighted_tree_sum(
                    [record.delta for record in records],
                    weights,
                )
                method_deltas[method_ids[lam]].append(
                    jax.tree_util.tree_map(np.asarray, bagged_delta)
                )
    finally:
        driver.chunk_size_bwd = original_chunk_size_bwd
        restore_driver_state(driver, snapshot)

    methods = {}
    validation_context = None
    for lam in lambdas:
        method_id = method_ids[lam]
        metrics, validation_context = _method_metrics_from_final_deltas_live(
            training,
            base_root=base_root,
            system_name=system_name,
            final_deltas=method_deltas[method_id],
            n_val=int(n_val),
            validation_round_samples=validation_round_samples,
            validation_sample_chunk=int(validation_sample_chunk),
            validation_seed_base=validation_seed_base,
            jvp_chunk=jvp_chunk,
            delta_batch_size=delta_batch_size,
        )
        methods[method_id] = {
            "method_id": method_id,
            "bagged_lambda": float(lam),
            "weights_per_repeat": [[1.0 / int(k)] * int(k)] * int(repeat_count),
            "mean_weights": [1.0 / int(k)] * int(k),
            **metrics,
        }
    if validation_context is None:
        raise RuntimeError(f"No bagged-lambda methods were evaluated for {ckpt_path}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "workspace": WORKSPACE_NAME,
        "system": system_name,
        "step": int(step),
        "source_step": int(step),
        "k": int(k),
        "repeat_count": int(repeat_count),
        "bagged_lambdas": [float(lam) for lam in lambdas],
        "lambda_star": float(lambda_star),
        "lambda_single_sr_tuned": float(lambda_star),
        "quantile_lambda_grid": [float(value) for value in quantile_lambda_grid],
        "lambda_quantiles": [float(value) for value in resolved_quantiles],
        "source_checkpoint": str(Path(ckpt_path).resolve()),
        "source_training_lambda": float(training["config"]["diag_shift"]),
        "n_val": int(n_val),
        "sample_materialization": "live_sampler",
        "train_seed_base": int(resolved_train_seed_base),
        "validation_seed": int(validation_context["validation_seed"]),
        "validation_round_samples": int(validation_context["round_samples"]),
        "spectrum_ref_path": str(spectrum_ref_path(base_root, system_name, step)),
        "methods": methods,
    }
    selected = select_bagged_lambda(methods)
    payload.update(
        selection_objective=BAGGED_SELECTION_OBJECTIVE,
        lambda_bagged_tuned=float(selected["bagged_lambda"]),
        bagged_tuned=selected,
    )
    result_path = bagged_lambda_eval_path(result_root, system_name, step, k=int(k))
    atomic_save(payload, result_path)
    return payload


def run_bagged_fixed_lambda(
    base_root,
    *,
    diagnostic_root=None,
    output_root=None,
    system_name: str,
    bagged_lambdas: tuple[float, ...] = DEFAULT_BAGGED_LAMBDAS,
    source_training_lambda: float = 1e-4,
    source_steps: str | None = "all",
    latest_only: bool = False,
    repeat_count: int = FIXED_DATA_REPEAT_COUNT,
    k: int = 4,
    n_val: int | None = None,
    validation_round_samples: int | None = None,
    validation_sample_chunk: int | None = None,
    jvp_chunk: int | None = None,
    delta_batch_size: int = 10,
    train_seed_base: int | None = None,
    validation_seed_base: int | None = None,
    spectrum_seed_base: int | None = None,
    lambda_quantiles=None,
    model_chunk_size_bwd: int | None = None,
) -> list[dict[str, Any]]:
    """Run bagged SR with separate checkpoint, diagnostic, and result roots."""

    system_name = normalize_system_name(system_name)
    result_root = base_root if output_root is None else output_root
    shared_root = result_root if diagnostic_root is None else diagnostic_root
    checkpoints = canonical_fixed_data_checkpoints(
        base_root,
        system_name,
        source_training_lambda=float(source_training_lambda),
        source_steps=source_steps,
        latest_only=latest_only,
    )
    if not checkpoints:
        raise SystemExit(
            f"No source checkpoints found for {system_name} at lambda_train={source_training_lambda:.3e}"
        )

    load_best_fixed_lambda_summary(shared_root)
    results = []
    for step, ckpt_path in checkpoints:
        print(f"[bagged-lambda] system={system_name} step={step}", flush=True)
        results.append(
            evaluate_bagged_fixed_lambda_checkpoint(
                ckpt_path,
                base_root=shared_root,
                output_root=result_root,
                system_name=system_name,
                bagged_lambdas=bagged_lambdas,
                k=k,
                repeat_count=repeat_count,
                n_val=n_val,
                validation_round_samples=validation_round_samples,
                validation_sample_chunk=validation_sample_chunk,
                jvp_chunk=jvp_chunk,
                delta_batch_size=delta_batch_size,
                train_seed_base=train_seed_base,
                validation_seed_base=validation_seed_base,
                spectrum_seed_base=spectrum_seed_base,
                lambda_quantiles=lambda_quantiles,
                model_chunk_size_bwd=model_chunk_size_bwd,
            )
        )
    return results
