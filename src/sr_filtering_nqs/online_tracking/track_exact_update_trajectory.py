#!/usr/bin/env python
"""Track stochastic optimizer updates against exact population SR on 4x4 systems.

This is a quick Figure 5 diagnostic script.  It runs short online trajectories
and, at each step, computes the exact dense population SR update at the current
state by full summation over the 4x4 Hilbert space.  The row written for step
t compares the update that will be applied at theta_t against that exact local
target delta*(theta_t).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "True")

ROOT = Path.cwd()

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
from flax import core as fcore
from jax.flatten_util import ravel_pytree
from netket import jax as nkjax
from netket.utils import struct

from advanced_drivers._src.callbacks.base import AbstractCallback
from advanced_drivers._src.driver.ngd.sr_srt_common import sr as dense_sr
from sr_filtering_nqs.large_scale.core.multilambda_sr import MultiLambdaSRCallback, _resolve_distribution

from sr_filtering_nqs.online_tracking.sweep_train import (
    METHODS,
    NTK_QUANTILES,
    PAPER_GRID,
    SYSTEMS,
    append_csv_row,
    atomic_write_json,
    atomic_write_pickle,
    count_parameters,
    exact_metrics,
    fidelity_with_ground_state,
    load_or_compute_exact_state,
    make_driver,
    make_model,
    make_system_bundle,
    make_vstate,
    method_grid_config,
    model_config,
    normalized_state_vector,
    parse_float_tuple,
    trace_distance_from_fidelity,
)


TRACK_HEADER = [
    "update_step",
    "state_step",
    "method",
    "ite_index",
    "mc_energy",
    "mc_energy_variance",
    "exact_energy",
    "exact_energy_gap",
    "exact_relative_gap_abs",
    "exact_relative_gap_signed",
    "exact_variance",
    "fidelity",
    "trace_distance",
    "ite_fidelity",
    "ite_trace_distance",
    "s_error",
    "s_relative_error",
    "s_cosine",
    "s_norm_update",
    "s_norm_ideal",
    "l2_error",
    "l2_relative_error",
    "l2_cosine",
    "l2_norm_update",
    "l2_norm_ideal",
    "ideal_diag_shift",
    "ideal_solve_wall_s",
    "metric_wall_s",
    "multilambda_weights",
    "multilambda_lambdas",
]


def json_ready(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): json_ready(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (np.complexfloating, complex)):
        value = complex(obj)
        return {"real": float(value.real), "imag": float(value.imag)}
    return obj


def numpy_tree(tree):
    if tree is None:
        return None

    def convert_leaf(leaf):
        try:
            return np.asarray(leaf)
        except TypeError:
            return leaf

    return jax.tree_util.tree_map(convert_leaf, tree)


def final_checkpoint_payload(
    *,
    step: int,
    driver,
    config: dict[str, Any],
    final_exact_metrics: dict[str, Any] | None,
    status: str,
    error: str | None,
) -> dict[str, Any]:
    multilambda = None
    if isinstance(getattr(driver, "info", None), dict):
        multilambda = driver.info.get("multilambda")
    return {
        "step": int(step),
        "status": status,
        "error": error,
        "parameters": numpy_tree(driver.state.parameters),
        "variables": numpy_tree(driver.state.variables),
        "sampler_state": numpy_tree(getattr(driver.state, "sampler_state", None)),
        "optimizer_state": numpy_tree(getattr(driver, "_optimizer_state", None)),
        "old_updates": numpy_tree(getattr(driver, "_old_updates", None)),
        "last_update": numpy_tree(getattr(driver, "_dp", None)),
        "config": config,
        "final_exact_metrics": final_exact_metrics,
        "multilambda": json_ready(multilambda),
    }


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def atomic_savez(target_path: Path, **arrays) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(target_path.parent), suffix=".npz.tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **arrays)
        os.replace(tmp_path, target_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def parse_csv(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def resolve_solver(name: str):
    solvers = {
        "pinv": nk.optimizer.solver.pinv,
        "pinv_smooth": nk.optimizer.solver.pinv_smooth,
    }
    return solvers[name]


def all_hilbert_states(hilbert) -> np.ndarray:
    states = np.asarray(hilbert.all_states())
    if states.ndim != 2:
        raise ValueError(f"Expected all_states with shape (n_states, n_sites), got {states.shape}")
    return states


def normalize_state_array(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=np.complex128)
    norm = float(np.linalg.norm(psi))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("Non-normalizable state vector")
    return np.asarray(psi / norm, dtype=np.complex128)


def state_vector_sha256(psi: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(psi, dtype=np.complex128))
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def float_token(value: float) -> str:
    return f"{float(value):.12g}".replace("-", "m").replace(".", "p")


def pure_state_overlap_metrics(reference: np.ndarray, psi: np.ndarray) -> dict[str, float]:
    reference = normalize_state_array(reference)
    psi = normalize_state_array(psi)
    fidelity = float(np.abs(np.vdot(reference, psi)) ** 2)
    fidelity = float(np.clip(fidelity, 0.0, 1.0))
    return {
        "ite_fidelity": fidelity,
        "ite_trace_distance": trace_distance_from_fidelity(fidelity),
    }


def state_vector_from_variables(
    bundle,
    model_name: str,
    variables,
    *,
    chunk_size: int,
) -> np.ndarray:
    vstate = nk.vqs.FullSumState(
        bundle.hilbert,
        model=make_model(bundle, model_name),
        chunk_size=int(chunk_size),
    )
    vstate.parameters = jax.tree_util.tree_map(np.asarray, variables["params"])
    return normalize_state_vector(vstate)


def logpsi_on_states(afun, variables, states: np.ndarray, *, chunk_size: int) -> np.ndarray:
    chunks = []
    for start in range(0, states.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), states.shape[0])
        chunks.append(np.asarray(afun(variables, jnp.asarray(states[start:stop]))))
    return np.concatenate(chunks, axis=0)


def born_pdf(afun, variables, states: np.ndarray, *, chunk_size: int) -> np.ndarray:
    logpsi = logpsi_on_states(afun, variables, states, chunk_size=chunk_size)
    logp = 2.0 * np.real(logpsi)
    logp -= float(np.max(logp))
    pdf = np.exp(logp)
    pdf_sum = float(np.sum(pdf))
    if not math.isfinite(pdf_sum) or pdf_sum <= 0.0:
        raise ValueError("Non-normalizable Born distribution")
    return np.asarray(pdf / pdf_sum, dtype=np.float64)


def flatten_real_tree(tree) -> np.ndarray:
    real_tree, _ = nkjax.tree_to_real(tree)
    flat, _ = ravel_pytree(real_tree)
    return np.asarray(flat, dtype=np.float64)


def l2_metrics(update, ideal) -> dict[str, float]:
    update_flat = flatten_real_tree(update)
    ideal_flat = flatten_real_tree(ideal)
    diff = update_flat - ideal_flat
    update_norm = float(np.linalg.norm(update_flat))
    ideal_norm = float(np.linalg.norm(ideal_flat))
    diff_norm = float(np.linalg.norm(diff))
    denom = max(update_norm * ideal_norm, 1e-30)
    return {
        "l2_error": diff_norm,
        "l2_relative_error": diff_norm / max(ideal_norm, 1e-30),
        "l2_cosine": float(np.dot(update_flat, ideal_flat) / denom),
        "l2_norm_update": update_norm,
        "l2_norm_ideal": ideal_norm,
    }


def jvp_prediction(
    afun,
    variables,
    states: np.ndarray,
    delta,
    *,
    chunk_size: int,
) -> np.ndarray:
    params = variables["params"]
    model_state, _ = fcore.pop(variables, "params")
    params_real, reconstruct = nkjax.tree_to_real(params)
    delta_real, _ = nkjax.tree_to_real(delta)

    @jax.jit
    def chunk_jvp(sample_chunk):
        def f(p_real):
            return afun({"params": reconstruct(p_real), **model_state}, sample_chunk)

        _, pred = jax.jvp(f, (params_real,), (delta_real,))
        return pred

    chunks = []
    for start in range(0, states.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), states.shape[0])
        chunks.append(np.asarray(chunk_jvp(jnp.asarray(states[start:stop]))))
    return np.concatenate(chunks, axis=0)


def centered_prediction_channels(pred: np.ndarray, pdf: np.ndarray, *, mode: str) -> tuple[np.ndarray, ...]:
    if mode == "complex":
        real = np.real(pred)
        imag = np.imag(pred)
        return (
            real - float(np.dot(pdf, real)),
            imag - float(np.dot(pdf, imag)),
        )
    real = np.real(pred)
    return (real - float(np.dot(pdf, real)),)


def channel_dot(a: tuple[np.ndarray, ...], b: tuple[np.ndarray, ...], pdf: np.ndarray) -> float:
    return float(sum(np.dot(pdf, ac * bc) for ac, bc in zip(a, b)))


def s_metric_update_metrics(
    *,
    afun,
    variables,
    states: np.ndarray,
    pdf: np.ndarray,
    update,
    ideal,
    mode: str,
    chunk_size: int,
) -> dict[str, float]:
    pred_update = jvp_prediction(afun, variables, states, update, chunk_size=chunk_size)
    pred_ideal = jvp_prediction(afun, variables, states, ideal, chunk_size=chunk_size)
    ch_update = centered_prediction_channels(pred_update, pdf, mode=mode)
    ch_ideal = centered_prediction_channels(pred_ideal, pdf, mode=mode)
    ch_diff = tuple(a - b for a, b in zip(ch_update, ch_ideal))

    s_error = channel_dot(ch_diff, ch_diff, pdf)
    s_norm_update = channel_dot(ch_update, ch_update, pdf)
    s_norm_ideal = channel_dot(ch_ideal, ch_ideal, pdf)
    denom = math.sqrt(max(s_norm_update, 0.0) * max(s_norm_ideal, 0.0))
    return {
        "s_error": s_error,
        "s_relative_error": s_error / max(s_norm_ideal, 1e-30),
        "s_cosine": channel_dot(ch_update, ch_ideal, pdf) / max(denom, 1e-30),
        "s_norm_update": s_norm_update,
        "s_norm_ideal": s_norm_ideal,
    }


def load_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must contain a dict payload: {path}")
    return checkpoint


def exact_metrics_from_variables(
    variables,
    bundle,
    exact,
    *,
    model_name: str,
    chunk_size: int,
) -> tuple[dict[str, float], np.ndarray]:
    vstate = nk.vqs.FullSumState(
        bundle.hilbert,
        model=make_model(bundle, model_name),
        chunk_size=int(chunk_size),
    )
    vstate.parameters = jax.tree_util.tree_map(np.asarray, variables["params"])
    psi = normalized_state_vector(vstate)
    stats = vstate.expect(bundle.hamiltonian)
    energy = float(np.real(complex(stats.Mean)))
    variance = float(np.real(complex(stats.Variance)))
    gap = energy - exact.energy
    abs_denom = abs(exact.energy) if abs(exact.energy) > 0.0 else 1.0
    signed_denom = exact.energy if abs(exact.energy) > 0.0 else 1.0
    if abs(gap) > 1e-14:
        v_score = float(bundle.n_sites * variance / (gap * gap))
    else:
        v_score = float("inf") if variance > 1e-14 else 0.0
    fidelity = fidelity_with_ground_state(psi, exact.vector)
    return {
        "exact_energy": energy,
        "exact_energy_per_site": energy / float(bundle.n_sites),
        "exact_variance": variance,
        "exact_variance_per_site": variance / float(bundle.n_sites),
        "exact_energy_gap": gap,
        "exact_relative_gap_abs": gap / abs_denom,
        "exact_relative_gap_signed": gap / signed_denom,
        "exact_v_score": v_score,
        "fidelity": fidelity,
        "trace_distance": trace_distance_from_fidelity(fidelity),
    }, psi


def exact_metrics_from_driver(
    driver,
    bundle,
    exact,
    *,
    model_name: str,
    chunk_size: int,
) -> dict[str, float]:
    metrics, _ = exact_metrics_from_variables(
        driver.state.variables,
        bundle,
        exact,
        model_name=model_name,
        chunk_size=chunk_size,
    )
    return metrics


def exact_metrics_and_state_from_driver(
    driver,
    bundle,
    exact,
    *,
    model_name: str,
    chunk_size: int,
) -> tuple[dict[str, float], np.ndarray]:
    return exact_metrics_from_variables(
        driver.state.variables,
        bundle,
        exact,
        model_name=model_name,
        chunk_size=chunk_size,
    )


def load_or_compute_ite_path(
    bundle,
    psi0: np.ndarray,
    *,
    learning_rate: float,
    steps: int,
    cache_dir: Path,
    force: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    psi0 = normalize_state_array(psi0)
    psi0_hash = state_vector_sha256(psi0)
    token = (
        f"{bundle.name}_L{bundle.lattice_size}_N{bundle.n_sites}"
        f"_lr{float_token(learning_rate)}_steps{int(steps)}_psi{psi0_hash[:16]}.npz"
    )
    cache_path = cache_dir / token
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as data:
            path = np.asarray(data["path"], dtype=np.complex128)
            cached_hash = str(np.asarray(data["psi0_sha256"]).reshape(()))
        if cached_hash != psi0_hash:
            raise ValueError(f"Cached ITE path has unexpected psi0 hash: {cache_path}")
        if path.shape != (int(steps) + 1, psi0.shape[0]):
            raise ValueError(f"Cached ITE path has unexpected shape {path.shape}: {cache_path}")
        return path, {
            "ite_cache_path": str(cache_path),
            "ite_cache_status": "loaded",
            "ite_initial_state_sha256": psi0_hash,
        }

    import scipy.sparse.linalg as spla

    cache_dir.mkdir(parents=True, exist_ok=True)
    hamiltonian = bundle.hamiltonian.to_sparse().astype(np.complex128)
    generator = (-float(learning_rate)) * hamiltonian
    path = np.empty((int(steps) + 1, psi0.shape[0]), dtype=np.complex128)
    path[0] = psi0
    t0 = time.time()
    print(
        f"Computing exact ITE path: steps={steps} lr={learning_rate} dim={psi0.shape[0]}",
        flush=True,
    )
    for index in range(1, int(steps) + 1):
        path[index] = normalize_state_array(spla.expm_multiply(generator, path[index - 1]))
        if index == 1 or index == int(steps) or index % 10 == 0:
            print(f"  ITE step {index:04d}/{int(steps):04d}", flush=True)

    atomic_savez(
        cache_path,
        path=path,
        psi0_sha256=np.asarray(psi0_hash),
        learning_rate=np.asarray(float(learning_rate)),
        steps=np.asarray(int(steps)),
        elapsed_wall_s=np.asarray(time.time() - t0),
    )
    return path, {
        "ite_cache_path": str(cache_path),
        "ite_cache_status": "computed",
        "ite_initial_state_sha256": psi0_hash,
    }


def prepare_shared_initial_checkpoint(args) -> None:
    if args.initial_checkpoint is not None:
        print(f"Using user-provided initial checkpoint: {args.initial_checkpoint.resolve()}")
        return

    run_dir = args.run_dir.resolve()
    token = (
        f"{args.system}_{args.model}_seed{int(args.seed)}"
        f"_samples{int(args.n_samples)}_L{int(args.lattice_size)}_h{float_token(args.tfim_h)}"
    )
    checkpoint_path = run_dir / "shared_initial" / f"{token}.pkl"
    if checkpoint_path.exists() and not args.force:
        args.initial_checkpoint = checkpoint_path
        print(f"Using existing shared initial checkpoint: {checkpoint_path}")
        return

    bundle = make_system_bundle(args.system, lattice_size=args.lattice_size, tfim_h=args.tfim_h)
    exact = load_or_compute_exact_state(
        bundle,
        cache_dir=args.exact_cache_dir,
        force=args.force_exact,
    )
    vstate = make_vstate(
        bundle=bundle,
        model_name=args.model,
        n_samples=args.n_samples,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )
    initial_metrics, psi0 = exact_metrics_from_variables(
        vstate.variables,
        bundle,
        exact,
        model_name=args.model,
        chunk_size=args.fullsum_chunk_size,
    )
    psi0_hash = state_vector_sha256(psi0)
    psi_path = checkpoint_path.with_suffix(".psi0.npz")
    atomic_savez(
        psi_path,
        psi0=np.asarray(psi0, dtype=np.complex128),
        psi0_sha256=np.asarray(psi0_hash),
    )
    payload = {
        "step": 0,
        "parameters": numpy_tree(vstate.parameters),
        "variables": numpy_tree(vstate.variables),
        "sampler_state": numpy_tree(getattr(vstate, "sampler_state", None)),
        "initial_state_sha256": psi0_hash,
        "initial_state_vector": str(psi_path),
        "initial_exact_metrics": initial_metrics,
        "config": {
            "system": args.system,
            "model": args.model,
            "seed": int(args.seed),
            "n_samples": int(args.n_samples),
            "lattice_size": int(args.lattice_size),
            "tfim_h": float(args.tfim_h),
            "chunk_size": int(args.chunk_size),
        },
    }
    atomic_write_pickle(checkpoint_path, payload)
    atomic_write(
        checkpoint_path.with_suffix(".json"),
        {
            "checkpoint": str(checkpoint_path),
            "initial_state_vector": str(psi_path),
            "initial_state_sha256": psi0_hash,
            "initial_exact_metrics": initial_metrics,
        },
    )
    args.initial_checkpoint = checkpoint_path
    print(f"Wrote shared initial checkpoint: {checkpoint_path}")
    print(f"Shared psi0 sha256: {psi0_hash}")


def exact_dense_sr_update(
    driver,
    states: np.ndarray,
    *,
    pdf: np.ndarray,
    diag_shift: float,
    solver_name: str,
    chunk_size: int,
):
    variables = driver.state.variables
    afun, variables = _resolve_distribution(driver, variables)
    samples = jnp.asarray(states)
    local_grad, _ = driver._kernel(afun, variables, samples, driver._ham)
    model_state, params = fcore.pop(variables, "params")
    delta, _, _ = dense_sr(
        afun,
        local_grad,
        params,
        model_state,
        samples,
        pdf=jnp.asarray(pdf),
        diag_shift=float(diag_shift),
        solver_fn=resolve_solver(solver_name),
        mode=driver.mode,
        proj_reg=None,
        momentum=None,
        old_updates=None,
        chunk_size=int(chunk_size),
        collect_quadratic_model=False,
        collect_gradient_statistics=False,
    )
    leaves = jax.tree_util.tree_leaves(delta)
    if leaves:
        jax.block_until_ready(leaves[0])
    return delta


class ExactUpdateTrackingCallback(AbstractCallback, mutable=True):
    _results_dir: str = struct.field(pytree_node=False)
    _method: str = struct.field(pytree_node=False)
    _model_name: str = struct.field(pytree_node=False)
    _bundle: Any = struct.field(pytree_node=False)
    _exact: Any = struct.field(pytree_node=False)
    _states: np.ndarray = struct.field(pytree_node=False)
    _ideal_diag_shift: float = struct.field(pytree_node=False)
    _ideal_solver: str = struct.field(pytree_node=False)
    _exact_chunk_size: int = struct.field(pytree_node=False)
    _fullsum_chunk_size: int = struct.field(pytree_node=False)
    _track_every: int = struct.field(pytree_node=False)
    _step_offset: int = struct.field(pytree_node=False)
    _ite_path: np.ndarray | None = struct.field(pytree_node=False)
    _order: int = struct.field(pytree_node=False)

    def __init__(
        self,
        *,
        results_dir: Path,
        method: str,
        model_name: str,
        bundle,
        exact,
        states: np.ndarray,
        ideal_diag_shift: float,
        ideal_solver: str,
        exact_chunk_size: int,
        fullsum_chunk_size: int,
        track_every: int,
        step_offset: int,
        ite_path: np.ndarray | None = None,
        order: int = 100,
    ):
        self._results_dir = str(results_dir)
        self._method = str(method)
        self._model_name = str(model_name)
        self._bundle = bundle
        self._exact = exact
        self._states = np.asarray(states)
        self._ideal_diag_shift = float(ideal_diag_shift)
        self._ideal_solver = str(ideal_solver)
        self._exact_chunk_size = int(exact_chunk_size)
        self._fullsum_chunk_size = int(fullsum_chunk_size)
        self._track_every = int(track_every)
        self._step_offset = int(step_offset)
        self._ite_path = None if ite_path is None else np.asarray(ite_path)
        self._order = int(order)

    @property
    def callback_order(self) -> int:
        return self._order

    def on_compute_update_end(self, step, log_data, driver):
        local_step = int(step) + 1
        update_step = self._step_offset + local_step
        if self._track_every <= 0 or local_step % self._track_every != 0:
            return

        metric_t0 = time.time()
        variables = driver.state.variables
        afun, variables = _resolve_distribution(driver, variables)
        pdf = born_pdf(
            afun,
            variables,
            self._states,
            chunk_size=self._exact_chunk_size,
        )

        solve_t0 = time.time()
        ideal = exact_dense_sr_update(
            driver,
            self._states,
            pdf=pdf,
            diag_shift=self._ideal_diag_shift,
            solver_name=self._ideal_solver,
            chunk_size=self._exact_chunk_size,
        )
        solve_wall = time.time() - solve_t0

        update = driver._dp
        s_metrics = s_metric_update_metrics(
            afun=afun,
            variables=variables,
            states=self._states,
            pdf=pdf,
            update=update,
            ideal=ideal,
            mode=driver.mode,
            chunk_size=self._exact_chunk_size,
        )
        flat_metrics = l2_metrics(update, ideal)
        exact_row, psi = exact_metrics_and_state_from_driver(
            driver,
            self._bundle,
            self._exact,
            model_name=self._model_name,
            chunk_size=self._fullsum_chunk_size,
        )
        ite_index = None
        ite_metrics = {"ite_fidelity": None, "ite_trace_distance": None}
        if self._ite_path is not None:
            ite_index = local_step - 1
            if ite_index < 0 or ite_index >= self._ite_path.shape[0]:
                raise IndexError(
                    f"ITE index {ite_index} outside cached path length {self._ite_path.shape[0]}"
                )
            ite_metrics = pure_state_overlap_metrics(self._ite_path[ite_index], psi)

        multilambda = None
        if isinstance(getattr(driver, "info", None), dict):
            multilambda = driver.info.get("multilambda")

        loss_stats = driver._loss_stats
        row = {
            "update_step": update_step,
            "state_step": update_step - 1,
            "method": self._method,
            "ite_index": ite_index,
            "mc_energy": None if loss_stats is None else float(np.real(complex(loss_stats.Mean))),
            "mc_energy_variance": None if loss_stats is None else float(np.real(complex(loss_stats.Variance))),
            "ideal_diag_shift": self._ideal_diag_shift,
            "ideal_solve_wall_s": solve_wall,
            "metric_wall_s": time.time() - metric_t0,
            "multilambda_weights": (
                None
                if not isinstance(multilambda, dict)
                else json.dumps(multilambda.get("weights"))
            ),
            "multilambda_lambdas": (
                None
                if not isinstance(multilambda, dict)
                else json.dumps(multilambda.get("lambda_grid"))
            ),
        }
        row.update(exact_row)
        row.update(ite_metrics)
        row.update(s_metrics)
        row.update(flat_metrics)
        append_csv_row(Path(self._results_dir) / "update_tracking.csv", TRACK_HEADER, row)
        print(
            "  "
            f"step={update_step:03d} "
            f"E={row['exact_energy']:.10f} "
            f"gap={row['exact_energy_gap']:.3e} "
            f"F={row['fidelity']:.8f} "
            f"D_ITE={row['ite_trace_distance'] if row['ite_trace_distance'] is not None else float('nan'):.3e} "
            f"Srel={row['s_relative_error']:.3e} "
            f"Scos={row['s_cosine']:.4f}",
            flush=True,
        )


def run_method(args, *, method: str) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    results_dir = run_dir / "results" / args.system / args.model / method
    if results_dir.exists() and args.force:
        shutil.rmtree(results_dir)
    if (results_dir / "summary.json").exists() and not args.force:
        with (results_dir / "summary.json").open() as handle:
            summary = json.load(handle)
        if summary.get("status") == "completed":
            print(f"[skip] completed branch exists: {results_dir}")
            return summary
        raise SystemExit(f"Existing incomplete branch; rerun with --force: {results_dir}")

    results_dir.mkdir(parents=True, exist_ok=True)
    bundle = make_system_bundle(args.system, lattice_size=args.lattice_size, tfim_h=args.tfim_h)
    exact = load_or_compute_exact_state(
        bundle,
        cache_dir=args.exact_cache_dir,
        force=args.force_exact,
    )
    states = all_hilbert_states(bundle.hilbert)
    paper_grid = parse_float_tuple(args.paper_grid, PAPER_GRID)
    quantiles = parse_float_tuple(args.ntk_quantiles, NTK_QUANTILES)
    grid, grid_meta = method_grid_config(method, paper_grid=paper_grid, quantiles=quantiles)
    dynamic_ntk_quantiles = (
        quantiles if method in ("mssr_ntk_quantile", "mssr_ema_ntk_quantile") else None
    )

    mode = "complex" if model_config(args.model)["family"] == "rbm" else None
    vstate = make_vstate(
        bundle=bundle,
        model_name=args.model,
        n_samples=args.n_samples,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )
    initial_checkpoint_step = 0
    expected_initial_sha = None
    if args.initial_checkpoint is not None:
        checkpoint_path = args.initial_checkpoint.resolve()
        checkpoint = load_checkpoint(checkpoint_path)
        initial_checkpoint_step = int(checkpoint.get("step", 0))
        expected_initial_sha = checkpoint.get("initial_state_sha256")
        variables = checkpoint.get("variables")
        parameters = checkpoint.get("parameters")
        sampler_state = checkpoint.get("sampler_state")
        if variables is not None:
            vstate.variables = jax.tree_util.tree_map(jnp.asarray, variables)
        if parameters is not None:
            vstate.parameters = jax.tree_util.tree_map(jnp.asarray, parameters)
        if sampler_state is not None and hasattr(vstate, "sampler_state"):
            vstate.sampler_state = jax.tree_util.tree_map(jnp.asarray, sampler_state)
        print(
            f"Loaded initial checkpoint step={initial_checkpoint_step} "
            f"path={checkpoint_path}"
        )
    step_offset = (
        int(args.step_offset)
        if args.step_offset is not None
        else int(initial_checkpoint_step)
    )
    n_parameters = count_parameters(vstate.parameters)
    diag_shift = float(args.sr_lambda)
    if grid is not None:
        diag_shift = float(max(grid))

    initial_metrics, initial_psi = exact_metrics_from_variables(
        vstate.variables,
        bundle,
        exact,
        model_name=args.model,
        chunk_size=args.fullsum_chunk_size,
    )
    initial_sha = state_vector_sha256(initial_psi)
    if expected_initial_sha is not None and str(expected_initial_sha) != initial_sha:
        raise ValueError(
            "Loaded initial checkpoint does not reproduce its recorded state hash: "
            f"expected={expected_initial_sha} actual={initial_sha}"
        )
    atomic_write(
        results_dir / "initial_state.json",
        {
            "step": int(step_offset),
            "initial_state_sha256": initial_sha,
            "metrics": initial_metrics,
        },
    )

    ite_path = None
    ite_meta: dict[str, Any] = {
        "ite_cache_path": None,
        "ite_cache_status": None,
        "ite_initial_state_sha256": None,
    }
    if args.track_ite:
        ite_path, ite_meta = load_or_compute_ite_path(
            bundle,
            initial_psi,
            learning_rate=args.learning_rate,
            steps=args.steps,
            cache_dir=run_dir / "ite_paths",
            force=bool(args.force_ite),
        )

    config = {
        "root": str(ROOT),
        "run_dir": str(run_dir),
        "results_dir": str(results_dir),
        "system": args.system,
        "display_name": bundle.display_name,
        "lattice_size": int(args.lattice_size),
        "tfim_h": None if bundle.tfim_h is None else float(bundle.tfim_h),
        "n_sites": int(bundle.n_sites),
        "hilbert_dim": int(bundle.hilbert_dim),
        "model": args.model,
        "model_config": model_config(args.model),
        "method": method,
        "seed": int(args.seed),
        "n_samples": int(args.n_samples),
        "n_parameters": int(n_parameters),
        "learning_rate": float(args.learning_rate),
        "total_steps": int(args.steps),
        "track_every": int(args.track_every),
        "step_offset": int(step_offset),
        "initial_checkpoint": (
            None if args.initial_checkpoint is None else str(args.initial_checkpoint.resolve())
        ),
        "initial_checkpoint_step": int(initial_checkpoint_step),
        "initial_state_sha256": initial_sha,
        "initial_exact_metrics": initial_metrics,
        "sr_lambda": float(args.sr_lambda),
        "spring_momentum": float(args.spring_momentum),
        "mssr_grid": None if grid is None else [float(value) for value in grid],
        "mssr_grid_meta": grid_meta,
        "mssr_dynamic_ntk_quantiles": dynamic_ntk_quantiles is not None,
        "mssr_lambda_quantiles": (
            None if dynamic_ntk_quantiles is None else [float(value) for value in dynamic_ntk_quantiles]
        ),
        "ideal_diag_shift": float(args.ideal_diag_shift),
        "ideal_solver": args.ideal_solver,
        "chunk_size": int(args.chunk_size),
        "chunk_size_bwd": int(args.chunk_size_bwd or args.n_samples),
        "exact_chunk_size": int(args.exact_chunk_size),
        "fullsum_chunk_size": int(args.fullsum_chunk_size),
        "track_ite": bool(args.track_ite),
        **ite_meta,
        "ground_state_energy": float(exact.energy),
        "ground_state_cache": str(exact.cache_path),
    }
    atomic_write_json(results_dir / "config.json", config)
    atomic_write(results_dir / "summary.json", {**config, "status": "running"})

    print("=" * 80)
    print(
        f"Exact-update trajectory: method={method} model={args.model} "
        f"system={args.system} seed={args.seed} samples={args.n_samples} steps={args.steps}"
    )
    print(f"Output: {results_dir}")
    print(
        f"N={bundle.n_sites} dim={bundle.hilbert_dim} P={n_parameters} "
        f"E0={exact.energy:.12f}"
    )
    print("=" * 80)

    driver = make_driver(
        bundle=bundle,
        vstate=vstate,
        method=method,
        learning_rate=args.learning_rate,
        diag_shift=diag_shift,
        spring_momentum=args.spring_momentum,
        chunk_size_bwd=int(args.chunk_size_bwd or args.n_samples),
        mode=mode,
    )
    driver.collect_residual_info = False

    callbacks = []
    if grid is not None:
        callbacks.append(
            MultiLambdaSRCallback(
                sr_variant="indep_batch",
                sr_weight_scheme="stacking",
                sr_lambda_grid=grid,
                sr_weight_fit_samples=int(args.n_samples),
                sr_lambda_quantiles=dynamic_ntk_quantiles,
                sr_lambda_ema_rho=None,
                sr_weight_ema_rho=(
                    float(args.mssr_ema_rho)
                    if method in ("mssr_ema_paper_fixed", "mssr_ema_ntk_quantile")
                    else None
                ),
            )
        )
    callbacks.append(
        ExactUpdateTrackingCallback(
            results_dir=results_dir,
            method=method,
            model_name=args.model,
            bundle=bundle,
            exact=exact,
            states=states,
            ideal_diag_shift=args.ideal_diag_shift,
            ideal_solver=args.ideal_solver,
            exact_chunk_size=args.exact_chunk_size,
            fullsum_chunk_size=args.fullsum_chunk_size,
            track_every=args.track_every,
            step_offset=step_offset,
            ite_path=ite_path,
        )
    )

    started = time.time()
    status = "error"
    error = None
    try:
        driver.run(n_iter=int(args.steps), out=None, callback=callbacks)
        status = "completed"
    except Exception as exc:
        status = "error"
        error = repr(exc)
        raise
    finally:
        final_step = step_offset + int(getattr(driver, "step_count", 0))
        final_exact_row = None
        final_exact_error = None
        try:
            final_exact_row, final_psi = exact_metrics_and_state_from_driver(
                driver,
                bundle,
                exact,
                model_name=args.model,
                chunk_size=args.fullsum_chunk_size,
            )
            if ite_path is not None:
                final_ite_index = max(0, min(final_step - step_offset, ite_path.shape[0] - 1))
                final_exact_row.update(
                    {
                        "ite_index": int(final_ite_index),
                        **pure_state_overlap_metrics(ite_path[final_ite_index], final_psi),
                    }
                )
            atomic_write(
                results_dir / "final_exact_metrics.json",
                {"step": final_step, **final_exact_row},
            )
        except Exception as metric_exc:
            final_exact_error = repr(metric_exc)

        checkpoint_payload = final_checkpoint_payload(
            step=final_step,
            driver=driver,
            config=config,
            final_exact_metrics=final_exact_row,
            status=status,
            error=error,
        )
        checkpoints_dir = results_dir / "checkpoints"
        checkpoint_path = checkpoints_dir / f"checkpoint_step{final_step:06d}_final.pkl"
        atomic_write_pickle(checkpoint_path, checkpoint_payload)
        atomic_write_pickle(results_dir / "checkpoint_final.pkl", checkpoint_payload)
        summary = {
            **config,
            "status": status,
            "error": error,
            "elapsed_wall_s": time.time() - started,
            "updated_at_epoch_s": time.time(),
            "tracking_csv": str(results_dir / "update_tracking.csv"),
            "final_step": final_step,
            "final_checkpoint": str(results_dir / "checkpoint_final.pkl"),
            "final_checkpoint_numbered": str(checkpoint_path),
            "final_exact_metrics": final_exact_row,
            "final_exact_metrics_error": final_exact_error,
        }
        atomic_write(results_dir / "summary.json", summary)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/online_tracking/run"),
    )
    parser.add_argument("--methods", type=str, default="sr,spring,mssr_paper_fixed")
    parser.add_argument("--system", choices=SYSTEMS, default="j1j2")
    parser.add_argument("--model", choices=("rbm_a2", "rbm_a4", "vit_small", "vit_medium"), default="rbm_a2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--track-every", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--lattice-size", type=int, default=4)
    parser.add_argument("--tfim-h", type=float, default=1.0)
    parser.add_argument("--sr-lambda", type=float, default=1e-4)
    parser.add_argument("--spring-momentum", type=float, default=0.8)
    parser.add_argument("--mssr-ema-rho", type=float, default=0.9)
    parser.add_argument("--paper-grid", type=str, default=",".join(str(value) for value in PAPER_GRID))
    parser.add_argument("--ntk-quantiles", type=str, default=",".join(str(value) for value in NTK_QUANTILES))
    parser.add_argument("--ideal-diag-shift", type=float, default=0.0)
    parser.add_argument("--ideal-solver", choices=("pinv", "pinv_smooth"), default="pinv")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--chunk-size-bwd", type=int, default=None)
    parser.add_argument("--exact-chunk-size", type=int, default=2048)
    parser.add_argument("--fullsum-chunk-size", type=int, default=4096)
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--step-offset", type=int, default=None)
    parser.add_argument(
        "--shared-initial-state",
        action="store_true",
        help="Create one explicit psi0 checkpoint and reuse it for every requested method.",
    )
    parser.add_argument(
        "--track-ite",
        action="store_true",
        help="Track trace distance to the exact imaginary-time trajectory from psi0.",
    )
    parser.add_argument("--force-ite", action="store_true")
    parser.add_argument(
        "--exact-cache-dir",
        type=Path,
        default=Path("outputs/online_tracking/exact_states"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-exact", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    methods = parse_csv(args.methods)
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown method(s): {unknown}. Valid methods: {METHODS}")

    if args.shared_initial_state:
        prepare_shared_initial_checkpoint(args)

    summaries = []
    for method in methods:
        summaries.append(run_method(args, method=method))

    manifest_path = args.run_dir.resolve() / "summary.json"
    atomic_write(
        manifest_path,
        {
            "status": "completed",
            "run_dir": str(args.run_dir.resolve()),
            "methods": list(methods),
            "branches": summaries,
        },
    )
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
