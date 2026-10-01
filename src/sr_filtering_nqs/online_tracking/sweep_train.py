#!/usr/bin/env python
"""Small Figure 5 sweep trainer with exact full-summation histories.

This script is intentionally self-contained: each invocation runs one branch
defined by (seed, sample count, system, ansatz, optimizer).  The launcher
scripts in ``Figure5_small/run{seed}_{samples}/`` fan out over the full grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np

import advanced_drivers as advd
from advanced_drivers.callbacks import AbstractCallback
from netket.utils import struct
from nqs_nets.net.wrappers import ViTNd

ROOT = Path.cwd()

from sr_filtering_nqs.large_scale.core.common import make_system, parse_sr_lambda_grid
from sr_filtering_nqs.large_scale.core.multilambda_sr import MultiLambdaSRCallback


SYSTEMS = ("ssm", "j1j2", "tfim2d")
MODELS = ("vit_small", "vit_medium", "rbm_a2", "rbm_a4")
METHODS = (
    "sr",
    "spring",
    "mssr_paper_fixed",
    "mssr_ntk_quantile",
    "mssr_ema_paper_fixed",
    "mssr_ema_ntk_quantile",
)
PAPER_GRID = (1e-2, 1e-4, 1e-6, 1e-8)
NTK_QUANTILES = (0.9, 0.7, 0.4, 0.1)
EXACT_STEPS_DEFAULT = tuple(range(10, 201, 10))

TRAIN_HEADER = [
    "step",
    "energy",
    "energy_per_site",
    "energy_sigma",
    "energy_variance",
    "learning_rate",
    "acceptance",
    "step_wall_time_s",
    "elapsed_wall_time_s",
    "multilambda_weights",
    "multilambda_lambdas",
]

EXACT_HEADER = [
    "step",
    "exact_energy",
    "exact_energy_per_site",
    "exact_variance",
    "exact_variance_per_site",
    "exact_energy_gap",
    "exact_relative_gap_abs",
    "exact_relative_gap_signed",
    "exact_v_score",
    "fidelity",
    "trace_distance",
]


class TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


@contextmanager
def tee_stdout(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    with log_path.open("a", buffering=1) as log_file:
        tee = TeeStream(old_stdout, log_file)
        sys.stdout = tee
        sys.stderr = tee
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


@dataclass(frozen=True)
class SystemBundle:
    name: str
    display_name: str
    lattice_size: int
    n_sites: int
    hilbert: Any
    graph: Any
    hamiltonian: Any
    hilbert_dim: int
    tfim_h: float | None = None


@dataclass(frozen=True)
class ExactState:
    energy: float
    vector: np.ndarray
    cache_path: Path
    computed: bool


def json_ready(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return [json_ready(value) for value in obj]
    if isinstance(obj, list):
        return [json_ready(value) for value in obj]
    if isinstance(obj, dict):
        return {str(key): json_ready(value) for key, value in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (np.complexfloating, complex)):
        value = complex(obj)
        return {"real": float(value.real), "imag": float(value.imag)}
    return obj


def atomic_write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def atomic_write_pickle(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".pkl.tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payload, handle)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def append_csv_row(path: Path, header: list[str], row: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in header})


def parse_float_tuple(raw: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if raw is None:
        return default
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def count_parameters(params) -> int:
    return int(sum(np.prod(np.asarray(leaf).shape) for leaf in jax.tree_util.tree_leaves(params)))


def method_label(method: str) -> str:
    labels = {
        "sr": "SR lambda=1e-4",
        "spring": "SPRING lambda=1e-4 momentum=0.8",
        "mssr_paper_fixed": "MS-SR paper_fixed",
        "mssr_ntk_quantile": "MS-SR ntk_quantile",
        "mssr_ema_paper_fixed": "MS-SR paper_fixed + EMA weights",
        "mssr_ema_ntk_quantile": "MS-SR ntk_quantile + EMA weights",
    }
    return labels[method]


def model_config(model_name: str) -> dict[str, Any]:
    if model_name == "vit_small":
        return {
            "family": "vit",
            "label": "ViT small",
            "depth": 1,
            "d_model": 16,
            "heads": 4,
            "expansion_factor": 4,
        }
    if model_name == "vit_medium":
        return {
            "family": "vit",
            "label": "ViT medium",
            "depth": 2,
            "d_model": 24,
            "heads": 4,
            "expansion_factor": 4,
        }
    if model_name == "rbm_a2":
        return {"family": "rbm", "label": "RBM alpha=2", "alpha": 2}
    if model_name == "rbm_a4":
        return {"family": "rbm", "label": "RBM alpha=4", "alpha": 4}
    raise ValueError(f"Unknown model={model_name!r}")


def make_system_bundle(name: str, *, lattice_size: int, tfim_h: float) -> SystemBundle:
    if name in ("j1j2", "ssm"):
        system = make_system(name, L=lattice_size)
        return SystemBundle(
            name=name,
            display_name=f"{name} {lattice_size}x{lattice_size}",
            lattice_size=lattice_size,
            n_sites=int(system.graph.n_nodes),
            hilbert=system.hilbert_space,
            graph=system.graph,
            hamiltonian=system.hamiltonian,
            hilbert_dim=int(system.hilbert_space.n_states),
            tfim_h=None,
        )
    if name == "tfim2d":
        graph = nk.graph.Square(length=lattice_size, pbc=True)
        hilbert = nk.hilbert.Spin(s=0.5, N=int(graph.n_nodes))
        hamiltonian = nk.operator.Ising(
            hilbert,
            graph,
            h=float(tfim_h),
            J=1.0,
            dtype=float,
        )
        return SystemBundle(
            name=name,
            display_name=f"TFIM {lattice_size}x{lattice_size} h={tfim_h:g}",
            lattice_size=lattice_size,
            n_sites=int(graph.n_nodes),
            hilbert=hilbert,
            graph=graph,
            hamiltonian=hamiltonian,
            hilbert_dim=int(hilbert.n_states),
            tfim_h=float(tfim_h),
        )
    raise ValueError(f"Unknown system={name!r}")


def make_model(bundle: SystemBundle, model_name: str):
    cfg = model_config(model_name)
    if cfg["family"] == "vit":
        class _ViTSystem:
            graph = bundle.graph

        return ViTNd(
            depth=int(cfg["depth"]),
            d_model=int(cfg["d_model"]),
            heads=int(cfg["heads"]),
            expansion_factor=int(cfg["expansion_factor"]),
            output_head="Vanilla",
            system=_ViTSystem(),
        ).network
    if cfg["family"] == "rbm":
        return nk.models.RBM(alpha=int(cfg["alpha"]), param_dtype=complex)
    raise ValueError(f"Unsupported model config family={cfg['family']!r}")


def make_sampler(bundle: SystemBundle, *, n_samples: int):
    if bundle.name in ("j1j2", "ssm"):
        return nk.sampler.MetropolisExchange(
            bundle.hilbert,
            graph=bundle.graph,
            d_max=2,
            n_chains=int(n_samples),
        )
    return nk.sampler.MetropolisLocal(bundle.hilbert, n_chains=int(n_samples))


def exact_cache_path(cache_dir: Path, bundle: SystemBundle) -> Path:
    if bundle.tfim_h is None:
        return cache_dir / f"{bundle.name}_L{bundle.lattice_size}_N{bundle.n_sites}.npz"
    token = f"{bundle.tfim_h:.8f}".rstrip("0").rstrip(".").replace(".", "p")
    return cache_dir / f"{bundle.name}_L{bundle.lattice_size}_N{bundle.n_sites}_h{token}.npz"


def load_or_compute_exact_state(bundle: SystemBundle, *, cache_dir: Path, force: bool = False) -> ExactState:
    cache_path = exact_cache_path(cache_dir, bundle)
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as data:
            return ExactState(
                energy=float(np.asarray(data["energy"]).reshape(())),
                vector=np.asarray(data["vector"]),
                cache_path=cache_path,
                computed=False,
            )

    eigenvalues, eigenvectors = nk.exact.lanczos_ed(
        bundle.hamiltonian,
        k=1,
        compute_eigenvectors=True,
    )
    vector = np.asarray(eigenvectors)
    if vector.ndim == 2:
        vector = vector[:, 0]
    vector = np.asarray(vector, dtype=np.result_type(vector.dtype, np.complex128))
    vector = vector / np.linalg.norm(vector)
    energy = float(np.asarray(eigenvalues).reshape(-1)[0])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, energy=np.asarray(energy), vector=vector)
    return ExactState(energy=energy, vector=vector, cache_path=cache_path, computed=True)


def normalized_state_vector(vstate) -> np.ndarray:
    psi = np.asarray(vstate.to_array(normalize=True))
    norm = float(np.linalg.norm(psi))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("FullSumState produced a non-normalizable vector")
    return psi / norm


def fidelity_with_ground_state(psi: np.ndarray, ground_state: np.ndarray) -> float:
    fidelity = float(np.abs(np.vdot(ground_state, psi)) ** 2)
    return float(np.clip(fidelity, 0.0, 1.0))


def trace_distance_from_fidelity(fidelity: float) -> float:
    return float(math.sqrt(max(0.0, 1.0 - float(fidelity))))


def exact_metrics(driver, bundle: SystemBundle, exact: ExactState, *, chunk_size: int) -> dict[str, float]:
    vstate = nk.vqs.FullSumState(
        bundle.hilbert,
        model=driver.state.model,
        variables=driver.state.variables,
        chunk_size=chunk_size,
    )
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
    }


def optimizer_info(driver):
    info = getattr(driver, "info", None)
    if not isinstance(info, dict):
        return None
    return json_ready(info.get("multilambda"))


def checkpoint_payload(
    *,
    step: int,
    driver,
    config: dict[str, Any],
    train_metrics: dict[str, Any] | None,
    exact_row: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "step": int(step),
        "parameters": jax.tree_util.tree_map(np.asarray, driver.state.parameters),
        "variables": jax.tree_util.tree_map(np.asarray, driver.state.variables),
        "config": config,
        "train_metrics": train_metrics,
        "exact_metrics": exact_row,
        "optimizer_update": optimizer_info(driver),
        "multilambda": optimizer_info(driver),
    }


class ExactHistoryCallback(AbstractCallback, mutable=True):
    _results_dir: str = struct.field(pytree_node=False)
    _config: dict = struct.field(pytree_node=False)
    _bundle: SystemBundle = struct.field(pytree_node=False)
    _exact: ExactState = struct.field(pytree_node=False)
    _exact_every: int = struct.field(pytree_node=False)
    _checkpoint_every: int = struct.field(pytree_node=False)
    _fullsum_chunk_size: int = struct.field(pytree_node=False)
    _learning_rate: float = struct.field(pytree_node=False)
    _step_offset: int = struct.field(pytree_node=False)
    _run_t0: float = struct.field(pytree_node=False)
    _step_t0: float = struct.field(pytree_node=False)
    _last_train_metrics: dict = struct.field(pytree_node=False)
    _last_exact_metrics: dict = struct.field(pytree_node=False)

    def __init__(
        self,
        *,
        results_dir: Path,
        config: dict[str, Any],
        bundle: SystemBundle,
        exact: ExactState,
        exact_every: int,
        checkpoint_every: int,
        fullsum_chunk_size: int,
        learning_rate: float,
        step_offset: int,
        run_start_time: float,
    ):
        self._results_dir = str(results_dir)
        self._config = config
        self._bundle = bundle
        self._exact = exact
        self._exact_every = int(exact_every)
        self._checkpoint_every = int(checkpoint_every)
        self._fullsum_chunk_size = int(fullsum_chunk_size)
        self._learning_rate = float(learning_rate)
        self._step_offset = int(step_offset)
        self._run_t0 = float(run_start_time)
        self._step_t0 = 0.0
        self._last_train_metrics = {}
        self._last_exact_metrics = {}

    @property
    def latest_train_metrics(self):
        return self._last_train_metrics

    @property
    def latest_exact_metrics(self):
        return self._last_exact_metrics

    def on_step_start(self, step, log_data, driver):
        self._step_t0 = time.time()

    def on_compute_update_end(self, step, log_data, driver):
        loss_stats = driver._loss_stats
        if loss_stats is None:
            return
        energy = float(np.real(complex(loss_stats.Mean)))
        variance = float(np.real(complex(loss_stats.Variance)))
        sigma = float(np.real(complex(loss_stats.Sigma)))
        if np.isnan(energy) or np.isnan(variance):
            driver._stop_run = True
            self._last_train_metrics = {
                "step": self._step_offset + int(step) + 1,
                "status": "nan",
            }
            return

        completed_step = self._step_offset + int(step) + 1
        multilambda = optimizer_info(driver)
        self._last_train_metrics = {
            "step": completed_step,
            "energy": energy,
            "energy_per_site": energy / float(self._bundle.n_sites),
            "energy_sigma": sigma,
            "energy_variance": variance,
            "learning_rate": self._learning_rate,
            "acceptance": None,
            "step_wall_time_s": None,
            "elapsed_wall_time_s": None,
            "multilambda_weights": (
                json.dumps(multilambda.get("weights")) if isinstance(multilambda, dict) else None
            ),
            "multilambda_lambdas": (
                json.dumps(multilambda.get("lambda_grid")) if isinstance(multilambda, dict) else None
            ),
        }

    def on_step_end(self, step, log_data, driver):
        completed_step = self._step_offset + int(step) + 1
        if not self._last_train_metrics:
            return

        acceptance = log_data.get("acceptance")
        row = dict(self._last_train_metrics)
        row["acceptance"] = None if acceptance is None else float(np.asarray(acceptance))
        row["step_wall_time_s"] = time.time() - self._step_t0
        row["elapsed_wall_time_s"] = time.time() - self._run_t0
        append_csv_row(Path(self._results_dir) / "train_metrics.csv", TRAIN_HEADER, row)
        self._last_train_metrics = row

        should_exact = self._exact_every > 0 and completed_step % self._exact_every == 0
        should_checkpoint = self._checkpoint_every > 0 and completed_step % self._checkpoint_every == 0
        exact_row = None
        if should_exact:
            exact_row = {"step": completed_step}
            exact_row.update(
                exact_metrics(
                    driver,
                    self._bundle,
                    self._exact,
                    chunk_size=self._fullsum_chunk_size,
                )
            )
            append_csv_row(Path(self._results_dir) / "exact_history.csv", EXACT_HEADER, exact_row)
            self._last_exact_metrics = exact_row
            print(
                "  "
                f"step={completed_step:04d} "
                f"E_exact={exact_row['exact_energy']:.10f} "
                f"gap={exact_row['exact_energy_gap']:.3e} "
                f"Var={exact_row['exact_variance']:.3e} "
                f"F={exact_row['fidelity']:.8f}",
                flush=True,
            )

        if should_checkpoint:
            payload = checkpoint_payload(
                step=completed_step,
                driver=driver,
                config=self._config,
                train_metrics=row,
                exact_row=exact_row,
            )
            atomic_write_pickle(
                Path(self._results_dir) / "checkpoints" / f"checkpoint_step{completed_step:06d}.pkl",
                payload,
            )


def write_summary(
    results_dir: Path,
    config: dict[str, Any],
    *,
    status: str,
    exact: ExactState | None,
    train_metrics: dict[str, Any] | None,
    exact_metrics_row: dict[str, Any] | None,
    error: str | None = None,
):
    summary = {
        "status": status,
        "updated_at_epoch_s": time.time(),
        "system": config["system"],
        "model": config["model"],
        "method": config["method"],
        "seed": config["seed"],
        "n_samples": config["n_samples"],
        "n_sites": config["n_sites"],
        "hilbert_dim": config["hilbert_dim"],
        "n_parameters": config["n_parameters"],
        "learning_rate": config["learning_rate"],
        "total_steps": config["total_steps"],
        "exact_every": config["exact_every"],
        "train_metrics": train_metrics,
        "exact_metrics": exact_metrics_row,
        "ground_state_energy": None if exact is None else exact.energy,
        "ground_state_cache": None if exact is None else str(exact.cache_path),
    }
    if error is not None:
        summary["error"] = error
    atomic_write_json(results_dir / "summary.json", summary)


def make_vstate(
    *,
    bundle: SystemBundle,
    model_name: str,
    n_samples: int,
    seed: int,
    chunk_size: int,
):
    sampler = make_sampler(bundle, n_samples=n_samples)
    model = make_model(bundle, model_name)
    return nk.vqs.MCState(
        sampler,
        model=model,
        n_samples=int(n_samples),
        chunk_size=int(chunk_size),
        seed=int(seed),
        sampler_seed=int(seed) + 1729,
    )


def make_driver(
    *,
    bundle: SystemBundle,
    vstate,
    method: str,
    learning_rate: float,
    diag_shift: float,
    spring_momentum: float,
    chunk_size_bwd: int,
    mode: str | None,
):
    optimizer = nk.optimizer.Sgd(learning_rate=float(learning_rate))
    return advd.driver.VMC_NG(
        hamiltonian=bundle.hamiltonian,
        optimizer=optimizer,
        diag_shift=float(diag_shift),
        proj_reg=None,
        momentum=float(spring_momentum) if method == "spring" else None,
        variational_state=vstate,
        chunk_size_bwd=int(chunk_size_bwd),
        use_ntk=True,
        linear_solver_fn=nk.optimizer.solver.pinv_smooth,
        mode=mode,
    )


def method_grid_config(
    method: str,
    *,
    paper_grid: tuple[float, ...],
    quantiles: tuple[float, ...],
) -> tuple[tuple[float, ...] | None, dict[str, Any]]:
    if method in ("mssr_paper_fixed", "mssr_ema_paper_fixed"):
        return tuple(float(value) for value in paper_grid), {
            "grid_mode": "paper_fixed",
            "grid": [float(value) for value in paper_grid],
            "dynamic_per_iteration": False,
        }
    if method in ("mssr_ntk_quantile", "mssr_ema_ntk_quantile"):
        return tuple(float(value) for value in paper_grid), {
            "grid_mode": "ntk_quantile",
            "dynamic_per_iteration": True,
            "quantiles": [float(value) for value in quantiles],
            "fallback_probe_grid": [float(value) for value in paper_grid],
        }
    return None, {}


def branch_relative_path(system: str, model: str, method: str) -> Path:
    return Path("results") / system / model / method


def branch_completed(results_dir: Path) -> bool:
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        return False
    try:
        with summary_path.open() as handle:
            summary = json.load(handle)
    except json.JSONDecodeError:
        return False
    return summary.get("status") == "completed"


def branch_has_data(results_dir: Path) -> bool:
    if not results_dir.exists():
        return False
    return any(results_dir.iterdir())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--total-steps", type=int, default=200)
    parser.add_argument("--exact-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--lattice-size", type=int, default=4)
    parser.add_argument("--tfim-h", type=float, default=1.0)
    parser.add_argument("--sr-lambda", type=float, default=1e-4)
    parser.add_argument("--spring-momentum", type=float, default=0.8)
    parser.add_argument("--mssr-ema-rho", type=float, default=0.9)
    parser.add_argument("--paper-grid", type=str, default=",".join(str(value) for value in PAPER_GRID))
    parser.add_argument("--ntk-quantiles", type=str, default=",".join(str(value) for value in NTK_QUANTILES))
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--chunk-size-bwd", type=int, default=None)
    parser.add_argument("--fullsum-chunk-size", type=int, default=4096)
    parser.add_argument("--exact-cache-dir", type=Path, default=Path("outputs/online_tracking/exact_states"))
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--step-offset", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-exact", action="store_true")
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must contain a dict payload: {path}")
    return checkpoint


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    results_dir = run_dir / branch_relative_path(args.system, args.model, args.method)
    if branch_completed(results_dir) and not args.force:
        print(f"[skip] completed branch exists: {results_dir}")
        return
    if branch_has_data(results_dir) and not args.force:
        raise SystemExit(f"Branch has existing incomplete data; rerun with --force: {results_dir}")

    if args.force and results_dir.exists():
        # Preserve logs from previous attempts but replace branch data files.
        for name in ("train_metrics.csv", "exact_history.csv", "config.json", "summary.json", "grid.json"):
            path = results_dir / name
            if path.exists():
                path.unlink()
        for path in (results_dir / "checkpoints").glob("checkpoint_step*.pkl"):
            path.unlink()

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "logs" / "train.log"

    with tee_stdout(log_path):
        print("=" * 80)
        print(
            f"Figure5_small branch: system={args.system} model={args.model} "
            f"method={args.method} seed={args.seed} samples={args.n_samples}"
        )
        print(f"Output: {results_dir}")
        print("=" * 80)

        paper_grid = parse_float_tuple(args.paper_grid, PAPER_GRID)
        quantiles = parse_float_tuple(args.ntk_quantiles, NTK_QUANTILES)
        bundle = make_system_bundle(args.system, lattice_size=args.lattice_size, tfim_h=args.tfim_h)
        exact = load_or_compute_exact_state(
            bundle,
            cache_dir=args.exact_cache_dir,
            force=args.force_exact,
        )
        print(
            f"Exact E0={exact.energy:.12f} | cache={exact.cache_path} "
            f"({'computed' if exact.computed else 'loaded'})"
        )

        chunk_size_bwd = args.chunk_size_bwd
        if chunk_size_bwd is None:
            chunk_size_bwd = int(args.n_samples)

        mode = "complex" if model_config(args.model)["family"] == "rbm" else None
        initial_vstate = make_vstate(
            bundle=bundle,
            model_name=args.model,
            n_samples=args.n_samples,
            seed=args.seed,
            chunk_size=args.chunk_size,
        )
        initial_checkpoint = None
        initial_checkpoint_step = 0
        if args.initial_checkpoint is not None:
            initial_checkpoint_path = args.initial_checkpoint.resolve()
            initial_checkpoint = load_checkpoint(initial_checkpoint_path)
            initial_checkpoint_step = int(initial_checkpoint.get("step", 0))
            variables = initial_checkpoint.get("variables")
            parameters = initial_checkpoint.get("parameters")
            if variables is not None:
                initial_vstate.variables = jax.tree_util.tree_map(jnp.asarray, variables)
            if parameters is not None:
                initial_vstate.parameters = jax.tree_util.tree_map(jnp.asarray, parameters)
            print(
                f"Loaded initial checkpoint step={initial_checkpoint_step} "
                f"path={initial_checkpoint_path}"
            )
        step_offset = (
            int(args.step_offset)
            if args.step_offset is not None
            else int(initial_checkpoint_step)
        )
        n_parameters = count_parameters(initial_vstate.parameters)
        grid, grid_meta = method_grid_config(
            args.method,
            paper_grid=paper_grid,
            quantiles=quantiles,
        )
        dynamic_ntk_quantiles = (
            quantiles if args.method in ("mssr_ntk_quantile", "mssr_ema_ntk_quantile") else None
        )

        if grid is not None:
            atomic_write_json(
                results_dir / "grid.json",
                {
                    **grid_meta,
                    "method": args.method,
                    "system": args.system,
                    "model": args.model,
                    "seed": int(args.seed),
                    "n_samples": int(args.n_samples),
                },
            )

        diag_shift = float(args.sr_lambda)
        if grid is not None:
            diag_shift = float(max(grid))

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
            "method": args.method,
            "method_label": method_label(args.method),
            "seed": int(args.seed),
            "n_samples": int(args.n_samples),
            "n_parameters": int(n_parameters),
            "learning_rate": float(args.learning_rate),
            "total_steps": int(args.total_steps),
            "exact_every": int(args.exact_every),
            "checkpoint_every": int(args.checkpoint_every),
            "step_offset": int(step_offset),
            "initial_checkpoint": (
                None if args.initial_checkpoint is None else str(args.initial_checkpoint.resolve())
            ),
            "initial_checkpoint_step": int(initial_checkpoint_step),
            "sr_lambda": float(args.sr_lambda),
            "spring_momentum": float(args.spring_momentum),
            "mssr_sr_variant": "indep_batch" if grid is not None else None,
            "mssr_weight_scheme": "stacking" if grid is not None else None,
            "mssr_grid": None if grid is None else [float(value) for value in grid],
            "mssr_grid_meta": grid_meta,
            "mssr_lambda_quantiles": (
                None
                if dynamic_ntk_quantiles is None
                else [float(value) for value in dynamic_ntk_quantiles]
            ),
            "mssr_dynamic_ntk_quantiles": dynamic_ntk_quantiles is not None,
            "mssr_weight_ema_rho": (
                float(args.mssr_ema_rho)
                if args.method in ("mssr_ema_paper_fixed", "mssr_ema_ntk_quantile")
                else None
            ),
            "linear_solver": "pinv_smooth",
            "mode": mode,
            "chunk_size": int(args.chunk_size),
            "chunk_size_bwd": int(chunk_size_bwd),
            "fullsum_chunk_size": int(args.fullsum_chunk_size),
            "exact_cache": str(exact.cache_path),
            "ground_state_energy": float(exact.energy),
        }
        atomic_write_json(results_dir / "config.json", config)
        write_summary(
            results_dir,
            config,
            status="running",
            exact=exact,
            train_metrics=None,
            exact_metrics_row=None,
        )

        print(
            f"N={bundle.n_sites} hilbert_dim={bundle.hilbert_dim} "
            f"P={n_parameters} samples={args.n_samples} lr={args.learning_rate}"
        )
        print(
            f"optimizer={args.method} diag_shift={diag_shift:g} "
            f"solver=pinv_smooth chunk_bwd={chunk_size_bwd}"
        )
        if grid is not None:
            if dynamic_ntk_quantiles is None:
                print(f"MS-SR grid={','.join(f'{value:g}' for value in grid)}")
            else:
                print(
                    "MS-SR dynamic NTK quantiles="
                    f"{','.join(f'{value:g}' for value in dynamic_ntk_quantiles)} "
                    "(lambda grid recomputed every optimizer step)"
                )

        driver = make_driver(
            bundle=bundle,
            vstate=initial_vstate,
            method=args.method,
            learning_rate=args.learning_rate,
            diag_shift=diag_shift,
            spring_momentum=args.spring_momentum,
            chunk_size_bwd=chunk_size_bwd,
            mode=mode,
        )
        driver.collect_residual_info = False

        run_start_time = time.time()
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
                        if args.method in ("mssr_ema_paper_fixed", "mssr_ema_ntk_quantile")
                        else None
                    ),
                )
            )
        history_callback = ExactHistoryCallback(
            results_dir=results_dir,
            config=config,
            bundle=bundle,
            exact=exact,
            exact_every=args.exact_every,
            checkpoint_every=args.checkpoint_every,
            fullsum_chunk_size=args.fullsum_chunk_size,
            learning_rate=args.learning_rate,
            step_offset=step_offset,
            run_start_time=run_start_time,
        )
        callbacks.append(history_callback)

        try:
            driver.run(n_iter=int(args.total_steps), out=None, callback=callbacks)
        except Exception as exc:
            write_summary(
                results_dir,
                config,
                status="error",
                exact=exact,
                train_metrics=history_callback.latest_train_metrics,
                exact_metrics_row=history_callback.latest_exact_metrics,
                error=str(exc),
            )
            raise

        if history_callback.latest_train_metrics.get("status") == "nan":
            write_summary(
                results_dir,
                config,
                status="error",
                exact=exact,
                train_metrics=history_callback.latest_train_metrics,
                exact_metrics_row=history_callback.latest_exact_metrics,
                error="NaN detected during training",
            )
            raise RuntimeError("NaN detected during training")

        write_summary(
            results_dir,
            config,
            status="completed",
            exact=exact,
            train_metrics=history_callback.latest_train_metrics,
            exact_metrics_row=history_callback.latest_exact_metrics,
        )
        if history_callback.latest_exact_metrics:
            latest = history_callback.latest_exact_metrics
            print(
                f"Completed: E_exact={latest['exact_energy']:.10f} "
                f"gap={latest['exact_energy_gap']:.3e} "
                f"F={latest['fidelity']:.8f} "
                f"trace={latest['trace_distance']:.3e}"
            )
        else:
            print("Completed.")


if __name__ == "__main__":
    main()
