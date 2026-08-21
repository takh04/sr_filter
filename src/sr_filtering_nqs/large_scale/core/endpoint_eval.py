"""High-precision endpoint evaluation and statistically valid shard pooling."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ENDPOINT_SCHEMA = "sr_filter.endpoint_eval.v1"
MAX_REPLICA_DIFFERENCE_Z = 5.0


def replica_difference_z(
    energies: Iterable[float],
    sigmas: Iterable[float],
) -> float:
    """Return the largest pairwise discrepancy in combined-standard-error units."""

    energy_values = np.asarray(tuple(energies), dtype=np.float64)
    sigma_values = np.asarray(tuple(sigmas), dtype=np.float64)
    if energy_values.ndim != 1 or sigma_values.shape != energy_values.shape:
        raise ValueError("Replica energies and standard errors must be matching vectors")
    if energy_values.size < 2:
        raise ValueError("At least two endpoint replicas are required")
    if not np.all(np.isfinite(energy_values)) or not np.all(np.isfinite(sigma_values)):
        raise ValueError("Replica energies and standard errors must be finite")
    if np.any(sigma_values < 0.0):
        raise ValueError("Replica standard errors must be nonnegative")

    maximum = 0.0
    for left in range(energy_values.size):
        for right in range(left + 1, energy_values.size):
            difference = abs(float(energy_values[left] - energy_values[right]))
            denominator = math.hypot(
                float(sigma_values[left]),
                float(sigma_values[right]),
            )
            if denominator == 0.0:
                if difference != 0.0:
                    return math.inf
                continue
            maximum = max(maximum, difference / denominator)
    return float(maximum)


@dataclass(frozen=True)
class EndpointEstimate:
    """A per-site endpoint estimate with an independent standard error."""

    energy_per_site: float
    sigma_per_site: float
    total_samples: int
    source: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.energy_per_site):
            raise ValueError(f"energy_per_site must be finite, got {self.energy_per_site}")
        if not math.isfinite(self.sigma_per_site) or self.sigma_per_site < 0.0:
            raise ValueError(f"sigma_per_site must be finite and nonnegative, got {self.sigma_per_site}")
        if int(self.total_samples) <= 0:
            raise ValueError(f"total_samples must be positive, got {self.total_samples}")

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        source: str | None = None,
    ) -> "EndpointEstimate":
        sigma = payload.get("sigma_per_site")
        if sigma is None:
            sigma = payload.get("sigma_per_site_binned")
        if sigma is None:
            raise ValueError(f"Endpoint payload {source or '<mapping>'} has no per-site sigma")
        return cls(
            energy_per_site=float(payload["energy_per_site"]),
            sigma_per_site=float(sigma),
            total_samples=int(payload["total_samples"]),
            source=source,
            metadata=dict(payload),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "energy_per_site": float(self.energy_per_site),
            "sigma_per_site": float(self.sigma_per_site),
            "total_samples": int(self.total_samples),
            "source": self.source,
            "metadata": None if self.metadata is None else dict(self.metadata),
        }


def load_endpoint_estimate(path: str | Path) -> EndpointEstimate:
    path = Path(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError(f"Endpoint payload must be a JSON object: {path}")
    return EndpointEstimate.from_mapping(payload, source=path.name)


def aggregate_endpoint_estimates(
    estimates: Iterable[EndpointEstimate],
    *,
    arm: str | None = None,
) -> dict[str, Any]:
    """Pool independent endpoint evaluations using their sample counts.

    Energies are sample-count weighted. Independent standard errors propagate
    through the same weights, which reproduces the equal-size two-run pooling
    used for the online SR/MS-SR comparison.
    """

    rows = tuple(estimates)
    if len(rows) < 2:
        raise ValueError("At least two independent endpoint replicas are required")
    required_common = (
        "experiment_id",
        "system",
        "arm",
        "method",
        "lambda_policy",
        "k",
        "branch_seed",
        "checkpoint_sha256",
        "checkpoint_step",
        "samples_per_round",
        "rounds",
        "burn_in_rounds",
    )
    common_metadata: dict[str, Any] = {}
    for key in required_common:
        values = []
        for row in rows:
            if row.metadata is None or row.metadata.get(key) is None:
                raise ValueError(f"Endpoint replica {row.source or '<mapping>'} lacks {key}")
            values.append(row.metadata[key])
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Endpoint replicas disagree on {key}: {values}")
        common_metadata[key] = values[0]
    common_arm = str(common_metadata["arm"])
    if str(common_metadata["method"]) != common_arm:
        raise ValueError(
            f"Endpoint arm/method mismatch: {common_arm!r} != {common_metadata['method']!r}"
        )
    if arm is not None and str(arm) != common_arm:
        raise ValueError(f"Requested arm {arm!r} does not match endpoint arm {common_arm!r}")
    checkpoint_sha256 = str(common_metadata["checkpoint_sha256"])
    if len(checkpoint_sha256) != 64:
        raise ValueError("Endpoint checkpoint_sha256 must contain 64 hexadecimal characters")
    try:
        int(checkpoint_sha256, 16)
    except ValueError as error:
        raise ValueError("Endpoint checkpoint_sha256 is not hexadecimal") from error

    replica_keys = []
    evaluation_seeds = []
    evaluation_replicas = []
    for row in rows:
        assert row.metadata is not None
        seed = row.metadata.get("eval_seed", row.metadata.get("evaluation_seed"))
        replica = row.metadata.get("eval_run", row.metadata.get("evaluation_replica"))
        repeat = row.metadata.get("repeat_index", row.metadata.get("repeat"))
        if seed is None or replica is None or repeat is None:
            raise ValueError(f"Endpoint replica {row.source or '<mapping>'} lacks seed/replica/repeat metadata")
        protocol_total = int(row.metadata["rounds"]) * int(row.metadata["samples_per_round"])
        if int(row.total_samples) != protocol_total:
            raise ValueError(
                f"Endpoint replica {row.source or '<mapping>'} has {row.total_samples} samples; "
                f"protocol requires {protocol_total}"
            )
        replica_keys.append((int(seed), int(replica), int(repeat)))
        evaluation_seeds.append(int(seed))
        evaluation_replicas.append(int(replica))
    if len(set(replica_keys)) != len(rows):
        raise ValueError(f"Duplicate endpoint replicas: {replica_keys}")
    if len(set(evaluation_seeds)) != len(rows):
        raise ValueError(f"Endpoint replicas must use distinct evaluation seeds: {evaluation_seeds}")
    if len(set(evaluation_replicas)) != len(rows):
        raise ValueError(f"Endpoint replica identifiers must be distinct: {evaluation_replicas}")

    sample_counts = np.asarray([row.total_samples for row in rows], dtype=np.float64)
    weights = sample_counts / float(sample_counts.sum())
    energies = np.asarray([row.energy_per_site for row in rows], dtype=np.float64)
    sigmas = np.asarray([row.sigma_per_site for row in rows], dtype=np.float64)
    difference_z = replica_difference_z(energies, sigmas)
    if not math.isfinite(difference_z) or difference_z > MAX_REPLICA_DIFFERENCE_Z:
        raise ValueError(
            "Endpoint replicas disagree beyond the predeclared consistency gate: "
            f"z={difference_z:.6g} > {MAX_REPLICA_DIFFERENCE_Z:g}"
        )
    pooled_energy = float(np.dot(weights, energies))
    pooled_sigma = float(np.sqrt(np.dot(np.square(weights), np.square(sigmas))))
    return {
        "schema": ENDPOINT_SCHEMA,
        "arm": common_arm,
        "aggregation": "sample_count_weighted_independent",
        "energy_per_site": pooled_energy,
        "sigma_per_site": pooled_sigma,
        "sigma_per_site_binned": pooled_sigma,
        "total_samples": int(sample_counts.sum()),
        "input_count": len(rows),
        "replica_difference_z": difference_z,
        "max_allowed_replica_difference_z": MAX_REPLICA_DIFFERENCE_Z,
        "replica_consistency_status": "pass",
        "inputs": [row.as_dict() for row in rows],
        "evaluation_replicas": evaluation_replicas,
        "evaluation_seeds": evaluation_seeds,
        **common_metadata,
    }


def aggregate_endpoint_files(
    paths: Iterable[str | Path],
    *,
    arm: str | None = None,
) -> dict[str, Any]:
    return aggregate_endpoint_estimates(
        (load_endpoint_estimate(path) for path in paths),
        arm=arm,
    )


def atomic_write_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def infer_n_sites(training: Mapping[str, Any], system_name: str) -> int:
    config = training.get("config", {})
    if config.get("n_sites") is not None:
        return int(config["n_sites"])
    if system_name == "tfim" and config.get("length") is not None:
        return int(config["length"])
    if config.get("lattice_size") is not None:
        return int(config["lattice_size"]) ** 2
    return 64 if system_name in {"j1j2", "ssm", "shastry"} else 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment_metadata() -> dict[str, Any]:
    def package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    import jax

    device = jax.devices()[0]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": package_version("jax"),
        "jaxlib": package_version("jaxlib"),
        "netket": package_version("netket"),
        "numpy": np.__version__,
        "jax_backend": jax.default_backend(),
        "device_count": len(jax.devices()),
        "device": str(device),
        "device_kind": device.device_kind,
        "device_platform": device.platform,
    }


def evaluate_checkpoint_endpoint(
    checkpoint: str | Path,
    *,
    experiment_id: str,
    system_name: str,
    rounds: int = 512,
    burn_in: int = 5,
    seed: int = 12345,
    samples_per_round: int = 8192,
    method: str,
    lambda_policy: str,
    k: int,
    branch_seed: int,
    replica: int,
    repeat: int,
) -> dict[str, Any]:
    """Evaluate a checkpoint with one seeded stream and sequential Monte Carlo rounds."""

    if int(rounds) < 2:
        raise ValueError(f"rounds must be at least 2, got {rounds}")
    if int(burn_in) < 0:
        raise ValueError(f"burn_in must be nonnegative, got {burn_in}")
    if int(samples_per_round) <= 0:
        raise ValueError(f"samples_per_round must be positive, got {samples_per_round}")
    if int(k) <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not experiment_id.strip() or not method.strip() or not lambda_policy.strip():
        raise ValueError("experiment_id, method, and lambda_policy must be nonempty")

    # Delayed import keeps endpoint aggregation lightweight and CPU-friendly.
    from sr_filtering_nqs.large_scale.core.fixed_data_protocol import (
        _load_training_driver,
        reset_vstate_sampler_to_seed,
    )
    from sr_filtering_nqs.large_scale.core.multilambda_sr import sample_prepared_batch

    checkpoint = Path(checkpoint)
    training = _load_training_driver(checkpoint, system_name=system_name)
    driver = training["driver"]
    n_sites = infer_n_sites(training, system_name)

    means: list[float] = []
    counts: list[int] = []
    all_counts: list[int] = []
    all_round_energies: list[float] = []
    # One evaluation replica is one sequential Markov stream.  Resetting at
    # every round would make the discarded rounds unable to thermalize the
    # retained chain, and changing chain_name would create separate cached
    # sampler streams.  Seed exactly once, discard the first burn_in draws,
    # then retain the following rounds from the same advancing state.
    reset_vstate_sampler_to_seed(driver.state, int(seed))
    chain_name = f"endpoint_eval_replica{int(replica)}"
    for round_index in range(int(rounds) + int(burn_in)):
        batch = sample_prepared_batch(
            driver,
            n_samples=int(samples_per_round),
            chain_name=chain_name,
        )
        local_energy = np.asarray(batch.local_loss, dtype=np.complex128)
        weights = np.asarray(batch.weights, dtype=np.float64)
        if int(local_energy.size) != int(samples_per_round):
            raise ValueError(
                f"Round {round_index} produced {local_energy.size} samples; "
                f"expected exactly {samples_per_round}"
            )
        all_counts.append(int(local_energy.size))
        weight_sum = float(np.sum(weights))
        if not math.isfinite(weight_sum) or weight_sum <= 0.0:
            raise ValueError(f"Invalid Monte Carlo weight sum at round {round_index}: {weight_sum}")
        round_mean = float(np.real(np.sum(local_energy * weights) / weight_sum))
        all_round_energies.append(round_mean)
        if round_index >= int(burn_in):
            means.append(round_mean)
            counts.append(int(local_energy.size))

    values = np.asarray(means, dtype=np.float64)
    mean_total = float(np.mean(values))
    sigma_round = float(np.std(values, ddof=1) / math.sqrt(values.size))
    block_count = values.size // 8
    if block_count > 2:
        blocks = values[: block_count * 8].reshape(block_count, 8).mean(axis=1)
        sigma_block = float(np.std(blocks, ddof=1) / math.sqrt(block_count))
    else:
        sigma_block = sigma_round
    sigma_total = max(sigma_round, sigma_block)
    return {
        "schema": ENDPOINT_SCHEMA,
        "experiment_id": experiment_id,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_step": int(training.get("step", -1)),
        "system": system_name,
        "arm": method,
        "method": method,
        "lambda_policy": lambda_policy,
        "k": int(k),
        "branch_seed": int(branch_seed),
        "evaluation_replica": int(replica),
        "repeat_index": int(repeat),
        "eval_run": int(replica),
        "evaluation_id": f"replica{int(replica)}_repeat{int(repeat)}",
        "n_sites": int(n_sites),
        "rounds": len(means),
        "burn_in": int(burn_in),
        "burn_in_rounds": int(burn_in),
        "total_samples": int(sum(counts)),
        "energy_total": mean_total,
        "energy_per_site": mean_total / n_sites,
        "sigma_per_site": sigma_total / n_sites,
        "sigma_per_site_binned": sigma_total / n_sites,
        "sigma_round_level": sigma_round / n_sites,
        "sigma_round_level_per_site": sigma_round / n_sites,
        "sigma_block8_level": sigma_block / n_sites,
        "sigma_block8_level_per_site": sigma_block / n_sites,
        "seed": int(seed),
        "evaluation_seed": int(seed),
        "eval_seed": int(seed),
        "sampler_seed": int(seed),
        "sampler_reset_count": 1,
        "sampler_progression": "single_seed_sequential_rounds",
        "sampler_chain_name": chain_name,
        "samples_per_round": int(samples_per_round),
        "all_round_sequence_indices": list(range(int(rounds) + int(burn_in))),
        "burn_in_round_sequence_indices": list(range(int(burn_in))),
        "retained_round_sequence_indices": list(
            range(int(burn_in), int(rounds) + int(burn_in))
        ),
        "round_energy_total": [float(value) for value in means],
        "burn_in_round_energy_total": all_round_energies[: int(burn_in)],
        "all_round_energy_total": all_round_energies,
        "round_sample_counts": counts,
        "burn_in_round_sample_counts": all_counts[: int(burn_in)],
        "all_round_sample_counts": all_counts,
        "run_status": "complete",
        "audit_status": "pass",
        "audit_checks": [
            "finite_positive_weight_sums",
            "exact_samples_per_all_rounds",
            "single_sampler_reset_then_sequential_rounds",
            "conservative_max_round_and_block8_standard_error",
        ],
        "environment": _environment_metadata(),
    }
