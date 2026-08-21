from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import sr_filtering_nqs.large_scale.core.fixed_data_protocol as fixed_data
import sr_filtering_nqs.large_scale.core.multilambda_sr as multilambda
from sr_filtering_nqs.large_scale.core import endpoint_eval
from sr_filtering_nqs.large_scale.core.endpoint_eval import (
    EndpointEstimate,
    aggregate_endpoint_estimates,
)


def test_endpoint_pooling_is_count_weighted_and_preserves_inputs():
    common = {
        "experiment_id": "online.test",
        "system": "j1j2",
        "arm": "sr",
        "method": "sr",
        "lambda_policy": "fixed_1e-4",
        "k": 1,
        "branch_seed": 0,
        "checkpoint_sha256": "a" * 64,
        "checkpoint_step": 500,
        "samples_per_round": 1,
        "burn_in_rounds": 5,
        "repeat_index": 1,
    }
    first = EndpointEstimate.from_mapping(
        {
            "energy_per_site": -2.0,
            "sigma_per_site": 0.04,
            "total_samples": 3,
            **common,
            "rounds": 3,
            "eval_seed": 1,
            "eval_run": 1,
        },
        source="first.json",
    )
    second = EndpointEstimate.from_mapping(
        {
            "energy_per_site": -1.96,
            "sigma_per_site": 0.08,
            "total_samples": 1,
            **common,
            "rounds": 1,
            "eval_seed": 2,
            "eval_run": 2,
        },
        source="second.json",
    )

    with pytest.raises(ValueError, match="disagree on rounds"):
        aggregate_endpoint_estimates((first, second), arm="sr")

    second.metadata["rounds"] = 3
    second = EndpointEstimate.from_mapping(
        {**second.metadata, "total_samples": 3}, source="second.json"
    )
    pooled = aggregate_endpoint_estimates((first, second), arm="sr")

    assert pooled["energy_per_site"] == -1.98
    np.testing.assert_allclose(
        pooled["sigma_per_site"],
        np.sqrt((0.5 * 0.04) ** 2 + (0.5 * 0.08) ** 2),
    )
    assert pooled["total_samples"] == 6
    assert pooled["checkpoint_sha256"] == "a" * 64
    assert pooled["inputs"][0]["metadata"]["eval_seed"] == 1
    assert pooled["inputs"][1]["source"] == "second.json"
    assert pooled["replica_consistency_status"] == "pass"
    assert pooled["replica_difference_z"] < 1.0


def test_endpoint_pooling_rejects_duplicate_or_mismatched_replicas():
    base = {
        "energy_per_site": -1.99,
        "sigma_per_site": 1e-5,
        "total_samples": 8,
        "experiment_id": "online.test",
        "system": "j1j2",
        "arm": "sr",
        "method": "sr",
        "lambda_policy": "fixed_1e-4",
        "k": 1,
        "branch_seed": 0,
        "checkpoint_sha256": "a" * 64,
        "checkpoint_step": 500,
        "samples_per_round": 4,
        "rounds": 2,
        "burn_in_rounds": 5,
        "repeat_index": 1,
        "eval_seed": 12345,
        "eval_run": 1,
    }
    first = EndpointEstimate.from_mapping(base, source="first.json")
    duplicate = EndpointEstimate.from_mapping(base, source="duplicate.json")
    with pytest.raises(ValueError, match="Duplicate endpoint replicas"):
        aggregate_endpoint_estimates((first, duplicate), arm="sr")

    mismatch = EndpointEstimate.from_mapping(
        {**base, "arm": "spring", "method": "spring", "eval_seed": 98765, "eval_run": 2},
        source="mismatch.json",
    )
    with pytest.raises(ValueError, match="disagree on arm"):
        aggregate_endpoint_estimates((first, mismatch), arm="sr")

    inconsistent = EndpointEstimate.from_mapping(
        {
            **base,
            "energy_per_site": -1.98,
            "eval_seed": 98765,
            "eval_run": 2,
        },
        source="inconsistent.json",
    )
    with pytest.raises(ValueError, match="consistency gate"):
        aggregate_endpoint_estimates((first, inconsistent), arm="sr")


def test_endpoint_runner_enforces_samples_and_records_every_round(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.mpack"
    checkpoint.write_bytes(b"checkpoint")
    driver = SimpleNamespace(state=object())
    monkeypatch.setattr(
        fixed_data,
        "_load_training_driver",
        lambda path, system_name: {
            "driver": driver,
            "config": {"n_sites": 2},
            "step": 17,
        },
    )
    reset_seeds = []
    monkeypatch.setattr(
        fixed_data,
        "reset_vstate_sampler_to_seed",
        lambda state, seed: reset_seeds.append(seed),
    )
    emitted = iter((2.0, 4.0, 6.0))
    chain_names = []

    def sample_batch(driver, *, n_samples, chain_name):
        chain_names.append(chain_name)
        value = next(emitted)
        return SimpleNamespace(
            local_loss=np.full(n_samples, value, dtype=np.complex128),
            weights=np.ones(n_samples, dtype=np.float64),
        )

    monkeypatch.setattr(multilambda, "sample_prepared_batch", sample_batch)
    monkeypatch.setattr(endpoint_eval, "_environment_metadata", lambda: {"python": "test"})

    payload = endpoint_eval.evaluate_checkpoint_endpoint(
        checkpoint,
        experiment_id="online.test",
        system_name="j1j2",
        rounds=2,
        burn_in=1,
        seed=11,
        samples_per_round=4,
        method="sr",
        lambda_policy="fixed_1e-4",
        k=1,
        branch_seed=0,
        replica=2,
        repeat=3,
    )

    assert reset_seeds == [11]
    assert chain_names == ["endpoint_eval_replica2"] * 3
    assert payload["experiment_id"] == "online.test"
    assert payload["arm"] == "sr"
    assert payload["eval_run"] == 2
    assert payload["evaluation_id"] == "replica2_repeat3"
    assert payload["eval_seed"] == 11
    assert payload["burn_in_rounds"] == 1
    assert payload["burn_in_round_energy_total"] == [2.0]
    assert payload["round_energy_total"] == [4.0, 6.0]
    assert payload["all_round_energy_total"] == [2.0, 4.0, 6.0]
    assert payload["sampler_seed"] == 11
    assert payload["sampler_reset_count"] == 1
    assert payload["sampler_progression"] == "single_seed_sequential_rounds"
    assert payload["all_round_sequence_indices"] == [0, 1, 2]
    assert payload["burn_in_round_sequence_indices"] == [0]
    assert payload["retained_round_sequence_indices"] == [1, 2]
    assert payload["all_round_sample_counts"] == [4, 4, 4]
    assert payload["total_samples"] == 8
    assert payload["samples_per_round"] == 4
    assert payload["audit_status"] == "pass"
