from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _curation_module():
    script = Path(__file__).resolve().parents[1] / "scripts/curation/curate_release_artifacts.py"
    spec = importlib.util.spec_from_file_location("sr_filter_release_curation", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _complete_payload() -> dict:
    all_energies = [-127.0 + index * 1e-6 for index in range(517)]
    return {
        "schema": "sr_filter.endpoint_eval.v1",
        "experiment_id": "online.test",
        "checkpoint_sha256": "a" * 64,
        "checkpoint_step": 500,
        "system": "j1j2",
        "arm": "sr",
        "method": "sr",
        "lambda_policy": "fixed_1e-4",
        "k": 1,
        "branch_seed": 0,
        "evaluation_replica": 1,
        "repeat_index": 1,
        "eval_run": 1,
        "rounds": 512,
        "burn_in_rounds": 5,
        "samples_per_round": 8192,
        "total_samples": 4_194_304,
        "energy_per_site": -1.99,
        "sigma_per_site": 1e-5,
        "sigma_round_level_per_site": 9e-6,
        "sigma_block8_level_per_site": 1e-5,
        "seed": 12345,
        "evaluation_seed": 12345,
        "eval_seed": 12345,
        "sampler_seed": 12345,
        "sampler_reset_count": 1,
        "sampler_progression": "single_seed_sequential_rounds",
        "sampler_chain_name": "endpoint_eval_replica1",
        "all_round_sequence_indices": list(range(517)),
        "burn_in_round_sequence_indices": list(range(5)),
        "retained_round_sequence_indices": list(range(5, 517)),
        "all_round_energy_total": all_energies,
        "burn_in_round_energy_total": all_energies[:5],
        "round_energy_total": all_energies[5:],
        "all_round_sample_counts": [8192] * 517,
        "burn_in_round_sample_counts": [8192] * 5,
        "round_sample_counts": [8192] * 512,
        "run_status": "complete",
        "audit_status": "pass",
        "environment": {
            "python": "3.12",
            "jax": "0.7.2",
            "jaxlib": "0.7.2",
            "netket": "3.21.0",
            "jax_backend": "gpu",
            "device_count": 1,
            "device_kind": "NVIDIA H100 80GB HBM3",
        },
    }


def test_endpoint_overlay_requires_one_sequential_sampler_stream(tmp_path):
    module = _curation_module()
    payload = _complete_payload()
    path = tmp_path / "sr_replica1.json"
    path.write_text(json.dumps(payload))

    overlays = module.validated_endpoint_overlays(
        tmp_path,
        branch_seed=0,
        experiment_id="online.test",
        checkpoint_hashes={"sr": "a" * 64},
        checkpoint_steps={"sr": 500},
    )
    assert overlays[("sr", 12345)]["round_energy_total"] == payload["round_energy_total"]

    payload["sampler_reset_count"] = 517
    payload["sampler_progression"] = "reseeded_each_round"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="sampler progression"):
        module.validated_endpoint_overlays(
            tmp_path,
            branch_seed=0,
            experiment_id="online.test",
            checkpoint_hashes={"sr": "a" * 64},
            checkpoint_steps={"sr": 500},
        )


def test_endpoint_overlay_rejects_missing_round_values(tmp_path):
    module = _curation_module()
    payload = _complete_payload()
    payload["round_energy_total"] = payload["round_energy_total"][:-1]
    (tmp_path / "sr_replica1.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="per-round energies"):
        module.validated_endpoint_overlays(
            tmp_path,
            branch_seed=0,
            experiment_id="online.test",
            checkpoint_hashes={"sr": "a" * 64},
            checkpoint_steps={"sr": 500},
        )


def test_endpoint_overlay_ignores_arm_until_accepted_checkpoint_exists(tmp_path):
    module = _curation_module()
    payload = _complete_payload()
    payload.update(
        {
            "arm": "ms_sr",
            "method": "ms_sr",
            "lambda_policy": "fixed_grid_1e-2_1e-3_1e-4_1e-5_stacking",
            "k": 4,
            "checkpoint_step": 125,
        }
    )
    (tmp_path / "legacy_ms_sr_replica1.json").write_text(json.dumps(payload))

    overlays = module.validated_endpoint_overlays(
        tmp_path,
        branch_seed=0,
        experiment_id="online.test",
        checkpoint_hashes={"sr": "a" * 64},
        checkpoint_steps={"sr": 500, "ms_sr": 125},
    )

    assert overlays == {}


def test_endpoint_replica_consistency_gate_is_predeclared():
    module = _curation_module()
    first = _complete_payload()
    second = {
        **first,
        "energy_per_site": first["energy_per_site"] + 1e-5,
        "evaluation_replica": 2,
        "eval_run": 2,
        "evaluation_seed": 98765,
        "eval_seed": 98765,
        "seed": 98765,
        "sampler_seed": 98765,
    }
    overlays = {("sr", 12345): first, ("sr", 98765): second}
    audit = module.endpoint_replica_consistency(overlays)
    assert audit["sr"]["status"] == "pass"
    assert audit["sr"]["replica_difference_z"] < 1.0

    second["energy_per_site"] = first["energy_per_site"] + 1e-3
    with pytest.raises(ValueError, match="consistency gate"):
        module.endpoint_replica_consistency(overlays)
