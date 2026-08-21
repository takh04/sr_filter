from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from sr_filtering_nqs.artifact_store import manifest_files

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/curation/stage_hf_dataset.py"
SPEC = importlib.util.spec_from_file_location("sr_filter_hf_stage", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
hf_stage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hf_stage)

CODE_SHA = "1" * 40
DATA_SHA = "2" * 40


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _endpoint(branch: int, arm: str, replica: int) -> dict:
    rounds = 512
    burn = 5
    samples = 8192
    return {
        "schema": "sr_filter.endpoint_eval.v1",
        "experiment_id": f"online.branch{branch}",
        "run_status": "complete",
        "audit_status": "pass",
        "branch_seed": branch,
        "arm": arm,
        "method": arm,
        "lambda_policy": "fixed" if arm != "ms_sr" else "fixed_grid",
        "k": 4 if arm == "ms_sr" else 1,
        "evaluation_replica": replica,
        "repeat_index": 1,
        "eval_run": replica,
        "eval_seed": 12345 if replica == 1 else 98765,
        "rounds": rounds,
        "burn_in_rounds": burn,
        "samples_per_round": samples,
        "total_samples": rounds * samples,
        "sampler_reset_count": 1,
        "sampler_progression": "single_seed_sequential_rounds",
        "round_energy_total": [-100.0] * rounds,
        "burn_in_round_energy_total": [-99.0] * burn,
        "all_round_energy_total": [-99.0] * burn + [-100.0] * rounds,
        "checkpoint_sha256": f"{branch + replica:x}" * 64,
        "energy_per_site": -1.0,
        "sigma_per_site": 0.001,
        "environment": {"python": "3.12", "jax": "0.7.2", "netket": "3.21.0"},
    }


def _make_inputs(tmp_path: Path, *, omit_endpoint: bool = False) -> dict[str, Path]:
    repo = tmp_path / "repo"
    checkpoint_root = tmp_path / "checkpoint_staging"
    endpoint_root = tmp_path / "endpoints"
    (repo / "artifacts/wall_clock/raw").mkdir(parents=True)
    (repo / "artifacts/online_training").mkdir(parents=True)
    checkpoint_rows = []
    for index in range(8):
        relative = Path("checkpoints") / f"checkpoint_{index}.pkl"
        path = checkpoint_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"sanitized checkpoint {index}\n".encode())
        checkpoint_rows.append(
            {
                "id": f"checkpoint/{index}",
                "path": relative.as_posix(),
                "status": "sanitized",
                "sanitized_sha256": hf_stage._sha256(path),
            }
        )
    _write_json(
        repo / "artifacts/external_checkpoints.json",
        {
            "schema": "sr_filter.external_checkpoint_manifest.v1",
            "status": "complete",
            "remote": {"repo_id": None, "repo_type": "dataset", "revision": None},
            "checkpoints": checkpoint_rows,
        },
    )

    for branch in (0, 1):
        for arm in ("sr", "spring", "ms_sr"):
            for replica in (1, 2):
                if omit_endpoint and branch == 1 and arm == "ms_sr" and replica == 2:
                    continue
                _write_json(
                    endpoint_root / f"branch_seed{branch}/{arm}_replica{replica}.json",
                    _endpoint(branch, arm, replica),
                )

    for pair in range(1, 4):
        for method in ("sr", "bagged_sr_k4", "ms_sr_k4"):
            _write_json(
                repo / f"artifacts/wall_clock/raw/pair{pair}_{method}.json",
                {"pair": pair, "method": method, "rows": [{"repeat": 1, "total_s": 1.0}]},
            )
    _write_json(repo / "artifacts/wall_clock/provenance.json", {"hardware": "H100"})
    (repo / "artifacts/wall_clock/sample_fingerprints.csv").write_text(
        "pair,method,fingerprint\n1,sr,abc\n", encoding="utf-8"
    )
    (repo / "artifacts/wall_clock/pairing_verification.csv").write_text(
        "pair,verified\n1,true\n", encoding="utf-8"
    )

    for branch in (0, 1):
        branch_root = repo / f"artifacts/online_training/branch_seed{branch}"
        _write_json(
            branch_root / "status.json",
            {"status": "complete", "camera_ready_gate": True},
        )
        _write_json(branch_root / "provenance.json", {"branch_seed": branch})
        branch_root.joinpath("training_curve.csv").write_text(
            "arm,budget,energy\nsr,1,-1.0\n", encoding="utf-8"
        )
    _write_json(repo / "artifacts/online_training/provenance.json", {"schema_version": 1})
    _write_json(repo / "artifacts/online_training/exclusions.json", {"excluded": []})
    (repo / "artifacts/online_training/checksums.sha256").write_text(
        "portable provenance placeholder\n", encoding="utf-8"
    )

    config = repo / "configs/release/hf_dataset.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        """schema = "sr_filter.hf_dataset_staging_config.v1"
repo_id = "takh04/sr-filter-artifacts"
repo_type = "dataset"
data_revision_placeholder = "PENDING_HF_COMMIT_SHA"

[inputs]
external_checkpoint_manifest = "artifacts/external_checkpoints.json"
online_artifact_root = "artifacts/online_training"
wall_clock_artifact_root = "artifacts/wall_clock"

[endpoint_contract]
schema = "sr_filter.endpoint_eval.v1"
branches = [0, 1]
arms = ["sr", "spring", "ms_sr"]
replicas = [1, 2]
rounds = 512
burn_in_rounds = 5
samples_per_round = 8192
sampler_reset_count = 1
sampler_progression = "single_seed_sequential_rounds"
max_replica_difference_z = 5.0
""",
        encoding="utf-8",
    )
    return {
        "repo": repo,
        "config": config,
        "checkpoints": checkpoint_root,
        "endpoints": endpoint_root,
    }


def _stage(monkeypatch, inputs: dict[str, Path], staging: Path):
    monkeypatch.setattr(hf_stage, "_git_state", lambda _root: (CODE_SHA, False))
    return hf_stage.stage_dataset(
        repo_root=inputs["repo"],
        config_path=inputs["config"],
        staging_root=staging,
        checkpoint_root=inputs["checkpoints"],
        endpoint_root=inputs["endpoints"],
        replace=False,
    )


def test_stage_complete_bundle_and_detect_tampering(tmp_path, monkeypatch):
    inputs = _make_inputs(tmp_path)
    staging = tmp_path / "staged"

    manifest = _stage(monkeypatch, inputs, staging)
    report = hf_stage.validate_staging(staging, require_complete=True)

    assert manifest["status"] == "ready_for_private_upload"
    assert report["counts"] == {
        "checkpoint": 8,
        "endpoint_rounds": 12,
        "online_provenance": 9,
        "wall_clock": 12,
    }
    assert manifest["code_revision"] == CODE_SHA
    assert manifest["data_revision"] == "PENDING_HF_COMMIT_SHA"
    assert "Never load an\nuntrusted pickle" in (staging / "README.md").read_text()
    assert b"/home/" not in (staging / "manifest.json").read_bytes()

    first = staging / manifest["files"][0]["path"]
    first.write_bytes(first.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="mismatch"):
        hf_stage.validate_staging(staging, require_complete=True)


def test_pending_stage_is_truthful_and_atomic(tmp_path, monkeypatch):
    inputs = _make_inputs(tmp_path, omit_endpoint=True)
    staging = tmp_path / "staged"
    staging.mkdir()
    (staging / "old.txt").write_text("old")
    monkeypatch.setattr(hf_stage, "_git_state", lambda _root: (CODE_SHA, False))

    with pytest.raises(ValueError, match="--replace"):
        hf_stage.stage_dataset(
            repo_root=inputs["repo"],
            config_path=inputs["config"],
            staging_root=staging,
            checkpoint_root=inputs["checkpoints"],
            endpoint_root=inputs["endpoints"],
            replace=False,
        )
    assert (staging / "old.txt").read_text() == "old"

    manifest = hf_stage.stage_dataset(
        repo_root=inputs["repo"],
        config_path=inputs["config"],
        staging_root=staging,
        checkpoint_root=inputs["checkpoints"],
        endpoint_root=inputs["endpoints"],
        replace=True,
    )
    assert manifest["status"] == "pending_inputs"
    assert {row["id"] for row in manifest["missing"]} == {"endpoint:branch1/ms_sr/replica2"}
    assert not (staging / "old.txt").exists()
    hf_stage.validate_staging(staging, require_complete=False)


def test_stage_rejects_inconsistent_endpoint_replicas(tmp_path, monkeypatch):
    inputs = _make_inputs(tmp_path)
    path = inputs["endpoints"] / "branch_seed0/sr_replica2.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["energy_per_site"] = -0.9
    _write_json(path, payload)

    with pytest.raises(ValueError, match="consistency gate"):
        _stage(monkeypatch, inputs, tmp_path / "staged")


def test_finalize_requires_exact_readback_and_pins_fetch_manifest(tmp_path, monkeypatch):
    inputs = _make_inputs(tmp_path)
    repo = inputs["repo"]
    staging = tmp_path / "staged"
    _stage(monkeypatch, inputs, staging)
    readback = tmp_path / "readback"
    shutil.copytree(staging, readback)

    _write_json(
        repo / "artifacts/hf_dataset_release.json",
        {"schema": "sr_filter.hf_dataset_release.v1", "status": "pending_local_bundle"},
    )
    _write_json(
        repo / "artifacts/manifest.json",
        {
            "schema": "sr_filter.artifact_manifest.v1",
            "files": [],
            "configs": [],
            "experiments": [],
        },
    )
    report = hf_stage.finalize_pinned_release(
        repo_root=repo,
        staging_root=staging,
        readback_root=readback,
        data_revision=DATA_SHA,
    )

    assert report["status"] == "verified_pinned"
    release = json.loads((repo / "artifacts/hf_dataset_release.json").read_text())
    assert release["data_revision"] == DATA_SHA
    assert release["verification"]["readback_verified"] is True
    assert all(f"/resolve/{DATA_SHA}/" in row["url"] for row in release["files"])
    external = json.loads((repo / "artifacts/external_checkpoints.json").read_text())
    assert external["remote"]["revision"] == DATA_SHA
    assert all(f"/resolve/{DATA_SHA}/" in row["url"] for row in external["checkpoints"])

    main = json.loads((repo / "artifacts/manifest.json").read_text())
    records = manifest_files(main, experiment_ids=["companion.camera_ready_artifacts.v1"])
    assert len(records) == len(release["files"])
    assert all(record["url"].startswith("https://") for record in records)


def test_finalize_rejects_readback_mismatch_before_checked_manifest_writes(tmp_path, monkeypatch):
    inputs = _make_inputs(tmp_path)
    repo = inputs["repo"]
    staging = tmp_path / "staged"
    _stage(monkeypatch, inputs, staging)
    readback = tmp_path / "readback"
    shutil.copytree(staging, readback)
    payload = json.loads((staging / "manifest.json").read_text())
    damaged = readback / payload["files"][0]["path"]
    damaged.write_bytes(b"wrong")
    original_external = (repo / "artifacts/external_checkpoints.json").read_bytes()

    with pytest.raises(ValueError, match="Readback byte mismatch"):
        hf_stage.finalize_pinned_release(
            repo_root=repo,
            staging_root=staging,
            readback_root=readback,
            data_revision=DATA_SHA,
        )
    assert (repo / "artifacts/external_checkpoints.json").read_bytes() == original_external
    assert not (repo / "artifacts/hf_dataset_release.json").exists()
