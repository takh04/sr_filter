#!/usr/bin/env python3
"""Stage, validate, and pin the SR Filter companion Hugging Face dataset.

This command never contacts Hugging Face.  ``stage`` assembles a sanitized
local directory, ``validate`` checks such a directory, and ``finalize`` pins a
previously uploaded-and-downloaded byte-for-byte readback in the checked-in
manifests.  Remote creation, upload, download, and visibility changes remain
explicit operator actions documented in ``docs/artifacts.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
import urllib.parse
import uuid
from collections import Counter
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/release/hf_dataset.toml"
DEFAULT_STAGING_ROOT = REPO_ROOT / "outputs/hf_dataset"
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "outputs/external_staging"
DEFAULT_ENDPOINT_ROOT = REPO_ROOT / "outputs/gpu/endpoints"

ARTIFACT_MANIFEST_SCHEMA = "sr_filter.artifact_manifest.v1"
DATASET_RELEASE_SCHEMA = "sr_filter.hf_dataset_release.v1"
DATASET_CONFIG_SCHEMA = "sr_filter.hf_dataset_staging_config.v1"
ENDPOINT_SCHEMA = "sr_filter.endpoint_eval.v1"
CODE_PLACEHOLDER = "PENDING_GIT_COMMIT_SHA"
DATA_PLACEHOLDER = "PENDING_HF_COMMIT_SHA"
REVISION_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
SHA256_RE = re.compile(r"[0-9a-f]{64}")

PRIVATE_BYTE_PATTERNS = (
    re.compile(rb"(?<![A-Za-z0-9_])/(?:home|Users)/[A-Za-z0-9._-]+/"),
    re.compile(rb"(?<![A-Za-z0-9_])/(?:scratch|gpfs|lustre|nfs)/[^\x00\r\n\t ]+"),
    re.compile(rb"(?i)(?<![A-Za-z0-9_])[A-Z]:[\\/]+Users[\\/]+[^\\/\x00\r\n\t ]+[\\/]"),
    re.compile(rb"beingin_" rb"overfitting_nqs", re.IGNORECASE),
    re.compile(rb"benign_" rb"overfitting_nqs", re.IGNORECASE),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str) -> Path:
    _require(isinstance(value, str) and bool(value), "Artifact paths must be nonempty strings")
    _require(
        "\\" not in value and "\x00" not in value and ":" not in value, f"Unsafe path: {value!r}"
    )
    pure = PurePosixPath(value)
    _require(
        not pure.is_absolute() and pure != PurePosixPath(".") and ".." not in pure.parts,
        f"Unsafe relative path: {value!r}",
    )
    return Path(*pure.parts)


def _confined_file(root: Path, relative: Path) -> Path:
    root = root.resolve(strict=True)
    _require(root.is_dir(), f"Input root is not a directory: {root}")
    unresolved = root / relative
    current = root
    for part in relative.parts:
        current /= part
        _require(not current.is_symlink(), f"Input path traverses a symbolic link: {current}")
    candidate = unresolved.resolve(strict=True)
    _require(candidate.is_relative_to(root), f"Input escapes its root: {candidate}")
    _require(
        candidate.is_file() and not candidate.is_symlink(),
        f"Input is not a regular file: {candidate}",
    )
    return candidate


def _assert_public_bytes(path: Path) -> None:
    payload = path.read_bytes()
    for pattern in PRIVATE_BYTE_PATTERNS:
        _require(
            pattern.search(payload) is None, f"Private path or repository name found in {path}"
        )


def _media_type(path: Path) -> str:
    explicit = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".pkl": "application/x-python-pickle",
        ".sha256": "text/plain",
    }
    return explicit.get(
        path.suffix.lower(), mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    )


def _git_state(repo_root: Path) -> tuple[str, bool]:
    try:
        revision = (
            subprocess.run(
                ["git", "-C", os.fspath(repo_root), "rev-parse", "--verify", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .lower()
        )
        dirty = bool(
            subprocess.run(
                ["git", "-C", os.fspath(repo_root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return CODE_PLACEHOLDER, True
    if REVISION_RE.fullmatch(revision) is None:
        return CODE_PLACEHOLDER, True
    return revision, dirty


def _copy_record(
    source: Path,
    destination_root: Path,
    destination: str,
    *,
    category: str,
    source_id: str,
) -> dict[str, Any]:
    relative = _safe_relative(destination)
    _assert_public_bytes(source)
    target = destination_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    _require(not target.exists(), f"Duplicate dataset target: {destination}")
    shutil.copyfile(source, target)
    return {
        "path": relative.as_posix(),
        "bytes": target.stat().st_size,
        "sha256": _sha256(target),
        "media_type": _media_type(target),
        "category": category,
        "source_id": source_id,
        "url": None,
    }


def _validate_endpoint(
    path: Path, *, branch: int, arm: str, replica: int, contract: Mapping[str, Any]
) -> dict[str, Any]:
    payload = _load_json(path)
    expected_rounds = int(contract["rounds"])
    expected_burn = int(contract["burn_in_rounds"])
    expected_samples = int(contract["samples_per_round"])
    _require(
        payload.get("schema") == contract.get("schema", ENDPOINT_SCHEMA),
        f"Wrong endpoint schema: {path}",
    )
    _require(
        payload.get("run_status") == "complete" and payload.get("audit_status") == "pass",
        f"Endpoint is not audited complete: {path}",
    )
    _require(int(payload.get("branch_seed", -1)) == branch, f"Wrong endpoint branch: {path}")
    _require(
        payload.get("arm") == arm and payload.get("method") == arm, f"Wrong endpoint arm: {path}"
    )
    _require(
        int(payload.get("evaluation_replica", -1)) == replica, f"Wrong endpoint replica: {path}"
    )
    _require(int(payload.get("rounds", -1)) == expected_rounds, f"Wrong retained rounds: {path}")
    _require(
        int(payload.get("burn_in_rounds", -1)) == expected_burn, f"Wrong burn-in rounds: {path}"
    )
    _require(
        int(payload.get("samples_per_round", -1)) == expected_samples,
        f"Wrong samples per round: {path}",
    )
    _require(
        int(payload.get("total_samples", -1)) == expected_rounds * expected_samples,
        f"Wrong endpoint sample total: {path}",
    )
    _require(
        int(payload.get("sampler_reset_count", -1)) == int(contract["sampler_reset_count"]),
        f"Wrong sampler reset count: {path}",
    )
    _require(
        payload.get("sampler_progression") == contract["sampler_progression"],
        f"Wrong sampler progression: {path}",
    )
    _require(
        len(payload.get("round_energy_total", [])) == expected_rounds,
        f"Missing retained round values: {path}",
    )
    _require(
        len(payload.get("burn_in_round_energy_total", [])) == expected_burn,
        f"Missing burn-in round values: {path}",
    )
    _require(
        len(payload.get("all_round_energy_total", [])) == expected_rounds + expected_burn,
        f"Missing all-round values: {path}",
    )
    _require(
        payload.get("environment") and isinstance(payload["environment"], dict),
        f"Missing endpoint environment: {path}",
    )
    checkpoint_sha = str(payload.get("checkpoint_sha256", "")).lower()
    _require(
        SHA256_RE.fullmatch(checkpoint_sha) is not None, f"Invalid endpoint checkpoint hash: {path}"
    )
    energy = float(payload.get("energy_per_site", math.nan))
    sigma = float(payload.get("sigma_per_site", math.nan))
    _require(math.isfinite(energy), f"Non-finite endpoint energy: {path}")
    _require(math.isfinite(sigma) and sigma > 0.0, f"Invalid endpoint standard error: {path}")
    return payload


def _dataset_card(
    *,
    repo_id: str,
    status: str,
    code_revision: str,
    data_revision: str,
    counts: Mapping[str, int],
    missing: list[dict[str, str]],
) -> str:
    missing_text = (
        "None."
        if not missing
        else "\n".join(f"- `{item['id']}`: {item['reason']}" for item in missing)
    )
    return f"""---
license: apache-2.0
pretty_name: SR Filter camera-ready artifacts
tags:
- quantum-machine-learning
- neural-quantum-states
- reproducibility
---

# SR Filter camera-ready artifacts

This companion dataset contains the large, auditable inputs for the camera-ready
release of *Stochastic Reconfiguration as Statistical Filtering*. It belongs to
[`takh04/sr_filter`](https://github.com/takh04/sr_filter); it is not a general-purpose
training dataset.

## Release state

- Bundle status: `{status}`
- Source-code revision: `{code_revision}`
- Immutable data revision: `{data_revision}`
- Repository: `{repo_id}`

The data revision is deliberately a placeholder until a private upload has been
downloaded again and every byte has been verified. Do not make this repository
public while the bundle status is pending.

## Contents

- Sanitized source/final checkpoints: {counts.get("checkpoint", 0)} (target: 8)
- Per-round endpoint evaluations: {counts.get("endpoint_rounds", 0)} (target: 12)
- Raw H100 wall-clock records and provenance: {counts.get("wall_clock", 0)}
- Sanitized online trajectories and provenance: {counts.get("online_provenance", 0)}

Endpoint records retain all five discarded burn-in rounds and all 512 retained
rounds of 8,192 samples, deterministic seeds, the environment, checkpoint hash,
and uncertainty diagnostics. The manifest records SHA-256 and byte size for every
payload file.

## Missing or incomplete inputs

{missing_text}

## Safety and licensing

Python pickle checkpoints can execute code when deserialized. **Never load an
untrusted pickle.** Verify the exact SHA-256 from `manifest.json`, obtain files
only from the pinned revision, and load them only in a trusted environment.

Original SR Filter code and released original artifacts are Apache-2.0 where
rights permit. Third-party components and metadata remain subject to their own
licenses; see the source repository's `NOTICE` and `THIRD_PARTY_NOTICES.md`.
"""


def _write_staging_metadata(root: Path, manifest: dict[str, Any]) -> None:
    counts = Counter(str(record["category"]) for record in manifest["files"])
    release = {
        "schema": DATASET_RELEASE_SCHEMA,
        "schema_version": 1,
        "repo_id": manifest["repo_id"],
        "repo_type": "dataset",
        "status": manifest["status"],
        "code_revision": manifest["code_revision"],
        "data_revision": manifest["data_revision"],
        "manifest_path": "manifest.json",
        "raw_base_url": manifest.get("base_url"),
        "readback_verified": manifest.get("readback_verified", False),
    }
    _atomic_json(root / "release.json", release)
    (root / "README.md").write_text(
        _dataset_card(
            repo_id=str(manifest["repo_id"]),
            status=str(manifest["status"]),
            code_revision=str(manifest["code_revision"]),
            data_revision=str(manifest["data_revision"]),
            counts=counts,
            missing=list(manifest.get("missing", [])),
        ),
        encoding="utf-8",
        newline="\n",
    )
    manifest["metadata_files"] = [
        {
            "path": name,
            "bytes": (root / name).stat().st_size,
            "sha256": _sha256(root / name),
        }
        for name in ("README.md", "release.json")
    ]
    _atomic_json(root / "manifest.json", manifest)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    _require(config.get("schema") == DATASET_CONFIG_SCHEMA, f"Wrong staging config schema: {path}")
    return config


def _stage_into(
    destination: Path,
    *,
    repo_root: Path,
    config: Mapping[str, Any],
    checkpoint_root: Path,
    endpoint_root: Path,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []

    inputs = config["inputs"]
    checkpoint_manifest_path = repo_root / _safe_relative(
        str(inputs["external_checkpoint_manifest"])
    )
    checkpoint_manifest = _load_json(checkpoint_manifest_path)
    checkpoint_rows = checkpoint_manifest.get("checkpoints", [])
    _require(
        isinstance(checkpoint_rows, list) and len(checkpoint_rows) == 8,
        "Checkpoint inventory must contain exactly eight records",
    )
    for row in checkpoint_rows:
        checkpoint_id = str(row.get("id", "<unknown>"))
        if row.get("status") != "sanitized":
            missing.append(
                {
                    "id": f"checkpoint:{checkpoint_id}",
                    "reason": str(row.get("status", "not sanitized")),
                }
            )
            continue
        relative = _safe_relative(str(row["path"]))
        try:
            source = _confined_file(checkpoint_root, relative)
        except (FileNotFoundError, ValueError):
            missing.append(
                {
                    "id": f"checkpoint:{checkpoint_id}",
                    "reason": "sanitized file is absent from the explicit checkpoint staging root",
                }
            )
            continue
        expected_sha = str(row.get("sanitized_sha256", "")).lower()
        _require(
            SHA256_RE.fullmatch(expected_sha) is not None,
            f"Missing sanitized hash: {checkpoint_id}",
        )
        _require(
            _sha256(source) == expected_sha, f"Sanitized checkpoint hash mismatch: {checkpoint_id}"
        )
        files.append(
            _copy_record(
                source,
                destination,
                relative.as_posix(),
                category="checkpoint",
                source_id=checkpoint_id,
            )
        )

    contract = config["endpoint_contract"]
    for branch in contract["branches"]:
        for arm in contract["arms"]:
            arm_payloads: list[dict[str, Any]] = []
            for replica in contract["replicas"]:
                relative = Path(f"branch_seed{int(branch)}") / f"{arm}_replica{int(replica)}.json"
                endpoint_id = f"branch{int(branch)}/{arm}/replica{int(replica)}"
                try:
                    source = _confined_file(endpoint_root, relative)
                except (FileNotFoundError, ValueError):
                    missing.append(
                        {
                            "id": f"endpoint:{endpoint_id}",
                            "reason": "audited per-round endpoint JSON is not present",
                        }
                    )
                    continue
                payload = _validate_endpoint(
                    source,
                    branch=int(branch),
                    arm=str(arm),
                    replica=int(replica),
                    contract=contract,
                )
                arm_payloads.append(payload)
                files.append(
                    _copy_record(
                        source,
                        destination,
                        f"endpoint_rounds/{relative.as_posix()}",
                        category="endpoint_rounds",
                        source_id=endpoint_id,
                    )
                )
            if len(arm_payloads) == len(contract["replicas"]):
                _require(len(arm_payloads) == 2, "Endpoint consistency audit requires two replicas")
                difference = abs(
                    float(arm_payloads[0]["energy_per_site"])
                    - float(arm_payloads[1]["energy_per_site"])
                )
                denominator = math.hypot(
                    float(arm_payloads[0]["sigma_per_site"]),
                    float(arm_payloads[1]["sigma_per_site"]),
                )
                difference_z = difference / denominator
                threshold = float(contract["max_replica_difference_z"])
                _require(
                    math.isfinite(difference_z) and difference_z <= threshold,
                    f"Endpoint replicas fail consistency gate for branch{int(branch)}/{arm}: "
                    f"z={difference_z:.6g} > {threshold:g}",
                )

    wall_root = repo_root / _safe_relative(str(inputs["wall_clock_artifact_root"]))
    raw_wall = sorted((wall_root / "raw").glob("*.json")) if wall_root.is_dir() else []
    expected_wall_names = {
        f"pair{pair}_{method}.json"
        for pair in range(1, 4)
        for method in ("sr", "bagged_sr_k4", "ms_sr_k4")
    }
    actual_wall_names = {path.name for path in raw_wall}
    for name in sorted(expected_wall_names - actual_wall_names):
        missing.append({"id": f"wall_clock:raw/{name}", "reason": "missing raw timing record"})
    _require(
        not (actual_wall_names - expected_wall_names), "Unexpected raw wall-clock files present"
    )
    for source in raw_wall:
        source = _confined_file(wall_root, Path("raw") / source.name)
        files.append(
            _copy_record(
                source,
                destination,
                f"wall_clock_raw/raw/{source.name}",
                category="wall_clock",
                source_id=f"artifacts/wall_clock/raw/{source.name}",
            )
        )
    for name in ("provenance.json", "sample_fingerprints.csv", "pairing_verification.csv"):
        try:
            source = _confined_file(wall_root, Path(name))
        except (FileNotFoundError, ValueError):
            missing.append(
                {"id": f"wall_clock:{name}", "reason": "sanitized provenance file is missing"}
            )
            continue
        files.append(
            _copy_record(
                source,
                destination,
                f"wall_clock_raw/{name}",
                category="wall_clock",
                source_id=f"artifacts/wall_clock/{name}",
            )
        )

    online_root = repo_root / _safe_relative(str(inputs["online_artifact_root"]))
    online_paths = (
        sorted(
            path
            for path in online_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".csv", ".json", ".sha256"}
        )
        if online_root.is_dir()
        else []
    )
    for branch in ("branch_seed0", "branch_seed1"):
        status_path = online_root / branch / "status.json"
        if not status_path.is_file():
            missing.append(
                {"id": f"online:{branch}/status.json", "reason": "missing online status"}
            )
            continue
        status = _load_json(status_path)
        if status.get("status") != "complete" or status.get("camera_ready_gate") is not True:
            missing.append(
                {"id": f"online:{branch}", "reason": f"status={status.get('status', 'unknown')}"}
            )
    for source in online_paths:
        relative = source.relative_to(online_root)
        source = _confined_file(online_root, relative)
        files.append(
            _copy_record(
                source,
                destination,
                f"online_provenance/{relative.as_posix()}",
                category="online_provenance",
                source_id=f"artifacts/online_training/{relative.as_posix()}",
            )
        )
    if not online_paths:
        missing.append(
            {"id": "online:provenance", "reason": "no sanitized online provenance files"}
        )

    target_paths = [str(record["path"]) for record in files]
    _require(len(target_paths) == len(set(target_paths)), "Duplicate dataset paths")
    code_revision, dirty = _git_state(repo_root)
    if code_revision == CODE_PLACEHOLDER:
        missing.append(
            {"id": "revision:code", "reason": "repository has no immutable Git revision"}
        )
    elif dirty:
        missing.append({"id": "revision:code", "reason": "source worktree is dirty"})
    status = "ready_for_private_upload" if not missing else "pending_inputs"
    manifest = {
        "schema": ARTIFACT_MANIFEST_SCHEMA,
        "schema_version": 1,
        "bundle_schema": "sr_filter.hf_dataset_bundle.v1",
        "repo_id": str(config["repo_id"]),
        "repo_type": "dataset",
        "status": status,
        "code_revision": code_revision,
        "code_worktree_was_dirty": dirty,
        "data_revision": str(config.get("data_revision_placeholder", DATA_PLACEHOLDER)),
        "base_url": None,
        "readback_verified": False,
        "missing": missing,
        "files": sorted(files, key=lambda row: str(row["path"])),
        "configs": [],
        "experiments": [
            {
                "id": "companion.camera_ready_artifacts.v1",
                "category": "companion_dataset",
                "status": status,
                "files": sorted(files, key=lambda row: str(row["path"])),
            }
        ],
        "security": {
            "pickle_warning": (
                "Python pickle is unsafe for untrusted content; verify SHA-256 "
                "and load only trusted project files."
            ),
            "private_paths_retained": False,
        },
    }
    _write_staging_metadata(destination, manifest)
    return manifest


def _atomic_replace_directory(temporary: Path, target: Path, *, replace: bool) -> None:
    _require(not target.is_symlink(), f"Refusing symbolic-link staging target: {target}")
    if not target.exists():
        temporary.replace(target)
        return
    _require(replace, f"Staging target already exists (pass --replace): {target}")
    _require(target.is_dir(), f"Staging target is not a directory: {target}")
    backup = target.parent / f".{target.name}.previous-{uuid.uuid4().hex}"
    target.replace(backup)
    try:
        temporary.replace(target)
    except BaseException:
        backup.replace(target)
        raise
    shutil.rmtree(backup)


def stage_dataset(
    *,
    repo_root: Path,
    config_path: Path,
    staging_root: Path,
    checkpoint_root: Path,
    endpoint_root: Path,
    replace: bool,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    staging_root = staging_root.absolute()
    staging_root.parent.mkdir(parents=True, exist_ok=True)
    _require(
        not staging_root.is_symlink(), f"Refusing symbolic-link staging target: {staging_root}"
    )
    config = _load_config(config_path.resolve(strict=True))
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{staging_root.name}.next-", dir=staging_root.parent)
    )
    try:
        manifest = _stage_into(
            temporary,
            repo_root=repo_root,
            config=config,
            checkpoint_root=checkpoint_root.resolve(strict=True),
            endpoint_root=endpoint_root.resolve(strict=True),
        )
        validate_staging(temporary, require_complete=False)
        _atomic_replace_directory(temporary, staging_root, replace=replace)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def validate_staging(root: Path, *, require_complete: bool) -> dict[str, Any]:
    root = root.resolve(strict=True)
    _require(root.is_dir(), f"Staging root is not a directory: {root}")
    manifest = _load_json(root / "manifest.json")
    _require(manifest.get("schema") == ARTIFACT_MANIFEST_SCHEMA, "Wrong dataset manifest schema")
    _require(
        manifest.get("bundle_schema") == "sr_filter.hf_dataset_bundle.v1", "Wrong bundle schema"
    )
    records = manifest.get("files")
    _require(isinstance(records, list), "Dataset manifest files must be a list")
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for record in records:
        _require(isinstance(record, dict), "Dataset file records must be objects")
        relative = _safe_relative(str(record.get("path", "")))
        _require(relative.as_posix() not in seen, f"Duplicate dataset record: {relative}")
        seen.add(relative.as_posix())
        path = _confined_file(root, relative)
        _assert_public_bytes(path)
        _require(path.stat().st_size == int(record.get("bytes", -1)), f"Byte mismatch: {relative}")
        _require(_sha256(path) == record.get("sha256"), f"SHA-256 mismatch: {relative}")
        counts[str(record.get("category"))] += 1
    for metadata in manifest.get("metadata_files", []):
        relative = _safe_relative(str(metadata["path"]))
        path = _confined_file(root, relative)
        _assert_public_bytes(path)
        _require(
            path.stat().st_size == int(metadata["bytes"]), f"Metadata byte mismatch: {relative}"
        )
        _require(_sha256(path) == metadata["sha256"], f"Metadata hash mismatch: {relative}")
    if manifest.get("status") == "ready_for_private_upload":
        _require(not manifest.get("missing"), "Ready bundle contains missing records")
        _require(counts["checkpoint"] == 8, "Ready bundle must have eight checkpoints")
        _require(counts["endpoint_rounds"] == 12, "Ready bundle must have twelve endpoint records")
        _require(
            counts["wall_clock"] == 12,
            "Ready bundle must have nine raw timing files plus three provenance tables",
        )
        _require(counts["online_provenance"] >= 7, "Ready bundle lacks online provenance")
        _require(
            REVISION_RE.fullmatch(str(manifest.get("code_revision", ""))) is not None,
            "Ready bundle lacks code revision",
        )
    if require_complete:
        _require(
            manifest.get("status") in {"ready_for_private_upload", "verified_pinned"},
            f"Bundle is not complete: {manifest.get('status')}",
        )
    return {
        "status": manifest.get("status"),
        "files": len(records),
        "counts": dict(sorted(counts.items())),
        "missing": len(manifest.get("missing", [])),
    }


def _pinned_url(repo_id: str, revision: str, path: str) -> str:
    encoded_path = "/".join(urllib.parse.quote(part, safe="") for part in PurePosixPath(path).parts)
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{encoded_path}"


def _refresh_local_record(main: dict[str, Any], repo_root: Path, relative_text: str) -> None:
    path = repo_root / _safe_relative(relative_text)
    _require(path.is_file(), f"Cannot refresh missing manifest target: {path}")
    record = {
        "path": relative_text,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "media_type": _media_type(path),
        "url": None,
    }
    files = [row for row in main.get("files", []) if row.get("path") != relative_text]
    files.append(record)
    main["files"] = sorted(files, key=lambda row: str(row["path"]))


def merge_release_into_main_manifest(
    main: dict[str, Any], release: Mapping[str, Any]
) -> dict[str, Any]:
    _require(main.get("schema") == ARTIFACT_MANIFEST_SCHEMA, "Wrong main artifact manifest schema")
    summary = {
        key: release.get(key)
        for key in (
            "schema",
            "status",
            "repo_id",
            "repo_type",
            "code_revision",
            "data_revision",
            "manifest_path",
            "manifest_url",
            "raw_base_url",
            "verification",
        )
    }
    main["companion_dataset"] = summary
    experiments = [
        row
        for row in main.get("experiments", [])
        if row.get("id") != "companion.camera_ready_artifacts.v1"
    ]
    if release.get("status") == "verified_pinned":
        remote_files = release.get("files")
        _require(isinstance(remote_files, list) and remote_files, "Pinned release has no files")
        experiments.append(
            {
                "id": "companion.camera_ready_artifacts.v1",
                "category": "companion_dataset",
                "status": "complete",
                "files": remote_files,
            }
        )
    main["experiments"] = experiments
    return main


def finalize_pinned_release(
    *,
    repo_root: Path,
    staging_root: Path,
    readback_root: Path,
    data_revision: str,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    staging_root = staging_root.resolve(strict=True)
    readback_root = readback_root.resolve(strict=True)
    revision = data_revision.lower()
    _require(
        REVISION_RE.fullmatch(revision) is not None,
        "Data revision must be an immutable 40- or 64-hex commit SHA",
    )
    validate_staging(staging_root, require_complete=True)
    manifest = _load_json(staging_root / "manifest.json")
    _require(
        manifest.get("status") == "ready_for_private_upload",
        "Only an unpinned ready bundle can be finalized",
    )
    code_revision = str(manifest.get("code_revision", "")).lower()
    _require(
        REVISION_RE.fullmatch(code_revision) is not None, "Staged code revision is not immutable"
    )
    current_revision, current_dirty = _git_state(repo_root)
    _require(
        current_revision == code_revision and not current_dirty,
        "Current clean Git revision must match the staged code revision",
    )

    verified = 0
    for record in manifest["files"]:
        relative = _safe_relative(str(record["path"]))
        path = _confined_file(readback_root, relative)
        _assert_public_bytes(path)
        _require(path.stat().st_size == int(record["bytes"]), f"Readback byte mismatch: {relative}")
        _require(_sha256(path) == record["sha256"], f"Readback digest mismatch: {relative}")
        verified += 1
    for record in manifest.get("metadata_files", []):
        relative = _safe_relative(str(record["path"]))
        path = _confined_file(readback_root, relative)
        _require(path.stat().st_size == int(record["bytes"]), f"Readback byte mismatch: {relative}")
        _require(_sha256(path) == record["sha256"], f"Readback digest mismatch: {relative}")
        verified += 1
    uploaded_manifest = _confined_file(readback_root, Path("manifest.json"))
    uploaded_manifest_sha256 = _sha256(uploaded_manifest)
    _require(
        uploaded_manifest_sha256 == _sha256(staging_root / "manifest.json"),
        "Downloaded manifest does not match the uploaded staging manifest",
    )
    verified += 1

    repo_id = str(manifest["repo_id"])
    raw_base = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/"
    pinned_files = []
    for record in manifest["files"]:
        pinned = dict(record)
        pinned["url"] = _pinned_url(repo_id, revision, str(record["path"]))
        pinned_files.append(pinned)
    manifest.update(
        {
            "status": "verified_pinned",
            "data_revision": revision,
            "base_url": raw_base,
            "readback_verified": True,
            "readback_verified_files": verified,
            "missing": [],
            "files": pinned_files,
            "experiments": [
                {
                    "id": "companion.camera_ready_artifacts.v1",
                    "category": "companion_dataset",
                    "status": "complete",
                    "files": pinned_files,
                }
            ],
        }
    )
    _write_staging_metadata(staging_root, manifest)
    manifest_url = _pinned_url(repo_id, revision, "manifest.json")
    release = {
        "schema": DATASET_RELEASE_SCHEMA,
        "schema_version": 1,
        "status": "verified_pinned",
        "repo_id": repo_id,
        "repo_type": "dataset",
        "code_revision": code_revision,
        "data_revision": revision,
        "manifest_path": "manifest.json",
        "manifest_url": manifest_url,
        "raw_base_url": raw_base,
        "files": pinned_files,
        "verification": {
            "readback_verified": True,
            "verified_files": verified,
            "method": "local SHA-256 and byte-size comparison after authenticated hf download",
            "uploaded_manifest_sha256": uploaded_manifest_sha256,
        },
    }

    external_path = repo_root / "artifacts/external_checkpoints.json"
    external = _load_json(external_path)
    checkpoints = external.get("checkpoints", [])
    _require(
        len(checkpoints) == 8 and all(row.get("status") == "sanitized" for row in checkpoints),
        "All eight checkpoints must be sanitized before pinning",
    )
    pinned_by_path = {str(row["path"]): row for row in pinned_files}
    for row in checkpoints:
        dataset_record = pinned_by_path.get(str(row["path"]))
        _require(
            dataset_record is not None, f"Checkpoint absent from uploaded dataset: {row['path']}"
        )
        _require(
            dataset_record["sha256"] == row["sanitized_sha256"],
            f"Checkpoint digest mismatch while pinning: {row['id']}",
        )
        row["url"] = dataset_record["url"]
    external["remote"] = {"repo_id": repo_id, "repo_type": "dataset", "revision": revision}
    external["status"] = "published_pinned"

    release_path = repo_root / "artifacts/hf_dataset_release.json"
    main_path = repo_root / "artifacts/manifest.json"
    main = _load_json(main_path)
    _atomic_json(external_path, external)
    _atomic_json(release_path, release)
    main = merge_release_into_main_manifest(main, release)
    _refresh_local_record(main, repo_root, "artifacts/external_checkpoints.json")
    _refresh_local_record(main, repo_root, "artifacts/hf_dataset_release.json")
    _atomic_json(main_path, main)
    return {
        "status": "verified_pinned",
        "repo_id": repo_id,
        "code_revision": code_revision,
        "data_revision": revision,
        "verified_files": verified,
        "manifest_url": manifest_url,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    stage = commands.add_parser("stage", help="Atomically assemble a local private-upload bundle")
    stage.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    stage.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    stage.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    stage.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    stage.add_argument("--endpoint-root", type=Path, default=DEFAULT_ENDPOINT_ROOT)
    stage.add_argument("--replace", action="store_true")
    stage.add_argument("--allow-pending", action="store_true")

    validate = commands.add_parser(
        "validate", help="Verify a staged dataset without network access"
    )
    validate.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    validate.add_argument("--allow-pending", action="store_true")

    finalize = commands.add_parser(
        "finalize",
        help="Pin an immutable, byte-verified private-upload readback in checked manifests",
    )
    finalize.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    finalize.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    finalize.add_argument("--readback-root", type=Path, required=True)
    finalize.add_argument("--data-revision", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "stage":
            report = stage_dataset(
                repo_root=args.repo_root,
                config_path=args.config,
                staging_root=args.staging_root,
                checkpoint_root=args.checkpoint_root,
                endpoint_root=args.endpoint_root,
                replace=args.replace,
            )
            result = {
                "status": report["status"],
                "staging_root": str(args.staging_root),
                "files": len(report["files"]),
                "missing": report["missing"],
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if report["status"] == "ready_for_private_upload" or args.allow_pending else 2
        if args.command == "validate":
            report = validate_staging(args.staging_root, require_complete=not args.allow_pending)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        report = finalize_pinned_release(
            repo_root=args.repo_root,
            staging_root=args.staging_root,
            readback_root=args.readback_root,
            data_revision=args.data_revision,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, KeyError, OSError, ValueError, tomllib.TOMLDecodeError) as error:
        parser = argparse.ArgumentParser(prog="stage_hf_dataset.py")
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
