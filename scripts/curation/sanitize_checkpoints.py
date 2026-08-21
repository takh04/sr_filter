#!/usr/bin/env python3
"""Sanitize trusted checkpoint pickles for external artifact staging.

This command intentionally unpickles its inputs. Run it only on trusted local
checkpoints after setting the environment variables named by the inventory.
Original checkpoints are opened read-only and are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import tempfile
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = REPO_ROOT / "configs/experiments/external_checkpoint_inventory.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument(
        "--public-manifest",
        type=Path,
        default=REPO_ROOT / "artifacts/external_checkpoints.json",
    )
    parser.add_argument(
        "--verify-runtime",
        action="store_true",
        help="Also reconstruct each variational driver; requires the experiment environment and an available accelerator.",
    )
    parser.add_argument(
        "--acknowledge-trusted-pickle-inputs",
        action="store_true",
        help="Required acknowledgement that every input pickle is trusted.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def sanitize_metadata_paths(
    value: Any,
    *,
    replacement_id: str,
    key_path: str = "",
) -> tuple[Any, list[dict[str, str]]]:
    replacements: list[dict[str, str]] = []
    if isinstance(value, dict):
        clean = {}
        for key, child in value.items():
            child_path = f"{key_path}.{key}" if key_path else str(key)
            clean_child, child_replacements = sanitize_metadata_paths(
                child,
                replacement_id=replacement_id,
                key_path=child_path,
            )
            clean[key] = clean_child
            replacements.extend(child_replacements)
        return clean, replacements
    if isinstance(value, list):
        clean_list = []
        for index, child in enumerate(value):
            clean_child, child_replacements = sanitize_metadata_paths(
                child,
                replacement_id=replacement_id,
                key_path=f"{key_path}[{index}]",
            )
            clean_list.append(clean_child)
            replacements.extend(child_replacements)
        return clean_list, replacements
    if isinstance(value, tuple):
        clean_items = []
        for index, child in enumerate(value):
            clean_child, child_replacements = sanitize_metadata_paths(
                child,
                replacement_id=replacement_id,
                key_path=f"{key_path}[{index}]",
            )
            clean_items.append(clean_child)
            replacements.extend(child_replacements)
        return tuple(clean_items), replacements
    if isinstance(value, (str, Path)) and Path(str(value)).is_absolute():
        replacement = f"artifact://{replacement_id}"
        replacements.append({"field": key_path, "replacement": replacement})
        return replacement, replacements
    return value, replacements


def array_signature(value: Any) -> list[tuple[str, tuple[int, ...], str]]:
    signature: list[tuple[str, tuple[int, ...], str]] = []

    def walk(node: Any, key_path: str) -> None:
        if isinstance(node, Mapping):
            for key in sorted(node, key=str):
                walk(node[key], f"{key_path}.{key}" if key_path else str(key))
        elif isinstance(node, (list, tuple)):
            for index, child in enumerate(node):
                walk(child, f"{key_path}[{index}]")
        elif hasattr(node, "shape") and hasattr(node, "dtype"):
            signature.append((key_path, tuple(int(item) for item in node.shape), str(node.dtype)))

    walk(value, "")
    return signature


def validate_roundtrip(original: dict[str, Any], sanitized_path: Path, expected_step: int) -> dict[str, Any]:
    with sanitized_path.open("rb") as handle:
        sanitized = pickle.load(handle)
    if not isinstance(sanitized, dict):
        raise TypeError(f"Sanitized checkpoint is not a mapping: {sanitized_path}")
    if int(sanitized.get("step", -1)) != int(expected_step):
        raise ValueError(f"Checkpoint step mismatch after sanitization: {sanitized_path}")
    if sorted(original) != sorted(sanitized):
        raise ValueError(f"Top-level checkpoint keys changed: {sanitized_path}")
    original_signature = array_signature(original.get("parameters"))
    sanitized_signature = array_signature(sanitized.get("parameters"))
    if original_signature != sanitized_signature or not sanitized_signature:
        raise ValueError(f"Parameter tree signature changed or is empty: {sanitized_path}")
    config = sanitized.get("config")
    if not isinstance(config, Mapping) or config.get("system") not in {"j1j2", "ssm", "tfim"}:
        raise ValueError(f"Checkpoint configuration cannot be reconstructed: {sanitized_path}")
    return {
        "status": "passed",
        "checkpoint_step": int(sanitized["step"]),
        "system": str(config["system"]),
        "parameter_array_count": len(sanitized_signature),
    }


def runtime_reconstruct(path: Path, system: str) -> dict[str, Any]:
    from sr_filtering_nqs.large_scale.core.fixed_data_protocol import _load_training_driver

    loaded = _load_training_driver(path, system_name=system)
    return {
        "status": "passed",
        "system": str(loaded["system"]),
        "checkpoint_step": int(loaded["step"]),
        "driver_type": type(loaded["driver"]).__name__,
    }


def main() -> None:
    args = parse_args()
    if not args.acknowledge_trusted_pickle_inputs:
        raise SystemExit("Refusing to unpickle inputs without --acknowledge-trusted-pickle-inputs")
    inventory_path = args.inventory.resolve()
    staging_root = args.staging_root.resolve()
    public_manifest = args.public_manifest.resolve()
    with inventory_path.open("rb") as handle:
        inventory = tomllib.load(handle)
    if inventory.get("schema") != "sr_filter.external_checkpoint_inventory.v1":
        raise ValueError("Unsupported external checkpoint inventory schema")

    staging_root.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in inventory["checkpoints"]:
        environment_variable = str(spec["environment_variable"])
        raw_path = os.environ.get(environment_variable, "").strip()
        if not raw_path:
            if spec["status"] == "complete":
                raise ValueError(f"Missing required environment variable {environment_variable}")
            records.append(
                {
                    "id": spec["id"],
                    "role": spec["role"],
                    "branch_seed": spec["branch_seed"],
                    "arm": spec.get("arm"),
                    "status": spec["status"],
                    "path": spec["staged_path"],
                    "url": None,
                    "bytes": None,
                    "original_sha256": spec.get("expected_original_sha256"),
                    "sanitized_sha256": None,
                    "verification": "pending_input",
                }
            )
            continue

        source_path = Path(raw_path).resolve(strict=True)
        destination = (staging_root / spec["staged_path"]).resolve()
        try:
            destination.relative_to(staging_root)
        except ValueError as error:
            raise ValueError(f"Staged path escapes staging root: {destination}") from error
        if destination == source_path:
            raise ValueError(f"Refusing to overwrite source checkpoint: {source_path}")
        original_digest = sha256(source_path)
        expected_digest = spec.get("expected_original_sha256")
        if expected_digest is not None and original_digest != expected_digest:
            raise ValueError(f"Original checkpoint checksum mismatch for {spec['id']}")

        with source_path.open("rb") as handle:
            original = pickle.load(handle)
        if not isinstance(original, dict):
            raise TypeError(f"Checkpoint must be a mapping: {source_path}")
        sanitized, replacements = sanitize_metadata_paths(
            original,
            replacement_id=str(spec["absolute_path_replacement_id"]),
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as handle:
            pickle.dump(sanitized, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary = Path(handle.name)
        temporary.replace(destination)
        payload = destination.read_bytes()
        forbidden = (
            b"/home/",
            b"/Users/",
            b"benign_" + b"overfitting_nqs",
            b"beingin_" + b"overfitting_nqs",
        )
        if any(token in payload for token in forbidden):
            raise ValueError(f"Private path/name remains in sanitized checkpoint {spec['id']}")
        roundtrip = validate_roundtrip(original, destination, int(spec["expected_checkpoint_step"]))
        runtime = (
            runtime_reconstruct(destination, roundtrip["system"])
            if args.verify_runtime
            else {"status": "not_requested"}
        )
        records.append(
            {
                "id": spec["id"],
                "role": spec["role"],
                "branch_seed": spec["branch_seed"],
                "arm": spec.get("arm"),
                "status": "sanitized",
                "path": spec["staged_path"],
                "url": None,
                "bytes": destination.stat().st_size,
                "original_sha256": original_digest,
                "sanitized_sha256": sha256(destination),
                "path_metadata_replacements": replacements,
                "pickle_roundtrip_verification": roundtrip,
                "runtime_reconstruction_verification": runtime,
            }
        )

    manifest = {
        "schema": "sr_filter.external_checkpoint_manifest.v1",
        "schema_version": 1,
        "status": (
            "ready_for_authenticated_upload"
            if all(record["status"] == "sanitized" for record in records)
            else "pending_gpu_outputs"
        ),
        "remote": {"repo_type": inventory["remote_repo_type"], "repo_id": None, "revision": None},
        "security": {
            "input_trust": "trusted local pickle inputs only",
            "consumer_requirement": "verify sanitized_sha256 before loading; pickle remains unsafe for untrusted content",
            "originals_modified": False,
        },
        "checkpoints": records,
    }
    atomic_json(staging_root / "manifest.json", manifest)
    atomic_json(public_manifest, manifest)
    print(json.dumps({"status": manifest["status"], "records": len(records), "manifest": str(public_manifest)}))


if __name__ == "__main__":
    main()
