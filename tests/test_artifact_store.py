from __future__ import annotations

import hashlib
import json

import pytest

from sr_filtering_nqs.artifact_store import (
    MANIFEST_SCHEMA,
    fetch_artifacts,
    manifest_files,
    safe_relative_path,
)


def test_fetch_artifacts_copies_and_verifies_local_payload(tmp_path):
    source_root = tmp_path / "source"
    source = source_root / "nested" / "payload.txt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"portable artifact\n")
    record = {
        "path": "nested/payload.txt",
        "bytes": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "experiments": [{"id": "demo.v1", "files": [record]}],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    output_root = tmp_path / "outputs"
    first = fetch_artifacts(
        manifest_path,
        output_root=output_root,
        source_root=source_root,
        experiment_ids=["demo.v1"],
    )
    second = fetch_artifacts(
        manifest_path,
        output_root=output_root,
        source_root=source_root,
        experiment_ids=["demo.v1"],
    )

    assert first["fetched"] == [str(output_root / "nested/payload.txt")]
    assert second["reused"] == [str(output_root / "nested/payload.txt")]
    assert (output_root / "nested/payload.txt").read_bytes() == source.read_bytes()


def test_artifact_manifest_filter_and_path_safety():
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "experiments": {
            "first": {"files": [{"path": "first.bin"}]},
            "second": {"files": [{"path": "second.bin"}]},
        },
    }
    assert [row["path"] for row in manifest_files(manifest, experiment_ids=["second"])] == [
        "second.bin"
    ]
    with pytest.raises(KeyError):
        manifest_files(manifest, experiment_ids=["missing"])
    with pytest.raises(ValueError):
        safe_relative_path("../outside")
    with pytest.raises(ValueError):
        safe_relative_path("/absolute")
    for nonportable in (r"..\outside", r"C:\outside", r"\\server\share"):
        with pytest.raises(ValueError, match="portable relative POSIX"):
            safe_relative_path(nonportable)


def test_fetch_rejects_symlink_escape_and_oversized_source(tmp_path):
    source_root = tmp_path / "source"
    (source_root / "nested").mkdir(parents=True)
    (source_root / "nested" / "payload.bin").write_bytes(b"too large")
    record = {
        "path": "nested/payload.bin",
        "bytes": 3,
        "sha256": hashlib.sha256(b"too").hexdigest(),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"schema": MANIFEST_SCHEMA, "experiments": [{"id": "demo", "files": [record]}]}
        )
    )

    output_root = tmp_path / "outputs"
    output_root.mkdir()
    escape = tmp_path / "escape"
    escape.mkdir()
    (output_root / "nested").symlink_to(escape, target_is_directory=True)
    with pytest.raises(ValueError, match="traverses"):
        fetch_artifacts(
            manifest_path,
            output_root=output_root,
            source_root=source_root,
            experiment_ids=["demo"],
        )
    assert not (escape / "payload.bin").exists()

    (output_root / "nested").unlink()
    with pytest.raises(ValueError, match="exceeds its declared size"):
        fetch_artifacts(
            manifest_path,
            output_root=output_root,
            source_root=source_root,
            experiment_ids=["demo"],
        )


def test_remote_manifests_require_https():
    with pytest.raises(ValueError, match="must use HTTPS"):
        fetch_artifacts("http://example.test/manifest.json")


def test_declared_byte_limits_fail_before_creating_output(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "experiments": [
                    {
                        "id": "demo",
                        "files": [
                            {"path": "one.bin", "bytes": 6, "sha256": "a" * 64},
                            {"path": "two.bin", "bytes": 5, "sha256": "b" * 64},
                        ],
                    }
                ],
            }
        )
    )

    per_file_output = tmp_path / "per-file-output"
    with pytest.raises(ValueError, match="per-file limit"):
        fetch_artifacts(
            manifest_path,
            output_root=per_file_output,
            experiment_ids=["demo"],
            max_file_bytes=5,
            max_total_bytes=20,
        )
    assert not per_file_output.exists()

    aggregate_output = tmp_path / "aggregate-output"
    with pytest.raises(ValueError, match="aggregate limit"):
        fetch_artifacts(
            manifest_path,
            output_root=aggregate_output,
            experiment_ids=["demo"],
            max_file_bytes=10,
            max_total_bytes=10,
        )
    assert not aggregate_output.exists()


def test_local_source_symlink_must_remain_inside_source_root(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (source_root / "payload.bin").symlink_to(outside)
    record = {
        "path": "payload.bin",
        "bytes": outside.stat().st_size,
        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"schema": MANIFEST_SCHEMA, "experiments": [{"id": "demo", "files": [record]}]}
        )
    )

    with pytest.raises(ValueError, match="escapes its source root"):
        fetch_artifacts(
            manifest_path,
            output_root=tmp_path / "outputs",
            source_root=source_root,
            experiment_ids=["demo"],
        )


def test_release_manifest_primary_files_and_config_are_resolved():
    artifact = {"path": "artifacts/demo.csv", "bytes": 1, "sha256": "a" * 64}
    config = {"path": "configs/demo.toml", "bytes": 1, "sha256": "b" * 64}
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "files": [artifact],
        "configs": [config],
        "experiments": [
            {
                "id": "demo.v1",
                "primary_files": ["artifacts/demo.csv"],
                "config": "configs/demo.toml",
            }
        ],
    }

    rows = manifest_files(manifest, experiment_ids=["demo.v1"])

    assert [row["path"] for row in rows] == ["artifacts/demo.csv", "configs/demo.toml"]
    assert all(row["experiment_id"] == "demo.v1" for row in rows)
