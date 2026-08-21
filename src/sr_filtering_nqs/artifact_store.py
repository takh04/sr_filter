"""Portable, checksum-verified access to external experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any


MANIFEST_SCHEMA = "sr_filter.artifact_manifest.v1"
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_MANIFEST_RECORDS = 10_000
DEFAULT_MAX_FILE_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024


def _is_url(value: str) -> bool:
    return urllib.parse.urlparse(value).scheme == "https"


def _open_https(url: str):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(f"Remote artifact URLs must use HTTPS, got {url!r}")
    response = urllib.request.urlopen(url, timeout=30)  # noqa: S310 - HTTPS checked above
    final_url = response.geturl()
    if urllib.parse.urlparse(final_url).scheme != "https":
        response.close()
        raise ValueError(f"Artifact download redirected away from HTTPS: {final_url!r}")
    return response


def _read_limited(handle, *, limit: int, description: str) -> bytes:
    payload = handle.read(int(limit) + 1)
    if len(payload) > int(limit):
        raise ValueError(f"{description} exceeds the {int(limit)}-byte limit")
    return payload


def load_manifest(locator: str | Path) -> tuple[dict[str, Any], str]:
    text_locator = str(locator)
    if _is_url(text_locator):
        with _open_https(text_locator) as response:
            payload = json.loads(
                _read_limited(
                    response,
                    limit=MAX_MANIFEST_BYTES,
                    description="Remote artifact manifest",
                )
            )
    elif urllib.parse.urlparse(text_locator).scheme:
        raise ValueError(f"Manifest URL must use HTTPS, got {text_locator!r}")
    else:
        manifest_path = Path(text_locator)
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ValueError(
                f"Artifact manifest exceeds the {MAX_MANIFEST_BYTES}-byte limit: "
                f"{manifest_path}"
            )
        payload = json.loads(manifest_path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError("Artifact manifest must be a JSON object")
    schema = payload.get("schema", payload.get("schema_version"))
    if schema != MANIFEST_SCHEMA:
        raise ValueError(f"Expected manifest schema {MANIFEST_SCHEMA!r}, got {schema!r}")
    return payload, text_locator


def _experiment_items(manifest: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    experiments = manifest.get("experiments", {})
    if isinstance(experiments, Mapping):
        return [(str(key), value) for key, value in experiments.items() if isinstance(value, Mapping)]
    if isinstance(experiments, list):
        rows = []
        for value in experiments:
            if not isinstance(value, Mapping) or not value.get("id"):
                raise ValueError("Each manifest experiment must be an object with an id")
            rows.append((str(value["id"]), value))
        return rows
    raise ValueError("manifest.experiments must be an object or list")


def manifest_files(
    manifest: Mapping[str, Any],
    *,
    experiment_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    selected = None if experiment_ids is None else set(experiment_ids)
    known: set[str] = set()
    rows: list[dict[str, Any]] = []

    top_files = manifest.get("files", [])
    if not isinstance(top_files, list):
        raise ValueError("manifest.files must be a list")
    top_records = {
        str(value.get("path", "")): dict(value)
        for value in top_files
        if isinstance(value, Mapping)
    }
    config_files = manifest.get("configs", [])
    if not isinstance(config_files, list):
        raise ValueError("manifest.configs must be a list")
    config_records = {
        str(value.get("path", "")): dict(value)
        for value in config_files
        if isinstance(value, Mapping)
    }
    if selected is None:
        rows.extend(top_records.values())
        rows.extend(config_records.values())

    for experiment_id, experiment in _experiment_items(manifest):
        known.add(experiment_id)
        if selected is None:
            continue
        if selected is not None and experiment_id not in selected:
            continue
        files = experiment.get("files")
        if files is None:
            primary_paths = experiment.get("primary_files", [])
            if not isinstance(primary_paths, list):
                raise ValueError(
                    f"Experiment {experiment_id!r} primary_files must be a list"
                )
            files = []
            for path in primary_paths:
                if str(path) not in top_records:
                    raise ValueError(
                        f"Experiment {experiment_id!r} references unknown file {path!r}"
                    )
                files.append(top_records[str(path)])
            config_path = experiment.get("config")
            if config_path:
                if str(config_path) not in config_records:
                    raise ValueError(
                        f"Experiment {experiment_id!r} references unknown config "
                        f"{config_path!r}"
                    )
                files.append(config_records[str(config_path)])
        if not isinstance(files, list):
            raise ValueError(f"Experiment {experiment_id!r} files must be a list")
        for value in files:
            if not isinstance(value, Mapping):
                raise ValueError(f"Experiment {experiment_id!r} has a non-object file record")
            row = dict(value)
            row.setdefault("experiment_id", experiment_id)
            rows.append(row)

    if selected is not None:
        missing = selected - known
        if missing:
            raise KeyError(f"Unknown experiment ids: {', '.join(sorted(missing))}")
    deduplicated: dict[str, dict[str, Any]] = {}
    for row in rows:
        path = str(row.get("path", ""))
        if not path:
            raise ValueError("Every artifact record needs a path")
        if path in deduplicated and deduplicated[path] != row:
            raise ValueError(f"Conflicting artifact records for {path}")
        deduplicated[path] = row
    if len(deduplicated) > MAX_MANIFEST_RECORDS:
        raise ValueError(
            f"Artifact manifest selects {len(deduplicated)} files; "
            f"limit is {MAX_MANIFEST_RECORDS}"
        )
    return [deduplicated[path] for path in sorted(deduplicated)]


def safe_relative_path(value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Artifact path must be a nonempty string, got {value!r}")
    if "\\" in value or "\x00" in value or ":" in value:
        raise ValueError(
            f"Artifact path must use portable relative POSIX syntax, got {value!r}"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or path == PurePosixPath("."):
        raise ValueError(f"Artifact path must be a safe relative POSIX path, got {value!r}")
    return Path(*path.parts)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(path: str | Path, record: Mapping[str, Any]) -> None:
    path = Path(path)
    expected_bytes = record.get("bytes")
    if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
        raise ValueError(
            f"Size mismatch for {path}: expected {expected_bytes}, got {path.stat().st_size}"
        )
    expected_sha256 = str(record.get("sha256", "")).lower()
    if len(expected_sha256) != 64:
        raise ValueError(f"Invalid SHA-256 in manifest for {record.get('path')!r}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )


def _source_for_record(
    record: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_locator: str,
    source_root: str | Path | None,
) -> str | Path:
    if record.get("url"):
        return str(record["url"])
    base_url = record.get("base_url", manifest.get("base_url"))
    if base_url:
        return urllib.parse.urljoin(str(base_url).rstrip("/") + "/", str(record["path"]))
    relative = safe_relative_path(str(record["path"]))
    if source_root is not None:
        return _confined_local_source(Path(source_root), relative)
    if _is_url(manifest_locator):
        return urllib.parse.urljoin(manifest_locator, str(record["path"]))
    manifest_path = Path(manifest_locator).resolve()
    candidate_roots = (manifest_path.parent, manifest_path.parent.parent)
    for candidate_root in candidate_roots:
        candidate = candidate_root / relative
        if candidate.is_file():
            return _confined_local_source(candidate_root, relative)
    return _confined_local_source(candidate_roots[0], relative, require_exists=False)


def _confined_local_source(
    source_root: Path,
    relative: Path,
    *,
    require_exists: bool = True,
) -> Path:
    resolved_root = source_root.resolve(strict=require_exists)
    candidate = (resolved_root / relative).resolve(strict=require_exists)
    if not candidate.is_relative_to(resolved_root):
        raise ValueError(f"Artifact source escapes its source root: {candidate}")
    if require_exists and not candidate.is_file():
        raise ValueError(f"Artifact source is not a regular file: {candidate}")
    return candidate


def _copy_source(
    source: str | Path,
    destination_handle,
    *,
    expected_bytes: int,
) -> None:
    if isinstance(source, str) and _is_url(source):
        source_handle = _open_https(source)
    elif isinstance(source, str) and urllib.parse.urlparse(source).scheme:
        raise ValueError(f"Artifact URL must use HTTPS, got {source!r}")
    else:
        source_handle = Path(source).open("rb")
    with source_handle:
        copied = 0
        while True:
            chunk = source_handle.read(min(1024 * 1024, expected_bytes + 1 - copied))
            if not chunk:
                break
            copied += len(chunk)
            if copied > expected_bytes:
                raise ValueError(
                    f"Artifact source exceeds its declared size of {expected_bytes} bytes"
                )
            destination_handle.write(chunk)


def _safe_destination(output_root: Path, relative: Path) -> Path:
    """Create a destination parent without traversing attacker-controlled links."""

    output_root.mkdir(parents=True, exist_ok=True)
    resolved_root = output_root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise ValueError(f"Artifact output root is not a directory: {output_root}")
    parent = resolved_root
    for part in relative.parts[:-1]:
        parent = parent / part
        if parent.exists() or parent.is_symlink():
            if parent.is_symlink() or not parent.is_dir():
                raise ValueError(f"Artifact output path traverses a non-directory or link: {parent}")
        else:
            parent.mkdir()
    destination = parent / relative.name
    if destination.is_symlink():
        raise ValueError(f"Artifact destination must not be a symbolic link: {destination}")
    if not destination.resolve(strict=False).is_relative_to(resolved_root):
        raise ValueError(f"Artifact destination escapes the output root: {destination}")
    return destination


def fetch_artifacts(
    manifest_locator: str | Path,
    *,
    output_root: str | Path = "outputs",
    experiment_ids: Iterable[str] | None = None,
    source_root: str | Path | None = None,
    force: bool = False,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> dict[str, Any]:
    manifest, normalized_locator = load_manifest(manifest_locator)
    records = manifest_files(manifest, experiment_ids=experiment_ids)
    max_file_bytes = int(max_file_bytes)
    max_total_bytes = int(max_total_bytes)
    if max_file_bytes <= 0 or max_total_bytes <= 0:
        raise ValueError("Artifact byte limits must be positive")
    prepared: list[tuple[Mapping[str, Any], Path, int]] = []
    declared_total = 0
    for record in records:
        relative = safe_relative_path(str(record["path"]))
        try:
            expected_bytes = int(record["bytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Artifact record needs an integer byte count: {record!r}") from error
        if expected_bytes < 0:
            raise ValueError(f"Artifact byte count must be nonnegative: {expected_bytes}")
        if expected_bytes > max_file_bytes:
            raise ValueError(
                f"Artifact {relative} declares {expected_bytes} bytes; "
                f"per-file limit is {max_file_bytes}"
            )
        declared_total += expected_bytes
        if declared_total > max_total_bytes:
            raise ValueError(
                f"Selected artifacts declare {declared_total} bytes; "
                f"aggregate limit is {max_total_bytes}"
            )
        prepared.append((record, relative, expected_bytes))
    output_root = Path(output_root)
    fetched = []
    reused = []
    for record, relative, expected_bytes in prepared:
        destination = _safe_destination(output_root, relative)
        if destination.is_file() and not force:
            verify_artifact(destination, record)
            reused.append(str(destination))
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = _source_for_record(
            record,
            manifest=manifest,
            manifest_locator=normalized_locator,
            source_root=source_root,
        )
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            try:
                _copy_source(source, temporary, expected_bytes=expected_bytes)
                temporary.flush()
                verify_artifact(temporary_path, record)
                temporary_path.replace(destination)
            except BaseException:
                temporary_path.unlink(missing_ok=True)
                raise
        fetched.append(str(destination))
    return {
        "schema": MANIFEST_SCHEMA,
        "manifest": normalized_locator,
        "output_root": str(output_root),
        "fetched": fetched,
        "reused": reused,
        "count": len(fetched) + len(reused),
    }
