#!/usr/bin/env python3
"""Audit files that would enter a public SR Filter source release.

The script is intentionally standard-library-only so CI can run it before
installing the project or executing any repository code.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tomllib
import urllib.parse
from collections.abc import Mapping
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_RELEASE_BYTES = 100 * 1024 * 1024


def joined(*parts: str) -> str:
    """Construct audit needles without making this source match itself."""

    return "".join(parts)


FORBIDDEN_TEXT = (
    joined("workspace/", "nqs_rev"),
    joined("beingin_", "overfitting_nqs"),
    joined("benign_", "overfitting_nqs"),
    joined("Anon", "NeurIPS2026"),
    joined("anonymous ", "review package"),
    joined("convergence", "_ai"),
    joined("MSSR_", "PRED_CHUNK"),
    joined("FIXED_DATA_", "WORKSPACE_NAME"),
    joined("FIXED_DATA_", "CHUNK_BWD_OVERRIDE"),
    joined("FIXED_DATA_", "LR_OVERRIDE"),
    joined("PII_", "TAU_J1J2"),
    joined("PII_", "TAU_SSM"),
    joined("PII_", "TAU_TFIM"),
)
FORBIDDEN_TEXT_DEFINITION_FILES = {
    Path("paper/scripts/check_camera_ready.py"),
    Path("scripts/audit_release.py"),
}

PRIVATE_PATH_RULES = (
    (
        "Unix user-home path",
        re.compile(rb"(?<![A-Za-z0-9_])/(?:home|Users)/[A-Za-z0-9._-]+/"),
    ),
    (
        "cluster-local absolute path",
        re.compile(rb"(?<![A-Za-z0-9_])/(?:scratch|gpfs|lustre|nfs)/[^\x00\r\n\t ]+"),
    ),
    (
        "Windows user-home path",
        re.compile(rb"(?i)(?<![A-Za-z0-9_])[A-Z]:[\\/]+Users[\\/]+[^\\/\x00\r\n\t ]+[\\/]"),
    ),
    (
        "file URI",
        re.compile(rb"(?i)file:///(?:home|Users|[A-Z]:/Users)/"),
    ),
)

SECRET_RULES = (
    ("GitHub token", re.compile(rb"gh[pousr]_[A-Za-z0-9_]{20,255}")),
    ("Hugging Face token", re.compile(rb"hf_[A-Za-z0-9]{30,}")),
    ("AWS access-key identifier", re.compile(rb"(?:AKIA|ASIA)[0-9A-Z]{16}")),
    ("Google API key", re.compile(rb"AIza[0-9A-Za-z_-]{35}")),
    ("Slack token", re.compile(rb"xox[baprs]-[0-9A-Za-z-]{20,}")),
    ("Stripe secret key", re.compile(rb"(?:sk|rk)_(?:live|test)_[0-9A-Za-z]{16,}")),
    (
        "private key",
        re.compile(rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    ),
    (
        "credential embedded in URL",
        re.compile(rb"(?i)https?://[^/@:\s]{1,128}:[^/@\s]{8,256}@"),
    ),
    (
        "assigned credential",
        re.compile(
            rb"(?i)(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
            rb"password|passwd)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}"
        ),
    ),
)

FORBIDDEN_PDF_BYTES = (
    joined("Quartz", " PDFContext").encode(),
    joined("mac", "OS Version").encode(),
    joined("Key", "note").encode(),
)

CACHE_OR_OUTPUT_PARTS = {
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "outputs",
}
COLLAPSED_FORBIDDEN_ROOTS = {
    "rebuttal_assets": (
        "legacy review-period package; curate its contents and remove this directory"
    ),
}
FORBIDDEN_FILENAMES = {
    ".DS_Store",
    ".env",
    ".netrc",
    ".pypirc",
    "Thumbs.db",
    "credentials.json",
    "id_dsa",
    "id_ed25519",
    "id_rsa",
}
GENERATED_SUFFIXES = (
    ".aux",
    ".bbl",
    ".bcf",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".nav",
    ".out",
    ".pyc",
    ".run.xml",
    ".snm",
    ".synctex.gz",
    ".toc",
    ".vrb",
    ".xdv",
)
ARCHIVE_SUFFIXES = (".7z", ".rar", ".tar", ".tar.gz", ".tgz", ".zip")
ALLOWED_PICKLE_ROOT = Path("artifacts/small_scale")

REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "CITATION.cff",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "pyproject.toml",
    "uv.lock",
    "artifacts/manifest.json",
    "artifacts/exclusions.json",
    "artifacts/hf_dataset_release.json",
    "configs/release/hf_dataset.toml",
    "configs/experiments/online_j1j2_branch2500_seed0.toml",
    "configs/experiments/online_j1j2_branch1800_seed1.toml",
    "configs/experiments/endpoint_eval_j1j2_seed0.toml",
    "configs/experiments/endpoint_eval_j1j2_seed1.toml",
    "configs/experiments/robustness_checkpoint_local.toml",
    "configs/experiments/wall_clock_h100.toml",
    "configs/experiments/hardware_h100.toml",
    ".github/workflows/ci.yml",
    ".github/workflows/cuda.yml",
)

REQUIRED_MARKERS = {
    "README.md": (
        "Stochastic Reconfiguration as Statistical Filtering",
        "https://github.com/takh04/sr_filter",
        "takh04/sr-filter-artifacts",
        "Never load an untrusted pickle",
    ),
    "LICENSE": ("Apache License", "Version 2.0, January 2004"),
    "NOTICE": ("Copyright 2026 Tak Hur", "NetKet Authors", "Google LLC"),
    "THIRD_PARTY_NOTICES.md": (
        "https://github.com/netket/netket",
        "https://github.com/google/neural-tangents",
        "Apache License 2.0",
    ),
    "CITATION.cff": (
        "family-names: Hur",
        "given-names: Tak",
        "Yonsei University",
        "https://github.com/takh04/sr_filter",
        "license: Apache-2.0",
    ),
    "SECURITY.md": ("takh0404@yonsei.ac.kr", "arbitrary code", "trust"),
    "pyproject.toml": (
        'name = "sr-filtering-nqs"',
        "Tak Hur",
        "https://github.com/takh04/sr_filter",
    ),
}


def release_paths_from_git() -> list[Path] | None:
    """Return tracked and non-ignored candidate files, or None outside Git."""

    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                os.fspath(ROOT),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    return sorted({ROOT / os.fsdecode(raw) for raw in result.stdout.split(b"\0") if raw})


def release_paths_from_tree() -> list[Path]:
    """Fallback for auditing an unpacked source archive without Git."""

    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if name != ".git")
        base = Path(dirpath)
        paths.extend(base / filename for filename in sorted(filenames))
    return paths


def release_paths() -> list[Path]:
    paths = release_paths_from_git()
    return paths if paths is not None else release_paths_from_tree()


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return repr(os.fspath(path))


def audit_path(path: Path) -> tuple[list[str], int]:
    failures: list[str] = []
    rel = path.relative_to(ROOT)
    rel_text = rel.as_posix()

    if rel.is_absolute() or ".." in rel.parts:
        return [f"unsafe release path: {rel_text}"], 0
    if any(part in CACHE_OR_OUTPUT_PARTS for part in rel.parts):
        failures.append("generated, runtime, or review-period path")
    if path.name in FORBIDDEN_FILENAMES:
        failures.append("credential, environment, or operating-system file")
    if path.name.endswith("~") or path.name.startswith(".#"):
        failures.append("editor backup file")
    if rel_text.endswith(GENERATED_SUFFIXES):
        failures.append("generated build file")
    if rel_text.lower().endswith(ARCHIVE_SUFFIXES):
        failures.append("bundled archive; publish its audited contents instead")
    if path.name.startswith("delta_bank") and path.suffix == ".pkl":
        failures.append("raw delta-bank pickle")
    if path.suffix == ".pkl" and ALLOWED_PICKLE_ROOT not in rel.parents:
        failures.append("pickle outside curated small-scale artifacts")

    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        failures.append("listed release file is missing")
        return failures, 0

    if stat.S_ISLNK(mode):
        target_text = os.readlink(path)
        target = Path(target_text)
        resolved = (path.parent / target).resolve(strict=False)
        try:
            resolved.relative_to(ROOT.resolve())
        except ValueError:
            failures.append("symlink resolves outside the repository")
        if target.is_absolute():
            failures.append("absolute symlink")
        if not path.exists():
            failures.append("broken symlink")
        return failures, len(os.fsencode(target_text))

    if not stat.S_ISREG(mode):
        failures.append("non-regular release entry")
        return failures, 0

    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        failures.append(f"file is {size} bytes; limit is {MAX_FILE_BYTES}")
    return failures, size


def audit_content(path: Path) -> list[str]:
    try:
        data = path.read_bytes()
    except (OSError, MemoryError) as exc:
        return [f"cannot read file: {exc}"]

    failures: list[str] = []
    if path.relative_to(ROOT) not in FORBIDDEN_TEXT_DEFINITION_FILES:
        lower_data = data.lower()
        for needle in FORBIDDEN_TEXT:
            if needle.lower().encode() in lower_data:
                failures.append(f"private/review text: {needle}")
    for name, pattern in PRIVATE_PATH_RULES:
        if pattern.search(data):
            failures.append(name)
    for name, pattern in SECRET_RULES:
        if pattern.search(data):
            failures.append(f"secret-like value ({name})")
    if path.suffix.lower() == ".pdf":
        for needle in FORBIDDEN_PDF_BYTES:
            if needle in data:
                failures.append("private authoring-tool PDF metadata")

    if path.suffix.lower() == ".json":
        try:
            json.loads(
                data,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-standard numeric constant {value}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            failures.append(f"invalid strict JSON: {exc}")
    elif path.suffix.lower() == ".toml":
        try:
            tomllib.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            failures.append(f"invalid TOML: {exc}")

    return failures


def audit_required_metadata(paths: set[Path]) -> list[str]:
    failures: list[str] = []
    for rel_text in REQUIRED_FILES:
        path = ROOT / rel_text
        if path not in paths or not path.is_file():
            failures.append(f"{rel_text}: required release file is missing")

    for rel_text, markers in REQUIRED_MARKERS.items():
        path = ROOT / rel_text
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"{rel_text}: required metadata is not UTF-8")
            continue
        normalized_text = " ".join(text.split())
        for marker in markers:
            if " ".join(marker.split()) not in normalized_text:
                failures.append(f"{rel_text}: missing required public metadata marker")
    return failures


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_records(
    manifest: Mapping[str, object],
) -> tuple[list[Mapping[str, object]], list[str]]:
    records: list[Mapping[str, object]] = []
    failures: list[str] = []
    top_files = manifest.get("files", [])
    if not isinstance(top_files, list):
        failures.append("artifacts/manifest.json: files must be a list")
    else:
        records.extend(item for item in top_files if isinstance(item, Mapping))
        if len(records) != len(top_files):
            failures.append("artifacts/manifest.json: every file record must be an object")

    experiments = manifest.get("experiments", {})
    if isinstance(experiments, Mapping):
        experiment_values = list(experiments.values())
    elif isinstance(experiments, list):
        experiment_values = experiments
    else:
        failures.append("artifacts/manifest.json: experiments must be an object or list")
        experiment_values = []
    for experiment in experiment_values:
        if not isinstance(experiment, Mapping):
            failures.append("artifacts/manifest.json: every experiment must be an object")
            continue
        files = experiment.get("files", [])
        if not isinstance(files, list):
            failures.append("artifacts/manifest.json: experiment files must be a list")
            continue
        for record in files:
            if isinstance(record, Mapping):
                records.append(record)
            else:
                failures.append("artifacts/manifest.json: every file record must be an object")
    return records, failures


def audit_artifact_manifest(release_set: set[Path]) -> list[str]:
    manifest_path = ROOT / "artifacts/manifest.json"
    if not manifest_path.is_file():
        return []
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []  # The generic content audit reports the parse error.
    if not isinstance(manifest, Mapping):
        return ["artifacts/manifest.json: root must be an object"]

    failures: list[str] = []
    if manifest.get("schema") != "sr_filter.artifact_manifest.v1":
        failures.append("artifacts/manifest.json: unexpected or missing schema")
    records, record_failures = manifest_records(manifest)
    failures.extend(record_failures)
    seen: set[str] = set()
    local_records: set[Path] = set()

    for index, record in enumerate(records):
        prefix = f"artifacts/manifest.json: record {index}"
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            failures.append(f"{prefix} has no path")
            continue
        pure_path = PurePosixPath(raw_path)
        if pure_path.is_absolute() or ".." in pure_path.parts or "\\" in raw_path:
            failures.append(f"{prefix} has an unsafe path")
            continue
        if raw_path in seen:
            failures.append(f"{prefix} duplicates path {raw_path!r}")
            continue
        seen.add(raw_path)

        expected_sha = record.get("sha256")
        if not isinstance(expected_sha, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
            failures.append(f"{prefix} has an invalid SHA-256 digest")
        expected_bytes = record.get("bytes")
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 0
        ):
            failures.append(f"{prefix} has an invalid byte count")

        relative = Path(*pure_path.parts)
        candidates = (ROOT / relative, ROOT / "artifacts" / relative)
        local_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        record_url = record.get("url") or record.get("base_url") or manifest.get("base_url")
        if local_path is not None:
            local_records.add(local_path)
            if isinstance(expected_bytes, int) and local_path.stat().st_size != expected_bytes:
                failures.append(f"{prefix} byte count does not match {display_path(local_path)}")
            if isinstance(expected_sha, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha):
                if sha256_file(local_path) != expected_sha:
                    failures.append(f"{prefix} digest does not match {display_path(local_path)}")
        elif not record_url:
            failures.append(f"{prefix} is neither checked in nor assigned a remote URL")

        if record_url:
            parsed = urllib.parse.urlparse(str(record_url))
            if parsed.scheme != "https":
                failures.append(f"{prefix} remote URL is not HTTPS")
            if parsed.hostname == "huggingface.co":
                if re.search(r"/resolve/(?:main|refs/heads/[^/]+)(?:/|$)", parsed.path):
                    failures.append(f"{prefix} uses a moving Hugging Face revision")
                elif "/resolve/" not in parsed.path:
                    failures.append(f"{prefix} does not pin a Hugging Face revision")

    for path in release_set:
        if path.suffix == ".pkl" and path.is_file() and path not in local_records:
            failures.append(f"{display_path(path)}: shipped pickle is not covered by the manifest")
    return failures


def audit_companion_dataset_release() -> list[str]:
    path = ROOT / "artifacts/hf_dataset_release.json"
    if not path.is_file():
        return []  # Required-file audit reports this separately.
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []  # Generic content audit reports malformed JSON.
    if not isinstance(payload, Mapping):
        return ["artifacts/hf_dataset_release.json: root must be an object"]
    failures: list[str] = []
    if payload.get("schema") != "sr_filter.hf_dataset_release.v1":
        failures.append("artifacts/hf_dataset_release.json: unexpected schema")
        return failures
    if payload.get("status") != "verified_pinned":
        failures.append("artifacts/hf_dataset_release.json: companion dataset is not verified and pinned")
        return failures
    revision = str(payload.get("data_revision", ""))
    code_revision = str(payload.get("code_revision", ""))
    if re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", revision) is None:
        failures.append("artifacts/hf_dataset_release.json: data revision is not immutable")
    if re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", code_revision) is None:
        failures.append("artifacts/hf_dataset_release.json: code revision is not immutable")
    records = payload.get("files")
    if not isinstance(records, list) or not records:
        failures.append("artifacts/hf_dataset_release.json: pinned file inventory is empty")
        return failures
    for record in records:
        if not isinstance(record, Mapping):
            failures.append("artifacts/hf_dataset_release.json: file record is not an object")
            continue
        url = str(record.get("url", ""))
        if f"/resolve/{revision}/" not in url or not url.startswith(
            "https://huggingface.co/datasets/takh04/sr-filter-artifacts/"
        ):
            failures.append("artifacts/hf_dataset_release.json: file URL is not pinned to its data revision")
            break
    verification = payload.get("verification")
    if not isinstance(verification, Mapping) or verification.get("readback_verified") is not True:
        failures.append("artifacts/hf_dataset_release.json: authenticated readback is not verified")
    return failures


def load_toml(path: Path) -> Mapping[str, object] | None:
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def audit_online_protocol(path: Path, expected_seed: int) -> list[str]:
    payload = load_toml(path)
    if payload is None:
        return []  # Missing/invalid TOML is reported by the generic audits.
    rel_text = display_path(path)
    failures: list[str] = []
    if payload.get("branch_rng_seed") != expected_seed:
        failures.append(f"{rel_text}: unexpected branch seed")
    status_text = str(payload.get("artifact_status", "")).lower()
    if not status_text or any(
        token in status_text for token in ("pending", "partial", "gap", "incomplete")
    ):
        failures.append(f"{rel_text}: online artifact status is not release-complete")
    if any(
        token in str(payload.get("experiment_id", "")).lower()
        for token in ("pending", "partial", "incomplete")
    ):
        failures.append(f"{rel_text}: experiment identifier marks a partial run")

    sampling = payload.get("sampling")
    if not isinstance(sampling, Mapping) or sampling.get("samples_per_batch") != 8192:
        failures.append(f"{rel_text}: expected 8192 samples per batch")
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return [*failures, f"{rel_text}: runs must be a list"]
    by_arm = {
        str(run.get("arm")): run
        for run in runs
        if isinstance(run, Mapping) and run.get("arm") is not None
    }
    expected = {
        "sr": (500, 1),
        "spring": (500, 1),
        "ms_sr": (125, 4),
    }
    if set(by_arm) != set(expected):
        failures.append(f"{rel_text}: expected exactly SR, SPRING, and MS-SR arms")
        return failures
    for arm, (steps, batches) in expected.items():
        run = by_arm[arm]
        if run.get("optimizer_steps") != steps or run.get("batches_per_update") != batches:
            failures.append(f"{rel_text}: {arm} does not implement the 500-unit budget")
        if arm == "sr" and run.get("status") == "reused_validated":
            validated = payload.get("validated_sr")
            checkpoint_sha = (
                validated.get("checkpoint_sha256")
                if isinstance(validated, Mapping)
                else None
            )
            validation_command = run.get("validation_command")
            if (
                not isinstance(validated, Mapping)
                or validated.get("status") != "complete"
                or validated.get("optimizer_steps") != steps
                or validated.get("batches_per_update") != batches
                or re.fullmatch(r"[0-9a-f]{64}", str(checkpoint_sha or "")) is None
                or not isinstance(validation_command, str)
                or str(checkpoint_sha) not in validation_command
                or "sha256sum -c" not in validation_command
            ):
                failures.append(
                    f"{rel_text}: reused SR arm lacks a complete digest-validated record"
                )
            continue
        command = run.get("command")
        required_fragments = (
            f"--schedule-total-steps {steps}",
            f"--phase-lengths {steps}",
            "--n-samples 8192",
            f"--seed {expected_seed}",
        )
        if not isinstance(command, str) or any(
            fragment not in command for fragment in required_fragments
        ):
            failures.append(f"{rel_text}: {arm} command disagrees with its protocol")
        if arm == "ms_sr" and isinstance(command, str):
            exact_fragments = (
                "--sr-update-protocol fixed_grid_exact_loo",
                "--sr-variant indep_batch",
                "--sr-weight-scheme stacking",
                "--sr-lambda-grid 1e-2,1e-3,1e-4,1e-5",
            )
            if any(fragment not in command for fragment in exact_fragments):
                failures.append(
                    f"{rel_text}: MS-SR command does not select the exact K-batch LOO driver"
                )
    return failures


def audit_endpoint_protocol(path: Path, expected_seed: int) -> list[str]:
    payload = load_toml(path)
    if payload is None:
        return []
    rel_text = display_path(path)
    failures: list[str] = []
    if payload.get("branch_rng_seed") != expected_seed:
        failures.append(f"{rel_text}: unexpected branch seed")
    status_text = str(payload.get("artifact_status", "")).lower()
    if not status_text or any(
        token in status_text for token in ("pending", "partial", "gap", "incomplete")
    ):
        failures.append(f"{rel_text}: endpoint artifact status is not release-complete")
    expected_scalars = {
        "rounds": 512,
        "burn_in_rounds": 5,
        "samples_per_round": 8192,
        "expected_runs_per_arm": 2,
        "expected_total_runs": 6,
        "max_replica_difference_z": 5.0,
    }
    for name, expected_value in expected_scalars.items():
        if payload.get(name) != expected_value:
            failures.append(f"{rel_text}: {name} must be {expected_value}")
    if payload.get("replica_seeds") != [12345, 98765]:
        failures.append(f"{rel_text}: unexpected endpoint replica seeds")
    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 3:
        failures.append(f"{rel_text}: expected three final checkpoint records")
    elif any(
        not isinstance(record, Mapping)
        or re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))) is None
        for record in checkpoints
    ):
        failures.append(f"{rel_text}: every endpoint checkpoint needs a SHA-256 digest")
    return failures


def audit_experiment_protocols() -> list[str]:
    config_root = ROOT / "configs/experiments"
    failures: list[str] = []
    for filename, seed in (
        ("online_j1j2_branch2500_seed0.toml", 0),
        ("online_j1j2_branch1800_seed1.toml", 1),
    ):
        path = config_root / filename
        if path.is_file():
            failures.extend(audit_online_protocol(path, seed))
    for filename, seed in (
        ("endpoint_eval_j1j2_seed0.toml", 0),
        ("endpoint_eval_j1j2_seed1.toml", 1),
    ):
        path = config_root / filename
        if path.is_file():
            failures.extend(audit_endpoint_protocol(path, seed))
    return failures


def main() -> int:
    failures: list[str] = []
    paths = release_paths()
    path_set = set(paths)
    total_size = 0
    collapsed_roots_seen: set[str] = set()

    failures.extend(audit_required_metadata(path_set))
    failures.extend(audit_artifact_manifest(path_set))
    failures.extend(audit_companion_dataset_release())
    failures.extend(audit_experiment_protocols())
    for path in paths:
        rel_text = display_path(path)
        try:
            root_part = path.relative_to(ROOT).parts[0]
        except (ValueError, IndexError):
            root_part = ""
        if root_part in COLLAPSED_FORBIDDEN_ROOTS and root_part not in collapsed_roots_seen:
            failures.append(f"{root_part}/: {COLLAPSED_FORBIDDEN_ROOTS[root_part]}")
            collapsed_roots_seen.add(root_part)
        try:
            path_failures, size = audit_path(path)
        except ValueError:
            path_failures, size = ["path is outside the repository"], 0
        total_size += size
        failures.extend(f"{rel_text}: {failure}" for failure in path_failures)
        if path.is_file() and not path.is_symlink() and size <= MAX_FILE_BYTES:
            failures.extend(f"{rel_text}: {failure}" for failure in audit_content(path))

    if total_size > MAX_RELEASE_BYTES:
        failures.append(
            f"release set is {total_size} bytes; limit is {MAX_RELEASE_BYTES}"
        )

    if failures:
        print(f"Release audit failed ({len(failures)} issue(s)):")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"Release audit passed ({len(paths)} files, {total_size} bytes checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
