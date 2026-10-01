#!/usr/bin/env python3
"""Verify the compact paper bundle and exclude generated/private research data."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts" / "paper"


def audit() -> list[str]:
    failures: list[str] = []
    manifest_path = BUNDLE / "manifest.json"
    if not manifest_path.is_file():
        return ["missing artifacts/paper/manifest.json"]
    manifest = json.loads(manifest_path.read_text())
    records = manifest.get("files", [])
    if not records:
        return ["paper data manifest is empty"]
    declared: set[str] = set()
    for record in records:
        name = record["path"]
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            failures.append(f"unsafe artifact path: {name}")
            continue
        if name in declared:
            failures.append(f"duplicate artifact: {name}")
        declared.add(name)
        path = BUNDLE / name
        if not path.is_file() or path.is_symlink():
            failures.append(f"missing or linked artifact: {name}")
            continue
        data = path.read_bytes()
        if len(data) != record["bytes"]:
            failures.append(f"artifact size mismatch: {name}")
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            failures.append(f"artifact checksum mismatch: {name}")
    actual = {p.relative_to(BUNDLE).as_posix() for p in BUNDLE.rglob("*") if p.is_file()}
    # The manifest describes data and provenance; README is ordinary documentation.
    extra = actual - declared - {"manifest.json", "README.md"}
    failures.extend(f"unlisted bundled file: {name}" for name in sorted(extra))
    other_artifacts = [p.name for p in (ROOT / "artifacts").iterdir() if p.name != "paper"]
    failures.extend(f"artifact outside the paper bundle: {name}" for name in other_artifacts)
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=False, capture_output=True,
    )
    if result.returncode == 0:
        paths = [ROOT / p for p in result.stdout.decode().split("\0") if p]
    else:
        paths = [p for folder in ("src", "artifacts", "paper", "scripts")
                 for p in (ROOT / folder).rglob("*") if p.is_file()]
    forbidden = {".pkl", ".pickle", ".mpack", ".ckpt", ".msgpack"}
    for path in paths:
        if not path.exists():  # staged or working-tree deletion
            continue
        rel = path.relative_to(ROOT)
        if path.is_symlink():
            failures.append(f"symlink in release: {rel}")
        if path.suffix in forbidden or "checkpoints" in rel.parts:
            failures.append(f"checkpoint/pickle in release: {rel}")
        if path.stat().st_size > 25 * 1024 * 1024:
            failures.append(f"unexpected large file in release: {rel}")
    return failures


def main() -> int:
    failures = audit()
    if failures:
        print("\n".join(f"ERROR: {failure}" for failure in failures))
        return 1
    print("Paper artifact checksums and repository scope verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
