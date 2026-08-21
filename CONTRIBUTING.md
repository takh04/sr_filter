# Contributing

Issues and focused pull requests are welcome. For scientific changes, describe
the protocol, random seeds, hardware, dependency lock, and expected numerical
tolerance. Do not commit raw checkpoints, cluster logs, private paths, or
machine-specific launch scripts.

Set up the CPU development environment and run the release checks:

```bash
uv sync --frozen --extra dev
uv run pytest
uv run python -m compileall -q src scripts
uv run python scripts/audit_release.py
```

Generated outputs belong under `outputs/` and are ignored. Compact artifacts
must use the documented long-form CSV/JSON schemas, include provenance, and be
regenerable by a checked-in command. Add or update tests whenever behavior or
an artifact schema changes.

By submitting a contribution, you agree that it is licensed under Apache-2.0
and that you have the right to contribute it. Preserve upstream copyright and
license headers.
