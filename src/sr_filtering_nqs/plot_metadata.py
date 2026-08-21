"""Deterministic metadata for reproducible Matplotlib PDF artifacts."""

from __future__ import annotations

import os
from datetime import datetime, timezone

DEFAULT_SOURCE_DATE_EPOCH = 1767225600  # 2026-01-01 00:00:00 UTC.


def pdf_metadata(title: str) -> dict[str, object]:
    """Return stable PDF metadata controlled by ``SOURCE_DATE_EPOCH``."""

    raw_epoch = os.environ.get("SOURCE_DATE_EPOCH", str(DEFAULT_SOURCE_DATE_EPOCH))
    try:
        timestamp = datetime.fromtimestamp(int(raw_epoch), tz=timezone.utc)
    except (ValueError, OverflowError, OSError) as error:
        raise ValueError(f"invalid SOURCE_DATE_EPOCH={raw_epoch!r}") from error
    return {
        "Title": title,
        "Author": "Tak Hur",
        "Creator": "sr-filter reproducibility CLI",
        "Producer": "Matplotlib PDF backend",
        "CreationDate": timestamp,
        "ModDate": timestamp,
    }
