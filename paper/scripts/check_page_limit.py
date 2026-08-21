#!/usr/bin/env python3
"""Fail when the labeled main-content boundary exceeds the page limit."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def label_page(aux_text: str, label: str) -> int:
    pattern = re.compile(
        rf"\\newlabel\{{{re.escape(label)}\}}\{{\{{[^}}]*\}}\{{(\d+)\}}"
    )
    match = pattern.search(aux_text)
    if not match:
        raise SystemExit(f"page-limit check: label {label!r} not found in aux file")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aux", type=Path)
    parser.add_argument("--max-pages", type=int, default=10)
    args = parser.parse_args()
    if not args.aux.is_file():
        raise SystemExit(f"page-limit check: missing {args.aux}; build the PDF first")
    aux_text = args.aux.read_text(encoding="utf-8", errors="replace")
    references_page = label_page(aux_text, "main:references:start")
    main_page = references_page - 1
    appendix_page = label_page(aux_text, "app:start")
    if main_page > args.max_pages:
        raise SystemExit(
            f"page-limit check: FAILED: main content ends on page {main_page}, "
            f"limit is {args.max_pages}"
        )
    print(
        f"page-limit check: PASS: main content ends on page {main_page}/{args.max_pages}; "
        f"references start on page {references_page}; appendix starts on page {appendix_page}"
    )


if __name__ == "__main__":
    main()
