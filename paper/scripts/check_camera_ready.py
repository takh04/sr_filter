#!/usr/bin/env python3
"""Static and build-artifact checks for the camera-ready manuscript."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PAPER_ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    PAPER_ROOT / "main.tex",
    PAPER_ROOT / "appendix.tex",
    PAPER_ROOT / "checklist.tex",
    PAPER_ROOT / "references.bib",
)
REQUIRED_FILES = (
    PAPER_ROOT / "literature_ratio.csv",
    PAPER_ROOT / "literature_ratio.md",
    PAPER_ROOT / "figures" / "p_over_ns_literature.pdf",
    PAPER_ROOT / "figures" / "figure2.pdf",
    PAPER_ROOT / "figures" / "figure3_tfim_fixed_data.pdf",
    PAPER_ROOT / "figures" / "figure4_mssr_ablation_oracle.pdf",
    PAPER_ROOT / "figures" / "figure5_equal_budget_online.pdf",
    PAPER_ROOT / "generated" / "online_comparison_results.tex",
    PAPER_ROOT / "generated" / "online_protocol_table.tex",
    PAPER_ROOT / "generated" / "online_artifact_status.json",
)
FORBIDDEN = {
    "anonymous author": re.compile(r"anonymous\s+author", re.IGNORECASE),
    "anonymous repository": re.compile(r"AnonNeurips|AnonNeurips2026", re.IGNORECASE),
    "checklist TODO macro": re.compile(r"\\(?:answer|justification)TODO"),
    "TODO/FIXME marker": re.compile(r"\b(?:TODO|FIXME)\b"),
    "figure placeholder": re.compile(r"figureplaceholder", re.IGNORECASE),
    "removed Figure 5 asset": re.compile(r"figure5_srel_logratio|mssr_update_tracking"),
    "legacy rebuttal artifact path": re.compile(r"rebuttal" + r"_assets", re.IGNORECASE),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    failures: list[str] = []
    texts: dict[Path, str] = {}
    for path in SOURCES:
        if not path.is_file():
            failures.append(f"missing source: {path.relative_to(PAPER_ROOT)}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        texts[path] = text
        for description, pattern in FORBIDDEN.items():
            if pattern.search(text):
                failures.append(f"{path.name}: contains {description}")

    main_text = texts.get(PAPER_ROOT / "main.tex", "")
    appendix_text = texts.get(PAPER_ROOT / "appendix.tex", "")
    checklist_text = texts.get(PAPER_ROOT / "checklist.tex", "")
    bibliography_text = texts.get(PAPER_ROOT / "references.bib", "")
    generated_results_path = PAPER_ROOT / "generated" / "online_comparison_results.tex"
    generated_results_text = (
        generated_results_path.read_text(encoding="utf-8", errors="replace")
        if generated_results_path.is_file()
        else ""
    )
    requirements = {
        "final NeurIPS style": r"\usepackage[final,main]{neurips_2026}" in main_text,
        "author metadata": "Tak Hur" in main_text and "Yonsei University" in main_text,
        "public repository URL": "https://github.com/takh04/sr_filter" in appendix_text
        and "https://github.com/takh04/sr_filter" in checklist_text,
        "new Figure 5 label": "fig:mssr_equal_budget_online"
        in (main_text + generated_results_text),
        "Figure 2 sample-error contract": all(
            token in main_text
            for token in (
                "100$ matched-noise",
                "10$ matched-state",
                "one sample standard error",
                r"0.53\pm0.008",
                r"3.39\pm0.073",
                r"2.53\pm0.18",
                r"1.59\pm0.079",
            )
        ),
        "Figure 3 draw/error contract": all(
            token in (main_text + appendix_text)
            for token in (
                "$m=20$ independent",
                "single-shift diagnostic banks use $m=20$",
                "one standard error across the ten checkpoint-level means",
            )
        ),
        "limitations": "sec:limitations" in main_text,
        "published Gravina metadata": re.search(
            r"@article\{gravina2410neural,.*?year=\{2025\}.*?doi=\{10\.22331/q-2025-07-22-1803\}",
            bibliography_text,
            re.DOTALL,
        ) is not None,
        "published Viteritti metadata": re.search(
            r"@article\{viteritti2024transformer,.*?volume=\{111\}.*?pages=\{134411\}.*?year=\{2025\}.*?doi=\{10\.1103/PhysRevB\.111\.134411\}",
            bibliography_text,
            re.DOTALL,
        ) is not None,
        "PsiFormer ICLR metadata": re.search(
            r"@inproceedings\{vonglehn2023psiformer,.*?booktitle=\{The Eleventh International Conference on Learning Representations\}.*?openreview\.net/forum\?id=xveTeHVlF7j",
            bibliography_text,
            re.DOTALL,
        ) is not None,
    }
    failures.extend(f"missing requirement: {name}" for name, present in requirements.items() if not present)

    for path in REQUIRED_FILES:
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing generated input: {path.relative_to(PAPER_ROOT)}")

    status_path = PAPER_ROOT / "generated" / "online_artifact_status.json"
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            failures.append("online_artifact_status.json is invalid JSON")
        else:
            if status.get("ready") is not True or len(status.get("validated_branches", [])) != 2:
                failures.append("online artifact status is not ready for two branches")
            repo_root = PAPER_ROOT.parent
            manifest_path = PAPER_ROOT / "online_artifacts.json"
            if status.get("manifest_sha256") != sha256(manifest_path):
                failures.append("online artifact status has a stale manifest checksum")
            for entry in status.get("inputs", []):
                input_path = repo_root / str(entry.get("path", ""))
                if not input_path.is_file():
                    failures.append(f"online status input is missing: {input_path}")
                elif entry.get("sha256") != sha256(input_path):
                    failures.append(f"online status input checksum is stale: {input_path}")

    log_path = PAPER_ROOT / "main.log"
    if log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        log_failures = (
            "There were undefined references",
            "Citation `",
            "multiply defined",
            "! LaTeX Error",
        )
        failures.extend(f"main.log contains {token!r}" for token in log_failures if token in log_text)

    if failures:
        rendered = "\n".join(f"  - {failure}" for failure in failures)
        raise SystemExit(f"camera-ready check: FAILED\n{rendered}")
    print("camera-ready check: PASS")


if __name__ == "__main__":
    main()
