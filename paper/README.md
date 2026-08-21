# Manuscript build

This directory is the canonical camera-ready source. It intentionally contains
no copied LaTeX auxiliary files from earlier builds.

Requirements:

- a TeX Live installation containing `latexmk`, `pdflatex`, `bibtex`, and the
  packages imported by `main.tex`;
- Python 3 with NumPy and Matplotlib for figure regeneration.

From this directory:

```bash
make pdf
make check
```

The build exports a fixed `SOURCE_DATE_EPOCH` (2026-01-01 00:00:00 UTC) and
`FORCE_SOURCE_DATE=1`. Both Matplotlib generators write explicit creation and
modification metadata from that epoch, and `latexmk` inherits the same values;
the main source also suppresses pdfTeX's variable trailer ID and implementation
provenance records. Set `SOURCE_DATE_EPOCH` on the command line to choose a
different reproducible timestamp.

`make pdf` regenerates Figure 4 directly from the curated checkpoint-level
ablation CSV and Figure 5 from the audited online-training artifacts, then runs
`latexmk`. Figure 5 is a
strict two-branch build gate. Its inputs are declared in
`online_artifacts.json` and must exist under:

```text
../artifacts/online_training/branch_seed0/
../artifacts/online_training/branch_seed1/
```

Each branch must supply `status.json`, `provenance.json`, `training_curve.csv`,
and `endpoint_evaluations.csv`. The generator validates completion, arm coverage,
equal-budget accounting, batch counts, endpoint audit status, and uncertainty
fields. In particular, every arm and branch must have two distinct audited
512-round endpoint replicas with exactly five burn-in rounds (4,194,304 samples
each); the legacy 128-round diagnostic does not satisfy the gate. Curves are plotted against 0--500 budget
units since the branch, with faint raw traces and a common 25-budget-unit
rolling mean after interpolation to a one-unit grid. Endpoint markers at budget
500 use the artifact's conservative one-standard-error field. Only then does
the generator write:

- `figures/figure5_equal_budget_online.pdf`;
- `generated/online_comparison_results.tex`;
- `generated/online_protocol_table.tex`;
- `generated/online_artifact_status.json` with input checksums.

`make check` additionally rejects anonymity markers, unfinished checklist
tokens, stale Figure 5 references, missing generated inputs, unresolved LaTeX
references/citations, and main content ending after page 10. The page boundary
is the page immediately preceding the `main:references:start` label; a
`\\clearpage` flushes all main-text floats before that label is written.
