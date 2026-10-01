# SR Filter

Code and compact results for **Stochastic Reconfiguration as Statistical Filtering
for Overparameterized Neural Quantum States**, by Tak Hur.

[arXiv paper](paper/arxiv/2609.23334v1.pdf) · [NeurIPS version](paper/main.pdf) ·
[Experiment guide](docs/reproducibility.md) ·
[Data inventory](docs/artifacts.md)

Both manuscript versions use the text and figures submitted as
[arXiv:2609.23334v1](https://arxiv.org/abs/2609.23334).
The [submitted sources](paper/arxiv/README.md) are preserved with checksums.

The repository contains the paper experiments: exact small-system regression
diagnostics, TFIM checkpoint diagnostics, MS-SR ablations, and the J1–J2 online
SR/MS-SR comparison. It also includes the optimized SR, bagged-SR, and MS-SR timing
implementation and the appendix's exact population-reference tracking code.
The numerical code uses JAX and NetKet.

## Install

Use Python 3.11 or 3.12. From the repository root:

```bash
uv sync --frozen --extra dev
uv run sr-filter --help
```

The default environment supports CPU plotting and tests. For CUDA 12 experiments
on Linux, use `uv sync --frozen --extra dev --extra cuda12`.

## Reproduce the figures

All five main-text figures can be regenerated from the bundled CSV, JSON, and
NumPy arrays without checkpoints or external downloads:

```bash
uv run sr-filter reproduce --input-root . --output-dir outputs/figures
```

Use `--recipe all` to also regenerate both appendix figures.

| Paper figure | Included data | Experiment implementation |
| --- | --- | --- |
| 1: parameter-to-sample ratios | Literature points and source notes | `literature_ratio.py` |
| 2: small-system diagnostics | Matched-noise and matched-state diagnostic arrays | `small_scale/`, `matched_state/` |
| 3: TFIM shift sweep | Checkpoint-level residuals and variances | `large_scale/core/fixed_data_protocol.py` |
| 4: MS-SR ablations | Checkpoint summaries and bagged-shift sweep | `fixed_data_protocol.py`, `bagged_lambda_protocol.py` |
| 5: online training and cost | Five J1–J2 training pairs, endpoint estimates, TFIM timing summaries | `large_scale/cli/train_sweep.py`, `large_scale/core/optimized_sr_benchmark.py` |

Implementation paths in this table are relative to `src/sr_filtering_nqs/`.
The [experiment guide](docs/reproducibility.md) maps the paper protocols to the
training and diagnostic commands. Running new training or checkpoint diagnostics
requires the corresponding model states and substantially more compute than
plotting the bundled results.

The repository preserves the available results used by the paper, including
appendix robustness diagnostics, raw timing records, and online-tracking
trajectories. [Archive coverage and known gaps](docs/archive-coverage.md) maps
them to the manuscript. The tracking source archive has missing, incomplete,
and conflicting records; all 96 published figure points are retained, and 81
can be independently recovered from the available trajectories. Model weights
and raw sample banks are outside this results bundle.

## Build the manuscript

Install Tectonic, or latexmk with a TeX Live installation, then run:

```bash
make -C paper camera-ready
```

`make -C paper sync` restores the NeurIPS manuscript from the submitted arXiv
sources. `make -C paper reproduce` generates numerical reproductions under
`outputs/figures/`, preserving the original submitted figure assets.

## Check the code and data

```bash
JAX_PLATFORMS=cpu uv run pytest
uv run python scripts/audit_release.py
uv build
```

See [CITATION.cff](CITATION.cff) for citation metadata. Code is licensed under
[Apache-2.0](LICENSE); bundled support-code attribution is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
