# SR Filter

Code and compact results for **Stochastic Reconfiguration as Statistical Filtering
for Overparameterized Neural Quantum States**, by Tak Hur.

[Read the paper on arXiv](https://arxiv.org/abs/2609.23334).

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
Use `uv run sr-filter COMMAND --help` for training, checkpoint diagnostics,
endpoint evaluation, and timing options. Running new training or checkpoint
diagnostics requires the corresponding model states and substantially more
compute than plotting the bundled results.

The repository preserves the available results used by the paper, including
appendix robustness diagnostics, raw timing records, and online-tracking
trajectories. The [data inventory](artifacts/paper/README.md) describes their
coverage and conventions. The tracking source archive has missing, incomplete,
and conflicting records; all 96 published figure points are retained, and 81
can be independently recovered from the available trajectories. Model weights
and raw sample banks are outside this results bundle.

## Check the code and data

```bash
JAX_PLATFORMS=cpu uv run pytest
uv run python scripts/audit_release.py
uv build
```

## Citation

```bibtex
@article{hur2026stochastic,
  title={Stochastic Reconfiguration as Statistical Filtering for Overparameterized Neural Quantum States},
  author={Hur, Tak},
  journal={arXiv preprint arXiv:2609.23334},
  year={2026}
}
```

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.
Code is licensed under
[Apache-2.0](LICENSE); bundled support-code attribution is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
