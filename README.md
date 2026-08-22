# SR Filtering for Neural Quantum States

Reference implementation and reproducibility package for **“Stochastic
Reconfiguration as Statistical Filtering for Overparameterized Neural Quantum
States.”**

The project studies stochastic reconfiguration (SR) through the lens of
statistical filtering and includes the checkpoint-local diagnostics,
multi-shift SR (MS-SR), online SR/SPRING/MS-SR comparisons, robustness
ablations, and wall-clock measurements used in the paper.

## Install

The reference environment uses Python 3.12, JAX 0.7.2, and NetKet 3.21.0.
Python 3.11 and 3.12 are tested. The default install is CPU-compatible:

```bash
git clone https://github.com/takh04/sr_filter.git
cd sr_filter
uv sync --frozen --extra dev
uv run sr-filter --help
uv run pytest
```

For NVIDIA CUDA 12 systems:

```bash
uv sync --frozen --extra dev --extra cuda12
uv run python -c "import jax; print(jax.devices())"
```

CUDA drivers and libraries are host requirements and are not installed by
this repository. The large online runs were measured on NVIDIA H100 80 GB
GPUs; CPU installation is intended for tests, artifact inspection, and figure
regeneration.

## Reproduce the paper artifacts

Compact, path-free inputs needed for plotting are versioned in `artifacts/`.
Most are CSV/JSON; two legacy small-scale inputs use Python pickle and are
called out in [`docs/artifacts.md`](docs/artifacts.md). Generated files default
to `./outputs` so an installed wheel never writes into site-packages.

```bash
uv run sr-filter reproduce --recipe curated --input-root . --output-dir outputs/figures
```

`--recipe online` regenerates the equal-budget comparison only after both
audited branch bundles are present; `--recipe all` requires both compact and
online inputs.

Online fixed-grid MS-SR uses the explicit trainer option
`--sr-update-protocol fixed_grid_exact_loo`; the frozen commands and exact
four-batch/four-solve budget are documented in
[`docs/reproducibility.md`](docs/reproducibility.md).

The checked-in compact artifact bundle is self-contained. Large checkpoints
and raw per-round records are intended for the companion Hugging Face dataset
[`takh04/sr-filter-artifacts`](https://huggingface.co/datasets/takh04/sr-filter-artifacts)
and are not stored in Git. Before a release advertises those files, the
dataset must be public and its immutable revision, byte counts, and SHA-256
digests must be added to the checked-in manifest. Fetch only manifest-listed
files:

```bash
uv run sr-filter artifact fetch --output-root outputs/artifacts
```

Exact experiment configurations are under `configs/`. GPU jobs write to an
explicit output directory and never depend on dated cluster paths. See
[`docs/reproducibility.md`](docs/reproducibility.md) for commands, expected
resources, and the figure-to-artifact map.

## Repository layout

- `src/sr_filtering_nqs/`: maintained package and command-line interfaces.
- `src/{advanced_drivers,netket_foundational,nqs_nets,nqs_support_core,spin_vmc}/`:
  bundled support code required by the experiments; these namespaces are
  internal and carry upstream provenance in `THIRD_PARTY_NOTICES.md`.
- `configs/`: portable online, diagnostic, endpoint, and timing configurations.
- `artifacts/`: compact sanitized data, manifests, and publication figures.
- `paper/`: camera-ready manuscript source, the built
  [`main.pdf`](paper/main.pdf), and reproducible build checks.
- `tests/`: CPU unit, schema, plotting, and package-install tests.

## Reproducibility boundaries

Completed, auditable camera-ready experiments are included. Interrupted
sweeps, debug launchers, machine-local logs, and an incomplete exploratory
sweep are intentionally excluded and recorded in the provenance manifest.
Online comparisons are controlled continuations from two checkpoints, not a
broad population-level ranking of optimizers.

Some archived research checkpoints use Python pickle. **Never load an
untrusted pickle.** The artifact fetch command accepts only manifest-listed
files, checks their digest, and never deserializes them. Commands that accept
a checkpoint or `.pkl` path treat it as trusted input and may deserialize it
without another prompt. A matching digest proves file identity, not safety.

## Citation and license

Please cite the paper using [`CITATION.cff`](CITATION.cff). Original project
code and released artifacts are licensed under Apache-2.0 unless a file or
third-party notice states otherwise. See [`LICENSE`](LICENSE) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Security and disclosure guidance is in [`SECURITY.md`](SECURITY.md).
Maintainer publication gates are listed in
[`docs/release-checklist.md`](docs/release-checklist.md).
