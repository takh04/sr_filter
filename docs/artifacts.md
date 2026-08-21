# Artifact contract

## Tiers

- **Git:** sanitized long-form CSV/JSON, compact summaries, plotting inputs,
  publication figures, configuration files, and manifests.
- **Companion dataset (when published):** audited source/final checkpoints,
  per-round endpoint evaluations, and raw timing records.
- **Private archive:** interrupted runs, exploratory sweeps, delta banks,
  machine-local logs, and cluster diagnostics.

New long-form public experiment records identify their experiment, system,
method, lambda policy, K, source checkpoint/step, branch seed where
applicable, repeat/round, metric, value, units, and source code revision.
Compact legacy summaries may retain a `method_id`; the curation process records
its normalized method and explicit K in the accompanying provenance records.

`artifacts/manifest.json` uses schema `sr_filter.artifact_manifest.v1`. Each
file has a safe relative POSIX path, byte count, and SHA-256 digest. Checked-in
files do not need a remote URL. Any future external record must use an
immutable Hugging Face revision; moving branch names such as `main` are not
release identifiers.

`sr-filter artifact fetch` rejects non-HTTPS remote locations and unsafe or
non-portable paths before writing. It also caps manifests at 8 MiB, selected
records at 10,000, individual downloads at 512 MiB, and the selected total at
2 GiB by default. The last two limits can be lowered or deliberately raised
with `--max-file-bytes` and `--max-total-bytes`; inspect the manifest before
raising them for a third-party source.

## Statistical fields

Summary records must identify the aggregation unit and distinguish standard
deviation from standard error. Figure 2 reports one sample standard error of
the exact-population excess-risk estimate across independently resampled
training batches: 100 trials for matched noise and 10 for matched state.
Figure 3 uses 20 independent update draws per shift and checkpoint; its plotted
points aggregate the ten fixed checkpoint-level diagnostic means, and its error
bars are their population standard deviation divided by square root 10.
Figure 4 uses the same checkpoint-level standard-error convention. Online
endpoint errors are Monte Carlo standard errors calculated from persisted
retained-round values. The Shastry–Sutherland negative result reports
descriptive mean and standard deviation across checkpoints.

## Provenance and exclusions

`artifacts/manifest.json` records paths, sizes, SHA-256 digests, schema
versions, and provenance. `artifacts/exclusions.json` records examined but
unpublished experiment outputs and the reason for exclusion. An artifact is
not publication-ready if it contains an absolute private path, an unrecorded
selection step, or a statistic that cannot be regenerated from released data.

## Pickle safety

The two legacy files below are Python pickles retained for the small-scale
plots:

- `artifacts/small_scale/matched_noise/rbm490_vit10.pkl`
- `artifacts/small_scale/matched_state/alternating_polish_vit_step0100_to_direct_rbm_a2_figure1_style.pkl`

Python pickle is an executable serialization format. Only load these exact
project files after checking their manifest digests, and never substitute a
file from an issue, pull request, shared drive, or untrusted URL. Digest
verification proves which bytes you have; it does not make arbitrary pickle
bytes safe. The fetch command downloads and verifies files without
deserializing them. Commands that receive a checkpoint or `.pkl` path may load
it without another confirmation prompt.

`make -C paper static-figures` verifies both listed byte counts and SHA-256
digests against `artifacts/manifest.json` before deserializing them to build
Figure 2.
