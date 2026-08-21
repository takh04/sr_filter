# Release artifact curation

The scripts in this directory convert a trusted experiment checkout into the
portable CSV/JSON bundle under `artifacts/`. They intentionally load historical
pickle inputs, so run them only on trusted data.

The frozen TOML files in `configs/experiments/` record exact commands and
protocols. The `sr-filter` CLI does not load those TOMLs directly.

## Compact result bundle

Set the trusted input root explicitly; no private checkout location is embedded
in emitted artifacts.

```bash
python scripts/curation/curate_release_artifacts.py \
  --source "${SR_FILTER_TRUSTED_EXPERIMENT_ROOT:?set SR_FILTER_TRUSTED_EXPERIMENT_ROOT}" \
  --online-results-root outputs/gpu/online \
  --endpoint-results-root outputs/gpu/endpoints

python scripts/curation/validate_release_artifacts.py
```

The online finalization command is idempotent. It imports MS-SR only from the
explicit `branch_seed0/mssr_exact` and `branch_seed1/mssr_exact` directories.
For each arm, the summary must say `completed`, the trajectory must contain
exactly 125 contiguous updates, and the trusted final checkpoint must prove
the `fixed_grid_exact_loo` contract: four candidate batches and four solves per
update, no base-driver or weight-fit sampling, and 500 cumulative units. The
trainer records that proof in a JSON audit sidecar bound to SHA-256 hashes of
the final checkpoint, configuration, and complete training-metrics CSV; the
curator verifies the sidecar without loading the online output pickle. The
historical `mssr` callback directories are never consulted. Seed 1's branch
origin is read from the step-1800 checkpoint's recorded energy history and
emitted as one `shared_sr` row at budget zero. Endpoint JSONs are accepted only
when their branch, accepted checkpoint hash, sampling contract, arm, and
deterministic seed all match. Pending rows contain empty measurement fields.

## External checkpoint staging

The external inventory names eight checkpoints: two branch sources and three
final arms for each of two branch seeds. Set the environment variables listed
in `configs/experiments/external_checkpoint_inventory.toml`, then run:

```bash
python scripts/curation/sanitize_checkpoints.py \
  --inventory configs/experiments/external_checkpoint_inventory.toml \
  --staging-root "${SR_FILTER_EXTERNAL_STAGING_ROOT:?set SR_FILTER_EXTERNAL_STAGING_ROOT}" \
  --public-manifest artifacts/external_checkpoints.json \
  --acknowledge-trusted-pickle-inputs
```

Add `--verify-runtime` on a machine with the experiment environment and an
available accelerator to reconstruct every checkpoint in addition to the
mandatory pickle round-trip and parameter-tree checks. The sanitizer never
modifies source files. It replaces absolute path metadata with logical artifact
identifiers and records original and sanitized SHA-256 digests separately.

Do not upload until all eight records are sanitized, runtime verification has
passed, and the remote dataset ID and immutable revision have been written into
the external manifest.
