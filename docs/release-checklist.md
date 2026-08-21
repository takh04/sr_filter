# Public release checklist

Complete this checklist from a clean clone before creating a version tag.

## Repository and identity

- Create the public repository at `https://github.com/takh04/sr_filter` and
  confirm that README, package metadata, citation metadata, and manuscript
  links resolve to it.
- Confirm that Tak Hur's Yonsei University affiliation and contact are current
  in `CITATION.cff`, `pyproject.toml`, the manuscript, and `SECURITY.md`.
- Remove `rebuttal_assets/`, machine-local output, caches, anonymous package
  wording, and any prior repository-name references.
- Run `python scripts/audit_release.py` and resolve every finding. Do not
  weaken a rule merely to admit a known private or generated file.

## Code and artifacts

- From a clean clone, run `uv sync --frozen --extra dev`, `uv run pytest`, and
  `uv run sr-filter reproduce --recipe curated --input-root .` on CPU.
- Build a wheel, install it outside the source tree, and run `sr-filter --help`.
- Verify every manifest byte count and SHA-256 digest. Review
  `artifacts/exclusions.json` for accurate, non-misleading exclusion reasons.
- Decide whether large checkpoints and raw per-round records are part of the
  release. If they are, publish `takh04/sr-filter-artifacts`, pin an immutable
  dataset revision in the manifest, and test a clean fetch. If they are not,
  describe that reproducibility boundary explicitly instead of publishing a
  dead link.
- Treat every pickle and checkpoint as trusted input. A checksum verifies
  identity but does not make pickle deserialization safe.
- Dispatch the CUDA smoke workflow on a trusted CUDA 12 runner when claiming
  GPU support; archive the runner, driver, CUDA, JAX, and device information.

## Manuscript and release

- Run `make -C paper clean camera-ready`. Confirm that both audited online
  branches are present, all generated-input checks pass, and main content ends
  by page 10.
- Inspect the final PDF visually and check its title, author, affiliation,
  links, figure labels, appendix references, and document metadata.
- Confirm that `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md` accompany the
  source and retain the upstream headers in derived files.
- Update the version and release date consistently in `pyproject.toml`,
  `CITATION.cff`, and `CHANGELOG.md`, then create the source archive from the
  exact tagged commit.
