# Security policy

## Supported versions

Security fixes are applied to the latest tagged release and, before the first
release, the current default branch. Historical research snapshots and
unpublished experiment branches are not supported.

## Reporting a vulnerability

Please report vulnerabilities privately to `tak.hur@uwaterloo.ca`; do not
open a public issue first. Include a minimal reproducer, affected version or
commit, impact, and any suggested mitigation. Do not include credentials,
private checkpoints, or personal data.

## Security model

SR Filter is an offline research command-line package, not a network service
or a multi-user execution environment. Repository source, the installed
Python environment, and explicitly selected local inputs are inside its trust
boundary. Configurations, manifests, checkpoints, CSV/JSON files, and URLs
supplied by another party are untrusted inputs.

Reportable issues include arbitrary code execution or file access that does
not require the documented trust decision, artifact path traversal or writes
outside the selected output root, bypasses of manifest integrity checks,
credential disclosure, and compromise of the package or CI supply chain.
The following invariants should hold:

- artifact paths remain beneath the caller-selected output root;
- downloads are limited to records in the selected manifest and are accepted
  only after their declared size and SHA-256 digest match;
- remote manifests use HTTPS and manifest, record, per-file, and aggregate
  byte limits are enforced before any output directory is created;
- downloading or plotting CSV/JSON artifacts does not deserialize a Python
  pickle; and
- generated results are written only to an explicit or documented relative
  output directory.

## Checkpoint safety

External experiment checkpoints can contain Python pickle payloads. The bundled
figure data do not use pickle. Pickle deserialization can execute arbitrary
code. The artifact fetch command verifies bytes but never deserializes them.
Commands that accept a pickle or checkpoint path are trusted-input interfaces
and may deserialize it without sandboxing or another confirmation prompt.

Only invoke those commands on files you created and trust. A matching checksum establishes identity, not safety; a
third party can publish a malicious file and its checksum together. Run
research workloads with least privilege and avoid mounting credentials or
unrelated data into the process.

## Scope

Reports about dependency vulnerabilities should identify a reachable path in
this package. Numerical disagreement, failed convergence, scientific model
limitations, and expected resource use from deliberately large GPU jobs are
valuable bug reports but are not security vulnerabilities unless they cross a
trust boundary described above.
