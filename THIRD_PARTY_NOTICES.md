# Third-party notices

This repository contains a minimal subset of support code required to execute
the published experiments. The bundled namespaces are implementation details,
not independent redistributions of the complete upstream projects.

## NetKet-derived files

Files carrying `Copyright ... The NetKet Authors` headers under
`src/advanced_drivers`, `src/netket_foundational`, and `src/nqs_support_core`
are derived from the [NetKet](https://github.com/netket/netket) project and are
distributed under the Apache License 2.0. Existing copyright and license
headers are retained in each source file.

## Neural Tangents-derived files

Files under `src/nqs_support_core/_src/external/neural_tangents` carrying
Google LLC copyright headers are derived from
[Neural Tangents](https://github.com/google/neural-tangents) and are
distributed under the Apache License 2.0. The bundled subset records upstream
commit `c17e770bb74f1771da7be4a69fabfa68b6078960` in its package initializer,
retains existing copyright and license headers, and includes the upstream
license at `src/nqs_support_core/_src/external/neural_tangents/LICENSE`.

The bundled Neural Tangents utility
`src/nqs_support_core/_src/external/neural_tangents/utils/dataclasses.py`
also records that it was copied and adapted from
[Flax](https://github.com/google/flax), which is distributed under the Apache
License 2.0. Its Google LLC copyright and Apache license header are retained.

The full Apache License 2.0 text is available in `LICENSE`. Dependencies
installed from package indexes remain governed by their respective licenses.
