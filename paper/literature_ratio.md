# Parameter-to-sample literature ledger

`literature_ratio.csv` is the machine-readable source for the introductory
parameter-to-sample figure. `P` follows each paper's parameter convention:
complex variational parameters count as one source parameter unless that paper
reports real scalar degrees of freedom. `N_s` is the raw optimizer batch size,
not an autocorrelation-adjusted effective sample size or a post-training
evaluation budget. The figure is therefore an order-of-magnitude historical
comparison, not a controlled benchmark across architectures.
`publication_year` records the bibliographic year; `plot_year` applies a small
horizontal jitter so multiple architectures from one source remain visible.

The `evidence` column has three levels:

- `explicit`: the paper reports the paired parameter and optimizer-sample counts;
- `derived`: `P` is computed from a paper-reported architecture equation and
  hyperparameters;
- `estimate`: the paper reports an approximate count or scale.

The generator validates every stored ratio against `parameters / samples` and
rejects nonpositive values, unknown evidence classes, unknown domains, duplicate
rows, and non-HTTPS source URLs. Qualitative historical anchors without a clean
paired `P,N_s` value are discussed and cited in the manuscript but deliberately
omitted from the quantitative scatter plot.

Reviewed examples without a clean paired optimizer count, including classical
Jastrow/Gutzwiller VMC, PauliNet, several convolutional NQS, and the 2025 FNQS,
are likewise omitted rather than assigned an inferred coordinate. Full source
titles and bibliographic metadata are in `references.bib`; exact extraction URLs
are recorded per row in the CSV.
