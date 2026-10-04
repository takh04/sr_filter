# Paper results and figure inputs

This bundle corresponds to Figures 1–5 and the appendix of the camera-ready manuscript.
It contains numerical diagnostics and source records used in those figures. It
contains no model parameters, checkpoints, cached tangent directions, sample
banks, or unrelated experiment campaigns. No Hugging Face download is needed
for figure regeneration.

From the repository root:

```bash
python -m sr_filtering_nqs.paper_figures --output-dir outputs/paper
```

The command checks `manifest.json` hashes before writing outputs. Select figures
with `--figures 1 3 5`. The input root may be set with `--artifact-root`; the
default resolves relative to the installed source checkout, independent of the
working directory. The Python API is `reproduce_paper(input_root, output_dir,
figures=None)` in `sr_filtering_nqs.paper_figures`.

| Figure | Local inputs | What regeneration does |
| --- | --- | --- |
| 1 | `figure1/literature_points.json`, `source_notes.json` | Replots the curated published parameter-to-sample ratios, with source-count conventions and the displayed-point audit. |
| 2 | `figure2/amplitude_diagnostics.npz`, `metadata.json` | Replots the four amplitude-regression diagnostic panels from safe numerical arrays, including repeated-solve risk uncertainties. |
| 3 | `figure3/checkpoint_metrics.csv`, `residual_repeats.csv` | Reaggregates ten TFIM checkpoints and eight shifts into the residual and minibatch-variance curves. |
| 4 | `figure4/checkpoint_values.csv`, `summary.csv`, `full_sweep.csv`, `selected_shifts.csv` | Reaggregates the corrected seven-method TFIM/J1-J2 ablation; the tuned-bagging shift minimizes the bagged residual itself. |
| 5 | `figure5/training_curves.csv`, `endpoint_pairs.csv`, `paired_timing.csv`, `timing_summary.csv`, `protocol.json` | Replots the five paired historical J1-J2 continuations and writes the endpoint and separate TFIM timing tables. |

The renderer reproduces figure content from saved results; it does not rerun the
original training, sampling, or checkpoint-dependent diagnostics. Rendered PDF
layout can vary with Matplotlib/font versions. Figure 2 uses a portable
four-panel layout. `source_hashes.json` identifies original inputs
before extraction using archive-relative paths; those archived files are not
required for rendering.

## Appendix and source records

The `appendix/` directory preserves all 96 tracking figure points, 27,769
available tracking rows, their configurations, three exact ground states, and
274 full numerical result payloads for the main and appendix diagnostics.
`figure5/timing_raw/` contains the nine raw paired timing outputs. The original
file hashes and missing-file inventory are in `archive_import.json`.

Run `python -m sr_filtering_nqs.appendix_figures --output-dir outputs/appendix`
to regenerate both appendix plots, or use `sr-filter reproduce --recipe all`
for every manuscript figure. All 96 tracking points are retained in the original
source table. The raw archive has 279 of 288 method records; four trajectories
are incomplete. Of 83 complete three-method comparisons, 81 recover the source
table and two disagree. `appendix/online_tracking/reconstruction_audit.json`
records both disagreements, including one that crosses the win threshold.

The original Figure 5 source2500 continuation directories are absent at their
recorded paths. Training observations, endpoint estimates, protocol, and original
hashes survive in the bundled tables. The endpoint states and individual
evaluation replicas are unavailable in this bundle. Trained states and raw
sample banks are kept separately from the public results.

## Data conventions

Figure 2 NPZ entries have names `panel0_*` through `panel3_*`: matched-noise RBM,
matched-noise ViT, matched-state RBM, matched-state ViT. Each panel retains
`target`, `prediction`, and `pi` over 65,536 basis states, representative
`train_indices`, and the repeated full-population `risk_samples`. These are
derived amplitude tangent predictions and probabilities, not a model
checkpoint. Figure 2 uses the real-amplitude covariance S_A, excluding phase
outputs. Its lambda is 1e-9, and uncertainties are sample SD divided by the square
root of 100 (matched-noise) or 10 (matched-state) repeated solves. The original
matched-noise graphic displayed sample SDs (about 0.08 and 0.73), although the
manuscript defined its uncertainties as standard errors. The public renderer
uses the corresponding SEMs (about 0.008 and 0.073), consistently with the
matched-state panels. `metadata.json` records both SD and SEM and this source
annotation discrepancy; the underlying risk samples are unchanged.

Figures 3 and 4 retain the source diagnostic convention: the target is twice the
local energy, giving the common factor of four in squared diagnostics. Their
error bars follow the manuscript's convention, checkpoint SD with ddof=0 divided
by sqrt(10). The raw repeat residuals let readers inspect the saved aggregation;
variance diagnostics are checkpoint summaries because direction vectors are
omitted. Figure 4 includes the eight-shift bagged-objective sweep only because
it determines the displayed tuned-bagging bar. The displayed J1-J2 rows use the
corrected complex-parameter layout.

Figure 5 contains only source checkpoint 2500 and all five branch seeds 104–108.
For each seed the 500 SR records and 125 historical MS-SR records contain the
energy measured before that update. The horizontal axis counts one candidate
SR update per SR step and four per historical MS-SR step. This nominal convention
does not account for the historical callback's additional solve or sampling.
`protocol.json` gives the relevant cost accounting. Endpoints use independent
Monte Carlo evaluations; table errors combine the individual Monte Carlo
standard errors. The paired interval is computed across five endpoint
energy differences conditional on one common source state. Panel C uses the
separate NTK-quantile TFIM implementation and medians of three paired device/seed
settings, not the historical J1-J2 implementation.
