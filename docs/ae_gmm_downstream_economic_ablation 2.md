# AE/GMM Downstream Economic Ablation

This experiment compares outcome-free AE/GMM representations through their
downstream base and meta economics. It is a fast representation screen, not an
untouched-OOS validation of state discovery.

## Fixed comparison contract

- `baseline_no_aegmm`: current model feature contracts with AE/GMM outputs removed.
- `baseline_current_full_context`: exact current frozen state and exact production
  selected-feature contracts; retained as the production-context control.
- `baseline_current_full_outputs`: current frozen state with every continuous
  AE/GMM output admitted, matching challenger capacity. This is the primary
  baseline for deltas and finalist selection.
- `candidate_k{3..8}_diag`: fixed-component challengers using the current ordered
  129-feature state input contract and outcome-free reference regularization search.
- Hard cluster IDs are excluded from model inputs by default.
- Labels, model parameters, sample weights, top-k basis and 1% round-trip cost are
  fixed across arms.

Candidate representations are fitted outcome-free on beginning/middle/end rows
from the complete available covariate period. This is deliberately
representation-transductive. Supervised models remain chronological:

The exact incumbent state is reused without reinterpretation. Its manifest must
be consulted separately: the older production artifact used outcome diagnostics
to select its GMM configuration even though its inference transform consumes
pre-entry features only. The matrix records that distinction explicitly.

- Base: train on the 365 days before the first OOS month, then score five months
  without refitting.
- Base-arm selection: first three OOS months only.
- Meta: train once on those first three months of frozen base predictions.
- Meta evaluation: final two months only.

## Staged command

The runner does nothing expensive with its default `--stage plan`.

```bash
PYTHONPATH=. python3 scripts/run_ae_gmm_economic_ablation_matrix.py \
  --labels-path data_perp/artifacts/20260713_s59_h5_fullthroughjul10_trailing_cost100bps_labels/labels \
  --feature-dir data_perp/features/20260711_070000 \
  --feature-list-csv data_perp/reports/s59_h5_singlecycle_aegmm_bme_fs_fixedparams_wf30_20260716_v1/base_raw_candidate_features.csv \
  --output-root data_perp/reports/ae_gmm_downstream_economic_ablation_20260718 \
  --oos-months 2026-02,2026-03,2026-04,2026-05,2026-06 \
  --stage all
```

For controlled operation or resumption, run `states`, `base`, `meta`, and
`report` separately. Existing complete artifacts are reused unless `--rerun` is
passed.

## Outputs

- Frozen state plus transform manifests for every fit arm.
- One fixed-window base ledger and saved model per arm.
- Base top-10/20/30 metrics overall, by month, week, side and archetype.
- Base ranking on months 1-3 and deltas against the current-state full-output
  capacity-matched baseline.
- Top-30 meta handoff and one fixed meta model for each finalist.
- Meta top-10/20/30 metrics and base-vs-meta deltas on months 4-5.
- Signed residual and positive/negative hit-surprise autocorrelation.
- Per-model feature-importance exports identifying AE/GMM feature usage.
- A cost audit that fails unless the embedded round-trip cost is 1%, preventing
  report-time fee subtraction or double counting.
