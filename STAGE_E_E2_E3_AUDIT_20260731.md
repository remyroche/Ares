# Stage-E E2/E3 audit

## Contract and reproducibility

- Canonical source: Stage-D compact v9; v10 remains only its reproducibility companion.
- Evaluation scope: 24,267 development-OOF rows, April--July 2024, with the canonical 12-hour purge and side-local monthly folds.
- M0 reproduces canonical v9 exactly: raw scores, calibrated scores and probabilities have maximum absolute error `0.0`; all actions are exactly equal.
- The v1 and v2 Stage-E E2/E3 runs are byte-identical for all six files.
- No later second-OOS rows were scored, inspected, selected on, or opened by this runner.

## Minimal-information result

| Arm | MAE bps | Spearman IC | ROC-AUC | Policy uplift vs continue, bps | M0 uplift retained |
|---|---:|---:|---:|---:|---:|
| M0 full A0 | 134.09 | 0.803 | 0.954 | 75.56 | 100.00% |
| M1 three action-state fields | 177.29 | 0.025 | 0.553 | 1.87 | 2.48% |
| M2 action state + cost + geometry | 134.68 | 0.801 | 0.955 | 75.51 | 99.93% |
| M3 entry-static only | 176.13 | 0.050 | 0.557 | 2.28 | 3.02% |
| M4 action state without estimated exit net | 177.29 | 0.025 | 0.553 | 1.87 | 2.48% |
| M5 estimated exit net only | 176.06 | 0.055 | 0.554 | -0.98 | -1.30% |
| M6 time to clear only | 176.71 | 0.026 | 0.554 | 1.11 | 1.47% |
| M7 action gross return only | 176.44 | 0.065 | 0.552 | 0.90 | 1.19% |
| M8 Ridge on M2 | 13.74 | 0.985 | 0.989 | 76.92 | 101.80% |
| M8 logistic on M2 | 122.50 | 0.941 | 0.989 | 76.92 | 101.80% |
| M8 depth-2 tree on M2 | 92.06 | 0.635 | 0.796 | 67.83 | 89.77% |
| M8 depth-3 tree on M2 | 62.49 | 0.864 | 0.905 | 70.64 | 93.48% |

M1 and M4 are numerically identical because the canonical train-only correlation reducer removes the redundant estimated-exit-net field in these folds. This is the expected consequence of retaining the frozen preprocessing discipline, not an omitted arm.

## Deletion and conditional permutation

- Deleting cost fields collapses uplift from `75.56` to `1.70` bps, retaining only `2.25%`.
- Deleting action state retains `72.50` bps, or `95.95%`.
- Deleting entry-static fields retains `75.51` bps, or `99.93%`.
- Policy geometry and side identity are removed as fold-local constants by the canonical side-local preprocessing, so deleting either changes nothing.
- There are no admitted upstream-model-output fields in sealed A0; that deletion is explicitly `NOT_APPLICABLE_EMPTY_FAMILY`.
- Conditional cost-family permutation within UTC day and side reduces uplift by `71.52` bps on average. Adding time-to-clear buckets still reduces it by `68.22` bps.
- Conditional action-state permutation reduces uplift by `5.26` and `4.79` bps under the two respective schemes. Entry-static permutation is economically null.

## Interpretation

The Stage-D result is not explained by three simple action-state variables or inherited entry-static information. It is dominated by the cost family, and a regularised linear model using M2 is stronger than the high-capacity M0 model on these development rows. `known_row_cost_bps` has a development Spearman correlation of approximately `0.928` with `delta_continue_bps`.

E2/E3 do not alone establish whether this is a causal threshold-state relationship or a defective/circular cost construction. They therefore do not independently select a Stage-E terminal outcome. E1 must reconcile the cost cancellation in `delta_continue = (continue_gross - cost) - (exit_gross - cost)` with the observed predictive dominance before any second-OOS or overlay advancement. If that reconciliation exposes target/cost circularity or non-live information, the predeclared outcome is `STAGE_D_PASS_REVOKED_TARGET_PROXY_OR_CAUSAL_DEFECT`; otherwise the signal remains subject to E4--E6 falsification.
