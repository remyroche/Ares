# TP6/SL4 canonical residual meta handover

## Scope and control

This workstream is independent of GAM.  The control is the exact no-GAM
canonical Base+Consensus stack in
[TP6_SL4_BASE_CONSENSUS_CANONICAL_20260807.md](TP6_SL4_BASE_CONSENSUS_CANONICAL_20260807.md):

```text
R3 score = P(clear) - 0.5 P(adverse)
train-only base map -> 8 consensus residual heads
canonical score = 0.75 base rank + 0.25 consensus rank
```

All economics use TP `+6 ATR`, SL `-4 ATR`, H12, 100 bps cost once, with one
pooled global ranking after monthly side normalization.  The control is not a
GAM arm.

## New residual target

For every held fold and side, fit a monotonic map on earlier resolved rows:

```text
CanonicalScore -> CanonicalExpectedNetBps
R_meta = ExactNetBps - CanonicalExpectedNetBps
```

The meta learner predicts ordinal residual grades with boundaries
`[-150, -50, +50, +150]` bps.  It uses native LambdaRank, 4-hour UTC × side
queries, depth 4, 12 leaves, 120 trees, learning rate 0.04, minimum child
fraction 3%, feature fraction/bagging 0.80, L1=1, L2=10, and max bin 63.
The residual rank is combined with the canonical rank using a frozen 75/25
blend.

## Feature blocks

The feature contract is deliberately compact:

- **Uncertainty:** dispersion/MAD/IQR/range across all eight consensus heads,
  agreement fraction, base-vs-consensus disagreement, R3 probabilities,
  probability entropy, top-two margin, and conviction.
- **Support/OOD:** train-support robust distance, missing fraction, extreme
  tail/outlier fractions, and low-support exposure.
- **Drift:** recent-vs-history context shift, covariance-break proxy,
  score-distribution shift, head-dispersion shift, and volatility/breadth/
  liquidity shifts.
- **Market state:** 21 existing causal fields covering volatility, breadth,
  dependence, liquidity, funding, leverage, and participation.

The raw eight-head OOF health artifact is
[canonical_head_health_2025.parquet](../data_perp/artifacts/tp6_sl4_canonical_head_health_2025_v1/canonical_head_health_2025.parquet).

## Three-fold feature selection

Selection used only the available pre-2025 history (July, September, and
November 2024 screens), with the same shallow LambdaRank screen in every fold,
Spearman correlation pruning at `|rho| >= .90`, and recurrence in at least two
of three chronological screens.  The selected subsets are in
[selected_features.json](../data_perp/artifacts/tp6_sl4_canonical_residual_feature_selection_20260808_v1/selected_features.json).

Selected block sizes:

| Block | Fields |
|---|---:|
| Uncertainty | 13 |
| Support/OOD | 7 |
| Drift | 3 |
| Market state | 14 |

## 2025 strict-OOF A–H ablation

The selected-feature results below are pooled across both sides.  Top-5 is the
predeclared primary tail.

| Arm | Inputs | Top-1 net | Top-2 net | Top-5 net | Top-10 net | Mean monthly Top-5 | Positive months |
|---|---|---:|---:|---:|---:|---:|---:|
| A control | Canonical Base+Consensus | −39.83 | −13.78 | **−12.30** | −22.57 | −11.69 | 7/12 |
| B | Uncertainty | +53.05 | +28.27 | **−3.39** | −25.63 | −13.51 | 6/12 |
| C | Support/OOD | −8.57 | +18.29 | −25.02 | −28.79 | −21.76 | 5/12 |
| D | Drift | −41.58 | +3.55 | −14.39 | −25.74 | −28.53 | 3/12 |
| E | Market state | +28.68 | −13.20 | −17.77 | −29.69 | −15.25 | 5/12 |
| F | Uncertainty + OOD | +15.01 | +0.41 | −18.24 | −29.73 | −24.94 | 5/12 |
| G | Uncertainty + OOD + drift | +25.24 | +14.98 | −12.25 | −33.69 | −22.04 | 6/12 |
| H | Full | +27.78 | +11.86 | −13.67 | −28.06 | −11.35 | 5/12 |

Side-local selected Top-5 net bps/trade:

| Arm | Long | Short |
|---|---:|---:|
| A control | +25.91 | −50.55 |
| B uncertainty | **+31.93** | −42.88 |
| C support/OOD | +39.70 | −65.95 |
| D drift | +14.54 | −43.06 |
| E market state | +4.25 | −41.03 |
| F uncertainty + OOD | +15.01 | −54.69 |
| G uncertainty + OOD + drift | +34.71 | −55.29 |
| H full | +18.49 | **−39.76** |

The apparent pooled improvement is mostly the short-side repair of a negative
short baseline; it is not a universal long/short uplift.  The full block does
not beat the uncertainty-only arm.

## Failure diagnostic

The same meta output was also read as an adverse-error proxy:

```text
failure_proxy = 1 - residual_rank
actual_failure = 1[R_meta <= -150 bps]
```

The selected 2025 diagnostic is effectively null: global failure AUC is 0.491
for uncertainty, 0.492 for uncertainty+OOD+drift, and 0.500 for the full arm.
The failure rate is approximately 24.7%.  There is not yet evidence for a
separate failure head.

## Later chronological validation

The selected uncertainty and full arms were refit on 2025 and applied once to
the untouched 20–23 July 2026 long population.  The later evaluator reproduces
the canonical control exactly: Top-5 gross/net `50.76 / -49.24` bps/trade.

| Arm | Top-1 net | Top-2 net | Top-5 net | Top-10 net | Top-5 mean across four days |
|---|---:|---:|---:|---:|---:|
| Canonical control | −47.37 | −24.44 | **−49.24** | −69.70 | −9.45 |
| Uncertainty residual | −32.55 | −8.74 | −75.75 | −75.36 | −38.63 |
| Full residual | −0.91 | −34.39 | −73.66 | −72.83 | −49.05 |

The new residual layer does not transport to this later period.  The four-day
window is a robustness rejection, not a sufficient positive validation sample;
a materially longer untouched chronological window is still required before
promoting any residual arm.

## Longer frozen-model chronology: January–July 10, 2026

The raw strict-OOS R3/TP6 panel also supports a longer window before the
July-20 later replay.  The canonical eight-head contract was materialized for
January through July 10, with every month’s heads fit only on rows available
before that month.  The residual meta model and its canonical score-to-bps map
were frozen on 2025 and never refit on 2026 outcomes.

| Arm | Top-1 net | Top-2 net | Top-5 net | Top-10 net | Mean monthly Top-5 | Positive months |
|---|---:|---:|---:|---:|---:|---:|
| Canonical control | −41.41 | −44.54 | **−26.51** | −27.48 | −31.44 | 1/7 |
| Uncertainty residual | −73.26 | −47.80 | −35.02 | −26.00 | −30.32 | 1/7 |
| Full residual | −21.18 | −25.02 | **−24.37** | −22.06 | **−16.13** | 3/7 |

The full arm improves the pooled long-window Top-5 by +2.14 bps/trade and has
better month stability, but this is driven by the short side: short Top-5 is
`+19.42` versus `+6.57` for control, while long Top-5 is `−55.86` versus
`−62.66`.  Uncertainty alone does not improve the pooled window.  This is a
promising robustness signal for the full block, not execution readiness: all
pooled tails remain net-negative and the long side remains materially weak.

The longer-window artifacts are in
[tp6_sl4_canonical_residual_meta_2026_window_20260808_v1](../data_perp/artifacts/tp6_sl4_canonical_residual_meta_2026_window_20260808_v1).

## Artifacts

- [Raw-head OOF health](../data_perp/artifacts/tp6_sl4_canonical_head_health_2025_v1/run_manifest.json)
- [Feature-selection audit](../data_perp/artifacts/tp6_sl4_canonical_residual_feature_selection_20260808_v1/selection_audit.parquet)
- [All-block ablation](../data_perp/artifacts/tp6_sl4_canonical_residual_meta_block_ablation_20260808_v1/TP6_SL4_CANONICAL_RESIDUAL_META_BLOCK_ABLATION_REPORT.md)
- [Selected-feature ablation](../data_perp/artifacts/tp6_sl4_canonical_residual_meta_block_ablation_selected_20260808_v1/TP6_SL4_CANONICAL_RESIDUAL_META_BLOCK_ABLATION_REPORT.md)
- [Later OOS](../data_perp/artifacts/tp6_sl4_canonical_residual_meta_later_oos_20260808_v2/TP6_SL4_CANONICAL_RESIDUAL_META_LATER_OOS_REPORT.md)

## Current decision

Do not promote a residual arm yet.  The uncertainty block is the strongest
2025 development candidate but fails both later tests.  The full block is the
best longer-window robustness candidate, improving month stability and the
short side, but it remains net-negative in every pooled tail and weakens the
long side.  The next step is a predeclared, longer 2026+ chronology with the
full feature subset, side-specific admission, and no additional architecture
or target selection on that data.
