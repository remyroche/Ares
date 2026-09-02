# Causal five-state and multi-geometry regime audit

## Implemented representation

The hourly regime sidecar now provides a frozen, strictly chronological
five-state market-geometry simplex:

1. quiet/choppy;
2. coherent directional expansion;
3. fragmented/idiosyncratic;
4. systemic stress/deleveraging;
5. recovery/settling.

The semantic coordinates are a train-only remapping of the fitted primary GMM
centroids.  They are not targets and do not use realised returns.  Signed
market direction is separately exposed, so an upward and a downward coherent
trend remain the same geometry state.  The sidecar also publishes all state
probabilities, entropy, top-two margin, state age, switch probability, and
the separate causal transition simplex (stable/onset/active/settling).

Four independent soft geometry views are additionally materialized:
trend/volatility, breadth/dependence, leverage/flow, and liquidity.  Inputs
are selected from pre-existing decision-time market features, kept disjoint
between specialists, and float32/vectorized.  State discovery prioritizes
persistent level/stress inputs over deltas/accelerations; transitions use the
forward movement dynamics.  GMM fitting uses a bounded proxy sample, while
the selected frozen model is fit only on prior rows.

## Causality and feature boundary

`oof_causal_market_regime_systems_2023q3_2024_20260803_v4` contains 11,712
hourly OOF rows and `candidate_oof_market_regimes.parquet` covers 237,246
matched candidates from 2023-09 through 2024-12.  Candidate joins are
backward-only (two-hour maximum lag).  No candidate has a source timestamp,
availability timestamp, or regime-train cutoff later than its decision time.
The semantic simplex has 100% coverage and maximum sum-to-one error
`1.79e-7`.

All latent memberships and the semantic primary simplex remain candidate-only
for the meta layer and are forbidden from base alpha.  The default meta
universe is unchanged: the matched economic gate below did not justify
automatic promotion.

## Sequential parameter assessment

The new selector is label-free: three chronological proxy windows, two seeds,
K screen at moderate stickiness, followed by a stickiness screen only for the
chosen K.  It records coverage, support, dwell, switch rate, confidence,
phase behaviour, posterior/centroid portability and cross-view redundancy;
it then uses Pareto plus one-standard-error selection.

Recommendations on the 2023Q3--2024 proxy:

| View | Result |
|---|---|
| Primary five-state | No K/stickiness candidate cleared all structural gates; K=5 itself had inadequate minimum support. |
| Trend/volatility | K=3, stickiness 0.00. |
| Breadth/dependence | K=3, stickiness 0.80. |
| Leverage/flow | K=3, stickiness 0.80. |
| Liquidity | No candidate cleared all structural gates. |

The three selected non-primary views were not redundant by the compact
permutation-invariant proxy (mean absolute Spearman 0.020--0.028).  This is
structural evidence only, not proof of trading value.

## Matched economic and portability replay

The causal Ridge replay ranks globally after one common-bps mapping, using the
same 237,246 candidates and 100-bps cost convention as its baseline.

| Arm, top 10% | Net bps/trade | Net rank IC | Change in net vs baseline |
|---|---:|---:|---:|
| Baseline | -107.95 | 0.0810 | 0.00 |
| Primary invariants | -106.78 | 0.0788 | +1.17 |
| Trend/volatility invariants | -105.94 | 0.0785 | +2.01 |
| Breadth/dependence invariants | -103.01 | 0.0771 | +4.94 |
| Primary semantic membership | -103.57 | 0.0766 | +4.38 |
| All geometry memberships (diagnostic only) | -113.70 | 0.0753 | -5.76 |

No arm improves both top-10 economics and rank IC, so none advances.  The
best economic arms also failed transport: on 2023Q4 -> 2024 the baseline was
-117.00 net bps and the semantic-primary arm was -108.72; on 2024H1 ->
2024H2 they were -75.33 and -87.95 respectively.  All tested arms remain
negative in both splits.

## Remaining gaps and next decision

The representation is now reliable enough for diagnosis, but there is no
evidence that it repairs the conversion/ranking problem.  The key gaps are:

1. the requested five-state ontology has insufficient recurrent support under
the current market-level proxy, especially at K=5;
2. portable structural geometry does not yet translate into incremental
candidate-level execution information;
3. liquidity is structurally unstable, and fold-local membership coordinates
are particularly unsafe;
4. the residual stack remains net-negative even before the regime extension.

Do not run a broad regime HPO next.  The credible next experiment is a
side-specific residual/reliability model using only the compact continuous
context and, at most, the individually selected breadth/dependence invariant
subset.  Admit any latent bundle only if a repeated matched OOF test improves
both rank IC and worst-period economics.

## Artifacts

- `data_perp/artifacts/oof_causal_market_regime_systems_2023q3_2024_20260803_v4`
- `data_perp/artifacts/causal_market_regime_parameter_funnel_20260803_v2`
- `data_perp/artifacts/regime_geometry_portability_ablation_20260803_v4`
