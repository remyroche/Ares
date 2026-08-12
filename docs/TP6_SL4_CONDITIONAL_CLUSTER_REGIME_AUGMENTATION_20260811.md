# TP6/SL4 conditional path layer — strict OOF regime/transition augmentation

Date: 2026-08-11  
Scope: long-only canonical 2025 TP6/SL4/H12 population

## What changed

The canonical cluster/path layer now receives the complete available causal
meta pool before train-only CMI selection:

- 104 previously available causal/meta fields;
- 209 finite fields from the strict OOF market-regime sidecar;
- 313 fields exposed to each fold's selector;
- seven to three-dimensional structural families retained from the canonical
  four-band path taxonomy (83 recurrent families);
- exact per-row target: cluster soft membership ×
  `(exact TP6/SL4 net bps − train-only base expected bps)`.

The regime sidecar was refit over 2023-09 through 2025-12 in chronological
quarterly blocks.  Every posterior, transition simplex, and continuous context
value is produced by a model fitted strictly before its evaluation block.

The canonical 2025 rows were joined by a backward as-of rule:

```text
source_utc <= candidate decision timestamp
and lag <= 2 hours
```

All 10,224 long rows joined successfully.  Future timestamps and invalid
sidecar provenance are rejected.  Geometry centroid-distance coordinates that
were genuinely unavailable because a fold had fewer states were excluded,
not imputed as zero.

## Regime/transition fields exposed

The sidecar includes:

- five primary state posterior coordinates;
- primary regime entropy, top-2 margin, OOD/uncertainty, state age and switch
  probability;
- stable/onset/active/settling transition probabilities and their entropy,
  margin, OOD and uncertainty;
- four geometry-specific soft regime systems (trend/volatility,
  breadth/dependence, leverage/flow, liquidity);
- prequential 90/180-day relative rank/z context, 4/24-hour changes and
  30/90-day relationship-break residuals.

The stable feature family is registered in `config.py` as
`CAUSAL_SOFT_REGIME_TRANSITION_META_FEATURE_KEYS`; it is meta-only and is not
added to the base alpha contract.

## Strictness checks

| Check | Result |
|---|---:|
| Canonical long rows | 10,224 |
| Regime join coverage | 100.0% |
| Path/family coverage | 100.0% |
| Regime train end < state availability | pass |
| State availability <= candidate timestamp | pass |
| Outcome fields in selector pool | 0 |
| Family/CMI discovery train-only | pass |
| Global ranking after monthly scoring | pass |

The 83-family matrix still represents only 43.8% of native absolute path
contribution mass on average; the remaining mass is retained as explicit
unassigned evidence rather than being treated as a zero path.

## Pooled global TP6/SL4 metrics

Net values include the 100 bps cost. Gross is exactly net + 100 bps.

| Arm | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% | Top 20% | Rank IC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base control | −36.16 | −68.77 | −3.13 | **+11.40** | −4.48 | −29.13 | 0.0641 |
| Cluster-only correction | **+18.97** | −130.78 | +1.95 | +8.50 | −10.97 | −32.68 | 0.0605 |
| Cluster + full regime/context pool | −9.41 | −130.78 | +1.95 | +5.52 | −10.45 | −29.40 | 0.0604 |

Adding the full regime/transition pool therefore did not improve the pooled
top-5 ranking: it moved the cluster-context arm from +8.91 to +5.52 bps/trade.
The cluster-only arm remained unchanged because the added fields are used only
in the context arm.

## Monthly top-5 net bps/trade

| Month | Base | Cluster-only | Cluster + regime/context |
|---|---:|---:|---:|
| Feb | +126.09 | +146.71 | +147.05 |
| Mar | −32.67 | +0.33 | −2.16 |
| Apr | +64.87 | +24.87 | +26.41 |
| May | −38.83 | −137.78 | −122.88 |
| Jun | +96.08 | +1.81 | +23.13 |
| Jul | +21.18 | +12.74 | +12.74 |
| Aug | −167.81 | −159.53 | −159.53 |
| Sep | −8.29 | −26.00 | −50.53 |
| Oct | −137.26 | −163.50 | −162.71 |
| Nov | +142.53 | +165.35 | +165.35 |
| Dec | +25.84 | +25.45 | +29.95 |

The regime/context arm helps June and December and slightly reduces the May
loss, but worsens September and does not repair the August/October failure.

## What the selector actually used

The train-only CMI audit recorded 1,008 selected field instances across the
11 held-month folds.  738 (73.2%) were regime/transition/continuous-context
fields, showing that the new inputs are not merely present in the schema—they
are being selected for cluster-specific residual models.

Frequently selected examples include:

- `continuous_regime__dependence__rank_180d`;
- `continuous_regime__dependence__z_90d`;
- `continuous_regime__dispersion__rank_180d`;
- `continuous_regime__relationship_break__isolation_dependence__residual_signed_30d`;
- `geometry_regime__breadth_dependence__state_p_2/p_3`;
- `geometry_regime__breadth_dependence__state_age_hours`;
- `geometry_regime__leverage_flow__state_centroid_distance_p_1`;
- primary `regime_state_p__0`, `regime_state_p__3`, and `regime_top2_margin`.

The highest CMI values were concentrated in relationship-break, dependence,
state-age, and leverage-flow fields, rather than in one universal primary
state coordinate.  This supports using the regime surface as conditional trust
context, not as a standalone score.

## Decision

The implementation requirement is satisfied: the canonical path layer now has
strictly causal five-state/transition and continuous regime inputs available
before feature selection, and the inputs are actually selected in the
cluster-specific models.

The economic advancement gate is **not** satisfied.  The regime/context arm
does not beat the existing base top-5 result and remains negative in the worst
months.  Do not promote it as a ranking correction.

The next useful experiment is not another broad feature sweep.  It should test
whether these regime fields improve *when to trust a path correction* by adding
explicit assignment-confidence/OOD gating or a residual reliability target,
while retaining the raw base ranking as the primary ordering.  The 43.8%
represented path-mass gap and the severe August/October failures remain open
diagnostics.

## Extended-history confirmation

To test whether the result depended on starting the taxonomy in 2025, the
strict meta-path materializer was extended through the available 2024 history.
It produced 20 completed held months (April–November 2024 and January–December
2025; the source has no December 2024 rows).  A matching extended base panel
and regime/meta pool contain 17,040 long rows, and the cluster runner evaluates
19 chronological held months, including January 2025.

The cross-fold taxonomy grew to 100 recurrent families.  Represented native
path mass was 38.5%, lower than the 43.8% in the 2025-only taxonomy because the
longer history introduces more structural variation.

| Arm | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% | Top 20% |
|---|---:|---:|---:|---:|---:|---:|
| Base control | −31.70 | −25.66 | −35.87 | −7.74 | −15.51 | −29.88 |
| Cluster-only | **+7.34** | −18.41 | +3.94 | −11.79 | −11.23 | −28.66 |
| Cluster + regime/context | **+32.46** | −18.22 | +3.61 | −20.68 | −9.92 | −25.11 |

At the extended top-5 tail, the base has a −6.87 bps monthly mean (9/19
positive months), cluster-only +6.97 (11/19), and cluster+context −1.64 (9/19).
The context arm is therefore not a robust improvement; the narrow top-0.5%
gain is not enough to offset the broad-tail deterioration.

The extended artifacts are:

- [extended family contract](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_canonical_meta_paths_20260811_extended_v1/);
- [extended base panel](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_extended_cluster_base_20260811_v1.parquet);
- [extended regime/meta pool](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_extended_cluster_meta_pool_regime_20260811_v1.parquet);
- [extended replay](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_conditional_cluster_residual_20260811_extended_regime_v1/).

## Causal reliability/OOD gate overlay

The extended replay was then passed through a predeclared trust overlay.  The
overlay does not refit a model, inspect held outcomes, or filter candidates; it
only multiplies the signed cluster correction by a causal trust weight before
the final global ranking.  Inputs were limited to path representation/margin
and strict OOF regime/transition support:

- path represented mass, entropy, and top-2 margin;
- primary regime OOD score, margin, and switch probability;
- transition OOD score, margin, and entropy.

The tested gates were `all`, represented mass ≥0.25, represented mass ≥0.50,
represented mass plus path margin, represented mass plus path/regime support,
and a clipped soft product of the same support terms.  Correction multipliers
were λ ∈ {0.25, 0.50, 0.75, 1.00}.  Ranking used the canonical global
`score desc, candidate_id asc` tie-break.

The best pooled top-5 arm was:

| Arm | Pooled top-5 net | Monthly mean | Worst month | Positive months |
|---|---:|---:|---:|---:|
| Base control | −7.74 bps | −4.66 bps | −172.74 bps | 11/19 |
| Cluster-only, no gate | −11.79 bps | +6.97 bps | −177.21 bps | 11/19 |
| Cluster-only + represented mass ≥0.25, λ=0.50 | **−7.01 bps** | **−0.83 bps** | −175.31 bps | **12/19** |
| Cluster + regime/context + best gate | −7.39 bps | −7.52 bps | −158.59 bps | 9/19 |

The best gate is a small improvement over the base on pooled top-5 and month
count, but it remains negative and does not improve the worst month.  The
stricter path/regime gate had only 7.3% positive-weight coverage; the soft gate
had a mean weight of 0.395.  These coverage facts are important: the result is
not evidence that a reliable admission subset has been found.

The complete artifacts are:

- [gate metrics](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_cluster_reliability_gate_20260811_v1/reliability_gate_metrics.parquet);
- [gate stability](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_cluster_reliability_gate_20260811_v1/reliability_gate_stability.parquet);
- [gate coverage](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_cluster_reliability_gate_20260811_v1/reliability_gate_coverage.parquet);
- [gate report](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_cluster_reliability_gate_20260811_v1/RELIABILITY_GATE_REPORT.md);
- [gate runner](/Users/remyroche/Documents/Ares/scripts/run_tp6_sl4_cluster_reliability_gate.py).

The reliability overlay therefore remains a diagnostic feature layer, not a
promoted economic correction.  The next substantive repair should model
per-path conversion/reliability directly using a prior-resolved target, rather
than adding more fixed trust thresholds.

## Explicit unassigned-path residual ablation

The frozen family contract represents only part of the native path contribution
mass.  To test whether the omitted mass was itself a useful conditional signal,
the remaining `cluster_path_unassigned_mass` was exposed as a separate soft
membership (`frozen_unassigned`) and given its own residual head.  The head
used the same 313-field causal/meta pool, train-only CMI selection, and the
same frozen-cluster score as its control.  It was not merged into any named
cluster.

On the 12 held 2025 months, the explicit head did not improve the frozen
cluster control:

| Arm | Top-0.5% net | Top-1% net | Top-2% net | Top-5% net | Top-10% net | Top-5 monthly mean | Worst month | Positive months |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | −98.12 | −106.56 | −32.76 | +2.57 | −8.94 | +2.58 | −167.81 | 6/12 |
| Frozen clusters | −99.73 | −103.32 | −12.04 | **+6.81** | −4.58 | +3.84 | −186.99 | 7/12 |
| Frozen clusters + unassigned | −109.19 | −103.32 | −12.04 | +1.72 | −6.00 | +4.02 | −206.39 | 7/12 |

The earlier-era transport check was also negative at pooled top-5:

| Arm | Top-2% net | Top-5% net | Top-10% net | Top-5 monthly mean | Worst month |
|---|---:|---:|---:|---:|---:|
| Base | +11.34 | **+3.27** | −26.82 | −23.08 | −90.14 |
| Frozen clusters | −2.40 | −12.03 | −23.95 | −12.41 | −122.32 |
| Frozen clusters + unassigned | **+13.31** | −14.49 | −23.54 | −3.00 | −85.15 |

The unassigned head therefore improves the earlier-era monthly average and
worst month but loses the pooled top-5 gate and worsens the 2025 pooled top-5
and worst month.  It is retained as a diagnostic coverage feature, not
promoted as a production correction.  Artifacts:

- [2025 unassigned replay](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_frozen_unassigned_ablation_20260812_v1/)
- [earlier-era transport replay](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_frozen_unassigned_transport_20260812_v1/)
- [unassigned runner](/Users/remyroche/Documents/Ares/scripts/run_tp6_sl4_frozen_unassigned_ablation.py)

## Frozen cross-fold cluster contract

The fold-by-fold cluster audit exposed an important semantic defect: the old
`cluster_00`, `cluster_01`, ... names were reused for different family
membership sets on different folds.  Those were valid fold-local diagnostics,
but not persistent specialists.  A separate replay now discovers the cluster
geometry once on the 2024 development population and applies the identical
family membership contract to every 2025 held month.

The frozen contract has 100 structural family inputs, five stable clusters,
and 313 configured causal/meta fields available before train-only CMI.  Each
cluster receives a separate soft target,
`membership × (exact TP6/SL4 net − train-only base expected bps)`, and at most
16 CMI-selected context fields.  No 2025 outcome enters clustering, feature
selection, or fitting.

| Arm | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% | Top 20% | Rank IC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base control | −98.12 | −106.56 | −32.76 | +2.57 | −8.94 | −29.54 | 0.0636 |
| Frozen cluster residual | −99.73 | −103.32 | −12.04 | **+6.81** | **−4.58** | **−28.08** | **0.0652** |

On the 12 held 2025 months, top-5 mean net improved from +2.58 to +3.84
bps/trade and positive months from 6/12 to 7/12.  This is the first replay in
this workstream to improve the broad top-5 tail with a persistent cluster
contract.  It is not yet promotion-grade: the worst month worsened from
−167.81 to −186.99 bps, and January, March, May, August, and October remain
negative.  The gain must therefore be tested on another untouched era and with
the reliability/conversion layer still outcome-free at the held boundary.

Artifacts:

- [frozen cluster contract](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_frozen_cluster_residual_20260812_v1/frozen_cluster_contract.json);
- [frozen OOS predictions](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_frozen_cluster_residual_20260812_v1/frozen_cluster_oos_predictions.parquet);
- [frozen feature selection](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_frozen_cluster_residual_20260812_v1/frozen_cluster_feature_selection.parquet);
- [frozen metrics](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_frozen_cluster_residual_20260812_v1/frozen_cluster_metrics.parquet);
- [frozen runner](/Users/remyroche/Documents/Ares/scripts/run_tp6_sl4_frozen_cluster_residual.py).

## Learned per-cluster reliability correction

The stable cluster layer was followed by a dedicated conversion learner.  For
each frozen cluster, the reliability target is:

```text
OOF soft-cluster residual target − OOF cluster-model prediction
```

The development predictions are leave-one-month-out within April–November
2024; April is excluded from the reliability fit because no earlier scored
month exists.  The reliability learner then scores 2025 only.  It receives the
same base/path/trust fields and a train-only CMI-selected subset of the full
313-field causal/meta pool.  Held-month outcomes never enter its fit or
feature selection.

| Arm | Top 1% net | Top 5% net | Top 10% net | Top-5 monthly mean | Worst month | Positive months |
|---|---:|---:|---:|---:|---:|---:|
| Base control | −106.56 | +2.57 | −8.94 | +2.58 | −167.81 | 6/12 |
| Frozen cluster | −103.32 | +6.81 | −4.58 | +3.84 | −186.99 | 7/12 |
| Reliability, α=0.25 | −103.32 | +2.68 | −6.63 | +6.73 | −204.67 | 7/12 |
| Reliability, α=0.50 | −103.32 | +6.77 | −3.69 | +4.61 | −204.67 | 7/12 |
| Reliability, α=0.75 | −110.92 | **+10.39** | **−3.66** | +1.99 | −204.67 | 7/12 |
| Reliability, α=1.00 | −148.66 | +13.36 | −3.85 | +1.00 | −204.67 | 7/12 |

The reliability learner improves the pooled top-5 point estimate at α≥0.50,
but it does not improve the worst month and the strongest variants damage the
top-1 tail.  It is therefore **not promoted**.  Its useful output is a
diagnostic per-cluster conversion-error feature; the remaining issue is
regime transport/admission, not the absence of a residual target.

Artifacts:

- [development OOF cluster predictions](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_frozen_cluster_reliability_20260812_v1/development_cluster_oof_predictions.parquet);
- [reliability OOS predictions](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_frozen_cluster_reliability_20260812_v1/frozen_cluster_reliability_oos_predictions.parquet);
- [reliability metrics](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_frozen_cluster_reliability_20260812_v1/frozen_cluster_reliability_metrics.parquet);
- [reliability feature selection](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_frozen_cluster_reliability_20260812_v1/reliability_feature_selection.parquet);
- [reliability runner](/Users/remyroche/Documents/Ares/scripts/run_tp6_sl4_frozen_cluster_reliability.py).

## Earlier-era transport audit

To test whether the 2025 gain was an era artifact, a separate contract was
discovered using only label-matured April 2024 rows (832 rows), then frozen
while May–November 2024 were scored.  The full 313-field pool was still
available before CMI and the same stable-contract procedure was used.

| Arm | Pooled top-2% net | Pooled top-5% net | Pooled top-10% net | Top-5 monthly mean | Worst month | Positive months |
|---|---:|---:|---:|---:|---:|---:|
| Base control | +11.34 | **+3.27** | −26.82 | −23.08 | −90.14 | 3/7 |
| Frozen cluster residual | −2.40 | −12.03 | −23.95 | −12.41 | −122.32 | 4/7 |

The frozen cluster contract modestly improves the unweighted monthly mean and
positive-month count, but loses 15.30 bps/trade at the pooled global top-5 and
worsens the worst month.  This is a cross-era transport failure: the positive
2025 top-5 result is not sufficient evidence of a general economic repair.

Transport artifact: [April-discovered contract replay](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_frozen_cluster_transport_20260812_v1/).

## Artifacts

- [OOF regime sidecar](/Users/remyroche/Documents/Ares/data_perp/artifacts/oof_causal_market_regime_systems_2023q3_2025_20260811_v1/)
- [Augmented canonical meta pool](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_canonical_meta_pool_regime_20260811_v1.parquet)
- [Augmented cluster replay](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_conditional_cluster_residual_20260811_regime_fullpool_v1/)
- [Augmentation script](/Users/remyroche/Documents/Ares/scripts/augment_tp6_sl4_canonical_meta_pool_with_oof_regime.py)
- [Cluster runner](/Users/remyroche/Documents/Ares/scripts/run_tp6_sl4_conditional_cluster_residual.py)
