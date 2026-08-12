# TP6/SL4 transport-first structural chain and GAM handover

Run: `data_perp/artifacts/tp6_sl4_archetype_cluster_vcgam_oof_20260814_v3`

Side: long.  Development is the 2024-04 through 2024-11 population; the OOF
evaluation is 2025-01 through 2025-12.  December 2024 is absent from the
source.  The exit contract is the existing TP6/SL4 panel; reported gross and
net values are the realised per-trade values in that panel.

## 1. What was integrated

The old structural branch used fold-local family IDs.  Those IDs do not have
stable meaning after refitting.  The new chain is:

```text
raw fitted tree
  -> raw leaf/path rule
  -> recurrent structural archetype
  -> soft top-3 archetype transport + explicit unmatched mass
  -> co-firing/economic cluster
  -> cluster-specific varying-coefficient GAM
  -> base expected bps + gamma * GAM residual
```

The contract is frozen before any 2025 outcomes are used.  Archetypes are
selected from recurrence across independent 2024 fits, not by reusing a
fold-local family ID.  Each new leaf receives at most three soft archetype
matches.  Unmatched contribution is retained rather than silently assigned to
the nearest archetype.

The principal GAM is constrained to zero exposure:

```text
residual_hat_i,k = exposure_i,k * (beta_k + spline(context_i))
score_i = base_expected_bps_i + gamma * sum_k membership_i,k * residual_hat_i,k
```

The explicit-intercept form is retained only as an ablation.  Cluster
membership is a weight/exposure; it is never multiplied into the target.

## 2. Raw rules, leaves, and paths

| Quantity | Value |
|---|---:|
| Independent monthly model fits | 20 |
| Fits covered | 2024-04..11, 2025-01..12 |
| Candidate rows | 17,040 (852 per fit) |
| Leaf-assignment columns per fit | 64 |
| Raw leaf/path catalogue rows | 18,122 |
| Unique structural signatures | 17,594 |
| Signatures recurring in at least 2 fits | 207 |
| Signatures recurring in at least 3 fits | 61 |
| Recurrent archetypes accepted | 11 |

The archetype gate was: recurrence in at least three fits, a separated-fold
gap of at least two, median training-leaf frequency at least 0.5%, and signed
tree-contribution consistency at least 80%.  All 11 accepted archetypes had
100% sign consistency and recurred in 3–7 fits.

### Accepted archetype semantics

The labels below are descriptive summaries of the actual path predicates;
they are not hand-coded semantic views.

| Archetype | Recurrence | Signed tree contribution | Median abs contribution | Median train frequency | Path predicate(s) |
|---|---:|---:|---:|---:|---|
| 0000 | 7 | − | 0.0158 | 0.069 | `flush_recovery_state` right band 2/3 |
| 0001 | 6 | + | 0.0148 | 0.056 | `asset_flush_exhaustion_score` left band 0/4 |
| 0002 | 5 | + | 0.0115 | 0.076 | `fund_abs_z_mkt_resid` right band 3/4 |
| 0003 | 5 | + | 0.0145 | 0.056 | `pct_assets_above_ema_fast` left band 0/4 |
| 0004 | 5 | + | 0.0147 | 0.040 | `pct_assets_above_intraday_vwap` left band 0/4 |
| 0005 | 4 | − | 0.0132 | 0.130 | `pct_assets_above_intraday_vwap` right band 3/4 |
| 0006 | 3 | + | 0.0274 | 0.039 | `price_down_oi_down_4h_rz` right; `hours_since_funding_sign_flip_24h_norm` right; `post_liquidation_rebound_score` left |
| 0007 | 3 | − | 0.0026 | 0.053 | `price_down_oi_down_4h_rz` right; funding-sign-flip field left |
| 0008 | 3 | − | 0.0150 | 0.171 | `post_flush_leverage_rebuild` right band 2/3 |
| 0009 | 3 | + | 0.0152 | 0.047 | `flush_recovery_state` left; `pct_assets_above_ema_fast` left |
| 0010 | 3 | − | 0.0101 | 0.081 | `coherence_24_ts_resid` left band 0/4 |

The signed contribution is the tree-path contribution sign, not a claim that
the feature is intrinsically bullish or bearish.  Economic effects were
measured separately on development rows.

## 3. Leaf-to-archetype transport and semantics quality

Soft matching used top-3 candidates, temperature 0.08, and an unmatched
threshold of 0.55.  The path score is 50% structural token Jaccard, 20%
interval-band agreement, 15% contribution magnitude, and 15% activation
frequency.  A contribution-weighted row exposure is then formed.

| Transport statistic | p10 | Median | p90 |
|---|---:|---:|---:|
| Best leaf similarity | 0.209 | 0.279 | 0.373 |
| Unmatched leaf probability | 0.322 | 0.493 | 0.621 |

Across all 20 fits, mean row matched mass was 0.514 (median 0.512); mean
unmatched mass was therefore about 0.486.  Development matched-mass means
ranged from 0.390 in 2024-04 to 0.571 in 2024-09.  Held-out 2025 means ranged
from 0.535 in January to 0.483 in December.  Virtually no row reached 0.75
matched mass (aggregate mean 0.02%); this is the main transport weakness.

The cluster stage uses a 0.10 archetype-exposure activation threshold for
co-firing statistics.  This is important: without it, soft top-3 probabilities
make almost every archetype technically nonzero and co-firing degenerates.

## 4. Cluster discovery and own metrics

Cluster selection used co-firing Jaccard/NPMI, signed contribution coherence,
economic coherence, balance, and chronological validation differentiation.
The selected contract was K=6.

| K | Silhouette | Selection score | Balance | Max mass | Min mass | Validation abs differentiation | Support | Valid |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 0.048 | −0.141 | 0.844 | 0.729 | 0.271 | 0.54 bps | 1.00 | No (balance) |
| 3 | 0.034 | −0.233 | 0.704 | 0.729 | 0.134 | 4.42 bps | 1.00 | No (balance) |
| 4 | 0.024 | 0.280 | 0.789 | 0.612 | 0.117 | 4.56 bps | 1.00 | Yes |
| 5 | 0.014 | 0.332 | 0.938 | 0.353 | 0.117 | 4.98 bps | 1.00 | Yes |
| 6 | **0.015** | **0.352** | **0.976** | 0.259 | 0.117 | 4.47 bps | **1.00** | **Yes / selected** |

The low silhouettes indicate that this is a weak geometric partition even
though it satisfies the balance/support gates.  The six clusters contain 4,
2, 2, 1, 1, and 1 archetypes respectively.  The three multi-archetype
clusters have mean pair similarity 0.390, 0.409, and 0.409; singleton
clusters have no within-cluster pair to validate.

Cluster-level OOF ordering (active rows only) was:

| Cluster | Zero-at-exposure mean active rank IC | Zero GAM Q5−Q1 net (mean bps) | Months with positive Q5−Q1 |
|---|---:|---:|---:|
| 00 | +0.0344 | +7.84 | 6/12 |
| 01 | +0.0194 | −6.75 | 6/12 |
| 02 | −0.0120 | +3.72 | 6/12 |
| 03 | +0.0138 | −15.26 | 5/12 |
| 04 | +0.0224 | +1.07 | 5/12 |
| 05 | +0.0185 | −3.33 | 5/12 |

Thus the required cluster-level gate is not met: the correction is not
positive and stable across all clusters.  Several clusters are directionally
reversed, and each cluster has a negative mean realised net among its active
rows (roughly −49 to −52 bps), so the GAM is trying to rank residuals inside a
population that is still economically poor on average.

## 5. GAM inputs and fitting

The available causal meta/context pool had 327 columns.  For every held month
and cluster, the selector computed a weighted binned-CMI proxy on prior rows
and selected at most 12 fields.  There were 72 cluster-month fits, 864
selected feature slots, and 36 unique selected fields.

The most persistent inputs were:

`rv_24h_peer_resid`, `q_iqr__price_x_oi_1d`, `median_spread_bps`,
`median_alt_minus_btc`, `avg_pair_corr_24h`, `cross_asset_corr_4h`,
`asset_flush_exhaustion_score`, `cross_asset_corr_1h`,
`breadth_dispersion`, `ob_depth_l10_to_qv_24h`, and 90/180-day dependence or
dispersion ranks.  Regime probabilities/state-age fields were selected when
their CMI was incremental for that cluster/month; they were not injected by
name or treated as universal semantics.

Inputs are train-standardised using training medians and MAD-like scales,
clipped to [-20, 20], then passed through a three-knot quadratic spline.  The
cluster's contribution exposure is multiplied into the spline design.  It is
also the sample weight through cluster membership.  This gives the principal
model the exact property that a row with no transported cluster exposure gets
no cluster correction.

Target:

```text
ordinary residual = realised net_bps − base_expected_bps
```

There is no multiplication of the target by membership and no future-context
field in the selector.

## 6. GAM/global OOF metrics

Global ranking is after score generation and uses all 10,224 2025 rows.  The
principal arm is the zero-at-exposure GAM.  Gamma is the multiplier on the
aggregated GAM residual.

| Arm | Tail | Trades | Gross bps/trade | Net bps/trade | Rank IC |
|---|---:|---:|---:|---:|---:|
| Base | 0.5% | 52 | 1.88 | −98.12 | 0.0636 |
| Zero GAM γ=.25 | 0.5% | 52 | 20.66 | −79.34 | 0.0639 |
| Zero GAM γ=.50 | 0.5% | 52 | 19.76 | −80.24 | 0.0644 |
| Zero GAM γ=1.00 | 0.5% | 52 | 4.11 | −95.89 | 0.0645 |
| Base | 1% | 103 | −6.56 | −106.56 | 0.0636 |
| Zero GAM γ=.25 | 1% | 103 | −8.17 | −108.17 | 0.0639 |
| Base | 2% | 205 | 67.24 | −32.76 | 0.0636 |
| Zero GAM γ=.25 | 2% | 205 | 71.07 | −28.93 | 0.0639 |
| Base | 5% | 512 | 102.57 | **+2.57** | 0.0636 |
| Zero GAM γ=.25 | 5% | 512 | 100.97 | **+0.97** | 0.0639 |
| Base | 10% | 1,023 | 91.06 | −8.94 | 0.0636 |
| Zero GAM γ=.25 | 10% | 1,023 | 91.97 | −8.03 | 0.0639 |
| Base | 20% | 2,045 | 70.46 | −29.54 | 0.0636 |
| Zero GAM γ=.25 | 20% | 2,045 | 71.87 | −28.13 | 0.0639 |

The zero GAM slightly improves top-0.5%, top-2%, top-10%, and top-20% net,
but gives back the baseline's positive top-5% net and remains negative at
every other meaningful tail.  The larger gamma arms do not repair this.

The intercept ablation is not admissible as the principal architecture.  Its
best-looking global result is still negative at top-5% (−3.72 bps for γ=1),
and it can assign a correction to an unmatched row.

### Monthly top-5% stability

| Arm | Mean | Median | Worst month | Positive months | Portability score | Mean monthly rank IC |
|---|---:|---:|---:|---:|---:|---:|
| Base | +2.58 | +6.45 | −167.81 | 6/12 | −208.48 | 0.06965 |
| Zero GAM γ=.25 | −19.71 | +7.17 | −170.99 | 6/12 | −233.73 | 0.06909 |
| Zero GAM γ=.50 | −19.71 | +7.17 | −170.99 | 6/12 | −233.73 | 0.06965 |
| Zero GAM γ=1.00 | −19.90 | +2.67 | −170.99 | 6/12 | −235.88 | 0.06960 |
| Intercept γ=1 | −12.46 | +20.38 | −169.32 | 7/12 | −223.54 | 0.06957 |

The mean is pulled down by January, March, May, August, and October.  The
positive months do not form a stable contiguous regime; the correction is
therefore not portable enough to promote.

## 7. Decision

The implementation is complete and the correctness checks pass, but the
structural branch does **not** advance the current stack.

What worked:

- recurrence-first archetypes eliminate dependence on fold-local family IDs;
- explicit unmatched mass exposes the true transport problem;
- the principal GAM obeys zero-at-exposure exactly;
- cluster support and chronological contract gates pass;
- top-0.5% and some broader tails improve slightly at small gamma.

What failed:

- transport quality is weak: about half of contribution mass is unmatched;
- archetype co-firing geometry is only weakly separated (selected silhouette
  0.015);
- cluster-level Q5−Q1 ordering is negative for three of six clusters and is
  not month-stable for the others;
- pooled top-5 net decreases from +2.57 to +0.97 bps and monthly portability
  becomes more negative.

The next repair should therefore be transport quality, not more GAM HPO:

1. Raise archetype recall with structural neighbourhoods or learned path
   embeddings while retaining recurrence/sign gates.
2. Add an explicit hard-top-1 activation channel alongside soft probabilities;
   use soft exposure for the GAM but hard/thresholded activation for co-firing.
3. Reject singleton clusters unless they recur in separated eras and have
   independent economic differentiation.
4. Re-run cluster discovery only after matched contribution mass is materially
   above the current ~0.51 mean.

## 8. Artifacts

- [Detailed handover report](TP6_SL4_ARCHETYPE_CLUSTER_VCGAM_20260814_HANDOVER.md)
- [Run report](../data_perp/artifacts/tp6_sl4_archetype_cluster_vcgam_oof_20260814_v3/TP6_SL4_ARCHETYPE_CLUSTER_VCGAM_REPORT.md)
- [Archetype contract](../data_perp/artifacts/tp6_sl4_archetype_cluster_vcgam_oof_20260814_v3/archetype_contract.json)
- [Cluster contract](../data_perp/artifacts/tp6_sl4_archetype_cluster_vcgam_oof_20260814_v3/cofiring_cluster_contract.json)
- [Transport audit](../data_perp/artifacts/tp6_sl4_archetype_cluster_vcgam_oof_20260814_v3/archetype_transport_by_month.parquet)
- [Cluster candidate audit](../data_perp/artifacts/tp6_sl4_archetype_cluster_vcgam_oof_20260814_v3/cluster_candidate_audit.parquet)
- [Cluster own metrics](../data_perp/artifacts/tp6_sl4_archetype_cluster_vcgam_oof_20260814_v3/cluster_own_metrics.parquet)
- [GAM context selection](../data_perp/artifacts/tp6_sl4_archetype_cluster_vcgam_oof_20260814_v3/cluster_gam_context_selection.parquet)
- [Global metrics](../data_perp/artifacts/tp6_sl4_archetype_cluster_vcgam_oof_20260814_v3/metrics_global.parquet)
- [Monthly stability](../data_perp/artifacts/tp6_sl4_archetype_cluster_vcgam_oof_20260814_v3/metrics_stability.parquet)
- [Correctness report](../data_perp/artifacts/tp6_sl4_archetype_cluster_vcgam_oof_20260814_v3/correctness_test_report.json)
