# Continuous-context successor audit — 2026-08-03

## Decision

Discrete latent-state memberships, state age, transition simplex, centroid
distances, and state priors are diagnostic-only. They are no longer in base or
meta shared feature allowlists.

## Implementation

The new lightweight sidecar covers 237,246 exact residual-OOF candidates from
2023-09 through 2024-12. It has 63 float32 meta-only fields: strict-prequential
90/180-day rank and z score, 4/24-hour change, and 30-day median distance for
nine continuous observables (trend, volatility, breadth, dependence, leverage,
funding, spread, turnover, dispersion). Every field has 100% coverage. It
performs a backward candidate identity join with `source_utc <= __ts__`, fits
no GMM, and reads no labels, outcomes, or candidate scores.

## Fixed residual-baseline results

All results are one pooled global ranking after common expected-net mapping.

| Arm | Top-1 net bps | Top-5 net bps | Top-10 net bps | Net IC |
|---|---:|---:|---:|---:|
| P0 frozen residual | +42.61 | -55.66 | -103.36 | 0.0916 |
| P2 Ridge, 9 fields + base interactions | +4.24 | -79.87 | -115.11 | 0.0795 |
| P3 spline + Ridge | +39.29 | -69.83 | -108.27 | 0.0798 |
| P4 shallow tree | +36.71 | -62.09 | -105.87 | 0.0897 |
| P2 sparse five-field Ridge | +7.82 | -78.92 | -111.26 | 0.0807 |
| P3 sparse five-field spline | +30.63 | -71.11 | -107.92 | 0.0796 |
| P4 sparse five-field tree | +44.97 | -64.32 | -104.86 | 0.0877 |

None advances. The sparse P4 arm has a small top-1 lift only; it loses top-5,
top-10, and IC. On the independent pre-2024 to 2024 split, top-10 net bps are
P0 -106.31, P2 -112.87, P3 -115.23, and P4 -108.61.

## Transport screen

Selection uses strict chronological, thresholded-net opportunity models,
held-out global-top-10 economic MDA, and a diagnostic-only calendar proxy. The
transport score is median cross-era MDA minus half its MAD; within-era MDA is a
separate gate.

| Field | Within MDA | Cross-era MDA | Score | Class |
|---|---:|---:|---:|---|
| volatility z90 | +14.98 | +5.52 | +3.02 | INVARIANT_CORE |
| dependence z90 | +5.22 | +7.69 | +7.14 | SMOOTHLY_CONDITIONED |
| leverage z90 | -0.16 | +5.48 | +5.12 | SMOOTHLY_CONDITIONED |
| spread z90 | +2.79 | +0.89 | +0.63 | SMOOTHLY_CONDITIONED |
| turnover z90 | +6.43 | +0.40 | -0.94 | SMOOTHLY_CONDITIONED |

No selected candidate was an era shortcut. Breadth, funding, trend-quality,
and dispersion z90 are rejected. Individual conditional information does not
make a broad residual reranker useful.

## Next valid family

Relationship breaks were then materialized as 16 meta-only fields: signed and
absolute prior-only rolling-OLS residuals at 30d and 90d for trend/breadth,
trend/turnover, volatility/dependence, and price/leverage. Two fields pass the
individual transport gate without an era shortcut: absolute price/leverage
break at 30d (within +18.17, cross-era +14.20 bps) and 90d (within +14.89,
cross-era +18.91 bps).

They still do not improve residual reranking when used as the only two context
inputs. The best P4 arm is +48.70 / -64.02 / -105.11 net bps at top 1/5/10
versus P0 +42.61 / -55.66 / -103.36; the pre-2024 to 2024 top-10 result is
-115.46 versus P0 -106.31. Thus the relationship-break representation is
retained as an individually useful candidate feature family, but no residual
baseline promotes. The complete fixed transport matrix confirms this: at
top-10, P4 is -115.46 versus P0 -106.31 on 2023Q4→2024 and -104.28 versus
P0 -84.44 on 2024H1→2024H2. P4's top-1 does improve in the second split
(+109.30 versus +87.33), but it cannot compensate for top-5/top-10 failure.

The next valid test is a reliability/error target that uses
these breaks to predict meta overestimation or correction-sign error, rather
than another direct expected-net location/reranking surface. Do not revisit
cluster K, stickiness, state priors, or monotonic trust shrinkage.

Focused tests pass: 13 tests across context generation, materialization,
transport selection, and residual baselines.

Artifacts: `causal_continuous_context_2023q3_2024_20260803_v1/`,
`continuous_context_residual_baselines_compact9_20260803_v1/`,
`continuous_context_transport_selection_compact9_20260803_v2/`, and
`continuous_context_residual_baselines_transport5_20260803_v1/`,
`continuous_relationship_break_transport_selection_20260803_v2/`, and
`continuous_context_residual_baselines_price_leverage_break_20260803_v2/`.
