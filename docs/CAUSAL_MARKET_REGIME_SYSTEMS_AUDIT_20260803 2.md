# Causal market-regime systems audit — 2026-08-03

## Scope and decision

This change materialises a compact, hourly, chronological-OOF regime sidecar
over 2023-09 through 2024-12.  It is a context/trust representation only: it
does not use outcomes, labels, scores, policy actions, or future path fields.
The candidate join is backward as-of and every candidate carries a strictly
earlier fold train-end timestamp.

The result is **not promoted as an admission feature set yet**.  It yields a
small pooled top-10 economic improvement but reduces rank IC and fails both
transport diagnostics.  The implementation and the data-quality diagnostics
are retained because they identify a concrete state-support problem rather
than silently treating unstable latent state IDs as reliable features.

## Implemented representation

At each hourly decision time the primary system emits a fixed five-state soft
posterior plus:

- `market_regime__state_p_0` … `state_p_4` (diagnostic only);
- entropy, top-two margin, state age, switch probability and train-reference
  OOD percentile;
- a forward-only action-facing simplex: stable / onset / active / settling,
  with entropy and top-two margin.

Four additional independently selected latent systems represent
trend-volatility, breadth-dependence, leverage-flow and liquidity.  Their
posterior coordinates are diagnostics only because a GMM component number is
fold-local.  The base feature allow-list contains only primary invariant
context and semantic phase probabilities (11 fields); the meta allow-list
adds the 20 geometry-specific invariant trust/context fields (31 fields
total).  Neither list admits a state ID, a raw posterior coordinate, a
delayed transition label, or an action field.

Inputs are pre-existing causal multiview features.  The materialiser reads
only its compact 76-field schema proxy from the 14,538-column source panel,
then uses a train-only median imputer, robust scaler clipped to +/-12 scaled
units, diagonal GMM, and forward recursion.  K is fixed at 5 for the primary
system; each geometry chooses K in {3,4,5,6} by a bounded training proxy.  A
label-free stickiness grid {0, .35, .60, .80} is ranked by confidence,
dwell, occupancy and temporal switching, with a structural persistence gate.

## Integrity and coverage

- 11,712 hourly rows and 237,246 exact candidate rows were materialised.
- Every selected context feature has 100% candidate coverage and is
  nonconstant.  Source input coverage is at least 99.85% in every
  system/fold.
- Unit/integration suite: 14 passed.  It covers simplex validity, strict
  causality under future mutation, forbidden outcome fields, exact candidate
  identity, and provenance/availability constraints.

## Matched OOF global-top-k evaluation

Evaluation uses the same 237,246 rows and one pooled global top-k after a
causal, trailing-180-day common-net-bps Ridge map.  It does not select a top
k per timestamp or per side.  Costs are approximately 100 bps/trade.

| Arm | Top 1% net bps | Top 5% net bps | Top 10% net bps | Top-10 net IC |
|---|---:|---:|---:|---:|
| A0 residual baseline | -7.29 | -78.19 | -107.95 | 0.0810 |
| A1 primary context | -5.41 | -72.97 | -102.46 | 0.0764 |
| A2 trend/volatility | -6.88 | -74.54 | -103.28 | 0.0761 |
| A2 breadth/dependence | +4.28 | -71.11 | -104.72 | 0.0711 |
| A2 leverage/flow | -11.19 | -76.01 | -102.32 | 0.0757 |
| A2 liquidity | -8.26 | -74.17 | -103.10 | 0.0770 |
| A3 all geometry | -7.33 | -72.99 | -105.64 | 0.0728 |

The best top-10 uplift is leverage/flow (+5.63 bps/trade), narrowly followed
by primary context (+5.48), but every regime arm lowers net IC.  Therefore no
arm satisfies the predeclared advancement requirement of positive top-10
economic uplift **and** positive IC uplift.  Breadth/dependence is a useful
research signal: it is the only arm with positive top-1 net (+4.28 bps), but
it does not scale to top-5/top-10.

## Portability and remaining gaps

Neither frozen cross-era transport test improves:

| Train -> test | Baseline top-10 net bps | Best regime arm | Best regime-arm net bps |
|---|---:|---|---:|
| 2023-Q4 -> 2024 | -117.00 | liquidity | -118.82 |
| 2024-H1 -> 2024-H2 | -75.33 | all geometry | -99.46 |

The primary system selects high stickiness (.80) in every chronological fold,
but fails the structural five-state persistence gate in all six: a component
has less than 2% occupancy in each fold (and median dwell is only five hours
in three).  This is a support/semantic-stability gap, not an economic tuning
failure.  Fold-local state diagnostics are written with the fold ID attached;
they must not be interpreted as state 0/1/… having a stable cross-era meaning.

The current stack also remains deeply negative after cost at top-10; the
features only make a modest gross-selection repair, not an executable policy.

## Recommended next research step

Keep the invariant continuous fields and phase representation available for
feature selection, but do not include them in a winner until a richer
candidate context can show joint IC/economic uplift.  For the five-state
system, compare a fixed K=3 control and a five-state model with a
minimum-occupancy-constrained re-fit/merge procedure.  Freeze semantic anchors
only if they are trained strictly before the evaluation era.  Then test the
primary and breadth systems within the existing residual-expert funnel using
per-side selected subsets, common-bps mapping, and the same pooled-global
ranking/transport gates.

## Artifacts

- `data_perp/artifacts/oof_causal_market_regime_systems_2023q3_2024_20260803_v1/`
- `data_perp/artifacts/regime_geometry_portability_ablation_20260803_v1/`

Both folders contain manifests, source hashes, fold diagnostics, coverage,
and the complete aggregate/monthly/side/phase/fold-local-state results.
