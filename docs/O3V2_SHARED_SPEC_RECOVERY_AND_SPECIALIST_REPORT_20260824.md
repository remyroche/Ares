# O3-v2 shared-spec recovery and specialist funnel

## Scope and status

The formerly unavailable shared specification was recovered through the raw
public shared-page payload, rather than the normal shared-link renderer.  This
document records the implemented offline research portion only.  Nothing here
changes live models, live bundles, MC1, admissions, policy, portfolio rules,
or exchange processes.

MDA is intentionally not included: it remains assigned to the separate user
pipeline.

## Frozen causal protocol

Every held score is written before any policy outcome join.  Specialist folds
use:

- top-30% timestamp-local base routing;
- three complete prior calendar months of resolved training data;
- a 28-day reserve excluded from fitting;
- policy and semantic labels whose availability timestamp is before the
  reserve boundary;
- training-distribution CDF references for specialist ranks;
- canonical reconciled rich-policy labels only after held scores are sealed.

The selected F1--F6 feature panels are parent-only: F5 contains target-free
current/BCF correction provenance but excludes a later O3 score.  That removes
the former circular score dependency and makes the specialist contract usable
from the first parent-score history month.  Coverage is 100% of routed rows in
every specialist held month.

## Completed target and support funnel

The strict 6-month/28-day target funnel retained:

- **T2 economic residual**, trained with the retained `SB3_error_semantic`
  support contract;
- **T6 rank error**, trained with `SB2_error_policy_state` support.

T6 family specialists fail their standalone OOS diagnostic and are rejected.
T2 is the only target advanced to the specialist stage.

## Specialist results

Selection was made on November--December 2025.  Later January--July 2026
metrics are retained for temporal portability, not re-selection.

### H1 family heads: T2

The two development-selected families are:

- **F4 state/transition**: selected causal market-state fields;
- **F5 parent-score provenance**: target-free current/BCF score,
  correctness, anchor, and disagreement coordinates.

Their ranks are complementary (mean Spearman approximately 0.20).  On the
nine held months, standalone rank diagnostics were:

| Head | Top-1% | Top-2% | Top-5% | Worst Top-5 month | Mean rank IC |
|---|---:|---:|---:|---:|---:|
| F4 state/transition | +328.82 | +284.87 | +180.51 | +49.46 | 0.073 |
| F5 parent provenance | +309.83 | +278.00 | +204.11 | +74.39 | 0.082 |
| Fixed 50/50 F4/F5 | +339.55 | +306.29 | +210.37 | +53.94 | — |

The all-family median is weaker because F2/F3/F6 are non-incremental or
adverse.  They are not carried into downstream candidates.

### H3 hybrid: T2 F4+F5

The predeclared hybrid trains one bounded head over the frozen union of F4 and
F5 features.  Across the same nine held months it produces:

| Top-1% | Top-2% | Top-5% | Worst Top-5 month | Mean rank IC |
|---:|---:|---:|---:|---:|
| +306.14 | +272.90 | +183.20 | +42.80 | 0.071 |

### H2 population heads: T2 F4+F5

Five predeclared population weighting heads were evaluated.  Development-only
selection chose `equal_archetype`; it retained the strongest later held
diagnostic among H2 variants.  The heads are highly redundant (pairwise rank
correlations 0.91--0.98), so only the equal-archetype head was sent to the
downstream test.

## Matched MC1 dual-admission and portfolio test

All arms below use the exact same May--July 2026 target-free parent population,
strict prequential MC1 class, dual current/BCF MC1 admission, canonical rich
policy labels, and constrained portfolio replay.  The comparison is limited
to these three months because a six-month prior homogeneous specialist score
ledger is required before MC1 can score a month.

| Arm | Threshold | Trades | Net bps/trade | Total net bps | Worst month | Worst week | Max DD | Total / |DD| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Matched current-stack control | 30 | 2,225 | +131.68 | +292,994 | +111.77 | +81.70 | −18.76% | 1.562m |
| H1 F4/F5 adapter | 30 | 2,364 | +129.67 | +306,537 | +116.16 | +86.96 | −24.05% | 1.275m |
| H3 F4+F5 hybrid | 30 | 2,339 | +127.53 | +298,290 | +113.15 | +90.26 | −20.22% | 1.476m |
| H2 equal-archetype | 30 | 2,337 | +128.59 | +300,506 | +116.59 | +93.42 | −24.34% | 1.234m |
| Matched current-stack control | 50 | 1,896 | +151.31 | +286,875 | +127.85 | +97.11 | −17.87% | 1.606m |
| H1 F4/F5 adapter | 50 | 1,975 | +146.77 | +289,868 | +134.53 | +110.73 | −19.45% | 1.490m |
| **H3 F4+F5 hybrid** | **50** | **1,991** | **+145.49** | **+289,678** | **+133.98** | **+110.33** | **−14.49%** | **1.999m** |
| H2 equal-archetype | 50 | 1,991 | +143.28 | +285,266 | +128.87 | +109.81 | −17.33% | 1.646m |

## Decision

- **Do not promote any arm to live/canonical.**
- Reject T6 specialists, the full six-family median, and the H2
  equal-archetype arm as production candidates.
- Retain **T2 H3 F4+F5 at the 50-bps threshold** as a research challenger:
  it sacrifices 5.81 bps/trade versus control but preserves a slightly higher
  total net contribution and improves total-net-to-drawdown by about 24.5%.
- The H1 F4/F5 adapter is not retained: its additional total bps come with a
  materially worse drawdown.

The next valid test is a later, untouched period.  It should keep the H3
contract, MC1 class, 50-bps threshold, policy, and portfolio constraints
fixed, then determine whether the risk-adjusted improvement transports.

## Relevant implementation and artifact paths

- `scripts/run_strict_r3_o3v2_feature_screen.py`
- `scripts/run_strict_r3_o3v2_specialist_funnel.py`
- `scripts/materialize_strict_r3_o3v2_t2_f4f5_adapter.py`
- `scripts/run_strict_r3_o3v2_mc1_portfolio.py`
- `tests/test_strict_r3_o3v2_contract.py`
- `data_perp/artifacts/strict_r3_o3v2_feature_screen_t2_parent_only_20260824_v1`
- `data_perp/artifacts/strict_r3_o3v2_specialist_t2_parent_only_20260824_v1`
- `data_perp/artifacts/strict_r3_o3v2_specialist_t2_h3_f4f5_20260824_v2`
- `data_perp/artifacts/strict_r3_o3v2_specialist_t2_h2_population_20260824_v2`
- `data_perp/artifacts/strict_r3_o3v2_t2_h3_f4f5_mc1_portfolio_20260824_v1`

Earlier failed startup-only artifacts are preserved as incomplete receipts and
must not be used for comparison: the pre-fix H3 run, first F4/F5 adapter
layout, and first H2 startup root.
