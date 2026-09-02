# Strict-R3 meta-layer optimisation decision — 2026-08-27

**Status:** `RESEARCH_ONLY_NOT_LIVE`
**Scope:** long-only, offline strict-prequential research. This document does
not change the deployed trader, the incumbent canonical handover, inference,
admission, or exchange behavior.

## Decision

Retain the existing live/canonical stack unchanged. The qualified meta-layer
research challenger is **U only**:

```text
incumbent_upstream_bps = 0.50 × efficiency_bps + 0.50 × timing_bps
U = unexpected-trailing / under-confidence LambdaRank coordinate
MC1 inputs = existing parent coordinates + U timestamp rank
admission = separate Current and BCF MC1 maps, both >= +50 bps
auction = unchanged chronological constrained portfolio, priority BCF MC1 EV
```

U is the parsimonious risk-adjusted winner on the Apr--Jul 2026 downstream
block. It improves EV/trade and drawdown with an economically immaterial
(-0.20%) reduction in total bps. The signed-state head `C` has the highest
total bps, but fails the predeclared no-worse-worst-day guardrail. It is
retained as diagnostic evidence only.

There is no target-free August 2026 source/feature panel for a genuinely later
untouched test. Consequently neither U nor C is promoted to the canonical or
live stack.

## Fixed base and causal inputs

The base is not retuned in this work:

```text
incumbent_upstream_bps = 0.50 × efficiency_bps + 0.50 × timing_bps
```

Each meta model receives its selected causal features plus the target-free base
geometry: enhanced/base, efficiency, timing, timestamp rank, query count,
query dispersion/range, top gaps, E--T disagreement, and base-component
dispersion. Scores are persisted before policy outcomes are joined.

The full candidate universe contained 1,407 numeric source fields; 1,094 pass
the coverage/variance hygiene gate. Final contracts use 50 selected causal
fields per head. The selection process combines coverage hygiene, strict-OOF
conditional IC/CMI, redundancy vetoes, randomized shallow-subspace evidence,
and full-model stability screening. Feature selection is recorded in:

```text
data_perp/artifacts/strict_r3_incumbent_meta_fullfeatures_selection_20260827_v3/
```

## Protocol

| Stage | Development evidence | Held evidence | Rule |
|---|---|---|---|
| Target/query and fine target/gain/truncation selection | earlier strict-OOF screens | no live authority | retain one predeclared winner per family |
| Feature selection and model HPO | Sep/Nov 2025, Jan/Mar 2026 | Apr--Jul 2026 reserved for MC1 | target-free scores, four complete-month fit histories plus 28-day reserve |
| MC1 and portfolio ablation | Apr--Jul 2026 | selected block, not untouched after selection | separate Current/BCF maps, dual +50 bps gate, one chronological constrained portfolio |
| Promotion | unavailable | no post-July source panel | requires a newly materialised later period |

All final MC1 fits use only prior resolved labels. Each Current and BCF family
has four scored held months, mean 206,942.5 training rows and minimum 202,381
training rows. Target-free identity rows are exact across the two family panels:
382,326 rows over 11 score months.

## Final head contracts and HPO

| Role | Family / target | Query | Features | HPO winner |
|---|---|---|---:|---|
| R | residual economic magnitude, sqrt-ATR quintiles | base-score band × 28-day block | 50 | depth 5 / 31 leaves, LR .05608, min-support .00770, no truncation |
| U | unexpected trailing opportunity beyond 1 ATR | timestamp | 50 | depth 2 / 3 leaves, LR .03774, min-support .00437, truncation 20 |
| O | adverse / over-confidence beyond 1.25 ATR | timestamp | 50 | depth 2 / 3 leaves, LR .03080, min-support .01634, truncation 5 |
| C | signed accurate/under/over-confidence state | timestamp | 50 | depth 5 / 31 leaves, LR .05243, min-support .00464, truncation 20 |

All models use 1,200 trees as an early-stop ceiling, a causal final-20%-of-query
early-stop split, calendar-stable fold seeds, and native LambdaRank. Over-score
orientation is explicitly reversed before ranking so higher rank means lower
adverse-surprise risk. The HPO scorer and final OOF scorer have exact raw-score
and rank parity for R/O/C on 137,027 development rows each; U has exact parity
on 382,326 rows against its prior winner receipt.

## Single-head downstream results

All values are realised canonical rich-policy net bps after the unchanged
dual-MC1 admission and constrained chronological auction, Apr--Jul 2026.

| Meta input | Entries | MC1-admitted | Net bps/trade | Total net bps | Worst month | Worst week | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| No meta control | 2,526 | 13,616 | +169.06 | +427,051 | +132.46 | +95.15 | -19.98% |
| R | 2,532 | 13,710 | +166.76 | +422,238 | +131.81 | +93.31 | -19.69% |
| **U** | **2,469** | **12,943** | **+172.62** | **+426,207** | **+132.76** | **+87.54** | **-13.91%** |
| O | 2,523 | 13,756 | +167.12 | +421,651 | +125.98 | +84.18 | -20.63% |
| C | 2,543 | 13,613 | +170.73 | +434,177 | +133.55 | +95.04 | -14.07% |

U versus control: +3.56 bps/trade, -844 total bps (-0.20%), -57 entries,
+0.29 bps worst month, -7.61 bps worst week, and +6.07pp less drawdown.

C versus control: +1.67 bps/trade, +7,126 total bps (+1.67%), +17 entries,
+1.09 bps worst month, -0.11 bps worst week, and +5.91pp less drawdown.
However C's worst daily mean is -82.47 bps versus -55.95 for control, so it
does not satisfy the predeclared downside guardrail.

## U portability and substitution diagnostics

| Measure | Control | U |
|---|---:|---:|
| Positive weeks | 18 / 18 | 18 / 18 |
| Worst daily mean | -55.95 | -5.95 |
| 5th-percentile daily mean | +54.30 | +56.14 |
| Worst trade | -600 | -600 |
| 5th-percentile symbol-day sum | -223.18 | -200.91 |
| Symbols traded | 112 | 112 |
| Largest symbol trade share | 2.97% | 2.99% |
| Trade-count HHI | .01537 | .01564 |

Post-selection diagnostic only, among dual-MC1-admitted candidates ranked by
BCF mapped EV within timestamp:

| Top-k | U-only | Control-only | Shared | U realised bps | Control realised bps | U delta |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 334 | 280 | 2,018 | +263.39 | +259.24 | +4.15 |
| 2 | 571 | 468 | 3,705 | +223.12 | +225.89 | -2.77 |

These statistics never enter fitting, mapping, admission, or the auction.

## Full R/U/O/C combination grid and parsimony

The full combination is not selected: it has +172.31 bps/trade but only
+419,060 total bps and -21.52% drawdown. Key alternatives:

| Combination | Entries | Net bps/trade | Total net bps | Max drawdown |
|---|---:|---:|---:|---:|
| C | 2,543 | +170.73 | +434,177 | -14.07% |
| R + C | 2,542 | +170.77 | +434,087 | -14.07% |
| U | 2,469 | +172.62 | +426,207 | -13.91% |
| R + U | 2,465 | +173.17 | +426,859 | -13.85% |
| R + U + O + C | 2,432 | +172.31 | +419,060 | -21.52% |

For the required full-combination leave-one-out diagnostic,
`Delta_X = Score(R,U,O,C) - Score(without X)`:

| Removed role | Delta entries | Delta EV/trade | Delta total bps | Delta worst month | Delta worst week | Delta max-DD |
|---|---:|---:|---:|---:|---:|---:|
| R | -5 | +0.09 | -632 | +0.06 | +2.05 | +0.00pp |
| U | -76 | +2.98 | -5,617 | +1.28 | -0.76 | -0.22pp |
| O | -10 | -1.55 | -5,503 | +1.09 | +0.38 | -0.66pp |
| C | -43 | +0.34 | -6,569 | +3.03 | +4.79 | -7.61pp |

The full grid is descriptive evidence; the accepted U decision is based on
the primary portfolio/risk criteria and the smallest sufficient contract, not
on an attempt to retain every potentially helpful feature coordinate.

## Artifacts and implementation

```text
config/strict_r3_incumbent_meta_family_hpo_finalists_20260827_v1.json
data_perp/artifacts/strict_r3_incumbent_meta_family_hpo_finalist_scores_20260827_v1/
data_perp/artifacts/strict_r3_incumbent_meta_family_hpo_mc1_combinations_20260827_v1/
data_perp/artifacts/strict_r3_incumbent_meta_under_u01_feature_screen_scores_20260827_v3/
data_perp/artifacts/strict_r3_incumbent_meta_under_u01_feature_screen_mc1_20260827_v1/
scripts/run_strict_r3_incumbent_meta_under_hpo_v1.py
scripts/score_strict_r3_incumbent_meta_selected_contracts_v1.py
scripts/run_strict_r3_incumbent_meta_mc1_combinations_v1.py
```

`run_strict_r3_incumbent_meta_under_hpo_v1.py` is now a generic,
predeclared-family HPO runner despite its historical filename. It accepts only
an exact arm from an immutable candidate configuration. The final scorer's
summary schema was corrected to use its emitted `family` and `query` fields.

## Required next validation

1. Materialise a target-free August-or-later candidate/source/feature panel
   with the same frozen 50/50 E/T upstream and full causal feature contract.
2. Freeze the U contract above, retrain only from prior resolved data, and
   evaluate the unchanged dual-MC1 +50 bps admission and portfolio rule.
3. Require no material loss versus control in total bps, no worst-day or
   trade-cluster degradation, and confirmation of U-only admission quality
   before any canonical or live promotion.
