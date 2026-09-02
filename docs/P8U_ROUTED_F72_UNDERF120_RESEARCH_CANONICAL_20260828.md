# P8U Router50 + F72 Base + Under F120 — Research Canonical (Historical v5)

> **Superseded for canonical use.** The capacity-corrected handover and live-promotion boundary are in [P8U_ROUTED_F72_UNDERF120_RESEARCH_CANONICAL_V6_20260828.md](P8U_ROUTED_F72_UNDERF120_RESEARCH_CANONICAL_V6_20260828.md). This v5 file remains an immutable historical research record; its capacity metrics used marked exposure rather than the corrected committed-initial-margin reservation semantics.

## Status

This is the canonical **long-only research stack** as of 2026-08-28. It is
offline only. It does not change the live stack, inference bundle, execution
policy, admission threshold, exchange access, or portfolio state.

The versioned machine-readable successor contract is
[`strict_r3_p8u_routed_f72_underf120_research_canonical_20260828_v5.json`](../config/strict_r3_p8u_routed_f72_underf120_research_canonical_20260828_v5.json). It inherits frozen model semantics and parameters unchanged, and records the August-27 target-free reconciliation receipt separately from selected-period evidence.

## Latest matched layer audit (2026-08-28)

The current frozen contract has been re-audited using only its pre-trained,
target-free OOF score receipts and the canonical rich 15-minute policy labels
with smooth capital protection.  The full immutable receipt is
[`strict_r3_p8u_full_layer_audit_20260828_v8`](../data_perp/artifacts/strict_r3_p8u_full_layer_audit_20260828_v8/).
It is a research-only audit: it did not change this configuration, any live
bundle, or the exchange.

The evidence separates the layers rather than treating a higher total bps
number as automatic promotion:

| Finding | Evidence | Decision consequence |
|---|---|---|
| Router | Current P8U Top-50 recall is lower than its prior frozen P8U on the exact Jul-2025–Jul-2026 intersection: −1.13pp / −0.95pp / −0.72pp / −0.59pp for rich-policy opportunities above +50/+100/+150/+200 bps. | Do not claim a Router improvement. |
| F72 Base | On the exact Apr–Jul-2026 common population, F72 adds +44.72/+36.32/+21.76/+12.00 bps at timestamp-local Top-1/2/5/10%, with +3.32/+3.80/+1.99/+0.63pp `>+50` hit-rate change. | F72 remains the credible Base tip-ranking advance. |
| Under F120 | Conditional MI given F72 is 0.1384 nats versus 0.0330 nats for prior T6/T9.  Its blend raises `>+50` hit rate by 4.55–8.55pp across Top-1–10%, but sacrifices F72 raw net bps at those same cuts. | Treat Under as a conservative confirmation signal, not a standalone alpha replacement. |
| MC1 / portfolio | At the frozen +50 dual gate and two-entry timestamp capacity, the current route has +997,959 total bps across 7,600 entries versus legacy-live +817,520 across 5,919, but lower unit EV (+131.31 vs +138.12), weaker worst month/week, and −30.06% vs −28.27% drawdown. | Qualified research challenger only; no live promotion. |

At one new entry per timestamp, the current route is the cleanest comparison:
5,733 entries at +156.44 bps/trade versus 4,445 at +156.78 bps/trade for
legacy, adding about +200.0k total bps with a 0.78pp smaller drawdown.  The
larger two-to-four-entry capacities increase total contribution but lose unit
quality and worsen drawdown; their selection is therefore not justified by
this audit.

The original matched audit covers Router through July 2026, F72 Base through
July 2026, Under through July 2026, and dual MC1 from November 2025 through
July 2026 after its three-month resolved-label warm-up.  It is preserved as
the selected-period layer comparison.  The versioned target-free extension
below independently adds August rather than substituting an older live score
family for the missing period.

That prerequisite is now complete for signals through **2026-08-27**.  The
append-only source materialisation contains 55,080 Router50 candidates over
648 timestamps, including the terminal 28-August 00:00 decision generated
from the final August signal hour.  The declared evaluation cutoff removes
that terminal decision, leaving **54,995 candidates over 647 timestamps**.
Complete rich-policy labels were joined only after frozen F72 and Under scores
were persisted.  The extension carries strict Base/Under scores to August
2026 and dual-MC1/portfolio evidence from November 2025 to August 27.
Because August was reconstructed after configuration selection and F72 joint
observed input coverage is 82.38%, it is retrospective reconciliation
evidence—not untouched promotion evidence.

| Extended end-to-end result | Pre-August (Nov-25–Jul-26) | Through Aug-27 signal | Change |
|---|---:|---:|---:|
| Shared-portfolio entries | 7,600 | 8,461 | +861 |
| Dual-MC1 admitted rows | 27,382 | 29,474 | +2,092 |
| Net EV/trade | +131.31 bps | +129.11 bps | −2.20 bps |
| Total net bps | +997,959 | +1,092,386 | +94,427 |
| Worst month / week | +68.28 / +50.44 bps | +68.28 / +39.42 bps | week lower |
| Max drawdown | −30.06% | −31.56% | −1.50 pp |

On the complete August 1–27 decision-day panel the portfolio made 861 entries
at +109.67 bps/trade, with no zero-entry day.  The final 28-August 00:00 UTC
decision was explicitly excluded from this scope.  The detailed daily and
risk receipt is
[`strict_r3_p8u_f72_underf120_extended_quality_aug27_20260828_v3`](../data_perp/artifacts/strict_r3_p8u_f72_underf120_extended_quality_aug27_20260828_v3/).

### August-27 gate and capacity sensitivity — reporting only

The frozen control remains the dual-MC1 **50-bps** gate with a maximum of two
new positions per timestamp.  No gate or capacity was selected from the
August extension.  The table makes the requested pre-portfolio admission
versus constrained portfolio trade-off explicit on the exact 2025-11 through
2026-08-27 decision scope.

| Dual gate / new-entry cap | Raw dual admits / EV | Portfolio entries / EV | Total net bps | Worst month / week | Max DD |
|---|---:|---:|---:|---:|---:|
| 30 bps / 1 | 45,136 / +106.56 | 6,751 / +148.47 | +1,002,350 | +89.01 / +26.25 | −31.81% |
| 30 bps / 2 | 45,136 / +106.56 | 9,126 / +119.06 | +1,086,505 | +64.65 / +32.52 | −42.54% |
| 40 bps / 2 | 35,837 / +123.61 | 8,916 / +123.20 | +1,098,447 | +65.08 / +36.96 | −43.12% |
| **50 bps / 2 (frozen)** | **29,474 / +137.08** | **8,461 / +129.11** | **+1,092,386** | **+68.28 / +39.42** | **−31.56%** |
| 50 bps / 1 | 29,474 / +137.08 | 6,342 / +153.41 | +972,948 | +88.98 / +26.25 | −31.81% |

At 50 bps/two entries, there are no zero-trade days, one day with fewer than
five trades, nine with fewer than ten, and 31.89 portfolio entries/day in
August.  The simulated wallet has no negative calendar-week closing return,
so Sortino is **undefined**, not reported as infinite.  This does not remove
the observed −31.56% intraperiod maximum drawdown.

### August-27 target-free layer check

On the exact 647-timestamp cutoff panel, P8U Router50 recall is 66.62%,
72.60%, 76.62%, and 80.15% for policy-net opportunities above +50/+100/+150/
+200 bps.  F72 Base remains materially sharper than the Current 75/25 blend
at the extreme tail; Under is therefore confirmation information, not a
replacement alpha rank.

| Timestamp-local cut | F72 Base net / >+50 hit | Current 75/25 net / >+50 hit |
|---|---:|---:|
| Top 1% | +210.83 / 69.24% | +109.33 / 69.24% |
| Top 2% | +163.63 / 65.53% | +85.81 / 67.31% |
| Top 5% | +93.02 / 57.19% | +52.54 / 62.29% |
| Top 10% | +53.37 / 51.40% | +29.59 / 57.46% |
| Top 15% | +33.89 / 47.87% | +20.50 / 53.79% |

Under conditional MI given F72 is 0.08081 nats in August.  The full receipt
is [`strict_r3_p8u_august01_27_layer_extension_20260828_v1`](../data_perp/artifacts/strict_r3_p8u_august01_27_layer_extension_20260828_v1/).

The saved, broader audit is
[`pipeline_metrics_with_router.md`](pipeline_metrics_with_router.md); it
contains the full matched precision, conditional-MI, MC1 floor, capacity,
oracle, and coverage tables.

## Frozen stack

```text
Point-in-time target-free candidates
  → P8U Router50 identity gate
  → F72 Raw-bps CatBoost Base
  → BCF family: Base rank
  → Current family: 75% Base rank + 25% Under F120 rank
  → independent strict-prequential BCF and Current MC1 expected-EV maps
  → BCF EV >= +50 bps AND Current EV >= +50 bps
  → one chronological constrained portfolio auction, ranked by BCF MC1 EV
```

`Under F120` is the **only** downstream head. Magnitude, Over-confidence,
signed State, multi-head blends, BCF-MC1 interaction demoters, and banded
CatBoost mappers are rejected for this research contract.

| Layer | Contract | Role | Authority |
|---|---|---|---|
| Candidate universe | Point-in-time target-free candidates | Establishes eligibility before an outcome exists | No future path/outcomes |
| Router | P8U top 50% within timestamp | Identity gate only | Not a numeric Base feature |
| Base | 72-field Raw-bps CatBoost QueryRMSE | Opportunity ordering | Produces Base timestamp rank |
| BCF | Base rank only | Precision-preserving family | Its MC1 EV orders the auction |
| Current | `0.75 × Base + 0.25 × Under F120` rank | Conservative confirmation family | Must clear its MC1 EV gate |
| Under F120 | 120-field XENDCG under-confidence head | Identifies Base under-confidence | Confirmation only; no direct auction rank |
| MC1 | Separate 3-month prequential Current / BCF maps | Maps each family to policy EV | Both must be >= +50 bps |
| Portfolio | One global chronological constrained auction | Applies portfolio constraints | Priority is BCF MC1 EV |

## Router and Base

P8U retains the exact timestamp-local top 50% of candidates. Its score is not
passed to the Base model and there is no second Base top-percent cutoff after
Router50.

### P8U Router50 — frozen economic recall gate

The Router is a **30-field LightGBM Rank-XENDCG** model.  It is fitted on three
months, excludes the last 28 days for label resolution, caps training at
120,000 rows, and persists target-free held-month scores before any outcome
join.  Its only inference authority is the exact timestamp-local top-50%
identity gate; its numeric score never enters the Base, Under, MC1, or auction
features.

| Item | Frozen value |
|---|---|
| Label | `P8u_floor100_cap250`: canonical policy net <= +100 bps is grade 0; positive excess is capped at 250 bps and assigned five ordered grades at 31.25/62.5/109.375/171.875 bps of excess |
| Query weighting | Each timestamp has total loss weight 1. Within a timestamp, `sqrt_excess` raises relative weight from 1x toward 2x for policy net from +100 to +350 bps; excess is capped at +250 bps |
| Objective | `rank_xendcg`; gains `[0,1,2,4,7,11]`; truncation 12; exact timestamp × long-side query |
| HPO winner | 1,000 trees; LR 0.0567571; depth 4; 15 leaves; min-child `max(500, 1.7038% of train rows)`; min gain 0.00321538; feature fraction 0.787355; subsample 0.727909; L1 0.0141675; L2 0.216746; max bin 127; early stop 30; inner validation 20%; seed 1729 |
| Feature contract | 30 causal fields, SHA-256 `c787eb4c432dee34b200aa4a861e695a9597e16adb24376510dedb47d550d284`; exact list in [`run_contract.json`](../data_perp/artifacts/strict_r3_p8u_router_oof_apr25_jul26_successorlabels_20260828_v1/run_contract.json) |

The optimized P8U Router challenger was not promoted: it improved standalone
router utility but did not improve the full downstream economics.  This
document therefore refers only to the frozen P8U control above.

The Base is `P8U_RAW_BPS_CATBOOST_QUERYRMSE_F72_TAIL125`:

| Item | Frozen value |
|---|---|
| Features | 72 causal P8U fields |
| Target geometry | Within each training fold, canonical rich-policy net bps is clipped at the train-only P2/P98, then five evenly spaced cut points between those two values form grades 0--5. Held rows are scored against their fitted model; held outcomes never determine cut points. |
| Query | Exact decision timestamp × long side |
| Learner | CatBoostRanker, QueryRMSE |
| Weights | `tail_linear_125`: raw weight = `1 + 0.125 × grade`; a monotone within-query projection then gives every timestamp query mean weight 1.0 while keeping each row in `[0.5, 2.0]` |
| Fit | 3 months; 28-day resolved-label reserve; 60,000 complete-query cap |
| Parameters | 2,000-tree ceiling; early stop 30; depth 5; LR 0.0650994; feature fraction 0.800651; bagging 0.709605; L2 2.235726; random strength 0.942890; seed 1729 |

Strict-OOF standalone evidence over Nov 2025–Jul 2026: ScoreStable 1.714 and
timestamp-local DTP2/DTP5/DTP10 of +190.30/+131.26/+86.14 bps. See
[`P8U_BASE_PRECISION_PRESERVATION_HANDOVER_20260828.md`](P8U_BASE_PRECISION_PRESERVATION_HANDOVER_20260828.md).

## Under F120

`xendcg_selected_under_bps100` is a 120-field causal timestamp-query LightGBM
Rank-XENDCG head. Its task is the +100-bps unexpected-under-confidence
condition relative to a causal Base-to-EV anchor.

| Item | Frozen value |
|---|---|
| Feature contract | 120 fields; scorer path-plus-bytes identity SHA-256 `bf662742bc72c8a2ccd0fdee21a1f6a354be23e40c64a83b807c2abb9921a900` (file-bytes SHA-256 `d5ad535b1d9cb38321b9c4325377fe87b41ac5d8c24913ac1483dadea5e912ea`) |
| Target geometry | Binary `under_bps100__timestamp`: valid rows receive 1 only when path MFE reaches the frozen +0.5 ATR trailing-activation level **and** `policy_net_bps − prequential_Base_anchor >= +100 bps`; otherwise 0. The 14-day blockwise, train-only isotonic Base anchor is never an inference feature. |
| Objective | `rank_xendcg`, exact timestamp query; the binary labels use the frozen gain vector below |
| Gains / truncation / sigmoid | `[0, 1, 2, 4, 7, 11, 16, 24]` / 12 / 1.0 |
| Parameters | 260 trees; LR 0.045; depth 4; 15 leaves; min child 350; feature fraction 0.80; bagging 0.82; L1 0.02; L2 8.0 |
| Sample weights / use | Unweighted fit; no weight is an inference input. Under receives the full 25% of the Current-family rank and has no direct auction authority. |

The feature contract is
[`under_f120.json`](../data_perp/artifacts/strict_r3_p8u_meta_under_fullfeatures_selection_20260828_v2/contracts/under_f120.json).

Its selection pipeline is full-universe hygiene → cross-era conditional IC/CMI
and redundancy checks → randomized subspace gain/tail-SHAP → group MDA → a
bounded SStableMeta ladder.  The final 120 fields and their family counts are
inside that frozen contract; they are all point-in-time causal inputs, while
Base/path/policy outcomes enter only during training-label construction.

| Training item | Frozen value |
|---|---|
| Train / reserve / cap | 4 months / 28 resolved-label days / 100,000 query-safe rows |
| Residual anchor | 14-day expanding, train-only Base-rank-to-policy-net anchor |
| Target | `under_bps100__timestamp`: unexpected favorable policy residual above +100 bps with the resolved path reaching the frozen 0.5-ATR trailing activation |
| Feature count in fitted matrix | 129: 120 selected causal fields plus nine deterministic Base-query geometry fields |
| Sample weights | Uniform; weights never enter inference features |

## MC1 mapping and admission

MC1 is not another ranking head.  It is a separate **absolute expected-policy-
net mapper** for each score family.  The BCF family receives Base rank only;
the Current family receives `0.75 × Base rank + 0.25 × Under rank`.  The two
maps are fit independently and both must clear the frozen +50-bps floor.

| Item | Frozen value |
|---|---|
| Inputs | `final_score`, `base_rank42`, `conditional_consensus_rank`, `upstream`, `ordinary_shadow_consensus_rank`, `correctness_rank` |
| Target | Canonical policy **net** bps; the fixed 100-bps policy cost is already included exactly once |
| Fit history | Prior 3 calendar months, valid labels resolved before the held month; day-balanced sample (top 50 daily rows plus up to 250 random remaining rows/day), capped at 50,000 |
| Estimator | HistGradientBoostingRegressor: depth 2, 80 iterations, LR 0.04, L2 20, min leaf 100, seed 1729 |
| Structural calibration | Timestamp-local ten score bands; the band curve is precision-shrunk to the global mean with `precision = n/(sd²+1)` and prior `80/250²`, then made monotonic by isotonic regression |
| Robustness | Training target clipped at its train-only P2/P98; a causal 21-day daily residual shift uses a 10% trimmed mean |
| Admission / auction | `BCF_MC1 >= +50` **and** `Current_MC1 >= +50`; one global chronological constrained auction; priority = BCF MC1 expected net bps |

There is no double fee debit in MC1 or the portfolio replay: `policy_net_bps`
already equals gross policy outcome less the fixed 100-bps all-in round-trip
cost.

## Matched constrained evidence

All figures below use canonical rich-policy **net** bps, independent
strict-prequential Current/BCF MC1 maps, the dual +50-bps gate, and one shared
chronological constrained portfolio. The supported matched evaluation covers
November 2025–July 2026: August–October 2025 are the required MC1 warm-up
months, not evaluation rows.

| Contract | Entries | MC1-admitted rows | Net EV/trade | Total net bps | Worst month | Worst week | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base-only | 8,017 | 36,107 | +123.39 | +989,208 | +65.95 | +46.39 | -51.59% |
| **Router50 + F72 Base + Under F120** | **7,600** | **27,382** | **+131.31** | **+997,959** | **+68.28** | **+50.44** | **-30.06%** |
| **Delta** | -417 | -8,725 | **+7.92** | **+8,751** | **+2.33** | **+4.04** | **+21.53 pp** |

The retained stack removes 5.2% of constrained entries while increasing both
unit EV and total realised contribution. Its most material advance is
downside: the matched maximum drawdown improves by 21.53 percentage points.

### Strict-OOF output coverage and monthly receipt

The layers have different earliest legitimate dates. The reporting receipt
does **not** fabricate in-sample scores or shorten a warm-up:

| Layer | Strict-OOF score coverage | Earliest constrained evaluation |
|---|---|---|
| Router / F72 Base | Mar 2025–Jul 2026 | Base timestamp-local metrics start Mar 2025; the single March history-only score is explicitly not a selection result |
| Under F120 | Aug 2025–Jul 2026 | Four prior Base-score months are available under the restored strict prehistory |
| Independent dual MC1 + portfolio | Aug 2025–Jul 2026 score history | Nov 2025–Jul 2026 after three prior months |

The November 2024 Router score, March 2025 Base score, and March 2025 Under
feature panel were rebuilt from target-free point-in-time identities. The
historical-only Base extension is explicitly recorded in its run manifest and
does not compute held outcome metrics. All Under folds use only earlier
Base/Under scores and labels resolved before their 28-day reserve.

The detailed month-by-month receipt is
[`P8U_F72_UNDERF120_MONTHLY_RECEIPT.md`](../data_perp/artifacts/strict_r3_p8u_f72_underf120_full_oof_reporting_20260828_v7/P8U_F72_UNDERF120_MONTHLY_RECEIPT.md).
It contains Base timestamp-local Top-1/2/5/10/20 results for every available
month, the conditional-quality metrics for Under F120, matched Base-only and
Base+Under dual-MC1 portfolio outcomes, and a separately labelled historical
live-BCF orientation benchmark.

Monthly matched results are retained in the reporting receipt. The nine-month
summary is:

| Month | Base-only: entries / EV | Base + Under: entries / EV | Δ entries | Δ EV/trade | Δ total bps |
|---|---:|---:|---:|---:|---:|
| Nov-25 | 965 / +141.36 | 926 / +147.61 | -39 | +6.25 | +274 |
| Dec-25 | 898 / +81.55 | 815 / +95.20 | -83 | +13.65 | +4,360 |
| Jan-26 | 886 / +128.49 | 851 / +140.49 | -35 | +12.01 | +5,719 |
| Feb-26 | 589 / +142.50 | 462 / +170.74 | -127 | +28.24 | -5,051 |
| Mar-26 | 918 / +163.06 | 914 / +167.71 | -4 | +4.65 | +3,599 |
| Apr-26 | 852 / +165.39 | 855 / +167.66 | +3 | +2.27 | +2,435 |
| May-26 | 982 / +102.96 | 979 / +104.26 | -3 | +1.30 | +965 |
| Jun-26 | 864 / +138.86 | 752 / +153.08 | -112 | +14.22 | -4,861 |
| Jul-26 | 1,063 / +65.95 | 1,046 / +68.28 | -17 | +2.33 | +1,311 |
| **Total** | **8,017 / +123.39** | **7,600 / +131.31** | **-417** | **+7.92** | **+8,751** |

### Current live-stack comparison

The current live score family is the sealed BCF/current-v5 dual-MC1 stack,
operated at a 30-bps dual floor. It was replayed over the same Nov 2025–Jul
2026 rich-policy outcomes and portfolio engine. It is not a paired candidate
test—the P8U Router50 population differs—but it is the correct operational
baseline.

| Stack / floor | Entries | Net EV/trade | Total net bps | Worst month | Worst week | Max DD |
|---|---:|---:|---:|---:|---:|---:|
| Current live BCF/current-v5, 30 bps | 6,944 | +121.57 | +844,166 | +93.50 | +52.46 | -36.34% |
| Current live BCF/current-v5, 50 bps | 5,940 | +137.02 | +813,885 | +98.52 | +76.71 | -25.22% |
| **P8U Router50 + F72 + Under F120, 50 bps** | **7,600** | **+131.31** | **+997,959** | +68.28 | +50.44 | **-30.06%** |

Against the actual 30-bps live operating floor, the research stack adds 656
entries, +9.74 bps/trade and +153,793 total bps, while reducing drawdown by
6.28 percentage points. Against the floor-matched 50-bps live control, it
adds 1,660 entries and +184,073 total bps but gives up 5.71 bps/trade and
has weaker worst-month/week results. This is a qualified research advance, not
live-promotion evidence.

### Dual-MC1 gate sensitivity (frozen-map full OOS replay)

The frozen P8U score and already-fitted independent Current/BCF MC1 maps
were replayed at a common 30/35/40/45/50/60/70/80-bps floor over the full
valid November 2025--July 2026 MC1 OOS interval. There is **no model or
mapper refit** in this comparison: only the dual admission floor changes.
Every arm uses one chronological constrained portfolio and the same rich
policy outcome labels (including the fixed 100-bps round-trip cost exactly
once). The 50-bps floor remains the frozen research choice: it is the best
balance of total contribution, worst-week performance, drawdown and
participation, without selecting a new floor on this evidence.

| Dual gate | MC1-admitted rows | Entries | Net EV/trade | Total net bps | Worst month | Worst week | Max drawdown | Mean trades/day | Max trades/day | Days <1 / <5 / <10 trades |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 bps | 42,210 | 8,193 | +120.85 | +990,131 | +64.65 | +32.52 | -35.52% | 30.01 | 44 | 0 / 0 / 2 |
| 35 bps | 37,589 | 8,123 | +124.06 | **+1,007,763** | +65.66 | +32.65 | -35.52% | 29.75 | 44 | 0 / 0 / 2 |
| 40 bps | 33,248 | 7,995 | +125.44 | +1,002,881 | +65.08 | +43.97 | -36.09% | 29.29 | 44 | 0 / 0 / 4 |
| 45 bps | 30,060 | 7,811 | +128.14 | +1,000,887 | +66.86 | +47.11 | -33.01% | 28.61 | 44 | 0 / 1 / 7 |
| **50 bps (retained)** | **27,382** | **7,600** | **+131.31** | +997,959 | +68.28 | **+50.44** | -30.06% | 27.84 | 44 | 0 / 1 / 9 |
| 60 bps | 22,659 | 7,095 | +135.74 | +963,067 | +65.92 | +45.73 | **-27.87%** | 25.99 | 45 | 0 / 2 / 13 |
| 70 bps | 18,083 | 6,473 | +145.17 | +939,681 | **+69.64** | +43.53 | -27.87% | 23.71 | 42 | 0 / 6 / 24 |
| 80 bps | 15,015 | 5,788 | **+151.97** | +879,624 | +66.08 | +46.25 | -29.13% | 21.20 | 39 | 3 / 19 / 41 |

The lower-tail EV figures below are quantiles of *active* UTC-day or
UTC-week mean net EV/trade. Zero-entry periods are deliberately excluded
from those quantiles and captured by the participation counts above.

| Dual gate | Day Q5 / Q10 / Q15 / Q20 EV/trade (bps) | Week Q5 / Q10 / Q15 / Q20 EV/trade (bps) |
|---:|---:|---:|
| 30 bps | -20.87 / +14.84 / +31.05 / +46.50 | +52.59 / +66.32 / +74.96 / +77.75 |
| 35 bps | -26.39 / +6.23 / +32.98 / +51.91 | +52.47 / +67.67 / +71.29 / +72.74 |
| 40 bps | -22.24 / +6.57 / +27.30 / +49.08 | +52.86 / +67.35 / +68.84 / +76.82 |
| 45 bps | -9.63 / +14.70 / +31.65 / +52.08 | +56.08 / +69.50 / +71.77 / +84.65 |
| **50 bps (retained)** | **-12.07 / +18.64 / +39.97 / +53.74** | **+53.86 / +70.41 / +77.12 / +90.67** |
| 60 bps | -10.23 / +12.95 / +42.04 / +57.33 | +61.02 / +71.33 / +87.73 / +98.90 |
| 70 bps | -10.24 / +25.15 / +49.38 / +72.95 | +71.83 / +83.03 / +96.66 / +102.22 |
| 80 bps | -14.89 / +31.24 / +51.00 / +68.13 | +50.78 / +88.25 / +101.34 / +104.15 |

The 60/70/80-bps gates improve unit EV but trade off a meaningful amount of
economic contribution and raise the number of thin-participation days. The
35-bps arm has the largest total bps in this already-observed period, but its
lower-tail and drawdown profile are weaker; it is not promoted by this
diagnostic.

Artifacts: `data_perp/artifacts/strict_r3_p8u_f72_underf120_gate_sweep_nov25jul26_20260828_v1/`;
runner: `scripts/report_strict_r3_p8u_f72_underf120_gate_sweep_v1.py`.

| Rejected alternative | Reason |
|---|---|
| Magnitude | +1.49k total bps, but worse drawdown (-30.69%) and weaker worst week |
| Under F72 | Higher unit EV but lower total bps and worse drawdown |
| Over / State | Lower total contribution and/or weaker risk profile |
| Multi-head blends | Reduced total economic contribution despite occasional higher unit EV |
| Bounded BCF interaction demoters | No material portfolio advance; best tiny total gain worsened drawdown |
| Banded CatBoost mapper | Lower EV/trade, lower total bps, and weaker downside profile |

## Causality and reproducibility

The source replay completed these checks:

- Base and Under scores are target-free and persisted before the policy join.
- Base/Under candidate IDs match exactly inside the frozen Router50 population.
- Current and BCF MC1 maps are separate and fit only on earlier resolved labels.
- Held outcomes are not model inputs; no pooled future rank is used.
- Both MC1 gates precede one chronological constrained portfolio.
- No live or exchange state was mutated.

Key artifacts:

- Base target-free score union: `data_perp/artifacts/strict_r3_p8u_tail125_base_history_mar25_jul26_fullprehistory_20260828_v1/`
- Under F120 score union: `data_perp/artifacts/strict_r3_p8u_meta_under_fullfeatures_xendcg_f120_aug25_jul26_fullprehistory_20260828_v4/`
- Under F120 selection: `data_perp/artifacts/strict_r3_p8u_meta_under_fullfeatures_selection_20260828_v2/`
- Matched Base-only MC1 / portfolio: `data_perp/artifacts/strict_r3_p8u_f72_baseonly_dual_mc1_nov25_jul26_fullprehistory_20260828_v1/`
- Matched Base + Under MC1 / portfolio: `data_perp/artifacts/strict_r3_p8u_f72_underf120_dual_mc1_nov25_jul26_fullprehistory_20260828_v1/`
- August target-free Base bridge: `data_perp/artifacts/strict_r3_p8u_f72_base_score_bridge_mar25_aug27_20260828_v1/`
- August target-free Under bridge: `data_perp/artifacts/strict_r3_p8u_under_f120_score_bridge_aug25_aug27_20260828_v1/`
- August target-free Router/Base/Under layer receipt: `data_perp/artifacts/strict_r3_p8u_august01_27_layer_extension_20260828_v1/`
- August rich-policy label successor: `data_perp/artifacts/strict_r3_p8u_router_policy_label_successor_fullprehistory_aug27_20260828_v1/`
- August-27 dual-MC1 / constrained-portfolio extension: `data_perp/artifacts/strict_r3_p8u_f72_underf120_dual_mc1_nov25_aug27_20260828_v1/`
- August-27 cutoff-aware gate/capacity sweep: `data_perp/artifacts/strict_r3_p8u_f72_underf120_gate_capacity_sweep_aug27_20260828_v2/`
- August-27 monthly/weekly/daily risk receipt: `data_perp/artifacts/strict_r3_p8u_f72_underf120_extended_quality_aug27_20260828_v3/`
- Full OOS reporting receipt: `scripts/report_strict_r3_p8u_f72_underf120_canonical_v1.py` and `data_perp/artifacts/strict_r3_p8u_f72_underf120_full_oof_reporting_20260828_v7/`
- Current-live baseline replay: `data_perp/artifacts/strict_r3_live_bcf_current_dual_reconciled_rich_portfolio_nov25jul26_20260828_v1/`
- Router prehistory score union: `data_perp/artifacts/strict_r3_p8u_router_oof_nov24_jul25_fullprehistory_20260828_v1/`
- Base materialisation: `scripts/run_strict_r3_p8u_precision_preservation_weight_funnel_v1.py --history-only`
- Under model scoring: `scripts/run_strict_r3_p8u_meta_lgbm_objective_screen_v1.py`
- Under feature selection: `scripts/select_strict_r3_p8u_meta_fullfeatures_v1.py`; cross-model screen `scripts/run_strict_r3_p8u_meta_crossmodel_v1.py`
- MC1 and constrained replay: `scripts/run_strict_r3_p8u_meta_mc1_combination_v1.py`
- Incremental target-free bridge: `scripts/build_strict_r3_target_free_score_ledger_bridge_v1.py`; appended outcome ledger: `scripts/append_strict_r3_policy_label_ledger_v1.py`
- Immutable lineage adapters: `scripts/materialize_strict_r3_p8u_router_score_union_v1.py`, `scripts/materialize_strict_r3_p8u_base_score_union_v1.py`, and `scripts/materialize_strict_r3_p8u_meta_score_union_v1.py`
- Meta-family handover: [`P8U_META_LAYER_FAMILY_SELECTION_HANDOVER_20260828.md`](P8U_META_LAYER_FAMILY_SELECTION_HANDOVER_20260828.md)

### Selection and reproduction map

| Layer | Target / model selection | Feature selection | OOS scoring and terminal replay |
|---|---|---|---|
| P8U Router50 | `scripts/run_strict_r3_economic_recall_router.py`; `scripts/run_strict_r3_router_hpo.py`; `scripts/select_strict_r3_economic_recall_router_hpo.py` | Frozen 30-field list in the Router `run_contract.json` | Router ledger union: `scripts/materialize_strict_r3_p8u_router_score_union_v1.py` |
| F72 Base | `scripts/run_strict_r3_p8u_precision_preservation_screen_v1.py`; `scripts/run_strict_r3_p8u_precision_preservation_loss_funnel_v1.py`; `scripts/run_strict_r3_p8u_precision_preservation_objective_funnel_v1.py`; `scripts/run_strict_r3_p8u_precision_preservation_weighted_cross_model_v1.py`; `scripts/run_strict_r3_p8u_precision_preservation_winner_hpo_v1.py`; `scripts/run_strict_r3_p8u_precision_preservation_weight_funnel_v1.py` | `scripts/run_strict_r3_p8u_precision_preservation_feature_prescreen_v1.py`; `scripts/run_strict_r3_p8u_precision_preservation_group_mda_beam_v1.py`; frozen F72 receipt | `scripts/run_strict_r3_p8u_precision_preservation_weight_funnel_v1.py --history-only`; `scripts/materialize_strict_r3_p8u_base_score_union_v1.py` |
| Under F120 | `scripts/run_strict_r3_p8u_meta_target_query_grid_v1.py`; `scripts/run_strict_r3_p8u_meta_lgbm_objective_screen_v1.py` | `scripts/select_strict_r3_p8u_meta_fullfeatures_v1.py`; `scripts/run_strict_r3_p8u_meta_crossmodel_v1.py` | `scripts/materialize_strict_r3_p8u_meta_score_union_v1.py` |
| Dual MC1 / portfolio | No target or HPO selection at terminal reporting; frozen independent Current / BCF maps only | N/A | `scripts/run_strict_r3_p8u_singlebase_true_dual_mc1_v1.py`; `scripts/report_strict_r3_p8u_f72_underf120_canonical_v1.py`; `scripts/report_strict_r3_p8u_f72_underf120_gate_sweep_v1.py`; `scripts/report_strict_r3_p8u_extended_quality_v1.py` |

## Promotion boundary

This is a research canonical, not a live promotion. Moving it beyond research
requires at least six common post-Meta months under this exact contract, a
later untouched forward period, complete inference/execution parity, and
separate explicit authorization.
