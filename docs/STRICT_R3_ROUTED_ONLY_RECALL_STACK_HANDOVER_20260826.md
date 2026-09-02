# Strict-R3 Routed-Only Recall Stack — Research Handover

**Status:** frozen current research winner as of 2026-08-27; not a live or
production promotion.
**Scope:** long-only, strict walk-forward April–July 2026, canonical reconciled rich-policy outcome ledger.

This handover is the canonical record for the routed-only architecture. It supersedes the older router-50 **full-base** handover for decisions about this stack. The older result remains historical evidence only because its base did not satisfy the routed-only training requirement.

## Freeze decision

The contract below is frozen as the **current research winner**.  It is the
control for subsequent Base, consensus, and mapper work.  In particular, the
Router50 single-Base XENDCG experiments documented in
`ROUTER_SINGLE_BASE_XENDCG_DOWNSTREAM_20260827.md` do not replace any layer of
this stack: their only favourable result is a two-month, fixed-R/U mini-MC1
screen and they have not beaten this stack in a matched full-contract replay.

The freeze applies to the exact router, three-way Base, T6/T9 heads, MC1
inputs, dual +50-bps admission, portfolio priority, and rich-policy outcome
contract listed below.  It does **not** grant live promotion.  April–July 2026
was used for model selection, and the deployed live score/map state has not
been replayed byte-for-byte under this research contract.  A successor must
first pass a predeclared, later untouched evaluation using the same
target-free candidate identities, policy labels, admission gate, and
chronological portfolio rules.

## Frozen research contract

```text
full causal candidate universe
  -> full-universe P8u recall router, exact top 50% per timestamp
  -> routed-only enhanced three-way base
  -> routed-only T6/T9 consensus heads
  -> Current and BCF MC1 maps with one router rank as an MC1 input
  -> both mapped EVs >= +50 bps
  -> BCF mapped-EV priority and chronological portfolio auction
  -> maximum two new entries per timestamp
  -> frozen rich 15-minute policy outcome
```

The router is a target-free gate at score time. The base and T6/T9 heads train only on rows inside its timestamp-local top 50%. The sole numeric router value retained downstream is `router_primary_rank`, supplied only to MC1.

## Router feature-selection and HPO

| Stage | Fields |
|---|---:|
| Causal full universe | 1,407 |
| Coverage/variance gate | 1,251 |
| Redundancy-veto pool | 355 |
| Frozen router contract | 30 |

Router target: `policy_net_bps >= +50`; selection aggregates equally over exact decision timestamps. The selector used gain/split stability, timestamp-local univariate recall rescue, Spearman redundancy veto, and a subset ladder. MDA was intentionally omitted. Frozen feature-contract SHA-256: `c787eb4c432dee34b200aa4a861e695a9597e16adb24376510dedb47d550d284`.

The P8u router is `rank_xendcg`, depth 4, 15 leaves, learning rate 0.05676, 12-rank truncation, 100–250-bps sqrt-excess weights, a 28-day reserve, and three prior training months. Its 13-fold top-50% diagnostic is ER50 0.8194, +50-bps winner recall 0.7237, and +100-bps recall 0.8015.

The rejected auxiliary family made the prior `router_primary_rank`, `router_primary_only_rank`, and `router_full_ae_rank` aliases byte-identical. Only the one distinct primary rank is now allowed into MC1. All earlier three-alias router-input receipts are superseded.

### Post-freeze full-universe add-back falsification — 2026-08-27

The frozen 30-field router remains the selected contract. A separate 1,407-field causal-universe screen retained 1,038 hygienic fields, then tested replacement and incremental add-back contracts. The guarded 150-field `prescreen_plus120` contract won both cheap screens and a first HPO pass, but the initial 30,000-row HPO was **not forward-valid**: it retained fewer than 500 hourly queries. That result is superseded.

The corrected HPO used a 120,000-row target-free cap, at least 727 eligible hourly training queries in every forward fold, the identical causal cap rule as the scorer, a three-month fit window, and the same 28-day reserve. It selected the same 150-field challenger at HPO stage (`S_stable=0.75281` versus `0.74760` for the HPO-tuned 30-field control). The only valid next check was an independent Apr--Jul 2026 target-free forward replay, followed by the common post-score timestamp-local utility evaluator.

| Forward arm | Fields | Mean `S_router` | `S_stable` | Q25 fold | Worst fold | R50 utility | R50 count | R100 count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HPO-tuned frozen control | 30 | **0.81798** | **0.81223** | 0.80587 | 0.79089 | **0.81704** | **0.79706** | **0.84333** |
| 150-field HPO challenger | 150 | 0.81696 | 0.81211 | **0.80792** | **0.79158** | 0.81605 | 0.79622 | 0.84197 |

The challenger improves the lower-tail fold diagnostics but loses average routing utility, R50, R100, and the predeclared aggregate stability score by 0.00011. It therefore fails the frozen-forward advance gate. No consensus, MC1, admission, or portfolio replay was run for it; omitting those stages prevents an unvalidated upstream change from being selected on downstream noise. The 30-field frozen router remains canonical.

Relevant immutable receipts:

| Purpose | Artifact |
|---|---|
| 1,038-field hygiene contract | `data_perp/artifacts/strict_r3_router_feature_hygiene_20260827_v1/` |
| Incremental add-back screen | `data_perp/artifacts/strict_r3_router_incremental_addback_20260827_v1/` |
| Corrected full-support challenger HPO | `data_perp/artifacts/strict_r3_router_addback_hpo_fullsupport_20260827_v1/` |
| Corrected full-support control HPO | `data_perp/artifacts/strict_r3_router_addback_hpo_fullsupport_control_20260827_v1/` |
| HPO feature contracts | `data_perp/artifacts/strict_r3_router_addback_hpo_fullsupport_{winner,control}_contract_20260827_v1.json` |
| Target-free Apr--Jul forward scores | `data_perp/artifacts/strict_r3_router_hpo{150,30}_fullsupport_forward_aprjul_20260827_v1/` |
| Common post-score utility receipts | `data_perp/artifacts/strict_r3_router_hpo{150,30}_fullsupport_utility_aprjul_20260827_v1/` |

The scorer repair in `scripts/run_strict_r3_economic_recall_router.py` ensures a target-free whole-query cap is applied exactly once. The HPO preparation now uses that same cap and refuses an insufficient-query training ledger. Neither repair changes the frozen selected router or any live configuration.

## Base: routed-only enhanced three-way model

The base is an equal common-bps mean of three chronological-OOF coordinates:

```text
mean(
  R3 P(clear) - 0.5 P(adverse),
  direct policy-conversion efficiency,
  negative time-to-meaningful-MFE
)
```

It fits on up to two preceding calendar months of routed rows, with labels resolved before the same-model 28-day reserve and a 6,000-row floor. All coordinates map only through earlier rich-policy outcomes.

| Base arm | Local top-2 EV | Local top-10 EV | Top-10 precision >50 | Top-2 portfolio EV/trade | Max DD | Sortino |
|---|---:|---:|---:|---:|---:|---:|
| B0 full-trained then routed — diagnostic | +80.21 | +41.08 | 50.18% | +67.20 | −35.47% | 56.92 |
| Enhanced full-trained then routed — diagnostic | +137.81 | +72.81 | 48.15% | +120.96 | −17.38% | 327.24 |
| Routed-only enhanced, no router input | +106.62 | +52.89 | 44.77% | +90.77 | −32.32% | 111.38 |
| Routed-only enhanced, router-rank input | **+122.49** | **+60.35** | **46.47%** | +91.77 | −31.10% | 85.52 |

The rank-input base improves raw score quality but not the completed downstream portfolio. The frozen end-to-end base therefore has **no numeric router input**.

## Consensus: T6/T9 only

Both correction heads train on the same routed population using three earlier calendar months and a target-free 28-day reserve.

| Head | Contract | Role |
|---|---|---|
| T6 | `cap80_ordinary` | ordinal policy-conversion correction |
| T9 | `cap120_equal_month` | equal-month exit-quality correction |

| T6/T9 input arm | BCF local top-2 EV | BCF local top-10 EV | Top-10 precision >50 | 50-bps portfolio EV/trade | Total bps | Max DD | Sortino |
|---|---:|---:|---:|---:|---:|---:|---:|
| **No router input — selected** | +146.88 | +68.30 | 50.50% | **+155.30** | **+417,457** | **−22.34%** | **383.92** |
| Router rank input | **+152.64** | +65.90 | **54.93%** | +148.27 | +371,555 | −26.64% | 232.90 |

Raw score quality alone is not enough: router rank loses 182 constrained entries, 7.04 bps/trade, 45.9k total bps, and risk-adjusted performance. It is rejected. In the selected arm, standalone top-2 EV is +160.81 bps for T6 and +154.37 bps for T9.

## MC1: selected authority and admission

Current and BCF are separately prequential expected-rich-policy-net mappers, trained on up to three prior scored months. Their target-free score panels are persisted before outcomes are joined. A candidate requires both mapped values above the same floor; BCF mapped EV sets auction priority.

| Floor | MC1 input | Entries | EV/trade | Total bps | Worst month | Worst week | Max DD | Sortino |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 30 | no router | 3,236 | +133.35 | +431,506 | +97.20 | +70.98 | −31.26% | 463.08 |
| 30 | router rank | 3,266 | +135.86 | +443,713 | +97.87 | +70.97 | −22.63% | 383.07 |
| 40 | no router | 2,994 | +141.09 | +422,409 | +102.22 | +75.31 | −22.12% | 416.90 |
| 40 | router rank | 2,950 | +149.92 | +442,261 | +106.90 | +82.97 | −21.83% | 631.97 |
| **50** | no router | 2,688 | +155.30 | +417,457 | +109.34 | +75.89 | −22.34% | 383.92 |
| **50** | **router rank — selected** | **2,648** | **+163.98** | **+434,216** | **+109.54** | **+78.30** | **−16.83%** | **2,102.93** |

At the selected +50-bps dual-MC1 gate, the single rank adds +8.67 bps/trade, +16.8k total bps, 5.51 percentage points of drawdown improvement, and +1,719 Sortino points. It trades 40 fewer constrained entries. Both arms have zero days without entries during the evaluation window.

## Waterfall and remaining opportunity

All rows are local top-two rich-policy EV except the last row, which is the chronological capacity-constrained portfolio. The oracle is post-hoc only.

| Stage | Top-2 EV | Total top-2 bps | Precision >50 |
|---|---:|---:|---:|
| Enhanced base, full universe | +137.76 | +804,785 | 58.59% |
| Routed-only selected base | +106.62 | +622,877 | 51.52% |
| Oracle within routed set — diagnostic | +655.30 | +3,828,244 | 99.97% |
| Selected T6/T9 BCF score | +146.88 | +858,083 | 59.29% |
| Dual MC1 >=50, before cross-time constraints | +217.08 | +1,004,054 | 67.57% |
| **Dual MC1 >=50 + global portfolio** | **+163.98** | **+434,216** | — |

The routed-only base specialization loses 31.19 bps/top-two against the full-trained diagnostic; T6/T9 recover +40.26 bps; MC1 contributes +70.20 bps; portfolio capacity reduces average from +217.08 to +163.98 bps/trade. The large oracle headroom means the next research focus is conversion within the routed set, not relaxing admission.

Within the selected +50-bps dual-MC1 admitted set, the post-hoc top-two oracle is +303.92 bps/trade. The BCF-MC1 timestamp-local ordering realizes +223.22 bps/trade before cross-time constraints, and the shared portfolio realizes +163.98 bps/trade. This separates remaining score/auction headroom from capacity headroom.

## Legacy comparison and decision

The nearest matched historical downstream B0/T6/T9 control at a dual 50-bps gate had 1,847 entries, +166.97 bps/trade, +308,398 total bps, +150.99 worst-month EV, and +106.96 worst-week EV. The selected stack adds 801 entries and +125,818 total bps, but gives up 2.99 bps/trade and has weaker worst-period means.

This is therefore a research challenger—not a live promotion. The 2026 period is selection evidence, not untouched validation.

An additional matched April–July 2026 comparison begins from the sealed legacy BCF/current score families, re-fits only their MC1 maps on this stack's reconciled rich-policy ledger, and uses the same dual gate and portfolio engine. It is deliberately distinct from a byte-identical deployed-map replay.

| Arm | Dual floor | Entries | Net EV/trade | Total net bps | Worst month | Worst week | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy BCF/current score families | 30 bps — operating floor | 2,805 | +122.68 | +344,110 | +104.25 | +72.89 | −19.82% |
| Legacy BCF/current score families | 50 bps — floor-matched | 2,289 | +137.56 | +314,871 | **+112.89** | +76.71 | −24.86% |
| **Selected Router-50 stack** | **50 bps** | **2,648** | **+163.98** | **+434,216** | +109.54 | **+78.30** | **−16.83%** |

At the matched 50-bps floor, Router-50 adds 359 entries, +26.42 bps/trade and +119,345 total bps. It improves worst-week EV by +1.60 bps and drawdown by 8.03 points, while giving up 3.35 bps on the worst-month mean. This is a compelling research delta but not a promotion: the period served model selection and the deployed map state was not replayed byte-for-byte.

## Causality, policy, and reproduction

The selected final receipts prove: target-free scoring before outcome joins; all base training rows router-selected; label resolution before each reserve; exact Current/BCF identities; no numeric router input to base/T6/T9; one router rank only in MC1; and maximum two new entries per timestamp.

The router-aware MC1 receipt reuses the immutable router-free routed-base score panel so that its input ablation changes only MC1. Consequently, a legacy receipt field describing whether that *invocation* refit the base is false; the source base receipt is explicitly `route_first_base_refit=true`. Reuse does not change the base's routed-only training lineage.

The rich policy has decision-time 15-minute entry, 48 completed 15-minute bars/H12, 100-bps cost once, trailing profit, smooth capital protection, adverse exit, and frozen stop geometry. Its main parameters are `sl_mult=4.37975`, `1.25 × ATR^1.3` with a 0.6%–5.0% absolute range, and smooth protection activation 1.5 ATR / strength 0.5 / power 1.5. Exact one-minute execution is a distinct policy contract.

| Purpose | Path |
|---|---|
| Full-universe feature selector | `data_perp/artifacts/strict_r3_fulluniverse_recall_selector_20260826_v2/` |
| Frozen router / HPO | `data_perp/artifacts/strict_r3_fulluniverse_recall_router_full30_frozen_jul25_jul26_20260826_v1/` |
| Frozen research configuration | `config/strict_r3_router50_routedonly_challenger_20260826_v1.json` |
| Routed-only base and router-free T6/T9 | `data_perp/artifacts/strict_r3_router50_baseN_metaN_mc1N_routedonly_20260826_v1/` |
| Selected router-aware MC1 map | `data_perp/artifacts/strict_r3_router50_baseN_metaN_mc1R_routedonly_20260826_v1/` |
| Waterfall | `data_perp/artifacts/strict_r3_router50_routedonly_final_waterfall_20260826_v5/` |
| Matched legacy score-family replay | `data_perp/artifacts/strict_r3_legacy_live_dual_reconciled_rich_portfolio_aprjul_20260826_v1/` |
| Base/meta/MC1 reports | `data_perp/artifacts/strict_r3_router50_{base_variant_metrics_distinct_router,routedonly_meta_input_metrics,routedonly_mc1_input_metrics}_20260826_*/` |
| Stack runner | `scripts/run_strict_r3_router_routed_base_stack.py` |
| Selector / router scripts | `scripts/run_strict_r3_fulluniverse_recall_selector.py`, `scripts/run_strict_r3_economic_recall_router.py` |
| Layer reports / waterfall | `scripts/report_strict_r3_router50_{base,meta,mc1}_metrics.py`, `scripts/report_strict_r3_router50_waterfall.py` |

No live configuration, exchange state, or execution policy was altered by this work.
