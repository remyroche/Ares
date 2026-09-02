# Strict-R3 incumbent research contract — retained upstream and downstream control

**Last updated:** 2026-08-26  
**Scope:** long-only, offline research. This document does not change the deployed/live trader.  
**Status:** **RETAIN INCUMBENT.** The 2026-08-26 B40/E55/T05 replacement is rejected after its matched residual, dual-MC1, and constrained-portfolio replay. The retained incumbent is the canonical control for all new base and meta research until a challenger beats it on the same identity, policy, admission, and portfolio contract.

> The remainder of this document records the earlier O3-v2 enhanced-base/T6/T9 challenger evidence. It is historical research context, **not** the canonical promotion decision. The authoritative 2026-08-26 decision record is [STRICT_R3_THREEWAY_HEAD_SELECTION_DECISION_20260826.md](STRICT_R3_THREEWAY_HEAD_SELECTION_DECISION_20260826.md).

## Canonical control as of 2026-08-26

The retained upstream is the immutable incumbent score family materialised in
`data_perp/artifacts/strict_r3_frozen_threeway_matched_control_targetfree_20260826_v2/`.
On its native source it is:

```text
efficiency_bps = strict-OOF direct path-efficiency coordinate
timing_bps     = strict-OOF direct time-to-opportunity coordinate
incumbent_upstream_bps = 0.50 × efficiency_bps + 0.50 × timing_bps
```

`base_bps` is retained as a causal disagreement/geometry coordinate for downstream consumers, but it is not mixed into the incumbent upstream ranking. The source is target-free, point-in-time, and immutable; policy outcomes are attached only after held scores have been persisted. Any replacement must regenerate all downstream consumers prequentially rather than reuse incumbent ranks or maps.

| Canonical stage | Retained contract |
|---|---|
| Candidate source | Immutable matched incumbent target-free panel; 120 causal source fields available to downstream consumers; no policy/outcome columns |
| Upstream ordering | `0.50 × efficiency_bps + 0.50 × timing_bps`; timestamp-local route under the source contract |
| Residual/consensus | Refit strictly prequentially on the exact incumbent upstream; no head or MC1 artifact from a different blend may be reused |
| EV mapping | Separate Current and BCF prequential MC1 expected-EV maps; both must clear the declared gate |
| Admission and execution | Dual-MC1 admission followed by one chronological constrained portfolio and the same canonical rich-policy outcome ledger |

### B0 F72 is not in the retained incumbent upstream

The 72-field B0 ranker is a **research challenger**, not the B0 coordinate in
the retained E/T 50/50 upstream. Its stronger standalone ranking was included
in the rejected B40/E55/T05 replacement family, whose full downstream replay
lost to the incumbent. It must therefore not be described as an active
incumbent feature/HPO contract.

| Component | Feature/HPO contract | Status |
|---|---|---|
| Incumbent E | Existing frozen 120-field causal contract; Huber direct-efficiency model | Active retained upstream coordinate |
| Incumbent T | Existing frozen 120-field causal contract; Huber direct-timing model | Active retained upstream coordinate |
| B0 F72 | 72 selected causal fields; policy-ordinal LambdaRank; depth 4, 27 leaves, learning rate 0.06959, min-leaf fraction 0.005335, feature/bagging fractions 0.85784/0.71416, L1/L2 0.15466/0.11575, min gain 0.000946, sigmoid 0.84051, truncation 5 | Research-only; rejected as part of the downstream replacement family |

`base_bps` remains a causal source coordinate available to downstream
consumers. It is not evidence that the F72 B0 model is active in the retained
upstream route.

### Matched incumbent economic reference

The valid current comparison is the April--July 2026 strict 50-bps dual-MC1 replay on the common candidate source. This—not raw-tail ranking—is the promotion reference.

| Arm | Entries | Net EV/trade | Total net bps | Worst month | Worst week | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| **Retained matched incumbent** | **2,597** | **+159.81** | **+415,035** | **+130.22** | **+93.53** | **−25.35%** |
| Rejected B40/E55/T05 replacement | 3,249 | +125.15 | +406,612 | +81.96 | +39.37 | −32.20% |
| Replacement minus incumbent | +652 | −34.66 | −8,423 | −48.27 | −54.16 | −6.85 pp |

At the 30-bps diagnostic gate the same conclusion holds: incumbent `3,117 / +136.93 / +426,805` versus B40/E55/T05 `3,738 / +105.80 / +395,464` (entries / bps per trade / total bps). The higher challenger participation is lower-quality marginal admission, not a promotion case.

### Immediate research rule

Meta feature selection must use a **strict-prequential incumbent T6/T9 ledger** built from this exact score family. The previously available homogeneous T6/T9 ledger is tied to the enhanced three-way source and is not a valid incumbent substitute. No result obtained with that mismatched geometry may be promoted into the incumbent stack.

## Archived O3-v2 challenger record — do not use for new runs or promotion

The sections below are retained only to preserve the original research evidence. Their three-way upstream, T6/T9 score ledgers, and MC1 results are not compatible with the retained incumbent source family and must not be reused in an incumbent run.

### 1. Historical challenger decision summary

The upstream is **not B0 alone**. It is the equal, common-bps mean of strict-OOF B0, direct-efficiency, and direct-timing predictions. B0 remains one of the three components because it carries complementary opportunity information and is the most stable reference coordinate.

The correction layer is frozen to exactly two heads:

| Head | Physical model | Selected support weighting | Decision |
|---|---|---|---|
| T6 rank-error ordinal | `cap80_ordinary` | uniform (`S0`) | retain |
| T9 exit-5 ordinal | `cap120_equal_month` | coarse triple-barrier state (`S5_tbm_coarse`) | retain |

T1, T2, T4 and T8 are excluded from this successor contract. Adding T8 to T6+T9 reduced every May–July raw-tail metric in the tested equal-contribution blend. The other heads did not improve the constrained strict-MC1 comparison.

Frozen two-head contract:

```text
data_perp/artifacts/strict_r3_o3v2_t6t9_consensus_contract_20260825_v1/
  selected_physical_slots.json
SHA-256: bdcd87049184f586e3a64e9a6fe5cf74907be5785132123f406bbefaed5e41bc
```

### 2. Historical challenger research flow

```text
target-free point-in-time candidates and 120 causal base fields
  -> strict-OOF B0 / direct-efficiency / direct-timing base predictions
  -> equal common-bps three-way base blend
  -> timestamp-local top-30% base route
  -> T6 and T9 only
  -> 75% base rank + 25% mean(T6 rank, T9 rank)
  -> strict prequential current and BCF MC1 maps
  -> dual MC1 expected EV admission threshold
  -> one chronological constrained portfolio auction
  -> canonical reconciled rich-policy outcome
```

All meta score receipts are target-free. Policy labels enter only after an immutable score panel is written. Each held fold uses six complete prior resolved calendar months and excludes a 28-day reserve from head fitting.

### 3. Historical challenger layer contracts

#### 3.1 Enhanced base

| Component | Target / meaning | Features and fit contract | Output |
|---|---|---|---|
| B0 | Strict R3 opportunity score: `P(clear) − 0.5 × P(adverse)` | frozen 120-field causal base contract; strict OOF | `base_bps` |
| Direct efficiency | Frozen direct efficiency prediction from selected source arm `S3_direct_efficiency_time_base_equal` | same point-in-time base population; strict OOF | `efficiency_bps` |
| Direct timing | Frozen direct timing prediction from the same selected direct source arm | same point-in-time base population; strict OOF | `timing_bps` |
| Enhanced base | `(base_bps + efficiency_bps + timing_bps) / 3` | no re-fit and no outcome input at blend time | `enhanced_base_bps`, timestamp rank, top-30% route |

Target-free producer:

```text
data_perp/artifacts/strict_r3_enhanced_base_threeway_targetfree_20260824_v1/
```

It has 120 base fields and records 100% required-field coverage. The output manifest seals the three-way equal blend.

#### 3.2 T6 and T9 correction heads

Both heads score only candidates that pass the enhanced-base timestamp-local top-30% route. They are trained separately each fold, using resolved pre-reserve labels; their held score receipts never contain policy outcomes or semantic labels.

| Head | Label | Model / query | Features | Training support |
|---|---|---|---:|---|
| T6 | Five ordinal bins of within-query error: `rank(realised policy net) − base_rank`, edges `−0.20, −0.05, 0.05, 0.20` | LightGBM L2 regressor, `cycle_4h_side`; 120 trees, depth 5, 31 leaves, min child 300, learning rate .035, feature/bagging .82, L1 .02, L2 2.0 | 102 | uniform |
| T9 | Five ordinal policy-exit states: stop, timeout, smooth-protection, regular trailing, large trailing | same tree geometry and query; `cap120_equal_month` physical slot | 73 | mild balanced support by coarse triple-barrier state (`S5_tbm_coarse`) |

T6 corrects under/over-ranking by the enhanced base. T9 is a conversion/exit-quality coordinate. They are not five-slot ensembles: each target is exactly one frozen physical model.

#### 3.3 MC1 mapping, admission, and policy

The validated initial T6+T9 test uses these MC1 inputs:

```text
base_rank42
base_anchor_bps
correctness_rank
T6 consensus_rank, T6 combined_rank
T9 consensus_rank, T9 combined_rank
```

It replaces incumbent correction coordinates rather than appending them. Each score family uses a strict-prequential MC1 fit on six complete prior months. Current and BCF family maps must both clear the declared threshold; the normal constrained auction is then replayed chronologically on the same canonical rich-policy net outcome ledger.

Policy ledger:

```text
data_perp/artifacts/strict_r3_enhanced_base_rich_policy_labels_reconciled_20260823_v1/
  canonical_reconciled_policy_labels.parquet
```

It is the reconciled rich execution-policy outcome, including the frozen stop/trailing/smooth-protection rules and one application of the specified trading cost.

### 4. Historical challenger metrics and comparison conventions

All score-layer tables below use valid rich-policy rows during **May–July 2026**. They are ranking diagnostics, not live admission results.

- **Global tail** pools valid rows across the three months, then takes the indicated score percentile.
- **Timestamp-local tail** takes the indicated percentile independently at each decision timestamp and then pools outcomes.
- The MC1 table is different: it is a strict prequential, dual-admission, constrained-portfolio replay. It is the primary economic comparison.

### 4.1 Base-layer net EV per selected trade (bps)

| Base score | Global top 1% | 2% | 5% | 10% | Timestamp-local top 1% | 2% | 5% | 10% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 control | 101.3 | 90.9 | 73.1 | 54.9 | 99.9 | 80.5 | 70.8 | 49.2 |
| Efficiency | 155.1 | 139.2 | 112.0 | 84.3 | 146.7 | 130.8 | 106.7 | 79.6 |
| Timing | 138.5 | 126.3 | 101.5 | 73.8 | 130.1 | 115.1 | 95.4 | 70.7 |
| **Enhanced three-way base** | **144.2** | **137.4** | **110.8** | **86.7** | **144.2** | **124.7** | **110.4** | **81.0** |
| Enhanced minus B0 | +42.9 | +46.5 | +37.7 | +31.8 | +44.3 | +44.2 | +39.6 | +31.8 |

The direct-efficiency score is strongest by these raw-tail metrics. The equal three-way blend is retained because it is the frozen, stricter common-bps architecture that preserves B0 and timing diversity for downstream reliability features. A replacement by one direct component has not yet demonstrated a stronger full meta/MC1/portfolio result on the same contract.

### 4.2 Meta-layer raw ranking metrics (bps per selected trade)

Meta values are evaluated only after the enhanced-base top-30% route. `Upstream` means `0.75 × enhanced-base rank + 0.25 × correction rank`.

| Score | Global top 1% | 2% | 5% | 10% | Timestamp-local top 1% | 2% | 5% | 10% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Current-control consensus | 174.0 | 170.6 | 178.3 | 158.6 | 141.7 | 141.7 | 141.6 | 120.1 |
| Current-control upstream | **362.1** | **279.6** | **219.1** | **174.9** | **164.9** | **164.9** | **164.9** | **133.6** |
| T6 alone | 363.2 | 349.9 | 277.9 | 205.7 | 190.9 | 190.9 | 190.4 | 147.1 |
| T9 alone | 45.8 | 57.8 | 48.2 | 37.5 | 38.5 | 38.5 | 38.5 | 31.4 |
| T6+T9 correction mean | 140.3 | 107.8 | 97.1 | 88.9 | 111.9 | 111.9 | 111.9 | 102.9 |
| **T6+T9 upstream** | 193.2 | 177.0 | 148.9 | 139.2 | 139.3 | 139.3 | 139.3 | 125.5 |
| T6+T9 upstream minus current-control upstream | −168.9 | −102.6 | −70.2 | −35.7 | −25.6 | −25.6 | −25.6 | −8.1 |

T6 has substantial standalone conditional signal. T9 is not useful as independently ranked alpha; it earns retention only as a diverse conversion coordinate within the downstream MC1 map. The current-control upstream is stronger at these raw May–July tails. No claim is made that the raw T6+T9 score dominates the control.

### 4.3 MC1 mapped-EV ranking diagnostics

This matched diagnostic isolates the third layer. It uses the **BCF-family MC1 expected-EV map** as the selection coordinate, attaches the same canonical policy label source, and restricts both rows to the exact **45,282 valid common candidate IDs** from May--July 2026. It does **not** apply admission or portfolio constraints; it answers whether MC1 produces a better expected-value ordering than the archived current-control map.

| MC1 map | Global top 1% | 2% | 5% | 10% | Timestamp-local top 1% | 2% | 5% | 10% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline BCF MC1 | 267.97 | 235.65 | 187.59 | 137.45 | 143.59 | 143.59 | 127.78 | 100.34 |
| **T6+T9 BCF MC1** | **386.85** | **308.91** | **215.68** | **158.62** | **169.62** | **169.62** | **150.40** | **114.26** |
| T6+T9 minus baseline | +118.88 | +73.26 | +28.09 | +21.17 | +26.03 | +26.03 | +22.62 | +13.92 |

The apparent equality of the timestamp-local top-1% and top-2% values is expected after top-30% routing: many timestamps have too few routed candidates for those two percentile cuts to select different rows. This is a ranking diagnostic only; the live-like economic comparison remains the dual-admission, constrained auction below.

### 4.4 Strict-MC1 constrained-portfolio replay

This is the valid full-stack comparison: same May–July 2026 candidate IDs, canonical rich-policy labels, six-month prequential MC1 fits, dual current/BCF MC1 admission at **+50 bps**, and one constrained chronological auction.

| MC1 input set | Portfolio entries | Net EV / trade | Total net bps | Worst month | Worst week | Max drawdown | Δ total bps vs control |
|---|---:|---:|---:|---:|---:|---:|---:|
| Live-control matched | 1,896 | **151.31** | 286,875 | 127.85 | **97.11** | −17.87% | — |
| T6 only | 2,168 | 141.95 | 307,740 | 131.78 | 89.13 | −18.21% | +20,865 |
| T9 only | 2,027 | 128.47 | 260,413 | 114.28 | 82.20 | −22.89% | −26,462 |
| **T6+T9** | **2,185** | 142.35 | **311,037** | **130.17** | 81.56 | **−14.98%** | **+24,161** |

T6+T9 is the best validated two-head aggregate: it adds 289 entries, +24.2k total net bps, improves the worst month by +2.32 bps/trade, and reduces drawdown by 2.88 percentage points. Its trade-level EV is 8.95 bps lower and its worst week is 15.54 bps lower than the control. It is consequently a qualified research challenger, not an automatic promotion.

### 5. Historical challenger support-ledger status

The earlier strict-MC1 table used the unweighted target-funnel receipt for both heads. T9’s selected `S5_tbm_coarse` support weighting was independently positive in the May–July raw forward screen, but it cannot be mixed with older unweighted outputs inside a strict prequential MC1 map.

Two target-free ledgers have been rebuilt for November 2025–July 2026:

| Head | Contract | Purpose |
|---|---|---|
| T6 | uniform, one `cap80_ordinary` head | homogeneous six-month MC1 history |
| T9 | `S5_tbm_coarse`, one `cap120_equal_month` head | homogeneous six-month MC1 history |

Each ledger contains target-free monthly receipts for November 2025 through July 2026. The independent support audit passed: all 18 family/month receipts are identity-unique, retain the required causal base-field coverage, and contain no policy or outcome columns. The next MC1 comparison must use only these homogeneous receipts; it must not mix the earlier unweighted T9 output into the T9 history. No additional historical head will be reintroduced.

Audit receipt:

```text
data_perp/artifacts/strict_r3_o3v2_t6t9_homogeneous_support_audit_20260825_v1/
  correctness_report.json  # passed: true
```

### 6. Historical challenger promotion gates

1. Exact target-free score and identity audit.
2. Six complete prior months under one identical head/support contract.
3. Constrained-portfolio total net contribution is not bought through unacceptable drawdown, worst-week, or concentration deterioration.
4. No material deterioration in EV per trade unless the increase in deployable entries and total net bps is sufficient under the same risk limits.
5. Stable month-level result, not one isolated trade, month, or symbol.
6. No change to the live bundle until a separately approved promotion decision.

### 7. Historical challenger scripts and artifacts

| Purpose | File |
|---|---|
| Enhanced-base source and original full-stack challenger | `scripts/run_strict_r3_enhanced_base_live_stack_challenger.py` |
| Target/head generation | `scripts/run_strict_r3_o3v2_target_funnel.py` |
| Support-weight generation | `scripts/run_strict_r3_o3v2_support_funnel.py`, `scripts/run_strict_r3_o3v2_support_funnel_v3.py` |
| Strict MC1 and constrained portfolio comparison | `scripts/run_strict_r3_o3v2_mc1_portfolio.py` |
| Fixed T6+T9 contract | `data_perp/artifacts/strict_r3_o3v2_t6t9_consensus_contract_20260825_v1/selected_physical_slots.json` |
| Validated initial T6+T9 MC1 result | `data_perp/artifacts/strict_r3_o3v2_mc1_portfolio_selected_slot_aggregate_20260825_v1_T6_T9/` |
| Current homogeneous T6 ledger | `data_perp/artifacts/strict_r3_o3v2_t6_uniform_homogeneous_202511_202607_20260825_v3/` |
| Current homogeneous T9 ledger | `data_perp/artifacts/strict_r3_o3v2_t9_tbm_homogeneous_202511_202607_20260825_v3/` |

### 8. Historical challenger causality requirements

- Candidate features are point-in-time and target-free.
- The base route is recomputed at each timestamp from `enhanced_base_bps`; no held-period percentile is used.
- T6/T9 train only on labels resolved before the pre-fit reserve.
- Support weights exist only in training and never enter score columns.
- Held score panels are persisted before the policy ledger is joined.
- MC1 uses six complete prior calendar months; current and BCF maps remain separate until the declared dual-admission rule.
- All portfolio metrics use the same reconciled rich-policy label source.
