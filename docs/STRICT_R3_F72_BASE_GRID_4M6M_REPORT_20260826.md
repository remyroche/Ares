# Strict-R3 F72 base-grid reconstruction and downstream decision

**Status:** completed offline research funnel; no promotion.

## Decision

The rebuilt strict-OOF F72 B score does not improve the full downstream
stack.  The router-matched incumbent remains the winner.  The F72 variants
were therefore not refined or promoted.

This is a decision about the full causal stack, not about a standalone
head metric.  Every arm below refits the same downstream consumers before
evaluation: five consensus heads, the two strict prequential MC1 family
maps, dual 50-bps admission, and one chronological portfolio with the
frozen rich policy.

## Scope and common contract

* Long-only, January--June 2026 held period.
* Point-in-time candidates and target-free 120-field downstream input
  contract.
* F72 router population: timestamp-local top-50% router eligibility.
* Consensus fit: the four preceding fully resolved months, with a separate
  28-day embargo/reserve.
* MC1 fit: four strictly preceding resolved calendar months, separately for
  current and BCF families.
* Admission: **both** family maps must estimate at least +50 bps.
* Portfolio: one global chronological state, frozen constraints, and the
  rich execution policy (including the existing smooth capital protection).
* Costs and policy labels are from the reconciled canonical-policy successor
  ledger; invalid paths are excluded from supervised fitting.

These are research results, not untouched promotion evidence.

## Repaired source lineage

The initial F72 comparison could not fit its earliest September 2025 fold
because the target-free B history began too late.  The repair reconstructs
causal router and B-score history once, rather than rebuilding it for each
grid cell.

| Layer | Artifact | Evidence |
|---|---|---|
| Reconciled policy labels | `strict_r3_enhanced_base_rich_policy_labels_reconciled_successor_20260826_v1` | 2,820,951 rows; 2,770,593 valid; invalid rows retained for coverage but excluded from fitting |
| Early causal feature history | `strict_r3_f72_early_router_features_20260826_history_contiguous_v1` | Apr-2024--Mar-2025; every month has at least 94.5% of rows with at least 90% of the frozen feature fields |
| Strict F72 B history | `strict_r3_f72_b_fullhistory_composite_20260826_v1` | Apr-2025--Jul-2026; early B scores use the no-score-geometry path, which was bit-identical to the established September scorer |
| E/T restricted scores | `strict_r3_incumbent_et_component_scores_f72router50_fullhistory_20260826_v1` | Target-free E/T values only on the frozen F72 router identities |

The early B path deliberately omits score-geometry coordinates that cannot
exist before the later enhanced score source begins.  It does not consume
outcomes, labels, or held-window ranks.  A September parity check against
the established scorer had 55,116 matching rows and zero score delta.

## Coarse grid

The coarse screen intentionally tested only materially different choices:

| Arm | Upstream coordinate |
|---|---|
| Router-matched incumbent control | Existing immutable upstream on the frozen router-50 identity intersection |
| B30/E35/T35 | Weighted timestamp-local rank: 30% F72 B, 35% incumbent E, 35% incumbent T |
| B60/E20/T20 | Weighted timestamp-local rank: 60% F72 B, 20% incumbent E, 20% incumbent T |
| B100 F72 | Strict-OOF F72 B timestamp-local rank; incumbent E/T raw geometry retained for downstream features |

## Full-stack constrained results

All values are net bps after the frozen rich policy and portfolio replay.

| Arm | Entries | Net EV/trade | Total net bps | Worst month | Worst week | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Live-source baseline | 3,627 | +155.60 | +564,362 | +128.42 | +43.58 | -27.81% |
| **Router-matched incumbent** | **3,787** | **+164.21** | **+621,849** | **+120.02** | **+84.45** | **-23.21%** |
| B30/E35/T35 | 4,078 | +134.97 | +550,393 | +85.41 | +58.87 | -32.21% |
| B60/E20/T20 | 3,930 | +149.04 | +585,735 | +94.73 | +68.56 | -30.32% |
| B100 F72 | 3,150 | +159.79 | +503,329 | +93.55 | +31.12 | -37.56% |

### Delta versus router-matched incumbent

| Arm | Entries | Net EV/trade | Total net bps | Worst month | Worst week | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| B30/E35/T35 | +291 | -29.24 | -71,456 | -34.61 | -25.59 | -9.00pp |
| B60/E20/T20 | +143 | -15.16 | -36,114 | -25.29 | -15.89 | -7.12pp |
| B100 F72 | -637 | -4.42 | -118,520 | -26.48 | -53.33 | -14.36pp |

The control is also better than the live-source baseline on the matched
replay: +160 entries, +8.61 bps/trade, +57,487 net bps, +40.87 bps in worst
week, and 4.60 percentage points less drawdown.  Its worst month is 8.40
bps lower but remains positive.

## Relation to the currently active live stack

The `live-source baseline` above is a historical matched control, **not** a
byte-identical replay of the contract active in live operations.  The active
sealed state is `strict_r3_kraken_live_state_v113_v181_hash_stability_capacity_fallback_live.json`, using inference overlay
`strict_r3_inference_overlay_long_v153_v181_hash_stability_capacity_fallback.json`.

Its economic stack is the BCF/current-v5 dual-MC1 contract:

```text
current-v5 top-30% timestamp-local route
and BCF score-family route
→ one family-specific MC1 map per score family
→ both expected EVs >= 30 bps
→ BCF MC1 EV auction priority
→ common 2-entry / 8-position / 80%-margin auction
→ rich SimplePolicyOptimiser parent plus Adaptive Exit V1 overlay
→ live-only executable spread, impact, and delay recheck
```

The grid control uses an offline historical reconstruction with a 50-bps
dual-map gate, a frozen router-50 identity source, a refit five-head
consensus layer, and no live executable-friction rejection.  It cannot be
promoted or called superior to the live stack from its own metrics.

The closest available comparison is the January--June 2026 historical
portfolio ledger for the active dual-30 BCF/current architecture.  It is
still an idealized historical execution replay, not a validation of actual
five-minute live fills.

| Contract, Jan--Jun 2026 | Entries | Net EV/trade | Total net bps | Worst month | Worst week | Max DD |
|---|---:|---:|---:|---:|---:|---:|
| Active BCF/current dual-30 historical proxy | 3,430 | **+171.81** | 589,299 | **+136.11** | +73.97 | -45.88% |
| Router-matched incumbent control | **3,787** | +164.21 | **+621,849** | +120.02 | **+84.45** | **-23.21%** |

The control adds 357 entries and +32,549 total bps, but loses 7.60 bps per
trade and 16.08 bps in its worst month.  Its drawdown figure is not
comparable because the portfolio state, candidate universe, threshold,
parent-policy replay, and live friction treatment differ.  This comparison
therefore changes no active live component: the F72 variants remain offline
non-promoted challengers.

## Monthly constrained portfolio economics

Each cell is `entries / net bps per realised trade`.

| Month | Live baseline | Router-matched incumbent | B30/E35/T35 | B60/E20/T20 | B100 F72 |
|---|---:|---:|---:|---:|---:|
| 2026-01 | 689 / +128.42 | 565 / +160.41 | 533 / +146.07 | 527 / +142.30 | 349 / +170.71 |
| 2026-02 | 435 / +203.02 | 441 / +222.15 | 383 / +225.76 | 397 / +232.65 | 335 / +212.23 |
| 2026-03 | 817 / +165.12 | 802 / +166.34 | 835 / +161.76 | 801 / +166.97 | 670 / +184.98 |
| 2026-04 | 767 / +155.57 | 698 / +194.96 | 901 / +130.35 | 907 / +155.50 | 816 / +156.14 |
| 2026-05 | 635 / +130.52 | 739 / +133.57 | 874 / +85.41 | 888 / +94.73 | 681 / +93.55 |
| 2026-06 | 284 / +177.67 | 542 / +120.02 | 552 / +106.73 | 410 / +145.06 | 299 / +192.64 |

All arms have six positive months, but the F72 substitutions weaken the
lower tail and/or reduce total economic contribution.  B100 has respectable
per-trade precision in several months, but it gives up too much capacity and
has the weakest weekly and drawdown profile.

## Causality and coverage audit

Every completed arm has:

* ten monthly downstream folds from September 2025 through June 2026;
* a documented four-month pre-reserve training interval and 28-day reserve
  before each held month;
* fully target-free feature coverage of 100% in the scored months;
* separate current and BCF MC1 fit audits, beginning only after sufficient
  strictly prior policy outcomes exist; and
* 100% resolved policy-outcome coverage for the January--June 2026
  constrained replay.

No score, map, rank, feature, or training input uses held outcomes.  No
held-window percentile is used in the construction of the upstream score.

## Result and next action

Keep the existing upstream contract.  Do **not** promote or further tune
F72 B as a direct B replacement from this result.  Subsequent research may
reuse the restored causal history, but should not treat the standalone F72
score's historical rank uplift as downstream economic proof.

## Primary result artifacts

* `data_perp/artifacts/strict_r3_f72_fullhistory_coarse_control_4m6m_20260826_v1`
* `data_perp/artifacts/strict_r3_f72_fullhistory_coarse_b30_4m6m_20260826_v1`
* `data_perp/artifacts/strict_r3_f72_fullhistory_coarse_b60_4m6m_20260826_v1`
* `data_perp/artifacts/strict_r3_f72_fullhistory_coarse_b100_4m6m_20260826_v1`
