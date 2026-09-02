# P8U Router50 + F72 Base + Under F120 — Canonical Research Handover v6

## Status

This is the canonical long-only research handover as of 2026-08-28. It supersedes the capacity interpretation in [the prior v5 handover](P8U_ROUTED_F72_UNDERF120_RESEARCH_CANONICAL_20260828.md), which is retained as historical evidence. The versioned contract is [v6](../config/strict_r3_p8u_routed_f72_underf120_research_canonical_20260828_v6.json).

This is not a live Kraken contract. It does not authorize exchange access, execution, or a live-bundle change. Evaluation covers 2025-11-01 through 2026-08-27 UTC. August is retrospective reconciliation evidence, not untouched promotion evidence. Policy outcomes are a 15-minute rich-policy proxy, not exact one-minute live-exit evidence.

## Architecture

    target-free point-in-time candidates
      → P8U Router: retain top 50% of identities per timestamp
      → F72 Base: Raw-bps CatBoost opportunity rank
      ├─ BCF family: Base rank only
      └─ Current family: 75% Base rank + 25% Under F120 rank
           → separate strict-prequential BCF and Current MC1 EV maps
           → admit only if both maps are at least +50 bps
           → shared chronological portfolio auction, ordered by BCF MC1 EV
           → rich policy outcome and committed-margin constraints

| Layer | Purpose | Authority |
|---|---|---|
| Candidate materialisation | Establish complete decision-time market universe | No future path, outcome, or label-validity input |
| P8U Router | Remove lowest-scoring half within timestamp | Candidate identity only; numeric Router score is not a later model feature |
| F72 Base | Rank economic opportunity | Base timestamp rank |
| Under F120 | Identify Base under-confidence | 25% of Current rank; no direct auction authority |
| BCF / Current | Preserve precision and independently require confirmation | BCF = Base; Current = .75 Base + .25 Under |
| Dual MC1 | Map score families to expected policy-net bps | Both maps must clear +50 bps |
| Auction | Enforce capital and concurrency limits | Priority = BCF MC1 expected EV |

## Causality contract

All Router/Base/Under candidate features are point-in-time. Target-free Base and Under scores are persisted before rich-policy outcomes are joined. The canonical policy-label materialisation is [canonical_reconciled_policy_labels.parquet](../data_perp/artifacts/strict_r3_p8u_router_policy_label_successor_fullprehistory_aug27_20260828_v1/canonical_reconciled_policy_labels.parquet).

The independent [replay realism audit](../data_perp/artifacts/strict_r3_p8u_f72_underf120_replay_realism_audit_20260828_v1/) verifies target-free score persistence, distinct BCF/Current coordinates, resolved-label chronology, one shared chronological portfolio, and no live/exchange mutation.

## Router: P8U Router50

P8U is a 30-field LightGBM Rank-XENDCG Router. It retains exactly the top 50% candidate IDs at each decision timestamp. Every retained identity flows to Base; there is no later Base top-percent gate.

| Property | Frozen setting |
|---|---|
| Target | P8u floor-100/cap-250: policy net <= +100 bps is grade 0; positive excess is capped at +250 bps and graded at 31.25/62.5/109.375/171.875 bps excess |
| Query and weights | Exact timestamp × long-side query; each query loss sums to 1; sqrt-excess raises row weight from 1x to 2x |
| Objective | Rank-XENDCG; gains [0,1,2,4,7,11]; truncation 12 |
| Fit | Three months; 28-day resolved-label reserve; 120,000-row cap |
| HPO | 1,000-tree ceiling; LR .0567571; depth 4; 15 leaves; child max(500, 1.7038% of rows); min gain .00321538; feature fraction .787355; bagging .727909; L1 .0141675; L2 .216746; max bin 127; 20% chronological inner validation; early stop 30; seed 1729 |

The 30 selected causal fields are liq_stop_safety_short_atr, mark_perp_dislocation, rv_rel_universe, range_24h_pct, ffd_rv_24h_04, upside_semivariance_24, ffd_rv_6h_06, t_be_proxy, dist_prior_day_high, upside_semivariance_8, dist_rolling_7d_high, asset_atr_level, vov_mad_60, vov_iqr_20, seasonality_strength, realized_volatility_24h, cvar_5pct, dist_prior_day_low, liquidity_ratio_peer_resid, mark_trigger_risk_10h, t_pl_proxy, price_rv_7d_robust_z, range_volatility, rvol_hod_base, range_per_volume, price_rv_15d_robust_z, ob_depth_l20_to_qv_24h, beta_eth_24h, beta_btc_24h, and rv_48h.

The exact field order is hash-bound in [run_contract.json](../data_perp/artifacts/strict_r3_p8u_router_oof_apr25_jul26_successorlabels_20260828_v1/run_contract.json), SHA-256 c787eb4c432dee34b200aa4a861e695a9597e16adb24376510dedb47d550d284. August 1–27 Router50 recall is 66.62%, 72.60%, 76.62%, and 80.15% for valid policy opportunities above +50/+100/+150/+200 bps. The optimized Router was rejected because it did not improve downstream economics.

## Base: F72 Raw-bps CatBoost QueryRMSE

F72 is the precision-first opportunity ranker and fully defines BCF. It is not the legacy R3 model. Training clips rich-policy net bps at train-only P2/P98, then builds six equal-width ordinal grades in every training fold.

| Property | Frozen setting |
|---|---|
| Contract | P8U_RAW_BPS_CATBOOST_QUERYRMSE_F72_TAIL125 |
| Query | Exact decision timestamp × long side |
| Weights | tail_linear_125: raw 1 + .125 × grade, then normalized within query to mean 1 and bounded to .5–2 |
| Fit | Three months; 28-day resolved-label reserve; 60,000 complete-query cap |
| HPO | 2,000-tree ceiling; early stop 30; depth 5; LR .0650994; feature fraction .800651; bagging .709605; L2 2.235726; random strength .942890; seed 1729 |

The exact 72-field contract and selection receipt is [selection.json](../data_perp/artifacts/strict_r3_b0_family_addback_20260826_v1_policy_ordinal_base_g3/selection.json). It contains causal mark/perp dislocation, multi-horizon volatility and FFD volatility, trend/efficiency, price/OI and funding state, support/resistance location, cross-asset beta/correlation, seasonality, liquidation/exhaustion, and liquidity fields. It must be loaded from that artifact by name and order rather than reselected at inference.

Strict-OOF Nov-2025–Jul-2026 evidence: ScoreStable 1.714 and timestamp-local DTP2/DTP5/DTP10 of +190.30/+131.26/+86.14 bps. In August, Base Top-1/2/5/10/15% per timestamp is +210.83/+163.63/+93.02/+53.37/+33.89 net bps, with >+50-bps hit rates 69.24/65.53/57.19/51.40/47.87%. See [the Base handover](P8U_BASE_PRECISION_PRESERVATION_HANDOVER_20260828.md).

## Under F120: confirmation head

Under F120 is the only retained correction head. It seeks Base under-confidence: target 1 requires a valid path to reach the .5-ATR trailing-activation level and policy net minus a prequential Base anchor to be at least +100 bps. The 14-day expanding isotonic anchor exists only for training-label construction.

| Property | Frozen setting |
|---|---|
| Objective | LightGBM Rank-XENDCG; timestamp query; gains [0,1,2,4,7,11,16,24]; truncation 12; sigmoid 1 |
| Fit | Four months; 28-day resolved-label reserve; 100,000 query-safe rows; uniform weights |
| HPO | 260 trees; LR .045; depth 4; 15 leaves; min child 350; feature fraction .80; bagging .82; L1 .02; L2 8 |
| Inputs | 120 selected causal fields plus 9 deterministic Base-query geometry fields |
| Score authority | Current = .75 Base + .25 Under; no direct auction priority |

The exact 120 fields, their order, feature-family accounting, selection trace, and hash are in [under_f120.json](../data_perp/artifacts/strict_r3_p8u_meta_under_fullfeatures_selection_20260828_v2/contracts/under_f120.json), SHA-256 85d6ada0e640e88a431801c7cd530931de818f700594fdcf4191cbe75b67f727. Selection was hygiene → cross-era conditional IC/CMI/redundancy → randomized subspace gain/tail-SHAP → group MDA → bounded SStableMeta. Under conditional MI given F72 is .1384 nats pre-August and .08081 nats in August. It improves confirmation/hit-rate structure but may dilute raw Base tail bps; this is why BCF and Current stay separate.

## BCF/Current and MC1 dual admission

BCF is Base rank only. Current is .75 Base rank + .25 Under rank. Each receives its own absolute expected-policy-net mapper. Inputs are final_score, base_rank42, conditional_consensus_rank, upstream, ordinary_shadow_consensus_rank, and correctness_rank.

MC1 is HistGradientBoostingRegressor: depth 2, 80 iterations, LR .04, L2 20, minimum leaf 100, seed 1729. It uses train-only P2/P98 target clipping, a ten-band monotone precision-shrunk score curve, day-balanced sampling (top 50 daily rows plus up to 250 random remaining rows), and a 50,000-row cap. A 21-day, 10%-trimmed, prior-resolved residual shift adjusts the static map once daily.

**Provenance blocker:** the persisted dual-MC1 artifact records a three-month fit window, whereas the current runner source declares six months. The source was modified after the artifact. A hash-bound rerun or source receipt must reconcile this before any live bundle is created. The shift is causal, but it has not reduced broad daily calibration error versus the static map; it remains research evidence, not a validated live calibrator.

Admission is:

    BCF MC1 expected policy-net EV >= +50 bps
    AND Current MC1 expected policy-net EV >= +50 bps

Only then is an item auctioned, ordered by BCF MC1 EV. Policy net already embeds the fixed 100-bps round-trip cost; it is not charged twice.

## Rich policy and portfolio

The parent policy is [frozen_policy.json](../data_perp/artifacts/strict_r3_rich_policy_smooth_protection_long_20260817_v1/frozen_policy.json), SHA-256 `e7508e523d6aaa8a03b0df8009f5e766bfca83c4f1b5fb1c1bd7816901041575`. It enters on the decision 15-minute open, uses 48 completed 15m bars/H12 and signal-time Wilder-14 ATR. It includes a volatility-transformed hard stop, smooth capital protection after 1.5 ATR MFE with strength .5 and power 1.5, dynamic trailing, fast-adverse exit, and H12 timeout. Protection is ratcheting only.

The shared portfolio permits eight open positions, 80% of wallet in **committed initial margin at entry**, and two new entries per timestamp in the frozen control. Marked exposure can exceed 80% after price moves; it is diagnostic and no longer reserves new capacity. The repair is in [portfolio_policy_replay.py](../extreme_price_movements/portfolio_policy_replay.py), with a regression test in [test_portfolio_policy_replay.py](../extreme_price_movements/tests/test_portfolio_policy_replay.py).

## Sealed 30/40/50-bps × capacity results

All results are rich-policy net bps, 2025-11-01 through 2026-08-27. Raw admits are after the dual MC1 gate but before the auction.

| Floor / cap | Raw admits / EV | Constrained entries / EV | Total bps | Worst month / week | Max DD |
|---|---:|---:|---:|---:|---:|
| 30 / 1 | 45,136 / +106.59 | 6,751 / +148.47 | +1,002,350 | +89.01 / +26.25 | −31.84% |
| 30 / 2 | 45,136 / +106.59 | 9,126 / +119.06 | +1,086,505 | +64.65 / +32.52 | −42.69% |
| 30 / 3 | 45,136 / +106.59 | 9,505 / +115.99 | +1,102,504 | +58.74 / +19.46 | −42.09% |
| 30 / 4 | 45,136 / +106.59 | 9,530 / +114.70 | +1,093,109 | +55.52 / +17.28 | −42.52% |
| 40 / 1 | 35,837 / +123.65 | 6,590 / +150.82 | +993,934 | +88.63 / +26.25 | −31.84% |
| 40 / 2 | 35,837 / +123.65 | 8,916 / +123.20 | +1,098,447 | +65.08 / +36.96 | −43.38% |
| 40 / 3 | 35,837 / +123.65 | 9,271 / +118.89 | +1,102,191 | +57.95 / +40.88 | −43.74% |
| 40 / 4 | 35,837 / +123.65 | 9,300 / +117.24 | +1,090,301 | +56.21 / +36.16 | −42.81% |
| 50 / 1 | 29,474 / +137.13 | 6,342 / +153.41 | +972,948 | +88.98 / +26.25 | −31.84% |
| **50 / 2 frozen control** | **29,474 / +137.13** | **8,461 / +129.11** | **+1,092,386** | **+68.28 / +39.42** | **−31.77%** |
| 50 / 3 | 29,474 / +137.13 | 8,748 / +125.72 | +1,099,829 | +61.64 / +38.75 | −33.52% |
| 50 / 4 | 29,474 / +137.13 | 8,780 / +124.49 | +1,093,032 | +58.75 / +37.85 | −34.01% |

The frozen 50/2 arm has 28.20 trades/calendar day, no zero-trade days, one day below five trades, nine days below ten, and a maximum 44 trades/day. Daily Q5/Q10/Q15/Q20 EV/trade is −23.35/+13.10/+32.47/+50.67 bps; weekly Q5/Q10/Q15/Q20 is +53.69/+63.11/+70.25/+84.60 bps.

| Month | Entries | Trades/day | Net EV/trade |
|---|---:|---:|---:|
| 2025-11 | 926 | 30.87 | +147.61 bps |
| 2025-12 | 815 | 26.29 | +95.20 bps |
| 2026-01 | 851 | 27.45 | +140.49 bps |
| 2026-02 | 462 | 16.50 | +170.74 bps |
| 2026-03 | 914 | 29.48 | +167.71 bps |
| 2026-04 | 855 | 28.50 | +167.66 bps |
| 2026-05 | 979 | 31.58 | +104.26 bps |
| 2026-06 | 752 | 25.07 | +153.08 bps |
| 2026-07 | 1,046 | 33.74 | +68.28 bps |
| 2026-08 (1–27) | 861 | 31.89 | +109.67 bps |

Exit counts: 3,837 smooth-capital-protection, 2,056 trailing, 1,740 H12 timeout, 775 hard-stop, and 53 fast-adverse. Authoritative outputs are [capacity sweep v3](../data_perp/artifacts/strict_r3_p8u_f72_underf120_gate_capacity_sweep_aug27_20260828_v3_committed_margin/), [quality v5](../data_perp/artifacts/strict_r3_p8u_f72_underf120_extended_quality_aug27_20260828_v5_committed_margin/), and [realism audit](../data_perp/artifacts/strict_r3_p8u_f72_underf120_replay_realism_audit_20260828_v1/).

## Scripts and promotion guide

| Layer | Primary implementation |
|---|---|
| Router | [run_strict_r3_economic_recall_router.py](../scripts/run_strict_r3_economic_recall_router.py), [run_strict_r3_router_hpo.py](../scripts/run_strict_r3_router_hpo.py) |
| Base | [run_strict_r3_p8u_precision_preservation_weight_funnel_v1.py](../scripts/run_strict_r3_p8u_precision_preservation_weight_funnel_v1.py), [run_strict_r3_p8u_precision_preservation_winner_hpo_v1.py](../scripts/run_strict_r3_p8u_precision_preservation_winner_hpo_v1.py) |
| Under | [select_strict_r3_p8u_meta_fullfeatures_v1.py](../scripts/select_strict_r3_p8u_meta_fullfeatures_v1.py), [run_strict_r3_p8u_meta_lgbm_objective_screen_v1.py](../scripts/run_strict_r3_p8u_meta_lgbm_objective_screen_v1.py) |
| MC1 | [run_strict_r3_enhanced_base_live_stack_challenger.py](../scripts/run_strict_r3_enhanced_base_live_stack_challenger.py) |
| Policy | [materialize_strict_r3_frozen_rich_policy_15m_labels.py](../scripts/materialize_strict_r3_frozen_rich_policy_15m_labels.py) |
| Replay/audit | [report_strict_r3_p8u_f72_underf120_gate_sweep_v1.py](../scripts/report_strict_r3_p8u_f72_underf120_gate_sweep_v1.py), [report_strict_r3_p8u_extended_quality_v1.py](../scripts/report_strict_r3_p8u_extended_quality_v1.py), [audit_strict_r3_p8u_replay_realism_v1.py](../scripts/audit_strict_r3_p8u_replay_realism_v1.py) |

A future live implementation must, in this order:

1. Reconcile the three-month MC1 artifact versus six-month runner-source discrepancy.
2. Seal one bundle with Router, F72, Under, imputers, feature order, rank references, MC1 models/band curves/shift state, policy state, and portfolio state.
3. Create the full target-free candidate universe at each decision timestamp; fail closed only rows lacking required contemporaneous inputs.
4. Apply Router50 → Base → Under → BCF/Current → dual MC1 >= 50 → BCF-EV auction in that exact order.
5. Run an offline exact-identity inference/replay parity test on persisted target-free inputs.
6. Materialise exact 1m exits and verify stateful exit parity.
7. Freeze the contract, collect later untouched evidence without retuning, and obtain a separate explicit exchange-writing authorization.

## Preproduction integrity gate — 2026-08-28

The selected research contract is now sealed into the immutable, hash-bound
[P8U preproduction bundle](../data_perp/artifacts/strict_r3_p8u_preproduction_bundle_20260828_v4/bundle.json), SHA-256 `cefe2a2db0b39ecc19a1561b3cee7bac94d9406a523f167c7b28a16e49dc5964`.
The [corresponding audit](../data_perp/artifacts/strict_r3_p8u_preproduction_bundle_20260828_v4/audit/correctness_report.json) verifies every referenced artifact before scoring and records the automatically generated feature inventory:

| Contract | Count | Rule |
|---|---:|---|
| Router | 30 | Score the complete point-in-time candidate population. |
| F72 Base | 72 | May consume only exact Router50 identities. |
| Under F120 | 120 | May consume only exact Router50 identities. |
| Complete materialisation union | 175 | Must exist in the active causal feature panel before Router scoring. |

The Router boundary is exact timestamp-local top 50%, with descending router
score, `candidate_id` as deterministic tie-break, and `ceil(0.50 × valid
candidates)` capacity. Base, Under, BCF, Current, and both MC1 maps are
identity-checked subsets of that result. There is no full-universe fallback.

Feature calculation restores and advances the append-only source panel plus
rolling, derived, Final14, and order-book state. It requests the sealed union
automatically and is prohibited from recomputing full raw history for a new
timestamp. Do not split feature calculation into Router-only and downstream
passes until a two-stage state graph has demonstrated byte-parity; for now,
compute the 175-field union once, then form the Router50 matrices.

This is deliberately **preproduction only**. The current August cached panels
cover Router and F72 but omit 84 Under fields, so they are explicitly rejected
as a shortcut. The recorded full causal Under panel contains 1,412 fields and
covers all 175 sealed fields; see the separate
[full-panel coverage audit](../data_perp/artifacts/strict_r3_p8u_preproduction_bundle_20260828_v4/audit_full_under_panel/correctness_report.json).
That proves the required inputs can be materialised, but a fresh incremental
materialisation must still prove exact feature parity before it becomes the
live source. In addition, serialised Router/Base/Under/MC1 packages are absent,
the MC1 three-versus-six-month receipt discrepancy remains unresolved, and
the policy is still a 15-minute proxy rather than an exact one-minute exit
parity contract. `order_submission` remains false until all four conditions
are remedied and a new explicitly approved live bundle is sealed.

### Inference efficiency contract

The sealed union is 175 fields (30 Router, 72 F72 Base, 120 Under F120),
with 11/18/25 Router–Base/Router–Under/Base–Under overlaps. Only three union
fields are currently in the incremental engine's audited long-memory fallback
set: `mkt_pct_price_up_oi_down_4h`, `price_rv_15d_robust_z`, and
`prior_volatility`. Every other selected field is already on the append-only
stateful path. These three are the priority for a future exact-state parity
promotion; do not replace their exact fallback with an approximation.

The materializer now accepts the bundle-generated
`required_feature_plan.json` via `--requested-features-json` and must use an
isolated `--feature-cache-namespace` derived from the sealed bundle. It then
loads only the 175 required model fields, restores the persistent transform
state, advances it one timestamp, and writes only the new rows.

On a 55,080-row August full causal panel, read-only projected I/O for the 178
identity-plus-model columns took 46.8 ms; Arrow-to-pandas conversion 5.4 ms;
the 85-row latest timestamp slice 3.3 ms; Router50 gate 3.2 ms; and Base plus
Under matrix construction 2.6 ms. These are not an end-to-end live latency
claim—the historical materialisation, source refresh, model scoring, and
state commit must still be benchmarked after an exact incremental bootstrap—
but they show the Router boundary and array matrix handling are not the
bottleneck.

Priority sequence for latency work:

1. Persist and warm-load the P8U source/rolling/derived/Final14/order-book
   state in one long-lived inference worker; atomically snapshot after each
   successful timestamp. Never deserialize models or recompute raw history on
   the hourly critical path.
2. Use Arrow column projection for the 175-field union and `float32` matrices
   for Router/Base/Under/MC1; the stateful materializer already performs
   vectorised timestamp/symbol gathering rather than a Python lookup loop.
3. Keep source I/O bounded and parallel, but score all four models in the same
   warmed process. Avoid process pools around loaded models or state; they
   copy memory and make state ownership ambiguous. Reserve the existing
   bounded process pool for independent CPU-heavy feature families only.
4. Benchmark an optional two-stage feature graph only after it produces
   byte-identical output: full-universe primitive/cross-sectional state →
   Router → Router50-only row-local downstream derivations. It may reduce
   post-router work by about half, but cannot change full-universe state
   semantics.

## Limitations

- August is reconciliation, not promotion evidence.
- This is 15m policy proxy evidence, not exact live 1m execution evidence.
- The MC1 train-window provenance mismatch blocks live packaging.
- The causal 21-day MC1 adjustment is not yet shown to improve broad calibration.
- Positive monthly EV does not negate the simulated −31.77% maximum drawdown.
- Any new feature, model, score blend, mapper, threshold, or policy creates a named challenger until separately validated.
