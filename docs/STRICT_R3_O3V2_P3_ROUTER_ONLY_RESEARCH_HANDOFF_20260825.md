# Strict-R3 O3-v2 P3 Router-only Research Handoff

**Published:** 2026-08-25  
**Status:** `RESEARCH_CHALLENGER_NOT_LIVE`  
**Scope:** long-only, offline research. This document records a strictly-OOF route experiment. It changes no live bundle, execution policy, exchange process, or admission threshold.

## Current status

**P3 is a completed pure-router challenger, not the final selected router.**
It remains eligible for the final cross-target feature-selection and portability
funnel, but it must never be promoted into an upstream/base coordinate.  The
selection decision is deliberately deferred until the common target, gain,
objective, truncation, feature, history and weighting evidence is complete.

Its valid architecture is **P3 as a pure timestamp-local route**, not a
replacement score:

```text
strict-OOF enhanced base / B0 / efficiency / timing
  -> P3 router: retain top 30% within each timestamp
  -> T6 + T9 correction heads
  -> Current and BCF prequential MC1 maps
  -> dual MC1 >= +50 bps admission
  -> BCF mapped-EV priority and a chronological constrained portfolio
```

The prior result that wrote P3 into `enhanced_base_bps`, `base_bps`, `efficiency_bps`, and `timing_bps` is invalid and diagnostic-only. The final run documented here preserves all four source coordinates exactly; P3 is absent from the consensus input panel.

## Frozen P3 challenger contract

| Component | Contract |
|---|---|
| Target | `P3_net_50_75_125_200_350`, an ordinal policy-net economic target |
| Model | LightGBM LambdaRank |
| Features | Frozen 120-field causal base contract, hash `b2c2725813d30c02ee298f82292d848d0e1133eb01be3f1398003163523ec2a1` |
| Training history | Six preceding months; 28-day resolved-label reserve; 240,000-row cap |
| Query weighting | Each exact decision timestamp has total loss weight one |
| Trees | 280 trees, depth 4, 31 leaves, learning rate 0.035 |
| Sampling | Feature fraction 0.78; subsample 0.78 |
| Regularisation | L1 0.05; L2 10.0; min child floor 300; min split gain 0.001 |
| Rank geometry | Truncation 12; gains `[0, .25, 1, 3, 7, 12]` |
| Router action | Deterministic timestamp-local top 30%, ties by candidate identity |

The full A--E auxiliary blend did not pass the router gate. The selected P3 receipt is primary-only; no auxiliary label group enters the deployed candidate contract.

## Downstream contract

The router does not become an alpha coordinate. On its retained candidates, the downstream system uses the actual strict-OOF enhanced-base values and only two correction heads:

| Layer | Contract |
|---|---|
| Base | Actual enhanced-base, B0, direct efficiency and direct timing scores preserved from source |
| Correction heads | `cap80_ordinary` and `cap120_equal_month` |
| Correction label | `direct_policy_economic_200_0_50_150` / ordinal LambdaRank |
| Consensus | Frozen 75/25 base-rank/consensus-rank control with generic correctness multiplier |
| MC1 | Independently prequential Current and BCF expected-EV maps, up to three prior scored months |
| Admission | Both MC1 maps >= +50 bps |
| Auction | BCF mapped expected EV; one global chronological portfolio |
| Outcome | Reconciled rich-policy net outcome, joined after target-free scores |

## Strict-OOF results

Evaluation is February--July 2026. All selected entries have outcome coverage of 100%.

| Default portfolio: 8 positions, 2 new entries/bar | Entries | Net EV/trade | Total net bps | Worst month | Worst week | Max DD |
|---|---:|---:|---:|---:|---:|---:|
| Matched enhanced-base route control | 3,857 | +150.37 | +579,960 | +126.07 | +82.99 | -20.27% |
| **P3 router-only** | **4,598** | **+155.99** | **+717,256** | **+126.66** | **+69.93** | **-36.30%** |
| P3 minus control | +741 | +5.63 | +137,296 | +0.60 | -13.06 | -16.03 pp |

P3 improves per-trade and total economics but creates more concurrent risk. The leading *portfolio-policy* challenger is a separate one-entry-per-hour cap, which changes no model or admission rule:

| Safer portfolio: 8 positions, 1 new entry/bar | Entries | Net EV/trade | Total net bps | Worst month | Worst week | Max DD | Sortino |
|---|---:|---:|---:|---:|---:|---:|---:|
| Matched enhanced-base route control | 2,929 | +168.62 | +493,878 | +130.86 | +101.62 | -22.09% | 0.955 |
| **P3 router-only** | **3,430** | **+173.30** | **+594,407** | **+137.57** | **+99.82** | **-22.60%** | **1.035** |
| P3 minus control | +501 | +4.68 | +100,530 | +6.70 | -1.80 | -0.50 pp | +0.080 |

P3's maximum mark-to-market drawdown in the safer run is -21.74%, versus -22.90% for the matched control; its growth-to-MTM-drawdown ratio is 179.79 versus 142.49, and ulcer index is 2.35% versus 2.42%.

## Monthly evidence: default 8x2 portfolio

| Month | Control entries / EV | P3 entries / EV | P3 total-net delta |
|---|---:|---:|---:|
| 2026-02 | 682 / +160.10 | 677 / +174.06 | +8,646 |
| 2026-03 | 872 / +143.01 | 893 / +156.59 | +15,130 |
| 2026-04 | 688 / +187.06 | 792 / +191.38 | +22,877 |
| 2026-05 | 718 / +126.07 | 867 / +126.66 | +19,303 |
| 2026-06 | 422 / +156.60 | 870 / +140.79 | +56,403 |
| 2026-07 | 475 / +127.95 | 499 / +151.73 | +14,938 |

The P3 incremental cohort is not a one-month result: it raises total policy net in every held month. June is the important trade-off: it expands participation considerably and raises total net, but dilutes EV/trade.

## Audit evidence

The hardened replay received these checks:

- `router_primary_rank` is absent from all nine target-free downstream monthly panels.
- No outcome, path-completeness, or target columns enter target-free panels.
- Exact `candidate_id`, timestamp and side identities align with source.
- All four original base coordinates have zero numerical delta in every month from November 2025 through July 2026.
- Current and BCF identities match exactly.
- Consensus heads train on three preceding calendar months plus a same-model, target-free 28-day reference reserve.
- MC1 fits use only prior scored rows whose policy labels were resolved before the held period.

## Selection boundary and next validation

P3 router-only and the 8x1 portfolio policy are **research challengers**, not live replacements. Before promotion, freeze their contracts and require a later untouched forward period with:

1. target-free identity and causal-label audits;
2. no deterioration in worst week, drawdown, concentration, or CVaR;
3. positive incremental P3 admissions after execution effects;
4. no dependence on June-like expansion or a small asset cluster; and
5. a separately authorised inference-parity and live-execution review.

## Reproduction receipts

| Purpose | Artifact |
|---|---|
| P3 router contract | `data_perp/artifacts/strict_r3_economic_recall_router_p3_h2_primaryonly_oof9_timestamp_20260825_v1/run_contract.json` |
| Hardened target-free downstream replay | `data_perp/artifacts/strict_r3_router_p3_h2_t6t9_direct_policy_hardened_20260825_v1/` |
| Hardened 8x1 safety result | `data_perp/artifacts/strict_r3_router_p3_h2_t6t9_direct_policy_hardened_safety_c8e1_20260825_v1/` |
| Matched B0 safety grid | `data_perp/artifacts/strict_r3_router_b0_portfolio_safety_20260825_v2/` |
| Router/downstream runner | `scripts/run_strict_r3_router_downstream.py` |
| Safety sweep | `scripts/sweep_strict_r3_router_only_portfolio_safety.py` |
