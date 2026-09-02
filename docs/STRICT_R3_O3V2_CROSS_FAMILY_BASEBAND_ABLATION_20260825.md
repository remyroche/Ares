# Strict-R3 O3-v2: cross-family and base-band ablation

## Scope

Offline long-only research only. No live bundle, admission rule, portfolio state, or execution process was modified.

The common forward evaluation is April--July 2026. Every score receipt is target-free when written. Each correction fold uses the preceding six calendar months with a 28-day resolved-label reserve; MC1 then uses the same prequential protocol. The matched comparison has 68,677 common candidate IDs and uses the reconciled rich-policy labels, dual current/BCF MC1 mapping, and one shared constrained portfolio.

## New cross-family feature-selection stage

Each F1--F6 family could contribute at most four additions beyond the frozen nine-field upstream core. The family winners were then pooled and considered one at a time in a second strict-OOF greedy pass. An addition was retained only when it improved the February--April 2026 development objective.

The final mixed contract retained only:

- `f5_delta_current_minus_bcf_final_score`
- `f3_q90_21d`
- `f5_current_base_anchor_bps`

It deliberately did not fill the cap. The sealed contract is `data_perp/artifacts/strict_r3_o3v2_g3_t6_final_cross_contract_20260825_v1.json`.

The role-specialist comparison selected H1, whose frozen two-head subset is the existing cap-100 ordinary rank plus the mixed cross-family head. It is not included in the MC1 result below: its earliest valid score month is February 2026, so it lacks the six complete preceding months required for a strict April--July MC1 fit. It remains a forward-only challenger.

## Query localization

The same retained T2/SB1 and T6/SB3 arms were compared on October--December 2025. Timestamp x base-score-band was selected. It improves the predeclared multi-tail stability score for both arms, while 4-hour x side is identical to timestamp x side in this implementation.

| Arm | Timestamp x side | Timestamp x base-band | 4-hour x side |
|---|---:|---:|---:|
| T2/SB1 stability score | 262.27 | **265.23** | 262.27 |
| T6/SB3 stability score | 274.00 | **277.42** | 274.00 |

## Matched MC1 and constrained-portfolio replay

All figures below are the April--July 2026 matched population at the 50-bps dual MC1 admission threshold. Delta is versus the current live-stack control on exactly the same eligible IDs.

| O3 inputs to MC1 | MC1 inputs | Trades | Net EV/trade | Delta EV/trade | Delta total net bps | Worst month | Worst week | Max drawdown |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Live control | -- | 1,901 | 154.50 | -- | -- | 130.52 | 94.57 | -27.81% |
| T2 only | Full arm + head ranks | 2,281 | 157.42 | +2.93 | +65,387 | 109.21 | 77.15 | -27.90% |
| T6 only | Full arm + head ranks | 2,298 | 157.68 | +3.19 | +68,659 | 112.70 | 87.80 | -14.34% |
| T2 + T6 | Head ranks only | 2,266 | 160.77 | +6.27 | +70,597 | 116.85 | 93.70 | -16.17% |
| T2 + T6 | Full summaries + head ranks | 2,283 | 161.83 | +7.33 | +75,757 | 112.44 | 86.57 | -17.30% |
| **T2 + T6** | **Aggregate summaries only** | **2,289** | **162.27** | **+7.77** | **+77,729** | **115.28** | **92.25** | **-16.29%** |

The aggregate-only input is the current research leader: it has the best EV/trade and total contribution, and individual head ranks do not add incremental value after the aggregate summaries are present. This is a research finding, not a promotion decision.

Monthly accepted-trade outcomes for the aggregate leader:

| Month | Trades | Net EV/trade (bps) | Total net bps |
|---|---:|---:|---:|
| 2026-04 | 620 | 235.11 | 145,765.54 |
| 2026-05 | 718 | 143.24 | 102,847.64 |
| 2026-06 | 604 | 115.28 | 69,629.22 |
| 2026-07 | 347 | 153.27 | 53,183.24 |

At 30 bps, the challenger increases participation and total net contribution but reduces EV/trade versus control. The 50-bps version is therefore the only threshold that advances to later falsification.

## Receipts

- Cross-family contract: `data_perp/artifacts/strict_r3_o3v2_g3_t6_final_cross_contract_20260825_v1.json`
- Base-band forward scores: `data_perp/artifacts/strict_r3_o3v2_support_forward_baseband_20260825_v1`
- Causality receipt (14/14 target-free, coverage and identity checks passed): `data_perp/artifacts/strict_r3_o3v2_support_chain_audit_baseband_20260825_v1/correctness_report.json`
- Aggregate MC1/portfolio result: `data_perp/artifacts/strict_r3_o3v2_mc1_portfolio_baseband_t2_t6_20260825_v1_aggregate`
