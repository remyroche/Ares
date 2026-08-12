# Frozen residual trust-overlay ablation

Date: 2026-08-10  
Input: frozen ATR2 specialists → q4h×side residual LambdaRank  
Rows: 388,494 strict-OOS candidates, July–November 2024

## Objective

The hierarchical conversion map failed because the score-to-net relationship
changes across sides and months. This test asks whether a conservative overlay
can suppress unreliable residual corrections without refitting the frozen
models.

For each decision day and side, only the preceding 21 calendar days are used,
with `label_available_ts = decision_ts + 13h` strictly before the current day.
The residual correction is:

`residual = frozen_score - prequential_base_expected_net_bps`.

The tested overlays shrink that residual toward zero using:

- `support`: effective soft-regime support;
- `ood`: Jensen–Shannon distance from the prior 21-day side regime mix;
- `ic`: recent side-local score/net rank IC;
- `combined`: support × OOD trust × IC trust.

Constants were fixed in advance: 2,000 support rows, JS scale 4, and IC scale
0.05. No realised test outcome is used in the current day's trust fields.

Implementation: `scripts/run_frozen_residual_trust_overlay_ablation.py`  
Artifacts: `data_perp/artifacts/frozen_residual_trust_overlay_20260810_v1/`

## Pooled global tails

| Arm | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% |
|---|---:|---:|---:|---:|---:|
| Raw control | −23.86 | −7.30 | **+16.01** | **+8.89** | −37.63 |
| Support trust | −26.28 | −7.91 | +16.10 | +9.10 | −40.32 |
| OOD trust | −26.47 | −9.59 | +15.50 | +7.87 | −41.95 |
| IC trust | −16.87 | −9.43 | +15.47 | +7.32 | −51.21 |
| Combined trust | −19.94 | −11.05 | +13.81 | +6.86 | −49.14 |

The apparent +0.22 bps support improvement at top 5% is not sufficient for
promotion: it does not improve the worst month and does not survive the side
and month gates.

## Monthly global top-5 net

| Arm | Jul | Aug | Sep | Oct | Nov | Mean | Worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw control | −51.55 | −171.07 | −58.25 | −81.07 | **+11.00** | −70.19 | −171.07 |
| Support trust | −56.38 | −177.04 | −59.54 | −81.74 | +8.29 | −73.28 | −177.04 |
| OOD trust | −54.24 | −188.44 | −59.25 | −80.88 | +6.64 | −75.23 | −188.44 |
| IC trust | −79.96 | −234.77 | −60.34 | −81.12 | +3.58 | −90.52 | −234.77 |
| Combined trust | −73.48 | −225.52 | −61.23 | −80.79 | +0.83 | −87.04 | −225.52 |

Every trust overlay worsens the mean monthly result and worst month.

## Per-side top-5 net

| Arm | Long | Short |
|---|---:|---:|
| Raw control | **+17.25** | −195.71 |
| Support trust | +17.14 | −195.83 |
| OOD trust | +15.90 | −196.41 |
| IC trust | +16.83 | −201.94 |
| Combined trust | +15.70 | −198.97 |

The trust fields do not identify a safe subset of the short side. They mostly
change which long rows enter the pooled tail while leaving the short-side
failure intact.

## Decision

`NO_TRUST_OVERLAY_ADVANCE`.

Support, OOD, and recent rank-IC information are causal and available, but a
post-fit residual shrinkage overlay does not repair the conversion failure.
Keep the raw frozen residual score as the control. The remaining gap requires a
newly trained conversion/reliability target that learns side-specific
under-conversion risk with enough historical support, rather than another
post-fit trust formula.
