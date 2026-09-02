# Adaptive-exit sequential funnel — 2026-08-13

## Decision status

The full sequential experiment is complete. **No winner is promoted automatically.**
Selection uses 2025 chronological OOF economics; 2026 is confirmation.

## Immutable execution contract

- stop: `4.15200064332387` ATR;
- baseline activation: `2.326224919759605` ATR;
- giveback: `0.10237198997143725` ATR;
- H12 timeout;
- 100-bps cost exactly once;
- decisions use completed hourly information and take effect on the next source bar;
- admission, entries and sizes are unchanged until the separate capacity-aware replay.

## Core sequential stages

### Gate

| Trial | Portability | 2025 uplift | Worst month | Positive months | 2026 uplift | Intervene | Winner Δ | Loser Δ | CVaR05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `G5_action_directional` | +32.12 | +37.47 | +17.96 | 100% | +16.48 | +95.47% | -1.31 | +100.63 | -694.23 |
| `G4_directional_disagreement` | +32.09 | +37.46 | +17.96 | 100% | +16.48 | +95.45% | -1.27 | +100.55 | -694.23 |
| `G7_extreme20_placebo` | +32.07 | +37.09 | +18.05 | 100% | +16.31 | +84.68% | -1.98 | +100.74 | -694.23 |
| `G0_no_gate` | +32.02 | +37.12 | +17.96 | 100% | +16.15 | +95.66% | -1.93 | +100.74 | -694.23 |
| `G1_raw_disagreement` | +31.94 | +37.17 | +17.96 | 100% | +16.13 | +90.98% | -1.85 | +100.74 | -694.23 |
| `G2_action_disagreement` | +31.89 | +37.01 | +17.99 | 100% | +16.27 | +94.91% | -0.65 | +98.36 | -694.79 |
| `G6_random20_placebo` | +31.00 | +36.45 | +17.88 | 100% | +16.29 | +93.95% | -1.51 | +98.29 | -695.31 |
| `G3_normalized_disagreement` | +28.66 | +34.83 | +16.63 | 100% | +14.25 | +94.20% | -0.98 | +93.15 | -698.83 |
| `G9_high_ood20` | +27.23 | +30.45 | +16.06 | 100% | +13.32 | +78.59% | -2.31 | +83.81 | -702.69 |
| `G8_low_support20` | +26.74 | +32.98 | +14.21 | 100% | +12.81 | +77.67% | -2.51 | +90.80 | -700.59 |

### Direction

| Trial | Portability | 2025 uplift | Worst month | Positive months | 2026 uplift | Intervene | Winner Δ | Loser Δ | CVaR05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `A1_decreases_only` | +32.35 | +37.40 | +17.89 | 100% | +16.66 | +48.19% | -1.38 | +100.58 | -694.23 |
| `A3_both` | +32.12 | +37.47 | +17.96 | 100% | +16.48 | +95.47% | -1.31 | +100.63 | -694.23 |
| `A4_down_free_up_corroborated` | +32.02 | +37.12 | +17.96 | 100% | +16.15 | +95.31% | -1.93 | +100.74 | -694.23 |
| `A5_up_free_down_corroborated` | +31.94 | +37.20 | +18.04 | 100% | +16.17 | +95.46% | -0.62 | +98.82 | -694.79 |
| `A6_separate_direction_thresholds` | +31.85 | +37.36 | +18.26 | 100% | +15.84 | +94.68% | -0.77 | +99.48 | -694.30 |
| `A0_frozen` | +0.00 | +0.00 | +0.00 | 0% | +0.00 | +0.00% | +0.00 | +0.00 | -735.43 |
| `A2_increases_only` | -0.23 | +0.34 | -0.23 | 56% | +0.42 | +86.22% | +0.30 | +0.39 | -735.43 |

### Authority

| Trial | Portability | 2025 uplift | Worst month | Positive months | 2026 uplift | Intervene | Winner Δ | Loser Δ | CVaR05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `deadband_0.50` | +32.39 | +37.01 | +17.88 | 100% | +16.78 | +43.36% | -1.34 | +99.50 | -695.12 |
| `gamma_0.75` | +32.35 | +37.40 | +17.89 | 100% | +16.66 | +48.19% | -1.38 | +100.58 | -694.23 |
| `gamma_down75_up25` | +32.35 | +37.40 | +17.89 | 100% | +16.66 | +48.19% | -1.38 | +100.58 | -694.23 |
| `deadband_0.00` | +32.35 | +37.40 | +17.89 | 100% | +16.66 | +48.19% | -1.38 | +100.58 | -694.23 |
| `budget_50` | +32.35 | +37.40 | +17.89 | 100% | +16.66 | +48.19% | -1.38 | +100.58 | -694.23 |
| `budget_75` | +32.35 | +37.40 | +17.89 | 100% | +16.66 | +48.19% | -1.38 | +100.58 | -694.23 |
| `budget_100` | +32.35 | +37.40 | +17.89 | 100% | +16.66 | +48.19% | -1.38 | +100.58 | -694.23 |
| `deadband_0.10` | +32.24 | +37.38 | +17.89 | 100% | +16.64 | +46.81% | -1.39 | +100.55 | -694.23 |
| `deadband_0.20` | +32.24 | +37.39 | +17.82 | 100% | +16.61 | +45.86% | -1.36 | +100.52 | -694.23 |
| `deadband_0.30` | +32.20 | +37.33 | +17.82 | 100% | +16.63 | +45.08% | -1.30 | +100.25 | -694.73 |
| `gamma_1.0` | +31.79 | +36.44 | +17.49 | 100% | +16.25 | +47.80% | -3.20 | +101.01 | -694.10 |
| `gamma_down100_up50` | +31.79 | +36.44 | +17.49 | 100% | +16.25 | +47.80% | -3.20 | +101.01 | -694.10 |
| `budget_30` | +31.63 | +36.53 | +17.82 | 100% | +16.66 | +42.50% | -1.68 | +98.77 | -694.73 |
| `gamma_0.5` | +30.98 | +35.66 | +17.62 | 100% | +16.40 | +48.71% | -0.14 | +93.97 | -695.55 |
| `gamma_down50_up25` | +30.98 | +35.66 | +17.62 | 100% | +16.40 | +48.71% | -0.14 | +93.97 | -695.55 |
| `budget_20` | +29.49 | +33.86 | +17.77 | 100% | +16.43 | +36.22% | -1.51 | +91.49 | -696.04 |
| `budget_10` | +28.91 | +33.37 | +16.98 | 100% | +16.09 | +35.37% | -1.63 | +90.37 | -696.04 |
| `gamma_0.25` | +13.36 | +15.94 | +6.39 | 100% | +7.15 | +48.71% | +0.38 | +41.29 | -720.76 |

### Uncertainty

| Trial | Portability | 2025 uplift | Worst month | Positive months | 2026 uplift | Intervene | Winner Δ | Loser Δ | CVaR05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `quantile_q70` | +32.46 | +37.36 | +18.15 | 100% | +17.90 | +41.07% | -0.48 | +99.00 | -695.38 |
| `quantile_q65` | +32.39 | +37.01 | +17.88 | 100% | +16.78 | +43.36% | -1.34 | +99.50 | -695.12 |
| `uncertainty_f4_plus_quantiles` | +32.39 | +37.01 | +17.88 | 100% | +16.78 | +43.36% | -1.34 | +99.50 | -695.12 |
| `uncertainty_quantile_width` | +32.33 | +36.86 | +17.88 | 100% | +16.67 | +46.77% | -1.60 | +99.51 | -695.04 |
| `uncertainty_none` | +32.31 | +36.89 | +17.88 | 100% | +16.68 | +46.99% | -1.61 | +99.61 | -695.04 |
| `uncertainty_seed_ensemble` | +32.12 | +36.64 | +17.25 | 100% | +17.32 | +42.47% | -1.70 | +99.08 | -695.04 |
| `quantile_q75` | +31.87 | +37.23 | +19.06 | 100% | +17.81 | +37.85% | +0.97 | +96.29 | -696.04 |
| `quantile_q60` | +31.85 | +36.49 | +17.47 | 100% | +17.51 | +45.04% | -2.80 | +100.50 | -694.10 |
| `quantile_q50` | +30.75 | +35.58 | +16.83 | 100% | +16.47 | +47.77% | -5.43 | +102.38 | -692.16 |
| `uncertainty_f4` | +29.14 | +32.47 | +17.18 | 100% | +14.71 | +32.25% | -0.98 | +86.96 | -697.74 |

### Simplify

| Trial | Portability | 2025 uplift | Worst month | Positive months | 2026 uplift | Intervene | Winner Δ | Loser Δ | CVaR05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `proposal_f1_trust_positive` | +32.38 | +37.11 | +18.46 | 100% | +17.89 | +42.11% | -0.58 | +98.49 | -695.36 |
| `proposal_f1_support_ood` | +32.36 | +37.04 | +18.28 | 100% | +17.52 | +40.12% | -0.31 | +97.88 | -695.80 |
| `proposal_f1_compact_trust` | +32.30 | +36.94 | +18.08 | 100% | +17.74 | +39.41% | -0.39 | +97.74 | -695.95 |
| `proposal_f1_trust_ev` | +32.18 | +36.90 | +18.18 | 100% | +17.78 | +43.49% | -0.71 | +98.15 | -695.72 |
| `veto_f4_rich` | +27.81 | +32.09 | +16.21 | 100% | +14.69 | +30.02% | -0.72 | +85.54 | -697.83 |
| `veto_f4_evolution` | +27.64 | +31.41 | +16.04 | 100% | +13.60 | +27.49% | -0.81 | +83.91 | -699.04 |
| `veto_f4_full` | +27.43 | +31.70 | +16.83 | 100% | +14.23 | +28.37% | -0.91 | +84.83 | -698.42 |
| `veto_p5` | +27.23 | +30.58 | +16.47 | 100% | +13.71 | +26.61% | -0.54 | +81.28 | -699.81 |
| `veto_p3` | +27.22 | +30.79 | +16.32 | 100% | +13.93 | +26.41% | -0.50 | +81.77 | -699.62 |
| `veto_f4_archetype` | +27.16 | +31.72 | +16.15 | 100% | +15.02 | +29.66% | -1.12 | +85.21 | -697.96 |
| `veto_f4_compact` | +27.14 | +31.19 | +15.72 | 100% | +14.18 | +26.25% | -0.74 | +83.20 | -698.67 |
| `veto_p2` | +27.12 | +30.62 | +16.47 | 100% | +13.80 | +26.40% | -0.51 | +81.33 | -699.57 |
| `veto_p4` | +26.93 | +30.51 | +16.22 | 100% | +13.55 | +26.37% | -0.62 | +81.21 | -699.44 |
| `veto_p1` | +26.89 | +31.21 | +16.30 | 100% | +13.58 | +26.86% | -0.38 | +82.68 | -699.24 |
| `veto_f4_trust` | +26.69 | +30.72 | +16.48 | 100% | +14.65 | +25.34% | -1.05 | +82.48 | -699.80 |
| `veto_p0` | +26.09 | +30.35 | +15.72 | 100% | +14.47 | +25.32% | -0.49 | +80.58 | -699.97 |

### Actionable

| Trial | Portability | 2025 uplift | Worst month | Positive months | 2026 uplift | Intervene | Winner Δ | Loser Δ | CVaR05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `actionable_all` | +32.38 | +37.11 | +18.46 | 100% | +17.89 | +42.11% | -0.58 | +98.49 | -695.36 |
| `schedule_every_hour` | +32.38 | +37.11 | +18.46 | 100% | +17.89 | +42.11% | -0.58 | +98.49 | -695.36 |
| `schedule_every_hour_deadband20` | +32.05 | +37.62 | +18.16 | 100% | +17.31 | +46.88% | -0.50 | +99.71 | -694.60 |
| `actionable_only` | +31.59 | +36.81 | +18.37 | 100% | +17.30 | +42.03% | -0.81 | +98.10 | -695.36 |
| `schedule_hour4` | +3.29 | +5.29 | +1.83 | 100% | +3.77 | +17.74% | -1.16 | +15.79 | -726.71 |
| `schedule_first` | -0.38 | -0.04 | -0.29 | 33% | +0.24 | +4.86% | -0.19 | +0.20 | -735.43 |
| `schedule_hour2` | -0.48 | +0.58 | -0.84 | 67% | +0.48 | +9.84% | -0.42 | +2.21 | -734.53 |

### Target

| Trial | Portability | 2025 uplift | Worst month | Positive months | 2026 uplift | Intervene | Winner Δ | Loser Δ | CVaR05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `T1_carried_contract` | +32.38 | +37.11 | +18.46 | 100% | +17.89 | +42.11% | -0.58 | +98.49 | -695.36 |
| `cap12` | +32.38 | +37.11 | +18.46 | 100% | +17.89 | +42.11% | -0.58 | +98.49 | -695.36 |
| `cap_train_p99` | +32.38 | +37.11 | +18.46 | 100% | +17.89 | +42.11% | -0.58 | +98.49 | -695.36 |
| `cap8` | +32.15 | +37.15 | +18.47 | 100% | +17.80 | +44.91% | -0.23 | +98.04 | -695.36 |
| `cap10` | +32.12 | +37.00 | +18.77 | 100% | +18.03 | +43.68% | -0.34 | +97.83 | -695.57 |
| `T5_probability_reaches_a0` | +29.45 | +33.64 | +14.99 | 100% | +12.78 | +54.80% | -9.11 | +103.29 | -691.36 |
| `T2_increment_above_current_mfe` | +27.97 | +29.82 | +11.81 | 100% | +10.41 | +70.34% | -16.19 | +104.78 | -690.15 |
| `T4_future_peak_minus_current` | +17.39 | +20.67 | +9.87 | 100% | +8.59 | +57.55% | -6.76 | +65.35 | -719.59 |
| `T3_next_hour_from_current` | +7.95 | +13.60 | +2.07 | 100% | +3.53 | +95.66% | -41.82 | +103.89 | -688.24 |

## Direct one-dimensional value and learned mapping

| Trial | Portability | 2025 uplift | Worst month | Positive months | 2026 uplift | Intervene | Winner Δ | Loser Δ | CVaR05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `M0_linear_incumbent` | +32.38 | +37.11 | +18.46 | 100% | +17.89 | +42.11% | -0.58 | +98.49 | -695.36 |
| `T6_direct_lambdarank` | +30.09 | +33.92 | +16.61 | 100% | +15.01 | +95.66% | -6.95 | +100.50 | -692.37 |
| `M1_isotonic` | +8.88 | +10.72 | +1.08 | 100% | +3.06 | +95.62% | -23.26 | +66.06 | -711.96 |
| `T6_direct_huber` | +7.82 | +13.74 | +2.07 | 100% | +3.67 | +83.84% | -41.54 | +103.78 | -688.32 |
| `M2_piecewise_monotonic` | +7.76 | +10.39 | +1.73 | 100% | +5.42 | +95.66% | -29.51 | +75.40 | -707.12 |

## Fine-path × hourly-proxy source factorial

| Trial | Portability | 2025 uplift | Worst month | Positive months | 2026 uplift | Intervene | Winner Δ | Loser Δ | CVaR05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `S4_f1_proxy_f1` | +27.24 | +31.82 | +15.21 | 100% | +16.19 | +77.01% | -1.46 | +83.48 | -690.57 |
| `S5_winner_proxy_f1` | +27.24 | +31.82 | +15.21 | 100% | +16.19 | +77.01% | -1.46 | +83.48 | -690.57 |
| `S6_winner_proxy_uncertainty` | +27.24 | +31.78 | +15.21 | 100% | +16.20 | +74.72% | -1.42 | +83.33 | -690.57 |
| `S3_frozen_proxy_f1` | +15.45 | +17.23 | +8.25 | 100% | +9.39 | +60.46% | -1.23 | +45.90 | -711.19 |
| `S1_f1_frozen` | +12.31 | +14.58 | +6.96 | 100% | +6.80 | +16.55% | -0.23 | +37.58 | -717.95 |
| `S2_winner_frozen` | +12.31 | +14.58 | +6.96 | 100% | +6.80 | +16.55% | -0.23 | +37.58 | -717.95 |
| `S0_frozen_frozen` | +0.00 | +0.00 | +0.00 | 0% | +0.00 | +0.00% | +0.00 | +0.00 | -734.33 |

### Hourly-proxy uncertainty challengers

| Trial | Portability | 2025 uplift | Worst month | Positive months | 2026 uplift | Intervene | Winner Δ | Loser Δ | CVaR05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `proxy_quantile_width` | +24.77 | +28.33 | +13.24 | 100% | +15.16 | +95.83% | -1.99 | +73.98 | -686.07 |
| `proxy_f1` | +24.63 | +28.39 | +13.24 | 100% | +15.15 | +99.61% | -2.04 | +74.22 | -686.07 |
| `proxy_ensemble` | +24.44 | +28.15 | +13.86 | 100% | +15.68 | +93.44% | -2.07 | +73.65 | -686.20 |
| `proxy_deadband20` | +24.42 | +28.41 | +13.28 | 100% | +15.15 | +99.54% | -1.97 | +74.15 | -686.07 |
| `proxy_ood` | +18.76 | +23.30 | +11.96 | 100% | +12.43 | +82.12% | -2.20 | +61.69 | -694.23 |
| `proxy_support` | +16.99 | +24.83 | +10.81 | 100% | +12.35 | +75.50% | -1.68 | +64.75 | -694.04 |
| `proxy_frozen` | +0.00 | +0.00 | +0.00 | 0% | +0.00 | +0.00% | +0.00 | +0.00 | -732.08 |

### Dynamic-capacity portfolio

| Metric | Frozen | Funnel winner | Change |
|---|---:|---:|---:|
| Trades | 8,453 | 8,622 | +169 |
| Trades/day | 14.72 | 15.01 | +0.29 |
| Net EV/trade (bps) | +163.09 | +189.36 | +26.27 |
| Sortino | 0.465 | 0.548 | +0.083 |
| Max drawdown | -76.53% | -76.53% | +0.00 pp |
| Worst week | -35.09% | -19.28% | +15.81 pp |

## Portability funnel

### Window

| Trial | Portability | 2025 uplift | Worst month | Positive months | 2026 uplift | Intervene | Winner Δ | Loser Δ | CVaR05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `window_9m` | +32.38 | +37.11 | +18.46 | 100% | +17.89 | +42.11% | -0.58 | +98.49 | -695.36 |
| `window_12m` | +32.38 | +37.11 | +18.46 | 100% | +17.86 | +42.11% | -0.58 | +98.49 | -695.36 |
| `window_expanding` | +32.38 | +37.11 | +18.46 | 100% | +17.86 | +42.11% | -0.58 | +98.49 | -695.36 |
| `window_6m` | +32.37 | +37.17 | +18.13 | 100% | +17.05 | +43.02% | -0.62 | +98.72 | -694.67 |
| `window_3m` | +31.53 | +36.61 | +17.53 | 100% | +17.31 | +44.97% | -0.35 | +96.82 | -695.39 |

### Weight

| Trial | Portability | 2025 uplift | Worst month | Positive months | 2026 uplift | Intervene | Winner Δ | Loser Δ | CVaR05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `weight_uniform` | +32.38 | +37.11 | +18.46 | 100% | +17.89 | +42.11% | -0.58 | +98.49 | -695.36 |
| `weight_recency` | +32.31 | +37.03 | +18.28 | 100% | +17.68 | +44.53% | -0.41 | +98.02 | -695.48 |
| `weight_equal_month_recency` | +32.16 | +36.89 | +18.28 | 100% | +17.78 | +44.42% | -0.48 | +97.76 | -695.48 |
| `weight_equal_month` | +32.13 | +37.13 | +18.61 | 100% | +17.93 | +42.88% | -0.55 | +98.51 | -695.36 |

### Regularization

| Trial | Portability | 2025 uplift | Worst month | Positive months | 2026 uplift | Intervene | Winner Δ | Loser Δ | CVaR05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `reg_d4_l15_l2_25` | +32.50 | +37.08 | +18.52 | 100% | +17.83 | +42.39% | -0.25 | +97.90 | -695.47 |
| `reg_d4_l15_l2_10` | +32.38 | +37.11 | +18.46 | 100% | +17.89 | +42.11% | -0.58 | +98.49 | -695.36 |
| `reg_d4_l15_l2_50` | +32.23 | +37.00 | +18.27 | 100% | +17.91 | +41.80% | -0.44 | +97.98 | -695.38 |
| `reg_d3_l7_l2_10` | +32.12 | +36.86 | +17.78 | 100% | +17.63 | +36.54% | -0.33 | +97.45 | -695.11 |
| `reg_d3_l7_l2_25` | +31.94 | +36.71 | +17.78 | 100% | +17.67 | +36.39% | -0.46 | +97.26 | -695.23 |

### Missing

| Trial | Portability | 2025 uplift | Worst month | Positive months | 2026 uplift | Intervene | Winner Δ | Loser Δ | CVaR05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `missing_median` | +32.50 | +37.08 | +18.52 | 100% | +17.83 | +42.39% | -0.25 | +97.90 | -695.47 |
| `missing_native` | +32.50 | +37.08 | +18.52 | 100% | +17.83 | +42.39% | -0.25 | +97.90 | -695.47 |
| `missing_median_indicators` | +32.50 | +37.08 | +18.52 | 100% | +17.83 | +42.39% | -0.25 | +97.90 | -695.47 |

### Seed

| Trial | Portability | 2025 uplift | Worst month | Positive months | 2026 uplift | Intervene | Winner Δ | Loser Δ | CVaR05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `seed_20260813` | +32.50 | +37.08 | +18.52 | 100% | +17.83 | +42.39% | -0.25 | +97.90 | -695.47 |
| `seed_20260814` | +32.24 | +36.97 | +18.20 | 100% | +17.83 | +44.44% | -0.38 | +97.80 | -695.57 |
| `seed_20260811` | +32.18 | +37.12 | +18.56 | 100% | +17.71 | +43.51% | -0.35 | +98.15 | -695.31 |
| `seed_20260812` | +32.05 | +37.02 | +18.31 | 100% | +17.82 | +43.65% | -0.34 | +97.87 | -695.57 |

## Terminal-contract dynamic portfolio

This reruns the auction after applying the portability winner
(L2 25) to the fine branch and retaining the frozen proxy winner.

| Metric | Frozen | Terminal contract | Change |
|---|---:|---:|---:|
| Trades | 8,453 | 8,622 | +169 |
| Trades/day | 14.72 | 15.01 | +0.29 |
| Net EV/trade (bps) | +163.09 | +189.36 | +26.27 |
| Sortino | 0.465 | 0.548 | +0.083 |
| Max drawdown | -76.53% | -76.53% | +0.00 pp |
| Worst week | -35.09% | -19.28% | +15.81 pp |

## Requirement coverage

| Item | Requirement | Evidence |
|---:|---|---|
| 1 | Raw/action/normalized/directional disagreement gates | `core/gate` |
| 2 | Asymmetric disagreement | `core/direction` |
| 3 | Raise/lower activation economics | `core/direction metrics` |
| 4 | Symmetric and asymmetric gamma | `core/authority` |
| 5 | Activation deadbands | `core/authority` |
| 6 | q50/q60/q65/q70/q75 | `core/uncertainty` |
| 7 | Multi-quantile and seed uncertainty | `core/uncertainty` |
| 8 | Blocked random/extreme/support/OOD placebos | `core/gate` |
| 9 | F4 block decomposition | `core/simplify` |
| 10 | Prediction-evolution P0–P5 | `core/simplify` |
| 11 | Compact trust injected into proposal | `core/simplify` |
| 12 | Actionable-only states | `core/actionable` |
| 13 | First/2h/4h/every-hour schedules | `core/actionable` |
| 14 | Fine × proxy source factorial | `completion/source_factorial` |
| 15 | Proxy uncertainty gate | `completion/source_factorial` |
| 16 | Direct one-dimensional Δnet | `completion/direct_and_mapping` |
| 17 | Isotonic and piecewise learned mappings | `completion/direct_and_mapping` |
| 18 | T1–T6 target family | `core/target + direct` |
| 19 | 8/10/12/p99 target caps | `core/target` |
| 20 | 3/6/9/12/expanding windows and recency | `completion/portability` |
| 21 | Depth/leaves/L2 regularization | `completion/portability` |
| 22 | Native/median/indicator missingness | `completion/portability` |
| 23 | 10/20/30/50/75/100% intervention budgets | `core/authority` |
| 24 | Sequential, non-factorial winner carry | `manifests and fit audits` |

## Reproduction

```bash
python3 scripts/run_adaptive_exit_sequential_funnel.py --out-dir data_perp/artifacts/adaptive_exit_sequential_funnel_20260813_v3 --max-train-states 40000
python3 scripts/run_adaptive_exit_sequential_funnel_completion.py --core-dir data_perp/artifacts/adaptive_exit_sequential_funnel_20260813_v3 --out-dir data_perp/artifacts/adaptive_exit_sequential_funnel_completion_20260813_v3 --max-train-states 40000
python3 scripts/report_adaptive_exit_sequential_funnel.py
python3 scripts/audit_adaptive_exit_sequential_funnel.py
```

The fail-closed completion audit passes 30/30 checks. Its machine-readable receipt is:
`data_perp/artifacts/adaptive_exit_sequential_funnel_completion_20260813_v3/correctness_test_report.json`.
