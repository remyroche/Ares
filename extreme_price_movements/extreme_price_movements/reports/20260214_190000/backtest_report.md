# Backtest Report — 20260214_190000
Generated: 2026-02-23 00:17 UTC

## Summary
- **Period**: 2026-02-03 to 2026-02-13 (10 days)
- **Total trades**: 170 (16.3/day)
- **Net PnL**: -0.049240
- **Sortino**: -0.3395
- **Max Drawdown**: -0.0565
- **Win Rate**: 38.24%
- **Profit Factor**: 0.355
- **Payoff Ratio**: 0.57
- **Avg Win**: +0.000416 | **Avg Loss**: -0.000727
- **Fee rate**: 35 bps
- **15m precision**: True

## Signal Parameters
- **k_long**: 10
- **k_short**: 10
- **size_k**: 4.0
- **size_max**: 0.15
- **size_min**: 0.03
- **size_q50**: 0.11931112367084162
- **size_q90**: 0.9993481421266157
- **size_q95**: 0.9998120990657471
- **size_q98**: 0.9999996727881719
- **size_x0**: 0.5
- **size_zcap**: 4.0
- **thr_long**: 0.0
- **thr_short**: -0.0

## Per-Bucket Performance
| Bucket | N | N/day | PnL | WR | Sortino | MaxDD | PF | AvgWin | AvgLoss | Payoff |
|--------|---|-------|-----|----|---------|----- -|----|--------|---------|--------|
| LONG_MR | 54 | 5.2 | -0.0057 | 48.15% | -0.288 | -0.0071 | 0.58 | +0.000296 | -0.000478 | 0.62 |
| LONG_TF | 41 | 3.9 | +0.0083 | 56.10% | 0.872 | -0.0033 | 1.91 | +0.000759 | -0.000507 | 1.50 |
| SHORT_MR | 35 | 3.4 | -0.0364 | 17.14% | -0.745 | -0.0357 | 0.02 | +0.000127 | -0.001280 | 0.10 |
| SHORT_TF | 40 | 3.8 | -0.0155 | 25.00% | -1.240 | -0.0154 | 0.07 | +0.000115 | -0.000556 | 0.21 |

## Exit Reasons
| Reason | N | % | PnL | WR | Avg Hold (h) |
|--------|---|---|-----|----|--------------|
| early_invalidation | 18 | 10.59% | -0.0077 | 16.67% | 0.5 |
| giveback_exit | 15 | 8.82% | +0.0043 | 73.33% | 0.0 |
| limit_not_filled | 28 | 16.47% | -0.0130 | 0.00% | 0.0 |
| stop_loss | 30 | 17.65% | -0.0064 | 40.00% | 0.1 |
| trailing_stop | 79 | 46.47% | -0.0264 | 49.37% | 0.2 |

## MAE / MFE Analysis
- **Global MAE**: mean=0.73%, med=0.47%, q90=1.74%
- **Global MFE**: mean=0.83%, med=0.55%, q90=1.80%
- **MFE/MAE ratio**: 1.13

### Per-Bucket MAE/MFE
| Bucket | MAE mean | MFE mean | MFE/MAE | Losers w/ MFE>0.5% | Winner Capture |
|--------|----------|----------|---------|---------------------|----------------|
| LONG_MR | 0.83% | 0.80% | 0.97 | 3.57% (1/28) | 0.74 |
| LONG_TF | 0.51% | 0.93% | 1.80 | 5.56% (1/18) | 0.84 |
| SHORT_MR | 0.66% | 0.83% | 1.27 | 72.41% (21/29) | -2.58 |
| SHORT_TF | 0.90% | 0.77% | 0.85 | 70.00% (21/30) | -1.09 |

## PnL Reconciliation
- **Gross PnL (pre-fee)**: +0.057435
- **Total fees**: +0.076252
- **Net PnL (post-fee)**: -0.018816
- **Gross Profit**: +0.027064
- **Gross Loss**: -0.076304

### Per-Bucket Contribution
| Bucket | N | Gross Profit | Gross Loss | Net PnL | PF | WR |
|--------|---|-------------|------------|---------|----|----|
| LONG_MR | 54 | +0.007693 | -0.013370 | -0.005677 | 0.58 | 48.15% |
| LONG_TF | 41 | +0.017453 | -0.009129 | +0.008324 | 1.91 | 56.10% |
| SHORT_MR | 35 | +0.000763 | -0.037118 | -0.036354 | 0.02 | 17.14% |
| SHORT_TF | 40 | +0.001154 | -0.016687 | -0.015533 | 0.07 | 25.00% |

## Daily Concentration
- **Max trades/day**: 31
- **Mean trades/day**: 17.0

### Per-Bucket Daily Max
| Bucket | Max/day | Mean/day |
|--------|---------|----------|
| LONG_MR | 14 | 6.0 |
| LONG_TF | 12 | 5.1 |
| SHORT_MR | 8 | 7.0 |
| SHORT_TF | 13 | 5.0 |

## Exit Stage Analysis
Exit stages: 0=initial SL, 1=break-even, 2=tight trail, 3=full trail

### Global
| Stage | N | % | PnL | WR | Avg Hold (h) |
|-------|---|---|-----|----|--------------|
| Stage 0 | 76 | 44.71% | -0.0271 | 19.74% | 0.2 |
| Stage 1 | 71 | 41.76% | -0.0217 | 50.70% | 0.2 |
| Stage 2 | 10 | 5.88% | -0.0003 | 60.00% | 0.1 |
| Stage 3 | 13 | 7.65% | -0.0001 | 61.54% | 0.0 |

### Per-Bucket Exit Stages
| Bucket | Stage | N | % | PnL | WR |
|--------|-------|---|---|-----|----|
| LONG_MR | 0 | 27 | 50.00% | -0.0128 | 0.00% |
| LONG_MR | 1 | 19 | 35.19% | +0.0041 | 100.00% |
| LONG_MR | 2 | 4 | 7.41% | +0.0000 | 75.00% |
| LONG_MR | 3 | 4 | 7.41% | +0.0030 | 100.00% |
| LONG_TF | 0 | 17 | 41.46% | -0.0089 | 0.00% |
| LONG_TF | 1 | 17 | 41.46% | +0.0065 | 100.00% |
| LONG_TF | 2 | 3 | 7.32% | +0.0006 | 66.67% |
| LONG_TF | 3 | 4 | 9.76% | +0.0100 | 100.00% |
| SHORT_MR | 0 | 14 | 40.00% | -0.0042 | 42.86% |
| SHORT_MR | 1 | 20 | 57.14% | -0.0240 | 0.00% |
| SHORT_MR | 3 | 1 | 2.86% | -0.0082 | 0.00% |
| SHORT_TF | 0 | 18 | 45.00% | -0.0013 | 50.00% |
| SHORT_TF | 1 | 15 | 37.50% | -0.0083 | 0.00% |
| SHORT_TF | 2 | 3 | 7.50% | -0.0010 | 33.33% |
| SHORT_TF | 3 | 4 | 10.00% | -0.0049 | 0.00% |

## Per-Regime Metrics

### Regime: G_VOL
| Value | N | PnL | WR | PF |
|-------|---|-----|----|----| 
| 0 | 47 | -0.0016 | 44.68% | 0.90 |
| 1 | 123 | -0.0476 | 35.77% | 0.21 |

### Regime: G_TREND
| Value | N | PnL | WR | PF |
|-------|---|-----|----|----| 
| 1 | 170 | -0.0492 | 38.24% | 0.35 |

## Weekly PnL
| Week | N | PnL | WR |
|------|---|-----|----|
| W06 | 122 | -0.0470 | 35.25% |
| W07 | 48 | -0.0022 | 45.83% |
