# MR vs Trend Classification Report: ETHUSDT
**Date:** 2025-12-04T19:39:17.024956
**Horizon:** 24 bars
**Threshold:** 2.50%
**Parameters:** SMA=20, Z=2.0, VWAP=96

## Metrics
**Trend (Class 1):** Precision: 0.30, Recall: 0.53, F1: 0.38
**MR (Class 2):** Precision: 0.78, Recall: 1.00, F1: 0.87
**Weighted Avg:** Precision: 0.86, Recall: 0.80, F1: 0.82

## Confusion Matrix
Noise (0): [21384  4533   607]
Trend (1): [1283 1942  458]
MR (2):    [   2   17 3790]

## Top 10 Features (Gain)
- **z_score_50:** 227.91
- **rsi_14:** 35.57
- **stoch_k:** 18.02
- **cmo_14:** 16.60
- **vol_ratio:** 13.81
- **sma_50_slope:** 13.61
- **vol_med_48:** 13.50
- **dist_vwap_96:** 13.48
- **macd_hist:** 12.35
- **atr_14_norm:** 12.20