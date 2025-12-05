# MR vs Trend Classification Report: ETHUSDT
**Date:** 2025-12-04T19:39:14.684158
**Horizon:** 24 bars
**Threshold:** 2.00%
**Parameters:** SMA=20, Z=2.0, VWAP=96

## Metrics
**Trend (Class 1):** Precision: 0.32, Recall: 0.49, F1: 0.39
**MR (Class 2):** Precision: 0.71, Recall: 0.99, F1: 0.83
**Weighted Avg:** Precision: 0.81, Recall: 0.75, F1: 0.77

## Confusion Matrix
Noise (0): [19807  5097   764]
Trend (1): [1838 2387  664]
MR (2):    [   4   20 3435]

## Top 10 Features (Gain)
- **z_score_50:** 214.54
- **rsi_14:** 32.47
- **stoch_k:** 18.42
- **cmo_14:** 15.55
- **vol_ratio:** 13.06
- **dist_vwap_96:** 12.05
- **sma_50_slope:** 11.97
- **macd_hist:** 11.97
- **vol_med_48:** 11.29
- **atr_14_norm:** 11.13