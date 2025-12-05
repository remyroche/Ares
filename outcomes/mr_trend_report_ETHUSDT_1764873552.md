# MR vs Trend Classification Report: ETHUSDT
**Date:** 2025-12-04T19:39:12.144569
**Horizon:** 24 bars
**Threshold:** 1.50%
**Parameters:** SMA=20, Z=2.0, VWAP=96

## Metrics
**Trend (Class 1):** Precision: 0.36, Recall: 0.48, F1: 0.41
**MR (Class 2):** Precision: 0.59, Recall: 1.00, F1: 0.74
**Weighted Avg:** Precision: 0.75, Recall: 0.70, F1: 0.72

## Confusion Matrix
Noise (0): [17848  5677  1081]
Trend (1): [2442 3132  934]
MR (2):    [   0   13 2889]

## Top 10 Features (Gain)
- **z_score_50:** 201.32
- **rsi_14:** 28.47
- **stoch_k:** 15.26
- **cmo_14:** 14.34
- **vol_ratio:** 12.53
- **sma_50_slope:** 10.68
- **dist_vwap_96:** 10.00
- **macd_hist:** 9.56
- **vol_med_48:** 8.85
- **atr_14_norm:** 8.68