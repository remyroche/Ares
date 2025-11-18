# Meta-Labeling Diagnostics Report

**Generated:** 2025-11-17 23:07:08

**Symbol:** ETHUSDT
**Timeframe:** 15m
**Horizon:** N/A bars

---


## 1. Label Distribution Analysis

- **Total labeled events:** 42945
- **Positive labels (profitable):** 12921 (30.1%)
- **Negative labels (unprofitable):** 30024 (69.9%)

✅ **OK:** Reasonable label balance (30.1%)

### Label Distribution Over Time

- **Daily positive rate - Mean:** 9.5%
- **Daily positive rate - Std:** 3.2%
- **Daily positive rate - Min:** 0.0%
- **Daily positive rate - Max:** 24.0%

⚠️ **Warning:** 768 days with <10% positive labels

## 2. Signal Coverage and Sparsity

- **Total samples:** 135775
- **Labeled samples:** 42945
- **Coverage:** 31.6%

✅ **OK:** Reasonable signal coverage (31.6%)

## 3. Feature-Label Correlation Analysis


### Top 10 Most Correlated Features:

- **event_win_rate_last_50:** 0.1907
- **event_mean_return_last_50:** 0.1212
- **returns_entropy:** 0.1081
- **atr_14:** 0.1035
- **atr_ratio:** 0.1026
- **volatility_ema:** 0.0986
- **volatility_1d:** 0.0979
- **volatility_20:** 0.0963
- **volatility_4h:** 0.0950
- **volatility_5:** 0.0778

### Correlation Health Check:


## 4. P&L Distribution Per Label

### Label = 1 (Profitable Signals):

- **Count:** 12921
- **Mean return:** 0.45%
- **Median return:** 0.26%
- **Std return:** 0.53%
- **% Actually positive:** 100.0%

### Label = 0 (Unprofitable Signals):

- **Count:** 30024
- **Mean return:** -0.40%
- **Median return:** -0.32%
- **Std return:** 0.34%
- **% Actually positive:** 0.0%

### Labeling Quality:

- **Label overlap:** 0.0%

✅ **OK:** Acceptable label overlap (0.0%)

## 5. Time-Series Stability and Regime Analysis

- **Win rate vs Volatility correlation:** 0.4780
- **Mean return vs Volatility correlation:** 0.1570

✅ **OK:** Win rate not strongly correlated with volatility

### Regime Shift Detection:

- **30-day rolling correlation (win rate vs vol) - Mean:** 0.4555
- **30-day rolling correlation (win rate vs vol) - Std:** 0.1539

## 6. Model Probability Diagnostics


### Calibration (Predicted Probability vs Actual Success Rate):


| Predicted Prob | Actual Success Rate | Count |
|---------------|---------------------|-------|
| 0.109 | 0.072 | 555 |
| 0.163 | 0.135 | 2907 |
| 0.219 | 0.200 | 5979 |
| 0.276 | 0.269 | 9809 |
| 0.332 | 0.332 | 8355 |
| 0.390 | 0.391 | 4935 |
| 0.448 | 0.445 | 2423 |
| 0.500 | 0.356 | 7791 |
| 0.563 | 0.511 | 176 |
| 0.623 | 0.667 | 15 |

- **Mean calibration error:** 0.033

✅ **OK:** Reasonable calibration

### Monotonicity and Slope Checks (Prob → Realized Return)

- **Monotonicity violations (adjacent bins):** 3 / 9
- **Approx. slope in high-probability region:** 0.020693

### Probability Distribution:

- **Mean probability:** 0.447
- **Median probability:** 0.500
- **Std probability:** 0.099

## 7. Feature Importance Analysis


### Top 20 Features by Importance:

1. **dist_from_recent_high_50:** 0.1107
2. **dist_from_recent_low_50:** 0.0727
3. **sma_slope:** 0.0382
4. **volatility_ema:** 0.0311
5. **momentum_20:** 0.0306
6. **volatility_5:** 0.0272
7. **volatility_1d:** 0.0269
8. **momentum_per_vol:** 0.0247
9. **volatility_4h:** 0.0243
10. **vol_of_vol:** 0.0202
11. **momentum_raw:** 0.0200
12. **vol_price_corr:** 0.0200
13. **volatility_1h:** 0.0195
14. **signal_density_50:** 0.0186
15. **volume_trend:** 0.0179
16. **log_ret:** 0.0174
17. **returns_entropy:** 0.0174
18. **atr_ratio:** 0.0173
19. **vol_ratio:** 0.0169
20. **price_vs_sma20:** 0.0169

### Feature Importance Health Check:


✅ **OK:** No single feature dominates

✅ Kalman-filtered features are being used (6 features)

✅ Volatility-based features in top 10: volatility_ema, volatility_5, volatility_1d, momentum_per_vol, volatility_4h, vol_of_vol

## 8. Kalman Smoothed Labels Analysis

- **Mean smoothed label:** 0.301
- **Median smoothed label:** 0.291
- **Std smoothed label:** 0.157
- **Correlation with binary labels:** 0.580
- **Correlation with realized returns:** 0.371

✅ **OK:** Good correlation between smoothed and binary labels

## 9. Event Mechanics and R-Multiples

These diagnostics describe how trades exit (profit, stop, or timeout), how long they stay open, and how large the realized return is relative to the configured stop-loss (R-multiple). They help verify that the triple-barrier / TPSL configuration produces economically meaningful events.


### Exit Reason Mix (Labeled Events)

- **timeout:** 75.6%
- **stop:** 19.9%
- **profit:** 4.5%

### Event Duration Distribution (Bars)

- **Mean duration:** 2.76
- **Median duration:** 3.00
- **90th percentile:** 3.00

### R-Multiple Distribution (Return / Stop Threshold)

- **R-multiple (label=1) mean:** 1.09
- **R-multiple (label=1) median:** 0.64
- **R-multiple (label=0) mean:** -0.97
- **R-multiple (label=0) median:** -0.77

## 10. Regime and Time-Conditional Label Checks

These metrics show how label base-rates and realized returns change across volatility regimes, trend states, and time-of-day/weekday. Large differences indicate that learnability and edge are regime-dependent, which is important for downstream model conditioning.


### Label Base-Rate by Volatility Regime

- **Regime high:** positive=35.7%, mean_return=-0.15%
- **Regime low:** positive=23.0%, mean_return=-0.14%
- **Regime medium:** positive=30.3%, mean_return=-0.15%

### Trend-Conditional (Price vs SMA20)

- **Strong uptrend:** positive=33.0%, mean_return=-0.14%
- **Strong downtrend:** positive=32.0%, mean_return=-0.14%

### Time-of-Day Positive Rates (Top/Bottom)

- **Top hours by positive rate:**
  - Hour 14: 36.1%
  - Hour 16: 33.7%
  - Hour 15: 33.3%
- **Bottom hours by positive rate:**
  - Hour 3: 25.8%
  - Hour 20: 27.6%
  - Hour 23: 27.8%

### Day-of-Week Positive Rates

- Day 0: 32.9%
- Day 1: 31.8%
- Day 2: 31.0%
- Day 3: 32.2%
- Day 4: 31.2%
- Day 5: 24.7%
- Day 6: 27.0%

## 11. Label–Return Separation and Information Content

This section quantifies how well the labels separate profitable from unprofitable events (effect size) and how much information they carry about the sign of future returns compared to a random baseline (mutual information and permutation tests).


### Separation Metrics

- **Mean return difference (label=1 - label=0):** 0.85%
- **Cohen's d effect size:** 2.075

### Information Content vs Permutation Baseline

- **Mutual information (labels vs realized sign):** 0.6116
- **Baseline MI (mean over permutations):** 0.0000

## 12. Target and Expected-Return Alignment

Here we check whether the continuous targets produced by the isotonic mapping are consistent with realized returns: higher targets should correspond to higher average realized P&L, and most non-zero targets should exceed transaction costs.


### Target vs Realized Return

- **Correlation (target, realized):** 0.072
- **MSE (target vs realized):** 0.000054

### Target/Return by Target Decile

- Decile 0: target=0.0003, realized=-0.0011

### Target Distribution Sanity

- **Non-zero target fraction (all samples):** 6.4%
- **Mean non-zero target:** 0.0003
- **Median non-zero target:** 0.0003
- **Fraction of targets below transaction cost (0.150%):** 98.5%

## 13. Cost-Aware Metrics and Threshold P&L

These diagnostics evaluate the meta-model from a trading perspective: global AUC/Brier/PR-AUC, and how mean and cumulative returns evolve as you raise the probability threshold used to filter trades. This helps choose probability cutoffs that are profitable after costs.


### Cost-Aware Classification Metrics (OOF)

- **AUC:** 0.607
- **Brier score:** 0.2067
- **Average precision (PR-AUC):** 0.365

### Threshold-Sweep P&L (Using Meta Probability)


| Threshold | Trades | Mean Return | Cum Return | Sharpe (per trade) |
|----------|--------|-------------|------------|---------------------|
| 0.50 | 7608 | -0.13% | -956.50% | -15.418 |
| 0.51 | 445 | 0.10% | 46.72% | 2.475 |
| 0.52 | 344 | 0.12% | 40.04% | 2.388 |
| 0.53 | 268 | 0.11% | 29.24% | 2.010 |
| 0.54 | 201 | 0.14% | 27.22% | 2.073 |
| 0.55 | 147 | 0.16% | 23.86% | 1.965 |
| 0.56 | 101 | 0.16% | 16.12% | 1.633 |
| 0.57 | 68 | 0.12% | 8.25% | 1.009 |
| 0.58 | 42 | 0.14% | 5.72% | 0.952 |
| 0.59 | 27 | 0.11% | 3.05% | 0.677 |
| 0.60 | 15 | 0.30% | 4.51% | 1.220 |
| 0.61 | 12 | 0.41% | 4.87% | 1.520 |
| 0.62 | 7 | 0.18% | 1.25% | 0.915 |
| 0.63 | 6 | 0.20% | 1.19% | 0.857 |
| 0.64 | 2 | 0.23% | 0.45% | 3.291 |
| 0.65 | 1 | 0.29% | 0.29% | nan |

## 14. Stability and Bootstrap Statistics

Finally, we assess how stable the labeling and model performance are across time (years and time-series folds) and how sensitive key metrics such as positive rate and label–return separation are under bootstrap resampling. Tight confidence intervals indicate robust signal rather than fragile noise.


### Per-Year Label and Return Stability

- **Year 2021:** positive=33.4%, mean_return=-0.15%, AUC=0.500
- **Year 2022:** positive=32.2%, mean_return=-0.15%, AUC=0.555
- **Year 2023:** positive=23.7%, mean_return=-0.15%, AUC=0.642
- **Year 2024:** positive=30.9%, mean_return=-0.15%, AUC=0.616
- **Year 2025:** positive=34.4%, mean_return=-0.14%, AUC=0.604

### Time-Series Fold Stability (Approximate)

- **Fold 1:** positive=28.8%, mean_return=-0.15%, AUC=0.618
- **Fold 2:** positive=21.9%, mean_return=-0.15%, AUC=0.649
- **Fold 3:** positive=30.1%, mean_return=-0.15%, AUC=0.616
- **Fold 4:** positive=31.7%, mean_return=-0.14%, AUC=0.614
- **Fold 5:** positive=34.2%, mean_return=-0.14%, AUC=0.602

### Bootstrap Label Statistics (Labeled Events)

- **Positive rate 95% CI:** [29.7%, 30.5%]
- **Mean return diff (label=1 - label=0) 95% CI:** [0.84%, 0.86%]

---


## Summary and Recommendations


### Key Findings:

1. Label balance: 30.1% positive
2. Signal coverage: 31.6%
3. Mean return (label=1): 0.45%
4. Mean return (label=0): -0.40%
5. Calibration error: 0.033