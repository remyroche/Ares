# Meta-Labeling Diagnostics Report

**Generated:** 2025-11-18 23:50:32

**Symbol:** ETHUSDT
**Timeframe:** 15m
**Horizon:** N/A bars

---


## 1. Label Distribution Analysis

- **Total labeled events:** 52696
- **Positive labels (profitable):** 12481 (23.7%)
- **Negative labels (unprofitable):** 40215 (76.3%)

⚠️ **Warning:** Low positive label rate (23.7%) - most signals are unprofitable

### Label Distribution Over Time

- **Daily positive rate - Mean:** 9.2%
- **Daily positive rate - Std:** 6.3%
- **Daily positive rate - Min:** 0.0%
- **Daily positive rate - Max:** 36.0%

⚠️ **Warning:** 813 days with <10% positive labels

## 2. Signal Coverage and Sparsity

- **Total samples:** 135775
- **Labeled samples:** 52696
- **Coverage:** 38.8%

✅ **OK:** Reasonable signal coverage (38.8%)

## 3. Feature-Label Correlation Analysis (Post-Filter)


### Top 10 Most Correlated Features (Post-Filter):

- **event_win_rate_last_50:** 0.2191
- **event_mean_return_last_50:** 0.2176
- **atr_ratio:** 0.1163
- **returns_entropy:** 0.1126
- **volatility_ema:** 0.1051
- **volatility_20:** 0.1012
- **returns_std_20:** 0.1012
- **volatility_4h:** 0.1007
- **atr_14:** 0.0997
- **range_4h:** 0.0996

### Correlation Health Check (Post-Filter):


## 3B. Pre- vs Post-Filter Comparison and Feature Shift


### Sample Counts and Retention

- **Pre-filter events (realized_return not NaN):** 66522
- **Pre-filter positive/negative (raw economic):** 21569 / 44953
- **Post-filter labeled events:** 52696
- **Post-filter positive/negative (binary_labels):** 12481 / 40215
- **Total retention (post / pre):** 79.2%
- **Positive retention:** 57.9%
- **Negative retention:** 89.5%

### Pre- vs Post-Filter Signal Quality

- **Pre-filter mean return (label=1/0):** 1.01% / -0.66%
- **Post-filter mean return (label=1/0):** 1.12% / -0.73%
- **Pre-filter Cohen's d (label=1 vs 0):** 5.761
- **Post-filter Cohen's d (label=1 vs 0):** 10.526
- **Pre-filter SNR (mean/std, label=1):** 2.982
- **Post-filter SNR (mean/std, label=1):** 6.103

### Largest Feature Correlation Shifts (|post|-|pre|)

- **signal_disagreement:** pre=-0.2533, post=-0.0831, Δ|corr|=-0.1703
- **event_win_rate_last_50:** pre=-0.0991, post=0.2191, Δ|corr|=0.1200
- **event_mean_return_last_50:** pre=0.3331, post=0.2176, Δ|corr|=-0.1155
- **atr_ratio:** pre=0.0451, post=0.1163, Δ|corr|=0.0711
- **returns_entropy:** pre=0.0479, post=0.1126, Δ|corr|=0.0647
- **range_1h:** pre=0.0356, post=0.0989, Δ|corr|=0.0633
- **volatility_ema:** pre=0.0457, post=0.1051, Δ|corr|=0.0594
- **returns_std_10:** pre=0.0389, post=0.0977, Δ|corr|=0.0588
- **returns_std_5:** pre=0.0342, post=0.0930, Δ|corr|=0.0587
- **volatility_5:** pre=0.0342, post=0.0930, Δ|corr|=0.0587

## 4. P&L Distribution Per Label

### Label = 1 (Profitable Signals):

- **Count:** 12481
- **Mean return:** 1.12%
- **Median return:** 1.03%
- **Std return:** 0.18%
- **% Actually positive:** 100.0%

### Label = 0 (Unprofitable Signals):

- **Count:** 40215
- **Mean return:** -0.73%
- **Median return:** -0.72%
- **Std return:** 0.17%
- **% Actually positive:** 0.0%

### Labeling Quality:

- **Label overlap:** 0.0%

✅ **OK:** Acceptable label overlap (0.0%)

### Cost-Aware Event Quality Summary

- **Transaction cost (per event, approx):** 0.150%
- **Unconditional mean event return:** -0.29%
- **Mean return (label=1) minus cost:** 0.97%
- **Fraction of labeled events with |return| < cost:** 0.0%

## 5. Time-Series Stability and Regime Analysis

- **Win rate vs Volatility correlation:** 0.4533
- **Mean return vs Volatility correlation:** 0.1354

✅ **OK:** Win rate not strongly correlated with volatility

### Rolling Daily SNR (Return / Std by Day)

- **Daily SNR mean:** -0.2444
- **Daily SNR std:** 0.4793
- **Daily SNR min/max:** -3.4667 / 1.3608

### Regime Shift Detection:

- **30-day rolling correlation (win rate vs vol) - Mean:** 0.3795
- **30-day rolling correlation (win rate vs vol) - Std:** 0.1547

## 6. Model Probability Diagnostics


### Calibration (Predicted Probability vs Actual Success Rate):


| Predicted Prob | Actual Success Rate | Count |
|---------------|---------------------|-------|
| 0.167 | 0.012 | 1132 |
| 0.222 | 0.051 | 2668 |
| 0.289 | 0.108 | 4052 |
| 0.356 | 0.155 | 5629 |
| 0.423 | 0.210 | 7158 |
| 0.495 | 0.264 | 17569 |
| 0.554 | 0.304 | 7610 |
| 0.618 | 0.360 | 5287 |
| 0.678 | 0.411 | 1470 |
| 0.742 | 0.438 | 121 |

- **Mean calibration error:** 0.223

⚠️ **Warning:** High calibration error - model probabilities may not be well-calibrated

### Monotonicity and Slope Checks (Prob → Realized Return)

- **Monotonicity violations (adjacent bins):** 0 / 9
- **Approx. slope in high-probability region:** 0.015146

### Probability Distribution:

- **Mean probability:** 0.485
- **Median probability:** 0.500
- **Std probability:** 0.077

## 7. Feature Importance Analysis


### Top 20 Features by Importance:

1. **high_dist_x_vol:** 0.1305
2. **vol_range_interaction:** 0.1177
3. **event_mean_return_last_50:** 0.0295
4. **bars_since_last_event:** 0.0251
5. **sma_slope:** 0.0219
6. **vol_ratio:** 0.0181
7. **returns_std_50:** 0.0150
8. **volatility_1d:** 0.0134
9. **dist_from_recent_high_5:** 0.0124
10. **volatility_ema:** 0.0123
11. **volatility_4h:** 0.0123
12. **momentum_20:** 0.0123
13. **dist_from_recent_high_10:** 0.0121
14. **vol_price_corr:** 0.0121
15. **close_max_50:** 0.0119
16. **ma_distance_raw:** 0.0110
17. **dist_from_recent_high_20:** 0.0110
18. **returns_1h:** 0.0110
19. **volume_trend:** 0.0106
20. **returns_entropy:** 0.0104

### Feature Importance Health Check:


✅ **OK:** No single feature dominates

✅ Kalman-filtered features are being used (6 features)

✅ Volatility-based features in top 10: high_dist_x_vol, vol_range_interaction, vol_ratio, volatility_1d, volatility_ema

## 8. Kalman Smoothed Labels Analysis

- **Mean smoothed label:** 0.236
- **Median smoothed label:** 0.235
- **Std smoothed label:** 0.106
- **Correlation with binary labels:** 0.296
- **Correlation with realized returns:** 0.265

⚠️ **Warning:** Low correlation between smoothed and binary labels - Kalman filter may be over-smoothing

## 9. Event Mechanics and R-Multiples

These diagnostics describe how trades exit (profit, stop, or timeout), how long they stay open, and how large the realized return is relative to the configured stop-loss (R-multiple). They help verify that the triple-barrier / TPSL configuration produces economically meaningful events.


### Exit Reason Mix (Labeled Events)

- **stop:** 72.2%
- **profit:** 23.7%
- **timeout:** 4.1%

### Event Duration Distribution (Bars)

- **Mean duration:** 8.66
- **Median duration:** 6.00
- **90th percentile:** 21.00

### R-Multiple Distribution (Return / Stop Threshold)

- **R-multiple (label=1) mean:** 1.94
- **R-multiple (label=1) median:** 1.82
- **R-multiple (label=0) mean:** -1.24
- **R-multiple (label=0) median:** -1.26

## 9B. Enhanced Event Quality Metrics

These advanced metrics help distinguish between efficient momentum capture and random drift trades. They measure path efficiency, timing quality, and entry/exit effectiveness.


### Path Efficiency Ratio (PER)

- **Mean PER:** 138.811
- **Median PER:** 1.774

✅ **OK:** Reasonable path efficiency

### Time-to-Outcome Ratio (TTO)

- **Mean TTO:** 0.541
- **Median TTO:** 0.375

✅ **OK:** TTO in healthy range [0.4, 0.6]

### MFE/MAE Ratio (Maximum Favorable vs Adverse Excursion)

- **Mean MFE/MAE:** 165.762
- **Median MFE/MAE:** 0.528

✅ **OK:** Favorable excursions exceed adverse excursions

## 9C. Target Volatility Health Check

Verifies that the continuous target values have sufficient variance to train a regression model. If targets are constant or near-zero, the model cannot learn meaningful patterns.

- **Target mean:** 0.000004
- **Target std:** 0.000058
- **Non-zero targets:** 653 / 135775 (0.5%)

⚠️ **Warning:** Very few non-zero targets (0.5%)

## 9D. Information Coefficient (IC)

Measures the rank correlation between predicted probabilities and realized returns. This is the purest measure of ranking ability, independent of absolute calibration.

- **Spearman IC (prob, return):** 0.0870
- **P-value:** 4.4959e-89

⚠️ **Caution:** Weak IC (|IC| < 0.1) - limited practical value

## 10. Regime and Time-Conditional Label Checks

These metrics show how label base-rates and realized returns change across volatility regimes, trend states, and time-of-day/weekday. Large differences indicate that learnability and edge are regime-dependent, which is important for downstream model conditioning.


### Label Base-Rate by Volatility Regime

- **Regime high:** positive=28.3%, mean_return=-0.23%
- **Regime low:** positive=18.7%, mean_return=-0.33%
- **Regime medium:** positive=22.6%, mean_return=-0.32%

### Trend-Conditional (Price vs SMA20)

- **Strong uptrend:** positive=25.1%, mean_return=-0.28%
- **Strong downtrend:** positive=28.3%, mean_return=-0.22%

### Time-of-Day Positive Rates (Top/Bottom)

- **Top hours by positive rate:**
  - Hour 14: 27.9%
  - Hour 13: 27.3%
  - Hour 15: 26.9%
- **Bottom hours by positive rate:**
  - Hour 3: 19.1%
  - Hour 2: 19.9%
  - Hour 4: 20.4%

### Day-of-Week Positive Rates

- Day 0: 25.3%
- Day 1: 24.6%
- Day 2: 25.2%
- Day 3: 24.9%
- Day 4: 24.7%
- Day 5: 18.6%
- Day 6: 22.4%

## 11. Label–Return Separation, Information Content, and Sample Size

This section quantifies how well the labels separate profitable from unprofitable events (effect size) and how much information they carry about the sign of future returns compared to a random baseline (mutual information and permutation tests).


### Separation Metrics

- **Mean return difference (label=1 - label=0):** 1.85%
- **Cohen's d effect size:** 10.526
- **Approx. required samples for 80% power (heuristic):** 0.1
- **Current labeled samples used in separation:** 52696.0

### Information Content vs Permutation Baseline

- **Mutual information (labels vs realized sign):** 0.5474
- **Baseline MI (mean over permutations):** 0.0000

## 12. Target and Expected-Return Alignment

Here we check whether the continuous targets produced by the isotonic mapping are consistent with realized returns: higher targets should correspond to higher average realized P&L, and most non-zero targets should exceed transaction costs.


### Target vs Realized Return

- **Correlation (target, realized):** 0.027
- **MSE (target vs realized):** 0.000110

### Target/Return by Target Decile

- Decile 0: target=0.0006, realized=0.0006
- Decile 1: target=0.0007, realized=0.0009
- Decile 2: target=0.0020, realized=0.0018

### Target Distribution Sanity

- **Non-zero target fraction (all samples):** 0.5%
- **Mean non-zero target:** 0.0008
- **Median non-zero target:** 0.0007
- **Fraction of targets below transaction cost (0.150%):** 92.6%

## 13. Cost-Aware Metrics and Threshold P&L

These diagnostics evaluate the meta-model from a trading perspective: global AUC/Brier/PR-AUC, and how mean and cumulative returns evolve as you raise the probability threshold used to filter trades. This helps choose probability cutoffs that are profitable after costs.


### Cost-Aware Classification Metrics (OOF)

- **AUC:** 0.641
- **Brier score:** 0.2229
- **Average precision (PR-AUC):** 0.329

### Threshold-Sweep P&L (Using Meta Probability)


| Threshold | Trades | Mean Return | Cum Return | Sharpe (per trade) |
|----------|--------|-------------|------------|---------------------|
| 0.50 | 26725 | -0.17% | -4590.36% | -31.812 |
| 0.51 | 15937 | -0.13% | -2120.71% | -18.723 |
| 0.52 | 14711 | -0.12% | -1785.23% | -16.311 |
| 0.53 | 13510 | -0.11% | -1513.46% | -14.348 |
| 0.54 | 12292 | -0.11% | -1323.17% | -13.107 |
| 0.55 | 11114 | -0.10% | -1081.48% | -11.207 |
| 0.56 | 9989 | -0.08% | -835.23% | -9.071 |
| 0.57 | 8903 | -0.07% | -645.66% | -7.377 |
| 0.58 | 7762 | -0.06% | -486.74% | -5.926 |
| 0.59 | 6736 | -0.05% | -356.93% | -4.627 |
| 0.60 | 5708 | -0.04% | -224.87% | -3.135 |
| 0.61 | 4791 | -0.03% | -129.83% | -1.957 |
| 0.62 | 3932 | -0.02% | -86.36% | -1.423 |
| 0.63 | 3160 | -0.01% | -29.24% | -0.530 |
| 0.64 | 2428 | 0.01% | 14.17% | 0.290 |
| 0.65 | 1853 | 0.02% | 40.96% | 0.946 |

## 14. Stability and Bootstrap Statistics

Finally, we assess how stable the labeling and model performance are across time (years and time-series folds) and how sensitive key metrics such as positive rate and label–return separation are under bootstrap resampling. Tight confidence intervals indicate robust signal rather than fragile noise.


### Per-Year Label and Return Stability

- **Year 2021:** positive=25.8%, mean_return=-0.28%, AUC=0.500
- **Year 2022:** positive=27.7%, mean_return=-0.22%, AUC=0.570
- **Year 2023:** positive=17.2%, mean_return=-0.39%, AUC=0.694
- **Year 2024:** positive=22.8%, mean_return=-0.31%, AUC=0.653
- **Year 2025:** positive=26.1%, mean_return=-0.25%, AUC=0.635

### Time-Series Fold Stability (Approximate)

- **Fold 1:** positive=25.9%, mean_return=-0.25%, AUC=0.633
- **Fold 2:** positive=15.0%, mean_return=-0.42%, AUC=0.708
- **Fold 3:** positive=21.7%, mean_return=-0.33%, AUC=0.666
- **Fold 4:** positive=24.1%, mean_return=-0.29%, AUC=0.642
- **Fold 5:** positive=25.8%, mean_return=-0.25%, AUC=0.639

### Bootstrap Label Statistics (Labeled Events)

- **Positive rate 95% CI:** [23.3%, 24.0%]
- **Mean return diff (label=1 - label=0) 95% CI:** [1.85%, 1.85%]

---


## Summary and Recommendations


### Key Findings:

1. Label balance: 23.7% positive
2. Signal coverage: 38.8%
3. Mean return (label=1): 1.12%
4. Mean return (label=0): -0.73%
5. Calibration error: 0.223