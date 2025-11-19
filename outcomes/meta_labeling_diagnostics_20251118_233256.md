# Meta-Labeling Diagnostics Report

**Generated:** 2025-11-18 23:32:56

**Symbol:** ETHUSDT
**Timeframe:** 15m
**Horizon:** N/A bars

---


## 1. Label Distribution Analysis

- **Total labeled events:** 53765
- **Positive labels (profitable):** 12481 (23.2%)
- **Negative labels (unprofitable):** 41284 (76.8%)

⚠️ **Warning:** Low positive label rate (23.2%) - most signals are unprofitable

### Label Distribution Over Time

- **Daily positive rate - Mean:** 9.2%
- **Daily positive rate - Std:** 6.3%
- **Daily positive rate - Min:** 0.0%
- **Daily positive rate - Max:** 36.0%

⚠️ **Warning:** 813 days with <10% positive labels

## 2. Signal Coverage and Sparsity

- **Total samples:** 135775
- **Labeled samples:** 53765
- **Coverage:** 39.6%

✅ **OK:** Reasonable signal coverage (39.6%)

## 3. Feature-Label Correlation Analysis (Post-Filter)


### Top 10 Most Correlated Features (Post-Filter):

- **event_win_rate_last_50:** 0.2237
- **event_mean_return_last_50:** 0.2192
- **atr_ratio:** 0.1207
- **returns_entropy:** 0.1169
- **volatility_ema:** 0.1085
- **volatility_20:** 0.1047
- **returns_std_20:** 0.1047
- **volatility_4h:** 0.1042
- **atr_14:** 0.1041
- **range_4h:** 0.1030

### Correlation Health Check (Post-Filter):


## 3B. Pre- vs Post-Filter Comparison and Feature Shift


### Sample Counts and Retention

- **Pre-filter events (realized_return not NaN):** 66522
- **Pre-filter positive/negative (raw economic):** 21569 / 44953
- **Post-filter labeled events:** 53765
- **Post-filter positive/negative (binary_labels):** 12481 / 41284
- **Total retention (post / pre):** 80.8%
- **Positive retention:** 57.9%
- **Negative retention:** 91.8%

### Pre- vs Post-Filter Signal Quality

- **Pre-filter mean return (label=1/0):** 1.01% / -0.66%
- **Post-filter mean return (label=1/0):** 1.12% / -0.72%
- **Pre-filter Cohen's d (label=1 vs 0):** 5.761
- **Post-filter Cohen's d (label=1 vs 0):** 9.836
- **Pre-filter SNR (mean/std, label=1):** 2.982
- **Post-filter SNR (mean/std, label=1):** 6.103

### Largest Feature Correlation Shifts (|post|-|pre|)

- **signal_disagreement:** pre=-0.2695, post=-0.0842, Δ|corr|=-0.1853
- **event_mean_return_last_50:** pre=0.3510, post=0.2192, Δ|corr|=-0.1318
- **event_win_rate_last_50:** pre=-0.1072, post=0.2237, Δ|corr|=0.1165
- **atr_ratio:** pre=0.0451, post=0.1207, Δ|corr|=0.0755
- **returns_entropy:** pre=0.0479, post=0.1169, Δ|corr|=0.0689
- **range_1h:** pre=0.0356, post=0.1030, Δ|corr|=0.0674
- **atr_14:** pre=0.0412, post=0.1041, Δ|corr|=0.0629
- **volatility_ema:** pre=0.0457, post=0.1085, Δ|corr|=0.0628
- **returns_std_10:** pre=0.0389, post=0.1014, Δ|corr|=0.0625
- **returns_std_5:** pre=0.0342, post=0.0967, Δ|corr|=0.0625

## 4. P&L Distribution Per Label

### Label = 1 (Profitable Signals):

- **Count:** 12481
- **Mean return:** 1.12%
- **Median return:** 1.03%
- **Std return:** 0.18%
- **% Actually positive:** 100.0%

### Label = 0 (Unprofitable Signals):

- **Count:** 41284
- **Mean return:** -0.72%
- **Median return:** -0.71%
- **Std return:** 0.19%
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

- **Win rate vs Volatility correlation:** 0.4593
- **Mean return vs Volatility correlation:** 0.1354

✅ **OK:** Win rate not strongly correlated with volatility

### Rolling Daily SNR (Return / Std by Day)

- **Daily SNR mean:** -0.2444
- **Daily SNR std:** 0.4793
- **Daily SNR min/max:** -3.4667 / 1.3608

### Regime Shift Detection:

- **30-day rolling correlation (win rate vs vol) - Mean:** 0.3811
- **30-day rolling correlation (win rate vs vol) - Std:** 0.1568

## 6. Model Probability Diagnostics


### Calibration (Predicted Probability vs Actual Success Rate):


| Predicted Prob | Actual Success Rate | Count |
|---------------|---------------------|-------|
| 0.160 | 0.007 | 1251 |
| 0.215 | 0.042 | 2816 |
| 0.283 | 0.105 | 4080 |
| 0.351 | 0.147 | 5638 |
| 0.418 | 0.200 | 7156 |
| 0.493 | 0.262 | 17628 |
| 0.551 | 0.292 | 7699 |
| 0.616 | 0.361 | 5701 |
| 0.677 | 0.408 | 1652 |
| 0.743 | 0.431 | 144 |

- **Mean calibration error:** 0.225

⚠️ **Warning:** High calibration error - model probabilities may not be well-calibrated

### Monotonicity and Slope Checks (Prob → Realized Return)

- **Monotonicity violations (adjacent bins):** 0 / 9
- **Approx. slope in high-probability region:** 0.015084

### Probability Distribution:

- **Mean probability:** 0.483
- **Median probability:** 0.500
- **Std probability:** 0.080

## 7. Feature Importance Analysis


### Top 20 Features by Importance:

1. **high_dist_x_vol:** 0.1277
2. **vol_range_interaction:** 0.1270
3. **event_mean_return_last_50:** 0.0274
4. **bars_since_last_event:** 0.0227
5. **sma_slope:** 0.0216
6. **vol_ratio:** 0.0184
7. **returns_std_50:** 0.0150
8. **dist_from_recent_high_5:** 0.0140
9. **volatility_1d:** 0.0133
10. **vol_price_corr:** 0.0131
11. **returns_1h:** 0.0129
12. **volatility_ema:** 0.0126
13. **volatility_4h:** 0.0124
14. **momentum_20:** 0.0123
15. **returns_4h:** 0.0120
16. **close_max_50:** 0.0117
17. **volume_trend:** 0.0113
18. **dist_from_recent_high_10:** 0.0112
19. **returns_entropy:** 0.0112
20. **dist_from_recent_high_20:** 0.0110

### Feature Importance Health Check:


✅ **OK:** No single feature dominates

✅ Kalman-filtered features are being used (6 features)

✅ Volatility-based features in top 10: high_dist_x_vol, vol_range_interaction, vol_ratio, volatility_1d, vol_price_corr

## 8. Kalman Smoothed Labels Analysis

- **Mean smoothed label:** 0.232
- **Median smoothed label:** 0.231
- **Std smoothed label:** 0.107
- **Correlation with binary labels:** 0.299
- **Correlation with realized returns:** 0.260

⚠️ **Warning:** Low correlation between smoothed and binary labels - Kalman filter may be over-smoothing

## 9. Event Mechanics and R-Multiples

These diagnostics describe how trades exit (profit, stop, or timeout), how long they stay open, and how large the realized return is relative to the configured stop-loss (R-multiple). They help verify that the triple-barrier / TPSL configuration produces economically meaningful events.


### Exit Reason Mix (Labeled Events)

- **stop:** 70.8%
- **profit:** 23.2%
- **timeout:** 6.0%

### Event Duration Distribution (Bars)

- **Mean duration:** 9.08
- **Median duration:** 6.00
- **90th percentile:** 22.00

### R-Multiple Distribution (Return / Stop Threshold)

- **R-multiple (label=1) mean:** 1.94
- **R-multiple (label=1) median:** 1.82
- **R-multiple (label=0) mean:** -1.22
- **R-multiple (label=0) median:** -1.26

## 9B. Enhanced Event Quality Metrics

These advanced metrics help distinguish between efficient momentum capture and random drift trades. They measure path efficiency, timing quality, and entry/exit effectiveness.


### Path Efficiency Ratio (PER)

- **Mean PER:** 136.166
- **Median PER:** 1.714

✅ **OK:** Reasonable path efficiency

### Time-to-Outcome Ratio (TTO)

- **Mean TTO:** 0.567
- **Median TTO:** 0.375

✅ **OK:** TTO in healthy range [0.4, 0.6]

### MFE/MAE Ratio (Maximum Favorable vs Adverse Excursion)

- **Mean MFE/MAE:** 162.499
- **Median MFE/MAE:** 0.541

✅ **OK:** Favorable excursions exceed adverse excursions

## 9C. Target Volatility Health Check

Verifies that the continuous target values have sufficient variance to train a regression model. If targets are constant or near-zero, the model cannot learn meaningful patterns.

- **Target mean:** 0.000004
- **Target std:** 0.000071
- **Non-zero targets:** 558 / 135775 (0.4%)

⚠️ **Warning:** Very few non-zero targets (0.4%)

## 9D. Information Coefficient (IC)

Measures the rank correlation between predicted probabilities and realized returns. This is the purest measure of ranking ability, independent of absolute calibration.

- **Spearman IC (prob, return):** 0.0788
- **P-value:** 6.9721e-75

⚠️ **Caution:** Weak IC (|IC| < 0.1) - limited practical value

## 10. Regime and Time-Conditional Label Checks

These metrics show how label base-rates and realized returns change across volatility regimes, trend states, and time-of-day/weekday. Large differences indicate that learnability and edge are regime-dependent, which is important for downstream model conditioning.


### Label Base-Rate by Volatility Regime

- **Regime high:** positive=27.9%, mean_return=-0.23%
- **Regime low:** positive=18.0%, mean_return=-0.32%
- **Regime medium:** positive=22.3%, mean_return=-0.32%

### Trend-Conditional (Price vs SMA20)

- **Strong uptrend:** positive=24.7%, mean_return=-0.28%
- **Strong downtrend:** positive=27.9%, mean_return=-0.22%

### Time-of-Day Positive Rates (Top/Bottom)

- **Top hours by positive rate:**
  - Hour 14: 27.6%
  - Hour 13: 27.1%
  - Hour 15: 26.5%
- **Bottom hours by positive rate:**
  - Hour 3: 18.4%
  - Hour 2: 19.4%
  - Hour 4: 19.7%

### Day-of-Week Positive Rates

- Day 0: 24.8%
- Day 1: 24.1%
- Day 2: 24.9%
- Day 3: 24.5%
- Day 4: 24.2%
- Day 5: 17.8%
- Day 6: 22.1%

## 11. Label–Return Separation, Information Content, and Sample Size

This section quantifies how well the labels separate profitable from unprofitable events (effect size) and how much information they carry about the sign of future returns compared to a random baseline (mutual information and permutation tests).


### Separation Metrics

- **Mean return difference (label=1 - label=0):** 1.84%
- **Cohen's d effect size:** 9.836
- **Approx. required samples for 80% power (heuristic):** 0.2
- **Current labeled samples used in separation:** 53765.0

### Information Content vs Permutation Baseline

- **Mutual information (labels vs realized sign):** 0.5418
- **Baseline MI (mean over permutations):** 0.0000

## 12. Target and Expected-Return Alignment

Here we check whether the continuous targets produced by the isotonic mapping are consistent with realized returns: higher targets should correspond to higher average realized P&L, and most non-zero targets should exceed transaction costs.


### Target vs Realized Return

- **Correlation (target, realized):** 0.038
- **MSE (target vs realized):** 0.000110

### Target/Return by Target Decile

- Decile 0: target=0.0010, realized=0.0010
- Decile 1: target=0.0063, realized=0.0063

### Target Distribution Sanity

- **Non-zero target fraction (all samples):** 0.4%
- **Mean non-zero target:** 0.0010
- **Median non-zero target:** 0.0010
- **Fraction of targets below transaction cost (0.150%):** 99.5%

## 13. Cost-Aware Metrics and Threshold P&L

These diagnostics evaluate the meta-model from a trading perspective: global AUC/Brier/PR-AUC, and how mean and cumulative returns evolve as you raise the probability threshold used to filter trades. This helps choose probability cutoffs that are profitable after costs.


### Cost-Aware Classification Metrics (OOF)

- **AUC:** 0.648
- **Brier score:** 0.2210
- **Average precision (PR-AUC):** 0.328

### Threshold-Sweep P&L (Using Meta Probability)


| Threshold | Trades | Mean Return | Cum Return | Sharpe (per trade) |
|----------|--------|-------------|------------|---------------------|
| 0.50 | 27041 | -0.17% | -4599.50% | -31.814 |
| 0.51 | 16209 | -0.13% | -2143.18% | -18.856 |
| 0.52 | 15013 | -0.12% | -1801.10% | -16.374 |
| 0.53 | 13816 | -0.11% | -1526.88% | -14.398 |
| 0.54 | 12605 | -0.10% | -1303.50% | -12.805 |
| 0.55 | 11463 | -0.09% | -1082.61% | -11.104 |
| 0.56 | 10331 | -0.08% | -813.69% | -8.739 |
| 0.57 | 9223 | -0.07% | -677.60% | -7.670 |
| 0.58 | 8134 | -0.06% | -484.98% | -5.801 |
| 0.59 | 7094 | -0.05% | -355.15% | -4.520 |
| 0.60 | 6077 | -0.04% | -264.73% | -3.620 |
| 0.61 | 5060 | -0.03% | -150.54% | -2.232 |
| 0.62 | 4197 | -0.02% | -85.31% | -1.376 |
| 0.63 | 3372 | -0.02% | -51.86% | -0.922 |
| 0.64 | 2673 | -0.01% | -22.69% | -0.448 |
| 0.65 | 2023 | -0.02% | -30.59% | -0.687 |

## 14. Stability and Bootstrap Statistics

Finally, we assess how stable the labeling and model performance are across time (years and time-series folds) and how sensitive key metrics such as positive rate and label–return separation are under bootstrap resampling. Tight confidence intervals indicate robust signal rather than fragile noise.


### Per-Year Label and Return Stability

- **Year 2021:** positive=25.6%, mean_return=-0.28%, AUC=0.500
- **Year 2022:** positive=27.4%, mean_return=-0.22%, AUC=0.576
- **Year 2023:** positive=16.4%, mean_return=-0.38%, AUC=0.706
- **Year 2024:** positive=22.4%, mean_return=-0.31%, AUC=0.659
- **Year 2025:** positive=25.8%, mean_return=-0.25%, AUC=0.637

### Time-Series Fold Stability (Approximate)

- **Fold 1:** positive=25.3%, mean_return=-0.25%, AUC=0.640
- **Fold 2:** positive=14.2%, mean_return=-0.41%, AUC=0.721
- **Fold 3:** positive=21.3%, mean_return=-0.33%, AUC=0.673
- **Fold 4:** positive=23.7%, mean_return=-0.29%, AUC=0.645
- **Fold 5:** positive=25.6%, mean_return=-0.25%, AUC=0.641

### Bootstrap Label Statistics (Labeled Events)

- **Positive rate 95% CI:** [22.8%, 23.6%]
- **Mean return diff (label=1 - label=0) 95% CI:** [1.83%, 1.84%]

---


## Summary and Recommendations


### Key Findings:

1. Label balance: 23.2% positive
2. Signal coverage: 39.6%
3. Mean return (label=1): 1.12%
4. Mean return (label=0): -0.72%
5. Calibration error: 0.225