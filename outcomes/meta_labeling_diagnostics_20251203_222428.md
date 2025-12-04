# Meta-Labeling Diagnostics Report

**Generated:** 2025-12-03 22:24:28

**Symbol:** ETHUSDT
**Timeframe:** 15m
**Horizon:** N/A bars

---


## 1. Label Distribution Analysis

- **Total labeled events:** 4128
- **Positive labels (profitable):** 2064 (50.0%)
- **Negative labels (unprofitable):** 2064 (50.0%)

✅ **OK:** Reasonable label balance (50.0%)

### Label Distribution Over Time

- **Daily positive rate - Mean:** 5.5%
- **Daily positive rate - Std:** 4.1%
- **Daily positive rate - Min:** 0.0%
- **Daily positive rate - Max:** 15.6%

⚠️ **Warning:** 337 days with <10% positive labels

## 2. Signal Coverage and Sparsity

- **Total samples:** 34135
- **Labeled samples:** 4128
- **Coverage:** 12.1%

✅ **OK:** Reasonable signal coverage (12.1%)

## 3. Feature-Label Correlation Analysis (Post-Filter)


### Top 10 Most Correlated Features (Post-Filter):

- **volatility_1d:** 0.2817
- **event_mean_return_last_50:** 0.2682
- **event_r_multiple_mean_last_50:** 0.2646
- **event_win_rate_last_50:** 0.2633
- **returns_std_50:** 0.1947
- **event_tto_mean_last_50:** -0.1928
- **low_dist_x_vol:** 0.1502
- **close_min_50:** -0.1250
- **close_max_10:** -0.1223
- **close_min_20:** -0.1223

### Correlation Health Check (Post-Filter):


## 3B. Pre- vs Post-Filter Comparison and Feature Shift


### Sample Counts and Retention

- **Pre-filter events (realized_return not NaN):** 8304
- **Pre-filter positive/negative (raw economic):** 2670 / 5634
- **Post-filter labeled events:** 4128
- **Post-filter positive/negative (binary_labels):** 2064 / 2064
- **Total retention (post / pre):** 49.7%
- **Positive retention:** 77.3%
- **Negative retention:** 36.6%

### Pre- vs Post-Filter Signal Quality

- **Pre-filter mean return (label=1/0):** 0.82% / -0.57%
- **Post-filter mean return (label=1/0):** 0.86% / -0.65%
- **Pre-filter Cohen's d (label=1 vs 0):** 7.056
- **Post-filter Cohen's d (label=1 vs 0):** 15.110
- **Pre-filter SNR (mean/std, label=1):** 4.059
- **Post-filter SNR (mean/std, label=1):** 6.665

### Largest Feature Correlation Shifts (|post|-|pre|)

- **signal_disagreement:** pre=0.3152, post=-0.0079, Δ|corr|=-0.3073
- **volatility_1d:** pre=-0.0066, post=0.2817, Δ|corr|=0.2751
- **returns_std_50:** pre=-0.0052, post=0.1947, Δ|corr|=0.1895
- **event_win_rate_last_50:** pre=0.4277, post=0.2633, Δ|corr|=-0.1645
- **low_dist_x_vol:** pre=-0.0047, post=0.1502, Δ|corr|=0.1456
- **close_min_50:** pre=-0.0154, post=-0.1250, Δ|corr|=0.1097
- **kalman_trend:** pre=-0.0150, post=-0.1219, Δ|corr|=0.1069
- **close_max_5:** pre=-0.0163, post=-0.1222, Δ|corr|=0.1059
- **close_max_50:** pre=-0.0143, post=-0.1202, Δ|corr|=0.1058
- **close_max_10:** pre=-0.0167, post=-0.1223, Δ|corr|=0.1056

## 4. P&L Distribution Per Label

### Label = 1 (Profitable Signals):

- **Count:** 2064
- **Mean return:** 0.86%
- **Median return:** 0.84%
- **Std return:** 0.13%
- **% Actually positive:** 100.0%

### Label = 0 (Unprofitable Signals):

- **Count:** 2064
- **Mean return:** -0.65%
- **Median return:** -0.66%
- **Std return:** 0.06%
- **% Actually positive:** 0.0%

### Labeling Quality:

- **Label overlap:** 0.0%

✅ **OK:** Acceptable label overlap (0.0%)

### Cost-Aware Event Quality Summary

- **Transaction cost (per event, approx):** 0.150%
- **Unconditional mean event return:** 0.11%
- **Mean return (label=1) minus cost:** 0.71%
- **Fraction of labeled events with |return| < cost:** 0.0%

## 5. Time-Series Stability and Regime Analysis

- **Win rate vs Volatility correlation:** 0.2089
- **Mean return vs Volatility correlation:** 0.0739

✅ **OK:** Win rate not strongly correlated with volatility

### Rolling Daily SNR (Return / Std by Day)

- **Daily SNR mean:** -7952.5932
- **Daily SNR std:** 69506.6688
- **Daily SNR min/max:** -728781.6552 / 3.8673

### Regime Shift Detection:

- **30-day rolling correlation (win rate vs vol) - Mean:** 0.3914
- **30-day rolling correlation (win rate vs vol) - Std:** 0.1280

## 6. Model Probability Diagnostics


### Calibration (Predicted Probability vs Actual Success Rate):


| Predicted Prob | Actual Success Rate | Count |
|---------------|---------------------|-------|
| 0.124 | 0.404 | 47 |
| 0.209 | 0.312 | 224 |
| 0.298 | 0.323 | 402 |
| 0.390 | 0.404 | 525 |
| 0.487 | 0.433 | 984 |
| 0.569 | 0.419 | 511 |
| 0.659 | 0.471 | 461 |
| 0.751 | 0.615 | 361 |
| 0.843 | 0.839 | 322 |
| 0.929 | 0.976 | 291 |

- **Mean calibration error:** 0.100

✅ **OK:** Reasonable calibration

### Monotonicity and Slope Checks (Prob → Realized Return)

- **Monotonicity violations (adjacent bins):** 2 / 9
- **Approx. slope in high-probability region:** 0.030155

### Probability Distribution:

- **Mean probability:** 0.506
- **Median probability:** 0.500
- **Std probability:** 0.072

## 7. Feature Importance Analysis


### Top 20 Features by Importance:

1. **volatility_1d:** 0.0857
2. **high_dist_x_vol:** 0.0410
3. **event_mean_return_last_50:** 0.0398
4. **vol_range_interaction:** 0.0290
5. **vol_price_corr:** 0.0259
6. **bars_since_last_event:** 0.0225
7. **dist_from_recent_high_20:** 0.0167
8. **atr_momentum:** 0.0140
9. **momentum_5_x_regime_medium:** 0.0138
10. **dist_from_recent_high_50:** 0.0121
11. **drawdown_100:** 0.0119
12. **volume_trend:** 0.0116
13. **volume_ratio:** 0.0114
14. **ma_distance_per_vol:** 0.0112
15. **regime_1_prob:** 0.0108
16. **volatility_4h_agg:** 0.0108
17. **regime_0_prob:** 0.0107
18. **atr_14:** 0.0101
19. **event_tto_mean_last_50:** 0.0100
20. **returns_std_10:** 0.0098

### Feature Importance Health Check:


✅ **OK:** No single feature dominates

✅ Kalman-filtered features are being used (6 features)

✅ Volatility-based features in top 10: volatility_1d, high_dist_x_vol, vol_range_interaction, vol_price_corr

## 8. Kalman Smoothed Labels Analysis

- **Mean smoothed label:** 0.503
- **Median smoothed label:** 0.497
- **Std smoothed label:** 0.424
- **Correlation with binary labels:** 0.970
- **Correlation with realized returns:** 0.960

✅ **OK:** Good correlation between smoothed and binary labels

## 9. Event Mechanics and R-Multiples

These diagnostics describe how trades exit (profit, stop, or timeout), how long they stay open, and how large the realized return is relative to the configured stop-loss (R-multiple). They help verify that the triple-barrier / TPSL configuration produces economically meaningful events.


### Exit Reason Mix (Labeled Events)

- **1:** 49.7%
- **0:** 47.8%
- **2:** 2.4%

### Event Duration Distribution (Bars)

- **Mean duration:** 4.78
- **Median duration:** 4.00
- **90th percentile:** 11.00

### R-Multiple Distribution (Return / Stop Threshold)

- **R-multiple (label=1) mean:** 1.75
- **R-multiple (label=1) median:** 1.72
- **R-multiple (label=0) mean:** -1.30
- **R-multiple (label=0) median:** -1.30

## 9B. Enhanced Event Quality Metrics

These advanced metrics help distinguish between efficient momentum capture and random drift trades. They measure path efficiency, timing quality, and entry/exit effectiveness.


### Path Efficiency Ratio (PER)

- **Mean PER (clipped 99.5%):** 0.561
- **Median PER:** 0.574
- **90th percentile PER:** 0.737
- **99th percentile PER:** 0.824

✅ **OK:** Good path efficiency on profitable events

### Time-to-Outcome Ratio (TTO)

- **Mean TTO:** 0.299
- **Median TTO:** 0.250

⚠️ **Warning:** TTO outside target range [0.4, 0.6]

### MFE/MAE Ratio (Maximum Favorable vs Adverse Excursion)

- **Mean MFE/MAE:** 388.061
- **Median MFE/MAE:** 1.682

✅ **OK:** Favorable excursions exceed adverse excursions

## 9C. Target Volatility Health Check

Verifies that the continuous target values have sufficient variance to train a regression model. If targets are constant or near-zero, the model cannot learn meaningful patterns.

- **Target mean:** -0.000297
- **Target std:** 0.001122
- **Non-zero targets:** 1125 / 34135 (3.3%)

✅ **OK:** Targets have sufficient variance

## 9D. Information Coefficient (IC)

Measures the rank correlation between predicted probabilities and realized returns. This is the purest measure of ranking ability, independent of absolute calibration.

- **Spearman IC (prob, return):** 0.2458
- **P-value:** 7.1080e-58

✅ **OK:** Meaningful rank correlation

## 10. Regime and Time-Conditional Label Checks

These metrics show how label base-rates and realized returns change across volatility regimes, trend states, and time-of-day/weekday. Large differences indicate that learnability and edge are regime-dependent, which is important for downstream model conditioning.


### Label Base-Rate by Volatility Regime

- **Regime high:** positive=70.7%, mean_return=0.44%
- **Regime low:** positive=37.9%, mean_return=-0.08%
- **Regime medium:** positive=50.1%, mean_return=0.10%

### Trend-Conditional (Price vs SMA20)

- **Strong uptrend:** positive=51.2%, mean_return=0.13%
- **Strong downtrend:** positive=45.8%, mean_return=0.05%

### Time-of-Day Positive Rates (Top/Bottom)

- **Top hours by positive rate:**
  - Hour 5: 61.3%
  - Hour 10: 59.4%
  - Hour 3: 56.2%
- **Bottom hours by positive rate:**
  - Hour 13: 38.8%
  - Hour 0: 44.0%
  - Hour 19: 44.2%

### Day-of-Week Positive Rates

- Day 0: 52.8%
- Day 1: 50.6%
- Day 2: 50.6%
- Day 3: 55.2%
- Day 4: 52.2%
- Day 5: 49.8%
- Day 6: 39.5%

## 11. Label–Return Separation, Information Content, and Sample Size

This section quantifies how well the labels separate profitable from unprofitable events (effect size) and how much information they carry about the sign of future returns compared to a random baseline (mutual information and permutation tests).


### Separation Metrics

- **Mean return difference (label=1 - label=0):** 1.52%
- **Cohen's d effect size:** 15.110
- **Approx. required samples for 80% power (heuristic):** 0.1
- **Current labeled samples used in separation:** 4128.0

### Information Content vs Permutation Baseline

- **Mutual information (labels vs realized sign):** 0.6931
- **Baseline MI (mean over permutations):** 0.0001

## 12. Target and Expected-Return Alignment

Here we check whether the continuous targets produced by the isotonic mapping are consistent with realized returns: higher targets should correspond to higher average realized P&L, and most non-zero targets should exceed transaction costs.


### Target vs Realized Return

- **Correlation (target, realized):** 0.374
- **MSE (target vs realized):** 0.000039

### Target/Return by Target Decile

- Decile 0: target=0.0009, realized=0.0024
- Decile 1: target=0.0015, realized=0.0030
- Decile 2: target=0.0032, realized=0.0047
- Decile 3: target=0.0044, realized=0.0059
- Decile 4: target=0.0058, realized=0.0073
- Decile 5: target=0.0058, realized=0.0081

### Target Distribution Sanity

- **Non-zero target fraction (all samples):** 3.3%
- **Mean non-zero target:** 0.0034
- **Median non-zero target:** 0.0032
- **Fraction of targets below transaction cost (0.150%):** 39.6%

## 13. Cost-Aware Metrics and Threshold P&L

These diagnostics evaluate the meta-model from a trading perspective: global AUC/Brier/PR-AUC, and how mean and cumulative returns evolve as you raise the probability threshold used to filter trades. This helps choose probability cutoffs that are profitable after costs.


### Cost-Aware Classification Metrics (OOF)

- **AUC:** 0.678
- **Brier score:** 0.2263
- **Average precision (PR-AUC):** 0.729

### Volatility Regime Breakdown

| Regime | Samples | Pos Rate | AUC | Mean Ret | Sharpe |
|--------|---------|----------|-----|----------|--------|
| Low | 1741 | 37.9% | 0.556 | -0.08% | -0.11 |
| Medium | 1370 | 50.1% | 0.676 | 0.10% | 0.13 |
| High | 1017 | 70.7% | 0.731 | 0.44% | 0.60 |

⚠️ **Warning:** Large win-rate disparity between regimes (low: 37.9%, high: 70.7%). Performance is highly regime-dependent.

### Threshold-Sweep P&L (Using Meta Probability)


| Threshold | Trades | Mean Return | Cum Return | Sharpe (per trade) |
|----------|--------|-------------|------------|---------------------|
| 0.50 | 2507 | 0.23% | 571.83% | 15.186 |
| 0.51 | 2029 | 0.27% | 551.07% | 16.461 |
| 0.52 | 1966 | 0.28% | 555.87% | 16.934 |
| 0.53 | 1908 | 0.29% | 560.63% | 17.410 |
| 0.54 | 1853 | 0.30% | 553.89% | 17.508 |
| 0.55 | 1801 | 0.31% | 563.64% | 18.169 |
| 0.56 | 1735 | 0.33% | 564.72% | 18.671 |
| 0.57 | 1680 | 0.34% | 563.87% | 19.008 |
| 0.58 | 1627 | 0.34% | 555.20% | 19.052 |
| 0.59 | 1567 | 0.35% | 555.40% | 19.524 |
| 0.60 | 1515 | 0.37% | 556.54% | 20.037 |
| 0.61 | 1463 | 0.38% | 556.08% | 20.531 |
| 0.62 | 1401 | 0.41% | 569.73% | 21.827 |
| 0.63 | 1348 | 0.42% | 569.50% | 22.459 |
| 0.64 | 1293 | 0.44% | 570.42% | 23.266 |
| 0.65 | 1252 | 0.46% | 575.33% | 24.206 |

## 14. Stability and Bootstrap Statistics

Finally, we assess how stable the labeling and model performance are across time (years and time-series folds) and how sensitive key metrics such as positive rate and label–return separation are under bootstrap resampling. Tight confidence intervals indicate robust signal rather than fragile noise.


### Per-Year Label and Return Stability

- **Year 2021:** positive=47.0%, mean_return=0.05%, AUC=0.523
- **Year 2022:** positive=50.7%, mean_return=0.12%, AUC=0.689
- **Year 2023:** positive=46.2%, mean_return=0.04%, AUC=0.718
- **Year 2024:** positive=72.2%, mean_return=0.42%, AUC=0.854
- **Year 2025:** positive=44.1%, mean_return=0.02%, AUC=0.738

### Time-Series Fold Stability (Approximate)

- **Fold 1:** positive=52.6%, mean_return=0.15%, AUC=0.703
- **Fold 2:** positive=42.0%, mean_return=-0.01%, AUC=0.608
- **Fold 3:** positive=50.8%, mean_return=0.12%, AUC=0.726
- **Fold 4:** positive=58.1%, mean_return=0.23%, AUC=0.761
- **Fold 5:** positive=50.7%, mean_return=0.12%, AUC=0.764

### Bootstrap Label Statistics (Labeled Events)

- **Positive rate 95% CI:** [48.6%, 51.9%]
- **Mean return diff (label=1 - label=0) 95% CI:** [1.51%, 1.52%]

---


## Summary and Recommendations


### Key Findings:

1. Label balance: 50.0% positive
2. Signal coverage: 12.1%
3. Mean return (label=1): 0.86%
4. Mean return (label=0): -0.65%
5. Calibration error: 0.100