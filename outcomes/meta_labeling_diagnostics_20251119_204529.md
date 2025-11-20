# Meta-Labeling Diagnostics Report

**Generated:** 2025-11-19 20:45:29

**Symbol:** ETHUSDT
**Timeframe:** 15m
**Horizon:** N/A bars

---


## 1. Label Distribution Analysis

- **Total labeled events:** 36238
- **Positive labels (profitable):** 18119 (50.0%)
- **Negative labels (unprofitable):** 18119 (50.0%)

✅ **OK:** Reasonable label balance (50.0%)

### Label Distribution Over Time

- **Daily positive rate - Mean:** 13.3%
- **Daily positive rate - Std:** 7.8%
- **Daily positive rate - Min:** 0.0%
- **Daily positive rate - Max:** 43.8%

⚠️ **Warning:** 489 days with <10% positive labels

## 2. Signal Coverage and Sparsity

- **Total samples:** 135775
- **Labeled samples:** 36238
- **Coverage:** 26.7%

✅ **OK:** Reasonable signal coverage (26.7%)

## 3. Feature-Label Correlation Analysis (Post-Filter)


### Top 10 Most Correlated Features (Post-Filter):

- **event_win_rate_last_50:** 0.4856
- **event_mean_return_last_50:** 0.4855
- **event_r_multiple_mean_last_50:** 0.4792
- **volatility_1d:** 0.4234
- **returns_std_50:** 0.3622
- **returns_entropy:** 0.3302
- **atr_ratio:** 0.3227
- **volatility_20:** 0.3061
- **returns_std_20:** 0.3061
- **volatility_ema:** 0.3042

### Correlation Health Check (Post-Filter):


## 3B. Pre- vs Post-Filter Comparison and Feature Shift


### Sample Counts and Retention

- **Pre-filter events (realized_return not NaN):** 66522
- **Pre-filter positive/negative (raw economic):** 21569 / 44953
- **Post-filter labeled events:** 36238
- **Post-filter positive/negative (binary_labels):** 18119 / 18119
- **Total retention (post / pre):** 54.5%
- **Positive retention:** 84.0%
- **Negative retention:** 40.3%

### Pre- vs Post-Filter Signal Quality

- **Pre-filter mean return (label=1/0):** 1.01% / -0.66%
- **Post-filter mean return (label=1/0):** 1.10% / -0.74%
- **Pre-filter Cohen's d (label=1 vs 0):** 5.761
- **Post-filter Cohen's d (label=1 vs 0):** 8.884
- **Pre-filter SNR (mean/std, label=1):** 2.982
- **Post-filter SNR (mean/std, label=1):** 4.667

### Largest Feature Correlation Shifts (|post|-|pre|)

- **volatility_1d:** pre=0.0435, post=0.4234, Δ|corr|=0.3799
- **returns_std_50:** pre=0.0434, post=0.3622, Δ|corr|=0.3187
- **returns_entropy:** pre=0.0479, post=0.3302, Δ|corr|=0.2822
- **atr_ratio:** pre=0.0451, post=0.3227, Δ|corr|=0.2776
- **volatility_20:** pre=0.0451, post=0.3061, Δ|corr|=0.2611
- **returns_std_20:** pre=0.0451, post=0.3061, Δ|corr|=0.2611
- **volatility_ema:** pre=0.0457, post=0.3042, Δ|corr|=0.2585
- **volatility_4h:** pre=0.0431, post=0.2933, Δ|corr|=0.2503
- **atr_14:** pre=0.0412, post=0.2798, Δ|corr|=0.2386
- **close_range_50:** pre=0.0451, post=0.2803, Δ|corr|=0.2352

## 4. P&L Distribution Per Label

### Label = 1 (Profitable Signals):

- **Count:** 18119
- **Mean return:** 1.10%
- **Median return:** 1.03%
- **Std return:** 0.24%
- **% Actually positive:** 100.0%

### Label = 0 (Unprofitable Signals):

- **Count:** 18119
- **Mean return:** -0.74%
- **Median return:** -0.72%
- **Std return:** 0.17%
- **% Actually positive:** 0.0%

### Labeling Quality:

- **Label overlap:** 0.0%

✅ **OK:** Acceptable label overlap (0.0%)

### Cost-Aware Event Quality Summary

- **Transaction cost (per event, approx):** 0.150%
- **Unconditional mean event return:** 0.18%
- **Mean return (label=1) minus cost:** 0.95%
- **Fraction of labeled events with |return| < cost:** 0.0%

## 5. Time-Series Stability and Regime Analysis

- **Win rate vs Volatility correlation:** 0.5859
- **Mean return vs Volatility correlation:** 0.1354

⚠️ **Warning:** Strong correlation between win rate and volatility - performance is regime-dependent

### Rolling Daily SNR (Return / Std by Day)

- **Daily SNR mean:** -0.2444
- **Daily SNR std:** 0.4793
- **Daily SNR min/max:** -3.4667 / 1.3608

### Regime Shift Detection:

- **30-day rolling correlation (win rate vs vol) - Mean:** nan
- **30-day rolling correlation (win rate vs vol) - Std:** nan

## 6. Model Probability Diagnostics


### Calibration (Predicted Probability vs Actual Success Rate):


| Predicted Prob | Actual Success Rate | Count |
|---------------|---------------------|-------|
| 0.112 | 0.091 | 1132 |
| 0.198 | 0.192 | 2665 |
| 0.293 | 0.272 | 3870 |
| 0.387 | 0.348 | 5624 |
| 0.486 | 0.525 | 10242 |
| 0.569 | 0.462 | 3665 |
| 0.664 | 0.529 | 2190 |
| 0.761 | 0.689 | 1433 |
| 0.861 | 0.927 | 1721 |
| 0.952 | 0.996 | 3696 |

- **Mean calibration error:** 0.055

✅ **OK:** Reasonable calibration

### Monotonicity and Slope Checks (Prob → Realized Return)

- **Monotonicity violations (adjacent bins):** 1 / 9
- **Approx. slope in high-probability region:** 0.036420

### Probability Distribution:

- **Mean probability:** 0.503
- **Median probability:** 0.500
- **Std probability:** 0.116

## 7. Feature Importance Analysis


### Top 20 Features by Importance:

1. **high_dist_x_vol:** 0.1302
2. **vol_range_interaction:** 0.1271
3. **event_mean_return_last_50:** 0.0985
4. **volatility_1d:** 0.0791
5. **bars_since_last_event:** 0.0584
6. **dist_from_recent_high_20:** 0.0297
7. **drawdown_100:** 0.0280
8. **sma_slope:** 0.0259
9. **vol_momentum_interaction:** 0.0214
10. **momentum_5_x_regime_medium:** 0.0187
11. **dist_from_recent_high_50:** 0.0177
12. **vol_ratio:** 0.0164
13. **event_r_multiple_mean_last_50:** 0.0162
14. **volatility_5:** 0.0157
15. **momentum_20:** 0.0156
16. **volatility_4h:** 0.0155
17. **event_tto_mean_last_50:** 0.0149
18. **dist_from_recent_high_10:** 0.0137
19. **momentum_per_vol:** 0.0113
20. **range_position:** 0.0109

### Feature Importance Health Check:


✅ **OK:** No single feature dominates

✅ Kalman-filtered features are being used (6 features)

✅ Volatility-based features in top 10: high_dist_x_vol, vol_range_interaction, volatility_1d, vol_momentum_interaction

## 8. Kalman Smoothed Labels Analysis

- **Mean smoothed label:** 0.499
- **Median smoothed label:** 0.437
- **Std smoothed label:** 0.263
- **Correlation with binary labels:** 0.539
- **Correlation with realized returns:** 0.528

✅ **OK:** Good correlation between smoothed and binary labels

## 9. Event Mechanics and R-Multiples

These diagnostics describe how trades exit (profit, stop, or timeout), how long they stay open, and how large the realized return is relative to the configured stop-loss (R-multiple). They help verify that the triple-barrier / TPSL configuration produces economically meaningful events.


### Exit Reason Mix (Labeled Events)

- **stop:** 48.7%
- **profit:** 45.9%
- **timeout:** 5.4%

### Event Duration Distribution (Bars)

- **Mean duration:** 12.26
- **Median duration:** 9.00
- **90th percentile:** 28.00

### R-Multiple Distribution (Return / Stop Threshold)

- **R-multiple (label=1) mean:** 1.83
- **R-multiple (label=1) median:** 1.81
- **R-multiple (label=0) mean:** -1.26
- **R-multiple (label=0) median:** -1.26

## 9B. Enhanced Event Quality Metrics

These advanced metrics help distinguish between efficient momentum capture and random drift trades. They measure path efficiency, timing quality, and entry/exit effectiveness.


### Path Efficiency Ratio (PER)

- **Mean PER:** 92.999
- **Median PER:** 0.860

✅ **OK:** Reasonable path efficiency

### Time-to-Outcome Ratio (TTO)

- **Mean TTO:** 0.767
- **Median TTO:** 0.562

⚠️ **Warning:** TTO outside target range [0.4, 0.6]

### MFE/MAE Ratio (Maximum Favorable vs Adverse Excursion)

- **Mean MFE/MAE:** 276.738
- **Median MFE/MAE:** 1.857

✅ **OK:** Favorable excursions exceed adverse excursions

## 9C. Target Volatility Health Check

Verifies that the continuous target values have sufficient variance to train a regression model. If targets are constant or near-zero, the model cannot learn meaningful patterns.

- **Target mean:** 0.000415
- **Target std:** 0.001821
- **Non-zero targets:** 8733 / 135775 (6.4%)

✅ **OK:** Targets have sufficient variance

## 9D. Information Coefficient (IC)

Measures the rank correlation between predicted probabilities and realized returns. This is the purest measure of ranking ability, independent of absolute calibration.

- **Spearman IC (prob, return):** 0.3428
- **P-value:** 0.0000e+00

✅ **OK:** Meaningful rank correlation

## 10. Regime and Time-Conditional Label Checks

These metrics show how label base-rates and realized returns change across volatility regimes, trend states, and time-of-day/weekday. Large differences indicate that learnability and edge are regime-dependent, which is important for downstream model conditioning.


### Label Base-Rate by Volatility Regime

- **Regime high:** positive=91.2%, mean_return=0.99%
- **Regime low:** positive=32.3%, mean_return=-0.12%
- **Regime medium:** positive=49.8%, mean_return=0.13%

### Trend-Conditional (Price vs SMA20)

- **Strong uptrend:** positive=55.0%, mean_return=0.27%
- **Strong downtrend:** positive=57.0%, mean_return=0.31%

### Time-of-Day Positive Rates (Top/Bottom)

- **Top hours by positive rate:**
  - Hour 5: 52.9%
  - Hour 9: 52.6%
  - Hour 6: 52.0%
- **Bottom hours by positive rate:**
  - Hour 23: 44.5%
  - Hour 22: 46.1%
  - Hour 20: 47.6%

### Day-of-Week Positive Rates

- Day 0: 47.5%
- Day 1: 50.9%
- Day 2: 57.3%
- Day 3: 59.7%
- Day 4: 57.2%
- Day 5: 44.3%
- Day 6: 38.1%

## 11. Label–Return Separation, Information Content, and Sample Size

This section quantifies how well the labels separate profitable from unprofitable events (effect size) and how much information they carry about the sign of future returns compared to a random baseline (mutual information and permutation tests).


### Separation Metrics

- **Mean return difference (label=1 - label=0):** 1.83%
- **Cohen's d effect size:** 8.884
- **Approx. required samples for 80% power (heuristic):** 0.2
- **Current labeled samples used in separation:** 36238.0

### Information Content vs Permutation Baseline

- **Mutual information (labels vs realized sign):** 0.6931
- **Baseline MI (mean over permutations):** 0.0000

## 12. Target and Expected-Return Alignment

Here we check whether the continuous targets produced by the isotonic mapping are consistent with realized returns: higher targets should correspond to higher average realized P&L, and most non-zero targets should exceed transaction costs.


### Target vs Realized Return

- **Correlation (target, realized):** 0.470
- **MSE (target vs realized):** 0.000050

### Target/Return by Target Decile

- Decile 0: target=0.0006, realized=0.0020
- Decile 1: target=0.0017, realized=0.0032
- Decile 2: target=0.0028, realized=0.0043
- Decile 3: target=0.0054, realized=0.0069
- Decile 4: target=0.0084, realized=0.0099
- Decile 5: target=0.0092, realized=0.0107
- Decile 6: target=0.0096, realized=0.0113

### Target Distribution Sanity

- **Non-zero target fraction (all samples):** 6.4%
- **Mean non-zero target:** 0.0064
- **Median non-zero target:** 0.0089
- **Fraction of targets below transaction cost (0.150%):** 15.6%

## 13. Cost-Aware Metrics and Threshold P&L

These diagnostics evaluate the meta-model from a trading perspective: global AUC/Brier/PR-AUC, and how mean and cumulative returns evolve as you raise the probability threshold used to filter trades. This helps choose probability cutoffs that are profitable after costs.


### Cost-Aware Classification Metrics (OOF)

- **AUC:** 0.756
- **Brier score:** 0.1953
- **Average precision (PR-AUC):** 0.787

### Threshold-Sweep P&L (Using Meta Probability)


| Threshold | Trades | Mean Return | Cum Return | Sharpe (per trade) |
|----------|--------|-------------|------------|---------------------|
| 0.50 | 18411 | 0.53% | 9721.87% | 77.648 |
| 0.51 | 13540 | 0.55% | 7401.40% | 68.947 |
| 0.52 | 12988 | 0.57% | 7370.40% | 70.665 |
| 0.53 | 12531 | 0.58% | 7326.78% | 72.038 |
| 0.54 | 12023 | 0.61% | 7334.71% | 74.564 |
| 0.55 | 11547 | 0.63% | 7314.33% | 76.817 |
| 0.56 | 11145 | 0.65% | 7299.52% | 78.903 |
| 0.57 | 10702 | 0.68% | 7253.25% | 81.097 |
| 0.58 | 10345 | 0.70% | 7194.56% | 82.774 |
| 0.59 | 9986 | 0.71% | 7128.07% | 84.558 |
| 0.60 | 9634 | 0.74% | 7109.64% | 87.506 |
| 0.61 | 9350 | 0.76% | 7066.05% | 89.659 |
| 0.62 | 9054 | 0.78% | 7027.05% | 92.429 |
| 0.63 | 8808 | 0.80% | 7016.82% | 95.565 |
| 0.64 | 8517 | 0.82% | 6960.61% | 98.566 |
| 0.65 | 8250 | 0.84% | 6910.84% | 101.956 |

## 14. Stability and Bootstrap Statistics

Finally, we assess how stable the labeling and model performance are across time (years and time-series folds) and how sensitive key metrics such as positive rate and label–return separation are under bootstrap resampling. Tight confidence intervals indicate robust signal rather than fragile noise.


### Per-Year Label and Return Stability

- **Year 2021:** positive=68.1%, mean_return=0.49%, AUC=0.500
- **Year 2022:** positive=63.8%, mean_return=0.44%, AUC=0.711
- **Year 2023:** positive=37.6%, mean_return=-0.05%, AUC=0.694
- **Year 2024:** positive=47.2%, mean_return=0.13%, AUC=0.762
- **Year 2025:** positive=58.9%, mean_return=0.35%, AUC=0.843

### Time-Series Fold Stability (Approximate)

- **Fold 1:** positive=55.5%, mean_return=0.28%, AUC=0.812
- **Fold 2:** positive=34.7%, mean_return=-0.10%, AUC=0.690
- **Fold 3:** positive=44.6%, mean_return=0.08%, AUC=0.747
- **Fold 4:** positive=49.5%, mean_return=0.17%, AUC=0.769
- **Fold 5:** positive=58.8%, mean_return=0.35%, AUC=0.840

### Bootstrap Label Statistics (Labeled Events)

- **Positive rate 95% CI:** [49.5%, 50.5%]
- **Mean return diff (label=1 - label=0) 95% CI:** [1.83%, 1.84%]

---


## Summary and Recommendations


### Key Findings:

1. Label balance: 50.0% positive
2. Signal coverage: 26.7%
3. Mean return (label=1): 1.10%
4. Mean return (label=0): -0.74%
5. Calibration error: 0.055