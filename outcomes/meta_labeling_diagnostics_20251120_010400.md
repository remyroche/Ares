# Meta-Labeling Diagnostics Report

**Generated:** 2025-11-20 01:04:00

**Symbol:** ETHUSDT
**Timeframe:** 15m
**Horizon:** N/A bars

---


## 1. Label Distribution Analysis

- **Total labeled events:** 22658
- **Positive labels (profitable):** 11329 (50.0%)
- **Negative labels (unprofitable):** 11329 (50.0%)

✅ **OK:** Reasonable label balance (50.0%)

### Label Distribution Over Time

- **Daily positive rate - Mean:** 8.3%
- **Daily positive rate - Std:** 5.0%
- **Daily positive rate - Min:** 0.0%
- **Daily positive rate - Max:** 26.0%

⚠️ **Warning:** 930 days with <10% positive labels

## 2. Signal Coverage and Sparsity

- **Total samples:** 135775
- **Labeled samples:** 22658
- **Coverage:** 16.7%

✅ **OK:** Reasonable signal coverage (16.7%)

## 3. Feature-Label Correlation Analysis (Post-Filter)


### Top 10 Most Correlated Features (Post-Filter):

- **volatility_1d:** 0.4258
- **event_win_rate_last_50:** 0.4037
- **event_mean_return_last_50:** 0.4036
- **event_r_multiple_mean_last_50:** 0.4029
- **returns_std_50:** 0.3779
- **returns_entropy:** 0.3403
- **atr_ratio:** 0.3321
- **volatility_ema:** 0.3244
- **volatility_20:** 0.3232
- **returns_std_20:** 0.3232

### Correlation Health Check (Post-Filter):


## 3B. Pre- vs Post-Filter Comparison and Feature Shift


### Sample Counts and Retention

- **Pre-filter events (realized_return not NaN):** 44589
- **Pre-filter positive/negative (raw economic):** 14255 / 30334
- **Post-filter labeled events:** 22658
- **Post-filter positive/negative (binary_labels):** 11329 / 11329
- **Total retention (post / pre):** 50.8%
- **Positive retention:** 79.5%
- **Negative retention:** 37.3%

### Pre- vs Post-Filter Signal Quality

- **Pre-filter mean return (label=1/0):** 1.05% / -0.68%
- **Post-filter mean return (label=1/0):** 1.21% / -0.82%
- **Pre-filter Cohen's d (label=1 vs 0):** 4.870
- **Post-filter Cohen's d (label=1 vs 0):** 9.543
- **Pre-filter SNR (mean/std, label=1):** 2.473
- **Post-filter SNR (mean/std, label=1):** 4.370

### Largest Feature Correlation Shifts (|post|-|pre|)

- **volatility_1d:** pre=0.0369, post=0.4258, Δ|corr|=0.3889
- **returns_std_50:** pre=0.0386, post=0.3779, Δ|corr|=0.3393
- **returns_entropy:** pre=0.0461, post=0.3403, Δ|corr|=0.2942
- **atr_ratio:** pre=0.0456, post=0.3321, Δ|corr|=0.2865
- **volatility_ema:** pre=0.0437, post=0.3244, Δ|corr|=0.2807
- **returns_std_20:** pre=0.0431, post=0.3232, Δ|corr|=0.2801
- **volatility_20:** pre=0.0431, post=0.3232, Δ|corr|=0.2801
- **volatility_4h:** pre=0.0421, post=0.3107, Δ|corr|=0.2686
- **close_range_50:** pre=0.0391, post=0.2996, Δ|corr|=0.2605
- **returns_std_10:** pre=0.0398, post=0.2836, Δ|corr|=0.2438

## 4. P&L Distribution Per Label

### Label = 1 (Profitable Signals):

- **Count:** 11329
- **Mean return:** 1.21%
- **Median return:** 1.31%
- **Std return:** 0.28%
- **% Actually positive:** 100.0%

### Label = 0 (Unprofitable Signals):

- **Count:** 11329
- **Mean return:** -0.82%
- **Median return:** -0.79%
- **Std return:** 0.12%
- **% Actually positive:** 0.0%

### Labeling Quality:

- **Label overlap:** 0.0%

✅ **OK:** Acceptable label overlap (0.0%)

### Cost-Aware Event Quality Summary

- **Transaction cost (per event, approx):** 0.150%
- **Unconditional mean event return:** 0.19%
- **Mean return (label=1) minus cost:** 1.06%
- **Fraction of labeled events with |return| < cost:** 0.0%

## 5. Time-Series Stability and Regime Analysis

- **Win rate vs Volatility correlation:** 0.6275
- **Mean return vs Volatility correlation:** 0.1368

⚠️ **Warning:** Strong correlation between win rate and volatility - performance is regime-dependent

### Rolling Daily SNR (Return / Std by Day)

- **Daily SNR mean:** -0.2445
- **Daily SNR std:** 0.5191
- **Daily SNR min/max:** -8.0022 / 1.1317

### Regime Shift Detection:

- **30-day rolling correlation (win rate vs vol) - Mean:** inf
- **30-day rolling correlation (win rate vs vol) - Std:** nan

## 6. Model Probability Diagnostics


### Calibration (Predicted Probability vs Actual Success Rate):


| Predicted Prob | Actual Success Rate | Count |
|---------------|---------------------|-------|
| 0.122 | 0.144 | 277 |
| 0.203 | 0.233 | 1327 |
| 0.295 | 0.293 | 2735 |
| 0.387 | 0.355 | 3970 |
| 0.487 | 0.496 | 6752 |
| 0.570 | 0.451 | 2480 |
| 0.662 | 0.502 | 1238 |
| 0.761 | 0.773 | 740 |
| 0.863 | 0.971 | 1049 |
| 0.957 | 0.999 | 2090 |

- **Mean calibration error:** 0.054

✅ **OK:** Reasonable calibration

### Monotonicity and Slope Checks (Prob → Realized Return)

- **Monotonicity violations (adjacent bins):** 1 / 9
- **Approx. slope in high-probability region:** 0.044817

### Probability Distribution:

- **Mean probability:** 0.502
- **Median probability:** 0.500
- **Std probability:** 0.086

## 7. Feature Importance Analysis


### Top 20 Features by Importance:

1. **volatility_1d:** 0.1464
2. **high_dist_x_vol:** 0.0731
3. **event_mean_return_last_50:** 0.0650
4. **dist_from_recent_high_20:** 0.0640
5. **vol_range_interaction:** 0.0604
6. **bars_since_last_event:** 0.0514
7. **momentum_5_x_regime_medium:** 0.0394
8. **vol_momentum_interaction:** 0.0382
9. **dist_from_recent_high_50:** 0.0322
10. **sma_slope:** 0.0299
11. **momentum_20:** 0.0206
12. **volatility_5:** 0.0177
13. **dist_from_recent_high_10:** 0.0159
14. **momentum_per_vol:** 0.0147
15. **volatility_4h:** 0.0145
16. **vol_ratio:** 0.0144
17. **returns_std_50:** 0.0136
18. **event_tto_mean_last_50:** 0.0115
19. **atr_momentum:** 0.0113
20. **drawdown_100:** 0.0104

### Feature Importance Health Check:


✅ **OK:** No single feature dominates

✅ Kalman-filtered features are being used (6 features)

✅ Volatility-based features in top 10: volatility_1d, high_dist_x_vol, vol_range_interaction, vol_momentum_interaction

## 8. Kalman Smoothed Labels Analysis

- **Mean smoothed label:** 0.495
- **Median smoothed label:** 0.481
- **Std smoothed label:** 0.428
- **Correlation with binary labels:** 0.936
- **Correlation with realized returns:** 0.926

✅ **OK:** Good correlation between smoothed and binary labels

## 9. Event Mechanics and R-Multiples

These diagnostics describe how trades exit (profit, stop, or timeout), how long they stay open, and how large the realized return is relative to the configured stop-loss (R-multiple). They help verify that the triple-barrier / TPSL configuration produces economically meaningful events.


### Exit Reason Mix (Labeled Events)

- **stop:** 47.0%
- **profit:** 41.0%
- **timeout:** 12.0%

### Event Duration Distribution (Bars)

- **Mean duration:** 12.60
- **Median duration:** 10.00
- **90th percentile:** 27.00

### R-Multiple Distribution (Return / Stop Threshold)

- **R-multiple (label=1) mean:** 1.72
- **R-multiple (label=1) median:** 1.87
- **R-multiple (label=0) mean:** -1.19
- **R-multiple (label=0) median:** -1.24

## 9B. Enhanced Event Quality Metrics

These advanced metrics help distinguish between efficient momentum capture and random drift trades. They measure path efficiency, timing quality, and entry/exit effectiveness.


### Path Efficiency Ratio (PER)

- **Mean PER:** 108.772
- **Median PER:** 0.879

✅ **OK:** Reasonable path efficiency

### Time-to-Outcome Ratio (TTO)

- **Mean TTO:** 0.787
- **Median TTO:** 0.625

⚠️ **Warning:** TTO outside target range [0.4, 0.6]

### MFE/MAE Ratio (Maximum Favorable vs Adverse Excursion)

- **Mean MFE/MAE:** 288.605
- **Median MFE/MAE:** 1.691

✅ **OK:** Favorable excursions exceed adverse excursions

## 9C. Target Volatility Health Check

Verifies that the continuous target values have sufficient variance to train a regression model. If targets are constant or near-zero, the model cannot learn meaningful patterns.

- **Target mean:** 0.000287
- **Target std:** 0.001679
- **Non-zero targets:** 4610 / 135775 (3.4%)

✅ **OK:** Targets have sufficient variance

## 9D. Information Coefficient (IC)

Measures the rank correlation between predicted probabilities and realized returns. This is the purest measure of ranking ability, independent of absolute calibration.

- **Spearman IC (prob, return):** 0.3439
- **P-value:** 0.0000e+00

✅ **OK:** Meaningful rank correlation

## 10. Regime and Time-Conditional Label Checks

These metrics show how label base-rates and realized returns change across volatility regimes, trend states, and time-of-day/weekday. Large differences indicate that learnability and edge are regime-dependent, which is important for downstream model conditioning.


### Label Base-Rate by Volatility Regime

- **Regime high:** positive=99.0%, mean_return=1.26%
- **Regime low:** positive=34.3%, mean_return=-0.14%
- **Regime medium:** positive=45.4%, mean_return=0.08%

### Trend-Conditional (Price vs SMA20)

- **Strong uptrend:** positive=55.0%, mean_return=0.29%
- **Strong downtrend:** positive=56.5%, mean_return=0.35%

### Time-of-Day Positive Rates (Top/Bottom)

- **Top hours by positive rate:**
  - Hour 4: 52.8%
  - Hour 13: 52.7%
  - Hour 15: 52.0%
- **Bottom hours by positive rate:**
  - Hour 17: 47.0%
  - Hour 23: 47.1%
  - Hour 0: 47.6%

### Day-of-Week Positive Rates

- Day 0: 49.7%
- Day 1: 54.2%
- Day 2: 57.7%
- Day 3: 57.1%
- Day 4: 53.4%
- Day 5: 42.1%
- Day 6: 39.1%

## 11. Label–Return Separation, Information Content, and Sample Size

This section quantifies how well the labels separate profitable from unprofitable events (effect size) and how much information they carry about the sign of future returns compared to a random baseline (mutual information and permutation tests).


### Separation Metrics

- **Mean return difference (label=1 - label=0):** 2.03%
- **Cohen's d effect size:** 9.543
- **Approx. required samples for 80% power (heuristic):** 0.2
- **Current labeled samples used in separation:** 22658.0

### Information Content vs Permutation Baseline

- **Mutual information (labels vs realized sign):** 0.6931
- **Baseline MI (mean over permutations):** 0.0000

## 12. Target and Expected-Return Alignment

Here we check whether the continuous targets produced by the isotonic mapping are consistent with realized returns: higher targets should correspond to higher average realized P&L, and most non-zero targets should exceed transaction costs.


### Target vs Realized Return

- **Correlation (target, realized):** 0.541
- **MSE (target vs realized):** 0.000040

### Target/Return by Target Decile

- Decile 0: target=0.0011, realized=0.0026
- Decile 1: target=0.0035, realized=0.0050
- Decile 2: target=0.0071, realized=0.0086
- Decile 3: target=0.0102, realized=0.0117
- Decile 4: target=0.0109, realized=0.0129

### Target Distribution Sanity

- **Non-zero target fraction (all samples):** 3.4%
- **Mean non-zero target:** 0.0085
- **Median non-zero target:** 0.0109
- **Fraction of targets below transaction cost (0.150%):** 15.8%

## 13. Cost-Aware Metrics and Threshold P&L

These diagnostics evaluate the meta-model from a trading perspective: global AUC/Brier/PR-AUC, and how mean and cumulative returns evolve as you raise the probability threshold used to filter trades. This helps choose probability cutoffs that are profitable after costs.


### Cost-Aware Classification Metrics (OOF)

- **AUC:** 0.730
- **Brier score:** 0.2021
- **Average precision (PR-AUC):** 0.772

### Threshold-Sweep P&L (Using Meta Probability)


| Threshold | Trades | Mean Return | Cum Return | Sharpe (per trade) |
|----------|--------|-------------|------------|---------------------|
| 0.50 | 11500 | 0.56% | 6479.48% | 59.617 |
| 0.51 | 8214 | 0.59% | 4852.10% | 53.260 |
| 0.52 | 7862 | 0.62% | 4848.64% | 54.866 |
| 0.53 | 7508 | 0.65% | 4882.83% | 57.298 |
| 0.54 | 7155 | 0.68% | 4862.67% | 59.198 |
| 0.55 | 6836 | 0.71% | 4866.60% | 61.563 |
| 0.56 | 6546 | 0.74% | 4856.63% | 63.830 |
| 0.57 | 6299 | 0.77% | 4837.64% | 65.758 |
| 0.58 | 6053 | 0.80% | 4828.05% | 68.272 |
| 0.59 | 5829 | 0.83% | 4830.77% | 71.175 |
| 0.60 | 5602 | 0.86% | 4803.14% | 73.821 |
| 0.61 | 5365 | 0.89% | 4752.23% | 76.534 |
| 0.62 | 5145 | 0.92% | 4726.14% | 80.175 |
| 0.63 | 4953 | 0.95% | 4705.26% | 84.243 |
| 0.64 | 4788 | 0.98% | 4686.17% | 88.442 |
| 0.65 | 4620 | 1.02% | 4701.71% | 95.416 |

## 14. Stability and Bootstrap Statistics

Finally, we assess how stable the labeling and model performance are across time (years and time-series folds) and how sensitive key metrics such as positive rate and label–return separation are under bootstrap resampling. Tight confidence intervals indicate robust signal rather than fragile noise.


### Per-Year Label and Return Stability

- **Year 2021:** positive=65.6%, mean_return=0.53%, AUC=0.500
- **Year 2022:** positive=62.3%, mean_return=0.47%, AUC=0.669
- **Year 2023:** positive=38.4%, mean_return=-0.06%, AUC=0.658
- **Year 2024:** positive=47.5%, mean_return=0.13%, AUC=0.742
- **Year 2025:** positive=56.8%, mean_return=0.34%, AUC=0.817

### Time-Series Fold Stability (Approximate)

- **Fold 1:** positive=54.8%, mean_return=0.30%, AUC=0.773
- **Fold 2:** positive=36.6%, mean_return=-0.10%, AUC=0.651
- **Fold 3:** positive=44.4%, mean_return=0.07%, AUC=0.738
- **Fold 4:** positive=49.5%, mean_return=0.18%, AUC=0.751
- **Fold 5:** positive=56.4%, mean_return=0.33%, AUC=0.806

### Bootstrap Label Statistics (Labeled Events)

- **Positive rate 95% CI:** [49.4%, 50.7%]
- **Mean return diff (label=1 - label=0) 95% CI:** [2.03%, 2.04%]

---


## Summary and Recommendations


### Key Findings:

1. Label balance: 50.0% positive
2. Signal coverage: 16.7%
3. Mean return (label=1): 1.21%
4. Mean return (label=0): -0.82%
5. Calibration error: 0.054