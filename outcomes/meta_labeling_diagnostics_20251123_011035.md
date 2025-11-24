# Meta-Labeling Diagnostics Report

**Generated:** 2025-11-23 01:10:35

**Symbol:** ETHUSDT
**Timeframe:** 15m
**Horizon:** N/A bars

---


## 1. Label Distribution Analysis

- **Total labeled events:** 31974
- **Positive labels (profitable):** 15987 (50.0%)
- **Negative labels (unprofitable):** 15987 (50.0%)

✅ **OK:** Reasonable label balance (50.0%)

### Label Distribution Over Time


⚠️ Index is not datetime, skipping time-series analysis

## 2. Signal Coverage and Sparsity

- **Total samples:** 173439
- **Labeled samples:** 31974
- **Coverage:** 18.4%

✅ **OK:** Reasonable signal coverage (18.4%)

## 3. Feature-Label Correlation Analysis (Post-Filter)


### Top 10 Most Correlated Features (Post-Filter):


### Correlation Health Check (Post-Filter):


## 3B. Pre- vs Post-Filter Comparison and Feature Shift


### Sample Counts and Retention

- **Pre-filter events (realized_return not NaN):** 56502
- **Pre-filter positive/negative (raw economic):** 15371 / 41131
- **Post-filter labeled events:** 31974
- **Post-filter positive/negative (binary_labels):** 15987 / 15987
- **Total retention (post / pre):** 56.6%
- **Positive retention:** 104.0%
- **Negative retention:** 38.9%

### Pre- vs Post-Filter Signal Quality

- **Pre-filter mean return (label=1/0):** 1.63% / -0.76%
- **Post-filter mean return (label=1/0):** 1.48% / -0.80%
- **Pre-filter Cohen's d (label=1 vs 0):** 6.106
- **Post-filter Cohen's d (label=1 vs 0):** 3.490
- **Pre-filter SNR (mean/std, label=1):** 2.443
- **Post-filter SNR (mean/std, label=1):** 1.613

⚠️ **Warning:** Post-filter effect size is materially worse than pre-filter – filters may be discarding informative events

### Largest Feature Correlation Shifts (|post|-|pre|)


## 4. P&L Distribution Per Label

### Label = 1 (Profitable Signals):

- **Count:** 15987
- **Mean return:** 1.48%
- **Median return:** 1.78%
- **Std return:** 0.92%
- **% Actually positive:** 90.9%

### Label = 0 (Unprofitable Signals):

- **Count:** 15987
- **Mean return:** -0.80%
- **Median return:** -0.81%
- **Std return:** 0.12%
- **% Actually positive:** 0.0%

### Labeling Quality:

- **Label overlap:** 4.5%

✅ **OK:** Acceptable label overlap (4.5%)

### Cost-Aware Event Quality Summary

- **Transaction cost (per event, approx):** 0.150%
- **Unconditional mean event return:** 0.34%
- **Mean return (label=1) minus cost:** 1.33%
- **Fraction of labeled events with |return| < cost:** 0.0%

## 5. Time-Series Stability and Regime Analysis


⚠️ Index is not datetime, skipping time-series analysis

## 6. Model Probability Diagnostics


### Calibration (Predicted Probability vs Actual Success Rate):


| Predicted Prob | Actual Success Rate | Count |
|---------------|---------------------|-------|
| 0.099 | 0.077 | 650 |
| 0.177 | 0.158 | 2267 |
| 0.274 | 0.236 | 2675 |
| 0.371 | 0.301 | 4130 |
| 0.484 | 0.425 | 10032 |
| 0.556 | 0.380 | 3267 |
| 0.651 | 0.490 | 1214 |
| 0.753 | 0.753 | 493 |
| 0.864 | 0.977 | 653 |
| 0.977 | 1.000 | 6593 |

- **Mean calibration error:** 0.068

✅ **OK:** Reasonable calibration

### Monotonicity and Slope Checks (Prob → Realized Return)

- **Monotonicity violations (adjacent bins):** 2 / 9
- **Approx. slope in high-probability region:** 0.040134

### Probability Distribution:

- **Mean probability:** 0.509
- **Median probability:** 0.500
- **Std probability:** 0.113

## 7. Feature Importance Analysis


### Top 20 Features by Importance:

1. **volatility_1d:** 0.1915
2. **vol_range_interaction:** 0.0823
3. **dist_from_recent_high_20:** 0.0705
4. **high_dist_x_vol:** 0.0547
5. **event_mean_return_last_50:** 0.0469
6. **vol_momentum_interaction:** 0.0348
7. **momentum_20_x_regime_medium:** 0.0270
8. **momentum_per_vol:** 0.0268
9. **momentum_20:** 0.0257
10. **volatility_5:** 0.0251
11. **atr_momentum:** 0.0248
12. **event_tto_mean_last_50:** 0.0244
13. **momentum_5_x_regime_medium:** 0.0228
14. **dist_from_recent_high_10:** 0.0225
15. **bars_since_last_event:** 0.0223
16. **dist_from_recent_high_50:** 0.0218
17. **event_r_multiple_mean_last_50:** 0.0198
18. **sma_slope:** 0.0193
19. **returns_std_50:** 0.0160
20. **drawdown_100:** 0.0133

### Feature Importance Health Check:


✅ **OK:** No single feature dominates

✅ Kalman-filtered features are being used (6 features)

✅ Volatility-based features in top 10: volatility_1d, vol_range_interaction, high_dist_x_vol, vol_momentum_interaction, momentum_per_vol, volatility_5

## 8. Kalman Smoothed Labels Analysis

- **Mean smoothed label:** 0.496
- **Median smoothed label:** 0.438
- **Std smoothed label:** 0.390
- **Correlation with binary labels:** 0.833
- **Correlation with realized returns:** 0.702

✅ **OK:** Good correlation between smoothed and binary labels

## 9. Event Mechanics and R-Multiples

These diagnostics describe how trades exit (profit, stop, or timeout), how long they stay open, and how large the realized return is relative to the configured stop-loss (R-multiple). They help verify that the triple-barrier / TPSL configuration produces economically meaningful events.


### Exit Reason Mix (Labeled Events)

- **1:** 53.1%
- **0:** 34.2%
- **2:** 12.6%

### Event Duration Distribution (Bars)

- **Mean duration:** 20.70
- **Median duration:** 15.00
- **90th percentile:** 49.00

### R-Multiple Distribution (Return / Stop Threshold)

- **R-multiple (label=1) mean:** 2.22
- **R-multiple (label=1) median:** 2.78
- **R-multiple (label=0) mean:** -1.23
- **R-multiple (label=0) median:** -1.23

## 9B. Enhanced Event Quality Metrics

These advanced metrics help distinguish between efficient momentum capture and random drift trades. They measure path efficiency, timing quality, and entry/exit effectiveness.


### Path Efficiency Ratio (PER)

- **Mean PER (clipped 99.5%):** 0.630
- **Median PER:** 0.682
- **90th percentile PER:** 0.836
- **99th percentile PER:** 0.907

✅ **OK:** Good path efficiency on profitable events

### Time-to-Outcome Ratio (TTO)

- **Mean TTO:** 1.294
- **Median TTO:** 0.938

⚠️ **Alert:** Mean TTO > 0.9 confirms excessive timeouts (not hitting barriers)

### MFE/MAE Ratio (Maximum Favorable vs Adverse Excursion)

- **Mean MFE/MAE:** 429.954
- **Median MFE/MAE:** 1.452

✅ **OK:** Favorable excursions exceed adverse excursions

## 9C. Target Volatility Health Check

Verifies that the continuous target values have sufficient variance to train a regression model. If targets are constant or near-zero, the model cannot learn meaningful patterns.

- **Target mean:** 0.000536
- **Target std:** 0.002397
- **Non-zero targets:** 8859 / 173439 (5.1%)

✅ **OK:** Targets have sufficient variance

## 9D. Information Coefficient (IC)

Measures the rank correlation between predicted probabilities and realized returns. This is the purest measure of ranking ability, independent of absolute calibration.

- **Spearman IC (prob, return):** 0.3417
- **P-value:** 0.0000e+00

✅ **OK:** Meaningful rank correlation

## 10. Regime and Time-Conditional Label Checks

These metrics show how label base-rates and realized returns change across volatility regimes, trend states, and time-of-day/weekday. Large differences indicate that learnability and edge are regime-dependent, which is important for downstream model conditioning.


### Label Base-Rate by Volatility Regime

- **Regime high:** positive=100.0%, mean_return=1.20%
- **Regime low:** positive=28.3%, mean_return=-0.11%
- **Regime medium:** positive=50.9%, mean_return=0.45%

### Trend-Conditional (Price vs SMA20)

- **Strong uptrend:** positive=57.8%, mean_return=0.49%
- **Strong downtrend:** positive=57.5%, mean_return=0.52%

## 11. Label–Return Separation, Information Content, and Sample Size

This section quantifies how well the labels separate profitable from unprofitable events (effect size) and how much information they carry about the sign of future returns compared to a random baseline (mutual information and permutation tests).


### Separation Metrics

- **Mean return difference (label=1 - label=0):** 2.29%
- **Cohen's d effect size:** 3.490
- **Approx. required samples for 80% power (heuristic):** 1.3
- **Current labeled samples used in separation:** 31974.0

### Information Content vs Permutation Baseline

- **Mutual information (labels vs realized sign):** 0.5369
- **Baseline MI (mean over permutations):** 0.0000

## 12. Target and Expected-Return Alignment

Here we check whether the continuous targets produced by the isotonic mapping are consistent with realized returns: higher targets should correspond to higher average realized P&L, and most non-zero targets should exceed transaction costs.


⚠️ Could not compute target/return alignment diagnostics: '<' not supported between instances of 'int' and 'Timestamp'

## 13. Cost-Aware Metrics and Threshold P&L

These diagnostics evaluate the meta-model from a trading perspective: global AUC/Brier/PR-AUC, and how mean and cumulative returns evolve as you raise the probability threshold used to filter trades. This helps choose probability cutoffs that are profitable after costs.


### Cost-Aware Classification Metrics (OOF)

- **AUC:** 0.800
- **Brier score:** 0.1723
- **Average precision (PR-AUC):** 0.836

### Volatility Regime Breakdown

| Regime | Samples | Pos Rate | AUC | Mean Ret | Sharpe |
|--------|---------|----------|-----|----------|--------|
| Low | 14871 | 28.3% | 0.604 | -0.11% | -0.09 |
| Medium | 10842 | 50.9% | 0.744 | 0.45% | 0.33 |
| High | 6261 | 100.0% | 0.928 | 1.20% | 0.97 |

⚠️ **Warning:** Large win-rate disparity between regimes (low: 28.3%, high: 100.0%). Performance is highly regime-dependent.

### Threshold-Sweep P&L (Using Meta Probability)


| Threshold | Trades | Mean Return | Cum Return | Sharpe (per trade) |
|----------|--------|-------------|------------|---------------------|
| 0.50 | 18233 | 0.71% | 13032.94% | 72.130 |
| 0.51 | 12398 | 0.89% | 11043.93% | 75.766 |
| 0.52 | 11955 | 0.92% | 11016.67% | 77.409 |
| 0.53 | 11539 | 0.95% | 10938.77% | 78.710 |
| 0.54 | 11128 | 0.98% | 10871.05% | 80.153 |
| 0.55 | 10733 | 1.01% | 10841.81% | 82.091 |
| 0.56 | 10327 | 1.04% | 10780.74% | 83.984 |
| 0.57 | 9969 | 1.08% | 10757.13% | 86.216 |
| 0.58 | 9665 | 1.11% | 10716.72% | 88.040 |
| 0.59 | 9425 | 1.13% | 10665.36% | 89.468 |
| 0.60 | 9204 | 1.16% | 10643.00% | 91.258 |
| 0.61 | 8988 | 1.18% | 10603.78% | 92.879 |
| 0.62 | 8791 | 1.20% | 10590.87% | 94.839 |
| 0.63 | 8586 | 1.22% | 10487.12% | 95.753 |
| 0.64 | 8438 | 1.23% | 10419.29% | 96.568 |
| 0.65 | 8301 | 1.25% | 10371.81% | 97.673 |

## 14. Stability and Bootstrap Statistics

Finally, we assess how stable the labeling and model performance are across time (years and time-series folds) and how sensitive key metrics such as positive rate and label–return separation are under bootstrap resampling. Tight confidence intervals indicate robust signal rather than fragile noise.


### Time-Series Fold Stability (Approximate)

- **Fold 1:** positive=41.3%, mean_return=0.11%, AUC=0.748
- **Fold 2:** positive=36.9%, mean_return=0.03%, AUC=0.767
- **Fold 3:** positive=80.0%, mean_return=0.97%, AUC=0.945
- **Fold 4:** positive=53.0%, mean_return=0.45%, AUC=0.844
- **Fold 5:** positive=54.1%, mean_return=0.43%, AUC=0.859

### Bootstrap Label Statistics (Labeled Events)

- **Positive rate 95% CI:** [49.4%, 50.5%]
- **Mean return diff (label=1 - label=0) 95% CI:** [2.27%, 2.30%]

---


## Summary and Recommendations


### Key Findings:

1. Label balance: 50.0% positive
2. Signal coverage: 18.4%
3. Mean return (label=1): 1.48%
4. Mean return (label=0): -0.80%
5. Calibration error: 0.068