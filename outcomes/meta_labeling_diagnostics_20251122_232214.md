# Meta-Labeling Diagnostics Report

**Generated:** 2025-11-22 23:22:14

**Symbol:** ETHUSDT
**Timeframe:** 15m
**Horizon:** N/A bars

---


## 1. Label Distribution Analysis

- **Total labeled events:** 31972
- **Positive labels (profitable):** 15986 (50.0%)
- **Negative labels (unprofitable):** 15986 (50.0%)

✅ **OK:** Reasonable label balance (50.0%)

### Label Distribution Over Time


⚠️ Index is not datetime, skipping time-series analysis

## 2. Signal Coverage and Sparsity

- **Total samples:** 173439
- **Labeled samples:** 31972
- **Coverage:** 18.4%

✅ **OK:** Reasonable signal coverage (18.4%)

## 3. Feature-Label Correlation Analysis (Post-Filter)


### Top 10 Most Correlated Features (Post-Filter):


### Correlation Health Check (Post-Filter):


## 3B. Pre- vs Post-Filter Comparison and Feature Shift


### Sample Counts and Retention

- **Pre-filter events (realized_return not NaN):** 56502
- **Pre-filter positive/negative (raw economic):** 15307 / 41195
- **Post-filter labeled events:** 31972
- **Post-filter positive/negative (binary_labels):** 15986 / 15986
- **Total retention (post / pre):** 56.6%
- **Positive retention:** 104.4%
- **Negative retention:** 38.8%

### Pre- vs Post-Filter Signal Quality

- **Pre-filter mean return (label=1/0):** 1.63% / -0.76%
- **Post-filter mean return (label=1/0):** 1.48% / -0.80%
- **Pre-filter Cohen's d (label=1 vs 0):** 6.048
- **Post-filter Cohen's d (label=1 vs 0):** 3.409
- **Pre-filter SNR (mean/std, label=1):** 2.419
- **Post-filter SNR (mean/std, label=1):** 1.577

⚠️ **Warning:** Post-filter effect size is materially worse than pre-filter – filters may be discarding informative events

### Largest Feature Correlation Shifts (|post|-|pre|)


## 4. P&L Distribution Per Label

### Label = 1 (Profitable Signals):

- **Count:** 15986
- **Mean return:** 1.48%
- **Median return:** 1.78%
- **Std return:** 0.94%
- **% Actually positive:** 90.5%Any

### Label = 0 (Unprofitable Signals):

- **Count:** 15986
- **Mean return:** -0.80%
- **Median return:** -0.80%
- **Std return:** 0.12%
- **% Actually positive:** 0.0%

### Labeling Quality:

- **Label overlap:** 4.7%

✅ **OK:** Acceptable label overlap (4.7%)

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
| 0.105 | 0.062 | 694 |
| 0.183 | 0.171 | 2277 |
| 0.281 | 0.230 | 2714 |
| 0.376 | 0.303 | 4297 |
| 0.486 | 0.426 | 9986 |
| 0.560 | 0.381 | 3113 |
| 0.653 | 0.515 | 1135 |
| 0.756 | 0.727 | 521 |
| 0.864 | 0.979 | 610 |
| 0.977 | 1.000 | 6625 |

- **Mean calibration error:** 0.072

✅ **OK:** Reasonable calibration

### Monotonicity and Slope Checks (Prob → Realized Return)

- **Monotonicity violations (adjacent bins):** 2 / 9
- **Approx. slope in high-probability region:** 0.038826

### Probability Distribution:

- **Mean probability:** 0.509
- **Median probability:** 0.500
- **Std probability:** 0.113

## 7. Feature Importance Analysis


### Top 20 Features by Importance:

1. **volatility_1d:** 0.1856
2. **vol_range_interaction:** 0.0824
3. **dist_from_recent_high_20:** 0.0723
4. **high_dist_x_vol:** 0.0531
5. **event_mean_return_last_50:** 0.0490
6. **vol_momentum_interaction:** 0.0327
7. **momentum_20_x_regime_medium:** 0.0295
8. **momentum_20:** 0.0273
9. **volatility_5:** 0.0264
10. **momentum_per_vol:** 0.0244
11. **sma_slope:** 0.0239
12. **dist_from_recent_high_50:** 0.0238
13. **dist_from_recent_high_10:** 0.0229
14. **atr_momentum:** 0.0227
15. **event_tto_mean_last_50:** 0.0226
16. **momentum_5_x_regime_medium:** 0.0226
17. **bars_since_last_event:** 0.0204
18. **returns_std_50:** 0.0175
19. **event_r_multiple_mean_last_50:** 0.0163
20. **drawdown_100:** 0.0137

### Feature Importance Health Check:


✅ **OK:** No single feature dominates

✅ Kalman-filtered features are being used (6 features)

✅ Volatility-based features in top 10: volatility_1d, vol_range_interaction, high_dist_x_vol, vol_momentum_interaction, volatility_5, momentum_per_vol

## 8. Kalman Smoothed Labels Analysis

- **Mean smoothed label:** 0.499
- **Median smoothed label:** 0.499
- **Std smoothed label:** 0.467
- **Correlation with binary labels:** 0.981
- **Correlation with realized returns:** 0.842

✅ **OK:** Good correlation between smoothed and binary labels

## 9. Event Mechanics and R-Multiples

These diagnostics describe how trades exit (profit, stop, or timeout), how long they stay open, and how large the realized return is relative to the configured stop-loss (R-multiple). They help verify that the triple-barrier / TPSL configuration produces economically meaningful events.


### Exit Reason Mix (Labeled Events)

- **1:** 53.3%
- **0:** 33.9%
- **2:** 12.8%

### Event Duration Distribution (Bars)

- **Mean duration:** 20.62
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

- **Mean PER:** 91.857
- **Median PER:** 0.899

✅ **OK:** Reasonable path efficiency

### Time-to-Outcome Ratio (TTO)

- **Mean TTO:** 1.289
- **Median TTO:** 0.938

⚠️ **Alert:** Mean TTO > 0.9 confirms excessive timeouts (not hitting barriers)

### MFE/MAE Ratio (Maximum Favorable vs Adverse Excursion)

- **Mean MFE/MAE:** 430.481
- **Median MFE/MAE:** 1.439

✅ **OK:** Favorable excursions exceed adverse excursions

## 9C. Target Volatility Health Check

Verifies that the continuous target values have sufficient variance to train a regression model. If targets are constant or near-zero, the model cannot learn meaningful patterns.

- **Target mean:** 0.000533
- **Target std:** 0.002356
- **Non-zero targets:** 9541 / 173439 (5.5%)

✅ **OK:** Targets have sufficient variance

## 9D. Information Coefficient (IC)

Measures the rank correlation between predicted probabilities and realized returns. This is the purest measure of ranking ability, independent of absolute calibration.

- **Spearman IC (prob, return):** 0.3342
- **P-value:** 0.0000e+00

✅ **OK:** Meaningful rank correlation

## 10. Regime and Time-Conditional Label Checks

These metrics show how label base-rates and realized returns change across volatility regimes, trend states, and time-of-day/weekday. Large differences indicate that learnability and edge are regime-dependent, which is important for downstream model conditioning.


### Label Base-Rate by Volatility Regime

- **Regime high:** positive=100.0%, mean_return=1.18%
- **Regime low:** positive=28.2%, mean_return=-0.10%
- **Regime medium:** positive=50.8%, mean_return=0.45%

### Trend-Conditional (Price vs SMA20)

- **Strong uptrend:** positive=57.7%, mean_return=0.48%
- **Strong downtrend:** positive=57.7%, mean_return=0.52%

## 11. Label–Return Separation, Information Content, and Sample Size

This section quantifies how well the labels separate profitable from unprofitable events (effect size) and how much information they carry about the sign of future returns compared to a random baseline (mutual information and permutation tests).


### Separation Metrics

- **Mean return difference (label=1 - label=0):** 2.28%
- **Cohen's d effect size:** 3.409
- **Approx. required samples for 80% power (heuristic):** 1.4
- **Current labeled samples used in separation:** 31972.0

### Information Content vs Permutation Baseline

- **Mutual information (labels vs realized sign):** 0.5320
- **Baseline MI (mean over permutations):** 0.0000

## 12. Target and Expected-Return Alignment

Here we check whether the continuous targets produced by the isotonic mapping are consistent with realized returns: higher targets should correspond to higher average realized P&L, and most non-zero targets should exceed transaction costs.


⚠️ Could not compute target/return alignment diagnostics: '<' not supported between instances of 'int' and 'Timestamp'

## 13. Cost-Aware Metrics and Threshold P&L

These diagnostics evaluate the meta-model from a trading perspective: global AUC/Brier/PR-AUC, and how mean and cumulative returns evolve as you raise the probability threshold used to filter trades. This helps choose probability cutoffs that are profitable after costs.


### Cost-Aware Classification Metrics (OOF)

- **AUC:** 0.801
- **Brier score:** 0.1722
- **Average precision (PR-AUC):** 0.837

### Volatility Regime Breakdown

| Regime | Samples | Pos Rate | AUC | Mean Ret | Sharpe |
|--------|---------|----------|-----|----------|--------|
| Low | 14839 | 28.2% | 0.605 | -0.10% | -0.09 |
| Medium | 10828 | 50.8% | 0.742 | 0.45% | 0.33 |
| High | 6305 | 100.0% | 0.928 | 1.18% | 0.94 |

⚠️ **Warning:** Large win-rate disparity between regimes (low: 28.2%, high: 100.0%). Performance is highly regime-dependent.

### Threshold-Sweep P&L (Using Meta Probability)


| Threshold | Trades | Mean Return | Cum Return | Sharpe (per trade) |
|----------|--------|-------------|------------|---------------------|
| 0.50 | 18176 | 0.71% | 12939.04% | 71.399 |
| 0.51 | 12378 | 0.89% | 10982.56% | 75.020 |
| 0.52 | 11915 | 0.91% | 10901.73% | 76.277 |
| 0.53 | 11475 | 0.94% | 10834.57% | 77.713 |
| 0.54 | 11095 | 0.98% | 10828.10% | 79.577 |
| 0.55 | 10668 | 1.01% | 10813.16% | 81.788 |
| 0.56 | 10359 | 1.04% | 10772.54% | 83.279 |
| 0.57 | 10037 | 1.07% | 10733.99% | 85.045 |
| 0.58 | 9712 | 1.10% | 10687.93% | 86.915 |
| 0.59 | 9433 | 1.13% | 10644.58% | 88.690 |
| 0.60 | 9180 | 1.15% | 10593.47% | 90.356 |
| 0.61 | 8970 | 1.17% | 10511.63% | 91.343 |
| 0.62 | 8766 | 1.20% | 10480.13% | 93.037 |
| 0.63 | 8599 | 1.21% | 10367.68% | 93.359 |
| 0.64 | 8439 | 1.22% | 10269.48% | 93.823 |
| 0.65 | 8302 | 1.23% | 10201.79% | 94.569 |

## 14. Stability and Bootstrap Statistics

Finally, we assess how stable the labeling and model performance are across time (years and time-series folds) and how sensitive key metrics such as positive rate and label–return separation are under bootstrap resampling. Tight confidence intervals indicate robust signal rather than fragile noise.


### Time-Series Fold Stability (Approximate)

- **Fold 1:** positive=41.2%, mean_return=0.11%, AUC=0.745
- **Fold 2:** positive=36.7%, mean_return=0.04%, AUC=0.771
- **Fold 3:** positive=80.8%, mean_return=0.97%, AUC=0.946
- **Fold 4:** positive=53.1%, mean_return=0.45%, AUC=0.842
- **Fold 5:** positive=54.3%, mean_return=0.44%, AUC=0.862

### Bootstrap Label Statistics (Labeled Events)

- **Positive rate 95% CI:** [49.4%, 50.5%]
- **Mean return diff (label=1 - label=0) 95% CI:** [2.26%, 2.29%]

---


## Summary and Recommendations


### Key Findings:

1. Label balance: 50.0% positive
2. Signal coverage: 18.4%
3. Mean return (label=1): 1.48%
4. Mean return (label=0): -0.80%
5. Calibration error: 0.072