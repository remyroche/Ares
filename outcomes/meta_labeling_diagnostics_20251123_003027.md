# Meta-Labeling Diagnostics Report

**Generated:** 2025-11-23 00:30:27

**Symbol:** ETHUSDT
**Timeframe:** 15m
**Horizon:** N/A bars

---


## 1. Label Distribution Analysis

- **Total labeled events:** 31970
- **Positive labels (profitable):** 15985 (50.0%)
- **Negative labels (unprofitable):** 15985 (50.0%)

✅ **OK:** Reasonable label balance (50.0%)

### Label Distribution Over Time


⚠️ Index is not datetime, skipping time-series analysis

## 2. Signal Coverage and Sparsity

- **Total samples:** 173439
- **Labeled samples:** 31970
- **Coverage:** 18.4%

✅ **OK:** Reasonable signal coverage (18.4%)

## 3. Feature-Label Correlation Analysis (Post-Filter)


### Top 10 Most Correlated Features (Post-Filter):


### Correlation Health Check (Post-Filter):


## 3B. Pre- vs Post-Filter Comparison and Feature Shift


### Sample Counts and Retention

- **Pre-filter events (realized_return not NaN):** 56502
- **Pre-filter positive/negative (raw economic):** 15289 / 41213
- **Post-filter labeled events:** 31970
- **Post-filter positive/negative (binary_labels):** 15985 / 15985
- **Total retention (post / pre):** 56.6%
- **Positive retention:** 104.6%
- **Negative retention:** 38.8%

### Pre- vs Post-Filter Signal Quality

- **Pre-filter mean return (label=1/0):** 1.63% / -0.76%
- **Post-filter mean return (label=1/0):** 1.47% / -0.80%
- **Pre-filter Cohen's d (label=1 vs 0):** 6.040
- **Post-filter Cohen's d (label=1 vs 0):** 3.391
- **Pre-filter SNR (mean/std, label=1):** 2.415
- **Post-filter SNR (mean/std, label=1):** 1.568

⚠️ **Warning:** Post-filter effect size is materially worse than pre-filter – filters may be discarding informative events

### Largest Feature Correlation Shifts (|post|-|pre|)


## 4. P&L Distribution Per Label

### Label = 1 (Profitable Signals):

- **Count:** 15985
- **Mean return:** 1.47%
- **Median return:** 1.78%
- **Std return:** 0.94%
- **% Actually positive:** 90.4%

### Label = 0 (Unprofitable Signals):

- **Count:** 15985
- **Mean return:** -0.80%
- **Median return:** -0.80%
- **Std return:** 0.13%
- **% Actually positive:** 0.0%

### Labeling Quality:

- **Label overlap:** 4.8%

✅ **OK:** Acceptable label overlap (4.8%)

### Cost-Aware Event Quality Summary

- **Transaction cost (per event, approx):** 0.150%
- **Unconditional mean event return:** 0.34%
- **Mean return (label=1) minus cost:** 1.32%
- **Fraction of labeled events with |return| < cost:** 0.0%

## 5. Time-Series Stability and Regime Analysis


⚠️ Index is not datetime, skipping time-series analysis

## 6. Model Probability Diagnostics


### Calibration (Predicted Probability vs Actual Success Rate):


| Predicted Prob | Actual Success Rate | Count |
|---------------|---------------------|-------|
| 0.103 | 0.052 | 705 |
| 0.179 | 0.157 | 2178 |
| 0.277 | 0.239 | 2685 |
| 0.374 | 0.296 | 4297 |
| 0.485 | 0.425 | 9996 |
| 0.558 | 0.395 | 3226 |
| 0.653 | 0.517 | 1144 |
| 0.756 | 0.713 | 470 |
| 0.859 | 0.964 | 581 |
| 0.978 | 1.000 | 6688 |

- **Mean calibration error:** 0.072

✅ **OK:** Reasonable calibration

### Monotonicity and Slope Checks (Prob → Realized Return)

- **Monotonicity violations (adjacent bins):** 2 / 9
- **Approx. slope in high-probability region:** 0.037232

### Probability Distribution:

- **Mean probability:** 0.509
- **Median probability:** 0.500
- **Std probability:** 0.113

## 7. Feature Importance Analysis


### Top 20 Features by Importance:

1. **volatility_1d:** 0.1729
2. **vol_range_interaction:** 0.0853
3. **dist_from_recent_high_20:** 0.0706
4. **high_dist_x_vol:** 0.0596
5. **event_mean_return_last_50:** 0.0500
6. **vol_momentum_interaction:** 0.0309
7. **momentum_20_x_regime_medium:** 0.0299
8. **momentum_per_vol:** 0.0271
9. **volatility_5:** 0.0266
10. **momentum_20:** 0.0259
11. **event_tto_mean_last_50:** 0.0246
12. **atr_momentum:** 0.0241
13. **bars_since_last_event:** 0.0235
14. **dist_from_recent_high_50:** 0.0234
15. **momentum_5_x_regime_medium:** 0.0233
16. **dist_from_recent_high_10:** 0.0217
17. **event_r_multiple_mean_last_50:** 0.0190
18. **sma_slope:** 0.0179
19. **returns_std_50:** 0.0173
20. **drawdown_100:** 0.0138

### Feature Importance Health Check:


✅ **OK:** No single feature dominates

✅ Kalman-filtered features are being used (6 features)

✅ Volatility-based features in top 10: volatility_1d, vol_range_interaction, high_dist_x_vol, vol_momentum_interaction, momentum_per_vol, volatility_5

## 8. Kalman Smoothed Labels Analysis

- **Mean smoothed label:** 0.496
- **Median smoothed label:** 0.437
- **Std smoothed label:** 0.391
- **Correlation with binary labels:** 0.834
- **Correlation with realized returns:** 0.695

✅ **OK:** Good correlation between smoothed and binary labels

## 9. Event Mechanics and R-Multiples

These diagnostics describe how trades exit (profit, stop, or timeout), how long they stay open, and how large the realized return is relative to the configured stop-loss (R-multiple). They help verify that the triple-barrier / TPSL configuration produces economically meaningful events.


### Exit Reason Mix (Labeled Events)

- **1:** 53.4%
- **0:** 33.9%
- **2:** 12.8%

### Event Duration Distribution (Bars)

- **Mean duration:** 20.59
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
- **90th percentile PER:** 0.835
- **99th percentile PER:** 0.908

✅ **OK:** Good path efficiency on profitable events

### Time-to-Outcome Ratio (TTO)

- **Mean TTO:** 1.287
- **Median TTO:** 0.938

⚠️ **Alert:** Mean TTO > 0.9 confirms excessive timeouts (not hitting barriers)

### MFE/MAE Ratio (Maximum Favorable vs Adverse Excursion)

- **Mean MFE/MAE:** 430.746
- **Median MFE/MAE:** 1.433

✅ **OK:** Favorable excursions exceed adverse excursions

## 9C. Target Volatility Health Check

Verifies that the continuous target values have sufficient variance to train a regression model. If targets are constant or near-zero, the model cannot learn meaningful patterns.

- **Target mean:** 0.000532
- **Target std:** 0.002332
- **Non-zero targets:** 10291 / 173439 (5.9%)

✅ **OK:** Targets have sufficient variance

## 9D. Information Coefficient (IC)

Measures the rank correlation between predicted probabilities and realized returns. This is the purest measure of ranking ability, independent of absolute calibration.

- **Spearman IC (prob, return):** 0.3354
- **P-value:** 0.0000e+00

✅ **OK:** Meaningful rank correlation

## 10. Regime and Time-Conditional Label Checks

These metrics show how label base-rates and realized returns change across volatility regimes, trend states, and time-of-day/weekday. Large differences indicate that learnability and edge are regime-dependent, which is important for downstream model conditioning.


### Label Base-Rate by Volatility Regime

- **Regime high:** positive=100.0%, mean_return=1.17%
- **Regime low:** positive=28.1%, mean_return=-0.10%
- **Regime medium:** positive=50.8%, mean_return=0.45%

### Trend-Conditional (Price vs SMA20)

- **Strong uptrend:** positive=57.7%, mean_return=0.48%
- **Strong downtrend:** positive=57.7%, mean_return=0.52%

## 11. Label–Return Separation, Information Content, and Sample Size

This section quantifies how well the labels separate profitable from unprofitable events (effect size) and how much information they carry about the sign of future returns compared to a random baseline (mutual information and permutation tests).


### Separation Metrics

- **Mean return difference (label=1 - label=0):** 2.27%
- **Cohen's d effect size:** 3.391
- **Approx. required samples for 80% power (heuristic):** 1.4
- **Current labeled samples used in separation:** 31970.0

### Information Content vs Permutation Baseline

- **Mutual information (labels vs realized sign):** 0.5306
- **Baseline MI (mean over permutations):** 0.0000

## 12. Target and Expected-Return Alignment

Here we check whether the continuous targets produced by the isotonic mapping are consistent with realized returns: higher targets should correspond to higher average realized P&L, and most non-zero targets should exceed transaction costs.


⚠️ Could not compute target/return alignment diagnostics: '<' not supported between instances of 'int' and 'Timestamp'

## 13. Cost-Aware Metrics and Threshold P&L

These diagnostics evaluate the meta-model from a trading perspective: global AUC/Brier/PR-AUC, and how mean and cumulative returns evolve as you raise the probability threshold used to filter trades. This helps choose probability cutoffs that are profitable after costs.


### Cost-Aware Classification Metrics (OOF)

- **AUC:** 0.802
- **Brier score:** 0.1718
- **Average precision (PR-AUC):** 0.837

### Volatility Regime Breakdown

| Regime | Samples | Pos Rate | AUC | Mean Ret | Sharpe |
|--------|---------|----------|-----|----------|--------|
| Low | 14828 | 28.1% | 0.606 | -0.10% | -0.09 |
| Medium | 10827 | 50.8% | 0.745 | 0.45% | 0.33 |
| High | 6315 | 100.0% | 0.928 | 1.17% | 0.93 |

⚠️ **Warning:** Large win-rate disparity between regimes (low: 28.1%, high: 100.0%). Performance is highly regime-dependent.

### Threshold-Sweep P&L (Using Meta Probability)


| Threshold | Trades | Mean Return | Cum Return | Sharpe (per trade) |
|----------|--------|-------------|------------|---------------------|
| 0.50 | 18185 | 0.71% | 12974.65% | 71.576 |
| 0.51 | 12411 | 0.89% | 11015.69% | 75.134 |
| 0.52 | 11945 | 0.92% | 10957.16% | 76.581 |
| 0.53 | 11489 | 0.95% | 10873.71% | 77.889 |
| 0.54 | 11074 | 0.98% | 10805.85% | 79.356 |
| 0.55 | 10685 | 1.01% | 10770.20% | 81.125 |
| 0.56 | 10332 | 1.04% | 10747.45% | 83.049 |
| 0.57 | 9985 | 1.07% | 10680.40% | 84.722 |
| 0.58 | 9673 | 1.10% | 10638.77% | 86.590 |
| 0.59 | 9418 | 1.12% | 10551.58% | 87.682 |
| 0.60 | 9145 | 1.14% | 10464.70% | 89.003 |
| 0.61 | 8938 | 1.17% | 10445.71% | 90.746 |
| 0.62 | 8758 | 1.18% | 10372.98% | 91.564 |
| 0.63 | 8582 | 1.20% | 10305.09% | 92.583 |
| 0.64 | 8437 | 1.21% | 10219.93% | 93.102 |
| 0.65 | 8286 | 1.22% | 10137.11% | 93.786 |

## 14. Stability and Bootstrap Statistics

Finally, we assess how stable the labeling and model performance are across time (years and time-series folds) and how sensitive key metrics such as positive rate and label–return separation are under bootstrap resampling. Tight confidence intervals indicate robust signal rather than fragile noise.


### Time-Series Fold Stability (Approximate)

- **Fold 1:** positive=41.0%, mean_return=0.11%, AUC=0.747
- **Fold 2:** positive=36.7%, mean_return=0.03%, AUC=0.771
- **Fold 3:** positive=81.1%, mean_return=0.97%, AUC=0.947
- **Fold 4:** positive=53.1%, mean_return=0.45%, AUC=0.845
- **Fold 5:** positive=54.3%, mean_return=0.44%, AUC=0.860

### Bootstrap Label Statistics (Labeled Events)

- **Positive rate 95% CI:** [49.4%, 50.5%]
- **Mean return diff (label=1 - label=0) 95% CI:** [2.26%, 2.28%]

---


## Summary and Recommendations


### Key Findings:

1. Label balance: 50.0% positive
2. Signal coverage: 18.4%
3. Mean return (label=1): 1.47%
4. Mean return (label=0): -0.80%
5. Calibration error: 0.072