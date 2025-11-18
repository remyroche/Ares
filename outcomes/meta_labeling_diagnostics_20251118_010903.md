# Meta-Labeling Diagnostics Report

**Generated:** 2025-11-18 01:09:03

**Symbol:** ETHUSDT
**Timeframe:** 15m
**Horizon:** N/A bars

---


## 1. Label Distribution Analysis

- **Total labeled events:** 12587
- **Positive labels (profitable):** 2363 (18.8%)
- **Negative labels (unprofitable):** 10224 (81.2%)

⚠️ **Warning:** Low positive label rate (18.8%) - most signals are unprofitable

### Label Distribution Over Time

- **Daily positive rate - Mean:** 1.7%
- **Daily positive rate - Std:** 2.9%
- **Daily positive rate - Min:** 0.0%
- **Daily positive rate - Max:** 16.7%

⚠️ **Warning:** 1381 days with <10% positive labels

## 2. Signal Coverage and Sparsity

- **Total samples:** 135775
- **Labeled samples:** 12587
- **Coverage:** 9.3%

✅ **OK:** Reasonable signal coverage (9.3%)

## 3. Feature-Label Correlation Analysis


### Top 10 Most Correlated Features:

- **event_win_rate_last_50:** 0.2329
- **event_mean_return_last_50:** 0.1929
- **atr_ratio:** 0.1578
- **returns_entropy:** 0.1513
- **volatility_1d:** 0.1420
- **volatility_ema:** 0.1420
- **volatility_20:** 0.1399
- **volatility_4h:** 0.1372
- **drawdown_100:** -0.1259
- **atr_14:** 0.1182

### Correlation Health Check:


## 4. P&L Distribution Per Label

### Label = 1 (Profitable Signals):

- **Count:** 2363
- **Mean return:** 3.31%
- **Median return:** 3.16%
- **Std return:** 0.99%
- **% Actually positive:** 100.0%

### Label = 0 (Unprofitable Signals):

- **Count:** 10224
- **Mean return:** -1.72%
- **Median return:** -1.36%
- **Std return:** 1.00%
- **% Actually positive:** 0.0%

### Labeling Quality:

- **Label overlap:** 0.0%

✅ **OK:** Acceptable label overlap (0.0%)

### Cost-Aware Event Quality Summary

- **Transaction cost (per event, approx):** 0.150%
- **Unconditional mean event return:** -0.77%
- **Mean return (label=1) minus cost:** 3.16%
- **Fraction of labeled events with |return| < cost:** 0.0%

## 5. Time-Series Stability and Regime Analysis

- **Win rate vs Volatility correlation:** 0.4479
- **Mean return vs Volatility correlation:** 0.1020

✅ **OK:** Win rate not strongly correlated with volatility

### Regime Shift Detection:

- **30-day rolling correlation (win rate vs vol) - Mean:** 0.4282
- **30-day rolling correlation (win rate vs vol) - Std:** 0.1475

## 6. Model Probability Diagnostics


### Calibration (Predicted Probability vs Actual Success Rate):


| Predicted Prob | Actual Success Rate | Count |
|---------------|---------------------|-------|
| 0.085 | 0.056 | 1682 |
| 0.135 | 0.134 | 3441 |
| 0.198 | 0.198 | 2120 |
| 0.261 | 0.247 | 1273 |
| 0.325 | 0.258 | 693 |
| 0.389 | 0.289 | 381 |
| 0.456 | 0.400 | 145 |
| 0.500 | 0.254 | 2812 |
| 0.588 | 0.314 | 35 |
| 0.641 | 0.600 | 5 |

- **Mean calibration error:** 0.083

✅ **OK:** Reasonable calibration

### Monotonicity and Slope Checks (Prob → Realized Return)

- **Monotonicity violations (adjacent bins):** 3 / 9
- **Approx. slope in high-probability region:** 0.043378

### Probability Distribution:

- **Mean probability:** 0.477
- **Median probability:** 0.500
- **Std probability:** 0.085

## 7. Feature Importance Analysis


### Top 20 Features by Importance:

1. **dist_from_recent_high_50:** 0.0834
2. **dist_from_recent_low_50:** 0.0725
3. **volatility_1d:** 0.0338
4. **sma_slope:** 0.0320
5. **volatility_regime:** 0.0273
6. **bars_since_last_event:** 0.0256
7. **momentum_20:** 0.0246
8. **momentum_per_vol:** 0.0245
9. **atr_ratio:** 0.0241
10. **vol_of_vol:** 0.0239
11. **vol_ratio:** 0.0237
12. **returns_entropy:** 0.0226
13. **range_position:** 0.0223
14. **volatility_5:** 0.0221
15. **vol_regime_high:** 0.0217
16. **bars_since_last_signal:** 0.0210
17. **volatility_4h:** 0.0207
18. **kalman_trend:** 0.0202
19. **vol_price_corr:** 0.0200
20. **volume_zscore:** 0.0196

### Feature Importance Health Check:


✅ **OK:** No single feature dominates

✅ Kalman-filtered features are being used (6 features)

✅ Volatility-based features in top 10: volatility_1d, volatility_regime, momentum_per_vol, vol_of_vol

## 8. Kalman Smoothed Labels Analysis

- **Mean smoothed label:** 0.181
- **Median smoothed label:** 0.057
- **Std smoothed label:** 0.239
- **Correlation with binary labels:** 0.756
- **Correlation with realized returns:** 0.652

✅ **OK:** Good correlation between smoothed and binary labels

## 9. Event Mechanics and R-Multiples

These diagnostics describe how trades exit (profit, stop, or timeout), how long they stay open, and how large the realized return is relative to the configured stop-loss (R-multiple). They help verify that the triple-barrier / TPSL configuration produces economically meaningful events.


### Exit Reason Mix (Labeled Events)

- **timeout:** 75.6%
- **profit:** 14.0%
- **stop:** 10.4%

### Event Duration Distribution (Bars)

- **Mean duration:** 14.53
- **Median duration:** 16.00
- **90th percentile:** 16.00

### R-Multiple Distribution (Return / Stop Threshold)

- **R-multiple (label=1) mean:** 1.07
- **R-multiple (label=1) median:** 1.02
- **R-multiple (label=0) mean:** -0.56
- **R-multiple (label=0) median:** -0.44

## 10. Regime and Time-Conditional Label Checks

These metrics show how label base-rates and realized returns change across volatility regimes, trend states, and time-of-day/weekday. Large differences indicate that learnability and edge are regime-dependent, which is important for downstream model conditioning.


### Label Base-Rate by Volatility Regime

- **Regime high:** positive=23.5%, mean_return=-0.70%
- **Regime low:** positive=12.1%, mean_return=-0.87%
- **Regime medium:** positive=15.3%, mean_return=-0.84%

### Trend-Conditional (Price vs SMA20)

- **Strong uptrend:** positive=21.0%, mean_return=-0.74%
- **Strong downtrend:** positive=23.8%, mean_return=-0.58%

### Time-of-Day Positive Rates (Top/Bottom)

- **Top hours by positive rate:**
  - Hour 14: 24.5%
  - Hour 18: 23.1%
  - Hour 12: 22.6%
- **Bottom hours by positive rate:**
  - Hour 3: 13.3%
  - Hour 2: 14.0%
  - Hour 0: 15.1%

### Day-of-Week Positive Rates

- Day 0: 22.8%
- Day 1: 18.4%
- Day 2: 20.2%
- Day 3: 20.5%
- Day 4: 18.6%
- Day 5: 13.2%
- Day 6: 14.9%

## 11. Label–Return Separation and Information Content

This section quantifies how well the labels separate profitable from unprofitable events (effect size) and how much information they carry about the sign of future returns compared to a random baseline (mutual information and permutation tests).


### Separation Metrics

- **Mean return difference (label=1 - label=0):** 5.02%
- **Cohen's d effect size:** 5.045

### Information Content vs Permutation Baseline

- **Mutual information (labels vs realized sign):** 0.4829
- **Baseline MI (mean over permutations):** 0.0000

## 12. Target and Expected-Return Alignment

Here we check whether the continuous targets produced by the isotonic mapping are consistent with realized returns: higher targets should correspond to higher average realized P&L, and most non-zero targets should exceed transaction costs.


### Target vs Realized Return

- **Correlation (target, realized):** nan
- **MSE (target vs realized):** 0.003854

### Target/Return by Target Decile


### Target Distribution Sanity

- **Non-zero target fraction (all samples):** 0.0%
- **Mean non-zero target:** 0.0744
- **Median non-zero target:** 0.0744
- **Fraction of targets below transaction cost (0.150%):** 0.0%

## 13. Cost-Aware Metrics and Threshold P&L

These diagnostics evaluate the meta-model from a trading perspective: global AUC/Brier/PR-AUC, and how mean and cumulative returns evolve as you raise an expected-return target threshold used to filter economically strong label=1 events. This helps choose cutoffs that are profitable after costs.


### Cost-Aware Classification Metrics (OOF)

- **AUC:** 0.634
- **Brier score:** 0.1612
- **Average precision (PR-AUC):** 0.250

### Threshold-Sweep P&L (Using Expected-Return Target, label=1 only)


| Target Mult (x cost) | Trades | Mean Return | Cum Return | Sharpe (per trade) |
|----------------------|--------|-------------|------------|---------------------|
| 1.0x | 3 | 3.75% | 11.24% | 8.169 |
| 1.5x | 3 | 3.75% | 11.24% | 8.169 |
| 2.0x | 3 | 3.75% | 11.24% | 8.169 |
| 2.5x | 3 | 3.75% | 11.24% | 8.169 |
| 3.0x | 3 | 3.75% | 11.24% | 8.169 |

## 14. Stability and Bootstrap Statistics

Finally, we assess how stable the labeling and model performance are across time (years and time-series folds) and how sensitive key metrics such as positive rate and label–return separation are under bootstrap resampling. Tight confidence intervals indicate robust signal rather than fragile noise.


### Per-Year Label and Return Stability

- **Year 2021:** positive=20.8%, mean_return=-0.73%, AUC=0.500
- **Year 2022:** positive=24.0%, mean_return=-0.69%, AUC=0.555
- **Year 2023:** positive=12.6%, mean_return=-0.85%, AUC=0.599
- **Year 2024:** positive=14.9%, mean_return=-0.89%, AUC=0.688
- **Year 2025:** positive=20.1%, mean_return=-0.71%, AUC=0.665

### Time-Series Fold Stability (Approximate)

- **Fold 1:** positive=20.2%, mean_return=-0.81%, AUC=0.566
- **Fold 2:** positive=12.8%, mean_return=-0.85%, AUC=0.647
- **Fold 3:** positive=13.9%, mean_return=-0.90%, AUC=0.706
- **Fold 4:** positive=15.7%, mean_return=-0.86%, AUC=0.680
- **Fold 5:** positive=19.7%, mean_return=-0.71%, AUC=0.662

### Bootstrap Label Statistics (Labeled Events)

- **Positive rate 95% CI:** [18.1%, 19.4%]
- **Mean return diff (label=1 - label=0) 95% CI:** [4.98%, 5.06%]

---


## Summary and Recommendations


### Key Findings:

1. Label balance: 18.8% positive
2. Signal coverage: 9.3%
3. Mean return (label=1): 3.31%
4. Mean return (label=0): -1.72%
5. Calibration error: 0.083