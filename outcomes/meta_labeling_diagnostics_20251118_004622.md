# Meta-Labeling Diagnostics Report

**Generated:** 2025-11-18 00:46:22

**Symbol:** ETHUSDT
**Timeframe:** 15m
**Horizon:** N/A bars

---


## 1. Label Distribution Analysis

- **Total labeled events:** 12636
- **Positive labels (profitable):** 2160 (17.1%)
- **Negative labels (unprofitable):** 10476 (82.9%)

⚠️ **Warning:** Low positive label rate (17.1%) - most signals are unprofitable

### Label Distribution Over Time

- **Daily positive rate - Mean:** 1.6%
- **Daily positive rate - Std:** 2.4%
- **Daily positive rate - Min:** 0.0%
- **Daily positive rate - Max:** 15.6%

⚠️ **Warning:** 1400 days with <10% positive labels

## 2. Signal Coverage and Sparsity

- **Total samples:** 135775
- **Labeled samples:** 12636
- **Coverage:** 9.3%

✅ **OK:** Reasonable signal coverage (9.3%)

## 3. Feature-Label Correlation Analysis


### Top 10 Most Correlated Features:

- **event_win_rate_last_50:** 0.2243
- **event_mean_return_last_50:** 0.1894
- **atr_ratio:** 0.1535
- **returns_entropy:** 0.1498
- **volatility_1d:** 0.1457
- **volatility_ema:** 0.1416
- **volatility_20:** 0.1386
- **volatility_4h:** 0.1341
- **atr_14:** 0.1133
- **drawdown_100:** -0.1131

### Correlation Health Check:


## 4. P&L Distribution Per Label

### Label = 1 (Profitable Signals):

- **Count:** 2160
- **Mean return:** 3.04%
- **Median return:** 2.94%
- **Std return:** 1.08%
- **% Actually positive:** 100.0%

### Label = 0 (Unprofitable Signals):

- **Count:** 10476
- **Mean return:** -1.46%
- **Median return:** -1.15%
- **Std return:** 0.87%
- **% Actually positive:** 0.0%

### Labeling Quality:

- **Label overlap:** 0.0%

✅ **OK:** Acceptable label overlap (0.0%)

### Cost-Aware Event Quality Summary

- **Transaction cost (per event, approx):** 0.150%
- **Unconditional mean event return:** -0.69%
- **Mean return (label=1) minus cost:** 2.89%
- **Fraction of labeled events with |return| < cost:** 0.0%

## 5. Time-Series Stability and Regime Analysis

- **Win rate vs Volatility correlation:** 0.4599
- **Mean return vs Volatility correlation:** 0.1353

✅ **OK:** Win rate not strongly correlated with volatility

### Regime Shift Detection:

- **30-day rolling correlation (win rate vs vol) - Mean:** 0.4393
- **30-day rolling correlation (win rate vs vol) - Std:** 0.1341

## 6. Model Probability Diagnostics


### Calibration (Predicted Probability vs Actual Success Rate):


| Predicted Prob | Actual Success Rate | Count |
|---------------|---------------------|-------|
| 0.070 | 0.062 | 2156 |
| 0.119 | 0.122 | 3531 |
| 0.180 | 0.196 | 2016 |
| 0.242 | 0.216 | 1067 |
| 0.306 | 0.258 | 561 |
| 0.367 | 0.323 | 313 |
| 0.429 | 0.320 | 172 |
| 0.500 | 0.236 | 2786 |
| 0.547 | 0.370 | 27 |
| 0.614 | 0.143 | 7 |

- **Mean calibration error:** 0.117

✅ **OK:** Reasonable calibration

### Monotonicity and Slope Checks (Prob → Realized Return)

- **Monotonicity violations (adjacent bins):** 4 / 9
- **Approx. slope in high-probability region:** -0.087422

### Probability Distribution:

- **Mean probability:** 0.475
- **Median probability:** 0.500
- **Std probability:** 0.091

## 7. Feature Importance Analysis


### Top 20 Features by Importance:

1. **dist_from_recent_high_50:** 0.0757
2. **dist_from_recent_low_50:** 0.0750
3. **sma_slope:** 0.0328
4. **volatility_1d:** 0.0322
5. **momentum_20:** 0.0269
6. **volatility_regime:** 0.0250
7. **momentum_per_vol:** 0.0249
8. **vol_ratio:** 0.0238
9. **volatility_5:** 0.0236
10. **vol_price_corr:** 0.0230
11. **atr_ratio:** 0.0229
12. **bars_since_last_event:** 0.0227
13. **volatility_4h:** 0.0221
14. **vol_of_vol:** 0.0219
15. **returns_entropy:** 0.0213
16. **kalman_trend:** 0.0209
17. **vol_regime_high:** 0.0199
18. **volatility_ema:** 0.0191
19. **ma_distance_kalman:** 0.0190
20. **volume_zscore:** 0.0187

### Feature Importance Health Check:


✅ **OK:** No single feature dominates

✅ Kalman-filtered features are being used (6 features)

✅ Volatility-based features in top 10: volatility_1d, volatility_regime, momentum_per_vol, vol_ratio, volatility_5, vol_price_corr

## 8. Kalman Smoothed Labels Analysis

- **Mean smoothed label:** 0.168
- **Median smoothed label:** 0.057
- **Std smoothed label:** 0.219
- **Correlation with binary labels:** 0.736
- **Correlation with realized returns:** 0.628

✅ **OK:** Good correlation between smoothed and binary labels

## 9. Event Mechanics and R-Multiples

These diagnostics describe how trades exit (profit, stop, or timeout), how long they stay open, and how large the realized return is relative to the configured stop-loss (R-multiple). They help verify that the triple-barrier / TPSL configuration produces economically meaningful events.


### Exit Reason Mix (Labeled Events)

- **timeout:** 81.1%
- **stop:** 9.6%
- **profit:** 9.3%

### Event Duration Distribution (Bars)

- **Mean duration:** 11.20
- **Median duration:** 12.00
- **90th percentile:** 12.00

### R-Multiple Distribution (Return / Stop Threshold)

- **R-multiple (label=1) mean:** 1.14
- **R-multiple (label=1) median:** 1.12
- **R-multiple (label=0) mean:** -0.55
- **R-multiple (label=0) median:** -0.43

## 10. Regime and Time-Conditional Label Checks

These metrics show how label base-rates and realized returns change across volatility regimes, trend states, and time-of-day/weekday. Large differences indicate that learnability and edge are regime-dependent, which is important for downstream model conditioning.


### Label Base-Rate by Volatility Regime

- **Regime high:** positive=21.5%, mean_return=-0.63%
- **Regime low:** positive=11.1%, mean_return=-0.73%
- **Regime medium:** positive=13.9%, mean_return=-0.75%

### Trend-Conditional (Price vs SMA20)

- **Strong uptrend:** positive=19.1%, mean_return=-0.66%
- **Strong downtrend:** positive=21.4%, mean_return=-0.57%

### Time-of-Day Positive Rates (Top/Bottom)

- **Top hours by positive rate:**
  - Hour 14: 21.7%
  - Hour 16: 21.0%
  - Hour 12: 20.5%
- **Bottom hours by positive rate:**
  - Hour 2: 10.0%
  - Hour 4: 12.8%
  - Hour 3: 13.4%

### Day-of-Week Positive Rates

- Day 0: 20.3%
- Day 1: 17.3%
- Day 2: 18.5%
- Day 3: 18.4%
- Day 4: 17.4%
- Day 5: 12.0%
- Day 6: 13.5%

## 11. Label–Return Separation and Information Content

This section quantifies how well the labels separate profitable from unprofitable events (effect size) and how much information they carry about the sign of future returns compared to a random baseline (mutual information and permutation tests).


### Separation Metrics

- **Mean return difference (label=1 - label=0):** 4.50%
- **Cohen's d effect size:** 4.966

### Information Content vs Permutation Baseline

- **Mutual information (labels vs realized sign):** 0.4574
- **Baseline MI (mean over permutations):** 0.0000

## 12. Target and Expected-Return Alignment

Here we check whether the continuous targets produced by the isotonic mapping are consistent with realized returns: higher targets should correspond to higher average realized P&L, and most non-zero targets should exceed transaction costs.


### Target vs Realized Return

- **Correlation (target, realized):** 0.000
- **MSE (target vs realized):** 0.000551

### Target/Return by Target Decile


### Target Distribution Sanity

- **Non-zero target fraction (all samples):** 24.4%
- **Mean non-zero target:** 0.0020
- **Median non-zero target:** 0.0020
- **Fraction of targets below transaction cost (0.150%):** 0.0%

## 13. Cost-Aware Metrics and Threshold P&L

These diagnostics evaluate the meta-model from a trading perspective: global AUC/Brier/PR-AUC, and how mean and cumulative returns evolve as you raise the probability threshold used to filter trades. This helps choose probability cutoffs that are profitable after costs.


### Cost-Aware Classification Metrics (OOF)

- **AUC:** 0.641
- **Brier score:** 0.1522
- **Average precision (PR-AUC):** 0.232

### Threshold-Sweep P&L (Using Meta Probability)


| Threshold | Trades | Mean Return | Cum Return | Sharpe (per trade) |
|----------|--------|-------------|------------|---------------------|
| 0.50 | 2781 | -0.55% | -1520.87% | -13.019 |
| 0.51 | 43 | 0.14% | 6.06% | 0.285 |
| 0.52 | 36 | -0.57% | -20.63% | -1.137 |
| 0.53 | 33 | -0.97% | -31.99% | -1.966 |
| 0.54 | 26 | -0.83% | -21.59% | -1.512 |
| 0.55 | 16 | -1.05% | -16.84% | -1.651 |
| 0.56 | 10 | -1.73% | -17.32% | -2.432 |
| 0.57 | 8 | -1.48% | -11.87% | -1.711 |
| 0.58 | 8 | -1.48% | -11.87% | -1.711 |
| 0.59 | 7 | -2.03% | -14.24% | -2.627 |
| 0.60 | 3 | -2.54% | -7.61% | -4.045 |
| 0.61 | 3 | -2.54% | -7.61% | -4.045 |
| 0.62 | 2 | -2.21% | -4.43% | -2.379 |
| 0.63 | 2 | -2.21% | -4.43% | -2.379 |
| 0.64 | 2 | -2.21% | -4.43% | -2.379 |
| 0.65 | 0 | N/A | N/A | N/A |

## 14. Stability and Bootstrap Statistics

Finally, we assess how stable the labeling and model performance are across time (years and time-series folds) and how sensitive key metrics such as positive rate and label–return separation are under bootstrap resampling. Tight confidence intervals indicate robust signal rather than fragile noise.


### Per-Year Label and Return Stability

- **Year 2021:** positive=19.5%, mean_return=-0.63%, AUC=0.500
- **Year 2022:** positive=22.4%, mean_return=-0.61%, AUC=0.551
- **Year 2023:** positive=11.8%, mean_return=-0.74%, AUC=0.607
- **Year 2024:** positive=13.4%, mean_return=-0.80%, AUC=0.691
- **Year 2025:** positive=17.4%, mean_return=-0.66%, AUC=0.674

### Time-Series Fold Stability (Approximate)

- **Fold 1:** positive=19.1%, mean_return=-0.69%, AUC=0.596
- **Fold 2:** positive=12.2%, mean_return=-0.71%, AUC=0.657
- **Fold 3:** positive=12.6%, mean_return=-0.78%, AUC=0.688
- **Fold 4:** positive=14.1%, mean_return=-0.80%, AUC=0.687
- **Fold 5:** positive=17.1%, mean_return=-0.67%, AUC=0.670

### Bootstrap Label Statistics (Labeled Events)

- **Positive rate 95% CI:** [16.5%, 17.6%]
- **Mean return diff (label=1 - label=0) 95% CI:** [4.44%, 4.54%]

---


## Summary and Recommendations


### Key Findings:

1. Label balance: 17.1% positive
2. Signal coverage: 9.3%
3. Mean return (label=1): 3.04%
4. Mean return (label=0): -1.46%
5. Calibration error: 0.117