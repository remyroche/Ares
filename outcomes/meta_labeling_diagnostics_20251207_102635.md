# Meta-Labeling Diagnostics Report

**Generated:** 2025-12-07 10:26:35

**Symbol:** ETHUSDT
**Timeframe:** 15m
**Horizon:** 20 bars

---


## 0. Signal Funnel (Primary Signals)

- **Total bars:** 140354
- **Raw non-zero signals:** 115496
- **Final consensus signals:** 122539 (ratio=1.061)
- **Raw long/short:** 58241/57255
- **Final long/short:** 59370/63169
- **Relaxed extra signals (strict=0 but raw≠0):** 3113
- **Raw signal density:** 0.82289 per bar
- **Final signal density:** 0.87307 per bar
- **Raw signals per day (approx):** 78.995
- **Final consensus signals per day (approx):** 83.812

ℹ️ **Note:** Consensus preserves most raw signals (final/raw > 0.90).

⚠️ **Warning:** Very dense primary signals (83.812 trades/day). Overlapping events may be frequent.

## 1. Label Distribution Analysis

- **Total labeled events:** 5110
- **Positive labels (profitable):** 2555 (50.0%)
- **Negative labels (unprofitable):** 2555 (50.0%)

✅ **OK:** Reasonable label balance (50.0%)

### Label Distribution Over Time

- **Daily positive rate - Mean:** 1.8%
- **Daily positive rate - Std:** 3.8%
- **Daily positive rate - Min:** 0.0%
- **Daily positive rate - Max:** 20.8%

⚠️ **Warning:** 1374 days with <10% positive labels

## 2. Signal Coverage and Sparsity

- **Total samples:** 140354
- **Labeled samples:** 5110
- **Coverage:** 3.6%

⚠️ **Warning:** Very sparse signals (3.6%) - consider lowering signal thresholds

## 3. Feature-Label Correlation Analysis (Post-Filter)


### Top 10 Most Correlated Features (Post-Filter):

- **volatility_1d:** 0.3474
- **event_mean_return_last_50:** 0.2824
- **event_r_multiple_mean_last_50:** 0.2811
- **event_win_rate_last_50:** 0.2797
- **returns_std_50:** 0.2659
- **event_tto_mean_last_50:** -0.2045
- **low_dist_x_vol:** 0.2009
- **zigzag_bars_since_pivot:** 0.1530
- **base_zigzag_bars_since_pivot:** 0.1530
- **returns_entropy:** 0.1477

### Correlation Health Check (Post-Filter):


## 3B. Pre- vs Post-Filter Comparison and Feature Shift


### Sample Counts and Retention

- **Pre-filter events (realized_return not NaN):** 43472
- **Pre-filter positive/negative (raw economic):** 11342 / 32130
- **Post-filter labeled events:** 5110
- **Post-filter positive/negative (binary_labels):** 2555 / 2555
- **Total retention (post / pre):** 11.8%
- **Positive retention:** 22.5%
- **Negative retention:** 8.0%

### Pre- vs Post-Filter Signal Quality

- **Pre-filter mean return (label=1/0):** 1.31% / -0.84%
- **Post-filter mean return (label=1/0):** 1.42% / -1.15%
- **Pre-filter Cohen's d (label=1 vs 0):** 4.391
- **Post-filter Cohen's d (label=1 vs 0):** 6.008
- **Pre-filter SNR (mean/std, label=1):** 2.392
- **Post-filter SNR (mean/std, label=1):** 2.353

### Largest Feature Correlation Shifts (|post|-|pre|)

- **volatility_1d:** pre=0.0309, post=0.3474, Δ|corr|=0.3165
- **returns_std_50:** pre=0.0330, post=0.2659, Δ|corr|=0.2329
- **low_dist_x_vol:** pre=0.0188, post=0.2009, Δ|corr|=0.1820
- **signal_disagreement:** pre=0.1399, post=0.0002, Δ|corr|=-0.1398
- **event_mean_return_last_50:** pre=0.1475, post=0.2824, Δ|corr|=0.1349
- **event_r_multiple_mean_last_50:** pre=0.1485, post=0.2811, Δ|corr|=0.1327
- **dist_from_recent_low_50:** pre=0.0167, post=0.1468, Δ|corr|=0.1301
- **is_sunday:** pre=-0.0055, post=-0.1248, Δ|corr|=0.1192
- **close_range_50:** pre=0.0269, post=0.1460, Δ|corr|=0.1191
- **trend_strength_4H:** pre=0.0010, post=0.1170, Δ|corr|=0.1160

## 4. P&L Distribution Per Label

### Label = 1 (Profitable Signals):

- **Count:** 2555
- **Mean return:** 1.42%
- **Median return:** 1.67%
- **Std return:** 0.60%
- **% Actually positive:** 96.4%

### Label = 0 (Unprofitable Signals):

- **Count:** 2555
- **Mean return:** -1.15%
- **Median return:** -1.17%
- **Std return:** 0.06%
- **% Actually positive:** 0.0%

### Labeling Quality:

- **Label overlap:** 1.8%

✅ **OK:** Acceptable label overlap (1.8%)

### Cost-Aware Event Quality Summary

- **Transaction cost (per event, approx):** 0.300%
- **Unconditional mean event return:** 0.13%
- **Mean return (label=1) minus cost:** 1.12%
- **Fraction of labeled events with |return| < cost:** 0.0%

## 5. Time-Series Stability and Regime Analysis

- **Win rate vs Volatility correlation:** 0.4402
- **Mean return vs Volatility correlation:** 0.1130

✅ **OK:** Win rate not strongly correlated with volatility

### Rolling Daily SNR (Return / Std by Day)

- **Daily SNR mean:** -0.3749
- **Daily SNR std:** 0.4317
- **Daily SNR min/max:** -2.6965 / 1.0886

### Regime Shift Detection:

- **30-day rolling correlation (win rate vs vol) - Mean:** 0.4453
- **30-day rolling correlation (win rate vs vol) - Std:** 0.1157

## 6. Model Probability Diagnostics


### Calibration (Predicted Probability vs Actual Success Rate):


| Predicted Prob | Actual Success Rate | Count |
|---------------|---------------------|-------|
| 0.153 | 0.250 | 68 |
| 0.226 | 0.249 | 273 |
| 0.306 | 0.298 | 557 |
| 0.393 | 0.328 | 512 |
| 0.495 | 0.504 | 2084 |
| 0.565 | 0.456 | 421 |
| 0.650 | 0.538 | 411 |
| 0.741 | 0.753 | 324 |
| 0.824 | 0.907 | 313 |
| 0.903 | 0.980 | 147 |

- **Mean calibration error:** 0.060

✅ **OK:** Reasonable calibration

### Monotonicity and Slope Checks (Prob → Realized Return)

- **Monotonicity violations (adjacent bins):** 3 / 9
- **Approx. slope in high-probability region:** 0.035427

### Probability Distribution:

- **Mean probability:** 0.500
- **Median probability:** 0.500
- **Std probability:** 0.032

⚠️ **Warning:** Very low probability variance - model may not be learning useful patterns

## 7. Feature Importance Analysis


### Top 20 Features by Importance:

1. **swing_1H:** 0.1278
2. **momentum_1h:** 0.0335
3. **signal_candle_trend_x_rsi_distance_50:** 0.0333
4. **range_1h:** 0.0258
5. **returns_mean_5:** 0.0257
6. **trend_base_cat:** 0.0226
7. **zigzag_last_pivot_idx:** 0.0205
8. **mtf_base_vs_htf_conflict:** 0.0199
9. **zigzag_trend_direction:** 0.0173
10. **signal_ma_distance_raw:** 0.0131
11. **range_4h:** 0.0125
12. **trend_regime:** 0.0101
13. **candle_reversal:** 0.0098
14. **volatility_1h_agg:** 0.0092
15. **close_min_5:** 0.0092
16. **zigzag_swing_magnitude:** 0.0088
17. **rsi_kalman:** 0.0088
18. **mtf_mixed_signals:** 0.0085
19. **zigzag_current_swing:** 0.0085
20. **mtf_confluence_class:** 0.0083

### Feature Importance Health Check:


✅ **OK:** No single feature dominates

✅ Kalman-filtered features are being used (6 features)

## 8. Kalman Smoothed Labels Analysis

- **Mean smoothed label:** 0.504
- **Median smoothed label:** 0.489
- **Std smoothed label:** 0.380
- **Correlation with binary labels:** 0.879
- **Correlation with realized returns:** 0.822

✅ **OK:** Good correlation between smoothed and binary labels

## 9. Event Mechanics and R-Multiples

These diagnostics describe how trades exit (profit, stop, or timeout), how long they stay open, and how large the realized return is relative to the configured stop-loss (R-multiple). They help verify that the triple-barrier / TPSL configuration produces economically meaningful events.


### Exit Reason Mix (Labeled Events)

- **1:** 50.5%
- **0:** 36.3%
- **2:** 13.2%

### Event Duration Distribution (Bars)

- **Mean duration:** 13.33
- **Median duration:** 11.00
- **90th percentile:** 27.00

### R-Multiple Distribution (Return / Stop Threshold)

- **R-multiple (label=1) mean:** 1.67
- **R-multiple (label=1) median:** 1.98
- **R-multiple (label=0) mean:** -1.35
- **R-multiple (label=0) median:** -1.34

## 9B. Enhanced Event Quality Metrics

These advanced metrics help distinguish between efficient momentum capture and random drift trades. They measure path efficiency, timing quality, and entry/exit effectiveness.


### Path Efficiency Ratio (PER)

- **Mean PER (clipped 99.5%):** 0.603
- **Median PER:** 0.617
- **90th percentile PER:** 0.766
- **99th percentile PER:** 0.827

✅ **OK:** Good path efficiency on profitable events

### Time-to-Outcome Ratio (TTO)

- **Mean TTO:** 0.666
- **Median TTO:** 0.550

⚠️ **Warning:** TTO outside target range [0.4, 0.6]

### MFE/MAE Ratio (Maximum Favorable vs Adverse Excursion)

- **Mean MFE/MAE:** 353.545
- **Median MFE/MAE:** 1.491

✅ **OK:** Favorable excursions exceed adverse excursions

## 9C. Target Volatility Health Check

Verifies that the continuous target values have sufficient variance to train a regression model. If targets are constant or near-zero, the model cannot learn meaningful patterns.

- **Target mean:** -0.000957
- **Target std:** 0.001759
- **Non-zero targets:** 977 / 140354 (0.7%)

⚠️ **Warning:** Very few non-zero targets (0.7%)

## 9D. Information Coefficient (IC)

Measures the rank correlation between predicted probabilities and realized returns. This is the purest measure of ranking ability, independent of absolute calibration.

- **Spearman IC (prob, return):** 0.2245
- **P-value:** 2.2403e-59

✅ **OK:** Meaningful rank correlation

## 10. Regime and Time-Conditional Label Checks

These metrics show how label base-rates and realized returns change across volatility regimes, trend states, and time-of-day/weekday. Large differences indicate that learnability and edge are regime-dependent, which is important for downstream model conditioning.


### Label Base-Rate by Volatility Regime

- **Regime high:** positive=74.6%, mean_return=0.69%
- **Regime low:** positive=29.9%, mean_return=-0.36%
- **Regime medium:** positive=43.6%, mean_return=0.00%

### Trend-Conditional (Price vs SMA20)

- **Strong uptrend:** positive=50.9%, mean_return=0.16%
- **Strong downtrend:** positive=50.0%, mean_return=0.16%

### Time-of-Day Positive Rates (Top/Bottom)

- **Top hours by positive rate:**
  - Hour 4: 62.9%
  - Hour 8: 60.6%
  - Hour 5: 58.9%
- **Bottom hours by positive rate:**
  - Hour 15: 38.7%
  - Hour 16: 40.4%
  - Hour 18: 42.1%

### Day-of-Week Positive Rates

- Day 0: 45.0%
- Day 1: 55.5%
- Day 2: 53.8%
- Day 3: 56.2%
- Day 4: 57.4%
- Day 5: 53.3%
- Day 6: 37.1%

## 11. Label–Return Separation, Information Content, and Sample Size

This section quantifies how well the labels separate profitable from unprofitable events (effect size) and how much information they carry about the sign of future returns compared to a random baseline (mutual information and permutation tests).


### Separation Metrics

- **Mean return difference (label=1 - label=0):** 2.57%
- **Cohen's d effect size:** 6.008
- **Approx. required samples for 80% power (heuristic):** 0.4
- **Current labeled samples used in separation:** 5110.0

### Information Content vs Permutation Baseline

- **Mutual information (labels vs realized sign):** 0.6150
- **Baseline MI (mean over permutations):** 0.0001

## 12. Target and Expected-Return Alignment

Here we check whether the continuous targets produced by the isotonic mapping are consistent with realized returns: higher targets should correspond to higher average realized P&L, and most non-zero targets should exceed transaction costs.


### Target vs Realized Return

- **Correlation (target, realized):** 0.217
- **MSE (target vs realized):** 0.000139

### Target/Return by Target Decile

- Decile 0: target=0.0010, realized=0.0018
- Decile 1: target=0.0032, realized=0.0042
- Decile 2: target=0.0048, realized=0.0053
- Decile 3: target=0.0079, realized=0.0116
- Decile 4: target=0.0093, realized=0.0076
- Decile 5: target=0.0107, realized=0.0094
- Decile 6: target=0.0111, realized=0.0096
- Decile 7: target=0.0115, realized=0.0089
- Decile 8: target=0.0125, realized=0.0106

### Target Distribution Sanity

- **Non-zero target fraction (all samples):** 0.7%
- **Mean non-zero target:** 0.0082
- **Median non-zero target:** 0.0102
- **Fraction of targets below transaction cost (0.300%):** 14.9%

## 13. Cost-Aware Metrics and Threshold P&L

These diagnostics evaluate the meta-model from a trading perspective: global AUC/Brier/PR-AUC, and how mean and cumulative returns evolve as you raise the probability threshold used to filter trades. This helps choose probability cutoffs that are profitable after costs.


### Cost-Aware Classification Metrics (OOF)

- **AUC:** 0.684
- **Brier score:** 0.2200
- **Average precision (PR-AUC):** 0.705

### Volatility Regime Breakdown

| Regime | Samples | Pos Rate | AUC | Mean Ret | Sharpe |
|--------|---------|----------|-----|----------|--------|
| Low | 1144 | 29.9% | 0.517 | -0.36% | -0.30 |
| Medium | 2404 | 43.6% | 0.620 | 0.00% | 0.00 |
| High | 1562 | 74.6% | 0.704 | 0.69% | 0.54 |

⚠️ **Warning:** Large win-rate disparity between regimes (low: 29.9%, high: 74.6%). Performance is highly regime-dependent.

### Threshold-Sweep P&L (Using Meta Probability)


| Threshold | Trades | Mean Return | Cum Return | Sharpe (per trade) |
|----------|--------|-------------|------------|---------------------|
| 0.50 | 3320 | 0.36% | 1199.55% | 15.485 |
| 0.51 | 1668 | 0.49% | 809.70% | 15.105 |
| 0.52 | 1628 | 0.51% | 828.56% | 15.724 |
| 0.53 | 1577 | 0.52% | 827.61% | 16.023 |
| 0.54 | 1531 | 0.54% | 822.36% | 16.211 |
| 0.55 | 1483 | 0.55% | 813.81% | 16.331 |
| 0.56 | 1438 | 0.57% | 816.25% | 16.705 |
| 0.57 | 1380 | 0.59% | 808.48% | 17.000 |
| 0.58 | 1328 | 0.62% | 817.34% | 17.672 |
| 0.59 | 1283 | 0.64% | 822.57% | 18.243 |
| 0.60 | 1238 | 0.66% | 812.59% | 18.416 |
| 0.61 | 1188 | 0.68% | 803.41% | 18.685 |
| 0.62 | 1134 | 0.70% | 798.62% | 19.202 |
| 0.63 | 1080 | 0.73% | 788.59% | 19.596 |
| 0.64 | 1035 | 0.75% | 776.16% | 19.877 |
| 0.65 | 988 | 0.77% | 757.26% | 19.993 |

## 14. Stability and Bootstrap Statistics

Finally, we assess how stable the labeling and model performance are across time (years and time-series folds) and how sensitive key metrics such as positive rate and label–return separation are under bootstrap resampling. Tight confidence intervals indicate robust signal rather than fragile noise.


### Per-Year Label and Return Stability

- **Year 2024:** positive=53.8%, mean_return=0.24%, AUC=0.500
- **Year 2025:** positive=49.4%, mean_return=0.11%, AUC=0.706

### Time-Series Fold Stability (Approximate)

- **Fold 4:** positive=53.8%, mean_return=0.25%, AUC=0.500
- **Fold 5:** positive=48.3%, mean_return=0.08%, AUC=0.736

### Bootstrap Label Statistics (Labeled Events)

- **Positive rate 95% CI:** [48.5%, 51.4%]
- **Mean return diff (label=1 - label=0) 95% CI:** [2.55%, 2.59%]

---


## Summary and Recommendations


### Key Findings:

1. Label balance: 50.0% positive
2. Signal coverage: 3.6%
3. Mean return (label=1): 1.42%
4. Mean return (label=0): -1.15%
5. Calibration error: 0.060