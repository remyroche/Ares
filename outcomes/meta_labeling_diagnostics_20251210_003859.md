# Meta-Labeling Diagnostics Report

**Generated:** 2025-12-10 00:38:59

**Symbol:** ETHUSDT
**Timeframe:** 15m
**Horizon:** 22 bars

**Final selected features (count):** 
61 features


**Final selected features (full list):**

- absorption_ratio
- close_max_50
- close_min_50
- close_range_10
- close_range_5
- close_range_50
- cvd_proxy
- day_of_week
- dist_from_recent_high_10
- dist_from_recent_low_5
- drawdown_100
- high_dist_x_vol
- hour
- hour_sin
- kalman_range_position
- kalman_trend_x_vol_ratio
- liquidity_gap_abs
- liquidity_gap_down
- log_ret
- low_dist_x_vol
- ma_distance_per_vol
- meta_trendiness
- meta_trendiness_x_meta_volatility_regime
- meta_volatility_regime
- meta_volume_shock
- meta_volume_shock_x_hour_cos
- meta_volume_shock_x_hour_sin
- momentum_10_x_regime_high
- momentum_20
- momentum_20_x_regime_high
- momentum_20_x_regime_medium
- momentum_5_x_regime_high
- momentum_per_vol
- ofi_proxy
- range_4h
- range_position
- range_position_x_vol_ratio
- return_autocorr_lag1_w50
- returns_1h
- returns_mean_50
- returns_std_10
- risk_score
- rsi_kalman
- signal_macd_hist_abs
- signal_macd_hist_long_abs
- signal_rsi_distance_50
- signal_rsi_long_distance_50
- signed_volume_ema
- sma_slope_22b
- smc_predicted
- sr_labeling_xgb_prob
- vol_of_vol
- vol_price_corr
- volatility_1d
- volatility_1h
- volatility_4h_agg
- volatility_5
- volatility_ema
- volume_pressure
- volume_trend
- volume_zscore

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

- **Total labeled events:** 30323
- **Positive labels (profitable):** 2743 (9.0%)
- **Negative labels (unprofitable):** 27580 (91.0%)

⚠️ **Warning:** Low positive label rate (9.0%) - most signals are unprofitable

### Label Distribution Over Time

- **Daily positive rate - Mean:** 2.0%
- **Daily positive rate - Std:** 3.6%
- **Daily positive rate - Min:** 0.0%
- **Daily positive rate - Max:** 19.8%

⚠️ **Warning:** 1376 days with <10% positive labels

## 2. Signal Coverage and Sparsity

- **Total samples:** 140354
- **Labeled samples:** 30323
- **Coverage:** 21.6%

✅ **OK:** Reasonable signal coverage (21.6%)

## 3. Feature-Label Correlation Analysis (Post-Filter)


### Top 10 Most Correlated Features (Post-Filter):

- **smc_predicted:** 0.4803
- **risk_score:** 0.4668
- **close_minus_vwap:** 0.2248
- **kalman_close_minus_vwap:** 0.2246
- **close_max_5:** 0.1803
- **close_min_5:** 0.1803
- **close_max_20:** 0.1801
- **close_max_10:** 0.1799
- **kalman_trend:** 0.1797
- **close_min_10:** 0.1797

### Correlation Health Check (Post-Filter):


## 3B. Pre- vs Post-Filter Comparison and Feature Shift


### Sample Counts and Retention

- **Pre-filter events (realized_return not NaN):** 33064
- **Pre-filter positive/negative (raw economic):** 9646 / 23418
- **Post-filter labeled events:** 30323
- **Post-filter positive/negative (binary_labels):** 2743 / 27580
- **Total retention (post / pre):** 91.7%
- **Positive retention:** 28.4%
- **Negative retention:** 117.8%

### Pre- vs Post-Filter Signal Quality

- **Pre-filter mean return (label=1/0):** 0.86% / -0.75%
- **Post-filter mean return (label=1/0):** 0.75% / -0.41%
- **Pre-filter Cohen's d (label=1 vs 0):** 7.210
- **Post-filter Cohen's d (label=1 vs 0):** 1.605
- **Pre-filter SNR (mean/std, label=1):** 6.202
- **Post-filter SNR (mean/std, label=1):** 1.799

⚠️ **Warning:** Post-filter effect size is materially worse than pre-filter – filters may be discarding informative events

### Largest Feature Correlation Shifts (|post|-|pre|)

- **smc_predicted:** pre=0.0257, post=0.4803, Δ|corr|=0.4546
- **risk_score:** pre=0.0222, post=0.4668, Δ|corr|=0.4446
- **close_minus_vwap:** pre=0.0215, post=0.2248, Δ|corr|=0.2033
- **kalman_close_minus_vwap:** pre=0.0220, post=0.2246, Δ|corr|=0.2026
- **close_min_5:** pre=0.0268, post=0.1803, Δ|corr|=0.1534
- **close_max_5:** pre=0.0271, post=0.1803, Δ|corr|=0.1532
- **close_min_10:** pre=0.0266, post=0.1797, Δ|corr|=0.1531
- **close_max_10:** pre=0.0272, post=0.1799, Δ|corr|=0.1527
- **close_min_20:** pre=0.0270, post=0.1796, Δ|corr|=0.1526
- **close_max_20:** pre=0.0278, post=0.1801, Δ|corr|=0.1523

## 4. P&L Distribution Per Label

### Label = 1 (Profitable Signals):

- **Count:** 2743
- **Mean return:** 0.75%
- **Median return:** 0.94%
- **Std return:** 0.42%
- **% Actually positive:** 93.6%

### Label = 0 (Unprofitable Signals):

- **Count:** 27580
- **Mean return:** -0.41%
- **Median return:** -0.86%
- **Std return:** 0.74%
- **% Actually positive:** 24.9%

### Labeling Quality:

- **Label overlap:** 23.3%

✅ **OK:** Acceptable label overlap (23.3%)

### Cost-Aware Event Quality Summary

- **Transaction cost (per event, approx):** 0.300%
- **Unconditional mean event return:** -0.30%
- **Mean return (label=1) minus cost:** 0.45%
- **Fraction of labeled events with |return| < cost:** 0.0%

## 5. Time-Series Stability and Regime Analysis

- **Win rate vs Volatility correlation:** 0.1024
- **Mean return vs Volatility correlation:** 0.0478

✅ **OK:** Win rate not strongly correlated with volatility

### Rolling Daily SNR (Return / Std by Day)

- **Daily SNR mean:** -0.4425
- **Daily SNR std:** 0.3987
- **Daily SNR min/max:** -3.1438 / 0.6720

### Regime Shift Detection:

- **30-day rolling correlation (win rate vs vol) - Mean:** 0.0464
- **30-day rolling correlation (win rate vs vol) - Std:** 0.2186

## 6. Model Probability Diagnostics


### Calibration (Predicted Probability vs Actual Success Rate):


| Predicted Prob | Actual Success Rate | Count |
|---------------|---------------------|-------|
| 0.006 | 0.051 | 19233 |
| 0.142 | 0.084 | 405 |
| 0.251 | 0.093 | 290 |
| 0.348 | 0.094 | 267 |
| 0.447 | 0.168 | 256 |
| 0.502 | 0.018 | 5261 |
| 0.651 | 0.326 | 230 |
| 0.755 | 0.314 | 363 |
| 0.856 | 0.335 | 702 |
| 0.969 | 0.337 | 3316 |

- **Mean calibration error:** 0.320

⚠️ **Warning:** High calibration error - model probabilities may not be well-calibrated

### Monotonicity and Slope Checks (Prob → Realized Return)

- **Monotonicity violations (adjacent bins):** 4 / 9
- **Approx. slope in high-probability region:** 0.000578

### Probability Distribution:

- **Mean probability:** 0.444
- **Median probability:** 0.500
- **Std probability:** 0.192

## 7. Feature Importance Analysis


### Top 20 Features by Importance:

1. **vol_of_vol:** 0.4264
2. **volatility_regime:** 0.2217
3. **liquidity_gap_abs:** 0.0870
4. **kalman_trend_x_vol_ratio:** 0.0345
5. **volume_trend:** 0.0082
6. **volatility_4h:** 0.0066
7. **volume_pressure:** 0.0051
8. **signed_volume_ema:** 0.0050
9. **cvd_proxy:** 0.0049
10. **kalman_range_position:** 0.0047
11. **smc_predicted:** 0.0046
12. **atr_14:** 0.0046
13. **is_good_hour:** 0.0046
14. **ofi_proxy:** 0.0045
15. **vol_price_corr:** 0.0043
16. **is_bad_hour:** 0.0043
17. **is_sunday:** 0.0043
18. **close_minus_vwap:** 0.0043
19. **risk_score:** 0.0043
20. **momentum_kalman:** 0.0042

### Feature Importance Health Check:


✅ **OK:** No single feature dominates

✅ Kalman-filtered features are being used (8 features)

✅ Volatility-based features in top 10: vol_of_vol, volatility_regime, kalman_trend_x_vol_ratio, volume_trend, volatility_4h, volume_pressure, signed_volume_ema

## 8. Kalman Smoothed Labels Analysis

- **Mean smoothed label:** 0.091
- **Median smoothed label:** 0.000
- **Std smoothed label:** 0.232
- **Correlation with binary labels:** 0.942
- **Correlation with realized returns:** 0.347

✅ **OK:** Good correlation between smoothed and binary labels

## 9. Event Mechanics and R-Multiples

These diagnostics describe how trades exit (profit, stop, or timeout), how long they stay open, and how large the realized return is relative to the configured stop-loss (R-multiple). They help verify that the triple-barrier / TPSL configuration produces economically meaningful events.


### Exit Reason Mix (Labeled Events)

- **1:** 67.2%
- **0:** 30.2%
- **2:** 2.6%

### Event Duration Distribution (Bars)

- **Mean duration:** 9.40
- **Median duration:** 6.00
- **90th percentile:** 23.00

### R-Multiple Distribution (Return / Stop Threshold)

- **R-multiple (label=1) mean:** 1.38
- **R-multiple (label=1) median:** 1.68
- **R-multiple (label=0) mean:** -0.76
- **R-multiple (label=0) median:** -1.53

## 9B. Enhanced Event Quality Metrics

These advanced metrics help distinguish between efficient momentum capture and random drift trades. They measure path efficiency, timing quality, and entry/exit effectiveness.


### Path Efficiency Ratio (PER)

- **Mean PER (clipped 99.5%):** 0.521
- **Median PER:** 0.527
- **90th percentile PER:** 0.658
- **99th percentile PER:** 0.730

✅ **OK:** Good path efficiency on profitable events

### Time-to-Outcome Ratio (TTO)

- **Mean TTO:** 0.427
- **Median TTO:** 0.273

✅ **OK:** TTO in healthy range [0.4, 0.6]

### MFE/MAE Ratio (Maximum Favorable vs Adverse Excursion)

- **Mean MFE/MAE:** 165.813
- **Median MFE/MAE:** 0.671

✅ **OK:** Favorable excursions exceed adverse excursions

## 9C. Target Volatility Health Check

Verifies that the continuous target values have sufficient variance to train a regression model. If targets are constant or near-zero, the model cannot learn meaningful patterns.

- **Target mean:** -0.001920
- **Target std:** 0.003490
- **Non-zero targets:** 0 / 140354 (0.0%)

⚠️ **Warning:** Very few non-zero targets (0.0%)

## 9D. Information Coefficient (IC)

Measures the rank correlation between predicted probabilities and realized returns. This is the purest measure of ranking ability, independent of absolute calibration.

- **Spearman IC (prob, return):** -0.0277
- **P-value:** 1.3821e-06

⚠️ **Warning:** Very weak IC (|IC| < 0.05) - model has minimal ranking ability

## 10. Regime and Time-Conditional Label Checks

These metrics show how label base-rates and realized returns change across volatility regimes, trend states, and time-of-day/weekday. Large differences indicate that learnability and edge are regime-dependent, which is important for downstream model conditioning.


### Label Base-Rate by Volatility Regime

- **Regime high:** positive=11.3%, mean_return=-0.32%
- **Regime low:** positive=4.9%, mean_return=-0.30%
- **Regime medium:** positive=9.7%, mean_return=-0.29%

### Trend-Conditional (Price vs SMA20)

- **Strong uptrend:** positive=10.5%, mean_return=-0.29%
- **Strong downtrend:** positive=9.7%, mean_return=-0.30%

### Time-of-Day Positive Rates (Top/Bottom)

- **Top hours by positive rate:**
  - Hour 4: 11.1%
  - Hour 3: 10.9%
  - Hour 6: 10.7%
- **Bottom hours by positive rate:**
  - Hour 15: 7.1%
  - Hour 23: 7.6%
  - Hour 13: 7.6%

### Day-of-Week Positive Rates

- Day 0: 9.4%
- Day 1: 9.4%
- Day 2: 8.1%
- Day 3: 9.1%
- Day 4: 9.1%
- Day 5: 9.6%
- Day 6: 8.7%

## 11. Label–Return Separation, Information Content, and Sample Size

This section quantifies how well the labels separate profitable from unprofitable events (effect size) and how much information they carry about the sign of future returns compared to a random baseline (mutual information and permutation tests).


### Separation Metrics

- **Mean return difference (label=1 - label=0):** 1.16%
- **Cohen's d effect size:** 1.605
- **Approx. required samples for 80% power (heuristic):** 6.2
- **Current labeled samples used in separation:** 30323.0

### Information Content vs Permutation Baseline

- **Mutual information (labels vs realized sign):** 0.0880
- **Baseline MI (mean over permutations):** 0.0000

## 12. Target and Expected-Return Alignment

Here we check whether the continuous targets produced by the isotonic mapping are consistent with realized returns: higher targets should correspond to higher average realized P&L, and most non-zero targets should exceed transaction costs.


## 13. Cost-Aware Metrics and Threshold P&L

These diagnostics evaluate the meta-model from a trading perspective: global AUC/Brier/PR-AUC, and how mean and cumulative returns evolve as you raise the probability threshold used to filter trades. This helps choose probability cutoffs that are profitable after costs.


### Cost-Aware Classification Metrics (OOF)

- **AUC:** 0.691
- **Brier score:** 0.1680
- **Average precision (PR-AUC):** 0.247

### Volatility Regime Breakdown

| Regime | Samples | Pos Rate | AUC | Mean Ret | Sharpe |
|--------|---------|----------|-----|----------|--------|
| Low | 7896 | 4.9% | 0.742 | -0.30% | -0.39 |
| Medium | 11261 | 9.7% | 0.716 | -0.29% | -0.37 |
| High | 11148 | 11.3% | 0.621 | -0.32% | -0.40 |

### Threshold-Sweep P&L (Using Meta Probability)


| Threshold | Trades | Mean Return | Cum Return | Sharpe (per trade) |
|----------|--------|-------------|------------|---------------------|
| 0.50 | 9872 | -0.32% | -3145.47% | -39.881 |
| 0.51 | 4787 | -0.31% | -1472.60% | -27.028 |
| 0.52 | 4767 | -0.31% | -1469.66% | -27.044 |
| 0.53 | 4749 | -0.31% | -1459.65% | -26.894 |
| 0.54 | 4733 | -0.31% | -1454.66% | -26.849 |
| 0.55 | 4717 | -0.31% | -1454.85% | -26.915 |
| 0.56 | 4698 | -0.31% | -1444.14% | -26.759 |
| 0.57 | 4681 | -0.31% | -1440.19% | -26.743 |
| 0.58 | 4657 | -0.31% | -1435.43% | -26.734 |
| 0.59 | 4631 | -0.31% | -1425.81% | -26.631 |
| 0.60 | 4610 | -0.31% | -1416.58% | -26.513 |
| 0.61 | 4584 | -0.31% | -1407.37% | -26.411 |
| 0.62 | 4562 | -0.31% | -1396.91% | -26.270 |
| 0.63 | 4539 | -0.30% | -1382.94% | -26.051 |
| 0.64 | 4516 | -0.30% | -1373.41% | -25.932 |
| 0.65 | 4503 | -0.30% | -1364.01% | -25.779 |

## 14. Stability and Bootstrap Statistics

Finally, we assess how stable the labeling and model performance are across time (years and time-series folds) and how sensitive key metrics such as positive rate and label–return separation are under bootstrap resampling. Tight confidence intervals indicate robust signal rather than fragile noise.


### Per-Year Label and Return Stability

- **Year 2021:** positive=1.4%, mean_return=-0.33%, AUC=0.500
- **Year 2022:** positive=0.4%, mean_return=-0.32%, AUC=0.562
- **Year 2023:** positive=0.7%, mean_return=-0.32%, AUC=0.552
- **Year 2024:** positive=4.8%, mean_return=-0.29%, AUC=0.479
- **Year 2025:** positive=34.6%, mean_return=-0.28%, AUC=0.485

### Time-Series Fold Stability (Approximate)

- **Fold 1:** positive=0.4%, mean_return=-0.31%, AUC=0.593
- **Fold 2:** positive=0.7%, mean_return=-0.33%, AUC=0.539
- **Fold 3:** positive=0.7%, mean_return=-0.29%, AUC=0.615
- **Fold 4:** positive=15.8%, mean_return=-0.27%, AUC=0.473
- **Fold 5:** positive=34.1%, mean_return=-0.29%, AUC=0.494

### Bootstrap Label Statistics (Labeled Events)

- **Positive rate 95% CI:** [8.7%, 9.3%]
- **Mean return diff (label=1 - label=0) 95% CI:** [1.14%, 1.18%]

---


## Summary and Recommendations


### Key Findings:

1. Label balance: 9.0% positive
2. Signal coverage: 21.6%
3. Mean return (label=1): 0.75%
4. Mean return (label=0): -0.41%
5. Calibration error: 0.320