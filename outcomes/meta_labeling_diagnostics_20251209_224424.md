# Meta-Labeling Diagnostics Report

**Generated:** 2025-12-09 22:44:24

**Symbol:** ETHUSDT
**Timeframe:** 15m
**Horizon:** 26 bars

**Final selected features (count):** 
98 features


**Final selected features (full list):**

- log_ret
- volatility_1h
- volatility_4h
- volatility_1d
- vol_of_vol
- risk_score
- mr_probability_dense
- sr_labeling_xgb_prob
- smc_predicted
- kalman_trend
- kalman_uncertainty
- rsi_kalman
- ma_distance_kalman
- momentum_kalman
- momentum_per_vol
- ma_distance_per_vol
- volatility_5
- volatility_20
- volatility_ratio
- volatility_ema
- sma_slope
- price_vs_sma20
- atr_14
- volume_ratio
- volume_trend
- vol_price_corr
- volume_zscore
- signed_volume_ema
- momentum_5
- momentum_10
- momentum_20
- return_autocorr_lag1_w50
- range_position
- close_minus_vwap
- hour
- day_of_week
- hour_sin
- hour_cos
- is_good_hour
- is_bad_hour
- is_sunday
- cvd_proxy
- volume_pressure
- ofi_proxy
- volume_imbalance
- absorption_ratio
- liquidity_gap_up
- liquidity_gap_down
- liquidity_gap_abs
- kalman_trend_x_vol_ratio
- range_position_x_vol_ratio
- signal_active
- signal_strength_all
- signal_rsi_macd_alignment
- signal_rsi_distance_50
- signal_rsi_long_distance_50
- signal_macd_hist_abs
- signal_macd_hist_long_abs
- trend_regime
- candle_trend
- candle_reversal
- signal_trend_regime_x_macd_hist_abs
- signal_candle_trend_x_rsi_distance_50
- returns_1h
- volatility_1h_agg
- range_1h
- returns_4h
- volatility_4h_agg
- range_4h
- close_range_5
- dist_from_recent_high_5
- dist_from_recent_low_5
- returns_std_10
- close_range_10
- dist_from_recent_high_10
- dist_from_recent_low_10
- close_min_20
- close_range_20
- dist_from_recent_high_20
- dist_from_recent_low_20
- returns_mean_50
- close_min_50
- close_max_50
- close_range_50
- dist_from_recent_high_50
- dist_from_recent_low_50
- momentum_5_x_regime_high
- momentum_5_x_regime_medium
- momentum_10_x_regime_high
- momentum_10_x_regime_medium
- momentum_20_x_regime_high
- momentum_20_x_regime_medium
- high_dist_x_vol
- low_dist_x_vol
- drawdown_100
- return_26b
- return_std_26b
- sma_slope_26b

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

- **Total labeled events:** 8132
- **Positive labels (profitable):** 5066 (62.3%)
- **Negative labels (unprofitable):** 3066 (37.7%)

✅ **OK:** Reasonable label balance (62.3%)

### Label Distribution Over Time

- **Daily positive rate - Mean:** 3.6%
- **Daily positive rate - Std:** 3.9%
- **Daily positive rate - Min:** 0.0%
- **Daily positive rate - Max:** 22.9%

⚠️ **Warning:** 1341 days with <10% positive labels

## 2. Signal Coverage and Sparsity

- **Total samples:** 140354
- **Labeled samples:** 8132
- **Coverage:** 5.8%

✅ **OK:** Reasonable signal coverage (5.8%)

## 3. Feature-Label Correlation Analysis (Post-Filter)


### Top 10 Most Correlated Features (Post-Filter):

- **mr_probability_dense:** -0.4699
- **smc_predicted:** -0.4432
- **risk_score:** -0.4196
- **signal_disagreement_ema:** -0.3379
- **signal_disagreement:** -0.3334
- **event_tto_mean_last_50:** 0.3108
- **volatility_1d:** 0.2200
- **close_minus_vwap:** -0.2185
- **close_min_50:** -0.1972
- **close_max_50:** -0.1929

### Correlation Health Check (Post-Filter):


## 3B. Pre- vs Post-Filter Comparison and Feature Shift


### Sample Counts and Retention

- **Pre-filter events (realized_return not NaN):** 43470
- **Pre-filter positive/negative (raw economic):** 11800 / 31670
- **Post-filter labeled events:** 8132
- **Post-filter positive/negative (binary_labels):** 5066 / 3066
- **Total retention (post / pre):** 18.7%
- **Positive retention:** 42.9%
- **Negative retention:** 9.7%

### Pre- vs Post-Filter Signal Quality

- **Pre-filter mean return (label=1/0):** 1.46% / -0.93%
- **Post-filter mean return (label=1/0):** 1.25% / -1.19%
- **Pre-filter Cohen's d (label=1 vs 0):** 4.786
- **Post-filter Cohen's d (label=1 vs 0):** 5.625
- **Pre-filter SNR (mean/std, label=1):** 2.506
- **Post-filter SNR (mean/std, label=1):** 2.303

### Largest Feature Correlation Shifts (|post|-|pre|)

- **mr_probability_dense:** pre=0.0238, post=-0.4699, Δ|corr|=0.4461
- **smc_predicted:** pre=0.0244, post=-0.4432, Δ|corr|=0.4188
- **risk_score:** pre=0.0211, post=-0.4196, Δ|corr|=0.3985
- **signal_disagreement_ema:** pre=0.0122, post=-0.3379, Δ|corr|=0.3257
- **signal_disagreement:** pre=0.1048, post=-0.3334, Δ|corr|=0.2286
- **close_minus_vwap:** pre=0.0160, post=-0.2185, Δ|corr|=0.2025
- **volatility_1d:** pre=0.0308, post=0.2200, Δ|corr|=0.1892
- **close_min_50:** pre=0.0269, post=-0.1972, Δ|corr|=0.1703
- **close_max_50:** pre=0.0282, post=-0.1929, Δ|corr|=0.1648
- **close_min_20:** pre=0.0284, post=-0.1904, Δ|corr|=0.1620

## 4. P&L Distribution Per Label

### Label = 1 (Profitable Signals):

- **Count:** 5066
- **Mean return:** 1.25%
- **Median return:** 1.14%
- **Std return:** 0.54%
- **% Actually positive:** 99.6%

### Label = 0 (Unprofitable Signals):

- **Count:** 3066
- **Mean return:** -1.19%
- **Median return:** -1.24%
- **Std return:** 0.10%
- **% Actually positive:** 0.0%

### Labeling Quality:

- **Label overlap:** 0.3%

✅ **OK:** Acceptable label overlap (0.3%)

### Cost-Aware Event Quality Summary

- **Transaction cost (per event, approx):** 0.300%
- **Unconditional mean event return:** 0.33%
- **Mean return (label=1) minus cost:** 0.95%
- **Fraction of labeled events with |return| < cost:** 0.0%

## 5. Time-Series Stability and Regime Analysis

- **Win rate vs Volatility correlation:** 0.0321
- **Mean return vs Volatility correlation:** 0.1187

✅ **OK:** Win rate not strongly correlated with volatility

### Rolling Daily SNR (Return / Std by Day)

- **Daily SNR mean:** -0.3680
- **Daily SNR std:** 0.4795
- **Daily SNR min/max:** -2.6759 / 1.2543

### Regime Shift Detection:

- **30-day rolling correlation (win rate vs vol) - Mean:** 0.4167
- **30-day rolling correlation (win rate vs vol) - Std:** 0.1421

## 6. Model Probability Diagnostics


### Calibration (Predicted Probability vs Actual Success Rate):


| Predicted Prob | Actual Success Rate | Count |
|---------------|---------------------|-------|
| 0.029 | 0.330 | 736 |
| 0.150 | 0.369 | 385 |
| 0.249 | 0.376 | 423 |
| 0.348 | 0.398 | 342 |
| 0.450 | 0.457 | 339 |
| 0.504 | 0.746 | 4403 |
| 0.648 | 0.489 | 313 |
| 0.752 | 0.571 | 312 |
| 0.852 | 0.607 | 318 |
| 0.957 | 0.750 | 561 |

- **Mean calibration error:** 0.174

⚠️ **Warning:** High calibration error - model probabilities may not be well-calibrated

### Monotonicity and Slope Checks (Prob → Realized Return)

- **Monotonicity violations (adjacent bins):** 1 / 9
- **Approx. slope in high-probability region:** 0.008082

### Probability Distribution:

- **Mean probability:** 0.499
- **Median probability:** 0.500
- **Std probability:** 0.056

## 7. Feature Importance Analysis


### Top 20 Features by Importance:

1. **vol_ratio:** 0.2049
2. **vol_regime_high:** 0.1189
3. **dist_from_recent_low_10:** 0.0608
4. **dist_from_recent_high_10:** 0.0603
5. **close_min_5:** 0.0341
6. **returns_mean_5:** 0.0275
7. **close_range_5:** 0.0262
8. **volatility_regime:** 0.0241
9. **volatility_1d:** 0.0206
10. **atr_14:** 0.0162
11. **signal_rsi_long_distance_50:** 0.0105
12. **signal_candle_trend_x_rsi_distance_50:** 0.0094
13. **volatility_ratio:** 0.0092
14. **dist_from_recent_high_5:** 0.0088
15. **signal_strength_all:** 0.0083
16. **momentum_kalman:** 0.0081
17. **close_max_5:** 0.0079
18. **returns_mean_10:** 0.0076
19. **cvd_proxy:** 0.0075
20. **ma_distance_per_vol:** 0.0075

### Feature Importance Health Check:


✅ **OK:** No single feature dominates

✅ Kalman-filtered features are being used (6 features)

✅ Volatility-based features in top 10: vol_ratio, vol_regime_high, volatility_regime, volatility_1d

## 8. Kalman Smoothed Labels Analysis

- **Mean smoothed label:** 0.625
- **Median smoothed label:** 0.691
- **Std smoothed label:** 0.359
- **Correlation with binary labels:** 0.822
- **Correlation with realized returns:** 0.710

✅ **OK:** Good correlation between smoothed and binary labels

## 9. Event Mechanics and R-Multiples

These diagnostics describe how trades exit (profit, stop, or timeout), how long they stay open, and how large the realized return is relative to the configured stop-loss (R-multiple). They help verify that the triple-barrier / TPSL configuration produces economically meaningful events.


### Exit Reason Mix (Labeled Events)

- **2:** 39.1%
- **1:** 37.3%
- **0:** 23.7%

### Event Duration Distribution (Bars)

- **Mean duration:** 22.17
- **Median duration:** 19.00
- **90th percentile:** 45.00

### R-Multiple Distribution (Return / Stop Threshold)

- **R-multiple (label=1) mean:** 1.41
- **R-multiple (label=1) median:** 1.27
- **R-multiple (label=0) mean:** -1.34
- **R-multiple (label=0) median:** -1.32

## 9B. Enhanced Event Quality Metrics

These advanced metrics help distinguish between efficient momentum capture and random drift trades. They measure path efficiency, timing quality, and entry/exit effectiveness.


### Path Efficiency Ratio (PER)

- **Mean PER (clipped 99.5%):** 0.517
- **Median PER:** 0.516
- **90th percentile PER:** 0.738
- **99th percentile PER:** 0.831

✅ **OK:** Good path efficiency on profitable events

### Time-to-Outcome Ratio (TTO)

- **Mean TTO:** 0.853
- **Median TTO:** 0.731

⚠️ **Warning:** TTO outside target range [0.4, 0.6]

### MFE/MAE Ratio (Maximum Favorable vs Adverse Excursion)

- **Mean MFE/MAE:** 304.024
- **Median MFE/MAE:** 3.132

✅ **OK:** Favorable excursions exceed adverse excursions

## 9C. Target Volatility Health Check

Verifies that the continuous target values have sufficient variance to train a regression model. If targets are constant or near-zero, the model cannot learn meaningful patterns.

- **Target mean:** -0.000453
- **Target std:** 0.001394
- **Non-zero targets:** 1640 / 140354 (1.2%)

✅ **OK:** Targets have sufficient variance

## 9D. Information Coefficient (IC)

Measures the rank correlation between predicted probabilities and realized returns. This is the purest measure of ranking ability, independent of absolute calibration.

- **Spearman IC (prob, return):** 0.1631
- **P-value:** 1.3242e-49

✅ **OK:** Meaningful rank correlation

## 10. Regime and Time-Conditional Label Checks

These metrics show how label base-rates and realized returns change across volatility regimes, trend states, and time-of-day/weekday. Large differences indicate that learnability and edge are regime-dependent, which is important for downstream model conditioning.


### Label Base-Rate by Volatility Regime

- **Regime high:** positive=78.5%, mean_return=0.75%
- **Regime low:** positive=55.8%, mean_return=0.12%
- **Regime medium:** positive=54.3%, mean_return=0.15%

### Trend-Conditional (Price vs SMA20)

- **Strong uptrend:** positive=62.1%, mean_return=0.33%
- **Strong downtrend:** positive=56.7%, mean_return=0.26%

### Time-of-Day Positive Rates (Top/Bottom)

- **Top hours by positive rate:**
  - Hour 2: 71.4%
  - Hour 4: 71.0%
  - Hour 3: 70.4%
- **Bottom hours by positive rate:**
  - Hour 18: 52.3%
  - Hour 15: 53.5%
  - Hour 13: 54.5%

### Day-of-Week Positive Rates

- Day 0: 54.5%
- Day 1: 66.4%
- Day 2: 63.4%
- Day 3: 68.9%
- Day 4: 68.0%
- Day 5: 68.6%
- Day 6: 50.6%

## 11. Label–Return Separation, Information Content, and Sample Size

This section quantifies how well the labels separate profitable from unprofitable events (effect size) and how much information they carry about the sign of future returns compared to a random baseline (mutual information and permutation tests).


### Separation Metrics

- **Mean return difference (label=1 - label=0):** 2.44%
- **Cohen's d effect size:** 5.625
- **Approx. required samples for 80% power (heuristic):** 0.5
- **Current labeled samples used in separation:** 8132.0

### Information Content vs Permutation Baseline

- **Mutual information (labels vs realized sign):** 0.6471
- **Baseline MI (mean over permutations):** 0.0001

## 12. Target and Expected-Return Alignment

Here we check whether the continuous targets produced by the isotonic mapping are consistent with realized returns: higher targets should correspond to higher average realized P&L, and most non-zero targets should exceed transaction costs.


### Target vs Realized Return

- **Correlation (target, realized):** 0.100
- **MSE (target vs realized):** 0.000212

### Target/Return by Target Decile

- Decile 0: target=0.0016, realized=0.0020
- Decile 1: target=0.0036, realized=0.0020
- Decile 2: target=0.0081, realized=0.0058
- Decile 3: target=0.0106, realized=0.0059
- Decile 4: target=0.0112, realized=0.0062
- Decile 5: target=0.0117, realized=0.0046
- Decile 6: target=0.0121, realized=0.0023

### Target Distribution Sanity

- **Non-zero target fraction (all samples):** 1.2%
- **Mean non-zero target:** 0.0090
- **Median non-zero target:** 0.0108
- **Fraction of targets below transaction cost (0.300%):** 15.4%

## 13. Cost-Aware Metrics and Threshold P&L

These diagnostics evaluate the meta-model from a trading perspective: global AUC/Brier/PR-AUC, and how mean and cumulative returns evolve as you raise the probability threshold used to filter trades. This helps choose probability cutoffs that are profitable after costs.


### Cost-Aware Classification Metrics (OOF)

- **AUC:** 0.614
- **Brier score:** 0.2594
- **Average precision (PR-AUC):** 0.710

### Volatility Regime Breakdown

| Regime | Samples | Pos Rate | AUC | Mean Ret | Sharpe |
|--------|---------|----------|-----|----------|--------|
| Low | 1969 | 55.8% | 0.615 | 0.12% | 0.10 |
| Medium | 3607 | 54.3% | 0.598 | 0.15% | 0.12 |
| High | 2540 | 78.5% | 0.515 | 0.75% | 0.65 |

### Threshold-Sweep P&L (Using Meta Probability)


| Threshold | Trades | Mean Return | Cum Return | Sharpe (per trade) |
|----------|--------|-------------|------------|---------------------|
| 0.50 | 5906 | 0.51% | 3014.32% | 33.772 |
| 0.51 | 1790 | 0.44% | 779.44% | 13.183 |
| 0.52 | 1757 | 0.44% | 779.92% | 13.320 |
| 0.53 | 1723 | 0.46% | 789.48% | 13.638 |
| 0.54 | 1696 | 0.46% | 786.72% | 13.721 |
| 0.55 | 1662 | 0.47% | 784.95% | 13.863 |
| 0.56 | 1631 | 0.48% | 790.87% | 14.123 |
| 0.57 | 1599 | 0.49% | 781.87% | 14.102 |
| 0.58 | 1566 | 0.50% | 776.68% | 14.171 |
| 0.59 | 1529 | 0.50% | 765.38% | 14.152 |
| 0.60 | 1503 | 0.51% | 760.89% | 14.205 |
| 0.61 | 1467 | 0.51% | 753.17% | 14.265 |
| 0.62 | 1435 | 0.53% | 760.23% | 14.590 |
| 0.63 | 1398 | 0.54% | 748.95% | 14.594 |
| 0.64 | 1378 | 0.54% | 744.92% | 14.644 |
| 0.65 | 1342 | 0.56% | 753.49% | 15.082 |

## 14. Stability and Bootstrap Statistics

Finally, we assess how stable the labeling and model performance are across time (years and time-series folds) and how sensitive key metrics such as positive rate and label–return separation are under bootstrap resampling. Tight confidence intervals indicate robust signal rather than fragile noise.


### Per-Year Label and Return Stability

- **Year 2021:** positive=100.0%, mean_return=0.92%, AUC=nan
- **Year 2022:** positive=100.0%, mean_return=0.91%, AUC=nan
- **Year 2023:** positive=100.0%, mean_return=0.85%, AUC=nan
- **Year 2024:** positive=70.1%, mean_return=0.41%, AUC=0.500
- **Year 2025:** positive=48.5%, mean_return=0.14%, AUC=0.635

### Time-Series Fold Stability (Approximate)

- **Fold 1:** positive=100.0%, mean_return=0.88%, AUC=nan
- **Fold 2:** positive=100.0%, mean_return=0.84%, AUC=nan
- **Fold 3:** positive=100.0%, mean_return=0.90%, AUC=nan
- **Fold 4:** positive=56.0%, mean_return=0.23%, AUC=0.500
- **Fold 5:** positive=47.7%, mean_return=0.12%, AUC=0.653

### Bootstrap Label Statistics (Labeled Events)

- **Positive rate 95% CI:** [61.3%, 63.5%]
- **Mean return diff (label=1 - label=0) 95% CI:** [2.43%, 2.45%]

---


## Summary and Recommendations


### Key Findings:

1. Label balance: 62.3% positive
2. Signal coverage: 5.8%
3. Mean return (label=1): 1.25%
4. Mean return (label=0): -1.19%
5. Calibration error: 0.174