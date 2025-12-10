# Meta-Labeling Diagnostics Report

**Generated:** 2025-12-10 01:25:00

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

- **Total labeled events:** 36744
- **Positive labels (profitable):** 4942 (13.4%)
- **Negative labels (unprofitable):** 31802 (86.6%)

⚠️ **Warning:** Low positive label rate (13.4%) - most signals are unprofitable

### Label Distribution Over Time

- **Daily positive rate - Mean:** 3.5%
- **Daily positive rate - Std:** 4.4%
- **Daily positive rate - Min:** 0.0%
- **Daily positive rate - Max:** 30.2%

⚠️ **Warning:** 1312 days with <10% positive labels

## 2. Signal Coverage and Sparsity

- **Total samples:** 140354
- **Labeled samples:** 36744
- **Coverage:** 26.2%

✅ **OK:** Reasonable signal coverage (26.2%)

## 3. Feature-Label Correlation Analysis (Post-Filter)


### Top 10 Most Correlated Features (Post-Filter):

- **smc_predicted:** 0.3304
- **risk_score:** 0.3237
- **close_minus_vwap:** 0.1642
- **kalman_close_minus_vwap:** 0.1629
- **close_min_5:** 0.1217
- **close_max_5:** 0.1214
- **close_min_10:** 0.1212
- **close_min_20:** 0.1209
- **close_max_10:** 0.1208
- **close_max_20:** 0.1207

### Correlation Health Check (Post-Filter):


## 3B. Pre- vs Post-Filter Comparison and Feature Shift


### Sample Counts and Retention

- **Pre-filter events (realized_return not NaN):** 43471
- **Pre-filter positive/negative (raw economic):** 10614 / 32857
- **Post-filter labeled events:** 36744
- **Post-filter positive/negative (binary_labels):** 4942 / 31802
- **Total retention (post / pre):** 84.5%
- **Positive retention:** 46.6%
- **Negative retention:** 96.8%

### Pre- vs Post-Filter Signal Quality

- **Pre-filter mean return (label=1/0):** 1.34% / -0.81%
- **Post-filter mean return (label=1/0):** 0.94% / -0.53%
- **Pre-filter Cohen's d (label=1 vs 0):** 5.276
- **Post-filter Cohen's d (label=1 vs 0):** 1.525
- **Pre-filter SNR (mean/std, label=1):** 2.668
- **Post-filter SNR (mean/std, label=1):** 1.224

⚠️ **Warning:** Post-filter effect size is materially worse than pre-filter – filters may be discarding informative events

### Largest Feature Correlation Shifts (|post|-|pre|)

- **smc_predicted:** pre=0.0185, post=0.3304, Δ|corr|=0.3119
- **risk_score:** pre=0.0167, post=0.3237, Δ|corr|=0.3070
- **close_minus_vwap:** pre=0.0111, post=0.1642, Δ|corr|=0.1531
- **kalman_close_minus_vwap:** pre=0.0113, post=0.1629, Δ|corr|=0.1516
- **close_min_5:** pre=0.0264, post=0.1217, Δ|corr|=0.0953
- **close_min_10:** pre=0.0262, post=0.1212, Δ|corr|=0.0950
- **close_max_5:** pre=0.0268, post=0.1214, Δ|corr|=0.0946
- **close_min_20:** pre=0.0264, post=0.1209, Δ|corr|=0.0944
- **close_max_10:** pre=0.0269, post=0.1208, Δ|corr|=0.0940
- **close_max_20:** pre=0.0273, post=0.1207, Δ|corr|=0.0934

## 4. P&L Distribution Per Label

### Label = 1 (Profitable Signals):

- **Count:** 4942
- **Mean return:** 0.94%
- **Median return:** 0.99%
- **Std return:** 0.77%
- **% Actually positive:** 88.6%

### Label = 0 (Unprofitable Signals):

- **Count:** 31802
- **Mean return:** -0.53%
- **Median return:** -1.01%
- **Std return:** 0.99%
- **% Actually positive:** 17.3%

### Labeling Quality:

- **Label overlap:** 16.5%

✅ **OK:** Acceptable label overlap (16.5%)

### Cost-Aware Event Quality Summary

- **Transaction cost (per event, approx):** 0.300%
- **Unconditional mean event return:** -0.33%
- **Mean return (label=1) minus cost:** 0.64%
- **Fraction of labeled events with |return| < cost:** 0.0%

## 5. Time-Series Stability and Regime Analysis

- **Win rate vs Volatility correlation:** 0.0447
- **Mean return vs Volatility correlation:** 0.1199

✅ **OK:** Win rate not strongly correlated with volatility

### Rolling Daily SNR (Return / Std by Day)

- **Daily SNR mean:** -0.4215
- **Daily SNR std:** 0.5428
- **Daily SNR min/max:** -8.9465 / 1.1698

### Regime Shift Detection:

- **30-day rolling correlation (win rate vs vol) - Mean:** -0.0718
- **30-day rolling correlation (win rate vs vol) - Std:** 0.2545

## 6. Model Probability Diagnostics


### Calibration (Predicted Probability vs Actual Success Rate):


| Predicted Prob | Actual Success Rate | Count |
|---------------|---------------------|-------|
| 0.040 | 0.130 | 3264 |
| 0.148 | 0.141 | 2141 |
| 0.249 | 0.128 | 2045 |
| 0.349 | 0.126 | 2181 |
| 0.450 | 0.125 | 2314 |
| 0.515 | 0.081 | 8671 |
| 0.649 | 0.142 | 3032 |
| 0.750 | 0.151 | 3677 |
| 0.849 | 0.153 | 4715 |
| 0.942 | 0.209 | 4704 |

- **Mean calibration error:** 0.374

⚠️ **Warning:** High calibration error - model probabilities may not be well-calibrated

### Monotonicity and Slope Checks (Prob → Realized Return)

- **Monotonicity violations (adjacent bins):** 5 / 9
- **Approx. slope in high-probability region:** 0.000186

### Probability Distribution:

- **Mean probability:** 0.514
- **Median probability:** 0.500
- **Std probability:** 0.145

## 7. Feature Importance Analysis


### Top 20 Features by Importance:

1. **vol_of_vol:** 0.2851
2. **volatility_regime:** 0.1634
3. **liquidity_gap_abs:** 0.0250
4. **kalman_trend_x_vol_ratio:** 0.0224
5. **volatility_4h:** 0.0173
6. **signed_volume_ema:** 0.0122
7. **volatility_20:** 0.0116
8. **is_bad_hour:** 0.0105
9. **momentum_5:** 0.0105
10. **meta_volatility_regime:** 0.0102
11. **sma_slope:** 0.0102
12. **vol_price_corr:** 0.0102
13. **rsi_kalman:** 0.0096
14. **is_sunday:** 0.0094
15. **is_good_hour:** 0.0094
16. **trade_aggressor_ratio:** 0.0093
17. **cvd_proxy:** 0.0092
18. **kalman_trend:** 0.0092
19. **volume_pressure:** 0.0092
20. **volume_trend:** 0.0092

### Feature Importance Health Check:


✅ **OK:** No single feature dominates

✅ Kalman-filtered features are being used (8 features)

✅ Volatility-based features in top 10: vol_of_vol, volatility_regime, kalman_trend_x_vol_ratio, volatility_4h, signed_volume_ema, volatility_20, meta_volatility_regime

## 8. Kalman Smoothed Labels Analysis

- **Mean smoothed label:** 0.134
- **Median smoothed label:** 0.000
- **Std smoothed label:** 0.290
- **Correlation with binary labels:** 0.968
- **Correlation with realized returns:** 0.430

✅ **OK:** Good correlation between smoothed and binary labels

## 9. Event Mechanics and R-Multiples

These diagnostics describe how trades exit (profit, stop, or timeout), how long they stay open, and how large the realized return is relative to the configured stop-loss (R-multiple). They help verify that the triple-barrier / TPSL configuration produces economically meaningful events.


### Exit Reason Mix (Labeled Events)

- **1:** 68.5%
- **0:** 19.4%
- **2:** 12.1%

### Event Duration Distribution (Bars)

- **Mean duration:** 13.17
- **Median duration:** 9.00
- **90th percentile:** 32.00

### R-Multiple Distribution (Return / Stop Threshold)

- **R-multiple (label=1) mean:** 1.35
- **R-multiple (label=1) median:** 1.40
- **R-multiple (label=0) mean:** -0.76
- **R-multiple (label=0) median:** -1.42

## 9B. Enhanced Event Quality Metrics

These advanced metrics help distinguish between efficient momentum capture and random drift trades. They measure path efficiency, timing quality, and entry/exit effectiveness.


### Path Efficiency Ratio (PER)

- **Mean PER (clipped 99.5%):** 0.581
- **Median PER:** 0.608
- **90th percentile PER:** 0.759
- **99th percentile PER:** 0.827

✅ **OK:** Good path efficiency on profitable events

### Time-to-Outcome Ratio (TTO)

- **Mean TTO:** 0.599
- **Median TTO:** 0.409

✅ **OK:** TTO in healthy range [0.4, 0.6]

### MFE/MAE Ratio (Maximum Favorable vs Adverse Excursion)

- **Mean MFE/MAE:** 162.088
- **Median MFE/MAE:** 0.601

✅ **OK:** Favorable excursions exceed adverse excursions

## 9C. Target Volatility Health Check

Verifies that the continuous target values have sufficient variance to train a regression model. If targets are constant or near-zero, the model cannot learn meaningful patterns.

- **Target mean:** 0.004950
- **Target std:** 0.009782
- **Non-zero targets:** 36207 / 140354 (25.8%)

✅ **OK:** Targets have sufficient variance

## 9D. Information Coefficient (IC)

Measures the rank correlation between predicted probabilities and realized returns. This is the purest measure of ranking ability, independent of absolute calibration.

- **Spearman IC (prob, return):** -0.0252
- **P-value:** 1.3071e-06

⚠️ **Warning:** Very weak IC (|IC| < 0.05) - model has minimal ranking ability

## 10. Regime and Time-Conditional Label Checks

These metrics show how label base-rates and realized returns change across volatility regimes, trend states, and time-of-day/weekday. Large differences indicate that learnability and edge are regime-dependent, which is important for downstream model conditioning.


### Label Base-Rate by Volatility Regime

- **Regime high:** positive=16.2%, mean_return=-0.33%
- **Regime low:** positive=10.3%, mean_return=-0.34%
- **Regime medium:** positive=12.8%, mean_return=-0.32%

### Trend-Conditional (Price vs SMA20)

- **Strong uptrend:** positive=15.2%, mean_return=-0.31%
- **Strong downtrend:** positive=12.4%, mean_return=-0.31%

### Time-of-Day Positive Rates (Top/Bottom)

- **Top hours by positive rate:**
  - Hour 3: 17.9%
  - Hour 2: 17.2%
  - Hour 1: 17.0%
- **Bottom hours by positive rate:**
  - Hour 15: 10.3%
  - Hour 16: 10.9%
  - Hour 14: 11.3%

### Day-of-Week Positive Rates

- Day 0: 12.3%
- Day 1: 14.6%
- Day 2: 12.5%
- Day 3: 14.3%
- Day 4: 13.3%
- Day 5: 15.2%
- Day 6: 12.3%

## 11. Label–Return Separation, Information Content, and Sample Size

This section quantifies how well the labels separate profitable from unprofitable events (effect size) and how much information they carry about the sign of future returns compared to a random baseline (mutual information and permutation tests).


### Separation Metrics

- **Mean return difference (label=1 - label=0):** 1.47%
- **Cohen's d effect size:** 1.525
- **Approx. required samples for 80% power (heuristic):** 6.9
- **Current labeled samples used in separation:** 36744.0

### Information Content vs Permutation Baseline

- **Mutual information (labels vs realized sign):** 0.1360
- **Baseline MI (mean over permutations):** 0.0000

## 12. Target and Expected-Return Alignment

Here we check whether the continuous targets produced by the isotonic mapping are consistent with realized returns: higher targets should correspond to higher average realized P&L, and most non-zero targets should exceed transaction costs.


### Target vs Realized Return

- **Correlation (target, realized):** 0.011
- **MSE (target vs realized):** 0.000717

### Target/Return by Target Decile

- Decile 0: target=0.0161, realized=-0.0033
- Decile 1: target=0.0209, realized=-0.0034
- Decile 2: target=0.0220, realized=-0.0035
- Decile 3: target=0.0229, realized=-0.0031
- Decile 4: target=0.0238, realized=-0.0027

### Target Distribution Sanity

- **Non-zero target fraction (all samples):** 25.8%
- **Mean non-zero target:** 0.0211
- **Median non-zero target:** 0.0211
- **Fraction of targets below transaction cost (0.300%):** 0.2%

## 13. Cost-Aware Metrics and Threshold P&L

These diagnostics evaluate the meta-model from a trading perspective: global AUC/Brier/PR-AUC, and how mean and cumulative returns evolve as you raise the probability threshold used to filter trades. This helps choose probability cutoffs that are profitable after costs.


### Cost-Aware Classification Metrics (OOF)

- **AUC:** 0.561
- **Brier score:** 0.3588
- **Average precision (PR-AUC):** 0.180

### Volatility Regime Breakdown

| Regime | Samples | Pos Rate | AUC | Mean Ret | Sharpe |
|--------|---------|----------|-----|----------|--------|
| Low | 9374 | 10.3% | 0.562 | -0.34% | -0.33 |
| Medium | 13678 | 12.8% | 0.555 | -0.32% | -0.29 |
| High | 13672 | 16.2% | 0.574 | -0.33% | -0.30 |

### Threshold-Sweep P&L (Using Meta Probability)


| Threshold | Trades | Mean Return | Cum Return | Sharpe (per trade) |
|----------|--------|-------------|------------|---------------------|
| 0.50 | 24772 | -0.33% | -8154.94% | -47.782 |
| 0.51 | 18450 | -0.33% | -6114.62% | -42.029 |
| 0.52 | 18205 | -0.33% | -6027.04% | -41.702 |
| 0.53 | 17973 | -0.33% | -5918.80% | -41.164 |
| 0.54 | 17722 | -0.33% | -5858.52% | -41.061 |
| 0.55 | 17442 | -0.33% | -5776.62% | -40.819 |
| 0.56 | 17176 | -0.33% | -5679.61% | -40.430 |
| 0.57 | 16893 | -0.33% | -5581.24% | -40.058 |
| 0.58 | 16619 | -0.33% | -5464.43% | -39.519 |
| 0.59 | 16355 | -0.33% | -5382.50% | -39.245 |
| 0.60 | 16092 | -0.33% | -5310.47% | -39.058 |
| 0.61 | 15815 | -0.33% | -5212.73% | -38.651 |
| 0.62 | 15501 | -0.33% | -5109.42% | -38.281 |
| 0.63 | 15217 | -0.33% | -5047.89% | -38.208 |
| 0.64 | 14891 | -0.33% | -4945.09% | -37.861 |
| 0.65 | 14593 | -0.33% | -4834.84% | -37.381 |

## 14. Stability and Bootstrap Statistics

Finally, we assess how stable the labeling and model performance are across time (years and time-series folds) and how sensitive key metrics such as positive rate and label–return separation are under bootstrap resampling. Tight confidence intervals indicate robust signal rather than fragile noise.


### Per-Year Label and Return Stability

- **Year 2021:** positive=7.0%, mean_return=-0.35%, AUC=0.500
- **Year 2022:** positive=4.9%, mean_return=-0.31%, AUC=0.518
- **Year 2023:** positive=7.6%, mean_return=-0.38%, AUC=0.562
- **Year 2024:** positive=10.6%, mean_return=-0.32%, AUC=0.522
- **Year 2025:** positive=34.0%, mean_return=-0.31%, AUC=0.524

### Time-Series Fold Stability (Approximate)

- **Fold 1:** positive=5.2%, mean_return=-0.34%, AUC=0.543
- **Fold 2:** positive=7.8%, mean_return=-0.40%, AUC=0.559
- **Fold 3:** positive=7.5%, mean_return=-0.32%, AUC=0.591
- **Fold 4:** positive=19.9%, mean_return=-0.30%, AUC=0.533
- **Fold 5:** positive=33.4%, mean_return=-0.32%, AUC=0.539

### Bootstrap Label Statistics (Labeled Events)

- **Positive rate 95% CI:** [13.1%, 13.7%]
- **Mean return diff (label=1 - label=0) 95% CI:** [1.44%, 1.49%]

---


## Summary and Recommendations


### Key Findings:

1. Label balance: 13.4% positive
2. Signal coverage: 26.2%
3. Mean return (label=1): 0.94%
4. Mean return (label=0): -0.53%
5. Calibration error: 0.374