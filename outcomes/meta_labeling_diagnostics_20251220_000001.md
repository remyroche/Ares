# Meta-Labeling Diagnostics Report

**Generated:** 2025-12-20 00:00:01

**Symbol:** ETHUSDT
**Timeframe:** 15m
**Horizon:** 22 bars

**Final selected features (count):** 
53 features


**Final selected features (full list):**

- absorption_ratio
- atr_14
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
- kaufman_efficiency_ratio
- liquidity_gap_abs
- liquidity_gap_down
- log_ret
- low_dist_x_vol
- ma_distance_per_vol
- meta_autocorr
- meta_efficiency
- meta_trendiness
- meta_volume_shock
- meta_volume_shock_x_hour_cos
- meta_volume_shock_x_hour_sin
- momentum_10_x_regime_high
- momentum_20
- momentum_20_x_regime_high
- momentum_20_x_regime_medium
- momentum_30
- momentum_5_x_regime_high
- momentum_per_vol
- ofi_proxy
- range_4h
- range_position
- return_autocorr_lag1_w50
- returns_1h
- returns_mean_50
- returns_std_10
- rolling_sharpe
- signed_volume_ema
- sma_slope_22b
- smc_predicted
- vol_price_corr
- volatility_4h_agg
- volatility_5
- volatility_ema
- volume_pressure
- volume_trend
- volume_zscore

---


## 0. Signal Funnel (Primary Signals)

- **Total bars:** 142619
- **Raw non-zero signals:** 0
- **Final consensus signals:** 90248 (ratio=0.000)
- **Raw long/short:** 0/0
- **Final long/short:** 0/0
- **Relaxed extra signals (strict=0 but raw≠0):** 0
- **Raw signal density:** 0.00000 per bar
- **Final signal density:** 0.63279 per bar
- **Raw signals per day (approx):** 0.000
- **Final consensus signals per day (approx):** 60.097

⚠️ **Warning:** Very dense primary signals (60.097 trades/day). Overlapping events may be frequent.

## 1. Label Distribution Analysis

- **Total labeled events:** 38349
- **Positive labels (profitable):** 24 (0.1%)
- **Negative labels (unprofitable):** 38325 (99.9%)

⚠️ **Warning:** Low positive label rate (0.1%) - most signals are unprofitable

### Label Distribution Over Time

- **Daily positive rate - Mean:** 0.0%
- **Daily positive rate - Std:** 0.5%
- **Daily positive rate - Min:** 0.0%
- **Daily positive rate - Max:** 18.8%

⚠️ **Warning:** 1488 days with <10% positive labels

## 2. Signal Coverage and Sparsity

- **Total samples:** 142619
- **Labeled samples:** 38349
- **Coverage:** 26.9%

✅ **OK:** Reasonable signal coverage (26.9%)

## 3. Feature-Label Correlation Analysis (Post-Filter)


### Top 10 Most Correlated Features (Post-Filter):

- **kalman_uncertainty:** -0.3066
- **event_tto_mean_last_50:** 0.1364
- **event_intensity_96:** 0.0455
- **momentum_20_x_regime_high:** 0.0258
- **cvd_proxy:** 0.0256
- **close_minus_vwap:** 0.0247
- **day_of_week:** -0.0243
- **volume_at_high_20:** 0.0241
- **range_4h:** -0.0240
- **pullback_depth_50:** -0.0229

### Correlation Health Check (Post-Filter):


## 3B. Pre- vs Post-Filter Comparison and Feature Shift


### Sample Counts and Retention

- **Pre-filter events (realized_return not NaN):** 38350
- **Pre-filter positive/negative (raw economic):** 20 / 38330
- **Post-filter labeled events:** 38349
- **Post-filter positive/negative (binary_labels):** 24 / 38325
- **Total retention (post / pre):** 100.0%
- **Positive retention:** 120.0%
- **Negative retention:** 100.0%

### Pre- vs Post-Filter Signal Quality

- **Pre-filter mean return (label=1/0):** 1.19% / -0.31%
- **Post-filter mean return (label=1/0):** 1.03% / -0.31%
- **Pre-filter Cohen's d (label=1 vs 0):** 4.257
- **Post-filter Cohen's d (label=1 vs 0):** 3.783
- **Pre-filter SNR (mean/std, label=1):** 1.335
- **Post-filter SNR (mean/std, label=1):** 1.142

⚠️ **Warning:** Post-filter effect size is materially worse than pre-filter – filters may be discarding informative events

### Largest Feature Correlation Shifts (|post|-|pre|)

- **kalman_uncertainty:** pre=-0.2807, post=-0.3066, Δ|corr|=0.0259
- **event_tto_mean_last_50:** pre=0.1228, post=0.1364, Δ|corr|=0.0136
- **hurst_kalman:** pre=-0.0061, post=-0.0008, Δ|corr|=-0.0053
- **sma_slope_22b:** pre=0.0123, post=0.0157, Δ|corr|=0.0034
- **close_min_10:** pre=0.0146, post=0.0179, Δ|corr|=0.0033
- **rolling_sharpe:** pre=0.0143, post=0.0176, Δ|corr|=0.0033
- **close_min_5:** pre=0.0180, post=0.0212, Δ|corr|=0.0032
- **momentum_30:** pre=0.0149, post=0.0182, Δ|corr|=0.0032
- **volume_x_rv_48:** pre=0.0044, post=0.0013, Δ|corr|=-0.0031
- **close_minus_vwap:** pre=0.0216, post=0.0247, Δ|corr|=0.0031

## 4. P&L Distribution Per Label

### Label = 1 (Profitable Signals):

- **Count:** 24
- **Mean return:** 1.03%
- **Median return:** 0.79%
- **Std return:** 0.90%
- **% Actually positive:** 100.0%

### Label = 0 (Unprofitable Signals):

- **Count:** 38325
- **Mean return:** -0.31%
- **Median return:** -0.17%
- **Std return:** 0.35%
- **% Actually positive:** 0.0%

### Labeling Quality:

- **Label overlap:** 0.0%

✅ **OK:** Acceptable label overlap (0.0%)

### Cost-Aware Event Quality Summary

- **Transaction cost (per event, approx):** 0.300%
- **Unconditional mean event return:** -0.31%
- **Mean return (label=1) minus cost:** 0.73%
- **Fraction of labeled events with |return| < cost:** 83.7%

## 5. Time-Series Stability and Regime Analysis

- **Win rate vs Volatility correlation:** 0.0042
- **Mean return vs Volatility correlation:** -0.1267

✅ **OK:** Win rate not strongly correlated with volatility

### Rolling Daily SNR (Return / Std by Day)

- **Daily SNR mean:** -1.0944
- **Daily SNR std:** 0.9689
- **Daily SNR min/max:** -13.2658 / 0.6270

### Regime Shift Detection:

- **30-day rolling correlation (win rate vs vol) - Mean:** -0.0594
- **30-day rolling correlation (win rate vs vol) - Std:** 0.0182

## 6. Model Probability Diagnostics


### Calibration (Predicted Probability vs Actual Success Rate):


| Predicted Prob | Actual Success Rate | Count |
|---------------|---------------------|-------|
| 0.000 | 0.000 | 31804 |
| 0.162 | 0.000 | 5 |
| 0.225 | 0.000 | 1 |
| 0.354 | 0.000 | 4 |
| 0.457 | 0.000 | 7 |
| 0.500 | 0.004 | 6518 |
| 0.647 | 0.000 | 6 |
| 0.726 | 0.000 | 2 |
| 0.812 | 0.000 | 1 |
| 0.969 | 0.000 | 1 |

- **Mean calibration error:** 0.485

⚠️ **Warning:** High calibration error - model probabilities may not be well-calibrated

### Monotonicity and Slope Checks (Prob → Realized Return)

- **Monotonicity violations (adjacent bins):** 2 / 9
- **Approx. slope in high-probability region:** 0.005984

### Probability Distribution:

- **Mean probability:** 0.388
- **Median probability:** 0.500
- **Std probability:** 0.208

## 7. Feature Importance Analysis


### Top 20 Features by Importance:

1. **momentum_ema:** 0.1357
2. **ma_distance_kalman:** 0.0869
3. **volume_ratio:** 0.0776
4. **momentum_per_vol:** 0.0728
5. **trend_x_participation_48:** 0.0724
6. **momentum_20:** 0.0644
7. **smc_predicted:** 0.0518
8. **kalman_range_position:** 0.0424
9. **returns_entropy:** 0.0406
10. **macro_trend_score_continuous:** 0.0400
11. **event_intensity_96:** 0.0303
12. **volume_spike:** 0.0275
13. **volume_x_rv_48:** 0.0255
14. **momentum_10:** 0.0227
15. **volatility_5:** 0.0203
16. **kalman_uncertainty:** 0.0200
17. **vol_move_confirmation:** 0.0170
18. **volume_trend:** 0.0120
19. **hour:** 0.0097
20. **signed_volume_ema:** 0.0088

### Feature Importance Health Check:


✅ **OK:** No single feature dominates

✅ Kalman-filtered features are being used (6 features)

✅ Volatility-based features in top 10: volume_ratio, momentum_per_vol

## 8. Kalman Smoothed Labels Analysis

- **Mean smoothed label:** 0.001
- **Median smoothed label:** 0.000
- **Std smoothed label:** 0.024
- **Correlation with binary labels:** 0.978
- **Correlation with realized returns:** 0.089

✅ **OK:** Good correlation between smoothed and binary labels

## 9. Event Mechanics and R-Multiples

These diagnostics describe how trades exit (profit, stop, or timeout), how long they stay open, and how large the realized return is relative to the configured stop-loss (R-multiple). They help verify that the triple-barrier / TPSL configuration produces economically meaningful events.


### Exit Reason Mix (Labeled Events)

- **0:** 83.6%
- **1:** 15.1%
- **2:** 1.2%

### Event Duration Distribution (Bars)

- **Mean duration:** 3.63
- **Median duration:** 1.00
- **90th percentile:** 9.00

### R-Multiple Distribution (Return / Stop Threshold)

- **R-multiple (label=1) mean:** nan
- **R-multiple (label=1) median:** nan
- **R-multiple (label=0) mean:** -0.38
- **R-multiple (label=0) median:** -0.19

## 9B. Enhanced Event Quality Metrics

These advanced metrics help distinguish between efficient momentum capture and random drift trades. They measure path efficiency, timing quality, and entry/exit effectiveness.


### Path Efficiency Ratio (PER)

- **Mean PER (clipped 99.5%):** 0.388
- **Median PER:** 0.388
- **90th percentile PER:** 0.709
- **99th percentile PER:** 0.853

⚠️ **Warning:** Median PER in [0.3, 0.5] – many winners meander significantly before paying off

### Time-to-Outcome Ratio (TTO)

- **Mean TTO:** 0.165
- **Median TTO:** 0.045

⚠️ **Warning:** TTO outside target range [0.4, 0.6]

### MFE/MAE Ratio (Maximum Favorable vs Adverse Excursion)

- **Mean MFE/MAE:** 170.588
- **Median MFE/MAE:** 1.332

✅ **OK:** Favorable excursions exceed adverse excursions

## 9C. Target Volatility Health Check

Verifies that the continuous target values have sufficient variance to train a regression model. If targets are constant or near-zero, the model cannot learn meaningful patterns.

- **Target mean:** 0.000008
- **Target std:** 0.001266
- **Non-zero targets:** 10 / 142619 (0.0%)

⚠️ **Warning:** Very few non-zero targets (0.0%)

## 9D. Information Coefficient (IC)

Measures the rank correlation between predicted probabilities and realized returns. This is the purest measure of ranking ability, independent of absolute calibration.

- **Spearman IC (prob, return):** 0.0153
- **P-value:** 2.6635e-03

⚠️ **Warning:** Very weak IC (|IC| < 0.05) - model has minimal ranking ability

## 10. Regime and Time-Conditional Label Checks

These metrics show how label base-rates and realized returns change across volatility regimes, trend states, and time-of-day/weekday. Large differences indicate that learnability and edge are regime-dependent, which is important for downstream model conditioning.


### Label Base-Rate by Volatility Regime

- **Regime high:** positive=0.2%, mean_return=-0.32%
- **Regime low:** positive=0.0%, mean_return=-0.31%
- **Regime medium:** positive=0.0%, mean_return=-0.31%

### Trend-Conditional (Price vs SMA20)

- **Strong uptrend:** positive=0.2%, mean_return=-0.31%
- **Strong downtrend:** positive=0.0%, mean_return=-0.31%

### Time-of-Day Positive Rates (Top/Bottom)

- **Top hours by positive rate:**
  - Hour 9: 0.2%
  - Hour 5: 0.1%
  - Hour 1: 0.1%
- **Bottom hours by positive rate:**
  - Hour 23: 0.0%
  - Hour 21: 0.0%
  - Hour 2: 0.0%

### Day-of-Week Positive Rates

- Day 0: 0.1%
- Day 1: 0.3%
- Day 2: 0.0%
- Day 3: 0.0%
- Day 4: 0.0%
- Day 5: 0.0%
- Day 6: 0.0%

## 11. Label–Return Separation, Information Content, and Sample Size

This section quantifies how well the labels separate profitable from unprofitable events (effect size) and how much information they carry about the sign of future returns compared to a random baseline (mutual information and permutation tests).


### Separation Metrics

- **Mean return difference (label=1 - label=0):** 1.34%
- **Cohen's d effect size:** 3.783
- **Approx. required samples for 80% power (heuristic):** 1.1
- **Current labeled samples used in separation:** 38349.0

### Information Content vs Permutation Baseline

- **Mutual information (labels vs realized sign):** 0.0052
- **Baseline MI (mean over permutations):** 0.0000

## 12. Target and Expected-Return Alignment

Here we check whether the continuous targets produced by the isotonic mapping are consistent with realized returns: higher targets should correspond to higher average realized P&L, and most non-zero targets should exceed transaction costs.


### Target vs Realized Return

- **Correlation (target, realized):** 0.281
- **MSE (target vs realized):** 0.023233

### Target/Return by Target Decile

- Decile 0: target=0.0292, realized=-0.0121
- Decile 1: target=0.0303, realized=-0.0017
- Decile 2: target=0.0465, realized=-0.0013
- Decile 3: target=0.0497, realized=-0.0013
- Decile 4: target=0.0557, realized=-0.0013
- Decile 5: target=0.0710, realized=-0.0013
- Decile 6: target=0.1023, realized=-0.0019
- Decile 7: target=0.1501, realized=-0.0013
- Decile 8: target=0.2115, realized=-0.0013
- Decile 9: target=0.3691, realized=-0.0013

### Target Distribution Sanity

- **Non-zero target fraction (all samples):** 0.0%
- **Mean non-zero target:** 0.1115
- **Median non-zero target:** 0.0634
- **Fraction of targets below transaction cost (0.300%):** 0.0%

## 13. Cost-Aware Metrics and Threshold P&L

These diagnostics evaluate the meta-model from a trading perspective: global AUC/Brier/PR-AUC, and how mean and cumulative returns evolve as you raise the probability threshold used to filter trades. This helps choose probability cutoffs that are profitable after costs.


### Cost-Aware Classification Metrics (OOF)

- **AUC:** 0.915
- **Brier score:** 0.0427
- **Average precision (PR-AUC):** 0.004

### Volatility Regime Breakdown

| Regime | Samples | Pos Rate | AUC | Mean Ret | Sharpe |
|--------|---------|----------|-----|----------|--------|
| Low | 10565 | 0.0% | N/A | -0.31% | -0.95 |
| Medium | 13547 | 0.0% | 0.917 | -0.31% | -0.88 |
| High | 14227 | 0.2% | 0.863 | -0.32% | -0.85 |

### Threshold-Sweep P&L (Using Meta Probability)


| Threshold | Trades | Mean Return | Cum Return | Sharpe (per trade) |
|----------|--------|-------------|------------|---------------------|
| 0.50 | 6527 | -0.32% | -2075.48% | -67.670 |
| 0.51 | 16 | -0.34% | -5.39% | -3.114 |
| 0.52 | 16 | -0.34% | -5.39% | -3.114 |
| 0.53 | 14 | -0.21% | -2.97% | -2.767 |
| 0.54 | 11 | -0.24% | -2.59% | -2.415 |
| 0.55 | 11 | -0.24% | -2.59% | -2.415 |
| 0.56 | 11 | -0.24% | -2.59% | -2.415 |
| 0.57 | 10 | -0.25% | -2.46% | -2.298 |
| 0.58 | 10 | -0.25% | -2.46% | -2.298 |
| 0.59 | 10 | -0.25% | -2.46% | -2.298 |
| 0.60 | 10 | -0.25% | -2.46% | -2.298 |
| 0.61 | 10 | -0.25% | -2.46% | -2.298 |
| 0.62 | 10 | -0.25% | -2.46% | -2.298 |
| 0.63 | 9 | -0.14% | -1.25% | -19.443 |
| 0.64 | 8 | -0.14% | -1.09% | -18.944 |
| 0.65 | 6 | -0.14% | -0.83% | -14.567 |

## 14. Stability and Bootstrap Statistics

Finally, we assess how stable the labeling and model performance are across time (years and time-series folds) and how sensitive key metrics such as positive rate and label–return separation are under bootstrap resampling. Tight confidence intervals indicate robust signal rather than fragile noise.


### Per-Year Label and Return Stability

- **Year 2021:** positive=1.5%, mean_return=-0.32%, AUC=0.500
- **Year 2022:** positive=0.0%, mean_return=-0.32%, AUC=nan
- **Year 2023:** positive=0.0%, mean_return=-0.31%, AUC=nan
- **Year 2024:** positive=0.0%, mean_return=-0.31%, AUC=nan
- **Year 2025:** positive=0.0%, mean_return=-0.31%, AUC=nan

### Time-Series Fold Stability (Approximate)

- **Fold 1:** positive=0.0%, mean_return=-0.31%, AUC=nan
- **Fold 2:** positive=0.0%, mean_return=-0.32%, AUC=nan
- **Fold 3:** positive=0.0%, mean_return=-0.31%, AUC=nan
- **Fold 4:** positive=0.0%, mean_return=-0.31%, AUC=nan
- **Fold 5:** positive=0.0%, mean_return=-0.32%, AUC=nan

### Bootstrap Label Statistics (Labeled Events)

- **Positive rate 95% CI:** [0.0%, 0.1%]
- **Mean return diff (label=1 - label=0) 95% CI:** [0.97%, 1.71%]

---


## Summary and Recommendations


### Key Findings:

1. Label balance: 0.1% positive
2. Signal coverage: 26.9%
3. Mean return (label=1): 1.03%
4. Mean return (label=0): -0.31%
5. Calibration error: 0.485