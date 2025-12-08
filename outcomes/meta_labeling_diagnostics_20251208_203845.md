# Meta-Labeling Diagnostics Report

**Generated:** 2025-12-08 20:38:45

**Symbol:** ETHUSDT
**Timeframe:** 15m
**Horizon:** 26 bars

**Final selected features (count):** 
60 features


**Final selected features (full list):**

- risk_score
- smc_predicted
- rv_z_short
- volatility_1d
- mr_probability_dense
- volume_zscore
- is_sunday
- close_max_50
- close_min_50
- kalman_trend
- low_dist_x_vol
- signal_trend_regime_x_macd_hist_abs
- atr_14
- close_range_10
- cvd_proxy
- volatility_1h_agg
- range_4h
- hour_sin
- close_range_50
- ofi_proxy
- return_std_26b
- volatility_4h
- volatility_ema
- returns_std_10
- drawdown_100
- volatility_20
- close_range_20
- range_1h
- close_min_20
- absorption_ratio
- high_dist_x_vol
- vol_of_vol
- dist_from_recent_low_50
- return_autocorr_lag1_w50
- liquidity_gap_abs
- signal_rsi_long_distance_50
- volatility_4h_agg
- dist_from_recent_high_50
- return_26b
- close_range_5
- signal_rsi_distance_50
- day_of_week
- sma_slope_26b
- rsi_kalman
- ma_distance_kalman
- dist_from_recent_high_20
- volatility_1h
- volatility_5
- momentum_per_vol
- signal_macd_hist_long_abs
- dist_from_recent_high_10
- momentum_20_x_rv_z
- close_minus_vwap
- hour_cos
- volume_imbalance
- volume_trend
- dist_from_recent_low_5
- sr_labeling_xgb_prob
- momentum_kalman
- returns_mean_50

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

- **Total labeled events:** 7822
- **Positive labels (profitable):** 5113 (65.4%)
- **Negative labels (unprofitable):** 2709 (34.6%)

✅ **OK:** Reasonable label balance (65.4%)

### Label Distribution Over Time

- **Daily positive rate - Mean:** 3.6%
- **Daily positive rate - Std:** 3.9%
- **Daily positive rate - Min:** 0.0%
- **Daily positive rate - Max:** 22.9%

⚠️ **Warning:** 1333 days with <10% positive labels

## 2. Signal Coverage and Sparsity

- **Total samples:** 140354
- **Labeled samples:** 7822
- **Coverage:** 5.6%

✅ **OK:** Reasonable signal coverage (5.6%)

## 3. Feature-Label Correlation Analysis (Post-Filter)


### Top 10 Most Correlated Features (Post-Filter):

- **mr_probability_dense:** -0.4540
- **smc_predicted:** -0.4236
- **risk_score:** -0.3971
- **signal_disagreement_ema:** -0.3619
- **signal_disagreement:** -0.3580
- **volatility_1d:** 0.2430
- **close_minus_vwap:** -0.2111
- **close_min_50:** -0.1923
- **close_max_50:** -0.1873
- **kalman_trend:** -0.1855

### Correlation Health Check (Post-Filter):


## 3B. Pre- vs Post-Filter Comparison and Feature Shift


### Sample Counts and Retention

- **Pre-filter events (realized_return not NaN):** 43470
- **Pre-filter positive/negative (raw economic):** 11800 / 31670
- **Post-filter labeled events:** 7822
- **Post-filter positive/negative (binary_labels):** 5113 / 2709
- **Total retention (post / pre):** 18.0%
- **Positive retention:** 43.3%
- **Negative retention:** 8.6%

### Pre- vs Post-Filter Signal Quality

- **Pre-filter mean return (label=1/0):** 1.46% / -0.93%
- **Post-filter mean return (label=1/0):** 1.26% / -1.19%
- **Pre-filter Cohen's d (label=1 vs 0):** 4.786
- **Post-filter Cohen's d (label=1 vs 0):** 5.316
- **Pre-filter SNR (mean/std, label=1):** 2.506
- **Post-filter SNR (mean/std, label=1):** 2.229

### Largest Feature Correlation Shifts (|post|-|pre|)

- **mr_probability_dense:** pre=0.0238, post=-0.4540, Δ|corr|=0.4302
- **smc_predicted:** pre=0.0244, post=-0.4236, Δ|corr|=0.3992
- **risk_score:** pre=0.0211, post=-0.3971, Δ|corr|=0.3760
- **signal_disagreement_ema:** pre=0.0098, post=-0.3619, Δ|corr|=0.3522
- **signal_disagreement:** pre=0.1074, post=-0.3580, Δ|corr|=0.2506
- **volatility_1d:** pre=0.0308, post=0.2430, Δ|corr|=0.2123
- **close_minus_vwap:** pre=0.0160, post=-0.2111, Δ|corr|=0.1951
- **close_min_50:** pre=0.0269, post=-0.1923, Δ|corr|=0.1654
- **close_max_50:** pre=0.0282, post=-0.1873, Δ|corr|=0.1592
- **close_min_20:** pre=0.0284, post=-0.1854, Δ|corr|=0.1570

## 4. P&L Distribution Per Label

### Label = 1 (Profitable Signals):

- **Count:** 5113
- **Mean return:** 1.26%
- **Median return:** 1.16%
- **Std return:** 0.56%
- **% Actually positive:** 99.2%

### Label = 0 (Unprofitable Signals):

- **Count:** 2709
- **Mean return:** -1.19%
- **Median return:** -1.24%
- **Std return:** 0.10%
- **% Actually positive:** 0.0%

### Labeling Quality:

- **Label overlap:** 0.5%

✅ **OK:** Acceptable label overlap (0.5%)

### Cost-Aware Event Quality Summary

- **Transaction cost (per event, approx):** 0.300%
- **Unconditional mean event return:** 0.41%
- **Mean return (label=1) minus cost:** 0.96%
- **Fraction of labeled events with |return| < cost:** 0.0%

## 5. Time-Series Stability and Regime Analysis

- **Win rate vs Volatility correlation:** 0.0412
- **Mean return vs Volatility correlation:** 0.1187

✅ **OK:** Win rate not strongly correlated with volatility

### Rolling Daily SNR (Return / Std by Day)

- **Daily SNR mean:** -0.3680
- **Daily SNR std:** 0.4795
- **Daily SNR min/max:** -2.6759 / 1.2543

### Regime Shift Detection:

- **30-day rolling correlation (win rate vs vol) - Mean:** 0.4107
- **30-day rolling correlation (win rate vs vol) - Std:** 0.1226

## 6. Model Probability Diagnostics


### Calibration (Predicted Probability vs Actual Success Rate):


| Predicted Prob | Actual Success Rate | Count |
|---------------|---------------------|-------|
| 0.500 | 0.654 | 7822 |

- **Mean calibration error:** 0.154

⚠️ **Warning:** High calibration error - model probabilities may not be well-calibrated

### Probability Distribution:

- **Mean probability:** 0.500
- **Median probability:** 0.500
- **Std probability:** 0.000

⚠️ **Warning:** Very low probability variance - model may not be learning useful patterns

## 7. Feature Importance Analysis


### Top 20 Features by Importance:

1. **vol_ratio_20_200:** 0.2055
2. **mr_probability_dense:** 0.1107
3. **signal_rsi_long_distance_50:** 0.0703
4. **signal_rsi_distance_50:** 0.0663
5. **volatility_4h:** 0.0346
6. **vol_of_vol:** 0.0346
7. **volatility_20:** 0.0241
8. **volatility_5:** 0.0229
9. **momentum_20:** 0.0144
10. **close_minus_vwap:** 0.0140
11. **liquidity_gap_abs:** 0.0135
12. **ofi_proxy:** 0.0129
13. **momentum_per_vol:** 0.0110
14. **volume_pressure:** 0.0107
15. **signal_count_active:** 0.0100
16. **rv_z_short:** 0.0100
17. **sr_labeling_xgb_prob:** 0.0100
18. **volatility_ema:** 0.0096
19. **volatility_1h:** 0.0095
20. **momentum_ema:** 0.0094

### Feature Importance Health Check:


✅ **OK:** No single feature dominates

✅ Kalman-filtered features are being used (5 features)

✅ Volatility-based features in top 10: vol_ratio_20_200, volatility_4h, vol_of_vol, volatility_20, volatility_5

## 8. Kalman Smoothed Labels Analysis

- **Mean smoothed label:** 0.657
- **Median smoothed label:** 0.770
- **Std smoothed label:** 0.355
- **Correlation with binary labels:** 0.827
- **Correlation with realized returns:** 0.707

✅ **OK:** Good correlation between smoothed and binary labels

## 9. Event Mechanics and R-Multiples

These diagnostics describe how trades exit (profit, stop, or timeout), how long they stay open, and how large the realized return is relative to the configured stop-loss (R-multiple). They help verify that the triple-barrier / TPSL configuration produces economically meaningful events.


### Exit Reason Mix (Labeled Events)

- **2:** 40.0%
- **1:** 34.5%
- **0:** 25.5%

### Event Duration Distribution (Bars)

- **Mean duration:** 22.40
- **Median duration:** 20.00
- **90th percentile:** 45.00

### R-Multiple Distribution (Return / Stop Threshold)

- **R-multiple (label=1) mean:** 1.42
- **R-multiple (label=1) median:** 1.30
- **R-multiple (label=0) mean:** -1.34
- **R-multiple (label=0) median:** -1.32

## 9B. Enhanced Event Quality Metrics

These advanced metrics help distinguish between efficient momentum capture and random drift trades. They measure path efficiency, timing quality, and entry/exit effectiveness.


### Path Efficiency Ratio (PER)

- **Mean PER (clipped 99.5%):** 0.521
- **Median PER:** 0.522
- **90th percentile PER:** 0.742
- **99th percentile PER:** 0.831

✅ **OK:** Good path efficiency on profitable events

### Time-to-Outcome Ratio (TTO)

- **Mean TTO:** 0.862
- **Median TTO:** 0.769

⚠️ **Warning:** TTO outside target range [0.4, 0.6]

### MFE/MAE Ratio (Maximum Favorable vs Adverse Excursion)

- **Mean MFE/MAE:** 331.195
- **Median MFE/MAE:** 3.437

✅ **OK:** Favorable excursions exceed adverse excursions

## 9C. Target Volatility Health Check

Verifies that the continuous target values have sufficient variance to train a regression model. If targets are constant or near-zero, the model cannot learn meaningful patterns.

- **Target mean:** -0.001489
- **Target std:** 0.002236
- **Non-zero targets:** 0 / 140354 (0.0%)

⚠️ **Warning:** Very few non-zero targets (0.0%)

## 9D. Information Coefficient (IC)

Measures the rank correlation between predicted probabilities and realized returns. This is the purest measure of ranking ability, independent of absolute calibration.

- **Spearman IC (prob, return):** nan
- **P-value:** nan

✅ **OK:** Meaningful rank correlation

## 10. Regime and Time-Conditional Label Checks

These metrics show how label base-rates and realized returns change across volatility regimes, trend states, and time-of-day/weekday. Large differences indicate that learnability and edge are regime-dependent, which is important for downstream model conditioning.


### Label Base-Rate by Volatility Regime

- **Regime high:** positive=56.8%, mean_return=0.26%
- **Regime low:** positive=74.3%, mean_return=0.61%
- **Regime medium:** positive=66.1%, mean_return=0.42%

### Trend-Conditional (Price vs SMA20)

- **Strong uptrend:** positive=65.0%, mean_return=0.41%
- **Strong downtrend:** positive=60.1%, mean_return=0.35%

### Time-of-Day Positive Rates (Top/Bottom)

- **Top hours by positive rate:**
  - Hour 4: 74.6%
  - Hour 5: 73.7%
  - Hour 2: 73.6%
- **Bottom hours by positive rate:**
  - Hour 15: 54.3%
  - Hour 18: 56.5%
  - Hour 13: 56.9%

### Day-of-Week Positive Rates

- Day 0: 56.3%
- Day 1: 69.6%
- Day 2: 67.6%
- Day 3: 73.2%
- Day 4: 71.4%
- Day 5: 72.9%
- Day 6: 52.2%

## 11. Label–Return Separation, Information Content, and Sample Size

This section quantifies how well the labels separate profitable from unprofitable events (effect size) and how much information they carry about the sign of future returns compared to a random baseline (mutual information and permutation tests).


### Separation Metrics

- **Mean return difference (label=1 - label=0):** 2.45%
- **Cohen's d effect size:** 5.316
- **Approx. required samples for 80% power (heuristic):** 0.6
- **Current labeled samples used in separation:** 7822.0

### Information Content vs Permutation Baseline

- **Mutual information (labels vs realized sign):** 0.6190
- **Baseline MI (mean over permutations):** 0.0001

## 12. Target and Expected-Return Alignment

Here we check whether the continuous targets produced by the isotonic mapping are consistent with realized returns: higher targets should correspond to higher average realized P&L, and most non-zero targets should exceed transaction costs.


## 13. Cost-Aware Metrics and Threshold P&L

These diagnostics evaluate the meta-model from a trading perspective: global AUC/Brier/PR-AUC, and how mean and cumulative returns evolve as you raise the probability threshold used to filter trades. This helps choose probability cutoffs that are profitable after costs.


### Cost-Aware Classification Metrics (OOF)

- **AUC:** 0.500
- **Brier score:** 0.2500
- **Average precision (PR-AUC):** 0.654

### Volatility Regime Breakdown

| Regime | Samples | Pos Rate | AUC | Mean Ret | Sharpe |
|--------|---------|----------|-----|----------|--------|
| Low | 1124 | 74.3% | 0.500 | 0.61% | 0.54 |
| Medium | 5095 | 66.1% | 0.500 | 0.42% | 0.34 |
| High | 1603 | 56.8% | 0.500 | 0.26% | 0.19 |

### Threshold-Sweep P&L (Using Meta Probability)


| Threshold | Trades | Mean Return | Cum Return | Sharpe (per trade) |
|----------|--------|-------------|------------|---------------------|
| 0.50 | 7822 | 0.41% | 3211.42% | 28.996 |
| 0.51 | 0 | N/A | N/A | N/A |
| 0.52 | 0 | N/A | N/A | N/A |
| 0.53 | 0 | N/A | N/A | N/A |
| 0.54 | 0 | N/A | N/A | N/A |
| 0.55 | 0 | N/A | N/A | N/A |
| 0.56 | 0 | N/A | N/A | N/A |
| 0.57 | 0 | N/A | N/A | N/A |
| 0.58 | 0 | N/A | N/A | N/A |
| 0.59 | 0 | N/A | N/A | N/A |
| 0.60 | 0 | N/A | N/A | N/A |
| 0.61 | 0 | N/A | N/A | N/A |
| 0.62 | 0 | N/A | N/A | N/A |
| 0.63 | 0 | N/A | N/A | N/A |
| 0.64 | 0 | N/A | N/A | N/A |
| 0.65 | 0 | N/A | N/A | N/A |

## 14. Stability and Bootstrap Statistics

Finally, we assess how stable the labeling and model performance are across time (years and time-series folds) and how sensitive key metrics such as positive rate and label–return separation are under bootstrap resampling. Tight confidence intervals indicate robust signal rather than fragile noise.


### Per-Year Label and Return Stability

- **Year 2021:** positive=100.0%, mean_return=0.92%, AUC=nan
- **Year 2022:** positive=100.0%, mean_return=0.91%, AUC=nan
- **Year 2023:** positive=100.0%, mean_return=0.85%, AUC=nan
- **Year 2024:** positive=76.2%, mean_return=0.59%, AUC=0.500
- **Year 2025:** positive=50.9%, mean_return=0.20%, AUC=0.500

### Time-Series Fold Stability (Approximate)

- **Fold 1:** positive=100.0%, mean_return=0.88%, AUC=nan
- **Fold 2:** positive=100.0%, mean_return=0.84%, AUC=nan
- **Fold 3:** positive=100.0%, mean_return=0.90%, AUC=nan
- **Fold 4:** positive=61.8%, mean_return=0.42%, AUC=0.500
- **Fold 5:** positive=49.8%, mean_return=0.17%, AUC=0.500

### Bootstrap Label Statistics (Labeled Events)

- **Positive rate 95% CI:** [64.5%, 66.1%]
- **Mean return diff (label=1 - label=0) 95% CI:** [2.43%, 2.46%]

---


## Summary and Recommendations


### Key Findings:

1. Label balance: 65.4% positive
2. Signal coverage: 5.6%
3. Mean return (label=1): 1.26%
4. Mean return (label=0): -1.19%
5. Calibration error: 0.154