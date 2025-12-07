# Meta-Labeling Diagnostics Report

**Generated:** 2025-12-07 13:46:46

**Symbol:** ETHUSDT
**Timeframe:** 15m
**Horizon:** 22 bars

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

- **Total labeled events:** 7324
- **Positive labels (profitable):** 4647 (63.4%)
- **Negative labels (unprofitable):** 2677 (36.6%)

✅ **OK:** Reasonable label balance (63.4%)

### Label Distribution Over Time

- **Daily positive rate - Mean:** 3.3%
- **Daily positive rate - Std:** 3.8%
- **Daily positive rate - Min:** 0.0%
- **Daily positive rate - Max:** 20.8%

⚠️ **Warning:** 1350 days with <10% positive labels

## 2. Signal Coverage and Sparsity

- **Total samples:** 140354
- **Labeled samples:** 7324
- **Coverage:** 5.2%

✅ **OK:** Reasonable signal coverage (5.2%)

## 3. Feature-Label Correlation Analysis (Post-Filter)


### Top 10 Most Correlated Features (Post-Filter):

- **event_win_rate_last_50:** 0.5051
- **event_mean_return_last_50:** 0.4758
- **event_r_multiple_mean_last_50:** 0.4693
- **mr_probability_dense:** -0.4268
- **smc_predicted:** -0.3935
- **zigzag_last_pivot_idx:** -0.3895
- **base_zigzag_last_pivot_idx:** -0.3895
- **risk_score:** -0.3675
- **signal_disagreement_ema:** -0.3397
- **signal_disagreement:** -0.3332

### Correlation Health Check (Post-Filter):


## 3B. Pre- vs Post-Filter Comparison and Feature Shift


### Sample Counts and Retention

- **Pre-filter events (realized_return not NaN):** 43471
- **Pre-filter positive/negative (raw economic):** 11370 / 32101
- **Post-filter labeled events:** 7324
- **Post-filter positive/negative (binary_labels):** 4647 / 2677
- **Total retention (post / pre):** 16.8%
- **Positive retention:** 40.9%
- **Negative retention:** 8.3%

### Pre- vs Post-Filter Signal Quality

- **Pre-filter mean return (label=1/0):** 1.32% / -0.85%
- **Post-filter mean return (label=1/0):** 1.12% / -1.09%
- **Pre-filter Cohen's d (label=1 vs 0):** 4.884
- **Post-filter Cohen's d (label=1 vs 0):** 4.851
- **Pre-filter SNR (mean/std, label=1):** 2.675
- **Post-filter SNR (mean/std, label=1):** 1.964

### Largest Feature Correlation Shifts (|post|-|pre|)

- **mr_probability_dense:** pre=0.0199, post=-0.4268, Δ|corr|=0.4069
- **zigzag_last_pivot_idx:** pre=0.0082, post=-0.3895, Δ|corr|=0.3813
- **base_zigzag_last_pivot_idx:** pre=0.0082, post=-0.3895, Δ|corr|=0.3813
- **smc_predicted:** pre=0.0210, post=-0.3935, Δ|corr|=0.3725
- **risk_score:** pre=0.0192, post=-0.3675, Δ|corr|=0.3482
- **signal_disagreement_ema:** pre=0.0000, post=-0.3397, Δ|corr|=0.3397
- **signal_disagreement:** pre=0.0928, post=-0.3332, Δ|corr|=0.2404
- **volatility_1d:** pre=0.0299, post=0.2661, Δ|corr|=0.2362
- **close_minus_vwap:** pre=0.0112, post=-0.1850, Δ|corr|=0.1738
- **event_mfe_mae_ratio_mean_last_50:** pre=0.2361, post=0.0634, Δ|corr|=-0.1727

## 4. P&L Distribution Per Label

### Label = 1 (Profitable Signals):

- **Count:** 4647
- **Mean return:** 1.12%
- **Median return:** 1.17%
- **Std return:** 0.57%
- **% Actually positive:** 97.1%

### Label = 0 (Unprofitable Signals):

- **Count:** 2677
- **Mean return:** -1.09%
- **Median return:** -1.11%
- **Std return:** 0.05%
- **% Actually positive:** 0.0%

### Labeling Quality:

- **Label overlap:** 1.8%

✅ **OK:** Acceptable label overlap (1.8%)

### Cost-Aware Event Quality Summary

- **Transaction cost (per event, approx):** 0.300%
- **Unconditional mean event return:** 0.31%
- **Mean return (label=1) minus cost:** 0.82%
- **Fraction of labeled events with |return| < cost:** 0.0%

## 5. Time-Series Stability and Regime Analysis

- **Win rate vs Volatility correlation:** 0.0508
- **Mean return vs Volatility correlation:** 0.1144

✅ **OK:** Win rate not strongly correlated with volatility

### Rolling Daily SNR (Return / Std by Day)

- **Daily SNR mean:** -0.3806
- **Daily SNR std:** 0.4434
- **Daily SNR min/max:** -2.5295 / 1.1166

### Regime Shift Detection:

- **30-day rolling correlation (win rate vs vol) - Mean:** 0.4332
- **30-day rolling correlation (win rate vs vol) - Std:** 0.1194

## 6. Model Probability Diagnostics


### Calibration (Predicted Probability vs Actual Success Rate):


| Predicted Prob | Actual Success Rate | Count |
|---------------|---------------------|-------|
| 0.198 | 0.056 | 18 |
| 0.277 | 0.182 | 33 |
| 0.375 | 0.746 | 3217 |
| 0.453 | 0.226 | 146 |
| 0.514 | 0.718 | 634 |
| 0.617 | 0.329 | 420 |
| 0.700 | 0.415 | 564 |
| 0.783 | 0.481 | 672 |
| 0.871 | 0.543 | 731 |
| 0.950 | 0.742 | 889 |

- **Mean calibration error:** 0.245

⚠️ **Warning:** High calibration error - model probabilities may not be well-calibrated

### Monotonicity and Slope Checks (Prob → Realized Return)

- **Monotonicity violations (adjacent bins):** 2 / 9
- **Approx. slope in high-probability region:** 0.026780

### Probability Distribution:

- **Mean probability:** 0.504
- **Median probability:** 0.500
- **Std probability:** 0.054

## 7. Feature Importance Analysis


### Top 20 Features by Importance:

1. **trend_1H:** 0.1301
2. **liquidity_gap_up:** 0.0908
3. **trend_4H:** 0.0607
4. **kalman_trend_x_vol_ratio:** 0.0527
5. **sma_slope_x_vol_ratio:** 0.0493
6. **cvd_proxy:** 0.0462
7. **ofi_proxy:** 0.0430
8. **is_bad_hour:** 0.0345
9. **base_zigzag_swing_cat:** 0.0243
10. **mtf_htf_bearish_count:** 0.0178
11. **swing_1H:** 0.0126
12. **vol_price_corr:** 0.0116
13. **trade_aggressor_ratio:** 0.0093
14. **momentum_5:** 0.0088
15. **return_autocorr_lag1_w50:** 0.0085
16. **base_trend_direction_cat:** 0.0082
17. **signed_volume_ema:** 0.0080
18. **mtf_all_aligned_up:** 0.0077
19. **atr_14:** 0.0071
20. **zigzag_trend_direction:** 0.0070

### Feature Importance Health Check:


✅ **OK:** No single feature dominates

✅ Kalman-filtered features are being used (6 features)

✅ Volatility-based features in top 10: kalman_trend_x_vol_ratio, sma_slope_x_vol_ratio

## 8. Kalman Smoothed Labels Analysis

- **Mean smoothed label:** 0.637
- **Median smoothed label:** 0.840
- **Std smoothed label:** 0.396
- **Correlation with binary labels:** 0.913
- **Correlation with realized returns:** 0.802

✅ **OK:** Good correlation between smoothed and binary labels

## 9. Event Mechanics and R-Multiples

These diagnostics describe how trades exit (profit, stop, or timeout), how long they stay open, and how large the realized return is relative to the configured stop-loss (R-multiple). They help verify that the triple-barrier / TPSL configuration produces economically meaningful events.


### Exit Reason Mix (Labeled Events)

- **1:** 37.3%
- **2:** 34.8%
- **0:** 27.9%

### Event Duration Distribution (Bars)

- **Mean duration:** 17.84
- **Median duration:** 15.00
- **90th percentile:** 37.00

### R-Multiple Distribution (Return / Stop Threshold)

- **R-multiple (label=1) mean:** 1.42
- **R-multiple (label=1) median:** 1.45
- **R-multiple (label=0) mean:** -1.37
- **R-multiple (label=0) median:** -1.37

## 9B. Enhanced Event Quality Metrics

These advanced metrics help distinguish between efficient momentum capture and random drift trades. They measure path efficiency, timing quality, and entry/exit effectiveness.


### Path Efficiency Ratio (PER)

- **Mean PER (clipped 99.5%):** 0.529
- **Median PER:** 0.534
- **90th percentile PER:** 0.729
- **99th percentile PER:** 0.815

✅ **OK:** Good path efficiency on profitable events

### Time-to-Outcome Ratio (TTO)

- **Mean TTO:** 0.811
- **Median TTO:** 0.682

⚠️ **Warning:** TTO outside target range [0.4, 0.6]

### MFE/MAE Ratio (Maximum Favorable vs Adverse Excursion)

- **Mean MFE/MAE:** 320.233
- **Median MFE/MAE:** 3.300

✅ **OK:** Favorable excursions exceed adverse excursions

## 9C. Target Volatility Health Check

Verifies that the continuous target values have sufficient variance to train a regression model. If targets are constant or near-zero, the model cannot learn meaningful patterns.

- **Target mean:** -0.001137
- **Target std:** 0.002236
- **Non-zero targets:** 2921 / 140354 (2.1%)

✅ **OK:** Targets have sufficient variance

## 9D. Information Coefficient (IC)

Measures the rank correlation between predicted probabilities and realized returns. This is the purest measure of ranking ability, independent of absolute calibration.

- **Spearman IC (prob, return):** -0.0022
- **P-value:** 8.4915e-01

⚠️ **Warning:** Very weak IC (|IC| < 0.05) - model has minimal ranking ability

## 10. Regime and Time-Conditional Label Checks

These metrics show how label base-rates and realized returns change across volatility regimes, trend states, and time-of-day/weekday. Large differences indicate that learnability and edge are regime-dependent, which is important for downstream model conditioning.


### Label Base-Rate by Volatility Regime

- **Regime high:** positive=83.0%, mean_return=0.70%
- **Regime low:** positive=51.5%, mean_return=0.02%
- **Regime medium:** positive=55.8%, mean_return=0.20%

### Trend-Conditional (Price vs SMA20)

- **Strong uptrend:** positive=63.1%, mean_return=0.32%
- **Strong downtrend:** positive=59.3%, mean_return=0.27%

### Time-of-Day Positive Rates (Top/Bottom)

- **Top hours by positive rate:**
  - Hour 4: 74.6%
  - Hour 5: 72.9%
  - Hour 8: 72.4%
- **Bottom hours by positive rate:**
  - Hour 15: 51.1%
  - Hour 16: 54.0%
  - Hour 14: 54.2%

### Day-of-Week Positive Rates

- Day 0: 57.1%
- Day 1: 68.8%
- Day 2: 66.4%
- Day 3: 70.1%
- Day 4: 70.9%
- Day 5: 67.9%
- Day 6: 49.4%

## 11. Label–Return Separation, Information Content, and Sample Size

This section quantifies how well the labels separate profitable from unprofitable events (effect size) and how much information they carry about the sign of future returns compared to a random baseline (mutual information and permutation tests).


### Separation Metrics

- **Mean return difference (label=1 - label=0):** 2.22%
- **Cohen's d effect size:** 4.851
- **Approx. required samples for 80% power (heuristic):** 0.7
- **Current labeled samples used in separation:** 7324.0

### Information Content vs Permutation Baseline

- **Mutual information (labels vs realized sign):** 0.5830
- **Baseline MI (mean over permutations):** 0.0001

## 12. Target and Expected-Return Alignment

Here we check whether the continuous targets produced by the isotonic mapping are consistent with realized returns: higher targets should correspond to higher average realized P&L, and most non-zero targets should exceed transaction costs.


### Target vs Realized Return

- **Correlation (target, realized):** 0.048
- **MSE (target vs realized):** 0.000175

### Target/Return by Target Decile

- Decile 0: target=0.0038, realized=0.0019
- Decile 1: target=0.0055, realized=-0.0004
- Decile 2: target=0.0061, realized=0.0022
- Decile 3: target=0.0063, realized=0.0032
- Decile 4: target=0.0066, realized=0.0021

### Target Distribution Sanity

- **Non-zero target fraction (all samples):** 2.1%
- **Mean non-zero target:** 0.0058
- **Median non-zero target:** 0.0061
- **Fraction of targets below transaction cost (0.300%):** 6.6%

## 13. Cost-Aware Metrics and Threshold P&L

These diagnostics evaluate the meta-model from a trading perspective: global AUC/Brier/PR-AUC, and how mean and cumulative returns evolve as you raise the probability threshold used to filter trades. This helps choose probability cutoffs that are profitable after costs.


### Cost-Aware Classification Metrics (OOF)

- **AUC:** 0.434
- **Brier score:** 0.3081
- **Average precision (PR-AUC):** 0.658

### Volatility Regime Breakdown

| Regime | Samples | Pos Rate | AUC | Mean Ret | Sharpe |
|--------|---------|----------|-----|----------|--------|
| Low | 1835 | 51.5% | 0.317 | 0.02% | 0.01 |
| Medium | 3152 | 55.8% | 0.423 | 0.20% | 0.16 |
| High | 2325 | 83.0% | 0.506 | 0.70% | 0.67 |

⚠️ **Warning:** Large win-rate disparity between regimes (low: 51.5%, high: 83.0%). Performance is highly regime-dependent.

### Threshold-Sweep P&L (Using Meta Probability)


| Threshold | Trades | Mean Return | Cum Return | Sharpe (per trade) |
|----------|--------|-------------|------------|---------------------|
| 0.50 | 3878 | 0.22% | 844.85% | 10.941 |
| 0.51 | 3488 | 0.15% | 538.08% | 7.104 |
| 0.52 | 3459 | 0.16% | 544.46% | 7.218 |
| 0.53 | 3430 | 0.16% | 563.84% | 7.505 |
| 0.54 | 3395 | 0.17% | 588.02% | 7.866 |
| 0.55 | 3365 | 0.18% | 595.58% | 8.005 |
| 0.56 | 3320 | 0.18% | 604.69% | 8.185 |
| 0.57 | 3283 | 0.18% | 600.07% | 8.169 |
| 0.58 | 3247 | 0.18% | 596.44% | 8.171 |
| 0.59 | 3194 | 0.20% | 624.43% | 8.630 |
| 0.60 | 3157 | 0.20% | 631.37% | 8.783 |
| 0.61 | 3109 | 0.20% | 634.54% | 8.901 |
| 0.62 | 3058 | 0.21% | 656.95% | 9.296 |
| 0.63 | 3004 | 0.22% | 665.61% | 9.511 |
| 0.64 | 2953 | 0.23% | 679.13% | 9.788 |
| 0.65 | 2900 | 0.24% | 696.40% | 10.134 |

## 14. Stability and Bootstrap Statistics

Finally, we assess how stable the labeling and model performance are across time (years and time-series folds) and how sensitive key metrics such as positive rate and label–return separation are under bootstrap resampling. Tight confidence intervals indicate robust signal rather than fragile noise.


### Per-Year Label and Return Stability

- **Year 2021:** positive=100.0%, mean_return=0.89%, AUC=nan
- **Year 2022:** positive=100.0%, mean_return=0.83%, AUC=nan
- **Year 2023:** positive=100.0%, mean_return=0.80%, AUC=nan
- **Year 2024:** positive=73.8%, mean_return=0.49%, AUC=0.500
- **Year 2025:** positive=50.7%, mean_return=0.13%, AUC=0.615

### Time-Series Fold Stability (Approximate)

- **Fold 1:** positive=100.0%, mean_return=0.82%, AUC=nan
- **Fold 2:** positive=100.0%, mean_return=0.79%, AUC=nan
- **Fold 3:** positive=100.0%, mean_return=0.85%, AUC=nan
- **Fold 4:** positive=60.9%, mean_return=0.32%, AUC=0.500
- **Fold 5:** positive=49.6%, mean_return=0.10%, AUC=0.694

### Bootstrap Label Statistics (Labeled Events)

- **Positive rate 95% CI:** [62.4%, 64.5%]
- **Mean return diff (label=1 - label=0) 95% CI:** [2.20%, 2.23%]

---


## Summary and Recommendations


### Key Findings:

1. Label balance: 63.4% positive
2. Signal coverage: 5.2%
3. Mean return (label=1): 1.12%
4. Mean return (label=0): -1.09%
5. Calibration error: 0.245