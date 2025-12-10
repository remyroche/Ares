# Meta-Labeling HPO Report

**Generated:** 2025-12-09 21:03:18 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 143
- **Total Trials:** 230
- **Best Edge:** 0.002252
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | 0.0000 | 100 |
| Stage 2 (Refinement) | medium | 0.0000 | 50 |
| Stage 3 (Production Proxy) | strong | 0.0022 | 35 |
| Stage 4 (Labeling Refinement) | strong | 0.0023 | 45 |

## Best Parameters (Highest Edge)

```json
{
  "cusum_threshold": 0.01748912331452132,
  "target_signal_density": 6.802718398396224,
  "econ_min_return_multiple": 1.3591824632152847,
  "iso_min_prob": 0.06219386291048011,
  "target_clip_high_q": 0.9357359165029273,
  "signal_strength_scale_max": 1.5712779286222458,
  "r_multiple_pos_threshold": 0.3957400557667297,
  "transaction_cost_mult": 0.663757439164945
}
```

## Per-Regime Metrics (Best Edge Configuration)

| Regime | n_events | trades_per_day | mean_pos | mean_neg | edge | AUC |
|--------|----------|----------------|----------|----------|------|-----|
| -1.0 | 20 | 0.018 | 0.00434 | -0.00688 | 0.000124 | 0.581 |
| 2.0 | 68 | 0.062 | 0.00910 | -0.01038 | 0.000695 | 0.581 |
| 4.0 | 709 | 0.647 | 0.00585 | -0.00844 | 0.001216 | 0.581 |
| 0.0 | 95 | 0.087 | 0.00693 | -0.00789 | 0.000571 | 0.581 |
| 3.0 | 72 | 0.066 | 0.00406 | -0.00719 | 0.000208 | 0.581 |
| 1.0 | 292 | 0.267 | 0.00622 | -0.00688 | 0.000857 | 0.581 |

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** 0.6268
- **Profitability:** 16.7694
- **Mean AUC:** 0.5807
- **Sharpe (Winners):** 0.5323

```json
{
  "cusum_threshold": 0.017489123715491137,
  "target_signal_density": 6.802456255263741,
  "econ_min_return_multiple": 1.3591306914352304,
  "iso_min_prob": 0.062293143215457036,
  "target_clip_high_q": 0.9357435489475957,
  "signal_strength_scale_max": 1.5713058271589926,
  "r_multiple_pos_threshold": 0.39582407140500675,
  "transaction_cost_mult": 0.6637824061203269,
  "horizon_bars": 22,
  "min_event_spacing": 3,
  "trail_distance": 0.9592151860987443,
  "profit_mult_min": 0.9421313143082265,
  "profit_mult_max": 1.2423430299134528,
  "stop_mult_min": 0.6304225923321515,
  "stop_mult_max": 1.1241740970938194,
  "kalman_Q": 0.0002882594698403212,
  "kalman_R": 0.0015279649349357728,
  "vol_baseline_window": 125,
  "profit_thr_base": 0.01928268227120649,
  "stop_to_profit_ratio": 0.3677666515720448,
  "volatility_ema_span": 38,
  "rolling_k_window": 477,
  "p_min": 0.33147003187542895,
  "p_max": 0.6028080876471551,
  "sigmoid_slope": 1.9379564088741894,
  "sigmoid_midpoint": 0.17323043239404481,
  "median_vol_lookback": 472,
  "regime_trend_lookback": 18,
  "k_tp_base": 1.7503775140454376,
  "k_sl_base": 1.9173341813001548,
  "trail_atr_mult": 2.257297947745488,
  "k_tp_long_mult": 1.1574338923356826,
  "k_sl_long_mult": 0.8957803562259118,
  "z_magnitude_scale": 0.35484337840736435,
  "barrier_trend_alpha": 0.3237097859940188,
  "barrier_trend_lookback": 21,
  "tp_min_mult": 0.6850608178708101,
  "tp_max_mult": 3.9546444904988407,
  "sl_min_mult": 0.3260930623035823,
  "sl_max_mult": 2.4852040542291878,
  "conditional_quantile_long": 0.568192641950727,
  "future_return_horizon": 4
}
```

## Diagnostics Summary (Best Configuration)

### Filtering & AUC Inflation

| Metric | Value |
|--------|-------|
| AUC (full labels) | 0.4983 |
| AUC (filtered labels) | 0.3321 |
| AUC inflation (filtered - full) | -0.1662 |
| Filtering is major contributor | False |
| AUC dominated by large moves | False |
| Precision collapse detected | False |

### Permutation-Importance Leakage Diagnostics

| Metric | Value |
|--------|-------|
| Baseline AUC (probe) | 0.7888 |
| AUC after dropping top-k features | 0.7397 |
| Delta AUC (baseline - dropped) | 0.0491 |
| God feature suspected | False |
| Top features | momentum_5_x_regime_high, momentum_10_x_regime_high, momentum_20_x_regime_high, signal_trend_regime_x_macd_hist_abs, kalman_trend_x_vol_ratio |

### Calibration Diagnostics

| Metric | Value |
|--------|-------|
| Well calibrated | False |
| Brier score | 0.5000 |
| Expected Calibration Error (ECE) | 0.0000 |
| Maximum Calibration Error (MCE) | 0.0000 |

### Mutual Information (Meta-Score vs Targets)

| Relationship | MI (bits) |
|--------------|-----------|
| Probabilities → Label | 0.0000 |
| Probabilities → Return sign | 0.0000 |

### Robustness & Class Overlap

| Metric | Value |
|--------|-------|
| Robust across folds | False |
| Worst fold AUC | 0.5434 |
| AUC CV std | 0.1009 |
| Easy problem detected | False |

#### Per-Fold AUC Summary

| Fold | AUC | n_test | ECE | Net P&L per trade |
|------|-----|--------|-----|-------------------|
| 0 | 0.5434 | 2066 | 0.1039 | -0.0026 |
| 1 | 0.7054 | 2066 | 0.0536 | 0.0052 |
| 2 | 0.7011 | 2066 | 0.0475 | 0.0046 |
| 3 | 0.8178 | 2066 | 0.0481 | 0.0065 |
| 4 | 0.8180 | 2066 | 0.0379 | 0.0060 |

#### AUC by Volatility Regime

| Regime | AUC | n_events |
|--------|-----|----------|
| low_vol | 0.6562 | 4092 |
| medium_vol | 0.7066 | 4216 |
| high_vol | 0.8874 | 4092 |

### Y-Shuffle Sanity Test

| Metric | Value |
|--------|-------|
| AUC with shuffled labels | 0.5028 |

A well-behaved model should have AUC≈0.5 under label shuffling; any materially higher value would indicate leakage or mis-specification.

### Lag-1 Stress Test (Look-Ahead Bias)

| Metric | Value |
|--------|-------|
| AUC (base, t features) | 0.7888 |
| AUC (lag-1 features) | 0.7883 |
| AUC difference (base - lag1) | 0.0004 |
| Look-ahead suspected | False |

### Dummy-Rule Volatility Baseline

| Metric | Value |
|--------|-------|
| AUC (raw, signed) | 0.6378 |
| AUC (absolute, best side) | 0.6378 |
| Samples used | 12400 |

## Regularization & Scoring

### Realistic P&L Edge Metric

The primary scoring metric is the **Realistic P&L Edge**:

```
Edge = (Mean_Return_Label1 - Transaction_Cost) × max(0, 2×AUC - 1)
```

This metric penalizes 'profitable but unlearnable' strategies more realistically:
- If AUC = 0.5 (random), Edge = 0 regardless of profitability
- If AUC = 1.0 (perfect), you capture full mean return minus cost

### Regularization Checks

All configurations were evaluated with:

1. **Isotonic Calibration:** Probabilities calibrated to align with real expected returns
2. **Temporal Stability:** Rolling window AUC variance penalty
3. **Learnability Threshold:** Mean AUC < 0.7 heavily penalized
4. **Profit/Stop Constraint:** Profit threshold must be ≥ 1.5× stop threshold
5. **Label Balance:** Entropy-based balance scoring
6. **Early Stopping:** Per-trial and global early stopping to prevent overfitting

## Gate Usage & Early-Exit Statistics

| Gate | Count |
|------|-------|
| econ_gate | 62 |
| rr_worst_rr | 25 |

## Artifacts

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251209_210318.json`
- Y-shuffle sanity tests to ensure no trivial leakage
- Robustness diagnostics across CV folds and volatility regimes
- Dummy volatility baseline AUC to benchmark meta-model value-add

## Recommended Next Step: Meta-Gated Backtest

To validate that the meta-labeling AUC and edge translate into tradable performance, run the meta-gated backtest using the meta_gating_config produced by feature_generation_meta_labeling_step. For example:

```bash
python3 src/launcher/ares_launcher.py \
  --step meta_gated_backtest \
  --symbol ETHUSDT --exchange binance --timeframe 15m --direction long --execution-mode full
```

The meta-gated backtest report (meta_gated_backtest_report_*.md) provides event-level P&L, trades-per-day, drawdowns, and cost stress tests for the diagnostic gate implied by the best HPO configuration.
