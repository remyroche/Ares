# Meta-Labeling HPO Report

**Generated:** 2025-12-09 20:16:25 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 31
- **Total Trials:** 116
- **Best Edge:** 0.001546
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | 0.0000 | 24 |
| Stage 2 (Refinement) | medium | 0.0000 | 12 |
| Stage 3 (Production Proxy) | strong | 0.0015 | 35 |
| Stage 4 (Labeling Refinement) | strong | 0.0015 | 45 |

## Best Parameters (Highest Edge)

```json
{
  "cusum_threshold": 0.017489124201200423,
  "target_signal_density": 6.93120968552656,
  "econ_min_return_multiple": 1.3595177278311787,
  "iso_min_prob": 0.06221919798690225,
  "target_clip_high_q": 0.9357322148908405,
  "signal_strength_scale_max": 1.5713105649845596,
  "r_multiple_pos_threshold": 0.39498732926338836,
  "transaction_cost_mult": 0.6636651670953451
}
```

## Per-Regime Metrics (Best Edge Configuration)

| Regime | n_events | trades_per_day | mean_pos | mean_neg | edge | AUC |
|--------|----------|----------------|----------|----------|------|-----|
| 3 | 428 | 2.675 | 0.00370 | -0.00711 | 0.001092 | 0.579 |
| 0 | 398 | 2.487 | 0.00325 | -0.00719 | 0.000773 | 0.579 |
| 1 | 998 | 6.237 | 0.00351 | -0.00589 | 0.001478 | 0.579 |
| 4 | 1070 | 6.688 | 0.00304 | -0.00811 | 0.001060 | 0.579 |
| -1 | 128 | 0.800 | 0.00349 | -0.00735 | 0.000524 | 0.579 |
| 2 | 109 | 0.681 | 0.00704 | -0.01059 | 0.001625 | 0.579 |

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** 0.2703
- **Profitability:** 3.7524
- **Mean AUC:** 0.6101
- **Sharpe (Winners):** 0.2791

```json
{
  "cusum_threshold": 0.017489123107365917,
  "target_signal_density": 6.931323641157351,
  "econ_min_return_multiple": 1.3600928228907583,
  "iso_min_prob": 0.06223837734887755,
  "target_clip_high_q": 0.9357456443092346,
  "signal_strength_scale_max": 1.571303811534864,
  "r_multiple_pos_threshold": 0.39450723913894187,
  "transaction_cost_mult": 0.6636053984183415,
  "horizon_bars": 25,
  "min_event_spacing": 4,
  "profit_thr_base": 0.019866134730223858,
  "stop_to_profit_ratio": 0.4132604850423981,
  "trail_distance": 0.9947299446713509,
  "vol_baseline_window": 165,
  "profit_mult_min": 0.9424035203650635,
  "profit_mult_max": 1.2398416311090323,
  "stop_mult_min": 0.6303461898166939,
  "stop_mult_max": 1.1241723294016686,
  "volatility_ema_span": 35,
  "rolling_k_window": 330,
  "kalman_Q": 0.00020812274976292244,
  "kalman_R": 0.0014592767221243935,
  "p_min": 0.2830996390800766,
  "p_max": 0.7136150745842493,
  "sigmoid_slope": 1.6429269923417227,
  "sigmoid_midpoint": 0.11919099394047533,
  "median_vol_lookback": 353,
  "regime_trend_lookback": 29,
  "k_tp_base": 2.803487959331716,
  "k_sl_base": 0.5237867005268957,
  "trail_atr_mult": 2.7073644919645194,
  "k_tp_long_mult": 1.2026818521010165,
  "k_sl_long_mult": 0.7113040876987025,
  "z_magnitude_scale": 0.15447391641131164,
  "barrier_trend_alpha": 0.2692748476036517,
  "barrier_trend_lookback": 34,
  "tp_min_mult": 0.7765693854556758,
  "tp_max_mult": 3.8972954750065205,
  "sl_min_mult": 0.48161074686262023,
  "sl_max_mult": 2.6056793891114713,
  "conditional_quantile_long": 0.6617283265097144,
  "future_return_horizon": 4
}
```

## Diagnostics Summary (Best Configuration)

### Filtering & AUC Inflation

| Metric | Value |
|--------|-------|
| AUC (full labels) | 0.4947 |
| AUC (filtered labels) | 0.5553 |
| AUC inflation (filtered - full) | 0.0606 |
| Filtering is major contributor | False |
| AUC dominated by large moves | False |
| Precision collapse detected | False |

### Permutation-Importance Leakage Diagnostics

| Metric | Value |
|--------|-------|
| Baseline AUC (probe) | 0.8762 |
| AUC after dropping top-k features | 0.8477 |
| Delta AUC (baseline - dropped) | 0.0285 |
| God feature suspected | False |
| Top features | momentum_5_x_regime_high, kalman_trend_x_vol_ratio, volatility_1d, momentum_10_x_regime_high, momentum_20_x_regime_high |

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
| Worst fold AUC | 0.5699 |
| AUC CV std | 0.0790 |
| Easy problem detected | False |

#### Per-Fold AUC Summary

| Fold | AUC | n_test | ECE | Net P&L per trade |
|------|-----|--------|-----|-------------------|
| 0 | 0.6093 | 326 | 0.1110 | -0.0008 |
| 1 | 0.7054 | 326 | 0.2053 | 0.0051 |
| 2 | 0.7618 | 326 | 0.0956 | 0.0029 |
| 3 | 0.5699 | 326 | 0.2012 | -0.0024 |
| 4 | 0.7624 | 326 | 0.1073 | 0.0029 |

#### AUC by Volatility Regime

| Regime | AUC | n_events |
|--------|-----|----------|
| low_vol | 0.8368 | 646 |
| medium_vol | 0.8589 | 664 |
| high_vol | 0.9537 | 646 |

### Y-Shuffle Sanity Test

| Metric | Value |
|--------|-------|
| AUC with shuffled labels | 0.5007 |

A well-behaved model should have AUC≈0.5 under label shuffling; any materially higher value would indicate leakage or mis-specification.

### Lag-1 Stress Test (Look-Ahead Bias)

| Metric | Value |
|--------|-------|
| AUC (base, t features) | 0.8762 |
| AUC (lag-1 features) | 0.8785 |
| AUC difference (base - lag1) | -0.0023 |
| Look-ahead suspected | False |

### Dummy-Rule Volatility Baseline

| Metric | Value |
|--------|-------|
| AUC (raw, signed) | 0.6561 |
| AUC (absolute, best side) | 0.6561 |
| Samples used | 1956 |

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
| econ_gate | 18 |
| rr_worst_rr | 6 |

## Artifacts

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251209_201625.json`
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
