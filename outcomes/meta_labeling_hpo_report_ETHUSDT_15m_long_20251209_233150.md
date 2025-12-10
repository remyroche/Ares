# Meta-Labeling HPO Report

**Generated:** 2025-12-09 23:31:50 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m | **Direction:** long

---

## Summary

- **Total Configurations Evaluated:** 205
- **Total Trials:** 230
- **Best Edge:** 0.004846
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | 0.0048 | 100 |
| Stage 2 (Refinement) | medium | 0.0015 | 50 |
| Stage 3 (Production Proxy) | strong | 0.0042 | 35 |
| Stage 4 (Labeling Refinement) | strong | 0.0043 | 45 |

## Best Parameters (Highest Edge)

```json
{
  "cusum_threshold": 0.017489123620356543,
  "target_signal_density": 6.802857225639665,
  "min_event_spacing": 4,
  "trail_distance": 0.9591950905182219,
  "profit_mult_min": 0.9421258996347891,
  "profit_mult_max": 1.242327588074971,
  "stop_mult_min": 0.6304281325962303,
  "stop_mult_max": 1.1241732191946008,
  "kalman_Q": 0.0002840261216160938,
  "kalman_R": 0.0014910678814530696,
  "label_low_q": 0.4449889084120344,
  "label_high_q": 0.7037926173728042,
  "econ_min_return_multiple": 1.338653755550544,
  "iso_min_prob": 0.057469784693780515,
  "target_clip_high_q": 0.9354449451990102,
  "signal_strength_scale_max": 1.5716188635988741,
  "r_multiple_pos_threshold": 0.39450680827552786,
  "transaction_cost_mult": 0.6674038747202805
}
```

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** 0.4099
- **Profitability:** -0.8746
- **Mean AUC:** 0.5340
- **Sharpe (Winners):** 2.2322

```json
{
  "cusum_threshold": 0.017489123620356543,
  "target_signal_density": 6.802857225639665,
  "min_event_spacing": 4,
  "trail_distance": 0.9591950905182219,
  "profit_mult_min": 0.9421258996347891,
  "profit_mult_max": 1.242327588074971,
  "stop_mult_min": 0.6304281325962303,
  "stop_mult_max": 1.1241732191946008,
  "kalman_Q": 0.0002840261216160938,
  "kalman_R": 0.0014910678814530696,
  "label_low_q": 0.4449889084120344,
  "label_high_q": 0.7037926173728042,
  "econ_min_return_multiple": 1.338653755550544,
  "iso_min_prob": 0.057469784693780515,
  "target_clip_high_q": 0.9354449451990102,
  "signal_strength_scale_max": 1.5716188635988741,
  "r_multiple_pos_threshold": 0.39450680827552786,
  "transaction_cost_mult": 0.6674038747202805,
  "horizon_bars": 22
}
```

## Diagnostics Summary (Best Configuration)

### Filtering & AUC Inflation

| Metric | Value |
|--------|-------|
| AUC (full labels) | 0.5110 |
| AUC (filtered labels) | 0.4471 |
| AUC inflation (filtered - full) | -0.0639 |
| Filtering is major contributor | False |
| AUC dominated by large moves | False |
| Precision collapse detected | False |

### Permutation-Importance Leakage Diagnostics

| Metric | Value |
|--------|-------|
| Baseline AUC (probe) | 0.7086 |
| AUC after dropping top-k features | 0.7086 |
| Delta AUC (baseline - dropped) | -0.0000 |
| God feature suspected | True |
| Top features | momentum_5_x_regime_high, signal_trend_regime_x_macd_hist_abs, kalman_trend_x_vol_ratio, vol_of_vol, momentum_10_x_regime_high |

### Calibration Diagnostics

| Metric | Value |
|--------|-------|
| Well calibrated | False |
| Brier score | 0.3996 |
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
| Worst fold AUC | 0.5538 |
| AUC CV std | 0.0563 |
| Easy problem detected | False |

#### Per-Fold AUC Summary

| Fold | AUC | n_test | ECE | Net P&L per trade |
|------|-----|--------|-----|-------------------|
| 0 | 0.5538 | 2551 | 0.0740 | -0.0012 |
| 1 | 0.5697 | 2551 | 0.0981 | -0.0007 |
| 2 | 0.5867 | 2551 | 0.0446 | 0.0005 |
| 3 | 0.6967 | 2551 | 0.0905 | 0.0037 |
| 4 | 0.6659 | 2551 | 0.0278 | 0.0021 |

#### AUC by Volatility Regime

| Regime | AUC | n_events |
|--------|-----|----------|
| low_vol | 0.7799 | 5051 |
| medium_vol | 0.7895 | 5205 |
| high_vol | 0.8668 | 5051 |

### Y-Shuffle Sanity Test

| Metric | Value |
|--------|-------|
| AUC with shuffled labels | 0.5000 |

A well-behaved model should have AUC≈0.5 under label shuffling; any materially higher value would indicate leakage or mis-specification.

### Lag-1 Stress Test (Look-Ahead Bias)

| Metric | Value |
|--------|-------|
| AUC (base, t features) | 0.7086 |
| AUC (lag-1 features) | 0.7063 |
| AUC difference (base - lag1) | 0.0022 |
| Look-ahead suspected | False |

### Dummy-Rule Volatility Baseline

| Metric | Value |
|--------|-------|
| AUC (raw, signed) | 0.6169 |
| AUC (absolute, best side) | 0.6169 |
| Samples used | 15307 |

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
| rr_worst_rr | 25 |

## Artifacts

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_long_20251209_233150.json`
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
