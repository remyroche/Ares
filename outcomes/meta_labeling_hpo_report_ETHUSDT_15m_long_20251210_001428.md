# Meta-Labeling HPO Report

**Generated:** 2025-12-10 00:14:28 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m | **Direction:** long

---

## Summary

- **Total Configurations Evaluated:** 194
- **Total Trials:** 230
- **Best Edge:** 0.002961
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | 0.0019 | 100 |
| Stage 2 (Refinement) | medium | 0.0003 | 50 |
| Stage 3 (Production Proxy) | strong | 0.0028 | 35 |
| Stage 4 (Labeling Refinement) | strong | 0.0030 | 45 |

## Best Parameters (Highest Edge)

```json
{
  "cusum_threshold": 0.017493750116268653,
  "target_signal_density": 6.8024381351908945,
  "label_low_q": 0.4189149509589369,
  "label_high_q": 0.6107278662378264,
  "econ_min_return_multiple": 1.3065871282448653,
  "iso_min_prob": 0.07489290844280615,
  "target_clip_high_q": 0.9798221603854687,
  "signal_strength_scale_max": 1.6167293283792525,
  "r_multiple_pos_threshold": 0.8448577021821315,
  "transaction_cost_mult": 0.7096857076137795
}
```

## Per-Regime Metrics (Best Edge Configuration)

| Regime | n_events | trades_per_day | mean_pos | mean_neg | edge | AUC |
|--------|----------|----------------|----------|----------|------|-----|
| 2.0 | 58 | 0.053 | 0.01784 | -0.00901 | 0.002393 | 0.577 |
| 0.0 | 73 | 0.067 | 0.01074 | -0.00923 | 0.001472 | 0.577 |
| 3.0 | 51 | 0.047 | 0.00513 | -0.00906 | 0.000428 | 0.577 |
| 4.0 | 559 | 0.511 | 0.00634 | -0.00895 | 0.001992 | 0.577 |
| 1.0 | 251 | 0.229 | 0.00647 | -0.00894 | 0.001375 | 0.577 |

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** 0.5670
- **Profitability:** 18.3308
- **Mean AUC:** 0.5769
- **Sharpe (Winners):** 0.6281

```json
{
  "cusum_threshold": 0.017493750116268653,
  "target_signal_density": 6.8024381351908945,
  "label_low_q": 0.4189149509589369,
  "label_high_q": 0.6107278662378264,
  "econ_min_return_multiple": 1.3065871282448653,
  "iso_min_prob": 0.07489290844280615,
  "target_clip_high_q": 0.9798221603854687,
  "signal_strength_scale_max": 1.6167293283792525,
  "r_multiple_pos_threshold": 0.8448577021821315,
  "transaction_cost_mult": 0.7096857076137795,
  "horizon_bars": 22,
  "min_event_spacing": 3,
  "trail_distance": 0.9592601063893501,
  "profit_mult_min": 0.9644336773977822,
  "profit_mult_max": 1.2507933675399303,
  "stop_mult_min": 0.6931035642171133,
  "stop_mult_max": 1.084173901542305,
  "kalman_Q": 0.0004582197351147646,
  "kalman_R": 0.0010639962449506444,
  "vol_baseline_window": 86,
  "profit_thr_base": 0.016518814465380487,
  "stop_to_profit_ratio": 0.3976509070338487
}
```

## Diagnostics Summary (Best Configuration)

### Filtering & AUC Inflation

| Metric | Value |
|--------|-------|
| AUC (full labels) | 0.4975 |
| AUC (filtered labels) | 0.3603 |
| AUC inflation (filtered - full) | -0.1372 |
| Filtering is major contributor | False |
| AUC dominated by large moves | False |
| Precision collapse detected | False |

### Permutation-Importance Leakage Diagnostics

| Metric | Value |
|--------|-------|
| Baseline AUC (probe) | 0.7613 |
| AUC after dropping top-k features | 0.7594 |
| Delta AUC (baseline - dropped) | 0.0019 |
| God feature suspected | True |
| Top features | momentum_5_x_regime_high, signal_trend_regime_x_macd_hist_abs, kalman_trend_x_vol_ratio, momentum_20_x_regime_high, volatility_1d |

### Calibration Diagnostics

| Metric | Value |
|--------|-------|
| Well calibrated | False |
| Brier score | 0.4816 |
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
| Worst fold AUC | 0.5918 |
| AUC CV std | 0.0705 |
| Easy problem detected | False |

#### Per-Fold AUC Summary

| Fold | AUC | n_test | ECE | Net P&L per trade |
|------|-----|--------|-----|-------------------|
| 0 | 0.5918 | 2801 | 0.0817 | -0.0016 |
| 1 | 0.6498 | 2801 | 0.0567 | -0.0015 |
| 2 | 0.6599 | 2801 | 0.0410 | -0.0019 |
| 3 | 0.7951 | 2801 | 0.0842 | -0.0002 |
| 4 | 0.7317 | 2801 | 0.0343 | -0.0000 |

#### AUC by Volatility Regime

| Regime | AUC | n_events |
|--------|-----|----------|
| low_vol | 0.7566 | 5546 |
| medium_vol | 0.7466 | 5714 |
| high_vol | 0.9054 | 5546 |

### Y-Shuffle Sanity Test

| Metric | Value |
|--------|-------|
| AUC with shuffled labels | 0.5000 |

A well-behaved model should have AUC≈0.5 under label shuffling; any materially higher value would indicate leakage or mis-specification.

### Lag-1 Stress Test (Look-Ahead Bias)

| Metric | Value |
|--------|-------|
| AUC (base, t features) | 0.7613 |
| AUC (lag-1 features) | 0.7581 |
| AUC difference (base - lag1) | 0.0032 |
| Look-ahead suspected | False |

### Dummy-Rule Volatility Baseline

| Metric | Value |
|--------|-------|
| AUC (raw, signed) | 0.5898 |
| AUC (absolute, best side) | 0.5898 |
| Samples used | 16806 |

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
| rr_worst_rr | 35 |
| tto_hard | 1 |

## Artifacts

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_long_20251210_001428.json`
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
