# Meta-Labeling HPO Report

**Generated:** 2025-12-08 19:35:29 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m | **Direction:** long

---

## Summary

- **Total Configurations Evaluated:** 194
- **Total Trials:** 230
- **Best Edge:** 0.006051
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | 0.0007 | 100 |
| Stage 2 (Refinement) | medium | 0.0000 | 50 |
| Stage 3 (Production Proxy) | strong | 0.0059 | 35 |
| Stage 4 (Labeling Refinement) | strong | 0.0061 | 45 |

## Best Parameters (Highest Edge)

```json
{
  "cusum_threshold": 0.017492280158452365,
  "target_signal_density": 6.802225601507517,
  "label_low_q": 0.4491096277528547,
  "label_high_q": 0.7054289945489125,
  "econ_min_return_multiple": 1.3104766934127894,
  "iso_min_prob": 0.05870692842710186,
  "target_clip_high_q": 0.936102827517865,
  "signal_strength_scale_max": 1.5771759796364133,
  "r_multiple_pos_threshold": 0.3370839735358726,
  "transaction_cost_mult": 0.6666971679180366
}
```

## Per-Regime Metrics (Best Edge Configuration)

| Regime | n_events | trades_per_day | mean_pos | mean_neg | edge | AUC |
|--------|----------|----------------|----------|----------|------|-----|
| 2.0 | 53 | 0.048 | 0.02076 | -0.01095 | 0.001870 | 0.546 |
| 4.0 | 507 | 0.463 | 0.01516 | -0.01078 | 0.004058 | 0.546 |
| 0.0 | 67 | 0.061 | 0.01762 | -0.01096 | 0.001751 | 0.546 |
| 3.0 | 48 | 0.044 | 0.01277 | -0.01108 | 0.001021 | 0.546 |
| 1.0 | 237 | 0.216 | 0.01376 | -0.01004 | 0.002479 | 0.546 |

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** 0.4645
- **Profitability:** 47.6300
- **Mean AUC:** 0.5458
- **Sharpe (Winners):** 1.9655

```json
{
  "cusum_threshold": 0.017492280158452365,
  "target_signal_density": 6.802225601507517,
  "label_low_q": 0.4491096277528547,
  "label_high_q": 0.7054289945489125,
  "econ_min_return_multiple": 1.3104766934127894,
  "iso_min_prob": 0.05870692842710186,
  "target_clip_high_q": 0.936102827517865,
  "signal_strength_scale_max": 1.5771759796364133,
  "r_multiple_pos_threshold": 0.3370839735358726,
  "transaction_cost_mult": 0.6666971679180366,
  "horizon_bars": 26,
  "min_event_spacing": 3,
  "trail_distance": 0.9591844604521964,
  "profit_mult_min": 0.9745632593865073,
  "profit_mult_max": 1.0726019260468238,
  "stop_mult_min": 0.6243195994577673,
  "stop_mult_max": 1.1363311498290924,
  "kalman_Q": 2.696521678581592e-05,
  "kalman_R": 0.0016332026649818281,
  "vol_baseline_window": 170,
  "profit_thr_base": 0.020565979823830648,
  "stop_to_profit_ratio": 0.40136787873576524
}
```

## Diagnostics Summary (Best Configuration)

### Filtering & AUC Inflation

| Metric | Value |
|--------|-------|
| AUC (full labels) | 0.4967 |
| AUC (filtered labels) | 0.5102 |
| AUC inflation (filtered - full) | 0.0136 |
| Filtering is major contributor | False |
| AUC dominated by large moves | False |
| Precision collapse detected | False |

### Permutation-Importance Leakage Diagnostics

| Metric | Value |
|--------|-------|
| Baseline AUC (probe) | 0.6719 |
| AUC after dropping top-k features | 0.6685 |
| Delta AUC (baseline - dropped) | 0.0034 |
| God feature suspected | True |
| Top features | signal_trend_regime_x_macd_hist_abs, volatility_1d, close_range_50, rv_z_short, close_min_20 |

### Calibration Diagnostics

| Metric | Value |
|--------|-------|
| Well calibrated | False |
| Brier score | 0.3961 |
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
| Worst fold AUC | 0.5131 |
| AUC CV std | 0.0340 |
| Easy problem detected | False |

#### Per-Fold AUC Summary

| Fold | AUC | n_test | ECE | Net P&L per trade |
|------|-----|--------|-----|-------------------|
| 0 | 0.5321 | 2579 | 0.0962 | -0.0040 |
| 1 | 0.5131 | 2579 | 0.0974 | -0.0024 |
| 2 | 0.5453 | 2579 | 0.0378 | -0.0006 |
| 3 | 0.5802 | 2579 | 0.0938 | 0.0022 |
| 4 | 0.6076 | 2579 | 0.0324 | 0.0016 |

#### AUC by Volatility Regime

| Regime | AUC | n_events |
|--------|-----|----------|
| low_vol | 0.7980 | 5108 |
| medium_vol | 0.7741 | 5262 |
| high_vol | 0.8010 | 5108 |

### Y-Shuffle Sanity Test

| Metric | Value |
|--------|-------|
| AUC with shuffled labels | 0.5000 |

A well-behaved model should have AUC≈0.5 under label shuffling; any materially higher value would indicate leakage or mis-specification.

### Lag-1 Stress Test (Look-Ahead Bias)

| Metric | Value |
|--------|-------|
| AUC (base, t features) | 0.6719 |
| AUC (lag-1 features) | 0.6666 |
| AUC difference (base - lag1) | 0.0054 |
| Look-ahead suspected | False |

### Dummy-Rule Volatility Baseline

| Metric | Value |
|--------|-------|
| AUC (raw, signed) | 0.5801 |
| AUC (absolute, best side) | 0.5801 |
| Samples used | 15478 |

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
| rr_worst_rr | 36 |

## Artifacts

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_long_20251208_193529.json`
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
