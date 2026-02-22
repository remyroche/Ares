# TBM Parameter Comparison Review

## Executive Summary

This report provides a thorough review of the Triple Barrier Model (TBM) parameter comparison results from [`compare_tbm_parameters.py`](extreme_price_movements/offline_optimisers/compare_tbm_parameters.py). The grid search evaluated **104 different TBM parameter configurations** across 12 cells (4 buckets × 3 horizons: H2, H4, H8).

**Critical Finding**: All 104 configurations fail production admissibility checks. The primary issues are:
1. **Floor Dominance**: TP floor binds too many configurations (>20% threshold)
2. **Low Min Cell AUC**: Minimum AUC across cells falls below 0.54 threshold
3. **Low Min Cell Bind Product**: Below 0.38 threshold

---

## Configuration Space Overview

### Parameters Tested

| Parameter | Values Tested | Description |
|-----------|-------------|-------------|
| `k_tp` | [0.4, 0.65, 0.8] | Take-profit multiplier |
| `sl_as_tp_pct` | [0.4, 0.5, 0.6, 0.7, 0.8, 1.0] | Stop-loss as % of TP |
| `mode` | [atr_norm, atr_norm_2A, atr_norm_2A_sl, atr_norm_2A_sl_path] | ATR normalization mode |
| `base_atr_window` | [336, 504, 672] | ATR lookback period |

### Best Configuration (by Sortino)

| Metric | Value |
|--------|-------|
| **Config ID** | CFG75750330C8 |
| **k_tp** | 0.8 |
| **sl_as_tp_pct** | 0.5 |
| **mode** | atr_norm_2A_sl_path |
| **Sortino** | 0.0841 |
| **IC Payoff** | 0.0236 |
| **Min Cell AUC** | 0.5152 |
| **Prod Adm Failures** | ~8-10 |

### Best Configuration (by IC Payoff)

| Metric | Value |
|--------|-------|
| **Config ID** | CFG25CE15AA73 |
| **k_tp** | 0.4 |
| **sl_as_tp_pct** | 0.4 |
| **mode** | atr_norm |
| **IC Payoff** | 0.0737 |
| **Sortino** | 0.0183 |
| **Min Cell AUC** | 0.3371 |
| **Prod Adm Failures** | 9 |

---

## Per-Cell Detailed Metrics

### Performance by k_tp

| k_tp | Mean IC Payoff | Max IC Payoff | Mean Sortino | Max Sortino | Mean Timeout Rate | Mean TP Hit Rate | Mean Prod Adm Failures |
|------|---------------|---------------|--------------|-------------|------------------|-----------------|----------------------|
| 0.40 | 0.0487 | 0.0737 | 0.0160 | 0.0183 | 38.6% | 37.1% | 9.52 |
| 0.65 | 0.0324 | 0.0350 | 0.0646 | ~0.08 | 46.1% | 28.3% | 10.0 |
| 0.80 | 0.0247 | 0.0326 | 0.0636 | ~0.08 | 46.1% | 28.1% | 8.96 |

### Key Observations by Parameter

1. **k_tp = 0.4** (Low TP):
   - ✅ Highest IC payoff (0.0487 mean, 0.0737 max)
   - ✅ Lowest timeout rate (38.6%)
   - ❌ Lowest sortino (0.016 mean)
   - ❌ Highest prod adm failures (9.52 mean)

2. **k_tp = 0.65** (Medium TP):
   - ✅ Best balance of sortino (0.0646 mean)
   - ❌ Highest timeout rate (46.1%)
   - ❌ All configs fail production (10 mean failures)

3. **k_tp = 0.8** (High TP):
   - ✅ Best sortino achievable (0.0841 max)
   - ✅ Lowest prod adm failures (8.96 mean)
   - ❌ Lowest IC payoff (0.0247 mean)
   - ❌ High timeout rate (46.1%)

---

## Production Admissibility Analysis

### Failure Distribution

| Failures | Count | % of Total |
|----------|-------|------------|
| 5 | 5 | 4.8% |
| 6 | 2 | 1.9% |
| 7 | 1 | 1.0% |
| 8 | 13 | 12.5% |
| 9 | 27 | 26.0% |
| 10 | 56 | 53.8% |

**Zero configurations pass all production admissibility checks.**

### Min Cell AUC Distribution

| Statistic | Value |
|-----------|-------|
| Minimum | 0.2918 |
| Maximum | 0.6047 |
| Mean | 0.4751 |
| **Configs ≥ 0.54** | **20 (19.2%)** |

The production threshold for min_cell_auc is 0.54, but only 20 out of 104 configs meet this requirement.

---

## Root Cause Analysis

### Issue 1: Floor Dominance

The TP floor (`tp_floor_bind_prod_agg`) is binding >20% of configurations across all tested parameter combinations. This indicates:
- The ATR-based floor calculation is too aggressive
- Market conditions may not support the current floor levels
- Need to adjust `tp_abs_lo_pct` or `tp_base_pct` parameters

### Issue 2: Stop-Loss Never Hits

Across all 104 configurations:
- **SL hit rate = 0%** for all configs
- Timeout rates range from 25% to 50%

This suggests the SL barriers are set too far from entry, making them unreachable within the timeout period.

### Issue 3: Mode Comparison

| Mode | Mean IC Payoff | Mean Sortino |
|------|---------------|--------------|
| atr_norm | ~0.05 | ~0.02 |
| atr_norm_2A | ~0.03 | ~0.06 |
| atr_norm_2A_sl | ~0.02 | ~0.06 |
| atr_norm_2A_sl_path | ~0.02 | ~0.08 |

The `atr_norm` mode produces highest IC payoff but lowest sortino. The `atr_norm_2A_sl_path` mode offers best risk-adjusted returns but with lower signal quality.

---

## Recommendations

### Immediate Actions

1. **Adjust TP Floor Calculation**:
   - Reduce `tp_abs_lo_pct` from 0.02 to 0.01
   - Lower `tp_base_pct` from 0.023 to 0.015
   - This should reduce floor dominance

2. **Tighten Stop-Loss Settings**:
   - Current sl_as_tp_pct=0.4-1.0 is too loose
   - Target sl_as_tp_pct=0.3-0.5 to ensure SL can actually be hit
   - This will reduce timeout rates

3. **Re-tune for Production**:
   - Target configurations with:
     - min_cell_auc ≥ 0.54
     - tp_floor_bind_prod_agg < 20%
     - min_cell_bind_prod ≥ 0.38

### Parameter Search Space for Next Iteration

Based on this analysis, suggest:

| Parameter | New Range | Rationale |
|-----------|-----------|-----------|
| k_tp | [0.6, 0.7, 0.8, 0.9, 1.0] | Focus on higher values for better sortino |
| sl_as_tp_pct | [0.3, 0.4, 0.5] | Tighter SL to reduce timeouts |
| tp_base_pct | [0.010, 0.015, 0.020] | Lower to reduce floor dominance |
| tp_abs_lo_pct | [0.008, 0.010, 0.012] | Reduce floor floor impact |

---

## Files Generated

- **Results CSV**: `extreme_price_movements/offline_optimisers/reports/tbm_parameter_comparison.csv`
- **Best Params**: `extreme_price_movements/offline_optimisers/reports/tbm_best_params.csv`

---

*Generated: 2026-02-20*
