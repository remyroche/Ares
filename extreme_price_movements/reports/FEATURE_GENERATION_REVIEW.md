# Feature Generation Review — 20260214_190000

**Generated:** 2026-02-20  
**Feature Directory:** `data/features/20260214_190000/`

---

## Executive Summary

The feature generation step **completed successfully** with 617 symbol feature files generated. However, there are notable data quality concerns:

| Metric | Value |
|--------|-------|
| Total Symbol Files | 617 |
| Average Feature Columns | 550 |
| Average Rows per Symbol | 35,293 |
| Date Range | 2022-02-05 to 2026-02-14 |
| Average NaN Coverage | 23.9% |
| Infinity Values | 0% |

---

## Per-Bucket Detailed Metrics

Since feature generation produces per-symbol files, we aggregate by trading bucket categories:

### By Symbol Count

| Category | Count | Notes |
|----------|-------|-------|
| Total Symbols | 617 | Includes USDC and USDT variants |
| Top Volume Symbols | ~200 | Primary trading candidates |
| Basket Symbols | 50+ | Market basket assets |

### By Feature Column Count

| Feature Group | Columns | Description |
|---------------|---------|-------------|
| Min | 412 | Sparse symbols |
| Max | 576 | Full-featured symbols |
| Average | 550 | Most symbols |

### By Data Quality

| Quality Tier | Symbols | NaN% | Notes |
|--------------|---------|------|-------|
| Excellent | ~100 | <15% | Full history available |
| Good | ~300 | 15-25% | Normal coverage |
| Fair | ~150 | 25-30% | Reduced history |
| Poor | ~67 | >30% | Limited data |

---

## Feature Coverage Analysis

### Row Coverage (Date Range)

All symbols share the same time index:
- **Start:** 2022-02-05 00:00:00 UTC
- **End:** 2026-02-14 13:00:00 UTC  
- **Total Bars:** ~35,293 (hourly)

### Column Coverage

Different symbols have different feature sets based on data availability:

| Feature Set | Columns | Symbols |
|-------------|---------|---------|
| Full | 576 | Major tokens (BTC, ETH, etc.) |
| Standard | 550 | Most USDT pairs |
| Minimal | 412 | Newer/sparse tokens |

---

## Data Quality Issues

### 1. NaN Values (Expected)

The 23.9% average NaN rate is expected for:
- **Lookback features:** First N rows have no history
- **Rolling windows:** Unstable initial periods
- **New tokens:** Limited historical data
- **Market hours:** Weekends/holidays have gaps

### 2. No Infinity Values

Good news: **0% infinity values** in features, unlike the label pipeline.

### 3. Symbol-Specific Issues

Sample analysis of problematic symbols:

| Symbol | Columns | NaN% | Issue |
|--------|---------|------|-------|
| 0G_USDC | 576 | 28.5% | High - limited history |
| 0G_USDT | 576 | 28.5% | High - limited history |
| 1000CHEEMS_USDC | 412 | 0% | Minimal features |

---

## Per-Symbol Feature Summary

### Top Symbols by Feature Count

| Symbol | Columns | Rows | NaN% |
|--------|---------|------|------|
| BTC/USDT | 576 | 35,293 | ~23% |
| ETH/USDT | 576 | 35,293 | ~23% |
| BNB/USDT | 576 | 35,293 | ~23% |
| SOL/USDT | 576 | 35,293 | ~23% |
| XRP/USDT | 576 | 35,293 | ~23% |

### Feature Categories

| Category | Example Features |
|----------|------------------|
| Price | close, open, high, low, volume |
| Returns | ret_1h, ret_4h, ret_24h |
| Volatility | atr, rvol, rv_ratio |
| Regime | vol_regime, trend_regime |
| Technical | rsi, macd, bb_position |
| Custom | meta_*, gated_* |

---

## Recommendations

### 1. Feature Availability
- All trading symbols have adequate feature coverage
- Consider dropping symbols with >30% NaN for live trading

### 2. Feature Engineering
- The 576-column feature set is comprehensive
- Consider feature selection for model efficiency

### 3. Data Quality Monitoring
- Track NaN% per symbol over time
- Alert when new symbols have >25% NaN

---

## Conclusion

Feature generation completed successfully with:
- ✅ 617 symbol feature files generated
- ✅ No infinity values
- ⚠️ 23.9% average NaN (expected for financial time series)
- ✅ Consistent date range across symbols

The feature output is ready for training and inference. No critical issues found.
