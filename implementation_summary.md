# Model Improvements - Implementation Summary

## ✅ Completed Tasks

### 1. Regularization Enhancement
**File**: `meta_model.py`
- Expanded alpha grid from [0.01, 0.1, 0.3, 1.0, 3.0] to [0.01, 0.1, 0.3, 1.0, 3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0]
- All strategies now use the same wider grid (uniform approach per user feedback)
- Expected impact: Better regularization tuning, especially for SHORT_TF overfitting

### 2. Gated Entry Features
**Files**: `features.py`, `config.py`

Added 4 new features for LONG_MR entry enhancement:

1. **`bounce_signal`**: Detects bounce after extreme move
   - Formula: `(close[t] > close[t-1]) AND (|ret1h| > 2×ATR)`
   - Purpose: Confirm reversal momentum

2. **`trap_strength`**: Custom trap quality metric
   - Combines: price at extreme + volume spike
   - Adapts to trend direction (long trap vs short trap)
   - Purpose: Identify high-quality reversal setups

3. **`volume_capitulation`**: Volume spike detection
   - Formula: `volume > 2× MA(24h)`
   - Purpose: Detect capitulation/exhaustion

4. **`entry_quality_composite`**: Weighted combination
   - Formula: `0.4×bounce + 0.3×trap + 0.3×volume`
   - Purpose: Single quality score for entry timing

All 4 features added to `meta_feature_keys` for meta model training.

### 3. Expanded TP/SL Grids
**File**: `training.py` (line 1788-1789)

- **`tp_mult_grid`**: [0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0] (was [1.0, 1.5, 2.0, 2.5, 3.0])
- **`sl_mult_grid`**: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0] (was [0.5, 1.0, 1.5, 2.0])
- **`trail_mult_grid`**: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0] (NEW - but needs implementation)

Expected impact: Find optimal risk parameters per strategy bucket.

---

## ⚠️ Pending Implementation

### 1. Trail Multiplier Optimization
**Status**: Grid parameter added to training.py, but function doesn't support it yet

**Required changes**:

#### A. Update `run_tp_sl_selection_fast` signature
**File**: `optimise_tpsl_ratio.py` (line ~758)

```python
def run_tp_sl_selection_fast(
    # ... existing params ...
    tp_mult_grid: Iterable[float] = (0.6, 0.8, 1.0, 1.25, 1.5),
    sl_mult_grid: Iterable[float] = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0),
    trail_mult_grid: Iterable[float] = (0.3, 0.5, 0.7, 1.0),  # NEW
    # ... rest of params ...
):
```

#### B. Add trail_mult to grid search loop
**File**: `optimise_tpsl_ratio.py` (line ~907)

Current loop:
```python
for tp_mult in tp_mult_grid:
    for sl_mult in sl_mult_grid:
        # ... optimization logic ...
```

Needs to become:
```python
for tp_mult in tp_mult_grid:
    for sl_mult in sl_mult_grid:
        for trail_mult in trail_mult_grid:
            # ... optimization logic ...
            # Store trail_mult in results
```

#### C. Update result storage
**File**: `training.py` (line ~1815)

Current (hardcoded):
```python
bucket_risk = {
    "trail_mult": 0.5 * summary.final_tp_mult,  # HARDCODED
    # ...
}
```

Should become:
```python
bucket_risk = {
    "trail_mult": summary.final_trail_mult,  # Use optimized value
    # ...
}
```

### 2. Dynamic Trail Implementation (Advanced)
**Status**: Not started

**Approach A - Vol-based trail** (linear formula):
```python
# In simulate_trade_hourly
vol_adjustment = a * predicted_vol_6h + b
effective_trail_mult = trail_mult * vol_adjustment
```

**Approach C - Profit-based trail** (accelerating):
```python
profit_ratio = (extreme - entry_px) / (activation_mult * barrier_pct)
if profit_ratio > 2.0:
    convexity_mult = 0.7  # Tighten to lock gains
elif profit_ratio > 1.5:
    convexity_mult = 0.85
else:
    convexity_mult = 1.0

effective_trail_mult = trail_mult * vol_adjustment * convexity_mult
```

**Historical optimization**: Find optimal (a, b) coefficients through backtesting.

### 3. Confidence vs Win Rate Analysis
**Status**: Not started

**Implementation**: Add to backtest reporting (likely in `pipeline_steps.py`)

```python
# After backtest completion
confidence_bins = pd.qcut(df["score"].abs(), q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

for bin_label in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
    bin_trades = df[confidence_bins == bin_label]
    wr = (bin_trades["ret"] > 0).mean()
    avg_ret = bin_trades["ret"].mean()
    print(f"  {bin_label}: n={len(bin_trades)}, WR={wr:.1%}, Avg Ret={avg_ret:+.4f}")
```

### 4. Vol_z Correlation Reporting
**Status**: Not started

**Implementation**: Add to backtest reporting

```python
# Vol_z binning
vol_bins = pd.cut(df["vol_z"], bins=[0, 1.0, 2.0, 3.0, 100], labels=["Low", "Med", "High", "Extreme"])

for bin_label in ["Low", "Med", "High", "Extreme"]:
    bin_trades = df[vol_bins == bin_label]
    wr = (bin_trades["ret"] > 0).mean()
    pnl = bin_trades["pnl"].sum()
    print(f"  {bin_label} vol_z: n={len(bin_trades)}, WR={wr:.1%}, PnL={pnl:+.4f}")

# Correlation
corr = df[["vol_z", "ret"]].corr().iloc[0, 1]
print(f"\nVol_z-Return Correlation: {corr:+.4f}")
```

### 5. Threshold Tuning
**Status**: Not started

**Implementation**: Add threshold grid to backtest optimization

```python
threshold_grid = {
    "long_mr": np.arange(0.01, 0.20, 0.02),
    "long_tf": np.arange(0.01, 0.15, 0.02),
    "short_mr": np.arange(-0.20, -0.01, 0.02),
    "short_tf": np.arange(-0.15, -0.01, 0.02),
}

# Grid search for optimal thresholds
# Store in signal_params
```

---

## Next Steps

1. **Immediate**: Implement trail_mult optimization in `optimise_tpsl_ratio.py`
2. **Short-term**: Add confidence and vol_z reporting to backtests
3. **Medium-term**: Implement dynamic trail logic (vol-based + profit-based)
4. **Long-term**: Add threshold tuning to optimization pipeline

## Expected Outcomes

After all improvements:
- SHORT_TF: Correlation -0.019 → +0.15, WR 32.6% → 44%
- LONG_MR: WR 34% → 42%
- LONG_TF: PnL +0.102 → +0.18
- Trail survival: 33% → 50%
- Overall PnL: 0.0012 → 0.10+
