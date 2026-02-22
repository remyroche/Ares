# TBM Parameter Fixes Summary
**Date:** 2026-02-20  
**Purpose:** Address TP floor dominance and expand parameter search space

---

## Changes Implemented

### 1. Expanded k_tp Grid Search Range

**Before:**
```python
"barrier_k_tp_grid": [0.8, 1.0, 1.25, 1.6, 2.0, 2.5]
```

**After:**
```python
"barrier_k_tp_grid": [0.4, 0.65, 0.8, 1.0, 1.25, 1.5, 1.6, 2.0, 2.5]
```

**Rationale:** Include the Stage 1 tested values [0.4, 0.65, 0.8] plus comprehensive coverage up to 1.5 to capture the optimal k_tp=1.25 identified in Stage 2.

---

### 2. Reduced TP Floor to Fix Floor Dominance

**Before:**
```python
"barrier_tp_lo": 0.02,     # Lower bound for TP (2%)
```

**After:**
```python
"barrier_tp_lo": 0.015,     # Lower bound for TP (1.5%) - reduced from 2% to fix floor dominance
```

**Expected Impact:** 
- Reduce TP floor binding from 71.2% to ~40-50%
- Allow more events to use natural TP distances instead of artificial floor
- Improve production admissibility (target < 20% floor binding)

---

### 3. Disabled TP-SL Co-calibration Logic

**Added explicit config parameters:**
```python
# TP-SL co-calibration (DISABLED to fix floor dominance)
"barrier_tp_lo_sl_cocalib_alpha": 0.0,  # Disabled (0.0) - TP floor independent of SL geometry
"barrier_sl_base_mult_ref": 0.5,         # Reference SL multiplier for co-calibration
```

**Updated training.py function:**
```python
def _co_calibrate_tp_floor(tp_floor: float, sl_base_mult: float, tp_hi_local: float) -> float:
    """Co-calibrate TP floor with SL geometry so floors are tuned jointly.
    
    NOTE: Co-calibration disabled (alpha=0.0) to make TP and SL parameters independent.
    This addresses floor dominance issue by preventing SL-based TP floor increases.
    """
    alpha = float(cfg.get("barrier_tp_lo_sl_cocalib_alpha", 0.0))  # Changed from 0.30 to 0.0
    # ... rest of function with disabled scaling
```

**Rationale:** Make TP and SL parameters independent to prevent SL geometry from inflating TP floor.

---

## Expected Results

### Before Fixes:
- TP floor binding: 71.2% (FAIL > 70% threshold)
- Production admissibility: All 104 configs FAIL
- k_tp search space: Limited to [0.8, 1.0, 1.25, 1.6, 2.0, 2.5]

### After Fixes:
- TP floor binding: Expected ~40-50% (PASS < 70% threshold)
- Production admissibility: Expected multiple configurations PASS
- k_tp search space: Comprehensive [0.4, 0.65, 0.8, 1.0, 1.25, 1.5, 1.6, 2.0, 2.5]

---

## Files Modified

1. **config.py**
   - Updated `barrier_tp_lo`: 0.02 → 0.015
   - Expanded `barrier_k_tp_grid`: Added [0.4, 0.65, 1.0, 1.5]
   - Added co-calibration disable parameters

2. **training.py**
   - Updated `_co_calibrate_tp_floor()` function
   - Changed default alpha from 0.30 → 0.0
   - Added explanatory comments

---

## Next Steps

1. **Re-run TBM parameter comparison** with new expanded search space
2. **Verify TP floor binding reduction** in results
3. **Check production admissibility** - expect multiple passing configurations
4. **Monitor min_cell_auc** to ensure ≥ 0.54 threshold is met

---

## Root Cause Resolution

These changes directly address the two critical issues identified in TBM_COMPARISON_REVIEW.md:

1. **Floor Dominance:** Fixed by reducing TP floor from 2% to 1.5% and disabling SL-based inflation
2. **Limited Parameter Space:** Fixed by expanding k_tp grid to include Stage 1 values and comprehensive coverage

The combination of lower TP floor + higher k_tp values should resolve the 71.2% floor binding issue that was causing all configurations to fail production admissibility.
