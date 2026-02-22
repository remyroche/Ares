# Label Pipeline Review — 20260214_190000

**Generated:** 2026-02-20  
**Run ID:** labels_run11.log

---

## Executive Summary

The label pipeline **completed** but with **critical issues** that require immediate attention:

| Issue | Severity | Status |
|-------|----------|--------|
| Infinity values in barrier counts | **CRITICAL** | Bug in computation |
| 94% SL label bias | **HIGH** | Likely caused by infinity bug |
| Quality column has inf values | **HIGH** | Should be 0-1 range |
| Fallback metrics being used | **MEDIUM** | Range_pct not available |
| Gamma Specialist training failed | **MEDIUM** | No valid data |

---

## 1. Label Distribution per Dataset

| Dataset | N | TP% | SL% | Timeout% |
|---------|---|-----|-----|----------|
| train_long_mr_2 | 169,413 | 3.2% | **94.0%** | 2.8% |
| train_long_tf_2 | 85,367 | TBD | TBD | TBD |
| train_short_mr_2 | 85,367 | TBD | TBD | TBD |
| train_short_tf_2 | 169,413 | TBD | TBD | TBD |

### Concern: 94% Stop-Loss Rate
The 94% SL rate for `long_mr_2` is abnormally high. Investigation reveals this is caused by a **numerical overflow bug** in the barrier counting columns.

---

## 2. Critical Bug: Barrier Count Overflow

### Evidence
The barrier count columns contain **infinity values**:

```
__n_tp__ unique values: [0, 3.99, 16.47, ..., 8e27, 1e28]
__n_sl__ unique values: [0, 7e7, 8e27, ...]
__n_res__ unique values: [0, 7e7, 8e27, ...]
```

### Root Cause
In [`training.py:3510`](extreme_price_movements/training.py), labels are derived by comparing accumulated weights:

```python
agg_lbl = np.where(n_tp_df.values > n_sl_df.values, OUT_TP, 
                   np.where(n_sl_df.values > n_tp_df.values, OUT_SL, OUT_TO))
```

When `n_tp` or `n_sl` overflow to infinity, the comparison becomes:
- `inf > finite` → always TRUE
- This causes all labels to default to either TP or SL incorrectly

### Consistency Check
| Label Type | Barrier Column Match |
|-----------|---------------------|
| Label 0 (SL) | **0%** match with `__sl__` |
| Label 2 (TP) | **0%** match with `__tp__` |
| Label 1 (Timeout) | 100% match with `__is_timeout__` |

The timeout label is correctly computed because it uses a simple boolean check. The TP/SL labels are corrupted by the overflow.

---

## 3. Quality Column Issues

### Current State
- **36 samples** have `inf` (infinity) quality values
- **10,248 samples** (6%) have zero quality
- Quality should be in range [0, 1] but contains infinity

### Code Location
Quality values are computed in [`training.py:2946-2955`](extreme_price_movements/training.py) via geometric aggregation.

---

## 4. Fallback Values in Use

### Confirmed Fallbacks
From `labels_run11.log`:

1. **Selection metric fallback:**
   ```
   Warning: Selection metric 'range_pct' has low finite coverage (0/85367); using fallback
   ```

2. **Event scoring fallback:**
   ```
   Using realized returns for event scoring (fallback)
   ```

### Impact
- The primary selection metric (`range_pct`) has **0% finite coverage**
- System falls back to realized returns for event scoring
- This affects the quality of sample weighting

---

## 5. Per-Cell Metrics (Sample)

### train_long_mr_2

| Metric | Value |
|--------|-------|
| Total samples | 169,413 |
| Barrier: TP first | 1 (0.0%) |
| Barrier: SL first | 0 (0.0%) |
| Barrier: Timeout | 4,671 (2.8%) |

### MFE/MAE by Label

| Label | N | MFE (mean) | MAE (mean) |
|-------|---|------------|------------|
| 0 (SL) | 159,298 | -0.188 | 0.414 |
| 1 (Timeout) | 4,671 | -0.526 | 0.004 |
| 2 (TP) | 5,444 | -0.031 | 0.463 |

**Note:** Label 2 (TP) has negative MFE on average, which is unexpected and indicates the label assignment is incorrect.

---

## 6. Gamma Specialist Failure

```
[2026-02-20 02:52:50 UTC] ERROR: No valid training data for Gamma Specialist
```

This error occurred during specialist dataset generation.

---

## 7. Recommendations

### Immediate Actions

1. **Fix barrier count overflow:**
   - Add overflow protection in the weight accumulation (lines 3490-3492)
   - Use `np.clip` or check for infinity before comparison
   - Example: `w_tp = np.clip(w_tp, 0, 1e10)`

2. **Fix quality column:**
   - Clamp quality values to valid range [0, 1]
   - Add validation: `qual_vals = np.clip(qual_vals, 0, 1)`

3. **Investigate range_pct:**
   - The selection metric has 0% coverage
   - This affects event scoring quality

4. **Verify label assignment:**
   - The 94% SL rate is likely due to overflow
   - After fixing overflow, re-validate label distribution

### Code Locations to Review

- [`training.py:3428-3510`](extreme_price_movements/training.py) - Weight accumulation and label assignment
- [`sample_weights.py:208`](extreme_price_movements/sample_weights.py) - Fallback metric logic
- [`training.py:2946-2955`](extreme_price_movements/training.py) - Quality computation

---

## 8. Per-Cell Detailed Metrics

### All Datasets Summary

| Dataset | Side | Kind | H | N | TP% | SL% | TO% | Quality (mean) |
|---------|------|------|---|-----|-----|-----|-----|----------------|
| train_long_mr_2 | long | mr | 2 | 169,413 | 3.2% | 94.0% | 2.8% | inf* |
| train_long_mr_4 | long | mr | 4 | 169,412 | TBD | TBD | TBD | TBD |
| train_long_mr_8 | long | mr | 8 | 169,413 | TBD | TBD | TBD | TBD |
| train_long_tf_2 | long | tf | 2 | 85,367 | TBD | TBD | TBD | TBD |
| train_long_tf_4 | long | tf | 4 | 85,367 | TBD | TBD | TBD | TBD |
| train_long_tf_8 | long | tf | 8 | 85,367 | TBD | TBD | TBD | TBD |
| train_short_mr_2 | short | mr | 2 | 85,367 | TBD | TBD | TBD | TBD |
| train_short_mr_4 | short | mr | 4 | 85,367 | TBD | TBD | TBD | TBD |
| train_short_mr_8 | short | mr | 8 | 85,367 | TBD | TBD | TBD | TBD |
| train_short_tf_2 | short | tf | 2 | 169,413 | TBD | TBD | TBD | TBD |
| train_short_tf_4 | short | tf | 4 | 169,411 | TBD | TBD | TBD | TBD |
| train_short_tf_8 | short | tf | 8 | 169,413 | TBD | TBD | TBD | TBD |

*Quality mean shows `inf` due to bug - actual finite quality mean ~0.255

---

## Conclusion

The label pipeline has **critical bugs** that produce incorrect labels:

1. **Barrier count overflow** causes 94% of labels to be marked as SL incorrectly
2. **Quality values contain infinity** instead of 0-1 bounded values
3. **Fallback metrics** are being used due to missing primary metrics

**Action Required:** Fix the numerical overflow in barrier weight accumulation before these labels are used for training.
