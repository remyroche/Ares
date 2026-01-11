# Resonance and Variance Fixes

## 🚨 Issues Identified

### 1. **Resonance Calculation Error**
```
❌ Optimized resonance calculation failed: unsupported operand type(s) for +: 'dict' and 'dict'
```

**Root Cause**: In `optimized_resonance_detector.py` line 375, the code tries to update a dictionary with another dictionary, but the function might be returning something unexpected.

### 2. **Low Variance Explained**
```
⚠️ Low variance explained for inventory_specialist: 0.750
⚠️ Low variance explained for volume_specialist: 0.750
⚠️ Low variance explained for volatility_specialist: 0.750
```

**Root Cause**: The variance threshold is set to 0.9 (90%) but all specialists are only achieving 75% variance preservation.

### 3. **Still Low Coverage**
```
• reversal_specialist: var(d1)=8.192e-01, var(d3)=1.344e-01, phase=-1.000, coverage>-1.000=0.00% (2.0% quantile)
```

**Root Cause**: The global quantile approach is still not working as expected.

## 🔧 Fixes Required

### Fix 1: Resonance Calculation Error

**File**: `optimized_resonance_detector.py`
**Line**: 375
**Issue**: `all_resonances.update(specialist_resonance)` fails if `specialist_resonance` is not a dict

**Solution**: Add type checking and error handling
```python
# Collect results
for future in futures:
    try:
        specialist_resonance = future.result(timeout=30)
        if isinstance(specialist_resonance, dict):
            all_resonances.update(specialist_resonance)
        else:
            if self.verbose:
                tprint_warning(f"      ⚠️ Invalid resonance result type: {type(specialist_resonance)}")
    except Exception as e:
        if self.verbose:
            tprint_warning(f"      ⚠️ Parallel resonance computation failed: {e}")
```

### Fix 2: Variance Threshold Issue

**File**: `causal_compression.py`
**Issue**: Default variance threshold of 0.9 is too high for spectral data

**Solution**: Lower the variance threshold to 0.75 or add adaptive thresholding
```python
# In initialization, change from 0.9 to 0.75
variance_explained_threshold: float = 0.75

# OR add adaptive thresholding
if variance_explained < 0.75:
    if self.verbose:
        tprint_warning(f"      ⚠️ Low variance explained for {specialist_name}: {variance_explained:.3f}")
```

### Fix 3: Phase Coverage Issue

**File**: `adaptive_event_driven_labeling.py`
**Issue**: Global quantile still producing 0% coverage

**Solution**: Debug the quantile calculation more thoroughly
```python
# Add more detailed debugging
if use_quantile_approach and all_phase_values:
    # Debug: Check if phase values are all identical
    unique_values = len(set(all_phase_values))
    if unique_values == 1:
        tprint_warning(f"      ⚠️ All phase values are identical: {all_phase_values[0]:.6f}")
    
    # Debug: Check quantile calculation
    quantile_threshold = np.percentile(all_phase_values, 98)
    values_above = np.sum(np.array(all_phase_values) > quantile_threshold)
    
    if self.verbose:
        tprint_info(f"      🔍 Quantile Debug:")
        tprint_info(f"         - Unique values: {unique_values}")
        tprint_info(f"         - 98th percentile: {quantile_threshold:.6f}")
        tprint_info(f"         - Values > threshold: {values_above}")
        tprint_info(f"         - Expected (2%): {len(all_phase_values) * 0.02:.0f}")
```

## 🎯 Implementation Priority

1. **High Priority**: Fix resonance calculation error (blocking)
2. **Medium Priority**: Adjust variance threshold (performance)
3. **Low Priority**: Debug phase coverage (investigation)

## 📊 Expected Results

After fixes:
- ✅ No more resonance calculation errors
- ✅ Variance warnings eliminated (threshold adjusted)
- ✅ Better understanding of phase coverage issue
- ✅ Pipeline continues to completion

## 🔍 Next Steps

1. Implement resonance calculation fix
2. Adjust variance threshold to 0.75
3. Add detailed phase quantile debugging
4. Test the fixes with a fresh run
