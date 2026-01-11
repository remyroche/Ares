# Breakout Coverage Discrepancy Analysis

## 🚨 Issue Identified

**Target**: 2.0% coverage  
**Actual**: 0.1% coverage  
**Discrepancy**: 20x lower than expected

## 🔍 Root Cause Analysis

### 1. **Quantile Calculation Logic Issue**

In `adaptive_event_driven_labeling.py` lines 383-387:

```python
# Calculate quantile-based threshold if enabled
if use_quantile_approach and all_phase_values:
    # Get threshold for top (100 - min_coverage_percent)% values
    quantile_threshold = np.percentile(all_phase_values, 100 - min_coverage_percent)
```

**Problem**: This calculates the 98th percentile (100 - 2.0 = 98), meaning only the TOP 2% of phase values should trigger breakouts.

### 2. **Coverage Calculation Issue**

In lines 450-454:

```python
if use_quantile_approach:
    total_periods = sum(len(mask) for mask in breakout_signals.values())
    total_breakouts = sum(np.sum(mask) for mask in breakout_signals.values())
    actual_coverage = (total_breakouts / total_periods * 100) if total_periods > 0 else 0
```

**Problem**: This calculates coverage across ALL specialists combined, but each specialist individually should have 2% coverage.

### 3. **Individual Specialist Coverage**

From the log output:
```
• entropy_specialist: var(d1)=1.342e+00, var(d3)=7.953e-02, phase=-0.998, coverage>-1.000=0.02% (2.0% quantile)
```

**Issue**: Each specialist is showing only 0.02% coverage instead of 2.0%.

## 🎯 Potential Causes

### 1. **Phase Distribution Skew**
- If phase values are heavily concentrated around the mean
- The 98th percentile might be very close to the 99.98th percentile
- Result: Very few values exceed the threshold

### 2. **Phase Series Quality**
- Phase calculations might be producing very similar values
- Low variance in phase series leads to poor breakout detection
- The entropy_specialist shows `phase=-0.998` (very consistent)

### 3. **Quantile Calculation Edge Case**
- With 132,484 data points, 2% = 2,649 periods
- 98th percentile should leave exactly 2,649 periods as breakouts
- But coverage is 0.02% = ~26 periods (100x less)

### 4. **NaN Handling**
- Phase series might contain NaN values
- `all_phase_values.extend(phase_series[~np.isnan(phase_series)].tolist())` filters NaNs
- But individual specialist calculations might not handle NaNs properly

## 🔧 Investigation Steps

### 1. **Check Phase Distribution**
```python
# Add debugging to understand phase distribution
print(f"Total phase values: {len(all_phase_values)}")
print(f"Phase range: [{np.min(all_phase_values):.4f}, {np.max(all_phase_values):.4f}]")
print(f"98th percentile: {np.percentile(all_phase_values, 98):.4f}")
print(f"99th percentile: {np.percentile(all_phase_values, 99):.4f}")
print(f"99.9th percentile: {np.percentile(all_phase_values, 99.9):.4f}")
```

### 2. **Verify Individual Specialist Coverage**
```python
# Check if each specialist individually has 2% coverage
for specialist_name in specialist_names:
    phase_series = phase_series_by_specialist[specialist_name]
    breakout_mask = phase_series > quantile_threshold
    individual_coverage = np.mean(breakout_mask.astype(float))
    print(f"{specialist_name}: {individual_coverage:.2%} coverage")
```

### 3. **Examine Phase Series Quality**
```python
# Check phase series statistics
for specialist_name in specialist_names:
    phase_series = phase_series_by_specialist[specialist_name]
    print(f"{specialist_name}:")
    print(f"  - Length: {len(phase_series)}")
    print(f"  - NaN count: {np.sum(np.isnan(phase_series))}")
    print(f"  - Mean: {np.nanmean(phase_series):.4f}")
    print(f"  - Std: {np.nanstd(phase_series):.4f}")
    print(f"  - Range: [{np.nanmin(phase_series):.4f}, {np.nanmax(phase_series):.4f}]")
```

## 🛠️ Potential Fixes

### 1. **Adjust Quantile Calculation**
```python
# Current: 98th percentile (top 2%)
quantile_threshold = np.percentile(all_phase_values, 100 - min_coverage_percent)

# Fix: Use per-specialist quantiles
for specialist_name in specialist_names:
    phase_series = phase_series_by_specialist[specialist_name]
    specialist_threshold = np.percentile(phase_series[~np.isnan(phase_series)], 100 - min_coverage_percent)
    breakout_mask = phase_series > specialist_threshold
```

### 2. **Increase Coverage Target**
```python
# If 2% is too restrictive, try 5% or 10%
min_coverage_percent = 5.0  # or 10.0
```

### 3. **Add Minimum Variance Filter**
```python
# Skip specialists with low phase variance
phase_std = np.nanstd(phase_series)
if phase_std < 0.01:  # Too low variance
    continue
```

### 4. **Debug Logging Enhancement**
```python
# Add detailed logging for quantile calculation
if use_quantile_approach and all_phase_values:
    quantile_threshold = np.percentile(all_phase_values, 100 - min_coverage_percent)
    expected_breakouts = len(all_phase_values) * min_coverage_percent / 100
    tprint_info(f"   🎯 Quantile analysis:")
    tprint_info(f"      - Total phase values: {len(all_phase_values)}")
    tprint_info(f"      - Expected breakouts: {expected_breakouts:.0f}")
    tprint_info(f"      - Quantile threshold: {quantile_threshold:.4f}")
    tprint_info(f"      - Values > threshold: {np.sum(np.array(all_phase_values) > quantile_threshold)}")
```

## 📊 Next Steps

1. **Add debug logging** to understand phase distribution
2. **Check individual specialist coverage** vs combined coverage
3. **Examine phase series quality** and variance
4. **Test per-specialist quantile calculation** instead of global
5. **Consider increasing coverage target** if 2% is too restrictive

## 🎯 Expected Outcome

After investigation and fixes:
- Each specialist should have ~2% coverage individually
- Combined coverage should be higher (multiple specialists)
- Breakout signals should be more evenly distributed
- Better alignment between target and actual coverage
