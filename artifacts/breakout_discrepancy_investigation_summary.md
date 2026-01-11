# Breakout Coverage Discrepancy Investigation Summary

## 🚨 Issue Summary

**Problem**: Structural breakout detection targeting 2.0% coverage but only achieving 0.1% coverage (20x discrepancy)

**Log Output**:
```
✅ Structural breakouts identified (2.0% quantile):
   - Breakout signals: 8
   - Dominant specialist: entropy_specialist_d1_d3_breakout
   - Breakout strength: 0.001
   - Actual coverage: 0.1% (target: 2.0%)
```

## 🔍 Root Cause Analysis

### 1. **Global vs Per-Specialist Quantile Issue**

**Current Logic**:
```python
# Global quantile calculation across ALL specialists
quantile_threshold = np.percentile(all_phase_values, 100 - min_coverage_percent)

# Applied to each specialist individually
breakout_mask = phase_series > quantile_threshold
```

**Problem**: Global 98th percentile might be much higher than individual specialist 98th percentiles, resulting in very few breakouts per specialist.

### 2. **Phase Distribution Concentration**

From the logs:
- `entropy_specialist: phase=-0.998` (very consistent phase)
- Low variance in phase series
- Most values clustered around the mean

### 3. **Mathematical Discrepancy**

**Expected**: 2% of 132,484 = 2,649 breakout periods per specialist
**Actual**: 0.1% of 132,484 = 132 breakout periods per specialist
**Ratio**: 20x lower than expected

## 🛠️ Implemented Debug Logging

### 1. **Global Quantile Analysis**
Added detailed logging for:
- Total phase values count
- Expected breakouts calculation
- Actual values above threshold
- Phase distribution statistics (mean, std, range)

### 2. **Per-Specialist Debug**
Added detailed logging for entropy_specialist:
- Individual specialist threshold vs global threshold
- Expected vs actual breakouts per specialist
- Phase distribution per specialist

## 🔧 Potential Fixes

### Fix 1: Per-Specialist Quantile Calculation

```python
# Instead of global quantile, use per-specialist quantiles
for specialist_name in specialist_names:
    phase_series = phase_series_by_specialist[specialist_name]
    specialist_phase_values = phase_series[~np.isnan(phase_series)]
    specialist_threshold = np.percentile(specialist_phase_values, 100 - min_coverage_percent)
    breakout_mask = phase_series > specialist_threshold
```

### Fix 2: Adjust Coverage Target

```python
# Increase from 2% to 5% or 10% if phase variance is too low
min_coverage_percent = 5.0  # or 10.0
```

### Fix 3: Hybrid Approach

```python
# Use per-specialist quantiles but ensure minimum coverage
specialist_threshold = np.percentile(specialist_phase_values, 100 - min_coverage_percent)
min_threshold = np.percentile(all_phase_values, 95)  # Global minimum
final_threshold = max(specialist_threshold, min_threshold)
```

## 📊 Expected Debug Output

With the new debug logging, we should see:

```
🎯 Quantile Analysis:
   - Total phase values: 1,184,356  (11 specialists × 132,484 periods)
   - Expected breakouts: 23,687
   - Quantile threshold: -0.9980
   - Values > threshold: 1,184
   - Phase range: [-1.0000, -0.9950]
   - Phase mean: -0.9985
   - Phase std: 0.0012

🔍 entropy_specialist Debug:
   - Phase values: 132,484
   - Expected breakouts: 2,650
   - Specialist threshold: -0.9982
   - Values > specialist threshold: 2,650
   - Using global threshold: -0.9980
   - Values > global threshold: 132
   - Phase range: [-1.0000, -0.9950]
   - Phase mean: -0.9985
   - Phase std: 0.0012
```

## 🎯 Next Steps

1. **Run the pipeline** with new debug logging to see actual phase distributions
2. **Analyze the output** to confirm if global vs per-specialist quantile is the issue
3. **Implement the appropriate fix** based on the debug results
4. **Verify the fix** by checking coverage reaches ~2% per specialist

## 📈 Success Criteria

- [ ] Debug logging shows phase distribution details
- [ ] Coverage discrepancy explained by quantile calculation method
- [ ] Fix implemented and tested
- [ ] Final coverage: ~2% per specialist (±0.5%)
- [ ] Total breakout signals: ~21,000 (11 specialists × 2,650)
- [ ] Breakout strength improves with more signals

## 🔬 Investigation Questions to Answer

1. **Is the phase variance too low** across all specialists?
2. **Are phase values too similar** between specialists?
3. **Is the global quantile approach** fundamentally flawed for this use case?
4. **Should we use per-specialist quantiles** instead?
5. **What is the optimal coverage target** for this dataset?

The debug logging will help answer these questions and guide the fix implementation.
