# CV Ratio Normalization Fix - Analysis

## Problem Identified

The CV ratio metric was dominating the composite score calculation due to its much larger dynamic range compared to other metrics.

### Metric Ranges
| Metric | Range | Normalization |
|--------|-------|---------------|
| Temporal Smoothness | 0.0 - 1.0 | Direct use |
| Balance Score | 0.0 - 1.0 | Direct use |
| Silhouette Score | -1.0 - 1.0 | Direct use (usually 0-1) |
| **CV Ratio** | **0.2 - 100+** | **Previously: tanh(x)** |

## Issue with Previous Approach: `tanh(cv_ratio)`

The `tanh()` function saturates very quickly:

| CV Ratio | tanh(CV Ratio) | Change from Previous |
|----------|----------------|---------------------|
| 0.2 | 0.197 | - |
| 0.5 | 0.462 | +0.265 (135% increase) |
| 1.0 | 0.762 | +0.300 (65% increase) |
| 2.0 | 0.964 | +0.202 (26% increase) |
| 5.0 | 0.9999 | +0.036 (4% increase) |
| 10.0 | 1.0000 | +0.0001 (0.01% increase) |
| 22.7 | 1.0000 | +0.0 (0% increase) |
| 100.0 | 1.0000 | +0.0 (0% increase) |

**Problem**: 
- Small changes in low CV ratios (0.2 → 1.0) cause huge score changes
- Large changes in high CV ratios (5 → 100) cause almost no score change
- This makes optimization hyper-sensitive to low CV ratio changes while ignoring high CV ratio differences

## Solution: Log-Scaled Transformation `tanh(log(1 + cv_ratio))`

The new approach uses logarithmic scaling before tanh to spread values more evenly:

| CV Ratio | log(1 + CV) | tanh(log(1 + CV)) | Change from Previous |
|----------|-------------|-------------------|---------------------|
| 0.2 | 0.182 | 0.180 | - |
| 0.5 | 0.405 | 0.384 | +0.204 (113% increase) |
| 1.0 | 0.693 | 0.600 | +0.216 (56% increase) |
| 2.0 | 1.099 | 0.800 | +0.200 (33% increase) |
| 5.0 | 1.792 | 0.947 | +0.147 (18% increase) |
| 10.0 | 2.398 | 0.983 | +0.036 (4% increase) |
| 22.7 | 3.153 | 0.996 | +0.013 (1% increase) |
| 100.0 | 4.615 | 0.9998 | +0.004 (0.4% increase) |

**Benefits**:
1. **More even distribution**: Changes are more proportional across the entire range
2. **Better differentiation**: High CV ratios (10, 20, 50) now show meaningful differences
3. **Prevents domination**: Low CV ratio changes no longer dominate the composite score
4. **Still bounded**: Output remains in [0, 1] range via tanh

## Impact on Composite Score

With 35% weight on CV ratio in the composite score:

### Example 1: Low CV Ratio Change (0.5 → 1.0)
**Old approach:**
- CV contribution: 0.462×0.35 = 0.162 → 0.762×0.35 = 0.267
- Change: +0.105 (64% of the 35% max weight)

**New approach:**
- CV contribution: 0.384×0.35 = 0.134 → 0.600×0.35 = 0.210
- Change: +0.076 (22% of the 35% max weight)

### Example 2: High CV Ratio Change (5.0 → 22.7)
**Old approach:**
- CV contribution: 0.9999×0.35 = 0.350 → 1.0×0.35 = 0.350
- Change: +0.000 (0% of the 35% max weight)

**New approach:**
- CV contribution: 0.947×0.35 = 0.331 → 0.996×0.35 = 0.349
- Change: +0.018 (5% of the 35% max weight)

## Summary

The log-scaled transformation (`tanh(log(1 + cv_ratio))`) provides:
- ✅ **More balanced contribution** across all metrics
- ✅ **Better differentiation** between parameter configurations
- ✅ **Prevents CV ratio from dominating** score changes
- ✅ **More stable optimization** focusing on meaningful differences
- ✅ **Consistent behavior** across the entire CV ratio range

## Files Updated
1. `hdp_hmm_isolated_tuning.py` - Main tuning script
2. `cluster_quality_assessor.py` - Quality assessment framework
3. `hierarchical_grid_search.py` - Hierarchical search
4. `hdp_hmm_progressive_tuning.py` - Progressive tuning
5. `hdp_hmm_iterative_grid.py` - Iterative grid search

All composite score calculations now use the improved log-scaled transformation.

