# Volatility Modeling: Technical Summary

## Core Improvements

### 1. **Unified Scale Architecture** 
**Problem**: Mixed units (annualized RV vs. price-normalized ATR) caused dimensional analysis errors.

**Solution**: All estimators now produce per-period return volatility:
- **Realized Volatility**: `rolling_std(returns)` (no annualization)
- **ATR Volatility**: `ATR / close` (same units as RV)
- **EWMA Volatility**: `sqrt(EWMA_variance)` (same units as RV)

**Impact**: Enables proper mathematical combination without unit conversion errors.

### 2. **Data-Driven Weight Learning**
**Problem**: Heuristic weight calculation with arbitrary "bonuses" and subjective tuning.

**Solution**: Projected gradient descent on the probability simplex:
```python
# Objective: Minimize MSE for predicting |r_{t+1}|
# Constraints: w ≥ 0, ∑w = 1
# Method: Projected gradient descent with Lipschitz step size
```

**Algorithm**:
1. Standardize features to prevent scale dominance
2. Estimate Lipschitz constant for step size
3. Iterate: `w_new = project_to_simplex(w - η∇f(w))`
4. Stop when `||w_new - w|| < tolerance`

**Impact**: Optimal weights based on actual predictive performance, no subjective tuning.

### 3. **Strict No Look-Ahead Policy**
**Problem**: `np.roll` wraparound and improper shifting caused data leakage.

**Solution**: 
- Replace `np.roll` with `pandas.shift(1)` for proper lag structure
- Target values use `shift(-1)` then drop tail
- All operations use trailing windows only

**Impact**: Ensures realistic backtesting and live trading compatibility.

### 4. **Enhanced ATR & EWMA Safety**
**ATR Improvements**:
- Removed `np.roll` wraparound bugs
- Fixed first-bar leakage using `high_low[0]` for first element
- Proper lag structure with `close.shift(1)`

**EWMA Improvements**:
- Removed manual zero-padding that injected artifacts
- Uses pandas `ewm()` directly for proper exponential weighting
- No look-ahead in variance calculation

**Impact**: More robust and mathematically correct volatility estimates.

### 5. **Data-Driven Normalization**
**Problem**: Arbitrary 1.0 cap and fixed floor values.

**Solution**: Percentile-based adaptive normalization:
```python
# Default: p0.5 floor, p99.5 cap
# Fallback: tiny absolute floor (1e-8)
# Optional: can be disabled for raw estimates
```

**Impact**: Adapts to data distribution, avoids arbitrary constants.

### 6. **Robust Error Handling**
**Problem**: Potential crashes with empty data and inconsistent indices.

**Solution**:
- Proper empty series creation with correct index structure
- Graceful fallbacks when data is insufficient
- Equal weights when learning fails
- Comprehensive input validation

**Impact**: Robust handling of edge cases and insufficient data.

## Mathematical Foundation

### Simplex Projection (Duchi et al. 2008)
```python
def project_to_simplex(v):
    """Project v onto {w: w≥0, ∑w=1}"""
    u = sort(v)[::-1]
    cssv = cumsum(u)
    rho = find(u * arange(1,n+1) > (cssv - 1))
    theta = (cssv[rho] - 1) / (rho + 1)
    return max(v - theta, 0)
```

### Projected Gradient Descent
```python
def fit_simplex_pg(X, y):
    """Minimize ||Xw - y||² subject to w≥0, ∑w=1"""
    # Standardize features
    Xs = X / X.std(axis=0)
    
    # Estimate Lipschitz constant
    L = 2 * ||Xs||² / n
    eta = 1 / L
    
    # Gradient descent with projection
    for iteration in range(max_iters):
        grad = (2/n) * Xs.T @ (Xs @ w - y)
        w_new = project_to_simplex(w - eta * grad)
        if ||w_new - w|| < tolerance:
            break
        w = w_new
    return w
```

## Configuration Enhancements

### New Parameters
```python
@dataclass
class VolatilityConfig:
    # Data-driven normalization
    use_percentile_floor_cap: bool = True
    floor_percentile: float = 0.5
    cap_percentile: float = 99.5
    absolute_floor: float = 1e-8
    
    # Weight learning
    combo_lookback: int = 252      # Training window
    combo_max_iters: int = 800     # Max iterations
    combo_tol: float = 1e-8        # Convergence tolerance
    
    # Quality control
    min_volatility_samples: int = 50
```

### Validation
- Comprehensive parameter validation
- Explicit error messages
- Range checks for all parameters
- Minimum sample requirements

## Performance Characteristics

### Computational Complexity
- **Weight Learning**: O(iterations × features × samples)
- **Simplex Projection**: O(features log features)
- **Component Calculation**: O(samples × window_size)

### Memory Usage
- Efficient pandas operations
- Proper dtype management (float64)
- Index alignment to avoid bloat

### Scalability
- Vectorized operations where possible
- Optional matrix operations integration
- Configurable training window size

## Quality Metrics

### Statistical Measures
- **Consistency**: `1 - mean_abs_change / mean_volatility`
- **Stability**: `1 - std / mean`
- **Percentiles**: p5, p10, p25, p50, p75, p90, p95

### Validation Checks
- Input data validation (OHLC, finite values)
- Output bounds checking
- Quality metric validation
- Error handling and fallbacks

## Migration Path

### Backward Compatibility
- Same public API interface
- Configuration defaults preserve behavior
- Graceful handling of legacy parameters

### Performance Impact
- Slightly increased computation for weight learning
- Offset by improved accuracy
- Optional smoothing can be disabled

## Conclusion

The enhanced volatility modeling provides:

1. **Mathematical Rigor**: Consistent units and proper statistical foundations
2. **Data-Driven Decisions**: No arbitrary tuning or subjective bonuses
3. **Temporal Integrity**: Strict no-look-ahead for realistic backtesting
4. **Robustness**: Graceful handling of edge cases and insufficient data
5. **Adaptability**: Percentile-based normalization adapts to data characteristics

This creates a solid foundation for volatility-aware labeling that can adapt to different market conditions while maintaining mathematical consistency and avoiding common pitfalls in volatility estimation.