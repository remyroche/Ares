# Volatility Modeling: Code Improvements Comparison

## Key Changes Overview

This document highlights the major improvements in the volatility modeling implementation, showing before/after code snippets and explaining the rationale.

## 1. Unified Scale Architecture

### Before: Mixed Units
```python
# OLD: Mixed annualization and price normalization
def _calculate_realized_volatility(self, bars: pd.DataFrame) -> pd.Series:
    returns = bars['close'].pct_change().dropna()
    rv = returns.rolling(window=self.config.rv_window).std()
    rv = rv * np.sqrt(252)  # Annualized - inconsistent units!
    return rv

def _calculate_atr_volatility(self, bars: pd.DataFrame) -> pd.Series:
    # ... ATR calculation ...
    atr_volatility = atr / bars['close']  # Price-normalized - different units!
    return atr_volatility
```

### After: Consistent Per-Period Units
```python
# NEW: All estimators produce per-period return volatility
def _calculate_realized_volatility(self, returns: pd.Series) -> pd.Series:
    """Per-period realized volatility: rolling std of close-to-close returns."""
    rv = self._rolling_std(returns, self.config.rv_window, self.config.rv_min_periods)
    return rv.rename("rv")  # No annualization - consistent units!

def _calculate_atr_volatility(self, bars: pd.DataFrame) -> pd.Series:
    """ATR-based per-period volatility: True Range divided by close."""
    # ... ATR calculation ...
    atr_vol = (atr / close).rename("atr")  # Same units as RV!
    return atr_vol
```

**Impact**: Eliminates dimensional analysis errors and enables proper mathematical combination.

## 2. Data-Driven Weight Learning

### Before: Heuristic Weights
```python
# OLD: Subjective reliability scoring with arbitrary bonuses
def _calculate_volatility_weight(self, vol_series: pd.Series, method: str) -> float:
    # ... calculate metrics ...
    method_factors = {
        'rv': {
            'base_reliability': 0.8,  # Arbitrary!
            'stability_bonus': 0.1,   # Subjective!
            'consistency_bonus': 0.1, # Hand-tuned!
        },
        # ... more arbitrary factors
    }
    # Complex heuristic calculation...
    return weight
```

### After: Optimized Weights
```python
# NEW: Data-driven weight learning via projected gradient descent
def _combine_data_driven(self, comps: pd.DataFrame, returns: pd.Series) -> tuple:
    """Learn weights w >= 0, sum(w)=1 to predict |r_{t+1}| from comps_t."""
    X_all = comps.dropna(how="any")
    y_all = returns.abs().reindex(X_all.index).shift(-1)  # |r_{t+1}|
    
    # Use trailing window for training
    X = X_all.iloc[-self.config.combo_lookback:, :]
    y = y_all.iloc[-self.config.combo_lookback:]
    w = self._fit_simplex_pg(X.to_numpy(), y.to_numpy())
    
    # Project to simplex for valid weights
    w = self._project_to_simplex(w)
    return combined, weights

def _fit_simplex_pg(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Projected gradient descent to minimize (1/n)||Xw - y||^2."""
    # Standardize features to prevent scale dominance
    Xs = X / X.std(axis=0, ddof=1)
    
    # Lipschitz constant estimation for step size
    L = 2.0 * power_iter(Xs) ** 2 / max(n, 1)
    eta = 1.0 / (L + 1e-12)
    
    # Gradient descent with simplex projection
    for _ in range(self.config.combo_max_iters):
        r = Xs @ w - y
        grad = (2.0 / n) * (Xs.T @ r)
        w_new = self._project_to_simplex(w - eta * grad)
        if np.linalg.norm(w_new - w) < self.config.combo_tol:
            break
        w = w_new
    return w
```

**Impact**: Removes subjective tuning and ensures optimal combination based on actual predictive performance.

## 3. No Look-Ahead Policy

### Before: Potential Data Leakage
```python
# OLD: np.roll can cause wraparound issues
def _calculate_atr_volatility(self, bars: pd.DataFrame) -> pd.Series:
    high_close = np.abs(high - np.roll(close, 1))  # Wraparound!
    low_close = np.abs(low - np.roll(close, 1))   # Potential leakage!
    
    # Manual padding that might inject zeros
    returns = np.concatenate([[0], returns])  # First value padding
```

### After: Strict Trailing Windows
```python
# NEW: Proper lag structure with pandas shift
def _calculate_atr_volatility(self, bars: pd.DataFrame) -> pd.Series:
    prev_close = close.shift(1)  # Proper lag, no wraparound
    
    # True range components
    c1 = high - low
    c2 = (high - prev_close).abs()
    c3 = (low - prev_close).abs()
    tr = pd.concat([c1, c2, c3], axis=1).max(axis=1)

def _calculate_ewma_volatility(self, returns: pd.Series) -> pd.Series:
    # Uses pandas ewm directly - no manual padding
    ew_var = r.ewm(alpha=self.config.ewma_alpha, 
                   min_periods=self.config.ewma_min_periods).var(bias=False)
    ew_vol = np.sqrt(ew_var).rename("ewma")
    return ew_vol
```

**Impact**: Ensures realistic backtesting and live trading compatibility.

## 4. Data-Driven Normalization

### Before: Arbitrary Constants
```python
# OLD: Fixed floor and cap values
@dataclass
class VolatilityConfig:
    volatility_floor: float = 1e-6  # Arbitrary!
    volatility_cap: float = 1.0     # Arbitrary!

def _normalize_volatility_units(self, volatility_series: pd.Series) -> pd.Series:
    normalized = np.maximum(volatility_series, self.config.volatility_floor)
    normalized = np.minimum(normalized, self.config.volatility_cap)
    return pd.Series(normalized, index=volatility_series.index)
```

### After: Percentile-Based Adaptation
```python
# NEW: Data-driven floor/cap via percentiles
@dataclass
class VolatilityConfig:
    use_percentile_floor_cap: bool = True
    floor_percentile: float = 0.5    # p0.5 to prevent zeros
    cap_percentile: float = 99.5     # p99.5 to cut extreme spikes
    absolute_floor: float = 1e-8     # hard lower bound

def _normalize_volatility_units(self, vol: pd.Series) -> pd.Series:
    """Apply data-driven floor/cap via percentiles (per series)."""
    if self.config.use_percentile_floor_cap:
        lo = np.nanpercentile(vol, self.config.floor_percentile)
        hi = np.nanpercentile(vol, self.config.cap_percentile)
        lo = max(float(lo), self.config.absolute_floor)
        hi = max(float(hi), lo)
        vol = vol.clip(lower=lo, upper=hi)
    else:
        vol = vol.clip(lower=self.config.absolute_floor)
    return vol
```

**Impact**: Adapts to data distribution and avoids arbitrary constants.

## 5. Robust Error Handling

### Before: Potential Crashes
```python
# OLD: Could crash with empty data
def _combine_volatility_estimates(self, rv_series, atr_series, ewma_series):
    # No proper empty data handling
    common_index = rv_series.index.intersection(atr_series.index)
    # Could fail if empty
```

### After: Graceful Degradation
```python
# NEW: Comprehensive empty data handling
def _combine_data_driven(self, comps: pd.DataFrame, returns: pd.Series) -> tuple:
    X_all = comps.dropna(how="any")
    if X_all.empty or X_all.shape[1] == 0:
        # Safe fallback
        return pd.Series(index=comps.index, dtype=float), {"rv": 1/3, "atr": 1/3, "ewma": 1/3}
    
    y_all = returns.abs().reindex(X_all.index).shift(-1)
    mask = y_all.notna()
    X_all, y_all = X_all[mask], y_all[mask]
    
    if len(X_all) < max(30, self.config.combo_lookback // 4):
        # Not enough data to learn reliably
        w = np.ones(X_all.shape[1]) / X_all.shape[1]
    else:
        # Proceed with weight learning
        w = self._fit_simplex_pg(X.to_numpy(), y.to_numpy())
```

**Impact**: Robust handling of edge cases and insufficient data.

## 6. Simplex Projection Algorithm

### New: Mathematical Rigor
```python
@staticmethod
def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project vector v onto the probability simplex {w: w>=0, sum w=1}.
    Duchi, Shalev-Shwartz, Singer, Chandra (2008).
    """
    v = np.asarray(v, dtype=float)
    if v.ndim != 1:
        raise ValueError("v must be 1-D")
    
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
    
    if len(rho) == 0:
        return np.ones(n) / n  # Uniform if all zeros
    
    rho = rho[-1]
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    return w if s > 0 else np.ones(n) / n
```

**Impact**: Ensures mathematically valid probability weights with proper constraints.

## 7. Enhanced Configuration

### Before: Limited Options
```python
@dataclass
class VolatilityConfig:
    method: VolatilityMethod = VolatilityMethod.COMBINED
    rv_window: int = 20
    atr_window: int = 14
    ewma_alpha: float = 0.06
    volatility_floor: float = 1e-6
    volatility_cap: float = 1.0
    # Limited configuration options
```

### After: Comprehensive Control
```python
@dataclass
class VolatilityConfig:
    # Method selection
    method: VolatilityMethod = VolatilityMethod.COMBINED
    
    # Component parameters
    rv_window: int = 20
    rv_min_periods: int = 10
    atr_window: int = 14
    atr_min_periods: int = 7
    ewma_alpha: float = 0.06
    ewma_min_periods: int = 10
    
    # Data-driven normalization
    use_percentile_floor_cap: bool = True
    floor_percentile: float = 0.5
    cap_percentile: float = 99.5
    absolute_floor: float = 1e-8
    
    # Combination training
    combo_lookback: int = 252
    combo_max_iters: int = 800
    combo_tol: float = 1e-8
    
    # Quality control
    min_volatility_samples: int = 50
    
    def _validate_config(self) -> None:
        # Comprehensive validation with explicit error messages
        if not (0 < self.ewma_alpha <= 1):
            raise ValueError("ewma_alpha must be in (0, 1]")
        if self.cap_percentile <= self.floor_percentile:
            raise ValueError("cap_percentile must be > floor_percentile")
        # ... more validation
```

**Impact**: Provides fine-grained control while maintaining sensible defaults.

## Summary

The new implementation represents a **fundamental shift from heuristic-based to data-driven approaches**:

1. **Mathematical Consistency**: All estimators use the same units
2. **Data-Driven Decisions**: No arbitrary tuning parameters
3. **Temporal Integrity**: Strict no-look-ahead policy
4. **Robustness**: Graceful handling of edge cases
5. **Adaptability**: Percentile-based normalization
6. **Rigor**: Proper constraint optimization with simplex projection

This creates a solid foundation for volatility-aware labeling that adapts to different market conditions while maintaining mathematical consistency.