# Feature Exclusion List for Regime Models Training

Based on analysis of the regime models training pipeline logs showing 418 non-finite values consistently, here's the recommended list of features to exclude and fix:

## 🚨 **CRITICAL ISSUES IDENTIFIED**

### 1. **Non-Finite Values (418 occurrences)**
**Root Cause**: Early rows in rolling calculations produce NaN values, plus some features inherently create infinite values.

**Features to Exclude**:
```python
EXCLUDED_FEATURES = [
    # Time-based features that create NaN in early rows
    'open_time', 'close_time', 'hour', 'day_of_week', 'is_weekend',
    
    # Price range features that create NaN without sufficient history
    'price_range', 'price_range_pct', 'body_size', 'body_size_pct',
    
    # Return-based features that create NaN in early periods
    'close_return', 'close_log_return', 'volume_return', 'volume_log_return',
    
    # Calculated features requiring sufficient lookback
    'quote_volume', 'trades',  # Order book columns not available
]
```

### 2. **VectorBT Indicator Failures**
**Root Cause**: `module 'vectorbt' has no attribute 'EMA', 'ADX'` - API incompatibility.

**Failed Indicators**:
```python
VECTORBT_FAILED_INDICATORS = [
    'vectorbt_ema_*',     # EMA indicators
    'vectorbt_adx_*',     # ADX indicators  
    'vectorbt_trend_comprehensive_*',  # Complex trend indicators
]
```

**Fix Required**: Update VectorBT compatibility layer or use pandas fallbacks.

### 3. **Missing Regime-Specific Features**
**Critical**: EWMA-style regime features appear to be missing from feature set.

**Required Regime Features**:
```python
REQUIRED_REGIME_FEATURES = [
    # EWMA (Exponentially Weighted Moving Average) features
    'ewma_10_close', 'ewma_20_close', 'ewma_50_close',
    
    # Adaptive moving averages for regime detection
    'kama_10_close', 'kama_20_close', 'mama_10_close',
    
    # Regime transition probabilities
    'regime_transition_prob_1', 'regime_transition_prob_5',
    
    # Multi-timeframe regime features
    'regime_strength_5min', 'regime_strength_15min', 'regime_strength_1h',
]
```

## 🔧 **IMMEDIATE FIXES REQUIRED**

### Fix 1: Data Validation Enhancement
```python
# In BalancedOptimizationStrategy._optimize_features()
def _handle_non_finite_values(self, data: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """Handle non-finite values gracefully."""
    # Count non-finite values before fixing
    non_finite_count = (~np.isfinite(data[feature_name])).sum()
    
    if non_finite_count > 0:
        tprint(f"⚠️ Found {non_finite_count} non-finite values in {feature_name}")
        
        # For time-based features, fill with reasonable defaults
        if feature_name in ['hour', 'day_of_week']:
            data[feature_name] = data[feature_name].fillna(method='bfill').fillna(0)
        
        # For price-based features, use forward fill
        elif feature_name in ['close_return', 'volume_return']:
            data[feature_name] = data[feature_name].fillna(0)
        
        # For range features, use small positive values
        elif 'range' in feature_name:
            data[feature_name] = data[feature_name].fillna(data['close'].rolling(20).std() * 0.1)
        
        # Drop completely invalid features
        elif feature_name in ['open_time', 'close_time', 'quote_volume', 'trades']:
            data = data.drop(columns=[feature_name])
            tprint(f"🗑️ Dropped invalid feature: {feature_name}")
    
    return data
```

### Fix 2: VectorBT Compatibility Update
```python
# In src/utils/vectorbt_compat.py
def get_ema(data, span=None, com=None, adjust=None):
    """VectorBT-compatible EMA implementation with fallback."""
    try:
        if hasattr(vbt, 'ta') and hasattr(vbt.ta, 'ema'):
            return vbt.ta.ema(data, span=span, com=com, adjust=adjust)
    except (AttributeError, ImportError):
        # Fallback to pandas
        return data.ewm(span=span, com=com, adjust=adjust).mean()

def get_adx(high, low, close, window=14):
    """VectorBT-compatible ADX implementation with fallback."""
    try:
        if hasattr(vbt, 'ta') and hasattr(vbt.ta, 'adx'):
            return vbt.ta.adx(high, low, close, window=window)
    except (AttributeError, ImportError):
        # Fallback to manual ADX calculation
        return _calculate_adx_fallback(high, low, close, window)
```

### Fix 3: Regime Feature Integration
```python
# In feature generation pipeline
def ensure_regime_features(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure required regime features are present."""
    
    # Check for EWMA features
    ewma_features = []
    for period in [10, 20, 50]:
        ewma_col = f'ewma_{period}_close'
        if ewma_col not in data.columns:
            data[ewma_col] = data['close'].ewm(span=period).mean()
            ewma_features.append(ewma_col)
    
    # Check for KAMA features
    kama_features = []
    for period in [10, 20]:
        kama_col = f'kama_{period}_close'
        if kama_col not in data.columns:
            data[kama_col] = _calculate_kama(data['close'], period)
            kama_features.append(kama_col)
    
    # Add regime strength indicators
    if 'regime_strength_5min' not in data.columns:
        data['regime_strength_5min'] = _calculate_regime_strength(data, '5min')
    
    tprint(f"✅ Added regime features: {ewma_features + kama_features}")
    return data
```

## 📊 **PERFORMANCE OPTIMIZATIONS**

### Memory Optimization
```python
# Reduce memory pressure during feature generation
class MemoryOptimizedFeatureGenerator:
    def __init__(self):
        self.chunk_size = 1000  # Process in chunks
        
    def generate_features_chunked(self, data: pd.DataFrame):
        """Generate features in memory-efficient chunks."""
        results = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data.iloc[i:i+self.chunk_size]
            chunk_features = self._generate_features(chunk)
            results.append(chunk_features)
            
            # Free memory
            del chunk
            import gc
            gc.collect()
        
        return pd.concat(results, ignore_index=True)
```

### CPU Optimization
```python
# Optimize CPU-intensive calculations
class CPUOptimizedCalculator:
    def __init__(self):
        self.use_numba = True  # Enable JIT compilation
        
    @numba.jit(nopython=True)
    def fast_rolling_mean(self, arr, window):
        """Fast rolling mean using numba."""
        return self._fast_rolling_mean(arr, window)
        
    def _fast_rolling_mean(self, arr, window):
        """Optimized rolling mean calculation."""
        result = np.empty(len(arr))
        cumulative_sum = np.cumsum(arr)
        for i in range(window, len(arr)):
            result[i] = (cumulative_sum[i] - cumulative_sum[i-window]) / window
        return result
```

## ✅ **VERIFICATION CHECKLIST**

### Before Running Pipeline:
- [ ] Review excluded features list
- [ ] Apply VectorBT compatibility fixes
- [ ] Ensure regime features are included
- [ ] Implement memory optimization
- [ ] Implement CPU optimization
- [ ] Test with smaller dataset first

### During Pipeline:
- [ ] Monitor non-finite value counts (should be < 50)
- [ ] Monitor CPU usage (should be < 85%)
- [ ] Monitor memory usage (should be < 80%)
- [ ] Verify VectorBT indicators work without fallbacks
- [ ] Confirm regime features are generated

### Expected Results:
- Non-finite values: **< 50** (vs current 418)
- CPU usage: **< 85%** (vs current 100%)
- Memory usage: **< 80%** (vs current 83%)
- VectorBT failures: **0** (vs current multiple)
- Regime features: **Complete set** (vs current missing)

## 🚀 **IMPLEMENTATION PRIORITY**

1. **URGENT**: Apply exclusion list to prevent 418 NaN values
2. **HIGH**: Fix VectorBT compatibility for EMA/ADX indicators
3. **HIGH**: Add missing EWMA/KAMA regime features
4. **MEDIUM**: Implement memory optimization to reduce pressure
5. **MEDIUM**: Implement CPU optimization for intensive calculations

## 📝 **NOTES**

- The 418 non-finite values suggest systematic issue, not random
- Missing regime EWMA features could significantly impact regime detection accuracy
- VectorBT failures suggest version compatibility issues that need addressing
- Performance issues compound each other (high CPU → high memory → more pressure)

## 🔍 **DEBUGGING COMMANDS**

```bash
# Check specific features causing NaN values
python -c "
import pandas as pd
data = pd.read_csv('your_data.csv')
for col in ['open_time', 'close_time', 'hour', 'price_range']:
    if col in data.columns:
        print(f'{col}: {(~data[col].isfinite()).sum()} NaN values')
"

# Test VectorBT compatibility
python -c "
try:
    import vectorbt as vbt
    print('✅ VectorBT available')
    print(f'EMA available: {hasattr(vbt.ta, \"ema\")}')
    print(f'ADX available: {hasattr(vbt.ta, \"adx\")}')
except Exception as e:
    print(f'❌ VectorBT error: {e}')
"
```

This exclusion list and fix set should resolve the major issues in your regime models training pipeline.
