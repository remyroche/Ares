# HDBSCAN Quick Tuning Guide

**Quick Reference** for tuning HDBSCAN regime detection parameters.

---

## 🎛️ Key Parameters to Adjust

### 1. **Target More/Fewer Regimes**

**Location**: `regime_discovery_config.py`

```python
# Want MORE regimes (6-10)?
min_cluster_size_pct: float = 0.003  # Lower = more clusters
min_cluster_size_floor: int = 3      # Lower = more clusters
target_regime_count_min: int = 6
target_regime_count_max: int = 10

# Want FEWER regimes (2-4)?
min_cluster_size_pct: float = 0.01   # Higher = fewer clusters
min_cluster_size_floor: int = 10     # Higher = fewer clusters
target_regime_count_min: int = 2
target_regime_count_max: int = 4
```

---

### 2. **Adjust Regime Count Penalty**

**Location**: `regime_discovery_config.py` or `enhanced_hyperparameter_optimizer.py`

```python
# Stronger enforcement of target range
regime_count_penalty: float = 0.3  # Higher penalty = stricter target

# Weaker enforcement (focus more on quality metrics)
regime_count_penalty: float = 0.1  # Lower penalty = more flexibility

# Disable completely (pure quality optimization)
enable_regime_count_objective: bool = False
```

**Rule of Thumb**: 
- `0.1-0.15`: Soft guidance
- `0.2`: Default (balanced)
- `0.3-0.4`: Strong enforcement

---

### 3. **Outlier Handling**

**Location**: `regime_discovery_config.py`

```python
# More aggressive outlier removal (cleaner clusters)
winsorize_limits: Tuple[float, float] = (0.05, 0.95)  # Remove top/bottom 5%

# Less aggressive (preserve extreme values)
winsorize_limits: Tuple[float, float] = (0.01, 0.99)  # Remove top/bottom 1%

# Disable winsorization
enable_winsorization: bool = False
```

---

### 4. **Rolling Normalization Window**

**Location**: `regime_discovery_config.py` or `optimized_preprocessor.py`

```python
# Shorter window = more responsive to recent changes
rolling_normalization_window: int = 30  # 30 bars (fast adaptation)

# Longer window = more stable, less sensitive
rolling_normalization_window: int = 120  # 120 bars (slow adaptation)

# Disable rolling normalization
enable_rolling_normalization: bool = False
```

**Rule of Thumb**:
- **30 bars**: High-frequency trading, fast regime changes
- **60 bars** (default): Balanced for most use cases
- **120 bars**: Longer-term regimes, more stability

---

### 5. **Cluster Selection Method**

**Location**: `regime_discovery_config.py` or `optimized_hdbscan_regime_discovery.py`

```python
# More balanced clusters (recommended for 4-8 regimes)
cluster_selection_method: str = 'leaf'

# More aggressive merging (tends toward fewer clusters)
cluster_selection_method: str = 'eom'
```

**When to Use**:
- **`leaf`**: Want 4-8+ regimes, balanced sizes
- **`eom`**: Want fewer regimes (2-4), quality over quantity

---

### 6. **Distance Metric**

**Location**: `regime_discovery_config.py` or `optimized_hdbscan_regime_discovery.py`

```python
# Robust to outliers (recommended)
metric: str = 'manhattan'

# Standard Euclidean distance
metric: str = 'euclidean'

# Normalized directional data
metric: str = 'cosine'
```

---

### 7. **Quantile Transformation**

**Location**: `optimized_preprocessor.py`

```python
# Ensure Gaussian features (recommended)
enable_quantile_transformation: bool = True
quantile_output_distribution: str = 'normal'

# Uniform distribution (alternative)
quantile_output_distribution: str = 'uniform'

# Disable quantile transformation
enable_quantile_transformation: bool = False
```

---

## 🚨 Common Issues & Fixes

### Issue 1: **Still Only 2 Regimes**

**Fixes** (try in order):

1. **Lower min_cluster_size**:
   ```python
   min_cluster_size_pct: float = 0.003
   min_cluster_size_floor: int = 3
   ```

2. **Increase regime count penalty**:
   ```python
   regime_count_penalty: float = 0.3
   ```

3. **Change to leaf method**:
   ```python
   cluster_selection_method: str = 'leaf'
   ```

4. **Try different metric**:
   ```python
   metric: str = 'manhattan'  # or 'cosine'
   ```

---

### Issue 2: **Too Many Regimes (12+)**

**Fixes**:

1. **Increase min_cluster_size**:
   ```python
   min_cluster_size_pct: float = 0.01
   min_cluster_size_floor: int = 10
   ```

2. **Adjust target range**:
   ```python
   target_regime_count_min: int = 4
   target_regime_count_max: int = 6  # Narrower range
   ```

3. **Use EOM method**:
   ```python
   cluster_selection_method: str = 'eom'
   ```

---

### Issue 3: **High Noise Ratio (>20%)**

**Fixes**:

1. **More aggressive winsorization**:
   ```python
   winsorize_limits: Tuple[float, float] = (0.05, 0.95)
   ```

2. **Enable quantile transformation**:
   ```python
   enable_quantile_transformation: bool = True
   ```

3. **Adjust rolling window**:
   ```python
   rolling_normalization_window: int = 90  # Longer window
   ```

---

### Issue 4: **Poor Economic Separation**

**Fixes**:

1. **Use rolling normalization**:
   ```python
   enable_rolling_normalization: bool = True
   rolling_normalization_window: int = 60
   ```

2. **Try RobustScaler**:
   ```python
   scaling_method: str = 'robust'
   ```

3. **Lower correlation threshold** (more features):
   ```python
   correlation_threshold: float = 0.75
   ```

---

## 🎯 Recommended Presets

### Preset 1: **Aggressive (6-8 regimes)**

```python
# regime_discovery_config.py
min_cluster_size_pct: float = 0.003
min_cluster_size_floor: int = 3
cluster_selection_method: str = 'leaf'
cluster_selection_epsilon: float = 0.0
target_regime_count_min: int = 6
target_regime_count_max: int = 8
regime_count_penalty: float = 0.3
```

---

### Preset 2: **Balanced (4-6 regimes)** ✅ DEFAULT

```python
# regime_discovery_config.py
min_cluster_size_pct: float = 0.005
min_cluster_size_floor: int = 5
cluster_selection_method: str = 'leaf'
cluster_selection_epsilon: float = 0.0
target_regime_count_min: int = 4
target_regime_count_max: int = 8
regime_count_penalty: float = 0.2
```

---

### Preset 3: **Conservative (3-4 regimes)**

```python
# regime_discovery_config.py
min_cluster_size_pct: float = 0.008
min_cluster_size_floor: int = 8
cluster_selection_method: str = 'eom'
cluster_selection_epsilon: float = 0.01
target_regime_count_min: int = 3
target_regime_count_max: int = 4
regime_count_penalty: float = 0.15
```

---

## 📊 Hyperparameter Tuning Strategy

### Step 1: **Set Target Range**
```python
target_regime_count_min: int = 4
target_regime_count_max: int = 8
```

### Step 2: **Run Initial Test**
- Check number of regimes in output report
- Note silhouette score and noise ratio

### Step 3: **Adjust Based on Results**

| Result | Action |
|--------|--------|
| Too few regimes (< 4) | Lower `min_cluster_size_pct` |
| Too many regimes (> 8) | Raise `min_cluster_size_pct` |
| High noise (> 20%) | More aggressive winsorization |
| Poor separation | Enable rolling normalization |
| Unstable regimes | Longer rolling window |

### Step 4: **Fine-Tune Penalty**
- If optimizer ignores target: **Increase penalty** (0.25-0.3)
- If quality metrics suffer: **Decrease penalty** (0.1-0.15)

---

## 🧪 Testing Checklist

After changing parameters:

- [ ] Run HDBSCAN regime discovery
- [ ] Check regime count in report
- [ ] Verify noise ratio < 15%
- [ ] Check economic separation > 20%
- [ ] Validate regime durations > 20 bars
- [ ] Review silhouette score > 0.3
- [ ] Test with different timeframes (15m, 1h, 4h)

---

## 📝 Quick Reference Table

| Want | Parameter | Value |
|------|-----------|-------|
| More regimes | `min_cluster_size_pct` | ↓ (0.003) |
| Fewer regimes | `min_cluster_size_pct` | ↑ (0.01) |
| Stricter target | `regime_count_penalty` | ↑ (0.3) |
| More flexible | `regime_count_penalty` | ↓ (0.1) |
| Less noise | `winsorize_limits` | (0.05, 0.95) |
| More responsive | `rolling_window` | ↓ (30) |
| More stable | `rolling_window` | ↑ (120) |
| Balanced clusters | `cluster_selection_method` | `'leaf'` |
| Fewer clusters | `cluster_selection_method` | `'eom'` |

---

## 🚀 Quick Start

**Default config (recommended starting point)**:

```python
from src.training.steps.market_analysis.hdbscan_clustering.config.regime_discovery_config import RegimeDiscoveryConfig

config = RegimeDiscoveryConfig(
    # Core HDBSCAN
    min_cluster_size_pct=0.005,
    min_cluster_size_floor=5,
    cluster_selection_method_options=['leaf', 'eom'],
    cluster_selection_epsilon=0.0,
    metric='manhattan',
    
    # Regime count target
    target_regime_count_min=4,
    target_regime_count_max=8,
    regime_count_penalty=0.2,
    
    # Preprocessing
    winsorize_limits=(0.02, 0.98),
    scaling_method='robust',
    enable_quantile_transformation=True,
    enable_rolling_normalization=True,
    rolling_normalization_window=60
)
```

**Or use the default config directly** (already optimized):

```python
from src.training.steps.market_analysis.hdbscan_clustering.optimization.optimized_hdbscan_regime_discovery import (
    OptimizedHDBSCANRegimeDiscovery,
    OptimizedHDBSCANRegimeDiscoveryConfig
)

# Use default config (already has all optimizations)
discovery = OptimizedHDBSCANRegimeDiscovery()

# Or customize
config = OptimizedHDBSCANRegimeDiscoveryConfig(
    execution_mode="light",
    target_regime_count_min=5,
    target_regime_count_max=7,
    regime_count_penalty=0.25
)
discovery = OptimizedHDBSCANRegimeDiscovery(config)
```

---

**Last Updated**: 2025-10-28  
**Version**: 1.0.0
