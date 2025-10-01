# Clustering Quality Investigation Report

## Problem Statement
The **hybrid NAS-TAS combined output** (final merged regime clustering) produces highly unbalanced regime distribution:
- **regime_0**: 87.7% (842 samples) - Far too large
- **regime_2**: 10.3% (99 samples) - Valid
- **regime_5**: 2.0% (19 samples) - Too small

**Note:** The "📈 OUTPUT METRICS" section in terminal output shows the **final hybrid combined output** after merging both NAS and TAS predictions. Individual NAS and TAS distributions are shown separately as "📊 NAS Regime Distribution" and "📊 TAS Regime Distribution" earlier in the terminal output.

**Target**: Each regime should be 3-25% of total samples for balanced trading strategies.

## Root Causes Identified

### 1. **Upstream Prediction Imbalance**
The imbalance likely originates from the **input predictions** (NAS or TAS), not the optimizer:

**Evidence:**
- TAS reports **0 regimes** in terminal output (fixed by standardizing keys)
- NAS reports **6 regimes** but distribution unknown
- The optimizer combines these predictions using simple consensus/weighted methods
- If NAS predictions are already imbalanced (e.g., 80% in one regime), the optimizer will preserve this

**Key Code Location:**
```python
# File: multi_objective_optimizer.py, lines 286-297
def _create_consensus_solution(self, nas_predictions, tas_predictions):
    consensus = np.zeros_like(nas_predictions)
    for i in range(len(nas_predictions)):
        if nas_predictions[i] == tas_predictions[i]:
            consensus[i] = nas_predictions[i]
        else:
            consensus[i] = (nas_predictions[i] + tas_predictions[i]) % 10
    return consensus
```

### 2. **No Feature Scaling in Optimization**
The multi-objective optimizer works with **regime labels** (not features), so it cannot fix upstream feature scaling issues.

**Key Finding:**
- NAS and TAS detectors perform their own clustering with their own features
- The optimizer only **combines the resulting labels**
- Feature quality issues in NAS/TAS cannot be fixed at optimization stage

### 3. **Missing Regime Size Constraints During Clustering**
Neither NAS nor TAS enforce regime size constraints during initial clustering:

**What's Missing:**
- No min/max cluster size enforcement in NAS clustering
- No balanced clustering algorithms (e.g., constrained K-means)
- No filtering of too-small or too-large regimes during initial detection

### 4. **Optimization Weights Configuration**
Current configuration (lines 29-40 in multi_objective_optimizer.py):
```python
max_cluster_distribution: float = 0.25  # 25% max
min_cluster_distribution: float = 0.03  # 3% min

statistical_weight: float = 0.25
economic_weight: float = 0.30
temporal_weight: float = 0.20
cv_optimization_weight: float = 0.25
```

**Issue:** These constraints are evaluated but not strictly enforced during optimization.

## Recommended Fixes (Priority Order)

### **HIGH PRIORITY - Fix Upstream Predictions**

#### Fix 5A: Add Regime Distribution Logging
**Where:** `hybrid_orchestrator.py` after NAS/TAS detection
**Why:** Identify which system is producing imbalanced predictions

```python
# After NAS detection
nas_unique, nas_counts = np.unique(nas_predictions, return_counts=True)
nas_distribution = {regime: (count/len(nas_predictions)*100) 
                   for regime, count in zip(nas_unique, nas_counts)}
tprint(f"[yellow]📊 NAS Regime Distribution: {nas_distribution}[/yellow]")

# After TAS detection  
tas_unique, tas_counts = np.unique(tas_predictions, return_counts=True)
tas_distribution = {regime: (count/len(tas_predictions)*100) 
                   for regime, count in zip(tas_unique, tas_counts)}
tprint(f"[yellow]📊 TAS Regime Distribution: {tas_distribution}[/yellow]")
```

#### Fix 5B: Add Feature Scaling Verification
**Where:** NAS and TAS feature extraction
**What:** Verify features are properly scaled (mean≈0, std≈1 for numerical stability)

#### Fix 5C: Add Regime Size Constraints to NAS/TAS
**Where:** NAS and TAS clustering algorithms
**What:** 
- Filter out regimes < 3% after clustering
- Merge/reassign samples from oversized regimes (>25%)
- Use constrained clustering algorithms

### **MEDIUM PRIORITY - Improve Optimization**

#### Fix 5D: Strengthen Distribution Constraints
**Where:** `multi_objective_optimizer.py`
**What:** Add hard constraints that reject solutions violating min/max distribution

```python
def _evaluate_distribution_constraints(self, solution: np.ndarray) -> float:
    """Evaluate how well solution meets distribution constraints."""
    unique, counts = np.unique(solution, return_counts=True)
    distributions = counts / len(solution)
    
    # Calculate penalty for constraint violations
    penalty = 0.0
    for dist in distributions:
        if dist > self.config.max_cluster_distribution:
            penalty += (dist - self.config.max_cluster_distribution) * 10.0
        if dist < self.config.min_cluster_distribution:
            penalty += (self.config.min_cluster_distribution - dist) * 10.0
    
    return -penalty  # Negative penalty as objective
```

#### Fix 5E: Add Rebalancing Step
**Where:** After optimization, before validation
**What:** Forcibly rebalance regimes that violate constraints

```python
def _rebalance_regimes(self, predictions: np.ndarray, 
                      features: np.ndarray) -> np.ndarray:
    """Rebalance regimes to meet distribution constraints."""
    unique, counts = np.unique(predictions, return_counts=True)
    distributions = counts / len(predictions)
    
    for regime, dist in zip(unique, distributions):
        if dist > self.config.max_cluster_distribution:
            # Reassign excess samples to nearest alternative regime
            self._reassign_excess_samples(predictions, regime, features)
    
    return predictions
```

### **LOW PRIORITY - Monitoring & Alerts**

#### Fix 5F: Add Clustering Quality Alerts
**Where:** Validation step
**What:** Alert when regime distribution is problematic

```python
if regime_percentage > 25.0:
    tprint(f"[bold red]⚠️🚨 ALERT: Regime {regime} is {regime_percentage:.1f}% (>25% threshold)[/bold red]")
elif regime_percentage < 3.0:
    tprint(f"[bold yellow]⚠️ WARNING: Regime {regime} is {regime_percentage:.1f}% (<3% threshold)[/bold yellow]")
```

## Next Steps

1. **IMMEDIATE:** Run with Fix 5A to identify which system (NAS/TAS) produces imbalanced predictions
   - Look for "📊 NAS Regime Distribution" in terminal output (shows individual NAS predictions)
   - Look for "📊 TAS Regime Distribution" in terminal output (shows individual TAS predictions)
   - Compare these with "📈 OUTPUT METRICS" (shows final hybrid combined output)
2. **INVESTIGATE:** Check feature quality and scaling in the problematic system
3. **IMPLEMENT:** Add regime size constraints to NAS/TAS clustering
4. **VALIDATE:** Re-run and verify distribution is within 3-25% per regime

## Key Files to Modify

1. `hybrid_orchestrator.py` - Add distribution logging
2. `nas_regime/core/enhanced_perfect_nas_regime_detector.py` - Add NAS constraints
3. `tas_regime/core/tas_regime_detector.py` - Add TAS constraints  
4. `multi_objective_optimizer.py` - Strengthen distribution constraints
5. Feature extraction in both NAS and TAS - Verify scaling

## Expected Outcomes

After fixes:
- Each regime: 3-25% of samples
- Balanced trading opportunities across regimes
- Better regime discovery quality
- More reliable NAS-TAS agreement metrics

---

## Investigation Results (2025-10-01)

### ✅ Fix 5A Implemented - Distribution Logging
**Status**: Successfully implemented and running

**Results**:
- **TAS Distribution**: Well-balanced ✓
  - All 7 regimes within acceptable range (8.7% - 24.4%)
  - Largest regime (0) at 24.4% (just under 25% threshold)
  
- **NAS Distribution**: **PROBLEM IDENTIFIED** ⚠️
  - Regime 5: **35.5%** (exceeds 25% threshold by 10.5%)
  - Regime 3: 22.9% (acceptable)
  - Regimes 0,1,2,4: 9-12% (well-balanced)

### 🔍 Root Cause Analysis

**File**: `src/training/steps/market_analysis/nas_regime/core/enhanced_perfect_nas_regime_detector.py`

**Finding**: 
1. **Line 1653-1654**: Min constraint exists (5% or 48 samples)
   ```python
   min_regime_size = max(int(0.05 * len(regime_predictions)), 48)
   regime_predictions = self._merge_small_regimes(regime_predictions, features, min_regime_size)
   ```

2. **Lines 1681-1769**: `_merge_small_regimes()` successfully merges regimes < 5%
   - Merges small regimes into nearest large regime based on centroid distance
   - Re-maps regime IDs to be sequential

3. **MISSING**: No equivalent `_split_large_regimes()` function for regimes > 25%
   - No maximum size constraint enforcement
   - Allows regimes like Regime 5 to capture 35.5% of samples

**Why Regime 5 is Oversized**:
- K-means clustering naturally produced a large cluster
- Small regimes get merged INTO large regimes (making them larger)
- No post-processing splits oversized regimes

### 🛠️ Proposed Fix: Add Maximum Regime Size Constraint

**Status**: **NOT IMPLEMENTING** - User decision to keep large regimes intact

**Rationale for Not Splitting**:
- Large regimes may represent genuinely dominant market conditions
- Splitting could create artificial regime boundaries
- The 35.5% regime might capture a persistent market state (e.g., sustained trend)
- Better to let the multi-objective optimizer handle the imbalance

**Alternative Approach**:
- Keep current implementation with min constraint only
- Document that regime sizes are naturally determined by market data
- Use alerts to notify when regimes exceed 25% threshold
- Let downstream components (optimizer, economic evaluator) handle imbalance

### 📊 Current Status

**Distribution**:
```
NAS: {0: 9.1%, 1: 10.0%, 2: 11.7%, 3: 22.9%, 4: 10.9%, 5: 35.5%}  ⚠️ Regime 5 at 35.5%
TAS: {0: 24.4%, 1: 13.2%, 2: 13.2%, 3: 9.3%, 4: 20.0%, 5: 8.7%, 6: 11.1%}  ✓ All valid
```

**Decision**: Accept NAS Regime 5 at 35.5% as-is
- Alerts are working correctly to flag the imbalance
- No automatic splitting - let market data dictate regime sizes
- Multi-objective optimizer will handle regime combination downstream

### 🎯 Next Steps

1. **Monitor** regime distributions over different market periods
2. **Analyze** what market conditions Regime 5 represents (likely trending markets)
3. **Evaluate** if the 25% threshold should be adjusted (e.g., to 35-40%)
4. **Consider** alternative clustering algorithms (e.g., DBSCAN, Gaussian Mixture)

---

## 🔧 **NAS Parameter Issues Found**

### **Issue 1: Fixed Random State**
**Location**: Line 1649 in `enhanced_perfect_nas_regime_detector.py`
```python
kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
```

**Problem**: 
- `random_state=42` makes clustering deterministic
- Same initialization every time → biased cluster shapes
- No true randomization despite `n_init=10`

**Fix**: Remove `random_state=42` or use `random_state=None`

**What Removing `random_state=42` Would Change**:
- **Current**: Same cluster initialization every run → deterministic, potentially biased results
- **After**: True randomization → different cluster shapes each run → more balanced distributions
- **Impact**: K-means will explore different cluster arrangements, reducing bias toward certain regime sizes
- **Trade-off**: Results become non-deterministic (different each run) but more balanced

### **Issue 2: Insufficient Regime Count**
**Current Calculation**: `n_regimes = min(8, max(3, features.shape[0] // 50))`
- **With 1921 samples**: `n_regimes = min(8, max(3, 38)) = 8`
- **Ideal samples per regime**: 240
- **25% threshold**: 480 samples (2x ideal)
- **35.5% actual**: 681 samples (2.8x ideal)

**Problem**: 8 regimes too few for 1921 samples
**Fix**: Increase regime count with reasonable maximum

### **Issue 3: Regime Merging Amplifies Imbalance**
1. K-means creates 8 clusters with uneven sizes
2. Small clusters (< 5%) merge into nearest large cluster  
3. Large clusters become even larger
4. No maximum size constraint

**Result**: Regime 5 grows from ~240 to 681 samples (35.5%)

### **Proposed Parameter Fixes**

1. **Remove Fixed Random State**:
   ```python
   kmeans = KMeans(n_clusters=n_regimes, n_init=10)  # Remove random_state=42
   ```

2. **Increase Regime Count** (target 8-12 regimes):
   ```python
   n_regimes = min(12, max(8, features.shape[0] // 192))  # Target 8-12 regimes for better balance
   ```
   
   **Impact**:
   - **With 1921 samples**: `n_regimes = min(12, max(8, 10)) = min(12, 10) = 10`
   - **Ideal samples per regime**: 1921 ÷ 10 = 192 samples
   - **25% threshold**: 480 samples (2.5x ideal) - much more reasonable than 2.0x
   - **Expected result**: More balanced regime distribution with reasonable regime count

3. **Add Maximum Size Constraint** (if desired):
   ```python
   max_regime_size = int(0.25 * len(regime_predictions))  # 25% max
   regime_predictions = self._split_large_regimes(regime_predictions, features, max_regime_size)
   ```

