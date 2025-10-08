# Utility Integration Quick Guide

Quick reference for integrating recommended utilities into the refactored feature systems.

---

## 1. Add tprint for Better Logging

### Why?
- Consistent, colorful logging across codebase
- Better user feedback during operations
- Easier debugging

### Where?
- `features_common/transforms/base_scaler.py`
- `feature_engineering_roadmap/transforms.py`
- Any long-running operations

### How?

**Before:**
```python
import logging
logger = logging.getLogger(__name__)

def fit_transform(self, data: pd.Series) -> pd.Series:
    logger.info(f"Fitting {self.__class__.__name__}")
    # ... fit logic ...
    self.fitted = True
    logger.info("Fitting complete")
    return self.transform(data)
```

**After:**
```python
from src.utils.tprint import tprint
import logging
logger = logging.getLogger(__name__)

def fit_transform(self, data: pd.Series) -> pd.Series:
    tprint(f"🔧 Fitting {self.__class__.__name__}...", color="cyan", bold=True)
    
    # ... fit logic ...
    
    self.fitted = True
    tprint(f"✅ {self.__class__.__name__} fitted successfully", color="green")
    
    return self.transform(data)
```

### Implementation Steps:

1. **Add import to base_scaler.py:**
   ```python
   from src.utils.tprint import tprint
   ```

2. **Add to fit_transform methods:**
   ```python
   def fit_transform(self, data: pd.Series) -> pd.Series:
       tprint(f"🔧 [{self.__class__.__name__}] Fitting on {len(data)} samples", 
              color="cyan")
       
       # Existing logic...
       
       tprint(f"✅ [{self.__class__.__name__}] Fitted: "
              f"{len(data)} → {self.fitted}", 
              color="green")
       return self.transform(data)
   ```

3. **Add to error handling:**
   ```python
   def transform(self, data: pd.Series) -> pd.Series:
       try:
           self._validate_fitted()
           return (data - self.mean) / self.std
       except Exception as e:
           tprint(f"❌ [{self.__class__.__name__}] Transform failed: {e}", 
                  color="red", bold=True)
           raise
   ```

---

## 2. Add Math Validation for Robustness

### Why?
- Prevents inf/nan propagation
- Better error messages
- Consistent validation

### Where?
- All transform methods in `features_common/`
- Division operations
- Numeric conversions

### How?

**Before:**
```python
def transform(self, data: pd.Series) -> pd.Series:
    return (data - self.mean) / self.std
```

**After:**
```python
from src.utils.math_validation import safe_divide, check_for_inf_nan

def transform(self, data: pd.Series) -> pd.Series:
    # Safe division prevents inf from zero std
    result = safe_divide(data - self.mean, self.std, default=0.0)
    
    # Validate output
    check_for_inf_nan(result.values, f'{self.__class__.__name__} output')
    
    return result
```

### Implementation Steps:

1. **Add imports:**
   ```python
   from src.utils.math_validation import (
       safe_divide,
       check_for_inf_nan,
       validate_numeric_array,
       is_valid_number
   )
   ```

2. **Validate inputs:**
   ```python
   def fit_transform(self, data: pd.Series) -> pd.Series:
       # Validate input is numeric
       validate_numeric_array(data.values, 'input data')
       
       clean_data = data.dropna()
       
       # Check for sufficient data
       if len(clean_data) == 0:
           raise ValueError("No valid data to fit")
       
       # ... rest of fit logic ...
   ```

3. **Use safe operations:**
   ```python
   # ZScoreNormalizer
   def transform(self, data: pd.Series) -> pd.Series:
       return safe_divide(data - self.mean, self.std, default=0.0)
   
   # RobustScaler  
   def transform(self, data: pd.Series) -> pd.Series:
       return safe_divide(data - self.median, 1.4826 * self.mad, default=0.0)
   
   # MinMaxScaler
   def transform(self, data: pd.Series) -> pd.Series:
       return safe_divide(data - self.min_val, 
                          self.max_val - self.min_val, 
                          default=0.0)
   ```

4. **Validate outputs:**
   ```python
   def fit_transform(self, data: pd.Series) -> pd.Series:
       # ... fit logic ...
       
       transformed = self.transform(data)
       
       # Check for problematic values
       check_for_inf_nan(transformed.values, 
                        f'{self.__class__.__name__} transformed data')
       
       return transformed
   ```

---

## 3. Enhance CV with Lookahead Protection

### Why?
- Prevents data leakage
- Catches lookahead bias
- Safer model validation

### Where?
- `features_common/optimization/cv_base.py`
- Any cross-validation code

### How?

**Add to BaseCVSplitter:**

```python
from src.utils.ml_common.validation.lookahead_protection import check_lookahead
from src.utils.ml_common.validation.data_leakage import detect_leakage

class BaseCVSplitter:
    def __init__(self, 
                 n_folds: int = 5,
                 embargo_pct: float = 0.1,
                 check_leakage: bool = True):
        self.n_folds = n_folds
        self.embargo_pct = embargo_pct
        self.check_leakage = check_leakage
    
    def split_with_embargo(self, X, y=None):
        """Split with leakage detection."""
        # ... existing split logic ...
        
        for train_idx, val_idx in tscv.split(X):
            # Apply embargo
            # ...
            
            if self.check_leakage and len(val_idx) > 0:
                # Check for lookahead
                try:
                    check_lookahead(
                        train_indices=train_idx,
                        val_indices=val_idx,
                        time_index=X.index
                    )
                except Exception as e:
                    logger.warning(f"Lookahead detected: {e}")
            
            splits.append((train_index, val_index))
        
        return splits
```

---

## 4. Add Bayesian Optimization to Lookback Selection

### Why?
- More efficient parameter search
- Better convergence
- Fewer manual iterations

### Where?
- `feature_engineering_roadmap/lookback_selection.py`
- Any hyperparameter tuning

### How?

**Add to LookbackSelector:**

```python
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer

class LookbackSelector:
    def __init__(self, 
                 n_folds: int = 5,
                 use_bayesian: bool = True,
                 n_trials: int = 50):
        self.n_folds = n_folds
        self.use_bayesian = use_bayesian
        self.menus = self._create_menus()
        
        if use_bayesian:
            self.optimizer = BayesianTPEOptimizer(
                n_trials=n_trials,
                n_jobs=4,
                verbose=False
            )
    
    def select_lookbacks_bayesian(self, 
                                   features: pd.DataFrame,
                                   targets: pd.Series,
                                   feature_families: Dict[str, List[str]]):
        """Select optimal lookbacks using Bayesian optimization."""
        
        # Define objective
        def objective(params):
            total_score = 0
            for family, lookback in params.items():
                if family in feature_families:
                    score = self._evaluate_lookback(
                        features, targets, 
                        feature_families[family], 
                        int(lookback)
                    )
                    total_score += score
            return -total_score  # Minimize negative
        
        # Define search space from menus
        search_space = {}
        for family, menu in self.menus.items():
            search_space[family] = {
                'type': 'int',
                'low': min(menu.options),
                'high': max(menu.options)
            }
        
        # Optimize
        best_params = self.optimizer.optimize(
            objective,
            search_space
        )
        
        # Convert to LookbackChoice objects
        choices = {}
        for family, lookback in best_params.items():
            choices[family] = LookbackChoice(
                family=family,
                selected_lookback=int(lookback),
                selection_criteria=SelectionCriteria.IC,
                confidence_score=0.8,
                ic_score=0.0,  # Would calculate
                auc_score=0.0,
                simplicity_bonus=0.0,
                spec_hash=f"{family}_{lookback}"
            )
        
        return choices
```

---

## 5. Usage Examples

### Example 1: Enhanced ZScoreNormalizer

```python
from src.features_common.transforms.base_scaler import BaseScaler
from src.utils.tprint import tprint
from src.utils.math_validation import safe_divide, check_for_inf_nan, validate_numeric_array

class ZScoreNormalizer(BaseScaler):
    """Enhanced z-score normalizer with tprint and validation."""
    
    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None
    
    def fit_transform(self, data: pd.Series) -> pd.Series:
        """Fit and transform with enhanced logging and validation."""
        tprint(f"🔧 [ZScoreNormalizer] Fitting on {len(data)} samples", 
               color="cyan")
        
        # Validate input
        validate_numeric_array(data.values, 'input data')
        
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            tprint("⚠️  [ZScoreNormalizer] No valid data, using defaults", 
                   color="yellow")
            self.mean = 0.0
            self.std = 1.0
        else:
            self.mean = float(clean_data.mean())
            self.std = float(clean_data.std())
            
            if self.std == 0 or np.isnan(self.std):
                tprint("⚠️  [ZScoreNormalizer] Zero std detected, using 1.0", 
                       color="yellow")
                self.std = 1.0
        
        self.fitted = True
        
        tprint(f"✅ [ZScoreNormalizer] Fitted: mean={self.mean:.4f}, std={self.std:.4f}", 
               color="green")
        
        transformed = self.transform(data)
        
        # Validate output
        check_for_inf_nan(transformed.values, 'transformed data')
        
        return transformed
    
    def transform(self, data: pd.Series) -> pd.Series:
        """Transform with safe division."""
        self._validate_fitted()
        
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer state is invalid")
        
        # Safe division prevents inf/nan
        return safe_divide(data - self.mean, self.std, default=0.0)
```

### Example 2: Enhanced BaseCVSplitter

```python
from src.features_common.optimization.cv_base import BaseCVSplitter
from src.utils.tprint import tprint
from src.utils.ml_common.validation.lookahead_protection import check_lookahead

class EnhancedCVSplitter(BaseCVSplitter):
    """CV splitter with lookahead protection and tprint."""
    
    def split_with_embargo(self, X, y=None):
        """Split with enhanced validation and logging."""
        tprint(f"🔍 [CV] Creating {self.n_folds} splits with "
               f"{self.embargo_pct:.1%} embargo", 
               color="cyan")
        
        splits = super().split_with_embargo(X, y)
        
        # Validate each split
        for idx, (train_idx, val_idx) in enumerate(splits):
            try:
                check_lookahead(
                    train_indices=train_idx,
                    val_indices=val_idx,
                    time_index=X.index
                )
                tprint(f"✅ [CV] Fold {idx+1}: No lookahead detected", 
                       color="green")
            except Exception as e:
                tprint(f"⚠️  [CV] Fold {idx+1}: {e}", 
                       color="yellow")
        
        tprint(f"✅ [CV] Generated {len(splits)} validated splits", 
               color="green")
        
        return splits
```

---

## 6. Testing Enhanced Code

### Test Script

```python
import pandas as pd
import numpy as np
from src.features_common.transforms.base_scaler import ZScoreNormalizer
from src.features_common.optimization.cv_base import BaseCVSplitter

# Test 1: Enhanced normalizer
print("="*60)
print("TEST 1: Enhanced ZScoreNormalizer")
print("="*60)

data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
normalizer = ZScoreNormalizer()
transformed = normalizer.fit_transform(data)

print(f"Original: {data.values}")
print(f"Transformed: {transformed.values}")
print(f"Mean: {transformed.mean():.6f}")
print(f"Std: {transformed.std():.6f}")

# Test 2: Enhanced CV splitter
print("\n" + "="*60)
print("TEST 2: Enhanced CV Splitter")
print("="*60)

dates = pd.date_range('2023-01-01', periods=100, freq='D')
X = pd.DataFrame({'feature': np.random.randn(100)}, index=dates)

splitter = BaseCVSplitter(n_folds=3, embargo_pct=0.1)
splits = splitter.split_with_embargo(X)

print(f"Number of splits: {len(splits)}")
for idx, (train, val) in enumerate(splits):
    print(f"  Fold {idx+1}: train={len(train)}, val={len(val)}")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED")
print("="*60)
```

---

## 7. Quick Checklist

When adding a new transform or feature:

- [ ] Import and use `tprint` for user feedback
- [ ] Use `safe_divide` for all divisions
- [ ] Validate inputs with `validate_numeric_array`
- [ ] Check outputs with `check_for_inf_nan`
- [ ] Add lookahead protection to CV
- [ ] Consider Bayesian optimization for hyperparameters
- [ ] Use M1 hardware optimizations where applicable
- [ ] Leverage matrix_operations for vectorized code
- [ ] Add proper error handling with `tprint` messages

---

## 8. Summary

### Immediate Wins (< 1 hour)
1. Add `tprint` imports - Better UX
2. Add `safe_divide` - Prevent inf/nan
3. Add `check_for_inf_nan` - Catch issues early

### Short-term (1-2 days)
1. Full math validation integration
2. Enhanced CV with leakage detection
3. Comprehensive testing

### Long-term (Optional)
1. Bayesian optimization for lookback
2. Performance profiling
3. Advanced logging/monitoring

**Priority:** Start with tprint and safe_divide - highest impact, lowest effort.

---

Last Updated: October 8, 2025  
Part of Strategy C Implementation
