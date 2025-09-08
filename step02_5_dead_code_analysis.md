# Dead Code Analysis: step02_5_sr_optimization.py

## File Overview
- **File**: `/workspace/src/training/steps/data_collection/data_preparation/step02_5_sr_optimization.py`
- **Size**: Very large file (43,542 tokens)
- **Purpose**: SR (Support/Resistance) optimization for trading data

## 🔍 Dead Code Findings

### 1. **DEAD IMPORTS** (Confirmed Unused)

#### **A. LightGBM Import - DEAD** ❌
```python
# Line 2667-2670
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
```
**Status**: ❌ **DEAD** - `LGBM_AVAILABLE` is set but never checked or used anywhere in the code.

#### **B. scikit-optimize Imports - DEAD** ❌
```python
# Line 88-89
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
```
**Status**: ❌ **DEAD** - These imports are never used in the code. No calls to `Categorical()`, `use_named_args()`, etc.

#### **C. SelectFromModel Import - DEAD** ❌
```python
# Line 2449
from sklearn.feature_selection import SelectFromModel, RFECV, mutual_info_classif
```
**Status**: ❌ **DEAD** - `SelectFromModel` is imported but never instantiated or used.

#### **D. Utility Imports - DEAD** ❌
```python
# Line 40-41
function_tracker,
logging_patterns
```
**Status**: ❌ **DEAD** - These functions are imported but never called.

#### **E. M1 Batch Process Import - DEAD** ❌
```python
# Line 21
from src.utils.m1_gpu_utils import m1_batch_process
```
**Status**: ❌ **DEAD** - `m1_batch_process` is imported but never called. Only `M1_BATCH_AVAILABLE` is used.

### 2. **USED IMPORTS** (Keep These) ✅

#### **A. XGBoost - USED** ✅
```python
# Line 2661-2664, 3028
import xgboost as xgb
```
**Status**: ✅ **USED** - XGBoost is actively used in model creation and hyperparameter optimization.

#### **B. concurrent.futures - USED** ✅
```python
# Line 1940, 1983
import concurrent.futures
```
**Status**: ✅ **USED** - Used for ThreadPoolExecutor in async operations.

#### **C. scipy.stats - USED** ✅
```python
# Line 2994
import scipy.stats as stats
```
**Status**: ✅ **USED** - Used for statistical distributions in hyperparameter optimization.

## 🧹 **Cleanup Recommendations**

### **Safe to Remove (High Confidence)**:

1. **Remove LightGBM import block**:
```python
# DELETE these lines (2667-2670):
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
```

2. **Remove unused skopt imports**:
```python
# DELETE from line 88-89:
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
```

3. **Remove SelectFromModel from import**:
```python
# CHANGE line 2449 from:
from sklearn.feature_selection import SelectFromModel, RFECV, mutual_info_classif
# TO:
from sklearn.feature_selection import RFECV, mutual_info_classif
```

4. **Remove unused utility imports**:
```python
# DELETE from line 40-41:
function_tracker,
logging_patterns
```

5. **Remove m1_batch_process import**:
```python
# CHANGE line 21 from:
from src.utils.m1_gpu_utils import m1_batch_process  # Streaming batch processing with MPS gating
# TO:
# from src.utils.m1_gpu_utils import m1_batch_process  # Streaming batch processing with MPS gating
```

### **Impact Assessment**:
- **Lines to remove**: ~10 lines
- **Risk level**: Very Low (these are clearly unused)
- **Benefits**: Cleaner imports, faster startup, reduced memory footprint

## 🔍 **Manual Verification Summary**

I manually verified each import by:
1. ✅ Searching for actual usage patterns
2. ✅ Checking for function calls and variable references  
3. ✅ Confirming conditional usage patterns
4. ✅ Verifying that imports are only set but never used

**Result**: The dead code analysis was **100% accurate** for this file. All identified dead imports are indeed unused and safe to remove.