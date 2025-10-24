# ML Model Trainer - YAML Configuration Integration

## Overview

The ML Model Trainer has been updated to work seamlessly with YAML configuration files that follow the specified format, with support for mode-based parameter reduction and proper integration with ares_launcher.

## ✅ **Key Changes Implemented**

### 1. **Multi-Timeframe Disabled** ✅
- **Change**: Commented out multi-timeframe processing as per requirements
- **Impact**: Simplified feature processing, timeframe handled by ares_launcher

### 2. **Timeframe from ares_launcher** ✅
- **Change**: Timeframe is now captured from ares_launcher instead of being hardcoded
- **Impact**: Better integration with the broader system

### 3. **Mode-Based Parameter Reduction** ✅
- **LIGHT Mode**: 90% reduction in n_estimators, early_stopping_rounds, cv_folds, n_trials
- **BLANK Mode**: 50% reduction in n_estimators, early_stopping_rounds, cv_folds, n_trials
- **FULL Mode**: No reduction (default)

```python
def _get_hpo_params(self, trial, model_config: Dict[str, Any], mode: str = "FULL") -> Dict[str, Any]:
    if mode == "LIGHT":
        reduction_factor = 0.1  # 90% reduction
    elif mode == "BLANK":
        reduction_factor = 0.5  # 50% reduction
    else:
        reduction_factor = 1.0  # No reduction
```

### 4. **Task Type from YAML** ✅
- **Source**: `targets.target_type` in YAML config
- **Mapping**: 
  - `"binary_classification"` → `"classification"`
  - `"multiclass_classification"` → `"classification"`
  - `"regression"` → `"regression"`

```python
def _infer_task_type_from_recipe(self, recipe: Dict[str, Any]) -> str:
    tt = (recipe.get("targets", {}).get("target_type") or "").lower()
    if tt in {"binary_classification", "multiclass_classification"}:
        return "classification"
    if tt in {"regression"}:
        return "regression"
    return "classification"  # safe default
```

### 5. **Metrics from YAML** ✅
- **Source**: `metrics.primary` and `metrics.secondary` in YAML
- **Mapping**: YAML metric names to sklearn scorers

```python
SCORER_MAP = {
    "f1_score": "f1",
    "precision": "precision", 
    "recall": "recall",
    "accuracy": "accuracy",
    "auc_roc": "roc_auc",
    "mse": "neg_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2_score": "r2",
}
```

### 6. **CV Strategy from YAML** ✅
- **Source**: `training.cv_strategy` and `training.cv_params` in YAML
- **Supported**: TimeSeriesSplit, PurgedCV, WalkForwardCV

```python
def _make_cv(self, recipe: Dict[str, Any]):
    name = (recipe.get("training", {}).get("cv_strategy") or "TimeSeriesSplit").lower()
    params = recipe.get("training", {}).get("cv_params", {})
    if name == "timeseriessplit":
        return TimeSeriesSplit(n_splits=params.get("n_splits", 5))
```

### 7. **Early Stopping with eval_set** ✅
- **Implementation**: Proper early stopping using temporal validation split
- **Support**: LightGBM, XGBoost, CatBoost with eval_set

```python
def _fit_with_early_stopping(self, model, X: np.ndarray, y: np.ndarray, recipe: Dict[str, Any], task_type: str):
    es = recipe.get("training", {}).get("early_stopping", {}).get("enabled", False)
    if not es:
        return model.fit(X, y)
    
    # Make temporal validation split
    tss = TimeSeriesSplit(n_splits=3)
    train_idx, val_idx = list(tss.split(X))[-1]
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    # Apply eval_set based on model type
    if "LGBM" in type(model).__name__.upper():
        return model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
```

### 8. **Special Model Types** ✅
- **LIGHTGBM_PATCHTST**: Treated as LightGBM with PatchTST feature gating
- **STACKER_LGBM_CALIBRATED**: Treated as LightGBM with stacking support

```python
if model_key in {"LIGHTGBM", "LIGHTGBM_PATCHTST", "STACKER_LGBM_CALIBRATED"}:
    from lightgbm import LGBMClassifier, LGBMRegressor
    cls = LGBMClassifier if is_classification else LGBMRegressor
    return cls(**merged, random_state=42, verbose=-1, n_jobs=1)
```

### 9. **Feature Gating for PatchTST** ✅
- **Check**: Verifies PatchTST features are enabled before using LIGHTGBM_PATCHTST
- **Validation**: Ensures `enable_patchtst_features` and `patchtst.enabled` are both true

```python
if model_config.get("type", "").upper() == "LIGHTGBM_PATCHTST":
    af = config.get("inputs", {}).get("analyst_features", {})
    fe = config.get("feature_engineering", {}).get("patchtst", {})
    if not (af.get("enable_patchtst_features") and fe.get("enabled")):
        raise ValueError("LIGHTGBM_PATCHTST selected but PatchTST features are disabled in config.")
```

### 10. **Probability Matrix Shape** ✅
- **Return**: Full probability matrix `(n, n_classes)` without slicing
- **Usage**: Callers can slice as needed for binary vs multi-class

```python
if hasattr(model, 'predict_proba'):
    return model.predict_proba(X)  # (n, n_classes) - full matrix
```

## **YAML Configuration Format Support**

### **Expected YAML Structure**
```yaml
# Task type definition
targets:
  target_type: "binary_classification"  # or "multiclass_classification", "regression"

# Metrics configuration
metrics:
  primary: "f1_score"  # Maps to sklearn "f1"
  secondary: "auc_roc"  # Maps to sklearn "roc_auc"

# Training configuration
training:
  cv_strategy: "TimeSeriesSplit"
  cv_params:
    n_splits: 5
    test_size: 0.2
  early_stopping:
    enabled: true
    patience: 50
  hyperparameter_optimization:
    direction: "maximize"
    metric: "f1_score"

# Model configuration
models:
  - type: "LIGHTGBM_PATCHTST"
    parameters:
      n_estimators: 1000
      learning_rate: 0.1
      early_stopping_rounds: 50

# Feature configuration
inputs:
  analyst_features:
    enable_patchtst_features: true

feature_engineering:
  patchtst:
    enabled: true
```

## **Mode-Based Parameter Reduction**

### **LIGHT Mode (90% Reduction)**
- `n_estimators`: 1000 → 100
- `cv_folds`: 5 → 2 (minimum)
- `max_trials`: 100 → 10
- `timeout`: 3600s → 360s

### **BLANK Mode (50% Reduction)**
- `n_estimators`: 1000 → 500
- `cv_folds`: 5 → 3 (minimum)
- `max_trials`: 100 → 50
- `timeout`: 3600s → 1800s

### **FULL Mode (No Reduction)**
- All parameters at full values
- Maximum performance and accuracy

## **Integration Benefits**

### **1. YAML-Driven Configuration**
- All training parameters controlled via YAML
- Easy experimentation and parameter tuning
- Consistent configuration across environments

### **2. Mode-Based Optimization**
- Quick testing with LIGHT mode
- Balanced performance with BLANK mode
- Full accuracy with FULL mode

### **3. Proper Early Stopping**
- Prevents overfitting with eval_set
- Works with all supported model types
- Configurable via YAML

### **4. Feature Gating**
- Prevents errors with missing features
- Clear validation messages
- Supports complex feature dependencies

### **5. Flexible Metrics**
- YAML-defined primary/secondary metrics
- Automatic sklearn scorer mapping
- Support for both classification and regression

## **Usage Examples**

### **Basic Usage with YAML**
```python
from src.training.steps.models_training.training.ml_model_trainer import MLModelTrainer, MLModelTrainerConfig, ModelType

# Create configuration
config = MLModelTrainerConfig(
    model_types=[ModelType.ANALYST_BASE, ModelType.TACTICIAN_BASE],
    mode="LIGHT",  # Use LIGHT mode for quick testing
    timeframe="15m"  # Set by ares_launcher
)

# Create trainer
trainer = MLModelTrainer(config)

# Train models with YAML configs
results = await trainer.train_models(data, config_paths)
```

### **YAML Configuration Example**
```yaml
# analyst_base_config.yaml
targets:
  target_type: "binary_classification"

metrics:
  primary: "f1_score"
  secondary: "auc_roc"

training:
  cv_strategy: "TimeSeriesSplit"
  cv_params:
    n_splits: 5
  early_stopping:
    enabled: true
    patience: 50

models:
  - type: "LIGHTGBM"
    parameters:
      n_estimators: 1000
      learning_rate: 0.1
      early_stopping_rounds: 50
```

The ML Model Trainer now fully supports YAML-based configuration with mode-based parameter reduction, proper early stopping, and seamless integration with ares_launcher.