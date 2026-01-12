# Huber Regressor Integration Plan for Layer 2 ML Models

This plan implements the integration of Huber Regressor for detection of monotonic constraints, automated interaction constraints, warm start initialization, and feature pruning to replace the current Ridge/LASSO approach in label_based_layer_2.py.

## Current State Analysis

The existing `huber_regressor_for_trees.py` provides advanced features but is **NOT INTEGRATED** into the ML models. The current implementation uses:
- Basic LASSO feature pruning (alpha=0.001)
- Ridge-based monotonic constraints (threshold 0.01)  
- No interaction constraints
- No warm start initialization
- Default model parameters that don't match user specifications

## Implementation Strategy

### Phase 1: Import and Integration Setup
1. **Add Huber Import** - Import `prepare_huber_teacher_outputs` from `huber_regressor_for_trees`
2. **Replace Feature Pruning** - Replace LASSO-based pruning with Huber coefficient analysis
3. **Update Constraints** - Replace Ridge constraints with Huber-based monotonic constraints
4. **Add Interaction Constraints** - Implement hierarchical clustering from Huber importance
5. **Add Warm Start** - Use Huber ensemble predictions for model initialization

### Phase 2: Model Configuration Updates
Update model configurations to match user specifications:

**XGBoost Updates:**
- `num_parallel_tree=7`
- `colsample_bynode=0.4` 
- `subsample=0.6`
- `reg_lambda=50`
- `min_child_weight=10`
- `gamma=1.1`
- `learning_rate=0.03`
- HPO ranges: `min_child_weight=[10-50]`, `gamma=[0.5-2]`, `colsample_bynode=[0.3-0.5]`, `max_depth=[4-6]`

**LightGBM Updates:**
- `path_smooth=20`
- `reg_lambda=10+` (HPO range higher)
- `extra_trees=True`
- `linear_tree=True` (for linear variants)
- `min_gain_to_split=[0.01-0.05]`
- `bagging_fraction=0.7`
- `feature_fraction=0.6`
- `lambda_l1=[0.1-5.0]`
- `max_bin=63`

**CatBoost Updates:**
- `subsample=0.6`
- `colsample_bylevel=0.5`
- `leaf_estimation_iterations=10`
- `l2_leaf_reg=20`
- `random_strength=5`
- `bootstrap_type='MVS'`

**ExtraTrees Updates:**
- Add `monotonic_cst` parameter (scikit-learn 1.4+ support)

### Phase 3: HPO Integration
1. **Median Early Stopping** - Implement Optuna median early stopping
2. **Update HPO Functions** - Modify `_optimize_xgb_params`, `_optimize_lgbm_params`, etc.
3. **Add Interaction Constraints** - Pass Huber interaction constraints to tree models
4. **Update Constants** - Modify `LAYER2_PROBE_CONSTANTS` and `LAYER2_MODEL_CONSTANTS`

## Detailed Implementation Steps

### Step 1: Update Imports and Constants
```python
# Add to imports section
from src.utils.huber_regressor_for_trees import prepare_huber_teacher_outputs

# Update model constants with user-specified parameters
HUBER_ENHANCED_CONSTANTS = {
    # XGB specific
    'num_parallel_tree': 7,
    'colsample_bynode': 0.4,
    'subsample': 0.6,
    'reg_lambda': 50,
    'min_child_weight': 10,
    'gamma': 1.1,
    'learning_rate': 0.03,
    # LGBM specific  
    'path_smooth': 20,
    'reg_lambda': 10,
    'extra_trees': True,
    'min_gain_to_split': 0.01,
    'bagging_fraction': 0.7,
    'feature_fraction': 0.6,
    'lambda_l1': 0.1,
    'max_bin': 63,
    # CatBoost specific
    'subsample': 0.6,
    'colsample_bylevel': 0.5,
    'leaf_estimation_iterations': 10,
    'l2_leaf_reg': 20,
    'random_strength': 5,
    'bootstrap_type': 'MVS'
}
```

### Step 2: Replace _run_model_race Function
Replace the current LASSO/Ridge approach with Huber-based preprocessing:

```python
def _run_model_race(self, X_train, y_train, X_val, y_val, w_train, environment_masks=None):
    """Run model race with Huber-enhanced feature selection and constraints."""
    
    # 1. Huber-based preprocessing
    huber_results = prepare_huber_teacher_outputs(
        X_train=X_train,
        y_train=y_train, 
        X_val=X_val,
        X_test=None,  # Not needed for race
        vol_proxy=X_train.get('volatility_1d', None),  # Use volatility as proxy
        epsilons=[1.1, 1.35, 1.75],
        alphas=[1e-4, 1e-3, 1e-2],
        pruning_percentile=15,
        corr_threshold=0.7,
        n_jobs=-1
    )
    
    # Extract Huber outputs
    selected_features = huber_results['selected_features']
    monotonic_constraints = huber_results['monotonic_constraints'] 
    interaction_constraints = huber_results['interaction_constraints']
    warm_start_predictions = huber_results['warm_start']
    
    # 2. Filter datasets using Huber-selected features
    X_train_final = X_train[selected_features]
    X_val_final = X_val[selected_features]
    
    # 3. Create candidates with Huber-enhanced constraints
    candidates = self._create_huber_enhanced_candidates(
        scale_pos_weight, X_train_final, y_train,
        environment_masks=environment_masks,
        monotonic_constraints=monotonic_constraints,
        interaction_constraints=interaction_constraints,
        warm_start=warm_start_predictions
    )
    
    # 4. Continue with existing race logic...
```

### Step 3: Update _create_irm_candidates Function
Modify candidate creation to use Huber outputs and user-specified parameters:

```python
def _create_huber_enhanced_candidates(self, scale_pos_weight, X_train, y_train, 
                                     environment_masks=None, monotonic_constraints=None,
                                     interaction_constraints=None, warm_start=None):
    """Create model candidates with Huber-enhanced constraints and parameters."""
    
    candidates = []
    
    # 1. LightGBM with Huber enhancements
    lgbm_params = HUBER_ENHANCED_CONSTANTS.copy()
    lgbm_params.update({
        'monotone_constraints': monotonic_constraints,
        'interaction_constraints': interaction_constraints,
        'scale_pos_weight': scale_pos_weight,
        # Warm start via init_score if available
        'init_score': warm_start.get('train') if warm_start else None
    })
    
    candidates.append({
        'name': 'LGBM_Huber_Enhanced',
        'model': lgb.LGBMClassifier(**lgbm_params)
    })
    
    # 2. XGBoost with Huber enhancements  
    xgb_constraints = tuple(monotonic_constraints.get(col, 0) for col in X_train.columns)
    xgb_params = {
        'n_estimators': 100,
        'learning_rate': 0.03,
        'num_parallel_tree': 7,
        'colsample_bynode': 0.4,
        'subsample': 0.6,
        'reg_lambda': 50,
        'min_child_weight': 10,
        'gamma': 1.1,
        'monotone_constraints': xgb_constraints,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': 1,
        'verbosity': 0,
        'use_label_encoder': False
    }
    
    candidates.append({
        'name': 'XGB_Huber_Enhanced', 
        'model': XGBClassifier(**xgb_params)
    })
    
    # 3. CatBoost with Huber enhancements
    catboost_params = {
        'iterations': 100,
        'learning_rate': 0.03,
        'depth': 5,
        'subsample': 0.6,
        'colsample_bylevel': 0.5,
        'leaf_estimation_iterations': 10,
        'l2_leaf_reg': 20,
        'random_strength': 5,
        'bootstrap_type': 'MVS',
        'monotone_constraints': monotonic_constraints,
        'scale_pos_weight': scale_pos_weight,
        'verbose': 0,
        'random_seed': 42,
        'thread_count': 1,
        'allow_writing_files': False
    }
    
    candidates.append({
        'name': 'CatBoost_Huber_Enhanced',
        'model': CatBoostClassifier(**catboost_params)
    })
    
    # 4. ExtraTrees with monotonic constraints (scikit-learn 1.4+)
    et_params = {
        'n_estimators': 1000,
        'max_features': 'log2',
        'min_samples_leaf': 0.02,
        'max_depth': None,
        'class_weight': 'balanced',
        'bootstrap': True,
        'random_state': 42,
        'n_jobs': 1,
        'monotonic_cst': tuple(monotonic_constraints.get(col, 0) for col in X_train.columns)
    }
    
    candidates.append({
        'name': 'ExtraTrees_Huber_Enhanced',
        'model': ExtraTreesClassifier(**et_params)
    })
    
    return candidates
```

### Step 4: Update HPO Functions
Modify HPO optimization functions to use new parameter ranges and median early stopping:

```python
def _optimize_xgb_params(self, X_train, y_train, X_val, y_val, w_train=None):
    """Optimize XGBoost with user-specified parameter ranges and median early stopping."""
    
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 4, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'min_child_weight': trial.suggest_int('min_child_weight', 10, 50),
            'gamma': trial.suggest_float('gamma', 0.5, 2.0),
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.3, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 10, 100, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'scale_pos_weight': self._compute_scale_pos_weight(y_train),
            'random_state': 42,
            'n_jobs': 1,
            'verbosity': 0,
            'use_label_encoder': False
        }
        
        # Add monotonic constraints if available from Huber
        if hasattr(self, '_current_monotonic_constraints'):
            params['monotone_constraints'] = self._current_monotonic_constraints
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=w_train,
                 eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=False)
        
        preds = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)
    
    # Add median early stopping
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=self.focal_hpo_n_trials, 
                   timeout=None, n_jobs=1, 
                   callbacks=[_median_early_stopping_callback])
    
    return study.best_params, study.best_value
```

### Step 5: Update Model Training Functions
Modify the main training functions in `_train_geometry_batch` to use Huber-enhanced models:

```python
# In train_single_geometry function:
if 'XGB' in best_name and XGBClassifier is not None:
    # Use Huber-enhanced parameters
    clf = XGBClassifier(
        max_depth=focal_params.get('max_depth', 5),
        learning_rate=focal_params.get('learning_rate', 0.03),
        n_estimators=focal_params.get('n_estimators', 100),
        min_child_weight=focal_params.get('min_child_weight', 10),
        gamma=focal_params.get('gamma', 1.1),
        subsample=focal_params.get('subsample', 0.6),
        colsample_bynode=focal_params.get('colsample_bynode', 0.4),
        reg_lambda=focal_params.get('reg_lambda', 50),
        reg_alpha=focal_params.get('reg_alpha', 0.0),
        monotone_constraints=getattr(self, '_current_monotonic_constraints', None),
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=1,
        verbosity=0,
        use_label_encoder=False
    )
```

## Implementation Benefits

1. **Enhanced Feature Selection** - Huber ensemble provides more robust feature pruning than LASSO
2. **Improved Monotonic Constraints** - Coefficient-based constraints are more principled than Ridge
3. **Interaction Constraints** - Hierarchical clustering captures feature relationships
4. **Warm Start Initialization** - Huber ensemble predictions provide better initialization
5. **User-Specified Parameters** - All models use the exact parameters requested
6. **Better HPO** - Median early stopping and appropriate parameter ranges

## Testing and Validation

1. **Unit Tests** - Test Huber preprocessing pipeline
2. **Integration Tests** - Verify model race with Huber enhancements
3. **Performance Tests** - Compare against current LASSO/Ridge baseline
4. **Parameter Validation** - Ensure all user-specified parameters are applied
5. **Constraint Validation** - Verify monotonic and interaction constraints work

## Rollout Strategy

1. **Phase 1** - Implement Huber integration in `_run_model_race`
2. **Phase 2** - Update model configurations and HPO functions  
3. **Phase 3** - Update training functions to use Huber-enhanced models
4. **Phase 4** - Add comprehensive testing and validation
5. **Phase 5** - Performance benchmarking and optimization

This plan provides a complete integration of Huber Regressor capabilities while maintaining backward compatibility and improving model performance through better feature selection, constraints, and initialization.
