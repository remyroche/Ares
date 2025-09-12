# HPO Timing and Tactician Training Strategy Analysis

## Question A: HPO Timing - Before or After Meta Model Training?

### Recommendation: **HPO should happen BEFORE meta model training**

### Rationale:

#### 1. **Base Model Optimization First**
- **Base models are the foundation** of the stacking ensemble
- Poor base models will limit the meta model's performance regardless of optimization
- Meta models can only be as good as the base model predictions they receive

#### 2. **Computational Efficiency**
- HPO on base models is computationally cheaper than on meta models
- Meta models require base model predictions, which are expensive to generate during HPO
- Early optimization reduces the search space for meta model optimization

#### 3. **Stability and Convergence**
- Well-optimized base models provide stable, high-quality predictions
- This stability helps meta models converge faster and more reliably
- Reduces the risk of meta model overfitting to poor base model predictions

#### 4. **Hierarchical Optimization Strategy**
```
Phase 1: Base Model HPO
├── GRU hyperparameters (Analyst)
├── NODE hyperparameters (Tactician)  
├── CatBoost hyperparameters
├── LightGBM hyperparameters
└── RandomForest hyperparameters

Phase 2: Meta Model HPO (with fixed base models)
├── Ridge hyperparameters
└── Ensemble weights
```

#### 5. **Implementation Strategy**
```python
# Phase 1: Base Model HPO
for model_name, model in base_models.items():
    hpo_result = optimize_hyperparameters(
        model=model,
        X_train=X_train,
        y_train=y_train,
        search_space=model_search_spaces[model_name],
        n_trials=100
    )
    optimized_models[model_name] = hpo_result.best_model

# Phase 2: Meta Model HPO (with optimized base models)
meta_hpo_result = optimize_hyperparameters(
    model=meta_model,
    X_train=create_meta_features(X_train, optimized_models),
    y_train=y_train,
    search_space=meta_search_space,
    n_trials=50
)
```

## Question B: Tactician Training Strategy - Per-Regime vs Whole Dataset?

### Recommendation: **Hybrid Approach - Per-Regime with Fallback**

### Rationale:

#### 1. **Data Scarcity Analysis**
- Tactician is only trained on Analyst's "green light" periods
- This significantly reduces available training data
- Per-regime training would further reduce data, potentially causing overfitting

#### 2. **Regime-Specific Characteristics**
- Different market regimes have different optimal trading strategies
- Entry timing, position sizing, and risk management vary by regime
- Per-regime models can capture regime-specific patterns better

#### 3. **Proposed Hybrid Strategy**

```python
def train_tactician_hybrid(X, y, regime_labels, min_samples_per_regime=1000):
    """
    Hybrid training strategy for Tactician:
    1. If regime has enough samples -> train per-regime model
    2. If regime has insufficient samples -> use global model with regime features
    3. Use ensemble of regime-specific and global models
    """
    
    regime_models = {}
    global_model = None
    
    for regime in unique_regimes:
        regime_mask = regime_labels == regime
        regime_samples = np.sum(regime_mask)
        
        if regime_samples >= min_samples_per_regime:
            # Train regime-specific model
            X_regime = X[regime_mask]
            y_regime = y[regime_mask]
            regime_models[regime] = train_tactician_model(X_regime, y_regime)
        else:
            # Use global model with regime features
            if global_model is None:
                X_with_regime = add_regime_features(X, regime_labels)
                global_model = train_tactician_model(X_with_regime, y)
    
    return regime_models, global_model
```

#### 4. **Implementation Details**

##### A. **Data Augmentation for Small Regimes**
```python
def augment_regime_data(X, y, regime, target_samples=1000):
    """Augment data for regimes with insufficient samples"""
    current_samples = len(X)
    if current_samples >= target_samples:
        return X, y
    
    # Use SMOTE or similar for synthetic data generation
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_aug, y_aug = smote.fit_resample(X, y)
    
    return X_aug, y_aug
```

##### B. **Regime-Aware Feature Engineering**
```python
def add_regime_features(X, regime_labels):
    """Add regime-specific features to help global model"""
    regime_features = []
    
    # One-hot encoding of regime
    regime_onehot = pd.get_dummies(regime_labels, prefix='regime')
    regime_features.append(regime_onehot.values)
    
    # Regime transition features
    regime_transitions = np.diff(regime_labels)
    regime_features.append(regime_transitions.reshape(-1, 1))
    
    # Regime duration features
    regime_durations = calculate_regime_durations(regime_labels)
    regime_features.append(regime_durations.reshape(-1, 1))
    
    return np.hstack([X] + regime_features)
```

##### C. **Ensemble Prediction Strategy**
```python
def predict_tactician_hybrid(X, regime_labels, regime_models, global_model):
    """Make predictions using hybrid approach"""
    predictions = []
    
    for i, regime in enumerate(regime_labels):
        if regime in regime_models:
            # Use regime-specific model
            pred = regime_models[regime].predict(X[i:i+1])
        else:
            # Use global model with regime features
            X_with_regime = add_regime_features(X[i:i+1], [regime])
            pred = global_model.predict(X_with_regime)
        
        predictions.append(pred)
    
    return np.array(predictions)
```

#### 5. **Minimum Sample Requirements**

| Regime Type | Min Samples | Strategy |
|-------------|-------------|----------|
| High Volatility | 500 | Per-regime if possible |
| Low Volatility | 1000 | Per-regime |
| Trending | 800 | Per-regime if possible |
| Sideways | 1200 | Per-regime |
| Transition | 200 | Global model only |

#### 6. **Performance Monitoring**

```python
def monitor_tactician_performance(predictions, y_true, regime_labels):
    """Monitor performance by regime"""
    performance_by_regime = {}
    
    for regime in unique_regimes:
        regime_mask = regime_labels == regime
        if np.sum(regime_mask) > 0:
            regime_pred = predictions[regime_mask]
            regime_true = y_true[regime_mask]
            
            performance_by_regime[regime] = {
                'mse': mean_squared_error(regime_true, regime_pred),
                'mae': mean_absolute_error(regime_true, regime_pred),
                'r2': r2_score(regime_true, regime_pred),
                'samples': len(regime_pred)
            }
    
    return performance_by_regime
```

## Implementation Priority

### Phase 1: Base Model HPO
1. Implement HPO for all base models (GRU, NODE, CatBoost, LightGBM, RandomForest)
2. Use Bayesian optimization with early stopping
3. Focus on model-specific hyperparameters

### Phase 2: Meta Model HPO
1. Implement HPO for Ridge meta models
2. Optimize ensemble weights
3. Use cross-validation with time series splits

### Phase 3: Tactician Hybrid Training
1. Implement regime detection and labeling
2. Implement hybrid training strategy
3. Add data augmentation for small regimes
4. Implement ensemble prediction logic

### Phase 4: Performance Monitoring
1. Add regime-specific performance tracking
2. Implement adaptive training strategies
3. Add model selection based on regime performance

## Expected Benefits

1. **Better Base Model Performance**: HPO before meta training ensures optimal base models
2. **Regime-Specific Optimization**: Tactician models tailored to market conditions
3. **Robust Fallback Strategy**: Global model ensures coverage for all regimes
4. **Computational Efficiency**: Hierarchical optimization reduces search space
5. **Better Generalization**: Hybrid approach balances specialization and generalization