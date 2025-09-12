# Multi-Output Stacking Ensemble Architecture & Pipeline Recap

## 🏗️ Overall Architecture

### Hierarchical Model Structure
```
┌─────────────────────────────────────────────────────────────────┐
│                    MARKET DATA (Cross-timeframe)                │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    ANALYST (5m timeframe)                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Per-Regime Training: Separate models for each market regime││
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐││
│  │  │    GRU      │ │  CatBoost   │ │  LightGBM   │ │   RF    │││
│  │  │ (Primary)   │ │ (Financial) │ │ (Speed)     │ │ (Meta)  │││
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘││
│  │                                                             ││
│  │  Meta-Model: Ridge (trained on features + base outputs)    ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  Outputs: [signal_strength, confidence, risk_score, regime_label]│
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      │ Green Light Signal (confidence > threshold)
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                  TACTICIAN (1m timeframe)                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Hybrid Training: Whole dataset + Analyst features          ││
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐││
│  │  │    NODE     │ │  CatBoost   │ │  LightGBM   │ │  Ridge  │││
│  │  │ (Primary)   │ │ (Regime)    │ │ (Speed)     │ │ (Meta)  │││
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘││
│  │                                                             ││
│  │  Meta-Model: Ridge (trained on features + base outputs)    ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  Outputs: [entry_timing, position_size, stop_loss, take_profit] │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Training Pipeline

### Phase 1: Data Preparation & Regime Detection
```python
# 1. Load cross-timeframe market data
market_data = load_cross_timeframe_data(
    timeframes=['1m', '5m', '15m', '30m', '1h'],
    features=['price', 'volume', 'volatility', 'momentum']
)

# 2. Detect market regimes using HMM
regime_detector = HMMRegimeDetector(n_components=3)
regime_labels = regime_detector.fit_predict(market_data)

# 3. Generate multi-output targets
analyst_targets = generate_analyst_targets(market_data, regime_labels)
tactician_targets = generate_tactician_targets(market_data, regime_labels)
```

### Phase 2: Hierarchical HPO (Base Models First)
```python
# Phase 1: Optimize base models
hpo_config = HierarchicalHPOConfig(
    phase1_config=HPOPhaseConfig(
        phase_name="base_models",
        models=analyst_base_models,  # GRU, CatBoost, LightGBM, RF
        search_spaces=base_model_search_spaces,
        n_trials=100
    ),
    phase2_config=HPOPhaseConfig(
        phase_name="meta_models", 
        models=analyst_meta_models,  # Ridge
        search_spaces=meta_model_search_spaces,
        n_trials=50
    )
)

hpo = HierarchicalHPO(hpo_config)
optimized_models = hpo.optimize_ensemble(X_train, y_train)
```

### Phase 3: Analyst Training (Per-Regime)
```python
# Train Analyst per-regime
analyst_trainer = RegimeAwareAnalystTrainer(
    config=RegimeAwareAnalystConfig(
        min_samples_per_regime=1000,
        enable_data_augmentation=True,
        augmentation_method="smote"
    )
)

analyst_results = analyst_trainer.train_analyst(
    X=market_data_5m,
    y=analyst_targets,
    regime_labels=regime_labels
)

# Results: Separate models for each regime + global fallback
# - regime_models: {regime_id: {output: {model_name: model}}}
# - regime_meta_models: {regime_id: {output: meta_model}}
# - global_model: Fallback for regimes with insufficient data
```

### Phase 4: Tactician Training (Hybrid Approach)
```python
# Train Tactician on whole dataset with Analyst features
tactician_trainer = HybridTacticianTrainer(
    config=HybridTacticianConfig(
        analyst_model_path="./analyst_models",
        analyst_threshold=0.6,
        enable_regime_features=True
    )
)

tactician_results = tactician_trainer.train_tactician(
    X=market_data_1m,
    y=tactician_targets,
    regime_labels=regime_labels
)

# Results: Single model trained on features + Analyst outputs
# - tactician_models: {output: {model_name: model}}
# - meta_model: {output: meta_model}
# - Uses Analyst predictions as additional features
```

### Phase 5: SHAP/LIME Explainability Integration
```python
# Generate explanations for all models
explainer = SHAPLIMEExplainer(
    config=ExplanationConfig(
        enable_shap=True,
        enable_lime=True,
        explain_all_outputs=True
    )
)

# Explain Analyst models
analyst_explanations = explainer.explain_stacking_ensemble(
    base_models=analyst_results['regime_models'][regime_id],
    meta_model=analyst_results['regime_meta_models'][regime_id],
    X=market_data_5m,
    output_names=analyst_output_names
)

# Explain Tactician models
tactician_explanations = explainer.explain_stacking_ensemble(
    base_models=tactician_results['tactician_models'],
    meta_model=tactician_results['meta_model'],
    X=combined_features,  # Features + Analyst outputs
    output_names=tactician_output_names
)
```

## 🎯 Model Specifications

### Analyst Model (5m timeframe)
- **Role**: Decides IF to trade and emits green light for Tactician
- **Training**: Per-regime with fallback to global model
- **Base Models**:
  - **GRU**: Primary model for time series patterns
  - **CatBoost**: Financial data and regime handling
  - **LightGBM**: Speed and robustness
  - **RandomForest**: Meta-model for ensemble
- **Meta-Model**: Ridge (trained on features + base model outputs)
- **Outputs**: 
  - `signal_strength`: Trading signal strength (-1 to 1)
  - `confidence`: Model confidence (0 to 1)
  - `risk_score`: Risk assessment (0 to 1)
  - `regime_label`: Market regime classification

### Tactician Model (1m timeframe)
- **Role**: Decides WHEN to trade (only when Analyst gives green light)
- **Training**: Hybrid approach on whole dataset with Analyst features
- **Base Models**:
  - **NODE**: Primary model for tabular data with attention
  - **CatBoost**: Regime handling and financial data
  - **LightGBM**: Speed and robustness
  - **Ridge**: Meta-model for ensemble
- **Meta-Model**: Ridge (trained on features + base model outputs)
- **Dependencies**: Requires Analyst outputs as input features
- **Outputs**:
  - `entry_timing`: Optimal entry timing (-1 to 1)
  - `position_size`: Position size recommendation (0 to 1)
  - `stop_loss`: Stop loss level (price-based)
  - `take_profit`: Take profit level (price-based)

## 🔧 Key Implementation Details

### 1. HPO Timing (✅ Implemented)
- **Phase 1**: Base model optimization first
- **Phase 2**: Meta model optimization with fixed base models
- **Rationale**: Base models are foundation; meta models can only be as good as base predictions

### 2. Analyst Per-Regime Training (✅ Implemented)
- **Strategy**: Separate models for each market regime
- **Fallback**: Global model for regimes with insufficient data
- **Data Augmentation**: SMOTE for small regimes
- **Minimum Samples**: 1000 samples per regime

### 3. Tactician Hybrid Training (✅ Implemented)
- **Strategy**: Whole dataset with Analyst features
- **Features**: Original features + Analyst model outputs + regime features
- **Dependency**: Only trains on periods where Analyst gives green light
- **Regime Awareness**: Uses regime features to understand market conditions

### 4. Meta Model Training (✅ Fixed)
- **Input**: Original features + base model predictions
- **Training**: `meta_features = np.hstack([X, base_predictions])`
- **Prediction**: `meta_pred = meta_model.predict(meta_features)`
- **Consistency**: Same approach for both training and prediction

### 5. SHAP/LIME Integration (✅ Implemented)
- **Coverage**: All models at every training step
- **Multi-Output**: Per-output explanations
- **Stacking**: Explains both base models and meta models
- **Caching**: Performance optimization for large-scale explanations

## 📊 Data Flow

### Training Flow
```
Market Data → Regime Detection → Multi-Output Target Generation
     ↓
Analyst Per-Regime Training → Analyst Model Outputs
     ↓
Tactician Hybrid Training (Features + Analyst Outputs) → Tactician Model Outputs
     ↓
SHAP/LIME Explanations → Model Interpretability
```

### Prediction Flow
```
New Market Data → Regime Classification
     ↓
Analyst Prediction → Green Light Check (confidence > threshold)
     ↓
Tactician Prediction (if green light) → Trading Decisions
     ↓
SHAP/LIME Explanations → Decision Interpretability
```

## 🚀 Performance Optimizations

### M1 Hardware Optimizations
- **GPU Acceleration**: Metal Performance Shaders for neural networks
- **Memory Optimization**: Efficient memory usage for large datasets
- **Parallel Processing**: Multi-core optimization for ensemble training
- **Caching**: Intelligent caching for repeated operations

### Computational Efficiency
- **Hierarchical HPO**: Reduces search space and computation time
- **Regime-Aware Training**: Focuses on relevant data for each regime
- **Hybrid Tactician**: Uses whole dataset efficiently with Analyst features
- **Explanation Caching**: Avoids recomputing explanations

## 📈 Expected Benefits

1. **Better Performance**: Optimized base models and proper meta model training
2. **Regime-Specific Optimization**: Models tailored to market conditions
3. **Enhanced Interpretability**: SHAP/LIME explanations for all decisions
4. **Robust Architecture**: Fallback strategies and comprehensive error handling
5. **Computational Efficiency**: Smart training strategies and hardware optimization
6. **Maintainable Codebase**: Clean organization and proper documentation

## 🔍 Monitoring & Validation

### Performance Metrics
- **Per-Output Metrics**: MSE, MAE, R² for each output
- **Overall Metrics**: Weighted average across all outputs
- **Regime-Specific Metrics**: Performance by market regime
- **Confidence Calibration**: Reliability of confidence scores

### Model Validation
- **Time Series CV**: Proper temporal validation
- **Regime Stability**: Performance consistency across regimes
- **Explanation Quality**: SHAP/LIME explanation reliability
- **Ensemble Diversity**: Base model diversity and meta model effectiveness

This architecture provides a robust, interpretable, and efficient multi-output stacking ensemble system that leverages both regime-specific and global patterns for optimal trading decisions.