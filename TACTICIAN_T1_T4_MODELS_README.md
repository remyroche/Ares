# Tactician T1-T4 Models Implementation

This document describes the implementation of the specialized Tactician T1-T4 models as requested.

## Overview

The Tactician T1-T4 models are a suite of specialized machine learning models designed for different aspects of trading signal generation:

- **T1-T3**: PatchTST-Enhanced Tree Models (LightGBM, XGBoost LambdaMART, CatBoost)
- **T4**: Causal Dilated TCN or TFT-Small (Sequence Models)

## Model Specifications

### T1-T3: PatchTST-Enhanced Tree Models

All T1-T3 models use PatchTST (Patch-based Time Series Transformer) embeddings to enhance tree-based models with temporal feature representations.

#### T1: PatchTST-LightGBM
- **Task**: Classification (up/down/none or two binaries) at horizon H
- **Architecture**: PatchTST-Embed + LightGBM
- **Loss Function**: Softmax (multi-class) or BCE (two-head)
- **Features**:
  - Monotone constraints for interpretable feature relationships
  - Fastest tree-based model for real-time inference
  - PatchTST attention mechanism for temporal patterns

#### T2: PatchTST-XGBoost LambdaMART
- **Task**: Pairwise ranking of candidate bars/windows by trade desirability
- **Architecture**: PatchTST-Embed + XGBoost with LambdaMART objective
- **Objective**: Pairwise ranking (rank:pairwise)
- **Setup**: Form groups by day/asset; label by realized payoff
- **Features**:
  - LambdaMART regularization for ranking optimization
  - Monotone constraints for financial feature relationships
  - Group-aware training for realistic evaluation

#### T3: PatchTST-CatBoost
- **Task**: Binary classification (up_hit@H and down_hit@H)
- **Architecture**: PatchTST-Embed + CatBoost
- **Loss Function**: BCE (two-head binary classification)
- **Features**:
  - Ordered boosting for better convergence
  - CatBoost depth 6-8 as specified
  - Early stopping for optimal generalization

### T4: Sequence Models

#### T4a: Causal Dilated TCN
- **Task**: Multi-label probabilities P(up_hit@H), P(down_hit@H) or regression for E[ret_H]
- **Architecture**: Causal Dilated Temporal Convolutional Network
- **Configuration**:
  - 6-8 residual blocks
  - 64 channels
  - Kernel size 3
  - Dilations: 1, 2, 4, 8, 16, 32, 64
  - Dropout: 0.1

#### T4b: TFT-Small (Alternative)
- **Task**: Ordinal regression or regression for E[ret_H]
- **Architecture**: Temporal Fusion Transformer (Small variant)
- **Configuration**:
  - Hidden size: 64
  - Attention heads: 4
  - Multiple layers with attention mechanisms

## Input Features

All models accept the following inputs:
- **Tabular features**: Technical indicators, price features, volume metrics
- **Regime features**: HMM states, market regime classifications
- **Analyst outputs**: Signal strength, confidence, risk scores
- **PatchTST embeddings**: Temporal feature representations (for T1-T3)

## Monotone Constraints

Tree models (T1-T3) implement monotone constraints for financial interpretability:

```python
monotone_constraints = {
    # Price features (positive relationship)
    'close_price': 1, 'open_price': 1, 'high_price': 1, 'low_price': 1,

    # Volume features (positive relationship)
    'volume': 1, 'quote_asset_volume': 1,

    # Volatility features (negative relationship)
    'volatility_10': -1, 'volatility_30': -1,

    # Technical indicators (no constraints)
    'rsi': 0, 'macd': 0, 'bollinger_position': 0
}
```

## Configuration Files

### Main Configuration
- **File**: `config/tactician_t1_t4_models_config.yaml`
- **Purpose**: Complete model specifications and hyperparameters

### Model Factory Integration
- **File**: `src/utils/ml_common/models/model_factory.py`
- **New Model Types**:
  - `PATCHTST_LIGHTGBM`
  - `PATCHTST_XGBOOST`
  - `PATCHTST_XGBOOST_LAMBDAMART`
  - `PATCHTST_CATBOOST`
  - `CAUSAL_DILATED_TCN`
  - `TFT_SMALL`

## Usage Examples

### Basic Model Creation

```python
from src.utils.ml_common.models.model_factory import EnhancedModelFactory, ModelType, ModelConfig

# Create T1 model
t1_config = ModelConfig(
    model_type=ModelType.PATCHTST_LIGHTGBM,
    model_name="t1_patchtst_lightgbm",
    n_outputs=3,
    model_params={
        'n_estimators': 2000,
        'learning_rate': 0.05,
        'max_depth': 8,
        'monotone_constraints': [1, 1, 1, 0, 0, -1, -1, 1, 0, 0],
        'patchtst_config': {
            'patch_len': 16,
            'stride': 8,
            'use_transformer_attention': True,
            'regime_aware': True
        }
    }
)

factory = EnhancedModelFactory()
t1_model = factory.create_model(t1_config)
```

### Training and Prediction

```python
# Fit the model
t1_model.fit(X_train, y_train)

# Make predictions
predictions = t1_model.predict(X_test)

# Get probabilities (for classification)
probabilities = t1_model.predict_proba(X_test)

# Get feature importance
importance = t1_model.get_feature_importance()
```

### PatchTST Wrapper Usage

```python
from src.training.steps.model_training.patchtst_wrapper import create_patchtst_wrapper

# Wrap any tree model with PatchTST
base_model = LightGBMRegressor(...)
patchtst_model = create_patchtst_wrapper(
    base_model,
    patch_len=16,
    stride=8,
    use_transformer_attention=True,
    regime_aware=True
)
```

## Implementation Details

### PatchTST Integration
- **Patch Creation**: Time series segmentation into fixed-length patches
- **Attention Mechanism**: Transformer-style attention for patch importance
- **Regime Awareness**: Regime-specific patch weighting
- **Feature Enhancement**: Combination of original and patch-based features

### Monotone Constraints Implementation
- **Advanced Method**: Uses XGBoost's advanced monotone constraints
- **Feature Engineering**: Automatic constraint assignment based on feature types
- **Validation**: Ensures constraints don't hurt performance significantly

### Loss Functions
- **Softmax**: Multi-class classification (T1)
- **BCE**: Binary cross-entropy for two-head classification (T3)
- **Pairwise Ranking**: LambdaMART objective for ranking tasks (T2)

## Performance Characteristics

### T1-T3 (Tree Models)
- **Speed**: Fastest inference (microsecond-level)
- **Memory**: Efficient memory usage
- **Interpretability**: Feature importance and monotone constraints
- **Scalability**: Handles large feature sets well

### T4 (Sequence Models)
- **Accuracy**: Superior for temporal pattern recognition
- **Latency**: Higher inference time but still suitable for real-time
- **Memory**: Higher memory requirements for sequence processing

## Integration with Existing Systems

### Analyst Integration
All models integrate with existing Analyst outputs:
- Signal strength filtering
- Confidence thresholding
- Risk score incorporation
- Regime label utilization

### Feature Pipeline
Models work with the existing feature engineering pipeline:
- Cross-timeframe features
- Technical indicators
- Market microstructure features
- Orderbook features

## Testing and Validation

### Model Validation
- **Cross-validation**: Time series split for realistic evaluation
- **Performance Metrics**: Task-specific evaluation metrics
- **Feature Importance**: Analysis of learned feature relationships
- **Monotone Constraint Validation**: Verification of constraint effectiveness

### Demonstration Script
- **File**: `examples/tactician_t1_t4_models_usage.py`
- **Purpose**: Complete usage demonstration for all T1-T4 models

## Future Enhancements

### Potential Improvements
1. **Advanced Monotone Constraints**: More sophisticated constraint learning
2. **Dynamic Patch Sizing**: Adaptive patch length based on market conditions
3. **Ensemble Integration**: Combine T1-T4 outputs for final trading decisions
4. **Online Learning**: Incremental model updates for concept drift

### Alternative Architectures
- **TFT-Large**: Larger transformer for more complex temporal patterns
- **WaveNet**: Alternative sequence model for causal convolutions
- **Neural ODE**: Continuous-time sequence modeling

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed (xgboost, lightgbm, catboost)
2. **Memory Issues**: Reduce batch size or sequence length for large datasets
3. **Monotone Constraint Failures**: Check feature ordering and constraint compatibility

### Debugging Tips
- Use verbose logging to track model creation and training
- Validate input shapes and data types
- Check feature importance for model health
- Monitor training curves for convergence issues

## Conclusion

The T1-T4 model suite provides a comprehensive approach to trading signal generation, combining the interpretability of tree models with the temporal pattern recognition capabilities of sequence models. The PatchTST enhancements improve the temporal awareness of tree models, while the causal dilated TCN provides state-of-the-art sequence modeling capabilities.

All models are designed for integration with the existing Tactician infrastructure and maintain compatibility with Analyst outputs and regime-based feature engineering.