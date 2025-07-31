# L1-L2 Regularization Implementation in Training Manager

## Overview

The TrainingManager (`/src/training/training_manager.py`) has been enhanced to include comprehensive L1-L2 regularization across all model types used in the Ares Trading Bot training pipeline.

## Implementation Summary

### 1. Regularization Configuration Management

The `TrainingManager` now includes:

- **`_get_regularization_config()`**: Extracts and organizes regularization parameters from `CONFIG['MODEL_TRAINING']['regularization']`
- **Centralized Configuration**: Single source of truth for all regularization parameters
- **Model-Specific Parameters**: Tailored regularization settings for each model type

### 2. Regularization Configuration Structure

```python
regularization_config = {
    # Base parameters from CONFIG
    'l1_alpha': 0.01,
    'l2_alpha': 0.001, 
    'dropout_rate': 0.2,
    
    # LightGBM specific
    'lightgbm': {
        'reg_alpha': 0.01,    # L1 regularization
        'reg_lambda': 0.001   # L2 regularization
    },
    
    # TensorFlow/Keras specific
    'tensorflow': {
        'l1_reg': 0.01,
        'l2_reg': 0.001,
        'dropout_rate': 0.2
    },
    
    # Sklearn specific
    'sklearn': {
        'alpha': 0.01,        # For Ridge/Lasso
        'l1_ratio': 0.5,      # For ElasticNet
        'C': 100.0            # For LogisticRegression
    },
    
    # TabNet specific
    'tabnet': {
        'lambda_sparse': 0.01,  # L1 regularization
        'lambda_l2': 0.001      # L2 regularization
    }
}
```

### 3. Component Integration

#### Training Pipeline Integration
- **`apply_regularization_to_components()`**: Injects regularization config into all ensemble instances
- **Automatic Application**: Called during component initialization in training pipeline
- **Validation**: `validate_and_report_regularization()` ensures proper configuration

#### Ensemble-Level Integration
- Each regime ensemble receives the full regularization configuration
- Base ensemble enhanced to use regularization parameters consistently
- Deep learning configurations updated with L1/L2 parameters

### 4. Model-Specific Implementations

#### LightGBM Models
**File**: `/src/analyst/predictive_ensembles/regime_ensembles/base_ensemble.py`

```python
def _get_lgbm_search_space(self, trial):
    if self.regularization_config and 'lightgbm' in self.regularization_config:
        reg_config = self.regularization_config['lightgbm']
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'reg_alpha': reg_config.get('reg_alpha', 0.01),    # Fixed L1 from config
            'reg_lambda': reg_config.get('reg_lambda', 0.001), # Fixed L2 from config
        }
```

#### TensorFlow/Keras Models
**File**: `/src/analyst/predictive_ensembles/regime_ensembles/bull_trend_ensemble.py`

```python
def _train_dl_model(self, X_seq, y_seq_encoded, num_classes, is_transformer=False):
    # Get regularization from config
    if self.regularization_config and 'tensorflow' in self.regularization_config:
        tf_config = self.regularization_config['tensorflow']
        l1_reg = tf_config.get('l1_reg', 0.01)
        l2_reg = tf_config.get('l2_reg', 0.001)
        dropout_rate = tf_config.get('dropout_rate', 0.2)
    
    # Create L1-L2 regularizer
    regularizer = l1_l2(l1=l1_reg, l2=l2_reg)
    
    # Apply to LSTM layers
    x = LSTM(
        self.dl_config["lstm_units"],
        kernel_regularizer=regularizer,
        recurrent_regularizer=regularizer
    )(inputs)
    
    # Apply to Dense layers
    outputs = Dense(
        num_classes, 
        activation='softmax',
        kernel_regularizer=regularizer
    )(x)
```

#### Sklearn Models
**File**: `/src/analyst/predictive_ensembles/regime_ensembles/base_ensemble.py`

```python
def _get_regularized_logistic_regression(self):
    if self.regularization_config and 'sklearn' in self.regularization_config:
        sklearn_config = self.regularization_config['sklearn']
        
        # Use ElasticNet penalty for combined L1/L2 regularization
        model = LogisticRegression(
            penalty='elasticnet',
            C=sklearn_config.get('C', 1.0),
            l1_ratio=sklearn_config.get('l1_ratio', 0.5),  # 0.5 = equal L1/L2
            solver='saga',  # Required for elasticnet
            random_state=42,
            max_iter=1000
        )
```

#### TabNet Models
**File**: `/src/analyst/predictive_ensembles/regime_ensembles/bull_trend_ensemble.py`

```python
def _train_tabnet_model(self, X_flat, y_flat_encoded):
    if self.regularization_config and 'tabnet' in self.regularization_config:
        tabnet_config = self.regularization_config['tabnet']
        lambda_sparse = tabnet_config.get('lambda_sparse', 0.01)
        lambda_l2 = tabnet_config.get('lambda_l2', 0.001)
    
    model = TabNetClassifier(
        lambda_sparse=lambda_sparse,  # L1 regularization
        reg_lambda=lambda_l2,         # L2 regularization
        verbose=0
    )
```

### 5. Configuration Source

The regularization parameters are read from:
```python
CONFIG['MODEL_TRAINING']['regularization'] = {
    'l1_alpha': 0.01,
    'l2_alpha': 0.001,
    'dropout_rate': 0.2
}
```

### 6. Training Pipeline Integration

The regularization is automatically applied during training:

```python
async def run_full_training(self, symbol: str, exchange_name: str = "BINANCE") -> bool:
    # Initialize components
    if not await self.initialize_components(symbol):
        return False
    
    # Apply L1-L2 regularization configuration to all components
    self.apply_regularization_to_components()
    
    # Validate and report regularization configuration
    if not self.validate_and_report_regularization():
        self.logger.warning("Regularization validation failed, but continuing with training...")
    
    # Continue with training steps...
```

### 7. Validation and Reporting

The implementation includes comprehensive validation:

- **Configuration Completeness**: Checks for required parameters
- **Value Validation**: Ensures parameters are within valid ranges
- **Detailed Reporting**: Logs all regularization settings for transparency
- **Model-Specific Reporting**: Shows how regularization is applied to each model type

### 8. Files Modified

1. **`/src/training/training_manager.py`**:
   - Added regularization configuration management
   - Added component application methods
   - Added validation and reporting
   - Integrated into training pipeline

2. **`/src/analyst/predictive_ensembles/regime_ensembles/base_ensemble.py`**:
   - Enhanced LightGBM regularization
   - Added regularized LogisticRegression method
   - Added regularization config attribute

3. **`/src/analyst/predictive_ensembles/regime_ensembles/bull_trend_ensemble.py`**:
   - Enhanced TensorFlow/Keras models with L1-L2 regularization
   - Enhanced TabNet models with regularization
   - Updated LogisticRegression to use base class method

4. **`/src/analyst/predictive_ensembles/regime_ensembles/bear_trend_ensemble.py`**:
   - Updated LogisticRegression to use regularized version

### 9. Benefits

- **Overfitting Prevention**: L1-L2 regularization helps prevent overfitting across all model types
- **Consistency**: Centralized configuration ensures consistent regularization across all models
- **Flexibility**: Easy to adjust regularization parameters from configuration
- **Transparency**: Comprehensive logging shows exactly how regularization is applied
- **Maintainability**: Clean separation of concerns and consistent implementation patterns

## Usage

When running training through the TrainingManager, L1-L2 regularization is automatically applied to all models based on the configuration in `CONFIG['MODEL_TRAINING']['regularization']`. The system will log detailed information about the regularization parameters being used for each model type.

## Conclusion

The implementation ensures that L1-L2 regularization is comprehensively applied across all model types in the training pipeline, providing a robust defense against overfitting while maintaining flexibility and transparency in configuration.
