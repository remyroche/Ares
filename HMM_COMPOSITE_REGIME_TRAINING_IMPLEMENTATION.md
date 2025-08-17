# HMM Composite Regime Training Implementation

## Overview

This document describes the implementation of regime-specific ensemble training using HMM composite regimes instead of traditional bull/bear/sideways regimes.

## Changes Made

### 1. **Step 3: Feature Engineering - Added HMM Composite Regime Splitting**

**File**: `src/training/steps/step3_feature_engineering.py`

**New Functionality**:
- Added `_hmm_composite_regime_splitting()` function that runs after feature engineering
- Loads HMM composite regime data from `{exchange}_{symbol}_hmm_composite_meta_{timeframe}.json`
- Splits data by `composite_cluster_id` for each regime
- Creates regime-specific parquet files in `data/training/regime_data/`
- Generates regime splitting summary in `{exchange}_{symbol}_hmm_composite_regime_splits.json`
- Creates gating matrix for ensemble training

**Output Structure**:
```
data/training/regime_data/
├── train_hmm_composite_0.parquet
├── train_hmm_composite_1.parquet
├── validation_hmm_composite_0.parquet
├── validation_hmm_composite_1.parquet
├── test_hmm_composite_0.parquet
└── test_hmm_composite_1.parquet

data/training/gating/
└── {exchange}_{symbol}_hmm_composite_gating.parquet
```

### 2. **Step 5: HMM-Based Training - Added Regime-Specific Training**

**File**: `src/training/steps/step5_hmm_based_training.py`

**New Functionality**:

#### **New Methods Added**:
- `_load_hmm_composite_regime_data()`: Loads regime-specific data splits
- `_train_regime_specific_models()`: Main method for regime-specific training
- `_train_lightgbm_model_regime()`: LightGBM training for specific regimes
- `_train_cnn_model_regime()`: CNN training for specific regimes (placeholder)
- `_train_tcn_model_regime()`: TCN training for specific regimes (placeholder)
- `_train_transformer_model_regime()`: Transformer training for specific regimes (placeholder)

#### **Modified Training Flow**:
1. **Primary**: Attempt regime-specific training using HMM composite regime data
2. **Fallback**: If regime data not available, use combined training (original approach)

#### **Training Results Structure**:
```python
training_results = {
    "1m": {
        "training_type": "regime_specific",
        "regime_models": {
            "hmm_composite_0": {
                "model": trained_model,
                "description": "Strong upward momentum with acceleration...",
                "architecture": "LightGBM",
                "data_sizes": {"train": 1000, "validation": 200, "test": 100}
            },
            "hmm_composite_1": {
                # ... similar structure
            }
        },
        "total_regimes": 2,
        "architecture": "LightGBM"
    }
}
```

## Key Benefits

### 1. **True Regime-Specific Training**
- Each HMM composite regime gets its own specialized model
- Models are trained only on data from their specific regime
- Better capture of regime-specific patterns and behaviors

### 2. **Improved Model Performance**
- Models can specialize in specific market conditions
- Reduced noise from other regimes during training
- Better generalization within each regime

### 3. **Enhanced Ensemble Capabilities**
- Multiple specialized models for different market conditions
- Gating matrix enables intelligent model selection
- Better adaptation to changing market regimes

### 4. **Descriptive Regime Names**
- Uses actual HMM archetype descriptions instead of generic "bull/bear/sideways"
- More informative regime identification
- Better understanding of what each model specializes in

## Implementation Details

### **Regime Data Structure**
Each regime split contains:
- All engineered features
- `composite_cluster_id`: The HMM regime identifier
- `regime_description`: Human-readable description of the regime
- `target`: The regime ID for training

### **Training Process**
1. **Data Loading**: Load regime-specific data for each HMM composite regime
2. **Model Training**: Train specialized model for each regime using only that regime's data
3. **Evaluation**: Evaluate each model on regime-specific validation/test data
4. **Ensemble Assembly**: Combine all regime models into an ensemble

### **Fallback Mechanism**
If regime-specific data is not available:
1. Log warning about missing regime data
2. Fall back to combined training (original approach)
3. Train single model on all data

## Usage

### **Automatic Execution**
The regime-specific training is automatically triggered when:
1. Step 3 completes and generates regime splits
2. Step 5 runs and finds regime-specific data available

### **Manual Execution**
To force regime-specific training:
```python
# Ensure regime data exists
await step3.execute(...)

# Run training with regime-specific data
await step5.execute(...)
```

## Future Enhancements

### **1. Complete Neural Network Support**
- Implement CNN, TCN, and Transformer training for regimes
- Add sequence-aware training for time series data
- Support for regime-specific hyperparameter optimization

### **2. Advanced Ensemble Methods**
- Dynamic model weighting based on regime probabilities
- Regime transition-aware ensemble selection
- Online learning for regime model adaptation

### **3. Performance Monitoring**
- Regime-specific performance metrics
- Regime transition prediction accuracy
- Model drift detection per regime

## Configuration

### **Step 3 Configuration**
```python
# Regime splitting is automatic when HMM composite data exists
# No additional configuration needed
```

### **Step 5 Configuration**
```python
# Regime-specific training is automatic when regime data exists
# Falls back to combined training if regime data not available
```

## Monitoring and Logging

### **Key Log Messages**
- `"🔄 Starting HMM composite regime data splitting..."`
- `"✅ HMM composite regime splitting completed: X regimes"`
- `"🎯 Training regime-specific models for {timeframe}"`
- `"✅ Trained X regime-specific models for {timeframe}"`

### **Performance Metrics**
- Per-regime model accuracy
- Regime data distribution
- Training time per regime
- Model ensemble performance

## Conclusion

This implementation provides a robust foundation for regime-specific ensemble training using HMM composite regimes. The system automatically adapts between regime-specific and combined training based on data availability, ensuring backward compatibility while enabling advanced ensemble capabilities.
