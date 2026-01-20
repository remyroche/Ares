# Checkpoint System Usage Guide

## Overview

The checkpoint system provides robust checkpointing for Layer 2.5, Layer 3, and Layer 4, enabling:
- **Resume execution** from any sub-step
- **Clean artifacts** from specific steps onwards  
- **Automatic replacement** of old checkpoints
- **Unified interface** across all layers

## Architecture

### Layer-Specific Sub-Steps

#### **Layer 2.5 Chaser** (12 sub-steps)
```
0: data_preparation       - Load and prepare data for chaser training
1: teacher_training       - Train BayesianRidge teacher models
2: teacher_validation     - Cross-validate teacher predictions
3: residual_computation   - Compute residuals and uncertainty weights
4: student_training_xgb   - Train XGBoost chaser students
5: student_training_lgb   - Train LightGBM chaser students
6: student_training_cat   - Train CatBoost chaser students
7: student_training_et    - Train ExtraTrees chaser students
8: model_selection        - Select top performing chaser models
9: ensemble_creation       - Create ensemble predictions
10: final_validation       - Final validation and performance metrics
11: artifact_saving         - Save models and predictions
```

#### **Layer 3 Meta-Models** (12 sub-steps)
```
0: data_loading           - Load OOF data and base model columns
1: entropy_bars_integration - Integrate entropy bars and specialized features
2: meta_features_engineering - Generate regime-aware and meta features
3: feature_clustering      - Apply mild MP-clustering for feature selection
4: layer25_integration     - Integrate Layer 2.5 chaser models (if available)
5: dual_head_training       - Train all model families (ET, LGBM, XGB, CatBoost, Huber, Ridge)
6: model_selection_12      - Select best models for 12-bar horizon
7: model_selection_48      - Select best models for 48-bar horizon
8: oof_predictions         - Generate OOF predictions for all models
9: race_reporting          - Generate comprehensive model race reports
10: enhanced_reporting      - Generate enhanced Layer 3 reports
11: final_processing        - Final validation and artifact saving
```

#### **Layer 4 Gate Models** (9 sub-steps)
```
0: data_preparation       - Load meta model OOF predictions and features
1: confidence_filtering    - Filter predictions by confidence threshold (>0.4)
2: feature_engineering    - Add regime and performance features
3: gate_model_training    - Train gate models (ExtraTrees vs Ridge)
4: gate_validation       - Validate gate model performance
5: final_predictions     - Generate final gated predictions
6: performance_analysis  - Compare meta vs gate performance
7: artifact_saving        - Save final models and predictions
8: deployment_prep        - Prepare for production deployment
```

## Usage Examples

### Basic Usage

#### **1. Get Checkpoint Manager**
```python
from src.training.steps.labeling.unified_checkpoint_manager import get_checkpoint_manager

# Get checkpoint manager for any layer
manager = get_checkpoint_manager('layer3', 'ETHUSDT')

# Or use layer-specific functions
from src.training.steps.labeling.layer3_checkpoint_manager import get_layer3_checkpoint_manager
manager = get_layer3_checkpoint_manager('ETHUSDT')
```

#### **2. Save Checkpoint**
```python
# Save checkpoint data
checkpoint_data = {
    'models': trained_models,
    'predictions': oof_predictions,
    'metadata': training_info
}

config = {
    'symbol': 'ETHUSDT',
    'timeframe': '15m',
    'fast_mode': False
}

path = manager.save_checkpoint('dual_head_training', checkpoint_data, config)
print(f"Checkpoint saved to: {path}")
```

#### **3. Load Checkpoint**
```python
# Load checkpoint data
data = manager.load_checkpoint('dual_head_training')
if data:
    models = data['models']
    predictions = data['predictions']
    print(f"Loaded checkpoint with {len(data)} keys")
else:
    print("No checkpoint found")
```

#### **4. Auto-Resume**
```python
from src.training.steps.labeling.unified_checkpoint_manager import UnifiedCheckpointManager

# Auto-resume from latest checkpoint
resume_step, manager = UnifiedCheckpointManager.auto_resume_pipeline('layer3', 'ETHUSDT')
print(f"Resuming from step: {resume_step}")

# Or check manually
latest = manager.get_latest_checkpoint()
if latest:
    step_name, metadata = latest
    print(f"Latest checkpoint: {step_name} at {metadata.timestamp}")
```

### Advanced Usage

#### **1. List All Checkpoints**
```python
# List all checkpoints for a symbol
checkpoints = manager.list_checkpoints()
for step_name, metadata in checkpoints:
    print(f"{step_name}: {metadata.timestamp} ({len(metadata.data_keys)} keys)")
```

#### **2. Delete Checkpoints**
```python
# Delete checkpoints from a specific step onwards
deleted_count = manager.delete_checkpoints_from('dual_head_training')
print(f"Deleted {deleted_count} checkpoints")
```

#### **3. Validate Data Before Saving**
```python
# Validate checkpoint data
try:
    manager.validate_checkpoint_data('dual_head_training', checkpoint_data)
    print("Data validation passed")
except ValueError as e:
    print(f"Validation failed: {e}")
```

#### **4. Get All Layer Managers**
```python
from src.training.steps.labeling.unified_checkpoint_manager import get_all_checkpoint_managers

# Get managers for all layers
all_managers = get_all_checkpoint_managers('ETHUSDT')
for layer_name, manager in all_managers.items():
    latest = manager.get_latest_checkpoint()
    if latest:
        step_name, metadata = latest
        print(f"{layer_name}: Latest checkpoint at {step_name}")
```

## Integration in Pipelines

### **Layer 3 Integration Example**
```python
from src.training.steps.labeling.layer3_checkpoint_manager import get_layer3_checkpoint_manager

def layer3_analyst_lgbm_with_checkpoints(df, config):
    symbol = config.get('symbol', 'ETHUSDT')
    manager = get_layer3_checkpoint_manager(symbol)
    
    # Auto-resume from latest checkpoint
    resume_step = manager.get_auto_resume_step()
    
    if resume_step == 'data_loading':
        # Start from beginning
        df_processed = load_and_process_data(df)
        manager.save_checkpoint('data_loading', {'df': df_processed}, config)
    
    if resume_step in ['data_loading', 'entropy_bars_integration']:
        # Load or compute entropy bars
        entropy_data = manager.load_checkpoint('entropy_bars_integration')
        if not entropy_data:
            entropy_data = compute_entropy_bars(df_processed)
            manager.save_checkpoint('entropy_bars_integration', entropy_data, config)
    
    if resume_step in ['data_loading', 'entropy_bars_integration', 'dual_head_training']:
        # Train models or load from checkpoint
        model_data = manager.load_checkpoint('dual_head_training')
        if not model_data:
            model_data = train_all_models(df_processed, entropy_data)
            manager.save_checkpoint('dual_head_training', model_data, config)
    
    # Continue with remaining steps...
    return df_processed, model_data
```

### **Layer 2.5 Integration Example**
```python
from src.training.steps.labeling.layer25_checkpoint_manager import get_layer25_checkpoint_manager

def train_chaser_with_checkpoints(X, y, config):
    symbol = config.get('symbol', 'ETHUSDT')
    manager = get_layer25_checkpoint_manager(symbol)
    
    # Auto-resume
    resume_step = manager.get_auto_resume_step()
    
    if resume_step == 'teacher_training':
        # Train teacher models
        teacher_data = train_teacher_models(X, y, config)
        manager.save_checkpoint('teacher_training', teacher_data, config)
    
    if resume_step in ['teacher_training', 'student_training_xgb']:
        # Train XGBoost students
        xgb_data = manager.load_checkpoint('student_training_xgb')
        if not xgb_data:
            teacher_data = manager.load_checkpoint('teacher_training')
            xgb_data = train_xgb_students(X, y, teacher_data, config)
            manager.save_checkpoint('student_training_xgb', xgb_data, config)
    
    # Continue with other student models...
    return all_student_models
```

## Storage Structure

### **Checkpoint Directory Layout**
```
versioned_artifacts/
├── layer2_checkpoints/
│   └── ETHUSDT/
│       ├── checkpoint_data_loading.h5
│       ├── checkpoint_data_loading.json
│       ├── checkpoint_causal_discovery.h5
│       ├── checkpoint_causal_discovery.json
│       └── ...
├── layer25_checkpoints/
│   └── ETHUSDT/
│       ├── checkpoint_teacher_training.h5
│       ├── checkpoint_teacher_training.json
│       ├── checkpoint_student_training_xgb.h5
│       └── ...
├── layer3_checkpoints/
│   └── ETHUSDT/
│       ├── checkpoint_dual_head_training.h5
│       ├── checkpoint_dual_head_training.json
│       └── ...
└── layer4_checkpoints/
    └── ETHUSDT/
        ├── checkpoint_gate_model_training.h5
        └── ...
```

### **File Formats**
- **`.h5`**: HDF5 format for DataFrames, Series, and arrays
- **`.json`**: Metadata and non-tabular data (models, dicts, etc.)

## Configuration

### **Checkpoint Configuration**
```python
config = {
    # Layer-specific settings
    'symbol': 'ETHUSDT',
    'timeframe': '15m',
    'execution_mode': 'blank',
    
    # Layer 2.5 specific
    'chaser_model_types': ['xgb', 'lgb', 'cat', 'et'],
    'top_n_models': 3,
    'winsor_k': 4.0,
    
    # Layer 3 specific
    'use_entropy_bars': True,
    'layer25_chaser_enabled': True,
    'fast_mode': False,
    
    # Layer 4 specific
    'confidence_threshold': 0.4,
    'gate_model_types': ['extratrees', 'ridge']
}
```

## Best Practices

### **1. Checkpoint Granularity**
- **Save after expensive operations** (model training, feature engineering)
- **Validate data before saving** to avoid corrupted checkpoints
- **Use meaningful step names** that reflect the operation

### **2. Error Handling**
```python
try:
    data = manager.load_checkpoint('expensive_step')
    if not data:
        # Compute and save
        data = compute_expensive_results()
        manager.save_checkpoint('expensive_step', data, config)
except Exception as e:
    logger.error(f"Checkpoint operation failed: {e}")
    # Continue without checkpoint or implement fallback
```

### **3. Memory Management**
- **Large DataFrames**: Save only essential columns
- **Models**: Consider saving only model artifacts, not full training state
- **Cleanup**: Delete old checkpoints when no longer needed

### **4. Reproducibility**
- **Config Hashing**: Automatic config versioning
- **Timestamps**: All checkpoints have creation timestamps
- **Metadata**: Complete traceability of data and parameters

## Troubleshooting

### **Common Issues**

#### **1. Corrupted Checkpoint Files**
```python
# The system automatically detects and deletes corrupted pickle files
# Check logs for: "⚠️ Corrupted pickle checkpoint file"
```

#### **2. Missing PyTables**
```python
# System falls back to pickle if PyTables not available
# Warning: "PyTables not available, using pickle for checkpoint"
```

#### **3. Invalid Data Validation**
```python
# Validation errors prevent saving invalid checkpoints
# Error: "❌ Invalid checkpoint for step: no trained models"
```

### **Debug Mode**
```python
# Enable detailed logging
import logging
logging.getLogger('src.training.steps.labeling.layer3_checkpoint_manager').setLevel(logging.DEBUG)

# Check checkpoint directory
manager = get_checkpoint_manager('layer3', 'ETHUSDT')
print(f"Checkpoint directory: {manager.checkpoint_dir}")
```

This checkpoint system provides robust, production-ready checkpointing for all layers with automatic resume capabilities and comprehensive error handling.
