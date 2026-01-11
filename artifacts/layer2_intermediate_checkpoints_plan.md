# Layer 2 Intermediate Checkpoints Plan

## 🎯 **Objective**
Add more granular checkpoints between feature_engineering (step 6) and geometry_optimization (step 7) to enable better resumption capabilities and debugging.

## 📊 **Current Pipeline Structure**
```
6: feature_engineering    ✅ (Apply causal denoising)
7: geometry_optimization  ✅ (De Prado protocol)
8: final_processing       ✅ (OOF analytics, reports)
```

## 🔍 **Gap Analysis**
Looking at the Layer 2 pipeline code between steps 6-7, I can see several intermediate operations:

### Current Flow (from label_based_layer_2.py lines 4700-4750):
```python
# 6. feature_engineering (current checkpoint)
engineered_df, causal_metadata = self._apply_causal_feature_engineering(enriched_df, augmented_graph)

# GAP: No checkpoint here
# 5. Causal Targets: Compute treatment effects and causal residuals
causal_targets_df = self._compute_causal_targets(engineered_df, causal_events_df, specialist_predictions)

# GAP: No checkpoint here  
# 6. IRM Training: Train base models with invariance penalty
causal_geometries, causal_selected_features = self._train_causal_models(...)

# 7. geometry_optimization (current checkpoint)
```

## 🚀 **Proposed Intermediate Checkpoints**

### **New Checkpoint 6A: causal_targets**
- **Position**: After `_compute_causal_targets()` 
- **Index**: 6.5 (between 6 and 7)
- **Data**: causal_targets_df, engineered_df, causal_events_df
- **Purpose**: Resume after causal target computation

### **New Checkpoint 6B: causal_model_training** 
- **Position**: After `_train_causal_models()`
- **Index**: 6.8 (between 6.5 and 7)
- **Data**: causal_geometries, causal_selected_features, causal_targets_df
- **Purpose**: Resume after causal model training but before geometry optimization

## 📝 **Updated Sub-step List**

```python
LAYER2_SUBSTEPS = [
    'data_loading',           # 0: Load market data, dollar bars
    'regime_generation',      # 1: Generate regimes via AdaptiveHunterRouter  
    'causal_initialization',  # 2: Initialize components, precompute features
    'causal_discovery',       # 3: Build causal DAG
    'specialist_training',    # 4: Train AEDL/traditional specialists
    'event_generation',       # 5: Generate causal surprise events
    'feature_engineering',    # 6: Apply causal denoising
    'causal_targets',         # 6.5: Compute causal targets ⭐ NEW
    'causal_model_training',  # 6.8: Train causal models ⭐ NEW
    'geometry_optimization',  # 7: De Prado protocol
    'final_processing',       # 8: OOF analytics, reports
]
```

## 🔧 **Implementation Steps**

### 1. **Update Checkpoint Manager**
```python
# In layer2_checkpoint_manager.py
LAYER2_SUBSTEPS = [
    'data_loading',           # 0
    'regime_generation',      # 1  
    'causal_initialization',  # 2
    'causal_discovery',       # 3
    'specialist_training',    # 4
    'event_generation',       # 5
    'feature_engineering',    # 6
    'causal_targets',         # 6.5  ⭐ NEW
    'causal_model_training',  # 6.8  ⭐ NEW
    'geometry_optimization',  # 7
    'final_processing',       # 8
]
```

### 2. **Add causal_targets Checkpoint**
```python
# In label_based_layer_2.py after line 4728
# Save causal_targets checkpoint
if self._checkpoints_enabled:
    self._checkpoint_manager.save_checkpoint('causal_targets', {
        'df': df,
        'engineered_df': engineered_df,
        'causal_events_df': causal_events_df,
        'causal_targets_df': causal_targets_df,
        'causal_graph': causal_graph,
        'augmented_graph': augmented_graph,
        'causal_metadata': causal_metadata,
        'specialist_predictions': specialist_predictions
    }, symbol, self._current_config)
```

### 3. **Add causal_model_training Checkpoint**
```python
# In label_based_layer_2.py after line 4740
# Save causal_model_training checkpoint  
if self._checkpoints_enabled:
    self._checkpoint_manager.save_checkpoint('causal_model_training', {
        'df': df,
        'engineered_df': engineered_df,
        'causal_events_df': causal_events_df,
        'causal_targets_df': causal_targets_df,
        'causal_geometries': [asdict(g) for g in causal_geometries],
        'causal_selected_features': causal_selected_features,
        'causal_graph': causal_graph,
        'augmented_graph': augmented_graph,
        'causal_metadata': causal_metadata,
        'specialist_predictions': specialist_predictions
    }, symbol, self._current_config)
```

### 4. **Update Resume Logic**
```python
# In _run_causal_denoising_pipeline() method
# Add resume logic for new checkpoints
if resume_from == 'causal_targets':
    # Load from causal_targets checkpoint
    # Skip to causal model training
elif resume_from == 'causal_model_training':
    # Load from causal_model_training checkpoint  
    # Skip to geometry optimization
```

## 🎯 **Benefits**

### **1. Faster Resumption**
- **Before**: Must redo causal targets + model training (expensive)
- **After**: Can resume from specific intermediate step

### **2. Better Debugging**
- **Isolate Issues**: Test causal targets separately from model training
- **Incremental Testing**: Validate each stage independently

### **3. Development Efficiency**
- **Faster Iteration**: Test model training changes without recomputing targets
- **Parameter Tuning**: Adjust model training parameters independently

### **4. Resource Savings**
- **Time**: Skip expensive recomputations
- **Memory**: Load only necessary intermediate data

## 📊 **Checkpoint Data Sizes**

### **causal_targets (6.5)**
- engineered_df: ~736 features × 132K rows (~100MB)
- causal_targets_df: ~10-20 target columns × 132K rows (~10MB)
- causal_events_df: sparse events (~5MB)
- **Total**: ~115MB

### **causal_model_training (6.8)**
- causal_geometries: ~50-100 geometry objects (~5MB)
- causal_selected_features: feature lists (~1MB)
- **Plus all previous data**: ~120MB total

## 🚀 **Usage Examples**

### **Resume from causal_targets**
```bash
python3 src/launcher/ares_launcher.py meta_labeling_hpo_sample_weighted \
  --symbol ETHUSDT --execution-mode full \
  --layer2-resume-from causal_targets
```

### **Resume from causal_model_training**
```bash
python3 src/launcher/ares_launcher.py meta_labeling_hpo_sample_weighted \
  --symbol ETHUSDT --execution-mode full \
  --layer2-resume-from causal_model_training
```

## ⚠️ **Considerations**

### **1. Storage Impact**
- **Additional**: ~235MB per symbol
- **Acceptable**: Given development efficiency gains

### **2. Backward Compatibility**
- **Old checkpoints**: Still work (missing new steps)
- **New logic**: Graceful fallback for missing intermediate checkpoints

### **3. Index Management**
- **Non-integer indices**: 6.5, 6.8 need special handling
- **Sorting**: Ensure proper order in resume logic

## ✅ **Implementation Priority**

1. **High Priority**: causal_targets checkpoint (saves expensive target computation)
2. **Medium Priority**: causal_model_training checkpoint (saves model training)
3. **Low Priority**: Update documentation and help text

## 🎯 **Success Criteria**

- [ ] Checkpoint manager updated with new steps
- [ ] causal_targets checkpoint implemented
- [ ] causal_model_training checkpoint implemented  
- [ ] Resume logic handles new checkpoints
- [ ] Testing confirms resumption works correctly
- [ ] No regression in existing checkpoint functionality

This plan provides granular resumption capabilities while maintaining backward compatibility and minimizing storage overhead.
