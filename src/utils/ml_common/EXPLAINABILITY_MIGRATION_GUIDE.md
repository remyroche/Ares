# Model Explainability Migration Guide

## Overview

This guide helps you migrate from the old component-specific explainability system (`src/explainability/`) to the new model-focused explainability system integrated with ML commons (`src/utils/ml_common/model_explainability.py`).

## Key Changes

### Old System (Component-Specific)
- **Location**: `src/explainability/`
- **Approach**: Separate explainers for each trading component
- **Classes**: `TacticianExplainer`, `AnalystExplainer`, `SRExplainer`, `HMMExplainer`
- **Integration**: Manual integration required
- **Usage**: Component-specific explanation formats

### New System (Model-Focused)
- **Location**: `src/utils/ml_common/model_explainability.py`
- **Approach**: Unified explainer for all ML models
- **Classes**: `ModelExplainabilityManager`, `ModelExplanationResult`
- **Integration**: Automatic integration with ML commons
- **Usage**: Model-specific explanations with unified format

## Migration Steps

### 1. Update Imports

**Old:**
```python
from src.explainability import (
    TacticianExplainer,
    AnalystExplainer,
    SRExplainer,
    HMMExplainer,
    ExplainabilityOrchestrator
)
```

**New:**
```python
from src.utils.ml_common import (
    ModelExplainabilityManager,
    ModelExplanationResult,
    explain_model_quick
)
```

### 2. Replace Component-Specific Explainers

**Old:**
```python
# Component-specific approach
tactician_explainer = TacticianExplainer(config)
analyst_explainer = AnalystExplainer(config)
sr_explainer = SRExplainer(config)
hmm_explainer = HMMExplainer(config)

# Manual explanation generation
tactician_explanation = tactician_explainer.explain_prediction(model, data)
```

**New:**
```python
# Model-focused approach
explainability_manager = ModelExplainabilityManager(config)

# Automatic explanation generation
explanation = explainability_manager.explain_model(
    model=model,
    X_train=X_train,
    X_test=X_test,
    model_id="my_model",
    model_type="RandomForestClassifier",
    feature_names=feature_names
)
```

### 3. Update Model Training Integration

**Old:**
```python
# Manual explainability integration
def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    
    # Manual explanation generation
    explainer = TacticianExplainer(config)
    explanation = explainer.explain_model(model, X_train, X_test)
    
    return {
        'model': model,
        'explanation': explanation
    }
```

**New:**
```python
# Automatic explainability integration
from src.utils.ml_common import EnhancedModelTrainer

def train_model(model, X_train, y_train, X_test, y_test):
    trainer = EnhancedModelTrainer({
        'enable_model_explanations': True,
        'explainability': {
            'enable_auto_explanations': True
        }
    })
    
    # Automatic explanation generation during training
    results = trainer.train_and_evaluate_model(
        model=model,
        model_name="my_model",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    
    return results  # Includes 'model_explanations' automatically
```

### 4. Update Model Registry Integration

**Old:**
```python
# Manual explanation saving
registry = ModelRegistry()
registry.save_model_with_metadata(model, metadata)

# Manual explanation loading
explanation = load_explanation_separately(model_id)
```

**New:**
```python
# Automatic explanation integration
from src.utils.ml_common import ModelRegistry, ModelExplainabilityManager

registry = ModelRegistry()
explainability_manager = ModelExplainabilityManager(model_registry=registry)

# Save model with automatic explanation
registry.save_model_with_metadata(model, metadata)

# Load model with automatic explanation
result = registry.load_model_with_validation(model_id)
if 'explanation' in result:
    explanation = result['explanation']
```

### 5. Update Explanation Usage

**Old:**
```python
# Component-specific explanation access
tactician_explanation = decision_trace.tactician_explanation
analyst_explanation = decision_trace.analyst_explanation

# Different formats for different components
tactician_shap = tactician_explanation.shap_values
analyst_lime = analyst_explanation.lime_explanation
```

**New:**
```python
# Unified explanation access
explanation = model_result['model_explanations']

# Unified format for all models
shap_values = explanation.get('shap_values')
lime_explanation = explanation.get('lime_explanation')
feature_importance = explanation.get('feature_importance')
confidence = explanation.get('explanation_confidence')
```

## Configuration Changes

### Old Configuration
```python
config = {
    'explainability': {
        'tactician': {
            'enable_shap': True,
            'enable_lime': True
        },
        'analyst': {
            'enable_shap': True,
            'enable_lime': False
        },
        'sr': {
            'enable_shap': False,
            'enable_lime': True
        },
        'hmm': {
            'enable_shap': True,
            'enable_lime': True
        }
    }
}
```

### New Configuration
```python
config = {
    'explainability': {
        'enable_auto_explanations': True,
        'enable_explanation_caching': True,
        'auto_explain_on_training': True,
        'auto_explain_on_prediction': False,
        'explanations': {
            'enable_shap': True,
            'enable_lime': True,
            'shap_sample_size': 100,
            'lime_sample_size': 10
        }
    }
}
```

## Benefits of Migration

### 1. Simplified Architecture
- **Before**: 4+ separate explainer classes
- **After**: 1 unified explainability manager

### 2. Automatic Integration
- **Before**: Manual integration required for each component
- **After**: Automatic integration with ML commons training

### 3. Model-Focused Approach
- **Before**: Component-specific explanations
- **After**: Model-specific explanations (RandomForest, Neural Network, etc.)

### 4. Unified Format
- **Before**: Different explanation formats per component
- **After**: Consistent `ModelExplanationResult` format

### 5. Better Caching
- **Before**: No built-in caching
- **After**: Automatic explanation caching and retrieval

### 6. Registry Integration
- **Before**: Manual explanation persistence
- **After**: Automatic explanation saving/loading with models

## Quick Migration Checklist

- [ ] Update imports to use `src.utils.ml_common`
- [ ] Replace component-specific explainers with `ModelExplainabilityManager`
- [ ] Update model training to use `EnhancedModelTrainer` with explainability
- [ ] Update model registry integration
- [ ] Update explanation access patterns
- [ ] Update configuration format
- [ ] Test explanation generation and retrieval
- [ ] Remove old explainability imports and usage

## Example Migration

See `explainability_integration_example.py` for a complete working example of the new system.

## Support

If you encounter issues during migration, check:
1. Import paths are correct
2. Configuration format matches new structure
3. Model training uses `EnhancedModelTrainer`
4. Model registry integration is properly configured

The new system is designed to be backward-compatible where possible, but some manual updates are required for optimal integration.