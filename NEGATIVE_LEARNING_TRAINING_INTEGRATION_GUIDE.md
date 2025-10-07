# Negative Learning Training Integration Guide

## 🎯 Current Status: **PARTIALLY WIRED**

The negative learning plugin is **partially wired** with ML training. Here's what's implemented and what needs to be done:

## ✅ **What's Already Implemented**

### 1. Core Plugin Components
- ✅ Failure context detection
- ✅ Negative learning feature generation
- ✅ Model constraints and sample weights
- ✅ Validation framework
- ✅ Pipeline integration classes

### 2. Training Integration Infrastructure
- ✅ `negative_learning_training_integration.py` - Integration wrapper
- ✅ `negative_learning_training_patches.py` - Function patching system
- ✅ Automatic feature enhancement
- ✅ Constraint application
- ✅ Sample weight integration

## ⚠️ **What Needs to be Done**

### 1. Initialize Negative Learning in Training Pipeline

You need to add initialization calls to your existing training pipeline:

```python
# In your main training pipeline (e.g., analyst_training_pipeline.py)
from src.training.steps.models_training.negative_learning_training_integration import (
    initialize_negative_learning_integration
)

# Initialize once per retrain cycle
def initialize_training_pipeline():
    # ... existing initialization code ...
    
    # Initialize negative learning
    negative_learning_config = {
        'analyst': {
            'max_negative_features': 8,
            'enable_gated_twins': True,
            'enable_exception_interactions': True
        },
        'tactician': {
            'max_negative_features': 6,
            'enable_gated_twins': True,
            'enable_exception_interactions': True
        }
    }
    
    nl_integration = initialize_negative_learning_integration(negative_learning_config)
    
    # Initialize with your training data
    init_results = nl_integration.initialize_for_training(
        analyst_features=analyst_features,
        analyst_target=analyst_target,
        tactician_features=tactician_features,
        tactician_target=tactician_target,
        analyst_outputs=analyst_outputs,
        retrain_timestamp=datetime.now()
    )
    
    return nl_integration
```

### 2. Apply Patches to Existing Training Functions

Add this to your training module imports:

```python
# At the top of your training modules
from src.training.steps.models_training.negative_learning_training_patches import (
    apply_negative_learning_patches
)

# Apply patches (this will automatically patch existing functions)
apply_negative_learning_patches()
```

### 3. Update Model Training Calls

The patches will automatically enhance your existing training calls, but you can also manually enhance them:

```python
# For Analyst training
from src.training.steps.models_training.negative_learning_training_patches import (
    enhance_analyst_training_data
)

# Enhance training data before calling your training function
enhanced_data, enhanced_columns, enhanced_weights, constraints = enhance_analyst_training_data(
    training_data, feature_columns, target_columns, sample_weight
)

# Add constraints to your model parameters
model_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'monotone_constraints': constraints.get('monotone_constraints'),
    # ... other parameters
}

# Use enhanced data for training
result = await your_analyst_training_function(
    enhanced_data, enhanced_columns, target_columns, enhanced_weights, **model_params
)
```

### 4. Update Tactician Training

```python
# For Tactician training
from src.training.steps.models_training.negative_learning_training_patches import (
    enhance_tactician_training_data
)

# Enhance training data
enhanced_data, enhanced_columns, enhanced_weights, constraints = enhance_tactician_training_data(
    training_data, feature_columns, target_columns, sample_weight, analyst_outputs
)

# Use enhanced data for training
result = await your_tactician_training_function(
    enhanced_data, enhanced_columns, target_columns, enhanced_weights,
    analyst_outputs=analyst_outputs, **constraints
)
```

## 🔧 **Step-by-Step Integration**

### Step 1: Add to Your Main Training Pipeline

```python
# In your main training pipeline file
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.training.steps.models_training.negative_learning_training_integration import (
    initialize_negative_learning_integration
)

class YourTrainingPipeline:
    def __init__(self, config):
        # ... existing initialization ...
        
        # Initialize negative learning
        self.nl_integration = initialize_negative_learning_integration(
            config.get('negative_learning', {})
        )
    
    async def train_models(self, analyst_data, tactician_data, analyst_outputs):
        # Initialize negative learning for this retrain cycle
        init_results = self.nl_integration.initialize_for_training(
            analyst_features=analyst_data['features'],
            analyst_target=analyst_data['target'],
            tactician_features=tactician_data['features'],
            tactician_target=tactician_data['target'],
            analyst_outputs=analyst_outputs
        )
        
        # Your existing training code will now automatically use enhanced features
        # due to the patches applied
```

### Step 2: Apply Patches in Training Modules

```python
# At the top of analyst_models_training.py
from src.training.steps.models_training.negative_learning_training_patches import (
    apply_negative_learning_patches
)

# Apply patches to existing functions
apply_negative_learning_patches()

# Your existing functions are now automatically enhanced
```

### Step 3: Update Model Parameters

```python
# In your model training code
def get_enhanced_model_params(pipeline_type='analyst'):
    """Get model parameters with negative learning constraints"""
    integration = get_negative_learning_integration()
    if integration is None:
        return {}
    
    constraints = integration.get_training_constraints(pipeline_type, 'lightgbm')
    
    base_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'max_depth': 4,
        'num_leaves': 16,
        'lambda_l2': 40,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.85,
        'learning_rate': 0.05
    }
    
    # Add negative learning constraints
    if constraints.get('monotone_constraints'):
        base_params['monotone_constraints'] = constraints['monotone_constraints']
    
    return base_params
```

### Step 4: Add Validation

```python
# Add validation after training
def validate_negative_learning_performance(pipeline_type='analyst'):
    """Validate negative learning performance"""
    integration = get_negative_learning_integration()
    if integration is None:
        return {}
    
    # Get validation results
    validation_results = integration.validate_training_performance(
        features_df, target, pipeline_type, analyst_outputs
    )
    
    return validation_results
```

## 🚀 **Quick Integration (Minimal Changes)**

If you want the quickest integration with minimal code changes:

### 1. Add One Import

```python
# At the top of your main training file
from src.training.steps.models_training.negative_learning_training_patches import (
    apply_negative_learning_patches
)

# Apply patches (this does everything automatically)
apply_negative_learning_patches()
```

### 2. Initialize Once Per Retrain

```python
# In your retrain function
from src.training.steps.models_training.negative_learning_training_integration import (
    initialize_negative_learning_integration
)

def retrain_models():
    # ... existing retrain code ...
    
    # Initialize negative learning
    nl_integration = initialize_negative_learning_integration()
    nl_integration.initialize_for_training(
        analyst_features, analyst_target,
        tactician_features, tactician_target,
        analyst_outputs
    )
    
    # Your existing training calls will now automatically use enhanced features
```

## 📊 **Expected Results**

After integration, you should see:

1. **Automatic Feature Enhancement**: Your training data will automatically include negative learning features
2. **Model Constraints**: Monotone constraints will be applied to tree models
3. **Sample Weights**: Uncertainty-based sample weights will be applied
4. **Performance Improvement**: Better performance in challenging market conditions
5. **Validation Reports**: Comprehensive validation of negative learning effectiveness

## 🔍 **Verification**

To verify the integration is working:

```python
# Check integration status
from src.training.steps.models_training.negative_learning_training_integration import (
    get_negative_learning_integration
)

integration = get_negative_learning_integration()
if integration:
    status = integration.get_integration_status()
    print(f"Negative learning initialized: {status['is_initialized']}")
    print(f"Analyst features: {status['analyst_negative_features']}")
    print(f"Tactician features: {status['tactician_negative_features']}")
```

## ⚠️ **Important Notes**

1. **Backward Compatibility**: All existing code will continue to work unchanged
2. **Performance**: Minimal overhead, features are cached
3. **Memory**: Efficient feature generation with garbage collection
4. **Time-Series Safety**: All features are built OOF and as-of joined
5. **Latency**: Estimated +30ms impact, within budget

## 🎯 **Next Steps**

1. **Choose Integration Method**: Quick (patches) or Manual (enhanced control)
2. **Add Initialization**: Call `initialize_negative_learning_integration()` once per retrain
3. **Apply Patches**: Add `apply_negative_learning_patches()` to your training modules
4. **Test**: Run your training pipeline and verify enhanced features are generated
5. **Monitor**: Check validation results and performance improvements

The negative learning plugin is ready to use - it just needs to be wired into your existing training pipeline with these integration steps!