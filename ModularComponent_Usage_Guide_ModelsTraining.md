# ModularComponent Usage Guide - Models Training

## Overview

The `ModularComponent` abstract class has been fully implemented with comprehensive functionality for creating modular, reusable components in the **models training** pipeline. This guide provides detailed usage instructions and examples specifically tailored for machine learning workflows.

## Fully Implemented Abstract Methods

### 1. `initialize() -> bool`
**Purpose**: Initialize the component and its resources.

**Implementation**: 
- Validates configuration using `validate_config()`
- Calls `_initialize_resources()` for component-specific setup
- Sets initialization flag and logs success/failure
- Returns `True` if successful, `False` otherwise

**Override**: Implement `_initialize_resources()` in subclasses for custom initialization logic.

### 2. `process(data: Any, **kwargs) -> Any`
**Purpose**: Process input data with comprehensive error handling.

**Implementation**:
- Checks if component is initialized
- Validates input data using `validate_input()`
- Verifies component can process data using `can_process()`
- Calls `_process_data()` for actual processing
- Handles errors gracefully with proper logging

**Override**: Implement `_process_data()` in subclasses for custom processing logic.

### 3. `validate_input(data: Any) -> ValidationResult`
**Purpose**: Comprehensive input validation with detailed results.

**Implementation**:
- Handles multiple data types (DataFrame, Series, ndarray, list, tuple, dict)
- Uses configurable validation rules from `_get_validation_rules()`
- Performs type-specific validation
- Includes component-specific validation via `_validate_component_specific()`
- Returns detailed ValidationResult with errors, warnings, and metadata

**Override**: Implement `_get_validation_rules()` and `_validate_component_specific()` in subclasses.

### 4. `cleanup() -> None`
**Purpose**: Cleanup resources and reset component state.

**Implementation**:
- Calls `_cleanup_resources()` for component-specific cleanup
- Clears component state
- Resets performance statistics
- Resets initialization flag
- Logs cleanup completion

**Override**: Implement `_cleanup_resources()` in subclasses for custom cleanup logic.

### 5. `get_component_info() -> Dict[str, Any]`
**Purpose**: Get comprehensive component metadata.

**Implementation**:
- Returns component name, type, version, description
- Includes initialization status and configuration
- Lists dependencies and capabilities
- Provides complete component information

**Override**: Override to add component-specific information.

### 6. `get_dependencies() -> List[str]`
**Purpose**: Get list of required dependencies.

**Implementation**:
- Returns default dependencies: `['torch', 'sklearn', 'pandas', 'numpy']`
- Can be overridden for component-specific dependencies

**Override**: Override to specify actual dependencies.

### 7. `get_output_schema() -> Dict[str, Any]`
**Purpose**: Get expected output schema.

**Implementation**:
- Returns generic schema with type, description, and metadata
- Can be overridden for specific output formats

**Override**: Override to specify actual output schema.

### 8. `get_required_config() -> List[str]`
**Purpose**: Get required configuration parameters.

**Implementation**:
- Returns empty list by default
- Can be overridden for specific configuration requirements

**Override**: Override to specify required configuration keys.

### 9. `can_process(data: Any) -> bool`
**Purpose**: Check if component can process given data.

**Implementation**:
- Validates data is not None
- Checks component is initialized
- Verifies data type compatibility
- Checks memory requirements
- Returns `True` if all checks pass

**Override**: Override for custom processing capability checks.

### 10. `get_processing_capabilities() -> Dict[str, Any]`
**Purpose**: Get component processing capabilities.

**Implementation**:
- Returns supported input/output types
- Indicates parallel processing support
- Specifies memory efficiency
- Lists processing features

**Override**: Override to specify actual capabilities.

### 11. `estimate_processing_time(data: Any) -> float`
**Purpose**: Estimate processing time for given data.

**Implementation**:
- Uses base processing time from config
- Calculates size-based factor
- Applies complexity factor
- Uses performance multiplier
- Returns estimated time in seconds

**Override**: Override for more accurate time estimation.

### 12. `get_memory_requirements(data: Any) -> Dict[str, Any]`
**Purpose**: Get memory requirements for processing data.

**Implementation**:
- Calculates base memory usage
- Handles pandas, numpy, and generic objects
- Applies overhead factor
- Returns estimated and peak memory usage

**Override**: Override for more accurate memory estimation.

## Helper Methods

### Concrete Methods (Available to all subclasses)

- **Configuration Management**: `get_config()`, `update_config()`, `validate_config()`
- **State Management**: `set_state()`, `get_state()`, `clear_state()`, `get_all_state()`
- **Performance Monitoring**: `get_performance_stats()`, `reset_stats()`
- **Lifecycle Management**: `is_initialized()`, `get_status()`
- **Serialization**: `serialize()`, `deserialize()`
- **Safe Processing**: `_safe_process()` - Wraps processing with error handling

### Abstract Helper Methods (Must be overridden)

- `_initialize_resources() -> bool` - Initialize component-specific resources
- `_cleanup_resources() -> None` - Cleanup component-specific resources
- `_process_data(data: Any, **kwargs) -> Any` - Process data with component logic
- `_get_validation_rules() -> Dict[str, Any]` - Get validation rules
- `_validate_component_specific(data: Any) -> Dict[str, Any]` - Component-specific validation

## Example Usage for Models Training

### 1. Basic Model Trainer Component

```python
from src.training.steps.models_training.unified_data_driven_pipeline.core.modular_architecture import (
    ModularComponent, create_modular_component
)

class ModelTrainer(ModularComponent):
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(name, config, logger)
        self.model_type = self.get_config('model_type', 'neural_network')
        self.training_config = self.get_config('training', {})
        self.version = "1.0.0"
        self.description = "Machine Learning Model Trainer"
    
    def _initialize_resources(self) -> bool:
        """Initialize component-specific resources."""
        try:
            self.set_state('initialized_at', time.time())
            self.set_state('training_epoch', 0)
            self.set_state('best_accuracy', 0.0)
            self.set_state('model_weights', None)
            return True
        except Exception as e:
            self.logger.error(f"Resource initialization failed: {e}")
            return False
    
    def _cleanup_resources(self) -> None:
        """Cleanup component-specific resources."""
        self.set_state('cleaned_up_at', time.time())
        self.set_state('model_weights', None)
    
    def _process_data(self, data: Any, **kwargs) -> Any:
        """Process data with component logic."""
        # Get training data
        X_train = data.get('X_train')
        y_train = data.get('y_train')
        X_val = data.get('X_val')
        y_val = data.get('y_val')
        
        # Initialize model
        model = self._create_model()
        
        # Training loop
        epochs = self.training_config.get('epochs', 100)
        for epoch in range(epochs):
            # Train one epoch
            train_loss = self._train_epoch(model, X_train, y_train)
            
            # Validate
            val_accuracy = self._validate_epoch(model, X_val, y_val)
            
            # Update state
            self.set_state('training_epoch', epoch + 1)
            self.set_state('current_accuracy', val_accuracy)
            
            # Save best model
            if val_accuracy > self.get_state('best_accuracy', 0.0):
                self.set_state('best_accuracy', val_accuracy)
                self.set_state('model_weights', model.state_dict())
            
            # Log progress
            self.logger.info(f"Epoch {epoch + 1}/{epochs}: Loss={train_loss:.4f}, Accuracy={val_accuracy:.4f}")
        
        # Return trained model
        return {
            'model': model,
            'weights': self.get_state('model_weights'),
            'accuracy': self.get_state('best_accuracy'),
            'epochs_trained': self.get_state('training_epoch')
        }
    
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for this component."""
        return {
            'min_samples': 100,
            'max_samples': 1000000,
            'required_keys': ['X_train', 'y_train'],
            'data_types': ['dict'],
            'X_train_shape': (None, None),
            'y_train_shape': (None,)
        }
    
    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        """Validate data with component-specific rules."""
        errors = []
        warnings = []
        metadata = {}
        
        if isinstance(data, dict):
            # Check required keys
            required_keys = ['X_train', 'y_train']
            for key in required_keys:
                if key not in data:
                    errors.append(f"Missing required key: {key}")
            
            # Check data shapes
            if 'X_train' in data and 'y_train' in data:
                X_train = data['X_train']
                y_train = data['y_train']
                
                if hasattr(X_train, 'shape') and hasattr(y_train, 'shape'):
                    metadata['X_train_shape'] = X_train.shape
                    metadata['y_train_shape'] = y_train.shape
                    
                    if len(X_train) != len(y_train):
                        errors.append("X_train and y_train must have same number of samples")
                    
                    if len(X_train) < 100:
                        warnings.append("Training data is small, consider more data")
        
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}
    
    def _create_model(self):
        """Create model based on configuration."""
        # Implement model creation logic
        pass
    
    def _train_epoch(self, model, X_train, y_train):
        """Train one epoch."""
        # Implement training logic
        pass
    
    def _validate_epoch(self, model, X_val, y_val):
        """Validate one epoch."""
        # Implement validation logic
        pass

# Usage
config = {
    'model_type': 'neural_network',
    'training': {
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001
    }
}

component = ModelTrainer('model_trainer', config)

# Initialize
if component.initialize():
    # Process data
    training_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val
    }
    result = component.process(training_data)
    # Cleanup
    component.cleanup()
```

### 2. Ensemble Training Component

```python
class EnsembleTrainer(ModularComponent):
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(name, config, logger)
        self.ensemble_config = self.get_config('ensemble', {})
        self.models = []
    
    def _initialize_resources(self) -> bool:
        """Initialize ensemble training resources."""
        try:
            self.set_state('initialized_at', time.time())
            self.set_state('ensemble_models', [])
            self.set_state('ensemble_weights', [])
            return True
        except Exception as e:
            self.logger.error(f"Ensemble initialization failed: {e}")
            return False
    
    def _process_data(self, data: Any, **kwargs) -> Any:
        """Process data with ensemble training logic."""
        # Get training data
        X_train = data.get('X_train')
        y_train = data.get('y_train')
        
        # Train multiple models
        models = []
        for i, model_config in enumerate(self.ensemble_config.get('models', [])):
            model = self._train_single_model(X_train, y_train, model_config)
            models.append(model)
            self.set_state('ensemble_models', models)
        
        # Calculate ensemble weights
        weights = self._calculate_ensemble_weights(models, X_train, y_train)
        self.set_state('ensemble_weights', weights)
        
        return {
            'models': models,
            'weights': weights,
            'ensemble_size': len(models)
        }
    
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for ensemble training."""
        return {
            'min_samples': 1000,
            'required_keys': ['X_train', 'y_train'],
            'data_types': ['dict'],
            'min_models': 2
        }
    
    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        """Validate data with ensemble-specific rules."""
        errors = []
        warnings = []
        metadata = {}
        
        if isinstance(data, dict):
            if 'X_train' in data and 'y_train' in data:
                X_train = data['X_train']
                y_train = data['y_train']
                
                if len(X_train) < 1000:
                    warnings.append("Ensemble training requires more data for stability")
                
                metadata['data_size'] = len(X_train)
        
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}
    
    def _train_single_model(self, X_train, y_train, model_config):
        """Train a single model in the ensemble."""
        # Implement single model training
        pass
    
    def _calculate_ensemble_weights(self, models, X_train, y_train):
        """Calculate ensemble weights."""
        # Implement weight calculation
        pass
```

### 3. ML Entry Timing Labeler Component

```python
class MLEntryTimingLabeler(ModularComponent):
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(name, config, logger)
        self.labeling_config = self.get_config('labeling', {})
        self.model = None
    
    def _initialize_resources(self) -> bool:
        """Initialize ML entry timing resources."""
        try:
            self.set_state('initialized_at', time.time())
            self.set_state('labeling_count', 0)
            self.set_state('model_loaded', False)
            
            # Load pre-trained model
            self.model = self._load_model()
            self.set_state('model_loaded', True)
            return True
        except Exception as e:
            self.logger.error(f"ML entry timing initialization failed: {e}")
            return False
    
    def _process_data(self, data: Any, **kwargs) -> Any:
        """Process data with ML entry timing logic."""
        # Get market data
        market_data = data.get('market_data')
        features = data.get('features')
        
        # Generate labels using ML model
        labels = self._generate_labels(market_data, features)
        
        # Update state
        count = self.get_state('labeling_count', 0)
        self.set_state('labeling_count', count + 1)
        
        return {
            'labels': labels,
            'confidence_scores': self._get_confidence_scores(market_data, features),
            'labeling_metadata': self._get_labeling_metadata()
        }
    
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for ML entry timing."""
        return {
            'min_samples': 100,
            'required_keys': ['market_data', 'features'],
            'data_types': ['dict'],
            'feature_columns': ['price', 'volume', 'technical_indicators']
        }
    
    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        """Validate data with ML entry timing specific rules."""
        errors = []
        warnings = []
        metadata = {}
        
        if isinstance(data, dict):
            if 'market_data' in data and 'features' in data:
                market_data = data['market_data']
                features = data['features']
                
                if len(market_data) < 100:
                    warnings.append("Insufficient market data for reliable labeling")
                
                metadata['market_data_size'] = len(market_data)
                metadata['features_size'] = len(features)
        
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}
    
    def _load_model(self):
        """Load pre-trained ML model."""
        # Implement model loading
        pass
    
    def _generate_labels(self, market_data, features):
        """Generate entry timing labels."""
        # Implement label generation
        pass
    
    def _get_confidence_scores(self, market_data, features):
        """Get confidence scores for labels."""
        # Implement confidence scoring
        pass
    
    def _get_labeling_metadata(self):
        """Get labeling metadata."""
        # Implement metadata collection
        pass
```

## Key Features for Models Training

1. **Comprehensive Error Handling**: All methods include proper error handling and logging
2. **Performance Monitoring**: Automatic performance statistics collection
3. **State Management**: Built-in state management for training data and model state
4. **Configuration Management**: Flexible configuration system for training parameters
5. **Validation Framework**: Comprehensive input validation for training data
6. **Serialization Support**: Built-in serialization for model persistence
7. **Memory Management**: Memory requirement estimation and checking
8. **Lifecycle Management**: Proper initialization and cleanup
9. **Extensibility**: Easy to extend with custom functionality
10. **Documentation**: Comprehensive docstrings and examples
11. **ML-Specific**: Optimized for machine learning workflows
12. **Model State Tracking**: Built-in support for model weights and training state

## Best Practices for Models Training

1. **Always call `initialize()`** before using the component
2. **Implement all abstract helper methods** for proper functionality
3. **Use `_safe_process()`** for automatic error handling and performance tracking
4. **Override validation methods** for component-specific validation
5. **Call `cleanup()`** when done with the component
6. **Use state management** for storing model weights and training state
7. **Implement proper error handling** in custom methods
8. **Provide accurate capability information** for better integration
9. **Track training progress** using state management
10. **Implement model checkpointing** for persistence
11. **Use performance monitoring** for training optimization
12. **Validate training data** thoroughly before processing