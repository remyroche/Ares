# ModularComponent Implementation Guide

## Overview

The `ModularComponent` abstract class has been fully implemented with comprehensive functionality for creating modular, reusable components in the unified data-driven pipeline.

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
- Handles multiple data types (DataFrame, Series, ndarray, list, tuple)
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
- Returns default dependencies: `['pandas', 'numpy']`
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

## Example Usage

```python
from src.training.steps.pre_training.unified_data_driven_pipeline.core.modular_architecture import (
    ModularComponent, create_modular_component
)

class MyCustomComponent(ModularComponent):
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(name, config, logger)
        self.my_param = self.get_config('my_param', 'default_value')
    
    def _initialize_resources(self) -> bool:
        # Initialize your resources here
        self.set_state('initialized_at', time.time())
        return True
    
    def _cleanup_resources(self) -> None:
        # Cleanup your resources here
        self.set_state('cleaned_up_at', time.time())
    
    def _process_data(self, data: Any, **kwargs) -> Any:
        # Implement your processing logic here
        return processed_data
    
    def _get_validation_rules(self) -> Dict[str, Any]:
        return {
            'min_size': 10,
            'max_size': 1000000,
            'required_attributes': ['required_column'],
            'data_types': ['pandas.DataFrame']
        }
    
    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        errors = []
        warnings = []
        metadata = {}
        
        # Add your validation logic here
        if isinstance(data, pd.DataFrame):
            if 'required_column' not in data.columns:
                errors.append("Missing required column")
        
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}

# Usage
config = {'my_param': 'custom_value'}
component = MyCustomComponent('my_component', config)

# Initialize
if component.initialize():
    # Process data
    result = component.process(data)
    # Cleanup
    component.cleanup()
```

## Key Features

1. **Comprehensive Error Handling**: All methods include proper error handling and logging
2. **Performance Monitoring**: Automatic performance statistics collection
3. **State Management**: Built-in state management for component data
4. **Configuration Management**: Flexible configuration system
5. **Validation Framework**: Comprehensive input validation
6. **Serialization Support**: Built-in serialization for persistence
7. **Memory Management**: Memory requirement estimation and checking
8. **Lifecycle Management**: Proper initialization and cleanup
9. **Extensibility**: Easy to extend with custom functionality
10. **Documentation**: Comprehensive docstrings and examples

## Best Practices

1. **Always call `initialize()`** before using the component
2. **Implement all abstract helper methods** for proper functionality
3. **Use `_safe_process()`** for automatic error handling and performance tracking
4. **Override validation methods** for component-specific validation
5. **Call `cleanup()`** when done with the component
6. **Use state management** for storing component-specific data
7. **Implement proper error handling** in custom methods
8. **Provide accurate capability information** for better integration