# HIGH COMPLEXITY REFACTORING GUIDE

## Overview
This guide provides specific refactoring strategies for functions with complexity > 30.

## General Principles

1. **Single Responsibility Principle**: Each function should do one thing well
2. **DRY (Don't Repeat Yourself)**: Extract common patterns into reusable functions
3. **Composition over Complexity**: Build complex behavior from simple functions
4. **Early Returns**: Use guard clauses to reduce nesting

## Specific Refactoring Strategies

### 1. Extract Method (Most Common)
- Break large functions into smaller, focused methods
- Each method should have complexity < 10
- Name methods clearly to describe what they do

### 2. Replace Conditional with Polymorphism
- Replace large if-elif chains with strategy/visitor patterns
- Use dictionaries to map conditions to handlers

### 3. Introduce Parameter Object
- Group related parameters into objects
- Reduces parameter lists and improves cohesion

### 4. Replace Nested Conditionals with Guard Clauses
```python
# Before
def process(data):
    if data is not None:
        if len(data) > 0:
            if validate(data):
                # actual processing
                
# After  
def process(data):
    if data is None:
        return None
    if len(data) == 0:
        return []
    if not validate(data):
        raise ValueError("Invalid data")
    
    # actual processing
```

### 5. Extract Class
- When a function has too many responsibilities
- Group related functionality into a class

## Function-Specific Recommendations

### VectorizedAdvancedFeatureEngineering.engineer_features (Complexity: 147)
**Problem**: Doing too many things in one method
**Solution**: 
1. Extract feature generation for each category into separate methods
2. Create a FeaturePipeline class to orchestrate the process
3. Use builder pattern for feature configuration

### VectorizedLabellingOrchestrator.orchestrate_labeling_and_feature_engineering (Complexity: 69)
**Problem**: Orchestrating too many operations inline
**Solution**:
1. Extract each major step into its own method
2. Create separate classes for labeling and feature engineering
3. Use template method pattern for the orchestration flow

### Step16ConfidenceCalibration.execute (Complexity: 46)
**Problem**: Too many calibration steps in one method
**Solution**:
1. Extract calibration for each model type into separate methods
2. Create CalibrationStrategy classes for different approaches
3. Use factory pattern to select appropriate calibration strategy

### DataCollectionStep._log_detailed_data_extract (Complexity: 41)
**Problem**: Complex logging logic with many branches
**Solution**:
1. Create separate logger classes for each data type
2. Use strategy pattern to select appropriate logger
3. Extract common logging functionality into base class
