# Training Module Structure

## Overview

This document defines the organization and structure of the training module to ensure clarity, maintainability, and scalability.

## Directory Structure

```
src/training/
├── __init__.py                     # Module initialization
├── PIPELINE_DOCUMENTATION.md       # Pipeline documentation
├── MODULE_STRUCTURE.md            # This file
├── step_config.py                 # Centralized step configuration
├── base_step.py                   # Base class for all steps
│
├── core/                          # Core training components
│   ├── __init__.py
│   ├── training_manager.py        # Main training manager
│   ├── step_executor.py           # Step execution logic
│   ├── dependency_manager.py      # Dependency management
│   └── progress_tracker.py        # Progress tracking
│
├── steps/                         # Training pipeline steps
│   ├── __init__.py
│   ├── data_preparation/          # Steps 1-2: Data preparation
│   │   ├── step01_data_collection.py
│   │   ├── step01_5_data_converter.py
│   │   └── step02_feature_engineering.py
│   │
│   ├── market_analysis/           # Steps 3-5: Market analysis
│   │   ├── step03_hmm_regime_discovery.py
│   │   ├── step04_regime_data_splitting.py
│   │   └── step05_triple_barrier_method.py
│   │
│   ├── feature_engineering/       # Steps 6-7: Feature engineering
│   │   ├── step06_advanced_features.py
│   │   └── step07_feature_selection.py
│   │
│   ├── model_training/            # Steps 8-11: Model training
│   │   ├── step08_regime_training.py
│   │   ├── step09_hmm_training.py
│   │   ├── step10_unified_intelligence.py
│   │   └── step11_analyst_creation.py
│   │
│   ├── advanced_training/         # Steps 12-15: Advanced training
│   │   ├── step12_analyst_enhancement.py
│   │   ├── step13_ensemble_creation.py
│   │   ├── step14_tactician_labeling.py
│   │   └── step15_tactician_training.py
│   │
│   ├── validation/                # Steps 16-20: Validation & optimization
│   │   ├── step16_confidence_calibration.py
│   │   ├── step17_parameter_optimization.py
│   │   ├── step18_walk_forward_validation.py
│   │   ├── step19_monte_carlo_validation.py
│   │   └── step20_ab_testing.py
│   │
│   └── persistence/               # Step 21: Model persistence
│       └── step21_model_saving.py
│
├── optimization/                  # Optimization utilities
│   ├── __init__.py
│   ├── hyperparameter_tuner.py
│   ├── memory_optimizer.py
│   ├── parallel_executor.py
│   └── cache_manager.py
│
├── utils/                         # Training-specific utilities
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_calculator.py
│   ├── model_evaluator.py
│   └── report_generator.py
│
├── validators/                    # Step validators
│   ├── __init__.py
│   ├── base_validator.py
│   └── step_validators/
│       ├── step01_validator.py
│       ├── step02_validator.py
│       └── ...
│
└── reports/                       # Generated reports
    ├── pipeline_execution/
    ├── step_execution/
    └── optimization_results/
```

## Module Hierarchy

### 1. Core Layer
- **Purpose**: Provide core orchestration and management functionality
- **Components**:
  - `training_manager.py`: Main entry point for training
  - `step_executor.py`: Handles individual step execution
  - `dependency_manager.py`: Manages step dependencies
  - `progress_tracker.py`: Tracks pipeline progress

### 2. Steps Layer
- **Purpose**: Implement individual pipeline steps
- **Organization**: Grouped by functionality
- **Pattern**: Each step inherits from `BaseStep`

### 3. Optimization Layer
- **Purpose**: Provide optimization and performance utilities
- **Components**:
  - Memory optimization
  - Parallel processing
  - Caching
  - Hyperparameter tuning

### 4. Utils Layer
- **Purpose**: Provide common utilities for steps
- **Components**:
  - Data loading and processing
  - Feature calculation
  - Model evaluation
  - Report generation

### 5. Validators Layer
- **Purpose**: Validate step inputs and outputs
- **Pattern**: Each step has a corresponding validator

## Naming Conventions

### Files
- Step files: `step{number}_{descriptive_name}.py`
- Validators: `step{number}_validator.py`
- Utilities: `{functionality}_{component}.py`

### Classes
- Steps: `{Descriptive}Step` (e.g., `DataCollectionStep`)
- Validators: `Step{Number}Validator`
- Managers: `{Component}Manager`

### Functions
- Public: `snake_case` (e.g., `execute_step`)
- Private: `_snake_case` (e.g., `_validate_input`)
- Async: `async_` prefix or `_async` suffix

## Import Guidelines

### Standard Import Order
1. Standard library imports
2. Third-party imports
3. Local application imports

### Example:
```python
# Standard library
import os
import json
from pathlib import Path
from typing import Dict, Any

# Third-party
import pandas as pd
import numpy as np

# Local application
from src.training.base_step import BaseStep
from src.utils.logger import system_logger
```

## Code Organization Guidelines

### Step Implementation
1. Inherit from `BaseStep`
2. Implement required abstract methods
3. Use decorators for error handling
4. Include comprehensive logging
5. Write corresponding validator

### Example Step Structure:
```python
class DataCollectionStep(BaseStep):
    """Step 1: Data Collection."""
    
    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        # Step-specific initialization
        
    def validate_inputs(self, training_input, pipeline_state):
        """Validate inputs."""
        # Input validation logic
        
    def execute_logic(self, training_input, pipeline_state):
        """Execute main logic."""
        # Step execution logic
        
    def validate_outputs(self, pipeline_state):
        """Validate outputs."""
        # Output validation logic
```

## Best Practices

1. **Single Responsibility**: Each module should have one clear purpose
2. **Dependency Injection**: Pass dependencies through constructors
3. **Configuration**: Use centralized configuration
4. **Error Handling**: Use decorators and proper exception handling
5. **Logging**: Comprehensive logging at appropriate levels
6. **Testing**: Each module should have corresponding tests
7. **Documentation**: Clear docstrings and inline comments

## Migration Plan

To migrate existing code to this structure:

1. **Phase 1**: Create new directory structure
2. **Phase 2**: Move steps to appropriate subdirectories
3. **Phase 3**: Refactor large files into smaller modules
4. **Phase 4**: Update imports throughout the codebase
5. **Phase 5**: Remove deprecated files
6. **Phase 6**: Update documentation

## Maintenance Guidelines

1. **Regular Reviews**: Review module organization quarterly
2. **Refactoring**: Refactor modules that exceed 500 lines
3. **Documentation**: Keep documentation up-to-date
4. **Dependencies**: Minimize inter-module dependencies
5. **Testing**: Maintain test coverage above 80%