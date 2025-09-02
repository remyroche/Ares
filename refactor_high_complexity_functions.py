#!/usr/bin/env python3
"""
Refactoring script for high-complexity functions in src/training/steps
This script provides refactoring suggestions and examples for the most critical functions
"""

import ast
import logging
from pathlib import Path
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplexityRefactorHelper:
    """Helper to suggest refactoring for high-complexity functions"""
    
    def __init__(self):
        self.critical_functions = [
            {
                'file': 'src/training/steps/vectorized_advanced_feature_engineering.py',
                'function': 'engineer_features',
                'complexity': 147,
                'line': 2289,
                'refactor_approach': 'extract_method_pattern'
            },
            {
                'file': 'src/training/steps/vectorized_labelling_orchestrator.py',
                'function': 'orchestrate_labeling_and_feature_engineering',
                'complexity': 69,
                'line': 278,
                'refactor_approach': 'extract_method_pattern'
            },
            {
                'file': 'src/training/steps/step16_confidence_calibration.py',
                'function': 'execute',
                'complexity': 46,
                'line': 93,
                'refactor_approach': 'extract_method_pattern'
            },
            {
                'file': 'src/training/steps/step1_data_collection.py',
                'function': '_log_detailed_data_extract',
                'complexity': 41,
                'line': 677,
                'refactor_approach': 'extract_method_pattern'
            }
        ]
    
    def generate_refactoring_examples(self):
        """Generate refactoring examples for critical functions"""
        examples = []
        
        # Example 1: Extract Method Pattern
        example1 = '''
# REFACTORING PATTERN: Extract Method
# For: VectorizedAdvancedFeatureEngineering.engineer_features (Complexity: 147)

# BEFORE: Single massive method with 147 complexity
def engineer_features(self, data, feature_config):
    # 500+ lines of code doing everything
    # - Validation
    # - Feature engineering
    # - Cross-timeframe features
    # - Interaction features
    # - Cleaning
    # - Logging
    ...

# AFTER: Broken into smaller, focused methods
def engineer_features(self, data, feature_config):
    """Main orchestrator method - complexity reduced to ~10"""
    # Step 1: Validate inputs
    validated_data = self._validate_input_data(data, feature_config)
    
    # Step 2: Engineer base features
    base_features = self._engineer_base_features(validated_data)
    
    # Step 3: Add time-based features
    time_features = self._engineer_time_features(base_features)
    
    # Step 4: Add cross-timeframe features
    cross_features = self._engineer_cross_timeframe_features(time_features)
    
    # Step 5: Add interaction features
    interaction_features = self._engineer_interaction_features(cross_features)
    
    # Step 6: Clean and validate final features
    final_features = self._clean_and_validate_features(interaction_features)
    
    # Step 7: Log summary
    self._log_engineering_summary(final_features)
    
    return final_features

def _validate_input_data(self, data, feature_config):
    """Validate input data and configuration"""
    # Extracted validation logic
    ...

def _engineer_base_features(self, data):
    """Engineer base technical indicators"""
    # Extracted base feature engineering
    ...

# Additional extracted methods...
'''
        examples.append(("Extract Method Pattern", example1))
        
        # Example 2: Strategy Pattern
        example2 = '''
# REFACTORING PATTERN: Strategy Pattern
# For: DataCollectionStep._log_detailed_data_extract (Complexity: 41)

# BEFORE: Giant method with many conditional branches
def _log_detailed_data_extract(self, data_dict):
    # Huge if-elif chain handling different data types
    if data_type == "klines":
        # 50 lines of klines logging
    elif data_type == "aggtrades":
        # 50 lines of aggtrades logging
    elif data_type == "futures":
        # 50 lines of futures logging
    ...

# AFTER: Strategy pattern with dedicated handlers
class DataLoggerStrategy:
    """Abstract base for data logging strategies"""
    def log(self, data): 
        raise NotImplementedError

class KlinesLogger(DataLoggerStrategy):
    def log(self, data):
        # Focused klines logging logic
        ...

class AggtradesLogger(DataLoggerStrategy):
    def log(self, data):
        # Focused aggtrades logging logic
        ...

def _log_detailed_data_extract(self, data_dict):
    """Simplified method using strategy pattern"""
    loggers = {
        'klines': KlinesLogger(),
        'aggtrades': AggtradesLogger(),
        'futures': FuturesLogger()
    }
    
    for data_type, data in data_dict.items():
        logger = loggers.get(data_type)
        if logger:
            logger.log(data)
'''
        examples.append(("Strategy Pattern", example2))
        
        # Example 3: Builder Pattern
        example3 = '''
# REFACTORING PATTERN: Builder Pattern
# For complex feature engineering with many steps

class FeatureBuilder:
    """Builder for complex feature engineering pipelines"""
    
    def __init__(self, data):
        self.data = data
        self.features = {}
    
    def add_technical_indicators(self):
        """Add technical indicators"""
        self.features['technical'] = self._compute_technical_indicators()
        return self
    
    def add_market_microstructure(self):
        """Add market microstructure features"""
        self.features['microstructure'] = self._compute_microstructure()
        return self
    
    def add_regime_features(self):
        """Add regime-based features"""
        self.features['regime'] = self._compute_regime_features()
        return self
    
    def build(self):
        """Combine all features"""
        return pd.concat(list(self.features.values()), axis=1)

# Usage:
features = (FeatureBuilder(data)
            .add_technical_indicators()
            .add_market_microstructure()
            .add_regime_features()
            .build())
'''
        examples.append(("Builder Pattern", example3))
        
        return examples
    
    def create_refactoring_guide(self):
        """Create a comprehensive refactoring guide"""
        guide = """# HIGH COMPLEXITY REFACTORING GUIDE

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
"""
        return guide
    
    def save_refactoring_guide(self):
        """Save the refactoring guide and examples"""
        # Save guide
        guide_path = Path("refactoring_guide.md")
        guide_path.write_text(self.create_refactoring_guide())
        logger.info(f"Saved refactoring guide to {guide_path}")
        
        # Save examples
        examples = self.generate_refactoring_examples()
        for i, (pattern_name, example) in enumerate(examples):
            example_path = Path(f"refactoring_example_{i+1}_{pattern_name.lower().replace(' ', '_')}.py")
            example_path.write_text(example)
            logger.info(f"Saved {pattern_name} example to {example_path}")


def main():
    """Main entry point"""
    helper = ComplexityRefactorHelper()
    helper.save_refactoring_guide()
    
    logger.info("\n" + "="*60)
    logger.info("REFACTORING RECOMMENDATIONS")
    logger.info("="*60)
    logger.info("1. Start with the highest complexity functions (>40)")
    logger.info("2. Apply Extract Method pattern as the primary approach")
    logger.info("3. Aim for function complexity < 10")
    logger.info("4. Use design patterns where appropriate")
    logger.info("5. Write tests before refactoring to ensure behavior is preserved")
    logger.info("="*60)


if __name__ == "__main__":
    main()