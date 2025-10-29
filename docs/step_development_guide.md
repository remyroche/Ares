# Step Development Guide

## Overview

The Ares trading system has been refactored to use an autonomous step-based architecture. Each step is a self-contained module that can be executed independently via the launcher, with automatic artifact management and outcome generation.

## Architecture

### Core Components

1. **BaseStep**: Abstract base class that all steps inherit from
2. **Artifact Manager**: Centralized artifact storage and retrieval system
3. **Step Registry**: Global registry for step discovery and execution
4. **Ares Launcher**: Simplified launcher that orchestrates step execution

### Step Lifecycle

```
Configuration → Step Execution → Artifact Generation → Outcome Report
```

## Creating a New Step

### 1. Basic Step Structure

```python
from src.training.steps.base_step import BaseStep
from typing import Dict, Any

class MyCustomStep(BaseStep):
    """
    Custom step for specific functionality.
    """
    
    def __init__(self, step_name: str = "my_custom_step"):
        super().__init__(step_name)
        # Initialize step-specific components
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the step logic.
        
        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - direction: Trading direction ('long' or 'short')
                - execution_mode: Execution mode ('light' or 'full')
                - Additional step-specific parameters
        
        Returns:
            Dictionary containing:
                - success: Boolean indicating success/failure
                - artifacts: List of artifact paths generated
                - metrics: Dictionary of execution metrics
                - Additional step-specific results
        """
        self.logger.info(f'Starting {self.step_name}')
        
        try:
            # Extract configuration
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'long')
            execution_mode = config.get('execution_mode', 'light')
            
            if not symbol:
                raise ValueError("Symbol is required")
            
            # Initialize artifacts and metrics
            artifacts = []
            metrics = {}
            
            # Set up artifact manager context
            self.artifact_manager.set_context(
                symbol=symbol,
                exchange=exchange,
                direction=direction,
                model='YourModelName'  # Replace with appropriate model name
            )
            
            # Perform step-specific logic
            result = await self._perform_step(path, timeframe, direction, execution_mode, config)
            
            # Save result as artifact (automatically generates Parquet + CSV if < 2000 rows)
            artifact_path = self._save_artifact(
                result,
                'step_result',
                'data'
            )
            artifacts.append(artifact_path)
            
            # Record metrics
            metrics.update({
                'items_processed': result.get('count', 0),
                'execution_time': result.get('duration', 0.0),
                'execution_mode': execution_mode
            })
            
            self.logger.info(f'✅ {self.step_name} completed successfully')
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics,
                'result': result
            }
            
        except Exception as e:
            self.logger.error(f'❌ {self.step_name} failed: {e}')
            return {
                'success': False,
                'artifacts': [],
                'metrics': {},
                'error': str(e)
            }
    
    async def _perform_step(self, symbol: str, timeframe: str, 
                           direction: str, execution_mode: str, 
                           config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform the actual step logic.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for analysis
            direction: Trading direction
            execution_mode: Execution mode
            config: Full configuration
            
        Returns:
            Step-specific result dictionary
        """
        # Implement your step logic here
        return {
            'count': 100,
            'duration': 1.5,
            'data': 'sample_data'
        }
```

### 2. Registering the Step

Create or update the `__init__.py` file in your step's directory:

```python
"""
Your Step Module.

This module registers all steps for autonomous execution.
"""

from src.training.steps.base_step import step_registry
from .your_step_file import YourStepClass

# Register steps
step_registry.register("your_step_name", YourStepClass)
```

### 3. Running the Step

```bash
# Run a single step
python ares_launcher.py step your_step_name --symbol ETHUSDT --timeframe 15m --direction long

# Run multiple steps
python ares_launcher.py steps step1,step2,step3 --symbol ETHUSDT --timeframe 15m

# Run an entire stage
python ares_launcher.py stage DATA_COLLECTION --symbol ETHUSDT --timeframe 15m
```

## Artifact Management

### Automatic Artifact Generation

The artifact manager automatically:
- Saves all data as Parquet files (primary format)
- Generates CSV files for DataFrames with < 2000 rows
- Compresses large datasets
- Tracks metadata and versioning

### Manual Artifact Management

```python
# Save an artifact
artifact_path = self._save_artifact(
    data,                    # Your data (DataFrame, dict, etc.)
    'artifact_name',        # Name for the artifact
    'data'                  # Type: 'data', 'model', 'metadata', etc.
)

# Retrieve an artifact
data = self._get_artifact('artifact_name', 'data')
```

## Step Categories

### DATA_COLLECTION
- **data_download**: Download raw data from exchanges
- **data_conversion**: Convert data formats and standardize
- **data_validation**: Validate data quality and integrity
- **data_preparation**: Prepare data for further processing

### MARKET_ANALYSIS
- **sr_detection**: Detect Support/Resistance levels
- **sr_clustering**: Generate SR clusters
- **sr_parameter_optimization**: Optimize SR detection parameters

### PRE_TRAINING
- **feature_generation_data_validation_step**: Enhanced data validation
- **feature_generation_period_lookback_optimization_step**: Period + lookback optimization
- **feature_generation_feature_generation_step**: Feature generation
- **feature_generation_feature_selection_step**: Feature selection

### MODEL_TRAINING
- **analyst_models_training**: Train Analyst models
- **tactician_models_training**: Train Tactician models
- **analyst_ensemble_training**: Train Analyst ensemble models
- **tactician_ensemble_training**: Train Tactician ensemble models

### BACKTESTING
- **final_parameters_optimization**: Final system parameters optimization
- **real_parameters_optimization**: Real trading parameters optimization

## Best Practices

### 1. Error Handling
- Always wrap main logic in try-catch blocks
- Provide meaningful error messages
- Log errors appropriately

### 2. Configuration
- Validate required configuration parameters
- Provide sensible defaults
- Document configuration options

### 3. Logging
- Use structured logging with appropriate levels
- Include progress indicators for long-running operations
- Log key metrics and results

### 4. Performance
- Use async/await for I/O operations
- Implement progress tracking for long operations
- Consider memory usage for large datasets

### 5. Testing
- Write unit tests for step logic
- Test with different configuration combinations
- Validate artifact generation

## Migration from Legacy Code

### Before (Legacy Pipeline)
```python
class LegacyPipeline:
    def __init__(self, config):
        self.config = config
    
    def run(self):
        # Complex pipeline logic
        pass
```

### After (Autonomous Step)
```python
class NewStep(BaseStep):
    def __init__(self, step_name: str = "new_step"):
        super().__init__(step_name)
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Simplified, focused logic
        pass
```

### Key Changes
1. **Inheritance**: Inherit from `BaseStep` instead of custom base classes
2. **Method Signature**: Use `async def execute(self, config: Dict[str, Any])`
3. **Return Format**: Return standardized dictionary with success, artifacts, metrics
4. **Artifact Management**: Use `self._save_artifact()` and `self._get_artifact()`
5. **Registration**: Register step in `__init__.py` file

## Troubleshooting

### Common Issues

1. **Step Not Found**: Ensure step is registered in `__init__.py`
2. **Import Errors**: Check all dependencies are available
3. **Configuration Errors**: Validate required parameters are provided
4. **Artifact Issues**: Check artifact manager context is set correctly

### Debug Mode

```bash
# Run with debug logging
python ares_launcher.py step your_step --symbol ETHUSDT --timeframe 15m --debug

# Run with verbose output
python ares_launcher.py step your_step --symbol ETHUSDT --timeframe 15m --verbose
```

## Examples

### Simple Data Processing Step
```python
class DataProcessingStep(BaseStep):
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Load data
        data = await self._load_data(config)
        
        # Process data
        processed_data = self._process_data(data)
        
        # Save result
        artifact_path = self._save_artifact(processed_data, 'processed_data', 'data')
        
        return {
            'success': True,
            'artifacts': [artifact_path],
            'metrics': {'rows_processed': len(processed_data)},
            'processed_data': processed_data
        }
```

### Complex ML Training Step
```python
class MLTrainingStep(BaseStep):
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Initialize training components
        self._initialize_training_components()
        
        # Train models
        models = await self._train_models(config)
        
        # Save models and metrics
        model_artifacts = []
        for model_name, model in models.items():
            artifact_path = self._save_artifact(model, f'{model_name}_model', 'model')
            model_artifacts.append(artifact_path)
        
        return {
            'success': True,
            'artifacts': model_artifacts,
            'metrics': {'models_trained': len(models)},
            'models': models
        }
```

This guide provides everything needed to develop, register, and execute autonomous steps in the Ares trading system.
