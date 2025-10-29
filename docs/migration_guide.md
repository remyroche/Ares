# Migration Guide: From Legacy Pipeline to Autonomous Steps

## Overview

This guide helps users migrate from the legacy pipeline-based architecture to the new autonomous step-based system. The new architecture provides better modularity, maintainability, and performance.

## Key Changes

### Architecture Changes

#### Before (Legacy)
```
Pipeline → Sub-Pipeline → Steps → Components
```

#### After (Autonomous Steps)
```
Launcher → Step Registry → Individual Steps
```

### Major Changes

1. **Eliminated Sub-Pipelines**: Direct step execution instead of complex pipeline orchestration
2. **Autonomous Steps**: Each step is self-contained and can run independently
3. **Simplified Launcher**: Single launcher for all step execution
4. **Enhanced Artifact Management**: Automatic format conversion and better organization
5. **Step Registry**: Global registry for step discovery and execution

## Migration Steps

### 1. Update Command Line Usage

#### Before (Legacy)
```bash
# Complex pipeline commands
python ares_launcher.py --mode sequential --sub_pipeline feature_generation_data_validation_step --symbol ETHUSDT --execution-mode light

python ares_launcher.py --start-from-step-name feature_generation_period_lookback_optimization_step --symbol ETHUSDT --execution-mode light

python ares_launcher.py --mode sequential --sub_pipeline feature_generation_feature_selection_step --stop-at-step 4 --symbol ETHUSDT --execution-mode light
```

#### After (Autonomous Steps)
```bash
# Simplified step commands
python ares_launcher.py step feature_generation_data_validation_step --symbol ETHUSDT --timeframe 15m --direction long --execution-mode light

python ares_launcher.py step feature_generation_period_lookback_optimization_step --symbol ETHUSDT --timeframe 15m --direction long --execution-mode light

python ares_launcher.py step feature_generation_feature_selection_step --symbol ETHUSDT --timeframe 15m --direction long --execution-mode light
```

### 2. Update Configuration

#### Before (Legacy)
```yaml
# Complex pipeline configuration
pipeline:
  mode: sequential
  sub_pipeline: feature_generation_data_validation_step
  start_from_step: feature_generation_period_lookback_optimization_step
  stop_at_step: 4
  execution_mode: light
```

#### After (Autonomous Steps)
```yaml
# Simplified step configuration
step:
  name: feature_generation_data_validation_step
  symbol: ETHUSDT
  timeframe: 15m
  direction: long
  execution_mode: light
```

### 3. Update Code Structure

#### Before (Legacy Pipeline)
```python
class LegacyPipeline:
    def __init__(self, config):
        self.config = config
        self.sub_pipelines = {}
    
    def run(self):
        # Complex pipeline orchestration
        for sub_pipeline in self.sub_pipelines:
            sub_pipeline.run()
```

#### After (Autonomous Step)
```python
from src.training.steps.base_step import BaseStep

class NewStep(BaseStep):
    def __init__(self, step_name: str = "new_step"):
        super().__init__(step_name)
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Simplified, focused logic
        pass
```

## Step-by-Step Migration

### 1. Identify Your Current Usage

#### Legacy Commands to Update
- `--mode sequential` → Use `step` or `steps` command
- `--sub_pipeline <name>` → Use `step <name>` command
- `--start-from-step-name <name>` → Use `step <name>` command
- `--stop-at-step <number>` → Use `steps` command with specific step names

#### Legacy Configuration to Update
- Pipeline configuration files
- Sub-pipeline specific settings
- Complex orchestration parameters

### 2. Update Command Line Scripts

#### Before
```bash
#!/bin/bash
# Legacy pipeline script
python ares_launcher.py --mode sequential --sub_pipeline feature_generation_data_validation_step --symbol ETHUSDT --execution-mode light
python ares_launcher.py --start-from-step-name feature_generation_period_lookback_optimization_step --symbol ETHUSDT --execution-mode light
```

#### After
```bash
#!/bin/bash
# New autonomous step script
python ares_launcher.py step feature_generation_data_validation_step --symbol ETHUSDT --timeframe 15m --direction long --execution-mode light
python ares_launcher.py step feature_generation_period_lookback_optimization_step --symbol ETHUSDT --timeframe 15m --direction long --execution-mode light
```

### 3. Update Configuration Files

#### Before
```yaml
# Legacy configuration
pipeline:
  mode: sequential
  sub_pipeline: feature_generation_data_validation_step
  execution_mode: light
  symbol: ETHUSDT
```

#### After
```yaml
# New configuration
step:
  name: feature_generation_data_validation_step
  symbol: ETHUSDT
  timeframe: 15m
  direction: long
  execution_mode: light
```

### 4. Update Custom Code

#### Before (Legacy Integration)
```python
from src.training.steps.pre_training.sub_pipeline import PreTrainingSubPipeline

# Legacy sub-pipeline usage
sub_pipeline = PreTrainingSubPipeline(config)
result = sub_pipeline.run()
```

#### After (Autonomous Step Integration)
```python
from src.training.steps.base_step import step_registry

# New autonomous step usage
step_class = step_registry.get_step("feature_generation_data_validation_step")
step = step_class()
result = await step.execute(config)
```

## Common Migration Scenarios

### Scenario 1: Running Individual Steps

#### Before
```bash
python ares_launcher.py --mode sequential --sub_pipeline feature_generation_data_validation_step --symbol ETHUSDT --execution-mode light
```

#### After
```bash
python ares_launcher.py step feature_generation_data_validation_step --symbol ETHUSDT --timeframe 15m --direction long --execution-mode light
```

### Scenario 2: Running Multiple Steps

#### Before
```bash
python ares_launcher.py --mode sequential --sub_pipeline feature_generation_data_validation_step --symbol ETHUSDT --execution-mode light
python ares_launcher.py --start-from-step-name feature_generation_period_lookback_optimization_step --symbol ETHUSDT --execution-mode light
```

#### After
```bash
python ares_launcher.py steps feature_generation_data_validation_step,feature_generation_period_lookback_optimization_step --symbol ETHUSDT --timeframe 15m --direction long --execution-mode light
```

### Scenario 3: Running Entire Stages

#### Before
```bash
python ares_launcher.py --mode sequential --sub_pipeline pre_training --symbol ETHUSDT --execution-mode light
```

#### After
```bash
python ares_launcher.py stage PRE_TRAINING --symbol ETHUSDT --timeframe 15m --direction long --execution-mode light
```

### Scenario 4: Custom Step Development

#### Before (Legacy Step)
```python
class LegacyStep:
    def __init__(self, config):
        self.config = config
    
    def run(self):
        # Step logic
        pass
```

#### After (Autonomous Step)
```python
from src.training.steps.base_step import BaseStep

class NewStep(BaseStep):
    def __init__(self, step_name: str = "new_step"):
        super().__init__(step_name)
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Step logic
        return {
            'success': True,
            'artifacts': [],
            'metrics': {}
        }
```

## Artifact Management Changes

### Before (Legacy)
```python
# Manual artifact management
artifact_path = save_artifact(data, 'artifact_name', 'data')
data = load_artifact('artifact_name', 'data')
```

### After (Autonomous Steps)
```python
# Automatic artifact management
artifact_path = self._save_artifact(data, 'artifact_name', 'data')
data = self._get_artifact('artifact_name', 'data')
```

### New Features
- **Automatic CSV Export**: DataFrames with < 2000 rows automatically generate CSV files
- **Better Organization**: Artifacts organized by symbol, exchange, direction, model
- **Enhanced Metadata**: Automatic metadata tracking and versioning

## Testing Migration

### 1. Test Individual Steps
```bash
# Test each step individually
python ares_launcher.py step data_download --symbol ETHUSDT --timeframe 15m --direction long --execution-mode light
python ares_launcher.py step sr_detection --symbol ETHUSDT --timeframe 15m --direction long --execution-mode light
```

### 2. Test Step Sequences
```bash
# Test step sequences
python ares_launcher.py steps data_download,sr_detection --symbol ETHUSDT --timeframe 15m --direction long --execution-mode light
```

### 3. Test Stage Execution
```bash
# Test stage execution
python ares_launcher.py stage DATA_COLLECTION --symbol ETHUSDT --timeframe 15m --direction long --execution-mode light
```

### 4. Verify Artifacts
```bash
# Check artifact generation
ls -la artifacts/ETHUSDT/binance/long/
```

## Troubleshooting Migration Issues

### Common Issues

#### 1. Step Not Found
```bash
# Check available steps
python ares_launcher.py list-steps

# Verify step registration
python ares_launcher.py step-info <step_name>
```

#### 2. Configuration Errors
```bash
# Check configuration
python ares_launcher.py check-config --symbol ETHUSDT --timeframe 15m
```

#### 3. Artifact Issues
```bash
# Check artifact status
python ares_launcher.py check-artifacts --symbol ETHUSDT --timeframe 15m
```

#### 4. Performance Issues
```bash
# Use light mode for testing
python ares_launcher.py step <step_name> --symbol ETHUSDT --timeframe 15m --direction long --execution-mode light
```

### Debug Mode
```bash
# Enable debug logging
python ares_launcher.py step <step_name> --symbol ETHUSDT --timeframe 15m --direction long --debug

# Verbose output
python ares_launcher.py step <step_name> --symbol ETHUSDT --timeframe 15m --direction long --verbose
```

## Benefits of Migration

### 1. Simplified Usage
- **Before**: Complex pipeline commands with multiple flags
- **After**: Simple, intuitive step commands

### 2. Better Performance
- **Before**: Complex orchestration overhead
- **After**: Direct step execution with optimized performance

### 3. Enhanced Modularity
- **Before**: Tightly coupled pipeline components
- **After**: Loosely coupled autonomous steps

### 4. Improved Maintainability
- **Before**: Complex pipeline logic
- **After**: Simple, focused step logic

### 5. Better Error Handling
- **Before**: Complex error propagation through pipelines
- **After**: Clear error handling at step level

## Migration Checklist

### Pre-Migration
- [ ] Backup current configuration files
- [ ] Document current usage patterns
- [ ] Identify custom integrations
- [ ] Plan migration timeline

### During Migration
- [ ] Update command line scripts
- [ ] Update configuration files
- [ ] Update custom code
- [ ] Test individual steps
- [ ] Test step sequences
- [ ] Verify artifact generation

### Post-Migration
- [ ] Update documentation
- [ ] Train team on new usage
- [ ] Monitor system performance
- [ ] Collect feedback
- [ ] Optimize based on usage patterns

## Support

For migration assistance:
1. Check the documentation in the `docs/` directory
2. Review the step reference guide
3. Test with light mode first
4. Use debug mode for troubleshooting
5. Create issues for specific problems

## Conclusion

The migration to autonomous steps provides significant benefits in terms of simplicity, performance, and maintainability. While the initial migration requires some effort, the long-term benefits make it worthwhile.

The new system is more intuitive, performant, and easier to maintain, making it a solid foundation for future development and enhancements.
