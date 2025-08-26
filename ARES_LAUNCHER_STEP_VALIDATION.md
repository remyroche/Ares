# Ares Launcher Step-Based Validation

## Overview

The `ares_launcher.py` has been enhanced to support starting at any step in the training pipeline with comprehensive validation. The system now ensures that previous steps are validated before proceeding to the next step, preventing pipeline failures due to missing or invalid data.

## New Features

### 1. Step-Based Commands

You can now start training from any specific step using dedicated commands:

```bash
# Start from specific steps with validation
python ares_launcher.py step1 --symbol ETHUSDT --exchange BINANCE --training-mode light
python ares_launcher.py step4 --symbol ETHUSDT --exchange BINANCE --training-mode blank
python ares_launcher.py step8 --symbol ETHUSDT --exchange BINANCE --training-mode full
python ares_launcher.py step5 --symbol ETHUSDT --exchange BINANCE --training-mode light --force
python ares_launcher.py step10 --symbol ETHUSDT --exchange BINANCE --training-mode blank --gui
```

### 2. Training Modes

Three training modes are available for step-based commands, each with pre-configured parameter values:

- **Light Mode** (`--training-mode light`): 30 days of data for quick testing and development
- **Blank Mode** (`--training-mode blank`): 180 days of data for development and validation (default)
- **Full Mode** (`--training-mode full`): 730 days of data for production training and backtesting

Each mode has optimized parameter configurations for its specific use case, including lookback periods, feature engineering parameters, and model training settings.

### 3. Available Step Commands

All pipeline steps are available as individual commands:

- `step1` - Data Collection
- `step1_5` - Data Converter
- `step2` - Data Reading
- `step3` - HMM Regime Discovery
- `step4` - Triple Barrier Method
- `step5` - Labeling
- `step6` - Feature Engineering
- `step7` - Regime Data Splitting
- `step8` - HMM-Based Training
- `step8_5` - Unified Regime Intelligence
- `step9` - Analyst Enhancement
- `step10` - Tactician Labeling
- `step11` - Tactician Specialist Training
- `step12` - Confidence Calibration
- `step13` - Final Parameters Optimization
- `step14` - Walk Forward Validation
- `step15` - Monte Carlo Validation
- `step16` - AB Testing
- `step17` - Saving

## Validation System

### 1. Pre-Execution Validation

Before starting any step, the system validates all previous steps:

```python
# Example validation flow for step5
step1_data_collection ✅
step1_5_data_converter ✅
step2_data_reading ✅
step3_hmm_regime_discovery ✅
step4_triple_barrier_method ✅
step5_labeling 🚀 (starting)
```

### 2. Step-by-Step Validation

Each step is validated after completion before proceeding to the next step:

```python
# EnhancedTrainingManager validation flow
step4_success = await step4_triple_barrier_method.run_step(...)
if step4_success:
    step4_validation = await self._run_step_validator("step4_triple_barrier_method", ...)
    if step4_validation.get("validation_passed", False):
        # Proceed to step5
        step5_success = await step5_labeling.run_step(...)
    else:
        # Stop pipeline - validation failed
        return False
```

### 3. Validation Reports

Comprehensive validation reports are generated:

```
================================================================================
📊 STEP VALIDATION REPORT
🎯 Symbol: ETHUSDT
🏢 Exchange: BINANCE
🚀 Starting from: step5_labeling
================================================================================
step1_data_collection                    ✅ PASSED
step1_5_data_converter                   ✅ PASSED
step2_data_reading                       ✅ PASSED
step3_hmm_regime_discovery               ✅ PASSED
step4_triple_barrier_method              ✅ PASSED
================================================================================
🎉 All previous steps validated successfully!
================================================================================
```

## Usage Examples

### Basic Step Execution

```bash
# Start from step4 with blank mode (default)
python ares_launcher.py step4 --symbol ETHUSDT --exchange BINANCE

    # Start from step8 with light mode for quick testing (30 days)
    python ares_launcher.py step8 --symbol ETHUSDT --exchange BINANCE --training-mode light

    # Start from step12 with full mode for production (730 days)
    python ares_launcher.py step12 --symbol ETHUSDT --exchange BINANCE --training-mode full
```

### Force Rerun

```bash
# Force rerun from step5, clearing previous progress
python ares_launcher.py step5 --symbol ETHUSDT --exchange BINANCE --force

# Force rerun with light mode
python ares_launcher.py step7 --symbol ETHUSDT --exchange BINANCE --training-mode light --force
```

### GUI Integration

```bash
# Start from step6 with GUI
python ares_launcher.py step6 --symbol ETHUSDT --exchange BINANCE --gui

# Start from step10 with blank mode and GUI
python ares_launcher.py step10 --symbol ETHUSDT --exchange BINANCE --training-mode blank --gui
```

## Validation Details

### 1. Step Dependencies

The system uses the `StepDependencyValidator` to determine which steps need to be validated:

```python
step_dependencies = {
    "step1_data_collection": [],
    "step1_5_data_converter": ["step1_data_collection"],
    "step2_data_reading": ["step1_5_data_converter"],
    "step3_hmm_regime_discovery": ["step2_data_reading"],
    "step4_triple_barrier_method": ["step3_hmm_regime_discovery"],
    # ... and so on
}
```

### 2. Validator Orchestration

The `ValidatorOrchestrator` manages validation execution:

```python
validator_orchestrator = ValidatorOrchestrator()
result = await validator_orchestrator.run_step_validator(
    step_name, training_input, pipeline_state, config
)
```

### 3. Validation Criteria

Each step has specific validation criteria:

- **File Existence**: Required output files must exist
- **Data Quality**: Data must meet quality standards
- **Structure Validation**: Required columns and formats
- **Size Validation**: Files must have reasonable sizes
- **Content Validation**: Data must be logically consistent

## Error Handling

### 1. Validation Failures

If validation fails, the pipeline stops with detailed error information:

```
❌ step4_triple_barrier_method validation failed: Triple barrier labels file not found
❌ Cannot start from step5_labeling - previous step validation failed
```

### 2. Graceful Degradation

The system provides fallback validation when full validation is not possible:

```python
try:
    # Full validation with validators
    result = await validator_orchestrator.run_step_validator(...)
except Exception as e:
    # Fallback to basic file existence check
    self.logger.warning(f"⚠️ Could not run validators: {e}")
    self.logger.warning("Proceeding with basic file existence check")
```

## Environment Variables

The system sets appropriate environment variables based on training mode:

```python
    # Light mode (30 days)
    os.environ["LIGHT_TRAINING_MODE"] = "1"
    os.environ["BLANK_TRAINING_MODE"] = "0"
    os.environ["FULL_TRAINING_MODE"] = "0"

    # Blank mode (180 days)
    os.environ["BLANK_TRAINING_MODE"] = "1"
    os.environ["LIGHT_TRAINING_MODE"] = "0"
    os.environ["FULL_TRAINING_MODE"] = "0"

    # Full mode (730 days)
    os.environ["FULL_TRAINING_MODE"] = "1"
    os.environ["LIGHT_TRAINING_MODE"] = "0"
    os.environ["BLANK_TRAINING_MODE"] = "0"
```

## Backward Compatibility

### 1. Legacy Commands

All existing commands continue to work:

```bash
# Legacy step-based commands (still supported)
python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step4_triple_barrier_method
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step5_labeling
```

### 2. Existing Modes

Traditional modes remain unchanged:

```bash
# Traditional modes
python ares_launcher.py light --symbol ETHUSDT --exchange BINANCE
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE
python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE
```

## Benefits

### 1. Reliability

- **Prevents Pipeline Failures**: Validation ensures data integrity before proceeding
- **Early Error Detection**: Issues are caught before expensive processing
- **Consistent Quality**: All steps meet quality standards

### 2. Flexibility

- **Granular Control**: Start from any step in the pipeline
- **Multiple Training Modes**: Light, blank, and full modes for different use cases
- **Force Rerun**: Clear previous progress when needed

### 3. Observability

- **Detailed Reports**: Comprehensive validation reports
- **Clear Error Messages**: Specific error information for debugging
- **Progress Tracking**: Step-by-step progress monitoring

### 4. Efficiency

- **Skip Completed Steps**: Only run necessary steps
- **Validation Caching**: Avoid redundant validation
- **Resource Optimization**: Use appropriate data sizes for different modes

## Troubleshooting

### Common Issues

1. **Validation Failures**: Check that previous steps completed successfully
2. **Missing Files**: Ensure data files exist in expected locations
3. **Permission Issues**: Verify file access permissions
4. **Memory Issues**: Use light mode for large datasets

### Debug Commands

```bash
# Check step status
python ares_launcher.py step4 --symbol ETHUSDT --exchange BINANCE --training-mode light

# Force rerun with validation
python ares_launcher.py step5 --symbol ETHUSDT --exchange BINANCE --force

# Run with GUI for visual debugging
python ares_launcher.py step6 --symbol ETHUSDT --exchange BINANCE --gui
```

## Future Enhancements

1. **Parallel Validation**: Validate multiple steps simultaneously
2. **Incremental Validation**: Only validate changed data
3. **Custom Validation Rules**: User-defined validation criteria
4. **Validation Profiles**: Predefined validation configurations
5. **Real-time Monitoring**: Live validation status updates