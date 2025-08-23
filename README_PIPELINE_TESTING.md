# Pipeline Testing with Ares Launcher

This directory contains comprehensive test scripts for testing the step1, step1_5, and step2 pipeline using ares_launcher and enhanced_training_manager with mock data.

## Overview

The test scripts verify that the complete pipeline works correctly by:
1. Generating realistic mock data
2. Running each step individually
3. Running the complete pipeline
4. Validating outputs
5. Testing different orchestration methods

## Test Scripts

### 1. `test_step1_step1_5_step2_pipeline.py`
**Comprehensive test script with multiple testing approaches**

This script provides the most thorough testing with:
- Individual step testing
- StepOrchestrator testing
- EnhancedTrainingManager testing
- ares_launcher testing
- Output validation

**Usage:**
```bash
python test_step1_step1_5_step2_pipeline.py
```

### 2. `test_pipeline_with_ares_launcher.py`
**Simplified test script focused on ares_launcher**

This script focuses specifically on testing with ares_launcher:
- Individual step testing with ares_launcher
- Complete pipeline testing
- Blank training mode testing
- Output validation

**Usage:**
```bash
python test_pipeline_with_ares_launcher.py
```

### 3. `run_pipeline_test.sh`
**Shell script for command-line testing**

This script demonstrates how to use ares_launcher directly from the command line:
- Step-by-step execution
- Environment setup
- Output validation
- File size verification

**Usage:**
```bash
./run_pipeline_test.sh
```

## Mock Data Generation

All test scripts include a `MockDataGenerator` class that creates realistic trading data:

### Data Types Generated:
- **Klines (OHLCV)**: 1-minute candlestick data with realistic price movements
- **Aggtrades**: Aggregated trade data with realistic volumes and prices
- **Futures**: Futures data including mark prices and funding rates

### Data Characteristics:
- 30 days of historical data
- Realistic ETH price movements (~$3000 base price)
- Proper timestamps and data formats
- Parquet file format for efficiency

## Testing Approaches

### 1. Individual Step Testing
Test each step in isolation:
```python
# Test step1
await tester.test_step1_data_collection()

# Test step1_5
await tester.test_step1_5_data_converter()

# Test step2
await tester.test_step2_feature_engineering()
```

### 2. Complete Pipeline Testing
Test the entire pipeline from step1 to step2:
```python
# Using StepOrchestrator
await tester.test_with_step_orchestrator()

# Using EnhancedTrainingManager
await tester.test_with_enhanced_training_manager()

# Using ares_launcher
await tester.test_with_ares_launcher()
```

### 3. Command Line Testing
Use ares_launcher directly:
```bash
# Test individual steps
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step1_data_collection --force-rerun
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step1_5_data_converter --force-rerun
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step2_feature_engineering --force-rerun

# Test complete pipeline
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --force-rerun
```

## Expected Outputs

### Step1 Outputs:
- `data_cache/klines_BINANCE_ETHUSDT_1m_consolidated.parquet`
- `data_cache/aggtrades_BINANCE_ETHUSDT_consolidated.parquet`

### Step1.5 Outputs:
- `data_cache/unified_BINANCE_ETHUSDT_1m.parquet`
- `data_cache/unified_BINANCE_ETHUSDT_1m_config.json`

### Step2 Outputs:
- `data/training/features_BINANCE_ETHUSDT_train.parquet`
- `data/training/features_BINANCE_ETHUSDT_val.parquet`
- `data/training/features_BINANCE_ETHUSDT_test.parquet`

## Environment Setup

The test scripts automatically set up the required environment:

### Environment Variables:
```bash
export BLANK_TRAINING_MODE=1
export FULL_TRAINING_MODE=0
export FORCE=1
```

### Directories Created:
- `data_cache/` - For step1 and step1_5 outputs
- `data/training/` - For step2 outputs
- `log/` - For logging files

## Running the Tests

### Prerequisites:
1. Ensure you're in the project root directory
2. All dependencies are installed
3. Python environment is properly configured

### Quick Start:
```bash
# Run the comprehensive test
python test_step1_step1_5_step2_pipeline.py

# Or run the simplified test
python test_pipeline_with_ares_launcher.py

# Or run the shell script
./run_pipeline_test.sh
```

### Expected Results:
- All tests should pass
- Mock data should be generated
- Pipeline outputs should be created
- File sizes should be reasonable

## Troubleshooting

### Common Issues:

1. **Import Errors**: Ensure you're running from the project root directory
2. **Missing Dependencies**: Install required packages from `requirements.txt`
3. **Permission Errors**: Make sure the shell script is executable (`chmod +x run_pipeline_test.sh`)
4. **Data Generation Issues**: Check that numpy and pandas are properly installed

### Debug Mode:
To run with more verbose logging, modify the environment:
```bash
export LOG_LEVEL=DEBUG
python test_pipeline_with_ares_launcher.py
```

## Integration with CI/CD

These test scripts can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions step
- name: Test Pipeline
  run: |
    python test_step1_step1_5_step2_pipeline.py
    if [ $? -ne 0 ]; then
      echo "Pipeline test failed"
      exit 1
    fi
```

## Performance Considerations

- Mock data generation uses 30 days of data for quick testing
- For production testing, increase the data period
- Tests use blank training mode for faster execution
- File sizes are optimized for testing purposes

## Contributing

When adding new pipeline steps:
1. Update the mock data generator if needed
2. Add new test methods to the test classes
3. Update the output validation
4. Update this README with new information