# Enhanced Step 1.5 Data Converter Validator

## Overview

The Enhanced Step 1.5 Data Converter Validator provides comprehensive function call monitoring, detailed outcome reporting, and health check mechanisms for data conversion validation. This enhanced version includes enterprise-grade monitoring capabilities that ensure thorough checks whenever a function is called, when functions call other functions, and when functions finish - all with detailed reports about outcomes.

## Features

### 🔍 **Comprehensive Function Call Monitoring**
- **FunctionCallMonitor Class**: Tracks every function call with unique IDs, execution times, memory usage, and call depth
- **Entry/Exit Logging**: Detailed logging for function entry, parameters, execution time, memory delta, and exit status
- **Input/Output Validation**: Validates function parameters and return values automatically
- **Call Stack Tracking**: Monitors nested function calls with depth tracking and call chain validation

### 📊 **Detailed Outcome Reporting**
- **Comprehensive Validation Reports**: Detailed reports with step-by-step validation results
- **Performance Analytics**: Execution time breakdown, memory usage tracking, and resource consumption
- **Error Categorization**: Categorized errors with severity levels and detailed error messages
- **Success/Failure Metrics**: Detailed metrics for validation success rates and failure analysis

### 🏥 **Health Check Mechanisms**
- **System Resource Monitoring**: CPU, memory, and disk usage monitoring
- **Data Integrity Checks**: Validates data files, schemas, and consistency
- **Component Availability**: Checks for required dependencies and system components
- **File System Health**: Monitors file system performance and accessibility
- **Memory Health**: Detects potential memory leaks and fragmentation
- **Disk Space Monitoring**: Tracks available disk space and usage patterns

### ⚡ **Performance Monitoring**
- **Execution Time Tracking**: Precise timing for each function and overall execution
- **Memory Usage Monitoring**: Real-time memory usage tracking with delta calculations
- **Resource Consumption**: CPU, memory, and disk usage monitoring
- **Performance Metrics**: Min/max/average execution times and success rates

### 🛡️ **Enhanced Error Handling**
- **Context Preservation**: Full error context including call stack, parameters, and execution state
- **Stack Trace Logging**: Complete stack traces with function call history
- **Recovery Mechanisms**: Graceful error handling with detailed error reporting
- **Error Classification**: Errors categorized by type, severity, and impact

## Installation

### Prerequisites
- Python >= 3.8
- pip package manager

### Core Dependencies
```bash
pip install pandas psutil
```

### Optional Dependencies (Recommended)
```bash
pip install pyarrow fastparquet
```

### Quick Setup
```bash
# Run the setup script
python setup_step01_5_enhanced.py

# Or install from requirements file
pip install -r requirements_step01_5_enhanced.txt
```

## Usage

### Basic Usage
```python
import asyncio
from step01_5_data_converter_validator import run_validator

async def main():
    training_input = {
        "symbol": "ETHUSDT",
        "exchange": "BINANCE",
        "timeframe": "1m",
        "data_dir": "data_cache"
    }
    
    pipeline_state = {}
    
    result = await run_validator(training_input, pipeline_state)
    
    print(f"Validation passed: {result['validation_passed']}")
    print(f"Function calls: {result['function_call_summary']['total_calls']}")
    print(f"Execution time: {result['duration']:.4f}s")

asyncio.run(main())
```

### Advanced Usage with Health Checks
```python
from step01_5_data_converter_validator import Step1_5DataConverterValidator

async def advanced_validation():
    config = {}
    validator = Step1_5DataConverterValidator(config)
    
    # Run health checks
    health_results = await validator.health_checker.run_comprehensive_health_check({
        'data_dir': 'data_cache',
        'symbol': 'ETHUSDT',
        'exchange': 'BINANCE',
        'timeframe': '1m'
    })
    
    print(f"System health: {health_results['overall_status']}")
    
    # Run validation with monitoring
    training_input = {
        "symbol": "ETHUSDT",
        "exchange": "BINANCE", 
        "timeframe": "1m",
        "data_dir": "data_cache"
    }
    
    pipeline_state = {}
    result = await validator.validate(training_input, pipeline_state)
    
    return result
```

## Configuration

### Environment Variables
```bash
# Optional: Set log level
export LOG_LEVEL=INFO

# Optional: Set data directory
export DATA_DIR=data_cache

# Optional: Enable debug mode
export DEBUG_MODE=true
```

### Configuration Options
The validator accepts configuration through the `CONFIG` object:
- `min_records`: Minimum records per file (default: 500)
- `min_files`: Minimum number of daily files (default: 1)
- `required_columns`: List of required columns for validation

## Monitoring and Reporting

### Function Call Monitoring
Every function call is monitored with:
- Unique call ID
- Execution time
- Memory usage delta
- Call depth
- Input/output validation
- Success/failure status

### Health Check System
Comprehensive health checks include:
- **System Resources**: CPU, memory, disk usage
- **Data Integrity**: File validation, schema checks
- **Component Availability**: Dependency checks
- **File System Health**: I/O performance testing
- **Memory Health**: Leak detection
- **Disk Space**: Available space monitoring

### Performance Metrics
Real-time metrics include:
- Function execution times (min/max/average)
- Memory usage patterns
- Call stack depth
- Success/failure rates
- Resource consumption

## Error Handling

### Error Categories
- **Critical**: System-level errors that prevent execution
- **Warning**: Issues that don't prevent execution but should be addressed
- **Info**: Informational messages about system state

### Error Recovery
- Graceful degradation when optional dependencies are missing
- Context preservation for debugging
- Detailed error messages with recommendations
- Automatic retry mechanisms where appropriate

## Testing

### Run Tests
```bash
# Run the built-in test suite
python step01_5_data_converter_validator.py

# Run with verbose output
python -u step01_5_data_converter_validator.py
```

### Test Coverage
The enhanced validator includes comprehensive tests for:
- Function call monitoring
- Health check system
- Error handling
- Performance monitoring
- Data validation
- Edge cases and error conditions

## Troubleshooting

### Common Issues

#### Missing Dependencies
```bash
# Error: ModuleNotFoundError: No module named 'pandas'
pip install pandas psutil
```

#### Permission Issues
```bash
# Error: Permission denied when accessing data directory
chmod 755 data_cache/
```

#### Memory Issues
```bash
# Error: Memory usage too high
# Solution: Increase available memory or optimize data processing
```

### Debug Mode
Enable debug mode for detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Optimization

### Recommendations
1. **Install pyarrow** for faster parquet file processing
2. **Monitor memory usage** during large data processing
3. **Use appropriate data types** to reduce memory footprint
4. **Enable parallel processing** where possible
5. **Regular health checks** to prevent issues

### Performance Tuning
- Adjust `min_records` and `min_files` based on your data size
- Monitor function call metrics to identify bottlenecks
- Use health check recommendations for system optimization

## Contributing

### Development Setup
```bash
# Install development dependencies
pip install pytest pytest-asyncio black flake8 mypy

# Run tests
pytest tests/

# Format code
black step01_5_data_converter_validator.py

# Lint code
flake8 step01_5_data_converter_validator.py

# Type check
mypy step01_5_data_converter_validator.py
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints for all functions
- Include comprehensive docstrings
- Add tests for new functionality

## License

This enhanced validator is part of the Ares trading system and follows the same licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the comprehensive error messages
3. Run health checks to identify system issues
4. Check the detailed function call reports for debugging

## Changelog

### Version 2.0 (Enhanced)
- Added comprehensive function call monitoring
- Implemented health check system
- Enhanced error handling and reporting
- Added performance monitoring
- Improved validation framework
- Added audit trail system

### Version 1.0 (Original)
- Basic data validation
- Simple error handling
- Basic reporting