# Comprehensive Training Pipeline with Enhanced Data Quality Management

## Overview

This directory contains a comprehensive training pipeline system that executes steps 1-7 of the enhanced training pipeline with integrated data quality monitoring, compatibility validation, format verification, and proper indexing at every step.

## Key Components

### 1. Comprehensive Pipeline Executor (`comprehensive_pipeline_executor.py`)

The main orchestrator that provides:
- Complete execution of steps 1-7
- Real-time data quality monitoring
- Comprehensive validation at each step
- Automated issue detection and reporting
- Performance optimization and resource management

### 2. Steps 1-7 Comprehensive Executor (`steps_1_7_comprehensive_executor.py`)

A detailed executor that:
- Systematically executes each step with validation
- Ensures data compatibility across all steps
- Validates data quality at each step
- Ensures format compatibility and standardization
- Maintains proper indexing and temporal alignment

### 3. Data Quality Monitor (`data_quality_monitor.py`)

A comprehensive monitoring system that provides:
- Real-time data quality monitoring
- Compatibility validation
- Format verification
- Index validation
- Continuous quality scoring
- Automated issue detection and reporting

## Pipeline Steps Overview

### Step 1: Data Collection
- Downloads and consolidates market data
- Ensures data completeness and quality
- Validates exchange-specific formats
- Performs gap detection and analysis

### Step 1.5: Data Converter
- Converts data to unified format
- Performs resampling and consolidation
- Ensures data integrity preservation
- Validates unified format compliance

### Step 2: Data Reading
- Reads unified data from step 1.5
- Performs comprehensive data quality validation
- Ensures proper file structure and accessibility
- Validates format consistency

### Step 3: HMM Regime Discovery
- Performs Hidden Markov Model regime discovery
- Identifies market regimes and states
- Ensures model convergence and quality
- Validates regime characteristics

### Step 4: Regime Data Splitting
- Creates unified dataset with regime labels
- Splits data by regime for processing
- Ensures proper data distribution
- Validates temporal consistency

### Step 5: Labeling
- Creates comprehensive labels for training
- Combines triple barrier labels with meta-labeling
- Ensures label quality and distribution
- Validates temporal alignment

### Step 6: Feature Engineering
- Creates comprehensive features (basic + advanced)
- Performs regime-aware optimization
- Ensures feature quality and correlation
- Validates data types and missing values

### Step 7: Enhanced Matrix Operations
- Performs advanced matrix operations
- Ensures computational accuracy
- Optimizes memory usage
- Validates performance metrics

## Data Quality Management

### Quality Metrics

The system monitors six key quality dimensions:

1. **Completeness**: Measures data completeness and missing value ratios
2. **Consistency**: Validates data relationships and constraints
3. **Validity**: Ensures data ranges and formats are correct
4. **Timeliness**: Checks data freshness and age
5. **Uniqueness**: Identifies duplicate records
6. **Accuracy**: Detects outliers and extreme values

### Quality Levels

- **Excellent** (≥0.95): High-quality data with minimal issues
- **Good** (≥0.85): Good quality data with minor issues
- **Acceptable** (≥0.75): Acceptable quality with some issues
- **Poor** (≥0.60): Poor quality requiring attention
- **Critical** (<0.60): Critical issues requiring immediate action

### Compatibility Validation

The system ensures:
- **Schema Compatibility**: Required columns and keys are present
- **Type Compatibility**: Data types match expected formats
- **Index Compatibility**: Proper temporal indexing
- **Temporal Alignment**: Consistent time intervals
- **Format Compatibility**: Expected file formats

## Usage

### Basic Usage

```python
import asyncio
from src.training.comprehensive_pipeline_executor import ComprehensivePipelineExecutor

# Configuration
config = {
    "SYMBOL": "ETHUSDT",
    "EXCHANGE": "BINANCE",
    "TIMEFRAME": "1m",
    "DATA_DIR": "data_cache",
    "LOOKBACK_DAYS": 1095,
    "project_version": "1.0.0",
    "data_quality_monitor": {
        "enable_real_time_monitoring": True,
        "alert_threshold": 0.8,
        "auto_fix_enabled": False
    }
}

# Training input
training_input = {
    "symbol": "ETHUSDT",
    "exchange": "BINANCE",
    "timeframe": "1m",
    "data_dir": "data_cache",
    "lookback_days": 1095
}

# Execute pipeline
async def main():
    executor = ComprehensivePipelineExecutor(config)
    comprehensive_report = await executor.execute_pipeline_with_quality_monitoring(training_input)
    await executor.print_execution_summary(comprehensive_report)

asyncio.run(main())
```

### Advanced Usage with Custom Configuration

```python
# Custom configuration with specific quality thresholds
config = {
    "SYMBOL": "BTCUSDT",
    "EXCHANGE": "BINANCE",
    "TIMEFRAME": "5m",
    "DATA_DIR": "data_cache",
    "LOOKBACK_DAYS": 730,  # 2 years
    "project_version": "1.0.0",
    "data_quality_monitor": {
        "enable_real_time_monitoring": True,
        "alert_threshold": 0.85,  # Higher threshold
        "auto_fix_enabled": True,  # Enable auto-fix
        "quality_thresholds": {
            "excellent": 0.98,
            "good": 0.90,
            "acceptable": 0.80,
            "poor": 0.70,
            "critical": 0.60
        }
    }
}

# Execute with custom parameters
training_input = {
    "symbol": "BTCUSDT",
    "exchange": "BINANCE",
    "timeframe": "5m",
    "data_dir": "data_cache",
    "lookback_days": 730,
    "custom_parameters": {
        "feature_engineering": {
            "include_advanced_features": True,
            "regime_optimization": True
        },
        "quality_validation": {
            "strict_mode": True,
            "auto_correction": True
        }
    }
}
```

## Output and Reporting

### Execution Summary

The system provides comprehensive execution summaries including:

- **Overall Success**: Whether the entire pipeline completed successfully
- **Success Rate**: Percentage of steps completed successfully
- **Execution Time**: Total time taken for the pipeline
- **Quality Metrics**: Overall quality scores and rates
- **Step-by-Step Results**: Detailed results for each step

### Quality Reports

Detailed quality reports include:

- **Quality Summary**: Overall quality statistics
- **Compatibility Summary**: Compatibility validation results
- **Format Summary**: Format verification results
- **Index Summary**: Index validation results
- **Recent Issues**: Recent quality issues and warnings

### MLflow Integration

All results are automatically logged to MLflow with:

- **Metrics**: Quality scores, execution times, success rates
- **Artifacts**: Data files, reports, and configurations
- **Reports**: Detailed execution and quality reports
- **Metadata**: Training parameters and execution context

## Monitoring and Alerts

### Real-time Monitoring

The system provides real-time monitoring of:

- **Data Quality**: Continuous quality scoring
- **Compatibility**: Real-time compatibility validation
- **Performance**: Execution time and resource usage
- **Errors**: Error detection and reporting

### Quality Alerts

Automatic alerts are triggered for:

- **Low Quality Data**: Quality scores below threshold
- **Compatibility Issues**: Data format or schema issues
- **Performance Issues**: Slow execution or resource problems
- **Critical Errors**: Pipeline failures or data corruption

### Alert Handling

Alerts include:

- **Warning Alerts**: Quality scores between 0.6-0.8
- **Critical Alerts**: Quality scores below 0.6
- **Detailed Information**: Issues, warnings, and recommendations
- **Auto-logging**: All alerts logged to MLflow

## Configuration Options

### Data Quality Monitor Configuration

```python
"data_quality_monitor": {
    "enable_real_time_monitoring": True,
    "alert_threshold": 0.8,
    "auto_fix_enabled": False,
    "quality_thresholds": {
        "excellent": 0.95,
        "good": 0.85,
        "acceptable": 0.75,
        "poor": 0.60,
        "critical": 0.50
    }
}
```

### Step-specific Configuration

Each step can have custom configuration:

```python
"step_configs": {
    "step1": {
        "data_collection": {
            "validate_gaps": True,
            "auto_fill_gaps": False
        }
    },
    "step3": {
        "hmm_regime_discovery": {
            "n_components": 3,
            "covariance_type": "full"
        }
    },
    "step6": {
        "feature_engineering": {
            "include_advanced_features": True,
            "regime_optimization": True
        }
    }
}
```

## Error Handling and Recovery

### Comprehensive Error Handling

The system provides:

- **Graceful Degradation**: Continues execution when possible
- **Detailed Error Reporting**: Comprehensive error information
- **Automatic Recovery**: Attempts to recover from errors
- **Rollback Capability**: Can rollback to previous states

### Error Types Handled

- **Data Quality Errors**: Missing data, format issues
- **Compatibility Errors**: Schema mismatches, type issues
- **Performance Errors**: Timeout, memory issues
- **System Errors**: File system, network issues

## Performance Optimization

### Memory Management

- **Efficient Data Processing**: Chunked processing for large datasets
- **Memory Monitoring**: Real-time memory usage tracking
- **Automatic Cleanup**: Memory cleanup after each step
- **Resource Optimization**: Optimized resource usage

### Execution Optimization

- **Parallel Processing**: Parallel execution where possible
- **Caching**: Intelligent caching of intermediate results
- **Progress Tracking**: Real-time progress monitoring
- **Performance Metrics**: Detailed performance tracking

## Validation and Testing

### Comprehensive Validation

Each step includes:

- **Prerequisites Validation**: Checks required inputs
- **Execution Validation**: Validates step execution
- **Output Validation**: Validates step outputs
- **Quality Validation**: Validates data quality

### Testing Support

The system supports:

- **Unit Testing**: Individual component testing
- **Integration Testing**: Full pipeline testing
- **Quality Testing**: Quality validation testing
- **Performance Testing**: Performance benchmarking

## Best Practices

### Configuration Best Practices

1. **Set Appropriate Thresholds**: Configure quality thresholds based on your requirements
2. **Enable Real-time Monitoring**: Always enable real-time monitoring for production
3. **Configure Alerts**: Set up appropriate alert thresholds
4. **Use Auto-fix Carefully**: Enable auto-fix only when appropriate

### Execution Best Practices

1. **Monitor Execution**: Always monitor pipeline execution
2. **Review Quality Reports**: Regularly review quality reports
3. **Address Alerts**: Promptly address quality alerts
4. **Validate Results**: Always validate final results

### Data Quality Best Practices

1. **Regular Quality Checks**: Perform regular quality checks
2. **Monitor Trends**: Monitor quality trends over time
3. **Address Issues**: Promptly address quality issues
4. **Document Changes**: Document any quality-related changes

## Troubleshooting

### Common Issues

1. **Low Quality Scores**: Check data sources and preprocessing
2. **Compatibility Issues**: Verify data formats and schemas
3. **Performance Issues**: Check system resources and configuration
4. **Execution Failures**: Review error logs and step configurations

### Debugging

1. **Enable Debug Logging**: Enable detailed logging for debugging
2. **Review Quality Reports**: Check quality reports for issues
3. **Validate Inputs**: Verify input data and configurations
4. **Check Dependencies**: Ensure all dependencies are available

## Support and Maintenance

### Regular Maintenance

- **Update Dependencies**: Keep dependencies updated
- **Monitor Performance**: Regular performance monitoring
- **Review Configurations**: Periodic configuration review
- **Backup Data**: Regular data backup

### Support Resources

- **Documentation**: Comprehensive documentation available
- **Logs**: Detailed logging for troubleshooting
- **Reports**: Quality and execution reports
- **Monitoring**: Real-time monitoring and alerts

## Conclusion

The comprehensive training pipeline with enhanced data quality management provides a robust, reliable, and efficient system for executing the complete training pipeline while ensuring data quality, compatibility, and proper indexing at every step. The integrated monitoring and validation systems provide confidence in the pipeline results and help identify and address issues early in the process.