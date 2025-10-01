# Regime Data Splitting Package

This package provides comprehensive regime data splitting functionality with enhanced error handling, validation, and reporting capabilities.

## 📁 Package Structure

```
regime_data_splitting/
├── __init__.py          # Package initialization and exports
├── component.py         # Main regime data splitting component
├── nas_tas_regime_data_splitting.py  # NAS/TAS regime data splitting with clustering integration
├── main.py             # Main step implementation with standardized data quality management
├── validator.py        # Comprehensive validation framework
└── README.md           # This file
```

## 🚀 Key Features

### **Silent Failure Prevention**
- Explicit dependency validation with fail-fast behavior
- Multi-stage validation checkpoints throughout the process
- Enhanced error context with detailed debugging information
- Graceful degradation with fallback mechanisms

### **Enhanced Reporting**
- Comprehensive metrics (10+ detailed metrics including quality scores)
- Data quality scoring (0-1 score based on data quality factors)
- Regime continuity scoring (0-1 score based on transition frequency)
- Actionable recommendations for continuous improvement
- Real-time execution tracking with detailed progress logging

### **Robust Error Handling**
- Explicit failure modes with clear error messages
- Multiple validation stages with detailed reporting
- Graceful degradation with proper logging
- Comprehensive error context for debugging

## 📦 Components

### **1. RegimeDataSplittingComponent (`component.py`)**
Main regime data splitting component with comprehensive error handling and reporting.

**Key Features:**
- Dependency validation with explicit error reporting
- Multi-stage validation (input, data, result validation)
- Comprehensive metrics tracking
- Quality scoring system
- Actionable recommendations

**Usage:**
```python
from src.training.steps.market_analysis.regime_data_splitting import RegimeDataSplittingComponent

component = RegimeDataSplittingComponent(config)
result = await component.execute(data, pipeline_state)

# Access enhanced artifacts
regime_data = result.artifacts['regime_data_splitting_result']
report = result.artifacts['regime_splitting_report']
validation = result.artifacts['regime_validation_results']
```

### **2. NasTasRegimeDataSplitting (`nas_tas_regime_data_splitting.py`)**
Enhanced implementation with HMM ML model integration and advanced features.

**Key Features:**
- HMM ML model integration for regime tagging
- Enhanced input validation with detailed error checking
- Execution metrics tracking with real-time monitoring
- Enhanced data quality assessment
- Improved error handling with graceful degradation

**Usage:**
```python
from src.training.steps.market_analysis.regime_data_splitting import NasTasRegimeDataSplitting

splitter = NasTasRegimeDataSplitting(config)
result = await splitter.execute(training_input, pipeline_state)

# Access execution metrics
metrics = result['execution_metrics']
data_quality_score = metrics['data_quality_score']
recommendations = metrics['recommendations']
```

### **3. RegimeDataSplittingStep (`main.py`)**
Main step implementation with standardized data quality management.

**Key Features:**
- Standardized data quality management
- Performance optimization components
- Memory management and optimization
- Comprehensive function monitoring

### **4. Step4RegimeDataSplittingValidator (`validator.py`)**
Comprehensive validation framework for regime data splitting.

**Key Features:**
- File existence validation
- DataFrame quality validation
- Statistics file validation
- Prerequisites and outputs validation

**Usage:**
```python
from src.training.steps.market_analysis.regime_data_splitting import run_validator

validation_result = await run_validator(training_input, pipeline_state)
```

## 📊 Data Classes

### **RegimeSplittingMetrics**
Comprehensive metrics for regime splitting operations:
- `total_data_points`: Number of data points processed
- `regime_count`: Number of regimes detected
- `regime_distribution`: Distribution of data across regimes
- `processing_time_seconds`: Execution time
- `data_quality_score`: 0-1 quality score
- `regime_continuity_score`: 0-1 continuity score
- `validation_checks_passed/failed`: Validation results
- `warnings_count/errors_count`: Issue tracking

### **RegimeSplittingReport**
Comprehensive report for regime splitting operations:
- `status`: Current operation status
- `metrics`: Detailed execution metrics
- `execution_summary`: High-level summary
- `validation_results`: Validation check results
- `warnings/errors`: Issue lists
- `recommendations`: Actionable suggestions
- `timestamp`: Execution timestamp

## 🔧 Quality Scoring System

### **Data Quality Score (0-1)**
- **Null Values (30% weight)**: Penalty for missing data
- **Duplicate Rows (20% weight)**: Penalty for duplicate entries
- **Infinite Values (30% weight)**: Penalty for infinite/invalid values
- **Invalid Prices (20% weight)**: Penalty for zero/negative prices

### **Regime Continuity Score (0-1)**
- Based on regime transition frequency
- Higher score = fewer transitions (more stable regimes)
- Penalty for excessive regime switching

## 🎯 Recommendations System

The package provides actionable recommendations based on execution results:

- **Data Quality**: Suggestions for improving data quality when score < 0.8
- **Regime Diversity**: Recommendations for regime discovery parameters
- **Continuity**: Suggestions for smoothing parameters when transitions are frequent
- **Performance**: Optimization suggestions for high processing times
- **Memory**: Recommendations for streaming processing when memory usage is high

## 📈 Expected Benefits

### **Reliability**
- Eliminated silent failures with explicit validation
- Comprehensive validation prevents invalid results
- Robust error handling for edge cases

### **Observability**
- Detailed metrics for all operations
- Quantified assessment of data and regime quality
- Specific recommendations for improvement
- Real-time monitoring of execution progress

### **Maintainability**
- Enhanced existing code without breaking changes
- Well-organized, documented structure
- Consistent error handling patterns
- Comprehensive logging for debugging

### **Performance**
- Streamlined execution path
- Better memory usage tracking
- Fast validation with early failure detection
- Performance insights for optimization

## 🔄 Migration Guide

### **From Old Locations**
```python
# Old imports
from src.training.steps.market_analysis.components.regime_data_splitting import RegimeDataSplittingComponent
from src.training.steps.market_analysis.regime_data_splitting.nas_tas_regime_data_splitting import NasTasRegimeDataSplitting

# New imports
from src.training.steps.market_analysis.regime_data_splitting import RegimeDataSplittingComponent, NasTasRegimeDataSplitting
```

### **Enhanced Features**
The moved components now include:
- Comprehensive error handling and validation
- Enhanced reporting with quality scores
- Actionable recommendations
- Silent failure prevention
- Real-time execution tracking

## 📝 Version History

- **v1.0.0**: Initial package creation with enhanced error handling and reporting
- Comprehensive regime data splitting with silent failure prevention
- Enhanced reporting with quality scores and recommendations
- Multi-stage validation and robust error handling