# Triple Barrier Labeling Package

This package provides a unified, robust implementation of triple barrier labeling for the market analysis pipeline.

## 🚀 Features

- **Unified Implementation**: Single, consolidated implementation replacing multiple overlapping files
- **Explicit Error Handling**: No silent failures - all errors are properly raised and reported
- **Comprehensive Validation**: Data quality validation with configurable thresholds
- **Enhanced Reporting**: Detailed execution reports with performance metrics
- **Hardware Optimization**: M1/M2/M3 Mac optimizations with proper fallbacks
- **Regime-Aware Support**: HMM regime integration for sophisticated labeling
- **Performance Monitoring**: Real-time progress tracking and metrics collection

## 📦 Package Structure

```
triple_barrier_labeling/
├── __init__.py              # Package initialization and public API
├── unified_labeler.py       # Main implementation
├── test_unified_labeler.py  # Comprehensive test suite
└── README.md               # This file
```

## 🔧 Quick Start

### Basic Usage

```python
from src.training.steps.market_analysis.triple_barrier_labeling import apply_triple_barrier_labeling

# Apply triple barrier labeling
result = apply_triple_barrier_labeling(data)

if result.success:
    labeled_data = result.labeled_data
    print(f"Generated {result.total_labels_generated} labels")
    print(f"Quality score: {result.data_quality_score:.2%}")
else:
    print(f"Labeling failed: {result.error_message}")
```

### Advanced Configuration

```python
from src.training.steps.market_analysis.triple_barrier_labeling import create_triple_barrier_labeler

# Create labeler with custom configuration
labeler = create_triple_barrier_labeler(
    profit_take_multiplier=0.003,      # 0.3% profit take
    stop_loss_multiplier=0.002,        # 0.2% stop loss
    time_barrier_minutes=45,           # 45-minute time barrier
    max_lookahead=150,                 # 150-period lookahead
    transaction_cost=0.001,            # 0.1% transaction cost
    binary_classification=True,        # Binary classification
    regime_aware=True,                 # Enable regime-aware labeling
    fail_on_validation_error=True,     # Fail fast on validation errors
    enable_hardware_optimizations=True # Enable hardware optimizations
)

result = labeler.apply_labeling(data)
```

## 📊 Main Classes

### UnifiedTripleBarrierLabeler

The main labeling class that orchestrates the entire triple barrier labeling process.

**Key Methods:**
- `apply_labeling(data)`: Main entry point for labeling
- `_validate_input_data(data)`: Comprehensive data validation
- `_apply_triple_barrier_labeling(data)`: Core labeling logic
- `_optimize_data(data)`: Hardware optimization

### TripleBarrierConfig

Configuration management with comprehensive validation.

**Key Parameters:**
- `profit_take_multiplier`: Profit take threshold (default: 0.002)
- `stop_loss_multiplier`: Stop loss threshold (default: 0.001)
- `time_barrier_minutes`: Time barrier in minutes (default: 30)
- `max_lookahead`: Maximum lookahead periods (default: 100)
- `transaction_cost`: Transaction cost percentage (default: 0.0008)
- `binary_classification`: Use binary classification (default: True)
- `regime_aware`: Enable regime-aware labeling (default: True)
- `fail_on_validation_error`: Fail fast on validation errors (default: True)

### TripleBarrierResult

Comprehensive execution result with detailed metrics.

**Key Attributes:**
- `success`: Execution success status
- `labeled_data`: Labeled DataFrame (if successful)
- `error_message`: Error message (if failed)
- `execution_duration`: Execution time in seconds
- `total_labels_generated`: Number of labels created
- `label_distribution`: Distribution of label values
- `data_quality_score`: Data quality score (0-1)
- `validation_passed`: Validation success status
- `validation_warnings`: List of validation warnings
- `performance_metrics`: Hardware optimization metrics

## 🛡️ Error Handling

The package provides explicit error handling with custom exception classes:

- `TripleBarrierError`: Base exception class
- `ValidationError`: Data validation failures
- `ConfigurationError`: Invalid configuration parameters
- `HardwareOptimizationError`: Hardware optimization failures
- `DataQualityError`: Insufficient data quality

## 📈 Validation Framework

### DataValidator

Comprehensive data validation with configurable thresholds:

- **OHLC Validation**: Price relationship validation
- **Missing Data Detection**: Configurable missing data thresholds
- **Regime Validation**: Regime data integrity checks
- **Quality Scoring**: Data quality assessment

### Validation Features

- Required column checking
- OHLC relationship validation
- Missing data ratio validation
- Non-positive price detection
- Regime distribution analysis

## ⚡ Performance Optimization

### HardwareManager

Unified hardware optimization with proper fallbacks:

- **M1 CPU Optimization**: Parallel processing acceleration
- **M1 Memory Optimization**: Memory usage optimization
- **M1 GPU Acceleration**: GPU acceleration support
- **Numba JIT**: Just-in-time compilation for critical loops

### Performance Features

- Automatic hardware detection
- Graceful fallback on optimization failures
- Memory usage monitoring
- Execution time tracking

## 🧪 Testing

The package includes a comprehensive test suite with 30+ test methods:

- **Configuration Tests**: Parameter validation
- **Data Validation Tests**: Input data validation
- **Core Functionality Tests**: Labeling logic
- **Error Handling Tests**: Exception handling
- **Performance Tests**: Scalability and timing

Run tests with:
```bash
python -m pytest src/training/steps/market_analysis/triple_barrier_labeling/test_unified_labeler.py -v
```

## 📋 Migration Guide

### From Old Implementation

**Old Code:**
```python
from src.training.steps.market_analysis.triple_barrier_labeling import MarketAnalysisTripleBarrierLabeling

labeler = MarketAnalysisTripleBarrierLabeling(config)
labeled_data = labeler.apply_triple_barrier_labeling(data)
```

**New Code:**
```python
from src.training.steps.market_analysis.triple_barrier_labeling import UnifiedTripleBarrierLabeler

labeler = UnifiedTripleBarrierLabeler(config)
result = labeler.apply_labeling(data)
labeled_data = result.labeled_data if result.success else None
```

### Key Changes

1. **Class Name**: `MarketAnalysisTripleBarrierLabeling` → `UnifiedTripleBarrierLabeler`
2. **Method Name**: `apply_triple_barrier_labeling()` → `apply_labeling()`
3. **Return Type**: DataFrame → `TripleBarrierResult` object
4. **Error Handling**: Silent failures → Explicit exceptions
5. **Reporting**: Basic logging → Comprehensive execution reports

## 🔮 Future Enhancements

- Real-time monitoring dashboard
- Advanced GPU acceleration
- Machine learning integration
- Enhanced validation metrics
- Performance profiling tools

## 📚 API Reference

See the package `__init__.py` file for the complete public API.

## 🤝 Contributing

When contributing to this package:

1. Follow the existing code structure
2. Add comprehensive tests for new features
3. Update documentation
4. Ensure backward compatibility
5. Follow error handling patterns

## 📄 License

This package is part of the market analysis pipeline and follows the same licensing terms.