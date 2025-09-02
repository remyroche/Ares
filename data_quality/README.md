# Unified Data Quality Orchestrator

This directory contains a comprehensive data quality orchestration system that unites all quality checking scripts and modules from the project into a single, easy-to-use interface.

## Overview

The `UnifiedQualityOrchestrator` class provides a unified entry point for comprehensive data quality assessment, validation, and monitoring. It integrates functionality from various data quality modules throughout the project:

- **Data validation and schema enforcement** from `src/utils/data_quality_framework.py`
- **Enhanced data quality validation** from `src/utils/enhanced_data_quality_validator.py`
- **Data quality monitoring** from `src/training/data_quality_monitor.py`
- **Multicollinearity analysis** from `scripts/assess_data_quality.py`
- **Feature-specific validation** from `feature_specific_validation.py`
- **Temporal data validation** from various temporal analysis modules
- **Dependency graph analysis** from `data_quality/mapping/dependency_graph.py`

## Features

### 🔍 Core Quality Validation
- **DataFrame Quality Checks**: NaN values, infinite values, constant columns, duplicates, data types
- **Schema Validation**: Enforces required columns, data types, and constraints for different data formats
- **Outlier Detection**: Integrates with enhanced outlier handler for comprehensive outlier analysis

### 📊 Advanced Analysis
- **Multicollinearity Detection**: VIF analysis and correlation matrix examination
- **Label Imbalance Analysis**: Comprehensive analysis for classification datasets
- **Feature Redundancy Analysis**: Identifies highly correlated features
- **Temporal Data Validation**: Timestamp format, gaps, duplicates, and future timestamps

### 🎯 Quality Metrics
- **Completeness**: Data completeness assessment
- **Consistency**: Data consistency validation
- **Validity**: Schema and constraint validation
- **Timeliness**: Temporal data quality
- **Uniqueness**: Duplicate detection
- **Accuracy**: Outlier and anomaly detection

### 📈 Reporting and Monitoring
- **Comprehensive Reports**: JSON-formatted detailed quality reports
- **Quality Scoring**: Overall quality assessment with recommendations
- **Issue Tracking**: Categorized issues by severity level
- **Recommendations**: Actionable insights for data quality improvement

## Installation and Dependencies

### Required Dependencies
```bash
pip install pandas numpy scikit-learn networkx
```

### Optional Dependencies
- `src/utils/enhanced_outlier_handler` - For advanced outlier detection
- `src/utils/logger` - For project-specific logging
- `src/utils/enhanced_mlflow_integration` - For MLflow integration

## Usage

### Command Line Interface

The orchestrator supports both single file and directory analysis:

#### Single File Analysis
```bash
# Basic usage
python data_quality/unified_quality_orchestrator.py --data_path /path/to/your/data.csv

# With custom context and output
python data_quality/unified_quality_orchestrator.py \
    --data_path /path/to/your/data.csv \
    --context "ETHUSDT 1h features" \
    --output quality_report.json

# With custom thresholds
python data_quality/unified_quality_orchestrator.py \
    --data_path /path/to/your/data.csv \
    --thresholds custom_thresholds.json
```

#### Directory Analysis
```bash
# Analyze all data files in a directory (auto-detect mode)
python data_quality/unified_quality_orchestrator.py --data_path /path/to/data/directory

# Force directory mode
python data_quality/unified_quality_orchestrator.py \
    --data_path /path/to/data/directory \
    --mode directory

# Quick directory scan (no full analysis)
python data_quality/unified_quality_orchestrator.py \
    --data_path /path/to/data/directory \
    --mode directory \
    --quick_scan

# Analyze specific file types in directory
python data_quality/unified_quality_orchestrator.py \
    --data_path /path/to/data/directory \
    --mode directory \
    --file_pattern "*.csv"

# Non-recursive directory analysis
python data_quality/unified_quality_orchestrator.py \
    --data_path /path/to/data/directory \
    --mode directory \
    --recursive false
```

#### Command Line Options
- `--data_path`: Path to data file or directory (required)
- `--context`: Context description for the data
- `--output`: Output file for the report
- `--thresholds`: JSON file with custom thresholds
- `--mode`: Analysis mode: `file`, `directory`, or `auto` (default: `auto`)
- `--recursive`: Search subdirectories recursively (default: `true`)
- `--file_pattern`: File pattern for directory analysis (default: `*`)
- `--quick_scan`: Quick directory scan without full analysis

### Python API

#### Single File Analysis
```python
from data_quality.unified_quality_orchestrator import UnifiedQualityOrchestrator
import pandas as pd

# Initialize orchestrator
orchestrator = UnifiedQualityOrchestrator()

# Load your data
data = pd.read_csv("your_data.csv")

# Generate comprehensive quality report
report = orchestrator.generate_comprehensive_report(data, "ETHUSDT 1h features")

# Save report
output_file = orchestrator.save_report(report, "my_quality_report.json")

# Access specific analysis results
quality_validation = report["quality_validation"]
multicollinearity = report["multicollinearity_analysis"]
feature_redundancy = report["feature_redundancy_analysis"]
temporal_validation = report["temporal_validation"]

# Get summary
summary = report["summary"]
print(f"Overall Quality: {summary['overall_quality']}")
print(f"Critical Issues: {summary['critical_issues']}")
print(f"Recommendations: {summary['recommendations']}")
```

#### Directory Analysis
```python
# Analyze all data files in a directory
directory_report = orchestrator.analyze_directory("/path/to/data/directory")

# Quick directory scan
scan_summary = orchestrator.get_directory_summary("/path/to/data/directory")

# Analyze specific file types
directory_report = orchestrator.analyze_directory(
    "/path/to/data/directory", 
    file_pattern="*.csv", 
    recursive=True
)

# Access directory summary
summary = directory_report["summary"]
print(f"Total files: {summary['total_files']}")
print(f"Success rate: {summary['success_rate']:.1%}")
print(f"Overall Quality: {summary['overall_quality']}")

# Access individual file results
for file_path, result in directory_report["file_results"].items():
    if "error" not in result:
        file_summary = result["summary"]
        print(f"{file_path}: {file_summary['overall_quality']}")
```

#### Batch File Analysis
```python
# Analyze multiple specific files
file_paths = ["file1.csv", "file2.csv", "file3.parquet"]
batch_report = orchestrator.analyze_file_batch(file_paths)

# Access batch summary
summary = batch_report["summary"]
print(f"Batch success rate: {summary['success_rate']:.1%}")
print(f"Total critical issues: {summary['critical_issues_total']}")
```

### Custom Thresholds

Create a JSON file with custom thresholds:

```json
{
    "max_nan_ratio": 0.05,
    "max_infinite_count": 10,
    "min_unique_values": 5,
    "max_constant_ratio": 0.9,
    "max_gap_hours": 24,
    "price_tolerance": 0.0001,
    "volume_tolerance": 0.0001,
    "max_correlation_threshold": 0.9,
    "min_feature_count": 50,
    "vif_threshold": 10.0
}
```

## Supported Data Formats

The orchestrator supports multiple data formats:

- **CSV** (`.csv`)
- **Parquet** (`.parquet`)
- **JSON** (`.json`)

## Data Schema Validation

The orchestrator includes predefined schemas for common data types:

### Klines Schema
- Required columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`
- Data types: All numeric except timestamp (int64)
- Constraints: All values must be positive

### Features Schema
- Required columns: `timestamp`
- Data types: `timestamp` as int64
- Constraints: Timestamp must be positive

### Labels Schema
- Required columns: `timestamp`, `label`
- Data types: `timestamp` as int64, `label` as int64
- Constraints: Both must be positive

## Quality Levels

The system categorizes data quality into five levels:

1. **EXCELLENT** (0 critical issues)
2. **GOOD** (1-2 critical issues)
3. **ACCEPTABLE** (3-5 critical issues)
4. **POOR** (6-10 critical issues)
5. **CRITICAL** (>10 critical issues)

## Issue Severity Levels

Issues are categorized by severity:

- **CRITICAL**: Data integrity problems that must be fixed
- **HIGH**: Significant quality issues requiring attention
- **MEDIUM**: Moderate issues that should be addressed
- **LOW**: Minor issues that can be monitored

## Output Reports

The orchestrator generates comprehensive JSON reports containing:

```json
{
    "context": "Data description",
    "timestamp": "2024-01-01T00:00:00",
    "data_shape": [1000, 50],
    "quality_validation": {
        "passed": true,
        "issue_count": 0,
        "warning_count": 2,
        "metrics": {...},
        "issues": [],
        "warnings": [...]
    },
    "multicollinearity_analysis": {
        "vif_scores": {...},
        "high_vif_features": [],
        "correlation_matrix": {...},
        "high_correlation_pairs": []
    },
    "feature_redundancy_analysis": {
        "redundant_pairs": [],
        "redundancy_ratio": 0.1,
        "recommendations": [...]
    },
    "temporal_validation": {
        "passed": true,
        "issue_count": 0,
        "metrics": {...}
    },
    "summary": {
        "overall_quality": "excellent",
        "critical_issues": 0,
        "recommendations": [...]
    }
}
```

## Integration with Existing Systems

The orchestrator is designed to integrate seamlessly with existing project components:

- **MLflow Integration**: Logs quality metrics and reports to MLflow
- **Pipeline Standards**: Follows project pipeline standards and conventions
- **Logging System**: Integrates with project logging infrastructure
- **Outlier Handling**: Uses enhanced outlier handler for advanced anomaly detection

## Examples

### Example 1: Basic Quality Check
```python
from data_quality.unified_quality_orchestrator import UnifiedQualityOrchestrator
import pandas as pd

# Load data
data = pd.read_csv("features_ETHUSDT_1h.csv")

# Check quality
orchestrator = UnifiedQualityOrchestrator()
quality_result = orchestrator.validate_dataframe_quality(data, "ETHUSDT 1h features")

if quality_result.passed:
    print("✅ Data quality check passed")
else:
    print(f"❌ Data quality check failed: {len(quality_result.issues)} issues")
```

### Example 2: Multicollinearity Analysis
```python
# Analyze multicollinearity
multicollinearity = orchestrator.analyze_multicollinearity(data)

if multicollinearity["high_vif_features"]:
    print(f"⚠️ High VIF features: {multicollinearity['high_vif_features']}")
else:
    print("✅ No multicollinearity issues detected")
```

### Example 3: Label Imbalance Analysis
```python
# Analyze label imbalance
if "label" in data.columns:
    labels = data["label"]
    imbalance_analysis = orchestrator.analyze_label_imbalance(labels)
    
    print(f"Imbalance Level: {imbalance_analysis['imbalance_level']}")
    print(f"Imbalance Ratio: {imbalance_analysis['imbalance_ratio']:.2f}")
    
    for rec in imbalance_analysis["recommendations"]:
        print(f"💡 {rec}")
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all required dependencies are installed
2. **Memory Issues**: For large datasets, consider processing in chunks
3. **Timestamp Issues**: Verify timestamp column format and timezone consistency
4. **Schema Mismatches**: Check that your data matches expected schemas

### Performance Tips

- Use Parquet format for large datasets
- Process data in chunks for very large files
- Adjust thresholds based on your specific use case
- Use custom thresholds for domain-specific requirements

## Contributing

To extend the orchestrator with new quality checks:

1. Add new validation methods to the `UnifiedQualityOrchestrator` class
2. Update the `generate_comprehensive_report` method to include new analyses
3. Add new metrics to the `QualityThresholds` dataclass if needed
4. Update this documentation with new features

## License

This module is part of the main project and follows the same license terms.