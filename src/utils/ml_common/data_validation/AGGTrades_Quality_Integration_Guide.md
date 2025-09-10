# Aggtrades Data Quality Verification - Integration Guide

This guide shows how to integrate the comprehensive aggtrades data quality verification system into your existing pipeline for regular data quality checks.

## Overview

The aggtrades quality verification system provides:
- **Timestamp gap detection** (configurable threshold, default 0.5s)
- **True duplicate detection and removal** (same timestamp + other columns)
- **Price sanity checks** (positive values, reasonable ranges, outlier detection)
- **Volume sanity checks** (positive values, reasonable ranges, outlier detection)
- **Comprehensive reporting and alerting**
- **Integration with existing validation framework**

## Quick Start

### Basic Usage

```python
from src.utils.ml_common.aggtrades_quality_verification import verify_aggtrades_quality

# Load your aggtrades data
data = pd.read_parquet('data/aggtrades_BTCUSDT.parquet')

# Basic quality verification
cleaned_data, report = verify_aggtrades_quality(data)

print(f"Quality score: {report.quality_score:.3f}")
print(f"Issues found: {len(report.issues)}")
```

### With Auto-Fix

```python
# Auto-fix common issues
cleaned_data, report = verify_aggtrades_quality(data, auto_fix=True)
```

## Integration Points

### 1. Data Loading Step

Add quality verification immediately after loading raw aggtrades data:

```python
def load_and_verify_aggtrades(exchange: str, symbol: str, data_dir: str = "data_cache/parquet"):
    """Load aggtrades data with quality verification."""
    from src.utils.ml_common.aggtrades_quality_verification import AggtradesQualityVerifier
    
    # Load data
    data = pd.read_parquet(f"{data_dir}/aggtrades_{exchange}_{symbol}_consolidated.parquet")
    
    # Quality verification configuration
    config = {
        'max_timestamp_gap_seconds': 0.5,
        'max_duplicate_ratio': 0.001,
        'duplicate_action': 'remove',
        'price_negative_action': 'fail',
        'volume_negative_action': 'fail'
    }
    
    # Verify quality
    verifier = AggtradesQualityVerifier(config)
    cleaned_data, report = verifier.verify_aggtrades_quality(data, auto_fix=True)
    
    # Log results
    logger.info(f"Data loading quality check: {report.quality_score:.3f}")
    
    # Export report
    verifier.export_quality_report(report, f"reports/quality_loading_{exchange}_{symbol}.json")
    
    return cleaned_data, report
```

### 2. Preprocessing Step

Verify quality after data preprocessing:

```python
def preprocess_aggtrades(data: pd.DataFrame, exchange: str, symbol: str):
    """Preprocess aggtrades data with quality verification."""
    from src.utils.ml_common.aggtrades_quality_verification import AggtradesQualityVerifier
    
    # Your preprocessing logic here
    processed_data = your_preprocessing_function(data)
    
    # Quality verification after preprocessing
    config = {
        'max_timestamp_gap_seconds': 0.5,
        'max_duplicate_ratio': 0.001,
        'duplicate_action': 'remove',
        'price_outlier_action': 'warn',
        'volume_outlier_action': 'warn'
    }
    
    verifier = AggtradesQualityVerifier(config)
    cleaned_data, report = verifier.verify_aggtrades_quality(processed_data)
    
    # Check for critical issues
    critical_issues = [issue for issue in report.issues if issue.severity == QualityIssueSeverity.CRITICAL]
    if critical_issues:
        raise ValueError(f"Critical quality issues after preprocessing: {len(critical_issues)}")
    
    return cleaned_data, report
```

### 3. Feature Engineering Step

Verify quality after feature engineering:

```python
def engineer_features(data: pd.DataFrame, exchange: str, symbol: str):
    """Engineer features with quality verification."""
    from src.utils.ml_common.aggtrades_quality_verification import AggtradesQualityVerifier
    
    # Your feature engineering logic here
    features_data = your_feature_engineering_function(data)
    
    # Quality verification for features
    config = {
        'max_timestamp_gap_seconds': 0.5,
        'max_duplicate_ratio': 0.001,
        'duplicate_action': 'remove',
        'price_outlier_action': 'warn',
        'volume_outlier_action': 'warn'
    }
    
    verifier = AggtradesQualityVerifier(config)
    cleaned_data, report = verifier.verify_aggtrades_quality(features_data)
    
    # Log quality metrics
    logger.info(f"Feature engineering quality: {report.quality_score:.3f}")
    
    return cleaned_data, report
```

### 4. Model Training Step

Verify quality before model training:

```python
def train_model(data: pd.DataFrame, exchange: str, symbol: str):
    """Train model with quality verification."""
    from src.utils.ml_common.aggtrades_quality_verification import AggtradesQualityVerifier
    
    # Final quality check before training
    config = {
        'max_timestamp_gap_seconds': 0.5,
        'max_duplicate_ratio': 0.001,
        'duplicate_action': 'remove',
        'price_negative_action': 'fail',
        'volume_negative_action': 'fail',
        'price_outlier_action': 'warn',
        'volume_outlier_action': 'warn'
    }
    
    verifier = AggtradesQualityVerifier(config)
    cleaned_data, report = verifier.verify_aggtrades_quality(data)
    
    # Ensure high quality before training
    if report.quality_score < 0.8:
        raise ValueError(f"Data quality too low for training: {report.quality_score:.3f}")
    
    # Train your model
    model = your_training_function(cleaned_data)
    
    return model, report
```

## Configuration Options

### Quality Thresholds

```python
config = {
    # Timestamp gap detection
    'max_timestamp_gap_seconds': 0.5,  # Maximum allowed gap in seconds
    
    # Duplicate detection
    'max_duplicate_ratio': 0.001,      # Maximum allowed duplicate ratio
    
    # Outlier detection
    'price_outlier_threshold': 5.0,    # Z-score threshold for price outliers
    'volume_outlier_threshold': 5.0,   # Z-score threshold for volume outliers
    
    # Value ranges
    'min_price': 0.000001,             # Minimum allowed price
    'max_price': 1000000.0,            # Maximum allowed price
    'min_volume': 0.0,                 # Minimum allowed volume
    'max_volume': 1e12,                # Maximum allowed volume
}
```

### Actions for Different Issue Types

```python
config = {
    # Actions: 'log_only', 'warn', 'remove', 'fail'
    'timestamp_gap_action': 'warn',      # What to do with timestamp gaps
    'duplicate_action': 'remove',        # What to do with duplicates
    'price_negative_action': 'fail',     # What to do with negative prices
    'price_outlier_action': 'warn',      # What to do with price outliers
    'volume_negative_action': 'fail',    # What to do with negative volumes
    'volume_outlier_action': 'warn',     # What to do with volume outliers
}
```

## Monitoring and Alerting

### Regular Quality Monitoring

```python
def setup_quality_monitoring():
    """Setup regular quality monitoring."""
    from src.utils.ml_common.aggtrades_quality_verification import AggtradesQualityVerifier
    
    # Monitoring configuration
    config = {
        'max_timestamp_gap_seconds': 0.5,
        'max_duplicate_ratio': 0.001,
        'duplicate_action': 'remove',
        'price_negative_action': 'fail',
        'volume_negative_action': 'fail',
        'price_outlier_action': 'warn',
        'volume_outlier_action': 'warn'
    }
    
    verifier = AggtradesQualityVerifier(config)
    
    def monitor_quality(data: pd.DataFrame, step_name: str):
        """Monitor quality for a specific step."""
        cleaned_data, report = verifier.verify_aggtrades_quality(data, auto_fix=True)
        
        # Check for critical issues
        critical_issues = [issue for issue in report.issues if issue.severity == QualityIssueSeverity.CRITICAL]
        error_issues = [issue for issue in report.issues if issue.severity == QualityIssueSeverity.ERROR]
        
        if critical_issues or error_issues:
            # Send alert
            alert_message = f"Quality issues in {step_name}: {len(critical_issues)} critical, {len(error_issues)} errors"
            send_alert(alert_message)
        
        return cleaned_data, report
    
    return monitor_quality
```

### Quality Dashboard Integration

```python
def generate_quality_dashboard(reports: List[QualityReport]):
    """Generate quality dashboard from multiple reports."""
    import matplotlib.pyplot as plt
    
    # Extract quality scores
    scores = [report.quality_score for report in reports]
    timestamps = [report.timestamp for report in reports]
    
    # Create quality trend chart
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, scores, marker='o')
    plt.title('Data Quality Trend')
    plt.ylabel('Quality Score')
    plt.xlabel('Time')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()
    
    # Summary statistics
    print(f"Average quality score: {np.mean(scores):.3f}")
    print(f"Minimum quality score: {np.min(scores):.3f}")
    print(f"Quality score std: {np.std(scores):.3f}")
```

## Integration with Existing Validation Framework

### Using with MLValidationSuite

```python
from src.utils.ml_common.validation_utils import MLValidationSuite
from src.utils.ml_common.aggtrades_quality_verification import AggtradesQualityVerifier

def validate_step_with_aggtrades_quality(config: Dict, data: pd.DataFrame):
    """Validate step with aggtrades quality verification."""
    
    # Standard ML validation
    validation_suite = MLValidationSuite()
    validation_result = await validation_suite.validate_step_execution(config, data)
    
    # Aggtrades-specific quality verification
    quality_config = {
        'max_timestamp_gap_seconds': 0.5,
        'max_duplicate_ratio': 0.001,
        'duplicate_action': 'remove'
    }
    
    quality_verifier = AggtradesQualityVerifier(quality_config)
    cleaned_data, quality_report = quality_verifier.verify_aggtrades_quality(data, auto_fix=True)
    
    # Combine results
    combined_result = {
        'ml_validation': validation_result,
        'quality_verification': {
            'quality_score': quality_report.quality_score,
            'issues_found': len(quality_report.issues),
            'cleaned_data_shape': cleaned_data.shape
        }
    }
    
    return cleaned_data, combined_result
```

## Best Practices

### 1. Regular Quality Checks

- Run quality verification at each major pipeline step
- Set up automated quality monitoring
- Track quality trends over time

### 2. Configuration Management

- Use different configurations for different steps
- Store configurations in version control
- Document configuration changes

### 3. Error Handling

- Handle critical quality issues appropriately
- Implement fallback strategies
- Log all quality issues for analysis

### 4. Performance Considerations

- Use auto-fix for non-critical issues
- Batch process multiple datasets
- Cache quality reports for large datasets

### 5. Reporting

- Export quality reports for each step
- Generate quality dashboards
- Set up alerts for quality degradation

## Example Pipeline Integration

```python
def complete_aggtrades_pipeline(exchange: str, symbol: str):
    """Complete aggtrades pipeline with quality verification."""
    
    # Step 1: Load and verify
    data, load_report = load_and_verify_aggtrades(exchange, symbol)
    
    # Step 2: Preprocess and verify
    processed_data, preprocess_report = preprocess_aggtrades(data, exchange, symbol)
    
    # Step 3: Engineer features and verify
    features_data, features_report = engineer_features(processed_data, exchange, symbol)
    
    # Step 4: Train model and verify
    model, training_report = train_model(features_data, exchange, symbol)
    
    # Generate overall quality summary
    all_reports = [load_report, preprocess_report, features_report, training_report]
    quality_summary = generate_quality_summary(all_reports)
    
    return model, quality_summary
```

## Troubleshooting

### Common Issues

1. **High timestamp gap count**: Check data collection process
2. **Many duplicates**: Verify deduplication logic
3. **Negative prices/volumes**: Check data source integrity
4. **High outlier count**: Review outlier detection thresholds

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger('src.utils.ml_common.aggtrades_quality_verification').setLevel(logging.DEBUG)

# Run with detailed output
cleaned_data, report = verify_aggtrades_quality(data, auto_fix=True)
```

This integration guide provides comprehensive coverage of how to integrate the aggtrades quality verification system into your existing pipeline. The system is designed to be flexible, configurable, and easy to integrate with your current workflow.