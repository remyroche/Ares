# Standardized Artifact Naming Guide

## Overview

This guide explains the standardized artifact naming pattern used throughout the enhanced training manager pipeline to ensure consistent, traceable, and well-organized MLflow artifacts.

## Naming Pattern

All artifacts follow the standardized pattern:

```
exchange_token_date_hourminute_NumberOfStep_Artifact
```

### Pattern Components

- **exchange**: Trading exchange (e.g., "BINANCE", "COINBASE")
- **token**: Trading symbol/token (e.g., "ETHUSDT", "BTCUSDT")
- **date**: Date in YYYYMMDD format (e.g., "20241201")
- **hourminute**: Time in HHMM format (e.g., "1430" for 2:30 PM)
- **NumberOfStep**: Step number without "step" prefix (e.g., "3", "6", "9", "21")
- **Artifact**: Artifact type/name (e.g., "composite_clusters", "features_train", "hmm_model")

### Examples

```
BINANCE_ETHUSDT_20241201_1430_3_composite_clusters.parquet
BINANCE_ETHUSDT_20241201_1430_6_features_train.parquet
BINANCE_ETHUSDT_20241201_1430_9_hmm_model.pkl
BINANCE_ETHUSDT_20241201_1430_21_training_summary.json
```

## Implementation

### 1. Utility Functions

The standardized naming is implemented through utility functions in `src/utils/enhanced_mlflow_integration.py`:

#### `generate_standardized_artifact_name()`
```python
def generate_standardized_artifact_name(
    exchange: str,
    token: str,
    step_number: str,
    artifact_type: str,
    extension: str = "",
    timestamp: Optional[datetime] = None
) -> str:
    """Generate standardized artifact name following the pattern."""
```

#### `log_step_dataframe_with_standardized_name()`
```python
def log_step_dataframe_with_standardized_name(
    config: Dict[str, Any],
    step_name: str,
    df: pd.DataFrame,
    artifact_type: str,
    run_id: Optional[str] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Log a DataFrame with standardized naming pattern."""
```

#### `log_step_artifact_with_standardized_name()`
```python
def log_step_artifact_with_standardized_name(
    config: Dict[str, Any],
    step_name: str,
    artifact_path: str,
    artifact_type: str,
    run_id: Optional[str] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Log an artifact with standardized naming pattern."""
```

#### `log_step_report()`
```python
def log_step_report(
    config: Dict[str, Any],
    step_name: str,
    report_data: Dict[str, Any],
    report_type: str,
    run_id: Optional[str] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Log a step report with standardized naming pattern."""
```

### 2. Step Integration

Each step should use these functions to log artifacts with standardized naming:

#### Step 3: HMM Regime Discovery
```python
# Log composite clusters DataFrame
artifact_name = log_step_dataframe_with_standardized_name(
    config=self.config,
    step_name="step3_hmm_regime_discovery",
    df=composite_df,
    artifact_type="composite_clusters",
    additional_metadata={
        "artifact_type": "composite_clusters",
        "dataframe_shape": list(composite_df.shape),
        "regime_count": len(composite_df.get("composite_cluster_id", []).unique()),
        "timeframe": timeframe,
    }
)

# Log regime discovery report
report_name = log_step_report(
    config=self.config,
    step_name="step3_hmm_regime_discovery",
    report_data=report_data,
    report_type="regime_discovery_report",
    additional_metadata={
        "hmm_states": regime_results["metrics"].get("hmm_states", 0),
        "composite_clusters": regime_results["metrics"].get("composite_clusters", 0),
    }
)
```

#### Step 6: Feature Engineering
```python
# Log training features DataFrame
train_artifact_name = log_step_dataframe_with_standardized_name(
    config=config,
    step_name="step6_feature_engineering",
    df=features_result["features_train"],
    artifact_type="features_train",
    additional_metadata={
        "artifact_type": "training_features",
        "feature_count": len(features_result["features_train"].columns),
        "sample_count": len(features_result["features_train"]),
        "timeframe": timeframe,
    }
)

# Log feature engineering report
report_name = log_step_report(
    config=config,
    step_name="step6_feature_engineering",
    report_data=report_data,
    report_type="feature_engineering_report",
    additional_metadata={
        "total_features": len(features_result["features_train"].columns),
        "feature_categories": len(features_result["metadata"].get("feature_categories", {})),
        "timeframe": timeframe,
    }
)
```

#### Step 9: HMM-Based Training
```python
# Log training summary with standardized naming
summary_artifact_name = log_step_artifact_with_standardized_name(
    config=self.config,
    step_name="step9_hmm_based_training",
    artifact_path=summary_path,
    artifact_type="training_summary",
    additional_metadata={
        "models_trained": len(training_results),
        "timeframes": list(training_results.keys()),
        "summary_type": "comprehensive_training_summary",
    }
)

# Log comprehensive training report
report_name = log_step_report(
    config=self.config,
    step_name="step9_hmm_based_training",
    report_data=report_data,
    report_type="hmm_training_report",
    additional_metadata={
        "models_trained": len(training_results),
        "timeframes": list(training_results.keys()),
        "model_architectures": list(self.model_architectures.keys()),
    }
)
```

#### Step 21: Saving
```python
# Log training summary with standardized naming
summary_artifact_name = log_step_artifact_with_standardized_name(
    config=self.config,
    step_name="step21_saving",
    artifact_path=temp_path,
    artifact_type="training_summary",
    additional_metadata={
        "summary_size": len(training_summary),
    }
)

# Log comprehensive final report
report_name = log_step_report(
    config=self.config,
    step_name="step21_saving",
    report_data=final_report_data,
    report_type="final_training_report",
    additional_metadata={
        "pipeline_steps_completed": len([k for k, v in pipeline_state.items() if v]),
        "pipeline_status": "completed",
    }
)
```

## Artifact Types and Naming Conventions

### DataFrames
- **Composite Clusters**: `composite_clusters`
- **Intensity Clusters**: `intensity_clusters`
- **Training Features**: `features_train`
- **Validation Features**: `features_val`
- **Processed Data**: `processed_data`

### Models
- **HMM Models**: `hmm_model`
- **K-means Models**: `kmeans_model`
- **Training Models**: `training_model`

### Reports
- **Regime Discovery Report**: `regime_discovery_report`
- **Feature Engineering Report**: `feature_engineering_report`
- **HMM Training Report**: `hmm_training_report`
- **Final Training Report**: `final_training_report`

### Metadata
- **Feature Metadata**: `feature_metadata`
- **Training Summary**: `training_summary`
- **Model Metadata**: `model_metadata`

## Benefits

### 1. Consistency
- All artifacts follow the same naming pattern
- Easy to identify and organize artifacts
- Consistent across all pipeline steps

### 2. Traceability
- Each artifact is associated with specific exchange, token, date, and step
- Easy to track artifacts through the pipeline
- Clear lineage from data to final models

### 3. Organization
- Artifacts are naturally sorted by exchange, token, date, and step
- Easy to find specific artifacts
- Clear hierarchy in MLflow artifact storage

### 4. Reproducibility
- Artifacts are timestamped for exact reproduction
- All context is preserved in the artifact name
- Easy to compare artifacts across different runs

## Best Practices

### 1. Always Use Standardized Functions
```python
# ✅ Good: Use standardized functions
artifact_name = log_step_dataframe_with_standardized_name(
    config=config,
    step_name="step_name",
    df=dataframe,
    artifact_type="artifact_type"
)

# ❌ Bad: Manual naming
log_step_dataframe(
    config=config,
    step_name="step_name",
    df=dataframe,
    artifact_name="custom_name"
)
```

### 2. Include Relevant Metadata
```python
# ✅ Good: Include comprehensive metadata
additional_metadata={
    "artifact_type": "composite_clusters",
    "dataframe_shape": list(df.shape),
    "regime_count": len(df.get("composite_cluster_id", []).unique()),
    "timeframe": timeframe,
    "processing_method": "hmm_analysis",
}
```

### 3. Log Reports for Complex Steps
```python
# ✅ Good: Log comprehensive reports
report_data = {
    "step_summary": {...},
    "metrics": {...},
    "artifacts": {...},
    "training_input": {...},
    "execution_timestamp": datetime.now().isoformat(),
}

report_name = log_step_report(
    config=config,
    step_name="step_name",
    report_data=report_data,
    report_type="step_report"
)
```

### 4. Handle Errors Gracefully
```python
try:
    artifact_name = log_step_dataframe_with_standardized_name(...)
    self.logger.info(f"✅ Logged artifact: {artifact_name}")
except Exception as e:
    self.logger.error(f"❌ Failed to log artifact: {e}")
    # Don't fail the step if MLflow logging fails
```

## Integration Checklist

For each pipeline step, ensure:

- [ ] Use `@with_enhanced_mlflow_logging()` decorator
- [ ] Log DataFrames with `log_step_dataframe_with_standardized_name()`
- [ ] Log artifacts with `log_step_artifact_with_standardized_name()`
- [ ] Log reports with `log_step_report()`
- [ ] Include comprehensive metadata
- [ ] Handle errors gracefully
- [ ] Log success messages with artifact names

## Example Implementation

See `docs/enhanced_mlflow_step_integration_template.py` for complete examples of how to implement standardized artifact naming in different types of pipeline steps.

## Validation

To validate that artifacts are properly named:

1. Check MLflow UI for consistent naming patterns
2. Verify all artifacts include exchange, token, date, and step information
3. Ensure reports are generated for complex steps
4. Confirm metadata is comprehensive and accurate

## Troubleshooting

### Common Issues

1. **Missing Metadata**: Ensure config contains required fields (trading_symbol, exchange_name, lookback_years)
2. **Incorrect Step Names**: Use exact step names (e.g., "step3_hmm_regime_discovery")
3. **File Not Found**: Check that artifact files exist before logging
4. **Permission Errors**: Ensure write permissions for MLflow artifact directory

### Debug Tips

1. Enable debug logging to see generated artifact names
2. Check MLflow run details for logged artifacts
3. Verify artifact paths in MLflow UI
4. Test with small datasets first