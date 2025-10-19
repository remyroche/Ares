# Comprehensive Report Generation Guide

## 📊 Overview

Each pre-training step now generates comprehensive .md reports with detailed metrics for financial, technical, and process troubleshooting. These reports provide:

- **Financial Metrics**: Performance indicators, risk metrics, feature performance
- **Technical Metrics**: System performance, data processing metrics, resource usage
- **Process Metrics**: Execution analysis, quality metrics, error tracking
- **Troubleshooting Guide**: Common issues, diagnostic commands, recommendations

## 🔧 Implementation Pattern

### 1. Report Generation in Each Step

Each step should call `artifact_manager.generate_comprehensive_report()` after saving artifacts:

```python
# Generate comprehensive report
self.logger.info("📊 Generating comprehensive analysis report...")

# Prepare metrics for the report
general_metrics = {
    'features_processed': result.get('features_processed', 0),
    'data_rows': result.get('data_rows', 0),
    # ... other general metrics
}

financial_metrics = {
    'sharpe_ratio': result.get('performance_metrics', {}).get('sharpe_ratio', 0.0),
    'max_drawdown': result.get('performance_metrics', {}).get('max_drawdown', 0.0),
    # ... other financial metrics
}

technical_metrics = {
    'memory_usage_mb': result.get('technical_metrics', {}).get('memory_usage_mb', 0.0),
    'execution_time_seconds': result.get('technical_metrics', {}).get('execution_time_seconds', 0.0),
    # ... other technical metrics
}

process_metrics = {
    'step_duration_seconds': result.get('technical_metrics', {}).get('execution_time_seconds', 0.0),
    'artifacts_generated': len(result.keys()),
    'dependencies_loaded': dependency_count,
    # ... other process metrics
}

# Generate the comprehensive report
report_path = self.artifact_manager.generate_comprehensive_report(
    step_name='step_name',
    metrics=general_metrics,
    financial_metrics=financial_metrics,
    technical_metrics=technical_metrics,
    process_metrics=process_metrics
)

if report_path:
    self.logger.info(f"✅ Comprehensive report generated: {report_path}")
else:
    self.logger.warning("⚠️ Failed to generate comprehensive report")
```

### 2. Metrics Categories

#### General Metrics
- `features_processed`: Number of features processed
- `data_rows`: Number of data rows processed
- `artifacts_generated`: Number of artifacts created
- `dependencies_loaded`: Number of dependencies loaded

#### Financial Metrics
- `sharpe_ratio`: Sharpe ratio of the strategy
- `max_drawdown`: Maximum drawdown
- `return_metrics`: Return-related metrics
- `risk_metrics`: Risk-related metrics
- `feature_performance`: Feature performance scores
- `target_correlation`: Correlation with targets

#### Technical Metrics
- `memory_usage_mb`: Memory usage in MB
- `execution_time_seconds`: Execution time in seconds
- `cpu_usage_percent`: CPU usage percentage
- `data_size_mb`: Data size in MB
- `rows_processed`: Number of rows processed
- `columns_processed`: Number of columns processed
- `throughput_rows_per_second`: Processing throughput
- `compression_ratio`: Data compression ratio

#### Process Metrics
- `step_duration_seconds`: Step duration in seconds
- `artifacts_generated`: Number of artifacts generated
- `dependencies_loaded`: Number of dependencies loaded
- `data_quality_score`: Data quality score (0-1)
- `validation_passed`: Whether validation passed
- `warnings_count`: Number of warnings
- `errors_count`: Number of errors
- `retry_count`: Number of retries

## 📋 Step-Specific Implementation

### Feature Generation Period Lookback Optimization Step

**Financial Metrics:**
- Sharpe ratio from period optimization
- Max drawdown from period optimization
- Return metrics from period optimization
- Risk metrics from period optimization
- Feature performance from MI scores
- Target correlation from correlation analysis

**Technical Metrics:**
- Memory usage during optimization
- Execution time for optimization
- CPU usage during optimization
- Data size processed
- Rows and columns processed
- Throughput metrics
- Compression ratio

**Process Metrics:**
- Step duration
- Artifacts generated (optimized_periods, optimized_lookbacks, etc.)
- Dependencies loaded (feature_generation_feature_generation_step, feature_generation_labeling_integration_step)
- Data quality score
- Validation status
- Warning and error counts

### Feature Generation Feature Selection Step

**Financial Metrics:**
- Feature selection performance
- Target alignment metrics
- Risk-adjusted returns
- Feature importance scores
- Correlation with targets

**Technical Metrics:**
- Memory usage during selection
- Execution time for selection
- Data size processed
- Features processed
- Selection algorithm performance

**Process Metrics:**
- Step duration
- Artifacts generated (selected_features, feature_selection_scores, etc.)
- Dependencies loaded (feature_generation_feature_generation_step, feature_generation_period_lookback_optimization_step, feature_generation_labeling_integration_step)
- Selection quality metrics
- Validation status

### Feature Generation Interaction Generation Steps (Analyst/Tactician)

**Financial Metrics:**
- Interaction feature performance
- Target correlation for interactions
- Feature combination effectiveness
- Risk metrics for interactions

**Technical Metrics:**
- Memory usage during interaction generation
- Execution time for interaction generation
- Data size processed
- Interaction features generated
- Algorithm performance

**Process Metrics:**
- Step duration
- Artifacts generated (interaction_features, interaction_metadata, etc.)
- Dependencies loaded (feature_generation_feature_selection_step, feature_generation_period_lookback_optimization_step, feature_generation_labeling_integration_step)
- Interaction quality metrics
- Validation status

### Feature Generation Final Feature Selection Step

**Financial Metrics:**
- Final feature selection performance
- Combined feature performance
- Target alignment metrics
- Risk-adjusted returns
- Feature importance scores

**Technical Metrics:**
- Memory usage during final selection
- Execution time for final selection
- Data size processed
- Features processed
- Selection algorithm performance

**Process Metrics:**
- Step duration
- Artifacts generated (selected_features_60/50/40, feature_scores, etc.)
- Dependencies loaded (feature_generation_feature_generation_step, feature_generation_period_lookback_optimization_step, feature_generation_interaction_generation_step_analyst, feature_generation_interaction_generation_step_tactician, feature_generation_labeling_integration_step)
- Final selection quality metrics
- Validation status

### Feature Generation Final Validation Step

**Financial Metrics:**
- Final validation performance
- Model performance metrics
- Risk metrics
- Return metrics
- Validation scores

**Technical Metrics:**
- Memory usage during validation
- Execution time for validation
- Data size processed
- Validation dataset size
- Algorithm performance

**Process Metrics:**
- Step duration
- Artifacts generated (final_dataset, final_validation_metrics, etc.)
- Dependencies loaded (feature_generation_final_feature_selection_step, feature_generation_labeling_integration_step)
- Validation quality metrics
- Final validation status

## 🔍 Report Structure

Each generated report includes:

1. **Executive Summary**: High-level overview with key metrics
2. **Financial Metrics**: Performance indicators and risk metrics
3. **Technical Metrics**: System performance and resource usage
4. **Process Metrics**: Execution analysis and quality metrics
5. **General Metrics**: Step-specific performance metrics
6. **Troubleshooting Guide**: Common issues and solutions
7. **Artifact Inventory**: Generated artifacts and dependencies
8. **Error Analysis**: Recent errors and warning indicators
9. **Recommendations**: Performance, financial, and process optimization suggestions

## 📊 Report Benefits

### For Financial Analysis
- Track performance metrics across steps
- Monitor risk-adjusted returns
- Analyze feature effectiveness
- Identify optimization opportunities

### For Technical Troubleshooting
- Monitor system resource usage
- Track execution performance
- Identify bottlenecks
- Optimize processing efficiency

### For Process Management
- Track step execution quality
- Monitor data quality
- Identify process improvements
- Ensure proper artifact flow

## 🚀 Usage

Reports are automatically generated after each step execution and saved to:
```
artifacts/pre_training/artifact_store/{step_name}/{step_name}_comprehensive_report.md
```

Reports can be viewed in any markdown viewer or text editor for detailed analysis and troubleshooting.

## 📈 Monitoring

Use these reports to:
- Monitor pipeline health
- Identify performance bottlenecks
- Track financial performance
- Troubleshoot issues
- Optimize processes
- Ensure data quality

## 🔧 Customization

Each step can customize its metrics based on its specific functionality:

1. **Add step-specific metrics** to the appropriate category
2. **Include domain-specific financial metrics** for trading strategies
3. **Add technical metrics** relevant to the step's operations
4. **Include process metrics** specific to the step's workflow

The report generation system is flexible and can accommodate any metrics relevant to the step's functionality.
