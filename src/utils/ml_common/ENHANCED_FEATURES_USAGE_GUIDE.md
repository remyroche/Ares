# Enhanced ML Common Features Usage Guide

This guide provides comprehensive examples for using the newly implemented features in the ML Common utilities.

## 1. Automated Feature Importance Analysis

### Basic Usage

```python
from src.utils.ml_common import (
    FeatureImportanceAnalyzer, FeatureImportanceConfig, ImportanceMethod,
    analyze_feature_importance, get_important_features
)
import pandas as pd
import numpy as np

# Load your data
data = pd.read_parquet("your_data.parquet")
X = data.drop(columns=['target'])
y = data['target']

# Quick analysis
important_features = get_important_features(X, y, k=20)

# Comprehensive analysis
config = FeatureImportanceConfig(
    methods=[
        ImportanceMethod.RANDOM_FOREST,
        ImportanceMethod.LASSO,
        ImportanceMethod.MUTUAL_INFO,
        ImportanceMethod.PERMUTATION
    ],
    top_k_features=20,
    save_results=True,
    generate_plots=True,
    output_directory="feature_importance_results"
)

analyzer = FeatureImportanceAnalyzer(config)
result = analyzer.analyze_features(X, y)

# Get top features by method
top_rf_features = result.get_top_features("random_forest", k=10)
top_ensemble_features = result.get_top_features("ensemble", k=15)

print(f"Top 10 Random Forest features: {top_rf_features}")
print(f"Top 15 Ensemble features: {top_ensemble_features}")
```

### Regime-Specific Analysis

```python
# If you have regime labels
regime_labels = data['regime']  # Your regime column

result = analyzer.analyze_features(X, y, regime_labels)

# Get regime-specific important features
for regime in regime_labels.unique():
    regime_mask = regime_labels == regime
    regime_features = result.get_top_features("ensemble", k=10)
    print(f"Regime {regime} top features: {regime_features}")
```

## 2. Automated Data Drift Detection

### Basic Drift Detection

```python
from src.utils.ml_common import (
    DataDriftDetector, DriftDetectionConfig, DriftMethod, DriftSeverity,
    detect_data_drift, get_drifted_features
)

# Load reference and current data
reference_data = pd.read_parquet("reference_data.parquet")
current_data = pd.read_parquet("current_data.parquet")

# Quick drift detection
drifted_features = get_drifted_features(reference_data, current_data)

# Comprehensive drift analysis
config = DriftDetectionConfig(
    methods=[
        DriftMethod.KS_TEST,
        DriftMethod.PSI,
        DriftMethod.WASSERSTEIN,
        DriftMethod.CHI_SQUARE
    ],
    drift_threshold=0.05,
    warning_threshold=0.1,
    critical_threshold=0.2,
    save_results=True,
    generate_plots=True,
    output_directory="drift_detection_results"
)

detector = DataDriftDetector(config)
report = detector.detect_drift(reference_data, current_data)

# Analyze results
print(f"Total features: {report.total_features}")
print(f"Drifted features: {report.drifted_features}")
print(f"Drift rate: {report.drifted_features / report.total_features:.2%}")

# Get severity summary
for severity, count in report.severity_summary.items():
    print(f"{severity.value}: {count} features")

# Get recommendations
for recommendation in report.recommendations:
    print(f"Recommendation: {recommendation}")
```

### Regime-Aware Drift Detection

```python
# If you have regime labels
regime_labels = current_data['regime']

report = detector.detect_drift(reference_data, current_data, regime_labels=regime_labels)

# Analyze regime-specific drift
for result in report.drift_results:
    if 'regime' in result.feature_name:
        print(f"Regime drift detected: {result.feature_name}")
```

## 3. HMM Ensemble Methods for Stability

### Basic HMM Ensemble

```python
from src.utils.ml_common import (
    HMMEnsembleManager, EnsembleConfig, EnsembleMethod, HMMType, HMMConfig,
    create_hmm_ensemble, get_stable_hmm_labels
)

# Load your data
data = pd.read_parquet("your_data.parquet")
X = data.select_dtypes(include=[np.number])

# Quick stable HMM labels
labels, quality = get_stable_hmm_labels(X, min_quality=0.7)
print(f"HMM quality score: {quality:.3f}")

# Comprehensive ensemble
config = EnsembleConfig(
    method=EnsembleMethod.VOTING,
    base_configs=[
        HMMConfig(n_components=2, covariance_type="full"),
        HMMConfig(n_components=3, covariance_type="diag"),
        HMMConfig(n_components=4, covariance_type="spherical"),
        HMMConfig(n_components=5, covariance_type="full"),
        HMMConfig(n_components=6, covariance_type="diag")
    ],
    voting_type="soft",
    save_results=True,
    generate_plots=True,
    output_directory="hmm_ensemble_results"
)

manager = HMMEnsembleManager(config)
result = manager.fit_ensemble(X)

print(f"Ensemble quality: {result.overall_quality:.3f}")
print(f"Diversity score: {result.diversity_score:.3f}")
print(f"Stability score: {result.stability_score:.3f}")
```

### Advanced Ensemble Methods

```python
# Bayesian Averaging Ensemble
bayesian_config = EnsembleConfig(
    method=EnsembleMethod.BAYESIAN_AVERAGING,
    base_configs=[
        HMMConfig(n_components=3, covariance_type="full"),
        HMMConfig(n_components=4, covariance_type="diag"),
        HMMConfig(n_components=5, covariance_type="spherical")
    ]
)

bayesian_result = create_hmm_ensemble(X, EnsembleMethod.BAYESIAN_AVERAGING, bayesian_config)

# Stacking Ensemble
stacking_config = EnsembleConfig(
    method=EnsembleMethod.STACKING,
    stacking_cv_folds=5,
    base_configs=[
        HMMConfig(n_components=2, covariance_type="full"),
        HMMConfig(n_components=3, covariance_type="diag"),
        HMMConfig(n_components=4, covariance_type="spherical")
    ]
)

stacking_result = create_hmm_ensemble(X, EnsembleMethod.STACKING, stacking_config)

# Compare ensemble methods
print(f"Voting ensemble quality: {result.overall_quality:.3f}")
print(f"Bayesian ensemble quality: {bayesian_result.overall_quality:.3f}")
print(f"Stacking ensemble quality: {stacking_result.overall_quality:.3f}")
```

## 4. Integrated Pipeline Example

### Complete Feature Engineering and Model Stability Pipeline

```python
from src.utils.ml_common import (
    FeatureImportanceAnalyzer, FeatureImportanceConfig, ImportanceMethod,
    DataDriftDetector, DriftDetectionConfig, DriftMethod,
    HMMEnsembleManager, EnsembleConfig, EnsembleMethod
)

def complete_analysis_pipeline(reference_data, current_data, target_column='target'):
    """
    Complete analysis pipeline combining all enhanced features.
    """
    results = {}
    
    # 1. Feature Importance Analysis
    print("🔍 Step 1: Feature Importance Analysis")
    X_ref = reference_data.drop(columns=[target_column])
    y_ref = reference_data[target_column]
    
    importance_config = FeatureImportanceConfig(
        methods=[ImportanceMethod.RANDOM_FOREST, ImportanceMethod.LASSO, ImportanceMethod.MUTUAL_INFO],
        top_k_features=20,
        save_results=True,
        output_directory="analysis_results/feature_importance"
    )
    
    importance_analyzer = FeatureImportanceAnalyzer(importance_config)
    importance_result = importance_analyzer.analyze_features(X_ref, y_ref)
    
    results['feature_importance'] = importance_result
    print(f"✅ Feature importance analysis completed. Top features: {importance_result.get_top_features('ensemble', k=5)}")
    
    # 2. Data Drift Detection
    print("🔍 Step 2: Data Drift Detection")
    X_cur = current_data.drop(columns=[target_column])
    
    drift_config = DriftDetectionConfig(
        methods=[DriftMethod.KS_TEST, DriftMethod.PSI, DriftMethod.WASSERSTEIN],
        drift_threshold=0.05,
        save_results=True,
        output_directory="analysis_results/drift_detection"
    )
    
    drift_detector = DataDriftDetector(drift_config)
    drift_report = drift_detector.detect_drift(X_ref, X_cur)
    
    results['drift_detection'] = drift_report
    print(f"✅ Drift detection completed. Drifted features: {drift_report.drifted_features}/{drift_report.total_features}")
    
    # 3. HMM Ensemble for Regime Detection
    print("🔍 Step 3: HMM Ensemble Regime Detection")
    
    # Use only important features for HMM
    important_features = importance_result.get_top_features("ensemble", k=15)
    X_hmm = X_cur[important_features]
    
    hmm_config = EnsembleConfig(
        method=EnsembleMethod.VOTING,
        voting_type="soft",
        save_results=True,
        output_directory="analysis_results/hmm_ensemble"
    )
    
    hmm_manager = HMMEnsembleManager(hmm_config)
    hmm_result = hmm_manager.fit_ensemble(X_hmm)
    
    results['hmm_ensemble'] = hmm_result
    print(f"✅ HMM ensemble completed. Quality: {hmm_result.overall_quality:.3f}")
    
    # 4. Generate Summary Report
    print("📊 Step 4: Generating Summary Report")
    
    summary = {
        'feature_importance': {
            'top_features': importance_result.get_top_features("ensemble", k=10),
            'stability_scores': importance_result.stability_scores
        },
        'drift_detection': {
            'drift_rate': drift_report.drifted_features / drift_report.total_features,
            'critical_features': [r.feature_name for r in drift_report.drift_results if r.severity.value == 'critical'],
            'recommendations': drift_report.recommendations
        },
        'hmm_ensemble': {
            'quality_score': hmm_result.overall_quality,
            'diversity_score': hmm_result.diversity_score,
            'stability_score': hmm_result.stability_score,
            'n_regimes': len(np.unique(hmm_result.ensemble_labels))
        }
    }
    
    # Save summary
    import json
    with open("analysis_results/summary_report.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("✅ Complete analysis pipeline finished!")
    return results, summary

# Usage
reference_data = pd.read_parquet("reference_data.parquet")
current_data = pd.read_parquet("current_data.parquet")

results, summary = complete_analysis_pipeline(reference_data, current_data)
```

## 5. Configuration Best Practices

### Feature Importance Configuration

```python
# For high-dimensional data
high_dim_config = FeatureImportanceConfig(
    methods=[ImportanceMethod.RANDOM_FOREST, ImportanceMethod.LASSO],
    top_k_features=50,
    n_jobs=-1,
    enable_parallel=True
)

# For time-series data
timeseries_config = FeatureImportanceConfig(
    methods=[ImportanceMethod.MUTUAL_INFO, ImportanceMethod.PERMUTATION],
    temporal_window=1000,
    stability_threshold=0.8
)
```

### Drift Detection Configuration

```python
# For production monitoring
production_config = DriftDetectionConfig(
    methods=[DriftMethod.PSI, DriftMethod.WASSERSTEIN],
    drift_threshold=0.03,  # Stricter threshold
    enable_alerts=True,
    alert_cooldown=1800  # 30 minutes
)

# For research/development
research_config = DriftDetectionConfig(
    methods=[DriftMethod.KS_TEST, DriftMethod.CHI_SQUARE, DriftMethod.JENSEN_SHANNON],
    drift_threshold=0.1,  # More lenient
    generate_plots=True,
    save_results=True
)
```

### HMM Ensemble Configuration

```python
# For stable regime detection
stable_config = EnsembleConfig(
    method=EnsembleMethod.BAYESIAN_AVERAGING,
    base_configs=[
        HMMConfig(n_components=3, covariance_type="full"),
        HMMConfig(n_components=4, covariance_type="diag"),
        HMMConfig(n_components=5, covariance_type="spherical")
    ],
    min_quality_score=0.7
)

# For exploratory analysis
exploratory_config = EnsembleConfig(
    method=EnsembleMethod.VOTING,
    base_configs=[
        HMMConfig(n_components=2, covariance_type="full"),
        HMMConfig(n_components=3, covariance_type="diag"),
        HMMConfig(n_components=4, covariance_type="spherical"),
        HMMConfig(n_components=5, covariance_type="full"),
        HMMConfig(n_components=6, covariance_type="diag"),
        HMMConfig(n_components=7, covariance_type="spherical")
    ],
    voting_type="soft"
)
```

## 6. Performance Optimization Tips

1. **Use Parallel Processing**: Enable `n_jobs=-1` and `enable_parallel=True` for large datasets
2. **Chunk Large Data**: Use `chunk_size` parameter for memory-efficient processing
3. **Selective Methods**: Choose only necessary detection methods to reduce computation time
4. **Quality Thresholds**: Set appropriate quality thresholds to filter out poor results early
5. **Caching**: Save results and reuse them for similar analyses

## 7. Troubleshooting

### Common Issues and Solutions

1. **Import Errors**: Ensure all dependencies are installed (`scikit-learn`, `scipy`, `hmmlearn`)
2. **Memory Issues**: Reduce `chunk_size` or use fewer parallel processes
3. **Convergence Issues**: Adjust HMM parameters or use different covariance types
4. **Poor Quality Scores**: Check data preprocessing and feature engineering steps

### Performance Monitoring

```python
# Monitor performance
import time
import psutil

def monitor_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        print(f"Execution time: {end_time - start_time:.2f}s")
        print(f"Memory usage: {end_memory - start_memory:.2f}MB")
        
        return result
    return wrapper

@monitor_performance
def analyze_features_with_monitoring(X, y):
    return analyze_feature_importance(X, y)
```

This guide provides comprehensive examples for using all the enhanced ML Common features. Each component is designed to work independently or as part of an integrated pipeline for robust machine learning workflows.