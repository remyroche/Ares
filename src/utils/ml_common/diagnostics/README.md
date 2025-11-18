# Signal-to-Noise Ratio (SNR) Diagnostics Module

## Overview

The SNR Diagnostics module provides comprehensive tools for assessing the predictability and signal quality of machine learning targets. It implements Phase 1 of a multi-phase diagnostic framework designed to answer fundamental questions about model performance and data quality.

**Key Questions Answered:**
- Is my target predictable, or am I fitting noise?
- How much signal exists in my features?
- Are my model results statistically significant?
- What is the confidence interval around my performance metrics?

## Features

### Core Capabilities (Phase 1 - Implemented)

1. **Cross-Validated Predictions**
   - Multiple model evaluation with proper CV to avoid overfitting
   - Support for grouped and stratified cross-validation
   - Automatic fold assignment tracking

2. **Comprehensive Metrics**
   - R² (Coefficient of Determination)
   - SNR = R²/(1-R²) - Signal-to-Noise Ratio
   - RMSE (Root Mean Squared Error)
   - nRMSE (Normalized RMSE)

3. **Statistical Validation**
   - Bootstrap confidence intervals (95% CI for R²)
   - Permutation testing (p-values for statistical significance)
   - Row-level resampling for robust estimation

4. **Rich Visualizations**
   - Predicted vs True scatter plots with density contours
   - Residual plots with LOWESS smoothing
   - SNR comparison bar charts across models
   - Automated plot generation and linking

5. **Comprehensive Reporting**
   - CSV reports with metrics and interpretations
   - Markdown reports with guidelines and analysis
   - JSON metrics for programmatic access
   - Parquet files with all cross-validated predictions

## Installation

The module is already integrated into the Ares codebase:

```python
from src.utils.ml_common.diagnostics import SNRDiagnostics
```

## Quick Start

### Basic Usage

```python
from src.utils.ml_common.diagnostics import SNRDiagnostics
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import numpy as np

# Prepare your data
X = your_features  # np.ndarray or pd.DataFrame
y = your_target    # np.ndarray or pd.Series

# Define models to evaluate
models = {
    'Ridge': Ridge(alpha=1.0),
    'Random_Forest': RandomForestRegressor(n_estimators=100, max_depth=8),
    'Gradient_Boosting': GradientBoostingRegressor(n_estimators=100)
}

# Initialize diagnostics
snr_diag = SNRDiagnostics(
    output_dir='./snr_output',
    cv_folds=5,
    bootstrap_iterations=1000,
    permutation_iterations=1000,
    random_state=42
)

# Run full diagnostic pipeline
cv_preds, metrics, plots, reports = snr_diag.run_full_diagnostics(
    models=models,
    X=X,
    y=y
)

# Access results
for model_name, m in metrics.items():
    print(f"{model_name}:")
    print(f"  R² = {m.r2:.4f}")
    print(f"  SNR = {m.snr:.4f}")
    print(f"  p-value = {m.permutation_pvalue:.4f}")
    print(f"  95% CI: [{m.bootstrap_ci_lower:.4f}, {m.bootstrap_ci_upper:.4f}]")
```

### Standalone Functions

```python
from src.utils.ml_common.diagnostics import (
    compute_snr_metrics,
    bootstrap_r2,
    permutation_test
)

# Quick metrics computation
metrics = compute_snr_metrics(y_true, y_pred)
print(f"R² = {metrics['r2']:.4f}, SNR = {metrics['snr']:.4f}")

# Bootstrap confidence interval
ci_lower, ci_upper = bootstrap_r2(y_true, y_pred, n_iterations=1000)
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Permutation test
pvalue = permutation_test(y_true, y_pred, n_permutations=1000)
print(f"p-value = {pvalue:.4f}")
```

## Integration with Meta-Labeling Step

The SNR diagnostics are automatically run as part of the `feature_generation_meta_labeling_step`:

```python
# In your config
config = {
    'symbol': 'BTCUSDT',
    'exchange': 'binance',
    'timeframe': '15m',
    # ... other config ...

    # SNR diagnostics parameters (optional)
    'snr_cv_folds': 5,
    'snr_bootstrap_iterations': 1000,
    'snr_permutation_iterations': 1000,
}

# The step will automatically:
# 1. Run SNR diagnostics on binary label prediction
# 2. Run SNR diagnostics on realized returns prediction
# 3. Generate comprehensive reports and visualizations
# 4. Save all results to outcomes/snr_diagnostics_{symbol}_{timeframe}/
```

## Output Files

When you run SNR diagnostics, the following files are generated:

### Directory Structure
```
outcomes/snr_diagnostics_{symbol}_{timeframe}/
├── cv_predictions.parquet              # Cross-validated predictions
├── snr_metrics.json                    # Metrics in JSON format
├── snr_diagnostics_report.csv          # Tabular metrics report
├── snr_diagnostics_report.md           # Comprehensive markdown report
├── snr_summary.json                    # Combined summary for both tasks
├── y_vs_ypred_{model_name}.png        # Scatter plots (one per model)
├── residuals_{model_name}.png         # Residual plots (one per model)
├── snr_comparison.png                  # SNR bar chart across models
└── regression/                         # Regression-specific outputs
    ├── cv_predictions.parquet
    ├── snr_metrics.json
    ├── snr_diagnostics_report.csv
    ├── snr_diagnostics_report.md
    └── *.png (plots)
```

### Report Contents

The **Markdown report** (`snr_diagnostics_report.md`) includes:
- Executive summary with best performing model
- Detailed metrics table for all models
- Interpretation guidelines for all metrics
- Model-specific analysis and recommendations
- Embedded visualizations
- Actionable next steps

The **CSV report** (`snr_diagnostics_report.csv`) includes:
- All numeric metrics
- Text interpretations for each metric
- Easy to import into Excel/Pandas for further analysis

## Interpretation Guidelines

### 1. R² (Coefficient of Determination)
- **R² > 0.40**: Strong predictable signal - meaningful modeling gains possible
- **0.10 < R² ≤ 0.40**: Weak-moderate signal - features matter more than model choice
- **R² ≤ 0.10**: Barely predictable - noise likely dominates

### 2. SNR (Signal-to-Noise Ratio)
- **SNR > 1**: Signal is stronger than noise - the target is learnable
- **0.3 < SNR ≤ 1**: Weak but real signal - more features or nonlinear models may help
- **SNR ≤ 0.3**: Noise overwhelms signal - predictability is fundamentally low

**Important**: SNR depends on both features AND model. Improvements can come from:
- Trying stronger models (deeper trees, neural networks)
- Adding engineered features
- If SNR rises → features were missing structure
- If SNR stays low → target may be intrinsically noisy

### 3. Permutation p-value
- **p < 0.01**: Statistically robust - model captures real pattern
- **0.01 ≤ p ≤ 0.20**: Weak/unstable signal - proceed with caution
- **p > 0.20**: No better than chance - label likely noisy

### 4. Bootstrap R² Confidence Interval
- **CI does NOT include 0**: Performance reliably above noise level
- **CI barely clears 0** (lower bound < 0.05): Signal present but fragile
- **CI spans below 0**: Performance may be indistinguishable from noise

### 5. What to Do Based on Results

#### Strong Signal (R² > 0.40, SNR > 1, p < 0.01)
✅ **Action**: Focus on model optimization and hyperparameter tuning
- Your target is predictable
- Experiment with different model architectures
- Fine-tune regularization and complexity parameters

#### Moderate Signal (0.10 < R² ≤ 0.40, 0.3 < SNR ≤ 1)
⚠️ **Action**: Feature engineering and model exploration
- Add engineered features (interactions, domain-specific transforms)
- Try ensemble methods or more complex architectures
- Investigate feature selection to reduce noise
- Examine residuals for patterns suggesting missing features

#### Weak Signal (R² ≤ 0.10, SNR ≤ 0.3, p > 0.20)
❌ **Action**: Fundamental investigation required
- Review data collection process for quality issues
- Question whether target definition is appropriate
- Consider if the prediction task is fundamentally feasible
- Explore alternative target formulations
- May need to pivot to a different problem

## API Reference

### SNRDiagnostics Class

```python
class SNRDiagnostics:
    def __init__(
        self,
        output_dir: Union[str, Path],
        cv_folds: int = 5,
        bootstrap_iterations: int = 1000,
        permutation_iterations: int = 1000,
        random_state: int = 42,
        verbose: bool = True
    )
```

**Parameters:**
- `output_dir`: Directory to save outputs (plots, data, reports)
- `cv_folds`: Number of cross-validation folds (default: 5)
- `bootstrap_iterations`: Number of bootstrap iterations for CI (default: 1000)
- `permutation_iterations`: Number of permutation iterations (default: 1000)
- `random_state`: Random seed for reproducibility (default: 42)
- `verbose`: Whether to print progress information (default: True)

**Methods:**

#### `cross_val_predictions(models, X, y, groups=None, stratify=False)`
Generate cross-validated predictions for multiple models.

**Returns:** pd.DataFrame with columns: id, y_true, y_pred, fold, model_name

#### `compute_metrics(cv_predictions=None)`
Compute SNR metrics from cross-validated predictions.

**Returns:** Dict[str, SNRMetrics]

#### `create_visualizations(cv_predictions=None, metrics=None)`
Create diagnostic visualizations.

**Returns:** Dict[str, Path] - mapping plot names to file paths

#### `generate_report(metrics=None, plot_paths=None)`
Generate comprehensive CSV and Markdown reports.

**Returns:** Tuple[Path, Path] - paths to (CSV report, Markdown report)

#### `run_full_diagnostics(models, X, y, groups=None, stratify=False)`
Run the complete diagnostic pipeline.

**Returns:** Tuple containing:
- cv_predictions: pd.DataFrame
- metrics: Dict[str, SNRMetrics]
- plot_paths: Dict[str, Path]
- report_paths: Tuple[Path, Path]

### SNRMetrics Dataclass

```python
@dataclass
class SNRMetrics:
    r2: float                    # R² score
    snr: float                   # Signal-to-noise ratio
    rmse: float                  # Root mean squared error
    nrmse: float                 # Normalized RMSE
    bootstrap_ci_lower: float    # 95% CI lower bound
    bootstrap_ci_upper: float    # 95% CI upper bound
    permutation_pvalue: float    # Permutation test p-value
    n_samples: int               # Number of samples
    model_name: str              # Model identifier
```

### Standalone Functions

#### `compute_snr_metrics(y_true, y_pred) -> Dict[str, float]`
Compute basic SNR metrics for a single model.

**Returns:** dict with r2, snr, rmse, nrmse

#### `bootstrap_r2(y_true, y_pred, n_iterations=1000, confidence_level=0.95, random_state=42) -> Tuple[float, float]`
Compute bootstrap confidence interval for R².

**Returns:** (lower_bound, upper_bound)

#### `permutation_test(y_true, y_pred, n_permutations=1000, random_state=42, metric='r2') -> float`
Perform permutation test for statistical significance.

**Returns:** p-value

#### `cross_val_predictions(model, X, y, cv=5, groups=None) -> np.ndarray`
Generate cross-validated predictions for a single model.

**Returns:** array of predictions

## Advanced Usage

### Custom Cross-Validation Strategy

```python
from sklearn.model_selection import TimeSeriesSplit

# Use time series CV
snr_diag = SNRDiagnostics(output_dir='./output')

# For time series data with groups
cv_preds, metrics, plots, reports = snr_diag.run_full_diagnostics(
    models=models,
    X=X,
    y=y,
    groups=time_groups  # Group labels for GroupKFold
)
```

### Classification Tasks (Binary/Multiclass)

For classification, wrap your classifier to output probabilities:

```python
from sklearn.base import BaseEstimator, ClassifierMixin

class ProbabilityWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        # Return probability of positive class
        return self.estimator.predict_proba(X)[:, 1]

# Use wrapper
models = {
    'Logistic_Regression': ProbabilityWrapper(
        LogisticRegression(max_iter=1000)
    )
}
```

### Adjusting Statistical Parameters

```python
# More conservative testing
snr_diag = SNRDiagnostics(
    output_dir='./output',
    bootstrap_iterations=5000,      # More bootstrap samples
    permutation_iterations=5000,    # More permutations
    cv_folds=10                     # More CV folds
)
```

## Future Phases (Roadmap)

### Phase 2: Signal-vs-Noise Attribution
- Model family sweep (linear, RF, GBM, MLP)
- Feature ablation (grouped + incremental)
- Synthetic signal injection tests
- Residual modeling
- Heteroscedastic residual prediction

### Phase 3: Noise Ceiling
- ICC (Intraclass Correlation Coefficient)
- Krippendorff's alpha
- Replicate-based checks
- Upper bounds on predictability

### Phase 4: Aleatoric vs Epistemic Uncertainty
- Ensemble uncertainty quantification
- Heteroscedastic models
- MC Dropout / Bayesian approximations
- Calibration analysis

### Phase 5: Subgroup & Spatio-Temporal Diagnostics
- Subgroup SNR analysis
- Residual autocorrelation
- Spatio-temporal structure detection

## Troubleshooting

### Issue: Low SNR across all models
**Possible causes:**
- Target is intrinsically noisy
- Features don't contain relevant information
- Data quality issues

**Solutions:**
1. Review data collection and preprocessing
2. Engineer domain-specific features
3. Check for label noise or annotation quality
4. Consider alternative target definitions

### Issue: High variance in bootstrap CI
**Possible causes:**
- Small sample size
- Non-stationary data
- Outliers or heteroscedasticity

**Solutions:**
1. Increase sample size if possible
2. Use robust scaling or outlier removal
3. Investigate data stationarity
4. Consider stratified sampling

### Issue: Permutation test shows p > 0.20 but R² > 0
**Interpretation:**
- The R² might be spurious or due to data leakage
- Check for temporal leakage in time series
- Verify CV strategy is appropriate
- Consider if features contain future information

## References

1. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*.
3. Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*.

## Support

For issues, questions, or feature requests, please:
1. Check the interpretation guidelines in this README
2. Review the generated markdown reports for model-specific guidance
3. Consult the comprehensive diagnostics output
4. Open an issue in the repository

---

**Version**: 1.0.0 (Phase 1)
**Last Updated**: 2025-11-18
**Author**: Ares ML Team
