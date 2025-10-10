# Feature Comparison Framework

This framework provides comprehensive tools to compare different feature engineering approaches, specifically focusing on VWAP-based features, volatility normalization, and their combinations.

## Overview

The framework compares 4 different feature versions:
1. **Initial**: Basic OHLCV data + fundamental technical indicators
2. **VWAP-based**: Features derived from Volume Weighted Average Price
3. **Vol-normalized**: Features normalized by volatility
4. **VWAP+Vol-normalized**: Combined VWAP and volatility normalization

## Features

- **Multiple Relevance Metrics**: LGBM with SHAP values, LASSO regression, Mutual Information, Correlation analysis
- **Robust Scaling**: Multiple scaling methods (Standard, Robust, MinMax, Quantile, Power) with validation
- **Robust Evaluation**: Spearman rank correlation, Bootstrap resampling (10 samples), Temporal stability analysis
- **Returns-Based Calculations**: All features use returns instead of raw prices for better statistical properties
- **Advanced Features**: Rolling averages, EWMA, lagged/lead features for predictive vs reactive analysis
- **Matrix Operations**: Hardware-optimized matrix operations for performance
- **Comprehensive Reporting**: JSON and Markdown reports with visualizations
- **Modular Design**: Easy to extend with new feature engineering approaches
- **Integration**: Works with existing feature modules in the codebase

## Quick Start

### Basic Usage

```python
from src.research.feature_comparison.run_comparison import FeatureComparisonRunner

# Initialize with your data
runner = FeatureComparisonRunner(data=your_data, task_type='regression')

# Run complete analysis
results = runner.run_complete_analysis()

# Print summary
runner.print_summary(results)
```

### Using Sample Data

```python
# Run with sample data
runner = FeatureComparisonRunner()
results = runner.run_complete_analysis()
```

## Module Structure

### Core Modules

- **`feature_comparison_utils.py`**: Utility functions to call scripts from different feature modules
- **`feature_versions.py`**: Manages the 4 different feature versions
- **`relevance_analyzer.py`**: Analyzes feature relevance using multiple methods
- **`comparison_report.py`**: Generates comprehensive comparison reports
- **`run_comparison.py`**: Main script to run the complete analysis

### Key Classes

#### FeatureComparisonUtils
- `create_vwap_features()`: Generate VWAP-based features
- `create_volatility_normalized_features()`: Generate volatility-normalized features
- `create_combined_features()`: Generate combined VWAP + volatility features
- `prepare_feature_versions()`: Prepare all 4 versions for comparison

#### FeatureVersions
- `generate_all_versions()`: Create all 4 feature versions
- `get_feature_matrix()`: Get feature matrix for specific version
- `get_version_info()`: Get information about each version
- `compare_feature_counts()`: Compare feature counts across versions

#### RelevanceAnalyzer
- `lgbm_shap_analysis()`: LGBM analysis with SHAP values
- `lasso_analysis()`: LASSO regression analysis
- `mutual_information_analysis()`: Mutual information analysis
- `comprehensive_analysis()`: Run all analysis methods

#### ComparisonReport
- `generate_comprehensive_report()`: Generate complete comparison report
- `generate_markdown_report()`: Generate markdown report
- `_generate_comparison_plots()`: Generate visualization plots

## Feature Engineering Approaches

### 1. Initial Features (Returns-Based)
- Returns and log returns
- Rolling averages (SMA, EWMA) of returns
- Returns volatility and higher moments (skewness, kurtosis)
- Lagged returns (reactive features)
- Lead returns (predictive features)
- Returns momentum and acceleration
- Volume-returns relationships

### 2. VWAP-based Features (Returns-Based)
- VWAP returns calculation
- Returns-VWAP ratios and differences
- VWAP returns momentum indicators
- VWAP returns volatility measures
- Volume-weighted returns

### 3. Volatility Normalized Features (Returns-Based)
- Returns normalized by rolling volatility
- Volatility-normalized rolling features
- Volatility regime features
- High/low volatility returns

### 4. Combined Features (Returns-Based)
- VWAP returns + volatility normalization
- Advanced combined momentum indicators
- Volatility regime-based features
- Cross-feature interactions

## Analysis Methods

### LGBM with SHAP
- Gradient boosting model for feature importance
- SHAP values for feature attribution
- Performance metrics (R², MSE for regression)

### LASSO Regression
- Sparse linear model
- Feature selection through regularization
- Cross-validated alpha selection

### Mutual Information
- Non-parametric dependency measure
- Captures non-linear relationships
- Robust to outliers

### Correlation Analysis
- Linear correlation with target
- Quick feature screening
- Baseline comparison

## Robust Evaluation Methods

### Robust Scaling
- **Standard Scaling**: Z-score normalization
- **Robust Scaling**: Uses median and IQR (outlier-resistant)
- **MinMax Scaling**: Scales to [0,1] range
- **Quantile Scaling**: Maps to uniform/normal distribution
- **Power Scaling**: Yeo-Johnson transformation
- **Validation**: Checks for NaN, infinite values, and scaling quality

### Spearman Rank Correlation
- Measures agreement between different feature importance methods
- Robust to outliers and non-linear relationships
- Provides significance testing (p-values)
- Identifies method consensus and disagreements

### Bootstrap Resampling
- Assesses feature importance variance across bootstrap samples
- Calculates coefficient of variation (CV) for stability
- Provides confidence intervals for feature rankings
- Identifies robust vs. unstable features

### Temporal Stability Analysis
- Analyzes feature importance consistency over time windows
- Calculates stability scores for each feature
- Identifies temporally stable features
- Measures ranking consistency across periods

## Advanced Features

### Returns-Based Calculations
- **No Raw Prices**: All calculations use returns for better statistical properties
- **Stationarity**: Returns are more stationary than prices
- **Risk-Adjusted**: Returns provide better risk-adjusted comparisons
- **Volatility Scaling**: Returns naturally scale with volatility

### Rolling and EWMA Features
- **Rolling Averages**: SMA and EWMA of returns with multiple windows (5, 10, 20, 50)
- **Rolling Statistics**: Standard deviation, skewness, kurtosis of returns
- **Exponentially Weighted**: EWMA with different decay factors
- **Matrix Optimized**: Uses vectorized operations for performance

### Lagged and Lead Features
- **Lagged Features**: Past returns (1, 2, 3, 5, 10 periods) for reactive analysis
- **Lead Features**: Future returns (1, 2, 3, 5 periods) for predictive analysis
- **Momentum Features**: Differences between current and lagged returns
- **Acceleration**: Second differences for trend analysis

### Matrix Operations Integration
- **Hardware Acceleration**: M1/M2/M3 GPU acceleration when available
- **Vectorized Operations**: Batch processing for multiple features
- **Memory Optimization**: Efficient memory usage for large datasets
- **Parallel Processing**: Multi-core CPU utilization

### Bootstrap Optimization
- **Reduced Samples**: 10 bootstrap samples instead of 50+ for faster analysis
- **Efficient Resampling**: Optimized sampling algorithms
- **Variance Assessment**: Coefficient of variation for feature stability
- **Performance Tracking**: Bootstrap performance metrics

## Output Reports

### JSON Report
- Complete analysis results
- Performance metrics
- Feature rankings
- Method-specific results

### Markdown Report
- Human-readable summary
- Performance comparison tables
- Top features by version
- Analysis insights

### Visualization Plots
- Feature count comparison
- Performance metrics comparison
- Top features heatmap
- Method agreement analysis
- Robust evaluation metrics
- Bootstrap stability analysis
- Temporal stability plots

## Example Output

```
FEATURE COMPARISON ANALYSIS SUMMARY
================================================================================

Feature Counts by Version:
----------------------------------------
initial                :   45 features
vwap_based            :   58 features
vol_normalized        :   52 features
vwap_vol_normalized   :   65 features

Performance Summary (LGBM R² Score):
----------------------------------------
initial                : 0.1234
vwap_based            : 0.1456
vol_normalized        : 0.1345
vwap_vol_normalized   : 0.1567

Top 5 Features by Version (Combined Ranking):
------------------------------------------------------------

vwap_based:
  1. vwap_momentum_20 (rank: 2.25)
  2. price_vwap_ratio (rank: 3.50)
  3. vwap_volatility_10 (rank: 4.75)
  4. vwap_momentum_10 (rank: 5.00)
  5. price_vwap_pct (rank: 6.25)

Robust Evaluation Metrics:
------------------------------------------------------------

vwap_based:
  Mean Rank Correlation: 0.742
  LGBM Mean CV: 0.156
  Mean Temporal Stability: 0.823
  Stable Features Count: 12
  Scaling Method: robust
```

## Integration with Existing Modules

The framework integrates with existing feature engineering modules:

- **feature_lookback_optimization**: For lookback period optimization
- **feature_generation**: For advanced feature generation
- **features_common**: For scalers and transforms
- **feature_selection**: For feature selection methods

## Requirements

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- lightgbm (optional, for LGBM analysis)
- shap (optional, for SHAP analysis)

## Usage Examples

See `example_usage.py` for detailed usage examples including:
- Basic comparison analysis
- Custom parameter configuration
- Specific version analysis
- Sample data generation

## File Structure

```
feature_comparison/
├── __init__.py
├── README.md
├── feature_comparison_utils.py
├── feature_versions.py
├── relevance_analyzer.py
├── comparison_report.py
├── run_comparison.py
├── example_usage.py
└── reports/
    ├── feature_comparison_report_YYYYMMDD_HHMMSS.json
    ├── feature_comparison_report_YYYYMMDD_HHMMSS.md
    ├── feature_count_comparison.png
    ├── performance_comparison.png
    └── top_features_comparison.png
```

## Contributing

To add new feature engineering approaches:

1. Extend `FeatureVersions` class with new version methods
2. Update `FeatureComparisonUtils` with new feature creation functions
3. Add new analysis methods to `RelevanceAnalyzer` if needed
4. Update report generation in `ComparisonReport`

## Notes

- The framework is designed to be modular and extensible
- All analysis methods include error handling and logging
- Reports are automatically timestamped and saved
- The framework works with both regression and classification tasks
- Sample data generation is included for testing purposes