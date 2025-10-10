# Multi-Target System Guide

This guide covers the comprehensive multi-target system that evaluates features against multiple targets including mean-reversion, trend-following, and other target families with proper metrics and reporting.

## Overview

The multi-target system evaluates features against 8 different target families with 24+ individual targets, providing a comprehensive assessment of feature performance across different market behaviors and prediction horizons.

## Target Families

### 1. Mean-Reversion Targets (H = 1, 2, 3)

**Idea**: Profit if future move opposes the current shock.

#### Regression Targets
- **MR Strength**: `y^MR_{t,H} = -r_t * R_{t->t+H}`
  - Positive when the next H bars reverse today's move
- **Risk-Adjusted MR**: `y~^MR_{t,H} = y^MR_{t,H} / σ_t(W)`
  - Normalized by rolling volatility

#### Classification Targets
- **MR Hit**: `y^MRcls_{t,H} = 1{sign(R_{t->t+H}) = -sign(r_t)}`
  - Binary indicator for opposite sign moves
- **MR Hit (Neutral)**: Same as above but ignores tiny |R| below threshold

**Metrics**: MAE/MSE, Pearson ρ for regression; AUC, F1, Brier for classification

### 2. Trend-Following Targets (H = 1, 2, 3)

**Idea**: Profit if future move continues the current direction.

#### Regression Targets
- **Trend Strength**: `y^TR_{t,H} = r_t * R_{t->t+H}`
  - Positive when trend continues
- **Risk-Adjusted Trend**: `y~^TR_{t,H} = y^TR_{t,H} / σ_t(W)`
  - Normalized by rolling volatility

#### Classification Targets
- **Trend Hit**: `y^TRcls_{t,H} = 1{sign(R_{t->t+H}) = sign(r_t)}`
  - Binary indicator for same sign moves
- **Trend Hit (Neutral)**: Same as above but ignores tiny |R| below threshold

**Metrics**: MAE/MSE, Pearson ρ for regression; AUC, F1, Brier for classification

### 3. Directional & Probability Targets

#### Binary Direction
- **Binary Direction**: `y^DIR_{t,H} = 1{R_{t->t+H} > 0}`
  - Simple up/down classification

#### Calibrated Probability
- **Calibrated Probability**: `Pr(R_{t->t+H} > 0)`
  - Rolling probability estimation
  - Evaluated with Brier/LogLoss + reliability curves

**Metrics**: AUC, F1, Brier, LogLoss, PR-AUC

### 4. Magnitude / Volatility Forecasting

#### Realized Volatility
- **Realized Vol**: `y^VOL_{t,H} = sqrt(sum(r_{t+i}^2))`
  - Sum of squared returns over horizon

#### Range-Based Volatility
- **Parkinson Vol**: Using high/low over horizon
  - `sqrt(0.25 * log(H/L)^2)`

**Metrics**: RMSE/MAE, MAPE, rank correlation

### 5. Tail Risk / Jump Likelihood

#### Left-Tail Events
- **Left-Tail**: `y^TAIL_{t,H} = 1{R_{t->t+H} < q_p}`
  - Historical quantile-based events (e.g., p=5%)

#### Right-Tail Events
- **Right-Tail**: `y^TAIL_{t,H} = 1{R_{t->t+H} > q_{1-p}}`
  - Upper tail events

**Metrics**: PR-AUC (imbalanced), recall@k

### 6. Breakout / Reversal Speed

#### VWAP Mean-Reversion Speed
- **VWAP MR Speed**: `y^VWAPMR_{t,H} = -basis_t * Δbasis_{t->t+H}`
  - Where `basis_t = P_t - VWAP_t`

#### Breakout Detection
- **Breakout Flag**: 1 if price exits rolling band (±kσ) and stays outside ≥M bars
  - Configurable std multiplier and minimum bars

**Metrics**: Precision@k, event F1

### 7. Risk-Adjusted Return

#### Sharpe-Like Metrics
- **Sharpe-Like**: `y^SR_{t,H} = R_{t->t+H} / σ_t(W)`
  - Risk-adjusted return over horizon

**Metrics**: MAE/MSE, Pearson ρ, rank correlation

### 8. Meta-Labeling (López de Prado)

#### Triple Barrier Method
- **Upper Barriers**: 0.6% and 1% profit-taking
- **Lower Barriers**: 0.3% and 0.5% profit-taking
- **Stop Loss**: 0.3%
- **Max Bars**: 3 bars maximum

**Meta-Labels**:
- 1.0: Hit upper barrier 2 or lower barrier 2
- 0.5: Hit upper barrier 1 or lower barrier 1
- -1.0: Hit stop loss
- 0.0: No barrier hit within max bars

**Metrics**: AUC, PR-AUC, realized hit ratio

## Configuration

### Timeframe and Horizons
```python
multi_target = MultiTargetSystem(
    horizons=[1, 2, 3],           # Prediction horizons in bars
    timeframe_minutes=15,         # 15-minute timeframe
    volatility_window=20,         # Window for volatility calculation
    neutral_threshold=0.001,      # Threshold for neutral zones
    tail_quantile=0.05,           # Quantile for tail risk events
    breakout_std_multiplier=2.0,  # Std multiplier for breakout detection
    breakout_min_bars=3,          # Minimum bars to stay outside band
    profit_taking_upper=0.006,    # 0.6% upper profit-taking
    profit_taking_lower=0.003,    # 0.3% lower profit-taking
    stop_loss=0.003,              # 0.3% stop loss
    max_bars=3                    # Maximum bars for triple barrier
)
```

## Usage Example

### Basic Usage with Data Loading

```python
from feature_comparison.multi_target_system import MultiTargetSystem
from datetime import datetime

# Initialize multi-target system
multi_target = MultiTargetSystem(
    horizons=[1, 2, 3],
    timeframe_minutes=15
)

# Run complete evaluation with automatic data loading
results = multi_target.run_complete_evaluation_with_data_loading(
    X=feature_matrix,
    symbol="ETHUSDT",
    interval="15m",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    data_type="raw",
    fallback_days=30
)

# Generate comprehensive report
report = multi_target.generate_multi_target_report(results)
print(report)
```

### Manual Data Loading

```python
from feature_comparison.multi_target_system import MultiTargetSystem

# Initialize multi-target system
multi_target = MultiTargetSystem(
    horizons=[1, 2, 3],
    timeframe_minutes=15
)

# Load market data using KlinesParquetManager
data = multi_target.load_market_data(
    symbol="ETHUSDT",
    interval="15m",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    data_type="raw",
    fallback_days=30
)

# Create all targets
targets = multi_target.create_all_targets(data)

# Evaluate features against all targets
results = multi_target.evaluate_features_against_targets(X, targets)

# Generate comprehensive report
report = multi_target.generate_multi_target_report(results)
print(report)
```

## Evaluation Metrics

### Regression Metrics
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of determination
- **Pearson Correlation**: Linear correlation
- **Spearman Correlation**: Rank correlation
- **MAPE**: Mean Absolute Percentage Error (if target > 0)

### Classification Metrics
- **AUC**: Area Under ROC Curve
- **F1 Score**: Harmonic mean of precision and recall
- **Brier Score**: Calibration measure
- **Log Loss**: Logarithmic loss
- **PR-AUC**: Area Under Precision-Recall Curve

### Overall Scoring
- **Regression**: `(abs(correlation) + abs(spearman_corr) + max(0, r2)) / 3`
- **Classification**: `(auc + f1 + (1 - brier)) / 3`

## Output Structure

### Target Families
```python
{
    'mean_reversion': DataFrame,      # MR targets
    'trend_following': DataFrame,     # Trend targets
    'directional': DataFrame,         # Direction targets
    'volatility': DataFrame,          # Volatility targets
    'tail_risk': DataFrame,           # Tail risk targets
    'breakout': DataFrame,            # Breakout targets
    'risk_adjusted': DataFrame,       # Risk-adjusted targets
    'meta_labeling': DataFrame        # Meta-labeling targets
}
```

### Evaluation Results
```python
{
    'target_families': {
        'family_name': {
            'targets': List[str],
            'feature_scores': Dict[str, Dict[str, float]],
            'regression_metrics': Dict[str, Dict[str, float]],
            'classification_metrics': Dict[str, Dict[str, float]],
            'best_features': List[str]
        }
    },
    'multi_target_summary': {
        'total_targets': int,
        'total_features_evaluated': int,
        'best_overall_features': List[Tuple[str, float]],
        'target_family_performance': Dict[str, Dict],
        'feature_consistency': Dict[str, float]
    },
    'best_features_by_target': Dict[str, List[Tuple[str, float]]],
    'correlation_analysis': {
        'target_correlations': Dict,
        'highly_correlated_pairs': List[Dict]
    }
}
```

## Key Features

### 1. Comprehensive Target Coverage
- **8 target families** with 24+ individual targets
- **Multiple horizons** (H=1,2,3) for each target
- **Regression and classification** variants where applicable

### 2. Proper Timeframe Handling
- **15-minute timeframe** with proper bar-based horizons
- **Neutral zones** to reduce label noise
- **Risk-adjusted** variants for all relevant targets

### 3. Advanced Target Types
- **Meta-labeling** with triple barrier method
- **Tail risk detection** with quantile-based events
- **Breakout detection** with microstructure awareness
- **VWAP-based** mean-reversion speed

### 4. Comprehensive Evaluation
- **Multiple metrics** for each target type
- **Feature consistency** across targets
- **Correlation analysis** between targets
- **Best features by target** identification

### 5. Production-Ready Features
- **Configurable parameters** for all target types
- **Robust error handling** for edge cases
- **Comprehensive reporting** with detailed analysis
- **Scalable evaluation** across large feature sets

### 6. KlinesParquetManager Integration
- **Real data loading** using KlinesParquetManager
- **Automatic fallback** to synthetic data if real data unavailable
- **Flexible data sources** (raw or processed data)
- **Date range filtering** with intelligent fallback
- **Data validation** and error handling

## Best Practices

### 1. Target Selection
- Use **multiple horizons** to test robustness
- Include both **regression and classification** variants
- Consider **risk-adjusted** targets for production use
- Use **neutral zones** to reduce noise

### 2. Feature Evaluation
- Evaluate features against **all target families**
- Look for **consistent performance** across targets
- Consider **target correlations** to avoid redundancy
- Use **comprehensive metrics** for each target type

### 3. Production Deployment
- Choose targets that align with **trading strategy**
- Consider **computational cost** of target calculation
- Monitor **target stability** over time
- Use **risk-adjusted** targets for position sizing

### 4. Model Selection
- Select features that perform well on **multiple targets**
- Prioritize **consistent features** over high-performing but unstable ones
- Consider **target-specific** feature selection
- Use **ensemble approaches** for different target types

## Integration with Other Systems

The multi-target system integrates seamlessly with other components:

```python
# With pre-screening pipeline
from feature_comparison.pre_screening_pipeline import PreScreeningPipeline
from feature_comparison.multi_target_system import MultiTargetSystem

# Pre-screen features
pipeline = PreScreeningPipeline()
prescreening_results = pipeline.run_pre_screening(X, y)

# Evaluate against multiple targets
multi_target = MultiTargetSystem()
targets = multi_target.create_all_targets(data)
results = multi_target.evaluate_features_against_targets(
    X, targets, prescreening_results['final_features']
)

# With acceleration and dilation
from feature_comparison.feature_acceleration_dilation_enhanced import EnhancedFeatureAccelerationDilation

# Generate variants
accel_dil_system = EnhancedFeatureAccelerationDilation()
variant_results = accel_dil_system.run_complete_evaluation(X, y)

# Evaluate variants against multiple targets
variant_target_results = multi_target.evaluate_features_against_targets(
    X, targets, variant_results['accepted_features']
)
```

This comprehensive multi-target system provides a robust foundation for evaluating features across different market behaviors and prediction horizons, ensuring that selected features are useful for a wide range of trading strategies and market conditions.