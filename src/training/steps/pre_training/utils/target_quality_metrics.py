"""
Target Quality Metrics Calculator

This module provides comprehensive metrics to assess target variable quality,
predictability, and information content for trading signal generation.

Metrics Categories:
1. Variance & Distribution
2. Autocorrelation & Self-Consistency
3. Distribution & Outliers
4. Target Entropy
5. Naive Feature-Free Baselines
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import mean_squared_error


def calculate_target_quality_metrics(
    labels: Union[pd.Series, pd.DataFrame],
    market_data: Optional[pd.DataFrame] = None,
    bins: int = 10,
    max_lag: int = 10
) -> Dict[str, Any]:
    """
    Calculate comprehensive target quality metrics.

    Args:
        labels: Target labels (Series or DataFrame with target_long/target_short)
        market_data: Optional market data for context
        bins: Number of bins for entropy calculation
        max_lag: Maximum lag for autocorrelation calculation

    Returns:
        Dictionary containing all target quality metrics
    """
    metrics = {}

    try:
        # Handle DataFrame vs Series
        if isinstance(labels, pd.DataFrame):
            # Use target_long if available, otherwise first numeric column
            if 'target_long' in labels.columns:
                target_series = labels['target_long']
            elif 'target_short' in labels.columns:
                target_series = labels['target_short']
            else:
                target_series = labels.iloc[:, 0]
        else:
            target_series = labels

        # Convert to numpy array for calculations
        y = target_series.dropna().values

        if len(y) == 0:
            return _empty_metrics()

        # Special handling for constant/all-zero targets: these are deterministic
        # (zero entropy) but should not be flagged as "highly noisy" or "skewed".
        if len(y) > 0 and np.allclose(y, y[0]):
            variance_metrics = _calculate_variance_metrics(y)
            metrics['variance_distribution'] = variance_metrics

            # Autocorrelation: constant target → no structure, but also not "noisy".
            metrics['autocorrelation'] = {
                'lag1_autocorrelation': 0.0,
                'mean_autocorrelation': 0.0,
                'max_abs_autocorrelation': 0.0,
                'autocorrelation_by_lag': {},
                'is_highly_noisy': False,
                'has_temporal_structure': False,
                'interpretation': 'CONSTANT - Target is constant; autocorrelation not informative',
            }

            # Distribution: constant value → symmetric, no outliers.
            constant_val = float(y[0])
            metrics['distribution_outliers'] = {
                'percentile_5': constant_val,
                'percentile_25': constant_val,
                'median': constant_val,
                'percentile_75': constant_val,
                'percentile_95': constant_val,
                'iqr': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'n_outliers': 0,
                'outlier_percentage': 0.0,
                'is_symmetric': True,
                'is_heavy_tailed': False,
                'has_many_outliers': False,
                'interpretation': 'CONSTANT - Target is constant; no outliers',
            }

            # Entropy & baselines behave as usual on this deterministic series.
            entropy_metrics = _calculate_entropy_metrics(y, bins)
            metrics['entropy'] = entropy_metrics

            baseline_metrics = _calculate_baseline_metrics(y)
            metrics['baseline_predictors'] = baseline_metrics

            # Overall assessment based on these metrics
            metrics['overall_assessment'] = _assess_overall_quality(metrics)

            return metrics

        # 1. Variance & Distribution Metrics
        variance_metrics = _calculate_variance_metrics(y)
        metrics['variance_distribution'] = variance_metrics

        # 2. Autocorrelation & Self-Consistency
        autocorr_metrics = _calculate_autocorrelation_metrics(y, max_lag)
        metrics['autocorrelation'] = autocorr_metrics

        # 3. Distribution & Outliers
        distribution_metrics = _calculate_distribution_metrics(y)
        metrics['distribution_outliers'] = distribution_metrics

        # 4. Target Entropy
        entropy_metrics = _calculate_entropy_metrics(y, bins)
        metrics['entropy'] = entropy_metrics

        # 5. Naive Feature-Free Baselines
        baseline_metrics = _calculate_baseline_metrics(y)
        metrics['baseline_predictors'] = baseline_metrics

        # 6. Overall Quality Assessment
        metrics['overall_assessment'] = _assess_overall_quality(metrics)

        return metrics

    except Exception as e:
        print(f"Error calculating target quality metrics: {e}")
        return _empty_metrics()


def _calculate_variance_metrics(y: np.ndarray) -> Dict[str, Any]:
    """
    Calculate variance, standard deviation, and coefficient of variation.

    Why: If the target is nearly constant, the model cannot learn anything.
    """
    metrics = {}

    # Basic statistics
    mean = np.mean(y)
    variance = np.var(y)
    std = np.std(y)

    # Coefficient of variation (normalized volatility)
    # CV = std / mean (only meaningful if mean != 0)
    if abs(mean) > 1e-10:
        cv = std / abs(mean)
    else:
        cv = np.inf if std > 0 else 0.0

    # Range and spread
    min_val = np.min(y)
    max_val = np.max(y)
    range_val = max_val - min_val

    # Assess if target is nearly constant
    is_constant = variance < 1e-10
    has_variation = variance > 1e-6

    metrics.update({
        'mean': float(mean),
        'variance': float(variance),
        'std_deviation': float(std),
        'coefficient_of_variation': float(cv) if not np.isinf(cv) else 'inf',
        'min': float(min_val),
        'max': float(max_val),
        'range': float(range_val),
        'is_nearly_constant': bool(is_constant),
        'has_sufficient_variation': bool(has_variation),
        'interpretation': 'GOOD - Target has sufficient variation' if has_variation
                         else 'BAD - Target is nearly constant, model cannot learn'
    })

    return metrics


def _calculate_autocorrelation_metrics(y: np.ndarray, max_lag: int = 10) -> Dict[str, Any]:
    """
    Calculate autocorrelation for time-series consistency.

    Why: Random fluctuations → low autocorrelation → high inherent noise.
    """
    metrics = {}

    try:
        # Calculate autocorrelation for different lags
        autocorr_values = []
        for lag in range(1, min(max_lag + 1, len(y))):
            if lag < len(y):
                corr = np.corrcoef(y[:-lag], y[lag:])[0, 1]
                autocorr_values.append(float(corr) if not np.isnan(corr) else 0.0)
            else:
                autocorr_values.append(0.0)

        # Summary statistics
        if autocorr_values:
            lag1_autocorr = autocorr_values[0] if len(autocorr_values) > 0 else 0.0
            mean_autocorr = np.mean(autocorr_values)
            max_autocorr = np.max(np.abs(autocorr_values))

            # Assess noise level based on autocorrelation
            is_noisy = abs(lag1_autocorr) < 0.1
            has_structure = abs(lag1_autocorr) > 0.3

            metrics.update({
                'lag1_autocorrelation': float(lag1_autocorr),
                'mean_autocorrelation': float(mean_autocorr),
                'max_abs_autocorrelation': float(max_autocorr),
                'autocorrelation_by_lag': {f'lag_{i+1}': float(v) for i, v in enumerate(autocorr_values)},
                'is_highly_noisy': bool(is_noisy),
                'has_temporal_structure': bool(has_structure),
                'interpretation': 'GOOD - Target has temporal structure' if has_structure
                                 else 'MODERATE - Some noise present' if not is_noisy
                                 else 'BAD - Target is highly noisy/random'
            })
        else:
            metrics.update({
                'lag1_autocorrelation': 0.0,
                'mean_autocorrelation': 0.0,
                'max_abs_autocorrelation': 0.0,
                'autocorrelation_by_lag': {},
                'is_highly_noisy': True,
                'has_temporal_structure': False,
                'interpretation': 'INSUFFICIENT DATA - Cannot compute autocorrelation'
            })
    except Exception as e:
        metrics.update({
            'error': str(e),
            'interpretation': 'ERROR - Failed to compute autocorrelation'
        })

    return metrics


def _calculate_distribution_metrics(y: np.ndarray) -> Dict[str, Any]:
    """
    Calculate distribution shape, skewness, kurtosis, and outlier detection.

    Why: Extreme skew or outliers may distort labels.
    """
    metrics = {}

    try:
        # Calculate percentiles
        percentiles = np.percentile(y, [5, 25, 50, 75, 95])
        q1, q3 = percentiles[1], percentiles[3]
        iqr = q3 - q1

        # Outlier detection using IQR method
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = (y < lower_bound) | (y > upper_bound)
        n_outliers = np.sum(outliers)
        outlier_percentage = (n_outliers / len(y)) * 100

        # Distribution shape
        from scipy.stats import skew, kurtosis
        skewness = skew(y)
        kurt = kurtosis(y)

        # Assess distribution quality
        is_symmetric = abs(skewness) < 0.5
        is_heavy_tailed = kurt > 3
        has_many_outliers = outlier_percentage > 5

        metrics.update({
            'percentile_5': float(percentiles[0]),
            'percentile_25': float(percentiles[1]),
            'median': float(percentiles[2]),
            'percentile_75': float(percentiles[3]),
            'percentile_95': float(percentiles[4]),
            'iqr': float(iqr),
            'skewness': float(skewness),
            'kurtosis': float(kurt),
            'n_outliers': int(n_outliers),
            'outlier_percentage': float(outlier_percentage),
            'is_symmetric': bool(is_symmetric),
            'is_heavy_tailed': bool(is_heavy_tailed),
            'has_many_outliers': bool(has_many_outliers),
            'interpretation': 'GOOD - Symmetric distribution with few outliers' if is_symmetric and not has_many_outliers
                             else 'MODERATE - Some skewness or outliers present'
        })
    except Exception as e:
        metrics.update({
            'error': str(e),
            'interpretation': 'ERROR - Failed to compute distribution metrics'
        })

    return metrics


def _calculate_entropy_metrics(y: np.ndarray, bins: int = 10) -> Dict[str, Any]:
    """
    Calculate target entropy after discretization.

    Why: High randomness means hard to predict. High entropy → noisy or very diverse target.
    """
    metrics = {}

    try:
        # For continuous targets, discretize into bins
        hist, bin_edges = np.histogram(y, bins=bins, density=True)

        # Normalize histogram to get probabilities
        # Add small epsilon to avoid log(0)
        hist_normalized = hist + 1e-10
        hist_normalized = hist_normalized / np.sum(hist_normalized)

        # Calculate Shannon entropy
        target_entropy = scipy_entropy(hist_normalized)

        # Max possible entropy for uniform distribution
        max_entropy = np.log(bins)

        # Normalized entropy (0 to 1)
        normalized_entropy = target_entropy / max_entropy if max_entropy > 0 else 0.0

        # Assess predictability based on entropy
        is_predictable = normalized_entropy < 0.5
        is_highly_diverse = normalized_entropy > 0.8

        metrics.update({
            'shannon_entropy': float(target_entropy),
            'max_possible_entropy': float(max_entropy),
            'normalized_entropy': float(normalized_entropy),
            'bins_used': int(bins),
            'is_predictable': bool(is_predictable),
            'is_highly_diverse': bool(is_highly_diverse),
            'interpretation': 'GOOD - Low entropy, more predictable' if is_predictable
                             else 'MODERATE - Moderate diversity' if not is_highly_diverse
                             else 'BAD - High entropy, very noisy/diverse'
        })
    except Exception as e:
        metrics.update({
            'error': str(e),
            'interpretation': 'ERROR - Failed to compute entropy'
        })

    return metrics


def _calculate_baseline_metrics(y: np.ndarray) -> Dict[str, Any]:
    """
    Calculate performance of naive feature-free baseline predictors.

    Why: These baselines provide lower bounds on model performance.
         If a complex model can't beat these, something is wrong.
    """
    metrics = {}

    try:
        # Need at least 2 samples for train/test split
        if len(y) < 2:
            return {'error': 'Insufficient data for baseline metrics'}

        # 1. Mean predictor baseline
        # Predict the mean of all targets
        mean_predictor = np.mean(y)
        mean_predictions = np.full_like(y, mean_predictor)
        mean_mse = mean_squared_error(y, mean_predictions)
        mean_rmse = np.sqrt(mean_mse)

        # 2. Median predictor baseline
        median_predictor = np.median(y)
        median_predictions = np.full_like(y, median_predictor)
        median_mse = mean_squared_error(y, median_predictions)
        median_rmse = np.sqrt(median_mse)

        # 3. Persistence predictor (previous value)
        # Predict next value = current value
        if len(y) > 1:
            persistence_predictions = np.roll(y, 1)
            persistence_predictions[0] = y[0]  # First value stays same
            persistence_mse = mean_squared_error(y, persistence_predictions)
            persistence_rmse = np.sqrt(persistence_mse)
        else:
            persistence_mse = 0.0
            persistence_rmse = 0.0

        # 4. Random sampling baseline
        # Randomly sample from target distribution
        np.random.seed(42)  # For reproducibility
        random_predictions = np.random.choice(y, size=len(y), replace=True)
        random_mse = mean_squared_error(y, random_predictions)
        random_rmse = np.sqrt(random_mse)

        # 5. Zero predictor (predict all zeros)
        zero_predictions = np.zeros_like(y)
        zero_mse = mean_squared_error(y, zero_predictions)
        zero_rmse = np.sqrt(zero_mse)

        # Find best baseline
        baselines = {
            'mean': mean_mse,
            'median': median_mse,
            'persistence': persistence_mse,
            'random': random_mse,
            'zero': zero_mse
        }
        best_baseline = min(baselines.items(), key=lambda x: x[1])

        metrics.update({
            'mean_predictor': {
                'mse': float(mean_mse),
                'rmse': float(mean_rmse),
                'predicted_value': float(mean_predictor)
            },
            'median_predictor': {
                'mse': float(median_mse),
                'rmse': float(median_rmse),
                'predicted_value': float(median_predictor)
            },
            'persistence_predictor': {
                'mse': float(persistence_mse),
                'rmse': float(persistence_rmse),
                'description': 'Predict next value = current value'
            },
            'random_sampling_predictor': {
                'mse': float(random_mse),
                'rmse': float(random_rmse),
                'description': 'Random sampling from target distribution'
            },
            'zero_predictor': {
                'mse': float(zero_mse),
                'rmse': float(zero_rmse),
                'description': 'Predict all zeros'
            },
            'best_baseline': {
                'name': best_baseline[0],
                'mse': float(best_baseline[1])
            },
            'interpretation': f'Best naive baseline: {best_baseline[0]} (MSE={best_baseline[1]:.6f}). '
                             f'Model must beat this to be useful.'
        })
    except Exception as e:
        metrics.update({
            'error': str(e),
            'interpretation': 'ERROR - Failed to compute baseline metrics'
        })

    return metrics


def _assess_overall_quality(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Provide an overall assessment of target quality based on all metrics.
    """
    assessment = {
        'quality_score': 0.0,
        'quality_grade': 'UNKNOWN',
        'issues_detected': [],
        'strengths_identified': [],
        'recommendations': []
    }

    try:
        score = 100.0  # Start with perfect score

        # Variance checks
        variance_metrics = metrics.get('variance_distribution', {})
        if not variance_metrics.get('has_sufficient_variation', True):
            score -= 30
            assessment['issues_detected'].append('Target has insufficient variation')
            assessment['recommendations'].append('Check if target calculation is correct')
        else:
            assessment['strengths_identified'].append('Target has good variance')

        # Autocorrelation checks
        autocorr_metrics = metrics.get('autocorrelation', {})
        if autocorr_metrics.get('is_highly_noisy', False):
            score -= 25
            assessment['issues_detected'].append('Target is highly noisy with low autocorrelation')
            assessment['recommendations'].append('Consider label smoothing or noise reduction')
        elif autocorr_metrics.get('has_temporal_structure', False):
            assessment['strengths_identified'].append('Target has temporal structure')

        # Distribution checks
        dist_metrics = metrics.get('distribution_outliers', {})
        if dist_metrics.get('has_many_outliers', False):
            score -= 15
            assessment['issues_detected'].append('Target has many outliers (>5%)')
            assessment['recommendations'].append('Consider outlier removal or robust scaling')

        if not dist_metrics.get('is_symmetric', True):
            score -= 10
            assessment['issues_detected'].append('Target distribution is skewed')
            assessment['recommendations'].append('Consider data transformation or resampling')

        # Entropy checks
        entropy_metrics = metrics.get('entropy', {})
        if not entropy_metrics.get('is_predictable', True):
            score -= 20
            assessment['issues_detected'].append('Target has high entropy (low predictability)')
            assessment['recommendations'].append('Target may be too noisy for accurate prediction')
        else:
            assessment['strengths_identified'].append('Target has reasonable predictability')

        # Baseline checks
        baseline_metrics = metrics.get('baseline_predictors', {})
        best_baseline = baseline_metrics.get('best_baseline', {})
        if best_baseline.get('name') == 'zero':
            assessment['recommendations'].append('Zero predictor is best baseline - check target scaling')

        # Determine quality grade
        assessment['quality_score'] = max(0.0, score)
        if score >= 80:
            assessment['quality_grade'] = 'EXCELLENT'
        elif score >= 60:
            assessment['quality_grade'] = 'GOOD'
        elif score >= 40:
            assessment['quality_grade'] = 'FAIR'
        elif score >= 20:
            assessment['quality_grade'] = 'POOR'
        else:
            assessment['quality_grade'] = 'CRITICAL'

        # Add general recommendations
        if not assessment['recommendations']:
            assessment['recommendations'].append('Target quality is good - proceed with model training')

    except Exception as e:
        assessment['error'] = str(e)

    return assessment


def _empty_metrics() -> Dict[str, Any]:
    """Return empty metrics structure when calculation fails."""
    return {
        'variance_distribution': {},
        'autocorrelation': {},
        'distribution_outliers': {},
        'entropy': {},
        'baseline_predictors': {},
        'overall_assessment': {
            'quality_score': 0.0,
            'quality_grade': 'UNKNOWN',
            'issues_detected': ['Failed to calculate metrics'],
            'strengths_identified': [],
            'recommendations': ['Check data quality and format']
        }
    }
