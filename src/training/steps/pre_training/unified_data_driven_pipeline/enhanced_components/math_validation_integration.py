"""
Math Validation Integration for UnifiedDataDrivenPipeline

This module provides comprehensive math validation integration for the unified pipeline,
enhancing all mathematical operations with robust error handling and validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import logging
import time
from dataclasses import dataclass

# Import math validation utilities
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_correlation, safe_covariance,
    safe_mean, safe_std, safe_percentile, safe_percentage_change,
    safe_weighted_average, safe_kelly_calculation, MathValidation,
    validate_numeric_array, safe_matrix_inverse, math_safe
)

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

@dataclass
class MathValidationResult:
    """Result of a math validation operation."""
    success: bool
    value: Any
    error_message: Optional[str] = None
    validation_time: float = 0.0
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

class MathValidationIntegration:
    """
    Comprehensive math validation integration for the unified pipeline.

    This class provides enhanced mathematical operations with robust validation,
    error handling, and performance monitoring for all pipeline calculations.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize math validation integration."""
        self.logger = logger or logging.getLogger(__name__)
        self.math_validator = MathValidation()
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'total_validation_time': 0.0
        }

        tprint_info("🔢 Math validation integration initialized")

    def validate_financial_metrics(self,
                                 returns: pd.Series,
                                 prices: Optional[pd.Series] = None,
                                 signals: Optional[pd.Series] = None) -> Dict[str, MathValidationResult]:
        """Validate and calculate financial metrics with comprehensive error handling."""
        results = {}
        start_time = time.time()

        try:
            # Validate input data
            returns = validate_finite(returns, "returns")
            if len(returns) == 0:
                raise ValueError("Returns series is empty")

            # Calculate Sharpe ratio
            results['sharpe_ratio'] = self._calculate_sharpe_ratio_safe(returns)

            # Calculate win rate
            results['win_rate'] = self._calculate_win_rate_safe(returns)

            # Calculate volatility
            results['volatility'] = self._calculate_volatility_safe(returns)

            # Calculate total return
            results['total_return'] = self._calculate_total_return_safe(returns)

            # Calculate additional metrics if prices and signals are provided
            if prices is not None and signals is not None:
                prices = validate_finite(prices, "prices")
                signals = validate_finite(signals, "signals")

                results['max_drawdown'] = self._calculate_max_drawdown_safe(prices, signals)
                results['calmar_ratio'] = self._calculate_calmar_ratio_safe(
                    results['total_return'].value,
                    results['max_drawdown'].value
                )

            # Calculate Sortino ratio
            results['sortino_ratio'] = self._calculate_sortino_ratio_safe(returns)

            self.validation_stats['successful_validations'] += 1

        except Exception as e:
            self.logger.error(f"Financial metrics validation failed: {e}")
            self.validation_stats['failed_validations'] += 1

            # Create failed results for all metrics
            for metric in ['sharpe_ratio', 'win_rate', 'volatility', 'total_return',
                          'max_drawdown', 'calmar_ratio', 'sortino_ratio']:
                if metric not in results:
                    results[metric] = MathValidationResult(
                        success=False,
                        value=0.0,
                        error_message=str(e)
                    )

        finally:
            self.validation_stats['total_validations'] += 1
            self.validation_stats['total_validation_time'] += time.time() - start_time

        return results

    def validate_statistical_metrics(self,
                                   data: pd.DataFrame,
                                   target: Optional[pd.Series] = None) -> Dict[str, MathValidationResult]:
        """Validate and calculate statistical metrics with comprehensive error handling."""
        results = {}
        start_time = time.time()

        try:
            # Validate input data
            data = validate_finite(data, "data")
            if len(data) == 0:
                raise ValueError("DataFrame is empty")

            # Calculate correlation matrix
            results['correlation_matrix'] = self._calculate_correlation_matrix_safe(data)

            # Calculate data quality score
            results['data_quality_score'] = self._calculate_data_quality_score_safe(data)

            # Calculate trend strength
            results['trend_strength'] = self._calculate_trend_strength_safe(data)

            # Calculate outliers ratio
            results['outliers_ratio'] = self._calculate_outliers_ratio_safe(data)

            # Calculate stability score
            results['stability_score'] = self._calculate_stability_score_safe(data)

            # Calculate additional metrics if target is provided
            if target is not None:
                target = validate_finite(target, "target")
                results['target_correlations'] = self._calculate_target_correlations_safe(data, target)
                results['mutual_information'] = self._calculate_mutual_information_safe(data, target)

            self.validation_stats['successful_validations'] += 1

        except Exception as e:
            self.logger.error(f"Statistical metrics validation failed: {e}")
            self.validation_stats['failed_validations'] += 1

            # Create failed results for all metrics
            for metric in ['correlation_matrix', 'data_quality_score', 'trend_strength',
                          'outliers_ratio', 'stability_score', 'target_correlations', 'mutual_information']:
                if metric not in results:
                    results[metric] = MathValidationResult(
                        success=False,
                        value=0.0,
                        error_message=str(e)
                    )

        finally:
            self.validation_stats['total_validations'] += 1
            self.validation_stats['total_validation_time'] += time.time() - start_time

        return results

    def validate_feature_metrics(self,
                               features: pd.DataFrame,
                               targets: Optional[pd.Series] = None) -> Dict[str, MathValidationResult]:
        """Validate and calculate feature metrics with comprehensive error handling."""
        results = {}
        start_time = time.time()

        try:
            # Validate input data
            features = validate_finite(features, "features")
            if len(features) == 0:
                raise ValueError("Features DataFrame is empty")

            # Calculate feature variance
            results['feature_variance'] = self._calculate_feature_variance_safe(features)

            # Calculate feature correlations
            results['feature_correlations'] = self._calculate_feature_correlations_safe(features)

            # Calculate feature importance (if targets provided)
            if targets is not None:
                targets = validate_finite(targets, "targets")
                results['feature_importance'] = self._calculate_feature_importance_safe(features, targets)
                results['feature_redundancy'] = self._calculate_feature_redundancy_safe(features, targets)

            # Calculate feature stability
            results['feature_stability'] = self._calculate_feature_stability_safe(features)

            self.validation_stats['successful_validations'] += 1

        except Exception as e:
            self.logger.error(f"Feature metrics validation failed: {e}")
            self.validation_stats['failed_validations'] += 1

            # Create failed results for all metrics
            for metric in ['feature_variance', 'feature_correlations', 'feature_importance',
                          'feature_redundancy', 'feature_stability']:
                if metric not in results:
                    results[metric] = MathValidationResult(
                        success=False,
                        value=0.0,
                        error_message=str(e)
                    )

        finally:
            self.validation_stats['total_validations'] += 1
            self.validation_stats['total_validation_time'] += time.time() - start_time

        return results

    def _calculate_sharpe_ratio_safe(self, returns: pd.Series) -> MathValidationResult:
        """Calculate Sharpe ratio with comprehensive validation."""
        start_time = time.time()

        try:
            mean_return = safe_mean(returns.values, default=0.0)
            std_return = safe_std(returns.values, default=0.0)

            sharpe_ratio = safe_divide(mean_return, std_return, default=0.0)
            sharpe_ratio = validate_range(sharpe_ratio, -10.0, 10.0, "sharpe_ratio")

            return MathValidationResult(
                success=True,
                value=float(sharpe_ratio),
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return MathValidationResult(
                success=False,
                value=0.0,
                error_message=str(e),
                validation_time=time.time() - start_time
            )

    def _calculate_win_rate_safe(self, returns: pd.Series) -> MathValidationResult:
        """Calculate win rate with comprehensive validation."""
        start_time = time.time()

        try:
            positive_returns = (returns > 0).astype(int)
            win_rate = safe_mean(positive_returns.values, default=0.0)
            win_rate = validate_range(win_rate, 0.0, 1.0, "win_rate")

            return MathValidationResult(
                success=True,
                value=float(win_rate),
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return MathValidationResult(
                success=False,
                value=0.0,
                error_message=str(e),
                validation_time=time.time() - start_time
            )

    def _calculate_volatility_safe(self, returns: pd.Series) -> MathValidationResult:
        """Calculate volatility with comprehensive validation."""
        start_time = time.time()

        try:
            volatility = safe_std(returns.values, default=0.0)
            volatility = validate_positive(volatility, "volatility")

            return MathValidationResult(
                success=True,
                value=float(volatility),
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return MathValidationResult(
                success=False,
                value=0.0,
                error_message=str(e),
                validation_time=time.time() - start_time
            )

    def _calculate_total_return_safe(self, returns: pd.Series) -> MathValidationResult:
        """Calculate total return with comprehensive validation."""
        start_time = time.time()

        try:
            total_return = float((1 + returns).prod() - 1)
            total_return = validate_finite(total_return, "total_return")

            return MathValidationResult(
                success=True,
                value=total_return,
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return MathValidationResult(
                success=False,
                value=0.0,
                error_message=str(e),
                validation_time=time.time() - start_time
            )

    def _calculate_max_drawdown_safe(self, prices: pd.Series, signals: pd.Series) -> MathValidationResult:
        """Calculate max drawdown with comprehensive validation."""
        start_time = time.time()

        try:
            returns = prices.pct_change().fillna(0.0)
            strategy_returns = signals.shift(1).fillna(0.0) * returns

            cumulative_returns = (1 + strategy_returns).cumprod()
            cumulative_returns = validate_finite(cumulative_returns, "cumulative_returns")

            running_max = cumulative_returns.expanding().max()
            drawdown = safe_divide(
                cumulative_returns - running_max,
                running_max,
                default=0.0
            )

            max_drawdown = safe_percentile(drawdown.values, 0.0, default=0.0)
            max_drawdown = validate_range(max_drawdown, -1.0, 0.0, "max_drawdown")

            return MathValidationResult(
                success=True,
                value=float(max_drawdown),
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return MathValidationResult(
                success=False,
                value=0.0,
                error_message=str(e),
                validation_time=time.time() - start_time
            )

    def _calculate_calmar_ratio_safe(self, total_return: float, max_drawdown: float) -> MathValidationResult:
        """Calculate Calmar ratio with comprehensive validation."""
        start_time = time.time()

        try:
            calmar_ratio = safe_divide(total_return, abs(max_drawdown), default=0.0)
            calmar_ratio = validate_finite(calmar_ratio, "calmar_ratio")

            return MathValidationResult(
                success=True,
                value=float(calmar_ratio),
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return MathValidationResult(
                success=False,
                value=0.0,
                error_message=str(e),
                validation_time=time.time() - start_time
            )

    def _calculate_sortino_ratio_safe(self, returns: pd.Series) -> MathValidationResult:
        """Calculate Sortino ratio with comprehensive validation."""
        start_time = time.time()

        try:
            mean_return = safe_mean(returns.values, default=0.0)
            negative_returns = returns[returns < 0]

            if len(negative_returns) == 0:
                return MathValidationResult(
                    success=True,
                    value=0.0,
                    validation_time=time.time() - start_time
                )

            downside_std = safe_std(negative_returns.values, default=0.0)
            sortino_ratio = safe_divide(mean_return, downside_std, default=0.0)
            sortino_ratio = validate_finite(sortino_ratio, "sortino_ratio")

            return MathValidationResult(
                success=True,
                value=float(sortino_ratio),
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return MathValidationResult(
                success=False,
                value=0.0,
                error_message=str(e),
                validation_time=time.time() - start_time
            )

    def _calculate_correlation_matrix_safe(self, data: pd.DataFrame) -> MathValidationResult:
        """Calculate correlation matrix with comprehensive validation."""
        start_time = time.time()

        try:
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data) == 0 or len(data) == 0:
                return MathValidationResult(
                    success=False,
                    value=pd.DataFrame(),
                    error_message="No numeric columns found",
                    validation_time=time.time() - start_time
                )

            corr_matrix = numeric_data.corr()
            corr_matrix = validate_finite(corr_matrix, "correlation_matrix")

            return MathValidationResult(
                success=True,
                value=corr_matrix,
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return MathValidationResult(
                success=False,
                value=pd.DataFrame(),
                error_message=str(e),
                validation_time=time.time() - start_time
            )

    def _calculate_data_quality_score_safe(self, data: pd.DataFrame) -> MathValidationResult:
        """Calculate data quality score with comprehensive validation."""
        start_time = time.time()

        try:
            # Calculate missing data ratio
            missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
            missing_ratio = validate_range(missing_ratio, 0.0, 1.0, "missing_ratio")

            # Calculate constant columns ratio
            constant_cols = (data.nunique() <= 1).sum()
            constant_ratio = constant_cols / len(data.columns)
            constant_ratio = validate_range(constant_ratio, 0.0, 1.0, "constant_ratio")

            # Calculate quality score
            quality_score = 1.0 - missing_ratio - constant_ratio
            quality_score = validate_range(quality_score, 0.0, 1.0, "quality_score")

            return MathValidationResult(
                success=True,
                value=float(quality_score),
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return MathValidationResult(
                success=False,
                value=0.0,
                error_message=str(e),
                validation_time=time.time() - start_time
            )

    def _calculate_trend_strength_safe(self, data: pd.DataFrame) -> MathValidationResult:
        """Calculate trend strength with comprehensive validation."""
        start_time = time.time()

        try:
            trend_strengths = []

            for col in data.select_dtypes(include=[np.number]).columns:
                series = data[col].dropna()
                if len(series) < 10:
                    continue

                series = validate_finite(series, f"series_{col}")
                if len(series) < 10:
                    continue

                # Calculate trend using linear regression
                x = np.arange(len(series))
                try:
                    from scipy import stats
                    slope, _, r_value, _, _ = stats.linregress(x, series)
                    r_abs = abs(validate_finite(r_value, f"r_value_{col}"))
                    r_abs = validate_range(r_abs, 0.0, 1.0, f"r_value_{col}")
                    trend_strengths.append(r_abs)
                except Exception:
                    continue

            trend_strength = safe_mean(trend_strengths, default=0.0)
            trend_strength = validate_range(trend_strength, 0.0, 1.0, "trend_strength")

            return MathValidationResult(
                success=True,
                value=float(trend_strength),
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return MathValidationResult(
                success=False,
                value=0.0,
                error_message=str(e),
                validation_time=time.time() - start_time
            )

    def _calculate_outliers_ratio_safe(self, data: pd.DataFrame) -> MathValidationResult:
        """Calculate outliers ratio with comprehensive validation."""
        start_time = time.time()

        try:
            outlier_ratios = []

            for col in data.select_dtypes(include=[np.number]).columns:
                series = data[col].dropna()
                if len(series) < 10:
                    continue

                series = validate_finite(series, f"series_{col}")
                if len(series) < 10:
                    continue

                # Use IQR method for outlier detection
                Q1 = safe_percentile(series.values, 25.0, default=0.0)
                Q3 = safe_percentile(series.values, 75.0, default=0.0)
                IQR = Q3 - Q1

                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = ((series < lower_bound) | (series > upper_bound)).sum()
                    outlier_ratio = outliers / len(series)
                    outlier_ratio = validate_range(outlier_ratio, 0.0, 1.0, f"outlier_ratio_{col}")
                    outlier_ratios.append(outlier_ratio)

            outliers_ratio = safe_mean(outlier_ratios, default=0.0)
            outliers_ratio = validate_range(outliers_ratio, 0.0, 1.0, "outliers_ratio")

            return MathValidationResult(
                success=True,
                value=float(outliers_ratio),
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return MathValidationResult(
                success=False,
                value=0.0,
                error_message=str(e),
                validation_time=time.time() - start_time
            )

    def _calculate_stability_score_safe(self, data: pd.DataFrame) -> MathValidationResult:
        """Calculate stability score with comprehensive validation."""
        start_time = time.time()

        try:
            stability_scores = []

            for col in data.select_dtypes(include=[np.number]).columns:
                series = data[col].dropna()
                if len(series) < 10:
                    continue

                series = validate_finite(series, f"series_{col}")
                if len(series) < 10:
                    continue

                # Calculate rolling standard deviation
                rolling_std = series.rolling(window=min(10, len(series)//2)).std()
                rolling_std = rolling_std.dropna()

                if len(rolling_std) > 0:
                    std_of_std = safe_std(rolling_std.values, default=0.0)
                    mean_std = safe_mean(rolling_std.values, default=0.0)

                    if mean_std > 0:
                        stability = 1.0 - safe_divide(std_of_std, mean_std, default=1.0)
                        stability = validate_range(stability, 0.0, 1.0, f"stability_{col}")
                        stability_scores.append(stability)

            stability_score = safe_mean(stability_scores, default=0.0)
            stability_score = validate_range(stability_score, 0.0, 1.0, "stability_score")

            return MathValidationResult(
                success=True,
                value=float(stability_score),
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return MathValidationResult(
                success=False,
                value=0.0,
                error_message=str(e),
                validation_time=time.time() - start_time
            )

    def _calculate_target_correlations_safe(self, data: pd.DataFrame, target: pd.Series) -> MathValidationResult:
        """Calculate target correlations with comprehensive validation."""
        start_time = time.time()

        try:
            correlations = {}

            for col in data.select_dtypes(include=[np.number]).columns:
                series = data[col].dropna()
                if len(series) < 10:
                    continue

                # Align series with target
                common_index = series.index.intersection(target.index)
                if len(common_index) < 10:
                    continue

                series_aligned = series.loc[common_index]
                target_aligned = target.loc[common_index]

                series_aligned = validate_finite(series_aligned, f"series_{col}")
                target_aligned = validate_finite(target_aligned, "target")

                if len(series_aligned) < 10:
                    continue

                correlation = safe_correlation(series_aligned.values, target_aligned.values, default=0.0)
                correlation = validate_range(correlation, -1.0, 1.0, f"correlation_{col}")
                correlations[col] = correlation

            return MathValidationResult(
                success=True,
                value=correlations,
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return MathValidationResult(
                success=False,
                value={},
                error_message=str(e),
                validation_time=time.time() - start_time
            )

    def _calculate_mutual_information_safe(self, data: pd.DataFrame, target: pd.Series) -> MathValidationResult:
        """Calculate mutual information with comprehensive validation."""
        start_time = time.time()

        try:
            from sklearn.feature_selection import mutual_info_regression

            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data) == 0 or len(data) == 0:
                return MathValidationResult(
                    success=False,
                    value={},
                    error_message="No numeric columns found",
                    validation_time=time.time() - start_time
                )

            # Align data with target
            common_index = numeric_data.index.intersection(target.index)
            if len(common_index) < 10:
                return MathValidationResult(
                    success=False,
                    value={},
                    error_message="Insufficient common data points",
                    validation_time=time.time() - start_time
                )

            data_aligned = numeric_data.loc[common_index]
            target_aligned = target.loc[common_index]

            data_aligned = validate_finite(data_aligned, "data")
            target_aligned = validate_finite(target_aligned, "target")

            # Calculate mutual information
            mi_scores = mutual_info_regression(data_aligned, target_aligned)

            mutual_info = {}
            for i, col in enumerate(data_aligned.columns):
                mi_score = validate_finite(mi_scores[i], f"mi_{col}")
                mi_score = validate_positive(mi_score, f"mi_{col}")
                mutual_info[col] = mi_score

            return MathValidationResult(
                success=True,
                value=mutual_info,
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return MathValidationResult(
                success=False,
                value={},
                error_message=str(e),
                validation_time=time.time() - start_time
            )

    def _calculate_feature_variance_safe(self, features: pd.DataFrame) -> MathValidationResult:
        """Calculate feature variance with comprehensive validation."""
        start_time = time.time()

        try:
            variances = {}

            for col in features.select_dtypes(include=[np.number]).columns:
                series = features[col].dropna()
                if len(series) < 2:
                    continue

                series = validate_finite(series, f"series_{col}")
                if len(series) < 2:
                    continue

                variance = safe_std(series.values, default=0.0) ** 2
                variance = validate_positive(variance, f"variance_{col}")
                variances[col] = variance

            return MathValidationResult(
                success=True,
                value=variances,
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return MathValidationResult(
                success=False,
                value={},
                error_message=str(e),
                validation_time=time.time() - start_time
            )

    def _calculate_feature_correlations_safe(self, features: pd.DataFrame) -> MathValidationResult:
        """Calculate feature correlations with comprehensive validation."""
        start_time = time.time()

        try:
            numeric_features = features.select_dtypes(include=[np.number])
            if numeric_features.empty:
                return MathValidationResult(
                    success=False,
                    value=pd.DataFrame(),
                    error_message="No numeric columns found",
                    validation_time=time.time() - start_time
                )

            corr_matrix = numeric_features.corr()
            corr_matrix = validate_finite(corr_matrix, "correlation_matrix")

            return MathValidationResult(
                success=True,
                value=corr_matrix,
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return MathValidationResult(
                success=False,
                value=pd.DataFrame(),
                error_message=str(e),
                validation_time=time.time() - start_time
            )

    def _calculate_feature_importance_safe(self, features: pd.DataFrame, targets: pd.Series) -> MathValidationResult:
        """Calculate feature importance with comprehensive validation."""
        start_time = time.time()

        try:
            from sklearn.ensemble import RandomForestRegressor

            numeric_features = features.select_dtypes(include=[np.number])
            if numeric_features.empty:
                return MathValidationResult(
                    success=False,
                    value={},
                    error_message="No numeric columns found",
                    validation_time=time.time() - start_time
                )

            # Align data with targets
            common_index = numeric_features.index.intersection(targets.index)
            if len(common_index) < 10:
                return MathValidationResult(
                    success=False,
                    value={},
                    error_message="Insufficient common data points",
                    validation_time=time.time() - start_time
                )

            features_aligned = numeric_features.loc[common_index]
            targets_aligned = targets.loc[common_index]

            features_aligned = validate_finite(features_aligned, "features")
            targets_aligned = validate_finite(targets_aligned, "targets")

            # Calculate feature importance
            rf = RandomForestRegressor(n_estimators=10, random_state=42)
            rf.fit(features_aligned, targets_aligned)

            importance = {}
            for i, col in enumerate(features_aligned.columns):
                imp_score = validate_finite(rf.feature_importances_[i], f"importance_{col}")
                imp_score = validate_positive(imp_score, f"importance_{col}")
                importance[col] = imp_score

            return MathValidationResult(
                success=True,
                value=importance,
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return MathValidationResult(
                success=False,
                value={},
                error_message=str(e),
                validation_time=time.time() - start_time
            )

    def _calculate_feature_redundancy_safe(self, features: pd.DataFrame, targets: pd.Series) -> MathValidationResult:
        """Calculate feature redundancy with comprehensive validation."""
        start_time = time.time()

        try:
            numeric_features = features.select_dtypes(include=[np.number])
            if numeric_features.empty:
                return MathValidationResult(
                    success=False,
                    value=0.0,
                    error_message="No numeric columns found",
                    validation_time=time.time() - start_time
                )

            # Calculate correlation matrix
            corr_matrix = numeric_features.corr()
            corr_matrix = validate_finite(corr_matrix, "correlation_matrix")

            # Calculate average absolute correlation (excluding diagonal)
            mask = np.ones_like(corr_matrix, dtype=bool)
            np.fill_diagonal(mask, False)

            avg_correlation = safe_mean(corr_matrix.values[mask], default=0.0)
            avg_correlation = validate_range(avg_correlation, 0.0, 1.0, "avg_correlation")

            return MathValidationResult(
                success=True,
                value=float(avg_correlation),
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return MathValidationResult(
                success=False,
                value=0.0,
                error_message=str(e),
                validation_time=time.time() - start_time
            )

    def _calculate_feature_stability_safe(self, features: pd.DataFrame) -> MathValidationResult:
        """Calculate feature stability with comprehensive validation."""
        start_time = time.time()

        try:
            stability_scores = []

            for col in features.select_dtypes(include=[np.number]).columns:
                series = features[col].dropna()
                if len(series) < 20:
                    continue

                series = validate_finite(series, f"series_{col}")
                if len(series) < 20:
                    continue

                # Split series into two halves
                mid_point = len(series) // 2
                first_half = series.iloc[:mid_point]
                second_half = series.iloc[mid_point:]

                if len(first_half) < 10 or len(second_half) < 10:
                    continue

                # Calculate statistics for each half
                mean1 = safe_mean(first_half.values, default=0.0)
                mean2 = safe_mean(second_half.values, default=0.0)
                std1 = safe_std(first_half.values, default=0.0)
                std2 = safe_std(second_half.values, default=0.0)

                # Calculate stability as inverse of relative change
                if std1 > 0 and std2 > 0:
                    mean_change = abs(mean2 - mean1) / ((abs(mean1) + abs(mean2)) / 2)
                    std_change = abs(std2 - std1) / ((std1 + std2) / 2)

                    stability = 1.0 - (mean_change + std_change) / 2
                    stability = validate_range(stability, 0.0, 1.0, f"stability_{col}")
                    stability_scores.append(stability)

            overall_stability = safe_mean(stability_scores, default=0.0)
            overall_stability = validate_range(overall_stability, 0.0, 1.0, "overall_stability")

            return MathValidationResult(
                success=True,
                value=float(overall_stability),
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return MathValidationResult(
                success=False,
                value=0.0,
                error_message=str(e),
                validation_time=time.time() - start_time
            )

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            'total_validations': self.validation_stats['total_validations'],
            'successful_validations': self.validation_stats['successful_validations'],
            'failed_validations': self.validation_stats['failed_validations'],
            'success_rate': safe_divide(
                self.validation_stats['successful_validations'],
                self.validation_stats['total_validations'],
                default=0.0
            ),
            'total_validation_time': self.validation_stats['total_validation_time'],
            'average_validation_time': safe_divide(
                self.validation_stats['total_validation_time'],
                self.validation_stats['total_validations'],
                default=0.0
            )
        }

    def reset_stats(self):
        """Reset validation statistics."""
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'total_validation_time': 0.0
        }

# Convenience functions
def create_math_validation_integration(logger: Optional[logging.Logger] = None) -> MathValidationIntegration:
    """Create a math validation integration instance."""
    return MathValidationIntegration(logger)

def validate_pipeline_calculation(func: Callable) -> Callable:
    """Decorator to validate pipeline calculations with math validation."""
    def wrapper(*args, **kwargs):
        try:
            # Create math validation integration
            math_validator = MathValidationIntegration()

            # Execute the function
            result = func(*args, **kwargs)

            # Validate the result if it's numeric
            if isinstance(result, (int, float, np.number)):
                result = validate_finite(result, "calculation_result")

            return result

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Pipeline calculation validation failed: {e}")
            return None

    return wrapper

# Export main classes and functions
__all__ = [
    'MathValidationIntegration',
    'MathValidationResult',
    'create_math_validation_integration',
    'validate_pipeline_calculation'
]
