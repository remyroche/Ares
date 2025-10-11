"""
VectorBT Feature Validation System

This module provides comprehensive validation capabilities for VectorBT
feature engineering components with advanced testing and quality assurance.

Features:
- Statistical validation and significance testing
- Backtesting and performance validation
- Cross-validation and robustness testing
- Feature stability and consistency analysis
- Out-of-sample testing and walk-forward analysis
- Quality metrics and reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
import warnings
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import VectorBT base classes
from src.training.steps.feature_engineering.vectorbt_base import (
    VectorBTFeatureGenerator, VectorBTConfig, VectorBTTechnicalIndicators
)

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success


class ValidationType(Enum):
    """Types of validation tests."""
    STATISTICAL = "statistical"
    PERFORMANCE = "performance"
    STABILITY = "stability"
    ROBUSTNESS = "robustness"
    OUT_OF_SAMPLE = "out_of_sample"
    WALK_FORWARD = "walk_forward"
    CROSS_VALIDATION = "cross_validation"


class ValidationMetric(Enum):
    """Validation metrics."""
    SHARPE_RATIO = "sharpe_ratio"
    INFORMATION_RATIO = "information_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    RETURN = "return"
    STABILITY = "stability"
    CONSISTENCY = "consistency"
    SIGNIFICANCE = "significance"


@dataclass
class VectorBTValidationConfig:
    """Configuration for VectorBT feature validation."""
    
    # Validation types to run
    enable_statistical_validation: bool = True
    enable_performance_validation: bool = True
    enable_stability_validation: bool = True
    enable_robustness_validation: bool = True
    enable_out_of_sample_validation: bool = True
    enable_walk_forward_validation: bool = True
    enable_cross_validation: bool = True
    
    # Statistical validation settings
    significance_level: float = 0.05
    min_correlation_threshold: float = 0.1
    max_correlation_threshold: float = 0.9
    
    # Performance validation settings
    min_sharpe_ratio: float = 0.5
    max_drawdown_threshold: float = 0.2
    min_information_ratio: float = 0.1
    
    # Stability validation settings
    stability_window: int = 50
    stability_threshold: float = 0.1
    consistency_threshold: float = 0.8
    
    # Cross-validation settings
    cv_folds: int = 5
    cv_strategy: str = "time_series"  # "time_series", "k_fold", "walk_forward"
    
    # Out-of-sample settings
    train_ratio: float = 0.7
    test_ratio: float = 0.3
    validation_ratio: float = 0.2
    
    # Walk-forward settings
    initial_train_size: int = 1000
    step_size: int = 100
    min_test_size: int = 100
    
    # Quality thresholds
    min_quality_score: float = 0.6
    max_failure_rate: float = 0.3
    
    # Performance settings
    enable_parallel: bool = True
    n_jobs: int = -1
    chunk_size: int = 1000


@dataclass
class ValidationResult:
    """Result of feature validation."""
    
    # Core validation results
    validation_passed: bool
    overall_score: float
    validation_details: Dict[str, Any]
    
    # Statistical validation results
    statistical_tests: Dict[str, Any]
    correlation_analysis: Dict[str, Any]
    significance_tests: Dict[str, Any]
    
    # Performance validation results
    performance_metrics: Dict[str, float]
    backtest_results: Dict[str, Any]
    risk_metrics: Dict[str, float]
    
    # Stability validation results
    stability_metrics: Dict[str, float]
    consistency_metrics: Dict[str, float]
    robustness_metrics: Dict[str, float]
    
    # Cross-validation results
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    
    # Out-of-sample results
    oos_performance: Dict[str, float]
    oos_metrics: Dict[str, Any]
    
    # Walk-forward results
    walk_forward_results: Dict[str, Any]
    walk_forward_metrics: Dict[str, float]
    
    # Quality assessment
    quality_score: float
    failure_rate: float
    recommendations: List[str]
    
    # Metadata
    feature_name: str
    validation_time: float
    validation_timestamp: datetime
    config_used: VectorBTValidationConfig


class VectorBTFeatureValidator:
    """
    Comprehensive feature validator for VectorBT features.
    
    Provides extensive validation capabilities including statistical testing,
    performance validation, stability analysis, and quality assessment.
    """
    
    def __init__(self, config: Optional[VectorBTValidationConfig] = None):
        """Initialize VectorBT feature validator."""
        self.config = config or VectorBTValidationConfig()
        self.logger = logging.getLogger('VectorBTFeatureValidator')
        
        # Validation history
        self.validation_history: List[ValidationResult] = []
        self.current_validation: Optional[ValidationResult] = None
        
        tprint_info("🔍 VectorBT Feature Validator initialized")
        tprint_info(f"   → Statistical validation: {self.config.enable_statistical_validation}")
        tprint_info(f"   → Performance validation: {self.config.enable_performance_validation}")
        tprint_info(f"   → Stability validation: {self.config.enable_stability_validation}")
        tprint_info(f"   → Out-of-sample validation: {self.config.enable_out_of_sample_validation}")
        tprint_info(f"   → Walk-forward validation: {self.config.enable_walk_forward_validation}")
    
    def validate_feature(
        self,
        feature_generator: VectorBTFeatureGenerator,
        data: pd.DataFrame,
        parameters: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Perform comprehensive validation of a VectorBT feature.
        
        Args:
            feature_generator: VectorBT feature generator to validate
            data: Input data for validation
            parameters: Optional parameters for the feature generator
            
        Returns:
            ValidationResult with comprehensive validation results
        """
        start_time = time.time()
        tprint_info(f"🔍 Validating feature: {feature_generator.config.name}")
        
        try:
            # Initialize validation result
            validation_result = ValidationResult(
                validation_passed=False,
                overall_score=0.0,
                validation_details={},
                statistical_tests={},
                correlation_analysis={},
                significance_tests={},
                performance_metrics={},
                backtest_results={},
                risk_metrics={},
                stability_metrics={},
                consistency_metrics={},
                robustness_metrics={},
                cv_scores=[],
                cv_mean=0.0,
                cv_std=0.0,
                oos_performance={},
                oos_metrics={},
                walk_forward_results={},
                walk_forward_metrics={},
                quality_score=0.0,
                failure_rate=0.0,
                recommendations=[],
                feature_name=feature_generator.config.name,
                validation_time=0.0,
                validation_timestamp=datetime.now(),
                config_used=self.config
            )
            
            # Generate features
            features = feature_generator.generate_vectorbt_features(data, parameters)
            
            # Run validation tests
            validation_scores = []
            
            # Statistical validation
            if self.config.enable_statistical_validation:
                tprint_info("📊 Running statistical validation")
                stat_results = self._run_statistical_validation(features, data)
                validation_result.statistical_tests = stat_results['tests']
                validation_result.correlation_analysis = stat_results['correlation']
                validation_result.significance_tests = stat_results['significance']
                validation_scores.append(stat_results['score'])
            
            # Performance validation
            if self.config.enable_performance_validation:
                tprint_info("📈 Running performance validation")
                perf_results = self._run_performance_validation(features, data)
                validation_result.performance_metrics = perf_results['metrics']
                validation_result.backtest_results = perf_results['backtest']
                validation_result.risk_metrics = perf_results['risk']
                validation_scores.append(perf_results['score'])
            
            # Stability validation
            if self.config.enable_stability_validation:
                tprint_info("🔒 Running stability validation")
                stability_results = self._run_stability_validation(features, data)
                validation_result.stability_metrics = stability_results['stability']
                validation_result.consistency_metrics = stability_results['consistency']
                validation_scores.append(stability_results['score'])
            
            # Robustness validation
            if self.config.enable_robustness_validation:
                tprint_info("🛡️ Running robustness validation")
                robustness_results = self._run_robustness_validation(feature_generator, data, parameters)
                validation_result.robustness_metrics = robustness_results['metrics']
                validation_scores.append(robustness_results['score'])
            
            # Cross-validation
            if self.config.enable_cross_validation:
                tprint_info("🔄 Running cross-validation")
                cv_results = self._run_cross_validation(feature_generator, data, parameters)
                validation_result.cv_scores = cv_results['scores']
                validation_result.cv_mean = cv_results['mean']
                validation_result.cv_std = cv_results['std']
                validation_scores.append(cv_results['score'])
            
            # Out-of-sample validation
            if self.config.enable_out_of_sample_validation:
                tprint_info("📤 Running out-of-sample validation")
                oos_results = self._run_out_of_sample_validation(feature_generator, data, parameters)
                validation_result.oos_performance = oos_results['performance']
                validation_result.oos_metrics = oos_results['metrics']
                validation_scores.append(oos_results['score'])
            
            # Walk-forward validation
            if self.config.enable_walk_forward_validation:
                tprint_info("🚶 Running walk-forward validation")
                wf_results = self._run_walk_forward_validation(feature_generator, data, parameters)
                validation_result.walk_forward_results = wf_results['results']
                validation_result.walk_forward_metrics = wf_results['metrics']
                validation_scores.append(wf_results['score'])
            
            # Calculate overall score
            if validation_scores:
                validation_result.overall_score = np.mean(validation_scores)
                validation_result.quality_score = validation_result.overall_score
            else:
                validation_result.overall_score = 0.0
                validation_result.quality_score = 0.0
            
            # Determine if validation passed
            validation_result.validation_passed = (
                validation_result.overall_score >= self.config.min_quality_score and
                validation_result.failure_rate <= self.config.max_failure_rate
            )
            
            # Generate recommendations
            validation_result.recommendations = self._generate_recommendations(validation_result)
            
            # Calculate validation time
            validation_result.validation_time = time.time() - start_time
            
            # Store results
            self.validation_history.append(validation_result)
            self.current_validation = validation_result
            
            # Log results
            status = "✅ PASSED" if validation_result.validation_passed else "❌ FAILED"
            tprint_success(f"🔍 Validation completed: {status} (Score: {validation_result.overall_score:.3f})")
            
            return validation_result
            
        except Exception as e:
            tprint_error(f"❌ Error validating feature: {e}")
            raise
    
    def _run_statistical_validation(
        self, 
        features: Dict[str, pd.Series], 
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run statistical validation tests."""
        try:
            results = {
                'tests': {},
                'correlation': {},
                'significance': {},
                'score': 0.0
            }
            
            # Get numeric features
            numeric_features = {
                name: series for name, series in features.items()
                if isinstance(series, pd.Series) and pd.api.types.is_numeric_dtype(series)
            }
            
            if not numeric_features:
                return results
            
            # Statistical tests
            test_results = {}
            
            for name, series in numeric_features.items():
                clean_series = series.dropna()
                if len(clean_series) < 10:
                    continue
                
                # Normality test
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(clean_series)
                    test_results[f'{name}_normality'] = {
                        'statistic': shapiro_stat,
                        'p_value': shapiro_p,
                        'is_normal': shapiro_p > self.config.significance_level
                    }
                except Exception:
                    test_results[f'{name}_normality'] = {'is_normal': False}
                
                # Stationarity test (ADF)
                try:
                    from statsmodels.tsa.stattools import adfuller
                    adf_stat, adf_p, _, _, adf_critical, _ = adfuller(clean_series)
                    test_results[f'{name}_stationarity'] = {
                        'statistic': adf_stat,
                        'p_value': adf_p,
                        'is_stationary': adf_p < self.config.significance_level,
                        'critical_values': adf_critical
                    }
                except Exception:
                    test_results[f'{name}_stationarity'] = {'is_stationary': False}
                
                # Autocorrelation test
                try:
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    lb_stat, lb_p = acorr_ljungbox(clean_series, lags=10, return_df=False)
                    test_results[f'{name}_autocorrelation'] = {
                        'statistic': lb_stat,
                        'p_value': lb_p,
                        'has_autocorrelation': lb_p < self.config.significance_level
                    }
                except Exception:
                    test_results[f'{name}_autocorrelation'] = {'has_autocorrelation': False}
            
            results['tests'] = test_results
            
            # Correlation analysis
            if len(numeric_features) > 1:
                feature_df = pd.DataFrame(numeric_features)
                correlation_matrix = feature_df.corr()
                
                # Check for high correlations
                high_correlations = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_val = abs(correlation_matrix.iloc[i, j])
                        if corr_val > self.config.max_correlation_threshold:
                            high_correlations.append({
                                'feature1': correlation_matrix.columns[i],
                                'feature2': correlation_matrix.columns[j],
                                'correlation': corr_val
                            })
                
                results['correlation'] = {
                    'matrix': correlation_matrix.to_dict(),
                    'high_correlations': high_correlations,
                    'max_correlation': correlation_matrix.abs().max().max()
                }
            
            # Significance tests
            significance_results = {}
            for name, series in numeric_features.items():
                clean_series = series.dropna()
                if len(clean_series) < 10:
                    continue
                
                # T-test against zero
                try:
                    t_stat, t_p = stats.ttest_1samp(clean_series, 0)
                    significance_results[f'{name}_ttest'] = {
                        'statistic': t_stat,
                        'p_value': t_p,
                        'is_significant': t_p < self.config.significance_level
                    }
                except Exception:
                    significance_results[f'{name}_ttest'] = {'is_significant': False}
            
            results['significance'] = significance_results
            
            # Calculate score
            score_components = []
            
            # Normality score
            normal_count = sum(1 for test in test_results.values() if test.get('is_normal', False))
            if test_results:
                score_components.append(normal_count / len(test_results))
            
            # Stationarity score
            stationary_count = sum(1 for test in test_results.values() if test.get('is_stationary', False))
            if test_results:
                score_components.append(stationary_count / len(test_results))
            
            # Significance score
            significant_count = sum(1 for test in significance_results.values() if test.get('is_significant', False))
            if significance_results:
                score_components.append(significant_count / len(significance_results))
            
            # Correlation score (penalty for high correlations)
            if 'correlation' in results and 'max_correlation' in results['correlation']:
                max_corr = results['correlation']['max_correlation']
                corr_score = max(0, 1 - (max_corr - self.config.max_correlation_threshold) / (1 - self.config.max_correlation_threshold))
                score_components.append(corr_score)
            
            if score_components:
                results['score'] = np.mean(score_components)
            
            return results
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in statistical validation: {e}")
            return {'tests': {}, 'correlation': {}, 'significance': {}, 'score': 0.0}
    
    def _run_performance_validation(
        self, 
        features: Dict[str, pd.Series], 
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run performance validation tests."""
        try:
            results = {
                'metrics': {},
                'backtest': {},
                'risk': {},
                'score': 0.0
            }
            
            # Get primary feature
            primary_feature = self._get_primary_feature(features)
            if primary_feature is None:
                return results
            
            # Calculate performance metrics
            returns = primary_feature.pct_change().dropna()
            if len(returns) == 0:
                return results
            
            # Basic performance metrics
            metrics = {
                'total_return': (1 + returns).prod() - 1,
                'annualized_return': returns.mean() * 252,
                'volatility': returns.std() * np.sqrt(252),
                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(returns),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis()
            }
            
            # Information ratio (vs price)
            price_returns = data['close'].pct_change().dropna()
            if len(price_returns) > 0:
                min_length = min(len(returns), len(price_returns))
                if min_length > 0:
                    excess_returns = returns.iloc[-min_length:] - price_returns.iloc[-min_length:]
                    metrics['information_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
            
            results['metrics'] = metrics
            
            # Risk metrics
            risk_metrics = {
                'var_95': np.percentile(returns, 5),
                'var_99': np.percentile(returns, 1),
                'cvar_95': returns[returns <= np.percentile(returns, 5)].mean(),
                'cvar_99': returns[returns <= np.percentile(returns, 1)].mean(),
                'downside_deviation': returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0,
                'sortino_ratio': returns.mean() / (returns[returns < 0].std() * np.sqrt(252)) if len(returns[returns < 0]) > 0 and returns[returns < 0].std() > 0 else 0
            }
            
            results['risk'] = risk_metrics
            
            # Simple backtesting simulation
            backtest_results = {
                'cumulative_return': (1 + returns).cumprod(),
                'drawdown': self._calculate_drawdown_series(returns),
                'win_rate': (returns > 0).mean(),
                'profit_factor': self._calculate_profit_factor(returns),
                'trades_count': len(returns),
                'avg_trade': returns.mean()
            }
            
            results['backtest'] = backtest_results
            
            # Calculate score
            score_components = []
            
            # Sharpe ratio score
            sharpe_score = min(1.0, max(0.0, metrics['sharpe_ratio'] / 2.0))  # Normalize to 0-1
            score_components.append(sharpe_score)
            
            # Drawdown score
            drawdown_score = max(0.0, 1.0 - metrics['max_drawdown'] / 0.5)  # Penalty for high drawdown
            score_components.append(drawdown_score)
            
            # Information ratio score
            if 'information_ratio' in metrics:
                ir_score = min(1.0, max(0.0, metrics['information_ratio'] / 1.0))  # Normalize to 0-1
                score_components.append(ir_score)
            
            # Win rate score
            win_rate_score = backtest_results['win_rate']
            score_components.append(win_rate_score)
            
            if score_components:
                results['score'] = np.mean(score_components)
            
            return results
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in performance validation: {e}")
            return {'metrics': {}, 'backtest': {}, 'risk': {}, 'score': 0.0}
    
    def _run_stability_validation(
        self, 
        features: Dict[str, pd.Series], 
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run stability validation tests."""
        try:
            results = {
                'stability': {},
                'consistency': {},
                'score': 0.0
            }
            
            # Get numeric features
            numeric_features = {
                name: series for name, series in features.items()
                if isinstance(series, pd.Series) and pd.api.types.is_numeric_dtype(series)
            }
            
            if not numeric_features:
                return results
            
            stability_metrics = {}
            consistency_metrics = {}
            
            for name, series in numeric_features.items():
                clean_series = series.dropna()
                if len(clean_series) < self.config.stability_window:
                    continue
                
                # Rolling statistics
                rolling_mean = clean_series.rolling(self.config.stability_window).mean()
                rolling_std = clean_series.rolling(self.config.stability_window).std()
                
                # Stability metrics
                stability_metrics[f'{name}_mean_stability'] = 1.0 / (1.0 + rolling_mean.std())
                stability_metrics[f'{name}_std_stability'] = 1.0 / (1.0 + rolling_std.std())
                stability_metrics[f'{name}_coefficient_variation'] = clean_series.std() / abs(clean_series.mean()) if clean_series.mean() != 0 else float('inf')
                
                # Consistency metrics
                consistency_metrics[f'{name}_trend_consistency'] = self._calculate_trend_consistency(clean_series)
                consistency_metrics[f'{name}_pattern_consistency'] = self._calculate_pattern_consistency(clean_series)
                consistency_metrics[f'{name}_outlier_ratio'] = self._calculate_outlier_ratio(clean_series)
            
            results['stability'] = stability_metrics
            results['consistency'] = consistency_metrics
            
            # Calculate score
            score_components = []
            
            # Stability score
            if stability_metrics:
                stability_scores = [v for v in stability_metrics.values() if not np.isinf(v)]
                if stability_scores:
                    score_components.append(np.mean(stability_scores))
            
            # Consistency score
            if consistency_metrics:
                consistency_scores = [v for v in consistency_metrics.values() if not np.isinf(v)]
                if consistency_scores:
                    score_components.append(np.mean(consistency_scores))
            
            if score_components:
                results['score'] = np.mean(score_components)
            
            return results
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in stability validation: {e}")
            return {'stability': {}, 'consistency': {}, 'score': 0.0}
    
    def _run_robustness_validation(
        self,
        feature_generator: VectorBTFeatureGenerator,
        data: pd.DataFrame,
        parameters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run robustness validation tests."""
        try:
            results = {
                'metrics': {},
                'score': 0.0
            }
            
            # Test with different parameter variations
            robustness_tests = []
            
            if parameters:
                # Test parameter variations
                for param_name, param_value in parameters.items():
                    if isinstance(param_value, (int, float)):
                        # Test ±10% variation
                        variation = param_value * 0.1
                        test_params = parameters.copy()
                        test_params[param_name] = param_value + variation
                        
                        try:
                            test_features = feature_generator.generate_vectorbt_features(data, test_params)
                            test_score = self._calculate_feature_quality_score(test_features)
                            robustness_tests.append(test_score)
                        except Exception:
                            continue
            
            # Test with different data subsets
            n_subsets = 5
            subset_size = len(data) // n_subsets
            
            for i in range(n_subsets):
                start_idx = i * subset_size
                end_idx = min((i + 1) * subset_size, len(data))
                subset_data = data.iloc[start_idx:end_idx]
                
                if len(subset_data) < 50:
                    continue
                
                try:
                    subset_features = feature_generator.generate_vectorbt_features(subset_data, parameters)
                    subset_score = self._calculate_feature_quality_score(subset_features)
                    robustness_tests.append(subset_score)
                except Exception:
                    continue
            
            # Calculate robustness metrics
            if robustness_tests:
                results['metrics'] = {
                    'mean_score': np.mean(robustness_tests),
                    'std_score': np.std(robustness_tests),
                    'min_score': np.min(robustness_tests),
                    'max_score': np.max(robustness_tests),
                    'coefficient_variation': np.std(robustness_tests) / np.mean(robustness_tests) if np.mean(robustness_tests) != 0 else float('inf')
                }
                
                # Robustness score (lower coefficient of variation = higher robustness)
                cv = results['metrics']['coefficient_variation']
                if not np.isinf(cv):
                    results['score'] = max(0.0, 1.0 - cv)
                else:
                    results['score'] = 0.0
            
            return results
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in robustness validation: {e}")
            return {'metrics': {}, 'score': 0.0}
    
    def _run_cross_validation(
        self,
        feature_generator: VectorBTFeatureGenerator,
        data: pd.DataFrame,
        parameters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run cross-validation tests."""
        try:
            results = {
                'scores': [],
                'mean': 0.0,
                'std': 0.0,
                'score': 0.0
            }
            
            if self.config.cv_strategy == "time_series":
                # Time series cross-validation
                cv_scores = self._time_series_cv(feature_generator, data, parameters)
            else:
                # K-fold cross-validation
                cv_scores = self._k_fold_cv(feature_generator, data, parameters)
            
            results['scores'] = cv_scores
            results['mean'] = np.mean(cv_scores) if cv_scores else 0.0
            results['std'] = np.std(cv_scores) if cv_scores else 0.0
            
            # Cross-validation score (higher mean, lower std = better)
            if cv_scores:
                mean_score = results['mean']
                std_score = results['std']
                results['score'] = mean_score * (1.0 - std_score)  # Penalty for high variance
            
            return results
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in cross-validation: {e}")
            return {'scores': [], 'mean': 0.0, 'std': 0.0, 'score': 0.0}
    
    def _run_out_of_sample_validation(
        self,
        feature_generator: VectorBTFeatureGenerator,
        data: pd.DataFrame,
        parameters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run out-of-sample validation tests."""
        try:
            results = {
                'performance': {},
                'metrics': {},
                'score': 0.0
            }
            
            # Split data
            train_size = int(len(data) * self.config.train_ratio)
            train_data = data.iloc[:train_size]
            test_data = data.iloc[train_size:]
            
            if len(train_data) < 50 or len(test_data) < 20:
                return results
            
            # Generate features on training data
            train_features = feature_generator.generate_vectorbt_features(train_data, parameters)
            
            # Generate features on test data
            test_features = feature_generator.generate_vectorbt_features(test_data, parameters)
            
            # Calculate performance metrics
            train_score = self._calculate_feature_quality_score(train_features)
            test_score = self._calculate_feature_quality_score(test_features)
            
            # Performance degradation
            performance_degradation = (train_score - test_score) / train_score if train_score > 0 else 1.0
            
            results['performance'] = {
                'train_score': train_score,
                'test_score': test_score,
                'performance_degradation': performance_degradation
            }
            
            # Out-of-sample metrics
            results['metrics'] = {
                'is_stable': performance_degradation < 0.3,  # Less than 30% degradation
                'degradation_level': performance_degradation,
                'test_quality': test_score
            }
            
            # Out-of-sample score
            results['score'] = test_score * (1.0 - performance_degradation)
            
            return results
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in out-of-sample validation: {e}")
            return {'performance': {}, 'metrics': {}, 'score': 0.0}
    
    def _run_walk_forward_validation(
        self,
        feature_generator: VectorBTFeatureGenerator,
        data: pd.DataFrame,
        parameters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run walk-forward validation tests."""
        try:
            results = {
                'results': [],
                'metrics': {},
                'score': 0.0
            }
            
            walk_forward_results = []
            initial_train_size = self.config.initial_train_size
            step_size = self.config.step_size
            min_test_size = self.config.min_test_size
            
            current_pos = initial_train_size
            
            while current_pos + min_test_size < len(data):
                # Training data
                train_data = data.iloc[:current_pos]
                
                # Test data
                test_end = min(current_pos + step_size, len(data))
                test_data = data.iloc[current_pos:test_end]
                
                if len(train_data) < 50 or len(test_data) < min_test_size:
                    current_pos += step_size
                    continue
                
                try:
                    # Generate features
                    train_features = feature_generator.generate_vectorbt_features(train_data, parameters)
                    test_features = feature_generator.generate_vectorbt_features(test_data, parameters)
                    
                    # Calculate scores
                    train_score = self._calculate_feature_quality_score(train_features)
                    test_score = self._calculate_feature_quality_score(test_features)
                    
                    walk_forward_results.append({
                        'train_score': train_score,
                        'test_score': test_score,
                        'performance_degradation': (train_score - test_score) / train_score if train_score > 0 else 1.0,
                        'train_size': len(train_data),
                        'test_size': len(test_data)
                    })
                    
                except Exception as e:
                    tprint_warning(f"⚠️ Error in walk-forward step: {e}")
                
                current_pos += step_size
            
            results['results'] = walk_forward_results
            
            if walk_forward_results:
                # Calculate walk-forward metrics
                test_scores = [r['test_score'] for r in walk_forward_results]
                degradations = [r['performance_degradation'] for r in walk_forward_results]
                
                results['metrics'] = {
                    'mean_test_score': np.mean(test_scores),
                    'std_test_score': np.std(test_scores),
                    'mean_degradation': np.mean(degradations),
                    'max_degradation': np.max(degradations),
                    'stability': 1.0 - np.std(test_scores) if test_scores else 0.0
                }
                
                # Walk-forward score
                mean_test_score = results['metrics']['mean_test_score']
                stability = results['metrics']['stability']
                results['score'] = mean_test_score * stability
            
            return results
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in walk-forward validation: {e}")
            return {'results': [], 'metrics': {}, 'score': 0.0}
    
    def _time_series_cv(
        self,
        feature_generator: VectorBTFeatureGenerator,
        data: pd.DataFrame,
        parameters: Optional[Dict[str, Any]]
    ) -> List[float]:
        """Perform time series cross-validation."""
        cv_scores = []
        n_samples = len(data)
        fold_size = n_samples // self.config.cv_folds
        
        for i in range(self.config.cv_folds):
            train_end = (i + 1) * fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, n_samples)
            
            if test_start >= n_samples:
                break
            
            train_data = data.iloc[:train_end]
            test_data = data.iloc[test_start:test_end]
            
            if len(train_data) < 50 or len(test_data) < 10:
                continue
            
            try:
                test_features = feature_generator.generate_vectorbt_features(test_data, parameters)
                score = self._calculate_feature_quality_score(test_features)
                cv_scores.append(score)
            except Exception as e:
                tprint_warning(f"⚠️ Error in CV fold {i}: {e}")
                continue
        
        return cv_scores
    
    def _k_fold_cv(
        self,
        feature_generator: VectorBTFeatureGenerator,
        data: pd.DataFrame,
        parameters: Optional[Dict[str, Any]]
    ) -> List[float]:
        """Perform k-fold cross-validation."""
        cv_scores = []
        n_samples = len(data)
        fold_size = n_samples // self.config.cv_folds
        
        for i in range(self.config.cv_folds):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_samples)
            
            if test_start >= n_samples:
                break
            
            test_indices = list(range(test_start, test_end))
            train_indices = list(range(0, test_start)) + list(range(test_end, n_samples))
            
            if len(train_indices) < 50 or len(test_indices) < 10:
                continue
            
            test_data = data.iloc[test_indices]
            
            try:
                test_features = feature_generator.generate_vectorbt_features(test_data, parameters)
                score = self._calculate_feature_quality_score(test_features)
                cv_scores.append(score)
            except Exception as e:
                tprint_warning(f"⚠️ Error in CV fold {i}: {e}")
                continue
        
        return cv_scores
    
    def _calculate_feature_quality_score(self, features: Dict[str, pd.Series]) -> float:
        """Calculate overall quality score for features."""
        try:
            if not features:
                return 0.0
            
            # Get numeric features
            numeric_features = {
                name: series for name, series in features.items()
                if isinstance(series, pd.Series) and pd.api.types.is_numeric_dtype(series)
            }
            
            if not numeric_features:
                return 0.0
            
            scores = []
            
            for name, series in numeric_features.items():
                clean_series = series.dropna()
                if len(clean_series) < 10:
                    continue
                
                # Stability score
                stability = 1.0 / (1.0 + clean_series.std())
                scores.append(stability)
                
                # Consistency score
                consistency = 1.0 - abs(clean_series.diff().mean()) / (clean_series.std() + 1e-8)
                scores.append(consistency)
                
                # Information content score
                information = clean_series.var() / (clean_series.mean()**2 + 1e-8)
                scores.append(min(1.0, information))
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating feature quality score: {e}")
            return 0.0
    
    def _get_primary_feature(self, features: Dict[str, pd.Series]) -> Optional[pd.Series]:
        """Get primary feature from feature dictionary."""
        if not features:
            return None
        
        # Priority order for primary feature selection
        priority_features = [
            'ratio', 'grade', 'score', 'signal', 'trend', 'momentum',
            'volatility', 'efficiency', 'coherence', 'strength'
        ]
        
        for priority in priority_features:
            for feature_name, feature_data in features.items():
                if priority in feature_name.lower() and isinstance(feature_data, pd.Series):
                    return feature_data
        
        # Fallback to first numeric series
        for feature_data in features.values():
            if isinstance(feature_data, pd.Series) and pd.api.types.is_numeric_dtype(feature_data):
                return feature_data
        
        # Last resort - return first series
        return next(iter(features.values()))
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return abs(drawdown.min())
        except Exception:
            return 0.0
    
    def _calculate_drawdown_series(self, returns: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown
        except Exception:
            return pd.Series(dtype=float)
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor."""
        try:
            positive_returns = returns[returns > 0].sum()
            negative_returns = abs(returns[returns < 0].sum())
            
            if negative_returns == 0:
                return float('inf')
            
            return positive_returns / negative_returns
        except Exception:
            return 0.0
    
    def _calculate_trend_consistency(self, series: pd.Series) -> float:
        """Calculate trend consistency."""
        try:
            if len(series) < 2:
                return 0.0
            
            # Calculate rolling trend direction
            rolling_trend = series.rolling(5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
            
            # Calculate consistency of trend direction
            trend_direction = np.sign(rolling_trend.diff())
            consistency = (trend_direction == trend_direction.shift(1)).mean()
            
            return consistency
        except Exception:
            return 0.0
    
    def _calculate_pattern_consistency(self, series: pd.Series) -> float:
        """Calculate pattern consistency."""
        try:
            if len(series) < 10:
                return 0.0
            
            # Calculate rolling patterns
            rolling_mean = series.rolling(5).mean()
            pattern = (series > rolling_mean).astype(int)
            
            # Calculate pattern consistency
            pattern_consistency = (pattern == pattern.shift(1)).mean()
            
            return pattern_consistency
        except Exception:
            return 0.0
    
    def _calculate_outlier_ratio(self, series: pd.Series) -> float:
        """Calculate outlier ratio."""
        try:
            if len(series) < 10:
                return 0.0
            
            # Calculate IQR
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            
            # Count outliers
            outliers = ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
            outlier_ratio = outliers / len(series)
            
            return 1.0 - outlier_ratio  # Higher is better
        except Exception:
            return 0.0
    
    def _generate_recommendations(self, validation_result: ValidationResult) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Overall score recommendations
        if validation_result.overall_score < 0.5:
            recommendations.append("Consider improving feature quality - overall score is low")
        
        # Statistical validation recommendations
        if validation_result.statistical_tests:
            normal_count = sum(1 for test in validation_result.statistical_tests.values() if test.get('is_normal', False))
            total_tests = len(validation_result.statistical_tests)
            if normal_count / total_tests < 0.5:
                recommendations.append("Consider data transformation to improve normality")
        
        # Performance validation recommendations
        if validation_result.performance_metrics:
            sharpe_ratio = validation_result.performance_metrics.get('sharpe_ratio', 0)
            if sharpe_ratio < 0.5:
                recommendations.append("Consider improving risk-adjusted returns")
            
            max_drawdown = validation_result.performance_metrics.get('max_drawdown', 0)
            if max_drawdown > 0.2:
                recommendations.append("Consider reducing maximum drawdown")
        
        # Stability validation recommendations
        if validation_result.stability_metrics:
            stability_scores = [v for v in validation_result.stability_metrics.values() if not np.isinf(v)]
            if stability_scores and np.mean(stability_scores) < 0.5:
                recommendations.append("Consider improving feature stability")
        
        # Cross-validation recommendations
        if validation_result.cv_std > 0.2:
            recommendations.append("Consider improving cross-validation consistency")
        
        # Out-of-sample recommendations
        if validation_result.oos_performance:
            degradation = validation_result.oos_performance.get('performance_degradation', 0)
            if degradation > 0.3:
                recommendations.append("Consider improving out-of-sample performance")
        
        return recommendations
    
    def get_validation_history(self) -> List[ValidationResult]:
        """Get validation history."""
        return self.validation_history.copy()
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary statistics."""
        if not self.validation_history:
            return {}
        
        passed_count = sum(1 for result in self.validation_history if result.validation_passed)
        total_count = len(self.validation_history)
        
        scores = [result.overall_score for result in self.validation_history]
        
        return {
            'total_validations': total_count,
            'passed_validations': passed_count,
            'pass_rate': passed_count / total_count if total_count > 0 else 0.0,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores)
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.validation_history.clear()
        self.current_validation = None
        tprint_info("🧹 VectorBT Feature Validator cleanup completed")


# Convenience functions
def create_vectorbt_validator(config: Optional[VectorBTValidationConfig] = None) -> VectorBTFeatureValidator:
    """Create VectorBT feature validator instance."""
    return VectorBTFeatureValidator(config)


def validate_vectorbt_feature(
    feature_generator: VectorBTFeatureGenerator,
    data: pd.DataFrame,
    config: Optional[VectorBTValidationConfig] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """Validate a single VectorBT feature."""
    validator = create_vectorbt_validator(config)
    return validator.validate_feature(feature_generator, data, parameters)