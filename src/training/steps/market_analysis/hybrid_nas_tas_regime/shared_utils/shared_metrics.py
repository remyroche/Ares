"""
Shared Metrics Utilities for Hybrid NAS-TAS Regime Detection.

Provides common metrics utilities that can be used by both NAS and TAS systems
for performance evaluation, model comparison, and result analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass
import time
from datetime import datetime
from enum import Enum
import json
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import existing utilities
try:
    from src.utils.common_operations import (
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer
    )
    HARDWARE_UTILS_AVAILABLE = True
except ImportError:
    HARDWARE_UTILS_AVAILABLE = False

try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_enhanced_matrix_operations,
        get_batch_matrix_processor
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics available."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    RANKING = "ranking"
    CLUSTERING = "clustering"
    FINANCIAL = "financial"
    TIME_SERIES = "time_series"

class MetricCategory(Enum):
    """Categories of metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    PR_AUC = "pr_auc"
    LOG_LOSS = "log_loss"
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    R2_SCORE = "r2_score"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    INFORMATION_RATIO = "information_ratio"
    TRACKING_ERROR = "tracking_error"
    BETA = "beta"
    ALPHA = "alpha"
    JENSEN_ALPHA = "jensen_alpha"
    TREYNOR_RATIO = "treynor_ratio"
    VOLATILITY = "volatility"
    SKEWNESS = "skewness"
    KURTOSIS = "kurtosis"
    VAR = "var"
    CVAR = "cvar"
    EXPECTED_SHORTFALL = "expected_shortfall"

@dataclass
class SharedMetricsConfig:
    """Configuration for shared metrics utilities."""
    # Metric types to calculate
    metric_types: List[MetricType] = None
    metric_categories: List[MetricCategory] = None

    # Financial metrics specific
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    benchmark_return: float = 0.08  # 8% annual benchmark return

    # Time series specific
    lookback_period: int = 252  # 1 year of trading days

    # Performance optimization
    use_hardware_acceleration: bool = True
    use_matrix_operations: bool = True
    batch_size: int = 1000
    memory_limit_gb: float = 8.0

    # Output settings
    save_results: bool = True
    output_dir: str = "metrics_results"
    verbose: bool = True

    def __post_init__(self):
        if self.metric_types is None:
            self.metric_types = [MetricType.CLASSIFICATION, MetricType.REGRESSION, MetricType.FINANCIAL]
        if self.metric_categories is None:
            self.metric_categories = [
                MetricCategory.ACCURACY,
                MetricCategory.PRECISION,
                MetricCategory.RECALL,
                MetricCategory.F1_SCORE,
                MetricCategory.ROC_AUC
            ]

@dataclass
class SharedMetricsResult:
    """Result from shared metrics calculation."""
    # Core metrics
    metrics: Dict[str, float]
    metric_categories: Dict[str, List[str]]

    # Performance metrics
    calculation_time: float = 0.0
    memory_usage_mb: float = 0.0

    # Metadata
    metric_types: List[str] = None
    n_samples: int = 0
    n_features: int = 0

    # Results
    success: bool = True
    error_message: Optional[str] = None
    hardware_optimization_applied: bool = False
    matrix_operations_used: bool = False

class SharedMetricsCalculator:
    """Shared metrics calculator for both NAS and TAS systems."""

    def __init__(self, config: SharedMetricsConfig):
        """Initialize the shared metrics calculator.

        Args:
            config: Shared metrics configuration
        """
        tprint_info("Initializing Shared Metrics Calculator")
        tprint_debug(f"Configuration: {config}")
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize hardware acceleration if available
        self.hardware_accelerator = None
        self.memory_optimizer = None
        self.cpu_optimizer = None

        if HARDWARE_UTILS_AVAILABLE and config.use_hardware_acceleration:
            try:
                tprint_info("Initializing hardware acceleration for metrics calculation")
                self.hardware_accelerator = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                tprint_success("Hardware acceleration initialized for shared metrics")
                self.logger.info("✅ Hardware acceleration initialized for shared metrics")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware acceleration not available: {e}")

        # Initialize matrix operations if available
        self.matrix_ops = None
        self.vectorized_core = None
        self.enhanced_ops = None
        self.batch_processor = None

        if MATRIX_OPERATIONS_AVAILABLE and config.use_matrix_operations:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.enhanced_ops = get_enhanced_matrix_operations()
                self.batch_processor = get_batch_matrix_processor()
                self.logger.info("✅ Matrix operations initialized for shared metrics")
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations not available: {e}")

        self.logger.info("✅ Shared Metrics Calculator initialized")
        self.logger.info(f"   Metric types: {[t.value for t in config.metric_types]}")
        self.logger.info(f"   Metric categories: {[c.value for c in config.metric_categories]}")
        self.logger.info(f"   Risk-free rate: {config.risk_free_rate}")
        self.logger.info(f"   Benchmark return: {config.benchmark_return}")

    def calculate_metrics(self,
                         y_true: pd.Series,
                         y_pred: np.ndarray,
                         y_pred_proba: Optional[np.ndarray] = None,
                         returns: Optional[pd.Series] = None,
                         benchmark_returns: Optional[pd.Series] = None,
                         additional_data: Optional[Dict[str, Any]] = None) -> SharedMetricsResult:
        """Calculate metrics using the configured strategy.

        Args:
            y_true: True labels/values
            y_pred: Predicted labels/values
            y_pred_proba: Predicted probabilities (optional)
            returns: Optional returns for financial metrics
            benchmark_returns: Optional benchmark returns for financial metrics
            additional_data: Optional additional data for metrics calculation

        Returns:
            SharedMetricsResult with calculated metrics
        """
        start_time = time.time()

        try:
            self.logger.info("📊 Starting shared metrics calculation")
            self.logger.info(f"   Data shape: {len(y_true)}")
            self.logger.info(f"   Metric types: {[t.value for t in self.config.metric_types]}")
            self.logger.info(f"   Metric categories: {[c.value for c in self.config.metric_categories]}")

            # Initialize metrics
            metrics = {}
            metric_categories = {}

            # Calculate metrics based on type
            for metric_type in self.config.metric_types:
                if metric_type == MetricType.CLASSIFICATION:
                    type_metrics, type_categories = self._calculate_classification_metrics(y_true, y_pred, y_pred_proba)
                elif metric_type == MetricType.REGRESSION:
                    type_metrics, type_categories = self._calculate_regression_metrics(y_true, y_pred)
                elif metric_type == MetricType.RANKING:
                    type_metrics, type_categories = self._calculate_ranking_metrics(y_true, y_pred, y_pred_proba)
                elif metric_type == MetricType.CLUSTERING:
                    type_metrics, type_categories = self._calculate_clustering_metrics(y_true, y_pred)
                elif metric_type == MetricType.FINANCIAL:
                    type_metrics, type_categories = self._calculate_financial_metrics(returns, benchmark_returns)
                elif metric_type == MetricType.TIME_SERIES:
                    type_metrics, type_categories = self._calculate_time_series_metrics(y_true, y_pred, additional_data)
                else:
                    type_metrics, type_categories = {}, {}

                # Merge metrics
                metrics.update(type_metrics)
                metric_categories.update(type_categories)

            # Calculate execution time
            calculation_time = time.time() - start_time

            self.logger.info(f"✅ Shared metrics calculation completed in {calculation_time:.2f}s")
            self.logger.info(f"   Calculated metrics: {len(metrics)}")

            return SharedMetricsResult(
                metrics=metrics,
                metric_categories=metric_categories,
                calculation_time=calculation_time,
                metric_types=[t.value for t in self.config.metric_types],
                n_samples=len(y_true),
                n_features=0,  # Would be calculated from additional data
                success=True,
                hardware_optimization_applied=self.hardware_accelerator is not None,
                matrix_operations_used=self.matrix_ops is not None
            )

        except Exception as e:
            calculation_time = time.time() - start_time
            self.logger.error(f"❌ Shared metrics calculation failed: {e}")

            return SharedMetricsResult(
                metrics={},
                metric_categories={},
                calculation_time=calculation_time,
                metric_types=[t.value for t in self.config.metric_types],
                n_samples=len(y_true),
                n_features=0,
                success=False,
                error_message=str(e)
            )

    def _calculate_classification_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                                        y_pred_proba: Optional[np.ndarray] = None) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """Calculate classification metrics."""
        try:
            metrics = {}
            categories = {}

            for category in self.config.metric_categories:
                if category == MetricCategory.ACCURACY:
                    from sklearn.metrics import accuracy_score
                    metrics['accuracy'] = accuracy_score(y_true, y_pred)
                    categories['accuracy'] = ['accuracy']

                elif category == MetricCategory.PRECISION:
                    from sklearn.metrics import precision_score
                    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                    categories['precision'] = ['precision']

                elif category == MetricCategory.RECALL:
                    from sklearn.metrics import recall_score
                    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                    categories['recall'] = ['recall']

                elif category == MetricCategory.F1_SCORE:
                    from sklearn.metrics import f1_score
                    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                    categories['f1_score'] = ['f1_score']

                elif category == MetricCategory.ROC_AUC:
                    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                        from sklearn.metrics import roc_auc_score
                        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                        categories['roc_auc'] = ['roc_auc']
                    else:
                        metrics['roc_auc'] = 0.0
                        categories['roc_auc'] = ['roc_auc']

                elif category == MetricCategory.PR_AUC:
                    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                        from sklearn.metrics import average_precision_score
                        metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba[:, 1])
                        categories['pr_auc'] = ['pr_auc']
                    else:
                        metrics['pr_auc'] = 0.0
                        categories['pr_auc'] = ['pr_auc']

                elif category == MetricCategory.LOG_LOSS:
                    if y_pred_proba is not None:
                        from sklearn.metrics import log_loss
                        metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                        categories['log_loss'] = ['log_loss']
                    else:
                        metrics['log_loss'] = 0.0
                        categories['log_loss'] = ['log_loss']

            return metrics, categories

        except Exception as e:
            self.logger.warning(f"⚠️ Classification metrics calculation failed: {e}")
            return {}, {}

    def _calculate_regression_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """Calculate regression metrics."""
        try:
            metrics = {}
            categories = {}

            for category in self.config.metric_categories:
                if category == MetricCategory.MAE:
                    from sklearn.metrics import mean_absolute_error
                    metrics['mae'] = mean_absolute_error(y_true, y_pred)
                    categories['mae'] = ['mae']

                elif category == MetricCategory.MSE:
                    from sklearn.metrics import mean_squared_error
                    metrics['mse'] = mean_squared_error(y_true, y_pred)
                    categories['mse'] = ['mse']

                elif category == MetricCategory.RMSE:
                    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
                    categories['rmse'] = ['rmse']

                elif category == MetricCategory.R2_SCORE:
                    from sklearn.metrics import r2_score
                    metrics['r2_score'] = r2_score(y_true, y_pred)
                    categories['r2_score'] = ['r2_score']

            return metrics, categories

        except Exception as e:
            self.logger.warning(f"⚠️ Regression metrics calculation failed: {e}")
            return {}, {}

    def _calculate_ranking_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                                 y_pred_proba: Optional[np.ndarray] = None) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """Calculate ranking metrics."""
        try:
            metrics = {}
            categories = {}

            # This would implement ranking metrics like NDCG, MAP, etc.
            # For now, return empty results

            return metrics, categories

        except Exception as e:
            self.logger.warning(f"⚠️ Ranking metrics calculation failed: {e}")
            return {}, {}

    def _calculate_clustering_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """Calculate clustering metrics."""
        try:
            metrics = {}
            categories = {}

            # This would implement clustering metrics like silhouette score, etc.
            # For now, return empty results

            return metrics, categories

        except Exception as e:
            self.logger.warning(f"⚠️ Clustering metrics calculation failed: {e}")
            return {}, {}

    def _calculate_financial_metrics(self, returns: Optional[pd.Series],
                                   benchmark_returns: Optional[pd.Series]) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """Calculate financial metrics."""
        try:
            metrics = {}
            categories = {}

            if returns is None:
                return metrics, categories

            # Calculate basic financial metrics
            for category in self.config.metric_categories:
                if category == MetricCategory.SHARPE_RATIO:
                    sharpe_ratio = self._calculate_sharpe_ratio(returns)
                    metrics['sharpe_ratio'] = sharpe_ratio
                    categories['sharpe_ratio'] = ['sharpe_ratio']

                elif category == MetricCategory.MAX_DRAWDOWN:
                    max_drawdown = self._calculate_max_drawdown(returns)
                    metrics['max_drawdown'] = max_drawdown
                    categories['max_drawdown'] = ['max_drawdown']

                elif category == MetricCategory.CALMAR_RATIO:
                    calmar_ratio = self._calculate_calmar_ratio(returns)
                    metrics['calmar_ratio'] = calmar_ratio
                    categories['calmar_ratio'] = ['calmar_ratio']

                elif category == MetricCategory.SORTINO_RATIO:
                    sortino_ratio = self._calculate_sortino_ratio(returns)
                    metrics['sortino_ratio'] = sortino_ratio
                    categories['sortino_ratio'] = ['sortino_ratio']

                elif category == MetricCategory.INFORMATION_RATIO:
                    if benchmark_returns is not None:
                        information_ratio = self._calculate_information_ratio(returns, benchmark_returns)
                        metrics['information_ratio'] = information_ratio
                        categories['information_ratio'] = ['information_ratio']

                elif category == MetricCategory.TRACKING_ERROR:
                    if benchmark_returns is not None:
                        tracking_error = self._calculate_tracking_error(returns, benchmark_returns)
                        metrics['tracking_error'] = tracking_error
                        categories['tracking_error'] = ['tracking_error']

                elif category == MetricCategory.BETA:
                    if benchmark_returns is not None:
                        beta = self._calculate_beta(returns, benchmark_returns)
                        metrics['beta'] = beta
                        categories['beta'] = ['beta']

                elif category == MetricCategory.ALPHA:
                    if benchmark_returns is not None:
                        alpha = self._calculate_alpha(returns, benchmark_returns)
                        metrics['alpha'] = alpha
                        categories['alpha'] = ['alpha']

                elif category == MetricCategory.JENSEN_ALPHA:
                    if benchmark_returns is not None:
                        jensen_alpha = self._calculate_jensen_alpha(returns, benchmark_returns)
                        metrics['jensen_alpha'] = jensen_alpha
                        categories['jensen_alpha'] = ['jensen_alpha']

                elif category == MetricCategory.TREYNOR_RATIO:
                    if benchmark_returns is not None:
                        treynor_ratio = self._calculate_treynor_ratio(returns, benchmark_returns)
                        metrics['treynor_ratio'] = treynor_ratio
                        categories['treynor_ratio'] = ['treynor_ratio']

                elif category == MetricCategory.VOLATILITY:
                    volatility = self._calculate_volatility(returns)
                    metrics['volatility'] = volatility
                    categories['volatility'] = ['volatility']

                elif category == MetricCategory.SKEWNESS:
                    skewness = self._calculate_skewness(returns)
                    metrics['skewness'] = skewness
                    categories['skewness'] = ['skewness']

                elif category == MetricCategory.KURTOSIS:
                    kurtosis = self._calculate_kurtosis(returns)
                    metrics['kurtosis'] = kurtosis
                    categories['kurtosis'] = ['kurtosis']

                elif category == MetricCategory.VAR:
                    var = self._calculate_var(returns)
                    metrics['var'] = var
                    categories['var'] = ['var']

                elif category == MetricCategory.CVAR:
                    cvar = self._calculate_cvar(returns)
                    metrics['cvar'] = cvar
                    categories['cvar'] = ['cvar']

                elif category == MetricCategory.EXPECTED_SHORTFALL:
                    expected_shortfall = self._calculate_expected_shortfall(returns)
                    metrics['expected_shortfall'] = expected_shortfall
                    categories['expected_shortfall'] = ['expected_shortfall']

            return metrics, categories

        except Exception as e:
            self.logger.warning(f"⚠️ Financial metrics calculation failed: {e}")
            return {}, {}

    def _calculate_time_series_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                                     additional_data: Optional[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """Calculate time series metrics."""
        try:
            metrics = {}
            categories = {}

            # This would implement time series specific metrics
            # For now, return empty results

            return metrics, categories

        except Exception as e:
            self.logger.warning(f"⚠️ Time series metrics calculation failed: {e}")
            return {}, {}

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        try:
            excess_returns = returns - self.config.risk_free_rate / 252  # Daily risk-free rate
            if returns.std() == 0:
                return 0.0
            return excess_returns.mean() / returns.std() * np.sqrt(252)  # Annualized

        except Exception:
            return 0.0

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            return drawdown.min()

        except Exception:
            return 0.0

    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio."""
        try:
            annual_return = returns.mean() * 252
            max_dd = abs(self._calculate_max_drawdown(returns))
            if max_dd == 0:
                return 0.0
            return annual_return / max_dd

        except Exception:
            return 0.0

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        try:
            excess_returns = returns - self.config.risk_free_rate / 252
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) == 0 or downside_returns.std() == 0:
                return 0.0
            return excess_returns.mean() / downside_returns.std() * np.sqrt(252)

        except Exception:
            return 0.0

    def _calculate_information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate information ratio."""
        try:
            excess_returns = returns - benchmark_returns
            if excess_returns.std() == 0:
                return 0.0
            return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

        except Exception:
            return 0.0

    def _calculate_tracking_error(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate tracking error."""
        try:
            excess_returns = returns - benchmark_returns
            return excess_returns.std() * np.sqrt(252)

        except Exception:
            return 0.0

    def _calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta."""
        try:
            if benchmark_returns.var() == 0:
                return 0.0
            return returns.cov(benchmark_returns) / benchmark_returns.var()

        except Exception:
            return 0.0

    def _calculate_alpha(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate alpha."""
        try:
            beta = self._calculate_beta(returns, benchmark_returns)
            return (returns.mean() - self.config.risk_free_rate / 252) - beta * (benchmark_returns.mean() - self.config.risk_free_rate / 252)

        except Exception:
            return 0.0

    def _calculate_jensen_alpha(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Jensen's alpha."""
        try:
            # Same as alpha for CAPM model
            return self._calculate_alpha(returns, benchmark_returns)

        except Exception:
            return 0.0

    def _calculate_treynor_ratio(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Treynor ratio."""
        try:
            beta = self._calculate_beta(returns, benchmark_returns)
            if beta == 0:
                return 0.0
            return (returns.mean() - self.config.risk_free_rate / 252) / beta * 252

        except Exception:
            return 0.0

    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate volatility."""
        try:
            return returns.std() * np.sqrt(252)

        except Exception:
            return 0.0

    def _calculate_skewness(self, returns: pd.Series) -> float:
        """Calculate skewness."""
        try:
            return returns.skew()

        except Exception:
            return 0.0

    def _calculate_kurtosis(self, returns: pd.Series) -> float:
        """Calculate kurtosis."""
        try:
            return returns.kurtosis()

        except Exception:
            return 0.0

    def _calculate_var(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk (VaR)."""
        try:
            return returns.quantile(confidence_level)

        except Exception:
            return 0.0

    def _calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        try:
            var = self._calculate_var(returns, confidence_level)
            return returns[returns <= var].mean()

        except Exception:
            return 0.0

    def _calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate expected shortfall."""
        try:
            # Same as CVaR
            return self._calculate_cvar(returns, confidence_level)

        except Exception:
            return 0.0

def create_shared_metrics_calculator(config: Optional[SharedMetricsConfig] = None) -> SharedMetricsCalculator:
    """Create a shared metrics calculator instance.

    Args:
        config: Optional shared metrics configuration

    Returns:
        SharedMetricsCalculator instance
    """
    if config is None:
        config = SharedMetricsConfig()
    return SharedMetricsCalculator(config)

def quick_shared_metrics(y_true: pd.Series,
                        y_pred: np.ndarray,
                        y_pred_proba: Optional[np.ndarray] = None,
                        returns: Optional[pd.Series] = None,
                        benchmark_returns: Optional[pd.Series] = None) -> SharedMetricsResult:
    """Quick shared metrics calculation with default settings.

    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        y_pred_proba: Predicted probabilities (optional)
        returns: Optional returns for financial metrics
        benchmark_returns: Optional benchmark returns for financial metrics

    Returns:
        SharedMetricsResult
    """
    config = SharedMetricsConfig()
    calculator = SharedMetricsCalculator(config)
    return calculator.calculate_metrics(y_true, y_pred, y_pred_proba, returns, benchmark_returns)
