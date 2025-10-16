"""
Unified Validation System

This module provides a unified validation and metrics system that combines
validation logic from both TAS and NAS regime detection systems.

Features:
- Unified validation logic
- Common metrics calculation
- Shared performance tracking
- Common reporting formats
- Cross-validation support
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

logger = logging.getLogger(__name__)

class ValidationType(Enum):
    """Types of validation."""
    CROSS_VALIDATION = "cross_validation"
    TIME_SERIES_VALIDATION = "time_series_validation"
    HOLDOUT_VALIDATION = "holdout_validation"
    BOOTSTRAP_VALIDATION = "bootstrap_validation"

class MetricType(Enum):
    """Types of metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    REGIME_STABILITY = "regime_stability"
    TRANSITION_ACCURACY = "transition_accuracy"

@dataclass
class ValidationConfig:
    """Configuration for unified validation."""

    # Validation parameters
    validation_type: ValidationType = ValidationType.TIME_SERIES_VALIDATION
    n_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42

    # Metrics to calculate
    metrics: List[MetricType] = field(default_factory=lambda: [
        MetricType.ACCURACY,
        MetricType.PRECISION,
        MetricType.RECALL,
        MetricType.F1_SCORE
    ])

    # Trading metrics
    enable_trading_metrics: bool = True
    risk_free_rate: float = 0.02

    # Regime-specific metrics
    enable_regime_metrics: bool = True
    stability_threshold: float = 0.7

    # Bootstrap validation
    enable_bootstrap: bool = True
    bootstrap_iterations: int = 100
    confidence_level: float = 0.95

@dataclass
class ValidationResult:
    """Result from unified validation."""

    # Validation success
    success: bool
    validation_type: str

    # Metrics
    metrics: Dict[str, float]
    metrics_std: Dict[str, float]
    metrics_ci: Dict[str, Tuple[float, float]]

    # Cross-validation results
    cv_scores: Optional[Dict[str, List[float]]] = None
    cv_mean: Optional[Dict[str, float]] = None
    cv_std: Optional[Dict[str, float]] = None

    # Trading performance
    trading_metrics: Optional[Dict[str, float]] = None

    # Regime-specific metrics
    regime_metrics: Optional[Dict[str, float]] = None

    # Bootstrap results
    bootstrap_results: Optional[Dict[str, Any]] = None

    # Metadata
    validation_timestamp: datetime = field(default_factory=datetime.now)
    n_samples: int = 0
    n_folds: int = 0
    execution_time: float = 0.0
    error_message: Optional[str] = None

class UnifiedValidationSystem:
    """
    Unified Validation System.

    Combines validation logic from both TAS and NAS regime detection systems.
    """

    def __init__(self, config: ValidationConfig):
        """Initialize unified validation system."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info("✅ Unified Validation System initialized")
        self.logger.info(f"   Validation type: {config.validation_type.value}")
        self.logger.info(f"   Metrics: {[m.value for m in config.metrics]}")
        self.logger.info(f"   Trading metrics: {config.enable_trading_metrics}")
        self.logger.info(f"   Regime metrics: {config.enable_regime_metrics}")

    def validate(self,
                 model: Any,
                 X: np.ndarray,
                 y: np.ndarray,
                 market_data: Optional[np.ndarray] = None,
                 regime_predictions: Optional[np.ndarray] = None) -> ValidationResult:
        """
        Perform unified validation.

        Args:
            model: Model to validate
            X: Features
            y: Target values
            market_data: Optional market data for trading metrics
            regime_predictions: Optional regime predictions

        Returns:
            Comprehensive validation result
        """
        start_time = time.time()

        try:
            self.logger.info("🔍 Starting unified validation...")
            self.logger.info(f"   Data shape: {X.shape}")
            self.logger.info(f"   Validation type: {self.config.validation_type.value}")

            # Perform validation based on type
            if self.config.validation_type == ValidationType.CROSS_VALIDATION:
                result = self._cross_validation(model, X, y, market_data, regime_predictions)
            elif self.config.validation_type == ValidationType.TIME_SERIES_VALIDATION:
                result = self._time_series_validation(model, X, y, market_data, regime_predictions)
            elif self.config.validation_type == ValidationType.HOLDOUT_VALIDATION:
                result = self._holdout_validation(model, X, y, market_data, regime_predictions)
            elif self.config.validation_type == ValidationType.BOOTSTRAP_VALIDATION:
                result = self._bootstrap_validation(model, X, y, market_data, regime_predictions)
            else:
                raise ValueError(f"Unknown validation type: {self.config.validation_type}")

            execution_time = time.time() - start_time
            result.execution_time = execution_time

            self.logger.info(f"✅ Unified validation completed in {execution_time:.2f}s")
            self.logger.info(f"   Success: {result.success}")
            if result.metrics:
                self.logger.info(f"   Accuracy: {result.metrics.get('accuracy', 0.0):.3f}")
                self.logger.info(f"   F1 Score: {result.metrics.get('f1_score', 0.0):.3f}")

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Unified validation failed: {e}")

            return ValidationResult(
                success=False,
                validation_type=self.config.validation_type.value,
                metrics={},
                metrics_std={},
                metrics_ci={},
                n_samples=len(X),
                n_folds=0,
                execution_time=execution_time,
                error_message=str(e)
            )

    def _cross_validation(self, model: Any, X: np.ndarray, y: np.ndarray,
                         market_data: Optional[np.ndarray],
                         regime_predictions: Optional[np.ndarray]) -> ValidationResult:
        """Perform cross-validation."""
        try:
            cv_scores = {}
            cv_results = {}

            # Calculate cross-validation scores for each metric
            for metric in self.config.metrics:
                if metric == MetricType.ACCURACY:
                    scores = cross_val_score(model, X, y, cv=self.config.n_folds, scoring='accuracy')
                elif metric == MetricType.PRECISION:
                    scores = cross_val_score(model, X, y, cv=self.config.n_folds, scoring='precision_weighted')
                elif metric == MetricType.RECALL:
                    scores = cross_val_score(model, X, y, cv=self.config.n_folds, scoring='recall_weighted')
                elif metric == MetricType.F1_SCORE:
                    scores = cross_val_score(model, X, y, cv=self.config.n_folds, scoring='f1_weighted')
                else:
                    # Custom metric calculation
                    scores = self._calculate_custom_metric_cv(model, X, y, metric)

                cv_scores[metric.value] = scores
                cv_results[metric.value] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'scores': scores
                }

            # Calculate final metrics
            metrics = {metric.value: cv_results[metric.value]['mean'] for metric in self.config.metrics}
            metrics_std = {metric.value: cv_results[metric.value]['std'] for metric in self.config.metrics}
            metrics_ci = {metric.value: self._calculate_confidence_interval(cv_results[metric.value]['scores'])
                         for metric in self.config.metrics}

            # Calculate trading metrics if enabled
            trading_metrics = None
            if self.config.enable_trading_metrics and market_data is not None:
                trading_metrics = self._calculate_trading_metrics(model, X, y, market_data)

            # Calculate regime metrics if enabled
            regime_metrics = None
            if self.config.enable_regime_metrics and regime_predictions is not None:
                regime_metrics = self._calculate_regime_metrics(model, X, y, regime_predictions)

            return ValidationResult(
                success=True,
                validation_type=self.config.validation_type.value,
                metrics=metrics,
                metrics_std=metrics_std,
                metrics_ci=metrics_ci,
                cv_scores=cv_scores,
                cv_mean=metrics,
                cv_std=metrics_std,
                trading_metrics=trading_metrics,
                regime_metrics=regime_metrics,
                n_samples=len(X),
                n_folds=self.config.n_folds
            )

        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}")
            raise

    def _time_series_validation(self, model: Any, X: np.ndarray, y: np.ndarray,
                               market_data: Optional[np.ndarray],
                               regime_predictions: Optional[np.ndarray]) -> ValidationResult:
        """Perform time series validation."""
        try:
            # Use TimeSeriesSplit for time series data
            tscv = TimeSeriesSplit(n_splits=self.config.n_folds)

            cv_scores = {}
            cv_results = {}

            for metric in self.config.metrics:
                scores = []

                for train_idx, test_idx in tscv.split(X):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    # Train model
                    model.fit(X_train, y_train)

                    # Make predictions
                    y_pred = model.predict(X_test)

                    # Calculate metric
                    if metric == MetricType.ACCURACY:
                        score = accuracy_score(y_test, y_pred)
                    elif metric == MetricType.PRECISION:
                        score = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    elif metric == MetricType.RECALL:
                        score = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    elif metric == MetricType.F1_SCORE:
                        score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    else:
                        score = self._calculate_custom_metric(y_test, y_pred, metric)

                    scores.append(score)

                cv_scores[metric.value] = scores
                cv_results[metric.value] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'scores': scores
                }

            # Calculate final metrics
            metrics = {metric.value: cv_results[metric.value]['mean'] for metric in self.config.metrics}
            metrics_std = {metric.value: cv_results[metric.value]['std'] for metric in self.config.metrics}
            metrics_ci = {metric.value: self._calculate_confidence_interval(cv_results[metric.value]['scores'])
                         for metric in self.config.metrics}

            # Calculate trading metrics if enabled
            trading_metrics = None
            if self.config.enable_trading_metrics and market_data is not None:
                trading_metrics = self._calculate_trading_metrics(model, X, y, market_data)

            # Calculate regime metrics if enabled
            regime_metrics = None
            if self.config.enable_regime_metrics and regime_predictions is not None:
                regime_metrics = self._calculate_regime_metrics(model, X, y, regime_predictions)

            return ValidationResult(
                success=True,
                validation_type=self.config.validation_type.value,
                metrics=metrics,
                metrics_std=metrics_std,
                metrics_ci=metrics_ci,
                cv_scores=cv_scores,
                cv_mean=metrics,
                cv_std=metrics_std,
                trading_metrics=trading_metrics,
                regime_metrics=regime_metrics,
                n_samples=len(X),
                n_folds=self.config.n_folds
            )

        except Exception as e:
            self.logger.error(f"Time series validation failed: {e}")
            raise

    def _holdout_validation(self, model: Any, X: np.ndarray, y: np.ndarray,
                           market_data: Optional[np.ndarray],
                           regime_predictions: Optional[np.ndarray]) -> ValidationResult:
        """Perform holdout validation."""
        try:
            # Split data
            split_idx = int(len(X) * (1 - self.config.test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = {}
            metrics_std = {}
            metrics_ci = {}

            for metric in self.config.metrics:
                if metric == MetricType.ACCURACY:
                    score = accuracy_score(y_test, y_pred)
                elif metric == MetricType.PRECISION:
                    score = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                elif metric == MetricType.RECALL:
                    score = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                elif metric == MetricType.F1_SCORE:
                    score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                else:
                    score = self._calculate_custom_metric(y_test, y_pred, metric)

                metrics[metric.value] = score
                metrics_std[metric.value] = 0.0  # No std for single holdout
                metrics_ci[metric.value] = (score, score)  # No CI for single holdout

            # Calculate trading metrics if enabled
            trading_metrics = None
            if self.config.enable_trading_metrics and market_data is not None:
                trading_metrics = self._calculate_trading_metrics(model, X, y, market_data)

            # Calculate regime metrics if enabled
            regime_metrics = None
            if self.config.enable_regime_metrics and regime_predictions is not None:
                regime_metrics = self._calculate_regime_metrics(model, X, y, regime_predictions)

            return ValidationResult(
                success=True,
                validation_type=self.config.validation_type.value,
                metrics=metrics,
                metrics_std=metrics_std,
                metrics_ci=metrics_ci,
                trading_metrics=trading_metrics,
                regime_metrics=regime_metrics,
                n_samples=len(X),
                n_folds=1
            )

        except Exception as e:
            self.logger.error(f"Holdout validation failed: {e}")
            raise

    def _bootstrap_validation(self, model: Any, X: np.ndarray, y: np.ndarray,
                             market_data: Optional[np.ndarray],
                             regime_predictions: Optional[np.ndarray]) -> ValidationResult:
        """Perform bootstrap validation."""
        try:
            bootstrap_scores = {}
            bootstrap_results = {}

            for metric in self.config.metrics:
                scores = []

                for i in range(self.config.bootstrap_iterations):
                    # Bootstrap sample
                    indices = np.random.choice(len(X), size=len(X), replace=True)
                    X_boot = X[indices]
                    y_boot = y[indices]

                    # Train model
                    model.fit(X_boot, y_boot)

                    # Make predictions on original test set
                    y_pred = model.predict(X)

                    # Calculate metric
                    if metric == MetricType.ACCURACY:
                        score = accuracy_score(y, y_pred)
                    elif metric == MetricType.PRECISION:
                        score = precision_score(y, y_pred, average='weighted', zero_division=0)
                    elif metric == MetricType.RECALL:
                        score = recall_score(y, y_pred, average='weighted', zero_division=0)
                    elif metric == MetricType.F1_SCORE:
                        score = f1_score(y, y_pred, average='weighted', zero_division=0)
                    else:
                        score = self._calculate_custom_metric(y, y_pred, metric)

                    scores.append(score)

                bootstrap_scores[metric.value] = scores
                bootstrap_results[metric.value] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'scores': scores
                }

            # Calculate final metrics
            metrics = {metric.value: bootstrap_results[metric.value]['mean'] for metric in self.config.metrics}
            metrics_std = {metric.value: bootstrap_results[metric.value]['std'] for metric in self.config.metrics}
            metrics_ci = {metric.value: self._calculate_confidence_interval(bootstrap_results[metric.value]['scores'])
                         for metric in self.config.metrics}

            # Calculate trading metrics if enabled
            trading_metrics = None
            if self.config.enable_trading_metrics and market_data is not None:
                trading_metrics = self._calculate_trading_metrics(model, X, y, market_data)

            # Calculate regime metrics if enabled
            regime_metrics = None
            if self.config.enable_regime_metrics and regime_predictions is not None:
                regime_metrics = self._calculate_regime_metrics(model, X, y, regime_predictions)

            return ValidationResult(
                success=True,
                validation_type=self.config.validation_type.value,
                metrics=metrics,
                metrics_std=metrics_std,
                metrics_ci=metrics_ci,
                bootstrap_results=bootstrap_results,
                trading_metrics=trading_metrics,
                regime_metrics=regime_metrics,
                n_samples=len(X),
                n_folds=self.config.bootstrap_iterations
            )

        except Exception as e:
            self.logger.error(f"Bootstrap validation failed: {e}")
            raise

    def _calculate_custom_metric_cv(self, model: Any, X: np.ndarray, y: np.ndarray, metric: MetricType) -> np.ndarray:
        """Calculate custom metric for cross-validation."""
        try:
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=self.config.random_state)
            scores = []

            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                score = self._calculate_custom_metric(y_test, y_pred, metric)
                scores.append(score)

            return np.array(scores)

        except Exception as e:
            self.logger.warning(f"Custom metric calculation failed: {e}")
            return np.zeros(self.config.n_folds)

    def _calculate_custom_metric(self, y_true: np.ndarray, y_pred: np.ndarray, metric: MetricType) -> float:
        """Calculate custom metric."""
        try:
            if metric == MetricType.SHARPE_RATIO:
                return self._calculate_sharpe_ratio(y_true, y_pred)
            elif metric == MetricType.MAX_DRAWDOWN:
                return self._calculate_max_drawdown(y_true, y_pred)
            elif metric == MetricType.WIN_RATE:
                return self._calculate_win_rate(y_true, y_pred)
            elif metric == MetricType.PROFIT_FACTOR:
                return self._calculate_profit_factor(y_true, y_pred)
            elif metric == MetricType.REGIME_STABILITY:
                return self._calculate_regime_stability(y_pred)
            elif metric == MetricType.TRANSITION_ACCURACY:
                return self._calculate_transition_accuracy(y_pred)
            else:
                return 0.0

        except Exception as e:
            self.logger.warning(f"Custom metric calculation failed: {e}")
            return 0.0

    def _calculate_trading_metrics(self, model: Any, X: np.ndarray, y: np.ndarray,
                                 market_data: np.ndarray) -> Dict[str, float]:
        """Calculate trading-specific metrics."""
        try:
            # Make predictions
            y_pred = model.predict(X)

            # Simulate trading
            returns = self._simulate_trading(market_data, y_pred)

            if len(returns) == 0:
                return {}

            # Calculate trading metrics
            total_return = np.prod(1 + returns) - 1
            sharpe_ratio = self._calculate_sharpe_ratio_from_returns(returns)
            max_drawdown = self._calculate_max_drawdown_from_returns(returns)
            win_rate = np.mean(returns > 0)

            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'volatility': np.std(returns)
            }

        except Exception as e:
            self.logger.warning(f"Trading metrics calculation failed: {e}")
            return {}

    def _calculate_regime_metrics(self, model: Any, X: np.ndarray, y: np.ndarray,
                                regime_predictions: np.ndarray) -> Dict[str, float]:
        """Calculate regime-specific metrics."""
        try:
            # Make predictions
            y_pred = model.predict(X)

            # Calculate regime stability
            regime_stability = self._calculate_regime_stability(regime_predictions)

            # Calculate transition accuracy
            transition_accuracy = self._calculate_transition_accuracy(regime_predictions)

            # Calculate regime consistency
            regime_consistency = self._calculate_regime_consistency(regime_predictions)

            return {
                'regime_stability': regime_stability,
                'transition_accuracy': transition_accuracy,
                'regime_consistency': regime_consistency,
                'n_regimes': len(np.unique(regime_predictions))
            }

        except Exception as e:
            self.logger.warning(f"Regime metrics calculation failed: {e}")
            return {}

    def _calculate_confidence_interval(self, scores: np.ndarray) -> Tuple[float, float]:
        """Calculate confidence interval for scores."""
        try:
            alpha = 1 - self.config.confidence_level
            lower = np.percentile(scores, alpha / 2 * 100)
            upper = np.percentile(scores, (1 - alpha / 2) * 100)
            return (lower, upper)

        except Exception:
            return (0.0, 0.0)

    def _simulate_trading(self, market_data: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """Simulate trading based on predictions."""
        try:
            if market_data.shape[1] < 4:
                return np.array([])

            close_prices = market_data[:, 3]
            returns = []

            for i in range(1, len(predictions)):
                if predictions[i] != predictions[i-1]:
                    # Regime change - simulate trade
                    trade_return = (close_prices[i] - close_prices[i-1]) / close_prices[i-1]
                    returns.append(trade_return)

            return np.array(returns)

        except Exception:
            return np.array([])

    def _calculate_sharpe_ratio_from_returns(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio from returns."""
        try:
            if len(returns) == 0 or np.std(returns) == 0:
                return 0.0

            excess_returns = returns - self.config.risk_free_rate / 252
            return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

        except Exception:
            return 0.0

    def _calculate_max_drawdown_from_returns(self, returns: np.ndarray) -> float:
        """Calculate max drawdown from returns."""
        try:
            if len(returns) == 0:
                return 0.0

            cumulative = np.cumprod(1 + returns)
            peak = cumulative[0]
            max_dd = 0.0

            for value in cumulative:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)

            return max_dd

        except Exception:
            return 0.0

    def _calculate_regime_stability(self, regime_predictions: np.ndarray) -> float:
        """Calculate regime stability."""
        try:
            if len(regime_predictions) < 2:
                return 0.0

            regime_changes = np.sum(np.diff(regime_predictions) != 0)
            total_periods = len(regime_predictions) - 1

            stability = 1.0 - (regime_changes / total_periods) if total_periods > 0 else 0.0
            return max(0.0, min(1.0, stability))

        except Exception:
            return 0.0

    def _calculate_transition_accuracy(self, regime_predictions: np.ndarray) -> float:
        """Calculate transition accuracy."""
        try:
            if len(regime_predictions) < 3:
                return 0.5

            unique_regimes = np.unique(regime_predictions)
            n_regimes = len(unique_regimes)

            if n_regimes < 2:
                return 0.5

            # Calculate transition matrix
            transition_matrix = np.zeros((n_regimes, n_regimes))

            for i in range(len(regime_predictions) - 1):
                current_regime = regime_predictions[i]
                next_regime = regime_predictions[i + 1]

                if current_regime in unique_regimes and next_regime in unique_regimes:
                    current_idx = np.where(unique_regimes == current_regime)[0][0]
                    next_idx = np.where(unique_regimes == next_regime)[0][0]
                    transition_matrix[current_idx, next_idx] += 1

            # Calculate transition accuracy
            total_transitions = np.sum(transition_matrix)
            if total_transitions > 0:
                diagonal_sum = np.trace(transition_matrix)
                transition_accuracy = diagonal_sum / total_transitions
            else:
                transition_accuracy = 0.5

            return min(transition_accuracy, 1.0)

        except Exception:
            return 0.0

    def _calculate_regime_consistency(self, regime_predictions: np.ndarray) -> float:
        """Calculate regime consistency."""
        try:
            if len(regime_predictions) < 2:
                return 0.0

            # Calculate consistency within each regime
            unique_regimes = np.unique(regime_predictions)
            consistency_scores = []

            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                regime_indices = np.where(regime_mask)[0]

                if len(regime_indices) > 1:
                    # Calculate consistency within regime
                    regime_consistency = 1.0 - np.sum(np.diff(regime_indices) > 1) / len(regime_indices)
                    consistency_scores.append(regime_consistency)

            return np.mean(consistency_scores) if consistency_scores else 0.0

        except Exception:
            return 0.0

    def _calculate_sharpe_ratio(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        try:
            # Simplified Sharpe ratio calculation
            returns = y_pred - y_true
            if np.std(returns) == 0:
                return 0.0

            return np.mean(returns) / np.std(returns)

        except Exception:
            return 0.0

    def _calculate_max_drawdown(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate max drawdown."""
        try:
            returns = y_pred - y_true
            cumulative = np.cumsum(returns)

            peak = cumulative[0]
            max_dd = 0.0

            for value in cumulative:
                if value > peak:
                    peak = value
                dd = (peak - value) / (peak + 1e-8)
                max_dd = max(max_dd, dd)

            return max_dd

        except Exception:
            return 0.0

    def _calculate_win_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate win rate."""
        try:
            returns = y_pred - y_true
            return np.mean(returns > 0)

        except Exception:
            return 0.0

    def _calculate_profit_factor(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate profit factor."""
        try:
            returns = y_pred - y_true
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]

            if len(negative_returns) == 0:
                return float('inf') if len(positive_returns) > 0 else 0.0

            return np.sum(positive_returns) / abs(np.sum(negative_returns))

        except Exception:
            return 0.0

# Convenience functions
def create_unified_validation_system(config: Optional[ValidationConfig] = None) -> UnifiedValidationSystem:
    """Create a unified validation system."""
    if config is None:
        config = ValidationConfig()
    return UnifiedValidationSystem(config)

def quick_validation(model: Any, X: np.ndarray, y: np.ndarray,
                    config: Optional[ValidationConfig] = None) -> ValidationResult:
    """Quick validation with default settings."""
    validator = create_unified_validation_system(config)
    return validator.validate(model, X, y)
