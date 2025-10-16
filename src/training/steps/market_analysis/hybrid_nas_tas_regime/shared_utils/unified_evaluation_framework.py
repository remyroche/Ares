"""
Unified Evaluation Framework

This module provides a comprehensive evaluation framework that consolidates
evaluation capabilities for both TAS and NAS architectures, including economic
significance, trading viability, regime-specific evaluation, and multi-objective
assessment.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import torch
from scipy import stats
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from .unified_architecture_config import ArchitectureType, OptimizationObjective
from .unified_economic_evaluator import UnifiedEconomicSignificanceEvaluator, EconomicEvaluationConfig
from .unified_trading_viability_evaluator import UnifiedTradingViabilityEvaluator, TradingViabilityConfig

# Import feature importance integration if available
try:
    from ...market_analysis.shared_utils.feature_importance_integration import (
        FeatureImportanceIntegrationManager, FeatureImportanceIntegrationConfig
    )
    FEATURE_IMPORTANCE_AVAILABLE = True
except ImportError:
    FEATURE_IMPORTANCE_AVAILABLE = False

logger = logging.getLogger(__name__)

class EvaluationType(Enum):
    """Types of evaluations."""
    BASIC = "basic"
    TRADING = "trading"
    ECONOMIC = "economic"
    REGIME_SPECIFIC = "regime_specific"
    MULTI_OBJECTIVE = "multi_objective"
    COMPREHENSIVE = "comprehensive"

class EvaluationMetric(Enum):
    """Types of evaluation metrics."""
    # Basic metrics
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"

    # Trading metrics
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"

    # Economic metrics
    ECONOMIC_SIGNIFICANCE = "economic_significance"
    TRADING_VIABILITY = "trading_viability"
    MARKET_IMPACT = "market_impact"
    LIQUIDITY_SCORE = "liquidity_score"

    # Risk metrics
    VAR_95 = "var_95"
    CVAR_95 = "cvar_95"
    VOLATILITY = "volatility"
    BETA = "beta"

    # Regime metrics
    REGIME_STABILITY = "regime_stability"
    ADAPTATION_SPEED = "adaptation_speed"
    TRANSITION_ACCURACY = "transition_accuracy"

@dataclass
class EvaluationResult:
    """Comprehensive evaluation result."""
    model_name: str
    architecture_type: ArchitectureType
    evaluation_type: EvaluationType
    timestamp: datetime = field(default_factory=datetime.now)

    # Basic performance metrics
    basic_metrics: Dict[EvaluationMetric, float] = field(default_factory=dict)

    # Trading performance metrics
    trading_metrics: Dict[EvaluationMetric, float] = field(default_factory=dict)

    # Economic significance metrics
    economic_metrics: Dict[EvaluationMetric, float] = field(default_factory=dict)

    # Risk metrics
    risk_metrics: Dict[EvaluationMetric, float] = field(default_factory=dict)

    # Regime-specific metrics
    regime_metrics: Dict[EvaluationMetric, float] = field(default_factory=dict)

    # Model characteristics
    model_complexity: float = 0.0
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage_mb: float = 0.0

    # Validation flags
    passed_economic_significance: bool = False
    passed_trading_viability: bool = False
    passed_risk_limits: bool = False
    passed_regime_stability: bool = False

    # Detailed analysis
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    feature_importance_analysis: Dict[str, Any] = field(default_factory=dict)
    error_analysis: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    evaluation_config: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

@dataclass
class EvaluationConfig:
    """Configuration for unified evaluation."""
    evaluation_type: EvaluationType = EvaluationType.COMPREHENSIVE

    # Basic evaluation
    enable_basic_metrics: bool = True
    basic_metrics: List[EvaluationMetric] = field(default_factory=lambda: [
        EvaluationMetric.ACCURACY, EvaluationMetric.PRECISION,
        EvaluationMetric.RECALL, EvaluationMetric.F1_SCORE
    ])

    # Trading evaluation
    enable_trading_metrics: bool = True
    trading_metrics: List[EvaluationMetric] = field(default_factory=lambda: [
        EvaluationMetric.SHARPE_RATIO, EvaluationMetric.MAX_DRAWDOWN,
        EvaluationMetric.WIN_RATE, EvaluationMetric.PROFIT_FACTOR
    ])

    # Economic evaluation
    enable_economic_metrics: bool = True
    economic_threshold: float = 0.7
    trading_viability_threshold: float = 0.6

    # Risk evaluation
    enable_risk_metrics: bool = True
    max_drawdown_threshold: float = 0.15
    var_confidence_level: float = 0.95

    # Regime evaluation
    enable_regime_metrics: bool = True
    regime_stability_threshold: float = 0.8
    min_regime_samples: int = 50

    # Cross-validation
    enable_cross_validation: bool = True
    cv_folds: int = 5
    cv_strategy: str = "temporal"

    # Feature importance analysis
    enable_feature_importance: bool = True
    feature_importance_methods: List[str] = field(default_factory=lambda: ["mutual_information", "f_classif"])
    feature_importance_threshold: float = 0.01

    # Bootstrap analysis
    enable_bootstrap: bool = True
    bootstrap_iterations: int = 100
    bootstrap_confidence: float = 0.95

    # Uncertainty quantification
    enable_uncertainty_quantification: bool = True
    uncertainty_samples: int = 100

class UnifiedEvaluationFramework:
    """Unified evaluation framework for TAS and NAS architectures."""

    def __init__(self,
                 architecture_type: ArchitectureType,
                 config: EvaluationConfig = None):
        """Initialize the unified evaluation framework.

        Args:
            architecture_type: Type of architecture (TAS/NAS/Hybrid)
            config: Evaluation configuration
        """
        self.architecture_type = architecture_type
        self.config = config or EvaluationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize specialized evaluators
        self._initialize_specialized_evaluators()

        # Initialize feature importance manager if available
        if FEATURE_IMPORTANCE_AVAILABLE and self.config.enable_feature_importance:
            try:
                importance_config = FeatureImportanceIntegrationConfig(
                    importance_methods=self.config.feature_importance_methods,
                    importance_threshold=self.config.feature_importance_threshold,
                    enable_pre_clustering_analysis=False,  # Not needed for model evaluation
                    enable_post_clustering_analysis=True,  # For regime analysis
                    enable_regime_characterization=True,
                    include_detailed_analysis=True
                )
                self.feature_importance_manager = FeatureImportanceIntegrationManager(importance_config)
                self.logger.info("✅ Feature importance manager initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Feature importance manager initialization failed: {e}")
                self.feature_importance_manager = None

        # Evaluation state
        self.evaluation_history: List[EvaluationResult] = []
        self.performance_baselines: Dict[str, float] = {}

        self.logger.info(f"✅ Unified Evaluation Framework initialized for {architecture_type.value}")
        self.logger.info(f"   Evaluation Type: {self.config.evaluation_type.value}")
        self.logger.info(f"   Cross-validation: {self.config.enable_cross_validation}")
        self.logger.info(f"   Bootstrap: {self.config.enable_bootstrap}")

    def _initialize_specialized_evaluators(self):
        """Initialize specialized evaluators."""
        try:
            # Initialize economic significance evaluator
            if self.config.enable_economic_metrics:
                economic_config = EconomicEvaluationConfig(
                    significance_threshold=self.config.economic_threshold,
                    price_impact_threshold=0.5,
                    volume_threshold=0.4,
                    volatility_threshold=0.5,
                    trend_threshold=0.6,
                    efficiency_threshold=0.5,
                    # Ensure all required attributes are set with defaults
                    price_impact_weight=0.25,
                    volume_significance_weight=0.15,
                    volatility_impact_weight=0.20,
                    trend_consistency_weight=0.15,
                    market_efficiency_weight=0.10,
                    economic_indicators_weight=0.10,
                    trading_opportunity_weight=0.05,
                    risk_adjustment_weight=0.05
                )
                self.economic_evaluator = UnifiedEconomicSignificanceEvaluator(economic_config)

            # Initialize trading viability evaluator
            if self.config.enable_trading_metrics:
                trading_config = TradingViabilityConfig(
                    viability_threshold=self.config.trading_viability_threshold,
                    min_trading_frequency=0.1,
                    max_trading_frequency=10.0,
                    min_position_duration=5.0,
                    max_position_duration=1440.0,
                    min_model_confidence=0.6,
                    min_risk_adjusted_return=0.1
                )
                self.trading_evaluator = UnifiedTradingViabilityEvaluator(trading_config)

            self.logger.info("✅ Specialized evaluators initialized")

        except Exception as e:
            self.logger.warning(f"Failed to initialize specialized evaluators: {e}")
            self.economic_evaluator = None
            self.trading_evaluator = None

    def evaluate_model(self,
                      model: Any,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      market_data: Optional[pd.DataFrame] = None,
                      regime_labels: Optional[np.ndarray] = None,
                      model_name: str = "unknown_model",
                      metadata: Dict[str, Any] = None) -> EvaluationResult:
        """Comprehensive model evaluation.

        Args:
            model: Model to evaluate
            X_test: Test features
            y_test: Test labels
            market_data: Optional market data for trading simulation
            regime_labels: Optional regime labels for regime-specific evaluation
            model_name: Name of the model
            metadata: Optional metadata

        Returns:
            Comprehensive evaluation result
        """
        self.logger.info(f"🔬 Starting comprehensive evaluation of {model_name}")
        start_time = time.time()

        # Create evaluation result
        result = EvaluationResult(
            model_name=model_name,
            architecture_type=self.architecture_type,
            evaluation_type=self.config.evaluation_type,
            evaluation_config=self.config.__dict__,
            notes=""
        )

        try:
            # Basic metrics evaluation
            if self.config.enable_basic_metrics:
                self._evaluate_basic_metrics(model, X_test, y_test, result)

            # Trading metrics evaluation
            if self.config.enable_trading_metrics and market_data is not None:
                self._evaluate_trading_metrics(model, X_test, y_test, market_data, result)

            # Economic significance evaluation
            if self.config.enable_economic_metrics and market_data is not None:
                self._evaluate_economic_metrics(model, X_test, y_test, market_data, result)

            # Risk metrics evaluation
            if self.config.enable_risk_metrics:
                self._evaluate_risk_metrics(model, X_test, y_test, result)

            # Regime-specific evaluation
            if self.config.enable_regime_metrics and regime_labels is not None:
                self._evaluate_regime_metrics(model, X_test, y_test, regime_labels, result)

            # Feature importance analysis if available
            if (self.config.enable_feature_importance and
                self.feature_importance_manager and
                regime_labels is not None):
                self._evaluate_feature_importance(model, X_test, y_test, regime_labels, result)

            # Model characteristics
            self._evaluate_model_characteristics(model, X_test, result)

            # Cross-validation if enabled
            if self.config.enable_cross_validation:
                self._perform_cross_validation(model, X_test, y_test, result)

            # Bootstrap analysis if enabled
            if self.config.enable_bootstrap:
                self._perform_bootstrap_analysis(model, X_test, y_test, result)

            # Uncertainty quantification if enabled
            if self.config.enable_uncertainty_quantification:
                self._perform_uncertainty_quantification(model, X_test, y_test, result)

            # Validation checks
            self._perform_validation_checks(result)

            # Calculate evaluation time
            result.training_time = time.time() - start_time

            # Store in history
            self.evaluation_history.append(result)

            self.logger.info(f"✅ Evaluation completed for {model_name}")
            self.logger.info(f"   Accuracy: {result.basic_metrics.get(EvaluationMetric.ACCURACY, 0):.4f}")
            self.logger.info(f"   Sharpe Ratio: {result.trading_metrics.get(EvaluationMetric.SHARPE_RATIO, 0):.4f}")
            self.logger.info(f"   Economic Significance: {result.economic_metrics.get(EvaluationMetric.ECONOMIC_SIGNIFICANCE, 0):.4f}")

            return result

        except Exception as e:
            self.logger.error(f"❌ Evaluation failed for {model_name}: {e}")
            result.notes = f"Evaluation failed: {str(e)}"
            result.training_time = time.time() - start_time
            return result

    def _evaluate_basic_metrics(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, result: EvaluationResult):
        """Evaluate basic performance metrics."""
        try:
            # Get predictions
            predictions = self._get_predictions(model, X_test)

            # Calculate basic metrics
            if len(np.unique(y_test)) == 2:  # Binary classification
                pred_classes = (predictions > 0.5).astype(int)

                result.basic_metrics[EvaluationMetric.ACCURACY] = accuracy_score(y_test, pred_classes)
                result.basic_metrics[EvaluationMetric.PRECISION] = precision_score(y_test, pred_classes, average='weighted', zero_division=0)
                result.basic_metrics[EvaluationMetric.RECALL] = recall_score(y_test, pred_classes, average='weighted', zero_division=0)
                result.basic_metrics[EvaluationMetric.F1_SCORE] = f1_score(y_test, pred_classes, average='weighted', zero_division=0)

                # ROC AUC if probabilities available
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(X_test)
                        if proba.ndim > 1 and proba.shape[1] == 2:
                            result.basic_metrics[EvaluationMetric.ROC_AUC] = roc_auc_score(y_test, proba[:, 1])
                    except Exception:
                        pass

            else:  # Regression or multi-class
                pred_classes = predictions.round() if predictions.dtype.kind == 'f' else predictions

                result.basic_metrics[EvaluationMetric.ACCURACY] = accuracy_score(y_test, pred_classes)

                # Regression metrics
                result.basic_metrics[EvaluationMetric.MSE] = mean_squared_error(y_test, predictions)
                result.basic_metrics[EvaluationMetric.MAE] = mean_absolute_error(y_test, predictions)
                result.basic_metrics[EvaluationMetric.R2_SCORE] = r2_score(y_test, predictions)

        except Exception as e:
            self.logger.warning(f"Basic metrics evaluation failed: {e}")

    def _evaluate_trading_metrics(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                                market_data: pd.DataFrame, result: EvaluationResult):
        """Evaluate trading-specific metrics."""
        try:
            # Generate trading signals
            predictions = self._get_predictions(model, X_test)

            # Convert predictions to trading signals
            if len(np.unique(y_test)) == 2:  # Binary
                trading_signals = (predictions > 0.5).astype(int) * 2 - 1  # Convert to -1, 1
            else:  # Multi-class or regression
                if hasattr(predictions, 'shape') and len(predictions.shape) > 1:
                    trading_signals = np.argmax(predictions, axis=1) - 1  # Convert to -1, 0, 1
                else:
                    trading_signals = np.sign(predictions - np.mean(predictions))  # Sign of deviation from mean

            # Simulate trading
            trading_results = self._simulate_trading(market_data, trading_signals)

            # Calculate trading metrics
            if len(trading_results['returns']) > 0:
                returns = np.array(trading_results['returns'])

                # Basic trading metrics
                result.trading_metrics[EvaluationMetric.SHARPE_RATIO] = self._calculate_sharpe_ratio(returns)
                result.trading_metrics[EvaluationMetric.MAX_DRAWDOWN] = self._calculate_max_drawdown(np.cumprod(1 + returns))
                result.trading_metrics[EvaluationMetric.WIN_RATE] = np.mean(returns > 0)
                result.trading_metrics[EvaluationMetric.PROFIT_FACTOR] = self._calculate_profit_factor(returns)
                result.trading_metrics[EvaluationMetric.CALMAR_RATIO] = self._calculate_calmar_ratio(returns)
                result.trading_metrics[EvaluationMetric.SORTINO_RATIO] = self._calculate_sortino_ratio(returns)

                # Trading viability using specialized evaluator
                if self.trading_evaluator:
                    viability_score = self.trading_evaluator.evaluate_trading_viability(
                        returns, trading_results
                    )
                    result.trading_metrics[EvaluationMetric.TRADING_VIABILITY] = viability_score

        except Exception as e:
            self.logger.warning(f"Trading metrics evaluation failed: {e}")

    def _evaluate_economic_metrics(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                                 market_data: pd.DataFrame, result: EvaluationResult):
        """Evaluate economic significance metrics."""
        try:
            if self.economic_evaluator:
                # Generate regime predictions if not provided
                regime_predictions = self._get_predictions(model, X_test)

                # Evaluate economic significance
                economic_scores = self.economic_evaluator.evaluate(
                    market_data.values, regime_predictions
                )

                result.economic_metrics[EvaluationMetric.ECONOMIC_SIGNIFICANCE] = np.mean(economic_scores)
                result.economic_metrics[EvaluationMetric.MARKET_IMPACT] = np.std(economic_scores)

        except Exception as e:
            self.logger.warning(f"Economic metrics evaluation failed: {e}")

    def _evaluate_risk_metrics(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, result: EvaluationResult):
        """Evaluate risk metrics."""
        try:
            # Get predictions
            predictions = self._get_predictions(model, X_test)

            # Calculate prediction errors
            errors = predictions - y_test

            # Risk metrics
            result.risk_metrics[EvaluationMetric.VOLATILITY] = np.std(errors)

            # Value at Risk
            confidence_level = self.config.var_confidence_level
            result.risk_metrics[EvaluationMetric.VAR_95] = np.percentile(errors, (1 - confidence_level) * 100)

            # Conditional Value at Risk
            var_threshold = result.risk_metrics[EvaluationMetric.VAR_95]
            tail_errors = errors[errors <= var_threshold]
            result.risk_metrics[EvaluationMetric.CVAR_95] = np.mean(tail_errors) if len(tail_errors) > 0 else var_threshold

            # Beta (if market data available)
            # This would require market returns data

        except Exception as e:
            self.logger.warning(f"Risk metrics evaluation failed: {e}")

    def _evaluate_regime_metrics(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                               regime_labels: np.ndarray, result: EvaluationResult):
        """Evaluate regime-specific metrics."""
        try:
            # Get predictions
            predictions = self._get_predictions(model, X_test)

            # Group by regime
            unique_regimes = np.unique(regime_labels)
            regime_performance = {}

            for regime in unique_regimes:
                regime_mask = regime_labels == regime
                if np.sum(regime_mask) >= self.config.min_regime_samples:
                    regime_y_true = y_test[regime_mask]
                    regime_y_pred = predictions[regime_mask]

                    # Calculate regime-specific metrics
                    regime_accuracy = accuracy_score(regime_y_true, regime_y_pred.round())

                    regime_performance[regime] = {
                        'accuracy': regime_accuracy,
                        'sample_count': np.sum(regime_mask),
                        'predictions_mean': np.mean(regime_y_pred),
                        'predictions_std': np.std(regime_y_pred)
                    }

            result.regime_performance = regime_performance

            # Calculate regime stability
            if regime_performance:
                regime_accuracies = [perf['accuracy'] for perf in regime_performance.values()]
                result.regime_metrics[EvaluationMetric.REGIME_STABILITY] = np.mean(regime_accuracies)
                result.regime_metrics[EvaluationMetric.ADAPTATION_SPEED] = 1.0 / (np.std(regime_accuracies) + 1e-8)

        except Exception as e:
            self.logger.warning(f"Regime metrics evaluation failed: {e}")

    def _evaluate_feature_importance(self, model: Any, X_test: np.ndarray,
                                   y_test: np.ndarray, regime_labels: np.ndarray,
                                   result: EvaluationResult):
        """Evaluate feature importance for regime discovery and interpretability."""
        try:
            # Prepare features for analysis
            # Try to get meaningful feature names from the model or data
            feature_names = []

            # Check if model has feature names
            if hasattr(model, 'feature_names_'):
                feature_names = getattr(model, 'feature_names_')
            elif hasattr(model, 'get_feature_names_out'):
                try:
                    feature_names = model.get_feature_names_out()
                except:
                    pass

            # If still no feature names, generate them
            if not feature_names and X_test.shape[1] > 0:
                feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]

            # Perform feature importance analysis
            importance_analysis = self.feature_importance_manager.analyze_post_clustering_regimes(
                X_test, feature_names, regime_labels
            )

            if importance_analysis:
                result.feature_importance_analysis = importance_analysis

                # Extract key insights for the result
                feature_ranking = importance_analysis.get('feature_importance_ranking', [])
                if feature_ranking:
                    # Store top feature importances
                    top_features = {}
                    for i, (feature_name, importance_score) in enumerate(feature_ranking[:20]):
                        top_features[feature_name] = float(importance_score)
                    result.feature_importance = top_features

                # Add interpretation to notes
                interpretation = importance_analysis.get('interpretation', '')
                if interpretation and result.notes:
                    result.notes += f" | Feature Importance: {interpretation}"
                elif interpretation:
                    result.notes = f"Feature Importance: {interpretation}"

                self.logger.info("✅ Feature importance analysis completed")
            else:
                self.logger.warning("⚠️ Feature importance analysis returned no results")

        except Exception as e:
            self.logger.warning(f"Feature importance evaluation failed: {e}")

    def _evaluate_model_characteristics(self, model: Any, X_test: np.ndarray, result: EvaluationResult):
        """Evaluate model characteristics."""
        try:
            # Model complexity
            if hasattr(model, 'get_n_params'):
                result.model_complexity = model.get_n_params()
            elif hasattr(model, 'n_features_in_'):
                result.model_complexity = model.n_features_in_ * 10  # Rough estimate
            else:
                result.model_complexity = 100  # Default estimate

            # Inference time
            start_time = time.time()
            _ = self._get_predictions(model, X_test[:100])  # Small batch for timing
            result.inference_time = (time.time() - start_time) / 100

            # Memory usage (rough estimate)
            if hasattr(model, 'get_n_params'):
                result.memory_usage_mb = model.get_n_params() * 4 / (1024 * 1024)  # 4 bytes per parameter
            else:
                result.memory_usage_mb = 10  # Default estimate

        except Exception as e:
            self.logger.warning(f"Model characteristics evaluation failed: {e}")

    def _perform_cross_validation(self, model: Any, X: np.ndarray, y: np.ndarray, result: EvaluationResult):
        """Perform cross-validation."""
        try:
            if self.config.cv_strategy == "temporal":
                cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            else:
                from sklearn.model_selection import KFold
                cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)

            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

            # Store CV results in error analysis
            result.error_analysis['cross_validation'] = {
                'mean_score': np.mean(cv_scores),
                'std_score': np.std(cv_scores),
                'scores': cv_scores.tolist(),
                'cv_folds': self.config.cv_folds
            }

        except Exception as e:
            self.logger.warning(f"Cross-validation failed: {e}")

    def _perform_bootstrap_analysis(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, result: EvaluationResult):
        """Perform bootstrap analysis for uncertainty quantification."""
        try:
            bootstrap_scores = []

            for i in range(self.config.bootstrap_iterations):
                # Bootstrap sample
                indices = np.random.choice(len(X_test), size=len(X_test), replace=True)
                X_boot = X_test[indices]
                y_boot = y_test[indices]

                # Get predictions
                pred_boot = self._get_predictions(model, X_boot)
                pred_classes = pred_boot.round() if pred_boot.dtype.kind == 'f' else pred_boot

                # Calculate accuracy
                accuracy_boot = accuracy_score(y_boot, pred_classes)
                bootstrap_scores.append(accuracy_boot)

            # Calculate confidence interval
            alpha = 1 - self.config.bootstrap_confidence
            ci_lower = np.percentile(bootstrap_scores, (alpha / 2) * 100)
            ci_upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)

            # Store bootstrap results
            result.error_analysis['bootstrap'] = {
                'mean_score': np.mean(bootstrap_scores),
                'std_score': np.std(bootstrap_scores),
                'confidence_interval': [ci_lower, ci_upper],
                'confidence_level': self.config.bootstrap_confidence,
                'iterations': self.config.bootstrap_iterations
            }

        except Exception as e:
            self.logger.warning(f"Bootstrap analysis failed: {e}")

    def _perform_uncertainty_quantification(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, result: EvaluationResult):
        """Perform uncertainty quantification."""
        try:
            # Get multiple predictions if model supports it
            if hasattr(model, 'predict_proba'):
                # Use probability predictions for uncertainty
                proba = model.predict_proba(X_test)
                if proba.ndim > 1:
                    # Use entropy as uncertainty measure
                    entropy = -np.sum(proba * np.log(proba + 1e-8), axis=1)
                    result.error_analysis['uncertainty'] = {
                        'mean_entropy': np.mean(entropy),
                        'std_entropy': np.std(entropy),
                        'max_entropy': np.max(entropy),
                        'min_entropy': np.min(entropy)
                    }

            # Alternative: Use prediction variance across multiple samples
            predictions_samples = []
            for _ in range(min(self.config.uncertainty_samples, 10)):  # Limit to 10 samples
                pred_sample = self._get_predictions(model, X_test)
                predictions_samples.append(pred_sample)

            if len(predictions_samples) > 1:
                predictions_array = np.array(predictions_samples)
                prediction_variance = np.var(predictions_array, axis=0)

                result.error_analysis['prediction_uncertainty'] = {
                    'mean_variance': np.mean(prediction_variance),
                    'std_variance': np.std(prediction_variance),
                    'max_variance': np.max(prediction_variance),
                    'min_variance': np.min(prediction_variance)
                }

        except Exception as e:
            self.logger.warning(f"Uncertainty quantification failed: {e}")

    def _perform_validation_checks(self, result: EvaluationResult):
        """Perform validation checks and set flags."""
        try:
            # Economic significance check
            economic_score = result.economic_metrics.get(EvaluationMetric.ECONOMIC_SIGNIFICANCE, 0)
            result.passed_economic_significance = economic_score >= self.config.economic_threshold

            # Trading viability check
            trading_score = result.trading_metrics.get(EvaluationMetric.TRADING_VIABILITY, 0)
            result.passed_trading_viability = trading_score >= self.config.trading_viability_threshold

            # Risk limits check
            max_drawdown = result.trading_metrics.get(EvaluationMetric.MAX_DRAWDOWN, 0)
            result.passed_risk_limits = abs(max_drawdown) <= self.config.max_drawdown_threshold

            # Regime stability check
            regime_stability = result.regime_metrics.get(EvaluationMetric.REGIME_STABILITY, 0)
            result.passed_regime_stability = regime_stability >= self.config.regime_stability_threshold

        except Exception as e:
            self.logger.warning(f"Validation checks failed: {e}")

    def _get_predictions(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get predictions from model."""
        try:
            if hasattr(model, 'predict'):
                return model.predict(X)
            elif hasattr(model, 'forward'):
                # PyTorch model
                with torch.no_grad():
                    X_tensor = torch.tensor(X, dtype=torch.float32)
                    predictions = model.forward(X_tensor)
                    return predictions.numpy()
            else:
                raise ValueError("Model does not support prediction")
        except Exception as e:
            self.logger.warning(f"Prediction failed: {e}")
            return np.zeros(len(X))

    def _simulate_trading(self, market_data: pd.DataFrame, signals: np.ndarray) -> Dict[str, Any]:
        """Simulate trading based on signals."""
        trading_results = {
            'returns': [],
            'positions': [],
            'trades': [],
            'trade_durations': []
        }

        try:
            # Simple trading simulation
            position = 0
            entry_price = 0
            trade_start = None

            for i, (signal, row) in enumerate(zip(signals, market_data.itertuples())):
                current_price = getattr(row, 'close', row[3] if len(row) > 3 else 1.0)

                if signal != 0 and position == 0:
                    # Enter position
                    position = signal
                    entry_price = current_price
                    trade_start = i
                elif signal != position and position != 0:
                    # Exit position
                    if trade_start is not None:
                        trade_duration = i - trade_start
                        trade_return = (current_price - entry_price) / entry_price * position

                        trading_results['returns'].append(trade_return)
                        trading_results['trades'].append({
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'return': trade_return,
                            'duration': trade_duration,
                            'direction': position
                        })
                        trading_results['trade_durations'].append(trade_duration)

                    position = 0
                    trade_start = None

        except Exception as e:
            self.logger.warning(f"Trading simulation failed: {e}")

        return trading_results

    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(cumulative_returns) == 0:
            return 0.0

        peak = cumulative_returns[0]
        max_dd = 0.0

        for value in cumulative_returns:
            if value > peak:
                peak = value
            dd = (peak - value) / (1 + peak) if peak != 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd

    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor."""
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        if len(negative_returns) == 0:
            return float('inf') if len(positive_returns) > 0 else 0.0

        gross_profit = np.sum(positive_returns)
        gross_loss = abs(np.sum(negative_returns))

        return gross_profit / gross_loss if gross_loss > 0 else 0.0

    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio."""
        if len(returns) == 0:
            return 0.0

        annual_return = np.prod(1 + returns) ** (252 / len(returns)) - 1
        max_dd = self._calculate_max_drawdown(np.cumprod(1 + returns))

        return annual_return / abs(max_dd) if max_dd != 0 else 0.0

    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0

        downside_deviation = np.std(downside_returns)
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)

    def compare_models(self, models: List[Tuple[Any, str]],
                      X_test: np.ndarray, y_test: np.ndarray,
                      market_data: Optional[pd.DataFrame] = None,
                      regime_labels: Optional[np.ndarray] = None) -> List[EvaluationResult]:
        """Compare multiple models."""
        results = []

        for model, name in models:
            result = self.evaluate_model(
                model=model,
                X_test=X_test,
                y_test=y_test,
                market_data=market_data,
                regime_labels=regime_labels,
                model_name=name
            )
            results.append(result)

        # Sort by overall performance score
        results.sort(key=lambda x: self._calculate_overall_score(x), reverse=True)

        return results

    def _calculate_overall_score(self, result: EvaluationResult) -> float:
        """Calculate overall performance score."""
        score = 0.0
        weight = 0.0

        # Basic metrics weight
        if result.basic_metrics:
            basic_score = np.mean(list(result.basic_metrics.values()))
            score += basic_score * 0.3
            weight += 0.3

        # Trading metrics weight
        if result.trading_metrics:
            trading_score = np.mean(list(result.trading_metrics.values()))
            score += trading_score * 0.3
            weight += 0.3

        # Economic metrics weight
        if result.economic_metrics:
            economic_score = np.mean(list(result.economic_metrics.values()))
            score += economic_score * 0.2
            weight += 0.2

        # Risk metrics weight (lower is better for some metrics)
        if result.risk_metrics:
            risk_metrics = result.risk_metrics
            risk_score = 1.0 - abs(risk_metrics.get(EvaluationMetric.MAX_DRAWDOWN, 0))
            score += risk_score * 0.1
            weight += 0.1

        # Regime metrics weight
        if result.regime_metrics:
            regime_score = np.mean(list(result.regime_metrics.values()))
            score += regime_score * 0.1
            weight += 0.1

        return score / weight if weight > 0 else 0.0

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
        if not self.evaluation_history:
            return {'error': 'No evaluation history available'}

        summary = {
            'total_evaluations': len(self.evaluation_history),
            'architecture_type': self.architecture_type.value,
            'evaluation_config': self.config.__dict__,
            'best_performing_model': None,
            'average_performance': {},
            'performance_trends': {},
            'validation_pass_rates': {}
        }

        # Find best performing model
        best_result = max(self.evaluation_history, key=lambda x: self._calculate_overall_score(x))
        summary['best_performing_model'] = {
            'name': best_result.model_name,
            'overall_score': self._calculate_overall_score(best_result),
            'evaluation_time': best_result.training_time
        }

        # Calculate average performance
        all_basic_metrics = defaultdict(list)
        all_trading_metrics = defaultdict(list)
        all_economic_metrics = defaultdict(list)

        for result in self.evaluation_history:
            for metric, value in result.basic_metrics.items():
                all_basic_metrics[metric].append(value)
            for metric, value in result.trading_metrics.items():
                all_trading_metrics[metric].append(value)
            for metric, value in result.economic_metrics.items():
                all_economic_metrics[metric].append(value)

        # Average metrics
        for metric, values in all_basic_metrics.items():
            summary['average_performance'][f'basic_{metric.value}'] = np.mean(values)

        for metric, values in all_trading_metrics.items():
            summary['average_performance'][f'trading_{metric.value}'] = np.mean(values)

        for metric, values in all_economic_metrics.items():
            summary['average_performance'][f'economic_{metric.value}'] = np.mean(values)

        # Validation pass rates
        summary['validation_pass_rates'] = {
            'economic_significance': sum(1 for r in self.evaluation_history if r.passed_economic_significance) / len(self.evaluation_history),
            'trading_viability': sum(1 for r in self.evaluation_history if r.passed_trading_viability) / len(self.evaluation_history),
            'risk_limits': sum(1 for r in self.evaluation_history if r.passed_risk_limits) / len(self.evaluation_history),
            'regime_stability': sum(1 for r in self.evaluation_history if r.passed_regime_stability) / len(self.evaluation_history)
        }

        return summary

# Convenience functions
def create_evaluation_framework(architecture_type: ArchitectureType,
                              evaluation_type: EvaluationType = EvaluationType.COMPREHENSIVE,
                              **kwargs) -> UnifiedEvaluationFramework:
    """Create an evaluation framework with default settings."""
    config = EvaluationConfig(evaluation_type=evaluation_type, **kwargs)
    return UnifiedEvaluationFramework(architecture_type=architecture_type, config=config)

def create_basic_evaluator(architecture_type: ArchitectureType) -> UnifiedEvaluationFramework:
    """Create a basic evaluation framework."""
    config = EvaluationConfig(
        evaluation_type=EvaluationType.BASIC,
        enable_trading_metrics=False,
        enable_economic_metrics=False,
        enable_risk_metrics=False,
        enable_regime_metrics=False,
        enable_cross_validation=False,
        enable_bootstrap=False
    )
    return UnifiedEvaluationFramework(architecture_type=architecture_type, config=config)

def create_trading_evaluator(architecture_type: ArchitectureType) -> UnifiedEvaluationFramework:
    """Create a trading-focused evaluation framework."""
    config = EvaluationConfig(
        evaluation_type=EvaluationType.TRADING,
        enable_trading_metrics=True,
        enable_economic_metrics=True,
        enable_risk_metrics=True,
        enable_regime_metrics=True
    )
    return UnifiedEvaluationFramework(architecture_type=architecture_type, config=config)
