"""
Advanced TAS Evaluator

This module provides comprehensive evaluation capabilities for TAS including:
- Economic significance validation
- Trading viability assessment
- Multi-objective optimization evaluation
- Uncertainty quantification
- Risk-adjusted performance metrics
- Regime-specific validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import torch

# Import tprint for comprehensive logging
try:
    from src.utils.tprint import (
        tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
        tprint_success, tprint_progress, tprint_performance, tprint_timer
    )
    TPRINT_AVAILABLE = True
except ImportError:
    # Fallback function if tprint is not available
    def tprint(message: str, color: str = "white", **kwargs):
        print(f"[TAS_EVALUATOR] {message}")
    def tprint_debug(message: str, **kwargs):
        print(f"[DEBUG] {message}")
    def tprint_info(message: str, **kwargs):
        print(f"[INFO] {message}")
    def tprint_warning(message: str, **kwargs):
        print(f"[WARNING] {message}")
    def tprint_error(message: str, **kwargs):
        print(f"[ERROR] {message}")
    def tprint_success(message: str, **kwargs):
        print(f"[SUCCESS] {message}")
    def tprint_progress(message: str, **kwargs):
        print(f"[PROGRESS] {message}")
    def tprint_performance(message: str, **kwargs):
        print(f"[PERFORMANCE] {message}")
    def tprint_timer(message: str, **kwargs):
        print(f"[TIMER] {message}")
    TPRINT_AVAILABLE = False

from ..core.tas_config import TASConfig, TradingObjective, MarketRegime

# Import unified utilities
from ...hybrid_nas_tas_regime.shared_utils import (
    UnifiedEconomicSignificanceEvaluator, EconomicEvaluationConfig,
    UnifiedTradingViabilityEvaluator, TradingViabilityConfig,
    UnifiedMultiObjectiveOptimizer, OptimizationConfig,
    UnifiedHardwareOptimizer, HardwareConfig,
    UnifiedRegimeAnalyzer, RegimeAnalysisConfig,
    UnifiedValidationSystem, ValidationConfig,
    create_unified_economic_evaluator, quick_economic_evaluation,
    create_unified_trading_viability_evaluator, quick_trading_viability_evaluation,
    create_unified_multi_objective_optimizer, quick_multi_objective_optimization,
    create_unified_hardware_optimizer, quick_hardware_optimization,
    create_unified_regime_analyzer, quick_regime_analysis,
    create_unified_validation_system, quick_validation
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of comprehensive model evaluation."""
    model_name: str
    architecture_type: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Core metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Trading-specific metrics
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration: float = 0.0

    # Risk metrics
    volatility: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    cvar_95: float = 0.0  # Conditional VaR 95%
    calmar_ratio: float = 0.0  # Return / Max Drawdown

    # Regime-specific metrics
    regime_stability_score: float = 0.0
    adaptation_speed: float = 0.0
    regime_transition_accuracy: float = 0.0

    # Economic significance
    economic_significance_score: float = 0.0
    trading_viability_score: float = 0.0
    market_regime_fit: float = 0.0

    # Model characteristics
    model_complexity: float = 0.0
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage_mb: float = 0.0

    # Uncertainty quantification
    accuracy_std: float = 0.0
    sharpe_ratio_std: float = 0.0
    prediction_confidence: float = 0.0

    # Validation flags
    passed_economic_significance: bool = False
    passed_trading_viability: bool = False
    passed_risk_limits: bool = False
    is_market_regime_aware: bool = False

    # Detailed breakdowns
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    error_analysis: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    evaluation_config: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


class TASEvaluator:
    """Advanced evaluator for Trading Architecture Search."""

    def __init__(self, config: TASConfig):
        """Initialize TAS evaluator.

        Args:
            config: TAS configuration
        """
        tprint("📊 Initializing TAS Evaluator", color="blue")
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        tprint(f"📊 Config: economic_threshold={config.economic_significance_threshold}, trading_threshold={config.trading_viability_threshold}", color="cyan")

        # Evaluation parameters
        tprint("⚙️ Setting evaluation parameters", color="yellow")
        self.min_trades_for_evaluation = 50
        self.confidence_level = 0.95
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        tprint(f"📊 Min trades: {self.min_trades_for_evaluation}, confidence: {self.confidence_level}, risk-free rate: {self.risk_free_rate}", color="cyan")

        # Validation thresholds
        tprint("🎯 Setting validation thresholds", color="yellow")
        self.economic_significance_threshold = config.economic_significance_threshold
        self.trading_viability_threshold = config.trading_viability_threshold
        self.max_drawdown_threshold = config.max_drawdown_threshold
        tprint(f"🎯 Economic: {self.economic_significance_threshold}, Trading: {self.trading_viability_threshold}, Max DD: {self.max_drawdown_threshold}", color="cyan")

        # Initialize unified utilities
        tprint("🔧 Initializing unified utilities", color="yellow")
        self._initialize_unified_utilities()
        tprint("✅ TAS Evaluator initialization complete", color="green")

    def _initialize_unified_utilities(self):
        """Initialize unified utilities for TAS evaluation."""
        tprint("🔧 Starting unified utilities initialization", color="yellow")
        try:
            # Initialize unified economic significance evaluator
            tprint("💰 Creating economic significance evaluator", color="yellow")
            economic_config = EconomicEvaluationConfig(
                significance_threshold=self.economic_significance_threshold,
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
            self.unified_economic_evaluator = create_unified_economic_evaluator(economic_config)
            tprint("✅ Economic significance evaluator created", color="green")
            
            # Initialize unified trading viability evaluator
            tprint("📈 Creating trading viability evaluator", color="yellow")
            trading_config = TradingViabilityConfig(
                viability_threshold=self.trading_viability_threshold,
                min_trading_frequency=0.1,
                max_trading_frequency=10.0,
                min_position_duration=5.0,
                max_position_duration=1440.0,
                min_model_confidence=0.6,
                min_risk_adjusted_return=0.1
            )
            self.unified_trading_evaluator = create_unified_trading_viability_evaluator(trading_config)
            tprint("✅ Trading viability evaluator created", color="green")
            
            # Initialize unified multi-objective optimizer
            tprint("🎯 Creating multi-objective optimizer", color="yellow")
            optimization_config = OptimizationConfig(
                objectives=['regime_accuracy', 'economic_significance', 'trading_viability', 'computational_efficiency'],
                objective_weights={
                    'regime_accuracy': 0.3,
                    'economic_significance': 0.25,
                    'trading_viability': 0.25,
                    'computational_efficiency': 0.2
                },
                max_iterations=100,
                population_size=50
            )
            self.unified_optimizer = create_unified_multi_objective_optimizer(optimization_config)
            tprint("✅ Multi-objective optimizer created", color="green")
            
            # Initialize unified hardware optimizer
            tprint("💻 Creating hardware optimizer", color="yellow")
            hardware_config = HardwareConfig(
                enable_hardware_optimization=True,
                max_memory_usage_gb=8.0,
                enable_gpu_acceleration=True,
                enable_performance_monitoring=True
            )
            self.unified_hardware_optimizer = create_unified_hardware_optimizer(hardware_config)
            tprint("✅ Hardware optimizer created", color="green")
            
            # Initialize unified regime analyzer
            tprint("📊 Creating regime analyzer", color="yellow")
            regime_config = RegimeAnalysisConfig(
                analysis_types=['stability', 'transitions', 'uncertainty'],
                stability_window=20,
                transition_window=10,
                uncertainty_method='entropy'
            )
            self.unified_regime_analyzer = create_unified_regime_analyzer(regime_config)
            tprint("✅ Regime analyzer created", color="green")
            
            # Initialize unified validation system
            tprint("✅ Creating validation system", color="yellow")
            validation_config = ValidationConfig(
                validation_type='time_series_validation',
                n_folds=5,
                test_size=0.2,
                metrics=['accuracy', 'precision', 'recall', 'f1_score'],
                enable_trading_metrics=True,
                enable_regime_metrics=True
            )
            self.unified_validator = create_unified_validation_system(validation_config)
            tprint("✅ Validation system created", color="green")
            
            self.logger.info("✅ Unified utilities initialized for TAS evaluation")
            tprint("✅ All unified utilities initialized successfully", color="green")
            
        except Exception as e:
            tprint(f"⚠️ Failed to initialize unified utilities: {e}", color="red")
            self.logger.warning(f"Failed to initialize unified utilities: {e}")
            # Set fallback evaluators
            self.unified_economic_evaluator = None
            self.unified_trading_evaluator = None
            self.unified_optimizer = None
            self.unified_hardware_optimizer = None
            self.unified_regime_analyzer = None
            self.unified_validator = None

    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      market_data: pd.DataFrame, regime_labels: Optional[pd.Series] = None,
                      model_name: str = "unknown", architecture_type: str = "unknown") -> EvaluationResult:
        """Comprehensive evaluation of a trading model.

        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test labels
            market_data: Market data for trading simulation
            regime_labels: Regime labels for regime-specific evaluation
            model_name: Name of the model
            architecture_type: Type of architecture

        Returns:
            Comprehensive evaluation result
        """
        self.logger.info(f"🔬 Starting comprehensive evaluation of {model_name}...")

        result = EvaluationResult(
            model_name=model_name,
            architecture_type=architecture_type,
            evaluation_config=self.config.get_validation_config()
        )

        try:
            # Basic model evaluation
            self._evaluate_basic_metrics(model, X_test, y_test, result)

            # Trading performance evaluation
            self._evaluate_trading_performance(model, X_test, y_test, market_data, result)

            # Risk analysis
            self._evaluate_risk_metrics(result)

            # Regime-specific evaluation
            if regime_labels is not None:
                self._evaluate_regime_performance(model, X_test, y_test, regime_labels, result)

            # Economic significance assessment
            self._evaluate_economic_significance(result)

            # Trading viability assessment
            self._evaluate_trading_viability(result)

            # Model characteristics
            self._evaluate_model_characteristics(model, X_test, result)

            # Uncertainty quantification
            self._evaluate_uncertainty(model, X_test, y_test, result)

            # Validation checks
            self._perform_validation_checks(result)

            self.logger.info(f"✅ Evaluation completed for {model_name}")
            self.logger.info(f"   Sharpe Ratio: {result.sharpe_ratio:.3f}")
            self.logger.info(f"   Max Drawdown: {result.max_drawdown:.3f}")
            self.logger.info(f"   Economic Significance: {result.economic_significance_score:.3f}")

            return result

        except Exception as e:
            self.logger.error(f"Evaluation failed for {model_name}: {e}")
            result.notes = f"Evaluation failed: {str(e)}"
            return result

    def _evaluate_basic_metrics(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, result: EvaluationResult):
        """Evaluate basic model performance metrics."""
        try:
            # Get predictions
            predictions = model.predict(X_test)

            # Handle different prediction formats
            if hasattr(predictions, 'shape') and len(predictions.shape) > 1:
                # Multi-class or probabilistic predictions
                pred_classes = np.argmax(predictions, axis=1)
            else:
                pred_classes = (predictions > 0.5).astype(int) if len(np.unique(y_test)) == 2 else predictions.round()

            # Calculate metrics
            result.accuracy = accuracy_score(y_test, pred_classes)
            result.precision = precision_score(y_test, pred_classes, average='weighted', zero_division=0)
            result.recall = recall_score(y_test, pred_classes, average='weighted', zero_division=0)
            result.f1_score = f1_score(y_test, pred_classes, average='weighted', zero_division=0)

            # Prediction confidence (if available)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)
                result.prediction_confidence = np.mean(np.max(proba, axis=1))
            elif hasattr(model, 'decision_function'):
                decision_scores = model.decision_function(X_test)
                result.prediction_confidence = np.mean(np.abs(decision_scores))

        except Exception as e:
            self.logger.warning(f"Basic metrics evaluation failed: {e}")

    def _evaluate_trading_performance(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                                    market_data: pd.DataFrame, result: EvaluationResult):
        """Evaluate trading-specific performance metrics."""
        try:
            # Generate trading signals
            predictions = model.predict(X_test)

            # Convert predictions to trading signals
            if hasattr(predictions, 'shape') and len(predictions.shape) > 1:
                trading_signals = np.argmax(predictions, axis=1) - 1  # Convert to -1, 0, 1
            else:
                trading_signals = (predictions > 0.5).astype(int) * 2 - 1  # Convert to -1, 1

            # Simulate trading
            trading_results = self._simulate_trading(market_data, trading_signals, result)

            # Calculate performance metrics
            if len(trading_results['returns']) > 0:
                returns = np.array(trading_results['returns'])

                result.total_return = np.prod(1 + returns) - 1
                result.sharpe_ratio = self._calculate_sharpe_ratio(returns)
                result.max_drawdown = self._calculate_max_drawdown(np.cumprod(1 + returns))
                result.win_rate = np.mean(returns > 0)
                result.volatility = np.std(returns)
                result.profit_factor = abs(np.sum(returns[returns > 0]) / np.sum(np.abs(returns[returns < 0])))

                # Average trade duration
                if 'trade_durations' in trading_results:
                    result.avg_trade_duration = np.mean(trading_results['trade_durations'])

        except Exception as e:
            self.logger.warning(f"Trading performance evaluation failed: {e}")

    def _simulate_trading(self, market_data: pd.DataFrame, signals: np.ndarray, result: EvaluationResult) -> Dict[str, Any]:
        """Simulate trading based on model signals."""
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
                current_price = row.close

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

            # Close any remaining position at the end
            if position != 0 and trade_start is not None:
                final_price = market_data['close'].iloc[-1]
                trade_return = (final_price - entry_price) / entry_price * position

                trading_results['returns'].append(trade_return)
                trading_results['trades'].append({
                    'entry_price': entry_price,
                    'exit_price': final_price,
                    'return': trade_return,
                    'duration': len(signals) - trade_start,
                    'direction': position
                })
                trading_results['trade_durations'].append(len(signals) - trade_start)

        except Exception as e:
            self.logger.warning(f"Trading simulation failed: {e}")

        return trading_results

    def _evaluate_risk_metrics(self, result: EvaluationResult):
        """Calculate comprehensive risk metrics."""
        try:
            returns = result.total_return  # This should be a time series, but we use simplified version

            # Value at Risk (95%)
            if isinstance(returns, (list, np.ndarray)) and len(returns) > 10:
                result.var_95 = np.percentile(returns, 5)

                # Conditional VaR (expected loss given VaR exceeded)
                worst_returns = returns[returns <= result.var_95]
                result.cvar_95 = np.mean(worst_returns) if len(worst_returns) > 0 else 0

            # Calmar Ratio (annual return / max drawdown)
            if result.max_drawdown > 0:
                annual_return = result.total_return  # Simplified
                result.calmar_ratio = annual_return / abs(result.max_drawdown)

        except Exception as e:
            self.logger.warning(f"Risk metrics evaluation failed: {e}")

    def _evaluate_regime_performance(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                                   regime_labels: pd.Series, result: EvaluationResult):
        """Evaluate performance across different market regimes."""
        try:
            predictions = model.predict(X_test)

            if hasattr(predictions, 'shape') and len(predictions.shape) > 1:
                pred_classes = np.argmax(predictions, axis=1)
            else:
                pred_classes = predictions.round()

            # Group by regime
            unique_regimes = regime_labels.unique()
            regime_performance = {}

            for regime in unique_regimes:
                regime_mask = regime_labels == regime
                if np.sum(regime_mask) > 10:  # Minimum samples for evaluation
                    regime_y_true = y_test[regime_mask]
                    regime_y_pred = pred_classes[regime_mask]

                    regime_accuracy = accuracy_score(regime_y_true, regime_y_pred)
                    regime_precision = precision_score(regime_y_true, regime_y_pred, average='weighted', zero_division=0)
                    regime_recall = recall_score(regime_y_true, regime_y_pred, average='weighted', zero_division=0)

                    regime_performance[regime] = {
                        'accuracy': regime_accuracy,
                        'precision': regime_precision,
                        'recall': regime_recall,
                        'sample_count': np.sum(regime_mask)
                    }

            result.regime_performance = regime_performance
            result.regime_stability_score = np.mean([perf['accuracy'] for perf in regime_performance.values()])
            result.is_market_regime_aware = True

        except Exception as e:
            self.logger.warning(f"Regime performance evaluation failed: {e}")

    def _evaluate_economic_significance(self, result: EvaluationResult):
        """Evaluate economic significance of the trading strategy."""
        try:
            # Multi-factor economic significance score
            sharpe_factor = min(1.0, result.sharpe_ratio / 2.0)  # Normalize Sharpe ratio
            drawdown_factor = max(0.0, 1 - abs(result.max_drawdown) / 0.3)  # Penalty for large drawdowns
            win_rate_factor = result.win_rate
            profit_factor = min(1.0, result.profit_factor / 2.0)  # Normalize profit factor

            # Regime stability bonus
            regime_bonus = result.regime_stability_score * 0.1

            # Calculate overall economic significance score
            result.economic_significance_score = (
                sharpe_factor * 0.4 +
                drawdown_factor * 0.3 +
                win_rate_factor * 0.2 +
                profit_factor * 0.1 +
                regime_bonus
            )

            # Set validation flag
            result.passed_economic_significance = result.economic_significance_score >= self.economic_significance_threshold

        except Exception as e:
            self.logger.warning(f"Economic significance evaluation failed: {e}")

    def _evaluate_trading_viability(self, result: EvaluationResult):
        """Evaluate trading viability considering practical constraints."""
        try:
            # Check trading frequency (should not be too high)
            trades_per_day = result.total_return / max(result.avg_trade_duration, 1) * 24 * 60 / 5  # Assuming 5-min bars
            frequency_factor = min(1.0, 50 / max(trades_per_day, 1))  # Prefer reasonable trading frequency

            # Check minimum trade duration
            duration_factor = min(1.0, result.avg_trade_duration / 30)  # Prefer trades lasting at least 30 minutes

            # Check model confidence
            confidence_factor = result.prediction_confidence

            # Check risk-adjusted returns
            risk_adjusted_factor = min(1.0, result.sharpe_ratio * result.calmar_ratio)

            # Calculate trading viability score
            result.trading_viability_score = (
                frequency_factor * 0.3 +
                duration_factor * 0.2 +
                confidence_factor * 0.3 +
                risk_adjusted_factor * 0.2
            )

            # Set validation flag
            result.passed_trading_viability = result.trading_viability_score >= self.trading_viability_threshold

        except Exception as e:
            self.logger.warning(f"Trading viability evaluation failed: {e}")

    def _evaluate_model_characteristics(self, model: Any, X_test: np.ndarray, result: EvaluationResult):
        """Evaluate model characteristics and performance."""
        try:
            # Model complexity (parameter count)
            if hasattr(model, 'get_n_params'):
                result.model_complexity = model.get_n_params()
            elif hasattr(model, 'n_features_in_'):
                result.model_complexity = model.n_features_in_ * 10  # Rough estimate
            else:
                result.model_complexity = 100  # Default estimate

            # Inference time
            import time
            start_time = time.time()
            _ = model.predict(X_test[:100])  # Small batch for timing
            result.inference_time = (time.time() - start_time) / 100

            # Memory usage (rough estimate)
            if hasattr(model, 'get_n_params'):
                result.memory_usage_mb = model.get_n_params() * 4 / (1024 * 1024)  # 4 bytes per parameter
            else:
                result.memory_usage_mb = 10  # Default estimate

        except Exception as e:
            self.logger.warning(f"Model characteristics evaluation failed: {e}")

    def _evaluate_uncertainty(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, result: EvaluationResult):
        """Evaluate model uncertainty using bootstrapping."""
        try:
            n_bootstrap = 10  # Reduced for speed
            bootstrap_scores = []

            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(len(X_test), size=len(X_test), replace=True)
                X_boot = X_test[indices]
                y_boot = y_test[indices]

                # Get predictions
                pred_boot = model.predict(X_boot)
                if hasattr(pred_boot, 'shape') and len(pred_boot.shape) > 1:
                    pred_classes = np.argmax(pred_boot, axis=1)
                else:
                    pred_classes = pred_boot.round()

                # Calculate accuracy
                accuracy_boot = accuracy_score(y_boot, pred_classes)
                bootstrap_scores.append(accuracy_boot)

            if bootstrap_scores:
                result.accuracy_std = np.std(bootstrap_scores)

                # Confidence interval
                ci_lower = np.percentile(bootstrap_scores, 2.5)
                ci_upper = np.percentile(bootstrap_scores, 97.5)

                result.prediction_confidence = (result.accuracy - ci_lower) / (ci_upper - ci_lower)

        except Exception as e:
            self.logger.warning(f"Uncertainty evaluation failed: {e}")

    def _perform_validation_checks(self, result: EvaluationResult):
        """Perform final validation checks."""
        try:
            # Risk limits check
            result.passed_risk_limits = (
                abs(result.max_drawdown) <= self.max_drawdown_threshold and
                result.sharpe_ratio >= self.config.risk_adjusted_return_threshold
            )

            # Overall validation
            validation_score = (
                result.economic_significance_score * 0.4 +
                result.trading_viability_score * 0.4 +
                (1 if result.passed_risk_limits else 0) * 0.2
            )

            result.notes = f"Validation score: {validation_score:.3f}"

        except Exception as e:
            self.logger.warning(f"Validation checks failed: {e}")

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio from returns series."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - self.risk_free_rate / 252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized

    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative returns."""
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


# Convenience functions for evaluation
def evaluate_tas_model(model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      market_data: pd.DataFrame, config: TASConfig,
                      model_name: str = "TAS_Model") -> EvaluationResult:
    """Convenience function for TAS model evaluation."""
    evaluator = TASEvaluator(config)
    return evaluator.evaluate_model(model, X_test, y_test, market_data, model_name=model_name)


def compare_tas_models(models: List[Tuple[Any, str]], X_test: np.ndarray, y_test: np.ndarray,
                      market_data: pd.DataFrame, config: TASConfig) -> List[EvaluationResult]:
    """Compare multiple TAS models and return evaluation results."""
    evaluator = TASEvaluator(config)
    results = []

    for model, name in models:
        result = evaluator.evaluate_model(model, X_test, y_test, market_data, model_name=name)
        results.append(result)

    # Sort by economic significance
    results.sort(key=lambda x: x.economic_significance_score, reverse=True)
    return results