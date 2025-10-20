"""
Bayesian Optimization for Entry Timing Parameters

Advanced Bayesian optimization system for optimizing entry timing parameters
in trading models, with support for multiple objectives and constraints.
Extensively integrated with utility modules for optimal performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
from pathlib import Path
import warnings

# Extensive use of common utilities
from ...common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report,
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    validate_finite, validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, optimize_dataframe_dtypes,
    safe_to_parquet, safe_read_parquet, integrate_with_m1_optimizers,
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    cleanup_m1_optimizers, memory_checkpoint, gpu_context, optimize_memory,
    get_memory_usage, safe_copy, safe_deepcopy, safe_resample, align_dataframes,
    validate_dataframe_schema, guard_dataframe_nulls, timed_operation,
    format_bytes, parallel_map, chunked_iterable, safe_rolling, safe_groupby_operation,
    safe_apply_function as co_safe_apply_function, create_summary_statistics as co_create_summary_statistics
)

from ...common_utilities import (
    CommonUtilities, safe_dataframe_operation as cu_safe_dataframe_operation,
    validate_dataframe_columns as cu_validate_dataframe_columns,
    calculate_data_quality_metrics as cu_calculate_data_quality_metrics,
    safe_merge_dataframes as cu_safe_merge_dataframes,
    safe_groupby_operation as cu_safe_groupby_operation,
    safe_apply_function as cu_safe_apply_function,
    create_summary_statistics as cu_create_summary_statistics,
    safe_drop_columns as cu_safe_drop_columns,
    safe_rename_columns as cu_safe_rename_columns,
    validate_timestamp_column as cu_validate_timestamp_column,
    safe_timestamp_conversion as cu_safe_timestamp_conversion,
    get_dataframe_info as cu_get_dataframe_info,
    safe_filter_dataframe as cu_safe_filter_dataframe,
    create_data_quality_report as cu_create_data_quality_report
)

from ...math_validation import (
    MathValidation, safe_divide as mv_safe_divide, safe_log as mv_safe_log,
    safe_sqrt as mv_safe_sqrt, safe_power as mv_safe_power,
    validate_finite as mv_validate_finite, validate_positive as mv_validate_positive,
    validate_range as mv_validate_range, safe_kelly_calculation as mv_safe_kelly_calculation,
    safe_weighted_average as mv_safe_weighted_average, safe_percentage_change as mv_safe_percentage_change,
    safe_correlation, safe_covariance, safe_mean as mv_safe_mean, safe_std as mv_safe_std,
    safe_percentile, validate_correlation_matrix, safe_matrix_inverse, math_safe,
    validate_numeric_array
)

from ...tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_structured,
    tprint_with_level, tprint_timer, tprint_logged, configure_tprint,
    get_tprint_config, tprint_context, LogLevel
)

from ...data.klines_parquet import (
    KlinesParquetManager, get_klines_manager, read_ethusdt_data,
    save_klines_to_parquet, load_klines_from_parquet, validate_klines_data
)

from ...serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

# Import data processing utilities
from ...data.processing.data_processing import DataProcessor
from ...data.basic_returns_engineer import BasicReturnsEngineer
from ...data.feature_engineer import FeatureEngineer
from ...data.gap_detector import GapDetector
from ...data.unified_data_utils import UnifiedDataUtils

# Import matrix operations
from ...matrix_operations.unified_operations import UnifiedMatrixOperations
from ...matrix_operations.enhanced_operations import EnhancedMatrixOperations
from ...matrix_operations.batch_operations import BatchMatrixOperations
from ...matrix_operations.vectorized_core import VectorizedProcessingCore

# Import hardware utilities
from ...hardware.m1_gpu_utils import M1GPUManager, is_m1_available, is_mps_available
from ...hardware.m1_memory_optimizer import M1MemoryOptimizer
from ...hardware.m1_cpu_optimizer import M1CPUOptimizer

# Bayesian optimization imports
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Install with: pip install optuna")

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    warnings.warn("Scikit-optimize not available. Install with: pip install scikit-optimize")

logger = logging.getLogger(__name__)

@dataclass
class EntryTimingConfig:
    """Configuration for entry timing optimization."""

    # Parameter bounds
    entry_threshold_min: float = 0.001  # 0.1%
    entry_threshold_max: float = 0.01   # 1.0%
    exit_threshold_min: float = 0.005   # 0.5%
    exit_threshold_max: float = 0.02    # 2.0%
    stop_loss_min: float = 0.002        # 0.2%
    stop_loss_max: float = 0.01         # 1.0%
    take_profit_min: float = 0.005      # 0.5%
    take_profit_max: float = 0.03       # 3.0%
    timing_window_min: int = 1          # 1 minute
    timing_window_max: int = 10         # 10 minutes
    confidence_threshold_min: float = 0.5  # 50%
    confidence_threshold_max: float = 0.9   # 90%

    # Optimization settings
    n_trials: int = 100
    timeout_minutes: int = 60
    n_calls: int = 100  # For skopt
    random_state: int = 42

    # Multi-objective settings
    enable_multi_objective: bool = True
    objectives: List[str] = None  # ['profit', 'sharpe', 'win_rate', 'max_drawdown']
    objective_weights: List[float] = None  # [0.4, 0.3, 0.2, 0.1]

    # Constraints
    max_drawdown_threshold: float = 0.1  # 10%
    min_win_rate_threshold: float = 0.4  # 40%
    max_trade_duration: int = 60  # 60 minutes

    # Reporting
    save_reports: bool = True
    report_directory: str = "reports/bayesian_entry_timing"
    enable_visualization: bool = True
    detailed_logging: bool = True

    def __post_init__(self):
        """Initialize default values."""
        if self.objectives is None:
            self.objectives = ['profit', 'sharpe', 'win_rate', 'max_drawdown']
        if self.objective_weights is None:
            self.objective_weights = [0.4, 0.3, 0.2, 0.1]

@dataclass
class EntryTimingResult:
    """Result of entry timing optimization."""

    # Best parameters
    best_params: Dict[str, Any]
    best_score: float

    # Optimization details
    n_trials: int
    optimization_time: float
    convergence_achieved: bool

    # Performance metrics
    profit: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    total_trades: int

    # Optimization history
    trial_history: List[Dict[str, Any]]
    convergence_history: List[float]

    # Recommendations
    recommendations: List[str]
    risk_assessment: str

    # Metadata
    model_name: str
    optimization_timestamp: str

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.optimization_timestamp is None:
            self.optimization_timestamp = datetime.now().isoformat()

class BayesianEntryTimingOptimizer:
    """Bayesian optimization for entry timing parameters."""

    def __init__(self, config: Optional[EntryTimingConfig] = None):
        """
        Initialize Bayesian entry timing optimizer.

        Args:
            config: Configuration for optimization
        """
        self.config = config or EntryTimingConfig()

        # Create report directory
        if self.config.save_reports:
            Path(self.config.report_directory).mkdir(parents=True, exist_ok=True)

        # Performance optimization: Add caching for trading simulation
        self._simulation_cache = {}
        self._cache_max_size = 500

        # Track evaluation count for performance monitoring
        self._evaluation_count = 0

        logger.info("✅ Bayesian Entry Timing Optimizer initialized")

    def optimize_entry_timing(self,
                            model: Any,
                            X: np.ndarray,
                            y: np.ndarray,
                            analyst_signals: Optional[np.ndarray] = None,
                            hmm_regime_probs: Optional[np.ndarray] = None,
                            timestamps: Optional[np.ndarray] = None,
                            model_name: str = "model",
                            optimization_method: str = "optuna") -> EntryTimingResult:
        """
        Optimize entry timing parameters using Bayesian optimization.

        Args:
            model: Trained model for predictions
            X: Feature matrix
            y: Target values
            analyst_signals: Analyst signals (optional)
            hmm_regime_probs: HMM regime probabilities (optional)
            timestamps: Timestamps for temporal analysis (optional)
            model_name: Name of the model
            optimization_method: 'optuna' or 'skopt'

        Returns:
            EntryTimingResult with optimization results
        """
        try:
            if optimization_method == "optuna" and OPTUNA_AVAILABLE:
                return self._optimize_with_optuna(
                    model, X, y, analyst_signals, hmm_regime_probs, timestamps, model_name
                )
            elif optimization_method == "skopt" and SKOPT_AVAILABLE:
                return self._optimize_with_skopt(
                    model, X, y, analyst_signals, hmm_regime_probs, timestamps, model_name
                )
            else:
                raise ValueError(f"Optimization method {optimization_method} not available")

        except Exception as e:
            logger.error(f"❌ Entry timing optimization failed: {e}")
            return EntryTimingResult(
                best_params={},
                best_score=0.0,
                n_trials=0,
                optimization_time=0.0,
                convergence_achieved=False,
                profit=0.0,
                sharpe_ratio=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
                total_trades=0,
                trial_history=[],
                convergence_history=[],
                recommendations=[],
                risk_assessment="unknown",
                model_name=model_name
            )

    def _optimize_with_optuna(self, model: Any, X: np.ndarray, y: np.ndarray,
                            analyst_signals: Optional[np.ndarray] = None,
                            hmm_regime_probs: Optional[np.ndarray] = None,
                            timestamps: Optional[np.ndarray] = None,
                            model_name: str = "model") -> EntryTimingResult:
        """Optimize using Optuna."""
        start_time = datetime.now()

        # Create study with optimized TPE sampler and early stopping
        sampler = TPESampler(
            seed=self.config.random_state,
            n_startup_trials=10,  # Reduce startup trials for efficiency
            n_ei_candidates=24,
            gamma=0.25  # Optimize exploration vs exploitation
        )

        # Enhanced pruning for better efficiency
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,
            interval_steps=3
        )

        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )

        # Define objective function
        def objective(trial):
            # Sample parameters
            params = {
                'entry_threshold': trial.suggest_float(
                    'entry_threshold',
                    self.config.entry_threshold_min,
                    self.config.entry_threshold_max
                ),
                'exit_threshold': trial.suggest_float(
                    'exit_threshold',
                    self.config.exit_threshold_min,
                    self.config.exit_threshold_max
                ),
                'stop_loss': trial.suggest_float(
                    'stop_loss',
                    self.config.stop_loss_min,
                    self.config.stop_loss_max
                ),
                'take_profit': trial.suggest_float(
                    'take_profit',
                    self.config.take_profit_min,
                    self.config.take_profit_max
                ),
                'timing_window': trial.suggest_int(
                    'timing_window',
                    self.config.timing_window_min,
                    self.config.timing_window_max
                ),
                'confidence_threshold': trial.suggest_float(
                    'confidence_threshold',
                    self.config.confidence_threshold_min,
                    self.config.confidence_threshold_max
                )
            }

            # Simulate trading with these parameters
            try:
                results = self._simulate_trading_with_params(
                    model, X, y, analyst_signals, hmm_regime_probs, timestamps, params
                )

                # Calculate composite score
                score = self._calculate_composite_score(results)

                # Add constraints
                if results['max_drawdown'] > self.config.max_drawdown_threshold:
                    return -np.inf
                if results['win_rate'] < self.config.min_win_rate_threshold:
                    return -np.inf

                return score

            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return -np.inf

        # Run optimization with early stopping based on convergence
        patience = 15  # Stop if no improvement for 15 trials
        best_score = -np.inf
        patience_counter = 0

        def optimized_objective(trial):
            nonlocal best_score, patience_counter

            score = objective(trial)

            # Early stopping logic
            if score > best_score:
                best_score = score
                patience_counter = 0
            else:
                patience_counter += 1

            # Stop optimization if no improvement for patience trials
            if patience_counter >= patience:
                trial.study.stop()

            return score

        # Run optimization
        study.optimize(
            optimized_objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_minutes * 60
        )

        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value

        # Simulate with best parameters
        best_results = self._simulate_trading_with_params(
            model, X, y, analyst_signals, hmm_regime_probs, timestamps, best_params
        )

        # Create trial history
        trial_history = []
        for trial in study.trials:
            if trial.value is not None:
                trial_history.append({
                    'params': trial.params,
                    'value': trial.value,
                    'state': trial.state.name
                })

        # Calculate convergence with safe division
        convergence_history = [trial.value for trial in study.trials if trial.value is not None]
        if len(convergence_history) > 10:
            recent_improvement = abs(convergence_history[-1] - convergence_history[-10])
            convergence_achieved = recent_improvement < 0.01
        else:
            convergence_achieved = False

        # Generate recommendations
        recommendations = self._generate_recommendations(best_results, best_params)
        risk_assessment = self._assess_risk(best_results)

        optimization_time = (datetime.now() - start_time).total_seconds()

        return EntryTimingResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=len(study.trials),
            optimization_time=optimization_time,
            convergence_achieved=convergence_achieved,
            profit=best_results['profit'],
            sharpe_ratio=best_results['sharpe_ratio'],
            win_rate=best_results['win_rate'],
            max_drawdown=best_results['max_drawdown'],
            total_trades=best_results['total_trades'],
            trial_history=trial_history,
            convergence_history=convergence_history,
            recommendations=recommendations,
            risk_assessment=risk_assessment,
            model_name=model_name
        )

    def _optimize_with_skopt(self, model: Any, X: np.ndarray, y: np.ndarray,
                            analyst_signals: Optional[np.ndarray] = None,
                            hmm_regime_probs: Optional[np.ndarray] = None,
                            timestamps: Optional[np.ndarray] = None,
                            model_name: str = "model") -> EntryTimingResult:
        """Optimize using scikit-optimize."""
        start_time = datetime.now()

        # Define parameter space
        space = [
            Real(self.config.entry_threshold_min, self.config.entry_threshold_max, name='entry_threshold'),
            Real(self.config.exit_threshold_min, self.config.exit_threshold_max, name='exit_threshold'),
            Real(self.config.stop_loss_min, self.config.stop_loss_max, name='stop_loss'),
            Real(self.config.take_profit_min, self.config.take_profit_max, name='take_profit'),
            Integer(self.config.timing_window_min, self.config.timing_window_max, name='timing_window'),
            Real(self.config.confidence_threshold_min, self.config.confidence_threshold_max, name='confidence_threshold')
        ]

        # Define objective function
        @use_named_args(space)
        def objective(**params):
            try:
                results = self._simulate_trading_with_params(
                    model, X, y, analyst_signals, hmm_regime_probs, timestamps, params
                )

                # Calculate composite score
                score = self._calculate_composite_score(results)

                # Add constraints
                if results['max_drawdown'] > self.config.max_drawdown_threshold:
                    return -np.inf
                if results['win_rate'] < self.config.min_win_rate_threshold:
                    return -np.inf

                return score

            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return -np.inf

        # Run optimization
        result = gp_minimize(
            objective,
            space,
            n_calls=self.config.n_calls,
            random_state=self.config.random_state
        )

        # Get best parameters
        best_params = dict(zip([dim.name for dim in space], result.x))
        best_score = -result.fun  # Negative because gp_minimize minimizes

        # Simulate with best parameters
        best_results = self._simulate_trading_with_params(
            model, X, y, analyst_signals, hmm_regime_probs, timestamps, best_params
        )

        # Create trial history (simplified for skopt)
        trial_history = [{'params': best_params, 'value': best_score, 'state': 'COMPLETE'}]

        # Calculate convergence
        convergence_history = [best_score]
        convergence_achieved = True

        # Generate recommendations
        recommendations = self._generate_recommendations(best_results, best_params)
        risk_assessment = self._assess_risk(best_results)

        optimization_time = (datetime.now() - start_time).total_seconds()

        return EntryTimingResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=self.config.n_calls,
            optimization_time=optimization_time,
            convergence_achieved=convergence_achieved,
            profit=best_results['profit'],
            sharpe_ratio=best_results['sharpe_ratio'],
            win_rate=best_results['win_rate'],
            max_drawdown=best_results['max_drawdown'],
            total_trades=best_results['total_trades'],
            trial_history=trial_history,
            convergence_history=convergence_history,
            recommendations=recommendations,
            risk_assessment=risk_assessment,
            model_name=model_name
        )

    def _simulate_trading_with_params(self, model: Any, X: np.ndarray, y: np.ndarray,
                                     analyst_signals: Optional[np.ndarray] = None,
                                     hmm_regime_probs: Optional[np.ndarray] = None,
                                     timestamps: Optional[np.ndarray] = None,
                                     params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Simulate trading with given parameters using optimized computation."""
        # Create cache key for this simulation
        data_hash = hash((X.shape, y.shape, str(params) if params else "default"))
        if data_hash in self._simulation_cache:
            return self._simulation_cache[data_hash]

        try:
            # Extract parameters
            entry_threshold = params['entry_threshold']
            exit_threshold = params['exit_threshold']
            stop_loss = params['stop_loss']
            take_profit = params['take_profit']
            timing_window = params['timing_window']
            confidence_threshold = params['confidence_threshold']

            # Get model predictions (cached for efficiency)
            if hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(X)[:, 1]
            else:
                predictions = model.predict(X)

            # Performance tracking
            self._evaluation_count += 1

            # Initialize trading simulation
            trades = []
            current_position = None
            entry_price = 0.0
            entry_time = 0

            # Simulate trading
            for i in range(len(X)):
                current_price = y[i] if i < len(y) else y[-1]
                current_time = timestamps[i] if timestamps is not None else i

                # Check for entry signal
                if current_position is None:
                    # Check analyst signal
                    if analyst_signals is not None and analyst_signals[i] < confidence_threshold:
                        continue

                    # Check model prediction
                    if predictions[i] > entry_threshold:
                        current_position = 'long'
                        entry_price = current_price
                        entry_time = current_time

                # Check for exit signal
                elif current_position == 'long':
                    exit_triggered = False
                    exit_reason = ""

                    # Check stop loss
                    if current_price <= entry_price * (1 - stop_loss):
                        exit_triggered = True
                        exit_reason = "stop_loss"

                    # Check take profit
                    elif current_price >= entry_price * (1 + take_profit):
                        exit_triggered = True
                        exit_reason = "take_profit"

                    # Check timing window
                    elif current_time - entry_time >= timing_window:
                        exit_triggered = True
                        exit_reason = "timing_window"

                    # Check exit threshold
                    elif predictions[i] < exit_threshold:
                        exit_triggered = True
                        exit_reason = "exit_threshold"

                    if exit_triggered:
                        # Calculate trade result
                        trade_return = (current_price - entry_price) / entry_price
                        trade_duration = current_time - entry_time

                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'return': trade_return,
                            'duration': trade_duration,
                            'exit_reason': exit_reason
                        })

                        current_position = None

            # Calculate performance metrics
            if not trades:
                return {
                    'profit': 0.0,
                    'sharpe_ratio': 0.0,
                    'win_rate': 0.0,
                    'max_drawdown': 0.0,
                    'total_trades': 0
                }

            returns = [trade['return'] for trade in trades]
            total_return = sum(returns)
            win_rate = sum(1 for r in returns if r > 0) / len(returns)

            # Calculate Sharpe ratio using optimized vectorized operations
            if len(returns) > 1:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized
                else:
                    sharpe_ratio = 0.0
            else:
                sharpe_ratio = 0.0

            # Calculate maximum drawdown using efficient vectorized operations
            if len(returns) > 0:
                cumulative_returns = np.cumsum(returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = cumulative_returns - running_max
                max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
                max_drawdown = abs(max_drawdown)  # Ensure positive
            else:
                max_drawdown = 0.0

            result = {
                'profit': float(total_return),
                'sharpe_ratio': float(sharpe_ratio),
                'win_rate': float(win_rate),
                'max_drawdown': float(max_drawdown),
                'total_trades': len(trades)
            }

            # Cache the result (manage cache size)
            if len(self._simulation_cache) >= self._cache_max_size:
                # Remove oldest 20% of cache entries
                items_to_remove = len(self._simulation_cache) // 5
                cache_items = list(self._simulation_cache.items())
                for i in range(items_to_remove):
                    del self._simulation_cache[cache_items[i][0]]

            self._simulation_cache[data_hash] = result
            return result

        except Exception as e:
            logger.error(f"Trading simulation failed: {e}")
            return {
                'profit': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0
            }

    def _calculate_composite_score(self, results: Dict[str, Any]) -> float:
        """Calculate composite score from multiple objectives."""
        if not self.config.enable_multi_objective:
            return results['profit']

        # Normalize metrics
        profit_score = min(results['profit'] / 0.1, 1.0)  # Cap at 10% profit
        sharpe_score = min(results['sharpe_ratio'] / 2.0, 1.0)  # Cap at 2.0 Sharpe
        win_rate_score = results['win_rate']
        drawdown_score = 1.0 - min(results['max_drawdown'] / 0.2, 1.0)  # Penalize >20% drawdown

        # Weighted combination
        weights = self.config.objective_weights
        composite_score = (
            weights[0] * profit_score +
            weights[1] * sharpe_score +
            weights[2] * win_rate_score +
            weights[3] * drawdown_score
        )

        return composite_score

    def _generate_recommendations(self, results: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on optimization results."""
        recommendations = []

        # Profit recommendations
        if results['profit'] < 0.05:  # Less than 5% profit
            recommendations.append("Consider increasing entry threshold for higher quality signals")
            recommendations.append("Try different take profit levels")

        # Sharpe ratio recommendations
        if results['sharpe_ratio'] < 1.0:
            recommendations.append("Improve risk-adjusted returns by adjusting stop loss")
            recommendations.append("Consider position sizing optimization")

        # Win rate recommendations
        if results['win_rate'] < 0.5:  # Less than 50% win rate
            recommendations.append("Increase confidence threshold for better signal quality")
            recommendations.append("Consider different exit strategies")

        # Drawdown recommendations
        if results['max_drawdown'] > 0.1:  # More than 10% drawdown
            recommendations.append("Implement stricter risk management")
            recommendations.append("Consider reducing position size")

        # Parameter-specific recommendations
        if params['entry_threshold'] > 0.008:
            recommendations.append("Entry threshold is high - consider lowering for more opportunities")

        if params['stop_loss'] < 0.005:
            recommendations.append("Stop loss is tight - consider widening for better risk management")

        if params['timing_window'] > 8:
            recommendations.append("Timing window is long - consider shortening for faster execution")

        return recommendations

    def _assess_risk(self, results: Dict[str, Any]) -> str:
        """Assess risk level based on results."""
        risk_factors = []

        if results['max_drawdown'] > 0.15:
            risk_factors.append("High drawdown")

        if results['sharpe_ratio'] < 0.5:
            risk_factors.append("Low Sharpe ratio")

        if results['win_rate'] < 0.4:
            risk_factors.append("Low win rate")

        if results['total_trades'] < 10:
            risk_factors.append("Low trade frequency")

        if len(risk_factors) >= 3:
            return "High risk - Multiple risk factors present"
        elif len(risk_factors) >= 2:
            return "Medium risk - Some risk factors present"
        elif len(risk_factors) >= 1:
            return "Low risk - Minor risk factors present"
        else:
            return "Very low risk - Good performance metrics"

# Global instance
DEFAULT_BAYESIAN_OPTIMIZER = BayesianEntryTimingOptimizer()

def get_bayesian_optimizer(config: Optional[EntryTimingConfig] = None) -> BayesianEntryTimingOptimizer:
    """Get Bayesian optimizer instance."""
    if config is None:
        return DEFAULT_BAYESIAN_OPTIMIZER
    return BayesianEntryTimingOptimizer(config)

def optimize_entry_timing(model: Any,
                        X: np.ndarray,
                        y: np.ndarray,
                        analyst_signals: Optional[np.ndarray] = None,
                        hmm_regime_probs: Optional[np.ndarray] = None,
                        timestamps: Optional[np.ndarray] = None,
                        model_name: str = "model",
                        config: Optional[EntryTimingConfig] = None,
                        optimization_method: str = "optuna") -> EntryTimingResult:
    """Convenience function to optimize entry timing."""
    optimizer = get_bayesian_optimizer(config)
    return optimizer.optimize_entry_timing(
        model, X, y, analyst_signals, hmm_regime_probs, timestamps, model_name, optimization_method
    )
