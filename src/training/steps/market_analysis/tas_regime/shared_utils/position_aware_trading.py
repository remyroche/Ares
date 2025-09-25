"""
Position Aware Trading - Comprehensive Implementation

This module provides comprehensive position-aware trading functionality using
unified evaluation framework and advanced ML utilities.

Features:
- Position-aware model evaluation
- Advanced risk management
- Multi-objective optimization
- Regime-specific trading strategies
- Hardware-optimized computations
- Comprehensive logging and monitoring
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
from pathlib import Path

# Import unified evaluation framework
from src.utils.nas_tas.shared_utils.unified_evaluation_framework import (
    UnifiedEvaluationFramework, EvaluationConfig
)
from src.utils.nas_tas.shared_utils.unified_architecture_config import ArchitectureType

# Import utility modules
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    safe_float, safe_int, validate_finite, validate_positive, validate_range,
    safe_kelly_calculation, safe_weighted_average, safe_percentage_change,
    safe_correlation, safe_covariance, safe_percentile,
    validate_correlation_matrix, safe_matrix_inverse, math_safe,
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    integrate_with_m1_optimizers, cleanup_m1_optimizers,
    memory_checkpoint, gpu_context, optimize_memory, get_memory_usage
)

from src.utils.common_utilities import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report
)

from src.utils.math_validation import (
    MathValidation, MathValidationError,
    safe_divide as math_safe_divide, safe_log as math_safe_log,
    safe_sqrt as math_safe_sqrt, safe_power as math_safe_power,
    validate_finite as math_validate_finite, validate_positive as math_validate_positive,
    validate_range as math_validate_range, safe_kelly_calculation as math_safe_kelly,
    safe_weighted_average as math_safe_weighted_avg, safe_percentage_change as math_safe_pct_change
)

from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_progress, tprint_performance, tprint_structured, tprint_with_level,
    configure_tprint, TPrintConfig, LogLevel
)

from src.utils.data.klines_parquet import KlinesParquetManager

# Import ML common utilities
try:
    from src.utils.ml_common import (
        EnhancedModelFactory, ModelType, ModelConfig,
        EnsembleManager, EnsembleType, EnsembleConfig,
        ParetoOptimizer, ParetoFront, ParetoFrontAnalyzer,
        UnifiedCrossValidator, UnifiedCVResult,
        MemoryOptimizer, ParallelProcessor, UnifiedCache,
        LookaheadProtection, MLTrainingSafeguards, RobustErrorHandler
    )
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False
    tprint_warning("ML Common utilities not available - using fallback implementations")

# Import matrix operations
try:
    from src.utils.matrix_operations.unified_operations import (
        M1EnhancedMatrixOperations, get_enhanced_matrix_operations
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False
    tprint_warning("Matrix operations not available - using fallback implementations")

# Import hardware optimizations
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUManager, is_m1_available, is_mps_available
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
    HARDWARE_OPT_AVAILABLE = True
except ImportError:
    HARDWARE_OPT_AVAILABLE = False
    tprint_warning("Hardware optimizations not available - using fallback implementations")


class PositionType(Enum):
    """Position type enumeration."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TradingRegime(Enum):
    """Trading regime enumeration."""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


@dataclass
class PositionConfig:
    """Configuration for position-aware trading."""
    
    # Position management
    max_position_size: float = 1.0
    min_position_size: float = 0.01
    position_sizing_method: str = "kelly"  # kelly, fixed, volatility_adjusted
    
    # Risk management
    max_drawdown: float = 0.2
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.1
    risk_free_rate: float = 0.02
    
    # Trading parameters
    min_confidence_threshold: float = 0.6
    max_trades_per_day: int = 10
    cooldown_period: int = 5  # minutes
    
    # Regime-specific parameters
    regime_adaptation: bool = True
    regime_confidence_threshold: float = 0.7
    
    # Hardware optimization
    use_m1_optimization: bool = True
    enable_gpu_acceleration: bool = True
    memory_optimization: bool = True
    
    # Logging and monitoring
    enable_detailed_logging: bool = True
    log_performance_metrics: bool = True
    save_trading_logs: bool = True
    
    # Advanced features
    enable_ensemble_trading: bool = True
    enable_regime_detection: bool = True
    enable_risk_management: bool = True
    enable_position_sizing: bool = True


@dataclass
class PositionMetrics:
    """Metrics for position evaluation."""
    
    # Basic metrics
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Position-specific metrics
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    kelly_criterion: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall: float = 0.0
    downside_deviation: float = 0.0
    
    # Regime-specific metrics
    regime_performance: Dict[str, float] = field(default_factory=dict)
    regime_accuracy: Dict[str, float] = field(default_factory=dict)
    
    # Hardware performance
    computation_time: float = 0.0
    memory_usage: float = 0.0
    gpu_utilization: float = 0.0


class PositionAwareTrading:
    """Comprehensive position-aware trading system."""
    
    def __init__(self, config: PositionConfig = None, evaluation_config: EvaluationConfig = None):
        """Initialize position-aware trading system.
        
        Args:
            config: Position trading configuration
            evaluation_config: Evaluation framework configuration
        """
        self.config = config or PositionConfig()
        self.evaluation_config = evaluation_config or EvaluationConfig()
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Initialize evaluation framework
        self.evaluator = UnifiedEvaluationFramework(
            architecture_type=ArchitectureType.TAS,
            config=self.evaluation_config
        )
        
        # Initialize hardware optimizations
        self._setup_hardware_optimizations()
        
        # Initialize ML components
        self._setup_ml_components()
        
        # Initialize position tracking
        self.positions = []
        self.current_position = None
        self.position_history = []
        
        # Initialize metrics
        self.metrics = PositionMetrics()
        
        # Initialize serialization
        self.serializer = UniversalSerializer()
        
        tprint_success("Position-aware trading system initialized")
    
    def _setup_logging(self):
        """Setup comprehensive logging."""
        if self.config.enable_detailed_logging:
            tprint_config = TPrintConfig(
                timestamp_format="%Y-%m-%d %H:%M:%S.%f",
                use_colors=True,
                output_to_file=True,
                output_file="logs/position_trading.log",
                enable_structured_logging=True
            )
            configure_tprint(tprint_config)
    
    def _setup_hardware_optimizations(self):
        """Setup hardware optimizations."""
        if not self.config.use_m1_optimization or not HARDWARE_OPT_AVAILABLE:
            return
        
        try:
            # Integrate M1 optimizations
            integration_result = integrate_with_m1_optimizers()
            if integration_result.get('success', False):
                tprint_success("M1 hardware optimizations enabled")
            else:
                tprint_warning(f"M1 optimization failed: {integration_result.get('error', 'Unknown error')}")
        except Exception as e:
            tprint_warning(f"Hardware optimization setup failed: {e}")
    
    def _setup_ml_components(self):
        """Setup ML components."""
        if not ML_COMMON_AVAILABLE:
            tprint_warning("ML Common not available - using basic implementations")
            return
        
        try:
            # Initialize ensemble manager if available
            if self.config.enable_ensemble_trading:
                self.ensemble_manager = EnsembleManager()
                tprint_info("Ensemble trading enabled")
            
            # Initialize memory optimizer
            self.memory_optimizer = MemoryOptimizer()
            
            # Initialize parallel processor
            self.parallel_processor = ParallelProcessor()
            
            # Initialize unified cache
            self.cache = UnifiedCache()
            
            # Initialize safeguards
            self.safeguards = MLTrainingSafeguards()
            self.error_handler = RobustErrorHandler()
            
        except Exception as e:
            tprint_warning(f"ML component setup failed: {e}")
    
    def evaluate_trading_performance(
        self, 
        model: Any, 
        X_test: np.ndarray, 
        y_test: np.ndarray, 
        positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate trading performance with position awareness.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            positions: Position data (optional)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with performance metrics
        """
        tprint_info("Starting position-aware trading evaluation")
        
        try:
            # Setup memory checkpoint
            with memory_checkpoint("trading_evaluation"):
                # Validate inputs
                self._validate_inputs(model, X_test, y_test, positions)
                
                # Generate predictions
                predictions = self._generate_predictions(model, X_test)
                
                # Calculate position metrics
                if positions is not None:
                    metrics = self._calculate_position_metrics(
                        predictions, y_test, positions
                    )
                else:
                    # Use unified evaluation framework
                    metrics = self.evaluator.evaluate_trading_performance(
                        model, X_test, y_test, positions, **kwargs
                    )
                
                # Add hardware performance metrics
                if self.config.log_performance_metrics:
                    metrics.update(self._get_hardware_metrics())
                
                # Update internal metrics
                self._update_metrics(metrics)
                
                tprint_success(f"Trading evaluation completed: Sharpe={metrics.get('sharpe_ratio', 0):.3f}")
                return metrics
                
        except Exception as e:
            tprint_error(f"Trading evaluation failed: {e}")
            raise
    
    def _validate_inputs(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, positions: Optional[np.ndarray]):
        """Validate input data."""
        if model is None:
            raise ValueError("Model cannot be None")
        
        if X_test is None or len(X_test) == 0:
            raise ValueError("X_test cannot be empty")
        
        if y_test is None or len(y_test) == 0:
            raise ValueError("y_test cannot be empty")
        
        if len(X_test) != len(y_test):
            raise ValueError("X_test and y_test must have same length")
        
        if positions is not None and len(positions) != len(y_test):
            raise ValueError("positions must have same length as y_test")
    
    def _generate_predictions(self, model: Any, X_test: np.ndarray) -> np.ndarray:
        """Generate model predictions."""
        try:
            # Use GPU context if available
            if self.config.enable_gpu_acceleration and HARDWARE_OPT_AVAILABLE:
                with gpu_context("model_prediction"):
                    predictions = model.predict(X_test)
            else:
                predictions = model.predict(X_test)
            
            # Validate predictions
            if not np.all(np.isfinite(predictions)):
                tprint_warning("Non-finite predictions detected, applying correction")
                predictions = np.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return predictions
            
        except Exception as e:
            tprint_error(f"Prediction generation failed: {e}")
            raise
    
    def _calculate_position_metrics(
        self, 
        predictions: np.ndarray, 
        y_test: np.ndarray, 
        positions: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate position-aware metrics."""
        try:
            # Basic performance metrics
            returns = self._calculate_returns(predictions, y_test, positions)
            
            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(returns)
            
            # Position-specific metrics
            position_metrics = self._calculate_position_specific_metrics(
                predictions, y_test, positions, returns
            )
            
            # Regime-specific metrics
            regime_metrics = self._calculate_regime_metrics(
                predictions, y_test, positions, returns
            )
            
            # Combine all metrics
            metrics = {
                **risk_metrics,
                **position_metrics,
                **regime_metrics,
                'total_return': np.sum(returns),
                'num_trades': len(returns),
                'avg_return': np.mean(returns)
            }
            
            return metrics
            
        except Exception as e:
            tprint_error(f"Metrics calculation failed: {e}")
            raise
    
    def _calculate_returns(
        self, 
        predictions: np.ndarray, 
        y_test: np.ndarray, 
        positions: np.ndarray
    ) -> np.ndarray:
        """Calculate position-based returns."""
        try:
            # Position-weighted returns
            returns = predictions * positions * y_test
            
            # Apply transaction costs if configured
            if hasattr(self.config, 'transaction_cost'):
                returns -= self.config.transaction_cost * np.abs(np.diff(positions, prepend=0))
            
            return returns
            
        except Exception as e:
            tprint_error(f"Returns calculation failed: {e}")
            raise
    
    def _calculate_risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate risk metrics."""
        try:
            if len(returns) == 0:
                return {'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'var_95': 0.0}
            
            # Sharpe ratio
            sharpe = safe_divide(np.mean(returns), np.std(returns)) * np.sqrt(252)
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                sortino = safe_divide(np.mean(returns), np.std(downside_returns)) * np.sqrt(252)
            else:
                sortino = sharpe
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns)
            
            # Value at Risk (95%)
            var_95 = np.percentile(returns, 5)
            
            # Expected Shortfall
            es = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else 0.0
            
            return {
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'expected_shortfall': es
            }
            
        except Exception as e:
            tprint_error(f"Risk metrics calculation failed: {e}")
            return {'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'var_95': 0.0}
    
    def _calculate_position_specific_metrics(
        self, 
        predictions: np.ndarray, 
        y_test: np.ndarray, 
        positions: np.ndarray, 
        returns: np.ndarray
    ) -> Dict[str, float]:
        """Calculate position-specific metrics."""
        try:
            # Win rate
            winning_trades = np.sum(returns > 0)
            total_trades = len(returns)
            win_rate = safe_divide(winning_trades, total_trades)
            
            # Average win/loss
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            avg_win = np.mean(wins) if len(wins) > 0 else 0.0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
            
            # Profit factor
            total_wins = np.sum(wins) if len(wins) > 0 else 0.0
            total_losses = np.abs(np.sum(losses)) if len(losses) > 0 else 1.0
            profit_factor = safe_divide(total_wins, total_losses)
            
            # Kelly criterion
            kelly = safe_kelly_calculation(win_rate, avg_win, abs(avg_loss))
            
            return {
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'kelly_criterion': kelly
            }
            
        except Exception as e:
            tprint_error(f"Position metrics calculation failed: {e}")
            return {'win_rate': 0.0, 'profit_factor': 0.0, 'kelly_criterion': 0.0}
    
    def _calculate_regime_metrics(
        self, 
        predictions: np.ndarray, 
        y_test: np.ndarray, 
        positions: np.ndarray, 
        returns: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate regime-specific metrics."""
        if not self.config.enable_regime_detection:
            return {}
        
        try:
            # Simple regime detection based on volatility
            volatility = np.std(returns)
            
            if volatility > 0.02:
                regime = TradingRegime.VOLATILE
            elif volatility < 0.005:
                regime = TradingRegime.RANGING
            else:
                regime = TradingRegime.TRENDING
            
            # Regime-specific performance
            regime_performance = {
                'regime': regime.value,
                'volatility': volatility,
                'regime_return': np.mean(returns)
            }
            
            return regime_performance
            
        except Exception as e:
            tprint_warning(f"Regime metrics calculation failed: {e}")
            return {}
    
    def _get_hardware_metrics(self) -> Dict[str, float]:
        """Get hardware performance metrics."""
        try:
            metrics = {
                'computation_time': 0.0,
                'memory_usage': get_memory_usage(),
                'gpu_utilization': 0.0
            }
            
            if HARDWARE_OPT_AVAILABLE:
                # Get M1-specific metrics
                gpu_manager = get_m1_gpu_manager()
                if gpu_manager:
                    gpu_info = gpu_manager.get_gpu_info()
                    metrics['gpu_utilization'] = gpu_info.get('utilization', 0.0)
            
            return metrics
            
        except Exception as e:
            tprint_warning(f"Hardware metrics collection failed: {e}")
            return {'computation_time': 0.0, 'memory_usage': 0.0, 'gpu_utilization': 0.0}
    
    def _update_metrics(self, metrics: Dict[str, Any]):
        """Update internal metrics."""
        try:
            # Update position metrics
            for key, value in metrics.items():
                if hasattr(self.metrics, key):
                    setattr(self.metrics, key, value)
            
            # Save metrics if configured
            if self.config.save_trading_logs:
                self._save_metrics(metrics)
                
        except Exception as e:
            tprint_warning(f"Metrics update failed: {e}")
    
    def _save_metrics(self, metrics: Dict[str, Any]):
        """Save metrics to file."""
        try:
            # Create logs directory
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            # Save as JSON
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"position_metrics_{timestamp}.json"
            filepath = logs_dir / filename
            
            self.serializer.save(metrics, str(filepath), format="json")
            tprint_info(f"Metrics saved to {filepath}")
            
        except Exception as e:
            tprint_warning(f"Metrics saving failed: {e}")
    
    def optimize_position_sizing(
        self, 
        returns: np.ndarray, 
        method: str = "kelly"
    ) -> float:
        """Optimize position sizing.
        
        Args:
            returns: Historical returns
            method: Sizing method (kelly, fixed, volatility_adjusted)
            
        Returns:
            Optimal position size
        """
        try:
            if method == "kelly":
                # Kelly criterion
                win_rate = np.mean(returns > 0)
                avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0.0
                avg_loss = np.abs(np.mean(returns[returns < 0])) if np.any(returns < 0) else 1.0
                
                kelly = safe_kelly_calculation(win_rate, avg_win, avg_loss)
                
                # Apply constraints
                kelly = max(self.config.min_position_size, 
                           min(kelly, self.config.max_position_size))
                
                return kelly
                
            elif method == "fixed":
                return self.config.max_position_size * 0.5
                
            elif method == "volatility_adjusted":
                # Volatility-adjusted sizing
                volatility = np.std(returns)
                target_vol = 0.02  # 2% target volatility
                
                size = safe_divide(target_vol, volatility)
                size = max(self.config.min_position_size, 
                          min(size, self.config.max_position_size))
                
                return size
                
            else:
                tprint_warning(f"Unknown sizing method: {method}")
                return self.config.max_position_size * 0.5
                
        except Exception as e:
            tprint_error(f"Position sizing optimization failed: {e}")
            return self.config.max_position_size * 0.5
    
    def detect_trading_regime(self, data: np.ndarray) -> TradingRegime:
        """Detect current trading regime.
        
        Args:
            data: Market data (prices, returns, etc.)
            
        Returns:
            Detected trading regime
        """
        try:
            if len(data) < 10:
                return TradingRegime.RANGING
            
            # Calculate regime indicators
            returns = np.diff(data) / data[:-1]
            volatility = np.std(returns)
            trend_strength = np.corrcoef(np.arange(len(returns)), returns)[0, 1]
            
            # Regime classification
            if volatility > 0.03:
                return TradingRegime.VOLATILE
            elif abs(trend_strength) > 0.3:
                return TradingRegime.TRENDING
            elif volatility < 0.01:
                return TradingRegime.RANGING
            else:
                return TradingRegime.BREAKOUT
                
        except Exception as e:
            tprint_warning(f"Regime detection failed: {e}")
            return TradingRegime.RANGING
    
    def calculate_risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive risk metrics.
        
        Args:
            returns: Portfolio returns
            
        Returns:
            Dictionary with risk metrics
        """
        try:
            if len(returns) == 0:
                return {'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
            
            # Basic risk metrics
            metrics = self._calculate_risk_metrics(returns)
            
            # Additional risk metrics
            metrics.update({
                'calmar_ratio': safe_divide(metrics['sharpe_ratio'], abs(metrics['max_drawdown'])),
                'downside_deviation': np.std(returns[returns < 0]) if np.any(returns < 0) else 0.0,
                'var_99': np.percentile(returns, 1),
                'skewness': self._calculate_skewness(returns),
                'kurtosis': self._calculate_kurtosis(returns)
            })
            
            return metrics
            
        except Exception as e:
            tprint_error(f"Risk metrics calculation failed: {e}")
            return {'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns."""
        try:
            if len(returns) < 3:
                return 0.0
            
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            
            if std_ret == 0:
                return 0.0
            
            skewness = np.mean(((returns - mean_ret) / std_ret) ** 3)
            return skewness
            
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns."""
        try:
            if len(returns) < 4:
                return 0.0
            
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            
            if std_ret == 0:
                return 0.0
            
            kurtosis = np.mean(((returns - mean_ret) / std_ret) ** 4) - 3
            return kurtosis
            
        except Exception:
            return 0.0
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            # Cleanup M1 optimizations
            if HARDWARE_OPT_AVAILABLE:
                cleanup_m1_optimizers()
            
            # Cleanup ML components
            if hasattr(self, 'cache'):
                self.cache.clear()
            
            tprint_info("Position-aware trading system cleaned up")
            
        except Exception as e:
            tprint_warning(f"Cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Convenience functions
def create_position_aware_trading(
    config: PositionConfig = None,
    evaluation_config: EvaluationConfig = None
) -> PositionAwareTrading:
    """Create position-aware trading system.
    
    Args:
        config: Position configuration
        evaluation_config: Evaluation configuration
        
    Returns:
        PositionAwareTrading instance
    """
    return PositionAwareTrading(config, evaluation_config)


def evaluate_trading_with_positions(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    positions: Optional[np.ndarray] = None,
    config: PositionConfig = None
) -> Dict[str, Any]:
    """Convenience function for position-aware evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        positions: Position data
        config: Position configuration
        
    Returns:
        Performance metrics
    """
    with create_position_aware_trading(config) as trading_system:
        return trading_system.evaluate_trading_performance(
            model, X_test, y_test, positions
        )


# Backward compatibility
def evaluate_trading(model, X_test, y_test, positions, **kwargs):
    """Backward compatibility function."""
    return evaluate_trading_with_positions(model, X_test, y_test, positions)


if __name__ == "__main__":
    # Example usage
    tprint_info("Position-aware trading system example")
    
    # Create system
    config = PositionConfig(
        max_position_size=0.5,
        min_position_size=0.01,
        enable_detailed_logging=True
    )
    
    with create_position_aware_trading(config) as trading_system:
        # Example evaluation
        np.random.seed(42)
        X_test = np.random.randn(100, 10)
        y_test = np.random.randn(100)
        positions = np.random.choice([-1, 0, 1], 100)
        
        # Mock model
        class MockModel:
            def predict(self, X):
                return np.random.randn(len(X))
        
        model = MockModel()
        
        # Evaluate
        metrics = trading_system.evaluate_trading_performance(
            model, X_test, y_test, positions
        )
        
        tprint_success(f"Evaluation completed: {metrics}")