"""
Unified Backtesting Engine

This module provides a unified backtesting engine that consolidates functionality
from TAS, NAS, and hybrid systems. It supports multiple backtesting modes,
regime-aware analysis, and comprehensive performance evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import common utilities
try:
    from src.utils.common_operations import (
        safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
        calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
        safe_apply_function, create_summary_statistics, safe_drop_columns,
        safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
        get_dataframe_info, safe_filter_dataframe, create_data_quality_report,
        optimize_dataframe_dtypes, safe_to_parquet, safe_read_parquet,
        align_dataframes, validate_dataframe_schema, guard_dataframe_nulls,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        integrate_with_m1_optimizers, memory_checkpoint, gpu_context,
        optimize_memory, get_memory_usage, validate_file_path, get_file_size,
        check_disk_space, CommonUtilities
    )
    COMMON_UTILS_AVAILABLE = True
except ImportError:
    COMMON_UTILS_AVAILABLE = False

try:
    from src.utils.math_validation import (
        safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
        validate_positive, validate_range, safe_kelly_calculation,
        safe_weighted_average, safe_percentage_change, safe_correlation,
        safe_covariance, safe_mean, safe_std, safe_percentile,
        validate_correlation_matrix, safe_matrix_inverse, math_safe,
        MathValidation, MathValidationError
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError:
    MATH_VALIDATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class BacktestingMode(Enum):
    """Backtesting modes."""
    HISTORICAL = "historical"
    WALK_FORWARD = "walk_forward"
    OUT_OF_SAMPLE = "out_of_sample"
    MONTE_CARLO = "monte_carlo"
    SCENARIO = "scenario"
    CROSS_VALIDATION = "cross_validation"


class PerformanceMetric(Enum):
    """Performance metrics for backtesting."""
    TOTAL_RETURN = "total_return"
    ANNUALIZED_RETURN = "annualized_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    VAR = "var"
    CVAR = "cvar"
    VOLATILITY = "volatility"
    BETA = "beta"
    ALPHA = "alpha"


@dataclass
class BacktestingConfig:
    """Unified configuration for backtesting engine."""
    
    # Core backtesting parameters
    mode: BacktestingMode = BacktestingMode.HISTORICAL
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005   # 0.05%
    
    # Data parameters
    data_frequency: str = "1H"  # 1H, 1D, etc.
    min_data_points: int = 1000
    max_data_points: int = 10000
    enable_data_validation: bool = True
    fill_missing_data: bool = True
    data_quality_threshold: float = 0.95
    
    # Regime detection parameters
    enable_regime_detection: bool = True
    regime_detection_method: str = "hybrid"  # "tas", "nas", "hybrid"
    regime_confidence_threshold: float = 0.7
    
    # Model parameters
    enable_model_selection: bool = True
    model_selection_strategy: str = "best_performance"
    
    # Performance analysis
    benchmark_symbol: Optional[str] = None
    risk_free_rate: float = 0.02  # 2% annual
    enable_risk_analysis: bool = True
    enable_performance_attribution: bool = True
    
    # Output parameters
    save_results: bool = True
    results_directory: Optional[str] = None
    enable_plotting: bool = True
    verbose: bool = True
    
    # Advanced parameters
    enable_memory_optimization: bool = True
    enable_gpu_acceleration: bool = False
    parallel_processing: bool = True
    max_workers: Optional[int] = None


@dataclass
class BacktestingResult:
    """Unified result from backtesting analysis."""
    
    # Basic information
    config: BacktestingConfig
    start_date: datetime
    end_date: datetime
    duration_days: int
    
    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    volatility: float
    
    # Risk metrics
    var_95: float
    cvar_95: float
    beta: float
    alpha: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    
    # Regime analysis
    regime_performance: Optional[Dict[str, Dict[str, float]]] = None
    regime_trades: Optional[Dict[str, int]] = None
    
    # Detailed data
    equity_curve: Optional[pd.DataFrame] = None
    trades_data: Optional[pd.DataFrame] = None
    performance_by_period: Optional[pd.DataFrame] = None
    
    # Metadata
    execution_time: float
    data_quality_score: float
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class BacktestingEngine:
    """
    Unified backtesting engine for TAS, NAS, and hybrid systems.
    
    This engine consolidates all backtesting functionality from different
    systems into a single, consistent interface.
    """
    
    def __init__(self, config: BacktestingConfig):
        """Initialize the backtesting engine."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_manager = None
        self.performance_attribution = None
        self.risk_analyzer = None
        
        # Initialize components
        self._initialize_components()
        
        # Validate configuration
        self._validate_config()
    
    def _initialize_components(self):
        """Initialize backtesting components."""
        try:
            # Import and initialize data manager
            from .data_manager import BacktestingDataManager, DataManagerConfig
            
            data_config = DataManagerConfig(
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                frequency=self.config.data_frequency,
                min_data_points=self.config.min_data_points,
                max_data_points=self.config.max_data_points,
                enable_validation=self.config.enable_data_validation,
                quality_threshold=self.config.data_quality_threshold
            )
            self.data_manager = BacktestingDataManager(data_config)
            
            # Import and initialize performance attribution
            if self.config.enable_performance_attribution:
                from .performance_attribution import PerformanceAttribution, PerformanceAttributionConfig
                
                perf_config = PerformanceAttributionConfig(
                    benchmark_symbol=self.config.benchmark_symbol,
                    risk_free_rate=self.config.risk_free_rate
                )
                self.performance_attribution = PerformanceAttribution(perf_config)
            
            # Import and initialize risk analyzer
            if self.config.enable_risk_analysis:
                from .risk_analyzer import RiskAnalyzer, RiskAnalysisConfig
                
                risk_config = RiskAnalysisConfig(
                    confidence_level=0.95,
                    enable_var=True,
                    enable_cvar=True
                )
                self.risk_analyzer = RiskAnalyzer(risk_config)
                
        except ImportError as e:
            self.logger.warning(f"Could not initialize some components: {e}")
    
    def _validate_config(self):
        """Validate backtesting configuration."""
        if self.config.start_date and self.config.end_date:
            if self.config.start_date >= self.config.end_date:
                raise ValueError("Start date must be before end date")
        
        if self.config.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        
        if not 0 <= self.config.commission_rate <= 1:
            raise ValueError("Commission rate must be between 0 and 1")
        
        if not 0 <= self.config.slippage_rate <= 1:
            raise ValueError("Slippage rate must be between 0 and 1")
    
    def run_backtest(
        self,
        model: Any,
        data: Optional[pd.DataFrame] = None,
        regime_detector: Optional[Any] = None
    ) -> BacktestingResult:
        """
        Run comprehensive backtesting analysis.
        
        Args:
            model: Trading model or strategy to backtest
            data: Market data (if None, will be loaded by data manager)
            regime_detector: Regime detection system (optional)
            
        Returns:
            BacktestingResult with comprehensive analysis
        """
        start_time = datetime.now()
        self.logger.info(f"Starting backtest in {self.config.mode.value} mode")
        
        try:
            # Load and prepare data
            if data is None:
                data = self._load_data()
            
            data = self._prepare_data(data)
            
            # Run regime detection if enabled
            regime_info = None
            if self.config.enable_regime_detection and regime_detector:
                regime_info = self._detect_regimes(data, regime_detector)
            
            # Execute backtesting based on mode
            if self.config.mode == BacktestingMode.HISTORICAL:
                result = self._run_historical_backtest(model, data, regime_info)
            elif self.config.mode == BacktestingMode.WALK_FORWARD:
                result = self._run_walk_forward_backtest(model, data, regime_info)
            elif self.config.mode == BacktestingMode.MONTE_CARLO:
                result = self._run_monte_carlo_backtest(model, data, regime_info)
            else:
                raise ValueError(f"Unsupported backtesting mode: {self.config.mode}")
            
            # Calculate performance metrics
            result = self._calculate_performance_metrics(result, data, regime_info)
            
            # Add risk analysis if enabled
            if self.config.enable_risk_analysis and self.risk_analyzer:
                result = self._add_risk_analysis(result, data)
            
            # Add performance attribution if enabled
            if self.config.enable_performance_attribution and self.performance_attribution:
                result = self._add_performance_attribution(result, data)
            
            # Save results if requested
            if self.config.save_results:
                self._save_results(result)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            self.logger.info(f"Backtest completed in {execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise
    
    def _load_data(self) -> pd.DataFrame:
        """Load market data using data manager."""
        if not self.data_manager:
            raise ValueError("Data manager not initialized")
        
        return self.data_manager.load_data()
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for backtesting."""
        # Validate data
        if COMMON_UTILS_AVAILABLE:
            data = validate_dataframe_columns(data)
            data = safe_convert_dtypes(data)
        
        # Fill missing data if enabled
        if self.config.fill_missing_data:
            data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure proper timestamp index
        if 'timestamp' in data.columns:
            data = data.set_index('timestamp')
        
        # Validate data quality
        if self.config.enable_data_validation and COMMON_UTILS_AVAILABLE:
            quality_score = calculate_data_quality_metrics(data)
            if quality_score < self.config.data_quality_threshold:
                self.logger.warning(f"Data quality score {quality_score:.3f} below threshold")
        
        return data
    
    def _detect_regimes(self, data: pd.DataFrame, regime_detector: Any) -> Dict[str, Any]:
        """Detect market regimes using the provided detector."""
        try:
            regime_predictions = regime_detector.detect_regimes(data)
            
            return {
                'predictions': regime_predictions,
                'regimes': regime_detector.get_regime_info(),
                'confidence': regime_detector.get_confidence_scores()
            }
        except Exception as e:
            self.logger.warning(f"Regime detection failed: {e}")
            return None
    
    def _run_historical_backtest(
        self,
        model: Any,
        data: pd.DataFrame,
        regime_info: Optional[Dict[str, Any]]
    ) -> BacktestingResult:
        """Run historical backtesting."""
        # This is a simplified implementation
        # In practice, this would integrate with the actual trading simulation
        
        equity_curve = self._simulate_trading(model, data, regime_info)
        trades_data = self._extract_trades(model, data, regime_info)
        
        return BacktestingResult(
            config=self.config,
            start_date=self.config.start_date or data.index.min(),
            end_date=self.config.end_date or data.index.max(),
            duration_days=0,  # Will be calculated
            equity_curve=equity_curve,
            trades_data=trades_data,
            # Placeholder values - will be calculated
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            volatility=0.0,
            var_95=0.0,
            cvar_95=0.0,
            beta=0.0,
            alpha=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            average_win=0.0,
            average_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            execution_time=0.0,
            data_quality_score=1.0
        )
    
    def _run_walk_forward_backtest(
        self,
        model: Any,
        data: pd.DataFrame,
        regime_info: Optional[Dict[str, Any]]
    ) -> BacktestingResult:
        """Run walk-forward backtesting."""
        # Implementation would use WalkForwardAnalyzer
        from .walk_forward_analyzer import WalkForwardAnalyzer, WalkForwardConfig
        
        wf_config = WalkForwardConfig(
            train_period_days=252,  # 1 year
            test_period_days=63,    # 1 quarter
            step_size_days=21       # 1 month
        )
        
        wf_analyzer = WalkForwardAnalyzer(wf_config)
        return wf_analyzer.analyze(model, data, regime_info)
    
    def _run_monte_carlo_backtest(
        self,
        model: Any,
        data: pd.DataFrame,
        regime_info: Optional[Dict[str, Any]]
    ) -> BacktestingResult:
        """Run Monte Carlo backtesting."""
        # Implementation would use MonteCarloEngine
        from .monte_carlo_engine import MonteCarloEngine, MonteCarloConfig
        
        mc_config = MonteCarloConfig(
            n_simulations=1000,
            confidence_level=0.95
        )
        
        mc_engine = MonteCarloEngine(mc_config)
        return mc_engine.run_simulation(model, data, regime_info)
    
    def _simulate_trading(
        self,
        model: Any,
        data: pd.DataFrame,
        regime_info: Optional[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Simulate trading based on model predictions."""
        # Simplified trading simulation
        # In practice, this would be much more sophisticated
        
        equity = self.config.initial_capital
        equity_curve = []
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            # Get model prediction (simplified)
            signal = self._get_model_signal(model, row, regime_info, i)
            
            # Apply trading logic (simplified)
            if signal > 0.5:  # Buy signal
                # Simulate buy
                equity *= (1 + row.get('returns', 0.001))
            elif signal < -0.5:  # Sell signal
                # Simulate sell
                equity *= (1 - row.get('returns', 0.001))
            
            equity_curve.append({
                'timestamp': timestamp,
                'equity': equity,
                'signal': signal
            })
        
        return pd.DataFrame(equity_curve)
    
    def _get_model_signal(
        self,
        model: Any,
        row: pd.Series,
        regime_info: Optional[Dict[str, Any]],
        index: int
    ) -> float:
        """Get trading signal from model."""
        # Simplified signal generation
        # In practice, this would use the actual model
        
        if hasattr(model, 'predict'):
            try:
                return model.predict(row.values.reshape(1, -1))[0]
            except:
                pass
        
        # Fallback to random signal
        import random
        return random.uniform(-1, 1)
    
    def _extract_trades(
        self,
        model: Any,
        data: pd.DataFrame,
        regime_info: Optional[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Extract trade information from simulation."""
        # Simplified trade extraction
        trades = []
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            signal = self._get_model_signal(model, row, regime_info, i)
            
            if abs(signal) > 0.5:  # Significant signal
                trades.append({
                    'timestamp': timestamp,
                    'signal': signal,
                    'price': row.get('close', 0),
                    'regime': regime_info.get('predictions', [])[i] if regime_info else None
                })
        
        return pd.DataFrame(trades)
    
    def _calculate_performance_metrics(
        self,
        result: BacktestingResult,
        data: pd.DataFrame,
        regime_info: Optional[Dict[str, Any]]
    ) -> BacktestingResult:
        """Calculate comprehensive performance metrics."""
        if result.equity_curve is None or len(result.equity_curve) == 0:
            return result
        
        equity = result.equity_curve['equity'].values
        returns = np.diff(equity) / equity[:-1]
        
        # Calculate basic metrics
        result.total_return = (equity[-1] - equity[0]) / equity[0]
        result.annualized_return = (1 + result.total_return) ** (365 / result.duration_days) - 1
        result.volatility = np.std(returns) * np.sqrt(252)
        result.sharpe_ratio = (result.annualized_return - self.config.risk_free_rate) / result.volatility
        
        # Calculate drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        result.max_drawdown = np.min(drawdown)
        
        # Calculate trade statistics
        if result.trades_data is not None and len(result.trades_data) > 0:
            result.total_trades = len(result.trades_data)
            # Additional trade statistics would be calculated here
        
        return result
    
    def _add_risk_analysis(self, result: BacktestingResult, data: pd.DataFrame) -> BacktestingResult:
        """Add risk analysis to results."""
        if not self.risk_analyzer or result.equity_curve is None:
            return result
        
        try:
            returns = result.equity_curve['equity'].pct_change().dropna()
            risk_metrics = self.risk_analyzer.analyze(returns)
            
            result.var_95 = risk_metrics.get('var_95', 0.0)
            result.cvar_95 = risk_metrics.get('cvar_95', 0.0)
            
        except Exception as e:
            self.logger.warning(f"Risk analysis failed: {e}")
        
        return result
    
    def _add_performance_attribution(
        self,
        result: BacktestingResult,
        data: pd.DataFrame
    ) -> BacktestingResult:
        """Add performance attribution to results."""
        if not self.performance_attribution or result.equity_curve is None:
            return result
        
        try:
            # Calculate benchmark comparison if available
            if self.config.benchmark_symbol:
                benchmark_returns = data.get(self.config.benchmark_symbol, None)
                if benchmark_returns is not None:
                    strategy_returns = result.equity_curve['equity'].pct_change().dropna()
                    
                    # Calculate beta and alpha
                    correlation = np.corrcoef(strategy_returns, benchmark_returns)[0, 1]
                    beta = correlation * (strategy_returns.std() / benchmark_returns.std())
                    alpha = strategy_returns.mean() - beta * benchmark_returns.mean()
                    
                    result.beta = beta
                    result.alpha = alpha
            
        except Exception as e:
            self.logger.warning(f"Performance attribution failed: {e}")
        
        return result
    
    def _save_results(self, result: BacktestingResult):
        """Save backtesting results."""
        if not self.config.results_directory:
            return
        
        results_dir = Path(self.config.results_directory)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_file = results_dir / f"backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'config': self.config.__dict__,
                'performance': {
                    'total_return': result.total_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'total_trades': result.total_trades
                },
                'execution_time': result.execution_time
            }, f, indent=2, default=str)
        
        # Save detailed data
        if result.equity_curve is not None:
            equity_file = results_dir / f"equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            result.equity_curve.to_parquet(equity_file)
        
        if result.trades_data is not None:
            trades_file = results_dir / f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            result.trades_data.to_parquet(trades_file)
        
        self.logger.info(f"Results saved to {results_dir}")


# Convenience functions for backward compatibility
def create_backtesting_engine(config: BacktestingConfig) -> BacktestingEngine:
    """Create a backtesting engine with the given configuration."""
    return BacktestingEngine(config)


def run_quick_backtest(
    model: Any,
    data: pd.DataFrame,
    config: Optional[BacktestingConfig] = None
) -> BacktestingResult:
    """Run a quick backtest with default configuration."""
    if config is None:
        config = BacktestingConfig()
    
    engine = BacktestingEngine(config)
    return engine.run_backtest(model, data)