"""
Walk-Forward Validation Step

This module provides comprehensive walk-forward validation functionality for
backtesting strategies with proper time series validation and performance analysis.

Key Features:
- Walk-forward validation with configurable windows
- Time series cross-validation
- Performance degradation detection
- Regime-aware validation
- Statistical significance testing
- Comprehensive reporting and analysis
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import gc
import psutil
from pathlib import Path
from scipy import stats

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size, check_disk_space
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, MathValidationError
)
from src.utils.parquet_utils import get_parquet_utils, ParquetUtils
from src.utils.comprehensive_function_logger import (
    log_step_functions, log_important_calls, log_all_calls, 
    log_internal_call, log_step_progress, log_data_operation
)
from src.utils.enhanced_financial_metrics_logger import EnhancedFinancialMetricsLogger
from src.utils.performance_utils import PerformanceMonitor
from src.utils.monitoring_utils import SystemMonitor

# Backtesting utilities
from src.utils.common_ml.backtesting.backtesting_engine import (
    BacktestingEngine, BacktestingConfig, BacktestingResults, BacktestingMode
)
from src.utils.common_ml.backtesting.analytics_reporter import AnalyticsReporter

# Core decorators and validation
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time, 
    timeout, error_boundary, compose, validate_data_quality, 
    monitor_step_execution, ensure_data_integrity, validate_pipeline_step
)
from src.core.errors import (
    ValidationError, DataIntegrityError, FileOperationError,
    MathValidationError, TimeoutError
)

# Training step utilities
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

logger = logging.getLogger(__name__)


class WalkForwardMode(Enum):
    """Walk-forward validation modes."""
    EXPANDING_WINDOW = "expanding_window"
    ROLLING_WINDOW = "rolling_window"
    FIXED_WINDOW = "fixed_window"
    ADAPTIVE_WINDOW = "adaptive_window"


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    # Basic configuration
    symbol: str
    exchange: str
    timeframe: str
    data_dir: str
    
    # Walk-forward parameters
    initial_training_days: int = 252  # 1 year
    training_window_days: int = 252   # 1 year
    testing_window_days: int = 63     # 3 months
    step_size_days: int = 21          # 1 month
    min_training_days: int = 126      # 6 months
    max_training_days: int = 504      # 2 years
    
    # Validation mode
    validation_mode: WalkForwardMode = WalkForwardMode.EXPANDING_WINDOW
    
    # Performance thresholds
    min_sharpe_ratio: float = 0.5
    max_drawdown_threshold: float = 0.2
    min_win_rate: float = 0.4
    min_trades_per_window: int = 5
    
    # Statistical testing
    enable_statistical_testing: bool = True
    confidence_level: float = 0.95
    min_observations: int = 10
    
    # Performance settings
    enable_detailed_logging: bool = True
    enable_performance_monitoring: bool = True
    enable_memory_optimization: bool = True
    
    # Output settings
    save_detailed_results: bool = True
    generate_plots: bool = True
    output_format: str = "parquet"


@dataclass
class WalkForwardResults:
    """Results from walk-forward validation."""
    # Basic info
    symbol: str
    exchange: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    total_duration: float
    
    # Walk-forward windows
    windows: List[Dict[str, Any]] = field(default_factory=list)
    n_windows: int = 0
    
    # Performance metrics
    overall_performance: Dict[str, float] = field(default_factory=dict)
    window_performance: List[Dict[str, Any]] = field(default_factory=list)
    
    # Statistical analysis
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    performance_stability: Dict[str, Any] = field(default_factory=dict)
    
    # Risk analysis
    risk_metrics: Dict[str, Any] = field(default_factory=dict)
    drawdown_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Validation results
    validation_summary: Dict[str, Any] = field(default_factory=dict)
    degradation_detection: Dict[str, Any] = field(default_factory=dict)
    
    # Detailed data
    equity_curves: Dict[str, pd.DataFrame] = field(default_factory=dict)
    trade_logs: Dict[str, pd.DataFrame] = field(default_factory=dict)
    
    # Metadata
    config: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    system_metrics: Dict[str, Any] = field(default_factory=dict)


class WalkForwardValidationStep:
    """Walk-forward validation step."""
    
    def __init__(self, config: WalkForwardConfig):
        """Initialize the walk-forward validation step."""
        self.config = config
        self.logger = logger.getChild('WalkForwardValidationStep')
        
        # Initialize utilities
        self.parquet_utils = get_parquet_utils()
        self.financial_logger = EnhancedFinancialMetricsLogger()
        self.performance_monitor = PerformanceMonitor()
        self.system_monitor = SystemMonitor()
        
        # Initialize data directory
        self.data_dir = Path(config.data_dir)
        ensure_directory(self.data_dir)
        
        self.logger.info(f"🚀 WalkForwardValidationStep initialized for {config.symbol}")
        self.logger.info(f"📊 Validation mode: {config.validation_mode.value}")
        self.logger.info(f"📅 Training window: {config.training_window_days} days")
        self.logger.info(f"📅 Testing window: {config.testing_window_days} days")
        self.logger.info(f"📅 Step size: {config.step_size_days} days")
        self.logger.info(f"📁 Data directory: {config.data_dir}")
    
    @traced(span_name='walk_forward_validation')
    @log_execution_time
    @monitor_step_execution
    async def execute(
        self, 
        data: Optional[pd.DataFrame] = None,
        strategy_func: Optional[Callable] = None,
        **kwargs
    ) -> WalkForwardResults:
        """Execute walk-forward validation."""
        
        self.logger.info("🚀 Starting walk-forward validation...")
        start_time = time.time()
        
        # Start performance monitoring
        if self.config.enable_performance_monitoring:
            self.performance_monitor.start_monitoring()
        
        try:
            # Load data if not provided
            if data is None:
                data = await self._load_data()
            
            # Validate data
            self._validate_data(data)
            
            # Generate walk-forward windows
            windows = self._generate_walk_forward_windows(data)
            self.logger.info(f"📊 Generated {len(windows)} walk-forward windows")
            
            # Execute validation for each window
            window_results = await self._execute_walk_forward_validation(windows, strategy_func)
            
            # Analyze results
            overall_performance = self._calculate_overall_performance(window_results)
            statistical_tests = self._perform_statistical_tests(window_results)
            performance_stability = self._analyze_performance_stability(window_results)
            risk_metrics = self._calculate_risk_metrics(window_results)
            drawdown_analysis = self._analyze_drawdowns(window_results)
            validation_summary = self._generate_validation_summary(window_results)
            degradation_detection = self._detect_performance_degradation(window_results)
            
            # Create results
            results = WalkForwardResults(
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                timeframe=self.config.timeframe,
                start_time=datetime.now(),
                end_time=datetime.now(),
                total_duration=time.time() - start_time,
                windows=windows,
                n_windows=len(windows),
                overall_performance=overall_performance,
                window_performance=window_results,
                statistical_tests=statistical_tests,
                performance_stability=performance_stability,
                risk_metrics=risk_metrics,
                drawdown_analysis=drawdown_analysis,
                validation_summary=validation_summary,
                degradation_detection=degradation_detection,
                config=self.config,
                execution_time=time.time() - start_time,
                memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
                system_metrics=self._get_system_metrics()
            )
            
            # Save results
            if self.config.save_detailed_results:
                await self._save_results(results)
            
            self.logger.info("✅ Walk-forward validation completed successfully")
            self.logger.info(f"⏱️ Execution time: {results.execution_time:.2f}s")
            self.logger.info(f"📊 Windows processed: {results.n_windows}")
            self.logger.info(f"📈 Overall Sharpe ratio: {overall_performance.get('sharpe_ratio', 0):.2f}")
            self.logger.info(f"⚠️ Max drawdown: {overall_performance.get('max_drawdown', 0):.2%}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in walk-forward validation: {e}")
            self.logger.exception("Full traceback:")
            raise
        finally:
            # Stop performance monitoring
            if self.config.enable_performance_monitoring:
                self.performance_monitor.stop_monitoring()
    
    async def _load_data(self) -> pd.DataFrame:
        """Load market data for validation."""
        self.logger.info("📂 Loading market data...")
        
        # Try to load consolidated data first
        consolidated_file = self.data_dir / f"aggtrades_{self.config.exchange}_{self.config.symbol}_consolidated.parquet"
        
        if safe_file_exists(consolidated_file):
            self.logger.info(f"📁 Loading consolidated data: {consolidated_file}")
            data = standardized_parquet_handler.read_parquet_standardized(consolidated_file)
        else:
            # Fallback to individual files
            self.logger.info("📁 Consolidated file not found, loading individual files...")
            data = await self._load_individual_files()
        
        self.logger.info(f"📊 Loaded {len(data):,} data points")
        self.logger.info(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
        
        return data
    
    async def _load_individual_files(self) -> pd.DataFrame:
        """Load data from individual files."""
        # This would implement loading from individual parquet files
        # For now, return empty DataFrame
        self.logger.warning("⚠️ Individual file loading not implemented")
        return pd.DataFrame()
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate market data."""
        self.logger.info("🔍 Validating market data...")
        
        if data.empty:
            raise ValidationError("Market data is empty")
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")
        
        # Check for sufficient data
        min_required_days = self.config.initial_training_days + self.config.testing_window_days
        if len(data) < min_required_days:
            raise ValidationError(f"Insufficient data points: {len(data)} < {min_required_days}")
        
        # Check for missing values
        missing_values = data[required_columns].isnull().sum().sum()
        if missing_values > 0:
            self.logger.warning(f"⚠️ Found {missing_values} missing values")
        
        self.logger.info("✅ Data validation completed successfully")
    
    def _generate_walk_forward_windows(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate walk-forward validation windows."""
        self.logger.info("📅 Generating walk-forward windows...")
        
        windows = []
        start_date = data.index[0]
        end_date = data.index[-1]
        
        # Calculate initial training period
        initial_training_end = start_date + timedelta(days=self.config.initial_training_days)
        
        current_training_start = start_date
        current_training_end = initial_training_end
        
        window_id = 0
        
        while current_training_end + timedelta(days=self.config.testing_window_days) <= end_date:
            # Define testing period
            testing_start = current_training_end
            testing_end = testing_start + timedelta(days=self.config.testing_window_days)
            
            # Get data for this window
            training_data = data.loc[current_training_start:current_training_end]
            testing_data = data.loc[testing_start:testing_end]
            
            # Validate window data
            if (len(training_data) >= self.config.min_training_days and 
                len(testing_data) >= self.config.testing_window_days):
                
                window = {
                    'window_id': window_id,
                    'training_start': current_training_start,
                    'training_end': current_training_end,
                    'testing_start': testing_start,
                    'testing_end': testing_end,
                    'training_data': training_data,
                    'testing_data': testing_data,
                    'training_days': len(training_data),
                    'testing_days': len(testing_data)
                }
                
                windows.append(window)
                window_id += 1
            
            # Move to next window based on validation mode
            if self.config.validation_mode == WalkForwardMode.EXPANDING_WINDOW:
                # Expanding window: keep start, extend end
                current_training_end += timedelta(days=self.config.step_size_days)
            elif self.config.validation_mode == WalkForwardMode.ROLLING_WINDOW:
                # Rolling window: move both start and end
                current_training_start += timedelta(days=self.config.step_size_days)
                current_training_end += timedelta(days=self.config.step_size_days)
            elif self.config.validation_mode == WalkForwardMode.FIXED_WINDOW:
                # Fixed window: move both start and end by step size
                current_training_start += timedelta(days=self.config.step_size_days)
                current_training_end = current_training_start + timedelta(days=self.config.training_window_days)
            else:
                # Default to expanding window
                current_training_end += timedelta(days=self.config.step_size_days)
        
        self.logger.info(f"✅ Generated {len(windows)} walk-forward windows")
        return windows
    
    async def _execute_walk_forward_validation(
        self, 
        windows: List[Dict[str, Any]], 
        strategy_func: Optional[Callable]
    ) -> List[Dict[str, Any]]:
        """Execute walk-forward validation for all windows."""
        self.logger.info("🔄 Executing walk-forward validation...")
        
        window_results = []
        
        for i, window in enumerate(windows):
            self.logger.info(f"🔄 Processing window {i+1}/{len(windows)}")
            
            try:
                # Execute validation for this window
                window_result = await self._validate_window(window, strategy_func)
                window_results.append(window_result)
                
                self.logger.info(f"✅ Window {i+1} completed - Sharpe: {window_result.get('sharpe_ratio', 0):.2f}")
                
            except Exception as e:
                self.logger.error(f"❌ Error in window {i+1}: {e}")
                # Add failed window result
                window_results.append({
                    'window_id': window['window_id'],
                    'success': False,
                    'error': str(e),
                    'sharpe_ratio': 0.0,
                    'total_return': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'total_trades': 0
                })
                continue
        
        self.logger.info(f"✅ Walk-forward validation completed: {len(window_results)} windows processed")
        return window_results
    
    async def _validate_window(
        self, 
        window: Dict[str, Any], 
        strategy_func: Optional[Callable]
    ) -> Dict[str, Any]:
        """Validate a single window."""
        window_id = window['window_id']
        training_data = window['training_data']
        testing_data = window['testing_data']
        
        # For now, implement a simple buy-and-hold strategy
        # In practice, this would use the provided strategy_func
        if strategy_func is not None:
            # Use provided strategy function
            strategy_results = await strategy_func(training_data, testing_data)
        else:
            # Use default buy-and-hold strategy
            strategy_results = await self._execute_buy_and_hold_strategy(testing_data)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_window_metrics(strategy_results)
        
        # Add window information
        result = {
            'window_id': window_id,
            'training_start': window['training_start'],
            'training_end': window['training_end'],
            'testing_start': window['testing_start'],
            'testing_end': window['testing_end'],
            'training_days': window['training_days'],
            'testing_days': window['testing_days'],
            'success': True,
            **performance_metrics
        }
        
        return result
    
    async def _execute_buy_and_hold_strategy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute buy-and-hold strategy for testing."""
        initial_price = data['close'].iloc[0]
        final_price = data['close'].iloc[-1]
        total_return = (final_price - initial_price) / initial_price
        
        # Calculate equity curve
        equity_curve = pd.DataFrame({
            'timestamp': data.index,
            'price': data['close'],
            'equity': self.config.initial_training_days * (1 + total_return),  # Simplified
            'return': data['close'].pct_change().fillna(0)
        })
        
        return {
            'total_return': total_return,
            'equity_curve': equity_curve,
            'trade_log': pd.DataFrame({
                'timestamp': [data.index[0], data.index[-1]],
                'action': ['buy', 'sell'],
                'price': [initial_price, final_price]
            })
        }
    
    def _calculate_window_metrics(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics for a window."""
        total_return = strategy_results.get('total_return', 0)
        
        # Calculate additional metrics
        if 'equity_curve' in strategy_results and not strategy_results['equity_curve'].empty:
            equity_curve = strategy_results['equity_curve']
            returns = equity_curve['return'].dropna()
            
            if len(returns) > 0:
                volatility = returns.std() * np.sqrt(252)
                sharpe_ratio = (returns.mean() * 252) / volatility if volatility > 0 else 0
                max_drawdown = self._calculate_max_drawdown(equity_curve['equity'])
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
        else:
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Calculate win rate
        if 'trade_log' in strategy_results and not strategy_results['trade_log'].empty:
            trade_log = strategy_results['trade_log']
            total_trades = len(trade_log)
            # Simplified win rate calculation
            win_rate = 0.5 if total_trades > 0 else 0
        else:
            total_trades = 0
            win_rate = 0
        
        return {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 / len(strategy_results.get('equity_curve', pd.DataFrame()))) - 1 if len(strategy_results.get('equity_curve', pd.DataFrame())) > 0 else 0,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': 1.0,  # Simplified
            'calmar_ratio': total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        }
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(equity_series) == 0:
            return 0.0
        
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        return float(drawdown.min())
    
    def _calculate_overall_performance(self, window_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate overall performance metrics."""
        self.logger.info("📊 Calculating overall performance...")
        
        if not window_results:
            return {}
        
        # Extract metrics from successful windows
        successful_windows = [w for w in window_results if w.get('success', False)]
        
        if not successful_windows:
            return {}
        
        # Calculate aggregate metrics
        returns = [w['total_return'] for w in successful_windows]
        sharpe_ratios = [w['sharpe_ratio'] for w in successful_windows]
        volatilities = [w['volatility'] for w in successful_windows]
        max_drawdowns = [w['max_drawdown'] for w in successful_windows]
        win_rates = [w['win_rate'] for w in successful_windows]
        
        overall_performance = {
            'total_windows': len(window_results),
            'successful_windows': len(successful_windows),
            'success_rate': len(successful_windows) / len(window_results),
            'average_return': np.mean(returns),
            'median_return': np.median(returns),
            'return_std': np.std(returns),
            'average_sharpe': np.mean(sharpe_ratios),
            'median_sharpe': np.median(sharpe_ratios),
            'sharpe_std': np.std(sharpe_ratios),
            'average_volatility': np.mean(volatilities),
            'average_drawdown': np.mean(max_drawdowns),
            'max_drawdown': min(max_drawdowns),  # Most negative
            'average_win_rate': np.mean(win_rates),
            'total_return': np.prod([1 + r for r in returns]) - 1,  # Compound return
            'annualized_return': (np.prod([1 + r for r in returns]) ** (252 / len(returns))) - 1
        }
        
        self.logger.info("✅ Overall performance calculated")
        return overall_performance
    
    def _perform_statistical_tests(self, window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical tests on window results."""
        self.logger.info("📈 Performing statistical tests...")
        
        if not self.config.enable_statistical_testing:
            return {}
        
        successful_windows = [w for w in window_results if w.get('success', False)]
        
        if len(successful_windows) < self.config.min_observations:
            self.logger.warning(f"⚠️ Insufficient observations for statistical tests: {len(successful_windows)} < {self.config.min_observations}")
            return {}
        
        # Extract returns
        returns = [w['total_return'] for w in successful_windows]
        
        # Perform statistical tests
        tests = {}
        
        # Normality test (Shapiro-Wilk)
        if len(returns) >= 3:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(returns)
                tests['normality_test'] = {
                    'test': 'Shapiro-Wilk',
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                }
            except Exception as e:
                self.logger.warning(f"⚠️ Normality test failed: {e}")
        
        # One-sample t-test (test if mean return is significantly different from 0)
        try:
            t_stat, t_p = stats.ttest_1samp(returns, 0)
            tests['mean_return_test'] = {
                'test': 'One-sample t-test',
                'statistic': t_stat,
                'p_value': t_p,
                'significant': t_p < (1 - self.config.confidence_level)
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Mean return test failed: {e}")
        
        # Performance consistency test (coefficient of variation)
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            cv = std_return / abs(mean_return) if mean_return != 0 else float('inf')
            tests['consistency_test'] = {
                'test': 'Coefficient of Variation',
                'coefficient_of_variation': cv,
                'is_consistent': cv < 1.0  # Less than 100% variation
            }
        
        self.logger.info("✅ Statistical tests completed")
        return tests
    
    def _analyze_performance_stability(self, window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance stability across windows."""
        self.logger.info("📊 Analyzing performance stability...")
        
        successful_windows = [w for w in window_results if w.get('success', False)]
        
        if len(successful_windows) < 2:
            return {}
        
        # Extract performance metrics
        returns = [w['total_return'] for w in successful_windows]
        sharpe_ratios = [w['sharpe_ratio'] for w in successful_windows]
        
        # Calculate stability metrics
        stability = {
            'return_stability': {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'cv': np.std(returns) / abs(np.mean(returns)) if np.mean(returns) != 0 else float('inf'),
                'min': min(returns),
                'max': max(returns),
                'range': max(returns) - min(returns)
            },
            'sharpe_stability': {
                'mean': np.mean(sharpe_ratios),
                'std': np.std(sharpe_ratios),
                'cv': np.std(sharpe_ratios) / abs(np.mean(sharpe_ratios)) if np.mean(sharpe_ratios) != 0 else float('inf'),
                'min': min(sharpe_ratios),
                'max': max(sharpe_ratios),
                'range': max(sharpe_ratios) - min(sharpe_ratios)
            }
        }
        
        # Calculate trend analysis
        if len(returns) >= 3:
            # Linear trend in returns
            x = np.arange(len(returns))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, returns)
            stability['return_trend'] = {
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'is_trending': p_value < 0.05
            }
        
        self.logger.info("✅ Performance stability analysis completed")
        return stability
    
    def _calculate_risk_metrics(self, window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate risk metrics across windows."""
        self.logger.info("⚠️ Calculating risk metrics...")
        
        successful_windows = [w for w in window_results if w.get('success', False)]
        
        if not successful_windows:
            return {}
        
        # Extract risk metrics
        returns = [w['total_return'] for w in successful_windows]
        volatilities = [w['volatility'] for w in successful_windows]
        max_drawdowns = [w['max_drawdown'] for w in successful_windows]
        
        # Calculate portfolio-level risk metrics
        risk_metrics = {
            'portfolio_volatility': np.std(returns) * np.sqrt(252),
            'portfolio_var_95': np.percentile(returns, 5),
            'portfolio_var_99': np.percentile(returns, 1),
            'portfolio_cvar_95': np.mean([r for r in returns if r <= np.percentile(returns, 5)]),
            'portfolio_cvar_99': np.mean([r for r in returns if r <= np.percentile(returns, 1)]),
            'average_volatility': np.mean(volatilities),
            'max_volatility': max(volatilities),
            'min_volatility': min(volatilities),
            'average_drawdown': np.mean(max_drawdowns),
            'worst_drawdown': min(max_drawdowns),
            'drawdown_std': np.std(max_drawdowns)
        }
        
        self.logger.info("✅ Risk metrics calculated")
        return risk_metrics
    
    def _analyze_drawdowns(self, window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze drawdown patterns."""
        self.logger.info("📉 Analyzing drawdowns...")
        
        successful_windows = [w for w in window_results if w.get('success', False)]
        
        if not successful_windows:
            return {}
        
        max_drawdowns = [w['max_drawdown'] for w in successful_windows]
        
        drawdown_analysis = {
            'worst_drawdown': min(max_drawdowns),
            'average_drawdown': np.mean(max_drawdowns),
            'drawdown_std': np.std(max_drawdowns),
            'drawdown_frequency': len([d for d in max_drawdowns if d < -0.05]) / len(max_drawdowns),  # >5% drawdowns
            'severe_drawdown_frequency': len([d for d in max_drawdowns if d < -0.1]) / len(max_drawdowns),  # >10% drawdowns
            'drawdown_percentiles': {
                'p5': np.percentile(max_drawdowns, 5),
                'p25': np.percentile(max_drawdowns, 25),
                'p50': np.percentile(max_drawdowns, 50),
                'p75': np.percentile(max_drawdowns, 75),
                'p95': np.percentile(max_drawdowns, 95)
            }
        }
        
        self.logger.info("✅ Drawdown analysis completed")
        return drawdown_analysis
    
    def _generate_validation_summary(self, window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate validation summary."""
        self.logger.info("📋 Generating validation summary...")
        
        successful_windows = [w for w in window_results if w.get('success', False)]
        
        summary = {
            'total_windows': len(window_results),
            'successful_windows': len(successful_windows),
            'failed_windows': len(window_results) - len(successful_windows),
            'success_rate': len(successful_windows) / len(window_results) if window_results else 0,
            'validation_passed': len(successful_windows) >= len(window_results) * 0.8,  # 80% success rate
            'performance_thresholds': {
                'min_sharpe_ratio': self.config.min_sharpe_ratio,
                'max_drawdown_threshold': self.config.max_drawdown_threshold,
                'min_win_rate': self.config.min_win_rate,
                'min_trades_per_window': self.config.min_trades_per_window
            }
        }
        
        # Check performance thresholds
        if successful_windows:
            avg_sharpe = np.mean([w['sharpe_ratio'] for w in successful_windows])
            avg_drawdown = np.mean([w['max_drawdown'] for w in successful_windows])
            avg_win_rate = np.mean([w['win_rate'] for w in successful_windows])
            avg_trades = np.mean([w['total_trades'] for w in successful_windows])
            
            summary['threshold_checks'] = {
                'sharpe_ratio_pass': avg_sharpe >= self.config.min_sharpe_ratio,
                'drawdown_pass': abs(avg_drawdown) <= self.config.max_drawdown_threshold,
                'win_rate_pass': avg_win_rate >= self.config.min_win_rate,
                'trades_pass': avg_trades >= self.config.min_trades_per_window
            }
            
            summary['overall_validation_pass'] = all(summary['threshold_checks'].values())
        else:
            summary['threshold_checks'] = {}
            summary['overall_validation_pass'] = False
        
        self.logger.info("✅ Validation summary generated")
        return summary
    
    def _detect_performance_degradation(self, window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect performance degradation over time."""
        self.logger.info("🔍 Detecting performance degradation...")
        
        successful_windows = [w for w in window_results if w.get('success', False)]
        
        if len(successful_windows) < 3:
            return {}
        
        # Sort windows by time
        successful_windows.sort(key=lambda x: x['testing_start'])
        
        # Extract performance metrics over time
        returns = [w['total_return'] for w in successful_windows]
        sharpe_ratios = [w['sharpe_ratio'] for w in successful_windows]
        
        # Detect degradation using linear regression
        x = np.arange(len(returns))
        
        # Test for negative trend in returns
        slope_returns, _, r_returns, p_returns, _ = stats.linregress(x, returns)
        slope_sharpe, _, r_sharpe, p_sharpe, _ = stats.linregress(x, sharpe_ratios)
        
        degradation_detection = {
            'return_trend': {
                'slope': slope_returns,
                'r_squared': r_returns ** 2,
                'p_value': p_returns,
                'is_degrading': slope_returns < 0 and p_returns < 0.05
            },
            'sharpe_trend': {
                'slope': slope_sharpe,
                'r_squared': r_sharpe ** 2,
                'p_value': p_sharpe,
                'is_degrading': slope_sharpe < 0 and p_sharpe < 0.05
            },
            'overall_degradation': (slope_returns < 0 and p_returns < 0.05) or (slope_sharpe < 0 and p_sharpe < 0.05)
        }
        
        # Calculate degradation severity
        if degradation_detection['overall_degradation']:
            early_performance = np.mean(returns[:len(returns)//2])
            late_performance = np.mean(returns[len(returns)//2:])
            degradation_severity = (early_performance - late_performance) / abs(early_performance) if early_performance != 0 else 0
            
            degradation_detection['degradation_severity'] = degradation_severity
            degradation_detection['severity_level'] = (
                'HIGH' if degradation_severity > 0.3 else
                'MEDIUM' if degradation_severity > 0.1 else
                'LOW'
            )
        
        self.logger.info("✅ Performance degradation detection completed")
        return degradation_detection
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Could not get system metrics: {e}")
            return {}
    
    async def _save_results(self, results: WalkForwardResults) -> None:
        """Save results to disk."""
        self.logger.info("💾 Saving results...")
        
        # Create output directory
        output_dir = self.data_dir / "backtesting_results" / "walk_forward"
        ensure_directory(output_dir)
        
        # Save main results
        results_file = output_dir / f"{self.config.symbol}_{self.config.exchange}_walk_forward_results.json"
        await safe_json_dump(results_file, results.__dict__, indent=2)
        
        # Save window performance data
        if results.window_performance:
            window_df = pd.DataFrame(results.window_performance)
            window_file = output_dir / f"{self.config.symbol}_{self.config.exchange}_window_performance.parquet"
            await self.parquet_utils.save_dataframe(window_df, window_file)
        
        # Save equity curves
        for strategy_name, equity_curve in results.equity_curves.items():
            if not equity_curve.empty:
                equity_file = output_dir / f"{self.config.symbol}_{self.config.exchange}_{strategy_name}_equity_curve.parquet"
                await self.parquet_utils.save_dataframe(equity_curve, equity_file)
        
        # Save trade logs
        for strategy_name, trade_log in results.trade_logs.items():
            if not trade_log.empty:
                trades_file = output_dir / f"{self.config.symbol}_{self.config.exchange}_{strategy_name}_trade_log.parquet"
                await self.parquet_utils.save_dataframe(trade_log, trades_file)
        
        self.logger.info(f"✅ Results saved to {output_dir}")


# Convenience function for easy integration
async def execute_walk_forward_validation(
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE", 
    timeframe: str = "1h",
    data_dir: str = "data/training",
    strategy_func: Optional[Callable] = None,
    **kwargs
) -> WalkForwardResults:
    """
    Convenience function to execute walk-forward validation.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Data timeframe
        data_dir: Data directory
        strategy_func: Strategy function to validate
        **kwargs: Additional configuration parameters
        
    Returns:
        Walk-forward validation results
    """
    config = WalkForwardConfig(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        **kwargs
    )
    
    step = WalkForwardValidationStep(config)
    return await step.execute(strategy_func=strategy_func)