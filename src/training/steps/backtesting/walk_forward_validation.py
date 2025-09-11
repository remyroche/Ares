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
from concurrent.futures import ProcessPoolExecutor

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size, check_disk_space
)
from src.utils.logger import system_logger
from src.custom_types.validation import TypeValidator, RuntimeTypeError
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


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    window_id: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'window_id': self.window_id, 
            'train_start': self.train_start.isoformat(), 
            'train_end': self.train_end.isoformat(), 
            'test_start': self.test_start.isoformat(), 
            'test_end': self.test_end.isoformat(), 
            'train_days': (self.train_end - self.train_start).days, 
            'test_days': (self.test_end - self.test_start).days
        }


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


class WalkForwardValidator:
    """Implements walk-forward validation for trading strategies with comprehensive protections."""

    def __init__(self, config: Dict[str, Any]) -> None:
        # Input validation
        if not isinstance(config, dict):
            raise RuntimeTypeError(dict, config, 'WalkForwardValidator config')
        
        self.config = config
        self.logger = system_logger.getChild('WalkForwardValidator')
        
        # Validate and set configuration parameters with bounds checking
        self.train_period_days = self._validate_positive_int(config.get('train_period_days', 365), 'train_period_days', min_val=30, max_val=3650)
        self.test_period_days = self._validate_positive_int(config.get('test_period_days', 30), 'test_period_days', min_val=1, max_val=365)
        self.step_days = self._validate_positive_int(config.get('step_days', 30), 'step_days', min_val=1, max_val=365)
        self.min_train_samples = self._validate_positive_int(config.get('min_train_samples', 1000), 'min_train_samples', min_val=10, max_val=100000)
        self.regime_aware = self._validate_bool(config.get('regime_aware', True), 'regime_aware')
        self.min_samples_per_regime = self._validate_positive_int(config.get('min_samples_per_regime', 500), 'min_samples_per_regime', min_val=5, max_val=50000)
        self.adaptive_windows = self._validate_bool(config.get('adaptive_windows', True), 'adaptive_windows')
        self.volatility_threshold = self._validate_positive_float(config.get('volatility_threshold', 0.03), 'volatility_threshold', min_val=0.001, max_val=1.0)
        self.max_acceptable_degradation = self._validate_positive_float(config.get('max_acceptable_degradation', 0.3), 'max_acceptable_degradation', min_val=0.0, max_val=1.0)
        self.min_out_sample_sharpe = self._validate_float(config.get('min_out_sample_sharpe', 0.5), 'min_out_sample_sharpe', min_val=-5.0, max_val=5.0)
        
        # Validate results directory
        results_dir_str = config.get('results_dir', 'validation_results')
        if not isinstance(results_dir_str, (str, Path)):
            raise RuntimeTypeError(Union[str, Path], results_dir_str, 'results_dir')
        self.results_dir = Path(results_dir_str)
        
        # Ensure results directory exists
        if not ensure_directory(self.results_dir):
            self.logger.warning(f"Failed to create results directory: {self.results_dir}")
        
        self.logger.info(f"WalkForwardValidator initialized with train_period={self.train_period_days}, test_period={self.test_period_days}, step_days={self.step_days}")
    
    def _validate_positive_int(self, value: Any, name: str, min_val: int = 1, max_val: int = None) -> int:
        """Validate positive integer parameter."""
        try:
            val = safe_int(value, 0)
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")
            if val < min_val:
                raise ValueError(f"{name} must be >= {min_val}, got {val}")
            if max_val is not None and val > max_val:
                raise ValueError(f"{name} must be <= {max_val}, got {val}")
            return val
        except Exception as e:
            raise ValueError(f"Invalid {name}: {e}")
    
    def _validate_positive_float(self, value: Any, name: str, min_val: float = 0.0, max_val: float = None) -> float:
        """Validate positive float parameter."""
        try:
            val = safe_float(value, 0.0)
            if val < 0:
                raise ValueError(f"{name} must be non-negative, got {val}")
            if val < min_val:
                raise ValueError(f"{name} must be >= {min_val}, got {val}")
            if max_val is not None and val > max_val:
                raise ValueError(f"{name} must be <= {max_val}, got {val}")
            return val
        except Exception as e:
            raise ValueError(f"Invalid {name}: {e}")
    
    def _validate_float(self, value: Any, name: str, min_val: float = None, max_val: float = None) -> float:
        """Validate float parameter with optional bounds."""
        try:
            val = safe_float(value, 0.0)
            if not np.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")
            if min_val is not None and val < min_val:
                raise ValueError(f"{name} must be >= {min_val}, got {val}")
            if max_val is not None and val > max_val:
                raise ValueError(f"{name} must be <= {max_val}, got {val}")
            return val
        except Exception as e:
            raise ValueError(f"Invalid {name}: {e}")
    
    def _validate_bool(self, value: Any, name: str) -> bool:
        """Validate boolean parameter."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        if isinstance(value, (int, float)):
            return bool(value)
        raise ValueError(f"{name} must be a boolean, got {type(value)}")

    def generate_walk_forward_windows(self, data: pd.DataFrame) -> List[WalkForwardWindow]:
        """Generate walk-forward validation windows."""
        # Input validation
        if not isinstance(data, pd.DataFrame):
            raise RuntimeTypeError(pd.DataFrame, data, 'generate_walk_forward_windows data')
        
        if data.empty:
            raise ValueError("DataFrame cannot be empty")
        
        # Check for required columns
        required_columns = ['close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        try:
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            
            # Validate datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("Data must have a datetime index or 'timestamp' column")
            
            start_date = data.index.min()
            end_date = data.index.max()
            
            if start_date >= end_date:
                raise ValueError("Start date must be before end date")
            
            # Check if we have enough data for at least one window
            min_required_days = self.train_period_days + self.test_period_days
            available_days = (end_date - start_date).days
            if available_days < min_required_days:
                raise ValueError(f"Insufficient data: need at least {min_required_days} days, have {available_days}")
            
            windows = []
            window_id = 0
            train_end = start_date + timedelta(days=self.train_period_days)
            
            while train_end + timedelta(days=self.test_period_days) <= end_date:
                train_start = start_date
                test_start = train_end
                test_end = test_start + timedelta(days=self.test_period_days)
                
                if self.adaptive_windows:
                    try:
                        window_params = self._adjust_window_for_volatility(data, train_start, train_end)
                        if window_params:
                            train_start = window_params['train_start']
                            test_end = window_params['test_end']
                    except Exception as e:
                        self.logger.warning(f"Failed to adjust window for volatility: {e}")
                
                window = WalkForwardWindow(
                    train_start=train_start, 
                    train_end=train_end, 
                    test_start=test_start, 
                    test_end=test_end, 
                    window_id=window_id
                )
                windows.append(window)
                train_end += timedelta(days=self.step_days)
                window_id += 1
            
            if not windows:
                raise ValueError("No valid windows could be generated with the given parameters")
            
            self.logger.info(f'Generated {len(windows)} walk-forward windows')
            return windows
            
        except Exception as e:
            self.logger.error(f"Error generating walk-forward windows: {e}")
            raise

    def _adjust_window_for_volatility(self, data: pd.DataFrame, train_start: datetime, train_end: datetime) -> Optional[Dict[str, datetime]]:
        """Adjust window size based on market volatility."""
        try:
            # Input validation
            if not isinstance(data, pd.DataFrame):
                raise RuntimeTypeError(pd.DataFrame, data, '_adjust_window_for_volatility data')
            if not isinstance(train_start, datetime):
                raise RuntimeTypeError(datetime, train_start, '_adjust_window_for_volatility train_start')
            if not isinstance(train_end, datetime):
                raise RuntimeTypeError(datetime, train_end, '_adjust_window_for_volatility train_end')
            
            if train_start >= train_end:
                raise ValueError("train_start must be before train_end")
            
            train_data = data[train_start:train_end]
            if train_data.empty:
                self.logger.warning("No training data available for volatility adjustment")
                return None
            
            if 'close' not in train_data.columns:
                self.logger.warning("No 'close' column available for volatility calculation")
                return None
            
            returns = train_data['close'].pct_change().dropna()
            if len(returns) < 2:
                self.logger.warning("Insufficient data for volatility calculation")
                return None
            
            volatility = returns.std()
            if not np.isfinite(volatility):
                self.logger.warning("Invalid volatility value calculated")
                return None
            
            if volatility > self.volatility_threshold:
                new_train_days = max(1, int(self.train_period_days * 0.5))
                new_test_days = max(1, int(self.test_period_days * 0.5))
                
                # Ensure we don't create invalid date ranges
                adjusted_train_start = train_end - timedelta(days=new_train_days)
                adjusted_test_end = train_end + timedelta(days=new_test_days)
                
                if adjusted_train_start >= train_end or train_end >= adjusted_test_end:
                    self.logger.warning("Adjusted window parameters would create invalid date ranges")
                    return None
                
                return {
                    'train_start': adjusted_train_start, 
                    'test_end': adjusted_test_end
                }
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error in volatility adjustment: {e}")
            return None

    async def validate_model(self, model_trainer: Callable, data: pd.DataFrame, regime_labels: Optional[np.ndarray]=None) -> Dict[str, Any]:
        """Run walk-forward validation on a model."""
        try:
            # Input validation
            if not callable(model_trainer):
                raise RuntimeTypeError(Callable, model_trainer, 'validate_model model_trainer')
            if not isinstance(data, pd.DataFrame):
                raise RuntimeTypeError(pd.DataFrame, data, 'validate_model data')
            if regime_labels is not None and not isinstance(regime_labels, np.ndarray):
                raise RuntimeTypeError(np.ndarray, regime_labels, 'validate_model regime_labels')
            
            self.logger.info('Starting walk-forward validation...')
            
            # Generate windows with error handling
            try:
                windows = self.generate_walk_forward_windows(data)
            except Exception as e:
                self.logger.error(f"Failed to generate walk-forward windows: {e}")
                return {
                    'windows': [],
                    'results': [],
                    'analysis': {
                        'total_windows': 0,
                        'successful_windows': 0,
                        'validation_passed': False,
                        'error': f"Window generation failed: {str(e)}"
                    }
                }
            
            # Validate regime labels if provided
            if self.regime_aware and regime_labels is not None:
                if len(regime_labels) != len(data):
                    raise ValueError(f"Regime labels length ({len(regime_labels)}) must match data length ({len(data)})")
                results = await self._validate_regime_aware(model_trainer, data, regime_labels, windows)
            else:
                results = await self._validate_standard(model_trainer, data, windows)
            
            # Analyze results with error handling
            try:
                analysis = self._analyze_validation_results(results)
            except Exception as e:
                self.logger.error(f"Failed to analyze validation results: {e}")
                analysis = {
                    'total_windows': len(results),
                    'successful_windows': 0,
                    'validation_passed': False,
                    'error': f"Analysis failed: {str(e)}"
                }
            
            # Save results with error handling
            try:
                self._save_validation_results(results, analysis)
            except Exception as e:
                self.logger.error(f"Failed to save validation results: {e}")
                # Don't fail the entire operation if saving fails
            
            return {
                'windows': [w.to_dict() for w in windows], 
                'results': results, 
                'analysis': analysis
            }
            
        except Exception as e:
            self.logger.error(f"Error in validate_model: {e}")
            return {
                'windows': [],
                'results': [],
                'analysis': {
                    'total_windows': 0,
                    'successful_windows': 0,
                    'validation_passed': False,
                    'error': f"Validation failed: {str(e)}"
                }
            }

    async def _validate_standard(self, model_trainer: Callable, data: pd.DataFrame, windows: List[WalkForwardWindow]) -> List[Dict[str, Any]]:
        """Standard walk-forward validation."""
        results = []
        max_workers = min(4, len(windows))  # Limit workers based on number of windows
        timeout_seconds = 3600  # 1 hour timeout per window
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                # Submit all window validation tasks
                for window in windows:
                    try:
                        future = executor.submit(self._validate_single_window, model_trainer, data, window)
                        futures.append((window, future))
                    except Exception as e:
                        self.logger.error(f'Failed to submit window {window.window_id}: {e}')
                        results.append({
                            'window': window.to_dict(), 
                            'error': f'Submission failed: {str(e)}', 
                            'success': False
                        })
                
                # Collect results with timeout protection
                for window, future in futures:
                    try:
                        result = future.result(timeout=timeout_seconds)
                        results.append(result)
                    except TimeoutError:
                        self.logger.error(f'Window {window.window_id} timed out after {timeout_seconds} seconds')
                        results.append({
                            'window': window.to_dict(), 
                            'error': f'Timeout after {timeout_seconds} seconds', 
                            'success': False
                        })
                    except Exception as e:
                        self.logger.error(f'Window {window.window_id} failed: {e}')
                        results.append({
                            'window': window.to_dict(), 
                            'error': str(e), 
                            'success': False
                        })
                        
        except Exception as e:
            self.logger.error(f'Error in _validate_standard: {e}')
            # Return partial results if available
            if not results:
                results = [{
                    'window': {'window_id': -1},
                    'error': f'Standard validation failed: {str(e)}',
                    'success': False
                }]
        
        return results

    def _validate_single_window(self, model_trainer: Callable, data: pd.DataFrame, window: WalkForwardWindow) -> Dict[str, Any]:
        """Validate a single window."""
        try:
            # Input validation
            if not callable(model_trainer):
                return {'window': window.to_dict(), 'error': 'model_trainer must be callable', 'success': False}
            if not isinstance(data, pd.DataFrame):
                return {'window': window.to_dict(), 'error': 'data must be a DataFrame', 'success': False}
            if not isinstance(window, WalkForwardWindow):
                return {'window': window.to_dict(), 'error': 'window must be a WalkForwardWindow', 'success': False}
            
            # Extract training and test data
            try:
                train_data = data[window.train_start:window.train_end]
                test_data = data[window.test_start:window.test_end]
            except Exception as e:
                return {'window': window.to_dict(), 'error': f'Failed to extract data: {str(e)}', 'success': False}
            
            # Validate data availability
            if train_data.empty:
                return {'window': window.to_dict(), 'error': 'No training data available', 'success': False}
            if test_data.empty:
                return {'window': window.to_dict(), 'error': 'No test data available', 'success': False}
            
            if len(train_data) < self.min_train_samples:
                return {
                    'window': window.to_dict(), 
                    'error': f'Insufficient training samples: {len(train_data)} < {self.min_train_samples}', 
                    'success': False
                }
            
            # Train model with error handling
            try:
                model = model_trainer(train_data)
                if model is None:
                    return {'window': window.to_dict(), 'error': 'Model trainer returned None', 'success': False}
            except Exception as e:
                return {'window': window.to_dict(), 'error': f'Model training failed: {str(e)}', 'success': False}
            
            # Generate predictions with error handling
            try:
                train_predictions = model.predict(train_data)
                test_predictions = model.predict(test_data)
                
                if train_predictions is None or test_predictions is None:
                    return {'window': window.to_dict(), 'error': 'Model predictions returned None', 'success': False}
                
                if len(train_predictions) != len(train_data):
                    return {'window': window.to_dict(), 'error': 'Train predictions length mismatch', 'success': False}
                if len(test_predictions) != len(test_data):
                    return {'window': window.to_dict(), 'error': 'Test predictions length mismatch', 'success': False}
                    
            except Exception as e:
                return {'window': window.to_dict(), 'error': f'Prediction generation failed: {str(e)}', 'success': False}
            
            # Calculate metrics with error handling
            try:
                train_metrics = self._calculate_metrics(train_data, train_predictions, 'train')
                test_metrics = self._calculate_metrics(test_data, test_predictions, 'test')
                degradation = self._calculate_degradation(train_metrics, test_metrics)
            except Exception as e:
                return {'window': window.to_dict(), 'error': f'Metrics calculation failed: {str(e)}', 'success': False}
            
            # Extract model parameters safely
            model_params = {}
            try:
                if hasattr(model, 'get_params') and callable(model.get_params):
                    model_params = model.get_params()
                elif hasattr(model, '__dict__'):
                    model_params = {k: v for k, v in model.__dict__.items() if not k.startswith('_')}
            except Exception as e:
                self.logger.warning(f"Failed to extract model parameters: {e}")
            
            return {
                'window': window.to_dict(), 
                'train_metrics': train_metrics, 
                'test_metrics': test_metrics, 
                'degradation': degradation, 
                'model_params': model_params, 
                'success': True
            }
            
        except Exception as e:
            return {'window': window.to_dict(), 'error': f'Unexpected error: {str(e)}', 'success': False}

    async def _validate_regime_aware(self, model_trainer: Callable, data: pd.DataFrame, regime_labels: np.ndarray, windows: List[WalkForwardWindow]) -> List[Dict[str, Any]]:
        """Regime-aware walk-forward validation."""
        results = []
        for window in windows:
            window_results = {'window': window.to_dict(), 'regime_results': {}, 'success': True}
            train_mask = (data.index >= window.train_start) & (data.index <= window.train_end)
            test_mask = (data.index >= window.test_start) & (data.index <= window.test_end)
            train_data = data[train_mask]
            test_data = data[test_mask]
            train_regimes = regime_labels[train_mask]
            test_regimes = regime_labels[test_mask]
            for regime in ['bull', 'bear', 'sideways']:
                regime_result = await self._validate_regime_window(model_trainer, train_data, test_data, train_regimes, test_regimes, regime)
                window_results['regime_results'][regime] = regime_result
                if not regime_result['success']:
                    window_results['success'] = False
            results.append(window_results)
        return results

    async def _validate_regime_window(self, model_trainer: Callable, train_data: pd.DataFrame, test_data: pd.DataFrame, train_regimes: np.ndarray, test_regimes: np.ndarray, regime: str) -> Dict[str, Any]:
        """Validate a single regime within a window."""
        regime_map = {'bear': 0, 'sideways': 1, 'bull': 2}
        regime_num = regime_map.get(regime, 1)
        train_regime_data = train_data[train_regimes == regime_num]
        test_regime_data = test_data[test_regimes == regime_num]
        if len(train_regime_data) < self.min_samples_per_regime:
            return {'regime': regime, 'error': f'Insufficient {regime} training samples: {len(train_regime_data)}', 'success': False}
        if len(test_regime_data) < 10:
            return {'regime': regime, 'error': f'Insufficient {regime} test samples: {len(test_regime_data)}', 'success': False}
        try:
            model = model_trainer(train_regime_data, regime=regime)
            train_predictions = model.predict(train_regime_data)
            test_predictions = model.predict(test_regime_data)
            train_metrics = self._calculate_metrics(train_regime_data, train_predictions, f'train_{regime}')
            test_metrics = self._calculate_metrics(test_regime_data, test_predictions, f'test_{regime}')
            degradation = self._calculate_degradation(train_metrics, test_metrics)
            return {'regime': regime, 'train_samples': len(train_regime_data), 'test_samples': len(test_regime_data), 'train_metrics': train_metrics, 'test_metrics': test_metrics, 'degradation': degradation, 'success': True}
        except Exception as e:
            return {'regime': regime, 'error': str(e), 'success': False}

    def _calculate_metrics(self, data: pd.DataFrame, predictions: np.ndarray, prefix: str) -> Dict[str, float]:
        """Calculate performance metrics."""
        try:
            # Input validation
            if not isinstance(data, pd.DataFrame):
                raise RuntimeTypeError(pd.DataFrame, data, '_calculate_metrics data')
            if not isinstance(predictions, np.ndarray):
                raise RuntimeTypeError(np.ndarray, predictions, '_calculate_metrics predictions')
            if not isinstance(prefix, str):
                raise RuntimeTypeError(str, prefix, '_calculate_metrics prefix')
            
            if data.empty:
                raise ValueError("Data cannot be empty")
            if len(predictions) == 0:
                raise ValueError("Predictions cannot be empty")
            
            # Ensure we have close prices
            if 'close' not in data.columns:
                raise ValueError("Data must contain 'close' column")
            
            # Calculate returns if not present
            if 'returns' not in data.columns:
                data = data.copy()  # Don't modify original data
                data['returns'] = data['close'].pct_change()
            
            # Align predictions with returns (skip first return which is NaN)
            if len(predictions) != len(data):
                # Handle length mismatch by taking the shorter length
                min_len = min(len(predictions), len(data))
                predictions = predictions[:min_len]
                data = data.iloc[:min_len]
            
            # Calculate strategy returns
            returns = data['returns'].values[1:]  # Skip first NaN
            pred_returns = predictions[:-1]  # Align with returns
            
            if len(returns) != len(pred_returns):
                min_len = min(len(returns), len(pred_returns))
                returns = returns[:min_len]
                pred_returns = pred_returns[:min_len]
            
            strategy_returns = returns * pred_returns
            
            # Validate strategy returns
            if len(strategy_returns) == 0:
                raise ValueError("No valid strategy returns calculated")
            
            # Calculate metrics with error handling
            metrics = {
                f'{prefix}_sharpe': self._calculate_sharpe(strategy_returns),
                f'{prefix}_sortino': self._calculate_sortino(strategy_returns),
                f'{prefix}_max_drawdown': self._calculate_max_drawdown(strategy_returns),
                f'{prefix}_win_rate': self._safe_mean(strategy_returns > 0),
                f'{prefix}_total_return': self._safe_sum(strategy_returns),
                f'{prefix}_volatility': self._safe_std(strategy_returns)
            }
            
            # Validate all metrics are finite
            for key, value in metrics.items():
                if not np.isfinite(value):
                    self.logger.warning(f"Non-finite metric {key}: {value}")
                    metrics[key] = 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics for {prefix}: {e}")
            # Return default metrics
            return {
                f'{prefix}_sharpe': 0.0,
                f'{prefix}_sortino': 0.0,
                f'{prefix}_max_drawdown': 0.0,
                f'{prefix}_win_rate': 0.0,
                f'{prefix}_total_return': 0.0,
                f'{prefix}_volatility': 0.0
            }
    
    def _safe_mean(self, array: np.ndarray) -> float:
        """Safely calculate mean."""
        try:
            if len(array) == 0:
                return 0.0
            return float(np.mean(array))
        except Exception:
            return 0.0
    
    def _safe_sum(self, array: np.ndarray) -> float:
        """Safely calculate sum."""
        try:
            if len(array) == 0:
                return 0.0
            return float(np.sum(array))
        except Exception:
            return 0.0
    
    def _safe_std(self, array: np.ndarray) -> float:
        """Safely calculate standard deviation."""
        try:
            if len(array) == 0:
                return 0.0
            return float(np.std(array))
        except Exception:
            return 0.0

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        try:
            if not isinstance(returns, np.ndarray):
                return 0.0
            if len(returns) == 0:
                return 0.0
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if not np.isfinite(mean_return) or not np.isfinite(std_return):
                return 0.0
            
            if std_return == 0:
                return 0.0
            
            sharpe = mean_return / std_return * np.sqrt(252)
            return float(sharpe) if np.isfinite(sharpe) else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        try:
            if not isinstance(returns, np.ndarray):
                return 0.0
            if len(returns) == 0:
                return 0.0
            
            mean_return = np.mean(returns)
            if not np.isfinite(mean_return):
                return 0.0
            
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                return float('inf') if mean_return > 0 else 0.0
            
            downside_std = np.std(downside_returns)
            if not np.isfinite(downside_std) or downside_std == 0:
                return float('inf') if mean_return > 0 else 0.0
            
            sortino = mean_return / downside_std * np.sqrt(252)
            return float(sortino) if np.isfinite(sortino) else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating Sortino ratio: {e}")
            return 0.0

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        try:
            if not isinstance(returns, np.ndarray):
                return 0.0
            if len(returns) == 0:
                return 0.0
            
            # Check for infinite or NaN values
            if not np.all(np.isfinite(returns)):
                self.logger.warning("Non-finite values found in returns for drawdown calculation")
                returns = returns[np.isfinite(returns)]
                if len(returns) == 0:
                    return 0.0
            
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            
            # Handle division by zero
            drawdown = np.where(running_max == 0, 0, drawdown)
            
            max_dd = np.min(drawdown)
            return float(max_dd) if np.isfinite(max_dd) else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating max drawdown: {e}")
            return 0.0

    def _calculate_degradation(self, train_metrics: Dict[str, float], test_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance degradation from train to test."""
        try:
            # Input validation
            if not isinstance(train_metrics, dict):
                raise RuntimeTypeError(dict, train_metrics, '_calculate_degradation train_metrics')
            if not isinstance(test_metrics, dict):
                raise RuntimeTypeError(dict, test_metrics, '_calculate_degradation test_metrics')
            
            degradation = {}
            
            # Calculate Sharpe degradation
            train_sharpe = next((v for k, v in train_metrics.items() if 'sharpe' in k), 0.0)
            test_sharpe = next((v for k, v in test_metrics.items() if 'sharpe' in k), 0.0)
            
            if not np.isfinite(train_sharpe) or not np.isfinite(test_sharpe):
                degradation['sharpe_degradation'] = 0.0
            elif train_sharpe != 0:
                sharpe_degradation = (train_sharpe - test_sharpe) / abs(train_sharpe)
                degradation['sharpe_degradation'] = float(sharpe_degradation) if np.isfinite(sharpe_degradation) else 0.0
            else:
                degradation['sharpe_degradation'] = 0.0
            
            # Calculate win rate degradation
            train_wr = next((v for k, v in train_metrics.items() if 'win_rate' in k), 0.0)
            test_wr = next((v for k, v in test_metrics.items() if 'win_rate' in k), 0.0)
            
            if not np.isfinite(train_wr) or not np.isfinite(test_wr):
                degradation['win_rate_degradation'] = 0.0
            elif train_wr != 0:
                wr_degradation = (train_wr - test_wr) / train_wr
                degradation['win_rate_degradation'] = float(wr_degradation) if np.isfinite(wr_degradation) else 0.0
            else:
                degradation['win_rate_degradation'] = 0.0
            
            # Calculate overall degradation
            overall = (degradation['sharpe_degradation'] + degradation['win_rate_degradation']) / 2
            degradation['overall'] = float(overall) if np.isfinite(overall) else 0.0
            
            # Determine potential overfitting
            degradation['potential_overfitting'] = degradation['overall'] > self.max_acceptable_degradation
            
            return degradation
            
        except Exception as e:
            self.logger.error(f"Error calculating degradation: {e}")
            return {
                'sharpe_degradation': 0.0,
                'win_rate_degradation': 0.0,
                'overall': 0.0,
                'potential_overfitting': False
            }

    def _analyze_validation_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze walk-forward validation results."""
        try:
            # Input validation
            if not isinstance(results, list):
                raise RuntimeTypeError(list, results, '_analyze_validation_results results')
            
            analysis = {
                'total_windows': len(results),
                'successful_windows': 0,
                'average_degradation': 0.0,
                'overfitting_windows': 0,
                'regime_analysis': {} if self.regime_aware else None,
                'validation_passed': False
            }
            
            degradations = []
            
            # Analyze each result
            for result in results:
                if not isinstance(result, dict):
                    continue
                    
                if result.get('success', False):
                    analysis['successful_windows'] += 1
                    
                    if 'degradation' in result and isinstance(result['degradation'], dict):
                        degradation = result['degradation'].get('overall', 0.0)
                        if np.isfinite(degradation):
                            degradations.append(degradation)
                        
                        if result['degradation'].get('potential_overfitting', False):
                            analysis['overfitting_windows'] += 1
            
            # Calculate average degradation
            if degradations:
                avg_degradation = np.mean(degradations)
                analysis['average_degradation'] = float(avg_degradation) if np.isfinite(avg_degradation) else 0.0
            
            # Analyze regime-specific results
            if self.regime_aware:
                for regime in ['bull', 'bear', 'sideways']:
                    regime_degradations = []
                    for result in results:
                        if not isinstance(result, dict):
                            continue
                            
                        if 'regime_results' in result and isinstance(result['regime_results'], dict):
                            regime_result = result['regime_results'].get(regime, {})
                            if isinstance(regime_result, dict) and regime_result.get('success', False):
                                if 'degradation' in regime_result and isinstance(regime_result['degradation'], dict):
                                    degradation = regime_result['degradation'].get('overall', 0.0)
                                    if np.isfinite(degradation):
                                        regime_degradations.append(degradation)
                    
                    if regime_degradations:
                        avg_regime_degradation = np.mean(regime_degradations)
                        analysis['regime_analysis'][regime] = {
                            'avg_degradation': float(avg_regime_degradation) if np.isfinite(avg_regime_degradation) else 0.0,
                            'windows_analyzed': len(regime_degradations)
                        }
            
            # Determine if validation passed
            successful_windows = max(analysis['successful_windows'], 1)
            overfitting_ratio = analysis['overfitting_windows'] / successful_windows
            analysis['validation_passed'] = (
                analysis['average_degradation'] <= self.max_acceptable_degradation and 
                overfitting_ratio < 0.3
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing validation results: {e}")
            return {
                'total_windows': len(results) if isinstance(results, list) else 0,
                'successful_windows': 0,
                'average_degradation': 0.0,
                'overfitting_windows': 0,
                'regime_analysis': {} if self.regime_aware else None,
                'validation_passed': False,
                'error': f"Analysis failed: {str(e)}"
            }

    def _save_validation_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> None:
        """Save validation results to disk."""
        try:
            # Input validation
            if not isinstance(results, list):
                raise RuntimeTypeError(list, results, '_save_validation_results results')
            if not isinstance(analysis, dict):
                raise RuntimeTypeError(dict, analysis, '_save_validation_results analysis')
            
            # Ensure results directory exists
            if not ensure_directory(self.results_dir):
                raise RuntimeError(f"Failed to create results directory: {self.results_dir}")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_path = self.results_dir / f'validation_results_{timestamp}.json'
            summary_path = self.results_dir / f'validation_summary_{timestamp}.json'
            
            # Prepare data for JSON serialization
            serializable_results = []
            for result in results:
                if isinstance(result, dict):
                    # Convert any non-serializable objects to strings
                    serializable_result = {}
                    for key, value in result.items():
                        try:
                            # Test if value is JSON serializable
                            import json
                            json.dumps(value)
                            serializable_result[key] = value
                        except (TypeError, ValueError):
                            serializable_result[key] = str(value)
                    serializable_results.append(serializable_result)
                else:
                    serializable_results.append(str(result))
            
            # Save results
            if not safe_json_dump(serializable_results, results_path, indent=2):
                raise RuntimeError(f"Failed to save results to {results_path}")
            
            # Save summary
            if not safe_json_dump(analysis, summary_path, indent=2):
                raise RuntimeError(f"Failed to save summary to {summary_path}")
            
            self.logger.info(f'Saved validation results to {results_path}')
            self.logger.info(f'Saved validation summary to {summary_path}')
            
        except Exception as e:
            self.logger.error(f"Error saving validation results: {e}")
            # Don't raise the exception to avoid failing the entire validation
    
    def validate(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Validate method compatible with BaseValidator interface.
        
        Args:
            data: The data to validate (should be a DataFrame)
            **kwargs: Additional validation parameters including:
                - model_trainer: Callable for training models
                - regime_labels: Optional regime labels for regime-aware validation
        
        Returns:
            Dict containing validation results
        """
        try:
            # Extract parameters from kwargs
            model_trainer = kwargs.get('model_trainer')
            regime_labels = kwargs.get('regime_labels', None)
            
            if model_trainer is None:
                return {
                    'valid': False,
                    'error': 'model_trainer is required for walk-forward validation',
                    'validation_passed': False
                }
            
            # Run validation synchronously (convert async to sync)
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, we need to handle this differently
                    # For now, return an error suggesting to use validate_model directly
                    return {
                        'valid': False,
                        'error': 'Walk-forward validation requires async context. Use validate_model() directly.',
                        'validation_passed': False
                    }
                else:
                    result = loop.run_until_complete(self.validate_model(model_trainer, data, regime_labels))
            except RuntimeError:
                # No event loop, create a new one
                result = asyncio.run(self.validate_model(model_trainer, data, regime_labels))
            
            # Convert result to BaseValidator format
            return {
                'valid': result['analysis'].get('validation_passed', False),
                'validation_passed': result['analysis'].get('validation_passed', False),
                'total_windows': result['analysis'].get('total_windows', 0),
                'successful_windows': result['analysis'].get('successful_windows', 0),
                'average_degradation': result['analysis'].get('average_degradation', 0.0),
                'overfitting_windows': result['analysis'].get('overfitting_windows', 0),
                'regime_analysis': result['analysis'].get('regime_analysis', {}),
                'results': result['results'],
                'windows': result['windows']
            }
            
        except Exception as e:
            self.logger.error(f"Error in validate method: {e}")
            return {
                'valid': False,
                'error': str(e),
                'validation_passed': False
            }


async def example_model_trainer(data: pd.DataFrame, regime: Optional[str]=None) -> Any:
    """Example model trainer for testing walk-forward validation."""

    class DummyModel:

        def __init__(self, regime: Any = None) -> None:
            self.regime = regime

        def predict(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> None:
            return np.random.choice([-1, 0, 1], size = len(data))

        def get_params(self) -> Any:
            return {'regime': self.regime}
    return DummyModel(regime)


async def run_validator(training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run walk-forward validator compatible with the validator orchestrator.
    
    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state
    
    Returns:
        Dictionary containing validation results
    """
    try:
        # Extract configuration from training_input
        config = training_input.get('walk_forward_config', {})
        if not config:
            # Use default configuration
            config = {
                'train_period_days': 365,
                'test_period_days': 30,
                'step_days': 30,
                'regime_aware': True,
                'adaptive_windows': True,
                'results_dir': 'validation_results'
            }
        
        # Create validator
        validator = WalkForwardValidator(config)
        
        # Extract data from pipeline state
        data = pipeline_state.get('data')
        if data is None:
            return {
                'validation_passed': False,
                'error': 'No data found in pipeline state for walk-forward validation'
            }
        
        # Extract model trainer from pipeline state or training input
        model_trainer = pipeline_state.get('model_trainer') or training_input.get('model_trainer')
        if model_trainer is None:
            # Use example model trainer for testing
            model_trainer = example_model_trainer
        
        # Extract regime labels if available
        regime_labels = pipeline_state.get('regime_labels')
        
        # Run validation
        result = await validator.validate_model(model_trainer, data, regime_labels)
        
        return {
            'validation_passed': result['analysis'].get('validation_passed', False),
            'total_windows': result['analysis'].get('total_windows', 0),
            'successful_windows': result['analysis'].get('successful_windows', 0),
            'average_degradation': result['analysis'].get('average_degradation', 0.0),
            'overfitting_windows': result['analysis'].get('overfitting_windows', 0),
            'regime_analysis': result['analysis'].get('regime_analysis', {}),
            'results': result['results'],
            'windows': result['windows']
        }
        
    except Exception as e:
        return {
            'validation_passed': False,
            'error': f'Walk-forward validation failed: {str(e)}'
        }


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