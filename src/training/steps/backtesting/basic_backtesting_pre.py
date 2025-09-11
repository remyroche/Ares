"""
Basic Backtesting Pre-Optimization Step

This module provides baseline backtesting functionality before parameter optimization,
establishing performance benchmarks and identifying areas for improvement.

Key Features:
- Baseline performance measurement
- Pre-optimization strategy validation
- Performance benchmark establishment
- Data quality assessment
- Risk metrics calculation
- Comprehensive logging and reporting
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


class BaselineStrategyType(Enum):
    """Types of baseline strategies for pre-optimization backtesting."""
    BUY_AND_HOLD = "buy_and_hold"
    RANDOM_WALK = "random_walk"
    SIMPLE_MA = "simple_moving_average"
    RSI_STRATEGY = "rsi_strategy"
    BOLLINGER_BANDS = "bollinger_bands"
    CUSTOM = "custom"


@dataclass
class BasicBacktestingPreConfig:
    """Configuration for basic backtesting pre-optimization."""
    # Basic configuration
    symbol: str
    exchange: str
    timeframe: str
    data_dir: str
    
    # Backtesting parameters
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_position_size: float = 0.1
    
    # Baseline strategy configuration
    baseline_strategies: List[BaselineStrategyType] = field(default_factory=lambda: [
        BaselineStrategyType.BUY_AND_HOLD,
        BaselineStrategyType.SIMPLE_MA,
        BaselineStrategyType.RSI_STRATEGY
    ])
    
    # Technical indicator parameters
    ma_period: int = 20
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    bb_period: int = 20
    bb_std_dev: float = 2.0
    
    # Risk management
    max_drawdown_threshold: float = 0.2
    stop_loss_threshold: float = 0.05
    take_profit_threshold: float = 0.1
    
    # Performance settings
    enable_detailed_logging: bool = True
    enable_performance_monitoring: bool = True
    enable_memory_optimization: bool = True
    
    # Output settings
    save_detailed_results: bool = True
    generate_plots: bool = True
    output_format: str = "parquet"


@dataclass
class BasicBacktestingPreResults:
    """Results from basic backtesting pre-optimization."""
    # Basic info
    symbol: str
    exchange: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    total_duration: float
    
    # Baseline strategy results
    baseline_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Performance benchmarks
    performance_benchmarks: Dict[str, float] = field(default_factory=dict)
    
    # Risk metrics
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Data quality assessment
    data_quality_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Optimization recommendations
    optimization_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Detailed data
    equity_curves: Dict[str, pd.DataFrame] = field(default_factory=dict)
    trade_logs: Dict[str, pd.DataFrame] = field(default_factory=dict)
    
    # Metadata
    config: BasicBacktestingPreConfig = field(default_factory=BasicBacktestingPreConfig)
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    system_metrics: Dict[str, Any] = field(default_factory=dict)


class BasicBacktestingPreStep:
    """Basic backtesting pre-optimization step."""
    
    def __init__(self, config: BasicBacktestingPreConfig):
        """Initialize the basic backtesting pre-optimization step."""
        self.config = config
        self.logger = logger.getChild('BasicBacktestingPreStep')
        
        # Initialize utilities
        self.parquet_utils = get_parquet_utils()
        self.financial_logger = EnhancedFinancialMetricsLogger()
        self.performance_monitor = PerformanceMonitor()
        self.system_monitor = SystemMonitor()
        
        # Initialize data directory
        self.data_dir = Path(config.data_dir)
        ensure_directory(self.data_dir)
        
        self.logger.info(f"🚀 BasicBacktestingPreStep initialized for {config.symbol}")
        self.logger.info(f"📊 Baseline strategies: {[s.value for s in config.baseline_strategies]}")
        self.logger.info(f"💰 Initial capital: ${config.initial_capital:,.2f}")
        self.logger.info(f"📁 Data directory: {config.data_dir}")
    
    @traced(span_name='basic_backtesting_pre')
    @log_execution_time
    @monitor_step_execution
    async def execute(
        self, 
        data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> BasicBacktestingPreResults:
        """Execute basic backtesting pre-optimization."""
        
        self.logger.info("🚀 Starting basic backtesting pre-optimization...")
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
            
            # Assess data quality
            data_quality_metrics = await self._assess_data_quality(data)
            
            # Execute baseline strategies
            baseline_results = await self._execute_baseline_strategies(data)
            
            # Calculate performance benchmarks
            performance_benchmarks = self._calculate_performance_benchmarks(baseline_results)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(baseline_results)
            
            # Generate optimization recommendations
            optimization_recommendations = self._generate_optimization_recommendations(
                baseline_results, performance_benchmarks, risk_metrics
            )
            
            # Create results
            results = BasicBacktestingPreResults(
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                timeframe=self.config.timeframe,
                start_time=datetime.now(),
                end_time=datetime.now(),
                total_duration=time.time() - start_time,
                baseline_results=baseline_results,
                performance_benchmarks=performance_benchmarks,
                risk_metrics=risk_metrics,
                data_quality_metrics=data_quality_metrics,
                optimization_recommendations=optimization_recommendations,
                config=self.config,
                execution_time=time.time() - start_time,
                memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
                system_metrics=self._get_system_metrics()
            )
            
            # Save results
            if self.config.save_detailed_results:
                await self._save_results(results)
            
            self.logger.info("✅ Basic backtesting pre-optimization completed successfully")
            self.logger.info(f"⏱️ Execution time: {results.execution_time:.2f}s")
            self.logger.info(f"📊 Baseline strategies tested: {len(baseline_results)}")
            self.logger.info(f"💡 Optimization recommendations: {len(optimization_recommendations)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in basic backtesting pre-optimization: {e}")
            self.logger.exception("Full traceback:")
            raise
        finally:
            # Stop performance monitoring
            if self.config.enable_performance_monitoring:
                self.performance_monitor.stop_monitoring()
    
    async def _load_data(self) -> pd.DataFrame:
        """Load market data for backtesting."""
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
        if len(data) < 100:
            raise ValidationError(f"Insufficient data points: {len(data)} < 100")
        
        # Check for missing values
        missing_values = data[required_columns].isnull().sum().sum()
        if missing_values > 0:
            self.logger.warning(f"⚠️ Found {missing_values} missing values")
        
        self.logger.info("✅ Data validation completed successfully")
    
    async def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality metrics."""
        self.logger.info("🔍 Assessing data quality...")
        
        quality_metrics = {
            'total_records': len(data),
            'date_range': {
                'start': data.index[0].isoformat() if not data.empty else None,
                'end': data.index[-1].isoformat() if not data.empty else None,
                'duration_days': (data.index[-1] - data.index[0]).days if len(data) > 1 else 0
            },
            'missing_values': data.isnull().sum().to_dict(),
            'duplicate_records': data.duplicated().sum(),
            'data_types': data.dtypes.to_dict(),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'price_statistics': {},
            'volume_statistics': {}
        }
        
        # Calculate price statistics
        if 'close' in data.columns:
            close_prices = data['close'].dropna()
            quality_metrics['price_statistics'] = {
                'mean': float(close_prices.mean()),
                'std': float(close_prices.std()),
                'min': float(close_prices.min()),
                'max': float(close_prices.max()),
                'median': float(close_prices.median()),
                'skewness': float(close_prices.skew()),
                'kurtosis': float(close_prices.kurtosis())
            }
        
        # Calculate volume statistics
        if 'volume' in data.columns:
            volume = data['volume'].dropna()
            quality_metrics['volume_statistics'] = {
                'mean': float(volume.mean()),
                'std': float(volume.std()),
                'min': float(volume.min()),
                'max': float(volume.max()),
                'median': float(volume.median())
            }
        
        self.logger.info("✅ Data quality assessment completed")
        return quality_metrics
    
    async def _execute_baseline_strategies(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Execute baseline strategies for comparison."""
        self.logger.info("🎯 Executing baseline strategies...")
        
        results = {}
        
        for strategy_type in self.config.baseline_strategies:
            self.logger.info(f"🔄 Executing {strategy_type.value} strategy...")
            
            try:
                if strategy_type == BaselineStrategyType.BUY_AND_HOLD:
                    strategy_results = await self._execute_buy_and_hold(data)
                elif strategy_type == BaselineStrategyType.SIMPLE_MA:
                    strategy_results = await self._execute_simple_ma(data)
                elif strategy_type == BaselineStrategyType.RSI_STRATEGY:
                    strategy_results = await self._execute_rsi_strategy(data)
                elif strategy_type == BaselineStrategyType.BOLLINGER_BANDS:
                    strategy_results = await self._execute_bollinger_bands(data)
                else:
                    self.logger.warning(f"⚠️ Strategy {strategy_type.value} not implemented")
                    continue
                
                results[strategy_type.value] = strategy_results
                self.logger.info(f"✅ {strategy_type.value} strategy completed")
                
            except Exception as e:
                self.logger.error(f"❌ Error executing {strategy_type.value} strategy: {e}")
                continue
        
        self.logger.info(f"✅ Baseline strategies execution completed: {len(results)} strategies")
        return results
    
    async def _execute_buy_and_hold(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute buy and hold strategy."""
        initial_price = data['close'].iloc[0]
        final_price = data['close'].iloc[-1]
        total_return = (final_price - initial_price) / initial_price
        
        # Calculate equity curve
        equity_curve = pd.DataFrame({
            'timestamp': data.index,
            'price': data['close'],
            'equity': self.config.initial_capital * (1 + total_return),
            'return': data['close'].pct_change().fillna(0)
        })
        
        # Calculate basic metrics
        returns = data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (returns.mean() * 252) / volatility if volatility > 0 else 0
        
        return {
            'strategy_type': 'buy_and_hold',
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 / len(data)) - 1,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self._calculate_max_drawdown(equity_curve['equity']),
            'total_trades': 1,
            'win_rate': 1.0 if total_return > 0 else 0.0,
            'equity_curve': equity_curve,
            'trade_log': pd.DataFrame({
                'timestamp': [data.index[0], data.index[-1]],
                'action': ['buy', 'sell'],
                'price': [initial_price, final_price],
                'shares': [self.config.initial_capital / initial_price, 0],
                'value': [self.config.initial_capital, self.config.initial_capital * (1 + total_return)]
            })
        }
    
    async def _execute_simple_ma(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute simple moving average strategy."""
        # Calculate moving average
        data_copy = data.copy()
        data_copy['ma'] = data_copy['close'].rolling(window=self.config.ma_period).mean()
        
        # Generate signals
        data_copy['signal'] = 0
        data_copy.loc[data_copy['close'] > data_copy['ma'], 'signal'] = 1
        data_copy.loc[data_copy['close'] < data_copy['ma'], 'signal'] = -1
        
        # Calculate position changes
        data_copy['position'] = data_copy['signal'].diff()
        
        # Execute trades
        portfolio = {
            'cash': self.config.initial_capital,
            'shares': 0.0,
            'equity': self.config.initial_capital,
            'trades': []
        }
        
        for i, (timestamp, row) in enumerate(data_copy.iterrows()):
            if row['position'] != 0:  # Position change
                if row['position'] > 0:  # Buy
                    shares_to_buy = (portfolio['cash'] * self.config.max_position_size) / row['close']
                    cost = shares_to_buy * row['close'] * (1 + self.config.commission_rate)
                    if cost <= portfolio['cash']:
                        portfolio['cash'] -= cost
                        portfolio['shares'] += shares_to_buy
                        portfolio['trades'].append({
                            'timestamp': timestamp,
                            'action': 'buy',
                            'price': row['close'],
                            'shares': shares_to_buy,
                            'cost': cost
                        })
                elif row['position'] < 0 and portfolio['shares'] > 0:  # Sell
                    proceeds = portfolio['shares'] * row['close'] * (1 - self.config.commission_rate)
                    portfolio['cash'] += proceeds
                    portfolio['trades'].append({
                        'timestamp': timestamp,
                        'action': 'sell',
                        'price': row['close'],
                        'shares': portfolio['shares'],
                        'proceeds': proceeds
                    })
                    portfolio['shares'] = 0.0
            
            # Update equity
            portfolio['equity'] = portfolio['cash'] + (portfolio['shares'] * row['close'])
        
        # Calculate final metrics
        final_equity = portfolio['equity']
        total_return = (final_equity - self.config.initial_capital) / self.config.initial_capital
        
        # Create equity curve
        equity_curve = pd.DataFrame({
            'timestamp': data_copy.index,
            'price': data_copy['close'],
            'ma': data_copy['ma'],
            'signal': data_copy['signal'],
            'equity': [portfolio['equity']] * len(data_copy)  # Simplified
        })
        
        # Create trade log
        trade_log = pd.DataFrame(portfolio['trades']) if portfolio['trades'] else pd.DataFrame()
        
        return {
            'strategy_type': 'simple_ma',
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 / len(data)) - 1,
            'volatility': data_copy['close'].pct_change().std() * np.sqrt(252),
            'sharpe_ratio': 0.0,  # Would need to calculate properly
            'max_drawdown': 0.0,  # Would need to calculate properly
            'total_trades': len(portfolio['trades']),
            'win_rate': 0.0,  # Would need to calculate properly
            'equity_curve': equity_curve,
            'trade_log': trade_log
        }
    
    async def _execute_rsi_strategy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute RSI strategy."""
        # Calculate RSI
        data_copy = data.copy()
        delta = data_copy['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_period).mean()
        rs = gain / loss
        data_copy['rsi'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        data_copy['signal'] = 0
        data_copy.loc[data_copy['rsi'] < self.config.rsi_oversold, 'signal'] = 1  # Buy
        data_copy.loc[data_copy['rsi'] > self.config.rsi_overbought, 'signal'] = -1  # Sell
        
        # Calculate position changes
        data_copy['position'] = data_copy['signal'].diff()
        
        # Execute trades (similar to MA strategy)
        portfolio = {
            'cash': self.config.initial_capital,
            'shares': 0.0,
            'equity': self.config.initial_capital,
            'trades': []
        }
        
        for i, (timestamp, row) in enumerate(data_copy.iterrows()):
            if row['position'] != 0:  # Position change
                if row['position'] > 0:  # Buy
                    shares_to_buy = (portfolio['cash'] * self.config.max_position_size) / row['close']
                    cost = shares_to_buy * row['close'] * (1 + self.config.commission_rate)
                    if cost <= portfolio['cash']:
                        portfolio['cash'] -= cost
                        portfolio['shares'] += shares_to_buy
                        portfolio['trades'].append({
                            'timestamp': timestamp,
                            'action': 'buy',
                            'price': row['close'],
                            'shares': shares_to_buy,
                            'cost': cost
                        })
                elif row['position'] < 0 and portfolio['shares'] > 0:  # Sell
                    proceeds = portfolio['shares'] * row['close'] * (1 - self.config.commission_rate)
                    portfolio['cash'] += proceeds
                    portfolio['trades'].append({
                        'timestamp': timestamp,
                        'action': 'sell',
                        'price': row['close'],
                        'shares': portfolio['shares'],
                        'proceeds': proceeds
                    })
                    portfolio['shares'] = 0.0
            
            # Update equity
            portfolio['equity'] = portfolio['cash'] + (portfolio['shares'] * row['close'])
        
        # Calculate final metrics
        final_equity = portfolio['equity']
        total_return = (final_equity - self.config.initial_capital) / self.config.initial_capital
        
        # Create equity curve
        equity_curve = pd.DataFrame({
            'timestamp': data_copy.index,
            'price': data_copy['close'],
            'rsi': data_copy['rsi'],
            'signal': data_copy['signal'],
            'equity': [portfolio['equity']] * len(data_copy)  # Simplified
        })
        
        # Create trade log
        trade_log = pd.DataFrame(portfolio['trades']) if portfolio['trades'] else pd.DataFrame()
        
        return {
            'strategy_type': 'rsi_strategy',
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 / len(data)) - 1,
            'volatility': data_copy['close'].pct_change().std() * np.sqrt(252),
            'sharpe_ratio': 0.0,  # Would need to calculate properly
            'max_drawdown': 0.0,  # Would need to calculate properly
            'total_trades': len(portfolio['trades']),
            'win_rate': 0.0,  # Would need to calculate properly
            'equity_curve': equity_curve,
            'trade_log': trade_log
        }
    
    async def _execute_bollinger_bands(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute Bollinger Bands strategy."""
        # Calculate Bollinger Bands
        data_copy = data.copy()
        data_copy['ma'] = data_copy['close'].rolling(window=self.config.bb_period).mean()
        data_copy['std'] = data_copy['close'].rolling(window=self.config.bb_period).std()
        data_copy['upper_band'] = data_copy['ma'] + (data_copy['std'] * self.config.bb_std_dev)
        data_copy['lower_band'] = data_copy['ma'] - (data_copy['std'] * self.config.bb_std_dev)
        
        # Generate signals
        data_copy['signal'] = 0
        data_copy.loc[data_copy['close'] < data_copy['lower_band'], 'signal'] = 1  # Buy
        data_copy.loc[data_copy['close'] > data_copy['upper_band'], 'signal'] = -1  # Sell
        
        # Calculate position changes
        data_copy['position'] = data_copy['signal'].diff()
        
        # Execute trades (similar to other strategies)
        portfolio = {
            'cash': self.config.initial_capital,
            'shares': 0.0,
            'equity': self.config.initial_capital,
            'trades': []
        }
        
        for i, (timestamp, row) in enumerate(data_copy.iterrows()):
            if row['position'] != 0:  # Position change
                if row['position'] > 0:  # Buy
                    shares_to_buy = (portfolio['cash'] * self.config.max_position_size) / row['close']
                    cost = shares_to_buy * row['close'] * (1 + self.config.commission_rate)
                    if cost <= portfolio['cash']:
                        portfolio['cash'] -= cost
                        portfolio['shares'] += shares_to_buy
                        portfolio['trades'].append({
                            'timestamp': timestamp,
                            'action': 'buy',
                            'price': row['close'],
                            'shares': shares_to_buy,
                            'cost': cost
                        })
                elif row['position'] < 0 and portfolio['shares'] > 0:  # Sell
                    proceeds = portfolio['shares'] * row['close'] * (1 - self.config.commission_rate)
                    portfolio['cash'] += proceeds
                    portfolio['trades'].append({
                        'timestamp': timestamp,
                        'action': 'sell',
                        'price': row['close'],
                        'shares': portfolio['shares'],
                        'proceeds': proceeds
                    })
                    portfolio['shares'] = 0.0
            
            # Update equity
            portfolio['equity'] = portfolio['cash'] + (portfolio['shares'] * row['close'])
        
        # Calculate final metrics
        final_equity = portfolio['equity']
        total_return = (final_equity - self.config.initial_capital) / self.config.initial_capital
        
        # Create equity curve
        equity_curve = pd.DataFrame({
            'timestamp': data_copy.index,
            'price': data_copy['close'],
            'ma': data_copy['ma'],
            'upper_band': data_copy['upper_band'],
            'lower_band': data_copy['lower_band'],
            'signal': data_copy['signal'],
            'equity': [portfolio['equity']] * len(data_copy)  # Simplified
        })
        
        # Create trade log
        trade_log = pd.DataFrame(portfolio['trades']) if portfolio['trades'] else pd.DataFrame()
        
        return {
            'strategy_type': 'bollinger_bands',
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 / len(data)) - 1,
            'volatility': data_copy['close'].pct_change().std() * np.sqrt(252),
            'sharpe_ratio': 0.0,  # Would need to calculate properly
            'max_drawdown': 0.0,  # Would need to calculate properly
            'total_trades': len(portfolio['trades']),
            'win_rate': 0.0,  # Would need to calculate properly
            'equity_curve': equity_curve,
            'trade_log': trade_log
        }
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(equity_series) == 0:
            return 0.0
        
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        return float(drawdown.min())
    
    def _calculate_performance_benchmarks(self, baseline_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance benchmarks from baseline results."""
        self.logger.info("📊 Calculating performance benchmarks...")
        
        benchmarks = {}
        
        if not baseline_results:
            return benchmarks
        
        # Extract returns
        returns = [result['total_return'] for result in baseline_results.values()]
        sharpe_ratios = [result['sharpe_ratio'] for result in baseline_results.values()]
        volatilities = [result['volatility'] for result in baseline_results.values()]
        max_drawdowns = [result['max_drawdown'] for result in baseline_results.values()]
        
        # Calculate benchmarks
        benchmarks['best_return'] = max(returns) if returns else 0.0
        benchmarks['worst_return'] = min(returns) if returns else 0.0
        benchmarks['average_return'] = np.mean(returns) if returns else 0.0
        benchmarks['median_return'] = np.median(returns) if returns else 0.0
        benchmarks['return_std'] = np.std(returns) if len(returns) > 1 else 0.0
        
        benchmarks['best_sharpe'] = max(sharpe_ratios) if sharpe_ratios else 0.0
        benchmarks['worst_sharpe'] = min(sharpe_ratios) if sharpe_ratios else 0.0
        benchmarks['average_sharpe'] = np.mean(sharpe_ratios) if sharpe_ratios else 0.0
        
        benchmarks['best_volatility'] = min(volatilities) if volatilities else 0.0
        benchmarks['worst_volatility'] = max(volatilities) if volatilities else 0.0
        benchmarks['average_volatility'] = np.mean(volatilities) if volatilities else 0.0
        
        benchmarks['best_drawdown'] = min(max_drawdowns) if max_drawdowns else 0.0
        benchmarks['worst_drawdown'] = max(max_drawdowns) if max_drawdowns else 0.0
        benchmarks['average_drawdown'] = np.mean(max_drawdowns) if max_drawdowns else 0.0
        
        self.logger.info("✅ Performance benchmarks calculated")
        return benchmarks
    
    def _calculate_risk_metrics(self, baseline_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate risk metrics from baseline results."""
        self.logger.info("⚠️ Calculating risk metrics...")
        
        risk_metrics = {}
        
        if not baseline_results:
            return risk_metrics
        
        # Calculate portfolio-level risk metrics
        all_returns = []
        all_volatilities = []
        all_drawdowns = []
        
        for result in baseline_results.values():
            if 'equity_curve' in result and not result['equity_curve'].empty:
                equity_curve = result['equity_curve']
                if 'return' in equity_curve.columns:
                    returns = equity_curve['return'].dropna()
                    all_returns.extend(returns.tolist())
                    all_volatilities.append(returns.std() * np.sqrt(252))
                
                if 'equity' in equity_curve.columns:
                    max_dd = self._calculate_max_drawdown(equity_curve['equity'])
                    all_drawdowns.append(max_dd)
        
        # Calculate aggregate risk metrics
        if all_returns:
            risk_metrics['portfolio_volatility'] = np.std(all_returns) * np.sqrt(252)
            risk_metrics['portfolio_var_95'] = np.percentile(all_returns, 5)
            risk_metrics['portfolio_var_99'] = np.percentile(all_returns, 1)
            risk_metrics['portfolio_cvar_95'] = np.mean([r for r in all_returns if r <= risk_metrics['portfolio_var_95']])
            risk_metrics['portfolio_cvar_99'] = np.mean([r for r in all_returns if r <= risk_metrics['portfolio_var_99']])
        
        if all_volatilities:
            risk_metrics['average_volatility'] = np.mean(all_volatilities)
            risk_metrics['max_volatility'] = max(all_volatilities)
            risk_metrics['min_volatility'] = min(all_volatilities)
        
        if all_drawdowns:
            risk_metrics['average_drawdown'] = np.mean(all_drawdowns)
            risk_metrics['max_drawdown'] = min(all_drawdowns)  # Most negative
            risk_metrics['min_drawdown'] = max(all_drawdowns)  # Least negative
        
        self.logger.info("✅ Risk metrics calculated")
        return risk_metrics
    
    def _generate_optimization_recommendations(
        self, 
        baseline_results: Dict[str, Dict[str, Any]], 
        performance_benchmarks: Dict[str, float],
        risk_metrics: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on baseline results."""
        self.logger.info("💡 Generating optimization recommendations...")
        
        recommendations = []
        
        # Analyze performance gaps
        if performance_benchmarks:
            best_return = performance_benchmarks.get('best_return', 0)
            average_return = performance_benchmarks.get('average_return', 0)
            
            if best_return > average_return * 1.2:
                recommendations.append({
                    'category': 'PERFORMANCE',
                    'priority': 'HIGH',
                    'title': 'Significant Performance Variation',
                    'description': f'Best strategy ({best_return:.2%}) significantly outperforms average ({average_return:.2%})',
                    'action': 'Investigate factors driving performance differences and optimize underperforming strategies',
                    'impact': 'HIGH'
                })
        
        # Analyze risk metrics
        if risk_metrics:
            max_drawdown = risk_metrics.get('max_drawdown', 0)
            if abs(max_drawdown) > self.config.max_drawdown_threshold:
                recommendations.append({
                    'category': 'RISK_MANAGEMENT',
                    'priority': 'HIGH',
                    'title': 'High Maximum Drawdown',
                    'description': f'Maximum drawdown ({max_drawdown:.2%}) exceeds threshold ({self.config.max_drawdown_threshold:.2%})',
                    'action': 'Implement better risk management and position sizing',
                    'impact': 'HIGH'
                })
            
            portfolio_volatility = risk_metrics.get('portfolio_volatility', 0)
            if portfolio_volatility > 0.3:  # 30% annual volatility
                recommendations.append({
                    'category': 'RISK_MANAGEMENT',
                    'priority': 'MEDIUM',
                    'title': 'High Portfolio Volatility',
                    'description': f'Portfolio volatility ({portfolio_volatility:.2%}) is high',
                    'action': 'Consider diversification or volatility reduction strategies',
                    'impact': 'MEDIUM'
                })
        
        # Analyze strategy performance
        if baseline_results:
            strategy_performance = [(name, result['total_return']) for name, result in baseline_results.items()]
            strategy_performance.sort(key=lambda x: x[1], reverse=True)
            
            if len(strategy_performance) > 1:
                best_strategy = strategy_performance[0]
                worst_strategy = strategy_performance[-1]
                
                if best_strategy[1] - worst_strategy[1] > 0.1:  # 10% difference
                    recommendations.append({
                        'category': 'STRATEGY_OPTIMIZATION',
                        'priority': 'MEDIUM',
                        'title': 'Strategy Performance Gap',
                        'description': f'Best strategy ({best_strategy[0]}: {best_strategy[1]:.2%}) significantly outperforms worst ({worst_strategy[0]}: {worst_strategy[1]:.2%})',
                        'action': f'Focus optimization efforts on {best_strategy[0]} strategy and investigate {worst_strategy[0]} weaknesses',
                        'impact': 'MEDIUM'
                    })
        
        # Data quality recommendations
        recommendations.append({
            'category': 'DATA_QUALITY',
            'priority': 'LOW',
            'title': 'Data Quality Monitoring',
            'description': 'Implement continuous data quality monitoring',
            'action': 'Set up automated data quality checks and alerts',
            'impact': 'LOW'
        })
        
        self.logger.info(f"✅ Generated {len(recommendations)} optimization recommendations")
        return recommendations
    
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
    
    async def _save_results(self, results: BasicBacktestingPreResults) -> None:
        """Save results to disk."""
        self.logger.info("💾 Saving results...")
        
        # Create output directory
        output_dir = self.data_dir / "backtesting_results" / "basic_pre"
        ensure_directory(output_dir)
        
        # Save main results
        results_file = output_dir / f"{self.config.symbol}_{self.config.exchange}_basic_backtesting_pre_results.json"
        await safe_json_dump(results_file, results.__dict__, indent=2)
        
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
async def execute_basic_backtesting_pre(
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE", 
    timeframe: str = "1h",
    data_dir: str = "data/training",
    **kwargs
) -> BasicBacktestingPreResults:
    """
    Convenience function to execute basic backtesting pre-optimization.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Data timeframe
        data_dir: Data directory
        **kwargs: Additional configuration parameters
        
    Returns:
        Basic backtesting pre-optimization results
    """
    config = BasicBacktestingPreConfig(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        **kwargs
    )
    
    step = BasicBacktestingPreStep(config)
    return await step.execute()