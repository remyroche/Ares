"""
Backtesting Engine with M1 Hardware Optimizations

This module provides a comprehensive backtesting engine with walk-forward validation,
utilizing M1 GPU, memory, and CPU optimizations for maximum performance.
"""

import asyncio
import logging

# Optional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import gc

# Optional imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# M1 Optimization imports
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager, m1_backtesting_simulate
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer, parallel_backtesting_worker

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
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time, 
    timeout, error_boundary, compose, validate_data_quality, 
    monitor_step_execution, ensure_data_integrity, validate_pipeline_step
)
from src.core.errors import (
    ValidationError, DataIntegrityError, FileOperationError,
    MathValidationError, TimeoutError
)

logger = logging.getLogger(__name__)


class BacktestingMode(Enum):
    """Backtesting execution modes."""
    WALK_FORWARD = "walk_forward"
    FIXED_WINDOW = "fixed_window"
    EXPANDING_WINDOW = "expanding_window"
    ROLLING_WINDOW = "rolling_window"


@dataclass
class BacktestingConfig:
    """Configuration for backtesting engine."""
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
    
    # Walk-forward parameters
    training_window_days: int = 252  # 1 year
    testing_window_days: int = 63    # 3 months
    step_size_days: int = 21         # 1 month
    
    # M1 optimization settings
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    memory_limit_gb: float = 8.0
    max_workers: Optional[int] = None
    
    # Performance settings
    enable_caching: bool = True
    cache_size_mb: int = 100
    enable_profiling: bool = False

    # Risk management
    max_drawdown_threshold: float = 0.2
    stop_loss_threshold: float = 0.05
    take_profit_threshold: float = 0.1

    # Turnover and capacity diagnostics
    capacity_limit: float = 1.0  # Maximum acceptable turnover relative to capital
    market_impact_coefficient: float = 0.0005  # Impact penalty per unit turnover
    turnover_warning_threshold: float = 0.8  # Warn when utilization exceeds this fraction

    # Validation settings
    min_trades_for_validation: int = 10
    confidence_level: float = 0.95

    # Output settings
    save_detailed_results: bool = True
    generate_plots: bool = True
    output_format: str = "parquet"  # parquet, csv, json


@dataclass
class BacktestingResults:
    """Results from backtesting execution."""
    # Basic info
    symbol: str
    exchange: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    total_duration: float
    
    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    
    # Risk metrics
    volatility: float
    var_95: float
    cvar_95: float
    beta: float
    alpha: float

    # Turnover diagnostics
    turnover: float = 0.0
    average_holding_period_days: float = 0.0
    capacity_utilization: float = 0.0
    capacity_limit: float = 0.0
    market_impact_cost: float = 0.0

    # Walk-forward results
    walk_forward_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Detailed data
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    
    # Metadata
    config: BacktestingConfig = field(default_factory=BacktestingConfig)
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    optimization_used: List[str] = field(default_factory=list)


class WalkForwardValidator:
    """Walk-forward validation engine with M1 optimizations."""
    
    def __init__(self, config: BacktestingConfig):
        """Initialize walk-forward validator."""
        self.config = config
        self.logger = logger.getChild('WalkForwardValidator')
        
        # Initialize M1 optimizers
        self.m1_gpu = get_m1_gpu_manager() if config.enable_gpu_acceleration else None
        self.m1_memory = get_m1_memory_optimizer(
            memory_limit_gb=config.memory_limit_gb
        ) if config.enable_memory_optimization else None
        self.m1_cpu = get_m1_cpu_optimizer(
            max_workers=config.max_workers
        ) if config.enable_parallel_processing else None
        
        # Initialize utilities
        self.parquet_utils = get_parquet_utils()
        
        self.logger.info(f"🚀 WalkForwardValidator initialized for {config.symbol}")
        self.logger.info(f"⚡ GPU acceleration: {config.enable_gpu_acceleration}")
        self.logger.info(f"🧠 Memory optimization: {config.enable_memory_optimization}")
        self.logger.info(f"🔄 Parallel processing: {config.enable_parallel_processing}")
    
    @traced(span_name='walk_forward_validation')
    async def validate(
        self, 
        data: pd.DataFrame,
        strategy_func: Callable,
        **strategy_kwargs
    ) -> List[Dict[str, Any]]:
        """Perform walk-forward validation with M1 optimizations."""
        
        self.logger.info("🔄 Starting walk-forward validation...")
        start_time = time.time()
        
        # Prepare data
        data = self._prepare_data(data)
        
        # Generate walk-forward windows
        windows = self._generate_windows(data)
        self.logger.info(f"📊 Generated {len(windows)} walk-forward windows")
        
        # Execute validation
        if self.config.enable_parallel_processing and len(windows) > 4:
            results = await self._parallel_validation(windows, strategy_func, **strategy_kwargs)
        else:
            results = await self._sequential_validation(windows, strategy_func, **strategy_kwargs)
        
        execution_time = time.time() - start_time
        self.logger.info(f"✅ Walk-forward validation completed in {execution_time:.2f}s")
        
        return results
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for backtesting."""
        # Ensure proper datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            else:
                raise ValidationError("Data must have datetime index or timestamp column")
        
        # Sort by time
        data = data.sort_index()
        
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")
        
        # Clean data
        data = data.dropna()
        
        return data
    
    def _generate_windows(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate walk-forward windows."""
        windows = []
        start_date = data.index[0]
        end_date = data.index[-1]
        
        current_start = start_date
        training_days = timedelta(days=self.config.training_window_days)
        testing_days = timedelta(days=self.config.testing_window_days)
        step_days = timedelta(days=self.config.step_size_days)
        
        while current_start + training_days + testing_days <= end_date:
            training_end = current_start + training_days
            testing_start = training_end
            testing_end = testing_start + testing_days
            
            # Get data for this window
            training_data = data.loc[current_start:training_end]
            testing_data = data.loc[testing_start:testing_end]
            
            if len(training_data) > 0 and len(testing_data) > 0:
                windows.append({
                    'window_id': len(windows),
                    'training_start': current_start,
                    'training_end': training_end,
                    'testing_start': testing_start,
                    'testing_end': testing_end,
                    'training_data': training_data,
                    'testing_data': testing_data
                })
            
            current_start += step_days
        
        return windows
    
    async def _parallel_validation(
        self, 
        windows: List[Dict[str, Any]], 
        strategy_func: Callable,
        **strategy_kwargs
    ) -> List[Dict[str, Any]]:
        """Execute validation in parallel using M1 CPU optimizer."""
        self.logger.info(f"⚡ Executing parallel validation with {self.m1_cpu.max_workers} workers")
        
        # Create tasks for parallel execution
        tasks = []
        for window in windows:
            task = self.m1_cpu.submit_task(
                self._validate_window,
                window, strategy_func, **strategy_kwargs
            )
            tasks.append(task)
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Window {i} failed: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _sequential_validation(
        self, 
        windows: List[Dict[str, Any]], 
        strategy_func: Callable,
        **strategy_kwargs
    ) -> List[Dict[str, Any]]:
        """Execute validation sequentially."""
        results = []
        
        for i, window in enumerate(windows):
            self.logger.info(f"🔄 Processing window {i+1}/{len(windows)}")
            
            try:
                result = await self._validate_window(window, strategy_func, **strategy_kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Window {i} failed: {e}")
                continue
        
        return results
    
    async def _validate_window(
        self, 
        window: Dict[str, Any], 
        strategy_func: Callable,
        **strategy_kwargs
    ) -> Dict[str, Any]:
        """Validate a single window."""
        window_id = window['window_id']
        training_data = window['training_data']
        testing_data = window['testing_data']
        
        # Train strategy on training data
        strategy_params = await strategy_func(training_data, mode='train', **strategy_kwargs)
        
        # Test strategy on testing data
        if self.config.enable_gpu_acceleration and self.m1_gpu.should_use_gpu(len(testing_data), "backtesting"):
            # Use GPU acceleration for large datasets
            results = await self._gpu_backtest(testing_data, strategy_params, strategy_func)
        else:
            # Use CPU backtesting
            results = await self._cpu_backtest(testing_data, strategy_params, strategy_func)
        
        return {
            'window_id': window_id,
            'training_period': f"{window['training_start']} to {window['training_end']}",
            'testing_period': f"{window['testing_start']} to {window['testing_end']}",
            'results': results,
            'strategy_params': strategy_params
        }
    
    async def _gpu_backtest(
        self, 
        data: pd.DataFrame, 
        strategy_params: Dict[str, Any], 
        strategy_func: Callable
    ) -> Dict[str, Any]:
        """Execute backtesting using GPU acceleration."""
        self.logger.info("🚀 Using GPU acceleration for backtesting")
        
        # Convert data to GPU-compatible format
        gpu_data = self._prepare_gpu_data(data)
        
        # Execute GPU backtesting
        results = await m1_backtesting_simulate(
            gpu_data, 
            strategy_params, 
            self.config,
            strategy_func
        )
        
        return results
    
    async def _cpu_backtest(
        self, 
        data: pd.DataFrame, 
        strategy_params: Dict[str, Any], 
        strategy_func: Callable
    ) -> Dict[str, Any]:
        """Execute backtesting using CPU."""
        # Initialize portfolio
        portfolio = {
            'cash': self.config.initial_capital,
            'position': 0.0,
            'equity': self.config.initial_capital,
            'trades': []
        }
        
        # Execute strategy on each bar
        for i, (timestamp, bar) in enumerate(data.iterrows()):
            # Get signal from strategy
            signal = await strategy_func(
                data.iloc[:i+1], 
                mode='predict', 
                params=strategy_params
            )
            
            # Execute trade based on signal
            if signal is not None:
                await self._execute_trade(portfolio, bar, signal, timestamp)
        
        # Calculate performance metrics
        return self._calculate_metrics(portfolio, data)
    
    def _prepare_gpu_data(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare data for GPU processing."""
        # Convert to numpy array with required features
        features = ['open', 'high', 'low', 'close', 'volume']
        gpu_data = data[features].values.astype(np.float32)
        
        return gpu_data
    
    async def _execute_trade(
        self,
        portfolio: Dict[str, Any],
        bar: pd.Series,
        signal: Dict[str, Any],
        timestamp: pd.Timestamp
    ):
        """Execute a trade based on signal."""
        if signal['action'] == 'buy' and portfolio['position'] == 0:
            # Calculate position size
            position_size = min(
                signal.get('size', 0.1),
                self.config.max_position_size
            )
            
            # Calculate shares to buy
            shares = (portfolio['cash'] * position_size) / bar['close']
            
            # Execute buy
            cost = shares * bar['close'] * (1 + self.config.commission_rate)
            if cost <= portfolio['cash']:
                portfolio['cash'] -= cost
                portfolio['position'] = shares
                
                portfolio['trades'].append({
                    'timestamp': timestamp,
                    'action': 'buy',
                    'price': bar['close'],
                    'shares': shares,
                    'cost': cost
                })
        
        elif signal['action'] == 'sell' and portfolio['position'] > 0:
            # Execute sell
            proceeds = portfolio['position'] * bar['close'] * (1 - self.config.commission_rate)
            portfolio['cash'] += proceeds
            
            portfolio['trades'].append({
                'timestamp': timestamp,
                'action': 'sell',
                'price': bar['close'],
                'shares': portfolio['position'],
                'proceeds': proceeds
            })
            
            portfolio['position'] = 0.0
        
        # Update equity
        portfolio['equity'] = portfolio['cash'] + (portfolio['position'] * bar['close'])

    def _calculate_turnover_metrics(
        self,
        trades: List[Dict[str, Any]],
        initial_equity: float,
        final_equity: float
    ) -> Dict[str, float]:
        """Calculate turnover, holding period, and capacity utilization metrics."""
        if not trades:
            return {
                'turnover': 0.0,
                'average_holding_period_days': 0.0,
                'capacity_utilization': 0.0,
                'capacity_limit': self.config.capacity_limit,
                'market_impact_cost': 0.0
            }

        sorted_trades = sorted(trades, key=lambda t: t.get('timestamp'))
        total_notional = 0.0
        holding_periods: List[float] = []
        open_positions: List[Dict[str, Any]] = []

        for trade in sorted_trades:
            price = float(trade.get('price', 0.0))
            shares = float(trade.get('shares', 0.0))
            total_notional += abs(price * shares)

            action = str(trade.get('action', '')).lower()
            if action == 'buy':
                open_positions.append(trade)
            elif action == 'sell' and open_positions:
                entry_trade = open_positions.pop(0)
                entry_time = entry_trade.get('timestamp')
                exit_time = trade.get('timestamp')

                if isinstance(entry_time, pd.Timestamp) and isinstance(exit_time, pd.Timestamp):
                    holding_period = max((exit_time - entry_time).total_seconds() / 86400, 0.0)
                    holding_periods.append(holding_period)

        average_equity = (initial_equity + final_equity) / 2 if final_equity > 0 else initial_equity
        turnover = total_notional / average_equity if average_equity > 0 else 0.0

        capacity_limit = self.config.capacity_limit if self.config.capacity_limit > 0 else None
        if capacity_limit:
            capacity_utilization = turnover / capacity_limit
        else:
            capacity_utilization = turnover

        market_impact_cost = turnover * self.config.market_impact_coefficient

        if capacity_limit:
            if capacity_utilization > 1.0:
                self.logger.warning(
                    "⚠️ Capacity limit exceeded: utilization %.2fx exceeds limit %.2fx",
                    capacity_utilization,
                    1.0
                )
            elif capacity_utilization > self.config.turnover_warning_threshold:
                self.logger.warning(
                    "⚠️ Capacity utilization approaching limit: %.2f%% of allowable turnover",
                    capacity_utilization * 100
                )

        average_holding_period = float(np.mean(holding_periods)) if holding_periods else 0.0

        return {
            'turnover': turnover,
            'average_holding_period_days': average_holding_period,
            'capacity_utilization': capacity_utilization,
            'capacity_limit': capacity_limit or 0.0,
            'market_impact_cost': market_impact_cost
        }

    def _calculate_metrics(self, portfolio: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if not portfolio['trades']:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0
            }
        
        # Calculate returns
        initial_equity = self.config.initial_capital
        final_equity = portfolio['equity']
        turnover_metrics = self._calculate_turnover_metrics(portfolio['trades'], initial_equity, final_equity)

        raw_total_return = (final_equity - initial_equity) / initial_equity
        total_return = raw_total_return - turnover_metrics['market_impact_cost']

        # Calculate trade statistics
        trades = portfolio['trades']
        buy_trades = [t for t in trades if t['action'] == 'buy']
        sell_trades = [t for t in trades if t['action'] == 'sell']
        
        if len(buy_trades) > 0 and len(sell_trades) > 0:
            # Calculate P&L for each trade pair
            pnl_list = []
            for i in range(min(len(buy_trades), len(sell_trades))):
                buy_price = buy_trades[i]['price']
                sell_price = sell_trades[i]['price']
                pnl = (sell_price - buy_price) / buy_price
                pnl_list.append(pnl)
            
            win_rate = len([p for p in pnl_list if p > 0]) / len(pnl_list) if pnl_list else 0.0
            avg_return = np.mean(pnl_list) if pnl_list else 0.0
            volatility = np.std(pnl_list) if len(pnl_list) > 1 else 0.0
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0.0
        else:
            win_rate = 0.0
            sharpe_ratio = 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': 0.0,  # TODO: Calculate actual drawdown
            'win_rate': win_rate,
            'total_trades': len(trades),
            'final_equity': final_equity,
            'turnover': turnover_metrics['turnover'],
            'average_holding_period_days': turnover_metrics['average_holding_period_days'],
            'capacity_utilization': turnover_metrics['capacity_utilization'],
            'capacity_limit': turnover_metrics['capacity_limit'],
            'market_impact_cost': turnover_metrics['market_impact_cost'],
            'raw_total_return': raw_total_return
        }


class BacktestingEngine:
    """Main backtesting engine with comprehensive M1 optimizations."""
    
    def __init__(self, config: BacktestingConfig):
        """Initialize backtesting engine."""
        self.config = config
        self.logger = logger.getChild('BacktestingEngine')
        
        # Initialize components
        self.walk_forward_validator = WalkForwardValidator(config)
        
        # Initialize M1 optimizers
        self.m1_gpu = get_m1_gpu_manager() if config.enable_gpu_acceleration else None
        self.m1_memory = get_m1_memory_optimizer(
            memory_limit_gb=config.memory_limit_gb
        ) if config.enable_memory_optimization else None
        self.m1_cpu = get_m1_cpu_optimizer(
            max_workers=config.max_workers
        ) if config.enable_parallel_processing else None
        
        # Initialize utilities
        self.parquet_utils = get_parquet_utils()
        
        self.logger.info(f"🚀 BacktestingEngine initialized for {config.symbol}")
    
    @traced(span_name='execute_backtesting')
    async def execute(
        self, 
        data: pd.DataFrame,
        strategy_func: Callable,
        **strategy_kwargs
    ) -> BacktestingResults:
        """Execute comprehensive backtesting with M1 optimizations."""
        
        self.logger.info("🚀 Starting comprehensive backtesting...")
        start_time = time.time()
        
        # Memory optimization context
        if self.m1_memory:
            with self.m1_memory.optimization_context():
                results = await self._execute_backtesting(data, strategy_func, **strategy_kwargs)
        else:
            results = await self._execute_backtesting(data, strategy_func, **strategy_kwargs)
        
        execution_time = time.time() - start_time
        results.execution_time = execution_time
        
        # Log memory usage
        if self.m1_memory:
            results.memory_usage_mb = self.m1_memory.get_current_memory_usage_mb()
        
        self.logger.info(f"✅ Backtesting completed in {execution_time:.2f}s")
        self.logger.info(f"📊 Total return: {results.total_return:.2%}")
        self.logger.info(f"📈 Sharpe ratio: {results.sharpe_ratio:.2f}")
        self.logger.info(
            "🔁 Turnover: %.2f%% | Avg holding period: %.2f days",
            results.turnover * 100,
            results.average_holding_period_days
        )
        capacity_limit_display = results.capacity_limit * 100 if results.capacity_limit else 0.0
        self.logger.info(
            "📦 Capacity utilization: %.2f%% of limit %.2f%%",
            results.capacity_utilization * 100,
            capacity_limit_display
        )

        return results
    
    async def _execute_backtesting(
        self, 
        data: pd.DataFrame,
        strategy_func: Callable,
        **strategy_kwargs
    ) -> BacktestingResults:
        """Execute the actual backtesting logic."""
        
        # Perform walk-forward validation
        walk_forward_results = await self.walk_forward_validator.validate(
            data, strategy_func, **strategy_kwargs
        )
        
        # Aggregate results
        aggregated_results = self._aggregate_results(walk_forward_results)
        
        # Create comprehensive results
        results = BacktestingResults(
            symbol=self.config.symbol,
            exchange=self.config.exchange,
            timeframe=self.config.timeframe,
            start_date=data.index[0],
            end_date=data.index[-1],
            total_duration=(data.index[-1] - data.index[0]).total_seconds() / 86400,  # days
            **aggregated_results,
            walk_forward_results=walk_forward_results,
            config=self.config,
            optimization_used=self._get_optimization_used()
        )
        
        return results
    
    def _aggregate_results(self, walk_forward_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate walk-forward results into overall metrics."""
        if not walk_forward_results:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'calmar_ratio': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'volatility': 0.0,
                'var_95': 0.0,
                'cvar_95': 0.0,
                'beta': 0.0,
                'alpha': 0.0,
                'turnover': 0.0,
                'average_holding_period_days': 0.0,
                'capacity_utilization': 0.0,
                'capacity_limit': self.config.capacity_limit,
                'market_impact_cost': 0.0
            }

        # Extract metrics from each window
        returns = [r['results']['total_return'] for r in walk_forward_results if 'results' in r]
        sharpe_ratios = [r['results']['sharpe_ratio'] for r in walk_forward_results if 'results' in r]
        win_rates = [r['results']['win_rate'] for r in walk_forward_results if 'results' in r]
        total_trades = [r['results']['total_trades'] for r in walk_forward_results if 'results' in r]
        turnovers = [r['results'].get('turnover', 0.0) for r in walk_forward_results if 'results' in r]
        holding_periods = [r['results'].get('average_holding_period_days', 0.0) for r in walk_forward_results if 'results' in r]
        capacity_utilizations = [r['results'].get('capacity_utilization', 0.0) for r in walk_forward_results if 'results' in r]
        market_impacts = [r['results'].get('market_impact_cost', 0.0) for r in walk_forward_results if 'results' in r]

        # Calculate aggregated metrics
        total_return = np.mean(returns) if returns else 0.0
        annualized_return = total_return * (252 / self.config.testing_window_days) if self.config.testing_window_days > 0 else 0.0
        sharpe_ratio = np.mean(sharpe_ratios) if sharpe_ratios else 0.0
        win_rate = np.mean(win_rates) if win_rates else 0.0
        total_trades_sum = sum(total_trades) if total_trades else 0
        average_turnover = float(np.mean(turnovers)) if turnovers else 0.0
        average_holding_period = float(np.mean(holding_periods)) if holding_periods else 0.0
        average_capacity_utilization = float(np.mean(capacity_utilizations)) if capacity_utilizations else 0.0
        average_market_impact = float(np.mean(market_impacts)) if market_impacts else 0.0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': 0.0,  # TODO: Calculate Sortino ratio
            'max_drawdown': 0.0,   # TODO: Calculate max drawdown
            'calmar_ratio': 0.0,   # TODO: Calculate Calmar ratio
            'total_trades': total_trades_sum,
            'winning_trades': int(total_trades_sum * win_rate),
            'losing_trades': int(total_trades_sum * (1 - win_rate)),
            'win_rate': win_rate,
            'profit_factor': 0.0,  # TODO: Calculate profit factor
            'average_win': 0.0,    # TODO: Calculate average win
            'average_loss': 0.0,   # TODO: Calculate average loss
            'volatility': np.std(returns) if len(returns) > 1 else 0.0,
            'var_95': np.percentile(returns, 5) if returns else 0.0,
            'cvar_95': 0.0,        # TODO: Calculate CVaR
            'beta': 0.0,           # TODO: Calculate beta
            'alpha': 0.0,          # TODO: Calculate alpha
            'turnover': average_turnover,
            'average_holding_period_days': average_holding_period,
            'capacity_utilization': average_capacity_utilization,
            'capacity_limit': self.config.capacity_limit,
            'market_impact_cost': average_market_impact
        }
    
    def _get_optimization_used(self) -> List[str]:
        """Get list of optimizations used."""
        optimizations = []
        
        if self.config.enable_gpu_acceleration and self.m1_gpu:
            optimizations.append("m1_gpu_acceleration")
        
        if self.config.enable_memory_optimization and self.m1_memory:
            optimizations.append("m1_memory_optimization")
        
        if self.config.enable_parallel_processing and self.m1_cpu:
            optimizations.append("m1_parallel_processing")
        
        return optimizations
    
    async def save_results(self, results: BacktestingResults, output_dir: str) -> None:
        """Save backtesting results to disk."""
        ensure_directory(output_dir)
        
        # Save detailed results
        if self.config.save_detailed_results:
            results_file = f"{output_dir}/{self.config.symbol}_{self.config.exchange}_backtesting_results.json"
            await safe_json_dump(results_file, results.__dict__)
            self.logger.info(f"💾 Results saved to {results_file}")
        
        # Save walk-forward results
        if results.walk_forward_results:
            wf_file = f"{output_dir}/{self.config.symbol}_{self.config.exchange}_walk_forward_results.parquet"
            wf_df = pd.DataFrame(results.walk_forward_results)
            await self.parquet_utils.save_dataframe(wf_df, wf_file)
            self.logger.info(f"💾 Walk-forward results saved to {wf_file}")
        
        # Save equity curve if available
        if not results.equity_curve.empty:
            equity_file = f"{output_dir}/{self.config.symbol}_{self.config.exchange}_equity_curve.parquet"
            await self.parquet_utils.save_dataframe(results.equity_curve, equity_file)
            self.logger.info(f"💾 Equity curve saved to {equity_file}")
        
        # Save trade log if available
        if not results.trade_log.empty:
            trades_file = f"{output_dir}/{self.config.symbol}_{self.config.exchange}_trade_log.parquet"
            await self.parquet_utils.save_dataframe(results.trade_log, trades_file)
            self.logger.info(f"💾 Trade log saved to {trades_file}")