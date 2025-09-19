"""
A/B/C Testing Framework for Multi-Model Paper Trading

This module provides a comprehensive framework for testing 3+ models simultaneously
with paper trading, statistical validation, and performance comparison.

Key Features:
- Multi-model orchestration and coordination
- Paper trading with realistic market simulation
- Statistical significance testing
- Performance monitoring and alerting
- Risk management across all models
- Comprehensive reporting and visualization
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

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

# Model management
from src.utils.standardized_model_manager import StandardizedModelManager
from src.utils.ml_common.models.model_factory import EnhancedModelFactory, ModelConfig, ModelType

# Backtesting components
from src.training.steps.backtesting.ab_testing import ABTestingStep, ABTestingConfig, ABTestingResults
from src.utils.common_ml.backtesting.ab_testing_engine import ABTestingEngine, ABTestConfig, ABTestResults
from src.utils.ml_common.vectorized_backtesting import VectorizedBacktestingEngine, VectorizedBacktestConfig

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

logger = logging.getLogger(__name__)


class TestMode(Enum):
    """Testing modes for the framework."""
    PAPER_TRADING = "paper_trading"
    BACKTESTING = "backtesting"
    HYBRID = "hybrid"
    LIVE_SIMULATION = "live_simulation"


class ModelStatus(Enum):
    """Model status during testing."""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class ModelTestConfig:
    """Configuration for individual model testing."""
    # Model identification
    model_id: str
    model_name: str
    model_type: str
    
    # Model instance or factory config
    model_instance: Optional[Any] = None
    model_factory_config: Optional[ModelConfig] = None
    
    # Trading parameters
    initial_capital: float = 100000.0
    max_position_size: float = 0.1
    risk_per_trade: float = 0.02
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.1
    
    # Model-specific parameters
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Testing parameters
    enable_risk_management: bool = True
    enable_position_sizing: bool = True
    enable_stop_loss: bool = True
    enable_take_profit: bool = True
    
    # Performance tracking
    track_detailed_metrics: bool = True
    save_trade_log: bool = True
    save_performance_data: bool = True


@dataclass
class ABCTestingConfig:
    """Configuration for A/B/C testing framework."""
    # Basic configuration
    test_name: str
    test_description: str
    symbol: str
    exchange: str
    timeframe: str
    data_dir: str
    
    # Test duration and timing
    start_date: datetime
    end_date: datetime
    test_mode: TestMode = TestMode.PAPER_TRADING
    
    # Model configurations
    model_configs: List[ModelTestConfig] = field(default_factory=list)
    
    # Statistical testing
    enable_statistical_testing: bool = True
    confidence_level: float = 0.95
    alpha: float = 0.05
    min_sample_size: int = 100
    
    # Risk management
    global_risk_limit: float = 0.2
    max_concurrent_positions: int = 5
    correlation_threshold: float = 0.7
    
    # Performance settings
    enable_parallel_execution: bool = True
    max_workers: int = 4
    enable_memory_optimization: bool = True
    memory_limit_gb: float = 8.0
    
    # Monitoring and alerting
    enable_real_time_monitoring: bool = True
    performance_check_interval: int = 300  # seconds
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'max_drawdown': 0.15,
        'min_sharpe_ratio': 0.5,
        'max_correlation': 0.8
    })
    
    # Output settings
    save_detailed_results: bool = True
    generate_plots: bool = True
    output_format: str = "parquet"


@dataclass
class ModelTestResults:
    """Results from individual model testing."""
    # Model identification
    model_id: str
    model_name: str
    model_type: str
    
    # Test period
    start_time: datetime
    end_time: datetime
    duration: float
    
    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    
    # Risk metrics
    volatility: float
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
    
    # Portfolio metrics
    final_portfolio_value: float
    peak_portfolio_value: float
    final_position: float
    
    # Detailed data
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    
    # Metadata
    status: ModelStatus = ModelStatus.COMPLETED
    error_message: Optional[str] = None
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0


@dataclass
class ABCTestingResults:
    """Results from A/B/C testing framework."""
    # Test information
    test_name: str
    test_description: str
    symbol: str
    exchange: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    total_duration: float
    
    # Model results
    model_results: List[ModelTestResults] = field(default_factory=list)
    
    # Statistical analysis
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    performance_ranking: List[Dict[str, Any]] = field(default_factory=list)
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Risk analysis
    risk_metrics: Dict[str, Any] = field(default_factory=dict)
    diversification_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    best_performing_model: Optional[str] = None
    most_robust_model: Optional[str] = None
    
    # Metadata
    config: ABCTestingConfig = field(default_factory=ABCTestingConfig)
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    system_metrics: Dict[str, Any] = field(default_factory=dict)


class PaperTradingEngine:
    """Paper trading engine for realistic market simulation."""
    
    def __init__(self, config: ModelTestConfig):
        """Initialize paper trading engine."""
        self.config = config
        self.logger = logger.getChild(f'PaperTradingEngine_{config.model_id}')
        
        # Portfolio state
        self.cash = config.initial_capital
        self.position = 0.0
        self.portfolio_value = config.initial_capital
        self.peak_value = config.initial_capital
        
        # Trade tracking
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        
        self.logger.info(f"🚀 PaperTradingEngine initialized for {config.model_name}")
        self.logger.info(f"💰 Initial capital: ${config.initial_capital:,.2f}")
    
    def execute_trade(self, signal: Dict[str, Any], market_data: pd.Series, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """Execute a trade based on signal and market data."""
        
        action = signal.get('action', 'hold')
        size = signal.get('size', 0.0)
        price = market_data['close']
        
        trade_result = {
            'timestamp': timestamp,
            'action': action,
            'price': price,
            'size': size,
            'executed': False,
            'reason': 'no_action'
        }
        
        if action == 'buy' and self.position == 0:
            # Calculate position size
            position_value = min(
                self.cash * self.config.max_position_size,
                self.cash * size if size > 0 else self.cash * self.config.max_position_size
            )
            
            shares = position_value / price
            
            # Check if we have enough cash
            if position_value <= self.cash:
                self.cash -= position_value
                self.position = shares
                
                trade_result.update({
                    'executed': True,
                    'shares': shares,
                    'value': position_value,
                    'reason': 'buy_executed'
                })
                
                self.trades.append(trade_result.copy())
                self.total_trades += 1
                
                self.logger.debug(f"✅ Buy executed: {shares:.4f} shares at ${price:.2f}")
        
        elif action == 'sell' and self.position > 0:
            # Execute sell
            proceeds = self.position * price
            self.cash += proceeds
            
            # Calculate P&L
            cost_basis = sum([t['value'] for t in self.trades if t['action'] == 'buy' and t['executed']])
            pnl = proceeds - cost_basis
            
            trade_result.update({
                'executed': True,
                'shares': self.position,
                'value': proceeds,
                'pnl': pnl,
                'reason': 'sell_executed'
            })
            
            self.trades.append(trade_result.copy())
            self.total_trades += 1
            
            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            self.total_pnl += pnl
            self.position = 0.0
            
            self.logger.debug(f"✅ Sell executed: {self.position:.4f} shares at ${price:.2f}, P&L: ${pnl:.2f}")
        
        # Update portfolio value
        self.portfolio_value = self.cash + (self.position * price)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        
        # Record equity curve
        self.equity_curve.append({
            'timestamp': timestamp,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'position': self.position,
            'position_value': self.position * price
        })
        
        return trade_result
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        
        if not self.equity_curve:
            return {}
        
        # Convert to DataFrame for easier calculation
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        equity_df = equity_df.set_index('timestamp')
        
        # Calculate returns
        returns = equity_df['portfolio_value'].pct_change().dropna()
        
        # Basic metrics
        total_return = (self.portfolio_value - self.config.initial_capital) / self.config.initial_capital
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = (returns.mean() * 252) / volatility if volatility > 0 else 0
        
        # Drawdown
        peak = equity_df['portfolio_value'].expanding().max()
        drawdown = (equity_df['portfolio_value'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Trade metrics
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        profit_factor = self._calculate_profit_factor()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'final_portfolio_value': self.portfolio_value,
            'peak_portfolio_value': self.peak_value
        }
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor."""
        if not self.trades:
            return 0.0
        
        gross_profit = sum([t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0])
        gross_loss = abs(sum([t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) < 0]))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0


class ABCTestingFramework:
    """Comprehensive A/B/C testing framework for multi-model paper trading."""
    
    def __init__(self, config: ABCTestingConfig):
        """Initialize the A/B/C testing framework."""
        self.config = config
        self.logger = logger.getChild('ABCTestingFramework')
        
        # Initialize components
        self.model_manager = StandardizedModelManager()
        self.model_factory = EnhancedModelFactory()
        self.parquet_utils = get_parquet_utils()
        self.performance_monitor = PerformanceMonitor()
        self.system_monitor = SystemMonitor()
        
        # Initialize data directory
        self.data_dir = Path(config.data_dir)
        ensure_directory(self.data_dir)
        
        # Model instances and paper trading engines
        self.models: Dict[str, Any] = {}
        self.paper_trading_engines: Dict[str, PaperTradingEngine] = {}
        self.model_status: Dict[str, ModelStatus] = {}
        
        self.logger.info(f"🚀 ABCTestingFramework initialized for {config.test_name}")
        self.logger.info(f"📊 Testing {len(config.model_configs)} models")
        self.logger.info(f"📅 Test period: {config.start_date} to {config.end_date}")
        self.logger.info(f"🎯 Test mode: {config.test_mode.value}")
    
    @traced(span_name='abc_testing')
    @log_execution_time
    @monitor_step_execution
    async def execute_testing(self) -> ABCTestingResults:
        """Execute comprehensive A/B/C testing."""
        
        self.logger.info("🚀 Starting A/B/C testing...")
        start_time = time.time()
        
        # Initialize memory optimizer
        from ..memory_optimizer import memory_managed_backtesting
        
        with memory_managed_backtesting("abc_testing") as memory_optimizer:
            try:
                # Initialize models and paper trading engines
                await self._initialize_models()
                
                # Load market data
                market_data = await self._load_market_data()
                
                # Optimize market data for memory efficiency
                market_data = memory_optimizer.optimize_dataframe(market_data)
                
                # Execute paper trading
                if self.config.test_mode == TestMode.PAPER_TRADING:
                    await self._execute_paper_trading(market_data)
                elif self.config.test_mode == TestMode.BACKTESTING:
                    await self._execute_backtesting(market_data)
                else:
                    raise ValueError(f"Unsupported test mode: {self.config.test_mode}")
                
                # Collect results
                model_results = await self._collect_model_results()
                
                # Perform statistical analysis
                statistical_analysis = await self._perform_statistical_analysis(model_results)
                
                # Generate recommendations
                recommendations = await self._generate_recommendations(model_results, statistical_analysis)
                
                # Create comprehensive results
                results = ABCTestingResults(
                    test_name=self.config.test_name,
                    test_description=self.config.test_description,
                    symbol=self.config.symbol,
                    exchange=self.config.exchange,
                    timeframe=self.config.timeframe,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    total_duration=time.time() - start_time,
                    model_results=model_results,
                    statistical_tests=statistical_analysis.get('statistical_tests', {}),
                    performance_ranking=statistical_analysis.get('performance_ranking', []),
                    correlation_matrix=statistical_analysis.get('correlation_matrix', pd.DataFrame()),
                    risk_metrics=statistical_analysis.get('risk_metrics', {}),
                    diversification_metrics=statistical_analysis.get('diversification_metrics', {}),
                    recommendations=recommendations,
                    best_performing_model=statistical_analysis.get('best_performing_model'),
                    most_robust_model=statistical_analysis.get('most_robust_model'),
                    config=self.config,
                    execution_time=time.time() - start_time,
                    memory_usage_mb=memory_optimizer.get_current_memory_stats().process_memory_mb,
                    system_metrics=self._get_system_metrics()
                )
                
                # Save results
                if self.config.save_detailed_results:
                    await self._save_results(results)
                
                self.logger.info("✅ A/B/C testing completed successfully")
                self.logger.info(f"⏱️ Execution time: {results.execution_time:.2f}s")
                self.logger.info(f"📊 Models tested: {len(model_results)}")
                
                return results
                
            except Exception as e:
                self.logger.error(f"❌ Error in A/B/C testing: {e}")
                self.logger.exception("Full traceback:")
                raise
            finally:
                # Cleanup
                await self._cleanup()
    
    async def _initialize_models(self) -> None:
        """Initialize all models and paper trading engines."""
        self.logger.info("🔧 Initializing models and paper trading engines...")
        
        for model_config in self.config.model_configs:
            try:
                # Initialize model
                if model_config.model_instance is not None:
                    model = model_config.model_instance
                elif model_config.model_factory_config is not None:
                    model = self.model_factory.create_model(model_config.model_factory_config)
                else:
                    raise ValueError(f"No model instance or factory config provided for {model_config.model_id}")
                
                self.models[model_config.model_id] = model
                
                # Initialize paper trading engine
                paper_engine = PaperTradingEngine(model_config)
                self.paper_trading_engines[model_config.model_id] = paper_engine
                
                # Set initial status
                self.model_status[model_config.model_id] = ModelStatus.ACTIVE
                
                self.logger.info(f"✅ Initialized {model_config.model_name} ({model_config.model_id})")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize {model_config.model_id}: {e}")
                self.model_status[model_config.model_id] = ModelStatus.ERROR
    
    async def _load_market_data(self) -> pd.DataFrame:
        """Load market data for the test period using unified data loader."""
        from ..unified_data_loader import DataLoadingConfig, get_unified_data_loader
        
        self.logger.info("📂 Loading market data...")
        
        try:
            # Create loading configuration
            loading_config = DataLoadingConfig(
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                timeframe=self.config.timeframe,
                data_dir=str(self.data_dir),
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                enable_memory_optimization=True,
                memory_limit_mb=1500.0  # Higher limit for ABC testing
            )
            
            # Load data using unified loader
            loader = get_unified_data_loader()
            loaded_data = loader.load_data(loading_config)
            
            self.logger.info(f"✅ Loaded data via unified loader:")
            self.logger.info(f"   📊 Records: {len(loaded_data.data):,}")
            self.logger.info(f"   🧠 Memory: {loaded_data.memory_usage_mb:.1f}MB")
            self.logger.info(f"   🎯 Quality: {loaded_data.data_quality_score:.2f}")
            
            return loaded_data.data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load data via unified loader: {e}")
            self.logger.warning("⚠️ Falling back to sample data generation...")
            return self._generate_sample_data()
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample market data for testing."""
        self.logger.info("🔄 Generating sample market data...")
        
        # Generate date range
        date_range = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq=self.config.timeframe
        )
        
        # Generate sample OHLCV data
        np.random.seed(42)
        n_periods = len(date_range)
        
        # Start with base price
        base_price = 100.0
        returns = np.random.normal(0.0001, 0.02, n_periods)  # Small positive drift with volatility
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLCV
        data = []
        for i, (timestamp, price) in enumerate(zip(date_range, prices)):
            # Generate realistic OHLC from close price
            volatility = abs(np.random.normal(0, 0.01))
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            open_price = prices[i-1] if i > 0 else price
            volume = np.random.uniform(1000, 10000)
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df = df.set_index('timestamp')
        
        return df
    
    async def _execute_paper_trading(self, market_data: pd.DataFrame) -> None:
        """Execute paper trading for all models."""
        self.logger.info("📈 Executing paper trading...")
        
        # Process each time period
        for timestamp, bar in market_data.iterrows():
            # Get signals from all active models
            signals = await self._get_model_signals(timestamp, bar)
            
            # Execute trades for each model
            for model_id, signal in signals.items():
                if self.model_status[model_id] == ModelStatus.ACTIVE:
                    try:
                        paper_engine = self.paper_trading_engines[model_id]
                        trade_result = paper_engine.execute_trade(signal, bar, timestamp)
                        
                        # Log significant trades
                        if trade_result['executed']:
                            self.logger.debug(f"📊 {model_id}: {trade_result['action']} executed at ${trade_result['price']:.2f}")
                    
                    except Exception as e:
                        self.logger.error(f"❌ Error executing trade for {model_id}: {e}")
                        self.model_status[model_id] = ModelStatus.ERROR
            
            # Check for risk limits
            await self._check_risk_limits()
            
            # Update monitoring
            if self.config.enable_real_time_monitoring:
                await self._update_monitoring()
    
    async def _get_model_signals(self, timestamp: pd.Timestamp, bar: pd.Series) -> Dict[str, Dict[str, Any]]:
        """Get trading signals from all models."""
        signals = {}
        
        for model_id, model in self.models.items():
            if self.model_status[model_id] == ModelStatus.ACTIVE:
                try:
                    # This is a placeholder - in practice, you would call the model's predict method
                    # with appropriate features and get trading signals
                    signal = await self._generate_sample_signal(model_id, bar)
                    signals[model_id] = signal
                
                except Exception as e:
                    self.logger.error(f"❌ Error getting signal from {model_id}: {e}")
                    signals[model_id] = {'action': 'hold', 'size': 0.0}
        
        return signals
    
    async def _generate_sample_signal(self, model_id: str, bar: pd.Series) -> Dict[str, Any]:
        """Generate trading signal using improved strategy (replaces random generation)."""
        try:
            from ..improved_trading_strategies import StrategyFactory, StrategyType
            
            # Create different strategies for different models to simulate diversity
            strategy_types = {
                'model_a': StrategyType.TREND_FOLLOWING,
                'model_b': StrategyType.MEAN_REVERSION,
                'model_c': StrategyType.MOMENTUM,
                'model_d': StrategyType.VOLATILITY_BREAKOUT,
                'model_e': StrategyType.ADAPTIVE
            }
            
            # Determine strategy type based on model_id
            strategy_type = StrategyType.ADAPTIVE  # Default
            for key, stype in strategy_types.items():
                if key in model_id.lower():
                    strategy_type = stype
                    break
            
            # Create strategy
            strategy = StrategyFactory.create_strategy(strategy_type)
            
            # Create minimal DataFrame for signal generation
            # Note: In practice, you would have access to historical data
            # For now, we'll create a simple DataFrame with the current bar
            if hasattr(self, '_historical_data') and not self._historical_data.empty:
                signal = strategy.generate_signal(self._historical_data, bar.name)
            else:
                # Fallback to simple technical analysis
                signal = self._generate_technical_signal(bar)
            
            return {
                'action': signal.action,
                'size': signal.position_size,
                'confidence': signal.confidence,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'reasoning': signal.reasoning
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating improved signal for {model_id}: {e}")
            # Fallback to simple technical signal
            return self._generate_technical_signal(bar)
    
    def _generate_technical_signal(self, bar: pd.Series) -> Dict[str, Any]:
        """Generate simple technical signal as fallback."""
        # Simple momentum-based signal
        if hasattr(bar, 'close') and hasattr(bar, 'open'):
            price_change = (bar['close'] - bar['open']) / bar['open']
            
            if price_change > 0.005:  # 0.5% gain
                return {
                    'action': 'buy',
                    'size': 0.1,
                    'confidence': min(0.8, 0.5 + abs(price_change) * 10)
                }
            elif price_change < -0.005:  # 0.5% loss
                return {
                    'action': 'sell',
                    'size': 0.1,
                    'confidence': min(0.8, 0.5 + abs(price_change) * 10)
                }
        
        return {'action': 'hold', 'size': 0.0, 'confidence': 0.5}
    
    async def _check_risk_limits(self) -> None:
        """Check global risk limits across all models."""
        # Calculate total exposure
        total_exposure = 0.0
        for paper_engine in self.paper_trading_engines.values():
            total_exposure += abs(paper_engine.position)
        
        # Check if exposure exceeds limits
        if total_exposure > self.config.global_risk_limit:
            self.logger.warning(f"⚠️ Total exposure {total_exposure:.2%} exceeds limit {self.config.global_risk_limit:.2%}")
            # Implement risk management actions here
    
    async def _update_monitoring(self) -> None:
        """Update real-time monitoring."""
        # This would implement real-time monitoring and alerting
        pass
    
    async def _collect_model_results(self) -> List[ModelTestResults]:
        """Collect results from all models."""
        self.logger.info("📊 Collecting model results...")
        
        results = []
        for model_config in self.config.model_configs:
            model_id = model_config.model_id
            
            try:
                paper_engine = self.paper_trading_engines[model_id]
                performance_metrics = paper_engine.get_performance_metrics()
                
                # Create equity curve DataFrame
                equity_curve = pd.DataFrame(paper_engine.equity_curve)
                if not equity_curve.empty:
                    equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
                    equity_curve = equity_curve.set_index('timestamp')
                
                # Create trade log DataFrame
                trade_log = pd.DataFrame(paper_engine.trades)
                
                # Calculate daily returns
                daily_returns = pd.Series()
                if not equity_curve.empty:
                    daily_returns = equity_curve['portfolio_value'].pct_change().dropna()
                
                result = ModelTestResults(
                    model_id=model_id,
                    model_name=model_config.model_name,
                    model_type=model_config.model_type,
                    start_time=self.config.start_date,
                    end_time=self.config.end_date,
                    duration=(self.config.end_date - self.config.start_date).total_seconds() / 86400,
                    **performance_metrics,
                    equity_curve=equity_curve,
                    trade_log=trade_log,
                    daily_returns=daily_returns,
                    status=self.model_status[model_id]
                )
                
                results.append(result)
                self.logger.info(f"✅ Collected results for {model_config.model_name}")
                
            except Exception as e:
                self.logger.error(f"❌ Error collecting results for {model_id}: {e}")
                # Create error result
                error_result = ModelTestResults(
                    model_id=model_id,
                    model_name=model_config.model_name,
                    model_type=model_config.model_type,
                    start_time=self.config.start_date,
                    end_time=self.config.end_date,
                    duration=0.0,
                    status=ModelStatus.ERROR,
                    error_message=str(e)
                )
                results.append(error_result)
        
        return results
    
    async def _perform_statistical_analysis(self, model_results: List[ModelTestResults]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        self.logger.info("📈 Performing statistical analysis...")
        
        if not model_results:
            return {}
        
        # Extract performance metrics
        metrics_df = pd.DataFrame([{
            'model_id': r.model_id,
            'model_name': r.model_name,
            'total_return': r.total_return,
            'sharpe_ratio': r.sharpe_ratio,
            'max_drawdown': r.max_drawdown,
            'win_rate': r.win_rate,
            'volatility': r.volatility
        } for r in model_results if r.status == ModelStatus.COMPLETED])
        
        if metrics_df.empty:
            return {}
        
        # Performance ranking
        performance_ranking = []
        for metric in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']:
            if metric in metrics_df.columns:
                sorted_models = metrics_df.sort_values(metric, ascending=(metric == 'max_drawdown')).head(3)
                performance_ranking.append({
                    'metric': metric,
                    'top_models': sorted_models[['model_id', 'model_name', metric]].to_dict('records')
                })
        
        # Correlation analysis
        correlation_matrix = pd.DataFrame()
        if len(model_results) > 1:
            # Extract daily returns for correlation analysis
            returns_data = {}
            for result in model_results:
                if not result.daily_returns.empty:
                    returns_data[result.model_id] = result.daily_returns
            
            if len(returns_data) > 1:
                returns_df = pd.DataFrame(returns_data)
                correlation_matrix = returns_df.corr()
        
        # Risk metrics
        risk_metrics = {
            'average_volatility': metrics_df['volatility'].mean(),
            'max_volatility': metrics_df['volatility'].max(),
            'average_drawdown': metrics_df['max_drawdown'].mean(),
            'max_drawdown': metrics_df['max_drawdown'].min()
        }
        
        # Diversification metrics
        diversification_metrics = {}
        if not correlation_matrix.empty:
            avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            diversification_metrics = {
                'average_correlation': avg_correlation,
                'diversification_ratio': 1 - avg_correlation
            }
        
        # Statistical tests (if enabled)
        statistical_tests = {}
        if self.config.enable_statistical_testing and len(model_results) > 1:
            statistical_tests = await self._perform_statistical_tests(model_results)
        
        # Find best performing and most robust models
        best_performing_model = None
        most_robust_model = None
        
        if not metrics_df.empty:
            # Best performing: highest Sharpe ratio
            best_performing = metrics_df.loc[metrics_df['sharpe_ratio'].idxmax()]
            best_performing_model = best_performing['model_id']
            
            # Most robust: lowest volatility with positive returns
            robust_models = metrics_df[metrics_df['total_return'] > 0]
            if not robust_models.empty:
                most_robust = robust_models.loc[robust_models['volatility'].idxmin()]
                most_robust_model = most_robust['model_id']
        
        return {
            'statistical_tests': statistical_tests,
            'performance_ranking': performance_ranking,
            'correlation_matrix': correlation_matrix,
            'risk_metrics': risk_metrics,
            'diversification_metrics': diversification_metrics,
            'best_performing_model': best_performing_model,
            'most_robust_model': most_robust_model
        }
    
    async def _perform_statistical_tests(self, model_results: List[ModelTestResults]) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        # This would implement various statistical tests
        # For now, return empty dict
        return {}
    
    async def _generate_recommendations(self, model_results: List[ModelTestResults], 
                                      statistical_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Performance-based recommendations
        if statistical_analysis.get('best_performing_model'):
            recommendations.append({
                'category': 'PERFORMANCE',
                'priority': 'HIGH',
                'title': 'Best Performing Model',
                'description': f"Model {statistical_analysis['best_performing_model']} shows the best risk-adjusted returns",
                'action': 'Consider deploying this model in production',
                'confidence': 'HIGH'
            })
        
        # Risk-based recommendations
        risk_metrics = statistical_analysis.get('risk_metrics', {})
        if risk_metrics.get('max_drawdown', 0) < -0.2:
            recommendations.append({
                'category': 'RISK',
                'priority': 'HIGH',
                'title': 'High Drawdown Risk',
                'description': f"Maximum drawdown of {risk_metrics['max_drawdown']:.2%} exceeds acceptable limits",
                'action': 'Implement additional risk management measures',
                'confidence': 'HIGH'
            })
        
        # Diversification recommendations
        diversification_metrics = statistical_analysis.get('diversification_metrics', {})
        if diversification_metrics.get('average_correlation', 0) > 0.8:
            recommendations.append({
                'category': 'DIVERSIFICATION',
                'priority': 'MEDIUM',
                'title': 'High Model Correlation',
                'description': f"Average correlation of {diversification_metrics['average_correlation']:.2f} indicates low diversification",
                'action': 'Consider adding more diverse models to the portfolio',
                'confidence': 'MEDIUM'
            })
        
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
    
    async def _save_results(self, results: ABCTestingResults) -> None:
        """Save comprehensive results to disk."""
        self.logger.info("💾 Saving results...")
        
        # Create output directory
        output_dir = self.data_dir / "abc_testing_results" / self.config.test_name
        ensure_directory(output_dir)
        
        # Save main results
        results_file = output_dir / f"{self.config.test_name}_abc_test_results.json"
        await safe_json_dump(results_file, results.__dict__, indent=2)
        
        # Save individual model results
        for model_result in results.model_results:
            model_file = output_dir / f"{model_result.model_id}_results.json"
            await safe_json_dump(model_file, model_result.__dict__, indent=2)
        
        # Save correlation matrix
        if not results.correlation_matrix.empty:
            corr_file = output_dir / f"{self.config.test_name}_correlation_matrix.parquet"
            await self.parquet_utils.save_dataframe(results.correlation_matrix, corr_file)
        
        self.logger.info(f"✅ Results saved to {output_dir}")
    
    async def _cleanup(self) -> None:
        """Cleanup resources."""
        self.logger.info("🧹 Cleaning up resources...")
        
        # Clear model instances
        self.models.clear()
        self.paper_trading_engines.clear()
        self.model_status.clear()
        
        # Force garbage collection
        gc.collect()
        
        self.logger.info("✅ Cleanup completed")


# Convenience function for easy integration
async def execute_abc_testing(
    test_name: str,
    model_configs: List[ModelTestConfig],
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE",
    timeframe: str = "1h",
    data_dir: str = "data/training",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    **kwargs
) -> ABCTestingResults:
    """
    Convenience function to execute A/B/C testing.
    
    Args:
        test_name: Name of the test
        model_configs: List of model configurations
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Data timeframe
        data_dir: Data directory
        start_date: Test start date
        end_date: Test end date
        **kwargs: Additional configuration parameters
        
    Returns:
        A/B/C testing results
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now()
    
    config = ABCTestingConfig(
        test_name=test_name,
        test_description=f"A/B/C testing for {len(model_configs)} models",
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date,
        model_configs=model_configs,
        **kwargs
    )
    
    framework = ABCTestingFramework(config)
    return await framework.execute_testing()