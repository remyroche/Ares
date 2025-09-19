"""
A/B Testing Step

This module provides comprehensive A/B testing functionality for comparing
different trading strategies with statistical validation and significance testing.

Key Features:
- A/B testing with statistical validation
- Multiple statistical tests (t-test, Mann-Whitney U, etc.)
- Effect size calculation and power analysis
- Sample size determination
- Confidence interval estimation
- Comprehensive reporting and recommendations
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
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency

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
from src.utils.common_ml.backtesting.ab_testing_engine import (
    ABTestingEngine, ABTestConfig, ABTestResults, TestType, MetricType, StatisticalTest
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


class ABTestStrategyType(Enum):
    """Types of strategies for A/B testing."""
    CONTROL = "control"
    TREATMENT = "treatment"
    BASELINE = "baseline"
    OPTIMIZED = "optimized"
    CUSTOM = "custom"


@dataclass
class ABTestingConfig:
    """Configuration for A/B testing step."""
    # Basic configuration
    symbol: str
    exchange: str
    timeframe: str
    data_dir: str
    
    # Test configuration
    test_name: str
    test_description: str
    control_strategy: str = "baseline"
    treatment_strategy: str = "optimized"
    
    # Sample configuration
    min_sample_size: int = 100
    max_sample_size: int = 10000
    target_sample_size: Optional[int] = None
    sampling_method: str = "random"  # random, stratified, systematic
    
    # Statistical configuration
    statistical_tests: List[str] = field(default_factory=lambda: ["t_test", "mann_whitney_u"])
    confidence_level: float = 0.95
    alpha: float = 0.05
    power: float = 0.8
    effect_size: Optional[float] = None
    
    # Performance settings
    enable_detailed_logging: bool = True
    enable_performance_monitoring: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    
    # Output settings
    save_detailed_results: bool = True
    generate_plots: bool = True
    output_format: str = "parquet"


@dataclass
class ABTestingResults:
    """Results from A/B testing step."""
    # Basic info
    symbol: str
    exchange: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    total_duration: float
    
    # Test configuration
    test_name: str
    test_description: str
    
    # Sample information
    control_group_size: int
    treatment_group_size: int
    total_sample_size: int
    
    # Test results
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    
    # Performance comparison
    performance_comparison: Dict[str, Any] = field(default_factory=dict)
    
    # Effect size and power
    effect_size_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Confidence intervals
    confidence_intervals: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Detailed data
    control_group_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    treatment_group_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Metadata
    config: ABTestingConfig = field(default_factory=ABTestingConfig)
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    system_metrics: Dict[str, Any] = field(default_factory=dict)


class ABTestingStep:
    """A/B testing step."""
    
    def __init__(self, config: ABTestingConfig):
        """Initialize the A/B testing step."""
        self.config = config
        self.logger = logger.getChild('ABTestingStep')
        
        # Initialize utilities
        self.parquet_utils = get_parquet_utils()
        self.financial_logger = EnhancedFinancialMetricsLogger()
        self.performance_monitor = PerformanceMonitor()
        self.system_monitor = SystemMonitor()
        
        # Initialize data directory
        self.data_dir = Path(config.data_dir)
        ensure_directory(self.data_dir)
        
        # Initialize A/B testing engine
        self.ab_test_config = ABTestConfig(
            symbol=config.symbol,
            exchange=config.exchange,
            timeframe=config.timeframe,
            data_dir=config.data_dir,
            test_name=config.test_name,
            test_description=config.test_description,
            control_group_name=config.control_strategy,
            treatment_group_name=config.treatment_strategy,
            min_sample_size=config.min_sample_size,
            max_sample_size=config.max_sample_size,
            target_sample_size=config.target_sample_size,
            sampling_method=config.sampling_method,
            statistical_tests=[StatisticalTest(TestType.T_TEST, MetricType.CONTINUOUS, alpha=config.alpha)],
            confidence_level=config.confidence_level,
            enable_gpu_acceleration=True,
            enable_memory_optimization=config.enable_memory_optimization,
            enable_parallel_processing=config.enable_parallel_processing
        )
        
        self.ab_testing_engine = ABTestingEngine(self.ab_test_config)
        
        self.logger.info(f"🚀 ABTestingStep initialized for {config.test_name}")
        self.logger.info(f"📊 Control strategy: {config.control_strategy}")
        self.logger.info(f"📊 Treatment strategy: {config.treatment_strategy}")
        self.logger.info(f"📊 Statistical tests: {config.statistical_tests}")
        self.logger.info(f"📁 Data directory: {config.data_dir}")
    
    @traced(span_name='ab_testing')
    @log_execution_time
    @monitor_step_execution
    async def execute(
        self, 
        control_data: Optional[pd.DataFrame] = None,
        treatment_data: Optional[pd.DataFrame] = None,
        control_strategy_func: Optional[Callable] = None,
        treatment_strategy_func: Optional[Callable] = None,
        **kwargs
    ) -> ABTestingResults:
        """Execute A/B testing."""
        
        self.logger.info("🚀 Starting A/B testing...")
        start_time = time.time()
        
        # Start performance monitoring
        if self.config.enable_performance_monitoring:
            self.performance_monitor.start_monitoring()
        
        # Initialize memory optimizer
        from .memory_optimizer import get_backtesting_memory_optimizer, memory_managed_backtesting
        
        with memory_managed_backtesting("ab_testing") as memory_optimizer:
            try:
                # Load or generate data if not provided
                if control_data is None or treatment_data is None:
                    control_data, treatment_data = await self._load_or_generate_data(
                        control_strategy_func, treatment_strategy_func
                    )
                
                # Optimize DataFrames for memory efficiency
                if control_data is not None:
                    control_data = memory_optimizer.optimize_dataframe(control_data)
                if treatment_data is not None:
                    treatment_data = memory_optimizer.optimize_dataframe(treatment_data)
                
                # Validate data
                self._validate_data(control_data, treatment_data)
                
                # Execute A/B testing
                ab_test_results = await self.ab_testing_engine.execute(
                    control_data=control_data,
                    treatment_data=treatment_data,
                    metric_columns=['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
                )
                
                # Perform additional analysis
                performance_comparison = self._compare_performance(control_data, treatment_data)
                effect_size_analysis = self._analyze_effect_size(ab_test_results)
                confidence_intervals = self._calculate_confidence_intervals(ab_test_results)
                recommendations = self._generate_recommendations(ab_test_results)
                
                # Create results
                results = ABTestingResults(
                    symbol=self.config.symbol,
                    exchange=self.config.exchange,
                    timeframe=self.config.timeframe,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    total_duration=time.time() - start_time,
                    test_name=self.config.test_name,
                    test_description=self.config.test_description,
                    control_group_size=len(control_data),
                    treatment_group_size=len(treatment_data),
                    total_sample_size=len(control_data) + len(treatment_data),
                    statistical_tests=ab_test_results.test_results,
                    performance_comparison=performance_comparison,
                    effect_size_analysis=effect_size_analysis,
                    confidence_intervals=confidence_intervals,
                    recommendations=recommendations,
                    control_group_data=control_data,
                    treatment_group_data=treatment_data,
                    config=self.config,
                    execution_time=time.time() - start_time,
                    memory_usage_mb=memory_optimizer.get_current_memory_stats().process_memory_mb,
                    system_metrics=self._get_system_metrics()
                )
                
                # Save results
                if self.config.save_detailed_results:
                    await self._save_results(results)
                
                self.logger.info("✅ A/B testing completed successfully")
                self.logger.info(f"⏱️ Execution time: {results.execution_time:.2f}s")
                self.logger.info(f"📊 Total tests: {len(ab_test_results.test_results)}")
                self.logger.info(f"✅ Significant tests: {ab_test_results.significant_tests}")
                self.logger.info(f"🎯 Overall conclusion: {ab_test_results.overall_conclusion}")
                
                return results
                
            except Exception as e:
                self.logger.error(f"❌ Error in A/B testing: {e}")
                self.logger.exception("Full traceback:")
                raise
            finally:
                # Stop performance monitoring
                if self.config.enable_performance_monitoring:
                    self.performance_monitor.stop_monitoring()
    
    async def _load_or_generate_data(
        self, 
        control_strategy_func: Optional[Callable] = None,
        treatment_strategy_func: Optional[Callable] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load or generate data for A/B testing."""
        self.logger.info("📂 Loading or generating data for A/B testing...")
        
        # Try to load existing data first
        control_data = await self._load_strategy_data(self.config.control_strategy)
        treatment_data = await self._load_strategy_data(self.config.treatment_strategy)
        
        # If data not found, generate it
        if control_data.empty or treatment_data.empty:
            self.logger.info("📊 Generating strategy data...")
            
            # Load market data
            market_data = await self._load_market_data()
            
            # Generate control strategy data
            if control_data.empty and control_strategy_func is not None:
                control_data = await self._generate_strategy_data(
                    market_data, control_strategy_func, self.config.control_strategy
                )
            elif control_data.empty:
                control_data = await self._generate_baseline_strategy_data(market_data)
            
            # Generate treatment strategy data
            if treatment_data.empty and treatment_strategy_func is not None:
                treatment_data = await self._generate_strategy_data(
                    market_data, treatment_strategy_func, self.config.treatment_strategy
                )
            elif treatment_data.empty:
                treatment_data = await self._generate_optimized_strategy_data(market_data)
        
        self.logger.info(f"📊 Control group size: {len(control_data):,}")
        self.logger.info(f"📊 Treatment group size: {len(treatment_data):,}")
        
        return control_data, treatment_data
    
    async def _load_market_data(self) -> pd.DataFrame:
        """Load market data using unified data loader."""
        from .unified_data_loader import DataLoadingConfig, get_unified_data_loader
        
        # Create loading configuration
        loading_config = DataLoadingConfig(
            symbol=self.config.symbol,
            exchange=self.config.exchange,
            timeframe=self.config.timeframe,
            data_dir=self.config.data_dir,
            enable_memory_optimization=True,
            memory_limit_mb=1000.0
        )
        
        # Load data using unified loader
        loader = get_unified_data_loader()
        loaded_data = loader.load_data(loading_config)
        
        self.logger.info(f"✅ Loaded data via unified loader:")
        self.logger.info(f"   📊 Records: {len(loaded_data.data):,}")
        self.logger.info(f"   🧠 Memory: {loaded_data.memory_usage_mb:.1f}MB")
        self.logger.info(f"   🎯 Quality: {loaded_data.data_quality_score:.2f}")
        
        return loaded_data.data
    
    async def _load_strategy_data(self, strategy_name: str) -> pd.DataFrame:
        """Load existing strategy data."""
        strategy_file = self.data_dir / "backtesting_results" / f"{strategy_name}_results.parquet"
        
        if safe_file_exists(strategy_file):
            self.logger.info(f"📁 Loading {strategy_name} strategy data: {strategy_file}")
            return standardized_parquet_handler.read_parquet_standardized(strategy_file)
        else:
            self.logger.info(f"📁 No existing data found for {strategy_name} strategy")
            return pd.DataFrame()
    
    async def _generate_strategy_data(
        self, 
        market_data: pd.DataFrame, 
        strategy_func: Callable, 
        strategy_name: str
    ) -> pd.DataFrame:
        """Generate strategy data using provided function."""
        self.logger.info(f"🔄 Generating {strategy_name} strategy data...")
        
        # Execute strategy function
        strategy_results = await strategy_func(market_data)
        
        # Convert to DataFrame format
        if isinstance(strategy_results, dict):
            strategy_data = pd.DataFrame([strategy_results])
        elif isinstance(strategy_results, pd.DataFrame):
            strategy_data = strategy_results
        else:
            # Convert other formats to DataFrame
            strategy_data = pd.DataFrame({'result': [strategy_results]})
        
        return strategy_data
    
    async def _generate_baseline_strategy_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate baseline strategy data using improved buy-and-hold implementation."""
        from .improved_trading_strategies import create_baseline_strategy
        
        self.logger.info("🔄 Generating baseline strategy data...")
        
        if market_data.empty or len(market_data) < 10:
            return pd.DataFrame([{
                'strategy': 'baseline',
                'total_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'profit_factor': 0.0,
                'calmar_ratio': 0.0
            }])
        
        try:
            # Create baseline strategy
            baseline_strategy = create_baseline_strategy()
            
            # Simple buy-and-hold execution with validation
            initial_price = market_data['close'].iloc[0]
            final_price = market_data['close'].iloc[-1]
            
            if not (validate_finite(initial_price) and validate_finite(final_price) and 
                    validate_positive(initial_price) and validate_positive(final_price)):
                raise ValueError("Invalid price data")
            
            total_return = (final_price - initial_price) / initial_price
            
            # Calculate proper metrics
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
            sharpe_ratio = (returns.mean() * 252 - 0.02) / volatility if volatility > 0 else 0  # Risk-free rate = 2%
            
            # Calculate max drawdown properly
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calculate win rate (for buy-and-hold, it's based on positive periods)
            positive_periods = (returns > 0).sum()
            win_rate = positive_periods / len(returns) if len(returns) > 0 else 0.0
            
            baseline_data = pd.DataFrame([{
                'strategy': 'baseline',
                'total_return': total_return,
                'annualized_return': (1 + total_return) ** (252 / len(market_data)) - 1,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': 1,
                'profit_factor': 1.0 if total_return > 0 else 0.0,
                'calmar_ratio': total_return / abs(max_drawdown) if max_drawdown != 0 else 0
            }])
            
            return baseline_data
            
        except Exception as e:
            self.logger.error(f"❌ Error generating baseline strategy data: {e}")
            return pd.DataFrame([{
                'strategy': 'baseline',
                'total_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'profit_factor': 0.0,
                'calmar_ratio': 0.0
            }])
    
    async def _generate_optimized_strategy_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate optimized strategy data using improved adaptive strategy."""
        from .improved_trading_strategies import create_optimized_strategy
        
        self.logger.info("🔄 Generating optimized strategy data...")
        
        if market_data.empty or len(market_data) < 50:
            return pd.DataFrame([{
                'strategy': 'optimized',
                'total_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'profit_factor': 0.0,
                'calmar_ratio': 0.0
            }])
        
        try:
            # Create optimized adaptive strategy
            optimized_strategy = create_optimized_strategy()
            
            # Execute strategy
            portfolio = {
                'cash': 100000.0,
                'shares': 0.0,
                'trades': []
            }
            
            # Generate signals and execute trades
            for i in range(50, len(market_data)):  # Start after sufficient lookback
                timestamp = market_data.index[i]
                current_price = market_data.loc[timestamp, 'close']
                
                # Generate signal
                signal = optimized_strategy.generate_signal(market_data, timestamp)
                
                # Execute trade based on signal
                if signal.action == 'buy' and portfolio['shares'] == 0:
                    # Calculate position size
                    volatility = market_data['close'].pct_change().tail(20).std()
                    position_size = optimized_strategy.risk_manager.calculate_position_size(
                        signal, portfolio['cash'] + portfolio['shares'] * current_price, volatility
                    )
                    
                    shares_to_buy = (portfolio['cash'] * position_size) / current_price
                    cost = shares_to_buy * current_price
                    
                    if cost <= portfolio['cash']:
                        portfolio['cash'] -= cost
                        portfolio['shares'] += shares_to_buy
                        portfolio['trades'].append({
                            'action': 'buy',
                            'price': current_price,
                            'timestamp': timestamp,
                            'shares': shares_to_buy,
                            'cost': cost,
                            'pnl': 0
                        })
                
                elif signal.action == 'sell' and portfolio['shares'] > 0:
                    proceeds = portfolio['shares'] * current_price
                    
                    # Calculate P&L
                    last_buy = next((t for t in reversed(portfolio['trades']) if t['action'] == 'buy'), None)
                    pnl = proceeds - last_buy['cost'] if last_buy else 0
                    
                    portfolio['cash'] += proceeds
                    portfolio['trades'].append({
                        'action': 'sell',
                        'price': current_price,
                        'timestamp': timestamp,
                        'shares': portfolio['shares'],
                        'proceeds': proceeds,
                        'pnl': pnl
                    })
                    portfolio['shares'] = 0.0
            
            # Calculate final metrics
            final_value = portfolio['cash'] + (portfolio['shares'] * market_data['close'].iloc[-1])
            total_return = (final_value - 100000.0) / 100000.0
            
            # Calculate proper performance metrics
            trade_df = pd.DataFrame(portfolio['trades'])
            
            if not trade_df.empty and 'pnl' in trade_df.columns:
                sell_trades = trade_df[trade_df['action'] == 'sell']
                if not sell_trades.empty:
                    winning_trades = len(sell_trades[sell_trades['pnl'] > 0])
                    total_trades = len(sell_trades)
                    win_rate = winning_trades / total_trades
                    
                    gross_profit = sell_trades[sell_trades['pnl'] > 0]['pnl'].sum()
                    gross_loss = abs(sell_trades[sell_trades['pnl'] < 0]['pnl'].sum())
                    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                else:
                    win_rate = 0.0
                    total_trades = 0
                    profit_factor = 0.0
            else:
                win_rate = 0.0
                total_trades = 0
                profit_factor = 0.0
            
            # Calculate risk metrics
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
            sharpe_ratio = (total_return * 252 - 0.02) / volatility if volatility > 0 else 0
            
            # Calculate max drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            optimized_data = pd.DataFrame([{
                'strategy': 'optimized',
                'total_return': total_return,
                'annualized_return': (1 + total_return) ** (252 / len(market_data)) - 1,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'profit_factor': profit_factor,
                'calmar_ratio': total_return / abs(max_drawdown) if max_drawdown != 0 else 0
            }])
            
            return optimized_data
            
        except Exception as e:
            self.logger.error(f"❌ Error generating optimized strategy data: {e}")
            return pd.DataFrame([{
                'strategy': 'optimized',
                'total_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'profit_factor': 0.0,
                'calmar_ratio': 0.0
            }])
    
    def _calculate_max_drawdown(self, price_series: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(price_series) == 0:
            return 0.0
        
        peak = price_series.expanding().max()
        drawdown = (price_series - peak) / peak
        return float(drawdown.min())
    
    def _validate_data(self, control_data: pd.DataFrame, treatment_data: pd.DataFrame) -> None:
        """Validate A/B testing data."""
        self.logger.info("🔍 Validating A/B testing data...")
        
        if control_data.empty:
            raise ValidationError("Control group data is empty")
        
        if treatment_data.empty:
            raise ValidationError("Treatment group data is empty")
        
        # Check required columns
        required_columns = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        missing_control = [col for col in required_columns if col not in control_data.columns]
        missing_treatment = [col for col in required_columns if col not in treatment_data.columns]
        
        if missing_control:
            raise ValidationError(f"Missing columns in control data: {missing_control}")
        
        if missing_treatment:
            raise ValidationError(f"Missing columns in treatment data: {missing_treatment}")
        
        # Check sample sizes
        if len(control_data) < self.config.min_sample_size:
            raise ValidationError(f"Control group too small: {len(control_data)} < {self.config.min_sample_size}")
        
        if len(treatment_data) < self.config.min_sample_size:
            raise ValidationError(f"Treatment group too small: {len(treatment_data)} < {self.config.min_sample_size}")
        
        self.logger.info("✅ Data validation completed successfully")
    
    def _compare_performance(
        self, 
        control_data: pd.DataFrame, 
        treatment_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compare performance between control and treatment groups."""
        self.logger.info("📊 Comparing performance...")
        
        comparison = {
            'control_group_stats': {},
            'treatment_group_stats': {},
            'performance_differences': {},
            'relative_improvements': {}
        }
        
        # Calculate statistics for each group
        for col in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']:
            if col in control_data.columns and col in treatment_data.columns:
                control_values = control_data[col].dropna()
                treatment_values = treatment_data[col].dropna()
                
                comparison['control_group_stats'][col] = {
                    'mean': float(control_values.mean()),
                    'std': float(control_values.std()),
                    'median': float(control_values.median()),
                    'min': float(control_values.min()),
                    'max': float(control_values.max())
                }
                
                comparison['treatment_group_stats'][col] = {
                    'mean': float(treatment_values.mean()),
                    'std': float(treatment_values.std()),
                    'median': float(treatment_values.median()),
                    'min': float(treatment_values.min()),
                    'max': float(treatment_values.max())
                }
                
                # Calculate differences
                mean_diff = treatment_values.mean() - control_values.mean()
                comparison['performance_differences'][col] = {
                    'mean_difference': float(mean_diff),
                    'relative_difference': float(mean_diff / abs(control_values.mean())) if control_values.mean() != 0 else 0.0
                }
                
                # Calculate relative improvements
                if col == 'max_drawdown':  # Lower is better for drawdown
                    improvement = control_values.mean() - treatment_values.mean()
                else:  # Higher is better for other metrics
                    improvement = treatment_values.mean() - control_values.mean()
                
                comparison['relative_improvements'][col] = {
                    'absolute_improvement': float(improvement),
                    'relative_improvement': float(improvement / abs(control_values.mean())) if control_values.mean() != 0 else 0.0,
                    'improvement_direction': 'positive' if improvement > 0 else 'negative'
                }
        
        self.logger.info("✅ Performance comparison completed")
        return comparison
    
    def _analyze_effect_size(self, ab_test_results: ABTestResults) -> Dict[str, Any]:
        """Analyze effect size and statistical power."""
        self.logger.info("📈 Analyzing effect size...")
        
        effect_size_analysis = {
            'overall_effect_size': ab_test_results.effect_size,
            'statistical_power': ab_test_results.statistical_power,
            'minimum_detectable_effect': ab_test_results.minimum_detectable_effect,
            'effect_size_interpretation': self._interpret_effect_size(ab_test_results.effect_size),
            'power_interpretation': self._interpret_power(ab_test_results.statistical_power)
        }
        
        # Calculate effect size for individual metrics
        if hasattr(ab_test_results, 'control_group_stats') and hasattr(ab_test_results, 'treatment_group_stats'):
            metric_effect_sizes = {}
            
            for metric in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']:
                if metric in ab_test_results.control_group_stats and metric in ab_test_results.treatment_group_stats:
                    control_mean = ab_test_results.control_group_stats[metric]['mean']
                    treatment_mean = ab_test_results.treatment_group_stats[metric]['mean']
                    control_std = ab_test_results.control_group_stats[metric]['std']
                    treatment_std = ab_test_results.treatment_group_stats[metric]['std']
                    
                    # Calculate Cohen's d
                    pooled_std = np.sqrt((control_std**2 + treatment_std**2) / 2)
                    cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
                    
                    metric_effect_sizes[metric] = {
                        'cohens_d': float(cohens_d),
                        'interpretation': self._interpret_effect_size(abs(cohens_d))
                    }
            
            effect_size_analysis['metric_effect_sizes'] = metric_effect_sizes
        
        self.logger.info("✅ Effect size analysis completed")
        return effect_size_analysis
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude."""
        abs_effect_size = abs(effect_size)
        
        if abs_effect_size < 0.2:
            return "negligible"
        elif abs_effect_size < 0.5:
            return "small"
        elif abs_effect_size < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_power(self, power: float) -> str:
        """Interpret statistical power."""
        if power < 0.5:
            return "low"
        elif power < 0.8:
            return "moderate"
        else:
            return "high"
    
    def _calculate_confidence_intervals(self, ab_test_results: ABTestResults) -> Dict[str, Any]:
        """Calculate confidence intervals for performance differences."""
        self.logger.info("📊 Calculating confidence intervals...")
        
        confidence_intervals = {}
        
        # Extract confidence intervals from test results
        for test_result in ab_test_results.test_results:
            if 'confidence_interval' in test_result:
                metric = test_result['metric']
                ci = test_result['confidence_interval']
                
                confidence_intervals[metric] = {
                    'lower_bound': float(ci[0]),
                    'upper_bound': float(ci[1]),
                    'width': float(ci[1] - ci[0]),
                    'contains_zero': ci[0] <= 0 <= ci[1]
                }
        
        self.logger.info("✅ Confidence intervals calculated")
        return confidence_intervals
    
    def _generate_recommendations(self, ab_test_results: ABTestResults) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on A/B test results."""
        self.logger.info("💡 Generating recommendations...")
        
        recommendations = []
        
        # Overall conclusion recommendations
        if "Significant difference detected" in ab_test_results.overall_conclusion:
            if ab_test_results.effect_size >= 0.2:  # Large effect
                recommendations.append({
                    'category': 'DEPLOYMENT',
                    'priority': 'HIGH',
                    'title': 'Deploy Treatment Strategy',
                    'description': f'Treatment strategy shows significant improvement with large effect size ({ab_test_results.effect_size:.2f})',
                    'action': 'Deploy the treatment strategy in production',
                    'confidence': 'HIGH'
                })
            elif ab_test_results.effect_size >= 0.1:  # Medium effect
                recommendations.append({
                    'category': 'DEPLOYMENT',
                    'priority': 'MEDIUM',
                    'title': 'Consider Deploying Treatment Strategy',
                    'description': f'Treatment strategy shows significant improvement with medium effect size ({ab_test_results.effect_size:.2f})',
                    'action': 'Consider deploying the treatment strategy with careful monitoring',
                    'confidence': 'MEDIUM'
                })
        elif "No significant difference detected" in ab_test_results.overall_conclusion:
            recommendations.append({
                'category': 'OPTIMIZATION',
                'priority': 'MEDIUM',
                'title': 'No Significant Difference Found',
                'description': 'No significant difference between control and treatment strategies',
                'action': 'Continue with control strategy or investigate treatment strategy further',
                'confidence': 'MEDIUM'
            })
        
        # Statistical power recommendations
        if ab_test_results.statistical_power < 0.8:
            recommendations.append({
                'category': 'METHODOLOGY',
                'priority': 'MEDIUM',
                'title': 'Low Statistical Power',
                'description': f'Statistical power is low ({ab_test_results.statistical_power:.2f}), which may lead to false negatives',
                'action': 'Increase sample size or effect size to improve statistical power',
                'confidence': 'HIGH'
            })
        
        # Sample size recommendations
        if ab_test_results.total_sample_size < 1000:
            recommendations.append({
                'category': 'METHODOLOGY',
                'priority': 'LOW',
                'title': 'Consider Larger Sample Size',
                'description': f'Current sample size ({ab_test_results.total_sample_size}) may be limiting statistical power',
                'action': 'Consider increasing sample size for more reliable results',
                'confidence': 'MEDIUM'
            })
        
        # Effect size recommendations
        if ab_test_results.effect_size < 0.1:
            recommendations.append({
                'category': 'STRATEGY',
                'priority': 'LOW',
                'title': 'Small Effect Size',
                'description': f'Effect size is small ({ab_test_results.effect_size:.2f}), indicating limited practical significance',
                'action': 'Consider optimizing the treatment strategy to achieve larger effect sizes',
                'confidence': 'MEDIUM'
            })
        
        self.logger.info(f"✅ Generated {len(recommendations)} recommendations")
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
    
    async def _save_results(self, results: ABTestingResults) -> None:
        """Save results to disk."""
        self.logger.info("💾 Saving results...")
        
        # Create output directory
        output_dir = self.data_dir / "backtesting_results" / "ab_testing"
        ensure_directory(output_dir)
        
        # Save main results
        results_file = output_dir / f"{self.config.test_name}_ab_test_results.json"
        await safe_json_dump(results_file, results.__dict__, indent=2)
        
        # Save control group data
        if not results.control_group_data.empty:
            control_file = output_dir / f"{self.config.test_name}_control_group_data.parquet"
            await self.parquet_utils.save_dataframe(results.control_group_data, control_file)
        
        # Save treatment group data
        if not results.treatment_group_data.empty:
            treatment_file = output_dir / f"{self.config.test_name}_treatment_group_data.parquet"
            await self.parquet_utils.save_dataframe(results.treatment_group_data, treatment_file)
        
        self.logger.info(f"✅ Results saved to {output_dir}")


# Convenience function for easy integration
async def execute_ab_testing(
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE", 
    timeframe: str = "1h",
    data_dir: str = "data/training",
    test_name: str = "strategy_comparison",
    control_strategy: str = "baseline",
    treatment_strategy: str = "optimized",
    **kwargs
) -> ABTestingResults:
    """
    Convenience function to execute A/B testing.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Data timeframe
        data_dir: Data directory
        test_name: Name of the A/B test
        control_strategy: Control strategy name
        treatment_strategy: Treatment strategy name
        **kwargs: Additional configuration parameters
        
    Returns:
        A/B testing results
    """
    config = ABTestingConfig(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        test_name=test_name,
        control_strategy=control_strategy,
        treatment_strategy=treatment_strategy,
        **kwargs
    )
    
    step = ABTestingStep(config)
    return await step.execute()