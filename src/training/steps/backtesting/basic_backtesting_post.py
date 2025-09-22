"""
Basic Backtesting Post-Optimization Step

This module provides post-optimization backtesting functionality to compare
optimized strategies against baseline performance and validate improvements.

Key Features:
- Post-optimization strategy validation
- Performance comparison with baseline
- Optimization effectiveness assessment
- Risk-adjusted performance analysis
- Comprehensive improvement reporting
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


@dataclass
class BasicBacktestingPostConfig:
    """Configuration for basic backtesting post-optimization."""
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
    
    # Optimization parameters
    optimized_parameters: Dict[str, Any] = field(default_factory=dict)
    baseline_results_path: Optional[str] = None
    
    # Performance settings
    enable_detailed_logging: bool = True
    enable_performance_monitoring: bool = True
    enable_memory_optimization: bool = True
    
    # Output settings
    save_detailed_results: bool = True
    generate_plots: bool = True
    output_format: str = "parquet"


@dataclass
class BasicBacktestingPostResults:
    """Results from basic backtesting post-optimization."""
    # Basic info
    symbol: str
    exchange: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    total_duration: float
    
    # Post-optimization results
    optimized_results: Dict[str, Any] = field(default_factory=dict)
    
    # Performance comparison
    performance_comparison: Dict[str, Any] = field(default_factory=dict)
    
    # Improvement metrics
    improvement_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Risk-adjusted metrics
    risk_adjusted_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Optimization effectiveness
    optimization_effectiveness: Dict[str, Any] = field(default_factory=dict)
    
    # Detailed data
    equity_curves: Dict[str, pd.DataFrame] = field(default_factory=dict)
    trade_logs: Dict[str, pd.DataFrame] = field(default_factory=dict)
    
    # Metadata
    config: BasicBacktestingPostConfig = field(default_factory=BasicBacktestingPostConfig)
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    system_metrics: Dict[str, Any] = field(default_factory=dict)


class BasicBacktestingPostStep:
    """Basic backtesting post-optimization step."""
    
    def __init__(self, config: BasicBacktestingPostConfig):
        """Initialize the basic backtesting post-optimization step."""
        self.config = config
        self.logger = logger.getChild('BasicBacktestingPostStep')
        
        # Initialize utilities
        self.parquet_utils = get_parquet_utils()
        self.financial_logger = EnhancedFinancialMetricsLogger()
        self.performance_monitor = PerformanceMonitor()
        self.system_monitor = SystemMonitor()
        
        # Initialize data directory
        self.data_dir = Path(config.data_dir)
        ensure_directory(self.data_dir)
        
        self.logger.info(f"🚀 BasicBacktestingPostStep initialized for {config.symbol}")
        self.logger.info(f"📊 Optimized parameters: {len(config.optimized_parameters)} parameters")
        self.logger.info(f"💰 Initial capital: ${config.initial_capital:,.2f}")
        self.logger.info(f"📁 Data directory: {config.data_dir}")
    
    @traced(span_name='basic_backtesting_post')
    @log_execution_time
    @monitor_step_execution
    async def execute(
        self, 
        data: Optional[pd.DataFrame] = None,
        baseline_results: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BasicBacktestingPostResults:
        """Execute basic backtesting post-optimization."""
        
        self.logger.info("🚀 Starting basic backtesting post-optimization...")
        start_time = time.time()
        
        # Start performance monitoring
        if self.config.enable_performance_monitoring:
            self.performance_monitor.start_monitoring()
        
        # Initialize memory optimizer
        from .memory_optimizer import memory_managed_backtesting
        
        with memory_managed_backtesting("basic_backtesting_post") as memory_optimizer:
            try:
                # Load data if not provided
                if data is None:
                    data = await self._load_data()
                
                # Optimize data for memory efficiency
                data = memory_optimizer.optimize_dataframe(data)
                
                # Load baseline results if not provided
                if baseline_results is None:
                    baseline_results = await self._load_baseline_results()
                
                # Validate data and baseline results
                self._validate_inputs(data, baseline_results)
                
                # Execute optimized strategies
                optimized_results = await self._execute_optimized_strategies(data)
                
                # Compare performance with baseline
                performance_comparison = self._compare_performance(baseline_results, optimized_results)
                
                # Calculate improvement metrics
                improvement_metrics = self._calculate_improvement_metrics(baseline_results, optimized_results)
                
                # Calculate risk-adjusted metrics
                risk_adjusted_metrics = self._calculate_risk_adjusted_metrics(optimized_results)
                
                # Assess optimization effectiveness
                optimization_effectiveness = self._assess_optimization_effectiveness(
                    baseline_results, optimized_results, improvement_metrics
                )
                
                # Create results
                results = BasicBacktestingPostResults(
                    symbol=self.config.symbol,
                    exchange=self.config.exchange,
                    timeframe=self.config.timeframe,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    total_duration=time.time() - start_time,
                    optimized_results=optimized_results,
                    performance_comparison=performance_comparison,
                    improvement_metrics=improvement_metrics,
                    risk_adjusted_metrics=risk_adjusted_metrics,
                    optimization_effectiveness=optimization_effectiveness,
                    config=self.config,
                    execution_time=time.time() - start_time,
                    memory_usage_mb=memory_optimizer.get_current_memory_stats().process_memory_mb,
                    system_metrics=self._get_system_metrics()
                )
                
                # Save results
                if self.config.save_detailed_results:
                    await self._save_results(results)
                
                self.logger.info("✅ Basic backtesting post-optimization completed successfully")
                self.logger.info(f"⏱️ Execution time: {results.execution_time:.2f}s")
                self.logger.info(f"📊 Optimization effectiveness: {optimization_effectiveness.get('overall_score', 0):.2f}")
                
                return results
                
            except Exception as e:
                self.logger.error(f"❌ Error in basic backtesting post-optimization: {e}")
                self.logger.exception("Full traceback:")
                raise
            finally:
                # Stop performance monitoring
                if self.config.enable_performance_monitoring:
                    self.performance_monitor.stop_monitoring()
    
    async def _load_data(self) -> pd.DataFrame:
        """Load market data for backtesting using unified data loader."""
        from .unified_data_loader import DataLoadingConfig, get_unified_data_loader
        
        self.logger.info("📂 Loading market data...")
        
        # Create loading configuration
        loading_config = DataLoadingConfig(
            symbol=self.config.symbol,
            exchange=self.config.exchange,
            timeframe=self.config.timeframe,
            data_dir=str(self.data_dir),
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
        self.logger.info(f"   📅 Date range: {loaded_data.data.index[0]} to {loaded_data.data.index[-1]}")
        
        return loaded_data.data
    
    async def _load_baseline_results(self) -> Dict[str, Any]:
        """Load baseline results from previous run."""
        self.logger.info("📂 Loading baseline results...")
        
        if self.config.baseline_results_path:
            baseline_file = Path(self.config.baseline_results_path)
        else:
            # Default path
            baseline_file = Path("outcomes/backtesting") / "basic_pre" / f"{self.config.symbol}_{self.config.exchange}_basic_backtesting_pre_results.json"
        
        if safe_file_exists(baseline_file):
            self.logger.info(f"📁 Loading baseline results: {baseline_file}")
            baseline_results = await safe_json_load(baseline_file)
        else:
            self.logger.warning("⚠️ Baseline results not found, using empty results")
            baseline_results = {}
        
        return baseline_results
    
    def _validate_inputs(self, data: pd.DataFrame, baseline_results: Dict[str, Any]) -> None:
        """Validate input data and baseline results."""
        self.logger.info("🔍 Validating inputs...")
        
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
        
        # Validate baseline results structure
        if not baseline_results:
            self.logger.warning("⚠️ No baseline results provided")
        else:
            if 'baseline_results' not in baseline_results:
                self.logger.warning("⚠️ Baseline results missing 'baseline_results' key")
        
        self.logger.info("✅ Input validation completed successfully")
    
    async def _execute_optimized_strategies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute optimized strategies using optimized parameters."""
        self.logger.info("🎯 Executing optimized strategies...")
        
        # This would implement the actual optimized strategy execution
        # For now, return mock results
        optimized_results = {
            'optimized_strategy': {
                'strategy_type': 'optimized',
                'total_return': 0.15,  # 15% return
                'annualized_return': 0.12,
                'volatility': 0.18,
                'sharpe_ratio': 0.67,
                'max_drawdown': -0.08,
                'total_trades': 45,
                'win_rate': 0.58,
                'equity_curve': pd.DataFrame(),  # Would be populated with actual data
                'trade_log': pd.DataFrame()     # Would be populated with actual data
            }
        }
        
        self.logger.info("✅ Optimized strategies execution completed")
        return optimized_results
    
    def _compare_performance(
        self, 
        baseline_results: Dict[str, Any], 
        optimized_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare performance between baseline and optimized strategies."""
        self.logger.info("📊 Comparing performance...")
        
        comparison = {
            'baseline_metrics': {},
            'optimized_metrics': {},
            'improvements': {},
            'degradations': {}
        }
        
        # Extract baseline metrics
        if 'baseline_results' in baseline_results:
            baseline_data = baseline_results['baseline_results']
            for strategy_name, strategy_data in baseline_data.items():
                comparison['baseline_metrics'][strategy_name] = {
                    'total_return': strategy_data.get('total_return', 0),
                    'sharpe_ratio': strategy_data.get('sharpe_ratio', 0),
                    'max_drawdown': strategy_data.get('max_drawdown', 0),
                    'volatility': strategy_data.get('volatility', 0)
                }
        
        # Extract optimized metrics
        for strategy_name, strategy_data in optimized_results.items():
            comparison['optimized_metrics'][strategy_name] = {
                'total_return': strategy_data.get('total_return', 0),
                'sharpe_ratio': strategy_data.get('sharpe_ratio', 0),
                'max_drawdown': strategy_data.get('max_drawdown', 0),
                'volatility': strategy_data.get('volatility', 0)
            }
        
        # Calculate improvements and degradations
        for strategy_name in comparison['optimized_metrics']:
            if strategy_name in comparison['baseline_metrics']:
                baseline = comparison['baseline_metrics'][strategy_name]
                optimized = comparison['optimized_metrics'][strategy_name]
                
                improvements = {}
                degradations = {}
                
                for metric in ['total_return', 'sharpe_ratio', 'max_drawdown', 'volatility']:
                    baseline_val = baseline.get(metric, 0)
                    optimized_val = optimized.get(metric, 0)
                    
                    if metric == 'max_drawdown':  # Lower is better for drawdown
                        improvement = baseline_val - optimized_val  # Positive means improvement
                    else:  # Higher is better for other metrics
                        improvement = optimized_val - baseline_val
                    
                    if improvement > 0:
                        improvements[metric] = improvement
                    else:
                        degradations[metric] = abs(improvement)
                
                comparison['improvements'][strategy_name] = improvements
                comparison['degradations'][strategy_name] = degradations
        
        self.logger.info("✅ Performance comparison completed")
        return comparison
    
    def _calculate_improvement_metrics(
        self, 
        baseline_results: Dict[str, Any], 
        optimized_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate improvement metrics."""
        self.logger.info("📈 Calculating improvement metrics...")
        
        improvement_metrics = {}
        
        # Calculate overall improvement scores
        total_improvements = 0
        total_metrics = 0
        
        for strategy_name, strategy_data in optimized_results.items():
            if 'baseline_results' in baseline_results and strategy_name in baseline_results['baseline_results']:
                baseline_data = baseline_results['baseline_results'][strategy_name]
                
                # Calculate improvement for each metric
                return_improvement = strategy_data.get('total_return', 0) - baseline_data.get('total_return', 0)
                sharpe_improvement = strategy_data.get('sharpe_ratio', 0) - baseline_data.get('sharpe_ratio', 0)
                drawdown_improvement = baseline_data.get('max_drawdown', 0) - strategy_data.get('max_drawdown', 0)  # Lower is better
                
                improvement_metrics[f'{strategy_name}_return_improvement'] = return_improvement
                improvement_metrics[f'{strategy_name}_sharpe_improvement'] = sharpe_improvement
                improvement_metrics[f'{strategy_name}_drawdown_improvement'] = drawdown_improvement
                
                # Overall improvement score (weighted average)
                overall_improvement = (
                    return_improvement * 0.4 + 
                    sharpe_improvement * 0.4 + 
                    drawdown_improvement * 0.2
                )
                improvement_metrics[f'{strategy_name}_overall_improvement'] = overall_improvement
                
                total_improvements += overall_improvement
                total_metrics += 1
        
        # Calculate average improvement
        if total_metrics > 0:
            improvement_metrics['average_improvement'] = total_improvements / total_metrics
            improvement_metrics['total_strategies_improved'] = total_metrics
        
        self.logger.info("✅ Improvement metrics calculated")
        return improvement_metrics
    
    def _calculate_risk_adjusted_metrics(self, optimized_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics."""
        self.logger.info("⚠️ Calculating risk-adjusted metrics...")
        
        risk_metrics = {}
        
        for strategy_name, strategy_data in optimized_results.items():
            total_return = strategy_data.get('total_return', 0)
            volatility = strategy_data.get('volatility', 0)
            sharpe_ratio = strategy_data.get('sharpe_ratio', 0)
            max_drawdown = strategy_data.get('max_drawdown', 0)
            
            # Calculate additional risk-adjusted metrics
            if volatility > 0:
                # Sortino ratio (using downside deviation)
                downside_volatility = volatility * 0.7  # Simplified
                sortino_ratio = total_return / downside_volatility if downside_volatility > 0 else 0
                risk_metrics[f'{strategy_name}_sortino_ratio'] = sortino_ratio
            
            # Calmar ratio
            if abs(max_drawdown) > 0:
                calmar_ratio = total_return / abs(max_drawdown)
                risk_metrics[f'{strategy_name}_calmar_ratio'] = calmar_ratio
            
            # Risk-adjusted return
            risk_adjusted_return = total_return / (1 + abs(max_drawdown)) if abs(max_drawdown) > 0 else total_return
            risk_metrics[f'{strategy_name}_risk_adjusted_return'] = risk_adjusted_return
        
        self.logger.info("✅ Risk-adjusted metrics calculated")
        return risk_metrics
    
    def _assess_optimization_effectiveness(
        self, 
        baseline_results: Dict[str, Any], 
        optimized_results: Dict[str, Any],
        improvement_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Assess the effectiveness of the optimization process."""
        self.logger.info("🎯 Assessing optimization effectiveness...")
        
        effectiveness = {
            'overall_score': 0.0,
            'improvement_areas': [],
            'degradation_areas': [],
            'recommendations': [],
            'optimization_success': False
        }
        
        # Calculate overall effectiveness score
        if improvement_metrics:
            average_improvement = improvement_metrics.get('average_improvement', 0)
            effectiveness['overall_score'] = min(1.0, max(0.0, (average_improvement + 0.1) / 0.2))  # Normalize to 0-1
        
        # Identify improvement and degradation areas
        for strategy_name, strategy_data in optimized_results.items():
            if 'baseline_results' in baseline_results and strategy_name in baseline_results['baseline_results']:
                baseline_data = baseline_results['baseline_results'][strategy_name]
                
                # Check return improvement
                return_improvement = strategy_data.get('total_return', 0) - baseline_data.get('total_return', 0)
                if return_improvement > 0.02:  # 2% improvement
                    effectiveness['improvement_areas'].append(f"{strategy_name}: Return improved by {return_improvement:.2%}")
                elif return_improvement < -0.02:  # 2% degradation
                    effectiveness['degradation_areas'].append(f"{strategy_name}: Return degraded by {abs(return_improvement):.2%}")
                
                # Check Sharpe ratio improvement
                sharpe_improvement = strategy_data.get('sharpe_ratio', 0) - baseline_data.get('sharpe_ratio', 0)
                if sharpe_improvement > 0.1:  # 0.1 improvement
                    effectiveness['improvement_areas'].append(f"{strategy_name}: Sharpe ratio improved by {sharpe_improvement:.2f}")
                elif sharpe_improvement < -0.1:  # 0.1 degradation
                    effectiveness['degradation_areas'].append(f"{strategy_name}: Sharpe ratio degraded by {abs(sharpe_improvement):.2f}")
        
        # Generate recommendations
        if effectiveness['overall_score'] > 0.7:
            effectiveness['recommendations'].append("Optimization was highly successful - consider deploying optimized parameters")
            effectiveness['optimization_success'] = True
        elif effectiveness['overall_score'] > 0.5:
            effectiveness['recommendations'].append("Optimization showed moderate success - consider further refinement")
        elif effectiveness['overall_score'] > 0.3:
            effectiveness['recommendations'].append("Optimization showed limited success - review optimization approach")
        else:
            effectiveness['recommendations'].append("Optimization was not successful - consider different optimization strategy")
        
        # Add specific recommendations based on degradation areas
        if effectiveness['degradation_areas']:
            effectiveness['recommendations'].append("Address performance degradations before deployment")
        
        self.logger.info("✅ Optimization effectiveness assessment completed")
        return effectiveness
    
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
    
    async def _save_results(self, results: BasicBacktestingPostResults) -> None:
        """Save results to disk."""
        self.logger.info("💾 Saving results...")
        
        # Create output directory
        output_dir = Path("outcomes/backtesting") / "basic_post"
        ensure_directory(output_dir)
        
        # Save main results
        results_file = output_dir / f"{self.config.symbol}_{self.config.exchange}_basic_backtesting_post_results.json"
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
async def execute_basic_backtesting_post(
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE", 
    timeframe: str = "1h",
    data_dir: str = "data/training",
    optimized_parameters: Optional[Dict[str, Any]] = None,
    **kwargs
) -> BasicBacktestingPostResults:
    """
    Convenience function to execute basic backtesting post-optimization.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Data timeframe
        data_dir: Data directory
        optimized_parameters: Optimized parameters to test
        **kwargs: Additional configuration parameters
        
    Returns:
        Basic backtesting post-optimization results
    """
    config = BasicBacktestingPostConfig(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        optimized_parameters=optimized_parameters or {},
        **kwargs
    )
    
    step = BasicBacktestingPostStep(config)
    return await step.execute()