"""
Monte Carlo Simulation Step

This module provides comprehensive Monte Carlo simulation functionality for
backtesting strategies with statistical analysis and risk assessment.

Key Features:
- Monte Carlo simulation with configurable parameters
- Statistical analysis of simulation results
- Risk metrics calculation (VaR, CVaR, etc.)
- Performance distribution analysis
- Confidence interval estimation
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
from src.utils.common_ml.backtesting.monte_carlo_engine import (
    MonteCarloEngine, MonteCarloConfig, MonteCarloResults, SimulationType, SimulationParameters
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


class MonteCarloSimulationType(Enum):
    """Types of Monte Carlo simulations."""
    PRICE_SIMULATION = "price_simulation"
    PORTFOLIO_SIMULATION = "portfolio_simulation"
    STRATEGY_SIMULATION = "strategy_simulation"
    RISK_SIMULATION = "risk_simulation"
    REGIME_SIMULATION = "regime_simulation"


@dataclass
class MonteCarloSimulationConfig:
    """Configuration for Monte Carlo simulation step."""
    # Basic configuration
    symbol: str
    exchange: str
    timeframe: str
    data_dir: str
    
    # Simulation parameters
    simulation_type: MonteCarloSimulationType = MonteCarloSimulationType.STRATEGY_SIMULATION
    n_simulations: int = 10000
    n_periods: int = 252  # Trading days in a year
    initial_capital: float = 100000.0
    
    # Price simulation parameters
    drift: float = 0.05  # Annual drift
    volatility: float = 0.2  # Annual volatility
    jump_probability: float = 0.05  # Probability of jumps
    jump_size: float = 0.1  # Size of jumps
    
    # Strategy parameters
    strategy_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Risk parameters
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    var_horizon: int = 1  # Days for VaR calculation
    
    # Performance settings
    enable_detailed_logging: bool = True
    enable_performance_monitoring: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    
    # Output settings
    save_detailed_results: bool = True
    generate_plots: bool = True
    output_format: str = "parquet"
    
    # Random seed
    random_seed: int = 42


@dataclass
class MonteCarloSimulationResults:
    """Results from Monte Carlo simulation step."""
    # Basic info
    symbol: str
    exchange: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    total_duration: float
    
    # Simulation results
    simulation_results: Dict[str, Any] = field(default_factory=dict)
    
    # Statistical analysis
    statistical_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Risk metrics
    risk_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Performance distribution
    performance_distribution: Dict[str, Any] = field(default_factory=dict)
    
    # Confidence intervals
    confidence_intervals: Dict[str, Any] = field(default_factory=dict)
    
    # Detailed data
    simulated_paths: np.ndarray = field(default_factory=lambda: np.array([]))
    final_values: np.ndarray = field(default_factory=lambda: np.array([]))
    returns: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Metadata
    config: MonteCarloSimulationConfig = field(default_factory=MonteCarloSimulationConfig)
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    system_metrics: Dict[str, Any] = field(default_factory=dict)


class MonteCarloSimulationStep:
    """Monte Carlo simulation step."""
    
    def __init__(self, config: MonteCarloSimulationConfig):
        """Initialize the Monte Carlo simulation step."""
        self.config = config
        self.logger = logger.getChild('MonteCarloSimulationStep')
        
        # Initialize utilities
        self.parquet_utils = get_parquet_utils()
        self.financial_logger = EnhancedFinancialMetricsLogger()
        self.performance_monitor = PerformanceMonitor()
        self.system_monitor = SystemMonitor()
        
        # Initialize data directory
        self.data_dir = Path(config.data_dir)
        ensure_directory(self.data_dir)
        
        # Initialize Monte Carlo engine
        self.monte_carlo_config = MonteCarloConfig(
            symbol=config.symbol,
            exchange=config.exchange,
            timeframe=config.timeframe,
            data_dir=config.data_dir,
            simulation_type=SimulationType.STRATEGY_SIMULATION,
            simulation_params=SimulationParameters(
                n_simulations=config.n_simulations,
                n_periods=config.n_periods,
                initial_value=config.initial_capital,
                drift=config.drift,
                volatility=config.volatility,
                jump_probability=config.jump_probability,
                jump_size=config.jump_size,
                random_seed=config.random_seed
            ),
            enable_gpu_acceleration=True,
            enable_memory_optimization=config.enable_memory_optimization,
            enable_parallel_processing=config.enable_parallel_processing
        )
        
        self.monte_carlo_engine = MonteCarloEngine(self.monte_carlo_config)
        
        self.logger.info(f"🚀 MonteCarloSimulationStep initialized for {config.symbol}")
        self.logger.info(f"🎲 Simulation type: {config.simulation_type.value}")
        self.logger.info(f"📊 Number of simulations: {config.n_simulations:,}")
        self.logger.info(f"📅 Number of periods: {config.n_periods}")
        self.logger.info(f"💰 Initial capital: ${config.initial_capital:,.2f}")
        self.logger.info(f"📁 Data directory: {config.data_dir}")
    
    @traced(span_name='monte_carlo_simulation')
    @log_execution_time
    @monitor_step_execution
    async def execute(
        self, 
        data: Optional[pd.DataFrame] = None,
        strategy_func: Optional[Callable] = None,
        **kwargs
    ) -> MonteCarloSimulationResults:
        """Execute Monte Carlo simulation."""
        
        self.logger.info("🚀 Starting Monte Carlo simulation...")
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
            
            # Execute Monte Carlo simulation
            monte_carlo_results = await self.monte_carlo_engine.simulate(
                historical_data=data,
                custom_params=self._create_simulation_parameters(data)
            )
            
            # Analyze simulation results
            statistical_analysis = self._perform_statistical_analysis(monte_carlo_results)
            risk_metrics = self._calculate_risk_metrics(monte_carlo_results)
            performance_distribution = self._analyze_performance_distribution(monte_carlo_results)
            confidence_intervals = self._calculate_confidence_intervals(monte_carlo_results)
            
            # Create results
            results = MonteCarloSimulationResults(
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                timeframe=self.config.timeframe,
                start_time=datetime.now(),
                end_time=datetime.now(),
                total_duration=time.time() - start_time,
                simulation_results=monte_carlo_results.__dict__,
                statistical_analysis=statistical_analysis,
                risk_metrics=risk_metrics,
                performance_distribution=performance_distribution,
                confidence_intervals=confidence_intervals,
                simulated_paths=monte_carlo_results.simulated_paths,
                final_values=monte_carlo_results.final_values,
                returns=monte_carlo_results.returns,
                config=self.config,
                execution_time=time.time() - start_time,
                memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
                system_metrics=self._get_system_metrics()
            )
            
            # Save results
            if self.config.save_detailed_results:
                await self._save_results(results)
            
            self.logger.info("✅ Monte Carlo simulation completed successfully")
            self.logger.info(f"⏱️ Execution time: {results.execution_time:.2f}s")
            self.logger.info(f"📊 Simulations completed: {monte_carlo_results.n_simulations:,}")
            self.logger.info(f"📈 Mean final value: ${monte_carlo_results.mean_final_value:,.2f}")
            self.logger.info(f"⚠️ VaR 95%: {monte_carlo_results.var_95:.2%}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in Monte Carlo simulation: {e}")
            self.logger.exception("Full traceback:")
            raise
        finally:
            # Stop performance monitoring
            if self.config.enable_performance_monitoring:
                self.performance_monitor.stop_monitoring()
    
    async def _load_data(self) -> pd.DataFrame:
        """Load market data for simulation."""
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
    
    def _create_simulation_parameters(self, data: pd.DataFrame) -> SimulationParameters:
        """Create simulation parameters from historical data."""
        self.logger.info("🔧 Creating simulation parameters from historical data...")
        
        # Calculate parameters from historical data
        returns = data['close'].pct_change().dropna()
        
        # Calculate drift and volatility from historical data
        if len(returns) > 0:
            drift = returns.mean() * 252  # Annualized
            volatility = returns.std() * np.sqrt(252)  # Annualized
        else:
            drift = self.config.drift
            volatility = self.config.volatility
        
        # Create simulation parameters
        sim_params = SimulationParameters(
            n_simulations=self.config.n_simulations,
            n_periods=self.config.n_periods,
            initial_value=self.config.initial_capital,
            drift=drift,
            volatility=volatility,
            jump_probability=self.config.jump_probability,
            jump_size=self.config.jump_size,
            random_seed=self.config.random_seed
        )
        
        self.logger.info(f"📊 Calculated drift: {drift:.2%}")
        self.logger.info(f"📊 Calculated volatility: {volatility:.2%}")
        
        return sim_params
    
    def _perform_statistical_analysis(self, monte_carlo_results: MonteCarloResults) -> Dict[str, Any]:
        """Perform statistical analysis on simulation results."""
        self.logger.info("📈 Performing statistical analysis...")
        
        final_values = monte_carlo_results.final_values
        returns = monte_carlo_results.returns
        
        if len(final_values) == 0 or len(returns) == 0:
            return {}
        
        # Basic statistics
        analysis = {
            'final_values': {
                'mean': float(np.mean(final_values)),
                'median': float(np.median(final_values)),
                'std': float(np.std(final_values)),
                'min': float(np.min(final_values)),
                'max': float(np.max(final_values)),
                'skewness': float(stats.skew(final_values)),
                'kurtosis': float(stats.kurtosis(final_values))
            },
            'returns': {
                'mean': float(np.mean(returns)),
                'median': float(np.median(returns)),
                'std': float(np.std(returns)),
                'min': float(np.min(returns)),
                'max': float(np.max(returns)),
                'skewness': float(stats.skew(returns)),
                'kurtosis': float(stats.kurtosis(returns))
            }
        }
        
        # Normality tests
        if len(returns) >= 3:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(returns)
                analysis['normality_tests'] = {
                    'shapiro_wilk': {
                        'statistic': float(shapiro_stat),
                        'p_value': float(shapiro_p),
                        'is_normal': shapiro_p > 0.05
                    }
                }
            except Exception as e:
                self.logger.warning(f"⚠️ Normality test failed: {e}")
        
        # Percentiles
        analysis['percentiles'] = {
            'final_values': {
                'p1': float(np.percentile(final_values, 1)),
                'p5': float(np.percentile(final_values, 5)),
                'p10': float(np.percentile(final_values, 10)),
                'p25': float(np.percentile(final_values, 25)),
                'p50': float(np.percentile(final_values, 50)),
                'p75': float(np.percentile(final_values, 75)),
                'p90': float(np.percentile(final_values, 90)),
                'p95': float(np.percentile(final_values, 95)),
                'p99': float(np.percentile(final_values, 99))
            },
            'returns': {
                'p1': float(np.percentile(returns, 1)),
                'p5': float(np.percentile(returns, 5)),
                'p10': float(np.percentile(returns, 10)),
                'p25': float(np.percentile(returns, 25)),
                'p50': float(np.percentile(returns, 50)),
                'p75': float(np.percentile(returns, 75)),
                'p90': float(np.percentile(returns, 90)),
                'p95': float(np.percentile(returns, 95)),
                'p99': float(np.percentile(returns, 99))
            }
        }
        
        self.logger.info("✅ Statistical analysis completed")
        return analysis
    
    def _calculate_risk_metrics(self, monte_carlo_results: MonteCarloResults) -> Dict[str, Any]:
        """Calculate risk metrics from simulation results."""
        self.logger.info("⚠️ Calculating risk metrics...")
        
        final_values = monte_carlo_results.final_values
        returns = monte_carlo_results.returns
        
        if len(final_values) == 0 or len(returns) == 0:
            return {}
        
        risk_metrics = {
            'value_at_risk': {
                'var_95': float(monte_carlo_results.var_95),
                'var_99': float(monte_carlo_results.var_99)
            },
            'conditional_value_at_risk': {
                'cvar_95': float(monte_carlo_results.cvar_95),
                'cvar_99': float(monte_carlo_results.cvar_99)
            },
            'expected_shortfall': float(monte_carlo_results.expected_shortfall)
        }
        
        # Calculate additional risk metrics
        risk_metrics['downside_deviation'] = float(np.std(returns[returns < 0])) if len(returns[returns < 0]) > 0 else 0.0
        risk_metrics['upside_deviation'] = float(np.std(returns[returns > 0])) if len(returns[returns > 0]) > 0 else 0.0
        
        # Calculate maximum drawdown for each simulation path
        if monte_carlo_results.simulated_paths.size > 0:
            max_drawdowns = []
            for path in monte_carlo_results.simulated_paths:
                peak = np.maximum.accumulate(path)
                drawdown = (path - peak) / peak
                max_drawdowns.append(np.min(drawdown))
            
            risk_metrics['maximum_drawdown'] = {
                'mean': float(np.mean(max_drawdowns)),
                'std': float(np.std(max_drawdowns)),
                'min': float(np.min(max_drawdowns)),
                'max': float(np.max(max_drawdowns)),
                'percentiles': {
                    'p5': float(np.percentile(max_drawdowns, 5)),
                    'p25': float(np.percentile(max_drawdowns, 25)),
                    'p50': float(np.percentile(max_drawdowns, 50)),
                    'p75': float(np.percentile(max_drawdowns, 75)),
                    'p95': float(np.percentile(max_drawdowns, 95))
                }
            }
        
        # Calculate probability of loss
        risk_metrics['probability_of_loss'] = float(len(returns[returns < 0]) / len(returns))
        risk_metrics['probability_of_positive_return'] = float(len(returns[returns > 0]) / len(returns))
        
        # Calculate tail risk metrics
        if len(returns) > 0:
            tail_returns = returns[returns <= np.percentile(returns, 5)]
            risk_metrics['tail_risk'] = {
                'mean_tail_return': float(np.mean(tail_returns)) if len(tail_returns) > 0 else 0.0,
                'tail_volatility': float(np.std(tail_returns)) if len(tail_returns) > 1 else 0.0
            }
        
        self.logger.info("✅ Risk metrics calculated")
        return risk_metrics
    
    def _analyze_performance_distribution(self, monte_carlo_results: MonteCarloResults) -> Dict[str, Any]:
        """Analyze performance distribution."""
        self.logger.info("📊 Analyzing performance distribution...")
        
        final_values = monte_carlo_results.final_values
        returns = monte_carlo_results.returns
        
        if len(final_values) == 0 or len(returns) == 0:
            return {}
        
        # Performance categories
        performance_categories = {
            'excellent': len(returns[returns > 0.2]) / len(returns),  # >20% return
            'good': len(returns[(returns > 0.1) & (returns <= 0.2)]) / len(returns),  # 10-20% return
            'moderate': len(returns[(returns > 0.0) & (returns <= 0.1)]) / len(returns),  # 0-10% return
            'poor': len(returns[(returns > -0.1) & (returns <= 0.0)]) / len(returns),  # -10-0% return
            'very_poor': len(returns[returns <= -0.1]) / len(returns)  # <-10% return
        }
        
        # Risk categories based on drawdown
        risk_categories = {
            'low_risk': 0.0,  # Would need to calculate from individual paths
            'medium_risk': 0.0,
            'high_risk': 0.0
        }
        
        # Calculate risk categories if we have individual paths
        if monte_carlo_results.simulated_paths.size > 0:
            max_drawdowns = []
            for path in monte_carlo_results.simulated_paths:
                peak = np.maximum.accumulate(path)
                drawdown = (path - peak) / peak
                max_drawdowns.append(np.min(drawdown))
            
            max_drawdowns = np.array(max_drawdowns)
            risk_categories = {
                'low_risk': len(max_drawdowns[max_drawdowns > -0.05]) / len(max_drawdowns),  # <5% drawdown
                'medium_risk': len(max_drawdowns[(max_drawdowns <= -0.05) & (max_drawdowns > -0.15)]) / len(max_drawdowns),  # 5-15% drawdown
                'high_risk': len(max_drawdowns[max_drawdowns <= -0.15]) / len(max_drawdowns)  # >15% drawdown
            }
        
        distribution_analysis = {
            'performance_categories': performance_categories,
            'risk_categories': risk_categories,
            'distribution_metrics': {
                'coefficient_of_variation': float(np.std(returns) / abs(np.mean(returns))) if np.mean(returns) != 0 else float('inf'),
                'sharpe_ratio': float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0.0,
                'sortino_ratio': float(np.mean(returns) / np.std(returns[returns < 0])) if len(returns[returns < 0]) > 0 and np.std(returns[returns < 0]) > 0 else 0.0
            }
        }
        
        self.logger.info("✅ Performance distribution analysis completed")
        return distribution_analysis
    
    def _calculate_confidence_intervals(self, monte_carlo_results: MonteCarloResults) -> Dict[str, Any]:
        """Calculate confidence intervals for various metrics."""
        self.logger.info("📊 Calculating confidence intervals...")
        
        final_values = monte_carlo_results.final_values
        returns = monte_carlo_results.returns
        
        if len(final_values) == 0 or len(returns) == 0:
            return {}
        
        confidence_intervals = {}
        
        # Confidence intervals for final values
        for confidence_level in self.config.confidence_levels:
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            confidence_intervals[f'final_values_{int(confidence_level*100)}'] = {
                'lower': float(np.percentile(final_values, lower_percentile)),
                'upper': float(np.percentile(final_values, upper_percentile)),
                'range': float(np.percentile(final_values, upper_percentile) - np.percentile(final_values, lower_percentile))
            }
        
        # Confidence intervals for returns
        for confidence_level in self.config.confidence_levels:
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            confidence_intervals[f'returns_{int(confidence_level*100)}'] = {
                'lower': float(np.percentile(returns, lower_percentile)),
                'upper': float(np.percentile(returns, upper_percentile)),
                'range': float(np.percentile(returns, upper_percentile) - np.percentile(returns, lower_percentile))
            }
        
        # Parametric confidence intervals for mean return
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            n = len(returns)
            
            for confidence_level in self.config.confidence_levels:
                alpha = 1 - confidence_level
                t_critical = stats.t.ppf(1 - alpha/2, n-1)
                margin_of_error = t_critical * (std_return / np.sqrt(n))
                
                confidence_intervals[f'mean_return_{int(confidence_level*100)}_parametric'] = {
                    'lower': float(mean_return - margin_of_error),
                    'upper': float(mean_return + margin_of_error),
                    'margin_of_error': float(margin_of_error)
                }
        
        self.logger.info("✅ Confidence intervals calculated")
        return confidence_intervals
    
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
    
    async def _save_results(self, results: MonteCarloSimulationResults) -> None:
        """Save results to disk."""
        self.logger.info("💾 Saving results...")
        
        # Create output directory
        output_dir = self.data_dir / "backtesting_results" / "monte_carlo"
        ensure_directory(output_dir)
        
        # Save main results
        results_file = output_dir / f"{self.config.symbol}_{self.config.exchange}_monte_carlo_results.json"
        await safe_json_dump(results_file, results.__dict__, indent=2)
        
        # Save simulated paths
        if results.simulated_paths.size > 0:
            paths_file = output_dir / f"{self.config.symbol}_{self.config.exchange}_simulated_paths.parquet"
            paths_df = pd.DataFrame(results.simulated_paths)
            await self.parquet_utils.save_dataframe(paths_df, paths_file)
        
        # Save final values
        if results.final_values.size > 0:
            values_file = output_dir / f"{self.config.symbol}_{self.config.exchange}_final_values.parquet"
            values_df = pd.DataFrame({'final_value': results.final_values})
            await self.parquet_utils.save_dataframe(values_df, values_file)
        
        # Save returns
        if results.returns.size > 0:
            returns_file = output_dir / f"{self.config.symbol}_{self.config.exchange}_returns.parquet"
            returns_df = pd.DataFrame({'return': results.returns})
            await self.parquet_utils.save_dataframe(returns_df, returns_file)
        
        self.logger.info(f"✅ Results saved to {output_dir}")


# Convenience function for easy integration
async def execute_monte_carlo_simulation(
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE", 
    timeframe: str = "1h",
    data_dir: str = "data/training",
    n_simulations: int = 10000,
    strategy_func: Optional[Callable] = None,
    **kwargs
) -> MonteCarloSimulationResults:
    """
    Convenience function to execute Monte Carlo simulation.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Data timeframe
        data_dir: Data directory
        n_simulations: Number of simulations to run
        strategy_func: Strategy function to simulate
        **kwargs: Additional configuration parameters
        
    Returns:
        Monte Carlo simulation results
    """
    config = MonteCarloSimulationConfig(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        n_simulations=n_simulations,
        **kwargs
    )
    
    step = MonteCarloSimulationStep(config)
    return await step.execute(strategy_func=strategy_func)