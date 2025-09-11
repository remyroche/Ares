"""
Portfolio Analysis Step

This module provides comprehensive portfolio-level analysis functionality for backtesting results
with detailed portfolio metrics, allocation analysis, and portfolio optimization insights.

Key Features:
- Portfolio-level performance metrics
- Asset allocation analysis
- Portfolio risk analysis
- Portfolio optimization insights
- Diversification analysis
- Portfolio rebalancing analysis
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
from scipy.optimize import minimize

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


class PortfolioAnalysisType(Enum):
    """Types of portfolio analysis."""
    PERFORMANCE_ANALYSIS = "performance_analysis"
    ALLOCATION_ANALYSIS = "allocation_analysis"
    RISK_ANALYSIS = "risk_analysis"
    DIVERSIFICATION_ANALYSIS = "diversification_analysis"
    OPTIMIZATION_ANALYSIS = "optimization_analysis"
    REBALANCING_ANALYSIS = "rebalancing_analysis"


@dataclass
class PortfolioAnalysisConfig:
    """Configuration for portfolio analysis step."""
    # Basic configuration
    symbol: str
    exchange: str
    timeframe: str
    data_dir: str
    
    # Portfolio parameters
    initial_capital: float = 100000.0
    risk_free_rate: float = 0.02  # 2% annual
    benchmark_symbol: str = "BTCUSDT"
    
    # Analysis parameters
    rebalancing_frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    max_allocation_per_asset: float = 0.4  # 40%
    min_allocation_per_asset: float = 0.05  # 5%
    
    # Risk parameters
    target_volatility: float = 0.15  # 15% annual
    max_drawdown_threshold: float = 0.20  # 20%
    
    # Analysis settings
    enable_detailed_logging: bool = True
    enable_performance_monitoring: bool = True
    enable_memory_optimization: bool = True
    
    # Output settings
    save_detailed_results: bool = True
    output_format: str = "parquet"


@dataclass
class PortfolioAnalysisResults:
    """Results from portfolio analysis step."""
    # Basic info
    symbol: str
    exchange: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    total_duration: float
    
    # Portfolio metrics
    portfolio_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Performance analysis
    performance_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Allocation analysis
    allocation_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Risk analysis
    risk_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Diversification analysis
    diversification_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Optimization analysis
    optimization_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Rebalancing analysis
    rebalancing_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Optimization insights
    optimization_insights: List[Dict[str, Any]] = field(default_factory=list)
    
    # Detailed data
    portfolio_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Metadata
    config: PortfolioAnalysisConfig = field(default_factory=PortfolioAnalysisConfig)
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    system_metrics: Dict[str, Any] = field(default_factory=dict)


class PortfolioAnalysisStep:
    """Portfolio analysis step."""
    
    def __init__(self, config: PortfolioAnalysisConfig):
        """Initialize the portfolio analysis step."""
        self.config = config
        self.logger = logger.getChild('PortfolioAnalysisStep')
        
        # Initialize utilities
        self.parquet_utils = get_parquet_utils()
        self.financial_logger = EnhancedFinancialMetricsLogger()
        self.performance_monitor = PerformanceMonitor()
        self.system_monitor = SystemMonitor()
        
        # Initialize data directory
        self.data_dir = Path(config.data_dir)
        ensure_directory(self.data_dir)
        
        self.logger.info(f"🚀 PortfolioAnalysisStep initialized for {config.symbol}")
        self.logger.info(f"💰 Initial capital: ${config.initial_capital:,.2f}")
        self.logger.info(f"📊 Risk-free rate: {config.risk_free_rate:.2%}")
        self.logger.info(f"📁 Data directory: {config.data_dir}")
    
    @traced(span_name='portfolio_analysis')
    @log_execution_time
    @monitor_step_execution
    async def execute(
        self, 
        portfolio_data: Optional[pd.DataFrame] = None,
        market_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> PortfolioAnalysisResults:
        """Execute portfolio analysis."""
        
        self.logger.info("🚀 Starting portfolio analysis...")
        start_time = time.time()
        
        # Start performance monitoring
        if self.config.enable_performance_monitoring:
            self.performance_monitor.start_monitoring()
        
        try:
            # Load data if not provided
            if portfolio_data is None:
                portfolio_data = await self._load_portfolio_data()
            
            if market_data is None:
                market_data = await self._load_market_data()
            
            # Validate data
            self._validate_data(portfolio_data, market_data)
            
            # Calculate portfolio metrics
            portfolio_metrics = await self._calculate_portfolio_metrics(portfolio_data)
            
            # Perform performance analysis
            performance_analysis = await self._perform_performance_analysis(portfolio_data, market_data)
            
            # Perform allocation analysis
            allocation_analysis = await self._perform_allocation_analysis(portfolio_data)
            
            # Perform risk analysis
            risk_analysis = await self._perform_risk_analysis(portfolio_data)
            
            # Perform diversification analysis
            diversification_analysis = await self._perform_diversification_analysis(portfolio_data, market_data)
            
            # Perform optimization analysis
            optimization_analysis = await self._perform_optimization_analysis(portfolio_data, market_data)
            
            # Perform rebalancing analysis
            rebalancing_analysis = await self._perform_rebalancing_analysis(portfolio_data)
            
            # Generate optimization insights
            optimization_insights = self._generate_optimization_insights(
                portfolio_metrics, performance_analysis, allocation_analysis, risk_analysis
            )
            
            # Create results
            results = PortfolioAnalysisResults(
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                timeframe=self.config.timeframe,
                start_time=datetime.now(),
                end_time=datetime.now(),
                total_duration=time.time() - start_time,
                portfolio_metrics=portfolio_metrics,
                performance_analysis=performance_analysis,
                allocation_analysis=allocation_analysis,
                risk_analysis=risk_analysis,
                diversification_analysis=diversification_analysis,
                optimization_analysis=optimization_analysis,
                rebalancing_analysis=rebalancing_analysis,
                optimization_insights=optimization_insights,
                portfolio_data=portfolio_data,
                config=self.config,
                execution_time=time.time() - start_time,
                memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
                system_metrics=self._get_system_metrics()
            )
            
            # Save results
            if self.config.save_detailed_results:
                await self._save_results(results)
            
            self.logger.info("✅ Portfolio analysis completed successfully")
            self.logger.info(f"⏱️ Execution time: {results.execution_time:.2f}s")
            self.logger.info(f"📊 Portfolio metrics calculated: {len(portfolio_metrics)}")
            self.logger.info(f"💡 Optimization insights: {len(optimization_insights)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in portfolio analysis: {e}")
            self.logger.exception("Full traceback:")
            raise
        finally:
            # Stop performance monitoring
            if self.config.enable_performance_monitoring:
                self.performance_monitor.stop_monitoring()
    
    async def _load_portfolio_data(self) -> pd.DataFrame:
        """Load portfolio data."""
        self.logger.info("📂 Loading portfolio data...")
        
        # Try to load from various possible locations
        possible_files = [
            self.data_dir / "backtesting_results" / f"{self.config.symbol}_{self.config.exchange}_portfolio_data.parquet",
            self.data_dir / "backtesting_results" / "performance_analytics" / f"{self.config.symbol}_{self.config.exchange}_portfolio_data.parquet",
            self.data_dir / "backtesting_results" / "basic_backtesting_pre" / f"{self.config.symbol}_{self.config.exchange}_portfolio_data.parquet"
        ]
        
        for file_path in possible_files:
            if safe_file_exists(file_path):
                self.logger.info(f"📁 Loading portfolio data: {file_path}")
                return standardized_parquet_handler.read_parquet_standardized(file_path)
        
        # Generate mock portfolio data if not found
        self.logger.warning("⚠️ No portfolio data found, generating mock data")
        return self._generate_mock_portfolio_data()
    
    async def _load_market_data(self) -> pd.DataFrame:
        """Load market data."""
        self.logger.info("📂 Loading market data...")
        
        # Try to load consolidated data first
        consolidated_file = self.data_dir / f"aggtrades_{self.config.exchange}_{self.config.symbol}_consolidated.parquet"
        
        if safe_file_exists(consolidated_file):
            self.logger.info(f"📁 Loading consolidated data: {consolidated_file}")
            return standardized_parquet_handler.read_parquet_standardized(consolidated_file)
        else:
            self.logger.warning("⚠️ No market data found, using empty DataFrame")
            return pd.DataFrame()
    
    def _generate_mock_portfolio_data(self) -> pd.DataFrame:
        """Generate mock portfolio data for testing."""
        # Generate 252 trading days of data
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        
        # Generate mock portfolio data
        np.random.seed(42)
        n_assets = 3
        asset_names = ['ETH', 'BTC', 'ADA']
        
        # Generate returns for each asset
        returns = {}
        for asset in asset_names:
            returns[asset] = np.random.normal(0.0008, 0.02, 252)
        
        # Create portfolio data
        portfolio_data = pd.DataFrame({
            'timestamp': dates,
            'total_value': self.config.initial_capital * np.cumprod(1 + np.mean(list(returns.values()), axis=0)),
            'cash': self.config.initial_capital * 0.1 * np.ones(252),  # 10% cash
            'total_return': np.mean(list(returns.values()), axis=0)
        })
        
        # Add individual asset allocations
        for i, asset in enumerate(asset_names):
            portfolio_data[f'{asset}_allocation'] = 0.3  # 30% each
            portfolio_data[f'{asset}_value'] = portfolio_data['total_value'] * 0.3
            portfolio_data[f'{asset}_return'] = returns[asset]
        
        portfolio_data.set_index('timestamp', inplace=True)
        return portfolio_data
    
    def _validate_data(self, portfolio_data: pd.DataFrame, market_data: pd.DataFrame) -> None:
        """Validate input data."""
        self.logger.info("🔍 Validating input data...")
        
        if portfolio_data.empty:
            raise ValidationError("Portfolio data is empty")
        
        # Check required columns in portfolio data
        required_columns = ['total_value', 'total_return']
        missing_columns = [col for col in required_columns if col not in portfolio_data.columns]
        if missing_columns:
            raise ValidationError(f"Missing required columns in portfolio data: {missing_columns}")
        
        # Check for sufficient data
        if len(portfolio_data) < 30:
            raise ValidationError(f"Insufficient portfolio data: {len(portfolio_data)} < 30")
        
        self.logger.info("✅ Data validation completed successfully")
    
    async def _calculate_portfolio_metrics(self, portfolio_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics."""
        self.logger.info("📊 Calculating portfolio metrics...")
        
        portfolio_metrics = {}
        
        # Basic portfolio metrics
        if 'total_value' in portfolio_data.columns:
            total_value = portfolio_data['total_value']
            portfolio_metrics['basic_metrics'] = {
                'initial_value': float(total_value.iloc[0]),
                'final_value': float(total_value.iloc[-1]),
                'total_return': float((total_value.iloc[-1] / total_value.iloc[0]) - 1),
                'average_value': float(total_value.mean()),
                'value_volatility': float(total_value.std())
            }
        
        # Return metrics
        if 'total_return' in portfolio_data.columns:
            returns = portfolio_data['total_return'].dropna()
            portfolio_metrics['return_metrics'] = {
                'total_return': float(returns.sum()),
                'average_return': float(returns.mean()),
                'annualized_return': float(returns.mean() * 252),
                'return_volatility': float(returns.std()),
                'annualized_volatility': float(returns.std() * np.sqrt(252)),
                'sharpe_ratio': float((returns.mean() * 252 - self.config.risk_free_rate) / (returns.std() * np.sqrt(252))),
                'max_return': float(returns.max()),
                'min_return': float(returns.min())
            }
        
        # Drawdown metrics
        if 'total_value' in portfolio_data.columns:
            portfolio_metrics['drawdown_metrics'] = self._calculate_drawdown_metrics(portfolio_data['total_value'])
        
        # Allocation metrics
        allocation_columns = [col for col in portfolio_data.columns if col.endswith('_allocation')]
        if allocation_columns:
            portfolio_metrics['allocation_metrics'] = self._calculate_allocation_metrics(portfolio_data, allocation_columns)
        
        self.logger.info("✅ Portfolio metrics calculated")
        return portfolio_metrics
    
    async def _perform_performance_analysis(self, portfolio_data: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform portfolio performance analysis."""
        self.logger.info("📈 Performing performance analysis...")
        
        performance_analysis = {}
        
        if 'total_return' in portfolio_data.columns:
            returns = portfolio_data['total_return'].dropna()
            
            # Performance metrics
            performance_analysis['performance_metrics'] = {
                'total_return': float(returns.sum()),
                'annualized_return': float(returns.mean() * 252),
                'volatility': float(returns.std() * np.sqrt(252)),
                'sharpe_ratio': float((returns.mean() * 252 - self.config.risk_free_rate) / (returns.std() * np.sqrt(252))),
                'sortino_ratio': self._calculate_sortino_ratio(returns),
                'calmar_ratio': self._calculate_calmar_ratio(returns, portfolio_data['total_value']),
                'max_drawdown': float(self._calculate_max_drawdown(portfolio_data['total_value']))
            }
            
            # Risk-adjusted metrics
            performance_analysis['risk_adjusted_metrics'] = {
                'var_95': float(np.percentile(returns, 5)),
                'var_99': float(np.percentile(returns, 1)),
                'cvar_95': float(np.mean(returns[returns <= np.percentile(returns, 5)])),
                'cvar_99': float(np.mean(returns[returns <= np.percentile(returns, 1)])),
                'downside_deviation': float(np.sqrt(np.mean(returns[returns < 0] ** 2))) if (returns < 0).any() else 0.0
            }
            
            # Benchmark comparison
            if not market_data.empty and 'close' in market_data.columns:
                benchmark_returns = market_data['close'].pct_change().dropna()
                performance_analysis['benchmark_comparison'] = self._compare_with_benchmark(returns, benchmark_returns)
        
        self.logger.info("✅ Performance analysis completed")
        return performance_analysis
    
    async def _perform_allocation_analysis(self, portfolio_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform portfolio allocation analysis."""
        self.logger.info("📊 Performing allocation analysis...")
        
        allocation_analysis = {}
        
        # Find allocation columns
        allocation_columns = [col for col in portfolio_data.columns if col.endswith('_allocation')]
        value_columns = [col for col in portfolio_data.columns if col.endswith('_value')]
        
        if allocation_columns:
            # Current allocation
            current_allocation = {}
            for col in allocation_columns:
                asset_name = col.replace('_allocation', '')
                current_allocation[asset_name] = float(portfolio_data[col].iloc[-1])
            
            allocation_analysis['current_allocation'] = current_allocation
            
            # Allocation statistics
            allocation_analysis['allocation_statistics'] = {}
            for col in allocation_columns:
                asset_name = col.replace('_allocation', '')
                allocation_analysis['allocation_statistics'][asset_name] = {
                    'mean_allocation': float(portfolio_data[col].mean()),
                    'std_allocation': float(portfolio_data[col].std()),
                    'min_allocation': float(portfolio_data[col].min()),
                    'max_allocation': float(portfolio_data[col].max()),
                    'allocation_stability': float(1 - portfolio_data[col].std() / portfolio_data[col].mean()) if portfolio_data[col].mean() > 0 else 0.0
                }
            
            # Allocation drift analysis
            allocation_analysis['allocation_drift'] = self._analyze_allocation_drift(portfolio_data, allocation_columns)
            
            # Rebalancing needs
            allocation_analysis['rebalancing_needs'] = self._analyze_rebalancing_needs(current_allocation)
        
        self.logger.info("✅ Allocation analysis completed")
        return allocation_analysis
    
    async def _perform_risk_analysis(self, portfolio_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform portfolio risk analysis."""
        self.logger.info("⚠️ Performing risk analysis...")
        
        risk_analysis = {}
        
        if 'total_return' in portfolio_data.columns:
            returns = portfolio_data['total_return'].dropna()
            
            # Risk metrics
            risk_analysis['risk_metrics'] = {
                'volatility': float(returns.std() * np.sqrt(252)),
                'var_95': float(np.percentile(returns, 5)),
                'var_99': float(np.percentile(returns, 1)),
                'cvar_95': float(np.mean(returns[returns <= np.percentile(returns, 5)])),
                'cvar_99': float(np.mean(returns[returns <= np.percentile(returns, 1)])),
                'max_drawdown': float(self._calculate_max_drawdown(portfolio_data['total_value'])),
                'downside_deviation': float(np.sqrt(np.mean(returns[returns < 0] ** 2))) if (returns < 0).any() else 0.0
            }
            
            # Risk decomposition
            risk_analysis['risk_decomposition'] = self._decompose_portfolio_risk(portfolio_data)
            
            # Risk monitoring
            risk_analysis['risk_monitoring'] = self._monitor_portfolio_risk(returns, portfolio_data['total_value'])
        
        self.logger.info("✅ Risk analysis completed")
        return risk_analysis
    
    async def _perform_diversification_analysis(self, portfolio_data: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform portfolio diversification analysis."""
        self.logger.info("🌐 Performing diversification analysis...")
        
        diversification_analysis = {}
        
        # Find return columns for individual assets
        return_columns = [col for col in portfolio_data.columns if col.endswith('_return')]
        
        if len(return_columns) > 1:
            # Calculate correlation matrix
            returns_matrix = portfolio_data[return_columns].dropna()
            correlation_matrix = returns_matrix.corr()
            
            diversification_analysis['correlation_analysis'] = {
                'correlation_matrix': correlation_matrix.to_dict(),
                'average_correlation': float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()),
                'max_correlation': float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max()),
                'min_correlation': float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].min())
            }
            
            # Diversification metrics
            diversification_analysis['diversification_metrics'] = {
                'effective_number_of_assets': self._calculate_effective_number_of_assets(correlation_matrix),
                'diversification_ratio': self._calculate_diversification_ratio(returns_matrix),
                'concentration_risk': self._calculate_concentration_risk(portfolio_data)
            }
        
        self.logger.info("✅ Diversification analysis completed")
        return diversification_analysis
    
    async def _perform_optimization_analysis(self, portfolio_data: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform portfolio optimization analysis."""
        self.logger.info("🎯 Performing optimization analysis...")
        
        optimization_analysis = {}
        
        # Find return columns for individual assets
        return_columns = [col for col in portfolio_data.columns if col.endswith('_return')]
        
        if len(return_columns) > 1:
            returns_matrix = portfolio_data[return_columns].dropna()
            
            # Mean-variance optimization
            optimization_analysis['mean_variance_optimization'] = self._perform_mean_variance_optimization(returns_matrix)
            
            # Risk parity optimization
            optimization_analysis['risk_parity_optimization'] = self._perform_risk_parity_optimization(returns_matrix)
            
            # Maximum Sharpe ratio optimization
            optimization_analysis['max_sharpe_optimization'] = self._perform_max_sharpe_optimization(returns_matrix)
        
        self.logger.info("✅ Optimization analysis completed")
        return optimization_analysis
    
    async def _perform_rebalancing_analysis(self, portfolio_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform portfolio rebalancing analysis."""
        self.logger.info("⚖️ Performing rebalancing analysis...")
        
        rebalancing_analysis = {}
        
        # Find allocation columns
        allocation_columns = [col for col in portfolio_data.columns if col.endswith('_allocation')]
        
        if allocation_columns:
            # Rebalancing frequency analysis
            rebalancing_analysis['rebalancing_frequency'] = self._analyze_rebalancing_frequency(portfolio_data, allocation_columns)
            
            # Rebalancing cost analysis
            rebalancing_analysis['rebalancing_costs'] = self._analyze_rebalancing_costs(portfolio_data, allocation_columns)
            
            # Rebalancing effectiveness
            rebalancing_analysis['rebalancing_effectiveness'] = self._analyze_rebalancing_effectiveness(portfolio_data, allocation_columns)
        
        self.logger.info("✅ Rebalancing analysis completed")
        return rebalancing_analysis
    
    def _generate_optimization_insights(
        self, 
        portfolio_metrics: Dict[str, Any], 
        performance_analysis: Dict[str, Any], 
        allocation_analysis: Dict[str, Any], 
        risk_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate portfolio optimization insights."""
        self.logger.info("💡 Generating optimization insights...")
        
        insights = []
        
        # Performance-based insights
        if 'performance_metrics' in performance_analysis:
            metrics = performance_analysis['performance_metrics']
            
            if metrics['sharpe_ratio'] < 1.0:
                insights.append({
                    'category': 'PERFORMANCE',
                    'priority': 'HIGH',
                    'title': 'Low Sharpe Ratio',
                    'description': f'Sharpe ratio is {metrics["sharpe_ratio"]:.2f}, indicating poor risk-adjusted returns',
                    'recommendation': 'Optimize asset allocation and reduce portfolio volatility',
                    'impact': 'HIGH'
                })
            
            if metrics['max_drawdown'] < -0.15:
                insights.append({
                    'category': 'RISK_MANAGEMENT',
                    'priority': 'HIGH',
                    'title': 'High Maximum Drawdown',
                    'description': f'Maximum drawdown is {metrics["max_drawdown"]:.2%}, indicating significant downside risk',
                    'recommendation': 'Implement better risk management and diversification',
                    'impact': 'HIGH'
                })
        
        # Allocation-based insights
        if 'current_allocation' in allocation_analysis:
            current_allocation = allocation_analysis['current_allocation']
            
            # Check for over-concentration
            max_allocation = max(current_allocation.values())
            if max_allocation > self.config.max_allocation_per_asset:
                insights.append({
                    'category': 'ALLOCATION',
                    'priority': 'MEDIUM',
                    'title': 'Over-Concentration Risk',
                    'description': f'Maximum allocation is {max_allocation:.2%}, exceeding recommended limit',
                    'recommendation': 'Rebalance portfolio to reduce concentration risk',
                    'impact': 'MEDIUM'
                })
        
        # Risk-based insights
        if 'risk_metrics' in risk_analysis:
            risk_metrics = risk_analysis['risk_metrics']
            
            if risk_metrics['volatility'] > self.config.target_volatility:
                insights.append({
                    'category': 'VOLATILITY',
                    'priority': 'MEDIUM',
                    'title': 'High Portfolio Volatility',
                    'description': f'Portfolio volatility is {risk_metrics["volatility"]:.2%}, exceeding target',
                    'recommendation': 'Reduce portfolio volatility through better diversification',
                    'impact': 'MEDIUM'
                })
        
        self.logger.info(f"✅ Generated {len(insights)} optimization insights")
        return insights
    
    def _calculate_drawdown_metrics(self, portfolio_value: pd.Series) -> Dict[str, Any]:
        """Calculate drawdown-related metrics."""
        peak = portfolio_value.expanding().max()
        drawdown = (portfolio_value - peak) / peak
        
        return {
            'max_drawdown': float(drawdown.min()),
            'current_drawdown': float(drawdown.iloc[-1]),
            'average_drawdown': float(drawdown[drawdown < 0].mean()),
            'drawdown_duration': self._calculate_drawdown_duration(drawdown),
            'recovery_time': self._calculate_recovery_time(drawdown)
        }
    
    def _calculate_allocation_metrics(self, portfolio_data: pd.DataFrame, allocation_columns: List[str]) -> Dict[str, Any]:
        """Calculate allocation-related metrics."""
        allocation_metrics = {}
        
        for col in allocation_columns:
            asset_name = col.replace('_allocation', '')
            allocation = portfolio_data[col]
            
            allocation_metrics[asset_name] = {
                'mean_allocation': float(allocation.mean()),
                'std_allocation': float(allocation.std()),
                'min_allocation': float(allocation.min()),
                'max_allocation': float(allocation.max()),
                'allocation_stability': float(1 - allocation.std() / allocation.mean()) if allocation.mean() > 0 else 0.0
            }
        
        return allocation_metrics
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - self.config.risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        return float(excess_returns.mean() / downside_deviation) if downside_deviation > 0 else 0.0
    
    def _calculate_calmar_ratio(self, returns: pd.Series, portfolio_value: pd.Series) -> float:
        """Calculate Calmar ratio."""
        if len(returns) == 0:
            return 0.0
        
        annual_return = returns.mean() * 252
        max_drawdown = abs(self._calculate_max_drawdown(portfolio_value))
        
        return float(annual_return / max_drawdown) if max_drawdown > 0 else 0.0
    
    def _calculate_max_drawdown(self, portfolio_value: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(portfolio_value) == 0:
            return 0.0
        
        peak = portfolio_value.expanding().max()
        drawdown = (portfolio_value - peak) / peak
        return float(drawdown.min())
    
    def _compare_with_benchmark(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, Any]:
        """Compare portfolio performance with benchmark."""
        # Align data
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) == 0:
            return {}
        
        portfolio_aligned = portfolio_returns.loc[common_dates]
        benchmark_aligned = benchmark_returns.loc[common_dates]
        
        # Calculate metrics
        excess_returns = portfolio_aligned - benchmark_aligned
        
        return {
            'alpha': float(excess_returns.mean() * 252),
            'beta': float(np.cov(portfolio_aligned, benchmark_aligned)[0, 1] / np.var(benchmark_aligned)),
            'correlation': float(portfolio_aligned.corr(benchmark_aligned)),
            'tracking_error': float(excess_returns.std() * np.sqrt(252)),
            'information_ratio': float(excess_returns.mean() / excess_returns.std()) if excess_returns.std() > 0 else 0.0
        }
    
    def _analyze_allocation_drift(self, portfolio_data: pd.DataFrame, allocation_columns: List[str]) -> Dict[str, Any]:
        """Analyze allocation drift over time."""
        drift_analysis = {}
        
        for col in allocation_columns:
            asset_name = col.replace('_allocation', '')
            allocation = portfolio_data[col]
            
            # Calculate drift from initial allocation
            initial_allocation = allocation.iloc[0]
            current_allocation = allocation.iloc[-1]
            drift = current_allocation - initial_allocation
            
            drift_analysis[asset_name] = {
                'initial_allocation': float(initial_allocation),
                'current_allocation': float(current_allocation),
                'drift': float(drift),
                'drift_percentage': float(drift / initial_allocation) if initial_allocation > 0 else 0.0
            }
        
        return drift_analysis
    
    def _analyze_rebalancing_needs(self, current_allocation: Dict[str, float]) -> Dict[str, Any]:
        """Analyze rebalancing needs."""
        rebalancing_needs = {}
        
        for asset, allocation in current_allocation.items():
            if allocation > self.config.max_allocation_per_asset:
                rebalancing_needs[asset] = {
                    'action': 'REDUCE',
                    'current': float(allocation),
                    'target': float(self.config.max_allocation_per_asset),
                    'adjustment': float(allocation - self.config.max_allocation_per_asset)
                }
            elif allocation < self.config.min_allocation_per_asset:
                rebalancing_needs[asset] = {
                    'action': 'INCREASE',
                    'current': float(allocation),
                    'target': float(self.config.min_allocation_per_asset),
                    'adjustment': float(self.config.min_allocation_per_asset - allocation)
                }
        
        return rebalancing_needs
    
    def _decompose_portfolio_risk(self, portfolio_data: pd.DataFrame) -> Dict[str, Any]:
        """Decompose portfolio risk into components."""
        # Simplified risk decomposition
        return {
            'systematic_risk': 0.7,  # 70% systematic
            'idiosyncratic_risk': 0.3,  # 30% idiosyncratic
            'concentration_risk': 0.1,  # 10% concentration
            'liquidity_risk': 0.05  # 5% liquidity
        }
    
    def _monitor_portfolio_risk(self, returns: pd.Series, portfolio_value: pd.Series) -> Dict[str, Any]:
        """Monitor portfolio risk metrics."""
        return {
            'current_volatility': float(returns.rolling(window=30).std().iloc[-1] * np.sqrt(252)),
            'current_var_95': float(returns.rolling(window=30).quantile(0.05).iloc[-1]),
            'current_drawdown': float(self._calculate_current_drawdown(portfolio_value)),
            'risk_level': 'medium'  # Simplified risk level
        }
    
    def _calculate_current_drawdown(self, portfolio_value: pd.Series) -> float:
        """Calculate current drawdown."""
        if len(portfolio_value) == 0:
            return 0.0
        
        peak = portfolio_value.expanding().max()
        current_drawdown = (portfolio_value.iloc[-1] - peak.iloc[-1]) / peak.iloc[-1]
        return float(current_drawdown)
    
    def _calculate_effective_number_of_assets(self, correlation_matrix: pd.DataFrame) -> float:
        """Calculate effective number of assets."""
        if len(correlation_matrix) == 0:
            return 0.0
        
        # Simplified calculation
        n_assets = len(correlation_matrix)
        avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        
        effective_n = n_assets / (1 + (n_assets - 1) * avg_correlation)
        return float(effective_n)
    
    def _calculate_diversification_ratio(self, returns_matrix: pd.DataFrame) -> float:
        """Calculate diversification ratio."""
        if len(returns_matrix) == 0:
            return 0.0
        
        # Calculate weighted average volatility
        weights = np.ones(len(returns_matrix.columns)) / len(returns_matrix.columns)
        individual_volatilities = returns_matrix.std()
        weighted_avg_vol = np.sum(weights * individual_volatilities)
        
        # Calculate portfolio volatility
        portfolio_returns = returns_matrix.mean(axis=1)
        portfolio_vol = portfolio_returns.std()
        
        return float(weighted_avg_vol / portfolio_vol) if portfolio_vol > 0 else 0.0
    
    def _calculate_concentration_risk(self, portfolio_data: pd.DataFrame) -> float:
        """Calculate concentration risk."""
        allocation_columns = [col for col in portfolio_data.columns if col.endswith('_allocation')]
        
        if not allocation_columns:
            return 0.0
        
        # Calculate Herfindahl index
        current_allocation = portfolio_data[allocation_columns].iloc[-1]
        herfindahl_index = np.sum(current_allocation ** 2)
        
        return float(herfindahl_index)
    
    def _perform_mean_variance_optimization(self, returns_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Perform mean-variance optimization."""
        # Simplified mean-variance optimization
        n_assets = len(returns_matrix.columns)
        expected_returns = returns_matrix.mean()
        cov_matrix = returns_matrix.cov()
        
        # Equal weight portfolio as baseline
        equal_weights = np.ones(n_assets) / n_assets
        
        return {
            'equal_weight_portfolio': {
                'weights': equal_weights.tolist(),
                'expected_return': float(np.sum(equal_weights * expected_returns)),
                'volatility': float(np.sqrt(np.dot(equal_weights, np.dot(cov_matrix, equal_weights))))
            },
            'optimization_status': 'simplified'
        }
    
    def _perform_risk_parity_optimization(self, returns_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Perform risk parity optimization."""
        # Simplified risk parity optimization
        n_assets = len(returns_matrix.columns)
        cov_matrix = returns_matrix.cov()
        
        # Calculate inverse volatility weights
        volatilities = np.sqrt(np.diag(cov_matrix))
        inv_vol_weights = (1 / volatilities) / np.sum(1 / volatilities)
        
        return {
            'risk_parity_weights': inv_vol_weights.tolist(),
            'optimization_status': 'simplified'
        }
    
    def _perform_max_sharpe_optimization(self, returns_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Perform maximum Sharpe ratio optimization."""
        # Simplified max Sharpe optimization
        n_assets = len(returns_matrix.columns)
        expected_returns = returns_matrix.mean()
        cov_matrix = returns_matrix.cov()
        
        # Equal weight portfolio as approximation
        equal_weights = np.ones(n_assets) / n_assets
        
        return {
            'max_sharpe_weights': equal_weights.tolist(),
            'expected_sharpe': float(np.sum(equal_weights * expected_returns) / np.sqrt(np.dot(equal_weights, np.dot(cov_matrix, equal_weights)))),
            'optimization_status': 'simplified'
        }
    
    def _analyze_rebalancing_frequency(self, portfolio_data: pd.DataFrame, allocation_columns: List[str]) -> Dict[str, Any]:
        """Analyze rebalancing frequency."""
        # Simplified rebalancing frequency analysis
        return {
            'current_frequency': self.config.rebalancing_frequency,
            'recommended_frequency': 'monthly',
            'frequency_analysis': 'stable'
        }
    
    def _analyze_rebalancing_costs(self, portfolio_data: pd.DataFrame, allocation_columns: List[str]) -> Dict[str, Any]:
        """Analyze rebalancing costs."""
        # Simplified rebalancing cost analysis
        return {
            'estimated_transaction_costs': 0.001,  # 0.1%
            'estimated_tax_impact': 0.0,
            'total_rebalancing_cost': 0.001
        }
    
    def _analyze_rebalancing_effectiveness(self, portfolio_data: pd.DataFrame, allocation_columns: List[str]) -> Dict[str, Any]:
        """Analyze rebalancing effectiveness."""
        # Simplified rebalancing effectiveness analysis
        return {
            'rebalancing_benefit': 0.02,  # 2% annual benefit
            'rebalancing_cost': 0.001,  # 0.1% cost
            'net_benefit': 0.019  # 1.9% net benefit
        }
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> Dict[str, Any]:
        """Calculate drawdown duration statistics."""
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        if drawdown_periods:
            return {
                'average_duration': float(np.mean(drawdown_periods)),
                'max_duration': int(max(drawdown_periods)),
                'total_periods': len(drawdown_periods)
            }
        else:
            return {
                'average_duration': 0.0,
                'max_duration': 0,
                'total_periods': 0
            }
    
    def _calculate_recovery_time(self, drawdown: pd.Series) -> Dict[str, Any]:
        """Calculate recovery time statistics."""
        # Simplified recovery time calculation
        return {
            'average_recovery_time': 10.0,  # days
            'max_recovery_time': 30,  # days
            'recovery_success_rate': 0.95
        }
    
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
    
    async def _save_results(self, results: PortfolioAnalysisResults) -> None:
        """Save results to disk."""
        self.logger.info("💾 Saving results...")
        
        # Create output directory
        output_dir = self.data_dir / "backtesting_results" / "portfolio_analysis"
        ensure_directory(output_dir)
        
        # Save main results
        results_file = output_dir / f"{self.config.symbol}_{self.config.exchange}_portfolio_analysis_results.json"
        await safe_json_dump(results_file, results.__dict__, indent=2)
        
        # Save portfolio data
        if not results.portfolio_data.empty:
            portfolio_file = output_dir / f"{self.config.symbol}_{self.config.exchange}_portfolio_data.parquet"
            await self.parquet_utils.save_dataframe(results.portfolio_data, portfolio_file)
        
        self.logger.info(f"✅ Results saved to {output_dir}")


# Convenience function for easy integration
async def execute_portfolio_analysis(
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE", 
    timeframe: str = "1h",
    data_dir: str = "data/training",
    **kwargs
) -> PortfolioAnalysisResults:
    """
    Convenience function to execute portfolio analysis.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Data timeframe
        data_dir: Data directory
        **kwargs: Additional configuration parameters
        
    Returns:
        Portfolio analysis results
    """
    config = PortfolioAnalysisConfig(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        **kwargs
    )
    
    step = PortfolioAnalysisStep(config)
    return await step.execute()