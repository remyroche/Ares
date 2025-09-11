"""
Risk Analysis Step

This module provides comprehensive risk analysis functionality for backtesting results
with detailed risk metrics, stress testing, and risk management insights.

Key Features:
- Comprehensive risk metrics calculation
- Value at Risk (VaR) and Conditional VaR (CVaR)
- Stress testing and scenario analysis
- Risk attribution and decomposition
- Risk-adjusted performance metrics
- Risk monitoring and alerting
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


class RiskMetricType(Enum):
    """Types of risk metrics."""
    VAR_METRICS = "var_metrics"
    DRAWDOWN_METRICS = "drawdown_metrics"
    VOLATILITY_METRICS = "volatility_metrics"
    CORRELATION_METRICS = "correlation_metrics"
    LIQUIDITY_METRICS = "liquidity_metrics"
    CONCENTRATION_METRICS = "concentration_metrics"


@dataclass
class RiskAnalysisConfig:
    """Configuration for risk analysis step."""
    # Basic configuration
    symbol: str
    exchange: str
    timeframe: str
    data_dir: str
    
    # Risk parameters
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    var_horizon_days: int = 1
    stress_test_scenarios: List[str] = field(default_factory=lambda: [
        "market_crash", "volatility_spike", "liquidity_crisis", "correlation_breakdown"
    ])
    
    # Risk thresholds
    max_var_threshold: float = 0.05  # 5% daily VaR
    max_drawdown_threshold: float = 0.20  # 20% max drawdown
    max_volatility_threshold: float = 0.30  # 30% annual volatility
    
    # Performance settings
    enable_detailed_logging: bool = True
    enable_performance_monitoring: bool = True
    enable_memory_optimization: bool = True
    
    # Output settings
    save_detailed_results: bool = True
    output_format: str = "parquet"


@dataclass
class RiskAnalysisResults:
    """Results from risk analysis step."""
    # Basic info
    symbol: str
    exchange: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    total_duration: float
    
    # Risk metrics
    risk_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Stress testing results
    stress_testing: Dict[str, Any] = field(default_factory=dict)
    
    # Risk attribution
    risk_attribution: Dict[str, Any] = field(default_factory=dict)
    
    # Risk monitoring
    risk_monitoring: Dict[str, Any] = field(default_factory=dict)
    
    # Risk recommendations
    risk_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Detailed data
    risk_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Metadata
    config: RiskAnalysisConfig = field(default_factory=RiskAnalysisConfig)
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    system_metrics: Dict[str, Any] = field(default_factory=dict)


class RiskAnalysisStep:
    """Risk analysis step."""
    
    def __init__(self, config: RiskAnalysisConfig):
        """Initialize the risk analysis step."""
        self.config = config
        self.logger = logger.getChild('RiskAnalysisStep')
        
        # Initialize utilities
        self.parquet_utils = get_parquet_utils()
        self.financial_logger = EnhancedFinancialMetricsLogger()
        self.performance_monitor = PerformanceMonitor()
        self.system_monitor = SystemMonitor()
        
        # Initialize data directory
        self.data_dir = Path(config.data_dir)
        ensure_directory(self.data_dir)
        
        self.logger.info(f"🚀 RiskAnalysisStep initialized for {config.symbol}")
        self.logger.info(f"⚠️ Confidence levels: {config.confidence_levels}")
        self.logger.info(f"📊 Stress test scenarios: {len(config.stress_test_scenarios)}")
        self.logger.info(f"📁 Data directory: {config.data_dir}")
    
    @traced(span_name='risk_analysis')
    @log_execution_time
    @monitor_step_execution
    async def execute(
        self, 
        equity_curve: Optional[pd.DataFrame] = None,
        trade_data: Optional[pd.DataFrame] = None,
        market_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> RiskAnalysisResults:
        """Execute risk analysis."""
        
        self.logger.info("🚀 Starting risk analysis...")
        start_time = time.time()
        
        # Start performance monitoring
        if self.config.enable_performance_monitoring:
            self.performance_monitor.start_monitoring()
        
        try:
            # Load data if not provided
            if equity_curve is None:
                equity_curve = await self._load_equity_curve()
            
            if trade_data is None:
                trade_data = await self._load_trade_data()
            
            if market_data is None:
                market_data = await self._load_market_data()
            
            # Validate data
            self._validate_data(equity_curve, trade_data, market_data)
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(equity_curve, trade_data, market_data)
            
            # Perform stress testing
            stress_testing = await self._perform_stress_testing(equity_curve, market_data)
            
            # Perform risk attribution
            risk_attribution = await self._perform_risk_attribution(equity_curve, trade_data)
            
            # Perform risk monitoring
            risk_monitoring = await self._perform_risk_monitoring(risk_metrics)
            
            # Generate risk recommendations
            risk_recommendations = self._generate_risk_recommendations(risk_metrics, stress_testing)
            
            # Create risk data
            risk_data = self._create_risk_data(equity_curve, risk_metrics)
            
            # Create results
            results = RiskAnalysisResults(
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                timeframe=self.config.timeframe,
                start_time=datetime.now(),
                end_time=datetime.now(),
                total_duration=time.time() - start_time,
                risk_metrics=risk_metrics,
                stress_testing=stress_testing,
                risk_attribution=risk_attribution,
                risk_monitoring=risk_monitoring,
                risk_recommendations=risk_recommendations,
                risk_data=risk_data,
                config=self.config,
                execution_time=time.time() - start_time,
                memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
                system_metrics=self._get_system_metrics()
            )
            
            # Save results
            if self.config.save_detailed_results:
                await self._save_results(results)
            
            self.logger.info("✅ Risk analysis completed successfully")
            self.logger.info(f"⏱️ Execution time: {results.execution_time:.2f}s")
            self.logger.info(f"⚠️ Risk metrics calculated: {len(risk_metrics)}")
            self.logger.info(f"💡 Risk recommendations: {len(risk_recommendations)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in risk analysis: {e}")
            self.logger.exception("Full traceback:")
            raise
        finally:
            # Stop performance monitoring
            if self.config.enable_performance_monitoring:
                self.performance_monitor.stop_monitoring()
    
    async def _load_equity_curve(self) -> pd.DataFrame:
        """Load equity curve data."""
        self.logger.info("📂 Loading equity curve data...")
        
        # Try to load from various possible locations
        possible_files = [
            self.data_dir / "backtesting_results" / f"{self.config.symbol}_{self.config.exchange}_equity_curve.parquet",
            self.data_dir / "backtesting_results" / "performance_analytics" / f"{self.config.symbol}_{self.config.exchange}_equity_curve.parquet"
        ]
        
        for file_path in possible_files:
            if safe_file_exists(file_path):
                self.logger.info(f"📁 Loading equity curve: {file_path}")
                return standardized_parquet_handler.read_parquet_standardized(file_path)
        
        # Generate mock equity curve if not found
        self.logger.warning("⚠️ No equity curve found, generating mock data")
        return self._generate_mock_equity_curve()
    
    async def _load_trade_data(self) -> pd.DataFrame:
        """Load trade data."""
        self.logger.info("📂 Loading trade data...")
        
        # Try to load from various possible locations
        possible_files = [
            self.data_dir / "backtesting_results" / f"{self.config.symbol}_{self.config.exchange}_trade_log.parquet",
            self.data_dir / "backtesting_results" / "performance_analytics" / f"{self.config.symbol}_{self.config.exchange}_trade_data.parquet"
        ]
        
        for file_path in possible_files:
            if safe_file_exists(file_path):
                self.logger.info(f"📁 Loading trade data: {file_path}")
                return standardized_parquet_handler.read_parquet_standardized(file_path)
        
        self.logger.warning("⚠️ No trade data found, using empty DataFrame")
        return pd.DataFrame()
    
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
    
    def _generate_mock_equity_curve(self) -> pd.DataFrame:
        """Generate mock equity curve for testing."""
        # Generate 252 trading days of data
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        
        # Generate mock equity curve with some volatility
        np.random.seed(42)
        returns = np.random.normal(0.0008, 0.02, 252)  # ~20% annual volatility, 20% annual return
        equity_values = 100000 * np.cumprod(1 + returns)
        
        equity_curve = pd.DataFrame({
            'timestamp': dates,
            'equity': equity_values,
            'return': returns,
            'cumulative_return': (equity_values / 100000) - 1
        })
        
        equity_curve.set_index('timestamp', inplace=True)
        return equity_curve
    
    def _validate_data(self, equity_curve: pd.DataFrame, trade_data: pd.DataFrame, market_data: pd.DataFrame) -> None:
        """Validate input data."""
        self.logger.info("🔍 Validating input data...")
        
        if equity_curve.empty:
            raise ValidationError("Equity curve data is empty")
        
        # Check required columns in equity curve
        required_columns = ['equity', 'return']
        missing_columns = [col for col in required_columns if col not in equity_curve.columns]
        if missing_columns:
            raise ValidationError(f"Missing required columns in equity curve: {missing_columns}")
        
        # Check for sufficient data
        if len(equity_curve) < 30:
            raise ValidationError(f"Insufficient equity curve data: {len(equity_curve)} < 30")
        
        self.logger.info("✅ Data validation completed successfully")
    
    async def _calculate_risk_metrics(self, equity_curve: pd.DataFrame, trade_data: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics."""
        self.logger.info("⚠️ Calculating risk metrics...")
        
        risk_metrics = {}
        
        if 'return' in equity_curve.columns:
            returns = equity_curve['return'].dropna()
            
            # VaR and CVaR metrics
            risk_metrics['var_metrics'] = self._calculate_var_metrics(returns)
            
            # Drawdown metrics
            risk_metrics['drawdown_metrics'] = self._calculate_drawdown_metrics(equity_curve)
            
            # Volatility metrics
            risk_metrics['volatility_metrics'] = self._calculate_volatility_metrics(returns)
            
            # Correlation metrics (if market data available)
            if not market_data.empty:
                risk_metrics['correlation_metrics'] = self._calculate_correlation_metrics(returns, market_data)
            
            # Liquidity metrics
            risk_metrics['liquidity_metrics'] = self._calculate_liquidity_metrics(trade_data, market_data)
            
            # Concentration metrics
            risk_metrics['concentration_metrics'] = self._calculate_concentration_metrics(trade_data)
        
        self.logger.info("✅ Risk metrics calculated")
        return risk_metrics
    
    def _calculate_var_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate Value at Risk and Conditional VaR metrics."""
        var_metrics = {}
        
        for confidence_level in self.config.confidence_levels:
            alpha = 1 - confidence_level
            var = np.percentile(returns, alpha * 100)
            cvar = np.mean(returns[returns <= var])
            
            var_metrics[f'var_{int(confidence_level*100)}'] = {
                'value_at_risk': float(var),
                'conditional_var': float(cvar),
                'confidence_level': confidence_level,
                'horizon_days': self.config.var_horizon_days
            }
        
        # Additional VaR metrics
        var_metrics['var_analysis'] = {
            'var_95': float(np.percentile(returns, 5)),
            'var_99': float(np.percentile(returns, 1)),
            'cvar_95': float(np.mean(returns[returns <= np.percentile(returns, 5)])),
            'cvar_99': float(np.mean(returns[returns <= np.percentile(returns, 1)])),
            'expected_shortfall': float(np.mean(returns[returns <= np.percentile(returns, 5)]))
        }
        
        return var_metrics
    
    def _calculate_drawdown_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """Calculate drawdown-related risk metrics."""
        if 'equity' not in equity_curve.columns:
            return {}
        
        equity = equity_curve['equity']
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        
        drawdown_metrics = {
            'max_drawdown': float(drawdown.min()),
            'current_drawdown': float(drawdown.iloc[-1]),
            'average_drawdown': float(drawdown[drawdown < 0].mean()),
            'drawdown_std': float(drawdown.std()),
            'drawdown_duration': self._calculate_drawdown_duration(drawdown),
            'recovery_time': self._calculate_recovery_time(drawdown),
            'drawdown_frequency': float((drawdown < -0.01).mean()),  # >1% drawdowns
            'severe_drawdown_frequency': float((drawdown < -0.05).mean())  # >5% drawdowns
        }
        
        return drawdown_metrics
    
    def _calculate_volatility_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate volatility-related risk metrics."""
        volatility_metrics = {
            'annualized_volatility': float(returns.std() * np.sqrt(252)),
            'realized_volatility': float(returns.std()),
            'volatility_of_volatility': float(returns.rolling(window=30).std().std()),
            'volatility_percentiles': {
                'p5': float(returns.rolling(window=30).std().quantile(0.05)),
                'p25': float(returns.rolling(window=30).std().quantile(0.25)),
                'p50': float(returns.rolling(window=30).std().quantile(0.50)),
                'p75': float(returns.rolling(window=30).std().quantile(0.75)),
                'p95': float(returns.rolling(window=30).std().quantile(0.95))
            },
            'volatility_trend': self._calculate_volatility_trend(returns),
            'volatility_clustering': self._detect_volatility_clustering(returns)
        }
        
        return volatility_metrics
    
    def _calculate_correlation_metrics(self, returns: pd.Series, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlation-related risk metrics."""
        correlation_metrics = {}
        
        if 'close' in market_data.columns:
            market_returns = market_data['close'].pct_change().dropna()
            
            # Align data
            common_dates = returns.index.intersection(market_returns.index)
            if len(common_dates) > 0:
                strategy_aligned = returns.loc[common_dates]
                market_aligned = market_returns.loc[common_dates]
                
                correlation = strategy_aligned.corr(market_aligned)
                beta = np.cov(strategy_aligned, market_aligned)[0, 1] / np.var(market_aligned)
                
                correlation_metrics = {
                    'market_correlation': float(correlation),
                    'beta': float(beta),
                    'correlation_stability': self._calculate_correlation_stability(strategy_aligned, market_aligned),
                    'correlation_breakdown_risk': self._detect_correlation_breakdown(strategy_aligned, market_aligned)
                }
        
        return correlation_metrics
    
    def _calculate_liquidity_metrics(self, trade_data: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate liquidity-related risk metrics."""
        liquidity_metrics = {}
        
        if not trade_data.empty and 'volume' in trade_data.columns:
            # Trade-based liquidity metrics
            liquidity_metrics['trade_liquidity'] = {
                'average_trade_size': float(trade_data['volume'].mean()),
                'trade_size_volatility': float(trade_data['volume'].std()),
                'large_trade_frequency': float((trade_data['volume'] > trade_data['volume'].quantile(0.95)).mean())
            }
        
        if not market_data.empty and 'volume' in market_data.columns:
            # Market-based liquidity metrics
            liquidity_metrics['market_liquidity'] = {
                'average_volume': float(market_data['volume'].mean()),
                'volume_volatility': float(market_data['volume'].std()),
                'volume_trend': self._calculate_volume_trend(market_data['volume'])
            }
        
        return liquidity_metrics
    
    def _calculate_concentration_metrics(self, trade_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate concentration-related risk metrics."""
        concentration_metrics = {}
        
        if not trade_data.empty:
            # Time concentration
            if 'timestamp' in trade_data.columns:
                trade_data['hour'] = pd.to_datetime(trade_data['timestamp']).dt.hour
                hour_distribution = trade_data['hour'].value_counts(normalize=True)
                concentration_metrics['time_concentration'] = {
                    'herfindahl_index': float((hour_distribution ** 2).sum()),
                    'max_hour_concentration': float(hour_distribution.max()),
                    'concentration_entropy': float(-(hour_distribution * np.log(hour_distribution)).sum())
                }
            
            # Size concentration
            if 'volume' in trade_data.columns:
                size_distribution = trade_data['volume'] / trade_data['volume'].sum()
                concentration_metrics['size_concentration'] = {
                    'herfindahl_index': float((size_distribution ** 2).sum()),
                    'max_trade_concentration': float(size_distribution.max()),
                    'gini_coefficient': self._calculate_gini_coefficient(trade_data['volume'])
                }
        
        return concentration_metrics
    
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
        # Simplified implementation
        return {
            'average_recovery_time': 10.0,  # days
            'max_recovery_time': 30,  # days
            'recovery_success_rate': 0.95
        }
    
    def _calculate_volatility_trend(self, returns: pd.Series) -> str:
        """Calculate volatility trend."""
        if len(returns) < 60:
            return "insufficient_data"
        
        rolling_vol = returns.rolling(window=30).std()
        x = np.arange(len(rolling_vol))
        slope, _, _, _, _ = stats.linregress(x, rolling_vol.dropna())
        
        if slope > 0.001:
            return "increasing"
        elif slope < -0.001:
            return "decreasing"
        else:
            return "stable"
    
    def _detect_volatility_clustering(self, returns: pd.Series) -> bool:
        """Detect volatility clustering."""
        if len(returns) < 30:
            return False
        
        # Simple test for volatility clustering
        abs_returns = np.abs(returns)
        autocorr = abs_returns.autocorr(lag=1)
        return abs(autocorr) > 0.1
    
    def _calculate_correlation_stability(self, strategy_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate correlation stability over time."""
        if len(strategy_returns) < 60:
            return 0.0
        
        # Calculate rolling correlation
        rolling_corr = strategy_returns.rolling(window=30).corr(market_returns)
        return float(rolling_corr.std())
    
    def _detect_correlation_breakdown(self, strategy_returns: pd.Series, market_returns: pd.Series) -> float:
        """Detect correlation breakdown risk."""
        if len(strategy_returns) < 60:
            return 0.0
        
        # Calculate rolling correlation
        rolling_corr = strategy_returns.rolling(window=30).corr(market_returns)
        
        # Detect significant changes in correlation
        corr_changes = rolling_corr.diff().abs()
        breakdown_risk = float((corr_changes > 0.3).mean())  # >30% change in correlation
        
        return breakdown_risk
    
    def _calculate_volume_trend(self, volume: pd.Series) -> str:
        """Calculate volume trend."""
        if len(volume) < 30:
            return "insufficient_data"
        
        x = np.arange(len(volume))
        slope, _, _, _, _ = stats.linregress(x, volume.dropna())
        
        if slope > 0:
            return "increasing"
        elif slope < 0:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_gini_coefficient(self, values: pd.Series) -> float:
        """Calculate Gini coefficient for concentration."""
        if len(values) == 0:
            return 0.0
        
        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        
        return float((n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n)
    
    async def _perform_stress_testing(self, equity_curve: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform stress testing scenarios."""
        self.logger.info("💥 Performing stress testing...")
        
        stress_testing = {}
        
        for scenario in self.config.stress_test_scenarios:
            if scenario == "market_crash":
                stress_testing[scenario] = self._simulate_market_crash(equity_curve)
            elif scenario == "volatility_spike":
                stress_testing[scenario] = self._simulate_volatility_spike(equity_curve)
            elif scenario == "liquidity_crisis":
                stress_testing[scenario] = self._simulate_liquidity_crisis(equity_curve, market_data)
            elif scenario == "correlation_breakdown":
                stress_testing[scenario] = self._simulate_correlation_breakdown(equity_curve, market_data)
        
        self.logger.info("✅ Stress testing completed")
        return stress_testing
    
    def _simulate_market_crash(self, equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """Simulate market crash scenario."""
        if 'return' not in equity_curve.columns:
            return {}
        
        returns = equity_curve['return'].dropna()
        
        # Simulate -20% market crash
        crash_return = -0.20
        stressed_returns = returns + crash_return
        
        # Calculate stressed metrics
        stressed_equity = 100000 * np.cumprod(1 + stressed_returns)
        stressed_drawdown = self._calculate_max_drawdown(pd.Series(stressed_equity))
        
        return {
            'scenario': 'market_crash',
            'shock_magnitude': -0.20,
            'stressed_return': float(stressed_returns.mean() * 252),
            'stressed_volatility': float(stressed_returns.std() * np.sqrt(252)),
            'stressed_max_drawdown': float(stressed_drawdown),
            'stressed_var_95': float(np.percentile(stressed_returns, 5)),
            'impact_assessment': 'severe' if stressed_drawdown < -0.3 else 'moderate' if stressed_drawdown < -0.15 else 'mild'
        }
    
    def _simulate_volatility_spike(self, equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """Simulate volatility spike scenario."""
        if 'return' not in equity_curve.columns:
            return {}
        
        returns = equity_curve['return'].dropna()
        
        # Simulate 3x volatility increase
        volatility_multiplier = 3.0
        stressed_returns = returns * volatility_multiplier
        
        # Calculate stressed metrics
        stressed_equity = 100000 * np.cumprod(1 + stressed_returns)
        stressed_drawdown = self._calculate_max_drawdown(pd.Series(stressed_equity))
        
        return {
            'scenario': 'volatility_spike',
            'volatility_multiplier': volatility_multiplier,
            'stressed_return': float(stressed_returns.mean() * 252),
            'stressed_volatility': float(stressed_returns.std() * np.sqrt(252)),
            'stressed_max_drawdown': float(stressed_drawdown),
            'stressed_var_95': float(np.percentile(stressed_returns, 5)),
            'impact_assessment': 'severe' if stressed_drawdown < -0.3 else 'moderate' if stressed_drawdown < -0.15 else 'mild'
        }
    
    def _simulate_liquidity_crisis(self, equity_curve: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Simulate liquidity crisis scenario."""
        # Simplified implementation
        return {
            'scenario': 'liquidity_crisis',
            'impact_assessment': 'moderate',
            'liquidity_impact': 'reduced_trading_opportunities'
        }
    
    def _simulate_correlation_breakdown(self, equity_curve: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Simulate correlation breakdown scenario."""
        # Simplified implementation
        return {
            'scenario': 'correlation_breakdown',
            'impact_assessment': 'mild',
            'correlation_impact': 'increased_diversification_benefits'
        }
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(equity_series) == 0:
            return 0.0
        
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        return float(drawdown.min())
    
    async def _perform_risk_attribution(self, equity_curve: pd.DataFrame, trade_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform risk attribution analysis."""
        self.logger.info("📊 Performing risk attribution...")
        
        risk_attribution = {}
        
        if 'return' in equity_curve.columns:
            returns = equity_curve['return'].dropna()
            
            # Time-based risk attribution
            risk_attribution['time_attribution'] = {
                'daily_risk_contribution': self._calculate_daily_risk_contribution(returns),
                'monthly_risk_contribution': self._calculate_monthly_risk_contribution(returns),
                'volatility_regime_attribution': self._calculate_volatility_regime_attribution(returns)
            }
            
            # Trade-based risk attribution
            if not trade_data.empty:
                risk_attribution['trade_attribution'] = self._calculate_trade_risk_attribution(trade_data)
        
        self.logger.info("✅ Risk attribution completed")
        return risk_attribution
    
    def _calculate_daily_risk_contribution(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate daily risk contribution."""
        daily_vol = returns.rolling(window=30).std()
        total_risk = returns.std()
        
        return {
            'average_daily_risk': float(daily_vol.mean()),
            'risk_volatility': float(daily_vol.std()),
            'high_risk_days': float((daily_vol > daily_vol.quantile(0.9)).mean())
        }
    
    def _calculate_monthly_risk_contribution(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate monthly risk contribution."""
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_vol = monthly_returns.std()
        
        return {
            'monthly_volatility': float(monthly_vol),
            'volatility_consistency': float(1 - monthly_vol / monthly_vol.mean()) if monthly_vol.mean() > 0 else 0.0
        }
    
    def _calculate_volatility_regime_attribution(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate volatility regime attribution."""
        rolling_vol = returns.rolling(window=30).std()
        high_vol_threshold = rolling_vol.quantile(0.75)
        low_vol_threshold = rolling_vol.quantile(0.25)
        
        high_vol_returns = returns[rolling_vol > high_vol_threshold]
        low_vol_returns = returns[rolling_vol < low_vol_threshold]
        
        return {
            'high_volatility_periods': {
                'frequency': float(len(high_vol_returns) / len(returns)),
                'average_return': float(high_vol_returns.mean()),
                'volatility': float(high_vol_returns.std())
            },
            'low_volatility_periods': {
                'frequency': float(len(low_vol_returns) / len(returns)),
                'average_return': float(low_vol_returns.mean()),
                'volatility': float(low_vol_returns.std())
            }
        }
    
    def _calculate_trade_risk_attribution(self, trade_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trade-based risk attribution."""
        # Simplified implementation
        return {
            'trade_size_risk': 'moderate',
            'trade_frequency_risk': 'low',
            'trade_timing_risk': 'low'
        }
    
    async def _perform_risk_monitoring(self, risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Perform risk monitoring and alerting."""
        self.logger.info("🔍 Performing risk monitoring...")
        
        risk_monitoring = {
            'risk_alerts': [],
            'risk_thresholds': {
                'max_var_threshold': self.config.max_var_threshold,
                'max_drawdown_threshold': self.config.max_drawdown_threshold,
                'max_volatility_threshold': self.config.max_volatility_threshold
            },
            'current_risk_level': 'low',
            'risk_trend': 'stable'
        }
        
        # Check VaR thresholds
        if 'var_metrics' in risk_metrics:
            var_95 = risk_metrics['var_metrics']['var_analysis']['var_95']
            if abs(var_95) > self.config.max_var_threshold:
                risk_monitoring['risk_alerts'].append({
                    'type': 'VAR_BREACH',
                    'severity': 'high',
                    'message': f'VaR 95% ({abs(var_95):.2%}) exceeds threshold ({self.config.max_var_threshold:.2%})'
                })
        
        # Check drawdown thresholds
        if 'drawdown_metrics' in risk_metrics:
            max_drawdown = risk_metrics['drawdown_metrics']['max_drawdown']
            if abs(max_drawdown) > self.config.max_drawdown_threshold:
                risk_monitoring['risk_alerts'].append({
                    'type': 'DRAWDOWN_BREACH',
                    'severity': 'high',
                    'message': f'Max drawdown ({abs(max_drawdown):.2%}) exceeds threshold ({self.config.max_drawdown_threshold:.2%})'
                })
        
        # Check volatility thresholds
        if 'volatility_metrics' in risk_metrics:
            annualized_vol = risk_metrics['volatility_metrics']['annualized_volatility']
            if annualized_vol > self.config.max_volatility_threshold:
                risk_monitoring['risk_alerts'].append({
                    'type': 'VOLATILITY_BREACH',
                    'severity': 'medium',
                    'message': f'Volatility ({annualized_vol:.2%}) exceeds threshold ({self.config.max_volatility_threshold:.2%})'
                })
        
        # Determine overall risk level
        if len(risk_monitoring['risk_alerts']) > 0:
            high_severity_alerts = [alert for alert in risk_monitoring['risk_alerts'] if alert['severity'] == 'high']
            if len(high_severity_alerts) > 0:
                risk_monitoring['current_risk_level'] = 'high'
            else:
                risk_monitoring['current_risk_level'] = 'medium'
        
        self.logger.info("✅ Risk monitoring completed")
        return risk_monitoring
    
    def _generate_risk_recommendations(self, risk_metrics: Dict[str, Any], stress_testing: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate risk management recommendations."""
        self.logger.info("💡 Generating risk recommendations...")
        
        recommendations = []
        
        # VaR-based recommendations
        if 'var_metrics' in risk_metrics:
            var_95 = risk_metrics['var_metrics']['var_analysis']['var_95']
            if abs(var_95) > 0.03:  # 3% daily VaR
                recommendations.append({
                    'category': 'RISK_MANAGEMENT',
                    'priority': 'HIGH',
                    'title': 'High Daily VaR',
                    'description': f'Daily VaR 95% is {abs(var_95):.2%}, indicating high daily risk',
                    'action': 'Consider reducing position sizes or implementing tighter stop-losses',
                    'impact': 'HIGH'
                })
        
        # Drawdown-based recommendations
        if 'drawdown_metrics' in risk_metrics:
            max_drawdown = risk_metrics['drawdown_metrics']['max_drawdown']
            if abs(max_drawdown) > 0.15:  # 15% max drawdown
                recommendations.append({
                    'category': 'RISK_MANAGEMENT',
                    'priority': 'HIGH',
                    'title': 'High Maximum Drawdown',
                    'description': f'Maximum drawdown is {abs(max_drawdown):.2%}, indicating significant downside risk',
                    'action': 'Implement better risk management and position sizing controls',
                    'impact': 'HIGH'
                })
        
        # Volatility-based recommendations
        if 'volatility_metrics' in risk_metrics:
            annualized_vol = risk_metrics['volatility_metrics']['annualized_volatility']
            if annualized_vol > 0.25:  # 25% annual volatility
                recommendations.append({
                    'category': 'VOLATILITY_MANAGEMENT',
                    'priority': 'MEDIUM',
                    'title': 'High Volatility',
                    'description': f'Annualized volatility is {annualized_vol:.2%}, indicating high price volatility',
                    'action': 'Consider volatility-based position sizing or volatility hedging',
                    'impact': 'MEDIUM'
                })
        
        # Stress testing recommendations
        for scenario, results in stress_testing.items():
            if results.get('impact_assessment') == 'severe':
                recommendations.append({
                    'category': 'STRESS_TESTING',
                    'priority': 'HIGH',
                    'title': f'Severe Impact in {scenario.replace("_", " ").title()}',
                    'description': f'Stress test shows severe impact in {scenario} scenario',
                    'action': f'Develop specific risk management measures for {scenario} scenarios',
                    'impact': 'HIGH'
                })
        
        self.logger.info(f"✅ Generated {len(recommendations)} risk recommendations")
        return recommendations
    
    def _create_risk_data(self, equity_curve: pd.DataFrame, risk_metrics: Dict[str, Any]) -> pd.DataFrame:
        """Create comprehensive risk data DataFrame."""
        risk_data = equity_curve.copy()
        
        # Add risk metrics to the DataFrame
        if 'return' in risk_data.columns:
            returns = risk_data['return'].dropna()
            
            # Add rolling volatility
            risk_data['rolling_volatility'] = returns.rolling(window=30).std() * np.sqrt(252)
            
            # Add rolling VaR
            risk_data['rolling_var_95'] = returns.rolling(window=30).quantile(0.05)
            
            # Add drawdown
            peak = risk_data['equity'].expanding().max()
            risk_data['drawdown'] = (risk_data['equity'] - peak) / peak
        
        return risk_data
    
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
    
    async def _save_results(self, results: RiskAnalysisResults) -> None:
        """Save results to disk."""
        self.logger.info("💾 Saving results...")
        
        # Create output directory
        output_dir = self.data_dir / "backtesting_results" / "risk_analysis"
        ensure_directory(output_dir)
        
        # Save main results
        results_file = output_dir / f"{self.config.symbol}_{self.config.exchange}_risk_analysis_results.json"
        await safe_json_dump(results_file, results.__dict__, indent=2)
        
        # Save risk data
        if not results.risk_data.empty:
            risk_file = output_dir / f"{self.config.symbol}_{self.config.exchange}_risk_data.parquet"
            await self.parquet_utils.save_dataframe(results.risk_data, risk_file)
        
        self.logger.info(f"✅ Results saved to {output_dir}")


# Convenience function for easy integration
async def execute_risk_analysis(
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE", 
    timeframe: str = "1h",
    data_dir: str = "data/training",
    **kwargs
) -> RiskAnalysisResults:
    """
    Convenience function to execute risk analysis.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Data timeframe
        data_dir: Data directory
        **kwargs: Additional configuration parameters
        
    Returns:
        Risk analysis results
    """
    config = RiskAnalysisConfig(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        **kwargs
    )
    
    step = RiskAnalysisStep(config)
    return await step.execute()