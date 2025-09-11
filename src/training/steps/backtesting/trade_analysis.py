"""
Trade Analysis Step

This module provides comprehensive trade-level analysis functionality for backtesting results
with detailed trade statistics, performance metrics, and trade pattern analysis.

Key Features:
- Trade-level performance metrics
- Trade pattern analysis
- Trade timing analysis
- Trade size analysis
- Trade correlation analysis
- Trade optimization insights
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


class TradeAnalysisType(Enum):
    """Types of trade analysis."""
    PERFORMANCE_ANALYSIS = "performance_analysis"
    PATTERN_ANALYSIS = "pattern_analysis"
    TIMING_ANALYSIS = "timing_analysis"
    SIZE_ANALYSIS = "size_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"
    OPTIMIZATION_ANALYSIS = "optimization_analysis"


@dataclass
class TradeAnalysisConfig:
    """Configuration for trade analysis step."""
    # Basic configuration
    symbol: str
    exchange: str
    timeframe: str
    data_dir: str
    
    # Analysis parameters
    min_trade_duration: int = 1  # minutes
    max_trade_duration: int = 1440  # 24 hours
    min_trade_size: float = 0.001
    max_trade_size: float = 1000.0
    
    # Performance thresholds
    min_profit_threshold: float = 0.001  # 0.1%
    max_loss_threshold: float = -0.05  # -5%
    
    # Analysis settings
    enable_detailed_logging: bool = True
    enable_performance_monitoring: bool = True
    enable_memory_optimization: bool = True
    
    # Output settings
    save_detailed_results: bool = True
    output_format: str = "parquet"


@dataclass
class TradeAnalysisResults:
    """Results from trade analysis step."""
    # Basic info
    symbol: str
    exchange: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    total_duration: float
    
    # Trade statistics
    trade_statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Performance analysis
    performance_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Pattern analysis
    pattern_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Timing analysis
    timing_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Size analysis
    size_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Correlation analysis
    correlation_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Optimization insights
    optimization_insights: List[Dict[str, Any]] = field(default_factory=list)
    
    # Detailed data
    trade_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Metadata
    config: TradeAnalysisConfig = field(default_factory=TradeAnalysisConfig)
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    system_metrics: Dict[str, Any] = field(default_factory=dict)


class TradeAnalysisStep:
    """Trade analysis step."""
    
    def __init__(self, config: TradeAnalysisConfig):
        """Initialize the trade analysis step."""
        self.config = config
        self.logger = logger.getChild('TradeAnalysisStep')
        
        # Initialize utilities
        self.parquet_utils = get_parquet_utils()
        self.financial_logger = EnhancedFinancialMetricsLogger()
        self.performance_monitor = PerformanceMonitor()
        self.system_monitor = SystemMonitor()
        
        # Initialize data directory
        self.data_dir = Path(config.data_dir)
        ensure_directory(self.data_dir)
        
        self.logger.info(f"🚀 TradeAnalysisStep initialized for {config.symbol}")
        self.logger.info(f"📊 Min trade duration: {config.min_trade_duration} minutes")
        self.logger.info(f"📊 Max trade duration: {config.max_trade_duration} minutes")
        self.logger.info(f"📁 Data directory: {config.data_dir}")
    
    @traced(span_name='trade_analysis')
    @log_execution_time
    @monitor_step_execution
    async def execute(
        self, 
        trade_data: Optional[pd.DataFrame] = None,
        market_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> TradeAnalysisResults:
        """Execute trade analysis."""
        
        self.logger.info("🚀 Starting trade analysis...")
        start_time = time.time()
        
        # Start performance monitoring
        if self.config.enable_performance_monitoring:
            self.performance_monitor.start_monitoring()
        
        try:
            # Load data if not provided
            if trade_data is None:
                trade_data = await self._load_trade_data()
            
            if market_data is None:
                market_data = await self._load_market_data()
            
            # Validate data
            self._validate_data(trade_data, market_data)
            
            # Calculate trade statistics
            trade_statistics = await self._calculate_trade_statistics(trade_data)
            
            # Perform performance analysis
            performance_analysis = await self._perform_performance_analysis(trade_data)
            
            # Perform pattern analysis
            pattern_analysis = await self._perform_pattern_analysis(trade_data)
            
            # Perform timing analysis
            timing_analysis = await self._perform_timing_analysis(trade_data)
            
            # Perform size analysis
            size_analysis = await self._perform_size_analysis(trade_data)
            
            # Perform correlation analysis
            correlation_analysis = await self._perform_correlation_analysis(trade_data, market_data)
            
            # Generate optimization insights
            optimization_insights = self._generate_optimization_insights(
                trade_statistics, performance_analysis, pattern_analysis
            )
            
            # Create results
            results = TradeAnalysisResults(
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                timeframe=self.config.timeframe,
                start_time=datetime.now(),
                end_time=datetime.now(),
                total_duration=time.time() - start_time,
                trade_statistics=trade_statistics,
                performance_analysis=performance_analysis,
                pattern_analysis=pattern_analysis,
                timing_analysis=timing_analysis,
                size_analysis=size_analysis,
                correlation_analysis=correlation_analysis,
                optimization_insights=optimization_insights,
                trade_data=trade_data,
                config=self.config,
                execution_time=time.time() - start_time,
                memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
                system_metrics=self._get_system_metrics()
            )
            
            # Save results
            if self.config.save_detailed_results:
                await self._save_results(results)
            
            self.logger.info("✅ Trade analysis completed successfully")
            self.logger.info(f"⏱️ Execution time: {results.execution_time:.2f}s")
            self.logger.info(f"📊 Total trades analyzed: {len(trade_data)}")
            self.logger.info(f"💡 Optimization insights: {len(optimization_insights)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in trade analysis: {e}")
            self.logger.exception("Full traceback:")
            raise
        finally:
            # Stop performance monitoring
            if self.config.enable_performance_monitoring:
                self.performance_monitor.stop_monitoring()
    
    async def _load_trade_data(self) -> pd.DataFrame:
        """Load trade data."""
        self.logger.info("📂 Loading trade data...")
        
        # Try to load from various possible locations
        possible_files = [
            self.data_dir / "backtesting_results" / f"{self.config.symbol}_{self.config.exchange}_trade_log.parquet",
            self.data_dir / "backtesting_results" / "performance_analytics" / f"{self.config.symbol}_{self.config.exchange}_trade_data.parquet",
            self.data_dir / "backtesting_results" / "basic_backtesting_pre" / f"{self.config.symbol}_{self.config.exchange}_trade_log.parquet"
        ]
        
        for file_path in possible_files:
            if safe_file_exists(file_path):
                self.logger.info(f"📁 Loading trade data: {file_path}")
                return standardized_parquet_handler.read_parquet_standardized(file_path)
        
        # Generate mock trade data if not found
        self.logger.warning("⚠️ No trade data found, generating mock data")
        return self._generate_mock_trade_data()
    
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
    
    def _generate_mock_trade_data(self) -> pd.DataFrame:
        """Generate mock trade data for testing."""
        # Generate 100 mock trades
        np.random.seed(42)
        n_trades = 100
        
        # Generate trade data
        trade_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=n_trades, freq='H'),
            'side': np.random.choice(['BUY', 'SELL'], n_trades),
            'size': np.random.uniform(0.1, 10.0, n_trades),
            'price': np.random.uniform(1000, 2000, n_trades),
            'fee': np.random.uniform(0.001, 0.01, n_trades),
            'pnl': np.random.normal(0, 50, n_trades),
            'duration_minutes': np.random.uniform(1, 1440, n_trades)
        })
        
        # Add some realistic patterns
        trade_data['pnl'] = trade_data['pnl'] * (1 + 0.1 * np.sin(np.arange(n_trades) * 0.1))
        trade_data['size'] = trade_data['size'] * (1 + 0.2 * np.cos(np.arange(n_trades) * 0.05))
        
        return trade_data
    
    def _validate_data(self, trade_data: pd.DataFrame, market_data: pd.DataFrame) -> None:
        """Validate input data."""
        self.logger.info("🔍 Validating input data...")
        
        if trade_data.empty:
            raise ValidationError("Trade data is empty")
        
        # Check required columns in trade data
        required_columns = ['timestamp', 'side', 'size', 'price']
        missing_columns = [col for col in required_columns if col not in trade_data.columns]
        if missing_columns:
            raise ValidationError(f"Missing required columns in trade data: {missing_columns}")
        
        # Check for sufficient data
        if len(trade_data) < 10:
            raise ValidationError(f"Insufficient trade data: {len(trade_data)} < 10")
        
        self.logger.info("✅ Data validation completed successfully")
    
    async def _calculate_trade_statistics(self, trade_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive trade statistics."""
        self.logger.info("📊 Calculating trade statistics...")
        
        trade_statistics = {}
        
        # Basic trade counts
        trade_statistics['basic_counts'] = {
            'total_trades': len(trade_data),
            'buy_trades': len(trade_data[trade_data['side'] == 'BUY']),
            'sell_trades': len(trade_data[trade_data['side'] == 'SELL']),
            'profitable_trades': len(trade_data[trade_data.get('pnl', 0) > 0]),
            'losing_trades': len(trade_data[trade_data.get('pnl', 0) < 0])
        }
        
        # Trade size statistics
        if 'size' in trade_data.columns:
            trade_statistics['size_statistics'] = {
                'average_size': float(trade_data['size'].mean()),
                'median_size': float(trade_data['size'].median()),
                'std_size': float(trade_data['size'].std()),
                'min_size': float(trade_data['size'].min()),
                'max_size': float(trade_data['size'].max()),
                'size_percentiles': {
                    'p25': float(trade_data['size'].quantile(0.25)),
                    'p75': float(trade_data['size'].quantile(0.75)),
                    'p90': float(trade_data['size'].quantile(0.90)),
                    'p95': float(trade_data['size'].quantile(0.95))
                }
            }
        
        # Trade price statistics
        if 'price' in trade_data.columns:
            trade_statistics['price_statistics'] = {
                'average_price': float(trade_data['price'].mean()),
                'median_price': float(trade_data['price'].median()),
                'std_price': float(trade_data['price'].std()),
                'min_price': float(trade_data['price'].min()),
                'max_price': float(trade_data['price'].max()),
                'price_range': float(trade_data['price'].max() - trade_data['price'].min())
            }
        
        # Trade duration statistics
        if 'duration_minutes' in trade_data.columns:
            trade_statistics['duration_statistics'] = {
                'average_duration': float(trade_data['duration_minutes'].mean()),
                'median_duration': float(trade_data['duration_minutes'].median()),
                'std_duration': float(trade_data['duration_minutes'].std()),
                'min_duration': float(trade_data['duration_minutes'].min()),
                'max_duration': float(trade_data['duration_minutes'].max())
            }
        
        # Trade PnL statistics
        if 'pnl' in trade_data.columns:
            pnl = trade_data['pnl'].dropna()
            trade_statistics['pnl_statistics'] = {
                'total_pnl': float(pnl.sum()),
                'average_pnl': float(pnl.mean()),
                'median_pnl': float(pnl.median()),
                'std_pnl': float(pnl.std()),
                'min_pnl': float(pnl.min()),
                'max_pnl': float(pnl.max()),
                'win_rate': float((pnl > 0).mean()),
                'profit_factor': float(pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum())) if (pnl < 0).any() else float('inf')
            }
        
        self.logger.info("✅ Trade statistics calculated")
        return trade_statistics
    
    async def _perform_performance_analysis(self, trade_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform trade performance analysis."""
        self.logger.info("📈 Performing performance analysis...")
        
        performance_analysis = {}
        
        if 'pnl' in trade_data.columns:
            pnl = trade_data['pnl'].dropna()
            
            # Performance metrics
            performance_analysis['performance_metrics'] = {
                'total_return': float(pnl.sum()),
                'average_return': float(pnl.mean()),
                'return_volatility': float(pnl.std()),
                'sharpe_ratio': float(pnl.mean() / pnl.std()) if pnl.std() > 0 else 0.0,
                'max_drawdown': float(self._calculate_max_drawdown(pnl)),
                'win_rate': float((pnl > 0).mean()),
                'loss_rate': float((pnl < 0).mean()),
                'break_even_rate': float((pnl == 0).mean())
            }
            
            # Risk metrics
            performance_analysis['risk_metrics'] = {
                'var_95': float(np.percentile(pnl, 5)),
                'var_99': float(np.percentile(pnl, 1)),
                'cvar_95': float(np.mean(pnl[pnl <= np.percentile(pnl, 5)])),
                'cvar_99': float(np.mean(pnl[pnl <= np.percentile(pnl, 1)])),
                'downside_deviation': float(np.sqrt(np.mean(pnl[pnl < 0] ** 2))) if (pnl < 0).any() else 0.0
            }
            
            # Performance consistency
            performance_analysis['consistency_metrics'] = {
                'consecutive_wins': self._calculate_consecutive_wins(pnl),
                'consecutive_losses': self._calculate_consecutive_losses(pnl),
                'performance_stability': self._calculate_performance_stability(pnl)
            }
        
        self.logger.info("✅ Performance analysis completed")
        return performance_analysis
    
    async def _perform_pattern_analysis(self, trade_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform trade pattern analysis."""
        self.logger.info("🔍 Performing pattern analysis...")
        
        pattern_analysis = {}
        
        # Side pattern analysis
        if 'side' in trade_data.columns:
            pattern_analysis['side_patterns'] = {
                'buy_sell_ratio': float(len(trade_data[trade_data['side'] == 'BUY']) / len(trade_data[trade_data['side'] == 'SELL'])) if (trade_data['side'] == 'SELL').any() else float('inf'),
                'side_alternation': self._calculate_side_alternation(trade_data['side']),
                'side_clustering': self._detect_side_clustering(trade_data['side'])
            }
        
        # Size pattern analysis
        if 'size' in trade_data.columns:
            pattern_analysis['size_patterns'] = {
                'size_trend': self._calculate_size_trend(trade_data['size']),
                'size_volatility': float(trade_data['size'].rolling(window=10).std().mean()),
                'size_clustering': self._detect_size_clustering(trade_data['size'])
            }
        
        # Price pattern analysis
        if 'price' in trade_data.columns:
            pattern_analysis['price_patterns'] = {
                'price_trend': self._calculate_price_trend(trade_data['price']),
                'price_volatility': float(trade_data['price'].rolling(window=10).std().mean()),
                'price_momentum': self._calculate_price_momentum(trade_data['price'])
            }
        
        # Duration pattern analysis
        if 'duration_minutes' in trade_data.columns:
            pattern_analysis['duration_patterns'] = {
                'duration_trend': self._calculate_duration_trend(trade_data['duration_minutes']),
                'duration_volatility': float(trade_data['duration_minutes'].rolling(window=10).std().mean()),
                'duration_clustering': self._detect_duration_clustering(trade_data['duration_minutes'])
            }
        
        self.logger.info("✅ Pattern analysis completed")
        return pattern_analysis
    
    async def _perform_timing_analysis(self, trade_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform trade timing analysis."""
        self.logger.info("⏰ Performing timing analysis...")
        
        timing_analysis = {}
        
        if 'timestamp' in trade_data.columns:
            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(trade_data['timestamp']):
                trade_data['timestamp'] = pd.to_datetime(trade_data['timestamp'])
            
            # Hour analysis
            trade_data['hour'] = trade_data['timestamp'].dt.hour
            timing_analysis['hour_analysis'] = {
                'most_active_hour': int(trade_data['hour'].mode().iloc[0]) if not trade_data['hour'].mode().empty else 0,
                'hour_distribution': trade_data['hour'].value_counts(normalize=True).to_dict(),
                'hour_volatility': float(trade_data['hour'].value_counts().std())
            }
            
            # Day of week analysis
            trade_data['day_of_week'] = trade_data['timestamp'].dt.dayofweek
            timing_analysis['day_analysis'] = {
                'most_active_day': int(trade_data['day_of_week'].mode().iloc[0]) if not trade_data['day_of_week'].mode().empty else 0,
                'day_distribution': trade_data['day_of_week'].value_counts(normalize=True).to_dict(),
                'weekend_activity': float((trade_data['day_of_week'] >= 5).mean())
            }
            
            # Time interval analysis
            if len(trade_data) > 1:
                time_intervals = trade_data['timestamp'].diff().dt.total_seconds() / 60  # minutes
                timing_analysis['interval_analysis'] = {
                    'average_interval': float(time_intervals.mean()),
                    'median_interval': float(time_intervals.median()),
                    'interval_volatility': float(time_intervals.std()),
                    'min_interval': float(time_intervals.min()),
                    'max_interval': float(time_intervals.max())
                }
        
        self.logger.info("✅ Timing analysis completed")
        return timing_analysis
    
    async def _perform_size_analysis(self, trade_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform trade size analysis."""
        self.logger.info("📏 Performing size analysis...")
        
        size_analysis = {}
        
        if 'size' in trade_data.columns:
            size = trade_data['size']
            
            # Size distribution analysis
            size_analysis['distribution'] = {
                'size_skewness': float(stats.skew(size)),
                'size_kurtosis': float(stats.kurtosis(size)),
                'size_entropy': self._calculate_entropy(size),
                'size_gini': self._calculate_gini_coefficient(size)
            }
            
            # Size vs performance analysis
            if 'pnl' in trade_data.columns:
                pnl = trade_data['pnl'].dropna()
                size_aligned = size.loc[pnl.index]
                
                # Correlation between size and PnL
                size_pnl_corr = size_aligned.corr(pnl)
                size_analysis['size_performance'] = {
                    'size_pnl_correlation': float(size_pnl_corr),
                    'large_trade_performance': float(pnl[size_aligned > size_aligned.quantile(0.8)].mean()) if len(pnl[size_aligned > size_aligned.quantile(0.8)]) > 0 else 0.0,
                    'small_trade_performance': float(pnl[size_aligned < size_aligned.quantile(0.2)].mean()) if len(pnl[size_aligned < size_aligned.quantile(0.2)]) > 0 else 0.0
                }
            
            # Size optimization analysis
            size_analysis['optimization'] = {
                'optimal_size_range': self._find_optimal_size_range(trade_data),
                'size_efficiency': self._calculate_size_efficiency(trade_data),
                'size_risk_ratio': self._calculate_size_risk_ratio(trade_data)
            }
        
        self.logger.info("✅ Size analysis completed")
        return size_analysis
    
    async def _perform_correlation_analysis(self, trade_data: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform trade correlation analysis."""
        self.logger.info("🔗 Performing correlation analysis...")
        
        correlation_analysis = {}
        
        # Internal trade correlations
        if 'size' in trade_data.columns and 'price' in trade_data.columns:
            correlation_analysis['internal_correlations'] = {
                'size_price_correlation': float(trade_data['size'].corr(trade_data['price'])),
                'size_duration_correlation': float(trade_data['size'].corr(trade_data.get('duration_minutes', 0))),
                'price_duration_correlation': float(trade_data['price'].corr(trade_data.get('duration_minutes', 0)))
            }
        
        # Market correlation analysis
        if not market_data.empty and 'close' in market_data.columns and 'pnl' in trade_data.columns:
            # Align data by timestamp
            market_returns = market_data['close'].pct_change().dropna()
            trade_pnl = trade_data['pnl'].dropna()
            
            # Find common time periods
            if 'timestamp' in trade_data.columns:
                trade_data['timestamp'] = pd.to_datetime(trade_data['timestamp'])
                market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
                
                # Align by hour
                trade_data['hour'] = trade_data['timestamp'].dt.floor('H')
                market_data['hour'] = market_data['timestamp'].dt.floor('H')
                
                # Group by hour and calculate correlations
                trade_hourly = trade_data.groupby('hour')['pnl'].sum()
                market_hourly = market_data.groupby('hour')['close'].last().pct_change().dropna()
                
                common_hours = trade_hourly.index.intersection(market_hourly.index)
                if len(common_hours) > 0:
                    correlation_analysis['market_correlations'] = {
                        'trade_market_correlation': float(trade_hourly.loc[common_hours].corr(market_hourly.loc[common_hours])),
                        'correlation_stability': self._calculate_correlation_stability(trade_hourly.loc[common_hours], market_hourly.loc[common_hours])
                    }
        
        self.logger.info("✅ Correlation analysis completed")
        return correlation_analysis
    
    def _generate_optimization_insights(
        self, 
        trade_statistics: Dict[str, Any], 
        performance_analysis: Dict[str, Any], 
        pattern_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate trade optimization insights."""
        self.logger.info("💡 Generating optimization insights...")
        
        insights = []
        
        # Performance-based insights
        if 'performance_metrics' in performance_analysis:
            metrics = performance_analysis['performance_metrics']
            
            if metrics['win_rate'] < 0.5:
                insights.append({
                    'category': 'PERFORMANCE',
                    'priority': 'HIGH',
                    'title': 'Low Win Rate',
                    'description': f'Win rate is {metrics["win_rate"]:.2%}, indicating poor trade selection',
                    'recommendation': 'Improve entry/exit criteria and risk management',
                    'impact': 'HIGH'
                })
            
            if metrics['sharpe_ratio'] < 1.0:
                insights.append({
                    'category': 'RISK_ADJUSTED_RETURN',
                    'priority': 'MEDIUM',
                    'title': 'Low Sharpe Ratio',
                    'description': f'Sharpe ratio is {metrics["sharpe_ratio"]:.2f}, indicating poor risk-adjusted returns',
                    'recommendation': 'Optimize position sizing and reduce volatility',
                    'impact': 'MEDIUM'
                })
        
        # Pattern-based insights
        if 'side_patterns' in pattern_analysis:
            side_patterns = pattern_analysis['side_patterns']
            
            if side_patterns['buy_sell_ratio'] > 2.0 or side_patterns['buy_sell_ratio'] < 0.5:
                insights.append({
                    'category': 'TRADE_BALANCE',
                    'priority': 'MEDIUM',
                    'title': 'Imbalanced Buy/Sell Ratio',
                    'description': f'Buy/sell ratio is {side_patterns["buy_sell_ratio"]:.2f}, indicating potential bias',
                    'recommendation': 'Review strategy logic for directional bias',
                    'impact': 'MEDIUM'
                })
        
        # Size-based insights
        if 'size_statistics' in trade_statistics:
            size_stats = trade_statistics['size_statistics']
            
            if size_stats['std_size'] / size_stats['average_size'] > 0.5:
                insights.append({
                    'category': 'POSITION_SIZING',
                    'priority': 'MEDIUM',
                    'title': 'High Size Volatility',
                    'description': f'Size volatility is {size_stats["std_size"]/size_stats["average_size"]:.2%}, indicating inconsistent position sizing',
                    'recommendation': 'Implement more consistent position sizing rules',
                    'impact': 'MEDIUM'
                })
        
        self.logger.info(f"✅ Generated {len(insights)} optimization insights")
        return insights
    
    def _calculate_max_drawdown(self, pnl: pd.Series) -> float:
        """Calculate maximum drawdown from PnL series."""
        if len(pnl) == 0:
            return 0.0
        
        cumulative_pnl = pnl.cumsum()
        peak = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - peak) / peak
        return float(drawdown.min())
    
    def _calculate_consecutive_wins(self, pnl: pd.Series) -> int:
        """Calculate maximum consecutive wins."""
        if len(pnl) == 0:
            return 0
        
        wins = (pnl > 0).astype(int)
        consecutive_wins = 0
        max_consecutive = 0
        
        for win in wins:
            if win:
                consecutive_wins += 1
                max_consecutive = max(max_consecutive, consecutive_wins)
            else:
                consecutive_wins = 0
        
        return max_consecutive
    
    def _calculate_consecutive_losses(self, pnl: pd.Series) -> int:
        """Calculate maximum consecutive losses."""
        if len(pnl) == 0:
            return 0
        
        losses = (pnl < 0).astype(int)
        consecutive_losses = 0
        max_consecutive = 0
        
        for loss in losses:
            if loss:
                consecutive_losses += 1
                max_consecutive = max(max_consecutive, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return max_consecutive
    
    def _calculate_performance_stability(self, pnl: pd.Series) -> float:
        """Calculate performance stability metric."""
        if len(pnl) < 10:
            return 0.0
        
        rolling_mean = pnl.rolling(window=10).mean()
        stability = 1 - (rolling_mean.std() / abs(rolling_mean.mean())) if rolling_mean.mean() != 0 else 0.0
        return float(stability)
    
    def _calculate_side_alternation(self, sides: pd.Series) -> float:
        """Calculate side alternation frequency."""
        if len(sides) < 2:
            return 0.0
        
        alternations = (sides != sides.shift(1)).sum()
        return float(alternations / (len(sides) - 1))
    
    def _detect_side_clustering(self, sides: pd.Series) -> bool:
        """Detect side clustering."""
        if len(sides) < 10:
            return False
        
        # Simple test for clustering
        buy_ratio = (sides == 'BUY').mean()
        expected_alternation = 2 * buy_ratio * (1 - buy_ratio)
        actual_alternation = self._calculate_side_alternation(sides)
        
        return actual_alternation < expected_alternation * 0.8
    
    def _calculate_size_trend(self, sizes: pd.Series) -> str:
        """Calculate size trend."""
        if len(sizes) < 10:
            return "insufficient_data"
        
        x = np.arange(len(sizes))
        slope, _, _, _, _ = stats.linregress(x, sizes)
        
        if slope > 0:
            return "increasing"
        elif slope < 0:
            return "decreasing"
        else:
            return "stable"
    
    def _detect_size_clustering(self, sizes: pd.Series) -> bool:
        """Detect size clustering."""
        if len(sizes) < 10:
            return False
        
        # Simple test for clustering
        size_std = sizes.std()
        rolling_std = sizes.rolling(window=5).std()
        clustering = (rolling_std < size_std * 0.5).mean()
        
        return clustering > 0.3
    
    def _calculate_price_trend(self, prices: pd.Series) -> str:
        """Calculate price trend."""
        if len(prices) < 10:
            return "insufficient_data"
        
        x = np.arange(len(prices))
        slope, _, _, _, _ = stats.linregress(x, prices)
        
        if slope > 0:
            return "increasing"
        elif slope < 0:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_price_momentum(self, prices: pd.Series) -> float:
        """Calculate price momentum."""
        if len(prices) < 5:
            return 0.0
        
        returns = prices.pct_change().dropna()
        momentum = returns.rolling(window=5).mean().iloc[-1] if len(returns) >= 5 else 0.0
        return float(momentum)
    
    def _calculate_duration_trend(self, durations: pd.Series) -> str:
        """Calculate duration trend."""
        if len(durations) < 10:
            return "insufficient_data"
        
        x = np.arange(len(durations))
        slope, _, _, _, _ = stats.linregress(x, durations)
        
        if slope > 0:
            return "increasing"
        elif slope < 0:
            return "decreasing"
        else:
            return "stable"
    
    def _detect_duration_clustering(self, durations: pd.Series) -> bool:
        """Detect duration clustering."""
        if len(durations) < 10:
            return False
        
        # Simple test for clustering
        duration_std = durations.std()
        rolling_std = durations.rolling(window=5).std()
        clustering = (rolling_std < duration_std * 0.5).mean()
        
        return clustering > 0.3
    
    def _calculate_entropy(self, values: pd.Series) -> float:
        """Calculate entropy of values."""
        if len(values) == 0:
            return 0.0
        
        # Discretize values into bins
        bins = pd.cut(values, bins=10, duplicates='drop')
        probabilities = bins.value_counts(normalize=True)
        
        # Calculate entropy
        entropy = -(probabilities * np.log2(probabilities)).sum()
        return float(entropy)
    
    def _calculate_gini_coefficient(self, values: pd.Series) -> float:
        """Calculate Gini coefficient."""
        if len(values) == 0:
            return 0.0
        
        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        
        return float((n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n)
    
    def _find_optimal_size_range(self, trade_data: pd.DataFrame) -> Dict[str, float]:
        """Find optimal size range based on performance."""
        if 'size' not in trade_data.columns or 'pnl' not in trade_data.columns:
            return {}
        
        # Group by size quartiles and calculate performance
        size_quartiles = pd.qcut(trade_data['size'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        performance_by_quartile = trade_data.groupby(size_quartiles)['pnl'].agg(['mean', 'std', 'count'])
        
        # Find best performing quartile
        best_quartile = performance_by_quartile['mean'].idxmax()
        best_performance = performance_by_quartile.loc[best_quartile, 'mean']
        
        return {
            'best_quartile': best_quartile,
            'best_performance': float(best_performance),
            'size_range': 'optimization_needed'
        }
    
    def _calculate_size_efficiency(self, trade_data: pd.DataFrame) -> float:
        """Calculate size efficiency metric."""
        if 'size' not in trade_data.columns or 'pnl' not in trade_data.columns:
            return 0.0
        
        # Calculate return per unit of size
        return_per_size = trade_data['pnl'] / trade_data['size']
        efficiency = return_per_size.mean() / return_per_size.std() if return_per_size.std() > 0 else 0.0
        
        return float(efficiency)
    
    def _calculate_size_risk_ratio(self, trade_data: pd.DataFrame) -> float:
        """Calculate size to risk ratio."""
        if 'size' not in trade_data.columns or 'pnl' not in trade_data.columns:
            return 0.0
        
        # Calculate risk-adjusted size
        pnl_std = trade_data['pnl'].std()
        if pnl_std == 0:
            return 0.0
        
        risk_adjusted_size = trade_data['size'] / pnl_std
        return float(risk_adjusted_size.mean())
    
    def _calculate_correlation_stability(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate correlation stability over time."""
        if len(series1) < 10:
            return 0.0
        
        # Calculate rolling correlation
        rolling_corr = series1.rolling(window=5).corr(series2)
        return float(rolling_corr.std())
    
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
    
    async def _save_results(self, results: TradeAnalysisResults) -> None:
        """Save results to disk."""
        self.logger.info("💾 Saving results...")
        
        # Create output directory
        output_dir = self.data_dir / "backtesting_results" / "trade_analysis"
        ensure_directory(output_dir)
        
        # Save main results
        results_file = output_dir / f"{self.config.symbol}_{self.config.exchange}_trade_analysis_results.json"
        await safe_json_dump(results_file, results.__dict__, indent=2)
        
        # Save trade data
        if not results.trade_data.empty:
            trade_file = output_dir / f"{self.config.symbol}_{self.config.exchange}_trade_data.parquet"
            await self.parquet_utils.save_dataframe(results.trade_data, trade_file)
        
        self.logger.info(f"✅ Results saved to {output_dir}")


# Convenience function for easy integration
async def execute_trade_analysis(
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE", 
    timeframe: str = "1h",
    data_dir: str = "data/training",
    **kwargs
) -> TradeAnalysisResults:
    """
    Convenience function to execute trade analysis.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Data timeframe
        data_dir: Data directory
        **kwargs: Additional configuration parameters
        
    Returns:
        Trade analysis results
    """
    config = TradeAnalysisConfig(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        **kwargs
    )
    
    step = TradeAnalysisStep(config)
    return await step.execute()