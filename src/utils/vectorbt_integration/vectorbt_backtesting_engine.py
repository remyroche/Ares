"""
VectorBT Backtesting Engine

High-performance backtesting engine using VectorBT for ultra-fast vectorized operations.
This engine provides significant performance improvements over custom implementations.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from pathlib import Path
import warnings

from src.utils.logger import get_logger
from src.utils.math_validation import validate_finite, validate_positive, safe_divide

logger = get_logger('VectorBTBacktestingEngine')


class VectorBTMode(Enum):
    """VectorBT execution modes."""
    FAST = "fast"  # Maximum speed, minimal features
    BALANCED = "balanced"  # Good speed with essential features
    COMPREHENSIVE = "comprehensive"  # Full features with detailed analysis


@dataclass
class VectorBTConfig:
    """Configuration for VectorBT backtesting engine."""
    # Core settings
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    
    # VectorBT specific settings
    mode: VectorBTMode = VectorBTMode.BALANCED
    use_gpu: bool = False  # VectorBT GPU support
    chunked: bool = True  # Use chunked processing for large datasets
    chunk_size: int = 10000
    
    # Portfolio settings
    max_position_size: float = 0.1
    min_position_size: float = 0.01
    rebalance_frequency: str = 'daily'
    
    # Performance settings
    enable_progress_bar: bool = True
    enable_parallel: bool = True
    n_jobs: int = -1  # Use all available cores
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    
    # Advanced features
    enable_shorting: bool = True
    enable_fractional_shares: bool = True
    enable_dynamic_sizing: bool = True
    
    # Output settings
    save_results: bool = True
    output_dir: str = "vectorbt_results"
    generate_plots: bool = True


@dataclass
class VectorBTResults:
    """Results from VectorBT backtesting."""
    portfolio: vbt.Portfolio
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    trade_analysis: Dict[str, Any]
    execution_time: float
    memory_usage: float
    
    # Additional analysis
    drawdown_analysis: Dict[str, Any] = field(default_factory=dict)
    returns_analysis: Dict[str, Any] = field(default_factory=dict)
    factor_analysis: Dict[str, Any] = field(default_factory=dict)


class VectorBTBacktestingEngine:
    """
    High-performance backtesting engine using VectorBT.
    
    This engine leverages VectorBT's ultra-fast vectorized operations to provide
    significant performance improvements over custom implementations while maintaining
    compatibility with existing research frameworks.
    """
    
    def __init__(self, config: Optional[VectorBTConfig] = None):
        """Initialize VectorBT backtesting engine."""
        self.config = config or VectorBTConfig()
        self.logger = logger.getChild('VectorBTEngine')
        
        # Configure VectorBT settings
        self._configure_vectorbt()
        
        # Performance tracking
        self.performance_stats = {
            'total_simulations': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0,
            'memory_peak': 0.0
        }
        
        self.logger.info("✅ VectorBT Backtesting Engine initialized")
        self.logger.info(f"📊 Mode: {self.config.mode.value}")
        self.logger.info(f"📊 GPU enabled: {self.config.use_gpu}")
        self.logger.info(f"📊 Chunked processing: {self.config.chunked}")
    
    def _configure_vectorbt(self):
        """Configure VectorBT global settings."""
        try:
            # Set VectorBT settings
            vbt.settings.set_theme("dark")
            vbt.settings['array_wrapper']['freq_precision'] = 's'
            
            # Configure chunked processing
            if self.config.chunked:
                vbt.settings['array_wrapper']['chunked'] = True
                vbt.settings['array_wrapper']['chunk_size'] = self.config.chunk_size
            
            # Configure parallel processing
            if self.config.enable_parallel:
                vbt.settings['array_wrapper']['n_jobs'] = self.config.n_jobs
            
            self.logger.info("✅ VectorBT configured successfully")
            
        except Exception as e:
            self.logger.warning(f"⚠️ VectorBT configuration warning: {e}")
    
    def run_backtest(self, 
                    prices: Union[pd.DataFrame, pd.Series, np.ndarray],
                    signals: Union[pd.DataFrame, pd.Series, np.ndarray],
                    entries: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
                    exits: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
                    **kwargs) -> VectorBTResults:
        """
        Run VectorBT backtest simulation.
        
        Args:
            prices: Asset prices (OHLCV data)
            signals: Trading signals (-1, 0, 1 for short, neutral, long)
            entries: Entry signals (optional, derived from signals if not provided)
            exits: Exit signals (optional, derived from signals if not provided)
            **kwargs: Additional parameters
            
        Returns:
            VectorBTResults object with comprehensive analysis
        """
        start_time = time.perf_counter()
        self.logger.info("🚀 Starting VectorBT backtest simulation")
        
        try:
            # Validate and prepare inputs
            prices, signals, entries, exits = self._prepare_inputs(prices, signals, entries, exits)
            
            # Create portfolio using VectorBT
            portfolio = self._create_portfolio(prices, signals, entries, exits, **kwargs)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(portfolio)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(portfolio)
            
            # Analyze trades
            trade_analysis = self._analyze_trades(portfolio)
            
            # Additional analysis based on mode
            if self.config.mode in [VectorBTMode.BALANCED, VectorBTMode.COMPREHENSIVE]:
                drawdown_analysis = self._analyze_drawdowns(portfolio)
                returns_analysis = self._analyze_returns(portfolio)
            else:
                drawdown_analysis = {}
                returns_analysis = {}
            
            # Factor analysis for comprehensive mode
            if self.config.mode == VectorBTMode.COMPREHENSIVE:
                factor_analysis = self._analyze_factors(portfolio, prices)
            else:
                factor_analysis = {}
            
            # Calculate execution metrics
            execution_time = time.perf_counter() - start_time
            memory_usage = self._get_memory_usage()
            
            # Update performance stats
            self._update_performance_stats(execution_time)
            
            # Create results object
            results = VectorBTResults(
                portfolio=portfolio,
                performance_metrics=performance_metrics,
                risk_metrics=risk_metrics,
                trade_analysis=trade_analysis,
                execution_time=execution_time,
                memory_usage=memory_usage,
                drawdown_analysis=drawdown_analysis,
                returns_analysis=returns_analysis,
                factor_analysis=factor_analysis
            )
            
            # Save results if configured
            if self.config.save_results:
                self._save_results(results)
            
            self.logger.info(f"✅ VectorBT backtest completed in {execution_time:.3f}s")
            self.logger.info(f"📊 Final portfolio value: ${portfolio.value().iloc[-1]:,.2f}")
            self.logger.info(f"📊 Total return: {performance_metrics.get('total_return', 0)*100:.2f}%")
            self.logger.info(f"📊 Sharpe ratio: {performance_metrics.get('sharpe_ratio', 0):.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ VectorBT backtest failed: {e}")
            raise
    
    def _prepare_inputs(self, prices, signals, entries, exits):
        """Prepare and validate input data."""
        # Convert to pandas if needed
        if isinstance(prices, np.ndarray):
            prices = pd.DataFrame(prices)
        if isinstance(signals, np.ndarray):
            signals = pd.DataFrame(signals)
        
        # Ensure prices is OHLCV format
        if isinstance(prices, pd.DataFrame):
            if 'close' in prices.columns:
                prices = prices['close']
            elif prices.shape[1] == 1:
                prices = prices.iloc[:, 0]
        
        # Prepare signals
        if entries is None:
            entries = (signals > 0).astype(int)
        if exits is None:
            exits = (signals < 0).astype(int)
        
        # Validate data
        if len(prices) != len(signals):
            raise ValueError(f"Prices and signals length mismatch: {len(prices)} vs {len(signals)}")
        
        return prices, signals, entries, exits
    
    def _create_portfolio(self, prices, signals, entries, exits, **kwargs):
        """Create VectorBT portfolio from signals."""
        try:
            # Configure portfolio parameters
            portfolio_kwargs = {
                'init_cash': self.config.initial_capital,
                'fees': self.config.commission_rate,
                'slippage': self.config.slippage_rate,
                'freq': self.config.rebalance_frequency,
                'seed': kwargs.get('seed', 42)
            }
            
            # Add risk management if configured
            if self.config.stop_loss:
                portfolio_kwargs['stop_loss'] = self.config.stop_loss
            if self.config.take_profit:
                portfolio_kwargs['take_profit'] = self.config.take_profit
            if self.config.trailing_stop:
                portfolio_kwargs['trailing_stop'] = self.config.trailing_stop
            
            # Create portfolio using VectorBT
            if self.config.mode == VectorBTMode.FAST:
                # Fast mode: use simple signal-based portfolio
                portfolio = vbt.Portfolio.from_signals(
                    prices, entries, exits, **portfolio_kwargs
                )
            else:
                # Balanced/Comprehensive mode: use advanced portfolio creation
                portfolio = vbt.Portfolio.from_signals(
                    prices, entries, exits, 
                    size=kwargs.get('size', None),
                    size_type=kwargs.get('size_type', 'amount'),
                    **portfolio_kwargs
                )
            
            return portfolio
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create VectorBT portfolio: {e}")
            raise
    
    def _calculate_performance_metrics(self, portfolio: vbt.Portfolio) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        try:
            # Basic metrics
            total_return = portfolio.total_return()
            annualized_return = portfolio.annualized_return()
            volatility = portfolio.annualized_volatility()
            sharpe_ratio = portfolio.sharpe_ratio()
            
            # Additional metrics
            max_drawdown = portfolio.max_drawdown()
            calmar_ratio = portfolio.calmar_ratio()
            sortino_ratio = portfolio.sortino_ratio()
            
            # Trade metrics
            total_trades = portfolio.orders.count()
            win_rate = portfolio.trades.win_rate()
            profit_factor = portfolio.trades.profit_factor()
            
            # Validate metrics
            metrics = {
                'total_return': float(validate_finite(total_return, 'total_return')),
                'annualized_return': float(validate_finite(annualized_return, 'annualized_return')),
                'volatility': float(validate_finite(volatility, 'volatility')),
                'sharpe_ratio': float(validate_finite(sharpe_ratio, 'sharpe_ratio')),
                'max_drawdown': float(validate_finite(max_drawdown, 'max_drawdown')),
                'calmar_ratio': float(validate_finite(calmar_ratio, 'calmar_ratio')),
                'sortino_ratio': float(validate_finite(sortino_ratio, 'sortino_ratio')),
                'total_trades': int(total_trades) if not pd.isna(total_trades) else 0,
                'win_rate': float(validate_finite(win_rate, 'win_rate')),
                'profit_factor': float(validate_finite(profit_factor, 'profit_factor'))
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate performance metrics: {e}")
            return {}
    
    def _calculate_risk_metrics(self, portfolio: vbt.Portfolio) -> Dict[str, float]:
        """Calculate risk metrics."""
        try:
            # Value at Risk
            var_95 = portfolio.value_at_risk(0.05)
            var_99 = portfolio.value_at_risk(0.01)
            
            # Expected Shortfall
            es_95 = portfolio.expected_shortfall(0.05)
            es_99 = portfolio.expected_shortfall(0.01)
            
            # Beta (if benchmark provided)
            beta = portfolio.beta() if hasattr(portfolio, 'beta') else 0.0
            
            # Information ratio
            information_ratio = portfolio.information_ratio() if hasattr(portfolio, 'information_ratio') else 0.0
            
            risk_metrics = {
                'var_95': float(validate_finite(var_95, 'var_95')),
                'var_99': float(validate_finite(var_99, 'var_99')),
                'es_95': float(validate_finite(es_95, 'es_95')),
                'es_99': float(validate_finite(es_99, 'es_99')),
                'beta': float(validate_finite(beta, 'beta')),
                'information_ratio': float(validate_finite(information_ratio, 'information_ratio'))
            }
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate risk metrics: {e}")
            return {}
    
    def _analyze_trades(self, portfolio: vbt.Portfolio) -> Dict[str, Any]:
        """Analyze trading activity."""
        try:
            trades = portfolio.trades
            
            if len(trades.records_readable) == 0:
                return {'total_trades': 0, 'analysis': 'No trades executed'}
            
            # Trade statistics
            trade_stats = {
                'total_trades': len(trades.records_readable),
                'winning_trades': len(trades.winning.records_readable),
                'losing_trades': len(trades.losing.records_readable),
                'avg_trade_duration': trades.duration.mean(),
                'avg_trade_return': trades.returns.mean(),
                'max_trade_return': trades.returns.max(),
                'min_trade_return': trades.returns.min(),
                'avg_win': trades.winning.returns.mean() if len(trades.winning.records_readable) > 0 else 0,
                'avg_loss': trades.losing.returns.mean() if len(trades.losing.records_readable) > 0 else 0
            }
            
            return trade_stats
            
        except Exception as e:
            self.logger.error(f"❌ Failed to analyze trades: {e}")
            return {}
    
    def _analyze_drawdowns(self, portfolio: vbt.Portfolio) -> Dict[str, Any]:
        """Analyze drawdown characteristics."""
        try:
            drawdowns = portfolio.drawdowns
            
            if len(drawdowns.records_readable) == 0:
                return {'max_drawdown': 0, 'analysis': 'No drawdowns recorded'}
            
            drawdown_stats = {
                'max_drawdown': drawdowns.max_drawdown(),
                'avg_drawdown': drawdowns.max_drawdown.mean(),
                'max_drawdown_duration': drawdowns.duration.max(),
                'avg_drawdown_duration': drawdowns.duration.mean(),
                'drawdown_count': len(drawdowns.records_readable)
            }
            
            return drawdown_stats
            
        except Exception as e:
            self.logger.error(f"❌ Failed to analyze drawdowns: {e}")
            return {}
    
    def _analyze_returns(self, portfolio: vbt.Portfolio) -> Dict[str, Any]:
        """Analyze return characteristics."""
        try:
            returns = portfolio.returns()
            
            if len(returns) == 0:
                return {'analysis': 'No returns data'}
            
            returns_stats = {
                'mean_return': returns.mean(),
                'std_return': returns.std(),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'positive_returns_pct': (returns > 0).mean(),
                'negative_returns_pct': (returns < 0).mean(),
                'zero_returns_pct': (returns == 0).mean()
            }
            
            return returns_stats
            
        except Exception as e:
            self.logger.error(f"❌ Failed to analyze returns: {e}")
            return {}
    
    def _analyze_factors(self, portfolio: vbt.Portfolio, prices) -> Dict[str, Any]:
        """Analyze factor exposures and performance attribution."""
        try:
            # This would require additional factor data
            # For now, return basic factor analysis
            factor_analysis = {
                'market_beta': portfolio.beta() if hasattr(portfolio, 'beta') else 0.0,
                'momentum_exposure': 0.0,  # Would need momentum factor data
                'value_exposure': 0.0,     # Would need value factor data
                'size_exposure': 0.0,      # Would need size factor data
                'analysis_note': 'Factor analysis requires additional factor data'
            }
            
            return factor_analysis
            
        except Exception as e:
            self.logger.error(f"❌ Failed to analyze factors: {e}")
            return {}
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    def _update_performance_stats(self, execution_time: float):
        """Update performance statistics."""
        self.performance_stats['total_simulations'] += 1
        self.performance_stats['total_execution_time'] += execution_time
        self.performance_stats['average_execution_time'] = (
            self.performance_stats['total_execution_time'] / 
            self.performance_stats['total_simulations']
        )
    
    def _save_results(self, results: VectorBTResults):
        """Save results to disk."""
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save portfolio data
            portfolio_path = output_dir / "portfolio.pkl"
            results.portfolio.save(portfolio_path)
            
            # Save metrics
            metrics_path = output_dir / "metrics.json"
            import json
            with open(metrics_path, 'w') as f:
                json.dump({
                    'performance_metrics': results.performance_metrics,
                    'risk_metrics': results.risk_metrics,
                    'trade_analysis': results.trade_analysis,
                    'execution_time': results.execution_time,
                    'memory_usage': results.memory_usage
                }, f, indent=2)
            
            self.logger.info(f"💾 Results saved to {output_dir}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save results: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def benchmark_against_custom(self, prices, signals, custom_engine=None):
        """Benchmark VectorBT against custom backtesting engine."""
        self.logger.info("🔬 Benchmarking VectorBT against custom engine")
        
        # Run VectorBT backtest
        start_time = time.perf_counter()
        vectorbt_results = self.run_backtest(prices, signals)
        vectorbt_time = time.perf_counter() - start_time
        
        # Run custom engine if provided
        if custom_engine:
            start_time = time.perf_counter()
            custom_results = custom_engine.run_backtest(prices, signals)
            custom_time = time.perf_counter() - start_time
            
            speedup = custom_time / vectorbt_time if vectorbt_time > 0 else float('inf')
            
            self.logger.info(f"📊 VectorBT time: {vectorbt_time:.3f}s")
            self.logger.info(f"📊 Custom engine time: {custom_time:.3f}s")
            self.logger.info(f"📊 Speedup: {speedup:.2f}x")
            
            return {
                'vectorbt_time': vectorbt_time,
                'custom_time': custom_time,
                'speedup': speedup,
                'vectorbt_results': vectorbt_results,
                'custom_results': custom_results
            }
        
        return {'vectorbt_time': vectorbt_time, 'vectorbt_results': vectorbt_results}


# Convenience functions
def run_vectorbt_backtest(prices, signals, config=None, **kwargs) -> VectorBTResults:
    """Convenience function to run VectorBT backtest."""
    engine = VectorBTBacktestingEngine(config)
    return engine.run_backtest(prices, signals, **kwargs)


def benchmark_vectorbt_vs_custom(prices, signals, custom_engine, config=None):
    """Convenience function to benchmark VectorBT vs custom engine."""
    engine = VectorBTBacktestingEngine(config)
    return engine.benchmark_against_custom(prices, signals, custom_engine)