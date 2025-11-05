"""
VectorBT Integration for Statsmodels Clustering

This module provides integration between statsmodels regime switching models and VectorBT
for backtesting and portfolio analysis, enabling comprehensive evaluation of regime-based strategies.

Key Features:
- Regime-based backtesting with VectorBT
- Portfolio optimization across regimes
- Performance analysis by regime
- Strategy comparison and validation
- Risk metrics calculation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from dataclasses import dataclass, field
import logging
import time
from pathlib import Path

# Import VectorBT
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None

# Import utilities
try:
    from src.utils.tprint import (
        tprint_info, tprint_success, tprint_warning, tprint_error,
        tprint_structured
    )
except ImportError:
    def tprint_info(msg): print(f'ℹ️  {msg}')
    def tprint_success(msg): print(f'✅ {msg}')
    def tprint_warning(msg): print(f'⚠️  {msg}')
    def tprint_error(msg): print(f'❌ {msg}')
    def tprint_structured(data, level="INFO"):
        for key, value in data.items():
            print(f'🔧 {key}: {value}')


@dataclass
class VectorBTConfig:
    """
    Configuration for VectorBT integration with regime switching models.
    
    Defines how regime switching results should be integrated with
    VectorBT for backtesting and analysis.
    """
    # Backtesting settings
    enable_backtesting: bool = True
    initial_cash: float = 10000.0
    fees: float = 0.001
    slippage: float = 0.0005
    
    # Portfolio settings
    enable_portfolio_optimization: bool = True
    rebalance_frequency: str = '1M'  # Monthly rebalancing
    max_positions: int = 10
    
    # Regime-specific settings
    enable_regime_strategies: bool = True
    regime_specific_weights: bool = True
    regime_transition_signals: bool = True
    
    # Performance analysis
    calculate_regime_metrics: bool = True
    calculate_drawdowns: bool = True
    calculate_rolling_metrics: bool = True
    
    # Risk analysis
    calculate_var: bool = True
    var_confidence: float = 0.05
    calculate_sharpe: bool = True
    risk_free_rate: float = 0.02
    
    # Output settings
    save_results: bool = True
    output_dir: Optional[str] = None
    plot_results: bool = True


@dataclass
class VectorBTResult:
    """
    Result container for VectorBT integration operations.
    
    Contains backtesting results, performance metrics, and regime analysis.
    """
    # Backtesting results
    portfolio_value: Optional[pd.Series] = None
    returns: Optional[pd.Series] = None
    positions: Optional[pd.DataFrame] = None
    trades: Optional[pd.DataFrame] = None
    
    # Performance metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    
    # Regime-specific results
    regime_performance: Optional[Dict[str, Any]] = None
    regime_transitions: Optional[pd.DataFrame] = None
    
    # Risk metrics
    var_5: float = 0.0
    var_1: float = 0.0
    volatility: float = 0.0
    
    # Processing metadata
    processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    # Additional data
    benchmark_returns: Optional[pd.Series] = None
    metrics_summary: Optional[Dict[str, Any]] = None


class VectorBTIntegration:
    """
    Integration between statsmodels regime switching and VectorBT.
    
    Provides comprehensive backtesting and analysis capabilities for
    regime-based trading strategies.
    """
    
    def __init__(self, config: Optional[VectorBTConfig] = None):
        """
        Initialize VectorBT integration.
        
        Args:
            config: Configuration for VectorBT integration
        """
        if not VECTORBT_AVAILABLE:
            tprint_warning("⚠️ VectorBT not available - install with: pip install vectorbt")
            self.available = False
            return
        
        self.config = config or VectorBTConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.available = True
        
        tprint_info("🔧 Initialized VectorBT Integration")
    
    def backtest_regime_strategy(self,
                               prices: pd.DataFrame,
                               regime_labels: np.ndarray,
                               regime_probabilities: Optional[np.ndarray] = None,
                               signals: Optional[pd.DataFrame] = None) -> VectorBTResult:
        """
        Backtest regime-based trading strategy.
        
        Args:
            prices: Price data for assets
            regime_labels: Regime labels for each time period
            regime_probabilities: Optional regime probabilities
            signals: Optional trading signals
            
        Returns:
            VectorBTResult with backtesting results
        """
        if not self.available:
            return VectorBTResult(
                success=False,
                error_message="VectorBT not available"
            )
        
        start_time = time.time()
        result = VectorBTResult()
        
        try:
            tprint_info("🔄 Backtesting regime-based strategy")
            
            # Validate inputs
            self._validate_backtest_inputs(prices, regime_labels, signals)
            
            # Create regime-specific strategies
            if self.config.enable_regime_strategies:
                regime_strategies = self._create_regime_strategies(prices, regime_labels, signals)
            else:
                regime_strategies = self._create_default_strategy(prices, signals)
            
            # Run backtesting
            if self.config.enable_backtesting:
                backtest_result = self._run_vectorbt_backtest(prices, regime_strategies, regime_labels)
                result.portfolio_value = backtest_result['portfolio_value']
                result.returns = backtest_result['returns']
                result.positions = backtest_result['positions']
                result.trades = backtest_result['trades']
            
            # Calculate performance metrics
            if result.returns is not None:
                performance_metrics = self._calculate_performance_metrics(result.returns)
                result.total_return = performance_metrics['total_return']
                result.annual_return = performance_metrics['annual_return']
                result.sharpe_ratio = performance_metrics['sharpe_ratio']
                result.max_drawdown = performance_metrics['max_drawdown']
                result.win_rate = performance_metrics['win_rate']
                result.volatility = performance_metrics['volatility']
            
            # Calculate regime-specific performance
            if self.config.calculate_regime_metrics:
                regime_performance = self._calculate_regime_performance(
                    result.returns, regime_labels, regime_probabilities
                )
                result.regime_performance = regime_performance
            
            # Calculate risk metrics
            if self.config.calculate_var and result.returns is not None:
                risk_metrics = self._calculate_risk_metrics(result.returns)
                result.var_5 = risk_metrics['var_5']
                result.var_1 = risk_metrics['var_1']
            
            # Analyze regime transitions
            if self.config.regime_transition_signals:
                result.regime_transitions = self._analyze_regime_transitions(regime_labels)
            
            # Create benchmark
            result.benchmark_returns = self._create_benchmark(prices)
            
            # Generate metrics summary
            result.metrics_summary = self._create_metrics_summary(result)
            
            result.processing_time = time.time() - start_time
            
            # Save results if requested
            if self.config.save_results:
                self._save_results(result)
            
            tprint_success(f"✅ Regime backtesting completed in {result.processing_time:.2f}s")
            tprint_structured({
                "total_return": f"{result.total_return:.2%}",
                "sharpe_ratio": f"{result.sharpe_ratio:.2f}",
                "max_drawdown": f"{result.max_drawdown:.2%}",
                "win_rate": f"{result.win_rate:.2%}"
            }, level="INFO")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Regime backtesting failed: {e}")
            result.success = False
            result.error_message = str(e)
            result.processing_time = time.time() - start_time
            return result
    
    def optimize_portfolio_by_regime(self,
                                   prices: pd.DataFrame,
                                   regime_labels: np.ndarray,
                                   returns: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Optimize portfolio weights for each regime.
        
        Args:
            prices: Price data for assets
            regime_labels: Regime labels
            returns: Optional returns data
            
        Returns:
            Dictionary with regime-specific optimal weights
        """
        if not self.available:
            return {'error': 'VectorBT not available'}
        
        try:
            tprint_info("🎯 Optimizing portfolio by regime")
            
            # Calculate returns if not provided
            if returns is None:
                returns = prices.pct_change().dropna()
            
            # Group data by regime
            regime_weights = {}
            unique_regimes = np.unique(regime_labels)
            
            for regime in unique_regimes:
                regime_mask = regime_labels == regime
                regime_returns = returns[regime_mask]
                
                if len(regime_returns) > 0:
                    # Optimize weights for this regime
                    weights = self._optimize_regime_weights(regime_returns)
                    regime_weights[f'regime_{regime}'] = weights
            
            # Calculate overall performance
            if regime_weights:
                overall_performance = self._evaluate_regime_weights(
                    returns, regime_labels, regime_weights
                )
                regime_weights['overall_performance'] = overall_performance
            
            tprint_success(f"✅ Portfolio optimization completed for {len(regime_weights)} regimes")
            return regime_weights
            
        except Exception as e:
            tprint_error(f"❌ Portfolio optimization failed: {e}")
            return {'error': str(e)}
    
    def compare_strategies(self,
                         strategies: Dict[str, pd.DataFrame],
                         prices: pd.DataFrame,
                         regime_labels: np.ndarray) -> Dict[str, Any]:
        """
        Compare multiple strategies across regimes.
        
        Args:
            strategies: Dictionary of strategy returns
            prices: Price data
            regime_labels: Regime labels
            
        Returns:
            Comparison results and analysis
        """
        if not self.available:
            return {'error': 'VectorBT not available'}
        
        try:
            tprint_info(f"📊 Comparing {len(strategies)} strategies")
            
            comparison_results = {}
            
            # Backtest each strategy
            for strategy_name, strategy_returns in strategies.items():
                result = self.backtest_regime_strategy(
                    prices, regime_labels, signals=strategy_returns
                )
                comparison_results[strategy_name] = result
            
            # Create comparison table
            comparison_table = self._create_comparison_table(comparison_results)
            
            # Analyze strategy performance by regime
            regime_comparison = self._compare_strategies_by_regime(
                comparison_results, regime_labels
            )
            
            # Determine best strategy
            best_strategy = self._determine_best_strategy(comparison_results)
            
            results = {
                'strategy_results': comparison_results,
                'comparison_table': comparison_table,
                'regime_comparison': regime_comparison,
                'best_strategy': best_strategy
            }
            
            tprint_success(f"✅ Strategy comparison completed")
            return results
            
        except Exception as e:
            tprint_error(f"❌ Strategy comparison failed: {e}")
            return {'error': str(e)}
    
    def _validate_backtest_inputs(self, prices, regime_labels, signals):
        """Validate inputs for backtesting."""
        if not isinstance(prices, pd.DataFrame):
            raise ValueError("prices must be a pandas DataFrame")
        
        if len(prices) != len(regime_labels):
            raise ValueError("prices and regime_labels must have same length")
        
        if signals is not None and len(signals) != len(prices):
            raise ValueError("signals and prices must have same length")
    
    def _create_regime_strategies(self, prices, regime_labels, signals):
        """Create regime-specific trading strategies."""
        strategies = {}
        
        # Create base strategy from signals if provided
        if signals is not None:
            base_strategy = signals
        else:
            # Create simple momentum strategy
            returns = prices.pct_change()
            base_strategy = (returns > 0).astype(int)
        
        # Apply regime-specific modifications
        if self.config.regime_specific_weights:
            for regime in np.unique(regime_labels):
                regime_mask = regime_labels == regime
                regime_strategy = base_strategy.copy()
                
                # Apply regime-specific logic
                if regime == 0:  # Bull regime
                    regime_strategy[regime_mask] *= 1.2  # Increase exposure
                elif regime == 1:  # Bear regime
                    regime_strategy[regime_mask] *= 0.8  # Decrease exposure
                
                strategies[f'regime_{regime}'] = regime_strategy
        
        return strategies
    
    def _create_default_strategy(self, prices, signals):
        """Create default trading strategy."""
        if signals is not None:
            return signals
        
        # Simple buy-and-hold strategy
        return pd.DataFrame(1, index=prices.index, columns=prices.columns)
    
    def _run_vectorbt_backtest(self, prices, strategies, regime_labels):
        """Run VectorBT backtesting."""
        # Combine strategies based on current regime
        combined_strategy = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        
        for regime in np.unique(regime_labels):
            regime_mask = regime_labels == regime
            regime_strategy = strategies.get(f'regime_{regime}')
            
            if regime_strategy is not None:
                combined_strategy[regime_mask] = regime_strategy[regime_mask]
        
        # Run VectorBT backtesting
        portfolio = vbt.Portfolio.from_signals(
            prices=prices,
            entries=(combined_strategy > 0),
            exits=(combined_strategy <= 0),
            init_cash=self.config.initial_cash,
            fees=self.config.fees,
            slippage=self.config.slippage
        )
        
        return {
            'portfolio_value': portfolio.value(),
            'returns': portfolio.returns(),
            'positions': portfolio.positions(),
            'trades': portfolio.trades()
        }
    
    def _calculate_performance_metrics(self, returns):
        """Calculate performance metrics."""
        metrics = {}
        
        # Total return
        metrics['total_return'] = (1 + returns).prod() - 1
        
        # Annual return
        trading_days = 252
        metrics['annual_return'] = (1 + returns).prod() ** (trading_days / len(returns)) - 1
        
        # Sharpe ratio
        if self.config.calculate_sharpe:
            excess_returns = returns - self.config.risk_free_rate / trading_days
            metrics['sharpe_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(trading_days)
        
        # Maximum drawdown
        if self.config.calculate_drawdowns:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            metrics['max_drawdown'] = drawdown.min()
        
        # Win rate
        metrics['win_rate'] = (returns > 0).mean()
        
        # Volatility
        metrics['volatility'] = returns.std() * np.sqrt(trading_days)
        
        return metrics
    
    def _calculate_regime_performance(self, returns, regime_labels, regime_probabilities):
        """Calculate performance metrics by regime."""
        regime_performance = {}
        
        for regime in np.unique(regime_labels):
            regime_mask = regime_labels == regime
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) > 0:
                metrics = self._calculate_performance_metrics(regime_returns)
                regime_performance[f'regime_{regime}'] = metrics
        
        return regime_performance
    
    def _calculate_risk_metrics(self, returns):
        """Calculate risk metrics."""
        metrics = {}
        
        # Value at Risk
        if self.config.calculate_var:
            metrics['var_5'] = returns.quantile(self.config.var_confidence)
            metrics['var_1'] = returns.quantile(0.01)
        
        return metrics
    
    def _analyze_regime_transitions(self, regime_labels):
        """Analyze regime transitions."""
        transitions = []
        
        for i in range(1, len(regime_labels)):
            if regime_labels[i] != regime_labels[i-1]:
                transitions.append({
                    'time': i,
                    'from_regime': regime_labels[i-1],
                    'to_regime': regime_labels[i]
                })
        
        return pd.DataFrame(transitions)
    
    def _create_benchmark(self, prices):
        """Create benchmark returns."""
        # Equal-weight portfolio
        returns = prices.pct_change().dropna()
        equal_weights = np.ones(len(prices.columns)) / len(prices.columns)
        benchmark_returns = (returns * equal_weights).sum(axis=1)
        
        return benchmark_returns
    
    def _create_metrics_summary(self, result):
        """Create comprehensive metrics summary."""
        summary = {
            'performance': {
                'total_return': result.total_return,
                'annual_return': result.annual_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'volatility': result.volatility
            },
            'risk': {
                'var_5': result.var_5,
                'var_1': result.var_1
            }
        }
        
        if result.regime_performance:
            summary['regime_performance'] = result.regime_performance
        
        return summary
    
    def _optimize_regime_weights(self, returns):
        """Optimize portfolio weights for a specific regime."""
        # Simple mean-variance optimization
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Equal risk contribution portfolio (simplified)
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(len(mean_returns))
        
        weights = inv_cov @ ones
        weights = weights / weights.sum()
        
        return pd.Series(weights, index=returns.columns)
    
    def _evaluate_regime_weights(self, returns, regime_labels, regime_weights):
        """Evaluate performance of regime-specific weights."""
        portfolio_returns = pd.Series(0, index=returns.index)
        
        for regime in np.unique(regime_labels):
            regime_mask = regime_labels == regime
            weights = regime_weights.get(f'regime_{regime}')
            
            if weights is not None:
                regime_returns = (returns[regime_mask] * weights).sum(axis=1)
                portfolio_returns[regime_mask] = regime_returns
        
        return self._calculate_performance_metrics(portfolio_returns)
    
    def _create_comparison_table(self, comparison_results):
        """Create comparison table for strategies."""
        table_data = []
        
        for strategy_name, result in comparison_results.items():
            table_data.append({
                'strategy': strategy_name,
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate
            })
        
        return pd.DataFrame(table_data).set_index('strategy')
    
    def _compare_strategies_by_regime(self, comparison_results, regime_labels):
        """Compare strategy performance by regime."""
        regime_comparison = {}
        
        for regime in np.unique(regime_labels):
            regime_data = {}
            
            for strategy_name, result in comparison_results.items():
                if result.regime_performance and f'regime_{regime}' in result.regime_performance:
                    regime_data[strategy_name] = result.regime_performance[f'regime_{regime}']
            
            regime_comparison[f'regime_{regime}'] = regime_data
        
        return regime_comparison
    
    def _determine_best_strategy(self, comparison_results):
        """Determine best strategy based on multiple metrics."""
        best_metrics = {}
        
        # Best by total return
        best_return = max(comparison_results.items(), key=lambda x: x[1].total_return)
        best_metrics['total_return'] = best_return[0]
        
        # Best by Sharpe ratio
        best_sharpe = max(comparison_results.items(), key=lambda x: x[1].sharpe_ratio)
        best_metrics['sharpe_ratio'] = best_sharpe[0]
        
        # Best by risk-adjusted return (Sharpe / drawdown)
        best_risk_adj = max(
            comparison_results.items(),
            key=lambda x: x[1].sharpe_ratio / abs(x[1].max_drawdown) if x[1].max_drawdown != 0 else 0
        )
        best_metrics['risk_adjusted'] = best_risk_adj[0]
        
        return best_metrics
    
    def _save_results(self, result):
        """Save VectorBT results to files."""
        if not self.config.output_dir:
            return
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save portfolio value
        if result.portfolio_value is not None:
            result.portfolio_value.to_csv(output_dir / 'portfolio_value.csv')
        
        # Save returns
        if result.returns is not None:
            result.returns.to_csv(output_dir / 'returns.csv')
        
        # Save metrics summary
        if result.metrics_summary:
            import json
            with open(output_dir / 'metrics_summary.json', 'w') as f:
                json.dump(result.metrics_summary, f, indent=2, default=str)
        
        tprint_info(f"💾 VectorBT results saved to {output_dir}")


# Convenience functions for VectorBT integration
def create_vectorbt_integration(config: Optional[VectorBTConfig] = None) -> VectorBTIntegration:
    """
    Create VectorBT integration with default configuration.
    
    Args:
        config: Optional VectorBT configuration
        
    Returns:
        VectorBTIntegration instance
    """
    return VectorBTIntegration(config)


def backtest_regime_strategy(prices: pd.DataFrame,
                           regime_labels: np.ndarray,
                           initial_cash: float = 10000.0) -> VectorBTResult:
    """
    Convenience function to backtest regime strategy.
    
    Args:
        prices: Price data for assets
        regime_labels: Regime labels
        initial_cash: Initial cash for backtesting
        
    Returns:
        VectorBTResult with backtesting results
    """
    config = VectorBTConfig(initial_cash=initial_cash)
    integration = create_vectorbt_integration(config)
    return integration.backtest_regime_strategy(prices, regime_labels)