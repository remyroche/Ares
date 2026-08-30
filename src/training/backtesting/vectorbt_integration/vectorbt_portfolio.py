"""
VectorBT Portfolio Simulation

Enhanced portfolio simulation using VectorBT for improved performance and functionality.
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import time
from dataclasses import dataclass

from .vectorbt_base import VectorBTBase, VectorBTError
from .vectorbt_config import VectorBTConfig, PortfolioMode

logger = logging.getLogger(__name__)

@dataclass
class PortfolioResult:
    """Result from portfolio simulation."""
    portfolio: Any  # VectorBT Portfolio object
    returns: pd.Series
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, Any]
    execution_time: float

class VectorBTPortfolio(VectorBTBase):
    """
    VectorBT Portfolio Simulation
    
    Provides high-performance portfolio simulation using VectorBT.
    """
    
    def __init__(self, config: VectorBTConfig):
        """Initialize VectorBT portfolio simulator."""
        super().__init__(config)
        self.portfolio = None
        self.last_result = None
        
        self.logger.info("VectorBT Portfolio initialized")
    
    def simulate_portfolio(self, 
                          data: pd.DataFrame,
                          entries: Optional[pd.Series] = None,
                          exits: Optional[pd.Series] = None,
                          **kwargs) -> PortfolioResult:
        """Simulate portfolio with given data and signals."""
        start_time = time.time()
        
        try:
            # Validate and prepare data
            data = self.validate_data(data)
            
            # Prepare signals
            entries, exits = self.prepare_signals(data, entries, exits)
            
            # Create portfolio based on mode
            if self.config.portfolio_mode == PortfolioMode.SIMPLE:
                portfolio = self._create_simple_portfolio(data, entries, exits, **kwargs)
            elif self.config.portfolio_mode == PortfolioMode.ADVANCED:
                portfolio = self._create_advanced_portfolio(data, entries, exits, **kwargs)
            else:
                portfolio = self._create_custom_portfolio(data, entries, exits, **kwargs)
            
            # Calculate metrics
            metrics = self._calculate_portfolio_metrics(portfolio)
            
            # Extract results
            returns = portfolio.returns()
            equity_curve = portfolio.value()
            trades = portfolio.trades.records_readable
            
            execution_time = time.time() - start_time
            
            # Create result
            result = PortfolioResult(
                portfolio=portfolio,
                returns=returns,
                equity_curve=equity_curve,
                trades=trades,
                metrics=metrics,
                execution_time=execution_time
            )
            
            # Store for later use
            self.portfolio = portfolio
            self.last_result = result
            
            self.log_performance("simulate_portfolio", execution_time)
            self.logger.info(f"Portfolio simulation completed in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Portfolio simulation failed: {e}")
            raise VectorBTError(f"Portfolio simulation failed: {e}")
    
    def _create_simple_portfolio(self, 
                                data: pd.DataFrame,
                                entries: pd.Series,
                                exits: pd.Series,
                                **kwargs) -> Any:
        """Create simple portfolio simulation."""
        try:
            portfolio = vbt.Portfolio.from_signals(
                close=data['close'],
                entries=entries,
                exits=exits,
                init_cash=self.config.init_cash,
                fees=self.config.fees,
                slippage=self.config.slippage,
                freq=self.config.freq,
                **kwargs
            )
            
            self.logger.debug("Simple portfolio created")
            return portfolio
            
        except Exception as e:
            self.logger.error(f"Simple portfolio creation failed: {e}")
            raise
    
    def _create_advanced_portfolio(self, 
                                  data: pd.DataFrame,
                                  entries: pd.Series,
                                  exits: pd.Series,
                                  **kwargs) -> Any:
        """Create advanced portfolio simulation with additional features."""
        try:
            # Prepare additional parameters
            portfolio_kwargs = {
                'init_cash': self.config.init_cash,
                'fees': self.config.fees,
                'slippage': self.config.slippage,
                'freq': self.config.freq,
                'max_position_size': self.config.max_position_size,
                **kwargs
            }
            
            # Add stop loss and take profit if configured
            if self.config.stop_loss is not None:
                portfolio_kwargs['stop_loss'] = self.config.stop_loss
            
            if self.config.take_profit is not None:
                portfolio_kwargs['take_profit'] = self.config.take_profit
            
            portfolio = vbt.Portfolio.from_signals(
                close=data['close'],
                entries=entries,
                exits=exits,
                **portfolio_kwargs
            )
            
            self.logger.debug("Advanced portfolio created")
            return portfolio
            
        except Exception as e:
            self.logger.error(f"Advanced portfolio creation failed: {e}")
            raise
    
    def _create_custom_portfolio(self, 
                                data: pd.DataFrame,
                                entries: pd.Series,
                                exits: pd.Series,
                                **kwargs) -> Any:
        """Create custom portfolio simulation."""
        try:
            # Use custom parameters from config
            custom_params = self.config.custom_params.copy()
            custom_params.update(kwargs)
            
            portfolio = vbt.Portfolio.from_signals(
                close=data['close'],
                entries=entries,
                exits=exits,
                init_cash=self.config.init_cash,
                fees=self.config.fees,
                slippage=self.config.slippage,
                freq=self.config.freq,
                **custom_params
            )
            
            self.logger.debug("Custom portfolio created")
            return portfolio
            
        except Exception as e:
            self.logger.error(f"Custom portfolio creation failed: {e}")
            raise
    
    def _calculate_portfolio_metrics(self, portfolio: Any) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics."""
        try:
            # Get basic stats
            stats = portfolio.stats()
            
            # Calculate additional metrics
            returns = portfolio.returns()
            equity_curve = portfolio.value()
            
            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(returns, equity_curve)
            
            # Performance metrics
            performance_metrics = self._calculate_performance_metrics(returns, equity_curve)
            
            # Trade metrics
            trade_metrics = self._calculate_trade_metrics(portfolio)
            
            # Combine all metrics
            metrics = {
                'basic_stats': stats,
                'risk_metrics': risk_metrics,
                'performance_metrics': performance_metrics,
                'trade_metrics': trade_metrics
            }
            
            self.logger.debug("Portfolio metrics calculated")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Portfolio metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_risk_metrics(self, returns: pd.Series, equity_curve: pd.Series) -> Dict[str, Any]:
        """Calculate risk metrics."""
        try:
            # Value at Risk (VaR)
            var_95 = returns.quantile(0.05)
            var_99 = returns.quantile(0.01)
            
            # Expected Shortfall (CVaR)
            cvar_95 = returns[returns <= var_95].mean()
            cvar_99 = returns[returns <= var_99].mean()
            
            # Maximum Drawdown
            peak = equity_curve.cummax()
            drawdown = (equity_curve - peak) / peak
            max_drawdown = drawdown.min()
            
            # Downside deviation
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'max_drawdown': max_drawdown,
                'downside_deviation': downside_deviation,
                'volatility': returns.std() * np.sqrt(252),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis()
            }
            
        except Exception as e:
            self.logger.error(f"Risk metrics calculation failed: {e}")
            return {}
    
    def _calculate_performance_metrics(self, returns: pd.Series, equity_curve: pd.Series) -> Dict[str, Any]:
        """Calculate performance metrics."""
        try:
            # Basic returns
            total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            
            # Risk-adjusted returns
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
            
            # Calmar ratio
            peak = equity_curve.cummax()
            drawdown = (equity_curve - peak) / peak
            max_drawdown = abs(drawdown.min())
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
            
            # Win rate
            win_rate = (returns > 0).mean()
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'volatility': volatility
            }
            
        except Exception as e:
            self.logger.error(f"Performance metrics calculation failed: {e}")
            return {}
    
    def _calculate_trade_metrics(self, portfolio: Any) -> Dict[str, Any]:
        """Calculate trade-specific metrics."""
        try:
            trades = portfolio.trades.records_readable
            
            if len(trades) == 0:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'profit_factor': 0,
                    'avg_trade_duration': 0
                }
            
            # Basic trade statistics
            total_trades = len(trades)
            winning_trades = len(trades[trades['PnL'] > 0])
            losing_trades = len(trades[trades['PnL'] < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # P&L analysis
            pnl = trades['PnL']
            avg_win = pnl[pnl > 0].mean() if winning_trades > 0 else 0
            avg_loss = pnl[pnl < 0].mean() if losing_trades > 0 else 0
            
            # Profit factor
            gross_profit = pnl[pnl > 0].sum() if winning_trades > 0 else 0
            gross_loss = abs(pnl[pnl < 0].sum()) if losing_trades > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
            
            # Trade duration
            if 'Duration' in trades.columns:
                avg_trade_duration = trades['Duration'].mean()
            else:
                avg_trade_duration = 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'avg_trade_duration': avg_trade_duration,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss
            }
            
        except Exception as e:
            self.logger.error(f"Trade metrics calculation failed: {e}")
            return {}
    
    def optimize_portfolio(self, 
                          data: pd.DataFrame,
                          parameter_ranges: Dict[str, Tuple[float, float]],
                          objective: str = 'sharpe_ratio',
                          n_trials: int = 100) -> Dict[str, Any]:
        """Optimize portfolio parameters."""
        try:
            self.logger.info(f"Starting portfolio optimization with {n_trials} trials")
            
            best_score = float('-inf')
            best_params = {}
            optimization_history = []
            
            for i in range(n_trials):
                # Generate random parameters
                params = {}
                for param, (min_val, max_val) in parameter_ranges.items():
                    params[param] = np.random.uniform(min_val, max_val)
                
                try:
                    # Create portfolio with parameters
                    portfolio = vbt.Portfolio.from_signals(
                        close=data['close'],
                        entries=params.get('entries', pd.Series(False, index=data.index)),
                        exits=params.get('exits', pd.Series(False, index=data.index)),
                        init_cash=self.config.init_cash,
                        fees=params.get('fees', self.config.fees),
                        slippage=params.get('slippage', self.config.slippage),
                        freq=self.config.freq
                    )
                    
                    # Calculate objective score
                    if objective == 'sharpe_ratio':
                        returns = portfolio.returns()
                        score = returns.mean() / returns.std() if returns.std() > 0 else 0
                    elif objective == 'total_return':
                        score = (portfolio.value().iloc[-1] / portfolio.value().iloc[0]) - 1
                    else:
                        score = 0
                    
                    # Update best if improved
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                    
                    # Store history
                    optimization_history.append({
                        'trial': i + 1,
                        'params': params.copy(),
                        'score': score
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Trial {i+1} failed: {e}")
                    continue
            
            self.logger.info(f"Optimization completed. Best score: {best_score:.4f}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'optimization_history': optimization_history,
                'n_trials': n_trials,
                'objective': objective
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            raise VectorBTError(f"Portfolio optimization failed: {e}")
    
    def compare_portfolios(self, 
                          portfolios: List[PortfolioResult],
                          names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare multiple portfolios."""
        try:
            if names is None:
                names = [f"Portfolio_{i+1}" for i in range(len(portfolios))]
            
            comparison = {}
            
            for i, (portfolio_result, name) in enumerate(zip(portfolios, names)):
                comparison[name] = {
                    'total_return': portfolio_result.metrics['performance_metrics']['total_return'],
                    'sharpe_ratio': portfolio_result.metrics['performance_metrics']['sharpe_ratio'],
                    'max_drawdown': portfolio_result.metrics['risk_metrics']['max_drawdown'],
                    'win_rate': portfolio_result.metrics['performance_metrics']['win_rate'],
                    'total_trades': portfolio_result.metrics['trade_metrics']['total_trades']
                }
            
            # Find best portfolio for each metric
            best_portfolios = {}
            for metric in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']:
                if metric == 'max_drawdown':
                    # For drawdown, lower is better
                    best_portfolio = min(comparison.items(), key=lambda x: abs(x[1][metric]))
                else:
                    # For other metrics, higher is better
                    best_portfolio = max(comparison.items(), key=lambda x: x[1][metric])
                
                best_portfolios[metric] = best_portfolio[0]
            
            return {
                'comparison': comparison,
                'best_portfolios': best_portfolios
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio comparison failed: {e}")
            raise VectorBTError(f"Portfolio comparison failed: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of the last portfolio simulation."""
        if self.last_result is None:
            return {'error': 'No portfolio simulation performed yet'}
        
        return {
            'execution_time': self.last_result.execution_time,
            'total_return': self.last_result.metrics['performance_metrics']['total_return'],
            'sharpe_ratio': self.last_result.metrics['performance_metrics']['sharpe_ratio'],
            'max_drawdown': self.last_result.metrics['risk_metrics']['max_drawdown'],
            'total_trades': self.last_result.metrics['trade_metrics']['total_trades'],
            'win_rate': self.last_result.metrics['performance_metrics']['win_rate']
        }