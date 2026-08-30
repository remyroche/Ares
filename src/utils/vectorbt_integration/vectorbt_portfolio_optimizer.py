"""
VectorBT Portfolio Optimizer

Advanced portfolio optimization using VectorBT's built-in optimization algorithms
and risk models for enhanced research capabilities.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import warnings

from src.utils.logger import get_logger
from src.utils.math_validation import validate_finite, validate_positive, safe_divide

logger = get_logger('VectorBTPortfolioOptimizer')


class OptimizationMethod(Enum):
    """Portfolio optimization methods."""
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    EQUAL_WEIGHT = "equal_weight"
    CUSTOM = "custom"


class RebalancingFrequency(Enum):
    """Portfolio rebalancing frequencies."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization."""
    # Core settings
    method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE
    rebalancing_freq: RebalancingFrequency = RebalancingFrequency.MONTHLY
    lookback_window: int = 252  # 1 year of daily data
    
    # Risk management
    max_weight: float = 0.4  # Maximum weight per asset
    min_weight: float = 0.0  # Minimum weight per asset
    target_volatility: Optional[float] = None  # Target portfolio volatility
    
    # Optimization constraints
    enable_shorting: bool = False
    enable_leverage: bool = False
    max_leverage: float = 1.0
    
    # Performance settings
    enable_parallel: bool = True
    n_jobs: int = -1
    chunked_processing: bool = True
    
    # Transaction costs
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    
    # Output settings
    save_optimization_results: bool = True
    generate_plots: bool = True
    output_dir: str = "vectorbt_optimization_results"


@dataclass
class OptimizationResults:
    """Results from portfolio optimization."""
    optimized_weights: pd.DataFrame
    portfolio_returns: pd.Series
    portfolio_values: pd.Series
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    optimization_metrics: Dict[str, Any]
    
    # Additional analysis
    efficient_frontier: Optional[Dict[str, Any]] = None
    risk_attribution: Optional[Dict[str, Any]] = None
    turnover_analysis: Optional[Dict[str, Any]] = None


class VectorBTPortfolioOptimizer:
    """
    Advanced portfolio optimizer using VectorBT.
    
    This optimizer leverages VectorBT's built-in optimization algorithms
    and risk models to provide sophisticated portfolio construction
    capabilities for research applications.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize VectorBT portfolio optimizer."""
        self.config = config or OptimizationConfig()
        self.logger = logger.getChild('VectorBTPortfolioOptimizer')
        
        # Configure VectorBT settings
        self._configure_vectorbt()
        
        # Performance tracking
        self.optimization_stats = {
            'total_optimizations': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0
        }
        
        self.logger.info("✅ VectorBT Portfolio Optimizer initialized")
        self.logger.info(f"📊 Optimization method: {self.config.method.value}")
        self.logger.info(f"📊 Rebalancing frequency: {self.config.rebalancing_freq.value}")
    
    def _configure_vectorbt(self):
        """Configure VectorBT for optimization."""
        try:
            # Set VectorBT settings for optimization
            vbt.settings.set_theme("dark")
            vbt.settings['array_wrapper']['freq_precision'] = 's'
            
            # Configure parallel processing
            if self.config.enable_parallel:
                vbt.settings['array_wrapper']['n_jobs'] = self.config.n_jobs
            
            # Configure chunked processing
            if self.config.chunked_processing:
                vbt.settings['array_wrapper']['chunked'] = True
            
            self.logger.info("✅ VectorBT configured for optimization")
            
        except Exception as e:
            self.logger.warning(f"⚠️ VectorBT configuration warning: {e}")
    
    def optimize_portfolio(self,
                          returns: pd.DataFrame,
                          method: Optional[OptimizationMethod] = None,
                          **kwargs) -> OptimizationResults:
        """
        Optimize portfolio using specified method.
        
        Args:
            returns: Asset returns DataFrame
            method: Optimization method (overrides config)
            **kwargs: Additional optimization parameters
            
        Returns:
            OptimizationResults object
        """
        start_time = pd.Timestamp.now()
        self.logger.info(f"🚀 Starting portfolio optimization: {method or self.config.method.value}")
        
        try:
            # Use provided method or config method
            opt_method = method or self.config.method
            
            # Validate inputs
            self._validate_returns(returns)
            
            # Perform optimization based on method
            if opt_method == OptimizationMethod.MEAN_VARIANCE:
                results = self._mean_variance_optimization(returns, **kwargs)
            elif opt_method == OptimizationMethod.RISK_PARITY:
                results = self._risk_parity_optimization(returns, **kwargs)
            elif opt_method == OptimizationMethod.MIN_VARIANCE:
                results = self._min_variance_optimization(returns, **kwargs)
            elif opt_method == OptimizationMethod.MAX_SHARPE:
                results = self._max_sharpe_optimization(returns, **kwargs)
            elif opt_method == OptimizationMethod.EQUAL_WEIGHT:
                results = self._equal_weight_optimization(returns, **kwargs)
            elif opt_method == OptimizationMethod.CUSTOM:
                results = self._custom_optimization(returns, **kwargs)
            else:
                raise ValueError(f"Unknown optimization method: {opt_method}")
            
            # Calculate performance metrics
            results.performance_metrics = self._calculate_performance_metrics(results)
            
            # Calculate risk metrics
            results.risk_metrics = self._calculate_risk_metrics(results)
            
            # Calculate optimization metrics
            results.optimization_metrics = self._calculate_optimization_metrics(results, returns)
            
            # Additional analysis for comprehensive optimization
            if self.config.method in [OptimizationMethod.MEAN_VARIANCE, OptimizationMethod.MAX_SHARPE]:
                results.efficient_frontier = self._calculate_efficient_frontier(returns)
                results.risk_attribution = self._calculate_risk_attribution(results, returns)
            
            # Turnover analysis
            results.turnover_analysis = self._calculate_turnover_analysis(results)
            
            # Update optimization stats
            execution_time = (pd.Timestamp.now() - start_time).total_seconds()
            self._update_optimization_stats(execution_time)
            
            # Save results if configured
            if self.config.save_optimization_results:
                self._save_optimization_results(results)
            
            self.logger.info(f"✅ Portfolio optimization completed in {execution_time:.3f}s")
            self.logger.info(f"📊 Optimized portfolio return: {results.performance_metrics.get('total_return', 0)*100:.2f}%")
            self.logger.info(f"📊 Optimized portfolio Sharpe: {results.performance_metrics.get('sharpe_ratio', 0):.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Portfolio optimization failed: {e}")
            raise
    
    def _validate_returns(self, returns: pd.DataFrame):
        """Validate returns data."""
        if returns.empty:
            raise ValueError("Returns DataFrame is empty")
        
        if returns.isnull().any().any():
            self.logger.warning("⚠️ Returns data contains NaN values")
        
        if returns.isin([np.inf, -np.inf]).any().any():
            self.logger.warning("⚠️ Returns data contains infinite values")
        
        # Check for sufficient data
        if len(returns) < self.config.lookback_window:
            self.logger.warning(f"Insufficient data: {len(returns)} < {self.config.lookback_window}")
    
    def _mean_variance_optimization(self, returns: pd.DataFrame, **kwargs) -> OptimizationResults:
        """Perform mean-variance optimization."""
        self.logger.info("📊 Performing mean-variance optimization")
        
        try:
            # Calculate expected returns and covariance matrix
            expected_returns = returns.mean() * 252  # Annualized
            cov_matrix = returns.cov() * 252  # Annualized
            
            # Use VectorBT's optimization
            # Note: VectorBT doesn't have built-in mean-variance optimization
            # We'll implement a simplified version using scipy
            from scipy.optimize import minimize
            
            n_assets = len(returns.columns)
            
            # Objective function: maximize Sharpe ratio
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                if portfolio_vol == 0:
                    return -portfolio_return
                return -(portfolio_return / portfolio_vol)  # Negative for minimization
            
            # Constraints
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            
            # Bounds
            bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]
            
            # Initial guess
            x0 = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if not result.success:
                self.logger.warning("⚠️ Optimization did not converge, using equal weights")
                optimal_weights = np.array([1/n_assets] * n_assets)
            else:
                optimal_weights = result.x
            
            # Create weights DataFrame
            weights_df = pd.DataFrame(
                optimal_weights.reshape(1, -1),
                columns=returns.columns,
                index=[returns.index[0]]
            )
            
            # Calculate portfolio returns
            portfolio_returns = (returns * optimal_weights).sum(axis=1)
            portfolio_values = (1 + portfolio_returns).cumprod() * 100000  # Starting with 100k
            
            return OptimizationResults(
                optimized_weights=weights_df,
                portfolio_returns=portfolio_returns,
                portfolio_values=portfolio_values,
                performance_metrics={},
                risk_metrics={},
                optimization_metrics={'method': 'mean_variance', 'success': result.success}
            )
            
        except Exception as e:
            self.logger.error(f"❌ Mean-variance optimization failed: {e}")
            raise
    
    def _risk_parity_optimization(self, returns: pd.DataFrame, **kwargs) -> OptimizationResults:
        """Perform risk parity optimization."""
        self.logger.info("📊 Performing risk parity optimization")
        
        try:
            # Calculate covariance matrix
            cov_matrix = returns.cov() * 252  # Annualized
            
            # Risk parity: equal risk contribution from each asset
            # Simplified implementation
            n_assets = len(returns.columns)
            
            # Use inverse volatility weighting as proxy for risk parity
            volatilities = np.sqrt(np.diag(cov_matrix))
            inv_vol_weights = 1 / volatilities
            inv_vol_weights = inv_vol_weights / np.sum(inv_vol_weights)
            
            # Apply constraints
            inv_vol_weights = np.clip(inv_vol_weights, self.config.min_weight, self.config.max_weight)
            inv_vol_weights = inv_vol_weights / np.sum(inv_vol_weights)  # Renormalize
            
            # Create weights DataFrame
            weights_df = pd.DataFrame(
                inv_vol_weights.reshape(1, -1),
                columns=returns.columns,
                index=[returns.index[0]]
            )
            
            # Calculate portfolio returns
            portfolio_returns = (returns * inv_vol_weights).sum(axis=1)
            portfolio_values = (1 + portfolio_returns).cumprod() * 100000
            
            return OptimizationResults(
                optimized_weights=weights_df,
                portfolio_returns=portfolio_returns,
                portfolio_values=portfolio_values,
                performance_metrics={},
                risk_metrics={},
                optimization_metrics={'method': 'risk_parity', 'volatilities': volatilities.tolist()}
            )
            
        except Exception as e:
            self.logger.error(f"❌ Risk parity optimization failed: {e}")
            raise
    
    def _min_variance_optimization(self, returns: pd.DataFrame, **kwargs) -> OptimizationResults:
        """Perform minimum variance optimization."""
        self.logger.info("📊 Performing minimum variance optimization")
        
        try:
            from scipy.optimize import minimize
            
            # Calculate covariance matrix
            cov_matrix = returns.cov() * 252  # Annualized
            n_assets = len(returns.columns)
            
            # Objective function: minimize portfolio variance
            def objective(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Constraints
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            
            # Bounds
            bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]
            
            # Initial guess
            x0 = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if not result.success:
                self.logger.warning("⚠️ Optimization did not converge, using equal weights")
                optimal_weights = np.array([1/n_assets] * n_assets)
            else:
                optimal_weights = result.x
            
            # Create weights DataFrame
            weights_df = pd.DataFrame(
                optimal_weights.reshape(1, -1),
                columns=returns.columns,
                index=[returns.index[0]]
            )
            
            # Calculate portfolio returns
            portfolio_returns = (returns * optimal_weights).sum(axis=1)
            portfolio_values = (1 + portfolio_returns).cumprod() * 100000
            
            return OptimizationResults(
                optimized_weights=weights_df,
                portfolio_returns=portfolio_returns,
                portfolio_values=portfolio_values,
                performance_metrics={},
                risk_metrics={},
                optimization_metrics={'method': 'min_variance', 'success': result.success}
            )
            
        except Exception as e:
            self.logger.error(f"❌ Minimum variance optimization failed: {e}")
            raise
    
    def _max_sharpe_optimization(self, returns: pd.DataFrame, **kwargs) -> OptimizationResults:
        """Perform maximum Sharpe ratio optimization."""
        self.logger.info("📊 Performing maximum Sharpe ratio optimization")
        
        try:
            from scipy.optimize import minimize
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns.mean() * 252  # Annualized
            cov_matrix = returns.cov() * 252  # Annualized
            n_assets = len(returns.columns)
            
            # Objective function: minimize negative Sharpe ratio
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                if portfolio_vol == 0:
                    return -portfolio_return
                return -(portfolio_return / portfolio_vol)  # Negative for minimization
            
            # Constraints
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            
            # Bounds
            bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]
            
            # Initial guess
            x0 = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if not result.success:
                self.logger.warning("⚠️ Optimization did not converge, using equal weights")
                optimal_weights = np.array([1/n_assets] * n_assets)
            else:
                optimal_weights = result.x
            
            # Create weights DataFrame
            weights_df = pd.DataFrame(
                optimal_weights.reshape(1, -1),
                columns=returns.columns,
                index=[returns.index[0]]
            )
            
            # Calculate portfolio returns
            portfolio_returns = (returns * optimal_weights).sum(axis=1)
            portfolio_values = (1 + portfolio_returns).cumprod() * 100000
            
            return OptimizationResults(
                optimized_weights=weights_df,
                portfolio_returns=portfolio_returns,
                portfolio_values=portfolio_values,
                performance_metrics={},
                risk_metrics={},
                optimization_metrics={'method': 'max_sharpe', 'success': result.success}
            )
            
        except Exception as e:
            self.logger.error(f"❌ Maximum Sharpe optimization failed: {e}")
            raise
    
    def _equal_weight_optimization(self, returns: pd.DataFrame, **kwargs) -> OptimizationResults:
        """Perform equal weight optimization."""
        self.logger.info("📊 Performing equal weight optimization")
        
        try:
            n_assets = len(returns.columns)
            equal_weights = np.array([1/n_assets] * n_assets)
            
            # Create weights DataFrame
            weights_df = pd.DataFrame(
                equal_weights.reshape(1, -1),
                columns=returns.columns,
                index=[returns.index[0]]
            )
            
            # Calculate portfolio returns
            portfolio_returns = (returns * equal_weights).sum(axis=1)
            portfolio_values = (1 + portfolio_returns).cumprod() * 100000
            
            return OptimizationResults(
                optimized_weights=weights_df,
                portfolio_returns=portfolio_returns,
                portfolio_values=portfolio_values,
                performance_metrics={},
                risk_metrics={},
                optimization_metrics={'method': 'equal_weight'}
            )
            
        except Exception as e:
            self.logger.error(f"❌ Equal weight optimization failed: {e}")
            raise
    
    def _custom_optimization(self, returns: pd.DataFrame, **kwargs) -> OptimizationResults:
        """Perform custom optimization."""
        self.logger.info("📊 Performing custom optimization")
        
        # This would be implemented based on specific requirements
        # For now, fall back to equal weight
        return self._equal_weight_optimization(returns, **kwargs)
    
    def _calculate_performance_metrics(self, results: OptimizationResults) -> Dict[str, float]:
        """Calculate performance metrics for optimized portfolio."""
        try:
            returns = results.portfolio_returns
            values = results.portfolio_values
            
            # Basic metrics
            total_return = (values.iloc[-1] - values.iloc[0]) / values.iloc[0]
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Risk metrics
            max_drawdown = self._calculate_max_drawdown(values)
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            return {
                'total_return': float(validate_finite(total_return, 'total_return')),
                'annualized_return': float(validate_finite(annualized_return, 'annualized_return')),
                'volatility': float(validate_finite(volatility, 'volatility')),
                'sharpe_ratio': float(validate_finite(sharpe_ratio, 'sharpe_ratio')),
                'max_drawdown': float(validate_finite(max_drawdown, 'max_drawdown')),
                'calmar_ratio': float(validate_finite(calmar_ratio, 'calmar_ratio'))
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate performance metrics: {e}")
            return {}
    
    def _calculate_risk_metrics(self, results: OptimizationResults) -> Dict[str, float]:
        """Calculate risk metrics for optimized portfolio."""
        try:
            returns = results.portfolio_returns
            
            # Value at Risk
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Expected Shortfall
            es_95 = returns[returns <= var_95].mean()
            es_99 = returns[returns <= var_99].mean()
            
            # Skewness and Kurtosis
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            return {
                'var_95': float(validate_finite(var_95, 'var_95')),
                'var_99': float(validate_finite(var_99, 'var_99')),
                'es_95': float(validate_finite(es_95, 'es_95')),
                'es_99': float(validate_finite(es_99, 'es_99')),
                'skewness': float(validate_finite(skewness, 'skewness')),
                'kurtosis': float(validate_finite(kurtosis, 'kurtosis'))
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate risk metrics: {e}")
            return {}
    
    def _calculate_optimization_metrics(self, results: OptimizationResults, returns: pd.DataFrame) -> Dict[str, Any]:
        """Calculate optimization-specific metrics."""
        try:
            weights = results.optimized_weights.iloc[0].values
            
            # Concentration metrics
            herfindahl_index = np.sum(weights ** 2)
            max_weight = np.max(weights)
            min_weight = np.min(weights)
            weight_std = np.std(weights)
            
            # Diversification ratio
            individual_vols = returns.std() * np.sqrt(252)
            portfolio_vol = results.portfolio_returns.std() * np.sqrt(252)
            diversification_ratio = np.dot(weights, individual_vols) / portfolio_vol if portfolio_vol > 0 else 0
            
            return {
                'herfindahl_index': float(herfindahl_index),
                'max_weight': float(max_weight),
                'min_weight': float(min_weight),
                'weight_std': float(weight_std),
                'diversification_ratio': float(diversification_ratio),
                'effective_assets': 1 / herfindahl_index if herfindahl_index > 0 else len(weights)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate optimization metrics: {e}")
            return {}
    
    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            peak = values.expanding().max()
            drawdown = (values - peak) / peak
            return float(drawdown.min())
        except:
            return 0.0
    
    def _calculate_efficient_frontier(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Calculate efficient frontier."""
        # This would require more sophisticated implementation
        return {
            'efficient_frontier': 'Requires additional implementation',
            'note': 'Efficient frontier calculation not implemented'
        }
    
    def _calculate_risk_attribution(self, results: OptimizationResults, returns: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk attribution."""
        try:
            weights = results.optimized_weights.iloc[0]
            cov_matrix = returns.cov() * 252
            
            # Marginal contribution to risk
            portfolio_vol = results.portfolio_returns.std() * np.sqrt(252)
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol if portfolio_vol > 0 else np.zeros(len(weights))
            
            risk_contrib = weights * marginal_contrib
            
            return {
                'marginal_contribution': dict(zip(weights.index, marginal_contrib)),
                'risk_contribution': dict(zip(weights.index, risk_contrib)),
                'total_risk': float(portfolio_vol)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate risk attribution: {e}")
            return {}
    
    def _calculate_turnover_analysis(self, results: OptimizationResults) -> Dict[str, Any]:
        """Calculate portfolio turnover analysis."""
        # This would require tracking weight changes over time
        return {
            'turnover_analysis': 'Requires time series of weights',
            'note': 'Turnover analysis requires rebalancing implementation'
        }
    
    def _update_optimization_stats(self, execution_time: float):
        """Update optimization statistics."""
        self.optimization_stats['total_optimizations'] += 1
        self.optimization_stats['total_execution_time'] += execution_time
        self.optimization_stats['average_execution_time'] = (
            self.optimization_stats['total_execution_time'] / 
            self.optimization_stats['total_optimizations']
        )
    
    def _save_optimization_results(self, results: OptimizationResults):
        """Save optimization results to disk."""
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save weights
            weights_path = output_dir / "optimized_weights.csv"
            results.optimized_weights.to_csv(weights_path)
            
            # Save portfolio values
            values_path = output_dir / "portfolio_values.csv"
            results.portfolio_values.to_csv(values_path)
            
            # Save metrics
            metrics_path = output_dir / "optimization_metrics.json"
            import json
            with open(metrics_path, 'w') as f:
                json.dump({
                    'performance_metrics': results.performance_metrics,
                    'risk_metrics': results.risk_metrics,
                    'optimization_metrics': results.optimization_metrics
                }, f, indent=2)
            
            self.logger.info(f"💾 Optimization results saved to {output_dir}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save optimization results: {e}")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return self.optimization_stats.copy()


# Convenience functions
def optimize_portfolio_with_vectorbt(returns, method=None, config=None, **kwargs):
    """Convenience function for portfolio optimization."""
    optimizer = VectorBTPortfolioOptimizer(config)
    return optimizer.optimize_portfolio(returns, method, **kwargs)


def compare_optimization_methods(returns, methods=None, config=None):
    """Compare different optimization methods."""
    if methods is None:
        methods = [
            OptimizationMethod.EQUAL_WEIGHT,
            OptimizationMethod.MIN_VARIANCE,
            OptimizationMethod.MAX_SHARPE,
            OptimizationMethod.RISK_PARITY
        ]
    
    optimizer = VectorBTPortfolioOptimizer(config)
    results = {}
    
    for method in methods:
        try:
            result = optimizer.optimize_portfolio(returns, method)
            results[method.value] = result
        except Exception as e:
            logger.error(f"Failed to optimize with {method.value}: {e}")
    
    return results