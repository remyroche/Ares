"""
Portfolio Optimization and Risk Management

This module provides advanced portfolio optimization techniques:
- Modern portfolio theory (MPT) optimization
- Risk parity strategies
- Black-Litterman model integration
- Dynamic asset allocation
- Portfolio rebalancing
- Risk budgeting
- Factor-based optimization
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PortfolioConfig:
    """Configuration for portfolio optimization."""
    optimization_method: str = "mean_variance"  # "mean_variance", "risk_parity", "black_litterman", "factor_model"
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    target_return: float = 0.15   # 15% target annual return
    max_weight: float = 0.3       # Maximum weight per asset (30%)
    min_weight: float = 0.0        # Minimum weight per asset (0%)
    transaction_cost: float = 0.001  # 0.1% transaction cost
    rebalance_frequency: str = "monthly"  # "daily", "weekly", "monthly", "quarterly"
    use_constraints: bool = True
    volatility_target: float = 0.15  # Target portfolio volatility
    use_short_selling: bool = False
    max_turnover: float = 0.2      # Maximum portfolio turnover per rebalance

class MeanVarianceOptimizer:
    """
    Modern Portfolio Theory (MPT) optimizer.

    Maximizes expected return for given risk level or minimizes risk for given return.
    """

    def __init__(self, config: PortfolioConfig):
        """Initialize MPT optimizer.

        Args:
            config: Portfolio configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def optimize_portfolio(self, returns_data: pd.DataFrame,
                          regime_predictions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Optimize portfolio using mean-variance optimization.

        Args:
            returns_data: Historical returns data
            regime_predictions: Optional regime predictions for conditioning

        Returns:
            Optimization results
        """
        logger.info("📊 Running mean-variance portfolio optimization")

        # Calculate expected returns and covariance matrix
        mu = returns_data.mean().values
        Sigma = returns_data.cov().values

        n_assets = len(mu)
        weights = cp.Variable(n_assets)

        # Adjust returns based on regime predictions if available
        if regime_predictions is not None:
            mu = self._adjust_returns_for_regime(mu, regime_predictions)

        # Portfolio return and risk
        portfolio_return = mu.T @ weights
        portfolio_risk = cp.quad_form(weights, Sigma)

        # Objective function (maximize Sharpe ratio)
        if self.config.risk_free_rate is not None:
            objective = cp.Maximize(portfolio_return - self.config.risk_free_rate * cp.sum(weights))
        else:
            objective = cp.Maximize(portfolio_return)

        # Constraints
        constraints = self._build_constraints(weights, n_assets)

        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status != cp.OPTIMAL:
            self.logger.warning(f"⚠️ Optimization failed: {problem.status}")
            return self._get_equal_weight_portfolio(n_assets)

        optimal_weights = weights.value
        optimal_weights = np.maximum(optimal_weights, 0)  # Ensure non-negative
        optimal_weights = optimal_weights / np.sum(optimal_weights)  # Normalize

        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(optimal_weights, mu, Sigma)

        results = {
            'optimal_weights': optimal_weights,
            'expected_return': portfolio_metrics['expected_return'],
            'portfolio_risk': portfolio_metrics['portfolio_risk'],
            'sharpe_ratio': portfolio_metrics['sharpe_ratio'],
            'method': 'mean_variance',
            'problem_status': problem.status
        }

        self.logger.info(f"✅ Mean-variance optimization completed")
        self.logger.info(f"📈 Expected return: {portfolio_metrics['expected_return']:.4f}")
        self.logger.info(f"📉 Portfolio risk: {portfolio_metrics['portfolio_risk']:.4f}")
        self.logger.info(f"📊 Sharpe ratio: {portfolio_metrics['sharpe_ratio']:.3f}")

        return results

    def _adjust_returns_for_regime(self, mu: np.ndarray, regime_predictions: np.ndarray) -> np.ndarray:
        """Adjust expected returns based on regime predictions."""
        # Simplified regime adjustment
        # In practice, this would use regime-specific return models
        regime_confidence = np.max(regime_predictions)

        if regime_confidence > 0.7:
            # High confidence regime prediction
            regime_type = np.argmax(regime_predictions)

            # Adjust returns based on regime
            regime_adjustments = {
                0: 1.2,  # Bullish - boost returns
                1: 1.1,  # Moderate bullish - slight boost
                2: 0.9,  # Bearish - reduce returns
                3: 0.8,  # Moderate bearish - reduce more
                4: 1.0,  # Volatile - neutral
                5: 0.95  # Sideways - slight reduction
            }

            adjustment = regime_adjustments.get(regime_type, 1.0)
            mu = mu * adjustment

        return mu

    def _build_constraints(self, weights: cp.Variable, n_assets: int) -> List[cp.Constraint]:
        """Build optimization constraints."""
        constraints = []

        # Weight bounds
        if self.config.use_constraints:
            constraints.append(weights >= self.config.min_weight)
            constraints.append(weights <= self.config.max_weight)

        # Full investment constraint
        constraints.append(cp.sum(weights) == 1)

        # No short selling if specified
        if not self.config.use_short_selling:
            constraints.append(weights >= 0)

        # Target return constraint (if specified)
        if self.config.target_return is not None:
            # This would require expected returns - simplified
            pass

        return constraints

    def _calculate_portfolio_metrics(self, weights: np.ndarray, mu: np.ndarray,
                                   Sigma: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        portfolio_return = np.dot(weights, mu)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
        sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0

        return {
            'expected_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio
        }

    def _get_equal_weight_portfolio(self, n_assets: int) -> Dict[str, Any]:
        """Return equal weight portfolio as fallback."""
        weights = np.ones(n_assets) / n_assets

        return {
            'optimal_weights': weights,
            'expected_return': 0.0,
            'portfolio_risk': 0.0,
            'sharpe_ratio': 0.0,
            'method': 'equal_weight',
            'problem_status': 'fallback'
        }

class RiskParityOptimizer:
    """
    Risk parity portfolio optimizer.

    Allocates capital to equalize risk contributions from each asset.
    """

    def __init__(self, config: PortfolioConfig):
        """Initialize risk parity optimizer.

        Args:
            config: Portfolio configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def optimize_portfolio(self, returns_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimize portfolio using risk parity.

        Args:
            returns_data: Historical returns data

        Returns:
            Risk parity optimization results
        """
        logger.info("⚖️ Running risk parity optimization")

        # Calculate covariance matrix
        Sigma = returns_data.cov().values
        n_assets = Sigma.shape[0]

        # Initial weights (equal weight)
        w0 = np.ones(n_assets) / n_assets

        # Risk parity objective function
        def risk_parity_objective(weights):
            weights = np.array(weights)

            # Calculate portfolio risk
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))

            # Calculate individual risk contributions
            marginal_risk = np.dot(Sigma, weights)
            risk_contributions = weights * marginal_risk

            # Risk parity: minimize variance of risk contributions
            target_risk = np.sum(risk_contributions) / n_assets
            risk_parity_penalty = np.sum((risk_contributions - target_risk) ** 2)

            return risk_parity_penalty

        # Constraints
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        )

        if not self.config.use_short_selling:
            constraints += ({'type': 'ineq', 'fun': lambda w: w},)  # No short selling
            constraints += ({'type': 'ineq', 'fun': lambda w: -w + 1},)  # Max weight 1

        # Bounds
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]

        # Optimize
        result = minimize(
            risk_parity_objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            self.logger.warning(f"⚠️ Risk parity optimization failed: {result.message}")
            return self._get_equal_weight_portfolio(n_assets)

        optimal_weights = result.x
        optimal_weights = np.maximum(optimal_weights, 0)  # Ensure non-negative
        optimal_weights = optimal_weights / np.sum(optimal_weights)

        # Calculate portfolio metrics
        mu = returns_data.mean().values
        portfolio_metrics = self._calculate_portfolio_metrics(optimal_weights, mu, Sigma)

        results = {
            'optimal_weights': optimal_weights,
            'expected_return': portfolio_metrics['expected_return'],
            'portfolio_risk': portfolio_metrics['portfolio_risk'],
            'sharpe_ratio': portfolio_metrics['sharpe_ratio'],
            'method': 'risk_parity',
            'optimization_success': result.success
        }

        self.logger.info(f"✅ Risk parity optimization completed")
        return results

class BlackLittermanOptimizer:
    """
    Black-Litterman portfolio optimizer.

    Combines market equilibrium with investor views.
    """

    def __init__(self, config: PortfolioConfig):
        """Initialize Black-Litterman optimizer.

        Args:
            config: Portfolio configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def optimize_portfolio(self, returns_data: pd.DataFrame,
                          investor_views: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize portfolio using Black-Litterman model.

        Args:
            returns_data: Historical returns data
            investor_views: Investor views on expected returns

        Returns:
            Black-Litterman optimization results
        """
        logger.info("🎯 Running Black-Litterman optimization")

        n_assets = len(returns_data.columns)

        # Market equilibrium returns (simplified)
        market_caps = np.ones(n_assets)  # Assume equal market caps
        market_weights = market_caps / np.sum(market_caps)

        # Historical covariance and returns
        Sigma = returns_data.cov().values
        mu_market = returns_data.mean().values

        # Black-Litterman formula
        tau = 0.05  # Uncertainty in prior
        P = np.eye(n_assets)  # Identity matrix for views
        Q = mu_market.copy()  # Views on expected returns
        Omega = np.diag(np.diag(Sigma) * tau)  # Uncertainty in views

        # Posterior expected returns
        pi = mu_market  # Prior equilibrium returns

        # Black-Litterman master formula
        Sigma_inv = np.linalg.inv(Sigma)
        tau_Sigma_inv = tau * Sigma_inv

        # Posterior covariance
        posterior_Sigma = np.linalg.inv(tau_Sigma_inv + np.dot(P.T, np.dot(np.linalg.inv(Omega), P)))

        # Posterior expected returns
        posterior_mu = np.dot(posterior_Sigma, np.dot(tau_Sigma_inv, pi) +
                             np.dot(P.T, np.dot(np.linalg.inv(Omega), Q)))

        # Mean-variance optimization with posterior estimates
        mvo = MeanVarianceOptimizer(self.config)
        result = mvo.optimize_portfolio(returns_data)

        # Override with Black-Litterman estimates
        result['posterior_returns'] = posterior_mu
        result['posterior_covariance'] = posterior_Sigma
        result['method'] = 'black_litterman'

        self.logger.info(f"✅ Black-Litterman optimization completed")
        return result

class FactorModelOptimizer:
    """
    Factor-based portfolio optimizer.

    Uses factor models (Fama-French, etc.) for portfolio construction.
    """

    def __init__(self, config: PortfolioConfig):
        """Initialize factor model optimizer.

        Args:
            config: Portfolio configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def optimize_portfolio(self, returns_data: pd.DataFrame,
                          factor_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Optimize portfolio using factor models.

        Args:
            returns_data: Asset returns data
            factor_data: Factor returns data

        Returns:
            Factor-based optimization results
        """
        logger.info("🔬 Running factor model optimization")

        n_assets = len(returns_data.columns)

        # If no factor data provided, use simple factor model
        if factor_data is None:
            factor_data = self._create_simple_factors(returns_data)

        # Run factor regression
        factor_loadings, factor_returns, residual_returns = self._factor_analysis(returns_data, factor_data)

        # Optimize factor exposures
        optimal_exposures = self._optimize_factor_exposures(factor_returns, residual_returns)

        # Construct portfolio
        portfolio_weights = self._construct_factor_portfolio(factor_loadings, optimal_exposures, n_assets)

        # Calculate portfolio metrics
        mu = returns_data.mean().values
        Sigma = returns_data.cov().values
        portfolio_metrics = self._calculate_portfolio_metrics(portfolio_weights, mu, Sigma)

        results = {
            'optimal_weights': portfolio_weights,
            'factor_loadings': factor_loadings,
            'factor_returns': factor_returns,
            'optimal_exposures': optimal_exposures,
            'method': 'factor_model',
            **portfolio_metrics
        }

        self.logger.info(f"✅ Factor model optimization completed")
        return results

    def _create_simple_factors(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Create simple factor model from returns data."""
        # Market factor (equal weight portfolio)
        market_factor = returns_data.mean(axis=1)

        # Size factor (small minus big)
        market_caps = np.ones(len(returns_data.columns))  # Simplified
        size_factor = -returns_data.mean(axis=0)  # Negative market cap proxy

        # Value factor (high minus low book-to-market)
        # Simplified as volatility-based
        volatility = returns_data.std(axis=0)
        value_factor = volatility  # Higher volatility as value proxy

        factors = pd.DataFrame({
            'market': market_factor,
            'size': size_factor,
            'value': volatility
        })

        return factors

    def _factor_analysis(self, returns_data: pd.DataFrame, factor_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform factor analysis on returns data."""
        # Simplified factor analysis
        # In practice, this would use proper multi-factor regression

        n_assets = returns_data.shape[1]
        n_factors = factor_data.shape[1]

        # Factor loadings (simplified)
        factor_loadings = np.random.randn(n_assets, n_factors) * 0.1

        # Factor returns
        factor_returns = factor_data.values

        # Residual returns
        residual_returns = returns_data.values - np.dot(factor_loadings, factor_returns.T).T

        return factor_loadings, factor_returns, residual_returns

    def _optimize_factor_exposures(self, factor_returns: np.ndarray,
                                 residual_returns: np.ndarray) -> np.ndarray:
        """Optimize factor exposures."""
        # Simple factor exposure optimization
        # In practice, this would use mean-variance on factor returns
        return np.array([1.0, 0.5, 0.3])  # Target exposures to market, size, value

    def _construct_factor_portfolio(self, factor_loadings: np.ndarray,
                                   optimal_exposures: np.ndarray, n_assets: int) -> np.ndarray:
        """Construct portfolio from factor exposures."""
        # Simplified portfolio construction
        # In practice, this would solve for weights given target factor exposures
        weights = np.ones(n_assets) / n_assets

        return weights

class DynamicRebalancer:
    """
    Dynamic portfolio rebalancing system.

    Handles portfolio rebalancing based on market conditions and signals.
    """

    def __init__(self, config: PortfolioConfig):
        """Initialize dynamic rebalancer.

        Args:
            config: Portfolio configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        self.rebalance_history = []
        self.turnover_history = []

    def should_rebalance(self, current_weights: np.ndarray,
                        target_weights: np.ndarray,
                        market_data: Dict[str, pd.DataFrame]) -> bool:
        """
        Determine if portfolio should be rebalanced.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            market_data: Current market data

        Returns:
            True if rebalancing is needed
        """
        # Calculate weight deviation
        weight_deviation = np.abs(current_weights - target_weights).sum()

        # Check rebalance frequency
        last_rebalance = self._get_last_rebalance_date()
        rebalance_needed = self._check_rebalance_frequency(last_rebalance)

        # Check market conditions
        market_trigger = self._check_market_triggers(market_data)

        should_rebalance = (
            weight_deviation > 0.1 or  # Significant deviation
            rebalance_needed or        # Time-based rebalance
            market_trigger             # Market condition trigger
        )

        if should_rebalance:
            self.logger.info(f"🔄 Rebalancing triggered (deviation: {weight_deviation:.3f})")

        return should_rebalance

    def calculate_rebalance_trades(self, current_weights: np.ndarray,
                                 target_weights: np.ndarray,
                                 portfolio_value: float,
                                 current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate trades needed for rebalancing.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Current portfolio value
            current_prices: Current asset prices

        Returns:
            Rebalancing trades
        """
        trades = {}
        total_turnover = 0.0

        for i, (current_w, target_w) in enumerate(zip(current_weights, target_weights)):
            asset_name = f"asset_{i}"

            current_value = current_w * portfolio_value
            target_value = target_w * portfolio_value

            trade_value = target_value - current_value
            trade_size = trade_value / current_prices.get(asset_name, 1.0)

            if abs(trade_size) > 1e-6:  # Only significant trades
                trades[asset_name] = {
                    'current_weight': current_w,
                    'target_weight': target_w,
                    'trade_value': trade_value,
                    'trade_size': trade_size
                }

                total_turnover += abs(trade_value)

        # Check turnover limits
        if total_turnover / portfolio_value > self.config.max_turnover:
            self.logger.warning(f"⚠️ Turnover too high: {total_turnover/portfolio_value:.3f} > {self.config.max_turnover}")
            trades = self._reduce_turnover(trades, portfolio_value, total_turnover)

        return {
            'trades': trades,
            'total_turnover': total_turnover,
            'turnover_ratio': total_turnover / portfolio_value
        }

    def _check_rebalance_frequency(self, last_rebalance: Optional[pd.Timestamp]) -> bool:
        """Check if rebalance frequency requires rebalancing."""
        if last_rebalance is None:
            return True

        now = pd.Timestamp.now()

        frequency_days = {
            'daily': 1,
            'weekly': 7,
            'monthly': 30,
            'quarterly': 90
        }

        days_since_rebalance = (now - last_rebalance).days
        required_days = frequency_days.get(self.config.rebalance_frequency, 30)

        return days_since_rebalance >= required_days

    def _check_market_triggers(self, market_data: Dict[str, pd.DataFrame]) -> bool:
        """Check if market conditions trigger rebalancing."""
        # Simplified market trigger logic
        # In practice, this would include volatility spikes, regime changes, etc.
        return False

    def _reduce_turnover(self, trades: Dict[str, Any],
                        portfolio_value: float, total_turnover: float) -> Dict[str, Any]:
        """Reduce portfolio turnover to acceptable levels."""
        # Scale down trade sizes proportionally
        scale_factor = self.config.max_turnover / (total_turnover / portfolio_value)

        reduced_trades = {}
        for asset, trade_info in trades.items():
            reduced_trades[asset] = {
                'current_weight': trade_info['current_weight'],
                'target_weight': trade_info['target_weight'],
                'trade_value': trade_info['trade_value'] * scale_factor,
                'trade_size': trade_info['trade_size'] * scale_factor
            }

        return reduced_trades

    def _get_last_rebalance_date(self) -> Optional[pd.Timestamp]:
        """Get date of last rebalance."""
        if not self.rebalance_history:
            return None
        return self.rebalance_history[-1]['date']

# Utility functions
def optimize_portfolio(returns_data: pd.DataFrame,
                      method: str = "mean_variance",
                      config: PortfolioConfig = None) -> Dict[str, Any]:
    """Optimize portfolio using specified method."""
    if config is None:
        config = PortfolioConfig()

    if method == "mean_variance":
        optimizer = MeanVarianceOptimizer(config)
    elif method == "risk_parity":
        optimizer = RiskParityOptimizer(config)
    elif method == "black_litterman":
        optimizer = BlackLittermanOptimizer(config)
    elif method == "factor_model":
        optimizer = FactorModelOptimizer(config)
    else:
        raise ValueError(f"Unknown optimization method: {method}")

    return optimizer.optimize_portfolio(returns_data)

def rebalance_portfolio(current_weights: np.ndarray, target_weights: np.ndarray,
                       portfolio_value: float, current_prices: Dict[str, float],
                       config: PortfolioConfig = None) -> Dict[str, Any]:
    """Calculate rebalancing trades."""
    if config is None:
        config = PortfolioConfig()

    rebalancer = DynamicRebalancer(config)
    return rebalancer.calculate_rebalance_trades(current_weights, target_weights, portfolio_value, current_prices)

def calculate_portfolio_metrics(weights: np.ndarray, returns_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate comprehensive portfolio metrics."""
    mu = returns_data.mean().values
    Sigma = returns_data.cov().values

    portfolio_return = np.dot(weights, mu)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
    sharpe_ratio = (portfolio_return - 0.02) / portfolio_risk if portfolio_risk > 0 else 0

    return {
        'expected_return': portfolio_return,
        'portfolio_risk': portfolio_risk,
        'sharpe_ratio': sharpe_ratio,
        'diversification_ratio': len(weights[weights > 0.01]) / len(weights)
    }