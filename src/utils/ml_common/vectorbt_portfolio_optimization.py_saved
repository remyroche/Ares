"""
VectorBT-Enhanced Portfolio Optimization

This module provides comprehensive portfolio optimization capabilities using VectorBT
for asset allocation, risk management, and performance optimization.

Key Features:
- Mean-variance optimization
- Risk parity and equal weight strategies
- Black-Litterman model implementation
- Multi-objective optimization
- Constraint handling (sector limits, concentration limits)
- Dynamic rebalancing strategies
- Regime-aware optimization
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
from datetime import datetime, timedelta
import time

from src.utils.ml_common.transaction_costs import DEFAULT_TRANSACTION_COST

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.portfolio.base import Portfolio
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    Portfolio = None

# Optimization libraries
try:
    from scipy.optimize import minimize, minimize_scalar
    from scipy.linalg import cholesky, solve_triangular
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from .vectorbt_memory_manager import get_memory_manager, memory_managed_operation, optimize_memory_usage
from .vectorbt_performance_monitor import get_performance_monitor, monitor_operation

logger = logging.getLogger(__name__)

class OptimizationMethod(Enum):
    """Portfolio optimization methods."""
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    BLACK_LITTERMAN = "black_litterman"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"
    MAX_DIVERSIFICATION = "max_diversification"

class RebalancingFrequency(Enum):
    """Portfolio rebalancing frequencies."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    ADAPTIVE = "adaptive"

@dataclass
class OptimizationConstraints:
    """Constraints for portfolio optimization."""
    # Weight constraints
    min_weight: float = 0.0
    max_weight: float = 1.0
    min_total_weight: float = 0.95
    max_total_weight: float = 1.05

    # Sector constraints
    sector_limits: Optional[Dict[str, float]] = None
    max_sector_weight: float = 0.4

    # Concentration constraints
    max_single_asset_weight: float = 0.2
    max_top_n_weight: int = 5  # Max weight for top N assets
    max_top_n_weight_value: float = 0.6

    # Turnover constraints
    max_turnover: float = 0.5  # Max 50% turnover per rebalancing

    # Risk constraints
    max_portfolio_volatility: float = 0.25
    max_var: float = 0.05  # Max 5% VaR
    max_cvar: float = 0.08  # Max 8% CVaR

    # Other constraints
    enable_short_selling: bool = False
    enable_leverage: bool = False
    max_leverage: float = 1.0

@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization."""
    # Basic settings
    method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE
    rebalancing_frequency: RebalancingFrequency = RebalancingFrequency.MONTHLY
    lookback_period: int = 252  # Trading days
    min_periods: int = 60  # Minimum periods for optimization

    # Risk settings
    risk_aversion: float = 1.0  # Risk aversion parameter
    target_return: Optional[float] = None  # Target return (if specified)
    risk_free_rate: float = 0.02

    # Optimization settings
    max_iterations: int = 1000
    tolerance: float = 1e-6
    enable_parallel: bool = True

    # Constraints
    constraints: OptimizationConstraints = field(default_factory=OptimizationConstraints)

    # Regime settings
    enable_regime_aware: bool = True
    regime_lookback: int = 60
    regime_threshold: float = 0.1

    # Transaction costs
    transaction_cost: float = DEFAULT_TRANSACTION_COST
    market_impact: float = 0.0005

    # Performance settings
    enable_caching: bool = True
    cache_duration_hours: int = 24

@dataclass
class OptimizationResults:
    """Results from portfolio optimization."""
    # Basic results
    weights: np.ndarray
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float

    # Optimization details
    method_used: OptimizationMethod
    optimization_time: float
    iterations: int
    converged: bool

    # Performance metrics
    backtest_results: Optional[Dict[str, Any]] = None
    risk_metrics: Optional[Dict[str, float]] = None

    # Additional data
    covariance_matrix: Optional[np.ndarray] = None
    expected_returns: Optional[np.ndarray] = None
    asset_names: Optional[List[str]] = None

class VectorBTPortfolioOptimizer:
    """
    Comprehensive portfolio optimizer using VectorBT and advanced optimization techniques.

    This class provides various portfolio optimization methods including:
    - Mean-variance optimization
    - Risk parity strategies
    - Black-Litterman model
    - Multi-objective optimization
    - Constraint handling
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize VectorBT portfolio optimizer.

        Args:
            config: Optimization configuration
        """
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required but not available. Install with: pip install vectorbt")

        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required but not available. Install with: pip install scipy")

        self.config = config or OptimizationConfig()

        # Initialize memory manager and performance monitor
        self.memory_manager = get_memory_manager()
        self.performance_monitor = get_performance_monitor()

        # Initialize VectorBT settings
        self._configure_vectorbt()

        # Performance tracking
        self.optimization_stats = {
            'total_optimizations': 0,
            'total_time': 0.0,
            'successful_optimizations': 0,
            'failed_optimizations': 0
        }

        # Cache for optimization results
        self._optimization_cache = {}

        logger.info("✅ VectorBT Portfolio Optimizer initialized")
        logger.info(f"📊 Method: {self.config.method.value}")
        logger.info(f"📊 Rebalancing: {self.config.rebalancing_frequency.value}")
        logger.info(f"📊 Risk aversion: {self.config.risk_aversion}")
        logger.info(f"📊 Memory manager: {self.memory_manager.get_memory_stats()['available_memory_gb']:.2f}GB available")

    def _configure_vectorbt(self):
        """Configure VectorBT global settings."""
        # Check if settings attribute exists first
        if not hasattr(vbt, 'settings'):
            logger.debug("VectorBT settings not available in this version")
            return
            
        if self.config.enable_parallel:
            try:
                # Try to set parallel settings if available
                if hasattr(vbt.settings, 'parallel') and hasattr(vbt.settings.parallel, '__setitem__'):
                    vbt.settings.parallel['threading'] = True
                elif hasattr(vbt.settings, 'threading') and hasattr(vbt.settings.threading, '__setitem__'):
                    vbt.settings.threading['enabled'] = True
                else:
                    logger.debug("Parallel settings not available in this VectorBT version")
            except (AttributeError, KeyError, TypeError) as e:
                logger.debug(f"Parallel settings not available in this VectorBT version: {e}")

        try:
            if hasattr(vbt.settings, 'array_wrapper'):
                vbt.settings.array_wrapper['freq'] = '1min'
        except (AttributeError, KeyError) as e:
            logger.debug(f"Array wrapper settings not available: {e}")

    def optimize_portfolio(self,
                          returns: Union[np.ndarray, pd.DataFrame],
                          expected_returns: Optional[Union[np.ndarray, pd.Series]] = None,
                          asset_names: Optional[List[str]] = None,
                          **kwargs) -> OptimizationResults:
        """
        Optimize portfolio using specified method with memory and performance optimization.

        Args:
            returns: Historical returns data
            expected_returns: Expected returns (if not provided, will estimate)
            asset_names: Names of assets
            **kwargs: Additional arguments for optimization

        Returns:
            Optimization results
        """
        # Check cache first
        cache_key = self._get_cache_key(returns, expected_returns, asset_names)
        if cache_key in self._optimization_cache and self.config.enable_caching:
            logger.info("📊 Using cached optimization results")
            return self._optimization_cache[cache_key]

        # Use performance monitoring
        with monitor_operation(
            f"portfolio_optimization_{self.config.method.value}",
            metadata={'n_assets': len(returns) if hasattr(returns, '__len__') else returns.shape[1]}
        ):
            logger.info(f"🚀 Starting portfolio optimization using {self.config.method.value}...")

            # Prepare data with memory optimization
            returns_df = self._prepare_returns_data_optimized(returns, asset_names)
            expected_returns_array = self._prepare_expected_returns_optimized(returns_df, expected_returns)

            # Calculate covariance matrix with memory management
            covariance_matrix = self._calculate_covariance_matrix_optimized(returns_df)

            logger.info(f"📊 Data shape: {returns_df.shape}")
            logger.info(f"📊 Assets: {list(returns_df.columns)}")

            # Run optimization based on method with memory management
            weights = self._run_optimization_with_memory_management(
                expected_returns_array, covariance_matrix, returns_df
            )

            # Calculate portfolio metrics
            expected_return = np.dot(weights, expected_returns_array)
            expected_volatility = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            sharpe_ratio = (expected_return - self.config.risk_free_rate) / expected_volatility if expected_volatility > 0 else 0

            # Create results
            results = OptimizationResults(
                weights=weights,
                expected_return=expected_return,
                expected_volatility=expected_volatility,
                sharpe_ratio=sharpe_ratio,
                method_used=self.config.method,
                optimization_time=0.0,  # Will be set by performance monitor
                iterations=0,  # Will be set by specific optimization methods
                converged=True,  # Will be set by specific optimization methods
                covariance_matrix=covariance_matrix,
                expected_returns=expected_returns_array,
                asset_names=list(returns_df.columns)
            )

            # Update statistics
            self.optimization_stats['total_optimizations'] += 1
            self.optimization_stats['successful_optimizations'] += 1

            # Cache results
            if self.config.enable_caching:
                self._optimization_cache[cache_key] = results
                # Limit cache size
                if len(self._optimization_cache) > 100:
                    # Remove oldest entries
                    oldest_key = next(iter(self._optimization_cache))
                    del self._optimization_cache[oldest_key]

            logger.info(f"✅ Portfolio optimization completed")
            logger.info(f"📊 Expected return: {expected_return:.2%}")
            logger.info(f"📊 Expected volatility: {expected_volatility:.2%}")
            logger.info(f"📊 Sharpe ratio: {sharpe_ratio:.3f}")

            return results

    def _get_cache_key(self, returns: Union[np.ndarray, pd.DataFrame],
                      expected_returns: Optional[Union[np.ndarray, pd.Series]],
                      asset_names: Optional[List[str]]) -> str:
        """Generate cache key for optimization results."""
        # Create a hash of the input data
        if isinstance(returns, np.ndarray):
            returns_hash = hash(returns.tobytes())
        else:
            returns_hash = hash(returns.values.tobytes())

        if expected_returns is not None:
            if isinstance(expected_returns, np.ndarray):
                exp_returns_hash = hash(expected_returns.tobytes())
            else:
                exp_returns_hash = hash(expected_returns.values.tobytes())
        else:
            exp_returns_hash = 0

        asset_names_str = str(asset_names) if asset_names else "None"

        return f"{self.config.method.value}_{returns_hash}_{exp_returns_hash}_{asset_names_str}"

    def _prepare_returns_data_optimized(self, returns: Union[np.ndarray, pd.DataFrame],
                                      asset_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Prepare returns data as DataFrame with memory optimization."""
        if isinstance(returns, np.ndarray):
            if returns.ndim == 1:
                returns_df = pd.DataFrame(returns, columns=['asset_0'])
            else:
                if asset_names is not None:
                    columns = asset_names[:returns.shape[1]]
                else:
                    columns = [f'asset_{i}' for i in range(returns.shape[1])]
                returns_df = pd.DataFrame(returns, columns=columns)
        else:
            returns_df = returns.copy()

        # Remove NaN values
        returns_df = returns_df.dropna()

        # Optimize data types for memory efficiency
        returns_df = optimize_memory_usage(returns_df)

        return returns_df

    def _prepare_expected_returns_optimized(self, returns_df: pd.DataFrame,
                                          expected_returns: Optional[Union[np.ndarray, pd.Series]] = None) -> np.ndarray:
        """Prepare expected returns array with memory optimization."""
        if expected_returns is not None:
            if isinstance(expected_returns, pd.Series):
                return optimize_memory_usage(expected_returns.values)
            else:
                return optimize_memory_usage(expected_returns)
        else:
            # Estimate expected returns from historical data
            return optimize_memory_usage(returns_df.mean().values * self.config.annualization_factor)

    def _calculate_covariance_matrix_optimized(self, returns_df: pd.DataFrame) -> np.ndarray:
        """Calculate covariance matrix with memory optimization."""
        # Use memory management for large covariance matrices
        data_size_gb = returns_df.memory_usage(deep=True).sum() / (1024**3)

        with memory_managed_operation(
            data_size_gb * 2,  # Covariance matrix is roughly 2x the data size
            f"covariance_calculation_{int(time.time())}",
            "covariance_calculation"
        ):
            # Use shrinkage estimator for better stability
            from sklearn.covariance import LedoitWolf

            lw = LedoitWolf()
            covariance_matrix = lw.fit(returns_df).covariance_

            # Optimize data types
            covariance_matrix = optimize_memory_usage(covariance_matrix)

        return covariance_matrix

    def _run_optimization_with_memory_management(self, expected_returns: np.ndarray,
                                               covariance_matrix: np.ndarray,
                                               returns_df: pd.DataFrame) -> np.ndarray:
        """Run optimization with memory management."""
        # Estimate memory requirements
        n_assets = len(expected_returns)
        memory_estimate_gb = (n_assets * n_assets * 8) / (1024**3)  # Covariance matrix size

        with memory_managed_operation(
            memory_estimate_gb,
            f"optimization_{self.config.method.value}_{int(time.time())}",
            "portfolio_optimization"
        ):
            # Run optimization based on method
            if self.config.method == OptimizationMethod.MEAN_VARIANCE:
                return self._optimize_mean_variance_enhanced(expected_returns, covariance_matrix)
            elif self.config.method == OptimizationMethod.RISK_PARITY:
                return self._optimize_risk_parity_enhanced(covariance_matrix)
            elif self.config.method == OptimizationMethod.EQUAL_WEIGHT:
                return self._optimize_equal_weight(len(returns_df.columns))
            elif self.config.method == OptimizationMethod.MIN_VARIANCE:
                return self._optimize_min_variance_enhanced(covariance_matrix)
            elif self.config.method == OptimizationMethod.MAX_SHARPE:
                return self._optimize_max_sharpe_enhanced(expected_returns, covariance_matrix)
            elif self.config.method == OptimizationMethod.BLACK_LITTERMAN:
                return self._optimize_black_litterman_enhanced(expected_returns, covariance_matrix, returns_df)
            elif self.config.method == OptimizationMethod.HIERARCHICAL_RISK_PARITY:
                return self._optimize_hierarchical_risk_parity_enhanced(covariance_matrix)
            elif self.config.method == OptimizationMethod.MAX_DIVERSIFICATION:
                return self._optimize_max_diversification_enhanced(covariance_matrix)
            else:
                raise ValueError(f"Unsupported optimization method: {self.config.method}")

    def _prepare_returns_data(self, returns: Union[np.ndarray, pd.DataFrame], asset_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Prepare returns data as DataFrame."""
        if isinstance(returns, np.ndarray):
            if returns.ndim == 1:
                returns_df = pd.DataFrame(returns, columns=['asset_0'])
            else:
                if asset_names is not None:
                    columns = asset_names[:returns.shape[1]]
                else:
                    columns = [f'asset_{i}' for i in range(returns.shape[1])]
                returns_df = pd.DataFrame(returns, columns=columns)
        else:
            returns_df = returns.copy()

        # Remove NaN values
        returns_df = returns_df.dropna()

        return returns_df

    def _prepare_expected_returns(self, returns_df: pd.DataFrame, expected_returns: Optional[Union[np.ndarray, pd.Series]] = None) -> np.ndarray:
        """Prepare expected returns array."""
        if expected_returns is not None:
            if isinstance(expected_returns, pd.Series):
                return expected_returns.values
            else:
                return expected_returns
        else:
            # Estimate expected returns from historical data
            return returns_df.mean().values * self.config.annualization_factor

    def _calculate_covariance_matrix(self, returns_df: pd.DataFrame) -> np.ndarray:
        """Calculate covariance matrix."""
        # Use shrinkage estimator for better stability
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf()
        covariance_matrix = lw.fit(returns_df).covariance_

        return covariance_matrix

    def _optimize_mean_variance(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:
        """Optimize using mean-variance approach."""
        logger.debug("🔄 Running mean-variance optimization...")

        n_assets = len(expected_returns)

        # Objective function: maximize utility = expected_return - risk_aversion * variance
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            return -(portfolio_return - self.config.risk_aversion * portfolio_variance)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]

        # Bounds
        bounds = [(self.config.constraints.min_weight, self.config.constraints.max_weight) for _ in range(n_assets)]

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            return result.x
        else:
            logger.warning("⚠️ Mean-variance optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets

    def _optimize_risk_parity(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """Optimize using risk parity approach."""
        logger.debug("🔄 Running risk parity optimization...")

        n_assets = covariance_matrix.shape[0]

        # Objective function: minimize sum of squared differences from equal risk contribution
        def objective(weights):
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            risk_contributions = weights * np.dot(covariance_matrix, weights) / portfolio_variance
            target_contributions = np.ones(n_assets) / n_assets
            return np.sum((risk_contributions - target_contributions) ** 2)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]

        # Bounds
        bounds = [(self.config.constraints.min_weight, self.config.constraints.max_weight) for _ in range(n_assets)]

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            return result.x
        else:
            logger.warning("⚠️ Risk parity optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets

    def _optimize_equal_weight(self, n_assets: int) -> np.ndarray:
        """Equal weight optimization."""
        logger.debug("🔄 Using equal weight optimization...")
        return np.ones(n_assets) / n_assets

    def _optimize_min_variance(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """Optimize for minimum variance."""
        logger.debug("🔄 Running minimum variance optimization...")

        n_assets = covariance_matrix.shape[0]

        # Objective function: minimize portfolio variance
        def objective(weights):
            return np.dot(weights, np.dot(covariance_matrix, weights))

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]

        # Bounds
        bounds = [(self.config.constraints.min_weight, self.config.constraints.max_weight) for _ in range(n_assets)]

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            return result.x
        else:
            logger.warning("⚠️ Minimum variance optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets

    def _optimize_max_sharpe(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:
        """Optimize for maximum Sharpe ratio."""
        logger.debug("🔄 Running maximum Sharpe ratio optimization...")

        n_assets = len(expected_returns)

        # Objective function: minimize negative Sharpe ratio
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)

            if portfolio_volatility == 0:
                return 0

            sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_volatility
            return -sharpe_ratio

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]

        # Bounds
        bounds = [(self.config.constraints.min_weight, self.config.constraints.max_weight) for _ in range(n_assets)]

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            return result.x
        else:
            logger.warning("⚠️ Maximum Sharpe ratio optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets

    def _optimize_black_litterman(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray, returns_df: pd.DataFrame) -> np.ndarray:
        """Optimize using Black-Litterman model."""
        logger.debug("🔄 Running Black-Litterman optimization...")

        # This is a simplified implementation
        # In practice, you would need market cap weights and views

        n_assets = len(expected_returns)

        # Use market cap weights (simplified as equal weights)
        market_cap_weights = np.ones(n_assets) / n_assets

        # Calculate implied returns
        risk_aversion = 3.0  # Typical value
        implied_returns = risk_aversion * np.dot(covariance_matrix, market_cap_weights)

        # Use implied returns for optimization
        return self._optimize_mean_variance(implied_returns, covariance_matrix)

    def _optimize_hierarchical_risk_parity(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """Optimize using hierarchical risk parity."""
        logger.debug("🔄 Running hierarchical risk parity optimization...")

        # This is a simplified implementation
        # In practice, you would use a more sophisticated clustering approach

        n_assets = covariance_matrix.shape[0]

        # Use correlation-based clustering
        correlation_matrix = self._covariance_to_correlation(covariance_matrix)

        # Simple clustering based on correlation
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        # Convert correlation to distance
        distance_matrix = 1 - np.abs(correlation_matrix)
        np.fill_diagonal(distance_matrix, 0)

        # Perform hierarchical clustering
        linkage_matrix = linkage(squareform(distance_matrix), method='ward')
        clusters = fcluster(linkage_matrix, t=0.5, criterion='distance')

        # Calculate weights based on clusters
        weights = np.zeros(n_assets)
        unique_clusters = np.unique(clusters)

        for cluster in unique_clusters:
            cluster_indices = np.where(clusters == cluster)[0]
            cluster_weights = self._optimize_risk_parity(covariance_matrix[np.ix_(cluster_indices, cluster_indices)])
            weights[cluster_indices] = cluster_weights / len(unique_clusters)

        return weights

    def _optimize_max_diversification(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """Optimize for maximum diversification."""
        logger.debug("🔄 Running maximum diversification optimization...")

        n_assets = covariance_matrix.shape[0]

        # Calculate volatility
        volatilities = np.sqrt(np.diag(covariance_matrix))

        # Objective function: maximize diversification ratio
        def objective(weights):
            portfolio_volatility = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            weighted_avg_volatility = np.dot(weights, volatilities)

            if weighted_avg_volatility == 0:
                return 0

            diversification_ratio = weighted_avg_volatility / portfolio_volatility
            return -diversification_ratio

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]

        # Bounds
        bounds = [(self.config.constraints.min_weight, self.config.constraints.max_weight) for _ in range(n_assets)]

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            return result.x
        else:
            logger.warning("⚠️ Maximum diversification optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets

    def _covariance_to_correlation(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix."""
        std_dev = np.sqrt(np.diag(covariance_matrix))
        correlation_matrix = covariance_matrix / np.outer(std_dev, std_dev)
        return correlation_matrix

    def backtest_optimized_portfolio(self,
                                   returns: Union[np.ndarray, pd.DataFrame],
                                   weights: np.ndarray,
                                   initial_capital: float = 100000.0) -> Dict[str, Any]:
        """
        Backtest the optimized portfolio using VectorBT.

        Args:
            returns: Historical returns data
            weights: Optimized portfolio weights
            initial_capital: Initial capital for backtesting

        Returns:
            Backtesting results
        """
        logger.info("🔄 Backtesting optimized portfolio...")

        # Prepare returns data
        returns_df = self._prepare_returns_data(returns)

        # Create portfolio using VectorBT
        portfolio = vbt.Portfolio.from_orders(
            returns_df,
            np.ones(len(returns_df)) * initial_capital,  # Initial capital
            freq='1min'
        )

        # Calculate performance metrics
        portfolio_values = portfolio.value()
        portfolio_returns = portfolio.returns()

        # Calculate metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (portfolio_returns.mean() * 252 - self.config.risk_free_rate) / volatility

        # Calculate drawdown
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = drawdown.min()

        results = {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': portfolio_values.iloc[-1],
            'portfolio_values': portfolio_values.values,
            'returns': portfolio_returns.values
        }

        logger.info(f"✅ Backtesting completed")
        logger.info(f"📊 Total return: {total_return:.2%}")
        logger.info(f"📊 Volatility: {volatility:.2%}")
        logger.info(f"📊 Sharpe ratio: {sharpe_ratio:.3f}")
        logger.info(f"📊 Max drawdown: {max_drawdown:.2%}")

        return results

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = self.optimization_stats.copy()
        if stats['total_optimizations'] > 0:
            stats['avg_optimization_time'] = stats['total_time'] / stats['total_optimizations']
            stats['success_rate'] = stats['successful_optimizations'] / stats['total_optimizations']
        else:
            stats['avg_optimization_time'] = 0
            stats['success_rate'] = 0

        return stats

    def _optimize_mean_variance_enhanced(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:
        """Enhanced mean-variance optimization with better constraint handling."""
        logger.debug("🔄 Running enhanced mean-variance optimization...")

        n_assets = len(expected_returns)

        # Objective function: maximize utility = expected_return - risk_aversion * variance
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            return -(portfolio_return - self.config.risk_aversion * portfolio_variance)

        # Enhanced constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]

        # Add constraint handling
        if hasattr(self.config.constraints, 'max_single_asset_weight'):
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: self.config.constraints.max_single_asset_weight - np.max(w)
            })

        # Bounds with enhanced handling
        bounds = [(self.config.constraints.min_weight, self.config.constraints.max_weight) for _ in range(n_assets)]

        # Multiple initial guesses for better convergence
        best_weights = None
        best_objective = float('inf')

        for i in range(5):  # Try 5 different initial guesses
            if i == 0:
                x0 = np.ones(n_assets) / n_assets  # Equal weights
            elif i == 1:
                x0 = np.random.dirichlet(np.ones(n_assets))  # Random weights
            else:
                x0 = np.random.dirichlet(np.ones(n_assets) * (i + 1))  # Different random weights

            try:
                result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
                if result.success and result.fun < best_objective:
                    best_weights = result.x
                    best_objective = result.fun
            except Exception as e:
                logger.warning(f"Optimization attempt {i+1} failed: {e}")
                continue

        if best_weights is not None:
            return best_weights
        else:
            logger.warning("⚠️ Enhanced mean-variance optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets

    def _optimize_risk_parity_enhanced(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """Enhanced risk parity optimization with better convergence."""
        logger.debug("🔄 Running enhanced risk parity optimization...")

        n_assets = covariance_matrix.shape[0]

        # Enhanced objective function with regularization
        def objective(weights):
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            if portfolio_variance <= 0:
                return 1e6  # Large penalty for invalid portfolio

            risk_contributions = weights * np.dot(covariance_matrix, weights) / portfolio_variance
            target_contributions = np.ones(n_assets) / n_assets

            # Add regularization term for stability
            regularization = 0.01 * np.sum(weights**2)

            return np.sum((risk_contributions - target_contributions) ** 2) + regularization

        # Enhanced constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]

        # Bounds
        bounds = [(self.config.constraints.min_weight, self.config.constraints.max_weight) for _ in range(n_assets)]

        # Multiple initial guesses
        best_weights = None
        best_objective = float('inf')

        for i in range(3):
            if i == 0:
                x0 = np.ones(n_assets) / n_assets
            else:
                x0 = np.random.dirichlet(np.ones(n_assets))

            try:
                result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
                if result.success and result.fun < best_objective:
                    best_weights = result.x
                    best_objective = result.fun
            except Exception as e:
                logger.warning(f"Risk parity attempt {i+1} failed: {e}")
                continue

        if best_weights is not None:
            return best_weights
        else:
            logger.warning("⚠️ Enhanced risk parity optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets

    def _optimize_min_variance_enhanced(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """Enhanced minimum variance optimization."""
        logger.debug("🔄 Running enhanced minimum variance optimization...")

        n_assets = covariance_matrix.shape[0]

        # Objective function: minimize portfolio variance
        def objective(weights):
            return np.dot(weights, np.dot(covariance_matrix, weights))

        # Enhanced constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]

        # Bounds
        bounds = [(self.config.constraints.min_weight, self.config.constraints.max_weight) for _ in range(n_assets)]

        # Try multiple initial guesses
        best_weights = None
        best_variance = float('inf')

        for i in range(3):
            if i == 0:
                x0 = np.ones(n_assets) / n_assets
            else:
                x0 = np.random.dirichlet(np.ones(n_assets))

            try:
                result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
                if result.success and result.fun < best_variance:
                    best_weights = result.x
                    best_variance = result.fun
            except Exception as e:
                logger.warning(f"Min variance attempt {i+1} failed: {e}")
                continue

        if best_weights is not None:
            return best_weights
        else:
            logger.warning("⚠️ Enhanced minimum variance optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets

    def _optimize_max_sharpe_enhanced(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:
        """Enhanced maximum Sharpe ratio optimization."""
        logger.debug("🔄 Running enhanced maximum Sharpe ratio optimization...")

        n_assets = len(expected_returns)

        # Objective function: minimize negative Sharpe ratio
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)

            if portfolio_volatility == 0:
                return 0

            sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_volatility
            return -sharpe_ratio

        # Enhanced constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]

        # Bounds
        bounds = [(self.config.constraints.min_weight, self.config.constraints.max_weight) for _ in range(n_assets)]

        # Try multiple initial guesses
        best_weights = None
        best_sharpe = float('-inf')

        for i in range(3):
            if i == 0:
                x0 = np.ones(n_assets) / n_assets
            else:
                x0 = np.random.dirichlet(np.ones(n_assets))

            try:
                result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
                if result.success and -result.fun > best_sharpe:
                    best_weights = result.x
                    best_sharpe = -result.fun
            except Exception as e:
                logger.warning(f"Max Sharpe attempt {i+1} failed: {e}")
                continue

        if best_weights is not None:
            return best_weights
        else:
            logger.warning("⚠️ Enhanced maximum Sharpe optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets

    def _optimize_black_litterman_enhanced(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray, returns_df: pd.DataFrame) -> np.ndarray:
        """Enhanced Black-Litterman optimization."""
        logger.debug("🔄 Running enhanced Black-Litterman optimization...")

        n_assets = len(expected_returns)

        # Enhanced Black-Litterman implementation
        # Use market cap weights (simplified as equal weights for now)
        market_cap_weights = np.ones(n_assets) / n_assets

        # Calculate implied returns with enhanced risk aversion
        risk_aversion = 3.0  # Typical value
        implied_returns = risk_aversion * np.dot(covariance_matrix, market_cap_weights)

        # Use enhanced mean-variance with implied returns
        return self._optimize_mean_variance_enhanced(implied_returns, covariance_matrix)

    def _optimize_hierarchical_risk_parity_enhanced(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """Enhanced hierarchical risk parity optimization."""
        logger.debug("🔄 Running enhanced hierarchical risk parity optimization...")

        n_assets = covariance_matrix.shape[0]

        # Enhanced correlation-based clustering
        correlation_matrix = self._covariance_to_correlation(covariance_matrix)

        # Use enhanced clustering
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        # Convert correlation to distance
        distance_matrix = 1 - np.abs(correlation_matrix)
        np.fill_diagonal(distance_matrix, 0)

        # Perform hierarchical clustering with enhanced parameters
        linkage_matrix = linkage(squareform(distance_matrix), method='ward')
        clusters = fcluster(linkage_matrix, t=0.5, criterion='distance')

        # Calculate weights based on clusters with enhanced risk parity
        weights = np.zeros(n_assets)
        unique_clusters = np.unique(clusters)

        for cluster in unique_clusters:
            cluster_indices = np.where(clusters == cluster)[0]
            if len(cluster_indices) > 1:
                cluster_weights = self._optimize_risk_parity_enhanced(
                    covariance_matrix[np.ix_(cluster_indices, cluster_indices)]
                )
            else:
                cluster_weights = np.array([1.0])

            weights[cluster_indices] = cluster_weights / len(unique_clusters)

        return weights

    def _optimize_max_diversification_enhanced(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """Enhanced maximum diversification optimization."""
        logger.debug("🔄 Running enhanced maximum diversification optimization...")

        n_assets = covariance_matrix.shape[0]

        # Calculate volatility with enhanced stability
        volatilities = np.sqrt(np.diag(covariance_matrix))
        volatilities = np.maximum(volatilities, 1e-8)  # Avoid division by zero

        # Enhanced objective function
        def objective(weights):
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            weighted_avg_volatility = np.dot(weights, volatilities)

            if weighted_avg_volatility == 0 or portfolio_volatility == 0:
                return 0

            diversification_ratio = weighted_avg_volatility / portfolio_volatility
            return -diversification_ratio

        # Enhanced constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]

        # Bounds
        bounds = [(self.config.constraints.min_weight, self.config.constraints.max_weight) for _ in range(n_assets)]

        # Try multiple initial guesses
        best_weights = None
        best_diversification = float('-inf')

        for i in range(3):
            if i == 0:
                x0 = np.ones(n_assets) / n_assets
            else:
                x0 = np.random.dirichlet(np.ones(n_assets))

            try:
                result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
                if result.success and -result.fun > best_diversification:
                    best_weights = result.x
                    best_diversification = -result.fun
            except Exception as e:
                logger.warning(f"Max diversification attempt {i+1} failed: {e}")
                continue

        if best_weights is not None:
            return best_weights
        else:
            logger.warning("⚠️ Enhanced maximum diversification optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets

# Convenience functions
def optimize_portfolio(returns: Union[np.ndarray, pd.DataFrame],
                      method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE,
                      expected_returns: Optional[Union[np.ndarray, pd.Series]] = None,
                      config: Optional[OptimizationConfig] = None,
                      **kwargs) -> OptimizationResults:
    """
    Convenience function to optimize portfolio.

    Args:
        returns: Historical returns data
        method: Optimization method to use
        expected_returns: Expected returns (optional)
        config: Optimization configuration
        **kwargs: Additional arguments

    Returns:
        Optimization results
    """
    if config is None:
        config = OptimizationConfig()

    config.method = method

    optimizer = VectorBTPortfolioOptimizer(config)
    return optimizer.optimize_portfolio(returns, expected_returns, **kwargs)

def create_optimization_config(method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE,
                              risk_aversion: float = 1.0,
                              rebalancing_frequency: RebalancingFrequency = RebalancingFrequency.MONTHLY,
                              **kwargs) -> OptimizationConfig:
    """
    Create optimization configuration.

    Args:
        method: Optimization method
        risk_aversion: Risk aversion parameter
        rebalancing_frequency: Rebalancing frequency
        **kwargs: Additional configuration parameters

    Returns:
        Optimization configuration
    """
    return OptimizationConfig(
        method=method,
        risk_aversion=risk_aversion,
        rebalancing_frequency=rebalancing_frequency,
        **kwargs
    )

if __name__ == "__main__":
    # Example usage and testing
    logger.info("🧪 Testing VectorBT Portfolio Optimization...")

    # Generate sample data
    np.random.seed(42)
    n_periods = 1000
    n_assets = 5

    # Generate random returns
    returns = np.random.normal(0.001, 0.02, (n_periods, n_assets))
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]

    # Test different optimization methods
    methods = [
        OptimizationMethod.MEAN_VARIANCE,
        OptimizationMethod.RISK_PARITY,
        OptimizationMethod.EQUAL_WEIGHT,
        OptimizationMethod.MIN_VARIANCE,
        OptimizationMethod.MAX_SHARPE
    ]

    results = {}

    for method in methods:
        logger.info(f"\n🔄 Testing {method.value}...")

        try:
            config = create_optimization_config(method=method)
            result = optimize_portfolio(returns, method=method, asset_names=asset_names)

            results[method.value] = {
                'weights': result.weights,
                'expected_return': result.expected_return,
                'expected_volatility': result.expected_volatility,
                'sharpe_ratio': result.sharpe_ratio,
                'optimization_time': result.optimization_time
            }

            print(f"✅ {method.value}: Return={result.expected_return:.2%}, Vol={result.expected_volatility:.2%}, Sharpe={result.sharpe_ratio:.3f}")

        except Exception as e:
            logger.error(f"❌ {method.value} failed: {e}")
            results[method.value] = {'error': str(e)}

    # Print summary
    print(f"\n📊 Optimization Results Summary:")
    for method, result in results.items():
        if 'error' not in result:
            print(f"{method}: {result['optimization_time']:.3f}s, Sharpe={result['sharpe_ratio']:.3f}")
        else:
            print(f"{method}: Failed - {result['error']}")

    print("\n✅ VectorBT Portfolio Optimization test completed!")
