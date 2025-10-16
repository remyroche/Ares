"""
Dynamic Target and Horizon Optimization for Multi-Horizon Profit Labeling

This module provides data-driven discovery and optimization of profit targets and
time horizons based on actual market patterns rather than fixed heuristics.

Key Optimization Components:
1. Data-Driven Target Discovery from Price Movement Patterns
2. Optimal Time Horizon Discovery for Each Target
3. Joint Target-Horizon Optimization
4. Multi-Objective Optimization (Hit Rate, Sharpe, Drawdown, Information Ratio)
5. Market Condition Adaptive Target Selection
6. Clustering-Based Target Identification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
from datetime import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optimization imports
from scipy.optimize import minimize, differential_evolution, basinhopping
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Optional optimization libraries
try:
    from skopt import gp_minimize, forest_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from src.utils.logger import get_logger
from src.training.steps.pre_training.multi_horizon_profit_labeler import MultiHorizonConfig

class OptimizationObjective(Enum):
    """Enumeration of optimization objectives."""
    HIT_RATE = "hit_rate"
    SHARPE_RATIO = "sharpe_ratio"
    INFORMATION_RATIO = "information_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MULTI_OBJECTIVE = "multi_objective"

class OptimizationMethod(Enum):
    """Enumeration of optimization methods."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    BASIN_HOPPING = "basin_hopping"
    GENETIC_ALGORITHM = "genetic_algorithm"
    OPTUNA_TPE = "optuna_tpe"

class TargetDiscoveryMethod(Enum):
    """Enumeration of target discovery methods."""
    PRICE_MOVEMENT_ANALYSIS = "price_movement_analysis"
    CLUSTERING_BASED = "clustering_based"
    STATISTICAL_LEVELS = "statistical_levels"
    SUPPORT_RESISTANCE = "support_resistance"
    VOLATILITY_SCALED = "volatility_scaled"
    MULTI_METHOD = "multi_method"

@dataclass
class DynamicOptimizationConfig:
    """Configuration for dynamic target and horizon optimization."""
    # Target discovery parameters
    target_discovery_method: TargetDiscoveryMethod = TargetDiscoveryMethod.MULTI_METHOD
    min_target_frequency: float = 0.05  # Minimum 5% hit rate
    max_target_frequency: float = 0.95  # Maximum 95% hit rate
    n_target_candidates: int = 20
    target_range: Tuple[float, float] = (0.001, 0.050)  # 0.1% to 5.0%

    # Horizon discovery parameters
    min_horizon: int = 1  # Minimum 1 period (5 minutes)
    max_horizon: int = 20  # Maximum 20 periods (100 minutes)
    horizon_step: int = 1

    # Optimization parameters
    optimization_method: OptimizationMethod = OptimizationMethod.BAYESIAN_OPTIMIZATION
    optimization_objective: OptimizationObjective = OptimizationObjective.MULTI_OBJECTIVE
    n_optimization_trials: int = 100
    optimization_timeout: int = 3600  # 1 hour

    # Multi-objective weights
    hit_rate_weight: float = 0.3
    sharpe_weight: float = 0.3
    drawdown_weight: float = 0.2
    information_weight: float = 0.2

    # Clustering parameters (for clustering-based discovery)
    n_clusters_range: Tuple[int, int] = (3, 10)
    clustering_features: List[str] = field(default_factory=lambda: [
        'price_change', 'volatility', 'volume_change', 'momentum'
    ])

    # Statistical parameters
    statistical_percentiles: List[float] = field(default_factory=lambda: [
        0.25, 0.5, 0.75, 0.9, 0.95, 0.99
    ])

    # Validation parameters
    validation_split: float = 0.3
    min_validation_samples: int = 100
    cross_validation_folds: int = 3

    # Performance constraints
    min_sharpe_ratio: float = 0.0
    max_drawdown_threshold: float = 0.20  # 20% maximum drawdown
    min_hit_rate: float = 0.1

    # Adaptive parameters
    market_condition_adaptive: bool = True
    regime_specific_optimization: bool = True

    # Parallel processing
    n_jobs: int = -1

@dataclass
class TargetHorizonCandidate:
    """Container for target-horizon candidate."""
    target: float
    horizon: int
    hit_rate: float
    sharpe_ratio: float
    max_drawdown: float
    information_ratio: float
    profit_factor: float
    total_trades: int
    avg_return: float
    std_return: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationResult:
    """Result container for target-horizon optimization."""
    optimal_targets: Dict[str, float]
    optimal_horizons: Dict[str, int]
    objective_score: float
    performance_metrics: Dict[str, float]
    validation_scores: Dict[str, float]
    candidate_results: List[TargetHorizonCandidate]
    optimization_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class DynamicTargetDiscovery:
    """
    Discover optimal profit targets from data patterns.

    This class analyzes historical price movements to identify natural profit
    levels that provide good risk/reward characteristics.
    """

    def __init__(self, config: Optional[DynamicOptimizationConfig] = None):
        """Initialize dynamic target discovery."""
        self.config = config or DynamicOptimizationConfig()
        self.logger = get_logger('DynamicTargetDiscovery')

        # Discovery state
        self.discovered_targets: List[float] = []
        self.target_statistics: Dict[float, Dict[str, float]] = {}

        self.logger.info('🎯 Dynamic Target Discovery initialized')
        self.logger.info(f'   → Discovery method: {self.config.target_discovery_method.value}')

    def discover_optimal_targets(self, market_data: pd.DataFrame) -> List[float]:
        """
        Discover optimal profit targets from market data.

        Args:
            market_data: OHLCV market data

        Returns:
            List of optimal profit target percentages
        """
        self.logger.info('🔍 Discovering optimal profit targets')

        if len(market_data) < 100:
            self.logger.warning('⚠️ Insufficient data for target discovery')
            return self._get_default_targets()

        # Select discovery method
        if self.config.target_discovery_method == TargetDiscoveryMethod.MULTI_METHOD:
            targets = self._multi_method_discovery(market_data)
        elif self.config.target_discovery_method == TargetDiscoveryMethod.PRICE_MOVEMENT_ANALYSIS:
            targets = self._price_movement_discovery(market_data)
        elif self.config.target_discovery_method == TargetDiscoveryMethod.CLUSTERING_BASED:
            targets = self._clustering_based_discovery(market_data)
        elif self.config.target_discovery_method == TargetDiscoveryMethod.STATISTICAL_LEVELS:
            targets = self._statistical_levels_discovery(market_data)
        elif self.config.target_discovery_method == TargetDiscoveryMethod.VOLATILITY_SCALED:
            targets = self._volatility_scaled_discovery(market_data)
        else:
            targets = self._price_movement_discovery(market_data)

        # Filter and validate targets
        valid_targets = self._validate_targets(targets, market_data)

        self.discovered_targets = valid_targets
        self.logger.info(f'✅ Discovered {len(valid_targets)} optimal targets')

        return valid_targets

    def _multi_method_discovery(self, market_data: pd.DataFrame) -> List[float]:
        """Combine multiple discovery methods."""
        all_targets = []

        # Price movement analysis
        try:
            price_targets = self._price_movement_discovery(market_data)
            all_targets.extend(price_targets)
        except Exception as e:
            self.logger.warning(f'Price movement discovery failed: {e}')

        # Statistical levels
        try:
            stat_targets = self._statistical_levels_discovery(market_data)
            all_targets.extend(stat_targets)
        except Exception as e:
            self.logger.warning(f'Statistical levels discovery failed: {e}')

        # Volatility scaled
        try:
            vol_targets = self._volatility_scaled_discovery(market_data)
            all_targets.extend(vol_targets)
        except Exception as e:
            self.logger.warning(f'Volatility scaled discovery failed: {e}')

        # Remove duplicates and sort
        unique_targets = sorted(list(set(all_targets)))

        # Select diverse set of targets
        if len(unique_targets) > self.config.n_target_candidates:
            # Select evenly spaced targets
            indices = np.linspace(0, len(unique_targets) - 1, self.config.n_target_candidates, dtype=int)
            unique_targets = [unique_targets[i] for i in indices]

        return unique_targets

    def _price_movement_discovery(self, market_data: pd.DataFrame) -> List[float]:
        """Discover targets based on actual price movement patterns."""
        if 'close' not in market_data.columns:
            return self._get_default_targets()

        prices = market_data['close']
        returns = prices.pct_change().dropna()

        # Calculate forward-looking returns for different horizons
        forward_returns = {}
        for horizon in range(1, min(21, len(returns) // 10)):  # Up to 20 periods or 10% of data
            forward_returns[horizon] = returns.rolling(horizon).sum().shift(-horizon)

        # Find natural profit levels based on return distributions
        targets = []

        for horizon, fwd_returns in forward_returns.items():
            valid_returns = fwd_returns.dropna()
            if len(valid_returns) < 50:
                continue

            # Find percentiles of positive returns
            positive_returns = valid_returns[valid_returns > 0]
            if len(positive_returns) < 20:
                continue

            # Extract target levels from return distribution
            percentiles = [0.25, 0.5, 0.75, 0.9, 0.95]
            for pct in percentiles:
                target = positive_returns.quantile(pct)
                if self.config.target_range[0] <= target <= self.config.target_range[1]:
                    targets.append(target)

        # Remove duplicates and sort
        targets = sorted(list(set(targets)))

        return targets[:self.config.n_target_candidates]

    def _clustering_based_discovery(self, market_data: pd.DataFrame) -> List[float]:
        """Discover targets using clustering of market conditions."""
        if len(market_data) < 100:
            return self._get_default_targets()

        # Engineer features for clustering
        features_df = self._engineer_clustering_features(market_data)

        if features_df.empty:
            return self._get_default_targets()

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df.fillna(0))

        # Find optimal number of clusters
        best_n_clusters = self._find_optimal_clusters(features_scaled)

        # Perform clustering
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)

        # Analyze returns for each cluster
        returns = market_data['close'].pct_change().shift(-1).fillna(0)
        targets = []

        for cluster_id in range(best_n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_returns = returns[cluster_mask]

            if len(cluster_returns) > 20:
                # Find representative target for this cluster
                positive_returns = cluster_returns[cluster_returns > 0]
                if len(positive_returns) > 10:
                    # Use 75th percentile as target
                    target = positive_returns.quantile(0.75)
                    if self.config.target_range[0] <= target <= self.config.target_range[1]:
                        targets.append(target)

        return sorted(targets)

    def _statistical_levels_discovery(self, market_data: pd.DataFrame) -> List[float]:
        """Discover targets based on statistical levels."""
        if 'close' not in market_data.columns:
            return self._get_default_targets()

        returns = market_data['close'].pct_change().dropna()

        # Calculate various statistical measures
        targets = []

        # Standard deviation multiples
        std = returns.std()
        for multiplier in [0.5, 1.0, 1.5, 2.0, 2.5]:
            target = std * multiplier
            if self.config.target_range[0] <= target <= self.config.target_range[1]:
                targets.append(target)

        # Percentiles of absolute returns
        abs_returns = abs(returns)
        for pct in self.config.statistical_percentiles:
            target = abs_returns.quantile(pct)
            if self.config.target_range[0] <= target <= self.config.target_range[1]:
                targets.append(target)

        # VaR-based targets
        try:
            for confidence in [0.95, 0.99]:
                var = abs(returns.quantile(1 - confidence))
                if self.config.target_range[0] <= var <= self.config.target_range[1]:
                    targets.append(var)
        except Exception:
            pass

        return sorted(list(set(targets)))

    def _volatility_scaled_discovery(self, market_data: pd.DataFrame) -> List[float]:
        """Discover targets scaled by volatility."""
        if 'close' not in market_data.columns:
            return self._get_default_targets()

        returns = market_data['close'].pct_change().dropna()

        # Calculate rolling volatility
        volatility = returns.rolling(20).std()

        # Create volatility-scaled targets
        targets = []
        base_targets = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # Multiples of volatility

        for multiplier in base_targets:
            # Use median volatility for scaling
            median_vol = volatility.median()
            if not np.isnan(median_vol) and median_vol > 0:
                target = multiplier * median_vol
                if self.config.target_range[0] <= target <= self.config.target_range[1]:
                    targets.append(target)

        return targets

    def _engineer_clustering_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for clustering analysis."""
        features = pd.DataFrame(index=market_data.index)

        if 'close' not in market_data.columns:
            return features

        prices = market_data['close']
        returns = prices.pct_change()

        # Price change features
        features['price_change'] = returns
        features['abs_price_change'] = abs(returns)

        # Volatility features
        features['volatility'] = returns.rolling(20).std()
        features['volatility_short'] = returns.rolling(5).std()

        # Momentum features
        features['momentum_5'] = prices / prices.shift(5) - 1
        features['momentum_10'] = prices / prices.shift(10) - 1

        # Volume features (if available)
        if 'volume' in market_data.columns:
            volume = market_data['volume']
            features['volume_change'] = volume.pct_change()
            features['volume_ratio'] = volume / volume.rolling(20).mean()

        # Range features (if OHLC available)
        if all(col in market_data.columns for col in ['high', 'low']):
            features['range'] = (market_data['high'] - market_data['low']) / prices
            features['price_position'] = (prices - market_data['low']) / (market_data['high'] - market_data['low'])

        return features

    def _find_optimal_clusters(self, features: np.ndarray) -> int:
        """Find optimal number of clusters using silhouette score."""
        min_clusters, max_clusters = self.config.n_clusters_range
        best_score = -1
        best_n_clusters = min_clusters

        for n_clusters in range(min_clusters, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(features)

                if len(set(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette score
                    score = silhouette_score(features, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters

            except Exception:
                continue

        return best_n_clusters

    def _validate_targets(self, targets: List[float], market_data: pd.DataFrame) -> List[float]:
        """Validate discovered targets against frequency constraints."""
        if not targets or 'close' not in market_data.columns:
            return self._get_default_targets()

        returns = market_data['close'].pct_change().shift(-1).dropna()
        valid_targets = []

        for target in targets:
            # Calculate hit rate for this target
            hits = (returns >= target).sum()
            hit_rate = hits / len(returns)

            # Check if hit rate is within acceptable range
            if self.config.min_target_frequency <= hit_rate <= self.config.max_target_frequency:
                valid_targets.append(target)

                # Store statistics
                self.target_statistics[target] = {
                    'hit_rate': hit_rate,
                    'total_hits': hits,
                    'avg_return_when_hit': returns[returns >= target].mean(),
                    'std_return_when_hit': returns[returns >= target].std()
                }

        # If no valid targets found, return defaults
        if not valid_targets:
            valid_targets = self._get_default_targets()

        return valid_targets[:self.config.n_target_candidates]

    def _get_default_targets(self) -> List[float]:
        """Get default target values."""
        return [0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020]

class DynamicHorizonOptimizer:
    """
    Optimize time horizons for each profit target.

    This class finds optimal time horizons that maximize hit rate while
    minimizing drawdown for each profit target.
    """

    def __init__(self, config: Optional[DynamicOptimizationConfig] = None):
        """Initialize dynamic horizon optimizer."""
        self.config = config or DynamicOptimizationConfig()
        self.logger = get_logger('DynamicHorizonOptimizer')

        # Optimization state
        self.optimal_horizons: Dict[float, int] = {}
        self.horizon_statistics: Dict[Tuple[float, int], Dict[str, float]] = {}

        self.logger.info('⏰ Dynamic Horizon Optimizer initialized')

    def discover_optimal_horizons(self,
                                targets: List[float],
                                market_data: pd.DataFrame) -> Dict[float, int]:
        """
        Discover optimal time horizons for each target.

        Args:
            targets: List of profit targets
            market_data: OHLCV market data

        Returns:
            Dictionary mapping targets to optimal horizons
        """
        self.logger.info(f'⏰ Discovering optimal horizons for {len(targets)} targets')

        if 'close' not in market_data.columns or len(market_data) < 100:
            return {target: 5 for target in targets}  # Default 5 periods

        optimal_horizons = {}

        for target in targets:
            self.logger.info(f'   → Optimizing horizon for target {target:.3f}')

            best_horizon = self._optimize_horizon_for_target(target, market_data)
            optimal_horizons[target] = best_horizon

            self.logger.info(f'     ✅ Optimal horizon: {best_horizon} periods')

        self.optimal_horizons = optimal_horizons
        return optimal_horizons

    def _optimize_horizon_for_target(self, target: float, market_data: pd.DataFrame) -> int:
        """Optimize horizon for a specific target."""
        prices = market_data['close']

        # Test different horizons
        horizon_scores = {}

        for horizon in range(self.config.min_horizon, self.config.max_horizon + 1, self.config.horizon_step):
            score = self._evaluate_target_horizon_combination(target, horizon, prices)
            horizon_scores[horizon] = score

            # Store statistics
            self.horizon_statistics[(target, horizon)] = score

        # Select horizon with best score
        if horizon_scores:
            best_horizon = max(horizon_scores.keys(), key=lambda h: horizon_scores[h]['composite_score'])
            return best_horizon

        return 5  # Default fallback

    def _evaluate_target_horizon_combination(self,
                                           target: float,
                                           horizon: int,
                                           prices: pd.Series) -> Dict[str, float]:
        """Evaluate a specific target-horizon combination."""
        results = {
            'hit_rate': 0.0,
            'avg_time_to_hit': horizon,
            'max_drawdown': 0.0,
            'avg_return': 0.0,
            'std_return': 0.0,
            'sharpe_ratio': 0.0,
            'composite_score': 0.0
        }

        if len(prices) < horizon + 50:
            return results

        # Calculate forward returns and hit statistics
        hits = []
        times_to_hit = []
        drawdowns = []
        returns = []

        for i in range(len(prices) - horizon):
            entry_price = prices.iloc[i]
            target_price = entry_price * (1 + target)

            # Look for hit within horizon
            future_prices = prices.iloc[i+1:i+horizon+1]

            hit = False
            time_to_hit = horizon
            max_adverse = 0.0

            for j, price in enumerate(future_prices):
                if price >= target_price:
                    hit = True
                    time_to_hit = j + 1
                    break

                # Calculate adverse excursion
                adverse = (entry_price - price) / entry_price
                max_adverse = max(max_adverse, adverse)

            hits.append(hit)
            if hit:
                times_to_hit.append(time_to_hit)

            # Calculate return
            final_price = future_prices.iloc[-1] if len(future_prices) > 0 else entry_price
            ret = (final_price - entry_price) / entry_price
            returns.append(ret)
            drawdowns.append(max_adverse)

        if not hits:
            return results

        # Calculate metrics
        results['hit_rate'] = np.mean(hits)
        results['avg_time_to_hit'] = np.mean(times_to_hit) if times_to_hit else horizon
        results['max_drawdown'] = np.max(drawdowns) if drawdowns else 0.0
        results['avg_return'] = np.mean(returns)
        results['std_return'] = np.std(returns)

        # Sharpe ratio
        if results['std_return'] > 0:
            results['sharpe_ratio'] = results['avg_return'] / results['std_return']

        # Composite score (weighted combination of metrics)
        hit_rate_score = results['hit_rate']
        sharpe_score = max(0, min(2, results['sharpe_ratio'] + 1)) / 2  # Normalize to 0-1
        drawdown_penalty = results['max_drawdown'] * 2  # Penalty for drawdown
        time_bonus = max(0, 1 - results['avg_time_to_hit'] / horizon)  # Bonus for faster hits

        results['composite_score'] = (
            hit_rate_score * 0.4 +
            sharpe_score * 0.3 +
            time_bonus * 0.2 -
            drawdown_penalty * 0.1
        )

        return results

class JointTargetHorizonOptimizer:
    """
    Joint optimization of targets and horizons using advanced optimization methods.

    This class performs multi-objective optimization to find the best combination
    of profit targets and time horizons.
    """

    def __init__(self, config: Optional[DynamicOptimizationConfig] = None):
        """Initialize joint optimizer."""
        self.config = config or DynamicOptimizationConfig()
        self.logger = get_logger('JointTargetHorizonOptimizer')

        # Optimization state
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_result: Optional[OptimizationResult] = None

        self.logger.info('🎯⏰ Joint Target-Horizon Optimizer initialized')
        self.logger.info(f'   → Method: {self.config.optimization_method.value}')
        self.logger.info(f'   → Objective: {self.config.optimization_objective.value}')

    def optimize_target_horizon_combinations(self, market_data: pd.DataFrame) -> OptimizationResult:
        """
        Optimize target-horizon combinations using multi-objective optimization.

        Args:
            market_data: OHLCV market data

        Returns:
            OptimizationResult with optimal parameters
        """
        self.logger.info('🚀 Starting joint target-horizon optimization')

        if len(market_data) < 200:
            self.logger.warning('⚠️ Insufficient data for joint optimization')
            return self._create_default_result()

        # Split data for validation
        split_idx = int(len(market_data) * (1 - self.config.validation_split))
        train_data = market_data.iloc[:split_idx]
        val_data = market_data.iloc[split_idx:]

        # Discover candidate targets
        target_discovery = DynamicTargetDiscovery(self.config)
        candidate_targets = target_discovery.discover_optimal_targets(train_data)

        if not candidate_targets:
            self.logger.warning('⚠️ No candidate targets discovered')
            return self._create_default_result()

        # Run optimization
        if self.config.optimization_method == OptimizationMethod.BAYESIAN_OPTIMIZATION and SKOPT_AVAILABLE:
            result = self._bayesian_optimization(candidate_targets, train_data, val_data)
        elif self.config.optimization_method == OptimizationMethod.OPTUNA_TPE and OPTUNA_AVAILABLE:
            result = self._optuna_optimization(candidate_targets, train_data, val_data)
        elif self.config.optimization_method == OptimizationMethod.DIFFERENTIAL_EVOLUTION:
            result = self._differential_evolution_optimization(candidate_targets, train_data, val_data)
        else:
            result = self._grid_search_optimization(candidate_targets, train_data, val_data)

        self.best_result = result
        self.logger.info(f'✅ Joint optimization completed with score: {result.objective_score:.4f}')

        return result

    def _bayesian_optimization(self,
                             candidate_targets: List[float],
                             train_data: pd.DataFrame,
                             val_data: pd.DataFrame) -> OptimizationResult:
        """Perform Bayesian optimization."""
        self.logger.info('🧠 Running Bayesian optimization')

        # Define search space
        space = []
        param_names = []

        # Target parameters
        for i, target in enumerate(candidate_targets[:5]):  # Limit to top 5 targets
            space.append(Real(target * 0.8, target * 1.2, name=f'target_{i}'))
            param_names.append(f'target_{i}')

        # Horizon parameters
        for i in range(len(candidate_targets[:5])):
            space.append(Integer(self.config.min_horizon, self.config.max_horizon, name=f'horizon_{i}'))
            param_names.append(f'horizon_{i}')

        # Define objective function
        @use_named_args(space)
        def objective(**params):
            # Extract parameters
            targets = {}
            horizons = {}

            for i in range(len(candidate_targets[:5])):
                targets[f'target_{i}'] = params[f'target_{i}']
                horizons[f'target_{i}'] = params[f'horizon_{i}']

            # Evaluate configuration
            score = self._evaluate_configuration(targets, horizons, train_data)

            # Store in history
            self.optimization_history.append({
                'targets': targets.copy(),
                'horizons': horizons.copy(),
                'score': score,
                'timestamp': datetime.now()
            })

            return -score  # Minimize negative score (maximize score)

        # Run optimization
        try:
            result = gp_minimize(
                func=objective,
                dimensions=space,
                n_calls=self.config.n_optimization_trials,
                random_state=42
            )

            # Extract best parameters
            best_params = dict(zip(param_names, result.x))
            best_targets = {k: v for k, v in best_params.items() if k.startswith('target_')}
            best_horizons = {k: v for k, v in best_params.items() if k.startswith('horizon_')}

            # Validate on validation data
            validation_scores = self._validate_configuration(best_targets, best_horizons, val_data)

            return OptimizationResult(
                optimal_targets=best_targets,
                optimal_horizons=best_horizons,
                objective_score=-result.fun,
                performance_metrics=validation_scores,
                validation_scores=validation_scores,
                candidate_results=[],
                optimization_history=self.optimization_history,
                metadata={
                    'method': 'bayesian_optimization',
                    'n_trials': len(result.x_iters),
                    'convergence': result.func_vals
                }
            )

        except Exception as e:
            self.logger.error(f'Bayesian optimization failed: {e}')
            return self._create_default_result()

    def _optuna_optimization(self,
                           candidate_targets: List[float],
                           train_data: pd.DataFrame,
                           val_data: pd.DataFrame) -> OptimizationResult:
        """Perform Optuna TPE optimization."""
        self.logger.info('🎯 Running Optuna TPE optimization')

        try:
            # Create study
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )

            # Define objective function
            def objective(trial):
                # Suggest parameters
                targets = {}
                horizons = {}

                for i, target in enumerate(candidate_targets[:5]):
                    targets[f'target_{i}'] = trial.suggest_float(
                        f'target_{i}', target * 0.8, target * 1.2
                    )
                    horizons[f'target_{i}'] = trial.suggest_int(
                        f'horizon_{i}', self.config.min_horizon, self.config.max_horizon
                    )

                # Evaluate configuration
                score = self._evaluate_configuration(targets, horizons, train_data)

                # Store in history
                self.optimization_history.append({
                    'targets': targets.copy(),
                    'horizons': horizons.copy(),
                    'score': score,
                    'timestamp': datetime.now()
                })

                return score

            # Run optimization
            study.optimize(objective, n_trials=self.config.n_optimization_trials)

            # Extract best parameters
            best_targets = {k: v for k, v in study.best_params.items() if k.startswith('target_')}
            best_horizons = {k: v for k, v in study.best_params.items() if k.startswith('horizon_')}

            # Validate on validation data
            validation_scores = self._validate_configuration(best_targets, best_horizons, val_data)

            return OptimizationResult(
                optimal_targets=best_targets,
                optimal_horizons=best_horizons,
                objective_score=study.best_value,
                performance_metrics=validation_scores,
                validation_scores=validation_scores,
                candidate_results=[],
                optimization_history=self.optimization_history,
                metadata={
                    'method': 'optuna_tpe',
                    'n_trials': len(study.trials),
                    'best_trial': study.best_trial.number
                }
            )

        except Exception as e:
            self.logger.error(f'Optuna optimization failed: {e}')
            return self._create_default_result()

    def _differential_evolution_optimization(self,
                                           candidate_targets: List[float],
                                           train_data: pd.DataFrame,
                                           val_data: pd.DataFrame) -> OptimizationResult:
        """Perform differential evolution optimization."""
        self.logger.info('🧬 Running differential evolution optimization')

        # Define bounds
        bounds = []

        # Target bounds
        for target in candidate_targets[:5]:
            bounds.append((target * 0.8, target * 1.2))

        # Horizon bounds
        for _ in candidate_targets[:5]:
            bounds.append((self.config.min_horizon, self.config.max_horizon))

        # Define objective function
        def objective(x):
            # Extract parameters
            n_targets = len(candidate_targets[:5])
            targets = {f'target_{i}': x[i] for i in range(n_targets)}
            horizons = {f'target_{i}': int(x[i + n_targets]) for i in range(n_targets)}

            # Evaluate configuration
            score = self._evaluate_configuration(targets, horizons, train_data)

            # Store in history
            self.optimization_history.append({
                'targets': targets.copy(),
                'horizons': horizons.copy(),
                'score': score,
                'timestamp': datetime.now()
            })

            return -score  # Minimize negative score

        try:
            # Run optimization
            result = differential_evolution(
                objective,
                bounds,
                maxiter=self.config.n_optimization_trials // 10,
                seed=42,
                workers=1  # Single worker to avoid issues
            )

            # Extract best parameters
            n_targets = len(candidate_targets[:5])
            best_targets = {f'target_{i}': result.x[i] for i in range(n_targets)}
            best_horizons = {f'target_{i}': int(result.x[i + n_targets]) for i in range(n_targets)}

            # Validate on validation data
            validation_scores = self._validate_configuration(best_targets, best_horizons, val_data)

            return OptimizationResult(
                optimal_targets=best_targets,
                optimal_horizons=best_horizons,
                objective_score=-result.fun,
                performance_metrics=validation_scores,
                validation_scores=validation_scores,
                candidate_results=[],
                optimization_history=self.optimization_history,
                metadata={
                    'method': 'differential_evolution',
                    'n_evaluations': result.nfev,
                    'success': result.success
                }
            )

        except Exception as e:
            self.logger.error(f'Differential evolution failed: {e}')
            return self._create_default_result()

    def _grid_search_optimization(self,
                                candidate_targets: List[float],
                                train_data: pd.DataFrame,
                                val_data: pd.DataFrame) -> OptimizationResult:
        """Perform grid search optimization."""
        self.logger.info('🔍 Running grid search optimization')

        # Create grid of parameters
        target_grid = candidate_targets[:3]  # Limit to top 3 for grid search
        horizon_grid = list(range(self.config.min_horizon, min(self.config.max_horizon + 1, 11), 2))

        best_score = -np.inf
        best_targets = {}
        best_horizons = {}

        total_combinations = len(target_grid) * len(horizon_grid)
        combination_count = 0

        for target in target_grid:
            for horizon in horizon_grid:
                combination_count += 1

                if combination_count % 10 == 0:
                    self.logger.info(f'   → Progress: {combination_count}/{total_combinations}')

                # Create configuration
                targets = {'target_0': target}
                horizons = {'target_0': horizon}

                # Evaluate configuration
                score = self._evaluate_configuration(targets, horizons, train_data)

                # Store in history
                self.optimization_history.append({
                    'targets': targets.copy(),
                    'horizons': horizons.copy(),
                    'score': score,
                    'timestamp': datetime.now()
                })

                # Update best if better
                if score > best_score:
                    best_score = score
                    best_targets = targets.copy()
                    best_horizons = horizons.copy()

        # Validate best configuration
        validation_scores = self._validate_configuration(best_targets, best_horizons, val_data)

        return OptimizationResult(
            optimal_targets=best_targets,
            optimal_horizons=best_horizons,
            objective_score=best_score,
            performance_metrics=validation_scores,
            validation_scores=validation_scores,
            candidate_results=[],
            optimization_history=self.optimization_history,
            metadata={
                'method': 'grid_search',
                'total_combinations': total_combinations
            }
        )

    def _evaluate_configuration(self,
                              targets: Dict[str, float],
                              horizons: Dict[str, int],
                              market_data: pd.DataFrame) -> float:
        """Evaluate a specific target-horizon configuration."""
        if 'close' not in market_data.columns or len(market_data) < 100:
            return 0.0

        prices = market_data['close']

        # Calculate performance for each target-horizon pair
        performance_metrics = []

        for target_key in targets:
            target = targets[target_key]
            horizon = horizons[target_key]

            # Calculate metrics for this target-horizon combination
            metrics = self._calculate_target_horizon_metrics(target, horizon, prices)
            performance_metrics.append(metrics)

        if not performance_metrics:
            return 0.0

        # Combine metrics based on optimization objective
        if self.config.optimization_objective == OptimizationObjective.MULTI_OBJECTIVE:
            return self._calculate_multi_objective_score(performance_metrics)
        elif self.config.optimization_objective == OptimizationObjective.SHARPE_RATIO:
            return np.mean([m['sharpe_ratio'] for m in performance_metrics])
        elif self.config.optimization_objective == OptimizationObjective.HIT_RATE:
            return np.mean([m['hit_rate'] for m in performance_metrics])
        else:
            return self._calculate_multi_objective_score(performance_metrics)

    def _calculate_target_horizon_metrics(self,
                                        target: float,
                                        horizon: int,
                                        prices: pd.Series) -> Dict[str, float]:
        """Calculate metrics for a specific target-horizon combination."""
        metrics = {
            'hit_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'information_ratio': 0.0,
            'profit_factor': 0.0
        }

        if len(prices) < horizon + 50:
            return metrics

        # Simulate trading strategy
        returns = []
        hits = 0
        total_trades = 0

        for i in range(len(prices) - horizon):
            entry_price = prices.iloc[i]
            target_price = entry_price * (1 + target)

            # Check if target is hit within horizon
            future_prices = prices.iloc[i+1:i+horizon+1]

            hit = any(price >= target_price for price in future_prices)

            if hit:
                hits += 1
                # Calculate return when target is hit
                ret = target  # Assume we exit at target
            else:
                # Calculate return at end of horizon
                final_price = future_prices.iloc[-1] if len(future_prices) > 0 else entry_price
                ret = (final_price - entry_price) / entry_price

            returns.append(ret)
            total_trades += 1

        if not returns or total_trades == 0:
            return metrics

        returns = np.array(returns)

        # Calculate metrics
        metrics['hit_rate'] = hits / total_trades

        if np.std(returns) > 0:
            metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns)

        # Max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = abs(np.min(drawdown))

        # Information ratio (similar to Sharpe but vs benchmark)
        benchmark_return = 0.0  # Assume cash benchmark
        excess_returns = returns - benchmark_return
        if np.std(excess_returns) > 0:
            metrics['information_ratio'] = np.mean(excess_returns) / np.std(excess_returns)

        # Profit factor
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        if len(negative_returns) > 0 and np.sum(negative_returns) != 0:
            metrics['profit_factor'] = abs(np.sum(positive_returns)) / abs(np.sum(negative_returns))
        elif len(positive_returns) > 0:
            metrics['profit_factor'] = 2.0  # High profit factor if no losses

        return metrics

    def _calculate_multi_objective_score(self, performance_metrics: List[Dict[str, float]]) -> float:
        """Calculate multi-objective score."""
        if not performance_metrics:
            return 0.0

        # Average metrics across all target-horizon combinations
        avg_hit_rate = np.mean([m['hit_rate'] for m in performance_metrics])
        avg_sharpe = np.mean([m['sharpe_ratio'] for m in performance_metrics])
        avg_drawdown = np.mean([m['max_drawdown'] for m in performance_metrics])
        avg_information = np.mean([m['information_ratio'] for m in performance_metrics])

        # Normalize and combine with weights
        hit_rate_score = min(1.0, max(0.0, avg_hit_rate))
        sharpe_score = min(1.0, max(0.0, (avg_sharpe + 2.0) / 4.0))  # Normalize Sharpe to 0-1
        drawdown_penalty = min(1.0, avg_drawdown / self.config.max_drawdown_threshold)
        information_score = min(1.0, max(0.0, (avg_information + 2.0) / 4.0))

        # Weighted combination
        composite_score = (
            self.config.hit_rate_weight * hit_rate_score +
            self.config.sharpe_weight * sharpe_score +
            self.config.information_weight * information_score -
            self.config.drawdown_weight * drawdown_penalty
        )

        return max(0.0, composite_score)

    def _validate_configuration(self,
                              targets: Dict[str, float],
                              horizons: Dict[str, int],
                              validation_data: pd.DataFrame) -> Dict[str, float]:
        """Validate configuration on validation data."""
        return self._evaluate_configuration_detailed(targets, horizons, validation_data)

    def _evaluate_configuration_detailed(self,
                                       targets: Dict[str, float],
                                       horizons: Dict[str, int],
                                       market_data: pd.DataFrame) -> Dict[str, float]:
        """Detailed evaluation of configuration."""
        detailed_metrics = {}

        if 'close' not in market_data.columns:
            return detailed_metrics

        prices = market_data['close']

        for target_key in targets:
            target = targets[target_key]
            horizon = horizons[target_key]

            metrics = self._calculate_target_horizon_metrics(target, horizon, prices)

            for metric_name, value in metrics.items():
                detailed_metrics[f'{target_key}_{metric_name}'] = value

        # Overall metrics
        if targets:
            detailed_metrics['overall_score'] = self._evaluate_configuration(targets, horizons, market_data)

        return detailed_metrics

    def _create_default_result(self) -> OptimizationResult:
        """Create default optimization result."""
        return OptimizationResult(
            optimal_targets={'target_0': 0.005},
            optimal_horizons={'target_0': 5},
            objective_score=0.0,
            performance_metrics={},
            validation_scores={},
            candidate_results=[],
            optimization_history=[],
            metadata={'method': 'default', 'error': 'optimization_failed'}
        )

# Convenience functions
def discover_optimal_targets_and_horizons(market_data: pd.DataFrame,
                                        config: Optional[DynamicOptimizationConfig] = None) -> OptimizationResult:
    """Convenience function for joint target-horizon optimization."""
    optimizer = JointTargetHorizonOptimizer(config)
    return optimizer.optimize_target_horizon_combinations(market_data)

def create_optimized_multi_horizon_config(market_data: pd.DataFrame,
                                         config: Optional[DynamicOptimizationConfig] = None) -> MultiHorizonConfig:
    """Create optimized MultiHorizonConfig from market data."""
    result = discover_optimal_targets_and_horizons(market_data, config)

    # Convert optimization result to MultiHorizonConfig
    multi_config = MultiHorizonConfig()

    # Extract targets and horizons
    targets = {}
    horizons = {}

    for key, value in result.optimal_targets.items():
        target_name = key.replace('target_', 'target_')  # Keep naming consistent
        targets[target_name] = value

    for key, value in result.optimal_horizons.items():
        horizon_name = key.replace('target_', 'horizon_')
        horizons[horizon_name] = value

    # Map to standard names if possible
    if targets:
        target_values = sorted(targets.values())
        standard_names = ['micro', 'small', 'medium', 'good', 'great']

        multi_config.profit_targets = {}
        for i, target in enumerate(target_values[:len(standard_names)]):
            multi_config.profit_targets[standard_names[i]] = target

    if horizons:
        horizon_values = sorted(horizons.values())
        standard_horizon_names = ['immediate', 'short']

        multi_config.time_horizons = {}
        for i, horizon in enumerate(horizon_values[:len(standard_horizon_names)]):
            multi_config.time_horizons[standard_horizon_names[i]] = horizon

    return multi_config
