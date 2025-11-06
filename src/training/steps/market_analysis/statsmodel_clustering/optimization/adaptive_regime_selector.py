"""
Adaptive Regime Count Selection

This module automatically selects the optimal number of regimes using multiple criteria:
1. BIC/AIC elbow detection (statistical fit)
2. Temporal stability (regime persistence across seeds)
3. Economic validation (Sharpe ratio improvement)
4. Cross-validation consistency

Expected Impact:
- More appropriate regime counts for different markets
- Reduced overfitting
- Better out-of-sample performance
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
import logging
from sklearn.metrics import adjusted_rand_score

# Import utilities
try:
    from src.utils.tprint import (
        tprint_info, tprint_success, tprint_warning, tprint_error
    )
except ImportError:
    def tprint_info(msg): print(f'ℹ️  {msg}')
    def tprint_success(msg): print(f'✅ {msg}')
    def tprint_warning(msg): print(f'⚠️  {msg}')
    def tprint_error(msg): print(f'❌ {msg}')

logger = logging.getLogger(__name__)


@dataclass
class RegimeSelectionResult:
    """Result container for regime selection."""
    optimal_k: int
    all_results: Dict[int, Dict[str, float]]
    selection_criteria: Dict[str, Any]
    scores_by_k: Dict[int, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdaptiveRegimeSelector:
    """
    Automatically select optimal number of regimes.

    Methods:
    1. BIC/AIC elbow detection
    2. Stability analysis across regime counts
    3. Economic validation (Sharpe improvement)
    4. Multi-criteria decision
    """

    def __init__(
        self,
        regime_range: Tuple[int, int] = (2, 10),
        n_stability_seeds: int = 5,
        random_state: int = 42
    ):
        """
        Initialize adaptive selector.

        Args:
            regime_range: (min_regimes, max_regimes) to test
            n_stability_seeds: Number of random seeds for stability analysis
            random_state: Base random seed
        """
        self.regime_range = regime_range
        self.n_stability_seeds = n_stability_seeds
        self.random_state = random_state
        self.logger = logging.getLogger(self.__class__.__name__)

    def select_optimal_regimes(
        self,
        data: np.ndarray,
        fit_func: Callable,
        returns: Optional[np.ndarray] = None,
        use_economic: bool = True
    ) -> RegimeSelectionResult:
        """
        Select optimal regime count via multiple criteria.

        Args:
            data: Input data (T, D)
            fit_func: Function to fit model (takes data, k, seed)
            returns: Optional returns for economic validation
            use_economic: Whether to use economic criteria

        Returns:
            RegimeSelectionResult with optimal k and analysis

        Criteria:
        1. BIC elbow (statistical fit)
        2. Temporal stability (regime persistence)
        3. Economic utility (Sharpe improvement) - if returns available
        4. Regime quality metrics
        """
        tprint_info(f"🔍 Testing regime counts from {self.regime_range[0]} to {self.regime_range[1]}")

        min_regimes, max_regimes = self.regime_range
        results = {}

        # Fit models for each k
        for k in range(min_regimes, max_regimes + 1):
            tprint_info(f"  📊 Testing k={k} regimes")

            try:
                # Fit model
                model_result = fit_func(data, k, self.random_state)

                # Calculate criteria
                results[k] = {
                    'bic': model_result.bic if hasattr(model_result, 'bic') else np.inf,
                    'aic': model_result.aic if hasattr(model_result, 'aic') else np.inf,
                    'log_likelihood': model_result.log_likelihood if hasattr(model_result, 'log_likelihood') else -np.inf,
                }

                # Temporal smoothness
                if hasattr(model_result, 'cluster_labels'):
                    from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
                        calculate_temporal_smoothness
                    )
                    results[k]['temporal_smoothness'] = calculate_temporal_smoothness(
                        model_result.cluster_labels
                    )
                else:
                    results[k]['temporal_smoothness'] = 0.0

                # Regime stability
                tprint_info(f"    🔄 Calculating stability for k={k}")
                results[k]['regime_stability'] = self._calculate_regime_stability(
                    data, k, fit_func
                )

                # Economic utility (if returns available)
                if use_economic and returns is not None and hasattr(model_result, 'cluster_labels'):
                    results[k]['economic_utility'] = self._calculate_economic_utility(
                        model_result.cluster_labels, returns
                    )
                else:
                    results[k]['economic_utility'] = 0.0

                tprint_success(
                    f"    ✅ k={k}: BIC={results[k]['bic']:.2f}, "
                    f"Stability={results[k]['regime_stability']:.3f}, "
                    f"Temporal={results[k]['temporal_smoothness']:.3f}"
                )

            except Exception as e:
                tprint_warning(f"    ⚠️ Failed to evaluate k={k}: {e}")
                results[k] = {
                    'bic': np.inf,
                    'aic': np.inf,
                    'log_likelihood': -np.inf,
                    'temporal_smoothness': 0.0,
                    'regime_stability': 0.0,
                    'economic_utility': 0.0
                }

        # Multi-criteria selection
        tprint_info("🤔 Selecting optimal k via multi-criteria analysis")
        optimal_k, selection_criteria, scores_by_k = self._select_via_multi_criteria(results)

        tprint_success(f"🏆 Optimal regime count selected: k={optimal_k}")
        tprint_info(f"    Selection criteria: {selection_criteria}")

        return RegimeSelectionResult(
            optimal_k=optimal_k,
            all_results=results,
            selection_criteria=selection_criteria,
            scores_by_k=scores_by_k,
            metadata={
                'regime_range': self.regime_range,
                'n_stability_seeds': self.n_stability_seeds,
                'use_economic': use_economic
            }
        )

    def _detect_elbow(self, criterion_values: List[float], k_values: List[int]) -> int:
        """
        Detect elbow point in criterion curve using second derivative.

        Args:
            criterion_values: Values of criterion for each k
            k_values: Corresponding k values

        Returns:
            Optimal k at elbow point
        """
        if len(criterion_values) < 3:
            # Not enough points for elbow detection
            return k_values[np.argmin(criterion_values)]

        criterion_array = np.array(criterion_values)

        # Calculate second derivative (curvature)
        first_diff = np.diff(criterion_array)
        second_diff = np.diff(first_diff)

        # Find maximum curvature (most negative for minimization criteria like BIC)
        if len(second_diff) > 0:
            elbow_idx = np.argmax(np.abs(second_diff)) + 2  # +2 because of two diffs

            # Ensure within bounds
            elbow_idx = min(elbow_idx, len(k_values) - 1)

            return k_values[elbow_idx]
        else:
            # Fallback: minimum criterion
            return k_values[np.argmin(criterion_values)]

    def _calculate_regime_stability(
        self,
        data: np.ndarray,
        k: int,
        fit_func: Callable
    ) -> float:
        """
        Calculate regime stability across random seeds.

        Fits model multiple times with different seeds and measures
        consistency via Adjusted Rand Index (ARI).

        Args:
            data: Input data
            k: Number of regimes
            fit_func: Function to fit model

        Returns:
            Mean pairwise ARI [0, 1] (higher = more stable)
        """
        labels_list = []

        for i in range(self.n_stability_seeds):
            seed = self.random_state + i

            try:
                result = fit_func(data, k, seed)
                if hasattr(result, 'cluster_labels'):
                    labels_list.append(result.cluster_labels)
            except Exception as e:
                self.logger.debug(f"Stability seed {seed} failed: {e}")
                continue

        if len(labels_list) < 2:
            return 0.0

        # Calculate pairwise ARI
        ari_scores = []
        for i in range(len(labels_list)):
            for j in range(i+1, len(labels_list)):
                ari = adjusted_rand_score(labels_list[i], labels_list[j])
                ari_scores.append(ari)

        # High mean ARI = stable
        return float(np.mean(ari_scores))

    def _calculate_economic_utility(
        self,
        labels: np.ndarray,
        returns: np.ndarray
    ) -> float:
        """
        Calculate economic utility via regime-conditional Sharpe ratio.

        Measures if regime detection improves risk-adjusted returns.

        Args:
            labels: Regime labels
            returns: Return series

        Returns:
            Mean Sharpe ratio across regimes (higher is better)
        """
        if len(labels) != len(returns):
            return 0.0

        unique_regimes = np.unique(labels)
        sharpe_ratios = []

        for regime in unique_regimes:
            regime_mask = labels == regime
            regime_returns = returns[regime_mask]

            if len(regime_returns) < 2:
                continue

            # Calculate Sharpe ratio for this regime
            mean_return = np.mean(regime_returns)
            std_return = np.std(regime_returns)

            if std_return > 1e-8:
                sharpe = mean_return / std_return
                # Annualize (assuming daily returns)
                sharpe_annualized = sharpe * np.sqrt(252)
                sharpe_ratios.append(sharpe_annualized)

        if not sharpe_ratios:
            return 0.0

        # Return mean absolute Sharpe (higher is better)
        return float(np.mean(np.abs(sharpe_ratios)))

    def _select_via_multi_criteria(
        self,
        results: Dict[int, Dict[str, float]]
    ) -> Tuple[int, Dict[str, Any], Dict[int, float]]:
        """
        Select optimal k using multiple criteria.

        Combines:
        1. BIC elbow
        2. Stability (max)
        3. Temporal smoothness (max)
        4. Economic utility (max)

        Args:
            results: Results dictionary for each k

        Returns:
            Tuple of (optimal_k, selection_criteria_dict, composite_scores_by_k)
        """
        k_values = sorted(results.keys())

        # Extract criteria
        bic_values = [results[k]['bic'] for k in k_values]
        stability_values = [results[k]['regime_stability'] for k in k_values]
        temporal_values = [results[k]['temporal_smoothness'] for k in k_values]
        economic_values = [results[k]['economic_utility'] for k in k_values]

        # 1. BIC elbow detection
        bic_elbow_k = self._detect_elbow(bic_values, k_values)

        # 2. Maximum stability
        stability_max_k = k_values[np.argmax(stability_values)]

        # 3. Maximum temporal smoothness
        temporal_max_k = k_values[np.argmax(temporal_values)]

        # 4. Maximum economic utility
        if max(economic_values) > 0:
            economic_max_k = k_values[np.argmax(economic_values)]
        else:
            economic_max_k = None

        # Composite scoring (normalize each criterion)
        composite_scores = {}

        for k in k_values:
            # Normalize BIC (lower is better, so invert)
            bic_score = 1.0 - (results[k]['bic'] - min(bic_values)) / (max(bic_values) - min(bic_values) + 1e-8)

            # Normalize stability (higher is better)
            stability_score = (results[k]['regime_stability'] - min(stability_values)) / (max(stability_values) - min(stability_values) + 1e-8)

            # Normalize temporal (higher is better)
            temporal_score = (results[k]['temporal_smoothness'] - min(temporal_values)) / (max(temporal_values) - min(temporal_values) + 1e-8)

            # Normalize economic (higher is better)
            if max(economic_values) > 0:
                economic_score = (results[k]['economic_utility'] - min(economic_values)) / (max(economic_values) - min(economic_values) + 1e-8)
            else:
                economic_score = 0.0

            # Weighted composite (adjust weights based on availability)
            if economic_max_k is not None:
                # All criteria available
                composite = (
                    0.30 * bic_score +
                    0.25 * stability_score +
                    0.25 * temporal_score +
                    0.20 * economic_score
                )
            else:
                # No economic data
                composite = (
                    0.40 * bic_score +
                    0.30 * stability_score +
                    0.30 * temporal_score
                )

            composite_scores[k] = composite

        # Select k with highest composite score
        optimal_k = max(composite_scores, key=composite_scores.get)

        # But prefer BIC elbow if composite scores are close
        if abs(composite_scores[optimal_k] - composite_scores[bic_elbow_k]) < 0.05:
            optimal_k = bic_elbow_k

        selection_criteria = {
            'bic_elbow_k': bic_elbow_k,
            'stability_max_k': stability_max_k,
            'temporal_max_k': temporal_max_k,
            'economic_max_k': economic_max_k,
            'composite_selected_k': optimal_k
        }

        return optimal_k, selection_criteria, composite_scores


def create_adaptive_selector(
    regime_range: Tuple[int, int] = (2, 10),
    n_stability_seeds: int = 5,
    random_state: int = 42
) -> AdaptiveRegimeSelector:
    """
    Factory function to create adaptive regime selector.

    Args:
        regime_range: (min, max) regime counts to test
        n_stability_seeds: Number of seeds for stability analysis
        random_state: Base random seed

    Returns:
        AdaptiveRegimeSelector instance
    """
    return AdaptiveRegimeSelector(
        regime_range=regime_range,
        n_stability_seeds=n_stability_seeds,
        random_state=random_state
    )
