"""
Enhanced Economic Evaluator for Hybrid NAS-TAS Regime Discovery.

Implements coefficient of variation (CV) optimization for volatility, returns, and volume,
with accessory CV based on momentum and entropy for regime clustering optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Initialize console for tprint
console = Console()

def tprint(*args, **kwargs):
    """Enhanced print function with rich formatting."""
    console.print(*args, **kwargs)

logger = logging.getLogger(__name__)

@dataclass
class EconomicEvaluationConfig:
    """Configuration for economic evaluation."""
    # CV optimization targets
    target_cluster_count_min: int = 6
    target_cluster_count_max: int = 15
    max_cluster_distribution: float = 0.25  # 25% max
    min_cluster_distribution: float = 0.03  # 3% min

    # CV weights
    volatility_cv_weight: float = 0.4
    returns_cv_weight: float = 0.3
    volume_cv_weight: float = 0.3

    # Removed accessory CV weights (momentum and entropy)

    # Dynamic weighting based on market conditions
    enable_dynamic_weighting: bool = True
    volatility_sensitivity: float = 0.3  # How much volatility affects weights
    trend_sensitivity: float = 0.2      # How much trend affects weights

    # Economic significance thresholds
    min_economic_significance: float = 0.5
    min_trading_viability: float = 0.5
    min_regime_duration: int = 5

    # Additional weights and thresholds for compatibility
    price_impact_weight: float = 0.25
    volume_significance_weight: float = 0.15
    volatility_impact_weight: float = 0.20
    trend_consistency_weight: float = 0.15
    market_efficiency_weight: float = 0.10
    economic_indicators_weight: float = 0.10
    trading_opportunity_weight: float = 0.05
    risk_adjustment_weight: float = 0.05
    significance_threshold: float = 0.6
    price_impact_threshold: float = 0.5
    volume_threshold: float = 0.4
    volatility_threshold: float = 0.5
    trend_threshold: float = 0.6
    efficiency_threshold: float = 0.5
    economic_indicators_lookback: int = 252
    economic_correlation_threshold: float = 0.3
    bootstrap_iterations: int = 100
    confidence_level: float = 0.95

    # Enhanced metrics
    enable_enhanced_price_analysis: bool = True
    enable_volume_pattern_analysis: bool = True
    enable_regime_transition_analysis: bool = True
    enable_cross_regime_correlation: bool = True

    # Regime-specific analysis
    enable_regime_specific_analysis: bool = True
    min_regime_samples: int = 50
    regime_stability_threshold: float = 0.7

    # TAS-specific enhancements
    enable_tree_based_analysis: bool = True
    tree_importance_threshold: float = 0.1
    tree_depth_penalty: float = 0.1
    tree_complexity_weight: float = 0.2

    # NAS-specific enhancements
    enable_neural_based_analysis: bool = True
    neural_confidence_threshold: float = 0.8
    neural_uncertainty_weight: float = 0.3
    neural_architecture_complexity: float = 0.1

    # Hybrid analysis
    enable_hybrid_analysis: bool = True
    hybrid_weighting: float = 0.5
    hybrid_consensus_threshold: float = 0.7

    # Position-aware analysis
    enable_position_aware_analysis: bool = True
    position_aware_config: Optional[Any] = None

    # Economic indicators
    enable_economic_indicators: bool = True
    enable_bootstrap_analysis: bool = True

class EnhancedEconomicEvaluator:
    """
    Enhanced economic evaluator with CV optimization for regime clustering.

    Optimizes coefficient of variation for volatility, returns, and volume
    while ensuring cluster distribution targets are met.
    """

    def __init__(self, config: Optional[EconomicEvaluationConfig] = None):
        """Initialize the enhanced economic evaluator."""
        self.config = config or EconomicEvaluationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def evaluate_regime_clustering(self, regime_predictions: np.ndarray,
                                 market_data: pd.DataFrame,
                                 features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Evaluate regime clustering with CV optimization and distribution targets.

        Args:
            regime_predictions: Regime predictions
            market_data: Market data with OHLCV
            features: Optional additional features

        Returns:
            Comprehensive evaluation results
        """
        try:
            tprint(Panel.fit(
                "[bold green]💰 Enhanced Economic Evaluator[/bold green]\n"
                f"Regime predictions: {len(regime_predictions)} samples\n"
                f"Market data shape: {market_data.shape if hasattr(market_data, 'shape') else 'Unknown'}\n"
                f"Features available: {features is not None}",
                title="Economic Evaluation Start",
                border_style="green"
            ))

            self.logger.info("💰 Starting enhanced economic evaluation with CV optimization")

            # Basic regime analysis
            tprint("[yellow]📊 Analyzing regime distribution...[/yellow]")
            regime_analysis = self._analyze_regime_distribution(regime_predictions)
            tprint(f"[green]✅ Found {regime_analysis.get('num_regimes', 0)} regimes[/green]")

            # CV optimization analysis
            tprint("[yellow]📈 Calculating CV optimization...[/yellow]")
            cv_analysis = self._calculate_cv_optimization(regime_predictions, market_data, features)
            tprint(f"[green]✅ CV analysis completed[/green]")

            # Economic significance analysis
            tprint("[yellow]💎 Calculating economic significance...[/yellow]")
            economic_analysis = self._calculate_economic_significance(regime_predictions, market_data)
            tprint(f"[green]✅ Economic analysis completed[/green]")

            # Trading viability analysis
            tprint("[yellow]📈 Calculating trading viability...[/yellow]")
            trading_analysis = self._calculate_trading_viability(regime_predictions, market_data)
            tprint(f"[green]✅ Trading analysis completed[/green]")

            # Multi-objective optimization score
            optimization_score = self._calculate_multi_objective_score(
                regime_analysis, cv_analysis, economic_analysis, trading_analysis
            )

            # Cluster distribution validation
            distribution_validation = self._validate_cluster_distribution(regime_analysis)

            results = {
                'regime_analysis': regime_analysis,
                'cv_optimization': cv_analysis,
                'economic_significance': economic_analysis,
                'trading_viability': trading_analysis,
                'optimization_score': optimization_score,
                'distribution_validation': distribution_validation,
                'overall_quality': self._calculate_overall_quality(
                    optimization_score, distribution_validation
                )
            }

            self.logger.info("✅ Enhanced economic evaluation completed")
            return results

        except Exception as e:
            self.logger.error(f"❌ Enhanced economic evaluation failed: {e}")
            return {'error': str(e), 'overall_quality': 0.0}

    def _analyze_regime_distribution(self, regime_predictions: np.ndarray) -> Dict[str, Any]:
        """Analyze regime distribution and basic statistics."""
        try:
            unique_regimes = np.unique(regime_predictions)
            n_regimes = len(unique_regimes)

            # Calculate regime sizes and percentages
            regime_sizes = {}
            regime_percentages = {}
            total_samples = len(regime_predictions)

            for regime in unique_regimes:
                size = np.sum(regime_predictions == regime)
                percentage = size / total_samples
                regime_sizes[regime] = size
                regime_percentages[regime] = percentage

            # Calculate distribution statistics
            percentages = list(regime_percentages.values())
            distribution_std = np.std(percentages)
            distribution_mean = np.mean(percentages)
            distribution_cv = distribution_std / distribution_mean if distribution_mean > 0 else 0

            # Check if distribution meets targets
            meets_cluster_count = self.config.target_cluster_count_min <= n_regimes <= self.config.target_cluster_count_max
            meets_max_distribution = all(p <= self.config.max_cluster_distribution for p in percentages)
            meets_min_distribution = all(p >= self.config.min_cluster_distribution for p in percentages)

            return {
                'n_regimes': n_regimes,
                'regime_sizes': regime_sizes,
                'regime_percentages': regime_percentages,
                'distribution_std': distribution_std,
                'distribution_mean': distribution_mean,
                'distribution_cv': distribution_cv,
                'meets_cluster_count': meets_cluster_count,
                'meets_max_distribution': meets_max_distribution,
                'meets_min_distribution': meets_min_distribution,
                'distribution_quality': self._calculate_distribution_quality(
                    n_regimes, percentages, meets_cluster_count, meets_max_distribution, meets_min_distribution
                )
            }

        except Exception as e:
            self.logger.error(f"❌ Regime distribution analysis failed: {e}")
            return {'error': str(e), 'n_regimes': 0, 'distribution_quality': 0.0}

    def _calculate_cv_optimization(self, regime_predictions: np.ndarray,
                                 market_data: pd.DataFrame,
                                 features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Calculate CV optimization for volatility, returns, and volume."""
        try:
            unique_regimes = np.unique(regime_predictions)
            cv_results = {}

            # Calculate returns and volatility
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                # Align regime_predictions with returns by dropping the first element
                # since pct_change() reduces length by 1
                aligned_regime_predictions = regime_predictions[1:]
            else:
                returns = pd.Series([0] * len(market_data))
                aligned_regime_predictions = regime_predictions

            # Calculate volume - align with returns length
            if 'volume' in market_data.columns:
                if len(returns) == len(market_data) - 1:
                    # If returns was created with pct_change().dropna(), align volume too
                    volume = market_data['volume'].iloc[1:]
                else:
                    volume = market_data['volume']
            else:
                volume = pd.Series([1] * len(returns))

            # Ensure all arrays have the same length
            min_length = min(len(aligned_regime_predictions), len(returns), len(volume))
            if min_length == 0:
                self.logger.warning("⚠️ No data available for CV optimization")
                return {'error': 'No data available', 'cv_optimization_score': 0.0}

            # Truncate all arrays to the same length
            aligned_regime_predictions = aligned_regime_predictions[:min_length]
            returns = returns.iloc[:min_length] if hasattr(returns, 'iloc') else returns[:min_length]
            volume = volume.iloc[:min_length] if hasattr(volume, 'iloc') else volume[:min_length]

            # Removed momentum and entropy calculations

            # Calculate CV for each regime
            for regime in unique_regimes:
                regime_mask = aligned_regime_predictions == regime
                regime_returns = returns[regime_mask]
                regime_volume = volume[regime_mask]

                # Basic CV calculations
                returns_cv = self._calculate_cv(regime_returns)
                volume_cv = self._calculate_cv(regime_volume)
                volatility_cv = self._calculate_cv(np.abs(regime_returns))  # Volatility CV

                # Weighted CV score (removed momentum and entropy CV)
                weighted_cv = (
                    self.config.volatility_cv_weight * volatility_cv +
                    self.config.returns_cv_weight * returns_cv +
                    self.config.volume_cv_weight * volume_cv
                )

                cv_results[regime] = {
                    'volatility_cv': volatility_cv,
                    'returns_cv': returns_cv,
                    'volume_cv': volume_cv,
                    'weighted_cv': weighted_cv,
                    'regime_size': np.sum(regime_mask)
                }

            # Calculate overall CV optimization metrics
            weighted_cvs = [result['weighted_cv'] for result in cv_results.values()]
            avg_weighted_cv = np.mean(weighted_cvs) if weighted_cvs else 0.0
            cv_std = np.std(weighted_cvs) if weighted_cvs else 0.0

            # CV optimization score (lower is better)
            cv_optimization_score = 1.0 / (1.0 + avg_weighted_cv) if avg_weighted_cv > 0 else 1.0

            return {
                'regime_cv_results': cv_results,
                'avg_weighted_cv': avg_weighted_cv,
                'cv_std': cv_std,
                'cv_optimization_score': cv_optimization_score,
                'best_cv_regime': min(cv_results.keys(), key=lambda k: cv_results[k]['weighted_cv']) if cv_results else None,
                'worst_cv_regime': max(cv_results.keys(), key=lambda k: cv_results[k]['weighted_cv']) if cv_results else None
            }

        except Exception as e:
            self.logger.error(f"❌ CV optimization calculation failed: {e}")
            return {'error': str(e), 'cv_optimization_score': 0.0}

    def _calculate_cv(self, data: Union[pd.Series, np.ndarray]) -> float:
        """Calculate coefficient of variation."""
        try:
            if len(data) == 0:
                return 0.0

            data = np.array(data)
            data = data[~np.isnan(data)]  # Remove NaN values

            if len(data) == 0:
                return 0.0

            mean_val = np.mean(data)
            std_val = np.std(data)

            if mean_val == 0:
                return 0.0 if std_val == 0 else float('inf')

            return std_val / abs(mean_val)

        except Exception as e:
            self.logger.error(f"❌ CV calculation failed: {e}")
            return 0.0

    def _calculate_returns_entropy(self, returns: pd.Series) -> pd.Series:
        """Calculate entropy based on returns."""
        try:
            # Discretize returns into bins
            n_bins = 10
            bins = pd.cut(returns, bins=n_bins, labels=False, include_lowest=True)

            # Calculate entropy for rolling window
            window_size = 20
            entropy_values = []

            for i in range(len(bins)):
                start_idx = max(0, i - window_size + 1)
                window_bins = bins.iloc[start_idx:i+1]

                # Calculate probability distribution
                bin_counts = window_bins.value_counts()
                probabilities = bin_counts / len(window_bins)

                # Calculate entropy
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                entropy_values.append(entropy)

            return pd.Series(entropy_values, index=returns.index)

        except Exception as e:
            self.logger.error(f"❌ Returns entropy calculation failed: {e}")
            return pd.Series([0] * len(returns), index=returns.index)

    def _calculate_economic_significance(self, regime_predictions: np.ndarray,
                                       market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate economic significance for each regime."""
        try:
            # Align regime_predictions with market_data length
            if len(regime_predictions) != len(market_data):
                # If regime_predictions is longer, truncate it to match market_data
                if len(regime_predictions) > len(market_data):
                    aligned_regime_predictions = regime_predictions[:len(market_data)]
                else:
                    # If regime_predictions is shorter, pad it with the last value
                    aligned_regime_predictions = np.pad(
                        regime_predictions,
                        (0, len(market_data) - len(regime_predictions)),
                        mode='edge'
                    )
            else:
                aligned_regime_predictions = regime_predictions

            unique_regimes = np.unique(aligned_regime_predictions)
            economic_scores = {}

            for regime in unique_regimes:
                regime_mask = aligned_regime_predictions == regime
                regime_data = market_data[regime_mask]

                if len(regime_data) < self.config.min_regime_duration:
                    economic_scores[regime] = 0.0
                    continue

                # Calculate economic metrics
                if 'close' in regime_data.columns:
                    returns = regime_data['close'].pct_change().dropna()
                    if len(returns) > 0:
                        mean_return = returns.mean()
                        volatility = returns.std()
                        sharpe_ratio = mean_return / volatility if volatility > 0 else 0
                        max_drawdown = self._calculate_max_drawdown(returns)
                    else:
                        mean_return = volatility = sharpe_ratio = max_drawdown = 0
                else:
                    mean_return = volatility = sharpe_ratio = max_drawdown = 0

                # Volume characteristics
                if 'volume' in regime_data.columns:
                    volume_mean = regime_data['volume'].mean()
                    volume_std = regime_data['volume'].std()
                    volume_consistency = 1 - (volume_std / volume_mean) if volume_mean > 0 else 0
                else:
                    volume_mean = volume_consistency = 0

                # Regime duration
                duration = len(regime_data)
                duration_score = min(duration / 100, 1.0)  # Normalize to 0-1

                # Combined economic significance score
                economic_score = (
                    0.3 * abs(sharpe_ratio) +  # Risk-adjusted return
                    0.2 * volume_consistency +  # Volume stability
                    0.2 * duration_score +      # Regime persistence
                    0.2 * abs(mean_return) +   # Absolute return
                    0.1 * (1 - max_drawdown)   # Drawdown penalty
                )

                economic_scores[regime] = min(economic_score, 1.0)

            # Calculate overall economic significance
            avg_economic_score = np.mean(list(economic_scores.values())) if economic_scores else 0.0
            significant_regimes = len([s for s in economic_scores.values() if s >= self.config.min_economic_significance])

            return {
                'regime_economic_scores': economic_scores,
                'avg_economic_score': avg_economic_score,
                'significant_regimes_count': significant_regimes,
                'economic_significance_ratio': significant_regimes / len(unique_regimes) if unique_regimes.size > 0 else 0.0
            }

        except Exception as e:
            self.logger.error(f"❌ Economic significance calculation failed: {e}")
            return {'error': str(e), 'avg_economic_score': 0.0}

    def _calculate_trading_viability(self, regime_predictions: np.ndarray,
                                   market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trading viability for each regime."""
        try:
            # Align regime_predictions with market_data length
            if len(regime_predictions) != len(market_data):
                # If regime_predictions is longer, truncate it to match market_data
                if len(regime_predictions) > len(market_data):
                    aligned_regime_predictions = regime_predictions[:len(market_data)]
                else:
                    # If regime_predictions is shorter, pad it with the last value
                    aligned_regime_predictions = np.pad(
                        regime_predictions,
                        (0, len(market_data) - len(regime_predictions)),
                        mode='edge'
                    )
            else:
                aligned_regime_predictions = regime_predictions

            unique_regimes = np.unique(aligned_regime_predictions)
            trading_scores = {}

            for regime in unique_regimes:
                regime_mask = aligned_regime_predictions == regime
                regime_data = market_data[regime_mask]

                if len(regime_data) < self.config.min_regime_duration:
                    trading_scores[regime] = 0.0
                    continue

                # Calculate trading metrics
                if 'close' in regime_data.columns:
                    returns = regime_data['close'].pct_change().dropna()
                    if len(returns) > 0:
                        # Trading frequency (regime changes) - calculate for current regime only
                        regime_changes = np.sum(aligned_regime_predictions[1:] != aligned_regime_predictions[:-1])
                        # Calculate stability based on regime data length instead of total predictions
                        stability_score = 1 - (regime_changes / len(aligned_regime_predictions)) if len(aligned_regime_predictions) > 0 else 1.0

                        # Return consistency
                        positive_returns = np.sum(returns > 0)
                        return_consistency = positive_returns / len(returns) if len(returns) > 0 else 0.5

                        # Volatility consistency
                        volatility = returns.std()
                        volatility_score = 1 - min(volatility / 0.1, 1.0)  # Penalize high volatility
                    else:
                        stability_score = return_consistency = volatility_score = 0
                else:
                    stability_score = return_consistency = volatility_score = 0

                # Volume liquidity
                if 'volume' in regime_data.columns:
                    volume_mean = regime_data['volume'].mean()
                    liquidity_score = min(volume_mean / 1000, 1.0)  # Normalize volume
                else:
                    liquidity_score = 0

                # Combined trading viability score
                trading_score = (
                    0.3 * stability_score +      # Regime stability
                    0.3 * return_consistency +    # Return consistency
                    0.2 * volatility_score +     # Volatility consistency
                    0.2 * liquidity_score        # Volume liquidity
                )

                trading_scores[regime] = min(trading_score, 1.0)

            # Calculate overall trading viability
            avg_trading_score = np.mean(list(trading_scores.values())) if trading_scores else 0.0
            viable_regimes = len([s for s in trading_scores.values() if s >= self.config.min_trading_viability])

            return {
                'regime_trading_scores': trading_scores,
                'avg_trading_score': avg_trading_score,
                'viable_regimes_count': viable_regimes,
                'trading_viability_ratio': viable_regimes / len(unique_regimes) if unique_regimes.size > 0 else 0.0
            }

        except Exception as e:
            self.logger.error(f"❌ Trading viability calculation failed: {e}")
            return {'error': str(e), 'avg_trading_score': 0.0}

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            return abs(drawdown.min())
        except Exception:
            return 0.0

    def _calculate_multi_objective_score(self, regime_analysis: Dict[str, Any],
                                       cv_analysis: Dict[str, Any],
                                       economic_analysis: Dict[str, Any],
                                       trading_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate multi-objective optimization score."""
        try:
            # Extract key metrics
            distribution_quality = regime_analysis.get('distribution_quality', 0.0)
            cv_optimization_score = cv_analysis.get('cv_optimization_score', 0.0)
            economic_score = economic_analysis.get('avg_economic_score', 0.0)
            trading_score = trading_analysis.get('avg_trading_score', 0.0)

            # Weighted multi-objective score
            multi_objective_score = (
                0.25 * distribution_quality +    # Cluster distribution
                0.25 * cv_optimization_score +  # CV optimization
                0.25 * economic_score +         # Economic significance
                0.25 * trading_score            # Trading viability
            )

            return {
                'multi_objective_score': multi_objective_score,
                'distribution_quality': distribution_quality,
                'cv_optimization_score': cv_optimization_score,
                'economic_score': economic_score,
                'trading_score': trading_score,
                'weighted_components': {
                    'distribution_weight': 0.25,
                    'cv_weight': 0.25,
                    'economic_weight': 0.25,
                    'trading_weight': 0.25
                }
            }

        except Exception as e:
            self.logger.error(f"❌ Multi-objective score calculation failed: {e}")
            return {'multi_objective_score': 0.0}

    def _validate_cluster_distribution(self, regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate cluster distribution against targets."""
        try:
            n_regimes = regime_analysis.get('n_regimes', 0)
            regime_percentages = regime_analysis.get('regime_percentages', {})
            percentages = list(regime_percentages.values())

            # Check cluster count target
            cluster_count_valid = self.config.target_cluster_count_min <= n_regimes <= self.config.target_cluster_count_max

            # Check distribution limits
            max_distribution_valid = all(p <= self.config.max_cluster_distribution for p in percentages)
            min_distribution_valid = all(p >= self.config.min_cluster_distribution for p in percentages)

            # Calculate distribution penalty
            distribution_penalty = 0.0
            if not cluster_count_valid:
                if n_regimes < self.config.target_cluster_count_min:
                    distribution_penalty += (self.config.target_cluster_count_min - n_regimes) * 0.1
                elif n_regimes > self.config.target_cluster_count_max:
                    distribution_penalty += (n_regimes - self.config.target_cluster_count_max) * 0.1

            if not max_distribution_valid:
                excess_regimes = [p for p in percentages if p > self.config.max_cluster_distribution]
                distribution_penalty += sum(excess_regimes) * 0.2

            if not min_distribution_valid:
                small_regimes = [p for p in percentages if p < self.config.min_cluster_distribution]
                distribution_penalty += sum([self.config.min_cluster_distribution - p for p in small_regimes]) * 0.2

            # Calculate validation score
            validation_score = max(0.0, 1.0 - distribution_penalty)

            return {
                'cluster_count_valid': cluster_count_valid,
                'max_distribution_valid': max_distribution_valid,
                'min_distribution_valid': min_distribution_valid,
                'distribution_penalty': distribution_penalty,
                'validation_score': validation_score,
                'all_targets_met': cluster_count_valid and max_distribution_valid and min_distribution_valid
            }

        except Exception as e:
            self.logger.error(f"❌ Cluster distribution validation failed: {e}")
            return {'validation_score': 0.0, 'all_targets_met': False}

    def _calculate_distribution_quality(self, n_regimes: int, percentages: List[float],
                                      meets_cluster_count: bool, meets_max_distribution: bool,
                                      meets_min_distribution: bool) -> float:
        """Calculate distribution quality score."""
        try:
            base_score = 0.0

            # Cluster count score
            if meets_cluster_count:
                base_score += 0.4
            else:
                # Penalty for being outside target range
                if n_regimes < self.config.target_cluster_count_min:
                    penalty = (self.config.target_cluster_count_min - n_regimes) * 0.05
                else:
                    penalty = (n_regimes - self.config.target_cluster_count_max) * 0.05
                base_score += max(0.0, 0.4 - penalty)

            # Distribution limits score
            if meets_max_distribution:
                base_score += 0.3
            if meets_min_distribution:
                base_score += 0.3

            return min(base_score, 1.0)

        except Exception as e:
            self.logger.error(f"❌ Distribution quality calculation failed: {e}")
            return 0.0

    def _calculate_overall_quality(self, optimization_score: Dict[str, float],
                                 distribution_validation: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        try:
            multi_objective_score = optimization_score.get('multi_objective_score', 0.0)
            validation_score = distribution_validation.get('validation_score', 0.0)

            # Weighted overall quality
            overall_quality = 0.7 * multi_objective_score + 0.3 * validation_score

            return min(overall_quality, 1.0)

        except Exception as e:
            self.logger.error(f"❌ Overall quality calculation failed: {e}")
            return 0.0

    def get_evaluation_summary(self, results: Dict[str, Any]) -> str:
        """Get a summary of the evaluation results."""
        try:
            overall_quality = results.get('overall_quality', 0.0)
            optimization_score = results.get('optimization_score', {})
            distribution_validation = results.get('distribution_validation', {})

            summary = f"""
            💰 Enhanced Economic Evaluation Summary:
            🎯 Overall Quality: {overall_quality:.3f}
            📊 Multi-objective Score: {optimization_score.get('multi_objective_score', 0.0):.3f}
            📈 Distribution Validation: {distribution_validation.get('validation_score', 0.0):.3f}
            ✅ All Targets Met: {distribution_validation.get('all_targets_met', False)}
            """

            return summary.strip()

        except Exception as e:
            return f"❌ Failed to generate evaluation summary: {e}"
