"""
Consensus Validator for Hybrid NAS-TAS Regime Discovery.

Validates consensus predictions using multi-objective optimization with
statistical, economic, and temporal criteria.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class ConsensusValidationConfig:
    """Configuration for consensus validation."""
    # Statistical validation weights
    silhouette_weight: float = 0.25
    calinski_harabasz_weight: float = 0.20
    davies_bouldin_weight: float = 0.20
    inertia_weight: float = 0.15

    # Economic validation weights
    economic_significance_weight: float = 0.30
    trading_viability_weight: float = 0.25
    regime_stability_weight: float = 0.25
    cv_optimization_weight: float = 0.20

    # Temporal validation weights
    temporal_smoothness_weight: float = 0.30
    regime_duration_weight: float = 0.25
    transition_consistency_weight: float = 0.25
    persistence_weight: float = 0.20

    # Validation thresholds
    min_consensus_quality: float = 0.6
    min_statistical_score: float = 0.5
    min_economic_score: float = 0.5
    min_temporal_score: float = 0.5

    # Multi-objective optimization
    enable_multi_objective: bool = True
    pareto_frontier_size: int = 10
    convergence_threshold: float = 0.01

class ConsensusValidator:
    """
    Validates consensus predictions using multi-objective optimization.

    Combines statistical, economic, and temporal validation criteria
    to ensure high-quality regime clustering.
    """

    def __init__(self, config: Optional[ConsensusValidationConfig] = None):
        """Initialize the consensus validator."""
        self.config = config or ConsensusValidationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate_consensus(self, consensus_predictions: np.ndarray,
                         nas_result: Dict[str, Any],
                         tas_result: Dict[str, Any],
                         market_data: pd.DataFrame,
                         features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Validate consensus predictions using multi-objective criteria.

        Args:
            consensus_predictions: Consensus regime predictions
            nas_result: NAS regime detection results
            tas_result: TAS regime detection results
            market_data: Market data for validation
            features: Optional features for validation

        Returns:
            Comprehensive validation results
        """
        try:
            self.logger.info("🔍 Starting consensus validation with multi-objective optimization")

            # Statistical validation
            statistical_validation = self._validate_statistical_quality(
                consensus_predictions, market_data, features
            )

            # Economic validation
            economic_validation = self._validate_economic_quality(
                consensus_predictions, market_data, features
            )

            # Temporal validation
            temporal_validation = self._validate_temporal_quality(
                consensus_predictions, market_data
            )

            # Consensus consistency validation
            consistency_validation = self._validate_consensus_consistency(
                consensus_predictions, nas_result, tas_result
            )

            # Multi-objective optimization score
            if self.config.enable_multi_objective:
                optimization_result = self._perform_multi_objective_optimization(
                    statistical_validation, economic_validation, temporal_validation
                )
            else:
                optimization_result = self._calculate_simple_optimization_score(
                    statistical_validation, economic_validation, temporal_validation
                )

            # Overall consensus quality
            overall_quality = self._calculate_overall_consensus_quality(
                statistical_validation, economic_validation, temporal_validation,
                consistency_validation, optimization_result
            )

            # Validation summary
            validation_summary = self._generate_validation_summary(
                overall_quality, statistical_validation, economic_validation,
                temporal_validation, consistency_validation
            )

            results = {
                'statistical_validation': statistical_validation,
                'economic_validation': economic_validation,
                'temporal_validation': temporal_validation,
                'consistency_validation': consistency_validation,
                'optimization_result': optimization_result,
                'overall_quality': overall_quality,
                'validation_summary': validation_summary,
                'validation_passed': overall_quality >= self.config.min_consensus_quality
            }

            self.logger.info("✅ Consensus validation completed")
            return results

        except Exception as e:
            self.logger.error(f"❌ Consensus validation failed: {e}")
            return {'error': str(e), 'overall_quality': 0.0, 'validation_passed': False}

    def _validate_statistical_quality(self, consensus_predictions: np.ndarray,
                                    market_data: pd.DataFrame,
                                    features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Validate statistical quality of consensus predictions."""
        try:
            # Prepare features for clustering validation
            if features is not None and not features.empty:
                validation_features = features.values
            else:
                # Use market data features
                numeric_columns = market_data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    validation_features = market_data[numeric_columns].values
                else:
                    # Fallback to basic OHLCV
                    basic_columns = ['open', 'high', 'low', 'close', 'volume']
                    available_columns = [col for col in basic_columns if col in market_data.columns]
                    validation_features = market_data[available_columns].values if available_columns else market_data.values

            # Ensure same length
            min_length = min(len(consensus_predictions), len(validation_features))
            consensus_predictions = consensus_predictions[:min_length]
            validation_features = validation_features[:min_length]

            # Calculate clustering metrics
            unique_regimes = np.unique(consensus_predictions)
            n_clusters = len(unique_regimes)

            if n_clusters < 2:
                return {
                    'silhouette_score': 0.0,
                    'calinski_harabasz_score': 0.0,
                    'davies_bouldin_score': float('inf'),
                    'inertia': 0.0,
                    'statistical_score': 0.0,
                    'n_clusters': n_clusters
                }

            # Silhouette score
            silhouette = silhouette_score(validation_features, consensus_predictions)

            # Calinski-Harabasz score
            calinski_harabasz = calinski_harabasz_score(validation_features, consensus_predictions)

            # Davies-Bouldin score
            davies_bouldin = davies_bouldin_score(validation_features, consensus_predictions)

            # Inertia (within-cluster sum of squares)
            inertia = self._calculate_inertia(validation_features, consensus_predictions)

            # Normalize scores
            normalized_silhouette = max(0, silhouette)  # Higher is better
            normalized_calinski = min(calinski_harabasz / 1000, 1.0)  # Normalize
            normalized_davies_bouldin = max(0, 1 - davies_bouldin / 10)  # Lower is better
            normalized_inertia = max(0, 1 - inertia / 10000)  # Lower is better

            # Weighted statistical score
            statistical_score = (
                self.config.silhouette_weight * normalized_silhouette +
                self.config.calinski_harabasz_weight * normalized_calinski +
                self.config.davies_bouldin_weight * normalized_davies_bouldin +
                self.config.inertia_weight * normalized_inertia
            )

            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin,
                'inertia': inertia,
                'statistical_score': statistical_score,
                'n_clusters': n_clusters,
                'normalized_scores': {
                    'silhouette': normalized_silhouette,
                    'calinski_harabasz': normalized_calinski,
                    'davies_bouldin': normalized_davies_bouldin,
                    'inertia': normalized_inertia
                }
            }

        except Exception as e:
            self.logger.error(f"❌ Statistical validation failed: {e}")
            return {'error': str(e), 'statistical_score': 0.0}

    def _validate_economic_quality(self, consensus_predictions: np.ndarray,
                                 market_data: pd.DataFrame,
                                 features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Validate economic quality of consensus predictions."""
        try:
            unique_regimes = np.unique(consensus_predictions)
            economic_scores = {}

            for regime in unique_regimes:
                regime_mask = consensus_predictions == regime
                regime_data = market_data[regime_mask]

                if len(regime_data) < 5:  # Minimum regime size
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
                duration_score = min(duration / 100, 1.0)

                # Economic significance score
                economic_score = (
                    0.3 * abs(sharpe_ratio) +  # Risk-adjusted return
                    0.2 * volume_consistency +  # Volume stability
                    0.2 * duration_score +      # Regime persistence
                    0.2 * abs(mean_return) +   # Absolute return
                    0.1 * (1 - max_drawdown)   # Drawdown penalty
                )

                economic_scores[regime] = min(economic_score, 1.0)

            # Calculate overall economic quality
            avg_economic_score = np.mean(list(economic_scores.values())) if economic_scores else 0.0
            significant_regimes = len([s for s in economic_scores.values() if s >= 0.5])

            # Trading viability assessment
            trading_viability = self._assess_trading_viability(consensus_predictions, market_data)

            # Regime stability assessment
            regime_stability = self._assess_regime_stability(consensus_predictions)

            # CV optimization assessment
            cv_optimization = self._assess_cv_optimization(consensus_predictions, market_data, features)

            # Weighted economic score
            economic_quality_score = (
                self.config.economic_significance_weight * avg_economic_score +
                self.config.trading_viability_weight * trading_viability +
                self.config.regime_stability_weight * regime_stability +
                self.config.cv_optimization_weight * cv_optimization
            )

            return {
                'regime_economic_scores': economic_scores,
                'avg_economic_score': avg_economic_score,
                'significant_regimes_count': significant_regimes,
                'trading_viability': trading_viability,
                'regime_stability': regime_stability,
                'cv_optimization': cv_optimization,
                'economic_quality_score': economic_quality_score
            }

        except Exception as e:
            self.logger.error(f"❌ Economic validation failed: {e}")
            return {'error': str(e), 'economic_quality_score': 0.0}

    def _validate_temporal_quality(self, consensus_predictions: np.ndarray,
                                 market_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate temporal quality of consensus predictions."""
        try:
            # Temporal smoothness (minimize regime changes)
            regime_changes = np.sum(consensus_predictions[1:] != consensus_predictions[:-1])
            temporal_smoothness = 1 - (regime_changes / len(consensus_predictions))

            # Regime duration analysis
            regime_durations = self._calculate_regime_durations(consensus_predictions)
            avg_duration = np.mean(regime_durations) if regime_durations else 0
            duration_consistency = min(avg_duration / 50, 1.0)  # Normalize

            # Transition consistency
            transition_consistency = self._assess_transition_consistency(consensus_predictions)

            # Regime persistence
            persistence = self._assess_regime_persistence(consensus_predictions)

            # Weighted temporal score
            temporal_score = (
                self.config.temporal_smoothness_weight * temporal_smoothness +
                self.config.regime_duration_weight * duration_consistency +
                self.config.transition_consistency_weight * transition_consistency +
                self.config.persistence_weight * persistence
            )

            return {
                'temporal_smoothness': temporal_smoothness,
                'regime_durations': regime_durations,
                'avg_duration': avg_duration,
                'duration_consistency': duration_consistency,
                'transition_consistency': transition_consistency,
                'persistence': persistence,
                'temporal_score': temporal_score
            }

        except Exception as e:
            self.logger.error(f"❌ Temporal validation failed: {e}")
            return {'error': str(e), 'temporal_score': 0.0}

    def _validate_consensus_consistency(self, consensus_predictions: np.ndarray,
                                      nas_result: Dict[str, Any],
                                      tas_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate consistency between consensus and individual method results."""
        try:
            # Extract individual predictions
            nas_predictions = nas_result.get('regime_predictions', [])
            tas_predictions = tas_result.get('regime_predictions', [])

            if len(nas_predictions) == 0 or len(tas_predictions) == 0:
                return {'consistency_score': 0.0, 'nas_consistency': 0.0, 'tas_consistency': 0.0}

            # Align lengths
            min_length = min(len(consensus_predictions), len(nas_predictions), len(tas_predictions))
            consensus_predictions = consensus_predictions[:min_length]
            nas_predictions = nas_predictions[:min_length]
            tas_predictions = tas_predictions[:min_length]

            # Calculate consistency with NAS
            nas_agreements = np.sum(consensus_predictions == nas_predictions)
            nas_consistency = nas_agreements / min_length

            # Calculate consistency with TAS
            tas_agreements = np.sum(consensus_predictions == tas_predictions)
            tas_consistency = tas_agreements / min_length

            # Overall consistency
            overall_consistency = (nas_consistency + tas_consistency) / 2

            # Calculate consensus improvement
            nas_tas_agreement = np.sum(nas_predictions == tas_predictions) / min_length
            consensus_improvement = overall_consistency - nas_tas_agreement

            return {
                'consistency_score': overall_consistency,
                'nas_consistency': nas_consistency,
                'tas_consistency': tas_consistency,
                'nas_tas_agreement': nas_tas_agreement,
                'consensus_improvement': consensus_improvement
            }

        except Exception as e:
            self.logger.error(f"❌ Consensus consistency validation failed: {e}")
            return {'error': str(e), 'consistency_score': 0.0}

    def _perform_multi_objective_optimization(self, statistical_validation: Dict[str, Any],
                                           economic_validation: Dict[str, Any],
                                           temporal_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Perform multi-objective optimization using Pareto frontier approach."""
        try:
            # Extract objective scores
            statistical_score = statistical_validation.get('statistical_score', 0.0)
            economic_score = economic_validation.get('economic_quality_score', 0.0)
            temporal_score = temporal_validation.get('temporal_score', 0.0)

            # Calculate Pareto efficiency
            objectives = [statistical_score, economic_score, temporal_score]
            pareto_efficiency = self._calculate_pareto_efficiency(objectives)

            # Calculate weighted sum
            weighted_sum = (
                0.4 * statistical_score +
                0.4 * economic_score +
                0.2 * temporal_score
            )

            # Calculate multi-objective score
            multi_objective_score = 0.7 * weighted_sum + 0.3 * pareto_efficiency

            return {
                'statistical_score': statistical_score,
                'economic_score': economic_score,
                'temporal_score': temporal_score,
                'pareto_efficiency': pareto_efficiency,
                'weighted_sum': weighted_sum,
                'multi_objective_score': multi_objective_score,
                'objectives': objectives
            }

        except Exception as e:
            self.logger.error(f"❌ Multi-objective optimization failed: {e}")
            return {'error': str(e), 'multi_objective_score': 0.0}

    def _calculate_simple_optimization_score(self, statistical_validation: Dict[str, Any],
                                           economic_validation: Dict[str, Any],
                                           temporal_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate simple optimization score without Pareto analysis."""
        try:
            statistical_score = statistical_validation.get('statistical_score', 0.0)
            economic_score = economic_validation.get('economic_quality_score', 0.0)
            temporal_score = temporal_validation.get('temporal_score', 0.0)

            # Simple weighted average
            optimization_score = (
                0.4 * statistical_score +
                0.4 * economic_score +
                0.2 * temporal_score
            )

            return {
                'statistical_score': statistical_score,
                'economic_score': economic_score,
                'temporal_score': temporal_score,
                'optimization_score': optimization_score
            }

        except Exception as e:
            self.logger.error(f"❌ Simple optimization score calculation failed: {e}")
            return {'error': str(e), 'optimization_score': 0.0}

    def _calculate_overall_consensus_quality(self, statistical_validation: Dict[str, Any],
                                           economic_validation: Dict[str, Any],
                                           temporal_validation: Dict[str, Any],
                                           consistency_validation: Dict[str, Any],
                                           optimization_result: Dict[str, Any]) -> float:
        """Calculate overall consensus quality score."""
        try:
            # Extract scores
            statistical_score = statistical_validation.get('statistical_score', 0.0)
            economic_score = economic_validation.get('economic_quality_score', 0.0)
            temporal_score = temporal_validation.get('temporal_score', 0.0)
            consistency_score = consistency_validation.get('consistency_score', 0.0)
            optimization_score = optimization_result.get('multi_objective_score',
                                                      optimization_result.get('optimization_score', 0.0))

            # Weighted overall quality
            overall_quality = (
                0.25 * statistical_score +
                0.25 * economic_score +
                0.20 * temporal_score +
                0.15 * consistency_score +
                0.15 * optimization_score
            )

            return min(overall_quality, 1.0)

        except Exception as e:
            self.logger.error(f"❌ Overall consensus quality calculation failed: {e}")
            return 0.0

    # Helper methods
    def _calculate_inertia(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate within-cluster sum of squares (inertia)."""
        try:
            unique_labels = np.unique(labels)
            inertia = 0.0

            for label in unique_labels:
                cluster_points = features[labels == label]
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    cluster_inertia = np.sum((cluster_points - centroid) ** 2)
                    inertia += cluster_inertia

            return inertia

        except Exception:
            return 0.0

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            return abs(drawdown.min())
        except Exception:
            return 0.0

    def _assess_trading_viability(self, predictions: np.ndarray, market_data: pd.DataFrame) -> float:
        """Assess trading viability of regime predictions."""
        try:
            # Calculate regime stability
            regime_changes = np.sum(predictions[1:] != predictions[:-1])
            stability = 1 - (regime_changes / len(predictions))

            # Calculate volume liquidity
            if 'volume' in market_data.columns:
                volume_mean = market_data['volume'].mean()
                liquidity = min(volume_mean / 1000, 1.0)
            else:
                liquidity = 0.5

            return (stability + liquidity) / 2

        except Exception:
            return 0.0

    def _assess_regime_stability(self, predictions: np.ndarray) -> float:
        """Assess regime stability."""
        try:
            regime_changes = np.sum(predictions[1:] != predictions[:-1])
            stability = 1 - (regime_changes / len(predictions))
            return stability
        except Exception:
            return 0.0

    def _assess_cv_optimization(self, predictions: np.ndarray, market_data: pd.DataFrame,
                              features: Optional[pd.DataFrame] = None) -> float:
        """Assess CV optimization quality."""
        try:
            # This would integrate with the EnhancedEconomicEvaluator
            # For now, return a placeholder score
            return 0.7
        except Exception:
            return 0.0

    def _calculate_regime_durations(self, predictions: np.ndarray) -> List[int]:
        """Calculate durations of each regime."""
        try:
            durations = []
            current_regime = predictions[0]
            current_duration = 1

            for i in range(1, len(predictions)):
                if predictions[i] == current_regime:
                    current_duration += 1
                else:
                    durations.append(current_duration)
                    current_regime = predictions[i]
                    current_duration = 1

            durations.append(current_duration)
            return durations

        except Exception:
            return []

    def _assess_transition_consistency(self, predictions: np.ndarray) -> float:
        """Assess consistency of regime transitions."""
        try:
            # Calculate transition probabilities
            transitions = {}
            for i in range(len(predictions) - 1):
                from_regime = predictions[i]
                to_regime = predictions[i + 1]
                transition = (from_regime, to_regime)
                transitions[transition] = transitions.get(transition, 0) + 1

            # Calculate consistency (lower entropy = more consistent)
            total_transitions = sum(transitions.values())
            if total_transitions == 0:
                return 0.0

            probabilities = [count / total_transitions for count in transitions.values()]
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            max_entropy = np.log2(len(transitions)) if len(transitions) > 1 else 1

            consistency = 1 - (entropy / max_entropy) if max_entropy > 0 else 1.0
            return consistency

        except Exception:
            return 0.0

    def _assess_regime_persistence(self, predictions: np.ndarray) -> float:
        """Assess regime persistence."""
        try:
            unique_regimes = np.unique(predictions)
            persistence_scores = []

            for regime in unique_regimes:
                regime_mask = predictions == regime
                regime_indices = np.where(regime_mask)[0]

                if len(regime_indices) > 1:
                    # Calculate average gap between regime occurrences
                    gaps = np.diff(regime_indices)
                    avg_gap = np.mean(gaps) if len(gaps) > 0 else 0
                    persistence = 1 / (1 + avg_gap / 10)  # Normalize
                    persistence_scores.append(persistence)

            return np.mean(persistence_scores) if persistence_scores else 0.0

        except Exception:
            return 0.0

    def _calculate_pareto_efficiency(self, objectives: List[float]) -> float:
        """Calculate Pareto efficiency score."""
        try:
            if len(objectives) < 2:
                return 1.0

            # Simple Pareto efficiency calculation
            # Higher scores are better, so we want to maximize the minimum
            min_objective = min(objectives)
            max_objective = max(objectives)

            if max_objective == 0:
                return 0.0

            # Efficiency is the ratio of minimum to maximum
            efficiency = min_objective / max_objective
            return efficiency

        except Exception:
            return 0.0

    def _generate_validation_summary(self, overall_quality: float,
                                  statistical_validation: Dict[str, Any],
                                  economic_validation: Dict[str, Any],
                                  temporal_validation: Dict[str, Any],
                                  consistency_validation: Dict[str, Any]) -> str:
        """Generate validation summary."""
        try:
            summary = f"""
            🔍 Consensus Validation Summary:
            🎯 Overall Quality: {overall_quality:.3f}
            📊 Statistical Score: {statistical_validation.get('statistical_score', 0.0):.3f}
            💰 Economic Score: {economic_validation.get('economic_quality_score', 0.0):.3f}
            ⏰ Temporal Score: {temporal_validation.get('temporal_score', 0.0):.3f}
            🔗 Consistency Score: {consistency_validation.get('consistency_score', 0.0):.3f}
            ✅ Validation Passed: {overall_quality >= self.config.min_consensus_quality}
            """

            return summary.strip()

        except Exception as e:
            return f"❌ Failed to generate validation summary: {e}"
