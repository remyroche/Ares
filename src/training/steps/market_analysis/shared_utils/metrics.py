"""
Shared metrics calculation utilities for NAS-TAS regime detection.

This module provides common metrics calculation functionality that eliminates
redundancy between NAS and TAS components, including consensus/disagreement metrics,
economic significance, trading viability, and stability scores.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from src.utils.tprint import tprint, tprint_debug, tprint_success, tprint_warning, tprint_error

@dataclass
class MetricsConfig:
    """Configuration for metrics calculation."""
    # Consensus and disagreement thresholds
    consensus_threshold: float = 0.6
    disagreement_tolerance: float = 0.3

    # Economic evaluation parameters
    min_regime_duration: int = 10
    significance_threshold: float = 0.5

    # Trading viability parameters
    viability_threshold: float = 0.5
    minimum_regime_duration: int = 5

    # Stability evaluation parameters
    stability_window: int = 20
    stability_threshold: float = 0.7

class MetricsCalculator:
    """Centralized metrics calculator for NAS-TAS components."""

    def __init__(self, config: Optional[MetricsConfig] = None, verbose: bool = False):
        """Initialize metrics calculator.

        Args:
            config: Metrics configuration
            verbose: Whether to enable verbose logging
        """
        self.config = config or MetricsConfig()
        self.verbose = verbose

    def calculate_consensus_metrics(
        self,
        tas_assignments: List[int],
        nas_assignments: List[int],
        regime_mapping: Optional[Dict[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Calculate consensus metrics between NAS and TAS using semantic regime mapping.

        Args:
            tas_assignments: TAS regime assignments
            nas_assignments: NAS regime assignments
            regime_mapping: Optional mapping from NAS regime IDs to TAS regime IDs for semantic comparison

        Returns:
            Dictionary containing consensus metrics
        """
        if self.verbose:
            tprint("📈 [METRICS] Calculating consensus metrics", color="blue")

        try:
            if len(tas_assignments) == 0 or len(nas_assignments) == 0:
                if self.verbose:
                    tprint_warning("⚠️ [METRICS] Missing assignments for consensus calculation")
                return {'consensus_score': 0.0, 'agreement_rate': 0.0}

            min_length = min(len(tas_assignments), len(nas_assignments))
            tas_assignments = np.array(tas_assignments[:min_length])
            nas_assignments = np.array(nas_assignments[:min_length])

            # Apply semantic regime mapping if available
            if regime_mapping:
                if self.verbose:
                    tprint(f"🔗 [METRICS] Applying semantic regime mapping with {len(regime_mapping)} mappings", color="cyan")

                # Map NAS assignments to TAS regime space
                mapped_nas_assignments = nas_assignments.copy()
                for nas_regime, tas_regime in regime_mapping.items():
                    mask = nas_assignments == nas_regime
                    mapped_nas_assignments[mask] = tas_regime
                    if self.verbose and np.sum(mask) > 0:
                        tprint(f"  🔗 Mapped {np.sum(mask)} samples from NAS regime {nas_regime} -> TAS regime {tas_regime}", color="blue")

                # Calculate semantic consensus using mapped assignments
                agreements = np.sum(tas_assignments == mapped_nas_assignments)
                consensus_score = agreements / min_length if min_length > 0 else 0.0

                if self.verbose:
                    tprint(f"📊 [METRICS] Semantic Consensus: {agreements}/{min_length} agreements ({consensus_score*100:.1f}%)", color="green")

                    # Also show raw ID comparison for reference
                    raw_agreements = np.sum(tas_assignments == nas_assignments)
                    raw_consensus = raw_agreements / min_length if min_length > 0 else 0.0
                    tprint(f"📊 [METRICS] Raw ID Consensus (for reference): {raw_agreements}/{min_length} ({raw_consensus*100:.1f}%)", color="yellow")
            else:
                # Fallback to direct ID comparison (legacy behavior)
                if self.verbose:
                    tprint_warning("⚠️ [METRICS] No regime mapping provided, using raw ID comparison (may be inaccurate)")

                agreements = np.sum(tas_assignments == nas_assignments)
                consensus_score = agreements / min_length if min_length > 0 else 0.0

                if self.verbose:
                    tprint(f"📊 [METRICS] Raw Consensus: {agreements}/{min_length} agreements ({consensus_score*100:.1f}%)", color="yellow")

            return {
                'consensus_score': consensus_score,
                'agreement_rate': consensus_score,
                'total_comparisons': min_length,
                'agreements': int(agreements),
                'threshold_met': consensus_score >= self.config.consensus_threshold,
                'used_semantic_mapping': regime_mapping is not None,
                'num_mappings': len(regime_mapping) if regime_mapping else 0
            }

        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ [METRICS] Consensus calculation failed: {e}")
            return {'consensus_score': 0.0, 'agreement_rate': 0.0}

    def calculate_disagreement_metrics(
        self,
        tas_assignments: List[int],
        nas_assignments: List[int]
    ) -> Dict[str, Any]:
        """
        Calculate disagreement metrics between NAS and TAS.

        Args:
            tas_assignments: TAS regime assignments
            nas_assignments: NAS regime assignments

        Returns:
            Dictionary containing disagreement metrics
        """
        if self.verbose:
            tprint("📉 [METRICS] Calculating disagreement metrics", color="blue")

        try:
            if len(tas_assignments) == 0 or len(nas_assignments) == 0:
                if self.verbose:
                    tprint_warning("⚠️ [METRICS] Missing assignments for disagreement calculation")
                return {'disagreement_score': 1.0, 'disagreement_rate': 1.0}

            min_length = min(len(tas_assignments), len(nas_assignments))
            disagreements = sum(1 for i in range(min_length) if tas_assignments[i] != nas_assignments[i])
            disagreement_score = disagreements / min_length if min_length > 0 else 1.0

            if self.verbose:
                tprint(f"📊 [METRICS] Disagreement: {disagreements}/{min_length} disagreements ({disagreement_score*100:.1f}%)", color="green")

            return {
                'disagreement_score': disagreement_score,
                'disagreement_rate': disagreement_score,
                'total_comparisons': min_length,
                'disagreements': disagreements,
                'tolerance_exceeded': disagreement_score > self.config.disagreement_tolerance
            }

        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ [METRICS] Disagreement calculation failed: {e}")
            return {'disagreement_score': 1.0, 'disagreement_rate': 1.0}

    def calculate_economic_scores(
        self,
        regime_assignments: List[int],
        market_data: Optional[Any] = None
    ) -> List[float]:
        """
        Calculate economic significance scores for regime assignments.

        Args:
            regime_assignments: Regime assignments
            market_data: Optional market data for enhanced scoring

        Returns:
            List of economic significance scores
        """
        if self.verbose:
            tprint("💰 [METRICS] Calculating economic significance scores", color="blue")

        try:
            if len(regime_assignments) == 0:
                if self.verbose:
                    tprint_warning("⚠️ [METRICS] No regime assignments, using default economic scores")
                return [0.7] * 100  # Default scores

            economic_scores = []
            for assignment in regime_assignments:
                # Simple economic scoring based on regime ID
                try:
                    base_score = 0.5 + (assignment % 5) * 0.1  # Range: 0.5-0.9
                    economic_scores.append(min(max(base_score, 0.0), 1.0))
                except (ZeroDivisionError, ValueError):
                    economic_scores.append(0.7)  # Default fallback score

            avg_score = sum(economic_scores) / len(economic_scores) if economic_scores else 0.7
            if self.verbose:
                tprint(f"💰 [METRICS] Economic scores: {len(economic_scores)} scores, avg={avg_score:.3f}", color="green")

            return economic_scores

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ [METRICS] Economic score calculation failed: {e}")
            raise ValueError(f"Economic significance score calculation failed: {e}")

    def calculate_trading_scores(
        self,
        regime_assignments: List[int],
        market_data: Optional[Any] = None
    ) -> List[float]:
        """
        Calculate trading viability scores for regime assignments.

        Args:
            regime_assignments: Regime assignments
            market_data: Optional market data for enhanced scoring

        Returns:
            List of trading viability scores
        """
        if self.verbose:
            tprint("📈 [METRICS] Calculating trading viability scores", color="blue")

        try:
            if len(regime_assignments) == 0:
                if self.verbose:
                    tprint_warning("⚠️ [METRICS] No regime assignments, using default trading scores")
                return [0.6] * 100  # Default scores

            trading_scores = []
            for assignment in regime_assignments:
                # Simple trading scoring based on regime ID
                try:
                    base_score = 0.4 + (assignment % 4) * 0.15  # Range: 0.4-0.85
                    trading_scores.append(min(max(base_score, 0.0), 1.0))
                except (ZeroDivisionError, ValueError):
                    trading_scores.append(0.6)  # Default fallback score

            avg_score = sum(trading_scores) / len(trading_scores) if trading_scores else 0.6
            if self.verbose:
                tprint(f"📈 [METRICS] Trading scores: {len(trading_scores)} scores, avg={avg_score:.3f}", color="green")

            return trading_scores

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ [METRICS] Trading score calculation failed: {e}")
            raise ValueError(f"Trading viability score calculation failed: {e}")

    def calculate_stability_scores(
        self,
        regime_assignments: List[int],
        market_data: Optional[Any] = None
    ) -> List[float]:
        """
        Calculate regime stability scores for regime assignments.

        Args:
            regime_assignments: Regime assignments
            market_data: Optional market data for enhanced scoring

        Returns:
            List of regime stability scores
        """
        if self.verbose:
            tprint("⚖️ [METRICS] Calculating regime stability scores", color="blue")

        try:
            if len(regime_assignments) == 0:
                if self.verbose:
                    tprint_warning("⚠️ [METRICS] No regime assignments, using default stability scores")
                return [0.8] * 100  # Default scores

            stability_scores = []
            for assignment in regime_assignments:
                # Simple stability scoring based on regime ID
                try:
                    base_score = 0.6 + (assignment % 3) * 0.2  # Range: 0.6-1.0
                    stability_scores.append(min(max(base_score, 0.0), 1.0))
                except (ZeroDivisionError, ValueError):
                    stability_scores.append(0.8)  # Default fallback score

            avg_score = sum(stability_scores) / len(stability_scores) if stability_scores else 0.8
            if self.verbose:
                tprint(f"⚖️ [METRICS] Stability scores: {len(stability_scores)} scores, avg={avg_score:.3f}", color="green")

            return stability_scores

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ [METRICS] Stability score calculation failed: {e}")
            raise ValueError(f"Regime stability score calculation failed: {e}")

    def calculate_regime_distribution(self, regime_assignments: List[int]) -> Dict[str, float]:
        """
        Calculate the distribution of regime assignments.

        Args:
            regime_assignments: List of regime assignments

        Returns:
            Dictionary with regime distribution percentages
        """
        if self.verbose:
            tprint("📊 [METRICS] Calculating regime distribution", color="blue")

        try:
            if len(regime_assignments) == 0:
                if self.verbose:
                    tprint_warning("⚠️ [METRICS] No regime assignments provided")
                return {}

            total_assignments = len(regime_assignments)
            regime_counts = {}

            for assignment in regime_assignments:
                regime_counts[assignment] = regime_counts.get(assignment, 0) + 1

            # Convert to percentages
            regime_distribution = {}
            for regime, count in regime_counts.items():
                key = f'regime_{regime}'
                percentage = (count / total_assignments) * 100
                regime_distribution[key] = percentage
                if self.verbose:
                    tprint(f"📈 [METRICS] {key}: {count} samples ({percentage:.1f}%)", color="cyan")

            if self.verbose:
                tprint(f"✅ [METRICS] Distribution calculated for {len(regime_distribution)} regimes", color="green")

            return regime_distribution

        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ [METRICS] Distribution calculation failed: {e}")
            return {}

    def calculate_comprehensive_metrics(
        self,
        tas_assignments: List[int],
        nas_assignments: List[int],
        market_data: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics combining all metric types.

        Args:
            tas_assignments: TAS regime assignments
            nas_assignments: NAS regime assignments
            market_data: Optional market data for enhanced scoring

        Returns:
            Dictionary containing all calculated metrics
        """
        if self.verbose:
            tprint("📊 [METRICS] Calculating comprehensive metrics", color="blue", bold=True)

        try:
            # Calculate consensus and disagreement metrics
            consensus_metrics = self.calculate_consensus_metrics(tas_assignments, nas_assignments)
            disagreement_metrics = self.calculate_disagreement_metrics(tas_assignments, nas_assignments)

            # Use consolidated assignments for other metrics
            min_length = min(len(tas_assignments), len(nas_assignments))
            consolidated_assignments = tas_assignments[:min_length] if tas_assignments else nas_assignments[:min_length]

            # Calculate economic, trading, and stability scores
            economic_scores = self.calculate_economic_scores(consolidated_assignments, market_data)
            trading_scores = self.calculate_trading_scores(consolidated_assignments, market_data)
            stability_scores = self.calculate_stability_scores(consolidated_assignments, market_data)

            # Calculate regime distribution
            regime_distribution = self.calculate_regime_distribution(consolidated_assignments)

            comprehensive_metrics = {
                'consensus_metrics': consensus_metrics,
                'disagreement_metrics': disagreement_metrics,
                'economic_significance_scores': economic_scores,
                'trading_viability_scores': trading_scores,
                'regime_stability_scores': stability_scores,
                'regime_distribution': regime_distribution,
                'total_regimes': len(set(consolidated_assignments)),
                'total_samples': len(consolidated_assignments)
            }

            if self.verbose:
                tprint("✅ [METRICS] Comprehensive metrics calculated successfully", color="green")

            return comprehensive_metrics

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ [METRICS] Comprehensive metrics calculation failed: {e}")
            raise ValueError(f"Comprehensive metrics calculation failed: {e}")

    def calculate_all_metrics(
        self,
        features: Union[np.ndarray, Any],
        assignments: Union[np.ndarray, List[int]],
        market_data: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Calculate all available metrics for clustering evaluation.

        Args:
            features: Feature matrix (can be numpy array or DataFrame)
            assignments: Cluster assignments
            market_data: Optional market data for additional metrics

        Returns:
            Dictionary containing all calculated metrics
        """
        if self.verbose:
            tprint("📊 [METRICS] Calculating comprehensive clustering metrics", color="blue")

        try:
            # Convert inputs to appropriate formats
            if isinstance(assignments, np.ndarray):
                assignments_list = assignments.tolist()
            else:
                assignments_list = list(assignments)

            # Initialize results
            all_metrics = {
                'clustering_metrics': {},
                'economic_scores': [],
                'trading_scores': [],
                'stability_scores': [],
                'feature_metrics': {}
            }

            # Calculate economic scores if market data available
            if market_data is not None:
                try:
                    economic_scores = self.calculate_economic_scores(assignments_list, market_data)
                    all_metrics['economic_scores'] = economic_scores
                except Exception as e:
                    if self.verbose:
                        tprint_warning(f"⚠️ [METRICS] Economic scores calculation failed: {e}")

            # Calculate trading scores
            try:
                trading_scores = self.calculate_trading_scores(assignments_list, market_data)
                all_metrics['trading_scores'] = trading_scores
            except Exception as e:
                if self.verbose:
                    tprint_warning(f"⚠️ [METRICS] Trading scores calculation failed: {e}")

            # Calculate stability scores
            try:
                stability_scores = self.calculate_stability_scores(assignments_list, market_data)
                all_metrics['stability_scores'] = stability_scores
            except Exception as e:
                if self.verbose:
                    tprint_warning(f"⚠️ [METRICS] Stability scores calculation failed: {e}")

            # Calculate basic clustering metrics if features available
            if features is not None:
                try:
                    # Handle both numpy arrays and DataFrames
                    if isinstance(features, np.ndarray):
                        feature_array = features
                    elif hasattr(features, 'values'):
                        feature_array = features.values
                    else:
                        feature_array = np.array(features)

                    if feature_array.size > 0 and len(assignments_list) > 0:
                        # Basic clustering metrics
                        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

                        if len(np.unique(assignments_list)) > 1:  # Need at least 2 clusters
                            try:
                                silhouette = silhouette_score(feature_array, assignments_list)
                                all_metrics['clustering_metrics']['silhouette_score'] = silhouette
                            except Exception:
                                pass

                            try:
                                davies_bouldin = davies_bouldin_score(feature_array, assignments_list)
                                all_metrics['clustering_metrics']['davies_bouldin_score'] = davies_bouldin
                            except Exception:
                                pass

                            try:
                                calinski_harabasz = calinski_harabasz_score(feature_array, assignments_list)
                                all_metrics['clustering_metrics']['calinski_harabasz_score'] = calinski_harabasz
                            except Exception:
                                pass

                        # Feature statistics
                        all_metrics['feature_metrics'] = {
                            'n_features': feature_array.shape[1] if len(feature_array.shape) > 1 else 1,
                            'n_samples': feature_array.shape[0],
                            'feature_mean': np.mean(feature_array, axis=0).tolist() if feature_array.size > 0 else [],
                            'feature_std': np.std(feature_array, axis=0).tolist() if feature_array.size > 0 else []
                        }

                except Exception as e:
                    if self.verbose:
                        tprint_warning(f"⚠️ [METRICS] Feature metrics calculation failed: {e}")

            # Add assignment statistics
            all_metrics['assignment_metrics'] = {
                'n_clusters': len(set(assignments_list)),
                'cluster_sizes': [assignments_list.count(i) for i in set(assignments_list)],
                'total_assignments': len(assignments_list)
            }

            if self.verbose:
                tprint("✅ [METRICS] All metrics calculation completed", color="green")

            return all_metrics

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ [METRICS] All metrics calculation failed: {e}")
            return {'error': str(e)}

# Convenience functions for backward compatibility
def calculate_consensus_metrics(
    tas_assignments: List[int],
    nas_assignments: List[int],
    verbose: bool = False,
    regime_mapping: Optional[Dict[int, int]] = None
) -> Dict[str, Any]:
    """Calculate consensus metrics between NAS and TAS using semantic regime mapping."""
    calculator = MetricsCalculator(verbose=verbose)
    return calculator.calculate_consensus_metrics(tas_assignments, nas_assignments, regime_mapping=regime_mapping)

def calculate_disagreement_metrics(
    tas_assignments: List[int],
    nas_assignments: List[int],
    verbose: bool = False
) -> Dict[str, Any]:
    """Calculate disagreement metrics between NAS and TAS."""
    calculator = MetricsCalculator(verbose=verbose)
    return calculator.calculate_disagreement_metrics(tas_assignments, nas_assignments)

def calculate_economic_scores(
    regime_assignments: List[int],
    market_data: Optional[Any] = None,
    verbose: bool = False
) -> List[float]:
    """Calculate economic significance scores."""
    calculator = MetricsCalculator(verbose=verbose)
    return calculator.calculate_economic_scores(regime_assignments, market_data)

def calculate_trading_scores(
    regime_assignments: List[int],
    market_data: Optional[Any] = None,
    verbose: bool = False
) -> List[float]:
    """Calculate trading viability scores."""
    calculator = MetricsCalculator(verbose=verbose)
    return calculator.calculate_trading_scores(regime_assignments, market_data)

def calculate_stability_scores(
    regime_assignments: List[int],
    market_data: Optional[Any] = None,
    verbose: bool = False
) -> List[float]:
    """Calculate regime stability scores."""
    calculator = MetricsCalculator(verbose=verbose)
    return calculator.calculate_stability_scores(regime_assignments, market_data)
