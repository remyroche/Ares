#!/usr/bin/env python3
"""
HMM-Appropriate Validation Metrics

This module implements validation metrics specifically designed for Hidden Markov Models
regime detection, replacing traditional clustering metrics that are inappropriate for
temporal regime modeling.

Key Metrics:
1. Temporal Coherence - Replaces Silhouette Score
2. Regime Transition Quality - Replaces Davies-Bouldin Score  
3. Economic Differentiation Index - Replaces Calinski-Harabasz Score
4. HMM-Specific Validation - Combines temporal, economic, and spatial validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings

logger = logging.getLogger(__name__)

@dataclass
class TemporalCoherenceMetrics:
    """Metrics for temporal coherence validation."""
    temporal_coherence: float
    avg_regime_duration: float
    duration_stability: float
    too_short_ratio: float
    regime_consistency: float
    interpretation: str

@dataclass
class TransitionQualityMetrics:
    """Metrics for regime transition quality validation."""
    transition_quality: float
    avg_persistence: float
    transition_entropy: float
    transition_clarity: float
    persistence_consistency: float
    interpretation: str

@dataclass
class EconomicDifferentiationMetrics:
    """Comprehensive economic differentiation metrics."""
    economic_differentiation: float
    return_differentiation: float
    volatility_differentiation: float
    sharpe_differentiation: float
    risk_return_tradeoff: float
    regime_economic_distinctness: float
    market_efficiency_impact: float
    regime_stats: Dict[int, Dict[str, float]]
    interpretation: str

@dataclass
class DetailedValidationReport:
    """Detailed validation report with comprehensive metrics."""
    execution_summary: Dict[str, Any]
    temporal_analysis: Dict[str, Any]
    transition_analysis: Dict[str, Any]
    economic_analysis: Dict[str, Any]
    spatial_analysis: Dict[str, Any]
    regime_characteristics: Dict[str, Any]
    comparative_analysis: Dict[str, Any]
    recommendations: Dict[str, Any]
    quality_assessment: Dict[str, Any]

@dataclass
class HMMValidationMetrics:
    """Combined HMM validation metrics."""
    hmm_quality_score: float
    temporal_coherence: TemporalCoherenceMetrics
    transition_quality: TransitionQualityMetrics
    economic_differentiation: EconomicDifferentiationMetrics
    spatial_coherence: Dict[str, float]
    regime_stability: Dict[str, float]
    overall_interpretation: str
    detailed_report: Optional[DetailedValidationReport] = None

class HMMValidationFramework:
    """Framework for HMM-appropriate validation metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_temporal_coherence(self, regime_sequence: np.ndarray, 
                                   min_duration_threshold: int = 5) -> TemporalCoherenceMetrics:
        """
        Calculate temporal coherence metrics for regime sequence.
        
        Replaces Silhouette Score with metrics that measure:
        - How consistently regimes persist over time
        - Stability of regime durations
        - Ratio of meaningful vs noise regime changes
        
        Args:
            regime_sequence: Array of regime assignments over time
            min_duration_threshold: Minimum meaningful regime duration
            
        Returns:
            TemporalCoherenceMetrics: Comprehensive temporal coherence analysis
        """
        try:
            if len(regime_sequence) < 10:
                return TemporalCoherenceMetrics(
                    temporal_coherence=0.0,
                    avg_regime_duration=1.0,
                    duration_stability=0.0,
                    too_short_ratio=1.0,
                    regime_consistency=0.0,
                    interpretation="Insufficient data for temporal analysis"
                )
            
            # Calculate regime duration statistics
            regime_changes = np.diff(regime_sequence) != 0
            regime_durations = []
            current_duration = 1
            
            for change in regime_changes:
                if change:
                    regime_durations.append(current_duration)
                    current_duration = 1
                else:
                    current_duration += 1
            regime_durations.append(current_duration)
            
            regime_durations = np.array(regime_durations)
            
            # Basic duration statistics
            avg_duration = np.mean(regime_durations)
            duration_std = np.std(regime_durations)
            
            # Duration stability (inverse coefficient of variation)
            duration_stability = 1.0 / (1.0 + duration_std / (avg_duration + 1e-10))
            
            # Ratio of regimes that are too short (noise)
            too_short_ratio = np.sum(regime_durations < min_duration_threshold) / len(regime_durations)
            
            # Regime consistency (how often the same regime appears consecutively)
            regime_consistency = 1.0 - (np.sum(regime_changes) / len(regime_changes))
            
            # Temporal coherence score (higher = more stable, meaningful regimes)
            temporal_coherence = (
                duration_stability * 0.4 +
                (1.0 - too_short_ratio) * 0.3 +
                regime_consistency * 0.3
            )
            
            # Ensure score is bounded [0, 1]
            temporal_coherence = max(0.0, min(1.0, temporal_coherence))
            
            return TemporalCoherenceMetrics(
                temporal_coherence=temporal_coherence,
                avg_regime_duration=avg_duration,
                duration_stability=duration_stability,
                too_short_ratio=too_short_ratio,
                regime_consistency=regime_consistency,
                interpretation=f"Temporal coherence: {temporal_coherence:.3f} - {'Excellent' if temporal_coherence > 0.8 else 'Good' if temporal_coherence > 0.6 else 'Needs improvement'}"
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating temporal coherence: {e}")
            return TemporalCoherenceMetrics(
                temporal_coherence=0.0,
                avg_regime_duration=1.0,
                duration_stability=0.0,
                too_short_ratio=1.0,
                regime_consistency=0.0,
                interpretation="Error in temporal coherence calculation"
            )
    
    def calculate_transition_quality(self, transition_matrix: np.ndarray) -> TransitionQualityMetrics:
        """
        Calculate regime transition quality metrics.
        
        Replaces Davies-Bouldin Score with metrics that measure:
        - Clarity and predictability of regime transitions
        - Stability of regime persistence
        - Information content of transition patterns
        
        Args:
            transition_matrix: Regime transition probability matrix
            
        Returns:
            TransitionQualityMetrics: Comprehensive transition quality analysis
        """
        try:
            if transition_matrix.size == 0 or transition_matrix.shape[0] == 0:
                return TransitionQualityMetrics(
                    transition_quality=0.0,
                    avg_persistence=0.0,
                    transition_entropy=float('inf'),
                    transition_clarity=0.0,
                    persistence_consistency=0.0,
                    interpretation="No transition matrix available"
                )
            
            n_regimes = transition_matrix.shape[0]
            
            # Calculate transition entropy (lower = more predictable)
            transition_entropy = 0.0
            valid_rows = 0
            
            for i in range(n_regimes):
                row = transition_matrix[i]
                # Avoid log(0) by adding small epsilon
                row_safe = row + 1e-10
                row_entropy = -np.sum(row * np.log(row_safe))
                if not np.isnan(row_entropy) and not np.isinf(row_entropy):
                    transition_entropy += row_entropy
                    valid_rows += 1
            
            if valid_rows > 0:
                transition_entropy /= valid_rows
            
            # Calculate regime persistence (diagonal elements)
            persistence_scores = np.diag(transition_matrix)
            avg_persistence = np.mean(persistence_scores)
            persistence_std = np.std(persistence_scores)
            
            # Persistence consistency (higher = more consistent across regimes)
            persistence_consistency = 1.0 / (1.0 + persistence_std / (avg_persistence + 1e-10))
            
            # Transition clarity (how clear the dominant transition is for each regime)
            max_transitions = np.max(transition_matrix, axis=1)
            avg_max_transition = np.mean(max_transitions)
            transition_clarity = avg_max_transition
            
            # Calculate transition asymmetry (how different forward vs backward transitions are)
            transition_asymmetry = 0.0
            if n_regimes > 1:
                # Measure how different transition probabilities are from uniform
                uniform_prob = 1.0 / n_regimes
                asymmetry_scores = []
                for i in range(n_regimes):
                    row = transition_matrix[i]
                    # Measure deviation from uniform distribution
                    deviation = np.mean(np.abs(row - uniform_prob))
                    asymmetry_scores.append(deviation)
                transition_asymmetry = np.mean(asymmetry_scores)
            
            # Overall transition quality score
            # Higher entropy is bad (more random), so we invert it
            entropy_score = max(0.0, 1.0 - min(transition_entropy / np.log(n_regimes), 1.0))
            
            transition_quality = (
                persistence_consistency * 0.3 +
                transition_clarity * 0.25 +
                entropy_score * 0.25 +
                transition_asymmetry * 0.2
            )
            
            # Ensure score is bounded [0, 1]
            transition_quality = max(0.0, min(1.0, transition_quality))
            
            return TransitionQualityMetrics(
                transition_quality=transition_quality,
                avg_persistence=avg_persistence,
                transition_entropy=transition_entropy,
                transition_clarity=transition_clarity,
                persistence_consistency=persistence_consistency,
                interpretation=f"Transition quality: {transition_quality:.3f} - {'Excellent' if transition_quality > 0.8 else 'Good' if transition_quality > 0.6 else 'Needs improvement'}"
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating transition quality: {e}")
            return TransitionQualityMetrics(
                transition_quality=0.0,
                avg_persistence=0.0,
                transition_entropy=float('inf'),
                transition_clarity=0.0,
                persistence_consistency=0.0,
                interpretation="Error in transition quality calculation"
            )
    
    def calculate_economic_differentiation(self, regime_data: pd.DataFrame, 
                                         returns_col: str = 'returns',
                                         price_col: str = 'close',
                                         volume_col: str = 'volume') -> EconomicDifferentiationMetrics:
        """
        Calculate comprehensive economic differentiation metrics.
        
        Replaces Calinski-Harabasz Score with thorough economic analysis:
        - Return differentiation across regimes
        - Risk-return tradeoff analysis
        - Market efficiency impact measurement
        - Regime economic distinctness
        
        Args:
            regime_data: DataFrame with regime assignments and market data
            returns_col: Column name for returns
            price_col: Column name for price data
            volume_col: Column name for volume data
            
        Returns:
            EconomicDifferentiationMetrics: Comprehensive economic analysis
        """
        try:
            if 'regime' not in regime_data.columns:
                raise ValueError("Regime column not found in data")
            
            unique_regimes = regime_data['regime'].unique()
            if len(unique_regimes) < 2:
                return EconomicDifferentiationMetrics(
                    economic_differentiation=0.0,
                    return_differentiation=0.0,
                    volatility_differentiation=0.0,
                    sharpe_differentiation=0.0,
                    risk_return_tradeoff=0.0,
                    regime_economic_distinctness=0.0,
                    market_efficiency_impact=0.0,
                    regime_stats={},
                    interpretation="Insufficient regimes for economic differentiation"
                )
            
            regime_stats = {}
            returns_by_regime = []
            volatilities_by_regime = []
            sharpes_by_regime = []
            volumes_by_regime = []
            
            # Calculate comprehensive statistics for each regime
            for regime in unique_regimes:
                regime_mask = regime_data['regime'] == regime
                regime_subset = regime_data[regime_mask]
                
                if len(regime_subset) < 10:  # Need minimum samples
                    continue
                
                # Calculate returns if not provided
                if returns_col not in regime_subset.columns and price_col in regime_subset.columns:
                    prices = regime_subset[price_col].values
                    regime_returns = np.diff(prices) / prices[:-1]
                elif returns_col in regime_subset.columns:
                    regime_returns = regime_subset[returns_col].values
                else:
                    continue
                
                # Remove any NaN or infinite values
                regime_returns = regime_returns[np.isfinite(regime_returns)]
                
                if len(regime_returns) == 0:
                    continue
                
                # Calculate regime statistics
                mean_return = np.mean(regime_returns)
                volatility = np.std(regime_returns)
                sharpe = mean_return / (volatility + 1e-10)
                skewness = stats.skew(regime_returns)
                kurtosis = stats.kurtosis(regime_returns)
                
                # Volume statistics
                if volume_col in regime_subset.columns:
                    volumes = regime_subset[volume_col].values
                    volumes = volumes[np.isfinite(volumes)]
                    avg_volume = np.mean(volumes) if len(volumes) > 0 else 0.0
                    volume_volatility = np.std(volumes) if len(volumes) > 0 else 0.0
                else:
                    avg_volume = 0.0
                    volume_volatility = 0.0
                
                regime_stats[regime] = {
                    'mean_return': mean_return,
                    'volatility': volatility,
                    'sharpe': sharpe,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'avg_volume': avg_volume,
                    'volume_volatility': volume_volatility,
                    'sample_size': len(regime_returns)
                }
                
                returns_by_regime.append(mean_return)
                volatilities_by_regime.append(volatility)
                sharpes_by_regime.append(sharpe)
                volumes_by_regime.append(avg_volume)
            
            if len(returns_by_regime) < 2:
                return EconomicDifferentiationMetrics(
                    economic_differentiation=0.0,
                    return_differentiation=0.0,
                    volatility_differentiation=0.0,
                    sharpe_differentiation=0.0,
                    risk_return_tradeoff=0.0,
                    regime_economic_distinctness=0.0,
                    market_efficiency_impact=0.0,
                    regime_stats=regime_stats,
                    interpretation="Insufficient regimes with valid data"
                )
            
            returns_by_regime = np.array(returns_by_regime)
            volatilities_by_regime = np.array(volatilities_by_regime)
            sharpes_by_regime = np.array(sharpes_by_regime)
            volumes_by_regime = np.array(volumes_by_regime)
            
            # 1. Return Differentiation (variance in returns across regimes)
            return_variance = np.var(returns_by_regime)
            return_mean_abs = np.mean(np.abs(returns_by_regime))
            return_differentiation = return_variance / (return_mean_abs + 1e-10)
            
            # 2. Volatility Differentiation (variance in volatility across regimes)
            volatility_variance = np.var(volatilities_by_regime)
            volatility_mean = np.mean(volatilities_by_regime)
            volatility_differentiation = volatility_variance / (volatility_mean + 1e-10)
            
            # 3. Sharpe Ratio Differentiation (variance in risk-adjusted returns)
            sharpe_variance = np.var(sharpes_by_regime)
            sharpe_mean_abs = np.mean(np.abs(sharpes_by_regime))
            sharpe_differentiation = sharpe_variance / (sharpe_mean_abs + 1e-10)
            
            # 4. Risk-Return Tradeoff Analysis
            # Calculate correlation between risk and return across regimes
            if len(returns_by_regime) > 2:
                risk_return_corr = np.corrcoef(returns_by_regime, volatilities_by_regime)[0, 1]
                if np.isnan(risk_return_corr):
                    risk_return_corr = 0.0
                # Higher correlation = better risk-return tradeoff
                risk_return_tradeoff = abs(risk_return_corr)
            else:
                risk_return_tradeoff = 0.0
            
            # 5. Regime Economic Distinctness
            # Measure how economically distinct regimes are using multiple dimensions
            economic_dimensions = np.column_stack([
                returns_by_regime,
                volatilities_by_regime,
                sharpes_by_regime,
                volumes_by_regime
            ])
            
            # Calculate pairwise distances between regimes in economic space
            if economic_dimensions.shape[0] > 1:
                distances = pdist(economic_dimensions, metric='euclidean')
                avg_economic_distance = np.mean(distances)
                economic_distance_std = np.std(distances)
                
                # Normalize by the scale of the data
                data_scale = np.std(economic_dimensions.flatten())
                regime_economic_distinctness = avg_economic_distance / (data_scale + 1e-10)
            else:
                regime_economic_distinctness = 0.0
            
            # 6. Market Efficiency Impact
            # Measure how much regime changes affect market efficiency metrics
            # Higher differentiation = more impact on market structure
            efficiency_metrics = []
            
            for regime, regime_stat in regime_stats.items():
                # Calculate regime-specific efficiency indicators
                sharpe_ratio = regime_stat['sharpe']
                volatility = regime_stat['volatility']
                skewness = abs(regime_stat['skewness'])  # Higher absolute skewness = less efficient
                
                # Combined efficiency score (higher = more efficient)
                efficiency_score = sharpe_ratio / (1.0 + volatility + skewness)
                efficiency_metrics.append(efficiency_score)
            
            if len(efficiency_metrics) > 1:
                efficiency_variance = np.var(efficiency_metrics)
                efficiency_mean = np.mean(efficiency_metrics)
                market_efficiency_impact = efficiency_variance / (efficiency_mean + 1e-10)
            else:
                market_efficiency_impact = 0.0
            
            # Overall economic differentiation score
            economic_differentiation = (
                min(return_differentiation, 1.0) * 0.25 +
                min(volatility_differentiation, 1.0) * 0.20 +
                min(sharpe_differentiation, 1.0) * 0.20 +
                risk_return_tradeoff * 0.15 +
                min(regime_economic_distinctness, 1.0) * 0.15 +
                min(market_efficiency_impact, 1.0) * 0.05
            )
            
            # Ensure score is bounded [0, 1]
            economic_differentiation = max(0.0, min(1.0, economic_differentiation))
            
            return EconomicDifferentiationMetrics(
                economic_differentiation=economic_differentiation,
                return_differentiation=return_differentiation,
                volatility_differentiation=volatility_differentiation,
                sharpe_differentiation=sharpe_differentiation,
                risk_return_tradeoff=risk_return_tradeoff,
                regime_economic_distinctness=regime_economic_distinctness,
                market_efficiency_impact=market_efficiency_impact,
                regime_stats=regime_stats,
                interpretation=f"Economic differentiation: {economic_differentiation:.3f} - {'Excellent' if economic_differentiation > 0.7 else 'Good' if economic_differentiation > 0.5 else 'Needs improvement'}"
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating economic differentiation: {e}")
            return EconomicDifferentiationMetrics(
                economic_differentiation=0.0,
                return_differentiation=0.0,
                volatility_differentiation=0.0,
                sharpe_differentiation=0.0,
                risk_return_tradeoff=0.0,
                regime_economic_distinctness=0.0,
                market_efficiency_impact=0.0,
                regime_stats={},
                interpretation="Error in economic differentiation calculation"
            )
    
    def calculate_regime_stability(self, regime_sequence: np.ndarray, 
                                 window_size: int = 100) -> Dict[str, float]:
        """
        Calculate regime stability metrics over time.
        
        Args:
            regime_sequence: Array of regime assignments over time
            window_size: Size of rolling window for stability analysis
            
        Returns:
            Dict containing stability metrics
        """
        try:
            if len(regime_sequence) < window_size:
                return {
                    'regime_stability_index': 0.0,
                    'stability_consistency': 0.0,
                    'regime_volatility': 1.0,
                    'interpretation': "Insufficient data for stability analysis"
                }
            
            n_windows = len(regime_sequence) // window_size
            regime_consistency = []
            
            for i in range(n_windows):
                window_start = i * window_size
                window_end = window_start + window_size
                window_regimes = regime_sequence[window_start:window_end]
                
                # Calculate regime consistency within window
                unique_regimes, counts = np.unique(window_regimes, return_counts=True)
                dominant_regime_ratio = np.max(counts) / len(window_regimes)
                regime_consistency.append(dominant_regime_ratio)
            
            if len(regime_consistency) > 0:
                stability_index = np.mean(regime_consistency)
                stability_consistency = 1.0 - np.std(regime_consistency)
                regime_volatility = 1.0 - stability_index
            else:
                stability_index = 0.0
                stability_consistency = 0.0
                regime_volatility = 1.0
            
            return {
                'regime_stability_index': stability_index,
                'stability_consistency': stability_consistency,
                'regime_volatility': regime_volatility,
                'interpretation': f"Regime stability: {stability_index:.3f} - {'Excellent' if stability_index > 0.8 else 'Good' if stability_index > 0.6 else 'Needs improvement'}"
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating regime stability: {e}")
            return {
                'regime_stability_index': 0.0,
                'stability_consistency': 0.0,
                'regime_volatility': 1.0,
                'interpretation': "Error in stability calculation"
            }
    
    def calculate_spatial_coherence(self, regime_data: pd.DataFrame, 
                                  feature_columns: List[str]) -> Dict[str, float]:
        """
        Calculate spatial coherence metrics for regime clusters.
        
        This provides the spatial clustering validation that's still important
        for ensuring regimes have internal similarity.
        
        Args:
            regime_data: DataFrame with regime assignments and features
            feature_columns: List of feature column names to use for spatial analysis
            
        Returns:
            Dict containing spatial coherence metrics
        """
        try:
            if 'regime' not in regime_data.columns:
                return {
                    'spatial_coherence': 0.0,
                    'intra_regime_similarity': 0.0,
                    'inter_regime_separation': 0.0,
                    'interpretation': "No regime column found"
                }
            
            # Select available feature columns
            available_features = [col for col in feature_columns if col in regime_data.columns]
            if not available_features:
                return {
                    'spatial_coherence': 0.0,
                    'intra_regime_similarity': 0.0,
                    'inter_regime_separation': 0.0,
                    'interpretation': "No feature columns available for spatial analysis"
                }
            
            unique_regimes = regime_data['regime'].unique()
            if len(unique_regimes) < 2:
                return {
                    'spatial_coherence': 0.0,
                    'intra_regime_similarity': 0.0,
                    'inter_regime_separation': 0.0,
                    'interpretation': "Insufficient regimes for spatial analysis"
                }
            
            # Extract feature data
            feature_data = regime_data[available_features].values
            regime_assignments = regime_data['regime'].values
            
            # Remove any rows with NaN values
            valid_mask = np.all(np.isfinite(feature_data), axis=1)
            feature_data = feature_data[valid_mask]
            regime_assignments = regime_assignments[valid_mask]
            
            if len(feature_data) == 0:
                return {
                    'spatial_coherence': 0.0,
                    'intra_regime_similarity': 0.0,
                    'inter_regime_separation': 0.0,
                    'interpretation': "No valid feature data available"
                }
            
            # Calculate intra-regime similarity (cohesion within regimes)
            intra_similarities = []
            for regime in unique_regimes:
                regime_mask = regime_assignments == regime
                regime_features = feature_data[regime_mask]
                
                if len(regime_features) > 1:
                    # Calculate average pairwise distance within regime
                    regime_distances = pdist(regime_features, metric='euclidean')
                    avg_intra_distance = np.mean(regime_distances)
                    intra_similarities.append(avg_intra_distance)
            
            if intra_similarities:
                avg_intra_similarity = np.mean(intra_similarities)
                # Invert so higher values = more similar (better)
                intra_regime_similarity = 1.0 / (1.0 + avg_intra_similarity)
            else:
                intra_regime_similarity = 0.0
            
            # Calculate inter-regime separation (distance between regime centroids)
            regime_centroids = []
            for regime in unique_regimes:
                regime_mask = regime_assignments == regime
                regime_features = feature_data[regime_mask]
                
                if len(regime_features) > 0:
                    centroid = np.mean(regime_features, axis=0)
                    regime_centroids.append(centroid)
            
            if len(regime_centroids) > 1:
                centroid_distances = pdist(regime_centroids, metric='euclidean')
                avg_inter_distance = np.mean(centroid_distances)
                inter_regime_separation = avg_inter_distance / (np.std(feature_data.flatten()) + 1e-10)
            else:
                inter_regime_separation = 0.0
            
            # Overall spatial coherence
            spatial_coherence = (intra_regime_similarity * 0.6 + 
                               min(inter_regime_separation, 1.0) * 0.4)
            
            return {
                'spatial_coherence': spatial_coherence,
                'intra_regime_similarity': intra_regime_similarity,
                'inter_regime_separation': inter_regime_separation,
                'interpretation': f"Spatial coherence: {spatial_coherence:.3f} - {'Excellent' if spatial_coherence > 0.7 else 'Good' if spatial_coherence > 0.5 else 'Needs improvement'}"
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating spatial coherence: {e}")
            return {
                'spatial_coherence': 0.0,
                'intra_regime_similarity': 0.0,
                'inter_regime_separation': 0.0,
                'interpretation': "Error in spatial coherence calculation"
            }
    
    def generate_detailed_report(self, regime_data: pd.DataFrame,
                               original_data: pd.DataFrame,
                               temporal_metrics: TemporalCoherenceMetrics,
                               transition_metrics: TransitionQualityMetrics,
                               economic_metrics: EconomicDifferentiationMetrics,
                               spatial_metrics: Dict[str, float],
                               stability_metrics: Dict[str, float],
                               hmm_quality_score: float) -> DetailedValidationReport:
        """
        Generate comprehensive detailed validation report.
        
        Args:
            regime_data: DataFrame with regime assignments
            original_data: Original market data
            temporal_metrics: Temporal coherence metrics
            transition_metrics: Transition quality metrics
            economic_metrics: Economic differentiation metrics
            spatial_metrics: Spatial coherence metrics
            stability_metrics: Regime stability metrics
            hmm_quality_score: Overall HMM quality score
            
        Returns:
            DetailedValidationReport: Comprehensive validation report
        """
        try:
            regime_sequence = regime_data['regime'].values
            unique_regimes = np.unique(regime_sequence)
            n_regimes = len(unique_regimes)
            n_samples = len(regime_sequence)
            
            # 1. Execution Summary
            execution_summary = {
                'validation_timestamp': datetime.now().isoformat(),
                'data_overview': {
                    'total_samples': n_samples,
                    'regime_count': n_regimes,
                    'unique_regimes': unique_regimes.tolist(),
                    'data_columns': list(original_data.columns),
                    'feature_columns': list(original_data.select_dtypes(include=[np.number]).columns)
                },
                'validation_method': 'HMM-appropriate metrics framework',
                'overall_quality_score': hmm_quality_score,
                'quality_grade': self._get_quality_grade(hmm_quality_score),
                'validation_status': 'PASSED' if hmm_quality_score > 0.6 else 'NEEDS_IMPROVEMENT'
            }
            
            # 2. Temporal Analysis
            temporal_analysis = {
                'temporal_coherence_score': temporal_metrics.temporal_coherence,
                'temporal_grade': self._get_quality_grade(temporal_metrics.temporal_coherence),
                'regime_duration_analysis': {
                    'average_duration': temporal_metrics.avg_regime_duration,
                    'duration_stability': temporal_metrics.duration_stability,
                    'duration_consistency': 'Excellent' if temporal_metrics.duration_stability > 0.8 else 'Good' if temporal_metrics.duration_stability > 0.6 else 'Needs improvement'
                },
                'regime_consistency_analysis': {
                    'consistency_score': temporal_metrics.regime_consistency,
                    'consistency_grade': self._get_quality_grade(temporal_metrics.regime_consistency),
                    'noise_ratio': temporal_metrics.too_short_ratio,
                    'noise_assessment': 'Low noise' if temporal_metrics.too_short_ratio < 0.2 else 'Moderate noise' if temporal_metrics.too_short_ratio < 0.4 else 'High noise'
                },
                'temporal_interpretation': temporal_metrics.interpretation,
                'temporal_recommendations': self._get_temporal_recommendations(temporal_metrics)
            }
            
            # 3. Transition Analysis
            transition_analysis = {
                'transition_quality_score': transition_metrics.transition_quality,
                'transition_grade': self._get_quality_grade(transition_metrics.transition_quality),
                'persistence_analysis': {
                    'average_persistence': transition_metrics.avg_persistence,
                    'persistence_consistency': transition_metrics.persistence_consistency,
                    'persistence_assessment': 'Excellent' if transition_metrics.avg_persistence > 0.8 else 'Good' if transition_metrics.avg_persistence > 0.6 else 'Needs improvement'
                },
                'transition_clarity_analysis': {
                    'transition_clarity': transition_metrics.transition_clarity,
                    'transition_entropy': transition_metrics.transition_entropy,
                    'predictability': 'High' if transition_metrics.transition_entropy < 1.0 else 'Moderate' if transition_metrics.transition_entropy < 2.0 else 'Low'
                },
                'transition_interpretation': transition_metrics.interpretation,
                'transition_recommendations': self._get_transition_recommendations(transition_metrics)
            }
            
            # 4. Economic Analysis
            economic_analysis = {
                'economic_differentiation_score': economic_metrics.economic_differentiation,
                'economic_grade': self._get_quality_grade(economic_metrics.economic_differentiation),
                'return_analysis': {
                    'return_differentiation': economic_metrics.return_differentiation,
                    'regime_returns': {f'regime_{k}': v for k, v in economic_metrics.regime_stats.items()},
                    'return_assessment': 'Well differentiated' if economic_metrics.return_differentiation > 0.5 else 'Moderately differentiated' if economic_metrics.return_differentiation > 0.3 else 'Poorly differentiated'
                },
                'volatility_analysis': {
                    'volatility_differentiation': economic_metrics.volatility_differentiation,
                    'regime_volatilities': {f'regime_{k}': v['volatility'] for k, v in economic_metrics.regime_stats.items()},
                    'volatility_assessment': 'Well differentiated' if economic_metrics.volatility_differentiation > 0.5 else 'Moderately differentiated' if economic_metrics.volatility_differentiation > 0.3 else 'Poorly differentiated'
                },
                'risk_return_analysis': {
                    'risk_return_tradeoff': economic_metrics.risk_return_tradeoff,
                    'regime_sharpes': {f'regime_{k}': v['sharpe'] for k, v in economic_metrics.regime_stats.items()},
                    'tradeoff_assessment': 'Strong correlation' if economic_metrics.risk_return_tradeoff > 0.7 else 'Moderate correlation' if economic_metrics.risk_return_tradeoff > 0.4 else 'Weak correlation'
                },
                'economic_distinctness': {
                    'regime_economic_distinctness': economic_metrics.regime_economic_distinctness,
                    'market_efficiency_impact': economic_metrics.market_efficiency_impact,
                    'distinctness_assessment': 'Highly distinct' if economic_metrics.regime_economic_distinctness > 0.7 else 'Moderately distinct' if economic_metrics.regime_economic_distinctness > 0.4 else 'Poorly distinct'
                },
                'economic_interpretation': economic_metrics.interpretation,
                'economic_recommendations': self._get_economic_recommendations(economic_metrics)
            }
            
            # 5. Spatial Analysis
            spatial_analysis = {
                'spatial_coherence_score': spatial_metrics.get('spatial_coherence', 0.0),
                'spatial_grade': self._get_quality_grade(spatial_metrics.get('spatial_coherence', 0.0)),
                'cluster_cohesion': {
                    'intra_regime_similarity': spatial_metrics.get('intra_regime_similarity', 0.0),
                    'cohesion_assessment': 'High cohesion' if spatial_metrics.get('intra_regime_similarity', 0.0) > 0.7 else 'Moderate cohesion' if spatial_metrics.get('intra_regime_similarity', 0.0) > 0.4 else 'Low cohesion'
                },
                'cluster_separation': {
                    'inter_regime_separation': spatial_metrics.get('inter_regime_separation', 0.0),
                    'separation_assessment': 'Well separated' if spatial_metrics.get('inter_regime_separation', 0.0) > 0.7 else 'Moderately separated' if spatial_metrics.get('inter_regime_separation', 0.0) > 0.4 else 'Poorly separated'
                },
                'spatial_interpretation': spatial_metrics.get('interpretation', 'Spatial analysis completed'),
                'spatial_recommendations': self._get_spatial_recommendations(spatial_metrics)
            }
            
            # 6. Regime Characteristics
            regime_characteristics = {
                'regime_distribution': self._analyze_regime_distribution(regime_sequence),
                'regime_stability': {
                    'stability_index': stability_metrics.get('regime_stability_index', 0.0),
                    'stability_consistency': stability_metrics.get('stability_consistency', 0.0),
                    'regime_volatility': stability_metrics.get('regime_volatility', 0.0),
                    'stability_assessment': 'Highly stable' if stability_metrics.get('regime_stability_index', 0.0) > 0.8 else 'Moderately stable' if stability_metrics.get('regime_stability_index', 0.0) > 0.6 else 'Unstable'
                },
                'regime_duration_distribution': self._analyze_duration_distribution(regime_sequence),
                'regime_transition_patterns': self._analyze_transition_patterns(regime_sequence)
            }
            
            # 7. Comparative Analysis
            comparative_analysis = {
                'traditional_vs_hmm_metrics': self._generate_comparative_analysis(regime_data, original_data),
                'regime_overlap_analysis': self._analyze_regime_overlap(regime_sequence, original_data),
                'market_behavior_consistency': self._analyze_market_behavior_consistency(regime_sequence, original_data)
            }
            
            # 8. Recommendations
            recommendations = {
                'immediate_actions': self._get_immediate_recommendations(hmm_quality_score, temporal_metrics, transition_metrics, economic_metrics),
                'improvement_suggestions': self._get_improvement_suggestions(temporal_metrics, transition_metrics, economic_metrics, spatial_metrics),
                'parameter_tuning': self._get_parameter_tuning_recommendations(hmm_quality_score),
                'feature_engineering': self._get_feature_engineering_recommendations(economic_metrics, spatial_metrics)
            }
            
            # 9. Quality Assessment
            quality_assessment = {
                'overall_grade': self._get_quality_grade(hmm_quality_score),
                'component_grades': {
                    'temporal_coherence': self._get_quality_grade(temporal_metrics.temporal_coherence),
                    'transition_quality': self._get_quality_grade(transition_metrics.transition_quality),
                    'economic_differentiation': self._get_quality_grade(economic_metrics.economic_differentiation),
                    'spatial_coherence': self._get_quality_grade(spatial_metrics.get('spatial_coherence', 0.0)),
                    'regime_stability': self._get_quality_grade(stability_metrics.get('regime_stability_index', 0.0))
                },
                'strengths': self._identify_strengths(temporal_metrics, transition_metrics, economic_metrics, spatial_metrics),
                'weaknesses': self._identify_weaknesses(temporal_metrics, transition_metrics, economic_metrics, spatial_metrics),
                'production_readiness': self._assess_production_readiness(hmm_quality_score, temporal_metrics, transition_metrics, economic_metrics),
                'ml_training_suitability': self._assess_ml_training_suitability(hmm_quality_score, economic_metrics)
            }
            
            return DetailedValidationReport(
                execution_summary=execution_summary,
                temporal_analysis=temporal_analysis,
                transition_analysis=transition_analysis,
                economic_analysis=economic_analysis,
                spatial_analysis=spatial_analysis,
                regime_characteristics=regime_characteristics,
                comparative_analysis=comparative_analysis,
                recommendations=recommendations,
                quality_assessment=quality_assessment
            )
            
        except Exception as e:
            self.logger.error(f"Error generating detailed report: {e}")
            return DetailedValidationReport(
                execution_summary={'error': str(e)},
                temporal_analysis={},
                transition_analysis={},
                economic_analysis={},
                spatial_analysis={},
                regime_characteristics={},
                comparative_analysis={},
                recommendations={},
                quality_assessment={}
            )
    
    def validate_hmm_regimes(self, regime_data: pd.DataFrame, 
                           original_data: pd.DataFrame,
                           feature_columns: Optional[List[str]] = None,
                           generate_detailed_report: bool = True) -> HMMValidationMetrics:
        """
        Comprehensive HMM regime validation using appropriate metrics.
        
        Args:
            regime_data: DataFrame with regime assignments
            original_data: Original market data
            feature_columns: List of feature columns for spatial analysis
            
        Returns:
            HMMValidationMetrics: Comprehensive validation results
        """
        try:
            if 'regime' not in regime_data.columns:
                raise ValueError("Regime column not found in regime_data")
            
            regime_sequence = regime_data['regime'].values
            
            # 1. Calculate temporal coherence
            temporal_metrics = self.calculate_temporal_coherence(regime_sequence)
            
            # 2. Calculate transition quality
            transition_matrix = self._calculate_transition_matrix(regime_sequence)
            transition_metrics = self.calculate_transition_quality(transition_matrix)
            
            # 3. Calculate economic differentiation
            combined_data = original_data.copy()
            combined_data['regime'] = regime_sequence
            economic_metrics = self.calculate_economic_differentiation(combined_data)
            
            # 4. Calculate regime stability
            stability_metrics = self.calculate_regime_stability(regime_sequence)
            
            # 5. Calculate spatial coherence (if features available)
            if feature_columns:
                spatial_metrics = self.calculate_spatial_coherence(regime_data, feature_columns)
            else:
                # Use available numeric columns as features
                numeric_cols = regime_data.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [col for col in numeric_cols if col != 'regime']
                spatial_metrics = self.calculate_spatial_coherence(regime_data, numeric_cols)
            
            # Calculate overall HMM quality score
            hmm_quality_score = (
                temporal_metrics.temporal_coherence * 0.25 +
                transition_metrics.transition_quality * 0.25 +
                economic_metrics.economic_differentiation * 0.25 +
                spatial_metrics['spatial_coherence'] * 0.15 +
                stability_metrics['regime_stability_index'] * 0.10
            )
            
            # Generate overall interpretation
            if hmm_quality_score > 0.8:
                overall_interpretation = "Excellent HMM regime detection with strong temporal coherence, clear transitions, and good economic differentiation"
            elif hmm_quality_score > 0.6:
                overall_interpretation = "Good HMM regime detection with acceptable temporal coherence and economic differentiation"
            else:
                overall_interpretation = "HMM regime detection needs improvement in temporal coherence, transitions, or economic differentiation"
            
            # Generate detailed report if requested
            detailed_report = None
            if generate_detailed_report:
                detailed_report = self.generate_detailed_report(
                    regime_data, original_data, temporal_metrics, transition_metrics,
                    economic_metrics, spatial_metrics, stability_metrics, hmm_quality_score
                )
            
            return HMMValidationMetrics(
                hmm_quality_score=hmm_quality_score,
                temporal_coherence=temporal_metrics,
                transition_quality=transition_metrics,
                economic_differentiation=economic_metrics,
                spatial_coherence=spatial_metrics,
                regime_stability=stability_metrics,
                overall_interpretation=overall_interpretation,
                detailed_report=detailed_report
            )
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive HMM validation: {e}")
            # Return minimal valid metrics
            return HMMValidationMetrics(
                hmm_quality_score=0.0,
                temporal_coherence=TemporalCoherenceMetrics(0.0, 1.0, 0.0, 1.0, 0.0, "Error"),
                transition_quality=TransitionQualityMetrics(0.0, 0.0, float('inf'), 0.0, 0.0, "Error"),
                economic_differentiation=EconomicDifferentiationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, "Error"),
                spatial_coherence={'spatial_coherence': 0.0, 'interpretation': "Error"},
                regime_stability={'regime_stability_index': 0.0, 'interpretation': "Error"},
                overall_interpretation="Error in validation calculation"
            )
    
    def _calculate_transition_matrix(self, regime_sequence: np.ndarray) -> np.ndarray:
        """Calculate transition probability matrix from regime sequence."""
        try:
            unique_regimes = np.unique(regime_sequence)
            n_regimes = len(unique_regimes)
            
            if n_regimes == 0:
                return np.array([])
            
            # Create regime to index mapping
            regime_to_index = {regime: i for i, regime in enumerate(unique_regimes)}
            
            # Initialize transition count matrix
            transition_counts = np.zeros((n_regimes, n_regimes))
            
            # Count transitions
            for i in range(len(regime_sequence) - 1):
                current_regime = regime_sequence[i]
                next_regime = regime_sequence[i + 1]
                
                current_idx = regime_to_index[current_regime]
                next_idx = regime_to_index[next_regime]
                
                transition_counts[current_idx, next_idx] += 1
            
            # Convert counts to probabilities
            row_sums = transition_counts.sum(axis=1)
            # Avoid division by zero
            row_sums[row_sums == 0] = 1
            transition_matrix = transition_counts / row_sums[:, np.newaxis]
            
            return transition_matrix
            
        except Exception as e:
            self.logger.error(f"Error calculating transition matrix: {e}")
            return np.array([])
    
    # Helper methods for detailed reporting
    def _get_quality_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 0.9:
            return 'A+'
        elif score >= 0.8:
            return 'A'
        elif score >= 0.7:
            return 'B+'
        elif score >= 0.6:
            return 'B'
        elif score >= 0.5:
            return 'C'
        elif score >= 0.4:
            return 'D'
        else:
            return 'F'
    
    def _get_temporal_recommendations(self, temporal_metrics: TemporalCoherenceMetrics) -> List[str]:
        """Generate recommendations for temporal coherence improvements."""
        recommendations = []
        
        if temporal_metrics.temporal_coherence < 0.6:
            recommendations.append("Consider increasing minimum regime duration threshold")
            recommendations.append("Review regime detection parameters for noise reduction")
        
        if temporal_metrics.too_short_ratio > 0.3:
            recommendations.append("Implement regime smoothing to reduce noise transitions")
            recommendations.append("Consider post-processing to merge short regimes")
        
        if temporal_metrics.duration_stability < 0.6:
            recommendations.append("Investigate regime duration variability sources")
            recommendations.append("Consider adaptive regime detection parameters")
        
        return recommendations
    
    def _get_transition_recommendations(self, transition_metrics: TransitionQualityMetrics) -> List[str]:
        """Generate recommendations for transition quality improvements."""
        recommendations = []
        
        if transition_metrics.transition_quality < 0.6:
            recommendations.append("Review transition probability estimation method")
            recommendations.append("Consider ensemble approaches for transition modeling")
        
        if transition_metrics.transition_entropy > 2.0:
            recommendations.append("Implement transition smoothing techniques")
            recommendations.append("Consider hierarchical regime modeling")
        
        if transition_metrics.avg_persistence < 0.6:
            recommendations.append("Investigate regime stability factors")
            recommendations.append("Consider regime persistence constraints")
        
        return recommendations
    
    def _get_economic_recommendations(self, economic_metrics: EconomicDifferentiationMetrics) -> List[str]:
        """Generate recommendations for economic differentiation improvements."""
        recommendations = []
        
        if economic_metrics.economic_differentiation < 0.5:
            recommendations.append("Enhance feature engineering for economic differentiation")
            recommendations.append("Consider regime-specific feature selection")
        
        if economic_metrics.return_differentiation < 0.3:
            recommendations.append("Add regime-aware return features")
            recommendations.append("Consider multi-timeframe return analysis")
        
        if economic_metrics.volatility_differentiation < 0.3:
            recommendations.append("Implement regime-specific volatility modeling")
            recommendations.append("Add volatility regime indicators")
        
        return recommendations
    
    def _get_spatial_recommendations(self, spatial_metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations for spatial coherence improvements."""
        recommendations = []
        
        if spatial_metrics.get('spatial_coherence', 0.0) < 0.5:
            recommendations.append("Review feature scaling and normalization")
            recommendations.append("Consider feature selection for better separation")
        
        if spatial_metrics.get('intra_regime_similarity', 0.0) < 0.4:
            recommendations.append("Investigate regime internal consistency")
            recommendations.append("Consider regime-specific feature engineering")
        
        return recommendations
    
    def _analyze_regime_distribution(self, regime_sequence: np.ndarray) -> Dict[str, Any]:
        """Analyze regime distribution characteristics."""
        unique_regimes, counts = np.unique(regime_sequence, return_counts=True)
        total_samples = len(regime_sequence)
        
        distribution = {}
        for regime, count in zip(unique_regimes, counts):
            percentage = (count / total_samples) * 100
            distribution[f'regime_{regime}'] = {
                'count': int(count),
                'percentage': round(percentage, 2),
                'assessment': 'Balanced' if 15 <= percentage <= 35 else 'Imbalanced' if percentage < 10 or percentage > 50 else 'Moderate'
            }
        
        # Calculate distribution balance
        percentages = [dist['percentage'] for dist in distribution.values()]
        distribution_std = np.std(percentages)
        balance_assessment = 'Well balanced' if distribution_std < 10 else 'Moderately balanced' if distribution_std < 20 else 'Poorly balanced'
        
        return {
            'regime_distribution': distribution,
            'distribution_balance': {
                'standard_deviation': round(distribution_std, 2),
                'assessment': balance_assessment
            },
            'total_regimes': len(unique_regimes),
            'total_samples': total_samples
        }
    
    def _analyze_duration_distribution(self, regime_sequence: np.ndarray) -> Dict[str, Any]:
        """Analyze regime duration distribution."""
        regime_changes = np.diff(regime_sequence) != 0
        regime_durations = []
        current_duration = 1
        
        for change in regime_changes:
            if change:
                regime_durations.append(current_duration)
                current_duration = 1
            else:
                current_duration += 1
        regime_durations.append(current_duration)
        
        regime_durations = np.array(regime_durations)
        
        return {
            'duration_statistics': {
                'mean': round(np.mean(regime_durations), 2),
                'median': round(np.median(regime_durations), 2),
                'std': round(np.std(regime_durations), 2),
                'min': int(np.min(regime_durations)),
                'max': int(np.max(regime_durations))
            },
            'duration_distribution': {
                'short_durations': int(np.sum(regime_durations < 5)),
                'medium_durations': int(np.sum((regime_durations >= 5) & (regime_durations < 20))),
                'long_durations': int(np.sum(regime_durations >= 20))
            }
        }
    
    def _analyze_transition_patterns(self, regime_sequence: np.ndarray) -> Dict[str, Any]:
        """Analyze regime transition patterns."""
        regime_changes = np.diff(regime_sequence) != 0
        transition_points = np.where(regime_changes)[0]
        
        if len(transition_points) > 0:
            transition_intervals = np.diff(transition_points)
            return {
                'transition_count': len(transition_points),
                'transition_frequency': len(transition_points) / len(regime_sequence),
                'average_transition_interval': round(np.mean(transition_intervals), 2) if len(transition_intervals) > 0 else 0,
                'transition_volatility': round(np.std(transition_intervals), 2) if len(transition_intervals) > 0 else 0
            }
        else:
            return {
                'transition_count': 0,
                'transition_frequency': 0.0,
                'average_transition_interval': 0,
                'transition_volatility': 0.0
            }
    
    def _generate_comparative_analysis(self, regime_data: pd.DataFrame, original_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comparative analysis between traditional and HMM metrics."""
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data for traditional metrics
            numeric_cols = original_data.select_dtypes(include=[np.number]).columns.tolist()
            feature_data = original_data[numeric_cols].values
            scaler = StandardScaler()
            feature_data_scaled = scaler.fit_transform(feature_data)
            
            regime_assignments = regime_data['regime'].values
            
            # Calculate traditional metrics
            traditional_metrics = {
                'silhouette_score': silhouette_score(feature_data_scaled, regime_assignments),
                'calinski_harabasz_score': calinski_harabasz_score(feature_data_scaled, regime_assignments),
                'davies_bouldin_score': davies_bouldin_score(feature_data_scaled, regime_assignments)
            }
            
            return {
                'traditional_metrics': traditional_metrics,
                'traditional_interpretation': 'These metrics assume spatial separation and are inappropriate for HMM regimes',
                'hmm_advantage': 'HMM metrics account for temporal dependencies and economic relevance',
                'recommendation': 'Use HMM-appropriate metrics for temporal regime validation'
            }
            
        except ImportError:
            return {
                'traditional_metrics': 'Not available - sklearn not installed',
                'interpretation': 'Traditional clustering metrics not calculated'
            }
        except Exception as e:
            return {
                'traditional_metrics': 'Error in calculation',
                'error': str(e)
            }
    
    def _analyze_regime_overlap(self, regime_sequence: np.ndarray, original_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze regime overlap characteristics."""
        unique_regimes = np.unique(regime_sequence)
        
        # Calculate regime statistics
        regime_stats = {}
        for regime in unique_regimes:
            regime_mask = regime_sequence == regime
            regime_data = original_data[regime_mask]
            
            if len(regime_data) > 0:
                numeric_cols = regime_data.select_dtypes(include=[np.number]).columns
                regime_stats[regime] = {
                    'sample_count': len(regime_data),
                    'mean_values': regime_data[numeric_cols].mean().to_dict(),
                    'std_values': regime_data[numeric_cols].std().to_dict()
                }
        
        return {
            'regime_statistics': regime_stats,
            'overlap_assessment': 'Natural regime overlap is expected in financial markets',
            'interpretation': 'Overlapping characteristics indicate realistic market behavior'
        }
    
    def _analyze_market_behavior_consistency(self, regime_sequence: np.ndarray, original_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market behavior consistency across regimes."""
        # This would analyze how market behavior patterns are consistent within regimes
        # and different between regimes
        return {
            'behavior_consistency': 'Analysis requires additional market behavior features',
            'recommendation': 'Implement market behavior pattern analysis'
        }
    
    def _get_immediate_recommendations(self, hmm_quality_score: float, temporal_metrics: TemporalCoherenceMetrics,
                                     transition_metrics: TransitionQualityMetrics, economic_metrics: EconomicDifferentiationMetrics) -> List[str]:
        """Get immediate action recommendations."""
        recommendations = []
        
        if hmm_quality_score < 0.6:
            recommendations.append("Priority: Improve overall HMM quality - focus on weakest component")
        
        if temporal_metrics.temporal_coherence < 0.6:
            recommendations.append("Action: Implement regime duration filtering to reduce noise")
        
        if transition_metrics.transition_quality < 0.6:
            recommendations.append("Action: Review transition probability estimation")
        
        if economic_metrics.economic_differentiation < 0.5:
            recommendations.append("Action: Enhance economic feature engineering")
        
        return recommendations
    
    def _get_improvement_suggestions(self, temporal_metrics: TemporalCoherenceMetrics, transition_metrics: TransitionQualityMetrics,
                                   economic_metrics: EconomicDifferentiationMetrics, spatial_metrics: Dict[str, float]) -> List[str]:
        """Get improvement suggestions."""
        suggestions = []
        
        suggestions.extend(self._get_temporal_recommendations(temporal_metrics))
        suggestions.extend(self._get_transition_recommendations(transition_metrics))
        suggestions.extend(self._get_economic_recommendations(economic_metrics))
        suggestions.extend(self._get_spatial_recommendations(spatial_metrics))
        
        return list(set(suggestions))  # Remove duplicates
    
    def _get_parameter_tuning_recommendations(self, hmm_quality_score: float) -> List[str]:
        """Get parameter tuning recommendations."""
        recommendations = []
        
        if hmm_quality_score < 0.6:
            recommendations.append("Consider adjusting HMM parameters: n_components, covariance_type")
            recommendations.append("Review optimization mode settings (light/blank/full)")
        
        if hmm_quality_score > 0.8:
            recommendations.append("Current parameters are well-tuned")
            recommendations.append("Consider fine-tuning for specific use case optimization")
        
        return recommendations
    
    def _get_feature_engineering_recommendations(self, economic_metrics: EconomicDifferentiationMetrics, spatial_metrics: Dict[str, float]) -> List[str]:
        """Get feature engineering recommendations."""
        recommendations = []
        
        if economic_metrics.economic_differentiation < 0.5:
            recommendations.append("Add regime-specific technical indicators")
            recommendations.append("Implement volatility regime features")
            recommendations.append("Consider cross-timeframe features")
        
        if spatial_metrics.get('spatial_coherence', 0.0) < 0.5:
            recommendations.append("Review feature scaling and normalization")
            recommendations.append("Consider feature selection algorithms")
        
        return recommendations
    
    def _identify_strengths(self, temporal_metrics: TemporalCoherenceMetrics, transition_metrics: TransitionQualityMetrics,
                          economic_metrics: EconomicDifferentiationMetrics, spatial_metrics: Dict[str, float]) -> List[str]:
        """Identify system strengths."""
        strengths = []
        
        if temporal_metrics.temporal_coherence > 0.8:
            strengths.append("Excellent temporal coherence - regimes are stable over time")
        
        if transition_metrics.transition_quality > 0.8:
            strengths.append("Excellent transition quality - clear regime transitions")
        
        if economic_metrics.economic_differentiation > 0.7:
            strengths.append("Strong economic differentiation - regimes are economically distinct")
        
        if spatial_metrics.get('spatial_coherence', 0.0) > 0.7:
            strengths.append("Good spatial coherence - regimes are internally consistent")
        
        return strengths
    
    def _identify_weaknesses(self, temporal_metrics: TemporalCoherenceMetrics, transition_metrics: TransitionQualityMetrics,
                           economic_metrics: EconomicDifferentiationMetrics, spatial_metrics: Dict[str, float]) -> List[str]:
        """Identify system weaknesses."""
        weaknesses = []
        
        if temporal_metrics.temporal_coherence < 0.6:
            weaknesses.append("Poor temporal coherence - regimes change too frequently")
        
        if transition_metrics.transition_quality < 0.6:
            weaknesses.append("Poor transition quality - regime transitions are unclear")
        
        if economic_metrics.economic_differentiation < 0.5:
            weaknesses.append("Weak economic differentiation - regimes are not economically distinct")
        
        if spatial_metrics.get('spatial_coherence', 0.0) < 0.5:
            weaknesses.append("Poor spatial coherence - regimes lack internal consistency")
        
        return weaknesses
    
    def _assess_production_readiness(self, hmm_quality_score: float, temporal_metrics: TemporalCoherenceMetrics,
                                   transition_metrics: TransitionQualityMetrics, economic_metrics: EconomicDifferentiationMetrics) -> Dict[str, Any]:
        """Assess production readiness."""
        if hmm_quality_score > 0.8:
            readiness = "Production Ready"
            confidence = "High"
        elif hmm_quality_score > 0.6:
            readiness = "Near Production Ready"
            confidence = "Moderate"
        else:
            readiness = "Not Production Ready"
            confidence = "Low"
        
        return {
            'readiness_level': readiness,
            'confidence': confidence,
            'requirements_met': {
                'temporal_stability': temporal_metrics.temporal_coherence > 0.6,
                'transition_clarity': transition_metrics.transition_quality > 0.6,
                'economic_relevance': economic_metrics.economic_differentiation > 0.5
            }
        }
    
    def _assess_ml_training_suitability(self, hmm_quality_score: float, economic_metrics: EconomicDifferentiationMetrics) -> Dict[str, Any]:
        """Assess suitability for ML training."""
        if hmm_quality_score > 0.7 and economic_metrics.economic_differentiation > 0.6:
            suitability = "Excellent for ML Training"
            recommendation = "Proceed with confidence - regimes provide strong signal for ML models"
        elif hmm_quality_score > 0.6 and economic_metrics.economic_differentiation > 0.5:
            suitability = "Good for ML Training"
            recommendation = "Suitable for ML training with some caution - consider regime-specific models"
        else:
            suitability = "Limited ML Training Value"
            recommendation = "Improve regime quality before ML training - weak economic differentiation"
        
        return {
            'suitability_level': suitability,
            'recommendation': recommendation,
            'regime_count_adequate': len(economic_metrics.regime_stats) >= 2,
            'economic_signal_strength': economic_metrics.economic_differentiation
        }