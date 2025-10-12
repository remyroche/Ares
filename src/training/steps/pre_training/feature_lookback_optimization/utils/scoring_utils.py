"""
Consolidated Scoring Utilities for Feature Lookback Optimization.

This module provides unified scoring methods to eliminate code duplication.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from ..error_handling.error_handler import safe_operation, safe_mi_calculation, safe_correlation_calculation


@dataclass
class ScoringConfig:
    """Configuration for scoring methods."""
    variance_penalty_cap: float = 0.3
    stability_penalty_factor: float = 0.1
    max_penalty_ratio: float = 0.5
    min_mi_threshold: float = 0.0
    correlation_to_mi_factor: float = 0.5


class UnifiedScoring:
    """Unified scoring methods to eliminate duplication."""
    
    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()
    
    @safe_operation("scale normalized scoring", default_value=0.0)
    def calculate_scale_normalized_score(
        self, 
        mean_mi: float, 
        std_mi: float, 
        stability_penalty: float, 
        lookback_penalty: float
    ) -> Dict[str, float]:
        """
        Calculate scale-normalized scoring with adaptive penalties.
        
        Args:
            mean_mi: Mean mutual information
            std_mi: Standard deviation of MI
            stability_penalty: Stability penalty (0 or 1)
            lookback_penalty: Lookback regularization penalty
            
        Returns:
            Dictionary with normalized score components
        """
        # Adaptive variance penalty: cap at configured percentage of mean_MI
        max_variance_penalty = mean_mi * self.config.variance_penalty_cap if mean_mi > 0 else 0.0
        variance_penalty = min(0.5 * std_mi, max_variance_penalty)
        
        # Scale-normalized stability penalty: proportional to mean_MI
        normalized_stability_penalty = (mean_mi * self.config.stability_penalty_factor) if stability_penalty > 0 else 0.0
        
        # Base objective (MI - variance penalty)
        base_objective = mean_mi - variance_penalty
        
        # Total penalties with cap to preserve MI signal
        total_penalties = normalized_stability_penalty + lookback_penalty
        max_penalty = abs(base_objective) * self.config.max_penalty_ratio if base_objective != 0 else 0.0
        capped_penalties = min(total_penalties, max_penalty)
        
        # Final normalized score - ensure non-negative
        final_score = max(0.0, base_objective - capped_penalties)
        
        return {
            'mean_mi': mean_mi,
            'std_mi': std_mi,
            'variance_penalty': variance_penalty,
            'normalized_stability_penalty': normalized_stability_penalty,
            'lookback_penalty': lookback_penalty,
            'base_objective': base_objective,
            'total_penalties': total_penalties,
            'capped_penalties': capped_penalties,
            'final_score': final_score
        }
    
    @safe_operation("mutual information calculation", default_value=0.0)
    def calculate_mutual_information(
        self, 
        feature_values: np.ndarray, 
        target_values: np.ndarray
    ) -> float:
        """
        Calculate mutual information with standardized error handling.
        
        Args:
            feature_values: Feature values array
            target_values: Target values array
            
        Returns:
            Mutual information score
        """
        return safe_mi_calculation(feature_values, target_values, default_value=0.0)
    
    @safe_operation("correlation calculation", default_value=0.0)
    def calculate_correlation(
        self, 
        x: np.ndarray, 
        y: np.ndarray
    ) -> float:
        """
        Calculate correlation with standardized error handling.
        
        Args:
            x: First array
            y: Second array
            
        Returns:
            Correlation coefficient
        """
        return safe_correlation_calculation(x, y, default_value=0.0)
    
    @safe_operation("correlation to MI conversion", default_value=0.0)
    def convert_correlation_to_mi(self, correlation: float) -> float:
        """
        Convert correlation to MI approximation for consistency.
        
        Args:
            correlation: Correlation coefficient
            
        Returns:
            MI approximation
        """
        if abs(correlation) < 0.999:  # Avoid log(0)
            try:
                mi_approx = self.config.correlation_to_mi_factor * np.log(1 - correlation**2) if correlation**2 < 1 else 0.0
                return max(0.0, -mi_approx)  # Ensure positive MI
            except (ValueError, OverflowError):
                return 0.0
        else:
            return 0.0
    
    @safe_operation("composite score calculation", default_value=0.0)
    def calculate_composite_score(
        self, 
        correlations: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate composite score using MI-consistent metrics.
        
        Args:
            correlations: Dictionary of correlation values
            weights: Optional weights for each correlation
            
        Returns:
            Composite score
        """
        if not correlations:
            return 0.0
        
        # Default equal weights if not provided
        if weights is None:
            weights = {key: 1.0 for key in correlations.keys()}
        
        # Convert all correlations to MI-consistent scale
        mi_scores = []
        total_weight = 0.0
        
        for key, corr in correlations.items():
            weight = weights.get(key, 1.0)
            mi_score = self.convert_correlation_to_mi(corr)
            mi_scores.append(mi_score * weight)
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return sum(mi_scores) / total_weight
    
    @safe_operation("multi-objective score calculation", default_value=0.0)
    def calculate_multi_objective_score(
        self, 
        targets: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate multi-objective score using weighted combination.
        
        Args:
            targets: Dictionary of target values
            weights: Optional weights for each target
            
        Returns:
            Multi-objective score
        """
        if not targets:
            return 0.0
        
        # Default equal weights if not provided
        if weights is None:
            weights = {key: 1.0 for key in targets.keys()}
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for key, value in targets.items():
            weight = weights.get(key, 1.0)
            weighted_sum += value * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    @safe_operation("penalty calculation", default_value=0.0)
    def calculate_lookback_penalty(
        self, 
        lookback: int, 
        preferred_lookback: int,
        penalty_strength: float = 0.1,
        penalty_exponent: float = 2.0
    ) -> float:
        """
        Calculate lookback regularization penalty.
        
        Args:
            lookback: Current lookback value
            preferred_lookback: Preferred lookback value
            penalty_strength: Strength of penalty
            penalty_exponent: Exponent for penalty calculation
            
        Returns:
            Penalty value
        """
        if penalty_strength <= 0:
            return 0.0
        
        deviation = abs(lookback - preferred_lookback)
        return penalty_strength * (deviation ** penalty_exponent)
    
    @safe_operation("stability score calculation", default_value=0.0)
    def calculate_stability_score(
        self, 
        scores: np.ndarray,
        min_samples: int = 5
    ) -> float:
        """
        Calculate stability score from a list of scores.
        
        Args:
            scores: Array of scores
            min_samples: Minimum samples required
            
        Returns:
            Stability score (0-1, higher is more stable)
        """
        if len(scores) < min_samples:
            return 0.0
        
        # Calculate coefficient of variation
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if mean_score == 0:
            return 0.0
        
        cv = std_score / abs(mean_score)
        stability_score = max(0.0, 1.0 - cv)  # Convert to 0-1 scale
        
        return stability_score
    
    @safe_operation("bootstrap validation", default_value={'mean_mi': 0.0, 'std_mi': 0.0, 'objective': 0.0})
    def calculate_bootstrap_validation(
        self, 
        feature_values: np.ndarray, 
        target_values: np.ndarray, 
        n_resamples: int = 10
    ) -> Dict[str, float]:
        """
        Calculate bootstrap validation for mutual information.
        
        Args:
            feature_values: Feature values array
            target_values: Target values array
            n_resamples: Number of bootstrap resamples
            
        Returns:
            Dictionary with validation results
        """
        # Align arrays
        min_length = min(len(feature_values), len(target_values))
        if min_length < 20:  # Need sufficient data for bootstrap
            return {'mean_mi': 0.0, 'std_mi': 0.0, 'objective': 0.0}
        
        feature_aligned = feature_values[:min_length]
        target_aligned = target_values[:min_length]
        
        # Remove NaN values
        valid_mask = ~(np.isnan(feature_aligned) | np.isnan(target_aligned))
        if not np.any(valid_mask):
            return {'mean_mi': 0.0, 'std_mi': 0.0, 'objective': 0.0}
        
        feature_clean = feature_aligned[valid_mask]
        target_clean = target_aligned[valid_mask]
        
        if len(feature_clean) < 20:
            return {'mean_mi': 0.0, 'std_mi': 0.0, 'objective': 0.0}
        
        # Generate bootstrap samples
        np.random.seed(42)  # For reproducibility
        mi_samples = []
        
        for _ in range(n_resamples):
            # Bootstrap sample
            indices = np.random.choice(len(feature_clean), size=len(feature_clean), replace=True)
            bootstrap_feature = feature_clean[indices]
            bootstrap_target = target_clean[indices]
            
            # Calculate MI for this sample
            mi_score = self.calculate_mutual_information(bootstrap_feature, bootstrap_target)
            mi_samples.append(mi_score)
        
        # Calculate statistics
        mean_mi = float(np.mean(mi_samples))
        std_mi = float(np.std(mi_samples))
        
        # Use scale-normalized scoring
        scoring_result = self.calculate_scale_normalized_score(
            mean_mi=mean_mi,
            std_mi=std_mi,
            stability_penalty=0.0,  # No stability penalty in bootstrap
            lookback_penalty=0.0   # No lookback penalty in bootstrap
        )
        
        return {
            'mean_mi': mean_mi,
            'std_mi': std_mi,
            'objective': scoring_result['final_score']
        }


# Global scoring instance
_global_scoring: Optional[UnifiedScoring] = None


def get_scoring_utils(config: Optional[ScoringConfig] = None) -> UnifiedScoring:
    """Get the global scoring utilities instance."""
    global _global_scoring
    if _global_scoring is None:
        _global_scoring = UnifiedScoring(config)
    return _global_scoring