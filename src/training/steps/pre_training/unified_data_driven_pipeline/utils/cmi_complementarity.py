"""
CMI Complementarity Scorer

This module provides Conditional Mutual Information (CMI) complementarity scoring
for feature selection in the Tactician pipeline. It maximizes feature-target MI
while minimizing redundancy with Analyst outputs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from datetime import datetime
import warnings

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_success

logger = system_logger.getChild('CMIComplementarity')

@dataclass
class CMIComplementarityConfig:
    """Configuration for CMI complementarity scoring."""
    
    # Feature selection budget
    per_family_budget: Tuple[int, int] = (5, 15)  # Min/max features per family
    upstream_multiplier: int = 3  # Total budget to RFE = 3× per-family
    max_total_features: int = 60  # Maximum total features to select
    
    # CMI computation settings
    enable_regime_awareness: bool = True  # Compute R(X|A) per regime
    compute_timeout_seconds: float = 300.0  # 5 min hard limit
    
    # Synergy settings
    enable_synergy: bool = True  # Enable synergy computation
    beta_synergy: float = 0.25  # Synergy bonus weight
    
    # Estimator settings
    estimator_type: str = 'ksg'  # ksg, gcmi, binned
    ksg_k: int = 3  # KSG estimator k parameter
    gcmi_bins: int = 10  # GCMI estimator bins
    binned_bins: int = 20  # Binned estimator bins
    
    # Performance settings
    enable_caching: bool = True
    cache_size_mb: int = 100
    enable_parallel: bool = True
    n_jobs: int = -1

@dataclass
class CMIComplementarityResult:
    """Result of CMI complementarity scoring."""
    
    selected_features: List[str]
    feature_scores: Dict[str, float]
    complementarity_scores: Dict[str, float]
    synergy_scores: Dict[str, float]
    computation_time: float
    estimator_used: str
    regime_aware: bool
    n_features_selected: int
    n_features_evaluated: int
    warnings: List[str]

class CMIComplementarityScorer:
    """
    CMI Complementarity Scorer for Tactician feature selection.
    
    This scorer maximizes feature-target MI while minimizing redundancy
    with Analyst outputs through adaptive estimators and hardware optimizations.
    """
    
    def __init__(self, config: CMIComplementarityConfig):
        """Initialize the CMI complementarity scorer."""
        self.config = config
        self.logger = logger.getChild('CMIComplementarityScorer')
        
        # Initialize estimators
        self._initialize_estimators()
        
        # Initialize caching
        self._initialize_caching()
        
        # Performance tracking
        self.computation_history = []
        self.performance_stats = {
            'total_computations': 0,
            'avg_computation_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self.logger.info("✅ CMI Complementarity Scorer initialized")
    
    def _initialize_estimators(self):
        """Initialize CMI estimators."""
        try:
            # Try to import CMI estimators
            from .cmi_estimators import CMIEstimator, CMIEstimatorConfig
            
            # Create estimator config
            estimator_config = CMIEstimatorConfig(
                estimator_type=self.config.estimator_type,
                ksg_k=self.config.ksg_k,
                gcmi_bins=self.config.gcmi_bins,
                binned_bins=self.config.binned_bins,
                enable_parallel=self.config.enable_parallel,
                n_jobs=self.config.n_jobs
            )
            
            self.estimator = CMIEstimator(estimator_config)
            self.estimator_available = True
            
        except ImportError:
            self.estimator = None
            self.estimator_available = False
            self.logger.warning("⚠️ CMI estimators not available - using placeholder")
    
    def _initialize_caching(self):
        """Initialize caching system."""
        if self.config.enable_caching:
            try:
                from src.utils.caching import get_cmi_cache_manager
                self.cache_manager = get_cmi_cache_manager(
                    max_size_mb=self.config.cache_size_mb
                )
                self.caching_available = True
            except ImportError:
                self.cache_manager = None
                self.caching_available = False
                self.logger.warning("⚠️ Caching not available")
        else:
            self.cache_manager = None
            self.caching_available = False
    
    def score_features(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        analyst_outputs: Optional[pd.DataFrame] = None,
        regime_labels: Optional[pd.Series] = None,
        feature_families: Optional[Dict[str, List[str]]] = None
    ) -> CMIComplementarityResult:
        """
        Score features using CMI complementarity.
        
        Args:
            features: Feature matrix
            targets: Target values
            analyst_outputs: Analyst outputs for conditioning
            regime_labels: Regime labels for regime-aware computation
            feature_families: Feature families for budget allocation
            
        Returns:
            CMIComplementarityResult with selected features and scores
        """
        start_time = datetime.now()
        warnings_list = []
        
        self.logger.info(f"🔄 Starting CMI complementarity scoring for {len(features.columns)} features")
        
        try:
            # Validate inputs
            self._validate_inputs(features, targets, analyst_outputs, regime_labels)
            
            # Check if estimator is available
            if not self.estimator_available:
                warnings_list.append("CMI estimator not available - using placeholder scoring")
                return self._placeholder_scoring(features, targets, feature_families, warnings_list)
            
            # Compute CMI complementarity scores
            complementarity_scores = self._compute_complementarity_scores(
                features, targets, analyst_outputs, regime_labels
            )
            
            # Compute synergy scores if enabled
            synergy_scores = {}
            if self.config.enable_synergy:
                synergy_scores = self._compute_synergy_scores(
                    features, targets, analyst_outputs, regime_labels
                )
            
            # Select features based on budget
            selected_features, feature_scores = self._select_features_by_budget(
                complementarity_scores, synergy_scores, feature_families
            )
            
            # Compute final result
            computation_time = (datetime.now() - start_time).total_seconds()
            
            result = CMIComplementarityResult(
                selected_features=selected_features,
                feature_scores=feature_scores,
                complementarity_scores=complementarity_scores,
                synergy_scores=synergy_scores,
                computation_time=computation_time,
                estimator_used=self.config.estimator_type,
                regime_aware=self.config.enable_regime_awareness,
                n_features_selected=len(selected_features),
                n_features_evaluated=len(features.columns),
                warnings=warnings_list
            )
            
            # Update performance stats
            self._update_performance_stats(computation_time)
            
            self.logger.info(f"✅ CMI complementarity scoring completed: "
                           f"{len(selected_features)} features selected in {computation_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ CMI complementarity scoring failed: {e}")
            warnings_list.append(f"Scoring failed: {str(e)}")
            
            # Return placeholder result on failure
            return self._placeholder_scoring(features, targets, feature_families, warnings_list)
    
    def _validate_inputs(self, features: pd.DataFrame, targets: pd.Series,
                        analyst_outputs: Optional[pd.DataFrame],
                        regime_labels: Optional[pd.Series]):
        """Validate input data."""
        if len(features) != len(targets):
            raise ValueError("Features and targets must have the same length")
        
        if analyst_outputs is not None and len(features) != len(analyst_outputs):
            raise ValueError("Features and analyst outputs must have the same length")
        
        if regime_labels is not None and len(features) != len(regime_labels):
            raise ValueError("Features and regime labels must have the same length")
    
    def _compute_complementarity_scores(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        analyst_outputs: Optional[pd.DataFrame],
        regime_labels: Optional[pd.Series]
    ) -> Dict[str, float]:
        """Compute CMI complementarity scores."""
        complementarity_scores = {}
        
        for feature_name in features.columns:
            try:
                feature_values = features[feature_name].values
                
                # Compute CMI complementarity score
                if analyst_outputs is not None:
                    # I(X; Y | A) = I(X; Y) - I(X; A) - I(Y; A) + I(X, Y; A)
                    score = self._compute_cmi_complementarity(
                        feature_values, targets.values, analyst_outputs.values
                    )
                else:
                    # Fallback to regular MI
                    score = self._compute_mutual_information(feature_values, targets.values)
                
                complementarity_scores[feature_name] = score
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to compute score for {feature_name}: {e}")
                complementarity_scores[feature_name] = 0.0
        
        return complementarity_scores
    
    def _compute_synergy_scores(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        analyst_outputs: Optional[pd.DataFrame],
        regime_labels: Optional[pd.Series]
    ) -> Dict[str, float]:
        """Compute synergy scores between features."""
        synergy_scores = {}
        
        # Simple synergy computation - can be enhanced
        for feature_name in features.columns:
            try:
                feature_values = features[feature_name].values
                
                # Compute synergy with other features
                synergy_score = 0.0
                for other_feature in features.columns:
                    if other_feature != feature_name:
                        other_values = features[other_feature].values
                        synergy_score += self._compute_synergy(
                            feature_values, other_values, targets.values
                        )
                
                synergy_scores[feature_name] = synergy_score / (len(features.columns) - 1)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to compute synergy for {feature_name}: {e}")
                synergy_scores[feature_name] = 0.0
        
        return synergy_scores
    
    def _compute_cmi_complementarity(
        self, X: np.ndarray, Y: np.ndarray, A: np.ndarray
    ) -> float:
        """Compute CMI complementarity I(X; Y | A)."""
        try:
            if self.estimator_available:
                return self.estimator.compute_cmi(X, Y, A)
            else:
                # Placeholder implementation
                return self._placeholder_cmi(X, Y, A)
        except Exception as e:
            self.logger.warning(f"⚠️ CMI computation failed: {e}")
            return 0.0
    
    def _compute_mutual_information(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute mutual information I(X; Y)."""
        try:
            if self.estimator_available:
                return self.estimator.compute_mi(X, Y)
            else:
                # Placeholder implementation
                return self._placeholder_mi(X, Y)
        except Exception as e:
            self.logger.warning(f"⚠️ MI computation failed: {e}")
            return 0.0
    
    def _compute_synergy(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
        """Compute synergy between features."""
        try:
            # Simple synergy computation
            mi_xy = self._compute_mutual_information(X, Y)
            mi_xz = self._compute_mutual_information(X, Z)
            mi_yz = self._compute_mutual_information(Y, Z)
            
            # Synergy as interaction information
            synergy = mi_xy - mi_xz - mi_yz
            return max(0.0, synergy)  # Only positive synergy
            
        except Exception as e:
            self.logger.warning(f"⚠️ Synergy computation failed: {e}")
            return 0.0
    
    def _placeholder_cmi(self, X: np.ndarray, Y: np.ndarray, A: np.ndarray) -> float:
        """Placeholder CMI computation."""
        # Simple correlation-based approximation
        try:
            from scipy.stats import pearsonr
            corr_xy = abs(pearsonr(X, Y)[0])
            corr_xa = abs(pearsonr(X, A.flatten())[0])
            corr_ya = abs(pearsonr(Y, A.flatten())[0])
            
            # Approximate CMI as correlation difference
            cmi = corr_xy - 0.5 * (corr_xa + corr_ya)
            return max(0.0, cmi)
        except:
            return 0.0
    
    def _placeholder_mi(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Placeholder MI computation."""
        try:
            from scipy.stats import pearsonr
            corr = abs(pearsonr(X, Y)[0])
            return corr * 0.5  # Rough approximation
        except:
            return 0.0
    
    def _select_features_by_budget(
        self,
        complementarity_scores: Dict[str, float],
        synergy_scores: Dict[str, float],
        feature_families: Optional[Dict[str, List[str]]]
    ) -> Tuple[List[str], Dict[str, float]]:
        """Select features based on budget constraints."""
        
        if feature_families is None:
            # Simple selection by score
            sorted_features = sorted(
                complementarity_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Apply synergy bonus
            final_scores = {}
            for feature, score in sorted_features:
                synergy_bonus = synergy_scores.get(feature, 0.0) * self.config.beta_synergy
                final_scores[feature] = score + synergy_bonus
            
            # Select top features
            sorted_final = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [f[0] for f in sorted_final[:self.config.max_total_features]]
            
            return selected_features, final_scores
        
        else:
            # Family-based selection
            selected_features = []
            feature_scores = {}
            
            for family_name, family_features in feature_families.items():
                # Get scores for this family
                family_scores = {
                    f: complementarity_scores.get(f, 0.0) + 
                       synergy_scores.get(f, 0.0) * self.config.beta_synergy
                    for f in family_features if f in complementarity_scores
                }
                
                # Select features from this family
                sorted_family = sorted(family_scores.items(), key=lambda x: x[1], reverse=True)
                family_budget = min(
                    self.config.per_family_budget[1],
                    max(self.config.per_family_budget[0], len(sorted_family))
                )
                
                family_selected = [f[0] for f in sorted_family[:family_budget]]
                selected_features.extend(family_selected)
                feature_scores.update(family_scores)
            
            # Limit total features
            if len(selected_features) > self.config.max_total_features:
                # Re-sort and select top features
                sorted_all = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
                selected_features = [f[0] for f in sorted_all[:self.config.max_total_features]]
            
            return selected_features, feature_scores
    
    def _placeholder_scoring(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        feature_families: Optional[Dict[str, List[str]]],
        warnings_list: List[str]
    ) -> CMIComplementarityResult:
        """Placeholder scoring when CMI estimators are not available."""
        
        # Simple correlation-based scoring
        feature_scores = {}
        for feature_name in features.columns:
            try:
                from scipy.stats import pearsonr
                corr = abs(pearsonr(features[feature_name], targets)[0])
                feature_scores[feature_name] = corr
            except:
                feature_scores[feature_name] = 0.0
        
        # Select top features
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:self.config.max_total_features]]
        
        return CMIComplementarityResult(
            selected_features=selected_features,
            feature_scores=feature_scores,
            complementarity_scores=feature_scores,
            synergy_scores={},
            computation_time=0.0,
            estimator_used='placeholder',
            regime_aware=False,
            n_features_selected=len(selected_features),
            n_features_evaluated=len(features.columns),
            warnings=warnings_list
        )
    
    def _update_performance_stats(self, computation_time: float):
        """Update performance statistics."""
        self.performance_stats['total_computations'] += 1
        
        # Update average computation time
        total = self.performance_stats['total_computations']
        current_avg = self.performance_stats['avg_computation_time']
        self.performance_stats['avg_computation_time'] = (
            (current_avg * (total - 1) + computation_time) / total
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'total_computations': self.performance_stats['total_computations'],
            'avg_computation_time': self.performance_stats['avg_computation_time'],
            'cache_hit_rate': (
                self.performance_stats['cache_hits'] / 
                max(1, self.performance_stats['cache_hits'] + self.performance_stats['cache_misses'])
            ),
            'estimator_available': self.estimator_available,
            'caching_available': self.caching_available
        }

# Convenience functions
def create_cmi_complementarity_scorer(config: Optional[CMIComplementarityConfig] = None) -> CMIComplementarityScorer:
    """Create CMI complementarity scorer instance."""
    return CMIComplementarityScorer(config or CMIComplementarityConfig())

def score_features_with_cmi_complementarity(
    features: pd.DataFrame,
    targets: pd.Series,
    analyst_outputs: Optional[pd.DataFrame] = None,
    config: Optional[CMIComplementarityConfig] = None
) -> CMIComplementarityResult:
    """Score features using CMI complementarity."""
    scorer = create_cmi_complementarity_scorer(config)
    return scorer.score_features(features, targets, analyst_outputs)
