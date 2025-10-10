"""
Two-Stage Scoring System for Efficient MI/IC Computation

This module implements a two-stage scoring system that uses cheap IC on samples
to shortlist features, then computes expensive MI only on the top features.

Key Features:
- Cheap IC computation on samples
- Feature shortlisting based on IC
- Expensive MI computation only on top features
- Vectorized binning for MI
- Memory-efficient processing
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

logger = logging.getLogger(__name__)


@dataclass
class TwoStageScoringConfig:
    """Configuration for two-stage scoring."""
    sample_ratio: float = 0.1  # Ratio of data to use for IC sampling
    ic_threshold: float = 0.01  # IC threshold for shortlisting
    top_k_features: int = 100  # Top K features to keep after IC filtering
    mi_bins: int = 5  # Number of bins for MI computation
    use_parallel: bool = True  # Use parallel processing
    max_workers: int = 4  # Number of workers for parallel processing
    memory_limit_mb: int = 500  # Memory limit for processing


class TwoStageScoring:
    """Two-stage scoring system for efficient feature selection."""
    
    def __init__(self, config: TwoStageScoringConfig):
        self.config = config
        self.scoring_stats = {}
        
        tprint_info("🎯 Two-stage scoring initialized")
        tprint_info(f"📊 Sample ratio: {config.sample_ratio}")
        tprint_info(f"📊 IC threshold: {config.ic_threshold}")
        tprint_info(f"📊 Top K features: {config.top_k_features}")
        tprint_info(f"📊 MI bins: {config.mi_bins}")
    
    def score_features(self, 
                      features: pd.DataFrame,
                      target: pd.Series,
                      feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Score features using two-stage approach.
        
        Args:
            features: Feature matrix
            target: Target vector
            feature_names: List of feature names to score
            
        Returns:
            Dictionary with scoring results
        """
        if feature_names is None:
            feature_names = list(features.columns)
        
        tprint_info(f"🎯 Scoring {len(feature_names)} features using two-stage approach")
        
        # Initialize statistics
        self.scoring_stats = {
            'total_features': len(feature_names),
            'sampled_features': 0,
            'ic_computations': 0,
            'mi_computations': 0,
            'shortlisted_features': 0,
            'processing_time': 0
        }
        
        start_time = time.time()
        
        # Stage 1: Cheap IC computation on sample
        tprint_info("📊 Stage 1: Computing IC on sample...")
        ic_scores = self._compute_ic_scores(features, target, feature_names)
        
        # Shortlist features based on IC scores
        shortlisted_features = self._shortlist_features(ic_scores)
        tprint_info(f"📊 Shortlisted {len(shortlisted_features)} features")
        
        # Stage 2: Expensive MI computation on shortlisted features
        tprint_info("📊 Stage 2: Computing MI on shortlisted features...")
        mi_scores = self._compute_mi_scores(features, target, shortlisted_features)
        
        # Combine results
        results = {
            'ic_scores': ic_scores,
            'mi_scores': mi_scores,
            'shortlisted_features': shortlisted_features,
            'final_ranking': self._rank_features(ic_scores, mi_scores, shortlisted_features)
        }
        
        # Update statistics
        self.scoring_stats['processing_time'] = time.time() - start_time
        
        tprint_success(f"✅ Scored {len(feature_names)} features")
        tprint_info(f"📊 IC computations: {self.scoring_stats['ic_computations']}")
        tprint_info(f"📊 MI computations: {self.scoring_stats['mi_computations']}")
        tprint_info(f"📊 Shortlisted features: {self.scoring_stats['shortlisted_features']}")
        
        return results
    
    def _compute_ic_scores(self, 
                          features: pd.DataFrame,
                          target: pd.Series,
                          feature_names: List[str]) -> Dict[str, float]:
        """Compute IC scores on a sample of data."""
        # Sample data for IC computation
        sample_size = int(len(features) * self.config.sample_ratio)
        sample_indices = np.random.choice(len(features), sample_size, replace=False)
        
        sample_features = features.iloc[sample_indices]
        sample_target = target.iloc[sample_indices]
        
        self.scoring_stats['sampled_features'] = sample_size
        
        # Compute IC scores
        ic_scores = {}
        
        if self.config.use_parallel and len(feature_names) > 10:
            ic_scores = self._compute_ic_parallel(sample_features, sample_target, feature_names)
        else:
            ic_scores = self._compute_ic_sequential(sample_features, sample_target, feature_names)
        
        self.scoring_stats['ic_computations'] = len(ic_scores)
        
        return ic_scores
    
    def _compute_ic_sequential(self, 
                             features: pd.DataFrame,
                             target: pd.Series,
                             feature_names: List[str]) -> Dict[str, float]:
        """Compute IC scores sequentially."""
        ic_scores = {}
        
        for feature_name in feature_names:
            if feature_name not in features.columns:
                continue
            
            try:
                ic = self._compute_ic(features[feature_name], target)
                ic_scores[feature_name] = ic
            except Exception as e:
                tprint_debug(f"⚠️ IC computation failed for {feature_name}: {e}")
                ic_scores[feature_name] = 0.0
        
        return ic_scores
    
    def _compute_ic_parallel(self, 
                           features: pd.DataFrame,
                           target: pd.Series,
                           feature_names: List[str]) -> Dict[str, float]:
        """Compute IC scores in parallel."""
        ic_scores = {}
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            for feature_name in feature_names:
                if feature_name in features.columns:
                    future = executor.submit(self._compute_ic, features[feature_name], target)
                    futures.append((feature_name, future))
            
            for feature_name, future in futures:
                try:
                    ic = future.result()
                    ic_scores[feature_name] = ic
                except Exception as e:
                    tprint_debug(f"⚠️ IC computation failed for {feature_name}: {e}")
                    ic_scores[feature_name] = 0.0
        
        return ic_scores
    
    def _compute_ic(self, feature: pd.Series, target: pd.Series) -> float:
        """Compute Information Coefficient (IC) between feature and target."""
        try:
            # Remove NaN values
            valid_mask = ~(feature.isna() | target.isna())
            if valid_mask.sum() < 10:
                return 0.0
            
            feature_clean = feature[valid_mask]
            target_clean = target[valid_mask]
            
            # Compute correlation
            correlation = feature_clean.corr(target_clean)
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _shortlist_features(self, ic_scores: Dict[str, float]) -> List[str]:
        """Shortlist features based on IC scores."""
        # Sort features by absolute IC score
        sorted_features = sorted(
            ic_scores.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Apply threshold and top-K filtering
        shortlisted = []
        for feature_name, ic_score in sorted_features:
            if abs(ic_score) >= self.config.ic_threshold:
                shortlisted.append(feature_name)
                if len(shortlisted) >= self.config.top_k_features:
                    break
        
        self.scoring_stats['shortlisted_features'] = len(shortlisted)
        
        return shortlisted
    
    def _compute_mi_scores(self, 
                          features: pd.DataFrame,
                          target: pd.Series,
                          feature_names: List[str]) -> Dict[str, float]:
        """Compute MI scores for shortlisted features."""
        mi_scores = {}
        
        if self.config.use_parallel and len(feature_names) > 5:
            mi_scores = self._compute_mi_parallel(features, target, feature_names)
        else:
            mi_scores = self._compute_mi_sequential(features, target, feature_names)
        
        self.scoring_stats['mi_computations'] = len(mi_scores)
        
        return mi_scores
    
    def _compute_mi_sequential(self, 
                             features: pd.DataFrame,
                             target: pd.Series,
                             feature_names: List[str]) -> Dict[str, float]:
        """Compute MI scores sequentially."""
        mi_scores = {}
        
        for feature_name in feature_names:
            if feature_name not in features.columns:
                continue
            
            try:
                mi = self._compute_mi(features[feature_name], target)
                mi_scores[feature_name] = mi
            except Exception as e:
                tprint_debug(f"⚠️ MI computation failed for {feature_name}: {e}")
                mi_scores[feature_name] = 0.0
        
        return mi_scores
    
    def _compute_mi_parallel(self, 
                           features: pd.DataFrame,
                           target: pd.Series,
                           feature_names: List[str]) -> Dict[str, float]:
        """Compute MI scores in parallel."""
        mi_scores = {}
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            for feature_name in feature_names:
                if feature_name in features.columns:
                    future = executor.submit(self._compute_mi, features[feature_name], target)
                    futures.append((feature_name, future))
            
            for feature_name, future in futures:
                try:
                    mi = future.result()
                    mi_scores[feature_name] = mi
                except Exception as e:
                    tprint_debug(f"⚠️ MI computation failed for {feature_name}: {e}")
                    mi_scores[feature_name] = 0.0
        
        return mi_scores
    
    def _compute_mi(self, feature: pd.Series, target: pd.Series) -> float:
        """Compute Mutual Information between feature and target."""
        try:
            # Remove NaN values
            valid_mask = ~(feature.isna() | target.isna())
            if valid_mask.sum() < 10:
                return 0.0
            
            feature_clean = feature[valid_mask]
            target_clean = target[valid_mask]
            
            # Determine if target is continuous or discrete
            if target_clean.dtype in ['object', 'category'] or target_clean.nunique() < 10:
                # Discrete target
                mi = mutual_info_classif(
                    feature_clean.values.reshape(-1, 1),
                    target_clean.values,
                    discrete_features=False,
                    random_state=42
                )[0]
            else:
                # Continuous target
                mi = mutual_info_regression(
                    feature_clean.values.reshape(-1, 1),
                    target_clean.values,
                    discrete_features=False,
                    random_state=42
                )[0]
            
            return mi if not np.isnan(mi) else 0.0
            
        except Exception:
            return 0.0
    
    def _rank_features(self, 
                      ic_scores: Dict[str, float],
                      mi_scores: Dict[str, float],
                      shortlisted_features: List[str]) -> List[Tuple[str, float, float, float]]:
        """Rank features by combined IC and MI scores."""
        rankings = []
        
        for feature_name in shortlisted_features:
            ic_score = ic_scores.get(feature_name, 0.0)
            mi_score = mi_scores.get(feature_name, 0.0)
            
            # Combined score (weighted average)
            combined_score = 0.6 * abs(ic_score) + 0.4 * mi_score
            
            rankings.append((feature_name, ic_score, mi_score, combined_score))
        
        # Sort by combined score
        rankings.sort(key=lambda x: x[3], reverse=True)
        
        return rankings
    
    def get_top_features(self, 
                        results: Dict[str, Any],
                        top_k: int = 50) -> List[str]:
        """Get top K features from scoring results."""
        rankings = results['final_ranking']
        return [feature_name for feature_name, _, _, _ in rankings[:top_k]]
    
    def get_scoring_statistics(self) -> Dict[str, Any]:
        """Get scoring statistics."""
        return self.scoring_stats


class VectorizedBinning:
    """Vectorized binning for efficient MI computation."""
    
    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins
    
    def bin_feature(self, feature: pd.Series) -> np.ndarray:
        """Bin feature values for MI computation."""
        try:
            # Remove NaN values
            valid_mask = ~feature.isna()
            if valid_mask.sum() < 10:
                return np.full(len(feature), -1)
            
            feature_clean = feature[valid_mask]
            
            # Use quantile-based binning
            bin_edges = np.quantile(feature_clean, np.linspace(0, 1, self.n_bins + 1))
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            
            # Bin the feature
            binned = np.digitize(feature_clean, bin_edges) - 1
            binned = np.clip(binned, 0, self.n_bins - 1)
            
            # Create full array with NaN handling
            result = np.full(len(feature), -1)
            result[valid_mask] = binned
            
            return result
            
        except Exception:
            return np.full(len(feature), -1)


# Global instances
_two_stage_scoring = None
_vectorized_binning = None

def get_two_stage_scoring() -> TwoStageScoring:
    """Get the global two-stage scoring instance."""
    global _two_stage_scoring
    if _two_stage_scoring is None:
        config = TwoStageScoringConfig()
        _two_stage_scoring = TwoStageScoring(config)
    return _two_stage_scoring

def get_vectorized_binning() -> VectorizedBinning:
    """Get the global vectorized binning instance."""
    global _vectorized_binning
    if _vectorized_binning is None:
        _vectorized_binning = VectorizedBinning()
    return _vectorized_binning

def score_features_two_stage(features: pd.DataFrame,
                           target: pd.Series,
                           feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Score features using two-stage approach.
    
    Args:
        features: Feature matrix
        target: Target vector
        feature_names: List of feature names to score
        
    Returns:
        Dictionary with scoring results
    """
    scoring = get_two_stage_scoring()
    return scoring.score_features(features, target, feature_names)

def get_top_features_from_scoring(results: Dict[str, Any], top_k: int = 50) -> List[str]:
    """Get top K features from scoring results."""
    scoring = get_two_stage_scoring()
    return scoring.get_top_features(results, top_k)