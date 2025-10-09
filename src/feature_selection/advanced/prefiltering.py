"""
Pre-filtering with mRMR + Spearman Correlation

This module implements pre-filtering using 70% mRMR and 30% Spearman correlation
for efficient feature reduction before advanced selection methods.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

# Import project utilities
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug
from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, HardwareConfig

logger = logging.getLogger(__name__)

class MRMRSpearmanPreFilter:
    """Pre-filter using mRMR and Spearman correlation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize mRMR-Spearman pre-filter."""
        self.config = config or {
            'mrmr_weight': 0.7,
            'spearman_weight': 0.3,
            'max_features': None,
            'min_features': 1,
            'correlation_threshold': 0.95,
            'enable_hardware_optimization': True,
            'n_jobs': -1,
            'random_state': 42
        }
        
        self.logger = logger.getChild('MRMRSpearmanPreFilter')
        
        # Initialize hardware optimization
        if self.config.get('enable_hardware_optimization', True):
            self.cpu_optimizer = M1CPUOptimizer()
            hw_config = HardwareConfig(
                cpu_optimization_level='balanced',
                enable_adaptive_optimization=True
            )
            self.hardware_manager = UnifiedHardwareManager(hw_config)
        else:
            self.cpu_optimizer = None
            self.hardware_manager = None
        
        # Performance tracking
        self.performance_stats = {
            'total_prefilters': 0,
            'avg_prefilter_time': 0.0,
            'features_removed': 0,
            'mrmr_selections': 0,
            'spearman_selections': 0
        }
        
        tprint_success("🔧 MRMRSpearmanPreFilter initialized")
    
    def prefilter_features(self, X: np.ndarray, y: np.ndarray,
                          target_features: int,
                          feature_names: Optional[List[str]] = None,
                          remove_correlated: bool = True) -> Dict[str, Any]:
        """Pre-filter features using mRMR + Spearman correlation."""
        tprint_info(f"🔧 Pre-filtering features: {X.shape} -> target: {target_features}")
        
        start_time = time.time()
        
        try:
            # Prepare feature names
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            n_total = X.shape[1]
            n_target = target_features
            
            # Calculate pre-filter count (50% of (total - target))
            prefilter_count = max(1, int(0.5 * (n_total - n_target)))
            prefilter_count = min(prefilter_count, n_total - n_target)
            
            tprint_debug(f"🔧 Pre-filter count: {prefilter_count} features")
            
            # Step 1: Calculate mRMR scores
            mrmr_scores = self._calculate_mrmr_scores(X, y, feature_names)
            
            # Step 2: Calculate Spearman correlation scores
            spearman_scores = self._calculate_spearman_scores(X, y, feature_names)
            
            # Step 3: Combine scores with weights
            combined_scores = self._combine_scores(mrmr_scores, spearman_scores)
            
            # Step 4: Select top features
            selected_indices = self._select_top_features(combined_scores, prefilter_count)
            
            # Step 5: Remove highly correlated features if requested
            if remove_correlated:
                selected_indices = self._remove_correlated_features(
                    X, selected_indices, feature_names
                )
            
            # Create feature mask
            feature_mask = np.zeros(n_total, dtype=bool)
            feature_mask[selected_indices] = True
            
            # Get filtered data
            X_filtered = X[:, selected_indices]
            filtered_feature_names = [feature_names[i] for i in selected_indices]
            
            # Update statistics
            end_time = time.time()
            execution_time = end_time - start_time
            self.performance_stats['total_prefilters'] += 1
            self.performance_stats['features_removed'] += n_total - len(selected_indices)
            self.performance_stats['avg_prefilter_time'] = (
                (self.performance_stats['avg_prefilter_time'] * (self.performance_stats['total_prefilters'] - 1) + 
                 execution_time) / self.performance_stats['total_prefilters']
            )
            
            result = {
                'success': True,
                'X_filtered': X_filtered,
                'feature_mask': feature_mask,
                'selected_indices': selected_indices,
                'filtered_feature_names': filtered_feature_names,
                'n_original': n_total,
                'n_filtered': len(selected_indices),
                'n_removed': n_total - len(selected_indices),
                'mrmr_scores': mrmr_scores,
                'spearman_scores': spearman_scores,
                'combined_scores': combined_scores,
                'execution_time': execution_time
            }
            
            tprint_success(f"✅ Pre-filtering completed: {n_total} -> {len(selected_indices)} features in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Pre-filtering failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _calculate_mrmr_scores(self, X: np.ndarray, y: np.ndarray, 
                              feature_names: List[str]) -> Dict[str, float]:
        """Calculate mRMR scores for all features."""
        tprint_debug("🔧 Calculating mRMR scores")
        
        try:
            # Determine if classification or regression
            is_classification = len(np.unique(y)) < 10 and np.all(y == y.astype(int))
            
            # Calculate mutual information
            if is_classification:
                mi_scores = mutual_info_classif(X, y, random_state=self.config['random_state'])
            else:
                mi_scores = mutual_info_regression(X, y, random_state=self.config['random_state'])
            
            # Calculate mRMR scores (relevance - redundancy)
            mrmr_scores = {}
            n_features = X.shape[1]
            
            for i in range(n_features):
                # Relevance (mutual information with target)
                relevance = mi_scores[i]
                
                # Redundancy (average mutual information with other features)
                redundancy = 0.0
                if n_features > 1:
                    other_features = [j for j in range(n_features) if j != i]
                    for j in other_features:
                        # Calculate mutual information between features
                        if is_classification:
                            mi_ij = mutual_info_classif(
                                X[:, [i, j]], np.zeros(X.shape[0]), 
                                random_state=self.config['random_state']
                            )[0]
                        else:
                            mi_ij = mutual_info_regression(
                                X[:, [i, j]], np.zeros(X.shape[0]), 
                                random_state=self.config['random_state']
                            )[0]
                        redundancy += mi_ij
                    redundancy /= len(other_features)
                
                # mRMR score = relevance - redundancy
                mrmr_score = relevance - redundancy
                mrmr_scores[feature_names[i]] = float(mrmr_score)
            
            self.performance_stats['mrmr_selections'] += 1
            return mrmr_scores
            
        except Exception as e:
            self.logger.warning(f"mRMR calculation failed: {e}")
            # Fallback to simple mutual information
            if is_classification:
                mi_scores = mutual_info_classif(X, y, random_state=self.config['random_state'])
            else:
                mi_scores = mutual_info_regression(X, y, random_state=self.config['random_state'])
            
            return {feature_names[i]: float(mi_scores[i]) for i in range(len(feature_names))}
    
    def _calculate_spearman_scores(self, X: np.ndarray, y: np.ndarray,
                                 feature_names: List[str]) -> Dict[str, float]:
        """Calculate Spearman correlation scores for all features."""
        tprint_debug("🔧 Calculating Spearman correlation scores")
        
        try:
            spearman_scores = {}
            
            for i, feature_name in enumerate(feature_names):
                # Calculate Spearman correlation with target
                correlation, _ = spearmanr(X[:, i], y)
                spearman_scores[feature_name] = float(abs(correlation))  # Use absolute value
            
            self.performance_stats['spearman_selections'] += 1
            return spearman_scores
            
        except Exception as e:
            self.logger.warning(f"Spearman calculation failed: {e}")
            # Fallback to simple correlation
            correlations = np.abs(np.corrcoef(X.T, y)[-1, :-1])
            return {feature_names[i]: float(correlations[i]) for i in range(len(feature_names))}
    
    def _combine_scores(self, mrmr_scores: Dict[str, float], 
                       spearman_scores: Dict[str, float]) -> Dict[str, float]:
        """Combine mRMR and Spearman scores with weights."""
        tprint_debug("🔧 Combining scores with weights")
        
        try:
            combined_scores = {}
            mrmr_weight = self.config['mrmr_weight']
            spearman_weight = self.config['spearman_weight']
            
            # Normalize scores to [0, 1] range
            mrmr_values = list(mrmr_scores.values())
            spearman_values = list(spearman_scores.values())
            
            if mrmr_values:
                mrmr_min, mrmr_max = min(mrmr_values), max(mrmr_values)
                mrmr_range = mrmr_max - mrmr_min if mrmr_max > mrmr_min else 1.0
                mrmr_normalized = {k: (v - mrmr_min) / mrmr_range for k, v in mrmr_scores.items()}
            else:
                mrmr_normalized = mrmr_scores
            
            if spearman_values:
                spearman_min, spearman_max = min(spearman_values), max(spearman_values)
                spearman_range = spearman_max - spearman_min if spearman_max > spearman_min else 1.0
                spearman_normalized = {k: (v - spearman_min) / spearman_range for k, v in spearman_scores.items()}
            else:
                spearman_normalized = spearman_scores
            
            # Combine normalized scores
            for feature_name in mrmr_scores.keys():
                mrmr_score = mrmr_normalized.get(feature_name, 0.0)
                spearman_score = spearman_normalized.get(feature_name, 0.0)
                combined_score = mrmr_weight * mrmr_score + spearman_weight * spearman_score
                combined_scores[feature_name] = float(combined_score)
            
            return combined_scores
            
        except Exception as e:
            self.logger.warning(f"Score combination failed: {e}")
            # Fallback to mRMR scores only
            return mrmr_scores
    
    def _select_top_features(self, combined_scores: Dict[str, float], 
                           n_features: int) -> List[int]:
        """Select top features based on combined scores."""
        tprint_debug(f"🔧 Selecting top {n_features} features")
        
        try:
            # Sort features by combined score
            sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Select top features
            selected_features = sorted_features[:n_features]
            selected_indices = []
            
            for feature_name, score in selected_features:
                # Find feature index (assuming feature_names are in order)
                try:
                    feature_index = int(feature_name.split('_')[-1])  # Extract index from feature_name
                    selected_indices.append(feature_index)
                except (ValueError, IndexError):
                    # Fallback: use position in sorted list
                    selected_indices.append(len(selected_indices))
            
            return selected_indices[:n_features]
            
        except Exception as e:
            self.logger.warning(f"Top feature selection failed: {e}")
            # Fallback: select first n_features
            return list(range(min(n_features, len(combined_scores))))
    
    def _remove_correlated_features(self, X: np.ndarray, selected_indices: List[int],
                                  feature_names: List[str]) -> List[int]:
        """Remove highly correlated features from selected features."""
        tprint_debug("🔧 Removing highly correlated features")
        
        try:
            if len(selected_indices) <= 1:
                return selected_indices
            
            # Get selected features data
            X_selected = X[:, selected_indices]
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(X_selected.T)
            
            # Find highly correlated pairs
            threshold = self.config.get('correlation_threshold', 0.95)
            to_remove = set()
            
            for i in range(len(selected_indices)):
                if i in to_remove:
                    continue
                
                for j in range(i + 1, len(selected_indices)):
                    if j in to_remove:
                        continue
                    
                    if abs(corr_matrix[i, j]) > threshold:
                        # Remove the feature with lower index (keep the first one)
                        to_remove.add(j)
            
            # Remove highly correlated features
            filtered_indices = [selected_indices[i] for i in range(len(selected_indices)) if i not in to_remove]
            
            tprint_debug(f"🔧 Removed {len(to_remove)} highly correlated features")
            return filtered_indices
            
        except Exception as e:
            self.logger.warning(f"Correlation removal failed: {e}")
            return selected_indices
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['total_prefilters'] > 0:
            stats['avg_features_removed'] = stats['features_removed'] / stats['total_prefilters']
            stats['mrmr_usage_ratio'] = stats['mrmr_selections'] / stats['total_prefilters']
            stats['spearman_usage_ratio'] = stats['spearman_selections'] / stats['total_prefilters']
        else:
            stats['avg_features_removed'] = 0.0
            stats['mrmr_usage_ratio'] = 0.0
            stats['spearman_usage_ratio'] = 0.0
        
        return stats
    
    def get_prefilter_insights(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Get insights about pre-filtering results."""
        if not result.get('success', False):
            return {'error': 'Pre-filtering failed'}
        
        insights = {
            'n_original': result['n_original'],
            'n_filtered': result['n_filtered'],
            'n_removed': result['n_removed'],
            'removal_ratio': result['n_removed'] / result['n_original'],
            'execution_time': result['execution_time'],
            'score_distribution': {},
            'top_features': []
        }
        
        # Analyze score distribution
        if 'combined_scores' in result:
            scores = list(result['combined_scores'].values())
            if scores:
                insights['score_distribution'] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'median': float(np.median(scores))
                }
        
        # Get top features
        if 'filtered_feature_names' in result:
            insights['top_features'] = result['filtered_feature_names'][:10]  # Top 10
        
        return insights

def create_mrmr_spearman_prefilter(config: Optional[Dict[str, Any]] = None) -> MRMRSpearmanPreFilter:
    """Create an mRMR-Spearman pre-filter."""
    return MRMRSpearmanPreFilter(config)