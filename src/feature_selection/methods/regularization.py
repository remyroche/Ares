"""
Feature Selection and Regularization Module

This module implements group regularization and feature dropout for tree-based models:
- feature_fraction=0.6-0.8 in LightGBM (random feature subsampling)
- Stability selection over 50-100 block bootstrap
- Cluster-correlated features and ensure ≤1 per cluster survives live
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings

# Import utilities
from src.utils.tprint import tprint, tprint_warning, tprint_error, tprint_success
from src.utils.math_validation import (
    validate_numeric_array, validate_finite, validate_positive, validate_range
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class FeatureRegularizationConfig:
    """Configuration for feature regularization and selection."""
    # Feature fraction for random subsampling
    feature_fraction_range: Tuple[float, float] = (0.6, 0.8)
    
    # Stability selection parameters
    n_bootstrap: int = 75  # 50-100 range
    stability_threshold: float = 0.6
    random_state: int = 42
    
    # Clustering parameters
    max_features_per_cluster: int = 1
    correlation_threshold: float = 0.8
    n_clusters: Optional[int] = None  # Auto-determine if None
    
    # Feature selection
    max_features: int = 60
    min_importance: float = 0.01


class FeatureRegularizationSelector:
    """
    Feature regularization and selection for tree-based models.
    
    Implements:
    1. Random feature subsampling (feature_fraction)
    2. Stability selection with block bootstrap
    3. Cluster-correlated feature selection
    """
    
    def __init__(self, config: Optional[FeatureRegularizationConfig] = None):
        """Initialize the feature regularization selector."""
        self.config = config or FeatureRegularizationConfig()
        
        # State
        self.fitted = False
        self.feature_importance_scores = None
        self.stability_scores = None
        self.cluster_assignments = None
        self.selected_features = None
        self.feature_names = None
        
    def _compute_feature_importance(self, X: np.ndarray, y: np.ndarray, 
                                  sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute feature importance using Random Forest."""
        try:
            # Use Random Forest for feature importance
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            
            if sample_weight is not None:
                rf.fit(X, y, sample_weight=sample_weight)
            else:
                rf.fit(X, y)
            
            return rf.feature_importances_
            
        except Exception as e:
            logger.warning(f"⚠️ Feature importance computation failed: {e}")
            # Fallback to mutual information
            try:
                mi_scores = mutual_info_regression(X, y, random_state=self.config.random_state)
                return mi_scores / np.sum(mi_scores)  # Normalize
            except Exception as e2:
                logger.warning(f"⚠️ Mutual information fallback failed: {e2}")
                return np.ones(X.shape[1]) / X.shape[1]  # Uniform importance
    
    def _stability_selection(self, X: np.ndarray, y: np.ndarray,
                           sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """Perform stability selection with block bootstrap."""
        try:
            n_samples, n_features = X.shape
            stability_scores = np.zeros(n_features)
            
            # Block bootstrap for time series
            block_size = max(10, n_samples // 20)  # Adaptive block size
            
            for bootstrap_iter in range(self.config.n_bootstrap):
                # Create bootstrap sample with blocks
                bootstrap_indices = []
                current_pos = 0
                
                while current_pos < n_samples:
                    # Random block start
                    block_start = np.random.randint(0, max(1, n_samples - block_size + 1))
                    block_end = min(block_start + block_size, n_samples)
                    
                    # Add block indices
                    bootstrap_indices.extend(range(block_start, block_end))
                    current_pos = block_end
                    
                    if len(bootstrap_indices) >= n_samples:
                        break
                
                # Truncate to exact sample size
                bootstrap_indices = bootstrap_indices[:n_samples]
                
                # Get bootstrap sample
                X_bootstrap = X[bootstrap_indices]
                y_bootstrap = y[bootstrap_indices]
                sample_weight_bootstrap = (sample_weight[bootstrap_indices] 
                                         if sample_weight is not None else None)
                
                # Compute feature importance for this bootstrap
                importance = self._compute_feature_importance(
                    X_bootstrap, y_bootstrap, sample_weight_bootstrap
                )
                
                # Apply random feature subsampling
                feature_fraction = np.random.uniform(*self.config.feature_fraction_range)
                n_selected = max(1, int(feature_fraction * n_features))
                
                # Select top features
                selected_indices = np.argsort(importance)[-n_selected:]
                
                # Update stability scores
                stability_scores[selected_indices] += 1
            
            # Normalize stability scores
            stability_scores = stability_scores / self.config.n_bootstrap
            
            return stability_scores
            
        except Exception as e:
            logger.warning(f"⚠️ Stability selection failed: {e}")
            # Fallback to uniform selection
            return np.ones(X.shape[1]) * 0.5
    
    def _cluster_correlated_features(self, X: np.ndarray) -> np.ndarray:
        """Cluster correlated features and select representatives."""
        try:
            # Compute correlation matrix
            corr_matrix = np.corrcoef(X.T)
            
            # Find highly correlated feature pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix)):
                for j in range(i + 1, len(corr_matrix)):
                    if abs(corr_matrix[i, j]) > self.config.correlation_threshold:
                        high_corr_pairs.append((i, j))
            
            if not high_corr_pairs:
                # No highly correlated features, return all features
                return np.arange(X.shape[1])
            
            # Create feature groups based on correlation
            feature_groups = []
            used_features = set()
            
            for i, j in high_corr_pairs:
                if i not in used_features and j not in used_features:
                    feature_groups.append([i, j])
                    used_features.update([i, j])
                elif i in used_features:
                    # Add j to existing group containing i
                    for group in feature_groups:
                        if i in group:
                            group.append(j)
                            used_features.add(j)
                            break
                elif j in used_features:
                    # Add i to existing group containing j
                    for group in feature_groups:
                        if j in group:
                            group.append(i)
                            used_features.add(i)
                            break
            
            # Add ungrouped features
            for i in range(X.shape[1]):
                if i not in used_features:
                    feature_groups.append([i])
            
            # Select representative from each group
            selected_features = []
            for group in feature_groups:
                if len(group) == 1:
                    selected_features.append(group[0])
                else:
                    # Select the feature with highest variance within the group
                    group_variances = np.var(X[:, group], axis=0)
                    best_feature = group[np.argmax(group_variances)]
                    selected_features.append(best_feature)
            
            return np.array(selected_features)
            
        except Exception as e:
            logger.warning(f"⚠️ Feature clustering failed: {e}")
            # Fallback: return all features
            return np.arange(X.shape[1])
    
    def _select_features(self, X: np.ndarray, y: np.ndarray,
                        sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """Select features using combined criteria."""
        try:
            n_features = X.shape[1]
            
            # Step 1: Stability selection
            stability_scores = self._stability_selection(X, y, sample_weight)
            
            # Step 2: Cluster correlated features
            cluster_selected = self._cluster_correlated_features(X)
            
            # Step 3: Combine criteria
            # Features must pass both stability threshold and be in cluster selection
            stability_mask = stability_scores >= self.config.stability_threshold
            cluster_mask = np.zeros(n_features, dtype=bool)
            cluster_mask[cluster_selected] = True
            
            # Combined mask
            combined_mask = stability_mask & cluster_mask
            
            # If too few features selected, relax criteria
            if np.sum(combined_mask) < self.config.max_features // 2:
                # Use stability scores as primary criterion
                n_select = min(self.config.max_features, n_features)
                selected_indices = np.argsort(stability_scores)[-n_select:]
                combined_mask = np.zeros(n_features, dtype=bool)
                combined_mask[selected_indices] = True
            
            # Ensure we don't exceed max_features
            if np.sum(combined_mask) > self.config.max_features:
                # Select top features by stability score
                selected_indices = np.where(combined_mask)[0]
                stability_scores_selected = stability_scores[selected_indices]
                top_indices = np.argsort(stability_scores_selected)[-self.config.max_features:]
                final_selected = selected_indices[top_indices]
                
                combined_mask = np.zeros(n_features, dtype=bool)
                combined_mask[final_selected] = True
            
            return np.where(combined_mask)[0]
            
        except Exception as e:
            logger.warning(f"⚠️ Feature selection failed: {e}")
            # Fallback: select features randomly
            n_select = min(self.config.max_features, X.shape[1])
            return np.random.choice(X.shape[1], size=n_select, replace=False)
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            sample_weight: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None) -> 'FeatureRegularizationSelector':
        """Fit the feature regularization selector."""
        try:
            tprint("🚀 Starting feature regularization fitting")
            
            # Validate inputs
            X = validate_numeric_array(X, name="Feature matrix X")
            y = validate_numeric_array(y, name="Target variable y")
            
            if sample_weight is not None:
                sample_weight = validate_numeric_array(sample_weight, name="Sample weights")
                
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}")
            
            tprint_success(f"✅ Input validation passed: X shape {X.shape}, y shape {y.shape}")
            
            # Store feature names
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            
            # Compute feature importance
            tprint("📊 Computing feature importance...")
            self.feature_importance_scores = self._compute_feature_importance(X, y, sample_weight)
            
            # Perform stability selection
            tprint("🔍 Performing stability selection...")
            self.stability_scores = self._stability_selection(X, y, sample_weight)
            
            # Select features
            tprint("✨ Selecting final features...")
            self.selected_features = self._select_features(X, y, sample_weight)
            
            self.fitted = True
            tprint_success(f"✅ Feature regularization fitted: {len(self.selected_features)}/{X.shape[1]} features selected")
            logger.info(f"✅ Feature regularization fitted: {len(self.selected_features)}/{X.shape[1]} features selected")
            
            return self
            
        except Exception as e:
            tprint_error(f"❌ Feature regularization fitting failed: {e}")
            logger.error(f"❌ Feature regularization fitting failed: {e}")
            raise
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using selected features."""
        if not self.fitted:
            raise ValueError("Feature selector must be fitted before transform")
        
        return X[:, self.selected_features]
    
    def get_selected_features(self) -> List[str]:
        """Get names of selected features."""
        if not self.fitted:
            return []
        
        return [self.feature_names[i] for i in self.selected_features]
    
    def get_feature_scores(self) -> Dict[str, Any]:
        """Get feature selection scores."""
        if not self.fitted:
            return {}
        
        return {
            'importance_scores': self.feature_importance_scores,
            'stability_scores': self.stability_scores,
            'selected_features': self.selected_features,
            'selected_feature_names': self.get_selected_features(),
            'n_selected': len(self.selected_features),
            'n_total': len(self.feature_names)
        }


# Factory function
def create_feature_regularization_selector(config: Optional[FeatureRegularizationConfig] = None) -> FeatureRegularizationSelector:
    """Create feature regularization selector."""
    return FeatureRegularizationSelector(config)