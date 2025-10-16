"""
Variance Inflation Factor (VIF) Module for Feature Selection

This module provides VIF-based multicollinearity detection and feature selection
capabilities for the NAS-TAS system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)


class VIFModule:
    """
    VIF-based multicollinearity detection and feature selection module.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize VIF module."""
        tprint_info("🚀 Initializing VIF Module")
        tprint_debug(f"Configuration: {config}")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # VIF parameters
        tprint_debug("⚙️ Setting VIF parameters...")
        self.vif_threshold = config.get('vif_threshold', 10.0)
        self.max_vif_threshold = config.get('max_vif_threshold', 5.0)
        self.stepwise_removal = config.get('stepwise_removal', True)
        self.standardize_features = config.get('standardize_features', True)
        self.min_features = config.get('min_features', 2)
        tprint_success("✅ VIF parameters configured")

        # Feature selection parameters
        tprint_debug("🔧 Setting feature selection parameters...")
        self.correlation_threshold = config.get('correlation_threshold', 0.9)
        self.enable_correlation_filtering = config.get('enable_correlation_filtering', True)
        self.enable_variance_filtering = config.get('enable_variance_filtering', True)
        self.variance_threshold = config.get('variance_threshold', 0.01)
        tprint_success("✅ Feature selection parameters configured")

        tprint_success("✅ VIF Module initialized")
        self.logger.info("✅ VIF Module initialized")

    def apply_vif_feature_selection(self, 
                                  X: Union[np.ndarray, pd.DataFrame],
                                  y: Optional[Union[np.ndarray, pd.Series]] = None,
                                  feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Apply VIF-based feature selection.

        Args:
            X: Feature matrix
            y: Optional target variable
            feature_names: Optional feature names

        Returns:
            Dictionary with VIF results and selected features
        """
        try:
            tprint("🔍 [VIF_FEATURE_SELECTION] Starting VIF feature selection", color="blue", bold=True)
            tprint_debug(f"📊 [VIF_FEATURE_SELECTION] Input shape: {X.shape}")
            self.logger.info("🔍 Starting VIF feature selection...")

            # Prepare data
            if isinstance(X, pd.DataFrame):
                X_np = X.values
                if feature_names is None:
                    feature_names = list(X.columns)
            else:
                X_np = np.asarray(X)
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(X_np.shape[1])]

            tprint_debug(f"📊 [VIF_FEATURE_SELECTION] Feature names: {len(feature_names)} features")

            # Pre-filter features if enabled
            if self.enable_correlation_filtering or self.enable_variance_filtering:
                tprint("🔧 [VIF_FEATURE_SELECTION] Pre-filtering features", color="cyan")
                X_filtered, filtered_names, filter_info = self._pre_filter_features(
                    X_np, feature_names
                )
                tprint_success(f"✅ [VIF_FEATURE_SELECTION] Pre-filtering completed: {X_filtered.shape[1]} features retained")
                tprint_debug(f"🔧 [VIF_FEATURE_SELECTION] Filter info: {filter_info}")
            else:
                X_filtered = X_np
                filtered_names = feature_names
                filter_info = {}

            # Standardize features if enabled
            if self.standardize_features:
                tprint("📊 [VIF_FEATURE_SELECTION] Standardizing features", color="cyan")
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_filtered)
                tprint_success("✅ [VIF_FEATURE_SELECTION] Features standardized")
            else:
                X_scaled = X_filtered

            # Calculate VIF scores
            tprint("📈 [VIF_FEATURE_SELECTION] Calculating VIF scores", color="cyan")
            vif_scores = self._calculate_vif_scores(X_scaled, filtered_names)
            tprint_success("✅ [VIF_FEATURE_SELECTION] VIF scores calculated")
            tprint_debug(f"📈 [VIF_FEATURE_SELECTION] VIF scores: {vif_scores}")

            # Remove features with high VIF
            tprint("🎯 [VIF_FEATURE_SELECTION] Removing high VIF features", color="cyan")
            selected_features, removal_info = self._remove_high_vif_features(
                X_scaled, filtered_names, vif_scores
            )
            tprint_success(f"✅ [VIF_FEATURE_SELECTION] High VIF features removed: {len(selected_features)} features retained")
            tprint_debug(f"🔧 [VIF_FEATURE_SELECTION] Removal info: {removal_info}")

            # Calculate final VIF scores for selected features
            if len(selected_features) > 1:
                tprint("📊 [VIF_FEATURE_SELECTION] Calculating final VIF scores", color="cyan")
                final_vif_scores = self._calculate_final_vif_scores(X_scaled, selected_features)
                tprint_success("✅ [VIF_FEATURE_SELECTION] Final VIF scores calculated")
            else:
                final_vif_scores = {}

            tprint_success(f"🎉 [VIF_FEATURE_SELECTION] VIF feature selection completed successfully")
            tprint_performance(f"⚡ [VIF_FEATURE_SELECTION] Final result: {len(selected_features)} features selected from {X.shape[1]} original features")
            
            return {
                'selected_features': selected_features,
                'selected_indices': [filtered_names.index(f) for f in selected_features if f in filtered_names],
                'vif_scores': vif_scores,
                'final_vif_scores': final_vif_scores,
                'removal_info': removal_info,
                'pre_filter_info': filter_info,
                'method': 'vif_based',
                'success': True
            }

        except Exception as e:
            tprint_error(f"❌ [VIF_FEATURE_SELECTION] VIF feature selection failed: {e}")
            tprint_debug(f"🔍 [VIF_FEATURE_SELECTION] Error details: {str(e)}")
            self.logger.error(f"VIF feature selection failed: {e}")
            return {
                'selected_features': [],
                'selected_indices': [],
                'method': 'vif_based',
                'success': False,
                'error': str(e)
            }

    def _pre_filter_features(self, 
                           X: np.ndarray, 
                           feature_names: List[str]) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """Pre-filter features based on correlation and variance."""
        try:
            filter_info = {}
            X_filtered = X.copy()
            filtered_names = feature_names.copy()
            original_shape = X.shape[1]

            # Variance filtering
            if self.enable_variance_filtering:
                tprint_debug("🔧 [VIF_FEATURE_SELECTION] Applying variance filtering")
                variance_filtered = self._filter_low_variance_features(X_filtered, filtered_names)
                X_filtered = variance_filtered['X']
                filtered_names = variance_filtered['names']
                filter_info['variance_filtered'] = {
                    'removed': original_shape - X_filtered.shape[1],
                    'threshold': self.variance_threshold
                }
                tprint_debug(f"🔧 [VIF_FEATURE_SELECTION] Variance filtering: {filter_info['variance_filtered']['removed']} features removed")

            # Correlation filtering
            if self.enable_correlation_filtering and X_filtered.shape[1] > 1:
                tprint_debug("🔧 [VIF_FEATURE_SELECTION] Applying correlation filtering")
                correlation_filtered = self._filter_correlated_features(X_filtered, filtered_names)
                X_filtered = correlation_filtered['X']
                filtered_names = correlation_filtered['names']
                filter_info['correlation_filtered'] = {
                    'removed': len(filtered_names) - X_filtered.shape[1],
                    'threshold': self.correlation_threshold
                }
                tprint_debug(f"🔧 [VIF_FEATURE_SELECTION] Correlation filtering: {filter_info['correlation_filtered']['removed']} features removed")

            return X_filtered, filtered_names, filter_info

        except Exception as e:
            self.logger.warning(f"Pre-filtering failed: {e}")
            return X, feature_names, {}

    def _filter_low_variance_features(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Filter features with low variance."""
        try:
            variances = np.var(X, axis=0)
            variance_mask = variances >= self.variance_threshold
            
            X_filtered = X[:, variance_mask]
            filtered_names = [name for i, name in enumerate(feature_names) if variance_mask[i]]
            
            return {
                'X': X_filtered,
                'names': filtered_names,
                'variances': variances,
                'mask': variance_mask
            }
        except Exception as e:
            self.logger.warning(f"Variance filtering failed: {e}")
            return {'X': X, 'names': feature_names}

    def _filter_correlated_features(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Filter highly correlated features."""
        try:
            if X.shape[1] <= 1:
                return {'X': X, 'names': feature_names}
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(X.T)
            np.fill_diagonal(corr_matrix, 0)  # Set diagonal to 0
            
            # Find highly correlated pairs
            to_remove = set()
            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    if abs(corr_matrix[i, j]) > self.correlation_threshold:
                        # Remove the feature with lower variance
                        if np.var(X[:, i]) < np.var(X[:, j]):
                            to_remove.add(i)
                        else:
                            to_remove.add(j)
            
            # Create mask for features to keep
            keep_mask = np.array([i not in to_remove for i in range(len(feature_names))])
            
            X_filtered = X[:, keep_mask]
            filtered_names = [name for i, name in enumerate(feature_names) if keep_mask[i]]
            
            return {
                'X': X_filtered,
                'names': filtered_names,
                'correlation_matrix': corr_matrix,
                'removed_indices': list(to_remove)
            }
        except Exception as e:
            self.logger.warning(f"Correlation filtering failed: {e}")
            return {'X': X, 'names': feature_names}

    def _calculate_vif_scores(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate VIF scores for all features."""
        try:
            vif_scores = {}
            
            for i, feature_name in enumerate(feature_names):
                # Get feature and other features
                y_feature = X[:, i]
                X_other = np.delete(X, i, axis=1)
                
                if X_other.shape[1] == 0:
                    vif_scores[feature_name] = 1.0
                    continue
                
                # Fit linear regression
                try:
                    reg = LinearRegression()
                    reg.fit(X_other, y_feature)
                    r_squared = reg.score(X_other, y_feature)
                    
                    # Calculate VIF
                    if r_squared >= 1.0:
                        vif_scores[feature_name] = float('inf')
                    else:
                        vif_scores[feature_name] = 1.0 / (1.0 - r_squared)
                except Exception as vif_e:
                    tprint_debug(f"⚠️ VIF calculation failed for {feature_name}: {vif_e}")
                    vif_scores[feature_name] = 1.0
            
            return vif_scores
        except Exception as e:
            self.logger.warning(f"VIF scores calculation failed: {e}")
            return {name: 1.0 for name in feature_names}

    def _remove_high_vif_features(self, 
                               X: np.ndarray, 
                               feature_names: List[str], 
                               vif_scores: Dict[str, float]) -> Tuple[List[str], Dict[str, Any]]:
        """Remove features with high VIF scores."""
        try:
            removal_info = {
                'removed_features': [],
                'removal_order': [],
                'final_vif_scores': {}
            }
            
            current_features = feature_names.copy()
            current_X = X.copy()
            removed_count = 0
            
            if self.stepwise_removal:
                # Stepwise removal: remove highest VIF feature iteratively
                while len(current_features) > self.min_features:
                    # Calculate current VIF scores
                    current_vif_scores = self._calculate_vif_scores(current_X, current_features)
                    
                    # Find feature with highest VIF
                    max_vif_feature = max(current_vif_scores.items(), key=lambda x: x[1])
                    feature_name, vif_score = max_vif_feature
                    
                    # Check if VIF is above threshold
                    if vif_score <= self.vif_threshold:
                        break
                    
                    # Remove feature
                    feature_index = current_features.index(feature_name)
                    current_features.remove(feature_name)
                    current_X = np.delete(current_X, feature_index, axis=1)
                    
                    removal_info['removed_features'].append(feature_name)
                    removal_info['removal_order'].append({
                        'feature': feature_name,
                        'vif_score': vif_score,
                        'iteration': removed_count + 1
                    })
                    
                    removed_count += 1
                    
                    # Safety check
                    if removed_count > len(feature_names):
                        break
            else:
                # Remove all features above threshold at once
                features_to_remove = [
                    feature for feature, vif_score in vif_scores.items() 
                    if vif_score > self.vif_threshold
                ]
                
                # Keep only features below threshold
                keep_mask = [feature not in features_to_remove for feature in feature_names]
                current_features = [feature for feature in feature_names if feature in current_features and feature not in features_to_remove]
                current_X = X[:, keep_mask]
                
                removal_info['removed_features'] = features_to_remove
                removal_info['removal_order'] = [
                    {'feature': feature, 'vif_score': vif_scores.get(feature, 0), 'iteration': 1}
                    for feature in features_to_remove
                ]
            
            # Calculate final VIF scores
            if len(current_features) > 1:
                final_vif_scores = self._calculate_vif_scores(current_X, current_features)
                removal_info['final_vif_scores'] = final_vif_scores
            
            return current_features, removal_info
            
        except Exception as e:
            self.logger.warning(f"High VIF feature removal failed: {e}")
            return feature_names, {'removed_features': [], 'removal_order': []}

    def _calculate_final_vif_scores(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate final VIF scores for selected features."""
        try:
            if len(feature_names) <= 1:
                return {name: 1.0 for name in feature_names}
            
            # Find indices of selected features
            selected_indices = []
            for feature_name in feature_names:
                for i, name in enumerate(feature_names):
                    if name == feature_name:
                        selected_indices.append(i)
                        break
            
            if len(selected_indices) == 0:
                return {name: 1.0 for name in feature_names}
            
            # Get selected features
            X_selected = X[:, selected_indices]
            
            # Calculate VIF scores
            return self._calculate_vif_scores(X_selected, feature_names)
        except Exception as e:
            self.logger.warning(f"Final VIF scores calculation failed: {e}")
            return {name: 1.0 for name in feature_names}

    def get_vif_scores(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Get VIF scores for features."""
        try:
            if self.standardize_features:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X
            
            return self._calculate_vif_scores(X_scaled, feature_names)
        except Exception as e:
            self.logger.error(f"VIF scores calculation failed: {e}")
            return {name: 1.0 for name in feature_names}

    def detect_multicollinearity(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Detect multicollinearity in features."""
        try:
            vif_scores = self.get_vif_scores(X, feature_names)
            
            # Categorize VIF scores
            low_vif = [name for name, score in vif_scores.items() if score < 5.0]
            moderate_vif = [name for name, score in vif_scores.items() if 5.0 <= score < 10.0]
            high_vif = [name for name, score in vif_scores.items() if score >= 10.0]
            
            return {
                'vif_scores': vif_scores,
                'low_vif': low_vif,
                'moderate_vif': moderate_vif,
                'high_vif': high_vif,
                'max_vif': max(vif_scores.values()) if vif_scores else 0.0,
                'avg_vif': np.mean(list(vif_scores.values())) if vif_scores else 0.0
            }
        except Exception as e:
            self.logger.error(f"Multicollinearity detection failed: {e}")
            return {
                'vif_scores': {},
                'low_vif': [],
                'moderate_vif': [],
                'high_vif': [],
                'max_vif': 0.0,
                'avg_vif': 0.0
            }


def create_vif_module(config: Dict[str, Any]) -> VIFModule:
    """Create VIF module."""
    return VIFModule(config)