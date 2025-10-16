"""
Principal Component Analysis (PCA) Module for Feature Selection

This module provides PCA-based dimensionality reduction and feature selection
capabilities for the NAS-TAS system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)


class PCAModule:
    """
    PCA-based feature selection and dimensionality reduction module.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize PCA module."""
        tprint_info("🚀 Initializing PCA Module")
        tprint_debug(f"Configuration: {config}")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # PCA parameters
        tprint_debug("⚙️ Setting PCA parameters...")
        self.n_components = config.get('n_components', None)
        self.variance_threshold = config.get('variance_threshold', 0.95)
        self.min_variance_explained = config.get('min_variance_explained', 0.01)
        self.whiten = config.get('whiten', False)
        self.random_state = config.get('random_state', 42)
        tprint_success("✅ PCA parameters configured")

        # Feature selection parameters
        tprint_debug("🔧 Setting feature selection parameters...")
        self.correlation_threshold = config.get('correlation_threshold', 0.9)
        self.variance_threshold_feature = config.get('variance_threshold_feature', 0.01)
        self.enable_correlation_filtering = config.get('enable_correlation_filtering', True)
        self.enable_variance_filtering = config.get('enable_variance_filtering', True)
        tprint_success("✅ Feature selection parameters configured")

        tprint_success("✅ PCA Module initialized")
        self.logger.info("✅ PCA Module initialized")

    def apply_pca_feature_selection(self, 
                                  X: Union[np.ndarray, pd.DataFrame],
                                  y: Optional[Union[np.ndarray, pd.Series]] = None,
                                  feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Apply PCA-based feature selection.

        Args:
            X: Feature matrix
            y: Optional target variable
            feature_names: Optional feature names

        Returns:
            Dictionary with PCA results and selected features
        """
        try:
            tprint("🔍 [PCA_FEATURE_SELECTION] Starting PCA feature selection", color="blue", bold=True)
            tprint_debug(f"📊 [PCA_FEATURE_SELECTION] Input shape: {X.shape}")
            self.logger.info("🔍 Starting PCA feature selection...")

            # Prepare data
            if isinstance(X, pd.DataFrame):
                X_np = X.values
                if feature_names is None:
                    feature_names = list(X.columns)
            else:
                X_np = np.asarray(X)
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(X_np.shape[1])]

            tprint_debug(f"📊 [PCA_FEATURE_SELECTION] Feature names: {len(feature_names)} features")

            # Pre-filter features if enabled
            if self.enable_correlation_filtering or self.enable_variance_filtering:
                tprint("🔧 [PCA_FEATURE_SELECTION] Pre-filtering features", color="cyan")
                X_filtered, filtered_names, filter_info = self._pre_filter_features(
                    X_np, feature_names
                )
                tprint_success(f"✅ [PCA_FEATURE_SELECTION] Pre-filtering completed: {X_filtered.shape[1]} features retained")
                tprint_debug(f"🔧 [PCA_FEATURE_SELECTION] Filter info: {filter_info}")
            else:
                X_filtered = X_np
                filtered_names = feature_names
                filter_info = {}

            # Standardize features
            tprint("📊 [PCA_FEATURE_SELECTION] Standardizing features", color="cyan")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_filtered)
            tprint_success("✅ [PCA_FEATURE_SELECTION] Features standardized")

            # Determine optimal number of components
            tprint("🎯 [PCA_FEATURE_SELECTION] Determining optimal components", color="cyan")
            optimal_components = self._determine_optimal_components(X_scaled)
            tprint_success(f"✅ [PCA_FEATURE_SELECTION] Optimal components: {optimal_components}")

            # Apply PCA
            tprint("🔄 [PCA_FEATURE_SELECTION] Applying PCA", color="cyan")
            pca_result = self._apply_pca(X_scaled, optimal_components)
            tprint_success("✅ [PCA_FEATURE_SELECTION] PCA applied successfully")

            # Extract feature importance from PCA
            tprint("📈 [PCA_FEATURE_SELECTION] Extracting feature importance", color="cyan")
            feature_importance = self._extract_feature_importance(
                pca_result, X_scaled, filtered_names
            )
            tprint_success("✅ [PCA_FEATURE_SELECTION] Feature importance extracted")

            # Select top features based on PCA loadings
            tprint("🎯 [PCA_FEATURE_SELECTION] Selecting top features", color="cyan")
            selected_features = self._select_features_from_pca(
                feature_importance, X_scaled.shape[1]
            )
            tprint_success(f"✅ [PCA_FEATURE_SELECTION] Selected {len(selected_features)} features")

            tprint_success(f"🎉 [PCA_FEATURE_SELECTION] PCA feature selection completed successfully")
            tprint_performance(f"⚡ [PCA_FEATURE_SELECTION] Final result: {len(selected_features)} features selected from {X.shape[1]} original features")
            
            return {
                'selected_features': selected_features,
                'selected_indices': [filtered_names.index(f) for f in selected_features if f in filtered_names],
                'pca_result': pca_result,
                'feature_importance': feature_importance,
                'explained_variance_ratio': pca_result.explained_variance_ratio_,
                'cumulative_variance': np.cumsum(pca_result.explained_variance_ratio_),
                'n_components': optimal_components,
                'pre_filter_info': filter_info,
                'method': 'pca_based',
                'success': True
            }

        except Exception as e:
            tprint_error(f"❌ [PCA_FEATURE_SELECTION] PCA feature selection failed: {e}")
            tprint_debug(f"🔍 [PCA_FEATURE_SELECTION] Error details: {str(e)}")
            self.logger.error(f"PCA feature selection failed: {e}")
            return {
                'selected_features': [],
                'selected_indices': [],
                'method': 'pca_based',
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
                tprint_debug("🔧 [PCA_FEATURE_SELECTION] Applying variance filtering")
                variance_filtered = self._filter_low_variance_features(X_filtered, filtered_names)
                X_filtered = variance_filtered['X']
                filtered_names = variance_filtered['names']
                filter_info['variance_filtered'] = {
                    'removed': original_shape - X_filtered.shape[1],
                    'threshold': self.variance_threshold_feature
                }
                tprint_debug(f"🔧 [PCA_FEATURE_SELECTION] Variance filtering: {filter_info['variance_filtered']['removed']} features removed")

            # Correlation filtering
            if self.enable_correlation_filtering and X_filtered.shape[1] > 1:
                tprint_debug("🔧 [PCA_FEATURE_SELECTION] Applying correlation filtering")
                correlation_filtered = self._filter_correlated_features(X_filtered, filtered_names)
                X_filtered = correlation_filtered['X']
                filtered_names = correlation_filtered['names']
                filter_info['correlation_filtered'] = {
                    'removed': len(filtered_names) - X_filtered.shape[1],
                    'threshold': self.correlation_threshold
                }
                tprint_debug(f"🔧 [PCA_FEATURE_SELECTION] Correlation filtering: {filter_info['correlation_filtered']['removed']} features removed")

            return X_filtered, filtered_names, filter_info

        except Exception as e:
            self.logger.warning(f"Pre-filtering failed: {e}")
            return X, feature_names, {}

    def _filter_low_variance_features(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Filter features with low variance."""
        try:
            variances = np.var(X, axis=0)
            variance_mask = variances >= self.variance_threshold_feature
            
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

    def _determine_optimal_components(self, X: np.ndarray) -> int:
        """Determine optimal number of PCA components."""
        try:
            if self.n_components is not None:
                return min(self.n_components, X.shape[1])
            
            # Find number of components that explain variance_threshold of variance
            pca_full = PCA()
            pca_full.fit(X)
            cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
            
            n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
            n_components = max(1, min(n_components, X.shape[1]))
            
            return n_components
        except Exception as e:
            self.logger.warning(f"Optimal components determination failed: {e}")
            return min(10, X.shape[1])

    def _apply_pca(self, X: np.ndarray, n_components: int) -> PCA:
        """Apply PCA transformation."""
        try:
            pca = PCA(
                n_components=n_components,
                whiten=self.whiten,
                random_state=self.random_state
            )
            pca.fit(X)
            return pca
        except Exception as e:
            self.logger.error(f"PCA application failed: {e}")
            raise

    def _extract_feature_importance(self, 
                                  pca: PCA, 
                                  X: np.ndarray, 
                                  feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from PCA loadings."""
        try:
            # Get component loadings
            components = pca.components_  # Shape: (n_components, n_features)
            
            # Calculate feature importance as sum of absolute loadings across components
            feature_importance = {}
            for i, feature_name in enumerate(feature_names):
                importance = np.sum(np.abs(components[:, i]))
                feature_importance[feature_name] = importance
            
            return feature_importance
        except Exception as e:
            self.logger.warning(f"Feature importance extraction failed: {e}")
            return {name: 0.0 for name in feature_names}

    def _select_features_from_pca(self, 
                                feature_importance: Dict[str, float], 
                                max_features: int) -> List[str]:
        """Select top features based on PCA importance."""
        try:
            # Sort features by importance
            sorted_features = sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Select top features
            n_features = min(len(sorted_features), max_features)
            selected_features = [feature for feature, _ in sorted_features[:n_features]]
            
            return selected_features
        except Exception as e:
            self.logger.warning(f"Feature selection from PCA failed: {e}")
            return []

    def get_pca_components(self, X: np.ndarray) -> np.ndarray:
        """Get PCA transformed components."""
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            pca = PCA(n_components=self.n_components, random_state=self.random_state)
            X_pca = pca.fit_transform(X_scaled)
            
            return X_pca
        except Exception as e:
            self.logger.error(f"PCA components extraction failed: {e}")
            return X

    def get_explained_variance_ratio(self, X: np.ndarray) -> np.ndarray:
        """Get explained variance ratio for all components."""
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            pca = PCA(random_state=self.random_state)
            pca.fit(X_scaled)
            
            return pca.explained_variance_ratio_
        except Exception as e:
            self.logger.error(f"Explained variance ratio calculation failed: {e}")
            return np.array([])


def create_pca_module(config: Dict[str, Any]) -> PCAModule:
    """Create PCA module."""
    return PCAModule(config)