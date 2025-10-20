"""
Dimensionality Reducer

This module provides comprehensive dimensionality reduction capabilities for
HDBSCAN-based regime discovery, including PCA, UMAP, t-SNE, and other
advanced techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)

@dataclass
class DimensionalityReducerConfig:
    """Configuration for dimensionality reduction."""
    # Method selection
    method: str = 'pca'  # 'pca', 'umap', 'tsne', 'ica', 'svd', 'isomap', 'lle', 'lda', 'random'
    
    # Common parameters
    n_components: int = 20
    random_state: int = 42
    
    # PCA parameters
    pca_whiten: bool = False
    pca_svd_solver: str = 'auto'
    
    # UMAP parameters
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = 'euclidean'
    umap_spread: float = 1.0
    
    # t-SNE parameters
    tsne_perplexity: float = 30.0
    tsne_early_exaggeration: float = 12.0
    tsne_learning_rate: float = 200.0
    tsne_n_iter: int = 1000
    
    # ICA parameters
    ica_algorithm: str = 'parallel'
    ica_fun: str = 'logcosh'
    ica_max_iter: int = 200
    
    # SVD parameters
    svd_algorithm: str = 'randomized'
    svd_n_iter: int = 5
    
    # Isomap parameters
    isomap_n_neighbors: int = 5
    isomap_metric: str = 'euclidean'
    
    # LLE parameters
    lle_n_neighbors: int = 5
    lle_method: str = 'standard'
    lle_reg: float = 0.001
    
    # Random projection parameters
    random_eps: float = 0.5
    random_density: float = 'auto'
    
    # Preprocessing
    standardize: bool = True
    remove_correlated: bool = True
    correlation_threshold: float = 0.95
    
    # Validation
    validate_input: bool = True
    min_samples: int = 10
    max_components: Optional[int] = None

class DimensionalityReducer:
    """
    Comprehensive dimensionality reducer for HDBSCAN regime discovery.
    
    Supports multiple dimensionality reduction techniques including PCA, UMAP,
    t-SNE, ICA, and other advanced methods.
    """
    
    def __init__(self, config: Optional[DimensionalityReducerConfig] = None):
        """
        Initialize dimensionality reducer.
        
        Args:
            config: Configuration for dimensionality reduction
        """
        self.config = config or DimensionalityReducerConfig()
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.reduction_stats = {}
        
    def reduce(self, 
               features: np.ndarray, 
               fit: bool = True,
               target: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reduce dimensionality of features.
        
        Args:
            features: Input feature matrix (n_samples, n_features)
            fit: Whether to fit the model (True) or transform only (False)
            target: Target variable for supervised methods (optional)
            
        Returns:
            Tuple of (reduced_features, reduction_info)
        """
        try:
            logger.info(f"📉 Starting dimensionality reduction using {self.config.method}...")
            
            # Validate input
            if self.config.validate_input:
                features = self._validate_input(features)
            
            # Preprocess features
            if fit:
                features = self._preprocess_features(features, fit=True)
            else:
                features = self._preprocess_features(features, fit=False)
            
            # Determine number of components
            n_components = self._determine_n_components(features)
            
            # Apply dimensionality reduction
            if fit:
                reduced_features, model = self._fit_reduction(features, n_components, target)
                self.model = model
            else:
                if self.model is None:
                    raise ValueError("Model not fitted. Call with fit=True first.")
                reduced_features = self._transform_features(features)
            
            # Calculate reduction statistics
            reduction_info = self._calculate_reduction_stats(features, reduced_features)
            self.reduction_stats = reduction_info
            
            logger.info(f"✅ Dimensionality reduction completed. Shape: {features.shape} -> {reduced_features.shape}")
            
            return reduced_features, reduction_info
            
        except Exception as e:
            logger.error(f"❌ Dimensionality reduction failed: {e}")
            # Return original features as fallback
            return features, {'error': str(e)}
    
    def _validate_input(self, features: np.ndarray) -> np.ndarray:
        """Validate input features."""
        try:
            # Check for NaN values
            if np.isnan(features).any():
                logger.warning("⚠️ Found NaN values, filling with 0")
                features = np.nan_to_num(features, nan=0.0)
            
            # Check for infinite values
            if np.isinf(features).any():
                logger.warning("⚠️ Found infinite values, clipping")
                features = np.clip(features, -1e10, 1e10)
            
            # Check minimum samples
            if len(features) < self.config.min_samples:
                raise ValueError(f"Insufficient samples: {len(features)} < {self.config.min_samples}")
            
            # Check for constant features
            feature_vars = np.var(features, axis=0)
            constant_features = feature_vars < 1e-10
            if constant_features.any():
                logger.warning(f"⚠️ Found {constant_features.sum()} constant features, removing them")
                features = features[:, ~constant_features]
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Input validation failed: {e}")
            return features
    
    def _preprocess_features(self, features: np.ndarray, fit: bool = True) -> np.ndarray:
        """Preprocess features before dimensionality reduction."""
        try:
            # Standardize features
            if self.config.standardize:
                if fit:
                    self.scaler = StandardScaler()
                    features = self.scaler.fit_transform(features)
                else:
                    if self.scaler is None:
                        raise ValueError("Scaler not fitted. Call with fit=True first.")
                    features = self.scaler.transform(features)
            
            # Remove highly correlated features
            if self.config.remove_correlated and features.shape[1] > 1:
                if fit:
                    features = self._remove_correlated_features(features)
                else:
                    # Use previously selected features
                    if hasattr(self, 'selected_features_mask'):
                        features = features[:, self.selected_features_mask]
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Feature preprocessing failed: {e}")
            return features
    
    def _remove_correlated_features(self, features: np.ndarray) -> np.ndarray:
        """Remove highly correlated features."""
        try:
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(features.T)
            
            # Find highly correlated pairs
            upper_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            high_corr_pairs = np.where((np.abs(corr_matrix) > self.config.correlation_threshold) & upper_tri)
            
            # Select features to keep
            features_to_keep = []
            features_to_remove = set()
            
            for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
                if i not in features_to_remove and j not in features_to_remove:
                    # Keep the feature with higher variance
                    if np.var(features[:, i]) >= np.var(features[:, j]):
                        features_to_keep.append(i)
                        features_to_remove.add(j)
                    else:
                        features_to_keep.append(j)
                        features_to_remove.add(i)
            
            # Add remaining features
            for i in range(features.shape[1]):
                if i not in features_to_remove:
                    features_to_keep.append(i)
            
            # Create mask for selected features
            self.selected_features_mask = np.zeros(features.shape[1], dtype=bool)
            self.selected_features_mask[features_to_keep] = True
            
            logger.info(f"✅ Removed {len(features_to_remove)} highly correlated features")
            
            return features[:, self.selected_features_mask]
            
        except Exception as e:
            logger.error(f"❌ Correlated feature removal failed: {e}")
            return features
    
    def _determine_n_components(self, features: np.ndarray) -> int:
        """Determine appropriate number of components."""
        try:
            n_components = self.config.n_components
            
            # Apply maximum components limit
            if self.config.max_components is not None:
                n_components = min(n_components, self.config.max_components)
            
            # Ensure n_components doesn't exceed feature dimensions
            n_components = min(n_components, features.shape[1])
            
            # Ensure n_components doesn't exceed sample count for some methods
            if self.config.method in ['lda']:
                n_components = min(n_components, features.shape[0] - 1)
            
            # Ensure minimum components
            n_components = max(1, n_components)
            
            return n_components
            
        except Exception as e:
            logger.error(f"❌ Component determination failed: {e}")
            return min(self.config.n_components, features.shape[1])
    
    def _fit_reduction(self, features: np.ndarray, n_components: int, target: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Any]:
        """Fit dimensionality reduction model."""
        try:
            if self.config.method == 'pca':
                model = PCA(
                    n_components=n_components,
                    whiten=self.config.pca_whiten,
                    svd_solver=self.config.pca_svd_solver,
                    random_state=self.config.random_state
                )
                
            elif self.config.method == 'umap':
                try:
                    import umap
                    model = umap.UMAP(
                        n_components=n_components,
                        n_neighbors=self.config.umap_n_neighbors,
                        min_dist=self.config.umap_min_dist,
                        metric=self.config.umap_metric,
                        spread=self.config.umap_spread,
                        random_state=self.config.random_state
                    )
                except ImportError:
                    logger.warning("⚠️ UMAP not available, falling back to PCA")
                    model = PCA(n_components=n_components, random_state=self.config.random_state)
                
            elif self.config.method == 'tsne':
                model = TSNE(
                    n_components=n_components,
                    perplexity=self.config.tsne_perplexity,
                    early_exaggeration=self.config.tsne_early_exaggeration,
                    learning_rate=self.config.tsne_learning_rate,
                    n_iter=self.config.tsne_n_iter,
                    random_state=self.config.random_state
                )
                
            elif self.config.method == 'ica':
                model = FastICA(
                    n_components=n_components,
                    algorithm=self.config.ica_algorithm,
                    fun=self.config.ica_fun,
                    max_iter=self.config.ica_max_iter,
                    random_state=self.config.random_state
                )
                
            elif self.config.method == 'svd':
                model = TruncatedSVD(
                    n_components=n_components,
                    algorithm=self.config.svd_algorithm,
                    n_iter=self.config.svd_n_iter,
                    random_state=self.config.random_state
                )
                
            elif self.config.method == 'isomap':
                model = Isomap(
                    n_components=n_components,
                    n_neighbors=self.config.isomap_n_neighbors,
                    metric=self.config.isomap_metric
                )
                
            elif self.config.method == 'lle':
                model = LocallyLinearEmbedding(
                    n_components=n_components,
                    n_neighbors=self.config.lle_n_neighbors,
                    method=self.config.lle_method,
                    reg=self.config.lle_reg,
                    random_state=self.config.random_state
                )
                
            elif self.config.method == 'lda':
                if target is None:
                    logger.warning("⚠️ LDA requires target variable, falling back to PCA")
                    model = PCA(n_components=n_components, random_state=self.config.random_state)
                else:
                    model = LinearDiscriminantAnalysis(n_components=n_components)
                
            elif self.config.method == 'random':
                model = GaussianRandomProjection(
                    n_components=n_components,
                    eps=self.config.random_eps,
                    random_state=self.config.random_state
                )
                
            else:
                logger.warning(f"⚠️ Unknown method {self.config.method}, falling back to PCA")
                model = PCA(n_components=n_components, random_state=self.config.random_state)
            
            # Fit the model
            if self.config.method == 'lda' and target is not None:
                reduced_features = model.fit_transform(features, target)
            else:
                reduced_features = model.fit_transform(features)
            
            # Store feature names
            self.feature_names = [f"{self.config.method.upper()}_{i+1}" for i in range(reduced_features.shape[1])]
            
            return reduced_features, model
            
        except Exception as e:
            logger.error(f"❌ Model fitting failed: {e}")
            # Fallback to PCA
            model = PCA(n_components=n_components, random_state=self.config.random_state)
            reduced_features = model.fit_transform(features)
            return reduced_features, model
    
    def _transform_features(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted model."""
        try:
            if self.model is None:
                raise ValueError("Model not fitted. Call with fit=True first.")
            
            return self.model.transform(features)
            
        except Exception as e:
            logger.error(f"❌ Feature transformation failed: {e}")
            return features
    
    def _calculate_reduction_stats(self, original_features: np.ndarray, reduced_features: np.ndarray) -> Dict[str, Any]:
        """Calculate reduction statistics."""
        try:
            stats = {
                'original_shape': original_features.shape,
                'reduced_shape': reduced_features.shape,
                'compression_ratio': original_features.shape[1] / reduced_features.shape[1],
                'variance_retained': 1.0,
                'method': self.config.method
            }
            
            # Calculate variance retained for PCA
            if self.config.method == 'pca' and hasattr(self.model, 'explained_variance_ratio_'):
                stats['variance_retained'] = np.sum(self.model.explained_variance_ratio_)
                stats['explained_variance_ratio'] = self.model.explained_variance_ratio_.tolist()
            
            # Calculate reconstruction error for other methods
            if self.config.method != 'pca':
                try:
                    reconstructed = self.model.inverse_transform(reduced_features)
                    mse = np.mean((original_features - reconstructed) ** 2)
                    stats['reconstruction_mse'] = mse
                except:
                    stats['reconstruction_mse'] = None
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Reduction stats calculation failed: {e}")
            return {'error': str(e)}
    
    def inverse_transform(self, reduced_features: np.ndarray) -> np.ndarray:
        """Inverse transform reduced features back to original space."""
        try:
            if self.model is None:
                raise ValueError("Model not fitted. Call with fit=True first.")
            
            if hasattr(self.model, 'inverse_transform'):
                return self.model.inverse_transform(reduced_features)
            else:
                logger.warning("⚠️ Model does not support inverse transform")
                return reduced_features
                
        except Exception as e:
            logger.error(f"❌ Inverse transform failed: {e}")
            return reduced_features
    
    def get_feature_names(self) -> List[str]:
        """Get reduced feature names."""
        return self.feature_names.copy()
    
    def get_reduction_stats(self) -> Dict[str, Any]:
        """Get reduction statistics."""
        return self.reduction_stats.copy()
    
    def get_model(self) -> Any:
        """Get fitted model."""
        return self.model
    
    def get_scaler(self) -> Any:
        """Get fitted scaler."""
        return self.scaler