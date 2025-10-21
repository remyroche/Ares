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

# Import enhanced hardware optimization tools
from src.utils.hardware import (
    smart_cache, auto_optimize, memory_efficient, performance_tracked,
    optimize_dataframe_default, optimize_numpy_array_default
)

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_progress, tprint_timer,
    tprint_logged, tprint_data_format, LogLevel
)

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
    
    # Regime-aware reduction
    enable_regime_aware_reduction: bool = True
    regime_detection_method: str = 'variance'  # 'variance', 'entropy', 'volatility'
    regime_window: int = 20
    regime_threshold: float = 0.1
    regime_specific_components: bool = True
    regime_adaptive_components: bool = True

class DimensionalityReducer:
    """
    Comprehensive dimensionality reducer for HDBSCAN regime discovery.
    
    Supports multiple dimensionality reduction techniques including PCA, UMAP,
    t-SNE, ICA, and other advanced methods.
    """
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def __init__(self, config: Optional[DimensionalityReducerConfig] = None):
        """
        Initialize dimensionality reducer.
        
        Args:
            config: Configuration for dimensionality reduction
        """
        tprint_info("Initializing DimensionalityReducer")
        
        self.config = config or DimensionalityReducerConfig()
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.reduction_stats = {}
        
        tprint_debug(f"Config: method={self.config.method}, n_components={self.config.n_components}")
        tprint_success("✅ DimensionalityReducer initialized")
        
    @tprint_logged(LogLevel.INFO, include_args=True)
    @smart_cache(ttl=3600)  # Cache dimensionality reduction results for 1 hour
    @auto_optimize(optimize_inputs=True, optimize_outputs=True)
    @memory_efficient(memory_threshold_mb=150.0, auto_cleanup=True)
    @performance_tracked(log_performance=True, track_memory=True)
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
            tprint_info(f"📉 Starting dimensionality reduction using {self.config.method}...")
            tprint_debug(f"Input shape: {features.shape}, fit={fit}, target_shape={target.shape if target is not None else 'None'}")
            
            # Data format validation for input features
            tprint_data_format(features, "input_features", LogLevel.DEBUG)
            if target is not None:
                tprint_data_format(target, "input_target", LogLevel.DEBUG)
            
            # Validate input
            if self.config.validate_input:
                with tprint_timer("Input validation"):
                    features = self._validate_input(features)
                    tprint_debug(f"After validation: {features.shape}")
                    # Data format validation after input validation
                    tprint_data_format(features, "validated_features", LogLevel.DEBUG)
            else:
                tprint_debug("Input validation skipped")
            
            # Preprocess features
            with tprint_timer("Feature preprocessing"):
                if fit:
                    features = self._preprocess_features(features, fit=True)
                    tprint_debug(f"After preprocessing (fit): {features.shape}")
                else:
                    features = self._preprocess_features(features, fit=False)
                    tprint_debug(f"After preprocessing (transform): {features.shape}")
            
            # Determine number of components
            with tprint_timer("Component determination"):
                n_components = self._determine_n_components(features)
                tprint_debug(f"Determined n_components: {n_components}")
            
            # Apply dimensionality reduction
            if fit:
                with tprint_timer(f"Dimensionality reduction fitting ({self.config.method})"):
                    if self.config.enable_regime_aware_reduction:
                        reduced_features, model = self._fit_regime_aware_reduction(features, n_components, target)
                    else:
                        reduced_features, model = self._fit_reduction(features, n_components, target)
                    self.model = model
                    tprint_debug(f"After fitting: {reduced_features.shape}")
                    # Data format validation after fitting
                    tprint_data_format(reduced_features, "fitted_reduced_features", LogLevel.DEBUG)
            else:
                with tprint_timer(f"Dimensionality reduction transform ({self.config.method})"):
                    if self.model is None:
                        raise ValueError("Model not fitted. Call with fit=True first.")
                    if self.config.enable_regime_aware_reduction:
                        reduced_features = self._transform_regime_aware_features(features)
                    else:
                        reduced_features = self._transform_features(features)
                    tprint_debug(f"After transform: {reduced_features.shape}")
                    # Data format validation after transform
                    tprint_data_format(reduced_features, "transformed_reduced_features", LogLevel.DEBUG)
            
            # Calculate reduction statistics
            with tprint_timer("Reduction statistics calculation"):
                reduction_info = self._calculate_reduction_stats(features, reduced_features)
                self.reduction_stats = reduction_info
                tprint_debug(f"Reduction info: {reduction_info}")
                # Data format validation for reduction info
                tprint_data_format(reduction_info, "reduction_info", LogLevel.DEBUG)
            
            # Data format validation for final output
            tprint_data_format(reduced_features, "final_reduced_features", LogLevel.DEBUG)
            
            tprint_success(f"✅ Dimensionality reduction completed. Shape: {features.shape} -> {reduced_features.shape}")
            
            return reduced_features, reduction_info
            
        except Exception as e:
            tprint_error(f"❌ Dimensionality reduction failed: {e}")
            # Return original features as fallback
            return features, {'error': str(e)}
    
    @tprint_logged(LogLevel.DEBUG, include_args=True)
    def _validate_input(self, features: np.ndarray) -> np.ndarray:
        """Validate input features."""
        try:
            tprint_debug(f"Validating input features with shape: {features.shape}")
            
            # Check for NaN values
            if np.isnan(features).any():
                tprint_warning("⚠️ Found NaN values, filling with 0")
                features = np.nan_to_num(features, nan=0.0)
                tprint_debug(f"After NaN handling: {features.shape}")
            
            # Check for infinite values
            if np.isinf(features).any():
                tprint_warning("⚠️ Found infinite values, clipping")
                features = np.clip(features, -1e10, 1e10)
                tprint_debug(f"After infinite handling: {features.shape}")
            
            # Check minimum samples
            if len(features) < self.config.min_samples:
                raise ValueError(f"Insufficient samples: {len(features)} < {self.config.min_samples}")
            
            # Check for constant features
            feature_vars = np.var(features, axis=0)
            constant_features = feature_vars < 1e-10
            if constant_features.any():
                tprint_warning(f"⚠️ Found {constant_features.sum()} constant features, removing them")
                features = features[:, ~constant_features]
                tprint_debug(f"After constant feature removal: {features.shape}")
            
            tprint_debug(f"Input validation completed. Final shape: {features.shape}")
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
    
    def _fit_regime_aware_reduction(self, features: np.ndarray, n_components: int, target: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Any]:
        """Fit regime-aware dimensionality reduction."""
        try:
            # Detect regimes
            regimes = self._detect_regimes(features)
            
            if regimes is not None and len(np.unique(regimes)) > 1:
                # Apply regime-aware reduction
                reduced_features, model = self._apply_regime_aware_reduction(features, n_components, target, regimes)
            else:
                # Fall back to standard reduction
                reduced_features, model = self._fit_reduction(features, n_components, target)
            
            return reduced_features, model
            
        except Exception as e:
            logger.error(f"❌ Regime-aware reduction fitting failed: {e}")
            return self._fit_reduction(features, n_components, target)
    
    def _transform_regime_aware_features(self, features: np.ndarray) -> np.ndarray:
        """Transform features using regime-aware model."""
        try:
            if self.model is None:
                raise ValueError("Model not fitted. Call with fit=True first.")
            
            # Detect regimes
            regimes = self._detect_regimes(features)
            
            if regimes is not None and len(np.unique(regimes)) > 1:
                # Apply regime-aware transformation
                return self._apply_regime_aware_transformation(features, regimes)
            else:
                # Fall back to standard transformation
                return self._transform_features(features)
            
        except Exception as e:
            logger.error(f"❌ Regime-aware transformation failed: {e}")
            return self._transform_features(features)
    
    def _detect_regimes(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Detect regimes in the feature data."""
        try:
            # Use first feature for regime detection
            primary_feature = features[:, 0]
            
            if self.config.regime_detection_method == 'variance':
                regimes = self._detect_regimes_by_variance(primary_feature)
            elif self.config.regime_detection_method == 'entropy':
                regimes = self._detect_regimes_by_entropy(primary_feature)
            elif self.config.regime_detection_method == 'volatility':
                regimes = self._detect_regimes_by_volatility(primary_feature)
            else:
                regimes = self._detect_regimes_by_variance(primary_feature)
            
            return regimes
            
        except Exception as e:
            logger.error(f"❌ Regime detection failed: {e}")
            return None
    
    def _detect_regimes_by_variance(self, feature: np.ndarray) -> np.ndarray:
        """Detect regimes based on variance changes."""
        try:
            window = self.config.regime_window
            threshold = self.config.regime_threshold
            
            regimes = np.zeros(len(feature))
            rolling_var = pd.Series(feature).rolling(window=window).var().values
            
            # Find variance change points
            var_changes = np.abs(np.diff(rolling_var)) > (threshold * np.nanmean(rolling_var))
            change_points = np.where(var_changes)[0]
            
            # Assign regime labels
            current_regime = 0
            for i in range(len(regimes)):
                if i in change_points:
                    current_regime = (current_regime + 1) % 3
                regimes[i] = current_regime
            
            return regimes
            
        except Exception as e:
            logger.debug(f"Variance-based regime detection failed: {e}")
            return np.zeros(len(feature))
    
    def _detect_regimes_by_entropy(self, feature: np.ndarray) -> np.ndarray:
        """Detect regimes based on entropy changes."""
        try:
            window = self.config.regime_window
            threshold = self.config.regime_threshold
            
            regimes = np.zeros(len(feature))
            
            # Calculate rolling entropy
            rolling_entropy = []
            for i in range(window, len(feature)):
                window_data = feature[i-window:i]
                # Discretize data
                hist, _ = np.histogram(window_data, bins=10)
                hist = hist / hist.sum()
                hist = hist[hist > 0]
                entropy = -np.sum(hist * np.log2(hist))
                rolling_entropy.append(entropy)
            
            rolling_entropy = np.array(rolling_entropy)
            
            # Find entropy change points
            entropy_changes = np.abs(np.diff(rolling_entropy)) > (threshold * np.std(rolling_entropy))
            change_points = np.where(entropy_changes)[0] + window
            
            # Assign regime labels
            current_regime = 0
            for i in range(len(regimes)):
                if i in change_points:
                    current_regime = (current_regime + 1) % 3
                regimes[i] = current_regime
            
            return regimes
            
        except Exception as e:
            logger.debug(f"Entropy-based regime detection failed: {e}")
            return np.zeros(len(feature))
    
    def _detect_regimes_by_volatility(self, feature: np.ndarray) -> np.ndarray:
        """Detect regimes based on volatility changes."""
        try:
            window = self.config.regime_window
            threshold = self.config.regime_threshold
            
            regimes = np.zeros(len(feature))
            rolling_vol = pd.Series(feature).rolling(window=window).std().values
            
            # Find volatility change points
            vol_changes = np.abs(np.diff(rolling_vol)) > (threshold * np.nanmean(rolling_vol))
            change_points = np.where(vol_changes)[0]
            
            # Assign regime labels
            current_regime = 0
            for i in range(len(regimes)):
                if i in change_points:
                    current_regime = (current_regime + 1) % 3
                regimes[i] = current_regime
            
            return regimes
            
        except Exception as e:
            logger.debug(f"Volatility-based regime detection failed: {e}")
            return np.zeros(len(feature))
    
    def _apply_regime_aware_reduction(self, features: np.ndarray, n_components: int, 
                                    target: Optional[np.ndarray], regimes: np.ndarray) -> Tuple[np.ndarray, Any]:
        """Apply regime-aware dimensionality reduction."""
        try:
            unique_regimes = np.unique(regimes)
            n_regimes = len(unique_regimes)
            
            if n_regimes < 2:
                # Not enough regimes, use standard reduction
                return self._fit_reduction(features, n_components, target)
            
            # Calculate regime-specific components
            if self.config.regime_specific_components:
                regime_components = self._calculate_regime_specific_components(features, regimes, n_components)
            else:
                regime_components = n_components
            
            # Apply reduction for each regime
            reduced_features = np.zeros((features.shape[0], n_components))
            regime_models = {}
            
            for regime in unique_regimes:
                regime_mask = regimes == regime
                regime_features = features[regime_mask]
                
                if len(regime_features) > 1:
                    # Determine components for this regime
                    if self.config.regime_adaptive_components:
                        regime_n_components = min(regime_components[regime], len(regime_features[0]))
                    else:
                        regime_n_components = min(n_components, len(regime_features[0]))
                    
                    # Apply reduction
                    regime_reduced, regime_model = self._fit_reduction(regime_features, regime_n_components, target)
                    regime_models[regime] = regime_model
                    
                    # Store results
                    if regime_reduced.shape[1] == n_components:
                        reduced_features[regime_mask] = regime_reduced
                    else:
                        # Pad or truncate to match n_components
                        if regime_reduced.shape[1] < n_components:
                            padded = np.zeros((regime_reduced.shape[0], n_components))
                            padded[:, :regime_reduced.shape[1]] = regime_reduced
                            reduced_features[regime_mask] = padded
                        else:
                            reduced_features[regime_mask] = regime_reduced[:, :n_components]
            
            # Store regime models
            self.regime_models = regime_models
            self.regimes = regimes
            
            # Create a combined model for transformation
            combined_model = self._create_combined_model(regime_models, regimes)
            
            return reduced_features, combined_model
            
        except Exception as e:
            logger.error(f"❌ Regime-aware reduction application failed: {e}")
            return self._fit_reduction(features, n_components, target)
    
    def _apply_regime_aware_transformation(self, features: np.ndarray, regimes: np.ndarray) -> np.ndarray:
        """Apply regime-aware transformation."""
        try:
            if not hasattr(self, 'regime_models') or self.regime_models is None:
                return self._transform_features(features)
            
            unique_regimes = np.unique(regimes)
            reduced_features = np.zeros((features.shape[0], self.config.n_components))
            
            for regime in unique_regimes:
                regime_mask = regimes == regime
                regime_features = features[regime_mask]
                
                if len(regime_features) > 0 and regime in self.regime_models:
                    regime_model = self.regime_models[regime]
                    
                    if hasattr(regime_model, 'transform'):
                        regime_reduced = regime_model.transform(regime_features)
                        
                        # Ensure correct dimensions
                        if regime_reduced.shape[1] == self.config.n_components:
                            reduced_features[regime_mask] = regime_reduced
                        elif regime_reduced.shape[1] < self.config.n_components:
                            padded = np.zeros((regime_reduced.shape[0], self.config.n_components))
                            padded[:, :regime_reduced.shape[1]] = regime_reduced
                            reduced_features[regime_mask] = padded
                        else:
                            reduced_features[regime_mask] = regime_reduced[:, :self.config.n_components]
            
            return reduced_features
            
        except Exception as e:
            logger.error(f"❌ Regime-aware transformation failed: {e}")
            return self._transform_features(features)
    
    def _calculate_regime_specific_components(self, features: np.ndarray, regimes: np.ndarray, n_components: int) -> Dict[int, int]:
        """Calculate regime-specific number of components."""
        try:
            unique_regimes = np.unique(regimes)
            regime_components = {}
            
            for regime in unique_regimes:
                regime_mask = regimes == regime
                regime_features = features[regime_mask]
                
                if len(regime_features) > 1:
                    # Calculate explained variance for this regime
                    if self.config.method == 'pca':
                        from sklearn.decomposition import PCA
                        pca = PCA()
                        pca.fit(regime_features)
                        explained_variance = pca.explained_variance_ratio_
                        
                        # Find number of components that explain 95% of variance
                        cumsum = np.cumsum(explained_variance)
                        n_comp = np.argmax(cumsum >= 0.95) + 1
                        regime_components[regime] = min(n_comp, n_components)
                    else:
                        # Default to n_components
                        regime_components[regime] = n_components
                else:
                    regime_components[regime] = 1
            
            return regime_components
            
        except Exception as e:
            logger.error(f"❌ Regime-specific components calculation failed: {e}")
            return {regime: n_components for regime in unique_regimes}
    
    def _create_combined_model(self, regime_models: Dict[int, Any], regimes: np.ndarray) -> Any:
        """Create a combined model for regime-aware transformation."""
        try:
            # Create a simple wrapper that delegates to regime-specific models
            class CombinedModel:
                def __init__(self, regime_models, regimes):
                    self.regime_models = regime_models
                    self.regimes = regimes
                
                def transform(self, features):
                    return self._transform(features)
                
                def _transform(self, features):
                    # This would need to be implemented based on the specific use case
                    # For now, return the first regime model's transform
                    if self.regime_models:
                        first_model = list(self.regime_models.values())[0]
                        if hasattr(first_model, 'transform'):
                            return first_model.transform(features)
                    return features
            
            return CombinedModel(regime_models, regimes)
            
        except Exception as e:
            logger.error(f"❌ Combined model creation failed: {e}")
            return None