#!/usr/bin/env python3
"""
Matrix Operations Integration for HMM Clustering

This module provides specialized matrix operations integration for HMM clustering,
leveraging the unified matrix operations framework for optimal performance.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import warnings

# Core dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, f_classif

# Import common utilities
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
from src.utils.math_validation import safe_divide, safe_log, safe_sqrt, safe_correlation
from src.utils.common_operations import safe_dataframe_operation
from src.utils.logger import system_logger

# Setup logging
logger = system_logger.getChild('MatrixOperationsIntegration')

@dataclass
class MatrixOperationsConfig:
    """Configuration for matrix operations integration."""
    # Scaling options
    scaling_method: str = 'standard'  # 'standard', 'minmax', 'robust', 'none'
    feature_normalization: bool = True
    outlier_handling: str = 'clip'  # 'clip', 'remove', 'none'
    
    # Dimensionality reduction
    enable_dimensionality_reduction: bool = False
    reduction_method: str = 'pca'  # 'pca', 'ica', 'none'
    n_components: Optional[int] = None
    variance_threshold: float = 0.95
    
    # Feature selection
    enable_feature_selection: bool = False
    selection_method: str = 'kbest'  # 'kbest', 'variance', 'correlation'
    n_features: Optional[int] = None
    correlation_threshold: float = 0.95
    
    # Matrix operations
    enable_matrix_optimization: bool = True
    use_vectorized_operations: bool = True
    enable_parallel_processing: bool = True
    chunk_size: int = 1000
    
    # Memory optimization
    memory_efficient: bool = True
    dtype: str = 'float32'  # 'float32', 'float64'
    
    # Validation
    enable_validation: bool = True
    enable_profiling: bool = False

class MatrixOperationsIntegration:
    """
    Matrix operations integration for HMM clustering.
    
    This class provides comprehensive matrix operations for feature processing,
    dimensionality reduction, and optimization specifically designed for HMM clustering.
    """
    
    def __init__(self, config: MatrixOperationsConfig):
        """Initialize matrix operations integration."""
        self.config = config
        self.logger = logger.getChild('MatrixOperationsIntegration')
        
        # Initialize unified matrix operations
        self.matrix_ops = UnifiedMatrixOperations()
        
        # Initialize scalers and transformers
        self.scaler = None
        self.dimensionality_reducer = None
        self.feature_selector = None
        
        # Performance tracking
        self.performance_metrics = {}
        self.operation_times = {}
        
        # State
        self.is_fitted = False
        self.feature_names = None
        self.n_features_original = None
        self.n_features_processed = None
        
        self.logger.info("🔧 Matrix Operations Integration initialized")
        self._log_capabilities()
    
    def _log_capabilities(self):
        """Log available matrix operations capabilities."""
        self.logger.info("🔧 Matrix Operations Capabilities:")
        self.logger.info(f"   Unified Matrix Operations: {'✅ Available' if self.matrix_ops else '❌ Not Available'}")
        self.logger.info(f"   Scaling Method: {self.config.scaling_method}")
        self.logger.info(f"   Dimensionality Reduction: {'✅ Enabled' if self.config.enable_dimensionality_reduction else '❌ Disabled'}")
        self.logger.info(f"   Feature Selection: {'✅ Enabled' if self.config.enable_feature_selection else '❌ Disabled'}")
        self.logger.info(f"   Matrix Optimization: {'✅ Enabled' if self.config.enable_matrix_optimization else '❌ Disabled'}")
        self.logger.info(f"   Memory Efficient: {'✅ Enabled' if self.config.memory_efficient else '❌ Disabled'}")
    
    def _create_scaler(self) -> Any:
        """Create appropriate scaler based on configuration."""
        if self.config.scaling_method == 'standard':
            return StandardScaler()
        elif self.config.scaling_method == 'minmax':
            return MinMaxScaler()
        elif self.config.scaling_method == 'robust':
            return RobustScaler()
        else:
            return None
    
    def _create_dimensionality_reducer(self) -> Any:
        """Create dimensionality reducer based on configuration."""
        if not self.config.enable_dimensionality_reduction:
            return None
        
        if self.config.reduction_method == 'pca':
            return PCA(
                n_components=self.config.n_components,
                whiten=True
            )
        elif self.config.reduction_method == 'ica':
            return FastICA(
                n_components=self.config.n_components,
                random_state=42
            )
        else:
            return None
    
    def _create_feature_selector(self) -> Any:
        """Create feature selector based on configuration."""
        if not self.config.enable_feature_selection:
            return None
        
        if self.config.selection_method == 'kbest':
            return SelectKBest(
                score_func=f_classif,
                k=self.config.n_features
            )
        else:
            return None
    
    def _optimize_matrix_for_clustering(self, data: np.ndarray) -> np.ndarray:
        """Optimize matrix for clustering using unified operations."""
        if not self.config.enable_matrix_optimization or not self.matrix_ops:
            return data
        
        self.logger.info("🔧 Optimizing matrix for clustering...")
        
        try:
            # Use unified matrix operations for optimization
            if hasattr(self.matrix_ops, 'optimize_for_clustering'):
                optimized_data = self.matrix_ops.optimize_for_clustering(data)
            else:
                # Fallback optimization
                optimized_data = self._fallback_matrix_optimization(data)
            
            return optimized_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Matrix optimization failed: {e}")
            return data
    
    def _fallback_matrix_optimization(self, data: np.ndarray) -> np.ndarray:
        """Fallback matrix optimization when unified operations are not available."""
        # Ensure data is contiguous
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        
        # Convert to appropriate dtype
        if self.config.memory_efficient and data.dtype != np.dtype(self.config.dtype):
            data = data.astype(self.config.dtype)
        
        return data
    
    def _handle_outliers(self, data: np.ndarray) -> np.ndarray:
        """Handle outliers in the data."""
        if self.config.outlier_handling == 'none':
            return data
        
        self.logger.info(f"🔍 Handling outliers using {self.config.outlier_handling} method...")
        
        try:
            if self.config.outlier_handling == 'clip':
                # Clip outliers to 3 standard deviations
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                
                data = np.clip(data, lower_bound, upper_bound)
                
            elif self.config.outlier_handling == 'remove':
                # Remove rows with outliers
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                z_scores = np.abs((data - mean) / std)
                outlier_mask = np.any(z_scores > 3, axis=1)
                data = data[~outlier_mask]
                
                self.logger.info(f"   Removed {np.sum(outlier_mask)} outlier rows")
            
            return data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Outlier handling failed: {e}")
            return data
    
    def _apply_dimensionality_reduction(self, data: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction if enabled."""
        if not self.config.enable_dimensionality_reduction or not self.dimensionality_reducer:
            return data
        
        self.logger.info(f"📉 Applying {self.config.reduction_method} dimensionality reduction...")
        
        try:
            start_time = time.time()
            
            # Fit and transform data
            reduced_data = self.dimensionality_reducer.fit_transform(data)
            
            # Log explained variance if PCA
            if self.config.reduction_method == 'pca' and hasattr(self.dimensionality_reducer, 'explained_variance_ratio_'):
                explained_variance = np.sum(self.dimensionality_reducer.explained_variance_ratio_)
                self.logger.info(f"   Explained variance: {explained_variance:.3f}")
            
            operation_time = time.time() - start_time
            self.operation_times['dimensionality_reduction'] = operation_time
            
            self.logger.info(f"   Reduced from {data.shape[1]} to {reduced_data.shape[1]} features in {operation_time:.3f}s")
            
            return reduced_data
            
        except Exception as e:
            self.logger.error(f"❌ Dimensionality reduction failed: {e}")
            return data
    
    def _apply_feature_selection(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply feature selection if enabled."""
        if not self.config.enable_feature_selection or not self.feature_selector:
            return data
        
        self.logger.info(f"🎯 Applying {self.config.selection_method} feature selection...")
        
        try:
            start_time = time.time()
            
            # For unsupervised feature selection, we need to create pseudo-labels
            if labels is None:
                # Use clustering to create pseudo-labels
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=min(5, data.shape[0] // 10), random_state=42)
                labels = kmeans.fit_predict(data)
            
            # Fit and transform data
            selected_data = self.feature_selector.fit_transform(data, labels)
            
            operation_time = time.time() - start_time
            self.operation_times['feature_selection'] = operation_time
            
            self.logger.info(f"   Selected {selected_data.shape[1]} features from {data.shape[1]} in {operation_time:.3f}s")
            
            return selected_data
            
        except Exception as e:
            self.logger.error(f"❌ Feature selection failed: {e}")
            return data
    
    def _apply_correlation_filtering(self, data: np.ndarray) -> np.ndarray:
        """Apply correlation-based feature filtering."""
        if self.config.selection_method != 'correlation':
            return data
        
        self.logger.info("🔗 Applying correlation-based feature filtering...")
        
        try:
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(data.T)
            
            # Find highly correlated features
            high_corr_pairs = []
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    if abs(corr_matrix[i, j]) > self.config.correlation_threshold:
                        high_corr_pairs.append((i, j))
            
            # Remove one feature from each highly correlated pair
            features_to_remove = set()
            for i, j in high_corr_pairs:
                # Keep the feature with higher variance
                var_i = np.var(data[:, i])
                var_j = np.var(data[:, j])
                if var_i < var_j:
                    features_to_remove.add(i)
                else:
                    features_to_remove.add(j)
            
            # Remove selected features
            features_to_keep = [i for i in range(data.shape[1]) if i not in features_to_remove]
            filtered_data = data[:, features_to_keep]
            
            self.logger.info(f"   Removed {len(features_to_remove)} highly correlated features")
            
            return filtered_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Correlation filtering failed: {e}")
            return data
    
    def _calculate_feature_importance(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate feature importance scores."""
        try:
            if labels is None:
                # Use variance as importance for unsupervised case
                importance = np.var(data, axis=0)
            else:
                # Use F-score for supervised case
                from sklearn.feature_selection import f_classif
                f_scores, _ = f_classif(data, labels)
                importance = f_scores
            
            return importance
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature importance calculation failed: {e}")
            return np.ones(data.shape[1])
    
    def fit_transform(self, data: Union[pd.DataFrame, np.ndarray], 
                     labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fit and transform data using matrix operations.
        
        Args:
            data: Input data (DataFrame or numpy array)
            labels: Optional labels for supervised feature selection
            
        Returns:
            Tuple of (transformed_data, metadata)
        """
        self.logger.info("🔧 Starting matrix operations fit and transform...")
        
        start_time = time.time()
        
        try:
            # Convert to numpy array if needed
            if isinstance(data, pd.DataFrame):
                self.feature_names = data.columns.tolist()
                data_array = data.values
            else:
                data_array = np.array(data)
            
            # Store original dimensions
            self.n_features_original = data_array.shape[1]
            
            # Validate data
            if self.config.enable_validation:
                self._validate_data(data_array)
            
            # Handle outliers
            data_array = self._handle_outliers(data_array)
            
            # Optimize matrix for clustering
            data_array = self._optimize_matrix_for_clustering(data_array)
            
            # Create and fit scaler
            if self.config.scaling_method != 'none':
                self.scaler = self._create_scaler()
                if self.scaler:
                    self.logger.info(f"📏 Scaling data using {self.config.scaling_method} scaler...")
                    scaler_start = time.time()
                    data_array = self.scaler.fit_transform(data_array)
                    self.operation_times['scaling'] = time.time() - scaler_start
            
            # Apply feature normalization
            if self.config.feature_normalization:
                self.logger.info("🔧 Applying feature normalization...")
                norm_start = time.time()
                data_array = self._normalize_features(data_array)
                self.operation_times['normalization'] = time.time() - norm_start
            
            # Apply correlation filtering
            data_array = self._apply_correlation_filtering(data_array)
            
            # Apply dimensionality reduction
            self.dimensionality_reducer = self._create_dimensionality_reducer()
            data_array = self._apply_dimensionality_reduction(data_array)
            
            # Apply feature selection
            self.feature_selector = self._create_feature_selector()
            data_array = self._apply_feature_selection(data_array, labels)
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(data_array, labels)
            
            # Store processed dimensions
            self.n_features_processed = data_array.shape[1]
            
            # Create metadata
            metadata = {
                'n_features_original': self.n_features_original,
                'n_features_processed': self.n_features_processed,
                'feature_importance': feature_importance,
                'operation_times': self.operation_times,
                'scaler_type': type(self.scaler).__name__ if self.scaler else None,
                'dimensionality_reducer_type': type(self.dimensionality_reducer).__name__ if self.dimensionality_reducer else None,
                'feature_selector_type': type(self.feature_selector).__name__ if self.feature_selector else None,
                'total_processing_time': time.time() - start_time
            }
            
            # Update state
            self.is_fitted = True
            
            self.logger.info(f"✅ Matrix operations completed in {metadata['total_processing_time']:.3f}s")
            self.logger.info(f"   Features: {self.n_features_original} → {self.n_features_processed}")
            
            return data_array, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Matrix operations failed: {e}")
            raise
    
    def _normalize_features(self, data: np.ndarray) -> np.ndarray:
        """Normalize features to unit norm."""
        try:
            # Calculate L2 norms
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            
            # Avoid division by zero
            norms = np.where(norms == 0, 1, norms)
            
            # Normalize
            normalized_data = data / norms
            
            return normalized_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature normalization failed: {e}")
            return data
    
    def _validate_data(self, data: np.ndarray):
        """Validate input data."""
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got {data.ndim}D")
        
        if data.shape[0] == 0:
            raise ValueError("Data is empty")
        
        if data.shape[1] == 0:
            raise ValueError("Data has no features")
        
        if not np.all(np.isfinite(data)):
            raise ValueError("Data contains non-finite values")
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform new data using fitted transformers."""
        if not self.is_fitted:
            raise ValueError("Must call fit_transform first")
        
        self.logger.info("🔄 Transforming new data...")
        
        try:
            # Convert to numpy array if needed
            if isinstance(data, pd.DataFrame):
                data_array = data.values
            else:
                data_array = np.array(data)
            
            # Apply scaling
            if self.scaler:
                data_array = self.scaler.transform(data_array)
            
            # Apply feature normalization
            if self.config.feature_normalization:
                data_array = self._normalize_features(data_array)
            
            # Apply dimensionality reduction
            if self.dimensionality_reducer:
                data_array = self.dimensionality_reducer.transform(data_array)
            
            # Apply feature selection
            if self.feature_selector:
                data_array = self.feature_selector.transform(data_array)
            
            return data_array
            
        except Exception as e:
            self.logger.error(f"❌ Data transformation failed: {e}")
            raise
    
    def get_feature_names(self) -> Optional[List[str]]:
        """Get feature names if available."""
        return self.feature_names
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        if hasattr(self.feature_selector, 'scores_'):
            return self.feature_selector.scores_
        return None
    
    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Get explained variance ratio if using PCA."""
        if (self.dimensionality_reducer and 
            hasattr(self.dimensionality_reducer, 'explained_variance_ratio_')):
            return self.dimensionality_reducer.explained_variance_ratio_
        return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'is_fitted': self.is_fitted,
            'n_features_original': self.n_features_original,
            'n_features_processed': self.n_features_processed,
            'feature_reduction_ratio': safe_divide(
                self.n_features_processed, 
                self.n_features_original, 
                1.0
            ) if self.n_features_original else 1.0,
            'operation_times': self.operation_times,
            'config': self.config.__dict__
        }


def create_matrix_operations_integration(config: Optional[MatrixOperationsConfig] = None) -> MatrixOperationsIntegration:
    """Factory function to create matrix operations integration instance."""
    if config is None:
        config = MatrixOperationsConfig()
    
    return MatrixOperationsIntegration(config)


# Example usage
if __name__ == "__main__":
    # Example usage
    logger.info("🔧 Matrix Operations Integration Example")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Generate sample data with some correlation
    data = np.random.randn(n_samples, n_features)
    
    # Add some correlation between features
    data[:, 5] = data[:, 0] + 0.1 * np.random.randn(n_samples)
    data[:, 10] = data[:, 1] + 0.1 * np.random.randn(n_samples)
    
    # Create configuration
    config = MatrixOperationsConfig(
        scaling_method='standard',
        enable_dimensionality_reduction=True,
        reduction_method='pca',
        n_components=10,
        enable_feature_selection=True,
        selection_method='kbest',
        n_features=15,
        enable_matrix_optimization=True,
        memory_efficient=True
    )
    
    # Create and use matrix operations integration
    matrix_ops = create_matrix_operations_integration(config)
    transformed_data, metadata = matrix_ops.fit_transform(data)
    
    # Print results
    print(f"Original features: {metadata['n_features_original']}")
    print(f"Processed features: {metadata['n_features_processed']}")
    print(f"Feature reduction ratio: {metadata['feature_reduction_ratio']:.3f}")
    print(f"Total processing time: {metadata['total_processing_time']:.3f}s")
    
    # Get performance summary
    summary = matrix_ops.get_performance_summary()
    print(f"Performance summary: {summary}")