"""
Optimized Dimensionality Reduction for HDBSCAN Clustering

This module provides optimized dimensionality reduction using VectorBT acceleration
and intelligent algorithm selection based on data characteristics.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import umap

# Import UnifiedVectorizationManager
from src.utils.ml_common.unified_vectorization_manager import (
    UnifiedVectorizationManager, 
    VectorizationConfig,
    get_unified_vectorization_manager
)

logger = logging.getLogger(__name__)

@dataclass
class DimensionalityReductionConfig:
    """Configuration for optimized dimensionality reduction."""
    # Algorithm selection
    primary_method: str = 'pca'  # 'pca', 'umap', 'tsne', 'ica', 'lda'
    fallback_method: str = 'pca'  # Fallback if primary fails
    
    # PCA parameters
    pca_n_components: Optional[int] = None  # Auto-select if None
    pca_variance_threshold: float = 0.95  # Retain 95% variance
    
    # UMAP parameters
    umap_n_components: int = 2
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = 'euclidean'
    
    # t-SNE parameters
    tsne_n_components: int = 2
    tsne_perplexity: float = 30.0
    tsne_learning_rate: float = 200.0
    tsne_n_iter: int = 1000
    
    # ICA parameters
    ica_n_components: Optional[int] = None
    ica_max_iter: int = 200
    
    # LDA parameters (requires labels)
    lda_n_components: Optional[int] = None
    
    # Memory optimization
    memory_efficient: bool = True
    chunk_size: int = 1000
    max_memory_gb: float = 8.0
    
    # VectorBT optimization
    enable_vectorbt: bool = True
    enable_gpu: bool = False
    
    # Performance optimization
    use_approximate_methods: bool = True  # Use faster approximations
    max_samples_for_tsne: int = 1000  # Limit samples for t-SNE
    max_samples_for_umap: int = 5000  # Limit samples for UMAP

class OptimizedDimensionalityReducer:
    """
    Optimized dimensionality reducer with intelligent algorithm selection
    and VectorBT acceleration.
    """
    
    def __init__(self, config: Optional[DimensionalityReductionConfig] = None):
        """Initialize the optimized dimensionality reducer."""
        self.config = config or DimensionalityReductionConfig()
        
        # Initialize UnifiedVectorizationManager
        vectorization_config = VectorizationConfig(
            enable_vectorbt=self.config.enable_vectorbt,
            enable_gpu=self.config.enable_gpu,
            memory_efficient=self.config.memory_efficient,
            max_memory_gb=self.config.max_memory_gb,
            chunk_size=self.config.chunk_size,
            enable_parallel=True
        )
        self.vectorization_manager = get_unified_vectorization_manager(vectorization_config)
        
        # Initialize reducers
        self._initialize_reducers()
        
        # Performance tracking
        self.performance_stats = {
            'reduction_time': 0.0,
            'original_dimensions': 0,
            'reduced_dimensions': 0,
            'variance_explained': 0.0,
            'memory_usage_mb': 0.0,
            'vectorbt_usage_rate': 0.0,
            'algorithm_used': None
        }
        
        logger.info("✅ OptimizedDimensionalityReducer initialized")
    
    def _initialize_reducers(self):
        """Initialize dimensionality reduction algorithms."""
        self.reducers = {
            'pca': PCA,
            'ica': FastICA,
            'umap': umap.UMAP,
            'tsne': TSNE,
            'lda': LinearDiscriminantAnalysis,
            'isomap': Isomap,
            'lle': LocallyLinearEmbedding
        }
    
    def reduce_dimensions(self, features_df: pd.DataFrame, 
                         labels: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Reduce dimensions using optimized algorithms.
        
        Args:
            features_df: Input features DataFrame
            labels: Optional labels for supervised methods (LDA)
            
        Returns:
            Reduced features DataFrame
        """
        start_time = time.time()
        logger.info(f"🚀 Starting dimensionality reduction for {features_df.shape[1]} features")
        
        # Validate input
        self._validate_features(features_df)
        
        # Select optimal algorithm
        algorithm = self._select_optimal_algorithm(features_df, labels)
        
        # Reduce dimensions
        try:
            reduced_features = self._reduce_with_algorithm(features_df, algorithm, labels)
            logger.info(f"✅ Dimensionality reduction completed: {reduced_features.shape[1]} dimensions")
        except Exception as e:
            logger.warning(f"⚠️ Primary algorithm failed: {e}, trying fallback")
            reduced_features = self._reduce_with_algorithm(features_df, self.config.fallback_method, labels)
        
        # Update performance stats
        reduction_time = time.time() - start_time
        self._update_performance_stats(features_df, reduced_features, reduction_time, algorithm)
        
        return reduced_features
    
    def _select_optimal_algorithm(self, features_df: pd.DataFrame, 
                                 labels: Optional[pd.Series] = None) -> str:
        """Select optimal dimensionality reduction algorithm based on data characteristics."""
        n_samples, n_features = features_df.shape
        
        # Check for supervised methods
        if labels is not None and self.config.primary_method == 'lda':
            return 'lda'
        
        # Check data size for expensive methods
        if n_samples > self.config.max_samples_for_tsne and self.config.primary_method == 'tsne':
            logger.info("🔄 Large dataset detected, switching to PCA for t-SNE")
            return 'pca'
        
        if n_samples > self.config.max_samples_for_umap and self.config.primary_method == 'umap':
            logger.info("🔄 Large dataset detected, switching to PCA for UMAP")
            return 'pca'
        
        # Check for high-dimensional data
        if n_features > 1000 and self.config.primary_method in ['tsne', 'umap']:
            logger.info("🔄 High-dimensional data detected, switching to PCA")
            return 'pca'
        
        return self.config.primary_method
    
    def _reduce_with_algorithm(self, features_df: pd.DataFrame, 
                              algorithm: str, 
                              labels: Optional[pd.Series] = None) -> pd.DataFrame:
        """Reduce dimensions using specified algorithm."""
        logger.info(f"🔄 Reducing dimensions using {algorithm}")
        
        if algorithm == 'pca':
            return self._reduce_with_pca(features_df)
        elif algorithm == 'umap':
            return self._reduce_with_umap(features_df)
        elif algorithm == 'tsne':
            return self._reduce_with_tsne(features_df)
        elif algorithm == 'ica':
            return self._reduce_with_ica(features_df)
        elif algorithm == 'lda':
            return self._reduce_with_lda(features_df, labels)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def _reduce_with_pca(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Reduce dimensions using PCA with VectorBT optimization."""
        try:
            # Determine number of components
            n_components = self._determine_pca_components(features_df)
            
            # Use VectorBT PCA if available
            if hasattr(self.vectorization_manager, 'pca_reduce'):
                reduced_features = self.vectorization_manager.pca_reduce(
                    features_df, 
                    n_components=n_components
                )
            else:
                # Use sklearn PCA
                pca = PCA(n_components=n_components, random_state=42)
                reduced_array = pca.fit_transform(features_df)
                
                # Create DataFrame
                component_names = [f'PC_{i+1}' for i in range(reduced_array.shape[1])]
                reduced_features = pd.DataFrame(
                    reduced_array,
                    index=features_df.index,
                    columns=component_names
                )
            
            logger.info(f"✅ PCA reduction completed: {reduced_features.shape[1]} components")
            return reduced_features
            
        except Exception as e:
            logger.error(f"❌ PCA reduction failed: {e}")
            raise
    
    def _reduce_with_umap(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Reduce dimensions using UMAP."""
        try:
            # Sample data if too large
            if len(features_df) > self.config.max_samples_for_umap:
                logger.info(f"🔄 Sampling data for UMAP: {len(features_df)} -> {self.config.max_samples_for_umap}")
                sampled_features = features_df.sample(n=self.config.max_samples_for_umap, random_state=42)
                sample_indices = sampled_features.index
            else:
                sampled_features = features_df
                sample_indices = features_df.index
            
            # Fit UMAP
            umap_reducer = umap.UMAP(
                n_components=self.config.umap_n_components,
                n_neighbors=self.config.umap_n_neighbors,
                min_dist=self.config.umap_min_dist,
                metric=self.config.umap_metric,
                random_state=42
            )
            
            reduced_array = umap_reducer.fit_transform(sampled_features)
            
            # Create DataFrame
            component_names = [f'UMAP_{i+1}' for i in range(reduced_array.shape[1])]
            reduced_features = pd.DataFrame(
                reduced_array,
                index=sample_indices,
                columns=component_names
            )
            
            logger.info(f"✅ UMAP reduction completed: {reduced_features.shape[1]} components")
            return reduced_features
            
        except Exception as e:
            logger.error(f"❌ UMAP reduction failed: {e}")
            raise
    
    def _reduce_with_tsne(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Reduce dimensions using t-SNE."""
        try:
            # Sample data if too large
            if len(features_df) > self.config.max_samples_for_tsne:
                logger.info(f"🔄 Sampling data for t-SNE: {len(features_df)} -> {self.config.max_samples_for_tsne}")
                sampled_features = features_df.sample(n=self.config.max_samples_for_tsne, random_state=42)
                sample_indices = sampled_features.index
            else:
                sampled_features = features_df
                sample_indices = features_df.index
            
            # Fit t-SNE
            tsne = TSNE(
                n_components=self.config.tsne_n_components,
                perplexity=self.config.tsne_perplexity,
                learning_rate=self.config.tsne_learning_rate,
                n_iter=self.config.tsne_n_iter,
                random_state=42
            )
            
            reduced_array = tsne.fit_transform(sampled_features)
            
            # Create DataFrame
            component_names = [f'tSNE_{i+1}' for i in range(reduced_array.shape[1])]
            reduced_features = pd.DataFrame(
                reduced_array,
                index=sample_indices,
                columns=component_names
            )
            
            logger.info(f"✅ t-SNE reduction completed: {reduced_features.shape[1]} components")
            return reduced_features
            
        except Exception as e:
            logger.error(f"❌ t-SNE reduction failed: {e}")
            raise
    
    def _reduce_with_ica(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Reduce dimensions using ICA."""
        try:
            # Determine number of components
            n_components = self.config.ica_n_components or min(features_df.shape[1], 50)
            
            # Fit ICA
            ica = FastICA(
                n_components=n_components,
                max_iter=self.config.ica_max_iter,
                random_state=42
            )
            
            reduced_array = ica.fit_transform(features_df)
            
            # Create DataFrame
            component_names = [f'IC_{i+1}' for i in range(reduced_array.shape[1])]
            reduced_features = pd.DataFrame(
                reduced_array,
                index=features_df.index,
                columns=component_names
            )
            
            logger.info(f"✅ ICA reduction completed: {reduced_features.shape[1]} components")
            return reduced_features
            
        except Exception as e:
            logger.error(f"❌ ICA reduction failed: {e}")
            raise
    
    def _reduce_with_lda(self, features_df: pd.DataFrame, 
                        labels: pd.Series) -> pd.DataFrame:
        """Reduce dimensions using LDA (supervised)."""
        try:
            if labels is None:
                raise ValueError("Labels required for LDA")
            
            # Determine number of components
            n_components = self.config.lda_n_components or min(len(labels.unique()) - 1, features_df.shape[1])
            
            # Fit LDA
            lda = LinearDiscriminantAnalysis(n_components=n_components)
            reduced_array = lda.fit_transform(features_df, labels)
            
            # Create DataFrame
            component_names = [f'LD_{i+1}' for i in range(reduced_array.shape[1])]
            reduced_features = pd.DataFrame(
                reduced_array,
                index=features_df.index,
                columns=component_names
            )
            
            logger.info(f"✅ LDA reduction completed: {reduced_features.shape[1]} components")
            return reduced_features
            
        except Exception as e:
            logger.error(f"❌ LDA reduction failed: {e}")
            raise
    
    def _determine_pca_components(self, features_df: pd.DataFrame) -> int:
        """Determine optimal number of PCA components."""
        if self.config.pca_n_components is not None:
            return self.config.pca_n_components
        
        # Use variance threshold
        pca = PCA(random_state=42)
        pca.fit(features_df)
        
        # Find number of components that explain variance threshold
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= self.config.pca_variance_threshold) + 1
        
        # Ensure reasonable bounds
        n_components = max(2, min(n_components, features_df.shape[1]))
        
        logger.info(f"🔄 Selected {n_components} PCA components (explains {cumulative_variance[n_components-1]:.2%} variance)")
        return n_components
    
    def _validate_features(self, features_df: pd.DataFrame):
        """Validate input features."""
        if not isinstance(features_df, pd.DataFrame):
            raise ValueError("Features must be a pandas DataFrame")
        
        if features_df.empty:
            raise ValueError("Features DataFrame cannot be empty")
        
        if features_df.shape[1] < 2:
            raise ValueError("At least 2 features required for dimensionality reduction")
    
    def _update_performance_stats(self, original_features: pd.DataFrame, 
                                 reduced_features: pd.DataFrame, 
                                 reduction_time: float, 
                                 algorithm: str):
        """Update performance statistics."""
        self.performance_stats['reduction_time'] = reduction_time
        self.performance_stats['original_dimensions'] = original_features.shape[1]
        self.performance_stats['reduced_dimensions'] = reduced_features.shape[1]
        self.performance_stats['algorithm_used'] = algorithm
        
        # Calculate memory usage
        memory_usage = reduced_features.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        self.performance_stats['memory_usage_mb'] = memory_usage
        
        # Get VectorBT usage rate
        vectorization_stats = self.vectorization_manager.get_performance_stats()
        self.performance_stats['vectorbt_usage_rate'] = vectorization_stats.get('vectorbt_usage_rate', 0)
        
        # Calculate variance explained (for PCA)
        if algorithm == 'pca':
            try:
                pca = PCA(random_state=42)
                pca.fit(original_features)
                variance_explained = np.sum(pca.explained_variance_ratio_[:reduced_features.shape[1]])
                self.performance_stats['variance_explained'] = variance_explained
            except:
                self.performance_stats['variance_explained'] = 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add vectorization manager stats
        vectorization_stats = self.vectorization_manager.get_performance_stats()
        stats['vectorization_stats'] = vectorization_stats
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'reduction_time': 0.0,
            'original_dimensions': 0,
            'reduced_dimensions': 0,
            'variance_explained': 0.0,
            'memory_usage_mb': 0.0,
            'vectorbt_usage_rate': 0.0,
            'algorithm_used': None
        }
        
        # Reset vectorization manager stats
        self.vectorization_manager.reset_stats()

# Convenience function for easy usage
def create_optimized_dimensionality_reducer(
    primary_method: str = 'pca',
    fallback_method: str = 'pca',
    pca_variance_threshold: float = 0.95,
    umap_n_components: int = 2,
    tsne_n_components: int = 2,
    memory_efficient: bool = True,
    enable_vectorbt: bool = True,
    enable_gpu: bool = False
) -> OptimizedDimensionalityReducer:
    """
    Create an optimized dimensionality reducer with specified configuration.
    
    Args:
        primary_method: Primary reduction method
        fallback_method: Fallback method if primary fails
        pca_variance_threshold: Variance threshold for PCA
        umap_n_components: Number of UMAP components
        tsne_n_components: Number of t-SNE components
        memory_efficient: Enable memory optimization
        enable_vectorbt: Enable VectorBT acceleration
        enable_gpu: Enable GPU acceleration
        
    Returns:
        OptimizedDimensionalityReducer instance
    """
    config = DimensionalityReductionConfig(
        primary_method=primary_method,
        fallback_method=fallback_method,
        pca_variance_threshold=pca_variance_threshold,
        umap_n_components=umap_n_components,
        tsne_n_components=tsne_n_components,
        memory_efficient=memory_efficient,
        enable_vectorbt=enable_vectorbt,
        enable_gpu=enable_gpu
    )
    
    return OptimizedDimensionalityReducer(config)

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 100
    
    # Create high-dimensional data
    features = np.random.randn(n_samples, n_features)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    features_df = pd.DataFrame(features, columns=feature_names)
    
    print(f"Original features: {features_df.shape}")
    
    # Create optimized dimensionality reducer
    reducer = create_optimized_dimensionality_reducer(
        primary_method='pca',
        pca_variance_threshold=0.95,
        memory_efficient=True,
        enable_vectorbt=True
    )
    
    # Reduce dimensions
    reduced_features = reducer.reduce_dimensions(features_df)
    
    print(f"Reduced features: {reduced_features.shape}")
    print(f"Performance stats: {reducer.get_performance_stats()}")
