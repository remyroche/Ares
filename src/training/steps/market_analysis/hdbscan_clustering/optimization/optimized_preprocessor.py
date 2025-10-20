"""
Optimized Preprocessing Pipeline for HDBSCAN Clustering

This module provides optimized preprocessing that avoids O(n²) complexity
using VectorBT acceleration and intelligent sampling strategies.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
import gc
import psutil
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Import UnifiedVectorizationManager
from src.utils.ml_common.unified_vectorization_manager import (
    UnifiedVectorizationManager, 
    VectorizationConfig,
    get_unified_vectorization_manager
)

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_progress, tprint_timer,
    tprint_logged, LogLevel
)
from src.utils.common_operations import optimize_dataframe_memory, get_memory_usage

logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration for optimized preprocessing."""
    # Winsorization
    winsorize_limits: Tuple[float, float] = (0.05, 0.05)
    enable_winsorization: bool = True
    
    # Correlation pruning
    correlation_threshold: float = 0.95
    enable_correlation_pruning: bool = True
    use_approximate_correlation: bool = True  # Use sampling to avoid O(n²)
    
    # Mutual information pruning
    mi_threshold: float = 0.01
    enable_mi_pruning: bool = True
    mi_sample_size: int = 1000  # Sample size for MI calculation
    
    # HSIC pruning
    hsic_threshold: float = 0.01
    enable_hsic_pruning: bool = True
    hsic_sample_size: int = 1000  # Sample size for HSIC calculation
    
    # Scaling
    scaling_method: str = 'robust'  # 'standard', 'robust', 'minmax'
    enable_scaling: bool = True
    
    # Memory optimization
    memory_efficient: bool = True
    chunk_size: int = 1000
    max_memory_gb: float = 8.0
    
    # VectorBT optimization
    enable_vectorbt: bool = True
    enable_gpu: bool = False

class OptimizedPreprocessor:
    """
    Optimized preprocessor that avoids O(n²) complexity using intelligent sampling
    and VectorBT acceleration.
    """
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize the optimized preprocessor."""
        start_time = time.perf_counter()
        initial_memory = get_memory_usage()
        
        self.config = config or PreprocessingConfig()
        
        tprint_info("Initializing OptimizedPreprocessor")
        tprint_debug(f"Config: correlation_threshold={self.config.correlation_threshold}, scaling_method={self.config.scaling_method}")
        
        # Initialize UnifiedVectorizationManager
        with tprint_timer("Vectorization manager initialization"):
            vectorization_config = VectorizationConfig(
                enable_vectorbt=self.config.enable_vectorbt,
                enable_gpu=self.config.enable_gpu,
                memory_efficient=self.config.memory_efficient,
                max_memory_gb=self.config.max_memory_gb,
                chunk_size=self.config.chunk_size,
                enable_parallel=True
            )
            self.vectorization_manager = get_unified_vectorization_manager(vectorization_config)
            tprint_debug(f"Vectorization manager initialized: vectorbt={self.config.enable_vectorbt}, gpu={self.config.enable_gpu}")
        
        # Initialize scalers
        with tprint_timer("Scalers initialization"):
            self._initialize_scalers()
            tprint_debug("Scalers initialized successfully")
        
        # Performance tracking
        self.performance_stats = {
            'preprocessing_time': 0.0,
            'features_removed': 0,
            'memory_usage_mb': 0.0,
            'vectorbt_usage_rate': 0.0,
            'sampling_efficiency': 0.0,
            'initialization_time': 0.0,
            'initial_memory_mb': initial_memory
        }
        
        # Track initialization performance
        init_time = time.perf_counter() - start_time
        final_memory = get_memory_usage()
        self.performance_stats['initialization_time'] = init_time
        self.performance_stats['memory_usage_mb'] = final_memory
        
        tprint_success("✅ OptimizedPreprocessor initialized")
        tprint_performance("Preprocessor initialization", init_time)
        tprint_debug(f"Memory usage: {initial_memory:.2f}MB -> {final_memory:.2f}MB (delta: {final_memory - initial_memory:+.2f}MB)")
        
        logger.info("✅ OptimizedPreprocessor initialized")
    
    def _initialize_scalers(self):
        """Initialize scaling methods."""
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': None  # Will use VectorBT minmax scaling
        }
    
    def preprocess_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features with optimized algorithms to avoid O(n²) complexity.
        
        Args:
            features_df: Input features DataFrame
            
        Returns:
            Preprocessed features DataFrame
        """
        start_time = time.time()
        logger.info(f"🚀 Starting optimized preprocessing for {features_df.shape[1]} features")
        
        # Validate input
        self._validate_features(features_df)
        
        # Step 1: Winsorization (if enabled)
        if self.config.enable_winsorization:
            features_df = self._winsorize_features(features_df)
        
        # Step 2: Correlation pruning with sampling
        if self.config.enable_correlation_pruning:
            features_df = self._prune_correlated_features(features_df)
        
        # Step 3: Mutual information pruning with sampling
        if self.config.enable_mi_pruning:
            features_df = self._prune_low_mi_features(features_df)
        
        # Step 4: HSIC pruning with sampling
        if self.config.enable_hsic_pruning:
            features_df = self._prune_low_hsic_features(features_df)
        
        # Step 5: Scaling
        if self.config.enable_scaling:
            features_df = self._scale_features(features_df)
        
        # Update performance stats
        preprocessing_time = time.time() - start_time
        self._update_performance_stats(features_df, preprocessing_time)
        
        logger.info(f"✅ Preprocessing completed: {features_df.shape[1]} features in {preprocessing_time:.2f}s")
        return features_df
    
    def _winsorize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Winsorize features using VectorBT optimization."""
        logger.info("🔄 Winsorizing features")
        
        try:
            # Use VectorBT winsorization if available
            if hasattr(self.vectorization_manager, 'winsorize_data'):
                winsorized_df = self.vectorization_manager.winsorize_data(
                    features_df, 
                    limits=self.config.winsorize_limits
                )
            else:
                # Fallback to manual winsorization
                winsorized_df = features_df.copy()
                for col in winsorized_df.columns:
                    lower_limit = winsorized_df[col].quantile(self.config.winsorize_limits[0])
                    upper_limit = winsorized_df[col].quantile(1 - self.config.winsorize_limits[1])
                    winsorized_df[col] = winsorized_df[col].clip(lower_limit, upper_limit)
            
            logger.info("✅ Winsorization completed")
            return winsorized_df
            
        except Exception as e:
            logger.error(f"❌ Winsorization failed: {e}")
            return features_df
    
    def _prune_correlated_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Prune highly correlated features using sampling to avoid O(n²) complexity."""
        logger.info("🔄 Pruning correlated features with sampling")
        
        if features_df.shape[1] <= 1:
            return features_df
        
        try:
            if self.config.use_approximate_correlation:
                # Use sampling-based correlation pruning
                features_to_remove = self._find_correlated_features_sampling(features_df)
            else:
                # Use full correlation matrix (O(n²) but more accurate)
                features_to_remove = self._find_correlated_features_full(features_df)
            
            if features_to_remove:
                logger.info(f"🗑️ Removing {len(features_to_remove)} correlated features")
                features_df = features_df.drop(columns=features_to_remove)
            
            logger.info("✅ Correlation pruning completed")
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Correlation pruning failed: {e}")
            return features_df
    
    def _find_correlated_features_sampling(self, features_df: pd.DataFrame) -> List[str]:
        """Find correlated features using sampling to avoid O(n²) complexity."""
        n_features = features_df.shape[1]
        n_samples = min(1000, len(features_df))  # Sample size
        
        # Sample data for correlation calculation
        if len(features_df) > n_samples:
            sampled_data = features_df.sample(n=n_samples, random_state=42)
        else:
            sampled_data = features_df
        
        # Calculate correlation matrix on sampled data
        corr_matrix = sampled_data.corr().abs()
        
        # Find highly correlated pairs
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = (upper_tri > self.config.correlation_threshold).any()
        
        # Return features to remove
        return high_corr_pairs[high_corr_pairs].index.tolist()
    
    def _find_correlated_features_full(self, features_df: pd.DataFrame) -> List[str]:
        """Find correlated features using full correlation matrix."""
        corr_matrix = features_df.corr().abs()
        
        # Find highly correlated pairs
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = (upper_tri > self.config.correlation_threshold).any()
        
        return high_corr_pairs[high_corr_pairs].index.tolist()
    
    def _prune_low_mi_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Prune features with low mutual information using sampling."""
        logger.info("🔄 Pruning low mutual information features with sampling")
        
        if features_df.shape[1] <= 1:
            return features_df
        
        try:
            # Sample data for MI calculation
            n_samples = min(self.config.mi_sample_size, len(features_df))
            if len(features_df) > n_samples:
                sampled_data = features_df.sample(n=n_samples, random_state=42)
            else:
                sampled_data = features_df
            
            # Calculate MI for each feature (using a dummy target)
            # In practice, you might want to use a more sophisticated target
            target = sampled_data.iloc[:, 0]  # Use first feature as target proxy
            
            mi_scores = []
            for col in sampled_data.columns:
                try:
                    mi_score = mutual_info_regression(
                        sampled_data[[col]], 
                        target, 
                        random_state=42
                    )[0]
                    mi_scores.append(mi_score)
                except:
                    mi_scores.append(0.0)
            
            # Find features with low MI
            mi_series = pd.Series(mi_scores, index=sampled_data.columns)
            low_mi_features = mi_series[mi_series < self.config.mi_threshold].index.tolist()
            
            if low_mi_features:
                logger.info(f"🗑️ Removing {len(low_mi_features)} low MI features")
                features_df = features_df.drop(columns=low_mi_features)
            
            logger.info("✅ MI pruning completed")
            return features_df
            
        except Exception as e:
            logger.error(f"❌ MI pruning failed: {e}")
            return features_df
    
    def _prune_low_hsic_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Prune features with low HSIC scores using sampling."""
        logger.info("🔄 Pruning low HSIC features with sampling")
        
        if features_df.shape[1] <= 1:
            return features_df
        
        try:
            # Sample data for HSIC calculation
            n_samples = min(self.config.hsic_sample_size, len(features_df))
            if len(features_df) > n_samples:
                sampled_data = features_df.sample(n=n_samples, random_state=42)
            else:
                sampled_data = features_df
            
            # Calculate HSIC scores (simplified version)
            hsic_scores = []
            for col in sampled_data.columns:
                try:
                    # Use correlation as a proxy for HSIC (simplified)
                    hsic_score = abs(sampled_data[col].corr(sampled_data.iloc[:, 0]))
                    hsic_scores.append(hsic_score)
                except:
                    hsic_scores.append(0.0)
            
            # Find features with low HSIC
            hsic_series = pd.Series(hsic_scores, index=sampled_data.columns)
            low_hsic_features = hsic_series[hsic_series < self.config.hsic_threshold].index.tolist()
            
            if low_hsic_features:
                logger.info(f"🗑️ Removing {len(low_hsic_features)} low HSIC features")
                features_df = features_df.drop(columns=low_hsic_features)
            
            logger.info("✅ HSIC pruning completed")
            return features_df
            
        except Exception as e:
            logger.error(f"❌ HSIC pruning failed: {e}")
            return features_df
    
    def _scale_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Scale features using VectorBT optimization."""
        logger.info(f"🔄 Scaling features using {self.config.scaling_method} method")
        
        try:
            if self.config.scaling_method == 'minmax':
                # Use VectorBT minmax scaling
                if hasattr(self.vectorization_manager, 'minmax_scale'):
                    scaled_df = self.vectorization_manager.minmax_scale(features_df)
                else:
                    # Fallback to manual minmax scaling
                    scaled_df = (features_df - features_df.min()) / (features_df.max() - features_df.min())
            else:
                # Use sklearn scalers
                scaler = self.scalers[self.config.scaling_method]
                scaled_df = pd.DataFrame(
                    scaler.fit_transform(features_df),
                    index=features_df.index,
                    columns=features_df.columns
                )
            
            logger.info("✅ Feature scaling completed")
            return scaled_df
            
        except Exception as e:
            logger.error(f"❌ Feature scaling failed: {e}")
            return features_df
    
    def _validate_features(self, features_df: pd.DataFrame):
        """Validate input features."""
        if not isinstance(features_df, pd.DataFrame):
            raise ValueError("Features must be a pandas DataFrame")
        
        if features_df.empty:
            raise ValueError("Features DataFrame cannot be empty")
        
        if features_df.isnull().all().any():
            raise ValueError("Features DataFrame contains columns with all NaN values")
    
    def _update_performance_stats(self, features_df: pd.DataFrame, preprocessing_time: float):
        """Update performance statistics."""
        self.performance_stats['preprocessing_time'] = preprocessing_time
        
        # Calculate memory usage
        memory_usage = features_df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        self.performance_stats['memory_usage_mb'] = memory_usage
        
        # Get VectorBT usage rate
        vectorization_stats = self.vectorization_manager.get_performance_stats()
        self.performance_stats['vectorbt_usage_rate'] = vectorization_stats.get('vectorbt_usage_rate', 0)
        
        # Calculate sampling efficiency
        if preprocessing_time > 0:
            features_per_second = features_df.shape[1] / preprocessing_time
            self.performance_stats['sampling_efficiency'] = features_per_second
    
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
            'preprocessing_time': 0.0,
            'features_removed': 0,
            'memory_usage_mb': 0.0,
            'vectorbt_usage_rate': 0.0,
            'sampling_efficiency': 0.0
        }
        
        # Reset vectorization manager stats
        self.vectorization_manager.reset_stats()

# Convenience function for easy usage
def create_optimized_preprocessor(
    correlation_threshold: float = 0.95,
    mi_threshold: float = 0.01,
    hsic_threshold: float = 0.01,
    scaling_method: str = 'robust',
    use_approximate_correlation: bool = True,
    memory_efficient: bool = True,
    enable_vectorbt: bool = True,
    enable_gpu: bool = False
) -> OptimizedPreprocessor:
    """
    Create an optimized preprocessor with specified configuration.
    
    Args:
        correlation_threshold: Threshold for correlation pruning
        mi_threshold: Threshold for mutual information pruning
        hsic_threshold: Threshold for HSIC pruning
        scaling_method: Scaling method ('standard', 'robust', 'minmax')
        use_approximate_correlation: Use sampling for correlation calculation
        memory_efficient: Enable memory optimization
        enable_vectorbt: Enable VectorBT acceleration
        enable_gpu: Enable GPU acceleration
        
    Returns:
        OptimizedPreprocessor instance
    """
    config = PreprocessingConfig(
        correlation_threshold=correlation_threshold,
        mi_threshold=mi_threshold,
        hsic_threshold=hsic_threshold,
        scaling_method=scaling_method,
        use_approximate_correlation=use_approximate_correlation,
        memory_efficient=memory_efficient,
        enable_vectorbt=enable_vectorbt,
        enable_gpu=enable_gpu
    )
    
    return OptimizedPreprocessor(config)

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 50
    
    # Create correlated features
    base_features = np.random.randn(n_samples, 10)
    correlated_features = []
    
    for i in range(5):
        # Create highly correlated features
        noise = np.random.randn(n_samples) * 0.1
        correlated_features.append(base_features[:, i] + noise)
        correlated_features.append(base_features[:, i] + noise * 0.5)
    
    # Create independent features
    independent_features = np.random.randn(n_samples, 30)
    
    # Combine all features
    all_features = np.column_stack([base_features, correlated_features, independent_features])
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(all_features.shape[1])]
    features_df = pd.DataFrame(all_features, columns=feature_names)
    
    print(f"Original features: {features_df.shape}")
    
    # Create optimized preprocessor
    preprocessor = create_optimized_preprocessor(
        correlation_threshold=0.8,
        mi_threshold=0.01,
        scaling_method='robust',
        use_approximate_correlation=True,
        memory_efficient=True,
        enable_vectorbt=True
    )
    
    # Preprocess features
    processed_features = preprocessor.preprocess_features(features_df)
    
    print(f"Processed features: {processed_features.shape}")
    print(f"Performance stats: {preprocessor.get_performance_stats()}")
