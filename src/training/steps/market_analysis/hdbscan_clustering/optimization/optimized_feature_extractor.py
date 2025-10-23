"""
Optimized Feature Extractor for HDBSCAN Clustering

This module provides optimized feature extraction using the UnifiedVectorizationManager
and the feature_generation/ system to leverage the existing feature bank.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Import UnifiedVectorizationManager
from src.utils.ml_common.unified_vectorization_manager import (
    UnifiedVectorizationManager, 
    VectorizationConfig,
    get_unified_vectorization_manager
)

# Import feature generation system
from src.feature_generation.core.feature_bank import FeatureBank
from src.feature_generation.categories.returns import ReturnsFeatureExtractor
from src.feature_generation.categories.volatility import VolatilityFeatureExtractor
from src.feature_generation.categories.volume import VolumeFeatureExtractor
from src.feature_generation.categories.entropy import EntropyFeatureExtractor
from src.feature_generation.categories.spectral_features import SpectralFeatureExtractor

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_progress, tprint_timer,
    tprint_logged, LogLevel
)
from src.utils.hardware import optimize_dataframe_default, get_memory_usage

logger = logging.getLogger(__name__)

@dataclass
class FeatureExtractionConfig:
    """Configuration for optimized feature extraction."""
    # Feature families to extract
    enable_returns: bool = True
    enable_volatility: bool = True
    enable_volume: bool = True
    enable_entropy: bool = True
    enable_spectral: bool = True
    
    # Parallel processing
    max_workers: Optional[int] = None
    use_multiprocessing: bool = True
    chunk_size: int = 1000
    
    # Memory optimization
    memory_efficient: bool = True
    max_memory_gb: float = 8.0
    
    # VectorBT optimization
    enable_vectorbt: bool = True
    enable_gpu: bool = False
    
    # Feature selection
    max_features_per_family: int = 50
    correlation_threshold: float = 0.95
    importance_threshold: float = 0.01

class OptimizedFeatureExtractor:
    """
    Optimized feature extractor using UnifiedVectorizationManager and feature_generation/.
    
    This class provides:
    - Parallel feature extraction using existing feature bank
    - Memory-efficient processing with chunking
    - VectorBT acceleration for financial operations
    - Intelligent feature selection and pruning
    """
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def __init__(self, config: Optional[FeatureExtractionConfig] = None):
        """Initialize the optimized feature extractor."""
        start_time = time.perf_counter()
        initial_memory = get_memory_usage()
        
        self.config = config or FeatureExtractionConfig()
        
        tprint_info("Initializing OptimizedFeatureExtractor")
        tprint_debug(f"Config: max_features_per_family={self.config.max_features_per_family}, max_workers={self.config.max_workers}")
        
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
        
        # Initialize feature extractors
        with tprint_timer("Feature extractors initialization"):
            self._initialize_feature_extractors()
            tprint_debug("Feature extractors initialized successfully")
        
        # Performance tracking
        self.performance_stats = {
            'total_features_extracted': 0,
            'extraction_time': 0.0,
            'memory_usage_mb': 0.0,
            'parallel_efficiency': 0.0,
            'vectorbt_usage_rate': 0.0,
            'feature_family_stats': {},
            'initialization_time': 0.0,
            'initial_memory_mb': initial_memory
        }
        
        # Track initialization performance
        init_time = time.perf_counter() - start_time
        final_memory = get_memory_usage()
        self.performance_stats['initialization_time'] = init_time
        self.performance_stats['memory_usage_mb'] = final_memory
        
        tprint_success("✅ OptimizedFeatureExtractor initialized")
        tprint_performance("Feature extractor initialization", init_time)
        tprint_debug(f"Memory usage: {initial_memory:.2f}MB -> {final_memory:.2f}MB (delta: {final_memory - initial_memory:+.2f}MB)")
        
        logger.info("✅ OptimizedFeatureExtractor initialized")
    
    def _initialize_feature_extractors(self):
        """Initialize feature extractors for each family."""
        self.feature_extractors = {}
        
        if self.config.enable_returns:
            self.feature_extractors['returns'] = ReturnsFeatureExtractor()
        
        if self.config.enable_volatility:
            self.feature_extractors['volatility'] = VolatilityFeatureExtractor()
        
        if self.config.enable_volume:
            self.feature_extractors['volume'] = VolumeFeatureExtractor()
        
        if self.config.enable_entropy:
            self.feature_extractors['entropy'] = EntropyFeatureExtractor()
        
        if self.config.enable_spectral:
            self.feature_extractors['spectral'] = SpectralFeatureExtractor()
    
    def extract_features(self, data: pd.DataFrame, 
                        symbol: str, 
                        timeframe: str) -> pd.DataFrame:
        """
        Extract optimized features using parallel processing and VectorBT acceleration.
        
        Args:
            data: OHLCV data
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            DataFrame with extracted features
        """
        start_time = time.time()
        logger.info(f"🚀 Starting optimized feature extraction for {symbol} {timeframe}")
        
        # Validate input data
        self._validate_data(data)
        
        # Determine optimal number of workers
        max_workers = self._determine_optimal_workers()
        
        # Extract features in parallel
        if max_workers > 1 and self.config.use_multiprocessing:
            features_df = self._extract_features_parallel(data, symbol, timeframe, max_workers)
        else:
            features_df = self._extract_features_sequential(data, symbol, timeframe)
        
        # Post-process features
        features_df = self._post_process_features(features_df, data)
        
        # Update performance stats
        extraction_time = time.time() - start_time
        self._update_performance_stats(features_df, extraction_time)
        
        logger.info(f"✅ Feature extraction completed: {features_df.shape[1]} features in {extraction_time:.2f}s")
        return features_df
    
    def _extract_features_parallel(self, data: pd.DataFrame, 
                                  symbol: str, 
                                  timeframe: str, 
                                  max_workers: int) -> pd.DataFrame:
        """Extract features using parallel processing."""
        logger.info(f"🔄 Extracting features in parallel with {max_workers} workers")
        
        # Prepare feature extraction tasks
        tasks = []
        for family_name, extractor in self.feature_extractors.items():
            tasks.append((family_name, extractor, data.copy(), symbol, timeframe))
        
        # Execute in parallel
        if self.config.use_multiprocessing:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self._extract_family_features, *task) for task in tasks]
                family_results = [future.result() for future in futures]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self._extract_family_features, *task) for task in tasks]
                family_results = [future.result() for future in futures]
        
        # Combine results
        features_df = self._combine_family_results(family_results, data.index)
        
        return features_df
    
    def _extract_features_sequential(self, data: pd.DataFrame, 
                                    symbol: str, 
                                    timeframe: str) -> pd.DataFrame:
        """Extract features sequentially."""
        logger.info("🔄 Extracting features sequentially")
        
        family_results = []
        for family_name, extractor in self.feature_extractors.items():
            result = self._extract_family_features(family_name, extractor, data, symbol, timeframe)
            family_results.append(result)
        
        # Combine results
        features_df = self._combine_family_results(family_results, data.index)
        
        return features_df
    
    def _extract_family_features(self, family_name: str, 
                                extractor: Any, 
                                data: pd.DataFrame, 
                                symbol: str, 
                                timeframe: str) -> Dict[str, pd.Series]:
        """Extract features for a single family using VectorBT optimization."""
        logger.debug(f"🔄 Extracting {family_name} features")
        
        try:
            # Use VectorBT optimization for feature extraction
            if hasattr(extractor, 'extract_features_vectorbt'):
                # Use VectorBT-optimized extraction
                features = extractor.extract_features_vectorbt(
                    data, 
                    symbol=symbol, 
                    timeframe=timeframe,
                    vectorization_manager=self.vectorization_manager
                )
            else:
                # Use standard extraction
                features = extractor.extract_features(data, symbol=symbol, timeframe=timeframe)
            
            # Limit features per family
            if len(features) > self.config.max_features_per_family:
                features = self._select_top_features(features, self.config.max_features_per_family)
            
            logger.debug(f"✅ {family_name} features extracted: {len(features)} features")
            return {family_name: features}
            
        except Exception as e:
            logger.error(f"❌ Failed to extract {family_name} features: {e}")
            return {family_name: pd.DataFrame()}
    
    def _combine_family_results(self, family_results: List[Dict[str, pd.DataFrame]], 
                               index: pd.Index) -> pd.DataFrame:
        """Combine results from all feature families."""
        logger.debug("🔄 Combining feature family results")
        
        all_features = []
        for family_result in family_results:
            for family_name, features_df in family_result.items():
                if not features_df.empty:
                    # Add family prefix to column names
                    features_df.columns = [f"{family_name}_{col}" for col in features_df.columns]
                    all_features.append(features_df)
        
        if not all_features:
            logger.warning("⚠️ No features extracted from any family")
            return pd.DataFrame(index=index)
        
        # Combine all features
        combined_features = pd.concat(all_features, axis=1)
        
        # Ensure index alignment
        combined_features = combined_features.reindex(index)
        
        logger.info(f"✅ Combined features: {combined_features.shape[1]} total features")
        return combined_features
    
    def _post_process_features(self, features_df: pd.DataFrame, 
                              original_data: pd.DataFrame) -> pd.DataFrame:
        """Post-process features with correlation pruning and importance filtering."""
        logger.info("🔄 Post-processing features")
        
        # Remove constant features
        features_df = self._remove_constant_features(features_df)
        
        # Remove highly correlated features
        features_df = self._remove_correlated_features(features_df)
        
        # Fill missing values
        features_df = self._fill_missing_values(features_df, original_data)
        
        # Final validation
        features_df = self._validate_features(features_df)
        
        logger.info(f"✅ Post-processing completed: {features_df.shape[1]} features remaining")
        return features_df
    
    def _remove_constant_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Remove constant or near-constant features."""
        # Remove features with zero variance
        constant_features = features_df.var() == 0
        if constant_features.any():
            logger.info(f"🗑️ Removing {constant_features.sum()} constant features")
            features_df = features_df.loc[:, ~constant_features]
        
        # Remove features with very low variance
        low_variance_features = features_df.var() < 1e-10
        if low_variance_features.any():
            logger.info(f"🗑️ Removing {low_variance_features.sum()} low-variance features")
            features_df = features_df.loc[:, ~low_variance_features]
        
        return features_df
    
    def _remove_correlated_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features."""
        if features_df.shape[1] <= 1:
            return features_df
        
        # Calculate correlation matrix
        corr_matrix = features_df.corr().abs()
        
        # Find highly correlated pairs
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = (upper_tri > self.config.correlation_threshold).any()
        
        if high_corr_pairs.any():
            # Remove features with high correlation
            features_to_remove = high_corr_pairs[high_corr_pairs].index
            logger.info(f"🗑️ Removing {len(features_to_remove)} highly correlated features")
            features_df = features_df.drop(columns=features_to_remove)
        
        return features_df
    
    def _fill_missing_values(self, features_df: pd.DataFrame, 
                            original_data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using forward fill and interpolation."""
        if features_df.isnull().any().any():
            logger.info("🔄 Filling missing values")
            
            # Forward fill first
            features_df = features_df.fillna(method='ffill')
            
            # Then backward fill
            features_df = features_df.fillna(method='bfill')
            
            # For any remaining NaN values, use median
            for col in features_df.columns:
                if features_df[col].isnull().any():
                    median_val = features_df[col].median()
                    features_df[col] = features_df[col].fillna(median_val)
        
        return features_df
    
    def _validate_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Validate final features."""
        # Check for infinite values
        if np.isinf(features_df).any().any():
            logger.warning("⚠️ Found infinite values, replacing with NaN")
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # Check for remaining NaN values
        if features_df.isnull().any().any():
            logger.warning("⚠️ Found remaining NaN values after filling")
        
        return features_df
    
    def _select_top_features(self, features_df: pd.DataFrame, 
                            max_features: int) -> pd.DataFrame:
        """Select top features based on variance and importance."""
        if features_df.shape[1] <= max_features:
            return features_df
        
        # Calculate feature importance (using variance as proxy)
        feature_importance = features_df.var().sort_values(ascending=False)
        
        # Select top features
        top_features = feature_importance.head(max_features).index
        return features_df[top_features]
    
    def _determine_optimal_workers(self) -> int:
        """Determine optimal number of workers for parallel processing."""
        if self.config.max_workers is not None:
            return self.config.max_workers
        
        # Use number of feature families as base
        num_families = len(self.feature_extractors)
        
        # Don't exceed CPU cores
        max_cores = mp.cpu_count()
        
        # Optimal is min of families and cores
        optimal_workers = min(num_families, max_cores)
        
        logger.info(f"🔄 Using {optimal_workers} workers for parallel processing")
        return optimal_workers
    
    def _validate_data(self, data: pd.DataFrame):
        """Validate input data."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
    
    def _update_performance_stats(self, features_df: pd.DataFrame, extraction_time: float):
        """Update performance statistics."""
        self.performance_stats['total_features_extracted'] = features_df.shape[1]
        self.performance_stats['extraction_time'] = extraction_time
        
        # Calculate memory usage
        memory_usage = features_df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        self.performance_stats['memory_usage_mb'] = memory_usage
        
        # Calculate parallel efficiency
        if extraction_time > 0:
            features_per_second = features_df.shape[1] / extraction_time
            self.performance_stats['parallel_efficiency'] = features_per_second
        
        # Get VectorBT usage rate from vectorization manager
        vectorization_stats = self.vectorization_manager.get_performance_stats()
        self.performance_stats['vectorbt_usage_rate'] = vectorization_stats.get('vectorbt_usage_rate', 0)
    
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
            'total_features_extracted': 0,
            'extraction_time': 0.0,
            'memory_usage_mb': 0.0,
            'parallel_efficiency': 0.0,
            'vectorbt_usage_rate': 0.0,
            'feature_family_stats': {}
        }
        
        # Reset vectorization manager stats
        self.vectorization_manager.reset_stats()

# Convenience function for easy usage
def create_optimized_feature_extractor(
    enable_returns: bool = True,
    enable_volatility: bool = True,
    enable_volume: bool = True,
    enable_entropy: bool = True,
    enable_spectral: bool = True,
    max_workers: Optional[int] = None,
    memory_efficient: bool = True,
    enable_vectorbt: bool = True,
    enable_gpu: bool = False
) -> OptimizedFeatureExtractor:
    """
    Create an optimized feature extractor with specified configuration.
    
    Args:
        enable_returns: Enable returns features
        enable_volatility: Enable volatility features
        enable_volume: Enable volume features
        enable_entropy: Enable entropy features
        enable_spectral: Enable spectral features
        max_workers: Maximum number of parallel workers
        memory_efficient: Enable memory optimization
        enable_vectorbt: Enable VectorBT acceleration
        enable_gpu: Enable GPU acceleration
        
    Returns:
        OptimizedFeatureExtractor instance
    """
    config = FeatureExtractionConfig(
        enable_returns=enable_returns,
        enable_volatility=enable_volatility,
        enable_volume=enable_volume,
        enable_entropy=enable_entropy,
        enable_spectral=enable_spectral,
        max_workers=max_workers,
        memory_efficient=memory_efficient,
        enable_vectorbt=enable_vectorbt,
        enable_gpu=enable_gpu
    )
    
    return OptimizedFeatureExtractor(config)

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(1000) * 0.01),
        'high': 100 + np.cumsum(np.random.randn(1000) * 0.01) + np.abs(np.random.randn(1000) * 0.5),
        'low': 100 + np.cumsum(np.random.randn(1000) * 0.01) - np.abs(np.random.randn(1000) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(1000) * 0.01),
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Create optimized feature extractor
    extractor = create_optimized_feature_extractor(
        enable_returns=True,
        enable_volatility=True,
        enable_volume=True,
        enable_entropy=True,
        enable_spectral=True,
        max_workers=4,
        memory_efficient=True,
        enable_vectorbt=True
    )
    
    # Extract features
    features = extractor.extract_features(data, symbol="BTCUSDT", timeframe="15m")
    
    print(f"Extracted {features.shape[1]} features")
    print(f"Performance stats: {extractor.get_performance_stats()}")
