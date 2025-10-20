"""
Optimized Regime Feature Processor

This module integrates the efficient regime feature selector with the existing
HDBSCAN clustering pipeline, providing a complete optimized feature processing
solution for regime discovery.

Key Features:
- Integration with existing OptimizedPreprocessor and OptimizedFeatureExtractor
- Efficient feature selection using mRMR and LASSO
- Regime-specific importance scoring
- Memory and computation optimization
- VectorBT acceleration
- Performance monitoring and statistics
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time

# Import existing optimization modules
from .optimized_preprocessor import OptimizedPreprocessor, PreprocessingConfig
from .optimized_feature_extractor import OptimizedFeatureExtractor, FeatureExtractionConfig
from .efficient_regime_feature_selector import (
    EfficientRegimeFeatureSelector, 
    EfficientFeatureSelectionConfig,
    create_efficient_regime_feature_selector
)

# Import optimization utilities
from src.utils.common_operations import (
    memory_monitor, optimize_dataframe_memory, safe_divide, safe_mean, safe_std,
    validate_finite, force_garbage_collection, get_memory_usage
)
from src.utils.math_validation import validate_positive, validate_range
from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error, tprint_performance

logger = logging.getLogger(__name__)

@dataclass
class OptimizedRegimeFeatureProcessorConfig:
    """Configuration for optimized regime feature processing."""
    # Feature extraction configuration
    enable_returns: bool = True
    enable_volatility: bool = True
    enable_volume: bool = True
    enable_entropy: bool = True
    enable_spectral: bool = True
    max_features_per_family: int = 50
    
    # Preprocessing configuration
    enable_winsorization: bool = True
    winsorize_limits: Tuple[float, float] = (0.05, 0.05)
    enable_correlation_pruning: bool = True
    correlation_threshold: float = 0.95
    enable_scaling: bool = True
    scaling_method: str = 'robust'
    
    # Feature selection configuration
    enable_feature_selection: bool = True
    selection_method: str = 'hybrid'  # 'mrmr', 'lasso', 'hybrid'
    k_features: int = 50
    enable_sampling: bool = True
    sample_size: int = 1000
    
    # Memory and performance optimization
    memory_efficient: bool = True
    chunk_size: int = 1000
    max_memory_gb: float = 8.0
    enable_vectorbt: bool = True
    enable_gpu: bool = False
    
    # Parallel processing
    max_workers: Optional[int] = None
    use_multiprocessing: bool = True
    
    # Regime-specific parameters
    regime_detection_method: str = 'volatility'
    n_regime_classes: int = 3
    regime_window: int = 20

class OptimizedRegimeFeatureProcessor:
    """
    Optimized regime feature processor that combines feature extraction,
    preprocessing, and selection for efficient regime discovery.
    
    This class integrates with the existing HDBSCAN clustering pipeline
    to provide a complete optimized feature processing solution.
    """
    
    def __init__(self, config: Optional[OptimizedRegimeFeatureProcessorConfig] = None):
        """Initialize the optimized regime feature processor."""
        self.config = config or OptimizedRegimeFeatureProcessorConfig()
        
        # Initialize components
        self._initialize_preprocessor()
        self._initialize_feature_extractor()
        self._initialize_feature_selector()
        
        # Performance tracking
        self.performance_stats = {
            'total_processing_time': 0.0,
            'feature_extraction_time': 0.0,
            'preprocessing_time': 0.0,
            'feature_selection_time': 0.0,
            'final_features_count': 0,
            'original_features_count': 0,
            'memory_usage_mb': 0.0,
            'vectorbt_usage_rate': 0.0
        }
        
        tprint_info("✅ OptimizedRegimeFeatureProcessor initialized")
    
    def _initialize_preprocessor(self):
        """Initialize the optimized preprocessor."""
        preprocessing_config = PreprocessingConfig(
            winsorize_limits=self.config.winsorize_limits,
            enable_winsorization=self.config.enable_winsorization,
            correlation_threshold=self.config.correlation_threshold,
            enable_correlation_pruning=self.config.enable_correlation_pruning,
            scaling_method=self.config.scaling_method,
            enable_scaling=self.config.enable_scaling,
            memory_efficient=self.config.memory_efficient,
            chunk_size=self.config.chunk_size,
            max_memory_gb=self.config.max_memory_gb,
            enable_vectorbt=self.config.enable_vectorbt,
            enable_gpu=self.config.enable_gpu
        )
        self.preprocessor = OptimizedPreprocessor(preprocessing_config)
    
    def _initialize_feature_extractor(self):
        """Initialize the optimized feature extractor."""
        extraction_config = FeatureExtractionConfig(
            enable_returns=self.config.enable_returns,
            enable_volatility=self.config.enable_volatility,
            enable_volume=self.config.enable_volume,
            enable_entropy=self.config.enable_entropy,
            enable_spectral=self.config.enable_spectral,
            max_workers=self.config.max_workers,
            use_multiprocessing=self.config.use_multiprocessing,
            chunk_size=self.config.chunk_size,
            memory_efficient=self.config.memory_efficient,
            max_memory_gb=self.config.max_memory_gb,
            enable_vectorbt=self.config.enable_vectorbt,
            enable_gpu=self.config.enable_gpu,
            max_features_per_family=self.config.max_features_per_family
        )
        self.feature_extractor = OptimizedFeatureExtractor(extraction_config)
    
    def _initialize_feature_selector(self):
        """Initialize the efficient regime feature selector."""
        selection_config = EfficientFeatureSelectionConfig(
            mrmr_k_features=self.config.k_features,
            selection_method=self.config.selection_method,
            enable_sampling=self.config.enable_sampling,
            sample_size=self.config.sample_size,
            memory_efficient=self.config.memory_efficient,
            chunk_size=self.config.chunk_size,
            max_memory_gb=self.config.max_memory_gb,
            enable_vectorbt=self.config.enable_vectorbt,
            enable_gpu=self.config.enable_gpu,
            regime_detection_method=self.config.regime_detection_method,
            n_regime_classes=self.config.n_regime_classes,
            regime_window=self.config.regime_window
        )
        self.feature_selector = EfficientRegimeFeatureSelector(selection_config)
    
    def process_features(self, data: pd.DataFrame, 
                        symbol: str, 
                        timeframe: str,
                        target: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Process features for regime discovery with full optimization pipeline.
        
        Args:
            data: OHLCV data
            symbol: Trading symbol
            timeframe: Data timeframe
            target: Target variable (optional, will create pseudo-target for regime discovery)
            
        Returns:
            Processed features DataFrame optimized for regime discovery
        """
        start_time = time.time()
        
        with memory_monitor("Optimized Regime Feature Processing"):
            tprint_info(f"🚀 Starting optimized regime feature processing for {symbol} {timeframe}")
            
            # Step 1: Feature Extraction
            tprint_info("🔄 Step 1: Feature Extraction")
            extraction_start = time.time()
            features_df = self.feature_extractor.extract_features(data, symbol, timeframe)
            extraction_time = time.time() - extraction_start
            self.performance_stats['feature_extraction_time'] = extraction_time
            
            tprint_success(f"✅ Feature extraction completed: {features_df.shape[1]} features in {extraction_time:.2f}s")
            
            # Step 2: Preprocessing
            tprint_info("🔄 Step 2: Preprocessing")
            preprocessing_start = time.time()
            processed_features = self.preprocessor.preprocess_features(features_df)
            preprocessing_time = time.time() - preprocessing_start
            self.performance_stats['preprocessing_time'] = preprocessing_time
            
            tprint_success(f"✅ Preprocessing completed: {processed_features.shape[1]} features in {preprocessing_time:.2f}s")
            
            # Step 3: Feature Selection (if enabled)
            if self.config.enable_feature_selection:
                tprint_info("🔄 Step 3: Feature Selection")
                selection_start = time.time()
                selected_features = self.feature_selector.select_features(processed_features, target)
                selection_time = time.time() - selection_start
                self.performance_stats['feature_selection_time'] = selection_time
                
                tprint_success(f"✅ Feature selection completed: {selected_features.shape[1]} features in {selection_time:.2f}s")
                
                final_features = selected_features
            else:
                final_features = processed_features
                self.performance_stats['feature_selection_time'] = 0.0
            
            # Step 4: Final optimization and validation
            final_features = self._finalize_features(final_features, data)
            
            # Update performance stats
            total_time = time.time() - start_time
            self._update_performance_stats(features_df, final_features, total_time)
            
            tprint_success(f"✅ Optimized regime feature processing completed: {final_features.shape[1]} features in {total_time:.2f}s")
            
            return final_features
    
    def _finalize_features(self, features_df: pd.DataFrame, 
                          original_data: pd.DataFrame) -> pd.DataFrame:
        """Finalize features with additional optimization and validation."""
        tprint_info("🔄 Finalizing features")
        
        # Memory optimization
        if self.config.memory_efficient:
            features_df = optimize_dataframe(features_df)
        
        # Final validation
        features_df = self._validate_final_features(features_df)
        
        # Ensure index alignment with original data
        features_df = features_df.reindex(original_data.index)
        
        # Fill any remaining NaN values
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        
        tprint_info("✅ Feature finalization completed")
        
        return features_df
    
    def _validate_final_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Validate final features for regime discovery."""
        # Remove constant features
        constant_features = features_df.var() == 0
        if constant_features.any():
            tprint_warning(f"⚠️ Removing {constant_features.sum()} constant features")
            features_df = features_df.loc[:, ~constant_features]
        
        # Remove features with infinite values
        infinite_features = np.isinf(features_df).any()
        if infinite_features.any():
            tprint_warning(f"⚠️ Removing {infinite_features.sum()} features with infinite values")
            features_df = features_df.loc[:, ~infinite_features]
        
        # Ensure minimum number of features
        if features_df.shape[1] < 5:
            tprint_warning("⚠️ Very few features remaining, this may affect clustering quality")
        
        return features_df
    
    def _update_performance_stats(self, original_features: pd.DataFrame, 
                                final_features: pd.DataFrame, 
                                total_time: float):
        """Update performance statistics."""
        self.performance_stats['total_processing_time'] = total_time
        self.performance_stats['final_features_count'] = final_features.shape[1]
        self.performance_stats['original_features_count'] = original_features.shape[1]
        
        # Calculate memory usage
        memory_usage = final_features.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        self.performance_stats['memory_usage_mb'] = memory_usage
        
        # Get VectorBT usage rate from components
        preprocessor_stats = self.preprocessor.get_performance_stats()
        extractor_stats = self.feature_extractor.get_performance_stats()
        selector_stats = self.feature_selector.get_performance_stats()
        
        vectorbt_rates = [
            preprocessor_stats.get('vectorbt_usage_rate', 0),
            extractor_stats.get('vectorbt_usage_rate', 0),
            selector_stats.get('vectorbt_usage_rate', 0)
        ]
        
        self.performance_stats['vectorbt_usage_rate'] = np.mean(vectorbt_rates)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add component stats
        stats['preprocessor_stats'] = self.preprocessor.get_performance_stats()
        stats['extractor_stats'] = self.feature_extractor.get_performance_stats()
        stats['selector_stats'] = self.feature_selector.get_performance_stats()
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_processing_time': 0.0,
            'feature_extraction_time': 0.0,
            'preprocessing_time': 0.0,
            'feature_selection_time': 0.0,
            'final_features_count': 0,
            'original_features_count': 0,
            'memory_usage_mb': 0.0,
            'vectorbt_usage_rate': 0.0
        }
        
        # Reset component stats
        self.preprocessor.reset_stats()
        self.feature_extractor.reset_stats()
        self.feature_selector.reset_stats()

# Convenience function for easy usage
def create_optimized_regime_feature_processor(
    k_features: int = 50,
    selection_method: str = 'hybrid',
    enable_feature_selection: bool = True,
    enable_sampling: bool = True,
    sample_size: int = 1000,
    memory_efficient: bool = True,
    enable_vectorbt: bool = True,
    enable_gpu: bool = False,
    max_workers: Optional[int] = None
) -> OptimizedRegimeFeatureProcessor:
    """
    Create an optimized regime feature processor with specified configuration.
    
    Args:
        k_features: Number of features to select
        selection_method: Selection method ('mrmr', 'lasso', 'hybrid')
        enable_feature_selection: Enable feature selection
        enable_sampling: Enable sampling for efficiency
        sample_size: Sample size for computations
        memory_efficient: Enable memory optimization
        enable_vectorbt: Enable VectorBT acceleration
        enable_gpu: Enable GPU acceleration
        max_workers: Maximum number of parallel workers
        
    Returns:
        OptimizedRegimeFeatureProcessor instance
    """
    config = OptimizedRegimeFeatureProcessorConfig(
        k_features=k_features,
        selection_method=selection_method,
        enable_feature_selection=enable_feature_selection,
        enable_sampling=enable_sampling,
        sample_size=sample_size,
        memory_efficient=memory_efficient,
        enable_vectorbt=enable_vectorbt,
        enable_gpu=enable_gpu,
        max_workers=max_workers
    )
    
    return OptimizedRegimeFeatureProcessor(config)

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
    
    # Create optimized regime feature processor
    processor = create_optimized_regime_feature_processor(
        k_features=30,
        selection_method='hybrid',
        enable_feature_selection=True,
        enable_sampling=True,
        sample_size=500,
        memory_efficient=True,
        enable_vectorbt=True,
        max_workers=4
    )
    
    # Process features
    processed_features = processor.process_features(data, symbol="BTCUSDT", timeframe="15m")
    
    print(f"Processed features: {processed_features.shape}")
    print(f"Performance stats: {processor.get_performance_stats()}")
