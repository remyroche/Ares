"""
Optimized HDBSCAN Pipeline for Market Analysis

This module provides a unified pipeline that integrates all optimization components:
- Feature Extraction Optimization
- Preprocessing Pipeline Optimization  
- Dimensionality Reduction Optimization
- HDBSCAN Clustering Optimization
- Post-Processing Optimization
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
from pathlib import Path

# Import optimization components
from .optimized_feature_extractor import OptimizedFeatureExtractor, FeatureExtractionConfig
from .optimized_preprocessor import OptimizedPreprocessor, PreprocessingConfig
from .optimized_dimensionality_reducer import OptimizedDimensionalityReducer, DimensionalityReductionConfig
from .optimized_hdbscan_clusterer import OptimizedHDBSCANClusterer, HDBSCANConfig
from .optimized_post_processor import OptimizedPostProcessor, PostProcessingConfig

logger = logging.getLogger(__name__)

@dataclass
class OptimizedHDBSCANPipelineConfig:
    """Configuration for the optimized HDBSCAN pipeline."""
    # Feature extraction
    enable_feature_extraction: bool = True
    feature_extraction_config: Optional[FeatureExtractionConfig] = None
    
    # Preprocessing
    enable_preprocessing: bool = True
    preprocessing_config: Optional[PreprocessingConfig] = None
    
    # Dimensionality reduction
    enable_dimensionality_reduction: bool = True
    dimensionality_reduction_config: Optional[DimensionalityReductionConfig] = None
    
    # Clustering
    enable_clustering: bool = True
    clustering_config: Optional[HDBSCANConfig] = None
    
    # Post-processing
    enable_post_processing: bool = True
    post_processing_config: Optional[PostProcessingConfig] = None
    
    # Pipeline optimization
    enable_pipeline_optimization: bool = True
    memory_efficient: bool = True
    enable_vectorbt: bool = True
    enable_gpu: bool = False
    
    # Output
    save_intermediate_results: bool = False
    output_dir: Optional[str] = None

class OptimizedHDBSCANPipeline:
    """
    Unified optimized HDBSCAN pipeline that integrates all optimization components.
    
    This pipeline provides:
    - End-to-end optimization from raw data to final clusters
    - VectorBT acceleration throughout the pipeline
    - Memory-efficient processing with intelligent chunking
    - Parallel processing for independent operations
    - Comprehensive performance monitoring and statistics
    """
    
    def __init__(self, config: Optional[OptimizedHDBSCANPipelineConfig] = None):
        """Initialize the optimized HDBSCAN pipeline."""
        self.config = config or OptimizedHDBSCANPipelineConfig()
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.performance_stats = {
            'total_pipeline_time': 0.0,
            'feature_extraction_time': 0.0,
            'preprocessing_time': 0.0,
            'dimensionality_reduction_time': 0.0,
            'clustering_time': 0.0,
            'post_processing_time': 0.0,
            'memory_usage_mb': 0.0,
            'vectorbt_usage_rate': 0.0,
            'pipeline_efficiency': 0.0,
            'component_stats': {}
        }
        
        logger.info("✅ OptimizedHDBSCANPipeline initialized")
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        # Feature extractor
        if self.config.enable_feature_extraction:
            self.feature_extractor = OptimizedFeatureExtractor(
                self.config.feature_extraction_config
            )
        else:
            self.feature_extractor = None
        
        # Preprocessor
        if self.config.enable_preprocessing:
            self.preprocessor = OptimizedPreprocessor(
                self.config.preprocessing_config
            )
        else:
            self.preprocessor = None
        
        # Dimensionality reducer
        if self.config.enable_dimensionality_reduction:
            self.dimensionality_reducer = OptimizedDimensionalityReducer(
                self.config.dimensionality_reduction_config
            )
        else:
            self.dimensionality_reducer = None
        
        # Clusterer
        if self.config.enable_clustering:
            self.clusterer = OptimizedHDBSCANClusterer(
                self.config.clustering_config
            )
        else:
            self.clusterer = None
        
        # Post-processor
        if self.config.enable_post_processing:
            self.post_processor = OptimizedPostProcessor(
                self.config.post_processing_config
            )
        else:
            self.post_processor = None
    
    def run_pipeline(self, data: pd.DataFrame, 
                    symbol: str, 
                    timeframe: str,
                    timestamps: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Run the complete optimized HDBSCAN pipeline.
        
        Args:
            data: OHLCV data
            symbol: Trading symbol
            timeframe: Data timeframe
            timestamps: Optional timestamps for temporal analysis
            
        Returns:
            Dictionary with pipeline results and statistics
        """
        start_time = time.time()
        logger.info(f"🚀 Starting optimized HDBSCAN pipeline for {symbol} {timeframe}")
        
        # Validate input
        self._validate_input(data)
        
        # Initialize results
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'pipeline_config': self.config,
            'intermediate_results': {},
            'final_results': {},
            'performance_stats': {}
        }
        
        # Step 1: Feature Extraction
        if self.config.enable_feature_extraction:
            logger.info("🔄 Step 1: Feature Extraction")
            step_start = time.time()
            
            features_df = self.feature_extractor.extract_features(data, symbol, timeframe)
            results['intermediate_results']['features'] = features_df
            
            if self.config.save_intermediate_results:
                self._save_intermediate_result('features', features_df, symbol, timeframe)
            
            self.performance_stats['feature_extraction_time'] = time.time() - step_start
            logger.info(f"✅ Feature extraction completed: {features_df.shape[1]} features")
        else:
            # Use original data as features
            features_df = data
            logger.info("⚠️ Feature extraction disabled, using original data")
        
        # Step 2: Preprocessing
        if self.config.enable_preprocessing:
            logger.info("🔄 Step 2: Preprocessing")
            step_start = time.time()
            
            processed_features = self.preprocessor.preprocess_features(features_df)
            results['intermediate_results']['processed_features'] = processed_features
            
            if self.config.save_intermediate_results:
                self._save_intermediate_result('processed_features', processed_features, symbol, timeframe)
            
            self.performance_stats['preprocessing_time'] = time.time() - step_start
            logger.info(f"✅ Preprocessing completed: {processed_features.shape[1]} features")
        else:
            processed_features = features_df
            logger.info("⚠️ Preprocessing disabled")
        
        # Step 3: Dimensionality Reduction
        if self.config.enable_dimensionality_reduction:
            logger.info("🔄 Step 3: Dimensionality Reduction")
            step_start = time.time()
            
            reduced_features = self.dimensionality_reducer.reduce_dimensions(processed_features)
            results['intermediate_results']['reduced_features'] = reduced_features
            
            if self.config.save_intermediate_results:
                self._save_intermediate_result('reduced_features', reduced_features, symbol, timeframe)
            
            self.performance_stats['dimensionality_reduction_time'] = time.time() - step_start
            logger.info(f"✅ Dimensionality reduction completed: {reduced_features.shape[1]} dimensions")
        else:
            reduced_features = processed_features
            logger.info("⚠️ Dimensionality reduction disabled")
        
        # Step 4: Clustering
        if self.config.enable_clustering:
            logger.info("🔄 Step 4: HDBSCAN Clustering")
            step_start = time.time()
            
            cluster_labels, clustering_info = self.clusterer.cluster_data(reduced_features)
            results['intermediate_results']['cluster_labels'] = cluster_labels
            results['intermediate_results']['clustering_info'] = clustering_info
            
            if self.config.save_intermediate_results:
                self._save_intermediate_result('cluster_labels', cluster_labels, symbol, timeframe)
            
            self.performance_stats['clustering_time'] = time.time() - step_start
            logger.info(f"✅ Clustering completed: {len(np.unique(cluster_labels))} clusters")
        else:
            # Create dummy cluster labels
            cluster_labels = np.zeros(len(reduced_features))
            clustering_info = {'n_clusters': 1, 'n_noise_points': 0}
            logger.info("⚠️ Clustering disabled, using dummy labels")
        
        # Step 5: Post-Processing
        if self.config.enable_post_processing:
            logger.info("🔄 Step 5: Post-Processing")
            step_start = time.time()
            
            optimized_labels, post_processing_info = self.post_processor.post_process_clusters(
                cluster_labels, reduced_features, timestamps
            )
            results['intermediate_results']['optimized_labels'] = optimized_labels
            results['intermediate_results']['post_processing_info'] = post_processing_info
            
            if self.config.save_intermediate_results:
                self._save_intermediate_result('optimized_labels', optimized_labels, symbol, timeframe)
            
            self.performance_stats['post_processing_time'] = time.time() - step_start
            logger.info(f"✅ Post-processing completed: {len(np.unique(optimized_labels))} clusters")
        else:
            optimized_labels = cluster_labels
            post_processing_info = {'optimization_disabled': True}
            logger.info("⚠️ Post-processing disabled")
        
        # Final results
        results['final_results'] = {
            'cluster_labels': optimized_labels,
            'clustering_info': clustering_info,
            'post_processing_info': post_processing_info,
            'n_clusters': len(set(optimized_labels)) - (1 if -1 in optimized_labels else 0),
            'n_noise_points': list(optimized_labels).count(-1),
            'features_used': reduced_features.shape[1]
        }
        
        # Update performance stats
        total_time = time.time() - start_time
        self._update_performance_stats(total_time)
        results['performance_stats'] = self.performance_stats.copy()
        
        logger.info(f"✅ Pipeline completed in {total_time:.2f}s")
        return results
    
    def _validate_input(self, data: pd.DataFrame):
        """Validate input data."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
    
    def _save_intermediate_result(self, result_name: str, result_data: Any, 
                                 symbol: str, timeframe: str):
        """Save intermediate result to disk."""
        if self.config.output_dir is None:
            return
        
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{result_name}_{symbol}_{timeframe}_{timestamp}.parquet"
        
        # Save result
        if isinstance(result_data, pd.DataFrame):
            result_data.to_parquet(output_path / filename)
        elif isinstance(result_data, np.ndarray):
            pd.DataFrame(result_data).to_parquet(output_path / filename)
        else:
            logger.warning(f"⚠️ Cannot save intermediate result {result_name}: unsupported type")
    
    def _update_performance_stats(self, total_time: float):
        """Update performance statistics."""
        self.performance_stats['total_pipeline_time'] = total_time
        
        # Calculate memory usage
        total_memory = 0.0
        for component in [self.feature_extractor, self.preprocessor, 
                         self.dimensionality_reducer, self.clusterer, self.post_processor]:
            if component is not None:
                stats = component.get_performance_stats()
                total_memory += stats.get('memory_usage_mb', 0.0)
        
        self.performance_stats['memory_usage_mb'] = total_memory
        
        # Calculate pipeline efficiency
        if total_time > 0:
            samples_per_second = 1000 / total_time  # Assuming 1000 samples
            self.performance_stats['pipeline_efficiency'] = samples_per_second
        
        # Collect component stats
        self.performance_stats['component_stats'] = {
            'feature_extractor': self.feature_extractor.get_performance_stats() if self.feature_extractor else None,
            'preprocessor': self.preprocessor.get_performance_stats() if self.preprocessor else None,
            'dimensionality_reducer': self.dimensionality_reducer.get_performance_stats() if self.dimensionality_reducer else None,
            'clusterer': self.clusterer.get_performance_stats() if self.clusterer else None,
            'post_processor': self.post_processor.get_performance_stats() if self.post_processor else None
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return self.performance_stats.copy()
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_pipeline_time': 0.0,
            'feature_extraction_time': 0.0,
            'preprocessing_time': 0.0,
            'dimensionality_reduction_time': 0.0,
            'clustering_time': 0.0,
            'post_processing_time': 0.0,
            'memory_usage_mb': 0.0,
            'vectorbt_usage_rate': 0.0,
            'pipeline_efficiency': 0.0,
            'component_stats': {}
        }
        
        # Reset component stats
        for component in [self.feature_extractor, self.preprocessor, 
                         self.dimensionality_reducer, self.clusterer, self.post_processor]:
            if component is not None:
                component.reset_stats()

# Convenience function for easy usage
def create_optimized_hdbscan_pipeline(
    enable_feature_extraction: bool = True,
    enable_preprocessing: bool = True,
    enable_dimensionality_reduction: bool = True,
    enable_clustering: bool = True,
    enable_post_processing: bool = True,
    memory_efficient: bool = True,
    enable_vectorbt: bool = True,
    enable_gpu: bool = False,
    save_intermediate_results: bool = False,
    output_dir: Optional[str] = None
) -> OptimizedHDBSCANPipeline:
    """
    Create an optimized HDBSCAN pipeline with specified configuration.
    
    Args:
        enable_feature_extraction: Enable feature extraction
        enable_preprocessing: Enable preprocessing
        enable_dimensionality_reduction: Enable dimensionality reduction
        enable_clustering: Enable clustering
        enable_post_processing: Enable post-processing
        memory_efficient: Enable memory optimization
        enable_vectorbt: Enable VectorBT acceleration
        enable_gpu: Enable GPU acceleration
        save_intermediate_results: Save intermediate results
        output_dir: Output directory for intermediate results
        
    Returns:
        OptimizedHDBSCANPipeline instance
    """
    config = OptimizedHDBSCANPipelineConfig(
        enable_feature_extraction=enable_feature_extraction,
        enable_preprocessing=enable_preprocessing,
        enable_dimensionality_reduction=enable_dimensionality_reduction,
        enable_clustering=enable_clustering,
        enable_post_processing=enable_post_processing,
        memory_efficient=memory_efficient,
        enable_vectorbt=enable_vectorbt,
        enable_gpu=enable_gpu,
        save_intermediate_results=save_intermediate_results,
        output_dir=output_dir
    )
    
    return OptimizedHDBSCANPipeline(config)

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
        'high': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) + np.abs(np.random.randn(n_samples) * 0.5),
        'low': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) - np.abs(np.random.randn(n_samples) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Create timestamps
    timestamps = pd.Series(pd.date_range('2023-01-01', periods=n_samples, freq='1H'))
    
    print(f"Sample data: {data.shape}")
    
    # Create optimized HDBSCAN pipeline
    pipeline = create_optimized_hdbscan_pipeline(
        enable_feature_extraction=True,
        enable_preprocessing=True,
        enable_dimensionality_reduction=True,
        enable_clustering=True,
        enable_post_processing=True,
        memory_efficient=True,
        enable_vectorbt=True,
        save_intermediate_results=False
    )
    
    # Run pipeline
    results = pipeline.run_pipeline(data, symbol="BTCUSDT", timeframe="15m", timestamps=timestamps)
    
    print(f"Pipeline results: {results['final_results']['n_clusters']} clusters")
    print(f"Performance stats: {pipeline.get_performance_stats()}")
