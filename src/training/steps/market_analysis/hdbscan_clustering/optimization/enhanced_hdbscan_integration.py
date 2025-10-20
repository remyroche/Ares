"""
Enhanced HDBSCAN Integration with Comprehensive Optimizations

This module demonstrates how to integrate all enhanced optimization components
for maximum performance in HDBSCAN clustering.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time

# Import enhanced optimization components
from .enhanced_memory_optimizer import (
    EnhancedMemoryOptimizer,
    MemoryOptimizationConfig,
    create_enhanced_memory_optimizer
)

from .enhanced_hyperparameter_optimizer import (
    EnhancedHyperparameterOptimizer,
    HDBSCANHyperparameterConfig,
    create_enhanced_hyperparameter_optimizer
)

from .enhanced_vectorized_processor import (
    EnhancedVectorizedProcessor,
    VectorizedProcessingConfig,
    create_enhanced_vectorized_processor
)

# Import feature generation components
from src.feature_generation.categories.entropy import create_default_entropy_generators
from src.feature_generation.categories.spectral_features import create_default_spectral_wavelet_generators
from src.feature_generation.categories.regime_features import create_default_regime_generators

logger = logging.getLogger(__name__)

@dataclass
class EnhancedHDBSCANConfig:
    """Configuration for enhanced HDBSCAN integration."""
    # Memory optimization
    memory_optimization: bool = True
    max_memory_gb: float = 8.0
    memory_cleanup_threshold: float = 0.8
    
    # Hyperparameter optimization
    hyperparameter_optimization: bool = True
    optimization_strategy: str = "hybrid"  # "grid", "tpe", "hybrid"
    n_trials: int = 50
    primary_metric: str = "silhouette"
    
    # Vectorized processing
    vectorized_processing: bool = True
    enable_vectorbt: bool = True
    enable_gpu: bool = False
    enable_parallel: bool = True
    
    # Feature generation
    enable_entropy_features: bool = True
    enable_spectral_features: bool = True
    enable_regime_features: bool = True
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    detailed_logging: bool = True

class EnhancedHDBSCANIntegration:
    """
    Enhanced HDBSCAN integration with comprehensive optimizations.
    
    Integrates:
    - Memory & Data Processing Optimization
    - Hyperparameter Optimization
    - Vectorized Computations
    - Feature Generation from feature_generation/ system
    """
    
    def __init__(self, config: Optional[EnhancedHDBSCANConfig] = None):
        """Initialize the enhanced HDBSCAN integration."""
        self.config = config or EnhancedHDBSCANConfig()
        
        # Initialize optimization components
        self.memory_optimizer = None
        self.hyperparameter_optimizer = None
        self.vectorized_processor = None
        
        # Initialize feature generators
        self.entropy_generators = []
        self.spectral_generators = []
        self.regime_generators = []
        
        # Performance tracking
        self.performance_stats = {
            'total_processing_time': 0.0,
            'memory_optimizations': 0,
            'hyperparameter_optimizations': 0,
            'vectorized_operations': 0,
            'feature_generations': 0,
            'clustering_operations': 0
        }
        
        # Initialize components
        self._initialize_components()
        
        logger.info("✅ EnhancedHDBSCANIntegration initialized")
    
    def _initialize_components(self):
        """Initialize all optimization components."""
        # Initialize memory optimizer
        if self.config.memory_optimization:
            self.memory_optimizer = create_enhanced_memory_optimizer(
                max_memory_gb=self.config.max_memory_gb,
                enable_memory_optimization=True,
                enable_data_validation=True,
                enable_safe_operations=True,
                enable_memory_monitoring=True
            )
        
        # Initialize hyperparameter optimizer
        if self.config.hyperparameter_optimization:
            self.hyperparameter_optimizer = create_enhanced_hyperparameter_optimizer(
                optimization_strategy=self.config.optimization_strategy,
                n_trials=self.config.n_trials,
                primary_metric=self.config.primary_metric,
                enable_parallel=self.config.enable_parallel,
                memory_efficient=True
            )
        
        # Initialize vectorized processor
        if self.config.vectorized_processing:
            self.vectorized_processor = create_enhanced_vectorized_processor(
                enable_vectorbt=self.config.enable_vectorbt,
                enable_gpu=self.config.enable_gpu,
                enable_parallel=self.config.enable_parallel,
                memory_efficient=True,
                max_memory_gb=self.config.max_memory_gb
            )
        
        # Initialize feature generators
        if self.config.enable_entropy_features:
            self.entropy_generators = create_default_entropy_generators()
        
        if self.config.enable_spectral_features:
            self.spectral_generators = create_default_spectral_wavelet_generators()
        
        if self.config.enable_regime_features:
            self.regime_generators = create_default_regime_generators()
    
    def process_data_with_optimizations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data with comprehensive optimizations."""
        start_time = time.time()
        
        logger.info(f"🚀 Processing data with optimizations: {data.shape}")
        
        # Step 1: Memory optimization
        if self.memory_optimizer:
            with self.memory_optimizer.memory_monitor("data_processing"):
                data = self.memory_optimizer.optimize_dataframe(data)
                self.performance_stats['memory_optimizations'] += 1
        
        # Step 2: Feature generation using feature_generation/ system
        features_df = self._generate_features(data)
        self.performance_stats['feature_generations'] += 1
        
        # Step 3: Vectorized processing
        if self.vectorized_processor:
            features_df = self._apply_vectorized_processing(features_df)
            self.performance_stats['vectorized_operations'] += 1
        
        # Step 4: Final memory optimization
        if self.memory_optimizer:
            features_df = self.memory_optimizer.optimize_dataframe(features_df)
        
        # Update performance stats
        processing_time = time.time() - start_time
        self.performance_stats['total_processing_time'] += processing_time
        
        logger.info(f"✅ Data processing completed: {processing_time:.2f}s")
        
        return features_df
    
    def _generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features using the feature_generation/ system."""
        features_df = data.copy()
        
        # Generate entropy features
        if self.entropy_generators:
            for generator in self.entropy_generators:
                try:
                    entropy_features = generator.generate(data)
                    if isinstance(entropy_features, pd.DataFrame):
                        features_df = pd.concat([features_df, entropy_features], axis=1)
                except Exception as e:
                    logger.warning(f"⚠️ Entropy feature generation failed: {e}")
        
        # Generate spectral features
        if self.spectral_generators:
            for generator in self.spectral_generators:
                try:
                    spectral_features = generator.generate(data)
                    if isinstance(spectral_features, pd.DataFrame):
                        features_df = pd.concat([features_df, spectral_features], axis=1)
                except Exception as e:
                    logger.warning(f"⚠️ Spectral feature generation failed: {e}")
        
        # Generate regime features
        if self.regime_generators:
            for generator in self.regime_generators:
                try:
                    regime_features = generator.generate(data)
                    if isinstance(regime_features, pd.DataFrame):
                        features_df = pd.concat([features_df, regime_features], axis=1)
                except Exception as e:
                    logger.warning(f"⚠️ Regime feature generation failed: {e}")
        
        return features_df
    
    def _apply_vectorized_processing(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply vectorized processing optimizations."""
        if not self.vectorized_processor:
            return features_df
        
        # Apply vectorized mathematical operations
        math_operations = ['log', 'sqrt', 'abs']
        features_df = self.vectorized_processor.vectorized_mathematical_operations(
            features_df, math_operations
        )
        
        # Apply vectorized rolling operations
        rolling_configs = [
            {'type': 'rolling', 'name': 'volatility', 'params': {'operation': 'std', 'window': 20}},
            {'type': 'rolling', 'name': 'momentum', 'params': {'operation': 'mean', 'window': 10}},
            {'type': 'rolling', 'name': 'trend', 'params': {'operation': 'corr', 'window': 30}}
        ]
        
        features_df = self.vectorized_processor.vectorized_feature_engineering(
            features_df, rolling_configs
        )
        
        return features_df
    
    def optimize_hyperparameters(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Optimize HDBSCAN hyperparameters."""
        if not self.hyperparameter_optimizer:
            logger.warning("⚠️ Hyperparameter optimizer not available")
            return {}
        
        logger.info("🔍 Starting hyperparameter optimization")
        
        # Optimize hyperparameters
        optimization_results = self.hyperparameter_optimizer.optimize_hyperparameters(features_df)
        
        self.performance_stats['hyperparameter_optimizations'] += 1
        
        logger.info(f"✅ Hyperparameter optimization completed: "
                   f"Best score: {optimization_results['best_score']:.3f}")
        
        return optimization_results
    
    def perform_optimized_clustering(self, features_df: pd.DataFrame, 
                                   hdbscan_params: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform optimized HDBSCAN clustering."""
        start_time = time.time()
        
        # Use optimized parameters if available
        if hdbscan_params is None:
            hdbscan_params = {
                'min_cluster_size': 10,
                'min_samples': 5,
                'cluster_selection_epsilon': 0.0,
                'cluster_selection_method': 'eom',
                'metric': 'euclidean'
            }
        
        # Perform clustering with vectorized optimization
        if self.vectorized_processor:
            cluster_labels, clustering_info = self.vectorized_processor.optimized_hdbscan_clustering(
                features_df, **hdbscan_params
            )
        else:
            # Fallback to standard HDBSCAN
            import hdbscan
            clusterer = hdbscan.HDBSCAN(**hdbscan_params)
            cluster_labels = clusterer.fit_predict(features_df)
            clustering_info = {
                'clusterer': clusterer,
                'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                'n_noise_points': list(cluster_labels).count(-1)
            }
        
        # Update performance stats
        clustering_time = time.time() - start_time
        self.performance_stats['clustering_operations'] += 1
        self.performance_stats['total_processing_time'] += clustering_time
        
        logger.info(f"✅ Clustering completed: {clustering_time:.2f}s, "
                   f"{clustering_info['n_clusters']} clusters found")
        
        return cluster_labels, clustering_info
    
    def get_comprehensive_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add memory optimizer stats
        if self.memory_optimizer:
            memory_stats = self.memory_optimizer.get_memory_stats()
            stats['memory_optimizer_stats'] = memory_stats
        
        # Add hyperparameter optimizer stats
        if self.hyperparameter_optimizer:
            hyperparameter_stats = self.hyperparameter_optimizer.get_optimization_results()
            stats['hyperparameter_optimizer_stats'] = hyperparameter_stats
        
        # Add vectorized processor stats
        if self.vectorized_processor:
            vectorized_stats = self.vectorized_processor.get_performance_stats()
            stats['vectorized_processor_stats'] = vectorized_stats
        
        return stats
    
    def reset_performance_stats(self):
        """Reset all performance statistics."""
        self.performance_stats = {
            'total_processing_time': 0.0,
            'memory_optimizations': 0,
            'hyperparameter_optimizations': 0,
            'vectorized_operations': 0,
            'feature_generations': 0,
            'clustering_operations': 0
        }
        
        # Reset component stats
        if self.memory_optimizer:
            self.memory_optimizer.reset_stats()
        
        if self.hyperparameter_optimizer:
            self.hyperparameter_optimizer.reset_optimization()
        
        if self.vectorized_processor:
            self.vectorized_processor.reset_stats()

# Convenience function
def create_enhanced_hdbscan_integration(
    memory_optimization: bool = True,
    hyperparameter_optimization: bool = True,
    vectorized_processing: bool = True,
    enable_vectorbt: bool = True,
    enable_gpu: bool = False,
    enable_parallel: bool = True,
    max_memory_gb: float = 8.0,
    optimization_strategy: str = "hybrid",
    n_trials: int = 50,
    primary_metric: str = "silhouette"
) -> EnhancedHDBSCANIntegration:
    """
    Create an enhanced HDBSCAN integration with specified configuration.
    
    Args:
        memory_optimization: Enable memory optimization
        hyperparameter_optimization: Enable hyperparameter optimization
        vectorized_processing: Enable vectorized processing
        enable_vectorbt: Enable VectorBT optimization
        enable_gpu: Enable GPU acceleration
        enable_parallel: Enable parallel processing
        max_memory_gb: Maximum memory usage in GB
        optimization_strategy: Hyperparameter optimization strategy
        n_trials: Number of optimization trials
        primary_metric: Primary evaluation metric
        
    Returns:
        EnhancedHDBSCANIntegration instance
    """
    config = EnhancedHDBSCANConfig(
        memory_optimization=memory_optimization,
        hyperparameter_optimization=hyperparameter_optimization,
        vectorized_processing=vectorized_processing,
        enable_vectorbt=enable_vectorbt,
        enable_gpu=enable_gpu,
        enable_parallel=enable_parallel,
        max_memory_gb=max_memory_gb,
        optimization_strategy=optimization_strategy,
        n_trials=n_trials,
        primary_metric=primary_metric
    )
    
    return EnhancedHDBSCANIntegration(config)
