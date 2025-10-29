"""
Enhanced HDBSCAN Regime Discovery

This module integrates all the enhanced components for comprehensive regime discovery
with proper parameter optimization, quality assessment, and temporal stability.
Enhanced with VectorBT optimizations and comprehensive tprint logging.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from dataclasses import dataclass
import warnings

# Import VectorBT optimizations
try:
    from src.vectorbt import (
        vbt, rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, VECTORBT_AVAILABLE
    )
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None

# Import tprint system
from src.utils.tprint import (
    tprint, tprint_data_preview, tprint_data_format, 
    tprint_performance
)

# Import hardware optimizations
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer

# Import our enhanced components
from unified_config import UnifiedHDBSCANConfig, create_unified_config
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import ClusterQualityAssessor, ClusterQualityMetrics
from feature_engineering import EnhancedFeatureEngineeringPipeline, FeatureEngineeringConfig
from chunked_processing import EnhancedChunkedProcessor, ChunkProcessingConfig

# Import HDBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    hdbscan = None

logger = logging.getLogger(__name__)


@dataclass
class EnhancedRegimeResult:
    """Enhanced regime discovery result with comprehensive metrics."""
    labels: np.ndarray
    n_clusters: int
    n_noise_points: int
    noise_ratio: float
    quality_metrics: ClusterQualityMetrics
    processing_info: Dict[str, Any]
    config: UnifiedHDBSCANConfig
    cluster_profiles: Optional[Dict[str, Any]] = None
    temporal_stability: Optional[float] = None
    economic_separation: Optional[float] = None
    validation_results: Optional[Dict[str, Any]] = None


class EnhancedHDBSCANRegimeDiscovery:
    """
    Enhanced HDBSCAN regime discovery with comprehensive quality assessment
    and intelligent parameter optimization.
    """
    
    def __init__(self, config: Optional[UnifiedHDBSCANConfig] = None):
        self.config = config or create_unified_config()
        self.quality_assessor = ClusterQualityAssessor()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.hardware_manager = UnifiedHardwareManager()
        self.memory_optimizer = M1MemoryOptimizer()
        
        # Initialize VectorBT optimizations
        self.vectorbt_available = VECTORBT_AVAILABLE
        if self.vectorbt_available:
            tprint("✅ VectorBT optimizations enabled for regime discovery")
        else:
            tprint("⚠️ VectorBT not available - using standard methods")
        
        # Initialize feature engineering pipeline
        feature_config = FeatureEngineeringConfig(
            correlation_threshold=self.config.feature_engineering.correlation_threshold,
            enable_feature_selection=self.config.feature_engineering.enable_feature_selection,
            feature_selection_method=self.config.feature_engineering.feature_selection_method,
            max_features=self.config.feature_engineering.max_features,
            enable_regime_features=self.config.feature_engineering.enable_regime_features,
            enable_entropy_features=self.config.feature_engineering.enable_entropy_features,
            enable_spectral_features=self.config.feature_engineering.enable_spectral_features,
            enable_temporal_features=self.config.feature_engineering.enable_temporal_features
        )
        self.feature_pipeline = EnhancedFeatureEngineeringPipeline(feature_config)
        
        tprint("🚀 Enhanced HDBSCAN regime discovery initialized")
        tprint_data_format("Configuration", self.config.__dict__)
        
        # Initialize chunked processor
        chunk_config = ChunkProcessingConfig(
            enable_chunked_processing=self.config.chunk_processing.enable_chunked_processing,
            chunk_size=self.config.chunk_processing.chunk_size,
            chunk_overlap=self.config.chunk_processing.chunk_overlap,
            enable_temporal_continuity=self.config.chunk_processing.enable_temporal_continuity,
            merge_similar_clusters=self.config.chunk_processing.merge_similar_clusters,
            similarity_threshold=self.config.chunk_processing.similarity_threshold
        )
        self.chunked_processor = EnhancedChunkedProcessor(chunk_config)
    
    def discover_regimes(self, 
                        data_df: pd.DataFrame,
                        timestamps: Optional[pd.Series] = None,
                        returns: Optional[np.ndarray] = None,
                        target: Optional[np.ndarray] = None) -> EnhancedRegimeResult:
        """
        Discover regimes using enhanced HDBSCAN clustering with VectorBT optimizations.
        
        Args:
            data_df: Input data DataFrame
            timestamps: Optional timestamps for temporal analysis
            returns: Optional returns data for economic analysis
            target: Optional target variable for supervised feature selection
            
        Returns:
            EnhancedRegimeResult with comprehensive clustering results
        """
        tprint("🚀 Starting enhanced regime discovery with VectorBT optimizations")
        tprint_data_preview(data_df, "Input data")
        
        start_time = time.time()
        
        tprint(f"📊 Data shape: {data_df.shape}")
        tprint(f"⚙️ Execution mode: {self.config.execution_mode.value}")
        
        # Step 1: Feature Engineering
        tprint("🔧 Step 1: VectorBT-optimized feature engineering")
        processed_features, feature_info = self.feature_pipeline.process_features(data_df, target)
        
        # Step 2: Determine processing strategy
        use_chunked = (
            self.config.chunk_processing.enable_chunked_processing and 
            len(processed_features) > self.config.chunk_processing.chunk_size
        )
        
        if use_chunked:
            tprint("🔄 Step 2: Chunked Processing with VectorBT optimizations")
            clustering_result = self._discover_regimes_chunked(
                processed_features, timestamps, returns
            )
        else:
            tprint("⚡ Step 2: Standard Processing with VectorBT optimizations")
            clustering_result = self._discover_regimes_standard(
                processed_features, timestamps, returns
            )
        
        # Step 3: Quality Assessment
        tprint("📊 Step 3: Comprehensive quality assessment")
        quality_metrics = self.quality_assessor.assess_clustering_quality(
            cluster_labels=clustering_result['labels'],
            features=processed_features.values,
            clusterer=clustering_result.get('clusterer'),
            timestamps=timestamps,
            returns=returns
        )
        
        # Step 4: Validation
        tprint("✅ Step 4: Quality validation")
        validation_results = self.config.validate_clustering_quality(
            clustering_result['labels'], 
            {
                'dbcv': quality_metrics.dbcv_score or 0.0,
                'silhouette_score': quality_metrics.silhouette_score or 0.0,
                'temporal_stability': quality_metrics.temporal_stability or 0.0,
                'economic_separation': quality_metrics.economic_separation or 0.0,
                'noise_ratio': quality_metrics.noise_ratio or 0.0
            }
        )
        
        # Step 5: Generate cluster profiles
        tprint("📋 Step 5: Generating cluster profiles")
        cluster_profiles = self._generate_cluster_profiles(
            clustering_result['labels'], 
            processed_features, 
            returns
        )
        
        # Compile processing info
        processing_info = {
            'feature_engineering': feature_info,
            'clustering': clustering_result,
            'processing_time': time.time() - start_time,
            'use_chunked_processing': use_chunked,
            'config': self.config.to_dict()
        }
        
        # Create result
        result = EnhancedRegimeResult(
            labels=clustering_result['labels'],
            n_clusters=clustering_result['n_clusters'],
            n_noise_points=clustering_result['n_noise_points'],
            noise_ratio=clustering_result['noise_ratio'],
            quality_metrics=quality_metrics,
            processing_info=processing_info,
            config=self.config,
            cluster_profiles=cluster_profiles,
            temporal_stability=quality_metrics.temporal_stability,
            economic_separation=quality_metrics.economic_separation,
            validation_results=validation_results
        )
        
        # Log results with tprint
        processing_time = processing_info['processing_time']
        tprint_performance(f"Regime discovery completed in {processing_time:.2f}s")
        tprint(f"📊 Clusters: {result.n_clusters}, Noise ratio: {result.noise_ratio:.3f}")
        tprint(f"✅ Quality validation passed: {validation_results['overall_passed']}")
        
        if not validation_results['overall_passed']:
            tprint("⚠️ Quality validation failed - see recommendations")
            for rec in validation_results['recommendations']:
                tprint(f"  - {rec}")
        
        tprint_data_preview(result.labels, "Final cluster labels")
        
        return result
    
    def _discover_regimes_standard(self, 
                                  features: pd.DataFrame,
                                  timestamps: Optional[pd.Series] = None,
                                  returns: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Discover regimes using standard (non-chunked) processing."""
        try:
            # Get adaptive parameters
            adaptive_params = self.config.get_adaptive_parameters(
                len(features), len(features.columns)
            )
            
            self.logger.info(f"Using adaptive parameters: {adaptive_params}")
            
            # Create HDBSCAN clusterer
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=adaptive_params['min_cluster_size'],
                min_samples=adaptive_params['min_samples'],
                cluster_selection_epsilon=adaptive_params['cluster_selection_epsilon'],
                cluster_selection_method=adaptive_params['cluster_selection_method'],
                metric=adaptive_params['metric'],
                alpha=adaptive_params['alpha']
            )
            
            # Perform clustering
            cluster_labels = clusterer.fit_predict(features.values)
            
            # Calculate basic metrics
            unique_labels = np.unique(cluster_labels)
            n_clusters = len(unique_labels[unique_labels != -1])
            n_noise_points = np.sum(cluster_labels == -1)
            noise_ratio = n_noise_points / len(cluster_labels) if len(cluster_labels) > 0 else 0.0
            
            return {
                'labels': cluster_labels,
                'n_clusters': n_clusters,
                'n_noise_points': n_noise_points,
                'noise_ratio': noise_ratio,
                'clusterer': clusterer,
                'parameters': adaptive_params
            }
            
        except Exception as e:
            self.logger.error(f"Standard clustering failed: {e}")
            # Return fallback result
            return {
                'labels': np.full(len(features), -1),
                'n_clusters': 0,
                'n_noise_points': len(features),
                'noise_ratio': 1.0,
                'clusterer': None,
                'parameters': {}
            }
    
    def _discover_regimes_chunked(self, 
                                 features: pd.DataFrame,
                                 timestamps: Optional[pd.Series] = None,
                                 returns: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Discover regimes using chunked processing."""
        try:
            # Create clustering function
            def clustering_func(features_chunk):
                # Handle both DataFrame and numpy array inputs
                if hasattr(features_chunk, 'columns'):
                    n_features = len(features_chunk.columns)
                else:
                    n_features = features_chunk.shape[1] if len(features_chunk.shape) > 1 else 1
                
                adaptive_params = self.config.get_adaptive_parameters(
                    len(features_chunk), n_features
                )
                
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=adaptive_params['min_cluster_size'],
                    min_samples=adaptive_params['min_samples'],
                    cluster_selection_epsilon=adaptive_params['cluster_selection_epsilon'],
                    cluster_selection_method=adaptive_params['cluster_selection_method'],
                    metric=adaptive_params['metric'],
                    alpha=adaptive_params['alpha']
                )
                
                # Perform clustering
                if hasattr(features_chunk, 'values'):
                    labels = clusterer.fit_predict(features_chunk.values)
                else:
                    labels = clusterer.fit_predict(features_chunk)
                return {
                    'labels': labels,
                    'clusterer': clusterer,
                    'parameters': adaptive_params
                }
            
            # Process chunks
            chunked_result = self.chunked_processor.process_chunks(
                features.values, clustering_func, timestamps
            )
            
            return chunked_result
            
        except Exception as e:
            self.logger.error(f"Chunked clustering failed: {e}")
            # Return fallback result
            return {
                'labels': np.full(len(features), -1),
                'n_clusters': 0,
                'n_noise_points': len(features),
                'noise_ratio': 1.0,
                'clusterer': None,
                'parameters': {}
            }
    
    def _generate_cluster_profiles(self, 
                                  cluster_labels: np.ndarray,
                                  features: pd.DataFrame,
                                  returns: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Generate detailed cluster profiles."""
        profiles = {}
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            
            cluster_mask = cluster_labels == label
            cluster_features = features[cluster_mask]
            
            # Basic statistics
            profile = {
                'cluster_id': int(label),
                'size': len(cluster_features),
                'size_ratio': len(cluster_features) / len(cluster_labels),
                'feature_means': cluster_features.mean().to_dict(),
                'feature_stds': cluster_features.std().to_dict()
            }
            
            # Economic profile if returns available
            if returns is not None and len(returns) == len(cluster_labels):
                cluster_returns = returns[cluster_mask]
                profile['economic'] = {
                    'avg_return': float(np.mean(cluster_returns)),
                    'volatility': float(np.std(cluster_returns)),
                    'sharpe_ratio': float(np.mean(cluster_returns) / np.std(cluster_returns)) if np.std(cluster_returns) > 0 else 0.0,
                    'min_return': float(np.min(cluster_returns)),
                    'max_return': float(np.max(cluster_returns))
                }
            
            profiles[f'cluster_{label}'] = profile
        
        return profiles
    
    def optimize_parameters(self, 
                           data_df: pd.DataFrame,
                           parameter_ranges: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """
        Optimize clustering parameters using grid search.
        
        Args:
            data_df: Input data DataFrame
            parameter_ranges: Optional parameter ranges to search
            
        Returns:
            Dictionary with optimization results
        """
        if parameter_ranges is None:
            parameter_ranges = {
                'min_cluster_size': [10, 20, 30, 50],
                'min_samples': [5, 10, 15, 20],
                'cluster_selection_epsilon': [0.0, 0.1, 0.2, 0.5],
                'metric': ['euclidean', 'manhattan', 'cosine']
            }
        
        self.logger.info("Starting parameter optimization")
        
        # Process features once
        processed_features, _ = self.feature_pipeline.process_features(data_df)
        
        best_score = -1
        best_params = {}
        best_result = None
        
        # Grid search
        for min_cluster_size in parameter_ranges['min_cluster_size']:
            for min_samples in parameter_ranges['min_samples']:
                for epsilon in parameter_ranges['cluster_selection_epsilon']:
                    for metric in parameter_ranges['metric']:
                        try:
                            # Create clusterer with current parameters
                            clusterer = hdbscan.HDBSCAN(
                                min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                cluster_selection_epsilon=epsilon,
                                metric=metric,
                                n_jobs=self.config.n_jobs
                            )
                            
                            # Perform clustering
                            labels = clusterer.fit_predict(processed_features.values)
                            
                            # Calculate quality metrics
                            quality_metrics = self.quality_assessor.assess_clustering_quality(
                                labels, processed_features.values, clusterer
                            )
                            
                            # Use DBCV as primary score
                            score = quality_metrics.dbcv_score or 0.0
                            
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    'min_cluster_size': min_cluster_size,
                                    'min_samples': min_samples,
                                    'cluster_selection_epsilon': epsilon,
                                    'metric': metric
                                }
                                best_result = {
                                    'labels': labels,
                                    'quality_metrics': quality_metrics,
                                    'clusterer': clusterer
                                }
                        
                        except Exception as e:
                            self.logger.warning(f"Parameter combination failed: {e}")
                            continue
        
        self.logger.info(f"Parameter optimization completed. Best score: {best_score:.4f}")
        self.logger.info(f"Best parameters: {best_params}")
        
        return {
            'best_score': best_score,
            'best_params': best_params,
            'best_result': best_result
        }
    
    def get_recommendations(self, result: EnhancedRegimeResult) -> List[str]:
        """Get recommendations for improving clustering results."""
        recommendations = []
        
        # Quality-based recommendations
        if result.quality_metrics.dbcv_score and result.quality_metrics.dbcv_score < 0.3:
            recommendations.append("DBCV score is low - consider adjusting min_cluster_size or min_samples")
        
        if result.quality_metrics.silhouette_score and result.quality_metrics.silhouette_score < 0.2:
            recommendations.append("Silhouette score is low - consider using different distance metric")
        
        if result.quality_metrics.temporal_stability and result.quality_metrics.temporal_stability < 0.7:
            recommendations.append("Temporal stability is low - consider enabling temporal smoothing")
        
        if result.quality_metrics.economic_separation and result.quality_metrics.economic_separation < 0.2:
            recommendations.append("Economic separation is low - consider adding more regime-specific features")
        
        if result.noise_ratio > 0.4:
            recommendations.append("High noise ratio - consider reducing min_cluster_size or min_samples")
        
        # Cluster count recommendations
        if result.n_clusters == 0:
            recommendations.append("No clusters found - consider reducing min_cluster_size")
        elif result.n_clusters > 10:
            recommendations.append("Too many clusters - consider increasing min_cluster_size")
        
        # Configuration recommendations
        if result.config.execution_mode.value == "light" and result.n_clusters < 3:
            recommendations.append("Consider using 'standard' or 'full' execution mode for better results")
        
        return recommendations


def create_enhanced_regime_discovery(config: Optional[UnifiedHDBSCANConfig] = None) -> EnhancedHDBSCANRegimeDiscovery:
    """Factory function to create an enhanced regime discovery instance."""
    return EnhancedHDBSCANRegimeDiscovery(config)