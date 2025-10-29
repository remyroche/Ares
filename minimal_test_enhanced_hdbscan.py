"""
Minimal Test for Enhanced HDBSCAN Components

This test directly imports and tests our enhanced components without
going through the full codebase import chain.
"""

import sys
import os
sys.path.append('/workspace')

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, Any
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_unified_config_direct():
    """Test unified configuration management directly."""
    logger.info("Testing unified configuration management...")
    
    # Import our modules directly
    sys.path.append('/workspace/src/training/steps/market_analysis/hdbscan_clustering')
    
    from unified_config import (
        UnifiedHDBSCANConfig, create_unified_config, ExecutionMode
    )
    
    # Test different execution modes
    for mode in ['light', 'standard', 'full', 'blank']:
        config = create_unified_config(execution_mode=mode)
        
        # Test adaptive parameters
        adaptive_params = config.get_adaptive_parameters(1000, 50)
        assert 'min_cluster_size' in adaptive_params
        assert 'min_samples' in adaptive_params
        assert 'metric' in adaptive_params
        
        # Test quality validation
        test_labels = np.array([0, 0, 1, 1, -1, -1])
        test_metrics = {
            'dbcv': 0.5,
            'silhouette_score': 0.3,
            'temporal_stability': 0.8,
            'economic_separation': 0.4,
            'noise_ratio': 0.33
        }
        
        validation = config.validate_clustering_quality(test_labels, test_metrics)
        assert 'overall_passed' in validation
        assert 'individual_checks' in validation
    
    logger.info("✓ Unified configuration management test passed")


def test_quality_assessment_direct():
    """Test comprehensive quality assessment directly."""
    logger.info("Testing quality assessment...")
    
    from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
        ClusterQualityAssessor, ClusterQualityMetrics
    )
    
    assessor = ClusterQualityAssessor()
    
    # Create test data
    n_samples = 500
    n_features = 10
    features = np.random.randn(n_samples, n_features)
    
    # Create test cluster labels
    cluster_labels = np.random.choice([0, 1, 2, -1], n_samples, p=[0.3, 0.3, 0.3, 0.1])
    
    # Test quality assessment
    quality_metrics = assessor.assess_clustering_quality(
        cluster_labels=cluster_labels,
        features=features,
        timestamps=pd.Series(pd.date_range('2023-01-01', periods=n_samples, freq='1H'))
    )
    
    assert isinstance(quality_metrics, ClusterQualityMetrics)
    assert quality_metrics.n_clusters >= 0
    assert quality_metrics.n_noise_points >= 0
    assert 0 <= quality_metrics.noise_ratio <= 1
    
    logger.info("✓ Quality assessment test passed")


def test_feature_engineering_direct():
    """Test enhanced feature engineering pipeline directly."""
    logger.info("Testing feature engineering...")
    
    from feature_engineering import (
        EnhancedFeatureEngineeringPipeline, FeatureEngineeringConfig
    )
    
    # Create test data
    n_samples = 500
    n_features = 10
    features = np.random.randn(n_samples, n_features)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    data_df = pd.DataFrame(features, columns=feature_names)
    
    # Add price-like features
    data_df['close'] = 100 + np.cumsum(np.random.normal(0, 0.01, n_samples))
    data_df['high'] = data_df['close'] * (1 + np.abs(np.random.normal(0, 0.02, n_samples)))
    data_df['low'] = data_df['close'] * (1 - np.abs(np.random.normal(0, 0.02, n_samples)))
    data_df['volume'] = np.random.uniform(1000, 10000, n_samples)
    
    # Test feature engineering
    config = FeatureEngineeringConfig(
        enable_feature_selection=True,
        max_features=15,
        enable_regime_features=True,
        enable_entropy_features=True,
        enable_spectral_features=True,
        enable_temporal_features=True
    )
    
    pipeline = EnhancedFeatureEngineeringPipeline(config)
    processed_features, processing_info = pipeline.process_features(data_df)
    
    assert isinstance(processed_features, pd.DataFrame)
    assert len(processed_features) == len(data_df)
    assert 'processing_steps' in processing_info
    assert 'final_features' in processing_info
    
    logger.info(f"✓ Feature engineering test passed - {processing_info['final_features']} features generated")


def test_chunked_processing_direct():
    """Test enhanced chunked processing directly."""
    logger.info("Testing chunked processing...")
    
    from chunked_processing import (
        EnhancedChunkedProcessor, ChunkProcessingConfig
    )
    
    # Create test data
    n_samples = 2000
    n_features = 15
    features = np.random.randn(n_samples, n_features)
    data_df = pd.DataFrame(features)
    
    # Test chunked processing
    config = ChunkProcessingConfig(
        enable_chunked_processing=True,
        chunk_size=500,
        chunk_overlap=0.1,
        enable_temporal_continuity=True,
        merge_similar_clusters=True
    )
    
    processor = EnhancedChunkedProcessor(config)
    
    # Create a simple clustering function
    def simple_clustering_func(features):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(features)
        return {'labels': labels}
    
    # Process chunks
    result = processor.process_chunks(
        data_df.values, 
        simple_clustering_func,
        timestamps=pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    )
    
    assert 'labels' in result
    assert 'n_clusters' in result
    assert 'processing_time' in result
    assert len(result['labels']) == len(data_df)
    
    logger.info(f"✓ Chunked processing test passed - {result['n_clusters']} clusters found")


def test_enhanced_regime_discovery_direct():
    """Test the complete enhanced regime discovery system directly."""
    logger.info("Testing enhanced regime discovery...")
    
    from enhanced_regime_discovery import (
        EnhancedHDBSCANRegimeDiscovery, EnhancedRegimeResult
    )
    from unified_config import create_unified_config
    from src.training.steps.market_analysis.clusters.cluster_quality_assessor import ClusterQualityMetrics
    
    # Create test data
    n_samples = 1000
    n_features = 20
    features = np.random.randn(n_samples, n_features)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    data_df = pd.DataFrame(features, columns=feature_names)
    
    # Add price-like features
    data_df['close'] = 100 + np.cumsum(np.random.normal(0, 0.01, n_samples))
    data_df['high'] = data_df['close'] * (1 + np.abs(np.random.normal(0, 0.02, n_samples)))
    data_df['low'] = data_df['close'] * (1 - np.abs(np.random.normal(0, 0.02, n_samples)))
    data_df['volume'] = np.random.uniform(1000, 10000, n_samples)
    data_df.index = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    
    # Test different execution modes
    for mode in ['light', 'standard']:
        logger.info(f"Testing {mode} mode...")
        
        config = create_unified_config(execution_mode=mode)
        discovery = EnhancedHDBSCANRegimeDiscovery(config)
        
        # Discover regimes
        result = discovery.discover_regimes(
            data_df=data_df,
            timestamps=data_df.index,
            returns=data_df['close'].pct_change().values
        )
        
        assert isinstance(result, EnhancedRegimeResult)
        assert len(result.labels) == len(data_df)
        assert result.n_clusters >= 0
        assert 0 <= result.noise_ratio <= 1
        assert isinstance(result.quality_metrics, ClusterQualityMetrics)
        assert isinstance(result.validation_results, dict)
        
        # Test recommendations
        recommendations = discovery.get_recommendations(result)
        assert isinstance(recommendations, list)
        
        logger.info(f"  {mode} mode: {result.n_clusters} clusters, noise ratio: {result.noise_ratio:.3f}")
        logger.info(f"  Quality validation passed: {result.validation_results['overall_passed']}")
    
    logger.info("✓ Enhanced regime discovery test passed")


def run_all_tests():
    """Run all tests."""
    logger.info("Starting enhanced HDBSCAN component tests...")
    
    try:
        test_unified_config_direct()
        test_quality_assessment_direct()
        test_feature_engineering_direct()
        test_chunked_processing_direct()
        test_enhanced_regime_discovery_direct()
        
        logger.info("🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)