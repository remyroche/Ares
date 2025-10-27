"""
Comprehensive Test for Enhanced HDBSCAN Integration

This test validates all the enhanced components working together:
- Unified configuration management
- Proper DBCV calculation
- Enhanced feature engineering
- Intelligent chunked processing
- Temporal stability validation
- Economic separation assessment
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, Any
import warnings

# Import our enhanced components
from src.training.steps.market_analysis.hdbscan_clustering.unified_config import (
    UnifiedHDBSCANConfig, create_unified_config, ExecutionMode
)
from src.training.steps.market_analysis.hdbscan_clustering.quality_assessment import (
    ComprehensiveQualityAssessor, QualityMetrics
)
from src.training.steps.market_analysis.hdbscan_clustering.feature_engineering import (
    EnhancedFeatureEngineeringPipeline, FeatureEngineeringConfig
)
from src.training.steps.market_analysis.hdbscan_clustering.chunked_processing import (
    EnhancedChunkedProcessor, ChunkProcessingConfig
)
from src.training.steps.market_analysis.hdbscan_clustering.enhanced_regime_discovery import (
    EnhancedHDBSCANRegimeDiscovery, EnhancedRegimeResult
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_data(n_samples: int = 1000, n_features: int = 20, 
                    n_regimes: int = 3, noise_level: float = 0.1) -> pd.DataFrame:
    """Create synthetic test data with known regimes."""
    np.random.seed(42)
    
    # Create regime labels
    regime_labels = np.random.choice(n_regimes, n_samples)
    
    # Create features with regime-specific characteristics
    features = np.zeros((n_samples, n_features))
    
    for regime in range(n_regimes):
        regime_mask = regime_labels == regime
        
        # Each regime has different mean and variance
        regime_mean = np.random.normal(0, 2, n_features)
        regime_std = np.random.uniform(0.5, 2.0, n_features)
        
        # Generate regime-specific data
        regime_data = np.random.normal(regime_mean, regime_std, (np.sum(regime_mask), n_features))
        features[regime_mask] = regime_data
    
    # Add noise
    noise = np.random.normal(0, noise_level, features.shape)
    features += noise
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(features, columns=feature_names)
    
    # Add price-like features
    df['close'] = 100 + np.cumsum(np.random.normal(0, 0.01, n_samples))
    df['high'] = df['close'] * (1 + np.abs(np.random.normal(0, 0.02, n_samples)))
    df['low'] = df['close'] * (1 - np.abs(np.random.normal(0, 0.02, n_samples)))
    df['volume'] = np.random.uniform(1000, 10000, n_samples)
    
    # Add timestamps
    df.index = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    
    return df, regime_labels


def test_unified_configuration():
    """Test unified configuration management."""
    logger.info("Testing unified configuration management...")
    
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


def test_quality_assessment():
    """Test comprehensive quality assessment."""
    logger.info("Testing quality assessment...")
    
    assessor = ComprehensiveQualityAssessor()
    
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
    
    assert isinstance(quality_metrics, QualityMetrics)
    assert quality_metrics.n_clusters >= 0
    assert quality_metrics.n_noise_points >= 0
    assert 0 <= quality_metrics.noise_ratio <= 1
    
    logger.info("✓ Quality assessment test passed")


def test_feature_engineering():
    """Test enhanced feature engineering pipeline."""
    logger.info("Testing feature engineering...")
    
    # Create test data
    data_df, _ = create_test_data(n_samples=500, n_features=10)
    
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


def test_chunked_processing():
    """Test enhanced chunked processing."""
    logger.info("Testing chunked processing...")
    
    # Create test data
    data_df, _ = create_test_data(n_samples=2000, n_features=15)
    
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
        timestamps=data_df.index
    )
    
    assert 'labels' in result
    assert 'n_clusters' in result
    assert 'processing_time' in result
    assert len(result['labels']) == len(data_df)
    
    logger.info(f"✓ Chunked processing test passed - {result['n_clusters']} clusters found")


def test_enhanced_regime_discovery():
    """Test the complete enhanced regime discovery system."""
    logger.info("Testing enhanced regime discovery...")
    
    # Create test data
    data_df, true_regimes = create_test_data(n_samples=1000, n_features=20, n_regimes=3)
    
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
        assert isinstance(result.quality_metrics, QualityMetrics)
        assert isinstance(result.validation_results, dict)
        
        # Test recommendations
        recommendations = discovery.get_recommendations(result)
        assert isinstance(recommendations, list)
        
        logger.info(f"  {mode} mode: {result.n_clusters} clusters, noise ratio: {result.noise_ratio:.3f}")
        logger.info(f"  Quality validation passed: {result.validation_results['overall_passed']}")
    
    logger.info("✓ Enhanced regime discovery test passed")


def test_parameter_optimization():
    """Test parameter optimization."""
    logger.info("Testing parameter optimization...")
    
    # Create test data
    data_df, _ = create_test_data(n_samples=500, n_features=15)
    
    # Test optimization
    config = create_unified_config(execution_mode='light')
    discovery = EnhancedHDBSCANRegimeDiscovery(config)
    
    # Test with limited parameter ranges for speed
    parameter_ranges = {
        'min_cluster_size': [10, 20],
        'min_samples': [5, 10],
        'cluster_selection_epsilon': [0.0, 0.1],
        'metric': ['euclidean', 'manhattan']
    }
    
    optimization_result = discovery.optimize_parameters(data_df, parameter_ranges)
    
    assert 'best_score' in optimization_result
    assert 'best_params' in optimization_result
    assert 'best_result' in optimization_result
    
    logger.info(f"✓ Parameter optimization test passed - best score: {optimization_result['best_score']:.4f}")


def test_integration_workflow():
    """Test complete integration workflow."""
    logger.info("Testing complete integration workflow...")
    
    # Create larger test dataset
    data_df, true_regimes = create_test_data(n_samples=2000, n_features=25, n_regimes=4)
    
    # Use standard mode for comprehensive testing
    config = create_unified_config(execution_mode='standard')
    discovery = EnhancedHDBSCANRegimeDiscovery(config)
    
    # Complete workflow
    start_time = time.time()
    
    result = discovery.discover_regimes(
        data_df=data_df,
        timestamps=data_df.index,
        returns=data_df['close'].pct_change().values
    )
    
    processing_time = time.time() - start_time
    
    # Validate results
    assert result.n_clusters > 0, "No clusters found"
    assert result.noise_ratio < 0.8, "Too much noise"
    assert result.quality_metrics.dbcv_score is not None, "DBCV score not calculated"
    assert result.temporal_stability is not None, "Temporal stability not calculated"
    assert result.economic_separation is not None, "Economic separation not calculated"
    
    # Log comprehensive results
    logger.info("=== INTEGRATION TEST RESULTS ===")
    logger.info(f"Processing time: {processing_time:.2f}s")
    logger.info(f"Clusters found: {result.n_clusters}")
    logger.info(f"Noise ratio: {result.noise_ratio:.3f}")
    logger.info(f"DBCV score: {result.quality_metrics.dbcv_score:.4f}")
    logger.info(f"Silhouette score: {result.quality_metrics.silhouette_score:.4f}")
    logger.info(f"Temporal stability: {result.temporal_stability:.4f}")
    logger.info(f"Economic separation: {result.economic_separation:.4f}")
    logger.info(f"Quality validation passed: {result.validation_results['overall_passed']}")
    
    if not result.validation_results['overall_passed']:
        logger.warning("Quality validation failed:")
        for rec in result.validation_results['recommendations']:
            logger.warning(f"  - {rec}")
    
    # Test recommendations
    recommendations = discovery.get_recommendations(result)
    if recommendations:
        logger.info("Recommendations:")
        for rec in recommendations:
            logger.info(f"  - {rec}")
    
    logger.info("✓ Complete integration workflow test passed")


def run_all_tests():
    """Run all tests."""
    logger.info("Starting comprehensive HDBSCAN integration tests...")
    
    try:
        test_unified_configuration()
        test_quality_assessment()
        test_feature_engineering()
        test_chunked_processing()
        test_enhanced_regime_discovery()
        test_parameter_optimization()
        test_integration_workflow()
        
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