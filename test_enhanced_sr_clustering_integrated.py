#!/usr/bin/env python3
"""
Test script for the enhanced SR clustering component with all integrated features.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_enhanced_sr_clustering():
    """Test the enhanced SR clustering component with all features."""
    try:
        from src.training.steps.market_analysis.components.sr_clustering import (
            SRClusteringComponent,
            EnhancedSRClusteringConfig
        )

        logger.info("🚀 Starting Enhanced SR Clustering Test")
        logger.info("=" * 60)

        # Initialize component
        component = SRClusteringComponent()

        # Test configuration with all enhanced features
        test_config = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframe': '15m',
            'direction': 'longs',
            'execution_mode': 'full',
            
            # Enhanced features
            'enable_hardware_optimization': True,
            'enable_vectorbt_optimization': True,
            'enable_memory_optimization': True,
            'enable_gpu_acceleration': True,
            'enable_data_leakage_detection': True,
            'enable_explainability': True,
            'enable_hpo_optimization': True,
            'enable_advanced_feature_engineering': True,
            'enable_regime_aware_clustering': True,
            'enable_ensemble_clustering': True,
            'clustering_algorithm': 'ensemble',
            'hpo_trials': 20
        }

        logger.info("📊 Test Configuration:")
        for key, value in test_config.items():
            logger.info(f"  {key}: {value}")

        logger.info("\n🔗 Executing Enhanced SR Clustering...")
        result = await component.execute(test_config)

        # Display results
        logger.info("\n📈 Clustering Results:")
        logger.info(f"  Success: {result['success']}")
        logger.info(f"  Total Clusters: {result['metrics']['total_clusters']}")
        logger.info(f"  Clustering Efficiency: {result['metrics']['clustering_efficiency']:.3f}")
        
        if 'clustering_result' in result:
            clustering_result = result['clustering_result']
            
            # Display quality metrics
            if 'quality_metrics' in clustering_result:
                quality_metrics = clustering_result['quality_metrics']
                logger.info("\n📊 Quality Metrics:")
                for key, value in quality_metrics.items():
                    if isinstance(value, float):
                        logger.info(f"  {key}: {value:.3f}")
                    else:
                        logger.info(f"  {key}: {value}")
            
            # Display performance metrics
            if 'performance_metrics' in clustering_result:
                perf_metrics = clustering_result['performance_metrics']
                logger.info("\n⚡ Performance Metrics:")
                for key, value in perf_metrics.items():
                    if isinstance(value, float):
                        logger.info(f"  {key}: {value:.3f}")
                    else:
                        logger.info(f"  {key}: {value}")
            
            # Display hardware metrics
            if 'hardware_metrics' in clustering_result:
                hw_metrics = clustering_result['hardware_metrics']
                logger.info("\n🖥️ Hardware Metrics:")
                for key, value in hw_metrics.items():
                    logger.info(f"  {key}: {value}")
            
            # Display data leakage results
            if 'data_leakage_results' in clustering_result:
                leakage_results = clustering_result['data_leakage_results']
                if leakage_results:
                    logger.info("\n🔍 Data Leakage Detection:")
                    for key, value in leakage_results.items():
                        logger.info(f"  {key}: {value}")
            
            # Display regime analysis
            if 'regime_analysis' in clustering_result:
                regime_analysis = clustering_result['regime_analysis']
                if regime_analysis:
                    logger.info("\n📊 Regime Analysis:")
                    for regime, analysis in regime_analysis.items():
                        logger.info(f"  {regime}: {analysis['count']} levels")
                        logger.info(f"    Price Range: {analysis['price_range']:.4f}")
                        logger.info(f"    Avg Strength: {analysis['strength_mean']:.3f}")
            
            # Display explainability results
            if 'explainability_results' in clustering_result:
                explain_results = clustering_result['explainability_results']
                if explain_results:
                    logger.info("\n🔍 Explainability Results:")
                    for key, value in explain_results.items():
                        if isinstance(value, float):
                            logger.info(f"  {key}: {value:.3f}")
                        else:
                            logger.info(f"  {key}: {value}")
            
            # Display enhancement features used
            if 'metadata' in clustering_result:
                metadata = clustering_result['metadata']
                logger.info(f"\n🔧 Enhancement Version: {metadata.get('enhancement_version', 'Unknown')}")
                logger.info("Features Used:")
                features_used = metadata.get('features_used', {})
                for feature, enabled in features_used.items():
                    status = "✅" if enabled else "❌"
                    logger.info(f"  {status} {feature}")

        logger.info("\n✅ Enhanced SR Clustering Test Completed Successfully!")
        return result

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        logger.error(f"❌ Error details: {traceback.format_exc()}")
        return None

async def test_individual_features():
    """Test individual enhanced features."""
    try:
        from src.training.steps.market_analysis.components.sr_clustering import (
            SRClusteringComponent,
            EnhancedSRClusteringConfig
        )

        logger.info("\n🧪 Testing Individual Enhanced Features")
        logger.info("=" * 50)

        component = SRClusteringComponent()

        # Test different clustering algorithms
        algorithms = ['hdbscan', 'dbscan', 'kmeans', 'spectral', 'gmm', 'ensemble']
        
        for algorithm in algorithms:
            logger.info(f"\n🔬 Testing {algorithm} clustering...")
            
            test_config = {
                'symbol': 'BTCUSDT',
                'exchange': 'binance',
                'timeframe': '1h',
                'direction': 'shorts',
                'execution_mode': 'light',
                'clustering_algorithm': algorithm,
                'enable_hardware_optimization': False,  # Disable for faster testing
                'enable_vectorbt_optimization': False,
                'enable_data_leakage_detection': False,
                'enable_explainability': False
            }
            
            try:
                result = await component.execute(test_config)
                if result['success']:
                    clusters = result['metrics']['total_clusters']
                    efficiency = result['metrics']['clustering_efficiency']
                    logger.info(f"  ✅ {algorithm}: {clusters} clusters, efficiency: {efficiency:.3f}")
                else:
                    logger.warning(f"  ⚠️ {algorithm}: Failed - {result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"  ❌ {algorithm}: Error - {e}")

        logger.info("\n✅ Individual Feature Testing Completed!")

    except Exception as e:
        logger.error(f"❌ Individual feature testing failed: {e}")

async def main():
    """Main test function."""
    logger.info("🚀 Enhanced SR Clustering Component Test Suite")
    logger.info("=" * 60)
    
    # Test 1: Full enhanced clustering
    logger.info("\n📋 Test 1: Full Enhanced Clustering")
    await test_enhanced_sr_clustering()
    
    # Test 2: Individual features
    logger.info("\n📋 Test 2: Individual Features")
    await test_individual_features()
    
    logger.info("\n🎉 All Tests Completed!")

if __name__ == "__main__":
    asyncio.run(main())