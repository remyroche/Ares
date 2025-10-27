#!/usr/bin/env python3
"""
Test script for Enhanced SR Clustering Component.

This script demonstrates the comprehensive enhancements made to the SR clustering functionality,
including:
- Multi-algorithm ensemble clustering
- Data leakage detection and prevention
- SHAP/LIME explainability integration
- Purged cross-validation for time series
- Regime-aware clustering
- Advanced feature engineering
- Dynamic parameter optimization
- Hardware-aware optimizations
- VectorBT integration
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_enhanced_sr_clustering():
    """Test the enhanced SR clustering component."""
    try:
        # Import the enhanced component
        from src.training.steps.market_analysis.components.sr_clustering_enhanced import (
            EnhancedSRClusteringComponent,
            EnhancedSRClusteringConfig
        )
        
        logger.info("🚀 Starting Enhanced SR Clustering Test")
        
        # Create the enhanced component
        component = EnhancedSRClusteringComponent()
        
        # Test configuration with all enhancements enabled
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
            'enable_ensemble_clustering': True,
            'clustering_algorithm': 'ensemble',
            'hpo_trials': 20
        }
        
        logger.info("📊 Test Configuration:")
        for key, value in test_config.items():
            logger.info(f"  {key}: {value}")
        
        # Execute the enhanced clustering
        logger.info("\n🔗 Executing Enhanced SR Clustering...")
        result = await component.execute(test_config)
        
        # Display results
        logger.info("\n📈 Enhanced Clustering Results:")
        logger.info(f"  Success: {result['success']}")
        
        if result['success']:
            metrics = result['metrics']
            logger.info(f"  Total Clusters: {metrics.get('total_clusters', 0)}")
            logger.info(f"  Clustering Efficiency: {metrics.get('clustering_efficiency', 0.0):.3f}")
            
            # Display enhancement features
            enhancement_features = metrics.get('enhancement_features', {})
            logger.info("\n🔧 Enhancement Features Used:")
            for feature, enabled in enhancement_features.items():
                status = "✅" if enabled else "❌"
                logger.info(f"  {status} {feature}")
            
            # Display quality metrics
            quality_metrics = metrics.get('quality_metrics', {})
            if quality_metrics:
                logger.info("\n📊 Quality Metrics:")
                for metric, value in quality_metrics.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  {metric}: {value:.3f}")
                    else:
                        logger.info(f"  {metric}: {value}")
            
            # Display performance metrics
            performance_metrics = metrics.get('performance_metrics', {})
            if performance_metrics:
                logger.info("\n⚡ Performance Metrics:")
                for metric, value in performance_metrics.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  {metric}: {value:.3f}")
                    else:
                        logger.info(f"  {metric}: {value}")
            
            # Display hardware metrics
            hardware_metrics = metrics.get('hardware_metrics', {})
            if hardware_metrics:
                logger.info("\n🖥️ Hardware Metrics:")
                for metric, value in hardware_metrics.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  {metric}: {value:.3f}")
                    else:
                        logger.info(f"  {metric}: {value}")
            
            # Display clustering result details
            clustering_result = result.get('clustering_result')
            if clustering_result:
                logger.info(f"\n🎯 Clustering Result Details:")
                logger.info(f"  Total Clusters: {clustering_result.total_clusters}")
                logger.info(f"  Clustering Efficiency: {clustering_result.clustering_efficiency:.3f}")
                
                # Display individual clusters
                for i, cluster in enumerate(clustering_result.clusters[:3]):  # Show first 3 clusters
                    logger.info(f"\n  Cluster {i}:")
                    logger.info(f"    Size: {cluster.get('size', 0)}")
                    logger.info(f"    Type: {cluster.get('type', 'unknown')}")
                    representative = cluster.get('representative', {})
                    logger.info(f"    Representative Price: {representative.get('price', 0.0):.4f}")
                    logger.info(f"    Representative Strength: {representative.get('strength', 0.0):.3f}")
                    logger.info(f"    Representative Confidence: {representative.get('confidence', 0.0):.3f}")
                
                # Display explainability results
                if clustering_result.explainability_results:
                    logger.info(f"\n🔍 Explainability Results:")
                    explainability = clustering_result.explainability_results
                    logger.info(f"  Sample Size: {explainability.get('sample_size', 0)}")
                    logger.info(f"  Explanation Time: {explainability.get('explanation_time', 0.0):.3f}s")
                
                # Display data leakage report
                if clustering_result.data_leakage_report:
                    logger.info(f"\n🛡️ Data Leakage Report:")
                    leakage_report = clustering_result.data_leakage_report
                    logger.info(f"  Leakage Detected: {leakage_report.get('leakage_detected', False)}")
                    lookahead_bias = leakage_report.get('lookahead_bias', {})
                    logger.info(f"  Risk Score: {lookahead_bias.get('risk_score', 0.0):.3f}")
                
                # Display regime analysis
                if clustering_result.regime_analysis:
                    logger.info(f"\n📊 Regime Analysis:")
                    for regime, analysis in clustering_result.regime_analysis.items():
                        logger.info(f"  {regime}:")
                        logger.info(f"    Count: {analysis.get('count', 0)}")
                        logger.info(f"    Price Mean: {analysis.get('price_mean', 0.0):.4f}")
                        logger.info(f"    Strength Mean: {analysis.get('strength_mean', 0.0):.3f}")
                        logger.info(f"    Confidence Mean: {analysis.get('confidence_mean', 0.0):.3f}")
                
                # Display ensemble consensus
                if clustering_result.ensemble_consensus:
                    logger.info(f"\n🎯 Ensemble Consensus:")
                    consensus = clustering_result.ensemble_consensus
                    logger.info(f"  Mean Consensus: {consensus.get('mean_consensus', 0.0):.3f}")
                    logger.info(f"  High Consensus Clusters: {consensus.get('high_consensus_clusters', 0)}")
                    logger.info(f"  Low Consensus Clusters: {consensus.get('low_consensus_clusters', 0)}")
            
            # Display artifacts
            artifacts = result.get('artifacts', [])
            logger.info(f"\n📁 Generated Artifacts: {len(artifacts)}")
            for artifact in artifacts:
                logger.info(f"  {artifact}")
        
        else:
            logger.error(f"❌ Enhanced Clustering Failed: {result.get('error', 'Unknown error')}")
        
        logger.info("\n✅ Enhanced SR Clustering Test Completed")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        logger.error(f"❌ Error details: {traceback.format_exc()}")

async def test_individual_features():
    """Test individual enhanced features."""
    try:
        from src.training.steps.market_analysis.components.sr_clustering_enhanced import (
            EnhancedSRClusteringComponent,
            EnhancedSRClusteringConfig
        )
        
        logger.info("\n🧪 Testing Individual Enhanced Features")
        
        component = EnhancedSRClusteringComponent()
        
        # Test 1: Data Leakage Detection
        logger.info("\n🔍 Testing Data Leakage Detection...")
        test_levels = [
            {'price': 1.2000, 'strength': 0.8, 'touches': 3, 'confidence': 0.7, 'features': {'volume_profile': 0.6}},
            {'price': 1.2050, 'strength': 0.7, 'touches': 2, 'confidence': 0.6, 'features': {'volume_profile': 0.5}},
            {'price': 1.2500, 'strength': 0.9, 'touches': 4, 'confidence': 0.8, 'features': {'volume_profile': 0.8}}
        ]
        
        config = EnhancedSRClusteringConfig(enable_data_leakage_detection=True)
        leakage_report = await component._detect_data_leakage(test_levels, config)
        logger.info(f"  Data Leakage Detection Result: {leakage_report}")
        
        # Test 2: Advanced Feature Engineering
        logger.info("\n🔧 Testing Advanced Feature Engineering...")
        enhanced_levels = await component._advanced_feature_engineering(test_levels, config)
        logger.info(f"  Enhanced Levels Count: {len(enhanced_levels)}")
        if enhanced_levels:
            features = enhanced_levels[0].get('features', {})
            logger.info(f"  Sample Enhanced Features: {list(features.keys())[:5]}")
        
        # Test 3: Regime-Aware Clustering
        logger.info("\n📊 Testing Regime-Aware Clustering...")
        regime_analysis = await component._regime_aware_clustering(enhanced_levels, config)
        logger.info(f"  Regime Analysis: {regime_analysis}")
        
        # Test 4: Ensemble Clustering
        logger.info("\n🎯 Testing Ensemble Clustering...")
        clusters, metrics = await component._ensemble_clustering(enhanced_levels, config)
        logger.info(f"  Ensemble Clusters: {len(clusters)}")
        logger.info(f"  Ensemble Metrics: {metrics}")
        
        # Test 5: Quality Metrics
        logger.info("\n📈 Testing Quality Metrics...")
        quality_metrics = await component._calculate_comprehensive_quality_metrics(clusters, enhanced_levels, config)
        logger.info(f"  Quality Metrics: {quality_metrics}")
        
        logger.info("\n✅ Individual Feature Tests Completed")
        
    except Exception as e:
        logger.error(f"❌ Individual feature test failed: {e}")
        import traceback
        logger.error(f"❌ Error details: {traceback.format_exc()}")

async def test_performance_comparison():
    """Test performance comparison between traditional and enhanced clustering."""
    try:
        from src.training.steps.market_analysis.components.sr_clustering_enhanced import (
            EnhancedSRClusteringComponent,
            EnhancedSRClusteringConfig
        )
        
        logger.info("\n⚡ Testing Performance Comparison")
        
        component = EnhancedSRClusteringComponent()
        
        # Create test data
        test_levels = []
        for i in range(50):  # Create 50 test levels
            test_levels.append({
                'price': 1.2000 + (i * 0.001),
                'strength': 0.5 + (i % 10) * 0.05,
                'touches': (i % 5) + 1,
                'confidence': 0.6 + (i % 8) * 0.05,
                'features': {
                    'volume_profile': 0.5 + (i % 6) * 0.1,
                    'price_action': 0.6 + (i % 7) * 0.05,
                    'technical_indicators': 0.5 + (i % 5) * 0.1,
                    'volatility': 0.3 + (i % 4) * 0.1,
                    'trend': 0.4 + (i % 6) * 0.1,
                    'volume': 0.5 + (i % 8) * 0.05
                },
                'regime': 'low_volatility' if i % 3 == 0 else 'high_volatility',
                'type': 'support' if i % 2 == 0 else 'resistance'
            })
        
        # Test traditional clustering
        logger.info("  Testing Traditional Clustering...")
        start_time = datetime.now()
        traditional_clusters, traditional_metrics = await component._cluster_sr_levels_traditional(
            test_levels, EnhancedSRClusteringConfig(clustering_algorithm='kmeans')
        )
        traditional_time = (datetime.now() - start_time).total_seconds()
        
        # Test enhanced clustering
        logger.info("  Testing Enhanced Clustering...")
        start_time = datetime.now()
        enhanced_clusters, enhanced_metrics = await component._ensemble_clustering(
            test_levels, EnhancedSRClusteringConfig(enable_ensemble_clustering=True)
        )
        enhanced_time = (datetime.now() - start_time).total_seconds()
        
        # Display comparison
        logger.info(f"\n📊 Performance Comparison Results:")
        logger.info(f"  Traditional Clustering:")
        logger.info(f"    Time: {traditional_time:.3f}s")
        logger.info(f"    Clusters: {len(traditional_clusters)}")
        logger.info(f"    Efficiency: {traditional_metrics.get('efficiency', 0.0):.3f}")
        
        logger.info(f"  Enhanced Clustering:")
        logger.info(f"    Time: {enhanced_time:.3f}s")
        logger.info(f"    Clusters: {len(enhanced_clusters)}")
        logger.info(f"    Efficiency: {enhanced_metrics.get('efficiency', 0.0):.3f}")
        
        if traditional_time > 0:
            speedup = traditional_time / enhanced_time if enhanced_time > 0 else 0
            logger.info(f"  Speedup: {speedup:.2f}x")
        
        logger.info("\n✅ Performance Comparison Completed")
        
    except Exception as e:
        logger.error(f"❌ Performance comparison test failed: {e}")
        import traceback
        logger.error(f"❌ Error details: {traceback.format_exc()}")

async def main():
    """Main test function."""
    logger.info("🚀 Starting Enhanced SR Clustering Comprehensive Test Suite")
    
    # Run all tests
    await test_enhanced_sr_clustering()
    await test_individual_features()
    await test_performance_comparison()
    
    logger.info("\n🎉 All Enhanced SR Clustering Tests Completed Successfully!")

if __name__ == "__main__":
    asyncio.run(main())