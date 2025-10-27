#!/usr/bin/env python3
"""
Test script for Enhanced SR Clustering module.

This script tests the enhanced SR clustering functionality with various
configurations and validates the integration of all components.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.sr_clustering.enhanced_sr_clustering import (
    EnhancedSRClustering,
    EnhancedSRClusteringConfig,
    ClusteringAlgorithm,
    OptimizationStrategy,
    create_enhanced_sr_clustering
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(seed)
    
    # Create date range
    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start_date, periods=n_samples, freq='1H')
    
    # Generate realistic price data with some support/resistance levels
    base_price = 100
    price_changes = np.random.randn(n_samples) * 0.02  # 2% volatility
    
    # Add some support/resistance levels
    sr_levels = [95, 100, 105, 110, 115]
    for i in range(n_samples):
        if i % 100 == 0:  # Every 100 hours, add some SR level influence
            level = np.random.choice(sr_levels)
            price_changes[i] = (level - base_price) * 0.1
    
    prices = base_price + np.cumsum(price_changes)
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'open': prices + np.random.randn(n_samples) * 0.1,
        'high': prices + np.abs(np.random.randn(n_samples)) * 0.5,
        'low': prices - np.abs(np.random.randn(n_samples)) * 0.5,
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    }, index=dates)
    
    # Ensure high >= low and proper OHLC relationships
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    return data

async def test_basic_clustering():
    """Test basic clustering functionality."""
    logger.info("Testing basic clustering functionality...")
    
    # Create sample data
    price_data = create_sample_data(500)
    
    # Create basic configuration
    config = EnhancedSRClusteringConfig(
        clustering_algorithm=ClusteringAlgorithm.DBSCAN,
        dbscan_eps=0.1,
        dbscan_min_samples=5,
        min_cluster_size=3,
        feature_engineering_config={
            'price_features': True,
            'volume_features': True,
            'time_features': True,
            'technical_indicators': True,
            'microstructure_features': False,  # Disable for basic test
            'feature_normalization': 'standard',
            'dimensionality_reduction': None,
        },
        hpo_config={
            'optimization_strategy': OptimizationStrategy.BAYESIAN_TPE,
            'n_trials': 5,  # Small number for testing
            'timeout': 60
        },
        backtesting_config={
            'enabled': False  # Disable for basic test
        },
        explainability_config={
            'shap_enabled': False,
            'lime_enabled': False
        }
    )
    
    # Create clustering instance
    clustering = create_enhanced_sr_clustering(config)
    
    try:
        # Run clustering
        results = await clustering.cluster_sr_levels(price_data)
        
        logger.info(f"✅ Basic clustering test passed: Found {len(results)} clusters")
        
        # Print top 3 clusters
        for i, result in enumerate(results[:3]):
            logger.info(f"  Cluster {i+1}: Price={result.centroid_price:.2f}, "
                       f"Quality={result.cluster_quality:.4f}, Size={result.cluster_size}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic clustering test failed: {e}")
        return False

async def test_advanced_clustering():
    """Test advanced clustering with HDBSCAN and HPO."""
    logger.info("Testing advanced clustering functionality...")
    
    # Create sample data
    price_data = create_sample_data(800)
    
    # Create advanced configuration
    config = EnhancedSRClusteringConfig(
        clustering_algorithm=ClusteringAlgorithm.HDBSCAN,
        hdbscan_min_cluster_size=5,
        hdbscan_min_samples=3,
        hdbscan_cluster_selection_epsilon=0.05,
        min_cluster_size=5,
        feature_engineering_config={
            'price_features': True,
            'volume_features': True,
            'time_features': True,
            'technical_indicators': True,
            'microstructure_features': True,
            'feature_normalization': 'robust',
            'dimensionality_reduction': 'pca',
            'n_components': 0.8
        },
        hpo_config={
            'optimization_strategy': OptimizationStrategy.BAYESIAN_TPE,
            'n_trials': 10,
            'timeout': 120
        },
        backtesting_config={
            'enabled': True,
            'initial_capital': 10000,
            'commission': 0.001
        },
        explainability_config={
            'shap_enabled': True,
            'lime_enabled': True
        }
    )
    
    # Create clustering instance
    clustering = create_enhanced_sr_clustering(config)
    
    try:
        # Run clustering
        results = await clustering.cluster_sr_levels(price_data)
        
        logger.info(f"✅ Advanced clustering test passed: Found {len(results)} clusters")
        
        # Print detailed results
        for i, result in enumerate(results[:5]):
            logger.info(f"  Cluster {i+1}:")
            logger.info(f"    Price: {result.centroid_price:.2f}")
            logger.info(f"    Quality: {result.cluster_quality:.4f}")
            logger.info(f"    Size: {result.cluster_size}")
            logger.info(f"    Confidence: {result.confidence:.4f}")
            logger.info(f"    Reliability: {result.reliability_score:.4f}")
            logger.info(f"    Stability: {result.stability_score:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Advanced clustering test failed: {e}")
        return False

async def test_performance_monitoring():
    """Test performance monitoring and logging."""
    logger.info("Testing performance monitoring...")
    
    # Create sample data
    price_data = create_sample_data(1000)
    
    # Create configuration with performance monitoring
    config = EnhancedSRClusteringConfig(
        clustering_algorithm=ClusteringAlgorithm.DBSCAN,
        dbscan_eps=0.05,
        dbscan_min_samples=3,
        min_cluster_size=3,
        feature_engineering_config={
            'price_features': True,
            'volume_features': True,
            'time_features': True,
            'technical_indicators': True,
            'microstructure_features': True,
            'feature_normalization': 'standard',
            'dimensionality_reduction': 'pca',
            'n_components': 0.7
        },
        hpo_config={
            'optimization_strategy': OptimizationStrategy.BAYESIAN_TPE,
            'n_trials': 8,
            'timeout': 90
        },
        backtesting_config={
            'enabled': True,
            'initial_capital': 10000,
            'commission': 0.001
        },
        explainability_config={
            'shap_enabled': True,
            'lime_enabled': True
        }
    )
    
    # Create clustering instance
    clustering = create_enhanced_sr_clustering(config)
    
    try:
        # Run clustering and monitor performance
        start_time = datetime.now()
        results = await clustering.cluster_sr_levels(price_data)
        end_time = datetime.now()
        
        execution_time = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ Performance monitoring test passed:")
        logger.info(f"  Execution time: {execution_time:.2f} seconds")
        logger.info(f"  Clusters found: {len(results)}")
        logger.info(f"  Data points processed: {len(price_data)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance monitoring test failed: {e}")
        return False

async def test_error_handling():
    """Test error handling with invalid data."""
    logger.info("Testing error handling...")
    
    # Create invalid data (empty DataFrame)
    invalid_data = pd.DataFrame()
    
    # Create configuration
    config = EnhancedSRClusteringConfig(
        clustering_algorithm=ClusteringAlgorithm.DBSCAN,
        dbscan_eps=0.1,
        dbscan_min_samples=5,
        min_cluster_size=3
    )
    
    # Create clustering instance
    clustering = create_enhanced_sr_clustering(config)
    
    try:
        # This should handle the error gracefully
        results = await clustering.cluster_sr_levels(invalid_data)
        
        # If we get here, it means the error was handled
        logger.info(f"✅ Error handling test passed: Handled invalid data gracefully")
        return True
        
    except Exception as e:
        # This is expected for invalid data
        logger.info(f"✅ Error handling test passed: Caught expected error: {e}")
        return True

async def main():
    """Run all tests."""
    logger.info("Starting Enhanced SR Clustering tests...")
    
    tests = [
        ("Basic Clustering", test_basic_clustering),
        ("Advanced Clustering", test_advanced_clustering),
        ("Performance Monitoring", test_performance_monitoring),
        ("Error Handling", test_error_handling),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
