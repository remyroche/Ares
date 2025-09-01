#!/usr/bin/env python3
"""
Test script for Enhanced Regime Clustering System

This script tests the enhanced clustering system with synthetic data to verify:
1. DBSCAN + Bayesian optimization
2. Noise point handling
3. Hybrid refinement
4. Comprehensive reporting
5. Quality metrics calculation
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_samples=1000, n_features=5, n_clusters=6):
    """Generate synthetic data with known clusters for testing."""
    logger.info(f"Generating synthetic data: {n_samples} samples, {n_features} features, {n_clusters} clusters")
    
    # Generate cluster centers
    np.random.seed(42)
    centers = np.random.randn(n_clusters, n_features) * 2
    
    # Generate data points around centers
    data = []
    labels = []
    samples_per_cluster = n_samples // n_clusters
    
    for i in range(n_clusters):
        cluster_data = np.random.randn(samples_per_cluster, n_features) * 0.5 + centers[i]
        data.append(cluster_data)
        labels.extend([i] * samples_per_cluster)
    
    # Add some noise points
    noise_points = np.random.randn(n_samples // 10, n_features) * 3
    data.append(noise_points)
    labels.extend([-1] * (n_samples // 10))  # -1 for noise
    
    # Combine all data
    features = np.vstack(data)
    labels = np.array(labels)
    
    # Shuffle data
    indices = np.random.permutation(len(features))
    features = features[indices]
    labels = labels[indices]
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    logger.info(f"Generated data shape: {features.shape}")
    logger.info(f"Unique labels: {np.unique(labels)}")
    
    return features, labels, feature_names

def test_enhanced_clustering():
    """Test the enhanced clustering system."""
    logger.info("🚀 Starting Enhanced Clustering System Test")
    
    try:
        # Import the enhanced clustering system
        from src.training.steps.enhanced_regime_clustering import EnhancedRegimeClustering
        
        # Generate test data
        features, true_labels, feature_names = generate_synthetic_data(
            n_samples=1000, 
            n_features=5, 
            n_clusters=6
        )
        
        # Test different configurations
        test_configs = [
            {
                "name": "Light Mode (2 clusters)",
                "config": {
                    "target_clusters": 2,
                    "bayesian_calls": 20,  # Reduced for faster testing
                    "max_iterations": 20,
                    "no_improvement_limit": 5
                }
            },
            {
                "name": "Blank Mode (4 clusters)",
                "config": {
                    "target_clusters": 4,
                    "bayesian_calls": 20,
                    "max_iterations": 20,
                    "no_improvement_limit": 5
                }
            },
            {
                "name": "Full Mode (20 clusters)",
                "config": {
                    "target_clusters": 20,
                    "bayesian_calls": 20,
                    "max_iterations": 20,
                    "no_improvement_limit": 5
                }
            }
        ]
        
        for test_case in test_configs:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing: {test_case['name']}")
            logger.info(f"{'='*60}")
            
            # Initialize enhanced clustering
            enhanced_clustering = EnhancedRegimeClustering(test_case["config"])
            
            # Run enhanced clustering
            results = enhanced_clustering.run_enhanced_clustering(features, feature_names)
            
            # Verify results
            assert "final_labels" in results, "Missing final_labels in results"
            assert "final_score_dict" in results, "Missing final_score_dict in results"
            assert "report" in results, "Missing report in results"
            
            # Check quality metrics
            final_score_dict = results["final_score_dict"]
            assert final_score_dict["n_clusters"] > 0, "No clusters found"
            assert final_score_dict["composite_score"] > -1000, "Invalid composite score"
            assert final_score_dict["coverage"] > 0.5, "Low data coverage"
            
            # Log results
            logger.info(f"✅ Test passed: {test_case['name']}")
            logger.info(f"   Final clusters: {final_score_dict['n_clusters']}")
            logger.info(f"   Composite score: {final_score_dict['composite_score']:.4f}")
            logger.info(f"   Coverage: {final_score_dict['coverage']:.3f}")
            logger.info(f"   Silhouette: {final_score_dict['silhouette']:.4f}")
            
            # Save report
            report_path = Path(f"test_report_{test_case['name'].replace(' ', '_').replace('(', '').replace(')', '')}.txt")
            with open(report_path, 'w') as f:
                f.write(results["report"])
            logger.info(f"   Report saved: {report_path}")
        
        logger.info(f"\n{'='*60}")
        logger.info("🎉 All Enhanced Clustering Tests Passed!")
        logger.info(f"{'='*60}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure the enhanced clustering module is properly installed")
        return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quality_metrics():
    """Test quality metrics calculation independently."""
    logger.info("\n🔍 Testing Quality Metrics Calculation")
    
    try:
        from src.training.steps.enhanced_regime_clustering import EnhancedRegimeClustering
        
        # Create simple test data
        np.random.seed(42)
        features = np.random.randn(100, 3)
        labels = np.random.randint(0, 3, 100)
        
        # Initialize clustering system
        config = {"target_clusters": 3}
        clustering = EnhancedRegimeClustering(config)
        
        # Test composite score calculation
        score_dict = clustering.calculate_composite_score(features, labels)
        
        # Verify all required metrics are present
        required_metrics = [
            "composite_score", "silhouette", "calinski_harabasz", 
            "davies_bouldin", "skew_penalty", "volatility_penalty",
            "n_clusters", "coverage"
        ]
        
        for metric in required_metrics:
            assert metric in score_dict, f"Missing metric: {metric}"
        
        logger.info(f"✅ Quality metrics test passed")
        logger.info(f"   Composite score: {score_dict['composite_score']:.4f}")
        logger.info(f"   Silhouette: {score_dict['silhouette']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Quality metrics test failed: {e}")
        return False

def test_noise_handling():
    """Test noise point handling."""
    logger.info("\n🔧 Testing Noise Point Handling")
    
    try:
        from src.training.steps.enhanced_regime_clustering import EnhancedRegimeClustering
        
        # Create data with noise points
        np.random.seed(42)
        features = np.random.randn(200, 3)
        labels = np.random.randint(0, 3, 200)
        
        # Add noise points
        labels[50:60] = -1  # 10 noise points
        
        config = {"target_clusters": 3}
        clustering = EnhancedRegimeClustering(config)
        
        # Test noise handling
        processed_labels = clustering.handle_noise_points(features, labels)
        
        # Verify no noise points remain
        assert -1 not in processed_labels, "Noise points not properly handled"
        
        logger.info(f"✅ Noise handling test passed")
        logger.info(f"   Original noise points: {sum(labels == -1)}")
        logger.info(f"   Processed noise points: {sum(processed_labels == -1)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Noise handling test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🧪 Enhanced Clustering System Test Suite")
    logger.info("=" * 50)
    
    # Run tests
    tests = [
        ("Quality Metrics", test_quality_metrics),
        ("Noise Handling", test_noise_handling),
        ("Full System", test_enhanced_clustering)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\nRunning test: {test_name}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("📊 Test Results Summary")
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
        logger.info("🎉 All tests passed! Enhanced clustering system is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())