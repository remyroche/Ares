#!/usr/bin/env python3
"""
Test script for Regime Clustering Pipeline.

This script tests the clustering pipeline with actual HMM regime discovery data
to verify the implementation works correctly.
"""

import json
import sys
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.training.steps.market_analysis.regime_clustering.main_clustering_pipeline import RegimeClusteringPipeline
from src.training.steps.market_analysis.regime_clustering.config import get_config_template


def test_basic_functionality():
    """Test basic clustering functionality."""
    print("🧪 Testing basic clustering functionality")
    
    # Use a small, conservative configuration for testing
    config = get_config_template('conservative')
    config_dict = config.to_dict()
    
    # Create pipeline
    pipeline = RegimeClusteringPipeline(config_dict)
    
    # Test with actual HMM outcome file
    hmm_outcome_path = "/workspace/outcomes/market_analysis_hmm_regime_discovery_outcome_20250920_095044.json"
    output_dir = "/workspace/outputs/regime_clustering_test"
    
    # Check if HMM outcome file exists
    if not Path(hmm_outcome_path).exists():
        print(f"❌ HMM outcome file not found: {hmm_outcome_path}")
        return False
    
    try:
        # Run clustering pipeline
        results = pipeline.run_clustering_pipeline(hmm_outcome_path, output_dir)
        
        # Verify results structure
        assert 'clustering_results' in results
        assert 'validation_results' in results
        assert 'cluster_characteristics' in results
        assert 'summary' in results
        
        # Verify clustering results
        clustering = results['clustering_results']['clustering_results']
        assert 'cluster_labels' in clustering
        assert 'cluster_stats' in clustering
        assert 'validation_metrics' in clustering
        
        # Verify cluster stats
        cluster_stats = clustering['cluster_stats']
        assert len(cluster_stats) > 0
        
        # Verify validation results
        validation = results['validation_results']
        assert 'validity' in validation
        assert 'size_distribution' in validation
        assert 'overall_quality' in validation
        
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_data_loading():
    """Test data loading functionality."""
    print("🧪 Testing data loading functionality")
    
    config = get_config_template('balanced')
    pipeline = RegimeClusteringPipeline(config.to_dict())
    
    hmm_outcome_path = "/workspace/outcomes/market_analysis_hmm_regime_discovery_outcome_20250920_095044.json"
    
    try:
        # Test loading HMM results
        regime_data = pipeline.clusterer.load_hmm_results(hmm_outcome_path)
        
        # Verify data structure
        assert 'regime_models' in regime_data
        assert 'regime_assignments' in regime_data
        assert 'metadata' in regime_data
        
        # Verify data content
        assert len(regime_data['regime_models']) > 0
        assert len(regime_data['regime_assignments']) > 0
        
        print(f"✅ Loaded {len(regime_data['regime_models'])} regimes with {len(regime_data['regime_assignments'])} samples")
        
        # Test coordinate parsing
        coordinates = pipeline.clusterer.parse_regime_coordinates()
        assert coordinates.shape[1] == 3  # 3D coordinates
        assert coordinates.shape[0] == len(regime_data['regime_models'])
        
        print(f"✅ Parsed {coordinates.shape[0]} regime coordinates")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False


def test_clustering_algorithm():
    """Test clustering algorithm functionality."""
    print("🧪 Testing clustering algorithm")
    
    config = get_config_template('balanced')
    pipeline = RegimeClusteringPipeline(config.to_dict())
    
    hmm_outcome_path = "/workspace/outcomes/market_analysis_hmm_regime_discovery_outcome_20250920_095044.json"
    
    try:
        # Load data
        pipeline.clusterer.load_hmm_results(hmm_outcome_path)
        pipeline.clusterer.parse_regime_coordinates()
        
        # Test clustering
        cluster_labels = pipeline.clusterer.perform_clustering()
        
        # Verify clustering results
        assert len(cluster_labels) == len(pipeline.clusterer.regime_coordinates)
        assert len(set(cluster_labels)) > 1  # More than one cluster
        
        print(f"✅ Clustering created {len(set(cluster_labels))} clusters")
        
        # Test size constraints
        pipeline.clusterer.apply_size_constraints()
        pipeline.clusterer.create_noise_cluster()
        pipeline.clusterer.calculate_cluster_statistics()
        
        # Verify cluster stats
        assert pipeline.clusterer.cluster_stats is not None
        assert len(pipeline.clusterer.cluster_stats) > 0
        
        print(f"✅ Cluster statistics calculated for {len(pipeline.clusterer.cluster_stats)} clusters")
        
        return True
        
    except Exception as e:
        print(f"❌ Clustering algorithm test failed: {e}")
        return False


def test_validation():
    """Test validation functionality."""
    print("🧪 Testing validation functionality")
    
    config = get_config_template('balanced')
    pipeline = RegimeClusteringPipeline(config.to_dict())
    validator = pipeline.validator
    
    hmm_outcome_path = "/workspace/outcomes/market_analysis_hmm_regime_discovery_outcome_20250920_095044.json"
    
    try:
        # Load and cluster data
        pipeline.clusterer.load_hmm_results(hmm_outcome_path)
        pipeline.clusterer.parse_regime_coordinates()
        pipeline.clusterer.perform_clustering()
        pipeline.clusterer.apply_size_constraints()
        pipeline.clusterer.create_noise_cluster()
        pipeline.clusterer.calculate_cluster_statistics()
        
        # Test validation
        validation_results = validator.validate_clustering_results(
            pipeline.clusterer.cluster_labels,
            pipeline.clusterer.regime_coordinates,
            pipeline.clusterer.cluster_stats,
            pipeline.clusterer.regime_data['regime_assignments']
        )
        
        # Verify validation structure
        assert 'internal_coherence' in validation_results
        assert 'validity' in validation_results
        assert 'distinction' in validation_results
        assert 'size_distribution' in validation_results
        assert 'overall_quality' in validation_results
        
        # Verify quality scores
        overall_quality = validation_results['overall_quality']
        assert 'overall_score' in overall_quality
        assert 'quality_level' in overall_quality
        assert 'recommendations' in overall_quality
        
        print(f"✅ Validation completed - Quality: {overall_quality['quality_level']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False


def test_analysis():
    """Test analysis functionality."""
    print("🧪 Testing analysis functionality")
    
    config = get_config_template('balanced')
    pipeline = RegimeClusteringPipeline(config.to_dict())
    analyzer = pipeline.analyzer
    
    hmm_outcome_path = "/workspace/outcomes/market_analysis_hmm_regime_discovery_outcome_20250920_095044.json"
    
    try:
        # Load and cluster data
        pipeline.clusterer.load_hmm_results(hmm_outcome_path)
        pipeline.clusterer.parse_regime_coordinates()
        pipeline.clusterer.perform_clustering()
        pipeline.clusterer.apply_size_constraints()
        pipeline.clusterer.create_noise_cluster()
        pipeline.clusterer.calculate_cluster_statistics()
        
        # Test analysis
        characteristics = analyzer.analyze_cluster_characteristics(
            pipeline.clusterer.cluster_stats,
            pipeline.clusterer.regime_coordinates,
            pipeline.clusterer.cluster_labels
        )
        
        # Verify analysis structure
        assert len(characteristics) == len(pipeline.clusterer.cluster_stats)
        
        for cluster_id, char in characteristics.items():
            assert 'interpretation' in char
            assert 'market_conditions' in char
            assert 'diversity_score' in char
        
        # Test cluster naming
        cluster_names = analyzer.generate_cluster_names(characteristics)
        assert len(cluster_names) == len(characteristics)
        
        print(f"✅ Analysis completed for {len(characteristics)} clusters")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("🚀 Running Regime Clustering Tests")
    print("="*50)
    
    tests = [
        test_data_loading,
        test_clustering_algorithm,
        test_validation,
        test_analysis,
        test_basic_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "="*50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Implementation is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)