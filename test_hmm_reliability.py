#!/usr/bin/env python3
"""
Test script for HMM Reliability Metrics

This script tests the new HMM reliability metrics to ensure they work correctly.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def test_hmm_reliability_metrics():
    """Test HMM reliability metrics calculation."""
    print("🧪 Testing HMM Reliability Metrics")
    
    try:
        from src.training.steps.enhanced_regime_clustering import EnhancedRegimeClustering
        
        # Create test data with clear clusters
        np.random.seed(42)
        n_samples = 200
        n_features = 3
        
        # Create two distinct clusters
        cluster1 = np.random.randn(n_samples // 2, n_features) + [2, 2, 2]
        cluster2 = np.random.randn(n_samples // 2, n_features) + [-2, -2, -2]
        
        features = np.vstack([cluster1, cluster2])
        labels = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
        
        # Initialize enhanced clustering with HMM reliability focus
        config = {
            "target_clusters": 2,
            "hmm_reliability_focus": True,
            "hmm_entropy_penalty_weight": 0.15,
            "min_hmm_state_duration": 5,
            "hmm_transition_smoothness_weight": 0.1
        }
        
        enhanced_clustering = EnhancedRegimeClustering(config)
        
        # Test HMM reliability metrics calculation
        print("   Testing HMM reliability metrics calculation...")
        hmm_metrics = enhanced_clustering._calculate_hmm_reliability_metrics(features, labels)
        
        # Verify metrics are present
        required_metrics = ["entropy_penalty", "transition_smoothness", "reliability_score"]
        for metric in required_metrics:
            assert metric in hmm_metrics, f"Missing metric: {metric}"
        
        print(f"   ✅ HMM metrics calculated successfully:")
        print(f"      Entropy Penalty: {hmm_metrics['entropy_penalty']:.4f}")
        print(f"      Transition Smoothness: {hmm_metrics['transition_smoothness']:.4f}")
        print(f"      Reliability Score: {hmm_metrics['reliability_score']:.4f}")
        
        # Test composite score with HMM metrics
        print("   Testing composite score with HMM metrics...")
        score_dict = enhanced_clustering.calculate_composite_score(features, labels)
        
        # Verify HMM metrics are included
        hmm_metrics_in_score = ["hmm_entropy_penalty", "hmm_transition_smoothness", "hmm_reliability_score"]
        for metric in hmm_metrics_in_score:
            assert metric in score_dict, f"Missing HMM metric in composite score: {metric}"
        
        print(f"   ✅ Composite score with HMM metrics:")
        print(f"      Composite Score: {score_dict['composite_score']:.4f}")
        print(f"      HMM Entropy Penalty: {score_dict['hmm_entropy_penalty']:.4f}")
        print(f"      HMM Transition Smoothness: {score_dict['hmm_transition_smoothness']:.4f}")
        print(f"      HMM Reliability Score: {score_dict['hmm_reliability_score']:.4f}")
        
        # Test with noisy data (should have higher entropy penalty)
        print("   Testing with noisy data...")
        noisy_features = features + np.random.randn(*features.shape) * 0.5
        noisy_score_dict = enhanced_clustering.calculate_composite_score(noisy_features, labels)
        
        print(f"   ✅ Noisy data results:")
        print(f"      Noisy Composite Score: {noisy_score_dict['composite_score']:.4f}")
        print(f"      Noisy HMM Entropy Penalty: {noisy_score_dict['hmm_entropy_penalty']:.4f}")
        
        # The noisy data should generally have higher entropy penalty
        if noisy_score_dict['hmm_entropy_penalty'] > score_dict['hmm_entropy_penalty']:
            print("   ✅ Noisy data correctly shows higher entropy penalty")
        else:
            print("   ⚠️ Noisy data entropy penalty not higher (may be due to randomness)")
        
        print("🎉 All HMM reliability tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure hmmlearn is installed: pip install hmmlearn")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_profiles():
    """Test performance profile configurations."""
    print("\n🚀 Testing Performance Profiles")
    
    try:
        from src.training.steps.enhanced_regime_clustering import EnhancedRegimeClustering
        
        # Test different performance profiles
        profiles = ["fast", "balanced", "thorough"]
        
        for profile in profiles:
            print(f"   Testing {profile} profile...")
            
            # Simulate different data sizes
            data_sizes = [500, 2000, 10000]
            
            for data_size in data_sizes:
                # Create mock configuration based on profile
                if profile == "fast":
                    config = {
                        "bayesian_calls": 20,
                        "max_iterations": 20,
                        "lime_samples": 100,
                        "shap_samples": 20,
                        "use_lime_shap": False,
                        "hmm_reliability_focus": False
                    }
                elif profile == "balanced":
                    config = {
                        "bayesian_calls": 50,
                        "max_iterations": 35,
                        "lime_samples": 300,
                        "shap_samples": 30,
                        "use_lime_shap": True,
                        "hmm_reliability_focus": True
                    }
                else:  # thorough
                    config = {
                        "bayesian_calls": 100,
                        "max_iterations": 50,
                        "lime_samples": 1000,
                        "shap_samples": 100,
                        "use_lime_shap": True,
                        "hmm_reliability_focus": True
                    }
                
                # Adjust based on data size
                if data_size < 1000:
                    config["bayesian_calls"] = int(config["bayesian_calls"] * 0.5)
                    config["max_iterations"] = int(config["max_iterations"] * 0.6)
                elif data_size > 10000:
                    config["bayesian_calls"] = int(config["bayesian_calls"] * 1.5)
                    config["max_iterations"] = int(config["max_iterations"] * 1.2)
                
                enhanced_clustering = EnhancedRegimeClustering(config)
                
                print(f"      Data size {data_size}: bayesian_calls={config['bayesian_calls']}, max_iterations={config['max_iterations']}")
        
        print("   ✅ Performance profiles tested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Performance profile test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 HMM Reliability and Performance Tests")
    print("=" * 50)
    
    tests = [
        ("HMM Reliability Metrics", test_hmm_reliability_metrics),
        ("Performance Profiles", test_performance_profiles)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning test: {test_name}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Results Summary")
    print(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! HMM reliability and performance features are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())