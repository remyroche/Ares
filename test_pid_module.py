#!/usr/bin/env python3
"""
Test script for the Partial Information Decompositor module.
"""

import numpy as np
import pandas as pd
from typing import List
import sys
import os

# Add the workspace to the path
sys.path.append('/workspace')

def create_sample_data(n_samples: int = 1000, n_features: int = 20) -> tuple:
    """Create sample data for testing PID analysis."""
    np.random.seed(42)
    
    # Create base features with some correlations
    X = np.random.randn(n_samples, n_features)
    
    # Add some correlated features
    X[:, 1] = X[:, 0] + 0.5 * np.random.randn(n_samples)  # Correlated with feature 0
    X[:, 2] = X[:, 0] * X[:, 1] + 0.3 * np.random.randn(n_samples)  # Interaction
    
    # Create timeframe features
    X[:, 3] = X[:, 0] + 0.2 * np.random.randn(n_samples)  # 1m timeframe
    X[:, 4] = X[:, 0] + 0.1 * np.random.randn(n_samples)  # 5m timeframe
    
    # Create target with some feature interactions
    y = (X[:, 0] * X[:, 1] + 
         0.5 * X[:, 2] + 
         0.3 * X[:, 3] * X[:, 4] + 
         np.random.randn(n_samples) * 0.1)
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]
    feature_names[3] = "price_1m"
    feature_names[4] = "price_5m"
    feature_names[5] = "volume_1h"
    feature_names[6] = "volume_4h"
    
    return X, y, feature_names

def test_pid_module():
    """Test the PID module functionality."""
    print("🧪 Testing Partial Information Decompositor Module")
    print("=" * 60)
    
    try:
        # Import the PID module
        from src.training.utils.feature_selection.partial_information_decompositor import (
            PartialInformationDecompositor, PIDConfig, PIDResult
        )
        print("✅ Successfully imported PID module")
        
        # Create sample data
        X, y, feature_names = create_sample_data(n_samples=500, n_features=15)
        print(f"📊 Created sample data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Initialize PID decompositor
        config = PIDConfig(
            synergy_threshold=0.05,
            redundancy_threshold=0.1,
            max_polynomial_degree=2,
            max_interaction_features=20,
            sample_size=300  # Limit sample size for testing
        )
        
        decompositor = PartialInformationDecompositor(config)
        print("✅ Initialized PID decompositor")
        
        # Run PID analysis
        print("\n🔍 Running PID analysis...")
        pid_result = decompositor.decompose_information(X, y, feature_names)
        
        print(f"✅ PID analysis completed in {pid_result.execution_time:.3f}s")
        print(f"📊 Feature pairs analyzed: {pid_result.feature_pairs_analyzed}")
        print(f"📊 Significant interactions found: {pid_result.significant_interactions}")
        
        # Display results
        print(f"\n📈 Polynomial features generated: {len(pid_result.polynomial_features)}")
        if pid_result.polynomial_features:
            print(f"   Examples: {pid_result.polynomial_features[:5]}")
        
        print(f"📈 Interaction features generated: {len(pid_result.interaction_features)}")
        if pid_result.interaction_features:
            print(f"   Examples: {pid_result.interaction_features[:5]}")
        
        print(f"📈 Cross-timeframe features generated: {len(pid_result.cross_timeframe_features)}")
        if pid_result.cross_timeframe_features:
            print(f"   Examples: {pid_result.cross_timeframe_features[:5]}")
        
        # Test feature matrix generation
        print(f"\n🔧 Testing feature matrix generation...")
        expanded_X, expanded_names = decompositor.generate_feature_matrix(X, feature_names, pid_result)
        print(f"✅ Expanded feature matrix: {X.shape} → {expanded_X.shape}")
        print(f"📊 Total features: {len(expanded_names)}")
        
        # Test feature importance scores
        importance_scores = decompositor.get_feature_importance_scores(pid_result)
        print(f"📊 Feature importance scores calculated for {len(importance_scores)} features")
        
        if importance_scores:
            top_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            print("   Top 5 features by importance:")
            for feat, score in top_features:
                print(f"     {feat}: {score:.4f}")
        
        print("\n✅ All PID module tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_framework():
    """Test integration with the main feature selection framework."""
    print("\n🧪 Testing PID integration with Feature Selection Framework")
    print("=" * 60)
    
    try:
        from src.training.utils.feature_selection import FeatureSelectionFramework
        
        # Create sample data
        X, y, feature_names = create_sample_data(n_samples=200, n_features=10)
        
        # Configure framework with PID analysis enabled
        config = {
            'mode': 'blank',  # Use minimal bootstrap for testing
            'partial_information_decompositor': {
                'synergy_threshold': 0.05,
                'redundancy_threshold': 0.1,
                'max_polynomial_degree': 2,
                'max_interaction_features': 10
            }
        }
        
        framework = FeatureSelectionFramework(config)
        print("✅ Initialized FeatureSelectionFramework with PID support")
        
        # Run comprehensive feature selection with PID analysis
        print("\n🔍 Running comprehensive feature selection with PID analysis...")
        results = framework.run_comprehensive_feature_selection(
            X, y, feature_names,
            target_features=8,
            enable_pid_analysis=True
        )
        
        if results['success']:
            print("✅ Comprehensive feature selection with PID completed successfully")
            
            # Check PID results
            pid_results = results.get('pid_results', {})
            if pid_results and pid_results.get('success', False):
                print(f"📊 PID analysis found {pid_results.get('significant_interactions', 0)} significant interactions")
                print(f"🔧 Generated {len(pid_results.get('polynomial_features', []))} polynomial features")
                print(f"🔧 Generated {len(pid_results.get('interaction_features', []))} interaction features")
                print(f"🔧 Generated {len(pid_results.get('cross_timeframe_features', []))} cross-timeframe features")
            else:
                print("⚠️ PID analysis was not successful")
            
            # Check final selection
            final_features = results.get('final_selected_features', [])
            print(f"🎯 Final feature selection: {len(final_features)} features")
            if final_features:
                print(f"   Selected features: {final_features}")
        else:
            print("❌ Comprehensive feature selection failed")
            return False
        
        print("\n✅ PID integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting PID Module Tests")
    print("=" * 60)
    
    # Test basic PID module
    test1_passed = test_pid_module()
    
    # Test integration with framework
    test2_passed = test_integration_with_framework()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   PID Module Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Integration Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! PID module is ready for use.")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed. Please check the implementation.")
        sys.exit(1)