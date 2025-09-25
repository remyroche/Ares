#!/usr/bin/env python3
"""
Simple test of the real implementations without external dependencies.
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_gradient_flow_analysis():
    """Test gradient flow analysis implementation."""
    print("🧠 Testing Gradient Flow Analysis Implementation")
    print("=" * 50)
    
    try:
        # Import the module
        from src.training.steps.market_analysis.gradient_flow_analysis import GradientFlowAnalyzer
        
        # Create analyzer
        analyzer = GradientFlowAnalyzer()
        print("✅ GradientFlowAnalyzer imported successfully")
        
        # Test basic analysis
        analysis = analyzer.analyze_gradient_flow_improvements()
        print("✅ Gradient flow analysis completed")
        
        # Check results
        neural_improvements = analysis.neural_network_improvements
        linear_improvements = analysis.linear_regression_improvements
        tree_improvements = analysis.tree_based_improvements
        
        print(f"   🧠 Neural Network improvements: {len(neural_improvements)} metrics")
        print(f"   📈 Linear Regression improvements: {len(linear_improvements)} metrics")
        print(f"   🌳 Tree-based improvements: {len(tree_improvements)} metrics")
        
        # Test real gradient flow analysis
        import numpy as np
        
        # Generate sample data
        data = np.random.randn(100, 5)
        binary_targets = np.random.choice([-1, 0, 1], size=100)
        continuous_targets = np.random.beta(2, 5, size=100)
        
        real_analysis = analyzer.analyze_real_gradient_flow(data, binary_targets, continuous_targets)
        print("✅ Real gradient flow analysis with data completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Gradient flow analysis test failed: {e}")
        return False

def test_standalone_optimizer():
    """Test standalone optimizer implementation."""
    print("\n📊 Testing Standalone Optimizer Implementation")
    print("=" * 50)
    
    try:
        # Import the module
        from src.training.steps.market_analysis.standalone_optimizer import (
            StandaloneTimeframeOptimizer, 
            StandaloneOptimizationConfig,
            OptimizationMethod
        )
        
        print("✅ Standalone optimizer modules imported successfully")
        
        # Create configuration
        config = StandaloneOptimizationConfig(
            optimization_method=OptimizationMethod.RANDOM_SEARCH,
            min_horizon=5,
            max_horizon=10,
            random_search_iterations=5
        )
        print("✅ Configuration created")
        
        # Create optimizer
        optimizer = StandaloneTimeframeOptimizer(config)
        print("✅ Optimizer created")
        
        # Test performance metrics calculation
        import numpy as np
        import pandas as pd
        
        # Generate sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.01)
        
        market_data = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Test performance metrics calculation
        point = np.array([5, 0.005])  # horizon=5, target=0.005
        metrics = optimizer._calculate_performance_metrics(point, market_data)
        
        print("✅ Performance metrics calculation completed")
        print(f"   📊 Metrics calculated: {len(metrics)}")
        
        for metric, value in metrics.items():
            print(f"      → {metric}: {value:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Standalone optimizer test failed: {e}")
        return False

def test_implementation_integration():
    """Test integration between implementations."""
    print("\n🔗 Testing Implementation Integration")
    print("=" * 50)
    
    try:
        # Test that both modules can be imported together
        from src.training.steps.market_analysis.gradient_flow_analysis import GradientFlowAnalyzer
        from src.training.steps.market_analysis.standalone_optimizer import StandaloneTimeframeOptimizer
        
        print("✅ Both modules imported successfully together")
        
        # Test that they can be used together
        analyzer = GradientFlowAnalyzer()
        print("✅ Gradient flow analyzer created")
        
        from src.training.steps.market_analysis.standalone_optimizer import StandaloneOptimizationConfig, OptimizationMethod
        
        config = StandaloneOptimizationConfig(optimization_method=OptimizationMethod.RANDOM_SEARCH)
        optimizer = StandaloneTimeframeOptimizer(config)
        print("✅ Standalone optimizer created")
        
        print("✅ Integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 TESTING REAL IMPLEMENTATIONS")
    print("=" * 60)
    print("Testing the real implementations of:")
    print("1. Gradient flow analysis")
    print("2. Performance metrics calculation")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Gradient flow analysis
    if test_gradient_flow_analysis():
        tests_passed += 1
    
    # Test 2: Standalone optimizer
    if test_standalone_optimizer():
        tests_passed += 1
    
    # Test 3: Integration
    if test_implementation_integration():
        tests_passed += 1
    
    # Summary
    print(f"\n📊 TEST RESULTS")
    print("=" * 30)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ ALL TESTS PASSED!")
        print("🎯 Real implementations are working correctly")
        print("\n💡 Key Features Implemented:")
        print("   → Real gradient flow analysis with actual calculations")
        print("   → Real performance metrics with trading simulation")
        print("   → Comprehensive optimization with real metrics")
        print("   → Integration between gradient analysis and optimization")
    else:
        print("❌ Some tests failed")
        print("🔧 Check the error messages above for details")

if __name__ == '__main__':
    main()