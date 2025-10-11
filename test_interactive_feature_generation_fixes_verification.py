#!/usr/bin/env python3
"""
Test Script to Verify Interactive Feature Generation Fixes

This script tests the fixes implemented for the interactive feature generation module
to ensure that features are actually generated and the pipeline works correctly.
"""

import asyncio
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

def create_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create test market data for feature generation."""
    np.random.seed(42)
    
    # Create time index
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='15min')
    
    # Generate realistic market data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_samples)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Add some trend and volatility
    trend = np.linspace(0, 0.1, n_samples)
    volatility = 0.5 + 0.3 * np.sin(np.linspace(0, 4*np.pi, n_samples))
    
    # Generate OHLCV data
    data = {
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices * (1 + trend + np.random.normal(0, volatility * 0.01, n_samples)),
        'volume': np.random.lognormal(10, 0.5, n_samples),
    }
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    for i in range(n_samples):
        data['high'][i] = max(data['high'][i], data['open'][i], data['close'][i])
        data['low'][i] = min(data['low'][i], data['open'][i], data['close'][i])
    
    df = pd.DataFrame(data, index=dates)
    
    # Add some additional features
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)  # Simple target
    
    return df

async def test_feature_generation_fixes():
    """Test the feature generation fixes."""
    tprint("🧪 Testing Interactive Feature Generation Fixes")
    tprint("=" * 60)
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.interactive_feature_generation_component import (
            InteractiveFeatureGenerationComponent, InteractiveFeatureGenerationConfig
        )
        
        # Create test data
        data = create_test_data(500)
        tprint_info(f"📊 Created test data: {data.shape}")
        
        # Test 1: Test comprehensive validation
        tprint("\n🔍 Test 1: Comprehensive Input Validation")
        component = InteractiveFeatureGenerationComponent()
        
        # Test with valid data
        training_input = {
            'data': data,
            'targets': {'target': data['target']}
        }
        pipeline_state = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframe': '15m'
        }
        
        # This should not raise an exception
        component._validate_inputs(training_input, pipeline_state)
        tprint_success("✅ Comprehensive validation passed for valid data")
        
        # Test with invalid data (should fail fast)
        try:
            invalid_data = pd.DataFrame()  # Empty data
            invalid_input = {'data': invalid_data, 'targets': {}}
            component._validate_inputs(invalid_input, pipeline_state)
            tprint_error("❌ Should have failed with empty data")
            return False
        except ValueError as e:
            tprint_success(f"✅ Fast-fail worked correctly: {e}")
        
        # Test 2: Test feature generation
        tprint("\n🏗️ Test 2: Feature Generation")
        
        # Create component with fixed configuration
        config = InteractiveFeatureGenerationConfig(
            symbol='ETHUSDT',
            exchange='binance',
            timeframe='15m',
            feature_budget_pre=50,
            feature_budget_post=(10, 20),
            interactions_cap=10,
            enable_matrix_optimization=True,
            enable_hardware_optimization=True,
            enable_parallel_processing=False,  # Disable for testing
            variance_threshold=1e-6,  # Use the fixed threshold
            top_k_per_family=10  # Use the fixed threshold
        )
        
        component = InteractiveFeatureGenerationComponent(config)
        
        # Execute component
        tprint_info("🚀 Executing interactive feature generation component...")
        result = await component.execute(training_input, pipeline_state)
        
        if result.success:
            tprint_success(f"✅ Component execution completed successfully!")
            tprint_info(f"⏱️ Execution time: {result.execution_time:.3f}s")
            
            # Check artifacts
            if hasattr(result, 'artifacts') and result.artifacts:
                artifacts = result.artifacts.interactive_feature_generation_result
                feature_count = len(artifacts.get('feature_names', []))
                selected_count = len(artifacts.get('selected_features', []))
                
                tprint_info(f"📊 Features generated: {feature_count}")
                tprint_info(f"📊 Selected features: {selected_count}")
                
                if feature_count > 0:
                    tprint_success("✅ Component generated features successfully!")
                    tprint_info(f"📊 Sample features: {artifacts.get('feature_names', [])[:10]}")
                    return True
                else:
                    tprint_error("❌ Component did not generate any features!")
                    return False
            else:
                tprint_error("❌ No artifacts found in result!")
                return False
        else:
            tprint_error(f"❌ Component execution failed: {result.error_message}")
            return False
            
    except Exception as e:
        tprint_error(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_memory_management():
    """Test memory management improvements."""
    tprint("\n💾 Test 3: Memory Management")
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.enhanced_optimized_orchestrator import (
            EnhancedOptimizedInteractionOrchestrator, EnhancedOptimizedConfig
        )
        
        # Create test data
        data = create_test_data(1000)
        
        # Create orchestrator with memory management
        config = EnhancedOptimizedConfig(
            max_memory_gb=2.0,  # Small memory limit for testing
            enable_parallel_processing=False,
            variance_threshold=1e-6,
            top_k_per_family=10
        )
        
        orchestrator = EnhancedOptimizedInteractionOrchestrator(config)
        
        # Test memory optimization
        training_input = {'data': data, 'target_column': 'target'}
        pipeline_state = {'symbol': 'ETHUSDT', 'timeframe': '15m'}
        
        result = await orchestrator.generate_features(training_input, pipeline_state)
        
        if result.success:
            tprint_success("✅ Memory management test passed")
            tprint_info(f"💾 Memory usage: {result.memory_usage_mb:.1f} MB")
            return True
        else:
            tprint_error(f"❌ Memory management test failed: {result.error_message}")
            return False
            
    except Exception as e:
        tprint_error(f"❌ Memory management test failed: {e}")
        return False

async def main():
    """Run all tests."""
    tprint("🚀 Starting Interactive Feature Generation Fix Verification")
    tprint("=" * 60)
    
    tests = [
        ("Feature Generation Fixes", test_feature_generation_fixes),
        ("Memory Management", test_memory_management),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        tprint(f"\n🧪 Running {test_name} Test...")
        tprint("-" * 40)
        
        try:
            success = await test_func()
            results.append((test_name, success))
            
            if success:
                tprint_success(f"✅ {test_name} test passed!")
            else:
                tprint_error(f"❌ {test_name} test failed!")
                
        except Exception as e:
            tprint_error(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    tprint("\n" + "=" * 60)
    tprint("📊 TEST SUMMARY")
    tprint("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        tprint(f"{status} {test_name}")
        if success:
            passed += 1
    
    tprint(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        tprint_success("🎉 All tests passed! The fixes are working correctly.")
        return True
    else:
        tprint_error(f"⚠️ {total - passed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)