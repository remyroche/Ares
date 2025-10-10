#!/usr/bin/env python3
"""
Test Script for Interactive Feature Generation Improvements

This script tests the improvements made to the interactive feature generation system:
1. ImportManager utility for reducing code duplication
2. Improved feature generation logic with better validation
3. Integration of both improvements in the main component
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

async def test_import_manager():
    """Test the ImportManager utility."""
    tprint("🧪 Testing ImportManager...")
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.import_manager import get_import_manager
        
        # Get import manager
        manager = get_import_manager()
        
        # Test tprint import (should work)
        tprint_result = manager.import_tprint()
        if tprint_result.status.value == "success":
            tprint_success("✅ tprint import successful")
        else:
            tprint_error("❌ tprint import failed")
            return False
        
        # Test optional import (should fail gracefully)
        optional_result = manager.safe_import("nonexistent.module", required=False)
        if optional_result.status.value == "failed":
            tprint_success("✅ Optional import failed gracefully")
        else:
            tprint_warning("⚠️ Optional import should have failed")
        
        # Test required import (should fail with exception)
        try:
            manager.safe_import("nonexistent.module", required=True)
            tprint_error("❌ Required import should have failed")
            return False
        except ImportError:
            tprint_success("✅ Required import failed as expected")
        
        # Test cache stats
        stats = manager.get_cache_stats()
        tprint_info(f"📊 Cache stats: {stats}")
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ ImportManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_improved_feature_generation():
    """Test the improved feature generation logic."""
    tprint("🧪 Testing Improved Feature Generation...")
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.feature_generation_utils import (
            ImprovedFeatureGenerator, FeatureGenerationConfig
        )
        
        # Create test data
        data = create_test_data(500)
        tprint_info(f"📊 Created test data: {data.shape}")
        
        # Test base feature generation
        config = FeatureGenerationConfig(
            enable_technical_indicators=True,
            enable_rolling_stats=True,
            enable_interaction_features=False,
            enable_cross_timeframe=False,
            min_valid_ratio=0.8,
            max_constant_ratio=0.1
        )
        
        generator = ImprovedFeatureGenerator(config)
        features = generator.generate_meaningful_features(data)
        
        if not features.empty:
            tprint_success(f"✅ Generated {len(features.columns)} base features")
            tprint_info(f"📊 Features shape: {features.shape}")
            
            # Test feature validation
            validation_result = generator.validator.validate_features(features)
            tprint_info(f"📊 Quality score: {validation_result['quality_score']:.3f}")
            tprint_info(f"📊 Finite ratio: {validation_result['finite_ratio']:.3f}")
            tprint_info(f"📊 Constant ratio: {validation_result['constant_ratio']:.3f}")
            
            if validation_result['passed']:
                tprint_success("✅ Feature validation passed")
            else:
                tprint_warning(f"⚠️ Feature validation issues: {validation_result['issues']}")
        else:
            tprint_error("❌ No features generated")
            return False
        
        # Test interaction features
        config.enable_interaction_features = True
        config.enable_technical_indicators = False
        config.enable_rolling_stats = False
        
        generator = ImprovedFeatureGenerator(config)
        interactions = generator.generate_interaction_features(data)
        
        if not interactions.empty:
            tprint_success(f"✅ Generated {len(interactions.columns)} interaction features")
        else:
            tprint_warning("⚠️ No interaction features generated")
        
        # Test cross-timeframe features
        config.enable_cross_timeframe = True
        config.enable_interaction_features = False
        
        generator = ImprovedFeatureGenerator(config)
        cross_tf = generator.generate_cross_timeframe_features(data)
        
        if not cross_tf.empty:
            tprint_success(f"✅ Generated {len(cross_tf.columns)} cross-timeframe features")
        else:
            tprint_warning("⚠️ No cross-timeframe features generated")
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ Improved feature generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_integrated_component():
    """Test the integrated component with improvements."""
    tprint("🧪 Testing Integrated Component...")
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.interactive_feature_generation_component import (
            InteractiveFeatureGenerationComponent, InteractiveFeatureGenerationConfig
        )
        
        # Create test data
        data = create_test_data(500)
        tprint_info(f"📊 Created test data: {data.shape}")
        
        # Create component config
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
            variance_threshold=1e-8,
            top_k_per_family=50
        )
        
        component = InteractiveFeatureGenerationComponent()
        
        # Create training input
        training_input = {
            'data': data,
            'targets': {'target': data['target']}
        }
        
        pipeline_state = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframe': '15m'
        }
        
        # Execute component
        tprint_info("🚀 Executing integrated component...")
        result = await component.execute(training_input, pipeline_state)
        
        if result.success:
            tprint_success(f"✅ Component execution completed successfully!")
            tprint_info(f"⏱️ Execution time: {result.execution_time:.3f}s")
            
            # Check artifacts
            if hasattr(result, 'artifacts') and result.artifacts:
                artifacts = result.artifacts.interactive_feature_generation_result
                tprint_info(f"📊 Features generated: {len(artifacts.get('feature_names', []))}")
                tprint_info(f"📊 Selected features: {len(artifacts.get('selected_features', []))}")
                tprint_info(f"📊 Interaction features: {len(artifacts.get('interaction_features', pd.DataFrame()).columns)}")
                tprint_info(f"📊 Cross-timeframe features: {len(artifacts.get('cross_timeframe_features', pd.DataFrame()).columns)}")
                
                if len(artifacts.get('feature_names', [])) > 0:
                    tprint_success("✅ Component generated features successfully!")
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
        tprint_error(f"❌ Integrated component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_code_duplication_reduction():
    """Test that code duplication has been reduced."""
    tprint("🧪 Testing Code Duplication Reduction...")
    
    try:
        # Test that ImportManager is being used
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.interactive_feature_generation_component import import_manager
        
        if import_manager is not None:
            tprint_success("✅ ImportManager is being used in main component")
            
            # Check cache stats
            stats = import_manager.get_cache_stats()
            tprint_info(f"📊 Import cache stats: {stats}")
            
            # Test that we can get imported modules
            tprint_module = import_manager.get_imported_module("src.utils.tprint")
            if tprint_module is not None:
                tprint_success("✅ Successfully retrieved tprint module from cache")
            else:
                tprint_warning("⚠️ Could not retrieve tprint module from cache")
            
            return True
        else:
            tprint_error("❌ ImportManager not found in main component")
            return False
            
    except Exception as e:
        tprint_error(f"❌ Code duplication reduction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all improvement tests."""
    tprint("🚀 Starting Interactive Feature Generation Improvement Tests")
    tprint("=" * 70)
    
    tests = [
        ("ImportManager Utility", test_import_manager),
        ("Improved Feature Generation", test_improved_feature_generation),
        ("Code Duplication Reduction", test_code_duplication_reduction),
        ("Integrated Component", test_integrated_component),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        tprint(f"\n🧪 Running {test_name} Test...")
        tprint("-" * 50)
        
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
    tprint("\n" + "=" * 70)
    tprint("📊 IMPROVEMENT TEST SUMMARY")
    tprint("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        tprint(f"{status} {test_name}")
        if success:
            passed += 1
    
    tprint(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        tprint_success("🎉 All improvement tests passed! The improvements are working correctly.")
        return True
    else:
        tprint_error(f"⚠️ {total - passed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)