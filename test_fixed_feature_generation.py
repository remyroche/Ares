#!/usr/bin/env python3
"""
Test Script for Fixed Interactive Feature Generation

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

async def test_feature_generation_utils():
    """Test the feature generation utilities directly."""
    tprint("🧪 Testing Feature Generation Utilities...")
    
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
            enable_cross_timeframe=False
        )
        
        generator = ImprovedFeatureGenerator(config)
        base_features = generator.generate_meaningful_features(data)
        
        tprint_success(f"✅ Generated {len(base_features.columns)} base features")
        tprint_info(f"📊 Base features shape: {base_features.shape}")
        
        if len(base_features.columns) > 0:
            tprint_info(f"📊 Sample base features: {list(base_features.columns[:10])}")
            return True
        else:
            tprint_error("❌ No base features generated!")
            return False
        
    except Exception as e:
        tprint_error(f"❌ Feature generation utilities test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_enhanced_orchestrator():
    """Test the enhanced orchestrator."""
    tprint("🧪 Testing Enhanced Orchestrator...")
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.enhanced_optimized_orchestrator import (
            EnhancedOptimizedInteractionOrchestrator, EnhancedOptimizedConfig
        )
        
        # Create test data
        data = create_test_data(500)
        tprint_info(f"📊 Created test data: {data.shape}")
        
        # Create orchestrator config
        config = EnhancedOptimizedConfig(
            enable_early_filtering=False,  # Disable for faster testing
            enable_interaction_pruning=False,  # Disable for faster testing
            enable_budgeted_optimization=False,  # Disable for faster testing
            enable_caching=False,  # Disable for faster testing
            enable_parallel_processing=False,  # Disable for testing
            max_workers=2,
            variance_threshold=1e-8,  # Use the fixed threshold
            top_k_per_family=50
        )
        
        orchestrator = EnhancedOptimizedInteractionOrchestrator(config)
        
        # Create training input
        training_input = {
            'data': data,
            'target_column': 'target'
        }
        
        pipeline_state = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframe': '15m',
            'execution_mode': 'test'
        }
        
        # Execute feature generation
        tprint_info("🚀 Executing feature generation...")
        result = await orchestrator.generate_features(training_input, pipeline_state)
        
        if result.success:
            tprint_success(f"✅ Feature generation completed successfully!")
            tprint_info(f"📊 Total features: {len(result.feature_names)}")
            tprint_info(f"📊 Selected features: {len(result.selected_features)}")
            tprint_info(f"📊 Interaction features: {len(result.interaction_features.columns)}")
            tprint_info(f"📊 Cross-timeframe features: {len(result.cross_timeframe_features.columns)}")
            tprint_info(f"⏱️ Execution time: {result.execution_time:.3f}s")
            tprint_info(f"💾 Memory usage: {result.memory_usage_mb:.1f} MB")
            
            # Check if we actually generated features
            if len(result.feature_names) > 0:
                tprint_success("✅ Features were actually generated!")
                return True
            else:
                tprint_error("❌ No features were generated!")
                return False
        else:
            tprint_error(f"❌ Feature generation failed: {result.error_message}")
            return False
            
    except Exception as e:
        tprint_error(f"❌ Enhanced orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_interactive_component():
    """Test the interactive feature generation component."""
    tprint("🧪 Testing Interactive Feature Generation Component...")
    
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
            variance_threshold=1e-8,  # Use the fixed threshold
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
        tprint_info("🚀 Executing interactive feature generation component...")
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
        tprint_error(f"❌ Interactive component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    tprint("🚀 Starting Fixed Interactive Feature Generation Tests")
    tprint("=" * 60)
    
    tests = [
        ("Feature Generation Utilities", test_feature_generation_utils),
        ("Enhanced Orchestrator", test_enhanced_orchestrator),
        ("Interactive Component", test_interactive_component),
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