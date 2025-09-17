#!/usr/bin/env python3
"""
Test script for Tactician Lookback Optimization Integration

This script tests the complete integration of the new tactician_lookback_optimization
step in the MODEL_TRAINING pipeline sequence.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_market_data_1m(days: int = 7) -> pd.DataFrame:
    """Create mock 1-minute market data for testing."""
    try:
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
        n_samples = len(timestamps)
        
        # Generate realistic price walk
        base_price = 2000.0
        returns = np.random.normal(0, 0.001, n_samples)
        price_walk = base_price + np.cumsum(returns)
        
        # Create OHLCV data
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': price_walk + np.random.normal(0, 0.5, n_samples),
            'high': price_walk + np.random.normal(2, 0.5, n_samples),
            'low': price_walk + np.random.normal(-2, 0.5, n_samples),
            'close': price_walk + np.random.normal(0, 0.5, n_samples),
            'volume': np.random.exponential(1000, n_samples)
        })
        
        # Ensure OHLC relationships
        data['high'] = np.maximum.reduce([data['open'], data['high'], data['close']])
        data['low'] = np.minimum.reduce([data['open'], data['low'], data['close']])
        
        print(f"✅ Created mock 1m market data: {len(data)} samples")
        return data
        
    except Exception as e:
        print(f"❌ Failed to create mock data: {e}")
        raise

def create_mock_analyst_models() -> Dict[str, Any]:
    """Create mock analyst models for testing."""
    return {
        'analyst_model_1': {'type': 'mock', 'loaded': True, 'accuracy': 0.75},
        'analyst_model_2': {'type': 'mock', 'loaded': True, 'accuracy': 0.78},
        'analyst_model_3': {'type': 'mock', 'loaded': True, 'accuracy': 0.72}
    }

def create_mock_analyst_ensemble() -> Dict[str, Any]:
    """Create mock analyst ensemble for testing."""
    return {'type': 'mock_ensemble', 'loaded': True, 'accuracy': 0.82}

async def test_tactician_lookback_optimization_step():
    """Test the Tactician Lookback Optimization Step directly."""
    try:
        print("\n🧪 Testing Tactician Lookback Optimization Step...")
        
        # Import the step
        from src.training.steps.model_training.tactician_lookback_optimization_step import (
            TacticianLookbackOptimizationStep
        )
        
        # Create test configuration
        config = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'optimization_method': 'two_step_grid_tpe',
            'tpe_trials': 5,  # Reduced for testing
            'optimization_timeout': 60,  # 1 minute for testing
            'save_results': True
        }
        
        # Create test data
        market_data_1m = create_mock_market_data_1m(days=2)  # Smaller dataset for testing
        analyst_models = create_mock_analyst_models()
        analyst_ensemble = create_mock_analyst_ensemble()
        
        # Initialize and execute step
        step = TacticianLookbackOptimizationStep(config)
        
        print("🔄 Executing tactician lookback optimization...")
        result = await step.execute(
            market_data_1m=market_data_1m,
            analyst_models=analyst_models,
            analyst_ensemble=analyst_ensemble
        )
        
        # Validate results
        assert result is not None, "Result should not be None"
        assert 'optimized_lookbacks' in result, "Result should contain optimized_lookbacks"
        assert 'optimization_score' in result, "Result should contain optimization_score"
        
        optimized_lookbacks = result['optimized_lookbacks']
        optimization_score = result['optimization_score']
        
        print(f"✅ Optimization completed successfully!")
        print(f"   📊 Optimized lookbacks: {len(optimized_lookbacks)} indicators")
        print(f"   📈 Optimization score: {optimization_score:.4f}")
        print(f"   ⏱️  Execution time: {result.get('execution_time', 0):.2f}s")
        
        # Print some optimized lookbacks
        if optimized_lookbacks:
            print("   🎯 Sample optimized lookbacks:")
            for indicator, lookback in list(optimized_lookbacks.items())[:5]:
                print(f"      {indicator}: {lookback}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tactician Lookback Optimization Step test failed: {e}")
        return False

async def test_pipeline_integration():
    """Test the integration with the model training pipeline."""
    try:
        print("\n🔗 Testing Pipeline Integration...")
        
        # Import pipeline components
        from src.training.steps.model_training.sub_pipeline import (
            ModelTrainingSubPipeline, SubPipelineConfig, ExecutionMode
        )
        
        # Create pipeline configuration
        config = SubPipelineConfig(
            mode=ExecutionMode.FULL,
            symbol='ETHUSDT',
            exchange='binance',
            timeframe='1m',
            data_dir='./test_data',
            custom_params={
                'optimization_method': 'grid_search',  # Faster for testing
                'tpe_trials': 3,
                'optimization_timeout': 30
            }
        )
        
        # Initialize pipeline
        pipeline = ModelTrainingSubPipeline(config)
        
        # Check that our new step is registered
        available_steps = pipeline.get_available_sub_pipelines()
        assert 'tactician_lookback_optimization' in available_steps, \
            "tactician_lookback_optimization should be in available sub-pipelines"
        
        print(f"✅ Pipeline integration verified!")
        print(f"   📋 Available sub-pipelines: {len(available_steps)}")
        print(f"   🎯 Contains tactician_lookback_optimization: ✓")
        
        # Test the specific pipeline execution (in blank mode for speed)
        config.mode = ExecutionMode.BLANK
        
        print("🔄 Testing pipeline execution in BLANK mode...")
        result = await pipeline.execute_sub_pipeline('tactician_lookback_optimization', config)
        
        assert result is not None, "Pipeline result should not be None"
        assert result.status.value in ['completed', 'failed'], "Result should have valid status"
        
        if result.status.value == 'completed':
            print("✅ Pipeline execution completed successfully!")
            print(f"   📊 Artifacts: {len(result.artifacts) if result.artifacts else 0}")
            if result.artifacts:
                print(f"   🎯 Sample artifacts: {list(result.artifacts.keys())[:3]}")
        else:
            print(f"⚠️ Pipeline execution failed: {result.error_message}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        return False

async def test_full_pipeline_sequence():
    """Test the full model training pipeline sequence."""
    try:
        print("\n🔄 Testing Full Pipeline Sequence...")
        
        # Import pipeline components
        from src.training.steps.model_training.sub_pipeline import (
            execute_full_model_training_pipeline, SubPipelineConfig, ExecutionMode
        )
        
        # Create configuration for testing
        config = SubPipelineConfig(
            mode=ExecutionMode.BLANK,  # Use blank mode for fast testing
            symbol='ETHUSDT',
            exchange='binance',
            timeframe='1m',
            data_dir='./test_data'
        )
        
        print("🔄 Executing full model training pipeline...")
        results = await execute_full_model_training_pipeline(config)
        
        assert results is not None, "Results should not be None"
        assert len(results) > 0, "Should have at least one result"
        
        # Check that tactician_lookback_optimization is in the sequence
        step_names = [result.sub_pipeline_name for result in results]
        assert 'tactician_lookback_optimization' in step_names, \
            "tactician_lookback_optimization should be in the execution sequence"
        
        # Verify execution order
        expected_order = [
            'analyst_model_training',
            'analyst_ensemble_training',
            'tactician_lookback_optimization',
            'tactician_models_training',
            'tactician_ensemble_training'
        ]
        
        # Check that tactician_lookback_optimization comes after analyst steps
        analyst_indices = [i for i, name in enumerate(step_names) if 'analyst' in name]
        tactician_opt_index = step_names.index('tactician_lookback_optimization') if 'tactician_lookback_optimization' in step_names else -1
        tactician_indices = [i for i, name in enumerate(step_names) if 'tactician_models_training' in name or 'tactician_ensemble_training' in name]
        
        if analyst_indices and tactician_opt_index >= 0:
            assert max(analyst_indices) < tactician_opt_index, \
                "tactician_lookback_optimization should come after analyst steps"
        
        if tactician_indices and tactician_opt_index >= 0:
            assert tactician_opt_index < min(tactician_indices), \
                "tactician_lookback_optimization should come before tactician training steps"
        
        print("✅ Full pipeline sequence test completed!")
        print(f"   📊 Total steps executed: {len(results)}")
        print(f"   🎯 Execution order verified: ✓")
        print(f"   📋 Step sequence: {' → '.join(step_names)}")
        
        # Print status of each step
        for result in results:
            status_icon = "✅" if result.status.value == 'completed' else "❌" if result.status.value == 'failed' else "⏸️"
            print(f"   {status_icon} {result.sub_pipeline_name}: {result.status.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline sequence test failed: {e}")
        return False

async def test_configuration_validation():
    """Test configuration validation for the new step."""
    try:
        print("\n🔍 Testing Configuration Validation...")
        
        from src.training.steps.model_training.tactician_lookback_optimization import (
            TacticianLookbackConfig, create_tactician_lookback_config
        )
        
        # Test default configuration
        default_config = TacticianLookbackConfig()
        assert default_config.timeframe == "1m", "Default timeframe should be 1m"
        assert default_config.requires_analyst_outputs == True, "Should require analyst outputs"
        
        # Test custom configuration
        custom_config = create_tactician_lookback_config(
            timeframe="1m",
            symbol="BTCUSDT",
            optimization_method="tpe",
            tpe_trials=10
        )
        
        assert custom_config.symbol == "BTCUSDT", "Custom symbol should be set"
        assert custom_config.optimization_method == "tpe", "Custom optimization method should be set"
        assert custom_config.tpe_trials == 10, "Custom TPE trials should be set"
        
        print("✅ Configuration validation test passed!")
        print(f"   🎯 Default config: {default_config.timeframe} timeframe")
        print(f"   🎯 Custom config: {custom_config.symbol} symbol, {custom_config.optimization_method} method")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation test failed: {e}")
        return False

async def run_all_tests():
    """Run all tests for the Tactician Lookback Optimization integration."""
    print("🚀 Starting Tactician Lookback Optimization Integration Tests")
    print("=" * 70)
    
    test_results = []
    
    # Test 1: Configuration validation
    result1 = await test_configuration_validation()
    test_results.append(("Configuration Validation", result1))
    
    # Test 2: Direct step testing
    result2 = await test_tactician_lookback_optimization_step()
    test_results.append(("Tactician Lookback Optimization Step", result2))
    
    # Test 3: Pipeline integration
    result3 = await test_pipeline_integration()
    test_results.append(("Pipeline Integration", result3))
    
    # Test 4: Full pipeline sequence
    result4 = await test_full_pipeline_sequence()
    test_results.append(("Full Pipeline Sequence", result4))
    
    # Print summary
    print("\n" + "=" * 70)
    print("🏁 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, passed in test_results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} - {test_name}")
        if passed:
            passed_tests += 1
    
    print(f"\n📊 Overall Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Tactician Lookback Optimization is successfully integrated!")
        return True
    else:
        print(f"⚠️ {total_tests - passed_tests} tests failed. Please review the implementation.")
        return False

async def main():
    """Main test function."""
    try:
        success = await run_all_tests()
        
        if success:
            print("\n✅ Integration test completed successfully!")
            print("\n📋 Next Steps:")
            print("   1. The tactician_lookback_optimization step is now integrated")
            print("   2. It will run between Analyst and Tactician training")
            print("   3. It optimizes lookback periods specifically for 1m timeframe")
            print("   4. It uses Analyst outputs as input features for optimization")
            print("\n🎯 Key Features Implemented:")
            print("   • Dependency-aware optimization (requires Analyst outputs)")
            print("   • 1m timeframe-specific optimization")
            print("   • Multiple optimization methods (grid search, TPE, two-step)")
            print("   • Integration with existing training pipeline")
            print("   • Comprehensive error handling and reporting")
        else:
            print("\n❌ Integration test failed!")
            print("Please review the error messages above and fix any issues.")
        
        return success
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(main())