#!/usr/bin/env python3

"""
Test script for unified training integration.

This script tests the new unified training step to ensure it works correctly
with ares_launcher and the UnifiedTrainingPipeline.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.steps.model_training.unified_models_training_step import UnifiedModelsTrainingStep


async def test_unified_training():
    """Test the unified training step."""
    print("🧪 Testing Unified Models Training Step...")
    
    # Test configuration
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'direction': 'long',
        'training_type': 'analyst_base',
        'execution_mode': 'light'
    }
    
    try:
        # Create unified training step
        training_step = UnifiedModelsTrainingStep()
        
        # Test execution
        print(f"Testing {config['training_type']} training...")
        result = await training_step.execute(config)
        
        # Print results
        print(f"✅ Test completed successfully!")
        print(f"Success: {result.get('success', False)}")
        print(f"Training type: {result.get('training_type', 'unknown')}")
        print(f"Execution time: {result.get('execution_time', 0.0):.2f}s")
        
        if result.get('artifacts'):
            print(f"Artifacts created: {len(result['artifacts'])}")
            for name, path in result['artifacts'].items():
                print(f"  - {name}: {path}")
        
        if result.get('metrics'):
            print(f"Metrics: {result['metrics']}")
        
        if result.get('error'):
            print(f"Error: {result['error']}")
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_all_training_types():
    """Test all training types."""
    training_types = ['analyst_base', 'analyst_ensemble', 'tactician_base', 'tactician_ensemble']
    
    print("🧪 Testing all training types...")
    
    results = {}
    for training_type in training_types:
        print(f"\n--- Testing {training_type} ---")
        
        config = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframe': '15m',
            'direction': 'long',
            'training_type': training_type,
            'execution_mode': 'light'
        }
        
        try:
            training_step = UnifiedModelsTrainingStep()
            result = await training_step.execute(config)
            results[training_type] = result.get('success', False)
            
            status = "✅ Success" if result.get('success', False) else "❌ Failed"
            print(f"{training_type}: {status}")
            
        except Exception as e:
            print(f"{training_type}: ❌ Error - {e}")
            results[training_type] = False
    
    # Summary
    print(f"\n📊 Test Summary:")
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    print(f"Successful: {successful}/{total}")
    
    for training_type, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {training_type}")
    
    return successful == total


if __name__ == "__main__":
    print("🚀 Starting Unified Training Tests...")
    
    # Test single training type
    success = asyncio.run(test_unified_training())
    
    if success:
        print("\n" + "="*50)
        # Test all training types
        all_success = asyncio.run(test_all_training_types())
        
        if all_success:
            print("\n🎉 All tests passed! Unified training is working correctly.")
        else:
            print("\n⚠️  Some tests failed. Check the output above for details.")
    else:
        print("\n❌ Basic test failed. Check the error messages above.")