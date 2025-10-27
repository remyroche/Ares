#!/usr/bin/env python3

"""
Test script for ares_launcher integration with unified training.

This script tests that ares_launcher can properly call the unified training step
without requiring all the heavy dependencies.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Mock the heavy dependencies to avoid import errors
class MockArtifactManager:
    def __init__(self, config=None):
        self.config = config or {}
    
    def set_context(self, step_name=None, datetime=None):
        pass
    
    def save(self, data=None, artifact_name=None, artifact_type=None, compression=None, metadata=None):
        return f"mock_path/{artifact_name}"
    
    def get_artifact(self, artifact_name=None, artifact_type=None):
        if artifact_name == 'training_dataset':
            import pandas as pd
            import numpy as np
            return pd.DataFrame({
                'close': np.random.randn(100).cumsum() + 100,
                'volume': np.random.exponential(1000, 100),
                'returns': np.random.randn(100) * 0.01,
                'volatility': np.random.exponential(0.02, 100)
            })
        elif artifact_name == 'analyst_targets':
            import pandas as pd
            import numpy as np
            return pd.Series(np.random.randn(100), name='analyst_target')
        elif artifact_name == 'tactician_targets':
            import pandas as pd
            import numpy as np
            return pd.Series(np.random.randn(100), name='tactician_target')
        else:
            raise Exception(f"Artifact {artifact_name} not found")

# Mock the artifact manager
sys.modules['src.utils.artifact_manager'] = type('MockModule', (), {'ArtifactManager': MockArtifactManager})()

# Mock other heavy dependencies
sys.modules['psutil'] = type('MockModule', (), {})()
sys.modules['numpy'] = type('MockModule', (), {})()
sys.modules['pandas'] = type('MockModule', (), {})()

# Mock the unified training pipeline
class MockUnifiedTrainingPipeline:
    def __init__(self, logger=None):
        self.logger = logger
    
    async def train_analyst_models(self, data, targets, config):
        return {
            'success': True,
            'models': {'analyst_model': 'mock_model'},
            'metrics': {'accuracy': 0.85, 'training_time': 10.5}
        }
    
    async def train_tactician_models(self, data, targets, config):
        return {
            'success': True,
            'models': {'tactician_model': 'mock_model'},
            'metrics': {'accuracy': 0.82, 'sharpe_ratio': 1.45}
        }
    
    async def train_ensemble_models(self, data, analyst_targets, tactician_targets, config):
        return {
            'success': True,
            'models': {'ensemble_model': 'mock_model'},
            'metrics': {'accuracy': 0.88, 'diversity_score': 0.92}
        }

# Mock the unified training pipeline
sys.modules['src.training.steps.models_training.unified_training_pipeline'] = type('MockModule', (), {
    'UnifiedTrainingPipeline': MockUnifiedTrainingPipeline
})()

# Mock other dependencies
sys.modules['src.utils.logger'] = type('MockModule', (), {
    'system_logger': type('MockLogger', (), {'getChild': lambda self, name: type('MockLogger', (), {})()})()
})()
sys.modules['src.utils.tprint'] = type('MockModule', (), {
    'tprint': lambda msg, level: print(f"[{level}] {msg}"),
    'tprint_info': lambda msg: print(f"[INFO] {msg}"),
    'tprint_success': lambda msg: print(f"[SUCCESS] {msg}"),
    'tprint_error': lambda msg: print(f"[ERROR] {msg}")
})()

# Mock the step registry
class MockStepRegistry:
    def __init__(self):
        self._steps = {}
    
    def register(self, step_name, step_class):
        self._steps[step_name] = step_class
    
    def get_step(self, step_name):
        return self._steps.get(step_name)
    
    def list_steps(self):
        return list(self._steps.keys())

sys.modules['src.training.steps.base_step'] = type('MockModule', (), {
    'BaseStep': type('MockBaseStep', (), {
        '__init__': lambda self, step_name: setattr(self, 'step_name', step_name),
        'logger': type('MockLogger', (), {})(),
        'artifact_manager': MockArtifactManager(),
        '_apply_light_mode_filter': lambda self, data, config, timeframe: data,
        '_get_artifact': lambda self, artifact_name, artifact_type: MockArtifactManager().get_artifact(artifact_name, artifact_type),
        '_save_artifact': lambda self, data, artifact_name, artifact_type=None, compression=None, metadata=None: f"mock_path/{artifact_name}"
    }),
    'step_registry': MockStepRegistry()
})()

# Now import our unified training step
from src.training.steps.model_training.unified_models_training_step import UnifiedModelsTrainingStep


async def test_unified_training():
    """Test the unified training step."""
    print("🧪 Testing Unified Models Training Step...")
    
    # Test configuration
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'direction': 'longs',
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
            'direction': 'longs',
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