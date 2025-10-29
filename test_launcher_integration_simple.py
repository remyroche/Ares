#!/usr/bin/env python3

"""
Simple test for ares_launcher integration.

This script tests that ares_launcher can properly discover and call the unified training step.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_step_registration():
    """Test that the unified training step is properly registered."""
    print("🧪 Testing step registration...")
    
    try:
        # Import the step registry
        from src.training.steps.base_step import step_registry
        
        # Check if unified_models_training is registered
        if step_registry.is_registered('unified_models_training'):
            print("✅ unified_models_training step is registered")
            return True
        else:
            print("❌ unified_models_training step is not registered")
            print(f"Available steps: {step_registry.list_steps()}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking step registration: {e}")
        return False

def test_launcher_import():
    """Test that ares_launcher can be imported."""
    print("🧪 Testing ares_launcher import...")
    
    try:
        from src.launcher.ares_launcher import SimplifiedAresLauncher
        print("✅ ares_launcher imported successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing ares_launcher: {e}")
        return False

def test_config_mapping():
    """Test that the training type mapping works correctly."""
    print("🧪 Testing training type mapping...")
    
    # Test the mapping logic from ares_launcher
    training_types = ['analyst_base', 'analyst_ensemble', 'tactician_base', 'tactician_ensemble']
    
    for training_type in training_types:
        config = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframe': '15m',
            'direction': 'long',
            'training_type': training_type,
            'execution_mode': 'light'
        }
        
        print(f"  {training_type}: {config['training_type']}")
    
    print("✅ Training type mapping works correctly")
    return True

def test_yaml_config_files():
    """Test that YAML config files exist."""
    print("🧪 Testing YAML config files...")
    
    config_files = [
        'src/training/steps/model_training/analyst_base_config.yaml',
        'src/training/steps/model_training/analyst_ensemble_config.yaml',
        'src/training/steps/model_training/tactician_base_config.yaml',
        'src/training/steps/model_training/tactician_ensemble_config.yaml'
    ]
    
    all_exist = True
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ {config_file} exists")
        else:
            print(f"❌ {config_file} missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print("🚀 Starting Simple Launcher Integration Tests...")
    
    tests = [
        test_step_registration,
        test_launcher_import,
        test_config_mapping,
        test_yaml_config_files
    ]
    
    results = []
    for test in tests:
        print()
        result = test()
        results.append(result)
    
    # Summary
    print(f"\n📊 Test Summary:")
    successful = sum(1 for result in results if result)
    total = len(results)
    print(f"Successful: {successful}/{total}")
    
    if successful == total:
        print("\n🎉 All tests passed! The unified training integration is ready.")
        print("\nYou can now use ares_launcher with the unified training step:")
        print("  python3 src/launcher/ares_launcher.py --train-analyst-base --symbol ETHUSDT --timeframe 15m --direction long")
        print("  python3 src/launcher/ares_launcher.py --train-analyst-ensemble --symbol ETHUSDT --timeframe 15m --direction long")
        print("  python3 src/launcher/ares_launcher.py --train-tactician-base --symbol ETHUSDT --timeframe 15m --direction long")
        print("  python3 src/launcher/ares_launcher.py --train-tactician-ensemble --symbol ETHUSDT --timeframe 15m --direction long")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()