#!/usr/bin/env python3
"""
Simple test for NAS Trainer implementation.
This test verifies the basic structure and functionality without external dependencies.
"""

import sys
import os
sys.path.append('/workspace')

def test_nas_trainer_import():
    """Test that the NAS Trainer can be imported."""
    try:
        # Test basic imports
        from nas_trainer import NASConfig, NASTrainer, create_sample_data
        print("✅ NAS Trainer imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_nas_config():
    """Test NAS configuration."""
    try:
        from nas_trainer import NASConfig
        
        config = NASConfig(
            search_strategy='random',
            max_trials=10,
            max_epochs=5,
            use_m1_optimization=False,  # Disable for testing
            verbose=False
        )
        
        print(f"✅ NAS Config created: {config.search_strategy}")
        print(f"   - Max trials: {config.max_trials}")
        print(f"   - Max epochs: {config.max_epochs}")
        print(f"   - M1 optimization: {config.use_m1_optimization}")
        
        return True
    except Exception as e:
        print(f"❌ NAS Config test failed: {e}")
        return False

def test_sample_data_creation():
    """Test sample data creation."""
    try:
        from nas_trainer import create_sample_data
        
        # Create small sample data
        X, y = create_sample_data(n_samples=50, n_features=5)
        
        print(f"✅ Sample data created:")
        print(f"   - X shape: {X.shape}")
        print(f"   - y shape: {y.shape}")
        print(f"   - X type: {type(X)}")
        print(f"   - y type: {type(y)}")
        
        return True
    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
        return False

def test_nas_trainer_initialization():
    """Test NAS Trainer initialization."""
    try:
        from nas_trainer import NASConfig, NASTrainer
        
        config = NASConfig(
            max_trials=5,
            max_epochs=3,
            use_m1_optimization=False,
            verbose=False
        )
        
        nas_trainer = NASTrainer(config)
        print("✅ NAS Trainer initialized successfully")
        
        # Test basic attributes
        print(f"   - Config: {nas_trainer.config.search_strategy}")
        print(f"   - Output dir: {nas_trainer.output_dir}")
        
        # Cleanup
        nas_trainer.cleanup()
        print("✅ NAS Trainer cleanup completed")
        
        return True
    except Exception as e:
        print(f"❌ NAS Trainer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_architecture_generation():
    """Test architecture generation."""
    try:
        from nas_trainer import NASConfig, NASTrainer
        
        config = NASConfig(
            max_trials=5,
            max_epochs=3,
            use_m1_optimization=False,
            verbose=False
        )
        
        nas_trainer = NASTrainer(config)
        
        # Test architecture generation
        architecture = nas_trainer.generate_architecture(0)
        
        print("✅ Architecture generated successfully")
        print(f"   - Trial ID: {architecture['trial_id']}")
        print(f"   - Layers: {architecture['n_layers']}")
        print(f"   - Learning rate: {architecture['learning_rate']}")
        print(f"   - Batch size: {architecture['batch_size']}")
        
        nas_trainer.cleanup()
        return True
    except Exception as e:
        print(f"❌ Architecture generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_preparation():
    """Test data preparation functionality."""
    try:
        from nas_trainer import NASConfig, NASTrainer, create_sample_data
        
        config = NASConfig(
            max_trials=5,
            max_epochs=3,
            use_m1_optimization=False,
            verbose=False
        )
        
        nas_trainer = NASTrainer(config)
        
        # Create sample data
        X, y = create_sample_data(n_samples=100, n_features=10)
        
        # Test data preparation
        data_splits = nas_trainer.prepare_data(X, y)
        
        print("✅ Data preparation successful")
        print(f"   - Train shape: {data_splits['X_train'].shape}")
        print(f"   - Val shape: {data_splits['X_val'].shape}")
        print(f"   - Test shape: {data_splits['X_test'].shape}")
        print(f"   - Feature names: {len(data_splits['feature_names'])}")
        
        nas_trainer.cleanup()
        return True
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Testing NAS Trainer Implementation")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_nas_trainer_import),
        ("Config Test", test_nas_config),
        ("Sample Data Test", test_sample_data_creation),
        ("Initialization Test", test_nas_trainer_initialization),
        ("Architecture Generation Test", test_architecture_generation),
        ("Data Preparation Test", test_data_preparation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! NAS Trainer implementation is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)