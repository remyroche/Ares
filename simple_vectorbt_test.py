#!/usr/bin/env python3
"""
Simple test to validate VectorBT optimizations in representation_learning module.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that the optimized modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test VectorBT rolling optimizer import
        from feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
        print("✅ VectorBTRollingOptimizer imported successfully")
        
        # Test unified vectorization manager import
        from utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager
        print("✅ UnifiedVectorizationManager imported successfully")
        
        # Test representation learning module import
        from feature_generation.categories.representation_learning import (
            PatchTSTRepresentationGenerator,
            TFTEncoderRepresentationGenerator,
            AutoencoderRepresentationGenerator,
            ContrastiveLearningGenerator
        )
        print("✅ Representation learning generators imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_vectorbt_optimizer():
    """Test VectorBTRollingOptimizer functionality."""
    print("\nTesting VectorBTRollingOptimizer...")
    
    try:
        from feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
        
        # Create optimizer
        optimizer = VectorBTRollingOptimizer(enable_gpu=False, enable_parallel=True)
        print("✅ VectorBTRollingOptimizer created successfully")
        
        # Test with sample data
        data = pd.Series(np.random.randn(1000))
        
        # Test rolling mean
        result = optimizer.rolling_mean(data, window=20)
        print(f"✅ Rolling mean test passed, result length: {len(result)}")
        
        # Test rolling std
        result = optimizer.rolling_std(data, window=20)
        print(f"✅ Rolling std test passed, result length: {len(result)}")
        
        return True
        
    except Exception as e:
        print(f"❌ VectorBTRollingOptimizer test failed: {e}")
        return False

def test_unified_manager():
    """Test UnifiedVectorizationManager functionality."""
    print("\nTesting UnifiedVectorizationManager...")
    
    try:
        from utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager, OperationType
        
        # Create manager
        manager = UnifiedVectorizationManager()
        print("✅ UnifiedVectorizationManager created successfully")
        
        # Test optimization stats
        stats = manager.get_optimization_stats()
        print(f"✅ Optimization stats retrieved: {type(stats)}")
        
        return True
        
    except Exception as e:
        print(f"❌ UnifiedVectorizationManager test failed: {e}")
        return False

def test_representation_generators():
    """Test representation learning generators."""
    print("\nTesting representation learning generators...")
    
    try:
        from feature_generation.categories.representation_learning import (
            PatchTSTRepresentationGenerator,
            TFTEncoderRepresentationGenerator,
            AutoencoderRepresentationGenerator,
            ContrastiveLearningGenerator
        )
        
        # Generate test data
        data = pd.DataFrame({
            'close': np.random.randn(1000).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 1000)
        })
        data.index = pd.date_range('2020-01-01', periods=1000, freq='1min')
        
        # Test each generator
        generators = [
            PatchTSTRepresentationGenerator(patch_length=8, num_patches=4, embedding_dim=16),
            TFTEncoderRepresentationGenerator(seq_length=20, hidden_size=16, num_heads=2),
            AutoencoderRepresentationGenerator(encoding_dim=8, sequence_length=20),
            ContrastiveLearningGenerator(embedding_dim=16, temperature=0.1)
        ]
        
        for i, generator in enumerate(generators):
            print(f"  Testing generator {i+1}: {generator.__class__.__name__}")
            
            # Check if optimizers are initialized
            assert hasattr(generator, 'rolling_optimizer'), "Missing rolling_optimizer"
            assert hasattr(generator, 'vectorization_manager'), "Missing vectorization_manager"
            
            # Test feature generation
            features = generator.generate_features(data)
            print(f"    ✅ Generated features with shape: {features.shape}")
        
        print("✅ All representation generators tested successfully")
        return True
        
    except Exception as e:
        print(f"❌ Representation generators test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Starting VectorBT optimization validation tests...\n")
    
    tests = [
        test_imports,
        test_vectorbt_optimizer,
        test_unified_manager,
        test_representation_generators
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All VectorBT optimization tests passed!")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)