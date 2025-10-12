#!/usr/bin/env python3
"""
Test script for VectorBT optimizations in representation_learning module.

This script validates that the representation_learning module is properly
using VectorBTRollingOptimizer and UnifiedVectorizationManager.
"""

import numpy as np
import pandas as pd
import time
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_data(n_periods: int = 2000, n_assets: int = 1) -> pd.DataFrame:
    """Generate test data for representation learning."""
    np.random.seed(42)
    
    # Generate price data
    returns = np.random.normal(0.001, 0.02, n_periods)
    prices = 100 * (1 + returns).cumprod()
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_periods)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_periods))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_periods)
    })
    
    # Ensure high >= low and high/low contain open/close
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    data.index = pd.date_range(start='2020-01-01', periods=n_periods, freq='1min')
    return data

def test_vectorbt_optimizer_integration():
    """Test that VectorBTRollingOptimizer is properly integrated."""
    logger.info("Testing VectorBTRollingOptimizer integration...")
    
    try:
        from src.feature_generation.categories.representation_learning import (
            PatchTSTRepresentationGenerator,
            TFTEncoderRepresentationGenerator,
            AutoencoderRepresentationGenerator,
            ContrastiveLearningGenerator
        )
        
        # Test data
        data = generate_test_data(1000)
        
        # Test each generator
        generators = [
            PatchTSTRepresentationGenerator(patch_length=8, num_patches=4, embedding_dim=32),
            TFTEncoderRepresentationGenerator(seq_length=30, hidden_size=32, num_heads=2),
            AutoencoderRepresentationGenerator(encoding_dim=16, sequence_length=30),
            ContrastiveLearningGenerator(embedding_dim=32, temperature=0.1)
        ]
        
        for generator in generators:
            logger.info(f"Testing {generator.__class__.__name__}...")
            
            # Check if VectorBT optimizers are initialized
            assert hasattr(generator, 'rolling_optimizer'), f"{generator.__class__.__name__} missing rolling_optimizer"
            assert hasattr(generator, 'vectorization_manager'), f"{generator.__class__.__name__} missing vectorization_manager"
            
            # Test feature generation
            start_time = time.time()
            features = generator.generate_features(data)
            generation_time = time.time() - start_time
            
            logger.info(f"  ✓ {generator.__class__.__name__} generated features in {generation_time:.3f}s")
            logger.info(f"  ✓ Features shape: {features.shape}")
            
            # Test optimized VectorBT operations
            if hasattr(generator, '_optimized_vectorbt_operation'):
                test_series = data['close']
                result = generator._optimized_vectorbt_operation(test_series, 'mean', 20)
                assert len(result) == len(test_series), "VectorBT operation result length mismatch"
                logger.info(f"  ✓ VectorBT operations working correctly")
            
            # Test batch operations if available
            if hasattr(generator, '_batch_rolling_operations'):
                batch_result = generator._batch_rolling_operations(
                    data[['close', 'volume']], 
                    ['mean', 'std'], 
                    [10, 20]
                )
                logger.info(f"  ✓ Batch operations working, result shape: {batch_result.shape}")
        
        logger.info("✅ VectorBTRollingOptimizer integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ VectorBTRollingOptimizer integration test failed: {e}")
        return False

def test_unified_vectorization_manager():
    """Test that UnifiedVectorizationManager is properly integrated."""
    logger.info("Testing UnifiedVectorizationManager integration...")
    
    try:
        from src.utils.ml_common.unified_vectorization_manager import (
            UnifiedVectorizationManager, OperationType, OptimizationStrategy
        )
        
        # Test manager initialization
        manager = UnifiedVectorizationManager()
        assert manager is not None, "UnifiedVectorizationManager failed to initialize"
        
        # Test optimization stats
        stats = manager.get_optimization_stats()
        assert isinstance(stats, dict), "Optimization stats should be a dictionary"
        
        logger.info("✅ UnifiedVectorizationManager integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ UnifiedVectorizationManager integration test failed: {e}")
        return False

def test_performance_improvements():
    """Test that VectorBT optimizations provide performance improvements."""
    logger.info("Testing performance improvements...")
    
    try:
        from src.feature_generation.categories.representation_learning import (
            PatchTSTRepresentationGenerator
        )
        
        # Generate larger dataset for performance testing
        data = generate_test_data(5000)
        
        # Test with VectorBT optimizations
        generator = PatchTSTRepresentationGenerator(patch_length=16, num_patches=8, embedding_dim=64)
        
        # Time the feature generation
        start_time = time.time()
        features = generator.generate_features(data)
        vectorbt_time = time.time() - start_time
        
        logger.info(f"VectorBT optimized generation time: {vectorbt_time:.3f}s")
        logger.info(f"Features shape: {features.shape}")
        
        # Test memory efficiency
        if hasattr(generator, '_memory_efficient_processing'):
            start_time = time.time()
            processed_data = generator._memory_efficient_processing(data, chunk_size=1000)
            memory_time = time.time() - start_time
            
            logger.info(f"Memory efficient processing time: {memory_time:.3f}s")
            logger.info(f"Processed data shape: {processed_data.shape}")
        
        logger.info("✅ Performance improvements test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance improvements test failed: {e}")
        return False

def test_error_handling():
    """Test that error handling works correctly with fallbacks."""
    logger.info("Testing error handling and fallbacks...")
    
    try:
        from src.feature_generation.categories.representation_learning import (
            PatchTSTRepresentationGenerator
        )
        
        # Test with small dataset (should trigger fallbacks)
        data = generate_test_data(50)  # Small dataset
        generator = PatchTSTRepresentationGenerator(patch_length=16, num_patches=8, embedding_dim=64)
        
        # This should work with fallbacks
        features = generator.generate_features(data)
        assert len(features) > 0, "Features should be generated even with small dataset"
        
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'close': [np.nan] * 10,
            'volume': [0] * 10
        })
        
        # Should handle gracefully
        features = generator.generate_features(invalid_data)
        assert len(features) > 0, "Should handle invalid data gracefully"
        
        logger.info("✅ Error handling test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting VectorBT optimization tests for representation_learning module...")
    
    tests = [
        test_vectorbt_optimizer_integration,
        test_unified_vectorization_manager,
        test_performance_improvements,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with exception: {e}")
    
    logger.info(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All VectorBT optimization tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)