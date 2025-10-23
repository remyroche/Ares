#!/usr/bin/env python3
"""
Test critical VectorBT components integration.

This script tests the VectorBTRollingOptimizer and UnifiedVectorizationManager
to ensure they're properly integrated with the new VectorBT production module.
"""

import sys
import os
from pathlib import Path

# Add workspace root to Python path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

import logging
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_vectorbt_rolling_optimizer():
    """Test VectorBTRollingOptimizer integration."""
    logger.info("🧪 Testing VectorBTRollingOptimizer...")
    
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import (
            VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
        )
        
        # Create test data
        data = pd.Series(np.random.randn(1000))
        
        # Test optimizer initialization
        optimizer = get_vectorbt_rolling_optimizer()
        logger.info(f"✅ VectorBTRollingOptimizer initialized: VectorBT={optimizer.use_vectorbt}")
        
        # Test basic rolling operations
        window = 20
        
        # Test rolling mean
        result_mean = optimizer.rolling_mean(data, window)
        logger.info(f"✅ Rolling mean: input_shape={data.shape}, output_shape={result_mean.shape}")
        
        # Test rolling std
        result_std = optimizer.rolling_std(data, window)
        logger.info(f"✅ Rolling std: input_shape={data.shape}, output_shape={result_std.shape}")
        
        # Test rolling correlation
        other_data = pd.Series(np.random.randn(1000))
        result_corr = optimizer.rolling_corr(data, other_data, window)
        logger.info(f"✅ Rolling correlation: input_shape={data.shape}, output_shape={result_corr.shape}")
        
        # Test performance stats
        stats = optimizer.get_performance_stats()
        logger.info(f"✅ Performance stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ VectorBTRollingOptimizer test failed: {e}")
        return False

def test_unified_vectorization_manager():
    """Test UnifiedVectorizationManager integration."""
    logger.info("🧪 Testing UnifiedVectorizationManager...")
    
    try:
        from src.feature_generation.utils.unified_vectorization_manager import (
            UnifiedVectorizationManager, get_unified_vectorization_manager
        )
        
        # Create test data
        data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000)
        })
        
        # Test manager initialization
        manager = get_unified_vectorization_manager()
        logger.info(f"✅ UnifiedVectorizationManager initialized: VectorBT={manager.vectorbt_available}")
        
        # Test vectorization operations
        operations = ['mean', 'std', 'min', 'max']
        window = 20
        
        # Test single operation
        result = manager.vectorize_features(data, operations, window)
        logger.info(f"✅ Single operation: input_shape={data.shape}, output_shape={result.shape}")
        
        # Test batch operations
        batch_result = manager.batch_vectorize_features(data, operations, window)
        logger.info(f"✅ Batch operations: input_shape={data.shape}, output_shape={batch_result.shape}")
        
        # Test performance stats
        stats = manager.get_performance_stats()
        logger.info(f"✅ Performance stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ UnifiedVectorizationManager test failed: {e}")
        return False

def test_vectorbt_imports():
    """Test VectorBT imports in critical components."""
    logger.info("🧪 Testing VectorBT imports...")
    
    try:
        # Test VectorBTRollingOptimizer imports
        from src.feature_generation.utils.vectorbt_rolling_optimizer import vbt, VECTORBT_AVAILABLE
        logger.info(f"✅ VectorBTRollingOptimizer imports: VectorBT={VECTORBT_AVAILABLE}")
        
        # Test UnifiedVectorizationManager imports
        from src.feature_generation.utils.unified_vectorization_manager import vbt, VECTORBT_AVAILABLE
        logger.info(f"✅ UnifiedVectorizationManager imports: VectorBT={VECTORBT_AVAILABLE}")
        
        # Test VectorBT functions
        from src.feature_generation.utils.unified_vectorization_manager import (
            rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max
        )
        logger.info("✅ VectorBT functions imported successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ VectorBT imports test failed: {e}")
        return False

def test_integration_consistency():
    """Test that both components use the same VectorBT module."""
    logger.info("🧪 Testing integration consistency...")
    
    try:
        # Import from both components
        from src.feature_generation.utils.vectorbt_rolling_optimizer import vbt as vbt1, VECTORBT_AVAILABLE as vb1
        from src.feature_generation.utils.unified_vectorization_manager import vbt as vbt2, VECTORBT_AVAILABLE as vb2
        
        # Check that they're the same
        if vbt1 is vbt2 and vb1 == vb2:
            logger.info("✅ Both components use the same VectorBT module")
            return True
        else:
            logger.error("❌ Components use different VectorBT modules")
            return False
            
    except Exception as e:
        logger.error(f"❌ Integration consistency test failed: {e}")
        return False

def main():
    """Run all critical component tests."""
    logger.info("🚀 Starting critical VectorBT components integration test")
    
    tests = [
        ("VectorBT Imports", test_vectorbt_imports),
        ("Integration Consistency", test_integration_consistency),
        ("VectorBTRollingOptimizer", test_vectorbt_rolling_optimizer),
        ("UnifiedVectorizationManager", test_unified_vectorization_manager),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All critical components are properly integrated!")
        return True
    else:
        logger.error("❌ Some critical components failed integration tests")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)