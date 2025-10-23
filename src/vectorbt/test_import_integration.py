#!/usr/bin/env python3
"""
Test VectorBT import integration for critical components.

This script tests that the critical components are properly importing
from the new VectorBT production module.
"""

import sys
import os
from pathlib import Path

# Add workspace root to Python path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_vectorbt_rolling_optimizer_imports():
    """Test VectorBTRollingOptimizer imports."""
    logger.info("🧪 Testing VectorBTRollingOptimizer imports...")
    
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import (
            vbt, VECTORBT_AVAILABLE, rolling_corr, rolling_cov
        )
        
        logger.info(f"✅ VectorBTRollingOptimizer imports successful")
        logger.info(f"   - vbt: {type(vbt)}")
        logger.info(f"   - VECTORBT_AVAILABLE: {VECTORBT_AVAILABLE}")
        logger.info(f"   - rolling_corr: {type(rolling_corr)}")
        logger.info(f"   - rolling_cov: {type(rolling_cov)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ VectorBTRollingOptimizer imports failed: {e}")
        return False

def test_unified_vectorization_manager_imports():
    """Test UnifiedVectorizationManager imports."""
    logger.info("🧪 Testing UnifiedVectorizationManager imports...")
    
    try:
        from src.feature_generation.utils.unified_vectorization_manager import (
            vbt, VECTORBT_AVAILABLE, rolling_mean, rolling_std, rolling_var,
            rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov,
            scale, rank, zscore, winsorize, clip, quantile
        )
        
        logger.info(f"✅ UnifiedVectorizationManager imports successful")
        logger.info(f"   - vbt: {type(vbt)}")
        logger.info(f"   - VECTORBT_AVAILABLE: {VECTORBT_AVAILABLE}")
        logger.info(f"   - rolling_mean: {type(rolling_mean)}")
        logger.info(f"   - rolling_std: {type(rolling_std)}")
        logger.info(f"   - rolling_var: {type(rolling_var)}")
        logger.info(f"   - rolling_min: {type(rolling_min)}")
        logger.info(f"   - rolling_max: {type(rolling_max)}")
        logger.info(f"   - rolling_sum: {type(rolling_sum)}")
        logger.info(f"   - rolling_apply: {type(rolling_apply)}")
        logger.info(f"   - rolling_corr: {type(rolling_corr)}")
        logger.info(f"   - rolling_cov: {type(rolling_cov)}")
        logger.info(f"   - scale: {type(scale)}")
        logger.info(f"   - rank: {type(rank)}")
        logger.info(f"   - zscore: {type(zscore)}")
        logger.info(f"   - winsorize: {type(winsorize)}")
        logger.info(f"   - clip: {type(clip)}")
        logger.info(f"   - quantile: {type(quantile)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ UnifiedVectorizationManager imports failed: {e}")
        return False

def test_import_consistency():
    """Test that both components import from the same VectorBT module."""
    logger.info("🧪 Testing import consistency...")
    
    try:
        # Import from both components
        from src.feature_generation.utils.vectorbt_rolling_optimizer import vbt as vbt1, VECTORBT_AVAILABLE as vb1
        from src.feature_generation.utils.unified_vectorization_manager import vbt as vbt2, VECTORBT_AVAILABLE as vb2
        
        # Check that they're the same
        if vbt1 is vbt2 and vb1 == vb2:
            logger.info("✅ Both components import from the same VectorBT module")
            return True
        else:
            logger.error("❌ Components import from different VectorBT modules")
            logger.error(f"   - vbt1: {vbt1}")
            logger.error(f"   - vbt2: {vbt2}")
            logger.error(f"   - vb1: {vb1}")
            logger.error(f"   - vb2: {vb2}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Import consistency test failed: {e}")
        return False

def test_no_fallback_logic():
    """Test that no fallback logic remains in critical components."""
    logger.info("🧪 Testing for remaining fallback logic...")
    
    try:
        # Check VectorBTRollingOptimizer
        with open('src/feature_generation/utils/vectorbt_rolling_optimizer.py', 'r') as f:
            content1 = f.read()
        
        # Check UnifiedVectorizationManager
        with open('src/feature_generation/utils/unified_vectorization_manager.py', 'r') as f:
            content2 = f.read()
        
        # Look for VectorBT-specific fallback patterns
        fallback_patterns = [
            'import vectorbt as vbt',
            'VECTORBT_AVAILABLE = False',
            'warnings.warn.*vectorbt',
            'fallback.*vectorbt'
        ]
        
        issues = []
        
        for pattern in fallback_patterns:
            if pattern in content1:
                issues.append(f"VectorBTRollingOptimizer contains: {pattern}")
            if pattern in content2:
                issues.append(f"UnifiedVectorizationManager contains: {pattern}")
        
        if issues:
            logger.warning("⚠️ Found potential fallback logic:")
            for issue in issues:
                logger.warning(f"   - {issue}")
            return False
        else:
            logger.info("✅ No fallback logic found in critical components")
            return True
            
    except Exception as e:
        logger.error(f"❌ Fallback logic test failed: {e}")
        return False

def main():
    """Run all import integration tests."""
    logger.info("🚀 Starting VectorBT import integration test")
    
    tests = [
        ("VectorBTRollingOptimizer Imports", test_vectorbt_rolling_optimizer_imports),
        ("UnifiedVectorizationManager Imports", test_unified_vectorization_manager_imports),
        ("Import Consistency", test_import_consistency),
        ("No Fallback Logic", test_no_fallback_logic),
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
        logger.info("🎉 All import integration tests passed!")
        return True
    else:
        logger.error("❌ Some import integration tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)