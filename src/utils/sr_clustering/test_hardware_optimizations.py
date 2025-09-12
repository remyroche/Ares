#!/usr/bin/env python3
"""
Test Hardware Optimizations Integration.

This script tests that all hardware optimization components are properly integrated
without requiring external dependencies.
"""

import sys
import os
import logging

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hardware_imports():
    """Test that hardware optimization modules can be imported."""
    logger.info("🔍 Testing hardware optimization imports...")
    
    try:
        # Test M1 optimizations
        from utils.hardware.m1_optimizations import M1MemoryOptimizer
        logger.info("✅ M1MemoryOptimizer imported successfully")
        
        from utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
        logger.info("✅ M1CPUOptimizer imported successfully")
        
        from utils.hardware.m1_gpu_utils import M1GPUManager
        logger.info("✅ M1GPUManager imported successfully")
        
        return True
        
    except ImportError as e:
        logger.warning(f"⚠️ Hardware optimization imports failed: {e}")
        return False

def test_parameter_optimization_engine():
    """Test that ParameterOptimizationEngine can be imported with hardware optimizations."""
    logger.info("🔍 Testing ParameterOptimizationEngine with hardware optimizations...")
    
    try:
        from utils.sr_clustering.parameter_optimization_engine import (
            ParameterOptimizationEngine, 
            ParameterOptimizationConfig
        )
        logger.info("✅ ParameterOptimizationEngine imported successfully")
        
        # Test configuration with hardware optimizations
        config = ParameterOptimizationConfig(
            enable_hardware_optimization=True,
            enable_parallel_processing=True,
            enable_gpu_acceleration=True,
            memory_limit_gb=8.0,
            chunk_size=1000
        )
        
        # Create engine
        engine = ParameterOptimizationEngine(config)
        logger.info("✅ ParameterOptimizationEngine created with hardware optimizations")
        
        # Check if hardware optimizers were initialized
        if engine.m1_memory_optimizer:
            logger.info("✅ M1 Memory Optimizer initialized")
        else:
            logger.info("⚠️ M1 Memory Optimizer not initialized")
        
        if engine.m1_cpu_optimizer:
            logger.info("✅ M1 CPU Optimizer initialized")
        else:
            logger.info("⚠️ M1 CPU Optimizer not initialized")
        
        if engine.m1_gpu_manager:
            logger.info("✅ M1 GPU Manager initialized")
        else:
            logger.info("⚠️ M1 GPU Manager not initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ParameterOptimizationEngine test failed: {e}")
        return False

def test_sr_backtesting_engine():
    """Test that SRBacktestingEngine can be imported with hardware optimizations."""
    logger.info("🔍 Testing SRBacktestingEngine with hardware optimizations...")
    
    try:
        from utils.sr_clustering.sr_backtesting_engine import (
            SRBacktestingEngine, 
            BacktestConfig
        )
        logger.info("✅ SRBacktestingEngine imported successfully")
        
        # Test configuration with hardware optimizations
        config = BacktestConfig(
            enable_m1_optimizations=True,
            enable_gpu_acceleration=True,
            enable_memory_optimization=True,
            memory_limit_gb=8.0,
            chunk_size=1000,
            enable_parallel_processing=True,
            enable_vectorized_operations=True,
            enable_caching=True,
            cache_size_mb=100,
            enable_numba_acceleration=True
        )
        
        # Create engine
        engine = SRBacktestingEngine(config)
        logger.info("✅ SRBacktestingEngine created with hardware optimizations")
        
        # Check if computation optimizations were initialized
        if hasattr(engine, '_cache') and engine._cache is not None:
            logger.info("✅ Caching initialized")
        else:
            logger.info("⚠️ Caching not initialized")
        
        if hasattr(engine, 'numba_available'):
            logger.info(f"✅ Numba availability: {engine.numba_available}")
        else:
            logger.info("⚠️ Numba availability not checked")
        
        if hasattr(engine, 'vectorized_ops'):
            logger.info(f"✅ Vectorized operations: {engine.vectorized_ops}")
        else:
            logger.info("⚠️ Vectorized operations not configured")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SRBacktestingEngine test failed: {e}")
        return False

def test_sub_pipeline_integration():
    """Test that sub-pipeline can be imported with hardware optimizations."""
    logger.info("🔍 Testing sub-pipeline integration...")
    
    try:
        from training.steps.market_analysis.sub_pipeline import MarketAnalysisSubPipeline
        logger.info("✅ MarketAnalysisSubPipeline imported successfully")
        
        # Check if the sub-pipeline has the parameter optimization method
        if hasattr(MarketAnalysisSubPipeline, '_sr_parameter_optimization_pipeline'):
            logger.info("✅ SR parameter optimization pipeline method exists")
        else:
            logger.warning("⚠️ SR parameter optimization pipeline method not found")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Sub-pipeline integration test failed: {e}")
        return False

def main():
    """Run all hardware optimization tests."""
    logger.info("🚀 Starting Hardware Optimization Integration Tests")
    logger.info("="*60)
    
    tests = [
        ("Hardware Imports", test_hardware_imports),
        ("Parameter Optimization Engine", test_parameter_optimization_engine),
        ("SR Backtesting Engine", test_sr_backtesting_engine),
        ("Sub-pipeline Integration", test_sub_pipeline_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        logger.info("-" * 40)
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.warning(f"⚠️ {test_name}: PARTIAL")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: FAILED - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All hardware optimization tests passed!")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed or had issues")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)