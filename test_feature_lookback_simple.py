#!/usr/bin/env python3
"""
Simplified test script for Feature Lookback Optimization in MARKET_ANALYSIS pipeline.

This script tests the core functionality without external dependencies.
"""

import sys
import os
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported."""
    logger.info("🧪 Testing imports...")
    
    try:
        # Test core imports
        from src.training.steps.market_analysis.sub_pipeline import (
            MarketAnalysisSubPipeline,
            SubPipelineConfig,
            ExecutionMode,
            ML_COMMONS_AVAILABLE
        )
        logger.info("✅ Core sub-pipeline imports successful")
        
        # Test feature optimization module
        from src.feature_engineering.feature_generation_optimization import (
            FeatureGenerationOptimizer,
            FeatureOptimizationConfig,
            OptimizationMethod
        )
        logger.info("✅ Feature optimization module imports successful")
        
        # Test ML commons availability
        if ML_COMMONS_AVAILABLE:
            logger.info("✅ ML commons are available")
        else:
            logger.info("ℹ️ ML commons not available - will use fallback methods")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_pipeline_initialization():
    """Test pipeline initialization."""
    logger.info("🧪 Testing pipeline initialization...")
    
    try:
        from src.training.steps.market_analysis.sub_pipeline import (
            MarketAnalysisSubPipeline,
            SubPipelineConfig,
            ExecutionMode
        )
        
        # Create pipeline
        pipeline = MarketAnalysisSubPipeline()
        logger.info("✅ Pipeline initialization successful")
        
        # Create config
        config = SubPipelineConfig(
            mode=ExecutionMode.BLANK,
            data_dir="test_data",
            exchange="binance",
            symbol="BTCUSDT",
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        logger.info("✅ Configuration creation successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline initialization failed: {e}")
        return False

def test_optimization_config():
    """Test optimization configuration."""
    logger.info("🧪 Testing optimization configuration...")
    
    try:
        from src.feature_engineering.feature_generation_optimization import (
            FeatureOptimizationConfig,
            OptimizationMethod
        )
        
        # Test different configurations
        configs = [
            FeatureOptimizationConfig(),
            FeatureOptimizationConfig(
                min_lookback=5,
                max_lookback=50,
                optimization_method=OptimizationMethod.STATISTICAL_ANALYSIS
            ),
            FeatureOptimizationConfig(
                optimization_method=OptimizationMethod.CROSS_VALIDATION,
                cv_folds=3
            )
        ]
        
        for i, config in enumerate(configs):
            logger.info(f"✅ Configuration {i+1} created successfully")
            logger.info(f"  Method: {config.optimization_method.value}")
            logger.info(f"  Lookback range: {config.min_lookback}-{config.max_lookback}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimization configuration test failed: {e}")
        return False

def test_optimizer_initialization():
    """Test optimizer initialization."""
    logger.info("🧪 Testing optimizer initialization...")
    
    try:
        from src.feature_engineering.feature_generation_optimization import (
            FeatureGenerationOptimizer,
            FeatureOptimizationConfig
        )
        
        # Test optimizer creation
        config = FeatureOptimizationConfig()
        optimizer = FeatureGenerationOptimizer(config)
        logger.info("✅ Optimizer initialization successful")
        
        # Test convenience function
        from src.feature_engineering.feature_generation_optimization import get_feature_optimizer
        optimizer2 = get_feature_optimizer()
        logger.info("✅ Convenience function successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimizer initialization failed: {e}")
        return False

def test_pipeline_methods():
    """Test that pipeline methods exist and are callable."""
    logger.info("🧪 Testing pipeline methods...")
    
    try:
        from src.training.steps.market_analysis.sub_pipeline import (
            MarketAnalysisSubPipeline,
            SubPipelineConfig,
            ExecutionMode
        )
        
        pipeline = MarketAnalysisSubPipeline()
        config = SubPipelineConfig(
            mode=ExecutionMode.BLANK,
            data_dir="test_data",
            exchange="binance",
            symbol="BTCUSDT",
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        # Test that the method exists
        if hasattr(pipeline, '_feature_lookback_optimization_pipeline'):
            logger.info("✅ _feature_lookback_optimization_pipeline method exists")
        else:
            logger.error("❌ _feature_lookback_optimization_pipeline method missing")
            return False
        
        # Test that the statistical optimization method exists
        if hasattr(pipeline, '_optimize_lookback_statistical'):
            logger.info("✅ _optimize_lookback_statistical method exists")
        else:
            logger.error("❌ _optimize_lookback_statistical method missing")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline methods test failed: {e}")
        return False

def test_blank_mode_execution():
    """Test blank mode execution (no actual data processing)."""
    logger.info("🧪 Testing blank mode execution...")
    
    try:
        import asyncio
        from src.training.steps.market_analysis.sub_pipeline import (
            MarketAnalysisSubPipeline,
            SubPipelineConfig,
            ExecutionMode
        )
        
        async def run_blank_test():
            pipeline = MarketAnalysisSubPipeline()
            config = SubPipelineConfig(
                mode=ExecutionMode.BLANK,
                data_dir="test_data",
                exchange="binance",
                symbol="BTCUSDT",
                start_date="2023-01-01",
                end_date="2023-12-31"
            )
            
            # Run the feature lookback optimization in blank mode
            artifacts = await pipeline._feature_lookback_optimization_pipeline(config)
            
            # Verify artifacts structure
            required_keys = ['optimization_results', 'optimal_lookbacks', 'optimization_metrics']
            for key in required_keys:
                if key not in artifacts:
                    raise ValueError(f"Missing required artifact key: {key}")
            
            # Verify optimal lookbacks
            optimal_lookbacks = artifacts['optimal_lookbacks']
            expected_indicators = ['rsi', 'sma', 'ema']
            for indicator in expected_indicators:
                if indicator not in optimal_lookbacks:
                    raise ValueError(f"Missing optimal lookback for {indicator}")
            
            logger.info(f"✅ Blank mode execution successful")
            logger.info(f"  Optimal lookbacks: {optimal_lookbacks}")
            return True
        
        # Run the async test
        result = asyncio.run(run_blank_test())
        return result
        
    except Exception as e:
        logger.error(f"❌ Blank mode execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_simple_test_suite():
    """Run simplified test suite."""
    logger.info("🚀 Starting simplified feature lookback optimization test suite...")
    
    tests = [
        ("imports", test_imports),
        ("pipeline_initialization", test_pipeline_initialization),
        ("optimization_config", test_optimization_config),
        ("optimizer_initialization", test_optimizer_initialization),
        ("pipeline_methods", test_pipeline_methods),
        ("blank_mode_execution", test_blank_mode_execution)
    ]
    
    results = {}
    passed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} test ---")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} test FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n📊 Test Results Summary:")
    logger.info(f"  Total tests: {len(tests)}")
    logger.info(f"  Passed: {passed}")
    logger.info(f"  Failed: {len(tests) - passed}")
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    if passed == len(tests):
        logger.info("\n🎉 All tests passed! Feature lookback optimization is properly implemented.")
        return True
    else:
        logger.warning(f"\n⚠️ {len(tests) - passed} tests failed. Review implementation.")
        return False

if __name__ == "__main__":
    success = run_simple_test_suite()
    
    if success:
        print("\n🎉 Feature Lookback Optimization Implementation: COMPLETE")
        sys.exit(0)
    else:
        print("\n❌ Feature Lookback Optimization Implementation: NEEDS ATTENTION")
        sys.exit(1)