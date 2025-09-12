#!/usr/bin/env python3
"""
Simple Market Analysis Integration Test

This script performs basic validation of the three main market analysis components:
1. SR Parameter Optimization - Optimize SR detection levels
2. SR Detection - Detect Support/Resistance levels  
3. SR Clustering - Generate SR clusters

The test verifies that all components can be imported and have the expected structure.
"""

import sys
import os
import logging
from pathlib import Path

# Add workspace to path
sys.path.append('/workspace')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported."""
    logger.info("🧪 Testing Module Imports")
    
    import_results = {}
    
    # Test SR Detection
    try:
        from src.training.steps.market_analysis.sr_detection import SRDetectionStep
        logger.info("✅ SRDetectionStep imported successfully")
        import_results['sr_detection'] = True
    except ImportError as e:
        logger.error(f"❌ Failed to import SRDetectionStep: {e}")
        import_results['sr_detection'] = False
    
    # Test SR Clustering
    try:
        from src.training.steps.market_analysis.sr_clustering import SRClusteringStep
        logger.info("✅ SRClusteringStep imported successfully")
        import_results['sr_clustering'] = True
    except ImportError as e:
        logger.error(f"❌ Failed to import SRClusteringStep: {e}")
        import_results['sr_clustering'] = False
    
    # Test Parameter Optimization Engine
    try:
        from src.utils.sr_clustering.parameter_optimization_engine import (
            ParameterOptimizationEngine, 
            ParameterOptimizationConfig
        )
        logger.info("✅ ParameterOptimizationEngine imported successfully")
        import_results['parameter_optimization'] = True
    except ImportError as e:
        logger.error(f"❌ Failed to import ParameterOptimizationEngine: {e}")
        import_results['parameter_optimization'] = False
    
    # Test SR Backtesting Engine
    try:
        from src.utils.sr_clustering.sr_backtesting_engine import (
            SRBacktestingEngine, 
            BacktestConfig
        )
        logger.info("✅ SRBacktestingEngine imported successfully")
        import_results['sr_backtesting'] = True
    except ImportError as e:
        logger.error(f"❌ Failed to import SRBacktestingEngine: {e}")
        import_results['sr_backtesting'] = False
    
    # Test Market Analysis Sub Pipeline
    try:
        from src.training.steps.market_analysis.sub_pipeline import (
            MarketAnalysisSubPipeline,
            SubPipelineConfig,
            ExecutionMode
        )
        logger.info("✅ MarketAnalysisSubPipeline imported successfully")
        import_results['market_analysis_pipeline'] = True
    except ImportError as e:
        logger.error(f"❌ Failed to import MarketAnalysisSubPipeline: {e}")
        import_results['market_analysis_pipeline'] = False
    
    return import_results

def test_class_structures():
    """Test that classes have expected methods and attributes."""
    logger.info("🧪 Testing Class Structures")
    
    structure_results = {}
    
    # Test SRDetectionStep structure
    try:
        from src.training.steps.market_analysis.sr_detection import SRDetectionStep
        
        # Check for expected methods
        expected_methods = ['execute', 'validate_config', 'get_status']
        missing_methods = []
        
        for method in expected_methods:
            if not hasattr(SRDetectionStep, method):
                missing_methods.append(method)
        
        if missing_methods:
            logger.warning(f"⚠️ SRDetectionStep missing methods: {missing_methods}")
            structure_results['sr_detection'] = False
        else:
            logger.info("✅ SRDetectionStep has all expected methods")
            structure_results['sr_detection'] = True
            
    except Exception as e:
        logger.error(f"❌ Error testing SRDetectionStep structure: {e}")
        structure_results['sr_detection'] = False
    
    # Test SRClusteringStep structure
    try:
        from src.training.steps.market_analysis.sr_clustering import SRClusteringStep
        
        # Check for expected methods
        expected_methods = ['execute', 'validate_config', 'get_status']
        missing_methods = []
        
        for method in expected_methods:
            if not hasattr(SRClusteringStep, method):
                missing_methods.append(method)
        
        if missing_methods:
            logger.warning(f"⚠️ SRClusteringStep missing methods: {missing_methods}")
            structure_results['sr_clustering'] = False
        else:
            logger.info("✅ SRClusteringStep has all expected methods")
            structure_results['sr_clustering'] = True
            
    except Exception as e:
        logger.error(f"❌ Error testing SRClusteringStep structure: {e}")
        structure_results['sr_clustering'] = False
    
    # Test ParameterOptimizationEngine structure
    try:
        from src.utils.sr_clustering.parameter_optimization_engine import ParameterOptimizationEngine
        
        # Check for expected methods
        expected_methods = ['optimize_parameters']
        missing_methods = []
        
        for method in expected_methods:
            if not hasattr(ParameterOptimizationEngine, method):
                missing_methods.append(method)
        
        if missing_methods:
            logger.warning(f"⚠️ ParameterOptimizationEngine missing methods: {missing_methods}")
            structure_results['parameter_optimization'] = False
        else:
            logger.info("✅ ParameterOptimizationEngine has all expected methods")
            structure_results['parameter_optimization'] = True
            
    except Exception as e:
        logger.error(f"❌ Error testing ParameterOptimizationEngine structure: {e}")
        structure_results['parameter_optimization'] = False
    
    return structure_results

def test_configuration_creation():
    """Test that configurations can be created."""
    logger.info("🧪 Testing Configuration Creation")
    
    config_results = {}
    
    # Test ParameterOptimizationConfig
    try:
        from src.utils.sr_clustering.parameter_optimization_engine import ParameterOptimizationConfig
        
        config = ParameterOptimizationConfig()
        logger.info("✅ ParameterOptimizationConfig created successfully")
        logger.info(f"   Optimization method: {config.optimization_method}")
        logger.info(f"   Objective metric: {config.objective_metric}")
        config_results['parameter_optimization_config'] = True
        
    except Exception as e:
        logger.error(f"❌ Error creating ParameterOptimizationConfig: {e}")
        config_results['parameter_optimization_config'] = False
    
    # Test BacktestConfig
    try:
        from src.utils.sr_clustering.sr_backtesting_engine import BacktestConfig
        
        config = BacktestConfig()
        logger.info("✅ BacktestConfig created successfully")
        config_results['backtest_config'] = True
        
    except Exception as e:
        logger.error(f"❌ Error creating BacktestConfig: {e}")
        config_results['backtest_config'] = False
    
    # Test SubPipelineConfig
    try:
        from src.training.steps.market_analysis.sub_pipeline import SubPipelineConfig, ExecutionMode
        
        config = SubPipelineConfig(
            mode=ExecutionMode.LIGHT,
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1h"
        )
        logger.info("✅ SubPipelineConfig created successfully")
        logger.info(f"   Mode: {config.mode}")
        logger.info(f"   Symbol: {config.symbol}")
        config_results['sub_pipeline_config'] = True
        
    except Exception as e:
        logger.error(f"❌ Error creating SubPipelineConfig: {e}")
        config_results['sub_pipeline_config'] = False
    
    return config_results

def test_file_existence():
    """Test that all required files exist."""
    logger.info("🧪 Testing File Existence")
    
    file_results = {}
    
    required_files = [
        'src/training/steps/market_analysis/sr_detection.py',
        'src/training/steps/market_analysis/sr_clustering.py',
        'src/training/steps/market_analysis/sub_pipeline.py',
        'src/utils/sr_clustering/parameter_optimization_engine.py',
        'src/utils/sr_clustering/sr_backtesting_engine.py',
        'src/utils/sr_clustering/backtesting_enhanced_clustering.py'
    ]
    
    for file_path in required_files:
        full_path = Path('/workspace') / file_path
        if full_path.exists():
            logger.info(f"✅ {file_path} exists")
            file_results[file_path] = True
        else:
            logger.error(f"❌ {file_path} does not exist")
            file_results[file_path] = False
    
    return file_results

def main():
    """Run all market analysis validation tests."""
    logger.info("🚀 Starting Market Analysis Validation Tests")
    logger.info("=" * 60)
    
    # Test results
    all_results = {}
    
    # Test file existence
    logger.info("\n1️⃣ Testing File Existence")
    logger.info("-" * 40)
    file_results = test_file_existence()
    all_results.update(file_results)
    
    # Test imports
    logger.info("\n2️⃣ Testing Module Imports")
    logger.info("-" * 40)
    import_results = test_imports()
    all_results.update(import_results)
    
    # Test class structures
    logger.info("\n3️⃣ Testing Class Structures")
    logger.info("-" * 40)
    structure_results = test_class_structures()
    all_results.update(structure_results)
    
    # Test configuration creation
    logger.info("\n4️⃣ Testing Configuration Creation")
    logger.info("-" * 40)
    config_results = test_configuration_creation()
    all_results.update(config_results)
    
    # Summary
    logger.info("\n📊 Test Results Summary")
    logger.info("=" * 60)
    
    total_tests = len(all_results)
    passed_tests = sum(all_results.values())
    
    for test_name, passed in all_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All validation tests passed! Market Analysis implementation is structurally complete.")
        return True
    else:
        logger.warning(f"⚠️ {total_tests - passed_tests} tests failed. Review implementation.")
        return False

if __name__ == "__main__":
    # Run the tests
    success = main()
    sys.exit(0 if success else 1)