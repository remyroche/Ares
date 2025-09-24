#!/usr/bin/env python3
"""
Simple test script for enhanced TAS regime system integration.

This script tests the integration of utility tools with the TAS regime system
without requiring external dependencies.
"""

import sys
import os
import logging
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that we can import the enhanced modules."""
    try:
        # Test utility imports
        from src.utils.common_operations import CommonUtilities
        from src.utils.math_validation import MathValidation
        from src.utils.serialization_utils import UniversalSerializer
        logger.info("✅ Utility modules imported successfully")
        
        # Test enhanced TAS engine
        from src.training.steps.market_analysis.tas_regime.core.tas_engine import (
            TreeArchitectureSearchEngine, TASEngineConfig
        )
        logger.info("✅ Enhanced TAS engine imported successfully")
        
        # Test enhanced regime detector
        from src.training.steps.market_analysis.tas_regime.core.tas_regime_detector import (
            TASRegimeDetector, TASRegimeConfig
        )
        logger.info("✅ Enhanced regime detector imported successfully")
        
        # Test enhanced backtesting engine
        from src.training.steps.market_analysis.tas_regime.backtesting.backtesting_engine import (
            BacktestingEngine, BacktestingConfig
        )
        logger.info("✅ Enhanced backtesting engine imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_utility_initialization():
    """Test utility tool initialization."""
    try:
        from src.utils.common_operations import CommonUtilities
        from src.utils.math_validation import MathValidation
        from src.utils.serialization_utils import UniversalSerializer
        
        # Test common utilities
        common_utils = CommonUtilities()
        logger.info("✅ Common utilities initialized")
        
        # Test math validation
        math_validator = MathValidation()
        logger.info("✅ Math validation initialized")
        
        # Test serializer
        serializer = UniversalSerializer()
        logger.info("✅ Universal serializer initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Utility initialization failed: {e}")
        return False

def test_enhanced_engine_initialization():
    """Test enhanced engine initialization."""
    try:
        from src.training.steps.market_analysis.tas_regime.core.tas_engine import (
            TreeArchitectureSearchEngine, TASEngineConfig
        )
        
        # Create configuration
        config = TASEngineConfig(
            enable_hardware_optimization=True,
            enable_meta_learning=True,
            enable_uncertainty_estimation=True,
            enable_regime_analysis=True,
            enable_real_time_adaptation=True
        )
        
        # Initialize engine
        engine = TreeArchitectureSearchEngine(config)
        logger.info("✅ Enhanced TAS engine initialized")
        
        # Test utility status
        status = engine._get_utility_status()
        logger.info(f"   Utility status: {status}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced engine initialization failed: {e}")
        return False

def test_enhanced_detector_initialization():
    """Test enhanced detector initialization."""
    try:
        from src.training.steps.market_analysis.tas_regime.core.tas_regime_detector import (
            TASRegimeDetector, TASRegimeConfig
        )
        
        # Create configuration
        config = TASRegimeConfig(
            n_regimes=3,
            enable_economic_evaluation=True,
            enable_uncertainty_quantification=True,
            enable_multi_scale_analysis=True
        )
        
        # Initialize detector
        detector = TASRegimeDetector(config)
        logger.info("✅ Enhanced regime detector initialized")
        
        # Test utility status
        status = detector._get_enhanced_utility_status()
        logger.info(f"   Enhanced utility status: {status}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced detector initialization failed: {e}")
        return False

def test_enhanced_backtesting_initialization():
    """Test enhanced backtesting initialization."""
    try:
        from src.training.steps.market_analysis.tas_regime.backtesting.backtesting_engine import (
            BacktestingEngine, BacktestingConfig
        )
        
        # Create configuration
        config = BacktestingConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000.0,
            enable_regime_aware_backtesting=True
        )
        
        # Initialize engine
        engine = BacktestingEngine(config)
        logger.info("✅ Enhanced backtesting engine initialized")
        
        # Test utility status
        status = engine._get_enhanced_utility_status()
        logger.info(f"   Enhanced utility status: {status}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced backtesting initialization failed: {e}")
        return False

def test_math_validation_basic():
    """Test basic math validation without external dependencies."""
    try:
        from src.utils.math_validation import MathValidation
        
        math_validator = MathValidation()
        
        # Test safe divide
        result = math_validator.safe_divide(10, 2)
        assert result == 5.0, f"Expected 5.0, got {result}"
        logger.info("✅ Safe divide working")
        
        # Test safe divide with zero
        result = math_validator.safe_divide(10, 0)
        assert result == 0.0, f"Expected 0.0, got {result}"
        logger.info("✅ Safe divide with zero working")
        
        # Test validation
        result = math_validator.validate_finite(5.0)
        assert result == 5.0, f"Expected 5.0, got {result}"
        logger.info("✅ Finite validation working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Math validation test failed: {e}")
        return False

def test_serialization_basic():
    """Test basic serialization without external dependencies."""
    try:
        from src.utils.serialization_utils import UniversalSerializer
        
        serializer = UniversalSerializer()
        
        # Test JSON serialization
        test_data = {"test": "value", "number": 42}
        success = serializer.save(test_data, "/tmp/test_data.json", format="json")
        assert success, "JSON save failed"
        logger.info("✅ JSON serialization working")
        
        # Test loading
        loaded_data = serializer.load("/tmp/test_data.json")
        assert loaded_data == test_data, f"Data mismatch: {loaded_data} != {test_data}"
        logger.info("✅ JSON deserialization working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Serialization test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting enhanced TAS regime system integration tests")
    
    tests = [
        ("Import Test", test_imports),
        ("Utility Initialization", test_utility_initialization),
        ("Enhanced Engine Initialization", test_enhanced_engine_initialization),
        ("Enhanced Detector Initialization", test_enhanced_detector_initialization),
        ("Enhanced Backtesting Initialization", test_enhanced_backtesting_initialization),
        ("Math Validation Basic", test_math_validation_basic),
        ("Serialization Basic", test_serialization_basic)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running test: {test_name}")
        try:
            if test_func():
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            failed += 1
    
    logger.info(f"\n📊 Test Results:")
    logger.info(f"   ✅ Passed: {passed}")
    logger.info(f"   ❌ Failed: {failed}")
    logger.info(f"   📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        logger.info("🎉 All tests passed! Enhanced TAS regime system is working correctly.")
        return True
    else:
        logger.error(f"⚠️ {failed} tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)