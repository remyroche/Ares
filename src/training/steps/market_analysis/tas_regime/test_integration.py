#!/usr/bin/env python3
"""
Test TAS Regime Integration

This script tests the integration of the TAS regime system in the market analysis pipeline.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add the project root to the path
sys.path.append('/workspace/src')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tas_regime_imports():
    """Test that TAS regime components can be imported."""
    try:
        # Test core imports
        from src.training.steps.market_analysis.tas_regime.core.tas_config import TASConfig, TASArchitectureType
        from src.training.steps.market_analysis.tas_regime.core.tas_engine import TreeArchitectureSearchEngine
        from src.training.steps.market_analysis.tas_regime.core.tree_cvlSA_architecture import TreeCVLSASearch
        
        logger.info("✅ Core TAS components imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import core TAS components: {e}")
        return False

def test_tas_regime_configuration():
    """Test TAS regime configuration."""
    try:
        
        # Test configuration creation
        config = TASConfig.create_advanced_trading_config()
        logger.info(f"✅ TAS configuration created: {config.architecture_type}")
        
        # Test CVLSA configuration
        cvlsa_config = TASConfig.create_cvlSA_tree_config()
        logger.info(f"✅ CVLSA configuration created: {cvlsa_config.architecture_type}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create TAS configuration: {e}")
        return False

def test_tas_regime_engine():
    """Test TAS regime engine initialization."""
    try:
        
        # Create configuration
        config = TASConfig.create_advanced_trading_config()
        
        # Initialize engine
        engine = TreeArchitectureSearchEngine(config)
        logger.info("✅ TAS engine initialized successfully")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize TAS engine: {e}")
        return False

def test_cvlSA_architecture():
    """Test CVLSA architecture."""
    try:
        
        # Create configuration
        config = TASConfig.create_cvlSA_tree_config()
        
        # Initialize CVLSA
        cvlsa = TreeCVLSASearch(config)
        logger.info("✅ CVLSA architecture initialized successfully")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize CVLSA architecture: {e}")
        return False

def test_market_analysis_integration():
    """Test integration with market analysis pipeline."""
    try:
        # Test that we can import from the market analysis module
        from src.training.steps.market_analysis.tas_regime import TASConfig, TreeArchitectureSearchEngine
        
        logger.info("✅ TAS regime integrated with market analysis pipeline")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to integrate with market analysis pipeline: {e}")
        return False

def main():
    """Run all integration tests."""
    logger.info("🧪 Testing TAS Regime Integration")
    logger.info("=" * 50)
    
    tests = [
        ("Import Test", test_tas_regime_imports),
        ("Configuration Test", test_tas_regime_configuration),
        ("Engine Test", test_tas_regime_engine),
        ("CVLSA Test", test_cvlSA_architecture),
        ("Market Analysis Integration", test_market_analysis_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name}...")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Test Results Summary")
    logger.info("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! TAS regime integration is working.")
        return True
    else:
        logger.error("❌ Some tests failed. Check the logs above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)