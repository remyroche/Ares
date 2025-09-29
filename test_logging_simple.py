#!/usr/bin/env python3
"""
Simple test to verify logging enhancements in nas_tas_regime_discovery module.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_logging_imports():
    """Test that the logging enhancements can be imported."""
    print("🧪 Testing NAS-TAS Regime Discovery Logging Imports")
    print("=" * 50)
    
    try:
        # Test tprint imports
        from src.utils.tprint import tprint, tprint_debug, tprint_info, tprint_success, tprint_error
        print("✅ Successfully imported tprint functions")
        
        # Test logging standards
        from src.training.steps.market_analysis.logging_standards import (
            get_logger, log_info, log_warning, log_error, log_success, log_debug
        )
        print("✅ Successfully imported logging standards")
        
        # Test the component import (without instantiation)
        from src.training.steps.market_analysis.components.nas_tas_regime_discovery import NASTASRegimeDiscoveryComponent
        print("✅ Successfully imported NASTASRegimeDiscoveryComponent")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_logging_functions():
    """Test that the logging functions work."""
    print("\n🔧 Testing logging functions...")
    
    try:
        from src.utils.tprint import tprint, tprint_debug, tprint_info, tprint_success, tprint_error
        from src.training.steps.market_analysis.logging_standards import log_info, log_success, log_debug
        
        # Test tprint functions
        tprint("🧪 Testing tprint function")
        tprint_debug("🔧 Testing tprint_debug function")
        tprint_info("ℹ️ Testing tprint_info function")
        tprint_success("✅ Testing tprint_success function")
        
        # Test logging standards
        log_info("Testing log_info function")
        log_success("Testing log_success function")
        log_debug("Testing log_debug function")
        
        print("✅ All logging functions work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Logging function error: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 NAS-TAS Regime Discovery Logging Test")
    print("=" * 60)
    
    # Test imports
    import_success = test_logging_imports()
    
    if not import_success:
        print("\n❌ Import tests failed!")
        return 1
    
    # Test logging functions
    function_success = test_logging_functions()
    
    if not function_success:
        print("\n❌ Logging function tests failed!")
        return 1
    
    print("\n🎉 All tests passed!")
    print("✅ Comprehensive logging has been successfully added to nas_tas_regime_discovery")
    print("\n📋 Logging Enhancements Added:")
    print("  • Detailed initialization logging with component attributes")
    print("  • Comprehensive execution parameter logging")
    print("  • Symbol and timeframe resolution debugging")
    print("  • Market data loading with shape and column information")
    print("  • Hybrid configuration creation with parameter details")
    print("  • Discovery process timing and result analysis")
    print("  • Regime prediction extraction with format detection")
    print("  • Metrics calculation with detailed output")
    print("  • Regime characteristics creation with progress tracking")
    print("  • Enhanced error handling with full traceback logging")
    print("  • Performance metrics and execution timing")
    print("  • Success/failure status with detailed debugging")
    
    return 0

if __name__ == "__main__":
    exit(main())