#!/usr/bin/env python3
"""
Test script for the new timestamped print system.

This script tests the timestamped print functionality to ensure it works
correctly without causing system failures, especially with numba compatibility.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_timestamped_prints():
    """Test basic timestamped print functionality."""
    print("\n" + "="*60)
    print("🧪 TESTING BASIC TIMESTAMPED PRINTS")
    print("="*60)
    
    try:
        from src.utils.logger import timestamped_print, enable_timestamped_prints, disable_timestamped_prints
        
        print("✅ Successfully imported timestamped print functions")
        
        # Test the timestamped_print function directly
        print("\n--- Testing timestamped_print function directly ---")
        timestamped_print("This is a test message with timestamp")
        timestamped_print("Another test message", "with multiple", "arguments")
        
        # Test enabling timestamped prints globally
        print("\n--- Testing global timestamped prints ---")
        print("Before enabling timestamped prints:")
        print("Regular print message")
        
        enable_success = enable_timestamped_prints(include_relative=True, include_session=True)
        if enable_success:
            print("✅ Successfully enabled timestamped prints")
            print("After enabling timestamped prints:")
            print("This should now have a timestamp")
            print("Multiple arguments", "should work", "too")
            
            # Test with different message types
            print("Info message")
            print("Warning message")
            print("Error message")
            
            # Test disabling
            disable_success = disable_timestamped_prints()
            if disable_success:
                print("✅ Successfully disabled timestamped prints")
                print("After disabling timestamped prints:")
                print("This should be back to regular print")
            else:
                print("❌ Failed to disable timestamped prints")
        else:
            print("❌ Failed to enable timestamped prints")
            
    except Exception as e:
        print(f"❌ Error testing basic timestamped prints: {e}")
        import traceback
        traceback.print_exc()

def test_enhanced_logger():
    """Test the enhanced logger system."""
    print("\n" + "="*60)
    print("🧪 TESTING ENHANCED LOGGER SYSTEM")
    print("="*60)
    
    try:
        from src.utils.enhanced_simple_logger import enhanced_system_logger
        
        print("✅ Successfully imported enhanced logger")
        
        # Test the enhanced logger
        logger = enhanced_system_logger
        logger.info("This is an info message from enhanced logger")
        logger.warning("This is a warning message from enhanced logger")
        logger.error("This is an error message from enhanced logger")
        logger.debug("This is a debug message from enhanced logger")
        
        # Test child logger
        child_logger = logger.getChild("TestComponent")
        child_logger.info("This is a message from child logger")
        child_logger.warning("This is a warning from child logger")
        
        print("✅ Enhanced logger test completed")
        
    except Exception as e:
        print(f"❌ Error testing enhanced logger: {e}")
        import traceback
        traceback.print_exc()

def test_numba_compatibility():
    """Test numba compatibility with timestamped prints."""
    print("\n" + "="*60)
    print("🧪 TESTING NUMBA COMPATIBILITY")
    print("="*60)
    
    try:
        # Test if numba is available
        try:
            import numba
            print(f"✅ Numba is available (version: {numba.__version__})")
            
            # Test numba compilation with timestamped prints
            from src.utils.logger import enable_timestamped_prints_after_numba
            
            print("Testing numba compilation with timestamped prints...")
            
            @numba.jit(nopython=True)
            def test_numba_function(x):
                return x * 2
            
            # This should work without issues
            result = test_numba_function(5)
            print(f"✅ Numba function executed successfully: {result}")
            
            # Now test enabling timestamped prints after numba
            enable_timestamped_prints_after_numba()
            print("✅ Timestamped prints enabled after numba loading")
            print("This message should have a timestamp")
            
        except ImportError:
            print("ℹ️ Numba is not available, skipping numba compatibility test")
            
    except Exception as e:
        print(f"❌ Error testing numba compatibility: {e}")
        import traceback
        traceback.print_exc()

def test_launcher_integration():
    """Test the launcher integration with timestamped prints."""
    print("\n" + "="*60)
    print("🧪 TESTING LAUNCHER INTEGRATION")
    print("="*60)
    
    try:
        # Test importing the launcher with enhanced logger
        print("Testing launcher import with enhanced logger...")
        
        # This should work without errors
        from src.launcher.ares_launcher import AresLauncher
        
        print("✅ Successfully imported AresLauncher")
        
        # Test creating launcher instance
        print("Creating launcher instance...")
        launcher = AresLauncher()
        print("✅ Successfully created launcher instance")
        
        # Test launcher logging
        launcher.logger.info("This is a test message from launcher logger")
        launcher.logger.warning("This is a warning from launcher logger")
        
        print("✅ Launcher integration test completed")
        
    except Exception as e:
        print(f"❌ Error testing launcher integration: {e}")
        import traceback
        traceback.print_exc()

def test_log_file_creation():
    """Test that log files are created properly."""
    print("\n" + "="*60)
    print("🧪 TESTING LOG FILE CREATION")
    print("="*60)
    
    try:
        from src.utils.enhanced_simple_logger import enhanced_system_logger
        
        # Create some log messages
        logger = enhanced_system_logger
        logger.info("Test log message 1")
        logger.warning("Test warning message")
        logger.error("Test error message")
        
        # Check if logs directory exists
        logs_dir = Path("logs")
        if logs_dir.exists():
            print(f"✅ Logs directory exists: {logs_dir}")
            
            # List log files
            log_files = list(logs_dir.glob("*.log"))
            if log_files:
                print(f"✅ Found {len(log_files)} log files:")
                for log_file in log_files:
                    print(f"  - {log_file.name} ({log_file.stat().st_size} bytes)")
                    
                    # Check if file has content
                    if log_file.stat().st_size > 0:
                        print(f"    ✅ File has content")
                    else:
                        print(f"    ❌ File is empty")
            else:
                print("❌ No log files found")
        else:
            print("❌ Logs directory does not exist")
            
    except Exception as e:
        print(f"❌ Error testing log file creation: {e}")
        import traceback
        traceback.print_exc()

def test_artifact_naming():
    """Test the artifact naming system."""
    print("\n" + "="*60)
    print("🧪 TESTING ARTIFACT NAMING SYSTEM")
    print("="*60)
    
    try:
        from src.utils.artifact_naming import get_artifact_naming_manager, create_outcome_filename
        
        # Test artifact naming manager
        manager = get_artifact_naming_manager({"bot_version": "aresv1"})
        
        # Test creating artifact names
        outcome_name = manager.create_artifact_name("data_collection", "data_download", "outcome", "json")
        print(f"✅ Outcome filename: {outcome_name}")
        
        model_name = manager.create_model_artifact_name("analyst", "model_training", "general_model_training")
        print(f"✅ Model filename: {model_name}")
        
        data_name = manager.create_data_artifact_name("features", "market_analysis", "feature_engineering")
        print(f"✅ Data filename: {data_name}")
        
        # Test convenience functions
        outcome_filename = create_outcome_filename("data_collection", "data_download", "aresv1")
        print(f"✅ Convenience function result: {outcome_filename}")
        
        print("✅ Artifact naming system test completed")
        
    except Exception as e:
        print(f"❌ Error testing artifact naming: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests."""
    print("🚀 STARTING TIMESTAMPED PRINT SYSTEM TESTS")
    print("="*80)
    print(f"Test started at: {datetime.now().isoformat()}")
    print("="*80)
    
    # Run all tests
    test_basic_timestamped_prints()
    test_enhanced_logger()
    test_numba_compatibility()
    test_launcher_integration()
    test_log_file_creation()
    test_artifact_naming()
    
    print("\n" + "="*80)
    print("🏁 ALL TESTS COMPLETED")
    print(f"Test finished at: {datetime.now().isoformat()}")
    print("="*80)

if __name__ == "__main__":
    main()