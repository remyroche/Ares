#!/usr/bin/env python3
"""
Test script to verify comprehensive logging integration with existing logging calls.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import CONFIG
from src.utils.comprehensive_logger import setup_comprehensive_logging
from src.utils.logger import system_logger, initialize_comprehensive_integration


def test_comprehensive_integration():
    """Test that comprehensive logging integrates with existing logging calls."""
    print("🧪 Testing comprehensive logging integration...")
    
    # Setup comprehensive logging
    comprehensive_logger = setup_comprehensive_logging(CONFIG)
    
    # Initialize comprehensive integration
    initialize_comprehensive_integration()
    
    # Test existing logging patterns
    print("📝 Testing existing logging patterns...")
    
    # Test system_logger.getChild() pattern (used throughout codebase)
    print(f"🔍 system_logger type: {type(system_logger)}")
    print(f"🔍 system_logger has getChild: {hasattr(system_logger, 'getChild')}")
    
    test_logger1 = system_logger.getChild("TestComponent1")
    print(f"🔍 test_logger1 type: {type(test_logger1)}")
    print(f"🔍 test_logger1 name: {test_logger1.name}")
    test_logger1.info("Test message from system_logger.getChild()")
    
    # Test get_logger() pattern
    from src.utils.logger import get_logger
    test_logger2 = get_logger("TestComponent2")
    test_logger2.info("Test message from get_logger()")
    
    # Test direct comprehensive logging
    component_logger = comprehensive_logger.get_component_logger("TestComponent3")
    component_logger.info("Test message from comprehensive logger")
    
    # Test specialized logging methods
    comprehensive_logger.log_system_info("System info test")
    comprehensive_logger.log_error("Error test")
    comprehensive_logger.log_trade("Trade test")
    comprehensive_logger.log_performance("Performance test")
    
    print("✅ Comprehensive integration test completed!")
    
    # Check if all logs appear in global log file
    log_dir = Path("log")
    if log_dir.exists():
        global_log_files = list(log_dir.glob("ares_global_*.log"))
        if global_log_files:
            latest_global_log = max(global_log_files, key=lambda x: x.stat().st_mtime)
            print(f"✅ Latest global log file: {latest_global_log.name}")
            
            # Check if all test messages are in the global log
            with open(latest_global_log, 'r') as f:
                content = f.read()
                test_messages = [
                    "Test message from system_logger.getChild()",
                    "Test message from get_logger()",
                    "Test message from comprehensive logger",
                    "System info test",
                    "Error test",
                    "Trade test",
                    "Performance test"
                ]
                
                missing_messages = []
                for msg in test_messages:
                    if msg not in content:
                        missing_messages.append(msg)
                
                if missing_messages:
                    print(f"❌ Missing messages in global log: {missing_messages}")
                else:
                    print("✅ All test messages found in global log file")
        else:
            print("❌ No global log files found")
    else:
        print("❌ Log directory not found")


if __name__ == "__main__":
    test_comprehensive_integration() 