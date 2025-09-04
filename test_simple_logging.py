#!/usr/bin/env python3
"""
Simple test for enhanced logging system without external dependencies
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_logging():
    """Test basic logging functionality."""
    print("🧪 Testing Basic Logging System...")
    
    try:
        # Test basic imports
        from src.training.steps.market_analysis.enhanced_logging_metrics import EnhancedPipelineLogger
        
        # Create logger
        logger = EnhancedPipelineLogger("test_logger")
        
        # Test basic logging
        logger.logger.info("✅ Basic logging test successful")
        logger.logger.info("🚀 Enhanced logging system is working")
        
        # Test pipeline start/end
        logger.start_pipeline("ETHUSDT", "BINANCE", "test_123")
        time.sleep(0.1)
        logger.end_pipeline(success=True)
        
        print("✅ Basic logging test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Basic logging test failed: {e}")
        return False


def test_progress_monitor():
    """Test progress monitoring without external dependencies."""
    print("🧪 Testing Progress Monitor...")
    
    try:
        from src.training.steps.market_analysis.progress_monitor import ProgressMonitor
        
        # Create monitor
        monitor = ProgressMonitor(update_interval=0.1)
        
        # Test basic functionality
        monitor.update_step_progress("test_step", 0.5, "Testing progress", "running")
        monitor.complete_step("test_step", True, "Test completed")
        
        print("✅ Progress monitor test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Progress monitor test failed: {e}")
        return False


def test_imports():
    """Test that all enhanced logging components can be imported."""
    print("🧪 Testing Imports...")
    
    try:
        # Test core imports
        from src.training.steps.market_analysis.enhanced_logging_metrics import (
            EnhancedPipelineLogger, 
            FeatureQualityMetrics, 
            RegimeQualityMetrics,
            StepMetrics
        )
        
        from src.training.steps.market_analysis.progress_monitor import (
            ProgressMonitor, 
            ProgressContext, 
            monitor_progress
        )
        
        from src.training.steps.market_analysis.enhanced_market_analysis_orchestrator import (
            MarketAnalysisPipelineOrchestrator
        )
        
        print("✅ All imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def main():
    """Run all simple tests."""
    print("🚀 Simple Enhanced Logging Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_basic_logging,
        test_progress_monitor
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced logging system is ready.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)