#!/usr/bin/env python3
"""
Test script for the Code Complexity Analysis Pipeline
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from complexity_pipeline import ComplexityPipeline
from config.complexity_config import ComplexityConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_test_file():
    """Create a simple test Python file"""
    test_code = '''
def simple_function():
    """A simple function with low complexity"""
    return "Hello, World!"

def complex_function(x, y, z):
    """A more complex function for testing"""
    if x > 0:
        if y > 0:
            if z > 0:
                result = x + y + z
            else:
                result = x + y
        else:
            if z > 0:
                result = x + z
            else:
                result = x
    else:
        if y > 0:
            if z > 0:
                result = y + z
            else:
                result = y
        else:
            if z > 0:
                result = z
            else:
                result = 0
    
    for i in range(10):
        if i % 2 == 0:
            result += i
        else:
            result -= i
    
    return result

class TestClass:
    """A test class for complexity analysis"""
    
    def __init__(self, value):
        self.value = value
    
    def method1(self):
        return self.value * 2
    
    def method2(self, multiplier):
        if multiplier > 0:
            return self.value * multiplier
        else:
            return 0
'''
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        return f.name


def test_pipeline():
    """Test the complexity analysis pipeline"""
    logger.info("Starting pipeline test...")
    
    try:
        # Create test file
        test_file = create_test_file()
        logger.info(f"Created test file: {test_file}")
        
        # Initialize pipeline
        pipeline = ComplexityPipeline()
        logger.info("Pipeline initialized successfully")
        
        # Test file analysis
        logger.info("Testing file analysis...")
        metrics = pipeline.analyze_file(test_file)
        
        logger.info(f"Analysis results:")
        logger.info(f"  File: {metrics.file_path}")
        logger.info(f"  PyExamine Score: {metrics.pyexamine_score}")
        logger.info(f"  Radon CC: {metrics.radon_cc}")
        logger.info(f"  Radon MI: {metrics.radon_mi}")
        logger.info(f"  Xenon Score: {metrics.xenon_score}")
        logger.info(f"  Combined Score: {metrics.combined_score}")
        
        # Test full analysis
        logger.info("Testing full analysis...")
        results = pipeline.run_full_analysis(test_file)
        
        logger.info("Full analysis completed successfully")
        logger.info(f"Results contain {len(results.get('file_analysis', {}))} file analyses")
        
        # Test tool availability
        logger.info("Checking tool availability...")
        from analyzers.pyexamine_analyzer import PyExamineAnalyzer
        from analyzers.radon_analyzer import RadonAnalyzer
        from analyzers.xenon_analyzer import XenonAnalyzer
        
        pyexamine = PyExamineAnalyzer(pipeline.config)
        radon = RadonAnalyzer(pipeline.config)
        xenon = XenonAnalyzer(pipeline.config)
        
        logger.info(f"PyExamine available: {pyexamine.is_available()}")
        logger.info(f"Radon available: {radon.is_available()}")
        logger.info(f"Xenon available: {xenon.is_available()}")
        
        # Clean up
        os.unlink(test_file)
        logger.info("Test file cleaned up")
        
        logger.info("Pipeline test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration loading"""
    logger.info("Testing configuration...")
    
    try:
        # Test default configuration
        config = ComplexityConfig()
        logger.info("Default configuration loaded successfully")
        
        # Test configuration values
        logger.info(f"PyExamine enabled: {config.enable_pyexamine}")
        logger.info(f"Radon enabled: {config.enable_radon}")
        logger.info(f"Xenon enabled: {config.enable_xenon}")
        logger.info(f"Complexity threshold: {config.complexity_threshold}")
        
        # Test configuration saving
        config_path = os.path.join(tempfile.gettempdir(), 'test_config.yaml')
        config.save_config(config_path)
        logger.info(f"Configuration saved to: {config_path}")
        
        # Test configuration loading
        config2 = ComplexityConfig(config_path)
        logger.info("Configuration loaded from file successfully")
        
        # Clean up
        os.unlink(config_path)
        logger.info("Test configuration cleaned up")
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    logger.info("=" * 50)
    logger.info("CODE COMPLEXITY PIPELINE TEST")
    logger.info("=" * 50)
    
    tests = [
        ("Configuration Test", test_configuration),
        ("Pipeline Test", test_pipeline),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name}...")
        logger.info("-" * 30)
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name:20} {status}")
    
    logger.info("-" * 50)
    logger.info(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("All tests passed! ✓")
        return 0
    else:
        logger.error("Some tests failed! ✗")
        return 1


if __name__ == '__main__':
    sys.exit(main())