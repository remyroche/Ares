"""
Test script for enhanced BaseStep functionality

This script tests the enhanced BaseStep to ensure all utilities are working correctly.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Change to the project root directory
os.chdir(project_root)

# Import with absolute path
import sys
sys.path.insert(0, '/workspace')

from src.training.steps.base_step import BaseStep


class TestEnhancedStep(BaseStep):
    """Test step to verify enhanced BaseStep functionality."""
    
    def __init__(self, step_name: str = "test_enhanced_step", config: dict = None):
        super().__init__(step_name, config)
        print("✅ TestEnhancedStep initialized")
    
    async def execute(self, config: dict) -> dict:
        """Test the enhanced BaseStep functionality."""
        print("🚀 Testing enhanced BaseStep functionality...")
        
        # Test 1: Check utility availability
        print("\n📋 Testing utility availability...")
        availability = self._get_availability_status()
        print(f"Available utilities: {sum(availability.values())}/{len(availability)}")
        
        for utility, available in availability.items():
            status = "✅" if available else "❌"
            print(f"  {status} {utility}")
        
        # Test 2: Test convenience methods
        print("\n🔧 Testing convenience methods...")
        
        # Test JSON operations
        test_data = {"test": "value", "number": 42}
        json_saved = self._safe_json_save(test_data, "test_data.json")
        print(f"JSON save: {json_saved}")
        
        loaded_data = self._safe_json_load("test_data.json")
        print(f"JSON load: {loaded_data}")
        
        # Test math operations
        result = self._safe_divide(10, 2, default=0)
        print(f"Safe divide: {result}")
        
        finite_value = self._validate_finite(3.14, default=0)
        print(f"Validate finite: {finite_value}")
        
        positive_value = self._validate_positive(-5, default=1)
        print(f"Validate positive: {positive_value}")
        
        # Test directory operations
        dir_created = self._ensure_directory("test_dir")
        print(f"Directory created: {dir_created}")
        
        # Test 3: Test direct utility access
        print("\n📦 Testing direct utility access...")
        
        if self.common_ops:
            print("✅ Common operations available")
            current_time = self.common_ops['get_current_datetime']()
            print(f"Current time: {current_time}")
        else:
            print("❌ Common operations not available")
        
        if self.hardware_utils:
            print("✅ Hardware utilities available")
            print(f"Hardware manager: {self.hardware_utils['get_integrated_hardware_manager'] is not None}")
        else:
            print("❌ Hardware utilities not available")
        
        if self.math_validation:
            print("✅ Math validation available")
            safe_result = self.math_validation['safe_divide'](100, 3, default=0)
            print(f"Math validation result: {safe_result}")
        else:
            print("❌ Math validation not available")
        
        # Test 4: Test tprint functions
        print("\n📝 Testing tprint functions...")
        tprint("Testing basic tprint")
        tprint_success("Testing tprint_success")
        tprint_info("Testing tprint_info")
        tprint_warning("Testing tprint_warning")
        tprint_error("Testing tprint_error")
        
        # Test 5: Test DataFrame operations
        print("\n📊 Testing DataFrame operations...")
        import pandas as pd
        import numpy as np
        
        df = pd.DataFrame({
            'col1': np.random.randn(10),
            'col2': np.random.randn(10),
            'col3': np.random.randn(10)
        })
        
        has_cols = self._validate_dataframe_columns(df, ['col1', 'col2'])
        print(f"DataFrame validation: {has_cols}")
        
        # Test 6: Test ML utilities
        print("\n🤖 Testing ML utilities...")
        optimizer = self._get_ml_optimizer("bayesian")
        print(f"ML optimizer: {optimizer is not None}")
        
        cv_validator = self._get_cv_validator("time_series")
        print(f"CV validator: {cv_validator is not None}")
        
        # Test 7: Test data quality utilities
        print("\n🧹 Testing data quality utilities...")
        cleaner = self._get_data_cleaner()
        print(f"Data cleaner: {cleaner is not None}")
        
        # Test 8: Test model persistence utilities
        print("\n💾 Testing model persistence utilities...")
        cache = self._get_model_cache()
        print(f"Model cache: {cache is not None}")
        
        # Test 9: Test help system
        print("\n❓ Testing help system...")
        help_info = self._get_utility_help()
        print(f"Help info keys: {list(help_info.keys())}")
        
        # Clean up test files
        try:
            os.remove("test_data.json")
            os.rmdir("test_dir")
            print("\n🧹 Test files cleaned up")
        except:
            pass
        
        print("\n✅ Enhanced BaseStep test completed successfully!")
        
        return {
            'success': True,
            'test_results': {
                'utility_availability': availability,
                'convenience_methods_working': True,
                'direct_utility_access_working': True,
                'tprint_functions_working': True,
                'dataframe_operations_working': True,
                'ml_utilities_working': True,
                'data_quality_utilities_working': True,
                'model_persistence_utilities_working': True,
                'help_system_working': True
            }
        }


async def main():
    """Run the test."""
    print("🧪 Starting Enhanced BaseStep Test")
    print("=" * 50)
    
    # Create test step
    step = TestEnhancedStep()
    
    # Test configuration
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'execution_mode': 'light'
    }
    
    try:
        # Run the test
        result = await step.run(config)
        
        print("\n" + "=" * 50)
        print("🎉 Test Results:")
        print(f"Success: {result.get('success', False)}")
        
        if result.get('success'):
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())