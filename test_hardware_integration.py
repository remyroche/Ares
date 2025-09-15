#!/usr/bin/env python3
"""
Test Hardware Integration and Computation Toolbox

This script tests the hardware integration and computation toolbox functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_hardware_integration_import():
    """Test that hardware integration can be imported."""
    try:
        from src.utils.matrix_operations.hardware_integration import (
            get_hardware_optimized_processor,
            HardwareConfig,
            optimize_matrix_operation
        )
        print("✅ Hardware integration imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import hardware integration: {e}")
        return False

def test_computation_toolbox_import():
    """Test that computation toolbox can be imported."""
    try:
        from src.utils.matrix_operations.computation_toolbox import (
            get_computation_toolbox,
            ComputationConfig,
            compute_trading_indicators_optimized,
            matrix_multiply_optimized,
            correlation_analysis_optimized,
            batch_process_optimized,
            optimize_dataframe_optimized,
            get_toolbox_performance_report
        )
        print("✅ Computation toolbox imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import computation toolbox: {e}")
        return False

def test_hardware_config():
    """Test hardware configuration."""
    try:
        from src.utils.matrix_operations.hardware_integration import HardwareConfig
        
        config = HardwareConfig(
            max_memory_gb=4.0,
            enable_gpu=True,
            auto_optimize_dtypes=True
        )
        
        print(f"✅ Hardware config created: {config.max_memory_gb}GB, GPU: {config.enable_gpu}")
        return True
    except Exception as e:
        print(f"❌ Hardware config test failed: {e}")
        return False

def test_computation_config():
    """Test computation configuration."""
    try:
        from src.utils.matrix_operations.computation_toolbox import ComputationConfig
        
        config = ComputationConfig(
            enable_gpu=True,
            max_memory_gb=4.0,
            auto_optimize_dtypes=True
        )
        
        print(f"✅ Computation config created: {config.max_memory_gb}GB, GPU: {config.enable_gpu}")
        return True
    except Exception as e:
        print(f"❌ Computation config test failed: {e}")
        return False

def test_hardware_processor_creation():
    """Test hardware processor creation."""
    try:
        from src.utils.matrix_operations.hardware_integration import (
            get_hardware_optimized_processor,
            HardwareConfig
        )
        
        config = HardwareConfig(max_memory_gb=2.0)
        processor = get_hardware_optimized_processor(config)
        
        print("✅ Hardware processor created successfully")
        return True
    except Exception as e:
        print(f"❌ Hardware processor creation failed: {e}")
        return False

def test_computation_toolbox_creation():
    """Test computation toolbox creation."""
    try:
        from src.utils.matrix_operations.computation_toolbox import (
            get_computation_toolbox,
            ComputationConfig
        )
        
        config = ComputationConfig(max_memory_gb=2.0)
        toolbox = get_computation_toolbox(config)
        
        print("✅ Computation toolbox created successfully")
        return True
    except Exception as e:
        print(f"❌ Computation toolbox creation failed: {e}")
        return False

def test_performance_reporting():
    """Test performance reporting functionality."""
    try:
        from src.utils.matrix_operations.computation_toolbox import get_computation_toolbox
        
        toolbox = get_computation_toolbox()
        report = toolbox.get_performance_report()
        
        print(f"✅ Performance report generated with {len(report.get('performance_history', []))} records")
        return True
    except Exception as e:
        print(f"❌ Performance reporting test failed: {e}")
        return False

def test_matrix_operations_integration():
    """Test integration with existing matrix operations."""
    try:
        from src.utils.matrix_operations import (
            get_hardware_performance_report,
            optimize_matrix_operation_with_hardware,
            get_processing_performance_stats
        )
        
        print("✅ Matrix operations hardware integration imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Matrix operations hardware integration import failed: {e}")
        return False

def test_computation_toolbox_functions():
    """Test computation toolbox convenience functions."""
    try:
        from src.utils.matrix_operations import (
            compute_trading_indicators_optimized,
            matrix_multiply_optimized,
            correlation_analysis_optimized,
            batch_process_optimized,
            optimize_dataframe_optimized,
            get_toolbox_performance_report,
            cleanup_toolbox_resources
        )
        
        print("✅ Computation toolbox convenience functions imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Computation toolbox convenience functions import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎯 HARDWARE INTEGRATION & COMPUTATION TOOLBOX TEST")
    print("=" * 60)
    
    tests = [
        ("Import Hardware Integration", test_hardware_integration_import),
        ("Import Computation Toolbox", test_computation_toolbox_import),
        ("Test Hardware Config", test_hardware_config),
        ("Test Computation Config", test_computation_config),
        ("Test Hardware Processor Creation", test_hardware_processor_creation),
        ("Test Computation Toolbox Creation", test_computation_toolbox_creation),
        ("Test Performance Reporting", test_performance_reporting),
        ("Test Matrix Operations Integration", test_matrix_operations_integration),
        ("Test Computation Toolbox Functions", test_computation_toolbox_functions),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All hardware integration tests passed!")
        print("\n✨ Hardware-optimized computation toolbox is ready!")
        print("\n📖 Usage Examples:")
        print("""
# Hardware-optimized trading indicators
from src.utils.matrix_operations import compute_trading_indicators_optimized
indicators = compute_trading_indicators_optimized(ohlcv_data)

# Hardware-optimized matrix operations
from src.utils.matrix_operations import matrix_multiply_optimized
result = matrix_multiply_optimized(matrix_a, matrix_b, use_gpu=True)

# Hardware-optimized correlation analysis
from src.utils.matrix_operations import correlation_analysis_optimized
corr_matrix, feature_importance = correlation_analysis_optimized(data)

# Performance monitoring
from src.utils.matrix_operations import get_toolbox_performance_report
report = get_toolbox_performance_report()
print(f"Total operations: {report['summary']['total_operations']}")
        """)
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())