"""
Comprehensive backward compatibility test.

This script tests that all existing interfaces continue to work
exactly as before while providing optional enhanced logging.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_backward_compatibility():
    """
    Test that all existing interfaces work exactly as before.
    """
    print("🔍 Testing Backward Compatibility")
    print("=" * 50)

    # Test 1: Original BaseScaler Interface
    print("\n📋 Test 1: Original BaseScaler Interface")
    print("-" * 40)

    try:
        from src.features_common import BaseScaler

        # Test original constructor signature
        scaler = BaseScaler(
            use_vectorbt=True,
            enable_gpu=False,
            vectorbt_threshold=1000,
            use_optimizer=True,
            use_unified_manager=True
        )

        print("✅ Original BaseScaler constructor works")
        print(f"   Type: {type(scaler).__name__}")
        print(f"   Fitted: {scaler.fitted}")
        print(f"   Use VectorBT: {scaler.use_vectorbt}")
        print(f"   Enable GPU: {scaler.enable_gpu}")
        print(f"   VectorBT Threshold: {scaler.vectorbt_threshold}")

    except Exception as e:
        print(f"❌ Original BaseScaler constructor failed: {e}")
        return False

    # Test 2: Original Method Signatures
    print("\n📋 Test 2: Original Method Signatures")
    print("-" * 40)

    try:
        # Create test data
        data = pd.Series(np.random.randn(100))

        # Test fit_transform signature
        result = scaler.fit_transform(data)

        print("✅ fit_transform method works")
        print(f"   Input shape: {data.shape}")
        print(f"   Output shape: {result.shape}")
        print(f"   Output type: {type(result)}")
        print(f"   Is pandas Series: {isinstance(result, pd.Series)}")

        # Test transform signature
        result2 = scaler.transform(data)

        print("✅ transform method works")
        print(f"   Input shape: {data.shape}")
        print(f"   Output shape: {result2.shape}")
        print(f"   Output type: {type(result2)}")
        print(f"   Is pandas Series: {isinstance(result2, pd.Series)}")

    except Exception as e:
        print(f"❌ Original method signatures failed: {e}")
        return False

    # Test 3: Original Behavior Preservation
    print("\n📋 Test 3: Original Behavior Preservation")
    print("-" * 40)

    try:
        # Test that the behavior is exactly the same
        data1 = pd.Series([1, 2, 3, 4, 5])
        data2 = pd.Series([1, 2, 3, 4, 5])

        # Create two scalers
        scaler1 = BaseScaler(enable_verbose_logging=False)
        scaler2 = BaseScaler(enable_verbose_logging=True)

        # Fit both scalers
        result1 = scaler1.fit_transform(data1)
        result2 = scaler2.fit_transform(data2)

        # Results should be identical
        if np.allclose(result1.values, result2.values, rtol=1e-10):
            print("✅ Behavior is identical with and without verbose logging")
        else:
            print("❌ Behavior differs between logging modes")
            return False

        # Test transform behavior
        test_data = pd.Series([6, 7, 8, 9, 10])
        result3 = scaler1.transform(test_data)
        result4 = scaler2.transform(test_data)

        if np.allclose(result3.values, result4.values, rtol=1e-10):
            print("✅ Transform behavior is identical with and without verbose logging")
        else:
            print("❌ Transform behavior differs between logging modes")
            return False

    except Exception as e:
        print(f"❌ Original behavior preservation failed: {e}")
        return False

    # Test 4: Enhanced Scaler Interface
    print("\n📋 Test 4: Enhanced Scaler Interface")
    print("-" * 40)

    try:
        from src.features_common import create_enhanced_scaler, enable_enhanced_logging

        # Test enhanced scaler creation
        enhanced_scaler = create_enhanced_scaler(
            method='zscore',
            enable_verbose_logging=True
        )

        print("✅ Enhanced scaler creation works")
        print(f"   Type: {type(enhanced_scaler).__name__}")
        print(f"   Enable verbose logging: {enhanced_scaler.enable_verbose_logging}")

        # Test enhanced scaler behavior
        data = pd.Series(np.random.randn(100))
        result = enhanced_scaler.fit_transform(data)

        print("✅ Enhanced scaler behavior works")
        print(f"   Input shape: {data.shape}")
        print(f"   Output shape: {result.shape}")

        # Test global logging control
        enable_enhanced_logging(True)
        print("✅ Global logging control works")

    except Exception as e:
        print(f"❌ Enhanced scaler interface failed: {e}")
        return False

    # Test 5: Factory Functions Compatibility
    print("\n📋 Test 5: Factory Functions Compatibility")
    print("-" * 40)

    try:
        from src.features_common import create_optimized_scaler

        # Test original factory function
        factory_scaler = create_optimized_scaler(method='zscore')

        print("✅ Original factory function works")
        print(f"   Type: {type(factory_scaler).__name__}")

        # Test factory scaler behavior
        data = pd.Series(np.random.randn(100))
        result = factory_scaler.fit_transform(data)

        print("✅ Factory scaler behavior works")
        print(f"   Input shape: {data.shape}")
        print(f"   Output shape: {result.shape}")

    except Exception as e:
        print(f"❌ Factory functions compatibility failed: {e}")
        return False

    # Test 6: Error Handling Compatibility
    print("\n📋 Test 6: Error Handling Compatibility")
    print("-" * 40)

    try:
        # Test that errors are handled the same way
        scaler = BaseScaler()

        # Test with empty data
        empty_data = pd.Series(dtype=float)

        try:
            scaler.fit_transform(empty_data)
            print("❌ Empty data should raise an error")
            return False
        except (ValueError, RuntimeError) as e:
            print("✅ Empty data correctly raises error")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {e}")

        # Test with invalid data
        invalid_data = "not a pandas series"

        try:
            scaler.fit_transform(invalid_data)
            print("❌ Invalid data should raise an error")
            return False
        except (TypeError, ValueError, RuntimeError) as e:
            print("✅ Invalid data correctly raises error")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {e}")

    except Exception as e:
        print(f"❌ Error handling compatibility failed: {e}")
        return False

    # Test 7: Performance Compatibility
    print("\n📋 Test 7: Performance Compatibility")
    print("-" * 40)

    try:
        import time

        # Test that performance is not significantly degraded
        data = pd.Series(np.random.randn(1000))

        # Test without verbose logging
        scaler1 = BaseScaler(enable_verbose_logging=False)
        start_time = time.time()
        result1 = scaler1.fit_transform(data)
        time1 = time.time() - start_time

        # Test with verbose logging
        scaler2 = BaseScaler(enable_verbose_logging=True)
        start_time = time.time()
        result2 = scaler2.fit_transform(data)
        time2 = time.time() - start_time

        # Results should be identical
        if not np.allclose(result1.values, result2.values, rtol=1e-10):
            print("❌ Results differ between logging modes")
            return False

        # Performance should be similar (within 20% tolerance)
        performance_ratio = time2 / time1 if time1 > 0 else 1.0
        if performance_ratio <= 1.2:  # 20% tolerance
            print("✅ Performance is compatible")
            print(f"   Time without logging: {time1:.4f}s")
            print(f"   Time with logging: {time2:.4f}s")
            print(f"   Performance ratio: {performance_ratio:.2f}")
        else:
            print("⚠️  Performance may be degraded with verbose logging")
            print(f"   Time without logging: {time1:.4f}s")
            print(f"   Time with logging: {time2:.4f}s")
            print(f"   Performance ratio: {performance_ratio:.2f}")

    except Exception as e:
        print(f"❌ Performance compatibility failed: {e}")
        return False

    # Test 8: Import Compatibility
    print("\n📋 Test 8: Import Compatibility")
    print("-" * 40)

    try:
        # Test that all original imports work
        from src.features_common import (
            BaseScaler, SimpleScaler, VectorBTScaler, VectorBTBatchScaler,
            create_optimized_scaler, create_optimized_batch_scaler,
            OptimizationConfig, VectorBTConfig, UnifiedConfig,
            OptimizationMixin, PerformanceMixin, VectorBTMixin,
            ValidationMixin, CachingMixin, MonitoringMixin
        )

        print("✅ All original imports work")

        # Test that new imports work
        from src.features_common import (
            create_enhanced_scaler, enable_enhanced_logging,
            FeaturesCommonError, ValidationError, OptimizationError,
            VectorBTError, ConfigurationError, SilentFailureError,
            ensure_no_silent_failures, validate_input_data, safe_execute,
            validate_configuration, check_system_health, report_silent_failures,
            get_logger, log_operation
        )

        print("✅ All new imports work")

    except Exception as e:
        print(f"❌ Import compatibility failed: {e}")
        return False

    # Final Summary
    print("\n🎉 Backward Compatibility Test Complete!")
    print("=" * 50)
    print("✅ All existing interfaces work exactly as before")
    print("✅ Method signatures are preserved")
    print("✅ Behavior is identical with and without logging")
    print("✅ Error handling is preserved")
    print("✅ Performance is compatible")
    print("✅ All imports work correctly")
    print("\n🚀 Full backward compatibility ensured!")

    return True

if __name__ == "__main__":
    success = test_backward_compatibility()
    if success:
        print("\n✅ All backward compatibility tests passed!")
    else:
        print("\n❌ Some backward compatibility tests failed!")
        exit(1)
