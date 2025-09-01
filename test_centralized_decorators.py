#!/usr/bin/env python3
"""
Test script for centralized decorators
"""

import pandas as pd
import numpy as np
import asyncio

def test_centralized_decorators():
    pass
    pass
    """Test that all centralized decorators can be imported and used."""

    print("🧪 Testing Centralized Decorators")
    print("=" * 50)

    try:
        # Test imports
            validate_data_quality,
            quality_gate,
            step_specific_ml_validation,
            auto_fix_data_quality_issues,
            monitor_feature_engineering,
            monitor_data_collection,
            deterministic_seed,
            idempotent_step,
            handle_errors,
            with_tracing_span
    except Exception as e:
        pass
    except Exception as e:
        pass
        )
        print("✅ All decorators imported successfully")

        # Test validate_data_quality decorator
        @validate_data_quality(validation_level="WARNING", context="test")
        def test_function_with_data_quality(df):
    pass
    pass
            return df

        # Test quality_gate decorator
        @quality_gate(min_quality_score=0.7, required_grade="C")
        def test_function_with_quality_gate(df):
    pass
    pass
            return df

        # Test step_specific_ml_validation decorator
        @step_specific_ml_validation("step3")
        def test_function_with_step_validation(df):
    pass
    pass
            return df

        # Test auto_fix_data_quality_issues decorator
        @auto_fix_data_quality_issues(context="test")
        def test_function_with_auto_fix(df):
    pass
    pass
            return df

        # Test monitor decorators
        @monitor_feature_engineering()
        def test_function_with_monitor(df):
    pass
    pass
            return df

        # Test other decorators
        @deterministic_seed(42)
        @idempotent_step()
        @handle_errors()
        @with_tracing_span("test")
        def test_function_with_multiple_decorators(df):
    pass
    pass
            return df

        print("✅ All decorators applied successfully")

        # Create test data
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1H')
        test_df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(len(dates)),
            'high': np.random.randn(len(dates)),
            'low': np.random.randn(len(dates)),
            'close': np.random.randn(len(dates)),
            'volume': np.random.randint(100, 1000, len(dates))
        })
        test_df.set_index('timestamp', inplace=True)

        # Test function calls
        result1 = test_function_with_data_quality(test_df)
        result2 = test_function_with_quality_gate(test_df)
        result3 = test_function_with_step_validation(test_df)
        result4 = test_function_with_auto_fix(test_df)
        result5 = test_function_with_monitor(test_df)
        result6 = test_function_with_multiple_decorators(test_df)

        print("✅ All decorated functions executed successfully")
        print("✅ Centralized decorators test completed successfully!")

        return True

    except Exception as e:
        print(f"❌ Error testing centralized decorators: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_async_decorators():
    """Test async decorators."""

    print("\\\n🧪 Testing Async Decorators")
    print("=" * 50)

    try:
        from src.utils.centralized_decorators import (
    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import validate_data_quality,
            validate_data_quality,
            quality_gate,
            auto_fix_data_quality_issues
        )

        # Test async function with decorators
        @validate_data_quality(validation_level="WARNING", context="async_test")
        @quality_gate(min_quality_score=0.7, required_grade="C")
        @auto_fix_data_quality_issues(context="async_test")
        async def test_async_function(df):
            await asyncio.sleep(0.1)  # Simulate async work
            return df

        # Create test data
        dates = pd.date_range(start='2023-01-01', end='2023-01-05', freq='1H')
        test_df = pd.DataFrame({
            'timestamp': dates,
            'value': np.random.randn(len(dates))
        })
        test_df.set_index('timestamp', inplace=True)

        # Test async function
        result = await test_async_function(test_df)

        print("✅ Async decorators test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error testing async decorators: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step3_quality_gate():
    pass
    pass
    """Test that step3 uses the correct quality_gate."""

    print("\\\n🧪 Testing Step3 Quality Gate")
    print("=" * 50)

    try:
        # Import step3

    except Exception as e:
        pass
    except Exception as e:
        pass
        # Check that it imports quality_gate from centralized_decorators
        import src.training.steps.step3_hmm_regime_discovery as step3_module

        # Verify the import
        if hasattr(step3_module, 'quality_gate'):
    pass
    pass
            print("✅ Step3 successfully imports quality_gate from centralized_decorators")
        else:
            print("❌ Step3 does not have quality_gate imported")
            return False

        print("✅ Step3 quality gate test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error testing step3 quality gate: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    pass
    pass
    # Run tests
    success1 = test_centralized_decorators()
    success2 = asyncio.run(test_async_decorators())
    success3 = test_step3_quality_gate()

    if all([success1, success2, success3]):
    pass
    pass
        print("\\\n🎉 All tests passed! Centralized decorators are working correctly.")
    else:
        print("\\\n❌ Some tests failed. Please check the implementation.")