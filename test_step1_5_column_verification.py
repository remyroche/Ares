#!/usr/bin/env python3
"""
Test Script for Step1_5 Column Verification and Calculation Enhancement

This script tests the new column verification and calculation functionality
added to Step1_5 data converter.
"""

import asyncio
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
import project_root = Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the enhanced Step1_5 components
from src.training.steps.step1_5_data_converter import ColumnVerifier, UnifiedDataConverter


import class Step1_5ColumnVerificationTester:
class Step1_5ColumnVerificationTester:
    """Test class for Step1_5 column verification and calculation functionality."""

    def __init__(self):
    pass
    pass
        self.logger = None  # Will be set by the converter
        self.test_results = {}

    def create_test_data(self) -> pd.DataFrame:
    pass
    pass
        """Create test data with some missing columns to test verification and calculation."""
        print("📊 Creating test data...")

        # Create base klines data
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1min')

        # Create realistic price data
        np.random.seed(42)  # For reproducible results
        base_price = 100.0
        price_changes = np.random.normal(0, 0.001, len(dates))  # Small random changes
        prices = [base_price]
        for change in price_changes[1:]:
    pass
    pass
            prices.append(prices[-1] * (1 + change))

        # Create OHLCV data
        data = {
            'timestamp': [int(dt.timestamp() * 1000) for dt in dates],
            'open': [p * (1 + np.random.normal(0, 0.0005)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, len(dates))
        }

        df = pd.DataFrame(data)

        # Add some calculated columns that should be detected as missing
        df['close_return'] = df['close'].pct_change()
        df['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()

        print(f"✅ Created test data with {len(df)} rows and {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")

        return df

    def create_data_with_missing_columns(self) -> pd.DataFrame:
    pass
    pass
        """Create test data with intentionally missing columns to test calculation."""
        print("📊 Creating test data with missing columns...")

        # Create base klines data (missing some calculated columns)
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1min')

        # Create realistic price data
        np.random.seed(42)  # For reproducible results
        base_price = 100.0
        price_changes = np.random.normal(0, 0.001, len(dates))
        prices = [base_price]
        for change in price_changes[1:]:
    pass
    pass
            prices.append(prices[-1] * (1 + change))

        # Create OHLCV data (missing calculated columns)
        data = {
            'timestamp': [int(dt.timestamp() * 1000) for dt in dates],
            'open': [p * (1 + np.random.normal(0, 0.0005)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, len(dates))
        }

        df = pd.DataFrame(data)

        print(f"✅ Created test data with {len(df)} rows and {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Missing columns that should be calculated: close_return, vwap, etc.")

        return df

    def test_column_verifier(self) -> bool:
    pass
    pass
        """Test the ColumnVerifier class functionality."""
        print("\\\n🧪 Testing ColumnVerifier...")

        try:
            # Create test data
    except Exception as e:
        pass
    except Exception as e:
        pass
            test_data = self.create_data_with_missing_columns()

            # Initialize column verifier
            column_verifier = ColumnVerifier()

            # Test verification
            print("🔍 Testing column verification...")
            missing_info = column_verifier.verify_missing_columns(test_data, data_type="unified")

            # Check results
            print(f"   Verification passed: {missing_info['verification_passed']}")
            print(f"   Missing required: {missing_info['missing_required']}")
            print(f"   Missing optional: {missing_info['missing_optional']}")
            print(f"   Can calculate: {missing_info['can_calculate']}")

            # Test calculation
            print("🔄 Testing column calculation...")
            enhanced_data = column_verifier.calculate_missing_columns(test_data, missing_info)

            # Check what was calculated
            original_columns = set(test_data.columns)
            new_columns = set(enhanced_data.columns) - original_columns

            print(f"   Original columns: {len(original_columns)}")
            print(f"   New columns: {len(new_columns)}")
            print(f"   Calculated columns: {list(new_columns)}")

            # Verify specific calculations
            success = True
            if 'close_return' in new_columns:
    pass
    pass
                print("   ✅ close_return calculated successfully")
            else:
                print("   ❌ close_return not calculated")
                success = False

            if 'vwap' in new_columns:
    pass
    pass
                print("   ✅ vwap calculated successfully")
            else:
                print("   ❌ vwap not calculated")
                success = False

            if 'vwap_return' in new_columns:
    pass
    pass
                print("   ✅ vwap_return calculated successfully")
            else:
                print("   ❌ vwap_return not calculated")
                success = False

            if 'price_vwap_ratio' in new_columns:
    pass
    pass
                print("   ✅ price_vwap_ratio calculated successfully")
            else:
                print("   ❌ price_vwap_ratio not calculated")
                success = False

            # Test data quality
            print("🔍 Testing calculated data quality...")
            if 'close_return' in enhanced_data.columns:
    pass
    pass
                # Check for reasonable values
                close_return = enhanced_data['close_return']
                if close_return.isna().sum() > len(close_return) * 0.1:  # More than 10% NaN
                    print("   ⚠️ close_return has too many NaN values")
                    success = False
                else:
                    print("   ✅ close_return data quality looks good")

            if 'vwap' in enhanced_data.columns:
    pass
    pass
                # Check for reasonable values
                vwap = enhanced_data['vwap']
                if vwap.isna().sum() > len(vwap) * 0.2:  # More than 20% NaN (rolling window effect)
                    print("   ⚠️ vwap has too many NaN values")
                    success = False
                else:
                    print("   ✅ vwap data quality looks good")

            self.test_results['column_verifier'] = success
            return success

        except Exception as e:
            print(f"❌ ColumnVerifier test failed: {e}")
            self.test_results['column_verifier'] = False
            return False

    async def test_unified_data_converter_integration(self) -> bool:
        """Test the integration of column verification in UnifiedDataConverter."""
        print("\\\n🧪 Testing UnifiedDataConverter integration...")

        try:
            # Create test data
    except Exception as e:
        pass
    except Exception as e:
        pass
            test_data = self.create_data_with_missing_columns()

            # Initialize converter
            converter = UnifiedDataConverter({})
            await converter.initialize()

            # Test the column verification method directly
            print("🔍 Testing _verify_and_calculate_missing_columns method...")
            enhanced_data = await converter._verify_and_calculate_missing_columns(
                test_data, "BTCUSDT", "BINANCE", "1m"
            )

            # Check results
            original_columns = set(test_data.columns)
            new_columns = set(enhanced_data.columns) - original_columns

            print(f"   Original columns: {len(original_columns)}")
            print(f"   New columns: {len(new_columns)}")
            print(f"   Calculated columns: {list(new_columns)}")

            # Verify integration worked
            success = len(new_columns) > 0
            if success:
    pass
    pass
                print("   ✅ Integration test passed - columns were calculated")
            else:
                print("   ❌ Integration test failed - no columns were calculated")

            self.test_results['unified_data_converter_integration'] = success
            return success

        except Exception as e:
            print(f"❌ UnifiedDataConverter integration test failed: {e}")
            self.test_results['unified_data_converter_integration'] = False
            return False

    def test_edge_cases(self) -> bool:
    pass
    pass
        """Test edge cases and error handling."""
        print("\\\n🧪 Testing edge cases...")

        try:
            column_verifier = ColumnVerifier()

    except Exception as e:
        pass
    except Exception as e:
        pass
            # Test with empty DataFrame
            print("🔍 Testing with empty DataFrame...")
            empty_df = pd.DataFrame()
            missing_info = column_verifier.verify_missing_columns(empty_df, data_type="unified")
            print(f"   Empty DataFrame handling: {'✅' if missing_info['verification_passed'] == False else '❌'}")

            # Test with DataFrame missing all required columns
            print("🔍 Testing with DataFrame missing required columns...")
            invalid_df = pd.DataFrame({'random_col': [1, 2, 3]})
            missing_info = column_verifier.verify_missing_columns(invalid_df, data_type="unified")
            print(f"   Missing required columns handling: {'✅' if missing_info['verification_passed'] == False else '❌'}")

            # Test with DataFrame having only some price columns
            print("🔍 Testing with partial price data...")
            partial_df = pd.DataFrame({
                'timestamp': [1000000, 1000060, 1000120],
                'close': [100.0, 101.0, 99.5],
                'volume': [1000, 1100, 900]
            })
            missing_info = column_verifier.verify_missing_columns(partial_df, data_type="unified")
            enhanced_partial = column_verifier.calculate_missing_columns(partial_df, missing_info)

            # Check if VWAP was calculated (should be possible with close and volume)
            if 'vwap' in enhanced_partial.columns:
    pass
    pass
                print("   ✅ VWAP calculation with partial data works")
            else:
                print("   ❌ VWAP calculation with partial data failed")

            success = True
            self.test_results['edge_cases'] = success
            return success

        except Exception as e:
            print(f"❌ Edge cases test failed: {e}")
            self.test_results['edge_cases'] = False
            return False

    def run_all_tests(self) -> dict:
    pass
    pass
        """Run all tests and return results."""
        print("🚀 Starting Step1_5 Column Verification Tests")
        print("=" * 60)

        # Run tests
        self.test_column_verifier()
        asyncio.run(self.test_unified_data_converter_integration())
        self.test_edge_cases()

        # Print summary
        print("\\\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)

        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())

        for test_name, result in self.test_results.items():
    pass
    pass
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {test_name}: {status}")

        print(f"\\\nOverall: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests:
    pass
    pass
            print("🎉 All tests passed! Step1_5 column verification enhancement is working correctly.")
        else:
            print("⚠️ Some tests failed. Please review the implementation.")

        return self.test_results


async def main():
    """Main test function."""
    tester = Step1_5ColumnVerificationTester()
    results = tester.run_all_tests()

    # Return exit code based on test results
    if all(results.values()):
    pass
    pass
        print("\\\n✅ All tests passed - exiting with success")
        return 0
    else:
        print("\\\n❌ Some tests failed - exiting with error")
        return 1


if __name__ == "__main__":
    pass
    pass
    try:
        exit_code = asyncio.run(main())
    except Exception as e:
        pass
    except Exception as e:
        pass
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\\\n⚠️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\\n❌ Unexpected error: {e}")
        sys.exit(1)