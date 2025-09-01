#!/usr/bin/env python3
"""
Comprehensive Test Script for Standardized Pipeline

This script tests all the standardized pipeline fixes including:
- Import management with consistent fallback patterns
- Directory structure standardization
- Timestamp format standardization
- Schema validation
- Data quality validation
- File naming conventions
- Metadata standards
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import pipeline standards
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards


import class StandardizedPipelineTester:
class StandardizedPipelineTester:
    """Comprehensive tester for standardized pipeline utilities."""

    def __init__(self):
    pass
    pass
        self.logger = pipeline_standards.logger
        self.test_results = {}

    def test_import_management(self) -> bool:
    pass
    pass
        """Test standardized import management."""
        self.logger.info("🧪 Testing import management...")

        try:
            # Test safe import with existing module
    except Exception as e:
        pass
    except Exception as e:
        pass
            pandas_module = PipelineStandards.safe_import("pandas", None, self.logger)
            if pandas_module is not None:
    pass
    pass
                self.logger.info("✅ Safe import of pandas successful")
            else:
                self.logger.error("❌ Safe import of pandas failed")
                return False

            # Test safe import with non-existent module
            fake_module = PipelineStandards.safe_import("fake_module_that_does_not_exist", "fallback", self.logger)
            if fake_module == "fallback":
    pass
    pass
                self.logger.info("✅ Safe import fallback working correctly")
            else:
                self.logger.error("❌ Safe import fallback not working")
                return False

            # Test environment dependency validation
            required_modules = ["pandas", "numpy", "fake_module"]
            availability = PipelineStandards.validate_environment_dependencies(required_modules, self.logger)

            if availability.get("pandas", False) and availability.get("numpy", False) and not availability.get("fake_module", True):
    pass
    pass
                self.logger.info("✅ Environment dependency validation working correctly")
            else:
                self.logger.error("❌ Environment dependency validation failed")
                return False

            self.test_results["import_management"] = True
            return True

        except Exception as e:
            self.logger.error(f"❌ Import management test failed: {e}")
            self.test_results["import_management"] = False
            return False

    def test_directory_structure(self) -> bool:
    pass
    pass
        """Test standardized directory structure."""
        self.logger.info("🧪 Testing directory structure...")

        try:
            # Test path building
    except Exception as e:
        pass
    except Exception as e:
        pass
            raw_data_path = pipeline_standards.build_path("raw_data", "BINANCE", "ETHUSDT")
            expected_raw_path = "data_cache/binance/ethusdt"

            if raw_data_path == expected_raw_path:
    pass
    pass
                self.logger.info("✅ Raw data path building correct")
            else:
                self.logger.error(f"❌ Raw data path building failed: expected {expected_raw_path}, got {raw_data_path}")
                return False

            # Test unified data path
            unified_data_path = pipeline_standards.build_path("unified_data", "BINANCE", "ETHUSDT")
            expected_unified_path = "data_cache/binance/ethusdt/unified"

            if unified_data_path == expected_unified_path:
    pass
    pass
                self.logger.info("✅ Unified data path building correct")
            else:
                self.logger.error(f"❌ Unified data path building failed: expected {expected_unified_path}, got {unified_data_path}")
                return False

            # Test with additional parameters
            processed_data_path = pipeline_standards.build_path("processed_data", "BINANCE", "ETHUSDT", timeframe="1m")
            expected_processed_path = "data_cache/binance/ethusdt/processed"

            if processed_data_path == expected_processed_path:
    pass
    pass
                self.logger.info("✅ Processed data path building correct")
            else:
                self.logger.error(f"❌ Processed data path building failed: expected {expected_processed_path}, got {processed_data_path}")
                return False

            self.test_results["directory_structure"] = True
            return True

        except Exception as e:
            self.logger.error(f"❌ Directory structure test failed: {e}")
            self.test_results["directory_structure"] = False
            return False

    def test_timestamp_standardization(self) -> bool:
    pass
    pass
        """Test timestamp format standardization."""
        self.logger.info("🧪 Testing timestamp standardization...")

        try:
            # Create test data with different timestamp formats
    except Exception as e:
        pass
    except Exception as e:
        pass
            test_data = []
            base_time = datetime.now()

            for i in range(100):
    pass
    pass
                timestamp = base_time + timedelta(minutes=i)
                test_data.append({
                    'timestamp': timestamp,  # datetime object
                    'open': 100.0 + i,
                    'high': 101.0 + i,
                    'low': 99.0 + i,
                    'close': 100.5 + i,
                    'volume': 1000.0 + i
                })

            df = pd.DataFrame(test_data)

            # Test conversion to int64 milliseconds
            df_int64 = pipeline_standards.standardize_timestamp(df, "timestamp", "int64")

            if df_int64["timestamp"].dtype == "int64":
    pass
    pass
                self.logger.info("✅ Timestamp conversion to int64 successful")
            else:
                self.logger.error(f"❌ Timestamp conversion to int64 failed: got {df_int64['timestamp'].dtype}")
                return False

            # Test conversion back to datetime
            df_datetime = pipeline_standards.standardize_timestamp(df_int64, "timestamp", "datetime64[ns]")

            if pd.api.types.is_datetime64_any_dtype(df_datetime["timestamp"]):
    pass
    pass
                self.logger.info("✅ Timestamp conversion to datetime successful")
            else:
                self.logger.error(f"❌ Timestamp conversion to datetime failed: got {df_datetime['timestamp'].dtype}")
                return False

            # Test timestamp validation
            validation_result = pipeline_standards.validate_timestamp_format(df_int64, "timestamp", "int64")

            if validation_result.passed:
    pass
    pass
                self.logger.info("✅ Timestamp validation passed")
            else:
                self.logger.error(f"❌ Timestamp validation failed: {validation_result.issues}")
                return False

            self.test_results["timestamp_standardization"] = True
            return True

        except Exception as e:
            self.logger.error(f"❌ Timestamp standardization test failed: {e}")
            self.test_results["timestamp_standardization"] = False
            return False

    def test_schema_validation(self) -> bool:
    pass
    pass
        """Test schema validation."""
        self.logger.info("🧪 Testing schema validation...")

        try:
            # Create test data with correct schema
    except Exception as e:
        pass
    except Exception as e:
        pass
            test_data = []
            base_time = datetime.now()

            for i in range(100):
    pass
    pass
                timestamp = base_time + timedelta(minutes=i)
                test_data.append({
                    'timestamp': int(timestamp.timestamp() * 1000),  # int64 milliseconds
                    'open': 100.0 + i,
                    'high': 101.0 + i,
                    'low': 99.0 + i,
                    'close': 100.5 + i,
                    'volume': 1000.0 + i
                })

            df = pd.DataFrame(test_data)

            # Test klines schema validation
            validation_result = pipeline_standards.validate_schema(df, "klines")

            if validation_result.passed:
    pass
    pass
                self.logger.info("✅ Klines schema validation passed")
            else:
                self.logger.error(f"❌ Klines schema validation failed: {validation_result.issues}")
                return False

            # Test schema enforcement
            df_enforced = pipeline_standards.enforce_schema(df, "klines")

            # Check that all required columns are present
            required_columns = pipeline_standards.SCHEMAS["klines"]["required_columns"]
            missing_columns = [col for col in required_columns if col not in df_enforced.columns]

            if not missing_columns:
    pass
    pass
                self.logger.info("✅ Schema enforcement successful")
            else:
                self.logger.error(f"❌ Schema enforcement failed: missing columns {missing_columns}")
                return False

            # Test with missing required column
            df_missing = df.drop(columns=['close'])
            validation_result_missing = pipeline_standards.validate_schema(df_missing, "klines")

            if not validation_result_missing.passed:
    pass
    pass
                self.logger.info("✅ Schema validation correctly detected missing column")
            else:
                self.logger.error("❌ Schema validation should have detected missing column")
                return False

            self.test_results["schema_validation"] = True
            return True

        except Exception as e:
            self.logger.error(f"❌ Schema validation test failed: {e}")
            self.test_results["schema_validation"] = False
            return False

    def test_data_quality_validation(self) -> bool:
    pass
    pass
        """Test data quality validation."""
        self.logger.info("🧪 Testing data quality validation...")

        try:
            # Create test data with some quality issues
    except Exception as e:
        pass
    except Exception as e:
        pass
            test_data = []
            base_time = datetime.now()

            for i in range(100):
    pass
    pass
                timestamp = base_time + timedelta(minutes=i)
                test_data.append({
                    'timestamp': int(timestamp.timestamp() * 1000),
                    'open': 100.0 + i,
                    'high': 101.0 + i,
                    'low': 99.0 + i,
                    'close': 100.5 + i,
                    'volume': 1000.0 + i
                })

            # Add some quality issues
            test_data[50]['close'] = np.nan  # Missing value
            test_data[51]['high'] = -1.0  # Negative price
            test_data[52]['timestamp'] = test_data[51]['timestamp']  # Duplicate timestamp

            df = pd.DataFrame(test_data)

            # Test data quality validation
            validation_result = pipeline_standards.validate_data_quality(df, "klines")

            if validation_result.quality_score > 0:
    pass
    pass
                self.logger.info(f"✅ Data quality validation completed (score: {validation_result.quality_score:.2f})")
            else:
                self.logger.error("❌ Data quality validation failed")
                return False

            # Check that issues were detected
            if len(validation_result.issues) > 0:
    pass
    pass
                self.logger.info(f"✅ Data quality validation correctly detected {len(validation_result.issues)} issues")
            else:
                self.logger.warning("⚠️ Data quality validation should have detected some issues")

            # Test with empty dataframe
            df_empty = pd.DataFrame()
            validation_result_empty = pipeline_standards.validate_data_quality(df_empty, "klines")

            if not validation_result_empty.passed:
    pass
    pass
                self.logger.info("✅ Data quality validation correctly detected empty dataframe")
            else:
                self.logger.error("❌ Data quality validation should have detected empty dataframe")
                return False

            self.test_results["data_quality_validation"] = True
            return True

        except Exception as e:
            self.logger.error(f"❌ Data quality validation test failed: {e}")
            self.test_results["data_quality_validation"] = False
            return False

    def test_file_naming(self) -> bool:
    pass
    pass
        """Test standardized file naming."""
        self.logger.info("🧪 Testing file naming conventions...")

        try:
            # Test klines file naming
    except Exception as e:
        pass
    except Exception as e:
        pass
            klines_file = pipeline_standards.generate_file_name("klines", "BINANCE", "ETHUSDT", "1m")
            expected_klines = "klines_BINANCE_ETHUSDT_1m_consolidated.parquet"

            if klines_file == expected_klines:
    pass
    pass
                self.logger.info("✅ Klines file naming correct")
            else:
                self.logger.error(f"❌ Klines file naming failed: expected {expected_klines}, got {klines_file}")
                return False

            # Test aggtrades file naming
            aggtrades_file = pipeline_standards.generate_file_name("aggtrades", "BINANCE", "ETHUSDT")
            expected_aggtrades = "aggtrades_BINANCE_ETHUSDT_consolidated.parquet"

            if aggtrades_file == expected_aggtrades:
    pass
    pass
                self.logger.info("✅ Aggtrades file naming correct")
            else:
                self.logger.error(f"❌ Aggtrades file naming failed: expected {expected_aggtrades}, got {aggtrades_file}")
                return False

            # Test validation report naming
            report_file = pipeline_standards.generate_file_name("validation_report", "BINANCE", "ETHUSDT", "1m")

            if "validation_report_BINANCE_ETHUSDT_1m_" in report_file and report_file.endswith(".json"):
    pass
    pass
                self.logger.info("✅ Validation report naming correct")
            else:
                self.logger.error(f"❌ Validation report naming failed: got {report_file}")
                return False

            self.test_results["file_naming"] = True
            return True

        except Exception as e:
            self.logger.error(f"❌ File naming test failed: {e}")
            self.test_results["file_naming"] = False
            return False

    def test_metadata_creation(self) -> bool:
    pass
    pass
        """Test metadata creation."""
        self.logger.info("🧪 Testing metadata creation...")

        try:
            # Test metadata creation
    except Exception as e:
        pass
    except Exception as e:
        pass
            metadata = pipeline_standards.create_metadata("klines", "BINANCE", "ETHUSDT", "1m")

            required_keys = ["schema_name", "exchange", "asset", "timeframe", "created_at", "pipeline_version"]
            missing_keys = [key for key in required_keys if key not in metadata]

            if not missing_keys:
    pass
    pass
                self.logger.info("✅ Metadata creation successful")
            else:
                self.logger.error(f"❌ Metadata creation failed: missing keys {missing_keys}")
                return False

            # Check specific values
            if metadata["schema_name"] == "klines":
    pass
    pass
                self.logger.info("✅ Schema name correct")
            else:
                self.logger.error(f"❌ Schema name incorrect: expected 'klines', got {metadata['schema_name']}")
                return False

            if metadata["exchange"] == "BINANCE":
    pass
    pass
                self.logger.info("✅ Exchange name correct")
            else:
                self.logger.error(f"❌ Exchange name incorrect: expected 'BINANCE', got {metadata['exchange']}")
                return False

            self.test_results["metadata_creation"] = True
            return True

        except Exception as e:
            self.logger.error(f"❌ Metadata creation test failed: {e}")
            self.test_results["metadata_creation"] = False
            return False

    def run_all_tests(self) -> Dict[str, bool]:
    pass
    pass
        """Run all tests and return results."""
        self.logger.info("🚀 Starting comprehensive standardized pipeline tests...")

        tests = [
            ("import_management", self.test_import_management),
            ("directory_structure", self.test_directory_structure),
            ("timestamp_standardization", self.test_timestamp_standardization),
            ("schema_validation", self.test_schema_validation),
            ("data_quality_validation", self.test_data_quality_validation),
            ("file_naming", self.test_file_naming),
            ("metadata_creation", self.test_metadata_creation),
        ]

        for test_name, test_func in tests:
    pass
    pass
            self.logger.info(f"\\\n{'='*60}")
            self.logger.info(f"🧪 Running {test_name} test...")
            self.logger.info(f"{'='*60}")

            try:
                success = test_func()
    except Exception as e:
        pass
    except Exception as e:
        pass
                if success:
    pass
    pass
                    self.logger.info(f"✅ {test_name} test PASSED")
                else:
                    self.logger.error(f"❌ {test_name} test FAILED")
            except Exception as e:
                self.logger.error(f"❌ {test_name} test ERROR: {e}")
                self.test_results[test_name] = False

        # Summary
        self.logger.info(f"\\\n{'='*60}")
        self.logger.info("📊 TEST SUMMARY")
        self.logger.info(f"{'='*60}")

        passed_tests = sum(1 for result in self.test_results.values() if result)
        total_tests = len(self.test_results)

        for test_name, result in self.test_results.items():
    pass
    pass
            status = "✅ PASSED" if result else "❌ FAILED"
            self.logger.info(f"   {test_name}: {status}")

        self.logger.info(f"\\\nOverall: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests:
    pass
    pass
            self.logger.info("🎉 All tests passed! Standardized pipeline is working correctly.")
        else:
            self.logger.error(f"⚠️ {total_tests - passed_tests} tests failed. Please review the issues above.")

        return self.test_results


async def main():
    """Main test function."""
    print("🚀 Starting Standardized Pipeline Tests")
    print("=" * 60)

    tester = StandardizedPipelineTester()
    results = tester.run_all_tests()

    # Return exit code based on test results
    if all(results.values()):
    pass
    pass
        print("\\\n🎉 All tests passed!")
        return 0
    else:
        print(f"\\\n⚠️ {sum(1 for r in results.values() if not r)} tests failed!")
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
        print("\\\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\\n❌ Test execution failed: {e}")
        sys.exit(1)