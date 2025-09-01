#!/usr/bin/env python3
"""
Comprehensive Data Quality and Formatting Testing Framework

This test validates all data quality and formatting measures including:
- Data validation and schema enforcement
- Data formatting and standardization
- Quality scoring and metrics
- Data cleaning and preprocessing
- Format validation and enforcement
- Cross-step format consistency
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.utils.data_formatting_framework import data_formatting_framework, DataFormat as FormatEnum
from src.utils.logger import system_logger


class DataQualityAndFormattingTester:
    """Comprehensive data quality and formatting testing framework."""

    def __init__(self):
        """Initialize tester."""
        self.logger = system_logger.getChild("DataQualityAndFormattingTester")
        self.quality_framework = data_quality_framework
        self.formatting_framework = data_formatting_framework
        self.test_results = {}
        self.start_time = time.time()

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all data quality and formatting tests."""
        self.logger.info("🔍 Starting Comprehensive Data Quality and Formatting Tests")

        test_suite = [
            ("test_data_validation", self.test_data_validation),
            ("test_data_formatting", self.test_data_formatting),
            ("test_data_cleaning", self.test_data_cleaning),
            ("test_data_profiling", self.test_data_profiling),
            ("test_quality_scoring", self.test_quality_scoring),
            ("test_format_validation", self.test_format_validation),
            ("test_timestamp_normalization", self.test_timestamp_normalization),
            ("test_missing_value_handling", self.test_missing_value_handling),
            ("test_data_type_standardization", self.test_data_type_standardization),
            ("test_cross_step_consistency", self.test_cross_step_consistency)
        ]

        for test_name, test_func in test_suite:
            try:
                self.logger.info(f"Running {test_name}...")
                result = test_func()
                self.test_results[test_name] = {
                    "status": "PASSED" if result else "FAILED",
                    "details": result
                }
                self.logger.info(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                self.logger.error(f"❌ {test_name} failed with exception: {e}")
                self.test_results[test_name] = {
                    "status": "ERROR",
                    "details": str(e)
                }

        return self.generate_test_report()

    def test_data_validation(self) -> bool:
        """Test data validation functionality."""
        self.logger.info("Testing data validation...")

        # Create test data
        test_data = self._create_test_klines_data()

        # Test validation
        validation_results = self.quality_framework.validate_data(test_data)

        # Check validation results
        if not validation_results["overall_passed"]:
            self.logger.error("Data validation failed")
            return False

        # Test specific validation rules
        schema_validation = self.quality_framework.validate_data(test_data, ["klines_schema"])
        if not schema_validation["overall_passed"]:
            self.logger.error("Schema validation failed")
            return False

        self.logger.info("Data validation tests passed")
        return True

    def test_data_formatting(self) -> bool:
        """Test data formatting functionality."""
        self.logger.info("Testing data formatting...")

        # Create test data with inconsistent format
        test_data = self._create_inconsistent_test_data()

        # Test formatting to klines format
        formatted_data = self.formatting_framework.standardize_format(test_data, FormatEnum.KLINES)

        # Validate formatted data
        validation_results = self.formatting_framework.validate_data_format(formatted_data, FormatEnum.KLINES)
        if not validation_results["valid"]:
            self.logger.error("Formatted data validation failed")
            return False

        # Test formatting to features format
        features_data = self._create_test_features_data()
        formatted_features = self.formatting_framework.standardize_format(features_data, FormatEnum.FEATURES)

        validation_results = self.formatting_framework.validate_data_format(formatted_features, FormatEnum.FEATURES)
        if not validation_results["valid"]:
            self.logger.error("Features formatting validation failed")
            return False

        self.logger.info("Data formatting tests passed")
        return True

    def test_data_cleaning(self) -> bool:
        """Test data cleaning functionality."""
        self.logger.info("Testing data cleaning...")

        # Create dirty test data
        dirty_data = self._create_dirty_test_data()

        # Test cleaning
        cleaned_data = self.quality_framework.clean_data(dirty_data)

        # Check that cleaning improved data quality
        dirty_quality = self.quality_framework.calculate_quality_score(dirty_data)
        cleaned_quality = self.quality_framework.calculate_quality_score(cleaned_data)

        if cleaned_quality <= dirty_quality:
            self.logger.error("Data cleaning did not improve quality")
            return False

        # Test specific cleaning operations
        # Test duplicate removal
        data_with_duplicates = pd.concat([dirty_data, dirty_data.iloc[:10]])
        cleaned_no_duplicates = self.quality_framework.clean_data(data_with_duplicates, {"remove_duplicates": True})

        if len(cleaned_no_duplicates) >= len(data_with_duplicates):
            self.logger.error("Duplicate removal failed")
            return False

        self.logger.info("Data cleaning tests passed")
        return True

    def test_data_profiling(self) -> bool:
        """Test data profiling functionality."""
        self.logger.info("Testing data profiling...")

        # Create test data
        test_data = self._create_test_klines_data()

        # Test profiling
        profile = self.quality_framework.profile_data(test_data)

        # Check profile structure
        required_keys = ["timestamp", "data_shape", "columns", "summary"]
        for key in required_keys:
            if key not in profile:
                self.logger.error(f"Missing key in profile: {key}")
                return False

        # Check profile content
        if profile["data_shape"] != test_data.shape:
            self.logger.error("Profile data shape mismatch")
            return False

        if len(profile["columns"]) != len(test_data.columns):
            self.logger.error("Profile columns count mismatch")
            return False

        self.logger.info("Data profiling tests passed")
        return True

    def test_quality_scoring(self) -> bool:
        """Test quality scoring functionality."""
        self.logger.info("Testing quality scoring...")

        # Create high-quality data
        high_quality_data = self._create_test_klines_data()

        # Create low-quality data
        low_quality_data = self._create_dirty_test_data()

        # Test quality scoring
        high_quality_score = self.quality_framework.calculate_quality_score(high_quality_data)
        low_quality_score = self.quality_framework.calculate_quality_score(low_quality_data)

        # Check that high-quality data has higher score
        if high_quality_score <= low_quality_score:
            self.logger.error("Quality scoring failed: high-quality data should have higher score")
            return False

        # Test quality metrics
        quality_report = self.quality_framework.get_quality_report(high_quality_data)

        required_metrics = ["completeness", "consistency", "accuracy", "timeliness"]
        for metric in required_metrics:
            if metric not in quality_report["quality_metrics"]:
                self.logger.error(f"Missing quality metric: {metric}")
                return False

        self.logger.info("Quality scoring tests passed")
        return True

    def test_format_validation(self) -> bool:
        """Test format validation functionality."""
        self.logger.info("Testing format validation...")

        # Test valid klines format
        valid_klines = self._create_test_klines_data()
        validation_results = self.formatting_framework.validate_data_format(valid_klines, FormatEnum.KLINES)

        if not validation_results["valid"]:
            self.logger.error("Valid klines format validation failed")
            return False

        # Test invalid format (missing required columns)
        invalid_data = valid_klines.drop(columns=["close"])
        validation_results = self.formatting_framework.validate_data_format(invalid_data, FormatEnum.KLINES)

        if validation_results["valid"]:
            self.logger.error("Invalid format validation should have failed")
            return False

        # Test format specification retrieval
        format_spec = self.formatting_framework.get_format_specification(FormatEnum.KLINES)
        required_keys = ["required_columns", "data_types", "column_order"]
        for key in required_keys:
            if key not in format_spec:
                self.logger.error(f"Missing key in format specification: {key}")
                return False

        self.logger.info("Format validation tests passed")
        return True

    def test_timestamp_normalization(self) -> bool:
        """Test timestamp normalization functionality."""
        self.logger.info("Testing timestamp normalization...")

        # Create test data with different timestamp formats
        test_data = self._create_test_klines_data()

        # Test unix seconds format
        normalized_data = self.formatting_framework.normalize_timestamps(test_data, "timestamp", "unix_seconds")

        # Check that timestamps are integers
        if not pd.api.types.is_integer_dtype(normalized_data["timestamp"]):
            self.logger.error("Timestamp normalization to unix_seconds failed")
            return False

        # Test ISO string format
        normalized_data = self.formatting_framework.normalize_timestamps(test_data, "timestamp", "iso_string")

        # Check that timestamps are strings
        if not pd.api.types.is_string_dtype(normalized_data["timestamp"]):
            self.logger.error("Timestamp normalization to ISO string failed")
            return False

        self.logger.info("Timestamp normalization tests passed")
        return True

    def test_missing_value_handling(self) -> bool:
        """Test missing value handling functionality."""
        self.logger.info("Testing missing value handling...")

        # Create test data with missing values
        test_data = self._create_test_klines_data()
        test_data.loc[0, "close"] = np.nan
        test_data.loc[1, "volume"] = np.nan

        # Test forward fill
        filled_data = self.formatting_framework.handle_missing_values(test_data, "forward_fill")

        # Check that missing values were handled
        if filled_data.isnull().sum().sum() > 0:
            self.logger.error("Forward fill did not handle all missing values")
            return False

        # Test median fill
        test_data.loc[0, "close"] = np.nan
        filled_data = self.formatting_framework.handle_missing_values(test_data, "median")

        if filled_data.isnull().sum().sum() > 0:
            self.logger.error("Median fill did not handle all missing values")
            return False

        self.logger.info("Missing value handling tests passed")
        return True

    def test_data_type_standardization(self) -> bool:
        """Test data type standardization functionality."""
        self.logger.info("Testing data type standardization...")

        # Create test data with mixed types
        test_data = self._create_mixed_type_test_data()

        # Test formatting to klines format (should standardize types)
        formatted_data = self.formatting_framework.standardize_format(test_data, FormatEnum.KLINES)

        # Check that numeric columns are float64
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            if col in formatted_data.columns:
                if not pd.api.types.is_float_dtype(formatted_data[col]):
                    self.logger.error(f"Column {col} is not float64 after standardization")
                    return False

        # Check that timestamp is int64
        if "timestamp" in formatted_data.columns:
            if not pd.api.types.is_integer_dtype(formatted_data["timestamp"]):
                self.logger.error("Timestamp is not int64 after standardization")
                return False

        self.logger.info("Data type standardization tests passed")
        return True

    def test_cross_step_consistency(self) -> bool:
        """Test cross-step format consistency."""
        self.logger.info("Testing cross-step format consistency...")

        # Create data for different steps
        klines_data = self._create_test_klines_data()
        features_data = self._create_test_features_data()
        labels_data = self._create_test_labels_data()

        # Format each to their respective formats
        formatted_klines = self.formatting_framework.standardize_format(klines_data, FormatEnum.KLINES)
        formatted_features = self.formatting_framework.standardize_format(features_data, FormatEnum.FEATURES)
        formatted_labels = self.formatting_framework.standardize_format(labels_data, FormatEnum.LABELS)

        # Check that all have timestamp column with same format
        timestamp_columns = ["timestamp"]
        for data, format_name in [(formatted_klines, "klines"), (formatted_features, "features"), (formatted_labels, "labels")]:
            if "timestamp" not in data.columns:
                self.logger.error(f"Missing timestamp column in {format_name}")
                return False

            if not pd.api.types.is_integer_dtype(data["timestamp"]):
                self.logger.error(f"Timestamp column in {format_name} is not integer")
                return False

        # Check that data can be merged on timestamp
        try:
            merged_data = formatted_klines.merge(formatted_features, on="timestamp", how="inner")
            merged_data = merged_data.merge(formatted_labels, on="timestamp", how="inner")
        except Exception as e:
            self.logger.error(f"Failed to merge data across steps: {e}")
            return False

        self.logger.info("Cross-step format consistency tests passed")
        return True

    def _create_test_klines_data(self) -> pd.DataFrame:
        """Create test klines data."""
        timestamps = pd.date_range(start="2024-01-01", periods=100, freq="1min").astype(np.int64) // 10**9
        data = {
            "timestamp": timestamps,
            "open": np.random.uniform(100, 200, 100),
            "high": np.random.uniform(200, 300, 100),
            "low": np.random.uniform(50, 100, 100),
            "close": np.random.uniform(100, 200, 100),
            "volume": np.random.uniform(1000, 10000, 100)
        }
        return pd.DataFrame(data)

    def _create_inconsistent_test_data(self) -> pd.DataFrame:
        """Create test data with inconsistent format."""
        timestamps = pd.date_range(start="2024-01-01", periods=100, freq="1min").astype(np.int64) // 10**9
        data = {
            "Timestamp": timestamps,  # Different case
            "OPEN": np.random.uniform(100, 200, 100),  # Different case
            "High": np.random.uniform(200, 300, 100),  # Different case
            "low": np.random.uniform(50, 100, 100),
            "Close": np.random.uniform(100, 200, 100),  # Different case
            "VOLUME": np.random.uniform(1000, 10000, 100)  # Different case
        }
        return pd.DataFrame(data)

    def _create_test_features_data(self) -> pd.DataFrame:
        """Create test features data."""
        timestamps = pd.date_range(start="2024-01-01", periods=100, freq="1min").astype(np.int64) // 10**9
        data = {
            "timestamp": timestamps,
            "feature_1": np.random.normal(0, 1, 100),
            "feature_2": np.random.normal(0, 1, 100),
            "feature_3": np.random.normal(0, 1, 100)
        }
        return pd.DataFrame(data)

    def _create_test_labels_data(self) -> pd.DataFrame:
        """Create test labels data."""
        timestamps = pd.date_range(start="2024-01-01", periods=100, freq="1min").astype(np.int64) // 10**9
        data = {
            "timestamp": timestamps,
            "label": np.random.randint(0, 3, 100),
            "label_probability": np.random.uniform(0, 1, 100)
        }
        return pd.DataFrame(data)

    def _create_dirty_test_data(self) -> pd.DataFrame:
        """Create dirty test data with quality issues."""
        timestamps = pd.date_range(start="2024-01-01", periods=100, freq="1min").astype(np.int64) // 10**9
        data = {
            "timestamp": timestamps,
            "open": np.random.uniform(100, 200, 100),
            "high": np.random.uniform(200, 300, 100),
            "low": np.random.uniform(50, 100, 100),
            "close": np.random.uniform(100, 200, 100),
            "volume": np.random.uniform(1000, 10000, 100)
        }

        df = pd.DataFrame(data)

        # Add quality issues
        df.loc[0, "close"] = np.nan  # Missing value
        df.loc[1, "volume"] = -1000  # Negative volume
        df.loc[2, "high"] = 50  # High < Low
        df.loc[3:5, "open"] = np.nan  # Multiple missing values

        return df

    def _create_mixed_type_test_data(self) -> pd.DataFrame:
        """Create test data with mixed types."""
        timestamps = pd.date_range(start="2024-01-01", periods=100, freq="1min").astype(np.int64) // 10**9
        data = {
            "timestamp": timestamps,
            "open": [str(x) for x in np.random.uniform(100, 200, 100)],  # Strings instead of floats
            "high": np.random.uniform(200, 300, 100),
            "low": np.random.uniform(50, 100, 100),
            "close": np.random.uniform(100, 200, 100),
            "volume": [int(x) for x in np.random.uniform(1000, 10000, 100)]  # Ints instead of floats
        }
        return pd.DataFrame(data)

    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        end_time = time.time()
        duration = end_time - self.start_time

        # Count results
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results.values() if r["status"] == "PASSED"])
        failed_tests = len([r for r in self.test_results.values() if r["status"] == "FAILED"])
        error_tests = len([r for r in self.test_results.values() if r["status"] == "ERROR"])

        # Get framework reports
        quality_report = self.quality_framework.get_quality_report(self._create_test_klines_data())
        formatting_report = self.formatting_framework.get_formatting_report(self._create_test_klines_data(), FormatEnum.KLINES)

        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "error_tests": error_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "duration_seconds": duration
            },
            "test_results": self.test_results,
            "quality_framework_report": quality_report,
            "formatting_framework_report": formatting_report,
            "framework_capabilities": {
                "available_formats": self.formatting_framework.list_available_formats(),
                "validation_rules": list(self.quality_framework.validation_rules.keys()),
                "quality_policies": self.quality_framework.quality_policies,
                "formatting_policies": self.formatting_framework.formatting_policies
            },
            "recommendations": self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        failed_tests = [name for name, result in self.test_results.items() if result["status"] == "FAILED"]
        error_tests = [name for name, result in self.test_results.items() if result["status"] == "ERROR"]

        if failed_tests:
            recommendations.append(f"Fix failed tests: {', '.join(failed_tests)}")

        if error_tests:
            recommendations.append(f"Investigate test errors: {', '.join(error_tests)}")

        # Check framework capabilities
        if len(self.quality_framework.validation_rules) < 5:
            recommendations.append("Add more validation rules for comprehensive data quality checking")

        if len(self.formatting_framework.standard_formats) < 4:
            recommendations.append("Add more standard data formats for comprehensive formatting")

        # Check quality policies
        if not self.quality_framework.quality_policies["strict_validation"]:
            recommendations.append("Enable strict validation for better data quality enforcement")

        if not self.quality_framework.quality_policies["auto_clean"]:
            recommendations.append("Consider enabling auto-cleaning for automatic data quality improvement")

        return recommendations


def main():
    """Main function to run data quality and formatting tests."""
    print("🔍 Comprehensive Data Quality and Formatting Testing Framework")
    print("=" * 70)

    tester = DataQualityAndFormattingTester()
    report = tester.run_all_tests()

    # Print summary
    summary = report["test_summary"]
    print(f"\n📊 Test Summary:")
    print(f"  Total Tests: {summary['total_tests']}")
    print(f"  Passed: {summary['passed_tests']}")
    print(f"  Failed: {summary['failed_tests']}")
    print(f"  Errors: {summary['error_tests']}")
    print(f"  Success Rate: {summary['success_rate']:.2%}")
    print(f"  Duration: {summary['duration_seconds']:.2f} seconds")

    # Print framework capabilities
    capabilities = report["framework_capabilities"]
    print(f"\n🔧 Framework Capabilities:")
    print(f"  Available Formats: {', '.join(capabilities['available_formats'])}")
    print(f"  Validation Rules: {len(capabilities['validation_rules'])}")
    print(f"  Quality Policies: {len(capabilities['quality_policies'])}")
    print(f"  Formatting Policies: {len(capabilities['formatting_policies'])}")

    # Print quality metrics
    quality_report = report["quality_framework_report"]
    print(f"\n📈 Quality Metrics:")
    print(f"  Overall Quality Score: {quality_report['quality_score']:.2%}")
    for metric, score in quality_report["quality_metrics"].items():
        print(f"  {metric.capitalize()}: {score:.2%}")

    # Print recommendations
    print(f"\n💡 Recommendations:")
    for rec in report["recommendations"]:
        print(f"  • {rec}")

    # Save detailed report
    report_file = "data_quality_and_formatting_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Detailed report saved to: {report_file}")

    # Return success if most tests passed
    return summary['success_rate'] >= 0.8


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)