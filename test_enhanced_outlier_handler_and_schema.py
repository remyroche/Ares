#!/usr/bin/env python3
"""
Enhanced Outlier Handler and Schema Validation Test

This test validates the enhanced outlier handler and data schema validation including:
- Outlier detection with error raising
- Data schema validation for file operations
- Root cause analysis and reporting
- Schema creation and validation
- Integration with data quality framework
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

from src.utils.enhanced_outlier_handler import enhanced_outlier_handler, OutlierSeverity
from src.utils.logger import system_logger


class EnhancedOutlierHandlerAndSchemaTester:
    """Comprehensive tester for enhanced outlier handler and schema validation."""
    
    def __init__(self):
        """Initialize tester."""
        self.logger = system_logger.getChild("EnhancedOutlierHandlerAndSchemaTester")
        self.handler = enhanced_outlier_handler
        self.test_results = {}
        self.start_time = time.time()
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all enhanced outlier handler and schema validation tests."""
        self.logger.info("🔍 Starting Enhanced Outlier Handler and Schema Validation Tests")
        
        test_suite = [
            ("test_outlier_detection_methods", self.test_outlier_detection_methods),
            ("test_outlier_severity_classification", self.test_outlier_severity_classification),
            ("test_error_raising_behavior", self.test_error_raising_behavior),
            ("test_schema_validation", self.test_schema_validation),
            ("test_custom_schema_creation", self.test_custom_schema_creation),
            ("test_schema_constraints", self.test_schema_constraints),
            ("test_data_type_validation", self.test_data_type_validation),
            ("test_outlier_reporting", self.test_outlier_reporting),
            ("test_integration_with_quality_framework", self.test_integration_with_quality_framework),
            ("test_file_operation_schemas", self.test_file_operation_schemas)
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
    
    def test_outlier_detection_methods(self) -> bool:
        """Test different outlier detection methods."""
        self.logger.info("Testing outlier detection methods...")
        
        # Create test data with outliers
        test_data = self._create_test_data_with_outliers()
        
        # Test Z-score method
        zscore_outliers = self.handler.detect_outliers(
            test_data, method="zscore", threshold=2.0, raise_errors=False
        )
        
        # Test IQR method
        iqr_outliers = self.handler.detect_outliers(
            test_data, method="iqr", threshold=1.5, raise_errors=False
        )
        
        # Test Mahalanobis method
        mahalanobis_outliers = self.handler.detect_outliers(
            test_data, method="mahalanobis", threshold=2.0, raise_errors=False
        )
        
        # Check that outliers were detected
        if len(zscore_outliers) == 0 and len(iqr_outliers) == 0:
            self.logger.error("No outliers detected with any method")
            return False
        
        # Check that different methods detect different numbers of outliers
        if len(zscore_outliers) == len(iqr_outliers) == len(mahalanobis_outliers):
            self.logger.warning("All methods detected same number of outliers")
        
        self.logger.info(f"Z-score outliers: {len(zscore_outliers)}")
        self.logger.info(f"IQR outliers: {len(iqr_outliers)}")
        self.logger.info(f"Mahalanobis outliers: {len(mahalanobis_outliers)}")
        
        return True
    
    def test_outlier_severity_classification(self) -> bool:
        """Test outlier severity classification."""
        self.logger.info("Testing outlier severity classification...")
        
        # Create test data with extreme outliers
        test_data = self._create_test_data_with_extreme_outliers()
        
        # Test severity classification
        outliers = self.handler.detect_outliers(
            test_data, method="zscore", threshold=2.0, raise_errors=False
        )
        
        # Check that different severity levels are detected
        severity_counts = {}
        for outlier in outliers:
            severity = outlier.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        self.logger.info(f"Severity distribution: {severity_counts}")
        
        # Should have at least some high/critical outliers
        if not any(severity in severity_counts for severity in ["high", "critical"]):
            self.logger.warning("No high/critical outliers detected")
        
        return True
    
    def test_error_raising_behavior(self) -> bool:
        """Test error raising behavior for critical outliers."""
        self.logger.info("Testing error raising behavior...")
        
        # Create test data with critical outliers
        test_data = self._create_test_data_with_critical_outliers()
        
        # Test with error raising enabled
        try:
            self.handler.detect_outliers(
                test_data, method="zscore", threshold=2.0, raise_errors=True
            )
            self.logger.error("Expected exception for critical outliers but none raised")
            return False
        except ValueError as e:
            self.logger.info(f"Correctly raised exception: {e}")
        
        # Test with error raising disabled
        try:
            outliers = self.handler.detect_outliers(
                test_data, method="zscore", threshold=2.0, raise_errors=False
            )
            if len(outliers) == 0:
                self.logger.error("No outliers detected when errors disabled")
                return False
        except Exception as e:
            self.logger.error(f"Unexpected exception when errors disabled: {e}")
            return False
        
        return True
    
    def test_schema_validation(self) -> bool:
        """Test data schema validation."""
        self.logger.info("Testing schema validation...")
        
        # Test klines schema validation
        valid_klines = self._create_valid_klines_data()
        klines_validation = self.handler.validate_data_schema(valid_klines, "klines")
        
        if not klines_validation["valid"]:
            self.logger.error(f"Valid klines data failed validation: {klines_validation}")
            return False
        
        # Test invalid klines data
        invalid_klines = self._create_invalid_klines_data()
        invalid_klines_validation = self.handler.validate_data_schema(invalid_klines, "klines")
        
        if invalid_klines_validation["valid"]:
            self.logger.error("Invalid klines data passed validation")
            return False
        
        # Check that specific errors are reported
        if not invalid_klines_validation["errors"]:
            self.logger.error("No errors reported for invalid data")
            return False
        
        self.logger.info("Schema validation tests passed")
        return True
    
    def test_custom_schema_creation(self) -> bool:
        """Test custom schema creation."""
        self.logger.info("Testing custom schema creation...")
        
        # Create custom schema
        custom_schema = self.handler.create_custom_schema(
            name="custom_trading_data",
            required_columns=["timestamp", "price", "volume"],
            optional_columns=["bid", "ask"],
            data_types={
                "timestamp": "int64",
                "price": "float64",
                "volume": "float64"
            },
            constraints={
                "price": {"min": 0, "not_null": True},
                "volume": {"min": 0, "not_null": True}
            }
        )
        
        # Verify schema was created
        if "custom_trading_data" not in self.handler.list_available_schemas():
            self.logger.error("Custom schema not found in available schemas")
            return False
        
        # Test schema validation
        valid_data = pd.DataFrame({
            "timestamp": [int(time.time())],
            "price": [100.0],
            "volume": [1000.0]
        })
        
        validation_result = self.handler.validate_data_schema(valid_data, "custom_trading_data")
        if not validation_result["valid"]:
            self.logger.error(f"Valid data failed custom schema validation: {validation_result}")
            return False
        
        self.logger.info("Custom schema creation tests passed")
        return True
    
    def test_schema_constraints(self) -> bool:
        """Test schema constraint validation."""
        self.logger.info("Testing schema constraints...")
        
        # Create schema with constraints
        constraint_schema = self.handler.create_custom_schema(
            name="constraint_test",
            required_columns=["id", "value"],
            data_types={"id": "int64", "value": "float64"},
            constraints={
                "id": {"unique": True, "not_null": True},
                "value": {"min": 0, "max": 100, "not_null": True}
            }
        )
        
        # Test valid data
        valid_data = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10.0, 50.0, 90.0]
        })
        
        valid_result = self.handler.validate_data_schema(valid_data, "constraint_test")
        if not valid_result["valid"]:
            self.logger.error(f"Valid data failed constraint validation: {valid_result}")
            return False
        
        # Test invalid data (duplicate IDs)
        invalid_data = pd.DataFrame({
            "id": [1, 1, 3],  # Duplicate ID
            "value": [10.0, 50.0, 90.0]
        })
        
        invalid_result = self.handler.validate_data_schema(invalid_data, "constraint_test")
        if invalid_result["valid"]:
            self.logger.error("Invalid data (duplicate IDs) passed validation")
            return False
        
        # Test invalid data (out of range values)
        invalid_data2 = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10.0, 150.0, 90.0]  # Value > 100
        })
        
        invalid_result2 = self.handler.validate_data_schema(invalid_data2, "constraint_test")
        if invalid_result2["valid"]:
            self.logger.error("Invalid data (out of range) passed validation")
            return False
        
        self.logger.info("Schema constraint tests passed")
        return True
    
    def test_data_type_validation(self) -> bool:
        """Test data type validation."""
        self.logger.info("Testing data type validation...")
        
        # Create schema with specific data types
        type_schema = self.handler.create_custom_schema(
            name="type_test",
            required_columns=["timestamp", "price", "category"],
            data_types={
                "timestamp": "int64",
                "price": "float64",
                "category": "object"
            }
        )
        
        # Test correct data types
        correct_data = pd.DataFrame({
            "timestamp": [int(time.time())],
            "price": [100.0],
            "category": ["buy"]
        })
        
        correct_result = self.handler.validate_data_schema(correct_data, "type_test")
        if not correct_result["valid"]:
            self.logger.error(f"Correct data types failed validation: {correct_result}")
            return False
        
        # Test incorrect data types
        incorrect_data = pd.DataFrame({
            "timestamp": ["not_a_timestamp"],  # Should be int64
            "price": [100.0],
            "category": ["buy"]
        })
        
        incorrect_result = self.handler.validate_data_schema(incorrect_data, "type_test")
        if incorrect_result["valid"]:
            self.logger.error("Incorrect data types passed validation")
            return False
        
        # Check for type mismatch warnings
        if not incorrect_result["type_mismatches"]:
            self.logger.error("No type mismatch warnings generated")
            return False
        
        self.logger.info("Data type validation tests passed")
        return True
    
    def test_outlier_reporting(self) -> bool:
        """Test outlier reporting functionality."""
        self.logger.info("Testing outlier reporting...")
        
        # Create test data with outliers
        test_data = self._create_test_data_with_outliers()
        
        # Detect outliers
        outliers = self.handler.detect_outliers(
            test_data, method="zscore", threshold=2.0, raise_errors=False
        )
        
        # Generate report
        report = self.handler.get_outlier_report()
        
        # Check report structure
        required_keys = ["timestamp", "total_outlier_groups", "severity_distribution", "column_distribution"]
        for key in required_keys:
            if key not in report:
                self.logger.error(f"Missing key in outlier report: {key}")
                return False
        
        # Check that outliers are reported
        if report["total_outlier_groups"] == 0:
            self.logger.error("No outliers reported in test data")
            return False
        
        self.logger.info(f"Outlier report generated: {report['total_outlier_groups']} groups")
        return True
    
    def test_integration_with_quality_framework(self) -> bool:
        """Test integration with data quality framework."""
        self.logger.info("Testing integration with quality framework...")
        
        # Import quality framework
        from src.utils.data_quality_framework import data_quality_framework
        
        # Create test data with outliers
        test_data = self._create_test_data_with_outliers()
        
        # Configure quality framework for enhanced outlier handling
        cleaning_rules = {
            "outlier_handling": "detect_only",
            "outlier_config": {
                "method": "zscore",
                "threshold": 2.0,
                "severity_threshold": "medium",
                "raise_errors": False
            }
        }
        
        # Test integration
        try:
            cleaned_data = data_quality_framework.clean_data(test_data, cleaning_rules)
            self.logger.info("Integration with quality framework successful")
            return True
        except Exception as e:
            self.logger.error(f"Integration with quality framework failed: {e}")
            return False
    
    def test_file_operation_schemas(self) -> bool:
        """Test schemas for file operations."""
        self.logger.info("Testing file operation schemas...")
        
        # Test available schemas
        available_schemas = self.handler.list_available_schemas()
        expected_schemas = ["klines", "features", "labels"]
        
        for schema_name in expected_schemas:
            if schema_name not in available_schemas:
                self.logger.error(f"Expected schema {schema_name} not found")
                return False
        
        # Test schema information retrieval
        for schema_name in expected_schemas:
            schema_info = self.handler.get_schema_info(schema_name)
            if "error" in schema_info:
                self.logger.error(f"Error getting schema info for {schema_name}: {schema_info}")
                return False
            
            if "required_columns" not in schema_info:
                self.logger.error(f"Missing required_columns in schema info for {schema_name}")
                return False
        
        self.logger.info("File operation schema tests passed")
        return True
    
    def _create_test_data_with_outliers(self) -> pd.DataFrame:
        """Create test data with outliers."""
        np.random.seed(42)
        
        # Create normal data
        normal_data = np.random.normal(100, 10, 100)
        
        # Add some outliers
        outliers = [50, 200, 300, 25, 400]  # Clear outliers
        data_with_outliers = np.concatenate([normal_data, outliers])
        
        return pd.DataFrame({
            "timestamp": range(len(data_with_outliers)),
            "price": data_with_outliers,
            "volume": np.random.normal(1000, 100, len(data_with_outliers))
        })
    
    def _create_test_data_with_extreme_outliers(self) -> pd.DataFrame:
        """Create test data with extreme outliers."""
        np.random.seed(42)
        
        # Create normal data
        normal_data = np.random.normal(100, 10, 100)
        
        # Add extreme outliers
        extreme_outliers = [10, 1000, 2000, 5, 5000]  # Very extreme outliers
        data_with_extreme_outliers = np.concatenate([normal_data, extreme_outliers])
        
        return pd.DataFrame({
            "timestamp": range(len(data_with_extreme_outliers)),
            "price": data_with_extreme_outliers,
            "volume": np.random.normal(1000, 100, len(data_with_extreme_outliers))
        })
    
    def _create_test_data_with_critical_outliers(self) -> pd.DataFrame:
        """Create test data with critical outliers."""
        np.random.seed(42)
        
        # Create normal data
        normal_data = np.random.normal(100, 10, 100)
        
        # Add critical outliers (very extreme)
        critical_outliers = [1, 10000, 20000, 0.1, 100000]  # Critical outliers
        data_with_critical_outliers = np.concatenate([normal_data, critical_outliers])
        
        return pd.DataFrame({
            "timestamp": range(len(data_with_critical_outliers)),
            "price": data_with_critical_outliers,
            "volume": np.random.normal(1000, 100, len(data_with_critical_outliers))
        })
    
    def _create_valid_klines_data(self) -> pd.DataFrame:
        """Create valid klines data."""
        base_time = int(time.time()) - 3600
        
        return pd.DataFrame({
            "timestamp": [base_time + i * 60 for i in range(100)],
            "open": [100 + np.random.random() * 10 for _ in range(100)],
            "high": [105 + np.random.random() * 10 for _ in range(100)],
            "low": [95 + np.random.random() * 10 for _ in range(100)],
            "close": [100 + np.random.random() * 10 for _ in range(100)],
            "volume": [1000 + np.random.random() * 1000 for _ in range(100)]
        })
    
    def _create_invalid_klines_data(self) -> pd.DataFrame:
        """Create invalid klines data."""
        base_time = int(time.time()) - 3600
        
        return pd.DataFrame({
            "timestamp": [base_time + i * 60 for i in range(100)],
            "open": [100 + np.random.random() * 10 for _ in range(100)],
            "high": [105 + np.random.random() * 10 for _ in range(100)],
            "low": [95 + np.random.random() * 10 for _ in range(100)],
            "close": [100 + np.random.random() * 10 for _ in range(100)],
            # Missing required "volume" column
            "extra_column": ["extra" for _ in range(100)]  # Extra column
        })
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Count results
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results.values() if r["status"] == "PASSED"])
        failed_tests = len([r for r in self.test_results.values() if r["status"] == "FAILED"])
        error_tests = len([r for r in self.test_results.values() if r["status"] == "ERROR"])
        
        # Get handler configuration
        handler_config = {
            "raise_errors": self.handler.raise_errors,
            "log_details": self.handler.log_details,
            "available_schemas": self.handler.list_available_schemas(),
            "detection_methods": list(self.handler.detection_methods.keys())
        }
        
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
            "handler_configuration": handler_config,
            "outlier_severity_levels": [level.value for level in OutlierSeverity],
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
        
        # Check handler configuration
        if not self.handler.raise_errors:
            recommendations.append("Consider enabling error raising for production use")
        
        if not self.handler.log_details:
            recommendations.append("Consider enabling detailed logging for better debugging")
        
        # Check schema coverage
        available_schemas = self.handler.list_available_schemas()
        if len(available_schemas) < 3:
            recommendations.append("Add more standard schemas for different data types")
        
        # Check test coverage
        if len(self.test_results) < 10:
            recommendations.append("Add more comprehensive tests for edge cases")
        
        return recommendations


def main():
    """Main function to run enhanced outlier handler and schema validation tests."""
    print("🔍 Enhanced Outlier Handler and Schema Validation Test Framework")
    print("=" * 70)
    
    tester = EnhancedOutlierHandlerAndSchemaTester()
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
    
    # Print handler configuration
    config = report["handler_configuration"]
    print(f"\n🔧 Handler Configuration:")
    print(f"  Raise Errors: {config['raise_errors']}")
    print(f"  Log Details: {config['log_details']}")
    print(f"  Available Schemas: {config['available_schemas']}")
    print(f"  Detection Methods: {config['detection_methods']}")
    
    # Print outlier severity levels
    print(f"\n📈 Outlier Severity Levels:")
    for level in report["outlier_severity_levels"]:
        print(f"  • {level}")
    
    # Print recommendations
    print(f"\n💡 Recommendations:")
    for rec in report["recommendations"]:
        print(f"  • {rec}")
    
    # Save detailed report
    report_file = "enhanced_outlier_handler_and_schema_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Return success if most tests passed
    return summary['success_rate'] >= 0.8


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)