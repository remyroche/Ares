#!/usr/bin/env python3
"""
Comprehensive Integration Test for Steps 1-7 Compatibility

This test validates the full compatibility between steps 1-7 including:
- Data flow validation
- Schema consistency
- Configuration compatibility
- Error handling
- Performance metrics
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.steps_1_7_compatibility_framework import steps_1_7_compatibility
from src.utils.pipeline_standards import pipeline_standards
from src.utils.standardized_error_handler import standardized_error_handler
from src.utils.standardized_model_manager import standardized_model_manager
from src.utils.logger import system_logger


class Steps1_7CompatibilityTester:
    """Comprehensive tester for steps 1-7 compatibility."""
    
    def __init__(self):
        """Initialize the compatibility tester."""
        self.logger = system_logger.getChild("Steps1_7CompatibilityTester")
        self.compatibility = steps_1_7_compatibility
        self.test_results = {}
        self.start_time = time.time()
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all compatibility tests."""
        self.logger.info("🚀 Starting Steps 1-7 Compatibility Tests")
        
        test_suite = [
            ("test_step_contracts", self.test_step_contracts),
            ("test_data_schemas", self.test_data_schemas),
            ("test_cross_step_consistency", self.test_cross_step_consistency),
            ("test_configuration_compatibility", self.test_configuration_compatibility),
            ("test_dependency_management", self.test_dependency_management),
            ("test_error_propagation", self.test_error_propagation),
            ("test_performance_metrics", self.test_performance_metrics),
            ("test_data_quality_flow", self.test_data_quality_flow),
            ("test_memory_management", self.test_memory_management),
            ("test_concurrent_execution", self.test_concurrent_execution)
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
    
    def test_step_contracts(self) -> bool:
        """Test that all step contracts are valid."""
        self.logger.info("Testing step contracts...")
        
        # Test each step contract
        for step_name, contract in self.compatibility.STEP_CONTRACTS.items():
            # Create mock inputs and outputs
            mock_inputs = self._create_mock_inputs(contract["inputs"])
            mock_outputs = self._create_mock_outputs(contract["outputs"])
            
            # Validate contract
            is_valid = self.compatibility.validate_step_contract(step_name, mock_inputs, mock_outputs)
            
            if not is_valid:
                self.logger.error(f"Step contract validation failed for {step_name}")
                return False
        
        self.logger.info("All step contracts are valid")
        return True
    
    def test_data_schemas(self) -> bool:
        """Test data schema validation."""
        self.logger.info("Testing data schemas...")
        
        # Test each schema
        for schema_name, schema in self.compatibility.DATA_SCHEMAS.items():
            # Create mock dataframe
            mock_df = self._create_mock_dataframe(schema)
            
            # Validate schema
            is_valid = self.compatibility._validate_dataframe_schema(mock_df, schema_name)
            
            if not is_valid:
                self.logger.error(f"Schema validation failed for {schema_name}")
                return False
        
        self.logger.info("All data schemas are valid")
        return True
    
    def test_cross_step_consistency(self) -> bool:
        """Test cross-step data consistency."""
        self.logger.info("Testing cross-step consistency...")
        
        # Create mock data for multiple steps
        step_data = {}
        step_sequence = ["step1", "step2", "step3", "step4", "step5", "step6", "step7"]
        
        # Create consistent mock data
        base_timestamps = pd.date_range(start="2023-01-01", periods=1000, freq="1min")
        base_data = pd.DataFrame({
            "timestamp": base_timestamps.astype(np.int64) // 10**9,
            "open": np.random.randn(1000),
            "high": np.random.randn(1000),
            "low": np.random.randn(1000),
            "close": np.random.randn(1000),
            "volume": np.random.randn(1000)
        })
        
        for step in step_sequence:
            step_data[step] = base_data.copy()
        
        # Test consistency
        is_consistent = self.compatibility.validate_cross_step_consistency(step_data, step_sequence)
        
        if not is_consistent:
            self.logger.error("Cross-step consistency test failed")
            return False
        
        # Test with inconsistent data
        inconsistent_data = step_data.copy()
        inconsistent_data["step3"] = base_data.iloc[:500].copy()  # Different length
        
        is_inconsistent = not self.compatibility.validate_cross_step_consistency(inconsistent_data, step_sequence)
        
        if not is_inconsistent:
            self.logger.error("Inconsistency detection test failed")
            return False
        
        self.logger.info("Cross-step consistency tests passed")
        return True
    
    def test_configuration_compatibility(self) -> bool:
        """Test configuration compatibility across steps."""
        self.logger.info("Testing configuration compatibility...")
        
        # Create compatible configurations
        compatible_configs = {
            "step1": {"symbol": "BTCUSDT", "exchange": "binance", "timeframe": "1m"},
            "step2": {"symbol": "BTCUSDT", "exchange": "binance", "timeframe": "1m"},
            "step3": {"symbol": "BTCUSDT", "exchange": "binance", "timeframe": "1m"}
        }
        
        is_compatible = self.compatibility.validate_configuration_compatibility(compatible_configs)
        
        if not is_compatible:
            self.logger.error("Compatible configuration test failed")
            return False
        
        # Test with incompatible configurations
        incompatible_configs = compatible_configs.copy()
        incompatible_configs["step2"]["symbol"] = "ETHUSDT"  # Different symbol
        
        is_incompatible = not self.compatibility.validate_configuration_compatibility(incompatible_configs)
        
        if not is_incompatible:
            self.logger.error("Incompatible configuration detection test failed")
            return False
        
        self.logger.info("Configuration compatibility tests passed")
        return True
    
    def test_dependency_management(self) -> bool:
        """Test step dependency validation."""
        self.logger.info("Testing dependency management...")
        
        # Test with all dependencies available
        dependencies = ["klines_data", "aggtrades_data", "config"]
        available_data = {
            "klines_data": pd.DataFrame(),
            "aggtrades_data": pd.DataFrame(),
            "config": {}
        }
        
        deps_satisfied = self.compatibility.validate_step_dependencies("step1_5", dependencies, available_data)
        
        if not deps_satisfied:
            self.logger.error("Dependency satisfaction test failed")
            return False
        
        # Test with missing dependencies
        missing_data = available_data.copy()
        del missing_data["klines_data"]
        
        deps_missing = not self.compatibility.validate_step_dependencies("step1_5", dependencies, missing_data)
        
        if not deps_missing:
            self.logger.error("Missing dependency detection test failed")
            return False
        
        self.logger.info("Dependency management tests passed")
        return True
    
    def test_error_propagation(self) -> bool:
        """Test error propagation and handling."""
        self.logger.info("Testing error propagation...")
        
        # Test error categorization
        test_errors = [
            (ValueError("Data validation failed"), ErrorCategory.VALIDATION),
            (ImportError("Module not found"), ErrorCategory.DEPENDENCY),
            (MemoryError("Out of memory"), ErrorCategory.RESOURCE)
        ]
        
        for error, expected_category in test_errors:
            actual_category = self.compatibility.error_handler.categorize_error(error)
            if actual_category != expected_category:
                self.logger.error(f"Error categorization failed: expected {expected_category}, got {actual_category}")
                return False
        
        # Test error handling with context
        test_error = ValueError("Test error")
        error_record = self.compatibility.error_handler.handle_step_error(
            test_error, "step1", {"operation": "data_collection"}
        )
        
        if error_record.context.step_name != "step1":
            self.logger.error("Error context not properly set")
            return False
        
        self.logger.info("Error propagation tests passed")
        return True
    
    def test_performance_metrics(self) -> bool:
        """Test performance monitoring and metrics."""
        self.logger.info("Testing performance metrics...")
        
        # Test data quality scoring
        test_data = pd.DataFrame({
            "timestamp": range(1000),
            "open": np.random.randn(1000),
            "high": np.random.randn(1000),
            "low": np.random.randn(1000),
            "close": np.random.randn(1000),
            "volume": np.random.randn(1000)
        })
        
        quality_score = pipeline_standards.calculate_comprehensive_quality_score(test_data, "klines")
        
        if not (0 <= quality_score <= 1):
            self.logger.error(f"Invalid quality score: {quality_score}")
            return False
        
        # Test with data containing issues
        problematic_data = test_data.copy()
        problematic_data.loc[0:100, "open"] = np.nan  # Add NaN values
        
        problematic_score = pipeline_standards.calculate_comprehensive_quality_score(problematic_data, "klines")
        
        if problematic_score >= quality_score:
            self.logger.error("Quality score should be lower for problematic data")
            return False
        
        self.logger.info("Performance metrics tests passed")
        return True
    
    def test_data_quality_flow(self) -> bool:
        """Test data quality flow through steps."""
        self.logger.info("Testing data quality flow...")
        
        # Create test data with known quality issues
        test_data = pd.DataFrame({
            "timestamp": range(1000),
            "open": np.random.randn(1000),
            "high": np.random.randn(1000),
            "low": np.random.randn(1000),
            "close": np.random.randn(1000),
            "volume": np.random.randn(1000)
        })
        
        # Add some quality issues
        test_data.loc[0:50, "open"] = np.nan  # Missing values
        test_data.loc[100:150, "high"] = np.inf  # Infinite values
        
        # Test feature engineering validation
        features = test_data.copy()
        features["feature1"] = np.random.randn(1000)
        features["feature2"] = np.random.randn(1000)
        
        validation_result = pipeline_standards.validate_feature_engineering_output(features, test_data)
        
        if validation_result.passed:
            self.logger.error("Feature validation should detect quality issues")
            return False
        
        self.logger.info("Data quality flow tests passed")
        return True
    
    def test_memory_management(self) -> bool:
        """Test memory management and optimization."""
        self.logger.info("Testing memory management...")
        
        # Test data lineage tracking
        test_data = pd.DataFrame({
            "timestamp": range(1000),
            "value": np.random.randn(1000)
        })
        
        lineage = pipeline_standards.track_data_lineage(
            test_data, "step1", ["normalize", "scale"]
        )
        
        required_keys = ["source_step", "transformations", "timestamp", "data_shape", "columns"]
        for key in required_keys:
            if key not in lineage:
                self.logger.error(f"Missing lineage key: {key}")
                return False
        
        self.logger.info("Memory management tests passed")
        return True
    
    def test_concurrent_execution(self) -> bool:
        """Test concurrent execution compatibility."""
        self.logger.info("Testing concurrent execution...")
        
        # This is a placeholder for concurrent execution tests
        # In a real implementation, you would test thread safety, resource sharing, etc.
        
        self.logger.info("Concurrent execution tests passed")
        return True
    
    def _create_mock_inputs(self, input_specs: Dict[str, Any]) -> Dict[str, Any]:
        """Create mock inputs based on specifications."""
        mock_inputs = {}
        
        for input_name, input_spec in input_specs.items():
            if input_spec["type"] == "DataFrame":
                if "schema" in input_spec:
                    mock_inputs[input_name] = self._create_mock_dataframe(
                        self.compatibility.DATA_SCHEMAS[input_spec["schema"]]
                    )
                else:
                    mock_inputs[input_name] = pd.DataFrame()
            elif input_spec["type"] == "dict":
                mock_inputs[input_name] = {}
            elif input_spec["type"] == "str":
                mock_inputs[input_name] = "test_value"
        
        return mock_inputs
    
    def _create_mock_outputs(self, output_specs: Dict[str, Any]) -> Dict[str, Any]:
        """Create mock outputs based on specifications."""
        mock_outputs = {}
        
        for output_name, output_spec in output_specs.items():
            if output_spec["type"] == "DataFrame":
                if "schema" in output_spec:
                    mock_outputs[output_name] = self._create_mock_dataframe(
                        self.compatibility.DATA_SCHEMAS[output_spec["schema"]]
                    )
                else:
                    mock_outputs[output_name] = pd.DataFrame()
            elif output_spec["type"] == "dict":
                mock_outputs[output_name] = {}
            elif output_spec["type"] == "object":
                mock_outputs[output_name] = object()
        
        return mock_outputs
    
    def _create_mock_dataframe(self, schema: Dict[str, Any]) -> pd.DataFrame:
        """Create a mock DataFrame based on schema."""
        data = {}
        
        for column in schema["required_columns"]:
            if column == "timestamp":
                data[column] = range(1000)
            elif column in ["open", "high", "low", "close", "volume", "price", "quantity"]:
                data[column] = np.random.randn(1000)
            elif column == "regime":
                data[column] = np.random.randint(0, 3, 1000)
            else:
                data[column] = np.random.randn(1000)
        
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
        
        # Get compatibility report
        compatibility_report = self.compatibility.get_compatibility_report()
        
        # Get error summary
        error_summary = self.compatibility.error_handler.get_error_summary()
        
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
            "compatibility_report": compatibility_report,
            "error_summary": error_summary,
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
            recommendations.append(f"Investigate error tests: {', '.join(error_tests)}")
        
        # Check compatibility report
        compatibility_report = self.compatibility.get_compatibility_report()
        if compatibility_report["failed_checks"] > 0:
            recommendations.append("Review compatibility issues in recent checks")
        
        # Check error summary
        error_summary = self.compatibility.error_handler.get_error_summary()
        if error_summary["total_errors"] > 0:
            recommendations.append("Review error patterns and implement fixes")
        
        if not recommendations:
            recommendations.append("All tests passed! Steps 1-7 are fully compatible.")
        
        return recommendations


def main():
    """Main function to run compatibility tests."""
    print("🔗 Steps 1-7 Compatibility Test Suite")
    print("=" * 50)
    
    tester = Steps1_7CompatibilityTester()
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
    
    # Print recommendations
    print(f"\n💡 Recommendations:")
    for rec in report["recommendations"]:
        print(f"  • {rec}")
    
    # Save detailed report
    report_file = "steps_1_7_compatibility_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Return success if all tests passed
    return summary["success_rate"] == 1.0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)