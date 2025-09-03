from __future__ import annotations
# src/training/steps/step7_enhanced_matrix_operations_validator.py

"""Validator for Step 7: Enhanced Matrix Operations."""

import json
from pathlib import Path
from typing import Any

from src.utils.base_validator import BaseValidator
from src.utils.common_operations import safe_json_load
from src.utils.logger import system_logger


class Step7EnhancedMatrixOperationsValidator(BaseValidator):
    """Validator for Step 7: Enhanced Matrix Operations."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the validator."""
        super().__init__("step7_enhanced_matrix_operations", config)
        self.logger = system_logger.getChild("Step7EnhancedMatrixOperationsValidator")

    def validate_step_prerequisites(self, symbol: str, exchange: str, timeframe: str) -> dict[str, Any]:
        """Validate prerequisites for Step 2.5."""
        validation_result = {
            "validation_passed": True,
            "warnings": [],
            "errors": [],
            "details": {},
        }

        try:
            # Check if step6_feature_engineering output exists
            step6_output_dir = Path("data/training")
            step6_files = list(step6_output_dir.glob(f"{exchange}_{symbol}_{timeframe}*features*.parquet"))

            if not step6_files:
                validation_result["validation_passed"] = False
                validation_result["errors"].append(
                    f"Step 6 feature engineering output not found for {exchange}_{symbol}_{timeframe}",
                )
            else:
                validation_result["details"]["step6_files_found"] = len(step6_files)
                validation_result["details"]["step6_files"] = [str(f) for f in step6_files]

            # Check if matrix operations directory exists
            matrix_ops_dir = Path("data/matrix_operations")
            if not matrix_ops_dir.exists():
                validation_result["warnings"].append(
                    f"Matrix operations directory does not exist: {matrix_ops_dir}",
                )

        except Exception as e:
            validation_result["validation_passed"] = False
            validation_result["errors"].append(f"Prerequisites validation failed: {str(e)}")

        return validation_result

    def validate_step_output(self, symbol: str, exchange: str, timeframe: str) -> dict[str, Any]:
        """Validate Step 2.5 output files and content."""
        validation_result = {
            "validation_passed": True,
            "warnings": [],
            "errors": [],
            "details": {},
        }

        try:
            # Define expected output files
            output_dir = Path("data/matrix_operations")
            expected_files = [
                f"{exchange}_{symbol}_{timeframe}_matrix_operations_config.json",
                f"{exchange}_{symbol}_{timeframe}_matrix_operations_results.json",
                f"{exchange}_{symbol}_{timeframe}_matrix_operations_summary.json",
            ]

            # Check if all expected files exist
            missing_files = []
            existing_files = []

            for filename in expected_files:
                file_path = output_dir / filename
                if file_path.exists():
                    existing_files.append(str(file_path))
                else:
                    missing_files.append(filename)

            if missing_files:
                validation_result["validation_passed"] = False
                validation_result["errors"].extend([
                    f"Missing matrix operations file: {f}" for f in missing_files
                ])
            else:
                validation_result["details"]["files_found"] = len(existing_files)
                validation_result["details"]["files"] = existing_files

            # Validate each file content
            if existing_files:
                file_validations = self._validate_file_contents(
                    output_dir, symbol, exchange, timeframe,
                )

                for file_validation in file_validations:
                    if not file_validation["valid"]:
                        validation_result["validation_passed"] = False
                        validation_result["errors"].extend(file_validation["errors"])

                    if file_validation["warnings"]:
                        validation_result["warnings"].extend(file_validation["warnings"])

                    validation_result["details"].update(file_validation["details"])

        except Exception as e:
            validation_result["validation_passed"] = False
            validation_result["errors"].append(f"Output validation failed: {str(e)}")

        return validation_result

    def _validate_file_contents(
        self,
        output_dir: Path,
        symbol: str,
        exchange: str,
        timeframe: str,
    ) -> list[dict[str, Any]]:
        """Validate the content of matrix operations files."""
        validations = []

        # Validate config file
        config_file = output_dir / f"{exchange}_{symbol}_{timeframe}_matrix_operations_config.json"
        if config_file.exists():
            config_validation = self._validate_config_file(config_file)
            validations.append(config_validation)

        # Validate results file
        results_file = output_dir / f"{exchange}_{symbol}_{timeframe}_matrix_operations_results.json"
        if results_file.exists():
            results_validation = self._validate_results_file(results_file)
            validations.append(results_validation)

        # Validate summary file
        summary_file = output_dir / f"{exchange}_{symbol}_{timeframe}_matrix_operations_summary.json"
        if summary_file.exists():
            summary_validation = self._validate_summary_file(summary_file)
            validations.append(summary_validation)

        return validations

    def _validate_config_file(self, config_file: Path) -> dict[str, Any]:
        """Validate matrix operations configuration file."""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "details": {},
        }

        try:
            config = safe_json_load(config_file)

            # Check required fields
            required_fields = [
                "enable_gpu_acceleration",
                "enable_sparse_optimizations",
                "enable_memory_optimization",
                "enable_parallel_processing",
                "condition_number_threshold",
                "min_eigenvalue_threshold",
                "correlation_threshold",
                "memory_threshold_gb",
                "batch_size",
                "max_iterations",
                "tolerance",
                "data_shape",
                "numeric_columns",
                "operations",
            ]

            missing_fields = [field for field in required_fields if field not in config]
            if missing_fields:
                validation["valid"] = False
                validation["errors"].append(f"Missing required fields: {missing_fields}")

            # Validate data types and ranges
            if "condition_number_threshold" in config:
                if not isinstance(config["condition_number_threshold"], int | float) or config["condition_number_threshold"] <= 0:
                    validation["valid"] = False
                    validation["errors"].append("condition_number_threshold must be a positive number")

            if "correlation_threshold" in config:
                if not isinstance(config["correlation_threshold"], int | float) or not (0 <= config["correlation_threshold"] <= 1):
                    validation["valid"] = False
                    validation["errors"].append("correlation_threshold must be between 0 and 1")

            if "operations" in config:
                expected_operations = [
                    "correlation_analysis",
                    "condition_number_check",
                    "eigenvalue_analysis",
                    "singular_value_decomposition",
                    "matrix_rank_analysis",
                ]

                if not isinstance(config["operations"], list):
                    validation["valid"] = False
                    validation["errors"].append("operations must be a list")
                else:
                    invalid_operations = [op for op in config["operations"] if op not in expected_operations]
                    if invalid_operations:
                        validation["warnings"].append(f"Unknown operations: {invalid_operations}")

            validation["details"]["config_fields"] = len(config)
            validation["details"]["operations_count"] = len(config.get("operations", []))

        except json.JSONDecodeError as e:
            validation["valid"] = False
            validation["errors"].append(f"Invalid JSON in config file: {str(e)}")
        except Exception as e:
            validation["valid"] = False
            validation["errors"].append(f"Config file validation failed: {str(e)}")

        return validation

    def _validate_results_file(self, results_file: Path) -> dict[str, Any]:
        """Validate matrix operations results file."""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "details": {},
        }

        try:
            results = safe_json_load(results_file)

            # Check if results contain expected operations
            expected_operations = [
                "correlation_analysis",
                "condition_number_check",
                "eigenvalue_analysis",
                "singular_value_decomposition",
                "matrix_rank_analysis",
            ]

            operations_found = []
            for operation in expected_operations:
                if operation in results:
                    operations_found.append(operation)

                    # Validate specific operation results
                    op_validation = self._validate_operation_results(operation, results[operation])
                    if not op_validation["valid"]:
                        validation["valid"] = False
                        validation["errors"].extend(op_validation["errors"])

                    if op_validation["warnings"]:
                        validation["warnings"].extend(op_validation["warnings"])

            if not operations_found:
                validation["valid"] = False
                validation["errors"].append("No matrix operations results found")

            validation["details"]["operations_found"] = operations_found
            validation["details"]["operations_count"] = len(operations_found)

        except json.JSONDecodeError as e:
            validation["valid"] = False
            validation["errors"].append(f"Invalid JSON in results file: {str(e)}")
        except Exception as e:
            validation["valid"] = False
            validation["errors"].append(f"Results file validation failed: {str(e)}")

        return validation

    def _validate_operation_results(self, operation: str, results: dict[str, Any]) -> dict[str, Any]:
        """Validate specific operation results."""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "details": {},
        }

        if operation == "correlation_analysis":
            if "correlation_matrix" not in results:
                validation["valid"] = False
                validation["errors"].append("correlation_analysis missing correlation_matrix")
            if "high_correlations" not in results:
                validation["warnings"].append("correlation_analysis missing high_correlations")

        elif operation == "condition_number_check":
            if "condition_number" not in results:
                validation["valid"] = False
                validation["errors"].append("condition_number_check missing condition_number")
            elif not isinstance(results["condition_number"], int | float) or results["condition_number"] <= 0:
                validation["valid"] = False
                validation["errors"].append("condition_number must be a positive number")

        elif operation == "eigenvalue_analysis":
            if "eigenvalues" not in results:
                validation["valid"] = False
                validation["errors"].append("eigenvalue_analysis missing eigenvalues")
            elif not isinstance(results["eigenvalues"], list):
                validation["valid"] = False
                validation["errors"].append("eigenvalues must be a list")

        elif operation == "singular_value_decomposition":
            if "singular_values" not in results:
                validation["valid"] = False
                validation["errors"].append("singular_value_decomposition missing singular_values")
            elif not isinstance(results["singular_values"], list):
                validation["valid"] = False
                validation["errors"].append("singular_values must be a list")

        elif operation == "matrix_rank_analysis":
            if "rank" not in results:
                validation["valid"] = False
                validation["errors"].append("matrix_rank_analysis missing rank")
            elif not isinstance(results["rank"], int) or results["rank"] < 0:
                validation["valid"] = False
                validation["errors"].append("rank must be a non-negative integer")

        return validation

    def _validate_summary_file(self, summary_file: Path) -> dict[str, Any]:
        """Validate matrix operations summary file."""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "details": {},
        }

        try:
            summary = safe_json_load(summary_file)

            # Check required fields
            required_fields = [
                "timestamp",
                "symbol",
                "exchange",
                "timeframe",
                "operations_performed",
                "data_shape",
                "numeric_columns",
            ]

            missing_fields = [field for field in required_fields if field not in summary]
            if missing_fields:
                validation["valid"] = False
                validation["errors"].append(f"Missing required fields: {missing_fields}")

            # Validate data types
            if "numeric_columns" in summary:
                if not isinstance(summary["numeric_columns"], int) or summary["numeric_columns"] < 0:
                    validation["valid"] = False
                    validation["errors"].append("numeric_columns must be a non-negative integer")

            if "operations_performed" in summary:
                if not isinstance(summary["operations_performed"], list):
                    validation["valid"] = False
                    validation["errors"].append("operations_performed must be a list")

            validation["details"]["summary_fields"] = len(summary)

        except json.JSONDecodeError as e:
            validation["valid"] = False
            validation["errors"].append(f"Invalid JSON in summary file: {str(e)}")
        except Exception as e:
            validation["valid"] = False
            validation["errors"].append(f"Summary file validation failed: {str(e)}")

        return validation

    def get_validation_summary(self, symbol: str, exchange: str, timeframe: str) -> dict[str, Any]:
        """Get a comprehensive validation summary for Step 2.5."""
        prerequisites = self.validate_step_prerequisites(symbol, exchange, timeframe)
        output = self.validate_step_output(symbol, exchange, timeframe)

        return {
            "step_name": "step7_enhanced_matrix_operations",
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "prerequisites_validation": prerequisites,
            "output_validation": output,
            "overall_validation_passed": prerequisites["validation_passed"] and output["validation_passed"],
            "total_warnings": len(prerequisites["warnings"]) + len(output["warnings"]),
            "total_errors": len(prerequisites["errors"]) + len(output["errors"]),
        }
