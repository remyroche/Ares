"""
Steps 1 - 7 Compatibility Framework

This module provides comprehensive compatibility management between steps 1 - 7 including:
- Data schema validation across steps
- Input/output contract validation
- Step dependency management
- Cross-step data consistency checks
- Configuration compatibility validation
- Error propagation handling
"""

import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path

# Try to import pandas, but make it optional
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    # Create a mock DataFrame class for when pandas is not available
    class MockDataFrame:
        def __init__(self, data=None):
            self.data = data or {}
            self.columns = list(self.data.keys()) if self.data else []
        
        def __len__(self):
            return len(next(iter(self.data.values()))) if self.data else 0
        
        def __getitem__(self, key):
            return self.data.get(key, [])
        
        def __contains__(self, key):
            return key in self.data
        
        @property
        def dtype(self):
            return "object"
    
    # Create a mock pandas module
    class MockPandas:
        DataFrame = MockDataFrame
    
    pd = MockPandas()

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from .pipeline_standards import PipelineStandards, pipeline_standards
from .logger import system_logger
from .error_handler import handle_errors
from .standardized_error_handler import standardized_error_handler, ErrorCategory, ErrorSeverity


class StepContract:
    """Defines the input/output contract for each step."""
    
    def __init__(self, step_name: str, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        self.step_name = step_name
        self.inputs = inputs
        self.outputs = outputs
        self.timestamp = datetime.now().isoformat()
    
    def validate_inputs(self, actual_inputs: Dict[str, Any]) -> bool:
        """Validate that actual inputs match the contract."""
        for input_name, input_spec in self.inputs.items():
            if input_spec.get("required", False) and input_name not in actual_inputs:
                return False
        return True
    
    def validate_outputs(self, actual_outputs: Dict[str, Any]) -> bool:
        """Validate that actual outputs match the contract."""
        for output_name, output_spec in self.outputs.items():
            if output_spec.get("required", False) and output_name not in actual_outputs:
                return False
        return True


class Steps1_7CompatibilityFramework:
    """Comprehensive compatibility framework for steps 1 - 7."""
    
    # Step definitions and their contracts
    STEP_CONTRACTS = {
        "step01_data_collection": {
            "inputs": {
                "config": {"type": "dict", "required": True},
                "symbol": {"type": "str", "required": True},
                "exchange": {"type": "str", "required": True},
                "timeframe": {"type": "str", "required": True}
            },
            "outputs": {
                "klines_data": {"type": "DataFrame", "required": True, "schema": "klines"},
                "aggtrades_data": {"type": "DataFrame", "required": True, "schema": "aggtrades"},
                "data_paths": {"type": "dict", "required": True},
                "metadata": {"type": "dict", "required": True}
            }
        },
        "step01_5_data_converter": {
            "inputs": {
                "klines_data": {"type": "DataFrame", "required": True, "schema": "klines"},
                "aggtrades_data": {"type": "DataFrame", "required": True, "schema": "aggtrades"},
                "config": {"type": "dict", "required": True}
            },
            "outputs": {
                "unified_data": {"type": "DataFrame", "required": True, "schema": "unified"},
                "conversion_metadata": {"type": "dict", "required": True}
            }
        },
        "step02_data_reading": {
            "inputs": {
                "unified_data": {"type": "DataFrame", "required": True, "schema": "unified"},
                "config": {"type": "dict", "required": True}
            },
            "outputs": {
                "validated_data": {"type": "DataFrame", "required": True, "schema": "unified"},
                "validation_report": {"type": "dict", "required": True},
                "quality_metrics": {"type": "dict", "required": True}
            }
        },
        "step03_hmm_regime_discovery": {
            "inputs": {
                "validated_data": {"type": "DataFrame", "required": True, "schema": "unified"},
                "config": {"type": "dict", "required": True}
            },
            "outputs": {
                "regime_labels": {"type": "DataFrame", "required": True, "schema": "regime_labels"},
                "hmm_model": {"type": "object", "required": True},
                "regime_metadata": {"type": "dict", "required": True}
            }
        },
        "step04_regime_data_splitting": {
            "inputs": {
                "validated_data": {"type": "DataFrame", "required": True, "schema": "unified"},
                "regime_labels": {"type": "DataFrame", "required": True, "schema": "regime_labels"},
                "config": {"type": "dict", "required": True}
            },
            "outputs": {
                "regime_datasets": {"type": "dict", "required": True},
                "splitting_metadata": {"type": "dict", "required": True}
            }
        },
        "step05_labeling": {
            "inputs": {
                "regime_datasets": {"type": "dict", "required": True},
                "config": {"type": "dict", "required": True}
            },
            "outputs": {
                "labeled_datasets": {"type": "dict", "required": True},
                "labeling_metadata": {"type": "dict", "required": True}
            }
        },
        "step06_feature_engineering": {
            "inputs": {
                "labeled_datasets": {"type": "dict", "required": True},
                "config": {"type": "dict", "required": True}
            },
            "outputs": {
                "feature_datasets": {"type": "dict", "required": True},
                "feature_metadata": {"type": "dict", "required": True},
                "feature_importance": {"type": "dict", "required": True}
            }
        },
        "step07_enhanced_matrix_operations": {
            "inputs": {
                "feature_datasets": {"type": "dict", "required": True},
                "config": {"type": "dict", "required": True}
            },
            "outputs": {
                "processed_datasets": {"type": "dict", "required": True},
                "processing_metadata": {"type": "dict", "required": True},
                "matrix_operations": {"type": "dict", "required": True}
            }
        }
    }

    # Data schemas for validation
    DATA_SCHEMAS = {
        "klines": {
            "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
            "optional_columns": ["quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"],
            "data_types": {
                "timestamp": "int64",
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64"
            }
        },
        "aggtrades": {
            "required_columns": ["timestamp", "price", "quantity"],
            "optional_columns": ["first_trade_id", "last_trade_id", "trade_time", "is_buyer_maker"],
            "data_types": {
                "timestamp": "int64",
                "price": "float64",
                "quantity": "float64"
            }
        },
        "unified": {
            "required_columns": ["timestamp", "open", "high", "low", "close", "volume", "price", "quantity"],
            "optional_columns": ["regime", "label", "split"],
            "data_types": {
                "timestamp": "int64",
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64",
                "price": "float64",
                "quantity": "float64"
            }
        },
        "regime_labels": {
            "required_columns": ["timestamp", "regime"],
            "optional_columns": ["regime_probability", "regime_confidence"],
            "data_types": {
                "timestamp": "int64",
                "regime": "int64"
            }
        }
    }

    def __init__(self):
        """Initialize the compatibility framework."""
        self.standards = pipeline_standards
        self.logger = system_logger.getChild("Steps1_7Compatibility")
        self.error_handler = standardized_error_handler
        self.compatibility_history: List[Dict[str, Any]] = []
        
        if not PANDAS_AVAILABLE:
            self.logger.warning("Pandas not available - using mock DataFrame implementation")

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step contract validation"
    )
    def validate_step_contract(self, step_name: str, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        """Validate that step inputs and outputs match the contract."""
        if step_name not in self.STEP_CONTRACTS:
            self.logger.error(f"Unknown step: {step_name}")
            return False

        contract = self.STEP_CONTRACTS[step_name]
        validation_result = True

        # Validate inputs
        for input_name, input_spec in contract["inputs"].items():
            if input_spec["required"] and input_name not in inputs:
                self.logger.error(f"Missing required input '{input_name}' for {step_name}")
                validation_result = False
            elif input_name in inputs:
                # Validate input type and schema
                if not self._validate_input(input_name, inputs[input_name], input_spec):
                    validation_result = False

        # Validate outputs
        for output_name, output_spec in contract["outputs"].items():
            if output_spec["required"] and output_name not in outputs:
                self.logger.error(f"Missing required output '{output_name}' for {step_name}")
                validation_result = False
            elif output_name in outputs:
                # Validate output type and schema
                if not self._validate_output(output_name, outputs[output_name], output_spec):
                    validation_result = False

        # Record validation result
        self._record_compatibility_check(step_name, "contract_validation", validation_result)

        return validation_result

    def _validate_input(self, input_name: str, input_value: Any, input_spec: Dict[str, Any]) -> bool:
        """Validate a single input against its specification."""
        try:
            # Type validation
            if input_spec["type"] == "DataFrame" and not isinstance(input_value, pd.DataFrame):
                self.logger.error(f"Input '{input_name}' must be a DataFrame")
                return False

            # Schema validation for DataFrames
            if input_spec["type"] == "DataFrame" and "schema" in input_spec:
                schema_name = input_spec["schema"]
                if not self._validate_dataframe_schema(input_value, schema_name):
                    return False

            return True
        except Exception as e:
            self.logger.error(f"Error validating input '{input_name}': {e}")
            return False

    def _validate_output(self, output_name: str, output_value: Any, output_spec: Dict[str, Any]) -> bool:
        """Validate a single output against its specification."""
        try:
            # Type validation
            if output_spec["type"] == "DataFrame" and not isinstance(output_value, pd.DataFrame):
                self.logger.error(f"Output '{output_name}' must be a DataFrame")
                return False

            # Schema validation for DataFrames
            if output_spec["type"] == "DataFrame" and "schema" in output_spec:
                schema_name = output_spec["schema"]
                if not self._validate_dataframe_schema(output_value, schema_name):
                    return False

            return True
        except Exception as e:
            self.logger.error(f"Error validating output '{output_name}': {e}")
            return False

    def _validate_dataframe_schema(self, df: pd.DataFrame, schema_name: str) -> bool:
        """Validate a DataFrame against a specific schema."""
        if schema_name not in self.DATA_SCHEMAS:
            self.logger.error(f"Unknown schema: {schema_name}")
            return False

        schema = self.DATA_SCHEMAS[schema_name]

        # Check required columns
        missing_columns = set(schema["required_columns"]) - set(df.columns)
        if missing_columns:
            self.logger.error(f"Missing required columns for schema '{schema_name}': {missing_columns}")
            return False

        # Check data types for required columns (only if pandas is available)
        if PANDAS_AVAILABLE:
            for column, expected_type in schema["data_types"].items():
                if column in df.columns:
                    actual_type = str(df[column].dtype)
                    if actual_type != expected_type:
                        self.logger.warning(f"Column '{column}' has type {actual_type}, expected {expected_type}")

        return True

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="cross-step data consistency validation"
    )
    def validate_cross_step_consistency(self, step_sequence: List[str], step_data: Dict[str, pd.DataFrame]) -> bool:
        """Validate data consistency across multiple steps."""
        if len(step_data) < 2:
            return True

        # Get reference dataframe (first step with data)
        reference_df = None
        reference_step = None
        for step in step_sequence:
            if step in step_data and step_data[step] is not None and len(step_data[step]) > 0:
                reference_df = step_data[step]
                reference_step = step
                break

        if reference_df is None:
            self.logger.error("No reference dataframe found for consistency validation")
            return False

        reference_length = len(reference_df)
        reference_timestamps = set(reference_df["timestamp"].values) if "timestamp" in reference_df.columns else set()

        consistency_issues = []

        # Check each step's data consistency
        for step in step_sequence:
            if step not in step_data or step_data[step] is None:
                continue

            df = step_data[step]

            # Check row count consistency
            if len(df) != reference_length:
                consistency_issues.append(f"Row count mismatch in {step}: {len(df)} vs {reference_length}")

            # Check timestamp consistency if available
            if "timestamp" in df.columns and reference_timestamps:
                df_timestamps = set(df["timestamp"].values)
                if df_timestamps != reference_timestamps:
                    missing_timestamps = reference_timestamps - df_timestamps
                    extra_timestamps = df_timestamps - reference_timestamps
                    if missing_timestamps or extra_timestamps:
                        consistency_issues.append(
                            f"Timestamp mismatch in {step}: missing={len(missing_timestamps)}, extra={len(extra_timestamps)}"
                        )

        if consistency_issues:
            for issue in consistency_issues:
                self.logger.warning(issue)
            return False

        self.logger.info("Cross-step data consistency validation passed")
        return True

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="configuration compatibility validation"
    )
    def validate_configuration_compatibility(self, configs: Dict[str, Dict[str, Any]]) -> bool:
        """Validate that configurations across steps are compatible."""
        if len(configs) < 2:
            return True

        # Extract common configuration parameters
        common_params = ["symbol", "exchange", "timeframe", "lookback_years"]
        compatibility_issues = []

        # Check common parameters across all configs
        for param in common_params:
            values = set()
            for step, config in configs.items():
                if param in config:
                    values.add(str(config[param]))

            if len(values) > 1:
                compatibility_issues.append(f"Parameter '{param}' has different values across steps: {values}")

        # Check for conflicting parameters
        conflicting_params = {
            "data_source": ["binance", "kucoin"],
            "timeframe": ["1m", "5m", "15m", "1h", "4h", "1d"]
        }

        for param, allowed_values in conflicting_params.items():
            for step, config in configs.items():
                if param in config and config[param] not in allowed_values:
                    compatibility_issues.append(f"Invalid value for '{param}' in {step}: {config[param]}")

        if compatibility_issues:
            for issue in compatibility_issues:
                self.logger.error(issue)
            return False

        self.logger.info("Configuration compatibility validation passed")
        return True

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step dependency validation"
    )
    def validate_step_dependencies(self, step_name: str, dependencies: List[str], available_data: Dict[str, Any]) -> bool:
        """Validate that all required dependencies are available."""
        missing_dependencies = []

        for dependency in dependencies:
            if dependency not in available_data or available_data[dependency] is None:
                missing_dependencies.append(dependency)

        if missing_dependencies:
            self.logger.error(f"Missing dependencies for {step_name}: {missing_dependencies}")
            return False

        self.logger.info(f"All dependencies satisfied for {step_name}")
        return True

    def _record_compatibility_check(self, step_name: str, check_type: str, result: bool, details: Optional[Dict[str, Any]] = None) -> None:
        """Record a compatibility check result."""
        check_record = {
            "step_name": step_name,
            "check_type": check_type,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }

        self.compatibility_history.append(check_record)

        # Keep history manageable
        if len(self.compatibility_history) > 1000:
            self.compatibility_history = self.compatibility_history[-500:]

    def get_compatibility_report(self, step_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate a compatibility report."""
        if step_name:
            filtered_history = [h for h in self.compatibility_history if h["step_name"] == step_name]
        else:
            filtered_history = self.compatibility_history

        report = {
            "total_checks": len(filtered_history),
            "passed_checks": len([h for h in filtered_history if h["result"]]),
            "failed_checks": len([h for h in filtered_history if not h["result"]]),
            "by_check_type": {},
            "by_step": {},
            "recent_issues": []
        }

        for check in filtered_history:
            # Count by check type
            check_type = check["check_type"]
            report["by_check_type"][check_type] = report["by_check_type"].get(check_type, 0) + 1

            # Count by step
            step = check["step_name"]
            report["by_step"][step] = report["by_step"].get(step, 0) + 1

        # Get recent failed checks
        recent_failures = [h for h in filtered_history[-10:] if not h["result"]]
        report["recent_issues"] = recent_failures

        return report

    def export_compatibility_report(self, file_path: str) -> bool:
        """Export compatibility report to JSON file."""
        try:
            report = self.get_compatibility_report()
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to export compatibility report: {e}")
            return False


# Global instance
steps_1_7_compatibility = Steps1_7CompatibilityFramework()