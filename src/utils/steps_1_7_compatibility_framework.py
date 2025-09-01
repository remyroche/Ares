"""
Steps 1 - 7 Compatibility Framework

This module provides comprehensive compatibility management between steps 1 - 7 including:
    pass - Data schema validation across steps - Input / output contract validation - Step dependency management - Cross - step data consistency checks - Configuration compatibility validation - Error propagation handling
"""

import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from .pipeline_standards import PipelineStandards, pipeline_standards
from .logger import system_logger
from .error_handler import handle_errors
from .standardized_error_handler import standardized_error_handler, ErrorCategory, ErrorSeverity

import class StepContract:
class StepContract:
    """Defines the input / output contract for each step."""

    def __init__(self, step_name: str, inputs: Dict[str, Any], outputs: Dict[str, Any]):
    pass
    pass
        self.step_name, step_name
        self.inputs, inputs
        self.outputs, outputs
        self.timestamp, datetime.now().isoformat()

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
    pass
    pass
        """Initialize the compatibility framework."""
        self.standards, pipeline_standards
        self.logger, system_logger.getChild("Steps1_7Compatibility")
        self.error_handler, standardized_error_handler
        self.compatibility_history: List[Dict[str, Any]] = []

    @handle_errors(
        exceptions=(Exception,),
        default_return = False,
        context="step contract validation"
    )
    def validate_step_contract(
        self,
        step_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any]
    ) -> bool:
        """Validate that a step's inputs and outputs match its contract.

        Args:
            step_name: Name of the step
            inputs: Step inputs
            outputs: Step outputs

        Returns:
            bool: True if contract is valid
        """
        if step_name not in self.STEP_CONTRACTS:
    pass
    pass
        self.logger.error(f"Unknown step: {step_name}")
        return False

        contract, self.STEP_CONTRACTS[step_name]
        validation_result, True

        # Validate inputs
        for input_name, input_spec in contract["inputs"].items():
    pass
    pass
        if input_spec["required"] and input_name not in inputs:
    pass
    pass
        self.logger.error(f"Missing required input '{input_name}' for {step_name}")
                validation_result, False
            elif input_name in inputs:
        # Validate input type and schema
        if not self._validate_input(input_name, inputs[input_name], input_spec):
    pass
    pass
                    validation_result, False

        # Validate outputs
        for output_name, output_spec in contract["outputs"].items():
    pass
    pass
        if output_spec["required"] and output_name not in outputs:
    pass
    pass
        self.logger.error(f"Missing required output '{output_name}' for {step_name}")
                validation_result, False
            elif output_name in outputs:
        # Validate output type and schema
        if not self._validate_output(output_name, outputs[output_name], output_spec):
    pass
    pass
                    validation_result, False

        # Record validation result
        self._record_compatibility_check(step_name, "contract_validation", validation_result)

        return validation_result

    def _validate_input(self, input_name: str, input_value: Any, input_spec: Dict[str, Any]) -> bool:
    pass
    pass
        """Validate a single input against its specification."""
        try:
        # Type validation
    except Exception as e:
        pass
    except Exception as e:
        pass
        if input_spec["type"] == "DataFrame" and not isinstance(input_value, pd.DataFrame):
    pass
    pass
        self.logger.error(f"Input '{input_name}' must be a DataFrame")
        return False

        # Schema validation for DataFrames
        if input_spec["type"] == "DataFrame" and "schema" in input_spec:
    pass
    pass
                schema_name, input_spec["schema"]
        if not self._validate_dataframe_schema(input_value, schema_name):
    pass
    pass
        return False

        return True
        except Exception as e:
        self.logger.error(f"Error validating input '{input_name}': {e}")
        return False

    def _validate_output(self, output_name: str, output_value: Any, output_spec: Dict[str, Any]) -> bool:
    pass
    pass
        """Validate a single output against its specification."""
        try:
        # Type validation
    except Exception as e:
        pass
    except Exception as e:
        pass
        if output_spec["type"] == "DataFrame" and not isinstance(output_value, pd.DataFrame):
    pass
    pass
        self.logger.error(f"Output '{output_name}' must be a DataFrame")
        return False

        # Schema validation for DataFrames
        if output_spec["type"] == "DataFrame" and "schema" in output_spec:
    pass
    pass
                schema_name, output_spec["schema"]
        if not self._validate_dataframe_schema(output_value, schema_name):
    pass
    pass
        return False

        return True
        except Exception as e:
        self.logger.error(f"Error validating output '{output_name}': {e}")
        return False

    def _validate_dataframe_schema(self, df: pd.DataFrame, schema_name: str) -> bool:
    pass
    pass
        """Validate a DataFrame against a schema."""
        if schema_name not in self.DATA_SCHEMAS:
    pass
    pass
        self.logger.error(f"Unknown schema: {schema_name}")
        return False

        schema, self.DATA_SCHEMAS[schema_name]

        # Check required columns
        missing_columns, set(schema["required_columns"]) - set(df.columns)
        if missing_columns:
    pass
    pass
        self.logger.error(f"Missing required columns for schema '{schema_name}': {missing_columns}")
        return False

        # Check data types for required columns
        for column, expected_type in schema["data_types"].items():
    pass
    pass
        if column in df.columns:
    pass
    pass
                actual_type, str(df[column].dtype)
        if actual_type != expected_type:
    pass
    pass
        self.logger.warning(f"Column '{column}' has type {actual_type}, expected {expected_type}")

        return True

    @handle_errors(
        exceptions=(Exception,),
        default_return = False,
        context="cross - step data consistency validation"
    )
    def validate_cross_step_consistency(
        self,
        step_data: Dict[str, pd.DataFrame],
        step_sequence: List[str]
    ) -> bool:
        """Validate data consistency across multiple steps.

        Args:
            step_data: Dictionary of dataframes from different steps
            step_sequence: Sequence of steps to validate

        Returns:
            bool: True if data is consistent across steps
        """
        if len(step_data) < 2:
    pass
    pass
        return True

        # Get reference dataframe (first step with data)
        reference_df, None
        reference_step, None
        for step in step_sequence:
    pass
    pass
        if step in step_data and step_data[step] is not None and len(step_data[step]) > 0:
    pass
    pass
                reference_df, step_data[step]
                reference_step, step
                break

        if reference_df is None:
    pass
    pass
        self.logger.error("No reference dataframe found for consistency validation")
        return False

        reference_length, len(reference_df)
        reference_timestamps, set(reference_df["timestamp"].values) if "timestamp" in reference_df.columns else set()

        consistency_issues = []

        # Check each step's data consistency
        for step in step_sequence:
    pass
    pass
        if step not in step_data or step_data[step] is None:
    pass
    pass
                continue

            df, step_data[step]

        # Check row count consistency
        if len(df) != reference_length:
    pass
    pass
                consistency_issues.append(f"Row count mismatch in {step}: {len(df)} vs {reference_length}")

        # Check timestamp consistency if available
        if "timestamp" in df.columns and reference_timestamps:
    pass
    pass
                df_timestamps, set(df["timestamp"].values)
        if df_timestamps != reference_timestamps:
    pass
    pass
                    missing_timestamps, reference_timestamps - df_timestamps
                    extra_timestamps, df_timestamps - reference_timestamps
        if missing_timestamps or extra_timestamps:
    pass
    pass
                        consistency_issues.append(
                            f"Timestamp mismatch in {step}: missing={len(missing_timestamps)}, extra={len(extra_timestamps)}"
                        )

        if consistency_issues:
    pass
    pass
        for issue in consistency_issues:
    pass
    pass
        self.logger.warning(issue)
        return False

        self.logger.info("Cross - step data consistency validation passed")
        return True

    @handle_errors(
        exceptions=(Exception,),
        default_return = False,
        context="configuration compatibility validation"
    )
    def validate_configuration_compatibility(
        self,
        configs: Dict[str, Dict[str, Any]]
    ) -> bool:
        """Validate that configurations are compatible across steps.

        Args:
            configs: Dictionary of configurations for each step

        Returns:
            bool: True if configurations are compatible
        """
        if len(configs) < 2:
    pass
    pass
        return True

        # Extract common configuration parameters
        common_params = ["symbol", "exchange", "timeframe", "lookback_years"]
        compatibility_issues = []

        # Check common parameters across all configs
        for param in common_params:
    pass
    pass
            values, set()
        for step, config in configs.items():
    pass
    pass
        if param in config:
    pass
    pass
                    values.add(str(config[param]))

        if len(values) > 1:
    pass
    pass
                compatibility_issues.append(f"Parameter '{param}' has different values across steps: {values}")

        # Check for conflicting parameters
        conflicting_params = {
            "data_source": ["binance", "kucoin"],
            "timeframe": ["1m", "5m", "15m", "1h", "4h", "1d"]
        }

        for param, allowed_values in conflicting_params.items():
    pass
    pass
        for step, config in configs.items():
    pass
    pass
        if param in config and config[param] not in allowed_values:
    pass
    pass
                    compatibility_issues.append(f"Invalid value for '{param}' in {step}: {config[param]}")

        if compatibility_issues:
    pass
    pass
        for issue in compatibility_issues:
    pass
    pass
        self.logger.error(issue)
        return False

        self.logger.info("Configuration compatibility validation passed")
        return True

    @handle_errors(
        exceptions=(Exception,),
        default_return = False,
        context="step dependency validation"
    )
    def validate_step_dependencies(
        self,
        step_name: str,
        dependencies: List[str],
        available_data: Dict[str, Any]
    ) -> bool:
        """Validate that all dependencies for a step are available.

        Args:
            step_name: Name of the step
            dependencies: List of required dependencies
            available_data: Available data from previous steps

        Returns:
            bool: True if all dependencies are satisfied
        """
        missing_dependencies = []

        for dependency in dependencies:
    pass
    pass
        if dependency not in available_data or available_data[dependency] is None:
    pass
    pass
                missing_dependencies.append(dependency)

        if missing_dependencies:
    pass
    pass
        self.logger.error(f"Missing dependencies for {step_name}: {missing_dependencies}")
        return False

        self.logger.info(f"All dependencies satisfied for {step_name}")
        return True

    def _record_compatibility_check(
        self,
        step_name: str,
        check_type: str,
        result: bool,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
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
    pass
    pass
        self.compatibility_history, self.compatibility_history[-500:]

    def get_compatibility_report(self, step_name: Optional[str] = None) -> Dict[str, Any]:
    pass
    pass
        """Get a compatibility report.

        Args:
            step_name: Optional step name to filter by

        Returns:
            Dict: Compatibility report
        """
        if step_name:
    pass
    pass
            filtered_history = [h for h in self.compatibility_history if h["step_name"] == step_name]
        else:
            filtered_history, self.compatibility_history

        report = {
            "total_checks": len(filtered_history),
            "passed_checks": len([h for h in filtered_history if h["result"]]),
            "failed_checks": len([h for h in filtered_history if not h["result"]]),
            "by_check_type": {},
            "by_step": {},
            "recent_issues": []
        }

        for check in filtered_history:
    pass
    pass
        # Count by check type
            check_type, check["check_type"]
            report["by_check_type"][check_type] = report["by_check_type"].get(check_type, 0) + 1

        # Count by step
            step, check["step_name"]
            report["by_step"][step] = report["by_step"].get(step, 0) + 1

        # Get recent failed checks
        recent_failures = [h for h in filtered_history[-10:] if not h["result"]]
        report["recent_issues"] = recent_failures

        return report

    def export_compatibility_report(self, file_path: str) -> bool:
    pass
    pass
        """Export compatibility report to file.

        Args:
            file_path: Path to export file

        Returns:
            bool: True if successful
        """
        try:
            report, self.get_compatibility_report()
    except Exception as e:
        pass
    except Exception as e:
        pass
        with open(file_path, 'w') as f:
                json.dump(report, f, indent = 2)
        return True
        except Exception as e:
        self.logger.error(f"Failed to export compatibility report: {e}")
        return False

# Global instance
steps_1_7_compatibility, Steps1_7CompatibilityFramework()