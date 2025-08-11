"""
Base validator class for training step validators.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from src.utils.warning_symbols import (
    failed,
    missing,
    validation_error,
    warning,
)


class BaseValidator(ABC):
    """Base class for all step validators."""

    def __init__(self, step_name: str, config: dict[str, Any]):
        self.step_name = step_name
        self.config = config
        self.logger = logging.getLogger(f"AresGlobal.{self.__class__.__name__}")
        self.validation_results = {}

    def print(self, message: str) -> None:
        """Proxy print to logger to keep output consistent in terminal."""
        self.logger.info(message)

    @abstractmethod
    async def validate(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> bool:
        """
        Validate a training step.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            bool: True if validation passed, False otherwise
        """

    def validate_error_absence(
        self,
        step_result: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        """
        Validate that the step completed without errors.

        Args:
            step_result: Step result dictionary

        Returns:
            Tuple[bool, Dict[str, Any]]: (passed, metrics)
        """
        try:
            # Check for errors in step result
            errors = step_result.get("errors", [])
            warnings = step_result.get("warnings", [])

            # Check for critical errors
            critical_errors = [
                e
                for e in errors
                if isinstance(e, dict) and e.get("severity") == "CRITICAL"
            ]

            metrics = {
                "total_errors": len(errors),
                "total_warnings": len(warnings),
                "critical_errors": len(critical_errors),
                "has_critical_errors": len(critical_errors) > 0,
                "error_messages": errors,
                "warning_messages": warnings,
            }

            # Step passes if no critical errors
            passed = len(critical_errors) == 0

            if not passed:
                self.logger.warning(
                    f"⚠️ Step {self.step_name} has {len(critical_errors)} critical errors",
                )

            return passed, metrics

        except Exception as e:
            self.print(validation_error(f"❌ Error in error absence validation: {e}"))
            return False, {"error": str(e)}

    def validate_file_exists(
        self,
        file_path: str,
        file_type: str,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Validate that a file exists.

        Args:
            file_path: Path to the file
            file_type: Type of file for logging

        Returns:
            Tuple[bool, Dict[str, Any]]: (passed, metrics)
        """
        try:
            import os

            exists = os.path.exists(file_path)

            metrics = {
                "file_path": file_path,
                "file_type": file_type,
                "exists": exists,
                "file_size": os.path.getsize(file_path) if exists else 0,
            }

            if not exists:
                self.print(missing(f"❌ {file_type} file not found: {file_path}"))

            return exists, metrics

        except Exception as e:
            self.print(validation_error(f"❌ Error in file existence validation: {e}"))
            return False, {"error": str(e)}

    def validate_data_quality(
        self,
        data: Any,
        data_name: str,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Validate basic data quality metrics.

        Args:
            data: Data to validate
            data_name: Name of the data for logging

        Returns:
            Tuple[bool, Dict[str, Any]]: (passed, metrics)
        """
        try:
            import pandas as pd

            if isinstance(data, pd.DataFrame):
                metrics = {
                    "data_type": "DataFrame",
                    "shape": data.shape,
                    "columns": list(data.columns),
                    "dtypes": data.dtypes.to_dict(),
                    "null_counts": data.isnull().sum().to_dict(),
                    "memory_usage": data.memory_usage(deep=True).sum(),
                    "has_duplicates": data.duplicated().sum(),
                    "empty": data.empty,
                }

                # Basic quality checks
                passed = not data.empty and data.shape[0] > 0 and data.shape[1] > 0

            elif isinstance(data, list | tuple):
                metrics = {
                    "data_type": type(data).__name__,
                    "length": len(data),
                    "empty": len(data) == 0,
                }
                passed = len(data) > 0

            elif isinstance(data, dict):
                metrics = {
                    "data_type": "dict",
                    "keys": list(data.keys()),
                    "length": len(data),
                    "empty": len(data) == 0,
                }
                passed = len(data) > 0

            else:
                metrics = {
                    "data_type": type(data).__name__,
                    "value": str(data)[:100],  # Truncate long values
                }
                passed = data is not None

            if not passed:
                self.print(failed(f"⚠️ {data_name} quality validation failed"))

            return passed, metrics

        except Exception as e:
            self.print(validation_error(f"❌ Error in data quality validation: {e}"))
            return False, {"error": str(e)}

    def validate_outcome_favorability(
        self,
        step_result: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        """
        Validate that the step outcome is favorable.

        Args:
            step_result: Step result dictionary

        Returns:
            Tuple[bool, Dict[str, Any]]: (passed, metrics)
        """
        try:
            # Extract outcome metrics
            metrics = step_result.get("metrics", {})
            performance = step_result.get("performance", {})

            # Check for success indicators
            success_indicators = [
                step_result.get("success", False),
                step_result.get("completed", False),
                step_result.get("status") == "SUCCESS",
            ]

            # Check for error indicators
            error_indicators = [
                step_result.get("error") is not None,
                step_result.get("failed", False),
                step_result.get("status") == "FAILED",
            ]

            # Determine if outcome is favorable
            has_success = any(success_indicators)
            has_errors = any(error_indicators)

            # Outcome is favorable if there's success and no errors
            passed = has_success and not has_errors

            outcome_metrics = {
                "has_success_indicators": has_success,
                "has_error_indicators": has_errors,
                "success_indicators": success_indicators,
                "error_indicators": error_indicators,
                "step_metrics": metrics,
                "performance_metrics": performance,
                "favorable": passed,
            }

            if not passed:
                # Build a concise reason summary and attach structured context
                status_value = step_result.get("status")
                error_value = step_result.get("error")
                reason_parts = []
                if not has_success:
                    reason_parts.append("no success indicator")
                if has_errors:
                    reason_parts.append("error indicator present")
                reason_summary = (
                    "; ".join(reason_parts) if reason_parts else "unspecified"
                )

                # Determine training mode (blank vs full)
                blank_mode = (
                    os.environ.get("BLANK_TRAINING_MODE", "0") == "1"
                    or bool(self.config.get("BLANK_TRAINING_MODE", False))
                    or bool(self.config.get("blank_training_mode", False))
                )

                # Build explicit indicator values summary for human-readable logs
                indicator_summary = (
                    f"success={step_result.get('success', False)}, "
                    f"completed={step_result.get('completed', False)}, "
                    f"status={status_value}"
                )

                # Build mode-aware message and extra context
                if blank_mode:
                    message = (
                        f"⚠️ Step {self.step_name} outcome is not favorable"
                        f" | status={status_value} | reasons={reason_summary}"
                        f" | success_indicators=({indicator_summary})"
                        f" | note=BLANK MODE: This can be normal with limited data"
                    )
                    note_text = "BLANK MODE: Outcome may be unfavourable due to small samples; continuing"
                    next_steps_text = "Optional: sanity-check labeled/feature files and label balance; acceptable to proceed in blank mode"
                else:
                    message = (
                        f"⚠️ Step {self.step_name} outcome is not favorable"
                        f" | status={status_value} | reasons={reason_summary}"
                        f" | success_indicators=({indicator_summary})"
                        f" | next_steps=Review labeled/feature files, label balance, and prior warnings"
                    )
                    note_text = "FULL MODE: Unfavorable outcome requires investigation"
                    next_steps_text = "Review step outputs (labeled data, feature files, label balance) and warnings"

                # Emit as a warning with structured fields for JSON logs
                self.logger.warning(
                    message,
                    extra={
                        "step_name": self.step_name,
                        "status": status_value,
                        "has_success_indicators": has_success,
                        "has_error_indicators": has_errors,
                        "success_indicators": success_indicators,
                        "error_indicators": error_indicators,
                        "success_field_values": {
                            "success": step_result.get("success", False),
                            "completed": step_result.get("completed", False),
                            "status": step_result.get("status"),
                        },
                        "expected_success_logic": "any(success, completed, status==SUCCESS)",
                        "step_metrics_keys": list(metrics.keys())
                        if isinstance(metrics, dict)
                        else [],
                        "performance_metrics_keys": list(performance.keys())
                        if isinstance(performance, dict)
                        else [],
                        "error_message": (
                            str(error_value)[:500] if error_value is not None else None
                        ),
                        "mode": "blank" if blank_mode else "full",
                        "note": note_text,
                        "next_steps": next_steps_text,
                    },
                )

            return passed, outcome_metrics

        except Exception as e:
            self.print(
                validation_error(f"❌ Error in outcome favorability validation: {e}"),
            )
            return False, {"error": str(e)}

    async def run_validation(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run the validation and return results.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Dict[str, Any]: Validation results dictionary
        """
        try:
            self.logger.info(f"🔍 Running validation for {self.step_name}...")

            # Run the validation
            validation_passed = await self.validate(training_input, pipeline_state)

            # Prepare results
            results = {
                "step_name": self.step_name,
                "validation_passed": validation_passed,
                "validation_results": self.validation_results.copy(),
                "timestamp": "2024-01-01T00:00:00",  # Placeholder timestamp
            }

            if validation_passed:
                self.logger.info(f"✅ Validation for {self.step_name} passed")
            else:
                self.print(failed(f"⚠️ Validation for {self.step_name} failed"))

            return results

        except Exception as e:
            self.logger.exception(
                f"❌ Error running validation for {self.step_name}: {e}",
            )
            return {
                "step_name": self.step_name,
                "validation_passed": False,
                "error": str(e),
                "validation_results": {},
                "timestamp": "2024-01-01T00:00:00",  # Placeholder timestamp
            }
