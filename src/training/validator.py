# src/training/validator.py

import time
from typing import Any

from src.training.training_validation_config import (
    VALIDATION_FUNCTIONS,
    can_proceed_to_step,
    get_progression_rules,
    get_validation_config,
)


class TrainingStepValidator:
    """Validates training steps and prevents progression on significant errors."""

    def __init__(self):
        self.step_errors = {}
        self.critical_errors = []
        self.warnings = []
        self.step_status = {}

    def add_error(self, step_name: str, error: str, severity: str = "ERROR"):
        """Add an error for a specific step."""
        if step_name not in self.step_errors:
            self.step_errors[step_name] = []

        self.step_errors[step_name].append(
            {"error": error, "severity": severity, "timestamp": time.time()},
        )

        if severity == "CRITICAL":
            self.critical_errors.append(f"{step_name}: {error}")

    def add_warning(self, step_name: str, warning: str):
        """Add a warning for a specific step."""
        if step_name not in self.warnings:
            self.warnings[step_name] = []
        self.warnings[step_name].append(warning)

    def set_step_status(self, step_name: str, status: str, details: str = ""):
        """Set the status of a step."""
        self.step_status[step_name] = {
            "status": status,  # SUCCESS, FAILED, WARNING, SKIPPED
            "details": details,
            "timestamp": time.time(),
        }

    def can_proceed_to_next_step(
        self,
        current_step: str,
        next_step: str,
    ) -> tuple[bool, str]:
        """Check if we can proceed to the next step based on current step status."""
        # Use the validation configuration to check progression rules
        can_proceed, message = can_proceed_to_step(
            current_step,
            next_step,
            self.step_status,
        )

        # Additional checks for critical errors
        if current_step in self.step_errors:
            critical_errors = [
                e for e in self.step_errors[current_step] if e["severity"] == "CRITICAL"
            ]
            if critical_errors:
                return (
                    False,
                    f"Cannot proceed to {next_step}: {len(critical_errors)} critical errors in {current_step}",
                )

        # Check step status
        if current_step in self.step_status:
            status = self.step_status[current_step]["status"]
            if status == "FAILED":
                # Check if the step can be skipped according to configuration
                current_rules = get_progression_rules(current_step)
                if not current_rules.get("can_skip", False):
                    return (
                        False,
                        f"Cannot proceed to {next_step}: {current_step} failed and cannot be skipped",
                    )
            elif status == "SKIPPED":
                return True, f"Proceeding to {next_step}: {current_step} was skipped"

        return can_proceed, message

    def get_step_summary(self) -> dict[str, Any]:
        """Get a summary of all step statuses and errors."""
        return {
            "step_status": self.step_status,
            "step_errors": self.step_errors,
            "critical_errors": self.critical_errors,
            "warnings": self.warnings,
        }

    def has_critical_errors(self) -> bool:
        """Check if there are any critical errors."""
        return len(self.critical_errors) > 0

    def get_critical_errors(self) -> list:
        """Get all critical errors."""
        return self.critical_errors.copy()

    def validate_step_results(
        self,
        step_name: str,
        results: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """Validate step results using the validation configuration."""
        if step_name in VALIDATION_FUNCTIONS:
            return VALIDATION_FUNCTIONS[step_name](results)
        return True, []

    def validate_step_thresholds(
        self,
        step_name: str,
        metrics: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """Validate step metrics against configured thresholds."""
        errors = []
        config = get_validation_config(step_name)

        for metric, threshold in config.items():
            if metric in metrics:
                value = metrics[metric]
                if isinstance(threshold, (int, float)):
                    if metric.startswith("min_") and value < threshold:
                        errors.append(f"{metric}: {value} < {threshold}")
                    elif metric.startswith("max_") and value > threshold:
                        errors.append(f"{metric}: {value} > {threshold}")

        return len(errors) == 0, errors
