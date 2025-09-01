# src/training/validator.py

import time
from typing import Any

from src.training.steps.data_preparation_components.training_validation_config import (
    VALIDATION_FUNCTIONS,
    can_proceed_to_step,
    get_progression_rules,
    get_validation_config,
)


class TrainingStepValidator:
    """Validates training steps and prevents progression on significant errors."""

    def __init__(self) -> None:
        self.step_errors = {}
        self.critical_errors = []
        self.warnings = []
        self.step_status = {}

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
        metrics: dict[str, float],
    ) -> tuple[bool, list[str]]:
        """Validate step metrics against configured thresholds."""
        config = get_validation_config(step_name)
        if not config or "thresholds" not in config:
            return True, []

        failed_thresholds = []
        for metric_name, threshold_config in config["thresholds"].items():
            if metric_name in metrics:
                value = metrics[metric_name]
                min_val = threshold_config.get("min")
                max_val = threshold_config.get("max")

                if min_val is not None and value < min_val:
                    failed_thresholds.append(
                        f"{metric_name} ({value}) below minimum threshold ({min_val})",
                    )
                elif max_val is not None and value > max_val:
                    failed_thresholds.append(
                        f"{metric_name} ({value}) above maximum threshold ({max_val})",
                    )

        return len(failed_thresholds) == 0, failed_thresholds
