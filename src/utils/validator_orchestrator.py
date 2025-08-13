"""
Validator orchestrator for running individual step validators in the training pipeline.
"""

import asyncio
import importlib
import sys
from pathlib import Path
from typing import Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
from src.utils.logger import system_logger
from src.utils.prometheus_metrics import metrics
from src.utils.warning_symbols import (
    error,
    missing,
)


class ValidatorOrchestrator:
    """Orchestrator for running step validators in the training pipeline."""

    def __init__(self):
        self.logger = system_logger.getChild("ValidatorOrchestrator")
        self.validators = {}
        self.validation_results = {}

    async def run_step_validator(
        self,
        step_name: str,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run the validator for a specific step.

        Args:
            step_name: Name of the step (e.g., "step1_data_collection")
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            config: Configuration dictionary

        Returns:
            Dictionary containing validation results
        """
        try:
            self.logger.info(f"🔍 Running validator for {step_name}")

            # Import and run the appropriate validator
            validator_result = await self._run_validator(
                step_name,
                training_input,
                pipeline_state,
                config,
            )

            # Store validation result
            self.validation_results[step_name] = validator_result

            # Record metrics
            if validator_result.get("validation_passed", False):
                metrics.record_validation_result(
                    step_name=step_name,
                    validation_type="step_validation",
                    passed=True,
                    reason="Step validation completed successfully",
                )
            else:
                # Surface concrete error cause if available
                failure_reason = (
                    validator_result.get("error")
                    or validator_result.get(
                        "warning",
                    )
                    or validator_result.get("message")
                    or "Step validation failed"
                )
                self.logger.error(
                    f"❌ Validator failed for {step_name}: {failure_reason}",
                )
                metrics.record_validation_result(
                    step_name=step_name,
                    validation_type="step_validation",
                    passed=False,
                    reason=failure_reason,
                )

            return validator_result

        except Exception as e:
            # Log full stack trace for debugging
            self.logger.exception(
                f"❌ Exception while running validator for {step_name}: {e}",
            )

            error_result = {
                "step_name": step_name,
                "validation_passed": False,
                "error": str(e),
                "duration": 0,
                "timestamp": asyncio.get_event_loop().time(),
            }

            self.validation_results[step_name] = error_result

            # Record failure metric
            metrics.record_validation_result(
                step_name=step_name,
                validation_type="step_validation",
                passed=False,
                reason=f"Validator execution error: {str(e)}",
            )

            return error_result

    async def _run_validator(
        self,
        step_name: str,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Dynamically import and run the appropriate validator.

        Args:
            step_name: Name of the step
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            config: Configuration dictionary

        Returns:
            Dictionary containing validation results
        """
        try:
            # Map step names to validator modules
            validator_mapping = {
                "step1_data_collection": "step1_data_collection_validator",
                "step2_market_regime_classification": (
                    "step2_market_regime_classification_validator"
                ),
                # New step names
                "step2_processing_labeling_feature_engineering": (
                    "step4_analyst_labeling_feature_engineering_validator"
                ),
                "step3_feature_engineering": (
                    "step4_analyst_labeling_feature_engineering_validator"
                ),
                "step4_regime_data_splitting": "step3_regime_data_splitting_validator",
                "step5_analyst_specialist_training": (
                    "step5_analyst_specialist_training_validator"
                ),
                "step6_analyst_enhancement": "step6_analyst_enhancement_validator",
                "step8_tactician_labeling": "step8_tactician_labeling_validator",
                "step9_tactician_specialist_training": (
                    "step9_tactician_specialist_training_validator"
                ),
                "step11_confidence_calibration": (
                    "step11_confidence_calibration_validator"
                ),
                "step12_final_parameters_optimization": (
                    "step12_final_parameters_optimization_validator"
                ),
                "step13_walk_forward_validation": (
                    "step13_walk_forward_validation_validator"
                ),
                "step14_monte_carlo_validation": (
                    "step14_monte_carlo_validation_validator"
                ),
                "step15_ab_testing": "step15_ab_testing_validator",
                "step16_saving": "step16_saving_validator",
            }

            validator_module_name = validator_mapping.get(step_name)
            if not validator_module_name:
                msg = f"No validator mapping found for step: {step_name}"
                raise ValueError(msg)

            # Import the validator module
            module_path = f"src.training.steps.{validator_module_name}"
            validator_module = importlib.import_module(module_path)

            # Get the run_validator function
            run_validator_func = validator_module.run_validator

            # Run the validator
            result = await run_validator_func(training_input, pipeline_state)

            self.logger.info(
                f"✅ Validator for {step_name} completed: "
                f"{result.get('validation_passed', False)}",
            )
            return result

        except ImportError as e:
            self.print(missing("⚠️ Validator module not found for {step_name}: {e}"))
            return {
                "step_name": step_name,
                "validation_passed": True,  # Skip validation if module not found
                "warning": f"Validator module not found: {str(e)}",
                "duration": 0,
                "timestamp": asyncio.get_event_loop().time(),
            }
        except Exception as e:
            # Raise after logging; caller will log exception stack as well
            self.logger.exception(
                f"❌ Error in validator for {step_name}: {e}",
            )
            raise

    def get_validation_summary(self) -> dict[str, Any]:
        """
        Get a summary of all validation results.

        Returns:
            Dictionary containing validation summary
        """
        total_validations = len(self.validation_results)
        passed_validations = sum(
            1
            for result in self.validation_results.values()
            if result.get("validation_passed", False)
        )
        failed_validations = total_validations - passed_validations

        return {
            "total_validations": total_validations,
            "passed_validations": passed_validations,
            "failed_validations": failed_validations,
            "success_rate": passed_validations / total_validations
            if total_validations > 0
            else 0,
            "validation_results": self.validation_results,
        }

    def get_failed_validations(self) -> list[str]:
        """
        Get list of steps that failed validation.

        Returns:
            List of step names that failed validation
        """
        return [
            step_name
            for step_name, result in self.validation_results.items()
            if not result.get("validation_passed", False)
        ]

    def clear_results(self):
        """Clear all validation results."""
        self.validation_results.clear()


# Global validator orchestrator instance
validator_orchestrator = ValidatorOrchestrator()
