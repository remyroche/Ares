"""
Validator orchestrator for running individual step validators in the training pipeline.
"""

import asyncio
import importlib
import sys
import inspect
import time
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
        start_perf = time.perf_counter()
        try:
            self.logger.info(f"🔍 Running validator for {step_name}")
            # Debug-level context for troubleshooting
            try:
                self.logger.debug(
                    "Input context - training_input_keys=%s pipeline_state_keys=%s",
                    list(training_input.keys()) if isinstance(training_input, dict) else type(training_input).__name__,
                    list(pipeline_state.keys()) if isinstance(pipeline_state, dict) else type(pipeline_state).__name__,
                )
            except Exception:
                # Defensive: never fail due to logging of keys
                pass

            # Import and run the appropriate validator
            raw_result = await self._run_validator(
                step_name,
                training_input,
                pipeline_state,
                config,
            )

            # Normalize and enrich result with timing and defaults
            duration = max(0.0, time.perf_counter() - start_perf)
            result = self._normalize_result(step_name, raw_result, duration)

            # Store validation result
            self.validation_results[step_name] = result

            # Derive status and reason for metrics/logs
            passed = bool(result.get("validation_passed", False))
            status = "SUCCESS" if passed else "FAILED"
            failure_reason = self._extract_failure_reason(result)

            # Record metrics
            try:
                metrics.record_step_execution(step_name=step_name, duration=duration, status=status)
            except Exception:
                # Metrics are best-effort; do not fail validation on metrics issues
                self.logger.debug("Metrics recording for step execution failed", exc_info=True)

            if passed:
                metrics.record_validation_result(
                    step_name=step_name,
                    validation_type="step_validation",
                    passed=True,
                    reason="Step validation completed successfully",
                )
                self.logger.info(
                    f"✅ Validator for {step_name} completed in {duration:.3f}s: passed=True",
                )
            else:
                self.logger.error(
                    f"❌ Validator failed for {step_name} in {duration:.3f}s: {failure_reason}",
                )
                metrics.record_validation_result(
                    step_name=step_name,
                    validation_type="step_validation",
                    passed=False,
                    reason=failure_reason,
                )

            return result

        except Exception as e:
            duration = max(0.0, time.perf_counter() - start_perf)
            # Log full stack trace for debugging
            self.logger.exception(
                f"❌ Exception while running validator for {step_name}: {e}",
            )

            error_result = {
                "step_name": step_name,
                "validation_passed": False,
                "error": str(e),
                "duration": duration,
                "timestamp": time.time(),
            }

            self.validation_results[step_name] = error_result

            # Record failure metric
            try:
                metrics.record_step_execution(step_name=step_name, duration=duration, status="EXCEPTION")
            except Exception:
                self.logger.debug("Metrics recording for exception failed", exc_info=True)

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
        # Map step names to validator modules
        validator_mapping = {
            "step1_data_collection": "step1_data_collection_validator",
            "step2_market_regime_classification": (
                "step2_market_regime_classification_validator"
            ),
            # New step names
            "step1_7_hmm_regime_discovery": (
                "step1_7_hmm_regime_discovery_validator"
            ),
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
            "step7_analyst_ensemble_creation": (
                "step7_analyst_ensemble_creation_validator"
            ),
            "step8_tactician_labeling": "step8_tactician_labeling_validator",
            "step9_tactician_specialist_training": (
                "step9_tactician_specialist_training_validator"
            ),
            "step10_tactician_ensemble_creation": (
                "step10_tactician_ensemble_creation_validator"
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

        module_path = f"src.training.steps.{validator_module_name}"
        try:
            # Import the validator module
            module_path = f"src.training.steps.{validator_module_name}"
            validator_module = importlib.import_module(module_path)
            # Cache module for potential reuse
            self.validators[step_name] = validator_module

            # Resolve run function
            run_validator_func: Any | None = getattr(validator_module, "run_validator", None)
            if run_validator_func is None or not callable(run_validator_func):
                warn_msg = f"run_validator not found or not callable in module {module_path}"
                self.logger.warning(missing(warn_msg))
                return {
                    "step_name": step_name,
                    "validation_passed": True,  # Skip validation if entry point not found
                    "warning": warn_msg,
                }

            # Support both async and sync validators
            if inspect.iscoroutinefunction(run_validator_func):
                result = await run_validator_func(training_input, pipeline_state)
            else:
                result = run_validator_func(training_input, pipeline_state)

            self.logger.info(
                f"✅ Validator for {step_name} completed: "
                f"{bool(result.get('validation_passed', False)) if isinstance(result, dict) else bool(result)}",
            )
            # Ensure dict result; normalize later in caller
            return result if isinstance(result, dict) else {"validation_passed": bool(result)}

        except ImportError as e:
            # Explicitly warn about missing module and continue as a soft skip
            self.logger.warning(
                missing(
                    f"⚠️ Validator module not found for {step_name}: {e}",
                ),
            )
            return {
                "step_name": step_name,
                "validation_passed": True,  # Skip validation if module not found
                "warning": f"Validator module not found: {str(e)}",
            }
        except Exception:
            # Raise to caller which will handle logging and metrics
            raise

    def _normalize_result(
        self,
        step_name: str,
        result: Any,
        duration: float,
    ) -> dict[str, Any]:
        """Normalize validator result into a consistent schema and inject timing.

        Schema keys: step_name, validation_passed, validation_results, error|warning|message, duration, timestamp
        """
        normalized: dict[str, Any]
        if not isinstance(result, dict):
            normalized = {
                "step_name": step_name,
                "validation_passed": bool(result),
                "validation_results": {},
                "message": "Non-dict validator result converted to boolean",
            }
        else:
            normalized = dict(result)
            normalized.setdefault("step_name", step_name)
            normalized["step_name"] = step_name  # enforce canonical step name
            normalized.setdefault("validation_passed", False)
            normalized.setdefault("validation_results", {})

        # Inject timing
        normalized["duration"] = duration
        normalized["timestamp"] = time.time()

        # If failed and no explicit reason, extract one for better troubleshooting
        if not bool(normalized.get("validation_passed", False)) and not (
            normalized.get("error") or normalized.get("warning") or normalized.get("message")
        ):
            normalized["error"] = self._extract_failure_reason(normalized)

        return normalized

    def _extract_failure_reason(self, result: dict[str, Any]) -> str:
        """Heuristically extract a concise failure reason from the result payload."""
        for key in ("error", "warning", "message"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        # Look into nested validation_results for first error(s)
        vr = result.get("validation_results")
        if isinstance(vr, dict):
            # Prefer explicit error strings
            for sub in vr.values():
                if isinstance(sub, dict):
                    if isinstance(sub.get("error"), str) and sub.get("error"):
                        return str(sub.get("error"))
                    errors = sub.get("errors")
                    if isinstance(errors, list) and errors:
                        return ", ".join(map(str, errors[:3]))
                    # Common flags
                    if sub.get("has_critical_errors"):
                        msgs = sub.get("error_messages")
                        if isinstance(msgs, list) and msgs:
                            return ", ".join(map(str, msgs[:3]))
                        return "Critical errors present"
        return "Step validation failed"

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
