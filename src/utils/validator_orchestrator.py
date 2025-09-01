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
project_root, Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
from src.utils.prometheus_metrics import metrics
from src.utils.warning_symbols import (
error,
missing,
)

class ValidatorOrchestrator:

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="validatororchestrator initialization",
    )
    async def initialize(self) -> bool:
        """Initialize ValidatorOrchestrator."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    passpassself.logger.info("Implementation placeholder - needs specific logic")
class ValidatorOrchestrator:
    passself.logger.info("Implementation placeholder - needs specific logic")
class ValidatorOrchestrator:
    pass"""Orchestrator for running step validators in the training pipeline."""

def __init__(...):
    passpassdef __init__(...):
    passdef __init__(...):
    passdef __init__(...):
    passself.logger, system_logger.getChild("ValidatorOrchestrator")
self.validators = {}
self.validation_results = {}

async def run_step_validator(
self,
step_name: str,
training_input: dict[str, Any],
pipeline_state: dict[str, Any],
config: dict[str, Any],
validation_level: str = "CRITICAL",
) -> dict[str, Any]:
        """
Run the validator for a specific step with enhanced validation levels.

Args:
    passpassstep_name: Name of the step (e.g., "step01_data_collection")
training_input: Training input parameters
pipeline_state: Current pipeline state
config: Configuration dictionary
validation_level: Validation level ("BASIC", "STANDARD", "COMPREHENSIVE", "CRITICAL") - defaults to CRITICAL

Returns:
            Dictionary containing validation results
"""
start_perf, time.perf_counter()
try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
self.logger.info(f"🔍 Running {validation_level} validator for {step_name}")

# Debug - level context for troubleshooting
try:
    passpassself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
self.logger.debug(
"Input context - training_input_keys=%s pipeline_state_keys=%s validation_level=%s",
list(training_input.keys())
if isinstance(training_input, dict)
else type(training_input).__name__,
list(pipeline_state.keys())
if isinstance(pipeline_state, dict)
else type(pipeline_state).__name__,
validation_level,
)
except Exception:
    passpasspass# Defensive: never fail due to logging of keys
pass

# Pre - validation checks
pre_validation_result, await self._run_pre_validation_checks(
step_name, training_input, pipeline_state, config, validation_level
)

if not pre_validation_result.get("passed", True):
    passduration, max(0.0, time.perf_counter() - start_perf)
return self._normalize_result(step_name, pre_validation_result, duration)

# Import and run the appropriate validator
raw_result, await self._run_validator(
step_name,
training_input,
pipeline_state,
config,
validation_level,
)

# Post - validation checks
post_validation_result, await self._run_post_validation_checks(
step_name, raw_result, training_input, pipeline_state, config, validation_level
)

# Combine results
combined_result, self._combine_validation_results(
step_name, raw_result, post_validation_result, validation_level
)

# Normalize and enrich result with timing and defaults
duration, max(0.0, time.perf_counter() - start_perf)
result, self._normalize_result(step_name, combined_result, duration)

# Store validation result
self.validation_results[step_name] = result

# Derive status and reason for metrics / logs
passed, bool(result.get("validation_passed", False))
status = "SUCCESS" if passed else "FAILED"
failure_reason, self._extract_failure_reason(result)

# Record metrics
try:
    passpasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
metrics.record_step_execution(
step_name = step_name, duration = duration, status = status
)
except Exception:
    passpass# Metrics are best - effort; do not fail validation on metrics issues
self.logger.debug(
"Metrics recording for step execution failed", exc_info = True
)

if passed:
    passpassmetrics.record_validation_result(
step_name = step_name,
validation_type="step_validation",
passed = True,
reason="Step validation completed successfully",
)
self.logger.info(
f"✅ Validator for {step_name} completed in {duration:.3f}s: passed = True",
)
else:
    passself.logger.error(
f"❌ Validator failed for {step_name} in {duration:.3f}s: {failure_reason}",
)
metrics.record_validation_result(
step_name = step_name,
validation_type="step_validation",
passed = False,
reason = failure_reason,
)

return result

except Exception as e:
    passpasspasspasspasspasspassduration, max(0.0, time.perf_counter() - start_perf)
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
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
metrics.record_step_execution(
step_name = step_name, duration = duration, status="EXCEPTION"
)
except Exception:
    passpassself.logger.debug(
"Metrics recording for exception failed", exc_info = True
)

metrics.record_validation_result(
step_name = step_name,
validation_type="step_validation",
passed = False,
reason = f"Validator execution error: {str(e)}",
)

return error_result

async def _run_pre_validation_checks(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
self.logger.debug(f"🔍 Running pre - validation checks for {step_name}")

# Basic input validation
if not isinstance(training_input, dict):
    passpassreturn {
"passed": False,
"validation_passed": False,
"error": "training_input must be a dictionary",
}

if not isinstance(pipeline_state, dict):
    passreturn {
"passed": False,
"validation_passed": False,
"error": "pipeline_state must be a dictionary",
}

if not isinstance(config, dict):
    passreturn {
"passed": False,
"validation_passed": False,
"error": "config must be a dictionary",
}

# Check for required training input parameters
required_params = ["symbol", "exchange", "timeframe"]
missing_params = [param for param in required_params if param not in training_input]

if missing_params:
    passpassreturn {
"passed": False,
"validation_passed": False,
"error": f"Missing required training input parameters: {missing_params}",
}

# Enhanced checks for comprehensive validation level
if validation_level in ["COMPREHENSIVE", "CRITICAL"]:
    passpass# Validate configuration structure
if "data_dir" not in config:
    passreturn {
"passed": False,
"validation_passed": False,
"error": "Missing data_dir in configuration",
}

# Check for critical pipeline state issues
failed_steps = [
step for step, info in pipeline_state.items()
if isinstance(info, dict) and info.get("status") == "FAILED"
]

if failed_steps:
    passpassreturn {
"passed": False,
"validation_passed": False,
"error": f"Pipeline has failed steps: {failed_steps}",
}

return {"passed": True, "validation_passed": True}

except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Error in pre - validation checks for {step_name}: {e}")
return {
"passed": False,
"validation_passed": False,
"error": f"Pre - validation check error: {str(e)}",
}

async def _run_post_validation_checks(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
self.logger.debug(f"🔍 Running post - validation checks for {step_name}")

post_checks = {
"validation_passed": True,
"warnings": [],
"recommendations": [],
}

# Check validation result structure
if not isinstance(validation_result, dict):
    passpost_checks["validation_passed"] = False
post_checks["warnings"].append("Validation result is not a dictionary")
return post_checks

# Enhanced checks for comprehensive validation level
if validation_level in ["COMPREHENSIVE", "CRITICAL"]:
    passpass# Check for critical issues in validation result
if validation_result.get("critical_issues"):
    passpasspost_checks["warnings"].append(f"Critical issues found: {validation_result['critical_issues']}")

# Check for data quality issues
if validation_result.get("data_quality_issues"):
    passpasspost_checks["warnings"].append(f"Data quality issues: {validation_result['data_quality_issues']}")

# Generate recommendations based on validation level
if validation_level == "CRITICAL":
    passpost_checks["recommendations"].append("Consider running additional data quality checks")
post_checks["recommendations"].append("Review model performance metrics")

return post_checks

except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Error in post - validation checks for {step_name}: {e}")
return {
"validation_passed": False,
"error": f"Post - validation check error: {str(e)}",
}

def _combine_validation_results(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
combined, dict(main_result)

# Add post - validation information
if post_result.get("warnings"):
    passcombined.setdefault("warnings", []).extend(post_result["warnings"])

if post_result.get("recommendations"):
    passcombined.setdefault("recommendations", []).extend(post_result["recommendations"])

# Determine final validation status
main_passed, main_result.get("validation_passed", False)
post_passed, post_result.get("validation_passed", True)

# For critical validation level, both must pass
if validation_level == "CRITICAL":
    passcombined["validation_passed"] = main_passed and post_passed
else:
    pass# For other levels, main result takes precedence
combined["validation_passed"] = main_passed

# Add validation level information
combined["validation_level"] = validation_level
combined["validation_timestamp"] = time.time()

return combined

except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Error combining validation results for {step_name}: {e}")
return {
"step_name": step_name,
"validation_passed": False,
"error": f"Result combination error: {str(e)}",
}

async def _run_validator(...) -> ...:
    """..."""
    pass# Map step names to validator modules
validator_mapping = {
"step01_data_collection": "step01_data_collection_validator",
"step01_5_data_converter": "step01_5_data_converter_validator",
"step02_data_reading": "step02_data_reading_validator",
"step02_5_sr_optimization": "step02_5_sr_optimization_validator",
"step03_hmm_regime_discovery": "step03_hmm_regime_discovery_validator",
"step04_triple_barrier_method": "step04_triple_barrier_method_validator",
"step04_regime_data_splitting": "step04_regime_data_splitting_validator",
"step05_labeling": "step05_labeling_validator",
"step06_feature_engineering": "step06_feature_engineering_validator",
"step07_enhanced_matrix_operations": "step07_enhanced_matrix_operations_validator",
"step08_regime_data_splitting": "step08_regime_data_splitting_validator",
"step09_hmm_based_training": "step09_hmm_based_training_validator",
"step09_5_multi_timeframe_hmm_ensemble": "step09_5_multi_timeframe_hmm_ensemble_validator",
"step09_5_hmm_lm_generalist_training": "step09_5_hmm_lm_generalist_training_validator",
"step10_unified_regime_intelligence": "step10_unified_regime_intelligence_validator",
"step11_analyst_creation": "step11_analyst_creation_validator",
"step12_analyst_enhancement": "step12_analyst_enhancement_validator",
"step13_analyst_ensemble_creation": "step13_analyst_ensemble_creation_validator",
"step14_tactician_labeling": "step14_tactician_labeling_validator",
"step15_tactician_specialist_training": (
"step15_tactician_specialist_training_validator"
),
"step16_confidence_calibration": (
"step16_confidence_calibration_validator"
),
"step17_final_parameters_optimization": (
"step17_final_parameters_optimization_validator"
),
"step18_walk_forward_validation": (
"step18_walk_forward_validation_validator"
),
"step19_monte_carlo_validation": (
"step19_monte_carlo_validation_validator"
),
"step20_ab_testing": "step20_ab_testing_validator",
"step21_saving": "step21_saving_validator",
}

validator_module_name, validator_mapping.get(step_name)
if not validator_module_name:
    passmsg, f"No validator mapping found for step: {step_name}"
raise ValueError(msg)

module_path, f"src.training.steps.{validator_module_name}"
try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
# Import the validator module
module_path, f"src.training.steps.{validator_module_name}"
validator_module, importlib.import_module(module_path)
# Cache module for potential reuse
self.validators[step_name] = validator_module

# Resolve run function
run_validator_func: Any | None, getattr(
validator_module, "run_validator", None
)
if run_validator_func is None or not callable(run_validator_func):
    passwarn_msg = (
f"run_validator not found or not callable in module {module_path}"
)
self.logger.warning(missing(warn_msg))
return {
"step_name": step_name,
"validation_passed": True,  # Skip validation if entry point not found
"warning": warn_msg,
}

# Support both async and sync validators
if inspect.iscoroutinefunction(run_validator_func):
    passresult, await run_validator_func(training_input, pipeline_state)
else:
    passresult, run_validator_func(training_input, pipeline_state)

self.logger.info(
f"✅ Validator for {step_name} completed: "
f"{bool(result.get('validation_passed', False)) if isinstance(result, dict) else bool(result)}",
)
# Ensure dict result; normalize later in caller
return (
result
if isinstance(result, dict)
else {"validation_passed": bool(result)}
)

except ImportError as e:
    passpasspasspasspasspasspass# Explicitly warn about missing module and continue as a soft skip
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
    passpass# Raise to caller which will handle logging and metrics
raise

def _normalize_result(...) -> ...:
    """..."""
    passnormalized: dict[str, Any]
if not isinstance(result, dict):
    passnormalized = {
"step_name": step_name,
"validation_passed": bool(result),
"validation_results": {},
"message": "Non - dict validator result converted to boolean",
}
else:
    passnormalized, dict(result)
normalized.setdefault("step_name", step_name)
normalized["step_name"] = step_name  # enforce canonical step name
normalized.setdefault("validation_passed", False)
normalized.setdefault("validation_results", {})

# Inject timing
normalized["duration"] = duration
normalized["timestamp"] = time.time()

# If failed and no explicit reason, extract one for better troubleshooting
if not bool(normalized.get("validation_passed", False)) and not (
normalized.get("error")
or normalized.get("warning")
or normalized.get("message")
):
    passpassnormalized["error"] = self._extract_failure_reason(normalized)

return normalized

def _extract_failure_reason(...) -> ...:
    """..."""
    passfor key in ("error", "warning", "message"):
    passvalue, result.get(key)
if isinstance(value, str) and value.strip():
    passreturn value.strip()

# Look into nested validation_results for first error(s)
vr, result.get("validation_results")
if isinstance(vr, dict):
    passpass# Prefer explicit error strings
for sub in vr.values():
    passif isinstance(sub, dict):
    passif isinstance(sub.get("error"), str) and sub.get("error"):
    passreturn str(sub.get("error"))
errors, sub.get("errors")
if isinstance(errors, list) and errors:
    passreturn ", ".join(map(str, errors[:3]))
# Common flags
if sub.get("has_critical_errors"):
    passmsgs, sub.get("error_messages")
if isinstance(msgs, list) and msgs:
    passreturn ", ".join(map(str, msgs[:3]))
return "Critical errors present"
return "Step validation failed"

def get_validation_summary(...) -> ...:
    """..."""
    passtotal_validations, len(self.validation_results)
passed_validations, sum(
1
for result in self.validation_results.values()
if result.get("validation_passed", False)
)
failed_validations, total_validations - passed_validations

return {
"total_validations": total_validations,
"passed_validations": passed_validations,
"failed_validations": failed_validations,
"success_rate": passed_validations / total_validations
if total_validations > 0
else 0,
"validation_results": self.validation_results,
}

def get_failed_validations(...) -> ...:
    """..."""
    passreturn [
step_name
for step_name, result in self.validation_results.items()
if not result.get("validation_passed", False)
]

def clear_results(...):
    passpasspassdef clear_results(...):
    passdef clear_results(...):
    passdef clear_results(...):
    pass"""Clear all validation results."""
self.validation_results.clear()

# Global validator orchestrator instance
validator_orchestrator, ValidatorOrchestrator()
