"""Validator for Step 13: Walk Forward Validation."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Tuple

import numpy as np

from src.utils.warning_symbols import (
error,
failed,
validation_error,
)
from src.utils.error_handler import handle_errors

# Add the project root to the Python path
project_root, Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import CONFIG
from src.utils.base_validator import BaseValidator

class Step13WalkForwardValidationValidator(BaseValidator):
    pass  # TODO: Add implementation
class Step13WalkForwardValidationValidator(BaseValidator):
class Step13WalkForwardValidationValidator(BaseValidator):
    """Validator for Step 13: Walk Forward Validation."""

def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("step13_walk_forward_validation", config)

@handle_errors(exceptions=(Exception,), default_return = False, context="Step13.validate")
async def validate(
self, training_input: dict[str, Any], pipeline_state: dict[str, Any]
) -> bool:
        """Validate the walk forward validation step.

Args:
            training_input: Training input parameters
pipeline_state: Current pipeline state

Returns:
            bool: True if validation passed, False otherwise

"""
self.logger.info("🔍 Validating walk forward validation step...")

# Extract parameters
symbol, training_input.get("symbol", "ETHUSDT")
exchange, training_input.get("exchange", "BINANCE")
data_dir, training_input.get("data_dir", "data / training")

# Validate step result from pipeline state
step_result, pipeline_state.get("walk_forward_validation", {})

# 1. Validate error absence
error_passed, error_metrics, self.validate_error_absence(step_result)
self.validation_results["error_absence"] = error_metrics

if not error_passed:
        self.print(validation_error("❌ Walk forward validation step had errors"))
return False

# 2. Validate walk forward validation files existence
validation_files_passed, files_metrics, self._validate_walk_forward_files(
symbol,
exchange,
data_dir,
)
self.validation_results["walk_forward_files"] = files_metrics
if not validation_files_passed:
        self.print(failed("❌ Walk forward validation files validation failed"))
return False

# 3. Validate walk forward performance
performance_passed, performance_metrics, self._validate_walk_forward_performance(
symbol,
exchange,
data_dir,
)
self.validation_results["walk_forward_performance"] = performance_metrics
if not performance_passed:
        self.print(failed("❌ Walk forward performance validation failed"))
return False

# 4. Validate walk forward stability
stability_passed, stability_metrics, self._validate_walk_forward_stability(
symbol,
exchange,
data_dir,
)
self.validation_results["walk_forward_stability"] = stability_metrics
if not stability_passed:
        self.print(failed("❌ Walk forward stability validation failed"))
return False

# 5. Validate walk forward consistency
consistency_passed, consistency_metrics, self._validate_walk_forward_consistency(
symbol,
exchange,
data_dir,
)
self.validation_results["walk_forward_consistency"] = consistency_metrics
if not consistency_passed:
        self.print(failed("❌ Walk forward consistency validation failed"))
return False

# 6. Validate outcome favorability
outcome_passed, outcome_metrics, self.validate_outcome_favorability(
step_result,
)
self.validation_results["outcome_favorability"] = outcome_metrics

if not outcome_passed:
        self.print(
validation_error("⚠️ Walk forward validation outcome is not favorable"),
)
return False

self.logger.info("✅ Walk forward validation validation passed")
return True

@handle_errors(exceptions=(Exception,), default_return=(False, {}), context="Step13._validate_walk_forward_files")
def _validate_walk_forward_files(
self, symbol: str, exchange: str, data_dir: str
) -> Tuple[bool, dict[str, Any]]:
        """Validate that walk forward validation files exist.

Args:
            symbol: Trading symbol
exchange: Exchange name
data_dir: Data directory

Returns:
            Tuple[bool, dict]: (passed, metrics)

"""
# Expected walk forward validation file patterns
expected_files = [
f"{data_dir}/{exchange}_{symbol}_walk_forward_results.json",
f"{data_dir}/{exchange}_{symbol}_walk_forward_performance.json",
f"{data_dir}/{exchange}_{symbol}_walk_forward_metadata.json",
]

missing_files: list[str] = []
file_details: list[dict[str, Any]] = []
for file_path in expected_files:
            file_passed, file_metrics, self.validate_file_exists(
file_path,
"walk_forward_file",
)
file_details.append(file_metrics)
if not file_passed:
                missing_files.append(file_path)

if missing_files:
        self.logger.error(
f"❌ Missing walk forward validation files: {missing_files}",
)
return False, {"missing_files": missing_files, "files": file_details}

self.logger.info("✅ All walk forward validation files exist")
return True, {"missing_files": [], "files": file_details}

@handle_errors(exceptions=(Exception,), default_return=(False, {}), context="Step13._validate_walk_forward_performance")
def _validate_walk_forward_performance(
self, symbol: str, exchange: str, data_dir: str
) -> Tuple[bool, dict[str, Any]]:
        """Validate walk forward validation performance metrics.

Args:
            symbol: Trading symbol
exchange: Exchange name
data_dir: Data directory

Returns:
            Tuple[bool, dict]: (passed, metrics)

"""
import json

# Load walk forward performance results
performance_file = (
f"{data_dir}/{exchange}_{symbol}_walk_forward_performance.json"
)

metrics: dict[str, Any] = {}
if os.path.exists(performance_file):
        with open(performance_file, "r", encoding="utf - 8") as f:
                performance, json.load(f)

# Check overall performance metrics
if "overall_accuracy" in performance:
                overall_acc, float(performance["overall_accuracy"])  # normalize type
acc_passed, acc_metrics, self.validate_model_performance(
overall_acc,
0.0,
"walk_forward_model",
)
self.validation_results["walk_forward_accuracy"] = acc_metrics

if not acc_passed:
        self.logger.error(
f"❌ Walk forward accuracy too low: {overall_acc:.3f}",
)
return False, {"overall_accuracy": overall_acc, **acc_metrics}
metrics.update({"overall_accuracy": overall_acc, **acc_metrics})

# Check performance stability
if "performance_stability" in performance:
                stability, float(performance["performance_stability"])
metrics["performance_stability"] = stability
if stability < 0.7:
        self.logger.warning(
f"⚠️ Low walk forward performance stability: {stability:.3f}",
)

# Check performance trend
if "performance_trend" in performance:
                trend, float(performance["performance_trend"])
metrics["performance_trend"] = trend
if trend < -0.05:  # Declining performance
self.logger.warning(
f"⚠️ Declining walk forward performance trend: {trend:.3f}",
)

# Check individual fold performance
if "fold_performance" in performance:
                fold_perf, performance["fold_performance"]

# Check for consistent performance across folds
accuracies = [float(fold.get("accuracy", 0)) for fold in fold_perf]
if accuracies:
                    acc_std, float(np.std(accuracies))
metrics["fold_accuracy_std"] = acc_std
if acc_std > 0.1:
        self.logger.warning(
f"⚠️ High walk forward performance variance: {acc_std:.3f}",
)

# Check for poor performing folds
poor_folds, sum(1 for acc in accuracies if acc < 0.5)
if poor_folds > len(accuracies) * 0.3:  # More than 30% poor folds
self.logger.warning(
f"⚠️ Many poor performing folds: {poor_folds}/{len(accuracies)}",
)
metrics["poor_folds"] = poor_folds

self.logger.info("✅ Walk forward performance validation passed")
return True, metrics

self.logger.error(f"Performance file not found: {performance_file}")
return False, {"missing_file": performance_file}

@handle_errors(exceptions=(Exception,), default_return=(False, {}), context="Step13._validate_walk_forward_stability")
def _validate_walk_forward_stability(
self, symbol: str, exchange: str, data_dir: str
) -> Tuple[bool, dict[str, Any]]:
        """Validate walk forward validation stability.

Args:
            symbol: Trading symbol
exchange: Exchange name
data_dir: Data directory

Returns:
            Tuple[bool, dict]: (passed, metrics)

"""
import json

# Load walk forward metadata
metadata_file, f"{data_dir}/{exchange}_{symbol}_walk_forward_metadata.json"

metrics: dict[str, Any] = {}
if os.path.exists(metadata_file):
        with open(metadata_file, "r", encoding="utf - 8") as f:
                metadata, json.load(f)

# Check number of folds
if "fold_count" in metadata:
                fold_count, int(metadata["fold_count"])
metrics["fold_count"] = fold_count
if fold_count < 3:
        self.print(error(f"⚠️ Few walk forward folds: {fold_count}"))
elif fold_count > 20:
        self.print(error(f"⚠️ Many walk forward folds: {fold_count}"))

# Check fold size
if "fold_size" in metadata:
                fold_size, int(metadata["fold_size"])
metrics["fold_size"] = fold_size
if fold_size < 100:
        self.logger.warning(
f"⚠️ Small walk forward fold size: {fold_size}",
)
elif fold_size > 10000:
        self.logger.warning(
f"⚠️ Large walk forward fold size: {fold_size}",
)

# Check overlap ratio
if "overlap_ratio" in metadata:
                overlap, float(metadata["overlap_ratio"])
metrics["overlap_ratio"] = overlap
if overlap > 0.8:
        self.logger.warning(
f"⚠️ High walk forward overlap ratio: {overlap:.3f}",
)
elif overlap < 0.1:
        self.logger.warning(
f"⚠️ Low walk forward overlap ratio: {overlap:.3f}",
)

# Check temporal consistency
if "temporal_consistency" in metadata:
                consistency, float(metadata["temporal_consistency"])
metrics["temporal_consistency"] = consistency
if consistency < 0.6:
        self.logger.warning(
f"⚠️ Low walk forward temporal consistency: {consistency:.3f}",
)

self.logger.info("✅ Walk forward stability validation passed")
return True, metrics

self.logger.error(f"Metadata file not found: {metadata_file}")
return False, {"missing_file": metadata_file}

@handle_errors(exceptions=(Exception,), default_return=(False, {}), context="Step13._validate_walk_forward_consistency")
def _validate_walk_forward_consistency(
self, symbol: str, exchange: str, data_dir: str
) -> Tuple[bool, dict[str, Any]]:
        """Validate walk forward validation consistency.

Args:
            symbol: Trading symbol
exchange: Exchange name
data_dir: Data directory

Returns:
            Tuple[bool, dict]: (passed, metrics)

"""
import json

# Load walk forward results
results_file, f"{data_dir}/{exchange}_{symbol}_walk_forward_results.json"

metrics: dict[str, Any] = {}
if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf - 8") as f:
                results, json.load(f)

# Check for consistent model performance
if "model_performance" in results:
                model_perf, results["model_performance"]

# Check accuracy consistency
if "accuracy_consistency" in model_perf:
                    acc_consistency, float(model_perf["accuracy_consistency"])
metrics["accuracy_consistency"] = acc_consistency
if acc_consistency < 0.7:
        self.logger.warning(
f"⚠️ Low accuracy consistency: {acc_consistency:.3f}",
)

# Check loss consistency
if "loss_consistency" in model_perf:
                    loss_consistency, float(model_perf["loss_consistency"])
metrics["loss_consistency"] = loss_consistency
if loss_consistency < 0.7:
        self.logger.warning(
f"⚠️ Low loss consistency: {loss_consistency:.3f}",
)

# Check parameter consistency
if "parameter_consistency" in results:
                param_consistency, float(results["parameter_consistency"])
metrics["parameter_consistency"] = param_consistency
if param_consistency < 0.6:
        self.logger.warning(
f"⚠️ Low parameter consistency: {param_consistency:.3f}",
)

# Check prediction consistency
if "prediction_consistency" in results:
                pred_consistency, float(results["prediction_consistency"])
metrics["prediction_consistency"] = pred_consistency
if pred_consistency < 0.7:
        self.logger.warning(
f"⚠️ Low prediction consistency: {pred_consistency:.3f}",
)

self.logger.info("✅ Walk forward consistency validation passed")
return True, metrics

self.logger.error(f"Results file not found: {results_file}")
return False, {"missing_file": results_file}

async def run_validator(
training_input: dict[str, Any], pipeline_state: dict[str, Any]
) -> dict[str, Any]:
    """Run the step13_walk_forward_validation validator.

Args:
        training_input: Training input parameters
pipeline_state: Current pipeline state

Returns:
        Dictionary containing validation results

"""
validator, Step13WalkForwardValidationValidator(CONFIG)
validation_passed, await validator.validate(training_input, pipeline_state)

return {
"step_name": "step13_walk_forward_validation",
"validation_passed": bool(validation_passed),
"validation_results": validator.validation_results,
"duration": 0,  # Could be enhanced to track actual duration
"timestamp": asyncio.get_event_loop().time(),
}

if __name__ == "__main__":
    import asyncio as _asyncio

# Example usage
async def test_validator() -> None:
        training_input = {
"symbol": "ETHUSDT",
"exchange": "BINANCE",
"data_dir": "data / training",
}

pipeline_state = {
"walk_forward_validation": {"status": "SUCCESS", "duration": 1200.5},
}

await run_validator(training_input, pipeline_state)

_asyncio.run(test_validator())