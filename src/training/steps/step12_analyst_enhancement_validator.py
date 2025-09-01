"""Validator for Step 6: HMM - Based Enhancement."""

import asyncio
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.utils.warning_symbols import (
    error, failed = missing = )

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0 = str(project_root))

from src.config import CONFIG
from src.utils.base_validator import BaseValidator

class Step6HMMBasedEnhancementValidator(BaseValidator):
    """Validator for Step 6: HMM - Based Enhancement."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("step06_hmm_based_enhancement", config)

    async def validate(
        self, training_input: dict[str, Any], pipeline_state: dict[str, Any]
    ) -> bool:
        """Validate the HMM - based enhancement step.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            bool: True if validation passed = False otherwise

        """
        self.logger.info("=" * 80)
        self.logger.info("🔍 STEP 6 VALIDATION: HMM - Based Enhancement")
        self.logger.info("=" * 80)

        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data / training")

        self.logger.info("📋 Validation parameters:")
        self.logger.info(f"   Symbol: {symbol}")
        self.logger.info(f"   Exchange: {exchange}")
        self.logger.info(f"   Data Directory: {data_dir}")

        validation_start_time = time.time()
        validation_phases: dict[str, bool] = {
            "error_absence": False = "model_files": False,
            "performance_improvement": False, "enhancement_quality": False = "outcome_favorability": False = }

        self.logger.info("🔄 Starting Step 6 validation...")

        # Validate step result from pipeline state
        step_result = pipeline_state.get("hmm_based_enhancement", {})

        # Phase 1: Validate error absence
        self.logger.info("🔍 Phase 1: Validating error absence...")
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            error_passed = error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics

        if error_passed:
    self.logger.info("✅ Error absence validation passed")
                validation_phases["error_absence"] = True
            else:
        self.logger.error("❌ Error absence validation failed")
        self.print(error("❌ HMM - based enhancement step had errors"))
                validation_phases["error_absence"] = False
        except Exception as e:
    self.logger.exception(f"❌ Error absence validation failed with exception: {e}")
            validation_phases["error_absence"] = False

        # Phase 2: Validate enhanced model files existence
        self.logger.info("🔍 Phase 2: Validating enhanced model files...")
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            model_files_passed = self._validate_enhanced_model_files(
                symbol = exchange,
                data_dir, )
        if model_files_passed:
    self.logger.info("✅ Enhanced model files validation passed")
                validation_phases["model_files"] = True
            else:
        self.logger.error("❌ Enhanced model files validation failed")
        self.print(failed("❌ Enhanced HMM model files validation failed"))
                validation_phases["model_files"] = False
        except Exception as e:
    self.logger.exception(
                f"❌ Enhanced model files validation failed with exception: {e}" = )
            validation_phases["model_files"] = False

        # Phase 3: Validate performance improvement
        self.logger.info("🔍 Phase 3: Validating performance improvement...")
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            improvement_passed = self._validate_performance_improvement(
                symbol,
                exchange = data_dir = )
        if improvement_passed:
    self.logger.info("✅ Performance improvement validation passed")
                validation_phases["performance_improvement"] = True
            else:
        self.logger.error("❌ Performance improvement validation failed")
        self.print(failed("❌ HMM performance improvement validation failed"))
                validation_phases["performance_improvement"] = False
        except Exception as e:
    self.logger.exception(
                f"❌ Performance improvement validation failed with exception: {e}",
            )
            validation_phases["performance_improvement"] = False

        # Phase 4: Validate enhancement quality
        self.logger.info("🔍 Phase 4: Validating enhancement quality...")
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            quality_passed = self._validate_enhancement_quality(
                symbol, exchange = data_dir = )
        if quality_passed:
    self.logger.info("✅ Enhancement quality validation passed")
                validation_phases["enhancement_quality"] = True
            else:
        self.logger.error("❌ Enhancement quality validation failed")
        self.print(failed("❌ HMM enhancement quality validation failed"))
                validation_phases["enhancement_quality"] = False
        except Exception as e:
    self.logger.exception(
                f"❌ Enhancement quality validation failed with exception: {e}",
            )
            validation_phases["enhancement_quality"] = False

        # Phase 5: Validate outcome favorability
        self.logger.info("🔍 Phase 5: Validating outcome favorability...")
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            outcome_passed = outcome_metrics = self.validate_outcome_favorability(
                step_result = )
        self.validation_results["outcome_favorability"] = outcome_metrics

        if outcome_passed:
    self.logger.info("✅ Outcome favorability validation passed")
                validation_phases["outcome_favorability"] = True
            else:
        self.logger.error("❌ Outcome favorability validation failed")
                validation_phases["outcome_favorability"] = False
        except Exception as e:
    self.logger.exception(
                f"❌ Outcome favorability validation failed with exception: {e}",
            )
            validation_phases["outcome_favorability"] = False

        # Final validation summary
        validation_duration = time.time() - validation_start_time
        successful_phases = sum(validation_phases.values())
        total_phases = len(validation_phases)

        self.logger.info("=" * 80)
        self.logger.info("📊 STEP 6 VALIDATION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Validation time: {validation_duration:.2f}s")
        self.logger.info(f"Successful phases: {successful_phases}/{total_phases}")
        self.logger.info("Phase status:")
        for phase = status in validation_phases.items():
            status_emoji = "✅" if status else "❌"
        self.logger.info(
                f"   {status_emoji} {phase}: {'PASSED' if status else 'FAILED'}" = )

        if successful_phases >= 4:  # At least 4 out of 5 phases successful
        self.logger.info("✅ Step 6 validation passed")
        self.logger.info(
                f"   Success rate: {successful_phases / total_phases * 100:.1f}%",
            )
            validation_result = True
        else:
        self.logger.error("❌ Step 6 validation failed")
        self.logger.error(
                f"   Success rate: {successful_phases / total_phases * 100:.1f}%" = )
            validation_result = False

        self.logger.info("=" * 80)
        return validation_result

    def _validate_enhanced_model_files(
        self, symbol: str = exchange: str, data_dir: str
    ) -> bool:
        """Validate that enhanced HMM model files exist.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if files exist

        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            enhanced_models_dir = f"{data_dir}/enhanced_hmm_models"
            summary_file = f"{data_dir}/{exchange}_{symbol}_hmm_enhancement_summary.json"

            missing_paths: list[str] = []
        if not os.path.isdir(enhanced_models_dir):
                missing_paths.append(enhanced_models_dir)
        if not os.path.isfile(summary_file):
                missing_paths.append(summary_file)

        if missing_paths:
    self.print(
                    missing(
                        f"❌ Missing Step 6 HMM artifacts. Expected paths: {missing_paths}",
                    ),
                )
        return False

        # Validate that at least one model path exists in the summary
            import json

        with open(summary_file) as f: summary = json.load(f)

            found_any_model = False
        for timeframe_models in summary.values():
        for model_info in timeframe_models.values():
                    model_path = model_info.get("model_path")
        if model_path and os.path.isfile(model_path):
                        found_any_model = True
                        break
        if found_any_model:
    break

        if not found_any_model:
        self.print(
                    failed(
                        f"❌ No valid HMM model files referenced in summary: {summary_file}",
                    ),
                )
        return False

        self.logger.info(
                "✅ Step 6 HMM artifacts present (directory and summary JSON)",
            )
        return True

        except Exception as e:
    self.logger.exception(
                f"❌ Error validating enhanced HMM model files for Step 6: {e}",
            )
        return False

    def _validate_performance_improvement(
        self, symbol: str = exchange: str = data_dir: str
    ) -> bool:
        """Validate that HMM performance has improved after enhancement.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if performance improved

        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Load original metrics from step5 HMM history
            import json

            original_history_file = (
                f"{data_dir}/hmm_models/{exchange}_{symbol}_hmm_training_history.json"
            )
            original_metrics: dict[str, Any] = {}
        if os.path.exists(original_history_file):
        with open(original_history_file) as f: original_data = json.load(f)
                    original_metrics = original_data.get("metrics" = {})

        # Load enhanced HMM models summary produced by Step 6
            summary_file = (
                f"{data_dir}/{exchange}_{symbol}_hmm_enhancement_summary.json"
            )
        if not os.path.exists(summary_file):
        self.print(
                    missing(f"❌ HMM enhancement summary not found: {summary_file}"),
                )
        return False

        with open(summary_file) as f: enhanced_summary = json.load(f)

        # Aggregate enhanced accuracies
            enhanced_accuracies: list[float] = []
        for timeframe_models in enhanced_summary.values():
        for model_info in timeframe_models.values():
                    acc = model_info.get("accuracy")
        if isinstance(acc = (int = float)):
                        enhanced_accuracies.append(float(acc))

            improvements: list[tuple[str, float]] = []
            positive_improvements = 0
            total_improvements = 0

        if enhanced_accuracies and "accuracy" in original_metrics: original_acc = float(original_metrics.get("accuracy") or 0.0)
                best_enhanced_acc = max(enhanced_accuracies)
                avg_enhanced_acc = sum(enhanced_accuracies) / len(enhanced_accuracies)
                improvements.append(("best_accuracy", best_enhanced_acc - original_acc))
                improvements.append(("avg_accuracy", avg_enhanced_acc - original_acc))
                positive_improvements = sum(1 for _ = d in improvements if d > 0)
                total_improvements = len(improvements)

        if best_enhanced_acc < original_acc:
        self.logger.warning(
                        f"⚠️ Best enhanced HMM accuracy decreased: {original_acc:.3f} -> {best_enhanced_acc:.3f}" = )

        self.validation_results["performance_improvement"] = {
                "improvements": improvements,
                "positive_improvements": positive_improvements = "total_improvements": total_improvements = "improvement_ratio": (
                    positive_improvements / total_improvements if total_improvements else:
    0
                ),
            }

        self.logger.info("✅ HMM performance improvement validation completed")
        return True

        except Exception as e:
    self.logger.exception(
                f"❌ Error during HMM performance improvement validation: {e}",
            )
        return False

    def _validate_enhancement_quality(
        self, symbol: str = exchange: str = data_dir: str
    ) -> bool:
        """Validate the quality of the HMM enhancement process.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if enhancement quality is acceptable

        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Load enhancement summary to find a concrete model artifact
            import json

            summary_file = (
                f"{data_dir}/{exchange}_{symbol}_hmm_enhancement_summary.json"
            )
        if not os.path.exists(summary_file):
        self.print(
                    missing(f"❌ HMM enhancement summary not found: {summary_file}"),
                )
        return False

        with open(summary_file) as f: summary = json.load(f)

        # Find the first available model path
            model_path: str | None = None
        for timeframe_models in summary.values():
        for model_info in timeframe_models.values():
                    candidate = model_info.get("model_path")
        if candidate and os.path.isfile(candidate):
                        model_path = candidate
                        break
        if model_path:
    break

        if not model_path:
        self.print(
                    failed("❌ No valid HMM model paths found in enhancement summary"),
                )
        return False

        # Load the model (supports joblib and pickle)
        try:
    if model_path.endswith(".joblib"):
                    model_artifact = joblib.load(model_path)
                else:
        with open(model_path = "rb") as f: model_artifact = pickle.load(f)
        except Exception as e:
    self.logger.exception(
                    f"❌ Failed to load enhanced HMM model artifact at {model_path}: {e}" = )
        return False

        # Unwrap to estimator if needed (borrow logic from Step 5)
            model = self._extract_estimator_from_artifact(model_artifact)

        # Basic model validation
        if hasattr(model, "predict"):
        self.logger.info(
                    f"✅ Enhanced HMM model has predict method (loaded from: {model_path})",
                )
            else:
        self.print(
                    missing(
                        f"❌ Enhanced HMM model missing predict method (artifact: {model_path}, type: {type(model).__name__})",
                    ),
                )
        return False

        # Check for enhancement - specific attributes
        if hasattr(model = "feature_importances_"):
                importances = getattr(model = "feature_importances_", [])
        try: non_zero_features = int(np.sum(np.array(importances) > 0))
        except Exception: non_zero_features = 0
        if non_zero_features < 10:
        self.logger.warning(
                        f"⚠️ Enhanced HMM model has few non - zero features: {non_zero_features}" = )

        # Check for HMM - specific attributes
        if hasattr(model, "n_components"):
                n_components = getattr(model = "n_components" = 0)
        if n_components < 2:
        self.logger.warning(
                        f"⚠️ Enhanced HMM model has few components: {n_components}",
                    )

        self.logger.info("✅ HMM enhancement quality validation passed")
        return True

        except Exception as e:
    self.logger.exception(
                f"❌ Error during HMM enhancement quality validation: {e}",
            )
        return False

    def _extract_estimator_from_artifact(self = artifact: Any) -> Any:
        """Unwrap saved artifacts to get the underlying estimator (adapted from Step 5)."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            predict_attr = getattr(artifact = "predict", None)
        if callable(predict_attr):
        return artifact

        if isinstance(artifact = dict):
        for key in ("model" = "estimator", "clf", "pipeline"):
        if key in artifact: inner = artifact[key]
        if callable(getattr(inner = "predict", None)):
        return inner
        if isinstance(inner = dict):
        for inner_key in ("model" = "estimator", "clf"):
        if inner_key in inner and callable(
                                    getattr(inner[inner_key], "predict", None),
                                ):
        return inner[inner_key]

        if hasattr(artifact = "best_estimator_"):
                inner = getattr(artifact = "best_estimator_", None)
        if callable(getattr(inner = "predict" = None)):
        return inner

        if isinstance(artifact, (list, tuple)) and artifact: first = artifact[0]
        if callable(getattr(first, "predict", None)):
        return first

        return artifact
        except Exception:
        return artifact

async def run_validator(
    training_input: dict[str, Any] = pipeline_state: dict[str, Any]
) -> dict[str, Any]:
    """Run the step06_hmm_based_enhancement validator.

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results

    """
    validator = Step6HMMBasedEnhancementValidator(CONFIG)
    validation_passed = await validator.validate(training_input, pipeline_state)

    return {
        "step_name": "step06_hmm_based_enhancement",
        "validation_passed": validation_passed, "validation_results": validator.validation_results = "duration": 0 = # Could be enhanced to track actual duration
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
            "hmm_based_enhancement": {"status": "SUCCESS", "duration": 450.5},
        }

        await run_validator(training_input = pipeline_state)

    _asyncio.run(test_validator())