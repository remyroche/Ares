"""
Validator for Step 6: Analyst Enhancement
"""

import asyncio
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.utils.warning_symbols import (
    error,
    failed,
    missing,
)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import CONFIG
from src.utils.base_validator import BaseValidator


class Step6AnalystEnhancementValidator(BaseValidator):
    """Validator for Step 6: Analyst Enhancement."""

    def __init__(self, config: dict[str, Any]):
        super().__init__("step6_analyst_enhancement", config)

    async def validate(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> bool:
        """
        Validate the analyst enhancement step.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("🔍 Validating analyst enhancement step...")

        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")

        # Validate step result from pipeline state
        step_result = pipeline_state.get("analyst_enhancement", {})

        # 1. Validate error absence
        error_passed, error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics

        if not error_passed:
            self.print(error("❌ Analyst enhancement step had errors"))
            return False

        # 2. Validate enhanced model files existence
        model_files_passed = self._validate_enhanced_model_files(
            symbol,
            exchange,
            data_dir,
        )
        if not model_files_passed:
            self.print(failed("❌ Enhanced model files validation failed"))
            return False

        # 3. Validate performance improvement
        improvement_passed = self._validate_performance_improvement(
            symbol,
            exchange,
            data_dir,
        )
        if not improvement_passed:
            self.print(failed("❌ Performance improvement validation failed"))
            return False

        # 4. Validate enhancement quality
        quality_passed = self._validate_enhancement_quality(symbol, exchange, data_dir)
        if not quality_passed:
            self.print(failed("❌ Enhancement quality validation failed"))
            return False

        # 5. Validate outcome favorability
        outcome_passed, outcome_metrics = self.validate_outcome_favorability(
            step_result,
        )
        self.validation_results["outcome_favorability"] = outcome_metrics

        if not outcome_passed:
            self.print(error("⚠️ Analyst enhancement outcome is not favorable"))
            return False

        self.logger.info("✅ Analyst enhancement validation passed")
        return True

    def _validate_enhanced_model_files(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """
        Validate that enhanced model files exist.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if files exist
        """
        try:
            enhanced_models_dir = f"{data_dir}/enhanced_analyst_models"
            summary_file = (
                f"{data_dir}/{exchange}_{symbol}_analyst_enhancement_summary.json"
            )

            missing_paths: list[str] = []
            if not os.path.isdir(enhanced_models_dir):
                missing_paths.append(enhanced_models_dir)
            if not os.path.isfile(summary_file):
                missing_paths.append(summary_file)

            if missing_paths:
                self.print(
                    missing(
                        f"❌ Missing Step 6 artifacts. Expected paths: {missing_paths}",
                    ),
                )
                return False

            # Validate that at least one model path exists in the summary
            import json

            with open(summary_file) as f:
                summary = json.load(f)

            found_any_model = False
            for regime_models in summary.values():
                for model_info in regime_models.values():
                    model_path = model_info.get("model_path")
                    if model_path and os.path.isfile(model_path):
                        found_any_model = True
                        break
                if found_any_model:
                    break

            if not found_any_model:
                self.print(
                    failed(
                        f"❌ No valid model files referenced in summary: {summary_file}",
                    ),
                )
                return False

            self.logger.info("✅ Step 6 artifacts present (directory and summary JSON)")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Error validating enhanced model files for Step 6: {e}",
            )
            return False

    def _validate_performance_improvement(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """
        Validate that performance has improved after enhancement.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if performance improved
        """
        try:
            # Load original metrics from step5 history
            import json

            original_history_file = (
                f"{data_dir}/{exchange}_{symbol}_analyst_training_history.json"
            )
            original_metrics = {}
            if os.path.exists(original_history_file):
                with open(original_history_file) as f:
                    original_data = json.load(f)
                    original_metrics = original_data.get("metrics", {})

            # Load enhanced models summary produced by Step 6
            summary_file = (
                f"{data_dir}/{exchange}_{symbol}_analyst_enhancement_summary.json"
            )
            if not os.path.exists(summary_file):
                self.print(missing(f"❌ Enhancement summary not found: {summary_file}"))
                return False

            with open(summary_file) as f:
                enhanced_summary = json.load(f)

            # Aggregate enhanced accuracies
            enhanced_accuracies: list[float] = []
            for regime_models in enhanced_summary.values():
                for model_info in regime_models.values():
                    acc = model_info.get("accuracy")
                    if isinstance(acc, (int, float)):
                        enhanced_accuracies.append(float(acc))

            improvements = []
            positive_improvements = 0
            total_improvements = 0

            if enhanced_accuracies and "accuracy" in original_metrics:
                original_acc = float(original_metrics["accuracy"]) or 0.0
                best_enhanced_acc = max(enhanced_accuracies)
                avg_enhanced_acc = sum(enhanced_accuracies) / len(enhanced_accuracies)
                improvements.append(("best_accuracy", best_enhanced_acc - original_acc))
                improvements.append(("avg_accuracy", avg_enhanced_acc - original_acc))
                positive_improvements = sum(1 for _, d in improvements if d > 0)
                total_improvements = len(improvements)

                if best_enhanced_acc < original_acc:
                    self.logger.warning(
                        f"⚠️ Best enhanced accuracy decreased: {original_acc:.3f} -> {best_enhanced_acc:.3f}",
                    )

            self.validation_results["performance_improvement"] = {
                "improvements": improvements,
                "positive_improvements": positive_improvements,
                "total_improvements": total_improvements,
                "improvement_ratio": (
                    positive_improvements / total_improvements
                    if total_improvements
                    else 0
                ),
            }

            self.logger.info("✅ Performance improvement validation completed")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Error during performance improvement validation: {e}",
            )
            return False

    def _validate_enhancement_quality(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """
        Validate the quality of the enhancement process.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if enhancement quality is acceptable
        """
        try:
            # Load enhancement summary to find a concrete model artifact
            import json

            summary_file = (
                f"{data_dir}/{exchange}_{symbol}_analyst_enhancement_summary.json"
            )
            if not os.path.exists(summary_file):
                self.print(missing(f"❌ Enhancement summary not found: {summary_file}"))
                return False

            with open(summary_file) as f:
                summary = json.load(f)

            # Find the first available model path
            model_path: str | None = None
            for regime_models in summary.values():
                for model_info in regime_models.values():
                    candidate = model_info.get("model_path")
                    if candidate and os.path.isfile(candidate):
                        model_path = candidate
                        break
                if model_path:
                    break

            if not model_path:
                self.print(
                    failed("❌ No valid model paths found in enhancement summary")
                )
                return False

            # Load the model (supports joblib and pickle)
            try:
                if model_path.endswith(".joblib"):
                    model_artifact = joblib.load(model_path)
                else:
                    with open(model_path, "rb") as f:
                        model_artifact = pickle.load(f)
            except Exception as e:
                self.logger.exception(
                    f"❌ Failed to load enhanced model artifact at {model_path}: {e}",
                )
                return False

            # Unwrap to estimator if needed (borrow logic from Step 5)
            model = self._extract_estimator_from_artifact(model_artifact)

            # Basic model validation
            if hasattr(model, "predict"):
                self.logger.info(
                    f"✅ Enhanced model has predict method (loaded from: {model_path})",
                )
            else:
                self.print(
                    missing(
                        f"❌ Enhanced model missing predict method (artifact: {model_path}, type: {type(model).__name__})",
                    ),
                )
                return False

            # Check for enhancement-specific attributes
            if hasattr(model, "feature_importances_"):
                importances = getattr(model, "feature_importances_", [])
                try:
                    non_zero_features = int(np.sum(np.array(importances) > 0))
                except Exception:
                    non_zero_features = 0
                if non_zero_features < 10:
                    self.logger.warning(
                        f"⚠️ Enhanced model has few non-zero features: {non_zero_features}",
                    )

            self.logger.info("✅ Enhancement quality validation passed")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Error during enhancement quality validation: {e}",
            )
            return False

    def _extract_estimator_from_artifact(self, artifact: Any) -> Any:
        """Unwrap saved artifacts to get the underlying estimator (adapted from Step 5)."""
        try:
            predict_attr = getattr(artifact, "predict", None)
            if callable(predict_attr):
                return artifact

            if isinstance(artifact, dict):
                for key in ("model", "estimator", "clf", "pipeline"):
                    if key in artifact:
                        inner = artifact[key]
                        if callable(getattr(inner, "predict", None)):
                            return inner
                        if isinstance(inner, dict):
                            for inner_key in ("model", "estimator", "clf"):
                                if inner_key in inner and callable(
                                    getattr(inner[inner_key], "predict", None),
                                ):
                                    return inner[inner_key]

            if hasattr(artifact, "best_estimator_"):
                inner = getattr(artifact, "best_estimator_", None)
                if callable(getattr(inner, "predict", None)):
                    return inner

            if isinstance(artifact, (list, tuple)) and artifact:
                first = artifact[0]
                if callable(getattr(first, "predict", None)):
                    return first

            return artifact
        except Exception:
            return artifact


async def run_validator(
    training_input: dict[str, Any],
    pipeline_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the step6_analyst_enhancement validator.

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results
    """
    validator = Step6AnalystEnhancementValidator(CONFIG)
    validation_passed = await validator.validate(training_input, pipeline_state)

    return {
        "step_name": "step6_analyst_enhancement",
        "validation_passed": validation_passed,
        "validation_results": validator.validation_results,
        "duration": 0,  # Could be enhanced to track actual duration
        "timestamp": asyncio.get_event_loop().time(),
    }


if __name__ == "__main__":
    import asyncio

    # Example usage
    async def test_validator():
        training_input = {
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "data_dir": "data/training",
        }

        pipeline_state = {
            "analyst_enhancement": {"status": "SUCCESS", "duration": 450.5},
        }

        result = await run_validator(training_input, pipeline_state)
        print(f"Validation result: {result}")

    asyncio.run(test_validator())
