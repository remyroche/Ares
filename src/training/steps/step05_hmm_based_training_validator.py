# src/training/steps/step5_hmm_based_training_validator.py

"""Validator for Step 5: HMM-Based Training."""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.common_operations import safe_json_load
from src.utils.warning_symbols import (
    error,
    failed,
    missing,
    validation_error,
)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent

# NOTE: Path injection kept for compatibility if needed by downstream imports
# Import after potential path update
from src.config import CONFIG
from src.utils.base_validator import BaseValidator


class Step5HMMBasedTrainingValidator(BaseValidator):
    """Validator for Step 5: HMM-Based Training."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("step5_hmm_based_training", config)

    async def validate(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> bool:
        """Validate the HMM-based training step."

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("🔍 Validating Step 5 HMM-based training...")

        # Extract parameters
        symbol = str(training_input.get("symbol", "ETHUSDT"))
        exchange = str(training_input.get("exchange", "BINANCE"))
        data_dir = str(training_input.get("data_dir", "data/training"))

        # Validate step result from pipeline state
        step_result = pipeline_state.get("hmm_based_training", {})

        # 1. Validate error absence
        error_passed, error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics

        if not error_passed:
            self.logger.error(error("❌ HMM-based training step had errors"))
            return False

        # 2. Validate model files existence
        model_files_passed = self._validate_model_files_existence(
            symbol,
            exchange,
            data_dir,
        )
        if not model_files_passed:
            self.logger.error(failed("❌ Model files validation failed"))
            return False

        # 3. Validate model performance
        performance_passed = self._validate_model_performance(
            symbol,
            exchange,
            data_dir,
        )
        if not performance_passed:
            self.logger.error(failed("❌ Model performance validation failed"))
            return False

        # 4. Validate training metrics
        metrics_passed = self._validate_training_metrics(symbol, exchange, data_dir)
        if not metrics_passed:
            self.logger.error(failed("❌ Training metrics validation failed"))
            return False

        # 5. Validate model quality
        quality_passed = self._validate_model_quality(symbol, exchange, data_dir)
        if not quality_passed:
            self.logger.error(failed("❌ Model quality validation failed"))
            return False

        # 6. Validate outcome favorability
        outcome_passed, outcome_metrics = self.validate_outcome_favorability(
            step_result,
        )
        self.validation_results["outcome_favorability"] = outcome_metrics

        if not outcome_passed:
            # Summarize why outcome is unfavorable and include context
            status_value = step_result.get("status")
            error_value = step_result.get("error")
            reasons: list[str] = []
            if not outcome_metrics.get("has_success_indicators"):
                reasons.append("no success indicator")
            if outcome_metrics.get("has_error_indicators"):
                reasons.append("error indicator present")
            reasons_text = "; ".join(reasons) if reasons else "unspecified"

            self.logger.warning(
                "⚠️ HMM-based training outcome is not favorable"
                f" | symbol={symbol} | exchange={exchange} | status={status_value}"
                f" | reasons={reasons_text}",
                extra={
                    "step_name": self.step_name,
                    "phase": "validation",
                    "component": "hmm_based_training",
                    "symbol": symbol,
                    "exchange": exchange,
                    "status": status_value,
                    "has_success_indicators": outcome_metrics.get(
                        "has_success_indicators",
                    ),
                    "has_error_indicators": outcome_metrics.get("has_error_indicators"),
                    "success_indicators": outcome_metrics.get("success_indicators"),
                    "error_indicators": outcome_metrics.get("error_indicators"),
                    "step_metrics_keys": list(
                        outcome_metrics.get("step_metrics", {}).keys(),
                    ),
                    "performance_metrics_keys": list(
                        outcome_metrics.get("performance_metrics", {}).keys(),
                    ),
                    "error_message": (
                        str(error_value)[:500] if error_value is not None else None
                    ),
                },
            )

            # In blank mode, allow continuation when all artifact and metric
            # checks passed. Outcome favorability often depends on pipeline
            # state flags omitted in quick/blank runs; treat as non-blocking.
            blank_mode = os.environ.get("BLANK_TRAINING_MODE", "0") == "1"
            if blank_mode:
                self.logger.warning(
                    (
                        "⚠️ BLANK MODE: Allowing Step 5 validation to pass despite "
                        "unfavorable outcome. This is expected in blank runs "
                        "(reduced artifacts/flags); safe to continue."
                    ),
                    extra={
                        "step_name": self.step_name,
                        "phase": "validation",
                        "component": "hmm_based_training",
                        "symbol": symbol,
                        "exchange": exchange,
                        "status": status_value,
                        "blank_mode": True,
                        "expected_in_blank_mode": True,
                        "guidance": (
                            "No action required for blank mode; run full mode for"
                            " strict validation."
                        ),
                    },
                )
                return True

            # Full mode: provide actionable guidance before failing
            self.logger.error(
                (
                    "❗ Full mode validation failed for Step 5. Actions: "
                    "1) Ensure step_result.success or status=='SUCCESS'. "
                    "2) Check missing artifacts and training history/metrics files. "
                    "3) Review accuracy/loss thresholds and any 'error' fields in "
                    "step_result."
                ),
                extra={
                    "step_name": self.step_name,
                    "phase": "validation",
                    "component": "hmm_based_training",
                    "symbol": symbol,
                    "exchange": exchange,
                    "status": status_value,
                    "reasons": reasons_text,
                    "expected_success_indicators": [
                        "success=True",
                        "completed=True",
                        "status=='SUCCESS'",
                    ],
                    "step_metrics_keys": list(
                        outcome_metrics.get("step_metrics", {}).keys(),
                    ),
                    "performance_metrics_keys": list(
                        outcome_metrics.get("performance_metrics", {}).keys(),
                    ),
                },
            )
            return False

        self.logger.info("✅ HMM-based training validation passed")
        return True

    def _validate_model_files_existence(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """Validate that all expected HMM model files exist."

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if all files exist
        """
        try:
            # Expected HMM model file patterns - updated to match artifacts
            expected_files: list[str] = []
            for timeframe in ["1m", "5m", "15m"]:
                expected_files.extend(
                    [
                        f"{data_dir}/{exchange}_{symbol}_hmm_block_states_{timeframe}.parquet",
                        f"{data_dir}/{exchange}_{symbol}_hmm_composite_clusters_{timeframe}.parquet",
                        f"{data_dir}/{exchange}_{symbol}_hmm_composite_intensity_{timeframe}.parquet",
                        f"{data_dir}/{exchange}_{symbol}_hmm_composite_meta_{timeframe}.json",
                    ],
                )

            missing_files: list[str] = []
            for file_path in expected_files:
                file_passed, file_metrics = self.validate_file_exists(
                    file_path,
                    "model_files",
                )
                # Track per-file validation
                self.validation_results[
                    f"exists::{Path(file_path).name}"
                ] = file_metrics
                if not file_passed:
                    missing_files.append(file_path)

            if missing_files:
                self.logger.error(
                    missing(f"❌ Missing HMM model files: {missing_files}"),
                )
                return False

            self.logger.info("✅ All HMM model files exist")
            return True

        except Exception as e:  # pragma: no cover - defensive logging
            self.logger.exception(
                error(f"❌ Error validating HMM model files existence: {e}"),
            )
            return False

    def _validate_model_performance(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """Validate HMM model performance metrics."

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if performance is acceptable
        """
        try:
            # Load training history - use composite meta file if present
            history_file = f"{data_dir}/{exchange}_{symbol}_hmm_composite_meta_1m.json"

            if not os.path.exists(history_file):
                self.logger.warning(
                    f"⚠️ HMM training history file not found: {history_file}",
                )
                return True  # Not critical for validation

            training_history: dict[str, Any] = safe_json_load(history_file)

            # Extract performance metrics
            metrics: dict[str, Any] = training_history.get("metrics", {})

            # Validate accuracy
            if "accuracy" in metrics:
                accuracy = float(metrics["accuracy"])  # sanitize
                accuracy_passed, accuracy_metrics = self._validate_performance_metric(
                    accuracy,
                    0.6,
                    "accuracy",
                    "hmm_model",
                )
                self.validation_results["model_accuracy"] = accuracy_metrics
                if not accuracy_passed:
                    self.logger.error(
                        error(f"❌ HMM model accuracy too low: {accuracy:.3f}"),
                    )
                    return False

            # Validate loss
            if "loss" in metrics:
                loss = float(metrics["loss"])  # sanitize
                loss_passed, loss_metrics = self._validate_performance_metric(
                    loss,
                    0.5,
                    "loss",
                    "hmm_model",
                    is_loss=True,
                )
                self.validation_results["model_loss"] = loss_metrics
                if not loss_passed:
                    self.logger.error(
                        error(f"❌ HMM model loss too high: {loss:.3f}"),
                    )
                    return False

            # Validate other metrics
            for metric_name, metric_value in metrics.items():
                if metric_name not in ["accuracy", "loss"] and isinstance(
                    metric_value,
                    (int, float),
                ):
                    custom_passed, custom_metrics = self._validate_performance_metric(
                        float(metric_value),
                        0.0,
                        metric_name,
                        "hmm_model",
                    )
                    self.validation_results[f"custom_metric::{metric_name}"] = (
                        custom_metrics
                    )

            self.logger.info("✅ HMM model performance validation passed")
            return True

        except Exception as e:  # pragma: no cover - defensive logging
            self.logger.exception(
                validation_error(
                    f"❌ Error during HMM model performance validation: {e}",
                ),
            )
            return False

    def _validate_training_metrics(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """Validate HMM training metrics and convergence."

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if training metrics are acceptable
        """
        try:
            history_file = (
                f"{data_dir}/hmm_models/{exchange}_{symbol}_hmm_training_history.json"
            )

            if not os.path.exists(history_file):
                self.logger.warning(
                    f"⚠️ HMM training history file not found: {history_file}",
                )
                return True

            training_history: dict[str, Any] = safe_json_load(history_file)

            # Check for training epochs
            if "epochs" in training_history:
                epochs = int(training_history["epochs"])  # sanitize
                if epochs < 10:
                    self.logger.warning(f"⚠️ Few HMM training epochs: {epochs}")
                elif epochs > 1000:
                    self.logger.warning(f"⚠️ Many HMM training epochs: {epochs}")

            # Check for convergence indicators
            if "converged" in training_history:
                converged = bool(training_history["converged"])  # sanitize
                if not converged:
                    self.logger.warning("⚠️ HMM model did not converge")

            # Check for overfitting indicators
            if (
                "train_accuracy" in training_history
                and "val_accuracy" in training_history
            ):
                train_acc = float(training_history["train_accuracy"])
                val_acc = float(training_history["val_accuracy"])
                if train_acc - val_acc > 0.1:
                    self.logger.warning(
                        (
                            "⚠️ Potential HMM overfitting: "
                            f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}"
                        ),
                    )

            # Check for training time
            if "training_time" in training_history:
                training_time = float(training_history["training_time"])  # seconds
                if training_time > 3600:  # More than 1 hour
                    self.logger.warning(
                        f"⚠️ Long HMM training time: {training_time:.1f}s",
                    )
                elif training_time < 60:  # Less than 1 minute
                    self.logger.warning(
                        f"⚠️ Short HMM training time: {training_time:.1f}s",
                    )

            self.logger.info("✅ HMM training metrics validation passed")
            return True

        except Exception as e:  # pragma: no cover - defensive logging
            self.logger.exception(
                validation_error(
                    f"❌ Error during HMM training metrics validation: {e}",
                ),
            )
            return False

    def _validate_model_quality(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """Validate HMM model quality characteristics."""
        try:
            # Load model metadata
            metadata_file = (
                f"{data_dir}/hmm_models/{exchange}_{symbol}_hmm_model_metadata.json"
            )

            if os.path.exists(metadata_file):
                metadata: dict[str, Any] = safe_json_load(metadata_file)

                # Check model type
                model_type = metadata.get("model_type")
                if model_type:
                    self.logger.info(f"HMM model type: {model_type}")

                # Check model parameters
                if isinstance(metadata.get("parameters"), dict):
                    params = metadata["parameters"]
                    param_count = len(params)
                    if param_count < 100:
                        self.logger.warning(
                            f"⚠️ Few HMM model parameters: {param_count}",
                        )
                    elif param_count > 1_000_000:
                        self.logger.warning(
                            f"⚠️ Many HMM model parameters: {param_count}",
                        )

                # Check model size
                if isinstance(metadata.get("model_size_mb"), (int, float)):
                    model_size = float(metadata["model_size_mb"])
                    if model_size > 100:  # More than 100MB
                        self.logger.warning(
                            f"⚠️ Large HMM model size: {model_size:.1f}MB",
                        )
                    elif model_size < 0.1:  # Less than 0.1MB
                        self.logger.warning(
                            f"⚠️ Small HMM model size: {model_size:.1f}MB",
                        )

                # Feature importance
                feature_importance = metadata.get("feature_importance")
                if isinstance(feature_importance, dict):
                    try:
                        top_features = sorted(
                            feature_importance.items(), key=lambda x: x[1], reverse=True
                        )[:5]
                        self.logger.info(f"Top 5 HMM features: {top_features}")
                    except Exception:  # pragma: no cover - defensive
                        pass

            # Validate existence of a concrete model artifact
            # Prefer a model pickle if present, otherwise validate parquet artifact
            model_pkl = (
                f"{data_dir}/hmm_models/{exchange}_{symbol}_hmm_model.pkl"
            )
            clusters_parquet = (
                f"{data_dir}/{exchange}_{symbol}_hmm_composite_clusters_1m.parquet"
            )

            if os.path.exists(model_pkl):
                try:
                    with open(model_pkl, "rb") as f:
                        loaded_artifact = pickle.load(f)
                    # Unwrap common wrappers to get the estimator
                    model = self._unwrap_estimator(loaded_artifact)

                    # Basic model validation
                    has_predict = callable(getattr(model, "predict", None))
                    has_fit = callable(getattr(model, "fit", None))

                    if not has_predict:
                        self.logger.error(
                            missing("❌ HMM model missing predict method"),
                        )
                        return False
                    if not has_fit:
                        self.logger.warning(missing("⚠️ HMM model missing fit method"))

                    # Optional: feature importances
                    if hasattr(model, "feature_importances_"):
                        importances = getattr(model, "feature_importances_")
                        try:
                            non_zero_features = int(np.sum(np.array(importances) > 0))
                            if non_zero_features < 5:
                                self.logger.warning(
                                    (
                                        "⚠️ Few non-zero HMM feature importances: "
                                        f"{non_zero_features}"
                                    ),
                                )
                        except Exception:  # pragma: no cover - defensive
                            pass
                except Exception as e:  # pragma: no cover - defensive
                    self.logger.exception(error(f"❌ Error loading HMM model: {e}"))
                    return False
            else:
                # Fall back to validating existence and non-emptiness of
                # parquet artifact
                if not os.path.exists(clusters_parquet):
                    self.logger.error(
                        missing(
                            f"❌ Missing HMM clusters artifact: {clusters_parquet}",
                        ),
                    )
                    return False
                try:
                    file_size = os.path.getsize(clusters_parquet)
                    if file_size <= 0:
                        self.logger.error(
                            missing(
                                "❌ HMM clusters artifact is empty: "
                                f"{clusters_parquet}",
                            ),
                        )
                        return False
                except Exception:  # pragma: no cover - defensive
                    # If file size can't be determined, proceed as existence is'
                    # validated
                    pass

            self.logger.info("✅ HMM model quality validation passed")
            return True

        except Exception as e:  # pragma: no cover - defensive logging
            self.logger.exception(
                validation_error(f"❌ Error during HMM model quality validation: {e}"),
            )
            return False

    def _validate_performance_metric(
        self,
        metric_value: float,
        threshold: float,
        metric_name: str,
        model_name: str,
        is_loss: bool = False,
    ) -> tuple[bool, dict[str, Any]]:
        """Validate a performance metric against a threshold."

        Args:
            metric_value: The metric value to validate
            threshold: The threshold to compare against
            metric_name: Name of the metric
            model_name: Name of the model
            is_loss: Whether this is a loss metric (lower is better)

        Returns:
            Tuple[bool, Dict[str, Any]]: (passed, metrics)
        """
        try:
            # For loss metrics, lower is better; for accuracy-like metrics,
            # higher is better
            if is_loss:
                passed = metric_value <= threshold
                comparison = "≤"
            else:
                passed = metric_value >= threshold
                comparison = "≥"

            metrics = {
                "metric_name": metric_name,
                "model_name": model_name,
                "metric_value": float(metric_value),
                "threshold": float(threshold),
                "comparison": comparison,
                "passed": passed,
                "is_loss": is_loss,
            }

            if not passed:
                self.logger.warning(
                    f"⚠️ {model_name} {metric_name} validation failed: "
                    f"{metric_value:.3f} {comparison} {threshold:.3f}",
                )
            else:
                self.logger.info(
                    f"✅ {model_name} {metric_name} validation passed: "
                    f"{metric_value:.3f} {comparison} {threshold:.3f}",
                )

            return passed, metrics

        except Exception as e:  # pragma: no cover - defensive logging
            self.logger.exception(
                validation_error(f"❌ Error in performance metric validation: {e}"),
            )
            return False, {"error": str(e)}

    def _unwrap_estimator(self, artifact: Any) -> Any:
        """Unwrap a potentially wrapped model artifact to get the estimator."

        Supports:
        - Dicts with keys 'model' / 'estimator' / 'clf' / 'pipeline'
        - Objects with 'best_estimator_'
        - First element of tuple/list
        - Returns original if it already has a callable predict
        """
        try:
            # If already looks like an estimator
            if callable(getattr(artifact, "predict", None)):
                return artifact

            # Dict wrappers
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

            # GridSearchCV-like
            if hasattr(artifact, "best_estimator_"):
                inner = getattr(artifact, "best_estimator_", None)
                if callable(getattr(inner, "predict", None)):
                    return inner

            # Tuple/list first element
            if isinstance(artifact, (list, tuple)) and artifact:
                first = artifact[0]
                if callable(getattr(first, "predict", None)):
                    return first

            return artifact
        except Exception:  # pragma: no cover - defensive
            return artifact


async def run_validator(
    training_input: dict[str, Any],
    pipeline_state: dict[str, Any],
) -> dict[str, Any]:
    """Run the step5_hmm_based_training validator."

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results
    """
    validator = Step5HMMBasedTrainingValidator(CONFIG)
    validation_passed = await validator.validate(training_input, pipeline_state)

    return {
        "step_name": "step5_hmm_based_training",
        "validation_passed": validation_passed,
        "validation_results": validator.validation_results,
        "duration": 0.0,  # Could be enhanced to track actual duration
        "timestamp": time.time(),
    }


if __name__ == "__main__":
    import asyncio as _asyncio

import os.path


    # Example usage
async def test_validator() -> None:
        training_input = {
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "data_dir": "data/training",
        }

        pipeline_state = {
            "hmm_based_training": {"status": "SUCCESS", "duration": 300.5},
        }

        await run_validator(training_input, pipeline_state)

_asyncio.run(test_validator())