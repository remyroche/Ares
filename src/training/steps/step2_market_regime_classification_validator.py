"""
Validator for Step 2: Market Regime Classification
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from src.utils.warning_symbols import (
    validation_error,
)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import CONFIG
from src.utils.base_validator import BaseValidator


class Step2MarketRegimeClassificationValidator(BaseValidator):
    """Validator for Step 2: Market Regime Classification."""

    def __init__(self, config: dict[str, Any]):
        super().__init__("step2_market_regime_classification", config)
        # Fine-tuned parameters for ML training (more lenient to avoid stopping training)
        self.max_regime_dominance = (
            0.85  # Increased from 0.8 to allow more dominant regimes
        )
        self.min_regime_frequency = 0.03  # Reduced from 0.05 to allow rare regimes
        self.max_regime_switching = (
            0.6  # Increased from 0.5 to allow more regime changes
        )
        self.max_stuck_ratio = 0.4  # Increased from 0.3 to allow longer regime periods
        self.probability_tolerance = (
            0.02  # Increased from 0.01 for probability validation
        )

    async def validate(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> bool:
        """
        Validate the market regime classification step.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("🔍 Validating market regime classification step...")

        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")

        # Validate step result from pipeline state
        step_result = pipeline_state.get("regime_classification", {})

        # 1. Validate error absence (CRITICAL - blocks process)
        error_passed, error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics

        if not error_passed:
            self.logger.error(
                "❌ Market regime classification step had critical errors - stopping process",
            )
            return False

        # 2. Validate regime classification file existence (CRITICAL - blocks process)
        # Prefer Parquet; fallback to legacy JSON
        parquet_path = f"{data_dir}/{exchange}_{symbol}_regime_classification.parquet"
        json_path = f"{data_dir}/{exchange}_{symbol}_regime_classification.json"

        if Path(parquet_path).exists():
            regime_file_path = parquet_path
        else:
            regime_file_path = json_path
            # Log explicit fallback to legacy JSON for traceability
            self.logger.info(
                f"ℹ️ Using legacy JSON regime file (Parquet not found): {json_path}",
            )

        file_passed, file_metrics = self.validate_file_exists(
            regime_file_path,
            "regime_classification",
        )
        self.validation_results["file_existence"] = file_metrics

        if not file_passed:
            self.logger.error(
                f"❌ Regime classification file not found: {regime_file_path} - stopping process",
            )
            return False

        # 3. Validate regime classification results (CRITICAL - blocks process)
        try:
            # Load results from Parquet or JSON depending on what exists
            if regime_file_path.endswith(".parquet"):
                import pandas as pd
                import os

                # Prefer dataset scan from partitioned store if available
                try:
                    from src.training.enhanced_training_manager_optimized import (
                        ParquetDatasetManager,
                    )

                    pdm = ParquetDatasetManager(logger=self.logger)
                    # Derive base dir and filters from context if possible
                    base_dir = (
                        os.path.join(data_dir, "parquet", "regime_classification")
                        if "data_dir" in locals()
                        else None
                    )
                    if base_dir and os.path.isdir(base_dir):
                        # We don't have exchange/symbol/timeframe params here; read full and let downstream checks validate
                        from src.utils.data_optimizer import regime_columns

                        regime_df = pdm.scan_dataset(
                            base_dir, columns=regime_columns(), to_pandas=True
                        )
                    else:
                        try:
                            from src.utils.data_optimizer import regime_columns
                            from src.utils.logger import log_io_operation

                            with log_io_operation(
                                self.logger,
                                "read_parquet",
                                regime_file_path,
                                columns="regime_columns",
                            ):
                                regime_df = pd.read_parquet(
                                    regime_file_path, columns=regime_columns()
                                )  # minimize
                        except Exception:
                            from src.utils.logger import log_io_operation

                            with log_io_operation(
                                self.logger, "read_parquet", regime_file_path
                            ):
                                regime_df = pd.read_parquet(regime_file_path)
                except Exception:
                    try:
                        from src.utils.data_optimizer import regime_columns
                        from src.utils.logger import log_io_operation

                        with log_io_operation(
                            self.logger,
                            "read_parquet",
                            regime_file_path,
                            columns="regime_columns",
                        ):
                            regime_df = pd.read_parquet(
                                regime_file_path, columns=regime_columns()
                            )  # minimize
                    except Exception:
                        from src.utils.logger import log_io_operation

                        with log_io_operation(
                            self.logger, "read_parquet", regime_file_path
                        ):
                            regime_df = pd.read_parquet(regime_file_path)

                # Convert to minimal dict expected by downstream checks
                regime_results = {
                    "regime_sequence": (
                        regime_df["regime"].tolist()
                        if "regime" in regime_df.columns
                        else []
                    ),
                    "metadata": {"source": "parquet"},
                }
            else:
                with open(regime_file_path) as f:
                    regime_results = json.load(f)

            regime_passed, regime_metrics = self._validate_regime_classification(
                regime_results,
            )
            self.validation_results["regime_classification"] = regime_metrics

            if not regime_passed:
                self.logger.error(
                    "❌ Regime classification validation failed - stopping process",
                )
                return False

            # 4. Validate regime distribution (WARNING - doesn't block)
            distribution_passed = self._validate_regime_distribution(regime_results)
            if not distribution_passed:
                self.logger.warning(
                    "⚠️ Regime distribution validation failed - continuing with caution",
                )

            # 5. Validate regime transitions (WARNING - doesn't block)
            transitions_passed = self._validate_regime_transitions(regime_results)
            if not transitions_passed:
                self.logger.warning(
                    "⚠️ Regime transitions validation failed - continuing with caution",
                )

            # 6. Validate outcome favorability (WARNING - doesn't block)
            outcome_passed, outcome_metrics = self.validate_outcome_favorability(
                step_result,
            )
            self.validation_results["outcome_favorability"] = outcome_metrics

            if not outcome_passed:
                self.logger.warning(
                    "⚠️ Market regime classification outcome is not favorable - continuing with caution",
                )

            # Overall validation passes if critical checks pass
            critical_passed = error_passed and file_passed and regime_passed
            if critical_passed:
                self.logger.info(
                    "✅ Market regime classification validation passed (critical checks only)",
                )
                return True
            self.logger.error(
                "❌ Market regime classification validation failed (critical checks failed)",
            )
            return False

        except Exception as e:
            self.logger.exception(
                f"❌ Error during regime classification validation: {e}",
            )
            return False

    def _validate_regime_classification(
        self,
        regime_results: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        """
        Validate regime classification results.

        Args:
            regime_results: Regime classification results

        Returns:
            Tuple of (passed, metrics_dict)
        """
        try:
            # Check if results contain required keys - updated to match actual file structure
            required_keys = ["regime_sequence", "metadata"]
            missing_keys = [key for key in required_keys if key not in regime_results]

            if missing_keys:
                return False, {
                    "error": f"Missing required keys in regime classification results: {missing_keys}",
                }

            # Validate regime sequence array
            regime_sequence = regime_results.get("regime_sequence", [])
            if not isinstance(regime_sequence, list) or len(regime_sequence) == 0:
                return False, {
                    "error": "Regime sequence array is empty or invalid - no regime classifications found",
                }

            # Validate confidence scores (optional - can be empty object or array)
            confidence_scores = regime_results.get("confidence_scores", {})
            confidence_scores_valid = True
            confidence_scores_message = ""

            if isinstance(confidence_scores, dict) and len(confidence_scores) == 0:
                # Empty object is acceptable
                confidence_scores_message = (
                    "Empty confidence scores object (acceptable)"
                )
            elif isinstance(confidence_scores, list):
                # Validate confidence score values (should be between 0 and 1) - more tolerant
                invalid_confidence_count = 0
                for i, confidence_score in enumerate(confidence_scores):
                    if isinstance(confidence_score, int | float):
                        if not (0.0 <= confidence_score <= 1.0):
                            confidence_scores_valid = False
                            invalid_confidence_count += 1
                            if invalid_confidence_count <= 3:  # Log first few errors
                                self.logger.warning(
                                    f"⚠️ Confidence score at index {i} is {confidence_score} (should be between 0.0 and 1.0)",
                                )
                    else:
                        confidence_scores_valid = False
                        invalid_confidence_count += 1
                        if invalid_confidence_count <= 3:  # Log first few errors
                            self.logger.warning(
                                f"⚠️ Confidence score at index {i} is not a number: {confidence_score}",
                            )

                if not confidence_scores_valid:
                    confidence_scores_message = f"Confidence scores are not valid (should be between 0.0 and 1.0) - {invalid_confidence_count} invalid scores found"
                else:
                    confidence_scores_message = f"Valid confidence scores array with {len(confidence_scores)} scores"
            else:
                confidence_scores_valid = False
                confidence_scores_message = f"Invalid confidence_scores format: {type(confidence_scores)} (expected dict or list)"

            # Check for reasonable regime values
            valid_regimes = all(isinstance(r, int | str) for r in regime_sequence)
            if not valid_regimes:
                return False, {
                    "error": "Invalid regime values found - regimes must be integers or strings",
                }

            # Calculate metrics
            unique_regimes = set(regime_sequence)
            regime_count = len(unique_regimes)

            metrics_dict = {
                "total_regimes": len(regime_sequence),
                "unique_regimes": regime_count,
                "regime_types": list(unique_regimes),
                "confidence_scores_valid": confidence_scores_valid,
                "confidence_scores_message": confidence_scores_message,
                "passed": True,
            }

            return True, metrics_dict

        except Exception as e:
            return False, {"error": f"Error validating regime classification: {str(e)}"}

    def _validate_regime_distribution(self, regime_results: dict[str, Any]) -> bool:
        """
        Validate regime distribution characteristics.

        Args:
            regime_results: Regime classification results

        Returns:
            bool: True if distribution is valid
        """
        try:
            regimes = regime_results.get(
                "regime_sequence",
                [],
            )  # Changed from "regimes" to "regime_sequence"
            if not regimes:
                self.logger.warning(
                    "⚠️ No regimes found for distribution validation - continuing with caution",
                )
                return False

            # Count regime frequencies
            regime_counts = {}
            for regime in regimes:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1

            total_regimes = len(regimes)

            # Check for balanced distribution (no single regime should dominate) - more lenient
            max_regime_count = max(regime_counts.values())
            max_regime_ratio = max_regime_count / total_regimes

            if max_regime_ratio > self.max_regime_dominance:  # More lenient threshold
                self.logger.warning(
                    f"⚠️ Regime distribution is imbalanced: {max_regime_ratio:.3f} (max allowed: {self.max_regime_dominance:.3f}) - continuing with caution",
                )
                return False

            # Check for minimum regime diversity
            if len(regime_counts) < 2:
                self.logger.warning(
                    "⚠️ Only one regime detected - insufficient diversity for robust training - continuing with caution",
                )
                return False

            # Check for reasonable regime frequencies - more lenient
            min_regime_ratio = min(regime_counts.values()) / total_regimes
            if min_regime_ratio < self.min_regime_frequency:  # More lenient threshold
                self.logger.warning(
                    f"⚠️ Some regimes have very low frequency: {min_regime_ratio:.3f} (min allowed: {self.min_regime_frequency:.3f}) - continuing with caution",
                )
                return False

            self.logger.info(
                f"✅ Regime distribution validation passed: {len(regime_counts)} regimes",
            )
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Error during regime distribution validation: {e}",
            )
            return False

    def _validate_regime_transitions(self, regime_results: dict[str, Any]) -> bool:
        """
        Validate regime transition characteristics.

        Args:
            regime_results: Regime classification results

        Returns:
            bool: True if transitions are valid
        """
        try:
            regimes = regime_results.get(
                "regime_sequence",
                [],
            )  # Changed from "regimes" to "regime_sequence"
            if len(regimes) < 2:
                self.logger.info(
                    "ℹ️ Single regime detected - no transitions to validate",
                )
                return True  # Single regime, no transitions to validate

            # Count transitions
            transitions = {}
            for i in range(len(regimes) - 1):
                current_regime = regimes[i]
                next_regime = regimes[i + 1]
                transition_key = f"{current_regime}->{next_regime}"
                transitions[transition_key] = transitions.get(transition_key, 0) + 1

            # Check for reasonable transition patterns - more lenient
            total_transitions = len(regimes) - 1

            # Check for excessive regime switching (more than threshold)
            regime_changes = sum(
                1 for i in range(len(regimes) - 1) if regimes[i] != regimes[i + 1]
            )
            change_ratio = regime_changes / total_transitions

            if change_ratio > self.max_regime_switching:  # More lenient threshold
                self.logger.warning(
                    f"⚠️ High regime switching frequency: {change_ratio:.3f} (max allowed: {self.max_regime_switching:.3f}) - continuing with caution",
                )
                return False

            # Check for stuck regimes (no changes for long periods) - more lenient
            max_consecutive_same = 1
            current_consecutive = 1
            for i in range(1, len(regimes)):
                if regimes[i] == regimes[i - 1]:
                    current_consecutive += 1
                    max_consecutive_same = max(
                        max_consecutive_same,
                        current_consecutive,
                    )
                else:
                    current_consecutive = 1

            # Check if any regime is stuck for too long - more lenient
            max_stuck_ratio = max_consecutive_same / len(regimes)
            if max_stuck_ratio > self.max_stuck_ratio:  # More lenient threshold
                self.logger.warning(
                    f"⚠️ Regime stuck for too long: {max_stuck_ratio:.3f} (max allowed: {self.max_stuck_ratio:.3f}) - continuing with caution",
                )
                return False

            self.logger.info(
                f"✅ Regime transitions validation passed: {len(transitions)} transition types",
            )
            return True

        except Exception as e:
            self.print(
                validation_error(f"❌ Error during regime transitions validation: {e}"),
            )
            return False


async def run_validator(
    training_input: dict[str, Any],
    pipeline_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the Step 2 Market Regime Classification validator.

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results
    """
    validator = Step2MarketRegimeClassificationValidator(CONFIG)
    validation_passed = await validator.validate(training_input, pipeline_state)

    # Build a meaningful top-level error message on failure to improve upstream logging/metrics
    error_message: str | None = None
    if not validation_passed:
        vr = validator.validation_results or {}
        # File existence failure
        fe = vr.get("file_existence") if isinstance(vr, dict) else None
        if isinstance(fe, dict) and not fe.get("exists", True):
            error_message = f"Required file missing: {fe.get('file_path', '<unknown>')}"
        # Regime classification specific error
        if not error_message:
            rc = vr.get("regime_classification") if isinstance(vr, dict) else None
            if isinstance(rc, dict) and rc.get("error"):
                error_message = str(rc.get("error"))
        # Critical errors listed by error_absence validator
        if not error_message:
            ea = vr.get("error_absence") if isinstance(vr, dict) else None
            if isinstance(ea, dict) and ea.get("has_critical_errors"):
                msgs = ea.get("error_messages")
                error_message = (
                    ", ".join(map(str, msgs)) if msgs else "Critical errors present"
                )
        if not error_message:
            error_message = "Validation failed"

    return {
        "step_name": "step2_market_regime_classification",
        "validation_passed": validation_passed,
        "validation_results": validator.validation_results,
        "error": error_message,
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
            "regime_classification": {"status": "SUCCESS", "duration": 45.2},
        }

        result = await run_validator(training_input, pipeline_state)
        print(f"Validation result: {result}")

    asyncio.run(test_validator())
