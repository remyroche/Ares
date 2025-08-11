"""
Validator for Step 8: Tactician Labeling
"""

import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.warning_symbols import (
    error,
    failed,
    missing,
    validation_error,
)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import CONFIG
from src.utils.base_validator import BaseValidator


class Step8TacticianLabelingValidator(BaseValidator):
    """Validator for Step 8: Tactician Labeling."""

    def __init__(self, config: dict[str, Any]):
        super().__init__("step8_tactician_labeling", config)

    async def validate(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> bool:
        """
        Validate the tactician labeling step.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("🔍 Validating tactician labeling step...")

        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")

        # Validate step result from pipeline state
        step_result = pipeline_state.get("tactician_labeling", {})

        # 1. Validate error absence
        error_passed, error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics

        if not error_passed:
            self.print(error("❌ Tactician labeling step had errors"))
            return False

        # 2. Validate tactician labeling files existence
        labeling_files_passed = self._validate_labeling_files_existence(
            symbol,
            exchange,
            data_dir,
        )
        if not labeling_files_passed:
            self.print(failed("❌ Tactician labeling files validation failed"))
            return False

        # 3. Validate signal quality
        signal_quality_passed = self._validate_signal_quality(
            symbol,
            exchange,
            data_dir,
        )
        if not signal_quality_passed:
            self.print(failed("❌ Signal quality validation failed"))
            return False

        # 4. Validate labeling consistency
        consistency_passed = self._validate_labeling_consistency(
            symbol,
            exchange,
            data_dir,
        )
        if not consistency_passed:
            self.print(failed("❌ Labeling consistency validation failed"))
            return False

        # 5. Validate signal distribution
        distribution_passed = self._validate_signal_distribution(
            symbol,
            exchange,
            data_dir,
        )
        if not distribution_passed:
            self.print(failed("❌ Signal distribution validation failed"))
            return False

        # 6. Validate outcome favorability
        outcome_passed, outcome_metrics = self.validate_outcome_favorability(
            step_result,
        )
        self.validation_results["outcome_favorability"] = outcome_metrics

        if not outcome_passed:
            self.print(error("⚠️ Tactician labeling outcome is not favorable"))
            return False

        self.logger.info("✅ Tactician labeling validation passed")
        return True

    def _validate_labeling_files_existence(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """
        Validate that tactician labeling files exist.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if files exist
        """
        try:
            # Expected tactician labeling file patterns
            expected_files = [
                f"{data_dir}/{exchange}_{symbol}_tactician_signals.pkl",
                f"{data_dir}/{exchange}_{symbol}_tactician_labels.pkl",
                f"{data_dir}/{exchange}_{symbol}_tactician_labeling_metadata.json",
            ]

            missing_files = []
            for file_path in expected_files:
                file_passed, file_metrics = self.validate_file_exists(
                    file_path,
                    "tactician_labeling_files",
                )
                if not file_passed:
                    missing_files.append(file_path)

            if missing_files:
                self.logger.error(
                    f"❌ Missing tactician labeling files: {missing_files}",
                )
                return False

            self.logger.info("✅ All tactician labeling files exist")
            return True

        except Exception:
            self.print(error("❌ Error validating tactician labeling files: {e}"))
            return False

    def _validate_signal_quality(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """
        Validate the quality of generated trading signals.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if signal quality is acceptable
        """
        try:
            # Load tactician signals (prefer Parquet)
            signals_parquet = (
                f"{data_dir}/{exchange}_{symbol}_tactician_signals.parquet"
            )
            signals_pickle = f"{data_dir}/{exchange}_{symbol}_tactician_signals.pkl"

            if os.path.exists(signals_parquet) or os.path.exists(signals_pickle):
                if os.path.exists(signals_parquet):
                    # Prefer dataset scan if labeled partition exists
                    try:
                        from src.training.enhanced_training_manager_optimized import (
                            ParquetDatasetManager,
                        )

                        pdm = ParquetDatasetManager(logger=self.logger)
                        part_base = os.path.join(data_dir, "parquet", "labeled")
                        if os.path.isdir(part_base):
                            filters = [
                                ("exchange", "==", exchange),
                                ("symbol", "==", symbol),
                            ]
                            t0 = getattr(self, "t0_ms", None)
                            t1 = getattr(self, "t1_ms", None)
                            if t0 is not None:
                                filters.append(("timestamp", ">=", int(t0)))
                            if t1 is not None:
                                filters.append(("timestamp", "<", int(t1)))
                            # Project only signal-related columns if present
                            columns = ["timestamp", "signal", "confidence"]
                            signals_data = pdm.scan_dataset(
                                part_base,
                                filters=filters,
                                columns=columns,
                                to_pandas=True,
                            )
                        else:
                            try:
                                from src.utils.logger import (
                                    log_io_operation,
                                    log_dataframe_overview,
                                )

                                with log_io_operation(
                                    self.logger,
                                    "read_parquet",
                                    signals_parquet,
                                    columns=True,
                                ):
                                    signals_data = pd.read_parquet(
                                        signals_parquet,
                                        columns=["timestamp", "signal", "confidence"],
                                    )
                                try:
                                    log_dataframe_overview(
                                        self.logger, signals_data, name="signals_data"
                                    )
                                except Exception:
                                    pass
                            except Exception:
                                from src.utils.logger import log_io_operation

                                with log_io_operation(
                                    self.logger, "read_parquet", signals_parquet
                                ):
                                    signals_data = pd.read_parquet(signals_parquet)
                    except Exception:
                        signals_data = pd.read_parquet(signals_parquet)
                else:
                    with open(signals_pickle, "rb") as f:
                        signals_data = pickle.load(f)

                if not isinstance(signals_data, pd.DataFrame):
                    signals_data = pd.DataFrame(signals_data)

                # Check for required signal columns
                required_columns = ["signal", "confidence", "timestamp"]
                missing_columns = [
                    col for col in required_columns if col not in signals_data.columns
                ]

                if missing_columns:
                    self.logger.error(
                        f"❌ Missing required signal columns: {missing_columns}",
                    )
                    return False

                # Validate signal values
                signals = signals_data["signal"]
                unique_signals = signals.unique()

                # Check for reasonable signal values (typically -1, 0, 1 or similar)
                if len(unique_signals) < 2:
                    self.print(error("❌ Insufficient signal diversity"))
                    return False

                if len(unique_signals) > 10:
                    self.print(error("⚠️ Many signal types: {len(unique_signals)}"))

                # Check signal consistency
                signal_counts = signals.value_counts()
                total_signals = len(signals)

                # Check for signal balance
                max_signal_count = signal_counts.max()
                min_signal_count = signal_counts.min()
                balance_ratio = (
                    min_signal_count / max_signal_count if max_signal_count > 0 else 0
                )

                if balance_ratio < 0.1:  # Very imbalanced signals
                    self.logger.warning(
                        f"⚠️ Imbalanced signal distribution: {balance_ratio:.3f}",
                    )

                # Check confidence values
                if "confidence" in signals_data.columns:
                    confidence = signals_data["confidence"]

                    # Check confidence range (should be 0-1 or similar)
                    if confidence.min() < 0 or confidence.max() > 1:
                        self.logger.warning(
                            "⚠️ Confidence values outside expected range [0,1]",
                        )

                    # Check for reasonable confidence distribution
                    low_confidence = (confidence < 0.3).sum()
                    high_confidence = (confidence > 0.7).sum()

                    if low_confidence > total_signals * 0.8:
                        self.print(error("⚠️ Too many low confidence signals"))

                    if high_confidence < total_signals * 0.1:
                        self.print(error("⚠️ Too few high confidence signals"))

                # Check for signal continuity
                signal_changes = (signals != signals.shift()).sum()
                change_ratio = signal_changes / total_signals

                if change_ratio > 0.5:
                    self.logger.warning(
                        f"⚠️ High signal change frequency: {change_ratio:.3f}",
                    )

                self.logger.info(
                    f"✅ Signal quality validation passed: {total_signals} signals",
                )
                return True

            self.print(missing("❌ Tactician signals file not found"))
            return False

        except Exception:
            self.print(
                validation_error("❌ Error during signal quality validation: {e}"),
            )
            return False

    def _validate_labeling_consistency(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """
        Validate consistency of tactician labeling.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if labeling is consistent
        """
        try:
            # Load tactician labels (prefer Parquet)
            labels_parquet = f"{data_dir}/{exchange}_{symbol}_tactician_labels.parquet"
            labels_pickle = f"{data_dir}/{exchange}_{symbol}_tactician_labels.pkl"

            if os.path.exists(labels_parquet) or os.path.exists(labels_pickle):
                if os.path.exists(labels_parquet):
                    try:
                        from src.training.enhanced_training_manager_optimized import (
                            ParquetDatasetManager,
                        )

                        pdm = ParquetDatasetManager(logger=self.logger)
                        part_base = os.path.join(data_dir, "parquet", "labeled")
                        if os.path.isdir(part_base):
                            filters = [
                                ("exchange", "==", exchange),
                                ("symbol", "==", symbol),
                            ]
                            t0 = getattr(self, "t0_ms", None)
                            t1 = getattr(self, "t1_ms", None)
                            if t0 is not None:
                                filters.append(("timestamp", ">=", int(t0)))
                            if t1 is not None:
                                filters.append(("timestamp", "<", int(t1)))
                            columns = ["timestamp", "label"]
                            labels_data = pdm.scan_dataset(
                                part_base,
                                filters=filters,
                                columns=columns,
                                to_pandas=True,
                            )
                        else:
                            try:
                                from src.utils.logger import (
                                    log_io_operation,
                                    log_dataframe_overview,
                                )

                                with log_io_operation(
                                    self.logger,
                                    "read_parquet",
                                    labels_parquet,
                                    columns=True,
                                ):
                                    labels_data = pd.read_parquet(
                                        labels_parquet, columns=["timestamp", "label"]
                                    )
                                try:
                                    log_dataframe_overview(
                                        self.logger, labels_data, name="labels_data"
                                    )
                                except Exception:
                                    pass
                            except Exception:
                                from src.utils.logger import log_io_operation

                                with log_io_operation(
                                    self.logger, "read_parquet", labels_parquet
                                ):
                                    labels_data = pd.read_parquet(labels_parquet)
                    except Exception:
                        labels_data = pd.read_parquet(labels_parquet)
                else:
                    with open(labels_pickle, "rb") as f:
                        labels_data = pickle.load(f)

                if not isinstance(labels_data, pd.DataFrame):
                    labels_data = pd.DataFrame(labels_data)

                # Check for required label columns
                if "label" not in labels_data.columns:
                    self.print(error("❌ No label column found in tactician labels"))
                    return False

                labels = labels_data["label"]

                # Check label values
                unique_labels = labels.unique()

                if len(unique_labels) < 2:
                    self.print(error("❌ Insufficient label diversity"))
                    return False

                # Check for label consistency with signals
                signals_parquet = (
                    f"{data_dir}/{exchange}_{symbol}_tactician_signals.parquet"
                )
                signals_pickle = f"{data_dir}/{exchange}_{symbol}_tactician_signals.pkl"

                if os.path.exists(signals_parquet) or os.path.exists(signals_pickle):
                    if os.path.exists(signals_parquet):
                        try:
                            from src.utils.logger import log_io_operation

                            with log_io_operation(
                                self.logger,
                                "read_parquet",
                                signals_parquet,
                                columns=True,
                            ):
                                signals_data = pd.read_parquet(
                                    signals_parquet,
                                    columns=["timestamp", "signal", "confidence"],
                                )
                        except Exception:
                            from src.utils.logger import log_io_operation

                            with log_io_operation(
                                self.logger, "read_parquet", signals_parquet
                            ):
                                signals_data = pd.read_parquet(signals_parquet)
                    else:
                        with open(signals_pickle, "rb") as f:
                            signals_data = pickle.load(f)

                    if not isinstance(signals_data, pd.DataFrame):
                        signals_data = pd.DataFrame(signals_data)

                    # Check if labels and signals have same length
                    if len(labels) != len(signals_data):
                        self.logger.error(
                            "❌ Labels and signals have different lengths",
                        )
                        return False

                    # Check for reasonable label-signal correlation
                    if "signal" in signals_data.columns:
                        signals = signals_data["signal"]

                        # Calculate correlation between labels and signals
                        try:
                            correlation = np.corrcoef(
                                labels.astype(float),
                                signals.astype(float),
                            )[0, 1]

                            if abs(correlation) < 0.1:
                                self.logger.warning(
                                    f"⚠️ Low correlation between labels and signals: {correlation:.3f}",
                                )
                            elif abs(correlation) > 0.95:
                                self.logger.warning(
                                    f"⚠️ Very high correlation between labels and signals: {correlation:.3f}",
                                )
                        except:
                            self.logger.warning(
                                "⚠️ Could not calculate label-signal correlation",
                            )

                # Check for label balance
                label_counts = labels.value_counts()
                total_labels = len(labels)

                min_label_count = label_counts.min()
                max_label_count = label_counts.max()
                balance_ratio = (
                    min_label_count / max_label_count if max_label_count > 0 else 0
                )

                if balance_ratio < 0.1:
                    self.logger.warning(
                        f"⚠️ Imbalanced label distribution: {balance_ratio:.3f}",
                    )

                # Check for missing labels
                null_labels = labels.isnull().sum()
                if null_labels > 0:
                    self.print(missing("⚠️ Found {null_labels} missing labels"))

                self.logger.info(
                    f"✅ Labeling consistency validation passed: {total_labels} labels",
                )
                return True

            self.print(missing("❌ Tactician labels file not found"))
            return False

        except Exception as e:
            self.logger.exception(
                f"❌ Error during labeling consistency validation: {e}",
            )
            return False

    def _validate_signal_distribution(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """
        Validate the distribution of trading signals.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if signal distribution is acceptable
        """
        try:
            # Load tactician labeling metadata
            metadata_file = (
                f"{data_dir}/{exchange}_{symbol}_tactician_labeling_metadata.json"
            )

            if os.path.exists(metadata_file):
                import json

                with open(metadata_file) as f:
                    metadata = json.load(f)

                # Check signal distribution metrics
                if "signal_distribution" in metadata:
                    signal_dist = metadata["signal_distribution"]

                    # Check for reasonable signal distribution
                    for signal_type, count in signal_dist.items():
                        if count < 10:
                            self.logger.warning(
                                f"⚠️ Very few signals of type {signal_type}: {count}",
                            )

                # Check signal frequency
                if "signal_frequency" in metadata:
                    signal_freq = metadata["signal_frequency"]

                    if signal_freq < 0.01:  # Very low signal frequency
                        self.logger.warning(
                            f"⚠️ Very low signal frequency: {signal_freq:.3f}",
                        )
                    elif signal_freq > 0.5:  # Very high signal frequency
                        self.logger.warning(
                            f"⚠️ Very high signal frequency: {signal_freq:.3f}",
                        )

                # Check signal quality metrics
                if "signal_quality_score" in metadata:
                    quality_score = metadata["signal_quality_score"]

                    if quality_score < 0.6:
                        self.logger.warning(
                            f"⚠️ Low signal quality score: {quality_score:.3f}",
                        )

                # Check labeling accuracy
                if "labeling_accuracy" in metadata:
                    labeling_acc = metadata["labeling_accuracy"]

                    if labeling_acc < 0.7:
                        self.logger.warning(
                            f"⚠️ Low labeling accuracy: {labeling_acc:.3f}",
                        )

                # Check signal consistency
                if "signal_consistency" in metadata:
                    consistency = metadata["signal_consistency"]

                    if consistency < 0.6:
                        self.logger.warning(
                            f"⚠️ Low signal consistency: {consistency:.3f}",
                        )

            # Load signals for additional validation
            signals_file = f"{data_dir}/{exchange}_{symbol}_tactician_signals.pkl"

            if os.path.exists(signals_file):
                with open(signals_file, "rb") as f:
                    signals_data = pickle.load(f)

                if not isinstance(signals_data, pd.DataFrame):
                    signals_data = pd.DataFrame(signals_data)

                if "signal" in signals_data.columns:
                    signals = signals_data["signal"]

                    # Check for signal clustering
                    signal_changes = (signals != signals.shift()).cumsum()
                    unique_clusters = signal_changes.nunique()

                    if unique_clusters < 5:
                        self.print(error("⚠️ Few signal clusters: {unique_clusters}"))
                    elif unique_clusters > 100:
                        self.logger.warning(
                            f"⚠️ Many signal clusters: {unique_clusters}",
                        )

                    # Check for signal persistence
                    signal_persistence = signals.groupby(signals).size()
                    avg_persistence = signal_persistence.mean()

                    if avg_persistence < 5:
                        self.logger.warning(
                            f"⚠️ Low signal persistence: {avg_persistence:.1f}",
                        )
                    elif avg_persistence > 100:
                        self.logger.warning(
                            f"⚠️ High signal persistence: {avg_persistence:.1f}",
                        )

            self.logger.info("✅ Signal distribution validation passed")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Error during signal distribution validation: {e}",
            )
            return False


async def run_validator(
    training_input: dict[str, Any],
    pipeline_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the step8_tactician_labeling validator.

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results
    """
    validator = Step8TacticianLabelingValidator(CONFIG)
    validation_passed = await validator.validate(training_input, pipeline_state)

    return {
        "step_name": "step8_tactician_labeling",
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
            "tactician_labeling": {"status": "SUCCESS", "duration": 240.5},
        }

        result = await run_validator(training_input, pipeline_state)
        print(f"Validation result: {result}")

    asyncio.run(test_validator())
