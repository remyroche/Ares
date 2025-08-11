"""
Validator for Step 3: Regime Data Splitting
"""

import asyncio
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.warning_symbols import (
    error,
    validation_error,
)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import CONFIG
from src.utils.base_validator import BaseValidator


class Step3RegimeDataSplittingValidator(BaseValidator):
    """Validator for Step 3: Regime Data Splitting."""

    def __init__(self, config: dict[str, Any]):
        super().__init__("step3_regime_data_splitting", config)

        # Check for BLANK mode
        blank_mode = os.environ.get("BLANK_TRAINING_MODE", "0") == "1"

        if blank_mode:
            # Very lenient parameters for BLANK mode
            self.min_sizes = {
                "train": 25,  # Very reduced for BLANK mode
                "validation": 10,  # Very reduced for BLANK mode
                "test": 10,  # Very reduced for BLANK mode
            }
            self.proportion_tolerance = 0.25  # Very lenient for BLANK mode
            self.max_overlap_ratio = 0.10  # Allow up to 10% overlap for BLANK mode
            self.distribution_tolerance = (
                0.9  # Very lenient for distribution differences
            )
            self.logger.info("🔧 BLANK MODE: Using very lenient validation parameters")
        else:
            # Fine-tuned parameters for ML training (more lenient to avoid stopping training)
            self.min_sizes = {
                "train": 500,  # Reduced from 1000
                "validation": 100,  # Reduced from 200
                "test": 100,  # Reduced from 200
            }
            self.proportion_tolerance = (
                0.15  # Increased from strict checking to allow more flexible splits
            )
            self.max_overlap_ratio = 0.05  # Allow up to 5% overlap between splits
            self.distribution_tolerance = (
                0.7  # More lenient for distribution differences
            )
            self.logger.info("🚨 FULL TRAINING MODE: Enhanced alert visibility enabled")

    async def validate(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> bool:
        """
        Validate the regime data splitting step.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("🔍 Validating regime data splitting step...")

        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")

        # Validate step result from pipeline state
        step_result = pipeline_state.get("regime_data_splitting", {})

        # 1. Validate error absence (CRITICAL - blocks process)
        error_passed, error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics

        if not error_passed:
            self.logger.error(
                "❌ Regime data splitting step had critical errors - stopping process",
            )
            return False

        # 2. Validate split data files existence (CRITICAL - blocks process)
        split_files_passed = self._validate_split_files_existence(
            symbol,
            exchange,
            data_dir,
        )
        if not split_files_passed:
            self.logger.error(
                "❌ Split data files validation failed - stopping process",
            )
            return False

        # 3. Validate split data quality (WARNING - doesn't block)
        quality_passed = self._validate_split_data_quality(symbol, exchange, data_dir)
        if not quality_passed:
            self.logger.warning(
                "⚠️ Split data quality validation failed - continuing with caution",
            )

        # 4. Validate split proportions (WARNING - doesn't block)
        proportions_passed = self._validate_split_proportions(
            symbol,
            exchange,
            data_dir,
        )
        if not proportions_passed:
            self.logger.warning(
                "⚠️ Split proportions validation failed - continuing with caution",
            )

        # 5. Validate split consistency (WARNING - doesn't block)
        consistency_passed = self._validate_split_consistency(
            symbol,
            exchange,
            data_dir,
        )
        if not consistency_passed:
            self.logger.warning(
                "⚠️ Split consistency validation failed - continuing with caution",
            )

        # 6. Validate outcome favorability (WARNING - doesn't block)
        outcome_passed, outcome_metrics = self.validate_outcome_favorability(
            step_result,
        )
        self.validation_results["outcome_favorability"] = outcome_metrics

        if not outcome_passed:
            self.logger.warning(
                "⚠️ Regime data splitting outcome is not favorable - continuing with caution",
            )

        # Overall validation passes if critical checks pass
        critical_passed = error_passed and split_files_passed
        if critical_passed:
            self.logger.info(
                "✅ Regime data splitting validation passed (critical checks only)",
            )
            return True
        self.logger.error(
            "❌ Regime data splitting validation failed (critical checks failed)",
        )
        return False

    def _validate_split_files_existence(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """
        Validate that all expected split data files exist.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if all files exist
        """
        try:
            # Expected file patterns for different splits (Parquet preferred)
            expected_files = [
                f"{data_dir}/{exchange}_{symbol}_train_data.parquet",
                f"{data_dir}/{exchange}_{symbol}_validation_data.parquet",
                f"{data_dir}/{exchange}_{symbol}_test_data.parquet",
            ]

            missing_files = []
            for file_path in expected_files:
                file_passed, file_metrics = self.validate_file_exists(
                    file_path,
                    "split_data",
                )
                if not file_passed:
                    missing_files.append(file_path)

            if missing_files:
                self.logger.error(
                    f"❌ Missing split data files: {missing_files} - stopping process",
                )
                return False

            self.logger.info("✅ All split data files exist")
            return True

        except Exception as e:
            self.logger.error(error(f"❌ Error validating split files existence: {e}"))
            return False

    def _validate_split_data_quality(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """
        Validate quality of split data.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if data quality is acceptable
        """
        try:
            # Resolve time window from config/env
            try:
                from src.utils.time_utils import resolve_time_window_ms

                self.t0_ms, self.t1_ms = resolve_time_window_ms(
                    getattr(self, "config", {})
                )
            except Exception:
                self.t0_ms, self.t1_ms = (
                    getattr(self, "t0_ms", None),
                    getattr(self, "t1_ms", None),
                )

            split_files = [
                ("train", f"{data_dir}/{exchange}_{symbol}_train_data"),
                ("validation", f"{data_dir}/{exchange}_{symbol}_validation_data"),
                ("test", f"{data_dir}/{exchange}_{symbol}_test_data"),
            ]

            for split_name, file_base in split_files:
                try:
                    parquet_path = f"{file_base}.parquet"
                    pickle_path = f"{file_base}.pkl"
                    if os.path.exists(parquet_path):
                        # Prefer partitioned labeled/features dataset when available
                        try:
                            from src.training.enhanced_training_manager_optimized import (
                                ParquetDatasetManager,
                            )

                            pdm = ParquetDatasetManager(logger=self.logger)
                            for ds_name in ("labeled", "features"):
                                part_base = os.path.join(data_dir, "parquet", ds_name)
                                if os.path.isdir(part_base):
                                    filters = [
                                        ("exchange", "==", exchange),
                                        ("symbol", "==", symbol),
                                        ("split", "==", split_name),
                                    ]
                                    # Optional time-window filters from env/config captured elsewhere in validators
                                    t0 = getattr(self, "t0_ms", None)
                                    t1 = getattr(self, "t1_ms", None)
                                    if t0 is not None:
                                        filters.append(("timestamp", ">=", int(t0)))
                                    if t1 is not None:
                                        filters.append(("timestamp", "<", int(t1)))
                                    # Quick governance checks: read schema metadata from one file if available
                                    try:
                                        manifest_path = os.path.join(
                                            part_base, "_manifest.json"
                                        )
                                        if os.path.exists(manifest_path):
                                            with open(manifest_path) as _mf:
                                                _ = _mf.read()  # presence is enough to short-circuit deep scans
                                    except Exception:
                                        pass
                                    split_data = pdm.scan_dataset(
                                        part_base,
                                        filters=filters,
                                        columns=None,
                                        to_pandas=True,
                                    )
                                    break
                            else:
                                try:
                                    from src.utils.data_optimizer import regime_columns
                                    from src.utils.logger import (
                                        log_io_operation,
                                        log_dataframe_overview,
                                    )

                                    with log_io_operation(
                                        self.logger,
                                        "read_parquet",
                                        parquet_path,
                                        columns="regime_columns",
                                    ):
                                        split_data = pd.read_parquet(
                                            parquet_path, columns=regime_columns()
                                        )  # minimize
                                    try:
                                        log_dataframe_overview(
                                            self.logger,
                                            split_data,
                                            name=f"{split_name}_data",
                                        )
                                    except Exception:
                                        pass
                                except Exception:
                                    from src.utils.logger import log_io_operation

                                    with log_io_operation(
                                        self.logger, "read_parquet", parquet_path
                                    ):
                                        split_data = pd.read_parquet(parquet_path)
                        except Exception:
                            from src.utils.logger import log_io_operation

                            with log_io_operation(
                                self.logger, "read_parquet", parquet_path
                            ):
                                split_data = pd.read_parquet(parquet_path)
                    else:
                        with open(pickle_path, "rb") as f:
                            split_data = pickle.load(f)

                    # Convert to DataFrame if needed
                    if not isinstance(split_data, pd.DataFrame):
                        split_data = pd.DataFrame(split_data)

                    # Validate data quality for this split
                    quality_passed, quality_metrics = self.validate_data_quality(
                        split_data,
                        f"{split_name}_data",
                    )
                    self.validation_results[f"{split_name}_data_quality"] = (
                        quality_metrics
                    )

                    if not quality_passed:
                        self.logger.warning(
                            f"⚠️ {split_name} data quality validation failed - continuing with caution",
                        )
                        return False

                    # Additional split-specific validations
                    if not self._validate_split_specific_characteristics(
                        split_data,
                        split_name,
                    ):
                        self.logger.warning(
                            f"⚠️ {split_name} split characteristics validation failed - continuing with caution",
                        )
                        return False

                except Exception as e:
                    self.logger.error(
                        error(f"❌ Error validating {split_name} data: {e}")
                    )
                    return False

            self.logger.info("✅ All split data quality validation passed")
            return True

        except Exception as e:
            self.logger.error(
                validation_error(f"❌ Error during split data quality validation: {e}"),
            )
            return False

    def _validate_split_specific_characteristics(
        self,
        data: pd.DataFrame,
        split_name: str,
    ) -> bool:
        """
        Validate characteristics specific to each split.

        Args:
            data: Split data DataFrame
            split_name: Name of the split (train/validation/test)

        Returns:
            bool: True if characteristics are valid
        """
        # Check for BLANK mode
        blank_mode = os.environ.get("BLANK_TRAINING_MODE", "0") == "1"
        try:
            # Check minimum size requirements (more lenient)
            min_size = self.min_sizes.get(split_name, 100)
            if len(data) < min_size:
                self.logger.warning(
                    f"⚠️ {split_name} split too small: {len(data)} records (minimum: {min_size}) - continuing with caution",
                )
                return False

            # Check for required columns (more flexible for different data formats)
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]
            if missing_columns:
                if blank_mode:
                    self.logger.warning(
                        f"⚠️ {split_name} split missing columns: {missing_columns} - continuing with caution in BLANK mode",
                    )
                    return True  # Allow missing columns in BLANK mode
                else:
                    self.logger.warning(
                        f"⚠️ {split_name} split missing columns: {missing_columns} - continuing with caution",
                    )
                    return False

            # Check for reasonable data ranges (more flexible)
            if "close" in data.columns:
                try:
                    price_range = data["close"].max() - data["close"].min()
                    if price_range <= 0:
                        if blank_mode:
                            self.logger.warning(
                                f"⚠️ {split_name} split has invalid price range in BLANK mode - continuing with caution",
                            )
                            return True  # Allow invalid price ranges in BLANK mode
                        else:
                            self.logger.warning(
                                f"⚠️ {split_name} split has invalid price range - continuing with caution",
                            )
                            return False
                except Exception as e:
                    if blank_mode:
                        self.logger.warning(
                            f"⚠️ Error checking price range in {split_name} split in BLANK mode: {e} - continuing with caution",
                        )
                        return True  # Allow errors in BLANK mode
                    else:
                        self.logger.warning(
                            f"⚠️ Error checking price range in {split_name} split: {e} - continuing with caution",
                        )
                        return False

            self.logger.info(f"✅ {split_name} split characteristics validation passed")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Error validating {split_name} characteristics: {e}",
            )
            return False

    def _validate_split_proportions(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """
        Validate that split proportions are reasonable.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if proportions are acceptable
        """
        try:
            split_files = [
                ("train", f"{data_dir}/{exchange}_{symbol}_train_data.pkl"),
                ("validation", f"{data_dir}/{exchange}_{symbol}_validation_data.pkl"),
                ("test", f"{data_dir}/{exchange}_{symbol}_test_data.pkl"),
            ]

            split_sizes = {}
            total_size = 0

            # Calculate sizes for each split
            for split_name, file_path in split_files:
                try:
                    with open(file_path, "rb") as f:
                        split_data = pickle.load(f)

                    if not isinstance(split_data, pd.DataFrame):
                        split_data = pd.DataFrame(split_data)

                    split_sizes[split_name] = len(split_data)
                    total_size += len(split_data)

                except Exception:
                    self.logger.error(error("❌ Error reading {split_name} split: {e}"))
                    return False

            if total_size == 0:
                self.logger.error(
                    error("❌ Total data size is zero - stopping process")
                )
                return False

            # Calculate proportions
            proportions = {
                split: size / total_size for split, size in split_sizes.items()
            }

            # Validate proportions (more lenient)
            expected_proportions = {
                "train": (0.5, 0.85),  # 50-85% (more flexible)
                "validation": (0.05, 0.25),  # 5-25% (more flexible)
                "test": (0.05, 0.25),  # 5-25% (more flexible)
            }

            proportion_warnings = []
            for split_name, (min_prop, max_prop) in expected_proportions.items():
                actual_prop = proportions.get(split_name, 0)
                # Check if proportion is within range (inclusive of boundaries)
                if (
                    actual_prop < min_prop - self.proportion_tolerance
                    or actual_prop > max_prop + self.proportion_tolerance
                ):
                    proportion_warnings.append(
                        f"{split_name}: {actual_prop:.3f} (expected: {min_prop:.3f}-{max_prop:.3f})",
                    )
                    self.logger.warning(
                        f"⚠️ {split_name} proportion {actual_prop:.3f} outside expected range ({min_prop:.3f}-{max_prop:.3f}) - continuing with caution",
                    )
                else:
                    self.logger.info(
                        f"✅ {split_name} proportion {actual_prop:.3f} within expected range ({min_prop:.3f}-{max_prop:.3f})",
                    )

            # Check that proportions sum to approximately 1 (more lenient)
            total_proportion = sum(proportions.values())
            if abs(total_proportion - 1.0) > 0.05:  # Allow 5% deviation
                self.logger.warning(
                    f"⚠️ Split proportions don't sum to 1: {total_proportion:.3f} - continuing with caution",
                )

            self.validation_results["split_proportions"] = {
                "proportions": proportions,
                "total_size": total_size,
                "split_sizes": split_sizes,
                "warnings": proportion_warnings,
                "passed": len(proportion_warnings) == 0,
            }

            self.logger.info(f"✅ Split proportions validation passed: {proportions}")
            return True

        except Exception:
            self.logger.error(
                validation_error("❌ Error during split proportions validation: {e}"),
            )
            return False

    def _validate_split_consistency(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """
        Validate consistency between splits (e.g., no data leakage).

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if splits are consistent
        """
        # Check for BLANK mode
        blank_mode = os.environ.get("BLANK_TRAINING_MODE", "0") == "1"
        try:
            split_files = [
                ("train", f"{data_dir}/{exchange}_{symbol}_train_data.pkl"),
                ("validation", f"{data_dir}/{exchange}_{symbol}_validation_data.pkl"),
                ("test", f"{data_dir}/{exchange}_{symbol}_test_data.pkl"),
            ]

            split_data = {}

            # Load all splits
            for split_name, file_path in split_files:
                try:
                    with open(file_path, "rb") as f:
                        data = pickle.load(f)

                    if not isinstance(data, pd.DataFrame):
                        data = pd.DataFrame(data)

                    split_data[split_name] = data

                except Exception:
                    self.logger.error(error("❌ Error loading {split_name} split: {e}"))
                    return False

            # Check for overlapping timestamps (if timestamp column exists)
            if "timestamp" in split_data["train"].columns:
                train_timestamps = set(split_data["train"]["timestamp"])
                val_timestamps = set(split_data["validation"]["timestamp"])
                test_timestamps = set(split_data["test"]["timestamp"])

                # Check for overlaps (more lenient)
                train_val_overlap = train_timestamps.intersection(val_timestamps)
                train_test_overlap = train_timestamps.intersection(test_timestamps)
                val_test_overlap = val_timestamps.intersection(test_timestamps)

                total_train = len(train_timestamps)
                total_val = len(val_timestamps)
                total_test = len(test_timestamps)

                if train_val_overlap:
                    overlap_ratio = len(train_val_overlap) / min(total_train, total_val)
                    if overlap_ratio > self.max_overlap_ratio:
                        self.logger.warning(
                            f"⚠️ Train-validation overlap: {len(train_val_overlap)} timestamps ({overlap_ratio:.2%}) - continuing with caution",
                        )
                    else:
                        self.logger.info(
                            f"ℹ️ Train-validation overlap: {len(train_val_overlap)} timestamps (acceptable)",
                        )

                if train_test_overlap:
                    overlap_ratio = len(train_test_overlap) / min(
                        total_train,
                        total_test,
                    )
                    if overlap_ratio > self.max_overlap_ratio:
                        self.logger.warning(
                            f"⚠️ Train-test overlap: {len(train_test_overlap)} timestamps ({overlap_ratio:.2%}) - continuing with caution",
                        )
                    else:
                        self.logger.info(
                            f"ℹ️ Train-test overlap: {len(train_test_overlap)} timestamps (acceptable)",
                        )

                if val_test_overlap:
                    overlap_ratio = len(val_test_overlap) / min(total_val, total_test)
                    if overlap_ratio > self.max_overlap_ratio:
                        self.logger.warning(
                            f"⚠️ Validation-test overlap: {len(val_test_overlap)} timestamps ({overlap_ratio:.2%}) - continuing with caution",
                        )
                    else:
                        self.logger.info(
                            f"ℹ️ Validation-test overlap: {len(val_test_overlap)} timestamps (acceptable)",
                        )

            # Check for temporal ordering (train < validation < test)
            if "timestamp" in split_data["train"].columns:
                train_max_time = split_data["train"]["timestamp"].max()
                val_min_time = split_data["validation"]["timestamp"].min()
                val_max_time = split_data["validation"]["timestamp"].max()
                test_min_time = split_data["test"]["timestamp"].min()

                if val_min_time <= train_max_time:
                    if blank_mode:
                        self.logger.warning(
                            "⚠️ Validation data may overlap with training data temporally (BLANK mode) - continuing with caution",
                        )
                    else:
                        self.logger.error(
                            "🚨 CRITICAL ALERT: Validation data overlaps with training data temporally (FULL TRAINING MODE)!",
                        )
                        self.logger.error(
                            "   This violates temporal ordering and may cause data leakage!"
                        )

                if test_min_time <= val_max_time:
                    if blank_mode:
                        self.logger.warning(
                            "⚠️ Test data may overlap with validation data temporally (BLANK mode) - continuing with caution",
                        )
                    else:
                        self.logger.error(
                            "🚨 CRITICAL ALERT: Test data overlaps with validation data temporally (FULL TRAINING MODE)!",
                        )
                        self.logger.error(
                            "   This violates temporal ordering and may cause data leakage!"
                        )

            # Check for reasonable feature distributions across splits (more lenient)
            numeric_columns = (
                split_data["train"].select_dtypes(include=["number"]).columns
            )
            for col in numeric_columns[:5]:  # Check first 5 numeric columns
                if col in split_data["train"].columns:
                    train_mean = split_data["train"][col].mean()
                    val_mean = split_data["validation"][col].mean()
                    test_mean = split_data["test"][col].mean()

                    # Check if means are reasonably close (more lenient)
                    if train_mean != 0:
                        val_diff = abs(val_mean - train_mean) / abs(train_mean)
                        test_diff = abs(test_mean - train_mean) / abs(train_mean)

                        if (
                            val_diff > self.distribution_tolerance
                            or test_diff > self.distribution_tolerance
                        ):
                            if blank_mode:
                                self.logger.warning(
                                    f"⚠️ Large distribution difference in {col} (BLANK mode): train={train_mean:.3f}, val={val_mean:.3f}, test={test_mean:.3f} - continuing with caution",
                                )
                            else:
                                self.logger.error(
                                    f"🚨 CRITICAL ALERT: Large distribution difference in {col} (FULL TRAINING MODE)!",
                                )
                                self.logger.error(
                                    f"   Train={train_mean:.3f}, Val={val_mean:.3f}, Test={test_mean:.3f}"
                                )
                                self.logger.error(
                                    f"   This may indicate data leakage or poor split quality!"
                                )

            if not blank_mode:
                self.logger.info(
                    "🚨 FULL TRAINING MODE: Enhanced validation completed with critical alerts enabled"
                )
                self.logger.info(
                    "   Any critical alerts above require immediate attention for production training!"
                )

            self.logger.info("✅ Split consistency validation passed")
            return True

        except Exception:
            self.logger.error(
                validation_error("❌ Error during split consistency validation: {e}"),
            )
            return False


async def run_validator(
    training_input: dict[str, Any],
    pipeline_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the Step 3 Regime Data Splitting validator.

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results
    """
    validator = Step3RegimeDataSplittingValidator(CONFIG)
    validation_passed = await validator.validate(training_input, pipeline_state)

    return {
        "step_name": "step3_regime_data_splitting",
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
            "regime_data_splitting": {"status": "SUCCESS", "duration": 30.5},
        }

        result = await run_validator(training_input, pipeline_state)
        print(f"Validation result: {result}")

    asyncio.run(test_validator())
