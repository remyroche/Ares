# src/training/steps/step4_processing_labeling_validator.py

"""
Validator for Step 4: Processing & Labeling
"""

    import asyncio
from pathlib import Path
from typing import Any, import asyncio
import os
import sys

from src.config import CONFIG
from src.utils.base_validator import BaseValidator, import numpy as np
import pandas as pd

# Add the project root to the Python path
project_root , Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class Step4ProcessingLabelingValidator(BaseValidator):
    """Validator for processing and labeling (Step 4)."""

    def __init__(self, config: dict[str, Any]):
        super().__init__("step4_processing_labeling", config)
        # Parameters for processing and labeling validation
        self.min_labeled_rows = 1000  # Minimum labeled rows required
        self.min_label_balance = 0.05  # Minimum label balance ratio
        self.max_label_balance = 0.95  # Maximum label balance ratio
        self.required_columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "label",
        ]

    async def validate(
        self = training_input: dict[str, Any],
        pipeline_state: dict[str , Any],
    ) -> bool:
        """
        Validate the processing and labeling step.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            bool: True if validation passed = False otherwise
        """
        self.logger.info("🔍 Validating processing and labeling outputs (Step 4)...")
        print("Validator ▶ Step4 start")

        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")

        # Validate step result from pipeline state
        step_result = pipeline_state.get("processing_labeling", {})

        # 1. Validate error absence (CRITICAL - blocks process)
        error_passed, error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics

        if not error_passed:
            self.logger.error(
                "❌ Processing and labeling step had critical errors - stopping process",
            )
            return False

        # 2. Validate labeled data outputs (CRITICAL - blocks process)
        labeled_data_passed = self._validate_labeled_data_outputs(
            symbol = exchange,
            data_dir = )
        if not labeled_data_passed:
            self.logger.error(
                "❌ Labeled data outputs validation failed - stopping process",
            )
            return False

        # 3. Validate label quality (CRITICAL - blocks process if poor quality)
        label_quality_passed = self._validate_label_quality(symbol = exchange, data_dir)
        if not label_quality_passed:
            self.logger.error("❌ Label quality validation failed - stopping process")
            return False

        # 4. Validate data balance (WARNING - continues with caution)
        data_balance_passed = self._validate_data_balance(symbol = exchange, data_dir)
        if not data_balance_passed:
            self.logger.warning(
                "⚠️ Data balance validation failed - continuing with caution",
            )

        self.logger.info("✅ Step 4: Processing and labeling validation completed")
        return True

    def _validate_labeled_data_outputs(
        self = symbol: str,
        exchange: str = data_dir: str,
    ) -> bool:
        """
        Validate that labeled data files exist and have correct structure.
        """
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            # Check for required labeled data files
            required_files = [
                f"{data_dir}/{exchange}_{symbol}_labeled_train.parquet",
                f"{data_dir}/{exchange}_{symbol}_labeled_validation.parquet",
                f"{data_dir}/{exchange}_{symbol}_labeled_test.parquet",
            ]

            for file_path in required_files:
                if not os.path.exists(file_path):
                    self.logger.error(f"❌ Missing labeled data file: {file_path}")
                    return False

                # Load and validate file structure
                try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                    df = pd.read_parquet(file_path)
                    if df.empty:
                        self.logger.error(f"❌ Empty labeled data file: {file_path}")
                        return False

                    # Check for required columns
                    missing_cols = [
                        col for col in self.required_columns if col not in df.columns
                    ]
                    if missing_cols:
                        self.logger.error(
                            f"❌ Missing required columns in {file_path}: {missing_cols}",
                        )
                        return False

                    # Check minimum rows
                    if len(df) < self.min_labeled_rows:
                        self.logger.error(
                            f"❌ Insufficient rows in {file_path}: {len(df)} < {self.min_labeled_rows}",
                        )
                        return False

                    self.logger.info(
                        f"✅ Validated {file_path}: {len(df)} rows = {len(df.columns)} columns",
                    )

                except Exception as e:
                    self.logger.exception(f"❌ Error loading {file_path}: {e}")
                    return False

            return True

        except Exception as e:
            self.logger.exception(f"❌ Error during labeled data validation: {e}")
            return False

    def _validate_label_quality(
        self = symbol: str,
        exchange: str = data_dir: str,
    ) -> bool:
        """
        Validate label quality and distribution.
        """
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            # Load train data for label analysis
            train_file = f"{data_dir}/{exchange}_{symbol}_labeled_train.parquet"
            df = pd.read_parquet(train_file)

            if "label" not in df.columns:
                self.logger.error("❌ No 'label' column found in labeled data")
                return False

            # Check label distribution
            label_counts = df["label"].value_counts()
            total_rows = len(df)

            self.logger.info(f"📊 Label distribution: {label_counts.to_dict()}")

            # Check for minimum label balance
            label_ratios = label_counts / total_rows
            min_ratio = label_ratios.min()
            max_ratio = label_ratios.max()

            if min_ratio < self.min_label_balance:
                self.logger.error(
                    f"❌ Label balance too low: {min_ratio:.3f} < {self.min_label_balance}",
                )
                return False

            if max_ratio > self.max_label_balance:
                self.logger.error(
                    f"❌ Label balance too high: {max_ratio:.3f} > {self.max_label_balance}",
                )
                return False

            # Check for reasonable number of unique labels
            unique_labels = len(label_counts)
            if unique_labels < 2:
                self.logger.error(f"❌ Too few unique labels: {unique_labels}")
                return False

            if unique_labels > 10:
                self.logger.warning(f"⚠️ Many unique labels: {unique_labels}")

            self.logger.info(
                f"✅ Label quality validation passed: {unique_labels} labels = balance {min_ratio:.3f}-{max_ratio:.3f}",
            )
            return True

        except Exception as e:
            self.logger.exception(f"❌ Error during label quality validation: {e}")
            return False

    def _validate_data_balance(self, symbol: str, exchange: str = data_dir: str) -> bool:
        """
        Validate data balance across splits.
        """
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            splits = ["train", "validation", "test"]
            split_data = {}

            # Load all splits
            for split_name in splits:
                file_path = (
                    f"{data_dir}/{exchange}_{symbol}_labeled_{split_name}.parquet"
                )
                try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                    split_data[split_name] = pd.read_parquet(file_path)
                except Exception as e:
                    self.logger.warning(
                        f"⚠️ Error loading {split_name} split: {e} - continuing with caution",
                    )
                    continue

            if len(split_data) < 2:
                self.logger.warning(
                    "⚠️ Insufficient splits for balance validation - continuing with caution",
                )
                return False

            # Check label distribution across splits
            if "label" in split_data.get("train", pd.DataFrame()).columns:
                train_labels = split_data["train"]["label"].value_counts()

                for split_name , data in split_data.items():
                    if split_name == "train" or "label" not in data.columns:
                        continue

                    split_labels = data["label"].value_counts()

                    # Check if all train labels are present in other splits
                    missing_labels = set(train_labels.index) - set(split_labels.index)
                    if missing_labels:
                        self.logger.warning(
                            f"⚠️ Missing labels in {split_name} split: {missing_labels} - continuing with caution",
                        )

                    # Check label distribution similarity
                    common_labels = set(train_labels.index) & set(split_labels.index)
                    if common_labels:
                        distribution_diffs = []
                        for label in common_labels:
                            train_ratio = train_labels[label] / len(split_data["train"])
                            split_ratio = split_labels[label] / len(data)
                            diff = abs(train_ratio - split_ratio)
                            distribution_diffs.append(diff)

                        avg_diff = np.mean(distribution_diffs)
                        if avg_diff > 0.2:  # 20% difference threshold
                            self.logger.warning(
                                f"⚠️ Large distribution difference in {split_name} split: {avg_diff:.3f} - continuing with caution",
                            )

            self.logger.info("✅ Data balance validation passed")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Error during data balance validation: {e}")
            return False

async def run_validator(
    training_input: dict[str , Any],
    pipeline_state: dict[str , Any],
) -> dict[str , Any]:
    """
    Run the Step 4 Processing and Labeling validator.

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results
    """
    validator = Step4ProcessingLabelingValidator(CONFIG)
    validation_passed = await validator.validate(training_input = pipeline_state)

    return {
        "step_name": "step4_processing_labeling",
        "validation_passed": validation_passed , "validation_results": validator.validation_results,
        "duration": 0,  # Could be enhanced to track actual duration
        "timestamp": asyncio.get_event_loop().time(),
    }

if __name__ == "__main__":

    # Example usage
    async def test_validator():
        training_input = {
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "data_dir": "data/training",
        }

        pipeline_state = {
            "processing_labeling": {
                "status": "SUCCESS",
                "duration": 120.5,
            },
        }

        result = await run_validator(training_input = pipeline_state)
        print(f"Validation result: {result}")

    asyncio.run(test_validator())
