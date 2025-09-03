#!/usr/bin/env python3
"""Validator for Step 5: Labeling."

This module validates the labeling step outputs.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.base_validator import BaseValidator
from src.utils.logger import system_logger
from src.utils.enhanced_validation_decorators import (
    validate_step5_comprehensive,
    smart_validation_cache
)
from src.utils.common_operations import safe_json_load

logger = system_logger.getChild("Step5LabelingValidator")


class Step5LabelingValidator(BaseValidator):
    """Validator for Step 5: Labeling."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("step5_labeling", config)
        self.logger = system_logger.getChild("Validator.Step5")

    @validate_step5_comprehensive
    async def validate_step5_labeling(
        self, symbol: str, exchange: str, data_dir: str, training_input: dict[str, Any]
    ) -> bool:
        """Validate Step 5: Labeling."

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            training_input: Training input data

        Returns:
            bool: True if validation passes
        """
        self.logger.info("🔍 Starting Step 5: Labeling validation")

        try:
            # Check if labeled data directory exists
            labeled_data_dir = Path(data_dir) / "training" / "labeled_data"
            if not labeled_data_dir.exists():
                self.logger.warning(
                    f"⚠️ Labeled data directory not found: {labeled_data_dir}"
                )
                return False

            # Validate labeled data files
            labeled_files = list(labeled_data_dir.glob("*.parquet"))
            if not labeled_files:
                self.logger.warning("⚠️ No labeled data files found")
                return False

            # Validate each labeled file
            for labeled_file in labeled_files:
                if not await self._validate_labeled_file(labeled_file):
                    return False

            # Check for labeling metadata file
            metadata_file = labeled_data_dir / f"{exchange}_{symbol}_1m_labeling_metadata.json"
            if not metadata_file.exists():
                self.logger.warning(f"⚠️ Labeling metadata file not found: {metadata_file}")
                return False

            # Validate metadata file
            if not await self._validate_metadata_file(metadata_file):
                return False

            self.logger.info("✅ Step 5: Labeling validation passed")
            return True

        except Exception as e:
            error_context = {
                "step": "step5_labeling",
                "symbol": symbol,
                "exchange": exchange,
                "data_dir": data_dir,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "timestamp": pd.Timestamp.now().isoformat()
            }
            self.logger.exception(f"❌ Step 5 validation failed: {error_context}")
            return False

    @smart_validation_cache(ttl_seconds=300)  # Cache for 5 minutes
    async def _validate_labeled_file(self, labeled_file: Path) -> bool:
        """Validate a labeled data file with caching."""
        try:
            self.logger.info(f"📁 Validating labeled file: {labeled_file.name}")

            # Use BaseValidator's file validation'
            file_exists, file_metrics = self.validate_file_exists(str(labeled_file), "labeled file")
            if not file_exists:
                return False

            # Load and validate the labeled file
            df = pd.read_parquet(labeled_file)

            # Use BaseValidator's DataFrame validation'
            df_valid, df_metrics = self.validate_dataframe_quality(
                df=df,
                min_rows=100,
                required_columns=["timestamp", "label"],
                check_data_types=True,
                check_value_ranges=True,
                check_duplicates=True,
                check_temporal_consistency=True
            )

            if not df_valid:
                self.logger.warning(f"⚠️ DataFrame validation failed for {labeled_file.name}")
                return False

            # Additional labeling-specific validation
            if "label" in df.columns:
                unique_labels = df["label"].nunique()
                if unique_labels < 2:
                    self.logger.warning(f"⚠️ Insufficient label diversity in {labeled_file.name}: {unique_labels} labels")
                    return False
                
                # Check label distribution
                label_counts = df["label"].value_counts()
                min_label_count = label_counts.min()
                if min_label_count < 10:  # Minimum samples per label
                    self.logger.warning(f"⚠️ Some labels have very few samples in {labeled_file.name}: min={min_label_count}")

            self.logger.info(f"✅ Labeled file validated: {labeled_file.name}")
            return True

        except Exception as e:
            error_context = {
                "file": str(labeled_file),
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
            self.logger.exception(f"❌ Failed to validate labeled file: {error_context}")
            return False

    @smart_validation_cache(ttl_seconds=600)  # Cache for 10 minutes
    async def _validate_metadata_file(self, metadata_file: Path) -> bool:
        """Validate the labeling metadata file with caching."""
        try:
            self.logger.info(f"📊 Validating metadata file: {metadata_file.name}")

            # Use BaseValidator's file validation'
            file_exists, file_metrics = self.validate_file_exists(str(metadata_file), "metadata file")
            if not file_exists:
                return False

            metadata = safe_json_load(metadata_file)

            # Check if metadata is a dictionary
            if not isinstance(metadata, dict):
                self.logger.warning("⚠️ Metadata file should contain a dictionary")
                return False

            # Check for required fields
            required_fields = ["labeling_method", "label_distribution", "total_samples", "labeling_timestamp"]
            missing_fields = [field for field in required_fields if field not in metadata]
            if missing_fields:
                self.logger.warning(f"⚠️ Missing required fields in metadata: {missing_fields}")
                return False

            # Validate label distribution
            label_distribution = metadata.get("label_distribution", {})
            if not isinstance(label_distribution, dict):
                self.logger.warning("⚠️ Label distribution should be a dictionary")
                return False

            # Validate total samples
            total_samples = metadata.get("total_samples", 0)
            if not isinstance(total_samples, int) or total_samples <= 0:
                self.logger.warning(f"⚠️ Invalid total_samples: {total_samples}")
                return False

            self.logger.info(f"✅ Metadata file validated: {metadata_file.name}")
            return True

        except Exception as e:
            error_context = {
                "file": str(metadata_file),
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
            self.logger.exception(f"❌ Failed to validate metadata file: {error_context}")
            return False

    def validate_step_prerequisites(self, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Validate prerequisites for Step 5 using BaseValidator methods."""
        validation_result = {
            "validation_passed": True,
            "warnings": [],
            "errors": [],
            "details": {}
        }

        try:
            # Check if step4_regime_data_splitting output exists using BaseValidator
            step4_output_dir = Path("data/training/regime_splits")
            step4_files = list(step4_output_dir.glob(f"{exchange}_{symbol}_{timeframe}*regime*.parquet"))
            
            if not step4_files:
                validation_result["validation_passed"] = False
                validation_result["errors"].append(
                    f"Step 4 regime data splitting output not found for {exchange}_{symbol}_{timeframe}"
                )
            else:
                # Validate each file using BaseValidator
                for file_path in step4_files:
                    file_valid, file_metrics = self.validate_file_exists(str(file_path), "step04 output file")
                    if not file_valid:
                        validation_result["warnings"].append(f"File validation failed: {file_path}")
                
                validation_result["details"]["step4_files_found"] = len(step4_files)
                validation_result["details"]["step4_files"] = [str(f) for f in step4_files]

        except Exception as e:
            validation_result["validation_passed"] = False
            validation_result["errors"].append(f"Prerequisites validation failed: {str(e)}")

        return validation_result

    def validate_step_output(self, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Validate Step 5 output files and content using BaseValidator methods."""
        validation_result = {
            "validation_passed": True,
            "warnings": [],
            "errors": [],
            "details": {}
        }

        try:
            # Define expected output files
            output_dir = Path("data/training/labeled_data")
            expected_files = [
                f"{exchange}_{symbol}_{timeframe}_labeled_data.parquet",
                f"{exchange}_{symbol}_{timeframe}_labeling_metadata.json"
            ]

            # Check if all expected files exist using BaseValidator
            missing_files = []
            existing_files = []
            
            for filename in expected_files:
                file_path = output_dir / filename
                file_valid, file_metrics = self.validate_file_exists(str(file_path), f"expected file: {filename}")
                
                if file_valid:
                    existing_files.append(str(file_path))
                else:
                    missing_files.append(filename)

            if missing_files:
                validation_result["validation_passed"] = False
                validation_result["errors"].extend([
                    f"Missing labeling file: {f}" for f in missing_files
                ])
            else:
                validation_result["details"]["files_found"] = len(existing_files)
                validation_result["details"]["files"] = existing_files

            # Validate file contents using BaseValidator
            if existing_files:
                for file_path in existing_files:
                    if file_path.endswith(".parquet"):
                        try:
                            df = pd.read_parquet(file_path)
                            # Use BaseValidator's DataFrame validation'
                            df_valid, df_metrics = self.validate_dataframe_quality(
                                df, min_rows=100, check_data_types=True
                            )
                            validation_result["details"][f"{Path(file_path).stem}_rows"] = len(df)
                            validation_result["details"][f"{Path(file_path).stem}_columns"] = list(df.columns)
                            validation_result["details"][f"{Path(file_path).stem}_valid"] = df_valid
                        except Exception as e:
                            validation_result["warnings"].append(f"Could not read parquet file {file_path}: {e}")

        except Exception as e:
            validation_result["validation_passed"] = False
            validation_result["errors"].append(f"Output validation failed: {str(e)}")

        return validation_result


async def run_validator(
    training_input: Dict[str, Any],
    pipeline_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Run validation for Step 5: Labeling."

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results
    """
    logger.info("🔍 Validating Step 5: Labeling")
    
    try:
        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        timeframe = training_input.get("timeframe", "1m")
        data_dir = training_input.get("data_dir", "data_cache")
        
        # Initialize validator with BaseValidator inheritance
        config = training_input.get("config", {})
        validator = Step5LabelingValidator(config)
        
        # Validate prerequisites using BaseValidator methods
        prereq_result = validator.validate_step_prerequisites(symbol, exchange, timeframe)
        
        # Validate step execution
        step_result = await validator.validate_step5_labeling(
            symbol, exchange, data_dir, training_input
        )
        
        # Validate outputs using BaseValidator methods
        output_result = validator.validate_step_output(symbol, exchange, timeframe)
        
        # Combine results
        validation_passed = (
            prereq_result["validation_passed"] and 
            step_result and 
            output_result["validation_passed"]
        )
        
        return {
            "step_name": "step5_labeling",
            "validation_passed": validation_passed,
            "prerequisites": prereq_result,
            "step_execution": step_result,
            "outputs": output_result,
            "warnings": prereq_result["warnings"] + output_result["warnings"],
            "errors": prereq_result["errors"] + output_result["errors"]
        }
        
    except Exception as e:
        error_context = {
            "step": "step5_labeling",
            "symbol": training_input.get("symbol", "UNKNOWN"),
            "exchange": training_input.get("exchange", "UNKNOWN"),
            "error_type": type(e).__name__,
            "error_message": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        logger.exception(f"❌ Step 5 validation failed: {error_context}")
        return {
            "step_name": "step5_labeling",
            "validation_passed": False,
            "error": str(e),
            "error_context": error_context
        }


if __name__ == "__main__":
    # Test the validator
    import asyncio
import datetime as datetime
    
    test_input = {
        "symbol": "ETHUSDT",
        "exchange": "BINANCE", 
        "timeframe": "1m",
        "data_dir": "data_cache",
        "config": {}
    }
    
    test_state = {}
    
    result = asyncio.run(run_validator(test_input, test_state))
    print(json.dumps(result, indent=2))