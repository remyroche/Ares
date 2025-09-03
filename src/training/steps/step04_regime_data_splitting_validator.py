#!/usr/bin/env python3
"""Validator for Step 4: Regime Data Splitting."

import pandas as pd
This module validates the regime data splitting step outputs with support for 10+ regimes.
"""
from src.core.domain import (

from src.core.decorators import validates
    smart_validation_cache,
    validate_step4_comprehensive
)

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


from src.utils.base_validator import BaseValidator
from src.utils.logger import system_logger
from src.utils.common_operations import safe_json_load

logger = system_logger.getChild("Step4RegimeDataSplittingValidator")


class Step4RegimeDataSplittingValidator(BaseValidator):
    """Validator for Step 4: Regime Data Splitting."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("step04_regime_data_splitting", config)
        self.logger = system_logger.getChild("Validator.Step4")

    @validates()
    async def validate_step4_regime_data_splitting(
        self, symbol: str, exchange: str, data_dir: str, training_input: dict[str, Any]
    ) -> bool:
        """Validate Step 4: Regime Data Splitting."

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            training_input: Training input data

        Returns:
            bool: True if validation passes
        """
        self.logger.info("🔍 Starting Step 4: Regime Data Splitting validation")

        try:
            # Check if regime data splitting directory exists
            regime_splits_dir = Path(data_dir) / "training" / "regime_splits"
            if not regime_splits_dir.exists():
                self.logger.warning(
                    f"⚠️ Regime splits directory not found: {regime_splits_dir}"
                )
                return False

            # Validate regime split files
            regime_files = list(regime_splits_dir.glob("*.parquet"))
            if not regime_files:
                self.logger.warning("⚠️ No regime split files found")
                return False

            # Validate each regime file
            for regime_file in regime_files:
                if not await self._validate_regime_file(regime_file):
                    return False

            # Check for regime statistics file
            stats_file = regime_splits_dir / f"{exchange}_{symbol}_1m_regime_statistics.json"
            if not stats_file.exists():
                self.logger.warning(f"⚠️ Regime statistics file not found: {stats_file}")
                return False

            # Validate statistics file
            if not await self._validate_statistics_file(stats_file):
                return False

            self.logger.info("✅ Step 4: Regime Data Splitting validation passed")
            return True

        except Exception as e:
            error_context = {
                "step": "step04_regime_data_splitting",
                "symbol": symbol,
                "exchange": exchange,
                "data_dir": data_dir,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "timestamp": pd.Timestamp.now().isoformat()
            }
            self.logger.exception(f"❌ Step 4 validation failed: {error_context}")
            return False

    @smart_validation_cache(ttl_seconds=300)  # Cache for 5 minutes
    async def _validate_regime_file(self, regime_file: Path) -> bool:
        """Validate a regime split file with caching."""
        try:
            self.logger.info(f"📁 Validating regime file: {regime_file.name}")

            # Use BaseValidator's file validation'
            file_exists, file_metrics = self.validate_file_exists(str(regime_file), "regime file")
            if not file_exists:
                return False

            # Load and validate the regime file
            df = pd.read_parquet(regime_file)

            # Use BaseValidator's DataFrame validation'
            df_valid, df_metrics = self.validate_dataframe_quality(
                df=df,
                min_rows=100,
                required_columns=["timestamp", "composite_cluster_id"],
                check_data_types=True,
                check_value_ranges=True,
                check_duplicates=True,
                check_temporal_consistency=True
            )

            if not df_valid:
                self.logger.warning(f"⚠️ DataFrame validation failed for {regime_file.name}")
                return False

            # Additional regime-specific validation
            if "composite_cluster_id" in df.columns:
                unique_regimes = df["composite_cluster_id"].nunique()
                if unique_regimes < 2 or unique_regimes > 50:
                    self.logger.warning(
                        f"⚠️ Unusual number of regimes ({unique_regimes}) in {regime_file.name}"
                    )

            self.logger.info(f"✅ Regime file validated: {regime_file.name}")
            return True

        except Exception as e:
            error_context = {
                "file": str(regime_file),
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
            self.logger.exception(f"❌ Failed to validate regime file: {error_context}")
            return False

    @smart_validation_cache(ttl_seconds=600)  # Cache for 10 minutes
    async def _validate_statistics_file(self, stats_file: Path) -> bool:
        """Validate the regime statistics file with caching."""
        try:
            self.logger.info(f"📊 Validating statistics file: {stats_file.name}")

            # Use BaseValidator's file validation'
            file_exists, file_metrics = self.validate_file_exists(str(stats_file), "statistics file")
            if not file_exists:
                return False

            stats_data = safe_json_load(stats_file)

            # Check if it's a dictionary'
            if not isinstance(stats_data, dict):
                self.logger.warning("⚠️ Statistics file should contain a dictionary")
                return False

            # Check for regime statistics
            if not stats_data:
                self.logger.warning("⚠️ Empty statistics data")
                return False

            # Validate each regime's statistics'
            for regime_id, stats in stats_data.items():
                if not isinstance(stats, dict):
                    self.logger.warning(f"⚠️ Invalid statistics format for regime {regime_id}")
                    return False

                # Check for basic statistics
                basic_fields = ["count", "percentage", "mean_volatility", "mean_momentum"]
                missing_basic = [field for field in basic_fields if field not in stats]
                if missing_basic:
                    self.logger.warning(
                        f"⚠️ Missing basic statistics for regime {regime_id}: {missing_basic}"
                    )
                    return False

            self.logger.info(f"✅ Statistics file validated: {stats_file.name}")
            return True

        except Exception as e:
            error_context = {
                "file": str(stats_file),
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
            self.logger.exception(f"❌ Failed to validate statistics file: {error_context}")
            return False

    def validates(self, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Validate prerequisites for Step 4 using BaseValidator methods."""
        validation_result = {
            "validation_passed": True,
            "warnings": [],
            "errors": [],
            "details": {}
        }

        try:
            # Check if step3_hmm_regime_discovery output exists using BaseValidator
            step3_output_dir = Path("data/training")
            step3_files = list(step3_output_dir.glob(f"{exchange}_{symbol}_{timeframe}*hmm*.parquet"))
            
            if not step3_files:
                validation_result["validation_passed"] = False
                validation_result["errors"].append(
                    f"Step 3 HMM regime discovery output not found for {exchange}_{symbol}_{timeframe}"
                )
            else:
                # Validate each file using BaseValidator
                for file_path in step3_files:
                    file_valid, file_metrics = self.validate_file_exists(str(file_path), "step03 output file")
                    if not file_valid:
                        validation_result["warnings"].append(f"File validation failed: {file_path}")
                
                validation_result["details"]["step3_files_found"] = len(step3_files)
                validation_result["details"]["step3_files"] = [str(f) for f in step3_files]

        except Exception as e:
            validation_result["validation_passed"] = False
            validation_result["errors"].append(f"Prerequisites validation failed: {str(e)}")

        return validation_result

    def validates(self, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Validate Step 4 output files and content using BaseValidator methods."""
        validation_result = {
            "validation_passed": True,
            "warnings": [],
            "errors": [],
            "details": {}
        }

        try:
            # Define expected output files
            output_dir = Path("data/training/regime_splits")
            expected_files = [
                f"{exchange}_{symbol}_{timeframe}_regime_splits.parquet",
                f"{exchange}_{symbol}_{timeframe}_regime_statistics.json"
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
                    f"Missing regime data splitting file: {f}" for f in missing_files
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
    """Run validation for Step 4: Regime Data Splitting."

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results
    """
    logger.info("🔍 Validating Step 4: Regime Data Splitting")
    
    try:
        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        timeframe = training_input.get("timeframe", "1m")
        data_dir = training_input.get("data_dir", "data_cache")
        
        # Initialize validator with BaseValidator inheritance
        config = training_input.get("config", {})
        validator = Step4RegimeDataSplittingValidator(config)
        
        # Validate prerequisites using BaseValidator methods
        prereq_result = validator.validate_step_prerequisites(symbol, exchange, timeframe)
        
        # Validate step execution
        step_result = await validator.validate_step4_regime_data_splitting(
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
            "step_name": "step04_regime_data_splitting",
            "validation_passed": validation_passed,
            "prerequisites": prereq_result,
            "step_execution": step_result,
            "outputs": output_result,
            "warnings": prereq_result["warnings"] + output_result["warnings"],
            "errors": prereq_result["errors"] + output_result["errors"]
        }
        
    except Exception as e:
        error_context = {
            "step": "step04_regime_data_splitting",
            "symbol": training_input.get("symbol", "UNKNOWN"),
            "exchange": training_input.get("exchange", "UNKNOWN"),
            "error_type": type(e).__name__,
            "error_message": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        logger.exception(f"❌ Step 4 validation failed: {error_context}")
        return {
            "step_name": "step04_regime_data_splitting",
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