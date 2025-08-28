#!/usr/bin/env python3
"""Validator for Step 7: Regime Data Splitting.

This module validates the regime data splitting step outputs.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.utils.centralized_decorators import (
    comprehensive_data_validation,
    handle_errors,
    memory_efficient,
    resource_monitor,
    secure_data_processing,
    validate_data_structure,
    with_tracing_span,
    quality_gate,
)

logger = system_logger.getChild("Step7RegimeDataSplittingValidator")


@with_tracing_span("validate_regime_data_splitting")
@quality_gate(
    min_quality_score=0.7,
    max_correlation=0.95,
    required_grade="C"
)
@comprehensive_data_validation
@handle_errors
@memory_efficient
@resource_monitor
@secure_data_processing
@validate_data_structure
async def run_validator(
    training_input: Dict[str, Any],
    pipeline_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Run validation for Step 7: Regime Data Splitting.

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results
    """
    logger.info("🔍 Validating Step 7: Regime Data Splitting")
    
    try:
        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        timeframe = training_input.get("timeframe", "1m")
        data_dir = training_input.get("data_dir", "data_cache")
        
        # Check if regime data splitting files exist
        train_path = Path(data_dir) / "training" / f"{exchange}_{symbol}_{timeframe}_regime_splits_train.parquet"
        validation_path = Path(data_dir) / "training" / f"{exchange}_{symbol}_{timeframe}_regime_splits_validation.parquet"
        
        if not train_path.exists():
            logger.error(f"❌ Regime splits train file not found: {train_path}")
            return {
                "step_name": "step7_regime_data_splitting",
                "validation_passed": False,
                "error": f"Regime splits train file not found: {train_path}",
            }
        
        if not validation_path.exists():
            logger.error(f"❌ Regime splits validation file not found: {validation_path}")
            return {
                "step_name": "step7_regime_data_splitting",
                "validation_passed": False,
                "error": f"Regime splits validation file not found: {validation_path}",
            }
        
        # Check file sizes
        train_file_size = train_path.stat().st_size
        validation_file_size = validation_path.stat().st_size
        
        if train_file_size == 0:
            logger.error(f"❌ Regime splits train file is empty: {train_path}")
            return {
                "step_name": "step7_regime_data_splitting",
                "validation_passed": False,
                "error": "Regime splits train file is empty",
            }
        
        if validation_file_size == 0:
            logger.error(f"❌ Regime splits validation file is empty: {validation_path}")
            return {
                "step_name": "step7_regime_data_splitting",
                "validation_passed": False,
                "error": "Regime splits validation file is empty",
            }
        
        # Try to read the files to validate structure
        try:
            import pandas as pd
            
            # Read train data
            train_data = pd.read_parquet(train_path)
            
            # Read validation data
            validation_data = pd.read_parquet(validation_path)
            
            # Check data quality
            if len(train_data) == 0:
                logger.error("❌ No data rows found in train split")
                return {
                    "step_name": "step7_regime_data_splitting",
                    "validation_passed": False,
                    "error": "No data rows found in train split",
                }
            
            if len(validation_data) == 0:
                logger.error("❌ No data rows found in validation split")
                return {
                    "step_name": "step7_regime_data_splitting",
                    "validation_passed": False,
                    "error": "No data rows found in validation split",
                }
            
            # Check for required columns
            required_columns = ["label", "composite_cluster_id"]
            missing_train_columns = [col for col in required_columns if col not in train_data.columns]
            missing_validation_columns = [col for col in required_columns if col not in validation_data.columns]
            
            if missing_train_columns:
                logger.error(f"❌ Missing required columns in train data: {missing_train_columns}")
                return {
                    "step_name": "step7_regime_data_splitting",
                    "validation_passed": False,
                    "error": f"Missing required columns in train data: {missing_train_columns}",
                }
            
            if missing_validation_columns:
                logger.error(f"❌ Missing required columns in validation data: {missing_validation_columns}")
                return {
                    "step_name": "step7_regime_data_splitting",
                    "validation_passed": False,
                    "error": f"Missing required columns in validation data: {missing_validation_columns}",
                }
            
            # Check label distribution
            train_label_counts = train_data["label"].value_counts()
            validation_label_counts = validation_data["label"].value_counts()
            
            logger.info(f"✅ Train label distribution: {train_label_counts.to_dict()}")
            logger.info(f"✅ Validation label distribution: {validation_label_counts.to_dict()}")
            
            # Check regime distribution
            train_regime_counts = train_data["composite_cluster_id"].value_counts()
            validation_regime_counts = validation_data["composite_cluster_id"].value_counts()
            
            logger.info(f"✅ Train regime distribution: {train_regime_counts.to_dict()}")
            logger.info(f"✅ Validation regime distribution: {validation_regime_counts.to_dict()}")
            
            # Check for reasonable split sizes
            total_samples = len(train_data) + len(validation_data)
            train_ratio = len(train_data) / total_samples
            validation_ratio = len(validation_data) / total_samples
            
            logger.info(f"✅ Train ratio: {train_ratio:.2%}")
            logger.info(f"✅ Validation ratio: {validation_ratio:.2%}")
            
            # Check for reasonable split ratios (typically 70-80% train, 20-30% validation)
            if train_ratio < 0.6 or train_ratio > 0.9:
                logger.warning(f"⚠️ Unusual train ratio: {train_ratio:.2%}")
            
            if validation_ratio < 0.1 or validation_ratio > 0.4:
                logger.warning(f"⚠️ Unusual validation ratio: {validation_ratio:.2%}")
            
            # Check for overlap in timestamps if available
            if "timestamp" in train_data.columns and "timestamp" in validation_data.columns:
                train_timestamps = set(train_data["timestamp"])
                validation_timestamps = set(validation_data["timestamp"])
                overlap = train_timestamps.intersection(validation_timestamps)
                
                if overlap:
                    logger.warning(f"⚠️ Found {len(overlap)} overlapping timestamps between train and validation")
                else:
                    logger.info("✅ No overlapping timestamps between train and validation")
            
            logger.info("✅ Step 7: Regime Data Splitting validation passed")
            return {
                "step_name": "step7_regime_data_splitting",
                "validation_passed": True,
                "train_file_path": str(train_path),
                "validation_file_path": str(validation_path),
                "train_shape": train_data.shape,
                "validation_shape": validation_data.shape,
                "train_ratio": train_ratio,
                "validation_ratio": validation_ratio,
                "train_label_distribution": train_label_counts.to_dict(),
                "validation_label_distribution": validation_label_counts.to_dict(),
            }
            
        except Exception as e:
            logger.error(f"❌ Error reading regime data splitting files: {e}")
            return {
                "step_name": "step7_regime_data_splitting",
                "validation_passed": False,
                "error": f"Error reading files: {e}",
            }
            
    except Exception as e:
        logger.exception(f"❌ Error in Step 7 validation: {e}")
        return {
            "step_name": "step7_regime_data_splitting",
            "validation_passed": False,
            "error": f"Validation error: {e}",
        }


if __name__ == "__main__":
    # Test the validator
    async def test():
        test_input = {
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "timeframe": "1m",
            "data_dir": "data_cache"
        }
        test_state = {}
        
        result = await run_validator(test_input, test_state)
        print(f"Validation result: {result}")

    asyncio.run(test())