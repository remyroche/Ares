#!/usr/bin/env python3
"""Validator for Step 6: Feature Engineering.

This module validates the feature engineering step outputs.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger

logger = system_logger.getChild("Step6FeatureEngineeringValidator")


async def run_validator(
    training_input: Dict[str, Any],
    pipeline_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Run validation for Step 6: Feature Engineering.

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results
    """
    logger.info("🔍 Validating Step 6: Feature Engineering")
    
    try:
        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        timeframe = training_input.get("timeframe", "1m")
        data_dir = training_input.get("data_dir", "data_cache")
        
        # Check if feature engineering files exist
        features_train_path = Path(data_dir) / "training" / f"{exchange}_{symbol}_features_train.parquet"
        features_metadata_path = Path(data_dir) / "training" / f"{exchange}_{symbol}_features_metadata.json"
        
        if not features_train_path.exists():
            logger.error(f"❌ Features train file not found: {features_train_path}")
            return {
                "step_name": "step6_feature_engineering",
                "validation_passed": False,
                "error": f"Features train file not found: {features_train_path}",
            }
        
        if not features_metadata_path.exists():
            logger.error(f"❌ Features metadata file not found: {features_metadata_path}")
            return {
                "step_name": "step6_feature_engineering",
                "validation_passed": False,
                "error": f"Features metadata file not found: {features_metadata_path}",
            }
        
        # Check file sizes
        train_file_size = features_train_path.stat().st_size
        metadata_file_size = features_metadata_path.stat().st_size
        
        if train_file_size == 0:
            logger.error(f"❌ Features train file is empty: {features_train_path}")
            return {
                "step_name": "step6_feature_engineering",
                "validation_passed": False,
                "error": "Features train file is empty",
            }
        
        if metadata_file_size == 0:
            logger.error(f"❌ Features metadata file is empty: {features_metadata_path}")
            return {
                "step_name": "step6_feature_engineering",
                "validation_passed": False,
                "error": "Features metadata file is empty",
            }
        
        # Try to read the files to validate structure
        try:
            import pandas as pd
            import json
            
            # Read features data
            features_data = pd.read_parquet(features_train_path)
            
            # Read metadata
            with open(features_metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check data quality
            if len(features_data) == 0:
                logger.error("❌ No data rows found in features")
                return {
                    "step_name": "step6_feature_engineering",
                    "validation_passed": False,
                    "error": "No data rows found in features",
                }
            
            # Check for reasonable number of features
            num_features = len(features_data.columns)
            if num_features < 10:
                logger.warning(f"⚠️ Very few features generated: {num_features}")
            
            # Check for basic OHLCV columns
            basic_columns = ["open", "high", "low", "close", "volume"]
            missing_basic = [col for col in basic_columns if col not in features_data.columns]
            if missing_basic:
                logger.warning(f"⚠️ Missing basic OHLCV columns: {missing_basic}")
            
            # Check for label column
            if "label" not in features_data.columns:
                logger.warning("⚠️ Label column not found in features data")
            
            # Check metadata structure
            if not isinstance(metadata, dict):
                logger.error("❌ Metadata is not a dictionary")
                return {
                    "step_name": "step6_feature_engineering",
                    "validation_passed": False,
                    "error": "Metadata is not a dictionary",
                }
            
            # Check for NaN values
            nan_count = features_data.isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"⚠️ Found {nan_count} NaN values in features data")
            
            # Check for infinite values
            inf_count = features_data.isin([float('inf'), float('-inf')]).sum().sum()
            if inf_count > 0:
                logger.warning(f"⚠️ Found {inf_count} infinite values in features data")
            
            logger.info(f"✅ Features data shape: {features_data.shape}")
            logger.info(f"✅ Number of features: {num_features}")
            logger.info(f"✅ Metadata keys: {list(metadata.keys())}")
            
            logger.info("✅ Step 6: Feature Engineering validation passed")
            return {
                "step_name": "step6_feature_engineering",
                "validation_passed": True,
                "features_file_path": str(features_train_path),
                "metadata_file_path": str(features_metadata_path),
                "data_shape": features_data.shape,
                "num_features": num_features,
                "nan_count": nan_count,
                "inf_count": inf_count,
            }
            
        except Exception as e:
            logger.error(f"❌ Error reading feature engineering files: {e}")
            return {
                "step_name": "step6_feature_engineering",
                "validation_passed": False,
                "error": f"Error reading files: {e}",
            }
            
    except Exception as e:
        logger.exception(f"❌ Error in Step 6 validation: {e}")
        return {
            "step_name": "step6_feature_engineering",
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