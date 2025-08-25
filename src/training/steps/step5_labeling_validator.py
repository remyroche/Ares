#!/usr/bin/env python3
"""Validator for Step 5: Labeling.

This module validates the labeling step outputs.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger

logger = system_logger.getChild("Step5LabelingValidator")


async def run_validator(
    training_input: Dict[str, Any],
    pipeline_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Run validation for Step 5: Labeling.

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
        
        # Check if labeled data file exists
        labeled_data_path = Path(data_dir) / "training" / f"{exchange}_{symbol}_{timeframe}_labeled_data.parquet"
        
        if not labeled_data_path.exists():
            logger.error(f"❌ Labeled data file not found: {labeled_data_path}")
            return {
                "step_name": "step5_labeling",
                "validation_passed": False,
                "error": f"Labeled data file not found: {labeled_data_path}",
            }
        
        # Check file size
        file_size = labeled_data_path.stat().st_size
        if file_size == 0:
            logger.error(f"❌ Labeled data file is empty: {labeled_data_path}")
            return {
                "step_name": "step5_labeling",
                "validation_passed": False,
                "error": "Labeled data file is empty",
            }
        
        # Try to read the file to validate structure
        try:
            import pandas as pd
            data = pd.read_parquet(labeled_data_path)
            
            # Check required columns
            required_columns = ["label", "triple_barrier_label"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                logger.error(f"❌ Missing required columns: {missing_columns}")
                return {
                    "step_name": "step5_labeling",
                    "validation_passed": False,
                    "error": f"Missing required columns: {missing_columns}",
                }
            
            # Check data quality
            if len(data) == 0:
                logger.error("❌ No data rows found")
                return {
                    "step_name": "step5_labeling",
                    "validation_passed": False,
                    "error": "No data rows found",
                }
            
            # Check label distribution
            label_counts = data["label"].value_counts()
            triple_barrier_counts = data["triple_barrier_label"].value_counts()
            
            logger.info(f"✅ Label distribution: {label_counts.to_dict()}")
            logger.info(f"✅ Triple barrier label distribution: {triple_barrier_counts.to_dict()}")
            
            # Check for reasonable label distribution
            if 0 in label_counts and label_counts[0] == len(data):
                logger.warning("⚠️ All labels are 0 (hold) - this might indicate an issue")
                return {
                    "step_name": "step5_labeling",
                    "validation_passed": True,  # Still pass but warn
                    "warning": "All labels are 0 (hold) - this might indicate an issue",
                }
            
            # Check for label confidence if available
            if "label_confidence" in data.columns:
                confidence_stats = data["label_confidence"].describe()
                logger.info(f"✅ Label confidence stats: {confidence_stats.to_dict()}")
                
                # Check if confidence is reasonable (between 0 and 1)
                if data["label_confidence"].min() < 0 or data["label_confidence"].max() > 1:
                    logger.warning("⚠️ Label confidence values outside [0,1] range")
            
            # Check for label source if available
            if "label_source" in data.columns:
                source_counts = data["label_source"].value_counts()
                logger.info(f"✅ Label source distribution: {source_counts.to_dict()}")
            
            logger.info("✅ Step 5: Labeling validation passed")
            return {
                "step_name": "step5_labeling",
                "validation_passed": True,
                "file_path": str(labeled_data_path),
                "data_shape": data.shape,
                "label_distribution": label_counts.to_dict(),
                "triple_barrier_distribution": triple_barrier_counts.to_dict(),
            }
            
        except Exception as e:
            logger.error(f"❌ Error reading labeled data file: {e}")
            return {
                "step_name": "step5_labeling",
                "validation_passed": False,
                "error": f"Error reading file: {e}",
            }
            
    except Exception as e:
        logger.exception(f"❌ Error in Step 5 validation: {e}")
        return {
            "step_name": "step5_labeling",
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