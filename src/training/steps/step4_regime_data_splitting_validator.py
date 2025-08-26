#!/usr/bin/env python3
"""Validator for Step 4: Regime Data Splitting.

This module validates the regime data splitting step outputs with support for 10+ regimes.
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

logger = system_logger.getChild("Step4RegimeDataSplittingValidator")


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
    """Run validation for Step 4: Regime Data Splitting.

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
        
        # Check if regime data directory exists
        regime_base_dir = Path(data_dir) / "training" / "regimes" / f"{exchange}_{symbol}_{timeframe}"
        
        if not regime_base_dir.exists():
            logger.error(f"❌ Regime data directory not found: {regime_base_dir}")
            return {
                "step_name": "step4_regime_data_splitting",
                "validation_passed": False,
                "error": f"Regime data directory not found: {regime_base_dir}",
            }
        
        # Check for regime metadata file
        metadata_file = regime_base_dir / "regime_metadata.json"
        if not metadata_file.exists():
            logger.error(f"❌ Regime metadata file not found: {metadata_file}")
            return {
                "step_name": "step4_regime_data_splitting",
                "validation_passed": False,
                "error": f"Regime metadata file not found: {metadata_file}",
            }
        
        # Load metadata
        try:
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            total_regimes = metadata.get("total_regimes", 0)
            regime_ids = metadata.get("regime_ids", [])
            
            logger.info(f"📊 Found {total_regimes} regimes: {regime_ids}")
            
        except Exception as e:
            logger.error(f"❌ Error reading regime metadata: {e}")
            return {
                "step_name": "step4_regime_data_splitting",
                "validation_passed": False,
                "error": f"Error reading metadata: {e}",
            }
        
        # Validate regime count
        if total_regimes < 3:
            logger.error(f"❌ Too few regimes: {total_regimes} (minimum 3 required)")
            return {
                "step_name": "step4_regime_data_splitting",
                "validation_passed": False,
                "error": f"Too few regimes: {total_regimes} (minimum 3 required)",
            }
        
        if total_regimes > 20:
            logger.warning(f"⚠️ Many regimes detected: {total_regimes} (maximum 20 supported)")
        
        # Validate each regime directory and data
        regime_validation_results = {}
        all_regimes_valid = True
        total_data_points = 0
        
        for regime_id in regime_ids:
            regime_dir = regime_base_dir / f"regime_{regime_id}"
            
            if not regime_dir.exists():
                logger.error(f"❌ Regime {regime_id} directory not found: {regime_dir}")
                regime_validation_results[regime_id] = {
                    "valid": False,
                    "error": "Directory not found"
                }
                all_regimes_valid = False
                continue
            
            # Check for regime data file
            regime_data_file = regime_dir / "regime_data.parquet"
            if not regime_data_file.exists():
                logger.error(f"❌ Regime {regime_id} data file not found: {regime_data_file}")
                regime_validation_results[regime_id] = {
                    "valid": False,
                    "error": "Data file not found"
                }
                all_regimes_valid = False
                continue
            
            # Check for regime stats file
            regime_stats_file = regime_dir / "regime_stats.json"
            if not regime_stats_file.exists():
                logger.warning(f"⚠️ Regime {regime_id} stats file not found: {regime_stats_file}")
            
            # Validate regime data
            try:
                import pandas as pd
                regime_data = pd.read_parquet(regime_data_file)
                
                data_points = len(regime_data)
                total_data_points += data_points
                
                # Check required columns
                required_columns = ["timestamp", "open", "high", "low", "close", "volume", "composite_cluster_id"]
                missing_columns = [col for col in required_columns if col not in regime_data.columns]
                
                if missing_columns:
                    logger.error(f"❌ Regime {regime_id} missing columns: {missing_columns}")
                    regime_validation_results[regime_id] = {
                        "valid": False,
                        "error": f"Missing columns: {missing_columns}"
                    }
                    all_regimes_valid = False
                    continue
                
                # Check data quality
                if data_points < 50:
                    logger.warning(f"⚠️ Regime {regime_id} has few data points: {data_points}")
                
                # Check for NaN values
                nan_count = regime_data[required_columns].isna().sum().sum()
                if nan_count > 0:
                    logger.warning(f"⚠️ Regime {regime_id} has {nan_count} NaN values")
                
                # Validate regime ID consistency
                regime_ids_in_data = regime_data['composite_cluster_id'].unique()
                if len(regime_ids_in_data) != 1 or regime_ids_in_data[0] != regime_id:
                    logger.error(f"❌ Regime {regime_id} has inconsistent regime IDs: {regime_ids_in_data}")
                    regime_validation_results[regime_id] = {
                        "valid": False,
                        "error": f"Inconsistent regime IDs: {regime_ids_in_data}"
                    }
                    all_regimes_valid = False
                    continue
                
                # Load and validate stats
                if regime_stats_file.exists():
                    try:
                        with open(regime_stats_file, 'r') as f:
                            stats = json.load(f)
                        
                        if stats.get("regime_id") != regime_id:
                            logger.warning(f"⚠️ Regime {regime_id} stats file has wrong regime ID: {stats.get('regime_id')}")
                        
                        if stats.get("data_points") != data_points:
                            logger.warning(f"⚠️ Regime {regime_id} stats data points mismatch: {stats.get('data_points')} vs {data_points}")
                    
                    except Exception as e:
                        logger.warning(f"⚠️ Error reading regime {regime_id} stats: {e}")
                
                regime_validation_results[regime_id] = {
                    "valid": True,
                    "data_points": data_points,
                    "nan_count": nan_count,
                    "file_size_mb": regime_data_file.stat().st_size / (1024 * 1024)
                }
                
                logger.info(f"✅ Regime {regime_id}: {data_points} data points, {nan_count} NaN values")
                
            except Exception as e:
                logger.error(f"❌ Error validating regime {regime_id}: {e}")
                regime_validation_results[regime_id] = {
                    "valid": False,
                    "error": str(e)
                }
                all_regimes_valid = False
        
        # Calculate validation summary
        valid_regimes = sum(1 for result in regime_validation_results.values() if result.get("valid", False))
        invalid_regimes = total_regimes - valid_regimes
        
        logger.info(f"📊 Regime validation summary:")
        logger.info(f"   - Total regimes: {total_regimes}")
        logger.info(f"   - Valid regimes: {valid_regimes}")
        logger.info(f"   - Invalid regimes: {invalid_regimes}")
        logger.info(f"   - Total data points: {total_data_points}")
        
        # Determine overall validation result
        validation_passed = all_regimes_valid and total_regimes >= 3
        
        if validation_passed:
            logger.info("✅ Step 4: Regime Data Splitting validation passed")
        else:
            logger.error("❌ Step 4: Regime Data Splitting validation failed")
        
        return {
            "step_name": "step4_regime_data_splitting",
            "validation_passed": validation_passed,
            "total_regimes": total_regimes,
            "valid_regimes": valid_regimes,
            "invalid_regimes": invalid_regimes,
            "total_data_points": total_data_points,
            "regime_validation_results": regime_validation_results,
            "metadata": metadata,
        }
        
    except Exception as e:
        logger.exception(f"❌ Error in Step 4 validation: {e}")
        return {
            "step_name": "step4_regime_data_splitting",
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