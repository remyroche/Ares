# src/training/steps/step3_hmm_regime_discovery_validator.py

"""
Validator for Step 3: HMM Regime Discovery

This validator ensures that the HMM regime discovery step has completed successfully
and generated all required artifacts for downstream steps.
"""

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors


logger = system_logger.getChild("Step3.HMMRegimeDiscovery.Validator")


@handle_errors(exceptions=(Exception,), default_return=False)
async def run_validator(
    training_input: Dict[str, Any], pipeline_state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate Step 3: HMM Regime Discovery completion and artifacts.
    
    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state
        
    Returns:
        Validation result dictionary
    """
    start_time = time.time()
    
    try:
        # Extract parameters
        symbol = training_input.get("symbol", "")
        exchange = training_input.get("exchange", "")
        timeframe = training_input.get("timeframe", "")
        data_dir = training_input.get("data_dir", "data/training")
        
        if not all([symbol, exchange, timeframe]):
            return {
                "validation_passed": False,
                "error": "Missing required parameters: symbol, exchange, timeframe",
                "validation_results": {}
            }
        
        logger.info(f"🔍 Validating Step 3: HMM Regime Discovery for {symbol} on {exchange} ({timeframe})")
        
        # Check pipeline state
        hmm_state = pipeline_state.get("hmm_regime_discovery", {})
        if not hmm_state.get("completed", False):
            return {
                "validation_passed": False,
                "error": "Step 3 HMM regime discovery not completed in pipeline state",
                "validation_results": {}
            }
        
        # Define required artifacts
        required_artifacts = [
            f"{exchange}_{symbol}_hmm_block_states_{timeframe}.parquet",
            f"{exchange}_{symbol}_hmm_composite_clusters_{timeframe}.parquet",
            f"{exchange}_{symbol}_hmm_composite_intensity_{timeframe}.parquet",
            f"{exchange}_{symbol}_hmm_composite_meta_{timeframe}.json"
        ]
        
        # Check for required artifacts
        missing_artifacts = []
        artifact_info = {}
        
        for artifact in required_artifacts:
            artifact_path = Path(data_dir) / artifact
            if artifact_path.exists():
                # Get file info
                stat = artifact_path.stat()
                artifact_info[artifact] = {
                    "exists": True,
                    "size_bytes": stat.st_size,
                    "modified_time": stat.st_mtime
                }
                
                # Validate file content for parquet files
                if artifact.endswith('.parquet'):
                    try:
                        df = pd.read_parquet(artifact_path)
                        artifact_info[artifact]["rows"] = len(df)
                        artifact_info[artifact]["columns"] = list(df.columns)
                        
                        # Check for required columns based on artifact type
                        if "block_states" in artifact:
                            required_cols = [col for col in df.columns if "state_id" in col or "p_state_" in col]
                            if not required_cols:
                                missing_artifacts.append(f"{artifact} - missing state columns")
                        
                        elif "composite_clusters" in artifact:
                            if "composite_cluster_id" not in df.columns:
                                missing_artifacts.append(f"{artifact} - missing composite_cluster_id column")
                        
                        elif "intensity" in artifact:
                            intensity_cols = [col for col in df.columns if "intensity_" in col]
                            if not intensity_cols:
                                missing_artifacts.append(f"{artifact} - missing intensity columns")
                                
                    except Exception as e:
                        missing_artifacts.append(f"{artifact} - failed to read: {str(e)}")
                
                # Validate JSON metadata
                elif artifact.endswith('.json'):
                    try:
                        with open(artifact_path, 'r') as f:
                            meta = json.load(f)
                        artifact_info[artifact]["metadata"] = meta
                        
                        # Check required metadata fields
                        required_meta_fields = ['symbol', 'exchange', 'timeframe', 'blocks_used', 'n_composite_clusters']
                        missing_fields = [field for field in required_meta_fields if field not in meta]
                        if missing_fields:
                            missing_artifacts.append(f"{artifact} - missing metadata fields: {missing_fields}")
                            
                    except Exception as e:
                        missing_artifacts.append(f"{artifact} - failed to read JSON: {str(e)}")
            else:
                missing_artifacts.append(artifact)
                artifact_info[artifact] = {"exists": False}
        
        # Check for HMM regimes directory artifacts (required for Step 4)
        hmm_regimes_dir = Path(data_dir) / "hmm_regimes"
        hmm_regimes_artifacts = [
            f"{exchange}_{symbol}_hmm_composite_clusters_{timeframe}.parquet"
        ]
        
        for artifact in hmm_regimes_artifacts:
            artifact_path = hmm_regimes_dir / artifact
            if artifact_path.exists():
                artifact_info[f"hmm_regimes/{artifact}"] = {
                    "exists": True,
                    "size_bytes": artifact_path.stat().st_size
                }
            else:
                missing_artifacts.append(f"hmm_regimes/{artifact}")
                artifact_info[f"hmm_regimes/{artifact}"] = {"exists": False}
        
        # Determine validation result
        validation_passed = len(missing_artifacts) == 0
        
        # Calculate validation metrics
        total_artifacts = len(required_artifacts) + len(hmm_regimes_artifacts)
        artifacts_found = total_artifacts - len(missing_artifacts)
        artifact_coverage = artifacts_found / total_artifacts if total_artifacts > 0 else 0
        
        validation_results = {
            "validation_passed": validation_passed,
            "artifact_coverage": artifact_coverage,
            "artifacts_found": artifacts_found,
            "total_artifacts": total_artifacts,
            "missing_artifacts": missing_artifacts,
            "artifact_info": artifact_info,
            "validation_time": time.time() - start_time
        }
        
        if validation_passed:
            logger.info(f"✅ Step 3 validation passed: {artifacts_found}/{total_artifacts} artifacts found")
        else:
            logger.error(f"❌ Step 3 validation failed: {len(missing_artifacts)} missing artifacts")
            logger.error(f"   Missing: {missing_artifacts}")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"❌ Step 3 validator error: {e}")
        return {
            "validation_passed": False,
            "error": str(e),
            "validation_results": {},
            "validation_time": time.time() - start_time
        }


# Legacy function for backward compatibility
async def run_step_validator(
    training_input: Dict[str, Any], pipeline_state: Dict[str, Any]
) -> Dict[str, Any]:
    """Legacy function name for backward compatibility."""
    return await run_validator(training_input, pipeline_state)
