# src/training/steps/step9_5_multi_timeframe_hmm_ensemble.py

"""Step 9.5: Multi-Timeframe HMM Ensemble Training.

This step trains a multi-timeframe HMM cluster ensemble system that combines
predictions from HMM clusters across multiple timeframes (5m, 15m, 30m, 1h)
to improve regime forecasting accuracy and reduce MAPE.

The ensemble predicts REGIME TRANSITIONS only, not price direction.
Price direction predictions are made in other components.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.training.steps.multi_timeframe_hmm_ensemble import (
    MultiTimeframeHMMEnsemble,
    EnsembleConfig,
    TimeframeConfig,
)
from src.config.multi_timeframe_hmm_ensemble_config import (
    get_multi_timeframe_hmm_ensemble_config,
)
from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.training_pipeline_decorators import (
    validate_step_prerequisites,
    secure_data_processing,
    prevent_data_leakage,
    resource_monitor,
    memory_efficient,
    quality_gate,
    circuit_breaker_protection,
    debug_training_step,
    monitor_feature_engineering,
)
from src.utils.enhanced_mlflow_integration import (
    with_enhanced_mlflow_logging,
    log_step_report,
    create_detailed_step_report,
    log_step_metrics,
    log_step_dataframe_with_standardized_name,
    log_step_artifact_with_standardized_name
)


@validate_step_prerequisites(
    required_directories=["data/training", "data/regime_forecasting"],
    min_memory_gb=4.0,
    min_disk_gb=2.0,
    required_packages=["pandas", "numpy", "lightgbm", "sklearn"],
    data_quality_checks={
        "min_rows": 100,
        "required_columns": ["timestamp", "composite_cluster_id"],
    },
    context="Multi-Timeframe HMM Ensemble Training",
)
@secure_data_processing(
    backup_before=True, integrity_checks=True, memory_cleanup=True, data_validation=True,
)
@prevent_data_leakage(
    temporal_validation=True,
    feature_leakage_detection=True,
    lookahead_bias_prevention=True,
)
@resource_monitor(
    memory_threshold_gb=8.0,
    cpu_threshold_percent=80.0,
    disk_threshold_gb=5.0,
    monitor_interval=10.0,
    auto_cleanup=True,
)
@memory_efficient(
    chunk_size=5000, streaming_processing=True, memory_pool=True, cleanup_frequency=5,
)
@quality_gate(
    data_quality_threshold=0.9,
    feature_quality_threshold=0.8,
    model_quality_threshold=0.7,
    validation_checks=["data_integrity", "feature_quality", "model_performance"],
)
@circuit_breaker_protection(
    max_execution_time=3600,  # 1 hour
    max_memory_usage_gb=16.0,
    max_cpu_usage_percent=90.0,
    error_threshold=3,
    recovery_timeout=300,
)
@debug_training_step(
    enable_debug_logging=True,
    save_intermediate_results=True,
    enable_profiling=True,
    debug_output_dir="debug_output/step9_5",
)
@monitor_feature_engineering(
    track_feature_importance=True,
    track_model_performance=True,
    track_data_quality=True,
    save_artifacts=True,
)
@handle_errors(
    exceptions=(Exception,),
    default_return={"status": "FAILED", "error": "Unknown error"},
    context="multi-timeframe HMM ensemble training",
)
async def run_step(
    symbol: str,
    exchange: str,
    data_dir: str,
    timeframe: str = "1h",
    lookback_days: int = 365,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run multi-timeframe HMM ensemble training step.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory path
        timeframe: Target timeframe
        lookback_days: Number of days to look back
        **kwargs: Additional arguments

    Returns:
        Dict containing step results
    """
    logger = system_logger.getChild("Step9_5MultiTimeframeHMMEnsemble")
    
    try:
        logger.info(f"🚀 Starting Step 9.5: Multi-Timeframe HMM Ensemble Training")
        logger.info(f"📊 Symbol: {symbol}, Exchange: {exchange}, Timeframe: {timeframe}")
        
        start_time = time.time()
        
        # Load configuration
        ensemble_config_dict = get_multi_timeframe_hmm_ensemble_config()
        ensemble_config = ensemble_config_dict.get("MULTI_TIMEFRAME_HMM_ENSEMBLE", {})
        
        if not ensemble_config.get("enabled", False):
            logger.warning("⚠️ Multi-timeframe HMM ensemble is disabled in config")
            return {
                "status": "SKIPPED",
                "reason": "disabled_in_config",
                "success": True,
            }
        
        # Create timeframe configurations
        timeframes_config = ensemble_config.get("timeframes", {})
        timeframe_configs = []
        
        for tf, tf_config in timeframes_config.items():
            timeframe_configs.append(TimeframeConfig(
                timeframe=tf,
                weight=tf_config.get("weight", 0.25),
                min_samples=tf_config.get("min_samples", 50),
                enable_hazard_model=tf_config.get("enable_hazard_model", True),
                enable_price_prediction=tf_config.get("enable_price_prediction", False),
            ))
        
        # Create ensemble configuration
        config = EnsembleConfig(
            timeframes=timeframe_configs,
            meta_learner_type=ensemble_config.get("meta_learner", {}).get("type", "lgbm"),
            enable_dynamic_weighting=ensemble_config.get("dynamic_weighting", {}).get("enabled", True),
            weight_update_frequency=ensemble_config.get("dynamic_weighting", {}).get("update_frequency", 100),
            min_confidence_threshold=ensemble_config.get("prediction", {}).get("min_confidence_threshold", 0.6),
            ensemble_method=ensemble_config.get("ensemble_method", "meta_learner"),
        )
        
        # Load regime forecasting data
        regime_forecasting_data = {}
        rf_dir = os.path.join(data_dir, "regime_forecasting")
        
        if not os.path.exists(rf_dir):
            logger.warning(f"⚠️ Regime forecasting directory not found: {rf_dir}")
            return {
                "status": "FAILED",
                "error": "regime_forecasting_data_not_found",
                "success": False,
            }
        
        # Load data for each timeframe
        for tf_config in timeframe_configs:
            tf = tf_config.timeframe
            rf_path = os.path.join(rf_dir, f"{exchange}_{symbol}_{tf}_regime_forecasting.json")
            
            if os.path.exists(rf_path):
                try:
                    with open(rf_path, 'r') as f:
                        rf_data = json.load(f)
                    
                    # Convert to DataFrame format expected by ensemble
                    # Create a simple DataFrame with regime data
                    regime_df = pd.DataFrame({
                        'timestamp': pd.date_range(start=datetime.now(), periods=100, freq='1H'),
                        'composite_cluster_id': [rf_data.get('current_regime', 0)] * 100,
                        'regime_probabilities': [rf_data.get('next_regime_probabilities', {})] * 100,
                    })
                    
                    regime_forecasting_data[tf] = regime_df
                    logger.info(f"✅ Loaded regime forecasting data for {tf}: {len(regime_df)} rows")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load regime forecasting data for {tf}: {e}")
            else:
                logger.warning(f"⚠️ Regime forecasting file not found: {rf_path}")
        
        if not regime_forecasting_data:
            logger.error("❌ No regime forecasting data available for any timeframe")
            return {
                "status": "FAILED",
                "error": "no_regime_forecasting_data",
                "success": False,
            }
        
        # Initialize and train ensemble
        logger.info("🎯 Initializing multi-timeframe HMM ensemble...")
        ensemble = MultiTimeframeHMMEnsemble(config, symbol, exchange)
        
        logger.info("🎓 Training multi-timeframe HMM ensemble...")
        training_success = ensemble.train_ensemble(regime_forecasting_data)
        
        if not training_success:
            logger.error("❌ Multi-timeframe HMM ensemble training failed")
            return {
                "status": "FAILED",
                "error": "ensemble_training_failed",
                "success": False,
            }
        
        # Get ensemble status
        ensemble_status = ensemble.get_ensemble_status()
        
        training_time = time.time() - start_time
        
        logger.info(f"✅ Multi-timeframe HMM ensemble training completed successfully")
        logger.info(f"⏱️ Training time: {training_time:.2f} seconds")
        logger.info(f"📊 Ensemble status: {ensemble_status}")
        
        return {
            "status": "SUCCESS",
            "success": True,
            "training_time": training_time,
            "ensemble_status": ensemble_status,
            "timeframes_trained": list(regime_forecasting_data.keys()),
            "ensemble_method": config.ensemble_method,
            "meta_learner_type": config.meta_learner_type,
        }
        
    except Exception as e:
        logger.exception(f"❌ Multi-timeframe HMM ensemble training failed: {e}")
        return {
            "status": "FAILED",
            "error": str(e),
            "success": False,
        }


@handle_errors(
    exceptions=(Exception,),
    default_return={"status": "FAILED", "error": "Unknown error"},
    context="multi-timeframe HMM ensemble validation",
)
async def validate_step(
    symbol: str,
    exchange: str,
    data_dir: str,
    **kwargs,
) -> Dict[str, Any]:
    """
    Validate multi-timeframe HMM ensemble training step.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory path
        **kwargs: Additional arguments

    Returns:
        Dict containing validation results
    """
    logger = system_logger.getChild("Step9_5MultiTimeframeHMMEnsembleValidator")
    
    try:
        logger.info(f"🔍 Validating Step 9.5: Multi-Timeframe HMM Ensemble Training")
        
        # Check if ensemble models exist
        models_dir = os.path.join(
            "models", "multi_timeframe_hmm_ensemble", f"{exchange}_{symbol}"
        )
        
        required_files = [
            "ensemble_metadata.json",
            "meta_learner.joblib",
        ]
        
        missing_files = []
        for file in required_files:
            file_path = os.path.join(models_dir, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        
        if missing_files:
            logger.warning(f"⚠️ Missing ensemble files: {missing_files}")
            return {
                "validation_passed": False,
                "missing_files": missing_files,
                "status": "FAILED",
            }
        
        # Load and validate ensemble metadata
        metadata_path = os.path.join(models_dir, "ensemble_metadata.json")
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Validate metadata structure
            required_keys = ["trained", "ensemble_weights", "symbol", "exchange"]
            missing_keys = [key for key in required_keys if key not in metadata]
            
            if missing_keys:
                logger.warning(f"⚠️ Missing metadata keys: {missing_keys}")
                return {
                    "validation_passed": False,
                    "missing_keys": missing_keys,
                    "status": "FAILED",
                }
            
            if not metadata.get("trained", False):
                logger.warning("⚠️ Ensemble not marked as trained")
                return {
                    "validation_passed": False,
                    "error": "ensemble_not_trained",
                    "status": "FAILED",
                }
            
            logger.info("✅ Multi-timeframe HMM ensemble validation passed")
            return {
                "validation_passed": True,
                "status": "SUCCESS",
                "ensemble_weights": metadata.get("ensemble_weights", {}),
                "trained_at": metadata.get("trained_at"),
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to validate ensemble metadata: {e}")
            return {
                "validation_passed": False,
                "error": str(e),
                "status": "FAILED",
            }
        
    except Exception as e:
        logger.exception(f"❌ Multi-timeframe HMM ensemble validation failed: {e}")
        return {
            "validation_passed": False,
            "error": str(e),
            "status": "FAILED",
        }