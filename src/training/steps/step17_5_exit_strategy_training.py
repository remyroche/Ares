# src/training/steps/step18_exit_strategy_training.py

"""
Step 17.5: Exit Strategy Training.
Trains ensemble ML models for trend reversal detection and exit timing optimization.
This step creates models that continuously calculate price action and determine
the likelihood of upcoming trend reversals for position closure decisions.
Uses all features from step6-7 and focuses on 1m and 5m timeframes.
"""

import asyncio
import json
import os
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

# Import exit strategy components
from src.tactician.exit_strategy_feature_engineering import ExitStrategyFeatureEngineering
from src.tactician.multi_timeframe_entry_models import MultiTimeframeEntryModels

# Import training pipeline decorators
from src.utils.training_pipeline_decorators import (
    circuit_breaker_protection,
    debug_training_step,
    memory_efficient,
    prevent_data_leakage,
    quality_gate,
    resource_monitor,
    secure_data_processing,
    validate_step_output,
    validate_step_prerequisites,
    monitor_feature_engineering,
)

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.decorators import guard_dataframe_nulls, with_tracing_span
from src.utils.enhanced_mlflow_integration import (
    with_enhanced_mlflow_logging, 
    log_step_dataframe, 
    log_step_metrics,
    log_step_dataframe_with_standardized_name,
    log_step_report,
    log_step_artifact_with_standardized_name
)


@validate_step_prerequisites(
    required_directories=["data/training", "data/hmm_regimes", "models"],
    min_memory_gb=4.0,
    min_disk_gb=2.0,
    required_packages=["pandas", "numpy", "sklearn", "xgboost", "lightgbm"],
    data_quality_checks={
        "min_rows": 5000,
        "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
        "data_validation": {
            "check_negative_prices": True,
            "check_price_relationships": True,
            "max_missing_ratio": 0.05,
            "min_data_points": 1000
        }
    },
    context="Exit Strategy Training",
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
    memory_threshold_gb=16.0,
    cpu_threshold_percent=80.0,
    disk_threshold_gb=10.0,
    monitor_interval=30.0,
    auto_cleanup=True,
)
@memory_efficient(
    chunk_size=10000, streaming_processing=True, memory_pool=True, cleanup_frequency=10,
)
@quality_gate(
    data_quality_threshold=0.95,
    feature_quality_threshold=0.85,
    model_quality_threshold=0.75,
    validation_checks=["data_integrity", "feature_quality", "model_performance"],
)
@debug_training_step(
    enable_debug_logging=True,
    save_intermediate_results=True,
    validate_predictions=True,
)
@circuit_breaker_protection(
    max_execution_time_hours=4.0,
    max_memory_usage_gb=32.0,
    max_cpu_usage_percent=90.0,
    error_threshold=5,
)
@monitor_feature_engineering(
    track_feature_importance=True,
    monitor_feature_stability=True,
    validate_feature_distributions=True,
)
@with_enhanced_mlflow_logging("step17_5_exit_strategy_training")
async def step17_5_exit_strategy_training(
    training_input: Dict[str, Any],
    pipeline_state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute Step 18: Exit Strategy Training.
    
    This step trains ensemble ML models for:
    1. Trend reversal detection
    2. Exit timing optimization
    3. Multi-timeframe entry models (1m, 5m, 15m)
    4. Position closure decision making
    
    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state
        
    Returns:
        Dict containing training results and model artifacts
    """
    try:
        logger = system_logger.getChild("Step17_5ExitStrategyTraining")
        logger.info("🚀 Starting Step 17.5: Exit Strategy Training")
        
        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")
        models_dir = training_input.get("models_dir", "models/exit_strategy")
        
        # Create output directories
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        
        # Load data from step6-7 (all features)
        data_loader = await _load_step6_7_data(data_dir, symbol, exchange, logger)
        if not data_loader["success"]:
            raise ValueError(f"Failed to load step6-7 data: {data_loader['error']}")
        
        # Use all features from step6-7 (no need to create new features)
        logger.info("📊 Using comprehensive features from step6-7 for ML training")
        
        # Initialize exit strategy components with existing features
        exit_strategy_components = await _initialize_exit_strategy_components(
            training_input, logger
        )
        if not exit_strategy_components["success"]:
            raise ValueError(f"Failed to initialize components: {exit_strategy_components['error']}")
        
        # Train exit strategy models
        training_results = await _train_exit_strategy_models(
            data_loader["data"],
            exit_strategy_components["components"],
            models_dir,
            logger
        )
        
        # Validate and save results
        validation_results = await _validate_training_results(
            training_results,
            exit_strategy_components["components"],
            logger
        )
        
        # Prepare output
        output = {
            "step_name": "step17_5_exit_strategy_training",
            "status": "COMPLETED",
            "timestamp": datetime.now(UTC).isoformat(),
            "training_results": training_results,
            "validation_results": validation_results,
            "model_artifacts": {
                "models_dir": models_dir,
                "feature_engineering_config": exit_strategy_components["feature_config"],
                "entry_models_config": exit_strategy_components["entry_config"]
            },
            "performance_metrics": {
                "overall_accuracy": validation_results.get("overall_accuracy", 0.0),
                "reversal_detection_accuracy": validation_results.get("reversal_accuracy", 0.0),
                "exit_timing_accuracy": validation_results.get("exit_timing_accuracy", 0.0),
                "multi_timeframe_accuracy": validation_results.get("multi_timeframe_accuracy", 0.0)
            }
        }
        
        # Log results to MLflow
        await _log_training_results(output, logger)
        
        logger.info("✅ Step 17.5: Exit Strategy Training completed successfully")
        return output
        
    except Exception as e:
        logger.error(f"❌ Step 17.5: Exit Strategy Training failed: {e}")
        return {
            "step_name": "step17_5_exit_strategy_training",
            "status": "FAILED",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat()
        }


async def _load_step6_7_data(
    data_dir: str, 
    symbol: str, 
    exchange: str, 
    logger: Any
) -> Dict[str, Any]:
    """
    Load comprehensive data from step6-7 with all features.
    
    Args:
        data_dir: Data directory path
        symbol: Trading symbol
        exchange: Exchange name
        logger: Logger instance
        
    Returns:
        Dict containing loaded data with all step6-7 features
    """
    try:
        logger.info("📊 Loading step6-7 data with all features...")
        
        # Load step6 processed data (basic features)
        step6_data_path = Path(data_dir) / f"{symbol}_{exchange}_step6_processed.parquet"
        if not step6_data_path.exists():
            return {
                "success": False,
                "error": f"Step6 processed data not found: {step6_data_path}"
            }
        
        # Load step7 enhanced data (advanced features)
        step7_data_path = Path(data_dir) / f"{symbol}_{exchange}_step7_enhanced.parquet"
        
        # Start with step6 data
        df = pd.read_parquet(step6_data_path)
        logger.info(f"   Loaded {len(df)} rows from step6 data with {len(df.columns)} features")
        
        # Merge step7 features if available
        if step7_data_path.exists():
            step7_df = pd.read_parquet(step7_data_path)
            logger.info(f"   Loaded step7 data with {len(step7_df.columns)} features")
            
            # Merge on timestamp/index
            if 'timestamp' in df.columns and 'timestamp' in step7_df.columns:
                df = df.merge(step7_df, on='timestamp', how='left', suffixes=('', '_step7'))
            else:
                # Merge on index
                df = df.join(step7_df, how='left')
            
            logger.info(f"   Combined data has {len(df.columns)} total features")
        
        # Load HMM regime data if available
        hmm_data_path = Path(data_dir) / f"{symbol}_{exchange}_hmm_regimes.parquet"
        hmm_regimes = None
        if hmm_data_path.exists():
            hmm_regimes = pd.read_parquet(hmm_data_path)
            logger.info(f"   Loaded HMM regime data: {len(hmm_regimes)} rows")
        
        # Create multi-timeframe data (1m and 5m only)
        timeframe_data = await _create_multi_timeframe_data(df, logger)
        
        return {
            "success": True,
            "data": {
                "main_data": df,
                "hmm_regimes": hmm_regimes,
                "timeframe_data": timeframe_data
            }
        }
        
    except Exception as e:
        logger.error(f"Step6-7 data loading failed: {e}")
        return {"success": False, "error": str(e)}


async def _create_multi_timeframe_data(df: pd.DataFrame, logger: Any) -> Dict[str, pd.DataFrame]:
    """
    Create multi-timeframe data for entry models.
    
    Args:
        df: Main dataframe
        logger: Logger instance
        
    Returns:
        Dict of dataframes for different timeframes
    """
    try:
        logger.info("🕐 Creating multi-timeframe data...")
        
        # Ensure timestamp index
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        
        timeframe_data = {}
        
        # 1-minute data (original)
        timeframe_data["1m"] = df.copy()
        logger.info(f"   1m data: {len(timeframe_data['1m'])} rows")
        
        # 5-minute data (resampled)
        df_5m = df.resample('5T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        timeframe_data["5m"] = df_5m
        logger.info(f"   5m data: {len(timeframe_data['5m'])} rows")
        

        
        return timeframe_data
        
    except Exception as e:
        logger.error(f"Multi-timeframe data creation failed: {e}")
        return {}


async def _initialize_exit_strategy_components(
    training_input: Dict[str, Any],
    logger: Any
) -> Dict[str, Any]:
    """
    Initialize exit strategy components.
    
    Args:
        training_input: Training input parameters
        logger: Logger instance
        
    Returns:
        Dict containing initialized components
    """
    try:
        logger.info("🔧 Initializing exit strategy components...")
        
        # Create configuration (1m and 5m only)
        config = {
            "exit_strategy": {
                "feature_engineering": {
                    "timeframes": ["1m", "5m"],
                    "lookback_periods": [10, 20, 50, 100],
                    "enable_profit_decay": True,
                    "enable_time_decay": True,
                    "enable_market_regime": True
                },
                "model_training": {
                    "ensemble_size": 7,
                    "cross_validation_folds": 5,
                    "min_accuracy_threshold": 0.75,
                    "feature_selection_method": "mutual_info"
                }
            },
            "entry_models": {
                "timeframes": ["1m", "5m"],
                "models_dir": "models/entry_models",
                "min_accuracy": 0.75,
                "ensemble_size": 5
            }
        }
        
        # Note: Feature engineering is now handled in step6, so we skip it here
        # All features are already available from step6-7 data
        logger.info("📊 Skipping feature engineering - using step6-7 features")
        
        # Initialize multi-timeframe entry models
        entry_models = MultiTimeframeEntryModels(config)
        if not await entry_models.initialize():
            return {
                "success": False,
                "error": "Entry models initialization failed"
            }
        
        logger.info("✅ Exit strategy components initialized successfully")
        
        return {
            "success": True,
            "components": {
                "entry_models": entry_models
            },
            "feature_config": config["exit_strategy"]["feature_engineering"],
            "entry_config": config["entry_models"]
        }
        
    except Exception as e:
        logger.error(f"Component initialization failed: {e}")
        return {"success": False, "error": str(e)}


async def _train_exit_strategy_models(
    data: Dict[str, Any],
    components: Dict[str, Any],
    models_dir: str,
    logger: Any
) -> Dict[str, Any]:
    """
    Train exit strategy models.
    
    Args:
        data: Training data
        components: Initialized components
        models_dir: Models directory
        logger: Logger instance
        
    Returns:
        Dict containing training results
    """
    try:
        logger.info("🎯 Training exit strategy models...")
        
        training_results = {}
        
        # Train multi-timeframe entry models
        logger.info("   Training multi-timeframe entry models...")
        entry_training_results = await components["entry_models"].train_models(
            data["timeframe_data"]
        )
        training_results["entry_models"] = entry_training_results
        
        # Train trend reversal detection models using step6-7 features
        logger.info("   Training trend reversal detection models...")
        reversal_training_results = await _train_reversal_detection_models(
            data["main_data"],
            models_dir,
            logger
        )
        training_results["reversal_models"] = reversal_training_results
        
        # Train exit timing models using step6-7 features
        logger.info("   Training exit timing models...")
        exit_timing_results = await _train_exit_timing_models(
            data["main_data"],
            models_dir,
            logger
        )
        training_results["exit_timing_models"] = exit_timing_results
        
        # Train ensemble exit decision model using step6-7 features
        logger.info("   Training ensemble exit decision model...")
        ensemble_results = await _train_ensemble_exit_model(
            data["main_data"],
            models_dir,
            logger
        )
        training_results["ensemble_model"] = ensemble_results
        
        logger.info("✅ Exit strategy model training completed")
        return training_results
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return {"error": str(e)}


async def _train_reversal_detection_models(
    df: pd.DataFrame,
    models_dir: str,
    logger: Any
) -> Dict[str, Any]:
    """
    Train trend reversal detection models using step6-7 features.
    
    Args:
        df: Training dataframe with step6-7 features
        models_dir: Models directory
        logger: Logger instance
        
    Returns:
        Dict containing training results
    """
    try:
        logger.info("      Training reversal detection models...")
        
        # Use existing features from step6-7 (no additional feature engineering needed)
        features_df = df.copy()
        
        # Create reversal labels
        reversal_labels = await _create_reversal_labels(df)
        
        if reversal_labels is None:
            return {"success": False, "error": "Failed to create reversal labels"}
        
        # Remove NaN values
        valid_mask = ~(features_df.isna().any(axis=1) | pd.isna(reversal_labels))
        features_df = features_df[valid_mask]
        reversal_labels = reversal_labels[valid_mask]
        
        # Train ensemble model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Cross-validation
        cv_scores = cross_val_score(model, features_df, reversal_labels, cv=5)
        accuracy = cv_scores.mean()
        
        # Train final model
        model.fit(features_df, reversal_labels)
        
        # Save model
        model_path = Path(models_dir) / "reversal_detection_model.joblib"
        joblib.dump(model, model_path)
        
        logger.info(f"      Reversal detection model accuracy: {accuracy:.3f}")
        
        return {
            "success": True,
            "accuracy": accuracy,
            "cv_scores": cv_scores.tolist(),
            "model_path": str(model_path),
            "feature_count": features_df.shape[1],
            "sample_count": len(features_df)
        }
        
    except Exception as e:
        logger.error(f"Reversal detection training failed: {e}")
        return {"success": False, "error": str(e)}


async def _train_exit_timing_models(
    df: pd.DataFrame,
    models_dir: str,
    logger: Any
) -> Dict[str, Any]:
    """
    Train exit timing models using step6-7 features.
    
    Args:
        df: Training dataframe with step6-7 features
        models_dir: Models directory
        logger: Logger instance
        
    Returns:
        Dict containing training results
    """
    try:
        logger.info("      Training exit timing models...")
        
        # Use existing features from step6-7 (no additional feature engineering needed)
        features_df = df.copy()
        
        # Create exit timing labels
        exit_timing_labels = await _create_exit_timing_labels(df)
        
        if exit_timing_labels is None:
            return {"success": False, "error": "Failed to create exit timing labels"}
        
        # Remove NaN values
        valid_mask = ~(features_df.isna().any(axis=1) | pd.isna(exit_timing_labels))
        features_df = features_df[valid_mask]
        exit_timing_labels = exit_timing_labels[valid_mask]
        
        # Train ensemble model
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        
        # Cross-validation
        cv_scores = cross_val_score(model, features_df, exit_timing_labels, cv=5)
        accuracy = cv_scores.mean()
        
        # Train final model
        model.fit(features_df, exit_timing_labels)
        
        # Save model
        model_path = Path(models_dir) / "exit_timing_model.joblib"
        joblib.dump(model, model_path)
        
        logger.info(f"      Exit timing model accuracy: {accuracy:.3f}")
        
        return {
            "success": True,
            "accuracy": accuracy,
            "cv_scores": cv_scores.tolist(),
            "model_path": str(model_path),
            "feature_count": features_df.shape[1],
            "sample_count": len(features_df)
        }
        
    except Exception as e:
        logger.error(f"Exit timing training failed: {e}")
        return {"success": False, "error": str(e)}


async def _train_ensemble_exit_model(
    df: pd.DataFrame,
    models_dir: str,
    logger: Any
) -> Dict[str, Any]:
    """
    Train ensemble exit decision model using step6-7 features.
    
    Args:
        df: Training dataframe with step6-7 features
        models_dir: Models directory
        logger: Logger instance
        
    Returns:
        Dict containing training results
    """
    try:
        logger.info("      Training ensemble exit decision model...")
        
        # Use existing features from step6-7 (no additional feature engineering needed)
        features_df = df.copy()
        
        # Create ensemble labels (combine reversal and timing)
        ensemble_labels = await _create_ensemble_exit_labels(df)
        
        if ensemble_labels is None:
            return {"success": False, "error": "Failed to create ensemble labels"}
        
        # Remove NaN values
        valid_mask = ~(features_df.isna().any(axis=1) | pd.isna(ensemble_labels))
        features_df = features_df[valid_mask]
        ensemble_labels = ensemble_labels[valid_mask]
        
        # Train ensemble model
        import xgboost as xgb
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # Cross-validation
        cv_scores = cross_val_score(model, features_df, ensemble_labels, cv=5)
        accuracy = cv_scores.mean()
        
        # Train final model
        model.fit(features_df, ensemble_labels)
        
        # Save model
        model_path = Path(models_dir) / "ensemble_exit_model.joblib"
        joblib.dump(model, model_path)
        
        logger.info(f"      Ensemble exit model accuracy: {accuracy:.3f}")
        
        return {
            "success": True,
            "accuracy": accuracy,
            "cv_scores": cv_scores.tolist(),
            "model_path": str(model_path),
            "feature_count": features_df.shape[1],
            "sample_count": len(features_df)
        }
        
    except Exception as e:
        logger.error(f"Ensemble exit model training failed: {e}")
        return {"success": False, "error": str(e)}


async def _create_reversal_labels(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Create trend reversal labels.
    
    Args:
        df: Input dataframe
        
    Returns:
        Array of reversal labels
    """
    try:
        # Calculate future price changes
        future_returns = df['close'].shift(-5) / df['close'] - 1
        
        # Define reversal threshold
        reversal_threshold = 0.01  # 1% price change
        
        # Create labels: 1 for reversal, 0 for continuation
        labels = np.where(np.abs(future_returns) > reversal_threshold, 1, 0)
        
        # Remove NaN values
        labels = labels[:-5]  # Remove last 5 rows
        
        return labels
        
    except Exception as e:
        return None


async def _create_exit_timing_labels(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Create exit timing labels.
    
    Args:
        df: Input dataframe
        
    Returns:
        Array of exit timing labels
    """
    try:
        # Calculate future returns
        future_returns = df['close'].shift(-10) / df['close'] - 1
        
        # Define exit timing criteria
        # Exit when future returns are negative or small positive
        exit_threshold = 0.002  # 0.2% threshold
        
        # Create labels: 1 for exit, 0 for hold
        labels = np.where(future_returns < exit_threshold, 1, 0)
        
        # Remove NaN values
        labels = labels[:-10]  # Remove last 10 rows
        
        return labels
        
    except Exception as e:
        return None


async def _create_ensemble_exit_labels(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Create ensemble exit labels combining reversal and timing.
    
    Args:
        df: Input dataframe
        
    Returns:
        Array of ensemble labels
    """
    try:
        # Get individual labels
        reversal_labels = await _create_reversal_labels(df)
        timing_labels = await _create_exit_timing_labels(df)
        
        if reversal_labels is None or timing_labels is None:
            return None
        
        # Align lengths
        min_length = min(len(reversal_labels), len(timing_labels))
        reversal_labels = reversal_labels[:min_length]
        timing_labels = timing_labels[:min_length]
        
        # Combine labels: exit if either reversal or timing suggests exit
        ensemble_labels = np.where((reversal_labels == 1) | (timing_labels == 1), 1, 0)
        
        return ensemble_labels
        
    except Exception as e:
        return None


async def _validate_training_results(
    training_results: Dict[str, Any],
    components: Dict[str, Any],
    logger: Any
) -> Dict[str, Any]:
    """
    Validate training results.
    
    Args:
        training_results: Training results
        components: Initialized components
        logger: Logger instance
        
    Returns:
        Dict containing validation results
    """
    try:
        logger.info("🔍 Validating training results...")
        
        validation_results = {}
        
        # Validate entry models
        if "entry_models" in training_results:
            entry_validation = await _validate_entry_models(
                training_results["entry_models"],
                logger
            )
            validation_results["entry_models"] = entry_validation
        
        # Validate reversal models
        if "reversal_models" in training_results:
            reversal_validation = await _validate_reversal_models(
                training_results["reversal_models"],
                logger
            )
            validation_results["reversal_models"] = reversal_validation
        
        # Validate exit timing models
        if "exit_timing_models" in training_results:
            timing_validation = await _validate_exit_timing_models(
                training_results["exit_timing_models"],
                logger
            )
            validation_results["exit_timing_models"] = timing_validation
        
        # Calculate overall metrics
        overall_metrics = await _calculate_overall_metrics(validation_results, logger)
        validation_results["overall_metrics"] = overall_metrics
        
        logger.info("✅ Training validation completed")
        return validation_results
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {"error": str(e)}


async def _validate_entry_models(
    entry_results: Dict[str, Any],
    logger: Any
) -> Dict[str, Any]:
    """
    Validate entry models.
    
    Args:
        entry_results: Entry model results
        logger: Logger instance
        
    Returns:
        Dict containing validation results
    """
    try:
        validation = {}
        
        for timeframe, result in entry_results.items():
            if result.get("success", False):
                accuracy = result.get("accuracy", 0.0)
                validation[timeframe] = {
                    "valid": accuracy >= 0.75,
                    "accuracy": accuracy,
                    "status": "PASS" if accuracy >= 0.75 else "FAIL"
                }
            else:
                validation[timeframe] = {
                    "valid": False,
                    "accuracy": 0.0,
                    "status": "FAIL",
                    "error": result.get("error", "Unknown error")
                }
        
        return validation
        
    except Exception as e:
        logger.error(f"Entry model validation failed: {e}")
        return {"error": str(e)}


async def _validate_reversal_models(
    reversal_results: Dict[str, Any],
    logger: Any
) -> Dict[str, Any]:
    """
    Validate reversal models.
    
    Args:
        reversal_results: Reversal model results
        logger: Logger instance
        
    Returns:
        Dict containing validation results
    """
    try:
        if reversal_results.get("success", False):
            accuracy = reversal_results.get("accuracy", 0.0)
            return {
                "valid": accuracy >= 0.70,
                "accuracy": accuracy,
                "status": "PASS" if accuracy >= 0.70 else "FAIL"
            }
        else:
            return {
                "valid": False,
                "accuracy": 0.0,
                "status": "FAIL",
                "error": reversal_results.get("error", "Unknown error")
            }
        
    except Exception as e:
        logger.error(f"Reversal model validation failed: {e}")
        return {"error": str(e)}


async def _validate_exit_timing_models(
    timing_results: Dict[str, Any],
    logger: Any
) -> Dict[str, Any]:
    """
    Validate exit timing models.
    
    Args:
        timing_results: Exit timing model results
        logger: Logger instance
        
    Returns:
        Dict containing validation results
    """
    try:
        if timing_results.get("success", False):
            accuracy = timing_results.get("accuracy", 0.0)
            return {
                "valid": accuracy >= 0.70,
                "accuracy": accuracy,
                "status": "PASS" if accuracy >= 0.70 else "FAIL"
            }
        else:
            return {
                "valid": False,
                "accuracy": 0.0,
                "status": "FAIL",
                "error": timing_results.get("error", "Unknown error")
            }
        
    except Exception as e:
        logger.error(f"Exit timing model validation failed: {e}")
        return {"error": str(e)}


async def _calculate_overall_metrics(
    validation_results: Dict[str, Any],
    logger: Any
) -> Dict[str, Any]:
    """
    Calculate overall validation metrics.
    
    Args:
        validation_results: Validation results
        logger: Logger instance
        
    Returns:
        Dict containing overall metrics
    """
    try:
        metrics = {
            "overall_accuracy": 0.0,
            "reversal_accuracy": 0.0,
            "exit_timing_accuracy": 0.0,
            "multi_timeframe_accuracy": 0.0,
            "total_models": 0,
            "passed_models": 0
        }
        
        # Calculate entry model metrics
        if "entry_models" in validation_results:
            entry_accuracies = []
            for timeframe, validation in validation_results["entry_models"].items():
                if isinstance(validation, dict) and "accuracy" in validation:
                    entry_accuracies.append(validation["accuracy"])
                    metrics["total_models"] += 1
                    if validation.get("valid", False):
                        metrics["passed_models"] += 1
            
            if entry_accuracies:
                metrics["multi_timeframe_accuracy"] = np.mean(entry_accuracies)
        
        # Calculate reversal model metrics
        if "reversal_models" in validation_results:
            reversal_validation = validation_results["reversal_models"]
            if isinstance(reversal_validation, dict) and "accuracy" in reversal_validation:
                metrics["reversal_accuracy"] = reversal_validation["accuracy"]
                metrics["total_models"] += 1
                if reversal_validation.get("valid", False):
                    metrics["passed_models"] += 1
        
        # Calculate exit timing model metrics
        if "exit_timing_models" in validation_results:
            timing_validation = validation_results["exit_timing_models"]
            if isinstance(timing_validation, dict) and "accuracy" in timing_validation:
                metrics["exit_timing_accuracy"] = timing_validation["accuracy"]
                metrics["total_models"] += 1
                if timing_validation.get("valid", False):
                    metrics["passed_models"] += 1
        
        # Calculate overall accuracy
        accuracies = [
            metrics["multi_timeframe_accuracy"],
            metrics["reversal_accuracy"],
            metrics["exit_timing_accuracy"]
        ]
        accuracies = [acc for acc in accuracies if acc > 0]
        
        if accuracies:
            metrics["overall_accuracy"] = np.mean(accuracies)
        
        # Calculate pass rate
        if metrics["total_models"] > 0:
            metrics["pass_rate"] = metrics["passed_models"] / metrics["total_models"]
        else:
            metrics["pass_rate"] = 0.0
        
        return metrics
        
    except Exception as e:
        logger.error(f"Overall metrics calculation failed: {e}")
        return {"error": str(e)}


async def _log_training_results(output: Dict[str, Any], logger: Any) -> None:
    """
    Log training results to MLflow.
    
    Args:
        output: Training output
        logger: Logger instance
    """
    try:
        logger.info("📊 Logging training results to MLflow...")
        
        # Log metrics
        await log_step_metrics("step17_5_exit_strategy_training", {
            "overall_accuracy": output["performance_metrics"]["overall_accuracy"],
            "reversal_detection_accuracy": output["performance_metrics"]["reversal_detection_accuracy"],
            "exit_timing_accuracy": output["performance_metrics"]["exit_timing_accuracy"],
            "multi_timeframe_accuracy": output["performance_metrics"]["multi_timeframe_accuracy"]
        })
        
        # Log model artifacts
        await log_step_artifact_with_standardized_name(
            "step17_5_exit_strategy_training",
            "training_results.json",
            json.dumps(output["training_results"], indent=2)
        )
        
        # Log validation results
        await log_step_artifact_with_standardized_name(
            "step17_5_exit_strategy_training",
            "validation_results.json",
            json.dumps(output["validation_results"], indent=2)
        )
        
        logger.info("✅ Training results logged to MLflow")
        
    except Exception as e:
        logger.error(f"MLflow logging failed: {e}")


# Export the main function
__all__ = ["step17_5_exit_strategy_training"]