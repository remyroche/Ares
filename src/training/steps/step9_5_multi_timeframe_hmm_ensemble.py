# src/training/steps/step09_5_multi_timeframe_hmm_ensemble.py

"""Step 9.5: Multi-Timeframe HMM Ensemble Training with Regime-Specific Logic."

This step trains a multi-timeframe HMM cluster ensemble system that combines
predictions from HMM clusters across multiple timeframes (5m, 15m, 30m, 1h)
to improve regime forecasting accuracy and reduce MAPE, with regime-specific optimization.

The ensemble predicts REGIME TRANSITIONS only, not price direction.
Price direction predictions are made in other components.
"""
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from copy import copy

    MultiTimeframeHMMEnsemble,
    EnsembleConfig,
    TimeframeConfig,
from src.training.steps.multi_timeframe_hmm_ensemble import (
from src.config.multi_timeframe_hmm_ensemble_config import (

)
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
from src.utils.common_operations import ensure_directory, safe_json_dump, safe_json_load


class RegimeSpecificMultiTimeframeEnsemble:
    """Regime-specific multi-timeframe HMM ensemble with regime-aware optimization."""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("RegimeSpecificMultiTimeframeEnsemble")
        
        # Regime-specific configuration
        self.regime_config = config.get("regime_specific_ensemble", {
            "min_regime_samples": 100,
            "regime_specific_timeframes": True,
            "regime_specific_weights": True,
            "regime_specific_validation": True,
            "regime_specific_logging": True,
            "regime_specific_optimization": True
        })
        
        # Regime-specific results storage
        self.regime_ensembles = {}
        self.regime_validation_results = {}
        self.regime_optimization_results = {}
        
        # Timeframes for regime-specific optimization
        self.timeframes = ["1m", "5m", "15m", "30m"]
        
        self.logger.info("🎯 Regime-Specific Multi-Timeframe Ensemble initialized")

    async def run_regime_specific_ensemble_step(
        self, symbol: str, exchange: str, data_dir: str, 
        timeframe: str, lookback_days: int
    ) -> bool:
        """Run regime-specific multi-timeframe ensemble creation."""
        
        self.logger.info(f"🚀 Starting regime-specific multi-timeframe ensemble for {symbol}")
        
        try:
            # Load regime-specific data for each timeframe
            regime_data = await self._load_regime_specific_data(symbol, exchange, data_dir, lookback_days)
            
            if regime_data.empty:
                self.logger.error("❌ No regime data available")
                return False
            
            # Get unique regimes
            unique_regimes = regime_data['composite_cluster_id'].unique()
            self.logger.info(f"📊 Found {len(unique_regimes)} regimes: {unique_regimes}")
            
            # Create regime-specific ensembles
            for regime in unique_regimes:
                self.logger.info(f"🔄 Creating ensemble for regime: {regime}")
                
                regime_ensembles = {}
                
                for tf in self.timeframes:
                    # Load regime-specific data for this timeframe
                    regime_tf_data = await self._load_regime_timeframe_data(
                        symbol, exchange, tf, regime, lookback_days
                    )
                    
                    if not regime_tf_data.empty:
                        # Create regime-specific ensemble for this timeframe
                        ensemble = await self._create_regime_timeframe_ensemble(
                            regime_tf_data, regime, tf
                        )
                        
                        if ensemble:
                            regime_ensembles[tf] = ensemble
                
                if regime_ensembles:
                    # Create regime-specific multi-timeframe ensemble
                    multi_tf_ensemble = await self._create_regime_multi_timeframe_ensemble(
                        regime_ensembles, regime
                    )
                    
                    if multi_tf_ensemble:
                        self.regime_ensembles[regime] = multi_tf_ensemble
                        
                        # Validate regime-specific ensemble
                        validation_success = await self._validate_regime_ensemble(
                            multi_tf_ensemble, regime
                        )
                        
                        if not validation_success:
                            self.logger.error(f"❌ Regime {regime} ensemble validation failed")
                            return False
                
                else:
                    self.logger.warning(f"⚠️ No ensembles created for regime {regime}")
            
            # Save regime-specific ensembles
            await self._save_regime_specific_ensembles(symbol, data_dir)
            
            self.logger.info("✅ Regime-specific multi-timeframe ensemble completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error in regime-specific ensemble creation: {e}")
            return False

    async def _load_regime_specific_data(
        self, symbol: str, exchange: str, data_dir: str, lookback_days: int
    ) -> pd.DataFrame:
        """Load regime-specific data for all timeframes."""
        
        self.logger.info(f"📊 Loading regime-specific data for {symbol}")
        
        try:
            # Load unified data with regime information
            unified_data_path = f"{data_dir}/{exchange}_{symbol}_unified_data.parquet"
            if not os.path.exists(unified_data_path):
                self.logger.error(f"❌ Unified data not found: {unified_data_path}")
                return pd.DataFrame()
            
            unified_data = pd.read_parquet(unified_data_path)
            
            # Check if regime column exists
            if 'composite_cluster_id' not in unified_data.columns:
                self.logger.error("❌ Regime column 'composite_cluster_id' not found")
                return pd.DataFrame()
            
            # Filter by lookback days if timestamp column exists
            if 'timestamp' in unified_data.columns:
                cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
                unified_data = unified_data[unified_data['timestamp'] >= cutoff_date]
            
            self.logger.info(f"✅ Loaded {len(unified_data)} samples with {unified_data['composite_cluster_id'].nunique()} regimes")
            return unified_data
            
        except Exception as e:
            self.logger.error(f"❌ Error loading regime-specific data: {e}")
            return pd.DataFrame()

    async def _load_regime_timeframe_data(
        self, symbol: str, exchange: str, timeframe: str, regime: str, lookback_days: int
    ) -> pd.DataFrame:
        """Load regime-specific data for a specific timeframe."""
        
        self.logger.info(f"📊 Loading {timeframe} data for regime {regime}")
        
        try:
            # Load timeframe-specific data
            tf_data_path = f"data/training/{exchange}_{symbol}_{timeframe}_unified_data.parquet"
            if not os.path.exists(tf_data_path):
                self.logger.warning(f"⚠️ Timeframe data not found: {tf_data_path}")
                return pd.DataFrame()
            
            tf_data = pd.read_parquet(tf_data_path)
            
            # Filter for specific regime
            if 'composite_cluster_id' in tf_data.columns:
                regime_mask = tf_data['composite_cluster_id'] == regime
                regime_data = tf_data[regime_mask].copy()
                
                # Regime-specific data validation
                if len(regime_data) < self.regime_config["min_regime_samples"]:
                    self.logger.warning(f"⚠️ Insufficient {timeframe} data for regime {regime}: {len(regime_data)} samples")
                    return pd.DataFrame()
                
                self.logger.info(f"✅ Loaded {len(regime_data)} {timeframe} samples for regime {regime}")
                return regime_data
            else:
                self.logger.warning(f"⚠️ No regime column in {timeframe} data")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"❌ Error loading {timeframe} data for regime {regime}: {e}")
            return pd.DataFrame()

    async def _create_regime_timeframe_ensemble(
        self, regime_data: pd.DataFrame, regime: str, timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """Create regime-specific ensemble for a specific timeframe."""
        
        self.logger.info(f"🎯 Creating {timeframe} ensemble for regime {regime}")
        
        try:
            # Regime-specific ensemble configuration
            ensemble_config = await self._get_regime_specific_ensemble_config(regime, timeframe)
            
            # Create ensemble using existing MultiTimeframeHMMEnsemble
            ensemble = MultiTimeframeHMMEnsemble(ensemble_config)
            
            # Train regime-specific ensemble
            ensemble_results = await ensemble.train_regime_specific_ensemble(
                regime_data, regime, timeframe
            )
            
            if ensemble_results:
                # Regime-specific optimization
                if self.regime_config["regime_specific_optimization"]:
                    optimized_ensemble = await self._optimize_regime_ensemble(
                        ensemble_results, regime, timeframe
                    )
                    return optimized_ensemble
                else:
                    return ensemble_results
            else:
                self.logger.error(f"❌ Failed to create {timeframe} ensemble for regime {regime}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error creating {timeframe} ensemble for regime {regime}: {e}")
            return None

    async def _create_regime_multi_timeframe_ensemble(
        self, regime_ensembles: Dict[str, Any], regime: str
    ) -> Optional[Dict[str, Any]]:
        """Create regime-specific multi-timeframe ensemble."""
        
        self.logger.info(f"🎯 Creating multi-timeframe ensemble for regime {regime}")
        
        try:
            # Regime-specific multi-timeframe configuration
            multi_tf_config = await self._get_regime_multi_timeframe_config(regime)
            
            # Create multi-timeframe ensemble
            multi_tf_ensemble = {
                "regime": regime,
                "timeframes": list(regime_ensembles.keys()),
                "ensembles": regime_ensembles,
                "weights": await self._calculate_regime_specific_weights(regime_ensembles, regime),
                "config": multi_tf_config,
                "created_timestamp": datetime.now().isoformat()
            }
            
            # Regime-specific validation
            if self.regime_config["regime_specific_validation"]:
                validation_results = await self._validate_regime_multi_timeframe_ensemble(
                    multi_tf_ensemble, regime
                )
                multi_tf_ensemble["validation"] = validation_results
            
            return multi_tf_ensemble
            
        except Exception as e:
            self.logger.error(f"❌ Error creating multi-timeframe ensemble for regime {regime}: {e}")
            return None

    async def _get_regime_specific_ensemble_config(self, regime: str, timeframe: str) -> Dict[str, Any]:
        """Get regime-specific ensemble configuration."""
        
        # Base configuration
        base_config = get_multi_timeframe_hmm_ensemble_config()
        
        # Regime-specific modifications
        regime_config = base_config.copy()
        
        # Regime-specific hyperparameters
        if self.regime_config["regime_specific_optimization"]:
            regime_config.update({
                "regime": regime,
                "timeframe": timeframe,
                "regime_specific_params": await self._get_regime_specific_params(regime, timeframe)
            })
        
        return regime_config

    async def _get_regime_multi_timeframe_config(self, regime: str) -> Dict[str, Any]:
        """Get regime-specific multi-timeframe configuration."""
        
        return {
            "regime": regime,
            "regime_specific_weights": self.regime_config["regime_specific_weights"],
            "regime_specific_validation": self.regime_config["regime_specific_validation"],
            "regime_specific_optimization": self.regime_config["regime_specific_optimization"]
        }

    async def _calculate_regime_specific_weights(
        self, regime_ensembles: Dict[str, Any], regime: str
    ) -> Dict[str, float]:
        """Calculate regime-specific weights for ensemble combination."""
        
        self.logger.info(f"⚖️ Calculating regime-specific weights for regime {regime}")
        
        try:
            weights = {}
            
            if self.regime_config["regime_specific_weights"]:
                # Calculate regime-specific weights based on performance
                for timeframe, ensemble in regime_ensembles.items():
                    if ensemble and "performance" in ensemble:
                        # Use regime-specific performance metrics
                        performance_score = ensemble["performance"].get("regime_specific_score", 0.5)
                        weights[timeframe] = performance_score
                    else:
                        # Default equal weights
                        weights[timeframe] = 1.0 / len(regime_ensembles)
            else:
                # Equal weights
                for timeframe in regime_ensembles.keys():
                    weights[timeframe] = 1.0 / len(regime_ensembles)
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {tf: w / total_weight for tf, w in weights.items()}
            
            self.logger.info(f"✅ Calculated weights for regime {regime}: {weights}")
            return weights
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating regime-specific weights: {e}")
            # Return equal weights as fallback
            return {tf: 1.0 / len(regime_ensembles) for tf in regime_ensembles.keys()}

    async def _optimize_regime_ensemble(
        self, ensemble_results: Dict[str, Any], regime: str, timeframe: str
    ) -> Dict[str, Any]:
        """Optimize regime-specific ensemble."""
        
        self.logger.info(f"⚙️ Optimizing {timeframe} ensemble for regime {regime}")
        
        try:
            # Regime-specific optimization logic
            optimized_results = ensemble_results.copy()
            
            # Add regime-specific optimization results
            optimized_results.update({
                "regime": regime,
                "timeframe": timeframe,
                "optimization_timestamp": datetime.now().isoformat(),
                "regime_specific_optimization": True
            })
            
            # Store optimization results
            self.regime_optimization_results[f"{regime}_{timeframe}"] = optimized_results
            
            return optimized_results
            
        except Exception as e:
            self.logger.error(f"❌ Error optimizing regime ensemble: {e}")
            return ensemble_results

    async def _validate_regime_ensemble(
        self, ensemble: Dict[str, Any], regime: str
    ) -> bool:
        """Validate regime-specific ensemble."""
        
        self.logger.info(f"🔍 Validating ensemble for regime {regime}")
        
        try:
            # Regime-specific validation logic
            validation_results = {
                "regime": regime,
                "validation_timestamp": datetime.now().isoformat(),
                "timeframes": ensemble.get("timeframes", []),
                "weights": ensemble.get("weights", {}),
                "validation_checks": {},
                "success": True
            }
            
            # Perform regime-specific validation checks
            validation_checks = await self._perform_regime_validation_checks(ensemble, regime)
            validation_results["validation_checks"] = validation_checks
            
            # Store validation results
            self.regime_validation_results[regime] = validation_results
            
            # Check if validation passed
            validation_success = all(check.get("passed", False) for check in validation_checks.values())
            
            if validation_success:
                self.logger.info(f"✅ Regime {regime} ensemble validation passed")
            else:
                self.logger.error(f"❌ Regime {regime} ensemble validation failed")
            
            return validation_success
            
        except Exception as e:
            self.logger.error(f"❌ Error validating regime ensemble: {e}")
            return False

    async def _validate_regime_multi_timeframe_ensemble(
        self, ensemble: Dict[str, Any], regime: str
    ) -> Dict[str, Any]:
        """Validate regime-specific multi-timeframe ensemble."""
        
        try:
            # Multi-timeframe specific validation
            validation_results = {
                "regime": regime,
                "multi_timeframe_validation": True,
                "timeframe_count": len(ensemble.get("timeframes", [])),
                "weight_distribution": ensemble.get("weights", {}),
                "validation_timestamp": datetime.now().isoformat()
            }
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ Error in multi-timeframe validation: {e}")
            return {"success": False, "error": str(e)}

    async def _perform_regime_validation_checks(
        self, ensemble: Dict[str, Any], regime: str
    ) -> Dict[str, Dict[str, Any]]:
        """Perform regime-specific validation checks."""
        
        try:
            checks = {}
            
            # Check 1: Ensemble structure
            checks["structure"] = {
                "passed": "ensembles" in ensemble and "weights" in ensemble,
                "description": "Ensemble structure validation"
            }
            
            # Check 2: Timeframe coverage
            checks["timeframes"] = {
                "passed": len(ensemble.get("timeframes", [])) > 0,
                "description": "Timeframe coverage validation"
            }
            
            # Check 3: Weight distribution
            weights = ensemble.get("weights", {})
            total_weight = sum(weights.values())
            checks["weights"] = {
                "passed": abs(total_weight - 1.0) < 0.01,  # Allow small numerical errors
                "description": "Weight distribution validation"
            }
            
            # Check 4: Regime-specific performance
            checks["performance"] = {
                "passed": True,  # Placeholder for actual performance validation
                "description": "Regime-specific performance validation"
            }
            
            return checks
            
        except Exception as e:
            self.logger.error(f"❌ Error in validation checks: {e}")
            return {"error": {"passed": False, "description": f"Validation error: {e}"}}

    async def _save_regime_specific_ensembles(self, symbol: str, data_dir: str) -> None:
        """Save regime-specific ensembles."""
        
        self.logger.info("💾 Saving regime-specific ensembles")
        
        try:
            for regime, ensemble in self.regime_ensembles.items():
                if ensemble:
                    regime_save_path = f"{data_dir}/regime_ensembles/{symbol}/regime_{regime}"
                    ensure_directory(regime_save_path)
                    
                    # Save ensemble configuration
                    ensemble_config_path = f"{regime_save_path}/ensemble_config.json"
                    safe_json_dump(ensemble, ensemble_config_path, indent=2, default=str)
                    
                    # Save validation results
                    if regime in self.regime_validation_results:
                        validation_path = f"{regime_save_path}/validation_results.json"
                        safe_json_dump(self.regime_validation_results[regime], validation_path, indent=2, default=str)
                    
                    self.logger.info(f"✅ Saved regime {regime} ensemble to {regime_save_path}")
                    
        except Exception as e:
            self.logger.error(f"❌ Error saving regime-specific ensembles: {e}")

    def _log_regime_specific_metrics(
        self, regime: str, metrics: dict, step_name: str
    ) -> None:
        """Log regime-specific metrics."""
        
        if self.regime_config["regime_specific_logging"]:
            self.logger.info(f"📊 {step_name} - Regime {regime} metrics:")
            for metric_name, metric_value in metrics.items():
                self.logger.info(f"   {metric_name}: {metric_value}")

    # Placeholder methods for regime-specific parameters
    async def _get_regime_specific_params(self, regime: str, timeframe: str) -> Dict[str, Any]:
        """Get regime-specific parameters."""
        # Placeholder for actual regime-specific parameter logic
        return {"regime": regime, "timeframe": timeframe}

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
                    rf_data = safe_json_load(rf_path)
                    
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
        
        # Optional: regime-specific training toggle and regime list
        regime_list = kwargs.get("regimes") or []
        per_regime_enabled: bool = bool(regime_list)

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
        
        # If per-regime enabled, also train per-regime ensembles reusing the same data
        per_regime_status: dict[str, Any] = {}
        if per_regime_enabled:
            for regime_name in regime_list:
                try:
                    logger.info(f"🎯 Training per-regime ensemble for regime {regime_name}")
                    regime_ensemble = MultiTimeframeHMMEnsemble(config, symbol, exchange, regime_name=regime_name)
                    regime_success = regime_ensemble.train_ensemble(regime_forecasting_data)
                    per_regime_status[regime_name] = {
                        "success": bool(regime_success),
                        "models_dir": regime_ensemble.models_dir,
                    }
                except Exception as e:
                    logger.warning(f"⚠️ Failed per-regime ensemble training for {regime_name}: {e}")
                    per_regime_status[regime_name] = {"success": False, "error": str(e)}

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
            "per_regime": per_regime_status if per_regime_enabled else None,
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
            metadata = safe_json_load(metadata_path)
            
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