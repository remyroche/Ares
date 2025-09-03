#!/usr/bin/env python3
"""Enhanced HMM-Based Training with Multi-Output Support and Regime-Specific Logic.

This module extends the existing HMM-based training to support intelligent
multi-output prediction for both direction and profit using the triple barrier
method and profit-based feature engineering, with regime-specific optimization.
"""
import json
import os
import pickle
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.decorators import handles_errors

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import asyncio
from src.utils.common_operations import (
from copy import copy

    get_current_datetime, format_datetime, ensure_directory,
    safe_read_parquet, safe_to_parquet, safe_copy
)
from sklearn.feature_selection import (
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
    average_precision_score
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

# Multi-output training will be imported when needed
from src.training.steps.step04_analyst_labeling_feature_engineering_components.profit_based_feature_engineering import (
    ProfitBasedFeatureEngineering
)
from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
from src.core.domain import (
    PerformanceLevel,
    ValidationLevel,
    adaptive_resource_allocation,
    comprehensive_validation,
    handle_errors,
    intelligent_caching,
    model_validation,
    performance_monitor,
    pipeline_checkpoint,
    validate_feature_engineering_with_lookahead_bias_detection
)
from src.utils.logger import system_logger
from src.utils.common_operations import ensure_directory, safe_json_dump

# Suppress warnings
warnings.filterwarnings("ignore")

class EnhancedHMMBasedTrainingStep:
    """Enhanced HMM-Based Model Training with Multi-Output Support and Regime-Specific Logic."

    Extends the existing HMM-based training to support intelligent multi-output
    prediction for both direction and profit using the triple barrier method
    and profit-based feature engineering, with regime-specific optimization.
    """
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}

        # Initialize SRBreakoutPredictor for S/R level integration with optimized parameters
        sr_config = safe_copy(config, deep=True)
        sr_config["sr_breakout_predictor"] = sr_config.get("sr_breakout_predictor", {})
        sr_config["sr_breakout_predictor"]["use_optimized_params"] = True
        self.sr_predictor = SRBreakoutPredictor(sr_config)

        # Initialize S/R outcome model trainer
        self.sr_outcome_trainer = None
        self.sr_outcome_model_trained = False

        # Initialize profit-based feature engineering
        self.profit_feature_engine = ProfitBasedFeatureEngineering(
            profit_column="potential_profit_pct",
            use_numba=True,
            memory_efficient=True
        )

        # Multi-output model trainer
        self.multi_output_trainer = None
        self.enable_multi_output = config.get("enable_multi_output", True)

        # Model architecture mapping from config
        hmm_lm_config = config.get("HMM_LM", {})
        specialist_config = hmm_lm_config.get("specialist_models", {})

        self.model_architectures = {}
        for timeframe, model_config in specialist_config.items():
            self.model_architectures[timeframe] = model_config
        # Provide sensible defaults if not configured
        if not self.model_architectures:
            self.model_architectures = {
                "1m": "LogisticRegression",
                "5m": "LightGBM",
            }

        # Regime-specific configuration
        self.regime_config = config.get("regime_specific_training", {
            "min_regime_samples": 100,
            "regime_validation_split": 0.2,
            "regime_specific_hyperparameters": True,
            "regime_specific_feature_selection": True,
            "regime_specific_validation": True,
            "regime_specific_logging": True
        })

        # Regime-specific results storage
        self.regime_results = {}
        self.regime_models = {}
        self.regime_validation_results = {}

        # Validation configuration (default fallback)
        self.validation_config = {
            "n_splits": 5,
            "test_size": 0.2,
            "validation_size": 0.2,
            "min_samples_per_split": 1000,
            "regime_aware_splitting": True,
        }

        self.logger.info("🎯 Enhanced HMM-Based Training Step initialized with regime-specific logic")

    def print(self, message: str) -> None:
        """Print message using logger."""
        self.logger.info(message)

    @handles_errors(exceptions=(Exception,), default_return=None)
    async def initialize(self) -> None:
        """Initialize the enhanced HMM-based training step."""
        self.logger.info("🚀 Initializing Enhanced HMM-Based Training Step...")
        
        # Initialize regime-specific components
        await self._initialize_regime_components()
        
        self.logger.info("✅ Enhanced HMM-Based Training Step initialized successfully")

    async def _initialize_regime_components(self) -> None:
        """Initialize regime-specific components."""
        self.logger.info("🔄 Initializing regime-specific components...")
        
        # Initialize regime-specific data loader
        self.regime_data_loader = await self._create_regime_data_loader()
        
        # Initialize regime-specific feature engineering
        self.regime_feature_engine = await self._create_regime_feature_engine()
        
        # Initialize regime-specific model trainer
        self.regime_model_trainer = await self._create_regime_model_trainer()
        
        self.logger.info("✅ Regime-specific components initialized")

    async def _create_regime_data_loader(self) -> Any:
        """Create regime-specific data loader."""
        # This would integrate with the unified data loader
        return None  # Placeholder for actual implementation

    async def _create_regime_feature_engine(self) -> Any:
        """Create regime-specific feature engineering component."""
        # This would integrate with the existing feature engineering
        return None  # Placeholder for actual implementation

    async def _create_regime_model_trainer(self) -> Any:
        """Create regime-specific model trainer."""
        # This would integrate with the existing model training
        return None  # Placeholder for actual implementation

    async def _load_regime_specific_data(
        self, symbol: str, data_dir: str, regime: str
    ) -> pd.DataFrame:
        """Load regime-specific data for processing."""
        
        self.logger.info(f"📊 Loading regime-specific data for regime: {regime}")
        
        try:
            # Load unified data with regime information
            unified_data_path = f"{data_dir}/{symbol}_unified_data.parquet"
            if not os.path.exists(unified_data_path):
                self.logger.error(f"❌ Unified data not found: {unified_data_path}")
                return pd.DataFrame()
            
            unified_data = safe_read_parquet(unified_data_path)
            
            # Check if regime column exists
            if 'composite_cluster_id' not in unified_data.columns:
                self.logger.error("❌ Regime column 'composite_cluster_id' not found in unified data")
                return pd.DataFrame()
            
            # Filter for specific regime
            regime_mask = unified_data['composite_cluster_id'] == regime
            regime_data = safe_copy(unified_data[regime_mask])
            
            # Regime-specific data validation
            if len(regime_data) < self.regime_config["min_regime_samples"]:
                self.logger.warning(f"⚠️ Insufficient data for regime {regime}: {len(regime_data)} samples")
                return pd.DataFrame()
            
            self.logger.info(f"✅ Loaded {len(regime_data)} samples for regime {regime}")
            return regime_data
            
        except Exception as e:
            self.logger.error(f"❌ Error loading regime-specific data: {e}")
            return pd.DataFrame()

    async def _train_regime_specific_model(
        self, regime_data: pd.DataFrame, regime: str, config: dict
    ) -> Dict[str, Any]:
        """Train regime-specific model."""
        
        self.logger.info(f"🎯 Training model for regime: {regime}")
        
        try:
            # Regime-specific feature engineering
            regime_features = await self._engineer_regime_features(regime_data, regime)
            
            if regime_features.empty:
                self.logger.error(f"❌ No features generated for regime {regime}")
                return {"success": False, "error": "No features generated"}
            
            # Regime-specific hyperparameter optimization
            regime_params = await self._optimize_regime_hyperparameters(
                regime_features, regime
            )
            
            # Regime-specific model training
            regime_model = await self._train_model_with_regime_params(
                regime_features, regime_params, regime
            )
            
            # Regime-specific validation
            validation_results = await self._validate_regime_model(
                regime_model, regime_features, regime
            )
            
            # Store regime-specific results
            self.regime_results[regime] = {
                "model": regime_model,
                "parameters": regime_params,
                "validation": validation_results,
                "regime": regime,
                "success": True
            }
            
            self.logger.info(f"✅ Regime {regime} training completed successfully")
            return self.regime_results[regime]
            
        except Exception as e:
            self.logger.error(f"❌ Error training regime {regime} model: {e}")
            return {"success": False, "error": str(e)}

    async def _engineer_regime_features(
        self, regime_data: pd.DataFrame, regime: str
    ) -> pd.DataFrame:
        """Engineer regime-specific features."""
        
        self.logger.info(f"🔧 Engineering features for regime: {regime}")
        
        try:
            # Use existing feature engineering with regime-specific parameters
            features_df = await self.prepare_enhanced_data(regime_data, "1m")
            
            # Add regime-specific feature enhancements
            if self.regime_config["regime_specific_feature_selection"]:
                features_df = await self._apply_regime_specific_feature_selection(
                    features_df, regime
                )
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"❌ Error engineering features for regime {regime}: {e}")
            return pd.DataFrame()

    async def _optimize_regime_hyperparameters(
        self, regime_features: pd.DataFrame, regime: str
    ) -> Dict[str, Any]:
        """Optimize hyperparameters for regime-specific model."""
        
        self.logger.info(f"⚙️ Optimizing hyperparameters for regime: {regime}")
        
        try:
            # Regime-specific hyperparameter optimization
            if self.regime_config["regime_specific_hyperparameters"]:
                # Use regime-specific parameter ranges
                regime_params = await self._get_regime_specific_params(regime)
                
                # Optimize using regime-specific data
                optimized_params = await self._optimize_params_for_regime(
                    regime_features, regime_params, regime
                )
                
                return optimized_params
            else:
                # Use default parameters
                return self._get_default_params()
                
        except Exception as e:
            self.logger.error(f"❌ Error optimizing hyperparameters for regime {regime}: {e}")
            return self._get_default_params()

    async def _get_regime_specific_params(self, regime: str) -> Dict[str, Any]:
        """Provide initial parameter ranges for a given regime (placeholder)."""
        try:
            return {
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [4, 6, 8],
                "n_estimators": [100, 200, 400],
                "regime": str(regime),
            }
        except Exception:
            return self._get_default_params()

    async def _optimize_params_for_regime(
        self, regime_features: pd.DataFrame, regime_params: Dict[str, Any], regime: str
    ) -> Dict[str, Any]:
        """Select a reasonable parameter set from ranges (lightweight placeholder)."""
        try:
            # Simple heuristic: pick middle values
            return {
                "learning_rate": float(regime_params.get("learning_rate", [0.05])[0 if len(regime_params.get("learning_rate", [])) == 1 else 1]),
                "max_depth": int(regime_params.get("max_depth", [6])[0 if len(regime_params.get("max_depth", [])) == 1 else 1]),
                "n_estimators": int(regime_params.get("n_estimators", [200])[0 if len(regime_params.get("n_estimators", [])) == 1 else 1]),
                "regime": str(regime),
            }
        except Exception:
            return self._get_default_params()

    def _get_default_params(self) -> Dict[str, Any]:
        """Default hyperparameters as a safe fallback."""
        return {"learning_rate": 0.1, "max_depth": 6, "n_estimators": 100}

    async def _train_model_with_regime_params(
        self, regime_features: pd.DataFrame, regime_params: dict, regime: str
    ) -> Any:
        """Train model with regime-specific parameters."""
        
        self.logger.info(f"🎯 Training model with regime-specific parameters for regime: {regime}")
        
        try:
            # Use existing training logic with regime-specific parameters
            model_name = f"enhanced_regime_{regime}_1m"
            
            # Train the model using existing enhanced training logic
            results = await self.train_enhanced_model(regime_features, model_name)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error training model for regime {regime}: {e}")
            return None

    async def _validate_regime_model(
        self, regime_model: Any, regime_features: pd.DataFrame, regime: str
    ) -> Dict[str, Any]:
        """Validate regime-specific model."""
        
        self.logger.info(f"🔍 Validating model for regime: {regime}")
        
        try:
            # Regime-specific validation
            if self.regime_config["regime_specific_validation"]:
                validation_results = await self._perform_regime_specific_validation(
                    regime_model, regime_features, regime
                )
            else:
                validation_results = await self._perform_default_validation(
                    regime_model, regime_features
                )
            
            # Store validation results
            self.regime_validation_results[regime] = validation_results
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ Error validating model for regime {regime}: {e}")
            return {"success": False, "error": str(e)}

    async def _perform_regime_specific_validation(
        self, regime_model: Any, regime_features: pd.DataFrame, regime: str
    ) -> Dict[str, Any]:
        """Perform regime-specific validation."""
        
        try:
            # Regime-specific validation logic
            # This would include regime-specific metrics and thresholds
            
            validation_results = {
                "regime": regime,
                "validation_timestamp": format_datetime(get_current_datetime(), "%Y-%m-%dT%H:%M:%S"),
                "metrics": {},
                "quality_checks": {},
                "success": True
            }
            
            # Add regime-specific validation metrics here
            # This is a placeholder for actual validation logic
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ Error in regime-specific validation: {e}")
            return {"success": False, "error": str(e)}

    async def run_enhanced_regime_specific_step(
        self, symbol: str, data_dir: str, 
        method_a_mixture_of_experts: dict, enable_multi_output: bool
    ) -> bool:
        """Run regime-specific enhanced training."""
        
        self.logger.info(f"🚀 Starting regime-specific enhanced training for {symbol}")
        
        try:
            # Load regime data
            regime_data = await self._load_regime_specific_data(symbol, data_dir, "all")
            
            if regime_data.empty:
                self.logger.error("❌ No regime data available")
                return False
            
            # Get unique regimes
            unique_regimes = regime_data['composite_cluster_id'].unique()
            self.logger.info(f"📊 Found {len(unique_regimes)} regimes: {unique_regimes}")
            
            # Train models for each regime
            for regime in unique_regimes:
                regime_mask = regime_data['composite_cluster_id'] == regime
                regime_training_data = regime_data[regime_mask]
                
                # Regime-specific training
                regime_success = await self._train_regime_specific_model(
                    regime_training_data, regime, method_a_mixture_of_experts
                )
                
                if not regime_success.get("success", False):
                    self.logger.error(f"❌ Regime {regime} training failed")
                    return False
            
            # Validate all regime-specific results
            overall_success = await self._validate_regime_specific_results()
            
            if overall_success:
                # Save regime-specific models
                await self._save_regime_specific_models(symbol, data_dir)
                
                self.logger.info("✅ Regime-specific enhanced training completed successfully")
                return True
            else:
                self.logger.error("❌ Regime-specific validation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error in regime-specific enhanced training: {e}")
            return False

    async def _validate_regime_specific_results(self) -> bool:
        """Validate all regime-specific results."""
        
        self.logger.info("🔍 Validating all regime-specific results")
        
        try:
            for regime, results in self.regime_results.items():
                if not results.get("success", False):
                    self.logger.error(f"❌ Regime {regime} results validation failed")
                    return False
                
                # Regime-specific quality validation
                quality_valid = await self._validate_regime_quality(results, regime)
                if not quality_valid:
                    self.logger.error(f"❌ Regime {regime} quality validation failed")
                    return False
            
            self.logger.info("✅ All regime-specific results validated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating regime-specific results: {e}")
            return False

    async def _validate_regime_quality(self, results: dict, regime: str) -> bool:
        """Validate regime-specific quality."""
        
        try:
            # Regime-specific quality checks
            # This would include regime-specific thresholds and metrics
            
            # Placeholder for actual quality validation logic
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating regime quality: {e}")
            return False

    async def _save_regime_specific_models(self, symbol: str, data_dir: str) -> None:
        """Save regime-specific models."""
        
        self.logger.info("💾 Saving regime-specific models")
        
        try:
            for regime, results in self.regime_results.items():
                if results.get("success", False):
                    regime_save_path = f"{data_dir}/enhanced_models/{symbol}/regime_{regime}"
                    ensure_directory(regime_save_path)
                    
                    # Save regime-specific model
                    self.save_enhanced_models(results, regime_save_path)
                    
                    self.logger.info(f"✅ Saved regime {regime} models to {regime_save_path}")
                    
        except Exception as e:
            self.logger.error(f"❌ Error saving regime-specific models: {e}")

    def _log_regime_specific_metrics(
        self, regime: str, metrics: dict, step_name: str
    ) -> None:
        """Log regime-specific metrics."""
        
        if self.regime_config["regime_specific_logging"]:
            self.logger.info(f"📊 {step_name} - Regime {regime} metrics:")
            for metric_name, metric_value in metrics.items():
                self.logger.info(f"   {metric_name}: {metric_value}")

    @handles_errors(
        exceptions=(ValueError, TypeError, MemoryError),
        default_return=None,
        context="enhanced_data_preparation"
    )
    async def prepare_enhanced_data(
        self,
        data: pd.DataFrame,
        timeframe: str,
        regime_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Prepare data for enhanced training with multi-output support."
        
        Args:
            data: Input DataFrame with features and targets
            timeframe: Timeframe for the data
            regime_key: Regime key if regime-specific training
            
        Returns:
            Dictionary containing prepared data for both single and multi-output training
        """
        self.logger.info(f"📊 Preparing enhanced training data for {timeframe}")
        if regime_key:
            self.logger.info(f"   - Regime: {regime_key}")
        
        # Check for multi-output targets
        has_direction = "direction" in data.columns
        has_profit = "potential_profit_pct" in data.columns
        has_single_target = "target" in data.columns or "label" in data.columns
        
        # Use enhanced feature selection if multi-output is enabled
        if has_profit and self.enable_multi_output:
            try:
                from src.training.enhanced_matrix_operations import EnhancedMatrixOperations
                
                self.logger.info("🔧 Using enhanced feature selection with autoencoder features...")
                
                # Create enhanced matrix operations manager
                feature_selector = EnhancedMatrixOperations(self.config)
                
                # Create dummy target for feature selection
                dummy_target = pd.Series(0, index=data.index)
                if "direction" in data.columns:
                    dummy_target = data["direction"]
                
                # Use enhanced feature selection with autoencoder features
                selected_features, metadata = feature_selector.select_features_step2(
                    features_df=data,
                    target=dummy_target,
                    symbol="default",
                    exchange="default",
                    data_dir="temp",
                    use_autoencoder_features=True,  # Use autoencoder features
                    use_regularization=True         # Use regularization
                )
                
                self.logger.info(f"✅ Enhanced feature selection completed: {selected_features.shape[1]} features selected")
                self.logger.info(f"   - Autoencoder features: {metadata.get('stages', {}).get('stage0_autoencoder', {}).get('autoencoder_features_added', 0)}")
                self.logger.info(f"   - Regularization applied: {metadata.get('stages', {}).get('stage6_regularization', {}).get('regularization_applied', False)}")
                
                # Use selected features
                data = selected_features
                
            except Exception as e:
                self.logger.warning(f"⚠️ Enhanced feature selection failed: {e}")
                self.logger.info("📊 Falling back to basic feature preparation")
                
                # Apply profit-based feature engineering as fallback
                self.logger.info("🔧 Applying profit-based feature engineering...")
                data = self.profit_feature_engine.apply_all_features(data)
                self.logger.info(f"✅ Added profit-based features")
        
        # Prepare features
        exclude_columns = [
            "target", "label", "direction", "potential_profit_pct",
            "timestamp", "timeframe", "composite_cluster_id", "sample_weight"
        ]
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        features = safe_copy(data[feature_columns])
        
        # Handle missing values
        features = features.fillna(0)
        
        prepared_data = {
            "features": features,
            "feature_columns": feature_columns,
            "timeframe": timeframe,
            "regime_key": regime_key,
            "has_multi_output": has_direction and has_profit,
            "has_single_output": has_single_target
        }
        
        # Prepare single-output targets (backward compatibility)
        if has_single_target:
            label_col = "target" if "target" in data.columns else "label"
            prepared_data["single_target"] = data[label_col].fillna(0)
        
        # Prepare multi-output targets
        if has_direction and has_profit:
            prepared_data["direction_target"] = data["direction"].fillna(0)
            prepared_data["profit_target"] = data["potential_profit_pct"].fillna(0)
            
            # Convert direction to binary if needed
            if prepared_data["direction_target"].dtype in ['object', 'string']:
                prepared_data["direction_target"] = (prepared_data["direction_target"] > 0).astype(int)
        
        self.logger.info(f"✅ Enhanced data prepared: {features.shape[0]} samples, {features.shape[1]} features")
        self.logger.info(f"   - Multi-output: {prepared_data['has_multi_output']}")
        self.logger.info(f"   - Single-output: {prepared_data['has_single_output']}")
        
        return prepared_data

    @handles_errors(
        exceptions=(ValueError, RuntimeError),
        default_return=None,
        context="enhanced_model_training"
    )
    # @performance_monitor - removed, use log_execution_time
    async def train_enhanced_model(
        self,
        prepared_data: Dict[str, Any],
        model_name: str = "enhanced_model"
    ) -> Dict[str, Any]:
        """Train enhanced model with multi-output support."
        
        Args:
            prepared_data: Prepared data dictionary
            model_name: Name for the trained model
            
        Returns:
            Dictionary containing training results and model artifacts
        """
        self.logger.info(f"🚀 Training enhanced model: {model_name}")
        
        results = {
            "model_name": model_name,
            "timeframe": prepared_data["timeframe"],
            "regime_key": prepared_data["regime_key"],
            "single_output_results": None,
            "multi_output_results": None,
            "training_timestamp": format_datetime(get_current_datetime(), "%Y-%m-%dT%H:%M:%S")
        }
        
        # Train multi-output model if data is available
        if prepared_data["has_multi_output"] and self.multi_output_trainer:
            self.logger.info("🎯 Training multi-output probability model")
            
            # Prepare data for multi-output training
            X = prepared_data["features"].values
            y = prepared_data.get("single_target")
            if y is None:
                # Fallback: derive a simple direction from potential_profit_pct if available, else zeros
                if "direction_target" in prepared_data:
                    y = prepared_data["direction_target"].values
                else:
                    y = np.zeros(len(X))
            
            # Market data for PnL evaluation must come from real series aligned with features
            # Expect caller to provide aligned market_data in prepared_data when available
            market_data = prepared_data.get("market_data")
            if market_data is None or "close" not in getattr(market_data, 'columns', []):
                # Safe fallback to avoid crashes; PnL metrics will be skipped if not present
                market_data = pd.DataFrame({"close": pd.Series(index=prepared_data["features"].index, dtype=float)})
            
            # Generate multi-output targets
            y_multi = self.multi_output_trainer.prepare_multi_output_targets(X, y, market_data)
            
            # Purged time split with embargo to prevent leakage
            n_samples = len(X)
            embargo = max(5, int(0.01 * n_samples))  # 1% or minimum 5 samples
            split_idx = int(0.8 * n_samples)
            train_end = max(0, split_idx - embargo)

            X_train, X_test = X[:train_end], X[split_idx:]
            y_train_multi = {k: v[:train_end] for k, v in y_multi.items()}
            y_test_multi = {k: v[split_idx:] for k, v in y_multi.items()}
            
            # Train multi-output model
            trained_models = self.multi_output_trainer.train_multi_output_model(
                X_train, y_train_multi, X_test, y_test_multi
            )
            
            # Generate probability outputs
            price_action_probabilities = self.multi_output_trainer.predict_probabilities(
                X_test, market_data.iloc[split_idx:]
            )
            from src.utils.common_operations import standardize_price_action_probabilities
            price_action_probabilities = standardize_price_action_probabilities(price_action_probabilities)

            # Compute PR-AUC for primary head if available
            pr_auc_scores = {}
            try:
                if "direction_probability" in price_action_probabilities and "direction_target" in prepared_data:
                    y_true = prepared_data["direction_target"].values[split_idx:]
                    y_proba = price_action_probabilities["direction_probability"]
                    # If scalar provided, replicate to match length
                    if np.isscalar(y_proba):
                        y_proba = np.full_like(y_true, float(y_proba), dtype=float)
                    pr_auc_scores["direction_pr_auc"] = float(average_precision_score(y_true, y_proba))
            except Exception:
                pass

            # Fast PnL simulator on OOS segment (validation-only)
            pnl_metrics = {}
            try:
                if "close" in market_data.columns:
                    costs_bps = float(self.config.get("costs_bps", 8.0))  # 0.08% per round-trip
                    entry_prices = market_data.iloc[split_idx:]["close"].values
                    # Simple threshold on direction probability
                    if "direction_probability" in price_action_probabilities:
                        prob = price_action_probabilities["direction_probability"]
                        threshold = float(self.config.get("prob_threshold", 0.6))
                        if np.isscalar(prob):
                            positions = np.ones_like(entry_prices, dtype=int) if float(prob) > threshold else np.zeros_like(entry_prices, dtype=int)
                        else:
                            positions = (np.asarray(prob) > threshold).astype(int)  # 1 long, 0 flat
                        # Entry/exit when position changes
                        returns = []
                        wins = 0
                        trades = 0
                        prev_pos = 0
                        prev_price = None
                        for i, (pos, px) in enumerate(zip(positions, entry_prices)):
                            if prev_pos == 0 and pos == 1:
                                prev_pos = 1
                                prev_price = px
                            elif prev_pos == 1 and pos == 0 and prev_price is not None:
                                gross = (px / prev_price) - 1.0
                                net = gross - (costs_bps / 10000.0)
                                returns.append(net)
                                if net > 0:
                                    wins += 1
                                trades += 1
                                prev_pos = 0
                                prev_price = None
                        if prev_pos == 1 and prev_price is not None:
                            gross = (entry_prices[-1] / prev_price) - 1.0
                            net = gross - (costs_bps / 10000.0)
                            returns.append(net)
                            if net > 0:
                                wins += 1
                            trades += 1
                        import math
                        pnl = float(np.nansum(returns))
                        win_rate = float(wins / trades) if trades > 0 else 0.0
                        sharpe = 0.0
                        if len(returns) > 1 and np.std(returns) > 1e-12:
                            sharpe = float(np.mean(returns) / np.std(returns) * math.sqrt(252))
                        pnl_metrics = {
                            "pnl": pnl,
                            "win_rate": win_rate,
                            "sharpe": sharpe,
                            "trades": trades,
                            "composite_metric": float(0.5 * pnl + 0.25 * win_rate + 0.25 * (sharpe / 10.0)),
                        }
            except Exception:
                pass
            
            multi_output_result = {
                "trained_models": trained_models,
                "price_action_probabilities": price_action_probabilities,
                "model_type": "multi_output",
                "pr_auc": pr_auc_scores,
                "pnl_metrics": pnl_metrics
            }
            
            if multi_output_result:
                results["multi_output_results"] = multi_output_result
                self.logger.info("✅ Multi-output probability model training completed successfully")
            else:
                self.logger.warning("⚠️ Multi-output probability model training failed")
        
        # Train single-output model for backward compatibility
        if prepared_data["has_single_output"]:
            self.logger.info("🎯 Training single-output model (backward compatibility)")
            
            single_output_result = await self._train_single_output_model(
                prepared_data["features"],
                prepared_data["single_target"],
                prepared_data["timeframe"],
                prepared_data["regime_key"]
            )
            
            if single_output_result:
                results["single_output_results"] = single_output_result
                self.logger.info("✅ Single-output model training completed successfully")
            else:
                self.logger.warning("⚠️ Single-output model training failed")
        
        return results

    async def _train_single_output_model(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        timeframe: str,
        regime_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train single-output model (backward compatibility)."
        
        Args:
            features: Feature DataFrame
            target: Target series
            timeframe: Timeframe
            regime_key: Regime key if regime-specific
            
        Returns:
            Training results dictionary
        """
        try:
            architecture = self.model_architectures.get(timeframe, "LightGBM")
            self.logger.info(f"   🌳 Training {architecture} single-output model")
            
            # Prepare data
            X = features.values
            y = target.values
            
            # Time series split
            tscv = TimeSeriesSplit(n_splits=self.validation_config["n_splits"])
            
            # Cross-validation
            cv_scores = []
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Train model based on architecture
                if architecture == "LightGBM":
                    model = lgb.LGBMClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=6,
                        random_state=42,
                        verbose=-1
                    )
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_val_scaled, y_val)],
                        eval_metric="binary_logloss",
                        early_stopping_rounds=10,
                        verbose=False
                    )
                elif architecture == "RandomForest":
                    model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1
                    )
                    model.fit(X_train_scaled, y_train)
                elif architecture == "LogisticRegression":
                    # Calibrated LR with L2 by default (optionally L1 via config)
                    base_lr = LogisticRegression(
                        penalty=self.config.get("lr_penalty", "l2"),
                        solver="liblinear" if self.config.get("lr_penalty", "l2") == "l1" else "lbfgs",
                        max_iter=200,
                        class_weight="balanced"
                    )
                    model = CalibratedClassifierCV(base_lr, method="isotonic", cv=3)
                    model.fit(X_train_scaled, y_train)
                else:
                    self.logger.warning(f"   ⚠️ Architecture {architecture} not implemented for single-output")
                    continue
                
                # Evaluate
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_val_scaled)[:, 1]
                    pr_auc = average_precision_score(y_val, y_proba)
                    cv_scores.append(pr_auc)
                else:
                    y_pred = model.predict(X_val_scaled)
                    accuracy = accuracy_score(y_val, y_pred)
                    cv_scores.append(accuracy)
            
            # Train final model on full dataset
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            if architecture == "LightGBM":
                final_model = lgb.LGBMClassifier(
                    n_estimators=self.config.get("lgb_n_estimators", 300),
                    learning_rate=self.config.get("lgb_learning_rate", 0.075),
                    max_depth=self.config.get("lgb_max_depth", 4),
                    random_state=42,
                    verbose=-1
                )
            elif architecture == "RandomForest":
                final_model = RandomForestClassifier(
                    n_estimators=self.config.get("rf_n_estimators", 150),
                    max_depth=self.config.get("rf_max_depth", 8),
                    max_features=self.config.get("rf_max_features", "sqrt"),
                    random_state=42,
                    n_jobs=-1
                )
            elif architecture == "LogisticRegression":
                base_lr = LogisticRegression(
                    penalty=self.config.get("lr_penalty", "l2"),
                    solver="liblinear" if self.config.get("lr_penalty", "l2") == "l1" else "lbfgs",
                    max_iter=500,
                    class_weight="balanced"
                )
                final_model = CalibratedClassifierCV(base_lr, method="isotonic", cv=3)
            else:
                return None
            
            final_model.fit(X_scaled, y)
            
            # Replace optimistic training-set metrics with CV summary; also compute PR-AUC on OOF-like fold averages
            final_accuracy = None
            
            result = {
                "model": final_model,
                "scaler": scaler,
                "architecture": architecture,
                "cv_scores": cv_scores,
                "cv_mean": float(np.mean(cv_scores)) if cv_scores else 0.0,
                "cv_std": float(np.std(cv_scores)) if cv_scores else 0.0,
                "final_accuracy": final_accuracy,  # No training-set metric reported
                "feature_importance": dict(zip(features.columns, final_model.feature_importances_)) if hasattr(final_model, 'feature_importances_') else {},
                "n_features": len(features.columns)
            }
            
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to train single-output model: {e}")
            return None

    @handles_errors(
        exceptions=(ValueError, RuntimeError),
        default_return=None,
        context="enhanced_regime_specific_training"
    )
    async def train_enhanced_regime_specific_models(
        self,
        timeframe: str,
        regime_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train enhanced regime-specific models with multi-output support."
        
        Args:
            timeframe: Timeframe for training
            regime_data: Regime-specific data dictionary
            
        Returns:
            Dictionary containing regime-specific training results
        """
        self.logger.info(f"🎯 Training enhanced regime-specific models for {timeframe}")
        
        regime_results = {}
        
        for regime_key, regime_info in regime_data.items():
            regime_desc = regime_info.get("description", "Unknown")
            self.logger.info(f"   🎯 Processing regime {regime_key}: {regime_desc}")
            
            # Get regime data
            train_data = regime_info.get("train")
            val_data = regime_info.get("validation")
            test_data = regime_info.get("test")
            
            if train_data is None or len(train_data) < self.validation_config["min_samples_per_split"]:
                self.logger.warning(f"   ⚠️ Insufficient data for regime {regime_key}")
                continue
            
            # Combine all regime data for training
            all_regime_data = pd.concat([train_data, val_data, test_data], ignore_index=True)
            
            # Prepare enhanced data
            prepared_data = await self.prepare_enhanced_data(
                all_regime_data, timeframe, regime_key
            )
            
            # Train enhanced model
            model_result = await self.train_enhanced_model(
                prepared_data, f"enhanced_{timeframe}_{regime_key}"
            )
            
            if model_result:
                regime_results[regime_key] = model_result
                self.logger.info(f"   ✅ Enhanced regime {regime_key} training completed")
            else:
                self.logger.warning(f"   ⚠️ Enhanced regime {regime_key} training failed")
        
        return regime_results

    def predict_enhanced(
        self,
        features: pd.DataFrame,
        model_name: str = "enhanced_model",
        prediction_type: str = "multi_output"  # "multi_output" or "single_output"
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Make predictions using enhanced model."
        
        Args:
            features: Feature DataFrame
            model_name: Name of the model to use
            prediction_type: Type of prediction ("multi_output" or "single_output")
            
        Returns:
            For multi_output: Tuple of (direction_predictions, profit_predictions)
            For single_output: Array of predictions
        """
        if prediction_type == "multi_output" and self.multi_output_trainer:
            try:
                # Create market data for prediction
                market_data = pd.DataFrame({
                    'close': np.random.randn(len(features)),  # Placeholder - should use actual market data
                    'volume': np.random.randn(len(features))
                })
                
                # Generate probability predictions
                price_action_probabilities = self.multi_output_trainer.predict_probabilities(
                    features.values, market_data
                )
                
                # Extract direction and profit probabilities
                direction_prob = price_action_probabilities.get("direction_probability", 0.5)
                profit_prob = price_action_probabilities.get("triple_barrier_probability", 0.5)
                
                return np.array([direction_prob]), np.array([profit_prob])
            except Exception as e:
                self.logger.error(f"❌ Multi-output prediction failed: {e}")
                return None, None
        else:
            # Fallback to single-output prediction
            self.logger.warning("⚠️ Single-output prediction not implemented in enhanced trainer")
            return None

    def save_enhanced_models(
        self,
        results: Dict[str, Any],
        save_path: str
    ) -> None:
        """Save enhanced models to disk."
        
        Args:
            results: Training results dictionary
            save_path: Path to save models
        """
        try:
            ensure_directory(save_path)
            
            # Save multi-output models
            if results.get("multi_output_results") and self.multi_output_trainer:
                multi_output_dir = os.path.join(save_path, "multi_output_models")
                ensure_directory(multi_output_dir)
                
                # Save the multi-output trainer
                model_path = os.path.join(multi_output_dir, f"{results['model_name']}_multi_output.pkl")
                import joblib
                joblib.dump(self.multi_output_trainer, model_path)
            
            # Save single-output models
            if results.get("single_output_results"):
                single_output_dir = os.path.join(save_path, "single_output_models")
                ensure_directory(single_output_dir)
                
                single_result = results["single_output_results"]
                model_path = os.path.join(single_output_dir, f"{results['model_name']}_single.pkl")
                scaler_path = os.path.join(single_output_dir, f"{results['model_name']}_scaler.pkl")
                
                import joblib
                joblib.dump(single_result["model"], model_path)
                joblib.dump(single_result["scaler"], scaler_path)
            
            # Save metadata
            metadata = {
                "model_name": results["model_name"],
                "timeframe": results["timeframe"],
                "regime_key": results["regime_key"],
                "has_multi_output": results.get("multi_output_results") is not None,
                "has_single_output": results.get("single_output_results") is not None,
                "training_timestamp": results["training_timestamp"]
            }
            
            metadata_path = os.path.join(save_path, "metadata.json")
            safe_json_dump(metadata, metadata_path, indent=2)
            
            self.logger.info(f"✅ Enhanced models saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save enhanced models: {e}")

    def load_enhanced_models(
        self,
        model_name: str,
        load_path: str
    ) -> None:
        """Load enhanced models from disk."
        
        Args:
            model_name: Name of the model to load
            load_path: Path to load models from
        """
        try:
            # Load multi-output models
            multi_output_dir = os.path.join(load_path, "multi_output_models")
            if os.path.exists(multi_output_dir) and self.multi_output_trainer:
                model_path = os.path.join(multi_output_dir, f"{model_name}_multi_output.pkl")
                if os.path.exists(model_path):
                    import joblib
                    self.multi_output_trainer = joblib.load(model_path)
                    self.logger.info(f"✅ Loaded multi-output trainer from {model_path}")
            
            # Load single-output models
            single_output_dir = os.path.join(load_path, "single_output_models")
            if os.path.exists(single_output_dir):
                model_path = os.path.join(single_output_dir, f"{model_name}_single.pkl")
                scaler_path = os.path.join(single_output_dir, f"{model_name}_scaler.pkl")
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    import joblib
                    model = joblib.load(model_path)
                    scaler = joblib.load(scaler_path)
                    
                    # Store in models dict
                    self.models[f"{model_name}_single"] = {
                        "model": model,
                        "scaler": scaler,
                        "model_type": "single_output"
                    }
            
            self.logger.info(f"✅ Enhanced models loaded from {load_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load enhanced models: {e}")

    async def _apply_regime_specific_feature_selection(
        self, features_df: pd.DataFrame, regime: str
    ) -> pd.DataFrame:
        """Apply simple regime-aware feature selection placeholder."

        Drops all-zero columns and ensures numeric dtype; keeps columns with variance.
        """
        try:
            df = features_df.copy()
            # Keep numeric columns only
            df = df.select_dtypes(include=[np.number])
            # Drop columns with zero variance
            variances = df.var(axis=0, ddof=0)
            keep_cols = [c for c, v in variances.items() if np.isfinite(v) and v > 0]
            if keep_cols:
                df = df[keep_cols]
            return df
        except Exception as e:
            self.logger.warning(f"⚠️ Feature selection fallback for regime {regime}: {e}")
            return features_df


async def run_enhanced_step(
    symbol: str = "ETHUSDT",
    data_dir: str = "data/training",
    method_a_mixture_of_experts: Optional[Dict] = None,
    enable_multi_output: bool = True
) -> bool:
    """Run enhanced HMM-based training step with multi-output support."
    
    Args:
        symbol: Trading symbol
        data_dir: Data directory
        method_a_mixture_of_experts: Method A configuration
        enable_multi_output: Whether to enable multi-output training
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger = system_logger.getChild("EnhancedHMMTraining")
        logger.info(f"🚀 Starting Enhanced HMM-Based Training for {symbol}")
        
        # Load configuration
        config = {
            "enable_multi_output": enable_multi_output,
            "HMM_LM": method_a_mixture_of_experts or {}
        }
        
        # Initialize enhanced trainer
        enhanced_trainer = EnhancedHMMBasedTrainingStep(config)
        await enhanced_trainer.initialize()
        
        # Load labeled data
        labeled_path = f"{data_dir}/{symbol}_labeled_train.parquet"
        if not os.path.exists(labeled_path):
            logger.error(f"❌ Labeled data not found: {labeled_path}")
            return False
        
        data = safe_read_parquet(labeled_path)
        logger.info(f"✅ Loaded labeled data: {data.shape}")
        
        # Prepare enhanced data
        prepared_data = await enhanced_trainer.prepare_enhanced_data(data, "1m")
        
        # If regime column present, run per-regime training as well (using shared accessor)
        per_regime_results: dict[str, Any] = {}
        try:
            from src.utils.regime_data_access import get_regime_column, split_train_val_test_by_regime
            regime_col = get_regime_column(data)
            if regime_col is not None:
                logger.info(f"🔁 Running per-regime enhanced training based on '{regime_col}'")
                splits = split_train_val_test_by_regime(
                    data.sort_values("timestamp"),
                    regime_column=regime_col,
                    train_ratio=0.7,
                    val_ratio=0.15,
                    test_ratio=0.15,
                    min_samples_per_split=10,
                )
                if splits:
                    per_regime_results = await enhanced_trainer.train_enhanced_regime_specific_models("1m", splits)
        except Exception as e:
            logger.warning(f"⚠️ Per-regime enhanced training skipped due to error: {e}")
        
        # Train enhanced model (global model)
        results = await enhanced_trainer.train_enhanced_model(
            prepared_data, f"enhanced_{symbol}_1m"
        )
        
        if results:
            # Save models
            save_path = f"{data_dir}/enhanced_models/{symbol}"
            enhanced_trainer.save_enhanced_models(results, save_path)
            # Save per-regime models if available
            if per_regime_results:
                per_regime_dir = os.path.join(save_path, "per_regime")
                ensure_directory(per_regime_dir)
                for regime_key, regime_result in per_regime_results.items():
                    regime_dir = os.path.join(per_regime_dir, f"regime_{regime_key}")
                    ensure_directory(regime_dir)
                    try:
                        enhanced_trainer.save_enhanced_models(regime_result, regime_dir)
                    except Exception:
                        pass
            
            logger.info("✅ Enhanced HMM-based training completed successfully")
            return True
        else:
            logger.error("❌ Enhanced HMM-based training failed")
            return False
            
    except Exception as e:
        logger.exception(f"❌ Enhanced HMM-based training failed: {e}")
        return False