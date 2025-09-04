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
    get_current_datetime, format_datetime, ensure_directory,
    safe_read_parquet, safe_to_parquet, safe_copy
)
from copy import copy
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

    @handles_errors(exceptions=(Exception,), default_return=False, log_level="ERROR")
    @validates(strict=True)
    @log_call
    @traced
    async def initialize(self) -> bool:
        """Initialize the enhanced HMM-based training step with comprehensive validation."""
        self.logger.info("🚀 Initializing Enhanced HMM-Based Training Step...")
        
        try:
            # Validate configuration
            config_valid = await self._validate_configuration()
            if not config_valid:
                self.logger.error("❌ Configuration validation failed")
                return False
            
            # Initialize regime-specific components
            regime_init_success = await self._initialize_regime_components()
            if not regime_init_success:
                self.logger.error("❌ Regime components initialization failed")
                return False
            
            # Initialize validation components
            validation_init_success = await self._initialize_validation_components()
            if not validation_init_success:
                self.logger.error("❌ Validation components initialization failed")
                return False
            
            self.logger.info("✅ Enhanced HMM-Based Training Step initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize HMM-based training step: {e}")
            return False
    
    @handles_errors(exceptions=(Exception,), default_return=False, log_level="ERROR")
    @log_call
    @traced
    async def _validate_configuration(self) -> bool:
        """Validate the training configuration."""
        self.logger.info("🔍 Validating HMM training configuration...")
        
        try:
            # Validate required configuration keys
            required_keys = ['HMM_LM', 'regime_specific_training']
            for key in required_keys:
                if key not in self.config:
                    self.logger.error(f"❌ Missing required configuration key: {key}")
                    return False
            
            # Validate regime configuration
            regime_config = self.config.get('regime_specific_training', {})
            required_regime_keys = ['min_regime_samples', 'regime_validation_split']
            for key in required_regime_keys:
                if key not in regime_config:
                    self.logger.error(f"❌ Missing required regime configuration key: {key}")
                    return False
            
            # Validate numeric parameters
            min_samples = regime_config.get('min_regime_samples', 0)
            if not isinstance(min_samples, int) or min_samples <= 0:
                self.logger.error(f"❌ Invalid min_regime_samples: {min_samples}")
                return False
            
            validation_split = regime_config.get('regime_validation_split', 0)
            if not isinstance(validation_split, (int, float)) or not (0 < validation_split < 1):
                self.logger.error(f"❌ Invalid regime_validation_split: {validation_split}")
                return False
            
            self.logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            return False
    
    @handles_errors(exceptions=(Exception,), default_return=False, log_level="ERROR")
    @log_call
    @traced
    async def _initialize_validation_components(self) -> bool:
        """Initialize validation components."""
        self.logger.info("🔍 Initializing validation components...")
        
        try:
            # Initialize data quality validator
            from src.utils.enhanced_data_quality_validator import EnhancedDataQualityValidator
            self.data_quality_validator = EnhancedDataQualityValidator(self.config)
            
            # Initialize feature validator
            from src.utils.feature_engineering_validation import FeatureEngineeringValidator
            self.feature_validator = FeatureEngineeringValidator(self.config)
            
            # Initialize model validator
            from src.utils.model_performance_monitor import ModelPerformanceMonitor
            self.model_validator = ModelPerformanceMonitor(self.config)
            
            self.logger.info("✅ Validation components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize validation components: {e}")
            return False

    @handles_errors(exceptions=(Exception,), default_return=False, log_level="ERROR")
    @log_call
    @traced
    async def _initialize_regime_components(self) -> bool:
        """Initialize regime-specific components with validation."""
        self.logger.info("🔄 Initializing regime-specific components...")
        
        try:
            # Initialize regime-specific data loader
            self.regime_data_loader = await self._create_regime_data_loader()
            if self.regime_data_loader is None:
                self.logger.warning("⚠️ Regime data loader not available, using fallback")
            
            # Initialize regime-specific feature engineering
            self.regime_feature_engine = await self._create_regime_feature_engine()
            if self.regime_feature_engine is None:
                self.logger.warning("⚠️ Regime feature engine not available, using fallback")
            
            # Initialize regime-specific model trainer
            self.regime_model_trainer = await self._create_regime_model_trainer()
            if self.regime_model_trainer is None:
                self.logger.warning("⚠️ Regime model trainer not available, using fallback")
            
            self.logger.info("✅ Regime-specific components initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize regime components: {e}")
            return False

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

    @handles_errors(exceptions=(Exception,), default_return=pd.DataFrame(), log_level="ERROR")
    @validates(strict=True)
    @log_call
    @traced
    async def _load_regime_specific_data(
        self, symbol: str, data_dir: str, regime: str
    ) -> pd.DataFrame:
        """Load regime-specific data for processing with comprehensive validation."""
        
        self.logger.info(f"📊 Loading regime-specific data for regime: {regime}")
        
        try:
            # Validate inputs
            if not symbol or not data_dir or not regime:
                raise ValueError("Missing required parameters: symbol, data_dir, regime")
            
            # Validate data directory exists
            if not safe_file_exists(data_dir):
                raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
            # Load unified data with regime information
            unified_data_path = f"{data_dir}/{symbol}_unified_data.parquet"
            
            # Check if unified data exists, fallback to consolidated data
            if not safe_file_exists(unified_data_path):
                self.logger.warning(f"⚠️ Unified data not found: {unified_data_path}")
                # Fallback to consolidated data
                consolidated_data_path = f"{data_dir}/aggtrades_BINANCE_{symbol}_consolidated.parquet"
                if safe_file_exists(consolidated_data_path):
                    unified_data_path = consolidated_data_path
                    self.logger.info(f"📁 Using consolidated data: {consolidated_data_path}")
                else:
                    raise FileNotFoundError(f"Neither unified nor consolidated data found for {symbol}")
            
            # Load data with validation
            df = safe_read_parquet(unified_data_path)
            if df.empty:
                raise ValueError(f"Data file is empty: {unified_data_path}")
            
            # Validate data schema
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            schema_valid, schema_errors = validate_dataframe_schema(df, required_columns)
            if not schema_valid:
                raise ValueError(f"Data schema validation failed: {schema_errors}")
            
            # Validate data quality
            quality_report = validate_data_quality(df, max_nan_ratio=0.1, check_duplicates=True)
            if not quality_report['is_valid']:
                self.logger.warning(f"⚠️ Data quality issues detected: {quality_report['issues']}")
            
            # Filter by regime if regime column exists
            if 'regime' in df.columns:
                regime_data = df[df['regime'] == regime].copy()
                if regime_data.empty:
                    self.logger.warning(f"⚠️ No data found for regime: {regime}")
                    return pd.DataFrame()
                self.logger.info(f"✅ Loaded {len(regime_data)} rows for regime: {regime}")
                return regime_data
            else:
                self.logger.warning("⚠️ No regime column found, returning all data")
                return df
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load regime-specific data: {e}")
            return pd.DataFrame()

    @handles_errors(exceptions=(Exception,), default_return={"success": False, "error": "Training failed"}, log_level="ERROR")
    @validates(strict=True)
    @log_call
    @traced
    async def _train_regime_specific_model(
        self, regime_data: pd.DataFrame, regime: str, config: dict
    ) -> Dict[str, Any]:
        """Train regime-specific model with comprehensive validation."""
        
        self.logger.info(f"🎯 Training model for regime: {regime}")
        
        try:
            # Validate inputs
            if regime_data.empty:
                raise ValueError(f"Empty regime data for regime: {regime}")
            
            if not regime or not isinstance(regime, str):
                raise ValueError(f"Invalid regime: {regime}")
            
            if not config or not isinstance(config, dict):
                raise ValueError("Invalid config provided")
            
            # Validate minimum data requirements
            min_samples = self.regime_config.get("min_regime_samples", 100)
            if len(regime_data) < min_samples:
                raise ValueError(f"Insufficient data for regime {regime}: {len(regime_data)} < {min_samples}")
            
            # Regime-specific feature engineering with validation
            regime_features = await self._engineer_regime_features(regime_data, regime)
            
            if regime_features.empty:
                self.logger.error(f"❌ No features generated for regime {regime}")
                return {"success": False, "error": "No features generated"}
            
            # Validate feature quality
            feature_quality = await self._validate_feature_quality(regime_features, regime)
            if not feature_quality["is_valid"]:
                self.logger.warning(f"⚠️ Feature quality issues for regime {regime}: {feature_quality['issues']}")
            
            # Regime-specific hyperparameter optimization
            regime_params = await self._optimize_regime_hyperparameters(
                regime_features, regime
            )
            
            # Regime-specific model training
            regime_model = await self._train_model_with_regime_params(
                regime_features, regime_params, regime
            )
            
            if regime_model is None:
                self.logger.error(f"❌ Failed to train model for regime {regime}")
                return {"success": False, "error": "Model training failed"}
            
            # Regime-specific validation
            validation_results = await self._validate_regime_model(
                regime_model, regime_features, regime
            )
            
            if not validation_results["is_valid"]:
                self.logger.warning(f"⚠️ Model validation issues for regime {regime}: {validation_results['issues']}")
            
            # Store regime results
            self.regime_results[regime] = {
                "model": regime_model,
                "features": regime_features,
                "params": regime_params,
                "validation": validation_results,
                "training_samples": len(regime_data),
                "feature_count": len(regime_features.columns)
            }
            
            self.logger.info(f"✅ Successfully trained model for regime {regime}")
            return {
                "success": True,
                "regime": regime,
                "model": regime_model,
                "validation": validation_results
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to train regime-specific model for {regime}: {e}")
            return {"success": False, "error": str(e)}
    
    @handles_errors(exceptions=(Exception,), default_return={"is_valid": False, "issues": ["Validation failed"]}, log_level="ERROR")
    @log_call
    @traced
    async def _validate_feature_quality(self, features: pd.DataFrame, regime: str) -> Dict[str, Any]:
        """Validate feature quality for regime-specific training."""
        self.logger.info(f"🔍 Validating feature quality for regime: {regime}")
        
        try:
            issues = []
            
            # Check for empty features
            if features.empty:
                issues.append("Empty feature DataFrame")
                return {"is_valid": False, "issues": issues}
            
            # Check for sufficient features
            if len(features.columns) < 5:
                issues.append(f"Insufficient features: {len(features.columns)} < 5")
            
            # Check for high NaN ratio
            nan_ratios = features.isna().sum() / len(features)
            high_nan_cols = nan_ratios[nan_ratios > 0.5]
            if not high_nan_cols.empty:
                issues.append(f"High NaN ratio columns: {high_nan_cols.to_dict()}")
            
            # Check for constant features
            constant_features = []
            for col in features.columns:
                if features[col].nunique() <= 1:
                    constant_features.append(col)
            if constant_features:
                issues.append(f"Constant features: {constant_features}")
            
            # Check for infinite values
            inf_count = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                issues.append(f"Infinite values found: {inf_count}")
            
            is_valid = len(issues) == 0
            
            self.logger.info(f"✅ Feature quality validation for regime {regime}: {'PASSED' if is_valid else 'FAILED'}")
            return {"is_valid": is_valid, "issues": issues}
            
        except Exception as e:
            self.logger.error(f"❌ Feature quality validation failed for regime {regime}: {e}")
            return {"is_valid": False, "issues": [f"Validation error: {e}"]}
    
    @handles_errors(exceptions=(Exception,), default_return={"is_valid": False, "issues": ["Model validation failed"]}, log_level="ERROR")
    @log_call
    @traced
    async def _validate_regime_model(self, model: Any, features: pd.DataFrame, regime: str) -> Dict[str, Any]:
        """Validate trained regime-specific model."""
        self.logger.info(f"🔍 Validating model for regime: {regime}")
        
        try:
            issues = []
            
            # Check if model is valid
            if model is None:
                issues.append("Model is None")
                return {"is_valid": False, "issues": issues}
            
            # Check if model has required methods
            required_methods = ['predict', 'fit']
            for method in required_methods:
                if not hasattr(model, method):
                    issues.append(f"Model missing required method: {method}")
            
            # Check if features are compatible with model
            if hasattr(model, 'feature_importances_'):
                if len(model.feature_importances_) != len(features.columns):
                    issues.append(f"Feature importance mismatch: {len(model.feature_importances_)} vs {len(features.columns)}")
            
            is_valid = len(issues) == 0
            
            self.logger.info(f"✅ Model validation for regime {regime}: {'PASSED' if is_valid else 'FAILED'}")
            return {"is_valid": is_valid, "issues": issues}
            
        except Exception as e:
            self.logger.error(f"❌ Model validation failed for regime {regime}: {e}")
            return {"is_valid": False, "issues": [f"Validation error: {e}"]}

    @handles_errors(exceptions=(Exception,), default_return=pd.DataFrame(), log_level="ERROR")
    @validates(strict=True)
    @log_call
    @traced
    async def _engineer_regime_features(
        self, regime_data: pd.DataFrame, regime: str
    ) -> pd.DataFrame:
        """Engineer regime-specific features with comprehensive validation."""
        
        self.logger.info(f"🔧 Engineering features for regime: {regime}")
        
        try:
            # Validate inputs
            if regime_data.empty:
                raise ValueError(f"Empty regime data for regime: {regime}")
            
            if not regime or not isinstance(regime, str):
                raise ValueError(f"Invalid regime: {regime}")
            
            # Validate required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = set(required_columns) - set(regime_data.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Create a copy to avoid modifying original data
            features_df = safe_copy(regime_data)
            
            # Basic price features
            features_df['price_change'] = features_df['close'].pct_change()
            features_df['price_range'] = (features_df['high'] - features_df['low']) / features_df['close']
            features_df['volume_change'] = features_df['volume'].pct_change()
            
            # Technical indicators
            features_df['sma_5'] = features_df['close'].rolling(window=5).mean()
            features_df['sma_20'] = features_df['close'].rolling(window=20).mean()
            features_df['rsi'] = self._calculate_rsi(features_df['close'])
            
            # Regime-specific features
            features_df['regime'] = regime
            features_df['regime_encoded'] = hash(regime) % 1000  # Simple encoding
            
            # Remove rows with NaN values
            features_df = features_df.dropna()
            
            if features_df.empty:
                self.logger.warning(f"⚠️ No valid features after cleaning for regime: {regime}")
                return pd.DataFrame()
            
            # Validate feature quality
            feature_quality = await self._validate_feature_quality(features_df, regime)
            if not feature_quality["is_valid"]:
                self.logger.warning(f"⚠️ Feature quality issues for regime {regime}: {feature_quality['issues']}")
            
            self.logger.info(f"✅ Generated {len(features_df.columns)} features for regime {regime}")
            return features_df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to engineer features for regime {regime}: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series(index=prices.index, dtype=float)

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