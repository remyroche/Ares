"""
Multi-Timeframe Training System

This module provides multi-timeframe model training capabilities that can be used
across all model types (general, analyst, tactician). It handles cross-timeframe
feature engineering, model coordination, and training orchestration.

Timeframes supported: 1m, 5m, 15m, 30m, 1h
(Removed 1d and 4h as requested)
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import gc
import psutil
from pathlib import Path

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size, check_disk_space
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, MathValidationError
)
from src.utils.parquet_utils import get_parquet_utils, ParquetUtils
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time, 
    timeout, error_boundary, compose, validate_data_quality, 
    monitor_step_execution, ensure_data_integrity, validate_pipeline_step
)
from src.utils.intensity_scaler import (
    get_intensity_from_environment, get_scaled_hpo_trials,
    get_scaled_hpo_timeout, log_intensity_info
)
from src.core.errors import (
    ValidationError, DataIntegrityError, FileOperationError,
    ConfigurationError, ModelTrainingError
)
from src.utils.logger import system_logger

# Import the existing multi-timeframe components
from src.analyst.multi_timeframe_feature_engineering import MultiTimeframeFeatureEngineering
from src.analyst.predictive_ensembles.multi_timeframe_ensemble import MultiTimeframeEnsemble

@dataclass
class TimeframeConfig:
    """Configuration for each timeframe in multi-timeframe training."""
    
    timeframe: str
    weight: float
    min_samples: int = 50
    enable_training: bool = True
    feature_engineering_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate timeframe configuration."""
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h']
        if self.timeframe not in valid_timeframes:
            raise ValueError(f"Invalid timeframe: {self.timeframe}. Must be one of {valid_timeframes}")
        
        if not 0 <= self.weight <= 1:
            raise ValueError(f"Weight must be between 0 and 1, got {self.weight}")

@dataclass
class MultiTimeframeTrainingConfig:
    """Configuration for multi-timeframe training."""
    
    timeframes: List[TimeframeConfig]
    enable_cross_timeframe_features: bool = True
    enable_timeframe_ensemble: bool = True
    ensemble_method: str = "weighted_average"  # "weighted_average", "meta_learner", "stacking"
    min_confidence_threshold: float = 0.6
    enable_dynamic_weighting: bool = True
    weight_update_frequency: int = 100
    
    def __post_init__(self):
        """Validate multi-timeframe training configuration."""
        if not self.timeframes:
            raise ValueError("At least one timeframe must be specified")
        
        total_weight = sum(tf.weight for tf in self.timeframes)
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Timeframe weights must sum to 1.0, got {total_weight}")
        
        valid_ensemble_methods = ["weighted_average", "meta_learner", "stacking"]
        if self.ensemble_method not in valid_ensemble_methods:
            raise ValueError(f"Invalid ensemble method: {self.ensemble_method}. Must be one of {valid_ensemble_methods}")

class MultiTimeframeTrainer:
    """Multi-timeframe model trainer that coordinates training across multiple timeframes."""
    
    def __init__(self, config: MultiTimeframeTrainingConfig, symbol: str, exchange: str):
        """Initialize the multi-timeframe trainer.
        
        Args:
            config: Multi-timeframe training configuration
            symbol: Trading symbol
            exchange: Exchange name
        """
        self.config = config
        self.symbol = symbol
        self.exchange = exchange
        self.logger = system_logger.getChild(f'MultiTimeframeTrainer_{symbol}_{exchange}')
        
        # Initialize components
        self.feature_engine = MultiTimeframeFeatureEngineering({
            'timeframes': [tf.timeframe for tf in config.timeframes],
            'enable_cross_timeframe_features': config.enable_cross_timeframe_features
        })
        
        self.ensemble = None
        if config.enable_timeframe_ensemble:
            self.ensemble = MultiTimeframeEnsemble({
                'timeframes': [tf.timeframe for tf in config.timeframes],
                'ensemble_method': config.ensemble_method,
                'min_confidence_threshold': config.min_confidence_threshold,
                'enable_dynamic_weighting': config.enable_dynamic_weighting,
                'weight_update_frequency': config.weight_update_frequency
            })
        
        # Training state
        self.trained_models: Dict[str, Any] = {}
        self.training_results: Dict[str, Any] = {}
        self.trained = False
        
        # Apply intensity scaling
        intensity_pct = get_intensity_from_environment()
        if intensity_pct < 1.0:
            self.config = self._apply_intensity_scaling(intensity_pct)
            self.logger.info(f"🔧 Applied intensity scaling ({intensity_pct*100:.0f}%) to multi-timeframe training config")
    
    def _apply_intensity_scaling(self, intensity_pct: float) -> MultiTimeframeTrainingConfig:
        """Apply intensity scaling to the configuration."""
        # Scale down the number of timeframes if intensity is low
        if intensity_pct < 0.5:
            # Keep only the most important timeframes
            important_timeframes = ['1m', '15m', '1h']
            scaled_timeframes = [tf for tf in self.config.timeframes if tf.timeframe in important_timeframes]
            
            # Renormalize weights
            total_weight = sum(tf.weight for tf in scaled_timeframes)
            for tf in scaled_timeframes:
                tf.weight = tf.weight / total_weight
            
            return MultiTimeframeTrainingConfig(
                timeframes=scaled_timeframes,
                enable_cross_timeframe_features=self.config.enable_cross_timeframe_features,
                enable_timeframe_ensemble=self.config.enable_timeframe_ensemble,
                ensemble_method=self.config.ensemble_method,
                min_confidence_threshold=self.config.min_confidence_threshold,
                enable_dynamic_weighting=self.config.enable_dynamic_weighting,
                weight_update_frequency=self.config.weight_update_frequency
            )
        
        return self.config
    
    @handles_errors(default_return=False, context='Multi-timeframe training')
    @log_execution_time
    async def train_models(self, training_data: Dict[str, pd.DataFrame], 
                          model_trainer: Any, model_config: Dict[str, Any]) -> bool:
        """Train models across multiple timeframes.
        
        Args:
            training_data: Dict mapping timeframe -> training DataFrame
            model_trainer: Model trainer instance (general, analyst, or tactician)
            model_config: Model training configuration
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info("🚀 Starting multi-timeframe model training...")
            start_time = time.time()
            
            # 1. Prepare cross-timeframe features
            if self.config.enable_cross_timeframe_features:
                self.logger.info("🔧 Preparing cross-timeframe features...")
                training_data = await self._prepare_cross_timeframe_features(training_data)
            
            # 2. Train models for each timeframe
            timeframe_results = {}
            for tf_config in self.config.timeframes:
                if not tf_config.enable_training:
                    self.logger.info(f"⏭️ Skipping {tf_config.timeframe} training (disabled)")
                    continue
                
                if tf_config.timeframe not in training_data:
                    self.logger.warning(f"⚠️ No training data for {tf_config.timeframe}, skipping")
                    continue
                
                self.logger.info(f"🔄 Training {tf_config.timeframe} models...")
                tf_start_time = time.time()
                
                # Train models for this timeframe
                success = await self._train_timeframe_models(
                    tf_config, training_data[tf_config.timeframe], model_trainer, model_config
                )
                
                tf_training_time = time.time() - tf_start_time
                timeframe_results[tf_config.timeframe] = {
                    'success': success,
                    'training_time': tf_training_time,
                    'models_trained': len(self.trained_models.get(tf_config.timeframe, {}))
                }
                
                if success:
                    self.logger.info(f"✅ {tf_config.timeframe} training completed in {tf_training_time:.2f}s")
                else:
                    self.logger.error(f"❌ {tf_config.timeframe} training failed")
            
            # 3. Train ensemble if enabled
            if self.config.enable_timeframe_ensemble and self.ensemble:
                self.logger.info("🧠 Training multi-timeframe ensemble...")
                ensemble_start_time = time.time()
                
                ensemble_success = await self._train_ensemble(training_data)
                ensemble_training_time = time.time() - ensemble_start_time
                
                if ensemble_success:
                    self.logger.info(f"✅ Ensemble training completed in {ensemble_training_time:.2f}s")
                else:
                    self.logger.error("❌ Ensemble training failed")
            
            # 4. Save training results
            await self._save_training_results()
            
            self.trained = True
            total_time = time.time() - start_time
            
            self.logger.info("✅ Multi-timeframe training completed!")
            self.logger.info(f"⏱️ Total training time: {total_time:.2f}s")
            self.logger.info("📊 Training summary:")
            for tf, results in timeframe_results.items():
                if results['success']:
                    self.logger.info(f"   - {tf}: {results['training_time']:.2f}s, {results['models_trained']} models")
                else:
                    self.logger.info(f"   - {tf}: FAILED")
            
            return True
            
        except Exception as e:
            self.logger.exception(f"💥 Error in multi-timeframe training: {e}")
            return False
    
    @handles_errors(default_return=training_data, context='Cross-timeframe feature preparation')
    async def _prepare_cross_timeframe_features(self, training_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare cross-timeframe features."""
        try:
            enhanced_data = {}
            
            for tf_config in self.config.timeframes:
                if tf_config.timeframe not in training_data:
                    continue
                
                # Get base features for this timeframe
                base_data = training_data[tf_config.timeframe].copy()
                
                # Add cross-timeframe features
                cross_features = self.feature_engine.generate_cross_timeframe_features(
                    base_data, tf_config.timeframe, training_data
                )
                
                if cross_features is not None and not cross_features.empty:
                    # Merge cross-timeframe features
                    enhanced_data[tf_config.timeframe] = pd.concat([base_data, cross_features], axis=1)
                    self.logger.info(f"📊 Added {len(cross_features.columns)} cross-timeframe features to {tf_config.timeframe}")
                else:
                    enhanced_data[tf_config.timeframe] = base_data
            
            return enhanced_data
            
        except Exception as e:
            self.logger.exception(f"💥 Error preparing cross-timeframe features: {e}")
            return training_data
    
    @handles_errors(default_return=False, context='Timeframe model training')
    async def _train_timeframe_models(self, tf_config: TimeframeConfig, 
                                    data: pd.DataFrame, model_trainer: Any, 
                                    model_config: Dict[str, Any]) -> bool:
        """Train models for a specific timeframe."""
        try:
            # Prepare features for this timeframe
            features = self.feature_engine.prepare_timeframe_features(data, tf_config.timeframe)
            
            if features is None or features.empty:
                self.logger.warning(f"⚠️ No features prepared for {tf_config.timeframe}")
                return False
            
            # Train models using the provided model trainer
            training_result = await model_trainer.train_models({
                'features': features,
                'timeframe': tf_config.timeframe,
                'config': {**model_config, **tf_config.feature_engineering_config}
            })
            
            if training_result and training_result.get('success', False):
                # Store trained models
                self.trained_models[tf_config.timeframe] = training_result.get('models', {})
                self.training_results[tf_config.timeframe] = training_result
                
                self.logger.info(f"✅ Trained {len(self.trained_models[tf_config.timeframe])} models for {tf_config.timeframe}")
                return True
            else:
                self.logger.error(f"❌ Model training failed for {tf_config.timeframe}")
                return False
                
        except Exception as e:
            self.logger.exception(f"💥 Error training {tf_config.timeframe} models: {e}")
            return False
    
    @handles_errors(default_return=False, context='Ensemble training')
    async def _train_ensemble(self, training_data: Dict[str, pd.DataFrame]) -> bool:
        """Train the multi-timeframe ensemble."""
        try:
            if not self.ensemble:
                return False
            
            # Prepare ensemble training data
            ensemble_data = {}
            for tf_config in self.config.timeframes:
                if tf_config.timeframe in training_data and tf_config.timeframe in self.trained_models:
                    ensemble_data[tf_config.timeframe] = training_data[tf_config.timeframe]
            
            # Train ensemble
            ensemble_result = await self.ensemble.train_ensemble(ensemble_data)
            
            if ensemble_result:
                self.logger.info("✅ Multi-timeframe ensemble trained successfully")
                return True
            else:
                self.logger.error("❌ Multi-timeframe ensemble training failed")
                return False
                
        except Exception as e:
            self.logger.exception(f"💥 Error training ensemble: {e}")
            return False
    
    @handles_errors(default_return=None, context='Training results saving')
    async def _save_training_results(self) -> None:
        """Save training results and models."""
        try:
            # Save training results
            results_data = {
                'config': self.config,
                'symbol': self.symbol,
                'exchange': self.exchange,
                'trained': self.trained,
                'trained_at': get_current_datetime(),
                'training_results': self.training_results,
                'timeframe_models_count': {
                    tf: len(models) for tf, models in self.trained_models.items()
                }
            }
            
            # Save to file
            results_path = f"data_cache/multi_timeframe_training_results_{self.symbol}_{self.exchange}_{get_current_datetime()}.json"
            ensure_directory(Path(results_path).parent)
            safe_json_dump(results_data, results_path)
            
            self.logger.info(f"💾 Training results saved to {results_path}")
            
        except Exception as e:
            self.logger.exception(f"💥 Error saving training results: {e}")
    
    @handles_errors(default_return=None, context='Multi-timeframe prediction')
    async def predict(self, prediction_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Get predictions from multi-timeframe models.
        
        Args:
            prediction_data: Dict mapping timeframe -> prediction DataFrame
            
        Returns:
            Dict with predictions and metadata
        """
        try:
            if not self.trained:
                self.logger.warning("⚠️ Models not trained, returning default prediction")
                return {
                    'prediction': 'HOLD',
                    'confidence': 0.0,
                    'timeframe_contributions': {},
                    'error': 'Models not trained'
                }
            
            # Get predictions from individual timeframes
            timeframe_predictions = {}
            timeframe_confidences = {}
            
            for tf_config in self.config.timeframes:
                if tf_config.timeframe not in prediction_data or tf_config.timeframe not in self.trained_models:
                    continue
                
                # Get prediction from this timeframe's models
                tf_pred = await self._get_timeframe_prediction(
                    tf_config.timeframe, prediction_data[tf_config.timeframe]
                )
                
                if tf_pred:
                    timeframe_predictions[tf_config.timeframe] = tf_pred
                    timeframe_confidences[tf_config.timeframe] = tf_pred.get('confidence', 0.0)
            
            # Combine predictions using ensemble if available
            if self.ensemble and timeframe_predictions:
                ensemble_pred = await self.ensemble.predict(timeframe_predictions)
                return ensemble_pred
            else:
                # Fallback to weighted average
                return self._weighted_average_prediction(timeframe_predictions, timeframe_confidences)
                
        except Exception as e:
            self.logger.exception(f"💥 Error in multi-timeframe prediction: {e}")
            return {
                'prediction': 'HOLD',
                'confidence': 0.0,
                'timeframe_contributions': {},
                'error': str(e)
            }
    
    @handles_errors(default_return=None, context='Timeframe prediction')
    async def _get_timeframe_prediction(self, timeframe: str, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get prediction from a specific timeframe's models."""
        try:
            if timeframe not in self.trained_models:
                return None
            
            # Prepare features
            features = self.feature_engine.prepare_timeframe_features(data, timeframe)
            
            if features is None or features.empty:
                return None
            
            # Get predictions from all models for this timeframe
            model_predictions = []
            model_confidences = []
            
            for model_name, model in self.trained_models[timeframe].items():
                try:
                    # Get prediction from this model
                    pred = model.predict(features)
                    confidence = getattr(model, 'confidence', 0.5)  # Default confidence
                    
                    model_predictions.append(pred)
                    model_confidences.append(confidence)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Error getting prediction from {model_name}: {e}")
                    continue
            
            if not model_predictions:
                return None
            
            # Combine predictions from all models in this timeframe
            avg_prediction = np.mean(model_predictions)
            avg_confidence = np.mean(model_confidences)
            
            return {
                'prediction': avg_prediction,
                'confidence': avg_confidence,
                'model_count': len(model_predictions)
            }
            
        except Exception as e:
            self.logger.exception(f"💥 Error getting {timeframe} prediction: {e}")
            return None
    
    def _weighted_average_prediction(self, timeframe_predictions: Dict[str, Dict[str, Any]], 
                                   timeframe_confidences: Dict[str, float]) -> Dict[str, Any]:
        """Combine predictions using weighted average."""
        try:
            if not timeframe_predictions:
                return {
                    'prediction': 'HOLD',
                    'confidence': 0.0,
                    'timeframe_contributions': {}
                }
            
            # Calculate weighted average
            total_weight = 0.0
            weighted_prediction = 0.0
            weighted_confidence = 0.0
            
            timeframe_contributions = {}
            
            for tf_config in self.config.timeframes:
                tf = tf_config.timeframe
                if tf in timeframe_predictions and tf in timeframe_confidences:
                    weight = tf_config.weight
                    prediction = timeframe_predictions[tf]['prediction']
                    confidence = timeframe_confidences[tf]
                    
                    weighted_prediction += prediction * weight
                    weighted_confidence += confidence * weight
                    total_weight += weight
                    
                    timeframe_contributions[tf] = {
                        'prediction': prediction,
                        'confidence': confidence,
                        'weight': weight,
                        'contribution': prediction * weight
                    }
            
            if total_weight > 0:
                final_prediction = weighted_prediction / total_weight
                final_confidence = weighted_confidence / total_weight
            else:
                final_prediction = 0.0
                final_confidence = 0.0
            
            return {
                'prediction': final_prediction,
                'confidence': final_confidence,
                'timeframe_contributions': timeframe_contributions,
                'ensemble_method': 'weighted_average'
            }
            
        except Exception as e:
            self.logger.exception(f"💥 Error in weighted average prediction: {e}")
            return {
                'prediction': 'HOLD',
                'confidence': 0.0,
                'timeframe_contributions': {},
                'error': str(e)
            }
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get training status and statistics."""
        return {
            'trained': self.trained,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timeframes': [tf.timeframe for tf in self.config.timeframes],
            'ensemble_enabled': self.config.enable_timeframe_ensemble,
            'ensemble_method': self.config.ensemble_method,
            'timeframe_models_count': {
                tf: len(models) for tf, models in self.trained_models.items()
            },
            'training_results_summary': {
                tf: {
                    'success': results.get('success', False),
                    'models_trained': len(results.get('models', {})),
                    'training_time': results.get('training_time', 0.0)
                } for tf, results in self.training_results.items()
            }
        }