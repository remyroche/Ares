"""
Analyst Base Trainer - Unified Training Architecture

This module provides the base trainer class for all Analyst model training,
consolidating common functionality and providing a unified interface.

Key Features:
- Unified training interface for all Analyst model types
- Common training patterns and lifecycle management
- Standardized configuration and validation
- Performance monitoring and checkpointing
- Error handling and recovery mechanisms
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from .base_trainer import BaseTrainer, TrainingConfig, TrainingResult, ValidationResult, PredictionResult, TrainingRole, ModelType
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_debug, tprint_performance
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.hardware.integrated_hardware_manager import (
    get_integrated_hardware_manager, WorkloadType
)
from src.utils.hardware.optimization_decorators import (
    smart_cache, auto_optimize, memory_efficient, performance_tracked
)
from src.utils.hardware.memory_optimized_decorators import (
    memory_optimized, comprehensive_memory_optimization, MemoryOptimizationLevel
)


class AnalystModelType(Enum):
    """Types of Analyst models."""
    LIGHTGBM = "lightgbm"
    LIGHTGBM_PATCHTST = "lightgbm_patchtst"
    CATBOOST = "catboost"
    STACKER_LGBM_CALIBRATED = "stacker_lgbm_calibrated"


@dataclass
class AnalystTrainingConfig(TrainingConfig):
    """Analyst-specific training configuration."""
    # Analyst-specific parameters
    enable_patchtst_features: bool = True
    enable_regime_features: bool = True
    enable_multi_timeframe: bool = True
    
    # Model-specific parameters
    lightgbm_params: Dict[str, Any] = field(default_factory=dict)
    catboost_params: Dict[str, Any] = field(default_factory=dict)
    stacker_params: Dict[str, Any] = field(default_factory=dict)
    
    # Feature engineering parameters
    patchtst_window_size: int = 96
    patchtst_patch_length: int = 16
    regime_lookback: int = 20
    
    # Validation parameters
    analyst_validation_split: float = 0.2
    analyst_cv_folds: int = 5


class AnalystBaseTrainer(BaseTrainer):
    """
    Base trainer for all Analyst model training.
    
    This class provides a unified interface for training different types of Analyst models
    while maintaining consistent patterns for configuration, validation, and error handling.
    """
    
    def __init__(self, config: AnalystTrainingConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the Analyst base trainer.
        
        Args:
            config: Analyst training configuration
            logger: Logger instance (optional)
        """
        # Set role to ANALYST
        config.role = TrainingRole.ANALYST
        
        super().__init__(config, logger)
        
        # Analyst-specific state
        self._analyst_state = {
            'patchtst_features_created': False,
            'regime_features_created': False,
            'multi_timeframe_features_created': False,
            'feature_engineering_completed': False
        }
        
        tprint_info(f"🔧 Initialized AnalystBaseTrainer for {config.timeframe}")
        self.logger.info(f"Initialized AnalystBaseTrainer for {config.timeframe}")
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=TrainingResult(
            success=False,
            error_message="Analyst training failed"
        ),
        context="analyst training"
    )
    async def train(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> TrainingResult:
        """
        Train Analyst models with given data.
        
        Args:
            data: Training data
            targets: Target variables (optional, can be inferred from data)
            
        Returns:
            Training result with model and metrics
        """
        try:
            tprint_info("📊 Starting Analyst model training...")
            self.logger.info("Starting Analyst model training...")
            
            start_time = time.time()
            
            # Preprocess data
            processed_data, processed_targets = self._preprocess_data(data, targets)
            
            # Create features
            feature_data = await self._create_analyst_features(processed_data)
            
            # Train models
            training_results = {}
            for model_type in self.config.model_types:
                tprint_info(f"🔧 Training {model_type.value} model...")
                
                model_result = await self._train_single_model(
                    model_type, feature_data, processed_targets
                )
                training_results[model_type.value] = model_result
                
                if model_result.success:
                    tprint_success(f"✅ {model_type.value} model trained successfully")
                else:
                    tprint_error(f"❌ {model_type.value} model training failed: {model_result.error_message}")
            
            # Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics(training_results)
            
            # Update training state
            self._training_state['training_completed'] = True
            self._training_state['training_started'] = True
            
            training_time = time.time() - start_time
            self._update_performance_metrics('training', training_time)
            
            # Create result
            result = TrainingResult(
                success=True,
                model=training_results,
                metrics=overall_metrics,
                training_time=training_time,
                metadata={
                    'analyst_state': self._analyst_state.copy(),
                    'models_trained': list(training_results.keys()),
                    'feature_engineering_completed': self._analyst_state['feature_engineering_completed']
                }
            )
            
            tprint_success(f"✅ Analyst training completed in {training_time:.2f}s")
            self.logger.info(f"Analyst training completed in {training_time:.2f}s")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Analyst training failed: {e}")
            self.logger.error(f"Analyst training failed: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e),
                training_time=time.time() - start_time if 'start_time' in locals() else 0.0
            )
    
    async def validate(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> ValidationResult:
        """
        Validate the trained Analyst models.
        
        Args:
            data: Validation data
            targets: Target variables (optional)
            
        Returns:
            Validation result with metrics
        """
        try:
            tprint_info("📊 Validating Analyst models...")
            self.logger.info("Validating Analyst models...")
            
            start_time = time.time()
            
            # Preprocess data
            processed_data, processed_targets = self._preprocess_data(data, targets)
            
            # Create features
            feature_data = await self._create_analyst_features(processed_data)
            
            # Validate models
            validation_metrics = {}
            for model_type in self.config.model_types:
                model_key = f"{model_type.value}_model"
                if model_key in self._model_state:
                    model = self._model_state[model_key]
                    metrics = await self._validate_single_model(model, model_type, feature_data, processed_targets)
                    validation_metrics[model_type.value] = metrics
            
            validation_time = time.time() - start_time
            self._update_performance_metrics('validation', validation_time)
            
            result = ValidationResult(
                success=True,
                metrics=validation_metrics,
                metadata={
                    'validation_time': validation_time,
                    'models_validated': list(validation_metrics.keys())
                }
            )
            
            tprint_success(f"✅ Analyst validation completed in {validation_time:.2f}s")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Analyst validation failed: {e}")
            self.logger.error(f"Analyst validation failed: {e}")
            return ValidationResult(
                success=False,
                error_message=str(e)
            )
    
    async def predict(self, data: pd.DataFrame) -> PredictionResult:
        """
        Make predictions with the trained Analyst models.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Prediction result
        """
        try:
            tprint_info("📊 Making Analyst predictions...")
            self.logger.info("Making Analyst predictions...")
            
            start_time = time.time()
            
            # Preprocess data
            processed_data, _ = self._preprocess_data(data, None)
            
            # Create features
            feature_data = await self._create_analyst_features(processed_data)
            
            # Make predictions
            predictions = {}
            probabilities = {}
            
            for model_type in self.config.model_types:
                model_key = f"{model_type.value}_model"
                if model_key in self._model_state:
                    model = self._model_state[model_key]
                    pred, prob = await self._predict_single_model(model, model_type, feature_data)
                    predictions[model_type.value] = pred
                    if prob is not None:
                        probabilities[model_type.value] = prob
            
            prediction_time = time.time() - start_time
            self._update_performance_metrics('prediction', prediction_time)
            
            result = PredictionResult(
                success=True,
                predictions=predictions,
                probabilities=probabilities if probabilities else None,
                metadata={
                    'prediction_time': prediction_time,
                    'models_used': list(predictions.keys())
                }
            )
            
            tprint_success(f"✅ Analyst predictions completed in {prediction_time:.2f}s")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Analyst prediction failed: {e}")
            self.logger.error(f"Analyst prediction failed: {e}")
            return PredictionResult(
                success=False,
                error_message=str(e)
            )
    
    def _create_model(self, model_type: ModelType) -> Any:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model to create
            
        Returns:
            Model instance
        """
        try:
            if model_type == ModelType.LIGHTGBM:
                return self._create_lightgbm_model()
            elif model_type == ModelType.CATBOOST:
                return self._create_catboost_model()
            elif model_type == ModelType.NEURAL_NETWORK:
                return self._create_neural_network_model()
            else:
                raise ValueError(f"Unsupported model type for Analyst: {model_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to create {model_type.value} model: {e}")
            return None
    
    def _get_feature_importance(self, model: Any) -> Optional[Dict[str, float]]:
        """
        Extract feature importance from model.
        
        Args:
            model: Trained model
            
        Returns:
            Feature importance dictionary
        """
        try:
            if hasattr(model, 'feature_importances_'):
                return dict(zip(model.feature_names_in_, model.feature_importances_))
            elif hasattr(model, 'get_feature_importance'):
                return model.get_feature_importance()
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
            return None
    
    async def _create_analyst_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create Analyst-specific features.
        
        Args:
            data: Input data
            
        Returns:
            Data with Analyst features
        """
        try:
            tprint_info("🔧 Creating Analyst features...")
            
            feature_data = data.copy()
            
            # Create PatchTST features if enabled
            if self.config.enable_patchtst_features:
                feature_data = await self._create_patchtst_features(feature_data)
                self._analyst_state['patchtst_features_created'] = True
            
            # Create regime features if enabled
            if self.config.enable_regime_features:
                feature_data = await self._create_regime_features(feature_data)
                self._analyst_state['regime_features_created'] = True
            
            # Create multi-timeframe features if enabled
            if self.config.enable_multi_timeframe:
                feature_data = await self._create_multi_timeframe_features(feature_data)
                self._analyst_state['multi_timeframe_features_created'] = True
            
            self._analyst_state['feature_engineering_completed'] = True
            tprint_success(f"✅ Created {feature_data.shape[1]} features")
            
            return feature_data
            
        except Exception as e:
            tprint_error(f"❌ Feature creation failed: {e}")
            self.logger.error(f"Feature creation failed: {e}")
            return data
    
    async def _create_patchtst_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create PatchTST (Patch Time Series Transformer) features for time series analysis."""
        try:
            tprint_debug("🔧 Creating PatchTST features...")
            
            # Create a copy to avoid modifying original data
            feature_data = data.copy()
            
            if 'close' not in data.columns:
                tprint_warning("⚠️ No 'close' column found for PatchTST features")
                return data
            
            # PatchTST parameters
            patch_len = 16  # Length of each patch
            stride = 8      # Stride between patches
            num_patches = 4 # Number of patches to create
            
            # Create patches from time series
            close_prices = data['close'].values
            
            # Ensure we have enough data
            if len(close_prices) < patch_len * num_patches:
                tprint_warning("⚠️ Insufficient data for PatchTST features")
                return data
            
            # Create patch features
            patch_features = []
            for i in range(0, len(close_prices) - patch_len + 1, stride):
                if len(patch_features) >= num_patches:
                    break
                patch = close_prices[i:i + patch_len]
                patch_features.append(patch)
            
            # Pad with zeros if we don't have enough patches
            while len(patch_features) < num_patches:
                patch_features.append(np.zeros(patch_len))
            
            # Convert to numpy array
            patch_array = np.array(patch_features)
            
            # Create PatchTST features
            for i, patch in enumerate(patch_array):
                # Patch statistics
                feature_data[f'patchtst_patch_{i}_mean'] = np.mean(patch)
                feature_data[f'patchtst_patch_{i}_std'] = np.std(patch)
                feature_data[f'patchtst_patch_{i}_min'] = np.min(patch)
                feature_data[f'patchtst_patch_{i}_max'] = np.max(patch)
                feature_data[f'patchtst_patch_{i}_range'] = np.max(patch) - np.min(patch)
                
                # Patch trends
                feature_data[f'patchtst_patch_{i}_trend'] = np.polyfit(range(len(patch)), patch, 1)[0]
                feature_data[f'patchtst_patch_{i}_r2'] = np.corrcoef(range(len(patch)), patch)[0, 1] ** 2
                
                # Patch volatility
                feature_data[f'patchtst_patch_{i}_volatility'] = np.std(np.diff(patch))
                
                # Patch momentum
                feature_data[f'patchtst_patch_{i}_momentum'] = patch[-1] - patch[0]
                feature_data[f'patchtst_patch_{i}_momentum_pct'] = (patch[-1] - patch[0]) / patch[0] if patch[0] != 0 else 0
            
            # Cross-patch features
            if len(patch_features) >= 2:
                # Patch correlation
                feature_data['patchtst_patch_correlation'] = np.corrcoef(patch_array[0], patch_array[1])[0, 1] if len(patch_array) >= 2 else 0
                
                # Patch similarity
                feature_data['patchtst_patch_similarity'] = np.corrcoef(patch_array[0], patch_array[1])[0, 1] if len(patch_array) >= 2 else 0
                
                # Patch divergence
                feature_data['patchtst_patch_divergence'] = np.mean(np.abs(patch_array[0] - patch_array[1])) if len(patch_array) >= 2 else 0
            
            # Temporal patch features
            feature_data['patchtst_temporal_consistency'] = np.mean([np.corrcoef(patch_array[i], patch_array[i+1])[0, 1] 
                                                                   for i in range(len(patch_array)-1)]) if len(patch_array) > 1 else 0
            
            # PatchTST attention-like features
            feature_data['patchtst_attention_weights'] = np.mean([np.var(patch) for patch in patch_array])
            feature_data['patchtst_patch_entropy'] = np.mean([-np.sum(patch * np.log(patch + 1e-8)) for patch in patch_array])
            
            # Multi-scale patch features
            if len(close_prices) >= patch_len * 2:
                # Longer patches
                long_patch_len = patch_len * 2
                long_patches = []
                for i in range(0, len(close_prices) - long_patch_len + 1, stride * 2):
                    if len(long_patches) >= 2:
                        break
                    long_patch = close_prices[i:i + long_patch_len]
                    long_patches.append(long_patch)
                
                if len(long_patches) >= 1:
                    long_patch = long_patches[0]
                    feature_data['patchtst_long_patch_mean'] = np.mean(long_patch)
                    feature_data['patchtst_long_patch_std'] = np.std(long_patch)
                    feature_data['patchtst_long_patch_trend'] = np.polyfit(range(len(long_patch)), long_patch, 1)[0]
            
            # Fill NaN values
            feature_data = feature_data.fillna(method='ffill').fillna(0)
            
            tprint_success(f"✅ Created {len([col for col in feature_data.columns if col.startswith('patchtst_')])} PatchTST features")
            return feature_data
            
        except Exception as e:
            tprint_error(f"❌ PatchTST feature creation failed: {e}")
            self.logger.error(f"PatchTST feature creation failed: {e}")
            return data
    
    async def _create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create regime detection and classification features."""
        try:
            tprint_debug("🔧 Creating regime features...")
            
            # Create a copy to avoid modifying original data
            feature_data = data.copy()
            
            if 'close' not in data.columns:
                tprint_warning("⚠️ No 'close' column found for regime features")
                return data
            
            # Price-based regime features
            close_prices = data['close']
            returns = close_prices.pct_change()
            
            # Volatility regime features
            vol_short = returns.rolling(20).std()
            vol_long = returns.rolling(50).std()
            feature_data['regime_vol_ratio'] = vol_short / vol_long
            feature_data['regime_vol_regime'] = (vol_short > vol_long * 1.2).astype(int)  # High vol regime
            feature_data['regime_vol_regime_low'] = (vol_short < vol_long * 0.8).astype(int)  # Low vol regime
            
            # Trend regime features
            sma_short = close_prices.rolling(20).mean()
            sma_medium = close_prices.rolling(50).mean()
            sma_long = close_prices.rolling(100).mean()
            
            feature_data['regime_trend_short'] = (close_prices > sma_short).astype(int)
            feature_data['regime_trend_medium'] = (close_prices > sma_medium).astype(int)
            feature_data['regime_trend_long'] = (close_prices > sma_long).astype(int)
            
            # Trend strength
            feature_data['regime_trend_strength'] = (sma_short - sma_long) / sma_long
            feature_data['regime_trend_consistency'] = ((close_prices > sma_short) & (sma_short > sma_medium) & (sma_medium > sma_long)).astype(int)
            
            # Market regime classification
            # Bull market: price above long MA, positive trend
            feature_data['regime_bull_market'] = ((close_prices > sma_long) & (feature_data['regime_trend_strength'] > 0.02)).astype(int)
            
            # Bear market: price below long MA, negative trend
            feature_data['regime_bear_market'] = ((close_prices < sma_long) & (feature_data['regime_trend_strength'] < -0.02)).astype(int)
            
            # Sideways market: price near long MA, low trend strength
            feature_data['regime_sideways_market'] = ((abs(feature_data['regime_trend_strength']) < 0.02) & 
                                                    (close_prices > sma_long * 0.95) & 
                                                    (close_prices < sma_long * 1.05)).astype(int)
            
            # Momentum regime features
            momentum_5 = close_prices.pct_change(5)
            momentum_20 = close_prices.pct_change(20)
            momentum_50 = close_prices.pct_change(50)
            
            feature_data['regime_momentum_5'] = momentum_5
            feature_data['regime_momentum_20'] = momentum_20
            feature_data['regime_momentum_50'] = momentum_50
            
            # Momentum regime classification
            feature_data['regime_momentum_strong'] = ((momentum_5 > 0.02) & (momentum_20 > 0.02) & (momentum_50 > 0.02)).astype(int)
            feature_data['regime_momentum_weak'] = ((abs(momentum_5) < 0.01) & (abs(momentum_20) < 0.01)).astype(int)
            feature_data['regime_momentum_reversal'] = ((momentum_5 * momentum_20 < 0) & (abs(momentum_5) > 0.01)).astype(int)
            
            # Volatility clustering regime
            vol_cluster = (vol_short > vol_short.rolling(10).mean() * 1.5).astype(int)
            feature_data['regime_vol_cluster'] = vol_cluster
            feature_data['regime_vol_cluster_duration'] = vol_cluster.groupby((vol_cluster != vol_cluster.shift()).cumsum()).cumsum()
            
            # Mean reversion regime features
            z_score = (close_prices - close_prices.rolling(50).mean()) / close_prices.rolling(50).std()
            feature_data['regime_z_score'] = z_score
            feature_data['regime_mean_reversion'] = (abs(z_score) > 2).astype(int)
            feature_data['regime_mean_reversion_strong'] = (abs(z_score) > 3).astype(int)
            
            # Range-bound regime
            high_20 = close_prices.rolling(20).max()
            low_20 = close_prices.rolling(20).min()
            range_size = (high_20 - low_20) / close_prices
            feature_data['regime_range_size'] = range_size
            feature_data['regime_range_bound'] = (range_size < 0.05).astype(int)  # Less than 5% range
            
            # Breakout regime features
            feature_data['regime_breakout_up'] = (close_prices > high_20.shift(1)).astype(int)
            feature_data['regime_breakout_down'] = (close_prices < low_20.shift(1)).astype(int)
            feature_data['regime_breakout_any'] = (feature_data['regime_breakout_up'] | feature_data['regime_breakout_down']).astype(int)
            
            # Volume regime features (if available)
            if 'volume' in data.columns:
                vol_ratio = data['volume'] / data['volume'].rolling(20).mean()
                feature_data['regime_volume_regime'] = (vol_ratio > 1.5).astype(int)
                feature_data['regime_volume_regime_low'] = (vol_ratio < 0.5).astype(int)
                
                # Volume-price relationship
                feature_data['regime_volume_price_corr'] = data['volume'].rolling(20).corr(close_prices)
                feature_data['regime_volume_divergence'] = ((close_prices > close_prices.shift(5)) & (vol_ratio < 1)).astype(int)
            
            # Time-based regime features
            if data.index.dtype == 'datetime64[ns]' or hasattr(data.index, 'hour'):
                feature_data['regime_hour'] = data.index.hour if hasattr(data.index, 'hour') else 0
                feature_data['regime_day_of_week'] = data.index.dayofweek if hasattr(data.index, 'dayofweek') else 0
                feature_data['regime_month'] = data.index.month if hasattr(data.index, 'month') else 0
                
                # Session-based regimes
                feature_data['regime_london_session'] = ((feature_data['regime_hour'] >= 8) & (feature_data['regime_hour'] < 16)).astype(int)
                feature_data['regime_ny_session'] = ((feature_data['regime_hour'] >= 13) & (feature_data['regime_hour'] < 21)).astype(int)
                feature_data['regime_asian_session'] = ((feature_data['regime_hour'] >= 0) & (feature_data['regime_hour'] < 8)).astype(int)
                
                # Weekend effect
                feature_data['regime_weekend'] = (feature_data['regime_day_of_week'] >= 5).astype(int)
            
            # Regime persistence features
            feature_data['regime_bull_persistence'] = feature_data['regime_bull_market'].groupby((feature_data['regime_bull_market'] != feature_data['regime_bull_market'].shift()).cumsum()).cumsum()
            feature_data['regime_bear_persistence'] = feature_data['regime_bear_market'].groupby((feature_data['regime_bear_market'] != feature_data['regime_bear_market'].shift()).cumsum()).cumsum()
            feature_data['regime_sideways_persistence'] = feature_data['regime_sideways_market'].groupby((feature_data['regime_sideways_market'] != feature_data['regime_sideways_market'].shift()).cumsum()).cumsum()
            
            # Regime transition features
            feature_data['regime_transition'] = ((feature_data['regime_bull_market'] != feature_data['regime_bull_market'].shift()) |
                                               (feature_data['regime_bear_market'] != feature_data['regime_bear_market'].shift()) |
                                               (feature_data['regime_sideways_market'] != feature_data['regime_sideways_market'].shift())).astype(int)
            
            # Regime stability score
            regime_stability = 1 - feature_data['regime_transition'].rolling(20).mean()
            feature_data['regime_stability_score'] = regime_stability
            
            # Fill NaN values
            feature_data = feature_data.fillna(method='ffill').fillna(0)
            
            tprint_success(f"✅ Created {len([col for col in feature_data.columns if col.startswith('regime_')])} regime features")
            return feature_data
            
        except Exception as e:
            tprint_error(f"❌ Regime feature creation failed: {e}")
            self.logger.error(f"Regime feature creation failed: {e}")
            return data
    
    async def _create_multi_timeframe_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create multi-timeframe analysis features."""
        try:
            tprint_debug("🔧 Creating multi-timeframe features...")
            
            # Create a copy to avoid modifying original data
            feature_data = data.copy()
            
            if 'close' not in data.columns:
                tprint_warning("⚠️ No 'close' column found for multi-timeframe features")
                return data
            
            close_prices = data['close']
            returns = close_prices.pct_change()
            
            # Define multiple timeframes
            timeframes = {
                'short': [5, 10, 15],
                'medium': [20, 30, 50],
                'long': [100, 200, 300]
            }
            
            # Price-based multi-timeframe features
            for tf_name, periods in timeframes.items():
                for period in periods:
                    # Moving averages
                    sma = close_prices.rolling(period).mean()
                    feature_data[f'mtf_{tf_name}_sma_{period}'] = sma
                    feature_data[f'mtf_{tf_name}_price_vs_sma_{period}'] = (close_prices - sma) / sma
                    
                    # Price position within range
                    high = close_prices.rolling(period).max()
                    low = close_prices.rolling(period).min()
                    feature_data[f'mtf_{tf_name}_position_{period}'] = (close_prices - low) / (high - low)
                    
                    # Volatility
                    vol = returns.rolling(period).std()
                    feature_data[f'mtf_{tf_name}_volatility_{period}'] = vol
                    
                    # Momentum
                    momentum = close_prices.pct_change(period)
                    feature_data[f'mtf_{tf_name}_momentum_{period}'] = momentum
                    
                    # Trend strength
                    trend_strength = (close_prices - close_prices.shift(period)) / close_prices.shift(period)
                    feature_data[f'mtf_{tf_name}_trend_strength_{period}'] = trend_strength
            
            # Cross-timeframe relationships
            # Short vs Medium
            feature_data['mtf_short_vs_medium_trend'] = (feature_data['mtf_short_sma_10'] - feature_data['mtf_medium_sma_20']) / feature_data['mtf_medium_sma_20']
            feature_data['mtf_short_vs_medium_momentum'] = feature_data['mtf_short_momentum_10'] - feature_data['mtf_medium_momentum_20']
            feature_data['mtf_short_vs_medium_vol'] = feature_data['mtf_short_volatility_10'] / feature_data['mtf_medium_volatility_20']
            
            # Medium vs Long
            feature_data['mtf_medium_vs_long_trend'] = (feature_data['mtf_medium_sma_50'] - feature_data['mtf_long_sma_100']) / feature_data['mtf_long_sma_100']
            feature_data['mtf_medium_vs_long_momentum'] = feature_data['mtf_medium_momentum_50'] - feature_data['mtf_long_momentum_100']
            feature_data['mtf_medium_vs_long_vol'] = feature_data['mtf_medium_volatility_50'] / feature_data['mtf_long_volatility_100']
            
            # Short vs Long
            feature_data['mtf_short_vs_long_trend'] = (feature_data['mtf_short_sma_5'] - feature_data['mtf_long_sma_200']) / feature_data['mtf_long_sma_200']
            feature_data['mtf_short_vs_long_momentum'] = feature_data['mtf_short_momentum_5'] - feature_data['mtf_long_momentum_200']
            feature_data['mtf_short_vs_long_vol'] = feature_data['mtf_short_volatility_5'] / feature_data['mtf_long_volatility_200']
            
            # Multi-timeframe alignment features
            # All timeframes bullish
            feature_data['mtf_all_bullish'] = ((feature_data['mtf_short_price_vs_sma_5'] > 0) &
                                             (feature_data['mtf_medium_price_vs_sma_20'] > 0) &
                                             (feature_data['mtf_long_price_vs_sma_100'] > 0)).astype(int)
            
            # All timeframes bearish
            feature_data['mtf_all_bearish'] = ((feature_data['mtf_short_price_vs_sma_5'] < 0) &
                                             (feature_data['mtf_medium_price_vs_sma_20'] < 0) &
                                             (feature_data['mtf_long_price_vs_sma_100'] < 0)).astype(int)
            
            # Divergence between timeframes
            feature_data['mtf_divergence_short_medium'] = ((feature_data['mtf_short_momentum_10'] > 0) & 
                                                         (feature_data['mtf_medium_momentum_20'] < 0)).astype(int)
            feature_data['mtf_divergence_medium_long'] = ((feature_data['mtf_medium_momentum_50'] > 0) & 
                                                        (feature_data['mtf_long_momentum_100'] < 0)).astype(int)
            feature_data['mtf_divergence_short_long'] = ((feature_data['mtf_short_momentum_5'] > 0) & 
                                                       (feature_data['mtf_long_momentum_200'] < 0)).astype(int)
            
            # Multi-timeframe volatility analysis
            feature_data['mtf_vol_short_medium_ratio'] = feature_data['mtf_short_volatility_10'] / feature_data['mtf_medium_volatility_20']
            feature_data['mtf_vol_medium_long_ratio'] = feature_data['mtf_medium_volatility_50'] / feature_data['mtf_long_volatility_100']
            feature_data['mtf_vol_short_long_ratio'] = feature_data['mtf_short_volatility_5'] / feature_data['mtf_long_volatility_200']
            
            # Volatility regime across timeframes
            feature_data['mtf_vol_regime_high'] = ((feature_data['mtf_vol_short_medium_ratio'] > 1.5) |
                                                 (feature_data['mtf_vol_medium_long_ratio'] > 1.5)).astype(int)
            feature_data['mtf_vol_regime_low'] = ((feature_data['mtf_vol_short_medium_ratio'] < 0.7) |
                                                (feature_data['mtf_vol_medium_long_ratio'] < 0.7)).astype(int)
            
            # Multi-timeframe momentum analysis
            feature_data['mtf_momentum_alignment'] = ((feature_data['mtf_short_momentum_10'] > 0) == 
                                                    (feature_data['mtf_medium_momentum_20'] > 0) == 
                                                    (feature_data['mtf_long_momentum_100'] > 0)).astype(int)
            
            # Momentum strength across timeframes
            feature_data['mtf_momentum_strength_short'] = abs(feature_data['mtf_short_momentum_10'])
            feature_data['mtf_momentum_strength_medium'] = abs(feature_data['mtf_medium_momentum_20'])
            feature_data['mtf_momentum_strength_long'] = abs(feature_data['mtf_long_momentum_100'])
            
            # Multi-timeframe position analysis
            feature_data['mtf_position_short'] = feature_data['mtf_short_position_10']
            feature_data['mtf_position_medium'] = feature_data['mtf_medium_position_50']
            feature_data['mtf_position_long'] = feature_data['mtf_long_position_200']
            
            # Position alignment
            feature_data['mtf_position_alignment'] = ((feature_data['mtf_position_short'] > 0.5) == 
                                                    (feature_data['mtf_position_medium'] > 0.5) == 
                                                    (feature_data['mtf_position_long'] > 0.5)).astype(int)
            
            # Multi-timeframe breakout analysis
            feature_data['mtf_breakout_short'] = (close_prices > close_prices.rolling(10).max().shift(1)).astype(int)
            feature_data['mtf_breakout_medium'] = (close_prices > close_prices.rolling(50).max().shift(1)).astype(int)
            feature_data['mtf_breakout_long'] = (close_prices > close_prices.rolling(200).max().shift(1)).astype(int)
            
            # Multi-timeframe breakout alignment
            feature_data['mtf_breakout_alignment'] = ((feature_data['mtf_breakout_short'] == 1) &
                                                    (feature_data['mtf_breakout_medium'] == 1) &
                                                    (feature_data['mtf_breakout_long'] == 1)).astype(int)
            
            # Time-based multi-timeframe features
            if data.index.dtype == 'datetime64[ns]' or hasattr(data.index, 'hour'):
                # Intraday patterns
                feature_data['mtf_hour'] = data.index.hour if hasattr(data.index, 'hour') else 0
                feature_data['mtf_day_of_week'] = data.index.dayofweek if hasattr(data.index, 'dayofweek') else 0
                
                # Session-based multi-timeframe analysis
                london_session = ((feature_data['mtf_hour'] >= 8) & (feature_data['mtf_hour'] < 16)).astype(int)
                ny_session = ((feature_data['mtf_hour'] >= 13) & (feature_data['mtf_hour'] < 21)).astype(int)
                asian_session = ((feature_data['mtf_hour'] >= 0) & (feature_data['mtf_hour'] < 8)).astype(int)
                
                # Multi-timeframe performance by session
                feature_data['mtf_london_performance'] = (feature_data['mtf_short_momentum_5'] * london_session)
                feature_data['mtf_ny_performance'] = (feature_data['mtf_short_momentum_5'] * ny_session)
                feature_data['mtf_asian_performance'] = (feature_data['mtf_short_momentum_5'] * asian_session)
            
            # Multi-timeframe correlation features
            feature_data['mtf_correlation_short_medium'] = feature_data['mtf_short_momentum_10'].rolling(20).corr(feature_data['mtf_medium_momentum_20'])
            feature_data['mtf_correlation_medium_long'] = feature_data['mtf_medium_momentum_20'].rolling(20).corr(feature_data['mtf_long_momentum_100'])
            feature_data['mtf_correlation_short_long'] = feature_data['mtf_short_momentum_10'].rolling(20).corr(feature_data['mtf_long_momentum_200'])
            
            # Multi-timeframe stability score
            stability_short = 1 - abs(feature_data['mtf_short_momentum_10']).rolling(10).std()
            stability_medium = 1 - abs(feature_data['mtf_medium_momentum_20']).rolling(10).std()
            stability_long = 1 - abs(feature_data['mtf_long_momentum_100']).rolling(10).std()
            
            feature_data['mtf_stability_short'] = stability_short
            feature_data['mtf_stability_medium'] = stability_medium
            feature_data['mtf_stability_long'] = stability_long
            feature_data['mtf_stability_overall'] = (stability_short + stability_medium + stability_long) / 3
            
            # Fill NaN values
            feature_data = feature_data.fillna(method='ffill').fillna(0)
            
            tprint_success(f"✅ Created {len([col for col in feature_data.columns if col.startswith('mtf_')])} multi-timeframe features")
            return feature_data
            
        except Exception as e:
            tprint_error(f"❌ Multi-timeframe feature creation failed: {e}")
            self.logger.error(f"Multi-timeframe feature creation failed: {e}")
            return data
    
    def _create_lightgbm_model(self):
        """Create LightGBM model."""
        try:
            import lightgbm as lgb
            
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                **self.config.lightgbm_params
            }
            
            return lgb.LGBMClassifier(**params)
            
        except ImportError:
            self.logger.error("LightGBM not available")
            return None
    
    def _create_catboost_model(self):
        """Create CatBoost model."""
        try:
            import catboost as cb
            
            params = {
                'iterations': 1000,
                'learning_rate': 0.1,
                'depth': 6,
                'verbose': False,
                **self.config.catboost_params
            }
            
            return cb.CatBoostClassifier(**params)
            
        except ImportError:
            self.logger.error("CatBoost not available")
            return None
    
    def _create_neural_network_model(self):
        """Create neural network model for Analyst predictions."""
        try:
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            
            # Neural network parameters optimized for Analyst (classification)
            params = {
                'hidden_layer_sizes': (150, 75, 25),  # Network architecture for classification
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,  # L2 regularization
                'learning_rate': 'adaptive',
                'learning_rate_init': 0.001,
                'max_iter': 1000,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 20,
                'random_state': self.config.random_seed,
                'warm_start': False,
                'momentum': 0.9,
                'nesterovs_momentum': True,
                'beta_1': 0.9,
                'beta_2': 0.999,
                'epsilon': 1e-8,
                'max_fun': 15000
            }
            
            # Update with custom parameters
            params.update(self.config.neural_network_params)
            
            # Create pipeline with scaling for better convergence
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPClassifier(**params))
            ])
            
            tprint_debug("🔧 Created neural network model for Analyst classification")
            return pipeline
            
        except ImportError as e:
            tprint_error(f"❌ scikit-learn not available for neural network: {e}")
            self.logger.error(f"scikit-learn not available for neural network: {e}")
            return None
        except Exception as e:
            tprint_error(f"❌ Failed to create neural network model: {e}")
            self.logger.error(f"Failed to create neural network model: {e}")
            return None
    
    async def _train_single_model(self, model_type: ModelType, data: pd.DataFrame, targets: pd.Series) -> TrainingResult:
        """Train a single model."""
        try:
            model = self._create_model(model_type)
            if model is None:
                return TrainingResult(
                    success=False,
                    error_message=f"Failed to create {model_type.value} model"
                )
            
            # Train the model
            model.fit(data, targets)
            
            # Store model
            model_key = f"{model_type.value}_model"
            self._model_state[model_key] = model
            
            # Get feature importance
            feature_importance = self._get_feature_importance(model)
            
            return TrainingResult(
                success=True,
                model=model,
                feature_importance=feature_importance,
                metadata={'model_type': model_type.value}
            )
            
        except Exception as e:
            return TrainingResult(
                success=False,
                error_message=str(e)
            )
    
    async def _validate_single_model(self, model: Any, model_type: ModelType, data: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """Validate a single model."""
        try:
            predictions = model.predict(data)
            probabilities = model.predict_proba(data)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {
                'accuracy': accuracy_score(targets, predictions),
                'precision': precision_score(targets, predictions, average='weighted'),
                'recall': recall_score(targets, predictions, average='weighted'),
                'f1': f1_score(targets, predictions, average='weighted')
            }
            
            if probabilities is not None:
                metrics['roc_auc'] = roc_auc_score(targets, probabilities)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Validation failed for {model_type.value}: {e}")
            return {}
    
    async def _predict_single_model(self, model: Any, model_type: ModelType, data: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions with a single model."""
        try:
            predictions = model.predict(data)
            probabilities = model.predict_proba(data)[:, 1] if hasattr(model, 'predict_proba') else None
            
            return predictions, probabilities
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {model_type.value}: {e}")
            return np.array([]), None
    
    def _calculate_overall_metrics(self, training_results: Dict[str, TrainingResult]) -> Dict[str, float]:
        """Calculate overall training metrics."""
        successful_models = [r for r in training_results.values() if r.success]
        
        return {
            'total_models': len(training_results),
            'successful_models': len(successful_models),
            'success_rate': len(successful_models) / len(training_results) if training_results else 0.0
        }
    
    def get_analyst_summary(self) -> Dict[str, Any]:
        """Get Analyst-specific training summary."""
        base_summary = self.get_training_summary()
        base_summary.update({
            'analyst_state': self._analyst_state.copy(),
            'feature_engineering': {
                'patchtst_features': self._analyst_state['patchtst_features_created'],
                'regime_features': self._analyst_state['regime_features_created'],
                'multi_timeframe_features': self._analyst_state['multi_timeframe_features_created'],
                'completed': self._analyst_state['feature_engineering_completed']
            }
        })
        return base_summary