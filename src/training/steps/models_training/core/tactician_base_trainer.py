"""
Tactician Base Trainer - Unified Training Architecture

This module provides the base trainer class for all Tactician model training,
consolidating common functionality and providing a unified interface.

Key Features:
- Unified training interface for all Tactician model types
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


class TacticianModelType(Enum):
    """Types of Tactician models."""
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    NEURAL_NETWORK = "neural_network"
    LINEAR = "linear"


@dataclass
class TacticianTrainingConfig(TrainingConfig):
    """Tactician-specific training configuration."""
    # Tactician-specific parameters
    enable_entry_timing: bool = True
    enable_exit_timing: bool = True
    enable_position_sizing: bool = True
    
    # Model-specific parameters
    lightgbm_params: Dict[str, Any] = field(default_factory=dict)
    catboost_params: Dict[str, Any] = field(default_factory=dict)
    neural_network_params: Dict[str, Any] = field(default_factory=dict)
    linear_params: Dict[str, Any] = field(default_factory=dict)
    
    # Timing parameters
    entry_lookback: int = 10
    exit_lookback: int = 5
    position_sizing_lookback: int = 20
    
    # Validation parameters
    tactician_validation_split: float = 0.2
    tactician_cv_folds: int = 5


class TacticianBaseTrainer(BaseTrainer):
    """
    Base trainer for all Tactician model training.
    
    This class provides a unified interface for training different types of Tactician models
    while maintaining consistent patterns for configuration, validation, and error handling.
    """
    
    def __init__(self, config: TacticianTrainingConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the Tactician base trainer.
        
        Args:
            config: Tactician training configuration
            logger: Logger instance (optional)
        """
        # Set role to TACTICIAN
        config.role = TrainingRole.TACTICIAN
        
        super().__init__(config, logger)
        
        # Tactician-specific state
        self._tactician_state = {
            'entry_timing_features_created': False,
            'exit_timing_features_created': False,
            'position_sizing_features_created': False,
            'timing_features_completed': False
        }
        
        tprint_info(f"🔧 Initialized TacticianBaseTrainer for {config.timeframe}")
        self.logger.info(f"Initialized TacticianBaseTrainer for {config.timeframe}")
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=TrainingResult(
            success=False,
            error_message="Tactician training failed"
        ),
        context="tactician training"
    )
    @comprehensive_memory_optimization(
        optimization_level=MemoryOptimizationLevel.AGGRESSIVE,
        enable_caching=True,
        enable_chunking=True,
        enable_gc=True,
        enable_pools=True
    )
    @performance_tracked(log_performance=True, track_memory=True)
    async def train(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> TrainingResult:
        """
        Train Tactician models with given data.
        
        Args:
            data: Training data
            targets: Target variables (optional, can be inferred from data)
            
        Returns:
            Training result with model and metrics
        """
        try:
            tprint_info("⚔️ Starting Tactician model training...")
            self.logger.info("Starting Tactician model training...")
            
            start_time = time.time()
            
            # Preview input data
            from src.utils.tprint import tprint_data_preview
            tprint_data_preview(data, "Input training data", max_rows=5, level="INFO")
            if targets is not None:
                tprint_data_preview(targets, "Input training targets", max_rows=10, level="INFO")
            
            # Preprocess data
            processed_data, processed_targets = self._preprocess_data(data, targets)
            
            # Preview preprocessed data
            tprint_data_preview(processed_data, "Preprocessed training data", max_rows=5, level="INFO")
            tprint_data_preview(processed_targets, "Preprocessed training targets", max_rows=10, level="INFO")
            
            # Create features
            feature_data = await self._create_tactician_features(processed_data)
            
            # Preview final training features
            tprint_data_preview(feature_data, "Final training features", max_rows=5, level="INFO")
            
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
                    'tactician_state': self._tactician_state.copy(),
                    'models_trained': list(training_results.keys()),
                    'timing_features_completed': self._tactician_state['timing_features_completed']
                }
            )
            
            tprint_success(f"✅ Tactician training completed in {training_time:.2f}s")
            self.logger.info(f"Tactician training completed in {training_time:.2f}s")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Tactician training failed: {e}")
            self.logger.error(f"Tactician training failed: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e),
                training_time=time.time() - start_time if 'start_time' in locals() else 0.0
            )
    
    async def validate(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> ValidationResult:
        """
        Validate the trained Tactician models.
        
        Args:
            data: Validation data
            targets: Target variables (optional)
            
        Returns:
            Validation result with metrics
        """
        try:
            tprint_info("⚔️ Validating Tactician models...")
            self.logger.info("Validating Tactician models...")
            
            start_time = time.time()
            
            # Preprocess data
            processed_data, processed_targets = self._preprocess_data(data, targets)
            
            # Create features
            feature_data = await self._create_tactician_features(processed_data)
            
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
            
            tprint_success(f"✅ Tactician validation completed in {validation_time:.2f}s")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Tactician validation failed: {e}")
            self.logger.error(f"Tactician validation failed: {e}")
            return ValidationResult(
                success=False,
                error_message=str(e)
            )
    
    async def predict(self, data: pd.DataFrame) -> PredictionResult:
        """
        Make predictions with the trained Tactician models.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Prediction result
        """
        try:
            tprint_info("⚔️ Making Tactician predictions...")
            self.logger.info("Making Tactician predictions...")
            
            start_time = time.time()
            
            # Preprocess data
            processed_data, _ = self._preprocess_data(data, None)
            
            # Create features
            feature_data = await self._create_tactician_features(processed_data)
            
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
            
            tprint_success(f"✅ Tactician predictions completed in {prediction_time:.2f}s")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Tactician prediction failed: {e}")
            self.logger.error(f"Tactician prediction failed: {e}")
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
            elif model_type == ModelType.LINEAR:
                return self._create_linear_model()
            else:
                raise ValueError(f"Unsupported model type for Tactician: {model_type}")
                
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
            elif hasattr(model, 'coef_'):
                return dict(zip(model.feature_names_in_, model.coef_))
            elif hasattr(model, 'get_feature_importance'):
                return model.get_feature_importance()
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
            return None
    
    @memory_efficient(memory_threshold_mb=200.0, auto_cleanup=True)
    @auto_optimize(optimize_inputs=True, optimize_outputs=True)
    async def _create_tactician_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create Tactician-specific features with enhanced optimization.
        
        Args:
            data: Input data
            
        Returns:
            Data with Tactician features
        """
        try:
            tprint_info("🔧 Creating Tactician features with enhanced optimization...")
            
            # Preview input data for feature creation
            from src.utils.tprint import tprint_data_preview
            tprint_data_preview(data, "Input data for feature creation", max_rows=5, level="DEBUG")
            
            # Use integrated hardware manager for optimized data processing
            hardware_manager = get_integrated_hardware_manager()
            feature_data = hardware_manager.process_data_with_optimization(
                data, WorkloadType.ML_TRAINING
            )
            
            # Create entry timing features if enabled
            if self.config.enable_entry_timing:
                feature_data = await self._create_entry_timing_features(feature_data)
                self._tactician_state['entry_timing_features_created'] = True
            
            # Create exit timing features if enabled
            if self.config.enable_exit_timing:
                feature_data = await self._create_exit_timing_features(feature_data)
                self._tactician_state['exit_timing_features_created'] = True
            
            # Create position sizing features if enabled
            if self.config.enable_position_sizing:
                feature_data = await self._create_position_sizing_features(feature_data)
                self._tactician_state['position_sizing_features_created'] = True
            
            self._tactician_state['timing_features_completed'] = True
            
            # Preview final Tactician features
            tprint_data_preview(feature_data, "Tactician features created", max_rows=5, level="INFO")
            
            tprint_success(f"✅ Created {feature_data.shape[1]} features with enhanced optimization")
            
            return feature_data
            
        except Exception as e:
            tprint_error(f"❌ Feature creation failed: {e}")
            self.logger.error(f"Feature creation failed: {e}")
            return data
    
    async def _create_entry_timing_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create entry timing features for tactical entry decisions."""
        try:
            tprint_debug("🔧 Creating entry timing features...")
            
            # Create a copy to avoid modifying original data
            feature_data = data.copy()
            
            # Lookback period for entry timing analysis
            lookback = self.config.entry_lookback
            
            # Price-based entry timing features
            if 'close' in data.columns:
                # Price momentum features
                feature_data['entry_price_momentum_5'] = data['close'].pct_change(5)
                feature_data['entry_price_momentum_10'] = data['close'].pct_change(10)
                feature_data['entry_price_momentum_20'] = data['close'].pct_change(20)
                
                # Price volatility for entry timing
                feature_data['entry_volatility_5'] = data['close'].rolling(5).std()
                feature_data['entry_volatility_10'] = data['close'].rolling(10).std()
                feature_data['entry_volatility_20'] = data['close'].rolling(20).std()
                
                # Price position within recent range
                feature_data['entry_price_position_10'] = (data['close'] - data['close'].rolling(10).min()) / (data['close'].rolling(10).max() - data['close'].rolling(10).min())
                feature_data['entry_price_position_20'] = (data['close'] - data['close'].rolling(20).min()) / (data['close'].rolling(20).max() - data['close'].rolling(20).min())
                
                # Price acceleration (second derivative)
                feature_data['entry_price_acceleration'] = data['close'].pct_change().diff()
                
                # Price breakout features
                feature_data['entry_breakout_high_10'] = (data['close'] > data['close'].rolling(10).max().shift(1)).astype(int)
                feature_data['entry_breakout_low_10'] = (data['close'] < data['close'].rolling(10).min().shift(1)).astype(int)
            
            # Volume-based entry timing features
            if 'volume' in data.columns:
                # Volume momentum
                feature_data['entry_volume_momentum_5'] = data['volume'].pct_change(5)
                feature_data['entry_volume_momentum_10'] = data['volume'].pct_change(10)
                
                # Volume relative to average
                feature_data['entry_volume_ratio_10'] = data['volume'] / data['volume'].rolling(10).mean()
                feature_data['entry_volume_ratio_20'] = data['volume'] / data['volume'].rolling(20).mean()
                
                # Volume spike detection
                feature_data['entry_volume_spike'] = (data['volume'] > data['volume'].rolling(20).mean() * 2).astype(int)
            
            # Technical indicator-based entry timing features
            if 'close' in data.columns and 'high' in data.columns and 'low' in data.columns:
                # RSI for entry timing
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                feature_data['entry_rsi'] = 100 - (100 / (1 + rs))
                
                # MACD for entry timing
                ema_12 = data['close'].ewm(span=12).mean()
                ema_26 = data['close'].ewm(span=26).mean()
                feature_data['entry_macd'] = ema_12 - ema_26
                feature_data['entry_macd_signal'] = feature_data['entry_macd'].ewm(span=9).mean()
                feature_data['entry_macd_histogram'] = feature_data['entry_macd'] - feature_data['entry_macd_signal']
                
                # Bollinger Bands for entry timing
                bb_period = 20
                bb_std = 2
                bb_middle = data['close'].rolling(bb_period).mean()
                bb_std_dev = data['close'].rolling(bb_period).std()
                feature_data['entry_bb_upper'] = bb_middle + (bb_std_dev * bb_std)
                feature_data['entry_bb_lower'] = bb_middle - (bb_std_dev * bb_std)
                feature_data['entry_bb_position'] = (data['close'] - feature_data['entry_bb_lower']) / (feature_data['entry_bb_upper'] - feature_data['entry_bb_lower'])
                feature_data['entry_bb_squeeze'] = (feature_data['entry_bb_upper'] - feature_data['entry_bb_lower']) / bb_middle
            
            # Time-based entry timing features
            if data.index.dtype == 'datetime64[ns]' or hasattr(data.index, 'hour'):
                # Hour of day effect
                feature_data['entry_hour'] = data.index.hour if hasattr(data.index, 'hour') else 0
                feature_data['entry_day_of_week'] = data.index.dayofweek if hasattr(data.index, 'dayofweek') else 0
                
                # Market session indicators
                feature_data['entry_is_london_session'] = ((feature_data['entry_hour'] >= 8) & (feature_data['entry_hour'] < 16)).astype(int)
                feature_data['entry_is_ny_session'] = ((feature_data['entry_hour'] >= 13) & (feature_data['entry_hour'] < 21)).astype(int)
                feature_data['entry_is_asian_session'] = ((feature_data['entry_hour'] >= 0) & (feature_data['entry_hour'] < 8)).astype(int)
            
            # Cross-asset entry timing features (if available)
            if 'spy' in data.columns or 'vix' in data.columns:
                if 'spy' in data.columns:
                    feature_data['entry_spy_correlation'] = data['close'].rolling(20).corr(data['spy'])
                    feature_data['entry_spy_momentum'] = data['spy'].pct_change(5)
                
                if 'vix' in data.columns:
                    feature_data['entry_vix_level'] = data['vix']
                    feature_data['entry_vix_momentum'] = data['vix'].pct_change(5)
                    feature_data['entry_fear_greed'] = (data['vix'] > data['vix'].rolling(20).quantile(0.8)).astype(int)
            
            # Fill NaN values created by rolling operations
            feature_data = feature_data.fillna(method='ffill').fillna(0)
            
            tprint_success(f"✅ Created {len([col for col in feature_data.columns if col.startswith('entry_')])} entry timing features")
            return feature_data
            
        except Exception as e:
            tprint_error(f"❌ Entry timing feature creation failed: {e}")
            self.logger.error(f"Entry timing feature creation failed: {e}")
            return data
    
    async def _create_exit_timing_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create exit timing features for tactical exit decisions."""
        try:
            tprint_debug("🔧 Creating exit timing features...")
            
            # Create a copy to avoid modifying original data
            feature_data = data.copy()
            
            # Lookback period for exit timing analysis
            lookback = self.config.exit_lookback
            
            # Price-based exit timing features
            if 'close' in data.columns:
                # Price reversal features
                feature_data['exit_price_reversal_3'] = data['close'].pct_change(3)
                feature_data['exit_price_reversal_5'] = data['close'].pct_change(5)
                feature_data['exit_price_reversal_10'] = data['close'].pct_change(10)
                
                # Price exhaustion signals
                feature_data['exit_price_exhaustion'] = (data['close'].pct_change() * data['close'].pct_change().shift(1) < 0).astype(int)
                
                # Price divergence features
                feature_data['exit_price_divergence'] = (data['close'].pct_change(5) * data['close'].pct_change(10) < 0).astype(int)
                
                # Support and resistance levels
                feature_data['exit_support_distance'] = (data['close'] - data['close'].rolling(20).min()) / data['close']
                feature_data['exit_resistance_distance'] = (data['close'].rolling(20).max() - data['close']) / data['close']
                
                # Price channel position
                feature_data['exit_channel_position'] = (data['close'] - data['close'].rolling(20).min()) / (data['close'].rolling(20).max() - data['close'].rolling(20).min())
                
                # Price acceleration for exit
                feature_data['exit_price_deceleration'] = -data['close'].pct_change().diff()
            
            # Volume-based exit timing features
            if 'volume' in data.columns:
                # Volume divergence (price up, volume down)
                price_change = data['close'].pct_change(5)
                volume_change = data['volume'].pct_change(5)
                feature_data['exit_volume_divergence'] = ((price_change > 0) & (volume_change < 0)).astype(int)
                
                # Volume exhaustion
                feature_data['exit_volume_exhaustion'] = (data['volume'] < data['volume'].rolling(10).mean() * 0.5).astype(int)
                
                # Volume spike for exit
                feature_data['exit_volume_spike'] = (data['volume'] > data['volume'].rolling(20).mean() * 1.5).astype(int)
            
            # Technical indicator-based exit timing features
            if 'close' in data.columns and 'high' in data.columns and 'low' in data.columns:
                # RSI for exit timing (overbought/oversold)
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                feature_data['exit_rsi'] = rsi
                feature_data['exit_rsi_overbought'] = (rsi > 70).astype(int)
                feature_data['exit_rsi_oversold'] = (rsi < 30).astype(int)
                
                # MACD divergence for exit
                ema_12 = data['close'].ewm(span=12).mean()
                ema_26 = data['close'].ewm(span=26).mean()
                macd = ema_12 - ema_26
                macd_signal = macd.ewm(span=9).mean()
                feature_data['exit_macd'] = macd
                feature_data['exit_macd_signal'] = macd_signal
                feature_data['exit_macd_divergence'] = ((macd > macd_signal) & (macd.shift(1) < macd_signal.shift(1))).astype(int)
                
                # Stochastic for exit timing
                low_14 = data['low'].rolling(14).min()
                high_14 = data['high'].rolling(14).max()
                k_percent = 100 * ((data['close'] - low_14) / (high_14 - low_14))
                d_percent = k_percent.rolling(3).mean()
                feature_data['exit_stoch_k'] = k_percent
                feature_data['exit_stoch_d'] = d_percent
                feature_data['exit_stoch_overbought'] = (k_percent > 80).astype(int)
                feature_data['exit_stoch_oversold'] = (k_percent < 20).astype(int)
                
                # Williams %R for exit
                feature_data['exit_williams_r'] = -100 * ((data['high'].rolling(14).max() - data['close']) / (data['high'].rolling(14).max() - data['low'].rolling(14).min()))
                feature_data['exit_williams_overbought'] = (feature_data['exit_williams_r'] > -20).astype(int)
                feature_data['exit_williams_oversold'] = (feature_data['exit_williams_r'] < -80).astype(int)
            
            # Time-based exit timing features
            if data.index.dtype == 'datetime64[ns]' or hasattr(data.index, 'hour'):
                # Time-based exit signals
                feature_data['exit_hour'] = data.index.hour if hasattr(data.index, 'hour') else 0
                feature_data['exit_day_of_week'] = data.index.dayofweek if hasattr(data.index, 'dayofweek') else 0
                
                # End of day exit signals
                feature_data['exit_near_close'] = (feature_data['exit_hour'] >= 15).astype(int)  # Near market close
                feature_data['exit_weekend_approaching'] = (feature_data['exit_day_of_week'] >= 4).astype(int)  # Friday
                
                # Session end signals
                feature_data['exit_london_close'] = (feature_data['exit_hour'] == 16).astype(int)
                feature_data['exit_ny_close'] = (feature_data['exit_hour'] == 21).astype(int)
            
            # Risk management exit features
            if 'close' in data.columns:
                # Stop loss levels
                feature_data['exit_stop_loss_2pct'] = (data['close'] < data['close'].rolling(20).max() * 0.98).astype(int)
                feature_data['exit_stop_loss_5pct'] = (data['close'] < data['close'].rolling(20).max() * 0.95).astype(int)
                
                # Take profit levels
                feature_data['exit_take_profit_2pct'] = (data['close'] > data['close'].rolling(20).min() * 1.02).astype(int)
                feature_data['exit_take_profit_5pct'] = (data['close'] > data['close'].rolling(20).min() * 1.05).astype(int)
                
                # Trailing stop features
                feature_data['exit_trailing_stop'] = (data['close'] < data['close'].rolling(10).max() * 0.97).astype(int)
            
            # Market condition exit features
            if 'close' in data.columns:
                # Trend change detection
                sma_short = data['close'].rolling(5).mean()
                sma_long = data['close'].rolling(20).mean()
                feature_data['exit_trend_change'] = ((sma_short > sma_long) != (sma_short.shift(1) > sma_long.shift(1))).astype(int)
                
                # Volatility expansion
                feature_data['exit_vol_expansion'] = (data['close'].rolling(5).std() > data['close'].rolling(20).std() * 1.5).astype(int)
                
                # Gap detection
                feature_data['exit_gap_up'] = (data['close'] > data['close'].shift(1) * 1.02).astype(int)
                feature_data['exit_gap_down'] = (data['close'] < data['close'].shift(1) * 0.98).astype(int)
            
            # Fill NaN values created by rolling operations
            feature_data = feature_data.fillna(method='ffill').fillna(0)
            
            tprint_success(f"✅ Created {len([col for col in feature_data.columns if col.startswith('exit_')])} exit timing features")
            return feature_data
            
        except Exception as e:
            tprint_error(f"❌ Exit timing feature creation failed: {e}")
            self.logger.error(f"Exit timing feature creation failed: {e}")
            return data
    
    async def _create_position_sizing_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create position sizing features for tactical position management."""
        try:
            tprint_debug("🔧 Creating position sizing features...")
            
            # Create a copy to avoid modifying original data
            feature_data = data.copy()
            
            # Lookback period for position sizing analysis
            lookback = self.config.position_sizing_lookback
            
            # Volatility-based position sizing features
            if 'close' in data.columns:
                # Historical volatility
                feature_data['position_volatility_5'] = data['close'].rolling(5).std()
                feature_data['position_volatility_10'] = data['close'].rolling(10).std()
                feature_data['position_volatility_20'] = data['close'].rolling(20).std()
                feature_data['position_volatility_50'] = data['close'].rolling(50).std()
                
                # Volatility percentile ranking
                feature_data['position_vol_percentile_20'] = data['close'].rolling(20).std().rolling(100).rank(pct=True)
                feature_data['position_vol_percentile_50'] = data['close'].rolling(20).std().rolling(200).rank(pct=True)
                
                # Volatility regime detection
                vol_short = data['close'].rolling(10).std()
                vol_long = data['close'].rolling(50).std()
                feature_data['position_vol_regime_high'] = (vol_short > vol_long * 1.5).astype(int)
                feature_data['position_vol_regime_low'] = (vol_short < vol_long * 0.5).astype(int)
                feature_data['position_vol_regime_normal'] = ((vol_short >= vol_long * 0.5) & (vol_short <= vol_long * 1.5)).astype(int)
            
            # Risk-adjusted return features
            if 'close' in data.columns:
                # Sharpe ratio approximation
                returns = data['close'].pct_change()
                feature_data['position_sharpe_20'] = returns.rolling(20).mean() / returns.rolling(20).std()
                feature_data['position_sharpe_50'] = returns.rolling(50).mean() / returns.rolling(50).std()
                
                # Sortino ratio approximation (downside deviation)
                downside_returns = returns.where(returns < 0, 0)
                feature_data['position_sortino_20'] = returns.rolling(20).mean() / downside_returns.rolling(20).std()
                feature_data['position_sortino_50'] = returns.rolling(50).mean() / downside_returns.rolling(50).std()
                
                # Calmar ratio approximation
                feature_data['position_calmar_20'] = returns.rolling(20).mean() / data['close'].rolling(20).max().pct_change()
                feature_data['position_calmar_50'] = returns.rolling(50).mean() / data['close'].rolling(50).max().pct_change()
            
            # Drawdown-based position sizing features
            if 'close' in data.columns:
                # Current drawdown
                rolling_max = data['close'].rolling(20).max()
                feature_data['position_drawdown_20'] = (data['close'] - rolling_max) / rolling_max
                
                rolling_max_50 = data['close'].rolling(50).max()
                feature_data['position_drawdown_50'] = (data['close'] - rolling_max_50) / rolling_max_50
                
                # Maximum drawdown
                feature_data['position_max_drawdown_20'] = feature_data['position_drawdown_20'].rolling(20).min()
                feature_data['position_max_drawdown_50'] = feature_data['position_drawdown_50'].rolling(50).min()
                
                # Drawdown duration
                feature_data['position_dd_duration'] = (feature_data['position_drawdown_20'] < 0).groupby((feature_data['position_drawdown_20'] >= 0).cumsum()).cumsum()
            
            # Kelly Criterion features
            if 'close' in data.columns:
                returns = data['close'].pct_change()
                win_rate = (returns > 0).rolling(20).mean()
                avg_win = returns.where(returns > 0).rolling(20).mean()
                avg_loss = returns.where(returns < 0).rolling(20).mean()
                
                # Kelly fraction approximation
                feature_data['position_kelly_20'] = win_rate - (1 - win_rate) / (avg_win / abs(avg_loss))
                feature_data['position_kelly_50'] = (returns > 0).rolling(50).mean() - (1 - (returns > 0).rolling(50).mean()) / (returns.where(returns > 0).rolling(50).mean() / abs(returns.where(returns < 0).rolling(50).mean()))
                
                # Conservative Kelly (half Kelly)
                feature_data['position_kelly_conservative_20'] = feature_data['position_kelly_20'] * 0.5
                feature_data['position_kelly_conservative_50'] = feature_data['position_kelly_50'] * 0.5
            
            # Market regime position sizing features
            if 'close' in data.columns:
                # Trend strength
                sma_short = data['close'].rolling(10).mean()
                sma_long = data['close'].rolling(30).mean()
                feature_data['position_trend_strength'] = (sma_short - sma_long) / sma_long
                
                # Trend consistency
                feature_data['position_trend_consistency'] = (data['close'] > sma_short).rolling(10).mean()
                
                # Market efficiency (random walk test)
                returns = data['close'].pct_change()
                feature_data['position_efficiency'] = returns.rolling(20).corr(returns.shift(1))
            
            # Volume-based position sizing features
            if 'volume' in data.columns and 'close' in data.columns:
                # Volume-weighted average price (VWAP) distance
                vwap = (data['close'] * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
                feature_data['position_vwap_distance'] = (data['close'] - vwap) / vwap
                
                # Volume profile position
                feature_data['position_volume_percentile'] = data['volume'].rolling(50).rank(pct=True)
                
                # Liquidity assessment
                feature_data['position_liquidity_score'] = (data['volume'] * data['close']).rolling(20).mean()
            
            # Correlation-based position sizing features
            if 'close' in data.columns:
                # Autocorrelation (momentum vs mean reversion)
                returns = data['close'].pct_change()
                feature_data['position_autocorr_1'] = returns.rolling(20).corr(returns.shift(1))
                feature_data['position_autocorr_5'] = returns.rolling(20).corr(returns.shift(5))
                
                # Cross-asset correlation (if available)
                if 'spy' in data.columns:
                    feature_data['position_spy_correlation'] = returns.rolling(20).corr(data['spy'].pct_change())
                if 'vix' in data.columns:
                    feature_data['position_vix_correlation'] = returns.rolling(20).corr(data['vix'].pct_change())
            
            # Risk parity features
            if 'close' in data.columns:
                # Equal weight portfolio equivalent
                feature_data['position_equal_weight'] = 1.0 / len(data.columns) if len(data.columns) > 0 else 1.0
                
                # Inverse volatility weighting
                vol = data['close'].rolling(20).std()
                feature_data['position_inv_vol_weight'] = (1 / vol) / (1 / vol).rolling(20).sum()
                
                # Risk parity score
                feature_data['position_risk_parity_score'] = feature_data['position_inv_vol_weight'] * feature_data['position_sharpe_20']
            
            # Time-based position sizing features
            if data.index.dtype == 'datetime64[ns]' or hasattr(data.index, 'hour'):
                # Time decay features
                feature_data['position_hour'] = data.index.hour if hasattr(data.index, 'hour') else 0
                feature_data['position_day_of_week'] = data.index.dayofweek if hasattr(data.index, 'dayofweek') else 0
                
                # Session-based position sizing
                feature_data['position_london_session'] = ((feature_data['position_hour'] >= 8) & (feature_data['position_hour'] < 16)).astype(int)
                feature_data['position_ny_session'] = ((feature_data['position_hour'] >= 13) & (feature_data['position_hour'] < 21)).astype(int)
                feature_data['position_asian_session'] = ((feature_data['position_hour'] >= 0) & (feature_data['position_hour'] < 8)).astype(int)
                
                # Weekend effect
                feature_data['position_weekend_effect'] = (feature_data['position_day_of_week'] >= 5).astype(int)
            
            # Portfolio heat features
            if 'close' in data.columns:
                # Portfolio heat (risk per trade)
                feature_data['position_portfolio_heat'] = feature_data['position_volatility_20'] * 0.02  # 2% risk per trade
                
                # Position size based on account risk
                feature_data['position_account_risk_1pct'] = 0.01 / feature_data['position_volatility_20']
                feature_data['position_account_risk_2pct'] = 0.02 / feature_data['position_volatility_20']
                feature_data['position_account_risk_5pct'] = 0.05 / feature_data['position_volatility_20']
            
            # Fill NaN values created by rolling operations
            feature_data = feature_data.fillna(method='ffill').fillna(0)
            
            # Cap extreme values to prevent numerical issues
            numeric_cols = feature_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col.startswith('position_'):
                    feature_data[col] = feature_data[col].clip(-10, 10)  # Cap between -10 and 10
            
            tprint_success(f"✅ Created {len([col for col in feature_data.columns if col.startswith('position_')])} position sizing features")
            return feature_data
            
        except Exception as e:
            tprint_error(f"❌ Position sizing feature creation failed: {e}")
            self.logger.error(f"Position sizing feature creation failed: {e}")
            return data
    
    def _create_lightgbm_model(self):
        """Create LightGBM model."""
        try:
            import lightgbm as lgb
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                **self.config.lightgbm_params
            }
            
            return lgb.LGBMRegressor(**params)
            
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
            
            return cb.CatBoostRegressor(**params)
            
        except ImportError:
            self.logger.error("CatBoost not available")
            return None
    
    def _create_neural_network_model(self):
        """Create neural network model for Tactician predictions."""
        try:
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            
            # Neural network parameters optimized for Tactician
            params = {
                'hidden_layer_sizes': (200, 100, 50),  # Deeper network for complex patterns
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
                ('mlp', MLPRegressor(**params))
            ])
            
            tprint_debug("🔧 Created neural network model with enhanced architecture")
            return pipeline
            
        except ImportError as e:
            tprint_error(f"❌ scikit-learn not available for neural network: {e}")
            self.logger.error(f"scikit-learn not available for neural network: {e}")
            return None
        except Exception as e:
            tprint_error(f"❌ Failed to create neural network model: {e}")
            self.logger.error(f"Failed to create neural network model: {e}")
            return None
    
    def _create_linear_model(self):
        """Create linear model."""
        try:
            from sklearn.linear_model import LinearRegression
            
            params = {
                **self.config.linear_params
            }
            
            return LinearRegression(**params)
            
        except ImportError:
            self.logger.error("scikit-learn not available")
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
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics = {
                'mse': mean_squared_error(targets, predictions),
                'mae': mean_absolute_error(targets, predictions),
                'r2': r2_score(targets, predictions),
                'rmse': np.sqrt(mean_squared_error(targets, predictions))
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Validation failed for {model_type.value}: {e}")
            return {}
    
    async def _predict_single_model(self, model: Any, model_type: ModelType, data: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions with a single model."""
        try:
            predictions = model.predict(data)
            
            # Preview predictions
            from src.utils.tprint import tprint_data_preview
            tprint_data_preview(predictions, f"Predictions from {model_type.value}", max_rows=10, level="DEBUG")
            
            # For regression models, we don't typically have probabilities
            # But we can calculate confidence intervals or uncertainty estimates
            probabilities = None
            
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
    
    def get_tactician_summary(self) -> Dict[str, Any]:
        """Get Tactician-specific training summary."""
        base_summary = self.get_training_summary()
        base_summary.update({
            'tactician_state': self._tactician_state.copy(),
            'timing_features': {
                'entry_timing': self._tactician_state['entry_timing_features_created'],
                'exit_timing': self._tactician_state['exit_timing_features_created'],
                'position_sizing': self._tactician_state['position_sizing_features_created'],
                'completed': self._tactician_state['timing_features_completed']
            }
        })
        return base_summary